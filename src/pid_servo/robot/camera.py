"""相机采集〔线B 维护〕—— cam_4 预览流的开关与取帧

设计要点：
    - 闭环只用一路相机、预览分辨率（configs camera 节）：bar_frac 是比例，
      与分辨率无关；540p 比 4K 采集/解码快一个量级，终拍才回 4K
    - 设备定位优先 V4L2 by-path（插拔不串号，学长 camera_config.yaml 同源），
      路径留空则退回用 cam_id 当 /dev/videoN 索引 —— TODO(S4上板) 现场 by-path 填入
    - 开流后先丢 settle_frames 帧（曝光未稳），这段由 start() 完成
    - 补光灯沿用学长 led.py 的 Modbus RTU 帧（Y1 继电器），闭环全程常亮
    - OpenCV/serial 惰性导入 + 采集器工厂可注入：PC 单测不碰任何硬件
"""

import time


class CameraError(RuntimeError):
    """相机打开/取帧失败"""


# ========== 补光灯（学长 led.py 的协议，Y1 继电器，Modbus RTU） ==========

#: 开/关 Y1 的两帧（寄存器 0x0000 与 0x0001，帧尾 CRC 与 led.py 逐字节一致）
LIGHT_FRAMES_ON = (bytes([0x01, 0x06, 0x00, 0x00, 0x00, 0x01, 0x48, 0x0A]),
                   bytes([0x01, 0x06, 0x00, 0x01, 0x00, 0x01, 0x19, 0xCA]))
LIGHT_FRAMES_OFF = (bytes([0x01, 0x06, 0x00, 0x00, 0x00, 0x00, 0x89, 0xCA]),
                    bytes([0x01, 0x06, 0x00, 0x01, 0x00, 0x00, 0xD8, 0x0A]))


class LightController:
    """LightController 补光灯开关（每帧独立开串口→发送→关闭，沿用 led.py 的写法）

    :param port: 串口设备（板端 /dev/ttyCH341USB0）
    :param baudrate: 38400（学长系统同值）
    :param opener: 打开串口的可调用（测试注入假件），默认 serial.Serial
    """

    def __init__(self, port, baudrate=38400, opener=None):
        self.port = port
        self.baudrate = baudrate
        self._opener = opener or self._serial_opener

    @staticmethod
    def _serial_opener(port, baudrate):
        import serial  # 惰性：板端依赖，PC 单测注入假 opener

        return serial.Serial(port, baudrate, timeout=1)

    def _send(self, frames):
        for frame in frames:
            ser = self._opener(self.port, self.baudrate)
            try:
                ser.write(frame)
            finally:
                ser.close()

    def on(self):
        self._send(LIGHT_FRAMES_ON)

    def off(self):
        self._send(LIGHT_FRAMES_OFF)


class _NoLight:
    """未启用补光灯时的空件"""

    def on(self):
        pass

    def off(self):
        pass


# ========== 相机后端 ==========

def _cv_capture_factory(device, fourcc, width, height, fps):
    """默认采集器工厂：OpenCV VideoCapture（板端）；device = by-path 字符串或索引"""
    import cv2  # 惰性：PC 单测注入假工厂

    cap = cv2.VideoCapture(device)
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps:
        cap.set(cv2.CAP_PROP_FPS, float(fps))
    return cap


def check_frame_quality(frame, blur_min_lap=100.0, brightness_min=40.0,
                        brightness_max=220.0):
    """check_frame_quality 确认照质检两道门：清晰度与亮度

    阈值标准与学长 Camera.check_frame_quality 同源（configs camera.quality 节）：
    模糊帧（Laplacian 方差低）与过暗/过曝帧不许当"完美照片"出库。

    :param frame: BGR 帧（numpy 数组）
    :param blur_min_lap: 灰度图 Laplacian 方差下限，低于 = 模糊
    :param brightness_min: 亮度均值合格下界
    :param brightness_max: 亮度均值合格上界
    :return: (ok, metrics)；metrics = {"laplacian_var", "brightness"}（保留 1 位小数）
    """
    import cv2  # 惰性：与模块其余部分一致

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())
    ok = lap >= blur_min_lap and brightness_min <= bright <= brightness_max
    return ok, {"laplacian_var": round(lap, 1), "brightness": round(bright, 1)}


class CameraBackend:
    """CameraBackend 单路预览流：start() 开流丢帧，read() 取一帧，close() 关流关灯

    :param capture_factory: 采集器工厂（测试注入假件），签名见 _cv_capture_factory
    :param device_path: V4L2 by-path 路径；空串则用 cam_id 当 /dev/videoN 索引
    :param light: 带 on()/off() 的灯控件；None = 不控灯
    """

    def __init__(self, capture_factory, device_path="", cam_id="4",
                 fourcc="MJPG", width=960, height=540, fps=30,
                 settle_frames=3, light=None):
        self._factory = capture_factory
        self.device = device_path or int(cam_id)
        self.fourcc = fourcc
        self.width = width
        self.height = height
        self.fps = fps
        self.settle_frames = int(settle_frames)
        self.light = light if light is not None else _NoLight()
        self._cap = None

    @classmethod
    def from_config(cls, cfg, capture_factory=None, light=None):
        """from_config 按完整配置字典的 camera 节构造（light 不传则按节内配置启停）"""
        from ..utils import get

        if light is None and get(cfg, "camera.light.enabled", False):
            light = LightController(get(cfg, "camera.light.port", "/dev/ttyCH341USB0"),
                                    float(get(cfg, "camera.light.baudrate", 38400)))
        return cls(
            capture_factory or _cv_capture_factory,
            device_path=get(cfg, "camera.device_path", ""),
            cam_id=str(get(cfg, "camera.cam_id", "4")),
            fourcc=get(cfg, "camera.fourcc", "MJPG"),
            width=int(get(cfg, "camera.preview_width", 960)),
            height=int(get(cfg, "camera.preview_height", 540)),
            fps=int(get(cfg, "camera.fps", 30)),
            settle_frames=int(get(cfg, "camera.settle_frames", 3)),
            light=light,
        )

    def start(self):
        """start 开流：打不开抛 CameraError；成功后丢 settle_frames 帧并开灯"""
        self._cap = self._factory(self.device, self.fourcc,
                                  self.width, self.height, self.fps)
        if not getattr(self._cap, "isOpened", lambda: True)():
            raise CameraError(f"相机打开失败: {self.device!r}")
        for _ in range(self.settle_frames):          # 丢掉曝光未稳的帧
            self._cap.read()
        self.light.on()
        return self

    def read(self):
        """read 取一帧 BGR；单次失败立即重试一次，仍失败返回 None（按丢帧处理）"""
        if self._cap is None:
            raise CameraError("相机未 start() 就 read()")
        for _ in range(2):
            ok, frame = self._cap.read()
            if ok:
                return frame
            time.sleep(0.05)
        return None

    def close(self):
        """close 关流关灯（幂等：仅在流开着时生效，灯只在真正开过时才关）"""
        if self._cap is None:
            return
        self.light.off()
        release = getattr(self._cap, "release", None)
        if release:
            release()
        self._cap = None

    def capture_still(self, width, height, quality_cfg=None):
        """capture_still 终拍确认照：关预览流 → 补光灯常亮 → 重开高分辨率流 → 质检取帧

        节奏参照学长 capture_group：开流后丢 settle 帧等曝光稳，拍一帧过
        质检（清晰度/亮度），不达标等 retry_wait_ms 再拍，最多 retries 次；
        全不合格时返回得分最高（最清晰）的一帧并 ok=False，交人工判断。
        返回后预览流保持关闭、补光灯关闭（终拍是流程最后一环），
        之后的 close() 是幂等空操作。

        :param width/height: 终拍分辨率（configs camera.still_width/still_height）
        :param quality_cfg: configs camera.quality 节（阈值与重拍节奏）；None 用内置默认
        :return: {"frame": 帧或 None, "ok": 是否质检通过,
                  "metrics": 质检指标, "attempts": 实际拍摄次数}
        :raises CameraError: 终拍流打不开（失败前会把灯关掉）
        """
        q = quality_cfg or {}
        blur_min = float(q.get("blur_min_lap", 100.0))
        b_min = float(q.get("brightness_min", 40.0))
        b_max = float(q.get("brightness_max", 220.0))
        retries = int(q.get("retries", 3))
        wait_s = float(q.get("retry_wait_ms", 1000)) / 1000.0

        if self._cap is not None:               # 释放预览流让出设备，灯保持常亮
            release = getattr(self._cap, "release", None)
            if release:
                release()
            self._cap = None
        self.light.on()

        cap = self._factory(self.device, self.fourcc, width, height, self.fps)
        if not getattr(cap, "isOpened", lambda: True)():
            self.light.off()
            raise CameraError(f"终拍打开失败: {self.device!r} @{width}x{height}")

        result = {"frame": None, "ok": False, "metrics": {}, "attempts": 0}
        best, best_lap = None, -1.0
        try:
            for _ in range(self.settle_frames):     # 曝光未稳的帧照旧丢弃
                cap.read()
            for i in range(retries):
                got, frame = cap.read()
                result["attempts"] = i + 1
                if not got:
                    time.sleep(wait_s)
                    continue
                passed, metrics = check_frame_quality(frame, blur_min, b_min, b_max)
                if passed:
                    result.update(frame=frame, ok=True, metrics=metrics)
                    return result
                if metrics["laplacian_var"] > best_lap:   # 兜底：留最清晰的一帧
                    best, best_lap = frame, metrics["laplacian_var"]
                    result["metrics"] = metrics
                time.sleep(wait_s)
        finally:
            release = getattr(cap, "release", None)
            if release:
                release()
            self.light.off()
        if best is not None:
            result["frame"] = best
        return result
