"""真机后端组装〔线B 维护〕—— observe()/move() 的真车实现

组装关系（组件全部注入，便于 mock 单测与现场替换）：
    observe() = camera.read() → spine_detector(frame) → 线A BarDetector.observe(frame, spine)
    move(u)   = mover.move(u)（robot/mover.py，正 = 后退）

每阶段耗时写入 Observation.debug（capture_ms / infer_ms / bar_ms / total_ms）——
「拍照到读数 0.8 秒花在哪」这个问题的答案就从这里出，S5 现场用它找瓶颈。

与 shelf_sim 的注入式约定一致：控制器只认 observe/move 两个回调，不知道对面是谁。
"""

import time

from ..observation import Observation


class RobotBackend:
    """RobotBackend 真机后端：真相机 + 真检测 + 真底盘的 observe/move 组装

    :param camera: 相机件，.read() 返回 BGR 帧（Step 3 CameraBackend）；无帧返回 None
    :param spine_detector: 书脊检测件，可调用对象 frame -> SpineBottom | None（Step 3 RKNN）
    :param mover: robot/mover.py 的 Mover 实例
    :param bar_detector: 线A perception.BarDetector 实例（横杆检测 + Observation 组装）
    :param source: 来源标注前缀，写进 Observation.source 供留痕区分
    """

    def __init__(self, camera, spine_detector, mover, bar_detector, source="robot"):
        self.camera = camera
        self.spine_detector = spine_detector
        self.mover = mover
        self.bar_detector = bar_detector
        self.source = source

    def observe(self):
        """observe 采集一帧并组装 Observation，耗时分解写进 debug"""
        t0 = time.perf_counter()
        frame = self.camera.read()
        t1 = time.perf_counter()

        if frame is None:
            # 相机没出帧：按"整帧丢失"如实上报（mode=none → 控制器走丢失分支）
            obs = Observation(frame_size=(0, 0), source=f"{self.source}+no_frame")
            obs.debug.update(capture_ms=round((t1 - t0) * 1000, 1), infer_ms=None,
                             bar_ms=None, total_ms=round((t1 - t0) * 1000, 1))
            return obs

        spine = None
        if self.spine_detector is not None:
            spine = self.spine_detector(frame)
        t2 = time.perf_counter()

        bar_name = getattr(self.bar_detector, "name", "bar")
        obs = self.bar_detector.observe(frame, spine, source=f"{self.source}+{bar_name}")
        t3 = time.perf_counter()

        obs.debug.update(
            capture_ms=round((t1 - t0) * 1000.0, 1),
            infer_ms=round((t2 - t1) * 1000.0, 1),
            bar_ms=round((t3 - t2) * 1000.0, 1),
            total_ms=round((t3 - t0) * 1000.0, 1),
        )
        return obs

    def move(self, delta_m):
        """move 委托底盘 Mover（正 = 后退），返回完成复核后的位姿"""
        return self.mover.move(delta_m)
