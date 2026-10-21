"""Step 3 测试 —— 相机后端 / 补光灯 / RKNN 书脊检测（全 PC 假件与合成数据）

不碰任何硬件：采集器工厂、串口 opener、RKNN runtime 全部注入假件。
协议类断言（Modbus 帧、坐标反变换）是上板前的最后防线。
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pid_servo.robot.camera import (LIGHT_FRAMES_OFF, LIGHT_FRAMES_ON,
                                    CameraBackend, CameraError, LightController)
from pid_servo.robot.spine_detector import (RknnSpineDetector, boxes_to_spine_stats,
                                            decode_obb_raw, letterbox)


# ========== 相机后端 ==========

class FakeCapture:
    def __init__(self, opened=True):
        self.opened = opened
        self.reads = 0
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        if not self.opened:
            return False, None
        self.reads += 1
        return True, np.zeros((2, 2, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class FakeLight:
    def __init__(self):
        self.events = []

    def on(self):
        self.events.append("on")

    def off(self):
        self.events.append("off")


def make_camera(opened=True, **over):
    made = {}

    def factory(device, fourcc, width, height, fps):
        made["args"] = (device, fourcc, width, height, fps)
        made["cap"] = cap = FakeCapture(opened=opened)
        return cap

    kwargs = dict(device_path="/dev/v4l/by-path/cam4", cam_id="4", fourcc="MJPG",
                  width=960, height=540, fps=30, settle_frames=3, light=FakeLight())
    kwargs.update(over)
    cam = CameraBackend(factory, **kwargs)
    return cam, made


def test_camera_start_drops_settle_frames_and_turns_light_on():
    cam, made = make_camera()
    cam.start()
    cap = made["cap"]
    assert cap.reads == 3                       # settle_frames 被丢弃
    assert cap.opened and cam.light.events == ["on"]
    frame = cam.read()
    assert frame is not None and frame.shape == (2, 2, 3)
    cam.close()
    assert cap.released and cam.light.events == ["on", "off"]
    cam.close()                                 # 幂等
    assert cam.light.events == ["on", "off"]


def test_camera_open_failure_raises():
    cam, _ = make_camera(opened=False)
    with pytest.raises(CameraError, match="相机打开失败"):
        cam.start()


def test_read_retries_then_reports_none():
    cam, made = make_camera()
    cam.start()
    cap = made["cap"]
    orig_read = cap.read

    def fail_then_fail():
        return False, None
    cap.read = fail_then_fail
    assert cam.read() is None                   # 两次都失败 → 按丢帧处理
    assert orig_read                            # 原读取函数存在（防御性占位）


def test_device_falls_back_to_cam_index():
    cam, made = make_camera(device_path="", cam_id="4")
    cam.start()
    assert made["args"][0] == 4                 # 空 by-path → /dev/videoN 索引


# ========== 补光灯协议（与学长 led.py 逐字节一致） ==========

def test_light_on_off_frames_exact():
    log = []

    def fake_opener(port, baud):
        log.append(("open", port, baud))
        ser = type("S", (), {})()
        ser.write = lambda b: log.append(("write", b))
        ser.close = lambda: log.append(("close",))
        return ser

    light = LightController("/dev/ttyCH341USB0", 38400, opener=fake_opener)
    light.on()
    light.off()

    expect = [("open", "/dev/ttyCH341USB0", 38400),
              ("write", LIGHT_FRAMES_ON[0]), ("close",),
              ("open", "/dev/ttyCH341USB0", 38400),
              ("write", LIGHT_FRAMES_ON[1]), ("close",),
              ("open", "/dev/ttyCH341USB0", 38400),
              ("write", LIGHT_FRAMES_OFF[0]), ("close",),
              ("open", "/dev/ttyCH341USB0", 38400),
              ("write", LIGHT_FRAMES_OFF[1]), ("close",)]
    assert log == expect


# ========== 书脊检测：纯数学 + 解码 ==========

def test_boxes_to_spine_stats_matches_legacy_math():
    """与学长 JSON 同形的框：底边 = cy + (|w·sinθ|+|h·cosθ|)/2，取最大；置信度取最弱"""
    b1 = (0.90, 44.53, 1168.125, 1733.34, 94.70, 91.707)   # 学长真实样例量级
    b2 = (0.80, 137.70, 1246.875, 1574.90, 89.08, 91.539)
    spine, roll = boxes_to_spine_stats([b1, b2])

    expect1 = 1168.125 + (abs(1733.34 * math.sin(math.radians(91.707)))
                          + abs(94.70 * math.cos(math.radians(91.707)))) / 2
    expect2 = 1246.875 + (abs(1574.90 * math.sin(math.radians(91.539)))
                          + abs(89.08 * math.cos(math.radians(91.539)))) / 2
    assert spine.y == pytest.approx(max(expect1, expect2), abs=1e-6)
    assert spine.confidence == pytest.approx(0.80)         # 最弱者
    assert spine.n_books == 2


def test_boxes_filtering_confidence_and_cap():
    boxes = [(0.4, 1, 1, 10, 10, 90)]                  # 低于门槛 → 剔除
    boxes += [(0.9, i * 10.0, 100.0, 100.0, 10.0, 90.0) for i in range(25)]
    spine, _ = boxes_to_spine_stats(boxes, min_confidence=0.5, max_boxes=20)
    assert spine.n_books == 20                          # 前 K 截断


def test_letterbox_geometry():
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    canvas, ratio, pad_x, pad_y = letterbox(frame, 1024)
    assert canvas.shape == (1024, 1024, 3)
    assert ratio == pytest.approx(1024 / 960)           # 受宽度限制
    assert (pad_x, pad_y) == (0, 224)                   # 1024 - 540*1.0667 = 576 → 上下各 224


def test_decode_maps_back_and_nms_suppresses():
    ratio, pad_x, pad_y = 1024 / 960, 0, 224
    cx_m, cy_m = 480 * ratio, 250 * ratio + pad_y       # 原图(480,250) → 模型系
    w_m, h_m = 300 * ratio, 40 * ratio
    raw = np.array([[[cx_m, cy_m, w_m, h_m, 0.92, 0.05, 90.0],      # 主检（书脊）
                     [cx_m + 1.0, cy_m, w_m, h_m, 0.85, 0.05, 90.0],  # 近重复 → NMS 压制
                     [100.0, 100.0, 50.0, 50.0, 0.30, 0.60, 45.0]]],  # 弱且是 shelf → 过滤
                   dtype=np.float64)
    boxes = decode_obb_raw(raw, ratio, pad_x, pad_y, conf_threshold=0.5, class_id=0)
    assert len(boxes) == 1
    conf, cx, cy, w, h, ang = boxes[0]
    assert conf == pytest.approx(0.92)
    assert (cx, cy, w, h) == pytest.approx((480, 250, 300, 40), abs=1e-3)
    assert ang == pytest.approx(90.0)


def test_spine_bottom_from_synthetic_inference():
    """假 RKNN runtime + 合成帧：端到端得到书本最低点（垂直书脊底边 = cy + w/2）"""
    ratio, pad_y = 1024 / 960, 224
    cx_m, cy_m = 480 * ratio, 250 * ratio + pad_y
    raw = np.array([[[cx_m, cy_m, 300 * ratio, 40 * ratio, 0.92, 0.05, 90.0]]],
                   dtype=np.float64)

    class FakeRuntime:
        def inference(self, inputs):
            assert inputs[0].shape == (1, 1024, 1024, 3)
            return raw

    det = RknnSpineDetector(model_path="fake.rknn", runtime=FakeRuntime())
    spine = det(np.zeros((540, 960, 3), dtype=np.uint8))
    assert spine is not None
    assert spine.y == pytest.approx(400.0, abs=1e-3)    # 250 + 300/2
    assert spine.confidence == pytest.approx(0.92)
    assert spine.n_books == 1
    assert det.last_roll_deg == pytest.approx(0.0)      # 90° 竖直长轴 → roll=0


def test_model_path_empty_raises_before_hardware():
    with pytest.raises(ValueError, match="model_path"):
        RknnSpineDetector(model_path="")


def test_from_config_reads_nodes():
    cfg = {"control": {"spine_detector": {"model_path": "m.rknn",
                                          "input_size": 1024, "class_id": 0}},
           "perception": {"spine": {"min_confidence": 0.5, "max_boxes": 20}}}
    det = RknnSpineDetector.from_config(cfg, runtime=object())
    assert det.model_path == "m.rknn" and det.min_confidence == 0.5
