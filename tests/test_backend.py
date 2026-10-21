"""Step 2 测试 —— RobotBackend 组装 + run_servo_real 干跑链路

全部 PC 本地：相机/书脊/横杆/底盘分别用假件或合成帧，不碰任何硬件。
"""

import importlib
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from pid_servo.observation import BarEdge, Observation, SpineBottom
from pid_servo.perception import BarDetector
from pid_servo.robot.backend import RobotBackend


# ========== 假件 ==========

class FakeCamera:
    """恒定返回预置帧；置 None 模拟相机失联"""

    def __init__(self, frame):
        self.frame = frame
        self.reads = 0

    def read(self):
        self.reads += 1
        return self.frame


class FakeSpineDetector:
    """记录收到的帧，返回预置 SpineBottom（Step 3 RKNN 件的占位形状）"""

    def __init__(self, spine):
        self.spine = spine
        self.seen_frames = []

    def __call__(self, frame):
        self.seen_frames.append(frame)
        return self.spine


class FakeBarDetector:
    """绕开图像处理，直接回一个带黄线的 Observation（形状与线A一致）"""

    name = "fake_bar"

    def observe(self, frame, spine, source=None):
        return Observation(frame_size=(0, 0) if frame is None else frame.shape[:2],
                           spine=spine,
                           bar=BarEdge(y=0.9 * (frame.shape[0] if frame is not None else 1),
                                       contrast=0.9, y_int=1),
                           source=source)


class FakeMover:
    def __init__(self):
        self.calls = []

    def move(self, delta_m):
        self.calls.append(delta_m)
        return {"x": 0.0, "y": 0.0, "yaw": 0.0}

    def stop(self):
        self.calls.append("STOP")


# ========== RobotBackend ==========

def test_observe_wires_camera_spine_bar_in_order():
    """相机帧 → 书脊检测 → 横杆组装，三件串成一条 Observation"""
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    cam = FakeCamera(frame)
    spine_det = FakeSpineDetector(SpineBottom(y=470.0, confidence=0.9, n_books=12))
    backend = RobotBackend(cam, spine_det, FakeMover(), FakeBarDetector(),
                           source="robot")

    obs = backend.observe()

    assert spine_det.seen_frames[0] is frame          # 书脊件收到的正是相机帧
    assert obs.spine is not None and obs.bar is not None
    assert obs.mode == "yolo_bar"
    assert obs.source.startswith("robot+fake_bar")
    for key in ("capture_ms", "infer_ms", "bar_ms", "total_ms"):
        assert key in obs.debug and obs.debug[key] is not None


def test_observe_camera_none_reports_lost_frame():
    """相机无帧：如实产出 mode=none 的观测（控制器会走丢失分支），不抛异常"""
    backend = RobotBackend(FakeCamera(None), FakeSpineDetector(None),
                           FakeMover(), FakeBarDetector())
    obs = backend.observe()
    assert obs.mode == "none"
    assert not obs.usable
    assert obs.debug["capture_ms"] is not None and obs.debug["bar_ms"] is None


def test_move_delegates_to_mover():
    mover = FakeMover()
    backend = RobotBackend(FakeCamera(None), None, mover, FakeBarDetector())
    p = backend.move(-0.07)
    assert mover.calls == [-0.07]
    assert p == {"x": 0.0, "y": 0.0, "yaw": 0.0}


# ---- 与线A真 BarDetector 的集成（合成帧，验证搜索区/读数整体可用） ----

def make_bar_frame(h=540, w=960, band_top=495, band_bottom=505):
    """灰底 + 下方一条黑横带（模拟横杆前缘的暗边）"""
    frame = np.full((h, w, 3), 200, dtype=np.uint8)
    frame[band_top:band_bottom, :] = 15
    return frame


def test_observe_with_real_bar_detector():
    """真横杆检测器 + 假相机/假书脊：观察结果应可用且读数落在横杆带附近"""
    frame = make_bar_frame()
    cam = FakeCamera(frame)
    spine_det = FakeSpineDetector(SpineBottom(y=470.0, confidence=0.9, n_books=12))
    backend = RobotBackend(cam, spine_det, FakeMover(), BarDetector(),
                           source="robot")

    obs = backend.observe()

    assert obs.usable, f"横杆未检出: {obs.debug}"
    frac = obs.bar_frac
    assert 0.85 <= frac <= 1.0, f"读数 {frac} 偏离横杆带（495~505/540）"
    assert frac == pytest.approx(500.0 / 540.0, abs=0.03)
    assert obs.debug["total_ms"] is not None


# ========== run_servo_real 干跑链路 ==========

@pytest.fixture()
def cli():
    return importlib.import_module("run_servo_real")


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_dry_run_end_to_end(cli, tmp_path):
    """干跑整链：闭环收敛 + trace.jsonl 结构完整（meta/iter/result）+ 逐轮耗时字段在"""
    out = str(tmp_path / "run")
    rc = cli.main(["--dry-run", "--d0", "0.9", "--d-star", "1.5",
                   "--noise-mult", "1", "--seed", "0", "--out", out])
    assert rc == 0, "seed=0 的 0.9→1.5 场景在 S3/S4 复验中均收敛"

    lines = read_jsonl(os.path.join(out, "trace.jsonl"))
    assert lines[0]["type"] == "meta" and lines[0]["target"] == 0.955
    iters = [ln for ln in lines if ln["type"] == "iter"]
    result = [ln for ln in lines if ln["type"] == "result"][0]

    assert result["converged"] is True and result["reason"] == "converged"
    assert len(iters) == result["iters"] >= 1
    for ln in iters:
        assert {"i", "mode", "state", "reading", "err", "out_m", "held"} <= set(ln)
        assert ln["total_ms"] is None          # 干跑无相机段，耗时如实为空
    assert iters[-1]["held"] >= 1


def test_dry_run_saturation_path_backs_off(cli, tmp_path):
    """太近起点（0.9m 出画区）：首轮必须是饱和/丢失态且指令为正（只许后退）"""
    out = str(tmp_path / "run")
    rc = cli.main(["--dry-run", "--d0", "0.9", "--d-star", "1.5",
                   "--noise-mult", "1", "--seed", "0", "--out", out])
    assert rc == 0
    lines = read_jsonl(os.path.join(out, "trace.jsonl"))
    iters = [ln for ln in lines if ln["type"] == "iter"]
    assert iters[0]["state"] in ("saturated", "lost")
    assert iters[0]["out_m"] > 0


def test_real_mode_refuses_safely_without_hardware(cli, tmp_path):
    """真机模式在无硬件的 PC 上：装配失败（相机打不开/模型未配）→ 返回 2，
    绝不带着残缺后端运行"""
    rc = cli.main(["--out", str(tmp_path / "run")])
    assert rc == 2
