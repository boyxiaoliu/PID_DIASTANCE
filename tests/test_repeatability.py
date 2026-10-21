"""Step 4 测试 —— σ 定点实测工具（假观测链，PC 可跑）"""

import importlib
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from pid_servo.observation import BarEdge, Observation, SpineBottom


def make_tool():
    return importlib.import_module("measure_repeatability")


def fake_builder(fracs_and_modes):
    """按脚本逐帧产观测：元素为 (bar_frac | None, mode)"""
    calls = {"i": 0}

    def observe():
        i = calls["i"]
        calls["i"] += 1
        frac, mode = fracs_and_modes[min(i, len(fracs_and_modes) - 1)]
        bar = None if frac is None else BarEdge(y=frac * 540, contrast=0.9, y_int=1)
        spine = SpineBottom(y=470.0, confidence=0.9, n_books=12) if frac is not None else None
        obs = Observation(frame_size=(540, 960), spine=spine, bar=bar,
                          source="fake")
        obs.debug.update(total_ms=12.3)
        return obs

    def cleanup():
        pass

    return observe, cleanup


def read_report(out):
    with open(os.path.join(out, "report.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def test_pass_case_repeatability_gate(tmp_path):
    tool = make_tool()
    script = [(0.955 + (0.001 if i % 2 else -0.001), "yolo_bar") for i in range(30)]
    out = str(tmp_path / "r")
    rc = tool.main(["--frames", "30", "--interval-ms", "0", "--out", out],
                   builder=lambda cfg: fake_builder(script))
    assert rc == 0
    report = read_report(out)
    assert report["n_valid"] == 30 and report["found_rate"] == 1.0
    assert report["gate_pass"] is True
    # 15 个 +0.001 与 15 个 -0.001 的样本标准差（ddof=1）
    assert report["std"] == pytest.approx(0.001017, abs=1e-5)


def test_fail_case_above_caution(tmp_path):
    tool = make_tool()
    script = [(f, "yolo_bar") for f in np.linspace(0.905, 1.005, 30)]
    out = str(tmp_path / "r")
    rc = tool.main(["--frames", "30", "--interval-ms", "0", "--out", out],
                   builder=lambda cfg: fake_builder(script))
    assert rc == 1
    report = read_report(out)
    assert report["gate_pass"] is False
    assert report["std"] > 0.02 and "警戒线" in report["verdict"]


def test_low_found_rate_is_flagged(tmp_path):
    tool = make_tool()
    script = ([(0.955, "yolo_bar")] * 20 + [(None, "none")] * 10)   # 检出率 67%
    out = str(tmp_path / "r")
    rc = tool.main(["--frames", "30", "--interval-ms", "0", "--out", out],
                   builder=lambda cfg: fake_builder(script))
    assert rc == 1
    report = read_report(out)
    assert report["found_rate"] == pytest.approx(20 / 30, abs=1e-3)   # 报告值按 4 位小数舍入
    assert "检出率" in report["verdict"]


def test_too_few_valid_frames_undecidable(tmp_path):
    tool = make_tool()
    script = ([(0.955, "yolo_bar")] * 3 + [(None, "none")] * 27)    # 有效 <5
    out = str(tmp_path / "r")
    rc = tool.main(["--frames", "30", "--interval-ms", "0", "--out", out],
                   builder=lambda cfg: fake_builder(script))
    assert rc == 1
    report = read_report(out)
    assert report["std"] is None and "不可判定" in report["verdict"]


def test_assembly_failure_returns_2(tmp_path):
    tool = make_tool()

    def bad_builder(cfg):
        raise FileNotFoundError("no camera")

    rc = tool.main(["--out", str(tmp_path / "r")], builder=bad_builder)
    assert rc == 2
