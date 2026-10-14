"""契约 Observation 的行为测试 —— 两线共用的接口语义，改动需格外谨慎"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pid_servo.observation import BarEdge, Observation, SpineBottom


def make(h=2160, w=3840, spine_y=2000.0, bar_y=1980.0, with_bar=True, with_spine=True):
    return Observation(
        frame_size=(h, w),
        spine=SpineBottom(y=spine_y, confidence=0.9, n_books=12) if with_spine else None,
        bar=BarEdge(y=bar_y, contrast=30.0, y_int=int(bar_y)) if with_bar else None,
        source="test",
    )


def test_bar_frac_is_normalized():
    obs = make(bar_y=1944.0)          # 1944 / 2160 = 0.90
    assert abs(obs.bar_frac - 0.90) < 1e-9


def test_bar_frac_none_when_bar_missing():
    obs = make(with_bar=False)
    assert obs.bar_frac is None
    assert not obs.usable
    assert obs.mode == "none"


def test_mode_yolo_bar():
    obs = make()
    assert obs.mode == "yolo_bar"
    assert obs.usable


def test_mode_bar_only_is_usable_fallback():
    obs = make(with_spine=False)
    assert obs.mode == "bar_only"
    assert obs.usable                 # 误差只依赖横杆，书脊丢失只降级不弃帧


def test_spine_bottom_semantics():
    s = SpineBottom(y=2002.75, confidence=0.84, n_books=7)
    assert s.n_books == 7 and s.y > 0


def test_empty_frame_is_not_usable():
    obs = Observation(frame_size=(0, 0), source="test")
    assert obs.bar_frac is None
    assert obs.mode == "none"
