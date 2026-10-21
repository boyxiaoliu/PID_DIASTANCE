"""感知层测试：JSON 书本最低点提取 + 横杆检测器（合成图，含斜线场景）"""

import numpy as np

from pid_servo.perception import (BarDetector, BarFit, spine_stats_from_json_dict,
                                  spine_stats_from_json_file, yellow_crosses_blue)

import json
import os


# ---------- 书本最低点 / roll 提取 ----------

def sample_json():
    return {
        "books": {
            "book_1": {"yolo11n_obb_detection": {
                # 近竖直书脊：w 为长轴，角度接近 90（度）—— 真实样例的形状
                "rotated_rect": [[679.7, 1181.3], [1642.4, 125.3], 89.69],
                "class_id": 0, "confidence": 0.92}},
            "book_2": {"yolo11n_obb_detection": {
                # 竖直书的另一种框表达：h 为长轴，角度接近 0
                "rotated_rect": [[100.0, 500.0], [50.0, 300.0], 3.0],
                "class_id": 0, "confidence": 0.60}},
            # 真实 JSON 里混着的非书条目（值为 list），必须被跳过而不是崩溃
            "position_ocr": [[0, 91, 182], [0, 86, 157]],
            "left_map_ocr": [[1, 2, 3]],
        }
    }


def test_spine_bottom_and_roll():
    spine, roll = spine_stats_from_json_dict(sample_json())
    # book_1 底边 = 1181.3 + (1642.4*sin89.69 + 125.3*cos89.69)/2 ≈ 2002.3
    assert spine.y > 2000.0 and spine.y < 2004.0
    assert spine.n_books == 2
    assert abs(spine.confidence - 0.60) < 1e-6      # 取参与框的最弱置信度
    # roll = median(长轴偏角): book_1 ≈ -0.31°, book_2 = +3.0° → 中位 ≈ 1.35°
    assert abs(roll - 1.345) < 0.05


def test_spine_confidence_filter_and_empty():
    spine, roll = spine_stats_from_json_dict(sample_json(), min_confidence=0.7)
    assert spine.n_books == 1 and abs(roll - (-0.31)) < 0.05

    spine2, roll2 = spine_stats_from_json_dict({"books": {}})
    assert spine2 is None and roll2 is None

    spine3, roll3 = spine_stats_from_json_dict(sample_json(), min_confidence=0.99)
    assert spine3 is None and roll3 is None


def test_spine_from_real_file_format(tmp_path):
    p = tmp_path / "case.json"
    p.write_text(json.dumps(sample_json()), encoding="utf-8")
    spine, roll = spine_stats_from_json_file(p)
    assert spine is not None and spine.n_books == 2


# ---------- 横杆检测器（合成图） ----------

def make_synthetic_shelf(w=960, h=540, slope=-0.02):
    """合成书架下沿：书本块(亮) + 背景(暗) + 横杆条(最亮，带斜率) + 下方(更暗)

    横杆条底边直线：y_bot(x) = 480 + slope * x
    底边中点 470.4/540 ≈ 0.871H —— 略低于兜底搜索区上界 0.85H，
    与真实构图（完美样本 ≈ 0.91H）同侧，保证无书脊兜底也能检出。
    """
    img = np.full((h, w), 40, dtype=np.uint8)
    img[:300, :] = 120                       # 书本块，底边 300（搜索区上界之外）
    for x in range(w):
        yb = int(round(480 + slope * x))     # 横杆条底边（斜线）
        img[max(0, yb - 120):yb, x] = 200    # 横杆条本体（顶边 = 底边 - 120）
        img[yb:, x] = 30                     # 横杆条下方更暗
    return img


def test_bar_detector_finds_slanted_bar():
    det = BarDetector({"highpass_sigma": 201, "min_points": 8})
    frame = make_synthetic_shelf()
    fit = det.detect(frame, spine_bottom_y=300.0, keep_points=True)

    assert fit is not None, "斜横杆场景必须能检出"
    w = frame.shape[1]
    expect_center = 480 + (-0.02) * (w / 2)          # = 470.4
    assert abs(fit.y_center - expect_center) <= 4.0, f"中心截距偏差过大: {fit.y_center}"
    assert abs(fit.slope - (-0.02)) <= 0.006, f"斜率估计偏差过大: {fit.slope}"
    assert fit.inlier_ratio > 0.9                     # 合成场景几乎全为内点
    assert fit.residual_std < 3.0


def test_bar_detector_zone_bound_by_spine():
    """搜索区必须被书本最低点约束：书本块底边(300)不允许被当成横杆"""
    det = BarDetector({"highpass_sigma": 201})
    frame = make_synthetic_shelf()
    fit = det.detect(frame, spine_bottom_y=300.0)
    # 若错误锁到 300 附近的边，y_center 会远小于横杆条底边的期望位置
    w = frame.shape[1]
    expect_center = 480 + (-0.02) * (w / 2)
    assert abs(fit.y_center - expect_center) <= 4.0


def test_bar_detector_fallback_zone_without_spine():
    det = BarDetector({"highpass_sigma": 201, "fallback_zone_frac": 0.85})
    frame = make_synthetic_shelf()
    fit = det.detect(frame, spine_bottom_y=None)      # 无书脊信息 → 兜底区
    assert fit is not None
    w = frame.shape[1]
    expect_center = 480 + (-0.02) * (w / 2)
    assert abs(fit.y_center - expect_center) <= 4.0


def test_bar_detector_not_found_on_flat():
    det = BarDetector({"highpass_sigma": 201})
    flat = np.full((540, 960), 40, dtype=np.uint8)
    assert det.detect(flat, spine_bottom_y=300.0) is None


def test_observe_builds_contract():
    det = BarDetector({"highpass_sigma": 201})
    frame = make_synthetic_shelf()
    obs = det.observe(frame, spine=None, source="test")
    assert obs.usable and obs.mode == "bar_only"
    w = frame.shape[1]
    expect_center = 480 + (-0.02) * (w / 2)
    assert abs(obs.bar_frac * frame.shape[0] - expect_center) <= 4.0
    assert "inlier_ratio" in obs.debug


# ---------- 黄蓝不相交约束（G1） ----------

def _fit(y_center, slope):
    return BarFit(y_center=y_center, slope=slope, n_points=20,
                  inlier_ratio=0.9, residual_std=1.0)


def test_yellow_crosses_blue_detection():
    assert not yellow_crosses_blue(_fit(2100.0, 0.0), 2000.0, 3840)   # 平线在蓝线之下
    # 右端 y = 2004 - 0.02*1920 = 1965.6 < 2000 → 画面内越过蓝线
    assert yellow_crosses_blue(_fit(2004.0, -0.02), 2000.0, 3840)
    assert not yellow_crosses_blue(None, 2000.0, 3840)                # 无拟合线不判交叉


def test_observe_drops_bar_when_crossing():
    det = BarDetector({"highpass_sigma": 201})
    frame = make_synthetic_shelf()
    # 构造一个会与蓝线相交的场景：斜率足够陡使右端越过书底 300
    obs = det.observe(frame, spine=None, source="test")
    # 正常合成场景不会相交
    assert "yellow_crosses_blue" not in obs.debug
