"""G1 实验工具：面识别方案基准 —— 纯面 / 面+线 两变体（用户发起的探索）

用法（项目根目录）：
    python tools/face_bench.py --cam 4

两个变体：
    纯面    搜索区内 Otsu 取亮部 + 横向形态学修补 → 连通块按块头门槛准入
            （面积/跨度/长宽比/带高度上下限）→ 读数 = 连通块质心 y
    面+线   选中连通块的"下边界"按列采点 → RANSAC 直线拟合 → 读数 = 画面中心截距
            （即"面定位、线读数"，附赠带高度作第二信号）

产出 runs/face_bench/：
    face_bench_cam{N}.csv   逐张两变体读数 + 与现行读数的偏差
    overlay_samples.png     代表帧叠加图（本人自检用）
所有阈值标 TODO(实验)，若采纳再进 configs。
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np

from pid_servo import utils
from pid_servo.perception import _ransac_line, yellow_crosses_blue, BarFit
from pid_servo.utils import imread_utf8, imwrite_utf8

# ---- 面识别参数（TODO(实验) 若采纳进 configs）----
CLOSE_KER = (151, 9)          # 横向修补核：把白牌和灰杆连成一块
GATE_W_FRAC = 0.40            # 面宽度 ≥ 40% 画宽（块头下限·跨度）
GATE_AREA_FRAC = 0.06         # 面积 ≥ 搜索区面积的 6%（块头下限·面积）
GATE_H_RANGE = (0.015, 0.09)  # 带高度占画高比的上限/下限（上限杀"整排书"冒充者）
GATE_ASPECT = 4.0             # 长宽比 ≥ 4
RANSAC_TOL = 12.0

MARK = {"4_B_097_L_8_1_4", "4_B_098_R_7_2_4"}


def face_component(img, sb):
    """搜索区内分割横杆带，返回准入的连通块掩码与元信息；不合格返回 (None, 原因)"""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lo = int((sb if sb else 0.85 * h) + 0.010 * h)
    zone = gray[lo:h, :]
    zh = zone.shape[0]
    if zh < 20:
        return None, "zone_too_thin", lo

    _, mask = cv2.threshold(zone, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, CLOSE_KER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best, best_reason = None, "no_component"
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw < GATE_W_FRAC * w:
            best_reason = best_reason if best is not None else f"too_narrow({bw / w:.2f}W)"
            continue
        if area < GATE_AREA_FRAC * zone.size:
            best_reason = f"too_small_area({area / zone.size:.2f})"
            continue
        hfrac = bh / h
        if not (GATE_H_RANGE[0] <= hfrac <= GATE_H_RANGE[1]):
            best_reason = f"bad_height({hfrac:.3f}H)"
            continue
        if bw / max(bh, 1) < GATE_ASPECT:
            best_reason = f"bad_aspect({bw / max(bh, 1):.1f})"
            continue
        if best is None or area > stats[best][4]:
            best = i
    if best is None:
        return None, best_reason, lo
    x, y, bw, bh, area = stats[best]
    comp = (lab == best)
    cy = float(np.mean(np.nonzero(comp)[0])) + lo          # 质心 y（全图坐标）
    return {"mask": comp, "lo": lo, "bbox": (x, y + lo, bw, bh),
            "area": int(area), "centroid_y": cy}, "ok", lo


def face_reading(img, sb):
    """纯面：质心读数。返回 (frac|None, info|reason)"""
    comp, reason, _ = face_component(img, sb)
    if comp is None:
        return None, reason
    return comp["centroid_y"] / img.shape[0], comp


def faceline_reading(img, sb):
    """面+线：下边界点云 → RANSAC → 中心截距 + 带高度。返回 (frac|None, slope, info)"""
    h, w = img.shape[:2]
    comp, reason, lo = face_component(img, sb)
    if comp is None:
        return None, None, reason
    mask = comp["mask"].astype(np.uint8)
    xs = np.nonzero(mask.any(axis=0))[0]
    if len(xs) < 30:
        return None, None, "too_few_columns"
    cols = xs[:: max(1, len(xs) // 120)]                   # 每块最多采 120 列
    pts = []
    for x in cols:
        col = np.nonzero(mask[:, x])[0]
        pts.append((x + 0.5, col.max() + 0.5 + lo))        # 该列最下沿
    P = np.asarray(pts, dtype=np.float32)
    k, b, inl = _ransac_line(P, RANSAC_TOL)
    y = float(k * w / 2 + b)
    fit = BarFit(y_center=y, slope=float(k), n_points=len(P),
                 inlier_ratio=float(inl.mean()), residual_std=0.0)
    if inl.mean() < 0.70 or yellow_crosses_blue(fit, sb, w):
        return None, float(k), {"reason": "line_health", "fit": fit, "comp": comp,
                                 "inl": inl, "pts": P}
    band_h = comp["bbox"][3] / h
    return y / h, float(k), {"fit": fit, "comp": comp, "band_h": band_h,
                             "inl": inl, "pts": P}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="4")
    args = ap.parse_args()

    cfg = utils.load()
    img_dir = os.path.join(utils.get(cfg, "data.dataset8_7"), "01_原图", f"cam_{args.cam}")
    spine = {}
    with open(os.path.join("runs", "s1_spine", f"spine_cam{args.cam}.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            spine[r["name"]] = float(r["spine_bottom"]) if r["spine_bottom"] else None
    cur = {}
    with open(os.path.join("runs", "s1_bar", f"bar_cam{args.cam}.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cur[r["name"]] = float(r["bar_frac"]) if (r["found"] and not r["reject"]) else None

    out_dir = os.path.join("runs", "face_bench")
    os.makedirs(out_dir, exist_ok=True)

    rows, samples = [], []
    for name in sorted(spine.keys()):
        img = imread_utf8(os.path.join(img_dir, name + ".jpg"))
        if img is None:
            continue
        sb = spine.get(name)
        f1, info1 = face_reading(img, sb)
        f2, slope2, info2 = faceline_reading(img, sb)
        rows.append({"name": name,
                     "face_frac": "" if f1 is None else f"{f1:.4f}",
                     "faceline_frac": "" if f2 is None else f"{f2:.4f}",
                     "cur_frac": "" if cur.get(name) is None else f"{cur[name]:.4f}",
                     "face_reason": info1 if f1 is None else "",
                     "faceline_reason": "" if f2 is not None else str(info2)})
        if name in MARK or len(samples) < 4 and np.random.default_rng(0).random() < 0.02:
            vis = img.copy()
            h, w = vis.shape[:2]
            if f2 is not None and isinstance(info2, dict) and "fit" in info2:
                ft = info2["fit"]; b0 = ft.y_center - ft.slope * (w / 2)
                cv2.line(vis, (0, int(b0)), (w, int(ft.slope * w + b0)), (0, 255, 255), 12, cv2.LINE_AA)
            if f1 is not None:
                cv2.putText(vis, f"FACE {f1:.4f}", (20, 2600), 0, 3, (0, 255, 0), 8, cv2.LINE_AA)
            if f2 is not None:
                cv2.putText(vis, f"FACE+LINE {f2:.4f}", (20, 2700), 0, 3, (0, 200, 255), 8, cv2.LINE_AA)
            if f1 is None:
                cv2.putText(vis, f"FACE NONE:{info1}", (20, 2600), 0, 2.5, (0, 0, 255), 6, cv2.LINE_AA)
            samples.append(cv2.resize(vis, (1280, 720)))
        if len(rows) % 50 == 0:
            print(f"  ... {len(rows)}")

    csv_path = os.path.join(out_dir, f"face_bench_cam{args.cam}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def summarize(key):
        vals = [float(r[key]) for r in rows if r[key]]
        pair = [(float(r[key]), r["cur_frac"]) for r in rows if r[key] and r["cur_frac"]]
        diffs = [a - float(b) for a, b in pair]
        s = f"{key}: 检出 {len(vals)}/247"
        if vals:
            s += f", 中位 {np.median(vals):.4f}"
        if diffs:
            s += f", 与现行偏差中位 {np.median(np.abs(diffs)):.4f} (P90 {np.percentile(np.abs(diffs), 90):.4f})"
        return s

    print("\n=== 面识别基准摘要 (cam_%s) ===" % args.cam)
    print(summarize("face_frac"))
    print(summarize("faceline_frac"))
    reasons = {}
    for r in rows:
        if not r["face_frac"]:
            key = r["face_reason"].split("(")[0]
            reasons[key] = reasons.get(key, 0) + 1
    print("纯面未检出原因分布:", reasons)
    print(f"CSV -> {csv_path}")

    if samples:
        sep = np.full((8, 1280, 3), 255, np.uint8)
        imwrite_utf8(os.path.join(out_dir, "overlay_samples.png"),
                     np.vstack([s for s in samples[:4]]))


if __name__ == "__main__":
    main()
