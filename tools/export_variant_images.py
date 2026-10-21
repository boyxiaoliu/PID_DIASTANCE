"""G1 工具：LSD / HoughLinesP+Canny 两种替代识别方案的全量结果图导出

用法（项目根目录）：
    python tools/export_variant_images.py --cam 4

产出（runs/ 下两个文件夹，各 247 张 + 说明）：
    识别图_cam{N}_LSD/     文件名 = 状态_读数_图名.jpg（按名排序=按读数排序）
    识别图_cam{N}_Hough/

背景（S1 实验档案"OpenCV 替代方案普查"节）：两方案均为"边缘线段→采样点云
→搜索区过滤→RANSAC 融合"，区别只在边缘源。健康口径与现行一致：内点率≥0.70
且点数≥8 才算有效读数。此导出仅供目检对比，非现役方案。
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np

from pid_servo import utils
from pid_servo.perception import _ransac_line
from pid_servo.utils import imread_utf8, imwrite_utf8

METHODS = ("lsd", "hough")
MARK = {"4_B_097_L_8_1_4": "★完美基准097", "4_B_098_R_7_2_4": "★偏远样例098"}

THICK, FONT, PR = 12, 2.4, 16


def points_along(x1, y1, x2, y2, step=24):
    n = max(2, int(np.hypot(x2 - x1, y2 - y1) // step))
    t = np.linspace(0, 1, n)
    return np.stack([x1 + (x2 - x1) * t, y1 + (y2 - y1) * t], axis=1)


def segments(gray2, kind, min_len_full, scale=0.5):
    """在缩放灰度图上找近水平线段，坐标还原到全图尺寸"""
    if kind == "lsd":
        lines = cv2.createLineSegmentDetector().detect(gray2)[0]
    else:
        edges = cv2.Canny(gray2, 40, 120)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                                minLineLength=int(0.06 * gray2.shape[1]), maxLineGap=20)
    segs = []
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = [v / scale for v in (x1, y1, x2, y2)]
            dx, dy = x2 - x1, y2 - y1
            if np.hypot(dx, dy) < min_len_full:
                continue
            if abs(dy) > 0.14 * max(abs(dx), 1):     # 近水平（≈8°内）
                continue
            segs.append((x1, y1, x2, y2))
    return segs


def variant_detect(img, sb, kind):
    """返回 (y_center|None, slope|None, 内点数, 总点数, 内点掩码, 点云)"""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g2 = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    segs = segments(g2, kind, min_len_full=0.04 * w)
    if not segs:
        return None, None, 0, 0, None, None
    pts = np.vstack([points_along(*s) for s in segs])
    lo = (sb if sb else 0.85 * h) + 0.010 * h
    pts = pts[pts[:, 1] >= lo]
    if len(pts) < 8:
        return None, None, 0, len(pts), None, pts
    k, b, inl = _ransac_line(pts.astype(np.float32), 12.0)
    ratio = inl.sum() / len(pts)
    if ratio < 0.70:
        return None, None, int(inl.sum()), len(pts), inl, pts
    return float(k * w / 2 + b), float(k), int(inl.sum()), len(pts), inl, pts


def draw(img, sb, y_center, slope, inl, pts):
    """评审页同款笔触：黄线=该方案读数，蓝线=书底，绿/红=内点/离群"""
    h, w = img.shape[:2]
    vis = img.copy()
    b0 = y_center - slope * (w / 2.0)
    if pts is not None:
        for (x, y), ok in zip(pts, inl):
            cv2.circle(vis, (int(x), int(y)), PR, (0, 255, 0) if ok else (0, 0, 255), -1, cv2.LINE_AA)
    cv2.line(vis, (0, int(b0)), (w, int(slope * w + b0)), (0, 255, 255), THICK, cv2.LINE_AA)
    if sb is not None:
        cv2.line(vis, (0, int(sb)), (w, int(sb)), (255, 128, 0), THICK, cv2.LINE_AA)
    cv2.putText(vis, f"y={y_center:.0f} ({y_center / h:.4f}H)",
                (20, 40 + PR * 3), cv2.FONT_HERSHEY_SIMPLEX, FONT, (0, 255, 255), THICK, cv2.LINE_AA)
    return vis


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
    names = sorted(spine.keys())

    outs = {m: os.path.join("runs", f"识别图_cam{args.cam}_{m.upper()}") for m in METHODS}
    for d in outs.values():
        os.makedirs(d, exist_ok=True)

    n_done = 0
    for name in names:
        img = imread_utf8(os.path.join(img_dir, name + ".jpg"))
        if img is None:
            print(f"[warn] 缺原图: {name}")
            continue
        sb = spine.get(name)
        for m in METHODS:
            y, slope, inl, npts, mask, pts = variant_detect(img, sb, m)
            if y is None:
                vis = img.copy()
                cv2.putText(vis, "NO LINE (inliers<0.7 or pts<8)", (20, 40 + PR * 3),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT, (0, 0, 255), THICK, cv2.LINE_AA)
                if sb:
                    cv2.line(vis, (0, int(sb)), (img.shape[1], int(sb)), (255, 128, 0), THICK, cv2.LINE_AA)
                fn = f"未检出_{name}.jpg"
            else:
                star = MARK.get(name, "")
                vis = draw(img, sb, y, slope, mask, pts)          # 先在 4K 上画
                fn = f"有效{star}_{y / 2160:.4f}_{name}.jpg"
            vis = cv2.resize(vis, (1920, 1080), interpolation=cv2.INTER_AREA)  # 后缩小
            imwrite_utf8(os.path.join(outs[m], fn), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        n_done += 1
        if n_done % 50 == 0:
            print(f"  ... {n_done}/{len(names)}")

    for m in METHODS:
        with open(os.path.join(outs[m], "说明.txt"), "w", encoding="utf-8") as f:
            f.write(f"""{m.upper()} 替代方案识别结果图（cam_{args.cam}）
========================================
黄线 = {m.upper()} 方案拟合的横杆线（左上角黄字为读数）
蓝线 = 书本最低点（搜索区上界）
绿点/红点 = 边缘采样点（绿=RANSAC 内点，红=离群）
有效口径：内点率 ≥ 0.70 且点数 ≥ 8（与现行方案同口径）
文件名 = 状态_读数_图名，按名称排序即按读数排序。
注意：该方案为 S1 普查实验产物（见 experiments/2026-10-17_S1误差标定/README.md），
非现役方案，仅供与 runs/识别图_cam{args.cam}/ 对比目检。
""")
    print(f"导出完成 -> {outs['lsd']} 与 {outs['hough']}（各 {n_done} 张）")


if __name__ == "__main__":
    main()
