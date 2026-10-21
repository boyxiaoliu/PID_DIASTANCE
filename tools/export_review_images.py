"""G1 工具：把 cam 全量识别结果图导出到一个文件夹（资源管理器直接翻）

用法（项目根目录）：
    python tools/export_review_images.py --cam 4

产出 runs/识别图_cam{N}/：
    {状态}_{读数}_{图名}.jpg   —— 按名称排序即按读数排序（有效组从近到远）
    直方图.png / roll直方图.png
    说明.txt                   —— 蓝线/黄线图例
图上笔画与评审页同款（加粗）：黄线=横杆下沿读数，蓝线=书本最低点（搜索区上界），
绿点=内点，红点=离群，左上角黄字=读数与斜度/内点率。
"""

import argparse
import csv
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2

from pid_servo import utils
from pid_servo.perception import BarDetector
from pid_servo.utils import imread_utf8, imwrite_utf8

MARK = {"4_B_097_L_8_1_4": "完美基准097", "4_B_098_R_7_2_4": "偏远样例098"}


def status_of(r):
    """CSV 行 → (文件名前缀状态, 排序用读数字符串)

    分类语义（G1 用户目检定稿）：黄蓝相交 = 下沿不可见 = 太近，与间距饱和同族，
    统一归入"饱和太近按1.0计"家族，判定依据缀在名字里。
    """
    rej = r["reject"]
    if not rej:
        return "有效", f"{float(r['bar_frac']):.4f}"
    if rej.startswith("saturated_too_close"):
        return "饱和太近按1.0计", f"{float(r['bar_frac']):.4f}" if r["bar_frac"] else "1.0000"
    if rej.startswith("saturated_not_found"):
        return "饱和太近按1.0计", "1.0000"
    if rej.startswith("yellow_crosses_blue"):
        return "饱和太近按1.0计_相交判定", f"{float(r['bar_frac']):.4f}" if r["bar_frac"] else ""
    if rej.startswith("inliers"):
        return "剔除内点不足", f"{float(r['bar_frac']):.4f}" if r["bar_frac"] else ""
    if rej.startswith("roll"):
        return "剔除roll歪", ""
    if rej.startswith("slope"):
        return "剔除斜度过大", ""
    return "未检出", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="4")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = utils.load()
    img_dir = os.path.join(utils.get(cfg, "data.dataset8_7"), "01_原图", f"cam_{args.cam}")
    det = BarDetector.from_config()

    spine = {}
    with open(os.path.join("runs", "s1_spine", f"spine_cam{args.cam}.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            spine[r["name"]] = float(r["spine_bottom"]) if r["spine_bottom"] else None

    with open(os.path.join("runs", "s1_bar", f"bar_cam{args.cam}.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out = args.out or os.path.join("runs", f"识别图_cam{args.cam}")
    os.makedirs(out, exist_ok=True)

    n = 0
    for r in rows:
        name = r["name"]
        img = imread_utf8(os.path.join(img_dir, name + ".jpg"))
        if img is None:
            print(f"[warn] 缺原图: {name}")
            continue
        sb = spine.get(name)
        fit = det.detect(img, spine_bottom_y=sb)
        overlay = det.draw(img, fit, sb, thick=12, font=2.4, pr=16)
        overlay = cv2.resize(overlay, (1920, 1080), interpolation=cv2.INTER_AREA)

        status, frac = status_of(r)
        star = MARK.get(name)
        if star:
            status = f"{status}★{star}"
        fn = f"{status}_{frac}_{name}.jpg" if frac else f"{status}_{name}.jpg"
        imwrite_utf8(os.path.join(out, fn), overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
        n += 1
        if n % 50 == 0:
            print(f"  ... {n}/{len(rows)}")

    # 直方图与说明
    for src, dst in ((os.path.join("runs", "s1_bar", f"bar_hist_cam{args.cam}.png"), "直方图_bar_frac.png"),
                     (os.path.join("runs", "s1_spine", f"roll_hist_cam{args.cam}.png"), "直方图_roll.png")):
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out, dst))

    with open(os.path.join(out, "说明.txt"), "w", encoding="utf-8") as f:
        f.write("""横杆识别结果图（cam_%s，共 %d 张）
========================================
黄线 = 横杆下沿的检测位置（控制信号，左上角黄字为其读数）
蓝线 = 书本最低点（搜索区上界，正常应在黄线上方不远处）
绿点/红点 = 找到的边缘点（绿=与黄线一致，红=捣乱）

文件名 = 状态_读数_图名，按名称排序即按读数排序。三大类：
  有效_              读数可信（0.92~0.99，从近到远）——闭环 PID 正常推进
  饱和太近按1.0计_   太近、横杆下沿不可见（含"相交判定"子类），
                     闭环读数按 1.0 处理 = 只许后退
  剔除内点不足_      结构乱、读数不可信（丢失）
  剔除roll歪_        相机旋转歪超 3°（丢失）
  未检出_            找不到横杆（丢失）
""" % (args.cam, len(rows)))

    size_mb = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)) / 1e6
    print(f"导出 {n} 张 -> {out}  （{size_mb:.0f} MB）")


if __name__ == "__main__":
    main()
