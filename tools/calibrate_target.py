"""S1-2/3/4 工具：横杆检测全量跑批 + 数据体检 + G1 候选图输出

用法（项目根目录）：
    python tools/calibrate_target.py --cam 4

流程：读 S1-1 的 spine CSV（书本最低点+roll）→ 逐张检测横杆（搜索区受书脊约束）
→ 按 health 阈值体检剔除 → 幸存样本画 bar_frac 分布 → 输出候选/样例 overlay。

产出（runs/s1_bar/，不入库）：
    bar_cam{N}.csv            逐张全部指标（含被剔除原因）
    bar_hist_cam{N}.png       幸存样本 bar_frac 直方图（标注 097/098）
    overlay_*.png             097/098 + 众数候选 + 各体检指标最差样本的可视化
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pid_servo import utils
from pid_servo.perception import BarDetector
from pid_servo.utils import imread_utf8, imwrite_utf8

#: 体检重点关注的样例与格位标记
MARK_NAMES = {"4_B_097_L_8_1_4", "4_B_098_R_7_2_4"}


def load_spine_csv(path):
    """读 S1-1 产出的 spine CSV → {name: (spine_bottom, roll_deg)}"""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sb = r["spine_bottom"]
            out[r["name"]] = (
                float(sb) if sb != "" else None,
                float(r["roll_deg"]) if r["roll_deg"] != "" else None,
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="4")
    ap.add_argument("--out", default=None, help="输出目录，默认 runs/s1_bar")
    ap.add_argument("--slope-max", type=float, default=None,
                    help="覆盖配置里的 slope_abs_max_deg（体检阈值扫描用）")
    args = ap.parse_args()

    cfg = utils.load()
    img_dir = os.path.join(utils.get(cfg, "data.dataset8_7"), "01_原图", f"cam_{args.cam}")
    spine_csv = os.path.join("runs", "s1_spine", f"spine_cam{args.cam}.csv")
    spine_info = load_spine_csv(spine_csv)

    det = BarDetector.from_config()
    roll_max = float(utils.get(cfg, "health.roll_max_deg", 3.0))
    min_inlier = float(utils.get(cfg, "health.min_inlier_ratio", 0.70))
    slope_max = args.slope_max if args.slope_max is not None else float(
        utils.get(cfg, "health.slope_abs_max_deg", 6.0))
    gap_min = float(utils.get(cfg, "health.gap_min_frac", 0.015))

    out_dir = args.out or os.path.join("runs", "s1_bar")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    names = sorted(spine_info.keys())
    for i, name in enumerate(names):
        spine_bottom, roll = spine_info[name]
        row = {"name": name, "found": 0, "bar_y": "", "bar_frac": "", "slope_deg": "",
               "inlier_ratio": "", "n_points": 0, "roll_deg": "" if roll is None else f"{roll:.2f}",
               "reject": ""}
        if roll is not None and abs(roll) > roll_max:
            row["reject"] = f"roll>{roll_max}deg"
            rows.append(row)
        else:
            img = imread_utf8(os.path.join(img_dir, name + ".jpg"))
            fit = det.detect(img, spine_bottom_y=spine_bottom) if img is not None else None
            if img is None:
                row["reject"] = "image_missing"
            elif fit is None:
                row["reject"] = "not_found"
            else:
                row.update(found=1, bar_y=f"{fit.y_center:.1f}",
                           bar_frac=f"{fit.y_center / img.shape[0]:.4f}",
                           slope_deg=f"{fit.slope_deg:.2f}",
                           inlier_ratio=f"{fit.inlier_ratio:.2f}",
                           n_points=fit.n_points)
                if fit.inlier_ratio < min_inlier:
                    row["reject"] = f"inliers<{min_inlier}"
                elif abs(fit.slope_deg) > slope_max:
                    row["reject"] = f"slope>{slope_max}deg"
                elif spine_bottom is not None:
                    # G1 用户目检发现的"黄蓝重合"态：横杆下沿出画、程序锁到上沿，
                    # 读数不可信且物理含义 = 太近（闭环里触发后退，绝不前进）
                    gap = fit.y_center / img.shape[0] - spine_bottom / img.shape[0]
                    if gap < gap_min:
                        row["reject"] = f"bar_top_suspect(间距{gap:.3f}<{gap_min})"
            rows.append(row)

        # overlay：样例 + 稍后按指标挑选的（先存样例，其余跑完统一挑）
        if name in MARK_NAMES and img is not None:
            imwrite_utf8(os.path.join(out_dir, f"overlay_{name}.png"),
                         det.draw(img, fit, spine_bottom))
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(names)}")

    # ---- 二次挑 overlay：众数候选 / 最差斜率 / 最差内点率 ----
    ok_rows = [r for r in rows if r["found"] and not r["reject"]]
    cache = {}

    def grab(name):
        if name not in cache:
            s, r_ = spine_info[name]
            img = imread_utf8(os.path.join(img_dir, name + ".jpg"))
            cache[name] = det.draw(img, det.detect(img, spine_bottom_y=s), s) if img is not None else None
        return cache[name]

    if ok_rows:
        med = float(np.median([float(r["bar_frac"]) for r in ok_rows]))
        near = sorted(ok_rows, key=lambda r: abs(float(r["bar_frac"]) - med))[:5]
        worst_slope = sorted(ok_rows, key=lambda r: -abs(float(r["slope_deg"])))[:3]
        worst_inl = sorted(ok_rows, key=lambda r: float(r["inlier_ratio"]))[:3]
        for r in near:
            imwrite_utf8(os.path.join(out_dir, f"overlay_near_{r['name']}.png"), grab(r["name"]))
        for r in worst_slope + worst_inl:
            imwrite_utf8(os.path.join(out_dir, f"overlay_worst_{r['name']}.png"), grab(r["name"]))

    # ---- CSV ----
    csv_path = os.path.join(out_dir, f"bar_cam{args.cam}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- 摘要 ----
    n_ok, n_roll, n_nf, n_inl, n_slope = (sum(1 for r in rows if c(r)) for c in (
        lambda r: r["found"] and not r["reject"],
        lambda r: r["reject"].startswith("roll"),
        lambda r: r["reject"] == "not_found",
        lambda r: r["reject"].startswith("inliers"),
        lambda r: r["reject"].startswith("slope")))
    print(f"\n=== cam_{args.cam} 横杆检测 + 体检摘要 ===")
    print(f"总数 {len(rows)} | 幸存 {n_ok} | 剔除: roll {n_roll} / 未检出 {n_nf} / 内点率 {n_inl} / 斜度 {n_slope}")

    def pct(v, ps=(5, 25, 50, 75, 90, 95)):
        a = np.percentile(v, ps)
        return ", ".join(f"P{p}={x:.4f}" for p, x in zip(ps, a))

    if ok_rows:
        fr = [float(r["bar_frac"]) for r in ok_rows]
        sl = [float(r["slope_deg"]) for r in ok_rows]
        il = [float(r["inlier_ratio"]) for r in ok_rows]
        print(f"bar_frac : {pct(fr)}")
        for tag, v in (("slope_deg", sl), ("inlier_ratio", il)):
            a = np.percentile(v, (50, 90, 99))
            print(f"{tag:13s}: P50={a[0]:.3f} P90={a[1]:.3f} P99={a[2]:.3f}")
        for mark in MARK_NAMES:
            m = next((r for r in rows if r["name"] == mark), None)
            print(f"  {mark}: " + (f"bar_frac={m['bar_frac']} slope={m['slope_deg']}° "
                                     f"inliers={m['inlier_ratio']} reject={m['reject'] or '无'}"
                                     if m and m["found"] else f"reject={m['reject'] if m else '?'}"))

        plt.figure(figsize=(8, 4.5))
        plt.hist(fr, bins=50, color="#4a7fb5")
        for name, color in (("4_B_097_L_8_1_4", "g"), ("4_B_098_R_7_2_4", "orange")):
            v = next((float(r["bar_frac"]) for r in rows if r["name"] == name and r["found"]), None)
            if v is not None:
                plt.axvline(v, color=color, lw=2, label=name[:14])
        plt.axvline(med, color="r", ls="--", lw=1, label=f"median={med:.4f}")
        plt.xlabel("bar_y / H"); plt.ylabel("frames"); plt.legend()
        plt.title(f"cam_{args.cam} bar_frac survivors n={len(fr)}")
        plt.tight_layout()
        hist_path = os.path.join(out_dir, f"bar_hist_cam{args.cam}.png")
        plt.savefig(hist_path, dpi=130); plt.close()
        print(f"直方图 -> {hist_path}")
    print(f"CSV -> {csv_path}")


if __name__ == "__main__":
    main()
