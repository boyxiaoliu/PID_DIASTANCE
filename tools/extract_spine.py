"""S1-1 工具：从检测 JSON 批量提取 书本最低点 + 相机 roll（中位书脊倾角）

用法（项目根目录）：
    python tools/extract_spine.py [--cam 4]

产出：
    runs/s1_spine/spine_cam{N}.csv   每张图一行（大目录，不入库）
    runs/s1_spine/roll_hist_cam{N}.png  roll 分布直方图（数据体检用）
控制台打印分布摘要。 CSV 列:
    name, spine_bottom, spine_frac, n_books, min_conf, roll_deg
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pid_servo import utils
from pid_servo.perception import spine_stats_from_json_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="4", help="相机 id（对应数据目录名）")
    ap.add_argument("--out", default=None, help="输出目录，默认 runs/s1_spine")
    args = ap.parse_args()

    cfg = utils.load()
    json_root = utils.get(cfg, "perception.spine.offline_json_root")
    min_conf = float(utils.get(cfg, "perception.spine.min_confidence", 0.5))
    max_boxes = int(utils.get(cfg, "perception.spine.max_boxes", 20))
    cam_dir = os.path.join(json_root, f"cam_{args.cam}")

    out_dir = args.out or os.path.join("runs", "s1_spine")
    os.makedirs(out_dir, exist_ok=True)

    names = sorted(f for f in os.listdir(cam_dir) if f.endswith(".json"))
    rows = []
    for i, fn in enumerate(names):
        stem = fn[: -len("_cam_{}.json".format(args.cam))] if fn.endswith(
            f"_cam_{args.cam}.json") else os.path.splitext(fn)[0]
        try:
            spine, roll = spine_stats_from_json_file(
                os.path.join(cam_dir, fn), min_confidence=min_conf, max_boxes=max_boxes)
        except Exception as e:                          # 单文件损坏不拖垮整批，如实留痕
            print(f"[warn] {fn}: {e}")
            spine, roll = None, None
        rows.append({
            "name": stem,
            "spine_bottom": "" if spine is None else f"{spine.y:.1f}",
            "spine_frac": "" if spine is None else f"{spine.y / 2160.0:.4f}",  # TODO(calib) 硬编码 2160：JSON 不含分辨率，4K 相机实测
            "n_books": 0 if spine is None else spine.n_books,
            "min_conf": "" if spine is None else f"{spine.confidence:.3f}",
            "roll_deg": "" if roll is None else f"{roll:.2f}",
        })
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(names)}")

    csv_path = os.path.join(out_dir, f"spine_cam{args.cam}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ---- 分布摘要 ----
    rolls = [float(r["roll_deg"]) for r in rows if r["roll_deg"] != ""]
    fracs = [float(r["spine_frac"]) for r in rows if r["spine_frac"] != ""]
    empties = sum(1 for r in rows if r["spine_bottom"] == "")
    print(f"\n=== cam_{args.cam} 书本最低点提取摘要 ===")
    print(f"JSON 总数 {len(rows)}，其中无书脊检出 {empties} 张")

    def pct(v, ps=(5, 25, 50, 75, 90, 95)):
        import numpy as np
        a = np.percentile(v, ps)
        return ", ".join(f"P{p}={x:.2f}" for p, x in zip(ps, a))

    if rolls:
        print(f"roll（度）: {pct(rolls)}   |roll|>3° 共 {sum(abs(r) > 3 for r in rolls)} 张")
        plt.figure(figsize=(8, 4.5))
        plt.hist(rolls, bins=60, color="#4a7fb5")
        plt.axvline(3, color="r", ls="--", lw=1); plt.axvline(-3, color="r", ls="--", lw=1)
        plt.xlabel("roll (deg)"); plt.ylabel("frames")
        plt.title(f"cam_{args.cam} spine-angle roll  n={len(rolls)}")
        plt.tight_layout()
        hist_path = os.path.join(out_dir, f"roll_hist_cam{args.cam}.png")
        plt.savefig(hist_path, dpi=130); plt.close()
        print(f"roll 直方图 -> {hist_path}")
    if fracs:
        print(f"spine_frac: {pct(fracs)}")
    print(f"CSV -> {csv_path}")


if __name__ == "__main__":
    main()
