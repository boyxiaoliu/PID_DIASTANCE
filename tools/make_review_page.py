"""G1 评审页生成 v2：247 张全量检测结果 → 独立 HTML（浏览器打开）

v2 改版（v2 之前版本底部特写只有 ~56px 高，没法读）：
    - 底部裁剪纵向拉伸 ~3 倍 → 横杆线/内点/离群点清晰可辨
    - 每卡大字号读数 + 精简指标；按 近拍/中距/远拍尾部分三段，各配看点
    - 直方图放大，画 0.93/0.96/0.99 分档线与 097/098 标注

用法（项目根目录）：
    python tools/make_review_page.py --cam 4
产出 runs/s1_bar/review_cam{N}.html（base64 内嵌，单文件可拷贝）
"""

import argparse
import base64
import csv
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np

from pid_servo import utils
from pid_servo.perception import BarDetector
from pid_servo.utils import imread_utf8

MARK = {"4_B_097_L_8_1_4": "★ 完美基准", "4_B_098_R_7_2_4": "★ 偏远样例"}

#: 分段定义：(名称, 下限, 上限, 看点提示)
TIERS = [
    ("近拍族 0.96 ~ 1.0（离得最近，横杆带最清晰）", 0.96, 1.01,
     "看点：黄线是否压在灰色横杆带上？这族是'完美照片'的候选基准。"),
    ("中距族 0.93 ~ 0.96", 0.93, 0.96,
     "看点：和近拍族是否同一结构（横杆+标签牌）？能否算'合格'？"),
    ("远拍尾部 < 0.93（重点怀疑区）", 0.0, 0.93,
     "看点：黄线锁的是横杆，还是斜搁板沿/暗区？这里混入假信号的概率最大。"),
]


def jpeg_b64(img, width, quality=55, target_h=None):
    """缩放（可独立指定高度做纵向拉伸）+ base64 JPEG"""
    h, w = img.shape[:2]
    tw = min(width, w)
    th = int(h * tw / w) if target_h is None else target_h
    if (tw, th) != (w, h):
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA if tw < w else cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def make_histogram(fracs, marks):
    """放大版直方图：0.93/0.96 分档线 + 097/098 标注"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.hist(fracs, bins=60, color="#7da7d9", edgecolor="white", linewidth=0.4)
    for x, lab in ((0.93, ""), (0.96, "")):
        ax.axvline(x, color="#888", ls="--", lw=1.2)
        ax.text(x, ax.get_ylim()[1] * 0.97, f" {x}", fontsize=9, color="#666", va="top")
    for v, name, color in marks:
        if v is not None:
            ax.axvline(v, color=color, lw=2.2, label=f"{name}={v:.3f}")
    ax.axvline(float(np.median(fracs)), color="crimson", ls="--", lw=1.5,
               label=f"中位数={np.median(fracs):.4f}")
    ax.set_xlabel("bar_frac（横杆读数 = 横杆y / 画面高，越大=越近）")
    ax.set_ylabel("张数")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


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
            spine[r["name"]] = (float(r["spine_bottom"]) if r["spine_bottom"] else None,
                                float(r["roll_deg"]) if r["roll_deg"] else None)

    with open(os.path.join("runs", "s1_bar", f"bar_cam{args.cam}.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ok = sorted((r for r in rows if r["found"] and not r["reject"]),
                key=lambda r: -float(r["bar_frac"]))
    rejected = [r for r in rows if r["reject"] and r["reject"] != "not_found"]
    notfound = [r for r in rows if r["reject"] == "not_found"]
    print(f"幸存 {len(ok)} / 剔除 {len(rejected)} / 未检出 {len(notfound)}")

    fr = [float(r["bar_frac"]) for r in ok]
    marks = []
    for name in MARK:
        m = next((r for r in rows if r["name"] == name and r["found"]), None)
        marks.append((float(m["bar_frac"]) if m else None, MARK[name][2:].strip(),
                      "green" if "097" in name else "orange"))
    hist_b64 = make_histogram(fr, marks) if fr else ""

    def card(r):
        name = r["name"]
        sb, roll = spine.get(name, (None, None))
        img = imread_utf8(os.path.join(img_dir, name + ".jpg"))
        if img is None:
            return ""
        fit = det.detect(img, spine_bottom_y=sb)
        overlay = det.draw(img, fit, sb)
        crop = overlay[1830:2160, :]                       # 底部 15% 区域
        big = jpeg_b64(crop, 960, 50, target_h=300)        # 纵向拉伸 ~3.3 倍
        frac = float(r["bar_frac"]) if r["bar_frac"] else None
        head = (f'<div class="frac">{frac:.4f}</div>' if frac is not None
                else '<div class="frac na">未检出</div>')
        badges = f'<span class="badge star">{MARK[name]}</span>' if name in MARK else ""
        if r["reject"]:
            badges += f'<span class="badge bad">{r["reject"]}</span>'
        stats = (f"斜度 {r['slope_deg'] or '-'}° ｜ 内点 {r['inlier_ratio'] or '-'} ｜ "
                 f"书底 {(sb / 2160):.3f} ｜ roll {r['roll_deg'] or '-'}°")
        cat = "rej" if r["reject"] == "not_found" else ("rej" if r["reject"] else "ok")
        return f'''<div class="card" data-cat="{cat}">
  <div class="head">{head}<div class="meta2"><b class="nm">{name}</b>{badges}<div class="st">{stats}</div></div></div>
  <img class="full" loading="lazy" src="data:image/jpeg;base64,{jpeg_b64(overlay, 680, 48)}" alt="整幅叠加图">
  <details><summary>底部放大特写（纵向拉伸）</summary>
    <img class="crop" loading="lazy" src="data:image/jpeg;base64,{big}" alt="底部特写(纵向拉伸)">
  </details>
</div>'''

    def tier_html(lo, hi):
        sel = [r for r in ok if lo < float(r["bar_frac"]) <= hi]
        if not sel:
            return ""
        return (f'<section><h2>{name_of(lo, hi)}（{len(sel)} 张）</h2>'
                f'<p class="hint">{hint_of(lo, hi)}</p>'
                f'<div class="grid">{"".join(card(r) for r in sel)}</div></section>')

    def name_of(lo, hi):
        return next(t[0] for t in TIERS if t[1] == lo and t[2] == hi)

    def hint_of(lo, hi):
        return next(t[3] for t in TIERS if t[1] == lo and t[2] == hi)

    sections = "".join(tier_html(lo, hi) for _, lo, hi, _ in reversed(TIERS))
    sections += (f'<section><h2>被剔除（{len(rejected)}）</h2><p class="hint">看点：剔除得对不对？'
                 f'有没有被冤枉的好图（尤其内点率 0.5~0.7 的）？</p>'
                 f'<div class="grid">{"".join(card(r) for r in rejected)}</div></section>')
    sections += (f'<section><h2>未检出（{len(notfound)}）</h2><p class="hint">看点：这些多是"太近撑满画面"'
                 f'还是"结构太乱"？太近的按设计应触发后退。</p>'
                 f'<div class="grid">{"".join(card(r) for r in notfound)}</div></section>')

    html = f'''<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>S1 横杆检测评审 v2 — cam_{args.cam}</title><style>
body{{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:0;background:#f2f3f5;color:#1b1b1b}}
header{{position:sticky;top:0;background:#fff;padding:10px 16px;box-shadow:0 1px 4px rgba(0,0,0,.12);z-index:9}}
h1{{font-size:16px;margin:0 0 6px}}
.meta{{font-size:12px;color:#555;line-height:1.6}}
.hist{{max-width:860px;width:100%;display:block;margin:8px auto 2px}}
.legend{{background:#fff;border:1px solid #e2e4e8;border-radius:10px;margin:10px 16px;padding:10px 14px;font-size:13px;line-height:1.9}}
.legend .sw{{display:inline-block;width:26px;height:0;border-top:4px solid;vertical-align:3px;margin-right:6px}}
.toolbar{{font-size:13px;margin-top:4px}}
section h2{{font-size:15px;margin:18px 16px 2px}}
.hint{{font-size:12px;color:#666;margin:2px 16px 8px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));gap:12px;padding:0 14px 8px}}
.card{{background:#fff;border:1px solid #e2e4e8;border-radius:10px;overflow:hidden}}
.head{{display:flex;gap:10px;align-items:center;padding:8px 10px}}
.frac{{font-size:22px;font-weight:700;color:#0b62a4;min-width:86px;font-variant-numeric:tabular-nums}}
.frac.na{{font-size:15px;color:#b3261e}}
.meta2{{flex:1;min-width:0}}
.nm{{font-size:12px;margin-right:6px;word-break:break-all}}
.badge{{font-size:11px;padding:1px 7px;border-radius:10px;vertical-align:1px}}
.badge.bad{{background:#fde8e8;color:#b3261e}} .badge.star{{background:#fff3cd;color:#7a5b00;font-weight:600}}
.st{{font-size:11.5px;color:#666;margin-top:2px}}
.full{{width:100%;display:block;background:#222}}
.crop{{width:100%;display:block;background:#ddd}}
details{{padding:4px 10px 8px;font-size:12px}} summary{{cursor:pointer;color:#777}}
details img{{width:100%;margin-top:6px}}
.hide{{display:none}}
</style></head><body>
<header><h1>S1 横杆检测评审 v2 — cam_{args.cam}（{len(rows)} 张）</h1>
<div class="meta">卡片主图 = <b>整幅照片</b>（带检测结果叠加）。大数字 = 程序量出的横杆读数（越大=离书架越近）。</div>
<div class="legend">
<b>图里的两条线，别搞混：</b><br>
<span class="sw" style="border-color:#00a8ff"></span><b>蓝线 = 书本的最低点</b>（YOLO 检测书脊算出来的）。近距离照片里它刚好贴着横杆上沿，看起来像"横杆线"，其实它是书的位置，只用来限定搜索范围（老师说的"像素上界"）。<br>
<span class="sw" style="border-color:#ffd400"></span><b>黄线 = 横杆读数（控制信号）</b>，锁横杆结构的下沿。机器人以后就靠这个数字决定前进/后退。<br>
<span class="sw" style="border-color:#00c853"></span>绿点 = 和黄线一致的边缘点；<span class="sw" style="border-color:#ff1744"></span>红点 = 捣乱的点。红点多 = 这张图量得不可靠。<br>
<b>为什么黄线能当地尺：</b>完美基准 097 黄线在 <b>0.989</b>（横杆快沉到画面底 = 离得近）；你判断"要前进"的 098 黄线在 <b>0.932</b>（横杆还悬着 = 离得远）。离得越近读数越大——数字和你眼睛的判断一致。
</div>
<img class="hist" src="data:image/png;base64,{hist_b64}" alt="bar_frac 直方图">
<div class="toolbar">显示：
<label><input type="checkbox" checked onchange="f('ok',this.checked)"> 有效</label>
<label><input type="checkbox" checked onchange="f('rej',this.checked)"> 剔除/未检出</label>
</div></header>
{sections}
<script>function f(cat, show){{document.querySelectorAll('.card[data-cat="'+cat+'"]').forEach(e => e.classList.toggle('hide', !show));}}</script>
</body></html>'''

    out = args.out or os.path.join("runs", "s1_bar", f"review_cam{args.cam}.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"评审页 v2 -> {out}  （{os.path.getsize(out) / 1e6:.1f} MB）")


if __name__ == "__main__":
    main()
