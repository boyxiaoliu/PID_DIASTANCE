"""G1 评审页生成：247 张全量检测结果 → 独立 HTML（浏览器打开）

用法（项目根目录）：
    python tools/make_review_page.py --cam 4

产出 runs/s1_bar/review_cam{N}.html：
    - 幸存样本按 bar_frac 从高到低（近拍→远拍），被剔除样本按原因分组
    - 每张卡片：底部检测特写（含拟合线/内点）+ 全图缩略 + 指标（读数/斜度/内点率/roll）
    - 097(完美基准)/098(偏远样例) 高亮；顶部嵌入 bar_frac 直方图
图片全部 base64 内嵌，单文件可拷贝给同伴。
"""

import argparse
import base64
import csv
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2

from pid_servo import utils
from pid_servo.perception import BarDetector
from pid_servo.utils import imread_utf8

MARK = {"4_B_097_L_8_1_4": "★ 完美基准", "4_B_098_R_7_2_4": "★ 偏远样例"}


def jpeg_b64(img, width, quality=60):
    """缩放 + 编码成 base64 JPEG"""
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii")


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

    rows = []
    with open(os.path.join("runs", "s1_bar", f"bar_cam{args.cam}.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ok = sorted((r for r in rows if r["found"] and not r["reject"]),
                key=lambda r: -float(r["bar_frac"]))
    rejected = [r for r in rows if r["reject"] and r["reject"] != "not_found"]
    notfound = [r for r in rows if r["reject"] == "not_found"]
    print(f"幸存 {len(ok)} / 剔除 {len(rejected)} / 未检出 {len(notfound)}")

    hist_path = os.path.join("runs", "s1_bar", f"bar_hist_cam{args.cam}.png")
    hist_b64 = base64.b64encode(open(hist_path, "rb").read()).decode("ascii") if os.path.exists(hist_path) else ""

    cards = []

    def build_card(r, img):
        name = r["name"]
        sb, roll = spine.get(name, (None, None))
        fit = det.detect(img, spine_bottom_y=sb)
        overlay = det.draw(img, fit, sb)
        crop = overlay[1950:2160, :]
        frac = r["bar_frac"] or "-"
        stats = (f"<span>读数 <b>{frac}</b></span><span>斜度 {r['slope_deg'] or '-'}°</span>"
                 f"<span>内点 {r['inlier_ratio'] or '-'}</span>"
                 f"<span>书底 {r['name'] and (sb / 2160 if sb else 0) and f'{sb / 2160:.3f}'}</span>"
                 f"<span>roll {r['roll_deg'] or '-'}°</span>")
        star = MARK.get(name, "")
        badges = ""
        if star:
            badges += f'<span class="badge star">{star}</span>'
        if r["reject"]:
            badges += f'<span class="badge bad">{r["reject"]}</span>'
        else:
            badges += '<span class="badge ok">幸存</span>'
        return f'''<div class="card" data-cat="{"rej" if r["reject"] else ("nf" if not r["found"] else "ok")}">
  <div class="head"><b>{name}</b>{badges}</div>
  <img class="crop" loading="lazy" src="data:image/jpeg;base64,{jpeg_b64(crop, 1024, 62)}" alt="底部检测特写">
  <div class="stats">{stats}</div>
  <details><summary>全图</summary>
    <img class="thumb" loading="lazy" src="data:image/jpeg;base64,{jpeg_b64(overlay, 480, 55)}" alt="全图叠加">
  </details>
</div>'''

    for i, r in enumerate(ok + rejected + notfound):
        img = imread_utf8(os.path.join(img_dir, r["name"] + ".jpg"))
        if img is None:
            continue
        cards.append(build_card(r, img))
        if (i + 1) % 40 == 0:
            print(f"  ... {i + 1}/{len(rows)}")

    html = f'''<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>S1 横杆检测评审 cam_{args.cam}</title><style>
body{{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:0;background:#f2f3f5;color:#1b1b1b}}
header{{position:sticky;top:0;background:#fff;padding:10px 16px;box-shadow:0 1px 4px rgba(0,0,0,.1);z-index:9}}
h1{{font-size:16px;margin:0 0 6px}} .meta{{font-size:12px;color:#555}}
.hist{{max-width:720px;width:100%;display:block;margin:8px 0}}
.toolbar{{margin:8px 0;font-size:13px}}
section h2{{font-size:14px;margin:16px 16px 6px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:10px;padding:0 10px 10px}}
.card{{background:#fff;border:1px solid #e2e4e8;border-radius:8px;overflow:hidden}}
.card .head{{display:flex;gap:6px;align-items:center;padding:6px 8px;font-size:12px;flex-wrap:wrap}}
.badge{{font-size:11px;padding:1px 7px;border-radius:10px}}
.badge.ok{{background:#e3f4e8;color:#137333}} .badge.bad{{background:#fde8e8;color:#b3261e}}
.badge.star{{background:#fff3cd;color:#7a5b00;font-weight:600}}
.stats{{font-size:12px;padding:4px 8px 6px;display:flex;flex-wrap:wrap;gap:10px;color:#444}}
.crop{{width:100%;display:block;background:#ddd}}
details{{padding:0 8px 8px;font-size:12px}} summary{{cursor:pointer;color:#666}}
details img{{width:100%;margin-top:6px}}
.hide{{display:none}}
</style></head><body>
<header><h1>S1 横杆检测评审 — cam_{args.cam}（{len(rows)} 张全量）</h1>
<div class="meta">幸存 {len(ok)}（按读数从高到低 = 从近拍到远拍） ｜ 剔除 {len(rejected)}（按原因） ｜ 未检出 {len(notfound)}。
看点：①读数 0.94~0.99 一族是否都是"合格构图" ②远拍尾部（读数&lt;0.93）锁的是不是横杆 ③完美照片对应读数该定多少。</div>
<img class="hist" src="data:image/png;base64,{hist_b64}" alt="bar_frac 直方图">
<div class="toolbar">显示：
<label><input type="checkbox" checked onchange="f('ok',this.checked)"> 幸存</label>
<label><input type="checkbox" checked onchange="f('rej',this.checked)"> 剔除</label>
<label><input type="checkbox" onchange="f('nf',this.checked)"> 未检出</label>
</div></header>
<section><h2>幸存样本（{len(ok)}）</h2><div class="grid">{''.join(c for c, r in zip(cards, ok + rejected + notfound) if not r["reject"] and r["found"])}</div></section>
<section><h2>被剔除（{len(rejected)}）</h2><div class="grid">{''.join(c for c, r in zip(cards, ok + rejected + notfound) if r["reject"] and r["reject"] != "not_found")}</div></section>
<section><h2>未检出（{len(notfound)}）</h2><div class="grid">{''.join(c for c, r in zip(cards, ok + rejected + notfound) if r["reject"] == "not_found")}</div></section>
<script>
function f(cat, show){{
  document.querySelectorAll('.card[data-cat="'+cat+'"]').forEach(el => el.classList.toggle('hide', !show));
}}
</script></body></html>'''

    out = args.out or os.path.join("runs", "s1_bar", f"review_cam{args.cam}.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"评审页 -> {out}  （{os.path.getsize(out) / 1e6:.1f} MB，浏览器直接打开）")


if __name__ == "__main__":
    main()
