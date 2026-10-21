"""G1 工具：人工标注页生成器 —— 分类对错盲标（三键判定，单人全量）

用法（项目根目录）：
    python tools/make_annotate_page.py --cam 4

产出 runs/annotate_cam{N}.html（自包含，浏览器打开）：
    每张图默认显示无叠加原图（盲标），按键 1=有效 2=饱和 3=丢失 4=没把握，
    d=偷看系统叠加图（判完后对照），0/←=上一张，自动跳下一张。
    进度存浏览器 localStorage；"导出CSV"随时可点，导出含系统判定对照、
    一致标志与危险标志（系统判有效但人工不认）。
判定顺序按固定种子洗牌——防止按文件名顺序猜出系统分类趋势。
"""

import argparse
import base64
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2

from pid_servo import utils
from pid_servo.utils import imread_utf8

STATES = {"有效": 1, "饱和": 2, "丢失": 3}


def sys_state_of(reject, found):
    """CSV 判定 → 三态语义：有效 / 饱和 / 丢失"""
    if found and not reject:
        return "有效"
    if reject and (reject.startswith("saturated") or reject.startswith("yellow_crosses_blue")):
        return "饱和"
    return "丢失"


def jpeg_b64(img, width, quality):
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="4")
    args = ap.parse_args()

    cfg = utils.load()
    raw_dir = os.path.join(utils.get(cfg, "data.dataset8_7"), "01_原图", f"cam_{args.cam}")
    ann_dir = os.path.join("runs", f"识别图_cam{args.cam}")

    rows = list(csv.DictReader(open(os.path.join("runs", "s1_bar", f"bar_cam{args.cam}.csv"),
                                    encoding="utf-8")))

    items = []
    for i, r in enumerate(rows):
        name = r["name"]
        raw = imread_utf8(os.path.join(raw_dir, name + ".jpg"))
        ann_path = os.path.join(ann_dir, f"有效_{r['bar_frac']}_{name}.jpg")
        if not os.path.exists(ann_path):       # 非有效帧：叠加图按状态前缀命名，兜底扫描
            ann_path = None
            for fn in os.listdir(ann_dir):
                if fn.endswith(name + ".jpg"):
                    ann_path = os.path.join(ann_dir, fn)
                    break
        ann = imread_utf8(ann_path) if ann_path else None
        if raw is None or ann is None:
            print(f"[warn] 缺图跳过: {name}")
            continue
        items.append({
            "name": name,
            "sys": sys_state_of(r["reject"], r["found"] == "1"),
            "raw": jpeg_b64(raw, 1024, 52),
            "ann": jpeg_b64(ann, 800, 45),
        })
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(rows)}")

    import json
    import random
    rng = random.Random(20261018)
    order = list(range(len(items)))
    rng.shuffle(order)

    html = '''<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>S1 分类盲标 — cam_4</title><style>
body{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:0;background:#17191c;color:#e8e8e8}
header{padding:10px 16px;background:#202329;position:sticky;top:0;z-index:9}
h1{font-size:15px;margin:0 0 4px}
.guide{font-size:12px;color:#bbb;line-height:1.7}
.bar{height:10px;background:#333;border-radius:5px;margin:8px 0;overflow:hidden}
.bar>div{height:100%;background:#3aa76d;width:0}
.wrap{max-width:1024px;margin:0 auto;padding:8px}
img{width:100%;display:block;border-radius:8px}
.btns{display:flex;gap:10px;margin:10px 0;justify-content:center}
button{font-size:16px;padding:10px 22px;border-radius:8px;border:1px solid #555;
       background:#2a2e34;color:#eee;cursor:pointer}
button:hover{background:#3a4048}
.tag{font-size:12px;color:#999;text-align:center;margin-top:4px}
.peek{position:fixed;right:12px;bottom:12px;display:none;max-width:420px;border:2px solid #888;border-radius:8px}
.info{font-size:13px;color:#ccc;text-align:center;margin:6px 0}
.done{color:#3aa76d}
</style></head><body>
<header><h1>S1 分类盲标（cam_4，247 张）—— 只判"系统归的类对不对"</h1>
<div class="guide">
判定标准（看原图回答"真实状态"）：<b>1 有效</b>=横杆下沿清晰可见、可给出读数；
<b>2 饱和</b>=太近，下沿不可见或与书底融为一体；
<b>3 丢失</b>=结构乱/相机歪/认不出横杆。<br>
快捷键：<b>1/2/3</b> 判定并自动下一张 ｜ <b>4</b>=没把握(留空) ｜ <b>d</b>=偷看系统叠加图(判完再看) ｜
<b>0 或 ←</b>=上一张改判。进度自动存浏览器，随时点"导出CSV"，全部标完导出交给 AI 合并分析。
</div>
<div class="bar"><div id="pbar"></div></div>
<div class="info" id="info"></div>
</header>
<div class="wrap">
<img id="rawImg" src="">
<div class="btns">
  <button onclick="judge(1)">1 有效</button>
  <button onclick="judge(2)">2 饱和</button>
  <button onclick="judge(3)">3 丢失</button>
  <button onclick="judge(4)">4 没把握</button>
</div>
<div class="tag" id="tag"></div>
</div>
<img id="peekImg" class="peek" src="" alt="系统叠加图">
<script>
const DATA = __DATA__;
const ORDER = __ORDER__;
const KEY = "annotate_cam4_v1";
let state = JSON.parse(localStorage.getItem(KEY) || "{}");
let pos = 0;
function save(){ try{ localStorage.setItem(KEY, JSON.stringify(state)); }catch(e){} }
function cur(){ return DATA[ORDER[pos]]; }
function render(){
  const it = cur();
  document.getElementById("rawImg").src = "data:image/jpeg;base64," + it.raw;
  document.getElementById("peekImg").src = "data:image/jpeg;base64," + it.ann;
  document.getElementById("peekImg").style.display = "none";
  const labeled = ORDER.filter(i => state[DATA[i].name]).length;
  document.getElementById("pbar").style.width = (100*labeled/DATA.length) + "%";
  document.getElementById("info").innerHTML =
    `第 ${pos+1}/${ORDER.length} 张 ｜ 已标 <span class="done">${labeled}</span>/${DATA.length}`;
  const v = state[it.name];
  document.getElementById("tag").textContent =
    v ? "已判：" + ({1:"有效",2:"饱和",3:"丢失",4:"没把握"})[v] + "（可改判）" : "未判定";
}
function judge(v){
  state[cur().name] = v > 3 ? null : v;
  if (v > 3) delete state[cur().name];
  save();
  if (pos < ORDER.length - 1) { pos++; render(); }
  else { render(); alert("到最后一张了！点右上\"导出CSV\"交给 AI。"); }
}
function back(){ if (pos > 0) { pos--; render(); } }
function togglePeek(){
  const p = document.getElementById("peekImg");
  p.style.display = p.style.display === "none" ? "block" : "none";
}
function exportCSV(){
  let rows = "name,system,human,agree,danger\\n";
  for (const it of DATA){
    const h = state[it.name];
    const hn = h ? ({1:"有效",2:"饱和",3:"丢失",4:""})[h] : "";
    const agree = h ? (hn === it.sys ? "1" : "0") : "";
    const danger = (it.sys === "有效" && h && h !== 1) ? "1" : "";
    rows += `${it.name},${it.sys},${hn},${agree},${danger}\\n`;
  }
  const blob = new Blob([rows], {type: "text/csv"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "annotate_cam4_human.csv";
  a.click();
}
document.addEventListener("keydown", e => {
  if (e.key === "1") judge(1);
  else if (e.key === "2") judge(2);
  else if (e.key === "3") judge(3);
  else if (e.key === "4") judge(4);
  else if (e.key === "d" || e.key === "D") togglePeek();
  else if (e.key === "0" || e.key === "ArrowLeft") back();
});
render();
</script></body></html>'''
    html = html.replace("__DATA__", json.dumps(items, ensure_ascii=False))
    html = html.replace("__ORDER__", json.dumps(order))

    out = os.path.join("runs", f"annotate_cam{args.cam}.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"标注页 -> {out}  （{os.path.getsize(out) / 1e6:.1f} MB，{len(items)} 张，浏览器打开）")


if __name__ == "__main__":
    main()
