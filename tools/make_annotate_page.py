"""G1 工具：人工标注页生成器 v3 —— 三分类盲标（布尔 + 比例有无，用户定稿语义）

用法（项目根目录）：
    python tools/make_annotate_page.py --cam 4

三分类语义（G1 用户定稿）：
    1 该给读数        横杆下沿清晰可见（布尔 True）
    2 不该给·蓝线在    太近，下沿不可见，但蓝线（书底/比例）还在画面最底边界（布尔 False，比例有）
    3 不该给·蓝线无    连书底都不该有——结构乱/相机歪/整帧没法看（布尔 False，比例无）

产出 runs/annotate_cam{N}.html：
    1/2/3 三键判定自动下一张，d=偷看系统叠加图，←=上一张；
    进度存浏览器（try/catch 保护）+ 导出/导入进度文件双保险；
    导出 CSV：name, sys_usable, sys_spine_frac, human_class, agree_bool,
              danger_false_pass（系统给了读数但人工判 2/3）。
v1→v2 教训：图片走相对路径引用（不内嵌，HTML 几十 KB），进度存储 try/catch 保护。
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pid_servo import utils


def sys_usable_of(reject, found):
    """系统布尔值：该帧有没有给出读数"""
    return 1 if (found and not reject) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="4")
    args = ap.parse_args()

    cfg = utils.load()
    raw_rel = f"../../图书馆数据8_7/01_原图/cam_{args.cam}"   # runs/ 的上一级的上一级
    ann_rel = f"识别图_cam{args.cam}"

    bar = {}
    with open(os.path.join("runs", "s1_bar", f"bar_cam{args.cam}.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            bar[r["name"]] = r
    spine_frac = {}
    with open(os.path.join("runs", "s1_spine", f"spine_cam{args.cam}.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            spine_frac[r["name"]] = r["spine_frac"]

    def ann_file_of(name, bar_frac, found):
        p1 = f"{ann_rel}/有效_{bar_frac}_{name}.jpg"
        if found == "1" and os.path.exists(os.path.join("runs", p1)):
            return p1
        for fn in os.listdir(os.path.join("runs", ann_rel)):
            if fn.endswith(name + ".jpg"):
                return f"{ann_rel}/{fn}"
        return None

    items = []
    for name, r in sorted(bar.items()):
        ann = ann_file_of(name, r["bar_frac"], r["found"])
        if ann is None:
            print(f"[warn] 缺叠加图跳过: {name}")
            continue
        items.append({"name": name,
                      "sys": sys_usable_of(r["reject"], r["found"]),
                      "spine": spine_frac.get(name, ""),
                      "raw": f"{raw_rel}/{name}.jpg", "ann": ann})

    order = sorted(range(len(items)), key=lambda i: (i * 7919 + 13) % len(items))  # 固定伪洗牌

    html = '''<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>S1 三分类盲标 — cam_4</title><style>
body{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:0;background:#17191c;color:#e8e8e8}
header{padding:10px 16px;background:#202329;position:sticky;top:0;z-index:9}
h1{font-size:15px;margin:0 0 4px}
.guide{font-size:12px;color:#bbb;line-height:1.7}
.bar{height:10px;background:#333;border-radius:5px;margin:8px 0;overflow:hidden}
.bar>div{height:100%;background:#3aa76d;width:0}
.wrap{max-width:1100px;margin:0 auto;padding:8px}
img{width:100%;display:block;border-radius:8px;background:#000;min-height:200px}
.btns{display:flex;gap:10px;margin:10px 0;justify-content:center}
button{font-size:15px;padding:10px 18px;border-radius:8px;border:1px solid #555;
       background:#2a2e34;color:#eee;cursor:pointer}
button:hover{background:#3a4048}
.tag{font-size:12px;color:#999;text-align:center;margin-top:4px}
.info{font-size:13px;color:#ccc;text-align:center;margin:6px 0}
.done{color:#3aa76d}
</style></head><body>
<header><h1>S1 三分类盲标（cam_4，__N__ 张）—— 这帧属于哪一类？</h1>
<div class="guide">
<b>1 该给读数</b> = 横杆下沿清晰可见（布尔 True）；<br>
<b>2 不该给·蓝线在</b> = 太近，下沿不可见，但蓝线（书底）还在画面最底边界一带（布尔 False，比例有）；<br>
<b>3 不该给·蓝线无</b> = 连书底都不该有——结构乱/相机歪/整帧没法看（布尔 False，比例无）。<br>
快捷键：<b>1/2/3</b> 判定并自动下一张 ｜ <b>d</b>=偷看系统叠加图(判完再对照) ｜ <b>←</b>=上一张改判。<br>
进度存浏览器（隔段点"导出进度"防丢）；标完点"导出标注CSV"交给 AI 出对账表。
</div>
<div class="bar"><div id="pbar"></div></div>
<div class="info" id="info"></div>
</header>
<div class="wrap">
<img id="rawImg" src="">
<div class="btns">
  <button onclick="judge(1)">1 该给读数</button>
  <button onclick="judge(2)">2 不该给·蓝线在</button>
  <button onclick="judge(3)">3 不该给·蓝线无</button>
</div>
<div class="tag" id="tag"></div>
</div>
<img id="peekImg" style="display:none;position:fixed;right:12px;bottom:12px;max-width:420px;
     border:2px solid #888;border-radius:8px;z-index:99" src="" alt="系统叠加图">
<script>
const DATA = __DATA__;
const ORDER = __ORDER__;
const KEY = "annotate_cam4_v5_class3";
const CLS = {1:"该给读数", 2:"不该给·蓝线在", 3:"不该给·蓝线无"};
function loadState(){
  try { return JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { return {}; }
}
let state = loadState();
let pos = 0;
function save(){ try { localStorage.setItem(KEY, JSON.stringify(state)); } catch(e) {} }
function cur(){ return DATA[ORDER[pos]]; }
function labeledCount(){ return ORDER.filter(i => state[DATA[i].name]).length; }
function render(){
  const it = cur();
  document.getElementById("rawImg").src = it.raw;
  const pk = document.getElementById("peekImg");
  pk.src = it.ann; pk.style.display = "none";
  document.getElementById("pbar").style.width = (100*labeledCount()/DATA.length) + "%";
  document.getElementById("info").innerHTML =
    `第 ${pos+1}/${ORDER.length} 张 ｜ 已标 <span class="done">${labeledCount()}</span>/${DATA.length}`;
  const v = state[it.name];
  document.getElementById("tag").textContent = v ? "已判：" + CLS[v] + "（可改判）" : "未判定";
}
function judge(v){
  state[cur().name] = v;
  save();
  if (pos < ORDER.length - 1) { pos++; }
  render();
}
function back(){ if (pos > 0) { pos--; render(); } }
function togglePeek(){
  const p = document.getElementById("peekImg");
  p.style.display = (p.style.display === "none") ? "block" : "none";
}
function downloadText(text, fname){
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], {type:"text/plain"}));
  a.download = fname; a.click();
}
function exportCSV(){
  let rows = "name,sys_usable,sys_spine_frac,human_class,agree_bool,danger_false_pass\n";
  for (const it of DATA){
    const h = state[it.name];
    if (!h){ rows += `${it.name},${it.sys},${it.spine},,,\n`; continue; }
    const agree = ((h === 1) === (it.sys === 1)) ? 1 : 0;
    const danger = (it.sys === 1 && h !== 1) ? 1 : 0;   // 危险假通过：系统给了读数、人工判 2/3
    rows += `${it.name},${it.sys},${it.spine},${h},${agree},${danger}\n`;
  }
  downloadText(rows, "annotate_cam4_human.csv");
}
function exportProgress(){ downloadText(JSON.stringify(state), "annotate_progress.json"); }
function importProgress(inp){
  const f = inp.files[0]; if (!f) return;
  const rd = new FileReader();
  rd.onload = () => { try { state = JSON.parse(rd.result); save(); render(); alert("进度已恢复"); }
                      catch(e) { alert("进度文件解析失败"); } };
  rd.readAsText(f);
}
document.addEventListener("keydown", e => {
  if (e.key === "1") judge(1);
  else if (e.key === "2") judge(2);
  else if (e.key === "3") judge(3);
  else if (e.key === "d" || e.key === "D") togglePeek();
  else if (e.key === "ArrowLeft") back();
});
render();
</script></body></html>'''
    html = (html.replace("__DATA__", json.dumps(items, ensure_ascii=False))
                .replace("__ORDER__", json.dumps(order))
                .replace("__N__", str(len(items))))

    out = os.path.join("runs", f"annotate_cam{args.cam}.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"标注页 v3 -> {out}  （{os.path.getsize(out) / 1024:.0f} KB，{len(items)} 张）")


if __name__ == "__main__":
    main()
