"""S3 蒙特卡洛闭环仿真 —— 控制器上岗考试

考试内容（docs/01_仿真设计_S3.md 定稿）：证明"读数 → 位移指令"这套控制器
（真机要跑的那份代码）在 0.9~3.4 米任意起点、含噪声和坏帧的情况下收敛，
且任何情况下不朝书架越线逼近。

用法（项目根目录）：

    python tools/sim_servo.py                  # 全网格 7×3×3×200 ≈ 12600 次
    python tools/sim_servo.py --quick          # 每格 20 种子冒烟
    python tools/sim_servo.py --single --d0 2.5 --noise-mult 3 --d-star 1.5
                                               # 单点逐轮明细
    python tools/sim_servo.py --set control.pid.kp=3.3 --out <dir>
                                               # 改增益对照

产出（--out 目录，默认 experiments/当日_S3闭环仿真/）：
    config_snapshot.yaml  当次全部参数快照
    metrics.csv           网格逐格指标
    failures.jsonl        未收敛样本逐轮轨迹（失败原因比成功率有用）
    sweep_metrics.csv     --kp-sweep 时的增益对照（仅最坏噪声档）
"""

import argparse
import csv
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import yaml

from pid_servo.servo import DistanceServo
from pid_servo.shelf_sim import run_sim
from pid_servo.utils import get, load


# ========== 配置覆盖（utils.apply_overrides 不存在，本地小实现） ==========

def apply_set(cfg, pairs):
    """把 ["control.pid.kp=3.3", ...] 写进配置字典，键不存在则报错防手滑"""
    for pair in pairs:
        key, _, val = pair.partition("=")
        if not _:
            raise ValueError(f"--set 需要 key=value 形式，收到: {pair}")
        cur = cfg
        keys = key.split(".")
        for k in keys[:-1]:
            if not isinstance(cur.get(k), dict):
                raise ValueError(f"--set 路径不存在: {key}")
            cur = cur[k]
        if keys[-1] not in cur:
            raise ValueError(f"--set 键不存在: {key}（先在 configs/default.yaml 里声明）")
        try:
            cur[keys[-1]] = yaml.safe_load(val)
        except yaml.YAMLError as e:
            raise ValueError(f"--set 值解析失败: {val}（{e}）")
    return cfg


# ========== 单次仿真 ==========

def sim_kwargs_from_cfg(cfg):
    """从 configs sim: 节组装 ShelfSim 模型参数（单一事实来源）"""
    return dict(
        A=get(cfg, "sim.A", 0.50),
        gap0_frac=get(cfg, "sim.gap0_frac", 0.058),
        dropout=get(cfg, "sim.dropout", 0.024),
        move_std=get(cfg, "sim.move_std", 0.02),
        bar_out_frac=get(cfg, "sim.bar_out_frac", 1.0),
        topedge_offset=get(cfg, "sim.topedge_offset", 0.008),
        topedge_std=get(cfg, "sim.topedge_std", 0.003),
        gap_noise_std=get(cfg, "sim.gap_noise_std", 0.005),
        img_h=int(get(cfg, "sim.img_h", 2160)),
        img_w=int(get(cfg, "sim.img_w", 3840)),
        d_floor=get(cfg, "sim.d_floor", 0.25),
    )


def analyze_run(res, sim, d_star, d0, margin_m):
    """从一次闭环提取考试指标（方向性 + 越线检查）

    - 方向性红线：饱和/丢失轮的指令必须 ≥ 0（不许前进）
    - 越线检查：合法起点（d0 ≥ d*−margin）下，动完后不得低于 d*−margin；
      起点已越线时，只要求不许比起点更近（先退出去是任务本身）
    """
    dir_violation = any(it.state in ("saturated", "lost") and it.out_m < 0
                        for it in res.trace)
    # 轨迹采样点 = 每轮 observe 时的 d（动前）；动后位置 = 下一轮动前 + 末距离
    post = [it.distance for it in res.trace[1:] if it.distance is not None]
    post.append(sim.distance)
    min_d_after = min(post)
    start_below = d0 < d_star - margin_m
    too_close = (not start_below) and min_d_after < d_star - margin_m
    moved_closer = start_below and min_d_after < d0 - 1e-9
    return dict(dir_violation=dir_violation, min_d_after=min_d_after,
                start_below=start_below, too_close=too_close,
                moved_closer=moved_closer)


def run_cell(servo, cfg, d_star, noise_mult, d0, n_seeds, sec_per_iter,
             margin_m, seed_base=20261018):
    """跑一个网格格子的 n_seeds 次闭环，返回聚合指标 + 失败样本列表"""
    noise_std = get(cfg, "sim.noise_std", 0.0299) * noise_mult
    sim_kwargs = sim_kwargs_from_cfg(cfg)
    conv = 0
    iters_ok = []
    min_ds = []
    dir_viol = too_close = moved_closer = 0
    reasons = {}
    failures = []

    for rep in range(n_seeds):
        # 种子由格点"数值"派生（非下标）：自定义网格也能跨机器复现
        seed = int(np.random.SeedSequence(
            [seed_base, int(d_star * 100), int(noise_mult * 100),
             int(d0 * 100), rep]).generate_state(1)[0])
        res, sim = run_sim(servo, d_star, d0, noise_std, seed=seed,
                           sim_kwargs=sim_kwargs)
        m = analyze_run(res, sim, d_star, d0, margin_m)
        conv += int(res.converged)
        if res.converged:
            iters_ok.append(res.iters)
        min_ds.append(m["min_d_after"])
        dir_viol += int(m["dir_violation"])
        too_close += int(m["too_close"])
        moved_closer += int(m["moved_closer"])
        reasons[res.reason] = reasons.get(res.reason, 0) + 1
        if not res.converged:
            failures.append(dict(
                d_star=d_star, noise_mult=noise_mult, d0=d0, seed=seed,
                reason=res.reason, d_final=round(sim.distance, 4),
                min_d_after=round(m["min_d_after"], 4),
                trace=[dict(i=it.i, state=it.state,
                            reading=None if it.reading is None else round(it.reading, 4),
                            err=None if it.err is None else round(it.err, 4),
                            out_m=round(it.out_m, 4), held=it.held,
                            d_observe=None if it.distance is None else round(it.distance, 4))
                       for it in res.trace]))

    its = np.array(iters_ok) if iters_ok else np.array([0])
    return dict(
        d_star=d_star, noise_mult=noise_mult, d0=d0, n_runs=n_seeds,
        n_converged=conv, conv_rate=conv / n_seeds,
        iters_mean=float(its.mean()), iters_p90=float(np.percentile(its, 90)),
        iters_max=int(its.max()),
        min_d_min=round(min(min_ds), 4), min_d_mean=round(float(np.mean(min_ds)), 4),
        n_dir_violation=dir_viol, n_too_close=too_close,
        n_moved_closer=moved_closer,
        fail_reasons=json.dumps(reasons, ensure_ascii=False),
        est_machine_min=round(sum(iters_ok) * sec_per_iter / 60.0, 1),
    ), failures


# ========== 打印 ==========

def print_summary(results, d_star_list, noise_mults, d0_list):
    """按 d* 分块打印 收敛率(d0×噪声) 矩阵 + 安全线统计"""
    print("\n" + "=" * 88)
    print("收敛率矩阵（行=初始距离 d0，列=噪声档，单元=收敛率/平均轮数）")
    print("=" * 88)
    for ds in d_star_list:
        print(f"\n--- d* = {ds:.1f} m ---")
        print(f"{'d0':>6} |" + "".join(f"  {n}×噪声        " for n in noise_mults))
        print("-" * (9 + 14 * len(noise_mults)))
        for d0 in d0_list:
            row = f"{d0:>6} |"
            for n in noise_mults:
                cell = next(r for r in results
                            if r["d_star"] == ds and r["noise_mult"] == n
                            and r["d0"] == d0)
                row += f"  {cell['conv_rate']:>5.1%} {cell['iters_mean']:4.1f}轮 "
            print(row)
    viol = sum(r["n_dir_violation"] for r in results)
    tc = sum(r["n_too_close"] for r in results)
    mc = sum(r["n_moved_closer"] for r in results)
    total = sum(r["n_runs"] for r in results)
    est_h = sum(r["est_machine_min"] for r in results) / 60.0
    print("-" * 88)
    print(f"方向性违规（饱和/丢失后前进）: {viol}   "
          f"合法起点越线: {tc}   起点已越线又更近: {mc}   （三项必须为 0）")
    print(f"总计 {total} 次闭环；折算真机耗时 ≈ {est_h:.1f} 小时"
          f"（收敛样本、每轮 5s 估算）")


def verdict(results, noise_mults):
    """按设计及格线给结论（docs/01_仿真设计_S3.md §四），返回 (是否全部通过, [逐条结论])

    硬门（必须为 0）：方向性违规（饱和/丢失后产生前进指令）。
    及格线：标称噪声收敛率 ≥99%，最高噪声档 ≥95%。
    安全线（d ≥ d*−0.15）：设计原文为记录项（每格 min(d)），此处报告计数与
    深度供分析 —— 应力档下它数学上不可为零（见实验 README 推导），不算硬门。
    """
    lines = []
    ok = True
    nominal = [r for r in results if r["noise_mult"] == 1]
    if nominal:
        worst = min(r["conv_rate"] for r in nominal)
        line = f"标称噪声(1×)最低收敛率 {worst:.1%}（及格线 ≥99%）"
        ok &= worst >= 0.99
        lines.append(("✅ " if worst >= 0.99 else "❌ ") + line)

    n_max = max(r["noise_mult"] for r in results)
    worst3 = min(r["conv_rate"] for r in results if r["noise_mult"] == n_max)
    line = f"最坏噪声({n_max:g}×)最低收敛率 {worst3:.1%}（及格线 ≥95%）"
    ok &= worst3 >= 0.95
    lines.append(("✅ " if worst3 >= 0.95 else "❌ ") + line)

    viol = sum(r["n_dir_violation"] for r in results)
    line = f"方向性违规（饱和/丢失后前进）合计 {viol}（设计硬门：必须为 0）"
    ok &= viol == 0
    lines.append(("✅ " if viol == 0 else "❌ ") + line)

    tc = sum(r["n_too_close"] for r in results)
    mc = sum(r["n_moved_closer"] for r in results)
    dmin = min(r["min_d_min"] for r in results)
    line = (f"安全线 d≥d*−0.15：合法起点越线 {tc} 起、已越线又更近 {mc} 起，"
            f"全程最小 d = {dmin:.3f}m（记录项：计数应趋 0，深度分析见 README）")
    lines.append(("✅ " if tc + mc == 0 else "⚠️ ") + line)
    return ok, lines


# ========== 主流程 ==========

D0_LIST, NOISE_MULTS, D_STAR_LIST = None, None, None   # main() 里从配置填充


def main():
    ap = argparse.ArgumentParser(description="S3 蒙特卡洛闭环仿真（控制器上岗考试）")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="产出目录，默认 experiments/<当日>_S3闭环仿真")
    ap.add_argument("--seeds", type=int, default=None, help="每格种子数，默认取配置 sim.seeds")
    ap.add_argument("--quick", action="store_true", help="冒烟模式：每格 20 种子")
    ap.add_argument("--d0", type=float, nargs="*", default=None, help="覆盖初始距离列表")
    ap.add_argument("--d-star", type=float, nargs="*", default=None, dest="d_star",
                    help="覆盖真实完美距离列表")
    ap.add_argument("--noise-mult", type=float, nargs="*", default=None,
                    dest="noise_mult", help="覆盖噪声倍率列表")
    ap.add_argument("--single", action="store_true", help="单点逐轮明细模式")
    ap.add_argument("--seed", type=int, default=0, help="--single 模式的种子")
    ap.add_argument("--kp-sweep", default=None, dest="kp_sweep",
                    help="增益对照：逗号分隔 kp 列表，只跑最坏噪声档，如 '3.0,3.3,4.5'")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="配置覆盖，如 --set control.pid.kp=3.3")
    args = ap.parse_args()

    global D0_LIST, NOISE_MULTS, D_STAR_LIST
    cfg = apply_set(load(args.config), args.overrides)

    D0_LIST = args.d0 or list(get(cfg, "sim.d0_list",
                                  [0.9, 1.2, 1.5, 2.0, 2.5, 3.0, 3.4]))
    NOISE_MULTS = args.noise_mult or list(get(cfg, "sim.noise_mults", [1, 2, 3]))
    D_STAR_LIST = args.d_star or list(get(cfg, "sim.d_star_list", [1.2, 1.5, 1.8]))
    if args.quick and args.seeds is None:
        args.seeds = 20
    n_seeds = args.seeds if args.seeds is not None else int(get(cfg, "sim.seeds", 200))
    sec_per_iter = float(get(cfg, "sim.sec_per_iter", 5))
    margin_m = float(get(cfg, "sim.safety_margin_m", 0.15))
    noise_std0 = float(get(cfg, "sim.noise_std", 0.0299))

    servo = DistanceServo.from_config(cfg)

    if args.single:
        # 单点逐轮明细：接合成书架，像真机一样observe→决策→move
        d0 = D0_LIST[0] if args.d0 is None else args.d0[0]
        ds = D_STAR_LIST[0] if args.d_star is None else args.d_star[0]
        nm = NOISE_MULTS[0] if args.noise_mult is None else args.noise_mult[0]
        print(f"单点: d0={d0}m  d*={ds}m  噪声={nm}×({nm * noise_std0:.4f})  "
              f"seed={args.seed}  target={servo.target_frac}±{servo.tol}")
        res, sim = run_sim(servo, ds, d0, nm * noise_std0, seed=args.seed,
                           sim_kwargs=sim_kwargs_from_cfg(cfg))
        print(f"{'轮':>3} {'状态':<10} {'读数':>7} {'误差':>8} {'指令':>7} "
              f"{'hold':>4} {'观测时d':>8}")
        last_d = d0
        for it in res.trace:
            r = "n/a" if it.reading is None else f"{it.reading:7.4f}"
            e = "n/a" if it.err is None else f"{it.err:+8.4f}"
            last_d = it.distance if it.distance is not None else last_d
            print(f"{it.i:>3} {it.state:<10} {r} {e} {it.out_m:>+7.3f} "
                  f"{it.held:>4} {last_d:>7.3f}")
        print(res.summary() + f"   末距离 {sim.distance:.3f}m")
        return 0 if res.converged else 1

    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "experiments", f"{date.today():%Y-%m-%d}_S3闭环仿真")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config_snapshot.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    print("=" * 88)
    print(f"S3 蒙特卡洛闭环仿真   读数模型 bar_frac = A + B/d + N(0, σ)   "
          f"target={servo.target_frac} ± {servo.tol}")
    print(f"网格: d0={D0_LIST} × 噪声={NOISE_MULTS}(σ0={noise_std0}) × "
          f"d*={D_STAR_LIST} × {n_seeds} 种子")
    print(f"PID: kp={servo.pid.cfg.kp} ki={servo.pid.cfg.ki} kd={servo.pid.cfg.kd} "
          f"限幅 {servo.pid.cfg.out_limit}/{servo.pid.cfg.out_limit_far}m "
          f"（|err|≥{servo.pid.cfg.far_threshold} 换档）")
    print("=" * 88)

    results, all_failures = [], []
    n_cells = len(D0_LIST) * len(NOISE_MULTS) * len(D_STAR_LIST)
    done = 0
    for ds in D_STAR_LIST:
        for nm in NOISE_MULTS:
            for d0 in D0_LIST:
                cell, fails = run_cell(servo, cfg, ds, nm, d0, n_seeds,
                                       sec_per_iter, margin_m)
                results.append(cell)
                all_failures.extend(fails)
                done += 1
                print(f"[{done:>2}/{n_cells}] d*={ds:.1f} 噪声{nm}× d0={d0:.1f}  "
                      f"收敛 {cell['n_converged']}/{cell['n_runs']} "
                      f"({cell['conv_rate']:.1%})  平均 {cell['iters_mean']:.1f} 轮  "
                      f"min_d_min={cell['min_d_min']:.3f}")

    # ---- 逐格指标与失败样本落盘 ----
    cols = list(results[0].keys())
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(results)
    with open(os.path.join(out_dir, "failures.jsonl"), "w", encoding="utf-8") as f:
        for rec in all_failures:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print_summary(results, D_STAR_LIST, NOISE_MULTS, D0_LIST)

    # ---- 增益对照（可选）：最坏噪声档下扫 kp ----
    if args.kp_sweep:
        sweep_rows = []
        nm_worst = max(NOISE_MULTS)
        base_kp = servo.pid.cfg.kp
        for kp_str in args.kp_sweep.split(","):
            kp = float(kp_str)
            servo.pid.cfg.kp = kp
            for ds in D_STAR_LIST:
                for d0 in D0_LIST:
                    cell, _ = run_cell(servo, cfg, ds, nm_worst, d0, n_seeds,
                                       sec_per_iter, margin_m)
                    cell["kp"] = kp
                    sweep_rows.append(cell)
            print(f"kp={kp}: {nm_worst}×噪声最低收敛率 "
                  f"{min(r['conv_rate'] for r in sweep_rows if r['kp'] == kp):.1%}")
        servo.pid.cfg.kp = base_kp
        with open(os.path.join(out_dir, "sweep_metrics.csv"), "w", newline="",
                  encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
            w.writeheader()
            w.writerows(sweep_rows)

    ok, lines = verdict(results, NOISE_MULTS)
    print("\n及格线判定（docs/01_仿真设计_S3.md）:")
    for ln in lines:
        print("  " + ln)
    print(("结论：全部通过 —— 有资格上真机（S5 复验）" if ok
           else "结论：未全过 —— 失败样本分析见 failures.jsonl，不要带病上车"))
    print(f"\n产出目录: {out_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    # Windows 控制台默认 GBK，中文输出/报错会乱码
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
