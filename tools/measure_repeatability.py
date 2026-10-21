"""S4 硬门禁工具 —— 定点单帧重复性实测（机器人不动，连拍 N 帧量 σ）

为什么存在（S3 实验 README 的硬性行动项）：
    S3 的"全过"结论是条件结论——前提是检测器逐帧重复性 σ ≤ 0.015。
    S1 拿到的 0.0299 是 247 张不同构图照片的散布（构图差异 ≠ 逐帧噪声），
    真单帧 σ 只能在真机上固定距离连拍实测。本工具就是测它的。

判定线（S3 实验定稿，来源见 experiments/2026-10-18_S3闭环仿真/README.md）：
    σ ≤ 0.015  重复性档门槛 —— S3"重复性档全过"结论直接适用，可上车
    σ ≤ 0.02   警戒线内 —— 保守按 2×~3× 档评估，或加宽容差带再上车
    σ > 0.02   超警戒线 —— 先排查（对焦锁定？光照？结构锁定？），再谈闭环

用法（板端项目根目录；机器人摆到拍摄位后保持不动）：

    python tools/measure_repeatability.py --frames 30
    python tools/measure_repeatability.py --frames 50 --interval-ms 300 --set camera.light.enabled=true

产出 runs/repeatability/<时间戳>/report.json（不入库；结论摘录进 experiments/ 留痕）。
退出码：0=达标  1=不达标/样本不足  2=装配失败
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pid_servo.utils import get, load
from sim_servo import apply_set

#: 判定线（比例域；来源 = S3 闭环仿真的两档场景设计，勿随手改）
GATE_REPEATABILITY = 0.015
GATE_CAUTION = 0.02


class _NoMove:
    """定点实测不允许移动——move 被调用即视为程序错误"""

    def move(self, delta_m):
        raise RuntimeError("定点实测不移动底盘")


def build_observer(cfg):
    """真机观测链（与闭环同一套：相机帧 → RKNN 书脊 → 横杆），不含底盘"""
    from pid_servo.perception import BarDetector
    from pid_servo.robot.backend import RobotBackend
    from pid_servo.robot.camera import CameraBackend
    from pid_servo.robot.spine_detector import RknnSpineDetector

    camera = CameraBackend.from_config(cfg)
    camera.start()
    try:
        spine = RknnSpineDetector.from_config(cfg)
        bar = BarDetector((cfg or {}).get("perception", {}).get("bar", {}))
        backend = RobotBackend(camera, spine, _NoMove(), bar,
                               source="repeatability")
    except Exception:
        camera.close()
        raise
    return backend.observe, camera.close


def summarize(frames, n_requested):
    """把逐帧记录折算成报告 dict（σ 用样本标准差 ddof=1）"""
    fracs = np.array([f["bar_frac"] for f in frames if f["bar_frac"] is not None])
    n_valid = int(fracs.size)
    report = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "n_requested": n_requested,
        "n_valid": n_valid,
        "found_rate": round(n_valid / n_requested, 4) if n_requested else 0.0,
    }
    if n_valid < 5:
        report.update(std=None, verdict="样本太少，不可判定（有效帧 <5）",
                      gate_pass=False)
        return report

    std = float(fracs.std(ddof=1))
    report.update(
        mean=round(float(fracs.mean()), 5),
        std=round(std, 5),
        p05=round(float(np.percentile(fracs, 5)), 5),
        p50=round(float(np.percentile(fracs, 50)), 5),
        p95=round(float(np.percentile(fracs, 95)), 5),
        min=round(float(fracs.min()), 5),
        max=round(float(fracs.max()), 5),
    )
    if std <= GATE_REPEATABILITY:
        gate_pass = True
        report.update(verdict=f"σ={std:.4f} ≤ {GATE_REPEATABILITY}：达到重复性档门槛，"
                              f"S3 重复性档'全过'结论直接适用", gate_pass=gate_pass)
    elif std <= GATE_CAUTION:
        report.update(verdict=f"σ={std:.4f} 落在门槛与警戒线({GATE_CAUTION})之间："
                              f"保守按 2×~3× 档评估，或加宽容差带后再上车", gate_pass=False)
    else:
        report.update(verdict=f"σ={std:.4f} > {GATE_CAUTION} 警戒线：先排查对焦/光照/"
                              f"结构锁定，再谈闭环", gate_pass=False)
    if report["found_rate"] < 0.9:
        # 检出率不达标 = 闭环里近 1/3 轮会走丢失/饱和分支，σ 再好也不能放行
        report["verdict"] += (f"；⚠️ 横杆检出率仅 {report['found_rate']:.0%}"
                              f"（<90%，感知侧需排查）")
        report["gate_pass"] = False
    return report


def main(argv=None, builder=None):
    """builder 可注入假观测链（单测用）；默认装配真机相机+RKNN+横杆"""
    ap = argparse.ArgumentParser(description="S4 硬门禁：定点单帧重复性实测")
    ap.add_argument("--config", default=None)
    ap.add_argument("--frames", type=int, default=30, help="连拍帧数（门禁要求 ≥20）")
    ap.add_argument("--interval-ms", type=int, dest="interval_ms", default=200,
                    help="两帧间隔（毫秒），0=连拍")
    ap.add_argument("--out", default=None, help="报告输出目录（默认 runs/repeatability/<时间戳>）")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="配置覆盖，如 --set camera.light.enabled=true")
    args = ap.parse_args(argv)

    cfg = apply_set(load(args.config), args.overrides)
    try:
        if builder is not None:
            observe, cleanup = builder(cfg)
        else:
            observe, cleanup = build_observer(cfg)
    except Exception as e:
        print(f"[装配失败] {type(e).__name__}: {e}")
        return 2

    print(f"定点实测：连拍 {args.frames} 帧（间隔 {args.interval_ms}ms），"
          f"期间机器人必须保持不动")
    frames = []
    try:
        for i in range(args.frames):
            obs = observe()
            frames.append({"i": i, "bar_frac": obs.bar_frac, "mode": obs.mode,
                           "total_ms": obs.debug.get("total_ms"),
                           "spine_n": (obs.spine.n_books if obs.spine else 0)})
            f = "n/a" if obs.bar_frac is None else f"{obs.bar_frac:.4f}"
            print(f"帧{i:>3} {obs.mode:<10} bar_frac={f}")
            if i < args.frames - 1 and args.interval_ms > 0:
                time.sleep(args.interval_ms / 1000.0)
    finally:
        cleanup()

    report = summarize(frames, args.frames)
    report["frames"] = frames
    report["gate"] = {"repeatability": GATE_REPEATABILITY, "caution": GATE_CAUTION}

    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "runs", "repeatability", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("-" * 72)
    print(f"有效帧 {report['n_valid']}/{report['n_requested']}"
          f"（检出率 {report['found_rate']:.0%}）")
    if report.get("std") is not None:
        print(f"bar_frac：均值 {report['mean']}  σ {report['std']}  "
              f"P5 {report['p05']}  P50 {report['p50']}  P95 {report['p95']}")
    print(f"判定：{report['verdict']}")
    print(f"报告已写入 {out_path}")
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    # Windows 控制台默认 GBK，中文输出/报错会乱码
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
