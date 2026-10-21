"""S4 真机单点闭环 CLI —— 一到点位后开始的"量→判→走"循环入口

用法（项目根目录）：

    python tools/run_servo_real.py --dry-run                       # 合成书架干跑（PC 可跑）
    python tools/run_servo_real.py --dry-run --d0 0.9 --d-star 1.5
    python tools/run_servo_real.py                                 # 真机模式（需板端硬件）

干跑与真机的唯一区别是 observe/move 两个回调的来源（注入式约定）：
    --dry-run  shelf_sim 合成书架（读数对移动有物理响应，链路完整可复现）
    真机       robot/backend.RobotBackend（相机帧 + RKNN 书脊 + 横杆检测 + 底盘 Mover）

安全：真机模式 Ctrl-C 先发急停再退出；相机/补光灯在 finally 里关闭；
装配阶段任何失败（相机打不开、模型缺失）一律安全退出，绝不带病运行。

产出（默认 runs/servo_real/<时间戳>/trace.jsonl，不入库；实验引用时拷贝到 experiments/）：
    每轮一行 {"type":"iter",...}（含分阶段耗时），收尾一行 {"type":"result",...}

退出码：0=收敛  1=未收敛/放弃  2=装配失败  130=中断
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # 复用 sim_servo 的装配函数

from pid_servo.servo import DistanceServo
from pid_servo.utils import get, load
from sim_servo import apply_set, sim_kwargs_from_cfg

#: 逐轮耗时 debug 键（robot/backend.py 写入；干跑没有相机段，记 None）
TIMING_KEYS = ("capture_ms", "infer_ms", "bar_ms", "total_ms")


def build_backend(cfg, servo, args):
    """按模式装配 observe/move 回调

    :return: (observe, move, source 描述, 额外元数据 dict, cleanup 可调用)
    cleanup 幂等，干跑为空操作；真机关流 + 关补光灯。
    """
    noise_std = float(get(cfg, "sim.noise_std", 0.0299)) * args.noise_mult

    if args.dry_run:
        from pid_servo.shelf_sim import ShelfSim

        sim = ShelfSim(d_star=args.d_star, d0=args.d0, target=servo.target_frac,
                       noise_std=noise_std, seed=args.seed,
                       **sim_kwargs_from_cfg(cfg))
        source = (f"dry-run:shelf_sim(d*={args.d_star}, d0={args.d0}, "
                  f"noise={args.noise_mult}x)")
        return sim.observe, sim.move, source, {"d_star": args.d_star, "d0": args.d0}, \
            (lambda: None)

    # ---- 真机模式：相机 → RKNN 书脊 → 横杆 → 底盘，全部按 configs 装配 ----
    from pid_servo.perception import BarDetector
    from pid_servo.robot.backend import RobotBackend
    from pid_servo.robot.camera import CameraBackend
    from pid_servo.robot.mover import Mover
    from pid_servo.robot.spine_detector import RknnSpineDetector

    camera = CameraBackend.from_config(cfg)
    try:
        camera.start()                       # 打不开立刻 CameraError（安全退出）
        spine = RknnSpineDetector.from_config(cfg)   # 模型常驻加载
        mover = Mover.from_config(cfg)
        bar = BarDetector.from_config(args.config)
        backend = RobotBackend(camera, spine, mover, bar, source="robot")
    except Exception:
        camera.close()                       # 装配中途失败也要关灯关流
        raise

    source = (f"robot(cam={get(cfg, 'camera.cam_id', '4')}, "
              f"{get(cfg, 'camera.preview_width', 960)}x"
              f"{get(cfg, 'camera.preview_height', 540)}, "
              f"chassis={get(cfg, 'control.mover.chassis_url', '')})")
    extra = {"spine_model": spine.model_path}
    return backend.observe, backend.move, source, extra, camera.close


def main(argv=None):
    ap = argparse.ArgumentParser(description="S4 真机单点距离闭环")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="用合成书架代替真车真相机（PC 可跑，链路与真机同构）")
    ap.add_argument("--d0", type=float, default=0.9, help="干跑初始距离（米）")
    ap.add_argument("--d-star", type=float, dest="d_star", default=1.5,
                    help="干跑真实完美距离（米）")
    ap.add_argument("--noise-mult", type=float, dest="noise_mult", default=1.0,
                    help="干跑读数噪声倍率（σ0 的倍数）")
    ap.add_argument("--seed", type=int, default=0, help="干跑随机种子")
    ap.add_argument("--out", default=None, help="trace 输出目录（默认 runs/servo_real/<时间戳>）")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="配置覆盖，如 --set control.pid.kp=3.3")
    args = ap.parse_args(argv)

    cfg = apply_set(load(args.config), args.overrides)
    servo = DistanceServo.from_config(cfg)

    try:
        observe, move, source, extra, cleanup = build_backend(cfg, servo, args)
    except Exception as e:                   # 装配失败（相机打不开/模型缺失等）安全退出
        print(f"[装配失败] {type(e).__name__}: {e}")
        return 2

    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "runs", "servo_real", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    trace_path = os.path.join(out_dir, "trace.jsonl")

    print(f"{'模式':<8} {source}")
    print(f"target={servo.target_frac} ± {servo.tol}  "
          f"max_iters={servo.max_iters}  trace → {trace_path}")

    trace = open(trace_path, "w", encoding="utf-8")
    meta = {"type": "meta", "ts": datetime.now().isoformat(timespec="seconds"),
            "argv": argv if argv is not None else sys.argv[1:], "source": source,
            "target": servo.target_frac, "tol": servo.tol, **extra}
    trace.write(json.dumps(meta, ensure_ascii=False) + "\n")

    last_timing = {}

    def observing():
        obs = observe()
        last_timing.clear()
        last_timing.update({k: obs.debug.get(k) for k in TIMING_KEYS})
        return obs

    def on_iter(it):
        line = {"type": "iter", **asdict(it), **last_timing}
        trace.write(json.dumps(line, ensure_ascii=False) + "\n")
        r = "n/a" if it.reading is None else f"{it.reading:7.4f}"
        e = "n/a" if it.err is None else f"{it.err:+8.4f}"
        print(f"轮{it.i:>3} {it.state:<10} {r} {e} 指令{it.out_m:>+7.3f} hold={it.held}")

    print("-" * 72)
    try:
        result = servo.run(observing, move, on_iter=on_iter)
    except KeyboardInterrupt:
        # 真机上 Ctrl-C 必须先停车再退出；干跑无运动件，直接落账
        try:
            stop = getattr(getattr(move, "__self__", None), "stop", None)
            if stop is not None:
                stop()
                print("已发急停（Ctrl-C）")
        finally:
            trace.write(json.dumps({"type": "result", "converged": False,
                                    "reason": "interrupted"}, ensure_ascii=False) + "\n")
            trace.close()
            cleanup()
        return 130

    tail = {"type": "result", "converged": result.converged, "reason": result.reason,
            "iters": result.iters,
            "reading_final": result.reading_final, "err_final": result.err_final}
    if hasattr(move, "__self__") and hasattr(move.__self__, "distance"):
        tail["final_distance"] = move.__self__.distance      # 干跑后端才报真距离
    trace.write(json.dumps(tail, ensure_ascii=False) + "\n")
    trace.close()
    cleanup()

    print("-" * 72)
    print(result.summary())
    print(f"trace 已写入 {trace_path}")
    return 0 if result.converged else 1


if __name__ == "__main__":
    # Windows 控制台默认 GBK，中文输出/报错会乱码
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
