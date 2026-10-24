"""S5 真机单点闭环 CLI —— "量→判→走→验→拍"全流程入口

用法（项目根目录）：

    python tools/run_servo_real.py --dry-run                       # 合成书架干跑（PC 可跑）
    python tools/run_servo_real.py --dry-run --d0 0.9 --d-star 1.5
    python tools/run_servo_real.py                                 # 真机模式（需板端硬件）

流程与"确认"语义（S5 新增后三环）：
    1) 闭环循环：每轮 拍照→跑模型→分析→运动，连续 3 轮中值在带内即跳出
       （max_iters=40 只是防打转的保险上限，不是固定次数）
    2) 终验：收敛后重新取帧单帧直判（堵 3 点中值的盲区：最新帧可能已出带
       而中值仍在带内）；出带就回到闭环再调，最多 control.loop.confirm_attempts 次
    3) 确认照：到位后按学长 capture_group 标准重开 4K 流拍一张，过质检
       （清晰度/亮度）存档 —— 闭环的验收证据
    4) 交接学长系统（可选，control.handoff.enabled / CLI --handoff）：把对齐后
       位姿写回底盘 POI（学长 ia.move 从底盘 POI 库取坐标，改完即原地瞬达），
       再向其 Flask API 派"指定列盘点"任务。交接失败不改退出码，但大声告警并落账。
       ⚠️ 现场交接前先重启学长系统（其 POI 坐标有进程内缓存，见 configs 注释）

干跑与真机的唯一区别是 observe/move/终拍 三个回调的来源（注入式约定）：
    --dry-run  shelf_sim 合成书架 + 合成终拍帧（链路与真机同构，PC 可复现）
    真机       robot/backend.RobotBackend + CameraBackend.capture_still(4K)

安全：真机模式 Ctrl-C 先发急停再退出；相机/补光灯在 finally 里关闭；
装配阶段任何失败（相机打不开、模型缺失）一律安全退出，绝不带病运行。

产出（默认 runs/servo_real/<时间戳>/，不入库；实验引用时拷贝到 experiments/）：
    trace.jsonl      每轮一行 {"type":"iter",...}（含分阶段耗时，终验帧 note="verify"），
                     收尾 {"type":"result",...}（含 confirmed）与 {"type":"final_shot",...}
    perfect_shot.jpg 到位后的确认照（control.final_shot.enabled=false 时不拍）
    final_meta.json  确认照元数据（终验读数/误差、质检指标、位姿、交接结果等）

退出码：0=到位且确认照合格  1=未到位（未收敛或终验出带，不拍确认照）  2=装配失败
        3=到位但确认照质检未通过（照片/指标已存档待人工看）  130=中断
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
from pid_servo.utils import get, imwrite_utf8, load
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


def run_handoff(cfg, client, pose):
    """run_handoff 执行交接：改 POI（可选）+ 派盘点任务

    :param cfg: 完整配置字典（读 control.handoff 节）
    :param client: HandoffClient 注入（单测假件）；None 按配置现建
    :param pose: 对齐确认后的位姿 {"x","y","yaw"}；None 时改 POI 步骤跳过并如实标注
    :return: 留痕字典（poi_updated / task_id / task_type / poi_key）
    :raises HandoffError: 任一步被拒
    """
    from pid_servo.robot.handoff import HandoffClient
    from pid_servo.robot.mover import ChassisHttp

    h = get(cfg, "control.handoff", {}) or {}
    poi_key = h.get("poi_key", "")
    task_type = int(h.get("task_type", 1))
    params = {k: h.get(k) for k in ("floor", "area", "rack", "side", "column")}
    if client is None:
        timeout = float(h.get("timeout_s", 10.0))
        client = HandoffClient(
            chassis=ChassisHttp(get(cfg, "control.mover.chassis_url", ""), timeout),
            senior=ChassisHttp(h.get("api_url", "http://127.0.0.1:5000"), timeout))
    summary = client.handoff(poi_key, pose, task_type, params,
                             update_poi=bool(h.get("update_poi", True)))
    summary.update(task_type=task_type, poi_key=poi_key)
    return summary


def synthetic_still(cfg):
    """synthetic_still 干跑终拍：合成一张带纹理的帧，走与真机同一条质检+存盘链路

    PC 上没有相机，但确认照的质检门与存盘路径必须每次干跑都被真实执行，
    不能在干跑里跳过 —— 否则"链路与真机同构"就是空话。
    """
    import cv2
    import numpy as np

    from pid_servo.robot.camera import check_frame_quality

    w = int(get(cfg, "camera.still_width", 3840))
    h = int(get(cfg, "camera.still_height", 2160))
    q = get(cfg, "camera.quality", {}) or {}
    frame = np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)
    cv2.putText(frame, "DRY-RUN FINAL SHOT", (60, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 4.0, (255, 255, 255), 12)
    ok, metrics = check_frame_quality(frame, float(q.get("blur_min_lap", 100.0)),
                                      float(q.get("brightness_min", 40.0)),
                                      float(q.get("brightness_max", 220.0)))
    return {"frame": frame, "ok": ok, "metrics": metrics, "attempts": 1}


def main(argv=None, builder=None, still_fn=None, handoff_client=None):
    """main CLI 入口

    :param argv: 命令行参数（None 用 sys.argv）
    :param builder: 装配函数注入（单测假件），默认 build_backend
    :param still_fn: 终拍函数注入（单测假件），返回 {"frame","ok","metrics","attempts"}；
        None 时按模式自动选择（真机 = 相机 capture_still，干跑 = 合成帧）
    :param handoff_client: 交接客户端注入（单测假件）；None 按 control.handoff 现建
    """
    ap = argparse.ArgumentParser(description="S5 真机单点距离闭环（量→判→走→验→拍→交接）")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="用合成书架代替真车真相机（PC 可跑，链路与真机同构）")
    ap.add_argument("--handoff", action="store_true",
                    help="到位后交接学长系统（临时强制开启，等价 control.handoff.enabled=true）")
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
    confirm_attempts = int(get(cfg, "control.loop.confirm_attempts", 3))
    shot_enabled = bool(get(cfg, "control.final_shot.enabled", True))
    still_w = int(get(cfg, "camera.still_width", 3840))
    still_h = int(get(cfg, "camera.still_height", 2160))
    quality_cfg = get(cfg, "camera.quality", {}) or {}

    try:
        observe, move, source, extra, cleanup = (builder or build_backend)(cfg, servo, args)
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
          f"max_iters={servo.max_iters}  终验≤{confirm_attempts}次  "
          f"确认照{'开' if shot_enabled else '关'}  trace → {trace_path}")

    trace = open(trace_path, "w", encoding="utf-8")
    handoff_on = bool(get(cfg, "control.handoff.enabled", False)) or args.handoff
    meta = {"type": "meta", "ts": datetime.now().isoformat(timespec="seconds"),
            "argv": argv if argv is not None else sys.argv[1:], "source": source,
            "target": servo.target_frac, "tol": servo.tol,
            "confirm_attempts": confirm_attempts, "final_shot": shot_enabled,
            "handoff": handoff_on, **extra}
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
        print(f"轮{it.i:>3} {it.state:<10} {r} {e} 指令{it.out_m:>+7.3f} "
              f"hold={it.held}" + ("  [终验]" if it.note == "verify" else ""))

    print("-" * 72)
    try:
        confirmed, result, verify_obs = servo.run_until_confirmed(
            observing, move, on_iter=on_iter, max_confirm_attempts=confirm_attempts)
    except KeyboardInterrupt:
        # 真机上 Ctrl-C 必须先停车再退出；干跑无运动件，直接落账
        try:
            stop = getattr(getattr(move, "__self__", None), "stop", None)
            if stop is not None:
                stop()
                print("已发急停（Ctrl-C）")
        finally:
            trace.write(json.dumps({"type": "result", "confirmed": False,
                                    "reason": "interrupted"}, ensure_ascii=False) + "\n")
            trace.close()
            cleanup()
        return 130

    tail = {"type": "result", "confirmed": confirmed,
            "converged": result.converged, "reason": result.reason,
            "iters": result.iters,
            "reading_final": result.reading_final, "err_final": result.err_final}
    if hasattr(move, "__self__") and hasattr(move.__self__, "distance"):
        tail["final_distance"] = move.__self__.distance      # 干跑后端才报真距离
    trace.write(json.dumps(tail, ensure_ascii=False) + "\n")

    print("-" * 72)
    print(result.summary())

    if not confirmed:
        trace.close()
        cleanup()
        print("终验未到位（收敛后新鲜帧出带/丢失），不拍确认照")
        return 1

    # ---- 确认照（4K，学长 capture_group 标准）与交接学长系统 ----
    backend_obj = getattr(move, "__self__", None)   # RobotBackend / ShelfSim / None(假件)

    # 位姿一次取够：确认照留痕与交接改 POI 都用它（底盘此时已停稳；
    # 拿不到不拦路——假件/网络抖动时留痕记 None，交接按 update_poi 跳过处理）
    pose = None
    mover_obj = getattr(backend_obj, "mover", None)
    if mover_obj is not None and hasattr(mover_obj, "get_pose"):
        try:
            pose = mover_obj.get_pose()
        except Exception:
            pose = None

    shot, shot_err = None, None
    if shot_enabled:
        if still_fn is None:
            cam_obj = getattr(backend_obj, "camera", None)
            if cam_obj is not None and hasattr(cam_obj, "capture_still"):
                still_fn = (lambda: cam_obj.capture_still(still_w, still_h, quality_cfg))
            else:
                still_fn = (lambda: synthetic_still(cfg))

        try:
            shot = still_fn()
        except KeyboardInterrupt:
            trace.write(json.dumps({"type": "result", "confirmed": False,
                                    "reason": "interrupted"}, ensure_ascii=False) + "\n")
            trace.close()
            cleanup()
            return 130
        except Exception as e:               # 4K 流打不开等：位置已到位，如实记账不拦报
            shot_err = f"{type(e).__name__}: {e}"

    shot_ok = (not shot_enabled) or (shot is not None and shot["ok"] and not shot_err)

    # 交接只在干净路径上执行（确认照关闭/合格）；质检未过先留人看，不自动派活
    handoff_summary, handoff_err = None, None
    if (bool(get(cfg, "control.handoff.enabled", False)) or args.handoff) and shot_ok:
        try:
            handoff_summary = run_handoff(cfg, handoff_client, pose)
            print(f"已交接学长系统：task_id={handoff_summary['task_id']}  "
                  f"POI={handoff_summary['poi_key']}  "
                  f"改坐标={handoff_summary['poi_updated']}")
        except KeyboardInterrupt:
            trace.write(json.dumps({"type": "result", "confirmed": False,
                                    "reason": "interrupted"}, ensure_ascii=False) + "\n")
            trace.close()
            cleanup()
            return 130
        except Exception as e:
            handoff_err = f"{type(e).__name__}: {e}"
            print(f"⚠️ 交接学长系统失败（不影响到位结论，已落账）：{handoff_err}")

    # ---- 落盘与收尾 ----
    shot_path = None
    if shot is not None and shot.get("frame") is not None:
        shot_path = os.path.join(out_dir, "perfect_shot.jpg")
        imwrite_utf8(shot_path, shot["frame"])

    final_meta = {
        "type": "final_shot",
        "ts": datetime.now().isoformat(timespec="seconds"),
        "size": [still_w, still_h],
        "verify": (None if verify_obs is None else {
            "bar_frac": verify_obs.bar_frac,
            "err": (None if verify_obs.bar_frac is None
                    else verify_obs.bar_frac - servo.target_frac),
            "mode": verify_obs.mode,
        }),
        "shot": (None if shot is None else {
            "ok": shot["ok"], "metrics": shot["metrics"],
            "attempts": shot["attempts"]}),
        "capture_error": shot_err,
        "pose": pose,
        "handoff": handoff_summary,
        "handoff_error": handoff_err,
    }
    meta_path = os.path.join(out_dir, "final_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(final_meta, f, ensure_ascii=False, indent=2)
    trace.write(json.dumps({"type": "final_shot",
                            "ok": bool(shot and shot["ok"]),
                            "capture_error": shot_err,
                            "handoff": handoff_summary,
                            "handoff_error": handoff_err},
                           ensure_ascii=False) + "\n")

    if shot_enabled and not shot_ok:
        print(f"确认照质检未通过（{shot_err or '清晰度/亮度不达标'}），"
              f"照片与指标已存档待人工看：{meta_path}")
        trace.close()
        cleanup()
        return 3

    print(f"已确认到位，确认照 → {shot_path}"
          if shot_enabled else
          "已确认到位（确认照已关闭：control.final_shot.enabled=false）")

    trace.close()
    cleanup()
    return 0


if __name__ == "__main__":
    # Windows 控制台默认 GBK，中文输出/报错会乱码
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
