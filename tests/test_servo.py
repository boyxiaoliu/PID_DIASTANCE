"""闭环状态机 + 饱和标度规则 单元测试

用假的 observe/move 回调，不依赖任何数据，也不碰真机。

    python -m pytest tests/test_servo.py
    python tests/test_servo.py
"""

from pid_servo.observation import BarEdge, Observation, SpineBottom
from pid_servo.servo import (DistanceServo, PID, STATE_LOST, STATE_NORMAL,
                             STATE_SATURATED, classify_observation)
from pid_servo.shelf_sim import ShelfSim, run_sim

TARGET = 0.955
IMG_H = 2160


# ========== 辅助 ==========

def _obs(bar_frac=None, spine_frac=None):
    """构造指定黄蓝位置的观测（比例 → 像素），None = 不检出"""
    return Observation(
        frame_size=(IMG_H, 3840),
        spine=None if spine_frac is None else SpineBottom(
            y=spine_frac * IMG_H, confidence=0.9, n_books=12),
        bar=None if bar_frac is None else BarEdge(
            y=bar_frac * IMG_H, contrast=20.0, y_int=int(bar_frac * IMG_H)),
        source="fake",
    )


def _servo(**kw):
    """按基准参数构造控制器，可被关键字覆盖。"""
    kw.setdefault("pid", {"kp": 3.7, "ki": 0.3, "kd": 0.2})
    kw.setdefault("target_frac", TARGET)
    kw.setdefault("tol", 0.03)
    return DistanceServo(**kw)


# ========== 标度表（classify_observation 三态） ==========

def test_scale_normal():
    """黄蓝间距充足 → 正常态，读数=实测 bar_frac。"""
    state, r = classify_observation(_obs(bar_frac=0.956, spine_frac=0.898))
    assert state == STATE_NORMAL
    assert abs(r - 0.956) < 1e-9


def test_scale_saturated_by_gap():
    """黄蓝重合（间距 < 0.015）→ 饱和态，读数记 1.0。"""
    state, r = classify_observation(_obs(bar_frac=0.912, spine_frac=0.910))
    assert state == STATE_SATURATED
    assert r == 1.0


def test_scale_saturated_by_spine95():
    """蓝线 ≥0.95 且黄线未检出 → 饱和态（书底贴出画下沿 = 太近）。"""
    state, r = classify_observation(_obs(bar_frac=None, spine_frac=0.96))
    assert state == STATE_SATURATED
    assert r == 1.0


def test_scale_lost():
    """黄蓝全无 → 丢失态。"""
    state, r = classify_observation(_obs(bar_frac=None, spine_frac=None))
    assert state == STATE_LOST
    assert r is None


def test_scale_bar_only_is_normal():
    """只有黄线（书脊丢失、兜底搜索区）→ 正常态可用。"""
    state, r = classify_observation(_obs(bar_frac=0.93, spine_frac=None))
    assert state == STATE_NORMAL
    assert abs(r - 0.93) < 1e-9


# ========== 收敛判定 ==========

def test_converges_on_constant_observation():
    """观测恒为容差带内时，连续 hold_frames 轮后收敛。"""
    servo = _servo(hold_frames=2)
    res = servo.run(lambda: _obs(bar_frac=TARGET + 0.001), lambda d: None)
    assert res.converged
    assert res.reason == "converged"


def test_does_not_converge_on_noisy_observation():
    """观测在带外抖动时不应误判收敛。"""
    state = {"n": 0}

    def obs():
        state["n"] += 1
        return _obs(bar_frac=TARGET + (0.2 if state["n"] % 2 else -0.2))

    res = _servo(max_iters=8).run(obs, lambda d: None)
    assert not res.converged


# ========== 退化分支 ==========

def test_backoff_when_lost():
    """丢失搁板时应后退，且步长为 backoff_m。"""
    servo = _servo(max_iters=4, backoff_m=0.25, max_backoff=10)
    moves = []
    servo.run(lambda: _obs(), moves.append)
    assert moves and all(m == 0.25 for m in moves)


def test_gives_up_after_max_backoff():
    """连续丢失超过 max_backoff 轮时放弃，并如实上报 lost。"""
    servo = _servo(max_iters=20, max_backoff=2)
    res = servo.run(lambda: _obs(), lambda d: None)
    assert res.reason == "lost"
    assert not res.converged


def test_saturated_backs_up_fixed_step():
    """饱和态按 sat_step_m 固定小步后退、不过 PID（积分不得被饱和读数污染）。"""
    servo = _servo(max_iters=5, sat_step_m=0.10)
    moves = []
    res = servo.run(lambda: _obs(bar_frac=0.912, spine_frac=0.910), moves.append)
    assert moves and all(m == 0.10 for m in moves)
    assert servo.pid.integral == 0.0          # 从未进过 PID
    assert all(it.state == STATE_SATURATED for it in res.trace)


def test_saturated_step_differs_from_lost_step():
    """饱和=看得见的小步退（sat_step_m），丢失=看不见的大步盲退（backoff_m）。"""
    sat_moves, lost_moves = [], []
    _servo(max_iters=3, sat_step_m=0.10, backoff_m=0.20).run(
        lambda: _obs(bar_frac=0.912, spine_frac=0.910), sat_moves.append)
    _servo(max_iters=3, sat_step_m=0.10, backoff_m=0.20).run(
        lambda: _obs(), lost_moves.append)
    assert sat_moves == [0.10] * 3
    assert lost_moves == [0.20] * 3


def test_saturated_does_not_count_toward_lost_giveup():
    """饱和连退不计入丢失放弃计数（标度表只给丢失设了放弃规则）。"""
    servo = _servo(max_iters=20, max_backoff=2, sat_step_m=0.20)
    res = servo.run(lambda: _obs(bar_frac=0.912, spine_frac=0.910), lambda d: None)
    assert res.reason == "max_iters"          # 撑满轮数而非判 lost
    assert all(it.out_m == 0.20 for it in res.trace)


def test_close_range_forward_cap():
    """近距离前进限幅：深档(|err|<0.06)限 0.05，带内远端(|err|≥0.06)限 0.15。"""
    # 深档：读数 0.90，err=−0.055 <0.06 → 限 0.05
    servo = _servo(pid={"kp": 3.7, "ki": 0, "kd": 0})
    moves = []
    servo.run(lambda: _obs(bar_frac=0.90, spine_frac=0.80), moves.append)
    assert moves[0] == -0.05
    # 带内远端：读数 0.86 ≥ 0.845 且 err=−0.095 ≥0.06 → 限 0.15
    servo2 = _servo(pid={"kp": 3.7, "ki": 0, "kd": 0})
    moves2 = []
    servo2.run(lambda: _obs(bar_frac=0.86, spine_frac=0.76), moves2.append)
    assert moves2[0] == -0.15
    # 带外（读数 0.70 < 0.845）不设限，按 PID 打到远档限幅
    servo3 = _servo(pid={"kp": 3.7, "ki": 0, "kd": 0})
    moves3 = []
    servo3.run(lambda: _obs(bar_frac=0.70, spine_frac=0.60), moves3.append)
    assert moves3[0] == -0.50


def test_forward_cap_hysteresis_blocks_noise_blinding():
    """迟滞窗：近 3 帧进过守卫带后，单帧噪声把读数打低也按深档限幅。

    S3 实测事故场景：贴近距离（真读数≈0.98）一帧 −2σ 噪声读到 0.78，
    旧逻辑守卫失明 + |err| 越过远档阈值 → −0.5m 大步跨线。
    """
    seq = [
        _obs(bar_frac=0.999, spine_frac=0.94),   # 贴近，进守卫带
        _obs(bar_frac=0.78, spine_frac=0.68),    # 噪声尖峰：读数掉出带，|err|=0.175
    ]
    servo = _servo(pid={"kp": 3.7, "ki": 0, "kd": 0}, max_iters=2)
    moves = []
    servo.run(lambda: seq.pop(0), moves.append)
    assert moves[0] > 0                          # 第 1 帧：太近，正常后退
    assert moves[1] == -0.05                     # 第 2 帧：迟滞生效，深档限幅而非 −0.50


def test_close_range_cap_never_limits_backward():
    """前进限幅只管前进：太近的正指令不受影响。"""
    servo = _servo(fwd_limit=0.05, close_band=0.11)
    moves = []
    servo.run(lambda: _obs(bar_frac=0.99, spine_frac=0.80), moves.append)
    assert moves[0] > 0                       # 后退方向不经过该限幅


def test_saturated_never_forward_after_integral_windup():
    """红线场景：积分积成负值后突进饱和，指令仍必须 ≥0（不许前进）。

    构造说明：默认 kp=3.7 时远距误差的比例项一步打满限幅，条件积分直接
    拒收（第一道防线，积分根本积不起来）；这里用小 kp 让积分真正积负，
    考验饱和分支固定步这第二道防线。
    """
    seq = [_obs(bar_frac=0.905, spine_frac=0.845)] * 8   # 太远，正常逼近（P 不饱和）
    seq.append(_obs(bar_frac=0.912, spine_frac=0.910))   # 突进饱和
    servo = _servo(pid={"kp": 1.5}, max_iters=9, sat_step_m=0.20)
    moves = []
    servo.run(lambda: seq.pop(0), moves.append)
    assert servo.pid.integral < 0             # 前置条件：积分确实积成了负值
    assert moves[-1] == 0.20                  # 饱和轮固定步后退，方向为正


# ========== 接线 ==========

def test_converges_on_shelf_sim():
    """接上合成书架后端应能收敛到 d* 附近（验证控制器接线）。"""
    servo = _servo(max_iters=20)
    res, sim = run_sim(servo, d_star=1.5, d0=2.8, noise_std=0.01, seed=1)
    assert res.converged
    assert abs(sim.distance - 1.5) < 0.15


def test_shelf_sim_too_close_retreats_only():
    """从太近的起点出发：先饱和/丢失后退，全程不许比起点更近。"""
    servo = _servo(max_iters=20)
    res, sim = run_sim(servo, d_star=1.5, d0=0.9, noise_std=0.01, seed=2)
    states = [it.state for it in res.trace]
    assert states[0] in (STATE_SATURATED, STATE_LOST)   # 0.9m 处读数已出量程
    assert all(it.out_m >= 0 for it in res.trace)       # 全程不前进
    assert sim.distance > 1.0                           # 确实退了出去


def test_shelf_sim_same_seed_same_trace():
    """固定种子必须可复现：同种子两次轨迹完全一致。"""
    outs = []
    for _ in range(2):
        servo = _servo(max_iters=20)
        res, sim = run_sim(servo, d_star=1.5, d0=2.0, noise_std=0.0299, seed=7)
        outs.append([(it.state, it.reading, it.out_m) for it in res.trace]
                    + [sim.distance])
    assert outs[0] == outs[1]


def test_shelf_sim_dropout_produces_lost():
    """坏帧（dropout）应表现为丢失态，走固定步后退。"""
    sim = ShelfSim(d_star=1.5, d0=1.5, dropout=1.0, seed=0)
    obs = sim.observe()
    assert obs.bar is None and obs.spine is None
    state, _ = classify_observation(obs)
    assert state == STATE_LOST


def test_pid_is_reset_between_runs():
    """两次 run 之间必须复位，否则上一轮的积分会污染下一轮。

    用小 kp：默认 kp=3.7 时 err=-0.30 的比例项一步打满限幅，条件积分
    拒收、积分恒为 0，制造不出残量（那是另一条测试的职责）。
    """
    servo = _servo(pid={"kp": 1.5})
    servo.run(lambda: _obs(bar_frac=TARGET - 0.30), lambda d: None)   # 制造积分残量
    residual = servo.pid.integral
    assert residual != 0.0

    servo.run(lambda: _obs(bar_frac=TARGET + 0.001), lambda d: None)
    # 第二轮的起始积分应为 0，不会带着上一轮的残量冲
    assert servo.pid.integral == 0.0 or abs(servo.pid.integral) < abs(residual)


def test_from_config_reads_contract_keys():
    """from_config 按新配置键路径取值（target.bar_y_frac / health.gap_min_frac）。"""
    cfg = {
        "control": {"pid": {"kp": 3.3}, "loop": {"max_iters": 15},
                    "saturation": {"spine_frac_sat": 0.94, "sat_reading": 1.0}},
        "target": {"bar_y_frac": 0.955, "bar_y_frac_tol": 0.03},
        "health": {"gap_min_frac": 0.02},
    }
    s = DistanceServo.from_config(cfg)
    assert s.pid.cfg.kp == 3.3
    assert s.max_iters == 15
    assert s.target_frac == 0.955 and s.tol == 0.03
    assert s.gap_min_frac == 0.02 and s.spine_frac_sat == 0.94


def test_pid_output_direction():
    """符号约定烟测：太近（e>0）出正=后退，太远（e<0）出负=前进。"""
    pid = PID({"kp": 3.7, "ki": 0, "kd": 0})
    assert pid.update(+0.10) > 0
    pid.reset()
    assert pid.update(-0.10) < 0


if __name__ == "__main__":
    import sys
    import traceback

    sys.stdout.reconfigure(encoding="utf-8")

    cases = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in cases:
        try:
            fn()
            print(f"√ {fn.__name__}")
        except Exception:
            failed += 1
            print(f"× {fn.__name__}")
            traceback.print_exc()

    print(f"\n{len(cases) - failed}/{len(cases)} 通过")
    raise SystemExit(1 if failed else 0)
