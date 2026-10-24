"""距离闭环 —— PID 控制器 + 闭环状态机 〔线B 维护〕

从 旧PID控距/src/pid_servo/servo.py 移植改造（同组自有代码），三处变化：

    1. 观测契约 span_frac → bar_frac（横杆像素 y 占画面高比例，AGENTS.md §6）
    2. 新增饱和标度规则（S1 实验 README 标度表，S3 设计定稿）：
         正常       黄线与蓝线间距 ≥ gap_min_frac → 实测 bar_frac 走 PID
         饱和-太近  间距 < gap_min_frac，或 蓝线≥spine_frac_sat 且黄线未检出
                    → 读数记 sat_reading(=1.0)，固定步后退（见下）
         丢失       黄线蓝线都没有 → 固定步后退，连退 max_backoff 次放弃
    3. 合成后端从 f=k/d 换成 shelf_sim 的 bar_frac ≈ A + B/d

符号约定（AGENTS.md §6，只此一处定义）::

    被控量 d   相机到书架的拍摄距离（米）
    观测量 r   bar_frac = 横杆下沿 y / 画面高，r ≈ A + B/d，距离越近值越大
    误差量 e = r - target            （比例域，无量纲）
    控制量 u = PID(e)                 单位米，正值 = 后退（远离书架）

    e > 0 → r 偏大 → 太近 → 后退；e < 0 → r 偏小 → 太远 → 前进

饱和态为什么走固定步后退而不进 PID：读数饱和意味着真误差比记下的
(sat_reading - target) 更大，且远距逼近阶段 PID 积分积成负值，突进饱和时
负积分可能把输出翻成前进 —— 这踩了「饱和后绝不前进」的安全红线。
固定步后退方向恒正、实现可证，比给 PID 打方向补丁更可靠。

⚠️ 默认增益是量纲推导初值（推导见 configs/default.yaml control.pid 注释），
以 S3 仿真定稿为准，真机复验。
"""

import math
from collections import deque
from dataclasses import dataclass, field, fields

from .observation import Observation


# ========== 观测标度（饱和规则） ==========

#: classify_observation 的三态（S1 实验 README 标度表的"状态"列）
STATE_NORMAL = "normal"        # 正常：实测 bar_frac，PID 推进
STATE_SATURATED = "saturated"  # 饱和-太近：读数记 sat_reading，只许后退
STATE_LOST = "lost"            # 丢失：固定步后退，连退 max_backoff 次放弃


def classify_observation(obs, gap_min_frac=0.015, spine_frac_sat=0.95,
                         sat_reading=1.0):
    """按饱和标度表把一帧 Observation 折算成 (状态, 读数)

    读数全程单调：远 →0.93 递增 → 完美(target) → 饱和 sat_reading（=该后退）。

    :param obs: 契约 Observation
    :param gap_min_frac: 黄线(横杆下沿)与蓝线(书底)最小正常间距，
        与 configs health.gap_min_frac 同源（DistanceServo.from_config 已接线）
    :param spine_frac_sat: 蓝线≥此值且黄线未检出 → 判"太近"饱和
    :param sat_reading: 饱和态读数（画面最下沿）
    :return: (state, reading)；lost 时 reading 为 None
    """
    bar_frac = obs.bar_frac
    if bar_frac is not None:
        if obs.spine is not None and obs.img_h > 0:
            gap = bar_frac - obs.spine.y / obs.img_h
            if gap < gap_min_frac:
                # 黄蓝重合 = 下沿出画、检测器锁到了上沿（G1 目检发现的失败模式）
                return STATE_SATURATED, sat_reading
        # 无蓝线但有黄线：bar_only 兜底态，读数可用（契约 usable 同判）
        return STATE_NORMAL, bar_frac

    # 黄线未检出
    if (obs.spine is not None and obs.img_h > 0
            and obs.spine.y / obs.img_h >= spine_frac_sat):
        # 书底贴着画面下沿 = 太近，横杆被书挡住/出画
        return STATE_SATURATED, sat_reading
    return STATE_LOST, None


# ========== PID 控制器 ==========

@dataclass
class PIDConfig:
    """
    PIDConfig PID 参数（默认值与 configs/default.yaml control.pid 保持一致）

    :param kp: 米 / 比例误差。量纲推导：∂bar_frac/∂d = -B/d²，d*=1.5、B≈0.68
        → 灵敏度 ≈ 0.30/m → kp ≈ 1/0.30 ≈ 3.3，取 3.70 待仿真调优 —— TODO(S3)
    :param ki: 米 / (比例·秒)
    :param kd: 米·秒 / 比例
    :param out_limit: 近距单次输出限幅（米），防一步冲过头
    :param i_limit: 积分项限幅（米），抗积分饱和
    :param deadband: 误差死区（比例），约合 4K 下 8.6px
    :param d_alpha: 微分低通系数（0~1），越小越平滑
    :param dt_default: 未提供 dt 时的默认采样周期（秒）
    :param out_limit_far: |err| 较大时的限幅（米），增益调度：远了走大步近了走小步
    :param far_threshold: 增益调度切换阈值
    """
    kp: float = 3.70
    ki: float = 0.30
    kd: float = 0.20
    out_limit: float = 0.20
    i_limit: float = 0.15
    deadband: float = 0.004
    d_alpha: float = 0.20
    dt_default: float = 0.5
    out_limit_far: float = 0.50
    far_threshold: float = 0.15

    @classmethod
    def from_dict(cls, d):
        """
        from_dict 从配置字典构造，忽略多余键（便于配置文件比代码先演进）

        :param d: 参数字典，可为 None
        :return: PIDConfig 实例
        """
        d = d or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class PID:
    """带死区、微分低通与条件积分抗饱和的 PID（自旧项目原样移植）"""

    def __init__(self, cfg=None):
        """
        :param cfg: PIDConfig 实例、参数字典或 None
        """
        self.cfg = cfg if isinstance(cfg, PIDConfig) else PIDConfig.from_dict(cfg)
        self.reset()

    # ---- 状态 ----

    def reset(self):
        """
        reset 复位积分与微分历史

        每次开始新一个 POI 的闭环前必须调用，否则上一个点的积分残留会
        变成一次莫名其妙的冲刺（DistanceServo.run 已自动调）。
        """
        self._i = 0.0
        self._prev_err = None
        self._d_filt = 0.0
        self._last_out = 0.0

    @property
    def integral(self):
        return self._i

    @property
    def last_output(self):
        return self._last_out

    # ---- 主计算 ----

    def update(self, err, dt=None):
        """
        update 输入误差，返回位移指令

        :param err: bar_frac - target，比例域无量纲
        :param dt: 距上次调用的时间间隔（秒），缺省用配置里的 dt_default。
            离线回放时底盘没有真实时间流逝，用固定 dt 保证可复现
        :return: 位移指令（米，正 = 后退）
        """
        c = self.cfg
        dt = dt if (dt is not None and dt > 0) else c.dt_default

        # 死区（扣除式，保证边界连续）
        if abs(err) <= c.deadband:
            err_eff = 0.0
        else:
            err_eff = err - math.copysign(c.deadband, err)

        # 比例项
        p = c.kp * err_eff

        # 微分项（一阶低通）
        if self._prev_err is None:
            d_raw = 0.0
        else:
            d_raw = (err_eff - self._prev_err) / dt
        self._d_filt = c.d_alpha * d_raw + (1.0 - c.d_alpha) * self._d_filt
        d = c.kd * self._d_filt

        # 积分项（先算候选，再看是否接受）
        i_cand = self._i + c.ki * err_eff * dt
        i_cand = max(-c.i_limit, min(c.i_limit, i_cand))

        # 增益调度：按误差大小选限幅档位
        lim = c.out_limit_far if abs(err_eff) >= c.far_threshold else c.out_limit

        out_unsat = p + i_cand + d
        out = max(-lim, min(lim, out_unsat))

        # 条件积分：未饱和就接受；已饱和时只接受「有利于退饱和」的方向
        if (out == out_unsat
                or (out_unsat > lim and err_eff < 0)
                or (out_unsat < -lim and err_eff > 0)):
            self._i = i_cand

        self._prev_err = err_eff
        self._last_out = out
        return out

    def in_deadband(self, err):
        """
        in_deadband 误差是否落在死区内（即「已经够好了」）

        :param err: 比例域误差
        :return: 是否在死区内
        """
        return abs(err) <= self.cfg.deadband


# ========== 闭环状态机 ==========

@dataclass
class Iteration:
    """
    Iteration 单轮闭环的留痕，实验记录直接序列化这个列表

    :param i: 轮次
    :param mode: 契约观测模式 yolo_bar / bar_only / none
    :param state: 标度状态 normal / saturated / lost（S3 新增）
    :param reading: 本轮标度后读数（饱和态 = sat_reading，丢失 = None）
    :param err: 本轮误差（比例域），丢失/饱和轮为 None
    :param out_m: 本轮下发的位移（米，正 = 后退）
    :param held: 累计连续命中容差带的轮数
    :param distance: 真实距离，仅仿真后端能提供（取自 obs.debug），用于对照
    :param note: 留痕标注。空 = 正常闭环轮；"verify" = 收敛后的终验帧
    """
    i: int
    mode: str
    state: str
    reading: float
    err: float
    out_m: float
    held: int
    distance: float = None
    note: str = ""


@dataclass
class ServoResult:
    """
    ServoResult 一次闭环的最终结果

    :param converged: 是否收敛
    :param reason: converged / max_iters / lost
    :param iters: 实际迭代轮数
    :param reading_final: 末轮标度后读数
    :param err_final: 末轮误差
    :param trace: 逐轮明细
    """
    converged: bool
    reason: str
    iters: int
    reading_final: float
    err_final: float
    trace: list = field(default_factory=list)

    def summary(self):
        """
        summary 生成一行中文摘要，供实验记录直接打印

        :return: 摘要字符串
        """
        e = "n/a" if self.err_final is None else f"{self.err_final:+.4f}"
        r = "n/a" if self.reading_final is None else f"{self.reading_final:.4f}"
        return (f"{'收敛' if self.converged else '未收敛'}"
                f"（{self.reason}）  迭代 {self.iters} 轮  "
                f"bar_frac={r}  误差={e}")


class DistanceServo:
    """
    DistanceServo 单自由度距离闭环

    感知与执行都是注入的可调用对象，不是硬编码依赖：同一份控制逻辑能跑在
    数值仿真、图像回放、真机三种后端上 —— 这是「先离线调好再上车」能成立
    的前提。控制参数在任何一种后端下调好，换后端不用改一行控制代码。
    """

    def __init__(self, pid=None, target_frac=0.955, tol=0.030, *,
                 max_iters=12, hold_frames=3, backoff_m=0.20,
                 max_backoff=3, dt=0.5,
                 gap_min_frac=0.015, spine_frac_sat=0.95, sat_reading=1.0,
                 sat_step_m=0.10, fwd_limit=0.05, close_band=0.11,
                 fwd_limit_far_zone=0.15, fwd_cap_err=0.06):
        """
        :param pid: PID 实例、参数字典或 None
        :param target_frac: 目标 bar_frac（configs target.bar_y_frac）
        :param tol: 容差带半宽（比例）
        :param max_iters: 单点最大迭代轮数，超限即放弃并如实上报
        :param hold_frames: 连续多少轮落在容差带内才判收敛
        :param backoff_m: 丢失（黄蓝全无）时每次固定后退步长（米）
        :param max_backoff: 连续丢失多少轮后放弃（饱和态不计入，见 run）
        :param dt: 传给 PID 的采样周期（秒）。离线仿真用固定值保证可复现
        :param gap_min_frac: 黄蓝最小正常间距（与 configs health.gap_min_frac 同源）
        :param spine_frac_sat: 蓝线≥此值且黄线未检出 → 判饱和
        :param sat_reading: 饱和态读数
        :param sat_step_m: 饱和态固定后退步长（米）。饱和=看得见但太近，小步
            退到带内即收敛；丢失=什么都看不见，才用 backoff_m 大步盲退
        :param fwd_limit: 近距离前进限幅（米）：读数进入 target−close_band 及以上
            时，前进（负）指令至多此值。安全装置，推导见 configs 注释与实验 README
        :param close_band: 前进限幅的 engages 读数带宽（比例域，见 fwd_limit）
        :param fwd_limit_far_zone: 限幅带内远端放宽档（米）：|err| ≥ fwd_cap_err 时
            前进至多此值 —— 防限幅带在远 d* 档伸得太深、爬行耗光迭代预算
        :param fwd_cap_err: 限幅带内两档切换的误差阈值（比例域）
        """
        self.pid = pid if isinstance(pid, PID) else PID(pid)
        self.target_frac = float(target_frac)
        self.tol = float(tol)
        self.max_iters = int(max_iters)
        self.hold_frames = int(hold_frames)
        self.backoff_m = float(backoff_m)
        self.max_backoff = int(max_backoff)
        self.dt = float(dt)
        self.gap_min_frac = float(gap_min_frac)
        self.spine_frac_sat = float(spine_frac_sat)
        self.sat_reading = float(sat_reading)
        self.sat_step_m = float(sat_step_m)
        self.fwd_limit = float(fwd_limit)
        self.close_band = float(close_band)
        self.fwd_limit_far_zone = float(fwd_limit_far_zone)
        self.fwd_cap_err = float(fwd_cap_err)

    # ---- 构造 ----

    @classmethod
    def from_config(cls, cfg):
        """
        from_config 按配置字典构造

        :param cfg: 完整配置字典，键路径见 configs/default.yaml
        :return: DistanceServo 实例
        """
        from .utils import get

        return cls(
            pid=PIDConfig.from_dict(get(cfg, "control.pid", {})),
            target_frac=get(cfg, "target.bar_y_frac", 0.955),
            tol=get(cfg, "target.bar_y_frac_tol", 0.030),
            max_iters=get(cfg, "control.loop.max_iters", 12),
            hold_frames=get(cfg, "control.loop.hold_frames", 3),
            backoff_m=get(cfg, "control.loop.backoff_m", 0.20),
            max_backoff=get(cfg, "control.loop.max_backoff", 3),
            gap_min_frac=get(cfg, "health.gap_min_frac", 0.015),
            spine_frac_sat=get(cfg, "control.saturation.spine_frac_sat", 0.95),
            sat_reading=get(cfg, "control.saturation.sat_reading", 1.0),
            sat_step_m=get(cfg, "control.loop.sat_step_m", 0.10),
            fwd_limit=get(cfg, "control.loop.fwd_limit", 0.05),
            close_band=get(cfg, "control.loop.close_band", 0.11),
            fwd_limit_far_zone=get(cfg, "control.loop.fwd_limit_far_zone", 0.15),
            fwd_cap_err=get(cfg, "control.loop.fwd_cap_err", 0.06),
        )

    # ---- 主循环 ----

    def run(self, observe, move, on_iter=None):
        """
        run 执行一次闭环，直到收敛 / 超限 / 丢失

        退化处理只后退、不前进：拍摄构图是「这一格撑满画面」，靠太近时横杆
        下沿出画/黄蓝重合（饱和态）、或黄蓝全丢（丢失态），离太远时结构缩小
        到画面中央反而不会丢 —— 所以退化几乎只指向「太近」一种解释，固定步
        后退到重新看见为止。两态的方向都恒为后退，绝不朝书架撞过去。

        收敛判据用 3 点中值滤波后再看是否连续命中，不是直接判单次误差，
        理由见旧项目 docs/02_闭环设计论证.md（抗单帧噪声误判）。

        :param observe: 无参回调，采集一帧并返回 Observation。由调用方负责
            让图像新鲜（真机上通常要先丢弃几帧等曝光稳定）
        :param move: 接受一个 float（米，正 = 后退）的回调。由调用方负责等待
            动作完成（真机要等底盘停稳 + 相机重新稳定，这是耗时的大头）
        :param on_iter: 每轮结束的回调，传 None 则不回调
        :return: ServoResult
        """
        self.pid.reset()

        trace = []
        hold = 0
        backoff_streak = 0
        obs = None
        reason = "max_iters"
        recent = deque(maxlen=3)          # 最近 3 轮正常态误差，用于中值滤波
        reading_window = deque(maxlen=3)  # 最近 3 轮读数，前进限幅的迟滞窗（防单帧噪声致盲）

        for i in range(self.max_iters):
            obs = observe()
            state, reading = classify_observation(
                obs, self.gap_min_frac, self.spine_frac_sat, self.sat_reading)

            err = None
            out = 0.0
            stop = None

            if state == STATE_NORMAL:
                # 正常：走 PID
                backoff_streak = 0
                err = reading - self.target_frac
                reading_window.append(reading)

                # 判定用中值滤波后的误差，控制仍用实时误差
                recent.append(err)
                err_judge = sorted(recent)[len(recent) // 2] if len(recent) == 3 else err

                hold = hold + 1 if abs(err_judge) <= self.tol else 0
                if hold >= self.hold_frames:
                    stop = "converged"
                else:
                    out = self.pid.update(err, self.dt)
                    # 近距离前进限幅（带 3 帧迟滞）：只要近 3 帧内任一读数进过
                    # 守卫带（target − close_band），就认定"可能已贴近安全线"：
                    #   当前帧读数被噪声打进带内且 |err| 大 → 放宽档 fwd_limit_far_zone
                    #   当前帧读数掉出带（守卫被噪声致盲）→ 最保守深档 fwd_limit
                    # 防的场景：贴近距离一帧 −2σ 噪声尖峰同时骗过读数门槛、又把
                    # |err| 推过远档阈值，诱发 −0.5m 大步一步跨过 d*−0.15m 线
                    # （S3 实验 3× 档实测，深度达 0.5m，见实验 README）
                    recent_max = max(reading_window)
                    if out < 0 and recent_max >= self.target_frac - self.close_band:
                        if reading >= self.target_frac - self.close_band and \
                                abs(err) >= self.fwd_cap_err:
                            cap = self.fwd_limit_far_zone
                        else:
                            cap = self.fwd_limit
                        out = max(out, -cap)
            elif state == STATE_SATURATED:
                # 太近：固定小步后退，不过 PID（原因见模块 docstring），
                # 也不计入丢失连退计数 —— 标度表只给"丢失"设了放弃规则
                hold = 0
                out = self.sat_step_m
            else:
                # 丢失：只后退
                hold = 0
                backoff_streak += 1
                if backoff_streak > self.max_backoff:
                    stop = "lost"
                else:
                    out = self.backoff_m

            # 本轮留痕（统一在这里构造，分支各自构造容易顾此失彼）
            it = Iteration(i, obs.mode, state, reading, err, out, hold,
                           distance=obs.debug.get("d_true"))
            trace.append(it)
            if on_iter:
                on_iter(it)

            if stop:
                reason = stop
                break
            move(out)

        final_reading = (trace[-1].reading if trace else None)
        return ServoResult(
            converged=(reason == "converged"),
            reason=reason,
            iters=len(trace),
            reading_final=final_reading,
            err_final=(None if final_reading is None
                       else final_reading - self.target_frac),
            trace=trace,
        )

    # ---- 收敛终验（S5 新增） ----

    def run_until_confirmed(self, observe, move, on_iter=None,
                            max_confirm_attempts=3):
        """run_until_confirmed 跑闭环直到「收敛 且 终验帧单帧直判在带内」

        循环内的收敛判据是 3 点中值滤波，有已知盲区：三帧误差取中值时，
        最新一帧可能已出带而中值仍在带内（项目记忆 S5 注意点）。所以每次
        收敛宣告后都要再取一帧新鲜数据做单帧终验：在带内才算真到位；
        出带（含饱和/丢失态）就回到闭环再调，终验最多 max_confirm_attempts
        轮，始终不达标则如实报未到位。

        :param observe: 同 run
        :param move: 同 run
        :param on_iter: 同 run；终验帧也走此回调留痕（Iteration.note="verify"）
        :param max_confirm_attempts: 终验尝试次数上限（每次含一轮完整闭环）
        :return: (confirmed, result, verify_obs)
            confirmed   是否「收敛且终验在带内」
            result      末次闭环的 ServoResult（未收敛时即那一次的结果）
            verify_obs  通过终验的那帧 Observation；confirmed=False 时为 None
        """
        result = None
        for _ in range(max_confirm_attempts):
            result = self.run(observe, move, on_iter=on_iter)
            if not result.converged:
                return False, result, None

            obs = observe()
            state, reading = classify_observation(
                obs, self.gap_min_frac, self.spine_frac_sat, self.sat_reading)
            ok = (state == STATE_NORMAL
                  and reading is not None
                  and abs(reading - self.target_frac) <= self.tol)
            if on_iter:
                err = None if reading is None else reading - self.target_frac
                on_iter(Iteration(result.iters, obs.mode, state, reading,
                                  err, 0.0, 0,
                                  distance=obs.debug.get("d_true"),
                                  note="verify"))
            if ok:
                return True, result, obs
        return False, result, None
