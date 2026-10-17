"""合成书架后端 —— 距离 → Observation 的物理模型 〔S3 仿真〕

读数模型（S3 设计定稿）：bar_frac(d) = A + B/d
    A = 0.50：d→∞ 时横杆趋于画面中心的地平线假设
    B = (target − A)·d*：使完美距离 d* 处读数恰为 target（真值未知，网格全跑）

几何细节（让三种观测状态从几何里自然涌现，而不是人为指定）：
    黄蓝间距 gap(d) = C_gap / d，C_gap = gap0·d*（gap0 = S1 干净群中位 0.058）
    蓝线(书底) spine_frac = bar_frac − gap（书在横杆上方，y 更小）
    噪声分解：黄线读数加 noise_std；间距加独立的 gap_noise_std——后者的依据是
        S1 真数据间距呈干净双峰（干净群/重合群在 0.015 处可分），若间距噪声
        与读数同量级，双峰早就糊成一片，gap 规则也就无从谈起。蓝线读数由
        "黄线 − 间距"推导，保证控制器算出的间距恰为 gap_true + 间距噪声。
    太近时按出画顺序退化：
        ① 横杆下沿出画（真读数 > bar_out_frac）→ 横杆带仍部分可见，检测器
           锁到上沿 = "黄蓝重合"态（G1 目检发现的失败模式）→ 控制器 gap 规则判饱和
        ② 书底也出画 → 黄蓝全无 → 控制器丢失分支 → 固定步后退
    另有整帧坏帧率 dropout（真数据 6/247 = 2.4%）

注入式设计（旧项目验证过的约定）：控制器只调 observe()/move()，不知道对面
是合成书架还是真车。仿真跑的代码 = 上车跑的代码。本模块验证的是控制器，
不是感知；感知质量由 153 张真图 + 用户目检闭环（S1）。结论单向有效：
仿真不过 = 必不过；仿真过了 = 有资格上车（S5 真机是最终关卡）。
"""

from dataclasses import dataclass

import numpy as np

from .observation import BarEdge, Observation, SpineBottom


@dataclass
class ShelfSim:
    """把「拍摄距离」映射成 Observation 的合成后端

    :param d_star: 真实完美距离（米）——控制器不知道，读数 target 在这里成立
    :param d0: 初始距离（米）
    :param target: 目标读数（与控制器 target_frac 一致，run_sim 自动对齐）
    :param A: 读数模型 A + B/d 的远端渐近值
    :param gap0_frac: d* 处的黄蓝间距（S1 干净群中位）
    :param noise_std: 黄线读数噪声标准差（比例域），压力测试按倍率加压
    :param gap_noise_std: 黄蓝间距的测量噪声（比例域）。不随压力倍率加压：
        加压对象是控制读数；间距规则的抗噪性由真数据双峰结构背书（见模块 docstring）
    :param dropout: 每次观测的整帧坏帧率（黄蓝全无，真数据 6/247）
    :param move_std: 每次移动的停稳散布 σ（米，假设值）
    :param bar_out_frac: 下沿出画阈值——真读数超过它 = 横杆下沿不可见
        （像素不可能超出画面，结构性阈值）
    :param topedge_offset/std: 黄蓝重合态里检测器锁到横杆上沿时，
        报出的黄线在蓝线之下的偏移（须 < gap_min_frac 才会触发饱和规则）
    :param img_h/img_w: 画面尺寸（4K，与真机 cam_4 一致）
    :param d_floor: 距离下限（米），防数值发散（旧项目同值）
    :param seed: 随机种子。固定种子保证实验可复现
    """
    d_star: float = 1.5
    d0: float = 1.5
    target: float = 0.955
    A: float = 0.50
    gap0_frac: float = 0.058
    noise_std: float = 0.0299
    gap_noise_std: float = 0.005
    dropout: float = 0.024
    move_std: float = 0.02
    bar_out_frac: float = 1.0
    topedge_offset: float = 0.008
    topedge_std: float = 0.003
    img_h: int = 2160
    img_w: int = 3840
    d_floor: float = 0.25
    seed: int = 0

    def __post_init__(self):
        # B 使 d* 处读数恰为 target；C_gap 使 d* 处黄蓝间距恰为 gap0_frac
        self.B = (self.target - self.A) * self.d_star
        self.C_gap = self.gap0_frac * self.d_star
        self.d = float(self.d0)
        self.rng = np.random.default_rng(self.seed)

    # ---- 后端接口（与真机适配层保持一致）----

    @property
    def distance(self):
        return self.d

    def move(self, delta_m):
        """
        move 执行位移，正 = 后退（远离书架），叠加底盘停稳散布

        :param delta_m: 位移量（米）
        """
        step = float(delta_m)
        if self.move_std > 0:
            step += self.rng.normal(0.0, self.move_std)
        self.d = max(self.d_floor, self.d + step)

    def observe(self):
        """
        observe 产生一次观测，不含任何图像处理 —— 直接由几何模型加噪声

        ⚠️ 随机数的抽取顺序被仿真实验复现依赖（改顺序会让历史记录数值对不上）：
            ① 坏帧抽取（无条件）→ ② 黄线噪声（黄线可见才抽）→ ③ 间距噪声
            （黄蓝双可见）/ 蓝线独噪（仅蓝线可见）→ ④ 上沿锁定抖动（重合态才抽）；
            move() 的散布独立抽取。

        :return: Observation（debug 带 d_true / reading_true / gap_true 供对照）
        """
        obs = Observation(frame_size=(self.img_h, self.img_w), source="shelf_sim")
        reading_true = self.A + self.B / self.d
        gap_true = self.C_gap / self.d
        spine_true = reading_true - gap_true
        obs.debug.update(d_true=self.d, reading_true=reading_true,
                         gap_true=gap_true)

        if self.rng.random() < self.dropout:
            return obs                      # 坏帧：黄蓝全无

        bar_visible = reading_true <= self.bar_out_frac
        spine_visible = spine_true <= 1.0

        if bar_visible:
            # 黄线 = 真读数 + 读数噪声
            y = (reading_true + self._noise()) * self.img_h
            obs.bar = BarEdge(y=min(y, self.img_h - 1.0), contrast=20.0,
                              y_int=int(y))
        if spine_visible:
            # 蓝线 = 黄线读数 − 带噪间距 → 控制器算出的间距 = gap_true + 间距噪声
            gap_meas = gap_true
            if self.gap_noise_std > 0:
                gap_meas += self.rng.normal(0.0, self.gap_noise_std)
            if obs.bar is None:
                y_spine = (spine_true + self._noise()) * self.img_h
            else:
                y_spine = obs.bar.y - gap_meas * self.img_h
            obs.spine = SpineBottom(y=min(y_spine, self.img_h - 1.0),
                                    confidence=0.90, n_books=12)
        if not bar_visible and spine_visible:
            # 重合态：下沿出画但横杆带仍在，检测器锁到上沿（G1 目检发现）
            y = (spine_true + self.topedge_offset
                 + self.rng.normal(0.0, self.topedge_std)) * self.img_h
            obs.bar = BarEdge(y=min(y, self.img_h - 1.0), contrast=20.0,
                              y_int=int(y))
        return obs

    def _noise(self):
        return self.rng.normal(0.0, self.noise_std) if self.noise_std > 0 else 0.0


def run_sim(servo, d_star, d0, noise_std, seed=0, sim_kwargs=None, on_iter=None):
    """在合成书架上跑一次闭环（工具与单元测试共用的接线）

    :param servo: DistanceServo 实例（run 内部自动复位 PID，可复用）
    :param d_star: 真实完美距离（米）
    :param d0: 初始距离（米）
    :param noise_std: 观测噪声标准差（比例域）
    :param seed: 随机种子
    :param sim_kwargs: 透传给 ShelfSim 的额外参数
    :param on_iter: 每轮回调 on_iter(it, sim)
    :return: (ServoResult, ShelfSim)
    """
    sim = ShelfSim(d_star=d_star, d0=d0, target=servo.target_frac,
                   noise_std=noise_std, seed=seed, **(sim_kwargs or {}))
    cb = (lambda it: on_iter(it, sim)) if on_iter else None
    res = servo.run(sim.observe, sim.move, on_iter=cb)
    return res, sim
