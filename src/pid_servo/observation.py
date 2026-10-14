"""感知 ↔ 控制 的数据契约（两线唯一接缝）

本模块只定义数据，不含任何检测或控制逻辑：
    线A perception.py 产出 Observation（离线 JSON / 真机 RKNN 两个来源都产出它）
    线B servo.py 只消费 Observation，不关心它从哪来

符号约定（AGENTS.md §6，只此一处定义）：
    d         拍摄距离（米）
    bar_frac  横杆像素 y / 画面高，bar_frac ≈ A + B/d，越近值越大
    e         bar_frac - target_bar_frac
    u         PID 输出（米），正值 = 后退（远离书架）

⚠️ 改动本文件需两线 Owner 都同意（AGENTS.md §2），提交信息加 `契约:` 前缀。
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BarEdge:
    """横杆（下搁板前沿/标签条）的一条水平边缘定位结果

    :param y: 亚像素 y 坐标（图像坐标系向下为正）
    :param contrast: 峰处灰度差（0~255），代表这条边的可信度
    :param y_int: 取整位置，仅用于调试图上画标记
    """
    y: float
    contrast: float
    y_int: int = 0


@dataclass(frozen=True)
class SpineBottom:
    """书本最低点：该帧所有书脊 OBB 框底边的最大 y

    由两个来源产出同一结构 —— 线A 离线读检测 JSON、线B 真机跑 RKNN：
    :param y: 最低点像素 y
    :param confidence: 参与统计的框中最低置信度（取最弱者，代表该统计的可信度）
    :param n_books: 参与统计的书脊框数，0 表示该帧没检出任何书脊
    """
    y: float
    confidence: float
    n_books: int


@dataclass
class Observation:
    """一帧图像对「当前书架层」的完整观测

    这是感知层与控制层之间唯一的数据契约。

    :param frame_size: 画面尺寸 (h, w)
    :param spine: 书本最低点；None = 该帧未检出书脊（YOLO 丢失或离线无 JSON）
    :param bar: 横杆定位；None = 未找到横杆
    :param source: 产出该观测的检测器/来源名，如 "offline_json+opencv" / "rknn+opencv"
    :param debug: 中间量，仅调试/可视化用，闭环逻辑不得依赖
    """
    frame_size: tuple
    spine: SpineBottom = None
    bar: BarEdge = None
    source: str = "unknown"
    debug: dict = field(default_factory=dict)

    @property
    def img_h(self):
        return self.frame_size[0]

    @property
    def img_w(self):
        return self.frame_size[1]

    @property
    def bar_frac(self):
        """横杆归一化 y 位置（误差信号的原材料），无横杆时 None"""
        if self.bar is None or self.img_h <= 0:
            return None
        return self.bar.y / self.img_h

    @property
    def mode(self):
        """观测模式，控制器按模式选策略：
        yolo_bar  书脊+横杆都有（正常态，横杆搜索区受书脊约束）
        bar_only  只有横杆（书脊丢失，兜底搜索区，可信度降级但可用）
        none      两者皆无（本帧不可用）
        """
        if self.bar is not None:
            return "yolo_bar" if self.spine is not None else "bar_only"
        return "none"

    @property
    def usable(self):
        """本帧能否用于闭环推进：误差只依赖横杆，横杆在即可用"""
        return self.bar is not None
