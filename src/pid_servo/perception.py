"""感知层〔线A 维护〕—— 书本最低点 + 横杆检测

两个来源产出同一契约（observation.py）：
    离线  从学长系统的检测 JSON 读 OBB 书脊框（本模块 spine_stats_from_json）
    真机  线B 调 RKNN 模型，组装成同样的 SpineBottom（S4 落地）

横杆检测 = 行剖面取峰（移植自 旧PID控距/src/pid_servo/detect.py，同组自有代码）
         + 全宽逐列扫描 + cv2.fitLine(DIST_HUBER) 稳健直线拟合。

斜视处理的设计决策（S1 规划讨论定稿，轮子来源见实验记录）：
    roll（画面旋转）和 yaw（斜视）都会把横杆从水平线变成斜线 —— 3D 直线在针孔
    投影下仍是 2D 直线，所以对边点云做一次稳健直线拟合通吃两种歪：
    中心截距 = 控制信号；斜率 = 斜度体检指标；内点率 = 结构可信度体检指标。
"""

import json
import math

from dataclasses import dataclass, field

import cv2
import numpy as np

from .observation import BarEdge, Observation, SpineBottom


# ========== 书本最低点：离线 JSON 来源 ==========

def spine_stats_from_json_dict(data, min_confidence=0.50, max_boxes=20):
    """从一份检测 JSON 提取 书本最低点 + 该帧倾角

    :param data: json.load 后的字典，结构见 图书馆数据8_7/02_检测JSON
    :param min_confidence: 置信度门槛，低于此的框不参与统计
    :param max_boxes: 按置信度取前 K 个框，防海量误检拉偏统计
    :return: (SpineBottom | None, roll_deg | None)
        roll_deg = 各框长轴偏离竖直角度的中位数（度，正 = 顶端偏右）。
        书在真实世界是竖直的，整帧一起偏 = 相机 roll —— 这就是免费水平仪。
    """
    boxes = []
    for rec in (data.get("books") or {}).values():
        # books 里混有 OCR 位置表等非书条目（position_ocr / left_map_ocr，
        # 值是 list），只认真正带书脊检测框的条目
        if not isinstance(rec, dict) or "yolo11n_obb_detection" not in rec:
            continue
        det = rec.get("yolo11n_obb_detection") or {}
        conf = float(det.get("confidence", 0.0))
        if conf < min_confidence:
            continue
        rr = det.get("rotated_rect")
        if not rr or len(rr) != 3:
            continue
        (cx, cy), (w, h), ang = rr
        boxes.append((conf, float(cx), float(cy), float(w), float(h), float(ang)))

    boxes.sort(key=lambda b: -b[0])
    boxes = boxes[:max_boxes]
    if not boxes:
        return None, None

    bottoms, tilts, confs = [], [], []
    for conf, cx, cy, w, h, ang in boxes:
        t = math.radians(ang)
        # 旋转矩形底边：中心 y + 长短轴在 y 向的半投影
        bottoms.append(cy + (abs(w * math.sin(t)) + abs(h * math.cos(t))) / 2.0)
        phi = (ang if w >= h else ang + 90.0) % 180.0   # 长轴方位角归一到 [0,180)
        tilts.append(phi - 90.0)
        confs.append(conf)

    spine = SpineBottom(y=max(bottoms), confidence=min(confs), n_books=len(boxes))
    # ⚠️ 此处角度按"度"解释 —— TODO(verify) 由 S1 体检的 roll 分布验证（应聚在 0° 附近）
    return spine, float(np.median(tilts))


def spine_stats_from_json_file(path, **kwargs):
    """spine_stats_from_json_dict 的文件版，方便工具直接调用"""
    with open(path, "r", encoding="utf-8") as f:
        return spine_stats_from_json_dict(json.load(f), **kwargs)


# ========== 剖面工具（移植自旧项目 detect.py） ==========

def highpass(prof, sigma):
    """一维高通：减去大核高斯模糊，只留局部起伏"""
    if sigma < 3 or prof.size < 3:
        return prof.astype(np.float32).copy()
    k = min(sigma | 1, (prof.size - 1) | 1)
    base = cv2.GaussianBlur(prof.reshape(-1, 1).astype(np.float32), (1, k), 0).ravel()
    return prof.astype(np.float32) - base


def _subpixel(v, i):
    """三点抛物线插值，峰位精确到亚像素"""
    if i <= 0 or i >= len(v) - 1:
        return float(i)
    a, b, c = float(v[i - 1]), float(v[i]), float(v[i + 1])
    denom = a - 2.0 * b + c
    if abs(denom) < 1e-9:
        return float(i)
    return i + max(-1.0, min(1.0, 0.5 * (a - c) / denom))


def _pick_peak(hp, lo, hi, min_contrast, rel_threshold, pick="bottommost"):
    """在 hp[lo:hi) 内选一条显著正峰，返回亚像素 y；位置优先，强度只做门槛"""
    lo, hi = max(0, lo), min(len(hp), hi)
    if hi - lo < 3:
        return None
    seg = hp[lo:hi]
    thr = max(min_contrast, rel_threshold * float(seg.max()))

    core = seg[1:-1]
    is_peak = (core > seg[:-2]) & (core >= seg[2:]) & (core >= thr)
    idx = np.flatnonzero(is_peak) + 1
    if idx.size == 0:
        return None

    if pick == "topmost":
        k = int(idx[0])
    elif pick == "bottommost":
        k = int(idx[-1])
    elif pick == "strongest":
        k = int(idx[int(np.argmax(seg[idx]))])
    else:
        raise ValueError(f"未知的 pick 策略: {pick}")
    return lo + _subpixel(seg, k)


def _ransac_line(P, tol, iters=500, seed=0):
    """RANSAC 直线拟合：找支持点最多的一条边（点云混有多种边时锁定主导边）

    背景：横杆下方沿宽度方向混有 横杆顶边/标签牌底边/立柱 等多种结构，
    M-estimator(HUBER) 会被少数派拉偏 —— S1 真图实验（097 内点率仅 45%）驱动的升级。
    固定种子保证批次可复现。

    :return: (k, b, inlier_mask)  最小二乘重拟合后的直线与内点掩码
    """
    rng = np.random.default_rng(seed)
    n = len(P)
    best_mask, best_k, best_b = None, 0.0, 0.0
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        dx = P[j, 0] - P[i, 0]
        if abs(dx) < 1e-3:
            continue
        k = (P[j, 1] - P[i, 1]) / dx
        b = P[i, 1] - k * P[i, 0]
        mask = np.abs(P[:, 1] - (k * P[:, 0] + b)) <= tol
        if best_mask is None or mask.sum() > best_mask.sum():
            best_mask, best_k, best_b = mask, k, b
    Q = P[best_mask]
    best_k, best_b = np.polyfit(Q[:, 0], Q[:, 1], 1)
    best_mask = np.abs(P[:, 1] - (best_k * P[:, 0] + best_b)) <= tol
    return float(best_k), float(best_b), best_mask


def yellow_crosses_blue(fit, spine_bottom_y, img_w):
    """黄线（横杆下沿拟合直线）在画面内是否触及/越过蓝线（书本最低点）

    物理不变量：横杆下沿恒在书底之下 → 两条线在画面内绝不允许相交。
    蓝线是水平线（单一 y 值），黄线是斜线，故只需检查黄线在画面左右两端
    的较低端是否 ≤ 蓝线（G1 用户约束）。
    """
    if fit is None or spine_bottom_y is None:
        return False
    y_left = fit.y_center - fit.slope * (img_w / 2.0)
    y_right = fit.y_center + fit.slope * (img_w / 2.0)
    return min(y_left, y_right) <= spine_bottom_y


# ========== 横杆检测器 ==========

@dataclass
class BarFit:
    """一次横杆直线拟合的全部产出（工具/体检直接用；进契约时压缩成 BarEdge）

    :param y_center: 拟合直线在画面水平中心 x=W/2 处的 y —— 控制信号
    :param slope: 直线斜率 dy/dx —— 斜度体检指标（roll+yaw 的合计表现）
    :param n_points: 参与拟合的点数
    :param inlier_ratio: 残差 <= inlier_tol_px 的点占比 —— 结构可信度体检指标
    :param residual_std: 内点残差标准差（像素），衡量这条线锁得稳不稳
    :param points: 全部扫描点 (N,2)，仅离线可视化用
    """
    y_center: float
    slope: float
    n_points: int
    inlier_ratio: float
    residual_std: float
    points: np.ndarray = field(default=None, repr=False)

    @property
    def slope_deg(self):
        return math.degrees(math.atan(self.slope))


class BarDetector:
    """横杆检测：窗口剖面取峰 → 全宽点云 → M-estimator 直线拟合"""

    name = "opencv_bar_fitline_v0"

    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.highpass_sigma = int(cfg.get("highpass_sigma", 201))
        self.spine_margin_frac = float(cfg.get("spine_margin_frac", 0.010))
        self.fallback_zone_frac = float(cfg.get("fallback_zone_frac", 0.85))
        self.min_contrast = float(cfg.get("min_contrast", 4.0))
        self.rel_threshold = float(cfg.get("rel_threshold", 0.40))
        self.win_frac = float(cfg.get("scan_win_frac", 0.07))
        self.step_frac = float(cfg.get("scan_step_frac", 0.035))
        self.pick = cfg.get("pick", "bottommost")
        self.min_points = int(cfg.get("min_points", 8))
        self.inlier_tol_px = float(cfg.get("inlier_tol_px", 12.0))
        self.fit_method = cfg.get("fit", "ransac")   # ransac=锁主导边 / huber=M-estimator

    @classmethod
    def from_config(cls, path=None):
        from . import utils
        return cls(utils.get(utils.load(path), "perception.bar", {}))

    def detect(self, frame, spine_bottom_y=None, keep_points=False):
        """对一帧检测横杆

        :param frame: BGR 或灰度图
        :param spine_bottom_y: 书本最低点像素 y（像素上界）；None 用兜底搜索区
        :param keep_points: 是否保留扫描点（离线可视化用，在线闭环关掉省内存）
        :return: BarFit | None（点太少/拟合退化 = 未检出）
        """
        if frame is None or getattr(frame, "size", 0) == 0:
            return None
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        # 搜索区上界：书本最低点往下让出一条余量；没有书脊信息时退到固定区
        if spine_bottom_y is not None:
            zone_top = spine_bottom_y + self.spine_margin_frac * h
        else:
            zone_top = self.fallback_zone_frac * h
        lo = int(zone_top)
        if h - lo < 3:
            return None          # 书本最低点已贴住画面底，横杆没有容身之处

        # 全宽逐列扫描（窗口重叠一半）
        win_w = max(8, int(self.win_frac * w))
        step = max(4, int(self.step_frac * w))
        pts = []
        for x0 in range(0, max(1, w - win_w + 1), step):
            band = gray[:, x0:x0 + win_w].astype(np.float32).mean(axis=1)
            hp = highpass(band, self.highpass_sigma)
            y = _pick_peak(hp, lo, h, self.min_contrast, self.rel_threshold, self.pick)
            if y is not None:
                pts.append((x0 + win_w / 2.0, y))
        if len(pts) < self.min_points:
            return None

        # 稳健直线拟合：RANSAC 锁主导边（真图上横杆下方混有多种边，S1 实验定）
        P = np.asarray(pts, dtype=np.float32)
        if self.fit_method == "ransac":
            k, b, inliers = _ransac_line(P, self.inlier_tol_px)
        else:
            vx, vy, px, py = cv2.fitLine(P, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            if abs(vx) < 1e-6:
                return None      # 近垂直不可能是横杆
            k, b = float(vy / vx), float(py - k * px)
            inliers = np.abs(P[:, 1] - (k * P[:, 0] + b)) <= self.inlier_tol_px
        y_center = float(k * (w / 2.0) + b)
        res = np.abs(P[:, 1] - (k * P[:, 0] + b))

        return BarFit(
            y_center=y_center,
            slope=k,
            n_points=len(P),
            inlier_ratio=float(inliers.mean()),
            residual_std=float(res[inliers].std()) if inliers.any() else float("inf"),
            points=P if keep_points else None,
        )

    def observe(self, frame, spine=None, source=None):
        """检测并组装成契约 Observation（闭环/真机统一入口）

        :param frame: 图像帧
        :param spine: SpineBottom 或 None（真机由 RKNN 产出后传入）
        :param source: 来源标注，None 用 "yolo_json+{检测器名}" / "none+{检测器名}"
        :return: Observation（bar_frac 可用时 usable=True）
        """
        fit = self.detect(frame, spine.y if spine is not None else None)
        if fit is not None and spine is not None and yellow_crosses_blue(fit, spine.y, frame.shape[1]):
            # 物理不变量被破坏 = 拟合锁错结构，读数作废（闭环按丢失处理 → 后退）
            obs = Observation(frame_size=frame.shape[:2], spine=spine, bar=None,
                              source=(source or f"{'spine' if spine else 'none'}+{self.name}"))
            obs.debug["reject"] = "yellow_crosses_blue"
            return obs
        obs = Observation(
            frame_size=frame.shape[:2] if frame is not None and getattr(frame, "size", 0) else (0, 0),
            spine=spine,
            bar=None if fit is None else BarEdge(y=fit.y_center, contrast=fit.inlier_ratio,
                                                 y_int=int(round(fit.y_center))),
            source=(source or f"{'spine' if spine else 'none'}+{self.name}"),
        )
        if fit is not None:
            obs.debug.update(slope=fit.slope, slope_deg=fit.slope_deg,
                             inlier_ratio=fit.inlier_ratio, n_points=fit.n_points)
        return obs

    def draw(self, frame, fit, spine_bottom_y=None, thick=3, font=1.1, pr=6):
        """可视化：拟合线、扫描点（绿=内点 红=离群）、搜索区上界，人工核对用

        :param thick/font/pr: 笔触/字号/点半径。在 4K 原图上画再缩小的场景
            （如评审页），传大值（如 thick=12）保证缩小后仍清晰可见。
        """
        vis = frame.copy()
        h, w = vis.shape[:2]
        if spine_bottom_y is not None:
            y_sb = int(spine_bottom_y)
            cv2.line(vis, (0, y_sb), (w, y_sb), (255, 128, 0), thick)
            cv2.putText(vis, f"spine_bottom={spine_bottom_y:.0f}", (20, max(40, y_sb - int(pr * 2))),
                        cv2.FONT_HERSHEY_SIMPLEX, font * 0.8, (255, 128, 0), thick, cv2.LINE_AA)
        if fit is None:
            cv2.putText(vis, "BAR NOT FOUND", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, font * 1.2, (0, 0, 255), thick, cv2.LINE_AA)
            return vis

        b = fit.y_center - fit.slope * (w / 2.0)
        if fit.points is not None:
            for x, y in fit.points:
                res = abs(y - (fit.slope * x + b))
                color = (0, 255, 0) if res <= self.inlier_tol_px else (0, 0, 255)
                cv2.circle(vis, (int(x), int(y)), pr, color, -1, cv2.LINE_AA)
        x1, x2 = 0, w
        cv2.line(vis, (x1, int(fit.slope * x1 + b)), (x2, int(fit.slope * x2 + b)),
                 (0, 255, 255), thick, cv2.LINE_AA)
        tag = (f"y={fit.y_center:.0f} ({fit.y_center / h:.3f}H)  "
               f"slope={fit.slope_deg:+.2f}deg  inliers={fit.inlier_ratio:.0%}")
        # 说明文字放左上角：底部正是黄线所在区域，别把它盖住
        cv2.putText(vis, tag, (20, 40 + int(pr * 3)), cv2.FONT_HERSHEY_SIMPLEX,
                    font, (0, 255, 255), thick, cv2.LINE_AA)
        return vis
