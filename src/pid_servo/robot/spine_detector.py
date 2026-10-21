"""书脊检测〔线B 维护〕—— RKNN OBB 模型 → 书本最低点（SpineBottom）

分工与复用：
    模型推理（板端件）在这里；「框 → 书本最低点」的数学不重复造——
    组装成学长检测 JSON 的同形字典后直接调用线A
    perception.spine_stats_from_json_dict（同一契约两处实现必须零分叉）。

坐标链（自洽约定，⚠️ TODO(S4上板) 若读数出现系统性偏差，优先核对本文件
    letterbox/decode 与学长 infer_sys 的 pre/post 约定差异）：
    原帧 --letterbox(ratio, pad)--> 模型输入 1024²
    模型输出 (1, 7, N)：[cx, cy, w, h, score_book, score_shelf, angle_deg]
    解码反变换：x_orig = (x_model - pad) / ratio；旋转框 NMS 后按类别取书脊
"""

import math

import numpy as np

from ..observation import SpineBottom
from ..perception import spine_stats_from_json_dict


# ========== 纯数学：合成框 → SpineBottom（复用线A实现，零分叉） ==========

def boxes_to_spine_stats(boxes, min_confidence=0.50, max_boxes=20):
    """把检测结果列表折算成 (SpineBottom | None, roll_deg | None)

    :param boxes: [(confidence, cx, cy, w, h, angle_deg), ...] 旋转矩形（度）
    内部组装成学长检测 JSON 同形字典，交给线A spine_stats_from_json_dict 统一处理
    （置信度过滤 / 前 K 框 / 底边取最大 / 置信度取最弱 / roll 取中位，全在线A实现里）。
    """
    books = {f"book_{i}": {"yolo11n_obb_detection": {
        "rotated_rect": [[cx, cy], [w, h], ang], "confidence": conf,
    }} for i, (conf, cx, cy, w, h, ang) in enumerate(boxes)}
    return spine_stats_from_json_dict({"books": books},
                                      min_confidence=min_confidence,
                                      max_boxes=max_boxes)


# ========== letterbox / 解码（自洽约定） ==========

def letterbox(frame, input_size=1024):
    """等比缩放 + 双侧补边到方图

    :return: (blob uint8 CHW RGB?, ratio, pad_x, pad_y)
    模型预处理通道序以学长部署为准 —— TODO(S4上板) 核对 BGR/RGB 与归一化
    """
    import cv2

    h, w = frame.shape[:2]
    ratio = min(input_size / w, input_size / h)
    nw, nh = int(round(w * ratio)), int(round(h * ratio))
    resized = cv2.resize(frame, (nw, nh))
    pad_x, pad_y = (input_size - nw) // 2, (input_size - nh) // 2
    canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, ratio, pad_x, pad_y


def _rotated_iou(a, b):
    """两个旋转矩形的 IoU（cv2 凸多边形求交，避免引入 shapely）"""
    import cv2

    ra = ((a[0], a[1]), (a[2], a[3]), a[4])
    rb = ((b[0], b[1]), (b[2], b[3]), b[4])
    pa, pb = cv2.boxPoints(ra), cv2.boxPoints(rb)
    inter, _ = cv2.intersectConvexConvex(pa.astype(np.float32), pb.astype(np.float32))
    area_a = abs(a[2] * a[3])
    area_b = abs(b[2] * b[3])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def decode_obb_raw(raw, ratio, pad_x, pad_y, conf_threshold=0.5,
                   class_id=0, iou_threshold=0.5, img_size=1024):
    """模型原始输出 → 原图坐标的 (conf, cx, cy, w, h, angle_deg) 列表

    :param raw: np.ndarray，形如 (1, 7, N) 或 (N, 7)，列 [cx,cy,w,h,s_book,s_shelf,angle]
    坐标列在模型输入系（含 pad），此处反变换回原图系；角度为度。
    ⚠️ 通道序/角度单位以板端联调为准（TODO(S4上板)）。
    """
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.shape[0] == 7 and arr.shape[-1] != 7:   # (7, N) → (N, 7)
        arr = arr.T
    if arr.ndim != 2 or arr.shape[1] < 7:
        raise ValueError(f"OBB 输出形状不符: {arr.shape}")

    cx = (arr[:, 0] - pad_x) / ratio
    cy = (arr[:, 1] - pad_y) / ratio
    w = arr[:, 2] / ratio
    h = arr[:, 3] / ratio
    scores = arr[:, 4 + class_id]
    angles = arr[:, 6]

    keep = []                                      # 保留下的是索引
    order = np.argsort(-scores)
    for i in order:
        if scores[i] < conf_threshold:
            break                                  # 已按分数降序，后面全低
        box = (cx[i], cy[i], w[i], h[i], angles[i])
        if all(_rotated_iou(box, (cx[j], cy[j], w[j], h[j], angles[j])) < iou_threshold
               for j in keep):
            keep.append(int(i))
    return [(float(scores[i]), float(cx[i]), float(cy[i]),
             float(w[i]), float(h[i]), float(angles[i])) for i in keep]


# ========== 板端检测器 ==========

class RknnSpineDetector:
    """RknnSpineDetector 模型常驻的书脊检测件（frame -> SpineBottom | None）

    rknnlite 惰性导入：PC 单测注入假 inference；模型只在 init 时加载一次
    （防板端 60s 空闲冷启动，闭环全程常驻）。

    :param model_path: .rknn 模型路径（学长 all_models/yolo11n_obb_model 同源）
    :param min_confidence / max_boxes: 语义与 perception.spine 节同源（传给线A实现）
    :param class_id: 0 = book 书脊（学长 metadata 同源）
    :param last_roll_deg: 附赠品——每次调用的帧倾角中位数（度），仅调试/留痕用
    """

    name = "rknn_obb_spine_v0"

    def __init__(self, model_path, input_size=1024, min_confidence=0.50,
                 max_boxes=20, class_id=0, conf_threshold=None,
                 iou_threshold=0.5, runtime=None):
        self.model_path = model_path
        self.input_size = int(input_size)
        self.min_confidence = float(min_confidence)
        self.max_boxes = int(max_boxes)
        self.class_id = int(class_id)
        self.conf_threshold = float(conf_threshold if conf_threshold is not None
                                    else min_confidence)
        self.iou_threshold = float(iou_threshold)
        self.last_roll_deg = None
        self._runtime = runtime            # 注入的推理后端（board: RKNNLite 包装）
        if runtime is None:
            self._runtime = self._init_runtime(model_path)

    @classmethod
    def from_config(cls, cfg, runtime=None):
        """from_config 从完整配置字典构造（部署参数取 control.spine_detector 节）"""
        from ..utils import get

        return cls(
            model_path=get(cfg, "control.spine_detector.model_path", ""),
            input_size=get(cfg, "control.spine_detector.input_size", 1024),
            class_id=get(cfg, "control.spine_detector.class_id", 0),
            min_confidence=get(cfg, "perception.spine.min_confidence", 0.50),
            max_boxes=get(cfg, "perception.spine.max_boxes", 20),
            runtime=runtime,
        )

    @staticmethod
    def _init_runtime(model_path):
        if not model_path:
            raise ValueError("control.spine_detector.model_path 未配置"
                             "（板端 .rknn 路径，见 configs 注释）")
        from rknnlite.api import RKNNLite  # 惰性：仅板端存在

        rknn = RKNNLite()
        if rknn.load_rknn(model_path) != 0:
            raise RuntimeError(f"RKNN 模型加载失败: {model_path}")
        if rknn.init_runtime() != 0:
            raise RuntimeError("RKNN runtime 初始化失败")
        return rknn

    def __call__(self, frame):
        """frame -> SpineBottom | None；无书脊框时返回 None（bar_only 兜底仍可用）"""
        canvas, ratio, pad_x, pad_y = letterbox(frame, self.input_size)
        raw = self._runtime.inference(inputs=[np.expand_dims(canvas, 0)])
        boxes = decode_obb_raw(raw, ratio, pad_x, pad_y,
                               conf_threshold=self.conf_threshold,
                               class_id=self.class_id,
                               iou_threshold=self.iou_threshold,
                               img_size=self.input_size)
        spine, roll = boxes_to_spine_stats(boxes, self.min_confidence,
                                           self.max_boxes)
        self.last_roll_deg = roll
        return spine
