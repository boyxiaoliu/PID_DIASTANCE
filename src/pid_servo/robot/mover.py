"""底盘移动器〔线B 维护〕—— Athena 2.0 REST 的增量位移原语（项目符号约定的"腿"）

底盘只有绝对坐标动作（无速度级控制，Athena 手册 §7 + 学长系统在用代码一致），
所以「u 米，正 = 后退」这个项目约定的翻译方式是：

    目标 = 当前位姿 沿视线方向（机头 yaw）推 -u 米      →  现算绝对坐标再发 MoveToAction

动作完成后读位姿复核，不达标重发（沿用学长系统 IA.move 的习惯，但有重发上限）。

三原语（与学长系统 chassis_control.py 逐字核对过，Athena 手册 §7.5 同源）：
    读位姿  GET    /api/core/slam/v1/localization/pose            → {x, y, yaw}
    发动作  POST   /api/core/motion/v1/actions                    → {action_id}
    查状态  GET    /api/core/motion/v1/actions/{action_id}        → state.status == 4 完成
    急停    DELETE /api/core/motion/v1/actions/:current

安全约定：
    - 动作等待超时：先补一发急停再抛错（宁可多停一次，不可盲走）
    - 位姿复核始终不达标：重发 max_recheck 次后抛错，绝不带病继续
    - 方向符号首次实机必须核对：configs invert_dir 一键取反，不用改代码
"""

import copy
import math
import time
from dataclasses import dataclass

from ..utils import get

# ---- 底盘 REST 路径（Athena 2.0 / Robot Agent，与学长系统逐字一致）----

PATH_POSE = "/api/core/slam/v1/localization/pose"
PATH_ACTIONS = "/api/core/motion/v1/actions"
PATH_ACTION_FMT = "/api/core/motion/v1/actions/{action_id}"      # GET 查状态
PATH_CURRENT = "/api/core/motion/v1/actions/:current"            # DELETE 急停

#: 查询接口里表示「动作已完成」的状态码（学长系统 IA.move 同值）
STATUS_DONE = 4

#: MoveToAction 请求模板（协议报文，非可调参数；precision/retry/speed 沿用学长系统填 0）
_MOVE_TEMPLATE = {
    "action_name": "slamtec.agent.actions.MoveToAction",
    "options": {
        "target": {"x": 0.0, "y": 0.0, "z": 0.0},
        "move_options": {
            "mode": 0,
            "flags": ["with_yaw", "precise"],
            "yaw": 0.0,
            "acceptable_precision": 0,
            "fail_retry_count": 0,
            "speed_ratio": 0,
        },
    },
}


class MoverError(RuntimeError):
    """底盘移动失败（超时/位姿复核不达标/响应不符合协议）——调用方应停下来看日志"""


# ========== HTTP 薄封装（测试时整体替换） ==========

class ChassisHttp:
    """requests 的极薄封装：只多出 base_url 与超时，便于单测注入假件

    requests 为板端依赖（学长系统环境自带），惰性导入——PC 单测注入 FakeHttp 即可。
    """

    def __init__(self, base_url, request_timeout_s=5.0):
        import requests  # 惰性：PC 单测不装 requests 也能全绿

        self.base_url = base_url.rstrip("/")
        self.request_timeout_s = float(request_timeout_s)
        self._requests = requests

    def get(self, path):
        return self._requests.get(self.base_url + path, timeout=self.request_timeout_s)

    def post_json(self, path, body):
        return self._requests.post(self.base_url + path, json=body,
                                   timeout=self.request_timeout_s)

    def put_json(self, path, body):
        return self._requests.put(self.base_url + path, json=body,
                                  timeout=self.request_timeout_s)

    def delete(self, path):
        return self._requests.delete(self.base_url + path, timeout=self.request_timeout_s)


# ========== 配置 ==========

@dataclass
class MoverConfig:
    """MoverConfig 底盘移动参数（configs control.mover 节，默认值与之一致）

    :param base_url: 底盘 Robot Agent 地址（学长系统 pipeline.yaml chassis_url 同源）
    :param action_timeout_s: 单次动作等待上限；超时先急停再抛 MoverError
    :param poll_interval_s: 动作状态轮询间隔
    :param request_timeout_s: 单个 HTTP 请求超时（学长系统同值 5s）
    :param position_tol_m: 完成后位姿复核容差（学长系统 check_position 同值 0.1m）
    :param max_recheck: 位姿复核不达标时的重发次数上限
    :param invert_dir: 方向取反开关。false = u 正值沿机头反方向退（标准约定）；
        首次实机若发现方向反了，改这一个键即可
    :param settle_ms: 移动完成后等停稳+曝光稳定的时长（自 control.saturation 移入）
    """

    base_url: str = "http://10.230.152.204:1448"
    action_timeout_s: float = 60.0
    poll_interval_s: float = 0.2
    request_timeout_s: float = 5.0
    position_tol_m: float = 0.10
    max_recheck: int = 1
    invert_dir: bool = False
    settle_ms: int = 800

    @classmethod
    def from_dict(cls, d):
        """from_dict 从配置字典构造，忽略多余键（配置可比代码先演进）"""
        d = d or {}
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_config(cls, cfg):
        """from_config 从完整配置字典的 control.mover 节构造"""
        return cls.from_dict(get(cfg, "control.mover", {}))


# ========== Mover ==========

class Mover:
    """Mover 增量位移原语：move(u) 接口与 shelf_sim.move 完全同形（正 = 后退）

    :param http: ChassisHttp 或同形假件（单测注入）
    :param cfg: MoverConfig
    """

    def __init__(self, http, cfg=None):
        self.http = http
        self.cfg = cfg if isinstance(cfg, MoverConfig) else MoverConfig.from_dict(cfg)

    @classmethod
    def from_config(cls, cfg, http=None):
        """from_config 按完整配置字典构造；http 为 None 时建真实 ChassisHttp"""
        mc = MoverConfig.from_config(cfg)
        if http is None:
            http = ChassisHttp(mc.base_url, mc.request_timeout_s)
        return cls(http, mc)

    # ---- 三原语 ----

    def get_pose(self):
        """get_pose 读当前位姿，返回 {x, y, yaw}（米/弧度）"""
        resp = self.http.get(PATH_POSE)
        data = self._json_of(resp, "get_pose")
        try:
            return {"x": float(data["x"]), "y": float(data["y"]),
                    "yaw": float(data["yaw"])}
        except (KeyError, TypeError, ValueError) as e:
            raise MoverError(f"位姿响应缺少字段: {data!r}（{e}）")

    def stop(self):
        """stop 急停：终止底盘当前动作（DELETE :current）"""
        resp = self.http.delete(PATH_CURRENT)
        self._json_of(resp, "stop")

    # ---- 项目约定接口：正 = 后退 ----

    def move(self, delta_m):
        """move 沿视线方向相对移动 |delta_m| 米，阻塞到完成+复核+settle

        :param delta_m: 位移（米，正 = 后退远离书架； invert_dir=true 时语义取反）
        :return: 完成复核后的位姿 {x, y, yaw}
        """
        delta_m = float(delta_m)
        if abs(delta_m) < 1e-9:                     # 零位移不值得跑一轮底盘
            time.sleep(self.cfg.settle_ms / 1000.0)
            return self.get_pose()

        pose = self.get_pose()
        # 机头 yaw 指向书架：u 正值（后退）沿机头反方向走
        sign = (+1.0 if self.cfg.invert_dir else -1.0) * delta_m
        tx = pose["x"] + sign * math.cos(pose["yaw"])
        ty = pose["y"] + sign * math.sin(pose["yaw"])

        for attempt in range(self.cfg.max_recheck + 1):
            action_id = self._post_move(tx, ty, pose["yaw"])
            self._wait_done(action_id)
            pose = self.get_pose()
            if self._close_enough(pose, tx, ty):
                break
            if attempt == self.cfg.max_recheck:
                raise MoverError(
                    f"位姿复核不达标: 目标({tx:.3f},{ty:.3f}) 实际({pose['x']:.3f},"
                    f"{pose['y']:.3f}) 已重发 {self.cfg.max_recheck} 次")
        if self.cfg.settle_ms > 0:
            time.sleep(self.cfg.settle_ms / 1000.0)
        return pose

    # ---- 内部 ----

    def _post_move(self, tx, ty, yaw):
        """_post_move 下发绝对坐标 MoveToAction，返回 action_id"""
        body = copy.deepcopy(_MOVE_TEMPLATE)
        body["options"]["target"]["x"] = float(tx)
        body["options"]["target"]["y"] = float(ty)
        body["options"]["move_options"]["yaw"] = float(yaw)
        resp = self.http.post_json(PATH_ACTIONS, body)
        data = self._json_of(resp, "post_move")
        if not isinstance(data, dict) or data.get("action_id") is None:
            raise MoverError(f"动作响应缺少 action_id: {data!r}")
        return data["action_id"]

    def _wait_done(self, action_id):
        """_wait_done 轮询动作状态直到 status==4；超时先急停再抛错"""
        path = PATH_ACTION_FMT.format(action_id=action_id)
        deadline = time.monotonic() + self.cfg.action_timeout_s
        last_status = None
        while True:
            resp = self.http.get(path)
            data = self._json_of(resp, "get_action")
            try:
                last_status = data["state"]["status"]
            except (KeyError, TypeError) as e:
                raise MoverError(f"状态响应缺少 state.status: {data!r}（{e}）")
            if last_status == STATUS_DONE:
                return
            if time.monotonic() >= deadline:
                try:
                    self.stop()
                except MoverError:
                    pass                    # 急停失败不掩盖原始超时错误
                raise MoverError(f"动作 {action_id} 等待超时"
                                 f"（{self.cfg.action_timeout_s:.0f}s，末状态 {last_status}），"
                                 f"已补发急停")
            time.sleep(self.cfg.poll_interval_s)

    def _close_enough(self, pose, tx, ty):
        """_close_enough 水平距离复核（学长系统 check_position 的 0.1m 同源）"""
        return math.hypot(pose["x"] - tx, pose["y"] - ty) <= self.cfg.position_tol_m

    @staticmethod
    def _json_of(resp, ctx):
        """_json_of 校验 HTTP 状态与 JSON 可解析性，统一抛 MoverError"""
        status = getattr(resp, "status_code", None)
        try:
            data = resp.json()
        except Exception as e:
            raise MoverError(f"{ctx}: 响应不是 JSON（status={status}，{e}）")
        if status is not None and status != 200:
            raise MoverError(f"{ctx}: HTTP {status}，body={data!r}")
        return data
