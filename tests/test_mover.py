"""Mover 单元测试 —— 全部走假 HTTP，PC 上不碰真车

覆盖：方向符号（u 正=后退 / invert 取反）、yaw 投影、动作完成判定、
超时急停、位姿复核重发、急停调用、配置装配、协议缺字段报错。
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from pid_servo.robot.mover import (Mover, MoverConfig, MoverError,
                                   PATH_ACTIONS, PATH_CURRENT, PATH_POSE)


class Resp:
    """最小 Response 假件：json() 返回预置数据，状态码默认 200"""

    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code

    def json(self):
        if isinstance(self._data, Exception):
            raise self._data
        return self._data


class FakeHttp:
    """按队列出数据的假底盘

    :param poses: 每次 GET pose 依次弹出一个（耗尽则保持最后一个）
    :param statuses: 每次查询动作状态依次弹出一个（耗尽则保持最后一个）
    :param action_ids: 每次 POST 动作依次弹出一个 action_id
    """

    def __init__(self, poses, statuses=(4,), action_ids=(101, 102, 103)):
        self.poses = list(poses)
        self.statuses = list(statuses)
        self.action_ids = list(action_ids)
        self.posted = []
        self.gets = []
        self.deletes = []

    def _pop(self, queue):
        return queue[-1] if len(queue) == 1 else queue.pop(0)

    def get(self, path):
        self.gets.append(path)
        if path == PATH_POSE:
            return Resp(self._pop(self.poses))
        return Resp({"state": {"status": self._pop(self.statuses)}})

    def post_json(self, path, body):
        self.posted.append((path, body))
        return Resp({"action_id": self._pop(self.action_ids)})

    def delete(self, path):
        self.deletes.append(path)
        return Resp({})


def make_mover(http, **over):
    params = dict(base_url="http://fake:1448", settle_ms=0, poll_interval_s=0.0)
    params.update(over)
    return Mover(http, MoverConfig(**params))


def pose(x=2.0, y=3.0, yaw=0.0):
    return {"x": x, "y": y, "yaw": yaw}


# ---- 方向符号 ----

def test_backward_positive_u_moves_against_heading():
    """u 正 = 后退：机头朝 +x（yaw=0）时，u=+0.2 的目标应在 x 减小侧"""
    http = FakeHttp(poses=[pose(x=2.0, yaw=0.0), pose(x=1.8, yaw=0.0)],
                    statuses=[4])
    make_mover(http).move(0.2)
    target = http.posted[0][1]["options"]["target"]
    assert target["x"] == pytest.approx(1.8)
    assert target["y"] == pytest.approx(3.0)


def test_forward_negative_u_moves_toward_heading():
    """u 负 = 前进：u=-0.05 的目标应在机头方向（x 增大侧）"""
    http = FakeHttp(poses=[pose(x=2.0, yaw=0.0), pose(x=2.05, yaw=0.0)],
                    statuses=[4])
    make_mover(http).move(-0.05)
    assert http.posted[0][1]["options"]["target"]["x"] == pytest.approx(2.05)


def test_invert_dir_flips_sign():
    """invert_dir=true：u 正值变为沿机头方向（现场发现方向反了的一键开关）"""
    http = FakeHttp(poses=[pose(x=2.0, yaw=0.0), pose(x=2.2, yaw=0.0)],
                    statuses=[4])
    make_mover(http, invert_dir=True).move(0.2)
    assert http.posted[0][1]["options"]["target"]["x"] == pytest.approx(2.2)


def test_yaw_projection():
    """机头 yaw=90°（朝 +y）时，后退 0.1 应作用于 y 减小方向"""
    http = FakeHttp(poses=[pose(x=5.0, y=7.0, yaw=math.pi / 2),
                           pose(x=5.0, y=6.9, yaw=math.pi / 2)],
                    statuses=[4])
    make_mover(http).move(0.1)
    target = http.posted[0][1]["options"]["target"]
    assert target["y"] == pytest.approx(6.9)
    assert target["x"] == pytest.approx(5.0)


def test_move_body_keeps_yaw_and_flags():
    """下发报文：yaw 保持当前机头角，flags 带 with_yaw+precise（协议与学长系统一致）"""
    http = FakeHttp(poses=[pose(yaw=0.3), pose(x=1.8, yaw=0.3)], statuses=[4])
    make_mover(http).move(0.2)
    opts = http.posted[0][1]["options"]
    assert opts["move_options"]["yaw"] == pytest.approx(0.3)
    assert opts["move_options"]["flags"] == ["with_yaw", "precise"]
    assert http.posted[0][0] == PATH_ACTIONS


def test_zero_delta_skips_chassis():
    """零位移不发动作、不查状态（伺服理论上不会出现，防御性）"""
    http = FakeHttp(poses=[pose()], statuses=[4])
    p = make_mover(http).move(0.0)
    assert http.posted == []
    assert p["x"] == pytest.approx(2.0)


# ---- 完成判定 / 超时 / 急停 ----

def test_done_status4_returns_final_pose():
    """status==4 判完成，返回复核后的位姿"""
    http = FakeHttp(poses=[pose(x=2.0), pose(x=1.8)], statuses=[2, 4])
    p = make_mover(http).move(0.2)
    assert p["x"] == pytest.approx(1.8)
    assert http.gets[-1] == PATH_POSE


def test_timeout_estops_and_raises():
    """等待超时：先补发急停（DELETE :current），再抛 MoverError"""
    http = FakeHttp(poses=[pose()], statuses=[2, 2, 2, 2])
    with pytest.raises(MoverError, match="超时"):
        make_mover(http, action_timeout_s=0.05, poll_interval_s=0.01).move(0.2)
    assert http.deletes == [PATH_CURRENT]


def test_stop_deletes_current():
    http = FakeHttp(poses=[pose()])
    make_mover(http).stop()
    assert http.deletes == [PATH_CURRENT]


# ---- 位姿复核 ----

def test_recheck_resends_when_position_off():
    """完成后实测位姿偏离目标超过容差 → 自动重发一次，第二次到位即通过"""
    http = FakeHttp(
        # 初始位姿 → 完成后第一次读（偏差 0.15m > 容差 0.10m）→ 重发完成后读（偏差 0）
        poses=[pose(x=2.0, yaw=0.0), pose(x=1.65, yaw=0.0), pose(x=1.80, yaw=0.0)],
        statuses=[4, 4], action_ids=[1, 2])
    make_mover(http, position_tol_m=0.10).move(0.2)
    assert len(http.posted) == 2


def test_recheck_exhausted_raises():
    """复核始终不达标：重发次数用尽后抛错（绝不带病继续）"""
    http = FakeHttp(
        poses=[pose(x=2.0, yaw=0.0), pose(x=1.65, yaw=0.0), pose(x=1.65, yaw=0.0)],
        statuses=[4, 4])
    with pytest.raises(MoverError, match="复核不达标"):
        make_mover(http, position_tol_m=0.10).move(0.2)


# ---- 配置装配 / 协议健壮性 ----

def test_from_config_reads_control_mover():
    cfg = {"control": {"mover": {"base_url": "http://10.0.0.9:1448",
                                 "invert_dir": True, "settle_ms": 0}}}
    m = Mover.from_config(cfg, http=FakeHttp(poses=[pose()]))
    assert m.cfg.base_url == "http://10.0.0.9:1448"
    assert m.cfg.invert_dir is True


def test_missing_action_id_raises():
    http = FakeHttp(poses=[pose(), pose()], statuses=[4], action_ids=[None])
    with pytest.raises(MoverError, match="action_id"):
        make_mover(http).move(0.2)


def test_http_error_status_raises():
    http = FakeHttp(poses=[pose()])
    http.post_json = lambda path, body: Resp({"error": "x"}, status_code=500)
    with pytest.raises(MoverError, match="500"):
        make_mover(http).move(0.2)
