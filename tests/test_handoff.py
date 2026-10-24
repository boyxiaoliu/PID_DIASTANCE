"""交接客户端测试 —— 改底盘 POI + 向学长系统派任务（全假件，PC 可跑）

接口形状依据学长系统源码（只读调研，不改他的代码）：
    底盘 artifact REST：chassis_control.py API_Url（GET/PUT /api/core/artifact/v1/pois）
    学长 Flask API：lr_restful_api.py（POST /api/connect、POST /api/task）
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pid_servo.robot.handoff import HandoffClient, HandoffError

#: 底盘 POI 库样例（形状照学长 ia.move 的解析：metadata.display_name 为键）
POI_A = {"uuid": "u-111", "pose": {"x": 1.0, "y": 2.0, "z": 0.0, "yaw": 0.5},
         "metadata": {"display_name": "4_B_094_L_8_1"}}
POI_B = {"uuid": "u-222", "pose": {"x": 3.0, "y": 4.0, "yaw": -1.2},
         "metadata": {"display_name": "4_B_094_L_9_1"}}


class FakeResp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = str(body)

    def json(self):
        return self._body


class FakeHttp:
    """ChassisHttp 同形假件：按 (方法, 路径前缀) 路由返回"""

    def __init__(self, routes):
        self.routes = routes            # (method, path_prefix, status, body)
        self.calls = []                 # (method, path, body)

    def get(self, path):
        return self._call("GET", path)

    def post_json(self, path, body):
        return self._call("POST", path, body)

    def put_json(self, path, body):
        return self._call("PUT", path, body)

    def _call(self, method, path, body=None):
        self.calls.append((method, path, body))
        for m, prefix, status, resp in self.routes:
            if m == method and path.startswith(prefix):
                return FakeResp(status, resp)
        return FakeResp(404, {"error": "no route"})


def make_client(task_ok=True, put_status=200):
    chassis = FakeHttp([
        ("GET", "/api/core/artifact/v1/pois", 200, [POI_A, POI_B]),
        ("PUT", "/api/core/artifact/v1/pois/", put_status, POI_A),
    ])
    senior = FakeHttp([
        ("POST", "/api/connect", 200, {"success": True}),
        ("POST", "/api/task", 200, {"success": task_ok, "task_id": 42}),
    ])
    return HandoffClient(chassis, senior), chassis, senior


PARAMS = {"floor": 4, "area": "B", "rack": 94, "side": "L", "column": 8}
POSE = {"x": 9.5, "y": -3.2, "yaw": 0.31}


def test_find_poi_by_display_name():
    client, _, _ = make_client()
    assert client.find_poi("4_B_094_L_8_1")["uuid"] == "u-111"


def test_find_poi_missing_raises():
    client, _, _ = make_client()
    with pytest.raises(HandoffError, match="display_name"):
        client.find_poi("4_B_094_L_9_9")


def test_update_poi_replaces_pose_keeps_identity():
    """只动 x/y/yaw：uuid、metadata、pose 里的 z 都原样保留（身份信息不能丢）"""
    client, chassis, _ = make_client()
    client.update_poi_pose("4_B_094_L_8_1", POSE)

    method, path, body = chassis.calls[-1]
    assert (method, path) == ("PUT", "/api/core/artifact/v1/pois/u-111")
    assert body["pose"]["x"] == 9.5 and body["pose"]["y"] == -3.2
    assert body["pose"]["yaw"] == 0.31
    assert body["pose"]["z"] == 0.0                     # 原 pose 的 z 保留
    assert body["uuid"] == "u-111"
    assert body["metadata"]["display_name"] == "4_B_094_L_8_1"


def test_update_poi_rejected_raises():
    client, _, _ = make_client(put_status=500)
    with pytest.raises(HandoffError, match="被拒绝"):
        client.update_poi_pose("4_B_094_L_8_1", POSE)


def test_submit_task_connects_first():
    """接口文档要求：提交任务前必须先 /api/connect"""
    client, _, senior = make_client()
    task_id = client.submit_task(1, PARAMS)
    assert task_id == 42
    assert [c[0] + ":" + c[1] for c in senior.calls] == \
        ["POST:/api/connect", "POST:/api/task"]
    assert senior.calls[-1][2] == {"task_type": 1, "params": PARAMS}


def test_submit_task_rejected_raises():
    client, _, _ = make_client(task_ok=False)
    with pytest.raises(HandoffError, match="被拒"):
        client.submit_task(1, PARAMS)


def test_handoff_full_flow():
    client, chassis, senior = make_client()
    summary = client.handoff("4_B_094_L_8_1", POSE, 1, PARAMS, update_poi=True)
    assert summary == {"poi_updated": True, "task_id": 42}
    assert any(c[0] == "PUT" for c in chassis.calls)
    assert any(c[1] == "/api/task" for c in senior.calls)


def test_handoff_without_pose_skips_poi_update():
    """没有位姿（如干跑后端）不拦任务：改 POI 如实标 None，任务照派"""
    client, chassis, _ = make_client()
    summary = client.handoff("4_B_094_L_8_1", None, 1, PARAMS, update_poi=True)
    assert summary["poi_updated"] is None and summary["task_id"] == 42
    assert not any(c[0] == "PUT" for c in chassis.calls)


def test_handoff_update_disabled():
    client, chassis, _ = make_client()
    summary = client.handoff("4_B_094_L_8_1", POSE, 1, PARAMS, update_poi=False)
    assert summary["poi_updated"] is False and summary["task_id"] == 42
    assert not any(c[0] == "PUT" for c in chassis.calls)
