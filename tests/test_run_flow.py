"""S5 流程测试 —— run_servo_real 的"闭环→终验→确认照"全链路（假件注入，PC 可跑）

    python -m pytest tests/test_run_flow.py
"""

import importlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from pid_servo.observation import BarEdge, Observation, SpineBottom


def make_tool():
    return importlib.import_module("run_servo_real")


def make_obs(frac):
    """按 bar_frac 构造观测（None = 黄蓝全无的丢失帧）

    书脊固定在 y=430（0.80H）：保证 0.86 这种带外帧仍是正常态（黄蓝间距
    ≥0.015），而不是被饱和规则接走 —— 那是 test_servo 的职责。
    """
    return Observation(
        frame_size=(540, 960),
        bar=None if frac is None else BarEdge(y=frac * 540, contrast=20.0, y_int=1),
        spine=None if frac is None else SpineBottom(y=430.0, confidence=0.9, n_books=12),
        source="fake",
    )


def scripted_builder(fracs):
    """按脚本逐帧产观测的注入式 builder（签名同 build_backend）；耗尽后停在最后一帧"""
    seq = [make_obs(f) for f in fracs]

    def builder(cfg, servo, args):
        def observe():
            return seq.pop(0) if len(seq) > 1 else seq[0]

        return observe, (lambda d: None), "fake", {}, (lambda: None)

    return builder


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_dry_run_full_flow_confirmed_with_photo(tmp_path):
    """干跑全流程：闭环收敛→终验在带内→确认照落盘，exit 0。"""
    tool = make_tool()
    out = str(tmp_path / "run")
    rc = tool.main(["--dry-run", "--out", out])
    assert rc == 0
    assert os.path.exists(os.path.join(out, "perfect_shot.jpg"))

    with open(os.path.join(out, "final_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["type"] == "final_shot" and meta["shot"]["ok"] is True
    assert meta["verify"] is not None
    assert abs(meta["verify"]["err"]) <= 0.030

    lines = read_jsonl(os.path.join(out, "trace.jsonl"))
    assert lines[-2]["type"] == "result" and lines[-2]["confirmed"] is True
    iters = [ln for ln in lines if ln["type"] == "iter"]
    assert any(ln.get("note") == "verify" for ln in iters)     # 终验帧进了留痕
    assert lines[-1]["type"] == "final_shot" and lines[-1]["ok"] is True


def test_verify_out_of_band_reenters_then_confirms(tmp_path):
    """终验出带（中值盲区场景）→ 自动回到闭环再调，第二次终验通过，exit 0。

    hold_frames=3（configs 现值）：3 帧在带内收敛 → 终验帧 0.86 出带 →
    再来 3 帧收敛 → 终验 0.950 在带内。
    """
    tool = make_tool()
    out = str(tmp_path / "run")
    fracs = [0.956, 0.956, 0.956, 0.86, 0.956, 0.956, 0.956, 0.950]
    rc = tool.main(["--dry-run", "--out", out], builder=scripted_builder(fracs))
    assert rc == 0

    lines = read_jsonl(os.path.join(out, "trace.jsonl"))
    verifies = [ln for ln in lines
                if ln["type"] == "iter" and ln.get("note") == "verify"]
    assert len(verifies) == 2                    # 第一次出带 + 第二次通过
    assert verifies[0]["err"] < -0.03            # 第一次终验确实在带外
    assert abs(verifies[1]["err"]) <= 0.030


def test_never_confirmed_exit_1_no_photo(tmp_path):
    """始终看不到书架（丢失态）→ 如实报未到位，exit 1，不拍确认照。"""
    tool = make_tool()
    out = str(tmp_path / "run")
    rc = tool.main(["--dry-run", "--out", out], builder=scripted_builder([None]))
    assert rc == 1
    assert not os.path.exists(os.path.join(out, "perfect_shot.jpg"))
    lines = read_jsonl(os.path.join(out, "trace.jsonl"))
    assert [ln for ln in lines if ln["type"] == "result"][-1]["confirmed"] is False


def test_final_shot_disabled_still_exit_0(tmp_path):
    """关闭确认照（--set control.final_shot.enabled=false）：只做到位确认，exit 0。

    final_meta.json 仍会落盘（终验/位姿/交接是结果记录），只是 shot 为 None。
    """
    tool = make_tool()
    out = str(tmp_path / "run")
    rc = tool.main(["--dry-run", "--out", out,
                    "--set", "control.final_shot.enabled=false"])
    assert rc == 0
    assert not os.path.exists(os.path.join(out, "perfect_shot.jpg"))
    meta = read_meta(out)
    assert meta["shot"] is None and meta["verify"] is not None


def test_quality_fail_exit_3_photo_archived(tmp_path):
    """到位但确认照质检不达标 → exit 3，兜底帧与指标仍要存档。"""
    tool = make_tool()

    def blurry_still():
        import numpy as np
        h, w = 216, 384
        smooth = np.tile(np.linspace(0, 255, w).astype(np.uint8), (h, 1))
        frame = np.stack([smooth] * 3, axis=-1)
        return {"frame": frame, "ok": False,
                "metrics": {"laplacian_var": 0.0, "brightness": 127.5},
                "attempts": 3}

    out = str(tmp_path / "run")
    rc = tool.main(["--dry-run", "--out", out], still_fn=blurry_still)
    assert rc == 3
    assert os.path.exists(os.path.join(out, "perfect_shot.jpg"))   # 兜底帧存档
    with open(os.path.join(out, "final_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["shot"]["ok"] is False and meta["shot"]["attempts"] == 3


def test_capture_error_exit_3_error_recorded(tmp_path):
    """终拍设备异常（如 4K 流打不开）→ 不掩盖"已到位"事实，exit 3 且错误落账。"""
    tool = make_tool()

    def broken_still():
        raise RuntimeError("4K 流打不开")

    out = str(tmp_path / "run")
    rc = tool.main(["--dry-run", "--out", out], still_fn=broken_still)
    assert rc == 3
    with open(os.path.join(out, "final_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    assert "RuntimeError" in meta["capture_error"]
    lines = read_jsonl(os.path.join(out, "trace.jsonl"))
    assert lines[-2]["confirmed"] is True        # 位置到位的事实不被终拍失败掩盖
    assert lines[-1]["type"] == "final_shot"


# ========== 交接学长系统（S5 新增） ==========

class FakeHandoffClient:
    """HandoffClient 假件：记录调用，可指定失败"""

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def handoff(self, poi_key, pose, task_type, params, update_poi=True):
        self.calls.append({"poi_key": poi_key, "pose": pose,
                           "task_type": task_type, "params": params,
                           "update_poi": update_poi})
        if self.fail:
            raise RuntimeError("学长系统没开机")
        return {"poi_updated": bool(pose), "task_id": 7}


def read_meta(out):
    with open(os.path.join(out, "final_meta.json"), encoding="utf-8") as f:
        return json.load(f)


def test_handoff_runs_with_flag(tmp_path):
    """--handoff 强制开启：交接被调用、结果进 final_meta，exit 0。"""
    tool = make_tool()
    client = FakeHandoffClient()
    out = str(tmp_path / "run")
    rc = tool.main(["--dry-run", "--out", out, "--handoff"],
                   handoff_client=client)
    assert rc == 0
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["task_type"] == 1
    assert call["params"] == {"floor": 4, "area": "B", "rack": 94,
                              "side": "L", "column": 8}
    assert call["pose"] is None                  # 干跑后端无 mover，位姿如实为空
    meta = read_meta(out)
    assert meta["handoff"]["task_id"] == 7
    assert meta["handoff_error"] is None


def test_handoff_disabled_by_default(tmp_path):
    """默认关：交接客户端一次都不被调用。"""
    tool = make_tool()
    client = FakeHandoffClient()
    out = str(tmp_path / "run")
    rc = tool.main(["--dry-run", "--out", out], handoff_client=client)
    assert rc == 0
    assert client.calls == []
    assert read_meta(out)["handoff"] is None


def test_handoff_failure_keeps_exit_0(tmp_path):
    """交接失败不改退出码（到位结论已成立），但错误要落账。"""
    tool = make_tool()
    client = FakeHandoffClient(fail=True)
    out = str(tmp_path / "run")
    rc = tool.main(["--dry-run", "--out", out, "--handoff"],
                   handoff_client=client)
    assert rc == 0
    meta = read_meta(out)
    assert "RuntimeError" in meta["handoff_error"]
    lines = read_jsonl(os.path.join(out, "trace.jsonl"))
    assert lines[-1]["type"] == "final_shot"
    assert lines[-1]["handoff"] is None
    assert "RuntimeError" in lines[-1]["handoff_error"]


def test_handoff_skipped_when_quality_fails(tmp_path):
    """确认照质检未过：先留人看，不自动交接（exit 3 且交接零调用）。"""
    tool = make_tool()
    client = FakeHandoffClient()

    def blurry_still():
        import numpy as np
        h, w = 216, 384
        smooth = np.tile(np.linspace(0, 255, w).astype(np.uint8), (h, 1))
        frame = np.stack([smooth] * 3, axis=-1)
        return {"frame": frame, "ok": False,
                "metrics": {"laplacian_var": 0.0, "brightness": 127.5},
                "attempts": 3}

    out = str(tmp_path / "run")
    rc = tool.main(["--dry-run", "--out", out], still_fn=blurry_still,
                   handoff_client=client)
    assert rc == 3
    assert client.calls == []
    assert read_meta(out)["handoff"] is None


if __name__ == "__main__":
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
