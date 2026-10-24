"""交接学长系统〔线B 维护〕—— 对齐完成后把活儿交给学长的盘点流水线

交接两步（都用学长系统现成的接口，不改他的任何代码）：

    1. 改 POI（底盘 REST，与 mover.py 同一个 chassis_url）：
       把 poi_key 的坐标改成我们对齐确认后的位姿。学长的 ia.move 是从底盘
       POI 库取坐标的（chassis_control.py move()），改完之后他再 move(poi_key)
       就是原地瞬达，不会把对齐好的车又开走 —— 这是"到位交接"的关键。
    2. 派任务（学长 Flask API，configs control.handoff.api_url）：
       POST /api/connect 后 POST /api/task，task_type=1（指定列盘点）正好
       对应单个点位。他的 GlobalControl 轮询到任务 → 逐 POI 拍照 → 异步推理
       匹配，与 workflow/任务执行流程.txt 描述一致。

⚠️ 现场注意（学长代码的行为，非本模块能控制）：
    ia.move 里的 POI 坐标有进程内缓存（self.pois，首次 move 时从底盘全量
    加载后不再刷新）。若学长系统已带着旧缓存运行，本模块改的 POI 对他不
    生效 —— 现场交接前先重启学长系统，或与学长商量在 move() 里去掉缓存。
    另 PUT POI 的路径形态（/{uuid}）上车首跑时先 GET 对比核对一遍。

连接注入：chassis/senior 两个 HTTP 件与 mover.ChassisHttp 同形（单测注入假件），
requests 惰性导入由 ChassisHttp 负责，PC 单测不装 requests 也能全绿。
"""

# ---- 底盘 artifact REST（与学长 chassis_control.py API_Url 逐字一致）----

PATH_POIS = "/api/core/artifact/v1/pois"          # GET 列表 / PUT 更新单个

# ---- 学长 Flask API（lr_restful_api.py，API 接口文档.md 同源）----

SENIOR_CONNECT = "/api/connect"                   # POST，提交任务前必须先连接
SENIOR_TASK = "/api/task"                         # POST {"task_type", "params"}


class HandoffError(RuntimeError):
    """交接失败（POI 找不到 / 更新被拒 / 学长 API 报错）——调用方如实记录"""


class HandoffClient:
    """HandoffClient 交接客户端：改底盘 POI + 向学长系统派盘点任务

    :param chassis: ChassisHttp 同形件，指向底盘（artifact API）
    :param senior: ChassisHttp 同形件，指向学长 Flask API
    """

    def __init__(self, chassis, senior):
        self.chassis = chassis
        self.senior = senior

    # ---- 第 1 步：改 POI ----

    def find_poi(self, poi_key):
        """find_poi 按 display_name 在底盘 POI 库里找点位

        :param poi_key: POI 的 display_name（学长 ia.move 用的键，如 "4_B_094_L_8_1"）
        :return: 底盘返回的 POI 字典（含 uuid/id、pose、metadata）
        :raises HandoffError: 底盘响应不合式或找不到该 POI
        """
        resp = self.chassis.get(PATH_POIS)
        status = getattr(resp, "status_code", None)
        try:
            pois = resp.json()
        except Exception as e:
            raise HandoffError(f"get_pois 响应不是 JSON（status={status}，{e}）")
        if status is not None and status != 200 or not isinstance(pois, list):
            raise HandoffError(f"get_pois 失败（status={status}）：{pois!r}")
        for poi in pois:
            if isinstance(poi, dict) and \
                    poi.get("metadata", {}).get("display_name") == poi_key:
                return poi
        raise HandoffError(
            f"底盘 POI 库里没有 display_name={poi_key!r} 的点位"
            f"（共 {len(pois)} 个，先在底盘端确认名字）")

    def update_poi_pose(self, poi_key, pose):
        """update_poi_pose 把指定 POI 的坐标改成对齐后的位姿

        只动 pose 的 x/y/yaw，其余字段（uuid、metadata 等）原样保留 ——
        那是底盘与学长系统识别这个点位的身份信息。

        :param poi_key: POI display_name
        :param pose: {"x", "y", "yaw"}（米/弧度）
        :return: 更新后的 POI 字典
        :raises HandoffError: 找不到点位或底盘拒绝更新
        """
        poi = self.find_poi(poi_key)
        poi_id = poi.get("uuid") or poi.get("id")
        if not poi_id:
            raise HandoffError(f"POI {poi_key!r} 缺少 uuid/id，无法更新（学长 "
                               f"update_poi 同样要求，见其 chassis_control.py）")

        new_poi = dict(poi)
        new_pose = dict(poi.get("pose") or {})    # 保留 z 等原字段
        new_pose.update(x=float(pose["x"]), y=float(pose["y"]),
                        yaw=float(pose["yaw"]))
        new_poi["pose"] = new_pose

        resp = self.chassis.put_json(f"{PATH_POIS}/{poi_id}", new_poi)
        status = getattr(resp, "status_code", None)
        if status != 200:
            raise HandoffError(f"更新 POI {poi_key!r} 被拒绝（status={status}，"
                               f"body={getattr(resp, 'text', '')!r}）——"
                               f"TODO(S5上板) 首跑先 GET 核对 PUT 路径形态")
        return new_poi

    # ---- 第 2 步：派任务 ----

    def submit_task(self, task_type, params):
        """submit_task 向学长系统提交盘点任务（先 connect 再 task，接口文档要求）

        :param task_type: 任务类型（1=指定列盘点，见学长 lr_restful_api.py）
        :param params: 任务参数字典，type=1 需 {floor, area, rack, side, column}
        :return: task_id
        :raises HandoffError: 连接或提交被拒
        """
        resp = self.senior.post_json(SENIOR_CONNECT, {})
        data = self._senior_json(resp, "connect")
        if not data.get("success", False):
            raise HandoffError(f"connect 学长系统失败：{data!r}")

        resp = self.senior.post_json(SENIOR_TASK,
                                     {"task_type": task_type, "params": params})
        data = self._senior_json(resp, "submit_task")
        if not data.get("success", False):
            raise HandoffError(f"提交任务被拒（task_type={task_type}，"
                               f"params={params!r}）：{data!r}")
        task_id = data.get("task_id")
        if task_id is None:
            raise HandoffError(f"任务响应缺少 task_id：{data!r}")
        return task_id

    # ---- 两步编排 ----

    def handoff(self, poi_key, pose, task_type, params, update_poi=True):
        """handoff 完整交接：改 POI（可选）→ 派任务

        :param poi_key: POI display_name
        :param pose: 对齐确认后的位姿 {"x","y","yaw"}；None 时跳过改 POI
        :param task_type: 任务类型
        :param params: 任务参数
        :param update_poi: 是否执行改 POI 步骤
        :return: 留痕字典 {"poi_updated": bool, "task_id": ...}
        """
        poi_updated = False
        if update_poi:
            if pose is None:
                # 没有位姿（如干跑后端没有 mover）不拦任务，如实标注跳过
                poi_updated = None
            else:
                self.update_poi_pose(poi_key, pose)
                poi_updated = True
        task_id = self.submit_task(task_type, params)
        return {"poi_updated": poi_updated, "task_id": task_id}

    @staticmethod
    def _senior_json(resp, ctx):
        """_senior_json 校验学长 API 的 HTTP 状态与 JSON 可解析性"""
        status = getattr(resp, "status_code", None)
        try:
            data = resp.json()
        except Exception as e:
            raise HandoffError(f"学长 API {ctx} 响应不是 JSON（status={status}，{e}）")
        if status is not None and status != 200:
            raise HandoffError(f"学长 API {ctx} HTTP {status}：{data!r}")
        return data
