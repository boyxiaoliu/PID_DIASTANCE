# 图书馆机器人 PID 控距（新版）

给图书馆盘点机器人的 POI 拍照流程加**视觉距离闭环**：
机器人按地图到达点位并调整角度 → 拍第一张照片 → YOLO 识别**书本最低点**（作横杆搜索区的像素上界）
→ 以**下方横杆的像素位置**为控制量做 PID（离得多了向前、少了向后）→ 收敛后拍出"完美照片"
（构图基准：`4_B_097_L_8_1_4.jpg`——书脊充满画面、横杆条压在画面底部约 9 成处）。

- 老路线的完整背景与旧代码：`D:\Claude\01_图书馆项目\旧PID控距\`（只读参考）
- 学长原生产系统：`旧PID控距\项目背景\program_server_v3...\`（真机阶段接口对接，不拷贝代码）

## 双机分工（两台电脑各跑一个 AI 会话）

| | 线A 感知·数据线 | 线B 控制·真机线 |
|---|---|---|
| 电脑 | 本机 | 同伴机 |
| Owner（拍板人） | 线A Owner | 线B Owner |
| 阶段 | S1 误差标定、S2 感知评估 | S3 控制仿真、S4 上车适配、S5 真机试验 |
| 独占文件 | `perception.py`、`utils.py`、`tools/calibrate_target.py`、`tools/eval_perception.py` | `servo.py`、`tools/sim_servo.py`、真机适配层 |
| configs 认领 | `perception`、`target` 节 | `control`、`camera` 节 |

协作规则（文件归属、契约保护、提交纪律）见 `AGENTS.md`——**两个会话都必须遵守**。

## 目录结构

```
AGENTS.md            项目级协作规则（两线共同遵守）
configs/default.yaml 全部可调参数（分节认领，零硬编码）
src/pid_servo/
  observation.py     感知↔控制的数据契约（两线唯一接缝，改动需两人同意）
  perception.py      〔线A〕横杆检测 + 书本最低点来源抽象
  servo.py           〔线B〕PID + 闭环状态机
tools/               按阶段逐步建：calibrate_target / eval_perception / sim_servo
tests/               单元测试（提交前 pytest 必须通过）
experiments/         实验留痕（YYYY-MM-DD_主题/）
runs/                工具大文件输出（不入库）
```

## 数据路径（外部只读，不入库）

| 路径 | 内容 |
|---|---|
| `D:\Claude\01_图书馆项目\图书馆数据8_7\` | 740 张四件套（原图/检测JSON/OBB整图/段图），cam_3/4/5 各 ~247 张 |
| `D:\Library_Project\ori_img_save\` | 980 张 4K 原图 + 239 个人工标注（类别 15 书脊 / 16 上搁板 / 18 下搁板 / 19 标签） |
| `旧PID控距\项目背景\...\keep_data\complete_data\` | 学长系统推理结果 JSON |

## 快速开始

```bash
pip install -r requirements.txt
pytest          # 全绿才算环境就绪
```

## 当前进度

- [x] S0 骨架 + git
- [ ] S1〔线A〕误差信号标定（G1：目检候选图定 target）
- [ ] S2〔线A〕感知评估（G2）
- [ ] S3〔线B〕控制移植 + 仿真（G3）
- [ ] S4〔线B〕真机适配层（G4：安全清单）
- [ ] S5〔线B〕真机单点闭环试验（G5）
