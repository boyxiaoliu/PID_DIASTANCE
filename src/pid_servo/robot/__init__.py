"""真机适配层〔线B 维护〕—— 把 DistanceServo 的 observe()/move() 接到真车

后端形状与 shelf_sim 保持一致（项目注入式约定）：
    observe() -> Observation   真机 = 相机帧 + RKNN 书脊 + 线A perception 横杆
    move(delta_m) -> pose      真机 = mover.Mover（底盘增量位移原语）
控制代码 servo.py 对二者的真实来源无感知——仿真跑的代码 = 上车跑的代码。

板端依赖（rknn / v4l / requests）一律惰性导入：PC 上只跑 mock 单测，不装也能全绿。
"""
