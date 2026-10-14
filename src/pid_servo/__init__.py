"""pid_servo —— 视觉距离闭环包

两线唯一接缝在 observation.py（契约）；线A 的 perception.py 产出 Observation，
线B 的 servo.py 消费 Observation。符号约定见包内模块 docstring 与 AGENTS.md §6。
"""

from .observation import BarEdge, Observation, SpineBottom

__all__ = ["Observation", "BarEdge", "SpineBottom"]
