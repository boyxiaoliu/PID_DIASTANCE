"""配置读写与小工具 〔线A 维护〕

约定：配置文件是 configs/default.yaml（项目根相对）；所有可调参数进配置，
代码出现魔法数字必须带 # TODO(calib) 与来源说明。
"""

import os

import yaml

#: 项目根 = src/pid_servo 的上两级
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: 默认配置文件路径
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "default.yaml")


def load(path=None):
    """
    load 读取 YAML 配置

    :param path: 配置文件路径，None 用 configs/default.yaml
    :return: 配置字典；文件不存在时抛 FileNotFoundError（配置缺失不该静默）
    """
    path = path or DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get(d, dotted, default=None):
    """
    get 按点路径取嵌套字典项，任一层缺失返回默认值

    :param d: 配置字典
    :param dotted: 点路径，如 "perception.bar.roi_x_fracs"
    :param default: 缺失时的默认值
    :return: 取到的值或默认值
    """
    cur = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
