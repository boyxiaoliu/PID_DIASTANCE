"""pytest 共享配置：把 src 加进 import 路径"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
