"""覆盖原 HSRL utils 的少量兼容修正。"""

from __future__ import annotations

from _loader import load_hsrl_module

_orig = load_hsrl_module("utils.py", "_hsrl_original_utils")

for _name, _value in vars(_orig).items():
    if not _name.startswith("__"):
        globals()[_name] = _value


class LinearScheduler(_orig.LinearScheduler):
    """修正 schedule_timesteps=0 时的除零问题。"""

    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        super().__init__(max(int(schedule_timesteps), 1), final_p, initial_p)


class SinScheduler(_orig.SinScheduler):
    """修正 schedule_timesteps=0 时的除零问题。"""

    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        super().__init__(max(int(schedule_timesteps), 1), final_p, initial_p)
