"""覆盖原 HSRL BaseModel 的 checkpoint 加载兼容性。"""

from __future__ import annotations

import torch

from _loader import load_hsrl_module

_orig = load_hsrl_module("model/general.py", "_hsrl_original_model_general")

for _name, _value in vars(_orig).items():
    if not _name.startswith("__"):
        globals()[_name] = _value


class BaseModel(_orig.BaseModel):
    """兼容 PyTorch 2.6+ 默认 weights_only=True 的 checkpoint 加载。"""

    def load_from_checkpoint(self, model_path, with_optimizer=True):
        print("Load (checkpoint) from " + model_path + ".checkpoint")
        try:
            checkpoint = torch.load(model_path + ".checkpoint", map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path + ".checkpoint", map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        if with_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model_path = model_path
