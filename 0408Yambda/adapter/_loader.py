"""按文件路径加载原 HSRL 模块，供 adapter 继承复用。"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


def hsrl_root() -> Path:
    adapter_root = Path(__file__).resolve().parent
    project_root = adapter_root.parent
    return Path(os.environ.get("HSRL_ROOT", project_root / "hsrl_core"))


def load_hsrl_module(relative_path: str, alias: str) -> ModuleType:
    """
    输入：
    - relative_path: 相对 HSRL root 的模块文件路径
    - alias: 加载到 sys.modules 的临时模块名

    输出：
    - 原 HSRL 模块对象
    """
    root = hsrl_root()
    if str(root) not in sys.path:
        sys.path.append(str(root))
    module_path = root / relative_path
    spec = importlib.util.spec_from_file_location(alias, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load HSRL module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module
