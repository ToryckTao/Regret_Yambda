"""把 0408Yambda/adapter 放到本地 hsrl_core 前面。"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def install_hsrl_adapter(hsrl_root: str | None = None) -> None:
    """
    输入：
    - hsrl_root: 可选 HSRL core 目录

    输出：
    - 修改 sys.path，使 adapter 覆盖本地 hsrl_core 的少量模块
    """
    adapter_root = Path(__file__).resolve().parent
    project_root = adapter_root.parent
    default_core = project_root / "hsrl_core"
    hsrl_path = Path(hsrl_root or os.environ.get("HSRL_ROOT", default_core))

    if not hsrl_path.exists():
        raise FileNotFoundError(f"HSRL source root not found: {hsrl_path}")

    # 先插 hsrl_core，再插 adapter，最终 sys.path[0] 是 adapter。
    for path in [hsrl_path, adapter_root]:
        path_str = str(path)
        if path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)
