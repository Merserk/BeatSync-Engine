#!/usr/bin/env python3
"""Runtime dependency compatibility checks for BeatSync Engine."""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Tuple


def check_numba_numpy_compat() -> Tuple[bool, str]:
    """Return (is_compatible, message)."""
    try:
        import numpy as np  # noqa: F401
    except Exception as exc:
        return False, f"NumPy 导入失败：{exc}"

    numpy_version = np.__version__

    try:
        import numba  # noqa: F401
        numba_version = numba.__version__
        return True, f"NumPy {numpy_version} / Numba {numba_version} 兼容"
    except Exception as exc:
        msg = str(exc)
        if "Numba needs NumPy 2.2 or less" in msg:
            return False, f"检测到版本冲突：{msg}"
        return False, f"Numba 导入失败：{msg}"


def fix_runtime_deps() -> int:
    """Install pinned dependency versions via current Python executable."""
    cmds = [
        [sys.executable, "-X", "utf8", "-m", "pip", "install", "--upgrade", "pip"],
        [sys.executable, "-X", "utf8", "-m", "pip", "install", "-r", "requirements.txt"],
    ]
    for cmd in cmds:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            return result.returncode
    return 0


def print_runtime_dependency_status() -> None:
    """Console-only status output; does not mutate environment."""
    ok, message = check_numba_numpy_compat()
    if ok:
        print(f"✅ 依赖检查通过：{message}")
    else:
        print(f"⚠️ 依赖检查警告：{message}")
        print("   如启动失败，请先运行 repair_env.bat 修复依赖。")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="check NumPy/Numba compatibility")
    parser.add_argument("--fix", action="store_true", help="install pinned runtime dependencies")
    args = parser.parse_args()

    if args.fix:
        return fix_runtime_deps()

    ok, message = check_numba_numpy_compat()
    if args.check:
        print(message)
        return 0 if ok else 1

    print_runtime_dependency_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
