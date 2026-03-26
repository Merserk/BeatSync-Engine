#!/usr/bin/env python3
"""
Portable runtime discovery helpers.

This module centralizes portable Python/CUDA environment setup so the app does
not depend on a hard-coded CUDA version folder name.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from pathlib import Path

CUDA_OVERRIDE_ENV = "BEATSYNC_CUDA_DIR"
PORTABLE_PYTHON_DIRNAME = "python-3.13.9-embed-amd64"
CUDA_VERSION_PATTERN = re.compile(r"^v(\d+)(?:\.(\d+))?(?:\.(\d+))?$", re.IGNORECASE)


@dataclass(frozen=True)
class PortableCudaCandidate:
    path: str
    name: str
    version: str
    version_key: tuple[int, int, int]
    major: str


@dataclass(frozen=True)
class PortableRuntimeConfig:
    script_dir: str
    portable_python_dir: str | None
    portable_python_exe: str | None
    using_portable_python: bool
    portable_cuda_dir: str | None
    portable_cuda_bin: str | None
    portable_cuda_lib: str | None
    portable_cuda_version: str | None
    using_portable_cuda: bool
    cupy_package_name: str | None
    cupy_cuda_major: str | None
    cuda_selection_source: str
    cuda_notice: str | None
    cuda_candidates: tuple[str, ...]

    @property
    def python_runtime_label(self) -> str:
        if self.using_portable_python and self.portable_python_exe:
            return f"Portable ({self.portable_python_exe})"
        return "System Python"

    @property
    def python_status_label(self) -> str:
        if self.using_portable_python and self.portable_python_dir:
            relative_dir = os.path.relpath(self.portable_python_dir, self.script_dir).replace("\\", "/")
            return f"Portable ({relative_dir}/)"
        return "System Python"

    @property
    def cuda_runtime_label(self) -> str:
        if self.using_portable_cuda and self.portable_cuda_dir:
            relative_dir = os.path.relpath(self.portable_cuda_dir, self.script_dir).replace("\\", "/")
            return f"Portable ({relative_dir})"
        return "System (or not available)"

    @property
    def cuda_short_label(self) -> str:
        if self.using_portable_cuda and self.portable_cuda_version:
            return f"Portable CUDA {self.portable_cuda_version}"
        if self.using_portable_cuda:
            return "Portable CUDA"
        return "System CUDA"


def _prepend_env_path(entry: str) -> None:
    """Prepend a path entry once without duplicating it in PATH-like vars."""
    if not entry:
        return
    current = os.environ.get("PATH", "")
    path_items = current.split(os.pathsep) if current else []
    if entry not in path_items:
        os.environ["PATH"] = entry + os.pathsep + current if current else entry


def _prepend_ld_library_path(entry: str) -> None:
    """Maintain LD_LIBRARY_PATH for environments that inspect it."""
    if not entry:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    path_items = current.split(os.pathsep) if current else []
    if entry not in path_items:
        os.environ["LD_LIBRARY_PATH"] = entry + os.pathsep + current if current else entry


def _resolve_cuda_bin_dir(path: Path) -> Path | None:
    """Support both toolkit layouts: bin/ and bin/x64/."""
    for candidate in (path / "bin" / "x64", path / "bin"):
        if candidate.is_dir():
            return candidate
    return None


def _resolve_cuda_lib_dir(path: Path) -> Path | None:
    """Support both toolkit layouts: lib/x64/ and lib/."""
    for candidate in (path / "lib" / "x64", path / "lib"):
        if candidate.is_dir():
            return candidate
    return None


def _is_valid_cuda_toolkit_dir(path: Path) -> bool:
    """A minimal portable CUDA folder needs runtime DLL and import library roots."""
    return _resolve_cuda_bin_dir(path) is not None and _resolve_cuda_lib_dir(path) is not None


def _parse_version_key(name: str) -> tuple[int, int, int] | None:
    match = CUDA_VERSION_PATTERN.match(name)
    if not match:
        return None
    parts = [int(group) if group is not None else 0 for group in match.groups()]
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def _read_cuda_version(path: Path) -> str | None:
    version_file = path / "version.json"
    if not version_file.is_file():
        return None
    try:
        data = json.loads(version_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    cuda_info = data.get("cuda") if isinstance(data, dict) else None
    if not isinstance(cuda_info, dict):
        return None
    version = cuda_info.get("version")
    return str(version).strip() if version else None


def _candidate_from_path(path: Path) -> PortableCudaCandidate | None:
    if not _is_valid_cuda_toolkit_dir(path):
        return None
    version_key = _parse_version_key(path.name)
    version = _read_cuda_version(path)
    if version_key is None:
        if version:
            version_key = tuple(int(part) for part in (version.split(".") + ["0", "0"])[:3])
            major = version.split(".", 1)[0]
        else:
            version_key = (0, 0, 0)
            major = ""
    else:
        major = str(version_key[0])
    if not version:
        version = path.name[1:] if path.name.lower().startswith("v") else path.name
    return PortableCudaCandidate(
        path=str(path),
        name=path.name,
        version=version,
        version_key=version_key,
        major=major,
    )


def _discover_cuda_candidates(cuda_root: Path) -> list[PortableCudaCandidate]:
    if not cuda_root.is_dir():
        return []
    candidates: list[PortableCudaCandidate] = []
    for child in cuda_root.iterdir():
        if not child.is_dir():
            continue
        if _parse_version_key(child.name) is None:
            continue
        candidate = _candidate_from_path(child)
        if candidate is not None:
            candidates.append(candidate)
    return sorted(candidates, key=lambda item: item.version_key, reverse=True)


def _detect_cupy_cuda_line() -> tuple[str | None, str | None, str | None]:
    """Return (package_name, cuda_major, notice)."""
    discovered: list[tuple[str, str]] = []
    for package_name, cuda_major in (("cupy-cuda12x", "12"), ("cupy-cuda13x", "13")):
        try:
            metadata.version(package_name)
        except metadata.PackageNotFoundError:
            continue
        discovered.append((package_name, cuda_major))

    if len(discovered) == 1:
        package_name, cuda_major = discovered[0]
        return package_name, cuda_major, None
    if len(discovered) > 1:
        package_names = ", ".join(name for name, _ in discovered)
        return None, None, f"Multiple CuPy CUDA packages are installed: {package_names}"
    return None, None, None


def _resolve_portable_cuda(script_dir: str) -> tuple[str | None, str | None, str | None, str]:
    cuda_root = Path(script_dir) / "bin" / "CUDA"
    cupy_package_name, cupy_cuda_major, cupy_notice = _detect_cupy_cuda_line()
    notices: list[str] = [cupy_notice] if cupy_notice else []

    override_dir = os.environ.get(CUDA_OVERRIDE_ENV)
    if override_dir:
        override_path = Path(os.path.expandvars(os.path.expanduser(override_dir))).resolve()
        override_candidate = _candidate_from_path(override_path)
        if override_candidate is not None:
            return (
                override_candidate.path,
                override_candidate.version,
                "override",
                " ".join(notices).strip(),
            )
        notices.append(f"{CUDA_OVERRIDE_ENV} ignored because it is not a valid CUDA toolkit: {override_path}")

    candidates = _discover_cuda_candidates(cuda_root)

    if cupy_cuda_major:
        matching = [candidate for candidate in candidates if candidate.major == cupy_cuda_major]
        if matching:
            selected = matching[0]
            return (
                selected.path,
                selected.version,
                f"{cupy_package_name}",
                " ".join(notices).strip(),
            )

    if len(candidates) == 1:
        selected = candidates[0]
        if cupy_cuda_major and selected.major != cupy_cuda_major:
            notices.append(
                f"Only portable CUDA folder found ({selected.name}), but it does not match installed {cupy_package_name}"
            )
        return selected.path, selected.version, "single-candidate", " ".join(notices).strip()

    candidate_names = ", ".join(candidate.name for candidate in candidates) if candidates else "(none)"
    if candidates and cupy_cuda_major:
        notices.append(f"No portable CUDA folder matched installed {cupy_package_name}; available folders: {candidate_names}")
    elif candidates:
        notices.append(f"Multiple portable CUDA folders found under {cuda_root}: {candidate_names}")
    else:
        notices.append(f"No valid portable CUDA folders found under {cuda_root}")

    return None, None, "unavailable", " ".join(notices).strip()


@lru_cache(maxsize=None)
def configure_portable_runtime(script_dir: str) -> PortableRuntimeConfig:
    """
    Discover and configure portable Python and CUDA paths for the current repo.

    The result is cached per-script directory so multiple imports do not repeat
    environment mutation or filesystem scans.
    """
    script_dir = os.path.abspath(script_dir)

    portable_python_dir = os.path.join(script_dir, "bin", PORTABLE_PYTHON_DIRNAME)
    portable_python_exe = os.path.join(portable_python_dir, "python.exe")
    using_portable_python = os.path.exists(portable_python_exe)
    if using_portable_python:
        _prepend_env_path(portable_python_dir)
        os.environ["PYTHONHOME"] = portable_python_dir
    else:
        portable_python_dir = None
        portable_python_exe = None

    portable_cuda_dir, portable_cuda_version, cuda_selection_source, cuda_notice = _resolve_portable_cuda(script_dir)
    using_portable_cuda = portable_cuda_dir is not None
    cuda_bin_dir = _resolve_cuda_bin_dir(Path(portable_cuda_dir)) if portable_cuda_dir else None
    cuda_lib_dir = _resolve_cuda_lib_dir(Path(portable_cuda_dir)) if portable_cuda_dir else None
    portable_cuda_bin = str(cuda_bin_dir) if cuda_bin_dir else None
    portable_cuda_lib = str(cuda_lib_dir) if cuda_lib_dir else None
    if using_portable_cuda and portable_cuda_dir and portable_cuda_bin and portable_cuda_lib:
        os.environ["CUDA_PATH"] = portable_cuda_dir
        os.environ["CUDA_HOME"] = portable_cuda_dir
        os.environ["CUDA_ROOT"] = portable_cuda_dir
        _prepend_env_path(portable_cuda_bin)
        _prepend_env_path(portable_cuda_lib)
        _prepend_ld_library_path(portable_cuda_lib)

    cupy_package_name, cupy_cuda_major, _ = _detect_cupy_cuda_line()
    candidates = tuple(candidate.name for candidate in _discover_cuda_candidates(Path(script_dir) / "bin" / "CUDA"))

    return PortableRuntimeConfig(
        script_dir=script_dir,
        portable_python_dir=portable_python_dir,
        portable_python_exe=portable_python_exe,
        using_portable_python=using_portable_python,
        portable_cuda_dir=portable_cuda_dir,
        portable_cuda_bin=portable_cuda_bin,
        portable_cuda_lib=portable_cuda_lib,
        portable_cuda_version=portable_cuda_version,
        using_portable_cuda=using_portable_cuda,
        cupy_package_name=cupy_package_name,
        cupy_cuda_major=cupy_cuda_major,
        cuda_selection_source=cuda_selection_source,
        cuda_notice=cuda_notice or None,
        cuda_candidates=candidates,
    )
