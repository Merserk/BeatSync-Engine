#!/usr/bin/env python3
"""Shared CPU, Taichi Vulkan, and NVENC helpers."""

import multiprocessing
from logger import get_gpu_info, check_nvenc, setup_environment


CPU_COUNT = multiprocessing.cpu_count()
MAX_THREADS = CPU_COUNT
PARALLEL_WORKERS = max(CPU_COUNT // 2, 4)


def get_cpu_count() -> int:
    """Get the detected logical CPU count."""
    return CPU_COUNT


def get_max_threads() -> int:
    """Get the maximum FFmpeg thread count used per encode."""
    return MAX_THREADS


def get_parallel_workers() -> int:
    """Get the default number of parallel clip workers."""
    return PARALLEL_WORKERS


setup_environment()
GPU_INFO = get_gpu_info()
GPU_AVAILABLE = GPU_INFO['available']

if GPU_AVAILABLE:
    import taichi_backend as GPU_BACKEND
else:
    GPU_BACKEND = None

NVENC_AVAILABLE = check_nvenc()
USE_GPU = GPU_AVAILABLE


def is_gpu_enabled() -> bool:
    """Return whether GPU acceleration is currently enabled."""
    return USE_GPU and GPU_AVAILABLE


def set_gpu_mode(enabled: bool) -> bool:
    """Enable or disable GPU acceleration."""
    global USE_GPU
    if enabled and not GPU_AVAILABLE:
        print("⚠️  Taichi Vulkan is unavailable, falling back to CPU")
        USE_GPU = False
        return False

    USE_GPU = enabled
    if enabled:
        print(f"🚀 GPU acceleration ENABLED (using {GPU_INFO['backend']})")
    else:
        print(f"💻 Using CPU mode")

    return USE_GPU


def rhythm_band_sums(stft, frequencies):
    if GPU_BACKEND is None:
        raise RuntimeError("Taichi Vulkan is unavailable")
    return GPU_BACKEND.rhythm_band_sums(stft, frequencies)


def frame_statistics(frames):
    if GPU_BACKEND is None:
        raise RuntimeError("Taichi Vulkan is unavailable")
    return GPU_BACKEND.frame_statistics(frames)


def harmonic_percussive_audio(y):
    if GPU_BACKEND is None:
        raise RuntimeError('Taichi Vulkan is unavailable')
    return GPU_BACKEND.harmonic_percussive_audio(y)


def clear_gpu_memory() -> None:
    """Finish Vulkan work; per-call array buffers are released by Taichi."""
    if GPU_BACKEND is not None:
        GPU_BACKEND.synchronize()


def release_gpu_memory() -> None:
    """Finish work before releasing application-owned array references."""
    clear_gpu_memory()
