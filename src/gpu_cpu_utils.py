#!/usr/bin/env python3
"""Shared CPU, GPU, CuPy, and NVENC helpers."""

import multiprocessing
import os

import numpy as np

from logger import get_gpu_info, check_nvenc, setup_environment, USING_PORTABLE_CUDA


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
os.environ.setdefault('CUPY_ACCELERATORS', 'cub')
GPU_INFO = get_gpu_info()
GPU_AVAILABLE = GPU_INFO['available']

if GPU_AVAILABLE:
    try:
        import cupy as cp
    except ImportError:
        cp = None
        GPU_AVAILABLE = False
else:
    cp = None

NVENC_AVAILABLE = check_nvenc()
USE_GPU = False


def is_gpu_enabled() -> bool:
    """Return whether GPU acceleration is currently enabled."""
    return USE_GPU and GPU_AVAILABLE


def set_gpu_mode(enabled: bool) -> bool:
    """Enable or disable GPU acceleration."""
    global USE_GPU
    if enabled and not GPU_AVAILABLE:
        print("⚠️  GPU requested but CuPy not available, falling back to CPU")
        USE_GPU = False
        return False

    USE_GPU = enabled
    if enabled:
        cuda_str = "Portable CUDA" if USING_PORTABLE_CUDA else "System CUDA"
        print(f"🚀 GPU acceleration ENABLED (using {cuda_str})")
    else:
        print(f"💻 Using CPU mode")

    return USE_GPU


def get_array_module(use_gpu: bool = None):
    """Get NumPy or CuPy based on GPU setting."""
    if use_gpu is None:
        use_gpu = USE_GPU
    return cp if (use_gpu and GPU_AVAILABLE) else np


def to_cpu(array):
    """Convert a CuPy array to a NumPy array if needed."""
    if GPU_AVAILABLE and cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return array


def to_gpu(array, use_gpu: bool = None):
    """Convert a NumPy array to a CuPy array if GPU mode is enabled."""
    if use_gpu is None:
        use_gpu = USE_GPU
    if use_gpu and GPU_AVAILABLE and cp is not None and isinstance(array, np.ndarray):
        return cp.asarray(array)
    return array


def clear_gpu_memory() -> None:
    """Synchronize GPU work while keeping CuPy's memory pool warm."""
    if GPU_AVAILABLE and cp is not None:
        cp.cuda.Stream.null.synchronize()
        if os.environ.get('BEATSYNC_RELEASE_GPU_MEMORY') == '1':
            release_gpu_memory()


def release_gpu_memory() -> None:
    """Release cached CuPy memory blocks back to the driver."""
    if GPU_AVAILABLE and cp is not None:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
