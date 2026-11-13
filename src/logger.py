#!/usr/bin/env python3
import os
import sys
import librosa
from typing import Dict
from ffmpeg_runtime import FFMPEG_BIN_DIR, FFMPEG_EXE, FFPROBE_EXE, get_runtime_info

# ============================================================================
# PATHS & DIRECTORIES
# ============================================================================

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)

# Add src to sys.path for easier imports
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

PORTABLE_PYTHON_DIR = os.path.join(ROOT_DIR, 'bin', 'python-3.13.17-embed-amd64')

USING_PORTABLE_PYTHON = os.path.exists(PORTABLE_PYTHON_DIR)
FFMPEG_FOUND = os.path.exists(FFMPEG_EXE)

# ============================================================================
# SYSTEM DETECTION
# ============================================================================

def setup_environment():
    """Configure the portable Python environment."""
    if FFMPEG_BIN_DIR not in os.environ.get('PATH', '').split(os.pathsep):
        os.environ['PATH'] = FFMPEG_BIN_DIR + os.pathsep + os.environ.get('PATH', '')
    # Python Setup
    if USING_PORTABLE_PYTHON:
        if PORTABLE_PYTHON_DIR not in os.environ.get('PATH', ''):
            os.environ['PATH'] = PORTABLE_PYTHON_DIR + os.pathsep + os.environ.get('PATH', '')
        os.environ['PYTHONHOME'] = PORTABLE_PYTHON_DIR

    # Make direct CLI runs behave like run.bat (-X utf8 / PYTHONIOENCODING).
    # Without this, Windows cp1252 consoles can crash on existing status icons.
    for stream in (getattr(sys, 'stdout', None), getattr(sys, 'stderr', None)):
        try:
            if hasattr(stream, 'reconfigure'):
                stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

def get_gpu_info() -> Dict:
    """Detect a verified Taichi Vulkan compute device, with CPU fallback."""
    try:
        from taichi_backend import get_backend_info
        return get_backend_info()
    except Exception as exc:
        return {'available': False, 'name': 'No Vulkan GPU', 'backend': 'Taichi Vulkan',
                'vulkan_version': 'None', 'error': str(exc)}


def check_nvenc() -> bool:
    """Verify that the driver can encode with NVENC on this machine."""
    try:
        from video_encoders import get_encoder_support
        return get_encoder_support('h264_nvenc')['available']
    except Exception:
        return False

# ============================================================================
# LOGGING & OUTPUT
# ============================================================================

CONSOLE_SEPARATOR = "=" * 70

def print_startup_banner():
    """Print a comprehensive startup report to the console."""
    from gpu_cpu_utils import CPU_COUNT

    gpu = get_gpu_info()
    from gpu_devices import get_discrete_gpu
    selected = get_discrete_gpu()
    nvenc = check_nvenc()
    cpu_count = CPU_COUNT
    
    print(CONSOLE_SEPARATOR)
    print("[*] BeatSync Engine V2 (aka MusicVideoCutter V7)")
    print(CONSOLE_SEPARATOR)
    
    python_type = "Portable" if USING_PORTABLE_PYTHON else "System"
    print(f"   Python: {python_type} ({sys.version.split()[0]})")
    print(f"   Selected dGPU: {selected['name']}" + (f" (Vulkan device {selected['index']})" if selected['available'] else ''))
    
    if gpu['available']:
        print(f"   GPU analysis: {gpu['backend']} | Vulkan: {gpu['vulkan_version']}")
        print(f"   GPU: {gpu['name']}")
    else:
        print(f"   GPU analysis: CPU ({gpu.get('error', 'Vulkan unavailable')})")

    runtime = get_runtime_info()
    print(f"   FFmpeg: {runtime['version']} (portable full build) | NVENC: {'[OK]' if nvenc else '[NO]'}")
    print(f"   CPU Optimization: {cpu_count} threads detected")
    print(f"   Librosa: {librosa.__version__}")
    print(CONSOLE_SEPARATOR + "\n")

if __name__ == '__main__':
    setup_environment()
    print_startup_banner()
