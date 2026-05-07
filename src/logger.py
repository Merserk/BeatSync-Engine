#!/usr/bin/env python3
import os
import sys
import subprocess
import librosa
from typing import Dict

# ============================================================================
# PATHS & DIRECTORIES
# ============================================================================

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)

# Add src to sys.path for easier imports
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

PORTABLE_CUDA_DIR = os.path.join(ROOT_DIR, 'bin', 'CUDA', 'v13.2')
PORTABLE_PYTHON_DIR = os.path.join(ROOT_DIR, 'bin', 'python-3.13.13-embed-amd64')
FFMPEG_BIN_DIR = os.path.join(ROOT_DIR, 'bin', 'ffmpeg')
FFMPEG_EXE = os.path.join(FFMPEG_BIN_DIR, 'ffmpeg.exe')

USING_PORTABLE_CUDA = os.path.exists(PORTABLE_CUDA_DIR)
USING_PORTABLE_PYTHON = os.path.exists(PORTABLE_PYTHON_DIR)
FFMPEG_FOUND = os.path.exists(FFMPEG_EXE)

# ============================================================================
# SYSTEM DETECTION
# ============================================================================

def setup_environment():
    """Configure portable environment variables for CUDA and Python."""
    # CUDA Setup
    if USING_PORTABLE_CUDA:
        os.environ['CUDA_PATH'] = PORTABLE_CUDA_DIR
        os.environ['CUDA_HOME'] = PORTABLE_CUDA_DIR
        os.environ['CUDA_ROOT'] = PORTABLE_CUDA_DIR
        
        cuda_bin = os.path.join(PORTABLE_CUDA_DIR, 'bin', 'x64')
        cuda_lib = os.path.join(PORTABLE_CUDA_DIR, 'lib', 'x64')
        
        # Add to PATH for DLL and Library discovery
        current_path = os.environ.get('PATH', '')
        for p in [cuda_bin, cuda_lib]:
            if os.path.exists(p) and p not in current_path:
                os.environ['PATH'] = p + os.pathsep + os.environ.get('PATH', '')
                
        # Linux compatibility (doesn't hurt on Windows)
        if 'LD_LIBRARY_PATH' in os.environ:
            os.environ['LD_LIBRARY_PATH'] = cuda_lib + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')
        else:
            os.environ['LD_LIBRARY_PATH'] = cuda_lib
    
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
    """Detect GPU hardware and CUDA runtime details via CuPy."""
    info = {
        'available': False,
        'name': 'No GPU',
        'cuda_version': 'None',
        'is_portable': USING_PORTABLE_CUDA
    }
    
    try:
        import cupy as cp
        info['available'] = True
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        info['name'] = props['name'].decode('utf-8')
        
        v = cp.cuda.runtime.runtimeGetVersion()
        # Format as Major.Minor (e.g., 13000 -> 13.0)
        info['cuda_version'] = f"{v // 1000}.{(v % 1000) // 10}"
    except ImportError:
        pass
    except Exception:
        info['name'] = 'Available (Detection Failed)'
        
    return info

def check_nvenc() -> bool:
    """Check if FFmpeg supports NVIDIA hardware encoding (NVENC)."""
    try:
        cmd = [FFMPEG_EXE if FFMPEG_FOUND else 'ffmpeg', '-encoders']
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return 'h264_nvenc' in result.stdout
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
    nvenc = check_nvenc()
    cpu_count = CPU_COUNT
    
    print(CONSOLE_SEPARATOR)
    print("[*] BeatSync Engine V2 (aka MusicVideoCutter V7)")
    print(CONSOLE_SEPARATOR)
    
    python_type = "Portable" if USING_PORTABLE_PYTHON else "System"
    print(f"   Python: {python_type} ({sys.version.split()[0]})")
    
    cuda_type = "Portable (v13.2)" if gpu['is_portable'] else "System"
    if gpu['available']:
        print(f"   CUDA: {cuda_type} | Runtime: {gpu['cuda_version']}")
        print(f"   GPU: {gpu['name']}")
    else:
        print(f"   CUDA: {cuda_type} | GPU: Not available (CPU only)")
        
    ffmpeg_type = "Portable" if FFMPEG_FOUND else "System"
    print(f"   FFmpeg: {ffmpeg_type} | NVENC: {'[OK]' if nvenc else '[NO]'}")
    print(f"   CPU Optimization: {cpu_count} threads detected")
    print(f"   Librosa: {librosa.__version__}")
    print(CONSOLE_SEPARATOR + "\n")

if __name__ == '__main__':
    setup_environment()
    print_startup_banner()
