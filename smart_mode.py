#!/usr/bin/env python3
"""
Smart Mode - Multi-Band Intelligent Beat Detection
- Kick drum (20-150 Hz) + Clap/Snare (150-4000 Hz) + Hi-hat (4000+ Hz)
- Frequency-based cut selection with presets
- GPU acceleration support
"""

import os
import sys

# Determine script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set up portable CUDA path BEFORE importing any CUDA-dependent libraries
PORTABLE_CUDA_DIR = os.path.join(SCRIPT_DIR, 'bin', 'CUDA', 'v13.0')
PORTABLE_CUDA_BIN = os.path.join(PORTABLE_CUDA_DIR, 'bin', 'x64')
PORTABLE_CUDA_LIB = os.path.join(PORTABLE_CUDA_DIR, 'lib', 'x64')

# Check if portable CUDA exists and configure environment
if os.path.exists(PORTABLE_CUDA_DIR):
    USING_PORTABLE_CUDA = True
    
    # Set CUDA environment variables
    os.environ['CUDA_PATH'] = PORTABLE_CUDA_DIR
    os.environ['CUDA_HOME'] = PORTABLE_CUDA_DIR
    os.environ['CUDA_ROOT'] = PORTABLE_CUDA_DIR
    
    # Add CUDA bin to PATH (for DLLs)
    if PORTABLE_CUDA_BIN not in os.environ.get('PATH', ''):
        os.environ['PATH'] = PORTABLE_CUDA_BIN + os.pathsep + os.environ.get('PATH', '')
    
    # Add CUDA lib to PATH (for library files)
    if os.path.exists(PORTABLE_CUDA_LIB):
        if PORTABLE_CUDA_LIB not in os.environ.get('PATH', ''):
            os.environ['PATH'] = PORTABLE_CUDA_LIB + os.pathsep + os.environ.get('PATH', '')
    
    # Set library path for Linux/Unix compatibility
    if 'LD_LIBRARY_PATH' in os.environ:
        os.environ['LD_LIBRARY_PATH'] = PORTABLE_CUDA_LIB + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')
    else:
        os.environ['LD_LIBRARY_PATH'] = PORTABLE_CUDA_LIB
else:
    USING_PORTABLE_CUDA = False

# Set up portable Python path
PORTABLE_PYTHON_DIR = os.path.join(SCRIPT_DIR, 'bin', 'python-3.13.9-embed-amd64')
PORTABLE_PYTHON_EXE = os.path.join(PORTABLE_PYTHON_DIR, 'python.exe')

# Check if we're using portable Python
if os.path.exists(PORTABLE_PYTHON_EXE):
    USING_PORTABLE_PYTHON = True
    if PORTABLE_PYTHON_DIR not in os.environ.get('PATH', ''):
        os.environ['PATH'] = PORTABLE_PYTHON_DIR + os.pathsep + os.environ.get('PATH', '')
    os.environ['PYTHONHOME'] = PORTABLE_PYTHON_DIR
else:
    USING_PORTABLE_PYTHON = False

# NOW import other modules (after CUDA and Python environment is set)
import numpy as np
import librosa
from typing import Tuple, List, Dict
import warnings

# Import FFmpeg processing functions
from ffmpeg_processing import detect_video_scene_changes, get_video_duration

# Import UI content for presets
from ui_content import SMART_PRESETS_CONFIG as SMART_PRESETS

# GPU Support
try:
    import cupy as cp
    GPU_AVAILABLE = True
    
    cuda_info = "Portable CUDA (bin/CUDA/v13.0)" if USING_PORTABLE_CUDA else "System CUDA"
    python_info = "Portable Python" if USING_PORTABLE_PYTHON else "System Python"
    
    print(f"✅ CuPy detected - GPU acceleration available")
    print(f"   Using: {cuda_info}")
    print(f"   Python: {python_info}")
    
    try:
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        gpu_name = props['name'].decode('utf-8')
        cuda_version = cp.cuda.runtime.runtimeGetVersion()
        print(f"   GPU: {gpu_name}")
        print(f"   CUDA Runtime: {cuda_version}")
    except Exception as e:
        print(f"   GPU: Available (name detection failed)")
        print(f"   Error: {e}")
except ImportError:
    GPU_AVAILABLE = False
    cp = None
    if USING_PORTABLE_PYTHON:
        print(f"⚠️  CuPy not available - using CPU only (Portable Python)")
    else:
        print(f"⚠️  CuPy not available - using CPU only")

# Suppress warnings
warnings.filterwarnings('ignore')

# Global GPU state
USE_GPU = False


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
    """Convert CuPy array to NumPy array if needed."""
    if GPU_AVAILABLE and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return array


def to_gpu(array):
    """Convert NumPy array to CuPy array if GPU enabled."""
    if USE_GPU and GPU_AVAILABLE and isinstance(array, np.ndarray):
        return cp.asarray(array)
    return array


def analyze_beats_smart(audio_file: str, start_time: float = 0.0, end_time: float = None) -> Tuple[np.ndarray, Dict]:
    """
    Smart Mode - Multi-band frequency beat detection.
    Analyzes Kick (bass) + Clap/Snare (mid) + Hi-hat (high) frequencies.
    
    Args:
        audio_file: Path to audio file
        start_time: Start time in seconds
        end_time: End time in seconds (None = full duration)
    
    Returns:
        Tuple of (beat_times array, beat_info dictionary)
    """
    gpu_status = "🚀 GPU" if USE_GPU and GPU_AVAILABLE else "💻 CPU"
    cuda_status = "Portable CUDA" if USING_PORTABLE_CUDA else "System CUDA/None"
    python_status = "Portable Python" if USING_PORTABLE_PYTHON else "System Python"
    
    print(f"🧠 Smart Mode - Multi-band frequency detection")
    print(f"   Mode: {gpu_status}")
    print(f"   CUDA: {cuda_status}")
    print(f"   Python: {python_status}")
    
    duration = None
    if end_time and end_time > start_time:
        duration = end_time - start_time
    
    # Load audio (always on CPU via librosa)
    print(f"   🎵 Loading audio file...")
    y, sr = librosa.load(audio_file, sr=22050, offset=start_time, duration=duration, mono=True)
    
    # Get array module (NumPy or CuPy)
    xp = get_array_module()
    
    # Transfer to GPU if enabled
    if USE_GPU and GPU_AVAILABLE:
        print(f"   📤 Transferring audio data to GPU ({cuda_status})...")
        y_processed = to_gpu(y)
    else:
        y_processed = y
    
    # Compute STFT for frequency analysis
    print(f"   🔊 Computing STFT on {gpu_status}...")
    stft = librosa.stft(y, n_fft=2048, hop_length=512)
    
    # Transfer STFT to GPU if enabled
    if USE_GPU and GPU_AVAILABLE:
        stft = to_gpu(stft)
    
    # Frequency bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    if USE_GPU and GPU_AVAILABLE:
        freqs = to_gpu(freqs)
    
    # Define frequency bands for rhythm detection
    print(f"   📊 Analyzing rhythm frequency bands on {gpu_status}:")
    print(f"      🥁 Kick drum: 20-150 Hz (bass punch)")
    print(f"      👏 Clap/Snare: 150-4000 Hz (mid-range snap)")
    print(f"      🎵 Hi-hat: 4000+ Hz (high-frequency)")
    
    # Frequency band masks
    kick_band = (freqs >= 20) & (freqs <= 150)
    clap_band = (freqs >= 150) & (freqs <= 4000)
    hihat_band = freqs >= 4000
    
    # Create onset envelopes for each band (GPU accelerated)
    print(f"   ⚡ Extracting frequency bands...")
    kick_onset = xp.sum(xp.abs(stft[kick_band, :]), axis=0)
    clap_onset = xp.sum(xp.abs(stft[clap_band, :]), axis=0)
    hihat_onset = xp.sum(xp.abs(stft[hihat_band, :]), axis=0)
    
    # Normalize onset envelopes (GPU accelerated)
    kick_onset = kick_onset / (xp.max(kick_onset) + 1e-6)
    clap_onset = clap_onset / (xp.max(clap_onset) + 1e-6)
    hihat_onset = hihat_onset / (xp.max(hihat_onset) + 1e-6)
    
    # Combined onset with kick and clap priority
    print(f"   🎯 Computing combined onset envelope...")
    combined_onset = (kick_onset * 3.0) + (clap_onset * 2.0) + (hihat_onset * 0.5)
    combined_onset = combined_onset / (xp.max(combined_onset) + 1e-6)
    
    # Convert back to CPU for librosa beat tracking
    if USE_GPU and GPU_AVAILABLE:
        print(f"   📥 Transferring results back to CPU for beat tracking...")
        combined_onset_cpu = to_cpu(combined_onset)
    else:
        combined_onset_cpu = combined_onset
    
    # Detect ALL beats using combined onset (librosa requires CPU)
    print(f"   🥁 Detecting beats...")
    try:
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=combined_onset_cpu,
            sr=sr,
            units='frames',
            hop_length=512,
            start_bpm=120,
            tightness=100
        )
    except TypeError:
        print("   ⚠️  Fallback: Using standard beat tracking")
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames', hop_length=512)
    
    # Convert to timestamps
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)
    
    # Analyze strength of kick and clap at each beat
    beat_info = {
        'times': beat_times,
        'kick_strength': [],
        'clap_strength': [],
        'combined_strength': [],
        'tempo': tempo[0] if tempo.size > 0 else 120.0,
        'is_strong_kick': [],
        'is_strong_clap': [],
        'is_strong_beat': []
    }
    
    # Sample strength around each beat (GPU accelerated)
    print(f"   💪 Analyzing beat strengths on {gpu_status}...")
    for beat_frame in beat_frames:
        if beat_frame < len(kick_onset):
            start_idx = max(0, beat_frame - 3)
            end_idx = min(len(kick_onset), beat_frame + 4)
            
            kick_val = float(xp.max(kick_onset[start_idx:end_idx]))
            clap_val = float(xp.max(clap_onset[start_idx:end_idx]))
            combined_val = float(xp.mean(combined_onset[start_idx:end_idx]))
            
            beat_info['kick_strength'].append(kick_val)
            beat_info['clap_strength'].append(clap_val)
            beat_info['combined_strength'].append(combined_val)
        else:
            beat_info['kick_strength'].append(0.0)
            beat_info['clap_strength'].append(0.0)
            beat_info['combined_strength'].append(0.0)
    
    # Convert to numpy arrays (always return CPU arrays)
    beat_info['kick_strength'] = np.array(beat_info['kick_strength'])
    beat_info['clap_strength'] = np.array(beat_info['clap_strength'])
    beat_info['combined_strength'] = np.array(beat_info['combined_strength'])
    
    # Calculate thresholds for "strong" beats
    kick_threshold_50 = np.percentile(beat_info['kick_strength'], 50)
    clap_threshold_50 = np.percentile(beat_info['clap_strength'], 50)
    
    # Mark which beats are strong
    for i in range(len(beat_times)):
        is_kick = beat_info['kick_strength'][i] > kick_threshold_50
        is_clap = beat_info['clap_strength'][i] > clap_threshold_50
        
        beat_info['is_strong_kick'].append(is_kick)
        beat_info['is_strong_clap'].append(is_clap)
        beat_info['is_strong_beat'].append(is_kick or is_clap)
    
    # Convert to numpy arrays
    beat_info['is_strong_kick'] = np.array(beat_info['is_strong_kick'])
    beat_info['is_strong_clap'] = np.array(beat_info['is_strong_clap'])
    beat_info['is_strong_beat'] = np.array(beat_info['is_strong_beat'])
    
    # Statistics
    num_strong_kicks = np.sum(beat_info['is_strong_kick'])
    num_strong_claps = np.sum(beat_info['is_strong_clap'])
    num_strong_beats = np.sum(beat_info['is_strong_beat'])
    
    print(f"   ✓ Detected {len(beat_times)} total beats at {beat_info['tempo']:.1f} BPM")
    print(f"   ✓ Strong kicks: {num_strong_kicks} ({num_strong_kicks/len(beat_times)*100:.1f}%)")
    print(f"   ✓ Strong claps: {num_strong_claps} ({num_strong_claps/len(beat_times)*100:.1f}%)")
    print(f"   ✓ Strong beats (kick or clap): {num_strong_beats} ({num_strong_beats/len(beat_times)*100:.1f}%)")
    
    if USE_GPU and GPU_AVAILABLE:
        print(f"   ♻️  Clearing GPU memory...")
        cp.get_default_memory_pool().free_all_blocks()
    
    return beat_times, beat_info


def select_beats_smart(beat_info: Dict, preset: str = 'normal') -> np.ndarray:
    """
    Intelligently select beats based on preset frequency.
    
    Args:
        beat_info: Dictionary from analyze_beats_smart()
        preset: Preset name (slower/slow/normal/fast/faster)
    
    Returns:
        Array of selected beat times
    """
    if preset not in SMART_PRESETS:
        print(f"   ⚠️  Unknown preset '{preset}', using 'normal'")
        preset = 'normal'
    
    config = SMART_PRESETS[preset]
    print(f"\n🎯 Smart Beat Selection: {preset.upper()}")
    print(f"   {config['description']}")
    print(f"   Frequency mode: {config['cut_frequency']}")
    
    beat_times = beat_info['times']
    kick_strength = beat_info['kick_strength']
    clap_strength = beat_info['clap_strength']
    is_strong_kick = beat_info['is_strong_kick']
    is_strong_clap = beat_info['is_strong_clap']
    is_strong_beat = beat_info['is_strong_beat']
    
    kick_threshold = np.percentile(kick_strength, config['kick_threshold'])
    clap_threshold = np.percentile(clap_strength, config['clap_threshold'])
    
    selected_beats = []
    last_selected_time = -999.0
    
    strong_beat_indices = np.where(is_strong_beat)[0]
    
    if config['cut_frequency'] == 'every_4th_strong_beat':
        print(f"   → Selecting every 4th strong beat from {len(strong_beat_indices)} strong beats")
        for i, idx in enumerate(strong_beat_indices):
            if i % 4 == 0:
                time = beat_times[idx]
                if time - last_selected_time >= config['min_interval']:
                    selected_beats.append(time)
                    last_selected_time = time
    
    elif config['cut_frequency'] == 'every_2nd_strong_beat':
        print(f"   → Selecting every 2nd strong beat from {len(strong_beat_indices)} strong beats")
        for i, idx in enumerate(strong_beat_indices):
            if i % 2 == 0:
                time = beat_times[idx]
                if time - last_selected_time >= config['min_interval']:
                    selected_beats.append(time)
                    last_selected_time = time
    
    elif config['cut_frequency'] == 'every_strong_beat':
        print(f"   → Selecting all {len(strong_beat_indices)} strong beats")
        for idx in strong_beat_indices:
            time = beat_times[idx]
            if time - last_selected_time >= config['min_interval']:
                selected_beats.append(time)
                last_selected_time = time
    
    elif config['cut_frequency'] == 'all_beats_prioritize_strong':
        print(f"   → Selecting all beats ({len(beat_times)} total) with priority on strong")
        for i, time in enumerate(beat_times):
            if is_strong_beat[i]:
                if time - last_selected_time >= config['min_interval']:
                    selected_beats.append(time)
                    last_selected_time = time
            else:
                if time - last_selected_time >= config['min_interval'] * 1.5:
                    selected_beats.append(time)
                    last_selected_time = time
    
    elif config['cut_frequency'] == 'all_beats_plus_subdivisions':
        print(f"   → Selecting all beats + subdivisions on strong kicks")
        very_strong_kick_threshold = np.percentile(kick_strength, 80)
        
        for i, time in enumerate(beat_times):
            if time - last_selected_time >= config['min_interval']:
                selected_beats.append(time)
                last_selected_time = time
                
                # Add subdivision on very strong kicks
                if kick_strength[i] > very_strong_kick_threshold and i < len(beat_times) - 1:
                    next_time = beat_times[i + 1]
                    mid_time = (time + next_time) / 2
                    
                    if (mid_time - time) >= config['min_interval'] and (next_time - mid_time) >= config['min_interval']:
                        selected_beats.append(mid_time)
    
    selected_beats = np.array(selected_beats)
    
    print(f"   ✓ Selected {len(selected_beats)} cuts from {len(beat_times)} detected beats")
    if len(selected_beats) > 1:
        avg_interval = np.mean(np.diff(selected_beats))
        print(f"   ✓ Average cut interval: {avg_interval:.3f}s ({1/avg_interval:.2f} cuts/sec)")
    
    return selected_beats


def avoid_scene_cuts(beat_times: np.ndarray, scene_changes: List[float], 
                     avoid_window: float = 0.3) -> np.ndarray:
    """Remove beat times that are too close to video scene changes."""
    if len(scene_changes) == 0:
        return beat_times
    
    print(f"\n🎬 Avoiding {len(scene_changes)} scene cuts (±{avoid_window}s window)")
    
    filtered_beats = []
    removed_count = 0
    
    for beat_time in beat_times:
        too_close = False
        for scene_time in scene_changes:
            if abs(beat_time - scene_time) < avoid_window:
                too_close = True
                removed_count += 1
                break
        
        if not too_close:
            filtered_beats.append(beat_time)
    
    print(f"   ✓ Removed {removed_count} beats near scene changes")
    print(f"   ✓ Kept {len(filtered_beats)} beats")
    
    return np.array(filtered_beats)


def get_preset_info(preset: str) -> Dict:
    """Get information about a smart preset."""
    return SMART_PRESETS.get(preset, SMART_PRESETS['normal'])


def list_presets() -> List[str]:
    """Get list of available presets."""
    return list(SMART_PRESETS.keys())


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return GPU_AVAILABLE


def get_gpu_info() -> str:
    """Get GPU information string."""
    if GPU_AVAILABLE and cp is not None:
        try:
            device = cp.cuda.Device()
            props = cp.cuda.runtime.getDeviceProperties(device.id)
            gpu_name = props['name'].decode('utf-8')
            cuda_version = cp.cuda.runtime.runtimeGetVersion()
            cuda_source = "Portable CUDA 13.0" if USING_PORTABLE_CUDA else f"CUDA {cuda_version}"
            return f"{gpu_name} ({cuda_source})"
        except Exception as e:
            cuda_source = "Portable CUDA" if USING_PORTABLE_CUDA else "System CUDA"
            return f"GPU Available ({cuda_source})"
    return "No GPU"


# Re-export detect_video_scene_changes from ffmpeg_processing
__all__ = [
    'analyze_beats_smart',
    'select_beats_smart',
    'avoid_scene_cuts',
    'detect_video_scene_changes',
    'get_preset_info',
    'list_presets',
    'set_gpu_mode',
    'is_gpu_available',
    'get_gpu_info'
]