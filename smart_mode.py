#!/usr/bin/env python3
"""
Smart Mode - Multi-Band Intelligent Beat Detection
- Kick drum (20-150 Hz) + Clap/Snare (150-4000 Hz) + Hi-hat (4000+ Hz)
- Frequency-based cut selection with presets
- GPU acceleration support
"""

import os
from typing import Dict, List, Tuple
import warnings

import librosa
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from runtime_env import configure_portable_runtime

RUNTIME = configure_portable_runtime(SCRIPT_DIR)
PORTABLE_CUDA_DIR = RUNTIME.portable_cuda_dir
USING_PORTABLE_CUDA = RUNTIME.using_portable_cuda
PORTABLE_PYTHON_DIR = RUNTIME.portable_python_dir
PORTABLE_PYTHON_EXE = RUNTIME.portable_python_exe
USING_PORTABLE_PYTHON = RUNTIME.using_portable_python

from ffmpeg_processing import detect_video_scene_changes
from ui_content import SMART_PRESETS_CONFIG as SMART_PRESETS

GPU_INIT_ERROR = None

if RUNTIME.cuda_notice:
    print(f"Portable CUDA note: {RUNTIME.cuda_notice}")

try:
    import cupy as cp

    GPU_AVAILABLE = True
    cuda_info = RUNTIME.cuda_runtime_label if USING_PORTABLE_CUDA else "System CUDA"
    python_info = "Portable Python" if USING_PORTABLE_PYTHON else "System Python"
    print("CuPy detected - GPU acceleration available")
    print(f"   Using: {cuda_info}")
    print(f"   Python: {python_info}")

    try:
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count < 1:
            raise RuntimeError("No CUDA devices detected")
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        gpu_name = props["name"].decode("utf-8")
        cuda_version = cp.cuda.runtime.runtimeGetVersion()
        print(f"   GPU: {gpu_name}")
        print(f"   CUDA Runtime: {cuda_version}")
    except Exception as exc:
        GPU_AVAILABLE = False
        GPU_INIT_ERROR = str(exc)
        print("CuPy detected, but CUDA runtime is not usable - falling back to CPU")
        print(f"   Using: {cuda_info}")
        print(f"   Python: {python_info}")
        print(f"   Error: {GPU_INIT_ERROR}")
except ImportError:
    GPU_AVAILABLE = False
    cp = None
    if USING_PORTABLE_PYTHON:
        print("CuPy not available - using CPU only (Portable Python)")
    else:
        print("CuPy not available - using CPU only")

if GPU_AVAILABLE and cp is not None:
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count < 1:
            raise RuntimeError("No CUDA devices detected")
    except Exception as exc:
        GPU_AVAILABLE = False
        GPU_INIT_ERROR = str(exc)
        print("CuPy detected, but CUDA runtime is not usable - falling back to CPU")
        print(f"   Error: {GPU_INIT_ERROR}")

warnings.filterwarnings("ignore")

USE_GPU = False


def _summarize_exception(exc: Exception) -> str:
    """Keep optional GPU fallback errors concise in the console."""
    message = str(exc).strip()
    return message.splitlines()[0] if message else exc.__class__.__name__


def _clear_cupy_memory() -> None:
    """Release any cached CuPy allocations after optional GPU work."""
    if cp is None:
        return
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def set_gpu_mode(enabled: bool) -> bool:
    """Enable or disable GPU acceleration."""
    global USE_GPU
    if enabled and not GPU_AVAILABLE:
        print("GPU requested but CuPy not available, falling back to CPU")
        USE_GPU = False
        return False
    USE_GPU = enabled
    if enabled:
        cuda_str = "Portable CUDA" if USING_PORTABLE_CUDA else "System CUDA"
        print(f"GPU acceleration ENABLED (using {cuda_str})")
    else:
        print("Using CPU mode")
    return USE_GPU


def get_array_module(use_gpu: bool = None):
    """Get NumPy or CuPy based on GPU setting."""
    if use_gpu is None:
        use_gpu = USE_GPU
    return cp if (use_gpu and GPU_AVAILABLE) else np


def to_cpu(array):
    """Convert CuPy array to NumPy array if needed."""
    if GPU_AVAILABLE and cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return array


def to_gpu(array):
    """Convert NumPy array to CuPy array if GPU enabled."""
    if USE_GPU and GPU_AVAILABLE and cp is not None and isinstance(array, np.ndarray):
        return cp.asarray(array)
    return array


def analyze_beats_smart(audio_file: str, start_time: float = 0.0, end_time: float = None) -> Tuple[np.ndarray, Dict]:
    """
    Smart Mode - Multi-band frequency beat detection.
    Analyzes Kick (bass) + Clap/Snare (mid) + Hi-hat (high) frequencies.
    """
    requested_gpu = USE_GPU and GPU_AVAILABLE
    gpu_status = "GPU" if requested_gpu else "CPU"
    cuda_status = "Portable CUDA" if USING_PORTABLE_CUDA else "System CUDA/None"
    python_status = "Portable Python" if USING_PORTABLE_PYTHON else "System Python"

    print("Smart Mode - Multi-band frequency detection")
    print(f"   Mode: {gpu_status}")
    print(f"   CUDA: {cuda_status}")
    print(f"   Python: {python_status}")

    duration = None
    if end_time and end_time > start_time:
        duration = end_time - start_time

    print("   Loading audio file...")
    y, sr = librosa.load(audio_file, sr=22050, offset=start_time, duration=duration, mono=True)

    analysis_on_gpu = False
    kick_onset = None
    clap_onset = None
    combined_onset_cpu = None

    if requested_gpu:
        try:
            print("   Computing STFT on GPU...")
            stft = librosa.stft(y, n_fft=2048, hop_length=512)

            print(f"   Transferring frequency data to GPU ({cuda_status})...")
            stft_gpu = to_gpu(stft)
            freqs_gpu = to_gpu(librosa.fft_frequencies(sr=sr, n_fft=2048))

            print("   Analyzing rhythm frequency bands on GPU:")
            print("      Kick drum: 20-150 Hz (bass punch)")
            print("      Clap/Snare: 150-4000 Hz (mid-range snap)")
            print("      Hi-hat: 4000+ Hz (high-frequency)")

            kick_band = (freqs_gpu >= 20) & (freqs_gpu <= 150)
            clap_band = (freqs_gpu >= 150) & (freqs_gpu <= 4000)
            hihat_band = freqs_gpu >= 4000

            print("   Extracting frequency bands...")
            kick_onset_gpu = cp.sum(cp.abs(stft_gpu[kick_band, :]), axis=0)
            clap_onset_gpu = cp.sum(cp.abs(stft_gpu[clap_band, :]), axis=0)
            hihat_onset_gpu = cp.sum(cp.abs(stft_gpu[hihat_band, :]), axis=0)

            kick_onset_gpu = kick_onset_gpu / (cp.max(kick_onset_gpu) + 1e-6)
            clap_onset_gpu = clap_onset_gpu / (cp.max(clap_onset_gpu) + 1e-6)
            hihat_onset_gpu = hihat_onset_gpu / (cp.max(hihat_onset_gpu) + 1e-6)

            print("   Computing combined onset envelope...")
            combined_onset_gpu = (kick_onset_gpu * 3.0) + (clap_onset_gpu * 2.0) + (hihat_onset_gpu * 0.5)
            combined_onset_gpu = combined_onset_gpu / (cp.max(combined_onset_gpu) + 1e-6)

            print("   Transferring results back to CPU for beat tracking...")
            kick_onset = to_cpu(kick_onset_gpu)
            clap_onset = to_cpu(clap_onset_gpu)
            combined_onset_cpu = to_cpu(combined_onset_gpu)
            analysis_on_gpu = True
        except Exception as exc:
            print(f"   GPU frequency analysis failed: {_summarize_exception(exc)}")
            print("   Falling back to CPU frequency analysis...")
        finally:
            _clear_cupy_memory()

    if combined_onset_cpu is None:
        print("   Computing STFT on CPU...")
        stft = librosa.stft(y, n_fft=2048, hop_length=512)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        print("   Analyzing rhythm frequency bands on CPU:")
        print("      Kick drum: 20-150 Hz (bass punch)")
        print("      Clap/Snare: 150-4000 Hz (mid-range snap)")
        print("      Hi-hat: 4000+ Hz (high-frequency)")

        kick_band = (freqs >= 20) & (freqs <= 150)
        clap_band = (freqs >= 150) & (freqs <= 4000)
        hihat_band = freqs >= 4000

        print("   Extracting frequency bands...")
        kick_onset = np.sum(np.abs(stft[kick_band, :]), axis=0)
        clap_onset = np.sum(np.abs(stft[clap_band, :]), axis=0)
        hihat_onset = np.sum(np.abs(stft[hihat_band, :]), axis=0)

        kick_onset = kick_onset / (np.max(kick_onset) + 1e-6)
        clap_onset = clap_onset / (np.max(clap_onset) + 1e-6)
        hihat_onset = hihat_onset / (np.max(hihat_onset) + 1e-6)

        print("   Computing combined onset envelope...")
        combined_onset_cpu = (kick_onset * 3.0) + (clap_onset * 2.0) + (hihat_onset * 0.5)
        combined_onset_cpu = combined_onset_cpu / (np.max(combined_onset_cpu) + 1e-6)

    print("   Detecting beats...")
    try:
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=combined_onset_cpu,
            sr=sr,
            units="frames",
            hop_length=512,
            start_bpm=120,
            tightness=100,
        )
    except TypeError:
        print("   Fallback: Using standard beat tracking")
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames", hop_length=512)

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)

    beat_info = {
        "times": beat_times,
        "kick_strength": [],
        "clap_strength": [],
        "combined_strength": [],
        "tempo": tempo[0] if tempo.size > 0 else 120.0,
        "is_strong_kick": [],
        "is_strong_clap": [],
        "is_strong_beat": [],
        "analysis_device": "gpu" if analysis_on_gpu else "cpu",
    }

    print(f"   Analyzing beat strengths on {'GPU' if analysis_on_gpu else 'CPU'}...")
    for beat_frame in beat_frames:
        if beat_frame < len(kick_onset):
            start_idx = max(0, beat_frame - 3)
            end_idx = min(len(kick_onset), beat_frame + 4)

            kick_val = float(np.max(kick_onset[start_idx:end_idx]))
            clap_val = float(np.max(clap_onset[start_idx:end_idx]))
            combined_val = float(np.mean(combined_onset_cpu[start_idx:end_idx]))

            beat_info["kick_strength"].append(kick_val)
            beat_info["clap_strength"].append(clap_val)
            beat_info["combined_strength"].append(combined_val)
        else:
            beat_info["kick_strength"].append(0.0)
            beat_info["clap_strength"].append(0.0)
            beat_info["combined_strength"].append(0.0)

    beat_info["kick_strength"] = np.array(beat_info["kick_strength"])
    beat_info["clap_strength"] = np.array(beat_info["clap_strength"])
    beat_info["combined_strength"] = np.array(beat_info["combined_strength"])

    kick_threshold_50 = np.percentile(beat_info["kick_strength"], 50)
    clap_threshold_50 = np.percentile(beat_info["clap_strength"], 50)

    for i in range(len(beat_times)):
        is_kick = beat_info["kick_strength"][i] > kick_threshold_50
        is_clap = beat_info["clap_strength"][i] > clap_threshold_50
        beat_info["is_strong_kick"].append(is_kick)
        beat_info["is_strong_clap"].append(is_clap)
        beat_info["is_strong_beat"].append(is_kick or is_clap)

    beat_info["is_strong_kick"] = np.array(beat_info["is_strong_kick"])
    beat_info["is_strong_clap"] = np.array(beat_info["is_strong_clap"])
    beat_info["is_strong_beat"] = np.array(beat_info["is_strong_beat"])

    num_strong_kicks = np.sum(beat_info["is_strong_kick"])
    num_strong_claps = np.sum(beat_info["is_strong_clap"])
    num_strong_beats = np.sum(beat_info["is_strong_beat"])

    print(f"   Detected {len(beat_times)} total beats at {beat_info['tempo']:.1f} BPM")
    print(f"   Strong kicks: {num_strong_kicks} ({num_strong_kicks/len(beat_times)*100:.1f}%)")
    print(f"   Strong claps: {num_strong_claps} ({num_strong_claps/len(beat_times)*100:.1f}%)")
    print(f"   Strong beats (kick or clap): {num_strong_beats} ({num_strong_beats/len(beat_times)*100:.1f}%)")

    return beat_times, beat_info


def select_beats_smart(beat_info: Dict, preset: str = "normal") -> np.ndarray:
    """
    Intelligently select beats based on preset frequency.

    Args:
        beat_info: Dictionary from analyze_beats_smart()
        preset: Preset name (slower/slow/normal/fast/faster)

    Returns:
        Array of selected beat times
    """
    if preset not in SMART_PRESETS:
        print(f"   Unknown preset '{preset}', using 'normal'")
        preset = "normal"

    config = SMART_PRESETS[preset]
    print(f"\nSmart Beat Selection: {preset.upper()}")
    print(f"   {config['description']}")
    print(f"   Frequency mode: {config['cut_frequency']}")

    beat_times = beat_info["times"]
    kick_strength = beat_info["kick_strength"]
    clap_strength = beat_info["clap_strength"]
    is_strong_beat = beat_info["is_strong_beat"]

    np.percentile(kick_strength, config["kick_threshold"])
    np.percentile(clap_strength, config["clap_threshold"])

    selected_beats = []
    last_selected_time = -999.0

    strong_beat_indices = np.where(is_strong_beat)[0]

    if config["cut_frequency"] == "every_4th_strong_beat":
        print(f"   Selecting every 4th strong beat from {len(strong_beat_indices)} strong beats")
        for i, idx in enumerate(strong_beat_indices):
            if i % 4 == 0:
                time = beat_times[idx]
                if time - last_selected_time >= config["min_interval"]:
                    selected_beats.append(time)
                    last_selected_time = time

    elif config["cut_frequency"] == "every_2nd_strong_beat":
        print(f"   Selecting every 2nd strong beat from {len(strong_beat_indices)} strong beats")
        for i, idx in enumerate(strong_beat_indices):
            if i % 2 == 0:
                time = beat_times[idx]
                if time - last_selected_time >= config["min_interval"]:
                    selected_beats.append(time)
                    last_selected_time = time

    elif config["cut_frequency"] == "every_strong_beat":
        print(f"   Selecting all {len(strong_beat_indices)} strong beats")
        for idx in strong_beat_indices:
            time = beat_times[idx]
            if time - last_selected_time >= config["min_interval"]:
                selected_beats.append(time)
                last_selected_time = time

    elif config["cut_frequency"] == "all_beats_prioritize_strong":
        print(f"   Selecting all beats ({len(beat_times)} total) with priority on strong beats")
        for i, time in enumerate(beat_times):
            if is_strong_beat[i]:
                if time - last_selected_time >= config["min_interval"]:
                    selected_beats.append(time)
                    last_selected_time = time
            elif time - last_selected_time >= config["min_interval"] * 1.5:
                selected_beats.append(time)
                last_selected_time = time

    elif config["cut_frequency"] == "all_beats_plus_subdivisions":
        print("   Selecting all beats + subdivisions on very strong kicks")
        very_strong_kick_threshold = np.percentile(kick_strength, 80)

        for i, time in enumerate(beat_times):
            if time - last_selected_time >= config["min_interval"]:
                selected_beats.append(time)
                last_selected_time = time

                if kick_strength[i] > very_strong_kick_threshold and i < len(beat_times) - 1:
                    next_time = beat_times[i + 1]
                    mid_time = (time + next_time) / 2

                    if (mid_time - time) >= config["min_interval"] and (next_time - mid_time) >= config["min_interval"]:
                        selected_beats.append(mid_time)

    selected_beats = np.array(selected_beats)

    print(f"   Selected {len(selected_beats)} cuts from {len(beat_times)} detected beats")
    if len(selected_beats) > 1:
        avg_interval = np.mean(np.diff(selected_beats))
        print(f"   Average cut interval: {avg_interval:.3f}s ({1/avg_interval:.2f} cuts/sec)")

    return selected_beats


def avoid_scene_cuts(beat_times: np.ndarray, scene_changes: List[float], avoid_window: float = 0.3) -> np.ndarray:
    """Remove beat times that are too close to video scene changes."""
    if len(scene_changes) == 0:
        return beat_times

    print(f"\nAvoiding {len(scene_changes)} scene cuts (+/- {avoid_window}s window)")

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

    print(f"   Removed {removed_count} beats near scene changes")
    print(f"   Kept {len(filtered_beats)} beats")

    return np.array(filtered_beats)


def get_preset_info(preset: str) -> Dict:
    """Get information about a smart preset."""
    return SMART_PRESETS.get(preset, SMART_PRESETS["normal"])


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
            gpu_name = props["name"].decode("utf-8")
            cuda_version = cp.cuda.runtime.runtimeGetVersion()
            cuda_source = RUNTIME.cuda_short_label if USING_PORTABLE_CUDA else f"CUDA {cuda_version}"
            return f"{gpu_name} ({cuda_source})"
        except Exception:
            cuda_source = "Portable CUDA" if USING_PORTABLE_CUDA else "System CUDA"
            return f"GPU Available ({cuda_source})"
    if GPU_INIT_ERROR:
        return f"Unavailable ({GPU_INIT_ERROR})"
    return "No GPU"


__all__ = [
    "analyze_beats_smart",
    "select_beats_smart",
    "avoid_scene_cuts",
    "detect_video_scene_changes",
    "get_preset_info",
    "list_presets",
    "set_gpu_mode",
    "is_gpu_available",
    "get_gpu_info",
]
