#!/usr/bin/env python3
"""
Manual Mode - Bass-Focused Beat Detection
Simple manual cut intensity control with bass frequency analysis
"""

import numpy as np
import librosa
from typing import Tuple, Dict
import warnings

warnings.filterwarnings('ignore')


def analyze_beats_manual(audio_file: str, start_time: float = 0.0, 
                        end_time: float = None, use_gpu: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    Manual mode beat analysis - bass-focused detection.
    Simple and straightforward beat tracking on bass frequencies (20-200 Hz).
    
    Args:
        audio_file: Path to audio file
        start_time: Start time in seconds
        end_time: End time in seconds (None = full duration)
        use_gpu: Use GPU acceleration if available
    
    Returns:
        Tuple of (beat_times array, beat_info dictionary)
    """
    # GPU support
    try:
        import cupy as cp
        gpu_available = use_gpu
    except ImportError:
        gpu_available = False
        cp = None
    
    duration = None
    if end_time and end_time > start_time:
        duration = end_time - start_time

    gpu_status = "🚀 GPU" if gpu_available else "💻 CPU"
    print(f"🎵 Manual Mode - Bass-focused beat detection")
    print(f"   Loading audio on {gpu_status}...")
    
    y, sr = librosa.load(audio_file, sr=22050, offset=start_time, duration=duration, mono=True)

    print(f"   📊 Analyzing bass frequencies (20-200 Hz)...")
    
    # Use GPU acceleration if available
    if gpu_available and cp is not None:
        print(f"   📤 Transferring audio to GPU...")
        
        # Compute STFT on CPU (librosa)
        stft = librosa.stft(y, n_fft=2048, hop_length=512)
        stft_gpu = cp.asarray(stft)
        
        # Frequency analysis on GPU
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        freqs_gpu = cp.asarray(freqs)
        
        bass_band = (freqs_gpu >= 20) & (freqs_gpu <= 200)
        bass_onset_env_gpu = cp.sum(cp.abs(stft_gpu[bass_band, :]), axis=0)
        
        # Transfer back to CPU
        print(f"   📥 Transferring results from GPU...")
        bass_onset_env = cp.asnumpy(bass_onset_env_gpu)
        
        # Clear GPU memory
        del stft_gpu, freqs_gpu, bass_band, bass_onset_env_gpu
        cp.get_default_memory_pool().free_all_blocks()
    else:
        # CPU processing
        stft = librosa.stft(y, n_fft=2048, hop_length=512)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        bass_band = (freqs >= 20) & (freqs <= 200)
        bass_onset_env = np.sum(np.abs(stft[bass_band, :]), axis=0)

    print(f"   🥁 Detecting beats...")
    try:
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=bass_onset_env, 
            sr=sr, 
            units='frames',
            hop_length=512
        )
    except TypeError:
        print("   ⚠️  Using fallback beat tracking (older librosa version)")
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames', hop_length=512)

    if tempo.size > 0:
        print(f"   ✓ Detected tempo: {tempo[0]:.2f} BPM")
    else:
        print("   ❌ Could not detect tempo")
        return np.array([]), {'tempo': 120.0, 'times': np.array([])}

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)
    print(f"   ✓ Detected {len(beat_times)} bass beats")
    
    beat_info = {
        'times': beat_times,
        'tempo': tempo[0] if tempo.size > 0 else 120.0
    }
    
    return beat_times, beat_info


def interpolate_beats(beat_times: np.ndarray, subdivisions: int) -> np.ndarray:
    """
    Interpolate additional beats between detected beats for more frequent cuts.
    
    Args:
        beat_times: Array of beat timestamps
        subdivisions: Number of subdivisions (e.g., 2 = double the beats)
    
    Returns:
        Array of interpolated beat times
    """
    if len(beat_times) < 2:
        return beat_times
    
    interpolated = []
    for i in range(len(beat_times) - 1):
        start = beat_times[i]
        end = beat_times[i + 1]
        for j in range(subdivisions):
            interpolated.append(start + (end - start) * j / subdivisions)
    
    interpolated.append(beat_times[-1])
    
    return np.array(interpolated)


def process_manual_intensity(beat_times: np.ndarray, cut_intensity: float) -> np.ndarray:
    """
    Process beat times based on manual cut intensity value.
    
    Args:
        beat_times: Original detected beat times
        cut_intensity: Intensity value (< 1.0 = subdivide, >= 1.0 = skip)
    
    Returns:
        Processed beat times array
    """
    if cut_intensity < 1.0:
        # Subdivide beats for more frequent cuts
        subdivisions = int(1.0 / cut_intensity)
        print(f"   ✂️  Subdividing beats by {subdivisions}x (more cuts)")
        return interpolate_beats(beat_times, subdivisions)
    else:
        # Skip beats for fewer cuts
        cut_intensity_int = int(cut_intensity)
        selected_beats = beat_times[::cut_intensity_int]
        print(f"   ✂️  Using every {cut_intensity_int} beats (fewer cuts)")
        return selected_beats


def get_manual_mode_info() -> Dict:
    """Get information about manual mode."""
    return {
        'name': 'Manual Mode',
        'description': 'Bass-focused beat detection with simple intensity control',
        'frequency_range': '20-200 Hz (bass frequencies)',
        'control': 'Cut Intensity slider',
        'intensity_explanation': {
            '< 1.0': 'Subdivide beats (MORE cuts)',
            '1.0': 'Every beat (standard)',
            '> 1.0': 'Skip beats (FEWER cuts)'
        }
    }