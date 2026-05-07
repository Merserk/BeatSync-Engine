#!/usr/bin/env python3
"""
Auto Mode package entrypoint + shared support.

This is the only non-stage file in src/auto_mode.
All tuning, helper utilities, and the public analyze_beats_auto() pipeline live here.
The remaining files are editable processing stages:
- stage1_audio.py
- stage2_features.py
- stage3_sections.py
- stage4_select.py
- stage5_qwen_scene_worker.py
- stage6_av_planner.py

Existing project imports remain compatible:
    from auto_mode import analyze_beats_auto
"""

import os
import sys
import warnings
from dataclasses import dataclass

# This package lives at src/auto_mode. Add src to sys.path so shared modules
# such as logger.py and gpu_cpu_utils.py remain importable.
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from logger import setup_environment

setup_environment()
warnings.filterwarnings("ignore")

@dataclass(frozen=True)
class AutoWaveConfig:
    """Creative + technical tuning for Auto Mode V3.2."""

    sr: int = 22050
    hop_length: int = 512
    n_fft: int = 2048

    # The main retune: fewer cuts, less flicker.
    # These are hard safety floors by energy class.
    low_energy_min_interval: float = 0.90
    medium_energy_min_interval: float = 0.58
    high_energy_min_interval: float = 0.38
    peak_energy_min_interval: float = 0.30

    # Avoid endless holds while still allowing cinematic breathing.
    low_energy_max_hold: float = 3.80
    medium_energy_max_hold: float = 2.80
    high_energy_max_hold: float = 1.85
    peak_energy_max_hold: float = 1.25

    # Grid sizes. The selector mostly moves by these beat steps.
    phrase_beats: int = 8
    bar_beats: int = 4

    # Micro/half-beat cuts were the main reason V3 could feel too busy.
    # V3.2 keeps them almost disabled and only uses them for rare huge impacts.
    enable_rare_micro_cuts: bool = True
    max_micro_cut_ratio: float = 0.025
    micro_min_gap: float = 0.34
    micro_percentile: float = 96.5

    # Smooth the energy-to-density curve so the edit behaves like waves rather
    # than a nervous switch reacting to every transient.
    wave_smooth_beats: int = 16
    section_min_seconds: float = 10.0

    # Global cap: protects against too many cuts in very dense music.
    target_cut_ratio_min: float = 0.22
    target_cut_ratio_max: float = 0.46

    # Auto Mode V4: analyze source footage and let the renderer use an
    # audio-visual clip plan instead of random source sampling.
    enable_video_analysis: bool = True
    enable_qwen_semantics: bool = True
    qwen_model_path: str = ""

    # Prefer strong downbeats/phrase anchors over off-grid novelty hits.
    anchor_bonus: float = 0.32
    phrase_bonus: float = 0.48


CONFIG = AutoWaveConfig()


# ---------------------------------------------------------------------------


from typing import Callable, Dict, List, Tuple
import librosa
import numpy as np
from gpu_cpu_utils import GPU_AVAILABLE, clear_gpu_memory

# ---------------------------------------------------------------------------
# Shared numerical helpers
# ---------------------------------------------------------------------------


def _to_float(value, default: float = 0.0) -> float:
    try:
        arr = np.asarray(value).reshape(-1)
        if arr.size:
            return float(arr[0])
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return default


def _normalize(values: np.ndarray, default: float = 0.0) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=default, posinf=default, neginf=default)
    lo = float(np.percentile(arr, 2))
    hi = float(np.percentile(arr, 98))
    if hi - lo < 1e-8:
        return np.zeros_like(arr) + default
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size < 3 or width <= 1:
        return arr
    width = int(max(1, min(width, max(1, arr.size))))
    if width % 2 == 0:
        width += 1
    pad = width // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(padded, kernel, mode="valid")


def _safe_percentile(values: np.ndarray, percentile: float, default: float = 0.0) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(np.percentile(arr, percentile))


def _unique_sorted(times: np.ndarray, min_gap: float) -> np.ndarray:
    arr = np.asarray(times, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return arr
    arr = np.sort(arr)

    out: List[float] = []
    for t in arr:
        t = float(t)
        if not out or t - out[-1] >= min_gap:
            out.append(t)
        else:
            # Keep the existing earlier beat; V3.2 prefers stable downbeat timing
            # over squeezing in nearby cuts.
            continue
    return np.asarray(out, dtype=float)


def _interp_to_beats(curve: np.ndarray, beat_times: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    if len(curve) == 0 or len(beat_times) == 0:
        return np.zeros(len(beat_times), dtype=float)
    curve_times = librosa.frames_to_time(np.arange(len(curve)), sr=sr, hop_length=hop_length)
    return np.interp(beat_times, curve_times, curve, left=float(curve[0]), right=float(curve[-1]))


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Import stage functions after shared support is defined.
# ---------------------------------------------------------------------------

from .stage1_audio import detect_master_beat_grid
from .stage2_features import analyze_wave_features
from .stage3_sections import analyze_sections
from .stage4_select import select_wave_cuts

# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------


def _build_audio_visual_profile(tempo: float, sections: List[Dict], features: Dict,
                                selected_beats: np.ndarray, beat_times: np.ndarray) -> Dict:
    wave = np.asarray(features.get("wave", []), dtype=float)
    impact = np.asarray(features.get("impact_score", []), dtype=float)
    rhythm = np.asarray(features.get("rhythm_score", []), dtype=float)

    avg_wave = float(np.mean(wave)) if wave.size else 0.5
    avg_impact = float(np.mean(impact)) if impact.size else 0.5
    peak_ratio = float(np.mean(wave >= _safe_percentile(wave, 82, 0.82))) if wave.size else 0.0
    avg_cut_interval = float(np.mean(np.diff(selected_beats))) if len(selected_beats) > 1 else 1.5
    drop_count = sum(1 for s in sections if s.get("type") in {"drop", "finale", "chorus"})

    if avg_cut_interval <= 0.72 or avg_impact >= 0.58:
        smart_preset = "rhythmic_hype_gmv_amv"
    elif avg_wave <= 0.42 and drop_count <= 1:
        smart_preset = "cinematic_soft_amv"
    elif peak_ratio >= 0.22 or drop_count >= 2:
        smart_preset = "hybrid_drop_story"
    else:
        smart_preset = "rhythmic_flow_gmv_amv"

    return {
        "tempo": float(tempo),
        "smart_preset": smart_preset,
        "average_wave": avg_wave,
        "average_impact": avg_impact,
        "average_rhythm": float(np.mean(rhythm)) if rhythm.size else 0.5,
        "peak_ratio": peak_ratio,
        "average_cut_interval": avg_cut_interval,
        "cut_count": int(len(selected_beats)),
        "beat_count": int(len(beat_times)),
        "section_types": [s.get("type", "body") for s in sections],
        "rhythm_preference": "rhythmic",
    }


def _notify_progress(progress_callback: Callable[[str], None] | None, stage_number: int) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(f"Stage {stage_number} is processing. Please wait.")
    except Exception:
        pass


def _notify_console(console_callback: Callable[[int, str], None] | None,
                    stage_number: int, message: str) -> None:
    if console_callback is None:
        return
    try:
        console_callback(stage_number, message)
    except Exception:
        pass


def analyze_beats_auto(audio_file: str, start_time: float = 0.0,
                       end_time: float = None, use_gpu: bool = False,
                       video_files: List[str] = None,
                       enable_video_analysis: bool = True,
                       enable_qwen_semantics: bool = True,
                       qwen_model_path: str = None,
                       progress_callback: Callable[[str], None] | None = None,
                       console_callback: Callable[[int, str], None] | None = None) -> Tuple[np.ndarray, Dict]:
    """
    Build a cleaner Auto Mode cut plan.

    The edit behaves like waves:
    - small waves: longer holds, mainly phrase/bar anchors;
    - medium waves: cuts every 2-4 beats;
    - big waves: tighter 1-2 beat rhythm, but only on strong musical impacts.
    """
    cfg = CONFIG

    print("🤖 AUTO MODE V4 - Audio-Visual Rhythmic GMV/AMV Planner")
    print("   Rhythm-first audio cuts + semantic video moment matching")

    duration = None
    if end_time and end_time > start_time:
        duration = end_time - start_time

    print("   🎵 Loading audio...")
    y, sr = librosa.load(audio_file, sr=cfg.sr, offset=start_time, duration=duration, mono=True)
    if y.size == 0:
        raise ValueError("Audio file is empty or could not be decoded.")

    audio_duration = len(y) / sr
    y = librosa.util.normalize(y)

    try:
        y_harmonic, y_percussive = librosa.effects.hpss(y)
    except Exception:
        y_harmonic, y_percussive = y, y

    _notify_progress(progress_callback, 1)
    print("   🥁 Step 1: Detecting stable beat grid...")
    beat_times, tempo, beat_frames, onset_env = detect_master_beat_grid(y_percussive, sr, cfg)
    if len(beat_times) < 2:
        raise ValueError("Auto Mode could not detect enough rhythmic events to build a cut plan.")
    print(f"      ✓ {len(beat_times)} beats detected at {tempo:.1f} BPM")
    _notify_console(console_callback, 1, f"Beat grid: {len(beat_times)} beats at {tempo:.1f} BPM")

    _notify_progress(progress_callback, 2)
    print("   🌊 Step 2: Reading energy waves and rhythm impacts...")
    features = analyze_wave_features(y, y_percussive, sr, beat_times, beat_frames, onset_env, cfg, use_gpu)
    wave = np.asarray(features.get("wave", []), dtype=float)
    impact = np.asarray(features.get("impact_score", []), dtype=float)
    rhythm = np.asarray(features.get("rhythm_score", []), dtype=float)
    _notify_console(console_callback, 2, "Energy and rhythm features ready")
    if wave.size:
        _notify_console(console_callback, 2, f"Energy wave: avg {float(np.mean(wave)):.2f}, peak {float(np.max(wave)):.2f}")
    if impact.size:
        strong_impacts = int(np.sum(impact >= _safe_percentile(impact, 88, 0.88)))
        _notify_console(console_callback, 2, f"Strong rhythm impacts: {strong_impacts}/{len(impact)} beats")
    if rhythm.size:
        _notify_console(console_callback, 2, f"Rhythm strength: avg {float(np.mean(rhythm)):.2f}, peak {float(np.max(rhythm)):.2f}")

    _notify_progress(progress_callback, 3)
    print("   🎼 Step 3: Detecting broad musical sections...")
    sections = analyze_sections(y, y_harmonic, y_percussive, sr, beat_times, features, cfg)
    print(f"      ✓ {len(sections)} sections")
    section_types = [str(s.get("type", "section")) for s in sections[:5]]
    _notify_console(console_callback, 3, f"Sections: {len(sections)}")
    if section_types:
        _notify_console(console_callback, 3, "Section types: " + ", ".join(section_types))
    if sections:
        longest = max(sections, key=lambda s: float(s.get("duration", 0.0)))
        _notify_console(
            console_callback,
            3,
            f"Longest section: {longest.get('type', 'section')} ({float(longest.get('duration', 0.0)):.1f}s)",
        )

    _notify_progress(progress_callback, 4)
    print("   🧠 Step 4: Selecting deliberate rhythmic cuts...")
    selected_beats, selection_info = select_wave_cuts(
        beat_times=beat_times,
        sections=sections,
        features=features,
        tempo=tempo,
        audio_duration=audio_duration,
        cfg=cfg,
    )

    if selected_beats.size == 0:
        selected_beats = _unique_sorted(beat_times[::4], cfg.low_energy_min_interval)

    cut_ratio = len(selected_beats) / max(1, len(beat_times)) * 100.0
    avg_interval = float(np.mean(np.diff(selected_beats))) if len(selected_beats) > 1 else 0.0
    print(f"   ✓ Selected {len(selected_beats)} clean cuts from {len(beat_times)} beats ({cut_ratio:.1f}%)")
    _notify_console(
        console_callback,
        4,
        f"Cut selection: {len(selected_beats)} cuts from {len(beat_times)} beats ({cut_ratio:.1f}%)",
    )
    if avg_interval:
        print(f"   ✓ Average visual rhythm: {avg_interval:.3f}s per cut ({1.0 / avg_interval:.2f} cuts/sec)")
        _notify_console(
            console_callback,
            4,
            f"Visual rhythm: {avg_interval:.3f}s per cut ({1.0 / avg_interval:.2f} cuts/sec)",
        )
    print("   ✓ Rhythm retune: stronger beat/bar/phrase sync")

    audio_visual_profile = _build_audio_visual_profile(
        tempo=tempo,
        sections=sections,
        features=features,
        selected_beats=selected_beats,
        beat_times=beat_times,
    )
    print(f"   ✓ Smart preset: {audio_visual_profile['smart_preset']}")
    _notify_console(console_callback, 4, f"Preset: {audio_visual_profile['smart_preset']}")

    video_analysis = None
    should_analyze_video = bool(
        cfg.enable_video_analysis and enable_video_analysis and video_files
    )
    if should_analyze_video:
        try:
            _notify_progress(progress_callback, 5)
            from video_analysis import DEFAULT_QWEN_MODEL_DIR, analyze_video_sources

            model_path = (
                qwen_model_path
                or cfg.qwen_model_path
                or DEFAULT_QWEN_MODEL_DIR
            )
            qwen_enabled = bool(cfg.enable_qwen_semantics and enable_qwen_semantics)
            if os.environ.get("BEATSYNC_DISABLE_QWEN", "0") == "1":
                qwen_enabled = False
            video_analysis = analyze_video_sources(
                video_files=video_files,
                audio_profile=audio_visual_profile,
                use_gpu=use_gpu,
                enable_ai=qwen_enabled,
                qwen_model_path=model_path,
            )
        except Exception as e:
            print(f"   ⚠️  Video analysis failed; renderer will use fallback sampling: {e}")
            _notify_console(console_callback, 5, f"Video analysis failed; fallback sampling: {e}")

    energy_profile = {
        "beat_energy": features["energy"],
        "energy_levels": features["energy_levels"],
        "rms": features["rms_curve"],
        "spectral_centroid": features["centroid_curve"],
        "zcr": features["flux_curve"],
        "wave": features["wave"],
        "arc": features["arc"],
    }

    rhythm_data = {
        "kick_strength": features["kick"],
        "clap_strength": features["clap"],
        "hihat_strength": features["hihat"],
        "bass_strength": features["bass"],
        "combined_strength": features["rhythm_score"],
        "impact_strength": features["impact_score"],
        "novelty_strength": features["novelty"],
        "is_strong_kick": features["is_strong_kick"],
        "is_strong_clap": features["is_strong_clap"],
        "is_strong_hihat": features["is_strong_hihat"],
        "is_strong_bass": features["is_strong_bass"],
        "is_bar_anchor": features["is_bar_anchor"],
        "is_phrase_anchor": features["is_phrase_anchor"],
    }

    beat_info = {
        "times": beat_times,
        "selected_times": selected_beats,
        "tempo": tempo,
        "sections": sections,
        "energy_profile": energy_profile,
        "rhythm_data": rhythm_data,
        "rhythm_patterns": {s["index"]: s.get("dominant_pattern", "mixed") for s in sections},
        "selection_info": selection_info,
        "audio_visual_profile": audio_visual_profile,
        "video_analysis": video_analysis,
        "audio_duration": audio_duration,
        "mode": "auto_v4_audio_visual_rhythmic_planner",
        "auto_style": "audio_visual_rhythmic_gmv_amv",
    }

    try:
        if use_gpu and GPU_AVAILABLE:
            clear_gpu_memory()
    except Exception:
        pass

    return selected_beats, beat_info

def get_auto_mode_info() -> str:
    """Return human-readable Auto Mode V3.2 summary for UI/debug use."""
    return """🤖 Auto Mode V4 - Audio-Visual Rhythmic GMV/AMV Planner

Tuning:
- Stronger downbeat/bar/phrase synchronization
- Smooth wave-like density: calm parts hold, big moments cut faster
- FFmpeg/OpenCV source-video analysis
- Optional Qwen3-VL semantic tags for action, beauty, emotion, and use-case
- Renderer chooses planned source moments instead of random clips
- Designed for rhythmic, professional GMV/AMV flow
"""


def analyze_beats_auto_fallback(audio_file: str, start_time: float = 0.0,
                                end_time: float = None, use_gpu: bool = False) -> Tuple[np.ndarray, Dict]:
    """Conservative fallback kept for compatibility."""
    duration = None
    if end_time and end_time > start_time:
        duration = end_time - start_time

    y, sr = librosa.load(audio_file, sr=CONFIG.sr, offset=start_time, duration=duration, mono=True)
    y = librosa.util.normalize(y)
    beat_times, tempo, _, _ = detect_master_beat_grid(y, sr, CONFIG)
    selected = _unique_sorted(beat_times[::4], CONFIG.low_energy_min_interval)
    beat_info = {
        "times": beat_times,
        "selected_times": selected,
        "tempo": tempo,
        "sections": [],
        "energy_profile": {},
        "rhythm_data": {},
        "rhythm_patterns": {},
        "selection_info": [],
        "mode": "auto_v3_2_fallback",
    }
    return selected, beat_info


__all__ = [
    "AutoWaveConfig",
    "CONFIG",
    "analyze_beats_auto",
    "analyze_beats_auto_fallback",
    "get_auto_mode_info",
]
