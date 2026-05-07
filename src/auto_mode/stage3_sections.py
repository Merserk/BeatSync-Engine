#!/usr/bin/env python3
"""Stage 3: broad musical section detection and labeling."""

from typing import Dict, List
import librosa
import numpy as np

from . import AutoWaveConfig
from . import _safe_percentile

def analyze_sections(y: np.ndarray, y_harmonic: np.ndarray, y_percussive: np.ndarray,
                     sr: int, beat_times: np.ndarray, features: Dict,
                     cfg: AutoWaveConfig) -> List[Dict]:
    duration = len(y) / sr
    if duration <= 0 or len(beat_times) == 0:
        return [{"index": 0, "start": 0.0, "end": duration, "duration": duration, "type": "body"}]

    boundaries: List[float] = [0.0, duration]

    try:
        chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr, hop_length=cfg.hop_length, n_fft=cfg.n_fft)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10, hop_length=cfg.hop_length)
        onset = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=cfg.hop_length)[np.newaxis, :]
        min_frames = min(chroma.shape[1], mfcc.shape[1], onset.shape[1])
        frame_features = np.vstack([
            librosa.util.normalize(chroma[:, :min_frames], axis=1),
            librosa.util.normalize(mfcc[:, :min_frames], axis=1),
            librosa.util.normalize(onset[:, :min_frames], axis=1),
        ])
        target_sections = int(np.clip(round(duration / 32.0), 3, 8))
        boundary_frames = librosa.segment.agglomerative(frame_features, k=target_sections)
        boundary_times = librosa.frames_to_time(boundary_frames, sr=sr, hop_length=cfg.hop_length)
        boundaries.extend([float(t) for t in boundary_times if 0.0 < t < duration])
    except Exception as e:
        print(f"      ⚠️ Structure clustering fallback: {e}")

    # Add only major transition peaks, not every novelty spike.
    novelty = np.asarray(features.get("novelty", []), dtype=float)
    wave = np.asarray(features.get("wave", []), dtype=float)
    if novelty.size == len(beat_times):
        threshold = _safe_percentile(novelty, 91, 0.90)
        last_added = -999.0
        for idx, value in enumerate(novelty):
            if value >= threshold and wave[idx] >= _safe_percentile(wave, 45, 0.45):
                t = float(beat_times[idx])
                if 0.0 < t < duration and t - last_added >= cfg.section_min_seconds:
                    boundaries.append(t)
                    last_added = t

    boundaries_arr = merge_boundaries(np.asarray(boundaries, dtype=float), duration, min_gap=cfg.section_min_seconds)
    if len(boundaries_arr) < 3:
        step = 32.0
        boundaries_arr = np.asarray([0.0] + list(np.arange(step, duration, step)) + [duration], dtype=float)

    sections: List[Dict] = []
    median_wave = _safe_percentile(features["wave"], 50, 0.5)

    for i in range(len(boundaries_arr) - 1):
        start = float(boundaries_arr[i])
        end = float(boundaries_arr[i + 1])
        if end - start < 1.0:
            continue
        beat_idx = np.where((beat_times >= start) & (beat_times < end))[0]
        if beat_idx.size:
            section_wave = float(np.mean(features["wave"][beat_idx]))
            section_impact = float(np.mean(features["impact_score"][beat_idx]))
            section_brightness = float(np.mean(features["brightness"][beat_idx]))
        else:
            section_wave = median_wave
            section_impact = 0.5
            section_brightness = 0.5

        rel_start = start / max(duration, 1e-6)
        rel_end = end / max(duration, 1e-6)
        section_type = classify_section(rel_start, rel_end, section_wave, section_impact, median_wave)
        dominant_pattern = detect_section_pattern(features, beat_idx)

        section = {
            "index": len(sections),
            "start": start,
            "end": end,
            "duration": end - start,
            "type": section_type,
            "energy": section_wave,
            "impact": section_impact,
            "brightness": section_brightness,
            "dominant_pattern": dominant_pattern,
        }
        sections.append(section)
        print(
            f"      • {section_type:<10} {start:6.1f}s → {end:6.1f}s "
            f"({end - start:5.1f}s), wave={section_wave:.2f}, pattern={dominant_pattern}"
        )

    return sections or [{"index": 0, "start": 0.0, "end": duration, "duration": duration, "type": "body"}]


def merge_boundaries(boundaries: np.ndarray, duration: float, min_gap: float) -> np.ndarray:
    arr = np.asarray(boundaries, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = np.clip(arr, 0.0, duration)
    arr = np.unique(np.round(arr, 3))
    arr = np.sort(arr)

    merged: List[float] = []
    for t in arr:
        if not merged:
            merged.append(float(t))
            continue
        if t in (0.0, duration) or t - merged[-1] >= min_gap:
            merged.append(float(t))

    if not merged or abs(merged[0]) > 1e-6:
        merged.insert(0, 0.0)
    if abs(merged[-1] - duration) > 1e-6:
        merged.append(float(duration))
    return np.asarray(merged, dtype=float)


def classify_section(rel_start: float, rel_end: float, wave: float, impact: float, median_wave: float) -> str:
    if rel_start < 0.11:
        return "intro" if wave <= median_wave * 1.18 else "hook"
    if rel_end > 0.91:
        return "outro" if wave <= median_wave * 1.18 else "finale"
    if wave >= max(0.72, median_wave * 1.22) and impact >= 0.58:
        return "drop"
    if wave >= max(0.60, median_wave * 1.12):
        return "chorus"
    if impact >= 0.64 and wave < median_wave:
        return "bridge"
    if wave < median_wave * 0.82:
        return "breakdown"
    return "verse"


def detect_section_pattern(features: Dict, beat_indices: np.ndarray) -> str:
    if beat_indices.size == 0:
        return "mixed"
    kick_ratio = float(np.mean(features["is_strong_kick"][beat_indices]))
    clap_ratio = float(np.mean(features["is_strong_clap"][beat_indices]))
    bass_ratio = float(np.mean(features["is_strong_bass"][beat_indices]))
    hihat_ratio = float(np.mean(features["is_strong_hihat"][beat_indices]))

    if kick_ratio > 0.42 and clap_ratio > 0.34:
        return "kick_clap"
    if kick_ratio >= max(clap_ratio, bass_ratio, hihat_ratio) and kick_ratio > 0.34:
        return "kick"
    if clap_ratio >= max(kick_ratio, bass_ratio, hihat_ratio) and clap_ratio > 0.34:
        return "clap"
    if bass_ratio >= max(kick_ratio, clap_ratio, hihat_ratio) and bass_ratio > 0.34:
        return "bass"
    if hihat_ratio > 0.46:
        return "hihat"
    return "mixed"
