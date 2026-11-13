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


def refine_musical_anchors(beat_times: np.ndarray, sections: List[Dict],
                           features: Dict, cfg: AutoWaveConfig) -> Dict:
    """Snap bar/phrase anchors to section starts and major musical events."""
    beats = np.asarray(beat_times, dtype=float)
    if beats.size == 0:
        return features

    bar_anchor = np.asarray(features.get("is_bar_anchor", np.zeros(beats.size, dtype=bool)), dtype=bool).copy()
    phrase_anchor = np.asarray(features.get("is_phrase_anchor", np.zeros(beats.size, dtype=bool)), dtype=bool).copy()
    if bar_anchor.size != beats.size:
        bar_anchor = np.zeros(beats.size, dtype=bool)
    if phrase_anchor.size != beats.size:
        phrase_anchor = np.zeros(beats.size, dtype=bool)

    impact = _feature_or_default(features, "impact_score", beats.size, 0.5)
    novelty = _feature_or_default(features, "novelty", beats.size, 0.0)
    confidence = _feature_or_default(features, "beat_confidence", beats.size, 0.5)
    wave = _feature_or_default(features, "wave", beats.size, 0.5)
    anchor_score = 0.42 * impact + 0.28 * confidence + 0.20 * novelty + 0.10 * wave

    phrase_size = max(1, int(cfg.phrase_beats))
    search_radius = max(1, min(3, phrase_size // 2))

    for section in sections or []:
        start = float(section.get("start", 0.0))
        end = float(section.get("end", start))
        section_indices = np.where((beats >= start) & (beats < end))[0]
        if section_indices.size == 0:
            continue

        first_local = int(section_indices[0])
        first_anchor = _best_anchor_near(first_local, anchor_score, search_radius)
        phrase_anchor[first_anchor] = True
        bar_anchor[first_anchor] = True

        local_positions = np.arange(0, section_indices.size, phrase_size, dtype=int)
        for local_pos in local_positions:
            idx = int(section_indices[min(local_pos, section_indices.size - 1)])
            best = _best_anchor_near(idx, anchor_score, search_radius)
            phrase_anchor[best] = True

    peak_threshold = max(
        _safe_percentile(anchor_score, 92, 0.90),
        _safe_percentile(anchor_score, 70, 0.70) + 0.08,
    )
    min_peak_gap = max(2, phrase_size // 2)
    last_peak = -999
    for idx in np.argsort(anchor_score)[::-1]:
        idx = int(idx)
        if anchor_score[idx] < peak_threshold:
            break
        if idx - last_peak < min_peak_gap and last_peak >= 0:
            continue
        phrase_anchor[idx] = True
        last_peak = idx

    downbeat_confidence = np.asarray(features.get("downbeat_confidence", np.zeros(beats.size)), dtype=float)
    if downbeat_confidence.size != beats.size:
        downbeat_confidence = np.zeros(beats.size, dtype=float)
    downbeat_confidence = np.maximum(downbeat_confidence, np.where(bar_anchor, 0.72, 0.0))
    downbeat_confidence = np.maximum(downbeat_confidence, np.where(phrase_anchor, 0.86, 0.0))

    features["is_bar_anchor"] = bar_anchor
    features["is_phrase_anchor"] = phrase_anchor
    features["downbeat_confidence"] = np.clip(downbeat_confidence, 0.0, 1.0)
    return features


def _feature_or_default(features: Dict, key: str, size: int, default: float) -> np.ndarray:
    arr = np.asarray(features.get(key, []), dtype=float)
    if arr.size != size:
        return np.zeros(size, dtype=float) + default
    return np.nan_to_num(arr, nan=default, posinf=default, neginf=default)


def _best_anchor_near(index: int, score: np.ndarray, radius: int) -> int:
    if score.size == 0:
        return max(0, index)
    lo = max(0, int(index) - int(radius))
    hi = min(score.size, int(index) + int(radius) + 1)
    if hi <= lo:
        return max(0, min(score.size - 1, int(index)))
    local = score[lo:hi]
    return int(lo + int(np.argmax(local)))


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
