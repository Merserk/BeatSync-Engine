#!/usr/bin/env python3
"""Stage 2: beat-synchronous energy wave and rhythm feature extraction."""

from typing import Dict, Tuple
import librosa
import numpy as np

from gpu_cpu_utils import GPU_AVAILABLE, rhythm_band_sums
from . import AutoWaveConfig
from . import _interp_to_beats, _normalize, _safe_percentile, _smooth

def analyze_wave_features(y: np.ndarray, y_percussive: np.ndarray, sr: int,
                          beat_times: np.ndarray, beat_frames: np.ndarray,
                          onset_env: np.ndarray, cfg: AutoWaveConfig,
                          use_gpu: bool = False) -> Dict:
    """Extract beat-synchronous energy/rhythm data with smooth wave behavior."""
    duration = len(y) / sr

    rms_curve = librosa.feature.rms(y=y, frame_length=cfg.n_fft, hop_length=cfg.hop_length)[0]
    centroid_curve = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length)[0]
    flux_curve = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=cfg.hop_length)

    rms_curve_n = _normalize(_smooth(rms_curve, 7))
    centroid_curve_n = _normalize(_smooth(centroid_curve, 7))
    flux_curve_n = _normalize(_smooth(flux_curve, 5))
    onset_n = _normalize(onset_env)

    rms = _interp_to_beats(rms_curve_n, beat_times, sr, cfg.hop_length)
    centroid = _interp_to_beats(centroid_curve_n, beat_times, sr, cfg.hop_length)
    flux = _interp_to_beats(flux_curve_n, beat_times, sr, cfg.hop_length)
    onset = _interp_to_beats(onset_n, beat_times, sr, cfg.hop_length)

    kick, bass, clap, hihat = analyze_rhythm_bands(y, sr, beat_times, cfg, use_gpu)

    # Musical novelty, but smoothed so it does not create twitchy cuts.
    novelty = _normalize(0.50 * flux + 0.35 * onset + 0.15 * np.abs(np.gradient(rms)))
    novelty = _smooth(novelty, 3)

    brightness = _normalize(0.65 * centroid + 0.35 * hihat)
    energy_raw = _normalize(0.55 * rms + 0.20 * flux + 0.15 * brightness + 0.10 * bass)

    # The important V3.2 change: a slower energy wave controls density.
    wave = _smooth(energy_raw, cfg.wave_smooth_beats)
    wave = _normalize(0.70 * wave + 0.30 * energy_raw)

    # Middle/finale arc lets the edit grow naturally without overcutting intro/outro.
    position = beat_times / max(duration, 1e-6)
    arc = np.sin(np.clip(position, 0.0, 1.0) * np.pi) ** 0.65
    arc = _normalize(0.58 * arc + 0.42 * wave)

    rhythm_score = _normalize(0.34 * kick + 0.26 * bass + 0.27 * clap + 0.08 * hihat + 0.05 * onset)
    impact_score = _normalize(0.42 * rhythm_score + 0.24 * novelty + 0.24 * wave + 0.10 * arc)
    tempo_stability = _tempo_stability_from_beats(beat_times)
    beat_confidence = _normalize(0.34 * onset + 0.28 * rhythm_score + 0.22 * impact_score + 0.16 * tempo_stability)

    high_thr = _safe_percentile(wave, 72, 0.72)
    peak_thr = _safe_percentile(wave, 88, 0.88)
    low_thr = _safe_percentile(wave, 30, 0.30)
    energy_levels = np.asarray([
        "peak" if e >= peak_thr else "high" if e >= high_thr else "low" if e <= low_thr else "medium"
        for e in wave
    ], dtype=object)

    kick_thr = _safe_percentile(kick, 72, 0.68)
    clap_thr = _safe_percentile(clap, 74, 0.68)
    hihat_thr = _safe_percentile(hihat, 82, 0.72)
    bass_thr = _safe_percentile(bass, 74, 0.68)

    beat_index = np.arange(len(beat_times), dtype=int)
    bar_phase, bar_phase_confidence = _infer_bar_phase(kick, bass, clap, hihat, cfg)
    bar_size = max(1, int(cfg.bar_beats))
    phrase_size = max(1, int(cfg.phrase_beats))
    is_bar_anchor = ((beat_index - bar_phase) % bar_size) == 0
    is_phrase_anchor = ((beat_index - bar_phase) % phrase_size) == 0
    downbeat_confidence = np.where(is_bar_anchor, 0.55 + 0.45 * beat_confidence, 0.18 * beat_confidence)

    return {
        "kick": kick,
        "bass": bass,
        "clap": clap,
        "hihat": hihat,
        "rms": rms,
        "centroid": centroid,
        "flux": flux,
        "onset": onset,
        "novelty": novelty,
        "energy": energy_raw,
        "wave": wave,
        "brightness": brightness,
        "rhythm_score": rhythm_score,
        "impact_score": impact_score,
        "tempo_stability": tempo_stability,
        "beat_confidence": beat_confidence,
        "downbeat_confidence": downbeat_confidence,
        "bar_phase": int(bar_phase),
        "bar_phase_confidence": float(bar_phase_confidence),
        "arc": arc,
        "energy_levels": energy_levels,
        "is_strong_kick": kick >= kick_thr,
        "is_strong_clap": clap >= clap_thr,
        "is_strong_hihat": hihat >= hihat_thr,
        "is_strong_bass": bass >= bass_thr,
        "is_bar_anchor": is_bar_anchor,
        "is_phrase_anchor": is_phrase_anchor,
        "rms_curve": rms_curve_n,
        "centroid_curve": centroid_curve_n,
        "flux_curve": flux_curve_n,
    }


def _tempo_stability_from_beats(beat_times: np.ndarray) -> np.ndarray:
    """Return per-beat stability against the local median beat interval."""
    beats = np.asarray(beat_times, dtype=float)
    if beats.size == 0:
        return np.asarray([], dtype=float)
    if beats.size < 3:
        return np.ones(beats.size, dtype=float)

    intervals = np.diff(beats)
    valid = intervals[np.isfinite(intervals) & (intervals > 1e-6)]
    if valid.size == 0:
        return np.ones(beats.size, dtype=float)

    median_interval = float(np.median(valid))
    if median_interval <= 1e-6:
        return np.ones(beats.size, dtype=float)

    local = np.empty(beats.size, dtype=float)
    local[0] = intervals[0]
    local[-1] = intervals[-1]
    local[1:-1] = 0.5 * (intervals[:-1] + intervals[1:])
    deviation = np.abs(local - median_interval) / median_interval
    return np.clip(1.0 - 2.2 * deviation, 0.0, 1.0)


def _infer_bar_phase(kick: np.ndarray, bass: np.ndarray, clap: np.ndarray,
                     hihat: np.ndarray, cfg: AutoWaveConfig) -> Tuple[int, float]:
    """Infer which beat index is most likely the downbeat of each bar."""
    bar_size = max(1, int(cfg.bar_beats))
    if bar_size <= 1:
        return 0, 1.0

    arrays = [np.asarray(v, dtype=float).reshape(-1) for v in (kick, bass, clap, hihat)]
    length = min((arr.size for arr in arrays), default=0)
    if length < bar_size:
        return 0, 0.0

    kick, bass, clap, hihat = [np.nan_to_num(arr[:length], nan=0.0) for arr in arrays]
    beat_index = np.arange(length, dtype=int)
    scores = []
    half_bar = max(1, bar_size // 2)

    for phase in range(bar_size):
        rel = (beat_index - phase) % bar_size
        downbeat = rel == 0
        backbeat = rel == half_bar
        offbeat = (rel % 2) == 1

        score = 0.0
        if np.any(downbeat):
            score += 0.44 * float(np.mean(kick[downbeat]))
            score += 0.25 * float(np.mean(bass[downbeat]))
            score -= 0.10 * float(np.mean(clap[downbeat]))
        if np.any(backbeat):
            score += 0.18 * float(np.mean(clap[backbeat]))
        if np.any(offbeat):
            score += 0.08 * float(np.mean(hihat[offbeat]))
            score += 0.05 * float(np.mean(clap[offbeat]))
        scores.append(score)

    score_arr = np.asarray(scores, dtype=float)
    best = int(np.argmax(score_arr))
    if score_arr.size < 2:
        return best, 0.0

    sorted_scores = np.sort(score_arr)
    best_score = float(sorted_scores[-1])
    runner_up = float(sorted_scores[-2])
    confidence = 0.0 if best_score <= 1e-8 else np.clip((best_score - runner_up) / best_score, 0.0, 1.0)
    return best, float(confidence)


def analyze_rhythm_bands(y: np.ndarray, sr: int, beat_times: np.ndarray,
                         cfg: AutoWaveConfig, use_gpu: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Beat-level kick/bass/clap/hihat strength."""
    stft = librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=cfg.n_fft)

    band_ranges = ((35, 145), (35, 220), (150, 4200), (4200, np.inf))
    curves = None
    if use_gpu and GPU_AVAILABLE:
        try:
            curves = rhythm_band_sums(stft, freqs)
        except Exception as exc:
            print(f"      Taichi rhythm analysis failed; using CPU: {exc}")
    if curves is None:
        magnitude = np.abs(stft)
        curves = [np.sum(magnitude[(freqs >= low) & (freqs <= high), :], axis=0)
                  for low, high in band_ranges]

    frame_times = librosa.frames_to_time(np.arange(stft.shape[1]), sr=sr, hop_length=cfg.hop_length)
    outputs: Dict[str, np.ndarray] = {}

    for name, curve in zip(("kick", "bass", "clap", "hihat"), curves):
        try:
            curve = _normalize(_smooth(np.asarray(curve, dtype=float), 3))
            outputs[name] = np.interp(beat_times, frame_times, curve, left=float(curve[0]), right=float(curve[-1]))
        except Exception:
            outputs[name] = np.zeros(len(beat_times), dtype=float)

    return outputs["kick"], outputs["bass"], outputs["clap"], outputs["hihat"]
