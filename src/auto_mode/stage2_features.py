#!/usr/bin/env python3
"""Stage 2: beat-synchronous energy wave and rhythm feature extraction."""

from typing import Dict, Tuple
import librosa
import numpy as np

from gpu_cpu_utils import GPU_AVAILABLE, cp
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

    is_phrase_anchor = np.zeros(len(beat_times), dtype=bool)
    is_bar_anchor = np.zeros(len(beat_times), dtype=bool)
    is_phrase_anchor[::max(1, cfg.phrase_beats)] = True
    is_bar_anchor[::max(1, cfg.bar_beats)] = True

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


def analyze_rhythm_bands(y: np.ndarray, sr: int, beat_times: np.ndarray,
                         cfg: AutoWaveConfig, use_gpu: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Beat-level kick/bass/clap/hihat strength."""
    stft = librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=cfg.n_fft)

    use_cupy = bool(use_gpu and GPU_AVAILABLE and cp is not None)
    xp = cp if use_cupy else np

    if use_cupy:
        try:
            stft_x = cp.asarray(stft)
            freqs_x = cp.asarray(freqs)
        except Exception:
            use_cupy = False
            xp = np
            stft_x = stft
            freqs_x = freqs
    else:
        stft_x = stft
        freqs_x = freqs

    magnitude = xp.abs(stft_x)
    bands = {
        "kick": (freqs_x >= 35) & (freqs_x <= 145),
        "bass": (freqs_x >= 35) & (freqs_x <= 220),
        "clap": (freqs_x >= 150) & (freqs_x <= 4200),
        "hihat": freqs_x >= 4200,
    }

    frame_times = librosa.frames_to_time(np.arange(magnitude.shape[1]), sr=sr, hop_length=cfg.hop_length)
    outputs: Dict[str, np.ndarray] = {}

    for name, mask in bands.items():
        try:
            curve = xp.sum(magnitude[mask, :], axis=0)
            if use_cupy:
                curve = cp.asnumpy(curve)
            curve = _normalize(_smooth(np.asarray(curve, dtype=float), 3))
            outputs[name] = np.interp(beat_times, frame_times, curve, left=float(curve[0]), right=float(curve[-1]))
        except Exception:
            outputs[name] = np.zeros(len(beat_times), dtype=float)

    return outputs["kick"], outputs["bass"], outputs["clap"], outputs["hihat"]
