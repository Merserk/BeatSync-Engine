#!/usr/bin/env python3
"""Stage 1: load-facing beat grid detection."""

from typing import Tuple
import librosa
import numpy as np

from . import AutoWaveConfig
from . import _normalize, _smooth, _to_float

def detect_master_beat_grid(y_percussive: np.ndarray, sr: int, cfg: AutoWaveConfig) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    onset_env = librosa.onset.onset_strength(
        y=y_percussive,
        sr=sr,
        hop_length=cfg.hop_length,
        aggregate=np.median,
    )
    onset_env = _normalize(_smooth(onset_env, 3))

    try:
        tempo_raw, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=cfg.hop_length,
            units="frames",
            start_bpm=120,
            tightness=120,
            trim=False,
        )
    except TypeError:
        tempo_raw, beat_frames = librosa.beat.beat_track(
            y=y_percussive,
            sr=sr,
            hop_length=cfg.hop_length,
            units="frames",
            start_bpm=120,
            tightness=120,
        )

    tempo = _to_float(tempo_raw, 120.0)
    beat_frames = np.asarray(beat_frames, dtype=int)

    if beat_frames.size < 2:
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=cfg.hop_length,
            backtrack=True,
            wait=4,
        )
        beat_frames = np.asarray(onset_frames, dtype=int)
        tempo = 120.0

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=cfg.hop_length)
    beat_times = np.asarray(beat_times, dtype=float)
    valid = np.isfinite(beat_times) & (beat_times >= 0.0)
    return beat_times[valid], tempo, beat_frames[valid], onset_env
