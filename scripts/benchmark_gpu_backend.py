"""Measure the actual analysis functions before and after a GPU backend change.

Decode and synthesize the inputs once, outside timing. Every timed call includes
host/device transfers and result retrieval. No Qwen inference or video export.
"""

import argparse
import contextlib
import io
import json
from pathlib import Path
import statistics
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / "output" / "gpu_backend_benchmark"
    out.mkdir(parents=True, exist_ok=True)
    init_started = time.perf_counter()
    import cv2
    import soundfile as sf
    import gpu_cpu_utils as gpu
    from auto_mode import AutoWaveConfig, analyze_beats_auto
    from auto_mode.stage2_features import analyze_rhythm_bands
    from video_analysis import _measure_frames_gpu, _measure_frame_samples

    assert gpu.GPU_AVAILABLE, gpu.GPU_INFO
    if args.label.lower() not in gpu.GPU_INFO["backend"].lower():
        raise ValueError(f"Requested label {args.label!r} does not match the active backend {gpu.GPU_INFO['backend']!r}")
    init_seconds = time.perf_counter() - init_started
    inputs_path = out / "inputs.npz"
    if inputs_path.exists():
        data = np.load(inputs_path)
        frames = data["frames"]
        audio = data["audio"]
        beat_times = data["beat_times"]
        sample_times = data["sample_times"]
        sr = int(data["sr"])
    else:
        video = next(root.glob("*.mkv"))
        cap = cv2.VideoCapture(str(video))
        decoded = []
        sample_times = np.linspace(119, 121, 6)
        try:
            for position in sample_times:
                cap.set(cv2.CAP_PROP_POS_MSEC, position * 1000)
                ok, frame = cap.read()
                assert ok
                height = round(frame.shape[0] * 360 / frame.shape[1])
                decoded.append(cv2.resize(frame, (360, height), interpolation=cv2.INTER_AREA))
        finally:
            cap.release()
        frames = np.stack(decoded)
        sr = 22050
        duration = 60.0
        t = np.arange(round(sr * duration), dtype=np.float64) / sr
        audio = (0.07 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        rng = np.random.default_rng(4271)
        for index, beat in enumerate(np.arange(0.25, duration - 0.1, 0.5)):
            start = round(beat * sr)
            pulse_t = np.arange(round(0.08 * sr)) / sr
            pulse = 0.6 * np.sin(2 * np.pi * 70 * pulse_t) * np.exp(-55 * pulse_t)
            if index % 2:
                pulse += 0.12 * rng.standard_normal(len(pulse_t)) * np.exp(-45 * pulse_t)
            audio[start:start + len(pulse_t)] += pulse.astype(np.float32)
        beat_times = np.arange(0.25, duration - 0.1, 0.5)
        np.savez_compressed(inputs_path, frames=frames, audio=audio, beat_times=beat_times, sample_times=sample_times, sr=sr)
    audio_path = out / "audio_60s.wav"
    if not audio_path.exists():
        sf.write(audio_path, audio, sr, subtype="FLOAT")
    cfg = AutoWaveConfig()
    frame_list = list(frames)

    def full_audio():
        with contextlib.redirect_stdout(io.StringIO()):
            return analyze_beats_auto(str(audio_path), use_gpu=True, enable_video_analysis=False, enable_qwen_semantics=False)

    cases = [
        ("video_gpu_metrics", lambda: _measure_frames_gpu(frame_list, sample_times, 119.0, 2.0), args.repeats),
        ("video_scoring_with_text", lambda: _measure_frame_samples(frame_list, sample_times, 119.0, 2.0, True), args.repeats),
        ("audio_rhythm_60s", lambda: analyze_rhythm_bands(audio, sr, beat_times, cfg, True), args.repeats),
        ("full_audio_analysis_60s", full_audio, 3),
    ]
    report = {"backend": args.label, "device": gpu.GPU_INFO, "initialization_seconds": init_seconds, "video_shape": list(frames.shape), "audio_seconds": len(audio) / sr, "sample_rate": sr, "measurements": {}}
    reference = {}
    for name, function, repeats in cases:
        started = time.perf_counter()
        result = function()
        gpu.clear_gpu_memory()
        cold_ms = (time.perf_counter() - started) * 1000
        for _ in range(2):
            function()
            gpu.clear_gpu_memory()
        times = []
        for _ in range(repeats):
            started = time.perf_counter()
            result = function()
            gpu.clear_gpu_memory()
            times.append((time.perf_counter() - started) * 1000)
        report["measurements"][name] = {"first_call_ms": cold_ms, "median_ms": statistics.median(times), "minimum_ms": min(times), "p95_ms": float(np.percentile(times, 95)), "repeats": repeats, "samples_ms": times}
        if name.startswith("video"):
            reference[name] = result
        elif name == "audio_rhythm_60s":
            reference[name] = [np.asarray(x).tolist() for x in result]
            cpu_result = analyze_rhythm_bands(audio, sr, beat_times, cfg, False)
            np.testing.assert_allclose(result, cpu_result, rtol=2e-4, atol=2e-5)
        else:
            reference[name] = {"cuts": result[0].tolist(), "tempo": result[1]["tempo"], "beat_times": result[1]["times"].tolist()}
        print(name, f"median {statistics.median(times):.3f} ms; first {cold_ms:.3f} ms", flush=True)
    label = args.label.lower()
    report["outputs"] = reference
    (out / (label + ".json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved", out / (label + ".json"), flush=True)


if __name__ == "__main__":
    main()
