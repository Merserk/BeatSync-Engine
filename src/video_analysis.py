#!/usr/bin/env python3
"""Video source analysis for Auto Mode.

This module builds a reusable library of source moments. FFmpeg/OpenCV provide
fast deterministic signals for every candidate; Qwen3.5 optionally adds
semantic editing tags for the best candidates.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Sequence

import cv2
import numpy as np

from clip_uniqueness import frame_fingerprints
from gpu_cpu_utils import GPU_AVAILABLE, frame_statistics
from gpu_video import get_gpu_video_path, iter_sampled_frames, decoded_analysis_frames
from ffmpeg_processing import (
    detect_video_scene_changes,
    get_video_duration,
    get_video_fps,
    get_video_resolution,
)
from logger import ROOT_DIR, setup_environment


setup_environment()

ANALYSIS_VERSION = "auto_av_analysis_v15_unique_footage"
DEFAULT_QWEN_MODEL_DIR = os.path.join(ROOT_DIR, "bin", "models")
DEFAULT_QWEN_GGUF_MODEL = os.path.join(DEFAULT_QWEN_MODEL_DIR, "Qwen3.5-2B-Q8_0.gguf")
DEFAULT_QWEN_MMPROJ_MODEL = os.path.join(DEFAULT_QWEN_MODEL_DIR, "mmproj-F16.gguf")
DEFAULT_LLAMA_CPP_DIR = os.path.join(ROOT_DIR, "bin", "llama-bin-win-vulkan-x64")
VIDEO_ANALYSIS_CACHE_DIR = os.path.join(ROOT_DIR, "input", "video_analysis_cache")
_LLAMA_VERSION_TOKENS: Dict[str, str] = {}


def _clamp(value: Any, lo: float = 0.0, hi: float = 1.0, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        v = default
    if not math.isfinite(v):
        v = default
    return max(lo, min(hi, v))


def _safe_name(path: str) -> str:
    return os.path.basename(path) or path


def _hash_text(text: str, length: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:length]


def _resolve_qwen_backend_paths(qwen_model_path: str | None) -> Dict[str, str]:
    llama_dir = os.environ.get("BEATSYNC_QWEN_LLAMA_DIR", DEFAULT_LLAMA_CPP_DIR)
    model_override = os.environ.get("BEATSYNC_QWEN_LLAMA_MODEL")
    mmproj_override = os.environ.get("BEATSYNC_QWEN_LLAMA_MMPROJ")

    requested = os.path.abspath(qwen_model_path) if qwen_model_path else ""
    if model_override:
        model_path = os.path.abspath(model_override)
    elif requested and os.path.isfile(requested) and requested.lower().endswith(".gguf"):
        model_path = requested
    else:
        candidates = []
        if requested:
            candidates.extend([
                os.path.join(requested, os.path.basename(DEFAULT_QWEN_GGUF_MODEL)),
                os.path.join(os.path.dirname(requested), os.path.basename(DEFAULT_QWEN_GGUF_MODEL)),
            ])
        candidates.append(DEFAULT_QWEN_GGUF_MODEL)
        model_path = next((path for path in candidates if os.path.exists(path)), DEFAULT_QWEN_GGUF_MODEL)

    if mmproj_override:
        mmproj_path = os.path.abspath(mmproj_override)
    else:
        candidates = [
            os.path.join(os.path.dirname(model_path), os.path.basename(DEFAULT_QWEN_MMPROJ_MODEL)),
            DEFAULT_QWEN_MMPROJ_MODEL,
        ]
        mmproj_path = next((path for path in candidates if os.path.exists(path)), DEFAULT_QWEN_MMPROJ_MODEL)

    return {
        "llama_dir": os.path.abspath(llama_dir),
        "server": os.path.abspath(os.path.join(llama_dir, "llama-server.exe")),
        "mtmd": os.path.abspath(os.path.join(llama_dir, "llama-mtmd-cli.exe")),
        "model": os.path.abspath(model_path),
        "mmproj": os.path.abspath(mmproj_path),
    }


def _qwen_backend_available(qwen_model_path: str | None) -> bool:
    paths = _resolve_qwen_backend_paths(qwen_model_path)
    return all(os.path.exists(paths[key]) for key in ["server", "mtmd", "model", "mmproj"])


def _qwen_backend_model_path(qwen_model_path: str | None) -> str:
    return _resolve_qwen_backend_paths(qwen_model_path)["model"]


def _path_signature_token(path: str) -> str:
    try:
        stat = os.stat(path)
        return f"{os.path.basename(path)}:{stat.st_size}:{int(stat.st_mtime)}"
    except OSError:
        return f"{os.path.basename(path)}:missing"


def _llama_version_token(llama_dir: str) -> str:
    llama_dir = os.path.abspath(llama_dir)
    cached = _LLAMA_VERSION_TOKENS.get(llama_dir)
    if cached:
        return cached

    mtmd = os.path.join(llama_dir, "llama-mtmd-cli.exe")
    token = _path_signature_token(mtmd)
    if os.path.exists(mtmd):
        env = os.environ.copy()
        env["PATH"] = llama_dir + os.pathsep + env.get("PATH", "")
        try:
            result = subprocess.run(
                [mtmd, "--version"],
                cwd=llama_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
                env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            version = ((result.stdout or "") + (result.stderr or "")).strip()
            if version:
                token = version.splitlines()[0][:120]
        except Exception:
            pass
    _LLAMA_VERSION_TOKENS[llama_dir] = token
    return token


def _qwen_backend_signature_token(qwen_model_path: str | None) -> str:
    paths = _resolve_qwen_backend_paths(qwen_model_path)
    required = ["server", "mtmd", "model", "mmproj"]
    if not all(os.path.exists(paths[key]) for key in required):
        return "ai_missing"
    raw = "|".join([
        "llama_vulkan",
        _path_signature_token(paths["model"]),
        _path_signature_token(paths["mmproj"]),
        _path_signature_token(paths["server"]),
        _path_signature_token(paths["mtmd"]),
        _llama_version_token(paths["llama_dir"]),
    ])
    return "ai_" + _hash_text(raw, length=20)


def _video_signature(video_file: str, enable_ai: bool, qwen_model_path: str | None) -> str:
    stat = os.stat(video_file)
    model_token = "no_ai"
    if enable_ai:
        model_token = _qwen_backend_signature_token(qwen_model_path)
    raw = "|".join([
        ANALYSIS_VERSION,
        os.path.abspath(video_file),
        str(stat.st_size),
        str(stat.st_mtime_ns),
        model_token,
        str(_qwen_max_windows()) if enable_ai else '0',
    ])
    return _hash_text(raw, length=24)


def _cache_path(video_file: str, enable_ai: bool, qwen_model_path: str | None) -> str:
    os.makedirs(VIDEO_ANALYSIS_CACHE_DIR, exist_ok=True)
    name = os.path.splitext(_safe_name(video_file))[0]
    return os.path.join(
        VIDEO_ANALYSIS_CACHE_DIR,
        f"{_hash_text(name, 8)}_{_video_signature(video_file, enable_ai, qwen_model_path)}.json",
    )


def _load_cache(path: str, require_ai: bool = False) -> Dict | None:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("analysis_version") == ANALYSIS_VERSION:
                if require_ai:
                    timings = data.get('timings') or {}
                    if (not data.get('ai_enabled') or data.get('ai_complete') is False or
                            ('qwen_tag_count' in timings and not timings['qwen_tag_count'])):
                        return None
                return data
    except Exception as e:
        print(f"   Warning: could not read video analysis cache: {e}")
    return None


def _save_cache(path: str, data: Dict) -> None:
    try:
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"   Warning: could not write video analysis cache: {e}")


def _fmt_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.1f}s"


def _env_int(name: str, default: int, lo: int = 1, hi: int | None = None) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    if hi is None:
        hi = max(lo, value)
    return max(lo, min(hi, value))


def _video_analysis_workers(video_count: int) -> int:
    # Finish each source's scan and scoring before opening the next. All AI
    # tagging follows this pass in one shared worker, with one active source.
    # Keep a single active source regardless of CPU count or inherited overrides.
    return 1


def _candidate_metric_workers(window_count: int, use_gpu: bool) -> int:
    if use_gpu or window_count < 80:
        return 1
    cpu = os.cpu_count() or 4
    default_workers = min(4, max(1, cpu // 4), window_count)
    return _env_int("BEATSYNC_CANDIDATE_METRIC_WORKERS", default_workers, lo=1, hi=max(1, window_count))


def _opencv_decode_threads() -> int:
    cpu = os.cpu_count() or 4
    default_threads = min(8, max(2, cpu // 2))
    return _env_int("BEATSYNC_OPENCV_FFMPEG_THREADS", default_threads, lo=1, hi=max(1, cpu))


def _open_video_capture(video_file: str) -> cv2.VideoCapture:
    """Open video with FFmpeg decoder threads configured for OpenCV.

    This keeps the same sampled frames and scoring formulas, but lets FFmpeg use
    more CPU cores while OpenCV decodes/seeks source frames.
    """
    threads = _opencv_decode_threads()
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", f"threads;{threads}")
    try:
        cv2.setNumThreads(threads)
    except Exception:
        pass
    return cv2.VideoCapture(video_file)


def analyze_video_sources(
    video_files: Sequence[str],
    audio_profile: Dict | None = None,
    use_gpu: bool = True,
    enable_ai: bool = True,
    qwen_model_path: str | None = None,
) -> Dict:
    """Analyze all source videos and return candidate moments for Auto Mode."""
    total_started = time.perf_counter()
    requested_qwen_model_path = qwen_model_path or DEFAULT_QWEN_MODEL_DIR
    qwen_model_path = _qwen_backend_model_path(requested_qwen_model_path)
    existing = [os.path.abspath(p) for p in video_files if p and os.path.exists(p)]
    if not existing:
        return {
            "analysis_version": ANALYSIS_VERSION,
            "videos": [],
            "candidates": [],
            "ai_enabled": False,
            "summary": "No source videos available for analysis.",
        }

    ai_available = bool(enable_ai and _qwen_max_windows() > 0 and _qwen_backend_available(requested_qwen_model_path))
    if ai_available:
        qwen_model_path = _qwen_backend_model_path(requested_qwen_model_path)
    print("\n   VIDEO ANALYSIS - Auto Mode visual library")
    print(f"   Source videos: {len(existing)}")
    print(f"   Qwen semantic tags: {'enabled' if ai_available else 'disabled/fallback'}")

    results_by_index: Dict[int, Dict] = {}
    jobs: List[Dict] = []
    cache_hits = 0
    visual_cache_hits = 0
    scanned_count = 0

    for idx, video_file in enumerate(existing, 1):
        cache_file = _cache_path(video_file, ai_available, qwen_model_path)
        cached = _load_cache(cache_file, require_ai=ai_available)
        if cached:
            cache_hits += 1
            print(f"   Reusing cached visual analysis {idx}/{len(existing)}: {_safe_name(video_file)}")
            results_by_index[idx] = cached
        else:
            visual_cache_file = _cache_path(video_file, False, None)
            if ai_available:
                visual = _load_cache(visual_cache_file)
                if visual:
                    # Older non-AI caches were sorted by editorial score. Restore
                    # scan order before selecting the same diverse Qwen samples.
                    visual['candidates'].sort(key=lambda c: (float(c.get('start', 0)), float(c.get('end', 0)), str(c.get('id', ''))))
                    visual.update(ai_enabled=False, ai_deferred=True, ai_complete=False)
                    results_by_index[idx] = visual
                    visual_cache_hits += 1
                    print(f"   Reusing scan/scoring cache {idx}/{len(existing)}: {_safe_name(video_file)}; Qwen pending")
            jobs.append({"index": idx, "video_file": video_file, "cache_file": cache_file,
                         "visual_cache_file": visual_cache_file})

    workers = _video_analysis_workers(len(jobs))
    if jobs:
        print("   Phase 1: scan and score source videos one by one (Qwen starts after all scans)")
        print("   Visual analysis workers: 1 (serial)")
        for job in jobs:
            if job['index'] in results_by_index:
                continue
            video_data = _analyze_single_video(
                job['video_file'], use_gpu, ai_available, qwen_model_path,
                audio_profile or {}, True, job['index'], len(existing))
            results_by_index[job['index']] = video_data
            scanned_count += 1
            # Save the unmerged visual results now. A failed AI batch, new model
            # or larger AI budget can reuse the scan without blending old tags
            # into the candidate scores a second time.
            _save_cache(job['visual_cache_file'], dict(video_data, ai_enabled=False,
                                                      ai_deferred=False, ai_complete=False))

    # One sequential visual pass followed by one shared model session. Legacy
    # concurrency/batch overrides cannot reintroduce per-video model loading.
    deferred_jobs = [
        job for job in jobs
        if results_by_index.get(job["index"], {}).get("ai_deferred") and ai_available
    ]
    qwen_run_stats = {}
    if deferred_jobs:
        print(f"   Phase 1 complete; phase 2: shared Qwen session for {len(deferred_jobs)} uncached video(s)")
        qwen_run_stats = _complete_deferred_qwen_batch(
            video_items=[(job, results_by_index[job["index"]]) for job in deferred_jobs],
            use_gpu=use_gpu,
            qwen_model_path=qwen_model_path,
            audio_profile=audio_profile or {},
            total_video_count=len(existing),
        ) or {}

    for job in jobs:
        video_data = results_by_index.get(job["index"])
        if video_data:
            video_data['candidates'].sort(key=lambda c: c.get('editorial_score', 0.), reverse=True)
            _save_cache(job["cache_file"], video_data)

    videos: List[Dict] = [results_by_index[i] for i in range(1, len(existing) + 1) if i in results_by_index]
    all_candidates: List[Dict] = []
    for video_data in videos:
        all_candidates.extend(video_data.get("candidates", []))
    highlight_pool = _build_highlight_pool(all_candidates)

    if not all_candidates:
        summary = "No usable candidate moments were found; renderer will use fallback sampling."
    else:
        action_avg = float(np.mean([c.get("action_score", 0.0) for c in all_candidates]))
        beauty_avg = float(np.mean([c.get("beauty_score", 0.0) for c in all_candidates]))
        quality_avg = float(np.mean([c.get("quality_score", 0.0) for c in all_candidates]))
        summary = (
            f"{len(all_candidates)} visual moments, "
            f"action={action_avg:.2f}, beauty={beauty_avg:.2f}, quality={quality_avg:.2f}"
        )

    total_elapsed = time.perf_counter() - total_started
    qwen_tag_count = sum(int((v.get("timings") or {}).get("qwen_tag_count", 0)) for v in videos)
    qwen_frame_count = sum(int((v.get("timings") or {}).get("qwen_frame_count", 0)) for v in videos)
    qwen_seconds = sum(float((v.get("timings") or {}).get("qwen_seconds", 0.0)) for v in videos)
    qwen_inference_seconds = sum(float((v.get("timings") or {}).get("qwen_inference_seconds", 0.0)) for v in videos)
    qwen_model_id = next(
        (
            str((v.get("timings") or {}).get("qwen_model_id"))
            for v in videos
            if (v.get("timings") or {}).get("qwen_model_id")
        ),
        "",
    )
    qwen_concurrency = next(
        (
            int((v.get("timings") or {}).get("qwen_concurrency"))
            for v in videos
            if (v.get("timings") or {}).get("qwen_concurrency")
        ),
        0,
    )
    qwen_peak_vram_gb = max(
        [float((v.get("timings") or {}).get("qwen_peak_vram_gb") or 0.0) for v in videos] or [0.0]
    )
    print(
        f"   Visual library ready: {summary} "
        f"[highlight pool {len(highlight_pool)}, total {_fmt_seconds(total_elapsed)}, "
        f"cache hits {cache_hits}/{len(existing)}]"
    )
    return {
        "analysis_version": ANALYSIS_VERSION,
        "videos": videos,
        "candidates": all_candidates,
        "highlight_pool": highlight_pool,
        "highlight_pool_size": len(highlight_pool),
        "ai_enabled": ai_available,
        "qwen_model_path": qwen_model_path if ai_available else None,
        "summary": summary,
        "analysis_seconds": total_elapsed,
        "cache_hits": cache_hits,
        "visual_cache_hits": visual_cache_hits,
        "visual_scanned_count": scanned_count,
        "pipeline_order": "scan_all_then_qwen",
        "qwen_worker_count": int(qwen_run_stats.get('worker_count', 0)),
        "qwen_server_start_count": int(qwen_run_stats.get('server_start_count', 0)),
        "qwen_run_seconds": float(qwen_run_stats.get('seconds', 0.)),
        "source_count": len(existing),
        "worker_count": workers,
        "qwen_tag_count": qwen_tag_count,
        "qwen_frame_count": qwen_frame_count,
        "qwen_seconds": qwen_seconds,
        "qwen_inference_seconds": qwen_inference_seconds,
        "qwen_model_id": qwen_model_id,
        "qwen_concurrency": qwen_concurrency,
        "qwen_peak_vram_gb": qwen_peak_vram_gb,
    }


def _analyze_single_video(
    video_file: str,
    use_gpu: bool,
    enable_ai: bool,
    qwen_model_path: str,
    audio_profile: Dict,
    defer_ai: bool = False,
    index: int | None = None,
    total: int | None = None,
) -> Dict:
    started = time.perf_counter()
    timings: Dict[str, float] = {}
    name = _safe_name(video_file)
    prefix = f"{index}/{total} " if index and total else ""
    print(f"   Analyzing video {prefix}{name}")

    step_started = time.perf_counter()
    duration = max(0.0, float(get_video_duration(video_file)))
    fps = max(1.0, float(get_video_fps(video_file)))
    width, height = get_video_resolution(video_file)
    timings["metadata_seconds"] = time.perf_counter() - step_started
    print(
        f"      Metadata: {duration:.1f}s, {fps:.3g} fps, "
        f"{int(width)}x{int(height)} [{_fmt_seconds(timings['metadata_seconds'])}]"
    )

    analysis_file = video_file
    gpu_analysis = None
    if use_gpu and os.environ.get('BEATSYNC_GPU_SINGLE_PASS', '1') != '0':
        try:
            tick = time.perf_counter()
            with decoded_analysis_frames(video_file) as frame_cache:
                timings['scene_detection_seconds'] = time.perf_counter() - tick
                timings['analysis_decode_backend'] = frame_cache.backend
                timings['gpu_analysis_sample_frames'] = frame_cache.count
                timings['gpu_analysis_sample_fps'] = frame_cache.analysis_fps
                scene_changes_gpu = frame_cache.scene_changes
                windows_gpu = _make_candidate_windows(_build_boundaries(scene_changes_gpu, duration), duration)
                tick = time.perf_counter()
                metrics_gpu = _measure_cached_windows(frame_cache, fps, windows_gpu,
                                                      _use_gpu_candidate_metrics(use_gpu))
                timings['candidate_scoring_seconds'] = time.perf_counter() - tick
                timings['color_conversion_backend'] = frame_cache.color_conversion_backend
                timings['frame_sampling_backend'] = 'GPU analysis samples at 6 fps'
                gpu_count = sum(m.get('metrics_backend') == 'Taichi Vulkan' for m in metrics_gpu)
                timings['gpu_scored_windows'] = gpu_count
                timings['candidate_metrics_backend'] = ('Taichi Vulkan' if gpu_count == len(metrics_gpu)
                                                        else 'CPU' if gpu_count == 0 else 'Mixed GPU/CPU fallback')
                gpu_analysis = scene_changes_gpu, windows_gpu, metrics_gpu
                print(f"      One-pass GPU source analysis: {frame_cache.backend}, {frame_cache.count} frames, "
                      f"{len(scene_changes_gpu)} cuts [{_fmt_seconds(timings['scene_detection_seconds'])}]; "
                      f"scoring {_fmt_seconds(timings['candidate_scoring_seconds'])}")
        except Exception as exc:
            timings['gpu_single_pass_error'] = str(exc)[-900:]
            print(f'      One-pass GPU analysis unavailable; using direct analysis: {str(exc)[-900:]}')

    scene_use_gpu = _use_gpu_scene_detection(use_gpu) and analysis_file == video_file
    if use_gpu and not scene_use_gpu and analysis_file == video_file:
        print("      Scene GPU acceleration disabled by BEATSYNC_GPU_SCENE_DETECTION")

    step_started = time.perf_counter()
    scene_changes = gpu_analysis[0] if gpu_analysis is not None else detect_video_scene_changes(
        analysis_file,
        threshold=0.27,
        use_gpu=scene_use_gpu,
        analysis_fps=6.0,
        analysis_width=768,
    )
    if gpu_analysis is None:
        timings["scene_detection_seconds"] = time.perf_counter() - step_started
    print(
        f"      ⏱ Scene detection total: {_fmt_seconds(timings['scene_detection_seconds'])} "
        f"({len(scene_changes)} scene cuts)"
    )

    step_started = time.perf_counter()
    boundaries = _build_boundaries(scene_changes, duration)
    windows = gpu_analysis[1] if gpu_analysis is not None else _make_candidate_windows(boundaries, duration)
    timings["window_build_seconds"] = time.perf_counter() - step_started
    print(
        f"      Candidate windows: {len(windows)} from {len(boundaries)} boundaries "
        f"[{_fmt_seconds(timings['window_build_seconds'])}]"
    )

    cap = _open_video_capture(analysis_file) if gpu_analysis is None else None
    candidates: List[Dict] = []
    if gpu_analysis is not None or cap.isOpened():
        gpu_candidate_metrics = _use_gpu_candidate_metrics(use_gpu)
        if gpu_candidate_metrics:
            print("      Candidate scoring: GPU decode/resize + Taichi Vulkan metrics")
        else:
            print("      Candidate scoring: CPU metrics")
        step_started = time.perf_counter()
        window_metrics = gpu_analysis[2] if gpu_analysis is not None else _measure_windows(cap, fps, windows, use_gpu=gpu_candidate_metrics,
                                         video_file=video_file if use_gpu and analysis_file == video_file else None,
                                         timings=timings)
        if analysis_file != video_file:
            timings.update(frame_sampling_backend='CPU low-resolution GPU proxy',
                           candidate_metrics_backend='Taichi Vulkan' if gpu_candidate_metrics else 'CPU')
        if gpu_analysis is None:
            timings["candidate_scoring_seconds"] = time.perf_counter() - step_started
        valid_metrics = 0
        for i, (window, metrics) in enumerate(zip(windows, window_metrics)):
            if not metrics:
                continue
            valid_metrics += 1
            candidate = _build_candidate(video_file, name, duration, i, window, metrics)
            candidates.append(candidate)
        if cap is not None:
            cap.release()
        print(
            f"      ⏱ Candidate scoring total: {_fmt_seconds(timings['candidate_scoring_seconds'])} "
            f"({valid_metrics}/{len(windows)} windows usable)"
        )
    else:
        print(f"      Warning: OpenCV could not open {name}; candidate analysis skipped.")

    qwen_seconds = 0.0
    if enable_ai and candidates and not defer_ai:
        try:
            step_started = time.perf_counter()
            qwen_info = _annotate_candidates_with_qwen(
                video_file=video_file,
                fps=fps,
                candidates=candidates,
                qwen_model_path=qwen_model_path,
                use_gpu=use_gpu,
                audio_profile=audio_profile,
                analysis_video_file=analysis_file,
            )
            qwen_seconds = time.perf_counter() - step_started
            timings.update(qwen_info or {})
            timings["qwen_seconds"] = qwen_seconds
            print(f"      ⏱ Qwen semantic analysis total: {_fmt_seconds(qwen_seconds)}")
        except Exception as e:
            print(f"      Warning: Qwen semantic analysis failed for {name}: {e}")
    elif enable_ai and candidates and defer_ai:
        timings["qwen_seconds"] = 0.0
        print("      Qwen semantic analysis: deferred until the visual pass completes")

    sort_started = time.perf_counter()
    if not defer_ai:
        candidates = sorted(candidates, key=lambda c: c.get("editorial_score", 0.0), reverse=True)
    timings["sort_seconds"] = time.perf_counter() - sort_started

    elapsed = time.perf_counter() - started
    timings["total_seconds"] = elapsed
    print(
        f"   Found {len(candidates)} usable visual moments in {_fmt_seconds(elapsed)} "
        f"(scene {_fmt_seconds(timings.get('scene_detection_seconds', 0))}, "
        f"scoring {_fmt_seconds(timings.get('candidate_scoring_seconds', 0))}, "
        f"qwen {_fmt_seconds(timings.get('qwen_seconds', 0))})"
    )

    return {
        "analysis_version": ANALYSIS_VERSION,
        "video_file": os.path.abspath(video_file),
        "analysis_video_file": os.path.abspath(analysis_file),
        "source_name": name,
        "duration": duration,
        "fps": fps,
        "width": int(width),
        "height": int(height),
        "scene_changes": scene_changes,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "analysis_seconds": elapsed,
        "timings": timings,
        "ai_enabled": bool(enable_ai and not defer_ai),
        "ai_deferred": bool(enable_ai and defer_ai),
    }


def _complete_deferred_qwen(
    video_data: Dict,
    use_gpu: bool,
    qwen_model_path: str,
    audio_profile: Dict,
    label: str = "",
) -> Dict:
    candidates = video_data.get("candidates") or []
    if not candidates:
        video_data["ai_deferred"] = False
        video_data["ai_enabled"] = False
        return

    name = _safe_name(video_data.get("video_file", video_data.get("source_name", "video")))
    print(f"   Running deferred Qwen semantic analysis {label}: {name}")
    started = time.perf_counter()
    qwen_info = {}
    try:
        qwen_info = _annotate_candidates_with_qwen(
            video_file=video_data["video_file"],
            fps=float(video_data.get("fps") or 24.0),
            candidates=candidates,
            qwen_model_path=qwen_model_path,
            use_gpu=use_gpu,
            audio_profile=audio_profile,
        )
    except Exception as e:
        print(f"      Warning: Qwen semantic analysis failed for {name}: {e}")

    qwen_seconds = time.perf_counter() - started
    candidates.sort(key=lambda c: c.get("editorial_score", 0.0), reverse=True)
    timings = video_data.setdefault("timings", {})
    timings.update(qwen_info or {})
    timings["qwen_seconds"] = qwen_seconds
    timings["total_seconds"] = float(timings.get("total_seconds", video_data.get("analysis_seconds", 0.0))) + qwen_seconds
    video_data["analysis_seconds"] = timings["total_seconds"]
    video_data["ai_deferred"] = False
    video_data["ai_enabled"] = True
    video_data["candidate_count"] = len(candidates)
    print(
        f"      ⏱ Deferred Qwen total: {_fmt_seconds(qwen_seconds)}; "
        f"video total now {_fmt_seconds(video_data['analysis_seconds'])}"
    )




def _qwen_max_windows() -> int:
    try:
        value = int(os.environ.get("BEATSYNC_QWEN_MAX_WINDOWS", "120"))
    except ValueError:
        value = 120
    return max(0, value)


def _complete_deferred_qwen_batch(
    video_items: Sequence[tuple[Dict, Dict]],
    use_gpu: bool,
    qwen_model_path: str,
    audio_profile: Dict,
    total_video_count: int,
) -> Dict:
    max_windows = _qwen_max_windows()
    if max_windows == 0:
        for _, video_data in video_items:
            video_data["ai_deferred"] = False
            video_data["ai_enabled"] = False
        print("   Qwen semantic analysis skipped (BEATSYNC_QWEN_MAX_WINDOWS=0).")
        return {}

    request_jobs: List[Dict] = []
    job_to_video: Dict[str, Dict] = {}
    selected_by_job: Dict[str, List[Dict]] = {}
    for job, video_data in video_items:
        candidates = video_data.get("candidates") or []
        ai_candidates = _select_ai_candidates(candidates, max_windows)
        idx = int(job.get("index") or 0)
        job_id = str(idx or len(request_jobs) + 1)
        name = _safe_name(video_data.get("video_file", video_data.get("source_name", "video")))
        print(
            f"   Qwen semantic analysis batch input {idx}/{total_video_count}: "
            f"{name} - {len(ai_candidates)} candidate moments"
        )
        if not ai_candidates:
            video_data["ai_deferred"] = False
            video_data["ai_enabled"] = False
            continue
        selected_by_job[job_id] = ai_candidates
        job_to_video[job_id] = video_data
        request_jobs.append({
            "job_id": job_id,
            "video_file": (video_data.get('analysis_video_file')
                           if os.path.isfile(video_data.get('analysis_video_file', '')) else video_data["video_file"]),
            "fps": float(video_data.get("fps") or 24.0),
            "candidates": [
                {
                    "id": c.get("id"),
                    "start": c.get("start"),
                    "end": c.get("end"),
                }
                for c in ai_candidates
            ],
        })

    if not request_jobs:
        return {}

    print(f"   Running one shared Qwen worker for {len(request_jobs)} video(s) (model stays loaded between sources)")
    batch_started = time.perf_counter()
    response = _run_qwen_worker_batch(
        jobs=request_jobs,
        qwen_model_path=qwen_model_path,
        use_gpu=use_gpu,
        audio_profile=audio_profile,
    )
    batch_seconds = time.perf_counter() - batch_started
    run_stats = {'worker_count': 1, 'server_start_count': int(response.get('server_start_count') or 0),
                 'seconds': batch_seconds}
    semantics_by_job = response.get("semantics_by_job") or {}
    timings_by_job = response.get("timings_by_job") or {}
    if not semantics_by_job:
        for video_data in job_to_video.values():
            video_data["ai_deferred"] = False
            video_data["ai_enabled"] = False
            video_data['ai_complete'] = False
        print("      Qwen llama.cpp returned no semantic response; AI analysis will retry on the next run.")
        print(f"      ⏱ Shared Qwen batch total: {_fmt_seconds(batch_seconds)}")
        return run_stats
    model_load_seconds = float(response.get("model_load_seconds") or 0.0)
    amortized_model = model_load_seconds / max(1, len(request_jobs))
    qwen_model_id = str(response.get("model_id") or "")
    qwen_concurrency = int(response.get("batch_size") or 0)
    qwen_peak_vram_gb = float(response.get("peak_vram_gb") or 0.0)

    for job_id, video_data in job_to_video.items():
        ai_candidates = selected_by_job.get(job_id, [])
        semantics = semantics_by_job.get(str(job_id), {})
        semantic_by_id = {str(k): v for k, v in semantics.items()} if isinstance(semantics, dict) else {}
        merged_count = 0
        for candidate in ai_candidates:
            semantic = semantic_by_id.get(str(candidate.get("id")))
            if semantic:
                _merge_semantic(candidate, semantic)
                merged_count += 1
        candidates = video_data.get("candidates") or []
        candidates.sort(key=lambda c: c.get("editorial_score", 0.0), reverse=True)
        timing = timings_by_job.get(str(job_id), {}) if isinstance(timings_by_job, dict) else {}
        qwen_seconds = (
            float(timing.get("prefetch_seconds") or 0.0)
            + float(timing.get("inference_seconds") or 0.0)
            + amortized_model
        )
        timings = video_data.setdefault("timings", {})
        timings["qwen_seconds"] = qwen_seconds
        timings["qwen_model_load_seconds_amortized"] = amortized_model
        timings["qwen_prefetch_seconds"] = float(timing.get("prefetch_seconds") or 0.0)
        timings["qwen_inference_seconds"] = float(timing.get("inference_seconds") or 0.0)
        timings["qwen_frame_count"] = int(timing.get("frame_count") or len(ai_candidates))
        timings["qwen_tag_count"] = int(timing.get("tag_count") or merged_count)
        timings['qwen_requested_frame_count'] = len(ai_candidates)
        timings["qwen_model_id"] = qwen_model_id
        timings["qwen_concurrency"] = qwen_concurrency
        timings["qwen_peak_vram_gb"] = qwen_peak_vram_gb
        timings["total_seconds"] = float(timings.get("total_seconds", video_data.get("analysis_seconds", 0.0))) + qwen_seconds
        video_data["analysis_seconds"] = timings["total_seconds"]
        video_data["ai_deferred"] = False
        video_data["ai_enabled"] = merged_count > 0
        video_data['ai_complete'] = merged_count == len(ai_candidates)
        video_data["candidate_count"] = len(candidates)
        print(
            f"      Qwen semantic tags merged: {merged_count}/{len(ai_candidates)} "
            f"for {_safe_name(video_data.get('video_file', 'video'))}"
        )
        if not video_data['ai_complete']:
            print('      Qwen tags incomplete; saved visual scan can be reused for the next AI attempt.')
        print(
            f"      ⏱ Shared-Qwen video cost: {_fmt_seconds(qwen_seconds)} "
            f"(prefetch {_fmt_seconds(timings['qwen_prefetch_seconds'])}, "
            f"inference {_fmt_seconds(timings['qwen_inference_seconds'])}, "
            f"model share {_fmt_seconds(amortized_model)})"
        )

    print(f"      ⏱ Shared Qwen batch total: {_fmt_seconds(batch_seconds)}")
    return run_stats


def _run_qwen_worker_batch(
    jobs: Sequence[Dict],
    qwen_model_path: str,
    use_gpu: bool,
    audio_profile: Dict,
) -> Dict:
    os.makedirs(VIDEO_ANALYSIS_CACHE_DIR, exist_ok=True)
    token = _hash_text(f"batch|{time.time()}|{len(jobs)}", 12)
    request_path = os.path.join(VIDEO_ANALYSIS_CACHE_DIR, f"qwen_batch_request_{token}.json")
    response_path = os.path.join(VIDEO_ANALYSIS_CACHE_DIR, f"qwen_batch_response_{token}.json")
    worker_path = os.path.join(ROOT_DIR, "src", "auto_mode", "stage5_qwen_scene_worker.py")
    request = {
        "jobs": list(jobs),
        "qwen_model_path": qwen_model_path,
        "use_gpu": bool(use_gpu),
        "audio_profile": audio_profile,
    }
    with open(request_path, "w", encoding="utf-8") as f:
        json.dump(request, f)

    env = _qwen_worker_environment()
    total_candidates = sum(len(job.get("candidates") or []) for job in jobs)
    timeout = max(1800, int(total_candidates * 75))
    process = None
    timer = None
    expired = threading.Event()

    def stop_worker():
        if process is None or process.poll() is not None:
            return
        # Stop only this worker's process tree, including its llama-server, so
        # a timeout cannot leave the model consuming VRAM in the background.
        if os.name == 'nt':
            try:
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
                               check=False, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            except Exception:
                pass
        if process.poll() is None:
            process.kill()

    def expire():
        expired.set()
        stop_worker()

    try:
        stderr_path = response_path + '.worker_stderr.log'
        with open(stderr_path, 'w+', encoding='utf8') as stderr:
            process = subprocess.Popen(
                [sys.executable, worker_path, '--request', request_path, '--response', response_path],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=stderr,
                text=True, encoding='utf8', errors='replace', env=env,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            timer = threading.Timer(timeout, expire)
            timer.daemon = True
            timer.start()
            for line in process.stdout:
                if line.strip():
                    print(f'      {line.rstrip()}', flush=True)
            code = process.wait(timeout=30)
            stderr.seek(0)
            error_tail = stderr.read().strip()[-2400:]
        if expired.is_set():
            print('      Qwen batch worker timed out; retaining completed tags and visual scans.')
        if code != 0:
            print(f"      Qwen batch worker failed: {error_tail}")
            if not os.path.isfile(response_path):
                return {}
            print('      Retaining completed video tags from the interrupted Qwen batch.')
        with open(response_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except subprocess.TimeoutExpired:
        print("      Qwen batch worker timed out; deterministic visual tags remain active.")
        return {}
    except Exception as e:
        print(f"      Qwen batch worker error: {e}")
        return {}
    finally:
        if timer:
            timer.cancel()
        stop_worker()
        if process:
            if process.poll() is None:
                process.wait(timeout=15)
            process.stdout.close()
        # Keep Qwen worker request/response files in video_analysis_cache for
        # reproducibility and debugging. The user explicitly wants this cache
        # folder to be preserved.
        pass

def _build_boundaries(scene_changes: Iterable[float], duration: float) -> List[float]:
    if duration <= 0:
        return [0.0]
    raw = [0.0] + [float(t) for t in scene_changes if 0.0 < float(t) < duration] + [duration]
    raw = sorted(set(round(t, 3) for t in raw))
    cleaned: List[float] = []
    for t in raw:
        if not cleaned or t - cleaned[-1] >= 0.35 or t in (0.0, duration):
            cleaned.append(t)
    if cleaned[-1] != duration:
        cleaned.append(duration)
    return cleaned


def _make_candidate_windows(boundaries: Sequence[float], duration: float) -> List[Dict]:
    windows: List[Dict] = []
    if duration <= 0:
        return windows

    max_window = 5.2
    min_window = 0.55
    fallback_step = 3.5

    if len(boundaries) < 2:
        starts = np.arange(0.0, max(duration - min_window, 0.0), fallback_step)
        return [
            {"start": float(s), "end": float(min(duration, s + max_window)), "kind": "fallback"}
            for s in starts
        ]

    for scene_idx in range(len(boundaries) - 1):
        start = float(boundaries[scene_idx])
        end = float(boundaries[scene_idx + 1])
        scene_duration = end - start
        if scene_duration < min_window:
            continue

        if scene_duration <= max_window:
            windows.append({"start": start, "end": end, "kind": "scene", "scene_index": scene_idx})
            continue

        chunks = max(1, int(math.ceil(scene_duration / max_window)))
        step = scene_duration / chunks
        for chunk_idx in range(chunks):
            chunk_start = start + chunk_idx * step
            chunk_end = min(end, chunk_start + max_window)
            if chunk_end - chunk_start >= min_window:
                windows.append({
                    "start": float(chunk_start),
                    "end": float(chunk_end),
                    "kind": "scene_chunk",
                    "scene_index": scene_idx,
                })

    return windows


def _measure_windows(cap: cv2.VideoCapture, fps: float, windows: Sequence[Dict],
                     use_gpu: bool = False, video_file: str | None = None,
                     timings: Dict | None = None) -> List[Dict]:
    plans = [_window_sample_plan(fps, float(w["start"]), float(w["end"])) for w in windows]
    targets = sorted({idx for plan in plans for idx in plan["frame_indices"]})
    if not targets:
        return [{} for _ in windows]

    frame_span = max(1, targets[-1] - targets[0] + 1)
    seek_ratio = frame_span / max(1, len(targets))
    sequential_limit = float(os.environ.get("BEATSYNC_SEQUENTIAL_SAMPLE_RATIO", "80"))
    use_sequential = (
        os.environ.get("BEATSYNC_SEQUENTIAL_WINDOW_SAMPLING", "1") != "0"
        and seek_ratio <= sequential_limit
    )

    def build_metric(plan: Dict, frame_map: Dict[int, np.ndarray]) -> Dict:
        frames = []
        sample_times = []
        for frame_idx, sample_time in zip(plan["frame_indices"], plan["sample_times"]):
            frame = frame_map.get(frame_idx)
            if frame is not None:
                frames.append(frame)
                sample_times.append(sample_time)
        return _measure_frame_samples(
            frames,
            np.asarray(sample_times, dtype=float),
            plan["start"],
            plan["duration"],
            use_gpu,
        )

    if use_sequential and video_file and os.environ.get('BEATSYNC_GPU_VIDEO_DECODE', '1') != '0':
        try:
            route = get_gpu_video_path(video_file)
            if route is not None:
                print(f'         GPU ordered sampling: {route.label}, {len(targets)} exact frames')
                # Release each frame once the last window using it completes.
                # Long sources therefore do not create gigabyte frame caches.
                references = {}
                completions = {}
                metrics = [{} for _ in plans]
                for i, plan in enumerate(plans):
                    for index in set(plan['frame_indices']):
                        references[index] = references.get(index, 0) + 1
                    if plan['frame_indices']:
                        completions.setdefault(max(plan['frame_indices']), []).append(i)
                frames = {}
                metric_seconds = 0.0
                started = time.perf_counter()
                for index, frame in iter_sampled_frames(video_file, targets, route=route):
                    frames[index] = frame
                    for i in completions.get(index, []):
                        tick = time.perf_counter()
                        metrics[i] = build_metric(plans[i], frames)
                        metric_seconds += time.perf_counter() - tick
                        for used in set(plans[i]['frame_indices']):
                            references[used] -= 1
                            if references[used] == 0:
                                frames.pop(used, None)
                elapsed = time.perf_counter() - started
                if timings is not None:
                    timings.update(frame_sampling_backend=route.label,
                                   candidate_metrics_backend='Taichi Vulkan' if use_gpu else 'CPU',
                                   frame_sampling_seconds=elapsed, metric_math_seconds=metric_seconds,
                                   sampled_frame_count=len(targets))
                print(f'         GPU sampling/scoring: {len(targets)} frames, {len(metrics)} windows '
                      f'[{_fmt_seconds(elapsed)}, metric math {_fmt_seconds(metric_seconds)}]')
                return metrics
        except Exception as exc:
            print(f'         GPU frame sampling failed; retrying CPU decode: {str(exc)[-900:]}')
            if timings is not None:
                timings['gpu_frame_sampling_error'] = str(exc)[-900:]
        if timings is not None:
            timings['frame_sampling_backend'] = 'CPU fallback'

    if use_sequential:
        print(
            f"         Ordered frame sampling: {len(targets)} target frames "
            f"(span/target={seek_ratio:.1f}, limit={sequential_limit:g})"
        )
        read_started = time.perf_counter()
        frame_map = _read_ordered_frames(cap, targets)
        read_elapsed = time.perf_counter() - read_started
        approx_ram_mb = sum(getattr(frame, "nbytes", 0) for frame in frame_map.values()) / (1024 * 1024)
        print(
            f"         Frame decode/cache: {len(frame_map)}/{len(targets)} frames "
            f"in RAM, ~{approx_ram_mb:.0f} MB [{_fmt_seconds(read_elapsed)}]"
        )

        metric_workers = _candidate_metric_workers(len(plans), use_gpu)
        metrics_started = time.perf_counter()
        if metric_workers > 1:
            print(f"         Metric CPU workers: {metric_workers}")
            with ThreadPoolExecutor(max_workers=metric_workers) as executor:
                metrics = list(executor.map(lambda plan: build_metric(plan, frame_map), plans))
        else:
            metrics = [build_metric(plan, frame_map) for plan in plans]
        metric_elapsed = time.perf_counter() - metrics_started
        usable = sum(1 for metric in metrics if metric)
        print(
            f"         Metric math: {usable}/{len(plans)} windows "
            f"[{_fmt_seconds(metric_elapsed)}, {usable / max(metric_elapsed, 1e-6):.1f} windows/s]"
        )
        return metrics

    print(
        f"         Random-seek sampling fallback: {len(targets)} target frames "
        f"(span/target={seek_ratio:.1f} > limit {sequential_limit:g})"
    )
    fallback_started = time.perf_counter()
    metrics = [
        _measure_window(cap, fps, float(w["start"]), float(w["end"]), use_gpu=use_gpu)
        for w in windows
    ]
    fallback_elapsed = time.perf_counter() - fallback_started
    usable = sum(1 for metric in metrics if metric)
    print(
        f"         Random-seek scoring: {usable}/{len(windows)} windows "
        f"[{_fmt_seconds(fallback_elapsed)}]"
    )
    return metrics


def _measure_cached_windows(frame_cache, fps, windows, use_gpu):
    """Batch Vulkan transfers/kernels and parallelize tiny CPU text checks."""
    metrics = []
    batch_size = 16
    gpu_metrics = use_gpu
    with ThreadPoolExecutor(max_workers=4) as text_workers:
        for first in range(0, len(windows), batch_size):
            items, stack = [], []
            plans = [_window_sample_plan(fps, window['start'], window['end'])
                     for window in windows[first:first + batch_size]]
            indices = sorted({index for plan in plans for index in plan['frame_indices']})
            indices = [index for index in indices if index < frame_cache.source_count]
            if indices:
                converted = frame_cache.read_bgr_batch(indices)
                frame_map = {index: _resize_for_analysis(frame, max_width=768)
                             for index, frame in zip(indices, converted)}
            else:
                frame_map = {}
            for plan in plans:
                frames, times = [], []
                for index in plan['frame_indices']:
                    frame = frame_map.get(index)
                    if frame is not None:
                        frames.append(frame)
                        times.append(frame_cache.source_index(index) / fps)
                items.append((frames, np.asarray(times), plan, len(stack)))
                stack.extend(frames)
            stats = None
            if stack and gpu_metrics:
                try:
                    stats = frame_statistics(np.stack(stack))
                except Exception as exc:
                    print(f'      Batched Vulkan metrics failed; using CPU metrics: {exc}')
                    gpu_metrics = False
            def measure(item):
                frames, times, plan, offset = item
                if not frames:
                    return {}
                if stats is None:
                    return _measure_frame_samples(frames, times, plan['start'], plan['duration'], False)
                result = _measure_frames_gpu(frames, times, plan['start'], plan['duration'],
                                             stats=stats[offset:offset + len(frames)])
                result['text_score'] = _text_overlay_score(frames)
                result['metrics_backend'] = 'Taichi Vulkan'
                result['visual_fingerprints'] = frame_fingerprints(frames)
                return result
            metrics.extend(text_workers.map(measure, items))
    return metrics


def _window_sample_plan(fps: float, start: float, end: float) -> Dict:
    duration = max(0.0, end - start)
    if duration <= 0.05:
        return {"start": start, "duration": duration, "sample_times": [], "frame_indices": []}

    sample_count = int(np.clip(math.ceil(duration * 1.4), 3, 8))
    sample_times = np.linspace(start + duration * 0.12, end - duration * 0.12, sample_count)
    frame_indices = [max(0, int(round(float(t) * fps))) for t in sample_times]
    return {
        "start": start,
        "duration": duration,
        "sample_times": sample_times,
        "frame_indices": frame_indices,
    }


def _read_ordered_frames(cap: cv2.VideoCapture, frame_indices: Sequence[int]) -> Dict[int, np.ndarray]:
    frames: Dict[int, np.ndarray] = {}
    if not frame_indices:
        return frames

    first = int(frame_indices[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    current = first

    for frame_idx in frame_indices:
        frame_idx = int(frame_idx)
        if frame_idx < current:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            current = frame_idx
        while current < frame_idx:
            if not cap.grab():
                break
            current += 1
        if current != frame_idx:
            continue
        ok, frame = cap.read()
        current += 1
        if not ok or frame is None:
            continue
        frames[frame_idx] = _resize_for_analysis(frame, max_width=768)
    return frames


def _measure_window(cap: cv2.VideoCapture, fps: float, start: float, end: float,
                    use_gpu: bool = False) -> Dict:
    plan = _window_sample_plan(fps, start, end)
    if plan["duration"] <= 0.05:
        return {}

    frames: List[np.ndarray] = []
    used_times: List[float] = []
    for frame_idx, sample_time in zip(plan["frame_indices"], plan["sample_times"]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = _resize_for_analysis(frame, max_width=768)
        frames.append(frame)
        used_times.append(float(sample_time))

    return _measure_frame_samples(frames, np.asarray(used_times, dtype=float), start, plan["duration"], use_gpu)


def _measure_frame_samples(frames: Sequence[np.ndarray], sample_times: np.ndarray,
                           start: float, duration: float, use_gpu: bool = False) -> Dict:
    if not frames:
        return {}

    text_score = _text_overlay_score(frames)
    if use_gpu and GPU_AVAILABLE:
        try:
            metrics = _measure_frames_gpu(frames, sample_times, start, duration)
            metrics["text_score"] = text_score
            metrics['metrics_backend'] = 'Taichi Vulkan'
            metrics['visual_fingerprints'] = frame_fingerprints(frames)
            return metrics
        except Exception as exc:
            # A driver/kernel failure must be visible; never report GPU work
            # while silently doing every subsequent window on the CPU.
            print(f'         Taichi frame metrics failed; using CPU metrics: {exc}')

    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    hsv_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for f in frames]

    brightness = float(np.mean([np.mean(g) / 255.0 for g in gray_frames]))
    contrast = float(np.mean([np.std(g) / 80.0 for g in gray_frames]))
    saturation = float(np.mean([np.mean(h[:, :, 1]) / 255.0 for h in hsv_frames]))
    blur_values = [cv2.Laplacian(g, cv2.CV_64F).var() for g in gray_frames]
    sharpness = _clamp(np.mean(blur_values) / 520.0)

    motion_values = []
    peak_offset = duration * 0.5
    for i in range(1, len(gray_frames)):
        diff = cv2.absdiff(gray_frames[i], gray_frames[i - 1])
        motion = float(np.mean(diff) / 42.0)
        motion_values.append(motion)
    if motion_values:
        motion = _clamp(np.mean(motion_values))
        peak_i = int(np.argmax(motion_values)) + 1
        peak_offset = float(sample_times[min(peak_i, len(sample_times) - 1)] - start)
    else:
        motion = 0.0

    colorfulness = _colorfulness(frames)
    darkness_penalty = _clamp((0.25 - brightness) / 0.25)
    blown_penalty = _clamp((brightness - 0.86) / 0.14)
    quality = _clamp(
        0.36 * sharpness
        + 0.22 * _clamp(contrast)
        + 0.18 * _clamp(saturation)
        + 0.14 * (1.0 - darkness_penalty)
        + 0.10 * (1.0 - blown_penalty)
    )

    return {
        "duration": duration,
        "brightness": _clamp(brightness),
        "contrast": _clamp(contrast),
        "saturation": _clamp(saturation),
        "sharpness": _clamp(sharpness),
        "motion": _clamp(motion),
        "colorfulness": _clamp(colorfulness),
        "quality_score": quality,
        "text_score": text_score,
        "peak_offset": _clamp(peak_offset, 0.0, duration, default=duration * 0.5),
        "metrics_backend": 'CPU',
        "visual_fingerprints": frame_fingerprints(frames),
    }


def _text_overlay_score(frames: Sequence[np.ndarray]) -> float:
    """Estimate readable on-screen text without an OCR dependency."""
    if not frames:
        return 0.0

    frame_scores: List[float] = []
    for frame in frames:
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        h, w = gray.shape[:2]
        if h < 40 or w < 80:
            continue

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 70, 170)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 2))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        line_count = 0
        weighted_area = 0.0
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw <= 0 or bh <= 0:
                continue
            rel_w = bw / float(w)
            rel_h = bh / float(h)
            aspect = bw / float(max(1, bh))
            if rel_w < 0.07 or rel_h < 0.012 or rel_h > 0.24 or aspect < 1.6:
                continue

            roi_gray = gray[y:y + bh, x:x + bw]
            roi_edges = edges[y:y + bh, x:x + bw]
            edge_density = float(np.count_nonzero(roi_edges)) / float(max(1, bw * bh))
            roi_contrast = float(np.std(roi_gray))
            if edge_density < 0.035 or roi_contrast < 18.0:
                continue

            vertical_position_weight = 1.0
            center_y = (y + bh * 0.5) / float(h)
            if center_y < 0.18 or center_y > 0.74:
                vertical_position_weight = 1.18

            line_count += 1
            weighted_area += min(0.075, rel_w * rel_h) * vertical_position_weight

        frame_scores.append(_clamp(0.18 * line_count + 5.8 * weighted_area))

    if not frame_scores:
        return 0.0
    return _clamp(float(np.percentile(frame_scores, 75)))


def _use_gpu_candidate_metrics(use_gpu: bool) -> bool:
    if not (use_gpu and GPU_AVAILABLE):
        return False
    return os.environ.get("BEATSYNC_GPU_CANDIDATE_METRICS", "1") != "0"


def _use_gpu_scene_detection(use_gpu: bool) -> bool:
    if not use_gpu:
        return False
    return os.environ.get("BEATSYNC_GPU_SCENE_DETECTION", "1") != "0"


def _measure_frames_gpu(frames: Sequence[np.ndarray], sample_times: np.ndarray,
                        start: float, duration: float, stats: np.ndarray | None = None) -> Dict:
    if stats is None:
        stats = frame_statistics(np.stack(frames, axis=0))
    # Small per-frame reductions run on the host after the fused Vulkan kernel.
    brightness = float(np.mean(stats[:, 0]) / 255.0)
    contrast = float(np.mean(np.sqrt(np.maximum(0.0, stats[:, 1] - stats[:, 0] ** 2))) / 80.0)
    saturation = float(np.mean(stats[:, 2]))
    sharpness = _clamp(float(np.mean(np.maximum(0.0, stats[:, 4] - stats[:, 3] ** 2))) / 520.0)
    if len(frames) > 1:
        motion_values = stats[1:, 9] / 42.0
        motion = _clamp(float(np.mean(motion_values)))
        peak_i = int(np.argmax(motion_values)) + 1
        peak_offset = float(sample_times[min(peak_i, len(sample_times) - 1)] - start)
    else:
        motion = 0.0
        peak_offset = duration * 0.5
    colorfulness = _clamp(float(np.mean(
        np.sqrt(np.maximum(0.0, stats[:, 6] - stats[:, 5] ** 2))
        + np.sqrt(np.maximum(0.0, stats[:, 8] - stats[:, 7] ** 2))
    )) / 95.0)

    darkness_penalty = _clamp((0.25 - brightness) / 0.25)
    blown_penalty = _clamp((brightness - 0.86) / 0.14)
    quality = _clamp(
        0.36 * sharpness
        + 0.22 * _clamp(contrast)
        + 0.18 * _clamp(saturation)
        + 0.14 * (1.0 - darkness_penalty)
        + 0.10 * (1.0 - blown_penalty)
    )

    return {
        "duration": duration,
        "brightness": _clamp(brightness),
        "contrast": _clamp(contrast),
        "saturation": _clamp(saturation),
        "sharpness": _clamp(sharpness),
        "motion": _clamp(motion),
        "colorfulness": _clamp(colorfulness),
        "quality_score": quality,
        "peak_offset": _clamp(peak_offset, 0.0, duration, default=duration * 0.5),
    }


def _resize_for_analysis(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    return cv2.resize(frame, (max_width, max(2, int(h * scale))), interpolation=cv2.INTER_AREA)


def _colorfulness(frames: Sequence[np.ndarray]) -> float:
    values = []
    for frame in frames:
        b, g, r = cv2.split(frame.astype("float"))
        rg = np.abs(r - g)
        yb = np.abs(0.5 * (r + g) - b)
        values.append((np.std(rg) + np.std(yb)) / 95.0)
    return _clamp(float(np.mean(values)) if values else 0.0)


def _quality_flags_for_values(brightness: float, contrast: float, saturation: float,
                              sharpness: float, quality: float, duration: float,
                              text_score: float = 0.0) -> List[str]:
    flags: List[str] = []
    if brightness < 0.12:
        flags.append("dark")
    if brightness > 0.93:
        flags.append("blown")
    if sharpness < 0.14:
        flags.append("blurry")
    if contrast < 0.08:
        flags.append("flat")
    if saturation < 0.06:
        flags.append("desaturated")
    if quality < 0.24:
        flags.append("low_quality")
    if duration < 0.45:
        flags.append("too_short")
    if text_score >= 0.58:
        flags.append("text_overlay")
    return flags


def _semantic_text_hint_score(semantic: Dict) -> float:
    if not isinstance(semantic, dict):
        return 0.0
    visible = semantic.get("visible_text")
    if visible is True or str(visible).strip().lower() in {"true", "yes", "1"}:
        return 0.82
    description = str(semantic.get("description", "")).lower()
    if not description:
        return 0.0
    if re.search(r"\b(credits?|subtitles?|captions?|title card|opening title|end title|text overlay|on-screen text|readable text|large text|words|letters)\b", description):
        return 0.76
    return 0.0


def _candidate_edit_roles(candidate: Dict) -> List[str]:
    tags = {str(t).lower() for t in candidate.get("tags", [])}
    semantic = candidate.get("semantic") or {}
    rec = str(semantic.get("recommended_use", "")).lower()
    action = _clamp(candidate.get("action_score", semantic.get("action_intensity", 0.0)))
    beauty = _clamp(candidate.get("beauty_score", semantic.get("beauty_score", 0.0)))
    tension = _clamp(candidate.get("tension_score", 0.0))
    soft = _clamp(candidate.get("soft_score", 0.0))
    motion = _clamp(candidate.get("motion", semantic.get("camera_motion", 0.0)))
    quality = _clamp(candidate.get("quality_score", semantic.get("visual_quality", 0.0)))

    roles = set()
    if rec in {"drop", "soft", "build", "transition", "flow", "filler"}:
        roles.add("flow" if rec == "filler" else rec)
    if action >= 0.58 or motion >= 0.62 or tags.intersection({"action", "combat", "chase", "explosion", "hype"}):
        roles.add("drop")
        roles.add("rhythm")
    if tension >= 0.55 or tags.intersection({"tension", "transition"}):
        roles.add("build")
    if beauty >= 0.56 or soft >= 0.58 or tags.intersection({"beauty", "soft", "sad"}):
        roles.add("soft")
    if quality >= 0.45:
        roles.add("flow")
    if not roles:
        roles.add("flow")
    return sorted(roles)


def _refresh_candidate_derived_fields(candidate: Dict) -> Dict:
    brightness = _clamp(candidate.get("brightness", 0.5), default=0.5)
    contrast = _clamp(candidate.get("contrast", 0.0))
    saturation = _clamp(candidate.get("saturation", 0.0))
    sharpness = _clamp(candidate.get("sharpness", 0.0))
    quality = _clamp(candidate.get("quality_score", 0.0))
    duration = max(0.0, float(candidate.get("duration", 0.0) or 0.0))
    action = _clamp(candidate.get("action_score", 0.0))
    beauty = _clamp(candidate.get("beauty_score", 0.0))
    tension = _clamp(candidate.get("tension_score", 0.0))
    soft = _clamp(candidate.get("soft_score", 0.0))
    motion = _clamp(candidate.get("motion", 0.0))
    colorfulness = _clamp(candidate.get("colorfulness", 0.0))
    semantic = candidate.get("semantic") or {}
    text_score = max(_clamp(candidate.get("text_score", 0.0)), _semantic_text_hint_score(semantic))

    flags = _quality_flags_for_values(brightness, contrast, saturation, sharpness, quality, duration, text_score)
    penalty = 0.0
    if "dark" in flags or "blown" in flags:
        penalty += 0.22
    if "blurry" in flags:
        penalty += 0.20
    if "low_quality" in flags:
        penalty += 0.18
    if "flat" in flags:
        penalty += 0.08

    impact_capacity = _clamp(0.42 * action + 0.22 * motion + 0.18 * tension + 0.12 * contrast + 0.06 * quality)
    highlight_score = _clamp(
        0.30 * quality
        + 0.22 * max(action, beauty, tension)
        + 0.17 * impact_capacity
        + 0.12 * saturation
        + 0.10 * contrast
        + 0.06 * colorfulness
        + 0.03 * soft
        - penalty
    )

    if highlight_score < 0.28 and "weak_highlight" not in flags:
        flags.append("weak_highlight")

    video_file = os.path.abspath(str(candidate.get("video_file", "")))
    scene_index = int(candidate.get("scene_index", 0) or 0)
    start = float(candidate.get("start", 0.0) or 0.0)
    end = float(candidate.get("end", start) or start)

    candidate["quality_flags"] = flags
    candidate["text_score"] = text_score
    candidate["impact_capacity"] = impact_capacity
    candidate["highlight_score"] = highlight_score
    candidate["flow_vector"] = {
        "brightness": brightness,
        "saturation": saturation,
        "motion": motion,
        "action": action,
        "beauty": beauty,
        "tension": tension,
        "quality": quality,
    }
    candidate["identity_key"] = f"{_hash_text(video_file, 10)}:scene:{scene_index}"
    candidate["usable_interval"] = {"start": start, "end": max(start, end)}
    candidate["edit_roles"] = _candidate_edit_roles(candidate)
    return candidate


def _candidate_is_highlight_usable(candidate: Dict) -> bool:
    flags = {str(flag) for flag in candidate.get("quality_flags", [])}
    hard_flags = {"dark", "blown", "blurry", "low_quality", "too_short", "weak_highlight"}
    if flags.intersection(hard_flags):
        return False
    if _clamp(candidate.get("highlight_score", 0.0)) < 0.30:
        return False
    return True


def _build_highlight_pool(candidates: Sequence[Dict]) -> List[Dict]:
    if not candidates:
        return []

    per_source_role_limit = _env_int("BEATSYNC_HIGHLIGHT_PER_SOURCE_ROLE", 18, lo=3, hi=80)
    global_limit = _env_int("BEATSYNC_HIGHLIGHT_GLOBAL_LIMIT", 1400, lo=50, hi=5000)
    time_bucket_seconds = float(_env_int("BEATSYNC_HIGHLIGHT_TIME_BUCKET_SECONDS", 12, lo=4, hi=60))

    usable: List[Dict] = []
    for candidate in candidates:
        refreshed = _refresh_candidate_derived_fields(candidate)
        if _candidate_is_highlight_usable(refreshed):
            usable.append(refreshed)
    grouped: Dict[tuple[str, str], List[Dict]] = {}
    for candidate in usable:
        video_file = str(candidate.get("video_file", ""))
        for role in candidate.get("edit_roles", ["flow"]):
            grouped.setdefault((video_file, str(role)), []).append(candidate)

    selected: Dict[str, Dict] = {}
    for items in grouped.values():
        ranked = sorted(items, key=lambda c: c.get("highlight_score", c.get("editorial_score", 0.0)), reverse=True)
        for candidate in ranked[:per_source_role_limit]:
            selected[str(candidate.get("id"))] = candidate

    bucketed: Dict[tuple[str, int], List[Dict]] = {}
    for candidate in usable:
        video_file = str(candidate.get("video_file", ""))
        center = float(candidate.get("center", candidate.get("start", 0.0)) or 0.0)
        bucket = int(max(0.0, center) // max(1.0, time_bucket_seconds))
        bucketed.setdefault((video_file, bucket), []).append(candidate)

    for items in bucketed.values():
        items.sort(
            key=lambda c: (
                c.get("highlight_score", c.get("editorial_score", 0.0)),
                c.get("quality_score", 0.0),
                c.get("impact_capacity", 0.0),
            ),
            reverse=True,
        )

    # The role seeds above intentionally keep the strongest moments. This fill
    # pass prevents overlapping roles in one source from collapsing the planner
    # pool to a tiny set of repeated clips.
    bucket_keys = sorted(bucketed, key=lambda key: (key[0], key[1]))
    bucket_offsets = {key: 0 for key in bucket_keys}
    while len(selected) < min(global_limit, len(usable)) and bucket_keys:
        added_this_round = 0
        next_keys: List[tuple[str, int]] = []
        for key in bucket_keys:
            items = bucketed[key]
            cursor = bucket_offsets.get(key, 0)
            while cursor < len(items):
                candidate = items[cursor]
                cid = str(candidate.get("id"))
                cursor += 1
                if cid not in selected:
                    selected[cid] = candidate
                    added_this_round += 1
                    break
            bucket_offsets[key] = cursor
            if cursor < len(items):
                next_keys.append(key)
        if added_this_round == 0:
            break
        bucket_keys = next_keys

    if not selected:
        fallback = sorted(
            [_refresh_candidate_derived_fields(c) for c in candidates],
            key=lambda c: c.get("highlight_score", c.get("editorial_score", 0.0)),
            reverse=True,
        )
        selected = {str(c.get("id")): c for c in fallback[:min(len(fallback), per_source_role_limit)]}

    pool = sorted(selected.values(), key=lambda c: c.get("highlight_score", c.get("editorial_score", 0.0)), reverse=True)
    return pool[:global_limit]


def _build_candidate(
    video_file: str,
    source_name: str,
    video_duration: float,
    index: int,
    window: Dict,
    metrics: Dict,
) -> Dict:
    start = float(window["start"])
    end = float(window["end"])
    duration = max(0.01, end - start)
    motion = metrics["motion"]
    quality = metrics["quality_score"]
    brightness = metrics["brightness"]
    saturation = metrics["saturation"]
    contrast = metrics["contrast"]
    sharpness = metrics["sharpness"]
    colorfulness = metrics["colorfulness"]
    text_score = _clamp(metrics.get("text_score", 0.0))

    balanced_light = 1.0 - min(abs(brightness - 0.52) / 0.52, 1.0)
    action = _clamp(0.58 * motion + 0.17 * contrast + 0.12 * saturation + 0.13 * quality)
    beauty = _clamp(0.34 * quality + 0.21 * colorfulness + 0.18 * saturation + 0.17 * balanced_light + 0.10 * sharpness)
    tension = _clamp(0.42 * motion + 0.22 * contrast + 0.18 * (1.0 - balanced_light) + 0.18 * saturation)
    soft = _clamp(0.55 * beauty + 0.25 * balanced_light + 0.20 * (1.0 - motion))
    editorial = _clamp(0.35 * max(action, beauty, tension) + 0.35 * quality + 0.16 * saturation + 0.14 * contrast)

    tags = _fallback_tags(action, beauty, tension, soft, quality)
    semantic = {
        "action_intensity": action,
        "beauty_score": beauty,
        "emotion": "hype" if action > 0.68 else "soft" if soft > 0.64 else "tension" if tension > 0.62 else "neutral",
        "combat": 0.0,
        "chase": _clamp(motion * 0.6),
        "explosion": 0.0,
        "character_focus": _clamp(0.35 * quality + 0.20 * sharpness + 0.10 * balanced_light),
        "camera_motion": motion,
        "visual_quality": quality,
        "recommended_use": "drop" if action > 0.68 else "soft" if soft > 0.64 else "build" if tension > 0.62 else "flow",
        "description": "",
    }

    candidate_id = f"{_hash_text(os.path.abspath(video_file), 10)}_{index:05d}_{int(start * 1000):08d}"
    candidate = {
        "id": candidate_id,
        "video_file": os.path.abspath(video_file),
        "source_name": source_name,
        "video_duration": video_duration,
        "start": start,
        "end": end,
        "duration": duration,
        "center": start + duration * 0.5,
        "peak_time": start + float(metrics.get("peak_offset", duration * 0.5)),
        "scene_index": int(window.get("scene_index", index)),
        "kind": window.get("kind", "scene"),
        "motion": motion,
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "sharpness": sharpness,
        "colorfulness": colorfulness,
        "text_score": text_score,
        "quality_score": quality,
        "action_score": action,
        "beauty_score": beauty,
        "tension_score": tension,
        "soft_score": soft,
        "editorial_score": editorial,
        "tags": tags,
        "semantic": semantic,
        "ai_analyzed": False,
    }
    candidate['visual_fingerprints'] = metrics.get('visual_fingerprints') or []
    return _refresh_candidate_derived_fields(candidate)


def _fallback_tags(action: float, beauty: float, tension: float, soft: float, quality: float) -> List[str]:
    tags: List[str] = []
    if action >= 0.68:
        tags.append("action")
    if beauty >= 0.62:
        tags.append("beauty")
    if tension >= 0.62:
        tags.append("tension")
    if soft >= 0.64:
        tags.append("soft")
    if quality >= 0.68:
        tags.append("clean")
    if not tags:
        tags.append("flow")
    return tags


def _select_ai_candidates(candidates: List[Dict], max_windows: int) -> List[Dict]:
    if len(candidates) <= max_windows:
        return candidates

    selected: Dict[str, Dict] = {}

    def add_many(items: Sequence[Dict], count: int) -> None:
        for item in items[:count]:
            selected[item["id"]] = item

    ranked_action = sorted(candidates, key=lambda c: c.get("action_score", 0.0), reverse=True)
    ranked_beauty = sorted(candidates, key=lambda c: c.get("beauty_score", 0.0), reverse=True)
    ranked_quality = sorted(candidates, key=lambda c: c.get("quality_score", 0.0), reverse=True)
    ranked_highlight = sorted(
        [_refresh_candidate_derived_fields(c) for c in candidates],
        key=lambda c: c.get("highlight_score", c.get("editorial_score", 0.0)),
        reverse=True,
    )

    add_many(ranked_highlight, max(1, max_windows // 3))
    add_many(ranked_action, max(1, max_windows // 3))
    add_many(ranked_beauty, max(1, max_windows // 4))
    add_many(ranked_quality, max(1, max_windows // 5))

    if len(selected) < max_windows:
        step = max(1, len(candidates) // max(1, max_windows - len(selected)))
        for item in candidates[::step]:
            selected[item["id"]] = item
            if len(selected) >= max_windows:
                break

    return list(selected.values())[:max_windows]


def _annotate_candidates_with_qwen(
    video_file: str,
    fps: float,
    candidates: List[Dict],
    qwen_model_path: str,
    use_gpu: bool,
    audio_profile: Dict,
    analysis_video_file: str | None = None,
) -> None:
    max_windows = int(os.environ.get("BEATSYNC_QWEN_MAX_WINDOWS", "120"))
    max_windows = max(0, max_windows)
    if max_windows == 0:
        print("   Qwen semantic analysis skipped (BEATSYNC_QWEN_MAX_WINDOWS=0).")
        return

    ai_candidates = _select_ai_candidates(candidates, max_windows)
    if not ai_candidates:
        return {"qwen_frame_count": 0, "qwen_tag_count": 0}

    print(f"   Qwen semantic analysis: {len(ai_candidates)} candidate moments")
    response = _run_qwen_worker(
        video_file=analysis_video_file or video_file,
        fps=fps,
        candidates=ai_candidates,
        qwen_model_path=qwen_model_path,
        use_gpu=use_gpu,
        audio_profile=audio_profile,
    )
    semantics = response.get("semantics") if isinstance(response, dict) else {}
    if not semantics:
        print("      Qwen returned no semantic tags; deterministic visual tags remain active.")
        return {
            "qwen_frame_count": len(ai_candidates),
            "qwen_tag_count": 0,
            "qwen_model_id": str(response.get("model_id") or "") if isinstance(response, dict) else "",
            "qwen_concurrency": int(response.get("batch_size") or 0) if isinstance(response, dict) else 0,
            "qwen_peak_vram_gb": float(response.get("peak_vram_gb") or 0.0) if isinstance(response, dict) else 0.0,
        }

    semantic_by_id = {str(k): v for k, v in semantics.items()}
    merged_count = 0
    for candidate in ai_candidates:
        semantic = semantic_by_id.get(str(candidate.get("id")))
        if semantic:
            _merge_semantic(candidate, semantic)
            merged_count += 1
    print(f"      Qwen semantic tags merged: {merged_count}/{len(ai_candidates)}")
    timing = response.get("timings_by_job", {}).get("single", {}) if isinstance(response, dict) else {}
    return {
        "qwen_frame_count": int(timing.get("frame_count") or len(ai_candidates)),
        "qwen_tag_count": int(timing.get("tag_count") or merged_count),
        "qwen_model_id": str(response.get("model_id") or "") if isinstance(response, dict) else "",
        "qwen_concurrency": int(response.get("batch_size") or 0) if isinstance(response, dict) else 0,
        "qwen_peak_vram_gb": float(response.get("peak_vram_gb") or 0.0) if isinstance(response, dict) else 0.0,
    }


def _run_qwen_worker(
    video_file: str,
    fps: float,
    candidates: Sequence[Dict],
    qwen_model_path: str,
    use_gpu: bool,
    audio_profile: Dict,
) -> Dict:
    os.makedirs(VIDEO_ANALYSIS_CACHE_DIR, exist_ok=True)
    token = _hash_text(f"{video_file}|{time.time()}", 12)
    request_path = os.path.join(VIDEO_ANALYSIS_CACHE_DIR, f"qwen_request_{token}.json")
    response_path = os.path.join(VIDEO_ANALYSIS_CACHE_DIR, f"qwen_response_{token}.json")
    worker_path = os.path.join(ROOT_DIR, "src", "auto_mode", "stage5_qwen_scene_worker.py")
    request = {
        "video_file": video_file,
        "fps": fps,
        "qwen_model_path": qwen_model_path,
        "use_gpu": bool(use_gpu),
        "audio_profile": audio_profile,
        "candidates": [
            {
                "id": c.get("id"),
                "start": c.get("start"),
                "end": c.get("end"),
            }
            for c in candidates
        ],
    }
    with open(request_path, "w", encoding="utf-8") as f:
        json.dump(request, f)

    env = _qwen_worker_environment()
    timeout = max(1800, int(len(candidates) * 75))
    try:
        result = subprocess.run(
            [sys.executable, worker_path, "--request", request_path, "--response", response_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            env=env,
        )
        if result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                print(f"      {line}")
        if result.returncode != 0:
            error_tail = (result.stderr or "").strip()[-1800:]
            print(f"      Qwen worker failed: {error_tail}")
            return {}
        with open(response_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except subprocess.TimeoutExpired:
        print("      Qwen worker timed out; deterministic visual tags remain active.")
        return {}
    except Exception as e:
        print(f"      Qwen worker error: {e}")
        return {}
    finally:
        # Keep Qwen worker request/response files in video_analysis_cache for
        # reproducibility and debugging. The user explicitly wants this cache
        # folder to be preserved.
        pass


def _qwen_worker_environment() -> Dict[str, str]:
    env = os.environ.copy()
    llama_dir = os.environ.get("BEATSYNC_QWEN_LLAMA_DIR", DEFAULT_LLAMA_CPP_DIR)
    llama_root = os.path.normcase(os.path.abspath(llama_dir))
    path_parts = []
    for part in env.get("PATH", "").split(os.pathsep):
        if not part:
            continue
        norm = os.path.normcase(os.path.abspath(part))
        if norm == llama_root:
            continue
        path_parts.append(part)
    env["PATH"] = os.pathsep.join([llama_dir] + path_parts)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _merge_semantic(candidate: Dict, semantic: Dict) -> None:
    if not semantic:
        return
    current = candidate.get("semantic", {})
    merged = dict(current)
    numeric_keys = [
        "action_intensity",
        "beauty_score",
        "combat",
        "chase",
        "explosion",
        "character_focus",
        "camera_motion",
        "visual_quality",
    ]
    for key in numeric_keys:
        if key in semantic:
            merged[key] = _clamp(semantic[key])
    for key in ["emotion", "recommended_use", "description"]:
        if semantic.get(key):
            merged[key] = str(semantic[key])[:160]
    if "visible_text" in semantic:
        merged["visible_text"] = bool(semantic.get("visible_text"))

    candidate["semantic"] = merged
    candidate["ai_analyzed"] = True

    deterministic_action = _clamp(candidate.get("action_score", 0.0))
    deterministic_motion = _clamp(candidate.get("motion", 0.0))
    semantic_action = _clamp(merged.get("action_intensity", 0.0))
    motion_gate = _clamp(0.35 + 0.65 * deterministic_motion)

    # Qwen is used as semantic guidance, not as the sole truth. Motion/quality
    # metrics stay dominant so a beautiful static frame is not hallucinated into
    # an action scene.
    action = _clamp(0.72 * deterministic_action + 0.28 * semantic_action * motion_gate)
    beauty = _clamp(0.65 * candidate.get("beauty_score", 0.0) + 0.35 * merged.get("beauty_score", 0.0))
    quality = _clamp(0.70 * candidate.get("quality_score", 0.0) + 0.30 * merged.get("visual_quality", 0.0))
    camera_motion = _clamp(0.75 * deterministic_motion + 0.25 * merged.get("camera_motion", deterministic_motion))
    tension = _clamp(0.48 * candidate.get("tension_score", 0.0) + 0.22 * camera_motion + 0.18 * action + 0.12 * merged.get("character_focus", 0.0))
    soft = _clamp(0.48 * beauty + 0.24 * merged.get("character_focus", 0.0) + 0.18 * (1.0 - action) + 0.10 * quality)

    candidate["action_score"] = action
    candidate["beauty_score"] = beauty
    candidate["quality_score"] = quality
    candidate["tension_score"] = tension
    candidate["soft_score"] = soft
    candidate["editorial_score"] = _clamp(0.30 * max(action, beauty, tension) + 0.42 * quality + 0.16 * candidate.get("saturation", 0.0) + 0.12 * candidate.get("contrast", 0.0))

    tags = set(_fallback_tags(action, beauty, tension, soft, quality))
    rec = str(merged.get("recommended_use", "")).lower()
    emotion = str(merged.get("emotion", "")).lower()
    if rec and (rec != "drop" or action >= 0.45):
        tags.add(rec)
    if emotion and (emotion != "hype" or action >= 0.45):
        tags.add(emotion)
    if merged.get("combat", 0.0) > 0.50 and action >= 0.38:
        tags.add("combat")
    if merged.get("chase", 0.0) > 0.50 and action >= 0.38:
        tags.add("chase")
    if merged.get("explosion", 0.0) > 0.55 and action >= 0.38:
        tags.add("explosion")
    candidate["tags"] = sorted(tags)
    _refresh_candidate_derived_fields(candidate)
