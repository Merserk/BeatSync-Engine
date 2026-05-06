#!/usr/bin/env python3
"""Standalone Qwen3-VL semantic tagging worker via vLLM's OpenAI API.

The main app keeps Auto Mode's candidate selection, prompt, JSON parsing, and
merge behavior outside this worker. This process only samples the same candidate
frames and sends them to a required local vLLM OpenAI-compatible server.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any, Dict, List
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from PIL import Image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    return parser.parse_args()


def _clamp(value, lo: float = 0.0, hi: float = 1.0, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        v = default
    if not np.isfinite(v):
        v = default
    return max(lo, min(hi, v))


def _parse_json_object(text: str) -> Dict:
    if not text:
        return {}
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _resize_frame(frame, max_width: int = 512):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    return cv2.resize(frame, (max_width, max(2, int(h * scale))), interpolation=cv2.INTER_AREA)


def _candidate_mid_frame(fps: float, candidate: Dict) -> int:
    start = float(candidate.get("start", 0.0))
    end = float(candidate.get("end", start))
    t = start + max(0.01, end - start) * 0.5
    return max(0, int(round(t * fps)))


def _frame_to_image(frame) -> Image.Image | None:
    if frame is None:
        return None
    frame = _resize_frame(frame, max_width=512)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _extract_frame(cap, fps: float, candidate: Dict) -> Image.Image | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, _candidate_mid_frame(fps, candidate))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return _frame_to_image(frame)


def _prefetch_candidate_frames(cap, fps: float, candidates: List[Dict]) -> List[Dict]:
    """Decode Qwen sample frames in timeline order and keep them in RAM.

    The old worker sought once per candidate in score order, which can thrash
    long video files. This keeps Auto Mode logic unchanged: every candidate uses
    the same midpoint frame as before, but frame reads happen in decode order.

    Large gaps still use seeking instead of blindly grabbing thousands of
    intermediate frames. Small gaps are advanced with grab(), which is much
    cheaper than a fresh random seek on many Windows/OpenCV/codec combinations.
    """
    started = time.perf_counter()
    if os.environ.get("BEATSYNC_QWEN_PREFETCH_FRAMES", "1") == "0":
        items = [{"candidate": c, "image": _extract_frame(cap, fps, c)} for c in candidates]
        ready = [item for item in items if item.get("image") is not None]
        elapsed = max(0.001, time.perf_counter() - started)
        print(
            f"Qwen frame prefetch disabled: {len(ready)}/{len(candidates)} frames "
            f"via direct seek in {elapsed:.1f}s",
            flush=True,
        )
        return ready

    try:
        max_gap = int(os.environ.get("BEATSYNC_QWEN_ORDERED_MAX_GAP", "96"))
    except ValueError:
        max_gap = 96
    max_gap = max(0, max_gap)

    plans = []
    for original_index, candidate in enumerate(candidates):
        plans.append({
            "original_index": original_index,
            "candidate": candidate,
            "frame_idx": _candidate_mid_frame(fps, candidate),
            "image": None,
        })

    seek_count = 0
    grab_count = 0
    current = None
    for plan in sorted(plans, key=lambda item: item["frame_idx"]):
        target = int(plan["frame_idx"])
        if current is None or target < current or target - current > max_gap:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            current = target
            seek_count += 1
        while current < target:
            if not cap.grab():
                break
            current += 1
            grab_count += 1
        if current != target:
            continue
        ok, frame = cap.read()
        current += 1
        if ok and frame is not None:
            plan["image"] = _frame_to_image(frame)

    plans.sort(key=lambda item: item["original_index"])
    ready = [p for p in plans if p.get("image") is not None]
    elapsed = max(0.001, time.perf_counter() - started)
    approx_ram_mb = sum(item["image"].width * item["image"].height * 3 for item in ready) / (1024 * 1024)
    print(
        f"Qwen frame prefetch: {len(ready)}/{len(candidates)} frames in RAM "
        f"(~{approx_ram_mb:.0f} MB, {seek_count} seeks, {grab_count} grabs, "
        f"max gap {max_gap}) in {elapsed:.1f}s",
        flush=True,
    )
    return ready


def _configure_opencv_ffmpeg_threads() -> int:
    """Let OpenCV/FFmpeg use more decoder threads for frame prefetch.

    This does not alter semantic logic; it only gives the codec more CPU workers
    while reading the exact same midpoint frames.
    """
    cpu = os.cpu_count() or 4
    try:
        default_threads = min(8, max(2, cpu // 2))
        threads = int(os.environ.get("BEATSYNC_OPENCV_FFMPEG_THREADS", str(default_threads)))
    except ValueError:
        threads = min(8, max(2, cpu // 2))
    threads = max(1, min(threads, max(1, cpu)))

    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", f"threads;{threads}")
    try:
        cv2.setNumThreads(threads)
    except Exception:
        pass
    return threads


def _normalize_semantic(data: Dict) -> Dict:
    out = {}
    for key in [
        "action_intensity",
        "beauty_score",
        "combat",
        "chase",
        "explosion",
        "character_focus",
        "camera_motion",
        "visual_quality",
    ]:
        if key in data:
            out[key] = _clamp(data[key])
    for key in ["emotion", "recommended_use", "description"]:
        if data.get(key):
            out[key] = str(data[key])[:160]
    return out


def _detect_local_gpu_total_gb() -> float:
    root = _project_root()
    python_exe = os.path.join(root, "bin", "vllm", "python-3.12.10-embed-amd64", "python.exe")
    if not os.path.exists(python_exe):
        return 0.0
    code = (
        "import torch\n"
        "print(torch.cuda.get_device_properties(0).total_memory/(1024**3) "
        "if torch.cuda.is_available() else 0)"
    )
    try:
        output = subprocess.check_output(
            [python_exe, "-c", code],
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        return float((output or "0").strip())
    except Exception:
        return 0.0


def _qwen_batch_size(_use_cuda: bool = True) -> int:
    total_gb = _detect_local_gpu_total_gb() if _is_local_vllm_url(_vllm_base_url()) else 0.0
    if total_gb <= 0:
        default_size = 8
    elif total_gb <= 8.5:
        default_size = 4
    elif total_gb <= 12.5:
        default_size = 6
    elif total_gb <= 18.0:
        default_size = 8
    elif total_gb <= 26.0:
        default_size = 12
    else:
        default_size = 16
    try:
        value = int(os.environ.get("BEATSYNC_QWEN_BATCH_SIZE", str(default_size)))
    except ValueError:
        value = default_size
    return max(1, min(value, 32))


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _vllm_base_url() -> str:
    value = (
        os.environ.get("BEATSYNC_VLLM_BASE_URL")
        or os.environ.get("VLLM_OPENAI_BASE_URL")
        or "http://127.0.0.1:8000/v1"
    )
    return value.rstrip("/")


def _vllm_api_key() -> str:
    return (
        os.environ.get("BEATSYNC_VLLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "EMPTY"
    )


def _vllm_server_error(base_url: str, model_path: str, detail: str) -> str:
    return (
        "ERROR: vLLM OpenAI-compatible API is required for Qwen semantic tags, "
        f"but no server is reachable at {base_url}.\n"
        f"Model: {model_path}\n"
        f"Details: {detail}"
    )


def _is_local_vllm_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def _vllm_wait_seconds() -> float:
    try:
        return max(30.0, float(os.environ.get("BEATSYNC_VLLM_START_TIMEOUT", "900")))
    except ValueError:
        return 900.0


def _start_local_vllm_server(base_url: str, model_path: str) -> str:
    root = _project_root()
    launcher = os.path.join(root, "src", "start_vllm_server.py")
    log_dir = os.path.join(root, "input", "video_analysis_cache")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "vllm_server.log")

    if not os.path.exists(launcher):
        raise RuntimeError(f"Local vLLM launcher not found: {launcher}")

    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = str(parsed.port or 8000)
    command = [
        sys.executable,
        launcher,
        "--host",
        host,
        "--port",
        port,
        "--model",
        model_path,
    ]

    print(
        f"vLLM server not running at {base_url}; starting local Qwen3-VL server...",
        flush=True,
    )
    print(f"vLLM server log: {log_path}", flush=True)
    log_file = open(log_path, "a", encoding="utf-8", errors="replace")
    log_file.write("\n\n=== BeatSync vLLM auto-start ===\n")
    log_file.write(" ".join(command) + "\n")
    log_file.flush()

    startupinfo = None
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
    subprocess.Popen(
        command,
        cwd=root,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
        startupinfo=startupinfo,
        close_fds=True,
    )
    log_file.close()
    return log_path


def _wait_for_vllm_models(base_url: str, model_path: str, log_path: str | None = None) -> Dict[str, Any]:
    deadline = time.perf_counter() + _vllm_wait_seconds()
    last_error = ""
    probe = VllmClient(base_url=base_url, api_key=_vllm_api_key(), model="")
    while time.perf_counter() < deadline:
        try:
            return probe.get("/models", timeout=8.0)
        except requests.RequestException as exc:
            last_error = str(exc)
            time.sleep(3.0)
    detail = last_error
    if log_path:
        detail = f"{detail}; vLLM server log: {log_path}"
    raise RuntimeError(_vllm_server_error(base_url, model_path, detail))


class VllmClient:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def get(self, path: str, timeout: float = 8.0) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        response = requests.get(url, headers=self.headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    def post(self, path: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        response = requests.post(url, headers=self.headers, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}


def _model_ids(models_response: Dict[str, Any]) -> List[str]:
    models = models_response.get("data") or []
    out: List[str] = []
    for item in models:
        if isinstance(item, dict) and item.get("id"):
            out.append(str(item["id"]))
    return out


def _resolve_vllm_model(base_url: str, model_path: str) -> str:
    requested = os.environ.get("BEATSYNC_VLLM_MODEL") or os.environ.get("VLLM_MODEL")
    probe = VllmClient(base_url=base_url, api_key=_vllm_api_key(), model=requested or "")
    try:
        models_response = probe.get("/models", timeout=8.0)
    except requests.RequestException as exc:
        if not _is_local_vllm_url(base_url):
            raise RuntimeError(_vllm_server_error(base_url, model_path, str(exc))) from exc
        log_path = _start_local_vllm_server(base_url, model_path)
        models_response = _wait_for_vllm_models(base_url, model_path, log_path)

    ids = _model_ids(models_response)
    if requested:
        if ids and requested not in ids:
            print(
                f"vLLM model override '{requested}' not listed by /v1/models; using it anyway.",
                flush=True,
            )
        return requested

    basename = os.path.basename(os.path.normpath(model_path))
    for model_id in ids:
        if model_id == basename or model_id == model_path:
            return model_id
    if ids:
        print(
            f"vLLM /v1/models did not list {basename}; using served model '{ids[0]}'.",
            flush=True,
        )
        return ids[0]
    return basename or model_path


def _connect_vllm(model_path: str) -> VllmClient:
    base_url = _vllm_base_url()
    model_id = _resolve_vllm_model(base_url, model_path)
    client = VllmClient(base_url=base_url, api_key=_vllm_api_key(), model=model_id)
    print(f"vLLM OpenAI API ready: {base_url}; model: {model_id}", flush=True)
    return client


def _image_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    try:
        quality = int(os.environ.get("BEATSYNC_QWEN_JPEG_QUALITY", "90"))
    except ValueError:
        quality = 85
    quality = max(60, min(95, quality))
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _message_content(data: Dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content) if content is not None else ""


def _generate_semantic(client: VllmClient, image: Image.Image, candidate: Dict, prompt: str) -> tuple[str, Dict]:
    payload = {
        "model": client.model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _image_data_url(image)}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": 160,
        "temperature": 0,
    }
    data = client.post("/chat/completions", payload, timeout=180.0)
    parsed = _normalize_semantic(_parse_json_object(_message_content(data)))
    return str(candidate["id"]), parsed


def _generate_semantics_batch(
    client: VllmClient,
    batch_items: List[Dict],
    prompt: str,
    request_workers: int,
) -> Dict[str, Dict]:
    semantics: Dict[str, Dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(request_workers, len(batch_items)))) as executor:
        futures = {
            executor.submit(
                _generate_semantic,
                client,
                item["image"],
                item["candidate"],
                prompt,
            ): item
            for item in batch_items
        }
        for future in as_completed(futures):
            candidate_id, parsed = future.result()
            if parsed:
                semantics[candidate_id] = parsed
    return semantics



def _build_prompt(audio_profile: Dict) -> str:
    style_hint = audio_profile.get("smart_preset", "rhythmic_gmv_amv")
    return (
        "You are tagging one source-video moment for professional AMV/GMV editing. "
        f"The music edit style is {style_hint}. "
        "Return JSON only. Keys: action_intensity, beauty_score, combat, chase, explosion, "
        "character_focus, camera_motion, visual_quality as numbers 0..1; "
        "emotion as one of soft,tension,hype,sad,neutral; "
        "recommended_use as one of drop,soft,build,transition,flow,filler; "
        "description under 12 words. Do not include markdown."
    )


def _run_semantics_for_video(
    *,
    client: VllmClient,
    batch_size: int,
    video_file: str,
    fps: float,
    candidates: List[Dict],
    prompt: str,
) -> tuple[Dict[str, Dict], Dict]:
    decode_threads = _configure_opencv_ffmpeg_threads()
    print(f"Qwen OpenCV decode threads: {decode_threads}", flush=True)
    cap = cv2.VideoCapture(video_file)
    semantics: Dict[str, Dict] = {}
    timings = {
        "prefetch_seconds": 0.0,
        "inference_seconds": 0.0,
        "frame_count": 0,
        "tag_count": 0,
    }
    prefetch_started = time.perf_counter()
    try:
        frame_items = _prefetch_candidate_frames(cap, fps, candidates)
        timings["prefetch_seconds"] = time.perf_counter() - prefetch_started
        timings["frame_count"] = len(frame_items)
        inference_started = time.perf_counter()
        idx = 0
        while idx < len(frame_items):
            current_size = min(batch_size, len(frame_items) - idx)
            while current_size > 0:
                scan_end = min(len(frame_items), idx + current_size)
                batch_items = frame_items[idx:scan_end]
                images = [item["image"] for item in batch_items]

                if not images:
                    idx = scan_end
                    break

                semantics.update(
                    _generate_semantics_batch(
                        client=client,
                        batch_items=batch_items,
                        prompt=prompt,
                        request_workers=current_size,
                    )
                )
                idx = scan_end
                break

            if idx % 10 == 0 or idx >= len(frame_items):
                elapsed = max(0.001, time.perf_counter() - inference_started)
                rate = idx / elapsed
                print(f"Qwen tagged {idx}/{len(frame_items)} ({rate:.2f}/s)", flush=True)
        elapsed = max(0.001, time.perf_counter() - inference_started)
        timings["inference_seconds"] = elapsed
        timings["tag_count"] = len(semantics)
        print(
            f"Qwen semantic inference total: {len(semantics)}/{len(frame_items)} tags "
            f"in {elapsed:.1f}s ({len(frame_items) / elapsed:.2f} candidates/s)",
            flush=True,
        )
    finally:
        cap.release()
    return semantics, timings


def main() -> None:
    args = _parse_args()
    whole_started = time.perf_counter()
    with open(args.request, "r", encoding="utf-8") as f:
        request = json.load(f)

    model_path = request["qwen_model_path"]
    audio_profile = request.get("audio_profile") or {}
    prompt = _build_prompt(audio_profile)

    client = _connect_vllm(model_path)
    batch_size = _qwen_batch_size(True)
    print(f"vLLM request concurrency: {batch_size}", flush=True)

    jobs = request.get("jobs")
    if not jobs:
        jobs = [{
            "job_id": "single",
            "video_file": request["video_file"],
            "fps": float(request.get("fps") or 24.0),
            "candidates": request.get("candidates") or [],
        }]
        legacy_single = True
    else:
        legacy_single = False

    semantics_by_job: Dict[str, Dict[str, Dict]] = {}
    timings_by_job: Dict[str, Dict] = {}
    for job_index, job in enumerate(jobs, 1):
        job_id = str(job.get("job_id", job_index))
        video_file = job["video_file"]
        fps = float(job.get("fps") or 24.0)
        candidates: List[Dict] = job.get("candidates") or []
        print(
            f"Qwen/vLLM job {job_index}/{len(jobs)}: {os.path.basename(video_file)} "
            f"({len(candidates)} candidates)",
            flush=True,
        )
        semantics, timings = _run_semantics_for_video(
            client=client,
            batch_size=batch_size,
            video_file=video_file,
            fps=fps,
            candidates=candidates,
            prompt=prompt,
        )
        semantics_by_job[job_id] = semantics
        timings_by_job[job_id] = timings

    total_seconds = time.perf_counter() - whole_started
    response = {
        "model_load_seconds": 0.0,
        "model_id": client.model,
        "batch_size": batch_size,
        "total_seconds": total_seconds,
        "timings_by_job": timings_by_job,
    }
    if legacy_single:
        response["semantics"] = semantics_by_job.get("single", {})
    else:
        response["semantics_by_job"] = semantics_by_job

    with open(args.response, "w", encoding="utf-8") as f:
        json.dump(response, f, indent=2)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr, flush=True)
        raise SystemExit(2)
