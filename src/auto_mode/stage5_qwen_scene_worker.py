#!/usr/bin/env python3
"""Standalone Qwen3-VL semantic tagging worker via Transformers.

The main app keeps Auto Mode's candidate selection, prompt, JSON parsing, and
merge behavior outside this worker. This process samples the same candidate
frames and tags them with the local Qwen3-VL model using PyTorch/Transformers.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
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


def _qwen_frame_width() -> int:
    try:
        value = int(os.environ.get("BEATSYNC_QWEN_FRAME_WIDTH", "512"))
    except ValueError:
        value = 512
    return max(224, min(value, 768))


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


def _frame_to_image(frame, max_width: int) -> Image.Image | None:
    if frame is None:
        return None
    frame = _resize_frame(frame, max_width=max_width)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _extract_frame(cap, fps: float, candidate: Dict, max_width: int) -> Image.Image | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, _candidate_mid_frame(fps, candidate))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return _frame_to_image(frame, max_width=max_width)


def _prefetch_candidate_frames(cap, fps: float, candidates: List[Dict], max_width: int) -> List[Dict]:
    """Decode Qwen sample frames in timeline order and keep them in RAM."""
    started = time.perf_counter()
    if os.environ.get("BEATSYNC_QWEN_PREFETCH_FRAMES", "1") == "0":
        items = [{"candidate": c, "image": _extract_frame(cap, fps, c, max_width)} for c in candidates]
        ready = [item for item in items if item.get("image") is not None]
        elapsed = max(0.001, time.perf_counter() - started)
        print(
            f"Qwen frame prefetch disabled: {len(ready)}/{len(candidates)} frames "
            f"via direct seek at max width {max_width} in {elapsed:.1f}s",
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
            plan["image"] = _frame_to_image(frame, max_width=max_width)

    plans.sort(key=lambda item: item["original_index"])
    ready = [p for p in plans if p.get("image") is not None]
    elapsed = max(0.001, time.perf_counter() - started)
    approx_ram_mb = sum(item["image"].width * item["image"].height * 3 for item in ready) / (1024 * 1024)
    print(
        f"Qwen frame prefetch: {len(ready)}/{len(candidates)} frames in RAM "
        f"(~{approx_ram_mb:.0f} MB, {seek_count} seeks, {grab_count} grabs, "
        f"max gap {max_gap}, max width {max_width}) in {elapsed:.1f}s",
        flush=True,
    )
    return ready


def _configure_opencv_ffmpeg_threads() -> int:
    """Let OpenCV/FFmpeg use more decoder threads for frame prefetch."""
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


def _detect_cuda_memory_gb() -> Tuple[float, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0, 0.0
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        return float(total_bytes) / (1024 ** 3), float(free_bytes) / (1024 ** 3)
    except Exception:
        return 0.0, 0.0


def _adaptive_qwen_batch_size(total_gb: float, free_gb: float) -> int:
    if total_gb <= 0:
        return 1
    elif total_gb <= 8.5:
        total_cap = 4
    elif total_gb <= 12.5:
        total_cap = 16
    elif total_gb <= 18.0:
        total_cap = 96
    else:
        total_cap = 128

    try:
        reserve_gb = float(os.environ.get("BEATSYNC_QWEN_VRAM_RESERVE_GB", "0"))
    except ValueError:
        reserve_gb = 0.0
    if reserve_gb <= 0:
        reserve_gb = max(1.75, total_gb * 0.12)

    if total_gb <= 8.5:
        default_per_item_gb = 0.50
    elif total_gb <= 12.5:
        default_per_item_gb = 0.25
    else:
        default_per_item_gb = 0.07
    try:
        per_item_gb = float(os.environ.get("BEATSYNC_QWEN_BATCH_ITEM_GB", str(default_per_item_gb)))
    except ValueError:
        per_item_gb = default_per_item_gb
    per_item_gb = max(0.03, per_item_gb)

    max_batch_is_override = "BEATSYNC_QWEN_MAX_BATCH_SIZE" in os.environ
    try:
        max_batch = int(os.environ.get("BEATSYNC_QWEN_MAX_BATCH_SIZE", "96"))
    except ValueError:
        max_batch = 96
    max_batch = max(1, min(max_batch, 128))
    if max_batch_is_override:
        total_cap = max(total_cap, max_batch)

    usable_gb = max(0.0, free_gb - reserve_gb)
    free_cap = max(1, int(usable_gb // per_item_gb))
    return max(1, min(total_cap, free_cap, max_batch))


def _qwen_batch_size(_use_cuda: bool = True) -> int:
    total_gb, free_gb = _detect_cuda_memory_gb()
    default_size = _adaptive_qwen_batch_size(total_gb, free_gb)
    try:
        value = int(os.environ.get("BEATSYNC_QWEN_BATCH_SIZE", str(default_size)))
    except ValueError:
        value = default_size
    value = max(1, min(value, 128))
    if total_gb > 0:
        source = "env override" if "BEATSYNC_QWEN_BATCH_SIZE" in os.environ else "adaptive"
        print(
            f"Qwen VRAM: total {total_gb:.1f} GB, free {free_gb:.1f} GB after model load; "
            f"batch {value} ({source})",
            flush=True,
        )
    return value


def _cuda_peak_used_gb(torch_module) -> float:
    try:
        if not torch_module.cuda.is_available():
            return 0.0
        free_bytes, total_bytes = torch_module.cuda.mem_get_info(0)
        driver_used = float(total_bytes - free_bytes) / (1024 ** 3)
        reserved = float(torch_module.cuda.max_memory_reserved(0)) / (1024 ** 3)
        allocated = float(torch_module.cuda.max_memory_allocated(0)) / (1024 ** 3)
        return max(driver_used, reserved, allocated)
    except Exception:
        return 0.0


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


class QwenTransformersClient:
    def __init__(self, model_path: str) -> None:
        started = time.perf_counter()
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.model_id = os.path.basename(os.path.normpath(model_path)) or model_path
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        dtype = self._torch_dtype()
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
            "dtype": dtype,
        }
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
            model_kwargs["attn_implementation"] = os.environ.get("BEATSYNC_QWEN_ATTN", "sdpa")

        try:
            self.model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
        except TypeError:
            model_kwargs.pop("attn_implementation", None)
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
            self.model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)

        if not torch.cuda.is_available():
            self.model.to("cpu")
        self.model.eval()
        self.device = self._model_device()
        self.max_new_tokens = self._max_new_tokens()
        self.load_seconds = time.perf_counter() - started
        print(
            f"Qwen Transformers model ready: {self.model_id}; device: {self.device}; "
            f"dtype: {dtype}; load {self.load_seconds:.1f}s",
            flush=True,
        )

    def _torch_dtype(self):
        torch = self.torch
        requested = os.environ.get("BEATSYNC_QWEN_TORCH_DTYPE", "auto").lower()
        if requested == "float32":
            return torch.float32
        if requested == "float16":
            return torch.float16
        if requested == "bfloat16":
            return torch.bfloat16
        if not torch.cuda.is_available():
            return torch.float32
        try:
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception:
            return torch.float16

    def _model_device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.torch.device("cuda:0" if self.torch.cuda.is_available() else "cpu")

    @staticmethod
    def _max_new_tokens() -> int:
        try:
            value = int(os.environ.get("BEATSYNC_QWEN_MAX_NEW_TOKENS", "128"))
        except ValueError:
            value = 128
        return max(32, min(value, 256))

    def _inputs_for_images(self, images: List[Image.Image], prompt: str) -> Dict[str, Any]:
        texts = []
        for image in images:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }]
            texts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

    def generate_batch(self, images: List[Image.Image], prompt: str) -> List[str]:
        torch = self.torch
        inputs = self._inputs_for_images(images, prompt)
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "use_cache": True,
        }
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)
        prompt_len = int(inputs["input_ids"].shape[1])
        generated_ids = output_ids[:, prompt_len:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)

    def clear_cache(self) -> None:
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def _is_out_of_memory(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def _generate_semantics_batch(
    client: QwenTransformersClient,
    batch_items: List[Dict],
    prompt: str,
) -> Dict[str, Dict]:
    semantics: Dict[str, Dict] = {}
    images = [item["image"] for item in batch_items]
    try:
        texts = client.generate_batch(images, prompt)
    except RuntimeError as exc:
        if len(batch_items) <= 1 or not _is_out_of_memory(exc):
            raise
        split_at = max(1, len(batch_items) // 2)
        print(
            f"Qwen batch of {len(batch_items)} hit GPU memory; retrying as "
            f"{split_at}+{len(batch_items) - split_at}.",
            flush=True,
        )
        client.clear_cache()
        semantics.update(_generate_semantics_batch(client, batch_items[:split_at], prompt))
        semantics.update(_generate_semantics_batch(client, batch_items[split_at:], prompt))
        return semantics

    for item, text in zip(batch_items, texts):
        parsed = _normalize_semantic(_parse_json_object(text))
        if parsed:
            semantics[str(item["candidate"]["id"])] = parsed
    return semantics


def _run_semantics_for_video(
    *,
    client: QwenTransformersClient,
    batch_size: int,
    video_file: str,
    fps: float,
    candidates: List[Dict],
    prompt: str,
) -> tuple[Dict[str, Dict], Dict]:
    decode_threads = _configure_opencv_ffmpeg_threads()
    print(f"Qwen OpenCV decode threads: {decode_threads}", flush=True)
    frame_width = _qwen_frame_width()
    print(f"Qwen frame max width: {frame_width}", flush=True)
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
        frame_items = _prefetch_candidate_frames(cap, fps, candidates, frame_width)
        timings["prefetch_seconds"] = time.perf_counter() - prefetch_started
        timings["frame_count"] = len(frame_items)
        inference_started = time.perf_counter()
        idx = 0
        while idx < len(frame_items):
            scan_end = min(len(frame_items), idx + batch_size)
            batch_items = frame_items[idx:scan_end]
            if not batch_items:
                break
            semantics.update(
                _generate_semantics_batch(
                    client=client,
                    batch_items=batch_items,
                    prompt=prompt,
                )
            )
            idx = scan_end
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

    client = QwenTransformersClient(model_path)
    batch_size = _qwen_batch_size(True)
    print(f"Qwen Transformers batch size: {batch_size}", flush=True)

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
            f"Qwen Transformers job {job_index}/{len(jobs)}: {os.path.basename(video_file)} "
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
        client.clear_cache()

    total_seconds = time.perf_counter() - whole_started
    response = {
        "model_load_seconds": client.load_seconds,
        "model_id": client.model_id,
        "batch_size": batch_size,
        "peak_vram_gb": _cuda_peak_used_gb(client.torch),
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
