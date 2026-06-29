#!/usr/bin/env python3
"""Standalone Qwen3-VL semantic tagging worker via llama.cpp Vulkan.

The main app keeps Auto Mode's candidate selection and merge behavior outside
this worker. This process samples the same candidate frames and tags them with
the bundled Qwen3-VL GGUF model through llama.cpp. It prefers a persistent
llama-server process so the model loads once, and falls back to llama-mtmd-cli
when the server path is unavailable.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import io
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[2]
BIN_DIR = ROOT_DIR / "bin"
DEFAULT_LLAMA_DIR = BIN_DIR / "llama-bin-win-vulkan-x64"
DEFAULT_MODEL = BIN_DIR / "models" / "Qwen3VL-2B-Instruct-Q8_0.gguf"
DEFAULT_MMPROJ = BIN_DIR / "models" / "mmproj-Qwen3VL-2B-Instruct-F16.gguf"

NUMERIC_KEYS = [
    "action_intensity",
    "beauty_score",
    "combat",
    "chase",
    "explosion",
    "character_focus",
    "camera_motion",
    "visual_quality",
]
ALLOWED_EMOTIONS = {"soft", "tension", "hype", "sad", "neutral"}
ALLOWED_USES = {"drop", "soft", "build", "transition", "flow", "filler"}
SEMANTIC_SCHEMA = {
    "type": "object",
    "properties": {
        key: {"type": "number", "minimum": 0, "maximum": 1}
        for key in NUMERIC_KEYS
    },
    "required": NUMERIC_KEYS + ["emotion", "recommended_use", "description"],
    "additionalProperties": False,
}
SEMANTIC_SCHEMA["properties"].update({
    "emotion": {"type": "string", "enum": sorted(ALLOWED_EMOTIONS)},
    "recommended_use": {"type": "string", "enum": sorted(ALLOWED_USES)},
    "description": {"type": "string"},
})


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


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(lo, min(hi, value))


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
    return _env_int("BEATSYNC_QWEN_FRAME_WIDTH", 512, lo=224, hi=768)


def _max_new_tokens() -> int:
    return _env_int("BEATSYNC_QWEN_MAX_NEW_TOKENS", 128, lo=32, hi=256)


def _ctx_sizes() -> List[int]:
    primary = _env_int("BEATSYNC_QWEN_LLAMA_CTX", 8192, lo=1024, hi=262144)
    fallback = _env_int("BEATSYNC_QWEN_LLAMA_CTX_FALLBACK", 4096, lo=1024, hi=262144)
    sizes = []
    for value in [primary, fallback]:
        if value not in sizes:
            sizes.append(value)
    return sizes


def _llama_slots() -> int:
    return _env_int("BEATSYNC_QWEN_LLAMA_SLOTS", 16, lo=1, hi=32)


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
    if not isinstance(data, dict):
        return {}

    out = {}
    for key in NUMERIC_KEYS:
        if key not in data:
            return {}
        out[key] = _clamp(data[key])

    emotion = str(data.get("emotion", "")).strip().lower()
    if emotion not in ALLOWED_EMOTIONS:
        return {}
    recommended_use = str(data.get("recommended_use", "")).strip().lower()
    if recommended_use not in ALLOWED_USES:
        return {}

    description = str(data.get("description", "")).strip()
    if not description:
        return {}
    out["emotion"] = emotion
    out["recommended_use"] = recommended_use
    out["description"] = description[:160]
    return out


def _semantic_from_text(text: str) -> Dict:
    return _normalize_semantic(_parse_json_object(text))


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


def _safe_file_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:80] or "item"


def _image_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _image_data_uri(image: Image.Image) -> str:
    encoded = base64.b64encode(_image_png_bytes(image)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _response_prefix(response_path: str) -> Path:
    return Path(response_path).with_suffix("")


def _tail(path: Path, limit: int = 2000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return text[-limit:]


def _append_log(path: Path, title: str, text: str) -> None:
    if not text:
        return
    with path.open("a", encoding="utf-8", errors="replace") as f:
        f.write(f"\n----- {title} -----\n")
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def _is_context_or_memory_error(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in [
        "out of memory",
        "memory allocation",
        "failed to allocate",
        "context",
        "ctx",
        "kv cache",
        "vram",
    ])


class LlamaPaths:
    def __init__(self, model_path: str | None) -> None:
        self.llama_dir = Path(os.environ.get("BEATSYNC_QWEN_LLAMA_DIR", str(DEFAULT_LLAMA_DIR)))
        self.server_exe = self.llama_dir / "llama-server.exe"
        self.mtmd_exe = self.llama_dir / "llama-mtmd-cli.exe"
        self.list_exe = self.llama_dir / "llama-cli.exe"
        self.model = self._resolve_model(model_path)
        self.mmproj = self._resolve_mmproj()

    def _resolve_model(self, model_path: str | None) -> Path:
        env_model = os.environ.get("BEATSYNC_QWEN_LLAMA_MODEL")
        if env_model:
            return Path(env_model)
        if model_path:
            requested = Path(model_path)
            if requested.is_file() and requested.suffix.lower() == ".gguf":
                return requested
            candidates = [
                requested / DEFAULT_MODEL.name,
                requested.parent / DEFAULT_MODEL.name,
                DEFAULT_MODEL,
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
        return DEFAULT_MODEL

    def _resolve_mmproj(self) -> Path:
        env_mmproj = os.environ.get("BEATSYNC_QWEN_LLAMA_MMPROJ")
        if env_mmproj:
            return Path(env_mmproj)
        candidates = [
            self.model.parent / DEFAULT_MMPROJ.name,
            DEFAULT_MMPROJ,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return DEFAULT_MMPROJ

    def validate(self) -> None:
        missing = [
            str(path)
            for path in [self.server_exe, self.mtmd_exe, self.model, self.mmproj]
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError("Missing llama.cpp/Qwen files: " + "; ".join(missing))

    @property
    def model_id(self) -> str:
        return self.model.stem


def _llama_env(paths: LlamaPaths) -> Dict[str, str]:
    env = os.environ.copy()
    path_parts = [str(paths.llama_dir)]
    for part in env.get("PATH", "").split(os.pathsep):
        if part and os.path.normcase(os.path.abspath(part)) != os.path.normcase(str(paths.llama_dir)):
            path_parts.append(part)
    env["PATH"] = os.pathsep.join(path_parts)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _parse_devices(text: str) -> List[Dict[str, Any]]:
    devices = []
    pattern = re.compile(r"^\s*(Vulkan\d+):\s*(.+?)\s*\((\d+)\s+MiB,\s*(\d+)\s+MiB free\)", re.I)
    for line in text.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        name = match.group(2).strip()
        lower = name.lower()
        integrated = any(token in lower for token in [
            "radeon(tm) graphics",
            "integrated",
            "uhd",
            "iris",
            "vega",
        ])
        discrete = any(token in lower for token in [
            "nvidia",
            "geforce",
            "rtx",
            "gtx",
            "quadro",
            "radeon rx",
            "arc",
        ]) and not integrated
        devices.append({
            "id": match.group(1),
            "name": name,
            "total_mib": int(match.group(3)),
            "free_mib": int(match.group(4)),
            "discrete": discrete,
            "integrated": integrated,
        })
    return devices


def _adaptive_llama_slots(device: Dict[str, Any] | None = None) -> int:
    if "BEATSYNC_QWEN_LLAMA_SLOTS" in os.environ:
        return _llama_slots()
    if not device:
        return 4
    free_mib = int(device.get("free_mib") or 0)
    if free_mib >= 14 * 1024:
        return 16
    if free_mib >= 10 * 1024:
        return 8
    if free_mib >= 6 * 1024:
        return 4
    return 2


def _select_vulkan_device(paths: LlamaPaths) -> Tuple[str | None, str]:
    override = os.environ.get("BEATSYNC_QWEN_LLAMA_DEVICE", "").strip()
    if override:
        if override.lower() in {"none", "cpu"}:
            return None, "CPU/no Vulkan override"
        return override, f"{override} (env override)"

    try:
        result = subprocess.run(
            [str(paths.list_exe), "--list-devices"],
            cwd=str(paths.llama_dir),
            env=_llama_env(paths),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
    except Exception as exc:
        return None, f"device list unavailable: {exc}"

    devices = _parse_devices((result.stdout or "") + "\n" + (result.stderr or ""))
    if not devices:
        return None, "no Vulkan devices reported"

    discrete = [device for device in devices if device["discrete"]]
    pool = discrete or devices
    selected = max(pool, key=lambda item: (item["free_mib"], item["total_mib"]))
    return selected["id"], f"{selected['id']}: {selected['name']}"


def _select_vulkan_device_info(paths: LlamaPaths) -> Tuple[str | None, str, Dict[str, Any] | None]:
    override = os.environ.get("BEATSYNC_QWEN_LLAMA_DEVICE", "").strip()
    if override:
        if override.lower() in {"none", "cpu"}:
            return None, "CPU/no Vulkan override", None
        return override, f"{override} (env override)", None

    try:
        result = subprocess.run(
            [str(paths.list_exe), "--list-devices"],
            cwd=str(paths.llama_dir),
            env=_llama_env(paths),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
    except Exception as exc:
        return None, f"device list unavailable: {exc}", None

    devices = _parse_devices((result.stdout or "") + "\n" + (result.stderr or ""))
    if not devices:
        return None, "no Vulkan devices reported", None

    discrete = [device for device in devices if device["discrete"]]
    pool = discrete or devices
    selected = max(pool, key=lambda item: (item["free_mib"], item["total_mib"]))
    return selected["id"], f"{selected['id']}: {selected['name']}", selected


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_json(method: str, url: str, payload: Dict | None = None, timeout: float = 30.0) -> Dict:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
    if not raw.strip():
        return {}
    return json.loads(raw)


class LlamaServerClient:
    def __init__(
        self,
        paths: LlamaPaths,
        device: str | None,
        ctx_size: int,
        response_path: str,
        slots: int,
    ) -> None:
        self.paths = paths
        self.device = device
        self.ctx_size = ctx_size
        self.port = _free_tcp_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.slots = max(1, min(32, int(slots)))
        prefix = _response_prefix(response_path)
        self.stdout_path = prefix.with_name(prefix.name + f"_llama_server_ctx{ctx_size}_stdout.log")
        self.stderr_path = prefix.with_name(prefix.name + f"_llama_server_ctx{ctx_size}_stderr.log")
        self._stdout_handle = self.stdout_path.open("w", encoding="utf-8", errors="replace")
        self._stderr_handle = self.stderr_path.open("w", encoding="utf-8", errors="replace")
        self.process: subprocess.Popen | None = None
        self.load_seconds = 0.0
        self._start()

    def _start(self) -> None:
        args = [
            str(self.paths.server_exe),
            "-m", str(self.paths.model),
            "--mmproj", str(self.paths.mmproj),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--ctx-size", str(self.ctx_size),
            "--batch-size", "2048",
            "--ubatch-size", "512",
            "--gpu-layers", "all",
            "--split-mode", "none",
            "--parallel", str(self.slots),
            "--cont-batching",
            "--timeout", "3600",
            "--mmproj-offload",
            "--reasoning", "off",
            "--alias", "qwen3vl",
            "--log-verbosity", "1",
            "--no-log-prefix",
        ]
        if self.device:
            args.extend(["--device", self.device])

        started = time.perf_counter()
        self.process = subprocess.Popen(
            args,
            cwd=str(self.paths.llama_dir),
            env=_llama_env(self.paths),
            stdout=self._stdout_handle,
            stderr=self._stderr_handle,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

        ready_timeout = _env_int("BEATSYNC_QWEN_LLAMA_SERVER_READY_TIMEOUT", 180, lo=15, hi=900)
        deadline = time.perf_counter() + ready_timeout
        last_error = ""
        while time.perf_counter() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited with {self.process.returncode}; "
                    f"stderr: {_tail(self.stderr_path)}"
                )
            for endpoint in ["/health", "/v1/models"]:
                try:
                    _http_json("GET", self.base_url + endpoint, timeout=2.0)
                    self.load_seconds = time.perf_counter() - started
                    return
                except Exception as exc:
                    last_error = str(exc)
            time.sleep(0.5)
        raise TimeoutError(f"llama-server readiness timed out after {ready_timeout}s: {last_error}")

    def close(self) -> None:
        process = self.process
        self.process = None
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        self._stdout_handle.close()
        self._stderr_handle.close()

    def generate(self, image: Image.Image, prompt: str) -> str:
        if not self.process or self.process.poll() is not None:
            raise RuntimeError("llama-server is not running")
        timeout = float(_env_int("BEATSYNC_QWEN_LLAMA_HTTP_TIMEOUT", 240, lo=30, hi=1800))
        payload = {
            "model": "qwen3vl",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _image_data_uri(image)}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
            "min_p": 0,
            "max_tokens": _max_new_tokens(),
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "beatsync_semantic_tag",
                    "strict": True,
                    "schema": SEMANTIC_SCHEMA,
                },
            },
        }
        try:
            data = _http_json("POST", self.base_url + "/v1/chat/completions", payload, timeout=timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama-server HTTP {exc.code}: {body}") from exc

        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(str(part.get("text") or ""))
                else:
                    parts.append(str(part))
            return "".join(parts)
        return str(content)


class LlamaMtmdClient:
    def __init__(self, paths: LlamaPaths, device: str | None, ctx_size: int, response_path: str) -> None:
        self.paths = paths
        self.device = device
        self.ctx_size = ctx_size
        prefix = _response_prefix(response_path)
        self.stdout_path = prefix.with_name(prefix.name + "_llama_cli_stdout.log")
        self.stderr_path = prefix.with_name(prefix.name + "_llama_cli_stderr.log")
        self.frame_dir = prefix.with_name(prefix.name + "_llama_frames")
        self.frame_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, image: Image.Image, prompt: str, item_id: str) -> str:
        image_path = self.frame_dir / f"{_safe_file_token(item_id)}.png"
        image.save(image_path, format="PNG")
        args = [
            str(self.paths.mtmd_exe),
            "-m", str(self.paths.model),
            "--mmproj", str(self.paths.mmproj),
            "--image", str(image_path),
            "-p", prompt,
            "-n", str(_max_new_tokens()),
            "--ctx-size", str(self.ctx_size),
            "--batch-size", "2048",
            "--ubatch-size", "512",
            "--gpu-layers", "all",
            "--split-mode", "none",
            "--mmproj-offload",
            "--temp", "0",
            "--top-k", "1",
            "--top-p", "1",
            "--min-p", "0",
            "--json-schema", json.dumps(SEMANTIC_SCHEMA, separators=(",", ":")),
            "--no-warmup",
            "--log-verbosity", "1",
            "--no-log-prefix",
        ]
        if self.device:
            args.extend(["--device", self.device])

        timeout = _env_int("BEATSYNC_QWEN_LLAMA_CLI_TIMEOUT", 300, lo=30, hi=3600)
        started = time.perf_counter()
        try:
            result = subprocess.run(
                args,
                cwd=str(self.paths.llama_dir),
                env=_llama_env(self.paths),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired:
            _append_log(self.stderr_path, f"{item_id} timeout", f"Timed out after {timeout}s")
            return ""

        elapsed = time.perf_counter() - started
        _append_log(self.stdout_path, f"{item_id} stdout ({elapsed:.1f}s)", result.stdout or "")
        _append_log(self.stderr_path, f"{item_id} stderr ({elapsed:.1f}s)", result.stderr or "")
        if result.returncode == 3221225786:
            _append_log(
                self.stderr_path,
                f"{item_id} interrupted",
                "llama-mtmd-cli exited with 3221225786 / 0xC000013A, usually an interruption.",
            )
            return ""
        if result.returncode != 0:
            combined = (result.stdout or "") + "\n" + (result.stderr or "")
            fallback_ctx = _ctx_sizes()[-1]
            if self.ctx_size != fallback_ctx and _is_context_or_memory_error(combined):
                _append_log(
                    self.stderr_path,
                    f"{item_id} ctx fallback",
                    f"Retrying llama-mtmd-cli with ctx {fallback_ctx} after exit code {result.returncode}.",
                )
                self.ctx_size = fallback_ctx
                return self.generate(image, prompt, item_id)
            _append_log(self.stderr_path, f"{item_id} exit", f"Exit code {result.returncode}")
            return ""
        return result.stdout or ""


class QwenLlamaClient:
    def __init__(self, model_path: str | None, response_path: str) -> None:
        self.paths = LlamaPaths(model_path)
        self.paths.validate()
        self.device, self.device_label, self.device_info = _select_vulkan_device_info(self.paths)
        self.target_slots = _adaptive_llama_slots(self.device_info)
        self.ctx_size = _ctx_sizes()[0]
        self.server: LlamaServerClient | None = None
        self.cli: LlamaMtmdClient | None = None
        self.load_seconds = 0.0
        self.batch_size = self.target_slots
        self.response_path = response_path
        print(f"Qwen llama.cpp model: {self.paths.model.name}", flush=True)
        print(f"Qwen llama.cpp mmproj: {self.paths.mmproj.name}", flush=True)
        print(f"Qwen llama.cpp Vulkan device: {self.device_label}", flush=True)
        print(f"Qwen llama.cpp target slots: {self.target_slots}", flush=True)
        self._start_server_or_prepare_cli()

    @property
    def model_id(self) -> str:
        return f"{self.paths.model_id} (llama.cpp Vulkan)"

    def _start_server_or_prepare_cli(self) -> None:
        if os.environ.get("BEATSYNC_QWEN_LLAMA_DISABLE_SERVER", "0") == "1":
            self._prepare_cli(_ctx_sizes()[0])
            print("Qwen llama.cpp server disabled; using llama-mtmd-cli fallback", flush=True)
            return

        last_error = ""
        for ctx_size in _ctx_sizes():
            try:
                self.server = LlamaServerClient(
                    self.paths,
                    self.device,
                    ctx_size,
                    self.response_path,
                    self.target_slots,
                )
                self.ctx_size = ctx_size
                self.load_seconds = self.server.load_seconds
                self.batch_size = self.server.slots
                print(
                    f"Qwen llama-server ready: ctx {ctx_size}, slots {self.batch_size}, "
                    f"load {self.load_seconds:.1f}s",
                    flush=True,
                )
                return
            except Exception as exc:
                last_error = str(exc)
                print(f"Qwen llama-server ctx {ctx_size} failed: {last_error[-600:]}", flush=True)
                if self.server:
                    self.server.close()
                    self.server = None
        self._prepare_cli(_ctx_sizes()[-1])
        print(f"Qwen falling back to llama-mtmd-cli: {last_error[-600:]}", flush=True)

    def _prepare_cli(self, ctx_size: int) -> None:
        self.server = None
        self.ctx_size = ctx_size
        self.batch_size = 1
        self.cli = LlamaMtmdClient(self.paths, self.device, ctx_size, self.response_path)

    def restart_server_with_slots(self, slots: int) -> bool:
        if os.environ.get("BEATSYNC_QWEN_LLAMA_DISABLE_SERVER", "0") == "1":
            return False
        if self.server:
            self.server.close()
            self.server = None
        self.cli = None
        self.target_slots = max(1, min(32, int(slots)))
        try:
            self.server = LlamaServerClient(
                self.paths,
                self.device,
                self.ctx_size,
                self.response_path,
                self.target_slots,
            )
            self.load_seconds += self.server.load_seconds
            self.batch_size = self.server.slots
            print(
                f"Qwen llama-server restarted: ctx {self.ctx_size}, slots {self.batch_size}, "
                f"load {self.server.load_seconds:.1f}s",
                flush=True,
            )
            return True
        except Exception as exc:
            print(f"Qwen llama-server restart failed: {str(exc)[-600:]}", flush=True)
            if self.server:
                self.server.close()
                self.server = None
            self._prepare_cli(self.ctx_size)
            return False

    def generate(self, image: Image.Image, prompt: str, item_id: str) -> str:
        if self.server:
            try:
                return self.server.generate(image, prompt)
            except Exception as exc:
                fallback_ctx = _ctx_sizes()[-1]
                if self.ctx_size != fallback_ctx and _is_context_or_memory_error(str(exc)):
                    print(
                        f"Qwen llama-server request hit context/VRAM limits; retrying ctx {fallback_ctx}",
                        flush=True,
                    )
                    self.server.close()
                    self.server = None
                    try:
                        self.server = LlamaServerClient(
                            self.paths,
                            self.device,
                            fallback_ctx,
                            self.response_path,
                            self.target_slots,
                        )
                        self.ctx_size = fallback_ctx
                        self.load_seconds += self.server.load_seconds
                        self.batch_size = self.server.slots
                        return self.server.generate(image, prompt)
                    except Exception as retry_exc:
                        print(f"Qwen llama-server ctx {fallback_ctx} retry failed: {retry_exc}", flush=True)
                print(f"Qwen llama-server request failed; switching to CLI fallback: {exc}", flush=True)
                if self.server:
                    self.server.close()
                self.server = None
                self._prepare_cli(self.ctx_size)
        if not self.cli:
            self._prepare_cli(self.ctx_size)
        return self.cli.generate(image, prompt, item_id)

    def close(self) -> None:
        if self.server:
            self.server.close()
            self.server = None


def _candidate_id(item: Dict, fallback_index: int = 0) -> str:
    return str(item["candidate"].get("id") or fallback_index)


def _generate_with_server(
    client: QwenLlamaClient,
    item: Dict,
    prompt: str,
    fallback_index: int = 0,
) -> Tuple[str, Dict, str]:
    item_id = _candidate_id(item, fallback_index)
    if not client.server:
        return item_id, {}, "llama-server is not running"
    try:
        text = client.server.generate(item["image"], prompt)
    except Exception as exc:
        return item_id, {}, str(exc)
    return item_id, _semantic_from_text(text), ""


def _generate_serial(
    client: QwenLlamaClient,
    item: Dict,
    prompt: str,
    fallback_index: int = 0,
) -> Tuple[str, Dict, str]:
    item_id = _candidate_id(item, fallback_index)
    try:
        text = client.generate(item["image"], prompt, item_id)
    except Exception as exc:
        return item_id, {}, str(exc)
    return item_id, _semantic_from_text(text), ""


def _run_inference_wave(
    client: QwenLlamaClient,
    wave_items: List[Dict],
    prompt: str,
    base_index: int,
) -> Dict[str, Dict]:
    if not wave_items:
        return {}

    semantics: Dict[str, Dict] = {}
    failed: List[Tuple[int, Dict, str]] = []
    server = client.server
    concurrency = max(1, int(client.batch_size or 1))

    if server and concurrency > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_item = {
                executor.submit(_generate_with_server, client, item, prompt, base_index + offset): (offset, item)
                for offset, item in enumerate(wave_items, 1)
            }
            for future in concurrent.futures.as_completed(future_to_item):
                offset, item = future_to_item[future]
                try:
                    item_id, semantic, error = future.result()
                except Exception as exc:
                    item_id, semantic, error = _candidate_id(item, base_index + offset), {}, str(exc)
                if semantic:
                    semantics[item_id] = semantic
                else:
                    failed.append((offset, item, error))
    else:
        for offset, item in enumerate(wave_items, 1):
            item_id, semantic, error = _generate_serial(client, item, prompt, base_index + offset)
            if semantic:
                semantics[item_id] = semantic
            else:
                failed.append((offset, item, error))

    if failed and client.server:
        retry_failed: List[Tuple[int, Dict, str]] = []
        for offset, item, _error in failed:
            item_id, semantic, error = _generate_with_server(client, item, prompt, base_index + offset)
            if semantic:
                semantics[item_id] = semantic
            else:
                retry_failed.append((offset, item, error))
        failed = retry_failed

    valid_ratio = len(semantics) / max(1, len(wave_items))
    if failed and client.server and client.batch_size > 1 and valid_ratio < 0.70:
        reduced_slots = max(1, int(client.batch_size) // 2)
        print(
            f"Qwen llama.cpp wave valid ratio {valid_ratio:.0%}; retrying with {reduced_slots} slots",
            flush=True,
        )
        if client.restart_server_with_slots(reduced_slots):
            return _run_inference_wave(client, wave_items, prompt, base_index)

    if failed and not client.server:
        for offset, item, _error in failed:
            item_id, semantic, _error = _generate_serial(client, item, prompt, base_index + offset)
            if semantic:
                semantics[item_id] = semantic

    return semantics


def _run_semantics_for_video(
    *,
    client: QwenLlamaClient,
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
            wave_size = max(1, int(client.batch_size or 1))
            wave_items = frame_items[idx:idx + wave_size]
            semantics.update(_run_inference_wave(client, wave_items, prompt, idx))
            idx += len(wave_items)
            elapsed = max(0.001, time.perf_counter() - inference_started)
            rate = idx / elapsed
            print(
                f"Qwen llama.cpp tagged {idx}/{len(frame_items)} "
                f"({rate:.2f}/s, batch {client.batch_size})",
                flush=True,
            )
        elapsed = max(0.001, time.perf_counter() - inference_started)
        timings["inference_seconds"] = elapsed
        timings["tag_count"] = len(semantics)
        print(
            f"Qwen llama.cpp semantic inference total: {len(semantics)}/{len(frame_items)} tags "
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

    model_path = request.get("qwen_model_path")
    audio_profile = request.get("audio_profile") or {}
    prompt = _build_prompt(audio_profile)

    client = QwenLlamaClient(model_path, args.response)
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
    try:
        for job_index, job in enumerate(jobs, 1):
            job_id = str(job.get("job_id", job_index))
            video_file = job["video_file"]
            fps = float(job.get("fps") or 24.0)
            candidates: List[Dict] = job.get("candidates") or []
            print(
                f"Qwen llama.cpp job {job_index}/{len(jobs)}: {os.path.basename(video_file)} "
                f"({len(candidates)} candidates)",
                flush=True,
            )
            semantics, timings = _run_semantics_for_video(
                client=client,
                video_file=video_file,
                fps=fps,
                candidates=candidates,
                prompt=prompt,
            )
            semantics_by_job[job_id] = semantics
            timings_by_job[job_id] = timings
    finally:
        client.close()

    total_seconds = time.perf_counter() - whole_started
    response = {
        "model_load_seconds": client.load_seconds,
        "model_id": client.model_id,
        "batch_size": client.batch_size,
        "peak_vram_gb": 0.0,
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
