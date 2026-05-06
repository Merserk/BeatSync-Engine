#!/usr/bin/env python3
"""Start the local vLLM OpenAI-compatible server with embedded Python 3.12."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
VLLM_DIR = ROOT_DIR / "bin" / "vllm"
PYTHON_EXE = VLLM_DIR / "python-3.12.10-embed-amd64" / "python.exe"
DEFAULT_MODEL = ROOT_DIR / "bin" / "models" / "Qwen3-VL-2B-Instruct"
CUDA_DIR = ROOT_DIR / "bin" / "CUDA" / "v13.2"
FP8_MIN_NVIDIA_CAPABILITY = 89


def _embedded_env() -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_PATH"] = str(CUDA_DIR)
    env["CUDA_HOME"] = str(CUDA_DIR)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["VLLM_USE_V1"] = env.get("VLLM_USE_V1", "1")
    env["PATH"] = os.pathsep.join([
        str(CUDA_DIR / "bin" / "x64"),
        str(CUDA_DIR / "bin"),
        str(PYTHON_EXE.parent / "Scripts"),
        str(PYTHON_EXE.parent),
        env.get("PATH", ""),
    ])
    return env


def _detect_gpu() -> dict:
    if not PYTHON_EXE.exists():
        return {}
    code = (
        "import json, torch\n"
        "if not torch.cuda.is_available():\n"
        "    print(json.dumps({}))\n"
        "else:\n"
        "    p=torch.cuda.get_device_properties(0)\n"
        "    free,total=torch.cuda.mem_get_info()\n"
        "    major,minor=torch.cuda.get_device_capability(0)\n"
        "    print(json.dumps({"
        "'name': p.name, "
        "'total_gb': total/(1024**3), "
        "'free_gb': free/(1024**3), "
        "'capability_major': major, "
        "'capability_minor': minor, "
        "'cuda': torch.version.cuda"
        "}))\n"
    )
    try:
        output = subprocess.check_output(
            [str(PYTHON_EXE), "-c", code],
            env=_embedded_env(),
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        return json.loads(output.strip() or "{}")
    except Exception as exc:
        print(f"Warning: could not detect GPU memory for vLLM auto profile: {exc}", flush=True)
        return {}


def _auto_profile() -> dict:
    gpu = _detect_gpu()
    total = float(gpu.get("total_gb") or 0.0)
    free = float(gpu.get("free_gb") or total or 0.0)

    if total <= 0:
        return {
            "name": "unknown",
            "gpu_memory_utilization": "0.80",
            "max_model_len": "4096",
            "max_num_seqs": "4",
            "max_num_batched_tokens": "4096",
        }

    if total <= 8.5:
        desired_util, seqs, model_len, batched, reserve = 0.82, 4, 4096, 4096, 0.75
    elif total <= 12.5:
        desired_util, seqs, model_len, batched, reserve = 0.86, 6, 4096, 8192, 0.85
    elif total <= 18.0:
        desired_util, seqs, model_len, batched, reserve = 0.89, 8, 6144, 12288, 1.00
    elif total <= 26.0:
        desired_util, seqs, model_len, batched, reserve = 0.91, 12, 8192, 16384, 1.25
    else:
        desired_util, seqs, model_len, batched, reserve = 0.92, 16, 8192, 24576, 1.50

    free_limit = max(0.60, min(0.94, (free - reserve) / total))
    util = max(0.60, min(desired_util, free_limit))
    return {
        "name": str(gpu.get("name") or "cuda"),
        "total_gb": f"{total:.1f}",
        "free_gb": f"{free:.1f}",
        "capability_major": str(gpu.get("capability_major") or 0),
        "capability_minor": str(gpu.get("capability_minor") or 0),
        "cuda": str(gpu.get("cuda") or ""),
        "gpu_memory_utilization": f"{util:.2f}",
        "max_model_len": str(model_len),
        "max_num_seqs": str(seqs),
        "max_num_batched_tokens": str(batched),
    }


def _pick(value: str | None, env_name: str, profile: dict, key: str) -> str:
    if value:
        return str(value)
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value
    return str(profile[key])


def _fp8_profile(profile: dict) -> dict[str, str | bool]:
    major = int(profile.get("capability_major") or 0)
    minor = int(profile.get("capability_minor") or 0)
    capability = major * 10 + minor
    name = str(profile.get("name") or "")
    is_nvidia = bool(profile.get("cuda")) and (
        "nvidia" in name.lower()
        or "geforce" in name.lower()
        or "rtx" in name.lower()
        or "quadro" in name.lower()
        or "tesla" in name.lower()
    )
    enabled = is_nvidia and capability >= FP8_MIN_NVIDIA_CAPABILITY
    reason = (
        f"NVIDIA compute capability {major}.{minor}"
        if enabled
        else f"requires NVIDIA compute capability >= 8.9, detected {major}.{minor}"
    )
    return {
        "enabled": enabled,
        "reason": reason,
        "quantization": "fp8",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--host", default=os.environ.get("BEATSYNC_VLLM_HOST", "127.0.0.1"))
    parser.add_argument("--port", default=os.environ.get("BEATSYNC_VLLM_PORT", "8000"))
    parser.add_argument("--served-model-name", default=os.environ.get("BEATSYNC_VLLM_MODEL", "Qwen3-VL-2B-Instruct"))
    parser.add_argument("--max-model-len")
    parser.add_argument("--max-num-seqs")
    parser.add_argument("--max-num-batched-tokens")
    parser.add_argument("--gpu-memory-utilization")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not PYTHON_EXE.exists():
        print(
            "ERROR: Embedded Python 3.12 runtime is missing. "
            "Run the vLLM installer after placing "
            "python-3.12.10-embed-amd64.zip in bin\\vllm.",
            file=sys.stderr,
        )
        return 1

    model = Path(args.model).resolve()
    if not model.exists():
        print(f"ERROR: Model path does not exist: {model}", file=sys.stderr)
        return 1

    env = _embedded_env()
    profile = _auto_profile()
    max_model_len = _pick(args.max_model_len, "BEATSYNC_VLLM_MAX_MODEL_LEN", profile, "max_model_len")
    max_num_seqs = _pick(args.max_num_seqs, "BEATSYNC_VLLM_MAX_NUM_SEQS", profile, "max_num_seqs")
    max_num_batched_tokens = _pick(
        args.max_num_batched_tokens,
        "BEATSYNC_VLLM_MAX_NUM_BATCHED_TOKENS",
        profile,
        "max_num_batched_tokens",
    )
    gpu_memory_utilization = _pick(
        args.gpu_memory_utilization,
        "BEATSYNC_VLLM_GPU_MEMORY",
        profile,
        "gpu_memory_utilization",
    )

    command = [
        str(PYTHON_EXE),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        str(args.host),
        "--port",
        str(args.port),
        "--model",
        str(model),
        "--served-model-name",
        str(args.served_model_name),
        "--trust-remote-code",
        "--max-model-len",
        max_model_len,
        "--max-num-seqs",
        max_num_seqs,
        "--max-num-batched-tokens",
        max_num_batched_tokens,
        "--gpu-memory-utilization",
        gpu_memory_utilization,
        "--no-enable-log-requests",
        "--uvicorn-log-level",
        "warning",
    ]
    fp8 = _fp8_profile(profile)
    if fp8["enabled"]:
        command.extend([
            "--quantization",
            str(fp8["quantization"]),
        ])

    print(f"Starting vLLM OpenAI server: http://{args.host}:{args.port}/v1", flush=True)
    print(f"Model: {model}", flush=True)
    print(f"Python: {PYTHON_EXE}", flush=True)
    print(
        "vLLM auto profile: "
        f"gpu={profile.get('name')} "
        f"total={profile.get('total_gb', '?')}GB "
        f"free={profile.get('free_gb', '?')}GB "
        f"gpu_memory_utilization={gpu_memory_utilization} "
        f"max_model_len={max_model_len} "
        f"max_num_seqs={max_num_seqs} "
        f"max_num_batched_tokens={max_num_batched_tokens} "
        f"fp8={'enabled' if fp8['enabled'] else 'disabled'} "
        f"({fp8['reason']})",
        flush=True,
    )
    if args.dry_run:
        print("Command:", " ".join(command), flush=True)
        return 0
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
