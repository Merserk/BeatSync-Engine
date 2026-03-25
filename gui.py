#!/usr/bin/env python3
import asyncio
import asyncio.proactor_events
import base64
import datetime
import multiprocessing
import os
import re
import shutil
import socket
import subprocess
import sys
from urllib.parse import unquote, urlparse
from pathlib import Path
from typing import Dict, List, Tuple, TypeAlias

import gradio as gr
import librosa

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from runtime_env import configure_portable_runtime

RUNTIME = configure_portable_runtime(SCRIPT_DIR)
PORTABLE_CUDA_DIR = RUNTIME.portable_cuda_dir
USING_PORTABLE_CUDA = RUNTIME.using_portable_cuda
PORTABLE_PYTHON_DIR = RUNTIME.portable_python_dir
PORTABLE_PYTHON_EXE = RUNTIME.portable_python_exe
USING_PORTABLE_PYTHON = RUNTIME.using_portable_python

if USING_PORTABLE_CUDA and PORTABLE_CUDA_DIR:
    print(f"Using Portable CUDA: {PORTABLE_CUDA_DIR}")
else:
    print(f"Portable CUDA not found under: {os.path.join(SCRIPT_DIR, 'bin', 'CUDA')}")
    print("   Will try to use system CUDA if available")
if RUNTIME.cuda_notice:
    print(f"   Note: {RUNTIME.cuda_notice}")

if USING_PORTABLE_PYTHON and PORTABLE_PYTHON_EXE:
    print(f"Using Portable Python: {PORTABLE_PYTHON_EXE}")
else:
    print("Portable Python not found, using system Python")

from ffmpeg_processing import DEFAULT_STANDARD_QUALITY, FFMPEG_FOUND, FFMPEG_PATH, check_nvenc_support, create_lossless_delivery_mp4, get_video_fps, get_video_resolution, is_browser_playable_video, normalize_quality_profile
from video_processor import CPU_COUNT, MAX_THREADS, PARALLEL_WORKERS, create_music_video, estimate_threads_per_job, get_local_temp_dir, get_video_files
from manual_mode import analyze_beats_manual, process_manual_intensity
from smart_mode import analyze_beats_smart, get_gpu_info, get_preset_info, is_gpu_available, select_beats_smart, set_gpu_mode
from auto_mode import analyze_beats_auto
from ui_content import *

GPU_RUNTIME_AVAILABLE = is_gpu_available()
CPU_ONLY_MODE = not GPU_RUNTIME_AVAILABLE
NVENC_AVAILABLE = GPU_RUNTIME_AVAILABLE and check_nvenc_support()
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEFAULT_GUI_PORT = 7860
RESOLUTION_PRESET_CHOICES = [
    ("Default (match first source video)", "default"),
    ("16:9 | 1280x720 (HD)", "1280x720"),
    ("16:9 | 1920x1080 (Full HD)", "1920x1080"),
    ("16:9 | 2560x1440 (QHD)", "2560x1440"),
    ("16:9 | 3840x2160 (4K UHD)", "3840x2160"),
    ("21:9 | 2560x1080 (UltraWide HD)", "2560x1080"),
    ("21:9 | 3440x1440 (UltraWide QHD)", "3440x1440"),
    ("21:9 | 3840x1600 (UltraWide 1600p)", "3840x1600"),
    ("9:16 | 720x1280 (Vertical HD)", "720x1280"),
    ("9:16 | 1080x1920 (Vertical Full HD)", "1080x1920"),
    ("9:16 | 1440x2560 (Vertical QHD)", "1440x2560"),
]

if hasattr(gr, "set_static_paths"):
    gr.set_static_paths(paths=[OUTPUT_DIR])

print(
    get_startup_header(
        CPU_COUNT,
        estimate_threads_per_job(PARALLEL_WORKERS),
        PARALLEL_WORKERS,
        RUNTIME.python_runtime_label if USING_PORTABLE_PYTHON else f"System ({sys.executable})",
        RUNTIME.cuda_runtime_label,
        librosa.__version__,
        "Portable (bin/ffmpeg/ffmpeg.exe)" if FFMPEG_FOUND else "System FFmpeg (portable not found)",
        GPU_RUNTIME_AVAILABLE,
        get_gpu_info(),
        NVENC_AVAILABLE,
    )
)
print(f"   Temp Directory: {get_local_temp_dir()}")
print(f"   Output Directory: {OUTPUT_DIR}")


def find_available_local_port(
    preferred_port: int,
    host: str = "127.0.0.1",
    fallback_count: int = 20,
) -> int:
    """Return the first available localhost port starting at preferred_port."""
    last_error = None
    for port in range(preferred_port, preferred_port + max(1, fallback_count) + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
                return port
            except OSError as exc:
                last_error = exc

    raise OSError(
        f"Could not find an available localhost port in range {preferred_port}-{preferred_port + max(1, fallback_count)}"
    ) from last_error


def resolve_gui_port() -> int:
    """Resolve the Gradio port, honoring GRADIO_SERVER_PORT when valid."""
    raw_port = os.environ.get("GRADIO_SERVER_PORT", "").strip()
    if raw_port:
        try:
            preferred_port = int(raw_port)
        except ValueError:
            print(f"Warning: Invalid GRADIO_SERVER_PORT={raw_port!r}; falling back to {DEFAULT_GUI_PORT}")
            preferred_port = DEFAULT_GUI_PORT
    else:
        preferred_port = DEFAULT_GUI_PORT

    return find_available_local_port(preferred_port=preferred_port)
print(f"{CONSOLE_SEPARATOR}\n")

StatusResult: TypeAlias = Tuple[str, str, Dict]
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac"}
STANDARD_QUALITY_LABELS = [("Fast", "fast"), ("Balanced", "balanced"), ("High", "high")]
POWERSHELL_EXECUTABLE = shutil.which("powershell.exe") or shutil.which("powershell") or "powershell"


def parse_resolution_choice(resolution_choice: str | None) -> tuple[int, int] | None:
    normalized = (resolution_choice or "default").strip().lower()
    if not normalized or normalized == "default":
        return None

    width_str, height_str = normalized.split("x", 1)
    width = int(width_str)
    height = int(height_str)
    if width <= 0 or height <= 0:
        raise ValueError("Custom resolution must use positive width and height values.")
    return width, height


def normalize_local_path(path_value: str) -> str | None:
    if not path_value:
        return None
    normalized = path_value.strip().strip('"').strip("'")
    if not normalized:
        return None

    normalized = unquote(normalized)

    parsed = urlparse(normalized)
    if parsed.scheme and parsed.scheme.lower() == "file":
        if parsed.netloc and parsed.path:
            normalized = f"//{parsed.netloc}{parsed.path}"
        else:
            normalized = parsed.path or normalized

    normalized = os.path.expandvars(os.path.expanduser(normalized))
    normalized = normalized.replace("/", os.sep)

    # Browsers and toolkits sometimes send Windows paths with a leading slash:
    # "/C:/Users/..." should still be treated as an absolute Windows path.
    if re.match(r"^[\\/]+[A-Za-z]:[\\/]", normalized):
        normalized = normalized.lstrip("\\/")

    # If a malformed value already has the project path prepended before a
    # drive-qualified Windows path, keep the last absolute drive path.
    drive_matches = list(re.finditer(r"[A-Za-z]:[\\/]", normalized))
    if len(drive_matches) > 1:
        normalized = normalized[drive_matches[-1].start():]

    if os.path.isabs(normalized):
        return os.path.normpath(normalized)

    return os.path.abspath(normalized)


def get_picker_start_dir(current_path: str | None, select_directory: bool) -> str:
    normalized = normalize_local_path(current_path or "")
    if normalized:
        if select_directory:
            if os.path.isdir(normalized):
                return normalized
        else:
            if os.path.isfile(normalized):
                parent_dir = os.path.dirname(normalized)
                if parent_dir and os.path.isdir(parent_dir):
                    return parent_dir
            if os.path.isdir(normalized):
                return normalized

        parent_dir = os.path.dirname(normalized)
        if parent_dir and os.path.isdir(parent_dir):
            return parent_dir

    home_dir = os.path.expanduser("~")
    return home_dir if os.path.isdir(home_dir) else SCRIPT_DIR


def escape_powershell_string(value: str) -> str:
    return value.replace("'", "''")


def run_native_dialog(powershell_script: str) -> str | None:
    if os.name != "nt":
        raise gr.Error("Native path browsing is currently available on Windows only.")

    encoded_script = base64.b64encode(powershell_script.encode("utf-16-le")).decode("ascii")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    try:
        result = subprocess.run(
            [
                POWERSHELL_EXECUTABLE,
                "-NoProfile",
                "-STA",
                "-ExecutionPolicy",
                "Bypass",
                "-EncodedCommand",
                encoded_script,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            creationflags=creationflags,
        )
    except FileNotFoundError as exc:
        raise gr.Error("Windows PowerShell was not found on this system.") from exc
    except subprocess.TimeoutExpired as exc:
        raise gr.Error("The file picker did not respond in time.") from exc

    if result.returncode != 0:
        stderr_text = (result.stderr or "").strip()
        raise gr.Error(f"Native file picker failed: {stderr_text or 'Unknown PowerShell error'}")

    selected_path = (result.stdout or "").strip()
    return selected_path or None


def browse_for_audio_file(current_path: str) -> str:
    start_dir = escape_powershell_string(get_picker_start_dir(current_path, select_directory=False))
    powershell_script = f"""
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::EnableVisualStyles()
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.InitialDirectory = '{start_dir}'
$dialog.Filter = 'Audio Files (*.mp3;*.wav;*.flac)|*.mp3;*.wav;*.flac|All Files (*.*)|*.*'
$dialog.Title = 'Select Audio File'
$dialog.Multiselect = $false
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    Write-Output $dialog.FileName
}}
"""
    selected_path = run_native_dialog(powershell_script)
    return normalize_local_path(selected_path or current_path or "") or ""


def browse_for_video_folder(current_path: str) -> str:
    start_dir = escape_powershell_string(get_picker_start_dir(current_path, select_directory=True))
    powershell_script = f"""
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::EnableVisualStyles()
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = 'Select Video Folder'
$dialog.SelectedPath = '{start_dir}'
$dialog.ShowNewFolderButton = $false
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    Write-Output $dialog.SelectedPath
}}
"""
    selected_path = run_native_dialog(powershell_script)
    return normalize_local_path(selected_path or current_path or "") or ""


def resolve_inputs(audio_path: str, video_folder: str, session_state: dict) -> tuple[str, List[str], dict]:
    resolved_audio = normalize_local_path(audio_path)
    if not resolved_audio:
        raise ValueError("Enter a local audio file path.")
    if not os.path.isfile(resolved_audio):
        raise FileNotFoundError(f"Audio file not found: {resolved_audio}")
    if Path(resolved_audio).suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError("Audio file must be MP3, WAV, or FLAC.")

    resolved_video_folder = normalize_local_path(video_folder)
    if not resolved_video_folder:
        raise ValueError("Enter a local video folder path.")
    if not os.path.isdir(resolved_video_folder):
        raise NotADirectoryError(f"Video folder not found: {resolved_video_folder}")

    resolved_video_paths = get_video_files(resolved_video_folder)

    session_state["resolved_audio_path"] = resolved_audio
    session_state["resolved_video_folder"] = resolved_video_folder
    session_state["resolved_video_paths"] = resolved_video_paths
    return resolved_audio, resolved_video_paths, session_state


def create_browser_preview(output_path: str, preview_path: str) -> str:
    if is_browser_playable_video(output_path):
        print("   Output is already browser-playable; no preview conversion needed.")
        return output_path

    attempts: List[tuple[str, List[str]]] = []
    preferred_hwaccel = ["-hwaccel", "cuda"] if not CPU_ONLY_MODE else ["-hwaccel", "auto"]
    cpu_fallback_hwaccels = [preferred_hwaccel]
    if not CPU_ONLY_MODE:
        cpu_fallback_hwaccels.append(["-hwaccel", "auto"])

    if NVENC_AVAILABLE:
        attempts.append(
            (
                "NVENC",
                [
                    FFMPEG_PATH,
                    *preferred_hwaccel,
                    "-i",
                    output_path,
                    "-c:v",
                    "h264_nvenc",
                    "-preset",
                    "p5",
                    "-cq",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-y",
                    preview_path,
                ],
            )
        )

    for hwaccel_args in cpu_fallback_hwaccels:
        attempt_name = "CPU (CUDA decode)" if hwaccel_args == ["-hwaccel", "cuda"] else "CPU"
        attempts.append(
            (
                attempt_name,
                [
                    FFMPEG_PATH,
                    *hwaccel_args,
                    "-i",
                    output_path,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-movflags",
                    "+faststart",
                    "-y",
                    preview_path,
                ],
            )
        )

    for encoder_name, preview_cmd in attempts:
        try:
            if os.path.exists(preview_path):
                os.remove(preview_path)
        except OSError:
            pass

        result = subprocess.run(preview_cmd, capture_output=True, text=True, timeout=180)
        if result.returncode == 0 and os.path.exists(preview_path):
            print(f"   Preview created for Gradio with {encoder_name}.")
            return preview_path

        stderr_tail = (result.stderr or "").strip().splitlines()
        error_line = stderr_tail[-1] if stderr_tail else "unknown FFmpeg error"
        print(f"   Preview creation with {encoder_name} failed: {error_line}")

    print("   Preview generation failed. Returning the original output path instead.")
    return output_path


def configure_asyncio_exception_filter() -> None:
    """Ignore benign Windows socket reset noise from closed browser transports."""
    if os.name == "nt":
        transport_type = getattr(asyncio.proactor_events, "_ProactorBasePipeTransport", None)
        original_method = getattr(transport_type, "_call_connection_lost", None)

        if transport_type and original_method and not getattr(original_method, "_beatsync_wrapped", False):
            def wrapped_call_connection_lost(self, exc):
                try:
                    return original_method(self, exc)
                except ConnectionResetError as reset_exc:
                    if getattr(reset_exc, "winerror", None) == 10054:
                        return None
                    raise

            wrapped_call_connection_lost._beatsync_wrapped = True
            transport_type._call_connection_lost = wrapped_call_connection_lost

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    def exception_handler(active_loop: asyncio.AbstractEventLoop, context: dict) -> None:
        exc = context.get("exception")
        message = str(context.get("message", ""))
        if (
            isinstance(exc, ConnectionResetError)
            and getattr(exc, "winerror", None) == 10054
            and "_ProactorBasePipeTransport._call_connection_lost" in message
        ):
            return

        active_loop.default_exception_handler(context)

    loop.set_exception_handler(exception_handler)


def process_video(audio_path: str, video_folder: str, generation_mode: str, cut_intensity: float, smart_preset: str, output_filename: str, direction: str, playback_speed_str: str, timing_offset: float, parallel_workers: int, processing_mode: str, standard_quality: str, create_prores_delivery_mp4: bool, custom_resolution: str, custom_fps: float, session_state: dict) -> StatusResult:
    try:
        session_state = session_state or {}
        resolved_audio_path, resolved_video_paths, session_state = resolve_inputs(
            audio_path, video_folder, session_state
        )

        use_gpu = GPU_RUNTIME_AVAILABLE
        set_gpu_mode(use_gpu)

        is_prores = processing_mode == "prores_proxy"
        use_nvenc = processing_mode in ["h264_nvenc", "hevc_nvenc"] and NVENC_AVAILABLE and use_gpu
        gpu_encoder = processing_mode if use_nvenc else "none"
        quality = normalize_quality_profile(standard_quality)
        threads_per_job = estimate_threads_per_job(parallel_workers)

        if generation_mode == "manual":
            mode_str, smart_mode = "MANUAL MODE", False
        elif generation_mode == "smart":
            mode_str, smart_mode = "SMART MODE", True
        else:
            mode_str, smart_mode = "AUTO MODE", False

        if is_prores:
            codec_info = "ProRes 422 Proxy (.mov) - Lossless"
            encoder_info = "Lossless Concatenation"
        elif use_nvenc:
            codec_info = f"{gpu_encoder.upper()} (.mp4) | {quality.capitalize()} quality"
            encoder_info = f"{gpu_encoder.upper()} | {quality.capitalize()}"
        else:
            codec_info = f"H.264 (.mp4) | {quality.capitalize()} quality"
            encoder_info = f"libx264 | {quality.capitalize()}"

        accel_str = "GPU ACCELERATED" if use_gpu else "CPU MODE"
        python_str = "Portable" if USING_PORTABLE_PYTHON else "System"
        cuda_str = "Portable" if USING_PORTABLE_CUDA else "System/None"

        output_fps = custom_fps if custom_fps is not None and custom_fps > 0 else get_video_fps(resolved_video_paths[0])
        selected_target_size = parse_resolution_choice(custom_resolution)
        resolved_target_size = selected_target_size or get_video_resolution(resolved_video_paths[0])
        resolution_info = (
            f"{resolved_target_size[0]}x{resolved_target_size[1]} (custom)"
            if selected_target_size is not None
            else f"{resolved_target_size[0]}x{resolved_target_size[1]} (auto-detected)"
        )

        name, _ = os.path.splitext(output_filename or "music_video.mp4")
        safe_name = os.path.basename(name) or "music_video"
        ext = ".mov" if is_prores else ".mp4"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_{timestamp}{ext}"
        output_path = os.path.join(OUTPUT_DIR, filename)

        speed_factor = {"Half Speed": 0.5, "Double Speed": 2.0}.get(playback_speed_str, 1.0)

        print(f"\n{CONSOLE_SEPARATOR}")
        print(f"BEAT ANALYSIS - {mode_str} ({accel_str})")
        print(f"{CONSOLE_SEPARATOR}")

        if generation_mode == "manual":
            beat_times, beat_info = analyze_beats_manual(resolved_audio_path, use_gpu=use_gpu)
            selected_beats = process_manual_intensity(beat_times, cut_intensity)
            intensity_param = cut_intensity
        elif generation_mode == "smart":
            beat_times, beat_info = analyze_beats_smart(resolved_audio_path)
            selected_beats = select_beats_smart(beat_info, preset=smart_preset)
            intensity_param = smart_preset
        else:
            selected_beats, beat_info = analyze_beats_auto(resolved_audio_path, use_gpu=use_gpu)
            intensity_param = "auto"
            beat_times = beat_info.get("times", selected_beats)

        print(f"\n{CONSOLE_SEPARATOR}")
        print("VIDEO CREATION")
        print(f"{CONSOLE_SEPARATOR}")

        create_music_video(
            resolved_audio_path,
            resolved_video_paths,
            selected_beats,
            intensity_param,
            output_file=output_path,
            direction=direction,
            speed_factor=speed_factor,
            timing_offset=timing_offset,
            max_workers=parallel_workers,
            smart_mode=smart_mode,
            beat_info=beat_info,
            lossless_mode=is_prores,
            use_gpu=use_gpu,
            gpu_encoder=gpu_encoder,
            fps=output_fps,
            target_size=resolved_target_size,
            quality=quality,
            mode_name=generation_mode,
        )

        preview_path = output_path
        preview_filename = None
        delivery_mp4_filename = None
        delivery_mp4_error = None
        preview_source_path = output_path
        if is_prores:
            if create_prores_delivery_mp4:
                print("Generating optional lossless MP4 delivery file...")
                delivery_mp4_filename = f"{safe_name}_{timestamp}_delivery_lossless.mp4"
                delivery_mp4_path = os.path.join(OUTPUT_DIR, delivery_mp4_filename)
                try:
                    preview_source_path = create_lossless_delivery_mp4(
                        output_path,
                        delivery_mp4_path,
                        prefer_cuda_decode=not CPU_ONLY_MODE,
                    )
                    print(f"   Delivery MP4 created: {delivery_mp4_filename}")
                except Exception as exc:
                    delivery_mp4_filename = None
                    delivery_mp4_error = str(exc)
                    print(f"   Delivery MP4 creation failed: {delivery_mp4_error}")

            print("Generating browser-friendly preview for ProRes output...")
            preview_filename = f"{safe_name}_{timestamp}_preview.mp4"
            preview_path = os.path.join(OUTPUT_DIR, preview_filename)
            preview_path = create_browser_preview(preview_source_path, preview_path)
        elif not is_browser_playable_video(output_path):
            print("Generating browser-friendly preview for non-playable output...")
            preview_filename = f"{safe_name}_{timestamp}_preview.mp4"
            preview_path = os.path.join(OUTPUT_DIR, preview_filename)
            preview_path = create_browser_preview(output_path, preview_path)

        gpu_info = f"GPU: {get_gpu_info()}" if use_gpu else "CPU"
        fps_info = f"{output_fps:.2f} FPS (custom)" if custom_fps else f"{output_fps:.2f} FPS (auto-detected)"
        audio_info = "PCM 24-bit (48kHz)" if is_prores else "AAC 320 kbps (48kHz)"

        if generation_mode == "smart":
            preset_info = get_preset_info(smart_preset)
            total_cuts = len(selected_beats) - 1
            status_msg = get_success_message_smart(
                smart_preset,
                preset_info,
                len(beat_times),
                beat_info.get("tempo", 120),
                total_cuts,
                python_str,
                cuda_str,
                threads_per_job,
                CPU_COUNT,
                parallel_workers,
                gpu_info,
                encoder_info,
                codec_info,
                fps_info,
                filename,
                audio_info,
            )
        elif generation_mode == "auto":
            total_cuts = len(selected_beats) - 1
            sections_info = beat_info.get("selection_info", [])
            status_msg = get_success_message_auto(
                total_cuts,
                len(beat_times),
                beat_info.get("tempo", 120),
                sections_info,
                python_str,
                cuda_str,
                threads_per_job,
                CPU_COUNT,
                parallel_workers,
                gpu_info,
                encoder_info,
                codec_info,
                fps_info,
                filename,
                audio_info,
            )
        else:
            if cut_intensity < 1.0:
                subdivisions = int(1.0 / cut_intensity)
                total_cuts = len(selected_beats) - 1
                status_msg = get_success_message_manual_subdivided(
                    total_cuts,
                    subdivisions,
                    len(beat_times),
                    beat_info.get("tempo", 120),
                    cut_intensity,
                    python_str,
                    cuda_str,
                    threads_per_job,
                    CPU_COUNT,
                    parallel_workers,
                    gpu_info,
                    encoder_info,
                    codec_info,
                    fps_info,
                    filename,
                    audio_info,
                )
            else:
                beats_used = len(selected_beats) - 1
                cut_intensity_int = int(cut_intensity)
                status_msg = get_success_message_manual_skipped(
                    beats_used,
                    cut_intensity_int,
                    len(beat_times),
                    beat_info.get("tempo", 120),
                    cut_intensity,
                    python_str,
                    cuda_str,
                    threads_per_job,
                    CPU_COUNT,
                    parallel_workers,
                    gpu_info,
                    encoder_info,
                    codec_info,
                    fps_info,
                    filename,
                    audio_info,
                )

        print(f"\n{CONSOLE_SEPARATOR}")
        print("PROCESS COMPLETE")
        print(f"{CONSOLE_SEPARATOR}\n")

        if is_prores and delivery_mp4_filename:
            status_msg += f"\nDelivery MP4: {delivery_mp4_filename}"
        elif is_prores and create_prores_delivery_mp4 and delivery_mp4_error:
            status_msg += f"\nDelivery MP4: Failed ({delivery_mp4_error})"
        if preview_filename and os.path.normcase(preview_path) != os.path.normcase(output_path):
            status_msg += f"\nBrowser Preview: {preview_filename}"
        status_msg += f"\nTarget Resolution: {resolution_info}"

        return preview_path, status_msg, session_state
    except Exception as e:
        import traceback

        traceback.print_exc()
        return None, f"Error: {str(e)}", session_state


def cleanup_on_startup():
    session_temp_base = get_local_temp_dir()
    try:
        if os.path.exists(session_temp_base):
            print("Cleaning up old session directories...")
            for item in os.listdir(session_temp_base):
                item_path = os.path.join(session_temp_base, item)
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                        print(f"   Removed: {item}")
                except Exception as e:
                    print(f"   Could not remove {item}: {e}")
            print("   Old sessions cleared.")
        else:
            os.makedirs(session_temp_base, exist_ok=True)
            print("   Created session temp directory")
    except Exception as e:
        print(f"   Warning: Could not clean up sessions: {e}")
        os.makedirs(session_temp_base, exist_ok=True)


def create_ui() -> gr.Blocks:
    python_status = (
        "✅ Portable (bin/python-3.13.9-embed-amd64/)"
        if USING_PORTABLE_PYTHON
        else "⚠️  System Python"
    )
    cuda_status = (
        f"✅ {RUNTIME.cuda_runtime_label}"
        if USING_PORTABLE_CUDA
        else "⚠️  System CUDA (or not available)"
    )
    ffmpeg_status = "✅ Portable (bin/ffmpeg/)" if FFMPEG_FOUND else "⚠️  System FFmpeg"
    ready_threads = estimate_threads_per_job(PARALLEL_WORKERS)

    app = gr.Blocks(title="BeatSync Engine", theme=gr.themes.Soft())
    with app:
        session_state = gr.State({})

        gr.Markdown(f"# {UI_TITLE}")
        gr.Markdown(UI_MAIN_DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Input Files")
                with gr.Row():
                    audio_input = gr.Textbox(
                        label=LABEL_AUDIO_FILE,
                        info=INFO_AUDIO_FILE,
                        placeholder=r"C:\Music\song.mp3",
                        scale=5,
                    )
                    browse_audio_btn = gr.Button("Browse...", scale=1, min_width=120)

                with gr.Row():
                    video_input = gr.Textbox(
                        label=LABEL_VIDEO_FOLDER,
                        info=INFO_VIDEO_FOLDER,
                        placeholder=r"D:\VideoClips",
                        scale=5,
                    )
                    browse_video_btn = gr.Button("Browse...", scale=1, min_width=120)

                with gr.Group():
                    gr.Markdown("### 🎯 Generation Mode")
                    generation_mode = gr.Radio(
                        choices=[
                            ("🤖 Auto Mode (Recommended)", "auto"),
                            ("🧠 Smart Mode", "smart"),
                            ("⚙️ Manual Mode", "manual"),
                        ],
                        value="auto",
                        label=LABEL_GENERATION_MODE,
                        info=INFO_GENERATION_MODE,
                    )

                    auto_group = gr.Group(visible=True)
                    with auto_group:
                        gr.Markdown(AUTO_MODE_DESCRIPTION)

                    smart_group = gr.Group(visible=False)
                    with smart_group:
                        gr.Markdown(SMART_MODE_DESCRIPTION)
                        smart_preset = gr.Radio(
                            choices=["slower", "slow", "normal", "fast", "faster"],
                            value="normal",
                            label=LABEL_CUT_PRESET,
                            info=INFO_CUT_PRESET,
                        )

                    manual_group = gr.Group(visible=False)
                    with manual_group:
                        gr.Markdown(MANUAL_MODE_DESCRIPTION)
                        cut_intensity = gr.Slider(
                            minimum=0.1,
                            maximum=16,
                            value=4,
                            step=0.1,
                            label=LABEL_CUT_INTENSITY,
                            info=INFO_CUT_INTENSITY,
                        )

                with gr.Group():
                    gr.Markdown("### ⚙️ Video Settings")
                    direction = gr.Radio(choices=["forward", "backward", "random"], value="forward", label=LABEL_DIRECTION, info=INFO_DIRECTION)
                    playback_speed = gr.Radio(choices=["Normal Speed", "Half Speed", "Double Speed"], value="Normal Speed", label=LABEL_PLAYBACK_SPEED, info=INFO_PLAYBACK_SPEED)
                    timing_offset = gr.Slider(minimum=-0.5, maximum=0.5, value=0.0, step=0.01, label=LABEL_TIMING_OFFSET, info=INFO_TIMING_OFFSET)
                    custom_resolution = gr.Dropdown(
                        choices=RESOLUTION_PRESET_CHOICES,
                        value="default",
                        label=LABEL_CUSTOM_RESOLUTION,
                        info=INFO_CUSTOM_RESOLUTION,
                    )
                    custom_fps = gr.Number(label=LABEL_CUSTOM_FPS, value=None, precision=2, info=INFO_CUSTOM_FPS)

                with gr.Group():
                    gr.Markdown("### 🎬 Processing Mode")
                    if NVENC_AVAILABLE:
                        processing_mode = gr.Radio(
                            choices=[("NVIDIA NVENC H.264", "h264_nvenc"), ("NVIDIA NVENC HEVC (H.265)", "hevc_nvenc"), ("CPU (H.264)", "cpu"), ("ProRes 422 Proxy (Precise Mode)", "prores_proxy")],
                            value="h264_nvenc",
                            label=LABEL_PROCESSING_MODE,
                            info=get_processing_mode_info_nvenc(),
                        )
                    else:
                        processing_mode = gr.Radio(
                            choices=[("CPU (H.264)", "cpu"), ("ProRes 422 Proxy (Precise Mode)", "prores_proxy")],
                            value="cpu",
                            label=LABEL_PROCESSING_MODE,
                            info=get_processing_mode_info_cpu(),
                        )
                    standard_quality = gr.Radio(
                        choices=STANDARD_QUALITY_LABELS,
                        value=DEFAULT_STANDARD_QUALITY,
                        label=LABEL_STANDARD_QUALITY,
                        info=INFO_STANDARD_QUALITY,
                    )
                    prores_delivery_mp4 = gr.Checkbox(
                        value=False,
                        visible=False,
                        label=LABEL_PRORES_DELIVERY_MP4,
                        info=INFO_PRORES_DELIVERY_MP4,
                    )

                with gr.Group():
                    gr.Markdown("### ⚙️ Performance Settings")
                    parallel_workers = gr.Slider(
                        minimum=1,
                        maximum=min(16, max(CPU_COUNT // 2, 4)),
                        value=PARALLEL_WORKERS,
                        step=1,
                        label=get_parallel_workers_label(PARALLEL_WORKERS),
                        info=get_parallel_workers_info(),
                    )

                with gr.Group():
                    gr.Markdown("### 📁 Output Settings")
                    output_filename = gr.Textbox(value="music_video.mp4", label=LABEL_OUTPUT_FILENAME, info=INFO_OUTPUT_FILENAME)

                process_btn = gr.Button("🎬 Create Music Video", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### 📺 Output")
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value=get_ready_status(
                        python_status,
                        cuda_status,
                        ready_threads,
                        CPU_COUNT,
                        ffmpeg_status,
                        GPU_RUNTIME_AVAILABLE,
                        get_gpu_info(),
                        NVENC_AVAILABLE,
                    ),
                    lines=16,
                    max_lines=25,
                )
                video_output = gr.Video(label="Generated Music Video", interactive=False)

        def toggle_mode(mode):
            return {
                manual_group: gr.update(visible=mode == "manual"),
                smart_group: gr.update(visible=mode == "smart"),
                auto_group: gr.update(visible=mode == "auto"),
            }

        def toggle_processing_options(mode):
            return gr.update(visible=mode == "prores_proxy")

        generation_mode.change(fn=toggle_mode, inputs=[generation_mode], outputs=[manual_group, smart_group, auto_group])
        processing_mode.change(
            fn=toggle_processing_options,
            inputs=[processing_mode],
            outputs=[prores_delivery_mp4],
        )
        browse_audio_btn.click(
            fn=browse_for_audio_file,
            inputs=[audio_input],
            outputs=[audio_input],
        )
        browse_video_btn.click(
            fn=browse_for_video_folder,
            inputs=[video_input],
            outputs=[video_input],
        )

        process_btn.click(
            fn=process_video,
            inputs=[
                audio_input,
                video_input,
                generation_mode,
                cut_intensity,
                smart_preset,
                output_filename,
                direction,
                playback_speed,
                timing_offset,
                parallel_workers,
                processing_mode,
                standard_quality,
                prores_delivery_mp4,
                custom_resolution,
                custom_fps,
                session_state,
            ],
            outputs=[video_output, status_output, session_state],
        )

    return app


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    configure_asyncio_exception_filter()
    cleanup_on_startup()
    launch_port = resolve_gui_port()
    print("Starting Gradio interface...")
    if launch_port != DEFAULT_GUI_PORT:
        print(f"   Preferred port {DEFAULT_GUI_PORT} is busy; using http://127.0.0.1:{launch_port}")
    else:
        print(f"   URL: http://127.0.0.1:{launch_port}")
    print("   Local path workflow: ENABLED")
    print(f"\n{CONSOLE_SEPARATOR}\n")

    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=launch_port,
        share=False,
        inbrowser=True,
        show_error=True,
    )
