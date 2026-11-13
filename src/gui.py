import os
import sys
import contextlib
import asyncio

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def _install_windows_asyncio_connection_reset_filter() -> None:
    """Hide benign Windows asyncio pipe resets after browser/subprocess shutdown.

    On Windows, asyncio's Proactor transport can log a scary traceback when a
    local socket or subprocess pipe is closed by the other side after the real
    work is already complete. The app result is not affected, so suppress only
    that exact WinError 10054 callback and let all other async errors through.
    """
    if os.name != "nt" or getattr(asyncio, "_beatsync_win10054_filter", False):
        return
    asyncio._beatsync_win10054_filter = True

    def is_benign_reset(exc: BaseException | None) -> bool:
        if not isinstance(exc, ConnectionResetError):
            return False
        winerror = getattr(exc, "winerror", None)
        errno_value = getattr(exc, "errno", None)
        return winerror == 10054 or errno_value == 10054 or "WinError 10054" in str(exc)

    # Directly patch the noisy Proactor pipe cleanup callback when available.
    try:
        from asyncio import proactor_events

        transport_cls = getattr(proactor_events, "_ProactorBasePipeTransport", None)
        original_call_lost = getattr(transport_cls, "_call_connection_lost", None)
        if transport_cls is not None and original_call_lost is not None:
            def quiet_call_connection_lost(self, exc):  # type: ignore[no-untyped-def]
                try:
                    return original_call_lost(self, exc)
                except ConnectionResetError as reset_exc:
                    if is_benign_reset(reset_exc):
                        return None
                    raise

            transport_cls._call_connection_lost = quiet_call_connection_lost
    except Exception:
        pass

    # Fallback for the same exception if it still reaches the loop logger.
    original_exception_handler = asyncio.BaseEventLoop.call_exception_handler

    def quiet_exception_handler(self, context):  # type: ignore[no-untyped-def]
        exc = context.get("exception") if isinstance(context, dict) else None
        handle = str(context.get("handle", "")) if isinstance(context, dict) else ""
        if is_benign_reset(exc) and "_ProactorBasePipeTransport._call_connection_lost" in handle:
            return None
        return original_exception_handler(self, context)

    asyncio.BaseEventLoop.call_exception_handler = quiet_exception_handler


_install_windows_asyncio_connection_reset_filter()

from logger import (
    setup_environment,
    USING_PORTABLE_PYTHON
)

# Initialize environment
setup_environment()
# Import modules after the portable environment is set
import gradio as gr
import tempfile
import shutil
import datetime
import multiprocessing
import queue
import re
import threading
import time
import socket
from typing import Callable, Iterator, TypeAlias, Tuple, Dict, List

# Import FFmpeg processing module
from ffmpeg_processing import get_video_fps, create_browser_preview
from ffmpeg_runtime import get_runtime_info
from video_encoders import VULKAN_ENCODERS, NVENC_ENCODERS, HARDWARE_ENCODERS, get_encoder_support

# Shared runtime settings
from gpu_cpu_utils import (
    CPU_COUNT,
    MAX_THREADS,
    PARALLEL_WORKERS,
    GPU_INFO,
    GPU_AVAILABLE,
    NVENC_AVAILABLE,
    set_gpu_mode,
)
from paths import (
    GRADIO_TEMP_DIR,
    get_input_dir,
    get_audio_input_dir,
    get_video_input_dir,
    get_output_dir,
)

gpu_data = GPU_INFO
gpu_info = f"{gpu_data['name']} ({gpu_data['backend']})" if gpu_data['available'] else "CPU Mode"

from video_processor import create_music_video

from auto_mode import analyze_beats_auto

# Import UI content
from ui_content import *

# Set environment variable for Gradio
os.environ['GRADIO_TEMP_DIR'] = GRADIO_TEMP_DIR

VideoFilesInput : TypeAlias = List[str]
StatusResult : TypeAlias = Tuple[str | None, str, Dict]
ExportResult : TypeAlias = Tuple[str | None, str, str | None, Dict]

STATUS_BOX_CSS = """
#status-output-box {
    min-height: 238px !important;
}

#status-output-box textarea {
    height: 186px !important;
    min-height: 186px !important;
    max-height: 186px !important;
    overflow-y: auto !important;
    resize: none !important;
}
"""


def _stage_status(stage_number: int) -> str:
    return f"Stage {stage_number} is processing. Please wait."


class QuietConsole:
    """Discard legacy verbose prints while the Gradio worker runs."""

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        pass


class StageConsoleLogger:
    """Small CMD logger: stage start, up to 5 useful lines, stage end."""

    def __init__(self, stream, max_lines_per_stage: int = 5):
        self.stream = stream
        self.max_lines_per_stage = max(1, int(max_lines_per_stage))
        self.stage_number: int | None = None
        self.stage_started = 0.0
        self.stage_line_count = 0
        self.total_started = time.perf_counter()

    def start_stage(self, stage_number: int) -> None:
        if self.stage_number == stage_number:
            return
        self.end_stage()
        self.stage_number = stage_number
        self.stage_started = time.perf_counter()
        self.stage_line_count = 0
        self._write(f"Stage {stage_number} processing started:\n")

    def stage_line(self, stage_number: int, message: str) -> None:
        if self.stage_number != stage_number:
            self.start_stage(stage_number)
        self.line(message)

    def line(self, message: str) -> None:
        if self.stage_number is None:
            return
        if self.stage_line_count >= self.max_lines_per_stage:
            return
        message = self._clean(message)
        if message:
            self._write(f"  {message}\n")
            self.stage_line_count += 1

    def end_stage(self) -> None:
        if self.stage_number is None:
            return
        elapsed = int(round(time.perf_counter() - self.stage_started))
        self._write(f"Stage {self.stage_number} ended in {elapsed} seconds.\n\n")
        self.stage_number = None
        self.stage_started = 0.0
        self.stage_line_count = 0

    def finish(self) -> None:
        self.end_stage()
        elapsed = int(round(time.perf_counter() - self.total_started))
        self._write(f"Total time processing: {elapsed} seconds\n")

    def _write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()

    def _clean(self, text: str) -> str:
        text = str(text).encode("ascii", "ignore").decode("ascii")
        return re.sub(r"\s+", " ", text).strip()


def _fmt_stage_seconds(seconds: float | int | None) -> str:
    try:
        return f"{float(seconds):.1f}s"
    except Exception:
        return "0.0s"


def _short_model_name(model_id: str | None) -> str:
    if not model_id:
        return ""
    return os.path.basename(str(model_id).rstrip("/\\")) or str(model_id)


def _stage5_summary(console_logger: StageConsoleLogger | None, video_analysis: Dict | None) -> None:
    if console_logger is None or not isinstance(video_analysis, dict):
        return

    source_count = int(video_analysis.get("source_count") or len(video_analysis.get("videos") or []))
    worker_count = int(video_analysis.get("worker_count") or 1)
    cache_hits = int(video_analysis.get("cache_hits") or 0)
    ai_enabled = bool(video_analysis.get("ai_enabled"))
    qwen_total = int(video_analysis.get("qwen_frame_count") or 0)
    qwen_tags = int(video_analysis.get("qwen_tag_count") or 0)
    model_id = _short_model_name(video_analysis.get("qwen_model_id"))
    batch_size = int(video_analysis.get("qwen_concurrency") or 0)

    console_logger.line(f"Source videos: {source_count}, visual workers: {worker_count}")
    if video_analysis.get('pipeline_order') == 'scan_all_then_qwen':
        console_logger.line('Video pipeline: scan all sources one by one, then shared Qwen tagging')
        console_logger.line(f"Qwen sessions this run: {video_analysis.get('qwen_worker_count', 0)}, "
                            f"server starts: {video_analysis.get('qwen_server_start_count', 0)}")
        visual_cache_hits = int(video_analysis.get('visual_cache_hits') or 0)
        if visual_cache_hits:
            console_logger.line(f'Scan/scoring cache reused for {visual_cache_hits} pending AI source(s)')
    if ai_enabled:
        qwen_bits = ["Qwen: enabled"]
        if model_id:
            qwen_bits.append(f"model {model_id}")
        if batch_size:
            qwen_bits.append(f"batch {batch_size}")
        console_logger.line(", ".join(qwen_bits))
        if qwen_total:
            inference_seconds = float(video_analysis.get("qwen_inference_seconds") or video_analysis.get("qwen_seconds") or 0.0)
            qwen_rate = (qwen_total / inference_seconds) if inference_seconds > 0 else 0.0
            peak_vram = float(video_analysis.get("qwen_peak_vram_gb") or 0.0)
            perf_bits = []
            if batch_size:
                perf_bits.append(f"batch {batch_size}")
            if peak_vram > 0:
                perf_bits.append(f"~{peak_vram:.2f} GB VRAM")
            if qwen_rate > 0:
                perf_bits.append(f"{qwen_rate:.2f} candidates/s")
            if perf_bits:
                console_logger.line(f"Qwen performance: {', '.join(perf_bits)}")
            console_logger.line(
                f"Qwen tags: {qwen_tags}/{qwen_total} in {_fmt_stage_seconds(video_analysis.get('qwen_seconds'))}"
            )
    else:
        console_logger.line("Qwen: disabled")

    summary = video_analysis.get("summary")
    if summary:
        console_logger.line(f"Visual library: {summary}")
    console_logger.line(
        f"Analysis time: {_fmt_stage_seconds(video_analysis.get('analysis_seconds'))}, cache {cache_hits}/{source_count}"
    )


def _stage6_summary(console_logger: StageConsoleLogger | None, beat_info: Dict | None) -> None:
    if console_logger is None or not isinstance(beat_info, dict):
        return

    render_info = beat_info.get("render_info") or {}
    if not render_info:
        return

    cuts = int(render_info.get("render_cuts") or 0)
    frames = int(render_info.get("timeline_frames") or 0)
    fps = render_info.get("output_fps")
    if cuts or frames:
        fps_text = f" @ {float(fps):.1f} FPS" if fps is not None else ""
        console_logger.line(f"Render timeline: {cuts} cuts, {frames} frames{fps_text}")

    clip_workers = render_info.get("clip_workers")
    requested_workers = render_info.get("requested_workers")
    worker_text = ""
    if clip_workers:
        worker_text = f", workers {clip_workers}"
        if requested_workers and requested_workers != clip_workers:
            worker_text += f"/{requested_workers}"
    encoder = render_info.get("encoder")
    if encoder or worker_text:
        console_logger.line(f"Encoder: {encoder or 'unknown'}{worker_text}")

    if render_info.get("audio_duration") is not None:
        console_logger.line(f"Audio duration: {float(render_info['audio_duration']):.2f} seconds")

    plan_summary = render_info.get("plan_summary") or {}
    if plan_summary:
        console_logger.line(
            "Planner: "
            f"{int(plan_summary.get('clip_count') or 0)} clips, "
            f"{int(plan_summary.get('source_count') or 0)} sources, "
            f"AI moments {int(plan_summary.get('ai_tagged') or 0)}"
        )
        console_logger.line(
            "Planner pool: "
            f"seed {int(plan_summary.get('highlight_seed_size') or plan_summary.get('highlight_pool_size') or 0)}, "
            f"expanded {int(plan_summary.get('planner_pool_size') or 0)}/"
            f"{int(plan_summary.get('planner_pool_target') or 0)}, "
            f"raw {int(plan_summary.get('raw_candidate_size') or 0)}, "
            f"reuse {int(plan_summary.get('candidate_reuse_count') or 0)}"
        )
        console_logger.line(
            "Unique footage: "
            f"{int(plan_summary.get('visual_verified_clip_count') or 0)}/"
            f"{int(plan_summary.get('clip_count') or 0)} cuts checked, "
            f"repeated scenes {int(plan_summary.get('global_scene_repetition_count') or 0)}, "
            f"source overlaps {int(plan_summary.get('source_overlap_count') or 0)}, "
            f"visual repeats {int(plan_summary.get('visual_duplicate_count') or 0)}, "
            f"verification {float(plan_summary.get('visual_verification_seconds') or 0):.2f}s"
        )
        console_logger.line(
            "Clean source shots: "
            f"{int(plan_summary.get('shot_verified_clip_count') or 0)}/"
            f"{int(plan_summary.get('clip_count') or 0)} verified, "
            f"{int(plan_summary.get('shot_refined_clip_count') or 0)} source intervals adjusted"
        )
        console_logger.line(
            "Avoid text: "
            f"{'on' if plan_summary.get('avoid_text_clips') else 'off'}, "
            f"filtered {int(plan_summary.get('text_candidates_filtered') or 0)}/"
            f"{int(plan_summary.get('text_candidate_count') or 0)}, "
            f"used {int(plan_summary.get('text_overlay_used_count') or 0)}"
        )
        console_logger.line(
            "Speed control: "
            f"{int(plan_summary.get('speed_slowed_count') or 0)} slowed, "
            f"{int(plan_summary.get('speed_sped_up_count') or 0)} sped up, "
            f"avg {float(plan_summary.get('average_playback_speed') or 1.0):.2f}x"
        )

    final_bits = []
    if render_info.get("target_resolution"):
        final_bits.append(f"resolution {render_info['target_resolution']}")
    if render_info.get("final_assembly_seconds") is not None:
        final_bits.append(f"assembly {_fmt_stage_seconds(render_info['final_assembly_seconds'])}")
    if final_bits:
        console_logger.line("Final: " + ", ".join(final_bits))

def find_launch_port(default_port: int = 7860, search_limit: int = 20) -> int:
    """Prefer the default Gradio port, then step forward if it is busy."""
    env_port = os.environ.get("GRADIO_SERVER_PORT")
    if env_port:
        try:
            return int(env_port)
        except ValueError:
            print(f"⚠️ Invalid GRADIO_SERVER_PORT={env_port!r}; using auto port search.")

    for port in range(default_port, default_port + search_limit):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                if port != default_port:
                    print(f"⚠️ Port {default_port} is busy. Starting BeatSync on port {port}.")
                return port

    raise OSError(f"Cannot find empty port in range: {default_port}-{default_port + search_limit - 1}")


def _as_existing_source_path(file_path: str | None) -> str | None:
    """Use the selected source file directly instead of copying it locally."""
    if not file_path:
        return None
    try:
        path = os.path.abspath(os.fspath(file_path))
    except TypeError:
        return None
    return path if os.path.isfile(path) else None


def _as_existing_source_paths(file_paths: VideoFilesInput) -> list[str]:
    if not file_paths:
        return []
    return [path for path in (_as_existing_source_path(p) for p in file_paths) if path]


def _process_video_impl(audio_file: str, video_files: VideoFilesInput,
                       output_filename: str, processing_mode: str,
                       custom_fps: float, auto_speed_clip_control: bool,
                       avoid_text_clips: bool,
                       session_state: dict,
                       progress_callback: Callable[[str], None] | None = None,
                       console_logger: StageConsoleLogger | None = None) -> StatusResult:
    total_started = time.perf_counter()
    session_state['export_path'] = None
    try:
        auto_speed_clip_control = True if auto_speed_clip_control is None else bool(auto_speed_clip_control)
        avoid_text_clips = True if avoid_text_clips is None else bool(avoid_text_clips)
        parallel_workers = PARALLEL_WORKERS

        # Initialize session state if needed
        if 'original_audio_path' not in session_state:
            session_state['original_audio_path'] = None
            session_state['original_video_paths'] = []
        if 'session_dir' not in session_state or not os.path.isdir(session_state['session_dir']):
            session_state['session_dir'] = tempfile.mkdtemp(prefix='beatsync_', dir=GRADIO_TEMP_DIR)
        session_dir = session_state['session_dir']

        # Handle audio by referencing the selected file path directly.
        if audio_file:
            if audio_file != session_state.get('original_audio_path'):
                local_audio_path = _as_existing_source_path(audio_file)
                if local_audio_path:
                    session_state['local_audio_path'] = local_audio_path
                    session_state['original_audio_path'] = audio_file
                else:
                    return None, '❌ Error: Could not access audio file', session_state
            else:
                local_audio_path = session_state.get('local_audio_path')
        else:
            return None, '❌ Error: No audio file selected', session_state

        # Handle videos by referencing selected file paths directly.
        if video_files:
            if video_files != session_state.get('original_video_paths'):
                local_video_paths = _as_existing_source_paths(video_files)
                if local_video_paths:
                    session_state['local_video_paths'] = local_video_paths
                    session_state['original_video_paths'] = video_files
                else:
                    return None, '❌ Error: Could not access video files', session_state
            else:
                local_video_paths = session_state.get('local_video_paths')
        else:
            return None, '❌ Error: No video files selected', session_state

        # Verify files exist
        if not local_audio_path or not os.path.exists(local_audio_path):
             return None, f"❌ Error: Audio file is missing or inaccessible.", session_state
        if not local_video_paths or not all(p and os.path.exists(p) for p in local_video_paths):
             return None, f"❌ Error: Video files are missing or inaccessible.", session_state
        
        # Set GPU mode
        use_gpu = GPU_AVAILABLE
        set_gpu_mode(use_gpu)
        
        # Determine processing mode
        is_prores = processing_mode == 'prores_hq'
        use_nvenc = processing_mode in NVENC_ENCODERS
        use_vulkan = processing_mode in VULKAN_ENCODERS
        if processing_mode in HARDWARE_ENCODERS:
            support = get_encoder_support(processing_mode)
            if not support['available']:
                raise RuntimeError(f"{HARDWARE_ENCODERS[processing_mode]} is unavailable: {support['error']}")
        use_hardware_encoder = use_nvenc or use_vulkan
        gpu_encoder = processing_mode if use_hardware_encoder else 'none'
        
        python_str = "Portable" if USING_PORTABLE_PYTHON else "System"
        gpu_backend_str = "Taichi Vulkan" if GPU_AVAILABLE else "CPU"

        # Determine FPS
        if custom_fps is not None and custom_fps > 0:
            output_fps = custom_fps
        else:
            output_fps = get_video_fps(local_video_paths[0])
            
        # Prepare output paths
        output_folder = get_output_dir()
        os.makedirs(output_folder, exist_ok=True)
        name, _ = os.path.splitext(output_filename)
        ext = '.mov' if is_prores else '.mp4'
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}{ext}"
        output_path = os.path.join(output_folder, filename)
        temp_output = os.path.join(session_dir, filename)

        selected_beats, beat_info = analyze_beats_auto(
            local_audio_path,
            use_gpu=use_gpu,
            video_files=local_video_paths,
            progress_callback=progress_callback,
            console_callback=lambda stage, message: console_logger.stage_line(stage, message) if console_logger else None,
        )
        beat_times = beat_info.get('times', selected_beats)
        _stage5_summary(console_logger, beat_info.get("video_analysis"))

        if progress_callback:
            progress_callback(_stage_status(6))

        # Create video
        result_path = create_music_video(
            local_audio_path, local_video_paths, selected_beats,
            output_file=temp_output, max_workers=parallel_workers,
            beat_info=beat_info, export_format='prores_hq' if is_prores else 'auto',
            use_gpu=use_gpu, gpu_encoder=gpu_encoder, fps=output_fps,
            auto_speed_clip_control=bool(auto_speed_clip_control),
            avoid_text_clips=bool(avoid_text_clips),
        )

        # Move to output folder
        shutil.move(result_path, output_path)
        temp_debug_path = os.path.splitext(result_path)[0] + ".clip_plan_debug.json"
        output_debug_path = os.path.splitext(output_path)[0] + ".clip_plan_debug.json"
        if os.path.exists(temp_debug_path):
            shutil.move(temp_debug_path, output_debug_path)
            render_info = beat_info.setdefault("render_info", {})
            render_info["clip_plan_debug"] = output_debug_path

        session_state['export_path'] = output_path
        preview_path = os.path.join(session_dir, f"{name}_{timestamp}_preview.mp4")
        preview_error = None
        try:
            create_browser_preview(output_path, preview_path)
        except Exception as exc:
            preview_path = None
            preview_error = str(exc)
            print(f'Preview failed; the completed export is available: {exc}')
        _stage6_summary(console_logger, beat_info)

        # Generate status message based on mode
        gpu_info = f"⚡ GPU: {GPU_INFO}" if use_gpu else "💻 CPU"
        fps_info = f"{output_fps:.2f} FPS (custom)" if custom_fps else f"{output_fps:.2f} FPS (auto-detected)"
        audio_info = "PCM 24-bit (48kHz)"
        
        if is_prores:
            codec_info = "ProRes 422 HQ (.mov)"
            encoder_info = "ProRes 422 HQ"
        elif use_hardware_encoder:
            encoder_label = VULKAN_ENCODERS.get(gpu_encoder, gpu_encoder.upper())
            codec_info = f"{encoder_label} (.mp4)"
            encoder_info = f"⚡ {encoder_label}"
        else:
            codec_info = "H.264 (.mp4)"
            encoder_info = "💻 libx264"

        render_info = beat_info.get("render_info") or {}
        total_cuts = int(render_info.get("render_cuts") or max(0, len(selected_beats) - 1))
        sections_info = beat_info.get('selection_info', [])
        total_processing_seconds = time.perf_counter() - total_started
        processing_label = gpu_encoder.upper() if use_hardware_encoder else ("PRORES_HQ" if is_prores else "H264_CPU")
        
        status_msg = get_success_message_auto(
            total_cuts, len(beat_times),
            beat_info.get('tempo', 120), sections_info,
            python_str, gpu_backend_str, MAX_THREADS, CPU_COUNT,
            parallel_workers, gpu_info, encoder_info,
            codec_info, fps_info, filename, audio_info,
            audio_duration=beat_info.get('audio_duration'),
            output_fps=output_fps,
            total_processing_seconds=total_processing_seconds,
            processing_label=processing_label
        )
        if preview_error:
            status_msg += '\nPreview could not be created. The exported video is available to download.'
        # Return preview path for display, keep session_state intact
        return preview_path, status_msg, session_state

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        import traceback
        traceback.print_exc()
        return None, error_msg, session_state


def process_video(audio_file: str, video_files: VideoFilesInput,
                 output_filename: str, processing_mode: str,
                 custom_fps: float, auto_speed_clip_control: bool,
                 avoid_text_clips: bool,
                 session_state: dict) -> Iterator[ExportResult]:
    status_queue: queue.Queue[str | None] = queue.Queue()
    result_queue: queue.Queue[StatusResult] = queue.Queue(maxsize=1)
    initial_status = _stage_status(1)
    console_logger = StageConsoleLogger(sys.__stdout__)
    quiet_console = QuietConsole()

    def progress_callback(message: str) -> None:
        status_queue.put(message)
        match = re.search(r"Stage (\d+) is processing", message)
        if match:
            console_logger.start_stage(int(match.group(1)))

    def worker() -> None:
        try:
            with contextlib.redirect_stdout(quiet_console), contextlib.redirect_stderr(quiet_console):
                result = _process_video_impl(
                    audio_file=audio_file,
                    video_files=video_files,
                    output_filename=output_filename,
                    processing_mode=processing_mode,
                    custom_fps=custom_fps,
                    auto_speed_clip_control=auto_speed_clip_control,
                    avoid_text_clips=avoid_text_clips,
                    session_state=session_state,
                    progress_callback=progress_callback,
                    console_logger=console_logger,
                )
        except Exception as e:
            console_logger.line(f"Error: {e}")
            result = None, f"❌ Error: {e}", session_state
        finally:
            console_logger.finish()
        result_queue.put(result)
        status_queue.put(None)

    thread = threading.Thread(target=worker, daemon=True)
    console_logger.start_stage(1)
    thread.start()

    last_status = initial_status
    yield None, initial_status, None, session_state

    while True:
        message = status_queue.get()
        if message is None:
            break
        if message != last_status:
            last_status = message
            yield None, message, None, session_state

    thread.join()
    preview, status, state = result_queue.get()
    yield preview, status, state.get('export_path'), state


def cleanup_on_startup():
    """
    Clean temporary runtime files on script start while preserving user inputs
    and the persistent video analysis cache.
    """
    input_base = get_input_dir()
    protected_dirs = {'audio', 'video', 'video_analysis_cache'}

    try:
        os.makedirs(get_audio_input_dir(), exist_ok=True)
        os.makedirs(get_video_input_dir(), exist_ok=True)
        os.makedirs(os.path.join(input_base, 'gradio_uploads'), exist_ok=True)

        if os.path.exists(input_base):
            for item in os.listdir(input_base):
                item_path = os.path.join(input_base, item)

                # Keep the latest user input files across restarts.
                if item in protected_dirs:
                    continue

                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                    elif os.path.isfile(item_path):
                        os.remove(item_path)
                except Exception as e:
                    print(f"   ⚠️  Could not clean {item}: {e}")

        # Recreate runtime temp upload folder after cleanup.
        os.makedirs(os.path.join(input_base, 'gradio_uploads'), exist_ok=True)

    except Exception as e:
        print(f"   ⚠️  Warning during startup cleanup: {e}")


def create_ui() -> gr.Blocks:
    # These definitions are needed within the function's scope
    python_status = "✅ Portable (bin/python-3.13.17-embed-amd64/)" if USING_PORTABLE_PYTHON else "⚠️  System Python"
    gpu_backend_status = "✅ Taichi (Vulkan)" if GPU_AVAILABLE else "⚠️ CPU fallback (Vulkan unavailable)"
    ffmpeg_status = f"✅ FFmpeg {get_runtime_info()['version']} (full)"
    
    app = gr.Blocks(title='BeatSync Engine', theme='ocean', css=STATUS_BOX_CSS)
    with app:
        session_state = gr.State({})

        gr.Markdown(f"# {UI_TITLE}")
        gr.Markdown(UI_MAIN_DESCRIPTION)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('### 📁 Input Files')
                audio_input = gr.File(label=LABEL_AUDIO_FILE, file_types=['.mp3', '.wav', '.flac'], type='filepath', elem_id='audio-file-input')
                video_input = gr.File(label=LABEL_VIDEO_FILES, file_count='multiple', file_types=['.mp4', '.mkv'], type='filepath', elem_id='video-files-input')

                with gr.Group():
                    gr.Markdown('### ⚙️ Video Settings')
                    custom_fps = gr.Number(label=LABEL_CUSTOM_FPS, value=None, precision=2, info=INFO_CUSTOM_FPS)

                with gr.Accordion('Advanced Settings', open=False):
                    auto_speed_clip_control = gr.Checkbox(
                        label='Auto speed clip control',
                        value=True,
                        info='Smart slowdowns and speed-ups only when Auto Mode finds a strong reason.'
                    )
                    avoid_text_clips = gr.Checkbox(
                        label='Avoid text',
                        value=True,
                        info='Skip source moments with obvious credits, titles, subtitles, captions, UI, or readable text when clean alternatives exist.'
                    )

                with gr.Group():
                    gr.Markdown(f'### 🎬 Processing Mode')
                    export_choices = []
                    for encoder, label in NVENC_ENCODERS.items():
                        if get_encoder_support(encoder)['available']:
                            export_choices.append((label, encoder))
                    vulkan_available = False
                    for encoder, label in VULKAN_ENCODERS.items():
                        if get_encoder_support(encoder)['available']:
                            export_choices.append((label, encoder))
                            vulkan_available = True
                    export_choices.extend([('CPU (H.264)', 'cpu'), ('ProRes 422 HQ', 'prores_hq')])
                    processing_mode = gr.Radio(
                        choices=export_choices,
                        value=export_choices[0][1],
                        label=LABEL_PROCESSING_MODE,
                        info=get_processing_mode_info(NVENC_AVAILABLE, vulkan_available),
                    )
                
                with gr.Group():
                    gr.Markdown('### 📁 Output Settings')
                    output_filename = gr.Textbox(value='music_video.mp4', label=LABEL_OUTPUT_FILENAME, info=INFO_OUTPUT_FILENAME)

                process_btn = gr.Button('🎬 Create Music Video', variant='primary', size='lg')

            with gr.Column(scale=1):
                gr.Markdown('### 📺 Output')
                status_output = gr.Textbox(label='Status', interactive=False, value=get_ready_status(python_status, gpu_backend_status, MAX_THREADS, CPU_COUNT, ffmpeg_status, GPU_AVAILABLE, gpu_info, NVENC_AVAILABLE), lines=4, max_lines=4, elem_id='status-output-box')
                video_output = gr.Video(label='Generated Music Video', interactive=False, elem_id='generated-video-output')
                export_download = gr.File(label='Download exported video', interactive=False)
                
        process_btn.click(
            fn=process_video,
            inputs=[
                audio_input, video_input,
                output_filename, processing_mode, custom_fps,
                auto_speed_clip_control,
                avoid_text_clips,
                session_state
            ],
            outputs=[video_output, status_output, export_download, session_state],
            show_progress='hidden'
        )

    return app

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # Clean up old files only on startup
    cleanup_on_startup()
    
    app = create_ui()
    launch_port = find_launch_port()
    app.launch(
        server_name="127.0.0.1",
        server_port=launch_port,
        share=False,
        inbrowser=True,
        show_error=True
    )
