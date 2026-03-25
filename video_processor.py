#!/usr/bin/env python3
import argparse
import gc
import multiprocessing
import os
import random
import sys
import time
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, TypeAlias

import librosa
import numpy as np

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

from ffmpeg_processing import (
    DEFAULT_STANDARD_QUALITY,
    STANDARD_QUALITY_CHOICES,
    check_nvenc_support,
    concatenate_videos_ffmpeg,
    convert_to_prores_proxy,
    extract_clip_segment_ffmpeg,
    extract_prores_segment_ffmpeg,
    frame_count_to_seconds,
    get_quality_summary,
    has_valid_video_stream,
    split_valid_video_files,
    get_video_duration,
    get_video_fps,
    get_video_resolution,
    normalize_quality_profile,
    seconds_to_frame_count,
)
from manual_mode import analyze_beats_manual, process_manual_intensity
from smart_mode import analyze_beats_smart, get_gpu_info, is_gpu_available, list_presets, select_beats_smart, set_gpu_mode
from auto_mode import analyze_beats_auto

GPU_AVAILABLE = is_gpu_available()
CPU_ONLY_MODE = not GPU_AVAILABLE
cp = None

warnings.filterwarnings("ignore", message=".*bytes wanted but 0 bytes read.*")

BeatTimes: TypeAlias = np.ndarray
VideoList: TypeAlias = List[str]


@dataclass(frozen=True)
class SegmentJob:
    index: int
    video_file: str
    start_time: float
    source_duration: float
    frame_count: int
    final_duration: float
    reverse: bool


CPU_COUNT = multiprocessing.cpu_count()
MAX_THREADS = CPU_COUNT
PARALLEL_WORKERS = max(CPU_COUNT // 2, 4)
LOCAL_TEMP_DIR = os.path.join(SCRIPT_DIR, "temp")
os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)

print(f"CPU Optimization: Detected {CPU_COUNT} threads")
print(f"Parallel Processing: {PARALLEL_WORKERS} workers for simultaneous clip processing")
print(f"Local Temp Directory: {LOCAL_TEMP_DIR}")
print("Python: Portable (bin/python-3.13.9-embed-amd64/)" if USING_PORTABLE_PYTHON else f"Python: System ({sys.executable})")
print(f"CUDA: {RUNTIME.cuda_runtime_label}" if USING_PORTABLE_CUDA else "CUDA: System (or not available)")
print(f"GPU Acceleration: AVAILABLE - {get_gpu_info()}" if GPU_AVAILABLE else f"GPU Acceleration: NOT AVAILABLE - {get_gpu_info()}")


def get_local_temp_dir() -> str:
    return LOCAL_TEMP_DIR


def create_temp_subdir() -> str:
    temp_subdir = os.path.join(LOCAL_TEMP_DIR, f"session_{uuid.uuid4().hex}")
    os.makedirs(temp_subdir, exist_ok=True)
    return temp_subdir


def estimate_threads_per_job(parallel_workers: int) -> int:
    workers = max(1, int(parallel_workers or 1))
    return max(1, CPU_COUNT // workers)


def get_chunk_size(max_workers: int) -> int:
    workers = max(1, int(max_workers or 1))
    return min(120, max(24, workers * 6))


NVENC_AVAILABLE = GPU_AVAILABLE and check_nvenc_support()
print("NVIDIA NVENC: AVAILABLE - Hardware video encoding enabled" if NVENC_AVAILABLE else "NVIDIA NVENC: NOT AVAILABLE - Using CPU encoding only")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a music video with cuts synchronized to detected beats"
    )
    parser.add_argument("mp3_file", type=str, help="Path to the input audio file (MP3/WAV/FLAC)")
    parser.add_argument("video_directory", type=str, help="Directory containing MP4/MKV video files")
    parser.add_argument(
        "cut_intensity",
        type=str,
        help='Cut intensity: number (0.1-16) or smart preset (slower/slow/normal/fast/faster) or "auto"',
    )
    parser.add_argument("-o", "--output", type=str, default="output_music_video.mkv", help="Output video file path (default: output_music_video.mkv)")
    parser.add_argument("-d", "--duration", type=float, default=2.0, help="Default clip duration in seconds (retained for compatibility)")
    parser.add_argument("-s", "--start-time", type=float, default=0.0, help="Start time in seconds for audio processing (default: 0.0)")
    parser.add_argument("-e", "--end-time", type=float, default=None, help="End time in seconds for audio processing (default: full duration)")
    parser.add_argument("--direction", type=str, choices=["forward", "backward", "random"], default="random", help="Video playback direction: forward, backward, or random")
    parser.add_argument("--offset", type=float, default=0.0, help="Timing offset in seconds to adjust video sync (default: 0.0)")
    parser.add_argument("--mode", type=str, choices=["manual", "smart", "auto"], default=None, help="Generation mode: manual, smart, or auto")
    parser.add_argument("--lossless", action="store_true", help="Enable precise ProRes 422 Proxy mode")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration (audio analysis + NVENC encoding)")
    parser.add_argument("--gpu-encoder", type=str, choices=["h264_nvenc", "hevc_nvenc", "none"], default="h264_nvenc", help="GPU encoder: h264_nvenc (H.264), hevc_nvenc (H.265), none (CPU)")
    parser.add_argument("--fps", type=float, default=None, help="Output FPS. If not specified, auto-detect from the input video")
    parser.add_argument("--quality", type=str, choices=list(STANDARD_QUALITY_CHOICES), default=DEFAULT_STANDARD_QUALITY, help="Standard export quality preset: fast, balanced, or high")
    return parser.parse_args()


def get_video_files(directory: str) -> VideoList:
    video_extensions = {".mp4", ".mkv"}
    video_files = sorted(
        str(path)
        for path in Path(directory).rglob("*")
        if path.is_file() and path.suffix.lower() in video_extensions
    )
    if not video_files:
        raise ValueError(f"No MP4/MKV files found in {directory} or its subfolders")
    return video_files


def determine_mode(args: argparse.Namespace) -> str:
    if args.mode:
        return args.mode
    if args.cut_intensity == "auto":
        return "auto"
    if args.cut_intensity in list_presets():
        return "smart"
    return "manual"


def resolve_selected_beats(mode: str, audio_file: str, cut_intensity: str, start_time: float, end_time: float | None, use_gpu: bool) -> tuple[np.ndarray, dict]:
    if mode == "auto":
        print("Using AUTO mode (extreme intelligence)")
        return analyze_beats_auto(
            audio_file,
            start_time=start_time,
            end_time=end_time,
            use_gpu=use_gpu,
        )

    if mode == "smart":
        print("Using SMART mode (multi-band analysis)")
        beat_times, beat_info = analyze_beats_smart(
            audio_file,
            start_time=start_time,
            end_time=end_time,
        )
        return select_beats_smart(beat_info, preset=cut_intensity), beat_info

    print("Using MANUAL mode (bass-focused)")
    beat_times, beat_info = analyze_beats_manual(
        audio_file,
        start_time=start_time,
        end_time=end_time,
        use_gpu=use_gpu,
    )
    try:
        intensity_value = float(cut_intensity)
    except ValueError:
        print("Invalid cut intensity, using 4.0")
        intensity_value = 4.0
    return process_manual_intensity(beat_times, intensity_value), beat_info


def choose_reverse(direction: str) -> bool:
    if direction == "backward":
        return True
    if direction == "random":
        return random.choice([True, False])
    return False


def plan_segments(video_files: VideoList, beat_times: BeatTimes, fps: float, speed_factor: float, direction: str, timing_offset: float) -> List[SegmentJob]:
    video_durations = {video_file: get_video_duration(video_file) for video_file in video_files}
    planned_segments: List[SegmentJob] = []
    origin_time = float(beat_times[0]) if len(beat_times) else 0.0
    boundary_frames = [seconds_to_frame_count(max(0.0, float(beat_time) - origin_time), fps) for beat_time in beat_times]

    for index in range(len(beat_times) - 1):
        frame_count = max(1, boundary_frames[index + 1] - boundary_frames[index])
        final_duration = frame_count_to_seconds(frame_count, fps)

        video_file = random.choice(video_files)
        video_duration = video_durations[video_file]
        source_duration = min(video_duration, final_duration * speed_factor)

        if video_duration > source_duration:
            clip_start = random.uniform(0, video_duration - source_duration)
        else:
            clip_start = 0.0

        adjusted_start = clip_start - timing_offset if timing_offset != 0.0 else clip_start
        max_start = max(0.0, video_duration - source_duration)
        adjusted_start = min(max(0.0, adjusted_start), max_start)

        planned_segments.append(
            SegmentJob(
                index=index,
                video_file=video_file,
                start_time=adjusted_start,
                source_duration=source_duration,
                frame_count=frame_count,
                final_duration=final_duration,
                reverse=choose_reverse(direction),
            )
        )

    return planned_segments


def render_standard_segment(
    job: SegmentJob,
    target_size: tuple[int, int],
    use_nvenc: bool,
    gpu_encoder: str,
    fps: float,
    quality: str,
    threads_per_job: int,
    temp_dir: str,
    temp_ext: str,
) -> tuple[int, str | None, str | None, str | None]:
    output_file = os.path.join(temp_dir, f"segment_{job.index:05d}{temp_ext}")
    success, recovery_label, failure_detail = extract_clip_segment_ffmpeg(
        video_file=job.video_file,
        start_time=job.start_time,
        duration=job.source_duration,
        output_file=output_file,
        fps=fps,
        target_size=target_size,
        reverse=job.reverse,
        speed_factor=job.source_duration / job.final_duration if job.final_duration > 0 else 1.0,
        use_nvenc=use_nvenc,
        gpu_encoder=gpu_encoder,
        quality=quality,
        threads_per_job=threads_per_job,
    )
    if not success:
        return job.index, None, failure_detail or "FFmpeg extraction failed", None
    return job.index, output_file, None, recovery_label


def render_prores_segment(job: SegmentJob, target_size: tuple[int, int], fps: float, prores_map: Dict[str, str], segments_dir: str) -> str:
    output_file = os.path.join(segments_dir, f"segment_{job.index:05d}.mov")
    speed_factor = job.source_duration / job.final_duration if job.final_duration > 0 else 1.0
    return extract_prores_segment_ffmpeg(
        video_file=prores_map[job.video_file],
        start_time=job.start_time,
        duration=job.source_duration,
        output_file=output_file,
        fps=fps,
        target_size=target_size,
        reverse=job.reverse,
        speed_factor=speed_factor,
    )


def render_standard_chunk(chunk_index: int, jobs: List[SegmentJob], target_size: tuple[int, int], use_nvenc: bool, gpu_encoder: str, fps: float, quality: str, threads_per_job: int, max_workers: int, session_temp_dir: str, chunk_output_dir: str, temp_ext: str) -> tuple[str, int, int]:
    chunk_temp_dir = os.path.join(session_temp_dir, f"chunk_{chunk_index:04d}")
    os.makedirs(chunk_temp_dir, exist_ok=True)
    rendered_segments: Dict[int, str] = {}
    failed_segments = 0
    recovered_segments = 0
    recovery_mode_counts: Dict[str, int] = {}

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {
                executor.submit(
                    render_standard_segment,
                    job,
                    target_size,
                    use_nvenc,
                    gpu_encoder,
                    fps,
                    quality,
                    threads_per_job,
                    chunk_temp_dir,
                    temp_ext,
                ): job
                for job in jobs
            }

            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    index, clip_path, error, recovery_label = future.result()
                except Exception as exc:
                    print(f"Warning: Error processing clip {job.index + 1}: {exc}")
                    failed_segments += 1
                    continue

                if error:
                    print(f"Warning: Clip {job.index + 1} failed: {error}")
                    failed_segments += 1
                    continue

                if clip_path:
                    rendered_segments[index] = clip_path
                if recovery_label:
                    recovered_segments += 1
                    recovery_mode_counts[recovery_label] = recovery_mode_counts.get(recovery_label, 0) + 1

        ordered_clips = [rendered_segments[job.index] for job in jobs if job.index in rendered_segments]
        ordered_clips, invalid_rendered_clips = split_valid_video_files(ordered_clips)
        if invalid_rendered_clips:
            failed_segments += len(invalid_rendered_clips)
            print(
                f"   Warning: chunk {chunk_index} dropped {len(invalid_rendered_clips)} rendered clip(s) with no readable video stream before assembly."
            )
        if not ordered_clips:
            raise ValueError(f"No valid clips were rendered for chunk {chunk_index}")

        if recovered_segments:
            recovery_summary = ", ".join(
                f"{label}={count}" for label, count in sorted(recovery_mode_counts.items())
            )
            print(
                f"   Note: chunk {chunk_index} recovered {recovered_segments} clip(s) after retry "
                f"({recovery_summary})."
            )

        if failed_segments:
            print(
                f"   Warning: chunk {chunk_index} skipped {failed_segments} clip(s) after retries; output will still be assembled from the successful segments."
            )

        chunk_output = os.path.join(chunk_output_dir, f"chunk_{chunk_index:04d}{temp_ext}")
        try:
            concatenate_videos_ffmpeg(
                video_files=ordered_clips,
                output_file=chunk_output,
                temp_dir=chunk_temp_dir,
                stream_copy=True,
            )
        except Exception as exc:
            print(f"   Warning: stream-copy chunk assembly failed ({exc}). Retrying chunk {chunk_index} with re-encode...")
            concatenate_videos_ffmpeg(
                video_files=ordered_clips,
                output_file=chunk_output,
                use_nvenc=use_nvenc,
                gpu_encoder=gpu_encoder,
                fps=fps,
                temp_dir=chunk_temp_dir,
                stream_copy=False,
                quality=quality,
                threads_per_job=threads_per_job,
            )

        if not has_valid_video_stream(chunk_output):
            raise ValueError(f"Chunk {chunk_index} output is missing a readable video stream")
        return chunk_output, len(ordered_clips), failed_segments
    finally:
        for file_name in os.listdir(chunk_temp_dir):
            file_path = os.path.join(chunk_temp_dir, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except OSError:
                pass
        try:
            os.rmdir(chunk_temp_dir)
        except OSError:
            pass


def render_and_assemble_standard(audio_file: str, segment_plan: List[SegmentJob], target_size: tuple[int, int], output_file: str, start_time: float, end_time: float | None, use_nvenc: bool, gpu_encoder: str, fps: float, max_workers: int, quality: str, session_temp_dir: str) -> str:
    if not segment_plan:
        raise ValueError("No segment plan was generated.")

    chunk_size = get_chunk_size(max_workers)
    threads_per_job = estimate_threads_per_job(max_workers)
    temp_ext = ".mp4" if os.path.splitext(output_file)[1].lower() == ".mp4" else ".mkv"
    chunk_output_dir = os.path.join(session_temp_dir, "chunks")
    os.makedirs(chunk_output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("STANDARD MODE: CHUNKED RENDER + STREAM-COPY ASSEMBLY")
    print("=" * 60)
    print(f"   Total clips: {len(segment_plan)}")
    print(f"   Parallel workers: {max_workers}")
    print(f"   Threads per CPU encode: {threads_per_job}")
    print(f"   Chunk size: {chunk_size}")
    print(f"   Quality: {get_quality_summary(use_nvenc, quality)}")
    print(f"   Encoder: {gpu_encoder.upper() if use_nvenc else 'LIBX264'}")
    print("=" * 60 + "\n")

    chunk_outputs: List[str] = []
    total_rendered = 0
    total_failed = 0

    for chunk_index, start_index in enumerate(range(0, len(segment_plan), chunk_size), start=1):
        chunk_jobs = segment_plan[start_index : start_index + chunk_size]
        print(f"Rendering chunk {chunk_index} with {len(chunk_jobs)} clips...")
        chunk_output, rendered_count, failed_count = render_standard_chunk(
            chunk_index=chunk_index,
            jobs=chunk_jobs,
            target_size=target_size,
            use_nvenc=use_nvenc,
            gpu_encoder=gpu_encoder,
            fps=fps,
            quality=quality,
            threads_per_job=threads_per_job,
            max_workers=max_workers,
            session_temp_dir=session_temp_dir,
            chunk_output_dir=chunk_output_dir,
            temp_ext=temp_ext,
        )
        chunk_outputs.append(chunk_output)
        total_rendered += rendered_count
        total_failed += failed_count
        progress = (min(start_index + len(chunk_jobs), len(segment_plan)) / len(segment_plan)) * 100
        print(f"   Progress: {total_rendered} rendered clips, {progress:.1f}% of plan processed")

    if not chunk_outputs:
        raise ValueError("No valid chunk outputs were created.")

    chunk_outputs, invalid_chunk_outputs = split_valid_video_files(chunk_outputs)
    if invalid_chunk_outputs:
        print(
            f"Warning: Dropping {len(invalid_chunk_outputs)} invalid chunk output(s) before final assembly."
        )
    if not chunk_outputs:
        raise ValueError("No valid chunk outputs remained for final assembly.")

    print("\n" + "=" * 60)
    print(f"FINAL ASSEMBLY: Concatenating {len(chunk_outputs)} chunk(s)")
    print("=" * 60 + "\n")

    concatenate_videos_ffmpeg(
        video_files=chunk_outputs,
        output_file=output_file,
        audio_file=audio_file,
        start_time=start_time,
        end_time=end_time,
        temp_dir=session_temp_dir,
        stream_copy=True,
        audio_codec="aac",
    )

    for chunk_output in chunk_outputs:
        try:
            if os.path.exists(chunk_output):
                os.remove(chunk_output)
        except OSError:
            pass

    if total_failed:
        print("\n" + "!" * 60)
        print(
            f"WARNING: {total_failed} clip(s) were skipped during standard rendering after retries."
        )
        print(
            "   The output file was still created, but timing or visual coverage may differ slightly from the original plan."
        )
        print("   Check the earlier FFmpeg warnings and consider rerunning if exact coverage matters.")
        print("!" * 60 + "\n")

    return output_file


def render_and_assemble_prores(audio_file: str, video_files: VideoList, segment_plan: List[SegmentJob], target_size: tuple[int, int], output_file: str, start_time: float, end_time: float | None, fps: float, session_temp_dir: str) -> str:
    if not segment_plan:
        raise ValueError("No segment plan was generated.")

    print("\n" + "=" * 60)
    print("LOSSLESS MODE: PRORES PRECISE WORKFLOW")
    print("=" * 60)

    prores_dir = os.path.join(session_temp_dir, "prores")
    segments_dir = os.path.join(session_temp_dir, "segments")
    os.makedirs(prores_dir, exist_ok=True)
    os.makedirs(segments_dir, exist_ok=True)

    used_sources = sorted({job.video_file for job in segment_plan})
    print(f"   Source videos used in plan: {len(used_sources)} / {len(video_files)}")
    print(f"   FPS: {fps}")
    print(f"   Target resolution: {target_size[0]}x{target_size[1]}")

    prores_map: Dict[str, str] = {}
    for index, video_file in enumerate(used_sources, start=1):
        print(f"Converting source {index}/{len(used_sources)} to ProRes proxy...")
        prores_map[video_file] = convert_to_prores_proxy(video_file, prores_dir, fps)

    segment_files: List[str] = []
    for index, job in enumerate(segment_plan, start=1):
        segment_files.append(
            render_prores_segment(
                job=job,
                target_size=target_size,
                fps=fps,
                prores_map=prores_map,
                segments_dir=segments_dir,
            )
        )
        if index % 10 == 0 or index == len(segment_plan):
            print(f"   Rendered {index}/{len(segment_plan)} ProRes segments")

    concatenate_videos_ffmpeg(
        video_files=segment_files,
        output_file=output_file,
        audio_file=audio_file,
        start_time=start_time,
        end_time=end_time,
        temp_dir=session_temp_dir,
        stream_copy=True,
        audio_codec="pcm_s24le",
    )

    return output_file


def cleanup_session_temp_dir(session_temp_dir: str) -> None:
    if not os.path.exists(session_temp_dir):
        return

    for root, dirs, files in os.walk(session_temp_dir, topdown=False):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                os.remove(file_path)
            except OSError:
                pass
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                os.rmdir(dir_path)
            except OSError:
                pass

    try:
        os.rmdir(session_temp_dir)
        print("Cleaned up session temp directory")
    except OSError as exc:
        print(f"Warning: Could not delete session temp directory: {exc}")


def create_music_video(audio_file: str, video_files: VideoList, beat_times: BeatTimes, cut_intensity, default_duration: float = 2.0, output_file: str = "output_music_video.mkv", start_time: float = 0.0, end_time: float | None = None, direction: str = "random", speed_factor: float = 1.0, timing_offset: float = 0.0, max_workers: int | None = None, smart_mode: bool = False, beat_info: dict | None = None, lossless_mode: bool = False, use_gpu: bool = False, gpu_encoder: str = "h264_nvenc", fps: float | None = None, target_size: tuple[int, int] | None = None, quality: str = DEFAULT_STANDARD_QUALITY, mode_name: str | None = None) -> str:
    if len(beat_times) == 0:
        raise ValueError("No beats were detected. Cannot create video.")

    if max_workers is None:
        max_workers = PARALLEL_WORKERS
    max_workers = max(1, int(max_workers))
    quality = normalize_quality_profile(quality)

    if fps is None:
        try:
            fps = get_video_fps(video_files[0])
            print(f"Auto-detected FPS from input video: {fps}")
        except Exception:
            fps = 30.0
            print(f"Could not detect FPS, using default: {fps}")
    else:
        print(f"Using custom FPS: {fps}")

    session_temp_dir = create_temp_subdir()
    print(f"Session temp directory: {session_temp_dir}")

    use_nvenc = use_gpu and NVENC_AVAILABLE and not lossless_mode and gpu_encoder != "none"
    threads_per_job = estimate_threads_per_job(max_workers)
    analysis_device = (beat_info or {}).get("analysis_device")
    gpu_status = "GPU" if analysis_device == "gpu" else "CPU" if analysis_device == "cpu" else ("GPU" if use_gpu and GPU_AVAILABLE else "CPU")
    encoder_status = gpu_encoder.upper() if use_nvenc else ("ProRes 422 Proxy" if lossless_mode else "libx264")
    python_status = "Portable" if USING_PORTABLE_PYTHON else "System"
    cuda_status = "Portable" if USING_PORTABLE_CUDA else "System/None"
    mode_name = mode_name or (beat_info.get("mode", "unknown") if beat_info else "unknown")

    print("Performance Settings:")
    print(f"   Python: {python_status}")
    print(f"   CUDA: {cuda_status}")
    print("   FFmpeg: Portable (bin/ffmpeg/ffmpeg.exe)")
    print(f"   Parallel clip workers: {max_workers}")
    print(f"   Threads per CPU encode: {threads_per_job}")
    print(f"   Audio analysis: {gpu_status}")
    print(f"   Video processing: {encoder_status}")
    print(f"   Output FPS: {fps}")
    print("   Frame-accurate mode: ENABLED")
    print(f"   Generation mode: {mode_name}")
    if not lossless_mode:
        print(f"   Standard quality: {get_quality_summary(use_nvenc, quality)}")
    if timing_offset != 0.0:
        offset_direction = "earlier" if timing_offset < 0 else "later"
        print(f"   Timing offset: {abs(timing_offset):.3f}s ({offset_direction})")
    print(f"   Export format: {os.path.splitext(output_file)[1].lower() or '.mkv'}")

    audio_duration = get_video_duration(audio_file)
    if end_time and end_time > start_time:
        audio_duration = end_time - start_time
    elif start_time > 0:
        audio_duration = audio_duration - start_time
    print(f"Audio duration: {audio_duration:.2f} seconds")

    selected_beats = beat_times
    if len(selected_beats) == 0 or selected_beats[0] > 0.1:
        selected_beats = np.insert(selected_beats, 0, 0)
    if selected_beats[-1] < audio_duration:
        selected_beats = np.append(selected_beats, audio_duration)

    if target_size is None:
        target_size = get_video_resolution(video_files[0])
        print(f"Target resolution: {target_size[0]}x{target_size[1]} (auto-detected)")
    else:
        print(f"Target resolution: {target_size[0]}x{target_size[1]} (custom)")
    print(f"Creating video with {len(selected_beats) - 1} cuts")
    if timing_offset != 0.0:
        print(f"Timing offset: {timing_offset:.3f}s (applied to video playback, not beats)")

    segment_plan = plan_segments(
        video_files=video_files,
        beat_times=selected_beats,
        fps=fps,
        speed_factor=speed_factor,
        direction=direction,
        timing_offset=timing_offset,
    )

    try:
        if lossless_mode:
            render_and_assemble_prores(
                audio_file=audio_file,
                video_files=video_files,
                segment_plan=segment_plan,
                target_size=target_size,
                output_file=output_file,
                start_time=start_time,
                end_time=end_time,
                fps=fps,
                session_temp_dir=session_temp_dir,
            )
        else:
            render_and_assemble_standard(
                audio_file=audio_file,
                segment_plan=segment_plan,
                target_size=target_size,
                output_file=output_file,
                start_time=start_time,
                end_time=end_time,
                use_nvenc=use_nvenc,
                gpu_encoder=gpu_encoder,
                fps=fps,
                max_workers=max_workers,
                quality=quality,
                session_temp_dir=session_temp_dir,
            )

        print("\n" + "=" * 60)
        print("VIDEO RENDER COMPLETE")
        print(f"   Output: {output_file}")
        print(f"   FPS: {fps} (frame-accurate)")
        print(f"   Total Cuts: {len(segment_plan)}")
        print("=" * 60 + "\n")
        return output_file
    finally:
        time.sleep(0.5)
        gc.collect()
        cleanup_session_temp_dir(session_temp_dir)


def main() -> None:
    args = parse_arguments()

    if not os.path.exists(args.mp3_file):
        raise FileNotFoundError("Audio file not found: " + args.mp3_file)
    if not os.path.isdir(args.video_directory):
        raise NotADirectoryError("Video directory not found: " + args.video_directory)

    if args.gpu:
        if GPU_AVAILABLE:
            set_gpu_mode(True)
            print(f"GPU acceleration ENABLED: {get_gpu_info()}")
        else:
            print("GPU requested but CuPy not available, using CPU")
            args.gpu = False

    mode = determine_mode(args)
    quality = normalize_quality_profile(args.quality)
    threads_per_job = estimate_threads_per_job(PARALLEL_WORKERS)

    python_str = "Portable (bin/python-3.13.9-embed-amd64/)" if USING_PORTABLE_PYTHON else f"System ({sys.executable})"
    cuda_str = RUNTIME.cuda_runtime_label if USING_PORTABLE_CUDA else "System/None"

    print("\n" + "=" * 60)
    print("MUSIC VIDEO CUTTER - FRAME-ACCURATE")
    print(f"   Python: {python_str}")
    print(f"   CUDA: {cuda_str}")
    print(f"   CPU Threads: {CPU_COUNT}")
    print("   FFmpeg: Portable (bin/ffmpeg/ffmpeg.exe)")
    print(f"   Audio Analysis: {'GPU' if args.gpu else 'CPU'}")
    if args.gpu and NVENC_AVAILABLE and not args.lossless and args.gpu_encoder != 'none':
        print(f"   Video Encoding: {args.gpu_encoder.upper()}")
    else:
        print(f"   Video Encoding: {'ProRes' if args.lossless else 'CPU'}")
    print(f"   Threads per CPU encode: {threads_per_job}")
    print(f"   FPS: {args.fps if args.fps else 'Auto-detect from input video'}")
    if args.offset != 0.0:
        print(f"   Timing Offset: {args.offset:.3f}s (applied to video playback)")
    print(f"   Mode: {mode.upper()}")
    if args.lossless:
        print("   Export: Lossless/Precise (ProRes 422 Proxy)")
    else:
        use_nvenc = args.gpu and NVENC_AVAILABLE and args.gpu_encoder != "none"
        print(f"   Export: Standard delivery | Quality: {get_quality_summary(use_nvenc, quality)}")
    print("=" * 60 + "\n")

    print(f"Audio file: {args.mp3_file}")
    selected_beats, beat_info = resolve_selected_beats(
        mode=mode,
        audio_file=args.mp3_file,
        cut_intensity=args.cut_intensity,
        start_time=args.start_time,
        end_time=args.end_time,
        use_gpu=args.gpu,
    )
    print(f"Selected {len(selected_beats)} cuts for video")

    video_files = get_video_files(args.video_directory)
    print(f"Found {len(video_files)} video files")

    output_file = args.output
    if args.lossless and not output_file.lower().endswith(".mov"):
        base, _ = os.path.splitext(output_file)
        output_file = base + ".mov"
        print(f"Changed output to .mov for Lossless mode: {output_file}")
    elif not args.lossless and not output_file.lower().endswith((".mkv", ".mp4")):
        base, _ = os.path.splitext(output_file)
        output_file = base + ".mkv"
        print(f"Changed output to .mkv: {output_file}")

    print("\nStarting video creation...\n")
    output_file = create_music_video(
        args.mp3_file,
        video_files,
        selected_beats,
        args.cut_intensity,
        default_duration=args.duration,
        output_file=output_file,
        start_time=args.start_time,
        end_time=args.end_time,
        direction=args.direction,
        timing_offset=args.offset,
        max_workers=PARALLEL_WORKERS,
        beat_info=beat_info,
        lossless_mode=args.lossless,
        use_gpu=args.gpu,
        gpu_encoder=args.gpu_encoder,
        fps=args.fps,
        quality=quality,
        mode_name=mode,
    )
    print(f"Music video created successfully: {output_file}")


if __name__ == "__main__":
    main()
