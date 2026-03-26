#!/usr/bin/env python3
"""
FFmpeg Processing Module - Frame-Accurate Video Operations
- Frame-perfect segment extraction
- Zero timing drift
- Shared concat + mux helpers
"""

import json
import multiprocessing
import os
import random
import re
import shutil
import subprocess
import uuid
from typing import Dict, List, Tuple

# Determine script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set up portable FFmpeg path
FFMPEG_PATH = os.path.join(SCRIPT_DIR, "bin", "ffmpeg", "ffmpeg.exe")
FFPROBE_PATH = os.path.join(SCRIPT_DIR, "bin", "ffmpeg", "ffprobe.exe")
FFMPEG_DIR = os.path.dirname(FFMPEG_PATH)


def prepend_path_once(entry: str) -> None:
    """Prepend a directory to PATH once, preserving existing entries."""
    if not entry:
        return

    current = os.environ.get("PATH", "")
    path_entries = [item for item in current.split(os.pathsep) if item]
    normalized_entry = os.path.normcase(os.path.normpath(entry))
    normalized_entries = {
        os.path.normcase(os.path.normpath(item))
        for item in path_entries
    }

    if normalized_entry not in normalized_entries:
        os.environ["PATH"] = entry + os.pathsep + current if current else entry

# Check if portable FFmpeg exists
if os.path.exists(FFMPEG_PATH):
    prepend_path_once(FFMPEG_DIR)
    os.environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_PATH
    os.environ["FFMPEG_BINARY"] = FFMPEG_PATH
    if os.path.exists(FFPROBE_PATH):
        os.environ["FFPROBE_BINARY"] = FFPROBE_PATH
    print(f"Using portable FFmpeg: {FFMPEG_PATH}")
    FFMPEG_FOUND = True
else:
    print(f"Portable FFmpeg not found at: {FFMPEG_PATH}")
    print("   Falling back to system FFmpeg")
    FFMPEG_PATH = "ffmpeg"
    FFPROBE_PATH = "ffprobe"
    FFMPEG_FOUND = False

CPU_COUNT = multiprocessing.cpu_count()
MAX_THREADS = CPU_COUNT

STANDARD_QUALITY_CHOICES = ("fast", "balanced", "high")
DEFAULT_STANDARD_QUALITY = "balanced"

CPU_QUALITY_SETTINGS: Dict[str, Dict[str, int | str]] = {
    "fast": {"preset": "veryfast", "crf": 24},
    "balanced": {"preset": "fast", "crf": 20},
    "high": {"preset": "medium", "crf": 18},
}

NVENC_QUALITY_SETTINGS: Dict[str, Dict[str, int | str]] = {
    "fast": {"preset": "p3", "cq": 28},
    "balanced": {"preset": "p5", "cq": 22},
    "high": {"preset": "p6", "cq": 18},
}


def check_nvenc_support() -> bool:
    """Check if FFmpeg can actually use NVIDIA NVENC hardware encoding."""
    try:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        result = subprocess.run(
            [FFMPEG_PATH, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=creationflags,
        )
        if "h264_nvenc" not in result.stdout:
            return False

        probe_cmd = [
            FFMPEG_PATH,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:r=1:d=1",
            "-frames:v",
            "1",
            "-c:v",
            "h264_nvenc",
            "-f",
            "null",
            "-",
        ]
        probe_result = subprocess.run(
            probe_cmd,
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=creationflags,
        )
        return probe_result.returncode == 0
    except Exception:
        return False


def normalize_quality_profile(quality: str | None) -> str:
    """Return a valid standard export quality preset."""
    normalized = (quality or DEFAULT_STANDARD_QUALITY).strip().lower()
    if normalized not in STANDARD_QUALITY_CHOICES:
        return DEFAULT_STANDARD_QUALITY
    return normalized


def get_quality_summary(use_nvenc: bool, quality: str) -> str:
    """Get a human-readable encoder summary for the selected quality profile."""
    quality = normalize_quality_profile(quality)
    if use_nvenc:
        settings = NVENC_QUALITY_SETTINGS[quality]
        return f"{quality.capitalize()} (preset {settings['preset']}, CQ {settings['cq']})"

    settings = CPU_QUALITY_SETTINGS[quality]
    return f"{quality.capitalize()} (preset {settings['preset']}, CRF {settings['crf']})"


def get_video_duration(video_file: str) -> float:
    """Get the duration of a video file using ffprobe."""
    try:
        probe_cmd = [
            FFPROBE_PATH,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_file,
        ]

        result = subprocess.run(
            probe_cmd, capture_output=True, text=True, timeout=10
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"   Could not get duration with ffprobe: {e}")
        return 10.0


def get_video_fps(video_file: str) -> float:
    """Get the FPS of a video file using ffprobe."""
    try:
        probe_cmd = [
            FFPROBE_PATH,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_file,
        ]

        result = subprocess.run(
            probe_cmd, capture_output=True, text=True, timeout=10
        )
        fps_str = result.stdout.strip()

        if "/" in fps_str:
            num, den = fps_str.split("/")
            return float(num) / float(den)
        return float(fps_str)
    except Exception as e:
        print(f"   Could not get video FPS with ffprobe: {e}")
        return 30.0


def get_video_resolution(video_file: str) -> Tuple[int, int]:
    """Get the resolution (width, height) of a video file using ffprobe."""
    try:
        probe_cmd = [
            FFPROBE_PATH,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            video_file,
        ]

        result = subprocess.run(
            probe_cmd, capture_output=True, text=True, timeout=10
        )
        data = json.loads(result.stdout)
        width = data["streams"][0]["width"]
        height = data["streams"][0]["height"]
        return (width, height)
    except Exception as e:
        print(f"   Could not get video resolution with ffprobe: {e}")
        return (1920, 1080)


def get_video_codec(video_file: str) -> str | None:
    """Get the primary video codec name using ffprobe."""
    try:
        probe_cmd = [
            FFPROBE_PATH,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_file,
        ]

        result = subprocess.run(
            probe_cmd, capture_output=True, text=True, timeout=10
        )
        codec = result.stdout.strip().splitlines()
        return codec[0].strip().lower() if codec else None
    except Exception as e:
        print(f"   Could not get video codec with ffprobe: {e}")
        return None


def is_browser_playable_video(video_file: str) -> bool:
    """Return True when the container/codec pair is browser-friendly for Gradio playback."""
    container = os.path.splitext(video_file)[1].lower()
    video_codec = get_video_codec(video_file)
    return (container, video_codec) in {
        (".mp4", "h264"),
        (".mp4", "av1"),
        (".ogg", "theora"),
        (".webm", "vp9"),
        (".webm", "vp8"),
        (".webm", "av1"),
    }


def has_valid_video_stream(video_file: str) -> bool:
    """Return True when the file exists, is non-empty, and ffprobe sees a video stream."""
    if not os.path.exists(video_file):
        return False
    try:
        if os.path.getsize(video_file) <= 0:
            return False
    except OSError:
        return False
    return get_video_codec(video_file) is not None


def split_valid_video_files(video_files: List[str]) -> Tuple[List[str], List[str]]:
    """Split paths into valid and invalid video files using ffprobe stream validation."""
    valid_files: List[str] = []
    invalid_files: List[str] = []
    for video_file in video_files:
        if has_valid_video_stream(video_file):
            valid_files.append(video_file)
        else:
            invalid_files.append(video_file)
    return valid_files, invalid_files


def seconds_to_frame_count(seconds: float, fps: float) -> int:
    """Convert seconds to exact frame count."""
    return int(round(seconds * fps))


def frame_count_to_seconds(frames: int, fps: float) -> float:
    """Convert frame count back to exact seconds."""
    return frames / fps


def convert_to_prores_proxy(video_file: str, output_dir: str, fps: float | None = None) -> str:
    """Convert video to ProRes 422 Proxy for precise editing."""
    filename = os.path.basename(video_file)
    name, _ = os.path.splitext(filename)
    output_file = os.path.join(output_dir, f"{name}_prores.mov")

    print(f"   Converting to ProRes 422 Proxy: {filename}")

    if fps is None:
        fps = get_video_fps(video_file)

    cmd = [
        FFMPEG_PATH,
        "-hwaccel",
        "auto",
        "-i",
        video_file,
        "-c:v",
        "prores",
        "-profile:v",
        "0",
        "-vendor",
        "apl0",
        "-pix_fmt",
        "yuv422p10le",
        "-an",
        "-r",
        str(fps),
        "-threads",
        str(MAX_THREADS),
        "-y",
        output_file,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )

        if result.returncode != 0:
            print(f"   FFmpeg error: {result.stderr}")
            raise Exception(f"ProRes conversion failed for {filename}")

        print(f"   ProRes conversion complete: {name}_prores.mov")
        return output_file

    except subprocess.TimeoutExpired:
        raise Exception(f"ProRes conversion timeout for {filename}")
    except Exception as e:
        raise Exception(f"ProRes conversion error: {str(e)}")


def is_mp4_path(path: str) -> bool:
    """Return True if the path uses an MP4-family container."""
    return os.path.splitext(path)[1].lower() in {".mp4", ".m4v"}


def create_lossless_delivery_mp4(
    input_file: str,
    output_file: str,
    prefer_cuda_decode: bool = False,
) -> str:
    """Create an optional lossless MP4 delivery file from a ProRes master."""
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    hwaccel_attempts = [["-hwaccel", "cuda"]] if prefer_cuda_decode else [["-hwaccel", "auto"]]
    if prefer_cuda_decode:
        hwaccel_attempts.append(["-hwaccel", "auto"])

    last_error = "unknown FFmpeg error"

    for hwaccel_args in hwaccel_attempts:
        try:
            if os.path.exists(output_file):
                os.remove(output_file)
        except OSError:
            pass

        cmd = [
            FFMPEG_PATH,
            *hwaccel_args,
            "-i",
            input_file,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "0",
            "-pix_fmt",
            "yuv422p10le",
            "-profile:v",
            "high422",
            "-c:a",
            "alac",
            "-movflags",
            "+faststart",
            "-y",
            output_file,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,
            creationflags=creationflags,
        )
        if result.returncode == 0 and os.path.exists(output_file):
            return output_file

        stderr_tail = (result.stderr or "").strip().splitlines()
        last_error = stderr_tail[-1] if stderr_tail else "unknown FFmpeg error"

    raise Exception(f"Lossless MP4 creation failed: {last_error}")


def build_standard_video_encode_args(
    use_nvenc: bool,
    gpu_encoder: str,
    quality: str,
    threads_per_job: int,
    cpu_encoder: str = "libx264",
) -> List[str]:
    """Build codec arguments for standard exports."""
    quality = normalize_quality_profile(quality)

    if use_nvenc:
        settings = NVENC_QUALITY_SETTINGS[quality]
        return [
            "-c:v",
            gpu_encoder,
            "-preset",
            str(settings["preset"]),
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            str(settings["cq"]),
            "-pix_fmt",
            "yuv420p",
        ]

    settings = CPU_QUALITY_SETTINGS[quality]
    args = [
        "-c:v",
        cpu_encoder,
        "-preset",
        str(settings["preset"]),
        "-crf",
        str(settings["crf"]),
        "-pix_fmt",
        "yuv420p",
        "-threads",
        str(max(1, int(threads_per_job))),
    ]
    if cpu_encoder == "libx265":
        args.extend(["-tag:v", "hvc1"])
    return args


def build_audio_input_args(audio_file: str, start_time: float, end_time: float | None) -> List[str]:
    """Build FFmpeg input args for optionally trimmed audio."""
    audio_args: List[str] = []
    if end_time and end_time > start_time:
        audio_args.extend(["-ss", str(start_time), "-t", str(end_time - start_time)])
    elif start_time > 0:
        audio_args.extend(["-ss", str(start_time)])

    audio_args.extend(["-i", audio_file])
    return audio_args


def write_concat_file(video_files: List[str], concat_file: str) -> None:
    """Write an FFmpeg concat demuxer file list."""
    with open(concat_file, "w", encoding="utf-8") as handle:
        for video_file in video_files:
            escaped_path = video_file.replace("\\", "/").replace("'", r"'\''")
            handle.write(f"file '{escaped_path}'\n")


def get_target_size_filters(target_size: Tuple[int, int]) -> List[str]:
    """Build aspect-ratio-safe fit-and-pad filters for a target frame size."""
    width, height = target_size
    return [
        f"scale={width}:{height}:force_original_aspect_ratio=decrease",
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black",
        "setsar=1",
    ]


def get_segment_timeout_seconds(
    duration: float,
    reverse: bool = False,
    prores: bool = False,
) -> int:
    """Return a conservative timeout for per-segment extraction jobs."""
    safe_duration = max(0.1, float(duration))
    if prores:
        base_timeout = 240
        timeout = base_timeout + int(safe_duration * 45)
        if reverse:
            timeout += 120
        return min(900, max(240, timeout))

    base_timeout = 180
    timeout = base_timeout + int(safe_duration * 30)
    if reverse:
        timeout += 90
    return min(360, max(180, timeout))


def summarize_ffmpeg_error(stderr: str | None, fallback: str) -> str:
    """Return the most useful FFmpeg error summary available."""
    lines = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
    if not lines:
        return fallback

    generic_lines = {
        "Conversion failed!",
        "Error while processing the decoded data for stream #0:0",
    }
    informative_lines = [line for line in lines if line not in generic_lines]
    if not informative_lines:
        return lines[-1]

    tail = informative_lines[-3:]
    return " | ".join(tail)


def extract_clip_segment_ffmpeg(
    video_file: str,
    start_time: float,
    duration: float,
    output_file: str,
    fps: float,
    target_size: Tuple[int, int] | None,
    reverse: bool,
    speed_factor: float,
    use_nvenc: bool,
    gpu_encoder: str = "h264_nvenc",
    quality: str = DEFAULT_STANDARD_QUALITY,
    threads_per_job: int = 1,
) -> Tuple[bool, str | None, str | None]:
    """Extract and encode a frame-accurate standard segment."""
    try:
        frame_count = seconds_to_frame_count(duration, fps)
        timeout_seconds = get_segment_timeout_seconds(duration, reverse=reverse, prores=False)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        cpu_fallback_encoder = "libx265" if gpu_encoder == "hevc_nvenc" else "libx264"

        filters = []

        if speed_factor != 1.0:
            filters.append(f"setpts={1.0 / speed_factor}*PTS")

        if reverse:
            filters.append("reverse")

        if target_size:
            filters.extend(get_target_size_filters(target_size))

        filters.append(f"fps={fps}")
        filter_complex = ",".join(filters)
        decode_attempts: List[Tuple[str, List[str], bool, bool, str]] = [("software decode", [], False, use_nvenc, cpu_fallback_encoder)]
        if use_nvenc:
            decode_attempts.insert(0, ("CUDA decode", ["-hwaccel", "cuda"], False, True, cpu_fallback_encoder))
            decode_attempts.append(("software decode with accurate seek", [], True, True, cpu_fallback_encoder))
            decode_attempts.append((f"CPU encode fallback ({cpu_fallback_encoder})", [], True, False, cpu_fallback_encoder))
        else:
            decode_attempts.append(("software decode with accurate seek", [], True, False, cpu_fallback_encoder))

        last_error = "FFmpeg extraction failed"

        for attempt_index, (attempt_label, hwaccel_args, accurate_seek, attempt_use_nvenc, attempt_cpu_encoder) in enumerate(decode_attempts, start=1):
            try:
                if os.path.exists(output_file):
                    os.remove(output_file)
            except OSError:
                pass

            cmd = [FFMPEG_PATH]
            cmd.extend(hwaccel_args)
            if accurate_seek:
                cmd.extend(["-i", video_file, "-ss", str(start_time), "-t", str(duration)])
            else:
                cmd.extend(["-ss", str(start_time), "-i", video_file, "-t", str(duration)])
            cmd.extend(["-vf", filter_complex])
            cmd.extend(["-vframes", str(frame_count)])
            cmd.extend(
                build_standard_video_encode_args(
                    use_nvenc=attempt_use_nvenc,
                    gpu_encoder=gpu_encoder,
                    quality=quality,
                    threads_per_job=threads_per_job,
                    cpu_encoder=attempt_cpu_encoder,
                )
            )
            cmd.extend(
                [
                    "-an",
                    "-vsync",
                    "cfr",
                    "-r",
                    str(fps),
                    "-fflags",
                    "+genpts",
                ]
            )

            if is_mp4_path(output_file):
                cmd.extend(["-movflags", "+faststart"])

            cmd.extend(["-y", output_file])

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    creationflags=creationflags,
                )
            except subprocess.TimeoutExpired:
                last_error = f"{attempt_label} timed out after {timeout_seconds} seconds"
            except Exception as exc:
                last_error = f"{attempt_label} failed: {exc}"
            else:
                if result.returncode == 0 and has_valid_video_stream(output_file):
                    recovery_label = attempt_label if attempt_index > 1 else None
                    return True, recovery_label, None
                if result.returncode == 0:
                    last_error = f"{attempt_label} produced an output file without a valid video stream"
                else:
                    last_error = summarize_ffmpeg_error(
                        result.stderr,
                        f"{attempt_label} failed",
                    )

        failure_detail = (
            f"{last_error} | source={os.path.basename(video_file)} | "
            f"start={start_time:.3f}s | duration={duration:.3f}s"
        )
        return False, None, failure_detail

    except Exception as e:
        return False, None, f"Error extracting clip: {e}"


def extract_prores_segment_ffmpeg(
    video_file: str,
    start_time: float,
    duration: float,
    output_file: str,
    fps: float,
    target_size: Tuple[int, int] | None,
    reverse: bool,
    speed_factor: float,
) -> str:
    """Extract a frame-accurate ProRes segment from a ProRes proxy source."""
    frame_count = seconds_to_frame_count(duration, fps)
    timeout_seconds = get_segment_timeout_seconds(duration, reverse=reverse, prores=True)
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    filters = []
    if speed_factor != 1.0:
        filters.append(f"setpts={1.0 / speed_factor}*PTS")
    if reverse:
        filters.append("reverse")
    if target_size:
        filters.extend(get_target_size_filters(target_size))
    filters.append(f"fps={fps}")

    cmd = [
        FFMPEG_PATH,
        "-hwaccel",
        "auto",
        "-ss",
        str(start_time),
        "-i",
        video_file,
        "-t",
        str(duration),
        "-vf",
        ",".join(filters),
        "-vframes",
        str(frame_count),
        "-c:v",
        "prores",
        "-profile:v",
        "0",
        "-vendor",
        "apl0",
        "-pix_fmt",
        "yuv422p10le",
        "-r",
        str(fps),
        "-an",
        "-threads",
        str(MAX_THREADS),
        "-y",
        output_file,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            creationflags=creationflags,
        )

        if result.returncode != 0:
            raise Exception(
                f"Segment extraction failed: {summarize_ffmpeg_error(result.stderr, 'unknown FFmpeg error')}"
            )

        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            raise Exception(f"Segment file not created or is empty: {output_file}")

        return output_file
    except subprocess.TimeoutExpired:
        raise Exception(f"Segment extraction timed out after {timeout_seconds} seconds")
    except Exception as e:
        raise Exception(f"ProRes segment extraction error: {str(e)}")


def concatenate_videos_ffmpeg(
    video_files: List[str],
    output_file: str,
    audio_file: str | None = None,
    start_time: float = 0.0,
    end_time: float | None = None,
    use_nvenc: bool = False,
    gpu_encoder: str = "h264_nvenc",
    fps: float = 30.0,
    temp_dir: str | None = None,
    stream_copy: bool = False,
    audio_codec: str = "aac",
    quality: str = DEFAULT_STANDARD_QUALITY,
    threads_per_job: int = MAX_THREADS,
) -> str:
    """Concatenate videos and optionally mux audio."""
    if temp_dir is None:
        temp_dir = os.path.dirname(output_file)

    os.makedirs(temp_dir, exist_ok=True)
    concat_file = os.path.join(temp_dir, f"concat_list_{uuid.uuid4().hex}.txt")
    write_concat_file(video_files, concat_file)

    should_stream_copy = stream_copy or output_file.lower().endswith(".mov")
    temp_video = os.path.join(
        temp_dir,
        f"video_only_{uuid.uuid4().hex}{os.path.splitext(output_file)[1] or '.mkv'}",
    )

    try:
        if should_stream_copy:
            print(f"   Concatenating {len(video_files)} segments with stream copy...")
            cmd = [
                FFMPEG_PATH,
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_file,
                "-fflags",
                "+genpts",
                "-c",
                "copy",
            ]
            if is_mp4_path(temp_video):
                cmd.extend(["-movflags", "+faststart"])
            cmd.extend(["-y", temp_video])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                raise Exception(f"Concatenation failed: {result.stderr}")

            if audio_file:
                mux_audio_track(
                    video_file=temp_video,
                    output_file=output_file,
                    audio_file=audio_file,
                    start_time=start_time,
                    end_time=end_time,
                    audio_codec=audio_codec,
                )
                if os.path.exists(temp_video):
                    os.remove(temp_video)
            else:
                shutil.move(temp_video, output_file)
        else:
            print(f"   Concatenating and encoding {len(video_files)} segments...")
            cmd = [FFMPEG_PATH]
            if use_nvenc:
                cmd.extend(["-hwaccel", "cuda"])
            else:
                cmd.extend(["-hwaccel", "auto"])

            cmd.extend(["-f", "concat", "-safe", "0", "-i", concat_file])

            if audio_file:
                cmd.extend(build_audio_input_args(audio_file, start_time, end_time))
                cmd.extend(["-map", "0:v", "-map", "1:a"])

            cmd.extend(
                build_standard_video_encode_args(
                    use_nvenc=use_nvenc,
                    gpu_encoder=gpu_encoder,
                    quality=quality,
                    threads_per_job=threads_per_job,
                )
            )

            if audio_file:
                if audio_codec == "pcm_s24le":
                    cmd.extend(["-c:a", "pcm_s24le", "-ar", "48000", "-ac", "2"])
                else:
                    cmd.extend(["-c:a", audio_codec, "-b:a", "320k", "-ar", "48000", "-ac", "2"])
                cmd.append("-shortest")

            cmd.extend(["-vsync", "cfr", "-r", str(fps)])
            if is_mp4_path(output_file):
                cmd.extend(["-movflags", "+faststart"])
            cmd.extend(["-y", output_file])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                raise Exception(f"Encoding failed: {result.stderr}")

        return output_file
    finally:
        if os.path.exists(concat_file):
            os.remove(concat_file)


def mux_audio_track(
    video_file: str,
    output_file: str,
    audio_file: str,
    start_time: float = 0.0,
    end_time: float | None = None,
    audio_codec: str = "aac",
) -> str:
    """Mux a video-only file with an audio track without re-encoding video."""
    cmd = [FFMPEG_PATH, "-i", video_file]
    cmd.extend(build_audio_input_args(audio_file, start_time, end_time))
    cmd.extend(["-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy"])

    if audio_codec == "pcm_s24le":
        cmd.extend(["-c:a", "pcm_s24le", "-ar", "48000", "-ac", "2"])
    else:
        cmd.extend(["-c:a", audio_codec, "-b:a", "320k", "-ar", "48000", "-ac", "2"])

    cmd.append("-shortest")

    if is_mp4_path(output_file):
        cmd.extend(["-movflags", "+faststart"])

    cmd.extend(["-y", output_file])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise Exception(f"Audio mux failed: {result.stderr}")

    return output_file


def detect_video_scene_changes(video_path: str, threshold: float = 0.3) -> List[float]:
    """Detect scene changes using FFmpeg's scene detection filter."""
    try:
        print(f"   Analyzing scene changes with FFmpeg: {os.path.basename(video_path)}")
        print(f"   Threshold: {threshold}")

        cmd = [
            FFMPEG_PATH,
            "-i",
            video_path,
            "-vf",
            f"select='gt(scene,{threshold})',showinfo",
            "-f",
            "null",
            "-",
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )

        scene_changes = []
        for line in result.stderr.split("\n"):
            if "pts_time:" in line:
                match = re.search(r"pts_time:([\d.]+)", line)
                if match:
                    scene_changes.append(float(match.group(1)))

        scene_changes = sorted(list(set(scene_changes)))
        print(f"   Found {len(scene_changes)} scene changes")
        return scene_changes

    except subprocess.TimeoutExpired:
        print("   Warning: Scene detection timeout")
        return []
    except Exception as e:
        print(f"   Warning: Could not analyze scene changes: {e}")
        return []
