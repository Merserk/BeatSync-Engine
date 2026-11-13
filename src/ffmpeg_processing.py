#!/usr/bin/env python3
"""
FFmpeg Processing Module - Frame-Accurate Video Operations
- Frame-perfect segment extraction
- Zero timing drift
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import subprocess
import time
import json
import random
import shutil
import uuid
import re
from typing import Tuple, List
from pathlib import Path
from fractions import Fraction
import hashlib
import math


from logger import (
    setup_environment,
    FFMPEG_EXE as FFMPEG_PATH,
    check_nvenc as check_nvenc_support,
)
from gpu_cpu_utils import MAX_THREADS, PARALLEL_WORKERS
from video_encoders import (VULKAN_ENCODERS, vulkan_device_args, vulkan_quality_args,
                            nvenc_quality_args, get_encoder_support,
                            nvenc_device_args, cuda_decode_args, get_prores_gpu_support)
from gpu_devices import get_cuda_gpu
from gpu_video import get_gpu_video_path
from ffmpeg_runtime import (FFPROBE_EXE, MediaError, get_runtime_info, run_media_command,
                            media_duration, video_fps, video_resolution, video_stream,
                            verify_video, fps_string, normalize_output_size)

# Initialize environment
setup_environment()

# Set up FFPROBE_PATH based on FFMPEG_PATH
FFPROBE_PATH = FFPROBE_EXE


def _run_media_command(cmd: List[str], timeout: int) -> subprocess.CompletedProcess[str]:
    """Run an FFmpeg/FFprobe command with consistent capture settings."""
    return run_media_command(cmd, timeout)



def _safe_remove_file(path: str | None) -> None:
    """Best-effort removal for temporary media files."""
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError as e:
            print(f"   ⚠️  Could not remove temporary file {os.path.basename(path)}: {e}")


def _short_ffmpeg_error(stderr: str, max_chars: int = 2200) -> str:
    """Return the useful tail of FFmpeg stderr without flooding the UI/log."""
    if not stderr:
        return ""
    text = stderr.strip()
    if len(text) <= max_chars:
        return text
    return "..." + text[-max_chars:]


def _fmt_seconds(seconds: float) -> str:
    try:
        value = float(seconds)
    except Exception:
        value = 0.0
    if value < 1.0:
        return f"{value * 1000:.0f}ms"
    return f"{value:.1f}s"


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _extract_scene_times(stderr: str) -> List[float]:
    """Parse showinfo pts_time values from FFmpeg stderr."""
    scene_changes: List[float] = []
    for line in stderr.split('\n'):
        if 'pts_time:' in line:
            match = re.search(r'pts_time:([\d.]+)', line)
            if match:
                try:
                    scene_changes.append(float(match.group(1)))
                except ValueError:
                    pass
    scene_changes = sorted(set(round(t, 3) for t in scene_changes if t >= 0.0))
    cleaned: List[float] = []
    for t in scene_changes:
        if not cleaned or t - cleaned[-1] >= 0.08:
            cleaned.append(t)
    return cleaned


def get_nvenc_quality_args(gpu_encoder: str, include_pix_fmt: bool = True) -> List[str]:
    """Use current NVENC options verified with the installed GPU driver."""
    return nvenc_quality_args(gpu_encoder, include_pix_fmt)


def get_cpu_h264_quality_args(include_pix_fmt: bool = True) -> List[str]:
    """Return lossless CPU H.264 settings."""
    args = [
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '0',
    ]
    if include_pix_fmt:
        args.extend(['-pix_fmt', 'yuv420p'])
    args.extend(['-threads', str(_render_threads())])
    return args


def get_hardware_encoding_args(gpu_encoder: str) -> List[str]:
    if gpu_encoder in VULKAN_ENCODERS:
        # Hardware Vulkan frames are produced by format=nv12,hwupload.
        return vulkan_quality_args(gpu_encoder)
    return get_nvenc_quality_args(gpu_encoder, include_pix_fmt=True)


def get_video_duration(video_file: str) -> float:
    return media_duration(video_file)


def get_video_fps(video_file: str) -> float:
    return video_fps(video_file)


def get_video_resolution(video_file: str) -> Tuple[int, int]:
    return video_resolution(video_file)


def seconds_to_frame_count(seconds: float, fps: float) -> int:
    """
    Convert seconds to exact frame count.
    This ensures frame-accurate timing with no drift.
    """
    return int(round(seconds * fps))


def frame_count_to_seconds(frames: int, fps: float) -> float:
    """
    Convert frame count back to exact seconds.
    This is the EXACT duration for the frame count.
    """
    return frames / fps


def _render_threads():
    default = max(1, min(8, MAX_THREADS // max(1, PARALLEL_WORKERS)))
    try:
        return max(1, min(MAX_THREADS, int(os.environ.get('BEATSYNC_FFMPEG_THREADS', default))))
    except ValueError:
        return default


def _base_command():
    get_runtime_info()
    return [FFMPEG_PATH, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
            '-threads', str(_render_threads()), '-filter_threads', str(_render_threads())]


def _container_args(output_file, fps, encoder=''):
    if Path(output_file).suffix.lower() not in ('.mp4', '.mov'):
        return []
    args = ['-movflags', '+faststart', '-video_track_timescale',
            str(Fraction(fps_string(fps)).numerator)]
    if encoder.startswith('hevc_'):
        args += ['-tag:v', 'hvc1']
    return args


def _scale_filters(target_size):
    if not target_size:
        return []
    width, height = normalize_output_size(target_size)
    return [f'scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2:reset_sar=1',
            f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2', 'setsar=1']


def _timing_filters(duration, output_duration, fps, target_size=None):
    rate = fps_string(fps)
    if not math.isfinite(float(duration)) or float(duration) <= 0:
        raise ValueError(f'Invalid source clip duration: {duration}')
    if not math.isfinite(float(output_duration)) or float(output_duration) <= 0:
        raise ValueError(f'Invalid output clip duration: {output_duration}')
    frames = max(1, seconds_to_frame_count(output_duration, fps))
    # Source time is independent of the output FPS. Rounding it up can consume
    # a frame from the next source shot, especially when retiming a clip.
    source_duration = float(duration)
    exact_output = frame_count_to_seconds(frames, fps)
    filters = [f'trim=duration={source_duration}',
               f'setpts={exact_output / source_duration:.10f}*(PTS-STARTPTS)']
    filters += _scale_filters(target_size)
    filters += [f'fps={rate}:start_time=0',
                f'tpad=stop_mode=clone:stop_duration={exact_output}']
    return filters, frames


def _verify_output(result, output_file, frames=None):
    if result.returncode:
        raise MediaError(_short_ffmpeg_error(result.stderr))
    if not os.path.isfile(output_file) or os.path.getsize(output_file) == 0:
        raise MediaError(f'FFmpeg did not produce a video: {output_file}')
    verify_video(output_file, frames)


def _prores_args():
    return ['-c:v', 'prores_ks', '-profile:v', '3', '-vendor', 'apl0', '-tag:v', 'apch',
            '-pix_fmt', 'yuv422p10le', '-threads', str(MAX_THREADS)]


def _prores_commands(video_file, output_file, fps, target_size, filters, frames=None, start_time=None, use_gpu=True,
                     source_duration=None):
    commands = []
    if use_gpu and get_prores_gpu_support()['available']:
        route = get_gpu_video_path(video_file, 'vulkan')
        if route:
            width, height = normalize_output_size(target_size or get_video_resolution(video_file))
            # Preserve ProRes's 10-bit 4:2:2 precision when software decoding
            # an unsupported input; an NV12 upload would discard both.
            upload = [] if route.hardware_decode else ['format=yuv422p10le', 'hwupload']
            gpu_filters = upload + route.scale_filters(width, height, contain=True, pixel_format='yuv422p10le')
            gpu_filters += filters
            cmd = _base_command() + route.input_args()
            if start_time is not None:
                cmd += ['-ss', str(start_time)]
            if source_duration is not None:
                cmd += ['-t', str(source_duration)]
            cmd += ['-i', video_file, '-map', '0:v:0', '-vf', ','.join(gpu_filters),
                    '-c:v', 'prores_ks_vulkan', '-profile:v', '3', '-vendor', 'apl0',
                    '-tag:v', 'apch', '-async_depth', '4']
            if frames is not None:
                cmd += ['-frames:v', str(frames)]
            cmd += ['-an', '-sn', '-dn', '-fps_mode', 'cfr'] + _container_args(output_file, fps) + [output_file]
            commands.append(cmd)
    cmd = _base_command()
    if start_time is not None:
        cmd += ['-ss', str(start_time)]
    if source_duration is not None:
        cmd += ['-t', str(source_duration)]
    cmd += ['-i', video_file, '-map', '0:v:0', '-vf', ','.join(_scale_filters(target_size) + filters)]
    if frames is not None:
        cmd += ['-frames:v', str(frames)]
    cmd += _prores_args() + ['-an', '-sn', '-dn', '-fps_mode', 'cfr']
    cmd += _container_args(output_file, fps) + [output_file]
    commands.append(cmd)
    return commands


def _run_prores_commands(commands, output_file, timeout, frames=None):
    for i, cmd in enumerate(commands):
        result = _run_media_command(cmd, timeout=timeout)
        if result.returncode and i + 1 < len(commands):
            print(f'   Vulkan ProRes failed; retrying CPU ProRes: {_short_ffmpeg_error(result.stderr, 600)}')
            continue
        _verify_output(result, output_file, frames)
        return


def convert_to_prores_hq(video_file: str, output_dir: str, fps: float = None,
                            target_size: Tuple[int, int] = None, use_gpu: bool = True) -> str:
    """Create 10-bit 4:2:2 ProRes HQ; subsequent stream copying is lossless."""
    os.makedirs(output_dir, exist_ok=True)
    name = Path(video_file).stem
    identity = hashlib.sha256(os.path.abspath(video_file).encode('utf8')).hexdigest()[:10]
    output_file = os.path.join(output_dir, f'{name}_{identity}_prores.mov')
    fps = get_video_fps(video_file) if fps is None else fps
    target_size = target_size or get_video_resolution(video_file)
    filters = [f'fps={fps_string(fps)}:start_time=0']
    commands = _prores_commands(video_file, output_file, fps, target_size, filters, use_gpu=use_gpu)
    print(f'   📹 Converting to ProRes 422 HQ: {Path(video_file).name}')
    _run_prores_commands(commands, output_file, timeout=max(600, int(get_video_duration(video_file) * 8)))
    return output_file


def extract_clip_segment_ffmpeg(video_file: str, start_time: float, duration: float,
                                output_file: str, fps: float, target_size: Tuple[int, int],
                                use_nvenc: bool, gpu_encoder: str = 'h264_nvenc',
                                output_duration: float = None) -> bool:
    """Render exact output frames. The legacy use_nvenc flag selects hardware encoding."""
    try:
        start_time = max(0.0, float(start_time))
        if not math.isfinite(start_time):
            raise ValueError('Invalid source start time')
        output_duration = duration if output_duration is None else output_duration
        filters, frames = _timing_filters(duration, output_duration, fps, target_size)
        use_vulkan = use_nvenc and gpu_encoder in VULKAN_ENCODERS
        commands = []
        route = None
        if use_nvenc and _env_flag('BEATSYNC_GPU_VIDEO_DECODE'):
            route = get_gpu_video_path(video_file, 'vulkan' if use_vulkan else 'cuda',
                                       hardware_decode=_env_flag('BEATSYNC_FFMPEG_CUDA_DECODE'))
        if route:
            timing, _ = _timing_filters(duration, output_duration, fps)
            width, height = normalize_output_size(target_size or get_video_resolution(video_file))
            gpu_filters = (timing[:2] + route.upload_filters()
                           + route.scale_filters(width, height, contain=True) + timing[2:])
            cmd = _base_command() + route.input_args()
            cmd += ['-ss', str(start_time), '-t', str(duration), '-i', video_file, '-map', '0:v:0',
                    '-vf', ','.join(gpu_filters), '-frames:v', str(frames)]
            cmd += vulkan_quality_args(gpu_encoder) if use_vulkan else nvenc_quality_args(gpu_encoder, include_pix_fmt=False)
            cmd += ['-an', '-sn', '-dn', '-fps_mode', 'cfr']
            cmd += _container_args(output_file, fps, gpu_encoder) + [output_file]
            commands.append(cmd)
        cmd = _base_command()
        if use_vulkan:
            cmd += vulkan_device_args(gpu_encoder)
        fallback_filters = filters + (['format=nv12', 'hwupload'] if use_vulkan else [])
        cmd += ['-ss', str(start_time), '-t', str(duration), '-i', video_file, '-map', '0:v:0',
                '-vf', ','.join(fallback_filters), '-frames:v', str(frames)]
        cmd += get_hardware_encoding_args(gpu_encoder) if use_nvenc else get_cpu_h264_quality_args()
        cmd += ['-an', '-sn', '-dn', '-fps_mode', 'cfr']
        cmd += _container_args(output_file, fps, gpu_encoder if use_nvenc else '') + [output_file]
        commands.append(cmd)
        for index, cmd in enumerate(commands):
            result = _run_media_command(cmd, timeout=max(120, int(output_duration * 20)))
            if result.returncode and index + 1 < len(commands):
                print(f'   ⚠️ GPU decode/filter failed; retrying CPU decode/filter with GPU encoding: {_short_ffmpeg_error(result.stderr, 500)}')
                continue
            _verify_output(result, output_file, frames)
            return True
    except Exception as exc:
        print(f'   ⚠️ Clip extraction failed: {exc}')
    return False


def extract_prores_segment_random(video_file: str, duration: float, fps: float,
                                  temp_dir: str, segment_index: int,
                                  start_time: float = None,
                                  output_duration: float = None, target_size: Tuple[int, int] = None,
                                  use_gpu: bool = True) -> str:
    output_file = os.path.join(temp_dir, f'segment_{segment_index:05d}.mov')
    video_duration = get_video_duration(video_file)
    output_duration = duration if output_duration is None else output_duration
    duration = min(duration, video_duration)
    max_start = max(0.0, video_duration - duration)
    start_time = random.uniform(0, max_start) if start_time is None else max(0.0, min(float(start_time), max_start))
    filters, frames = _timing_filters(duration, output_duration, fps)
    commands = _prores_commands(video_file, output_file, fps, target_size or get_video_resolution(video_file),
                               filters, frames=frames, start_time=start_time, use_gpu=use_gpu, source_duration=duration)
    _run_prores_commands(commands, output_file, timeout=max(180, int(output_duration * 20)), frames=frames)
    return output_file


def _concat_path(path):
    value = os.path.abspath(path).replace('\\', '/')
    if '\n' in value or '\r' in value:
        raise ValueError('Media filenames cannot contain line breaks')
    return "'" + value.replace("'", "'\\''") + "'"


def concatenate_videos_ffmpeg(video_files: List[str], output_file: str,
                              audio_file: str = None, start_time: float = 0.0,
                              end_time: float = None, use_nvenc: bool = False,
                              gpu_encoder: str = 'h264_nvenc', fps: float = 30.0,
                              temp_dir: str = None, output_duration: float = None,
                              clip_frame_counts: List[int] = None) -> str:
    """Assemble one frame-locked timeline and commit the verified output atomically."""
    if not video_files:
        raise ValueError('No clips to concatenate')
    fps_string(fps)
    temp_dir = temp_dir or os.path.dirname(os.path.abspath(output_file))
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    concat_file = os.path.join(temp_dir, f'concat_{uuid.uuid4().hex}.ffconcat')
    target = Path(output_file)
    staging = str(target.with_name(f'{target.stem}_{uuid.uuid4().hex}.partial{target.suffix}'))
    if clip_frame_counts is not None and len(clip_frame_counts) != len(video_files):
        raise ValueError('Clip frame counts must match the input clips')
    try:
        with open(concat_file, 'w', encoding='utf8', newline='\n') as script:
            script.write('ffconcat version 1.0\n')
            for index, file in enumerate(video_files):
                script.write(f'file {_concat_path(file)}\n')
                if clip_frame_counts is not None:
                    script.write(f'duration {frame_count_to_seconds(clip_frame_counts[index], fps):.12f}\n')
        if output_duration is None:
            output_duration = sum(get_video_duration(file) for file in video_files)
            if audio_file:
                audio_duration = get_video_duration(audio_file) - start_time
                if end_time is not None and end_time > start_time:
                    audio_duration = min(audio_duration, end_time - start_time)
                output_duration = min(output_duration, audio_duration)
        if output_duration <= 0 or not math.isfinite(output_duration):
            raise ValueError('Invalid final timeline duration')
        frames = max(1, seconds_to_frame_count(output_duration, fps))
        exact_duration = frame_count_to_seconds(frames, fps)
        is_prores = gpu_encoder == 'prores_ks' or (
            not use_nvenc and target.suffix.lower() == '.mov' and video_stream(video_files[0])['codec_name'] == 'prores')

        def build_command(copy):
            cmd = _base_command()
            if not copy and use_nvenc and gpu_encoder in VULKAN_ENCODERS:
                cmd += vulkan_device_args(gpu_encoder)
            cmd += ['-f', 'concat', '-safe', '0', '-i', concat_file]
            if audio_file:
                cmd += ['-ss', str(max(0.0, start_time))]
                if end_time is not None and end_time > start_time:
                    cmd += ['-t', str(end_time - start_time)]
                cmd += ['-i', audio_file]
            cmd += ['-map', '0:v:0']
            if audio_file:
                cmd += ['-map', '1:a:0', '-c:a', 'pcm_s24le', '-ar', '48000',
                        '-af', 'asetpts=PTS-STARTPTS']
            if copy:
                cmd += ['-c:v', 'copy']
            else:
                filters = [f'fps={fps_string(fps)}:start_time=0',
                           f'tpad=stop_mode=clone:stop_duration={exact_duration}']
                if use_nvenc and gpu_encoder in VULKAN_ENCODERS:
                    filters += ['format=nv12', 'hwupload']
                cmd += ['-vf', ','.join(filters)]
                if is_prores:
                    cmd += _prores_args()
                elif use_nvenc:
                    cmd += get_hardware_encoding_args(gpu_encoder)
                else:
                    cmd += get_cpu_h264_quality_args()
                cmd += ['-fps_mode', 'cfr']
            # Explicit end and frame count preserve the last reordered frame;
            # -shortest previously truncated some NVENC/Vulkan + PCM exports.
            cmd += ['-frames:v', str(frames), '-t', str(exact_duration), '-sn', '-dn']
            cmd += _container_args(staging, fps, gpu_encoder if use_nvenc else '')
            return cmd + [staging]

        copy_enabled = _env_flag('BEATSYNC_FAST_CONCAT_COPY', True)
        for copy in ([True, False] if copy_enabled else [False]):
            started = time.perf_counter()
            print('   🔗 Final assembly: ' + ('stream copy + music' if copy else 'full re-encode + music'))
            result = _run_media_command(build_command(copy), timeout=max(600, int(exact_duration * 10)))
            try:
                _verify_output(result, staging, frames)
            except MediaError as exc:
                if not copy:
                    raise
                print(f'   ⚠️ Copy assembly failed; retrying a full re-encode: {exc}')
                _safe_remove_file(staging)
                continue
            os.replace(staging, output_file)
            print(f'   ✓ Final assembly complete in {_fmt_seconds(time.perf_counter() - started)}')
            return output_file
    finally:
        _safe_remove_file(concat_file)
        _safe_remove_file(staging)


def create_browser_preview(video_file: str, output_file: str) -> str:
    """H.264/AAC MP4 for browser playback, retaining the original export separately."""
    target = Path(output_file)
    staging = str(target.with_name(f'{target.stem}_{uuid.uuid4().hex}.partial.mp4'))
    try:
        use_nvenc = get_encoder_support('h264_nvenc')['available']
        use_vulkan = not use_nvenc and get_encoder_support('h264_vulkan')['available']
        commands = []
        route = get_gpu_video_path(video_file, 'cuda' if use_nvenc else 'vulkan') if (use_nvenc or use_vulkan) else None
        if route:
            width, height = normalize_output_size(get_video_resolution(video_file))
            filters = route.upload_filters() + route.scale_filters(width, height, contain=True)
            gpu_cmd = _base_command() + route.input_args()
            gpu_cmd += ['-i', video_file, '-map', '0:v:0', '-map', '0:a:0?', '-vf', ','.join(filters)]
            if use_nvenc:
                gpu_cmd += ['-c:v', 'h264_nvenc'] + nvenc_device_args()
                gpu_cmd += ['-preset', 'p5', '-rc', 'vbr', '-cq', '23', '-b:v', '0']
            else:
                gpu_cmd += vulkan_quality_args('h264_vulkan')
            commands.append(gpu_cmd)
        cmd = _base_command()
        if use_vulkan:
            cmd += vulkan_device_args('h264_vulkan')
        cmd += ['-i', video_file, '-map', '0:v:0', '-map', '0:a:0?']
        if use_nvenc:
            cmd += ['-c:v', 'h264_nvenc'] + nvenc_device_args()
            cmd += ['-preset', 'p5', '-rc', 'vbr', '-cq', '23', '-b:v', '0']
        elif use_vulkan:
            cmd += ['-vf', 'format=nv12,hwupload'] + vulkan_quality_args('h264_vulkan')
        else:
            cmd += ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23']
        if not use_vulkan:
            cmd += ['-pix_fmt', 'yuv420p']
        commands.append(cmd)
        for index, command in enumerate(commands):
            command += ['-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-ac', '2',
                        '-movflags', '+faststart', '-sn', '-dn', staging]
            result = _run_media_command(command, timeout=max(180, int(get_video_duration(video_file) * 10)))
            if result.returncode and index + 1 < len(commands):
                print(f'   GPU preview decode/filter failed; retrying CPU decode with GPU encoding: {_short_ffmpeg_error(result.stderr, 500)}')
                continue
            _verify_output(result, staging)
            break
        os.replace(staging, output_file)
        return output_file
    finally:
        _safe_remove_file(staging)


def detect_video_scene_changes(video_path: str, threshold: float = 0.28,
                               use_gpu: bool = True,
                               analysis_fps: float = 8.0,
                               analysis_width: int = 768) -> List[float]:
    """
    Detect likely montage/scene changes using FFmpeg's scene score.

    GPU note:
    FFmpeg's `scene` comparison itself is a CPU video filter, but when GPU mode
    is enabled this uses CUDA decode + CUDA resize, then downloads a small
    analysis frame for the CPU scene score. That makes the expensive decode/scale
    stage GPU-assisted while preserving the same semantic scene-cut signal.
    If CUDA decode is unsupported for a source codec, it automatically falls
    back to the CPU analysis path.
    """
    try:
        print(f"   🎬 Analyzing scene changes: {os.path.basename(video_path)}")
        print(f"   Threshold: {threshold}")
        print(f"   Analysis: {analysis_fps:g} fps @ {analysis_width}px wide")

        filter_cpu = (
            f"fps={analysis_fps},"
            f"scale={analysis_width}:-2:flags=fast_bilinear,"
            f"select='gt(scene,{threshold})',showinfo"
        )
        commands = []
        route = get_gpu_video_path(video_path) if use_gpu else None
        if route is not None:
            width, height = get_video_resolution(video_path)
            analysis_height = max(2, int(round(height * analysis_width / width / 2)) * 2)
            # Reduce the frame rate while frames are still on the GPU. Doing
            # this after hwdownload unnecessarily transferred every frame.
            gpu_filters = [f'fps={analysis_fps}'] + route.upload_filters()
            gpu_filters += route.scale_filters(analysis_width, analysis_height)
            gpu_filters += ['hwdownload', 'format=nv12',
                            f"select='gt(scene,{threshold})'", 'showinfo']
            commands.append((
                f'GPU-assisted {route.label}',
                [
                    FFMPEG_PATH,
                    '-nostdin',
                    '-hide_banner',
                    *route.input_args(),
                    '-i', video_path,
                    '-vf', ','.join(gpu_filters),
                    '-an', '-sn', '-dn',
                    '-f', 'null',
                    '-'
                ],
            ))

        commands.append((
            'CPU fast scene score',
            [
                FFMPEG_PATH,
                '-nostdin',
                '-hide_banner',
                '-threads', str(MAX_THREADS),
                '-i', video_path,
                '-vf', filter_cpu,
                '-an', '-sn', '-dn',
                '-f', 'null',
                '-'
            ],
        ))

        last_error = ''
        for label, cmd in commands:
            if label.startswith('GPU'):
                print(f"   ⚡ GPU scene analysis: {route.label} + small CPU scene score")
            else:
                if use_gpu:
                    print("   ↪ GPU scene analysis unavailable/failed, using CPU fallback")
                else:
                    print("   💻 CPU scene analysis")

            command_started = time.perf_counter()
            result = _run_media_command(cmd, timeout=300)
            command_elapsed = time.perf_counter() - command_started
            if result.returncode == 0:
                cleaned = _extract_scene_times(result.stderr)
                print(
                    f"   ✓ Found {len(cleaned)} visual scene changes ({label}) "
                    f"in {command_elapsed:.1f}s"
                )
                return cleaned

            print(f"   ⚠️  Scene command failed after {command_elapsed:.1f}s ({label})")
            last_error = _short_ffmpeg_error(result.stderr, max_chars=1200)
            if label.startswith('GPU'):
                print(f"   ⚠️  GPU scene analysis failed, fallback enabled: {last_error}")

        print(f"   ⚠️  Warning: Scene detection failed: {last_error}")
        return []

    except subprocess.TimeoutExpired:
        print(f"   ⚠️  Warning: Scene detection timeout")
        return []
    except Exception as e:
        print(f"   ⚠️  Warning: Could not analyze scene changes: {e}")
        return []

def detect_video_keyframes(video_path: str, min_interval: float = 0.20) -> List[float]:
    """
    Detect codec keyframes/I-frames with ffprobe.

    Keyframes are not always creative scene cuts, but they are useful as a
    fallback signal for footage that was encoded with keyframes at montage cuts.
    """
    try:
        print(f"   🔑 Reading codec keyframes: {os.path.basename(video_path)}")
        cmd = [
            FFPROBE_PATH,
            '-v', 'error',
            '-select_streams', 'v:0',
            '-skip_frame', 'nokey',
            '-show_entries', 'frame=best_effort_timestamp_time,pkt_pts_time,pts_time',
            '-of', 'json',
            video_path,
        ]
        result = _run_media_command(cmd, timeout=120)
        if result.returncode != 0 or not result.stdout.strip():
            return []

        data = json.loads(result.stdout)
        keyframes = []
        for frame in data.get('frames', []):
            ts = frame.get('best_effort_timestamp_time') or frame.get('pkt_pts_time') or frame.get('pts_time')
            if ts is None:
                continue
            try:
                t = float(ts)
            except (TypeError, ValueError):
                continue
            if t >= 0.0:
                keyframes.append(round(t, 3))

        keyframes = sorted(set(keyframes))
        cleaned = []
        for t in keyframes:
            if not cleaned or t - cleaned[-1] >= min_interval:
                cleaned.append(t)

        print(f"   ✓ Found {len(cleaned)} codec keyframes")
        return cleaned
    except subprocess.TimeoutExpired:
        print("   ⚠️  Warning: Keyframe detection timeout")
        return []
    except Exception as e:
        print(f"   ⚠️  Warning: Could not read keyframes: {e}")
        return []
