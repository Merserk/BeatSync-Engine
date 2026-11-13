import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from logger import setup_environment

# Initialize environment
setup_environment()

# Import modules after the portable Python environment is set
import argparse
import json
import random
import shutil
from typing import TypeAlias, List, Dict, Tuple
import numpy as np
from pathlib import Path
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
import warnings
import time

from gpu_cpu_utils import (
    PARALLEL_WORKERS,
    GPU_INFO as gpu_info,
    GPU_AVAILABLE,
    NVENC_AVAILABLE,
    set_gpu_mode,
)
from paths import (
    get_processing_dir,
)

# Import FFmpeg processing module
from ffmpeg_processing import (
    get_video_duration,
    get_video_fps,
    get_video_resolution,
    extract_clip_segment_ffmpeg,
    extract_prores_segment_random,
    concatenate_videos_ffmpeg,
    seconds_to_frame_count,
    frame_count_to_seconds,
)
from auto_mode.stage6_av_planner import (build_planned_clip_sequence, summarize_clip_plan,
                                        verify_planned_clip_sequence)

# Import mode modules
from auto_mode import analyze_beats_auto
from video_encoders import VULKAN_ENCODERS, HARDWARE_ENCODERS, get_encoder_support, get_prores_gpu_support
from ffmpeg_runtime import normalize_output_size
from clip_uniqueness import UniqueFootageError, source_key, validate_unique_clips

warnings.filterwarnings('ignore', message='.*bytes wanted but 0 bytes read.*')

BeatTimes : TypeAlias = np.ndarray
VideoList : TypeAlias = List[str]


def _fmt_seconds(seconds: float) -> str:
    try:
        value = float(seconds)
    except Exception:
        value = 0.0
    if value < 1.0:
        return f"{value * 1000:.0f}ms"
    return f"{value:.1f}s"


def _env_int(name: str, default: int, lo: int = 1, hi: int | None = None) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


def _effective_clip_workers(requested_workers: int, use_nvenc: bool,
                            gpu_encoder: str = 'h264_nvenc') -> int:
    """Choose a stable FFmpeg worker count for clip extraction.

    Multiple simultaneous hardware encodes can fight for one encoder and
    decode/IO bandwidth. Capping the default is a performance-only change: the
    same clip plan, source times, frame counts, and encoder settings are used.
    """
    requested_workers = max(1, int(requested_workers or 1))
    if not use_nvenc:
        return requested_workers

    cpu = os.cpu_count() or 4
    default_nvenc_workers = min(requested_workers, 4, max(1, cpu // 2))
    return _env_int(
        "BEATSYNC_VULKAN_CLIP_WORKERS" if gpu_encoder in VULKAN_ENCODERS else "BEATSYNC_NVENC_CLIP_WORKERS",
        default_nvenc_workers,
        lo=1,
        hi=requested_workers,
    )


def _summarize_clip_timings(timings: List[float], total_duration: float) -> None:
    if not timings:
        return
    arr = np.asarray(timings, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    print(
        f"   ⏱ Clip extraction total: {_fmt_seconds(total_duration)} | "
        f"avg {float(np.mean(arr)):.2f}s, p50 {float(np.percentile(arr, 50)):.2f}s, "
        f"p90 {float(np.percentile(arr, 90)):.2f}s, max {float(np.max(arr)):.2f}s"
    )


def _hardware_encoding_enabled(use_gpu: bool, gpu_encoder: str,
                                is_prores: bool) -> bool:
    if is_prores:
        return False
    if gpu_encoder in HARDWARE_ENCODERS:
        support = get_encoder_support(gpu_encoder)
        if not support['available']:
            raise RuntimeError(f"{HARDWARE_ENCODERS[gpu_encoder]} is unavailable: {support['error']}")
        return True
    return False


def _cleanup_render_dir(path: str, root: str) -> None:
    target = Path(path).resolve()
    parent = Path(root).resolve()
    if target.parent != parent or not target.name.startswith('render_'):
        raise RuntimeError(f'Refusing to clean a directory outside this render: {target}')
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_clip_plan_debug(output_file: str, plan_summary: Dict, planned_clip_sequence: List[Dict]) -> str | None:
    if not planned_clip_sequence:
        return None
    output_dir = os.path.dirname(os.path.abspath(output_file)) or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(output_file))[0]
    debug_path = os.path.join(output_dir, f"{base_name}.clip_plan_debug.json")
    clip_fields = (
        "index", "video_file", "source_name", "start_time", "source_duration", "final_duration",
        "playback_speed", "speed_reason", "speed_confidence", "target", "impact_class",
        "score", "match_score", "flow_score", "highlight_score", "candidate_id",
        "identity_key", "source_time_bucket", "selection_level", "repetition_penalty",
        "repetition_flags", "repetition_repaired", "replaced_candidate_id",
        "highlight_pool_size", "highlight_seed_size", "planner_pool_size",
        "planner_pool_target", "raw_candidate_size", "planner_pool_deficit",
        "avoid_text_clips", "text_score", "text_overlay", "text_candidate_count",
        "text_candidates_filtered", "text_filter_fallback",
        "quality_flags", "low_quality", "audio_start", "audio_end",
        "scene_index", "visual_fingerprints", "source_guard_seconds", "visual_verified",
        "visual_verification_seconds", "visual_verification_repairs",
        "usable_source_start", "usable_source_end", "shot_verified", "shot_refined",
        "source_start_shift", "verified_shot_start", "verified_shot_end", "native_source_frames", "shot_boundary_count",
    )
    payload = {
        "output_file": os.path.abspath(output_file),
        "plan_summary": plan_summary,
        "clips": [
            {field: clip.get(field) for field in clip_fields if field in clip}
            for clip in planned_clip_sequence
        ],
    }
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2)
    return debug_path

# Log status on import if running directly
if __name__ == '__main__':
    if GPU_AVAILABLE:
        print(f"⚡ GPU available: {gpu_info['name']}")
    else:
        print(f"💻 GPU Acceleration: NOT AVAILABLE")


if __name__ == '__main__':
    if NVENC_AVAILABLE:
        print(f"🎬 NVIDIA NVENC: AVAILABLE - Hardware video encoding enabled")
    else:
        print(f"⚠️  NVIDIA NVENC: NOT AVAILABLE")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create an Auto Mode music video with rhythmic audio-video cuts'
    )
    parser.add_argument(
        'mp3_file',
        type=str,
        help='Path to the input audio file (MP3/WAV/FLAC)'
    )
    parser.add_argument(
        'video_directory',
        type=str,
        help='Directory containing MP4/MKV video files'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='output_music_video.mkv',
        help='Output video file path (default: output_music_video.mkv)'
    )
    parser.add_argument(
        '-s', '--start-time',
        type=float,
        default=0.0,
        help='Start time in seconds for audio processing (default: 0.0)'
    )
    parser.add_argument(
        '-e', '--end-time',
        type=float,
        default=None,
        help='End time in seconds for audio processing (default: full duration)'
    )
    parser.add_argument(
        '--export-format',
        choices=['auto', 'prores_hq'],
        default='auto',
        help='Export format: auto uses the selected H.264/H.265 encoder; prores_hq exports ProRes 422 HQ (.mov)'
    )
    parser.add_argument(
        '--gpu',
        action='store_true', default=True,
        help='Use the detected dGPU for analysis and hardware encoding (default)'
    )
    parser.add_argument('--cpu', dest='gpu', action='store_false',
                        help='Explicitly disable GPU analysis and automatic hardware export')
    parser.add_argument(
        '--gpu-encoder',
        type=str,
        choices=['h264_nvenc', 'hevc_nvenc', 'h264_vulkan', 'hevc_vulkan', 'none'],
        default=None,
        help='Video encoder, independent of analysis. Default: the first available dGPU encoder.'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=None,
        help='Output FPS (frames per second). If not specified, auto-detect from input video (default: auto)'
    )
    speed_group = parser.add_mutually_exclusive_group()
    speed_group.add_argument(
        '--auto-speed-control',
        dest='auto_speed_clip_control',
        action='store_true',
        default=True,
        help='Enable smart per-clip speed control in Auto Mode (default: enabled)'
    )
    speed_group.add_argument(
        '--no-auto-speed-control',
        dest='auto_speed_clip_control',
        action='store_false',
        help='Disable smart per-clip speed control'
    )
    text_group = parser.add_mutually_exclusive_group()
    text_group.add_argument(
        '--avoid-text',
        dest='avoid_text_clips',
        action='store_true',
        default=True,
        help='Avoid clips with obvious credits, titles, subtitles, captions, UI, or readable text (default: enabled)'
    )
    text_group.add_argument(
        '--no-avoid-text',
        dest='avoid_text_clips',
        action='store_false',
        help='Allow clips with visible text overlays'
    )

    return parser.parse_args()


def get_video_files(directory : str) -> VideoList:
    video_extensions = ['.mp4', '.MP4', '.mkv', '.MKV']
    video_files = []

    for ext in video_extensions:
        video_files.extend(Path(directory).glob(f'*{ext}'))

    if not video_files:
        raise ValueError(f'No MP4/MKV files found in {directory}')

    return [str(f) for f in video_files]


def build_frame_aligned_cut_timeline(beat_times: BeatTimes, audio_duration: float,
                                     fps: float):
    """
    Build the output cut timeline on absolute video frame numbers.

    This is the critical sync stage: each cut boundary is quantized once from its
    absolute beat time. Segment durations are then frame differences between
    absolute boundaries, so rounding error cannot accumulate from clip to clip.

    The first and last boundaries stay locked to the audio timeline.
    """
    if fps <= 0:
        raise ValueError(f"Invalid FPS for cut timeline: {fps}")
    if audio_duration <= 0:
        raise ValueError(f"Invalid audio duration: {audio_duration}")

    beats = np.asarray(beat_times, dtype=float).reshape(-1)
    beats = beats[np.isfinite(beats)]
    if beats.size == 0:
        raise ValueError("No valid beat times were provided.")

    beats = np.sort(beats)

    internal_beats = beats[(beats > 0.0) & (beats < audio_duration)]

    raw_cut_times = np.concatenate(([0.0], internal_beats, [audio_duration]))

    # Quantize ABSOLUTE cut positions, not per-segment durations. This removes
    # cumulative drift caused by round((beat[i+1] - beat[i]) * fps) on each clip.
    end_frame = max(1, seconds_to_frame_count(audio_duration, fps))
    cut_frames = np.rint(raw_cut_times * fps).astype(int)
    cut_frames = np.clip(cut_frames, 0, end_frame)
    cut_frames[0] = 0
    cut_frames[-1] = end_frame
    cut_frames = np.unique(cut_frames)

    if cut_frames.size == 0 or cut_frames[0] != 0:
        cut_frames = np.insert(cut_frames, 0, 0)
    if cut_frames[-1] != end_frame:
        cut_frames = np.append(cut_frames, end_frame)

    if cut_frames.size < 2:
        cut_frames = np.array([0, end_frame], dtype=int)

    segment_frames = np.diff(cut_frames).astype(int)
    segment_durations = segment_frames / fps
    cut_times = cut_frames / fps
    dropped_boundaries = max(0, raw_cut_times.size - cut_frames.size)

    return cut_times, segment_frames, segment_durations, dropped_boundaries


def create_clip_parallel(args):
    """
    Wrapper function for parallel clip creation using FFmpeg.

    Source intervals are allocated and checked before parallel rendering.
    """
    clip_started = time.perf_counter()
    planned_clip = None
    if len(args) >= 9:
        (i, video_file, final_duration, target_size,
         use_nvenc, gpu_encoder, temp_dir, fps, planned_clip) = args
    else:
        (i, video_file, final_duration, target_size,
         use_nvenc, gpu_encoder, temp_dir, fps) = args
    
    try:
        if planned_clip:
            video_file = planned_clip.get('video_file') or video_file
            video_duration = get_video_duration(video_file)
            source_duration = float(planned_clip.get('source_duration', final_duration))
            clip_start = float(planned_clip.get('start_time', 0.0))
            if clip_start < 0 or source_duration <= 0 or clip_start + source_duration > video_duration + 1e-3:
                raise UniqueFootageError('Planned source interval is outside the video; refusing to shift or repeat footage.')
        else:
            raise UniqueFootageError('A unique source interval must be allocated before rendering a clip.')
            
        
        # Output file
        temp_clip_path = os.path.join(temp_dir, f"temp_clip_{i}_{uuid.uuid4().hex}.mp4")
        
        extract_kwargs = {
            'video_file': video_file,
            'start_time': clip_start,
            'duration': source_duration,
            'output_file': temp_clip_path,
            'fps': fps,
            'target_size': target_size,
            'use_nvenc': use_nvenc,
            'gpu_encoder': gpu_encoder,
            'output_duration': final_duration,
        }

        success = extract_clip_segment_ffmpeg(**extract_kwargs)
        
        elapsed = time.perf_counter() - clip_started
        if not success:
            return (i, None, target_size, None, "FFmpeg extraction failed", elapsed)
        
        return (i, temp_clip_path, target_size, temp_clip_path, None, elapsed)
        
    except Exception as e:
        elapsed = time.perf_counter() - clip_started
        return (i, None, target_size, None, str(e), elapsed)


def _build_unique_fallback_plan(video_files, cut_times, durations):
    """Allocate unused source-frame ranges before any parallel worker starts."""
    sources = {}
    for path in video_files:
        key = source_key(path)
        if key in sources:
            continue
        source_fps = get_video_fps(path)
        frame_count = int(np.floor(get_video_duration(path) * source_fps + 1e-6))
        sources[key] = {'path': path, 'fps': source_fps, 'free': [(0, frame_count)]}
    plan = []
    for index, duration in enumerate(durations):
        choices = []
        for key, source in sources.items():
            needed = int(np.ceil(float(duration) * source['fps'] - 1e-6))
            for gap_index, (first, last) in enumerate(source['free']):
                if last - first >= needed:
                    choices.append((key, gap_index, needed))
        if not choices:
            raise UniqueFootageError('Not enough unused source footage to fill the edit. Add footage or shorten it.')
        key, gap_index, needed = random.choice(choices)
        source = sources[key]
        first, last = source['free'].pop(gap_index)
        start = random.randint(first, last - needed)
        # Reserve a frame between allocations so rounding cannot reuse a frame.
        if start - 1 > first:
            source['free'].append((first, start - 1))
        if start + needed + 1 < last:
            source['free'].append((start + needed + 1, last))
        plan.append({'index': index, 'video_file': source['path'], 'source_name': Path(source['path']).name,
                     'start_time': start / source['fps'], 'source_duration': float(duration),
                     'final_duration': float(duration), 'playback_speed': 1.0,
                     'selection_level': 'unique_source_fallback',
                     'audio_start': float(cut_times[index]), 'audio_end': float(cut_times[index + 1])})
    validate_unique_clips(plan)
    return plan


def create_music_video(audio_file: str, video_files: VideoList, beat_times: BeatTimes,
                      output_file: str = 'output_music_video.mkv',
                      start_time: float = 0.0, end_time: float = None,
                      max_workers: int = None,
                      beat_info: dict = None,
                      export_format: str = 'auto', use_gpu: bool = True,
                      gpu_encoder: str = None, fps: float = None,
                      auto_speed_clip_control: bool = True,
                      avoid_text_clips: bool = True) -> str:
    """
    Creates a music video with video clips cut to detected beats.
    
    **PURE FFMPEG IMPLEMENTATION - FRAME-ACCURATE**
    
    ✅ NO BATCH PROCESSING: FFmpeg handles memory independently
    ✅ FRAME-ACCURATE: Uses exact frame counts for zero drift
    ✅ NO CUMULATIVE ERROR: Each segment is precisely timed
    
    Args:
        audio_file: Path to audio file
        video_files: List of video file paths
        beat_times: Array of beat times (already processed by mode)
        output_file: Output file path
        start_time: Audio start time
        end_time: Audio end time
        max_workers: Number of parallel workers
        beat_info: Beat information dictionary
        export_format: 'auto' for H.264/H.265, or 'prores_hq' for ProRes 422 HQ
        use_gpu: Use GPU acceleration
        gpu_encoder: GPU encoder to use
        fps: Output FPS
    
    Returns:
        Path to output video file
    """
    if len(beat_times) == 0:
        raise ValueError("No beats were detected. Cannot create video.")
    if export_format not in ('auto', 'prores_hq'):
        raise ValueError(f'Unsupported export format: {export_format}')
    is_prores = export_format == 'prores_hq'

    video_creation_started = time.perf_counter()

    if max_workers is None:
        max_workers = PARALLEL_WORKERS

    # Verify the selected encoder before touching the processing directory.
    # use_nvenc is the historical helper argument for hardware encoding.
    if gpu_encoder is None and not is_prores:
        gpu_encoder = next((name for name in HARDWARE_ENCODERS
                            if use_gpu and get_encoder_support(name)['available']), 'none')
    use_nvenc = _hardware_encoding_enabled(use_gpu, gpu_encoder, is_prores)

    # Determine FPS to use
    if fps is None:
        # Auto-detect from first video file
        try:
            fps = get_video_fps(video_files[0])
            print(f"🎞️ Auto-detected FPS from input video: {fps}")
        except Exception as e:
            fps = 30.0
            print(f"⚠️ Could not detect FPS, using default: {fps}")
    else:
        print(f"🎞️ Using custom FPS: {fps}")

    # Each render owns its temporary directory. Parallel CLI/UI runs cannot
    # erase each other's clips.
    processing_root = get_processing_dir()
    session_temp_dir = os.path.join(processing_root, f'render_{uuid.uuid4().hex}')
    print(f"📁 Processing directory: {session_temp_dir}")

    # Determine processing mode
    requested_workers = max_workers
    max_workers = _effective_clip_workers(max_workers, use_nvenc, gpu_encoder)
    render_info = beat_info.setdefault("render_info", {}) if isinstance(beat_info, dict) else {}
    
    gpu_status = "⚡ GPU" if use_gpu and GPU_AVAILABLE else "💻 CPU"
    encoder_status = f"⚡ {gpu_encoder.upper()}" if use_nvenc else ("ProRes 422 HQ" if is_prores else "💻 CPU (libx264)")
    render_info.update({
        "clip_workers": int(max_workers),
        "requested_workers": int(requested_workers),
        "encoder": gpu_encoder.upper() if use_nvenc else ("PRORES_HQ" if is_prores else "H264_CPU"),
        "output_fps": float(fps),
        "auto_speed_clip_control": bool(auto_speed_clip_control),
        "avoid_text_clips": bool(avoid_text_clips),
    })
    # Determine mode name
    mode_name = beat_info.get('mode', 'unknown') if beat_info else 'unknown'
    
    print(f"🎬 Processing Settings:")
    if max_workers != requested_workers:
        print(f"   • Parallel clip workers: {max_workers} (requested {requested_workers}; hardware encoder contention cap)")
    else:
        print(f"   • Parallel clip workers: {max_workers}")
    print(f"   • Video processing: {encoder_status}")
    print(f"   • Output FPS: {fps}")
    print(f"   • Frame-accurate mode: ENABLED (zero drift)")
    print(f"   • Generation mode: {mode_name}")
    if is_prores:
        print(f"   • Export Format: Apple ProRes 422 HQ (.mov; 10-bit 4:2:2)")
    else:
        print(f"   • Export Format: H.264/H.265 ({Path(output_file).suffix})")

    # Get audio duration using ffprobe
    audio_duration = get_video_duration(audio_file)
    if end_time and end_time > start_time:
        audio_duration = end_time - start_time
    elif start_time > 0:
        audio_duration = audio_duration - start_time
    render_info["audio_duration"] = float(audio_duration)
    
    print(f"🎵 Audio duration: {audio_duration:.2f} seconds")


    # Build one frame-locked output timeline before creating clips.
    selected_beats, segment_frames, segment_durations, dropped_boundaries = build_frame_aligned_cut_timeline(
        beat_times, audio_duration, fps
    )
    total_clips = len(segment_durations)
    render_info.update({
        "render_cuts": int(total_clips),
        "timeline_boundaries": int(len(selected_beats)),
        "timeline_frames": int(sum(segment_frames)),
    })

    print(f"🎬 Creating video with {total_clips} frame-locked cuts")
    print(f"⏱️  Cut timeline: {len(selected_beats)} boundaries, {sum(segment_frames)} frames")
    if dropped_boundaries:
        print(f"   ⚠️  Dropped {dropped_boundaries} duplicate/too-close cut boundaries after frame quantization")
    planned_clip_sequence = build_planned_clip_sequence(
        cut_times=selected_beats,
        segment_durations=segment_durations,
        beat_info=beat_info,
        video_files=video_files,
        auto_speed_clip_control=auto_speed_clip_control,
        avoid_text_clips=avoid_text_clips,
    )
    if not planned_clip_sequence:
        print('   No visual library: allocating unused source intervals for every cut')
        planned_clip_sequence = _build_unique_fallback_plan(video_files, selected_beats, segment_durations)
    validate_unique_clips(planned_clip_sequence)
    allowed_sources = {source_key(path) for path in video_files}
    source_durations = {}
    def validate_source_bounds(plan):
        for clip in plan:
            path = clip['video_file']
            key = source_key(path)
            if key not in allowed_sources:
                raise UniqueFootageError('A planned clip is not from one of the selected input videos.')
            if key not in source_durations:
                source_durations[key] = get_video_duration(path)
            if clip['start_time'] + clip['source_duration'] > source_durations[key] + 1e-3:
                raise UniqueFootageError('A planned clip extends past its source; refusing to shift or repeat it.')
    validate_source_bounds(planned_clip_sequence)
    planned_clip_sequence = verify_planned_clip_sequence(
        planned_clip_sequence, selected_beats, segment_durations, beat_info, video_files,
        use_gpu=use_gpu, auto_speed_clip_control=auto_speed_clip_control, avoid_text_clips=avoid_text_clips)
    validate_source_bounds(planned_clip_sequence)
    render_info['visual_verification_seconds'] = max(
        float(item.get('visual_verification_seconds', 0)) for item in planned_clip_sequence)
    if planned_clip_sequence:
        plan_summary = summarize_clip_plan(planned_clip_sequence)
        if beat_info is not None:
            beat_info['clip_plan_summary'] = plan_summary
            render_info["plan_summary"] = plan_summary
        print(f"🧠 Auto visual planner: {plan_summary['clip_count']} planned clips")
        print(f"   Sources used: {plan_summary.get('source_count', 0)}")
        print(f"   Targets: {plan_summary.get('targets', {})}")
        print(f"   AI-tagged source moments used: {plan_summary.get('ai_tagged', 0)}")
        print(
            f"   Highlight pool: {plan_summary.get('highlight_pool_size', 0)}, "
            f"planner pool: {plan_summary.get('planner_pool_size', 0)}/"
            f"{plan_summary.get('planner_pool_target', 0)}, "
            f"raw candidates: {plan_summary.get('raw_candidate_size', 0)}, "
            f"candidate reuse: {plan_summary.get('candidate_reuse_count', 0)}, "
            f"scene repeats: {plan_summary.get('scene_repetition_count', 0)}"
        )
        print(
            f"   Avoid text: {'on' if plan_summary.get('avoid_text_clips') else 'off'}, "
            f"filtered {plan_summary.get('text_candidates_filtered', 0)}/"
            f"{plan_summary.get('text_candidate_count', 0)}, "
            f"used {plan_summary.get('text_overlay_used_count', 0)}"
        )
        print(
            f"   Whole-edit uniqueness: repeated scenes {plan_summary.get('global_scene_repetition_count', 0)}, "
            f"source overlaps {plan_summary.get('source_overlap_count', 0)}, "
            f"visual duplicates {plan_summary.get('visual_duplicate_count', 0)}, "
            f"every source frame checked in {plan_summary.get('visual_verification_seconds', 0):.2f}s"
        )
        print(f"   Clean source shots: {plan_summary.get('shot_verified_clip_count', 0)}/{len(planned_clip_sequence)} verified, "
              f"{plan_summary.get('shot_refined_clip_count', 0)} source intervals adjusted")
        print(
            f"   Match/flow: {plan_summary.get('avg_match_score', 0.0):.2f}/"
            f"{plan_summary.get('avg_flow_score', 0.0):.2f}"
        )
        print(
            f"   Speed control: {plan_summary.get('speed_slowed_count', 0)} slowed, "
            f"{plan_summary.get('speed_sped_up_count', 0)} sped up, "
            f"{plan_summary.get('speed_original_count', 0)} original "
            f"(avg {plan_summary.get('average_playback_speed', 1.0):.2f}x)"
        )
        try:
            debug_path = _write_clip_plan_debug(output_file, plan_summary, planned_clip_sequence)
            if debug_path:
                render_info["clip_plan_debug"] = debug_path
                print(f"   Clip plan debug: {debug_path}")
        except Exception as e:
            print(f"   ⚠️  Clip plan debug write failed: {e}")

    # ProRes HQ export shares the same verified, frame-locked edit plan.
    os.makedirs(session_temp_dir, exist_ok=False)
    if is_prores:
        print(f"\n{'='*60}")
        print(f"🎬 PRORES 422 HQ EXPORT")
        print(f"{'='*60}")
        
        # Encode only the selected original-source intervals. Encoding the
        # entire input first does not improve the exact output frame count.
        prores_fps = fps
        print(f"🎞️ Using FPS: {prores_fps} (for frame-perfect precision)")
        
        target_size = normalize_output_size(get_video_resolution(video_files[0]))
        render_info['target_resolution'] = f'{target_size[0]}x{target_size[1]}'
        
        # Encode selected original intervals to ProRes with exact frame counts.
        print(f"\n{'='*60}")
        print(f"✂️  EXTRACTING SEGMENTS (FRAME-PERFECT PRECISION)")
        print(f"{'='*60}")
        print(f"   Mode: Frame-accurate re-encoding")
        print(f"   Method: Exact frame count calculation")
        print(f"   Playback: forward")
        print(f"   Audio: Stripped (will add music at the end)")
        print(f"   FPS: {prores_fps} (fixed)")
        
        segment_files = [None] * len(segment_durations)
        segments_dir = os.path.join(session_temp_dir, 'segments')
        os.makedirs(segments_dir, exist_ok=True)
        gpu_prores = use_gpu and get_prores_gpu_support()['available']
        prores_workers = _env_int('BEATSYNC_PRORES_CLIP_WORKERS', min(4, max_workers), lo=1, hi=max_workers)
        render_info['clip_workers'] = prores_workers
        render_info['prores_encoder'] = 'prores_ks_vulkan' if gpu_prores else 'prores_ks'
        print(f"   ProRes encoder: {'Vulkan (dGPU)' if gpu_prores else 'CPU'}; workers: {prores_workers}")
        prores_jobs = []
        
        for i, exact_duration in enumerate(segment_durations):
            # Duration comes from the absolute frame-locked timeline.
            planned_clip = planned_clip_sequence[i]
            prores_file = os.path.abspath(planned_clip['video_file'])
            segment_start = float(planned_clip['start_time'])
            source_duration = float(planned_clip['source_duration'])

            prores_jobs.append((prores_file, source_duration, prores_fps, segments_dir, i,
                                 segment_start, exact_duration, target_size, gpu_prores))
        def prores_segment(values):
            return extract_prores_segment_random(*values)
        with ThreadPoolExecutor(max_workers=prores_workers) as executor:
            futures = {executor.submit(prores_segment, values): values[4] for values in prores_jobs}
            for completed, future in enumerate(as_completed(futures), 1):
                segment_files[futures[future]] = future.result()
                if completed % 10 == 0 or completed == total_clips:
                    print(f'   ✓ Extracted {completed}/{total_clips} segments (frame-perfect)')
        
        print(f"✓ Extracted all {len(segment_files)} segments (frame-perfect, video only)")
        
        # Concatenate and add audio
        print(f"\n{'='*60}")
        print(f"🔗 LOSSLESS CONCATENATION + MUSIC")
        print(f"{'='*60}")
        
        concatenate_videos_ffmpeg(
            video_files=segment_files,
            output_file=output_file,
            audio_file=audio_file,
            start_time=start_time,
            end_time=end_time,
            use_nvenc=False,  # ProRes uses stream copy
            gpu_encoder='prores_ks', fps=fps, temp_dir=session_temp_dir,
            output_duration=frame_count_to_seconds(sum(segment_frames), fps),
            clip_frame_counts=segment_frames,
        )
        
        print(f"\n{'='*60}")
        print(f"✅ PRORES HQ VIDEO CREATION COMPLETE!")
        print(f"   Output: {output_file}")
        print(f"   Method: Frame-perfect re-encoding + lossless concatenation")
        print(f"   Quality: ProRes 422 HQ (stream-copy assembly)")
        print(f"   Audio: Music track from input file")
        print(f"   FPS: {prores_fps} (fixed)")
        print(f"   Total Segments: {len(segment_files)}")
        print(f"   Timing Precision: Frame-perfect (zero drift)")
        print(f"{'='*60}\n")
        
        print("🧹 Cleaning up temporary files...")
        gc.collect()
        _cleanup_render_dir(session_temp_dir, processing_root)
        return output_file
    
    # STANDARD MODE - Direct parallel processing (NO BATCHES)
    else:
        # Get target resolution from first video
        target_size = normalize_output_size(get_video_resolution(video_files[0]))
        render_info["target_resolution"] = f"{target_size[0]}x{target_size[1]}"
        print(f"🎞️ Target resolution: {target_size[0]}x{target_size[1]}")
        
        print(f"\n{'='*60}")
        print(f"🎬 PROCESSING ALL CLIPS (No batch processing with FFmpeg)")
        print(f"   Total clips: {total_clips}")
        print(f"   Parallel workers: {max_workers}")
        print(f"   Frame-accurate: ENABLED")
        if use_nvenc:
            print(f"   Encoder: ⚡ {VULKAN_ENCODERS.get(gpu_encoder, gpu_encoder.upper())} (GPU-accelerated)")
        else:
            print(f"   Encoder: 💻 libx264 (CPU)")
        print(f"{'='*60}\n")
        
        clip_args = []
        for i, final_duration in enumerate(segment_durations):
            # Duration comes from the absolute frame-locked cut timeline.
            planned_clip = planned_clip_sequence[i]
            video_file = planned_clip['video_file']
            clip_args.append((i, video_file, final_duration,
                            target_size, use_nvenc, gpu_encoder, session_temp_dir, fps,
                            planned_clip))
        
        clip_files = [None] * len(clip_args)
        clip_timings: List[float] = []
        clip_stage_started = time.perf_counter()
        
        # Process all clips in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(create_clip_parallel, args): idx 
                for idx, args in enumerate(clip_args)
            }
            
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result_tuple = future.result()
                    if len(result_tuple) >= 6:
                        i, clip_path, new_target_size, temp_path, error, clip_elapsed = result_tuple
                    else:
                        i, clip_path, new_target_size, temp_path, error = result_tuple
                        clip_elapsed = 0.0
                    
                    if clip_elapsed:
                        clip_timings.append(float(clip_elapsed))
                    
                    if error:
                        print(f"⚠️  Warning: Clip {i+1} failed after {_fmt_seconds(clip_elapsed)}: {error}")
                        continue
                    
                    if clip_path is not None:
                        clip_files[idx] = clip_path
                        
                        completed += 1
                        if completed % 10 == 0 or completed == len(clip_args):
                            progress = (completed / len(clip_args)) * 100
                            elapsed = time.perf_counter() - clip_stage_started
                            rate = completed / max(0.001, elapsed)
                            print(
                                f"   ⚡ Progress: {completed}/{len(clip_args)} clips ({progress:.1f}%) "
                                f"[{_fmt_seconds(elapsed)}, {rate:.2f} clips/s]"
                            )
                    
                except Exception as e:
                    print(f"⚠️  Warning: Error processing clip: {str(e)}")
                    continue
        
        clip_stage_seconds = time.perf_counter() - clip_stage_started
        _summarize_clip_timings(clip_timings, clip_stage_seconds)

        # Do not silently drop failed clips. Dropping one segment compresses the
        # output timeline and makes every later cut drift against the audio.
        failed_count = sum(1 for f in clip_files if f is None)
        if failed_count:
            raise RuntimeError(
                f"{failed_count} clip(s) failed; refusing to concatenate an incomplete timeline."
            )

        if not clip_files:
            raise ValueError('No valid video clips could be created')
 
        print(f"\n{'='*60}")
        print(f"🎬 FINAL ASSEMBLY: Concatenating {len(clip_files)} clips")
        print(f"{'='*60}\n")
        
        # Concatenate all clips and add audio
        assembly_started = time.perf_counter()
        concatenate_videos_ffmpeg(
            video_files=clip_files,
            output_file=output_file,
            audio_file=audio_file,
            start_time=start_time,
            end_time=end_time,
            use_nvenc=use_nvenc,
            gpu_encoder=gpu_encoder,
            fps=fps,
            temp_dir=session_temp_dir,
            output_duration=frame_count_to_seconds(sum(segment_frames), fps),
            clip_frame_counts=segment_frames,
        )
        assembly_seconds = time.perf_counter() - assembly_started
        render_info["final_assembly_seconds"] = float(assembly_seconds)
        print(f"   ⏱ Final assembly total: {_fmt_seconds(assembly_seconds)}")
 
        print(f"\n🧹 Cleaning up resources...")
        
        # Cleanup clip files
        for clip_file in clip_files:
            try:
                if os.path.exists(clip_file):
                    os.remove(clip_file)
            except Exception as e:
                print(f"⚠️  Warning: Could not delete clip file: {e}")
        
        # Clean up processing directory
        try:
            if os.path.exists(session_temp_dir):
                _cleanup_render_dir(session_temp_dir, processing_root)
                print(f"✓ Cleaned up processing directory")
        except Exception as e:
            print(f"⚠️  Warning: Could not delete processing directory: {e}")
        
        gc.collect()
 
        print(f"\n{'='*60}")
        print(f"✅ VIDEO CREATION COMPLETE!")
        print(f"   Output: {output_file}")
        print(f"   FPS: {fps} (frame-accurate)")
        print(f"   Total Cuts: {total_clips}")
        print(f"   Zero Drift: Absolute frame-locked cut timeline")
        print(f"   Total video creation time: {_fmt_seconds(time.perf_counter() - video_creation_started)}")
        print(f"{'='*60}\n")
        
        return output_file
 
 
def main() -> None:
    args = parse_arguments()
 
    if not os.path.exists(args.mp3_file):
        raise FileNotFoundError('Audio file not found: ' + args.mp3_file)
 
    if not os.path.isdir(args.video_directory):
        raise NotADirectoryError('Video directory not found: ' + args.video_directory)
 
    # Enable GPU mode if requested
    if args.gpu:
        if GPU_AVAILABLE:
            set_gpu_mode(True)
            print(f"⚡ GPU acceleration ENABLED: {gpu_info}")
        else:
            print(f"⚠️ Taichi analysis unavailable; using supported dGPU video paths and CPU analysis fallback")
 
    is_prores = args.export_format == 'prores_hq'
    if args.gpu_encoder is None and not is_prores:
        args.gpu_encoder = next((name for name in HARDWARE_ENCODERS
                                 if args.gpu and get_encoder_support(name)['available']), 'none')
    hardware_export = _hardware_encoding_enabled(args.gpu, args.gpu_encoder, is_prores)
    print(f"\n{'='*60}")
    print(f"🎵 BEATSYNC ENGINE - AUTO MODE")
    print(f"   Audio Analysis: {'⚡ GPU' if args.gpu else '💻 CPU'}")
    if is_prores:
        print(f"   Video Encoding: {'⚡ Vulkan ProRes (dGPU)' if args.gpu and get_prores_gpu_support()['available'] else '💻 CPU ProRes'}")
    elif hardware_export:
        print(f"   Video Encoding: ⚡ {args.gpu_encoder.upper()}")
    else:
        print(f"   Video Encoding: 💻 CPU")
    if args.fps:
        print(f"   FPS: {args.fps} (custom)")
    else:
        print(f"   FPS: Auto-detect from input video")
    print(f"   Mode: AUTO")
    if is_prores:
        print(f"   Export: ProRes 422 HQ (.mov; 10-bit 4:2:2)")
    else:
        print(f"   Export: 📹 H.264/H.265 ({Path(args.output).suffix}) - Frame Accurate")
    print(f"{'='*60}\n")
    
    print(f'📁 Audio file: {args.mp3_file}')
    video_files = get_video_files(args.video_directory)
    print(f'✓ Found {len(video_files)} video files')
 
    print(f"🤖 Using AUTO mode")
    selected_beats, beat_info = analyze_beats_auto(
        args.mp3_file,
        start_time=args.start_time,
        end_time=args.end_time,
        use_gpu=args.gpu,
        video_files=video_files,
    )
 
    print(f'✓ Selected {len(selected_beats)} cuts for video')
 
    output_file = args.output
    if is_prores and not output_file.lower().endswith('.mov'):
        base, _ = os.path.splitext(output_file)
        output_file = base + '.mov'
        print(f'📝 Changed output to .mov for ProRes HQ: {output_file}')
    elif not is_prores and not output_file.lower().endswith(('.mp4', '.mkv')):
        base, _ = os.path.splitext(output_file)
        output_file = base + '.mkv'
        print(f'📝 Changed output to .mkv: {output_file}')
 
    print(f'\n🎬 Starting video creation (frame-accurate)...\n')
    output_file = create_music_video(
        args.mp3_file,
        video_files,
        selected_beats,  # Pass pre-processed beats
        output_file=output_file,
        start_time=args.start_time,
        end_time=args.end_time,
        max_workers=PARALLEL_WORKERS,
        beat_info=beat_info,
        export_format=args.export_format,
        use_gpu=args.gpu,
        gpu_encoder=args.gpu_encoder,
        fps=args.fps,
        auto_speed_clip_control=args.auto_speed_clip_control,
        avoid_text_clips=args.avoid_text_clips,
    )
 
    print(f'✅ Music video created successfully: {output_file}')
 
 
if __name__ == '__main__':
    main()
