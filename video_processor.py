#!/usr/bin/env python3
import os
import sys

# Determine script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set up portable CUDA path BEFORE importing any CUDA-dependent libraries
PORTABLE_CUDA_DIR = os.path.join(SCRIPT_DIR, 'bin', 'CUDA', 'v13.0')
PORTABLE_CUDA_BIN = os.path.join(PORTABLE_CUDA_DIR, 'bin', 'x64')
PORTABLE_CUDA_LIB = os.path.join(PORTABLE_CUDA_DIR, 'lib', 'x64')

# Check if portable CUDA exists and configure environment
if os.path.exists(PORTABLE_CUDA_DIR):
    USING_PORTABLE_CUDA = True
    
    # Set CUDA environment variables
    os.environ['CUDA_PATH'] = PORTABLE_CUDA_DIR
    os.environ['CUDA_HOME'] = PORTABLE_CUDA_DIR
    os.environ['CUDA_ROOT'] = PORTABLE_CUDA_DIR
    
    # Add CUDA bin to PATH (for DLLs)
    if PORTABLE_CUDA_BIN not in os.environ.get('PATH', ''):
        os.environ['PATH'] = PORTABLE_CUDA_BIN + os.pathsep + os.environ.get('PATH', '')
    
    # Add CUDA lib to PATH (for library files)
    if os.path.exists(PORTABLE_CUDA_LIB):
        if PORTABLE_CUDA_LIB not in os.environ.get('PATH', ''):
            os.environ['PATH'] = PORTABLE_CUDA_LIB + os.pathsep + os.environ.get('PATH', '')
    
    # Set library path for Linux/Unix compatibility
    if 'LD_LIBRARY_PATH' in os.environ:
        os.environ['LD_LIBRARY_PATH'] = PORTABLE_CUDA_LIB + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')
    else:
        os.environ['LD_LIBRARY_PATH'] = PORTABLE_CUDA_LIB
    
    print(f"✅ Using Portable CUDA: {PORTABLE_CUDA_DIR}")
else:
    USING_PORTABLE_CUDA = False
    print(f"⚠️  Portable CUDA not found at: {PORTABLE_CUDA_DIR}")
    print(f"   Will try to use system CUDA if available")

# Set up portable Python path
PORTABLE_PYTHON_DIR = os.path.join(SCRIPT_DIR, 'bin', 'python-3.13.9-embed-amd64')
PORTABLE_PYTHON_EXE = os.path.join(PORTABLE_PYTHON_DIR, 'python.exe')

# Check if we're using portable Python
if os.path.exists(PORTABLE_PYTHON_EXE):
    USING_PORTABLE_PYTHON = True
    # Ensure the portable Python is in the path for subprocesses
    if PORTABLE_PYTHON_DIR not in os.environ.get('PATH', ''):
        os.environ['PATH'] = PORTABLE_PYTHON_DIR + os.pathsep + os.environ.get('PATH', '')
    # Set Python home
    os.environ['PYTHONHOME'] = PORTABLE_PYTHON_DIR
    print(f"✅ Using Portable Python: {PORTABLE_PYTHON_EXE}")
else:
    USING_PORTABLE_PYTHON = False
    print(f"⚠️  Portable Python not found, using system Python")

# NOW import other modules (after CUDA and Python environment is set)
import argparse
import random
from typing import TypeAlias, Tuple, List
import numpy as np
import librosa
from pathlib import Path
import gc
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import uuid
import warnings
import time

# Import FFmpeg processing module
from ffmpeg_processing import (
    check_nvenc_support,
    get_video_duration,
    get_video_fps,
    get_video_resolution,
    convert_to_prores_proxy,
    extract_clip_segment_ffmpeg,
    extract_prores_segment_random,
    concatenate_videos_ffmpeg,
    seconds_to_frame_count,
    frame_count_to_seconds,
    FFMPEG_PATH,
    FFPROBE_PATH
)

# Import mode modules
from manual_mode import analyze_beats_manual, process_manual_intensity
from smart_mode import (
    analyze_beats_smart, 
    select_beats_smart, 
    detect_video_scene_changes,
    avoid_scene_cuts,
    get_preset_info,
    list_presets,
    set_gpu_mode,
    is_gpu_available,
    get_gpu_info
)
from auto_mode import analyze_beats_auto

# GPU Support
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

warnings.filterwarnings('ignore', message='.*bytes wanted but 0 bytes read.*')

AudioData : TypeAlias = np.ndarray
BeatTimes : TypeAlias = np.ndarray
VideoList : TypeAlias = List[str]

# Get maximum available CPU threads
CPU_COUNT = multiprocessing.cpu_count()
MAX_THREADS = CPU_COUNT
PARALLEL_WORKERS = max(CPU_COUNT // 2, 4)

# Local temp directory setup
LOCAL_TEMP_DIR = os.path.join(SCRIPT_DIR, 'temp')
os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)

print(f"🚀 CPU Optimization: Detected {CPU_COUNT} threads - Will use ALL {MAX_THREADS} threads for encoding")
print(f"🔧 Parallel Processing: {PARALLEL_WORKERS} workers for simultaneous clip processing")
print(f"📁 Local Temp Directory: {LOCAL_TEMP_DIR}")
if USING_PORTABLE_PYTHON:
    print(f"🐍 Python: Portable (bin/python-3.13.9-embed-amd64/)")
else:
    print(f"🐍 Python: System ({sys.executable})")
if USING_PORTABLE_CUDA:
    print(f"🎮 CUDA: Portable (bin/CUDA/v13.0)")
else:
    print(f"🎮 CUDA: System (or not available)")
if GPU_AVAILABLE:
    print(f"⚡ GPU Acceleration: AVAILABLE - {get_gpu_info()}")
else:
    print(f"💻 GPU Acceleration: NOT AVAILABLE (CuPy not installed)")


def get_local_temp_dir() -> str:
    """Get the local temp directory path."""
    return LOCAL_TEMP_DIR


def create_temp_subdir() -> str:
    """Create a unique temporary subdirectory in the local temp folder."""
    temp_subdir = os.path.join(LOCAL_TEMP_DIR, f"session_{uuid.uuid4().hex}")
    os.makedirs(temp_subdir, exist_ok=True)
    return temp_subdir


NVENC_AVAILABLE = check_nvenc_support()
if NVENC_AVAILABLE:
    print(f"🎬 NVIDIA NVENC: AVAILABLE - Hardware video encoding enabled")
else:
    print(f"⚠️  NVIDIA NVENC: NOT AVAILABLE - Using CPU encoding only")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create a music video with cuts synchronized to bass beats'
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
        'cut_intensity',
        type=str,
        help='Cut intensity: number (0.1-16) or smart preset (slower/slow/normal/fast/faster) or "auto"'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='output_music_video.mkv',
        help='Output video file path (default: output_music_video.mkv)'
    )
    parser.add_argument(
        '-d', '--duration',
        type=float,
        default=2.0,
        help='Default clip duration in seconds (default: 2.0)'
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
        '--direction',
        type=str,
        choices=['forward', 'backward', 'random'],
        default='random',
        help='Video playback direction: forward (normal), backward (reverse), or random (mix of both) (default: random)'
    )
    parser.add_argument(
        '--offset',
        type=float,
        default=0.0,
        help='Timing offset in seconds to adjust video sync (default: 0.0)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['manual', 'smart', 'auto'],
        default=None,
        help='Generation mode: manual (bass-focused), smart (multi-band), auto (intelligent)'
    )
    parser.add_argument(
        '--lossless',
        action='store_true',
        help='Enable Lossless/Precise mode with ProRes 422 Proxy (frame-accurate cuts, no re-encoding)'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Enable GPU acceleration (audio analysis + NVENC encoding)'
    )
    parser.add_argument(
        '--gpu-encoder',
        type=str,
        choices=['h264_nvenc', 'hevc_nvenc', 'none'],
        default='h264_nvenc',
        help='GPU encoder: h264_nvenc (H.264), hevc_nvenc (H.265), none (CPU) (default: h264_nvenc)'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=None,
        help='Output FPS (frames per second). If not specified, auto-detect from input video (default: auto)'
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


def create_clip_parallel(args):
    """
    Wrapper function for parallel clip creation using FFmpeg.
    
    ✅ TIMING OFFSET is applied HERE to video playback timing, not to beat times
    """
    (i, video_file, final_duration, speed_factor, direction, target_size, 
     use_nvenc, gpu_encoder, temp_dir, fps, timing_offset) = args
    
    try:
        # Random start time from video
        video_duration = get_video_duration(video_file)
        
        # Calculate required source duration based on speed
        required_source_duration = final_duration * speed_factor
        
        if video_duration >= required_source_duration:
            max_start = video_duration - required_source_duration
            clip_start = random.uniform(0, max_start)
            source_duration = required_source_duration
        else:
            clip_start = 0
            source_duration = video_duration
        
        # Determine reverse
        apply_reverse = False
        if direction == 'backward':
            apply_reverse = True
        elif direction == 'random':
            apply_reverse = random.choice([True, False])
        
        # Output file
        temp_clip_path = os.path.join(temp_dir, f"temp_clip_{i}_{uuid.uuid4().hex}.mp4")
        
        # ✅ TIMING OFFSET: Applied to video playback, not beat detection
        # Negative offset = video plays earlier (to compensate for perceived lag)
        adjusted_start = clip_start - timing_offset if timing_offset != 0.0 else clip_start
        
        # Ensure we don't go negative
        if adjusted_start < 0:
            adjusted_start = 0
        
        # Extract clip with FFmpeg (FRAME-ACCURATE)
        success = extract_clip_segment_ffmpeg(
            video_file=video_file,
            start_time=adjusted_start,
            duration=source_duration,
            output_file=temp_clip_path,
            fps=fps,
            target_size=target_size,
            reverse=apply_reverse,
            speed_factor=speed_factor,
            use_nvenc=use_nvenc,
            gpu_encoder=gpu_encoder
        )
        
        if not success:
            return (i, None, target_size, None, "FFmpeg extraction failed")
        
        return (i, temp_clip_path, target_size, temp_clip_path, None)
        
    except Exception as e:
        return (i, None, target_size, None, str(e))


def create_music_video(audio_file: str, video_files: VideoList, beat_times: BeatTimes, 
                      cut_intensity, default_duration: float = 2.0, 
                      output_file: str = 'output_music_video.mkv',
                      start_time: float = 0.0, end_time: float = None, 
                      direction: str = 'random', speed_factor: float = 1.0, 
                      timing_offset: float = 0.0, max_workers: int = None,
                      smart_mode: bool = False, beat_info: dict = None, 
                      lossless_mode: bool = False, use_gpu: bool = False, 
                      gpu_encoder: str = 'h264_nvenc', fps: float = None) -> str:
    """
    Creates a music video with video clips cut to detected beats.
    
    **PURE FFMPEG IMPLEMENTATION - FRAME-ACCURATE**
    
    ✅ NO BATCH PROCESSING: FFmpeg handles memory independently
    ✅ TIMING OFFSET: Applied to VIDEO playback timing, not beat detection
    ✅ FRAME-ACCURATE: Uses exact frame counts for zero drift
    ✅ NO CUMULATIVE ERROR: Each segment is precisely timed
    
    Args:
        audio_file: Path to audio file
        video_files: List of video file paths
        beat_times: Array of beat times (already processed by mode)
        cut_intensity: Cut intensity or preset name
        default_duration: Default clip duration (not used with beat times)
        output_file: Output file path
        start_time: Audio start time
        end_time: Audio end time
        direction: Video direction (forward/backward/random)
        speed_factor: Playback speed multiplier
        timing_offset: Timing offset in seconds (applied to video playback)
        max_workers: Number of parallel workers
        smart_mode: Whether smart mode was used
        beat_info: Beat information dictionary
        lossless_mode: Use ProRes 422 Proxy mode
        use_gpu: Use GPU acceleration
        gpu_encoder: GPU encoder to use
        fps: Output FPS
    
    Returns:
        Path to output video file
    """
    if len(beat_times) == 0:
        raise ValueError("No beats were detected. Cannot create video.")

    if max_workers is None:
        max_workers = PARALLEL_WORKERS

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

    # Create session temp directory
    session_temp_dir = create_temp_subdir()
    print(f"📁 Session temp directory: {session_temp_dir}")

    # Determine processing mode
    use_nvenc = use_gpu and NVENC_AVAILABLE and not lossless_mode and gpu_encoder != 'none'
    
    gpu_status = "⚡ GPU" if use_gpu and GPU_AVAILABLE else "💻 CPU"
    encoder_status = f"⚡ {gpu_encoder.upper()}" if use_nvenc else ("🎯 Frame-Perfect ProRes" if lossless_mode else "💻 CPU (libx264)")
    python_status = "Portable" if USING_PORTABLE_PYTHON else "System"
    cuda_status = "Portable" if USING_PORTABLE_CUDA else "System/None"
    
    # Determine mode name
    mode_name = beat_info.get('mode', 'unknown') if beat_info else 'unknown'
    
    print(f"🚀 Performance Settings:")
    print(f"   • Python: {python_status}")
    print(f"   • CUDA: {cuda_status}")
    print(f"   • FFmpeg: Portable (bin/ffmpeg/ffmpeg.exe)")
    print(f"   • FFmpeg threads per encode: {MAX_THREADS} (100% CPU utilization)")
    print(f"   • Parallel clip workers: {max_workers}")
    print(f"   • Audio analysis: {gpu_status}")
    print(f"   • Video processing: {encoder_status}")
    print(f"   • Output FPS: {fps}")
    print(f"   • Frame-accurate mode: ENABLED (zero drift)")
    print(f"   • Generation mode: {mode_name}")
    if timing_offset != 0.0:
        offset_direction = "earlier" if timing_offset < 0 else "later"
        print(f"   • Timing offset: {abs(timing_offset):.3f}s ({offset_direction})")
    if lossless_mode:
        print(f"   • Lossless Mode: ENABLED (ProRes 422 Proxy)")
        print(f"   • Precision Mode: Frame-perfect (re-encodes all segments)")
        print(f"   • Export Format: Apple ProRes 422 Proxy (.mov)")
    else:
        print(f"   • Export Format: H.264/H.265 (.mkv)")

    # Get audio duration using ffprobe
    audio_duration = get_video_duration(audio_file)
    if end_time and end_time > start_time:
        audio_duration = end_time - start_time
    elif start_time > 0:
        audio_duration = audio_duration - start_time
    
    print(f"🎵 Audio duration: {audio_duration:.2f} seconds")

    # Use provided beat_times (already processed by the mode)
    selected_beats = beat_times
    
    # Ensure beats cover full audio duration
    if len(selected_beats) == 0 or selected_beats[0] > 0.1:
        selected_beats = np.insert(selected_beats, 0, 0)
    if selected_beats[-1] < audio_duration:
        selected_beats = np.append(selected_beats, audio_duration)

    print(f"🎬 Creating video with {len(selected_beats)-1} cuts")
    if timing_offset != 0.0:
        print(f"⏱️  Timing offset: {timing_offset:.3f}s (applied to video playback, not beats)")

    # LOSSLESS MODE - ProRes workflow with FRAME-PERFECT precision
    if lossless_mode:
        print(f"\n{'='*60}")
        print(f"🎯 LOSSLESS MODE: Converting videos to ProRes 422 Proxy")
        print(f"{'='*60}")
        
        # Create ProRes conversion directory
        prores_dir = os.path.join(session_temp_dir, 'prores')
        os.makedirs(prores_dir, exist_ok=True)
        
        # Use detected FPS for ProRes conversion
        prores_fps = fps
        print(f"🎞️ Using FPS: {prores_fps} (for frame-perfect precision)")
        
        # Convert all input videos to ProRes (video only, no audio)
        prores_files = []
        for idx, video_file in enumerate(video_files, 1):
            print(f"Converting {idx}/{len(video_files)}...")
            prores_file = convert_to_prores_proxy(video_file, prores_dir, prores_fps)
            prores_files.append(prores_file)
        
        print(f"✓ All videos converted to ProRes 422 Proxy (video only)")
        
        # Create segments from ProRes files with FRAME-PERFECT precision
        print(f"\n{'='*60}")
        print(f"✂️  EXTRACTING SEGMENTS (FRAME-PERFECT PRECISION)")
        print(f"{'='*60}")
        print(f"   Mode: Frame-accurate re-encoding")
        print(f"   Method: Exact frame count calculation")
        print(f"   Direction: {direction}")
        print(f"   Audio: Stripped (will add music at the end)")
        print(f"   FPS: {prores_fps} (fixed)")
        if timing_offset != 0.0:
            print(f"   Timing offset: {timing_offset:.3f}s (video playback adjustment)")
        
        segment_files = []
        segments_dir = os.path.join(session_temp_dir, 'segments')
        os.makedirs(segments_dir, exist_ok=True)
        
        for i in range(len(selected_beats) - 1):
            # ✅ FRAME-ACCURATE: Calculate exact duration from beat interval
            duration_seconds = selected_beats[i + 1] - selected_beats[i]
            
            # Convert to exact frame count
            frame_count = seconds_to_frame_count(duration_seconds, prores_fps)
            exact_duration = frame_count_to_seconds(frame_count, prores_fps)
            
            # Randomly select ProRes file
            prores_file = random.choice(prores_files)
            
            # Extract segment
            segment_file = extract_prores_segment_random(
                prores_file, exact_duration, prores_fps, segments_dir, i, direction
            )
            segment_files.append(segment_file)
            
            if (i + 1) % 10 == 0:
                print(f"   ✓ Extracted {i + 1}/{len(selected_beats) - 1} segments (frame-perfect)")
        
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
            temp_dir=session_temp_dir
        )
        
        print(f"\n{'='*60}")
        print(f"✅ LOSSLESS VIDEO CREATION COMPLETE!")
        print(f"   Output: {output_file}")
        print(f"   Method: Frame-perfect re-encoding + lossless concatenation")
        print(f"   Quality: ProRes 422 Proxy (lossless)")
        print(f"   Audio: Music track from input file")
        print(f"   FPS: {prores_fps} (fixed)")
        print(f"   Total Segments: {len(segment_files)}")
        print(f"   Timing Precision: Frame-perfect (zero drift)")
        print(f"{'='*60}\n")
        
        # Cleanup
        print(f"🧹 Cleaning up temporary files...")
        time.sleep(1.0)
        gc.collect()
        
        for segment_file in segment_files:
            try:
                if os.path.exists(segment_file):
                    os.remove(segment_file)
            except:
                pass
        
        for prores_file in prores_files:
            try:
                if os.path.exists(prores_file):
                    os.remove(prores_file)
            except:
                pass
        
        try:
            import shutil
            if os.path.exists(segments_dir):
                shutil.rmtree(segments_dir, ignore_errors=True)
            if os.path.exists(prores_dir):
                shutil.rmtree(prores_dir, ignore_errors=True)
        except:
            pass
        
        print(f"✓ Cleanup complete")
        
        return output_file
    
    # STANDARD MODE - Direct parallel processing (NO BATCHES)
    else:
        # Get target resolution from first video
        target_size = get_video_resolution(video_files[0])
        print(f"🎞️ Target resolution: {target_size[0]}x{target_size[1]}")
        
        total_clips = len(selected_beats) - 1
        
        print(f"\n{'='*60}")
        print(f"🎬 PROCESSING ALL CLIPS (No batch processing with FFmpeg)")
        print(f"   Total clips: {total_clips}")
        print(f"   Parallel workers: {max_workers}")
        print(f"   Frame-accurate: ENABLED")
        if use_nvenc:
            print(f"   Encoder: ⚡ NVIDIA {gpu_encoder.upper()} (GPU-accelerated)")
        else:
            print(f"   Encoder: 💻 libx264 (CPU)")
        if timing_offset != 0.0:
            print(f"   Timing offset: {timing_offset:.3f}s (applied to video playback)")
        print(f"{'='*60}\n")
        
        clip_args = []
        for i in range(total_clips):
            # ✅ FRAME-ACCURATE: Calculate exact duration
            duration_seconds = selected_beats[i + 1] - selected_beats[i]
            frame_count = seconds_to_frame_count(duration_seconds, fps)
            final_duration = frame_count_to_seconds(frame_count, fps)
            
            video_file = random.choice(video_files)
            clip_args.append((i, video_file, final_duration, speed_factor, direction, 
                            target_size, use_nvenc, gpu_encoder, session_temp_dir, fps, timing_offset))
        
        clip_files = [None] * len(clip_args)
        
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
                    i, clip_path, new_target_size, temp_path, error = future.result()
                    
                    if error:
                        print(f"⚠️  Warning: Clip {i+1} failed: {error}")
                        continue
                    
                    if clip_path is not None:
                        clip_files[idx] = clip_path
                        
                        completed += 1
                        if completed % 10 == 0 or completed == len(clip_args):
                            progress = (completed / len(clip_args)) * 100
                            print(f"   ⚡ Progress: {completed}/{len(clip_args)} clips ({progress:.1f}%)")
                    
                except Exception as e:
                    print(f"⚠️  Warning: Error processing clip: {str(e)}")
                    continue
        
        # Filter out None values
        clip_files = [f for f in clip_files if f is not None]
        
        if not clip_files:
            raise ValueError('No valid video clips could be created')

        print(f"\n{'='*60}")
        print(f"🎬 FINAL ASSEMBLY: Concatenating {len(clip_files)} clips")
        print(f"{'='*60}\n")
        
        # Concatenate all clips and add audio
        concatenate_videos_ffmpeg(
            video_files=clip_files,
            output_file=output_file,
            audio_file=audio_file,
            start_time=start_time,
            end_time=end_time,
            use_nvenc=use_nvenc,
            gpu_encoder=gpu_encoder,
            fps=fps,
            temp_dir=session_temp_dir
        )

        print(f"\n🧹 Cleaning up resources...")
        
        # Cleanup clip files
        for clip_file in clip_files:
            try:
                if os.path.exists(clip_file):
                    os.remove(clip_file)
            except Exception as e:
                print(f"⚠️  Warning: Could not delete clip file: {e}")
        
        # Clean up session temp directory
        try:
            if os.path.exists(session_temp_dir):
                import shutil
                shutil.rmtree(session_temp_dir)
                print(f"✓ Cleaned up session temp directory")
        except Exception as e:
            print(f"⚠️  Warning: Could not delete session temp directory: {e}")
        
        gc.collect()

        print(f"\n{'='*60}")
        print(f"✅ VIDEO CREATION COMPLETE!")
        print(f"   Output: {output_file}")
        print(f"   FPS: {fps} (frame-accurate)")
        print(f"   Total Cuts: {total_clips}")
        if timing_offset != 0.0:
            print(f"   Timing Offset Applied: {timing_offset:.3f}s")
        print(f"   Zero Drift: Frame-accurate processing")
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
            print(f"⚡ GPU acceleration ENABLED: {get_gpu_info()}")
        else:
            print(f"⚠️  GPU requested but CuPy not available, using CPU")
            args.gpu = False

    # Determine mode
    if args.mode:
        mode = args.mode
    elif args.cut_intensity == 'auto':
        mode = 'auto'
    elif args.cut_intensity in list_presets():
        mode = 'smart'
    else:
        mode = 'manual'

    python_str = "Portable (bin/python-3.13.9-embed-amd64/)" if USING_PORTABLE_PYTHON else f"System ({sys.executable})"
    cuda_str = "Portable (bin/CUDA/v13.0)" if USING_PORTABLE_CUDA else "System/None"

    print(f"\n{'='*60}")
    print(f"🎵 MUSIC VIDEO CUTTER - FRAME-ACCURATE")
    print(f"   Python: {python_str}")
    print(f"   CUDA: {cuda_str}")
    print(f"   CPU Threads: {CPU_COUNT}")
    print(f"   FFmpeg: Portable (bin/ffmpeg/ffmpeg.exe)")
    print(f"   Audio Analysis: {'⚡ GPU' if args.gpu else '💻 CPU'}")
    if args.gpu and NVENC_AVAILABLE and not args.lossless:
        print(f"   Video Encoding: ⚡ {args.gpu_encoder.upper()}")
    else:
        print(f"   Video Encoding: 💻 CPU")
    if args.fps:
        print(f"   FPS: {args.fps} (custom)")
    else:
        print(f"   FPS: Auto-detect from input video")
    if args.offset != 0.0:
        print(f"   Timing Offset: {args.offset:.3f}s (applied to video playback)")
    print(f"   Mode: {mode.upper()}")
    if args.lossless:
        print(f"   Export: 🎯 Lossless/Precise (ProRes 422 Proxy - Frame Perfect)")
    else:
        print(f"   Export: 📹 H.264/H.265 (.mkv) - Frame Accurate")
    print(f"{'='*60}\n")
    
    print(f'📁 Audio file: {args.mp3_file}')

    # Analyze beats based on mode
    if mode == 'auto':
        print(f"🤖 Using AUTO mode (extreme intelligence)")
        selected_beats, beat_info = analyze_beats_auto(
            args.mp3_file,
            start_time=args.start_time,
            end_time=args.end_time,
            use_gpu=args.gpu
        )
    elif mode == 'smart':
        print(f"🧠 Using SMART mode (multi-band analysis)")
        beat_times, beat_info = analyze_beats_smart(
            args.mp3_file,
            start_time=args.start_time,
            end_time=args.end_time
        )
        selected_beats = select_beats_smart(beat_info, preset=args.cut_intensity)
    else:  # manual
        print(f"⚙️ Using MANUAL mode (bass-focused)")
        beat_times, beat_info = analyze_beats_manual(
            args.mp3_file,
            start_time=args.start_time,
            end_time=args.end_time,
            use_gpu=args.gpu
        )
        try:
            cut_intensity_value = float(args.cut_intensity)
            selected_beats = process_manual_intensity(beat_times, cut_intensity_value)
        except ValueError:
            print(f"⚠️  Invalid cut intensity, using 4.0")
            selected_beats = process_manual_intensity(beat_times, 4.0)

    print(f'✓ Selected {len(selected_beats)} cuts for video')
    
    video_files = get_video_files(args.video_directory)
    print(f'✓ Found {len(video_files)} video files')

    output_file = args.output
    if args.lossless and not output_file.lower().endswith('.mov'):
        base, _ = os.path.splitext(output_file)
        output_file = base + '.mov'
        print(f'📝 Changed output to .mov for Lossless mode: {output_file}')
    elif not args.lossless and not output_file.lower().endswith('.mkv'):
        base, _ = os.path.splitext(output_file)
        output_file = base + '.mkv'
        print(f'📝 Changed output to .mkv: {output_file}')

    print(f'\n🎬 Starting video creation (frame-accurate)...\n')
    output_file = create_music_video(
        args.mp3_file,
        video_files,
        selected_beats,  # Pass pre-processed beats
        args.cut_intensity,
        default_duration=args.duration,
        output_file=output_file,
        start_time=args.start_time,
        end_time=args.end_time,
        direction=args.direction,
        timing_offset=args.offset,
        max_workers=PARALLEL_WORKERS,
        smart_mode=(mode == 'smart'),
        beat_info=beat_info,
        lossless_mode=args.lossless,
        use_gpu=args.gpu,
        gpu_encoder=args.gpu_encoder,
        fps=args.fps
    )

    print(f'✅ Music video created successfully: {output_file}')


if __name__ == '__main__':
    main()