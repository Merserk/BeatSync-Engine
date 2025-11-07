#!/usr/bin/env python3
import os
import sys

# Determine script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add script directory to Python path for module imports
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

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
    
    # Set library path for Linux/Unix compatibility (not needed for Windows but doesn't hurt)
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

# NOW import other modules (after CUDA environment is set)
import gradio as gr
import librosa
import tempfile
import shutil
import datetime
import gc
import multiprocessing
import time
import subprocess
from typing import TypeAlias, Tuple, Any, Dict, List

# Import FFmpeg processing module
from ffmpeg_processing import check_nvenc_support, get_video_fps, FFMPEG_FOUND, FFMPEG_PATH

# Check NVENC support
NVENC_AVAILABLE = check_nvenc_support()

from video_processor import create_music_video, CPU_COUNT, MAX_THREADS, PARALLEL_WORKERS, GPU_AVAILABLE, get_local_temp_dir, create_temp_subdir

# Import mode modules
from manual_mode import analyze_beats_manual, process_manual_intensity
from smart_mode import analyze_beats_smart, select_beats_smart, list_presets, get_preset_info, set_gpu_mode, is_gpu_available, get_gpu_info
from auto_mode import analyze_beats_auto

# Import UI content
from ui_content import *

# Set Gradio to use local temp directory instead of system temp
GRADIO_TEMP_DIR = os.path.join(SCRIPT_DIR, 'temp', 'gradio_uploads')
os.makedirs(GRADIO_TEMP_DIR, exist_ok=True)

# Set environment variable for Gradio (but don't override tempfile.tempdir globally)
os.environ['GRADIO_TEMP_DIR'] = GRADIO_TEMP_DIR

# Prepare status strings for startup
python_str_startup = "Portable" if USING_PORTABLE_PYTHON else "System"
cuda_str_startup = "Portable" if USING_PORTABLE_CUDA else "System/None"
ffmpeg_str_startup = "✅ Portable" if FFMPEG_FOUND else "⚠️  System"

print(get_startup_header(
    CPU_COUNT, MAX_THREADS, PARALLEL_WORKERS,
    f"✅ Portable ({PORTABLE_PYTHON_EXE})" if USING_PORTABLE_PYTHON else f"System ({sys.executable})",
    "✅ Portable (bin/CUDA/v13.0)" if USING_PORTABLE_CUDA else "System (or not available)",
    librosa.__version__,
    f"✅ Portable (bin/ffmpeg/ffmpeg.exe)" if FFMPEG_FOUND else "⚠️  System FFmpeg (portable not found)",
    is_gpu_available(),
    get_gpu_info(),
    NVENC_AVAILABLE
))

# Print temp directory locations
local_temp = get_local_temp_dir()
print(f"   Temp Directory: {local_temp}")
print(f"   Gradio Uploads: {GRADIO_TEMP_DIR}")
print(f"{CONSOLE_SEPARATOR}\n")

VideoFilesInput : TypeAlias = List[str]
StatusResult : TypeAlias = Tuple[str, str, Dict]


def copy_to_local_temp(file_path: str, session_temp: str) -> str:
    """Copy uploaded file from Gradio temp to local session temp."""
    if not file_path or not os.path.exists(file_path):
        return None
    
    filename = os.path.basename(file_path)
    local_path = os.path.join(session_temp, filename)
    
    # Only copy if not already in session temp
    if not os.path.exists(local_path):
        try:
            shutil.copy2(file_path, local_path)
            print(f"   📥 Copied: {filename}")
        except Exception as e:
            print(f"   ⚠️  Error copying {filename}: {e}")
            return None
    
    return local_path


def process_video(audio_file: str, video_files: VideoFilesInput, 
                 generation_mode: str, cut_intensity: float, smart_preset: str,
                 output_filename: str, direction: str, playback_speed_str: str, 
                 timing_offset: float, parallel_workers: int, processing_mode: str,
                 custom_fps: float, session_state: dict) -> StatusResult:
    try:
        # Get or create session directory
        session_dir = session_state.get('session_dir')
        if not session_dir or not os.path.exists(session_dir):
            session_dir = create_temp_subdir()
            session_state['session_dir'] = session_dir
            session_state['original_audio_path'] = None
            session_state['original_video_paths'] = []
            print(f"✨ New session started. Temp dir: {session_dir}")

        # Handle audio file
        if audio_file:
            if audio_file != session_state.get('original_audio_path'):
                print(f"📥 Processing new audio file...")
                local_audio_path = copy_to_local_temp(audio_file, session_dir)
                if local_audio_path:
                    session_state['local_audio_path'] = local_audio_path
                    session_state['original_audio_path'] = audio_file
                    print(f"   ✓ Audio ready: {os.path.basename(local_audio_path)}")
                else:
                    return None, '❌ Error: Could not copy audio file', session_state
            else:
                local_audio_path = session_state.get('local_audio_path')
                print(f"♻️  Reusing existing audio file")
        else:
            return None, '❌ Error: No audio file uploaded', session_state

        # Handle video files
        if video_files:
            if video_files != session_state.get('original_video_paths'):
                print(f"📥 Processing {len(video_files)} video file(s)...")
                local_video_paths = []
                for vf in video_files:
                    if vf:
                        local_path = copy_to_local_temp(vf, session_dir)
                        if local_path:
                            local_video_paths.append(local_path)
                
                if local_video_paths:
                    session_state['local_video_paths'] = local_video_paths
                    session_state['original_video_paths'] = video_files
                    print(f"   ✓ {len(local_video_paths)} video(s) ready")
                else:
                    return None, '❌ Error: Could not copy video files', session_state
            else:
                local_video_paths = session_state.get('local_video_paths')
                print(f"♻️  Reusing existing video files")
        else:
            return None, '❌ Error: No valid video files uploaded', session_state

        # Verify files exist
        if not local_audio_path or not os.path.exists(local_audio_path):
             return None, f"❌ Error: Audio file is missing from session directory.", session_state
        if not local_video_paths or not all(p and os.path.exists(p) for p in local_video_paths):
             return None, f"❌ Error: Video files are missing from session directory.", session_state
        
        # Set GPU mode
        use_gpu = is_gpu_available()
        set_gpu_mode(use_gpu)
        
        # Determine processing mode
        is_prores = processing_mode == 'prores_proxy'
        use_nvenc = (processing_mode in ['h264_nvenc', 'hevc_nvenc']) and NVENC_AVAILABLE
        gpu_encoder = processing_mode if use_nvenc else 'none'
        
        # Determine generation mode
        if generation_mode == 'manual': 
            mode_str, smart_mode = "⚙️ MANUAL MODE", False
        elif generation_mode == 'smart': 
            mode_str, smart_mode = "🧠 SMART MODE", True
        else: 
            mode_str, smart_mode = "🤖 AUTO MODE", False
        
        # Codec and encoder strings
        if is_prores: 
            codec_str, encoder_str = "🎯 ProRes 422 Proxy", "🎯 Lossless Concatenation"
        elif use_nvenc: 
            codec_str, encoder_str = f"⚡ NVIDIA {gpu_encoder.upper()}", f"⚡ {gpu_encoder.upper()}"
        else: 
            codec_str, encoder_str = "💻 CPU H.264", "💻 CPU (libx264)"
        
        accel_str = "⚡ GPU ACCELERATED" if use_gpu else "💻 CPU MODE"
        python_str = "Portable" if USING_PORTABLE_PYTHON else "System"
        cuda_str = "Portable" if USING_PORTABLE_CUDA else "System/None"

        # Determine FPS
        if custom_fps is not None and custom_fps > 0:
            output_fps = custom_fps
        else:
            output_fps = get_video_fps(local_video_paths[0])
            
        # Prepare output paths
        output_folder = os.path.join(SCRIPT_DIR, 'output')
        os.makedirs(output_folder, exist_ok=True)
        name, _ = os.path.splitext(output_filename)
        ext = '.mov' if is_prores else '.mp4'
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}{ext}"
        output_path = os.path.join(output_folder, filename)
        temp_output = os.path.join(session_dir, filename)

        speed_factor = {'Half Speed': 0.5, 'Double Speed': 2.0}.get(playback_speed_str, 1.0)

        print(f"\n{CONSOLE_SEPARATOR}")
        print(f"🎵 BEAT ANALYSIS - {mode_str} ({accel_str})")
        print(f"{CONSOLE_SEPARATOR}")

        # Analyze beats based on mode
        if generation_mode == 'manual':
            beat_times, beat_info = analyze_beats_manual(local_audio_path, use_gpu=use_gpu)
            selected_beats = process_manual_intensity(beat_times, cut_intensity)
            intensity_param = cut_intensity
        elif generation_mode == 'smart':
            beat_times, beat_info = analyze_beats_smart(local_audio_path)
            selected_beats = select_beats_smart(beat_info, preset=smart_preset)
            intensity_param = smart_preset
        else: # auto
            selected_beats, beat_info = analyze_beats_auto(local_audio_path, use_gpu=use_gpu)
            intensity_param = 'auto'
            beat_times = beat_info.get('times', selected_beats)

        print(f"\n{CONSOLE_SEPARATOR}")
        print(f"🎬 VIDEO CREATION")
        print(f"{CONSOLE_SEPARATOR}")

        # Create video
        result_path = create_music_video(
            local_audio_path, local_video_paths, selected_beats, intensity_param,
            output_file=temp_output, direction=direction, speed_factor=speed_factor,
            timing_offset=timing_offset, max_workers=parallel_workers,
            smart_mode=smart_mode, beat_info=beat_info, lossless_mode=is_prores,
            use_gpu=use_gpu, gpu_encoder=gpu_encoder, fps=output_fps
        )

        # Move to output folder
        shutil.move(result_path, output_path)

        # Create preview for ProRes if needed
        preview_path = output_path
        if is_prores:
            print(f"🎬 Generating H.264 preview for ProRes file...")
            preview_filename = f"{name}_{timestamp}_preview.mp4"
            preview_path = os.path.join(session_dir, preview_filename)
            preview_cmd = [FFMPEG_PATH]
            if NVENC_AVAILABLE:
                preview_cmd.extend(['-hwaccel', 'cuda', '-c:v', 'h264_nvenc', '-preset', 'p5', '-cq', '23'])
            else:
                preview_cmd.extend(['-hwaccel', 'auto', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23'])
            preview_cmd.extend(['-i', output_path, '-pix_fmt', 'yuv420p', '-y', preview_path])
            subprocess.run(preview_cmd, capture_output=True, text=True, timeout=180)
            print(f"   ✓ Preview created for Gradio.")

        # Generate status message based on mode
        gpu_info = f"⚡ GPU: {get_gpu_info()}" if use_gpu else "💻 CPU"
        fps_info = f"{output_fps:.2f} FPS (custom)" if custom_fps else f"{output_fps:.2f} FPS (auto-detected)"
        audio_info = "PCM 24-bit (48kHz)"
        
        if is_prores:
            codec_info = "ProRes 422 Proxy (.mov) - Lossless"
            encoder_info = "🎯 Lossless Concatenation"
        elif use_nvenc:
            codec_info = f"{gpu_encoder.upper()} (.mp4)"
            encoder_info = f"⚡ {gpu_encoder.upper()}"
        else:
            codec_info = "H.264 (.mp4)"
            encoder_info = "💻 libx264"

        if generation_mode == 'smart':
            preset_info = get_preset_info(smart_preset)
            total_cuts = len(selected_beats) - 1
            
            status_msg = get_success_message_smart(
                smart_preset, preset_info, len(beat_times),
                beat_info.get('tempo', 120), total_cuts,
                python_str, cuda_str, MAX_THREADS, CPU_COUNT,
                parallel_workers, gpu_info, encoder_info,
                codec_info, fps_info, filename, audio_info
            )
        elif generation_mode == 'auto':
            total_cuts = len(selected_beats) - 1
            sections_info = beat_info.get('selection_info', [])
            
            status_msg = get_success_message_auto(
                total_cuts, len(beat_times),
                beat_info.get('tempo', 120), sections_info,
                python_str, cuda_str, MAX_THREADS, CPU_COUNT,
                parallel_workers, gpu_info, encoder_info,
                codec_info, fps_info, filename, audio_info
            )
        else:  # manual mode
            if cut_intensity < 1.0:
                subdivisions = int(1.0 / cut_intensity)
                total_cuts = len(selected_beats) - 1
                status_msg = get_success_message_manual_subdivided(
                    total_cuts, subdivisions, len(beat_times),
                    beat_info.get('tempo', 120), cut_intensity,
                    python_str, cuda_str, MAX_THREADS, CPU_COUNT,
                    parallel_workers, gpu_info, encoder_info,
                    codec_info, fps_info, filename, audio_info
                )
            else:
                beats_used = len(selected_beats) - 1
                cut_intensity_int = int(cut_intensity)
                status_msg = get_success_message_manual_skipped(
                    beats_used, cut_intensity_int, len(beat_times),
                    beat_info.get('tempo', 120), cut_intensity,
                    python_str, cuda_str, MAX_THREADS, CPU_COUNT,
                    parallel_workers, gpu_info, encoder_info,
                    codec_info, fps_info, filename, audio_info
                )

        print(f"\n{CONSOLE_SEPARATOR}")
        print(f"✅ PROCESS COMPLETE")
        print(f"{CONSOLE_SEPARATOR}\n")

        # Return preview path for display, keep session_state intact
        return preview_path, status_msg, session_state

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        import traceback
        traceback.print_exc()
        return None, error_msg, session_state


def cleanup_on_startup():
    """Clean up old session folders on script start only. Leave Gradio temp alone."""
    # Only clean up session directories, not Gradio uploads
    session_temp_base = get_local_temp_dir()
    try:
        if os.path.exists(session_temp_base):
            print(f"🧹 Cleaning up old session directories...")
            for item in os.listdir(session_temp_base):
                item_path = os.path.join(session_temp_base, item)
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                        print(f"   ✓ Removed: {item}")
                except Exception as e:
                    print(f"   ⚠️  Could not remove {item}: {e}")
            print(f"   ✓ Old sessions cleared.")
        else:
            os.makedirs(session_temp_base, exist_ok=True)
            print(f"   ✓ Created session temp directory")
    except Exception as e:
        print(f"   ⚠️  Warning: Could not clean up sessions: {e}")
        try:
            os.makedirs(session_temp_base, exist_ok=True)
        except:
            pass


def create_ui() -> gr.Blocks:
    # These definitions are needed within the function's scope
    python_status = "✅ Portable (bin/python-3.13.9-embed-amd64/)" if USING_PORTABLE_PYTHON else "⚠️  System Python"
    cuda_status = "✅ Portable (bin/CUDA/v13.0)" if USING_PORTABLE_CUDA else "⚠️  System CUDA (or not available)"
    ffmpeg_status = "✅ Portable (bin/ffmpeg/)" if FFMPEG_FOUND else "⚠️  System FFmpeg"
    
    app = gr.Blocks(title='BeatSync Engine', theme=gr.themes.Soft())
    with app:
        session_state = gr.State({})

        gr.Markdown(f"# {UI_TITLE}")
        gr.Markdown(UI_MAIN_DESCRIPTION)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('### 📁 Input Files')
                audio_input = gr.File(label=LABEL_AUDIO_FILE, file_types=['.mp3', '.wav', '.flac'], type='filepath')
                video_input = gr.File(label=LABEL_VIDEO_FILES, file_count='multiple', file_types=['.mp4', '.mkv'], type='filepath')

                with gr.Group():
                    gr.Markdown('### 🎯 Generation Mode')
                    generation_mode = gr.Radio(
                        choices=[('🤖 Auto Mode (Recommended)', 'auto'), ('🧠 Smart Mode', 'smart'), ('⚙️ Manual Mode', 'manual')],
                        value='auto', label=LABEL_GENERATION_MODE, info=INFO_GENERATION_MODE
                    )
                    
                    auto_group = gr.Group(visible=True)
                    with auto_group:
                        gr.Markdown(AUTO_MODE_DESCRIPTION)
                    
                    smart_group = gr.Group(visible=False)
                    with smart_group:
                        gr.Markdown(SMART_MODE_DESCRIPTION)
                        smart_preset = gr.Radio(
                            choices=['slower', 'slow', 'normal', 'fast', 'faster'], value='normal',
                            label=LABEL_CUT_PRESET, info=INFO_CUT_PRESET
                        )
                    
                    manual_group = gr.Group(visible=False)
                    with manual_group:
                        gr.Markdown(MANUAL_MODE_DESCRIPTION)
                        cut_intensity = gr.Slider(
                            minimum=0.1, maximum=16, value=4, step=0.1,
                            label=LABEL_CUT_INTENSITY, info=INFO_CUT_INTENSITY
                        )

                with gr.Group():
                    gr.Markdown('### ⚙️ Video Settings')
                    direction = gr.Radio(choices=['forward', 'backward', 'random'], value='forward', label=LABEL_DIRECTION, info=INFO_DIRECTION)
                    playback_speed = gr.Radio(choices=['Normal Speed', 'Half Speed', 'Double Speed'], value='Normal Speed', label=LABEL_PLAYBACK_SPEED, info=INFO_PLAYBACK_SPEED)
                    timing_offset = gr.Slider(minimum=-0.5, maximum=0.5, value=0.0, step=0.01, label=LABEL_TIMING_OFFSET, info=INFO_TIMING_OFFSET)
                    custom_fps = gr.Number(label=LABEL_CUSTOM_FPS, value=None, precision=2, info=INFO_CUSTOM_FPS)

                with gr.Group():
                    gr.Markdown(f'### 🎬 Processing Mode')
                    if NVENC_AVAILABLE:
                        processing_mode = gr.Radio(choices=[('NVIDIA NVENC H.264', 'h264_nvenc'), ('NVIDIA NVENC HEVC (H.265)', 'hevc_nvenc'), ('CPU (H.264)', 'cpu'), ('ProRes 422 Proxy (Precise Mode)', 'prores_proxy')], value='h264_nvenc', label=LABEL_PROCESSING_MODE, info=get_processing_mode_info_nvenc())
                    else:
                        processing_mode = gr.Radio(choices=[('CPU (H.264)', 'cpu'), ('ProRes 422 Proxy (Precise Mode)', 'prores_proxy')], value='cpu', label=LABEL_PROCESSING_MODE, info=get_processing_mode_info_cpu())
                
                with gr.Group():
                    gr.Markdown(f'### ⚙️ Performance Settings')
                    parallel_workers = gr.Slider(minimum=1, maximum=min(16, max(CPU_COUNT // 2, 4)), value=PARALLEL_WORKERS, step=1, label=get_parallel_workers_label(PARALLEL_WORKERS), info=get_parallel_workers_info())

                with gr.Group():
                    gr.Markdown('### 📁 Output Settings')
                    output_filename = gr.Textbox(value='music_video.mp4', label=LABEL_OUTPUT_FILENAME, info=INFO_OUTPUT_FILENAME)

                process_btn = gr.Button('🎬 Create Music Video', variant='primary', size='lg')

            with gr.Column(scale=1):
                gr.Markdown('### 📺 Output')
                status_output = gr.Textbox(label='Status', interactive=False, value=get_ready_status(python_status, cuda_status, MAX_THREADS, CPU_COUNT, ffmpeg_status, is_gpu_available(), get_gpu_info(), NVENC_AVAILABLE), lines=16, max_lines=25)
                video_output = gr.Video(label='Generated Music Video', interactive=False)
                
        def toggle_mode(mode):
            return {
                manual_group: gr.update(visible=mode == 'manual'),
                smart_group: gr.update(visible=mode == 'smart'),
                auto_group: gr.update(visible=mode == 'auto')
            }

        generation_mode.change(
            fn=toggle_mode,
            inputs=[generation_mode],
            outputs=[manual_group, smart_group, auto_group]
        )

        process_btn.click(
            fn=process_video,
            inputs=[
                audio_input, video_input, generation_mode, cut_intensity,
                smart_preset, output_filename, direction, playback_speed,
                timing_offset, parallel_workers, processing_mode, custom_fps,
                session_state
            ],
            outputs=[video_output, status_output, session_state]
        )

    return app

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # Clean up old files only on startup
    cleanup_on_startup()
    
    print(f"🌐 Starting Gradio interface...")
    print(f"   URL: http://127.0.0.1:7860")
    print(f"   Session persistence: ENABLED")
    print(f"   Files kept until app restart")
    print(f"\n{CONSOLE_SEPARATOR}\n")
    
    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True
    )