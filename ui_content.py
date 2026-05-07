#!/usr/bin/env python3
"""
UI Content for BeatSync Engine 
Streamlined for unified mode selection (Manual/Smart/Auto)
"""

# ============================================================================
# SMART MODE PRESETS
# ============================================================================

SMART_PRESETS_CONFIG = {
    'slower': {
        'cut_frequency': 'every_4th_strong_beat',
        'kick_threshold': 70,
        'clap_threshold': 70,
        'min_interval': 1.5,
        'description': 'Cinematic - Every 4th strong beat (fewest cuts)'
    },
    'slow': {
        'cut_frequency': 'every_2nd_strong_beat',
        'kick_threshold': 60,
        'clap_threshold': 60,
        'min_interval': 0.8,
        'description': 'Relaxed - Every 2nd strong beat'
    },
    'normal': {
        'cut_frequency': 'every_strong_beat',
        'kick_threshold': 50,
        'clap_threshold': 50,
        'min_interval': 0.4,
        'description': 'Standard - Every strong kick or clap'
    },
    'fast': {
        'cut_frequency': 'all_beats_prioritize_strong',
        'kick_threshold': 40,
        'clap_threshold': 40,
        'min_interval': 0.25,
        'description': 'Energetic - All beats, prioritize strong'
    },
    'faster': {
        'cut_frequency': 'all_beats_plus_subdivisions',
        'kick_threshold': 30,
        'clap_threshold': 30,
        'min_interval': 0.15,
        'description': 'Hyper - All beats + subdivisions (most cuts)'
    }
}

# ============================================================================
# MAIN UI CONTENT
# ============================================================================

UI_TITLE = "🎵 BeatSync Engine"

UI_MAIN_DESCRIPTION = """Create music videos that cut to the beat. Upload audio and video clips 
to automatically generate a video synchronized with your music's rhythm."""

# ============================================================================
# SYSTEM STATUS
# ============================================================================

def get_system_performance_info(cpu_count, max_threads, parallel_workers, python_status, 
                                cuda_status, ffmpeg_status, gpu_status, gpu_device, nvenc_status):
    """System performance overview."""
    return f"""## 🚀 System Status
- **Python**: {python_status} | **CUDA**: {cuda_status}
- **CPU**: {cpu_count} threads | **FFmpeg**: {ffmpeg_status}
- **GPU**: {gpu_status}{gpu_device} | **NVENC**: {nvenc_status}
- **Parallel Processing**: {parallel_workers} workers | **Temp**: Local project folder"""

PORTABLE_SETUP_INFO = """## 📦 Portable Setup
Self-contained installation - Python 3.13.9, CUDA 13.0, and FFmpeg included. 
No system dependencies required."""

GPU_ACCELERATION_INFO = """## ⚡ GPU Acceleration
- **Audio Analysis**: 5-10x faster with GPU (CuPy + CUDA)
- **Video Encoding**: 2-3x faster with NVENC hardware encoder
- **Auto-Detection**: Uses GPU automatically when NVIDIA card detected"""

NVENC_BENEFITS_INFO = """## 🎬 NVENC Hardware Encoding
2-3x faster than CPU encoding with comparable quality. Automatically enabled when available."""

EXPORT_OPTIONS_INFO = """## 🎬 Export Modes
- **NVIDIA NVENC H.264**: GPU-accelerated H.264 (fast, .mkv)
- **NVIDIA NVENC HEVC**: GPU-accelerated H.265 (smaller files, .mkv)
- **CPU H.264**: Software encoding (.mkv)
- **ProRes 422 Proxy**: Frame-perfect lossless (.mov)"""

PRORES_MODE_INFO = """## 🎯 ProRes 422 Proxy Mode
Frame-perfect cuts with zero quality loss. Converts input to ProRes (I-frames only), 
then uses lossless concatenation. Larger files, perfect accuracy.

**How it works:**
1. Converts input videos to ProRes 422 Proxy (all I-frames)
2. Extracts segments with exact frame counts (frame-perfect)
3. Concatenates with stream copy (zero quality loss)
4. Adds music track with PCM audio

**Best for:** Professional editing, archival, maximum quality"""

# ============================================================================
# MODE DESCRIPTIONS
# ============================================================================

MANUAL_MODE_DESCRIPTION = """**⚙️ Manual Mode - Bass-Focused**
- Simple bass frequency detection (20-200 Hz)
- Direct cut intensity control
- Best for: Simple projects, consistent rhythm

**Controls:**
- `< 1.0` = MORE cuts (subdivide beats)
- `1.0` = Every beat (standard)
- `> 1.0` = FEWER cuts (skip beats)"""

SMART_MODE_DESCRIPTION = """**🧠 Smart Mode - Multi-Band Analysis**
- Kick (20-150 Hz) + Clap/Snare (150-4000 Hz) + Hi-hat (4000+ Hz)
- Intelligent frequency-based selection
- Best for: Professional results, complex music

**Presets:**
- **Slower**: Every 4th strong beat (cinematic)
- **Slow**: Every 2nd strong beat (relaxed)
- **Normal**: Every strong beat (standard)
- **Fast**: All beats with priority (energetic)
- **Faster**: All beats + subdivisions (hyper)"""

AUTO_MODE_DESCRIPTION = """**🤖 Auto Mode - Extreme Intelligence**

Fully automatic music analysis with adaptive beat detection:

**Features:**
- 🎵 **Song Structure Detection**: Intro/Verse/Chorus/Bridge/Outro
- ⚡ **Energy Analysis**: High/Medium/Low energy per section
- 🎯 **Rhythm Pattern Recognition**: Kick/Clap/Bass/Hi-hat patterns
- 🧠 **Adaptive Cut Frequency**: More cuts in chorus, fewer in intro/outro
- 🎼 **Context-Aware Selection**: Follows musical structure

**How it works:**
- Analyzes your song's energy and structure
- Detects dominant rhythm patterns per section
- Automatically adjusts cut frequency
- More cuts in high-energy sections (chorus)
- Fewer cuts in low-energy sections (intro/outro)
- Follows kick-clap patterns intelligently

**Example:**
- Fast kick-clap track → Cuts on every kick and some claps
- Bass-heavy track → Follows bass rhythm
- High energy chorus → More frequent cuts
- Calm verse → Fewer cuts

**Best for:** Hands-off processing, optimal results without tweaking"""

# ============================================================================
# PERFORMANCE GUIDE
# ============================================================================

def get_performance_guide(cpu_count, parallel_workers, python_status, cuda_status, 
                         ffmpeg_status, gpu_available, gpu_info, nvenc_available):
    """Concise performance guide."""
    
    gpu_text = ""
    if gpu_available:
        nvenc_text = "NVENC: 2-3x faster encoding" if nvenc_available else ""
        gpu_text = f"""**GPU Acceleration** ({gpu_info}):
- Audio analysis: 5-10x faster
- {nvenc_text}

"""
    
    return f"""**System**: {cpu_count} CPU threads | {parallel_workers} parallel workers

**Portable Components**:
- Python: {python_status}
- CUDA: {cuda_status}
- FFmpeg: {ffmpeg_status}

{gpu_text}**Processing Modes**:
- **NVENC H.264**: GPU-accelerated (fastest)
- **NVENC HEVC**: GPU-accelerated (smaller files)
- **CPU H.264**: Software encoding
- **ProRes 422 Proxy**: Lossless (converts to ProRes, then concatenates)

**Temp Files**: `./temp/` folder (local only, auto-cleaned)

**FPS**: Leave empty to auto-detect from input video, or set custom (24/30/60)

**Speed Estimates** (3-min video):
- CPU only: ~2-3 min
- GPU + NVENC: ~45-90 sec ⚡
- ProRes: ~3-5 min (conversion) + instant (concat)

**RAM Usage**: Constant with FFmpeg (no batch processing needed)

**Parallel Workers**:
- More workers = faster processing
- Recommended: {parallel_workers} (auto-calculated from CPU)
- With NVENC: Can use more workers (GPU handles encoding)"""

# ============================================================================
# GUIDES
# ============================================================================

NVENC_GUIDE_ACTIVE = """**🚀 NVENC Auto-Enabled!**
GPU hardware encoding: 2-3x faster than CPU.
Select NVIDIA NVENC H.264 or HEVC for best performance.

---

"""

QUICK_START_GUIDE = """### 💡 Quick Start

**🤖 Auto Mode (Recommended):**
1. Upload audio + videos
2. Select **Auto Mode**
3. Click "Create Music Video"
4. Done! Automatic optimal cuts

**🧠 Smart Mode (Advanced):**
1. Upload audio + videos
2. Select **Smart Mode** → Choose preset
3. Select **NVIDIA NVENC H.264** (if available)
4. Click "Create Music Video"

**⚙️ Manual Mode (Simple):**
1. Upload audio + videos
2. Select **Manual Mode**
3. Adjust cut intensity slider
4. Click "Create Music Video"

**🎯 Lossless (ProRes):**
- Select **ProRes 422 Proxy** mode for frame-perfect quality

---

"""

# ============================================================================
# SYSTEM INFO PANEL
# ============================================================================

def get_system_info_panel(python_status, cuda_status, cpu_count, max_threads, 
                         ffmpeg_status, gpu_status, gpu_device, nvenc_status, 
                         librosa_version):
    """Compact system info."""
    return f"""**System:**
Python: {python_status} | CUDA: {cuda_status}
CPU: {cpu_count} cores ({max_threads} threads)
FFmpeg: {ffmpeg_status} | Librosa: {librosa_version}
GPU: {gpu_status}{gpu_device} | NVENC: {nvenc_status}

**Files:**
- Python: `bin/python-3.13.9-embed-amd64/`
- CUDA: `bin/CUDA/v13.0/`
- FFmpeg: `bin/ffmpeg/ffmpeg.exe`
- Temp: `./temp/` (local only)
- Output: `./output/`

**Modes:**
- ⚙️ Manual: Bass-focused, simple control
- 🧠 Smart: Multi-band, preset-based
- 🤖 Auto: Extreme intelligence, fully automatic"""

# ============================================================================
# STATUS MESSAGES
# ============================================================================

def get_ready_status(python_status, cuda_status, max_threads, cpu_count, ffmpeg_status,
                    gpu_available, gpu_info, nvenc_available):
    """Ready status message."""
    gpu_line = f'⚡ GPU: {gpu_info}\n' if gpu_available else '💻 CPU mode only\n'
    nvenc_line = f'🎬 NVENC: Enabled\n' if nvenc_available else ''
    
    return (f'✅ Ready to process!\n'
            f'🐍 Python: {python_status}\n'
            f'🎮 CUDA: {cuda_status}\n'
            f'💻 CPU: {max_threads}/{cpu_count} threads\n'
            f'📦 FFmpeg: {ffmpeg_status}\n'
            f'{gpu_line}{nvenc_line}'
            f'🎯 Modes: ⚙️ Manual | 🧠 Smart | 🤖 Auto\n'
            f'🎯 ProRes mode available\n'
            f'📁 Temp: Local folder only\n\n'
            f'Upload audio and video files to begin.')

# ============================================================================
# SUCCESS MESSAGES
# ============================================================================

def get_success_message_smart(preset, preset_info, total_beats, tempo, total_cuts,
                              python_str, cuda_str, max_threads, cpu_count,
                              parallel_workers, gpu_info, encoder_info,
                              codec_info, fps_info, filename, audio_info):
    """Success message for Smart Mode."""
    return f"""✅ Video created successfully!

🧠 Smart Mode: {preset.upper()}
   • {preset_info['description']}
   • {total_beats} beats detected at {tempo:.1f} BPM
   • {total_cuts} rhythm-based cuts

🚀 Performance:
   • Python: {python_str} | CUDA: {cuda_str}
   • CPU: {max_threads}/{cpu_count} threads | Workers: {parallel_workers}
   • Audio: {gpu_info} | Video: {encoder_info}

🎬 Export:
   • {codec_info} | {fps_info} | {audio_info}

📁 Output: {filename}"""


def get_success_message_manual_subdivided(total_cuts, subdivisions, total_beats, tempo,
                                         cut_intensity, python_str, cuda_str, max_threads,
                                         cpu_count, parallel_workers, gpu_info, encoder_info,
                                         codec_info, fps_info, filename, audio_info):
    """Success message for Manual mode with subdivisions."""
    return f"""✅ Video created successfully!

⚙️ Manual Mode: {total_cuts} cuts
   • Subdivided {subdivisions}x from {total_beats} beats
   • {tempo:.1f} BPM | Intensity: {cut_intensity}

🚀 Performance:
   • Python: {python_str} | CUDA: {cuda_str}
   • CPU: {max_threads}/{cpu_count} threads | Workers: {parallel_workers}
   • Audio: {gpu_info} | Video: {encoder_info}

🎬 Export:
   • {codec_info} | {fps_info} | {audio_info}

📁 Output: {filename}"""


def get_success_message_manual_skipped(beats_used, cut_intensity_int, total_beats, tempo,
                                      cut_intensity, python_str, cuda_str, max_threads,
                                      cpu_count, parallel_workers, gpu_info, encoder_info,
                                      codec_info, fps_info, filename, audio_info):
    """Success message for Manual mode with skipped beats."""
    return f"""✅ Video created successfully!

⚙️ Manual Mode: {beats_used} cuts
   • Every {cut_intensity_int} beats from {total_beats} detected
   • {tempo:.1f} BPM | Intensity: {cut_intensity}

🚀 Performance:
   • Python: {python_str} | CUDA: {cuda_str}
   • CPU: {max_threads}/{cpu_count} threads | Workers: {parallel_workers}
   • Audio: {gpu_info} | Video: {encoder_info}

🎬 Export:
   • {codec_info} | {fps_info} | {audio_info}

📁 Output: {filename}"""


def get_success_message_auto(total_cuts, total_beats, tempo, sections_info,
                            python_str, cuda_str, max_threads, cpu_count,
                            parallel_workers, gpu_info, encoder_info,
                            codec_info, fps_info, filename, audio_info):
    """Success message for Auto mode."""
    
    # Build section summary
    section_summary = ""
    if sections_info and len(sections_info) > 0:
        section_summary = "\n   • Sections analyzed:\n"
        for section in sections_info:
            section_summary += f"      - {section['section'].capitalize()}: {section['selected_beats']}/{section['total_beats']} beats ({section['selection_ratio']*100:.1f}%)\n"
    
    return f"""✅ Video created successfully!

🤖 Auto Mode: {total_cuts} cuts (Extreme Intelligence)
   • {total_beats} beats detected at {tempo:.1f} BPM
   • Automatic song structure analysis
   • Adaptive cut frequency per section{section_summary}
🚀 Performance:
   • Python: {python_str} | CUDA: {cuda_str}
   • CPU: {max_threads}/{cpu_count} threads | Workers: {parallel_workers}
   • Audio: {gpu_info} | Video: {encoder_info}

🎬 Export:
   • {codec_info} | {fps_info} | {audio_info}

📁 Output: {filename}"""


# ============================================================================
# CONSOLE MESSAGES
# ============================================================================

CONSOLE_SEPARATOR = "=" * 70

def get_startup_header(cpu_count, max_threads, parallel_workers, python_status, 
                      cuda_status, librosa_version, ffmpeg_status, gpu_available, 
                      gpu_info, nvenc_available):
    """Startup header."""
    gpu_line = f"   GPU: {gpu_info} (Auto-enabled)" if gpu_available else "   GPU: Not available (CPU only)"
    nvenc_line = f"   NVENC: Available (Auto-enabled)" if nvenc_available else "   NVENC: Not available"
    
    return f"""{CONSOLE_SEPARATOR}
🎵 BeatSync Engine
{CONSOLE_SEPARATOR}
   Python: {python_status}
   CUDA: {cuda_status}
   FFmpeg: {ffmpeg_status}
   Librosa: {librosa_version}
   CPU: {cpu_count} threads ({max_threads} max per encode)
   Parallel Workers: {parallel_workers}
   {gpu_line}
   {nvenc_line}
   Modes: ⚙️ Manual | 🧠 Smart | 🤖 Auto
   ProRes 422 Proxy: ENABLED"""

# ============================================================================
# INPUT LABELS & INFO
# ============================================================================

LABEL_AUDIO_FILE = "🎵 Audio File (MP3/WAV/FLAC)"
LABEL_VIDEO_FILES = "🎥 Video Files (MP4/MKV)"

# Generation Mode
LABEL_GENERATION_MODE = "🎯 Generation Mode"
INFO_GENERATION_MODE = "Choose how beats are detected and selected"

# Manual Mode
LABEL_CUT_INTENSITY = "✂️ Cut Intensity"
INFO_CUT_INTENSITY = "< 1.0 = MORE cuts (subdivide) | >= 1.0 = FEWER cuts (skip beats)"

# Smart Mode
LABEL_CUT_PRESET = "🎯 Cut Frequency Preset"
INFO_CUT_PRESET = "Slower = fewer cuts | Faster = more cuts"

# Video Settings
LABEL_DIRECTION = "🔄 Video Direction"
INFO_DIRECTION = "Forward/Backward/Random playback"

LABEL_PLAYBACK_SPEED = "⚡ Playback Speed"
INFO_PLAYBACK_SPEED = "Slow-motion/Normal/Fast-forward"

LABEL_TIMING_OFFSET = "⏱️ Timing Offset (seconds)"
INFO_TIMING_OFFSET = "Fine-tune sync: negative=earlier, positive=later (applied to video playback)"

LABEL_CUSTOM_FPS = "🎞️ Custom FPS (Frame Rate)"
INFO_CUSTOM_FPS = "Leave empty for auto-detect, or enter value (24/30/60)"

# GPU & Processing
LABEL_GPU_STATUS = "⚡ GPU Acceleration Status"

LABEL_PROCESSING_MODE = "🎬 Processing Mode"

# Performance
LABEL_PARALLEL_WORKERS = "⚡ Parallel Workers"
INFO_PARALLEL_WORKERS = "Clips processed simultaneously. More workers with GPU."

# Output
LABEL_OUTPUT_FILENAME = "📝 Output Filename"
INFO_OUTPUT_FILENAME = "Timestamp added automatically (.mkv or .mov)"

def get_gpu_status_info(gpu_available, gpu_info, nvenc_available):
    """GPU status info."""
    if gpu_available and nvenc_available:
        return f'✅ {gpu_info} | NVENC Enabled'
    elif gpu_available:
        return f'✅ {gpu_info} | NVENC Not available'
    else:
        return '❌ CPU mode only'

def get_processing_mode_info_nvenc():
    """Processing mode info with NVENC."""
    return 'GPU (NVENC): High quality | CPU: High quality | ProRes: Max quality'

def get_processing_mode_info_cpu():
    """Processing mode info without NVENC."""
    return 'CPU: H.264 encoding | ProRes: Max quality (NVENC not available)'

def get_parallel_workers_label(recommended_workers):
    """Parallel workers label."""
    return f'⚡ Parallel Workers (Recommended: {recommended_workers})'

def get_parallel_workers_info():
    """Parallel workers info."""
    return 'Clips processed simultaneously. More workers with GPU.'