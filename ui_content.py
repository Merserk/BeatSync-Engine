#!/usr/bin/env python3
"""UI content and status helpers for BeatSync Engine."""

SMART_PRESETS_CONFIG = {
    "slower": {
        "cut_frequency": "every_4th_strong_beat",
        "kick_threshold": 70,
        "clap_threshold": 70,
        "min_interval": 1.5,
        "description": "Cinematic - Every 4th strong beat (fewest cuts)",
    },
    "slow": {
        "cut_frequency": "every_2nd_strong_beat",
        "kick_threshold": 60,
        "clap_threshold": 60,
        "min_interval": 0.8,
        "description": "Relaxed - Every 2nd strong beat",
    },
    "normal": {
        "cut_frequency": "every_strong_beat",
        "kick_threshold": 50,
        "clap_threshold": 50,
        "min_interval": 0.4,
        "description": "Standard - Every strong kick or clap",
    },
    "fast": {
        "cut_frequency": "all_beats_prioritize_strong",
        "kick_threshold": 40,
        "clap_threshold": 40,
        "min_interval": 0.25,
        "description": "Energetic - All beats, prioritize strong",
    },
    "faster": {
        "cut_frequency": "all_beats_plus_subdivisions",
        "kick_threshold": 30,
        "clap_threshold": 30,
        "min_interval": 0.15,
        "description": "Hyper - All beats + subdivisions (most cuts)",
    },
}

UI_TITLE = "BeatSync Engine"
UI_MAIN_DESCRIPTION = (
    "Local-path beat-synced video builder for turning a song and a clip library "
    "into a finished music edit."
)
MANUAL_MODE_DESCRIPTION = """**Manual Mode - Bass-Focused**
- Simple bass frequency detection (20-200 Hz)
- Direct cut intensity control
- Best for: Simple projects, consistent rhythm

**Controls:**
- `< 1.0` = more cuts (subdivide beats)
- `1.0` = every beat
- `> 1.0` = fewer cuts (skip beats)"""
SMART_MODE_DESCRIPTION = """**Smart Mode - Multi-Band Analysis**
- Kick (20-150 Hz) + Clap/Snare (150-4000 Hz) + Hi-hat (4000+ Hz)
- Intelligent frequency-based selection
- Best for: Professional results, complex music

**Presets:**
- **Slower**: every 4th strong beat
- **Slow**: every 2nd strong beat
- **Normal**: every strong beat
- **Fast**: all beats with priority
- **Faster**: all beats plus subdivisions"""
AUTO_MODE_DESCRIPTION = """**Auto Mode - Structure-Aware**

Fully automatic music analysis with adaptive beat detection:

- Song structure detection: intro / verse / chorus / bridge / outro
- Energy analysis by section
- Rhythm pattern recognition
- Adaptive cut frequency based on musical context

**Best for:** hands-off processing and first-pass results."""
CONSOLE_SEPARATOR = "=" * 70


def get_ready_status(
    python_status,
    cuda_status,
    max_threads,
    cpu_count,
    ffmpeg_status,
    gpu_available,
    gpu_info,
    nvenc_available,
):
    gpu_line = f"Analysis: {gpu_info}\n" if gpu_available else "Analysis: CPU only\n"
    nvenc_line = (
        "Video encode: NVENC available\n"
        if nvenc_available
        else "Video encode: CPU / ProRes\n"
    )
    return (
        "Ready to render\n\n"
        f"Python: {python_status}\n"
        f"CUDA: {cuda_status}\n"
        f"CPU budget: {max_threads}/{cpu_count} threads per encode\n"
        f"FFmpeg: {ffmpeg_status}\n"
        f"{gpu_line}{nvenc_line}"
        "Modes: Manual | Smart | Auto\n"
        "Delivery paths: H.264 | HEVC | ProRes 422 Proxy\n"
        "Source handling: Local files stay in place\n\n"
        "Choose a local audio track and clip folder to begin."
    )


def get_success_message_smart(
    preset,
    preset_info,
    total_beats,
    tempo,
    total_cuts,
    python_str,
    cuda_str,
    max_threads,
    cpu_count,
    parallel_workers,
    gpu_info,
    encoder_info,
    codec_info,
    fps_info,
    filename,
    audio_info,
):
    return f"""Video created successfully!

Smart Mode: {preset.upper()}
   - {preset_info['description']}
   - {total_beats} beats detected at {tempo:.1f} BPM
   - {total_cuts} rhythm-based cuts

Performance:
   - Python: {python_str} | CUDA: {cuda_str}
   - CPU: {max_threads}/{cpu_count} threads | Workers: {parallel_workers}
   - Audio: {gpu_info} | Video: {encoder_info}

Export:
   - {codec_info} | {fps_info} | {audio_info}

Output: {filename}"""


def get_success_message_manual_subdivided(
    total_cuts,
    subdivisions,
    total_beats,
    tempo,
    cut_intensity,
    python_str,
    cuda_str,
    max_threads,
    cpu_count,
    parallel_workers,
    gpu_info,
    encoder_info,
    codec_info,
    fps_info,
    filename,
    audio_info,
):
    return f"""Video created successfully!

Manual Mode: {total_cuts} cuts
   - Subdivided {subdivisions}x from {total_beats} beats
   - {tempo:.1f} BPM | Intensity: {cut_intensity}

Performance:
   - Python: {python_str} | CUDA: {cuda_str}
   - CPU: {max_threads}/{cpu_count} threads | Workers: {parallel_workers}
   - Audio: {gpu_info} | Video: {encoder_info}

Export:
   - {codec_info} | {fps_info} | {audio_info}

Output: {filename}"""


def get_success_message_manual_skipped(
    beats_used,
    cut_intensity_int,
    total_beats,
    tempo,
    cut_intensity,
    python_str,
    cuda_str,
    max_threads,
    cpu_count,
    parallel_workers,
    gpu_info,
    encoder_info,
    codec_info,
    fps_info,
    filename,
    audio_info,
):
    return f"""Video created successfully!

Manual Mode: {beats_used} cuts
   - Every {cut_intensity_int} beats from {total_beats} detected
   - {tempo:.1f} BPM | Intensity: {cut_intensity}

Performance:
   - Python: {python_str} | CUDA: {cuda_str}
   - CPU: {max_threads}/{cpu_count} threads | Workers: {parallel_workers}
   - Audio: {gpu_info} | Video: {encoder_info}

Export:
   - {codec_info} | {fps_info} | {audio_info}

Output: {filename}"""


def get_success_message_auto(
    total_cuts,
    total_beats,
    tempo,
    sections_info,
    python_str,
    cuda_str,
    max_threads,
    cpu_count,
    parallel_workers,
    gpu_info,
    encoder_info,
    codec_info,
    fps_info,
    filename,
    audio_info,
):
    section_summary = ""
    if sections_info:
        section_summary = "\n   - Sections analyzed:\n"
        for section in sections_info:
            section_summary += (
                f"      * {section['section'].capitalize()}: "
                f"{section['selected_beats']}/{section['total_beats']} beats "
                f"({section['selection_ratio'] * 100:.1f}%)\n"
            )

    return f"""Video created successfully!

Auto Mode: {total_cuts} cuts
   - {total_beats} beats detected at {tempo:.1f} BPM
   - Automatic song structure analysis
   - Adaptive cut frequency per section{section_summary}
Performance:
   - Python: {python_str} | CUDA: {cuda_str}
   - CPU: {max_threads}/{cpu_count} threads | Workers: {parallel_workers}
   - Audio: {gpu_info} | Video: {encoder_info}

Export:
   - {codec_info} | {fps_info} | {audio_info}

Output: {filename}"""


def get_startup_header(
    cpu_count,
    max_threads,
    parallel_workers,
    python_status,
    cuda_status,
    librosa_version,
    ffmpeg_status,
    gpu_available,
    gpu_info,
    nvenc_available,
):
    gpu_line = (
        f"   GPU: {gpu_info} (Auto-enabled)"
        if gpu_available
        else "   GPU: Not available (CPU only)"
    )
    nvenc_line = (
        "   NVENC: Available (Auto-enabled)"
        if nvenc_available
        else "   NVENC: Not available"
    )
    return f"""{CONSOLE_SEPARATOR}
BeatSync Engine
{CONSOLE_SEPARATOR}
   Python: {python_status}
   CUDA: {cuda_status}
   FFmpeg: {ffmpeg_status}
   Librosa: {librosa_version}
   CPU: {cpu_count} threads ({max_threads} max per encode)
   Parallel Workers: {parallel_workers}
   {gpu_line}
   {nvenc_line}
   Modes: Manual | Smart | Auto
   ProRes 422 Proxy: ENABLED"""


LABEL_AUDIO_FILE = "Audio Track"
INFO_AUDIO_FILE = (
    "Choose a local MP3, WAV, or FLAC file. BeatSync reads it in place instead of uploading it."
)
LABEL_VIDEO_FOLDER = "Clip Folder"
INFO_VIDEO_FOLDER = (
    "Choose the local folder that contains your source clips. Subfolders are scanned recursively for MP4 and MKV files."
)
LABEL_GENERATION_MODE = "Beat Strategy"
INFO_GENERATION_MODE = "Choose how BeatSync should detect and prioritize edit points."
LABEL_CUT_INTENSITY = "Manual Intensity"
INFO_CUT_INTENSITY = (
    "Below 1.0 creates more cuts by subdividing beats. Values above 1.0 skip beats for a calmer edit."
)
LABEL_CUT_PRESET = "Smart Preset"
INFO_CUT_PRESET = (
    "Move from slower cinematic cutting toward faster, denser rhythm tracking."
)
LABEL_DIRECTION = "Playback Direction"
INFO_DIRECTION = "Choose whether source clips move forward, backward, or switch randomly."
LABEL_PLAYBACK_SPEED = "Playback Speed"
INFO_PLAYBACK_SPEED = "Adjust clip speed before BeatSync assembles the final edit."
LABEL_TIMING_OFFSET = "Timing Offset (seconds)"
INFO_TIMING_OFFSET = (
    "Fine-tune sync placement. Negative values shift video earlier; positive values shift it later."
)
LABEL_CUSTOM_RESOLUTION = "Target Resolution"
INFO_CUSTOM_RESOLUTION = (
    "Leave this on default to match the first source clip, or choose a preset frame. BeatSync preserves aspect ratio and pads to fit."
)
LABEL_CUSTOM_FPS = "Target FPS"
INFO_CUSTOM_FPS = "Leave empty for auto-detect, or enter a frame rate such as 24, 30, or 60."
LABEL_PROCESSING_MODE = "Export Pipeline"
LABEL_STANDARD_QUALITY = "Delivery Quality"
INFO_STANDARD_QUALITY = (
    "Used for CPU and NVENC delivery exports. ProRes uses its own quality-first workflow."
)
LABEL_PRORES_DELIVERY_MP4 = "Also create delivery MP4 (Lossless)"
INFO_PRORES_DELIVERY_MP4 = (
    "ProRes only. Keep the MOV master and also create a lossless MP4 delivery copy."
)
LABEL_OUTPUT_FILENAME = "Export Name"
INFO_OUTPUT_FILENAME = (
    "A timestamp is added automatically. The file extension follows the selected export pipeline."
)


def get_processing_mode_info_nvenc():
    return "NVENC is available. Use GPU delivery for speed or switch to ProRes for a quality-first master."


def get_processing_mode_info_cpu():
    return "Use CPU H.264 for universal delivery or ProRes when you need a quality-first master export."


def get_parallel_workers_label(recommended_workers):
    return f"Parallel Clip Workers (Recommended: {recommended_workers})"


def get_parallel_workers_info():
    return "Controls how many clips are processed at once. Higher values improve throughput but reduce threads available to each encode."
