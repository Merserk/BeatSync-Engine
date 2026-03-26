#!/usr/bin/env python3
"""
UI content and status helpers for BeatSync Engine.
"""

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
UI_MAIN_DESCRIPTION = """Create music videos that cut to the beat. Enter a local audio file path and
a local video folder path to generate a video synchronized with your music's rhythm!"""

MANUAL_MODE_DESCRIPTION = """**Manual Mode - Bass-Focused**
- Simple bass frequency detection (20-200 Hz)
- Direct cut intensity control
- Best for: Simple projects, consistent rhythm

**Controls:**
- `< 1.0` = MORE cuts (subdivide beats)
- `1.0` = Every beat (standard)
- `> 1.0` = FEWER cuts (skip beats)"""

SMART_MODE_DESCRIPTION = """**Smart Mode - Multi-Band Analysis**
- Kick (20-150 Hz) + Clap/Snare (150-4000 Hz) + Hi-hat (4000+ Hz)
- Intelligent frequency-based selection
- Best for: Professional results, complex music

**Presets:**
- **Slower**: Every 4th strong beat (cinematic)
- **Slow**: Every 2nd strong beat (relaxed)
- **Normal**: Every strong beat (standard)
- **Fast**: All beats with priority (energetic)
- **Faster**: All beats + subdivisions (hyper)"""

AUTO_MODE_DESCRIPTION = """**Auto Mode - Extreme Intelligence**

Fully automatic music analysis with adaptive beat detection:

- Song structure detection: intro / verse / chorus / bridge / outro
- Energy analysis by section
- Rhythm pattern recognition
- Adaptive cut frequency based on the musical context

**Best for:** Hands-off processing and optimal results without tweaking."""

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
    gpu_line = f"⚡ GPU: {gpu_info}\n" if gpu_available else "💻 CPU mode only\n"
    nvenc_line = "🎬 NVENC: Enabled\n" if nvenc_available else ""
    return (
        f"✅ Ready to process!\n"
        f"🐍 Python: {python_status}\n"
        f"🎮 CUDA: {cuda_status}\n"
        f"💻 CPU: {max_threads}/{cpu_count} threads\n"
        f"📦 FFmpeg: {ffmpeg_status}\n"
        f"{gpu_line}{nvenc_line}"
        f"🎯 Modes: ⚙️ Manual | 🧠 Smart | 🤖 Auto\n"
        f"🎯 ProRes mode available\n"
        f"📁 Temp: Local folder only\n\n"
        f"Enter a local audio path and video folder to begin."
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
BeatSync Engine 界面文案（中文）
统一模式选择：手动 / 智能 / 自动
"""

SMART_PRESETS_CONFIG = {
    'slower': {
        'cut_frequency': 'every_4th_strong_beat',
        'kick_threshold': 70,
        'clap_threshold': 70,
        'min_interval': 1.5,
        'description': '电影感：每第 4 个强拍切一次（切换最少）'
    },
    'slow': {
        'cut_frequency': 'every_2nd_strong_beat',
        'kick_threshold': 60,
        'clap_threshold': 60,
        'min_interval': 0.8,
        'description': '舒缓：每第 2 个强拍切一次'
    },
    'normal': {
        'cut_frequency': 'every_strong_beat',
        'kick_threshold': 50,
        'clap_threshold': 50,
        'min_interval': 0.4,
        'description': '标准：每个强 Kick / Clap 切一次'
    },
    'fast': {
        'cut_frequency': 'all_beats_prioritize_strong',
        'kick_threshold': 40,
        'clap_threshold': 40,
        'min_interval': 0.25,
        'description': '活力：全拍点切换，优先强拍'
    },
    'faster': {
        'cut_frequency': 'all_beats_plus_subdivisions',
        'kick_threshold': 30,
        'clap_threshold': 30,
        'min_interval': 0.15,
        'description': '高能：全拍点 + 细分拍（切换最多）'
    }
}

UI_TITLE = "🎵 BeatSync Engine"
UI_MAIN_DESCRIPTION = """把音乐与视频节奏自动对齐，一键生成卡点视频。\n上传音频和视频片段后即可开始处理。"""

MANUAL_MODE_DESCRIPTION = """**⚙️ 手动模式 - 低频优先**
- 基于 20-200Hz 低频节拍检测
- 直接控制切换强度
- 适合：节奏稳定、需求简单的项目

**控制说明：**
- `< 1.0` = 切换更多（细分拍点）
- `1.0` = 每拍切换（标准）
- `> 1.0` = 切换更少（跳拍）"""

SMART_MODE_DESCRIPTION = """**🧠 智能模式 - 多频段分析**
- Kick（20-150Hz）+ Clap/Snare（150-4000Hz）+ Hi-hat（4000Hz+）
- 按频段强度智能选择切点
- 适合：复杂编曲、追求更稳定观感

**预设：**
- **更慢**：每第 4 个强拍
- **慢速**：每第 2 个强拍
- **标准**：每个强拍
- **快速**：全拍点优先强拍
- **更快**：全拍点 + 细分拍"""

AUTO_MODE_DESCRIPTION = """**🤖 自动模式 - 全自动分析（推荐）**

自动完成歌曲结构与节奏分析，并按段落动态调整切换频率：

- 识别段落：前奏 / 主歌 / 副歌 / 过渡 / 尾声
- 识别能量：高 / 中 / 低能量区间
- 高能段落切换更多，低能段落切换更少
- 自动跟随主导节奏乐器

**适合：** 想快速获得稳定结果、尽量少调参数。"""


def get_ready_status(python_status, cuda_status, max_threads, cpu_count, ffmpeg_status,
                     gpu_available, gpu_info, nvenc_available):
    gpu_line = f'⚡ GPU：{gpu_info}\n' if gpu_available else '💻 仅 CPU 模式\n'
    nvenc_line = '🎬 NVENC：已启用\n' if nvenc_available else ''
    return (f'✅ 已就绪，等待处理！\n'
            f'🐍 Python：{python_status}\n'
            f'🎮 CUDA：{cuda_status}\n'
            f'💻 CPU：{max_threads}/{cpu_count} 线程\n'
            f'📦 FFmpeg：{ffmpeg_status}\n'
            f'{gpu_line}{nvenc_line}'
            f'🎯 模式：⚙️ 手动 | 🧠 智能 | 🤖 自动\n'
            f'🎯 支持 ProRes 模式\n'
            f'📁 临时目录：仅项目本地\n\n'
            f'请先上传音频与视频文件。')


def get_success_message_smart(preset, preset_info, total_beats, tempo, total_cuts,
                              python_str, cuda_str, max_threads, cpu_count,
                              parallel_workers, gpu_info, encoder_info,
                              codec_info, fps_info, filename, audio_info):
    return f"""✅ 视频生成成功！

🧠 智能模式：{preset.upper()}
   • {preset_info['description']}
   • 检测到 {total_beats} 个节拍，速度 {tempo:.1f} BPM
   • 生成 {total_cuts} 次节奏切换

🚀 性能信息：
   • Python：{python_str} | CUDA：{cuda_str}
   • CPU：{max_threads}/{cpu_count} 线程 | 并行 Worker：{parallel_workers}
   • 音频：{gpu_info} | 视频：{encoder_info}

🎬 导出信息：
   • {codec_info} | {fps_info} | {audio_info}

📁 输出文件：{filename}"""


def get_success_message_manual_subdivided(total_cuts, subdivisions, total_beats, tempo,
                                          cut_intensity, python_str, cuda_str, max_threads,
                                          cpu_count, parallel_workers, gpu_info, encoder_info,
                                          codec_info, fps_info, filename, audio_info):
    return f"""✅ 视频生成成功！

⚙️ 手动模式：{total_cuts} 次切换
   • 在 {total_beats} 个节拍基础上细分 {subdivisions} 倍
   • {tempo:.1f} BPM | 强度：{cut_intensity}

🚀 性能信息：
   • Python：{python_str} | CUDA：{cuda_str}
   • CPU：{max_threads}/{cpu_count} 线程 | 并行 Worker：{parallel_workers}
   • 音频：{gpu_info} | 视频：{encoder_info}

🎬 导出信息：
   • {codec_info} | {fps_info} | {audio_info}

📁 输出文件：{filename}"""


def get_success_message_manual_skipped(beats_used, cut_intensity_int, total_beats, tempo,
                                       cut_intensity, python_str, cuda_str, max_threads,
                                       cpu_count, parallel_workers, gpu_info, encoder_info,
                                       codec_info, fps_info, filename, audio_info):
    return f"""✅ 视频生成成功！

⚙️ 手动模式：{beats_used} 次切换
   • 在 {total_beats} 个节拍中，每 {cut_intensity_int} 拍切一次
   • {tempo:.1f} BPM | 强度：{cut_intensity}

🚀 性能信息：
   • Python：{python_str} | CUDA：{cuda_str}
   • CPU：{max_threads}/{cpu_count} 线程 | 并行 Worker：{parallel_workers}
   • 音频：{gpu_info} | 视频：{encoder_info}

🎬 导出信息：
   • {codec_info} | {fps_info} | {audio_info}

📁 输出文件：{filename}"""


def get_success_message_auto(total_cuts, total_beats, tempo, sections_info,
                             python_str, cuda_str, max_threads, cpu_count,
                             parallel_workers, gpu_info, encoder_info,
                             codec_info, fps_info, filename, audio_info):
    section_summary = ""
    if sections_info:
        section_summary = "\n   • 段落分析结果：\n"
        for section in sections_info:
            section_summary += (
                f"      - {section['section'].capitalize()}："
                f"{section['selected_beats']}/{section['total_beats']} 拍 "
                f"({section['selection_ratio']*100:.1f}%)\n"
            )

    return f"""✅ 视频生成成功！

🤖 自动模式：{total_cuts} 次切换
   • 检测到 {total_beats} 个节拍，速度 {tempo:.1f} BPM
   • 已完成歌曲结构自动分析
   • 各段落自适应切换频率{section_summary}
🚀 性能信息：
   • Python：{python_str} | CUDA：{cuda_str}
   • CPU：{max_threads}/{cpu_count} 线程 | 并行 Worker：{parallel_workers}
   • 音频：{gpu_info} | 视频：{encoder_info}

🎬 导出信息：
   • {codec_info} | {fps_info} | {audio_info}

📁 输出文件：{filename}"""


CONSOLE_SEPARATOR = "=" * 70


def get_startup_header(cpu_count, max_threads, parallel_workers, python_status,
                       cuda_status, librosa_version, ffmpeg_status, gpu_available,
                       gpu_info, nvenc_available):
    gpu_line = f"   GPU: {gpu_info} (自动启用)" if gpu_available else "   GPU: 不可用（仅 CPU）"
    nvenc_line = "   NVENC: 可用（自动启用）" if nvenc_available else "   NVENC: 不可用"

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
   Modes: Manual | Smart | Auto
   ProRes 422 Proxy: ENABLED"""


LABEL_AUDIO_FILE = "Audio File Path"
INFO_AUDIO_FILE = (
    "Absolute or relative local path to an MP3, WAV, or FLAC file, or use Browse"
)
LABEL_VIDEO_FOLDER = "Video Folder Path"
INFO_VIDEO_FOLDER = (
    "Absolute or relative local folder path containing MP4/MKV clips; subfolders are scanned recursively, or use Browse"
)
LABEL_GENERATION_MODE = "Generation Mode"
INFO_GENERATION_MODE = "Choose how beats are detected and selected"
LABEL_CUT_INTENSITY = "Cut Intensity"
INFO_CUT_INTENSITY = "< 1.0 = MORE cuts (subdivide) | >= 1.0 = FEWER cuts (skip beats)"
LABEL_CUT_PRESET = "Cut Frequency Preset"
INFO_CUT_PRESET = "Slower = fewer cuts | Faster = more cuts"
LABEL_DIRECTION = "Video Direction"
INFO_DIRECTION = "Forward / Backward / Random playback"
LABEL_PLAYBACK_SPEED = "Playback Speed"
INFO_PLAYBACK_SPEED = "Slow-motion / Normal / Fast-forward"
LABEL_TIMING_OFFSET = "Timing Offset (seconds)"
INFO_TIMING_OFFSET = (
    "Fine-tune sync: negative=earlier, positive=later (applied to video playback)"
)
LABEL_CUSTOM_RESOLUTION = "Custom Resolution"
INFO_CUSTOM_RESOLUTION = (
    "Leave at default to match the first source clip, or pick a common 16:9, 21:9, or 9:16 target. BeatSync preserves aspect ratio and pads to fit."
)
LABEL_CUSTOM_FPS = "Custom FPS (Frame Rate)"
INFO_CUSTOM_FPS = "Leave empty for auto-detect, or enter a value such as 24 / 30 / 60"
LABEL_PROCESSING_MODE = "Processing Mode"
LABEL_STANDARD_QUALITY = "Standard Export Quality"
INFO_STANDARD_QUALITY = "Used for CPU/NVENC exports only. ProRes ignores this setting."
LABEL_PRORES_DELIVERY_MP4 = "Also create delivery MP4 (Lossless)"
INFO_PRORES_DELIVERY_MP4 = "ProRes only. Keeps the .mov master and also creates a lossless H.264/ALAC .mp4 delivery copy."
LABEL_OUTPUT_FILENAME = "Output Filename"
INFO_OUTPUT_FILENAME = "Timestamp added automatically (.mp4 or .mov)"


def get_processing_mode_info_nvenc():
    return "GPU/CPU: Delivery exports with quality presets | ProRes: Precise lossless workflow"


def get_processing_mode_info_cpu():
    return (
        "CPU: H.264 encoding with quality presets | ProRes: Precise lossless workflow"
    )


def get_parallel_workers_label(recommended_workers):
    return f"Parallel Workers (Recommended: {recommended_workers})"


def get_parallel_workers_info():
    return "Clips processed simultaneously. More workers can help, but each CPU encode gets fewer threads."
   CPU: {cpu_count} 线程（单次编码最多 {max_threads}）
   Parallel Workers: {parallel_workers}
   {gpu_line}
   {nvenc_line}
   Modes: ⚙️ 手动 | 🧠 智能 | 🤖 自动
   ProRes 422 Proxy: 已启用"""


LABEL_AUDIO_FILE = "🎵 音频文件（MP3/WAV/FLAC）"
LABEL_VIDEO_FILES = "🎥 视频文件（MP4/MKV）"

LABEL_GENERATION_MODE = "🎯 生成模式"
INFO_GENERATION_MODE = "选择节拍检测与切换策略"

LABEL_CUT_INTENSITY = "✂️ 切换强度"
INFO_CUT_INTENSITY = "< 1.0 = 切换更多（细分）| >= 1.0 = 切换更少（跳拍）"

LABEL_CUT_PRESET = "🎯 切换频率预设"
INFO_CUT_PRESET = "更慢=更少切换 | 更快=更多切换"

LABEL_DIRECTION = "🔄 视频方向"
INFO_DIRECTION = "正放 / 倒放 / 随机"

LABEL_PLAYBACK_SPEED = "⚡ 播放速度"
INFO_PLAYBACK_SPEED = "慢放 / 正常 / 快放"

LABEL_TIMING_OFFSET = "⏱️ 时间偏移（秒）"
INFO_TIMING_OFFSET = "微调同步：负值更早，正值更晚（作用于视频播放）"

LABEL_CUSTOM_FPS = "🎞️ 自定义 FPS（帧率）"
INFO_CUSTOM_FPS = "留空为自动检测，或输入 24/30/60 等值"

LABEL_GPU_STATUS = "⚡ GPU 加速状态"
LABEL_PROCESSING_MODE = "🎬 处理模式"

LABEL_PARALLEL_WORKERS = "⚡ 并行 Worker"
INFO_PARALLEL_WORKERS = "同时处理多个片段。GPU 下可使用更多并行。"

LABEL_OUTPUT_FILENAME = "📝 输出文件名"
INFO_OUTPUT_FILENAME = "会自动追加时间戳（.mp4 或 .mov）"


def get_gpu_status_info(gpu_available, gpu_info, nvenc_available):
    if gpu_available and nvenc_available:
        return f"{gpu_info} | NVENC Enabled"
    if gpu_available:
        return f"{gpu_info} | NVENC Not available"
    return "CPU mode only"
        return f'✅ {gpu_info} | NVENC 已启用'
    if gpu_available:
        return f'✅ {gpu_info} | NVENC 不可用'
    return '❌ 仅 CPU 模式'


def get_processing_mode_info_nvenc():
    return 'GPU（NVENC）：高速高质量 | CPU：高质量 | ProRes：最高质量'


def get_processing_mode_info_cpu():
    return 'CPU：H.264 编码 | ProRes：最高质量（NVENC 不可用）'


def get_parallel_workers_label(recommended_workers):
    return f'⚡ 并行 Worker（推荐：{recommended_workers}）'


def get_parallel_workers_info():
    return '同时处理多个片段。GPU 下可使用更多并行。'
