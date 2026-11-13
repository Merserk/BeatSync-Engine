# BeatSync Engine - AI Audio-Visual Music Video Generator

[![Downloads](https://img.shields.io/github/downloads/Merserk/BeatSync-Engine/total.svg?style=flat-square&label=Downloads)](https://github.com/Merserk/BeatSync-Engine/releases)
[![Windows](https://img.shields.io/badge/Platform-Windows-0078D4?style=flat-square&logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![Python](https://img.shields.io/badge/Python-Portable%203.13.17-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Taichi Vulkan](https://img.shields.io/badge/GPU-Taichi%20Vulkan-000000?style=flat-square)](https://www.taichi-lang.org/)
[![NVENC](https://img.shields.io/badge/Encoding-NVENC-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new)
[![Vulkan Video](https://img.shields.io/badge/Encoding-Vulkan%20Video-AC162C?style=flat-square)](https://www.khronos.org/vulkan/)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-Portable%209.0.2-007808?style=flat-square&logo=ffmpeg&logoColor=white)](https://ffmpeg.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-F97316?style=flat-square)](https://www.gradio.app/)
[![Qwen3.5-2B](https://img.shields.io/badge/Vision-Qwen3.5--2B-blueviolet?style=flat-square)](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF)

A portable Windows app that creates beat-synchronized **AMV/GMV/music videos** from one audio track and one or more source videos.

BeatSync Engine analyzes music rhythm, energy, sections, and source-video moments, then builds a frame-locked edit timeline with optional **Qwen3.5-2B semantic scene matching**.

<img width="1402" height="743" alt="image" src="https://github.com/user-attachments/assets/c7323c0a-a49d-4596-adcc-ec0e8e23b774" />

> **Goal:** upload audio + video clips, click one button, and get a finished rhythmic music video with clean beat cuts, strong source-moment selection, and GPU-accelerated rendering when available.

---

## 🎥 Demo Video

[![Watch the BeatSync Engine demo video](https://img.youtube.com/vi/Iv00-5GkoOM/maxresdefault.jpg)](https://www.youtube.com/watch?v=Iv00-5GkoOM)

[![Watch the BeatSync Engine demo video](https://img.youtube.com/vi/jud_ZxxDjPg/maxresdefault.jpg)](https://www.youtube.com/watch?v=jud_ZxxDjPg)

> Watch the special showcase video for BeatSync Engine on YouTube.

---

## ✨ Features

*   **🎵 Automatic Beat Editing:** Detects a stable beat grid and cuts source footage to the music rhythm.
*   **🌊 Energy-Wave Cut Density:** Calm sections hold longer; drops, impacts, and high-energy parts cut faster.
*   **🎼 Song Structure Detection:** Finds broad intro, verse, chorus, bridge, drop, build, body, and outro-style sections.
*   **🥁 Rhythm Feature Analysis:** Reads kick, clap, bass, hi-hat, novelty, impact, bar anchors, and phrase anchors.
*   **🎬 Source Video Moment Library:** Scans source videos for motion, quality, scene changes, action, beauty, and usable moments.
*   **🧠 Qwen3.5-2B Semantic Tags:** Optional local llama.cpp Vulkan vision-language tagging (`thinking` disabled) for action, combat, chase, beauty, emotion, drops, builds, soft moments, and visible text detection.
*   **🎯 Audio-Visual Planner:** Chooses unique, shot-clean source moments with optional auto speed control and text avoidance instead of random clip sampling.
*   **⚡ GPU Acceleration:** Automatically uses the detected discrete GPU with Taichi Vulkan for audio separation and video scoring, llama.cpp Vulkan for Qwen3.5-2B, and FFmpeg hardware decoding, scaling, and encoding.
*   **🎞️ Frame-Locked Timeline:** Quantizes cut boundaries to absolute output frames and checks native shot boundaries to avoid timing drift or flash frames.
*   **🎬 Multiple Export Modes:** NVIDIA NVENC H.264, NVIDIA NVENC HEVC (H.265), Vulkan Video H.264, Vulkan Video H.265, CPU (H.264), and ProRes 422 HQ (`.mov`).
*   **📦 Portable Runtime:** Bundled portable Python 3.13.17, FFmpeg 9.0.2 full build, llama.cpp Vulkan (`b12360`), Taichi Vulkan, and local Qwen3.5-2B GGUF model files.
*   **🌐 Local Web UI:** Launches a Gradio interface at `http://127.0.0.1:7860` by default.
*   **🧹 Local Caches:** Reuses visual and semantic analysis data in `input/video_analysis_cache/` so repeated runs can be faster.

---

## 📦 Portable Folder Layout

A complete portable release is expected to look like this:

```text
BeatSync Engine V5/
├── install.bat                              # One-click Windows installer
├── run.bat                                  # One-click Windows launcher
├── requirements.txt                         # Python package requirements
├── scripts/
│   └── install.ps1                          # Portable runtime/model installer
├── bin/
│   ├── python-3.13.17-embed-amd64/          # Main portable Python 3.13.17 runtime
│   ├── ffmpeg/ffmpeg.exe                    # Portable FFmpeg 9.0.2 (full build)
│   ├── ffmpeg/ffprobe.exe                   # Portable FFprobe 9.0.2
│   ├── llama-bin-win-vulkan-x64/            # llama.cpp Vulkan backend (b12360)
│   └── models/
│       ├── Qwen3.5-2B-Q8_0.gguf             # Local Qwen3.5-2B GGUF model
│       └── mmproj-F16.gguf                  # Local Qwen3.5 vision projector
├── src/
│   ├── gui.py                               # Gradio web UI
│   ├── video_processor.py                   # Rendering pipeline and CLI
│   ├── video_analysis.py                    # Source-video visual library
│   ├── video_encoders.py                    # NVENC, Vulkan Video, and ProRes probes
│   ├── ffmpeg_runtime.py                    # Portable FFmpeg/FFprobe validation
│   ├── ffmpeg_processing.py                 # FFmpeg/FFprobe processing helpers
│   ├── gpu_devices.py                       # Discrete GPU discovery and backend matching
│   ├── gpu_cpu_utils.py                     # CPU, Taichi Vulkan, and NVENC helpers
│   ├── gpu_video.py                         # GPU decode and scaling paths
│   ├── taichi_backend.py                    # Taichi Vulkan audio/video compute kernels
│   ├── clip_uniqueness.py                   # Whole-edit duplicate clip prevention
│   ├── shot_integrity.py                    # Native shot-boundary verification
│   ├── logger.py                            # Portable environment and console setup
│   ├── paths.py                             # Local input/output path helpers
│   ├── auto_mode/
│   │   ├── __init__.py                      # Auto Mode pipeline entry point
│   │   ├── stage1_audio.py                  # Beat grid detection
│   │   ├── stage2_features.py               # Energy/rhythm features
│   │   ├── stage3_sections.py               # Music section analysis
│   │   ├── stage4_select.py                 # Rhythmic cut selection
│   │   ├── stage5_qwen_scene_worker.py      # Qwen3.5 / llama.cpp Vulkan worker
│   │   └── stage6_av_planner.py             # Audio-visual clip planner
│   └── ui_content.py                        # UI labels and status text
├── input/
│   ├── audio/                               # Audio input directory
│   ├── video/                               # Video input directory
│   ├── processing/                          # Temporary processing files
│   ├── gradio_uploads/                      # Gradio upload/session temp files
│   └── video_analysis_cache/                # Visual and semantic analysis cache
└── output/                                  # Final exported videos
```

> The app creates `input/` and `output/` subfolders automatically if they are missing.

---

## 🛠️ Quick Start - Windows Portable

1.  Download or extract BeatSync Engine V5.

2.  Keep the folder path simple, for example:

    ```text
    D:\AI\BeatSync Engine V5\
    ```

3.  Run the installer once if the portable runtime is not already included:

    ```bat
    install.bat
    ```

4.  Start the app:

    ```bat
    run.bat
    ```

5.  The browser opens automatically:

    ```text
    http://127.0.0.1:7860
    ```

6.  Upload:

    *   one audio file: `.mp3`, `.wav`, or `.flac`;
    *   one or more source videos: `.mp4` or `.mkv`.

7.  Choose a processing mode.

8.  Click:

    ```text
    🎬 Create Music Video
    ```

9.  Finished videos are saved to:

    ```text
    output/
    ```

---

## 🎮 How to Use the Web UI

### 1. Upload files

Use the left panel to upload an audio file and one or more source video files.

Gradio stores session uploads locally inside:

```text
input/gradio_uploads/
```

BeatSync reads the selected files directly from disk without creating duplicate copies.

### 2. Pick FPS

Leave **Custom FPS** empty to auto-detect the frame rate from the first source video (fractional rates such as `30000/1001` are preserved).

Common values:

```text
24
30
60
```

Under **Advanced Settings**, **Auto speed clip control** and **Avoid text** are enabled by default.

### 3. Choose processing mode

| Mode | Output | Best for |
|---|---:|---|
| **NVIDIA NVENC H.264** | `.mp4` | Fast, compatible GPU exports. |
| **NVIDIA NVENC HEVC (H.265)** | `.mp4` | Smaller GPU-encoded files for modern players. |
| **Vulkan Video H.264** | `.mp4` | Hardware H.264 on a Vulkan Video GPU. |
| **Vulkan Video H.265** | `.mp4` | Hardware HEVC on a Vulkan Video GPU. |
| **CPU (H.264)** | `.mp4` | Software `libx264` encoding fallback. |
| **ProRes 422 HQ** | `.mov` | High-quality 10-bit 4:2:2 export for editing handoff. |

Only hardware encoding modes that pass a real driver probe on the detected discrete GPU are shown.

### 4. Create the video

The CMD window shows clean stage progress like:

```text
Stage 1 processing started:
  Beat grid: 259 beats at 152.0 BPM
Stage 1 ended in 8 seconds.

Stage 5 processing started:
  Source videos: 1, visual workers: 1
  Qwen: enabled, model Qwen3.5-2B-Q8_0.gguf
  Qwen tags: 116/116 in 97.5s
  Visual library: 804 visual moments, action=0.54, beauty=0.47, quality=0.62
Stage 5 ended in 192 seconds.

Total time processing: 217 seconds
```

The UI displays an H.264/AAC browser preview and success statistics after the render finishes. **Download exported video** provides the original selected export (including PCM 24-bit audio) from `output/`.

---

## 🎵 Auto Mode Details

Auto Mode is the main creative engine.

### Stage 1 - Beat grid

Detects stable beat positions and tempo from the percussive part of the song using Taichi Vulkan harmonic/percussive audio separation when available.

### Stage 2 - Energy and rhythm

Builds beat-synchronous curves for:

*   RMS energy;
*   spectral brightness;
*   flux and novelty;
*   kick, clap, bass, and hi-hat strength;
*   impact score;
*   bar and phrase anchors.

### Stage 3 - Sections

Groups the song into broad musical sections so the edit can breathe instead of cutting every transient.

### Stage 4 - Cut selection

Selects a deliberate subset of beats. The selector prefers downbeats, bar anchors, phrase anchors, strong impacts, and section-aware cut density.

### Stage 5 - Video analysis + Qwen3.5-2B

Builds a visual library from the uploaded source videos.

Deterministic analysis (with Taichi Vulkan scoring and GPU decode/resize where supported) reads:

*   scene changes;
*   motion strength;
*   visual quality;
*   action score;
*   beauty score;
*   reusable candidate moments.

Optional local Qwen3.5-2B analysis through llama.cpp Vulkan (`thinking` disabled) adds semantic tags:

```text
action_intensity
beauty_score
combat
chase
explosion
character_focus
camera_motion
visual_quality
emotion
recommended_use
visible_text
```

### Stage 6 - Audio-visual render

The renderer builds a frame-accurate cut timeline, chooses unique source moments for each segment (verifying native shot boundaries and avoiding duplicate footage or unwanted text), extracts clips with FFmpeg, concatenates the result, and adds the music track.

---

## Runtime Notes

BeatSync does not require PyTorch, Transformers, or a separately installed CUDA Toolkit. It uses `taichi==1.7.4` with the Vulkan compute backend for audio separation, spectral reductions, color conversion, and candidate scoring, falling back to CPU when GPU compute is unavailable.

At startup, `src/gpu_devices.py` automatically detects one discrete GPU (excluding integrated GPUs) and routes Taichi Vulkan, llama.cpp Vulkan, FFmpeg hardware decoding/scaling, NVENC, and Vulkan Video to that card.

Qwen3.5-2B semantic tagging uses the bundled llama.cpp Vulkan backend (`b12360`, thinking disabled) and these GGUF files from [Unsloth's Qwen3.5-2B GGUF repository](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF):

```text
bin/llama-bin-win-vulkan-x64/
bin/models/Qwen3.5-2B-Q8_0.gguf
bin/models/mmproj-F16.gguf
```

Media processing uses the bundled portable **FFmpeg `9.0.2` full build** (`bin/ffmpeg/ffmpeg.exe` and `ffprobe.exe` with Vulkan and `libplacebo` support). Install current GPU drivers with Vulkan support for Taichi, Qwen3.5-2B, and Vulkan Video; NVIDIA NVENC modes require a supported NVIDIA GPU.

---

## 🎬 Export Modes

### NVIDIA NVENC H.264

Fast hardware-encoded H.264 export.

Best for:

*   general sharing;
*   YouTube/TikTok/Discord workflows;
*   fast iteration.

### NVIDIA NVENC HEVC (H.265)

Hardware-encoded H.265/HEVC export.

Best for:

*   smaller files;
*   archival previews;
*   modern playback devices.

### Vulkan Video H.264 / H.265

Hardware-encoded H.264 or H.265 (`hvc1` tagged) via FFmpeg Vulkan Video (`cqp`, QP 18, high-quality tuning) on a discrete GPU whose driver passes a real encoding probe.

Best for:

*   cross-vendor discrete GPUs supporting Vulkan Video encoding;
*   high-quality hardware exports without relying on NVENC.

### CPU (H.264)

Software encoding with `libx264`.

Best for:

*   systems without hardware video encoding;
*   compatibility fallback.

### ProRes 422 HQ

Encodes selected source intervals directly to 10-bit 4:2:2 ProRes 422 HQ (`.mov`) with exact frame counts, speed control, and PCM 24-bit audio, using Vulkan compute encoding on supported discrete GPUs with CPU fallback. Stream-copy assembly avoids re-encoding.

Best for:

*   editing handoff;
*   frame-perfect workflows;
*   high-quality 10-bit 4:2:2 video.

> ProRes files are larger. The UI creates an H.264/AAC preview for the Gradio video player while keeping the `.mov` output available to download.

---

*If BeatSync Engine saves you editing time, give the repository a star! ⭐*
