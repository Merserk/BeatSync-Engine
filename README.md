# BeatSync Engine - AI Audio-Visual Music Video Generator

[![Windows](https://img.shields.io/badge/Platform-Windows-0078D4?style=flat-square&logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![Python](https://img.shields.io/badge/Python-Portable%203.13-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/NVIDIA-CUDA%2013.0-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![NVENC](https://img.shields.io/badge/Encoding-NVENC-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-Portable-007808?style=flat-square&logo=ffmpeg&logoColor=white)](https://ffmpeg.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-F97316?style=flat-square)](https://www.gradio.app/)
[![Qwen3--VL](https://img.shields.io/badge/Vision-Qwen3--VL-blueviolet?style=flat-square)](https://huggingface.co/Qwen)
[![Downloads](https://img.shields.io/github/downloads/Merserk/BeatSync-Engine/total.svg?style=flat-square)](https://github.com/Merserk/BeatSync-Engine/releases)

A portable Windows app that creates beat-synchronized **AMV/GMV/music videos** from one audio track and one or more source videos.

BeatSync Engine analyzes music rhythm, energy, sections, and source-video moments, then builds a frame-locked edit timeline with optional **Qwen3-VL semantic scene matching**.

<img width="1402" height="743" alt="image" src="https://github.com/user-attachments/assets/c7323c0a-a49d-4596-adcc-ec0e8e23b774" />

> **Goal:** upload audio + video clips, click one button, and get a finished rhythmic music video with clean beat cuts, strong source-moment selection, and GPU-accelerated rendering when available.

---

## 🎥 Demo Video

[![Watch the BeatSync Engine demo video](https://img.youtube.com/vi/Iv00-5GkoOM/maxresdefault.jpg)](https://www.youtube.com/watch?v=Iv00-5GkoOM)

> Watch the special showcase video for BeatSync Engine on YouTube.

---

## ✨ Features

*   **🎵 Automatic Beat Editing:** Detects a stable beat grid and cuts source footage to the music rhythm.
*   **🌊 Energy-Wave Cut Density:** Calm sections hold longer; drops, impacts, and high-energy parts cut faster.
*   **🎼 Song Structure Detection:** Finds broad intro, verse, chorus, bridge, drop, build, body, and outro-style sections.
*   **🥁 Rhythm Feature Analysis:** Reads kick, clap, bass, hi-hat, novelty, impact, bar anchors, and phrase anchors.
*   **🎬 Source Video Moment Library:** Scans source videos for motion, quality, scene changes, action, beauty, and usable moments.
*   **🧠 Qwen3-VL Semantic Tags:** Optional local vision-language tagging for action, combat, chase, beauty, emotion, drops, builds, and soft moments.
*   **🎯 Audio-Visual Planner:** Chooses planned source moments instead of relying only on random clip sampling.
*   **⚡ NVIDIA GPU Acceleration:** Uses CuPy/CUDA for analysis when available and FFmpeg NVENC for fast H.264/H.265 encoding.
*   **🎞️ Frame-Locked Timeline:** Quantizes cut boundaries to absolute output frames to avoid timing drift.
*   **🎬 Multiple Export Modes:** NVENC H.264, NVENC HEVC, CPU H.264, and ProRes 422 Proxy precise mode.
*   **📦 Portable Runtime:** Designed around bundled Python, CUDA, FFmpeg, Transformers, and local model folders.
*   **🌐 Local Web UI:** Launches a Gradio interface at `http://127.0.0.1:7860` by default.
*   **🧹 Local Caches:** Reuses visual analysis data so repeated runs can be faster.

---

## 📦 Portable Folder Layout

A complete portable release is expected to look like this:

```text
BeatSync Engine/
├── install.bat                              # One-click Windows installer
├── run.bat                                  # One-click Windows launcher
├── requirements.txt                         # Python package requirements
├── scripts/
│   └── install.ps1                          # Portable runtime/model installer
├── bin/
│   ├── python-3.13.13-embed-amd64/          # Main portable Python runtime
│   ├── CUDA/v13.0/                          # Portable CUDA runtime files
│   ├── ffmpeg/ffmpeg.exe                    # Portable FFmpeg
│   ├── ffmpeg/ffprobe.exe                   # Portable FFprobe
│   └── models/
│       └── Qwen3-VL-2B-Instruct/            # Local vision-language model
├── src/
│   ├── gui.py                               # Gradio web UI
│   ├── video_processor.py                   # Rendering pipeline
│   ├── video_analysis.py                    # Source-video visual library
│   ├── ffmpeg_processing.py                 # FFmpeg/FFprobe helpers
│   ├── auto_mode/
│   │   ├── __init__.py                      # Auto Mode pipeline entry point
│   │   ├── stage1_audio.py                  # Beat grid detection
│   │   ├── stage2_features.py               # Energy/rhythm features
│   │   ├── stage3_sections.py               # Music section analysis
│   │   ├── stage4_select.py                 # Rhythmic cut selection
│   │   ├── stage5_qwen_scene_worker.py      # Qwen/Transformers semantic worker
│   │   └── stage6_av_planner.py             # Audio-visual clip planner
│   └── ui_content.py                        # UI labels and status text
├── input/
│   ├── audio/                               # Latest uploaded audio copy
│   ├── video/                               # Latest uploaded source videos
│   ├── processing/                          # Temporary processing files
│   ├── gradio_uploads/                      # Gradio upload/session temp files
│   └── video_analysis_cache/                # Visual analysis cache
└── output/                                  # Final exported videos
```

> The app creates `input/` and `output/` subfolders automatically if they are missing.

---

## 🛠️ Quick Start - Windows Portable

1.  Download or extract BeatSync Engine.

2.  Keep the folder path simple, for example:

    ```text
    D:\AI\BeatSync Engine\
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

BeatSync copies the latest files into:

```text
input/audio/
input/video/
```

This keeps processing local and avoids depending on browser upload temp paths.

### 2. Pick FPS

Leave **Custom FPS** empty to auto-detect FPS from the first source video.

Common values:

```text
24
30
60
```

### 3. Choose processing mode

| Mode | Output | Best for |
|---|---:|---|
| **NVIDIA NVENC H.264** | `.mp4` | Fast, compatible exports. |
| **NVIDIA NVENC HEVC** | `.mp4` | Smaller files, slower compatibility on old players. |
| **CPU H.264** | `.mp4` | Systems without NVENC. |
| **ProRes 422 Proxy** | `.mov` | Frame-perfect precise workflow and editing handoff. |

### 4. Create the video

The CMD window shows clean stage progress like:

```text
Stage 1 processing started:
  Beat grid: 259 beats at 152.0 BPM
Stage 1 ended in 8 seconds.

Stage 5 processing started:
  Source videos: 1
  Qwen: enabled
  Qwen tags: 116/116 in 97.5s
  Visual library: 804 visual moments, action=0.54, beauty=0.47, quality=0.62
Stage 5 ended in 192 seconds.

Total time processing: 217 seconds
```

The UI shows a preview and success stats after the render finishes.

---

## 🎵 Auto Mode Details

Auto Mode is the main creative engine.

### Stage 1 - Beat grid

Detects stable beat positions and tempo from the percussive part of the song.

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

### Stage 5 - Video analysis + Qwen3-VL

Builds a visual library from the uploaded source videos.

Deterministic analysis reads:

*   scene changes;
*   motion strength;
*   visual quality;
*   action score;
*   beauty score;
*   reusable candidate moments.

Optional local Qwen3-VL analysis through Transformers adds semantic tags like:

```text
action
combat
chase
explosion
character_focus
camera_motion
emotion
recommended_use
visual_quality
```

### Stage 6 - Audio-visual render

The renderer creates a frame-accurate cut timeline, chooses source moments for each segment, extracts clips with FFmpeg, concatenates the result, and adds the music track.

---

## 🎬 Export Modes

### NVIDIA NVENC H.264

Fast hardware-encoded H.264 export.

Best for:

*   general sharing;
*   YouTube/TikTok/Discord workflows;
*   fast iteration.

### NVIDIA NVENC HEVC

Hardware-encoded H.265/HEVC export.

Best for:

*   smaller files;
*   archival previews;
*   modern playback devices.

### CPU H.264

Software encoding with `libx264`.

Best for:

*   systems without NVIDIA NVENC;
*   compatibility fallback.

### ProRes 422 Proxy

Precise workflow that converts input videos to ProRes 422 Proxy, extracts frame-counted segments, concatenates, and adds the music track.

Best for:

*   editing handoff;
*   frame-perfect workflows;
*   maximum timeline stability.

> ProRes files are larger. The app can create an H.264 preview for the Gradio video player while keeping the `.mov` output.

---

*If BeatSync Engine saves you editing time, give the repository a star! ⭐*
