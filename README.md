# 🎵 BeatSync Engine - AI Audio-Visual Music Video Generator

[![Windows](https://img.shields.io/badge/Platform-Windows-0078D4?style=flat-square&logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![Python](https://img.shields.io/badge/Python-Portable%203.13-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/NVIDIA-CUDA%2013.2-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![NVENC](https://img.shields.io/badge/Encoding-NVENC-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-Portable-007808?style=flat-square&logo=ffmpeg&logoColor=white)](https://ffmpeg.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-F97316?style=flat-square)](https://www.gradio.app/)
[![Qwen3--VL](https://img.shields.io/badge/Vision-Qwen3--VL-blueviolet?style=flat-square)](https://huggingface.co/Qwen)
[![vLLM](https://img.shields.io/badge/Server-vLLM-000000?style=flat-square)](https://docs.vllm.ai/)
[![Downloads](https://img.shields.io/github/downloads/Merserk/BeatSync-Engine/total.svg?style=flat-square)](https://github.com/Merserk/BeatSync-Engine/releases)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue?style=flat-square)](https://www.gnu.org/licenses/agpl-3.0.html)

A portable Windows app that creates beat-synchronized **AMV/GMV/music videos** from one audio track and one or more source videos.

BeatSync Engine analyzes music rhythm, energy, sections, and source-video moments, then builds a frame-locked edit timeline with optional **Qwen3-VL semantic scene matching**.







> **Goal:** upload audio + video clips, click one button, and get a finished rhythmic music video with clean beat cuts, strong source-moment selection, and GPU-accelerated rendering when available.

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
*   **📦 Portable Runtime:** Designed around bundled Python, CUDA, FFmpeg, vLLM runtime, and local model folders.
*   **🌐 Local Web UI:** Launches a Gradio interface at `http://127.0.0.1:7860` by default.
*   **🧹 Local Caches:** Reuses visual analysis data so repeated runs can be faster.

---

## ✅ Support Matrix

| Platform | GPU / Mode | Status | Notes |
|---|---:|---:|---|
| **Windows 10/11 x64** | **NVIDIA CUDA + NVENC** | ✅ Recommended | Fastest path: GPU audio/visual acceleration, Qwen3-VL, and hardware encoding. |
| **Windows 10/11 x64** | **CPU H.264** | ✅ Supported | Works without NVIDIA acceleration, but video analysis and rendering are slower. |
| **Windows 10/11 x64** | **ProRes 422 Proxy** | ✅ Supported | Frame-perfect precise workflow. Larger `.mov` files. |
| **Windows 10/11 x64** | **AMD / Intel GPU** | ⚠️ CPU fallback | GPU acceleration is NVIDIA/CUDA-focused. AMD/Intel can still use CPU processing. |
| **Linux / WSL2** | **Manual run** | ⚠️ Not packaged target | Code may be portable, but the included launcher/runtime layout is Windows-focused. |

### Media support

| Input / Output | Supported |
|---|---|
| **Audio input** | `.mp3`, `.wav`, `.flac` |
| **Video input** | `.mp4`, `.mkv` |
| **GUI output** | `.mp4` for H.264/H.265, `.mov` for ProRes |
| **CLI output** | `.mkv` by default, `.mov` with `--lossless` |

---

## 📦 Portable Folder Layout

A complete portable release is expected to look like this:

```text
BeatSync Engine/
├── run.bat                                  # One-click Windows launcher
├── bin/
│   ├── python-3.13.13-embed-amd64/          # Main portable Python runtime
│   ├── CUDA/v13.2/                          # Portable CUDA runtime files
│   ├── ffmpeg/ffmpeg.exe                    # Portable FFmpeg
│   ├── ffmpeg/ffprobe.exe                   # Portable FFprobe
│   ├── vllm/
│   │   └── python-3.12.10-embed-amd64/      # Separate Python runtime for vLLM
│   └── models/
│       └── Qwen3-VL-2B-Instruct/            # Local vision-language model
├── src/
│   ├── gui.py                               # Gradio web UI
│   ├── video_processor.py                   # CLI + rendering pipeline
│   ├── video_analysis.py                    # Source-video visual library
│   ├── ffmpeg_processing.py                 # FFmpeg/FFprobe helpers
│   ├── start_vllm_server.py                 # Local vLLM launcher
│   ├── auto_mode/
│   │   ├── __init__.py                      # Auto Mode pipeline entry point
│   │   ├── stage1_audio.py                  # Beat grid detection
│   │   ├── stage2_features.py               # Energy/rhythm features
│   │   ├── stage3_sections.py               # Music section analysis
│   │   ├── stage4_select.py                 # Rhythmic cut selection
│   │   ├── stage5_qwen_scene_worker.py      # Qwen/vLLM semantic worker
│   │   └── stage6_av_planner.py             # Audio-visual clip planner
│   └── ui_content.py                        # UI labels and status text
├── input/
│   ├── audio/                               # Latest uploaded audio copy
│   ├── video/                               # Latest uploaded source videos
│   ├── processing/                          # Temporary processing files
│   ├── gradio_uploads/                      # Gradio upload/session temp files
│   └── video_analysis_cache/                # Visual analysis + vLLM logs/cache
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

3.  Start the app:

    ```bat
    run.bat
    ```

4.  The browser opens automatically:

    ```text
    http://127.0.0.1:7860
    ```

5.  Upload:

    *   one audio file: `.mp3`, `.wav`, or `.flac`;
    *   one or more source videos: `.mp4` or `.mkv`.

6.  Choose a processing mode.

7.  Click:

    ```text
    🎬 Create Music Video
    ```

8.  Finished videos are saved to:

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

## 🧠 Automatic Processing Flow

BeatSync Engine follows a fixed audio-to-video pipeline:

```text
Start
├── Launch portable Python from run.bat
├── Configure portable CUDA, FFmpeg, and UTF-8 console
├── Start local Gradio UI on 127.0.0.1
├── User uploads audio + videos
├── Stage 1: Detect stable beat grid
├── Stage 2: Extract energy and rhythm features
├── Stage 3: Detect broad musical sections
├── Stage 4: Select deliberate rhythmic cuts
├── Stage 5: Build visual library
│   ├── Read source-video metadata
│   ├── Detect scene changes and motion/quality windows
│   ├── Reuse cached source analysis when possible
│   └── Optionally tag best moments with Qwen3-VL via vLLM
├── Stage 6: Build and render the final video
│   ├── Quantize cut boundaries to output frames
│   ├── Plan source clips for each music segment
│   ├── Extract clips with FFmpeg
│   ├── Concatenate clips
│   └── Add the music track
└── Save final output to output/
```

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

### Stage 5 - Video analysis + Qwen

Builds a visual library from the uploaded source videos.

Deterministic analysis reads:

*   scene changes;
*   motion strength;
*   visual quality;
*   action score;
*   beauty score;
*   reusable candidate moments.

Optional Qwen3-VL analysis adds semantic tags like:

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

## 🧠 Qwen3-VL + vLLM Notes

BeatSync can use a local Qwen3-VL model through a local OpenAI-compatible vLLM server.

Default local endpoint:

```text
http://127.0.0.1:8000/v1
```

Default model folder:

```text
bin/models/Qwen3-VL-2B-Instruct/
```

Default vLLM runtime:

```text
bin/vllm/python-3.12.10-embed-amd64/python.exe
```

vLLM logs are written to:

```text
input/video_analysis_cache/vllm_server.log
```

### Disable Qwen semantic tags

Use this if VRAM is low or you only want deterministic visual analysis:

```bat
set BEATSYNC_DISABLE_QWEN=1
run.bat
```

### Use an external vLLM server

```bat
set BEATSYNC_VLLM_BASE_URL=http://127.0.0.1:8000/v1
set BEATSYNC_VLLM_MODEL=Qwen3-VL-2B-Instruct
run.bat
```

### Why Windows Firewall may ask for permission

The local vLLM server opens a local TCP port. It is designed for local access on `127.0.0.1`. If Windows asks whether Python can access public/private networks, this is usually the embedded vLLM Python process.

For local-only use, you normally do **not** need to expose it to the network.

---

## 🖥️ Command Line Usage

The web UI is recommended, but the core renderer can also be run directly:

```bat
bin\python-3.13.13-embed-amd64\python.exe -X utf8 src\video_processor.py "song.mp3" "input\video" -o "output_music_video.mkv" --gpu
```

### CPU render

```bat
bin\python-3.13.13-embed-amd64\python.exe -X utf8 src\video_processor.py "song.mp3" "input\video" -o "output_music_video.mkv"
```

### NVIDIA H.264

```bat
bin\python-3.13.13-embed-amd64\python.exe -X utf8 src\video_processor.py "song.mp3" "input\video" --gpu --gpu-encoder h264_nvenc
```

### NVIDIA HEVC

```bat
bin\python-3.13.13-embed-amd64\python.exe -X utf8 src\video_processor.py "song.mp3" "input\video" --gpu --gpu-encoder hevc_nvenc
```

### ProRes precise mode

```bat
bin\python-3.13.13-embed-amd64\python.exe -X utf8 src\video_processor.py "song.mp3" "input\video" --lossless -o "output_music_video.mov"
```

### Render only part of a song

```bat
bin\python-3.13.13-embed-amd64\python.exe -X utf8 src\video_processor.py "song.mp3" "input\video" --start-time 30 --end-time 90
```

### Custom FPS

```bat
bin\python-3.13.13-embed-amd64\python.exe -X utf8 src\video_processor.py "song.mp3" "input\video" --fps 60
```

---

## 🧩 Advanced Options

Set environment variables before running `run.bat`.

### Change Gradio port

```bat
set GRADIO_SERVER_PORT=7870
run.bat
```

### Disable Qwen

```bat
set BEATSYNC_DISABLE_QWEN=1
run.bat
```

### Limit Qwen windows

```bat
set BEATSYNC_QWEN_MAX_WINDOWS=60
run.bat
```

Use `0` to skip Qwen semantic analysis:

```bat
set BEATSYNC_QWEN_MAX_WINDOWS=0
run.bat
```

### Change Qwen batch size

```bat
set BEATSYNC_QWEN_BATCH_SIZE=4
run.bat
```

Useful for low-VRAM GPUs.

### Change vLLM GPU memory fraction

```bat
set BEATSYNC_VLLM_GPU_MEMORY=0.80
run.bat
```

### Change vLLM model length

```bat
set BEATSYNC_VLLM_MAX_MODEL_LEN=4096
run.bat
```

### Change vLLM startup timeout

```bat
set BEATSYNC_VLLM_START_TIMEOUT=900
run.bat
```

### Control visual-analysis workers

```bat
set BEATSYNC_VIDEO_ANALYSIS_WORKERS=2
run.bat
```

### Control NVENC clip workers

```bat
set BEATSYNC_NVENC_CLIP_WORKERS=4
run.bat
```

### Enable experimental GPU candidate metrics

```bat
set BEATSYNC_GPU_CANDIDATE_METRICS=1
run.bat
```

### Release CuPy memory aggressively

```bat
set BEATSYNC_RELEASE_GPU_MEMORY=1
run.bat
```

---

## 🧪 Verify the Portable Runtime

### Check main Python

```bat
bin\python-3.13.13-embed-amd64\python.exe -X utf8 --version
```

### Check FFmpeg

```bat
bin\ffmpeg\ffmpeg.exe -version
```

### Check NVENC support

```bat
bin\ffmpeg\ffmpeg.exe -encoders | findstr nvenc
```

Expected useful entries:

```text
h264_nvenc
hevc_nvenc
```

### Check CUDA from main Python

```bat
bin\python-3.13.13-embed-amd64\python.exe -X utf8 -c "import cupy as cp; print(cp.cuda.runtime.getDeviceProperties(0)['name'].decode())"
```

### Check vLLM dry run

```bat
bin\python-3.13.13-embed-amd64\python.exe -X utf8 src\start_vllm_server.py --dry-run
```

---

## 📁 Cache and Output Locations

| Folder | Purpose |
|---|---|
| `input/audio/` | Latest audio uploaded through the UI. |
| `input/video/` | Latest source videos uploaded through the UI. |
| `input/processing/` | Temporary render workspace. |
| `input/gradio_uploads/` | Gradio upload and session files. |
| `input/video_analysis_cache/` | Cached visual library JSON files and `vllm_server.log`. |
| `output/` | Final rendered videos. |

### Clear visual-analysis cache

```bat
rmdir /s /q input\video_analysis_cache
```

The folder will be recreated on the next run.

### Clear uploaded input copies

```bat
rmdir /s /q input\audio
rmdir /s /q input\video
```

The app will recreate them automatically.

---

## 🚀 Performance Tips

### Best performance setup

*   NVIDIA RTX GPU with enough VRAM for Qwen3-VL.
*   NVENC H.264 for fastest exports.
*   Source videos on an SSD.
*   Reuse the same source videos to benefit from `input/video_analysis_cache/`.
*   Leave FPS empty unless you specifically need 24, 30, or 60 FPS.

### Low VRAM setup

Try these options:

```bat
set BEATSYNC_QWEN_BATCH_SIZE=2
set BEATSYNC_QWEN_MAX_WINDOWS=40
set BEATSYNC_VLLM_GPU_MEMORY=0.70
run.bat
```

Or disable Qwen completely:

```bat
set BEATSYNC_DISABLE_QWEN=1
run.bat
```

### CPU-only setup

Use **CPU H.264** mode in the UI. It is slower but avoids CUDA/NVENC requirements.

---

## 🛟 Troubleshooting

### Browser does not open

Open the local URL manually:

```text
http://127.0.0.1:7860
```

If port `7860` is busy, BeatSync searches nearby ports. Check the CMD window for the exact URL.

### Windows Firewall asks about Python

This can happen when the local Gradio server or local vLLM server starts.

BeatSync binds to local host by default:

```text
127.0.0.1
```

For normal local use, you do not need a public network rule.

### `No MP4/MKV files found`

The CLI expects a folder containing `.mp4` or `.mkv` files:

```bat
src\video_processor.py "song.mp3" "input\video"
```

Make sure the second argument is a folder, not a single video file.

### Qwen or vLLM does not start

Check:

```text
input/video_analysis_cache/vllm_server.log
```

Also verify these paths exist:

```text
bin/vllm/python-3.12.10-embed-amd64/python.exe
bin/models/Qwen3-VL-2B-Instruct/
```

To continue without Qwen:

```bat
set BEATSYNC_DISABLE_QWEN=1
run.bat
```

### CUDA works, but NVENC is missing

Check FFmpeg encoders:

```bat
bin\ffmpeg\ffmpeg.exe -encoders | findstr nvenc
```

If no NVENC encoder appears, use **CPU H.264** or replace FFmpeg with a build that includes NVIDIA NVENC.

### Out of VRAM during Qwen stage

Lower Qwen load:

```bat
set BEATSYNC_QWEN_BATCH_SIZE=2
set BEATSYNC_QWEN_MAX_WINDOWS=40
set BEATSYNC_VLLM_GPU_MEMORY=0.70
run.bat
```

Or disable Qwen:

```bat
set BEATSYNC_DISABLE_QWEN=1
run.bat
```

### Out of VRAM during render stage

Use fewer parallel NVENC workers:

```bat
set BEATSYNC_NVENC_CLIP_WORKERS=1
run.bat
```

You can also use CPU H.264 mode.

### `ConnectionResetError: [WinError 10054]`

This is usually harmless Windows asyncio/vLLM cleanup noise after a local server or browser connection closes. If the output video was created successfully, the render is complete.

### Generated video is too large

Use NVENC HEVC for smaller output, or avoid ProRes unless you need a precise editing format.

### Cuts feel too fast

Use calmer source music or adjust the Auto Mode configuration in:

```text
src/auto_mode/__init__.py
```

Relevant tuning values include low/medium/high/peak minimum intervals and global cut-ratio limits.

---

## 🔄 Updating BeatSync Engine

For a portable release, update by replacing the application folder with a newer version.

To keep old renders, copy this folder before replacing:

```text
output/
```

To keep source-video analysis cache, copy:

```text
input/video_analysis_cache/
```

If you changed tuning values, back up:

```text
src/auto_mode/__init__.py
```

---

## 🧹 Uninstall

BeatSync is portable. To uninstall it, close the app and delete the folder:

```text
BeatSync Engine/
```

This does not remove NVIDIA drivers or other system-level GPU components.

---

## 📖 Project Lineage & Components

BeatSync Engine connects several pieces into one local workflow:

* 🟦 **BeatSync Engine**<br>
  The Auto Mode pipeline that analyzes audio, source videos, cut timing, and final clip planning.

* 🟧 **Gradio**<br>
  The local web interface used for uploads, status, preview, and export controls.

* 🟩 **NVIDIA CUDA / CuPy**<br>
  Optional acceleration path for GPU-enabled analysis.

* 🟩 **NVIDIA NVENC**<br>
  Hardware encoding path for fast H.264/H.265 exports.

* 🟢 **FFmpeg / FFprobe**<br>
  Media probing, frame-accurate extraction, clip conversion, concatenation, and final audio muxing.

* 🧠 **Qwen3-VL**<br>
  Optional semantic scene understanding for stronger action, beauty, emotion, and use-case matching.

* ⚙️ **vLLM**<br>
  Local OpenAI-compatible server used to run Qwen3-VL during the visual-analysis stage.

* 🎼 **Librosa**<br>
  Audio feature extraction for beat, rhythm, energy, and structure analysis.

---

## 🤝 Credits

*   **FFmpeg:** Media processing framework by the [FFmpeg project](https://ffmpeg.org/).
*   **Gradio:** Local web UI framework by [Gradio](https://www.gradio.app/).
*   **Librosa:** Python audio analysis library by the [librosa project](https://librosa.org/).
*   **CuPy:** NumPy-compatible GPU array library by the [CuPy team](https://cupy.dev/).
*   **NVIDIA CUDA / NVENC:** GPU runtime and hardware video encoding by [NVIDIA](https://developer.nvidia.com/).
*   **Qwen3-VL:** Vision-language model family by [Qwen](https://huggingface.co/Qwen).
*   **vLLM:** High-throughput local model server by the [vLLM project](https://docs.vllm.ai/).
*   **Python:** Runtime used by the portable application.

---

## 📜 License

BeatSync Engine is released under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

See the `LICENSE` file in the repository for the full license text.

---

## ⚠️ Disclaimer

BeatSync Engine is a local media-processing tool. Rendering speed, output quality, and Qwen/vLLM behavior depend on your source videos, audio quality, GPU, VRAM, FFmpeg build, and model files.

Only process media you have the right to use. Large video files and AI model inference can consume significant disk space, CPU, GPU, and VRAM.

---

*If BeatSync Engine saves you editing time, give the repository a star! ⭐*
