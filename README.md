# 🎵 BeatSync Engine

**AI-powered music video creator that automatically synchronizes video clips to the rhythm of your music with frame-perfect precision.**

<br>

<p align="center">
<img width="1418" height="1036" alt="image" src="https://github.com/user-attachments/assets/69542c8c-752c-4419-ab2f-18ac12300699" />
  <br/>
  <em>Create dynamic, beat-matched videos in just a few clicks.</em>
</p>

<p align="center">
  <a href="#-key-features"><strong>Key Features</strong></a> ·
  <a href="#-generation-modes-in-detail"><strong>Generation Modes</strong></a> ·
  <a href="#-installation"><strong>Installation</strong></a> ·
  <a href="#-how-to-use"><strong>How to Use</strong></a> ·
  <a href="#-command-line-interface-cli"><strong>CLI</strong></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue.svg" alt="Python 3.13">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/platform-Windows-lightgrey.svg" alt="Platform: Windows">
  <img src="https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia" alt="CUDA Support">
</p>

---

**BeatSync Engine** is a powerful desktop application designed to eliminate the tedious work of manual video editing. By leveraging advanced audio analysis and robust video processing, it intelligently analyzes any song, detects its core rhythmic structure, and automatically cuts and assembles video clips to create a seamless, professional-quality music video.

Whether you are a content creator, musician, or hobbyist, BeatSync Engine provides the tools you need to produce stunning visual content that is perfectly synchronized to your audio.

## ✨ Key Features

### Core Intelligence
*   **🤖 Intelligent Auto Mode**: Analyzes song structure (intro, verse, chorus), energy levels, and rhythm patterns to create optimal, context-aware cuts automatically.
*   **🧠 Smart Multi-Band Analysis**: Detects kick drums, snares/claps, and hi-hats across the frequency spectrum for precise, rhythm-based editing using presets.
*   **⚙️ Simple Manual Mode**: Provides direct control over cut frequency based on bass-heavy beats for simple and consistent results.

### Performance & Quality
*   **🎯 Frame-Perfect Processing**: Utilizes FFmpeg with exact frame calculations, ensuring **zero timing drift** and perfect synchronization.
*   **⚡ GPU Acceleration**: Supports **NVIDIA CUDA** for up to 10x faster audio analysis and **NVENC** for 2-3x faster video encoding.
*   **🎬 Lossless ProRes Workflow**: An optional "Precise Mode" converts source videos to **Apple ProRes 422 Proxy**, performs frame-perfect cuts, and concatenates them losslessly for maximum quality.
*   **🚀 Multi-Core Optimized**: Employs parallel processing to render video clips simultaneously, maximizing CPU and GPU efficiency.

### User Experience
*   **📦 Fully Portable**: Ships with its own embedded Python, CUDA, and FFmpeg. **No installation or dependencies required.** Just download, unzip, and run.
*   **🌐 Intuitive Web UI**: A clean and simple interface powered by Gradio makes video creation accessible to everyone.
*   **🔧 Advanced Customization**: Fine-tune your videos with controls for playback direction (forward, reverse, random), speed, timing offsets, and custom frame rates.

## 🤖 Generation Modes in Detail

BeatSync Engine offers three distinct modes to fit any workflow, from fully automated to manually controlled.

| Mode | Description | Best For |
| :--- | :--- | :--- |
| **🤖 Auto** | **Extreme Intelligence.** The most advanced mode. It performs a deep analysis of the song's structure, energy, and rhythm. It automatically varies cut density—more cuts in high-energy choruses, fewer in calm intros—and intelligently follows the dominant instruments. | **Effortless, optimal results.** Just upload your files and let the AI do the work. Perfect for complex music with varying tempo and energy. |
| **🧠 Smart** | **Preset-Based Rhythm.** Analyzes kick, clap, and hi-hat frequencies and uses presets (e.g., *'Slower', 'Normal', 'Hyper'*) to determine which beats to cut on. Offers a great balance between control and automation. | **Professional and consistent results.** Ideal for EDM, pop, and rock tracks where the kick/snare pattern is the driving force. |
| **⚙️ Manual** | **Simple Bass Focus.** Detects beats in the bass frequency range (20-200 Hz). A single "Cut Intensity" slider lets you decide whether to use every beat, skip beats, or even subdivide beats for more rapid cuts. | **Quick projects and simple control.** Great for hip-hop, lo-fi, or any genre with a clear and consistent bass line. |

## 🛠️ Installation

BeatSync Engine is designed to be completely portable and requires no setup.

**Prerequisites:**
*   **OS**: Windows 10/11
*   **GPU (Optional but Recommended)**: An NVIDIA GPU is required for CUDA (audio) and NVENC (video) acceleration. The application will run in CPU-only mode otherwise.

**Instructions:**
1.  Go to the [**Releases**](https://github.com/BeatSync/beatsync-engine/releases) page.
2.  Download the latest `BeatSync.Engine.vX.X.zip` file.
3.  Unzip the archive to your desired location.
4.  Run `run.bat` to start the application. Your browser will automatically open the user interface.

## 🚀 How to Use

1.  **Launch the App**: Double-click `run.bat`.
2.  **Upload Files**:
    *   Click to upload your **Audio File** (`.mp3`, `.wav`, `.flac`).
    *   Click to upload one or more **Video Files** (`.mp4`, `.mkv`).
3.  **Choose Generation Mode**:
    *   **🤖 Auto**: The recommended set-and-forget option.
    *   **🧠 Smart**: Select a cut frequency preset.
    *   **⚙️ Manual**: Adjust the cut intensity slider.
4.  **Configure Settings (Optional)**:
    *   **Video Direction**: `forward`, `backward`, or `random`.
    *   **Processing Mode**:
        *   `NVIDIA NVENC`: For fast, high-quality GPU encoding.
        *   `ProRes 422 Proxy`: For frame-perfect, lossless quality.
        *   `CPU (H.264)`: If you don't have an NVIDIA GPU.
    *   **Output Filename**: Set a custom name for your video.
5.  **Create Video**: Click the **"🎬 Create Music Video"** button and watch the progress in the console and UI.

Your final video will be saved in the `output` folder.

## 🖥️ Command-Line Interface (CLI)

For power users and automation, BeatSync Engine can be fully controlled via the command line.

**Usage:**
```bash
python video_processor.py <mp3_file> <video_directory> <cut_intensity> [options]
```

**Example (Smart Mode):**
```bash
python video_processor.py "C:\Music\my_song.mp3" "D:\VideoClips" "normal" --mode smart --gpu -o "my_smart_video.mp4"
```

**Example (Auto Mode):**
```bash
python video_processor.py "C:\Music\another_track.wav" "D:\VideoClips" "auto" --mode auto --gpu --lossless -o "prores_auto_video.mov"
```

**Example (Manual Mode):**
```bash
python video_processor.py "C:\Music\bass_heavy.mp3" "D:\VideoClips" "2.0" --mode manual --offset -0.05 -o "manual_video.mp4"
```
Run `python video_processor.py -h` for a full list of commands and options.

## 📁 Technology Stack

*   **Backend**: Python
*   **Audio Analysis**: `librosa`
*   **GPU Computing**: `cupy` (for CUDA)
*   **Video Processing**: `ffmpeg`
*   **Web UI**: `gradio`
*   **Core Numerics**: `numpy`

## ❤️ Acknowledgments
This project stands on the shoulders of giants. A huge thank you to the developers of the incredible open-source libraries that make BeatSync Engine possible.
