#!/usr/bin/env python3
import asyncio
import asyncio.proactor_events
import base64
import datetime
import html
import multiprocessing
import os
import re
import shutil
import socket
import subprocess
import sys
from urllib.parse import unquote, urlparse
from pathlib import Path
from typing import Dict, List, Tuple, TypeAlias

import gradio as gr
import librosa

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from runtime_env import configure_portable_runtime

RUNTIME = configure_portable_runtime(SCRIPT_DIR)
PORTABLE_CUDA_DIR = RUNTIME.portable_cuda_dir
USING_PORTABLE_CUDA = RUNTIME.using_portable_cuda
PORTABLE_PYTHON_DIR = RUNTIME.portable_python_dir
PORTABLE_PYTHON_EXE = RUNTIME.portable_python_exe
USING_PORTABLE_PYTHON = RUNTIME.using_portable_python

if USING_PORTABLE_CUDA and PORTABLE_CUDA_DIR:
    print(f"Using Portable CUDA: {PORTABLE_CUDA_DIR}")
else:
    print(f"Portable CUDA not found under: {os.path.join(SCRIPT_DIR, 'bin', 'CUDA')}")
    print("   Will try to use system CUDA if available")
if RUNTIME.cuda_notice:
    print(f"   Note: {RUNTIME.cuda_notice}")

if USING_PORTABLE_PYTHON and PORTABLE_PYTHON_EXE:
    print(f"Using Portable Python: {PORTABLE_PYTHON_EXE}")
else:
    print("Portable Python not found, using system Python")

from ffmpeg_processing import DEFAULT_STANDARD_QUALITY, FFMPEG_FOUND, FFMPEG_PATH, check_nvenc_support, create_lossless_delivery_mp4, get_video_fps, get_video_resolution, is_browser_playable_video, normalize_quality_profile
from video_processor import CPU_COUNT, MAX_THREADS, PARALLEL_WORKERS, create_music_video, estimate_threads_per_job, get_local_temp_dir, get_video_files
from manual_mode import analyze_beats_manual, process_manual_intensity
from smart_mode import analyze_beats_smart, get_gpu_info, get_preset_info, is_gpu_available, select_beats_smart, set_gpu_mode
from auto_mode import analyze_beats_auto
from ui_content import *

GPU_RUNTIME_AVAILABLE = is_gpu_available()
CPU_ONLY_MODE = not GPU_RUNTIME_AVAILABLE
NVENC_AVAILABLE = GPU_RUNTIME_AVAILABLE and check_nvenc_support()
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEFAULT_GUI_PORT = 7860
RESOLUTION_PRESET_CHOICES = [
    ("Default (match first source video)", "default"),
    ("16:9 | 1280x720 (HD)", "1280x720"),
    ("16:9 | 1920x1080 (Full HD)", "1920x1080"),
    ("16:9 | 2560x1440 (QHD)", "2560x1440"),
    ("16:9 | 3840x2160 (4K UHD)", "3840x2160"),
    ("21:9 | 2560x1080 (UltraWide HD)", "2560x1080"),
    ("21:9 | 3440x1440 (UltraWide QHD)", "3440x1440"),
    ("21:9 | 3840x1600 (UltraWide 1600p)", "3840x1600"),
    ("9:16 | 720x1280 (Vertical HD)", "720x1280"),
    ("9:16 | 1080x1920 (Vertical Full HD)", "1080x1920"),
    ("9:16 | 1440x2560 (Vertical QHD)", "1440x2560"),
]

if hasattr(gr, "set_static_paths"):
    gr.set_static_paths(paths=[OUTPUT_DIR])

print(
    get_startup_header(
        CPU_COUNT,
        estimate_threads_per_job(PARALLEL_WORKERS),
        PARALLEL_WORKERS,
        RUNTIME.python_runtime_label if USING_PORTABLE_PYTHON else f"System ({sys.executable})",
        RUNTIME.cuda_runtime_label,
        librosa.__version__,
        "Portable (bin/ffmpeg/ffmpeg.exe)" if FFMPEG_FOUND else "System FFmpeg (portable not found)",
        GPU_RUNTIME_AVAILABLE,
        get_gpu_info(),
        NVENC_AVAILABLE,
    )
)
print(f"   Temp Directory: {get_local_temp_dir()}")
print(f"   Output Directory: {OUTPUT_DIR}")


def find_available_local_port(
    preferred_port: int,
    host: str = "127.0.0.1",
    fallback_count: int = 20,
) -> int:
    """Return the first available localhost port starting at preferred_port."""
    last_error = None
    for port in range(preferred_port, preferred_port + max(1, fallback_count) + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
                return port
            except OSError as exc:
                last_error = exc

    raise OSError(
        f"Could not find an available localhost port in range {preferred_port}-{preferred_port + max(1, fallback_count)}"
    ) from last_error


def resolve_gui_port() -> int:
    """Resolve the Gradio port, honoring GRADIO_SERVER_PORT when valid."""
    raw_port = os.environ.get("GRADIO_SERVER_PORT", "").strip()
    if raw_port:
        try:
            preferred_port = int(raw_port)
        except ValueError:
            print(f"Warning: Invalid GRADIO_SERVER_PORT={raw_port!r}; falling back to {DEFAULT_GUI_PORT}")
            preferred_port = DEFAULT_GUI_PORT
    else:
        preferred_port = DEFAULT_GUI_PORT

    return find_available_local_port(preferred_port=preferred_port)
print(f"{CONSOLE_SEPARATOR}\n")

StatusResult: TypeAlias = Tuple[str, str, Dict]
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac"}
STANDARD_QUALITY_LABELS = [("Fast", "fast"), ("Balanced", "balanced"), ("High", "high")]
POWERSHELL_EXECUTABLE = shutil.which("powershell.exe") or shutil.which("powershell") or "powershell"

APP_THEME = gr.themes.Base(
    font=[
        gr.themes.GoogleFont("IBM Plex Sans"),
        gr.themes.GoogleFont("Space Grotesk"),
        "ui-sans-serif",
        "sans-serif",
    ],
    font_mono=[
        gr.themes.GoogleFont("IBM Plex Mono"),
        "Consolas",
        "monospace",
    ],
).set(
    body_background_fill="var(--beatsync-bg)",
    body_background_fill_dark="var(--beatsync-bg)",
    body_text_color="var(--beatsync-text)",
    body_text_color_dark="var(--beatsync-text)",
    body_text_color_subdued="var(--beatsync-subdued-text)",
    body_text_color_subdued_dark="var(--beatsync-subdued-text)",
    block_background_fill="transparent",
    block_background_fill_dark="transparent",
    block_border_color="var(--beatsync-line)",
    block_border_color_dark="var(--beatsync-line)",
    block_label_text_color="var(--beatsync-subdued-text)",
    block_label_text_color_dark="var(--beatsync-subdued-text)",
    block_title_background_fill="transparent",
    block_title_background_fill_dark="transparent",
    block_title_border_color="transparent",
    block_title_border_color_dark="transparent",
    block_title_border_width="0px",
    block_title_border_width_dark="0px",
    block_title_padding="0",
    block_title_radius="0px",
    block_title_text_color="var(--beatsync-text)",
    block_title_text_color_dark="var(--beatsync-text)",
    block_title_text_weight="600",
    block_info_text_color="var(--beatsync-subdued-text)",
    block_info_text_color_dark="var(--beatsync-subdued-text)",
    input_background_fill="var(--beatsync-input)",
    input_background_fill_dark="var(--beatsync-input)",
    input_background_fill_hover="var(--beatsync-input)",
    input_background_fill_hover_dark="var(--beatsync-input)",
    input_background_fill_focus="var(--beatsync-input)",
    input_background_fill_focus_dark="var(--beatsync-input)",
    input_placeholder_color="var(--beatsync-placeholder)",
    input_placeholder_color_dark="var(--beatsync-placeholder)",
    input_border_color="var(--beatsync-line-strong)",
    input_border_color_dark="var(--beatsync-line-strong)",
    input_border_color_focus="var(--beatsync-accent)",
    input_border_color_focus_dark="var(--beatsync-accent)",
    button_primary_background_fill="var(--beatsync-accent)",
    button_primary_background_fill_dark="var(--beatsync-accent)",
    button_primary_background_fill_hover="var(--beatsync-accent-strong)",
    button_primary_background_fill_hover_dark="var(--beatsync-accent-strong)",
    button_primary_border_color="var(--beatsync-accent)",
    button_primary_border_color_dark="var(--beatsync-accent)",
    button_primary_border_color_hover="var(--beatsync-accent-strong)",
    button_primary_border_color_hover_dark="var(--beatsync-accent-strong)",
    button_primary_text_color="var(--beatsync-primary-text)",
    button_primary_text_color_dark="var(--beatsync-primary-text)",
    button_primary_text_color_hover="var(--beatsync-primary-text)",
    button_primary_text_color_hover_dark="var(--beatsync-primary-text)",
    button_secondary_background_fill="var(--beatsync-muted-surface)",
    button_secondary_background_fill_dark="var(--beatsync-muted-surface)",
    button_secondary_background_fill_hover="var(--beatsync-muted-surface-strong)",
    button_secondary_background_fill_hover_dark="var(--beatsync-muted-surface-strong)",
    button_secondary_border_color="var(--beatsync-button-secondary-border)",
    button_secondary_border_color_dark="var(--beatsync-button-secondary-border)",
    button_secondary_border_color_hover="var(--beatsync-button-secondary-border)",
    button_secondary_border_color_hover_dark="var(--beatsync-button-secondary-border)",
    button_secondary_text_color="var(--beatsync-secondary-text)",
    button_secondary_text_color_dark="var(--beatsync-secondary-text)",
    button_secondary_text_color_hover="var(--beatsync-secondary-text)",
    button_secondary_text_color_hover_dark="var(--beatsync-secondary-text)",
    checkbox_background_color="var(--beatsync-input)",
    checkbox_background_color_dark="var(--beatsync-input)",
    checkbox_background_color_hover="var(--beatsync-surface-raised)",
    checkbox_background_color_hover_dark="var(--beatsync-surface-raised)",
    checkbox_background_color_focus="var(--beatsync-surface-raised)",
    checkbox_background_color_focus_dark="var(--beatsync-surface-raised)",
    checkbox_background_color_selected="var(--beatsync-surface)",
    checkbox_background_color_selected_dark="var(--beatsync-surface)",
    checkbox_border_color="var(--beatsync-line-strong)",
    checkbox_border_color_dark="var(--beatsync-line-strong)",
    checkbox_border_color_hover="var(--beatsync-accent)",
    checkbox_border_color_hover_dark="var(--beatsync-accent)",
    checkbox_border_color_focus="var(--beatsync-accent)",
    checkbox_border_color_focus_dark="var(--beatsync-accent)",
    checkbox_border_color_selected="var(--beatsync-accent)",
    checkbox_border_color_selected_dark="var(--beatsync-accent)",
    checkbox_label_background_fill="var(--beatsync-muted-surface)",
    checkbox_label_background_fill_dark="var(--beatsync-muted-surface)",
    checkbox_label_background_fill_hover="var(--beatsync-surface-raised)",
    checkbox_label_background_fill_hover_dark="var(--beatsync-surface-raised)",
    checkbox_label_background_fill_selected="var(--beatsync-accent-soft)",
    checkbox_label_background_fill_selected_dark="var(--beatsync-accent-soft)",
    checkbox_label_border_color="var(--beatsync-line-strong)",
    checkbox_label_border_color_dark="var(--beatsync-line-strong)",
    checkbox_label_border_color_hover="var(--beatsync-accent)",
    checkbox_label_border_color_hover_dark="var(--beatsync-accent)",
    checkbox_label_border_color_selected="var(--beatsync-accent)",
    checkbox_label_border_color_selected_dark="var(--beatsync-accent)",
    checkbox_label_text_color="var(--beatsync-text)",
    checkbox_label_text_color_dark="var(--beatsync-text)",
    checkbox_label_text_color_selected="var(--beatsync-text)",
    checkbox_label_text_color_selected_dark="var(--beatsync-text)",
    block_radius="18px",
    button_large_radius="12px",
    button_small_radius="10px",
)

APP_HEAD = """
<script>
(() => {
  const url = new URL(window.location.href);
  let changed = false;

  if (url.searchParams.get("__theme") !== "dark") {
    url.searchParams.set("__theme", "dark");
    changed = true;
  }

  if (url.searchParams.get("view") === "settings") {
    url.searchParams.delete("view");
    changed = true;
  }

  document.documentElement.classList.add("dark");
  document.documentElement.style.colorScheme = "dark";

  document.addEventListener(
    "DOMContentLoaded",
    () => {
      document.body.classList.add("dark");
      document.body.style.colorScheme = "dark";
    },
    { once: true },
  );

  if (changed) {
    window.history.replaceState({}, "", url.toString());
  }
})();
</script>
"""

APP_CSS = """
:root {
  color-scheme: dark;
  --beatsync-bg: #232c33;
  --beatsync-surface: rgba(251, 249, 255, 0.035);
  --beatsync-surface-raised: rgba(251, 249, 255, 0.055);
  --beatsync-muted-surface: rgba(251, 249, 255, 0.075);
  --beatsync-muted-surface-strong: rgba(251, 249, 255, 0.11);
  --beatsync-tile-surface: rgba(251, 249, 255, 0.06);
  --beatsync-tile-surface-hover: rgba(251, 249, 255, 0.1);
  --beatsync-tile-surface-selected: rgba(255, 102, 216, 0.16);
  --beatsync-tile-text: #fbf9ff;
  --beatsync-input: rgba(251, 249, 255, 0.045);
  --beatsync-text: #fbf9ff;
  --beatsync-muted: rgba(251, 249, 255, 0.76);
  --beatsync-subdued-text: rgba(251, 249, 255, 0.8);
  --beatsync-placeholder: rgba(251, 249, 255, 0.58);
  --beatsync-line: rgba(163, 124, 64, 0.26);
  --beatsync-line-strong: #a37c40;
  --beatsync-accent: #ff66d8;
  --beatsync-accent-strong: #ff66d8;
  --beatsync-accent-soft: rgba(255, 102, 216, 0.16);
  --beatsync-highlight: #a37c40;
  --beatsync-danger: #f71735;
  --beatsync-primary-text: #232c33;
  --beatsync-secondary-text: #fbf9ff;
  --beatsync-button-secondary-border: #a37c40;
  --beatsync-focus-ring: rgba(255, 102, 216, 0.34);
  --beatsync-radio-dot: #ff66d8;
  --beatsync-console: rgba(0, 0, 0, 0.16);
  --beatsync-console-text: #fbf9ff;
  --beatsync-ok: #ff66d8;
  --beatsync-warn: #fbf9ff;
}

html,
body {
  background: var(--beatsync-bg) !important;
  color: var(--beatsync-text) !important;
}

body {
  transition: background-color 0.18s ease, color 0.18s ease;
}

.gradio-container > footer,
.gradio-container footer.svelte-czcr5b,
.gradio-container button.settings,
.gradio-container .show-api,
.gradio-container .show-api-divider,
.gradio-container .api-docs,
.gradio-container #api-recorder-container {
  display: none !important;
}

.gradio-container {
  max-width: 1420px !important;
  padding: 24px 24px 36px !important;
}

.gradio-container,
.gradio-container label,
.gradio-container legend,
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container p,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container span {
  color: var(--beatsync-text);
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  background: var(--beatsync-input) !important;
  border-color: var(--beatsync-line-strong) !important;
}

.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
  color: var(--beatsync-placeholder) !important;
  opacity: 1 !important;
  -webkit-text-fill-color: var(--beatsync-placeholder) !important;
}

.gradio-container [data-testid="block-info"] {
  color: var(--beatsync-text) !important;
  -webkit-text-fill-color: var(--beatsync-text) !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
  opacity: 1 !important;
  font-weight: 600 !important;
}

.gradio-container button.primary,
.gradio-container a.primary {
  color: var(--beatsync-primary-text) !important;
  border-color: var(--beatsync-accent) !important;
}

.gradio-container button.primary:hover,
.gradio-container a.primary:hover {
  color: var(--beatsync-primary-text) !important;
  border-color: var(--beatsync-accent-strong) !important;
}

.gradio-container #process-btn[disabled] {
  opacity: 0.72 !important;
  cursor: progress !important;
}

.gradio-container button.secondary,
.gradio-container a.secondary {
  color: var(--beatsync-secondary-text) !important;
  background: var(--beatsync-muted-surface) !important;
  border-color: var(--beatsync-button-secondary-border) !important;
}

.gradio-container button.secondary:hover,
.gradio-container a.secondary:hover {
  color: var(--beatsync-secondary-text) !important;
  background: var(--beatsync-muted-surface-strong) !important;
  border-color: var(--beatsync-accent) !important;
}

.gradio-container button:focus-visible,
.gradio-container a:focus-visible,
.gradio-container input:focus-visible,
.gradio-container textarea:focus-visible,
.gradio-container select:focus-visible {
  outline: 3px solid var(--beatsync-focus-ring) !important;
  outline-offset: 2px !important;
}

.gradio-container input[role="listbox"] {
  color: var(--beatsync-text) !important;
  -webkit-text-fill-color: var(--beatsync-text) !important;
}

.gradio-container input[role="listbox"].subdued,
.gradio-container .subdued {
  color: var(--beatsync-subdued-text) !important;
  -webkit-text-fill-color: var(--beatsync-subdued-text) !important;
  opacity: 1 !important;
}

#app-shell {
  gap: 2.25rem;
  align-items: flex-start;
}

.app-header {
  display: flex;
  flex-direction: column;
  gap: 1.1rem;
  align-items: flex-start;
  margin-bottom: 1.75rem;
  padding-bottom: 1.25rem;
  border-bottom: 1px solid var(--beatsync-line);
}

.hero-kicker,
.section-kicker,
.insight-kicker {
  margin: 0 0 0.45rem;
  font-size: 0.77rem;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--beatsync-accent);
}

.hero-title,
.section-title,
.insight-title {
  margin: 0;
  font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
  letter-spacing: -0.03em;
  color: var(--beatsync-text);
}

.hero-title {
  font-size: clamp(2rem, 3.4vw, 3.1rem);
  line-height: 0.98;
}

.hero-body,
.section-body,
.insight-body,
.runtime-body,
.run-note,
.preview-note {
  margin: 0;
  color: var(--beatsync-muted);
  line-height: 1.6;
}

.hero-body {
  max-width: 62ch;
  margin-top: 0.9rem;
}

.header-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0.8rem 1.35rem;
  margin-top: 1rem;
}

.header-meta span {
  position: relative;
  padding-left: 0.9rem;
  color: var(--beatsync-muted);
  font-size: 0.95rem;
}

.header-meta span::before {
  content: "";
  position: absolute;
  left: 0;
  top: 0.55rem;
  width: 0.36rem;
  height: 0.36rem;
  border-radius: 999px;
  background: var(--beatsync-highlight);
}

.gradio-container label[data-testid$="-radio-label"] {
  gap: 0.68rem;
  border-color: var(--beatsync-line-strong) !important;
  background: var(--beatsync-tile-surface) !important;
  color: var(--beatsync-tile-text) !important;
  -webkit-text-fill-color: var(--beatsync-tile-text) !important;
  opacity: 1 !important;
}

.gradio-container label[data-testid$="-radio-label"]:hover {
  border-color: var(--beatsync-accent) !important;
  background: var(--beatsync-tile-surface-hover) !important;
}

.gradio-container label[data-testid$="-radio-label"].selected {
  border-color: var(--beatsync-accent) !important;
  background: var(--beatsync-tile-surface-selected) !important;
  color: var(--beatsync-tile-text) !important;
  -webkit-text-fill-color: var(--beatsync-tile-text) !important;
}

.gradio-container label[data-testid$="-radio-label"] span {
  color: var(--beatsync-tile-text) !important;
  -webkit-text-fill-color: var(--beatsync-tile-text) !important;
  opacity: 1 !important;
  font-weight: 600;
}

.gradio-container label[data-testid$="-radio-label"] * {
  color: var(--beatsync-tile-text) !important;
  -webkit-text-fill-color: var(--beatsync-tile-text) !important;
  opacity: 1 !important;
}

.gradio-container input[type="radio"] {
  appearance: none !important;
  -webkit-appearance: none !important;
  width: 1rem !important;
  height: 1rem !important;
  min-width: 1rem !important;
  min-height: 1rem !important;
  margin: 0 !important;
  padding: 0 !important;
  border-radius: 999px !important;
  border: 2px solid var(--beatsync-line-strong) !important;
  background: var(--beatsync-input) !important;
  box-shadow: none !important;
  display: inline-grid !important;
  place-items: center !important;
}

.gradio-container input[type="radio"]::before {
  content: "" !important;
  width: 0.42rem !important;
  height: 0.42rem !important;
  border-radius: 999px !important;
  transform: scale(0) !important;
  transition: transform 0.14s ease-in-out !important;
  background: var(--beatsync-radio-dot) !important;
}

.gradio-container input[type="radio"]:checked {
  border-color: var(--beatsync-accent) !important;
  background: var(--beatsync-surface) !important;
  background-image: radial-gradient(circle, var(--beatsync-radio-dot) 0 34%, transparent 35% 100%) !important;
}

.gradio-container input[type="radio"]:checked::before {
  transform: scale(1) !important;
}

.gradio-container input[type="radio"]:hover {
  border-color: var(--beatsync-accent) !important;
}

.surface-panel {
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  margin-bottom: 1.7rem !important;
}

.surface-panel > .wrap,
.surface-panel > div {
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

.surface-panel:last-child {
  margin-bottom: 0 !important;
}

.surface-panel .section-header {
  margin-bottom: 1rem;
  padding-bottom: 0.95rem;
  border-bottom: 1px solid var(--beatsync-line);
}

.section-title {
  font-size: 1.2rem;
  line-height: 1.08;
}

.section-body {
  margin-top: 0.4rem;
  max-width: 54ch;
}

.summary-grid,
.metric-grid {
  display: grid;
  gap: 1rem 1.3rem;
}

.summary-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
  margin-top: 1rem;
}

.metric-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
  margin-top: 1rem;
}

.summary-row,
.metric-row {
  padding: 0.85rem 0 0.95rem;
  border-top: 1px solid var(--beatsync-line);
}

.summary-row span,
.metric-row span {
  display: block;
  margin-bottom: 0.2rem;
  font-size: 0.77rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--beatsync-muted);
}

.summary-row strong,
.metric-row strong {
  display: block;
  font-size: 1rem;
  color: var(--beatsync-text);
}

.summary-row p,
.metric-row p {
  margin: 0.3rem 0 0;
  color: var(--beatsync-muted);
  line-height: 1.55;
}

.summary-row.full-span {
  grid-column: 1 / -1;
}

.summary-row.tone-ok strong {
  color: var(--beatsync-ok);
}

.summary-row.tone-warn strong {
  color: var(--beatsync-warn);
}

.insight-panel {
  margin-top: 0.9rem;
  padding-left: 1rem;
  border-left: 3px solid var(--beatsync-accent);
  background: transparent;
}

.insight-title {
  font-size: 1rem;
  line-height: 1.15;
}

.insight-body {
  margin-top: 0.4rem;
}

.insight-list {
  margin: 0.6rem 0 0;
  padding-left: 1rem;
  color: var(--beatsync-muted);
  line-height: 1.55;
}

.runtime-body,
.run-note,
.preview-note {
  margin-top: 0.85rem;
}

.inspector-column {
  position: sticky;
  top: 1rem;
  align-self: flex-start;
}

.browse-btn button,
#process-btn button {
  min-height: 3rem;
  font-weight: 600;
  letter-spacing: 0.01em;
}

#process-btn button {
  min-height: 3.2rem;
}

#status-log textarea {
  background: var(--beatsync-console) !important;
  color: var(--beatsync-console-text) !important;
  border: 1px solid var(--beatsync-line-strong) !important;
  border-radius: 14px !important;
  font-family: "IBM Plex Mono", Consolas, monospace !important;
  line-height: 1.52 !important;
}

#preview-panel video,
#preview-panel .video-preview,
#preview-panel [data-testid="video"],
#preview-panel [data-testid="media-container"] {
  border-radius: 14px !important;
  overflow: hidden !important;
  border: 1px solid var(--beatsync-line);
}

@media (max-width: 1180px) {
  .inspector-column {
    position: static;
  }
}

@media (max-width: 920px) {
  .gradio-container {
    padding: 16px 14px 28px !important;
  }

  .summary-grid,
  .metric-grid {
    grid-template-columns: 1fr;
  }
}
"""


def build_section_header(kicker: str, title: str, body: str) -> str:
    return f"""
    <div class="section-header">
      <p class="section-kicker">{html.escape(kicker)}</p>
      <h2 class="section-title">{html.escape(title)}</h2>
      <p class="section-body">{html.escape(body)}</p>
    </div>
    """


def build_summary_card(label: str, title: str, detail: str, tone: str = "neutral", full_span: bool = False) -> str:
    span_class = " full-span" if full_span else ""
    return f"""
    <article class="summary-row tone-{html.escape(tone)}{span_class}">
      <span>{html.escape(label)}</span>
      <strong>{html.escape(title)}</strong>
      <p>{html.escape(detail)}</p>
    </article>
    """


def build_metric_card(label: str, title: str, detail: str) -> str:
    return f"""
    <article class="metric-row">
      <span>{html.escape(label)}</span>
      <strong>{html.escape(title)}</strong>
      <p>{html.escape(detail)}</p>
    </article>
    """


def build_insight_panel(kicker: str, title: str, body: str, bullets: List[str]) -> str:
    bullet_items = "".join(f"<li>{html.escape(item)}</li>" for item in bullets)
    return f"""
    <div class="insight-panel">
      <p class="insight-kicker">{html.escape(kicker)}</p>
      <h3 class="insight-title">{html.escape(title)}</h3>
      <p class="insight-body">{html.escape(body)}</p>
      <ul class="insight-list">{bullet_items}</ul>
    </div>
    """


def build_app_hero(gpu_available: bool, nvenc_available: bool) -> str:
    runtime_summary = []
    runtime_summary.append("Local files stay in place")
    runtime_summary.append("Recursive MP4 and MKV scan")
    runtime_summary.append("NVENC delivery ready" if nvenc_available else "CPU and ProRes delivery ready")
    runtime_summary.append("GPU beat analysis ready" if gpu_available else "CPU beat analysis active")
    summary_markup = "".join(f"<span>{html.escape(item)}</span>" for item in runtime_summary)
    return f"""
    <section class="app-header">
      <div class="title-stack">
        <p class="hero-kicker">Application workspace</p>
        <h1 class="hero-title">{html.escape(UI_TITLE)}</h1>
        <p class="hero-body">
          Set source media, choose a beat strategy, and export a finished edit from the same screen.
        </p>
        <div class="header-meta">{summary_markup}</div>
      </div>
    </section>
    """


def build_runtime_overview(
    python_status: str,
    cuda_status: str,
    ffmpeg_status: str,
    ready_threads: int,
    cpu_count: int,
    gpu_available: bool,
    gpu_info: str,
    nvenc_available: bool,
) -> str:
    gpu_title = gpu_info if gpu_available else "CPU-only analysis"
    gpu_detail = "Beat detection will use the GPU." if gpu_available else "Beat detection will stay on the CPU."
    encoder_title = "NVENC delivery" if nvenc_available else "CPU / ProRes delivery"
    encoder_detail = "Fast MP4 delivery is available." if nvenc_available else "Delivery renders will use CPU-safe paths."
    metrics = "".join(
        [
            build_metric_card("Python", python_status, "Active runtime"),
            build_metric_card("CUDA", cuda_status, "Detected compute stack"),
            build_metric_card("FFmpeg", ffmpeg_status, "Render and preview pipeline"),
            build_metric_card("Thread budget", f"{ready_threads}/{cpu_count}", "Threads per encode job"),
            build_metric_card("Analysis", gpu_title, gpu_detail),
            build_metric_card("Video encode", encoder_title, encoder_detail),
        ]
    )
    return f"""
    <div>
      {build_section_header("System", "Runtime profile", "This is the environment BeatSync will use when you start a render.")}
      <div class="metric-grid">{metrics}</div>
      <p class="runtime-body">Source files are read in place. Intermediates render under temp/, and finished outputs are written to output/.</p>
    </div>
    """


def build_generation_mode_overview(mode: str) -> str:
    if mode == "manual":
        return build_insight_panel(
            "Beat strategy",
            "Manual emphasizes direct control.",
            "Use bass-focused detection when you already know how aggressively the edit should cut.",
            [
                "Good for simple or repetitive tracks.",
                "Lower values create denser cuts.",
                "Higher values skip beats for a calmer edit.",
            ],
        )
    if mode == "smart":
        return build_insight_panel(
            "Beat strategy",
            "Smart balances control and automation.",
            "Kick, clap, and higher-frequency detail are combined to produce more musical cut points.",
            [
                "Useful for layered percussion and changing energy.",
                "Presets shift from sparse to dense cutting.",
                "A strong default when you want control without manual tuning.",
            ],
        )
    return build_insight_panel(
        "Beat strategy",
        "Auto adapts to the song structure.",
        "BeatSync adjusts cut density across intros, verses, choruses, and breakdowns with minimal setup.",
        [
            "Best default for first-pass results.",
            "Works well on tracks with clear section changes.",
            "No extra tuning is required before rendering.",
        ],
    )


def build_processing_mode_overview(mode: str) -> str:
    if mode == "prores_proxy":
        return build_insight_panel(
            "Export pipeline",
            "ProRes prioritizes precision and finishing headroom.",
            "BeatSync writes a ProRes 422 Proxy master and generates a browser preview separately for in-app review.",
            [
                "Use when you need a master file rather than the fastest preview.",
                "Standard delivery quality presets are ignored.",
                "An optional lossless MP4 delivery copy can also be created.",
            ],
        )
    if mode == "hevc_nvenc":
        return build_insight_panel(
            "Export pipeline",
            "HEVC NVENC reduces file size.",
            "GPU encoding keeps exports quick while H.265 delivers smaller files than H.264.",
            [
                "Useful when storage or upload size matters.",
                "A browser preview is generated if the master is not directly playable.",
                "Delivery quality presets still apply.",
            ],
        )
    if mode == "h264_nvenc":
        return build_insight_panel(
            "Export pipeline",
            "H.264 NVENC is the fastest delivery path.",
            "Use this for quick, broadly compatible MP4 exports when the NVIDIA runtime is available.",
            [
                "Best for iteration and review.",
                "Quality presets balance speed against image quality.",
                "GPU encoding is used only when the runtime probe succeeds.",
            ],
        )
    return build_insight_panel(
        "Export pipeline",
        "CPU H.264 keeps the path universal.",
        "Choose this for maximum compatibility or when GPU encoding is unavailable on the current system.",
        [
            "Works on every supported runtime configuration.",
            "Delivery quality presets still apply.",
            "A safe fallback when portability matters more than speed.",
        ],
    )


def build_source_summary(audio_path: str, video_folder: str) -> str:
    normalized_audio = normalize_local_path(audio_path or "")
    normalized_folder = normalize_local_path(video_folder or "")
    cards: List[str] = []

    if not normalized_audio and not normalized_folder:
        return f"""
        <div class="summary-grid">
          {build_summary_card(
              "Preflight",
              "Select an audio track and a clip folder",
              "BeatSync will validate the media paths here before you start the render.",
              tone="neutral",
              full_span=True,
          )}
        </div>
        """

    if normalized_audio:
        if os.path.isfile(normalized_audio):
            audio_ext = Path(normalized_audio).suffix.lower()
            if audio_ext in SUPPORTED_AUDIO_EXTENSIONS:
                cards.append(
                    build_summary_card(
                        "Audio track",
                        Path(normalized_audio).name,
                        f"Ready to analyze as {audio_ext.lstrip('.').upper()} audio.",
                        tone="ok",
                    )
                )
            else:
                cards.append(
                    build_summary_card(
                        "Audio track",
                        Path(normalized_audio).name,
                        "The file exists, but it must be MP3, WAV, or FLAC.",
                        tone="warn",
                    )
                )
        else:
            cards.append(
                build_summary_card(
                    "Audio track",
                    "Path not found",
                    "Choose a valid local MP3, WAV, or FLAC file.",
                    tone="warn",
                )
            )
    else:
        cards.append(
            build_summary_card(
                "Audio track",
                "Not selected",
                "Choose the song that BeatSync should analyze.",
            )
        )

    if normalized_folder:
        if os.path.isdir(normalized_folder):
            try:
                video_paths = get_video_files(normalized_folder)
            except Exception as exc:
                cards.append(
                    build_summary_card(
                        "Clip folder",
                        Path(normalized_folder).name,
                        f"Folder could not be scanned: {exc}",
                        tone="warn",
                    )
                )
            else:
                if video_paths:
                    extensions = sorted({Path(path).suffix.lower().lstrip(".") for path in video_paths})
                    extension_text = ", ".join(ext.upper() for ext in extensions)
                    cards.append(
                        build_summary_card(
                            "Clip folder",
                            Path(normalized_folder).name,
                            f"{len(video_paths)} compatible clips found ({extension_text}).",
                            tone="ok",
                        )
                    )
                else:
                    cards.append(
                        build_summary_card(
                            "Clip folder",
                            Path(normalized_folder).name,
                            "The folder is valid, but no MP4 or MKV clips were found.",
                            tone="warn",
                        )
                    )
        else:
            cards.append(
                build_summary_card(
                    "Clip folder",
                    "Path not found",
                    "Choose a valid local folder. BeatSync scans subfolders recursively.",
                    tone="warn",
                )
            )
    else:
        cards.append(
            build_summary_card(
                "Clip folder",
                "Not selected",
                "Choose the folder that contains your source clips.",
            )
        )

    return f'<div class="summary-grid">{"".join(cards)}</div>'


def update_generation_mode_ui(mode: str):
    return (
        gr.update(visible=mode == "manual"),
        gr.update(visible=mode == "smart"),
        build_generation_mode_overview(mode),
    )


def recommend_output_filename(current_filename: str | None, processing_mode: str) -> str:
    raw_name = (current_filename or "").strip() or "music_video.mp4"
    base_name, ext = os.path.splitext(raw_name)
    normalized_ext = ext.lower()
    target_ext = ".mov" if processing_mode == "prores_proxy" else ".mp4"

    if not base_name:
        base_name = "music_video"

    if normalized_ext in {"", ".mp4", ".mov"}:
        return f"{base_name}{target_ext}"

    return raw_name


def update_processing_ui(mode: str, current_filename: str):
    is_prores = mode == "prores_proxy"
    return (
        gr.update(visible=not is_prores, interactive=not is_prores),
        gr.update(visible=is_prores, interactive=is_prores),
        build_processing_mode_overview(mode),
        recommend_output_filename(current_filename, mode),
    )


def build_render_started_feedback():
    return (
        gr.update(value="Creating...", interactive=False),
        "Render started.\nBeatSync is analyzing the selected sources and preparing the export pipeline.",
    )


def reset_render_button():
    return gr.update(value="Create Music Video", interactive=True)


def parse_resolution_choice(resolution_choice: str | None) -> tuple[int, int] | None:
    normalized = (resolution_choice or "default").strip().lower()
    if not normalized or normalized == "default":
        return None

    width_str, height_str = normalized.split("x", 1)
    width = int(width_str)
    height = int(height_str)
    if width <= 0 or height <= 0:
        raise ValueError("Custom resolution must use positive width and height values.")
    return width, height


def normalize_yes_no_choice(value: bool | str | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"yes", "true", "on", "1"}


def normalize_local_path(path_value: str) -> str | None:
    if not path_value:
        return None
    normalized = path_value.strip().strip('"').strip("'")
    if not normalized:
        return None

    normalized = unquote(normalized)

    parsed = urlparse(normalized)
    if parsed.scheme and parsed.scheme.lower() == "file":
        if parsed.netloc and parsed.path:
            normalized = f"//{parsed.netloc}{parsed.path}"
        else:
            normalized = parsed.path or normalized

    normalized = os.path.expandvars(os.path.expanduser(normalized))
    normalized = normalized.replace("/", os.sep)

    # Browsers and toolkits sometimes send Windows paths with a leading slash:
    # "/C:/Users/..." should still be treated as an absolute Windows path.
    if re.match(r"^[\\/]+[A-Za-z]:[\\/]", normalized):
        normalized = normalized.lstrip("\\/")

    # If a malformed value already has the project path prepended before a
    # drive-qualified Windows path, keep the last absolute drive path.
    drive_matches = list(re.finditer(r"[A-Za-z]:[\\/]", normalized))
    if len(drive_matches) > 1:
        normalized = normalized[drive_matches[-1].start():]

    if os.path.isabs(normalized):
        return os.path.normpath(normalized)

    return os.path.abspath(normalized)


def get_picker_start_dir(current_path: str | None, select_directory: bool) -> str:
    normalized = normalize_local_path(current_path or "")
    if normalized:
        if select_directory:
            if os.path.isdir(normalized):
                return normalized
        else:
            if os.path.isfile(normalized):
                parent_dir = os.path.dirname(normalized)
                if parent_dir and os.path.isdir(parent_dir):
                    return parent_dir
            if os.path.isdir(normalized):
                return normalized

        parent_dir = os.path.dirname(normalized)
        if parent_dir and os.path.isdir(parent_dir):
            return parent_dir

    home_dir = os.path.expanduser("~")
    return home_dir if os.path.isdir(home_dir) else SCRIPT_DIR


def escape_powershell_string(value: str) -> str:
    return value.replace("'", "''")


def run_native_dialog(powershell_script: str) -> str | None:
    if os.name != "nt":
        raise gr.Error("Native path browsing is currently available on Windows only.")

    encoded_script = base64.b64encode(powershell_script.encode("utf-16-le")).decode("ascii")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    try:
        result = subprocess.run(
            [
                POWERSHELL_EXECUTABLE,
                "-NoProfile",
                "-STA",
                "-ExecutionPolicy",
                "Bypass",
                "-EncodedCommand",
                encoded_script,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            creationflags=creationflags,
        )
    except FileNotFoundError as exc:
        raise gr.Error("Windows PowerShell was not found on this system.") from exc
    except subprocess.TimeoutExpired as exc:
        raise gr.Error("The file picker did not respond in time.") from exc

    if result.returncode != 0:
        stderr_text = (result.stderr or "").strip()
        raise gr.Error(f"Native file picker failed: {stderr_text or 'Unknown PowerShell error'}")

    selected_path = (result.stdout or "").strip()
    return selected_path or None


def browse_for_audio_file(current_path: str) -> str:
    start_dir = escape_powershell_string(get_picker_start_dir(current_path, select_directory=False))
    powershell_script = f"""
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::EnableVisualStyles()
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.InitialDirectory = '{start_dir}'
$dialog.Filter = 'Audio Files (*.mp3;*.wav;*.flac)|*.mp3;*.wav;*.flac|All Files (*.*)|*.*'
$dialog.Title = 'Select Audio File'
$dialog.Multiselect = $false
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    Write-Output $dialog.FileName
}}
"""
    selected_path = run_native_dialog(powershell_script)
    return normalize_local_path(selected_path or current_path or "") or ""


def browse_for_video_folder(current_path: str) -> str:
    start_dir = escape_powershell_string(get_picker_start_dir(current_path, select_directory=True))
    powershell_script = f"""
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::EnableVisualStyles()
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = 'Select Video Folder'
$dialog.SelectedPath = '{start_dir}'
$dialog.ShowNewFolderButton = $false
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    Write-Output $dialog.SelectedPath
}}
"""
    selected_path = run_native_dialog(powershell_script)
    return normalize_local_path(selected_path or current_path or "") or ""


def resolve_inputs(audio_path: str, video_folder: str, session_state: dict) -> tuple[str, List[str], dict]:
    resolved_audio = normalize_local_path(audio_path)
    if not resolved_audio:
        raise ValueError("Enter a local audio file path.")
    if not os.path.isfile(resolved_audio):
        raise FileNotFoundError(f"Audio file not found: {resolved_audio}")
    if Path(resolved_audio).suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError("Audio file must be MP3, WAV, or FLAC.")

    resolved_video_folder = normalize_local_path(video_folder)
    if not resolved_video_folder:
        raise ValueError("Enter a local video folder path.")
    if not os.path.isdir(resolved_video_folder):
        raise NotADirectoryError(f"Video folder not found: {resolved_video_folder}")

    resolved_video_paths = get_video_files(resolved_video_folder)

    session_state["resolved_audio_path"] = resolved_audio
    session_state["resolved_video_folder"] = resolved_video_folder
    session_state["resolved_video_paths"] = resolved_video_paths
    return resolved_audio, resolved_video_paths, session_state


def create_browser_preview(output_path: str, preview_path: str) -> str:
    if is_browser_playable_video(output_path):
        print("   Output is already browser-playable; no preview conversion needed.")
        return output_path

    attempts: List[tuple[str, List[str]]] = []
    preferred_hwaccel = ["-hwaccel", "cuda"] if not CPU_ONLY_MODE else ["-hwaccel", "auto"]
    cpu_fallback_hwaccels = [preferred_hwaccel]
    if not CPU_ONLY_MODE:
        cpu_fallback_hwaccels.append(["-hwaccel", "auto"])

    if NVENC_AVAILABLE:
        attempts.append(
            (
                "NVENC",
                [
                    FFMPEG_PATH,
                    *preferred_hwaccel,
                    "-i",
                    output_path,
                    "-c:v",
                    "h264_nvenc",
                    "-preset",
                    "p5",
                    "-cq",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-y",
                    preview_path,
                ],
            )
        )

    for hwaccel_args in cpu_fallback_hwaccels:
        attempt_name = "CPU (CUDA decode)" if hwaccel_args == ["-hwaccel", "cuda"] else "CPU"
        attempts.append(
            (
                attempt_name,
                [
                    FFMPEG_PATH,
                    *hwaccel_args,
                    "-i",
                    output_path,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-movflags",
                    "+faststart",
                    "-y",
                    preview_path,
                ],
            )
        )

    for encoder_name, preview_cmd in attempts:
        try:
            if os.path.exists(preview_path):
                os.remove(preview_path)
        except OSError:
            pass

        result = subprocess.run(preview_cmd, capture_output=True, text=True, timeout=180)
        if result.returncode == 0 and os.path.exists(preview_path):
            print(f"   Preview created for Gradio with {encoder_name}.")
            return preview_path

        stderr_tail = (result.stderr or "").strip().splitlines()
        error_line = stderr_tail[-1] if stderr_tail else "unknown FFmpeg error"
        print(f"   Preview creation with {encoder_name} failed: {error_line}")

    print("   Preview generation failed. Returning the original output path instead.")
    return output_path


def configure_asyncio_exception_filter() -> None:
    """Ignore benign Windows socket reset noise from closed browser transports."""
    if os.name == "nt":
        transport_type = getattr(asyncio.proactor_events, "_ProactorBasePipeTransport", None)
        original_method = getattr(transport_type, "_call_connection_lost", None)

        if transport_type and original_method and not getattr(original_method, "_beatsync_wrapped", False):
            def wrapped_call_connection_lost(self, exc):
                try:
                    return original_method(self, exc)
                except ConnectionResetError as reset_exc:
                    if getattr(reset_exc, "winerror", None) == 10054:
                        return None
                    raise

            wrapped_call_connection_lost._beatsync_wrapped = True
            transport_type._call_connection_lost = wrapped_call_connection_lost

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    def exception_handler(active_loop: asyncio.AbstractEventLoop, context: dict) -> None:
        exc = context.get("exception")
        message = str(context.get("message", ""))
        if (
            isinstance(exc, ConnectionResetError)
            and getattr(exc, "winerror", None) == 10054
            and "_ProactorBasePipeTransport._call_connection_lost" in message
        ):
            return

        active_loop.default_exception_handler(context)

    loop.set_exception_handler(exception_handler)


def process_video(audio_path: str, video_folder: str, generation_mode: str, cut_intensity: float, smart_preset: str, output_filename: str, direction: str, playback_speed_str: str, timing_offset: float, parallel_workers: int, processing_mode: str, standard_quality: str, create_prores_delivery_mp4: bool | str | None, custom_resolution: str, custom_fps: float, session_state: dict) -> StatusResult:
    try:
        session_state = session_state or {}
        resolved_audio_path, resolved_video_paths, session_state = resolve_inputs(
            audio_path, video_folder, session_state
        )

        use_gpu = GPU_RUNTIME_AVAILABLE
        set_gpu_mode(use_gpu)

        is_prores = processing_mode == "prores_proxy"
        create_prores_delivery_mp4 = normalize_yes_no_choice(create_prores_delivery_mp4)
        use_nvenc = processing_mode in ["h264_nvenc", "hevc_nvenc"] and NVENC_AVAILABLE and use_gpu
        gpu_encoder = processing_mode if use_nvenc else "none"
        quality = normalize_quality_profile(standard_quality)
        threads_per_job = estimate_threads_per_job(parallel_workers)

        if generation_mode == "manual":
            mode_str, smart_mode = "MANUAL MODE", False
        elif generation_mode == "smart":
            mode_str, smart_mode = "SMART MODE", True
        else:
            mode_str, smart_mode = "AUTO MODE", False

        if is_prores:
            codec_info = "ProRes 422 Proxy (.mov) - Lossless"
            encoder_info = "Lossless Concatenation"
        elif use_nvenc:
            codec_info = f"{gpu_encoder.upper()} (.mp4) | {quality.capitalize()} quality"
            encoder_info = f"{gpu_encoder.upper()} | {quality.capitalize()}"
        else:
            codec_info = f"H.264 (.mp4) | {quality.capitalize()} quality"
            encoder_info = f"libx264 | {quality.capitalize()}"

        accel_str = "GPU ACCELERATED" if use_gpu else "CPU MODE"
        python_str = "Portable" if USING_PORTABLE_PYTHON else "System"
        cuda_str = "Portable" if USING_PORTABLE_CUDA else "System/None"

        output_fps = custom_fps if custom_fps is not None and custom_fps > 0 else get_video_fps(resolved_video_paths[0])
        selected_target_size = parse_resolution_choice(custom_resolution)
        resolved_target_size = selected_target_size or get_video_resolution(resolved_video_paths[0])
        resolution_info = (
            f"{resolved_target_size[0]}x{resolved_target_size[1]} (custom)"
            if selected_target_size is not None
            else f"{resolved_target_size[0]}x{resolved_target_size[1]} (auto-detected)"
        )

        name, _ = os.path.splitext(output_filename or "music_video.mp4")
        safe_name = os.path.basename(name) or "music_video"
        ext = ".mov" if is_prores else ".mp4"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_{timestamp}{ext}"
        output_path = os.path.join(OUTPUT_DIR, filename)

        speed_factor = {"Half Speed": 0.5, "Double Speed": 2.0}.get(playback_speed_str, 1.0)

        print(f"\n{CONSOLE_SEPARATOR}")
        print(f"BEAT ANALYSIS - {mode_str} ({accel_str})")
        print(f"{CONSOLE_SEPARATOR}")

        if generation_mode == "manual":
            beat_times, beat_info = analyze_beats_manual(resolved_audio_path, use_gpu=use_gpu)
            selected_beats = process_manual_intensity(beat_times, cut_intensity)
            intensity_param = cut_intensity
        elif generation_mode == "smart":
            beat_times, beat_info = analyze_beats_smart(resolved_audio_path)
            selected_beats = select_beats_smart(beat_info, preset=smart_preset)
            intensity_param = smart_preset
        else:
            selected_beats, beat_info = analyze_beats_auto(resolved_audio_path, use_gpu=use_gpu)
            intensity_param = "auto"
            beat_times = beat_info.get("times", selected_beats)

        print(f"\n{CONSOLE_SEPARATOR}")
        print("VIDEO CREATION")
        print(f"{CONSOLE_SEPARATOR}")

        create_music_video(
            resolved_audio_path,
            resolved_video_paths,
            selected_beats,
            intensity_param,
            output_file=output_path,
            direction=direction,
            speed_factor=speed_factor,
            timing_offset=timing_offset,
            max_workers=parallel_workers,
            smart_mode=smart_mode,
            beat_info=beat_info,
            lossless_mode=is_prores,
            use_gpu=use_gpu,
            gpu_encoder=gpu_encoder,
            fps=output_fps,
            target_size=resolved_target_size,
            quality=quality,
            mode_name=generation_mode,
        )

        preview_path = output_path
        preview_filename = None
        delivery_mp4_filename = None
        delivery_mp4_error = None
        preview_source_path = output_path
        if is_prores:
            if create_prores_delivery_mp4:
                print("Generating optional lossless MP4 delivery file...")
                delivery_mp4_filename = f"{safe_name}_{timestamp}_delivery_lossless.mp4"
                delivery_mp4_path = os.path.join(OUTPUT_DIR, delivery_mp4_filename)
                try:
                    preview_source_path = create_lossless_delivery_mp4(
                        output_path,
                        delivery_mp4_path,
                        prefer_cuda_decode=not CPU_ONLY_MODE,
                    )
                    print(f"   Delivery MP4 created: {delivery_mp4_filename}")
                except Exception as exc:
                    delivery_mp4_filename = None
                    delivery_mp4_error = str(exc)
                    print(f"   Delivery MP4 creation failed: {delivery_mp4_error}")

            print("Generating browser-friendly preview for ProRes output...")
            preview_filename = f"{safe_name}_{timestamp}_preview.mp4"
            preview_path = os.path.join(OUTPUT_DIR, preview_filename)
            preview_path = create_browser_preview(preview_source_path, preview_path)
        elif not is_browser_playable_video(output_path):
            print("Generating browser-friendly preview for non-playable output...")
            preview_filename = f"{safe_name}_{timestamp}_preview.mp4"
            preview_path = os.path.join(OUTPUT_DIR, preview_filename)
            preview_path = create_browser_preview(output_path, preview_path)

        gpu_info = f"GPU: {get_gpu_info()}" if use_gpu else "CPU"
        fps_info = f"{output_fps:.2f} FPS (custom)" if custom_fps else f"{output_fps:.2f} FPS (auto-detected)"
        audio_info = "PCM 24-bit (48kHz)" if is_prores else "AAC 320 kbps (48kHz)"

        if generation_mode == "smart":
            preset_info = get_preset_info(smart_preset)
            total_cuts = len(selected_beats) - 1
            status_msg = get_success_message_smart(
                smart_preset,
                preset_info,
                len(beat_times),
                beat_info.get("tempo", 120),
                total_cuts,
                python_str,
                cuda_str,
                threads_per_job,
                CPU_COUNT,
                parallel_workers,
                gpu_info,
                encoder_info,
                codec_info,
                fps_info,
                filename,
                audio_info,
            )
        elif generation_mode == "auto":
            total_cuts = len(selected_beats) - 1
            sections_info = beat_info.get("selection_info", [])
            status_msg = get_success_message_auto(
                total_cuts,
                len(beat_times),
                beat_info.get("tempo", 120),
                sections_info,
                python_str,
                cuda_str,
                threads_per_job,
                CPU_COUNT,
                parallel_workers,
                gpu_info,
                encoder_info,
                codec_info,
                fps_info,
                filename,
                audio_info,
            )
        else:
            if cut_intensity < 1.0:
                subdivisions = int(1.0 / cut_intensity)
                total_cuts = len(selected_beats) - 1
                status_msg = get_success_message_manual_subdivided(
                    total_cuts,
                    subdivisions,
                    len(beat_times),
                    beat_info.get("tempo", 120),
                    cut_intensity,
                    python_str,
                    cuda_str,
                    threads_per_job,
                    CPU_COUNT,
                    parallel_workers,
                    gpu_info,
                    encoder_info,
                    codec_info,
                    fps_info,
                    filename,
                    audio_info,
                )
            else:
                beats_used = len(selected_beats) - 1
                cut_intensity_int = int(cut_intensity)
                status_msg = get_success_message_manual_skipped(
                    beats_used,
                    cut_intensity_int,
                    len(beat_times),
                    beat_info.get("tempo", 120),
                    cut_intensity,
                    python_str,
                    cuda_str,
                    threads_per_job,
                    CPU_COUNT,
                    parallel_workers,
                    gpu_info,
                    encoder_info,
                    codec_info,
                    fps_info,
                    filename,
                    audio_info,
                )

        print(f"\n{CONSOLE_SEPARATOR}")
        print("PROCESS COMPLETE")
        print(f"{CONSOLE_SEPARATOR}\n")

        if is_prores and delivery_mp4_filename:
            status_msg += f"\nDelivery MP4: {delivery_mp4_filename}"
        elif is_prores and create_prores_delivery_mp4 and delivery_mp4_error:
            status_msg += f"\nDelivery MP4: Failed ({delivery_mp4_error})"
        if preview_filename and os.path.normcase(preview_path) != os.path.normcase(output_path):
            status_msg += f"\nBrowser Preview: {preview_filename}"
        status_msg += f"\nTarget Resolution: {resolution_info}"

        return preview_path, status_msg, session_state
    except Exception as e:
        import traceback

        traceback.print_exc()
        return None, f"Error: {str(e)}", session_state


def cleanup_on_startup():
    session_temp_base = get_local_temp_dir()
    try:
        if os.path.exists(session_temp_base):
            print("Cleaning up old session directories...")
            for item in os.listdir(session_temp_base):
                item_path = os.path.join(session_temp_base, item)
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                        print(f"   Removed: {item}")
                except Exception as e:
                    print(f"   Could not remove {item}: {e}")
            print("   Old sessions cleared.")
        else:
            os.makedirs(session_temp_base, exist_ok=True)
            print("   Created session temp directory")
    except Exception as e:
        print(f"   Warning: Could not clean up sessions: {e}")
        os.makedirs(session_temp_base, exist_ok=True)


# Legacy pre-redesign layout kept only as reference.
def create_ui_legacy() -> gr.Blocks:
    """Deprecated compatibility alias for the active application UI."""
    return create_ui()

    python_status = (
        "✅ Portable (bin/python-3.13.9-embed-amd64/)"
        if USING_PORTABLE_PYTHON
        else "⚠️  System Python"
    )
    cuda_status = (
        f"✅ {RUNTIME.cuda_runtime_label}"
        if USING_PORTABLE_CUDA
        else "⚠️  System CUDA (or not available)"
    )
    ffmpeg_status = "✅ Portable (bin/ffmpeg/)" if FFMPEG_FOUND else "⚠️  System FFmpeg"
    ready_threads = estimate_threads_per_job(PARALLEL_WORKERS)

    app = gr.Blocks(title="BeatSync Engine", theme=gr.themes.Soft())
    with app:
        session_state = gr.State({})

        gr.Markdown(f"# {UI_TITLE}")
        gr.Markdown(UI_MAIN_DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Input Files")
                with gr.Row():
                    audio_input = gr.Textbox(
                        label=LABEL_AUDIO_FILE,
                        info=INFO_AUDIO_FILE,
                        placeholder=r"C:\Music\song.mp3",
                        scale=5,
                    )
                    browse_audio_btn = gr.Button("Browse...", scale=1, min_width=120)

                with gr.Row():
                    video_input = gr.Textbox(
                        label=LABEL_VIDEO_FOLDER,
                        info=INFO_VIDEO_FOLDER,
                        placeholder=r"D:\VideoClips",
                        scale=5,
                    )
                    browse_video_btn = gr.Button("Browse...", scale=1, min_width=120)

                with gr.Group():
                    gr.Markdown("### 🎯 Generation Mode")
                    generation_mode = gr.Radio(
                        choices=[
                            ("🤖 Auto Mode (Recommended)", "auto"),
                            ("🧠 Smart Mode", "smart"),
                            ("⚙️ Manual Mode", "manual"),
                        ],
                        value="auto",
                        label=LABEL_GENERATION_MODE,
                        info=INFO_GENERATION_MODE,
                    )

                    auto_group = gr.Group(visible=True)
                    with auto_group:
                        gr.Markdown(AUTO_MODE_DESCRIPTION)

                    smart_group = gr.Group(visible=False)
                    with smart_group:
                        gr.Markdown(SMART_MODE_DESCRIPTION)
                        smart_preset = gr.Radio(
                            choices=["slower", "slow", "normal", "fast", "faster"],
                            value="normal",
                            label=LABEL_CUT_PRESET,
                            info=INFO_CUT_PRESET,
                        )

                    manual_group = gr.Group(visible=False)
                    with manual_group:
                        gr.Markdown(MANUAL_MODE_DESCRIPTION)
                        cut_intensity = gr.Slider(
                            minimum=0.1,
                            maximum=16,
                            value=4,
                            step=0.1,
                            label=LABEL_CUT_INTENSITY,
                            info=INFO_CUT_INTENSITY,
                        )

                with gr.Group():
                    gr.Markdown("### ⚙️ Video Settings")
                    direction = gr.Radio(choices=["forward", "backward", "random"], value="forward", label=LABEL_DIRECTION, info=INFO_DIRECTION)
                    playback_speed = gr.Radio(choices=["Normal Speed", "Half Speed", "Double Speed"], value="Normal Speed", label=LABEL_PLAYBACK_SPEED, info=INFO_PLAYBACK_SPEED)
                    timing_offset = gr.Slider(minimum=-0.5, maximum=0.5, value=0.0, step=0.01, label=LABEL_TIMING_OFFSET, info=INFO_TIMING_OFFSET)
                    custom_resolution = gr.Dropdown(
                        choices=RESOLUTION_PRESET_CHOICES,
                        value="default",
                        label=LABEL_CUSTOM_RESOLUTION,
                        info=INFO_CUSTOM_RESOLUTION,
                    )
                    custom_fps = gr.Number(label=LABEL_CUSTOM_FPS, value=None, precision=2, info=INFO_CUSTOM_FPS)

                with gr.Group():
                    gr.Markdown("### 🎬 Processing Mode")
                    if NVENC_AVAILABLE:
                        processing_mode = gr.Radio(
                            choices=[("NVIDIA NVENC H.264", "h264_nvenc"), ("NVIDIA NVENC HEVC (H.265)", "hevc_nvenc"), ("CPU (H.264)", "cpu"), ("ProRes 422 Proxy (Precise Mode)", "prores_proxy")],
                            value="h264_nvenc",
                            label=LABEL_PROCESSING_MODE,
                            info=get_processing_mode_info_nvenc(),
                        )
                    else:
                        processing_mode = gr.Radio(
                            choices=[("CPU (H.264)", "cpu"), ("ProRes 422 Proxy (Precise Mode)", "prores_proxy")],
                            value="cpu",
                            label=LABEL_PROCESSING_MODE,
                            info=get_processing_mode_info_cpu(),
                        )
                    standard_quality = gr.Radio(
                        choices=STANDARD_QUALITY_LABELS,
                        value=DEFAULT_STANDARD_QUALITY,
                        label=LABEL_STANDARD_QUALITY,
                        info=INFO_STANDARD_QUALITY,
                    )
                    prores_delivery_mp4 = gr.Radio(
                        choices=[("No", "no"), ("Yes", "yes")],
                        value="no",
                        visible=False,
                        interactive=True,
                        label=LABEL_PRORES_DELIVERY_MP4,
                        info=INFO_PRORES_DELIVERY_MP4,
                    )

                with gr.Group():
                    gr.Markdown("### ⚙️ Performance Settings")
                    parallel_workers = gr.Slider(
                        minimum=1,
                        maximum=min(16, max(CPU_COUNT // 2, 4)),
                        value=PARALLEL_WORKERS,
                        step=1,
                        label=get_parallel_workers_label(PARALLEL_WORKERS),
                        info=get_parallel_workers_info(),
                    )

                with gr.Group():
                    gr.Markdown("### 📁 Output Settings")
                    output_filename = gr.Textbox(value="music_video.mp4", label=LABEL_OUTPUT_FILENAME, info=INFO_OUTPUT_FILENAME)

                process_btn = gr.Button("🎬 Create Music Video", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### 📺 Output")
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value=get_ready_status(
                        python_status,
                        cuda_status,
                        ready_threads,
                        CPU_COUNT,
                        ffmpeg_status,
                        GPU_RUNTIME_AVAILABLE,
                        get_gpu_info(),
                        NVENC_AVAILABLE,
                    ),
                    lines=16,
                    max_lines=25,
                )
                video_output = gr.Video(label="Generated Music Video", interactive=False)

        def toggle_mode(mode):
            return {
                manual_group: gr.update(visible=mode == "manual"),
                smart_group: gr.update(visible=mode == "smart"),
                auto_group: gr.update(visible=mode == "auto"),
            }

        def toggle_processing_options(mode):
            is_prores = mode == "prores_proxy"
            return gr.update(visible=is_prores, interactive=is_prores)

        generation_mode.change(fn=toggle_mode, inputs=[generation_mode], outputs=[manual_group, smart_group, auto_group])
        processing_mode.change(
            fn=toggle_processing_options,
            inputs=[processing_mode],
            outputs=[prores_delivery_mp4],
        )
        browse_audio_btn.click(
            fn=browse_for_audio_file,
            inputs=[audio_input],
            outputs=[audio_input],
        )
        browse_video_btn.click(
            fn=browse_for_video_folder,
            inputs=[video_input],
            outputs=[video_input],
        )

        process_btn.click(
            fn=process_video,
            inputs=[
                audio_input,
                video_input,
                generation_mode,
                cut_intensity,
                smart_preset,
                output_filename,
                direction,
                playback_speed,
                timing_offset,
                parallel_workers,
                processing_mode,
                standard_quality,
                prores_delivery_mp4,
                custom_resolution,
                custom_fps,
                session_state,
            ],
            outputs=[video_output, status_output, session_state],
        )

    return app

# Active application-oriented UI.
def create_ui() -> gr.Blocks:
    python_status = "Portable runtime" if USING_PORTABLE_PYTHON else "System Python"
    cuda_status = RUNTIME.cuda_runtime_label if USING_PORTABLE_CUDA else "System CUDA / not available"
    ffmpeg_status = "Portable FFmpeg" if FFMPEG_FOUND else "System FFmpeg"
    ready_threads = estimate_threads_per_job(PARALLEL_WORKERS)
    default_processing_mode = "h264_nvenc" if NVENC_AVAILABLE else "cpu"
    default_output_name = recommend_output_filename("music_video.mp4", default_processing_mode)
    smart_preset_choices = [
        (SMART_PRESETS_CONFIG[preset]["description"], preset)
        for preset in ["slower", "slow", "normal", "fast", "faster"]
    ]

    app = gr.Blocks(title="BeatSync Engine", theme=APP_THEME, css=APP_CSS, head=APP_HEAD, fill_width=True)
    with app:
        session_state = gr.State({})
        gr.Navbar(visible=False, main_page_name=False)

        gr.HTML(
            build_app_hero(GPU_RUNTIME_AVAILABLE, NVENC_AVAILABLE),
            elem_id="app-hero",
        )

        with gr.Row(elem_id="app-shell"):
            with gr.Column(scale=7, min_width=760, elem_classes="workspace-column"):
                with gr.Group(elem_classes="surface-panel"):
                    gr.HTML(
                        build_section_header(
                            "Session",
                            "Source media",
                            "Select the song and clip folder for this render.",
                        )
                    )
                    with gr.Row():
                        audio_input = gr.Textbox(
                            label=LABEL_AUDIO_FILE,
                            info=INFO_AUDIO_FILE,
                            placeholder=r"C:\Music\song.mp3",
                            scale=5,
                        )
                        browse_audio_btn = gr.Button(
                            "Browse audio",
                            scale=1,
                            min_width=150,
                            variant="secondary",
                            elem_classes="browse-btn",
                        )

                    with gr.Row():
                        video_input = gr.Textbox(
                            label=LABEL_VIDEO_FOLDER,
                            info=INFO_VIDEO_FOLDER,
                            placeholder=r"D:\VideoClips",
                            scale=5,
                        )
                        browse_video_btn = gr.Button(
                            "Browse folder",
                            scale=1,
                            min_width=150,
                            variant="secondary",
                            elem_classes="browse-btn",
                        )

                    source_summary = gr.HTML(build_source_summary("", ""))

                with gr.Row(equal_height=True):
                    with gr.Group(elem_classes="surface-panel"):
                        gr.HTML(
                            build_section_header(
                                "Analysis",
                                "Beat selection",
                                "Choose how the application should detect and prioritize edit points.",
                            )
                        )
                        generation_mode = gr.Radio(
                            choices=[
                                ("Auto (recommended)", "auto"),
                                ("Smart", "smart"),
                                ("Manual", "manual"),
                            ],
                            value="auto",
                            label=LABEL_GENERATION_MODE,
                            info=INFO_GENERATION_MODE,
                        )
                        mode_summary = gr.HTML(build_generation_mode_overview("auto"))

                        smart_group = gr.Group(visible=False)
                        with smart_group:
                            smart_preset = gr.Radio(
                                choices=smart_preset_choices,
                                value="normal",
                                label=LABEL_CUT_PRESET,
                                info=INFO_CUT_PRESET,
                            )

                        manual_group = gr.Group(visible=False)
                        with manual_group:
                            cut_intensity = gr.Slider(
                                minimum=0.1,
                                maximum=16,
                                value=4,
                                step=0.1,
                                label=LABEL_CUT_INTENSITY,
                                info=INFO_CUT_INTENSITY,
                            )

                    with gr.Group(elem_classes="surface-panel"):
                        gr.HTML(
                            build_section_header(
                                "Export",
                                "Output pipeline",
                                "Choose the render path that fits speed, compatibility, and finishing needs.",
                            )
                        )
                        if NVENC_AVAILABLE:
                            processing_mode = gr.Radio(
                                choices=[
                                    ("NVIDIA NVENC H.264", "h264_nvenc"),
                                    ("NVIDIA NVENC HEVC (H.265)", "hevc_nvenc"),
                                    ("CPU H.264", "cpu"),
                                    ("ProRes 422 Proxy", "prores_proxy"),
                                ],
                                value=default_processing_mode,
                                label=LABEL_PROCESSING_MODE,
                                info=get_processing_mode_info_nvenc(),
                            )
                        else:
                            processing_mode = gr.Radio(
                                choices=[
                                    ("CPU H.264", "cpu"),
                                    ("ProRes 422 Proxy", "prores_proxy"),
                                ],
                                value=default_processing_mode,
                                label=LABEL_PROCESSING_MODE,
                                info=get_processing_mode_info_cpu(),
                            )

                        processing_summary = gr.HTML(
                            build_processing_mode_overview(default_processing_mode)
                        )
                        standard_quality = gr.Radio(
                            choices=STANDARD_QUALITY_LABELS,
                            value=DEFAULT_STANDARD_QUALITY,
                            label=LABEL_STANDARD_QUALITY,
                            info=INFO_STANDARD_QUALITY,
                        )
                        prores_delivery_mp4 = gr.Radio(
                            choices=[("No", "no"), ("Yes", "yes")],
                            value="no",
                            visible=False,
                            interactive=True,
                            label=LABEL_PRORES_DELIVERY_MP4,
                            info=INFO_PRORES_DELIVERY_MP4,
                        )

                with gr.Row(equal_height=True):
                    with gr.Group(elem_classes="surface-panel"):
                        gr.HTML(
                            build_section_header(
                                "Playback",
                                "Clip behavior",
                                "Adjust direction, speed, and timing before you start the render.",
                            )
                        )
                        direction = gr.Radio(
                            choices=[
                                ("Forward", "forward"),
                                ("Backward", "backward"),
                                ("Random", "random"),
                            ],
                            value="forward",
                            label=LABEL_DIRECTION,
                            info=INFO_DIRECTION,
                        )
                        playback_speed = gr.Radio(
                            choices=[
                                ("Normal speed", "Normal Speed"),
                                ("Half speed", "Half Speed"),
                                ("Double speed", "Double Speed"),
                            ],
                            value="Normal Speed",
                            label=LABEL_PLAYBACK_SPEED,
                            info=INFO_PLAYBACK_SPEED,
                        )
                        timing_offset = gr.Slider(
                            minimum=-0.5,
                            maximum=0.5,
                            value=0.0,
                            step=0.01,
                            label=LABEL_TIMING_OFFSET,
                            info=INFO_TIMING_OFFSET,
                        )

                    with gr.Group(elem_classes="surface-panel"):
                        gr.HTML(
                            build_section_header(
                                "Render",
                                "Output settings",
                                "Set frame geometry, throughput, and file naming before you launch the render.",
                            )
                        )
                        custom_resolution = gr.Dropdown(
                            choices=RESOLUTION_PRESET_CHOICES,
                            value="default",
                            label=LABEL_CUSTOM_RESOLUTION,
                            info=INFO_CUSTOM_RESOLUTION,
                        )
                        custom_fps = gr.Number(
                            label=LABEL_CUSTOM_FPS,
                            value=None,
                            precision=2,
                            info=INFO_CUSTOM_FPS,
                        )
                        parallel_workers = gr.Slider(
                            minimum=1,
                            maximum=min(16, max(CPU_COUNT // 2, 4)),
                            value=PARALLEL_WORKERS,
                            step=1,
                            label=get_parallel_workers_label(PARALLEL_WORKERS),
                            info=get_parallel_workers_info(),
                        )
                        output_filename = gr.Textbox(
                            value=default_output_name,
                            label=LABEL_OUTPUT_FILENAME,
                            info=INFO_OUTPUT_FILENAME,
                        )
                        gr.HTML(
                            '<p class="run-note">A timestamp is appended automatically. ProRes renders export as MOV; CPU and NVENC delivery modes export as MP4.</p>'
                        )
                        process_btn = gr.Button(
                            "Create Music Video",
                            variant="primary",
                            size="lg",
                            elem_id="process-btn",
                        )

            with gr.Column(scale=5, min_width=360, elem_classes="inspector-column"):
                with gr.Group(elem_classes="surface-panel"):
                    gr.HTML(
                        build_runtime_overview(
                            python_status,
                            cuda_status,
                            ffmpeg_status,
                            ready_threads,
                            CPU_COUNT,
                            GPU_RUNTIME_AVAILABLE,
                            get_gpu_info(),
                            NVENC_AVAILABLE,
                        )
                    )

                with gr.Group(elem_classes="surface-panel"):
                    gr.HTML(
                        build_section_header(
                            "Render",
                            "Status",
                            "Readiness notes, success details, and output paths appear here.",
                        )
                    )
                    status_output = gr.Textbox(
                        label="Render log",
                        interactive=False,
                        value=get_ready_status(
                            python_status,
                            cuda_status,
                            ready_threads,
                            CPU_COUNT,
                            ffmpeg_status,
                            GPU_RUNTIME_AVAILABLE,
                            get_gpu_info(),
                            NVENC_AVAILABLE,
                        ),
                        lines=14,
                        max_lines=22,
                        elem_id="status-log",
                    )

                with gr.Group(elem_classes="surface-panel", elem_id="preview-panel"):
                    gr.HTML(
                        build_section_header(
                            "Preview",
                            "Latest output",
                            "The newest browser-playable render appears here after each run.",
                        )
                    )
                    gr.HTML(
                        '<p class="preview-note">When the master export is not directly playable in the browser, BeatSync automatically creates a preview copy for the Gradio player.</p>'
                    )
                    video_output = gr.Video(label="Latest preview", interactive=False)

        generation_mode.change(
            fn=update_generation_mode_ui,
            inputs=[generation_mode],
            outputs=[manual_group, smart_group, mode_summary],
        )
        processing_mode.change(
            fn=update_processing_ui,
            inputs=[processing_mode, output_filename],
            outputs=[standard_quality, prores_delivery_mp4, processing_summary, output_filename],
        )
        audio_input.change(
            fn=build_source_summary,
            inputs=[audio_input, video_input],
            outputs=[source_summary],
        )
        video_input.change(
            fn=build_source_summary,
            inputs=[audio_input, video_input],
            outputs=[source_summary],
        )
        browse_audio_event = browse_audio_btn.click(
            fn=browse_for_audio_file,
            inputs=[audio_input],
            outputs=[audio_input],
        )
        browse_audio_event.then(
            fn=build_source_summary,
            inputs=[audio_input, video_input],
            outputs=[source_summary],
        )
        browse_video_event = browse_video_btn.click(
            fn=browse_for_video_folder,
            inputs=[video_input],
            outputs=[video_input],
        )
        browse_video_event.then(
            fn=build_source_summary,
            inputs=[audio_input, video_input],
            outputs=[source_summary],
        )

        process_btn_event = process_btn.click(
            fn=build_render_started_feedback,
            outputs=[process_btn, status_output],
            queue=False,
        )
        process_btn_event = process_btn_event.then(
            fn=process_video,
            inputs=[
                audio_input,
                video_input,
                generation_mode,
                cut_intensity,
                smart_preset,
                output_filename,
                direction,
                playback_speed,
                timing_offset,
                parallel_workers,
                processing_mode,
                standard_quality,
                prores_delivery_mp4,
                custom_resolution,
                custom_fps,
                session_state,
            ],
            outputs=[video_output, status_output, session_state],
        )
        process_btn_event.then(
            fn=reset_render_button,
            outputs=[process_btn],
            queue=False,
        )

    return app


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    configure_asyncio_exception_filter()
    cleanup_on_startup()
    launch_port = resolve_gui_port()
    print("Starting Gradio interface...")
    if launch_port != DEFAULT_GUI_PORT:
        print(f"   Preferred port {DEFAULT_GUI_PORT} is busy; using http://127.0.0.1:{launch_port}")
    else:
        print(f"   URL: http://127.0.0.1:{launch_port}")
    print("   Local path workflow: ENABLED")
    print(f"\n{CONSOLE_SEPARATOR}\n")

    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=launch_port,
        share=False,
        inbrowser=True,
        show_error=True,
        show_api=False,
    )
