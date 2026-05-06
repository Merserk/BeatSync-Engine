#!/usr/bin/env python3
"""Shared project paths and directory helpers."""

import os

from logger import ROOT_DIR


INPUT_DIR = os.path.join(ROOT_DIR, 'input')
AUDIO_INPUT_DIR = os.path.join(INPUT_DIR, 'audio')
VIDEO_INPUT_DIR = os.path.join(INPUT_DIR, 'video')
PROCESSING_DIR = os.path.join(INPUT_DIR, 'processing')
GRADIO_TEMP_DIR = os.path.join(INPUT_DIR, 'gradio_uploads')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')


def ensure_project_dirs() -> None:
    """Create the standard project directories if they are missing."""
    for directory in [
        INPUT_DIR,
        AUDIO_INPUT_DIR,
        VIDEO_INPUT_DIR,
        PROCESSING_DIR,
        GRADIO_TEMP_DIR,
        OUTPUT_DIR,
    ]:
        os.makedirs(directory, exist_ok=True)


def get_input_dir() -> str:
    """Get the local input directory path."""
    return INPUT_DIR


def get_audio_input_dir() -> str:
    """Get the audio input directory path."""
    return AUDIO_INPUT_DIR


def get_video_input_dir() -> str:
    """Get the video input directory path."""
    return VIDEO_INPUT_DIR


def get_processing_dir() -> str:
    """Get the processing directory path."""
    return PROCESSING_DIR


def get_gradio_temp_dir() -> str:
    """Get the Gradio upload/temp directory path."""
    return GRADIO_TEMP_DIR


def get_output_dir() -> str:
    """Get the final output directory path."""
    return OUTPUT_DIR


ensure_project_dirs()
