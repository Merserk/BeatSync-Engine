"""One portable FFmpeg/FFprobe runtime for all BeatSync media commands."""

from functools import lru_cache
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
from gpu_devices import ffmpeg_gpu_environment


ROOT = Path(__file__).resolve().parents[1]
FFMPEG_BIN_DIR = str(ROOT / 'bin' / 'ffmpeg')
FFMPEG_EXE = str(Path(FFMPEG_BIN_DIR) / 'ffmpeg.exe')
FFPROBE_EXE = str(Path(FFMPEG_BIN_DIR) / 'ffprobe.exe')
MIN_FFMPEG_VERSION = (9, 0, 2)
_PROBE_LOCK = threading.RLock()


class MediaError(RuntimeError):
    """A media command or metadata check failed."""


def run_media_command(cmd, timeout=120):
    return subprocess.run(
        [str(value) for value in cmd], stdin=subprocess.DEVNULL,
        capture_output=True, text=True, encoding='utf-8', errors='replace',
        timeout=timeout, check=False,
        env=ffmpeg_gpu_environment(),
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
    )


def checked_output(cmd, timeout=30):
    try:
        result = run_media_command(cmd, timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MediaError(f"Could not run {Path(cmd[0]).name}: {exc}") from exc
    if result.returncode:
        raise MediaError(result.stderr.strip()[-2200:] or f"Media command exited with {result.returncode}")
    return result.stdout


@lru_cache(maxsize=1)
def get_runtime_info():
    """Require matching full portable binaries; never silently use PATH tools."""
    lines = []
    for exe in (FFMPEG_EXE, FFPROBE_EXE):
        if not Path(exe).is_file():
            raise MediaError(f"Missing {exe}. Run install.bat to install the latest FFmpeg full build.")
        lines.append(checked_output([exe, '-version']))
    versions = [re.search(r'^\w+ version (\S+)', value) for value in lines]
    if not all(versions) or versions[0][1] != versions[1][1]:
        raise MediaError('FFmpeg and FFprobe versions differ. Run install.bat to repair the portable full build.')
    release = re.match(r'(\d+)\.(\d+)(?:\.(\d+))?', versions[0][1])
    if release and tuple(int(value or 0) for value in release.groups()) < MIN_FFMPEG_VERSION:
        raise MediaError('FFmpeg 9.0.2 or newer is required. Run install.bat for the latest full build.')
    config = lines[0]
    if '--enable-vulkan' not in config or '--enable-libplacebo' not in config:
        raise MediaError('FFmpeg full build with Vulkan is required. Run install.bat to replace the current build.')
    listing = checked_output([FFMPEG_EXE, '-hide_banner', '-encoders'])
    encoders = set(re.findall(r'^\s*[VAS][A-Z.]{5}\s+(\S+)', listing, re.MULTILINE))
    required = {'libx264', 'prores_ks', 'h264_vulkan', 'hevc_vulkan', 'pcm_s24le', 'aac'}
    if required - encoders:
        raise MediaError(f"FFmpeg is missing encoders: {', '.join(sorted(required - encoders))}. Run install.bat.")
    return {'version': versions[0][1], 'full': True, 'ffmpeg': FFMPEG_EXE,
            'ffprobe': FFPROBE_EXE, 'encoders': sorted(encoders)}


def positive_number(value):
    try:
        result = float(Fraction(str(value)))
        return result if math.isfinite(result) and result > 0 else None
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def fps_string(fps):
    if positive_number(fps) is None:
        raise ValueError(f'Invalid output FPS: {fps}')
    value = Fraction(str(fps)).limit_denominator(100000)
    return str(value.numerator) if value.denominator == 1 else f'{value.numerator}/{value.denominator}'


def normalize_output_size(size):
    width, height = (int(value) for value in size)
    if width <= 0 or height <= 0:
        raise ValueError(f'Invalid output resolution: {size}')
    return max(2, width // 2 * 2), max(2, height // 2 * 2)


@lru_cache(maxsize=1024)
def _probe_file(path, size, mtime_ns):
    get_runtime_info()
    data = json.loads(checked_output([
        FFPROBE_EXE, '-v', 'error', '-show_format', '-show_streams', '-of', 'json', path,
    ]))
    if not data.get('streams'):
        raise MediaError(f'No media streams in {path}')
    return data


def probe_media(path):
    file = Path(path).resolve()
    try:
        stat = file.stat()
    except OSError as exc:
        raise MediaError(f'Cannot access media file {file}: {exc}') from exc
    # Serialize cache fills so parallel clips from the same source share one
    # metadata query. Changed files get a fresh key even at the same path.
    with _PROBE_LOCK:
        return _probe_file(str(file), stat.st_size, stat.st_mtime_ns)


def video_stream(path):
    videos = [s for s in probe_media(path)['streams'] if s.get('codec_type') == 'video'
              and not s.get('disposition', {}).get('attached_pic')]
    if not videos:
        raise MediaError(f'No video stream in {path}')
    return videos[0]


def media_duration(path):
    data = probe_media(path)
    videos = [s for s in data['streams'] if s.get('codec_type') == 'video'
              and not s.get('disposition', {}).get('attached_pic')]
    duration = positive_number(videos[0].get('duration')) if videos else None
    if duration is None and videos:
        ticks = positive_number(videos[0].get('duration_ts'))
        base = positive_number(videos[0].get('time_base'))
        if ticks and base:
            duration = ticks * base
    if duration is None:
        duration = positive_number(data.get('format', {}).get('duration'))
    if duration is None:
        duration = max((positive_number(s.get('duration')) or 0 for s in data['streams']), default=0)
    if not duration:
        raise MediaError(f'Missing or invalid duration in {path}')
    return duration


def video_fps(path):
    stream = video_stream(path)
    rate = positive_number(stream.get('avg_frame_rate')) or positive_number(stream.get('r_frame_rate'))
    if rate is None:
        raise MediaError(f'Missing or invalid video FPS in {path}')
    return rate


def video_resolution(path):
    stream = video_stream(path)
    width, height = int(stream.get('width', 0)), int(stream.get('height', 0))
    if min(width, height) <= 0:
        raise MediaError(f'Missing or invalid video dimensions in {path}')
    sar = positive_number(str(stream.get('sample_aspect_ratio', '1:1')).replace(':', '/')) or 1.0
    width = max(1, int(round(width * sar)))
    rotation = next((s.get('rotation', 0) for s in stream.get('side_data_list', []) if 'rotation' in s),
                    stream.get('tags', {}).get('rotate', 0))
    if abs(float(rotation)) % 180 == 90:
        width, height = height, width
    return width, height


def verify_video(path, expected_frames=None):
    stream = video_stream(path)
    if expected_frames is not None:
        count = positive_number(stream.get('nb_frames'))
        if count is None:
            data = json.loads(checked_output([
                FFPROBE_EXE, '-v', 'error', '-select_streams', 'v:0', '-count_packets',
                '-show_entries', 'stream=nb_read_packets', '-of', 'json', str(path),
            ], timeout=120))
            count = positive_number(data['streams'][0].get('nb_read_packets'))
        if count != expected_frames:
            raise MediaError(f'Incomplete video: expected {expected_frames} frames, found {count} in {path}')
    return stream
