"""Decode and resize on the automatically selected discrete GPU.

Hardware decoder support is tested on the actual input, not inferred from an
encoder listing. Unsupported codecs retain GPU filtering after CPU decoding.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import subprocess
import tempfile
import threading
import contextlib
import math
import queue
import re
from collections import deque

import numpy as np

from ffmpeg_runtime import (FFMPEG_EXE, MediaError, run_media_command,
                            video_resolution, video_stream, video_fps)
from gpu_devices import (get_cuda_gpu, get_discrete_gpu,
                         ffmpeg_gpu_environment)


_PROBE_LOCK = threading.RLock()


@dataclass(frozen=True)
class GPUVideoPath:
    backend: str
    hardware_decode: bool
    device_index: int

    @property
    def label(self):
        return f"{self.backend.upper()} {'decode + resize' if self.hardware_decode else 'resize (CPU decode)'}"

    def input_args(self):
        if self.backend == 'cuda':
            args = ['-init_hw_device', f'cuda=beatsync_frames:{self.device_index}',
                    '-filter_hw_device', 'beatsync_frames']
        else:
            args = ['-init_hw_device', f'vulkan=beatsync_frames:{self.device_index}',
                    '-filter_hw_device', 'beatsync_frames']
        if self.hardware_decode:
            args += ['-hwaccel', self.backend, '-hwaccel_device', 'beatsync_frames',
                     '-hwaccel_output_format', self.backend, '-extra_hw_frames', '64']
        return args

    def upload_filters(self):
        return [] if self.hardware_decode else ['format=nv12', 'hwupload']

    def scale_filters(self, width, height, contain=False, pixel_format='nv12'):
        if self.backend == 'cuda':
            scale = f'scale_cuda={width}:{height}:format={pixel_format}:interp_algo=lanczos'
            if contain:
                scale += ':force_original_aspect_ratio=decrease:force_divisible_by=2:reset_sar=1'
                return [scale, f'pad_cuda={width}:{height}:(ow-iw)/2:(oh-ih)/2', 'setsar=1']
            return [scale]
        scale = f'libplacebo=w={width}:h={height}:format={pixel_format}'
        if contain:
            scale += ':fit_mode=contain:reset_sar=1:fillcolor=black'
        # The named filter device and hardware frames both refer to the selected
        # physical card; libplacebo must not create its own default GPU context.
        return [scale]


@lru_cache(maxsize=64)
def _probe_path(path, size, modified, backend, hardware_decode, device_index):
    route = GPUVideoPath(backend, hardware_decode, device_index)
    cmd = [FFMPEG_EXE, '-nostdin', '-hide_banner', '-v', 'error'] + route.input_args()
    filters = route.upload_filters() + route.scale_filters(64, 64)
    if backend == 'vulkan' and hardware_decode:
        # libplacebo can automatically upload CPU frames, so it cannot prove
        # hardware decoding. scale_vulkan strictly requires Vulkan surfaces.
        filters = ['scale_vulkan=w=64:h=64:format=nv12']
    cmd += ['-i', path, '-map', '0:v:0', '-vf', ','.join(filters), '-frames:v', '1',
            '-an', '-sn', '-dn', '-f', 'null', '-']
    try:
        result = run_media_command(cmd, timeout=45)
        return result.returncode == 0, result.stderr.strip()[-800:]
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)


def get_gpu_video_path(video_file, backend=None, hardware_decode=True):
    """Return a verified GPU route, or None when GPU filtering cannot run."""
    device = get_discrete_gpu()
    if not device['available']:
        return None
    cuda = get_cuda_gpu()
    backend = backend or ('cuda' if cuda['available'] else 'vulkan')
    if backend == 'cuda' and not cuda['available']:
        return None
    index = cuda['index'] if backend == 'cuda' else device['index']
    file = Path(video_file).resolve()
    if not file.is_file():
        return None
    stat = file.stat()
    with _PROBE_LOCK:
        for hardware_decode in ((True, False) if hardware_decode else (False,)):
            valid, error = _probe_path(str(file), stat.st_size, stat.st_mtime_ns,
                                      backend, hardware_decode, index)
            if valid:
                return GPUVideoPath(backend, hardware_decode, index)
            if hardware_decode:
                print(f'   GPU decoder unavailable for {file.name}; testing GPU resize with CPU decode: {error[-300:]}')
        print(f'   GPU video processing unavailable for {file.name}: {error}')
    return None


def selection_expression(indices):
    """Balanced, lazy binary search: FFmpeg's expression recursion is bounded.

    A flat sum of thousands of eq() calls exceeds FFmpeg's parser depth and
    performs thousands of comparisons per decoded frame. if() evaluates only
    the selected branch, so this tree costs logarithmic comparisons.
    """
    values = sorted(set(int(i) for i in indices))
    if not values or values[0] < 0:
        raise ValueError('Expected nonnegative frame indices')
    def tree(start, end):
        if end - start == 1:
            return f'eq(n,{values[start]})'
        middle = (start + end) // 2
        return f'if(lt(n,{values[middle]}),{tree(start, middle)},{tree(middle, end)})'
    return tree(0, len(values))


def iter_sampled_frames(video_file, frame_indices, max_width=768, route=None):
    """Yield exact, ordered BGR source frames with GPU decode/scale.

    A filter file avoids Windows' command-line length limit. Frames stream
    through a pipe; the caller decides how much to retain. stderr goes to a
    temporary file so a full pipe cannot deadlock the decoder.
    """
    targets = sorted(set(int(i) for i in frame_indices))
    if not targets:
        return
    route = route or get_gpu_video_path(video_file)
    if route is None:
        raise MediaError('No working GPU video route')
    width, height = video_resolution(video_file)
    width_out = min(int(max_width), width)
    # YUV hardware surfaces require even dimensions. Aspect ratio is preserved
    # via even-rounded height derived from the source dimensions.
    width_out = max(2, width_out // 2 * 2)
    height_out = max(2, int(height * width_out / width) // 2 * 2)
    filters = [f"select='{selection_expression(targets)}'"]
    filters += route.upload_filters() + route.scale_filters(width_out, height_out)
    filters += ['hwdownload', 'format=nv12', 'format=bgr24']
    frame_bytes = width_out * height_out * 3
    with tempfile.TemporaryDirectory(prefix='beatsync_gpu_frames_') as temporary:
        filter_path = Path(temporary) / 'frames.txt'
        filter_path.write_text(','.join(filters), encoding='utf8')
        with (Path(temporary) / 'stderr.log').open('w+b') as stderr:
            cmd = [FFMPEG_EXE, '-nostdin', '-hide_banner', '-v', 'error'] + route.input_args()
            cmd += ['-i', str(video_file), '-map', '0:v:0', '-/filter:v', str(filter_path),
                    '-frames:v', str(len(targets)), '-an', '-sn', '-dn',
                    '-fps_mode', 'passthrough', '-f', 'rawvideo', '-']
            process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=stderr, env=ffmpeg_gpu_environment(),
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            deadline = threading.Timer(max(120, min(600, targets[-1] / 100)), process.kill)
            deadline.daemon = True
            deadline.start()
            try:
                for index in targets:
                    data = process.stdout.read(frame_bytes)
                    if len(data) != frame_bytes:
                        stderr.flush()
                        stderr.seek(0)
                        raise MediaError(f'GPU frame sampling stopped at source frame {index}: '
                                         + stderr.read().decode('utf8', errors='replace')[-1200:])
                    yield index, np.frombuffer(data, np.uint8).reshape(height_out, width_out, 3)
                code = process.wait(timeout=30)
                if code:
                    stderr.seek(0)
                    raise MediaError(stderr.read().decode('utf8', errors='replace')[-1200:])
            finally:
                deadline.cancel()
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=15)
                process.stdout.close()


@contextlib.contextmanager
def decoded_analysis_frames(video_file, max_width=768, route=None, analysis_fps=6.0):
    """One GPU pass supplies scene scores and dense low-resolution samples.

    Six samples per second exceed the per-window scorer's sampling rate.
    Raw low-resolution YUV frames use a temporary seekable spool, avoiding
    both another full source decode and lossy/redundant proxy encoding.
    Large spools roll to disk after 64 MB and are removed when analysis ends.
    """
    route = route or get_gpu_video_path(video_file)
    if route is None:
        raise MediaError('No working GPU video route')
    fps = video_fps(video_file)
    stream = video_stream(video_file)
    if stream.get('color_space') == 'bt2020c':
        raise MediaError('Constant-luminance BT.2020 requires native FFmpeg color conversion')
    if stream.get('avg_frame_rate') != stream.get('r_frame_rate'):
        raise MediaError('Variable frame rate requires the direct timestamp-aware source path')
    width, height = video_resolution(video_file)
    width = max(2, min(max_width, width) // 2 * 2)
    height = max(2, int(height * width / video_resolution(video_file)[0]) // 2 * 2)
    frame_bytes = width * height * 3 // 2
    filters = [f'fps={analysis_fps}'] + route.upload_filters() + route.scale_filters(width, height)
    filters += ['hwdownload', 'format=nv12']
    scene_changes = []
    previous = None
    previous_mafd = 0.0
    count = 0
    with tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024, mode='w+b') as spool, \
         tempfile.TemporaryFile(mode='w+b') as stderr:
        cmd = [FFMPEG_EXE, '-nostdin', '-hide_banner', '-v', 'error'] + route.input_args()
        cmd += ['-i', str(video_file), '-map', '0:v:0', '-vf', ','.join(filters),
                '-an', '-sn', '-dn', '-fps_mode', 'passthrough', '-f', 'rawvideo', '-']
        process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=stderr, env=ffmpeg_gpu_environment(),
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        deadline = threading.Timer(600, process.kill)
        deadline.daemon = True
        deadline.start()
        try:
            while data := process.stdout.read(frame_bytes):
                if len(data) != frame_bytes:
                    raise MediaError('GPU analysis produced an incomplete raw frame')
                spool.write(data)
                luma = np.frombuffer(data, np.uint8, count=width * height).astype(np.int16)
                if previous is not None:
                    mafd = float(np.abs(luma - previous).mean())
                    score = min(mafd, abs(mafd - previous_mafd)) / 100.0
                    if score > .27:
                        scene_changes.append(round(count / analysis_fps, 3))
                    previous_mafd = mafd
                previous = luma
                count += 1
            code = process.wait(timeout=30)
            if code or not count:
                stderr.seek(0)
                raise MediaError(stderr.read().decode('utf8', errors='replace')[-1200:] or 'GPU analysis produced no frames')
            spool.flush()
            class Frames:
                def __init__(self):
                    self.count = count
                    self.scene_changes = scene_changes
                    self.backend = route.label
                    self.analysis_fps = analysis_fps
                    self.source_fps = fps
                    self.colorspace = stream.get('color_space') or 'smpte170m'
                    self.full_range = stream.get('color_range') == 'pc'
                    self.source_count = round(count * fps / analysis_fps)
                    self.color_conversion_backend = 'Taichi Vulkan'
                def source_index(self, index):
                    sample = max(0, min(count - 1, round((index + 1) * analysis_fps / fps - .5)))
                    return max(0, math.ceil((sample + .5) * fps / analysis_fps) - 1)
                def read_bgr_batch(self, indices):
                    from taichi_backend import nv12_frames_to_bgr
                    raw = []
                    for index in indices:
                        sample = round((index + 1) * analysis_fps / fps - .5)
                        if not 0 <= index < self.source_count:
                            raise IndexError('Analysis sampling beyond the source video')
                        sample = max(0, min(count - 1, sample))
                        spool.seek(sample * frame_bytes)
                        raw.append(np.frombuffer(spool.read(frame_bytes), np.uint8).reshape(height * 3 // 2, width))
                    values = np.stack(raw)
                    if self.color_conversion_backend == 'Taichi Vulkan':
                        try:
                            return nv12_frames_to_bgr(values, self.colorspace, self.full_range)
                        except RuntimeError as exc:
                            self.color_conversion_backend = 'CPU fallback'
                            print(f'   Vulkan color conversion unavailable; using small CPU conversion: {exc}')
                    # Preserve the source's matrix/range in the fallback, too.
                    kr, kb = {'bt709': (.2126, .0722), 'bt2020nc': (.2627, .0593)}.get(self.colorspace, (.299, .114))
                    y = values[:, :height].astype(np.float32)
                    uv = values[:, height:].astype(np.float32)
                    u = np.repeat(np.repeat(uv[:, :, ::2], 2, axis=1), 2, axis=2) - 128
                    v = np.repeat(np.repeat(uv[:, :, 1::2], 2, axis=1), 2, axis=2) - 128
                    if not self.full_range:
                        y = (y - 16) * (255 / 219)
                        u, v = u * (255 / 224), v * (255 / 224)
                    r, b = y + (2 - 2 * kr) * v, y + (2 - 2 * kb) * u
                    g = y - kr * (2 - 2 * kr) / (1 - kr - kb) * v - kb * (2 - 2 * kb) / (1 - kr - kb) * u
                    return np.clip(np.rint(np.stack([b, g, r], axis=-1)), 0, 255).astype(np.uint8)
            yield Frames()
        finally:
            deadline.cancel()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=15)
            process.stdout.close()


def iter_interval_frames(video_file, start, duration, max_width=768, use_gpu=True, with_timestamps=False):
    """Decode every frame of one selected interval, without scanning the whole file.

    Accurate input seeking discards the GOP preroll. This is also suitable for
    VFR sources, since interval selection is by timestamp rather than frame index.
    """
    route = get_gpu_video_path(video_file) if use_gpu else None
    width, height = video_resolution(video_file)
    width_out = max(2, min(width, max_width) // 2 * 2)
    height_out = max(2, int(height * width_out / width) // 2 * 2)
    if route:
        filters = route.upload_filters() + route.scale_filters(width_out, height_out)
        filters += ['hwdownload', 'format=nv12', 'format=bgr24']
    else:
        filters = [f'scale={width_out}:{height_out}:flags=lanczos', 'format=bgr24']
    cmd = [FFMPEG_EXE, '-nostdin', '-hide_banner', '-v', 'error']
    if with_timestamps:
        cmd += ['-copyts']
        # Metadata prints native PTS without a pixel checksum or another decode.
        filters += ['metadata=mode=add:key=beatsync_frame:value=1',
                    "metadata=mode=print:file='pipe\\:2':direct=1"]
    if route:
        cmd += route.input_args()
    cmd += ['-ss', f'{start:.9f}', '-t', f'{duration:.9f}', '-i', str(video_file),
            '-map', '0:v:0', '-vf', ','.join(filters), '-an', '-sn', '-dn',
            '-fps_mode', 'passthrough', '-f', 'rawvideo', '-']
    frame_bytes = width_out * height_out * 3
    with tempfile.TemporaryFile(mode='w+b') as stderr:
        process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE if with_timestamps else stderr, env=ffmpeg_gpu_environment(),
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        timestamps = queue.Queue()
        errors = deque(maxlen=20)
        reader = None
        if with_timestamps:
            def read_metadata():
                for raw in process.stderr:
                    line = raw.decode('utf8', errors='replace').strip()
                    match = re.search(r'^frame:\d+\s+pts:\S+\s+pts_time:([-+\d.eE]+)', line)
                    if match:
                        timestamps.put(float(match.group(1)))
                    elif line != 'beatsync_frame=1':
                        errors.append(line)
            reader = threading.Thread(target=read_metadata, daemon=True)
            reader.start()
        deadline = threading.Timer(max(60, duration * 10), process.kill)
        deadline.daemon = True
        deadline.start()
        count = 0
        try:
            while data := process.stdout.read(frame_bytes):
                if len(data) != frame_bytes:
                    raise MediaError('Source verification produced an incomplete frame')
                count += 1
                frame = np.frombuffer(data, np.uint8).reshape(height_out, width_out, 3)
                if with_timestamps:
                    try:
                        timestamp = timestamps.get(timeout=10)
                    except queue.Empty as exc:
                        raise MediaError('Missing source-frame timestamp: ' + '\n'.join(errors)) from exc
                    yield timestamp, frame
                else:
                    yield frame
            code = process.wait(timeout=30)
            if code or not count:
                stderr.seek(0)
                message = '\n'.join(errors) if with_timestamps else stderr.read().decode('utf8', errors='replace')[-1200:]
                raise MediaError(message or
                                 'Source verification produced no frames')
        finally:
            deadline.cancel()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=15)
            process.stdout.close()
            if reader:
                reader.join(timeout=5)
                process.stderr.close()
