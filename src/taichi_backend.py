"""Vulkan kernels for BeatSync's dense audio and video analysis.

The public operations accept NumPy arrays and return host results. Taichi owns
the GPU transfers, and the runtime must successfully execute a Vulkan kernel
before this backend is advertised as available.
"""

import os
from pathlib import Path
import threading

import numpy as np
import taichi as ti
from gpu_devices import require_discrete_gpu


_LOCK = threading.RLock()
_INFO = None
_VIDEO_BLOCK = 256
_AUDIO_BLOCK = 128
_MEDIAN_NETWORK = tuple((i, i ^ stride, (i & span) == 0)
                        for span in (2, 4, 8, 16, 32)
                        for stride in (16, 8, 4, 2, 1) if stride < span
                        for i in range(32) if (i ^ stride) > i)


@ti.kernel
def _probe(result: ti.types.ndarray(dtype=ti.i32, ndim=1)):
    result[0] = 42


def get_backend_info():
    global _INFO
    with _LOCK:
        if _INFO is not None:
            return dict(_INFO)
        info = {"available": False, "name": "No Vulkan GPU", "backend": "Taichi Vulkan", "vulkan_version": "None", "version": ti.__version__}
        try:
            if os.environ.get("BEATSYNC_DISABLE_GPU") == "1":
                raise RuntimeError("Disabled by BEATSYNC_DISABLE_GPU")
            device = require_discrete_gpu()
            os.environ['TI_VISIBLE_DEVICE'] = str(device['index'])
            ti._lib.core.set_vulkan_visible_device(str(device["index"]))
            cache = Path(__file__).resolve().parents[1] / "bin" / "taichi-cache"
            ti.init(arch=ti.vulkan, enable_fallback=False, log_level=ti.ERROR,
                    offline_cache=True, offline_cache_file_path=str(cache), fast_math=False)
            if ti.lang.impl.current_cfg().arch != ti.vulkan:
                raise RuntimeError("Taichi did not select the Vulkan backend")
            result = np.zeros(1, dtype=np.int32)
            _probe(result)
            ti.sync()
            if result[0] != 42:
                raise RuntimeError("Vulkan compute verification failed")
            info.update(available=True, name=device["name"], device_index=device["index"], vulkan_version=device["version"])
        except Exception as exc:
            info["error"] = str(exc)
        _INFO = info
        return dict(info)


def _require_vulkan():
    info = get_backend_info()
    if not info["available"]:
        raise RuntimeError(info.get("error", "Taichi Vulkan is unavailable"))


@ti.func
def _gray(frames: ti.template(), frame, y, x):
    return (0.114 * ti.cast(frames[frame, y, x, 0], ti.f32)
            + 0.587 * ti.cast(frames[frame, y, x, 1], ti.f32)
            + 0.299 * ti.cast(frames[frame, y, x, 2], ti.f32))


@ti.kernel
def _frame_partials(frames: ti.types.ndarray(dtype=ti.u8, ndim=4),
                    partials: ti.types.ndarray(dtype=ti.f32, ndim=3)):
    for frame, block in ti.ndrange(frames.shape[0], partials.shape[1]):
        sums = ti.Vector.zero(ti.f32, 10)
        height, width = frames.shape[1], frames.shape[2]
        anchor_gray = _gray(frames, frame, 0, 0)
        anchor_b = ti.cast(frames[frame, 0, 0, 0], ti.f32)
        anchor_g = ti.cast(frames[frame, 0, 0, 1], ti.f32)
        anchor_r = ti.cast(frames[frame, 0, 0, 2], ti.f32)
        anchor_rg = ti.abs(anchor_r - anchor_g)
        anchor_yb = ti.abs(0.5 * (anchor_r + anchor_g) - anchor_b)
        for offset in range(_VIDEO_BLOCK):
            pixel = block * _VIDEO_BLOCK + offset
            if pixel < height * width:
                y, x = pixel // width, pixel % width
                b = ti.cast(frames[frame, y, x, 0], ti.f32)
                g = ti.cast(frames[frame, y, x, 1], ti.f32)
                r = ti.cast(frames[frame, y, x, 2], ti.f32)
                gray = 0.114 * b + 0.587 * g + 0.299 * r
                maximum, minimum = ti.max(r, g, b), ti.min(r, g, b)
                saturation = 0.0
                if maximum > 1e-6:
                    saturation = (maximum - minimum) / maximum
                up, down = ti.max(0, y - 1), ti.min(height - 1, y + 1)
                left, right = ti.max(0, x - 1), ti.min(width - 1, x + 1)
                if height > 1:
                    if y == 0:
                        up = 1
                    if y == height - 1:
                        down = height - 2
                if width > 1:
                    if x == 0:
                        left = 1
                    if x == width - 1:
                        right = width - 2
                lap = (_gray(frames, frame, up, x) + _gray(frames, frame, down, x)
                       + _gray(frames, frame, y, left) + _gray(frames, frame, y, right) - 4.0 * gray)
                rg, yb = ti.abs(r - g), ti.abs(0.5 * (r + g) - b)
                motion = 0.0
                if frame > 0:
                    motion = ti.abs(gray - _gray(frames, frame - 1, y, x))
                # Shift moments to preserve low contrast and uniform colors
                # without requiring float64 support on the Vulkan device.
                gray_delta, rg_delta, yb_delta = gray - anchor_gray, rg - anchor_rg, yb - anchor_yb
                sums += ti.Vector([gray_delta, gray_delta * gray_delta, saturation, lap, lap * lap,
                                   rg_delta, rg_delta * rg_delta, yb_delta, yb_delta * yb_delta, motion])
        for metric in ti.static(range(10)):
            partials[frame, block, metric] = sums[metric]


def frame_statistics(frames):
    """Per-frame sums of luminance, saturation, Laplacian, color and motion."""
    _require_vulkan()
    stack = np.ascontiguousarray(frames, dtype=np.uint8)
    if stack.ndim != 4 or stack.shape[3] != 3 or min(stack.shape[:3]) < 1:
        raise ValueError("Expected nonempty BGR frames with shape (frames, height, width, 3)")
    blocks = (stack.shape[1] * stack.shape[2] + _VIDEO_BLOCK - 1) // _VIDEO_BLOCK
    partials = np.empty((stack.shape[0], blocks, 10), dtype=np.float32)
    with _LOCK:
        _frame_partials(stack, partials)
    stats = np.sum(partials, axis=1, dtype=np.float64) / (stack.shape[1] * stack.shape[2])
    b, g, r = stack[:, 0, 0, :].astype(np.float64).T
    anchors = (0.114 * b + 0.587 * g + 0.299 * r, np.abs(r - g), np.abs(0.5 * (r + g) - b))
    for mean_index, square_index, anchor in zip((0, 5, 7), (1, 6, 8), anchors):
        shifted_mean = stats[:, mean_index].copy()
        stats[:, square_index] += 2 * anchor * shifted_mean + anchor ** 2
        stats[:, mean_index] += anchor
    return stats


@ti.kernel
def _rhythm_partials(spectrum: ti.types.ndarray(dtype=ti.f32, ndim=3),
                     bounds: ti.types.ndarray(dtype=ti.i32, ndim=2),
                     partials: ti.types.ndarray(dtype=ti.f32, ndim=3)):
    for frame, block in ti.ndrange(spectrum.shape[0], partials.shape[1]):
        sums = ti.Vector.zero(ti.f32, 4)
        for offset in range(_AUDIO_BLOCK):
            frequency = block * _AUDIO_BLOCK + offset
            if frequency < spectrum.shape[1]:
                real, imag = spectrum[frame, frequency, 0], spectrum[frame, frequency, 1]
                magnitude = ti.sqrt(real * real + imag * imag)
                for band in ti.static(range(4)):
                    if bounds[band, 0] <= frequency < bounds[band, 1]:
                        sums[band] += magnitude
        for band in ti.static(range(4)):
            partials[frame, block, band] = sums[band]


def rhythm_band_sums(stft, frequencies):
    """Sum complex spectral magnitudes for kick, bass, clap and hihat bands."""
    _require_vulkan()
    spectrum = np.ascontiguousarray(np.asarray(stft, dtype=np.complex64).T)
    values = spectrum.view(np.float32).reshape(spectrum.shape + (2,))
    frequencies = np.asarray(frequencies)
    bounds = np.asarray([(np.searchsorted(frequencies, low), np.searchsorted(frequencies, high, side="right"))
                         for low, high in ((35, 145), (35, 220), (150, 4200), (4200, np.inf))], dtype=np.int32)
    blocks = (spectrum.shape[1] + _AUDIO_BLOCK - 1) // _AUDIO_BLOCK
    partials = np.empty((spectrum.shape[0], blocks, 4), dtype=np.float32)
    with _LOCK:
        _rhythm_partials(values, bounds, partials)
    return np.sum(partials, axis=1, dtype=np.float64).T


@ti.func
def _reflect_index(index, length):
    value = index % (2 * length)
    if value >= length:
        value = 2 * length - 1 - value
    return value


@ti.func
def _median31(values):
    for left, right, ascending in ti.static(_MEDIAN_NETWORK):
        low, high = ti.min(values[left], values[right]), ti.max(values[left], values[right])
        if ti.static(ascending):
            values[left], values[right] = low, high
        else:
            values[left], values[right] = high, low
    return values[15]


@ti.kernel
def _hpss_masks(magnitude: ti.types.ndarray(dtype=ti.f32, ndim=2),
                masks: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    # librosa's default HPSS uses reflect-padded 31-bin median filters.
    # Both directions and the stable power-2 soft mask share one Vulkan pass.
    ti.loop_config(block_dim=64)
    for frequency, frame in ti.ndrange(magnitude.shape[0], magnitude.shape[1]):
        horizontal = ti.Vector.zero(ti.f32, 32)
        vertical = ti.Vector.zero(ti.f32, 32)
        for offset in ti.static(range(31)):
            horizontal[offset] = magnitude[frequency, _reflect_index(frame + offset - 15, magnitude.shape[1])]
            vertical[offset] = magnitude[_reflect_index(frequency + offset - 15, magnitude.shape[0]), frame]
        horizontal[31], vertical[31] = ti.math.inf, ti.math.inf
        harmonic, percussive = _median31(horizontal), _median31(vertical)
        maximum = ti.max(harmonic, percussive)
        mask = 0.5
        if maximum > 0:
            harmonic /= maximum
            percussive /= maximum
            mask = harmonic * harmonic / (harmonic * harmonic + percussive * percussive)
        masks[frequency, frame] = mask


def hpss_masks(spectrum):
    """Default HPSS harmonic mask, equivalent to librosa's two median filters."""
    _require_vulkan()
    magnitude = np.ascontiguousarray(np.abs(spectrum), dtype=np.float32)
    if magnitude.ndim != 2 or min(magnitude.shape) < 1 or not np.isfinite(magnitude).all():
        raise ValueError('Expected a finite, nonempty frequency/time spectrum')
    masks = np.empty_like(magnitude)
    with _LOCK:
        _hpss_masks(magnitude, masks)
    return masks


def harmonic_percussive_audio(y):
    """GPU median separation with the existing fast CPU FFT/reconstruction."""
    import librosa
    spectrum = librosa.stft(y)
    harmonic_mask = hpss_masks(spectrum)
    harmonic = librosa.istft(spectrum * harmonic_mask, length=len(y))
    percussive = librosa.istft(spectrum * (1.0 - harmonic_mask), length=len(y))
    return harmonic, percussive


def synchronize():
    if get_backend_info()["available"]:
        with _LOCK:
            ti.sync()


@ti.kernel
def _nv12_bgr(frames: ti.types.ndarray(dtype=ti.u8, ndim=3),
              colors: ti.types.ndarray(dtype=ti.f32, ndim=1),
              output: ti.types.ndarray(dtype=ti.u8, ndim=4)):
    for frame, y, x in ti.ndrange(output.shape[0], output.shape[1], output.shape[2]):
        height = output.shape[1]
        luminance = (ti.cast(frames[frame, y, x], ti.f32) - colors[0]) * colors[1]
        u = (ti.cast(frames[frame, height + y // 2, (x // 2) * 2], ti.f32) - 128.0) * colors[2]
        v = (ti.cast(frames[frame, height + y // 2, (x // 2) * 2 + 1], ti.f32) - 128.0) * colors[2]
        kr, kb = colors[3], colors[4]
        kg = 1.0 - kr - kb
        red = luminance + (2.0 - 2.0 * kr) * v
        blue = luminance + (2.0 - 2.0 * kb) * u
        green = luminance - kr * (2.0 - 2.0 * kr) / kg * v - kb * (2.0 - 2.0 * kb) / kg * u
        output[frame, y, x, 0] = ti.cast(ti.round(ti.min(255.0, ti.max(0.0, blue))), ti.u8)
        output[frame, y, x, 1] = ti.cast(ti.round(ti.min(255.0, ti.max(0.0, green))), ti.u8)
        output[frame, y, x, 2] = ti.cast(ti.round(ti.min(255.0, ti.max(0.0, red))), ti.u8)


def nv12_frames_to_bgr(frames, colorspace='bt709', full_range=False):
    """GPU YUV conversion with the source matrix/range, including BT.709/2020."""
    _require_vulkan()
    values = np.ascontiguousarray(frames, dtype=np.uint8)
    if values.ndim != 3 or values.shape[1] % 3 or values.shape[2] % 2:
        raise ValueError('Expected NV12 frames with even luma dimensions')
    height = values.shape[1] * 2 // 3
    if colorspace == 'bt2020c':
        raise ValueError('Constant-luminance BT.2020 requires the native FFmpeg color conversion path')
    kr, kb = {'bt709': (.2126, .0722), 'bt2020nc': (.2627, .0593)}.get(colorspace, (.299, .114))
    colors = np.asarray([0 if full_range else 16, 1 if full_range else 255 / 219,
                         1 if full_range else 255 / 224, kr, kb], dtype=np.float32)
    output = np.empty((len(values), height, values.shape[2], 3), dtype=np.uint8)
    with _LOCK:
        _nv12_bgr(values, colors, output)
    return output
