"""FFmpeg Vulkan Video settings and real hardware capability checks."""

from functools import lru_cache
import subprocess

from ffmpeg_runtime import FFMPEG_EXE, get_runtime_info, run_media_command
from gpu_devices import require_discrete_gpu, require_cuda_gpu


VULKAN_ENCODERS = {
    "h264_vulkan": "Vulkan Video H.264",
    "hevc_vulkan": "Vulkan Video H.265",
}
NVENC_ENCODERS = {'h264_nvenc': 'NVIDIA NVENC H.264', 'hevc_nvenc': 'NVIDIA NVENC HEVC (H.265)'}
HARDWARE_ENCODERS = {**NVENC_ENCODERS, **VULKAN_ENCODERS}


def vulkan_quality_args(encoder):
    if encoder not in VULKAN_ENCODERS:
        raise ValueError(f"Unsupported Vulkan Video encoder: {encoder}")
    return ["-c:v", encoder, "-rc_mode", "cqp", "-qp", "18",
            "-tune", "hq", "-usage", "transcode", "-profile:v",
            "high" if encoder == "h264_vulkan" else "main"]


def _device_args(index):
    return ["-init_hw_device", f"vulkan=beatsync_video:{index}",
            "-filter_hw_device", "beatsync_video"]


def nvenc_device_args():
    return ['-gpu', str(require_cuda_gpu()['index'])]


def cuda_decode_args():
    return ['-hwaccel', 'cuda', '-hwaccel_device', str(require_cuda_gpu()['index'])]


def _nvenc_args(encoder, tune='hq', include_pix_fmt=True):
    args = ['-c:v', encoder] + nvenc_device_args() + ['-preset', 'p7', '-tune', tune, '-rc', 'vbr',
            '-b:v', '0', '-cq', '1', '-multipass', 'fullres', '-rc-lookahead', '32',
            '-spatial-aq', '1', '-temporal-aq', '1', '-aq-strength', '12',
            '-b_ref_mode', 'middle', '-profile:v', 'high' if encoder == 'h264_nvenc' else 'main']
    if include_pix_fmt:
        args += ['-pix_fmt', 'yuv420p']
    return args


@lru_cache(maxsize=2)
def _probe_nvenc_encoder(encoder):
    try:
        device = require_cuda_gpu()
        if encoder not in get_runtime_info()['encoders']:
            return {'available': False, 'encoder': encoder, 'error': 'Encoder missing from FFmpeg'}
        error = ''
        # UHQ is optional; the actual GPU and driver must accept the complete
        # production configuration, rather than just a default encode probe.
        for tune in (('uhq', 'hq') if encoder == 'hevc_nvenc' else ('hq',)):
            cmd = [FFMPEG_EXE, '-nostdin', '-hide_banner', '-loglevel', 'error',
                   '-f', 'lavfi', '-i', 'color=black:s=256x256:r=30']
            cmd += _nvenc_args(encoder, tune)
            cmd += ['-frames:v', '3', '-an', '-f', 'null', '-']
            result = run_media_command(cmd, timeout=15)
            if result.returncode == 0:
                return {'available': True, 'encoder': encoder, 'tune': tune,
                        'device_index': device['index'], 'name': device['name']}
            error = result.stderr.strip()[-1800:]
        return {'available': False, 'encoder': encoder, 'name': device['name'],
                'device_index': device['index'], 'error': error}
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return {'available': False, 'encoder': encoder, 'error': str(exc)}


def get_encoder_support(encoder):
    if encoder in VULKAN_ENCODERS:
        return get_vulkan_encoder_support(encoder)
    if encoder in NVENC_ENCODERS:
        return dict(_probe_nvenc_encoder(encoder))
    return {'available': encoder in get_runtime_info()['encoders'], 'encoder': encoder}


def nvenc_quality_args(encoder, include_pix_fmt=True):
    if encoder not in NVENC_ENCODERS:
        raise ValueError(f'Unsupported NVENC encoder: {encoder}')
    support = get_encoder_support(encoder)
    if not support['available']:
        raise RuntimeError(f"{NVENC_ENCODERS[encoder]} is unavailable: {support['error']}")
    return _nvenc_args(encoder, support['tune'], include_pix_fmt)


@lru_cache(maxsize=2)
def _probe_vulkan_encoder(encoder):
    """Encode a frame; an encoder listing alone does not prove driver support."""
    try:
        device = require_discrete_gpu()
        if encoder not in get_runtime_info()['encoders']:
            return {'available': False, 'encoder': encoder, 'error': 'Encoder missing from FFmpeg'}
        cmd = [FFMPEG_EXE, "-nostdin", "-hide_banner", "-loglevel", "error"]
        cmd += _device_args(device["index"])
        cmd += ["-f", "lavfi", "-i", "color=black:s=256x256:r=30",
                "-vf", "format=nv12,hwupload"]
        cmd += vulkan_quality_args(encoder)
        cmd += ["-frames:v", "1", "-an", "-f", "null", "-"]
        result = run_media_command(cmd, timeout=15)
        info = {"available": result.returncode == 0, "device_index": device["index"],
                "name": device["name"], "encoder": encoder}
        if result.returncode:
            info['error'] = result.stderr.strip()[-1800:] or 'FFmpeg encoding probe failed'
        return info
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "encoder": encoder, "error": str(exc)}


def get_vulkan_encoder_support(encoder):
    if encoder not in VULKAN_ENCODERS:
        raise ValueError(f"Unsupported Vulkan Video encoder: {encoder}")
    return dict(_probe_vulkan_encoder(encoder))


def vulkan_device_args(encoder):
    support = get_vulkan_encoder_support(encoder)
    if not support["available"]:
        raise RuntimeError(f"{VULKAN_ENCODERS[encoder]} is unavailable: {support['error']}")
    return _device_args(support["device_index"])


@lru_cache(maxsize=1)
def get_prores_gpu_support():
    """Probe the production ProRes HQ profile on the detected dGPU's Vulkan compute."""
    encoder = 'prores_ks_vulkan'
    try:
        device = require_discrete_gpu()
        if encoder not in get_runtime_info()['encoders']:
            return {'available': False, 'error': 'Vulkan ProRes encoder missing from FFmpeg'}
        cmd = [FFMPEG_EXE, '-nostdin', '-v', 'error'] + _device_args(device['index'])
        cmd += ['-f', 'lavfi', '-i', 'color=black:s=256x256:r=24',
                '-vf', 'format=nv12,hwupload,libplacebo=format=yuv422p10le',
                '-c:v', encoder, '-profile:v', '3', '-vendor', 'apl0', '-tag:v', 'apch',
                '-async_depth', '4', '-frames:v', '1', '-f', 'null', '-']
        result = run_media_command(cmd, timeout=60)
        return {'available': result.returncode == 0, 'encoder': encoder,
                'device_index': device['index'], 'name': device['name'],
                'error': result.stderr.strip()[-1000:]}
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return {'available': False, 'encoder': encoder, 'error': str(exc)}
