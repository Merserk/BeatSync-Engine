"""Verify real GPU execution with deliberately conflicting iGPU overrides.

Uses the existing short FFmpeg and Qwen fixtures. Requires a working discrete
NVIDIA card with Vulkan Video/NVENC support to exercise every hardware path.
"""

from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import gpu_devices as devices


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'dgpu_update'


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gpu_commands = []
    report = {'passed': False, 'conflicting_overrides_ignored': True}
    overrides = {'BEATSYNC_TAICHI_DEVICE': '0', 'TI_VISIBLE_DEVICE': '0',
                 'BEATSYNC_VULKAN_VIDEO_DEVICE': '0', 'BEATSYNC_QWEN_LLAMA_DEVICE': 'Vulkan1',
                 'GGML_VK_VISIBLE_DEVICES': '0', 'CUDA_VISIBLE_DEVICES': '99',
                 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'LLAMA_ARG_DEVICE': 'Vulkan1',
                 'MTMD_BACKEND_DEVICE': 'Vulkan1', 'BEATSYNC_DISABLE_GPU': '0'}
    try:
        with patch.dict('os.environ', overrides):
            report['physical_devices'] = devices._vulkan_devices()
            selected = report['selected'] = devices.require_discrete_gpu()
            cuda = report['cuda'] = devices.require_cuda_gpu()
            assert selected['type'] == 2 and selected['vendor_id'] == 0x10de, selected
            assert cuda['uuid'] == selected['uuid'] or cuda['luid'] == selected['luid']
            print(f"Selected dGPU: {selected['name']} | Vulkan {selected['index']} | CUDA {cuda['index']}", flush=True)

            import taichi_backend
            report['taichi'] = taichi_backend.get_backend_info()
            assert report['taichi']['available'] and report['taichi']['device_index'] == selected['index']
            frames = np.full((3, 64, 96, 3), 128, dtype=np.uint8)
            stats = taichi_backend.frame_statistics(frames)
            np.testing.assert_allclose(stats[:, 0], 128, atol=1e-4)
            spectrum = np.ones((1025, 8), dtype=np.complex64)
            frequency = np.linspace(0, 11025, 1025)
            bands = taichi_backend.rhythm_band_sums(spectrum, frequency)
            assert bands.shape == (4, 8) and np.all(bands >= 0)
            print('PASS Taichi Vulkan frame/audio kernels on selected dGPU', flush=True)

            import ffmpeg_runtime as runtime
            import ffmpeg_processing as media
            import video_encoders as encoders
            import gpu_video
            real_run = runtime.run_media_command

            def traced_run(cmd, *args, **kwargs):
                hardware = False
                if '-c:v' in cmd and cmd[cmd.index('-c:v') + 1] in encoders.HARDWARE_ENCODERS:
                    encoder = cmd[cmd.index('-c:v') + 1]
                    if encoder in encoders.NVENC_ENCODERS:
                        assert '-gpu' in cmd and cmd[cmd.index('-gpu') + 1] == str(cuda['index']), cmd
                    else:
                        assert any(value.startswith('vulkan=') and value.endswith(':' + str(selected['index']))
                                   for value in cmd), cmd
                    hardware = True
                if '-hwaccel' in cmd:
                    backend = cmd[cmd.index('-hwaccel') + 1]
                    assert backend in ('cuda', 'vulkan'), cmd
                    device = cmd[cmd.index('-hwaccel_device') + 1]
                    ordinal = str(cuda['index']) if backend == 'cuda' else str(selected['index'])
                    assert device == ordinal or f'{backend}={device}:{ordinal}' in cmd, cmd
                    hardware = True
                if '-c:v' in cmd and cmd[cmd.index('-c:v') + 1] == 'prores_ks_vulkan':
                    assert any(value.startswith('vulkan=') and value.endswith(':' + str(selected['index'])) for value in cmd), cmd
                    hardware = True
                result = real_run(cmd, *args, **kwargs)
                if hardware:
                    gpu_commands.append({'argv': cmd, 'returncode': result.returncode,
                                         'stderr': result.stderr[-1000:] if result.returncode else ''})
                return result

            with ExitStack() as stack:
                for module in (runtime, media, encoders, gpu_video):
                    stack.enter_context(patch.object(module, 'run_media_command', side_effect=traced_run))
                ffmpeg = load_script('test_ffmpeg_smoke')
                ffmpeg.OUT = OUT / 'ffmpeg'
                ffmpeg.main()
                report['ffmpeg_report'] = str(ffmpeg.OUT / 'report.json')
                original_support = media.get_encoder_support
                def without_nvenc(encoder):
                    return {'available': False} if encoder in encoders.NVENC_ENCODERS else original_support(encoder)
                with patch.object(media, 'get_encoder_support', side_effect=without_nvenc):
                    preview = media.create_browser_preview(str(ffmpeg.OUT / 'prores_ks_app.mov'),
                                                           str(OUT / 'vulkan_preview.mp4'))
                assert [s['codec_name'] for s in runtime.probe_media(preview)['streams']] == ['h264', 'aac']
                runtime.verify_video(preview, 60)
                runtime.checked_output([runtime.FFMPEG_EXE, '-v', 'error', '-xerror', '-i', preview, '-f', 'null', '-'])
                report['vulkan_preview_without_nvenc'] = 'H.264/AAC, 60 frames, decode passed'
                scenes = media.detect_video_scene_changes(str(ffmpeg.OUT / 'scene_change.mp4'),
                                                         use_gpu=True, threshold=.27, analysis_fps=6, analysis_width=256)
                assert any(abs(t - 1.) <= 1 / 6 for t in scenes), scenes
                report['cuda_scene_detection'] = scenes
                production = [command for command in gpu_commands if '-c:v' in command['argv'] or '-frames:v' not in command['argv']]
                assert all(command['returncode'] == 0 for command in production), [c for c in production if c['returncode']]
                report['successful_hardware_commands'] = sum(c['returncode'] == 0 for c in gpu_commands)
                report['unsupported_decoder_probes'] = sum(c['returncode'] != 0 for c in gpu_commands)
            print(f'PASS {len(gpu_commands)} hardware commands explicitly pinned to selected dGPU', flush=True)

            qwen = load_script('test_qwen_smoke')
            device, label = qwen.worker._select_vulkan_device(qwen.worker.LlamaPaths(None))
            assert device is not None and selected['name'] in label, label
            qwen.main(OUT / 'qwen')
            report['qwen_report'] = str(OUT / 'qwen' / 'results.json')
            report['llama_visible_physical_device'] = qwen.worker._llama_env(qwen.worker.LlamaPaths(None))['GGML_VK_VISIBLE_DEVICES']
            report['llama_device_args'] = qwen.worker._llama_device_args(device)
            report['passed'] = True
    finally:
        (OUT / 'gpu_commands.json').write_text(json.dumps(gpu_commands, indent=2), encoding='utf8')
        (OUT / 'report.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf8')
    print('PASS mixed AMD iGPU / NVIDIA dGPU routing, including conflicting environment overrides', flush=True)


if __name__ == '__main__':
    main()
