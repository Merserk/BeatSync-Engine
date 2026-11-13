"""Measure 120 seconds of a real H.264 source through native decoder routes."""
import json
from pathlib import Path
import time
from ffmpeg_runtime import FFMPEG_EXE, run_media_command
from gpu_devices import require_cuda_gpu, require_discrete_gpu, get_dxgi_gpu

ROOT = Path(__file__).resolve().parents[1]


def main():
    cuda, vulkan = require_cuda_gpu()['index'], require_discrete_gpu()['index']
    routes = [
        ('cpu', [], 'fps=6,scale=384:216:flags=fast_bilinear'),
        ('cuda', ['-hwaccel', 'cuda', '-hwaccel_device', str(cuda), '-hwaccel_output_format', 'cuda'],
         'fps=6,scale_cuda=384:216,hwdownload,format=nv12'),
        ('cuvid_resize', ['-hwaccel', 'cuda', '-hwaccel_device', str(cuda), '-hwaccel_output_format', 'cuda',
                          '-c:v', 'h264_cuvid', '-resize', '384x216'], 'fps=6,hwdownload,format=nv12'),
        ('vulkan', ['-init_hw_device', f'vulkan=dgpu:{vulkan}', '-filter_hw_device', 'dgpu',
                    '-hwaccel', 'vulkan', '-hwaccel_device', 'dgpu', '-hwaccel_output_format', 'vulkan'],
         'fps=6,libplacebo=w=384:h=216:format=nv12,hwdownload,format=nv12'),
        ('d3d11', ['-init_hw_device', f'd3d11va=dgpu:{get_dxgi_gpu()["index"]}', '-filter_hw_device', 'dgpu',
                   '-hwaccel', 'd3d11va', '-hwaccel_device', 'dgpu', '-hwaccel_output_format', 'd3d11'],
         'fps=6,scale_d3d11=384:216:format=nv12,hwdownload,format=nv12'),
        ('d3d12', ['-init_hw_device', f'd3d12va=dgpu:{get_dxgi_gpu()["index"]}', '-filter_hw_device', 'dgpu',
                   '-hwaccel', 'd3d12va', '-hwaccel_device', 'dgpu', '-hwaccel_output_format', 'd3d12'],
         'fps=6,scale_d3d12=384:216:format=nv12,hwdownload,format=nv12'),
    ]
    report = []
    for label, args, vf in routes:
        cmd = [FFMPEG_EXE, '-nostdin', '-v', 'error'] + args
        cmd += ['-ss', '120', '-t', '120', '-i', str(ROOT / 'for amv' / '1.mkv'),
                '-map', '0:v:0', '-vf', vf, '-an', '-sn', '-dn', '-f', 'null', '-']
        tick = time.perf_counter()
        result = run_media_command(cmd, timeout=100)
        item = {'route': label, 'seconds': time.perf_counter() - tick, 'passed': result.returncode == 0, 'error': result.stderr[-500:]}
        report.append(item)
        print(json.dumps(item), flush=True)
    (ROOT / 'output' / 'gpu_priority' / 'video_route_benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
