"""Compare real source frame sampling and scene detection hardware paths."""
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from ffmpeg_runtime import FFMPEG_EXE
from gpu_devices import require_cuda_gpu, ffmpeg_gpu_environment
from ffmpeg_processing import detect_video_scene_changes
from video_analysis import _window_sample_plan

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'gpu_priority'


def selection(values):
    if len(values) == 1:
        return f'eq(n,{values[0]})'
    middle = len(values) // 2
    return f'if(lt(n,{values[middle]}),{selection(values[:middle])},{selection(values[middle:])})'


def main():
    cuda = require_cuda_gpu()['index']
    reports = []
    for source in (1, 2):
        video = ROOT / 'for amv' / f'{source}.mkv'
        data = next(json.loads(p.read_text(encoding='utf8')) for p in (OUT / 'baseline' / 'analysis_cache').glob('*.json')
                    if p.name[:8] in ('356a192b', 'da4b9237') and json.loads(p.read_text(encoding='utf8'))['source_name'] == video.name)
        targets = sorted({i for c in data['candidates'] for i in _window_sample_plan(24, c['start'], c['end'])['frame_indices']})
        script = OUT / f'select_{source}.txt'
        script.write_text("select='" + selection(targets) + "'" +
                          ',scale_cuda=360:202:interp_algo=lanczos,hwdownload,format=nv12,format=bgr24', encoding='utf8')
        cmd = [FFMPEG_EXE, '-nostdin', '-v', 'error', '-hwaccel', 'cuda', '-hwaccel_device', str(cuda),
               '-hwaccel_output_format', 'cuda', '-i', str(video), '-map', '0:v:0', '-/filter:v', str(script),
               '-frames:v', str(len(targets)), '-an', '-sn', '-dn', '-fps_mode', 'passthrough', '-f', 'rawvideo', '-']
        started = time.perf_counter()
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
                                env=ffmpeg_gpu_environment(), creationflags=subprocess.CREATE_NO_WINDOW)
        sampling = time.perf_counter() - started
        assert result.returncode == 0, result.stderr.decode('utf8', errors='replace')[-1000:]
        assert len(result.stdout) == len(targets) * 360 * 202 * 3, (len(result.stdout), len(targets))
        frames = np.frombuffer(result.stdout, np.uint8).reshape(-1, 202, 360, 3)
        cap = cv2.VideoCapture(str(video))
        quality = []
        for i in np.linspace(0, len(targets) - 1, 15, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, targets[i])
            ok, reference = cap.read()
            assert ok
            reference = cv2.resize(reference, (360, 202), interpolation=cv2.INTER_AREA)
            error = float(np.mean((reference.astype(float) - frames[i]) ** 2))
            quality.append(10 * np.log10(255 ** 2 / max(error, 1e-8)))
        cap.release()
        del frames, result
        started = time.perf_counter()
        cuts = detect_video_scene_changes(str(video), threshold=.27, use_gpu=True, analysis_fps=6, analysis_width=384)
        reports.append({'source': video.name, 'targets': len(targets), 'sampling_seconds': sampling,
                        'gpu_cpu_psnr_db': quality, 'scene_seconds': time.perf_counter() - started, 'scene_count': len(cuts), 'scene_cuts': cuts})
        print(json.dumps(reports[-1]), flush=True)
    (OUT / 'sampling_probe.json').write_text(json.dumps(reports, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
