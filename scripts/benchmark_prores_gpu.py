"""CPU/Vulkan ProRes HQ performance and real-source export validation."""
import json
from pathlib import Path
import random
import time

import numpy as np
import ffmpeg_processing as media
import ffmpeg_runtime as runtime
import video_processor
from video_encoders import get_prores_gpu_support

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'prores_hq_update' / 'benchmark'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audio = next((ROOT / 'for amv').glob('*.flac'))
    videos = [str(ROOT / 'for amv' / f'{i}.mkv') for i in (1, 2)]
    report = {'support': get_prores_gpu_support(), 'segments': [], 'prores_hq': []}
    assert report['support']['available'], report['support']
    for gpu in (False, True):
        tick = time.perf_counter()
        segment = media.extract_prores_segment_random(videos[0], 10, 24, str(OUT), 1 if gpu else 0,
                                                       start_time=120, target_size=(1920, 1080), use_gpu=gpu)
        seconds = time.perf_counter() - tick
        stream = runtime.verify_video(segment, 240)
        assert stream['codec_name'] == 'prores' and stream['profile'] == 'HQ'
        report['segments'].append({'gpu': gpu, 'seconds': seconds, 'frames': 240, 'output': segment})
    for gpu in (False, True):
        random.seed(9271)
        info = {}
        output = OUT / ('prores_hq_gpu.mov' if gpu else 'prores_hq_cpu.mov')
        tick = time.perf_counter()
        video_processor.create_music_video(str(audio), videos, np.asarray([0, 1.04, 2.37, 4.02, 5.21]),
                                           str(output), end_time=6, fps=24, export_format='prores_hq',
                                           use_gpu=gpu, beat_info=info)
        seconds = time.perf_counter() - tick
        stream = runtime.verify_video(output, 144)
        assert stream['codec_name'] == 'prores' and stream['profile'] == 'HQ'
        assert stream['pix_fmt'] == 'yuv422p10le'
        assert runtime.probe_media(output)['streams'][1]['codec_name'] == 'pcm_s24le'
        runtime.checked_output([media.FFMPEG_PATH, '-v', 'error', '-xerror', '-i', str(output), '-f', 'null', '-'], timeout=120)
        report['prores_hq'].append({'gpu': gpu, 'seconds': seconds, 'frames': 144, 'output': str(output)})
    (OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
