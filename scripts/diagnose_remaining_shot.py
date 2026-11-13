"""Inspect a native shot boundary found by the independent export audit."""
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from gpu_video import iter_interval_frames
from shot_integrity import native_shot_boundaries, inspect_source_clip, _correlation
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_cut_flashes import sheet

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'clean_cuts'
clip = json.loads((OUT / 'music_video_clean.clip_plan_debug.json').read_text(encoding='utf8'))['clips'][40]
start = clip['start_time']
native = list(iter_interval_frames(clip['video_file'], start - .35, clip['source_duration'] + .7,
                                  with_timestamps=True))
for width, height in ((768, 432), (384, 216), (256, 144)):
    images = [cv2.resize(im, (width, height), interpolation=cv2.INTER_AREA) for _, im in native]
    selected = [im for (t, _), im in zip(native, images) if start <= t < start + clip['source_duration']]
    print(width, 'context cuts', native_shot_boundaries(images), 'selected cuts', native_shot_boundaries(selected))
    gray = [cv2.cvtColor(cv2.resize(im, (128, 72), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
            for im in images]
    ref = np.median(np.stack(gray[:5]), axis=0)
    active = np.flatnonzero((ref.mean(axis=1) > 12) | (ref.std(axis=1) > 2))
    gray = [im[active[0]:active[-1] + 1] for im in gray]
    diffs = [float(np.abs(gray[i] - gray[i - 1]).mean()) for i in range(1, len(gray))]
    for i in range(len(gray) - 18, len(gray)):
        baseline = np.median(diffs[max(0, i-5):i-1] + diffs[i:min(len(diffs), i+4)])
        print('metrics', width, i, native[i][0], 'diff', round(diffs[i-1], 2), 'corr', round(_correlation(gray[i-1], gray[i]), 3),
              'baseline', round(float(baseline), 2))
sheet('remaining_source', [im for _, im in native[-24:]], [f'{t:.3f}s' for t, _ in native[-24:]], 6)
rendered = list(iter_interval_frames(str(OUT / 'music_video_clean.mp4'), clip['audio_start'], clip['final_duration'],
                                    with_timestamps=True))
sheet('remaining_output', [im for _, im in rendered[-18:]], [f'{t:.3f}s' for t, _ in rendered[-18:]], 6)
result = inspect_source_clip(clip)
print('inspection', {k: v for k, v in result.items() if k.startswith(('verified_shot', 'shot_', 'native_source', 'source_start'))})
