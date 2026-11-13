"""Trace reported flash frames to planned cuts and the original footage."""
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gpu_video import iter_sampled_frames, iter_interval_frames
from clip_uniqueness import frame_fingerprints

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'clean_cuts'
ORIGINAL = ROOT / 'output' / 'music_video_20261002_034313.mp4'


def source_path(clip):
    original = Path(clip['video_file'])
    return original if original.exists() else ROOT / 'for amv' / original.name


def sheet(name, frames, labels, columns=8):
    width, height = 320, 208
    canvas = Image.new('RGB', (columns * width, ((len(frames) + columns - 1) // columns) * height), '#151820')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 16)
    for i, (frame, label) in enumerate(zip(frames, labels)):
        x, y = (i % columns) * width, (i // columns) * height
        frame = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        canvas.paste(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), (x, y + 25))
        draw.text((x + 5, y + 3), label, fill='white', font=font)
    path = OUT / f'{name}.jpg'
    canvas.save(path, quality=93)
    return str(path)


def score(a, b):
    a = cv2.cvtColor(cv2.resize(a, (128, 72)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(cv2.resize(b, (128, 72)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.abs(a - b).mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = json.loads(ORIGINAL.with_suffix('.clip_plan_debug.json').read_text(encoding='utf8'))
    clips = plan['clips']
    fps = 24
    observations = []
    for reported, index in [(86, 41), (96, 47), (139, 68)]:
        clip = clips[index]
        last = round(clip['audio_end'] * fps)
        targets = range(last - 12, last + 6)
        output = list(iter_sampled_frames(str(ORIGINAL), targets, max_width=768))
        picture = sheet(f'output_{reported}', [image for _, image in output],
                        [f'{frame / fps:.4f}s #{frame} c{index if frame < last else index + 1}' for frame, _ in output], 6)
        start = clip['start_time'] + clip['source_duration'] - .75
        source = list(iter_interval_frames(str(source_path(clip)), start, 1.25, max_width=768))
        source_picture = sheet(f'source_{reported}', source,
                               [f'{start + n / fps:.4f}s ({n})' for n in range(len(source))], 6)
        transitions = [{'frame': n, 'approx_source_seconds': start + n / fps,
                        'difference': score(source[n-1], source[n])} for n in range(1, len(source))]
        observations.append({'reported': reported, 'clip_index': index, 'source': str(source_path(clip)),
                             'source_start': clip['start_time'], 'source_end': clip['start_time'] + clip['source_duration'],
                             'output_start': clip['audio_start'], 'output_end': clip['audio_end'],
                             'output_sheet': picture, 'source_sheet': source_picture,
                             'transitions': sorted(transitions, key=lambda item: item['difference'], reverse=True)[:5]})
        print(json.dumps(observations[-1], indent=2), flush=True)
    (OUT / 'diagnosis.json').write_text(json.dumps(observations, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
