"""Real encoder tests of native shot endpoints and independent source/output FPS."""
import json
from pathlib import Path

import cv2
import numpy as np

import ffmpeg_processing as media
import ffmpeg_runtime as runtime
from clip_uniqueness import ClipUniquenessIndex, frame_fingerprints
from gpu_video import iter_interval_frames
from shot_integrity import inspect_source_clip, native_shot_boundaries
from video_encoders import get_encoder_support

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'clean_cuts' / 'endpoint_exports'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(8731)
    a = cv2.resize(rng.integers(16, 120, (8, 12, 3), dtype=np.uint8), (512, 288), interpolation=cv2.INTER_NEAREST)
    b = cv2.resize(rng.integers(135, 240, (8, 12, 3), dtype=np.uint8), (512, 288), interpolation=cv2.INTER_NEAREST)
    raw = OUT / 'two_shots.bgr24'
    with raw.open('wb') as file:
        for i in range(60):
            file.write((a if i < 47 else b).tobytes())
    source = OUT / 'millisecond_pts.mkv'
    runtime.checked_output([media.FFMPEG_PATH, '-v', 'error', '-y', '-f', 'rawvideo', '-pixel_format', 'bgr24',
                            '-video_size', '512x288', '-framerate', '24', '-i', str(raw), '-c:v', 'libx264',
                            '-pix_fmt', 'yuv420p', str(source)], timeout=120)
    source_clip = {'video_file': str(source), 'start_time': .2, 'source_duration': 1.8, 'final_duration': 1.8,
                   'usable_source_start': 0., 'usable_source_end': 2.0}
    refined = inspect_source_clip(source_clip)
    assert refined['shot_verified'] and refined['shot_refined'], refined
    report = {'passed': False, 'exports': [], 'refined_start': refined['start_time']}
    bad = ClipUniquenessIndex([{'visual_fingerprints': frame_fingerprints([b])}])
    for encoder in ('none', 'h264_nvenc', 'hevc_nvenc', 'h264_vulkan', 'hevc_vulkan', 'prores_cpu', 'prores_vulkan'):
        if encoder.startswith('prores'):
            output = media.extract_prores_segment_random(str(source), 1.8, 30, str(OUT),
                                                          1 if encoder == 'prores_cpu' else 2,
                                                          start_time=refined['start_time'], output_duration=1.8,
                                                          target_size=(512, 288), use_gpu=encoder == 'prores_vulkan')
        else:
            if encoder != 'none':
                assert get_encoder_support(encoder)['available'], encoder
            output = str(OUT / (encoder + '.mp4'))
            assert media.extract_clip_segment_ffmpeg(str(source), refined['start_time'], 1.8, output,
                                                       30, (512, 288), encoder != 'none', encoder, 1.8)
        runtime.verify_video(output, 54)
        frames = list(iter_interval_frames(output, 0., 1.8, max_width=384))
        assert len(frames) == 54 and not native_shot_boundaries(frames), encoder
        assert not bad.conflicts({'visual_fingerprints': frame_fingerprints(frames)}), encoder
        runtime.checked_output([media.FFMPEG_PATH, '-v', 'error', '-xerror', '-i', output, '-f', 'null', '-'], timeout=120)
        report['exports'].append({'encoder': encoder, 'frames': 54, 'fps': 30, 'source_fps': 24,
                                  'next_shot_frames': 0, 'output': output})
        print(f'{encoder}: 54 frames, 24 to 30 fps, no next-shot frames', flush=True)
    # A source duration shorter than an output frame must not be expanded into
    # the next source shot. Exercise this with a real high-frame-rate source.
    short_raw = OUT / 'short_source.bgr24'
    short_raw.write_bytes(a.tobytes() * 3 + b.tobytes() * 9)
    short_source = OUT / 'high_fps.mkv'
    runtime.checked_output([media.FFMPEG_PATH, '-v', 'error', '-y', '-f', 'rawvideo', '-pixel_format', 'bgr24',
                            '-video_size', '512x288', '-framerate', '120', '-i', str(short_raw),
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(short_source)], timeout=120)
    output = str(OUT / 'short_exact_duration.mp4')
    assert media.extract_clip_segment_ffmpeg(str(short_source), 0., .020, output, 24, (512, 288), True,
                                               'h264_nvenc', 1 / 24)
    runtime.verify_video(output, 1)
    frames = list(iter_interval_frames(output, 0., 1 / 24, max_width=384))
    assert len(frames) == 1 and not bad.conflicts({'visual_fingerprints': frame_fingerprints(frames)})
    report['source_duration_rounding'] = {'source_seconds': .020, 'source_fps': 120, 'output_fps': 24,
                                         'frames': 1, 'next_shot_frames': 0}
    report['passed'] = True
    (OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
