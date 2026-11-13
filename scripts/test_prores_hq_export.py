"""Short real-footage HQ exports, including the previously reported shot endpoints."""
from contextlib import redirect_stdout
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np

import ffmpeg_processing as media
import ffmpeg_runtime as runtime
import video_processor as renderer
from gpu_video import iter_interval_frames
from shot_integrity import native_shot_boundaries
from video_encoders import get_prores_gpu_support

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'prores_hq_update' / 'smoke'
COUNTS = [43, 61, 58]
FPS = 24


def check(file, counts, codec='prores', audio=True):
    stream = runtime.verify_video(file, sum(counts))
    assert stream['codec_name'] == codec, stream
    assert Fraction(stream['avg_frame_rate']) == FPS, stream
    if codec == 'prores':
        assert stream['profile'] == 'HQ', stream
        assert stream['pix_fmt'] == 'yuv422p10le', stream
        assert stream['codec_tag_string'] == 'apch', stream
    sound = next((s for s in runtime.probe_media(file)['streams'] if s['codec_type'] == 'audio'), None)
    if audio:
        assert sound['codec_name'] == 'pcm_s24le' and sound['sample_rate'] == '48000', sound
        assert abs(float(sound['duration']) - sum(counts) / FPS) < 1 / FPS, sound
    runtime.checked_output([media.FFMPEG_PATH, '-nostdin', '-v', 'error', '-xerror', '-i', str(file),
                            '-map', '0', '-f', 'null', '-'], timeout=120)
    images = list(iter_interval_frames(str(file), 0, sum(counts) / FPS, max_width=384))
    assert len(images) == sum(counts), len(images)
    cursor = 0
    for count in counts:
        assert not native_shot_boundaries(images[cursor:cursor + count]), 'Unexpected shot inside a clip'
        cursor += count
    return {'output': str(file), 'codec': codec, 'profile': stream.get('profile'),
            'pixel_format': stream['pix_fmt'], 'frames': sum(counts), 'fps': FPS,
            'audio': sound['codec_name'] if sound else None, 'full_decode': 'passed',
            'within_clip_shot_check': 'passed'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_dir = ROOT / 'for amv'
    audio = next(source_dir.glob('*.flac'))
    sources = [str(source_dir / '5.mkv'), str(source_dir / '6.mkv')]
    cuts = np.asarray([0, *np.cumsum(COUNTS) / FPS])
    plan = []
    for index, (source, start, count) in enumerate([
        (sources[0], 663.1643333333334, 43), (sources[0], 2129.1223333333337, 61),
        (sources[1], 1679.5393333333334, 58),
    ]):
        plan.append({'index': index, 'video_file': source, 'source_name': Path(source).name,
                     'start_time': start, 'source_duration': count / FPS, 'final_duration': count / FPS,
                     'playback_speed': 1, 'audio_start': cuts[index], 'audio_end': cuts[index + 1],
                     'candidate_id': f'hq_case_{index}', 'selection_level': 'regression_fixture'})
    report = {'passed': False, 'gpu_support': get_prores_gpu_support(), 'exports': [],
              'test_note': 'Reuses three known source intervals; native shot and uniqueness checks run normally.'}
    assert report['gpu_support']['available'], report['gpu_support']
    with (OUT / 'render.log').open('w', encoding='utf8') as log, redirect_stdout(log):
        plans = []
        for label, gpu, export_format in [('hq_cpu', False, 'prores_hq'), ('hq_vulkan', True, 'prores_hq'),
                                           ('h264_regression', True, 'auto')]:
            file = OUT / (label + ('.mov' if export_format == 'prores_hq' else '.mp4'))
            info = {'mode': 'auto'}
            tick = time.perf_counter()
            with patch.object(renderer, 'build_planned_clip_sequence', side_effect=lambda **kwargs: deepcopy(plan)), \
                 patch.object(renderer, 'get_processing_dir', return_value=str(OUT / 'processing')):
                renderer.create_music_video(str(audio), sources, cuts, str(file), end_time=cuts[-1], fps=FPS,
                                            max_workers=2, beat_info=info, export_format=export_format, use_gpu=gpu,
                                            gpu_encoder='h264_nvenc' if export_format == 'auto' else None)
            result = check(file, COUNTS, 'h264' if export_format == 'auto' else 'prores')
            result.update(seconds=time.perf_counter() - tick, render_info=info['render_info'])
            report['exports'].append(result)
            data = json.loads(file.with_suffix('.clip_plan_debug.json').read_text(encoding='utf8'))
            assert data['plan_summary']['clean_shots']
            assert data['plan_summary']['visual_duplicate_count'] == 0
            plans.append([(c['video_file'], c['start_time'], c['source_duration'], c['final_duration']) for c in data['clips']])
        assert plans[0] == plans[1] == plans[2], 'Export format changed the verified edit plan'
        report['same_verified_plan_all_formats'] = True
        # Prove CPU retry and full-encode assembly cannot fall back to Proxy.
        real_run = media._run_media_command
        def fail_gpu_once(command, timeout):
            if 'prores_ks_vulkan' in command:
                return subprocess.CompletedProcess(command, 1, '', 'Injected Vulkan failure for retry verification')
            return real_run(command, timeout)
        with patch.object(media, '_run_media_command', side_effect=fail_gpu_once):
            segment = media.extract_prores_segment_random(sources[0], COUNTS[0] / FPS, FPS, str(OUT), 999,
                                                          start_time=plan[0]['start_time'], target_size=(512, 288),
                                                          use_gpu=True, output_duration=2)
        result = check(segment, [48], audio=False)
        report['exports'].append(dict(result, test='Vulkan failure -> CPU HQ retry; retimed 43 source frames -> 48 output frames'))
        fallback = OUT / 'hq_reencode_assembly.mov'
        with patch.dict('os.environ', {'BEATSYNC_FAST_CONCAT_COPY': '0'}):
            media.concatenate_videos_ffmpeg([segment], str(fallback), str(audio), fps=FPS,
                                            gpu_encoder='prores_ks', temp_dir=str(OUT), output_duration=2,
                                            clip_frame_counts=[48])
        report['exports'].append(dict(check(fallback, [48]), test='Full assembly re-encode retains HQ'))
        media.create_browser_preview(str(OUT / 'hq_vulkan.mov'), str(OUT / 'hq_preview.mp4'))
        runtime.verify_video(OUT / 'hq_preview.mp4', sum(COUNTS))
        runtime.checked_output([media.FFMPEG_PATH, '-v', 'error', '-xerror', '-i', str(OUT / 'hq_preview.mp4'),
                                '-f', 'null', '-'], timeout=120)
        report['browser_preview'] = 'passed; original HQ MOV retained'
    report['passed'] = True
    (OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps({'passed': True, 'exports': len(report['exports']), 'frames_per_full_export': sum(COUNTS),
                      'gpu': report['gpu_support']['name'], 'report': str(OUT / 'report.json')}, indent=2))


if __name__ == '__main__':
    main()
