"""Verify reported source-shot leaks and rebuild the actual 100-cut AMV."""
import argparse
import json
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

import numpy as np
import cv2

import auto_mode
import ffmpeg_processing as media
import ffmpeg_runtime as runtime
import video_analysis
import video_processor
from clip_uniqueness import ClipUniquenessIndex, frame_fingerprints, validate_unique_clips
from gpu_video import iter_sampled_frames, iter_interval_frames
from shot_integrity import inspect_source_clip, native_shot_boundaries
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_rendered_duplicates import audit
from diagnose_cut_flashes import sheet, source_path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'clean_cuts'
ORIGINAL = ROOT / 'output' / 'music_video_20261002_034313.mp4'


def full_shot_audit(path, clips):
    targets = range(round(clips[-1]['audio_end'] * 24))
    first = 0
    images = []
    found = []
    frame_count = 0
    for frame, image in iter_sampled_frames(str(path), targets, max_width=384):
        if frame >= round(clips[first]['audio_end'] * 24):
            for point in native_shot_boundaries(images):
                found.append({'clip': first, 'output_frame': round(clips[first]['audio_start'] * 24) + point})
            first += 1
            images = []
        images.append(image)
        frame_count += 1
    for point in native_shot_boundaries(images):
        found.append({'clip': first, 'output_frame': round(clips[first]['audio_start'] * 24) + point})
    return {'frames': frame_count, 'internal_source_shot_changes': found}


def check_output_transitions_against_source(path, clips, result):
    """Resolve detector flags with the actual camera motion around the source cut.

    An isolated final motion-blurred frame can look like a cut when the output
    comparison lacks the subsequent source frames. It must match the inspected
    source and have continuous surrounding motion before being dismissed.
    """
    confirmed = []
    continuous = []
    for flag in result['internal_source_shot_changes']:
        clip = clips[flag['clip']]
        start, duration = clip['start_time'], clip['source_duration']
        source = list(iter_interval_frames(clip['video_file'], max(0., start - .35), duration + .7,
                                           max_width=384, with_timestamps=True))
        points = native_shot_boundaries([im for _, im in source])
        if any(start <= source[n][0] < start + duration for n in points):
            confirmed.append(flag)
            continue
        rendered = next(iter_sampled_frames(str(path), [flag['output_frame']], max_width=384))[1]
        expected = start + (flag['output_frame'] / 24 - clip['audio_start']) * clip['playback_speed']
        def small(image):
            return cv2.resize(image, (128, 72), interpolation=cv2.INTER_AREA).astype(np.float32)
        candidates = [(t, float(np.abs(small(im) - small(rendered)).mean())) for t, im in source
                      if abs(t - expected) < .15]
        timestamp, difference = min(candidates, key=lambda match: match[1])
        assert difference < 6, ('Output transition does not match native footage', flag, difference)
        continuous.append(dict(flag, source_timestamp=timestamp, mean_pixel_difference=difference,
                               reason='continuous fast motion; matches native source, with no contextual shot change'))
    return dict(result, internal_source_shot_changes=confirmed, continuous_motion_flags=continuous,
                raw_transition_count=len(result['internal_source_shot_changes']))


def validate_full_export(old, report):
    output = OUT / 'music_video_clean.mp4'
    new = json.loads(output.with_suffix('.clip_plan_debug.json').read_text(encoding='utf8'))
    clips = new['clips']
    report['plan_summary'] = new['plan_summary']
    validate_unique_clips(clips)
    assert [(c['audio_start'], c['audio_end']) for c in clips] == [(c['audio_start'], c['audio_end']) for c in old]
    runtime.verify_video(output, 5434)
    runtime.checked_output([media.FFMPEG_PATH, '-v', 'error', '-xerror', '-i', str(output), '-map', '0', '-f', 'null', '-'], timeout=180)
    report['new_shot_audit'] = check_output_transitions_against_source(output, clips, full_shot_audit(output, clips))
    assert not report['new_shot_audit']['internal_source_shot_changes'], report['new_shot_audit']
    report['new_duplicate_audit'] = audit(output)
    assert not report['new_duplicate_audit']['matching_frame_count'], report['new_duplicate_audit']
    preview = OUT / 'music_video_clean_preview.mp4'
    media.create_browser_preview(str(output), str(preview))
    runtime.verify_video(preview, 5434)
    report.update(output=str(output), preview=str(preview), frames=5434)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--focused', action='store_true')
    parser.add_argument('--validate-existing', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    old = json.loads(ORIGINAL.with_suffix('.clip_plan_debug.json').read_text(encoding='utf8'))['clips']
    report = {'focused': [], 'passed': False}
    if args.validate_existing:
        report = json.loads((OUT / 'report.json').read_text(encoding='utf8'))
        tick = time.perf_counter()
        validate_full_export(old, report)
        report.update(passed=True, post_export_validation_seconds=time.perf_counter() - tick)
        (OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
        print(json.dumps(report, indent=2), flush=True)
        return
    tick = time.perf_counter()
    try:
        bounds = {41: (662.5, 665), 47: (2128.833, 2131.667), 68: (1679.5, 1682)}
        for reported, index in [(86, 41), (96, 47), (139, 68)]:
            clip = dict(old[index], video_file=str(source_path(old[index])),
                        usable_source_start=bounds[index][0], usable_source_end=bounds[index][1])
            refined = inspect_source_clip(clip)
            assert refined['shot_verified'], refined
            output = OUT / f'fixed_cut_{reported}.mp4'
            assert media.extract_clip_segment_ffmpeg(refined['video_file'], refined['start_time'],
                                                       refined['source_duration'], str(output), 24, (1920, 1080),
                                                       True, 'h264_nvenc', refined['final_duration'])
            count = round(refined['final_duration'] * 24)
            runtime.verify_video(output, count)
            frames = list(iter_interval_frames(str(output), 0, refined['final_duration'], max_width=384))
            assert len(frames) == count
            assert not native_shot_boundaries(frames), (index, native_shot_boundaries(frames))
            stray_frame = next(iter_sampled_frames(str(ORIGINAL), [round(old[index]['audio_end'] * 24) - 1], max_width=384))[1]
            assert not ClipUniquenessIndex([{'visual_fingerprints': frame_fingerprints(frames)}]).conflicts(
                {'visual_fingerprints': frame_fingerprints([stray_frame])}), index
            sheet(f'fixed_{reported}', frames[-12:], [f'fixed cut {index + 1}, frame {count - 12 + i}' for i in range(12)], 6)
            report['focused'].append({k: refined[k] for k in ('index', 'start_time', 'source_start_shift',
                                                               'verified_shot_end', 'native_source_frames', 'shot_boundary_count')})
            print(f'Focused cut {index + 1}: {count} frames, no internal cut or reported stray frame', flush=True)
        if not args.focused:
            report['original_shot_audit'] = full_shot_audit(ORIGINAL, old)
            sources = [str(source_path(next(c for c in old if Path(c['video_file']).name == name))) for name in ('5.mkv', '6.mkv')]
            audio = next((ROOT / 'for amv').glob('*.flac'))
            with patch.object(video_analysis, 'VIDEO_ANALYSIS_CACHE_DIR', str(OUT / 'analysis_cache')):
                _, info = auto_mode.analyze_beats_auto(str(audio), use_gpu=True, video_files=sources)
            output = OUT / 'music_video_clean.mp4'
            beats = np.asarray([c['audio_start'] for c in old] + [old[-1]['audio_end']])
            random.seed(9271)
            render_started = time.perf_counter()
            with patch.object(video_processor, 'get_processing_dir', return_value=str(OUT / 'processing')):
                video_processor.create_music_video(str(audio), sources, beats, str(output), beat_info=info,
                                                   fps=24, use_gpu=True, gpu_encoder='h264_nvenc',
                                                   auto_speed_clip_control=True, avoid_text_clips=True)
            report['render_seconds'] = time.perf_counter() - render_started
            validate_full_export(old, report)
        report['passed'] = True
    finally:
        report['total_seconds'] = time.perf_counter() - tick
        (OUT / ('focused_report.json' if args.focused else 'report.json')).write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
