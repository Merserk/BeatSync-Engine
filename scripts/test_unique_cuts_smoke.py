"""Reproduce and repair the user's repeated-shot AMV with its actual inputs."""
import json
from pathlib import Path
import random
import time
from unittest.mock import patch

import numpy as np
import auto_mode
from auto_mode.stage6_av_planner import audit_clip_plan_repetition
from clip_uniqueness import ClipUniquenessIndex, frame_fingerprints, validate_unique_clips
import ffmpeg_processing as media
import ffmpeg_runtime as runtime
from gpu_video import iter_sampled_frames
import video_analysis
import video_processor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'unique_cuts'


def rendered_visual_audit(video, clips, fps):
    targets = {}
    for index, clip in enumerate(clips):
        first = round(clip['audio_start'] * fps)
        count = round(clip['final_duration'] * fps)
        for fraction in (.15, .5, .85):
            frame = first + min(count - 1, max(0, round((count - 1) * fraction)))
            targets[frame] = index
    descriptors = [[] for _ in clips]
    for frame, image in iter_sampled_frames(str(video), targets, max_width=384):
        descriptors[targets[frame]].extend(frame_fingerprints([image]))
    seen = ClipUniquenessIndex()
    conflicts = []
    for index, fingerprints in enumerate(descriptors):
        item = {'index': index, 'visual_fingerprints': fingerprints}
        for reason, previous in seen.conflicts(item).items():
            conflicts.append({'previous': previous, 'index': index, 'reason': reason})
        seen.add(item)
    return {'sampled_frames': len(targets), 'conflicts': conflicts}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original = ROOT / 'output' / 'music_video_20261002_011441.mp4'
    original_plan = json.loads(original.with_suffix('.clip_plan_debug.json').read_text(encoding='utf8'))
    old_clips = original_plan['clips']
    sources = list(dict.fromkeys(clip['video_file'] for clip in old_clips))
    assert all(Path(path).is_file() for path in sources)
    audio = next((ROOT / 'for amv').glob('*.flac'))
    output = OUT / 'music_video_unique.mp4'
    preview = OUT / 'music_video_unique_preview.mp4'
    fps = 24
    report = {'original': str(original), 'sources': sources, 'audio': str(audio), 'passed': False,
              'original_repetition': audit_clip_plan_repetition(old_clips)}
    assert report['original_repetition']['source_overlap_count'] == 10
    tick = time.perf_counter()
    try:
        report['original_visual_audit'] = rendered_visual_audit(original, old_clips, fps)
        old_pair = []
        for _, image in iter_sampled_frames(str(original), [51 * fps, 124 * fps], max_width=384):
            old_pair.append(frame_fingerprints([image]))
        report['reported_frames_match'] = bool(ClipUniquenessIndex([
            {'index': 22, 'visual_fingerprints': old_pair[0]}
        ]).conflicts({'index': 60, 'visual_fingerprints': old_pair[1]}))
        with patch.object(video_analysis, 'VIDEO_ANALYSIS_CACHE_DIR', str(OUT / 'analysis_cache')):
            _, info = auto_mode.analyze_beats_auto(str(audio), use_gpu=True, video_files=sources)
        report['analysis'] = {k: v for k, v in info['video_analysis'].items()
                              if k not in ('candidates', 'highlight_pool', 'videos')}
        report['source_analysis'] = [
            {'source': data['source_name'], 'timings': data['timings'],
             'fingerprinted_candidates': sum(bool(c.get('visual_fingerprints')) for c in data['candidates'])}
            for data in info['video_analysis']['videos']]
        # Preserve every musical cut boundary from the user's export.
        beats = np.asarray([clip['audio_start'] for clip in old_clips] + [old_clips[-1]['audio_end']])
        random.seed(9271)
        render_tick = time.perf_counter()
        with patch.object(video_processor, 'get_processing_dir', return_value=str(OUT / 'processing')):
            video_processor.create_music_video(str(audio), sources, beats, str(output), fps=fps,
                                               beat_info=info, use_gpu=True, gpu_encoder='h264_nvenc',
                                               auto_speed_clip_control=False, avoid_text_clips=True)
        report['render_seconds'] = time.perf_counter() - render_tick
        new_plan = json.loads(output.with_suffix('.clip_plan_debug.json').read_text(encoding='utf8'))
        new_clips = new_plan['clips']
        validate_unique_clips(new_clips)
        report['plan_summary'] = new_plan['plan_summary']
        assert [(c['audio_start'], c['audio_end']) for c in new_clips] == [
            (c['audio_start'], c['audio_end']) for c in old_clips]
        runtime.verify_video(output, 5434)
        runtime.checked_output([media.FFMPEG_PATH, '-v', 'error', '-xerror', '-i', str(output),
                                '-map', '0', '-f', 'null', '-'], timeout=180)
        report['new_visual_audit'] = rendered_visual_audit(output, new_clips, fps)
        assert not report['new_visual_audit']['conflicts'], report['new_visual_audit']
        media.create_browser_preview(str(output), str(preview))
        runtime.verify_video(preview, 5434)
        report['output'] = str(output)
        report['preview'] = str(preview)
        report['frames'] = 5434
        report['passed'] = True
    finally:
        report['total_seconds'] = time.perf_counter() - tick
        (OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('source_analysis', 'analysis')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
