"""Uncached real-media analysis/render profiling for two supplied AMV sources."""

import argparse
import cProfile
import json
import os
from pathlib import Path
import pstats
import random
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--audio-seconds', type=float, default=60)
    parser.add_argument('--ai-windows', type=int, default=12)
    parser.add_argument('--reuse-analysis', action='store_true')
    args = parser.parse_args()
    out = ROOT / 'output' / 'gpu_priority' / args.label
    out.mkdir(parents=True, exist_ok=True)
    audio = next((ROOT / 'for amv').glob('*.flac'))
    videos = [ROOT / 'for amv' / (str(n) + '.mkv') for n in (1, 2)]
    os.environ['BEATSYNC_QWEN_MAX_WINDOWS'] = str(args.ai_windows)
    import auto_mode
    import video_analysis
    import video_processor
    import ffmpeg_processing as media
    import ffmpeg_runtime
    import gpu_cpu_utils
    from gpu_devices import get_discrete_gpu

    report = {'label': args.label, 'audio': str(audio), 'sources': [str(p) for p in videos],
              'audio_seconds': args.audio_seconds, 'qwen_windows_per_video': args.ai_windows,
              'gpu': get_discrete_gpu(), 'passed': False, 'stage_seconds': {}}
    profiler = cProfile.Profile()
    started = time.perf_counter()
    try:
        profiler.enable()
        original_stages = {name: getattr(auto_mode, name) for name in ('detect_master_beat_grid', 'analyze_wave_features', 'analyze_sections', 'select_wave_cuts')}
        def timed(name):
            def call(*values, **kwargs):
                tick = time.perf_counter()
                try:
                    return original_stages[name](*values, **kwargs)
                finally:
                    report['stage_seconds'][name] = time.perf_counter() - tick
            return call
        with patch.object(video_analysis, 'VIDEO_ANALYSIS_CACHE_DIR', str(out / 'analysis_cache')), \
             patch.object(video_analysis, '_load_cache', wraps=video_analysis._load_cache if args.reuse_analysis else lambda *a, **kw: None), \
             patch.object(auto_mode, 'detect_master_beat_grid', side_effect=timed('detect_master_beat_grid')), \
             patch.object(auto_mode, 'analyze_wave_features', side_effect=timed('analyze_wave_features')), \
             patch.object(auto_mode, 'analyze_sections', side_effect=timed('analyze_sections')), \
             patch.object(auto_mode, 'select_wave_cuts', side_effect=timed('select_wave_cuts')):
            tick = time.perf_counter()
            cuts, info = auto_mode.analyze_beats_auto(str(audio), end_time=args.audio_seconds,
                                                     use_gpu=True, video_files=[str(v) for v in videos])
            report['analysis_seconds'] = time.perf_counter() - tick
        visual = info.get('video_analysis') or {}
        assert visual.get('source_count') == 2 and visual.get('candidates'), visual
        assert all(v.get('candidates') for v in visual['videos']), visual['summary']
        report['video_analysis'] = {k: v for k, v in visual.items() if k not in ('candidates', 'highlight_pool', 'videos')}
        report['video_timings'] = [{'source': v['source_name'], 'timings': v.get('timings'),
                                  'candidate_count': len(v.get('candidates') or [])} for v in visual['videos']]
        report['audio_cuts'] = cuts.tolist()
        report['tempo'] = info['tempo']
        report['audio_separation_backend'] = info.get('audio_separation_backend')
        report['audio_separation_seconds'] = info.get('audio_separation_seconds')
        random.seed(9271)
        np.random.seed(9271)
        tick = time.perf_counter()
        output = out / 'amv_test.mp4'
        with patch.object(video_processor, 'get_processing_dir', return_value=str(out / 'processing')):
            video_processor.create_music_video(str(audio), [str(v) for v in videos], cuts,
                                               output_file=str(output), end_time=args.audio_seconds,
                                               beat_info=info, use_gpu=True, gpu_encoder='h264_nvenc', fps=24,
                                               auto_speed_clip_control=True, avoid_text_clips=True)
        report['render_seconds'] = time.perf_counter() - tick
        report['render_info'] = info.get('render_info')
        stream = ffmpeg_runtime.verify_video(output, round(args.audio_seconds * 24))
        assert stream['codec_name'] == 'h264'
        ffmpeg_runtime.checked_output([media.FFMPEG_PATH, '-v', 'error', '-xerror', '-i', str(output), '-map', '0', '-f', 'null', '-'], timeout=180)
        report['decoded_without_errors'] = True
        tick = time.perf_counter()
        preview = out / 'amv_preview.mp4'
        media.create_browser_preview(str(output), str(preview))
        report['preview_seconds'] = time.perf_counter() - tick
        assert [s['codec_name'] for s in ffmpeg_runtime.probe_media(preview)['streams']] == ['h264', 'aac']
        report['output'] = str(output)
        report['preview'] = str(preview)
        report['passed'] = True
    finally:
        profiler.disable()
        report['total_seconds'] = time.perf_counter() - started
        (out / 'report.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf8')
        profiler.dump_stats(str(out / 'profile.pstats'))
        with (out / 'profile.txt').open('w', encoding='utf8') as file:
            pstats.Stats(profiler, stream=file).sort_stats('cumulative').print_stats(65)
    print('Pipeline report:', out / 'report.json', flush=True)
    print('Total seconds:', round(report['total_seconds'], 2), flush=True)


if __name__ == '__main__':
    main()
