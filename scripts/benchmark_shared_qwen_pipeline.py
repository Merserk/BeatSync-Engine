"""Compare the old interleaved and new staged pipeline on ten real inputs."""
from contextlib import redirect_stdout
import importlib.util
import json
from pathlib import Path
import random
import time
from unittest.mock import patch

import numpy as np

import ffmpeg_processing as media
import ffmpeg_runtime as runtime
from gpu_devices import get_discrete_gpu
from gpu_video import get_gpu_video_path, iter_interval_frames
from shot_integrity import native_shot_boundaries
import video_analysis
import video_processor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'pipeline_batch'


def fixtures():
    folder = OUT / 'fixtures'
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(10):
        source = ROOT / 'for amv' / f'{index % 9 + 1}.mkv'
        output = folder / f'input_{index + 1:02d}.mkv'
        if not output.exists():
            runtime.checked_output([media.FFMPEG_PATH, '-v', 'error', '-y', '-ss', str(300 + index * 47),
                                    '-t', '12', '-i', str(source), '-map', '0:v:0', '-c:v', 'copy',
                                    '-an', '-sn', '-dn', '-avoid_negative_ts', 'make_zero', str(output)], timeout=120)
        runtime.verify_video(output)
        paths.append(str(output))
    return paths


def run_analysis(module, label, paths):
    cache = OUT / label / 'analysis_cache'
    cache.mkdir(parents=True, exist_ok=True)
    existing_responses = set(cache.glob('qwen*response*.json'))
    existing_requests = set(cache.glob('qwen*request*.json'))
    events = []
    analyze = module._analyze_single_video
    single = module._run_qwen_worker
    batch = module._run_qwen_worker_batch

    def scan(*args, **kwargs):
        events.append({'stage': 'scan', 'source': Path(args[0]).name})
        return analyze(*args, **kwargs)

    def qwen_single(*args, **kwargs):
        events.append({'stage': 'qwen', 'sources': [Path(kwargs['video_file']).name]})
        return single(*args, **kwargs)

    def qwen_batch(*args, **kwargs):
        events.append({'stage': 'qwen', 'sources': [Path(j['video_file']).name for j in kwargs['jobs']]})
        return batch(*args, **kwargs)

    started = time.perf_counter()
    with patch.object(module, 'VIDEO_ANALYSIS_CACHE_DIR', str(cache)), \
         patch.object(module, '_analyze_single_video', side_effect=scan), \
         patch.object(module, '_run_qwen_worker', side_effect=qwen_single), \
         patch.object(module, '_run_qwen_worker_batch', side_effect=qwen_batch), \
         patch.object(module, '_load_cache', return_value=None), \
         (OUT / f'{label}.log').open('w', encoding='utf8') as log, redirect_stdout(log):
        result = module.analyze_video_sources(paths, use_gpu=True, enable_ai=True)
    elapsed = time.perf_counter() - started
    response_files = [p for p in cache.glob('qwen*response*.json') if p not in existing_responses]
    responses = [json.loads(p.read_text(encoding='utf8')) for p in response_files]
    responses = [r for r in responses if 'semantics' in r or 'semantics_by_job' in r]
    requests = [json.loads(p.read_text(encoding='utf8')) for p in cache.glob('qwen*request*.json') if p not in existing_requests]
    requested = {}
    for request in requests:
        for job in request.get('jobs') or [request]:
            requested[job['video_file']] = [{'id': c['id'], 'start': c['start'], 'end': c['end']}
                                           for c in job['candidates']]
    requested_count = sum(len(v) for v in requested.values())
    server_pids = {}
    for path in response_files:
        response = json.loads(path.read_text(encoding='utf8'))
        request = json.loads(path.with_name(path.name.replace('response', 'request')).read_text(encoding='utf8'))
        for job in request.get('jobs') or [dict(request, job_id='single')]:
            server_pids[Path(job['video_file']).name] = response['timings_by_job'][job['job_id']]['server_pid']
    assert result['source_count'] == 10 and len(requested) == 10
    assert result['qwen_tag_count'] == requested_count, (label, result['qwen_tag_count'], requested_count)
    assert all(c.get('ai_analyzed') for v in result['videos'] for c in v['candidates']
               if c['id'] in {x['id'] for x in requested[v['video_file']]})
    stats = {
        'seconds': elapsed, 'events': events, 'worker_launches': len(responses),
        'server_starts': sum(r['server_start_count'] for r in responses),
        'model_load_seconds': sum(r['model_load_seconds'] for r in responses),
        'semantic_tags': result['qwen_tag_count'], 'requested_frames': requested_count,
        'requested': requested,
        'server_pids_by_source': server_pids,
        'source_stats': [{'name': v['source_name'], 'candidate_count': v['candidate_count'],
                          'timings': v['timings']} for v in result['videos']],
    }
    (OUT / label / 'stats.json').write_text(json.dumps(stats, indent=2), encoding='utf8')
    return result, stats


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {'passed': False, 'gpu': get_discrete_gpu(), 'sample_resolution': 768,
              'qwen_window_budget': video_analysis._qwen_max_windows()}
    started = time.perf_counter()
    try:
        paths = fixtures()
        # Warm capabilities and numerical kernels before either timed run.
        for path in paths:
            get_gpu_video_path(path)
        with (OUT / 'warmup.log').open('w', encoding='utf8') as log, redirect_stdout(log):
            video_analysis._analyze_single_video(paths[0], True, False, '', {}, True)
        spec = importlib.util.spec_from_file_location('video_analysis_before_batch', OUT / 'before' / 'video_analysis.py')
        before = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(before)
        old, report['before'] = run_analysis(before, 'before_run', paths)
        print('Old pipeline completed:', round(report['before']['seconds'], 2), 'seconds', flush=True)
        new, report['after'] = run_analysis(video_analysis, 'after_run', paths)
        print('Shared pipeline completed:', round(report['after']['seconds'], 2), 'seconds', flush=True)
        assert report['before']['requested'] == report['after']['requested'], 'Semantic sample choices changed'
        old_by_id = {c['id']: c for c in old['candidates']}
        new_by_id = {c['id']: c for c in new['candidates']}
        assert old_by_id.keys() == new_by_id.keys()
        preserved = ('start', 'end', 'scene_index', 'motion', 'brightness', 'contrast', 'sharpness', 'visual_fingerprints')
        for cid, candidate in old_by_id.items():
            for field in preserved:
                assert candidate[field] == new_by_id[cid][field], (cid, field)
        assert report['after']['worker_launches'] == 1
        assert [e['stage'] for e in report['after']['events']] == ['scan'] * 10 + ['qwen']
        report['quality'] = {'identical_candidates': len(new_by_id), 'same_requested_frames': True,
                             'same_scan_descriptors': True, 'all_requested_tags_valid': True}
        # A repeat run should launch neither a video scan nor a model worker.
        with patch.object(video_analysis, 'VIDEO_ANALYSIS_CACHE_DIR', str(OUT / 'after_run' / 'analysis_cache')), \
             patch.object(video_analysis, '_analyze_single_video', side_effect=AssertionError('Cached video was rescanned')), \
             patch.object(video_analysis, '_run_qwen_worker_batch', side_effect=AssertionError('Cached AI reloaded')), \
             (OUT / 'cache_reuse.log').open('w', encoding='utf8') as log, redirect_stdout(log):
            tick = time.perf_counter()
            cached = video_analysis.analyze_video_sources(paths)
            report['cache_reuse'] = {'seconds': time.perf_counter() - tick, 'hits': cached['cache_hits'],
                                     'qwen_workers': cached['qwen_worker_count']}
        assert cached['cache_hits'] == 10
        report['seconds_saved'] = report['before']['seconds'] - report['after']['seconds']
        report['speedup'] = report['before']['seconds'] / report['after']['seconds']
        report['inputs'] = paths
        # Keep the existing rendering checks in the same real-input exercise.
        audio = next((ROOT / 'for amv').glob('*.flac'))
        output = OUT / 'shared_pipeline_test.mp4'
        info = {'video_analysis': new}
        random.seed(9271)
        with patch.object(video_processor, 'get_processing_dir', return_value=str(OUT / 'processing')), \
             (OUT / 'render.log').open('w', encoding='utf8') as log, redirect_stdout(log):
            video_processor.create_music_video(str(audio), paths, np.asarray([0., 2., 4., 6., 8.]),
                                               str(output), end_time=10., beat_info=info, fps=24,
                                               use_gpu=True, gpu_encoder='h264_nvenc')
        runtime.verify_video(output, 240)
        runtime.checked_output([media.FFMPEG_PATH, '-v', 'error', '-xerror', '-i', str(output), '-f', 'null', '-'], timeout=120)
        plan = json.loads(output.with_suffix('.clip_plan_debug.json').read_text(encoding='utf8'))
        assert plan['plan_summary']['unique_footage'] and plan['plan_summary']['clean_shots']
        for clip in plan['clips']:
            frames = list(iter_interval_frames(str(output), clip['audio_start'], clip['final_duration'], max_width=384))
            assert not native_shot_boundaries(frames), clip['index']
        report['render'] = {'file': str(output), 'frames': 240, 'full_decode': 'passed',
                             'unique_footage': True, 'clean_shots': True}
        report['passed'] = True
    finally:
        report['total_seconds'] = time.perf_counter() - started
        (OUT / 'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print('Benchmark report:', OUT / 'benchmark.json', flush=True)


if __name__ == '__main__':
    main()
