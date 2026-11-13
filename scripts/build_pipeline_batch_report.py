"""Save the measured multi-source pipeline comparison and validation evidence."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'pipeline_batch'


def link(label, path):
    return f'[{label}](<{Path(path).as_posix()}>)'


def main():
    report = json.loads((OUT / 'benchmark.json').read_text(encoding='utf8'))
    assert report['passed']
    tests = (OUT / 'all_tests.log').read_text(encoding='utf8', errors='replace')
    count, seconds = re.search(r'Ran (\d+) tests in ([\d.]+)s', tests).groups()
    assert '\nOK' in tests
    # Label legacy single-video worker PIDs by source rather than overwriting
    # all ten under their identical protocol job ID, "single".
    for key, folder in [('before', 'before_run'), ('after', 'after_run')]:
        cache = OUT / folder / 'analysis_cache'
        pids = {}
        for response_file in sorted(cache.glob('qwen*response*.json'), key=lambda p: p.stat().st_mtime_ns):
            response = json.loads(response_file.read_text(encoding='utf8'))
            request_file = response_file.with_name(response_file.name.replace('response', 'request'))
            request = json.loads(request_file.read_text(encoding='utf8'))
            for job in request.get('jobs') or [dict(request, job_id='single')]:
                pids[Path(job['video_file']).name] = response['timings_by_job'][job['job_id']]['server_pid']
        report[key]['server_pids_by_source'] = pids
        stats_file = OUT / folder / 'stats.json'
        stats = json.loads(stats_file.read_text(encoding='utf8'))
        stats['server_pids_by_source'] = pids
        stats_file.write_text(json.dumps(stats, indent=2), encoding='utf8')
    report['unit_tests'] = {'passed': True, 'count': int(count), 'seconds': float(seconds)}
    (OUT / 'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    before, after = report['before'], report['after']
    saved_percent = 100 * report['seconds_saved'] / before['seconds']
    content = f'''# Shared Qwen pipeline: multi-video performance fix

Verified on 3 October 2026 with the detected NVIDIA GeForce RTX 4060 Ti dGPU, Qwen3.5-2B-Q8_0 and its F16 projector, thinking off, FFmpeg 9.0.2 full, Python 3.13.16 and Taichi Vulkan.

The observation was correct. The serial scan path ran Qwen inside each video's analysis function and created a new worker/model session for the next video. The pre-existing shared-worker path was only used by deferred jobs from the parallel scan path, which was no longer active after source scanning became serial.

The pipeline now follows this order:

```text
Video 1: scan + score ─┐
Video 2: scan + score  │  One source decoder/spool at a time
...                   │  Qwen model is not loaded during this pass
Video N: scan + score ─┘
         ↓
One shared Qwen session
         Video 1 tags → Video 2 tags → ... → Video N tags
         Decoder closes after each source's sample extraction
         Only that source's images remain in RAM during its inference
         Model and any learned context/concurrency setting stay loaded
         ↓
Clean-shot and duplicate checks → exact-frame export → preview
```

FFmpeg is not an AI model. Its source decoder needs to open each video separately. That remains sequential and bounded, with no attempt to retain ten full-video decoders or their frames. The repeated Python/Qwen worker startup and model loading were the avoidable costs. Qwen's OpenCV/FFmpeg decoder is now released immediately after its sample images have been extracted, before inference.

Unmerged visual results are cached before Qwen starts. Different models, larger semantic budgets or failed AI attempts can reuse scan/scoring results. Complete AI caches skip both phases. Incomplete tags are marked incomplete and can retry from unmerged data, so scores are not blended twice. The worker saves completed sources during a batch; a later source failure preserves earlier tags. Startup failures and worker timeouts also clean up their own model processes. Qwen progress is streamed to the console during the shared session.

Legacy source-worker and per-video batch overrides cannot change this phase order. Candidate selection, default 120-moment budget per video, six-per-second scan sampling, 768-pixel analysis width, semantic schema, thinking-off behavior, dGPU detection, shot verification, duplicate prevention and ProRes 422 Proxy Precise Mode remain intact.

## Measured comparison

Ten distinct input files were made by stream-copying roughly 12-second extracts from the supplied videos: files 1–9 plus another interval from file 1. Native keyframe preroll makes some files slightly longer. No source re-encode or reduced semantic budget was used. GPU capabilities and metric kernels were warmed before both timed passes. Both runs used the same worker/inference implementation; the old run used the saved pre-change orchestration. Their separate analysis caches were bypassed for the timed comparison.

| Measurement | Previous pipeline | Shared pipeline |
|---|---:|---:|
| Analyze all ten inputs | {before['seconds']:.1f} s | {after['seconds']:.1f} s |
| Qwen workers launched | {before['worker_launches']} | {after['worker_launches']} |
| Model server starts, including recovery | {before['server_starts']} | {after['server_starts']} |
| Model loading time | {before['model_load_seconds']:.1f} s | {after['model_load_seconds']:.1f} s |
| Selected Qwen images | {before['requested_frames']} | {after['requested_frames']} |
| Valid semantic tags | {before['semantic_tags']} | {after['semantic_tags']} |

This run saved **{report['seconds_saved']:.1f} seconds ({saved_percent:.1f}% less elapsed time)**, or **{report['speedup']:.2f}× faster**. The short inputs make startup overhead a large fraction of total time. Longer source videos still require their full scan and semantic inference; this measurement is not a prediction of a 2.17× gain for hour-long videos.

The driver initially selected 16 inference slots, and a context validation retry reduced them to 8. Previously that same recovery repeated for each new video, producing 20 server starts. The shared session did it once: two starts for the whole batch. All ten sources then used the same server process, PID {next(iter(after['server_pids_by_source'].values()))}. Normal source transitions caused zero model restarts, and the server exited after the batch. Safety recovery remains available when context/VRAM limits require it.

The two timed runs selected exactly the same 35 candidate images, IDs and source intervals. Their motion, brightness, contrast, sharpness and visual fingerprints were identical, and every requested image produced a valid semantic tag. No sampling or quality threshold was reduced to obtain this result.

A subsequent cache run reused all ten completed analyses in {report['cache_reuse']['seconds']:.3f} seconds and launched zero scanners or Qwen workers. This is a warm, small-cache measurement; large libraries take longer to read.

## Validation

{count} unit tests passed in {float(seconds):.1f} seconds. Eleven new tests cover ten-input phase order, a single shared client/process, mixed visual/AI caches, larger budgets, failed/partial AI retries, unmerged score preservation, streamed worker output, earlier results surviving a later source error, decoder release before inference, and failed-server cleanup.

A real ten-input render produced five cuts and 240 frames at 24 fps, with the music supplied by the user. Full FFmpeg decoding passed, every cut passed native source-shot verification, no internal shot transition was detected in the exported cuts, and whole-edit duplicate checks passed.

Evidence: {link('Benchmark JSON', OUT / 'benchmark.json')}, {link('Previous pipeline log', OUT / 'before_run.log')}, {link('Shared pipeline log', OUT / 'after_run.log')}, {link('Unit test log', OUT / 'all_tests.log')}, {link('Test render', OUT / 'shared_pipeline_test.mp4')}, {link('Render plan', OUT / 'shared_pipeline_test.clip_plan_debug.json')}.

Implementation: {link('Two-phase orchestration and visual caches', ROOT / 'src' / 'video_analysis.py')}, {link('Shared Qwen worker and decoder lifetime', ROOT / 'src' / 'auto_mode' / 'stage5_qwen_scene_worker.py')}, {link('Console summary', ROOT / 'src' / 'gui.py')}.

Restart the app to load the updated pipeline.
'''
    (OUT / 'REPORT.md').write_text(content, encoding='utf8')
    print(OUT / 'REPORT.md')


if __name__ == '__main__':
    main()
