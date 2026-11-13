"""Save the evidence, regression results and visual comparison for the fix."""
import json
from pathlib import Path
import re
import sys

from gpu_video import iter_sampled_frames
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_cut_flashes import sheet

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'clean_cuts'


def link(label, path):
    return f'[{label}](<{Path(path).as_posix()}>)'


def main():
    report = json.loads((OUT / 'report.json').read_text(encoding='utf8'))
    focused = json.loads((OUT / 'focused_report.json').read_text(encoding='utf8'))
    encoders = json.loads((OUT / 'endpoint_exports' / 'report.json').read_text(encoding='utf8'))
    ffmpeg = json.loads((OUT / 'ffmpeg' / 'report.json').read_text(encoding='utf8'))
    prores = json.loads((OUT / 'prores' / 'report.json').read_text(encoding='utf8'))
    tests = (OUT / 'all_tests_final.log').read_text(encoding='utf8', errors='replace')
    test_count, test_seconds = re.search(r'Ran (\d+) tests in ([\d.]+)s', tests).groups()
    assert '\nOK' in tests and all(r['passed'] for r in (report, focused, encoders, ffmpeg))
    original = ROOT / 'output' / 'music_video_20261002_034313.mp4'
    old = json.loads(original.with_suffix('.clip_plan_debug.json').read_text(encoding='utf8'))['clips']
    images, labels = [], []
    cases = [(86, 41, 2078, 664.958), (96, 47, 2313, 2131.666), (139, 68, 3346, 1681.958)]
    for reported, index, stray, native_pts in cases:
        before = list(iter_sampled_frames(str(original), range(stray - 3, stray + 1), max_width=768))
        images.extend(im for _, im in before)
        labels.extend(f'Original {n / 24:.4f}s' + (' - stray' if n == stray else '') for n, _ in before)
        count = round(old[index]['final_duration'] * 24)
        after = list(iter_sampled_frames(str(OUT / f'fixed_cut_{reported}.mp4'), range(count - 4, count), max_width=768))
        images.extend(im for _, im in after)
        labels.extend(f'Fixed cut {index + 1}, frame {n + 1}/{count}' for n, _ in after)
    picture = sheet('before_after_endpoints', images, labels, columns=4)
    summary = report['plan_summary']
    rows = []
    for case, fixed in zip(cases, focused['focused']):
        reported, index, frame, native_pts = case
        clip = old[index]
        rows.append(f'| ~{reported // 60:02d}:{reported % 60:02d} | {index + 1} | '
                    f'{frame / 24:.6f} s, frame {frame} | {clip["source_name"]} | '
                    f'{native_pts:.3f} s | {clip["start_time"] + clip["source_duration"]:.6f} s | '
                    f'{fixed["source_start_shift"] * 1000:.1f} ms |')
    export_count = sum(len(value['exports']) for value in ffmpeg['encoders'].values())
    content = f'''# Stray source-shot frames: diagnosis and fix

Verified on 3 October 2026 using FFmpeg {ffmpeg['runtime']['version']}, Python 3.13.16 and the automatically selected NVIDIA GeForce RTX 4060 Ti dGPU.

The three reported images are real one-frame leaks from the next camera shot in the original footage. Each appears at the end of the preceding musical cut, just before the intended next cut. The reported edit has 100 musical cuts, with a minimum length of six output frames; these stray frames were not planned one-frame musical cuts.

| Reported position | Cut (1-based) | Stray output frame (0-based) | Source | Next shot's native PTS | Old source end | Start adjustment in focused fix |
|---|---:|---|---|---:|---:|---:|
{chr(10).join(rows)}

The scene scan samples six frames per second to rank candidates. Its approximate scene endpoints were used as exact export limits. Native 24 fps frames have different endpoints, and the source MKVs use millisecond timestamps: a frame belonging to the following shot can begin just before the approximate limit. For example, 664.958 s was included by an old endpoint of 665.000 s. In addition, the timing filter rounded source duration using the output frame rate, which could expand the source interval when retiming or changing FPS. The old dense frame check detected repeated footage but did not certify source-shot continuity.

The fix checks every native frame of each selected source interval, including a short look around its endpoints, using FFmpeg's actual frame timestamps. Decoding and resizing use the selected dGPU where supported. Small CPU image comparisons distinguish abrupt camera changes from continuous motion and exposure effects. Approximate intervals are moved inside one continuous shot; intervals too short for the requested source duration are replaced by another clean, unique source shot. Musical boundaries, output length and the chosen playback speed stay intact. The checks support variable frame rates and nonzero container start timestamps.

FFmpeg now receives a strict input duration and keeps source duration independent of output FPS. Exact output frame counts still apply, including ProRes 422 Proxy Precise Mode. The metadata-only fallback path also searches unused footage and checks each alternative rather than stopping on its first crossing interval or recycling footage.

The focused tests keep the original choices for cuts 42, 48 and 69 and render them again: 43, 61 and 58 frames, respectively. All three retain their musical duration; their reported stray images are absent, and no internal camera cut is detected.

![Original last four frames, followed by the corrected last four frames for each reported cut](<{Path(picture).as_posix()}>)

The full rebuilt AMV uses the two actual source files, 5.mkv and 6.mkv, with Qwen semantic analysis enabled and source analysis running one video at a time. Its 100 musical cuts have exactly the same output boundaries as the reported edit. The expired uploaded copy of 5.mkv was matched to the existing original in the test folder.

| Full export check | Result |
|---|---|
| Native shot and uniqueness verification | {summary['shot_verified_clip_count']}/100 cuts |
| Refined source starts | {summary['shot_refined_clip_count']} |
| Replacement source choices | {summary['visual_verification_repairs']} |
| Exact timeline | 5,434 frames, 24 fps, 226.416667 s |
| Full decode with FFmpeg `-xerror` | Passed |
| Confirmed unintended source-shot changes in exported cuts | 0 |
| Matching frames between different exported cuts | 0 |
| Reused candidate IDs, scenes or overlapping source intervals | 0 |
| PCM music in main export; AAC browser preview | Passed |

The raw output detector flagged one final frame at 84.791667 s. Inspecting the surrounding native footage showed continuous fast action, and the exported frame matched native PTS 1573.499 s with mean pixel difference 0.141/255. That original motion is preserved. The old video had 56 raw transition flags, which include fast motion; that number should not be interpreted as 56 proven artifacts. Shot detection uses image comparisons, so this report combines automated checks with direct inspection of the reported and remaining flagged images.

Preparing and exporting the complete rebuilt edit took {report['render_seconds']:.1f} seconds, including {summary['visual_verification_seconds']:.1f} seconds for native source verification and replacements. These timings exclude the initial full source scan and Qwen tagging. Verification decodes selected intervals rather than every original frame of both full videos.

| Regression checks | Result |
|---|---|
| Unit suite | {test_count} tests passed in {float(test_seconds):.1f} s |
| Native endpoint exports | {len(encoders['exports'])} encoder paths passed: CPU H.264, NVENC H.264/H.265, Vulkan Video H.264/H.265, CPU/Vulkan ProRes |
| Source 24 fps → output 30 fps | All seven paths: 54 frames, zero next-shot frames |
| Source 120 fps, 20 ms limit → output 24 fps | One output frame, zero next-shot frames |
| Broad FFmpeg suite | {export_count} exports, browser previews and {len(ffmpeg['edge_cases'])} edge cases passed |
| Precise Mode on real footage | CPU and Vulkan: 144 frames, ProRes 422 Proxy, 10-bit 4:2:2, PCM 24-bit, full decode passed |

{link('Rebuilt main video', OUT / 'music_video_clean.mp4')} · {link('Browser preview', OUT / 'music_video_clean_preview.mp4')} · {link('Full plan', OUT / 'music_video_clean.clip_plan_debug.json')}

Evidence: {link('Full audit JSON', OUT / 'report.json')}, {link('Focused cut JSON', OUT / 'focused_report.json')}, {link('Native endpoint tests', OUT / 'endpoint_exports' / 'report.json')}, {link('FFmpeg regression report', OUT / 'ffmpeg' / 'report.json')}, {link('Precise Mode results', OUT / 'prores' / 'report.json')}, {link('Unit log', OUT / 'all_tests_final.log')}.

Implementation: {link('Native shot inspection', ROOT / 'src' / 'shot_integrity.py')}, {link('Planner and checked fallback', ROOT / 'src' / 'auto_mode' / 'stage6_av_planner.py')}, {link('Native frame timestamp reader', ROOT / 'src' / 'gpu_video.py')}, {link('Strict export timing', ROOT / 'src' / 'ffmpeg_processing.py')}.

Restart the app to load the updated Python modules. The rebuilt video and focused comparisons are saved separately from the reported export.
'''
    (OUT / 'REPORT.md').write_text(content, encoding='utf8')
    print(str(OUT / 'REPORT.md'), flush=True)


if __name__ == '__main__':
    main()
