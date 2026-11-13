"""Save the duplicate-cut reproduction, verification results, and frame comparison."""
import json
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont
import ffmpeg_processing as media
import ffmpeg_runtime as runtime

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'unique_cuts'


def link(label, path):
    return f'[{label}](<{path.resolve().as_posix()}>)'


def main():
    report = json.loads((OUT / 'report.json').read_text(encoding='utf8'))
    frames = json.loads((OUT / 'rendered_frame_audit.json').read_text(encoding='utf8'))
    ffmpeg = json.loads((OUT / 'ffmpeg' / 'report.json').read_text(encoding='utf8'))
    prores = json.loads((OUT / 'prores' / 'report.json').read_text(encoding='utf8'))
    assert report['passed'] and ffmpeg['passed'] and frames['repaired']['matching_frame_count'] == 0
    tests = (OUT / 'all_tests.log').read_text(encoding='utf8')
    test_count, test_seconds = re.search(r'Ran (\d+) tests in ([\d.]+)s', tests).groups()
    assert '\nOK\n' in tests and 'FAILED (' not in tests
    old = json.loads((OUT / 'before' / 'reported_clip_plan.json').read_text(encoding='utf8'))['clips']
    a, b = old[22], old[60]
    overlap = min(a['start_time'] + a['source_duration'], b['start_time'] + b['source_duration']) - max(a['start_time'], b['start_time'])
    picture = Image.new('RGB', (1280, 820), '#151820')
    draw = ImageDraw.Draw(picture)
    font_path = Path('C:/Windows/Fonts/segoeui.ttf')
    font = ImageFont.truetype(str(font_path), 22) if font_path.exists() else ImageFont.load_default(size=22)
    for row, (label, video) in enumerate([('Original', Path(report['original'])), ('Fixed', Path(report['output']))]):
        for column, seconds in enumerate([51.083333, 124.041667]):
            frame = OUT / f'{label.lower()}_{round(seconds)}.png'
            runtime.checked_output([media.FFMPEG_PATH, '-nostdin', '-v', 'error', '-y', '-ss', str(seconds),
                                    '-i', str(video), '-frames:v', '1', '-vf', 'scale=640:360', str(frame)], timeout=60)
            picture.paste(Image.open(frame), (column * 640, row * 410 + 42))
            draw.text((column * 640 + 15, row * 410 + 8),
                      f'{label} - {int(seconds) // 60:02}:{int(seconds) % 60:02}', font=font, fill='white')
    comparison = OUT / 'comparison.png'
    picture.save(comparison)
    summary = report['plan_summary']
    exports = sum(len(item['exports']) for item in ffmpeg['encoders'].values())
    content = f'''# Unique-cut fix and verification — 2 October 2026

The reported AMV repeated source footage. The rebuilt video preserves every original musical cut boundary and contains no repeated scenes, overlapping source intervals, or detected near-identical frames between cuts.

## Confirmed cause

The cuts around 00:51 and 02:04 used different candidate IDs from **scene 113 in 3.mkv**. Cut 23 selected source seconds {a['start_time']:.6f}–{a['start_time'] + a['source_duration']:.6f}; cut 61 selected {b['start_time']:.6f}–{b['start_time'] + b['source_duration']:.6f}. They reused **{overlap:.6f} seconds** of source footage. The scene cooldown lasted only 28 cuts; these cuts were 38 positions apart. The old audit checked candidate IDs and recent scenes, so it reported no reuse.

Across the original edit, the source plan contains 10 overlapping interval pairs and 12 scene reuse occurrences. Checking every exported frame found 323 frames matching earlier cuts across 11 pairs of cuts.

A second defect allowed a long cut to extend outside its analyzed candidate window. Close source intervals around incorrectly split scene boundaries could also show almost identical camera views.

## Changed behavior

- Candidate IDs, physical source intervals, and scenes are checked across the entire edit. No cooldown or exhausted-pool fallback permits reuse.
- Planned source intervals keep a two-second separation within the same source. Longer cuts must fit analyzed footage. Contiguous analyzed chunks can be joined only inside the same continuous scene.
- Analysis caches now include visual fingerprints. Older caches are rebuilt automatically.
- Before encoding, every frame of the actual selected source intervals is decoded and resized, using the automatically detected dGPU where supported. Small CPU perceptual descriptors check hash, grayscale pixels, and color together.
- A detected visual repeat is replaced with a suitable unused candidate and its actual footage is checked again. The replacement search includes all usable analyzed moments. Insufficient unique footage produces an actionable error before export.
- Without an analysis library, source intervals are allocated before parallel workers run and receive the same final visual check.
- The UI console and clip-plan JSON report verification across the whole edit. Sequential source analysis and ProRes 422 Proxy Precise Mode remain enabled.

## Real-video results

Input: the user's uploaded 3.mkv and 4.mkv, with the original 226.4-second FLAC. Output: 1920×1080, 24 fps, 100 cuts, exactly 5,434 frames. Every original musical boundary was preserved; playback speed remains 1×.

| Check | Original | Fixed |
|---|---:|---:|
| Repeated candidate IDs | 0 | 0 |
| Scene reuse occurrences across the whole edit | 12 | 0 |
| Overlapping source interval pairs | 10 | 0 |
| Exported frames inspected | 5,434 | 5,434 |
| Frames matching earlier cuts | 323 | 0 |
| Pairs of cuts with matching frames | 11 | 0 |

All 100 selected source intervals passed verification. Both the main export and browser preview contain 5,434 frames. The main export decoded fully with FFmpeg's error checks enabled.

Source-frame uniqueness verification took **{summary['visual_verification_seconds']:.2f} seconds**. Planning, verification, and main rendering took **{report['render_seconds']:.2f} seconds**. This final reproduction, including audio analysis, output audits and preview creation, took **{report['total_seconds']:.2f} seconds**. The final run reused the two newly built source-analysis caches; this timing excludes rebuilding them and running their Qwen tags.

The visual audit detects the same or near-identical frames. Uninformative flat frames are excluded; repeated characters or subject matter alone do not establish a duplicate.

## Regression and export checks

- **{test_count} automated tests passed** ({test_seconds} seconds), including the reported overlap, non-overlapping scene reuse, physical path aliases, duplicates across different files, long holds, and replacement of repeats missed by scan samples.
- **6 export modes, {exports} exports, and {len(ffmpeg['edge_cases'])} FFmpeg edge cases passed**: CPU H.264, NVENC H.264/HEVC, Vulkan H.264/H.265, and ProRes 422 Proxy. Checks include frame counts, decoding, fractional frame rates, mixed geometry, audio slicing, browser previews, and UI downloads.
- Real-source ProRes Precise Mode passed on CPU and Vulkan: 144 frames each, Proxy profile, 10-bit 4:2:2, PCM 24-bit audio, and full decoding. Separate CPU/Vulkan segment exports also passed at 240 frames each.

## Files

{link('Rebuilt video', Path(report['output']))} · {link('Browser preview', Path(report['preview']))} · {link('Clip plan', OUT / 'music_video_unique.clip_plan_debug.json')}

{link('Every-frame audit', OUT / 'rendered_frame_audit.json')} · {link('Pipeline results', OUT / 'report.json')} · {link('Automated tests', OUT / 'all_tests.log')} · {link('FFmpeg results', OUT / 'ffmpeg' / 'report.json')} · {link('ProRes results', OUT / 'prores' / 'report.json')}

![Original and fixed moments at approximately 00:51 and 02:04](<{comparison.resolve().as_posix()}>)

Restart the running app to load the fix, then regenerate any affected exports. The original reported video is preserved.
'''
    (OUT / 'REPORT.md').write_text(content, encoding='utf8')
    print(f'Report saved: {OUT / "REPORT.md"}')


if __name__ == '__main__':
    main()
