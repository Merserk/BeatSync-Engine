"""Check every rendered frame for near matches in other output clips."""
from collections import Counter
import json
from pathlib import Path
import time

from clip_uniqueness import ClipUniquenessIndex, frame_fingerprints
from gpu_video import iter_sampled_frames

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'unique_cuts'


def audit(path):
    plan = json.loads(path.with_suffix('.clip_plan_debug.json').read_text(encoding='utf8'))['clips']
    frame_to_clip = {}
    for index, item in enumerate(plan):
        for frame in range(round(item['audio_start'] * 24), round(item['audio_end'] * 24)):
            frame_to_clip[frame] = index
    seen = ClipUniquenessIndex()
    current = -1
    fingerprints = []
    conflicts = Counter()
    samples = []
    tick = time.perf_counter()
    for frame, image in iter_sampled_frames(str(path), frame_to_clip, max_width=384):
        clip_index = frame_to_clip[frame]
        if clip_index != current:
            if fingerprints:
                seen.add({'index': current, 'visual_fingerprints': fingerprints})
            current, fingerprints = clip_index, []
        values = frame_fingerprints([image])
        found = seen.conflicts({'index': current, 'visual_fingerprints': values})
        if 'visual_duplicate' in found:
            previous = found['visual_duplicate']
            conflicts[(previous, current)] += 1
            if len(samples) < 30:
                samples.append({'earlier_clip': previous, 'clip': current, 'output_seconds': frame / 24})
        fingerprints.extend(values)
    return {'file': str(path), 'frames': len(frame_to_clip), 'seconds': time.perf_counter() - tick,
            'matching_frame_count': sum(conflicts.values()),
            'pairs': [{'earlier_clip': a, 'clip': b, 'matching_frames': count}
                      for (a, b), count in conflicts.items()], 'examples': samples}


def main():
    report = {}
    for label, path in [('original', ROOT / 'output' / 'music_video_20261002_011441.mp4'),
                        ('repaired', OUT / 'music_video_unique.mp4')]:
        report[label] = audit(path)
        print(label + ': ' + json.dumps({k: v for k, v in report[label].items() if k != 'examples'}), flush=True)
    (OUT / 'rendered_frame_audit.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    assert report['repaired']['matching_frame_count'] == 0, report['repaired']


if __name__ == '__main__':
    main()
