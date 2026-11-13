"""Keep a rendered cut inside one source shot, using native frame timestamps."""
import cv2
import numpy as np

from clip_uniqueness import frame_fingerprints
from ffmpeg_runtime import probe_media
from gpu_video import iter_interval_frames


def _correlation(a, b):
    a = a - float(a.mean())
    b = b - float(b.mean())
    scale = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.sum(a * b) / scale) if scale > 1e-5 else 1.0


def native_shot_boundaries(images):
    """Detect abrupt shot changes; ordinary motion and exposure flashes stay intact.

    The small CPU comparison runs on frames already decoded/resized on the GPU.
    Absolute difference alone is insufficient: a hard cut also needs a structural
    change and a jump relative to the surrounding motion or color distribution.
    """
    if len(images) < 2:
        return []
    gray = [cv2.cvtColor(cv2.resize(image, (128, 72), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY).astype(np.float32) for image in images]
    reference = np.median(np.stack(gray[:min(5, len(gray))]), axis=0)
    active = np.flatnonzero((reference.mean(axis=1) > 12) | (reference.std(axis=1) > 2))
    if active.size:
        gray = [image[active[0]:active[-1] + 1] for image in gray]
    differences = [float(np.abs(gray[i] - gray[i-1]).mean()) for i in range(1, len(gray))]
    result = []
    for i, difference in enumerate(differences, 1):
        if difference < 18 or _correlation(gray[i-1], gray[i]) >= .60:
            continue
        neighboring = differences[max(0, i-5):i-1] + differences[i:min(len(differences), i+4)]
        baseline = float(np.median(neighboring)) if neighboring else 0.0
        hist_a = cv2.calcHist([gray[i-1]], [0], None, [16], [0, 256])
        hist_b = cv2.calcHist([gray[i]], [0], None, [16], [0, 256])
        color_change = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA)
        if difference >= max(30., baseline * 2.5) or (difference >= 25 and color_change >= .60):
            result.append(i)
    # A brief flat white/black exposure flash between matching views is an
    # effect in the original shot, rather than a different camera view.
    ignored = set()
    for left in result:
        for right in range(left + 1, min(len(gray), left + 4)):
            if right not in result:
                continue
            if all(float(gray[n].std()) < 5 for n in range(left, right)) and \
                    _correlation(gray[left-1], gray[right]) > .90:
                ignored.update((left, right))
    return [i for i in result if i not in ignored]


def inspect_source_clip(clip, use_gpu=True):
    """Refine an approximate source interval and fingerprint its consumed frames.

    A short look around both endpoints catches the next shot before export.
    Output duration and playback speed are preserved. If the requested source
    duration cannot fit a continuous shot, the caller must select another one.
    """
    start = float(clip['start_time'])
    duration = float(clip['source_duration'])
    data = probe_media(clip['video_file'])
    offset = float(data.get('format', {}).get('start_time', 0) or 0)
    source_end = float(data['format']['duration'])
    scan_start = max(0., start - .35)
    scan_end = min(source_end, start + duration + .35)

    def decode(gpu):
        # Retain only small comparison images, rather than a full decoded clip.
        return [(timestamp, cv2.resize(image, (256, 144), interpolation=cv2.INTER_AREA))
                for timestamp, image in iter_interval_frames(clip['video_file'], scan_start, scan_end - scan_start,
                                                              use_gpu=gpu, with_timestamps=True)]
    try:
        frames = decode(use_gpu)
    except Exception as exc:
        if not use_gpu:
            raise
        print(f'   GPU shot verification failed; retrying CPU decoding: {exc}')
        frames = decode(False)
    timestamps = np.asarray([timestamp - offset for timestamp, _ in frames], dtype=float)
    images = [image for _, image in frames]
    if not images or not np.all(np.diff(timestamps) > 0):
        raise ValueError('Native source verification requires increasing frame timestamps.')
    period = float(np.median(np.diff(timestamps))) if len(frames) > 1 else 1 / 24
    cuts = native_shot_boundaries(images)
    # Millisecond MKV timestamps and decimal FFmpeg arguments need a small
    # exclusive-boundary margin. This is not an arbitrary frame deletion.
    guard = max(.002, period / 100)
    allowed_start = float(clip.get('usable_source_start', scan_start))
    allowed_end = float(clip.get('usable_source_end', scan_end))
    options = []
    positions = [0] + cuts + [len(frames)]
    for first, last in zip(positions[:-1], positions[1:]):
        lo = float(max(allowed_start, timestamps[first] + (guard if first in cuts else 0)))
        hi = float(min(allowed_end, timestamps[last] - guard if last < len(frames) else min(source_end, timestamps[-1] + period)))
        if hi - lo + 1e-7 < duration:
            continue
        proposed_start = max(lo, min(start, hi - duration))
        options.append((abs(proposed_start - start), proposed_start, lo, hi))
    if not options:
        return dict(clip, shot_verified=False, visual_fingerprints=[],
                    shot_integrity_failure='requested duration crosses a source-shot boundary')
    _, refined_start, lo, hi = min(options)
    selected = [image for timestamp, image in zip(timestamps, images)
                if refined_start - 1e-7 <= timestamp < refined_start + duration - 1e-7]
    if not selected:
        raise ValueError('No native source frames fit the verified source interval.')
    return dict(clip, start_time=refined_start, shot_verified=True,
                shot_refined=abs(refined_start - start) > 1e-7,
                source_start_shift=refined_start - start,
                verified_shot_start=float(lo), verified_shot_end=float(hi),
                native_source_frames=len(selected),
                visual_fingerprints=frame_fingerprints(selected),
                shot_boundary_count=int(sum(start < timestamps[n] < start + duration for n in cuts)))
