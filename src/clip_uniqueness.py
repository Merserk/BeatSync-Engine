"""Whole-edit source and visual uniqueness checks shared by planner and renderer."""
from collections import defaultdict
from functools import lru_cache
import base64
import math
import os

import cv2
import numpy as np


class UniqueFootageError(ValueError):
    """A complete edit cannot be made without repeating available footage."""


@lru_cache(maxsize=4096)
def source_key(path):
    path = os.path.normcase(os.path.realpath(os.path.abspath(str(path))))
    try:
        stat = os.stat(path)
        if stat.st_ino:
            return ('file', stat.st_dev, stat.st_ino)
    except OSError:
        pass
    return ('path', path)


def scene_key(item):
    if not item.get('video_file'):
        return None
    scene = item.get('scene_index')
    identity = str(item.get('identity_key') or '')
    if scene is None and ':scene:' in identity:
        scene = identity.rsplit(':scene:', 1)[1]
    if scene is None:
        return (source_key(item['video_file']), identity) if identity else None
    return (source_key(item['video_file']), str(scene))


def source_interval(item):
    try:
        start = float(item['start_time'])
        duration = float(item['source_duration'])
    except (KeyError, ValueError, TypeError):
        return None
    if not math.isfinite(start + duration) or start < 0 or duration <= 0:
        return None
    return start, start + duration


def frame_fingerprints(frames):
    """Small CPU descriptors of already-downloaded frames; no extra decoding.

    A perceptual hash is only a lookup key. Pixel and color checks must also
    match, preventing similar composition alone from excluding a different shot.
    Uninformative blank frames cannot establish a visual duplicate.
    """
    result = []
    for frame in frames:
        small = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        active = np.flatnonzero((gray.mean(axis=1) > 12) | (gray.std(axis=1) > 2))
        if active.size:
            small = small[active[0]:active[-1] + 1]
            gray = gray[active[0]:active[-1] + 1]
        if float(gray.std()) < 8:
            continue
        dct = cv2.dct(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))
        low = dct[:8, :8].reshape(-1)
        bits = low > np.median(low[1:])
        bits[0] = False
        hashed = sum(int(bit) << i for i, bit in enumerate(bits))
        thumb = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        item = {'hash': f'{hashed:016x}', 'pixels': base64.b64encode(thumb.tobytes()).decode('ascii'),
                'color': np.rint(small.mean(axis=(0, 1))).astype(int).tolist()}
        if item not in result:
            result.append(item)
    return result


def _bands(hashed):
    # <=4 differing bits must leave at least one of these five bands unchanged.
    for band in range(5):
        yield band, (hashed >> (13 * band)) & 8191


@lru_cache(maxsize=16384)
def _pixels(encoded):
    values = np.frombuffer(base64.b64decode(encoded), np.uint8)
    if values.size != 256:
        raise ValueError('Invalid visual fingerprint pixels')
    return values.astype(np.int16)


def _fingerprint_matches(a, b):
    if (int(a['hash'], 16) ^ int(b['hash'], 16)).bit_count() > 4:
        return False
    if max(abs(x - y) for x, y in zip(a['color'], b['color'])) > 12:
        return False
    difference = np.abs(_pixels(a['pixels']) - _pixels(b['pixels']))
    return float(difference.mean()) <= 6 and float(np.percentile(difference, 95)) <= 18


class ClipUniquenessIndex:
    """Index the whole edit, without an expiring scene/candidate cooldown."""
    def __init__(self, clips=()):
        self.ids = {}
        self.scenes = {}
        self.intervals = defaultdict(list)
        self.visual = defaultdict(list)
        for clip in clips:
            self.add(clip)

    def conflicts(self, clip):
        conflicts = {}
        cid = str(clip.get('candidate_id', clip.get('id')) or '')
        if cid and cid in self.ids:
            conflicts['candidate_reused'] = self.ids[cid]
        scene = scene_key(clip)
        if scene is not None and scene in self.scenes:
            conflicts['source_scene_reused'] = self.scenes[scene]
        interval = source_interval(clip)
        if interval and clip.get('video_file'):
            for start, end, index, margin in self.intervals[source_key(clip['video_file'])]:
                overlap = min(end, interval[1]) - max(start, interval[0])
                if overlap > 1e-6:
                    conflicts['source_footage_overlap'] = index
                    break
                if overlap > -max(margin, float(clip.get('source_guard_seconds', 0))) + 1e-6:
                    conflicts['source_proximity'] = index
                    break
        for fingerprint in clip.get('visual_fingerprints') or []:
            hashed = int(fingerprint['hash'], 16)
            tested = set()
            for band in _bands(hashed):
                for other, index in self.visual.get(band, ()):
                    token = id(other)
                    if token not in tested:
                        tested.add(token)
                        if _fingerprint_matches(fingerprint, other):
                            conflicts['visual_duplicate'] = index
                            break
                if 'visual_duplicate' in conflicts:
                    break
            if 'visual_duplicate' in conflicts:
                break
        return conflicts

    def add(self, clip):
        index = clip.get('index', 0)
        cid = str(clip.get('candidate_id', clip.get('id')) or '')
        if cid:
            self.ids.setdefault(cid, index)
        scene = scene_key(clip)
        if scene is not None:
            self.scenes.setdefault(scene, index)
        interval = source_interval(clip)
        if interval and clip.get('video_file'):
            self.intervals[source_key(clip['video_file'])].append(
                (*interval, index, float(clip.get('source_guard_seconds', 0))))
        for fingerprint in clip.get('visual_fingerprints') or []:
            for band in _bands(int(fingerprint['hash'], 16)):
                self.visual[band].append((fingerprint, index))


def validate_unique_clips(clips):
    index = ClipUniquenessIndex()
    for position, clip in enumerate(clips):
        conflict = index.conflicts(clip)
        if conflict:
            earlier = min(conflict.values())
            raise UniqueFootageError(
                f'Cuts {earlier + 1} and {position + 1} repeat source footage or a scene '
                f'({", ".join(sorted(conflict))}). Add different footage or shorten the edit.'
            )
        if source_interval(clip) is None:
            raise UniqueFootageError(f'Cut {position + 1} has an invalid source interval.')
        index.add(dict(clip, index=position))


def load_clip_fingerprints(clip, use_gpu=True):
    """Verify the actual consumed source frames, including long clip endpoints."""
    from gpu_video import iter_interval_frames
    def load(gpu):
        result = []
        for frame in iter_interval_frames(clip['video_file'], clip['start_time'],
                                          clip['source_duration'], use_gpu=gpu):
            result.extend(frame_fingerprints([frame]))
        return result
    try:
        return load(use_gpu)
    except Exception as exc:
        if not use_gpu:
            raise
        print(f'   GPU source verification failed; retrying CPU decoding: {exc}')
        return load(False)
