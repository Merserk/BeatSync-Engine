#!/usr/bin/env python3
"""Audio-visual clip planner for Auto Mode."""

from __future__ import annotations

import hashlib
import os
import random
import time
from collections import Counter, defaultdict
from typing import Dict, List, Sequence

import numpy as np
from clip_uniqueness import (ClipUniquenessIndex, UniqueFootageError, validate_unique_clips,
                             load_clip_fingerprints, source_key, scene_key, source_interval)
from ffmpeg_runtime import probe_media
from shot_integrity import inspect_source_clip


RECENT_CANDIDATE_COOLDOWN = 40
RECENT_SCENE_COOLDOWN = 28
RECENT_TIME_BUCKET_COOLDOWN = 28
SOURCE_TIME_BUCKET_SECONDS = 12.0
SOURCE_GUARD_SECONDS = 2.0


def _clamp(value, lo: float = 0.0, hi: float = 1.0, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        v = default
    if not np.isfinite(v):
        v = default
    return max(lo, min(hi, v))


def _stable_rng(*parts) -> random.Random:
    raw = "|".join(str(p) for p in parts)
    seed = int(hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:12], 16)
    return random.Random(seed)


def _env_int(name: str, default: int, lo: int = 1, hi: int | None = None) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    if hi is None:
        hi = max(lo, value)
    return max(lo, min(hi, value))


def build_planned_clip_sequence(
    cut_times: Sequence[float],
    segment_durations: Sequence[float],
    beat_info: Dict | None,
    video_files: Sequence[str],
    auto_speed_clip_control: bool = True,
    avoid_text_clips: bool = True,
) -> List[Dict]:
    """Build exact source clip choices for every output segment.

    Returns an empty list when no visual library is present, which tells the
    renderer to allocate non-overlapping fallback intervals. An exhausted visual
    library raises an error instead of silently recycling clips.
    """
    beat_info = beat_info or {}
    video_analysis = beat_info.get("video_analysis") or {}
    cut_times_arr = np.asarray(cut_times, dtype=float)
    durations_arr = np.asarray(segment_durations, dtype=float)
    if cut_times_arr.size < 2 or durations_arr.size == 0:
        return []

    candidates, planner_diagnostics = _build_planner_candidate_pool(
        video_analysis,
        int(durations_arr.size),
        avoid_text_clips=avoid_text_clips,
    )
    if not candidates:
        return []

    profiles = _build_segment_profiles(cut_times_arr, durations_arr, beat_info)
    try:
        planned = _build_sequence_plan(candidates, profiles, auto_speed_clip_control=auto_speed_clip_control)
    except UniqueFootageError:
        # Search every usable raw moment before declaring the library exhausted.
        raw = list(video_analysis.get('candidates') or [])
        expanded = {_candidate_id(c): c for c in (
            _prepare_candidate_for_planner(item) for item in raw + _extended_scene_candidates(raw))
            if _candidate_usable_for_planner(c, avoid_text_clips=avoid_text_clips and
                                            not planner_diagnostics.get('text_filter_fallback'))}
        if len(expanded) <= len(candidates):
            raise
        candidates = list(expanded.values())
        print(f'   Expanding the unique-scene search to all {len(candidates)} usable moments')
        planned = _build_sequence_plan(candidates, profiles, auto_speed_clip_control=auto_speed_clip_control)
        planner_diagnostics['planner_pool_size'] = len(candidates)
        planner_diagnostics['planner_pool_expanded'] = True

    if len(planned) != len(durations_arr):
        raise UniqueFootageError('Could not fill the edit with unique scenes. Add footage or shorten the edit.')
    for item in planned:
        item.update(planner_diagnostics)
    return planned


def verify_planned_clip_sequence(plan, cut_times, segment_durations, beat_info, video_files,
                                 use_gpu=True, auto_speed_clip_control=True, avoid_text_clips=True,
                                 verify_shots=True):
    """Inspect all consumed source frames and replace any remaining repeated shot.

    Scan samples guide selection, but cannot certify the endpoints of the final
    intervals. Verification decodes just those intervals, one at a time, on the
    selected GPU where supported. Every replacement must pass the same check.
    """
    started = time.perf_counter()
    verified = [dict(item) for item in plan]
    cache = {}

    def inspect(item):
        key = (source_key(item['video_file']), float(item['start_time']), float(item['source_duration']),
               item.get('usable_source_start'), item.get('usable_source_end'))
        if key not in cache:
            cache[key] = (inspect_source_clip(item, use_gpu=use_gpu) if verify_shots else
                          dict(item, visual_fingerprints=load_clip_fingerprints(item, use_gpu=use_gpu), shot_verified=True))
        checked = dict(item)
        fields = ('start_time', 'visual_fingerprints', 'shot_verified', 'shot_refined',
                  'source_start_shift', 'verified_shot_start', 'verified_shot_end',
                  'native_source_frames', 'shot_boundary_count', 'shot_integrity_failure')
        checked.update({k: cache[key][k] for k in fields if k in cache[key]})
        return checked

    print(f'   Checking native source frames for clean shots and duplicates ({len(verified)} cuts)')
    for position, item in enumerate(verified):
        item = verified[position] = inspect(item)
        item['visual_verified'] = bool(item.get('shot_verified'))
        if (position + 1) % 10 == 0 or position + 1 == len(verified):
            print(f'   Source uniqueness check: {position + 1}/{len(verified)} cuts')

    profiles = _build_segment_profiles(np.asarray(cut_times, dtype=float),
                                       np.asarray(segment_durations, dtype=float), beat_info or {})
    analysis = (beat_info or {}).get('video_analysis') or {}
    pool, diagnostics = _build_planner_candidate_pool(analysis, len(verified), avoid_text_clips)
    allowed = {source_key(path) for path in video_files}
    # Repair may need a moment outside the initial highlight pool. Search all
    # usable, selected-source moments before reporting insufficient footage.
    raw = list(analysis.get('candidates') or [])
    prepared = [_prepare_candidate_for_planner(item) for item in raw + _extended_scene_candidates(raw) + pool]
    candidates = {_candidate_id(item): item for item in prepared
                  if item.get('video_file') and source_key(item['video_file']) in allowed
                  and _candidate_usable_for_planner(item, avoid_text_clips=avoid_text_clips and
                                                   not diagnostics.get('text_filter_fallback'))}
    index = ClipUniquenessIndex()
    repairs = 0
    for position, item in enumerate(verified):
        conflict = index.conflicts(item)
        if not item.get('shot_verified'):
            conflict['source_shot_boundary'] = position
        if conflict:
            print(f'   Replacing cut {position + 1}: {", ".join(sorted(conflict))}')
            ranked = _rank_profile_candidates(list(candidates.values()), profiles[position], len(candidates))
            excluded = set()
            others = ClipUniquenessIndex(c for i, c in enumerate(verified) if i != position)
            replacement = None
            while len(excluded) < len(candidates):
                proposed = _find_repetition_replacement(
                    position, verified, ranked, profiles[position], auto_speed_clip_control,
                    excluded_ids=excluded)
                if proposed is None:
                    break
                excluded.add(str(proposed['candidate_id']))
                proposed = inspect(proposed)
                if proposed.get('shot_verified') and not others.conflicts(proposed):
                    replacement = dict(item)
                    replacement.update(proposed)
                    replacement.update(visual_verified=True, repetition_repaired=True,
                                       replaced_candidate_id=item.get('candidate_id'))
                    replacement.pop('shot_integrity_failure', None)
                    break
            if replacement is None and not raw and not pool:
                replacement = _find_clean_fallback_replacement(item, verified, position, video_files, inspect)
            if replacement is None:
                raise UniqueFootageError(
                    f'Cannot fill cut {position + 1} with a clean, unique source shot. '
                    'Add different footage or shorten the edit; no duplicate export was created.')
            verified[position] = item = replacement
            repairs += 1
        index.add(item)

    validate_unique_clips(verified)
    elapsed = time.perf_counter() - started
    for item in verified:
        item['visual_verification_seconds'] = elapsed
        item['visual_verification_repairs'] = repairs
    print(f'   Unique footage verified: {len(verified)} cuts, {repairs} replacements, {elapsed:.2f}s')
    return _annotate_clip_plan_repetition(verified)


def _find_clean_fallback_replacement(item, plan, position, video_files, inspect):
    """Check unused intervals when the caller has no analyzed scene library.

    This preserves the metadata-only export path without letting a random seek
    cross a camera cut. Each alternative receives the same native-frame and
    whole-edit uniqueness checks as an analyzed candidate.
    """
    duration = float(item['source_duration'])
    others = [clip for i, clip in enumerate(plan) if i != position]
    index = ClipUniquenessIndex(others)
    sources = {}
    for path in video_files:
        key = source_key(path)
        if key in sources:
            continue
        end = float(probe_media(path)['format']['duration'])
        reserved = sorted((max(0., interval[0] - SOURCE_GUARD_SECONDS),
                           min(end, interval[1] + SOURCE_GUARD_SECONDS))
                          for clip in others if clip.get('video_file') and source_key(clip['video_file']) == key
                          if (interval := source_interval(clip)) is not None)
        gaps = []
        cursor = 0.
        for lo, hi in reserved:
            if lo - cursor >= duration:
                gaps.append((cursor, lo))
            cursor = max(cursor, hi)
        if end - cursor >= duration:
            gaps.append((cursor, end))
        sources[key] = (path, gaps)
    choices = [(path, lo, hi) for path, gaps in sources.values() for lo, hi in gaps]
    if not choices:
        return None
    rng = _stable_rng('clean_source_fallback', position, duration, sorted(str(key) for key in sources))
    # A failed continuous shot is discarded; no candidate is reused to fill an
    # exhausted library. The bound prevents a pathological source from hanging.
    for attempt in range(128):
        path, lo, hi = rng.choice(choices)
        start = rng.uniform(lo, hi - duration)
        proposed = dict(item, video_file=path, source_name=os.path.basename(path), start_time=start,
                        usable_source_start=lo, usable_source_end=hi,
                        candidate_id=hashlib.sha1(f'{source_key(path)}:{start:.9f}:{duration}'.encode()).hexdigest(),
                        selection_level='verified_source_fallback', source_guard_seconds=SOURCE_GUARD_SECONDS)
        for key in ('scene_index', 'identity_key', 'source_time_bucket', 'visual_fingerprints',
                    'shot_integrity_failure', 'shot_verified', 'shot_refined', 'source_start_shift',
                    'verified_shot_start', 'verified_shot_end', 'native_source_frames', 'shot_boundary_count'):
            proposed.pop(key, None)
        checked = inspect(proposed)
        if checked.get('shot_verified') and not index.conflicts(checked):
            checked.update(visual_verified=True, repetition_repaired=True,
                           replaced_candidate_id=item.get('candidate_id'))
            return checked
    return None


def summarize_clip_plan(plan: Sequence[Dict]) -> Dict:
    if not plan:
        return {"clip_count": 0, "targets": {}, "ai_tagged": 0}
    repetition_audit = audit_clip_plan_repetition(plan)
    targets = Counter(str(item.get("target", "flow")) for item in plan)
    impact_targets = Counter(str(item.get("impact_class", item.get("target", "flow"))) for item in plan)
    candidate_counts = Counter(str(item.get("candidate_id")) for item in plan if item.get("candidate_id"))
    candidate_reuse_count = sum(max(0, count - 1) for count in candidate_counts.values())
    scene_repetition_count = 0
    previous_identity = None
    for item in plan:
        identity = item.get("identity_key")
        if identity and previous_identity and identity == previous_identity:
            scene_repetition_count += 1
        previous_identity = identity
    ai_tagged = sum(1 for item in plan if item.get("ai_analyzed"))
    source_count = len(set(item.get("video_file") for item in plan))
    match_scores = [float(item.get("match_score", 0.0)) for item in plan if item.get("match_score") is not None]
    flow_scores = [float(item.get("flow_score", 0.0)) for item in plan if item.get("flow_score") is not None]
    low_quality_used = sum(1 for item in plan if item.get("low_quality"))
    text_overlay_used = sum(1 for item in plan if item.get("text_overlay"))
    highlight_pool_size = max([int(item.get("highlight_pool_size") or 0) for item in plan] or [0])
    highlight_seed_size = max([int(item.get("highlight_seed_size") or 0) for item in plan] or [0])
    planner_pool_size = max([int(item.get("planner_pool_size") or 0) for item in plan] or [0])
    planner_pool_target = max([int(item.get("planner_pool_target") or 0) for item in plan] or [0])
    raw_candidate_size = max([int(item.get("raw_candidate_size") or 0) for item in plan] or [0])
    planner_pool_deficit = max([int(item.get("planner_pool_deficit") or 0) for item in plan] or [0])
    avoid_text_clips = any(bool(item.get("avoid_text_clips")) for item in plan)
    text_candidates_filtered = max([int(item.get("text_candidates_filtered") or 0) for item in plan] or [0])
    text_candidate_count = max([int(item.get("text_candidate_count") or 0) for item in plan] or [0])
    text_filter_fallback = any(bool(item.get("text_filter_fallback")) for item in plan)
    speeds = [_clamp(item.get("playback_speed", 1.0), lo=0.05, hi=8.0, default=1.0) for item in plan]
    slowed_count = sum(1 for speed in speeds if speed < 0.98)
    sped_up_count = sum(1 for speed in speeds if speed > 1.02)
    original_speed_count = len(plan) - slowed_count - sped_up_count
    strongest_speed = max(speeds, key=lambda speed: abs(speed - 1.0)) if speeds else 1.0
    return {
        "clip_count": len(plan),
        "targets": dict(targets),
        "ai_tagged": ai_tagged,
        "source_count": source_count,
        "highlight_pool_size": highlight_pool_size,
        "highlight_seed_size": highlight_seed_size,
        "planner_pool_size": planner_pool_size,
        "planner_pool_target": planner_pool_target,
        "raw_candidate_size": raw_candidate_size,
        "planner_pool_deficit": planner_pool_deficit,
        "candidate_reuse_count": candidate_reuse_count,
        "scene_repetition_count": scene_repetition_count,
        "avg_match_score": float(np.mean(match_scores)) if match_scores else 0.0,
        "avg_flow_score": float(np.mean(flow_scores)) if flow_scores else 0.0,
        "low_quality_used_count": low_quality_used,
        "text_overlay_used_count": text_overlay_used,
        "avoid_text_clips": avoid_text_clips,
        "text_candidate_count": text_candidate_count,
        "text_candidates_filtered": text_candidates_filtered,
        "text_filter_fallback": text_filter_fallback,
        "impact_target_counts": dict(impact_targets),
        "speed_slowed_count": slowed_count,
        "speed_sped_up_count": sped_up_count,
        "speed_original_count": original_speed_count,
        "average_playback_speed": float(np.mean(speeds)) if speeds else 1.0,
        "strongest_playback_speed": float(strongest_speed),
        "repeated_motif_count": repetition_audit.get("repeated_motif_count", 0),
        "repeated_motifs": repetition_audit.get("repeated_motifs", []),
        "recent_candidate_repeat_count": repetition_audit.get("recent_candidate_repeat_count", 0),
        "recent_scene_repeat_count": repetition_audit.get("recent_scene_repeat_count", 0),
        "scene_time_bucket_repetition_count": repetition_audit.get("scene_time_bucket_repetition_count", 0),
        "reused_candidate_ids": repetition_audit.get("reused_candidate_ids", []),
        "global_scene_repetition_count": repetition_audit.get("global_scene_repetition_count", 0),
        "source_overlap_count": repetition_audit.get("source_overlap_count", 0),
        "source_proximity_count": repetition_audit.get("source_proximity_count", 0),
        "visual_duplicate_count": repetition_audit.get("visual_duplicate_count", 0),
        "visual_verified_clip_count": sum(bool(item.get('visual_verified')) for item in plan),
        "visual_verification_repairs": max(int(item.get('visual_verification_repairs', 0)) for item in plan),
        "visual_verification_seconds": max(float(item.get('visual_verification_seconds', 0)) for item in plan),
        "shot_verified_clip_count": sum(bool(item.get('shot_verified')) for item in plan),
        "shot_refined_clip_count": sum(bool(item.get('shot_refined')) for item in plan),
        "clean_shots": all(bool(item.get('shot_verified')) for item in plan),
        "unique_footage": not any(repetition_audit.get(k, 0) for k in (
            "global_scene_repetition_count", "source_overlap_count", "source_proximity_count", "visual_duplicate_count")) and not candidate_reuse_count,
    }


def _build_planner_candidate_pool(video_analysis: Dict, clip_count: int,
                                  avoid_text_clips: bool = True) -> tuple[List[Dict], Dict]:
    highlight_seed = list(video_analysis.get("highlight_pool") or []) if isinstance(video_analysis, dict) else []
    raw_candidates = list(video_analysis.get("candidates") or []) if isinstance(video_analysis, dict) else []
    raw_candidate_size = len(raw_candidates)
    extended = _extended_scene_candidates(raw_candidates or highlight_seed)
    raw_candidates += extended
    source_candidates = raw_candidates or highlight_seed
    target_size = _planner_target_pool_size(clip_count, len(source_candidates))

    selected: Dict[str, Dict] = {}
    seed_prepared = [_prepare_candidate_for_planner(c) for c in highlight_seed + extended if c.get("video_file")]
    for candidate in seed_prepared:
        if _candidate_usable_for_planner(candidate, avoid_text_clips=avoid_text_clips):
            selected[_candidate_id(candidate)] = candidate

    raw_prepared = [_prepare_candidate_for_planner(c) for c in raw_candidates if c.get("video_file")]
    usable_raw_all = [candidate for candidate in raw_prepared if _candidate_usable_for_planner(candidate, avoid_text_clips=False)]
    usable_raw_clean = [candidate for candidate in usable_raw_all if not _candidate_has_text_overlay(candidate)]
    seed_clean = [candidate for candidate in seed_prepared if _candidate_usable_for_planner(candidate, avoid_text_clips=avoid_text_clips)]
    seed_all = [candidate for candidate in seed_prepared if _candidate_usable_for_planner(candidate, avoid_text_clips=False)]
    text_candidate_count = sum(1 for candidate in (usable_raw_all or seed_all) if _candidate_has_text_overlay(candidate))
    text_filter_fallback = bool(avoid_text_clips and len(usable_raw_clean or seed_clean) < max(1, clip_count))

    if avoid_text_clips and not text_filter_fallback:
        expansion_candidates = usable_raw_clean or seed_clean
    else:
        expansion_candidates = usable_raw_all or seed_all
    available_count = len(expansion_candidates)
    desired_size = min(target_size, available_count) if available_count else len(selected)

    if len(selected) < desired_size:
        _fill_selected_time_diverse(selected, expansion_candidates, desired_size)

    if not selected:
        fallback_candidates = raw_prepared or seed_prepared
        desired_fallback = min(target_size, len(fallback_candidates))
        _fill_selected_time_diverse(selected, fallback_candidates, desired_fallback)

    pool = list(selected.values())
    diagnostics = {
        "highlight_pool_size": int(video_analysis.get("highlight_pool_size") or len(highlight_seed)) if isinstance(video_analysis, dict) else len(highlight_seed),
        "highlight_seed_size": len(highlight_seed),
        "raw_candidate_size": raw_candidate_size,
        "planner_pool_target": int(target_size),
        "planner_pool_size": len(pool),
        "planner_pool_deficit": max(0, int(target_size) - len(pool)),
        "avoid_text_clips": bool(avoid_text_clips),
        "text_candidate_count": int(text_candidate_count),
        "text_candidates_filtered": int(text_candidate_count if avoid_text_clips and not text_filter_fallback else 0),
        "text_filter_fallback": bool(text_filter_fallback),
    }
    return pool, diagnostics


def _extended_scene_candidates(candidates):
    """Join analyzed chunks of one continuous scene for longer musical holds."""
    groups = defaultdict(list)
    for item in candidates:
        if item.get('kind') == 'scene_chunk' and item.get('scene_index') is not None:
            groups[scene_key(item)].append(item)
    extended = []
    for items in groups.values():
        items = sorted(items, key=lambda item: float(item['start']))
        if len(items) < 2:
            continue
        end = float(items[0]['end'])
        continuous = True
        for item in items[1:]:
            if float(item['start']) > end + .001:
                continuous = False
                break
            end = max(end, float(item['end']))
        if not continuous:
            continue
        start = float(items[0]['start'])
        joined = dict(max(items, key=lambda item: float(item.get('highlight_score', 0))))
        joined.update(id=f'{joined["id"]}_full_scene', kind='full_scene', start=start, end=end,
                      duration=end - start, center=(start + end) / 2,
                      usable_interval={'start': start, 'end': end},
                      quality_score=min(float(item.get('quality_score', .5)) for item in items),
                      text_score=max(float(item.get('text_score', 0)) for item in items),
                      quality_flags=sorted({flag for item in items for flag in item.get('quality_flags', [])}),
                      visual_fingerprints=[value for item in items for value in item.get('visual_fingerprints', [])])
        extended.append(joined)
    return extended


def _planner_target_pool_size(clip_count: int, available_count: int) -> int:
    target = max(clip_count * 4, clip_count + 80, 240)
    if available_count <= 0:
        return target
    return min(target, int(available_count))


def _candidate_usable_for_planner(candidate: Dict, avoid_text_clips: bool = True) -> bool:
    flags = {str(flag) for flag in candidate.get("quality_flags", [])}
    hard_flags = {"dark", "blown", "blurry", "low_quality", "too_short", "weak_highlight"}
    if flags.intersection(hard_flags):
        return False
    if avoid_text_clips and _candidate_has_text_overlay(candidate):
        return False
    if _clamp(candidate.get("highlight_score", 0.0)) < 0.30:
        return False
    return bool(candidate.get("video_file"))


def _candidate_has_text_overlay(candidate: Dict) -> bool:
    flags = {str(flag) for flag in candidate.get("quality_flags", [])}
    if "text_overlay" in flags:
        return True
    return _clamp(candidate.get("text_score", 0.0)) >= 0.58


def _fill_selected_time_diverse(selected: Dict[str, Dict], candidates: Sequence[Dict], target_size: int) -> None:
    if len(selected) >= target_size:
        return
    bucketed: Dict[str, List[Dict]] = {}
    for candidate in candidates:
        bucketed.setdefault(_source_time_bucket(candidate), []).append(candidate)
    for items in bucketed.values():
        items.sort(key=_planner_pool_sort_key, reverse=True)

    bucket_keys = sorted(bucketed)
    bucket_offsets = {key: 0 for key in bucket_keys}
    while len(selected) < target_size and bucket_keys:
        added_this_round = 0
        next_keys: List[str] = []
        for key in bucket_keys:
            items = bucketed[key]
            cursor = bucket_offsets.get(key, 0)
            while cursor < len(items):
                candidate = items[cursor]
                cursor += 1
                cid = _candidate_id(candidate)
                if cid not in selected:
                    selected[cid] = candidate
                    added_this_round += 1
                    break
            bucket_offsets[key] = cursor
            if cursor < len(items):
                next_keys.append(key)
            if len(selected) >= target_size:
                break
        if added_this_round == 0:
            break
        bucket_keys = next_keys

    if len(selected) < target_size:
        for candidate in sorted(candidates, key=_planner_pool_sort_key, reverse=True):
            selected.setdefault(_candidate_id(candidate), candidate)
            if len(selected) >= target_size:
                break


def _planner_pool_sort_key(candidate: Dict) -> tuple[float, float, float, float]:
    return (
        _clamp(candidate.get("highlight_score", candidate.get("editorial_score", 0.0))),
        _clamp(candidate.get("quality_score", 0.0)),
        _clamp(candidate.get("impact_capacity", 0.0)),
        _clamp(candidate.get("duration", 0.0), hi=20.0),
    )


def _build_segment_profiles(cut_times: np.ndarray, segment_durations: np.ndarray, beat_info: Dict) -> List[Dict]:
    beat_times = np.asarray(beat_info.get("times", []), dtype=float)
    energy_profile = beat_info.get("energy_profile") or {}
    rhythm_data = beat_info.get("rhythm_data") or {}
    sections = beat_info.get("sections") or []

    wave = np.asarray(energy_profile.get("wave", []), dtype=float)
    arc = np.asarray(energy_profile.get("arc", []), dtype=float)
    impact = np.asarray(rhythm_data.get("impact_strength", []), dtype=float)
    rhythm = np.asarray(rhythm_data.get("combined_strength", []), dtype=float)
    novelty = np.asarray(rhythm_data.get("novelty_strength", []), dtype=float)

    profiles: List[Dict] = []
    for i, duration in enumerate(segment_durations):
        start = float(cut_times[i])
        end = float(cut_times[i + 1])
        mid = (start + end) * 0.5
        local_wave = _interp_feature(mid, beat_times, wave, 0.5)
        local_arc = _interp_feature(mid, beat_times, arc, 0.5)
        local_impact = _interp_feature(start, beat_times, impact, 0.5)
        local_rhythm = _interp_feature(start, beat_times, rhythm, 0.5)
        local_novelty = _interp_feature(start, beat_times, novelty, 0.4)
        section = _section_at(sections, mid)
        target = _target_for_segment(section, local_wave, local_impact, local_rhythm, local_novelty, local_arc)
        impact_class = _impact_class_for_segment(target, local_wave, local_impact, local_rhythm, local_novelty, local_arc)
        profiles.append({
            "index": i,
            "start": start,
            "end": end,
            "duration": float(duration),
            "mid": mid,
            "wave": local_wave,
            "impact": local_impact,
            "rhythm": local_rhythm,
            "novelty": local_novelty,
            "arc": local_arc,
            "section": section,
            "section_type": section.get("type", "body") if section else "body",
            "target": target,
            "impact_class": impact_class,
        })
    return profiles


def _interp_feature(time_s: float, beat_times: np.ndarray, values: np.ndarray, default: float) -> float:
    if beat_times.size == 0 or values.size != beat_times.size:
        return default
    return _clamp(np.interp(time_s, beat_times, values, left=float(values[0]), right=float(values[-1])), default=default)


def _section_at(sections: Sequence[Dict], time_s: float) -> Dict:
    for section in sections:
        if float(section.get("start", 0.0)) <= time_s < float(section.get("end", 0.0)):
            return section
    return sections[-1] if sections else {}


def _target_for_segment(section: Dict, wave: float, impact: float, rhythm: float, novelty: float, arc: float) -> str:
    section_type = section.get("type", "body")
    if section_type in {"drop", "finale"} and (wave >= 0.58 or impact >= 0.55):
        return "drop"
    if impact >= 0.76 or (wave >= 0.78 and rhythm >= 0.62):
        return "drop"
    if section_type in {"breakdown", "intro", "outro"} and wave <= 0.54:
        return "soft"
    if wave <= 0.32 and impact <= 0.48:
        return "soft"
    if section_type in {"bridge", "hook"} or novelty >= 0.68 or (arc >= 0.62 and wave >= 0.48):
        return "build"
    if wave >= 0.58 and rhythm >= 0.54:
        return "rhythm"
    return "flow"


def _impact_class_for_segment(target: str, wave: float, impact: float, rhythm: float, novelty: float, arc: float) -> str:
    if target == "drop" or impact >= 0.76 or (wave >= 0.82 and rhythm >= 0.62):
        return "drop"
    if target == "build" or novelty >= 0.66 or (arc >= 0.60 and wave >= 0.50):
        return "build"
    if target == "soft" or (wave <= 0.34 and impact <= 0.50):
        return "soft"
    if target == "rhythm" or rhythm >= 0.56:
        return "rhythm"
    return "flow"


def _score_candidate(candidate: Dict, profile: Dict) -> float:
    return _score_candidate_components(candidate, profile, previous_candidate=None)["total"]


def _prepare_candidate_for_planner(candidate: Dict) -> Dict:
    prepared = dict(candidate)
    semantic = prepared.get("semantic") or {}
    quality = _clamp(prepared.get("quality_score", semantic.get("visual_quality", 0.5)), default=0.5)
    action = _clamp(prepared.get("action_score", semantic.get("action_intensity", 0.0)))
    beauty = _clamp(prepared.get("beauty_score", semantic.get("beauty_score", 0.0)))
    tension = _clamp(prepared.get("tension_score", 0.0))
    soft = _clamp(prepared.get("soft_score", 0.0))
    motion = _clamp(prepared.get("motion", semantic.get("camera_motion", 0.0)))
    contrast = _clamp(prepared.get("contrast", 0.0))
    saturation = _clamp(prepared.get("saturation", 0.0))
    brightness = _clamp(prepared.get("brightness", 0.5), default=0.5)
    text_score = _clamp(prepared.get("text_score", (prepared.get("semantic") or {}).get("visible_text", 0.0)))

    if not prepared.get("id"):
        prepared["id"] = _candidate_stable_id(prepared)
    if "impact_capacity" not in prepared:
        prepared["impact_capacity"] = _clamp(0.42 * action + 0.22 * motion + 0.18 * tension + 0.12 * contrast + 0.06 * quality)
    if "highlight_score" not in prepared:
        prepared["highlight_score"] = _clamp(
            0.34 * quality + 0.22 * max(action, beauty, tension) + 0.18 * prepared["impact_capacity"]
            + 0.14 * saturation + 0.12 * contrast
        )
    if "flow_vector" not in prepared:
        prepared["flow_vector"] = {
            "brightness": brightness,
            "saturation": saturation,
            "motion": motion,
            "action": action,
            "beauty": beauty,
            "tension": tension,
            "quality": quality,
        }
    if "identity_key" not in prepared:
        prepared["identity_key"] = f"{prepared.get('video_file')}:{prepared.get('scene_index', prepared.get('id'))}"
    if "usable_interval" not in prepared:
        prepared["usable_interval"] = {
            "start": float(prepared.get("start", 0.0) or 0.0),
            "end": float(prepared.get("end", prepared.get("start", 0.0)) or 0.0),
        }
    if "edit_roles" not in prepared:
        prepared["edit_roles"] = sorted(_candidate_roles(prepared))
    if "quality_flags" not in prepared:
        flags = []
        if brightness < 0.12:
            flags.append("dark")
        if quality < 0.24:
            flags.append("low_quality")
        if text_score >= 0.58:
            flags.append("text_overlay")
        prepared["quality_flags"] = flags
    if "text_score" not in prepared:
        prepared["text_score"] = text_score
    prepared["source_time_bucket"] = _source_time_bucket(prepared)
    return prepared


def _candidate_stable_id(candidate: Dict) -> str:
    raw = "|".join([
        str(candidate.get("video_file", "")),
        f"{float(candidate.get('start', 0.0) or 0.0):.3f}",
        f"{float(candidate.get('end', 0.0) or 0.0):.3f}",
        str(candidate.get("scene_index", "")),
    ])
    return "auto_" + hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _build_sequence_plan(candidates: Sequence[Dict], profiles: Sequence[Dict],
                         auto_speed_clip_control: bool = True) -> List[Dict]:
    if not candidates or not profiles:
        return []

    beam_width = _env_int("BEATSYNC_SEQUENCE_BEAM_WIDTH", 8, lo=2, hi=24)
    candidate_limit = _env_int("BEATSYNC_SEQUENCE_CANDIDATE_LIMIT", 48, lo=8, hi=180)
    expansion_limit = _env_int("BEATSYNC_SEQUENCE_EXPANSION_LIMIT", 16, lo=4, hi=60)
    unique_video_count = len({str(c.get("video_file")) for c in candidates if c.get("video_file")})
    candidate_count = len({str(c.get("id")) for c in candidates if c.get("id")}) or len(candidates)

    ranked_by_profile = [
        _rank_profile_candidates(candidates, profile, candidate_limit)
        for profile in profiles
    ]

    states = [{
        "score": 0.0,
        "clips": [],
        "used_ids": set(),
        "usage": Counter(),
        "recent_videos": [],
        "recent_ids": [],
        "recent_identities": [],
        "recent_time_buckets": [],
        "last_candidate": None,
        "last_identity": None,
        "last_id": None,
        "last_role": None,
        "role_streak": 0,
    }]

    for index, profile in enumerate(profiles):
        next_states: List[Dict] = []
        ranked = ranked_by_profile[index]
        for state in states:
            options = _legal_candidate_options(
                ranked=ranked,
                profile=profile,
                state=state,
                candidate_count=candidate_count,
                unique_video_count=unique_video_count,
                limit=expansion_limit,
                auto_speed_clip_control=auto_speed_clip_control,
            )
            for candidate, components in options:
                clip = _materialize_clip(
                    candidate,
                    profile,
                    index,
                    components,
                    auto_speed_clip_control=auto_speed_clip_control,
                )
                role = _primary_role(candidate, profile)
                cid = str(candidate.get("id") or "")
                video_file = str(candidate.get("video_file") or "")
                usage = Counter(state["usage"])
                if cid:
                    usage[cid] += 1
                if video_file:
                    usage[video_file] += 1
                used_ids = set(state["used_ids"])
                if cid:
                    used_ids.add(cid)
                recent_videos = list(state["recent_videos"])
                if video_file:
                    recent_videos.append(video_file)
                recent_videos = recent_videos[-5:]
                recent_ids = list(state.get("recent_ids") or [])
                if cid:
                    recent_ids.append(cid)
                recent_ids = recent_ids[-RECENT_CANDIDATE_COOLDOWN:]
                recent_identities = list(state.get("recent_identities") or [])
                identity = candidate.get("identity_key")
                if identity:
                    recent_identities.append(identity)
                recent_identities = recent_identities[-RECENT_SCENE_COOLDOWN:]
                recent_time_buckets = list(state.get("recent_time_buckets") or [])
                time_bucket = _source_time_bucket(candidate)
                if time_bucket:
                    recent_time_buckets.append(time_bucket)
                recent_time_buckets = recent_time_buckets[-RECENT_TIME_BUCKET_COOLDOWN:]
                role_streak = int(state["role_streak"]) + 1 if role == state["last_role"] else 1
                reuse_penalty = 0.75 * max(0, usage[cid] - 1) if cid else 0.0

                next_states.append({
                    "score": float(state["score"]) + float(components["total"]) - reuse_penalty,
                    "clips": list(state["clips"]) + [clip],
                    "used_ids": used_ids,
                    "usage": usage,
                    "recent_videos": recent_videos,
                    "recent_ids": recent_ids,
                    "recent_identities": recent_identities,
                    "recent_time_buckets": recent_time_buckets,
                    "last_candidate": candidate,
                    "last_identity": identity,
                    "last_id": cid,
                    "last_role": role,
                    "role_streak": role_streak,
                })

        if not next_states:
            raise UniqueFootageError(
                f'Not enough distinct usable scenes to fill all {len(profiles)} cuts without repetition '
                f'(stopped at cut {index + 1}). Add different footage or shorten the edit.'
            )
        next_states.sort(key=lambda item: item["score"], reverse=True)
        states = next_states[:beam_width]

    best = max(states, key=lambda item: item["score"])
    planned = _repair_repetition_plan(
        list(best["clips"]),
        candidates,
        profiles,
        ranked_by_profile,
        auto_speed_clip_control=auto_speed_clip_control,
    )
    validate_unique_clips(planned)
    return _annotate_clip_plan_repetition(planned)


def _rank_profile_candidates(candidates: Sequence[Dict], profile: Dict, limit: int) -> List[Dict]:
    target = str(profile.get("target", "flow"))
    impact_class = str(profile.get("impact_class", target))
    preferred = []
    fallback = []
    for candidate in candidates:
        roles = _candidate_roles(candidate)
        item = (candidate, _score_candidate_components(candidate, profile, previous_candidate=None)["total"])
        if target in roles or impact_class in roles:
            preferred.append(item)
        else:
            fallback.append(item)

    preferred.sort(key=lambda item: item[1], reverse=True)
    fallback.sort(key=lambda item: item[1], reverse=True)
    ranked = [item[0] for item in preferred]
    ranked.extend(item[0] for item in fallback)
    return ranked or [item[0] for item in fallback[:limit]]


def _legal_candidate_options(
    ranked: Sequence[Dict],
    profile: Dict,
    state: Dict,
    candidate_count: int,
    unique_video_count: int,
    limit: int,
    auto_speed_clip_control: bool = True,
) -> List[tuple[Dict, Dict]]:
    uniqueness = ClipUniquenessIndex(state.get('clips') or [])
    used_ids = set(state.get("used_ids") or set())
    scoped = [candidate for candidate in ranked if _candidate_matches_profile(candidate, profile)]
    if not scoped:
        scoped = list(ranked)
    unused_scoped_available = any(_candidate_id(candidate) not in used_ids for candidate in scoped)
    unused_any_available = any(_candidate_id(candidate) not in used_ids for candidate in ranked)
    levels = [
        {"name": "strict_target", "reuse": False, "scope": "target", "recent": True, "time": True, "video": True, "role": True},
        {"name": "strict_target_video_relaxed", "reuse": False, "scope": "target", "recent": True, "time": True, "video": False, "role": True},
        {"name": "unused_target_recent_relaxed", "reuse": False, "scope": "target", "recent": False, "time": False, "video": False, "role": True},
        {"name": "unused_any_strict", "reuse": False, "scope": "all", "recent": True, "time": True, "video": False, "role": True},
        {"name": "unused_any_relaxed", "reuse": False, "scope": "all", "recent": False, "time": False, "video": False, "role": False},
    ]
    for level in levels:
        if level["scope"] == "target" and not unused_scoped_available and not level.get("reuse"):
            continue
        if level.get("reuse") and unused_any_available:
            continue
        search_space = scoped if level["scope"] == "target" else ranked
        options: List[tuple[Dict, Dict]] = []
        for candidate in search_space:
            if not _candidate_allowed(candidate, profile, state, candidate_count, unique_video_count, level):
                continue
            if uniqueness.conflicts(candidate):
                continue
            components = _score_candidate_components(candidate, profile, state.get("last_candidate"))
            clip = _materialize_clip(candidate, profile, int(profile.get('index', 0)), components,
                                     auto_speed_clip_control=auto_speed_clip_control)
            if not _clip_fits_candidate(clip, candidate):
                continue
            if uniqueness.conflicts(clip):
                continue
            cid = _candidate_id(candidate)
            repetition_penalty = _repetition_penalty(candidate, state, level)
            if repetition_penalty > 0.0:
                components = dict(components)
                components["total"] -= repetition_penalty
                components["repetition_penalty"] = repetition_penalty
            if cid and state["usage"][cid] > 0:
                components = dict(components)
                components["total"] -= min(1.35, 0.75 * state["usage"][cid])
            components = dict(components)
            components["selection_level"] = level["name"]
            options.append((candidate, components))
        if options:
            options.sort(key=lambda item: item[1]["total"], reverse=True)
            return options[:limit]
    return []


def _candidate_allowed(candidate: Dict, profile: Dict, state: Dict,
                       candidate_count: int, unique_video_count: int,
                       level: Dict[str, bool]) -> bool:
    cid = _candidate_id(candidate)
    identity = candidate.get("identity_key")
    video_file = str(candidate.get("video_file") or "")
    used_ids = state.get("used_ids") or set()

    if cid and cid == state.get("last_id"):
        return False
    if identity and identity == state.get("last_identity"):
        return False
    if cid and cid in used_ids and not level.get("reuse"):
        return False
    if level.get("recent") and cid and cid in set(state.get("recent_ids") or []):
        return False
    if level.get("recent") and identity and identity in set(state.get("recent_identities") or []):
        return False
    time_bucket = _source_time_bucket(candidate)
    if level.get("time") and time_bucket and time_bucket in set(state.get("recent_time_buckets") or []):
        return False
    if level.get("video") and unique_video_count > 1 and video_file:
        cooldown = _same_video_cooldown(profile)
        if video_file in list(state.get("recent_videos") or [])[-cooldown:]:
            return False
    if level.get("role"):
        role = _primary_role(candidate, profile)
        if role == state.get("last_role") and int(state.get("role_streak") or 0) >= _semantic_streak_cap(profile):
            return False
    return True


def _candidate_id(candidate: Dict) -> str:
    return str(candidate.get("id") or _candidate_stable_id(candidate))


def _candidate_matches_profile(candidate: Dict, profile: Dict) -> bool:
    roles = _candidate_roles(candidate)
    target = str(profile.get("target", "flow"))
    impact_class = str(profile.get("impact_class", target))
    return target in roles or impact_class in roles


def _source_time_bucket(candidate: Dict, bucket_seconds: float = SOURCE_TIME_BUCKET_SECONDS) -> str:
    video_file = str(candidate.get("video_file") or "")
    try:
        center = float(candidate.get("center", candidate.get("peak_time", candidate.get("start", 0.0))) or 0.0)
    except Exception:
        center = 0.0
    bucket = int(max(0.0, center) // max(1.0, bucket_seconds))
    return f"{video_file}:{bucket}"


def _repetition_penalty(candidate: Dict, state: Dict, level: Dict) -> float:
    penalty = 0.0
    cid = _candidate_id(candidate)
    identity = candidate.get("identity_key")
    time_bucket = _source_time_bucket(candidate)
    usage = state.get("usage") or Counter()
    if cid and usage[cid] > 0:
        penalty += 0.75 * usage[cid]
    if cid and cid in set(state.get("recent_ids") or []):
        penalty += 0.38
    if identity and identity in set(state.get("recent_identities") or []):
        penalty += 0.44
    if time_bucket and time_bucket in set(state.get("recent_time_buckets") or []):
        penalty += 0.34
    if not level.get("recent"):
        penalty += 0.05
    return penalty


def _repair_repetition_plan(
    plan: Sequence[Dict],
    candidates: Sequence[Dict],
    profiles: Sequence[Dict],
    ranked_by_profile: Sequence[Sequence[Dict]],
    auto_speed_clip_control: bool = True,
) -> List[Dict]:
    if not plan:
        return []
    candidate_ids = {_candidate_id(candidate) for candidate in candidates}
    if len(candidate_ids) <= 1:
        return [dict(item) for item in plan]

    repaired = [dict(item) for item in plan]
    total_repairs = 0
    for _ in range(2):
        problem_indexes = _repetition_problem_indexes(repaired)
        if not problem_indexes:
            break
        changed = False
        for index in sorted(problem_indexes):
            if index >= len(repaired) or index >= len(profiles):
                continue
            replacement = _find_repetition_replacement(
                index=index,
                plan=repaired,
                ranked=ranked_by_profile[index],
                profile=profiles[index],
                auto_speed_clip_control=auto_speed_clip_control,
            )
            if not replacement:
                continue
            old_id = repaired[index].get("candidate_id")
            replacement["repetition_repaired"] = True
            replacement["replaced_candidate_id"] = old_id
            repaired[index] = replacement
            total_repairs += 1
            changed = True
        if not changed:
            break

    if total_repairs:
        for item in repaired:
            item["plan_repair_count"] = total_repairs
    return repaired


def _find_repetition_replacement(
    index: int,
    plan: Sequence[Dict],
    ranked: Sequence[Dict],
    profile: Dict,
    auto_speed_clip_control: bool,
    excluded_ids=(),
) -> Dict | None:
    used_elsewhere = {
        str(item.get("candidate_id"))
        for pos, item in enumerate(plan)
        if pos != index and item.get("candidate_id")
    }
    current_id = str(plan[index].get("candidate_id") or "") if index < len(plan) else ""
    uniqueness = ClipUniquenessIndex(item for pos, item in enumerate(plan) if pos != index)
    state = _state_from_clips(plan[:index])
    passes = [
        {"profile": True, "recent": True, "time": True},
        {"profile": True, "recent": False, "time": False},
        {"profile": False, "recent": True, "time": True},
        {"profile": False, "recent": False, "time": False},
    ]
    for repair_pass in passes:
        best: tuple[float, Dict, Dict] | None = None
        for candidate in ranked:
            cid = _candidate_id(candidate)
            if cid in excluded_ids:
                continue
            if cid == current_id:
                continue
            if cid in used_elsewhere:
                continue
            if repair_pass["profile"] and not _candidate_matches_profile(candidate, profile):
                continue
            if _candidate_conflicts_with_neighbors(candidate, plan, index):
                continue
            if repair_pass["recent"] and _candidate_has_recent_conflict(candidate, state):
                continue
            if repair_pass["time"] and _source_time_bucket(candidate) in set(state.get("recent_time_buckets") or []):
                continue
            components = _score_candidate_components(candidate, profile, state.get("last_candidate"))
            replacement = _materialize_clip(candidate, profile, index, components,
                                            auto_speed_clip_control=auto_speed_clip_control)
            if not _clip_fits_candidate(replacement, candidate):
                continue
            if uniqueness.conflicts(replacement):
                continue
            penalty = _repetition_penalty(candidate, state, {"recent": repair_pass["recent"]})
            score = float(components["total"]) - penalty
            if best is None or score > best[0]:
                adjusted = dict(components)
                adjusted["total"] = score
                adjusted["selection_level"] = "repetition_repair"
                adjusted["repetition_penalty"] = penalty
                best = (score, candidate, adjusted)
        if best is not None:
            return _materialize_clip(
                best[1],
                profile,
                index,
                best[2],
                auto_speed_clip_control=auto_speed_clip_control,
            )
    return None


def _candidate_conflicts_with_neighbors(candidate: Dict, plan: Sequence[Dict], index: int) -> bool:
    cid = _candidate_id(candidate)
    identity = candidate.get("identity_key")
    for neighbor_index in (index - 1, index + 1):
        if neighbor_index < 0 or neighbor_index >= len(plan):
            continue
        neighbor = plan[neighbor_index]
        if cid and cid == str(neighbor.get("candidate_id") or ""):
            return True
        if identity and identity == neighbor.get("identity_key"):
            return True
    return False


def _candidate_has_recent_conflict(candidate: Dict, state: Dict) -> bool:
    cid = _candidate_id(candidate)
    identity = candidate.get("identity_key")
    time_bucket = _source_time_bucket(candidate)
    return (
        (cid and cid in set(state.get("recent_ids") or []))
        or (identity and identity in set(state.get("recent_identities") or []))
        or (time_bucket and time_bucket in set(state.get("recent_time_buckets") or []))
    )


def _state_from_clips(clips: Sequence[Dict]) -> Dict:
    state = {
        "used_ids": set(),
        "usage": Counter(),
        "recent_videos": [],
        "recent_ids": [],
        "recent_identities": [],
        "recent_time_buckets": [],
        "last_candidate": None,
        "last_identity": None,
        "last_id": None,
        "last_role": None,
        "role_streak": 0,
    }
    for clip in clips:
        cid = str(clip.get("candidate_id") or "")
        identity = clip.get("identity_key")
        video_file = str(clip.get("video_file") or "")
        time_bucket = _clip_time_bucket(clip)
        role = str(clip.get("target", clip.get("impact_class", "flow")))
        if cid:
            state["used_ids"].add(cid)
            state["usage"][cid] += 1
            state["recent_ids"].append(cid)
            state["recent_ids"] = state["recent_ids"][-RECENT_CANDIDATE_COOLDOWN:]
        if video_file:
            state["usage"][video_file] += 1
            state["recent_videos"].append(video_file)
            state["recent_videos"] = state["recent_videos"][-5:]
        if identity:
            state["recent_identities"].append(identity)
            state["recent_identities"] = state["recent_identities"][-RECENT_SCENE_COOLDOWN:]
        if time_bucket:
            state["recent_time_buckets"].append(time_bucket)
            state["recent_time_buckets"] = state["recent_time_buckets"][-RECENT_TIME_BUCKET_COOLDOWN:]
        state["last_candidate"] = clip
        state["last_identity"] = identity
        state["last_id"] = cid
        state["role_streak"] = int(state["role_streak"]) + 1 if role == state["last_role"] else 1
        state["last_role"] = role
    return state


def _clip_time_bucket(clip: Dict) -> str:
    existing = clip.get("source_time_bucket")
    if existing:
        return str(existing)
    video_file = str(clip.get("video_file") or "")
    try:
        start = float(clip.get("start_time", clip.get("audio_start", 0.0)) or 0.0)
    except Exception:
        start = 0.0
    bucket = int(max(0.0, start) // SOURCE_TIME_BUCKET_SECONDS)
    return f"{video_file}:{bucket}"


def _repetition_problem_indexes(plan: Sequence[Dict]) -> set[int]:
    problem = set(_motif_repeated_indexes(plan))
    uniqueness = ClipUniquenessIndex()
    seen_ids: set[str] = set()
    recent_ids: List[str] = []
    recent_identities: List[str] = []
    recent_time_buckets: List[str] = []
    for index, clip in enumerate(plan):
        if uniqueness.conflicts(clip):
            problem.add(index)
        uniqueness.add(dict(clip, index=index))
        cid = str(clip.get("candidate_id") or "")
        identity = clip.get("identity_key")
        time_bucket = _clip_time_bucket(clip)
        if index > 0 and cid and (cid in seen_ids or cid in recent_ids):
            problem.add(index)
        if index > 0 and identity and identity in recent_identities:
            problem.add(index)
        if index > 0 and time_bucket and time_bucket in recent_time_buckets:
            problem.add(index)
        if cid:
            seen_ids.add(cid)
            recent_ids.append(cid)
            recent_ids = recent_ids[-RECENT_CANDIDATE_COOLDOWN:]
        if identity:
            recent_identities.append(identity)
            recent_identities = recent_identities[-RECENT_SCENE_COOLDOWN:]
        if time_bucket:
            recent_time_buckets.append(time_bucket)
            recent_time_buckets = recent_time_buckets[-RECENT_TIME_BUCKET_COOLDOWN:]
    return problem


def _annotate_clip_plan_repetition(plan: Sequence[Dict]) -> List[Dict]:
    annotated = [dict(item) for item in plan]
    uniqueness = ClipUniquenessIndex()
    candidate_counts = Counter(str(item.get("candidate_id")) for item in annotated if item.get("candidate_id"))
    motif_indexes = set(_motif_repeated_indexes(annotated))
    recent_ids: List[str] = []
    recent_identities: List[str] = []
    recent_time_buckets: List[str] = []
    for index, clip in enumerate(annotated):
        flags = list(uniqueness.conflicts(clip))
        uniqueness.add(dict(clip, index=index))
        cid = str(clip.get("candidate_id") or "")
        identity = clip.get("identity_key")
        time_bucket = _clip_time_bucket(clip)
        clip["source_time_bucket"] = time_bucket
        if cid and candidate_counts[cid] > 1:
            flags.append("candidate_reused")
        if cid and cid in recent_ids:
            flags.append("recent_candidate_repeat")
        if identity and identity in recent_identities:
            flags.append("recent_scene_repeat")
        if time_bucket and time_bucket in recent_time_buckets:
            flags.append("recent_time_bucket_repeat")
        if index in motif_indexes:
            flags.append("repeated_motif")
        clip["repetition_flags"] = sorted(set(flags))
        if cid:
            recent_ids.append(cid)
            recent_ids = recent_ids[-RECENT_CANDIDATE_COOLDOWN:]
        if identity:
            recent_identities.append(identity)
            recent_identities = recent_identities[-RECENT_SCENE_COOLDOWN:]
        if time_bucket:
            recent_time_buckets.append(time_bucket)
            recent_time_buckets = recent_time_buckets[-RECENT_TIME_BUCKET_COOLDOWN:]
    return annotated


def audit_clip_plan_repetition(plan: Sequence[Dict]) -> Dict:
    if not plan:
        return {
            "repeated_motif_count": 0,
            "repeated_motifs": [],
            "recent_candidate_repeat_count": 0,
            "recent_scene_repeat_count": 0,
            "scene_time_bucket_repetition_count": 0,
            "reused_candidate_ids": [],
            "global_scene_repetition_count": 0,
            "source_overlap_count": 0,
            "source_proximity_count": 0,
            "visual_duplicate_count": 0,
        }
    annotated = _annotate_clip_plan_repetition(plan)
    candidate_counts = Counter(str(item.get("candidate_id")) for item in annotated if item.get("candidate_id"))
    motifs = _detect_repeated_motifs(annotated)
    return {
        "repeated_motif_count": len(motifs),
        "repeated_motifs": motifs[:12],
        "recent_candidate_repeat_count": sum(
            1 for item in annotated if "recent_candidate_repeat" in item.get("repetition_flags", [])
        ),
        "recent_scene_repeat_count": sum(
            1 for item in annotated if "recent_scene_repeat" in item.get("repetition_flags", [])
        ),
        "scene_time_bucket_repetition_count": sum(
            1 for item in annotated if "recent_time_bucket_repeat" in item.get("repetition_flags", [])
        ),
        "reused_candidate_ids": sorted(
            cid for cid, count in candidate_counts.items() if cid and count > 1
        )[:40],
        "global_scene_repetition_count": sum('source_scene_reused' in item['repetition_flags'] for item in annotated),
        "source_overlap_count": sum('source_footage_overlap' in item['repetition_flags'] for item in annotated),
        "source_proximity_count": sum('source_proximity' in item['repetition_flags'] for item in annotated),
        "visual_duplicate_count": sum('visual_duplicate' in item['repetition_flags'] for item in annotated),
    }


def _detect_repeated_motifs(plan: Sequence[Dict]) -> List[Dict]:
    ids = [str(item.get("candidate_id") or "") for item in plan]
    ids = [cid if cid else f"missing_{index}" for index, cid in enumerate(ids)]
    motifs: List[Dict] = []
    max_size = min(8, len(ids) // 2)
    for size in range(2, max_size + 1):
        index = 0
        while index + size * 2 <= len(ids):
            pattern = ids[index:index + size]
            if len(set(pattern)) <= 1:
                index += 1
                continue
            repeats = 1
            cursor = index + size
            while cursor + size <= len(ids) and ids[cursor:cursor + size] == pattern:
                repeats += 1
                cursor += size
            if repeats >= 2:
                motifs.append({
                    "start_index": index,
                    "length": size,
                    "repeats": repeats,
                    "pattern": pattern[:8],
                })
                index = cursor
            else:
                index += 1
    return motifs


def _motif_repeated_indexes(plan: Sequence[Dict]) -> set[int]:
    indexes: set[int] = set()
    for motif in _detect_repeated_motifs(plan):
        start = int(motif.get("start_index", 0))
        length = int(motif.get("length", 0))
        repeats = int(motif.get("repeats", 0))
        for repeat_index in range(1, repeats):
            block_start = start + repeat_index * length
            indexes.update(range(block_start, block_start + length))
    return indexes


def _same_video_cooldown(profile: Dict) -> int:
    target = str(profile.get("target", "flow"))
    if target in {"drop", "rhythm"}:
        return 3
    if target == "build":
        return 4
    return 5


def _semantic_streak_cap(profile: Dict) -> int:
    target = str(profile.get("target", "flow"))
    if target in {"drop", "build"}:
        return 5
    return 3


def _candidate_roles(candidate: Dict) -> set[str]:
    roles = {str(role).lower() for role in candidate.get("edit_roles", []) if role}
    tags = {str(tag).lower() for tag in candidate.get("tags", []) if tag}
    semantic = candidate.get("semantic") or {}
    rec = str(semantic.get("recommended_use", "")).lower()
    if rec:
        roles.add("flow" if rec == "filler" else rec)
    if tags.intersection({"action", "combat", "chase", "explosion", "hype"}):
        roles.update({"drop", "rhythm"})
    if tags.intersection({"tension", "transition"}):
        roles.add("build")
    if tags.intersection({"soft", "beauty", "sad"}):
        roles.add("soft")
    if not roles:
        roles.add("flow")
    return roles


def _primary_role(candidate: Dict, profile: Dict) -> str:
    target = str(profile.get("target", "flow"))
    impact_class = str(profile.get("impact_class", target))
    roles = _candidate_roles(candidate)
    if target in roles:
        return target
    if impact_class in roles:
        return impact_class
    for role in ("drop", "build", "rhythm", "soft", "flow"):
        if role in roles:
            return role
    return "flow"


def _score_candidate_components(candidate: Dict, profile: Dict, previous_candidate: Dict | None = None) -> Dict:
    target = profile.get("target", "flow")
    impact_class = profile.get("impact_class", target)
    semantic = candidate.get("semantic") or {}
    tags = {str(t).lower() for t in candidate.get("tags", [])}
    quality = _clamp(candidate.get("quality_score", semantic.get("visual_quality", 0.5)), default=0.5)
    action = _clamp(candidate.get("action_score", semantic.get("action_intensity", 0.0)))
    beauty = _clamp(candidate.get("beauty_score", semantic.get("beauty_score", 0.0)))
    tension = _clamp(candidate.get("tension_score", 0.0))
    soft = _clamp(candidate.get("soft_score", 0.0))
    motion = _clamp(candidate.get("motion", semantic.get("camera_motion", 0.0)))
    character = _clamp(semantic.get("character_focus", 0.0))
    combat = _clamp(semantic.get("combat", 0.0))
    chase = _clamp(semantic.get("chase", 0.0))
    explosion = _clamp(semantic.get("explosion", 0.0))
    highlight = _clamp(candidate.get("highlight_score", candidate.get("editorial_score", quality)), default=quality)
    impact_capacity = _clamp(candidate.get("impact_capacity", max(action, motion, tension)))

    semantic_match = _semantic_match_score(candidate, profile, tags, semantic)
    desired_impact = _desired_visual_impact(profile)
    impact_match = _clamp(1.0 - abs(impact_capacity - desired_impact) * 1.45)
    if impact_class == "drop":
        impact_match = _clamp(0.58 * impact_match + 0.42 * max(action, motion, combat, chase, explosion))
    elif impact_class == "soft":
        impact_match = _clamp(0.56 * impact_match + 0.28 * beauty + 0.16 * soft - 0.20 * action)

    brightness = _clamp(candidate.get("brightness", 0.5), default=0.5)
    flow_score = _flow_score(previous_candidate, candidate, profile)
    quality_penalty = _quality_penalty(candidate, brightness, quality)
    duration_penalty = _duration_penalty(candidate, profile)

    total = (
        0.32 * highlight
        + 0.27 * impact_match
        + 0.20 * semantic_match
        + 0.15 * flow_score
        + 0.06 * quality
        - quality_penalty
        - duration_penalty
    )
    rng = _stable_rng(profile.get("index"), candidate.get("id"), target)
    total += rng.random() * 0.004

    return {
        "total": _clamp(total, lo=-1.0, hi=2.0),
        "match_score": _clamp(0.55 * impact_match + 0.45 * semantic_match),
        "impact_match": impact_match,
        "semantic_match": semantic_match,
        "flow_score": flow_score,
        "highlight_score": highlight,
        "quality_penalty": quality_penalty,
        "duration_penalty": duration_penalty,
    }


def _semantic_match_score(candidate: Dict, profile: Dict, tags: set[str], semantic: Dict) -> float:
    target = str(profile.get("target", "flow"))
    roles = _candidate_roles(candidate)
    quality = _clamp(candidate.get("quality_score", semantic.get("visual_quality", 0.5)), default=0.5)
    action = _clamp(candidate.get("action_score", semantic.get("action_intensity", 0.0)))
    beauty = _clamp(candidate.get("beauty_score", semantic.get("beauty_score", 0.0)))
    tension = _clamp(candidate.get("tension_score", 0.0))
    soft = _clamp(candidate.get("soft_score", 0.0))
    motion = _clamp(candidate.get("motion", semantic.get("camera_motion", 0.0)))
    character = _clamp(semantic.get("character_focus", 0.0))
    combat = _clamp(semantic.get("combat", 0.0))
    chase = _clamp(semantic.get("chase", 0.0))
    explosion = _clamp(semantic.get("explosion", 0.0))

    role_bonus = 0.18 if target in roles else 0.0
    if target == "drop":
        base = 0.40 * action + 0.18 * motion + 0.12 * combat + 0.10 * chase + 0.08 * explosion + 0.12 * quality
        tag_bonus = 0.10 if tags.intersection({"action", "combat", "chase", "explosion", "hype"}) else 0.0
    elif target == "soft":
        base = 0.42 * beauty + 0.19 * soft + 0.15 * character + 0.14 * (1.0 - action) + 0.10 * quality
        tag_bonus = 0.10 if tags.intersection({"soft", "beauty", "sad"}) else 0.0
    elif target == "build":
        base = 0.36 * tension + 0.18 * motion + 0.16 * character + 0.15 * action + 0.15 * quality
        tag_bonus = 0.10 if tags.intersection({"tension", "transition"}) else 0.0
    elif target == "rhythm":
        base = 0.28 * action + 0.24 * motion + 0.20 * quality + 0.16 * tension + 0.12 * beauty
        tag_bonus = 0.07 if tags.intersection({"action", "flow", "clean"}) else 0.0
    else:
        base = 0.30 * quality + 0.22 * beauty + 0.18 * action + 0.16 * tension + 0.14 * soft
        tag_bonus = 0.06 if tags.intersection({"flow", "clean", "beauty"}) else 0.0
    return _clamp(base + role_bonus + tag_bonus)


def _desired_visual_impact(profile: Dict) -> float:
    target = str(profile.get("impact_class", profile.get("target", "flow")))
    audio_impact = _clamp(profile.get("impact", 0.5), default=0.5)
    wave = _clamp(profile.get("wave", 0.5), default=0.5)
    if target == "drop":
        return _clamp(max(0.74, 0.55 * audio_impact + 0.45 * wave))
    if target == "build":
        return _clamp(max(0.56, 0.48 * audio_impact + 0.52 * wave))
    if target == "rhythm":
        return _clamp(max(0.48, 0.62 * audio_impact + 0.38 * wave))
    if target == "soft":
        return _clamp(min(0.36, 0.35 * audio_impact + 0.25 * wave))
    return _clamp(0.42 * audio_impact + 0.38 * wave + 0.20 * 0.5)


def _flow_score(previous_candidate: Dict | None, candidate: Dict, profile: Dict) -> float:
    if not previous_candidate:
        return 0.74

    prev = _flow_vector(previous_candidate)
    curr = _flow_vector(candidate)
    brightness_delta = abs(curr["brightness"] - prev["brightness"])
    saturation_delta = abs(curr["saturation"] - prev["saturation"])
    motion_delta = abs(curr["motion"] - prev["motion"])
    action_delta = abs(curr["action"] - prev["action"])
    target = str(profile.get("target", "flow"))

    if target == "drop":
        motion_lift = max(0.0, curr["motion"] - prev["motion"])
        action_lift = max(0.0, curr["action"] - prev["action"])
        return _clamp(0.58 + 0.20 * motion_lift + 0.16 * action_lift - 0.16 * brightness_delta - 0.08 * saturation_delta)
    if target == "build":
        motion_lift = max(0.0, curr["motion"] - prev["motion"])
        return _clamp(0.78 + 0.12 * motion_lift - 0.30 * brightness_delta - 0.16 * saturation_delta - 0.10 * action_delta)
    if target == "soft":
        return _clamp(0.88 - 0.42 * brightness_delta - 0.24 * motion_delta - 0.16 * saturation_delta - 0.12 * action_delta)
    return _clamp(0.86 - 0.36 * brightness_delta - 0.28 * motion_delta - 0.20 * saturation_delta - 0.10 * action_delta)


def _flow_vector(candidate: Dict) -> Dict[str, float]:
    vector = candidate.get("flow_vector") or {}
    semantic = candidate.get("semantic") or {}
    return {
        "brightness": _clamp(vector.get("brightness", candidate.get("brightness", 0.5)), default=0.5),
        "saturation": _clamp(vector.get("saturation", candidate.get("saturation", 0.5)), default=0.5),
        "motion": _clamp(vector.get("motion", candidate.get("motion", semantic.get("camera_motion", 0.0)))),
        "action": _clamp(vector.get("action", candidate.get("action_score", semantic.get("action_intensity", 0.0)))),
        "beauty": _clamp(vector.get("beauty", candidate.get("beauty_score", semantic.get("beauty_score", 0.0)))),
        "tension": _clamp(vector.get("tension", candidate.get("tension_score", 0.0))),
        "quality": _clamp(vector.get("quality", candidate.get("quality_score", semantic.get("visual_quality", 0.5))), default=0.5),
    }


def _quality_penalty(candidate: Dict, brightness: float, quality: float) -> float:
    flags = {str(flag) for flag in candidate.get("quality_flags", [])}
    penalty = 0.0
    if brightness < 0.13 or "dark" in flags:
        penalty += 0.26
    if "blown" in flags:
        penalty += 0.22
    if "blurry" in flags:
        penalty += 0.22
    if quality < 0.24 or "low_quality" in flags:
        penalty += 0.20
    if "weak_highlight" in flags:
        penalty += 0.12
    return penalty


def _duration_penalty(candidate: Dict, profile: Dict) -> float:
    required_source = max(0.05, float(profile.get("duration", 0.05)))
    candidate_duration = max(0.05, float(candidate.get("duration", required_source)))
    if candidate_duration >= required_source * 0.80:
        return 0.0
    if candidate_duration >= required_source * 0.55:
        return 0.08
    return 0.18


def _speed_decision(candidate: Dict, profile: Dict, components: Dict,
                    auto_speed_clip_control: bool) -> Dict:
    final_duration = max(0.05, float(profile.get("duration", 0.05)))
    if not auto_speed_clip_control:
        return {"speed": 1.0, "confidence": 0.0, "reason": "disabled"}

    quality_flags = {str(flag) for flag in candidate.get("quality_flags", [])}
    if quality_flags.intersection({"dark", "blown", "blurry", "low_quality", "weak_highlight"}):
        return {"speed": 1.0, "confidence": 0.0, "reason": "quality_guard"}

    highlight = _clamp(candidate.get("highlight_score", components.get("highlight_score", 0.0)))
    match = _clamp(components.get("match_score", 0.0))
    flow = _clamp(components.get("flow_score", 0.0))
    confidence = _clamp(0.42 * highlight + 0.38 * match + 0.20 * flow)
    if confidence < 0.64:
        return {"speed": 1.0, "confidence": confidence, "reason": "low_confidence"}

    target = str(profile.get("target", "flow"))
    impact_class = str(profile.get("impact_class", target))
    wave = _clamp(profile.get("wave", 0.5), default=0.5)
    impact = _clamp(profile.get("impact", 0.5), default=0.5)
    motion = _clamp(candidate.get("motion", 0.0))
    action = _clamp(candidate.get("action_score", 0.0))
    beauty = _clamp(candidate.get("beauty_score", 0.0))
    soft = _clamp(candidate.get("soft_score", 0.0))
    tension = _clamp(candidate.get("tension_score", 0.0))

    speed = 1.0
    reason = "original"
    if impact_class == "drop" and highlight >= 0.72 and match >= 0.66 and max(action, motion) >= 0.62:
        if impact >= 0.88 and wave >= 0.82 and highlight >= 0.82 and max(action, motion) >= 0.78:
            speed = 0.75
            reason = "major_drop_slowdown"
        else:
            speed = 0.85
            reason = "drop_highlight_slowdown"
    elif impact_class == "soft" and highlight >= 0.70 and match >= 0.66 and max(beauty, soft) >= 0.68:
        speed = 0.88
        reason = "emotional_soft_slowdown"
    elif target in {"build", "rhythm"} and highlight >= 0.66 and match >= 0.62 and flow >= 0.54:
        if target == "build" and tension >= 0.66 and wave >= 0.62 and motion >= 0.58:
            speed = 1.20
            reason = "build_momentum_speedup"
        elif max(action, motion) >= 0.58:
            speed = 1.15
            reason = "rhythm_momentum_speedup"
    elif impact_class == "build" and highlight >= 0.74 and tension >= 0.72 and motion >= 0.66 and flow >= 0.58:
        speed = 1.35
        reason = "rare_transition_speedup"

    if abs(speed - 1.0) <= 0.02:
        return {"speed": 1.0, "confidence": confidence, "reason": reason}

    if not _has_source_room_for_speed(candidate, final_duration, speed):
        return {"speed": 1.0, "confidence": confidence, "reason": "source_boundary_guard"}
    if flow < 0.48 and speed > 1.0:
        return {"speed": 1.0, "confidence": confidence, "reason": "flow_guard"}

    return {"speed": speed, "confidence": confidence, "reason": reason}


def _has_source_room_for_speed(candidate: Dict, final_duration: float, speed: float) -> bool:
    requested = max(0.05, final_duration * speed)
    candidate_duration = max(0.0, float(candidate.get("duration", 0.0) or 0.0))
    usable = candidate.get("usable_interval") or {}
    usable_start = float(usable.get("start", candidate.get("start", 0.0)) or 0.0)
    usable_end = float(usable.get("end", candidate.get("end", usable_start)) or usable_start)
    usable_duration = max(0.0, usable_end - usable_start)
    available = usable_duration if usable_end > usable_start else candidate_duration
    if available <= 0.0:
        video_duration = max(0.0, float(candidate.get("video_duration", 0.0) or 0.0))
        start = max(0.0, float(candidate.get("start", 0.0) or 0.0))
        available = max(0.0, video_duration - start)
    return available + 1e-6 >= requested


def _clip_fits_candidate(clip: Dict, candidate: Dict) -> bool:
    """A long cut must stay inside the analyzed usable moment and source file."""
    usable = candidate.get('usable_interval') or {}
    start = float(usable.get('start', candidate.get('start', 0)) or 0)
    end = float(usable.get('end', candidate.get('end', start)) or start)
    clip_start = float(clip['start_time'])
    clip_end = clip_start + float(clip['source_duration'])
    source_end = float(candidate.get('video_duration', end) or end)
    return (all(np.isfinite(v) for v in (start, end, clip_start, clip_end, source_end))
            and clip_start >= start - 1e-6 and clip_end <= min(end, source_end) + 1e-6)


def _materialize_clip(candidate: Dict, profile: Dict, index: int, components: Dict | None = None,
                      auto_speed_clip_control: bool = True) -> Dict:
    components = components or _score_candidate_components(candidate, profile, previous_candidate=None)
    final_duration = max(0.05, float(profile["duration"]))
    target = profile.get("target", "flow")
    impact_class = profile.get("impact_class", target)
    speed_info = _speed_decision(candidate, profile, components, auto_speed_clip_control)
    playback_speed = _clamp(speed_info.get("speed", 1.0), lo=0.25, hi=3.0, default=1.0)
    source_duration = max(0.05, final_duration * playback_speed)
    video_duration = max(source_duration, float(candidate.get("video_duration", source_duration)))

    if target == "drop":
        anchor = float(candidate.get("peak_time", candidate.get("center", candidate.get("start", 0.0))))
        align = 0.22
    elif target == "soft":
        anchor = float(candidate.get("center", candidate.get("start", 0.0)))
        align = 0.50
    elif target == "build":
        anchor = float(candidate.get("peak_time", candidate.get("center", candidate.get("start", 0.0))))
        align = 0.62
    elif target == "rhythm":
        anchor = float(candidate.get("peak_time", candidate.get("center", candidate.get("start", 0.0))))
        align = 0.46
    else:
        anchor = float(candidate.get("center", candidate.get("start", 0.0)))
        align = 0.50

    start_time = anchor - source_duration * align
    usable = candidate.get("usable_interval") or {}
    usable_start = max(0.0, float(usable.get("start", candidate.get("start", 0.0)) or 0.0))
    usable_end = max(usable_start, float(usable.get("end", candidate.get("end", usable_start)) or usable_start))
    if usable_end - usable_start >= source_duration:
        start_time = max(usable_start, min(start_time, usable_end - source_duration))
    start_time = max(0.0, min(start_time, max(0.0, video_duration - source_duration)))
    quality_flags = list(candidate.get("quality_flags", []))
    low_quality = bool({str(flag) for flag in quality_flags}.intersection({"dark", "blown", "blurry", "low_quality"}))
    text_overlay = _candidate_has_text_overlay(candidate)

    return {
        "index": index,
        "video_file": candidate.get("video_file"),
        "source_name": candidate.get("source_name"),
        "start_time": start_time,
        "source_duration": source_duration,
        "final_duration": final_duration,
        "playback_speed": playback_speed,
        "speed_reason": speed_info.get("reason", "original"),
        "speed_confidence": speed_info.get("confidence", 0.0),
        "target": target,
        "impact_class": impact_class,
        "score": components.get("total", _score_candidate(candidate, profile)),
        "match_score": components.get("match_score"),
        "flow_score": components.get("flow_score"),
        "highlight_score": components.get("highlight_score", candidate.get("highlight_score")),
        "selection_level": components.get("selection_level", "unknown"),
        "repetition_penalty": components.get("repetition_penalty", 0.0),
        "candidate_id": candidate.get("id"),
        "identity_key": candidate.get("identity_key"),
        "scene_index": candidate.get("scene_index"),
        "visual_fingerprints": candidate.get("visual_fingerprints") or [],
        "source_guard_seconds": SOURCE_GUARD_SECONDS,
        "usable_source_start": usable_start,
        "usable_source_end": usable_end,
        "source_time_bucket": _source_time_bucket(candidate),
        "quality_flags": quality_flags,
        "low_quality": low_quality,
        "text_score": candidate.get("text_score", 0.0),
        "text_overlay": text_overlay,
        "tags": list(candidate.get("tags", [])),
        "ai_analyzed": bool(candidate.get("ai_analyzed")),
        "audio_start": profile.get("start"),
        "audio_end": profile.get("end"),
        "wave": profile.get("wave"),
        "impact": profile.get("impact"),
        "impact_capacity": candidate.get("impact_capacity"),
    }
