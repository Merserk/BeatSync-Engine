#!/usr/bin/env python3
"""Audio-visual clip planner for Auto Mode."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, deque
from typing import Dict, List, Sequence

import numpy as np


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


def build_planned_clip_sequence(
    cut_times: Sequence[float],
    segment_durations: Sequence[float],
    beat_info: Dict | None,
    video_files: Sequence[str],
) -> List[Dict]:
    """Build exact source clip choices for every output segment.

    Returns an empty list when no visual library is present, which tells the
    renderer to keep its old fallback sampling.
    """
    beat_info = beat_info or {}
    video_analysis = beat_info.get("video_analysis") or {}
    candidates = list(video_analysis.get("candidates") or [])
    candidates = [c for c in candidates if c.get("video_file")]
    if not candidates:
        return []

    cut_times_arr = np.asarray(cut_times, dtype=float)
    durations_arr = np.asarray(segment_durations, dtype=float)
    if cut_times_arr.size < 2 or durations_arr.size == 0:
        return []

    profiles = _build_segment_profiles(cut_times_arr, durations_arr, beat_info)
    recent_ids = deque(maxlen=10)
    recent_videos = deque(maxlen=5)
    usage = Counter()
    planned: List[Dict] = []

    for i, profile in enumerate(profiles):
        candidate = _choose_candidate(
            candidates=candidates,
            profile=profile,
            recent_ids=recent_ids,
            recent_videos=recent_videos,
            usage=usage,
            index=i,
        )
        if not candidate:
            continue
        planned_clip = _materialize_clip(
            candidate=candidate,
            profile=profile,
            index=i,
        )
        planned.append(planned_clip)
        recent_ids.append(candidate.get("id"))
        recent_videos.append(candidate.get("video_file"))
        usage[candidate.get("id")] += 1
        usage[candidate.get("video_file")] += 1

    if len(planned) != len(durations_arr):
        return []
    return planned


def summarize_clip_plan(plan: Sequence[Dict]) -> Dict:
    if not plan:
        return {"clip_count": 0, "targets": {}, "ai_tagged": 0}
    targets = Counter(str(item.get("target", "flow")) for item in plan)
    ai_tagged = sum(1 for item in plan if item.get("ai_analyzed"))
    source_count = len(set(item.get("video_file") for item in plan))
    return {
        "clip_count": len(plan),
        "targets": dict(targets),
        "ai_tagged": ai_tagged,
        "source_count": source_count,
    }


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


def _choose_candidate(
    candidates: Sequence[Dict],
    profile: Dict,
    recent_ids: deque,
    recent_videos: deque,
    usage: Counter,
    index: int,
) -> Dict | None:
    best_candidate = None
    best_score = -999.0
    rng = _stable_rng(index, profile.get("target"), profile.get("start"))

    for candidate in candidates:
        score = _score_candidate(candidate, profile)
        cid = candidate.get("id")
        video_file = candidate.get("video_file")

        if cid in recent_ids:
            score -= 0.28
        if video_file in recent_videos:
            score -= 0.10
        score -= min(0.28, usage[cid] * 0.10)
        score -= min(0.18, usage[video_file] * 0.012)

        required_source = max(0.05, profile["duration"])
        candidate_duration = max(0.05, float(candidate.get("duration", required_source)))
        if candidate_duration < required_source * 0.55:
            score -= 0.18

        score += rng.random() * 0.015
        if score > best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate


def _score_candidate(candidate: Dict, profile: Dict) -> float:
    target = profile.get("target", "flow")
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

    tag_bonus = 0.0
    if target in tags:
        tag_bonus += 0.08
    if target == "drop" and tags.intersection({"action", "combat", "chase", "explosion", "hype"}):
        tag_bonus += 0.12
    if target == "soft" and tags.intersection({"soft", "beauty", "sad"}):
        tag_bonus += 0.10
    if target == "build" and tags.intersection({"tension", "transition"}):
        tag_bonus += 0.10

    if target == "drop":
        match = 0.46 * action + 0.16 * motion + 0.12 * combat + 0.10 * chase + 0.08 * explosion + 0.08 * quality
    elif target == "soft":
        match = 0.45 * beauty + 0.18 * soft + 0.13 * character + 0.14 * (1.0 - action) + 0.10 * quality
    elif target == "build":
        match = 0.40 * tension + 0.18 * motion + 0.15 * character + 0.14 * action + 0.13 * quality
    elif target == "rhythm":
        match = 0.30 * action + 0.24 * motion + 0.18 * quality + 0.16 * tension + 0.12 * beauty
    else:
        match = 0.28 * quality + 0.24 * beauty + 0.20 * action + 0.16 * tension + 0.12 * soft

    brightness = _clamp(candidate.get("brightness", 0.5), default=0.5)
    visibility_penalty = 0.0
    if brightness < 0.13:
        visibility_penalty += 0.18
    if quality < 0.24:
        visibility_penalty += 0.16

    return _clamp(match + tag_bonus + 0.12 * quality - visibility_penalty, lo=-1.0, hi=2.0)


def _materialize_clip(candidate: Dict, profile: Dict, index: int) -> Dict:
    final_duration = max(0.05, float(profile["duration"]))
    source_duration = final_duration
    video_duration = max(source_duration, float(candidate.get("video_duration", source_duration)))
    target = profile.get("target", "flow")

    if target == "drop":
        anchor = float(candidate.get("peak_time", candidate.get("center", candidate.get("start", 0.0))))
        align = 0.36
    elif target == "soft":
        anchor = float(candidate.get("center", candidate.get("start", 0.0)))
        align = 0.50
    elif target == "build":
        anchor = float(candidate.get("peak_time", candidate.get("center", candidate.get("start", 0.0))))
        align = 0.48
    else:
        anchor = float(candidate.get("center", candidate.get("start", 0.0)))
        align = 0.44

    start_time = anchor - source_duration * align
    start_time = max(0.0, min(start_time, max(0.0, video_duration - source_duration)))

    return {
        "index": index,
        "video_file": candidate.get("video_file"),
        "source_name": candidate.get("source_name"),
        "start_time": start_time,
        "source_duration": source_duration,
        "final_duration": final_duration,
        "target": target,
        "score": _score_candidate(candidate, profile),
        "candidate_id": candidate.get("id"),
        "tags": list(candidate.get("tags", [])),
        "ai_analyzed": bool(candidate.get("ai_analyzed")),
        "audio_start": profile.get("start"),
        "audio_end": profile.get("end"),
        "wave": profile.get("wave"),
        "impact": profile.get("impact"),
    }
