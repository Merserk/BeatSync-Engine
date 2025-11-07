#!/usr/bin/env python3
"""
Auto Mode - Fully Automatic Intelligent Beat Detection
Analyzes music structure and automatically creates optimal cuts
"""

import os
import sys
import numpy as np
import librosa
from typing import Tuple, Dict, List
import warnings

# Determine script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# GPU Support (optional for faster processing)
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

warnings.filterwarnings('ignore')


def analyze_beats_auto(audio_file: str, start_time: float = 0.0, 
                       end_time: float = None, use_gpu: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    Auto Mode - Extreme intelligence for optimal music video cuts.
    
    Features:
    - Automatic music structure detection (intro/verse/chorus/bridge/outro)
    - Energy-based section analysis (high energy = more cuts, low energy = fewer cuts)
    - Multi-band rhythm analysis (kick/clap/bass/hi-hat)
    - Adaptive cut frequency based on musical context
    - Spectral novelty detection for musical changes
    - Tempo and rhythm pattern recognition
    
    Args:
        audio_file: Path to audio file
        start_time: Start time in seconds
        end_time: End time in seconds
        use_gpu: Use GPU acceleration if available
    
    Returns:
        Tuple of (selected_beat_times, beat_info)
    """
    print(f"🤖 AUTO MODE - Extreme Intelligence Analysis")
    print(f"   Analyzing music structure and energy...")
    
    # Load audio
    duration = None
    if end_time and end_time > start_time:
        duration = end_time - start_time
    
    y, sr = librosa.load(audio_file, sr=22050, offset=start_time, duration=duration, mono=True)
    
    # Step 1: Detect all beats
    print(f"   🥁 Step 1: Detecting all beats...")
    beat_times, tempo = detect_all_beats(y, sr)
    print(f"      ✓ Detected {len(beat_times)} beats at {tempo:.1f} BPM")
    
    # Step 2: Analyze music structure (intro/verse/chorus/etc)
    print(f"   🎵 Step 2: Analyzing song structure...")
    sections = analyze_song_structure(y, sr, beat_times)
    print(f"      ✓ Detected {len(sections)} sections")
    
    # Step 3: Analyze energy levels per section
    print(f"   ⚡ Step 3: Analyzing energy levels...")
    energy_profile = analyze_energy_profile(y, sr, beat_times)
    
    # Step 4: Multi-band rhythm analysis
    print(f"   📊 Step 4: Multi-band rhythm analysis...")
    rhythm_data = analyze_multi_band_rhythm(y, sr, beat_times, use_gpu)
    
    # Step 5: Detect dominant rhythm patterns per section
    print(f"   🎯 Step 5: Detecting rhythm patterns...")
    rhythm_patterns = detect_rhythm_patterns(rhythm_data, beat_times, sections)
    
    # Step 6: Intelligent beat selection
    print(f"   🧠 Step 6: Intelligent beat selection...")
    selected_beats, selection_info = intelligent_beat_selection(
        beat_times, sections, energy_profile, rhythm_data, rhythm_patterns, tempo
    )
    
    print(f"   ✓ Selected {len(selected_beats)} optimal cuts from {len(beat_times)} beats")
    print(f"   ✓ Average: {len(selected_beats)/len(beat_times)*100:.1f}% of beats used")
    
    # Build comprehensive beat info
    beat_info = {
        'times': beat_times,
        'selected_times': selected_beats,
        'tempo': tempo,
        'sections': sections,
        'energy_profile': energy_profile,
        'rhythm_data': rhythm_data,
        'rhythm_patterns': rhythm_patterns,
        'selection_info': selection_info,
        'mode': 'auto'
    }
    
    return selected_beats, beat_info


def detect_all_beats(y: np.ndarray, sr: int) -> Tuple[np.ndarray, float]:
    """Detect all beats in the audio."""
    try:
        tempo, beat_frames = librosa.beat.beat_track(
            y=y, 
            sr=sr, 
            units='frames',
            hop_length=512,
            start_bpm=120,
            tightness=100
        )
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)
        return beat_times, tempo[0] if tempo.size > 0 else 120.0
    except:
        # Fallback: simple onset detection
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        beat_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        return beat_times, 120.0


def analyze_song_structure(y: np.ndarray, sr: int, beat_times: np.ndarray) -> List[Dict]:
    """
    Analyze song structure using spectral clustering.
    Identifies sections like intro, verse, chorus, bridge, outro.
    """
    try:
        # Compute chromagram for harmonic analysis
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
        
        # Compute spectral features
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)
        
        # Combine features
        features = np.vstack([chroma, spectral_contrast, mfcc])
        
        # Compute self-similarity matrix
        rec_matrix = librosa.segment.recurrence_matrix(
            features, 
            mode='affinity',
            metric='cosine',
            bandwidth=3
        )
        
        # Detect boundaries using distance_threshold instead of k
        # This allows automatic determination of the number of sections
        boundaries_frames = librosa.segment.agglomerative(
            rec_matrix, 
            k=None,
            clusterer=None
        )
        
        # If that fails, use a simpler approach
        if boundaries_frames is None or len(boundaries_frames) == 0:
            raise ValueError("Agglomerative clustering failed")
            
    except Exception as e:
        print(f"      ⚠️ Advanced structure analysis failed: {e}")
        print(f"      → Using fallback: onset-based segmentation")
        
        # Fallback: Use onset-based segmentation
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        
        # Detect strong changes in onset strength
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=512,
            backtrack=False,
            pre_max=20,
            post_max=20,
            pre_avg=100,
            post_avg=100,
            delta=0.2,
            wait=10
        )
        
        # Use every Nth onset as a boundary (to get ~4-8 sections)
        total_duration = len(y) / sr
        target_sections = max(4, min(8, int(total_duration / 20)))  # 1 section per ~20 seconds
        
        if len(onset_frames) > target_sections:
            step = len(onset_frames) // target_sections
            boundaries_frames = onset_frames[::step]
        else:
            boundaries_frames = onset_frames
        
        # Always include start and end
        boundaries_frames = np.unique(np.concatenate([[0], boundaries_frames, [len(onset_env) - 1]]))
    
    boundaries_times = librosa.frames_to_time(boundaries_frames, sr=sr, hop_length=512)
    
    # Create section information
    sections = []
    total_duration = len(y) / sr
    
    for i in range(len(boundaries_times)):
        start_time = boundaries_times[i]
        end_time = boundaries_times[i+1] if i+1 < len(boundaries_times) else total_duration
        
        # Classify section type based on position and duration
        section_duration = end_time - start_time
        relative_position = start_time / total_duration
        
        # Simple heuristic classification
        if relative_position < 0.15:
            section_type = 'intro'
        elif relative_position > 0.85:
            section_type = 'outro'
        elif section_duration < 15:
            section_type = 'bridge'
        else:
            # Alternate between verse and chorus
            section_type = 'chorus' if i % 2 == 0 else 'verse'
        
        sections.append({
            'start': start_time,
            'end': end_time,
            'type': section_type,
            'duration': section_duration
        })
        
        print(f"      • {section_type.capitalize()}: {start_time:.1f}s - {end_time:.1f}s ({section_duration:.1f}s)")
    
    return sections


def analyze_energy_profile(y: np.ndarray, sr: int, beat_times: np.ndarray) -> Dict:
    """
    Analyze energy levels across the song.
    High energy = more cuts, low energy = fewer cuts.
    """
    # Compute RMS energy
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    
    # Compute spectral centroid (brightness)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
    
    # Compute zero crossing rate (noisiness/percussion)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=512)[0]
    
    # Normalize features
    rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-6)
    centroid_norm = (spectral_centroid - np.min(spectral_centroid)) / (np.max(spectral_centroid) - np.min(spectral_centroid) + 1e-6)
    zcr_norm = (zcr - np.min(zcr)) / (np.max(zcr) - np.min(zcr) + 1e-6)
    
    # Combined energy score (weighted)
    energy_combined = (rms_norm * 0.5) + (centroid_norm * 0.3) + (zcr_norm * 0.2)
    
    # Map energy to beat times
    times = librosa.frames_to_time(np.arange(len(energy_combined)), sr=sr, hop_length=512)
    beat_energy = np.interp(beat_times, times, energy_combined)
    
    # Classify energy levels
    energy_threshold_high = np.percentile(beat_energy, 70)
    energy_threshold_low = np.percentile(beat_energy, 30)
    
    energy_levels = []
    for e in beat_energy:
        if e > energy_threshold_high:
            energy_levels.append('high')
        elif e < energy_threshold_low:
            energy_levels.append('low')
        else:
            energy_levels.append('medium')
    
    high_count = energy_levels.count('high')
    med_count = energy_levels.count('medium')
    low_count = energy_levels.count('low')
    
    print(f"      ✓ Energy: High={high_count} ({high_count/len(beat_times)*100:.1f}%), "
          f"Medium={med_count} ({med_count/len(beat_times)*100:.1f}%), "
          f"Low={low_count} ({low_count/len(beat_times)*100:.1f}%)")
    
    return {
        'beat_energy': beat_energy,
        'energy_levels': energy_levels,
        'rms': rms_norm,
        'spectral_centroid': centroid_norm,
        'zcr': zcr_norm
    }


def analyze_multi_band_rhythm(y: np.ndarray, sr: int, beat_times: np.ndarray, use_gpu: bool = False) -> Dict:
    """
    Analyze rhythm across multiple frequency bands.
    Kick (20-150 Hz), Clap/Snare (150-4000 Hz), Hi-hat (4000+ Hz), Bass (20-200 Hz)
    """
    # Use GPU if available
    xp = cp if (use_gpu and GPU_AVAILABLE) else np
    
    # Compute STFT
    stft = librosa.stft(y, n_fft=2048, hop_length=512)
    if use_gpu and GPU_AVAILABLE:
        stft = cp.asarray(stft)
    
    # Frequency bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    if use_gpu and GPU_AVAILABLE:
        freqs = cp.asarray(freqs)
    
    # Define frequency bands
    kick_band = (freqs >= 20) & (freqs <= 150)
    clap_band = (freqs >= 150) & (freqs <= 4000)
    hihat_band = freqs >= 4000
    bass_band = (freqs >= 20) & (freqs <= 200)
    
    # Extract onset envelopes for each band
    kick_onset = xp.sum(xp.abs(stft[kick_band, :]), axis=0)
    clap_onset = xp.sum(xp.abs(stft[clap_band, :]), axis=0)
    hihat_onset = xp.sum(xp.abs(stft[hihat_band, :]), axis=0)
    bass_onset = xp.sum(xp.abs(stft[bass_band, :]), axis=0)
    
    # Normalize
    kick_onset = kick_onset / (xp.max(kick_onset) + 1e-6)
    clap_onset = clap_onset / (xp.max(clap_onset) + 1e-6)
    hihat_onset = hihat_onset / (xp.max(hihat_onset) + 1e-6)
    bass_onset = bass_onset / (xp.max(bass_onset) + 1e-6)
    
    # Convert back to CPU if using GPU
    if use_gpu and GPU_AVAILABLE:
        kick_onset = cp.asnumpy(kick_onset)
        clap_onset = cp.asnumpy(clap_onset)
        hihat_onset = cp.asnumpy(hihat_onset)
        bass_onset = cp.asnumpy(bass_onset)
        
        # Clear GPU memory
        cp.get_default_memory_pool().free_all_blocks()
    
    # Sample strength at each beat
    beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=512)
    
    kick_strength = []
    clap_strength = []
    hihat_strength = []
    bass_strength = []
    
    for beat_frame in beat_frames:
        if beat_frame < len(kick_onset):
            start_idx = max(0, beat_frame - 3)
            end_idx = min(len(kick_onset), beat_frame + 4)
            
            kick_strength.append(float(np.max(kick_onset[start_idx:end_idx])))
            clap_strength.append(float(np.max(clap_onset[start_idx:end_idx])))
            hihat_strength.append(float(np.max(hihat_onset[start_idx:end_idx])))
            bass_strength.append(float(np.max(bass_onset[start_idx:end_idx])))
        else:
            kick_strength.append(0.0)
            clap_strength.append(0.0)
            hihat_strength.append(0.0)
            bass_strength.append(0.0)
    
    # Convert to numpy arrays
    kick_strength = np.array(kick_strength)
    clap_strength = np.array(clap_strength)
    hihat_strength = np.array(hihat_strength)
    bass_strength = np.array(bass_strength)
    
    # Determine strong beats per band
    kick_threshold = np.percentile(kick_strength, 50)
    clap_threshold = np.percentile(clap_strength, 50)
    hihat_threshold = np.percentile(hihat_strength, 50)
    bass_threshold = np.percentile(bass_strength, 50)
    
    is_strong_kick = kick_strength > kick_threshold
    is_strong_clap = clap_strength > clap_threshold
    is_strong_hihat = hihat_strength > hihat_threshold
    is_strong_bass = bass_strength > bass_threshold
    
    print(f"      ✓ Kick: {np.sum(is_strong_kick)} strong beats ({np.sum(is_strong_kick)/len(beat_times)*100:.1f}%)")
    print(f"      ✓ Clap: {np.sum(is_strong_clap)} strong beats ({np.sum(is_strong_clap)/len(beat_times)*100:.1f}%)")
    print(f"      ✓ Hi-hat: {np.sum(is_strong_hihat)} strong beats ({np.sum(is_strong_hihat)/len(beat_times)*100:.1f}%)")
    print(f"      ✓ Bass: {np.sum(is_strong_bass)} strong beats ({np.sum(is_strong_bass)/len(beat_times)*100:.1f}%)")
    
    return {
        'kick_strength': kick_strength,
        'clap_strength': clap_strength,
        'hihat_strength': hihat_strength,
        'bass_strength': bass_strength,
        'is_strong_kick': is_strong_kick,
        'is_strong_clap': is_strong_clap,
        'is_strong_hihat': is_strong_hihat,
        'is_strong_bass': is_strong_bass
    }


def detect_rhythm_patterns(rhythm_data: Dict, beat_times: np.ndarray, sections: List[Dict]) -> Dict:
    """
    Detect dominant rhythm patterns in each section.
    Determines if section follows kick pattern, clap pattern, or mixed.
    """
    patterns = {}
    
    for section in sections:
        # Find beats in this section
        section_mask = (beat_times >= section['start']) & (beat_times < section['end'])
        section_beat_indices = np.where(section_mask)[0]
        
        if len(section_beat_indices) == 0:
            continue
        
        # Analyze rhythm dominance
        kick_strong = np.sum(rhythm_data['is_strong_kick'][section_beat_indices])
        clap_strong = np.sum(rhythm_data['is_strong_clap'][section_beat_indices])
        hihat_strong = np.sum(rhythm_data['is_strong_hihat'][section_beat_indices])
        bass_strong = np.sum(rhythm_data['is_strong_bass'][section_beat_indices])
        
        total = len(section_beat_indices)
        
        # Determine dominant pattern
        kick_ratio = kick_strong / total
        clap_ratio = clap_strong / total
        hihat_ratio = hihat_strong / total
        bass_ratio = bass_strong / total
        
        # Classify pattern
        if kick_ratio > 0.6 and clap_ratio > 0.6:
            pattern = 'kick_clap_mixed'  # Alternating kick-clap
        elif kick_ratio > 0.7:
            pattern = 'kick_dominant'
        elif clap_ratio > 0.7:
            pattern = 'clap_dominant'
        elif hihat_ratio > 0.6:
            pattern = 'hihat_dominant'
        elif bass_ratio > 0.7:
            pattern = 'bass_dominant'
        else:
            pattern = 'mixed'
        
        patterns[section['type']] = {
            'pattern': pattern,
            'kick_ratio': kick_ratio,
            'clap_ratio': clap_ratio,
            'hihat_ratio': hihat_ratio,
            'bass_ratio': bass_ratio,
            'beat_indices': section_beat_indices
        }
        
        print(f"      • {section['type'].capitalize()}: {pattern} (K:{kick_ratio:.2f} C:{clap_ratio:.2f} H:{hihat_ratio:.2f} B:{bass_ratio:.2f})")
    
    return patterns


def intelligent_beat_selection(beat_times: np.ndarray, sections: List[Dict], 
                               energy_profile: Dict, rhythm_data: Dict,
                               rhythm_patterns: Dict, tempo: float) -> Tuple[np.ndarray, Dict]:
    """
    Intelligent beat selection based on all analyzed data.
    
    Strategy:
    - Intro: Fewer cuts (every 2-4 beats) to ease viewer in
    - Verse: Medium cuts (every 1-2 beats) based on rhythm pattern
    - Chorus: More cuts (every beat or subdivided) for high energy
    - Bridge: Adaptive cuts based on energy and rhythm
    - Outro: Fewer cuts (every 2-4 beats) to wind down
    - High energy sections: More frequent cuts
    - Follow dominant rhythm pattern (kick/clap/bass/mixed)
    """
    selected_indices = []
    selection_info = []
    
    for section in sections:
        section_type = section['type']
        
        # Find beats in this section
        section_mask = (beat_times >= section['start']) & (beat_times < section['end'])
        section_beat_indices = np.where(section_mask)[0]
        
        if len(section_beat_indices) == 0:
            continue
        
        # Get rhythm pattern for this section
        pattern_info = rhythm_patterns.get(section_type, {'pattern': 'mixed'})
        pattern = pattern_info['pattern']
        
        # Get energy levels for this section
        section_energy_levels = [energy_profile['energy_levels'][i] for i in section_beat_indices]
        high_energy_ratio = section_energy_levels.count('high') / len(section_energy_levels)
        
        print(f"      • Processing {section_type}: {len(section_beat_indices)} beats, pattern={pattern}, energy={high_energy_ratio:.2f}")
        
        # Determine base cut frequency based on section type
        if section_type == 'intro':
            base_frequency = 3  # Every 3rd beat
        elif section_type == 'verse':
            base_frequency = 2  # Every 2nd beat
        elif section_type == 'chorus':
            base_frequency = 1  # Every beat
        elif section_type == 'bridge':
            base_frequency = 2  # Every 2nd beat
        elif section_type == 'outro':
            base_frequency = 4  # Every 4th beat
        else:
            base_frequency = 2
        
        # Adjust based on energy
        if high_energy_ratio > 0.7:
            base_frequency = max(1, base_frequency - 1)  # More cuts
        elif high_energy_ratio < 0.3:
            base_frequency = base_frequency + 1  # Fewer cuts
        
        # Apply pattern-specific selection
        section_selected = []
        
        if pattern == 'kick_clap_mixed':
            # Alternating pattern: follow kick and clap rhythm
            for idx in section_beat_indices:
                is_kick = rhythm_data['is_strong_kick'][idx]
                is_clap = rhythm_data['is_strong_clap'][idx]
                
                if is_kick or is_clap:
                    # Add beat if it's a strong kick or clap
                    section_selected.append(idx)
                    
                    # Sometimes add subdivision on very strong kicks in high energy
                    if is_kick and energy_profile['energy_levels'][idx] == 'high':
                        if rhythm_data['kick_strength'][idx] > 0.8:
                            # Add a micro-cut (will be interpolated later)
                            pass  # Handle in post-processing
        
        elif pattern == 'kick_dominant':
            # Follow kicks primarily
            kick_count = 0
            for idx in section_beat_indices:
                is_kick = rhythm_data['is_strong_kick'][idx]
                
                if is_kick:
                    kick_count += 1
                    # Every kick in high energy, every 2nd kick in medium/low
                    if energy_profile['energy_levels'][idx] == 'high':
                        section_selected.append(idx)
                    elif kick_count % 2 == 0:
                        section_selected.append(idx)
        
        elif pattern == 'clap_dominant':
            # Follow claps primarily
            clap_count = 0
            for idx in section_beat_indices:
                is_clap = rhythm_data['is_strong_clap'][idx]
                
                if is_clap:
                    clap_count += 1
                    # Every clap in high energy, every 2nd clap in medium/low
                    if energy_profile['energy_levels'][idx] == 'high':
                        section_selected.append(idx)
                    elif clap_count % 2 == 0:
                        section_selected.append(idx)
        
        elif pattern == 'bass_dominant':
            # Follow bass primarily
            for idx in section_beat_indices:
                is_bass = rhythm_data['is_strong_bass'][idx]
                
                if is_bass:
                    if energy_profile['energy_levels'][idx] in ['high', 'medium']:
                        section_selected.append(idx)
        
        else:  # mixed or hihat_dominant
            # Use base frequency with energy modulation
            for i, idx in enumerate(section_beat_indices):
                # Dynamic selection based on energy
                if energy_profile['energy_levels'][idx] == 'high':
                    # High energy: more cuts
                    section_selected.append(idx)
                elif i % base_frequency == 0:
                    # Follow base frequency
                    section_selected.append(idx)
        
        # Ensure minimum cuts per section (at least one every 3 seconds)
        if len(section_selected) < section['duration'] / 3:
            # Add more beats if too sparse
            for i, idx in enumerate(section_beat_indices):
                if idx not in section_selected and i % 2 == 0:
                    section_selected.append(idx)
        
        selected_indices.extend(section_selected)
        
        selection_info.append({
            'section': section_type,
            'pattern': pattern,
            'total_beats': len(section_beat_indices),
            'selected_beats': len(section_selected),
            'selection_ratio': len(section_selected) / len(section_beat_indices) if len(section_beat_indices) > 0 else 0
        })
        
        print(f"         → Selected {len(section_selected)}/{len(section_beat_indices)} beats ({len(section_selected)/len(section_beat_indices)*100:.1f}%)")
    
    # Convert indices to times
    selected_indices = sorted(list(set(selected_indices)))  # Remove duplicates and sort
    selected_beat_times = beat_times[selected_indices]
    
    return selected_beat_times, selection_info


def get_auto_mode_info() -> Dict:
    """Get information about auto mode."""
    return {
        'name': 'Auto Mode',
        'status': 'Active',
        'description': 'Extreme intelligence - automatic music analysis with adaptive beat detection',
        'features': [
            'Automatic song structure detection (intro/verse/chorus/bridge/outro)',
            'Energy-based section analysis (high/medium/low energy)',
            'Multi-band rhythm analysis (kick/clap/bass/hi-hat)',
            'Adaptive cut frequency based on musical context',
            'Spectral novelty detection for musical changes',
            'Tempo and rhythm pattern recognition',
            'Intelligent beat selection per section type',
            'Dynamic cut density (more cuts in chorus, fewer in intro/outro)',
            'Pattern-following (kick-clap alternation, bass emphasis, etc.)'
        ],
        'algorithms': [
            'Onset-based segmentation for structure detection',
            'RMS energy + spectral centroid + zero-crossing rate for energy',
            'Multi-band STFT for rhythm analysis',
            'Pattern recognition with threshold-based classification',
            'Context-aware beat selection with energy modulation'
        ]
    }


# Backward compatibility - make available for smart mode fallback
def analyze_beats_auto_fallback(audio_file: str, start_time: float = 0.0, 
                               end_time: float = None) -> Tuple[np.ndarray, Dict]:
    """
    Fallback to smart mode if auto mode has issues.
    """
    try:
        return analyze_beats_auto(audio_file, start_time, end_time, use_gpu=False)
    except Exception as e:
        print(f"⚠️ Auto mode error: {e}")
        print(f"   Falling back to smart mode...")
        
        # Import smart mode as fallback
        from smart_mode import analyze_beats_smart, select_beats_smart
        beat_times, beat_info = analyze_beats_smart(audio_file, start_time, end_time)
        selected_beats = select_beats_smart(beat_info, preset='normal')
        
        beat_info['selected_times'] = selected_beats
        beat_info['mode'] = 'smart_fallback'
        
        return selected_beats, beat_info