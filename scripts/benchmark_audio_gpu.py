"""Numerical and timing comparison of CPU/GPU audio separation and complete cuts."""
import json
from pathlib import Path
import time
import contextlib
import io

import librosa
import numpy as np
import taichi_backend
from auto_mode import analyze_beats_auto

ROOT = Path(__file__).resolve().parents[1]


def main():
    audio = next((ROOT / 'for amv').glob('*.flac'))
    y, sr = librosa.load(audio, sr=22050)
    y = librosa.util.normalize(y)
    # Warm CPU FFT compilation; warm GPU kernels have a separate cold figure.
    librosa.effects.hpss(y[:22050])
    tick = time.perf_counter()
    taichi_backend.harmonic_percussive_audio(y[:22050])
    cold = time.perf_counter() - tick
    timings = {}
    outputs = {}
    for name, function in (('cpu', librosa.effects.hpss), ('gpu', taichi_backend.harmonic_percussive_audio)):
        timings[name] = []
        for _ in range(3):
            tick = time.perf_counter()
            outputs[name] = function(y)
            timings[name].append(time.perf_counter() - tick)
    errors = [float(np.max(np.abs(a - b))) for a, b in zip(outputs['cpu'], outputs['gpu'])]
    assert max(errors) < 5e-6, errors
    cuts = {}
    complete = {}
    for gpu in (False, True):
        tick = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            values, info = analyze_beats_auto(str(audio), use_gpu=gpu, enable_video_analysis=False)
        cuts[str(gpu)] = values.tolist()
        complete[str(gpu)] = {'seconds': time.perf_counter() - tick, 'tempo': info['tempo'],
                              'cut_count': len(values), 'separation_backend': info['audio_separation_backend']}
    np.testing.assert_allclose(cuts['False'], cuts['True'], rtol=0, atol=1e-6)
    result = {'audio_seconds': len(y)/sr, 'gpu_cold_one_second': cold, 'separation_seconds': timings,
              'max_waveform_errors': errors, 'full_audio_analysis': complete, 'cuts_identical': True}
    print(json.dumps(result, indent=2), flush=True)
    (ROOT / 'output' / 'gpu_priority' / 'audio_benchmark.json').write_text(json.dumps(result, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
