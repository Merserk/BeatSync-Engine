"""Compare real full-source GPU analysis scans, parallel versus sequential."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from gpu_devices import get_discrete_gpu
from gpu_video import decoded_analysis_frames, get_gpu_video_path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'gpu_priority' / 'scan_concurrency.json'


def main():
    sources = [ROOT / 'for amv' / f'{i}.mkv' for i in (1, 2)]
    routes = [get_gpu_video_path(str(source)) for source in sources]
    assert all(route and route.hardware_decode for route in routes), routes
    report = {'gpu': get_discrete_gpu(), 'passed': False, 'runs': []}

    def scan(index):
        tick = time.perf_counter()
        with decoded_analysis_frames(str(sources[index]), route=routes[index]) as frames:
            item = {'source': sources[index].name, 'backend': frames.backend,
                    'sample_frames': frames.count, 'scene_changes': frames.scene_changes}
        item['seconds'] = time.perf_counter() - tick
        print(json.dumps({k: v for k, v in item.items() if k != 'scene_changes'}), flush=True)
        return item

    try:
        for mode in ('parallel', 'sequential'):
            print('Starting ' + mode, flush=True)
            tick = time.perf_counter()
            if mode == 'parallel':
                with ThreadPoolExecutor(max_workers=2) as pool:
                    items = list(pool.map(scan, range(2)))
            else:
                items = [scan(i) for i in range(2)]
            run = {'mode': mode, 'wall_seconds': time.perf_counter() - tick, 'sources': items}
            report['runs'].append(run)
            print(mode + ' total: ' + str(round(run['wall_seconds'], 3)), flush=True)
        first, second = report['runs']
        assert all(a['sample_frames'] == b['sample_frames'] and a['scene_changes'] == b['scene_changes']
                   for a, b in zip(first['sources'], second['sources']))
        report['passed'] = True
    finally:
        OUT.write_text(json.dumps(report, indent=2), encoding='utf8')
    print('Report: ' + str(OUT), flush=True)


if __name__ == '__main__':
    main()
