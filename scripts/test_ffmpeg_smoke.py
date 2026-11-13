"""Real short exports for every encoder plus FFmpeg media edge cases."""

from contextlib import redirect_stdout
from fractions import Fraction
import json
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import numpy as np

import ffmpeg_processing as media
import ffmpeg_runtime as runtime
import video_processor as renderer
from video_encoders import get_encoder_support, HARDWARE_ENCODERS


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'ffmpeg_update'


def run(args):
    return runtime.checked_output(args, timeout=120)


def verify(file, codec, frames, fps=30, size=(512, 288), audio=False):
    video = runtime.verify_video(file, frames)
    assert video['codec_name'] == codec, video
    assert Fraction(video['avg_frame_rate']) == Fraction(runtime.fps_string(fps)), video
    assert (video['width'], video['height']) == size, video
    assert video.get('sample_aspect_ratio', '1:1') == '1:1', video
    streams = runtime.probe_media(file)['streams']
    sound = next((s for s in streams if s['codec_type'] == 'audio'), None)
    if audio:
        assert sound and sound['codec_name'] == 'pcm_s24le', sound
        assert sound['sample_rate'] == '48000', sound
        assert abs(float(sound['duration']) - frames / fps) < 1 / fps, sound
    run([media.FFMPEG_PATH, '-nostdin', '-v', 'error', '-xerror', '-i', str(file), '-map', '0', '-f', 'null', '-'])
    return {'file': str(file.relative_to(ROOT)), 'codec': codec, 'frames': frames,
            'fps': video['avg_frame_rate'], 'size': size, 'decode': 'passed', 'audio': audio}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {'runtime': runtime.get_runtime_info(), 'encoders': {}, 'edge_cases': {}, 'passed': False}
    source = OUT / "O'Brien – тест.mp4"
    music = OUT / 'music_4s.wav'
    # The app now rejects repeated visual content. A looping test pattern and
    # audio as long as the source cannot provide multiple separated unique cuts.
    # Pan steadily across a unique pattern. Each frame differs, while the
    # movement is continuous rather than hundreds of one-frame source shots.
    raw = OUT / 'unique_source.bgr24'
    rng = np.random.default_rng(9271)
    panorama = rng.integers(16, 240, (12, 380, 3), dtype=np.uint8)
    with raw.open('wb') as data:
        for index in range(360):
            blocks = panorama[:, index:index + 20]
            data.write(np.repeat(np.repeat(blocks, 30, axis=0), 32, axis=1).tobytes())
    run([media.FFMPEG_PATH, '-v', 'error', '-y', '-f', 'rawvideo', '-pixel_format', 'bgr24',
         '-video_size', '640x360', '-framerate', '30', '-i', str(raw),
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)])
    run([media.FFMPEG_PATH, '-v', 'error', '-y', '-f', 'lavfi', '-i', 'sine=frequency=997:sample_rate=48000',
         '-t', '4', '-c:a', 'pcm_s24le', str(music)])
    try:
        with (OUT / 'pipeline.log').open('w', encoding='utf8') as log, redirect_stdout(log):
            for encoder in ('none', 'h264_nvenc', 'hevc_nvenc', 'h264_vulkan', 'hevc_vulkan', 'prores_ks'):
                started = time.perf_counter()
                support = get_encoder_support(encoder) if encoder != 'none' else {'available': True}
                assert support['available'], support
                codec = 'hevc' if encoder.startswith('hevc') else 'prores' if encoder == 'prores_ks' else 'h264'
                prores = encoder == 'prores_ks'
                source_file = media.convert_to_prores_hq(str(source), str(OUT / 'prores_sources'), 30, (512, 288)) if prores else str(source)
                clips, checks = [], []
                for index, (duration, output_duration, frame_count) in enumerate([(.5, 1., 30), (1., .5, 15), (.5, .5, 15)]):
                    file = OUT / f"{encoder}_clip_{index}' тест{'.mov' if prores else '.mp4'}"
                    if prores:
                        segment = media.extract_prores_segment_random(source_file, duration, 30, str(OUT), index, index * .5, output_duration)
                        shutil.move(segment, file)
                    else:
                        assert media.extract_clip_segment_ffmpeg(source_file, index * .5, duration, str(file),
                                                                30, (513, 289), encoder in HARDWARE_ENCODERS, encoder, output_duration)
                    checks.append(verify(file, codec, frame_count))
                    clips.append(str(file.relative_to(ROOT)))  # Relative and apostrophe/Unicode paths.
                for fast in ('1', '0'):
                    file = OUT / f"{encoder}_{'copy' if fast == '1' else 'reencode'}{'.mov' if prores else '.mp4'}"
                    with patch.dict('os.environ', {'BEATSYNC_FAST_CONCAT_COPY': fast}):
                        media.concatenate_videos_ffmpeg(clips, str(file), str(music), start_time=.5, end_time=2.5,
                                                       use_nvenc=encoder in HARDWARE_ENCODERS, gpu_encoder=encoder, fps=30,
                                                       temp_dir=str(OUT), output_duration=2., clip_frame_counts=[30, 15, 15])
                    checks.append(verify(file, codec, 60, audio=True))
                app_file = OUT / f"{encoder}_app{'.mov' if prores else '.mp4'}"
                info = {'mode': 'auto'}
                # CPU analysis with every hardware encoder; mixed input geometry
                # in ProRes mode is covered separately below.
                with patch.object(renderer, 'get_processing_dir', return_value=str(OUT / 'processing')), \
                     patch.object(renderer, 'get_video_resolution', return_value=(512, 288)):
                    renderer.create_music_video(str(music), [str(source)], np.array([0, .5, 1, 1.5, 2]),
                                                str(app_file), end_time=2., max_workers=2, beat_info=info,
                                                export_format='prores_hq' if prores else 'auto', use_gpu=False, gpu_encoder=encoder, fps=30,
                                                auto_speed_clip_control=False, avoid_text_clips=False)
                checks.append(verify(app_file, codec, 60, audio=True))
                preview = OUT / f'{encoder}_preview.mp4'
                media.create_browser_preview(str(app_file), str(preview))
                streams = runtime.probe_media(preview)['streams']
                assert [s['codec_name'] for s in streams] == ['h264', 'aac'], streams
                run([media.FFMPEG_PATH, '-v', 'error', '-xerror', '-i', str(preview), '-f', 'null', '-'])
                report['encoders'][encoder] = {'support': support, 'exports': checks, 'preview': 'H.264/AAC passed',
                                               'elapsed_seconds': time.perf_counter() - started}

            ntsc = OUT / 'ntsc_source.mp4'
            run([media.FFMPEG_PATH, '-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc2=size=512x288:rate=30000/1001',
                 '-t', '2', '-c:v', 'libx264', str(ntsc)])
            fps = media.get_video_fps(str(ntsc))
            assert runtime.fps_string(fps) == '30000/1001'
            ntsc_clips = []
            for index in range(4):
                file = OUT / f'ntsc_{index}.mp4'
                assert media.extract_clip_segment_ffmpeg(str(ntsc), index * .2, 6 / fps, str(file), fps, (512, 288), False)
                ntsc_clips.append(str(file))
            file = OUT / 'ntsc_final.mp4'
            media.concatenate_videos_ffmpeg(ntsc_clips, str(file), str(music), fps=fps, temp_dir=str(OUT),
                                            output_duration=24 / fps, clip_frame_counts=[6] * 4)
            report['edge_cases']['fractional_fps'] = verify(file, 'h264', 24, fps=fps, audio=True)

            rotated = OUT / 'rotated.mp4'
            run([media.FFMPEG_PATH, '-v', 'error', '-y', '-display_rotation:v:0', '90', '-i', str(source), '-c', 'copy', str(rotated)])
            assert media.get_video_resolution(str(rotated)) == (360, 640)
            rotated_clip = OUT / 'rotated_clip.mp4'
            assert media.extract_clip_segment_ffmpeg(str(rotated), 0, .5, str(rotated_clip), 30, (360, 640), False)
            report['edge_cases']['rotation'] = verify(rotated_clip, 'h264', 15, size=(360, 640))

            mixed_folder = OUT / 'other_source'
            mixed_folder.mkdir(exist_ok=True)
            square = mixed_folder / source.name
            run([media.FFMPEG_PATH, '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=red:size=512x512:rate=30',
                 '-t', '2', '-c:v', 'libx264', str(square)])
            mixed = OUT / 'mixed_prores.mov'
            with patch.object(renderer, 'get_processing_dir', return_value=str(OUT / 'processing')), \
                 patch.object(renderer, 'get_video_resolution', return_value=(512, 288)):
                renderer.create_music_video(str(music), [str(source), str(square)], np.array([0, .5, 1, 1.5, 2]),
                                            str(mixed), end_time=2., export_format='prores_hq', gpu_encoder='prores_ks',
                                            fps=30, beat_info={'mode': 'auto'}, auto_speed_clip_control=False, avoid_text_clips=False)
            report['edge_cases']['mixed_geometry_and_identical_basenames'] = verify(mixed, 'prores', 60, audio=True)

            protected = OUT / 'failed_export_preserves_previous.mp4'
            protected.write_bytes(b'previous export remains intact')
            try:
                media.concatenate_videos_ffmpeg([str(OUT / 'missing.mp4')], str(protected), output_duration=1., temp_dir=str(OUT))
            except RuntimeError:
                assert protected.read_bytes() == b'previous export remains intact'
            else:
                raise AssertionError('Invalid inputs unexpectedly succeeded')
            report['edge_cases']['failed_export_preserves_existing'] = 'passed'

            repaired = OUT / 'copy_validation_fallback.mp4'
            media.concatenate_videos_ffmpeg([ntsc_clips[0]], str(repaired), str(music), fps=fps, temp_dir=str(OUT), output_duration=12 / fps)
            report['edge_cases']['incomplete_copy_reencode_fallback'] = verify(repaired, 'h264', 12, fps=fps, audio=True)

            corrupt = OUT / 'corrupt.mp4'
            corrupt.write_bytes(b'not a valid media file')
            for probe in (media.get_video_duration, media.get_video_fps, media.get_video_resolution):
                try:
                    probe(str(corrupt))
                except runtime.MediaError:
                    pass
                else:
                    raise AssertionError('Corrupt input returned fabricated metadata')
            report['edge_cases']['corrupt_input_errors'] = 'passed'

            cpu_preview = OUT / 'cpu_browser_preview.mp4'
            with patch.object(media, 'get_encoder_support', return_value={'available': False}):
                media.create_browser_preview(str(OUT / 'prores_ks_app.mov'), str(cpu_preview))
            assert [s['codec_name'] for s in runtime.probe_media(cpu_preview)['streams']] == ['h264', 'aac']
            report['edge_cases']['preview_without_nvenc'] = 'passed'

            scene_source = OUT / 'scene_change.mp4'
            run([media.FFMPEG_PATH, '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=red:s=512x288:r=30:d=1',
                 '-f', 'lavfi', '-i', 'color=blue:s=512x288:r=30:d=1',
                 '-filter_complex', '[0:v][1:v]concat=n=2:v=1:a=0', '-c:v', 'libx264', '-g', '30', str(scene_source)])
            scenes = media.detect_video_scene_changes(str(scene_source), threshold=.27, analysis_fps=6, analysis_width=256)
            keys = media.detect_video_keyframes(str(scene_source))
            assert any(abs(t - 1.) <= 1 / 6 for t in scenes), scenes
            assert keys and abs(keys[0]) < .01, keys
            report['edge_cases']['scene_and_keyframe_detection'] = {'scenes': scenes, 'keyframes': keys}

            sliced_raw, source_raw = OUT / 'sliced_audio.f32', OUT / 'original_audio.f32'
            for input_file, raw_file in [(OUT / 'none_copy.mp4', sliced_raw), (music, source_raw)]:
                run([media.FFMPEG_PATH, '-v', 'error', '-y', '-i', str(input_file), '-map', '0:a:0',
                     '-ac', '1', '-ar', '48000', '-c:a', 'pcm_f32le', '-f', 'f32le', str(raw_file)])
            original = np.fromfile(source_raw, dtype=np.float32)
            sliced = np.fromfile(sliced_raw, dtype=np.float32)
            assert len(sliced) == 96000, len(sliced)
            error = float(np.max(np.abs(sliced - original[24000:120000])))
            assert error < 1e-6, error
            report['edge_cases']['sample_accurate_audio_slice'] = {'samples': len(sliced), 'max_error': error}

            cuda_clip = OUT / 'optional_cuda_decode.mp4'
            with patch.dict('os.environ', {'BEATSYNC_FFMPEG_CUDA_DECODE': '1'}):
                assert media.extract_clip_segment_ffmpeg(str(source), .5, .5, str(cuda_clip), 30, (512, 288), True, 'h264_nvenc')
            report['edge_cases']['optional_cuda_decode'] = verify(cuda_clip, 'h264', 15)

            # Verify that the UI returns a browser preview and the original
            # high-quality export as distinct downloadable files.
            import gui
            with patch.object(gui, 'analyze_beats_auto', return_value=(np.array([0, .5, 1]), {'mode': 'auto'})), \
                 patch.object(gui, 'get_output_dir', return_value=str(OUT)), \
                 patch.object(renderer, 'get_processing_dir', return_value=str(OUT / 'processing')):
                state = {}
                preview, status, state = gui._process_video_impl(str(music), [str(source)], 'ui_export.mp4', 'cpu', 30,
                                                               False, False, state)
            assert preview and state['export_path'] and preview != state['export_path'], status
            assert [s['codec_name'] for s in runtime.probe_media(preview)['streams']] == ['h264', 'aac']
            assert [s['codec_name'] for s in runtime.probe_media(state['export_path'])['streams']] == ['h264', 'pcm_s24le']
            report['edge_cases']['ui_preview_and_original_download'] = 'passed'
        report['passed'] = True
    finally:
        (OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(f"Passed: {len(report['encoders'])} export modes, {sum(len(v['exports']) for v in report['encoders'].values())} exports, browser previews, and {len(report['edge_cases'])} edge cases.")


if __name__ == '__main__':
    main()
