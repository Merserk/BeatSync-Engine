"""Short real exports through BeatSync's clip, concat, and renderer paths."""

import contextlib
import json
from pathlib import Path
import random
import subprocess
import time
from unittest.mock import patch

import numpy as np

import ffmpeg_processing as media
import video_processor as renderer
from video_encoders import VULKAN_ENCODERS, get_vulkan_encoder_support


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "vulkan_export_test"


def run(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:])
    return result.stdout


def verify(path, encoder, frames, audio=False):
    data = json.loads(run([media.FFPROBE_PATH, "-v", "error", "-count_frames",
                           "-show_streams", "-show_format", "-of", "json", str(path)]))
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    assert video["codec_name"] == ("h264" if encoder == "h264_vulkan" else "hevc"), video
    assert int(video["nb_read_frames"]) == frames, video
    assert video["avg_frame_rate"] == "30/1", video
    assert (video["width"], video["height"]) == (512, 288), video
    if encoder == "hevc_vulkan":
        assert video["codec_tag_string"] == "hvc1", video
    if audio:
        sound = next(s for s in data["streams"] if s["codec_type"] == "audio")
        assert sound["codec_name"] == "pcm_s24le", sound
        assert sound["sample_rate"] == "48000", sound
        assert abs(float(sound["duration"]) - 2.0) <= 1 / 30, sound
    run([media.FFMPEG_PATH, "-nostdin", "-v", "error", "-xerror", "-i", str(path),
         "-map", "0", "-f", "null", "-"])
    return {"path": str(path.relative_to(ROOT)), "codec": video["codec_name"],
            "frames": int(video["nb_read_frames"]), "fps": video["avg_frame_rate"],
            "duration": float(data["format"]["duration"]), "decode": "passed",
            "audio": "PCM 24-bit 48 kHz" if audio else None}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"ffmpeg": run([media.FFMPEG_PATH, "-version"]).splitlines()[0],
              "support": {}, "exports": {}, "passed": False}
    source, audio = OUT / "source.mp4", OUT / "music.wav"
    run([media.FFMPEG_PATH, "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=640x360:rate=30", "-t", "4", "-c:v", "libx264", "-crf", "18",
         "-pix_fmt", "yuv420p", str(source)])
    run([media.FFMPEG_PATH, "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
         "sine=frequency=440:sample_rate=48000", "-t", "2", "-c:a", "pcm_s24le", str(audio)])
    try:
        for encoder in VULKAN_ENCODERS:
            report["support"][encoder] = get_vulkan_encoder_support(encoder)
            assert report["support"][encoder]["available"], report["support"][encoder]
            started = time.perf_counter()
            checks = []
            with (OUT / f"{encoder}_pipeline.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
                clips = []
                # Slow down, speed up, and retain normal speed; all resize.
                for i, (start, duration, output_duration, frames) in enumerate([
                    (0.1, 0.5, 1.0, 30), (1.0, 1.0, 0.5, 15), (2.0, 0.5, 0.5, 15)
                ]):
                    clip = OUT / f"{encoder}_clip_{i}.mp4"
                    assert media.extract_clip_segment_ffmpeg(
                        str(source), start, duration, str(clip), 30, (512, 288),
                        use_nvenc=True, gpu_encoder=encoder, output_duration=output_duration)
                    checks.append(verify(clip, encoder, frames))
                    clips.append(str(clip))
                for fast in (True, False):
                    output = OUT / f"{encoder}_{'copy' if fast else 'reencode'}.mp4"
                    with patch.dict("os.environ", {"BEATSYNC_FAST_CONCAT_COPY": "1" if fast else "0"}):
                        media.concatenate_videos_ffmpeg(
                            clips, str(output), audio_file=str(audio), use_nvenc=True,
                            gpu_encoder=encoder, fps=30, temp_dir=str(OUT))
                    checks.append(verify(output, encoder, 60, audio=True))

                # Exercise top-level rendering with analysis on CPU and NVENC off.
                output = OUT / f"{encoder}_app.mp4"
                info = {"mode": "auto", "tempo": 120}
                random.seed(42)
                with patch.object(renderer, "NVENC_AVAILABLE", False), \
                     patch.object(renderer, "get_processing_dir", return_value=str(OUT / f"{encoder}_processing")), \
                     patch.object(renderer, "get_video_resolution", return_value=(512, 288)):
                    renderer.create_music_video(
                        str(audio), [str(source)], np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
                        output_file=str(output), max_workers=2, beat_info=info,
                        use_gpu=False, gpu_encoder=encoder, fps=30,
                        auto_speed_clip_control=False, avoid_text_clips=False)
                assert info["render_info"]["encoder"] == encoder.upper(), info
                checks.append(verify(output, encoder, 60, audio=True))
            report["exports"][encoder] = {"checks": checks, "elapsed_seconds": time.perf_counter() - started}
            print(f"{encoder}: {len(checks)} exports passed (clipping, retiming, concat copy/reencode, CPU-analysis render)")

        import gui
        original_support = gui.get_encoder_support
        def without_nvenc(encoder):
            return {"available": False, "error": "disabled in test"} if encoder in gui.NVENC_ENCODERS else original_support(encoder)
        with patch.object(gui, "NVENC_AVAILABLE", False), patch.object(gui, "get_encoder_support", side_effect=without_nvenc):
            app = gui.create_ui()
        radio = next(c for c in app.config["components"] if c["type"] == "radio")
        values = {value for label, value in radio["props"]["choices"]}
        assert set(VULKAN_ENCODERS).issubset(values), values
        report["ui_without_nvenc"] = radio["props"]["choices"]
        report["passed"] = True
        print("GUI exposes both Vulkan options with NVENC disabled.")
    finally:
        (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
