import datetime
import sys
import unittest
import warnings
from pathlib import Path
from unittest import mock

warnings.simplefilter("ignore", ResourceWarning)


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gui


class FixedDateTime(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 3, 27, 12, 0, 0, tzinfo=tz)


class GuiRegressionTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        try:
            loop = gui.asyncio.get_event_loop()
        except RuntimeError:
            return
        if not loop.is_closed():
            loop.close()

    def test_create_ui_builds(self):
        app = gui.create_ui()
        self.assertEqual(type(app).__name__, "Blocks")

    def test_update_processing_ui_switches_export_controls(self):
        quality_update, prores_update, summary_markup, output_name = gui.update_processing_ui(
            "prores_proxy",
            "demo.mp4",
        )

        self.assertEqual(quality_update["visible"], False)
        self.assertEqual(quality_update["interactive"], False)
        self.assertEqual(prores_update["visible"], True)
        self.assertEqual(prores_update["interactive"], True)
        self.assertIn("ProRes", summary_markup)
        self.assertEqual(output_name, "demo.mov")

    def test_render_button_feedback_updates(self):
        button_update, status_text = gui.build_render_started_feedback()

        self.assertEqual(button_update["value"], "Creating...")
        self.assertFalse(button_update["interactive"])
        self.assertIn("Render started.", status_text)
        self.assertEqual(gui.reset_render_button()["value"], "Create Music Video")

    def test_source_summary_reports_valid_local_media(self):
        audio_path = r"C:\media\track.mp3"
        clip_dir = r"C:\media\clips"
        with (
            mock.patch.object(gui, "normalize_local_path", side_effect=lambda value: value),
            mock.patch.object(gui.os.path, "isfile", side_effect=lambda value: value == audio_path),
            mock.patch.object(gui.os.path, "isdir", side_effect=lambda value: value == clip_dir),
            mock.patch.object(gui, "get_video_files", return_value=[r"C:\media\clips\a.mp4", r"C:\media\clips\b.mkv"]),
        ):
            summary_markup = gui.build_source_summary(str(audio_path), str(clip_dir))

        self.assertIn("Ready to analyze as MP3 audio.", summary_markup)
        self.assertIn("2 compatible clips found", summary_markup)

    def test_process_video_standard_delivery_path(self):
        session_state = {"seed": "value"}
        with (
            mock.patch.object(gui, "resolve_inputs", return_value=("C:\\audio.mp3", ["C:\\clips\\a.mp4"], session_state)),
            mock.patch.object(gui, "set_gpu_mode"),
            mock.patch.object(
                gui,
                "analyze_beats_auto",
                return_value=([0.0, 1.0, 2.0], {"tempo": 120.0, "times": [0.0, 1.0, 2.0], "selection_info": []}),
            ),
            mock.patch.object(gui, "create_music_video") as create_music_video,
            mock.patch.object(gui, "get_video_fps", return_value=30.0),
            mock.patch.object(gui, "get_video_resolution", return_value=(1920, 1080)),
            mock.patch.object(gui, "is_browser_playable_video", return_value=True),
            mock.patch.object(gui, "get_gpu_info", return_value="GPU Ready"),
            mock.patch.object(gui.datetime, "datetime", FixedDateTime),
        ):
            preview_path, status_text, returned_session = gui.process_video(
                audio_path="C:\\audio.mp3",
                video_folder="C:\\clips",
                generation_mode="auto",
                cut_intensity=4.0,
                smart_preset="normal",
                output_filename="deliverable.mp4",
                direction="forward",
                playback_speed_str="Normal Speed",
                timing_offset=0.0,
                parallel_workers=2,
                processing_mode="cpu",
                standard_quality="balanced",
                create_prores_delivery_mp4="no",
                custom_resolution="default",
                custom_fps=None,
                session_state=session_state,
            )

        self.assertTrue(preview_path.endswith(".mp4"))
        self.assertIn("Video created successfully!", status_text)
        self.assertIn("Target Resolution: 1920x1080 (auto-detected)", status_text)
        self.assertIs(returned_session, session_state)
        create_music_video.assert_called_once()

    def test_process_video_prores_delivery_copy_path(self):
        session_state = {"seed": "value"}
        with (
            mock.patch.object(gui, "resolve_inputs", return_value=("C:\\audio.mp3", ["C:\\clips\\a.mp4"], session_state)),
            mock.patch.object(gui, "set_gpu_mode"),
            mock.patch.object(
                gui,
                "analyze_beats_auto",
                return_value=([0.0, 1.0, 2.0], {"tempo": 120.0, "times": [0.0, 1.0, 2.0], "selection_info": []}),
            ),
            mock.patch.object(gui, "create_music_video"),
            mock.patch.object(gui, "create_lossless_delivery_mp4", return_value="C:\\output\\deliverable_delivery_lossless.mp4") as create_delivery_mp4,
            mock.patch.object(gui, "create_browser_preview", return_value="C:\\output\\deliverable_preview.mp4"),
            mock.patch.object(gui, "get_video_fps", return_value=24.0),
            mock.patch.object(gui, "get_video_resolution", return_value=(1280, 720)),
            mock.patch.object(gui, "get_gpu_info", return_value="GPU Ready"),
            mock.patch.object(gui.datetime, "datetime", FixedDateTime),
        ):
            preview_path, status_text, returned_session = gui.process_video(
                audio_path="C:\\audio.mp3",
                video_folder="C:\\clips",
                generation_mode="auto",
                cut_intensity=4.0,
                smart_preset="normal",
                output_filename="master.mov",
                direction="forward",
                playback_speed_str="Normal Speed",
                timing_offset=0.0,
                parallel_workers=2,
                processing_mode="prores_proxy",
                standard_quality="balanced",
                create_prores_delivery_mp4="yes",
                custom_resolution="1280x720",
                custom_fps=24.0,
                session_state=session_state,
            )

        self.assertEqual(preview_path, "C:\\output\\deliverable_preview.mp4")
        self.assertIn("Delivery MP4:", status_text)
        self.assertIn("Browser Preview:", status_text)
        self.assertIn("Target Resolution: 1280x720 (custom)", status_text)
        self.assertIs(returned_session, session_state)
        create_delivery_mp4.assert_called_once()


if __name__ == "__main__":
    unittest.main()
