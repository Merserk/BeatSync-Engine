# BeatSync Engine

BeatSync Engine is a portable Windows app that builds beat-synced music videos from a local audio file and a folder of local video clips.

The current workflow is local-path based. The GUI does not upload and duplicate your source media into a Gradio cache or BeatSync session folder before processing. Source files are read in place, rendered clips are created under `temp/`, and finished outputs are written to `output/`.

## Highlights

- Local-path workflow for lower disk usage and faster startup
- Native Windows `Browse...` pickers for audio files and video folders
- Manual, Smart, and Auto beat selection modes
- Frame-based segment planning for stable output duration
- Standard H.264/NVENC export with chunked rendering and stream-copy assembly
- ProRes 422 Proxy precise mode with browser-friendly preview generation
- Optional ProRes secondary export: lossless delivery MP4
- Boot-time GPU/NVENC runtime probing so unusable GPU paths fall back cleanly
- Portable runtime support for Python, CUDA, and FFmpeg under `bin/`

## Supported Media

- Audio input: `.mp3`, `.wav`, `.flac`
- Video source folder: `.mp4`, `.mkv`

## Portable Layout

The repo expects these tools under `bin/`:

- `bin/python-3.13.9-embed-amd64/python.exe`
- `bin/CUDA/v13.0`
- `bin/ffmpeg/ffmpeg.exe`
- `bin/ffmpeg/ffprobe.exe`

`run.bat`, `gui.py`, and `video_processor.py` are wired to those paths.

## Launching The App

1. Place the portable runtime files under `bin/`.
2. Run `run.bat`.
3. The Gradio UI starts locally in your browser.

## GUI Workflow

1. Enter a local audio file path or use `Browse...`.
2. Enter a local video folder path or use `Browse...`.
3. Choose a generation mode:
   - `manual`: bass-focused beat detection
   - `smart`: multi-band beat selection with presets
   - `auto`: structure-aware automatic selection
4. Choose a processing mode:
   - `CPU (H.264)`
   - `NVIDIA NVENC H.264`
   - `NVIDIA NVENC HEVC (H.265)`
   - `ProRes 422 Proxy (Precise Mode)`
   NVENC options only appear when the boot-time runtime probe confirms they are usable.
5. For standard CPU/NVENC exports, choose a quality preset:
   - `fast`
   - `balanced`
   - `high`
6. In ProRes mode, you can optionally enable `Also create delivery MP4 (Lossless)` to keep the `.mov` master and create a second lossless `.mp4`.
7. Optionally adjust direction, playback speed, timing offset, worker count, FPS, and output filename.
8. Click `Create Music Video`.

Finished files are written to `output/`.

## Processing Pipelines

### Standard Export

Standard CPU and NVENC exports now use this pipeline:

1. Plan all segments in frames
2. Render each segment directly at the final quality
3. Concatenate segment chunks with stream copy
4. Concatenate chunk files with stream copy
5. Mux the final audio track

This avoids the older "encode every segment, then fully re-encode the final video again" workflow.

### ProRes Precise Mode

ProRes mode is the quality-first path:

1. Convert only the source videos actually used by the segment plan to ProRes 422 Proxy
2. Extract frame-accurate ProRes segments
3. Concatenate them losslessly
4. Write the final `.mov` to `output/`
5. Optionally create a lossless `.mp4` delivery copy
6. Generate an H.264 preview in `output/` for UI playback

## Disk Usage Notes

- Source audio and source video files are used directly from their original locations.
- BeatSync still uses `temp/` for render intermediates and ProRes working files.
- Standard exports now process in chunks, which lowers peak temp usage compared with keeping every rendered clip around until the very end.
- Session temp folders are cleaned up after successful runs.

## CPU And GPU Notes

- If CUDA is unavailable or unusable, audio analysis falls back to CPU automatically.
- The app determines CPU-only mode at boot and only exposes GPU/NVENC paths when the runtime probe succeeds.
- NVENC is not treated as available just because FFmpeg lists the encoder; BeatSync now performs a real startup probe.
- ProRes preview generation prefers CUDA/NVENC when the app is not in CPU-only mode, then falls back cleanly to CPU-compatible paths if needed.
- The optional ProRes delivery MP4 is a secondary export, not a replacement for the `.mov` master.

## CLI Usage

Basic usage:

```bash
python video_processor.py <audio_file> <video_directory> <cut_intensity> [options]
```

Examples:

Manual mode:

```bash
python video_processor.py "C:\Music\track.wav" "D:\Clips" 2.0 --mode manual --quality fast -o "manual.mp4"
```

Smart mode:

```bash
python video_processor.py "C:\Music\track.wav" "D:\Clips" normal --mode smart --quality balanced -o "smart.mp4"
```

Auto mode:

```bash
python video_processor.py "C:\Music\track.wav" "D:\Clips" auto --mode auto --quality high -o "auto.mp4"
```

Lossless ProRes mode:

```bash
python video_processor.py "C:\Music\track.wav" "D:\Clips" 2.0 --mode manual --lossless -o "precise.mov"
```

Useful options:

- `--mode {manual,smart,auto}`
- `--quality {fast,balanced,high}`
- `--direction {forward,backward,random}`
- `--offset <seconds>`
- `--fps <value>`
- `--gpu`
- `--gpu-encoder {h264_nvenc,hevc_nvenc,none}`
- `--lossless`

Run `python video_processor.py -h` for the full argument list.

## Repository Notes

- `bin/` is ignored by Git for local portable runtime files.
- `.vscode/` is ignored by Git for local IDE settings.
- Outputs are written to `output/`.
- Temporary processing files are written to `temp/`.

## Validation Notes

- A workstation follow-up checklist for GPU/NVENC validation lives in `TEST_TODO.md`.

## License

AGPL-3.0. See `LICENSE`.
