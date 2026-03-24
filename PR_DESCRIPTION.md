# Pull Request Description

## Suggested Title

Local-path GUI workflow and render pipeline refactor to reduce disk usage and improve throughput

## Summary

This PR replaces the Gradio upload-based media flow with a local-path workflow and refactors the standard render pipeline to reduce duplicate disk usage, lower peak temp usage, and remove an unnecessary final video re-encode.

## Problem

The previous workflow had two major cost centers:

- The GUI upload flow staged media into Gradio-managed files.
- BeatSync then copied source media again into session temp folders under `temp/`.

For large source libraries, this could dramatically increase disk usage and startup time before rendering even began.

The standard export path also rendered temp clips and then performed a full final video re-encode, which added extra processing time and another quality-loss step for standard delivery formats.

## What Changed

### GUI and Input Handling

- Replaced GUI upload widgets with:
  - a local audio file path
  - a local video folder path
- Added native Windows `Browse...` pickers that write the selected local path back into the text fields without reintroducing upload copies.
- Added path normalization and validation for local input paths.
- Removed BeatSync source-media copying into `temp/session_*`.
- Stored resolved source paths in session state instead of copied files.
- Configured Gradio to serve files directly from `output/`.

### Standard Render Pipeline

- Split the processing flow into clearer stages around input resolution, segment planning, and render/assembly.
- Standard CPU/NVENC exports now:
  - render segments directly at final quality
  - concatenate chunk outputs with stream copy
  - concatenate the final video with stream copy
  - mux the audio track separately
- Added chunked assembly to reduce peak temp disk usage.
- Added CPU thread budgeting so parallel workers do not oversubscribe all CPU threads per FFmpeg job.

### Quality and CLI

- Added standard export quality presets:
  - `fast`
  - `balanced`
  - `high`
- Exposed `--quality {fast,balanced,high}` in the CLI.
- Added matching GUI quality controls for standard exports.

### Timing and Stability Fixes

- Updated segment planning to use cumulative frame boundaries so final output duration remains frame-accurate across the full timeline.
- Fixed auto-mode advanced song-structure analysis so it no longer fails against the current `librosa` / `scikit-learn` stack.
- Improved GPU availability detection so the app falls back cleanly to CPU when CUDA is importable but not actually usable.
- Changed NVENC detection from a simple encoder-list check to a real runtime probe at startup.
- Improved ProRes preview generation to:
  - write previews to `output/`
  - follow the boot-time CPU-only / GPU-mode decision
  - retry through CPU-compatible fallback paths if GPU preview generation fails
  - only report preview success when the preview file actually exists
- Added an optional ProRes secondary export path:
  - `Also create delivery MP4 (Lossless)`
  - keeps the `.mov` master
  - creates a second lossless `.mp4` delivery file when enabled
- Cleaned up runtime logging so the selected generation mode is reported correctly.

### Documentation

- Updated the README to describe the new local-path workflow, standard quality presets, current portable runtime expectations, and the revised processing pipeline.

## Files of Interest

- `gui.py`
- `video_processor.py`
- `ffmpeg_processing.py`
- `smart_mode.py`
- `ui_content.py`
- `README.md`
- `TEST_TODO.md`
- `.gitignore`

## User-Facing Impact

- The GUI now expects local file/folder paths instead of uploaded files.
- The GUI now includes native Windows browse buttons for the local-path workflow.
- Source media is used directly from its original location.
- Standard exports should start faster, use less extra disk space, and avoid an unnecessary final video re-encode.
- ProRes preview behavior is more reliable on systems where NVENC is present in FFmpeg but not usable at runtime.
- ProRes mode can optionally create a second lossless MP4 delivery file while keeping the `.mov` master.

## Breaking Change

Yes.

The GUI is now explicitly local-path based and assumes a local desktop workflow. This is a better fit for BeatSync's portable desktop usage, but it is not suitable for a remote-hosted or browser-only upload workflow without additional changes.

## Why This Approach

This PR takes the larger refactor rather than a narrow "stop copying uploads" patch because the upload path itself was only part of the cost. The biggest practical win comes from addressing both sides of the problem:

- eliminate duplicated source-media staging
- remove the extra final standard-video re-encode

That gives a meaningful reduction in both disk usage and render time.

## Testing Performed

### Static Validation

- Portable Python compile pass completed successfully for the main modules.
- Import checks completed successfully using the portable runtime.
- `video_processor.py -h` completed successfully.
- GUI construction (`create_ui()`) completed successfully after the later browse/runtime changes.

### Live Runtime Validation

Validated on a CPU-only test machine using local sample media:

- CLI `manual` mode: passed
- CLI `smart` mode: passed
- CLI `auto` mode: passed
- GUI local-path CPU export: passed
- GUI ProRes export: passed
- GUI ProRes preview fallback path: passed

Validated with targeted regression/probe scripts:

- Auto-mode advanced structure analysis: passed after clustering fix
- ProRes preview helper on CPU-only machine: passed
- Optional lossless delivery MP4 helper: passed

### Behavior Verified

- No source-media copies were created under `temp/session_*`.
- Session temp directories were cleaned up after successful runs.
- Standard export outputs had the expected audio/video streams and durations.
- Auto-mode output duration landed exactly on the expected frame-aligned total after the cumulative-frame fix.
- ProRes output produced `.mov` with H.264 preview written into `output/`.

## Environment Notes

- CUDA was not usable on the validation machine because the installed driver/runtime combination reported `cudaErrorInsufficientDriver`.
- CPU paths were fully exercised.
- GPU/NVENC full end-to-end validation is still needed on a compatible machine.
- A workstation follow-up checklist now lives in `TEST_TODO.md`.

## Risks and Review Notes

- The GUI input model changed substantially, so reviewers should focus on whether the project wants to remain local-desktop-first.
- The native browse buttons are Windows-specific and intentionally align with the project's portable desktop usage.
- The render pipeline changed materially in standard mode, so review should pay close attention to:
  - concat compatibility assumptions
  - final mux behavior
  - thread allocation logic
  - frame-accuracy at segment boundaries
- `ui_content.py` was simplified while updating the workflow text, so reviewers may want to compare any removed copy or labels they consider important.

## Not In Scope

- Full redesign of the ProRes pipeline beyond shared helpers and preview reliability improvements
- GPU validation on a machine with working CUDA drivers
- Remote-safe file upload semantics

## Follow-Up Recommendations

- Run a GPU/NVENC validation pass on a machine with compatible NVIDIA drivers.
- Consider additional documentation or release notes calling out the GUI workflow change.

## Reviewer Checklist

- Confirm the local-path GUI model is acceptable for the upstream project direction.
- Confirm the chunked standard assembly logic preserves expected output behavior.
- Confirm the new quality presets and defaults are acceptable.
- Confirm the README reflects the intended supported workflow.
- Confirm whether `.gitignore` changes for `bin/` and `__pycache__/` should be included in the upstream PR.

## Optional PR Metadata

### Screenshots / Video

- Add updated UI screenshots showing the local path inputs and quality selector.

### Linked Issues

- Add issue references here if the upstream repo tracks this problem.
