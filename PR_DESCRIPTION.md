# Pull Request Description

## Suggested Title

Local-path workflow, render pipeline cleanup, and portable CUDA 12 workstation compatibility

## Summary

This PR combines the local desktop workflow refactor with the follow-up GPU/runtime fixes discovered during workstation validation.

The result is a branch that:

- uses local file and folder paths instead of upload staging
- avoids duplicating source media into BeatSync temp folders
- scans nested video folders recursively
- streamlines standard CPU/NVENC output assembly
- improves timing stability and startup capability checks
- adds safer GPU fallback behavior
- adds custom target-resolution presets with aspect-ratio-safe fit-and-pad output
- generates browser-friendly previews for non-browser-playable exports
- hardens HEVC segment retries and chunk assembly so recoverable per-clip failures do not kill the export
- makes the portable CUDA runtime version-aware instead of hard-coded to `v13.0`
- restores a Pascal-friendly portable stack by moving the recommended CUDA line to `12.9.x`

## Problem

The original workflow had two separate issues:

1. The GUI and processing flow created unnecessary disk churn by staging media through uploads and then copying source files again into BeatSync temp folders.
2. GPU validation on the GTX 1080 Ti workstation exposed that the portable `CUDA 13.x` plus `cupy-cuda13x` stack was not viable for Pascal GPU beat analysis, failing in CuPy/NVRTC during runtime kernel compilation.

Those two threads ended up touching the same startup/runtime surface, so this PR documents both the workflow refactor and the workstation compatibility work that followed.

## What Changed

### GUI and Input Handling

- Replaced GUI upload widgets with:
  - a local audio file path
  - a local video folder path
- Added native Windows `Browse...` pickers for both inputs.
- Normalized and validated local paths before processing.
- Changed video-folder discovery to search `.mp4` and `.mkv` clips recursively.
- Stopped copying source media into BeatSync session temp folders.
- Stored resolved source paths in session state.
- Configured Gradio to serve finished outputs directly from `output/`.
- Added target-resolution presets in the GUI for common `16:9`, `21:9`, and `9:16` outputs.

### Standard Render Pipeline

- Reworked standard CPU/NVENC exports around clearer planning, rendering, and assembly stages.
- Standard exports now:
  - render clips directly at final quality
  - assemble chunk outputs with stream copy
  - assemble the final video with stream copy
  - mux audio separately
- Added chunked assembly to reduce peak temp disk usage.
- Added CPU thread budgeting so parallel FFmpeg jobs do not oversubscribe all logical cores.
- Added standard quality presets:
  - `fast`
  - `balanced`
  - `high`
- Exposed matching quality controls in both GUI and CLI.

### Timing and Export Reliability

- Updated segment planning to use cumulative frame boundaries so long outputs stay aligned to the expected total duration.
- Fixed Auto Mode advanced structure analysis against the current `librosa` / `scikit-learn` stack.
- Improved mode/status logging so runtime reports match the actual selected processing path.
- Switched resizing to fit-and-pad instead of stretch so mixed-aspect-ratio source clips keep their proportions.
- Added browser-preview generation for non-browser-playable outputs so HEVC and ProRes finish cleanly in the GUI.
- Added better Windows startup behavior by resolving an open local Gradio port instead of failing when `7860` is busy.
- Added FFmpeg/ffprobe startup path setup so Gradio preview probing works with the portable FFmpeg bundle.

### GPU, NVENC, and Portable Runtime

- Added stronger boot-time GPU runtime probing so CUDA being importable is no longer treated as proof that GPU processing will actually work.
- Changed NVENC detection from a passive encoder-list check to a real runtime probe.
- Added beat-analysis CPU fallback in Manual, Smart, and Auto modes when GPU analysis fails at runtime.
- Added `analysis_device` reporting so the export/log layer can report whether analysis actually ran on GPU or CPU.
- Introduced `runtime_env.py` as a shared helper for portable runtime discovery and environment setup.
- Added version-aware portable CUDA discovery under `bin/CUDA/`.
- Added the `BEATSYNC_CUDA_DIR` override for explicit CUDA toolkit selection.
- Removed hard-coded `bin/CUDA/v13.0` assumptions from the app and launcher.
- Simplified `run.bat` so Python owns CUDA discovery.
- Shifted the recommended Pascal-compatible portable stack to:
  - `CUDA 12.9.x`
  - `cupy-cuda12x==13.6.0`

### ProRes and Preview Paths

- Improved ProRes preview generation so it:
  - writes previews to `output/`
  - follows the startup CPU-only vs GPU-mode decision
  - retries through CPU-safe fallbacks when the GPU preview path fails
  - only reports preview success when the preview file actually exists
- Added an optional lossless delivery MP4 secondary export while keeping the `.mov` master.
- Added the same browser-preview handling for standard non-browser-playable outputs such as HEVC delivery files.

### Segment Retry and Assembly Hardening

- Standard segment extraction now retries through safer fallback paths instead of failing immediately on the first NVENC issue.
- Added validation so rendered segments and chunk outputs must contain a readable video stream before they are accepted.
- Added chunk-assembly fallback from stream-copy concat to re-encode concat if stream-copy chunk assembly fails.
- Reduced retry log spam by summarizing recovered segment retries per chunk while keeping real per-clip failures detailed.

### Documentation and Validation Assets

- Updated `README.md` to describe:
  - the local-path workflow
  - standard quality presets
  - the revised processing pipeline
  - portable CUDA auto-discovery
  - the `BEATSYNC_CUDA_DIR` override
  - the recommended Pascal-compatible CUDA 12 stack
- Updated `TEST_TODO.md` with current workstation status, known failure history, and remaining GPU validation work.

## Files of Interest

- `gui.py`
- `video_processor.py`
- `ffmpeg_processing.py`
- `auto_mode.py`
- `smart_mode.py`
- `manual_mode.py`
- `runtime_env.py`
- `run.bat`
- `README.md`
- `TEST_TODO.md`
- `.gitignore`

## User-Facing Impact

- The GUI is now local-path based instead of upload based.
- Users get native Windows browse buttons for selecting the audio file and video folder.
- Source media is used in place instead of being duplicated into BeatSync temp folders.
- Standard exports should start faster, use less extra disk space, and avoid an unnecessary final re-encode.
- GPU/NVENC options only appear when the startup probe says they are truly usable.
- On Pascal GPUs, the documented portable CUDA recommendation is now `12.9.x`, not `13.x`.
- If a GPU analysis path fails unexpectedly, the app falls back cleanly to CPU instead of crashing.

## Breaking Change

Yes.

The GUI is now explicitly built around a local desktop workflow. That is a better fit for BeatSync's portable Windows use case, but it is not equivalent to the previous upload-oriented model.

## Why This Approach

This branch favors a larger but coherent fix over a narrow one-off patch.

The disk-usage and throughput improvements required more than just removing one copy step, and the workstation validation work made it clear that the runtime layer also needed cleanup:

- local-path flow removes redundant file staging
- standard assembly removes an unnecessary final re-encode
- startup probing prevents false-positive GPU/NVENC availability
- version-aware portable CUDA discovery avoids baking one toolkit version into the app
- CUDA 12 restores a viable Pascal path for portable CuPy GPU analysis

## Testing Performed

### Static Validation

- Portable Python compile pass completed successfully for the touched Python modules.
- Import checks completed successfully using the portable runtime.
- GUI startup import checks completed successfully after the runtime discovery changes.

### CPU-Oriented Validation From Earlier In The Branch

Validated on a CPU-only test machine using local sample media:

- CLI `manual` mode: passed
- CLI `smart` mode: passed
- CLI `auto` mode: passed
- GUI local-path CPU export: passed
- GUI ProRes export: passed
- GUI ProRes preview fallback path: passed
- Optional lossless delivery MP4 helper: passed

### Workstation Validation On March 25, 2026

Validated on the GTX 1080 Ti workstation:

- GUI startup detection: passed
  - GPU available
  - `CPU_ONLY_MODE=False`
  - NVENC startup probe passed
- Native `Browse...` buttons:
  - audio path selection passed
  - video folder selection passed
- Initial `Auto Mode` plus `NVIDIA NVENC H.264` run on the old `CUDA 13.x` stack:
  - failed during CuPy/NVRTC kernel compilation on Pascal
  - error signature matched `invalid value for --gpu-architecture (-arch)`
- CPU fallback patch:
  - prevented that failure from aborting analysis
- Portable runtime downgrade:
  - portable toolkit moved to `bin/CUDA/v12.9`
  - portable Python package moved from `cupy-cuda13x` to `cupy-cuda12x==13.6.0`
- Workstation GPU smoke tests after the downgrade:
  - CuPy runtime reported `12090`
  - CuPy JIT kernel probe passed
  - Manual beat analysis reported `analysis_device=gpu`
  - Smart beat analysis reported `analysis_device=gpu`
  - Auto beat analysis reported `analysis_device=gpu`
- Full GUI exports after the downgrade:
  - `Auto Mode` plus `NVIDIA NVENC H.264`: passed
  - `Auto Mode` plus `NVIDIA NVENC HEVC (H.265)`: passed
  - `Smart Mode` with NVENC/HEVC: passed
  - `Manual Mode` with NVENC/HEVC: passed
  - ProRes master export: passed
  - ProRes preview generation on the GPU-enabled workstation path: passed
- Mixed-source HEVC export stability follow-up:
  - initial per-segment NVENC failures were reproduced on some clips
  - retry and validation hardening now recovers those segments without skipped-clip output in the successful end-to-end test
- GUI polish follow-up:
  - non-browser-playable outputs now generate an explicit preview before `PROCESS COMPLETE`
  - the Windows Gradio socket-reset traceback after successful preview handling was suppressed
  - the app now falls forward to the next open localhost port when `7860` is already in use

## Remaining Validation

These still remain open in `TEST_TODO.md`:

- ProRes preview fallback when the preferred GPU preview path fails
- `Also create delivery MP4 (Lossless)` plus playback validation of the generated delivery file
- longer-track sync verification from start to finish
- custom-FPS validation in both standard and ProRes modes

## Risks and Review Notes

- This is a broad PR. Review it as a workflow/runtime branch rather than a single isolated bugfix.
- The GUI input model changed substantially, so reviewers should confirm the project wants to stay local-desktop-first.
- The standard render pipeline changed materially, so reviewers should pay close attention to:
  - concat assumptions
  - chunk assembly
  - final mux behavior
  - thread allocation logic
  - frame-accuracy at segment boundaries
- Mixed-source HEVC exports are much more robust now, but some clips can still trigger recoverable first-pass NVENC segment failures before the fallback chain succeeds.
- The runtime layer is better than before, but only portable path discovery is centralized today. Full capability-state centralization is still a follow-up refactor.
- The README and checklist now reflect a Pascal-friendly CUDA 12 recommendation; reviewers should treat that as intentional, based on the workstation failure against `CUDA 13.x`.

## Not In Scope

- Full redesign of the ProRes pipeline beyond preview reliability and optional delivery MP4 support
- Remote-safe upload semantics
- Final cleanup of all overlapping GPU/NVENC capability checks into one shared capability-state module

## Follow-Up Recommendations

- Finish the remaining real-media GUI validation items in `TEST_TODO.md` on the downgraded workstation stack.
- Complete the planned refactor to centralize full capability detection, not just runtime path discovery.
- Consider release notes or migration notes calling out:
  - the local-path GUI workflow
  - the new portable CUDA auto-discovery behavior
  - the Pascal recommendation to use `CUDA 12.9.x`
  - the new target-resolution presets and fit-and-pad output behavior

## Reviewer Checklist

- Confirm the local-path GUI workflow is the intended product direction.
- Confirm the chunked standard assembly path preserves expected output behavior.
- Confirm the quality preset defaults are acceptable.
- Confirm the runtime discovery behavior under `bin/CUDA/` is acceptable.
- Confirm the `BEATSYNC_CUDA_DIR` override is acceptable.
- Confirm the Pascal-oriented CUDA 12 recommendation is acceptable.
- Confirm the README and `TEST_TODO.md` reflect the current state of the branch.

## Optional PR Metadata

### Screenshots / Video

- Add an updated startup screenshot from the GPU workstation.
- Add one successful NVENC export log after the CUDA 12 re-test.

### Linked Issues

- Add issue references here if the upstream repo tracks this work.
