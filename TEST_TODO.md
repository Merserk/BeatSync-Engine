# Workstation Test TODO

Use this checklist on a machine with working NVIDIA drivers and CUDA/NVENC support.

## Priority Tests

- [ ] Confirm boot-time detection shows GPU available and `CPU_ONLY_MODE = False`.
- [ ] Confirm boot-time NVENC probe passes and NVENC options appear in the GUI.
- [ ] Verify native `Browse...` buttons still work on the workstation build for:
  - audio file selection
  - video folder selection
- [ ] Run a full GUI export in `Auto Mode` with `NVIDIA NVENC H.264`.
- [ ] Run a full GUI export in `Auto Mode` with `NVIDIA NVENC HEVC (H.265)`.
- [ ] Run at least one `Smart Mode` export with NVENC enabled.
- [ ] Run at least one `Manual Mode` export with NVENC enabled.

## ProRes Validation

- [ ] Run a ProRes export and confirm the `.mov` master is created successfully.
- [ ] Confirm ProRes preview generation prefers CUDA/NVENC when GPU mode is available.
- [ ] Confirm ProRes preview still falls back cleanly if the GPU preview path fails.
- [ ] Test `Also create delivery MP4 (Lossless)` and confirm both files are produced:
  - `.mov` master
  - `_delivery_lossless.mp4`
- [ ] Open the generated lossless delivery MP4 and confirm playback is correct.

## Behavior Checks

- [ ] Confirm no source media is copied into `temp/` during normal processing.
- [ ] Confirm standard exports still clean up session temp folders after success.
- [ ] Confirm output timing stays in sync from start to finish on a longer track.
- [ ] Confirm custom FPS still works as expected in both standard and ProRes modes.
- [ ] Confirm absolute manual paths and browsed paths both resolve correctly.

## Things To Watch For

- [ ] NVENC listed in FFmpeg but still failing at runtime.
- [ ] CUDA decode issues during preview generation.
- [ ] Browser playback issues with generated preview MP4 files.
- [ ] Unexpected extra disk usage when ProRes preview and optional delivery MP4 are both enabled.
- [ ] Any mismatch between GUI status text and actual output files on disk.

## Suggested Evidence To Capture

- [ ] Screenshot of startup status on the GPU workstation.
- [ ] One successful NVENC H.264 export log.
- [ ] One successful NVENC HEVC export log.
- [ ] One successful ProRes export log with optional delivery MP4 enabled.
