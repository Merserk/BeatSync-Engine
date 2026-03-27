## Summary

This PR covers the UI, documentation, and regression-testing work completed after `dev/PR_DESCRIPTION_26.03.26.md`.

It is intentionally limited to the current Gradio application surface and its supporting docs/tests.

The branch now:

- refines BeatSync into a darker application-style desktop UI
- improves interaction reliability around ProRes delivery options
- adds visible render-start feedback on the primary action
- cleans up duplicated UI copy/helpers
- documents the current UX more clearly in `README.md`
- adds a rerunnable regression suite under `dev/`

## Problem

The post-PR UI iteration exposed three follow-up gaps:

1. The Gradio surface still behaved too much like a generic webpage instead of an application workspace.
2. Some UI controls were unreliable or unclear in use, especially the ProRes delivery-copy option and the main render button state.
3. There was no tracked regression suite covering the current GUI helper/state logic, so every UI pass depended on manual retesting.

## What Changed

### Application UI

- Kept the current app in permanent dark mode and continued refining the desktop application styling.
- Locked the visible surface away from Gradio settings/API chrome so accidental theme changes do not leak back into the app.
- Preserved the current application-oriented sectioning and status/preview inspector layout.
- Added immediate button-state feedback when a render starts:
  - `Create Music Video` changes to `Creating...`
  - the button becomes non-interactive during the run
  - a short render-start status message is shown immediately

### ProRes Delivery Option

- Replaced the unreliable ProRes delivery-copy checkbox with a deterministic `Yes / No` radio control.
- Normalized the radio value back to a boolean in the processing path so backend behavior stays unchanged.
- Kept the ProRes-only visibility rules tied to the selected export pipeline.

### Code Cleanup

- Removed the duplicate/overridden UI copy blocks from `ui_content.py` and kept one active source of truth for labels and status text.
- Marked the old `create_ui_legacy()` path as a compatibility alias to the active UI so it cannot diverge from the real application surface.
- Added a tracked regression module under `dev/test_regression.py`.
- Updated `.gitignore` so `dev/*.md` and `dev/test_*.py` are trackable instead of being swallowed by the blanket `dev/` ignore rule.

### README

- Updated the README to describe the current application-style Gradio UI.
- Documented the `Yes / No` ProRes delivery-copy control.
- Documented the immediate render-start feedback behavior.
- Added the current regression test command and a short summary of what it covers.

## Files Of Interest

- `gui.py`
- `ui_content.py`
- `README.md`
- `.gitignore`
- `dev/test_regression.py`

## User-Facing Impact

- The UI reads more clearly as an application workspace than a generic form page.
- Users now get immediate confirmation that a render request was accepted.
- The ProRes delivery-copy option is more reliable and easier to understand.
- The repo now contains a repeatable regression command for the current GUI helper/state surface.

## Testing Performed

### Static Validation

- Portable Python compile check:
  - `gui.py`
  - `ui_content.py`
  - `dev/test_regression.py`

### Regression Suite

Executed with the portable runtime:

```bash
bin\python-3.13.9-embed-amd64\python.exe -m unittest discover -s dev -p "test_*.py" -v
```

Covered cases:

- `create_ui()` construction
- processing-mode UI state switching
- render-start button feedback helpers
- source preflight summary behavior
- standard delivery orchestration glue
- ProRes master plus optional delivery MP4 orchestration glue

### Notes

- The regression run passes, but Gradio on Windows still emits a non-failing `ResourceWarning` about an unclosed Proactor event loop during the UI-build test.

## Risks And Review Notes

- The active app surface is now the clear source of truth, but `gui.py` is still large and would benefit from a later extraction of UI helper/render-building code into smaller modules.
- The regression suite focuses on orchestration and UI state, not on full media-processing end-to-end renders with real sample assets.
- The ProRes delivery selector changed component type from checkbox to radio intentionally, to avoid the Gradio checkbox interaction issue seen in this branch.

## Not In Scope

- Changes to the core beat-analysis algorithms
- Changes to the standard or ProRes render engines beyond UI wiring and option normalization
- A full module split of `gui.py`

## Reviewer Checklist

- Confirm the darker application-style surface is acceptable as the default GUI direction.
- Confirm the `Yes / No` ProRes delivery selector is clearer than the previous checkbox.
- Confirm the immediate render-start feedback matches expected UX.
- Confirm the README reflects the current product surface.
- Confirm the new regression suite is a reasonable baseline for future GUI changes.
