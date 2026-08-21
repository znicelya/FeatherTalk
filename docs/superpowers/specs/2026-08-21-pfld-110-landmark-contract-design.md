# PFLD 110-Point Landmark Contract Design

Date: 2026-08-21  
Status: Approved for implementation

## Purpose

Align the Rust landmark file contract with the existing Python PFLD pipeline. The Python model outputs 110 landmark points (220 normalized coordinates), adds `mean_face.txt`, maps them into the detector crop, and writes all 110 points to each `.lms` file.

## Scope

Included:

- Change strict `.lms` parsing from exactly 68 points to exactly 110 points.
- Preserve finite, non-negative coordinate validation and line-aware errors.
- Preserve face geometry indices used by Python: points `1`, `31`, and `52`.
- Preserve crop, mouth-mask, and audio-window constants.
- Update public API and tests to document the 110-point contract.

Excluded:

- PFLD model execution or Burn integration.
- Checkpoint or `mean_face.txt` conversion.
- Image resizing, crop extraction, or pixel buffers.
- A 110-to-68 landmark mapping.

## Public Contract

`read_landmarks(&Path)` returns a read-only `Landmarks` value containing exactly 110 points. Blank lines are ignored; every non-empty line contains exactly two finite, non-negative float tokens. Wrong counts and malformed values return `PreprocessError`.

`compute_face_bbox` continues to use point indices 1, 52, and 31 exactly as the Python implementation does. The existing `CropSpec`, `MaskRect`, and `audio_window_indices` contracts do not change.

## Acceptance

- Landmark tests use 110 points and reject 109/111 points.
- Geometry tests construct 110-point values while retaining the same indexed points.
- Public API tests continue to use only crate-root exports.
- `cargo fmt --check`, `cargo clippy -p feathertalk-preprocess --all-targets --all-features -- -D warnings`, `cargo test -p feathertalk-preprocess --all-targets`, and `git diff --check` pass.
