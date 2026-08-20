# Preprocess Geometry Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent `feathertalk-preprocess` crate for strict 68-point landmark parsing, deterministic face geometry, fixed crop/mask constants, and eight-frame audio-window indices.

**Architecture:** Separate modules handle landmark parsing, face geometry, and audio-window indexing. The crate owns no image/audio tensors and performs no media or model operations.

**Tech Stack:** Rust 1.92 edition 2024, standard library, `thiserror`, and `tempfile` for tests.

## Global Constraints

- Workspace member: `rust/crates/feathertalk-preprocess`.
- Runtime dependencies: only the standard library and `thiserror`; `tempfile` is test-only.
- No dependency on media, project, model, image/OpenCV, Burn, WGPU, GPUI, or FFmpeg crates.
- Landmark files contain exactly 68 non-empty `x y` points; reject malformed, extra-token, non-finite, negative, invalid-UTF-8, and wrong-count input.
- Preserve point indices `1`, `31`, `52`, crop `168`, inner `160`, border `4`, mask `(5,5,150,145)`.
- Audio windows contain 8 slots centered on the current frame with half-window 4; boundary slots are `None`.

---

### Task 1: Bootstrap types and strict landmark parsing

**Files:** modify `rust/Cargo.toml` and generated `rust/Cargo.lock`; create `rust/crates/feathertalk-preprocess/Cargo.toml`, `src/lib.rs`, `src/error.rs`, `src/landmarks.rs`, and `tests/landmarks.rs`.

**Produces:** `Point`, `Landmarks`, `PreprocessError`, and `read_landmarks(&Path) -> Result<Landmarks, PreprocessError>`; `Landmarks::points() -> &[Point]` is read-only.

- [ ] Write tests for: valid 68 points with blank-line skipping; malformed/extra-token line; 67-point count; `NaN`; negative coordinate; invalid UTF-8; missing file.
- [ ] Run `cargo test -p feathertalk-preprocess --test landmarks`; expect failure because the crate/parser do not exist.
- [ ] Add the workspace member and manifest with only `thiserror` plus test-only `tempfile`.
- [ ] Implement `read_landmarks` with `fs::read`, UTF-8 conversion, one-based line numbers, `split_whitespace`, exactly two float tokens, `is_finite`, non-negative checks, and final count exactly 68.
- [ ] Define errors: `Io { operation, path, source }`, `InvalidUtf8 { path }`, `InvalidLine { path, line, message }`, `WrongLandmarkCount { path, expected, actual }`, `NonFiniteCoordinate { path, line }`, `NegativeCoordinate { path, line }`, `InvalidGeometry { field, message }`, and `FrameIndexOutOfRange { frame_index, frame_count }`.
- [ ] Run `cargo test -p feathertalk-preprocess --test landmarks`; expect all parser tests to pass.
- [ ] Commit with `feat: parse strict landmark files`.

### Task 2: Implement deterministic face geometry and crop constants

**Files:** create `src/geometry.rs` and `tests/geometry.rs`; modify `src/lib.rs`.

**Consumes:** `Landmarks`, `Point`, and `PreprocessError` from Task 1. **Produces:** `FaceBoundingBox`, `MaskRect`, `CropSpec`, `compute_face_bbox(&Landmarks) -> Result<FaceBoundingBox, PreprocessError>`, and `default_crop_spec() -> CropSpec`.

- [ ] Write tests for bbox `{ xmin: 10, ymin: 20, xmax: 50, ymax: 60 }` using points 1/31/52, invalid non-positive width, exact crop constants, and `crop_size == inner_size + 2 * border`.
- [ ] Run `cargo test -p feathertalk-preprocess --test geometry`; expect failure because geometry types/functions do not exist.
- [ ] Implement `xmin = int(point[1].x)`, `ymin = int(point[52].y)`, `xmax = int(point[31].x)`, `width = xmax - xmin`, reject `width <= 0`, and return `ymax = ymin + width`.
- [ ] Return crop `168`, inner `160`, border `4`, mouth mask `MaskRect { x: 5, y: 5, width: 150, height: 145 }`; derive equality for tests and keep access read-only.
- [ ] Run `cargo fmt --all; cargo test -p feathertalk-preprocess --test geometry`.
- [ ] Commit with `feat: add preprocess face geometry contract`.

### Task 3: Add audio-window indices and public API acceptance

**Files:** create `src/audio_window.rs`, `tests/audio_window.rs`, and `tests/public_api.rs`; modify `src/lib.rs`.

**Produces:** `audio_window_indices(frame_index: usize, frame_count: usize) -> Result<[Option<usize>; 8], PreprocessError>`.

- [ ] Write tests asserting windows for `(0,10)`, `(5,10)`, and `(9,10)`, plus errors for `(0,0)` and `(10,10)`.
- [ ] Run `cargo test -p feathertalk-preprocess --test audio_window`; expect failure because the function does not exist.
- [ ] Implement rejection for empty/out-of-range frame counts and offsets `-4..4`, returning `Some(index)` in range and `None` at boundaries.
- [ ] Add crate-root API coverage that parses a temporary 68-point file, calls bbox/crop/window APIs, and binds `Landmarks::points()` to `&[Point]` without private-module access.
- [ ] Run `cargo fmt --check`, `cargo clippy -p feathertalk-preprocess --all-targets --all-features -- -D warnings`, `cargo test -p feathertalk-preprocess --all-targets`, and `git diff --check`; expect all to exit 0.
- [ ] Commit with `feat: add preprocess audio window contract`.

## Plan Self-Review

- Spec coverage: parsing, 68-point count, structured errors, geometry indices/truncation, exact crop constants, audio boundaries, read-only API, tests, and dependency exclusions map to Tasks 1-3.
- Placeholder scan: no TODO, TBD, or undefined implementation steps remain.
- Type consistency: every later-task type and signature is produced by an earlier task.
- Scope: no media decoding, pixel operations, model inference, tensor allocation, manifest writes, or GPU dependencies are introduced.
