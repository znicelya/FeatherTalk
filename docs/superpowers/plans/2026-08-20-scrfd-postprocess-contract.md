# SCRFD Postprocess Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent `feathertalk-face` crate for SCRFD resize geometry, anchor generation, detection decoding, coordinate mapping, and deterministic NMS.

**Architecture:** Split preprocessing geometry, level decoding, and NMS into focused modules. The crate accepts numerical slices supplied by a later model adapter and owns no image buffers or model runtime state.

**Tech Stack:** Rust 1.92 edition 2024, standard library, `thiserror`, and `tempfile` for tests.

## Global Constraints

- Workspace member: `rust/crates/feathertalk-face`.
- Runtime dependencies are only the standard library and `thiserror`; `tempfile` is test-only.
- No OpenCV, image, ONNX Runtime, Burn, WGPU, GPUI, `feathertalk-preprocess`, or model dependencies.
- Model input is exactly `640x640`; strides are exactly `8, 16, 32`; anchors per location exactly `2`.
- Default confidence threshold is `0.1`; default NMS IoU threshold is `0.5`.
- All caller-provided numerical input is finite-checked and length-checked; no public API panics.
- Bboxes use `[x, y, width, height]` in original-image coordinates.

---

### Task 1: Bootstrap types, resize transform, and anchor centers

**Files:** modify `rust/Cargo.toml` and generated `rust/Cargo.lock`; create `rust/crates/feathertalk-face/Cargo.toml`, `src/lib.rs`, `src/error.rs`, `src/preprocess.rs`, and `tests/preprocess.rs`.

**Produces:** `ImageSize`, `ResizeTransform`, `FaceError`, `resize_with_padding`, and `generate_anchor_centers`.

- [ ] Write tests for square `640x640`, portrait, landscape with odd padding, zero dimensions, and exact stride anchor counts/order for `8/16/32`.
- [ ] Run `cargo test -p feathertalk-face --test preprocess`; expect failure because the crate/functions do not exist.
- [ ] Add the workspace member and manifest with only `thiserror` plus test-only `tempfile`.
- [ ] Implement `resize_with_padding`: reject zero dimensions; use fixed model `640x640`; square stays `640x640`; portrait uses `new_height=640`, `new_width=floor(640/(height/width))`; landscape uses `new_width=640`, `new_height=floor(640*(height/width))+1`; split padding with floor on left/top and remainder on right/bottom; set `scale_x=input.width/new_width`, `scale_y=input.height/new_height`.
- [ ] Implement `generate_anchor_centers`: accept only model `640x640`, strides `8/16/32`, anchors `2`; emit row-major y-then-x `[x*stride,y*stride]`, repeating each center twice.
- [ ] Run `cargo fmt --all; cargo test -p feathertalk-face --test preprocess`; expect all geometry tests to pass.
- [ ] Commit with `feat: add SCRFD resize and anchor geometry`.

### Task 2: Decode levels and map detections to original coordinates

**Files:** create `src/decode.rs` and `tests/decode.rs`; modify `src/lib.rs` and `src/error.rs`.

**Consumes:** Task 1 `ImageSize`, `ResizeTransform`, and `FaceError`. **Produces:** `Detection` and `decode_level`.

- [ ] Write tests for exact bbox/keypoint decoding, tensor length mismatch, non-finite scores/distances, padding/scale mapping, clipping to source bounds, and invalid/non-positive decoded area.
- [ ] Run `cargo test -p feathertalk-face --test decode`; expect failure because `Detection` and `decode_level` do not exist.
- [ ] Define `Detection { bbox: [f32;4], score: f32, keypoints: [[f32;2];5] }` and errors `InvalidTensorLength { level, field, expected, actual }`, `NonFiniteValue { level, field, index }`, `InvalidDetectionGeometry { index }`.
- [ ] Require all four input slices to have equal lengths; check every score and distance is finite.
- [ ] Decode bbox with `x1=cx-left*stride`, `y1=cy-top*stride`, `x2=cx+right*stride`, `y2=cy+bottom*stride`; decode each keypoint as `[cx+dx*stride, cy+dy*stride]`.
- [ ] Map model coordinates using `(value-pad)*scale`, clip bbox and keypoints to source bounds, return bbox `[x1,y1,x2-x1,y2-y1]`, and reject non-positive clipped area.
- [ ] Run `cargo fmt --all; cargo test -p feathertalk-face --test decode`; expect all decode tests to pass.
- [ ] Commit with `feat: decode SCRFD detection levels`.

### Task 3: Implement deterministic NMS and public acceptance

**Files:** create `src/nms.rs`, `tests/nms.rs`, and `tests/public_api.rs`; modify `src/lib.rs`.

**Consumes:** `Detection` and `FaceError` from Task 2. **Produces:** `DetectionConfig`, `non_max_suppression`.

- [ ] Write tests for default thresholds, low-score filtering, overlapping suppression, non-overlap retention, threshold equality, equal-score stable index ordering, invalid thresholds, non-finite values, and non-positive geometry.
- [ ] Run `cargo test -p feathertalk-face --test nms`; expect failure because NMS types/functions do not exist.
- [ ] Implement `DetectionConfig::default()` as `0.1` and `0.5`; reject non-finite or out-of-range thresholds.
- [ ] Validate scores and `[x,y,width,height]`, filter scores strictly below threshold, sort by score descending then original index ascending, and suppress candidates whose continuous IoU is strictly greater than the threshold.
- [ ] Return original input indices in deterministic keep order.
- [ ] Add crate-root public API coverage for value types and all four public functions without private-module access.
- [ ] Run `cargo fmt --check`, `cargo clippy -p feathertalk-face --all-targets --all-features -- -D warnings`, `cargo test -p feathertalk-face --all-targets`, and `git diff --check`; expect all to exit 0.
- [ ] Commit with `feat: add deterministic SCRFD NMS contract`.

## Plan Self-Review

- Spec coverage: fixed detector parameters, resize/padding, anchor order/count, decode formulas, coordinate mapping/clipping, finite/length validation, NMS semantics, stable errors, public API, and dependency exclusions map to Tasks 1-3.
- Placeholder scan: no TODO, TBD, or undefined implementation steps remain.
- Type consistency: every type and function used by later tasks is produced by an earlier task with matching names and signatures.
- Scope: no model execution, image manipulation, PFLD, frame anomaly logic, or asset writes are introduced.
