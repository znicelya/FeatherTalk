# Face Crop Geometry Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Extend `feathertalk-face` with deterministic Python-compatible square crop geometry.

**Architecture:** Add a focused crop module beside SCRFD resize, decode, and NMS. Reuse `ImageSize` and `FaceError`; return value types containing requested, source, padding, and origin geometry. No pixels or model state are owned.

**Tech Stack:** Rust 1.92 edition 2024, standard library, `thiserror`.

## Global Constraints

- Input bbox is `[x, y, width, height]` in source-image coordinates.
- Input image dimensions are non-zero.
- Bbox values and intermediate arithmetic must be finite and checked.
- The square size is `int(max(w, h) * 1.05)` after Python-compatible integer edges.
- Runtime dependencies remain only the standard library and `thiserror`.
- No OpenCV, image, Burn, WGPU, GPUI, FFmpeg, or model dependency is added.

---

### Task 1: Add deterministic face crop geometry

**Files:**
- Modify: `rust/crates/feathertalk-face/src/lib.rs`
- Modify: `rust/crates/feathertalk-face/src/error.rs`
- Create: `rust/crates/feathertalk-face/src/crop.rs`
- Create: `rust/crates/feathertalk-face/tests/crop.rs`
- Modify: `rust/crates/feathertalk-face/tests/public_api.rs`

**Interfaces:**
- Consumes existing `ImageSize` and `FaceError`.
- Produces `RectI`, `Padding`, `FaceCropGeometry`, and `compute_face_crop_geometry`.

- [ ] Write failing tests for normal square, wide/tall boxes, all four boundary cases, invalid image/bbox inputs, and public crate-root imports.
- [ ] Run `cargo test -p feathertalk-face --test crop`; expect failure because the crop module and API do not exist.
- [ ] Implement checked integer conversion, Python-compatible floor/truncation, centered square expansion, clipping, padding, and origin reporting.
- [ ] Export the new types and function from the crate root.
- [ ] Run `cargo fmt --check`, `cargo clippy -p feathertalk-face --all-targets --all-features -- -D warnings`, `cargo test -p feathertalk-face --all-targets`, and `git diff --check`.
- [ ] Commit with `feat: add deterministic face crop geometry`.

## Self-Review

- The module computes geometry only; it does not allocate or mutate image buffers.
- Existing SCRFD decode/NMS behavior remains unchanged.
- Origin semantics are explicit so PFLD output mapping can consume them without reimplementing padding logic.
