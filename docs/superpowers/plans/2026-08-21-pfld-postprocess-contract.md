# PFLD Numeric Postprocess Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add an independent `feathertalk-pfld` crate that maps validated PFLD output vectors to 110 integer landmark points.

**Architecture:** Keep the crate numerical and model-free. A small decode module validates vectors and crop geometry, performs Python-compatible addition, scaling, truncation, and offset mapping, and returns an immutable value type.

**Tech Stack:** Rust 1.92 edition 2024, standard library, `thiserror`, and `tempfile` for tests.

## Global Constraints

- Model output and mean-face vectors contain exactly 220 finite `f32` values.
- The output contains exactly 110 `(x, y)` landmark pairs.
- Crop width and height are positive; offsets are signed `i32` values.
- Coordinate conversion truncates toward zero before adding offsets.
- No Burn, WGPU, GPUI, image, FFmpeg, checkpoint, or model dependencies.

---

### Task 1: Bootstrap the PFLD numeric postprocess crate

**Files:**
- Modify: `rust/Cargo.toml` and generated `rust/Cargo.lock`
- Create: `rust/crates/feathertalk-pfld/Cargo.toml`
- Create: `rust/crates/feathertalk-pfld/src/lib.rs`
- Create: `rust/crates/feathertalk-pfld/src/error.rs`
- Create: `rust/crates/feathertalk-pfld/src/decode.rs`
- Create: `rust/crates/feathertalk-pfld/tests/decode.rs`
- Create: `rust/crates/feathertalk-pfld/tests/public_api.rs`

**Interfaces:**
- Produces `CropGeometry`, `LandmarkPoint`, `PFLDLandmarks`, `PfldError`, `PFLD_OUTPUT_VALUE_COUNT`, `PFLD_LANDMARK_COUNT`, and `decode_landmarks`.
- Depends only on the standard library and `thiserror`.

- [ ] Write failing tests for exact mapping, negative offsets, truncation toward zero, zero vectors, length mismatch, non-finite values, zero crop dimensions, and coordinate overflow.
- [ ] Run `cargo test -p feathertalk-pfld --test decode`; expect failure because the crate and API do not exist.
- [ ] Add the workspace member and minimal crate manifest.
- [ ] Implement structured errors, immutable output types, finite/length/dimension validation, checked truncation, and mapping formulas.
- [ ] Add crate-root public API coverage using only public imports.
- [ ] Run `cargo fmt --check`, `cargo clippy -p feathertalk-pfld --all-targets --all-features -- -D warnings`, `cargo test -p feathertalk-pfld --all-targets`, and `git diff --check`.
- [ ] Commit with `feat: add PFLD numeric postprocess contract`.

## Self-Review

- The crate does not depend on `feathertalk-preprocess`; consumers can serialize returned points into the existing 110-point `.lms` format.
- No unsafe float-to-integer cast occurs before range validation.
- No model execution or checkpoint parsing is introduced.
