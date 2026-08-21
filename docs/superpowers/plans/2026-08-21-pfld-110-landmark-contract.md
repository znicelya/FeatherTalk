# PFLD 110-Point Landmark Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Align `feathertalk-preprocess` with Python by requiring exactly 110 landmark points per `.lms` file.

**Architecture:** Keep the existing parser, geometry, and audio-window modules. Change only the shared landmark count constant and tests that construct landmark fixtures; no model or image dependencies are added.

**Tech Stack:** Rust 1.92 edition 2024, standard library, `thiserror`, and `tempfile` for tests.

## Global Constraints

- The `.lms` contract requires exactly 110 points.
- Coordinates must remain finite and non-negative.
- Python geometry indices remain unchanged: `1`, `31`, and `52`.
- Runtime dependencies remain only the standard library and `thiserror`.
- No PFLD model execution, Burn, image, FFmpeg, or checkpoint conversion is introduced.

---

### Task 1: Update landmark count contract

**Files:**
- Modify: `rust/crates/feathertalk-preprocess/src/landmarks.rs`
- Modify: `rust/crates/feathertalk-preprocess/tests/landmarks.rs`
- Modify: `rust/crates/feathertalk-preprocess/tests/geometry.rs`
- Modify: `rust/crates/feathertalk-preprocess/tests/public_api.rs` only if fixture helpers require it

**Interfaces:**
- Consumes the existing `Landmarks`, `Point`, `read_landmarks`, and `compute_face_bbox` APIs.
- Produces the same APIs with the parser requiring exactly 110 points.

- [ ] Write tests first: change the fixture helper to create 110 points; assert 110 parses; assert 109 and 111 fail with `WrongLandmarkCount`.
- [ ] Run `cargo test -p feathertalk-preprocess --test landmarks`; expect failure because production validation still expects 68.
- [ ] Change the production expected count from 68 to 110 and export `PFLD_LANDMARK_COUNT` from the crate root.
- [ ] Run `cargo fmt --all` and `cargo test -p feathertalk-preprocess --all-targets`; expect all tests to pass.
- [ ] Run `cargo clippy -p feathertalk-preprocess --all-targets --all-features -- -D warnings` and `git diff --check`.
- [ ] Commit with `feat: align landmarks with PFLD 110-point output`.

## Self-Review

- No 110-to-68 mapping is added because Python writes and consumes all 110 points.
- Existing geometry indices remain valid because they are below 110.
- No unrelated preprocess behavior changes.
