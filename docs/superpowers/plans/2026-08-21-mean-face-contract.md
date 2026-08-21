# PFLD Mean Face Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Embed the fixed PFLD mean face in `feathertalk-pfld`, add a default decoder entry point, and retain strict external-file loading without changing existing slice-based decoding.

**Architecture:** Add a focused `mean_face` module to `feathertalk-pfld` containing the fixed `[f32; 220]` constant and compatibility file parser. Typed decoders delegate to the existing numerical mapper so coordinate behavior remains single-sourced.

**Tech Stack:** Rust 1.92 edition 2024, standard library, `thiserror`, and `tempfile` for temporary parser fixtures.

## Global Constraints

- `mean_face.txt` contains exactly 220 finite `f32` values.
- The fixed values are compiled into `feathertalk-pfld` as `MEAN_FACE`.
- Parsing accepts all Rust `split_whitespace` separators, including spaces, tabs, LF, and CRLF.
- Existing `decode_landmarks(&[f32], &[f32], CropGeometry)` remains source-compatible.
- New typed decoding delegates to the existing slice decoder.
- No Burn, image, FFmpeg, checkpoint, WGPU, GPUI, or model dependency is added.

---

### Task 1: Implement embedded mean face, strict loading, and typed decode

**Files:**
- Modify: `rust/crates/feathertalk-pfld/src/lib.rs`
- Modify: `rust/crates/feathertalk-pfld/src/error.rs`
- Modify: `rust/crates/feathertalk-pfld/src/decode.rs`
- Create: `rust/crates/feathertalk-pfld/src/mean_face.rs`
- Create: `rust/crates/feathertalk-pfld/tests/mean_face.rs`
- Modify: `rust/crates/feathertalk-pfld/tests/public_api.rs`

**Interfaces:**
- Consumes `PFLD_OUTPUT_VALUE_COUNT`, `CropGeometry`, `PFLDLandmarks`, and `decode_landmarks`.
- Produces `MEAN_FACE`, `MeanFace`, `MeanFace::values`, `read_mean_face`, `decode_landmarks_with_mean_face`, and `decode_landmarks_with_default_mean_face`.

- [ ] Write failing tests for the embedded constant, default decoder, repository fixture, whitespace variants, missing/invalid UTF-8 files, malformed and non-finite tokens, wrong counts, read-only values, typed-decoder parity, and public crate-root imports.
- [ ] Run `cargo test -p feathertalk-pfld --test mean_face`; expect failure because the reader, type, and typed decoder do not exist.
- [ ] Add `PfldError` variants for I/O, invalid UTF-8, malformed mean-face token, and wrong count; remove `Eq` derives if the retained I/O source prevents them.
- [ ] Implement `MEAN_FACE` as a compile-time `[f32; 220]` constant copied from the fixed baseline values.
- [ ] Implement `read_mean_face` with `fs::read`, UTF-8 conversion, `split_whitespace`, indexed parsing, finite validation, exact count, and `Vec<f32>` to `[f32; 220]` conversion.
- [ ] Implement `decode_landmarks_with_mean_face` as a delegation to `decode_landmarks(model_output, mean_face.values(), crop)`.
- [ ] Implement `decode_landmarks_with_default_mean_face` as a delegation through `MEAN_FACE`.
- [ ] Export all new public types and functions from the crate root.
- [ ] Run `cargo fmt --check`, `cargo clippy -p feathertalk-pfld --all-targets --all-features -- -D warnings`, `cargo test -p feathertalk-pfld --all-targets`, and `git diff --check`.
- [ ] Commit with `feat: load PFLD mean face contract`.

## Self-Review

- The fixed values are intentionally committed as a crate constant; no checkpoint or unrelated model weights are copied.
- Existing numerical behavior remains in `decode_landmarks`; the typed API cannot drift from it.
- Parse errors report stable categories and deterministic token indices.
