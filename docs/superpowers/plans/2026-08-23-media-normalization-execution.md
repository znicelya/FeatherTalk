# Media Normalization Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add strict bundled-FFmpeg media probing and atomic normalization to `feathertalk-media`.

**Architecture:** Preserve validated path/layout types. Add immutable metadata types, a pure FFprobe parser, fixed argv builders, an injectable bounded runner, output verification, streaming hashes, and an injected two-file commit state machine.

**Tech Stack:** Rust 1.92 edition 2024, standard process/filesystem APIs, `serde`, `serde_json`, `sha2`, `tempfile`, and `thiserror`.

## Global Constraints

- Source input is canonicalized by `validate_input`.
- Tool paths are absolute; production rejects symlinks and non-files.
- Source probe requires exactly one video and one audio stream.
- Video target is 25 FPS, `mpeg4`, `yuv420p`, MP4.
- Audio target is 16,000 Hz, mono, `pcm_s16le`/`s16`, WAV.
- Never invoke a shell or concatenate paths into strings.
- Probe/captured output limits are 1 MiB; timeout/duration maximum is 24 hours.
- Preserve existing destinations until both new files pass verification.
- Do not add frame extraction, models, manifests, worker RPC, cancellation, or synthesis.
- Every implementation step follows red-green-refactor.

---

### Task 1: Add immutable media execution types

**Files:**
- Modify: `rust/crates/feathertalk-media/Cargo.toml`
- Modify: `rust/crates/feathertalk-media/src/model.rs`
- Modify: `rust/crates/feathertalk-media/src/error.rs`
- Modify: `rust/crates/feathertalk-media/src/lib.rs`
- Create: `rust/crates/feathertalk-media/tests/probe_types.rs`

**Interfaces:** Produces `MediaToolchain::new(PathBuf, PathBuf, Duration)`,
`FrameRate`, `ProbeFormat`, `VideoMetadata`, `AudioMetadata`, `MediaProbe`,
`MediaArtifact`, and `NormalizedMedia`, all with immutable accessors.

- [ ] Write the failing type tests for valid values, read-only accessors, relative executable paths, and zero/over-limit timeout.
- [ ] Run `cargo test -p feathertalk-media --test probe_types` and confirm the expected missing-type failure.
- [ ] Add serde/sha2 dependencies and implement the minimal types and checked constructors.
- [ ] Export types from the crate root and rerun the focused test to green.
- [ ] Run `cargo fmt --all -- --check` and commit `feat: define media execution types`.

### Task 2: Parse bounded FFprobe metadata

**Files:**
- Create: `rust/crates/feathertalk-media/src/probe.rs`
- Modify: `rust/crates/feathertalk-media/src/lib.rs`
- Modify: `rust/crates/feathertalk-media/src/error.rs`
- Create: `rust/crates/feathertalk-media/tests/probe_parser.rs`

**Interfaces:** Produces `parse_probe_json(bytes: &[u8]) -> Result<MediaProbe,
MediaError>` and crate-private `parse_probe_json_for(bytes, StreamExpectation)`.

- [ ] Write inline JSON tests for valid one-video/one-audio input, missing/duplicate streams, malformed ratios, invalid dimensions, durations, counts, and unknown extra fields.
- [ ] Run `cargo test -p feathertalk-media --test probe_parser` and confirm RED.
- [ ] Implement bounded UTF-8/JSON parsing, exact stream cardinality, rational rate parsing, and checked count conversion.
- [ ] Run the parser test and commit `feat: parse strict media probe metadata`.

### Task 3: Build fixed command specifications

**Files:**
- Create: `rust/crates/feathertalk-media/src/commands.rs`
- Modify: `rust/crates/feathertalk-media/src/lib.rs`
- Create: `rust/crates/feathertalk-media/tests/commands.rs`

**Interfaces:** Produces immutable `CommandSpec`, `probe_command`,
`video_normalization_command`, and `audio_normalization_command`.

- [ ] Write exact argv tests for all fixed flags, maps, filters, codecs, formats, and metadata removal.
- [ ] Add a hostile native path test and assert it remains one `OsString` argument.
- [ ] Run command tests and confirm RED because builders are absent.
- [ ] Implement builders with only validated executable/source/temp-path variables; rerun tests green.
- [ ] Commit `feat: define safe media tool commands`.

### Task 4: Execute bounded processes and source probes

**Files:**
- Create: `rust/crates/feathertalk-media/src/process.rs`
- Create: `rust/crates/feathertalk-media/src/execution.rs`
- Modify: `rust/crates/feathertalk-media/src/lib.rs`
- Create: `rust/crates/feathertalk-media/tests/probe_execution.rs`

**Interfaces:** Produces `ProcessOutput`, `ProcessRunner`, `SystemProcessRunner`,
`probe_media_with_runner`, and public `probe_media`.

- [ ] Write fake-runner tests for valid output, non-zero exit, timeout, output overflow, and malformed JSON; assert exact probe command count.
- [ ] Run focused tests RED.
- [ ] Map fake outputs to stable errors and parse successful stdout.
- [ ] Add platform-gated real runner tests for capture, non-zero exit, timeout/kill, and 1 MiB overflow.
- [ ] Implement executable validation, concurrent pipe draining, polling, kill/reap, and bounded capture.
- [ ] Run focused tests and commit `feat: execute bounded media probes`.

### Task 5: Stage, normalize, verify, and hash outputs

**Files:**
- Create: `rust/crates/feathertalk-media/src/normalize.rs`
- Modify: `rust/crates/feathertalk-media/src/lib.rs`
- Create: `rust/crates/feathertalk-media/tests/normalization_execution.rs`

**Interfaces:** Produces `normalize_media_with_runner` up to verified staged outputs,
`MediaArtifact`, and final `NormalizedMedia` after commit integration.

- [ ] Write a successful fake-runner test that queues source/video-only/audio-only probes, writes temp bytes, and asserts command order, metadata, hashes, and byte counts.
- [ ] Run the focused test RED.
- [ ] Implement unique suffix-preserving temp files, two FFmpeg calls, fsync, post-probes, contract checks, and streaming SHA-256.
- [ ] Add tests for wrong codec/pixel format/FPS, wrong audio properties, duration delta, missing temp output, and second FFmpeg failure; assert old destinations are unchanged.
- [ ] Run `cargo test -p feathertalk-media --test normalization_execution` green.

### Task 6: Commit both outputs with rollback

**Files:**
- Create: `rust/crates/feathertalk-media/src/commit.rs`
- Modify: `rust/crates/feathertalk-media/src/normalize.rs`
- Create: `rust/crates/feathertalk-media/tests/output_commit.rs`

**Interfaces:** Produces crate-private `commit_output_pair(video_temp, audio_temp,
layout, file_ops)` and production `SystemFileOps`.

- [ ] Write tests for absent destinations, replacement, invalid late destinations, backup failure, first rename failure, and second rename failure.
- [ ] Run `cargo test -p feathertalk-media --test output_commit` RED.
- [ ] Implement same-directory backups, ordered renames, reverse rollback, and primary/rollback error reporting.
- [ ] Integrate commit into normalization and assert only invocation-owned temp/backup paths are removed.
- [ ] Run focused tests and commit `feat: atomically normalize media outputs`.

### Task 7: Full validation and integration

**Files:** Modify only files required by validation or cross-platform corrections.

- [ ] Run `cargo fmt --all -- --check`.
- [ ] Run `cargo clippy -p feathertalk-media --all-targets --all-features -- -D warnings`.
- [ ] Run `cargo test -p feathertalk-media --all-targets`.
- [ ] Run `cargo test --workspace --all-targets`.
- [ ] Run `git diff --check`; inspect staged paths for `demo/` or temporary files.
- [ ] Commit validation corrections if any, merge into `main`, and rerun media tests on merged `main`.

## Plan Self-Review

- Spec coverage: types, parser, argv, bounded execution, verification, hashes,
  atomic replacement, rollback, and integration each map to a task.
- Placeholder scan: no deferred implementation or unspecified error path remains.
- Type consistency: later interfaces are introduced in earlier tasks.
- Scope: frame extraction, models, manifests, worker RPC, and rendering remain out.
