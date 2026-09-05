# Migrate Legacy Features Worker Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert legacy rank-three `f32` NumPy feature files into no-clobber FeatherTalk `.f32` artifacts through the worker and CLI.

**Architecture:** Add a focused worker importer that validates paths and the fixed `[video_frames, 2, 1024]` matrix contract, then reuses the existing audio artifact writer. Add the command dispatch and unconditional handshake entry, followed by a thin CLI subcommand and protocol/e2e tests.

**Tech Stack:** Rust 2024, existing `ndarray`, `ndarray-npy`, `feathertalk-audio`, `serde_json`, clap 4, tempfile. No new dependencies.

## Global Constraints

- Rust commands run from `E:\workspace\github\FeatherTalk\rust`; git commands run from `E:\workspace\github\FeatherTalk`.
- Do not modify `feathertalk-domain`; do not add dependencies or environment variables.
- CPU-only; no ffmpeg and no access to `demo/`.
- Production Rust has no `unwrap`, `expect`, `panic!`, unchecked arithmetic, or unchecked integer conversion.
- Chinese user-visible strings; English identifiers, comments, docs, and technical detail.
- Stage explicit paths only; never push.

### Task 1: Worker importer and payload

**Files:**
- Create: `rust/crates/feathertalk-worker/src/migrating_features.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs`
- Test: `rust/crates/feathertalk-worker/tests/migrating_features.rs`

- [ ] Write tests for valid conversion/payload, absolute-path and destination checks, invalid NPY contract, cancellation, and no-clobber.
- [ ] Run `cargo test -p feathertalk-worker --test migrating_features` and observe the missing module/API failure.
- [ ] Implement `execute_migrate_legacy_features(params, token, reporter)`, a cancellation error, strict admission, `ReadNpyExt` conversion, `FeatureMatrix::new`, `write_feature_file_no_clobber`, and JSON payload.
- [ ] Re-run the focused tests and then `cargo fmt --all -- --check`.
- [ ] Commit with `feat(worker): migrate legacy features into artifacts`.

### Task 2: Handshake and command dispatch

**Files:**
- Modify: `rust/crates/feathertalk-worker/src/handshake.rs`
- Modify: `rust/crates/feathertalk-worker/src/commands.rs`
- Modify: `rust/crates/feathertalk-worker/src/error_map.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs`
- Test: `rust/crates/feathertalk-worker/tests/migrating_features.rs`, `rust/crates/feathertalk-worker/tests/handshake.rs`, `rust/crates/feathertalk-worker/tests/commands.rs`

- [ ] Add failing assertions that the handshake lists the command and direct execution reports `Preparing`, `Importing`, then completion, while cancellation maps to `Cancelled`.
- [ ] Run focused worker tests and observe unsupported dispatch/missing handshake.
- [ ] Add the command after `ImportLegacyModel`, map importer errors with a dedicated `legacy_feature_task_error`, and preserve cancellation.
- [ ] Run worker all-target tests, format, and clippy.
- [ ] Commit with `feat(worker): serve the migrate-legacy-features command`.

### Task 3: CLI command

**Files:**
- Modify: `rust/crates/feathertalk-cli/src/cli.rs`
- Modify: `rust/crates/feathertalk-cli/src/run.rs`
- Test: `rust/crates/feathertalk-cli/tests/cli.rs`

- [ ] Add failing parser/request tests for the two positional paths, empty-path rejection, and exact path preservation.
- [ ] Run focused CLI tests and observe the unknown command/request mismatch.
- [ ] Add `Command::MigrateLegacyFeatures { source, destination }` and build `Request::MigrateLegacyFeatures` with local empty-path checks.
- [ ] Run CLI tests, format, and clippy.
- [ ] Commit with `feat(cli): add the migrate-legacy-features subcommand`.

### Task 4: Real worker coverage and final verification

**Files:**
- Modify: `rust/crates/feathertalk-cli/tests/real_worker.rs`

- [ ] Add a gated test driven only by `FEATHERTALK_WORKER_LEGACY_FEATURES`; create a temporary destination, run the release CLI, parse the completed payload, and verify the artifact can be read and no-clobber holds. Skip clearly when the variable is absent or the source is not suitable. Never discover or open `.MOV` files.
- [ ] Run `cargo test --workspace --all-targets`, `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and `git diff --check`.
- [ ] Review the diff and commit with `test(cli): migrate legacy features end to end`.

## Done when

- Worker handshake and CLI both expose the command.
- Valid `[N, 2, 1024]` `f32` NPY files become versioned `.f32` files with a stable JSON report.
- Invalid inputs, cancellations, and destination collisions are safe and mapped consistently.
- Full workspace verification passes and only the pre-existing untracked `demo/kanghui_training_video_featherhubert_188_latest/` remains outside committed changes.
