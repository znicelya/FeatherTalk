# Export Model Package Worker Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a standard model package from a training checkpoint through the worker and the CLI, as `docs/superpowers/specs/2026-09-05-export-model-package-worker-command-design.md` describes.

**Architecture:** A worker module owns admission, the plan it derives from the checkpoint manifest and state, and the JSON payload. A second entry point takes an already-resolved architecture variant and performs the restore/fuse/publish sequence through `write_model_package`, which keeps the publication testable with the micro configuration. Handshake, dispatch and error mapping follow, then a thin CLI subcommand and real-worker coverage.

**Tech Stack:** Rust 2024, existing `feathertalk-export`, `feathertalk-training`, `feathertalk-models`, `burn`, `serde_json`, `time`, clap 4, tempfile. No new dependencies.

## Global Constraints

- Rust commands run from `E:\workspace\github\FeatherTalk\rust`; git commands run from `E:\workspace\github\FeatherTalk`.
- Do not modify `feathertalk-domain`; do not add dependencies or environment variables.
- CPU-only; no ffmpeg and no access to `demo/`.
- Production Rust has no `unwrap`, `expect`, `panic!`, unchecked arithmetic, or unchecked integer conversion.
- Chinese user-visible strings; English identifiers, comments, docs, and technical detail.
- Stage explicit paths only; never push.

### Task 1: Worker export module and publication

**Files:**
- Create: `rust/crates/feathertalk-worker/src/exporting.rs`
- Create: `rust/crates/feathertalk-worker/src/export.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs`
- Test: `rust/crates/feathertalk-worker/tests/exporting.rs`

- [ ] Write tests for admission (relative source, a published package as source, a missing license bundle, an existing destination, an unknown model kind), cancellation before and after the record load, and one full publication from a micro checkpoint asserting the manifest and the payload.
- [ ] Run `cargo test -p feathertalk-worker --test exporting` and observe the missing module/API failure.
- [ ] Implement `ExportModelPackageError`, `check_export_paths`, `ExportPlan` with its constructor from the checkpoint metadata, the `training_mode` slug, the payload builder, `publish_checkpoint_package`, and `execute_export_model_package`.
- [ ] Re-run the focused tests, then `cargo fmt --all -- --check` and clippy for the worker crate.
- [ ] Commit with `feat(worker): export training checkpoints into model packages`.

### Task 2: Handshake, dispatch and error mapping

**Files:**
- Modify: `rust/crates/feathertalk-worker/src/handshake.rs`
- Modify: `rust/crates/feathertalk-worker/src/commands.rs`
- Modify: `rust/crates/feathertalk-worker/src/error_map.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs`
- Test: `rust/crates/feathertalk-worker/tests/exporting.rs`, `rust/crates/feathertalk-worker/tests/handshake.rs`, `rust/crates/feathertalk-worker/tests/runtime.rs`, `rust/crates/feathertalk-worker/tests/process_boundary.rs`

- [ ] Add failing assertions that the handshake always lists `export_model_package` after `migrate_legacy_features`, and that dispatch reports the mapped failure and maps cancellation to `Cancelled`.
- [ ] Run the focused worker tests and observe the unsupported dispatch and the missing handshake entry.
- [ ] Add the command to `supported_commands`, add the `Request::ExportModelPackage` arm, and add `export_task_error` with summary `模型导出失败`.
- [ ] Run worker all-target tests, format, and clippy.
- [ ] Commit with `feat(worker): serve the export-model-package command`.

### Task 3: CLI command

**Files:**
- Modify: `rust/crates/feathertalk-cli/src/cli.rs`
- Modify: `rust/crates/feathertalk-cli/src/run.rs`

- [ ] Add failing parser/request tests for the two positional paths, empty-path rejection, and exact path preservation.
- [ ] Run `cargo test -p feathertalk-cli --lib` and observe the unknown command/request mismatch.
- [ ] Add `Command::ExportModelPackage { source, destination }` and build `Request::ExportModelPackage` with local empty-path checks only.
- [ ] Run CLI tests, format, and clippy.
- [ ] Commit with `feat(cli): add the export-model-package subcommand`.

### Task 4: Real worker coverage and final verification

**Files:**
- Modify: `rust/crates/feathertalk-cli/tests/real_worker.rs`

- [ ] Assert the real handshake lists `export_model_package`, and add a test that exporting a directory which is not a checkpoint exits 1 and leaves the destination absent. Never discover or open anything under `demo/`.
- [ ] Run `cargo test --workspace --all-targets`, `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and `git diff --check`.
- [ ] Review the diff and commit with `test(cli): export a model package end to end`.

## Done when

- Worker handshake and CLI both expose the command.
- A valid UNet checkpoint with a license bundle beside it becomes a published package whose manifest records the checkpoint as its source and the training recipe as its mode.
- Unknown kinds, missing licenses, cancellations and destination collisions are safe, mapped consistently, and leave nothing behind.
- Full workspace verification passes and only the pre-existing untracked `demo/kanghui_training_video_featherhubert_188_latest/` remains outside committed changes.
