# Import Legacy Model Worker Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve `Request::ImportLegacyModel` in `feathertalk-worker` and expose `feathertalk import-legacy-model <SOURCE> <KIND> <DESTINATION>`, converting supported legacy `.pth` files into standard no-clobber model packages.

**Architecture:** The worker validates the frozen domain parameters, reads `LICENSES.json` beside the source, imports supported FeatherHuBERT or Original UNet weights on the CPU, and delegates package staging, manifest creation, validation, and atomic publication to the existing export crate. Unsupported kinds fail before any destination is created. The command reports `Preparing`, `Importing`, and `Completed`; import/package failures use one stable model-incompatible error mapping.

**Tech Stack:** Rust edition 2024, `feathertalk-domain`, `feathertalk-weights`, `feathertalk-export`, Burn CPU backends, `serde_json`, clap 4, tempfile fixtures. No new dependencies or environment variables.

## Global Constraints

- Run every `cargo`, `rustfmt`, and `clippy` command from `E:\workspace\github\FeatherTalk\rust`; run every `git` command from `E:\workspace\github\FeatherTalk`.
- Do not modify `feathertalk-domain`; use its existing `ImportLegacyModelParams`, `LegacyModelKind`, `Request::ImportLegacyModel`, `TaskKind::ImportLegacyModel`, `TaskStage`, `TaskError`, and payload protocol.
- CPU only; do not launch ffmpeg, ffprobe, or any external process.
- `source` is an absolute, regular non-symlink file ending in `.pth` or `.pth.tar`; `destination` is absolute, absent, and has an existing parent; licenses are exactly `source.parent()/LICENSES.json`.
- Supported kinds are `FeatherHubert` and `OriginalUnet`; `Pfld` and `MobileOneUnet` fail explicitly with `ErrorCode::ModelIncompatible` and never create a partial destination.
- Use the worker’s current UTC RFC3339 timestamp and `WorkerConfig::worker_version()` for package metadata.
- User-facing strings are Chinese; identifiers, comments, doc comments, and technical `detail` strings are English.
- No `unwrap`, `expect`, `panic!`, panicking indexing, or unchecked arithmetic outside tests; use checked conversions and `ok_or_else`.
- Stage explicit paths only, never anything under `demo/`; do not push. Each task has its own test-first cycle and exact commit message.

## File Structure

```text
rust/crates/feathertalk-worker/src/lib.rs             module/public exports
rust/crates/feathertalk-worker/src/handshake.rs       unconditional capability
rust/crates/feathertalk-worker/src/error_map.rs       legacy error mapping
rust/crates/feathertalk-worker/src/importing.rs        validation and package builders
rust/crates/feathertalk-worker/src/commands.rs        request dispatch
rust/crates/feathertalk-worker/src/runtime.rs         exact handshake vectors
rust/crates/feathertalk-worker/tests/importing.rs     importer unit tests
rust/crates/feathertalk-cli/src/cli.rs                subcommand and kind parser
rust/crates/feathertalk-cli/src/run.rs                request construction
rust/crates/feathertalk-cli/tests/real_worker.rs      gated real-package e2e
```

### Task 1: Announce the command

**Files:** Modify `worker/src/handshake.rs`, `worker/src/runtime.rs`, `worker/tests/process_boundary.rs`, and any exact supported-command vectors in worker tests.

- [ ] Add a failing test asserting `TaskKind::ImportLegacyModel` is present for an empty configuration and in the JSON ready frame.
- [ ] Run the focused handshake/process-boundary tests and observe the missing command.
- [ ] Insert `TaskKind::ImportLegacyModel` immediately after `InspectModel` in `supported_commands`; update all exact vectors.
- [ ] Run `cargo test -p feathertalk-worker --test process_boundary --test runtime` and `cargo fmt --all -- --check`.
- [ ] Commit `git add rust/crates/feathertalk-worker/src/handshake.rs rust/crates/feathertalk-worker/src/runtime.rs rust/crates/feathertalk-worker/tests/process_boundary.rs rust/crates/feathertalk-worker/tests/runtime.rs; git commit -m "feat(worker): announce the import-legacy-model command"`.

### Task 2: Map legacy import failures

**Files:** Modify `worker/src/error_map.rs` and `worker/src/lib.rs`; test in `worker/tests/importing.rs` or a focused error-map test.

- [ ] Add failing assertions that a legacy `WeightImportError` and `PackageError` map to `ErrorCode::ModelIncompatible`, summary `模型导入失败`, recovery `ReimportModel`, and the requested `TaskStage`.
- [ ] Run the focused test and verify the mapper does not exist.
- [ ] Implement `legacy_task_error(error: &impl Display, stage: TaskStage) -> TaskError`, preserving bounded English technical detail and mapping both importer/package errors through it; export it from `lib.rs`.
- [ ] Run the focused worker tests, format, and clippy for the worker crate.
- [ ] Commit `git add rust/crates/feathertalk-worker/src/error_map.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/tests/importing.rs; git commit -m "feat(worker): map legacy import failures"`.

### Task 3: Import legacy models into packages

**Files:** Create `worker/src/importing.rs`; modify `worker/src/lib.rs`; create `worker/tests/importing.rs` fixtures/tests.

- [ ] Write failing tests for relative source, symlink source, wrong extension, missing licenses, existing destination, unsupported `Pfld`/`MobileOneUnet`, and successful FeatherHuBERT/Original UNet package reports. Build tiny valid fixtures with existing weights/export APIs; never read the protected `.MOV`.
- [ ] Run the importer tests and confirm they fail before the module exists.
- [ ] Implement `execute_import_legacy_model(params, config, token, reporter) -> Result<ImportLegacyModelReport, ImportLegacyModelError>` (or equivalent worker-internal result): validate paths and kind, check cancellation before import and before publication, report `Importing`, load `LICENSES.json`, use `OffsetDateTime::now_utc().format(&Rfc3339)`, and pass `config.worker_version()`.
- [ ] For FeatherHuBERT call `build_feather_hubert_package(&FeatherHubertPackageRequest { source, licenses, destination, created_at, minimum_app_version })`; for Original UNet initialize `OriginalUnetConfig::production()`, call `import_into::<CpuBackend, OriginalUnet<CpuBackend>>` with `LegacyImportRequest`, then call `write_model_package` with the production config factory. Return source hash, model hash, tensor count, total elements, model kind, and architecture version from the published manifest/report.
- [ ] Ensure unsupported kinds and all validation/import/package failures return the Task 2 mapping without creating destination directories; add a JSON conversion helper for the nine-field completed payload.
- [ ] Run `cargo test -p feathertalk-worker --test importing`, then worker all-target tests, format, and clippy.
- [ ] Commit `git add rust/crates/feathertalk-worker/src/importing.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/tests/importing.rs; git commit -m "feat(worker): import legacy models into packages"`.

### Task 4: Execute the worker command

**Files:** Modify `worker/src/commands.rs`, `worker/src/lib.rs`, and `worker/src/runtime.rs` if reporter wiring requires a helper.

- [ ] Add a failing command test constructing `Request::ImportLegacyModel` with a valid fixture and asserting `Preparing`, `Importing`, then `Completed` payload; add cancellation-before-import and cancellation-before-publish tests.
- [ ] Run the focused command tests and observe the unsupported/missing dispatch.
- [ ] Add the `Request::ImportLegacyModel(params)` match arm, call the importer with the runtime token/reporter, map cancellation to `CommandOutcome::Cancelled`, and map failures through `legacy_task_error` at the correct stage.
- [ ] Run worker command/runtime tests, format, and clippy.
- [ ] Commit `git add rust/crates/feathertalk-worker/src/commands.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/src/runtime.rs rust/crates/feathertalk-worker/tests/importing.rs rust/crates/feathertalk-worker/tests/runtime.rs; git commit -m "feat(worker): execute the import-legacy-model command"`.

### Task 5: Add the CLI subcommand

**Files:** Modify `cli/src/cli.rs`, `cli/src/run.rs`, and CLI unit tests.

- [ ] Add failing parser tests for `import-legacy-model SOURCE feather-hubert DEST`, all four accepted kind spellings, missing/relative arguments, and exact request path preservation.
- [ ] Run focused CLI tests and verify clap reports the unknown command/request mismatch.
- [ ] Add `Command::ImportLegacyModel { source, kind, destination }` and a local `LegacyModelKindArg` `ValueEnum`; map `feather-hubert`, `pfld`, `original-unet`, and `mobileone-unet` to domain kinds and construct `Request::ImportLegacyModel` without canonicalizing paths.
- [ ] Wire normal JSON/progress/error exit handling through the existing CLI runner; keep user-facing help and rejection text Chinese.
- [ ] Run `cargo test -p feathertalk-cli --lib`, format, and clippy.
- [ ] Commit `git add rust/crates/feathertalk-cli/src/cli.rs rust/crates/feathertalk-cli/src/run.rs rust/crates/feathertalk-cli/tests; git commit -m "feat(cli): add the import-legacy-model subcommand"`.

### Task 6: Import a real model end to end

**Files:** Modify `cli/tests/real_worker.rs` only.

- [ ] Add a gated test that reads an explicit source path from `FEATHERTALK_WORKER_LEGACY_MODEL`, uses a sibling `LICENSES.json`, creates a fresh destination under a temp directory, runs `feathertalk import-legacy-model`, parses one completed JSON object, checks source hash preservation, manifest hashes/architecture, tensor statistics, and destination no-clobber behavior. If the variable or license file is absent, print a clear skip. Never discover or open `.MOV` files.
- [ ] Run the test before implementation/build and capture the expected skip or failure; then build release worker/CLI and run it with an explicit `.pth` source when available.
- [ ] Verify the complete workspace: `cargo test --workspace --all-targets`, `cargo fmt --all -- --check`, and `cargo clippy --workspace --all-targets -- -D warnings`.
- [ ] Commit `git add rust/crates/feathertalk-cli/tests/real_worker.rs; git commit -m "test(cli): import a legacy model end to end"`.

## Final verification

From `E:\workspace\github\FeatherTalk\rust`, run the full test, format, and clippy commands above and inspect `git status --short`; only the six commits and the pre-existing untracked `demo/kanghui_training_video_featherhubert_188_latest/` directory may remain. Do not stage, read, modify, delete, or push that directory.

