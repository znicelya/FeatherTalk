# Inspect Model Worker Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve `Request::InspectModel` in `feathertalk-worker` and expose it as `feathertalk inspect-model <source>`, so any directory that holds either an exported model package or a training checkpoint answers three questions from its manifests alone: what is in it, how big it is, and whether this build can use it.

**Architecture:** The two readers already exist and already validate. `feathertalk-export::read_package_manifest` walks an exported package; `feathertalk-training::read_training_checkpoint` walks a checkpoint pair. Neither deserialises a tensor. This slice is the thin layer above them: `inspecting.rs` decides which of the two layouts a directory is by probing for the one file that only that layout has, collects the on-disk size of every file the manifest names, and turns manifest-versus-build disagreements into a list of stable reason codes; `inspect_result.rs` shapes one JSON object with the same eighteen keys for both layouts; `inspect.rs` is the command body. No weights are loaded, so no backend is instantiated and no toolchain is needed -- which is why the handshake announces the command unconditionally.

**Tech Stack:** Rust edition 2024 (rust-version 1.94), `serde_json` for the payload, `tempfile` for fixtures. burn `=0.21.0` appears only inside tests, where `NdArray<f32>` and `Autodiff<NdArray<f32>>` write the real fixture packages and checkpoints the reader then inspects. clap 4 for the CLI. No new dependency in any manifest.

**Design:** `docs/superpowers/specs/2026-09-04-inspect-model-worker-command-design.md`

## Global Constraints

- Run every `cargo`, `rustfmt` and `clippy` command from `E:\workspace\github\FeatherTalk\rust`. Run every `git` command from `E:\workspace\github\FeatherTalk`.
- The wire protocol is frozen and `feathertalk-domain` is not edited at all. `TaskKind::InspectModel`, `InspectModelParams { source }`, `Request::InspectModel`, `ErrorCode::ModelIncompatible` and `TaskStage::Preparing` all landed in earlier slices; this slice only starts using them.
- Nothing outside `feathertalk-worker` and `feathertalk-cli` changes. Both readers, `render_variant`, `checkpoint_descriptor`, `package_task_error` and `training_task_error` are consumed exactly as they are.
- No new environment variable and no new capability flag. Inspection reads files; it never shells out and never needs ffmpeg, SCRFD, PFLD, FeatherHuBERT or VGG19.
- CPU only, and only in tests: fixtures are written with `NdArray<f32>` / `Autodiff<NdArray<f32>>` because a checkpoint has to be produced by the same types that wrote the committed ones.
- Weights are never deserialised (design section 3). Size and hash come from the manifest plus `fs::symlink_metadata`; `model.safetensors`, `model.bin` and `optimizer.bin` are never opened.
- Chinese only inside user-facing string literals (task-error summaries, CLI help, rejection reasons). Identifiers, comments, doc comments and error `detail` text are English.
- No `unwrap`, `expect`, `panic!`, panicking index or panicking arithmetic outside `#[cfg(test)]` and `tests/`. Every `u64` to `usize` conversion is a `try_from`. Prefer `ok_or_else` over `ok_or`. Never `.clone()` a `Copy` device.
- Incompatibility reasons are `&'static str` codes, not sentences: the CLI and the future UI decide the wording. The list is sorted by nothing -- it is built in a fixed order, which the tests assert.
- rustfmt defaults apply (`max_width` 100). Run `cargo fmt --all` after each implementation step instead of hand-sorting imports or guessing where a call wraps.
- Every unit test is offline and self-contained: a `tempfile::tempdir`, a fixture written by the test itself, no environment variable. Only `feathertalk-cli/tests/real_worker.rs` (Task 7) touches a real model directory, and it skips unless its variables are set.
- Stage explicit paths only. Never stage anything under `demo/`. One commit per task with the exact message given. Do not push.

## One Deviation From The Design

`InspectedFile.bytes_on_disk` is `Option<u64>`, not `u64`. Design section 3 lists the on-disk size beside the manifest size as though it is always a number, because both readers guarantee the file exists before a manifest is ever parsed. It is still a `Option`: `symlink_metadata` can fail between the reader's walk and this one -- a file removed, a permission changed -- and the honest answer is then "unknown", not zero. `None` counts as a `file_size` incompatibility, so the payload never claims agreement it did not verify, and the JSON carries `null` rather than a lie.

## File Structure

```
rust/crates/feathertalk-worker/src/lib.rs               + the inspect modules and their public surface
rust/crates/feathertalk-worker/src/handshake.rs         + TaskKind::InspectModel, unconditionally
rust/crates/feathertalk-worker/src/admission.rs         + check_model_source
rust/crates/feathertalk-worker/src/inspecting.rs        new: layout classification, file sizes, compatibility reasons
rust/crates/feathertalk-worker/src/inspect_result.rs    new: the eighteen-key payload
rust/crates/feathertalk-worker/src/inspect.rs           new: the command body
rust/crates/feathertalk-worker/src/commands.rs          + the Request::InspectModel arm
rust/crates/feathertalk-worker/tests/support/mod.rs     + package and checkpoint fixtures
rust/crates/feathertalk-worker/tests/inspecting.rs      new: classification, admission, compatibility
rust/crates/feathertalk-worker/tests/inspect_result.rs  new: payload shape
rust/crates/feathertalk-worker/tests/inspect.rs         new: the command body end to end, offline
rust/crates/feathertalk-worker/tests/handshake.rs       + inspect_model in every handshake vector
rust/crates/feathertalk-worker/tests/runtime.rs         + the same vectors
rust/crates/feathertalk-cli/src/cli.rs                  + the inspect-model subcommand
rust/crates/feathertalk-cli/src/run.rs                  + the build_request arm and its inline tests
rust/crates/feathertalk-cli/tests/real_worker.rs        + the gated end-to-end inspection
```

Read `docs/superpowers/specs/2026-09-04-inspect-model-worker-command-design.md` once before Task 1. Every "why" below is a pointer back into it.

---

### Task 1: Announce the inspect-model command

**Files:**

- Modify: `rust/crates/feathertalk-worker/src/handshake.rs:23` (one line)
- Modify: `rust/crates/feathertalk-worker/src/lib.rs:1-7` (the head doc comment still stops at `train`)
- Test: `rust/crates/feathertalk-worker/tests/handshake.rs` (every exact-vector assertion, plus one renamed test)
- Test: `rust/crates/feathertalk-worker/tests/runtime.rs` (the same vectors, at `:459-477` and `:1058-1067`)

**Interfaces:**

- Produces: `supported_commands(&config)` starting with `[TaskKind::ValidateProject, TaskKind::InspectModel]` for every possible configuration.
- Consumes: `WorkerConfig::from_values`, `TaskKind::InspectModel`, `ready_frame`.

**Why now:** design section 8 -- inspection reads manifests, so it has no precondition to gate on, and a worker with nothing configured must still announce it. Doing it first also means every later task is written against a handshake that already tells the truth. `runtime.rs::unsupported_reason` needs no new arm: its `_ =>` catch-all covers the kinds that are still unserved, and after this task `InspectModel` never reaches it.

- [ ] **Step 1: Write the failing test**

In `rust/crates/feathertalk-worker/tests/handshake.rs`, rename `a_worker_without_a_media_toolchain_only_offers_project_validation` (line 55) and extend it -- the old name stops being true the moment the command is announced, and a second test asserting the same vector would be a copy:

```rust
#[test]
fn a_worker_without_a_media_toolchain_only_offers_the_toolchain_free_commands() {
    let config = WorkerConfig::from_values(None, None, None);
    assert!(config.media().is_none());
    assert!(
        config
            .media_rejection()
            .is_some_and(|reason| reason.contains(ENV_FFPROBE))
    );
    let frame = ready_frame(&config);
    frame.validate().unwrap();
    // Inspection reads manifests, so it is announced with no toolchain at all.
    assert_eq!(
        frame.supported_commands,
        vec![TaskKind::ValidateProject, TaskKind::InspectModel]
    );
    assert!(!frame.capabilities.ffmpeg);
    assert_eq!(supported_commands(&config).len(), 2);
}
```

Then add `TaskKind::InspectModel,` immediately after `TaskKind::ValidateProject,` in every other exact vector in the file (`:41-47`, `:145-154`, `:182`, `:204`, `:241-248`, `:271-274`, `:296`) and in `tests/runtime.rs` (`:459-467`, `:477`, `:1058-1067`). Order matters: the vectors are compared with `assert_eq!`, and `supported_commands` pushes in a fixed order.

- [ ] **Step 2: Run test to verify it fails**

```powershell
cargo test -p feathertalk-worker --test handshake
```

Every extended vector fails on the missing `inspect_model`.

- [ ] **Step 3: Write minimal implementation**

`src/handshake.rs`, replacing line 23:

```rust
    // Neither command needs a toolchain: one walks a project directory, the
    // other reads a model manifest, so both are always available.
    let mut commands = vec![TaskKind::ValidateProject, TaskKind::InspectModel];
```

And correct the head doc comment in `src/lib.rs`, which has been listing the served commands since the first slice:

```rust
//! This slice serves `validate_project`, `probe_media`, `normalize_media`,
//! `extract_frames`, `extract_features`, `lock_asset_package`, `train`,
//! `render` and `inspect_model` on the CPU.
```

- [ ] **Step 4: Run test to verify it passes**

```powershell
cargo test -p feathertalk-worker --test handshake
cargo test -p feathertalk-worker --test runtime
cargo fmt --all -- --check
cargo clippy -p feathertalk-worker --all-targets -- -D warnings
```

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/src/handshake.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/tests/handshake.rs rust/crates/feathertalk-worker/tests/runtime.rs
git commit -m "feat(worker): announce the inspect-model command"
```

---

### Task 2: Classify a model source directory

**Files:**

- Create: `rust/crates/feathertalk-worker/src/inspecting.rs`
- Modify: `rust/crates/feathertalk-worker/src/admission.rs` (append `check_model_source`)
- Modify: `rust/crates/feathertalk-worker/src/lib.rs` (the module and its exports)
- Test: `rust/crates/feathertalk-worker/tests/inspecting.rs` (new)

**Interfaces:**

- Produces: `ModelSourceKind { ModelPackage, TrainingCheckpoint }` with `as_slug() -> &'static str` (`"model_package"` / `"training_checkpoint"`).
- Produces: `model_source_kind(source: &Path) -> Result<ModelSourceKind, TaskError>`.
- Produces: `admission::check_model_source(source: &Path) -> Result<(), TaskError>` (crate-private).
- Consumes: `MODEL_FILE_NAME` (`model.safetensors`), `CHECKPOINT_MODEL_FILE_NAME` (`model.bin`), `admission::invalid_request`, `error_map::clamp`.

**Why now:** design section 2 -- one command serves two directory layouts, so everything downstream branches on this answer. The probe is the cheapest possible one: each layout has a model file the other cannot have, and the two names differ (`model.safetensors` versus `model.bin`), so a single `symlink_metadata` per candidate decides it without opening either file. Ambiguous and empty directories are rejected here rather than guessed at, because guessing would hand the wrong reader a directory and turn a clear「无法识别」into a confusing parse error.

- [ ] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-worker/tests/inspecting.rs`. Classification never opens a file, so every fixture is a one-byte placeholder:

```rust
use std::{fs, path::Path};

use feathertalk_domain::{ErrorCode, TaskStage};
use feathertalk_worker::{ModelSourceKind, model_source_kind};

fn touch(path: &Path) {
    fs::write(path, b"x").unwrap();
}

fn directory(root: &Path, name: &str) -> std::path::PathBuf {
    let dir = root.join(name);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn a_directory_with_a_safetensors_model_is_a_package() {
    let root = tempfile::tempdir().unwrap();
    let dir = directory(root.path(), "package");
    touch(&dir.join("model.safetensors"));
    touch(&dir.join("manifest.json"));
    assert_eq!(
        model_source_kind(&dir).unwrap(),
        ModelSourceKind::ModelPackage
    );
}

#[test]
fn a_directory_with_a_binary_model_is_a_checkpoint() {
    let root = tempfile::tempdir().unwrap();
    let dir = directory(root.path(), "checkpoint");
    touch(&dir.join("model.bin"));
    touch(&dir.join("manifest.json"));
    assert_eq!(
        model_source_kind(&dir).unwrap(),
        ModelSourceKind::TrainingCheckpoint
    );
}

#[test]
fn a_directory_holding_both_model_files_is_refused() {
    let root = tempfile::tempdir().unwrap();
    let dir = directory(root.path(), "both");
    touch(&dir.join("model.safetensors"));
    touch(&dir.join("model.bin"));
    let error = model_source_kind(&dir).unwrap_err();
    assert_eq!(error.code, ErrorCode::ModelIncompatible);
    assert_eq!(error.summary, "无法识别的模型目录");
    assert_eq!(error.stage, TaskStage::Preparing);
}

#[test]
fn a_directory_holding_neither_model_file_is_refused() {
    let root = tempfile::tempdir().unwrap();
    let dir = directory(root.path(), "empty");
    touch(&dir.join("manifest.json"));
    let error = model_source_kind(&dir).unwrap_err();
    assert_eq!(error.code, ErrorCode::ModelIncompatible);
}

#[test]
fn a_relative_source_is_refused_before_any_probe() {
    let error = model_source_kind(Path::new("model")).unwrap_err();
    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.summary, "模型目录必须是绝对路径");
}

#[test]
fn a_file_instead_of_a_directory_is_refused() {
    let root = tempfile::tempdir().unwrap();
    let file = root.path().join("model.safetensors");
    touch(&file);
    let error = model_source_kind(&file).unwrap_err();
    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.summary, "模型目录不可用");
}

#[test]
fn a_missing_source_is_refused() {
    let root = tempfile::tempdir().unwrap();
    let error = model_source_kind(&root.path().join("absent")).unwrap_err();
    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.summary, "模型目录不可用");
}

#[test]
fn both_kinds_have_a_stable_slug() {
    assert_eq!(ModelSourceKind::ModelPackage.as_slug(), "model_package");
    assert_eq!(
        ModelSourceKind::TrainingCheckpoint.as_slug(),
        "training_checkpoint"
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
cargo test -p feathertalk-worker --test inspecting
```

It fails to compile: `ModelSourceKind` and `model_source_kind` do not exist.

- [ ] **Step 3: Write minimal implementation**

`src/admission.rs`, appended before `invalid_request`:

```rust
/// What has to hold before a model directory is read: a real directory at an
/// absolute path. Which of the two layouts it is is `inspecting`'s question,
/// and whether the layout is complete is the readers' question.
pub(crate) fn check_model_source(source: &Path) -> Result<(), TaskError> {
    if !source.is_absolute() {
        return Err(invalid_request(
            "模型目录必须是绝对路径",
            format!("source {} is not absolute", source.display()),
        ));
    }
    let metadata = fs::symlink_metadata(source).map_err(|error| {
        invalid_request("模型目录不可用", format!("{}: {error}", source.display()))
    })?;
    if !metadata.is_dir() {
        return Err(invalid_request(
            "模型目录不可用",
            format!("{} is not a directory", source.display()),
        ));
    }
    Ok(())
}
```

`src/inspecting.rs`, the whole file for now:

```rust
use std::{fs, path::Path};

use feathertalk_domain::{ErrorCode, TaskError, TaskStage};
use feathertalk_export::MODEL_FILE_NAME;
use feathertalk_training::CHECKPOINT_MODEL_FILE_NAME;

use crate::{admission::check_model_source, error_map::clamp};

/// The two directory layouts `inspect_model` accepts (design section 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSourceKind {
    ModelPackage,
    TrainingCheckpoint,
}

impl ModelSourceKind {
    /// The wire spelling. It goes into the payload's `source_kind`, so it is
    /// part of the protocol and must not drift.
    pub fn as_slug(self) -> &'static str {
        match self {
            Self::ModelPackage => "model_package",
            Self::TrainingCheckpoint => "training_checkpoint",
        }
    }
}

/// Decide the layout from the one file only that layout has. An exported
/// package carries `model.safetensors`; a checkpoint carries `model.bin`.
/// Neither file is opened -- the digests in the manifests are the source of
/// truth for content, and reading weights is out of scope (design section 3).
pub fn model_source_kind(source: &Path) -> Result<ModelSourceKind, TaskError> {
    check_model_source(source)?;
    let package = is_regular_file(&source.join(MODEL_FILE_NAME));
    let checkpoint = is_regular_file(&source.join(CHECKPOINT_MODEL_FILE_NAME));
    match (package, checkpoint) {
        (true, false) => Ok(ModelSourceKind::ModelPackage),
        (false, true) => Ok(ModelSourceKind::TrainingCheckpoint),
        // Both means the directory is two things at once and neither reader
        // would be right; none means it is not a model directory at all.
        _ => Err(unrecognized(source)),
    }
}

fn is_regular_file(path: &Path) -> bool {
    fs::symlink_metadata(path)
        .map(|metadata| metadata.is_file())
        .unwrap_or(false)
}

fn unrecognized(source: &Path) -> TaskError {
    TaskError::new(
        ErrorCode::ModelIncompatible,
        "无法识别的模型目录",
        &clamp(&format!(
            "{} holds neither exactly one {MODEL_FILE_NAME} nor exactly one {CHECKPOINT_MODEL_FILE_NAME}",
            source.display()
        )),
        TaskStage::Preparing,
    )
}
```

`src/lib.rs` gains `mod inspecting;` and `pub use inspecting::{ModelSourceKind, model_source_kind};`.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cargo test -p feathertalk-worker --test inspecting
cargo fmt --all
cargo clippy -p feathertalk-worker --all-targets -- -D warnings
```

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/src/admission.rs rust/crates/feathertalk-worker/src/inspecting.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/tests/inspecting.rs
git commit -m "feat(worker): classify a model source directory"
```

---

### Task 3: Judge model compatibility

**Files:**

- Modify: `rust/crates/feathertalk-worker/src/inspecting.rs` (append the file listing and the two judges)
- Modify: `rust/crates/feathertalk-worker/src/lib.rs` (the new exports)
- Test: `rust/crates/feathertalk-worker/tests/support/mod.rs` (three fixtures)
- Test: `rust/crates/feathertalk-worker/tests/inspecting.rs` (append the compatibility tests)

**Interfaces:**

- Produces: `InspectedFile { file_name: String, bytes: u64, sha256: String, bytes_on_disk: Option<u64> }` with `agrees(&self) -> bool`.
- Produces: `package_files(dir, &ModelPackageManifest) -> Vec<InspectedFile>` (model, licenses) and `checkpoint_files(dir, &TrainingCheckpointManifest) -> Vec<InspectedFile>` (model, optimizer, training state).
- Produces: `package_incompatibilities(&ModelPackageManifest, &[InspectedFile], worker_version: &str) -> Vec<&'static str>` and `checkpoint_incompatibilities(&TrainingCheckpointManifest, &[InspectedFile]) -> Vec<&'static str>`.
- Consumes: `render_variant`, `RenderVariant::configuration`, `checkpoint_descriptor`, the manifests' own fields.

**Why now:** design section 4 -- "compatible" is the one answer the readers cannot give. `read_package_manifest` proves a package is well-formed; whether *this* build can run it is a different question, and it has exactly two shapes. A package declares `minimum_app_version`, so the check is a version comparison against `worker_version()`. A checkpoint declares `model_kind`, `architecture_version` and `model_config_sha256`, so the check is whether `render_variant` knows the kind and whether the descriptor this build would compute for that variant matches -- the same comparison `execute_render` makes before it loads weights, which is why an incompatible checkpoint can now be spotted without starting a render.

Version comparison is written here rather than pulled in: `semver` is not a workspace dependency, `minimum_app_version` is already validated to three numeric components by `ModelPackageManifest::validate`, and the constraint above forbids a new dependency. An unparseable version on either side counts as incompatible -- refusing to guess is the honest answer, and `ModelPackageManifest::validate` has already rejected the malformed manifest case.

- [ ] **Step 1: Write the failing test**

First the fixtures. `tests/support/mod.rs` opens with `#![allow(dead_code)]`, so a helper only one test binary uses is fine. Add:

```rust
/// Publishes a micro FeatherHuBERT package under `root/{name}` and returns it.
///
/// Ported from `tests/features.rs::published_package` with the minimum app
/// version as a parameter, because compatibility is exactly what varies here.
/// The real writer rather than hand-written files: the manifest declares the
/// size and digest of everything beside it, and only the writer makes them agree.
pub fn published_package(root: &Path, name: &str, minimum_app_version: &str) -> PathBuf {
    let source_path = root.join(format!("{name}-source.pth"));
    fs::write(&source_path, b"source-fixture").expect("the source fixture is written");
    let source_sha256 = hex::encode(Sha256::digest(b"source-fixture"));
    let licenses_path = root.join(format!("{name}-LICENSES.input.json"));
    let licenses = LicenseBundle {
        schema_version: 1,
        entries: vec![LicenseEntry {
            component: "synthetic FeatherHuBERT fixture".to_owned(),
            license_id: "LicenseRef-Test".to_owned(),
            source_url: "https://example.invalid/feather-hubert".to_owned(),
            notice: "test-only local record".to_owned(),
        }],
    };
    fs::write(&licenses_path, serde_json::to_vec(&licenses).expect("the bundle serialises"))
        .expect("the licenses fixture is written");
    let config = FeatherHubertConfig::parity_micro();
    let device = RenderDevice::default();
    let model = config.init::<RenderBackend>(&device);
    let request = PackageBuildRequest {
        destination: root.join(name),
        description: ModelDescription::feather_hubert(config.clone()),
        source_path,
        source: SourceManifest {
            format: "test".to_owned(),
            identifier: "feather-hubert-fixture".to_owned(),
            version: "1".to_owned(),
            file_name: format!("{name}-source.pth"),
            sha256: source_sha256,
            url: None,
        },
        licenses_path,
        created_at: "2026-08-27T00:00:00Z".to_owned(),
        minimum_app_version: minimum_app_version.to_owned(),
        training: TrainingManifest::default(),
    };
    write_model_package::<RenderBackend, FeatherHubertEncoder<RenderBackend>, _>(
        &request,
        &model,
        &device,
        |device| config.init::<RenderBackend>(device),
    )
    .expect("the package is written");
    request.destination
}

/// The training state a checkpoint carries. A copy of the private `state()` in
/// `tests/render.rs`: every field is one `TrainingCheckpointState::validate`
/// insists on, and the three cross-checked values come from `training_config`
/// rather than from a literal. `tests/render.rs` keeps its own copy -- folding
/// them together would mean re-verifying a committed test for no change in
/// behaviour.
pub fn checkpoint_state(project_dir: &Path, frame_count: u64) -> TrainingCheckpointState {
    let params = TrainParams {
        project_dir: project_dir.to_path_buf(),
        mode: DomainTrainingMode::Baseline,
        variant: UnetVariant::OriginalUnet,
        epochs: 1,
        resume: false,
    };
    let config = training_config(&params);
    let batch_size = config.batch_size;
    let temporal_stride = config.temporal_stride;
    TrainingCheckpointState {
        schema_version: TRAINING_STATE_SCHEMA_VERSION,
        epoch: 1,
        global_step: 2,
        random_seed: TRAINING_SEED,
        data_loader: DataLoaderState {
            schema_version: DATA_LOADER_STATE_SCHEMA_VERSION,
            random_algorithm: RandomAlgorithm::Splitmix64FisherYatesV1,
            config: DataLoaderConfig {
                batch_size,
                seed: TRAINING_SEED,
                sampling: SamplingConfig {
                    kind: SamplingKind::SingleFrame,
                    temporal_stride,
                },
            },
            frame_count,
            epoch: 1,
            next_position: 0,
        },
        training_config: config,
        asset_provenance: Provenance {
            entries: BTreeMap::new(),
        },
        model_provenance: Provenance {
            entries: BTreeMap::new(),
        },
    }
}

/// Writes a real checkpoint whose manifest carries `descriptor`. `directory`
/// must not exist: `save_training_checkpoint` creates it.
pub fn write_checkpoint(directory: &Path, descriptor: CheckpointDescriptor) {
    let device = TrainDevice::default();
    save_training_checkpoint::<TrainBackend, _, _>(
        directory,
        &model(&device),
        &AdamConfig::new().init(),
        descriptor,
        checkpoint_state(directory, 2),
    )
    .expect("the checkpoint is written");
}
```

New imports for `tests/support/mod.rs`: `std::collections::BTreeMap`, `std::fs`, `burn::optim::AdamConfig`, `feathertalk_domain::{TRAINING_STATE_SCHEMA_VERSION is training's}`, the export types `LicenseBundle, LicenseEntry, ModelDescription, PackageBuildRequest, SourceManifest, TrainingManifest, write_model_package`, `feathertalk_models::feather_hubert::{FeatherHubertConfig, FeatherHubertEncoder}`, the training items `CheckpointDescriptor, DATA_LOADER_STATE_SCHEMA_VERSION, DataLoaderConfig, DataLoaderState, Provenance, RandomAlgorithm, SamplingConfig, SamplingKind, TRAINING_STATE_SCHEMA_VERSION, TrainingCheckpointState, save_training_checkpoint`, `feathertalk_worker::TRAINING_SEED`, `sha2::{Digest, Sha256}`. Run `cargo fmt --all` and let it group them.

Then append to `tests/inspecting.rs` -- and add `mod support;` plus the fixture imports at the top:

```rust
#[test]
fn a_package_this_build_satisfies_is_compatible() {
    let root = tempfile::tempdir().unwrap();
    let dir = published_package(root.path(), "package", "0.1.0");
    let manifest = read_package_manifest(&dir).unwrap();
    let files = package_files(&dir, &manifest);
    assert_eq!(files.len(), 2);
    assert_eq!(files[0].file_name, "model.safetensors");
    assert_eq!(files[1].file_name, "LICENSES.json");
    assert!(files.iter().all(|file| file.agrees()));
    assert!(package_incompatibilities(&manifest, &files, "0.1.0").is_empty());
}

#[test]
fn a_package_that_wants_a_newer_app_is_incompatible() {
    let root = tempfile::tempdir().unwrap();
    let dir = published_package(root.path(), "package", "9.9.9");
    let manifest = read_package_manifest(&dir).unwrap();
    let files = package_files(&dir, &manifest);
    assert_eq!(
        package_incompatibilities(&manifest, &files, "0.1.0"),
        vec!["minimum_app_version"]
    );
}

#[test]
fn an_unparseable_worker_version_is_incompatible() {
    let root = tempfile::tempdir().unwrap();
    let dir = published_package(root.path(), "package", "0.1.0");
    let manifest = read_package_manifest(&dir).unwrap();
    let files = package_files(&dir, &manifest);
    assert_eq!(
        package_incompatibilities(&manifest, &files, "0.1"),
        vec!["minimum_app_version"]
    );
}

#[test]
fn a_truncated_model_file_is_reported_by_size() {
    let root = tempfile::tempdir().unwrap();
    let dir = published_package(root.path(), "package", "0.1.0");
    let manifest = read_package_manifest(&dir).unwrap();
    fs::write(dir.join("model.safetensors"), b"truncated").unwrap();
    let files = package_files(&dir, &manifest);
    assert!(!files[0].agrees());
    assert_eq!(
        package_incompatibilities(&manifest, &files, "0.1.0"),
        vec!["file_size"]
    );
}

#[test]
fn a_checkpoint_this_build_can_rebuild_is_compatible() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("checkpoint");
    let variant = render_variant("original_unet").unwrap();
    let descriptor = checkpoint_descriptor(&variant.configuration()).unwrap();
    write_checkpoint(&dir, descriptor);
    let checkpoint = read_training_checkpoint(&dir).unwrap();
    let files = checkpoint_files(&dir, &checkpoint.manifest);
    assert_eq!(files.len(), 3);
    assert_eq!(files[0].file_name, "model.bin");
    assert_eq!(files[1].file_name, "optimizer.bin");
    assert_eq!(files[2].file_name, "training-state.json");
    assert!(checkpoint_incompatibilities(&checkpoint.manifest, &files).is_empty());
}

#[test]
fn a_checkpoint_of_an_unknown_kind_is_incompatible() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("checkpoint");
    write_checkpoint(&dir, CheckpointDescriptor::new("legacy_unet", "v1", "0".repeat(64)));
    let checkpoint = read_training_checkpoint(&dir).unwrap();
    let files = checkpoint_files(&dir, &checkpoint.manifest);
    // The kind decides which configuration to compare against, so an unknown
    // kind is the only reason reported: the rest cannot be computed.
    assert_eq!(
        checkpoint_incompatibilities(&checkpoint.manifest, &files),
        vec!["model_kind"]
    );
}

#[test]
fn a_checkpoint_of_another_architecture_is_incompatible() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("checkpoint");
    let variant = render_variant("original_unet").unwrap();
    let mine = checkpoint_descriptor(&variant.configuration()).unwrap();
    write_checkpoint(
        &dir,
        CheckpointDescriptor::new("original_unet", "unet-burn-v0", &mine.model_config_sha256),
    );
    let checkpoint = read_training_checkpoint(&dir).unwrap();
    let files = checkpoint_files(&dir, &checkpoint.manifest);
    assert_eq!(
        checkpoint_incompatibilities(&checkpoint.manifest, &files),
        vec!["architecture_version"]
    );
}

#[test]
fn a_checkpoint_of_another_configuration_is_incompatible() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("checkpoint");
    let variant = render_variant("original_unet").unwrap();
    let mine = checkpoint_descriptor(&variant.configuration()).unwrap();
    write_checkpoint(
        &dir,
        CheckpointDescriptor::new("original_unet", &mine.architecture_version, "0".repeat(64)),
    );
    let checkpoint = read_training_checkpoint(&dir).unwrap();
    let files = checkpoint_files(&dir, &checkpoint.manifest);
    assert_eq!(
        checkpoint_incompatibilities(&checkpoint.manifest, &files),
        vec!["model_config_sha256"]
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
cargo test -p feathertalk-worker --test inspecting
```

It fails to compile on the four new functions and `InspectedFile`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/inspecting.rs`:

```rust
/// The reason codes a client may see. They are identifiers, not sentences: the
/// CLI and the UI own the wording, and a rename here is a protocol change.
const REASON_MINIMUM_APP_VERSION: &str = "minimum_app_version";
const REASON_MODEL_KIND: &str = "model_kind";
const REASON_ARCHITECTURE_VERSION: &str = "architecture_version";
const REASON_MODEL_CONFIG_SHA256: &str = "model_config_sha256";
const REASON_FILE_SIZE: &str = "file_size";

/// One file a manifest names, as the manifest describes it and as the disk
/// answers. `bytes_on_disk` is `None` when the file could not be stated at all,
/// which is reported as a size disagreement rather than as a zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InspectedFile {
    pub file_name: String,
    pub bytes: u64,
    pub sha256: String,
    pub bytes_on_disk: Option<u64>,
}

impl InspectedFile {
    /// Whether the disk agrees with the manifest about the size. The digest is
    /// not recomputed: hashing gigabytes to answer a listing would make an
    /// interactive command a long job (design section 3).
    pub fn agrees(&self) -> bool {
        self.bytes_on_disk == Some(self.bytes)
    }

    fn from_manifest(directory: &Path, file_name: &str, bytes: u64, sha256: &str) -> Self {
        Self {
            file_name: file_name.to_owned(),
            bytes,
            sha256: sha256.to_owned(),
            bytes_on_disk: fs::symlink_metadata(directory.join(file_name))
                .ok()
                .filter(|metadata| metadata.is_file())
                .map(|metadata| metadata.len()),
        }
    }
}

/// The two files an exported package manifest names, in manifest order.
pub fn package_files(directory: &Path, manifest: &ModelPackageManifest) -> Vec<InspectedFile> {
    [&manifest.model, &manifest.licenses]
        .into_iter()
        .map(|file| {
            InspectedFile::from_manifest(directory, &file.file_name, file.bytes, &file.sha256)
        })
        .collect()
}

/// The three files a checkpoint manifest names, in manifest order.
pub fn checkpoint_files(
    directory: &Path,
    manifest: &TrainingCheckpointManifest,
) -> Vec<InspectedFile> {
    [
        &manifest.model,
        &manifest.optimizer,
        &manifest.training_state,
    ]
    .into_iter()
    .map(|file| InspectedFile::from_manifest(directory, &file.file_name, file.bytes, &file.sha256))
    .collect()
}

/// Why this build cannot use this package, in a fixed order.
pub fn package_incompatibilities(
    manifest: &ModelPackageManifest,
    files: &[InspectedFile],
    worker_version: &str,
) -> Vec<&'static str> {
    let mut reasons = Vec::new();
    if !version_at_least(worker_version, &manifest.minimum_app_version) {
        reasons.push(REASON_MINIMUM_APP_VERSION);
    }
    push_file_size(&mut reasons, files);
    reasons
}

/// Why this build cannot use this checkpoint, in a fixed order.
pub fn checkpoint_incompatibilities(
    manifest: &TrainingCheckpointManifest,
    files: &[InspectedFile],
) -> Vec<&'static str> {
    let mut reasons = Vec::new();
    match render_variant(&manifest.model_kind) {
        // Without a known kind there is no configuration to compare against, so
        // the architecture and digest checks are not "passed", they are unasked.
        None => reasons.push(REASON_MODEL_KIND),
        Some(variant) => match checkpoint_descriptor(&variant.configuration()) {
            Err(_) => reasons.push(REASON_MODEL_KIND),
            Ok(mine) => {
                if manifest.architecture_version != mine.architecture_version {
                    reasons.push(REASON_ARCHITECTURE_VERSION);
                }
                if manifest.model_config_sha256 != mine.model_config_sha256 {
                    reasons.push(REASON_MODEL_CONFIG_SHA256);
                }
            }
        },
    }
    push_file_size(&mut reasons, files);
    reasons
}

fn push_file_size(reasons: &mut Vec<&'static str>, files: &[InspectedFile]) {
    if files.iter().any(|file| !file.agrees()) {
        reasons.push(REASON_FILE_SIZE);
    }
}

/// Whether `have` is at least `want`, both `major.minor.patch`. Written here
/// rather than pulled in: `semver` is not a workspace dependency, and
/// `ModelPackageManifest::validate` has already pinned the manifest side to
/// three numeric components. An unparseable version is not "at least" anything.
fn version_at_least(have: &str, want: &str) -> bool {
    match (parse_version(have), parse_version(want)) {
        (Some(have), Some(want)) => have >= want,
        _ => false,
    }
}

fn parse_version(text: &str) -> Option<(u64, u64, u64)> {
    let mut parts = text.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((major, minor, patch))
}
```

New imports: `feathertalk_export::ModelPackageManifest`, `feathertalk_training::TrainingCheckpointManifest`, and `crate::{checkpoint_descriptor, rendering::render_variant}` -- whichever path `cargo fmt` and the existing module layout prefer. `src/lib.rs` exports `InspectedFile`, `checkpoint_files`, `checkpoint_incompatibilities`, `package_files` and `package_incompatibilities`.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cargo test -p feathertalk-worker --test inspecting
cargo fmt --all
cargo clippy -p feathertalk-worker --all-targets -- -D warnings
```

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/src/inspecting.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/tests/inspecting.rs rust/crates/feathertalk-worker/tests/support/mod.rs
git commit -m "feat(worker): judge model compatibility"
```

---

### Task 4: Report an inspected model

**Files:**

- Create: `rust/crates/feathertalk-worker/src/inspect_result.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs`
- Test: `rust/crates/feathertalk-worker/tests/inspect_result.rs` (new)

**Interfaces:**

- Produces: `InspectedModel<'a> { Package(&'a ModelPackageManifest), Checkpoint(&'a TrainingCheckpointMetadata) }`.
- Produces: `InspectSummary<'a> { source_kind, source_path, model, files, incompatibilities }`.
- Produces: `inspect_to_json(&InspectSummary<'_>) -> Value`, eighteen keys for either layout.
- Consumes: `ModelSourceKind::as_slug`, `InspectedFile`, `TensorSpec`, both `TrainingMode` enums.

**Why now:** design section 6 -- the payload is the whole visible product of this command, and the two layouts have to answer with the same shape or every client would need two parsers. Eighteen keys, always present, `null` where the layout genuinely has nothing to say: a package has no `epoch`, a checkpoint has no `parameter_count` because counting one would mean reading the record. A struct rather than a long argument list, for the reason `train_result.rs` gives: most of these are strings and counters, so a wrong order would type-check.

- [ ] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-worker/tests/inspect_result.rs`:

```rust
mod support;

use std::path::Path;

use feathertalk_export::read_package_manifest;
use feathertalk_training::{CheckpointDescriptor, read_training_checkpoint};
use feathertalk_worker::{
    InspectSummary, InspectedModel, ModelSourceKind, checkpoint_descriptor, checkpoint_files,
    inspect_to_json, package_files, render_variant,
};
use support::{published_package, write_checkpoint};

/// Every key the payload promises, sorted. Both layouts answer with all of them.
const KEYS: [&str; 18] = [
    "architecture_version",
    "compatible",
    "created_at",
    "epoch",
    "files",
    "global_step",
    "incompatibilities",
    "inputs",
    "minimum_app_version",
    "model_config_sha256",
    "model_kind",
    "outputs",
    "parameter_count",
    "schema_version",
    "source_kind",
    "source_path",
    "tensor_count",
    "training_mode",
];

fn keys(value: &serde_json::Value) -> Vec<String> {
    let mut names: Vec<String> = value
        .as_object()
        .expect("the payload is an object")
        .keys()
        .cloned()
        .collect();
    names.sort();
    names
}

#[test]
fn a_package_payload_answers_every_key() {
    let root = tempfile::tempdir().unwrap();
    let dir = published_package(root.path(), "package", "0.1.0");
    let manifest = read_package_manifest(&dir).unwrap();
    let files = package_files(&dir, &manifest);
    let payload = inspect_to_json(&InspectSummary {
        source_kind: ModelSourceKind::ModelPackage,
        source_path: &dir,
        model: InspectedModel::Package(&manifest),
        files: &files,
        incompatibilities: &[],
    });

    assert_eq!(keys(&payload), KEYS);
    assert_eq!(payload["source_kind"], "model_package");
    assert_eq!(payload["source_path"], dir.display().to_string());
    assert_eq!(payload["schema_version"], 1);
    assert_eq!(payload["model_kind"], "feather_hubert");
    assert_eq!(payload["training_mode"], "inference");
    // Counting a checkpoint's parameters would mean reading its record, so only
    // a package -- whose manifest states them -- reports these two.
    assert!(payload["parameter_count"].as_u64().unwrap() > 0);
    assert!(payload["tensor_count"].as_u64().unwrap() > 0);
    assert_eq!(payload["inputs"][0]["name"], "waveform");
    assert_eq!(payload["outputs"][0]["name"], "hidden");
    assert!(payload["inputs"][0]["shape"].is_array());
    assert_eq!(payload["minimum_app_version"], "0.1.0");
    assert_eq!(payload["created_at"], "2026-08-27T00:00:00Z");
    assert!(payload["model_config_sha256"].is_null());
    assert!(payload["epoch"].is_null());
    assert!(payload["global_step"].is_null());
    assert_eq!(payload["files"].as_array().unwrap().len(), 2);
    assert_eq!(payload["files"][0]["file_name"], "model.safetensors");
    assert_eq!(payload["files"][0]["bytes"], payload["files"][0]["bytes_on_disk"]);
    assert_eq!(payload["compatible"], true);
    assert_eq!(payload["incompatibilities"].as_array().unwrap().len(), 0);
}

#[test]
fn a_checkpoint_payload_answers_the_same_keys() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("checkpoint");
    let variant = render_variant("original_unet").unwrap();
    write_checkpoint(&dir, checkpoint_descriptor(&variant.configuration()).unwrap());
    let checkpoint = read_training_checkpoint(&dir).unwrap();
    let files = checkpoint_files(&dir, &checkpoint.manifest);
    let payload = inspect_to_json(&InspectSummary {
        source_kind: ModelSourceKind::TrainingCheckpoint,
        source_path: &dir,
        model: InspectedModel::Checkpoint(&checkpoint),
        files: &files,
        incompatibilities: &[],
    });

    assert_eq!(keys(&payload), KEYS);
    assert_eq!(payload["source_kind"], "training_checkpoint");
    assert_eq!(payload["model_kind"], "original_unet");
    assert_eq!(payload["training_mode"], "baseline");
    assert_eq!(payload["epoch"], 1);
    assert_eq!(payload["global_step"], 2);
    assert!(payload["model_config_sha256"].is_string());
    assert!(payload["parameter_count"].is_null());
    assert!(payload["tensor_count"].is_null());
    assert!(payload["created_at"].is_null());
    assert!(payload["minimum_app_version"].is_null());
    assert_eq!(payload["inputs"].as_array().unwrap().len(), 0);
    assert_eq!(payload["outputs"].as_array().unwrap().len(), 0);
    assert_eq!(payload["files"].as_array().unwrap().len(), 3);
    assert_eq!(payload["compatible"], true);
}

#[test]
fn any_reason_makes_the_model_incompatible() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("checkpoint");
    write_checkpoint(&dir, CheckpointDescriptor::new("legacy_unet", "v1", "0".repeat(64)));
    let checkpoint = read_training_checkpoint(&dir).unwrap();
    let files = checkpoint_files(&dir, &checkpoint.manifest);
    let payload = inspect_to_json(&InspectSummary {
        source_kind: ModelSourceKind::TrainingCheckpoint,
        source_path: Path::new("/models/legacy"),
        model: InspectedModel::Checkpoint(&checkpoint),
        files: &files,
        incompatibilities: &["model_kind"],
    });

    assert_eq!(payload["compatible"], false);
    assert_eq!(payload["incompatibilities"], serde_json::json!(["model_kind"]));
}
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
cargo test -p feathertalk-worker --test inspect_result
```

It fails to compile: `InspectSummary`, `InspectedModel` and `inspect_to_json` do not exist.

- [ ] **Step 3: Write minimal implementation**

Create `src/inspect_result.rs`:

```rust
//! The JSON payload an inspected model returns.

use std::path::Path;

use feathertalk_export::{
    ModelPackageManifest, TensorSpec, TrainingMode as PackageTrainingMode,
};
use feathertalk_training::{TrainingCheckpointMetadata, TrainingMode as CheckpointTrainingMode};
use serde_json::{Value, json};

use crate::{InspectedFile, ModelSourceKind};

/// Which of the two layouts was read, with the manifest that came out of it.
#[derive(Debug)]
pub enum InspectedModel<'a> {
    Package(&'a ModelPackageManifest),
    Checkpoint(&'a TrainingCheckpointMetadata),
}

/// What an inspected model has to say for itself.
#[derive(Debug)]
pub struct InspectSummary<'a> {
    pub source_kind: ModelSourceKind,
    pub source_path: &'a Path,
    pub model: InspectedModel<'a>,
    pub files: &'a [InspectedFile],
    /// The reason codes from `inspecting`; empty means usable by this build.
    pub incompatibilities: &'a [&'static str],
}

/// Shapes the payload the `completed` event of an inspect task carries.
///
/// Both arms are written out in full rather than merged from a common base: the
/// key set is the contract (design section 6), and spelling it twice is what
/// makes a missing key a diff instead of a runtime surprise. Where a layout has
/// no answer the value is `null`, never a zero -- a checkpoint with
/// `parameter_count: 0` would read as an empty model.
pub fn inspect_to_json(summary: &InspectSummary<'_>) -> Value {
    let compatible = summary.incompatibilities.is_empty();
    let source_path = path_text(summary.source_path);
    let files = files_json(summary.files);
    match summary.model {
        InspectedModel::Package(manifest) => json!({
            "source_kind": summary.source_kind.as_slug(),
            "source_path": source_path,
            "schema_version": manifest.schema_version,
            "model_kind": manifest.model_type.as_str(),
            "architecture_version": manifest.architecture_version.as_str(),
            // A package is a published artifact, not a resume point: it carries
            // no configuration digest to match a checkpoint against.
            "model_config_sha256": Value::Null,
            "parameter_count": manifest.tensors.total_elements,
            "tensor_count": manifest.tensors.tensor_count,
            "inputs": specs_json(&manifest.inputs),
            "outputs": specs_json(&manifest.outputs),
            "training_mode": package_mode_slug(manifest.training.mode),
            "epoch": Value::Null,
            "global_step": Value::Null,
            "created_at": manifest.created_at.as_str(),
            "minimum_app_version": manifest.minimum_app_version.as_str(),
            "files": files,
            "compatible": compatible,
            "incompatibilities": summary.incompatibilities,
        }),
        InspectedModel::Checkpoint(checkpoint) => json!({
            "source_kind": summary.source_kind.as_slug(),
            "source_path": source_path,
            "schema_version": checkpoint.manifest.schema_version,
            "model_kind": checkpoint.manifest.model_kind.as_str(),
            "architecture_version": checkpoint.manifest.architecture_version.as_str(),
            "model_config_sha256": checkpoint.manifest.model_config_sha256.as_str(),
            // Counting parameters means reading the record, which this command
            // does not do (design section 3).
            "parameter_count": Value::Null,
            "tensor_count": Value::Null,
            "inputs": Value::Array(Vec::new()),
            "outputs": Value::Array(Vec::new()),
            "training_mode": checkpoint_mode_slug(checkpoint.state.training_config.mode),
            "epoch": checkpoint.state.epoch,
            "global_step": checkpoint.state.global_step,
            // A checkpoint manifest records the toolchain that wrote it, not a
            // timestamp or an app floor.
            "created_at": Value::Null,
            "minimum_app_version": Value::Null,
            "files": files,
            "compatible": compatible,
            "incompatibilities": summary.incompatibilities,
        }),
    }
}

fn specs_json(specs: &[TensorSpec]) -> Value {
    Value::Array(
        specs
            .iter()
            .map(|spec| {
                json!({
                    "name": spec.name.as_str(),
                    // A dynamic axis is -1 in the manifest and stays -1 here.
                    "shape": spec.shape,
                    "dtype": spec.dtype.as_str(),
                })
            })
            .collect(),
    )
}

fn files_json(files: &[InspectedFile]) -> Value {
    Value::Array(
        files
            .iter()
            .map(|file| {
                json!({
                    "file_name": file.file_name.as_str(),
                    "bytes": file.bytes,
                    "sha256": file.sha256.as_str(),
                    "bytes_on_disk": file.bytes_on_disk,
                })
            })
            .collect(),
    )
}

/// Matched exhaustively rather than serialised, so a fifth mode is a compile
/// error here instead of a surprise string in the payload.
fn package_mode_slug(mode: PackageTrainingMode) -> &'static str {
    match mode {
        PackageTrainingMode::Inference => "inference",
        PackageTrainingMode::Baseline => "baseline",
        PackageTrainingMode::MouthRoi => "mouth_roi",
        PackageTrainingMode::MouthRoiTemporal => "mouth_roi_temporal",
    }
}

/// The checkpoint enum has no `Inference`: a checkpoint is by definition mid-training.
fn checkpoint_mode_slug(mode: CheckpointTrainingMode) -> &'static str {
    match mode {
        CheckpointTrainingMode::Baseline => "baseline",
        CheckpointTrainingMode::MouthRoi => "mouth_roi",
        CheckpointTrainingMode::MouthRoiTemporal => "mouth_roi_temporal",
    }
}

fn path_text(path: &Path) -> String {
    path.display().to_string()
}
```

`src/lib.rs` gains `mod inspect_result;` and `pub use inspect_result::{InspectSummary, InspectedModel, inspect_to_json};`.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cargo test -p feathertalk-worker --test inspect_result
cargo fmt --all
cargo clippy -p feathertalk-worker --all-targets -- -D warnings
```

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/src/inspect_result.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/tests/inspect_result.rs
git commit -m "feat(worker): report an inspected model"
```

---

### Task 5: Execute the inspect-model command

**Files:**

- Create: `rust/crates/feathertalk-worker/src/inspect.rs`
- Modify: `rust/crates/feathertalk-worker/src/commands.rs:160-176` (a new arm before the catch-all)
- Modify: `rust/crates/feathertalk-worker/src/lib.rs`
- Test: `rust/crates/feathertalk-worker/tests/inspect.rs` (new)

**Interfaces:**

- Produces: `execute_inspect_model(&InspectModelParams, &WorkerConfig, &CancellationToken) -> CommandOutcome`.
- Consumes: `model_source_kind`, `read_package_manifest`, `read_training_checkpoint`, `package_task_error`, `training_task_error`, `inspect_to_json`, `WorkerConfig::worker_version`.

**Why now:** the parts exist; this is the body that joins them. Signature notes, all from design sections 5 and 7: no `TaskReporter` parameter and no progress events, because reading three manifests has no phases worth reporting and a progress event per file would be noise; `&WorkerConfig` only for `worker_version()`, which is the value the package compatibility check compares against; and the cancellation token is checked twice -- once before the readers and once before the payload is built -- so a cancel that arrives during a slow directory walk is still honoured. No toolchain guard in `commands.rs`: the handshake announces the command unconditionally, so a guard could only ever produce an unreachable rejection.

- [ ] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-worker/tests/inspect.rs`:

```rust
mod support;

use std::{fs, path::Path};

use feathertalk_domain::{ErrorCode, InspectModelParams, TaskError, TaskStage};
use feathertalk_media::CancellationToken;
use feathertalk_training::CheckpointDescriptor;
use feathertalk_worker::{
    CommandOutcome, WorkerConfig, checkpoint_descriptor, execute_inspect_model, render_variant,
};
use serde_json::Value;
use support::{published_package, write_checkpoint};

/// Inspection needs no toolchain at all, which is the point of the command.
fn config() -> WorkerConfig {
    WorkerConfig::from_values(None, None, None)
}

fn params(source: &Path) -> InspectModelParams {
    InspectModelParams {
        source: source.to_path_buf(),
    }
}

fn failed(outcome: CommandOutcome) -> TaskError {
    match outcome {
        CommandOutcome::Failed(error) => error,
        other => panic!("expected a failure, got {other:?}"),
    }
}

fn completed(outcome: CommandOutcome) -> Value {
    match outcome {
        CommandOutcome::Completed(Some(payload)) => payload,
        other => panic!("expected a payload, got {other:?}"),
    }
}

#[test]
fn a_relative_source_is_refused_before_anything_is_read() {
    let outcome = execute_inspect_model(
        &params(Path::new("models/hubert")),
        &config(),
        &CancellationToken::new(),
    );
    let error = failed(outcome);
    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.summary, "模型目录必须是绝对路径");
    assert_eq!(error.stage, TaskStage::Preparing);
}

#[test]
fn a_cancelled_token_stops_before_the_first_read() {
    let root = tempfile::tempdir().unwrap();
    let token = CancellationToken::new();
    token.cancel();
    // The source need not even exist: nothing is read.
    let outcome = execute_inspect_model(&params(&root.path().join("absent")), &config(), &token);
    assert!(matches!(outcome, CommandOutcome::Cancelled));
}

#[test]
fn a_real_package_is_inspected() {
    let root = tempfile::tempdir().unwrap();
    let dir = published_package(root.path(), "package", "0.1.0");
    let payload = completed(execute_inspect_model(
        &params(&dir),
        &config(),
        &CancellationToken::new(),
    ));
    assert_eq!(payload["source_kind"], "model_package");
    assert_eq!(payload["model_kind"], "feather_hubert");
    assert_eq!(payload["compatible"], true);
    assert_eq!(payload["files"].as_array().unwrap().len(), 2);
}

#[test]
fn a_real_checkpoint_is_inspected() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("checkpoint");
    let variant = render_variant("original_unet").unwrap();
    write_checkpoint(&dir, checkpoint_descriptor(&variant.configuration()).unwrap());
    let payload = completed(execute_inspect_model(
        &params(&dir),
        &config(),
        &CancellationToken::new(),
    ));
    assert_eq!(payload["source_kind"], "training_checkpoint");
    assert_eq!(payload["model_kind"], "original_unet");
    assert_eq!(payload["global_step"], 2);
    assert_eq!(payload["compatible"], true);
}

#[test]
fn an_incompatible_checkpoint_is_still_reported() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("checkpoint");
    write_checkpoint(&dir, CheckpointDescriptor::new("legacy_unet", "v1", "0".repeat(64)));
    let payload = completed(execute_inspect_model(
        &params(&dir),
        &config(),
        &CancellationToken::new(),
    ));
    // An unusable model is an answer, not an error (design section 4).
    assert_eq!(payload["compatible"], false);
    assert_eq!(payload["incompatibilities"], serde_json::json!(["model_kind"]));
}

#[test]
fn a_directory_that_is_neither_layout_is_refused() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("mystery");
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("weights.pth"), b"x").unwrap();
    let error = failed(execute_inspect_model(
        &params(&dir),
        &config(),
        &CancellationToken::new(),
    ));
    assert_eq!(error.code, ErrorCode::ModelIncompatible);
    assert_eq!(error.summary, "无法识别的模型目录");
}

#[test]
fn a_broken_manifest_is_a_model_error() {
    let root = tempfile::tempdir().unwrap();
    let dir = published_package(root.path(), "package", "0.1.0");
    fs::write(dir.join("manifest.json"), b"not json").unwrap();
    let error = failed(execute_inspect_model(
        &params(&dir),
        &config(),
        &CancellationToken::new(),
    ));
    assert_eq!(error.code, ErrorCode::ModelIncompatible);
    assert_eq!(error.stage, TaskStage::Preparing);
}
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
cargo test -p feathertalk-worker --test inspect
```

It fails to compile: `execute_inspect_model` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `src/inspect.rs`:

```rust
//! The `inspect_model` command: what a model directory says about itself.

use feathertalk_domain::{InspectModelParams, TaskStage};
use feathertalk_export::read_package_manifest;
use feathertalk_media::CancellationToken;
use feathertalk_training::read_training_checkpoint;

use crate::{
    CommandOutcome, InspectSummary, InspectedModel, ModelSourceKind, WorkerConfig,
    checkpoint_files, checkpoint_incompatibilities, error_map::package_task_error,
    error_map::training_task_error, inspect_result::inspect_to_json, inspecting::model_source_kind,
    package_files, package_incompatibilities,
};

/// Reads the manifests of a model directory and reports what is in it.
///
/// No reporter and no progress events: three manifests and a handful of
/// `symlink_metadata` calls have no phase a client could act on (design section
/// 7). The token is checked twice all the same -- before the readers and before
/// the payload -- so a cancel during a slow directory walk still lands.
pub fn execute_inspect_model(
    params: &InspectModelParams,
    config: &WorkerConfig,
    token: &CancellationToken,
) -> CommandOutcome {
    if token.is_cancelled() {
        return CommandOutcome::Cancelled;
    }
    let source = params.source.as_path();
    let kind = match model_source_kind(source) {
        Ok(kind) => kind,
        Err(error) => return CommandOutcome::Failed(error),
    };
    let payload = match kind {
        ModelSourceKind::ModelPackage => {
            let manifest = match read_package_manifest(source) {
                Ok(manifest) => manifest,
                Err(error) => return CommandOutcome::Failed(package_task_error(&error)),
            };
            let files = package_files(source, &manifest);
            let reasons = package_incompatibilities(&manifest, &files, config.worker_version());
            if token.is_cancelled() {
                return CommandOutcome::Cancelled;
            }
            inspect_to_json(&InspectSummary {
                source_kind: kind,
                source_path: source,
                model: InspectedModel::Package(&manifest),
                files: &files,
                incompatibilities: &reasons,
            })
        }
        ModelSourceKind::TrainingCheckpoint => {
            let checkpoint = match read_training_checkpoint(source) {
                Ok(checkpoint) => checkpoint,
                Err(error) => {
                    return CommandOutcome::Failed(training_task_error(&error, TaskStage::Preparing));
                }
            };
            let files = checkpoint_files(source, &checkpoint.manifest);
            let reasons = checkpoint_incompatibilities(&checkpoint.manifest, &files);
            if token.is_cancelled() {
                return CommandOutcome::Cancelled;
            }
            inspect_to_json(&InspectSummary {
                source_kind: kind,
                source_path: source,
                model: InspectedModel::Checkpoint(&checkpoint),
                files: &files,
                incompatibilities: &reasons,
            })
        }
    };
    CommandOutcome::Completed(Some(payload))
}
```

`src/commands.rs`, a new arm immediately before `other =>`:

```rust
        // No toolchain guard: inspection reads manifests, so the handshake
        // announces it unconditionally and there is nothing to reject on.
        Request::InspectModel(params) => execute_inspect_model(params, config, token),
```

`src/lib.rs` gains `mod inspect;` and `pub use inspect::execute_inspect_model;`. Let `cargo fmt --all` settle the import block in `inspect.rs`.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cargo test -p feathertalk-worker --test inspect
cargo test -p feathertalk-worker
cargo fmt --all
cargo clippy -p feathertalk-worker --all-targets -- -D warnings
```

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/src/commands.rs rust/crates/feathertalk-worker/src/inspect.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/tests/inspect.rs
git commit -m "feat(worker): execute the inspect-model command"
```

---

### Task 6: Add the inspect-model subcommand

**Files:**

- Modify: `rust/crates/feathertalk-cli/src/cli.rs` (a variant after `Render`, before `Capabilities`)
- Modify: `rust/crates/feathertalk-cli/src/run.rs` (the `build_request` arm and two inline tests)

**Interfaces:**

- Produces: `feathertalk inspect-model <SOURCE>`, building `Request::InspectModel(InspectModelParams { source })`.
- Consumes: `reject_empty`, `InspectModelParams`.

**Why now:** the worker serves the command; nothing reaches it until clap knows the verb. One positional argument and no flags (design section 7): the worker decides which of the two layouts the directory is, so asking the operator to declare it would be asking them to repeat what the files already say -- and to be wrong sometimes.

No branch in `cli/src/render.rs`: that file's hint tells an operator which environment variable a rejected command needed, and inspection is never rejected for a missing toolchain.

- [ ] **Step 1: Write the failing test**

Append to the `mod tests` block in `rust/crates/feathertalk-cli/src/run.rs`:

```rust
    #[test]
    fn inspect_model_refuses_an_empty_path_by_name() {
        let error = build_request(&Command::InspectModel {
            source: PathBuf::new(),
        })
        .expect_err("an empty source is refused");
        assert_eq!(error, "模型目录不能为空。");
    }

    #[test]
    fn inspect_model_carries_the_source_into_the_request() {
        let request = build_request(&Command::InspectModel {
            source: PathBuf::from("models/hubert"),
        })
        .expect("the arguments are accepted")
        .expect("inspect-model needs a task");
        let Request::InspectModel(params) = request else {
            panic!("inspect-model must build an InspectModel request");
        };
        // Relative here, absolute demanded by the worker: whether a path is
        // usable is the worker's judgement, like every other path in this file.
        assert_eq!(params.source, PathBuf::from("models/hubert"));
    }
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
cargo test -p feathertalk-cli
```

It fails to compile: `Command::InspectModel` does not exist.

- [ ] **Step 3: Write minimal implementation**

`src/cli.rs`, between `Render` and `Capabilities`:

```rust
    /// 检视模型：读取模型包或训练检查点的清单，报告类型、参数量、哈希与兼容状态
    InspectModel {
        /// 模型包目录或训练检查点目录
        source: PathBuf,
    },
```

`src/run.rs`, a new arm after `Command::Render`:

```rust
        Command::InspectModel { source } => {
            reject_empty(source, "模型目录")?;
            Ok(Some(Request::InspectModel(InspectModelParams {
                source: source.clone(),
            })))
        }
```

plus `InspectModelParams` in the `feathertalk_domain` import list.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cargo test -p feathertalk-cli
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
```

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-cli/src/cli.rs rust/crates/feathertalk-cli/src/run.rs
git commit -m "feat(cli): add the inspect-model subcommand"
```

---

### Task 7: Inspect a real model package end to end

**Files:**

- Test: `rust/crates/feathertalk-cli/tests/real_worker.rs` (one new test, plus one assertion in `capabilities_reports_the_real_handshake` at `:94-97`)

**Interfaces:**

- Consumes: `worker_or_skip`, `real_dir("HUBERT_DIR")`, `run`, `code`, `stdout`, `stderr`.

**Why now:** every unit test above inspects a package this repository's own code wrote. This one inspects the FeatherHuBERT package on the machine, through the real binaries and the real JSON Lines protocol, which is the only way to prove the payload survives the round trip. It needs no ffmpeg and no demo clip -- inspection reads manifests -- so it is the cheapest end-to-end test in the file.

- [ ] **Step 1: Write the failing test**

```rust
/// The real FeatherHuBERT package, read through the real protocol. No ffmpeg and
/// no demo clip: inspection reads manifests and nothing else.
#[test]
fn a_real_model_package_is_inspected_end_to_end() {
    let Some(worker) = worker_or_skip("a_real_model_package_is_inspected_end_to_end") else {
        return;
    };
    let Some(hubert) = real_dir("HUBERT_DIR") else {
        println!(
            "skipping a_real_model_package_is_inspected_end_to_end: it needs \
             FEATHERTALK_WORKER_HUBERT_DIR"
        );
        return;
    };
    let source_arg = hubert.to_string_lossy().into_owned();
    let output = run(&worker, &["inspect-model", &source_arg], &[]);
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));
    let inspected: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("stdout is exactly one JSON document");

    assert_eq!(inspected["source_kind"], "model_package");
    assert_eq!(inspected["source_path"], hubert.display().to_string());
    assert_eq!(inspected["schema_version"], 1);
    assert_eq!(inspected["model_kind"], "feather_hubert");
    assert_eq!(inspected["architecture_version"], "feather-hubert-burn-v1");
    // A published package is not a resume point and carries no step counter.
    assert!(inspected["model_config_sha256"].is_null());
    assert!(inspected["epoch"].is_null());
    assert_eq!(inspected["training_mode"], "inference");
    assert!(inspected["parameter_count"].as_u64().unwrap() > 0);
    assert!(inspected["tensor_count"].as_u64().unwrap() > 0);
    assert_eq!(inspected["inputs"][0]["name"], "waveform");
    assert_eq!(inspected["outputs"][0]["name"], "hidden");

    let files = inspected["files"].as_array().expect("files is an array");
    assert_eq!(files.len(), 2);
    for file in files {
        assert_eq!(file["sha256"].as_str().unwrap().len(), 64);
        // The package on this machine is intact, so the manifest and the disk
        // agree -- which is also what makes `compatible` true below.
        assert_eq!(file["bytes"], file["bytes_on_disk"]);
    }

    assert_eq!(inspected["compatible"], true);
    assert_eq!(inspected["incompatibilities"].as_array().unwrap().len(), 0);
}
```

And in `capabilities_reports_the_real_handshake`, beside the `validate_project` assertion (the comment above it says "the one command the worker always advertises" and is now wrong):

```rust
    // `CPU_ADAPTER_ID` and the two commands the worker always advertises.
    assert!(text.contains("cpu-0"), "{text}");
    assert!(text.contains("validate_project"), "{text}");
    assert!(text.contains("inspect_model"), "{text}");
```

- [ ] **Step 2: Run test to verify it fails**

Without the release binaries built, both tests skip. With them built against `main` before Task 1, the handshake assertion fails on the missing `inspect_model` and the new test fails because the CLI does not know the verb. Build and run:

```powershell
$env:FEATHERTALK_REQUIRE_E2E = "1"
$env:FEATHERTALK_WORKER_HUBERT_DIR = "$env:TEMP\ft_hubert_e2e\package"
cargo build --release -p feathertalk-worker -p feathertalk-cli
cargo test --release -p feathertalk-cli --test real_worker -- --nocapture *> "$env:TEMP\ft_inspect_e2e.log"
Select-String -Path "$env:TEMP\ft_inspect_e2e.log" -Pattern "test result:|panicked at|skipping"
```

- [ ] **Step 3: Write minimal implementation**

None: Tasks 1 to 6 are the implementation. If this test fails, the fix belongs to whichever task owns the behaviour, not here.

- [ ] **Step 4: Run test to verify it passes**

The gated run above, plus the whole suite:

```powershell
cargo test --workspace --all-targets
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

The other end-to-end tests still need ffmpeg, ffprobe and both model directories; `a_real_second_is_extracted_end_to_end` skips for missing SCRFD/PFLD, which is expected on this machine.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-cli/tests/real_worker.rs
git commit -m "test(cli): inspect a real model package end to end"
```

---

## Done When

- `feathertalk inspect-model <source>` prints one JSON object for either an exported model package or a training checkpoint, naming the model kind, the architecture version, the files with their manifest and on-disk sizes, and whether this build can use it.
- The handshake lists `inspect_model` for every configuration, including a worker with nothing configured at all.
- A directory that is neither layout, and a directory whose manifest does not parse, both fail as `MODEL_INCOMPATIBLE` with 「请重新导入模型文件」 as the advice; an intact but unusable model succeeds and says why it is unusable.
- No weights are read: inspecting a multi-gigabyte package is as fast as inspecting a micro one.
- `feathertalk-domain` is untouched, and no new dependency or environment variable appears anywhere.
