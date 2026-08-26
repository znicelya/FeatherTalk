# Training Checkpoint Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strict, atomic Rust training checkpoint coordinator that restores Burn model parameters, Adam state, DataLoader position, epoch, global step, configuration, and provenance across process boundaries.

**Architecture:** Keep checkpoint schema and public coordinator APIs in `feathertalk-training/src/checkpoint.rs`; keep filesystem, hashing, recorder, and staging mechanics in `checkpoint_io.rs`. Burn full-precision binary records are written independently for the model and optimizer so Burn persists `ParamId`; a versioned JSON state and manifest validate all compatibility before either record is loaded. Existing DataLoader and loss APIs remain unchanged.

**Tech Stack:** Rust 1.92, Burn 0.21 `BinFileRecorder<FullPrecisionSettings>`, serde/serde_json, SHA-256, standard filesystem primitives, existing `feathertalk-training` crate, CPU `NdArray` test backend.

## Global Constraints

- Use Rust 1.92 and Burn exactly 0.21.0.
- Training records use `burn::record::BinFileRecorder<burn::record::FullPrecisionSettings>`; do not replace them with SafeTensors.
- Model and optimizer records are separate files and must preserve Burn `ParamId`.
- `TRAINING_CHECKPOINT_MANIFEST_SCHEMA_VERSION = 1`; `TRAINING_STATE_SCHEMA_VERSION = 1`.
- A checkpoint directory contains exactly `manifest.json`, `model.bin`, `optimizer.bin`, and `training-state.json`.
- Reject unknown JSON fields, missing/extra directory entries, symlinks, invalid hashes, incompatible provenance, and existing destination directories.
- Validate all manifest/state compatibility and file hashes before loading Burn tensors.
- Loading returns new model/optimizer values and never mutates the caller's templates on failure.
- Saving writes to a same-parent staging directory, syncs completed files, writes the manifest last, and atomically renames the staging directory.
- Preserve the existing DataLoader schema and deterministic algorithm; embed its `DataLoaderState` without storing a permutation.
- `epoch` must equal `data_loader.epoch`; `random_seed` must equal `data_loader.config.seed`; `global_step` increases only after a committed optimizer update.
- Float configuration values must be finite and non-negative; SHA-256 values are 64 lowercase hexadecimal characters.
- Do not read, modify, stage, commit, or delete `demo/kanghui_training_video_featherhubert_188_latest/`.
- Never use `git add .`; stage only the paths listed by each task.
- Do not use subagents; execute this plan inline with `superpowers:executing-plans`.

---

## File Map

- Create `rust/crates/feathertalk-training/src/checkpoint.rs`: strict schema types, validation, compatibility checks, public save/load functions, and restored-state container.
- Create `rust/crates/feathertalk-training/src/checkpoint_io.rs`: file names, bounded reads, SHA-256, recorder calls, staging directory creation/cleanup, file sync, and atomic rename helpers.
- Modify `rust/crates/feathertalk-training/src/error.rs`: structured checkpoint validation, compatibility, directory, and recorder errors while preserving existing variants.
- Modify `rust/crates/feathertalk-training/src/lib.rs`: module declaration and crate-root exports.
- Create `rust/crates/feathertalk-training/tests/checkpoint_schema.rs`: exact JSON and validation contracts that do not require Burn model tensors.
- Create `rust/crates/feathertalk-training/tests/checkpoint_recovery.rs`: CPU model/Adam cross-instance restore and next-step equivalence.
- Create `rust/crates/feathertalk-training/tests/checkpoint_atomicity.rs`: exact directory, hash, symlink, staging cleanup, and destination preservation behavior.

---

### Task 1: Define checkpoint schema, training configuration, and errors

**Files:**
- Create: `rust/crates/feathertalk-training/src/checkpoint.rs`
- Modify: `rust/crates/feathertalk-training/src/error.rs`
- Modify: `rust/crates/feathertalk-training/src/lib.rs`
- Test: `rust/crates/feathertalk-training/tests/checkpoint_schema.rs`

**Interfaces:**
- Produces `TRAINING_CHECKPOINT_MANIFEST_SCHEMA_VERSION: u32 = 1` and `TRAINING_STATE_SCHEMA_VERSION: u32 = 1`.
- Produces `TrainingMode`, `TrainingConfig`, `Provenance`, `CheckpointFileManifest`, `TrainingCheckpointState`, `CheckpointDescriptor`, `TrainingCheckpointManifest`, `CheckpointCompatibility`, and `RestoredTrainingState<M, O>`.
- Produces `validate()` methods on every persisted schema and `CheckpointCompatibility::validate_manifest_state(...)`.
- Keeps `save_training_checkpoint` and `load_training_checkpoint` declarations private until Tasks 3 and 4 implement them.

- [ ] **Step 1: Write the failing schema tests**

Create `checkpoint_schema.rs` with these concrete contracts:

```rust
use feathertalk_training::{
    DataLoaderConfig, DataLoaderState, RandomAlgorithm, SamplingConfig, SamplingKind,
    CheckpointDescriptor, CheckpointCompatibility, Provenance, TrainingCheckpointManifest,
    TrainingCheckpointState, TrainingConfig, TrainingError, TrainingMode,
    DATA_LOADER_STATE_SCHEMA_VERSION, TRAINING_STATE_SCHEMA_VERSION,
};
use std::collections::BTreeMap;

fn loader_state() -> DataLoaderState {
    DataLoaderState {
        schema_version: DATA_LOADER_STATE_SCHEMA_VERSION,
        random_algorithm: RandomAlgorithm::Splitmix64FisherYatesV1,
        config: DataLoaderConfig {
            batch_size: 2,
            seed: 17,
            sampling: SamplingConfig { kind: SamplingKind::SingleFrame, temporal_stride: 0 },
        },
        frame_count: 5,
        epoch: 3,
        next_position: 4,
    }
}

fn state() -> TrainingCheckpointState {
    TrainingCheckpointState {
        schema_version: TRAINING_STATE_SCHEMA_VERSION,
        epoch: 3,
        global_step: 14,
        random_seed: 17,
        data_loader: loader_state(),
        training_config: TrainingConfig {
            mode: TrainingMode::Baseline,
            batch_size: 2,
            learning_rate: 1e-3,
            total_epochs: 10,
            temporal_stride: 0,
            mouth_weight: 0.0,
            temporal_weight: 0.0,
            temporal_mouth_weight: 0.0,
            perceptual_weight: 0.01,
        },
        asset_provenance: Provenance { entries: BTreeMap::from([("assets".into(), "a".repeat(64))]) },
        model_provenance: Provenance { entries: BTreeMap::from([("vgg19".into(), "b".repeat(64))]) },
    }
}

#[test]
fn state_json_is_schema_one_and_round_trips_exactly() {
    let value = state();
    value.validate().unwrap();
    let json = serde_json::to_string(&value).unwrap();
    assert!(json.contains("\"schema_version\":1"));
    assert!(json.contains("\"global_step\":14"));
    assert!(!json.contains("permutation"));
    assert_eq!(serde_json::from_str::<TrainingCheckpointState>(&json).unwrap(), value);
}

#[test]
fn unknown_fields_and_inconsistent_progress_are_rejected() {
    let json = serde_json::to_value(state()).unwrap();
    let mut extra = json.clone();
    extra["unexpected"] = true.into();
    assert!(serde_json::from_value::<TrainingCheckpointState>(extra).is_err());

    let mut mismatch = state();
    mismatch.epoch = 2;
    assert!(matches!(mismatch.validate(), Err(TrainingError::InvalidCheckpoint(_))));

    let mut bad_hash = state();
    bad_hash.asset_provenance.entries.insert("bad".into(), "ABC".into());
    assert!(matches!(bad_hash.validate(), Err(TrainingError::InvalidCheckpoint(_))));
}

#[test]
fn manifest_descriptor_and_compatibility_use_fixed_identifiers() {
    let descriptor = CheckpointDescriptor::new("original-unet", "original-unet-v1", "c".repeat(64));
    descriptor.validate().unwrap();
    assert_eq!(descriptor.optimizer_kind, "adam");
    assert_eq!(descriptor.optimizer_schema_version, 1);
    let compatibility = CheckpointCompatibility::new(descriptor.clone(), state().training_config.clone(), 5);
    compatibility.validate().unwrap();
}
```

- [ ] **Step 2: Run the focused test and observe the expected RED failure**

Run from `rust/`:

```powershell
cargo test -p feathertalk-training --test checkpoint_schema
```

Expected: compilation fails because checkpoint types and `TrainingError::InvalidCheckpoint` do not exist.

- [ ] **Step 3: Add structured error variants**

Append these variants to `TrainingError` without changing existing display text:

```rust
#[error("invalid training checkpoint: {0}")]
InvalidCheckpoint(String),
#[error("training checkpoint compatibility error: {0}")]
CheckpointCompatibility(String),
#[error("training checkpoint directory error: {0}")]
CheckpointDirectory(String),
```

- [ ] **Step 4: Implement strict schema types and validation**

Implement all persisted structs with `#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]` where applicable and `#[serde(deny_unknown_fields)]`. Use `f64` fields in `TrainingConfig` and implement `PartialEq` manually only if derive constraints require it.

Use these exact serialized enums:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingMode { Baseline, MouthRoi, MouthRoiTemporal }
```

Implement `TrainingConfig::validate()` to require finite, non-negative `learning_rate`, all loss weights, `batch_size > 0`, `total_epochs > 0`, and mode/temporal-stride consistency. Implement `TrainingCheckpointState::validate()` to require schema one, `epoch == data_loader.epoch`, `random_seed == data_loader.config.seed`, matching batch size and temporal stride, a valid nested DataLoader state, and valid provenance. Implement `CheckpointDescriptor::new()` to fill fixed Adam identifiers and `validate()` to enforce 64-hex model config hash. Implement `CheckpointFileManifest::validate(expected_name)` for exact name, positive byte count, and lowercase SHA-256. Implement `TrainingCheckpointManifest::validate()` for exact schema, record format `burn-bin-full-precision-v1`, Adam/schema-one identifiers, exact file names, and matching `training_state_sha256`.

Implement `CheckpointCompatibility::validate()` and `validate_manifest_state()` to compare model kind, architecture, model config hash, optimizer identifiers, training config, frame count, asset provenance, and model provenance before any record load.

- [ ] **Step 5: Export the schema and run GREEN**

Add `mod checkpoint;` and crate-root `pub use` exports in `src/lib.rs`, then run:

```powershell
cargo test -p feathertalk-training --test checkpoint_schema
```

Expected: all schema tests pass.

- [ ] **Step 6: Commit the schema slice**

```powershell
git add rust/crates/feathertalk-training/src/checkpoint.rs rust/crates/feathertalk-training/src/error.rs rust/crates/feathertalk-training/src/lib.rs rust/crates/feathertalk-training/tests/checkpoint_schema.rs
git commit -m "feat: define training checkpoint schema"
```

---

### Task 2: Add Burn record and filesystem primitives

**Files:**
- Create: `rust/crates/feathertalk-training/src/checkpoint_io.rs`
- Modify: `rust/crates/feathertalk-training/src/checkpoint.rs`
- Test: `rust/crates/feathertalk-training/tests/checkpoint_recovery.rs`

**Interfaces:**
- Produces private `MODEL_FILE_NAME`, `OPTIMIZER_FILE_NAME`, `STATE_FILE_NAME`, `MANIFEST_FILE_NAME` constants.
- Produces private `record_model`, `record_optimizer`, `load_model_record`, `load_optimizer_record` helpers using `BinFileRecorder<FullPrecisionSettings>`.
- Produces private `sha256_file`, `write_synced_file`, `validate_checkpoint_directory`, and staging cleanup helpers.

- [ ] **Step 1: Write the failing Burn cross-instance test**

Create a small test-only module in `checkpoint_recovery.rs`:

```rust
use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{Tensor, backend::{AutodiffBackend, Backend}},
};
use feathertalk_models::backend::{CpuAutodiffBackend, CpuBackend};

#[derive(Module, Debug)]
struct TinyModel<B: Backend> { linear: Linear<B> }

impl<B: Backend> TinyModel<B> {
    fn new(device: &B::Device) -> Self { Self { linear: LinearConfig::new(2, 1).init(device) } }
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> { self.linear.forward(input) }
}

fn step<B: AutodiffBackend>(
    model: TinyModel<B>, optimizer: &mut impl Optimizer<TinyModel<B>, B>,
    input: Tensor<B, 2>, target: Tensor<B, 2>,
) -> TinyModel<B> {
    let loss = (model.forward(input) - target).abs().mean();
    let grads = GradientsParams::from_grads(loss.backward(), &model);
    optimizer.step(1e-2, model, grads)
}
```

Add a test that performs one step, calls the not-yet-implemented record helper through the public save API, creates fresh model/optimizer instances, and expects the checkpoint load to compile and return them.

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
cargo test -p feathertalk-training --test checkpoint_recovery
```

Expected: compilation fails because the coordinator and I/O helpers are absent.

- [ ] **Step 3: Implement recorder helpers**

Use the exact Burn types:

```rust
use burn::{
    module::{AutodiffModule, Module},
    optim::Optimizer,
    record::{BinFileRecorder, FileRecorder, FullPrecisionSettings, Recorder},
    tensor::backend::AutodiffBackend,
};

type FullRecorder = BinFileRecorder<FullPrecisionSettings>;

fn write_model<B, M>(model: &M, stem: &Path) -> Result<PathBuf, TrainingError>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Clone,
{
    let recorder = FullRecorder::default();
    recorder.record(model.clone().into_record(), stem.to_path_buf())
        .map_err(|error| TrainingError::Store(error.to_string()))?;
    Ok(stem.with_extension("bin"))
}
```

Implement the analogous optimizer helper with `optimizer.clone().to_record()`. Load records through `recorder.load::<M::Record>(stem.to_path_buf(), device)` and `recorder.load::<O::Record>(...)`, mapping `RecorderError` to `TrainingError::Store`. Do not deserialize records before manifest/hash validation.

- [ ] **Step 4: Implement bounded file hashing and sync helpers**

`sha256_file` must read in 64 KiB chunks and return lowercase `hex::encode(Sha256::finalize())`. `write_synced_file` must use `OpenOptions::create_new(true)` to write bytes, call `sync_all`, and return the byte count and hash. `validate_checkpoint_directory` must reject symlink directories/files and require sorted entries exactly equal to:

```rust
vec![
    "manifest.json".to_owned(),
    "model.bin".to_owned(),
    "optimizer.bin".to_owned(),
    "training-state.json".to_owned(),
]
```

On Windows, `sync_directory` is a no-op after file `sync_all`; on Unix, open the directory and call `sync_all`. Keep all platform branching inside `checkpoint_io.rs`.

- [ ] **Step 5: Run focused compilation and tests**

```powershell
cargo test -p feathertalk-training --test checkpoint_recovery
```

Expected: the test now reaches coordinator behavior rather than missing-record compilation errors.

- [ ] **Step 6: Commit the record primitive slice**

```powershell
git add rust/crates/feathertalk-training/src/checkpoint_io.rs rust/crates/feathertalk-training/src/checkpoint.rs rust/crates/feathertalk-training/tests/checkpoint_recovery.rs
git commit -m "feat: add Burn checkpoint record primitives"
```

---

### Task 3: Implement atomic checkpoint saving

**Files:**
- Modify: `rust/crates/feathertalk-training/src/checkpoint.rs`
- Modify: `rust/crates/feathertalk-training/src/checkpoint_io.rs`
- Test: `rust/crates/feathertalk-training/tests/checkpoint_atomicity.rs`

**Interfaces:**
- Produces `save_training_checkpoint<B, M, O>(destination, model, optimizer, descriptor, state) -> Result<TrainingCheckpointManifest, TrainingError>`.
- The returned manifest contains computed byte lengths and hashes and is the only manifest considered published.

- [ ] **Step 1: Write failing save and atomicity tests**

In `checkpoint_atomicity.rs`, create a valid tiny model/optimizer and valid state, then assert:

```rust
let destination = root.path().join("checkpoint-000001");
let manifest = save_training_checkpoint::<CpuAutodiffBackend, _, _>(
    &destination, &model, &optimizer, descriptor(), state(),
).unwrap();
assert_eq!(manifest.schema_version, 1);
assert_eq!(std::fs::read_dir(&destination).unwrap().count(), 4);
assert_eq!(std::fs::read(destination.join("manifest.json")).unwrap(),
           serde_json::to_vec(&manifest).unwrap());
```

Add tests that a second save to the same destination returns `TrainingError::CheckpointDirectory`, that a deliberately invalid state leaves no `.staging` directory, and that an existing checkpoint's file bytes remain unchanged after a failed save.

- [ ] **Step 2: Run tests to verify RED**

```powershell
cargo test -p feathertalk-training --test checkpoint_atomicity
```

Expected: compilation fails because `save_training_checkpoint` is not implemented.

- [ ] **Step 3: Implement staging lifecycle**

Implement the following order in `save_training_checkpoint`:

```text
validate descriptor and state
create destination parent
reject destination symlink/existing path
create unique sibling staging directory
write model.bin and optimizer.bin with Burn recorder
serialize and sync training-state.json
hash all three data files
construct and validate manifest
serialize and sync manifest.json last
sync staging parent
rename staging -> destination
sync destination parent
return manifest
```

Use a process ID plus an `AtomicU64` counter for a unique staging suffix; do not add a random dependency. Wrap the body in a guard that removes only the newly-created staging directory on error. Never remove or rename an existing destination.

- [ ] **Step 4: Implement manifest construction and state cross-checks**

Construct `CheckpointFileManifest` values from the actual files, set `training_state_sha256` equal to the state file hash, set `record_format` to `burn-bin-full-precision-v1`, and set `burn_version` from `env!("CARGO_PKG_VERSION")` of the Burn core package if exposed; otherwise use the locked literal `0.21.0`. Set `rust_version` to `rustc_version_runtime` only if already available; otherwise use the workspace toolchain literal `1.92.0` and validate it as non-empty.

Before writing any file, require state/config/DataLoader consistency. After writing each record, call `sync_all`; after writing JSON, call `sync_all`; compute hashes from the committed staging bytes, not from in-memory values.

- [ ] **Step 5: Run focused tests and inspect the directory contract**

```powershell
cargo test -p feathertalk-training --test checkpoint_atomicity
Get-ChildItem -Force <temporary-checkpoint-parent>
```

Expected: save tests pass, exactly four files exist, no staging directory remains after success or failure, and an existing destination is untouched.

- [ ] **Step 6: Commit atomic save**

```powershell
git add rust/crates/feathertalk-training/src/checkpoint.rs rust/crates/feathertalk-training/src/checkpoint_io.rs rust/crates/feathertalk-training/tests/checkpoint_atomicity.rs
git commit -m "feat: publish training checkpoints atomically"
```

---

### Task 4: Implement strict checkpoint loading before tensor access

**Files:**
- Modify: `rust/crates/feathertalk-training/src/checkpoint.rs`
- Modify: `rust/crates/feathertalk-training/src/checkpoint_io.rs`
- Test: `rust/crates/feathertalk-training/tests/checkpoint_atomicity.rs`
- Test: `rust/crates/feathertalk-training/tests/checkpoint_recovery.rs`

**Interfaces:**
- Produces `load_training_checkpoint<B, M, O>(directory, model_template, optimizer_template, device, expected) -> Result<RestoredTrainingState<M, O>, TrainingError>`.
- Guarantees no Burn recorder call occurs until all directory, JSON, compatibility, byte-count, and SHA-256 checks pass.

- [ ] **Step 1: Write failing strict-load tests**

After saving a valid checkpoint, mutate one condition at a time and assert the exact phase fails:

```rust
let mut manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
manifest["unexpected"] = true.into();
std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
assert!(matches!(load_training_checkpoint::<CpuAutodiffBackend, _, _>(
    &destination, &template_model, &template_optimizer, &device, &compatibility
), Err(TrainingError::InvalidCheckpoint(_))));
```

Add cases for missing `optimizer.bin`, an extra `notes.txt`, a modified model byte, a symlink in place of `model.bin` (where the platform permits symlink creation), wrong model config hash, wrong asset hash, wrong DataLoader frame count, and wrong optimizer schema. Record a `Cell<bool>`-backed test helper is not needed: the test must prove preflight by replacing record files with invalid bytes while making the manifest/state mismatch; the loader must report compatibility/JSON error rather than a Burn decode error.

- [ ] **Step 2: Run the strict-load tests to verify RED**

```powershell
cargo test -p feathertalk-training --test checkpoint_recovery --test checkpoint_atomicity
```

Expected: tests fail because the loader is not yet implemented or loads records before validation.

- [ ] **Step 3: Implement preflight directory and manifest validation**

Implement this exact load order:

```rust
validate_checkpoint_directory(directory)?;
let manifest = read_strict_manifest(directory)?;
manifest.validate()?;
let state = read_strict_state(directory)?;
state.validate()?;
expected.validate()?;
expected.validate_manifest_state(&manifest, &state)?;
validate_declared_file(directory.join("model.bin"), &manifest.model)?;
validate_declared_file(directory.join("optimizer.bin"), &manifest.optimizer)?;
validate_declared_file(directory.join("training-state.json"), &manifest.training_state)?;
```

Use bounded reads (manifest <= 64 KiB, state <= 256 KiB) and `serde_json::from_slice`; reject trailing/unknown data through the strict JSON parser and exact file set check. Validate `manifest.training_state_sha256` against the declared state file hash before any record load.

- [ ] **Step 4: Implement candidate-only Burn restoration**

After preflight, load `M::Record` with the full recorder and build `candidate_model = model_template.clone().load_record(record)`. Then load `O::Record` and build `candidate_optimizer = optimizer_template.clone().load_record(record)`. Return both only after both loads succeed:

```rust
Ok(RestoredTrainingState { model: candidate_model, optimizer: candidate_optimizer, state, manifest })
```

Map all recorder failures to `TrainingError::Store` with the file and phase in the message. Never assign to `*model_template` or `*optimizer_template` (the arguments are shared references).

- [ ] **Step 5: Run strict-load tests GREEN**

```powershell
cargo test -p feathertalk-training --test checkpoint_recovery --test checkpoint_atomicity
```

Expected: all malformed directory, hash, compatibility, symlink, and preflight-order tests pass.

- [ ] **Step 6: Commit strict loading**

```powershell
git add rust/crates/feathertalk-training/src/checkpoint.rs rust/crates/feathertalk-training/src/checkpoint_io.rs rust/crates/feathertalk-training/tests/checkpoint_recovery.rs rust/crates/feathertalk-training/tests/checkpoint_atomicity.rs
git commit -m "feat: restore checkpoints with strict compatibility checks"
```

---

### Task 5: Prove Adam momentum, progress, and DataLoader continuation equivalence

**Files:**
- Modify: `rust/crates/feathertalk-training/tests/checkpoint_recovery.rs`
- Modify: `rust/crates/feathertalk-training/src/checkpoint.rs` only if a tested invariant is missing

**Interfaces:**
- Verifies the public coordinator against real Burn `NdArray` autodiff and the existing `TrainingDataLoader`.

- [ ] **Step 1: Write the uninterrupted-versus-restored test**

Use `CpuAutodiffBackend`, seed the backend before constructing each model, and run this sequence:

```text
continuous model/Adam: step(input0,target0), step(input1,target1)
interrupted model/Adam: step(input0,target0), save checkpoint(state global_step=1)
fresh model/Adam: load checkpoint, step(input1,target1)
compare every model parameter and the second loss with max_abs_error <= 1e-4
```

Collect parameter values through `model.clone().into_record()` and the full recorder bytes, or visit all float parameters and compare their `TensorData` values. Assert that the fresh model's parameter IDs after load equal the interrupted model's IDs before save by comparing serialized model record bytes. Assert `optimizer` is not empty after the first step by requiring the resumed second step to match; this catches lost Adam momentum.

Embed a DataLoader state with a final partial batch (`frame_count=5`, `batch_size=2`, cursor at `4`) and assert the returned state preserves epoch, cursor, seed, and config exactly.

- [ ] **Step 2: Run the test and observe any RED behavior**

```powershell
cargo test -p feathertalk-training --test checkpoint_recovery -- --nocapture
```

If the comparison fails, inspect model record load order and optimizer record load order; do not loosen the tolerance or reset optimizer state.

- [ ] **Step 3: Make the minimal correction and rerun GREEN**

The only permitted implementation corrections are those needed to preserve `ParamId`, use the same full-precision recorder settings, or validate the progress/DataLoader invariants. Re-run the focused test after each correction and retain the numerical assertions.

- [ ] **Step 4: Commit the equivalence proof**

```powershell
git add rust/crates/feathertalk-training/tests/checkpoint_recovery.rs rust/crates/feathertalk-training/src/checkpoint.rs
git commit -m "test: prove Adam checkpoint recovery equivalence"
```

---

### Task 6: Full verification, review, and integration

**Files:**
- Modify only files required by verification findings; never touch the protected demo directory.

- [ ] **Step 1: Run formatting and focused checks**

From `rust/` run:

```powershell
cargo fmt --all -- --check
cargo test -p feathertalk-training --test checkpoint_schema
cargo test -p feathertalk-training --test checkpoint_recovery
cargo test -p feathertalk-training --test checkpoint_atomicity
cargo clippy -p feathertalk-training --all-targets -- -D warnings
```

- [ ] **Step 2: Run the complete workspace verification**

```powershell
cargo test --workspace --all-targets
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
```

Record exit codes and the number of ignored hardware/license tests. Do not claim completion from a partial command.

- [ ] **Step 3: Perform the inline requirements review**

Check each requirement against code and tests: Burn full-precision records, ParamId restoration, exact four-file directory, manifest-last atomic publish, preflight-before-tensor-load, strict unknown-field rejection, provenance/config checks, DataLoader cursor preservation, final partial batch, no destination overwrite, staging cleanup, and next-step Adam equivalence. Use `git diff --check`, `git status --short`, and `git diff --name-only` to verify no protected path is present.

- [ ] **Step 4: Commit any verification-only fixes explicitly**

```powershell
git add rust/crates/feathertalk-training/src/checkpoint.rs rust/crates/feathertalk-training/src/checkpoint_io.rs rust/crates/feathertalk-training/src/error.rs rust/crates/feathertalk-training/src/lib.rs rust/crates/feathertalk-training/tests/checkpoint_schema.rs rust/crates/feathertalk-training/tests/checkpoint_recovery.rs rust/crates/feathertalk-training/tests/checkpoint_atomicity.rs
git commit -m "chore: verify training checkpoint recovery"
```

- [ ] **Step 5: Integrate the isolated branch**

Use `finishing-a-development-branch` after a fresh full test run. The base branch is `main`; merge locally only after the merged result is independently verified. Remove only the checkpoint worktree and branch; leave `frame-face-pipeline`, `media-normalization-execution`, `pfld-burn-inference`, and the protected untracked demo directory untouched.

- [ ] **Step 6: Continue automatically to the next milestone slice**

After the merged checkpoint result is green, reread the migration design's milestone-three requirements and start the next independent spec/plan cycle for training metrics/preview artifacts or the next unmet milestone-three contract. Do not stop merely because this checkpoint slice is complete.

---

## Plan Self-Review

- **Spec coverage:** schema and provenance are covered by Task 1; Burn record format and ParamId persistence by Task 2 and Task 5; atomic staging and manifest-last publication by Task 3; strict preflight and failure atomicity by Task 4; progress/DataLoader and numerical equivalence by Task 5; full verification and continuation by Task 6.
- **Placeholder scan:** every task names files, commands, concrete assertions, and expected outcomes; no unspecified implementation step is relied upon.
- **Type consistency:** `CheckpointDescriptor`, `TrainingCheckpointState`, `TrainingCheckpointManifest`, `CheckpointCompatibility`, `save_training_checkpoint`, `load_training_checkpoint`, and `RestoredTrainingState` have the same names and argument order in all tasks.
- **Scope check:** this plan contains only checkpoint persistence/recovery; metrics, previews, GPUI, and deployment remain later slices as required by the spec.
- **Protected path check:** the forbidden demo directory is explicitly excluded from every command and commit path.
