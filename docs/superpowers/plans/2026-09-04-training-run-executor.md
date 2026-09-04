# Training Run Executor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `feathertalk-training-run` crate that turns a locked project dataset plus a `TrainingConfig` into executed optimizer steps, telemetry, checkpoints, and preview artifacts.

**Architecture:** The existing crates already provide the pieces: `feathertalk-training` owns config validation, the sampling data loader, the loss functions, telemetry types, and checkpoint IO; `feathertalk-training-data` turns samples into stacked burn tensors; `feathertalk-models` owns the UNet graphs. This slice adds the missing wiring layer — a trait that marks a model as trainable, two step functions that map a training mode onto the right loss, and a `TrainingRunner` that pulls batches, tracks progress, projects metrics, and saves/restores state. Nothing in the worker, CLI, or wire protocol changes.

**Tech Stack:** Rust edition 2024 (rust-version 1.94), burn `=0.21.0` with the NdArray autodiff backend in tests, tempfile 3.20 for fixtures.

**Design:** `docs/superpowers/specs/2026-09-04-training-run-executor-design.md`

## Global Constraints

- Run every `cargo`, `rustfmt`, and `clippy` command from `E:\workspace\github\FeatherTalk\rust`. Run every `git` command from `E:\workspace\github\FeatherTalk`.
- Exactly three additive changes touch existing crates: the new trait file in `feathertalk-models` plus its two lines in `unet/mod.rs`, the `sample_count` visibility bump, and the new `TrainingDataLoader::dataset()` accessor. `adam_train_step`, `feathertalk-training-data`, the worker, the CLI, and the wire protocol stay unchanged.
- The new crate's dependencies are exactly `burn.workspace`, `feathertalk-models` (`default-features = false`), `feathertalk-training`, `feathertalk-training-data`; its dev-dependencies are exactly `feathertalk-audio`, `feathertalk-inference`, `feathertalk-preprocess`, `feathertalk-project`, `tempfile.workspace`. No `thiserror` — every error is a `feathertalk_training::TrainingError`.
- Fixed tensor contract, never parameterized: 6 input channels, 3 output channels, 1 mask channel, inner size 160, audio `16*32*32`, `FEATURE_DIMS` 1024, 2 tokens per frame, `PREVIEW_TENSOR_ELEMENTS` 76800, `PREVIEW_TENSOR_SHAPE` `[3, 160, 160]`.
- Mode mapping, copied verbatim from the design: `Baseline` → `SingleFrame` sampling, stride 0, `SingleFrameBatch`, `baseline_loss`, all three optional losses `None`. `MouthRoi` → `SingleFrame`, stride 0, `SingleFrameBatch`, `mouth_roi_loss`, `mouth` is `Some`. `MouthRoiTemporal` → `TemporalPair`, stride `config.temporal_stride`, `TemporalBatch`, `temporal_loss`, all three optional losses `Some`.
- No `unwrap`, `expect`, `panic!`, panicking index, or panicking arithmetic outside `#[cfg(test)]` and `tests/`. Counters use `checked_add` and surface `TrainingError::DataLoaderOverflow`; progress math uses `saturating_mul`/`saturating_add`/`saturating_sub`.
- Never `.clone()` a `Copy` device — clippy's `clone_on_copy` is an error under `-D warnings`. Prefer `ok_or_else` over `ok_or`.
- The repo has no `rustfmt.toml`, so rustfmt defaults apply: `max_width` 100, `fn_call_width` 60, `chain_width` 60, `struct_lit_width` 18, `imports_layout` Mixed, `reorder_imports` true. Every code block below is already in rustfmt's output shape — type it as written.
- clippy runs with `-D warnings` on 1.94. `#[allow(clippy::too_many_arguments)]` is the established repo idiom and is required on `build_preview_artifact` (9 arguments). `TrainingRunner::new` (6) and `TrainingRunner::metrics` (5) do not need it.
- Every new test runs offline on `Autodiff<NdArray<f32>>` with `OriginalUnetConfig::parity_micro()`, stub feature extractors, and a synthetic locked-project fixture. No test may read an environment variable, invoke ffmpeg, touch a GPU, or load VGG19.
- Stage explicit paths only. Never stage anything under `demo/`. One commit per task with the exact message given. Do not push.

## File Structure

```
rust/Cargo.toml                                                     workspace members gain feathertalk-training-run
rust/Cargo.lock                                                     lock entry for the new crate
rust/crates/feathertalk-models/src/unet/training_graph.rs           TrainableTalkingHead and its two impls
rust/crates/feathertalk-models/src/unet/mod.rs                      + mod training_graph; + pub use
rust/crates/feathertalk-models/tests/training_graph.rs              trait bound and forward-shape coverage
rust/crates/feathertalk-training/src/data.rs                        sample_count becomes pub; new dataset() accessor
rust/crates/feathertalk-training/tests/data_loader.rs               + coverage for both new accessors
rust/crates/feathertalk-training-run/Cargo.toml                     new crate manifest
rust/crates/feathertalk-training-run/src/lib.rs                     module wiring and the public surface
rust/crates/feathertalk-training-run/src/step.rs                    mode -> loader config, and the two step functions
rust/crates/feathertalk-training-run/src/loss.rs                    LossValues: scalars out of LossBreakdown
rust/crates/feathertalk-training-run/src/runner.rs                  TrainingRunner, StepReport, metrics, checkpoints
rust/crates/feathertalk-training-run/src/preview.rs                 build_preview_artifact
rust/crates/feathertalk-training-run/tests/support/mod.rs           backend aliases, stub extractors, config builder
rust/crates/feathertalk-training-run/tests/fixture/mod.rs           locked-project fixture and gradient frame reader
rust/crates/feathertalk-training-run/tests/mode_mapping.rs          mode -> loader config, incl. rejections
rust/crates/feathertalk-training-run/tests/single_frame_step.rs     one single-frame step and its loss shape
rust/crates/feathertalk-training-run/tests/temporal_step.rs         one temporal step and the sample-major reshape
rust/crates/feathertalk-training-run/tests/runner_progress.rs       stepping across epoch boundaries
rust/crates/feathertalk-training-run/tests/metrics.rs               TrainingMetrics projection
rust/crates/feathertalk-training-run/tests/checkpoint_round_trip.rs save, restore, resume determinism
rust/crates/feathertalk-training-run/tests/non_finite_loss.rs       NaN loss poisoning
rust/crates/feathertalk-training-run/tests/preview.rs               preview artifact contents and round-trip
```

---

### Task 1: The trainable talking-head boundary

**Files:**

- Create: `rust/crates/feathertalk-models/src/unet/training_graph.rs`
- Modify: `rust/crates/feathertalk-models/src/unet/mod.rs`
- Test: `rust/crates/feathertalk-models/tests/training_graph.rs`

**Interfaces:**

- Consumes: `OriginalUnet<B>::forward(&self, image: Tensor<B, 4>, audio: Tensor<B, 4>) -> Tensor<B, 4>` and the identical `MobileOneUnet<B>::forward`, both already public.
- Produces: `pub trait TrainableTalkingHead<B: Backend> { fn forward_training(&self, image: Tensor<B, 4>, audio: Tensor<B, 4>) -> Tensor<B, 4>; }`, exported as `feathertalk_models::unet::TrainableTalkingHead`. Tasks 4, 5, 6, and 9 bound their models on it.

**Why now:** Every later task needs one name for "a model you may train". Without it the step functions would have to be written twice, once per UNet flavour. The trait is deliberately *not* implemented for `MobileOneUnetInference`, which is what keeps a reparameterized inference graph out of the training path — and that exclusion is what the `compile_fail` doctest pins down.

- [ ] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-models/tests/training_graph.rs`:

```rust
use burn::tensor::Tensor;
use feathertalk_models::{
    backend::CpuBackend,
    unet::{
        MobileOneUnet, MobileOneUnetConfig, OriginalUnet, OriginalUnetConfig,
        TrainableTalkingHead,
    },
};

type CpuDevice = burn::backend::ndarray::NdArrayDevice;

fn assert_trainable_talking_head<M: TrainableTalkingHead<CpuBackend>>() {}

fn image(device: &CpuDevice) -> Tensor<CpuBackend, 4> {
    Tensor::zeros([1, 6, 160, 160], device)
}

fn audio(device: &CpuDevice) -> Tensor<CpuBackend, 4> {
    Tensor::zeros([1, 16, 32, 32], device)
}

#[test]
fn both_training_graphs_implement_the_public_training_trait() {
    assert_trainable_talking_head::<OriginalUnet<CpuBackend>>();
    assert_trainable_talking_head::<MobileOneUnet<CpuBackend>>();
}

#[test]
fn training_trait_forward_preserves_the_fixed_unet_contract() {
    let device = CpuDevice::default();

    let original = OriginalUnetConfig::parity_micro().init::<CpuBackend>(&device);
    let original_output = original.forward_training(image(&device), audio(&device));
    assert_eq!(original_output.dims(), [1, 3, 160, 160]);

    let mobile = MobileOneUnetConfig::parity_micro().init::<CpuBackend>(&device);
    let mobile_output = mobile.forward_training(image(&device), audio(&device));
    assert_eq!(mobile_output.dims(), [1, 3, 160, 160]);
}
```

`feathertalk_models::unet` is the only import path for these types — the crate root does not re-export them.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-models --test training_graph`
Expected: FAIL to compile with `error[E0432]: unresolved import` / `no TrainableTalkingHead in unet`.

- [ ] **Step 3: Write minimal implementation**

Create `rust/crates/feathertalk-models/src/unet/training_graph.rs`:

```rust
use burn::tensor::{Tensor, backend::Backend};

use super::{MobileOneUnet, OriginalUnet};

/// Common training boundary for talking-head UNet graphs.
///
/// The reparameterized MobileOne inference graph must not cross this boundary:
///
/// ```compile_fail
/// use feathertalk_models::{
///     backend::CpuBackend,
///     unet::{MobileOneUnetConfig, TrainableTalkingHead},
/// };
///
/// fn require_training_graph<M: TrainableTalkingHead<CpuBackend>>(_model: &M) {}
///
/// let device = Default::default();
/// let inference_graph = MobileOneUnetConfig::parity_micro()
///     .init::<CpuBackend>(&device)
///     .reparameterize();
/// require_training_graph(&inference_graph);
/// ```
pub trait TrainableTalkingHead<B: Backend> {
    fn forward_training(&self, image: Tensor<B, 4>, audio: Tensor<B, 4>) -> Tensor<B, 4>;
}

impl<B: Backend> TrainableTalkingHead<B> for OriginalUnet<B> {
    fn forward_training(&self, image: Tensor<B, 4>, audio: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward(image, audio)
    }
}

impl<B: Backend> TrainableTalkingHead<B> for MobileOneUnet<B> {
    fn forward_training(&self, image: Tensor<B, 4>, audio: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward(image, audio)
    }
}
```

Then edit `rust/crates/feathertalk-models/src/unet/mod.rs`. The `mod` declarations are one per line and alphabetical, so `mod training_graph;` goes immediately after `mod model;`:

```rust
mod model;
mod training_graph;
```

The `pub use` list is also alphabetical, so the re-export goes last, after `pub use model::OriginalUnet;`:

```rust
pub use model::OriginalUnet;
pub use training_graph::TrainableTalkingHead;
```

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-models --test training_graph
cargo test -p feathertalk-models --doc unet::training_graph
cargo fmt --all -- --check
cargo clippy -p feathertalk-models --all-targets -- -D warnings
```

Expected: 2 integration tests pass, 1 doctest passes (the `compile_fail` block counts as a pass when it fails to compile), fmt is clean, clippy is clean. `--all-targets` skips doctests, which is why the `--doc` run is separate.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-models/src/unet/training_graph.rs rust/crates/feathertalk-models/src/unet/mod.rs rust/crates/feathertalk-models/tests/training_graph.rs
git commit -m "feat(models): add a trainable talking-head boundary"
```

---

### Task 2: Expose the sample count and the loader dataset

**Files:**

- Modify: `rust/crates/feathertalk-training/src/data.rs` (line 67, and after line 263)
- Test: `rust/crates/feathertalk-training/tests/data_loader.rs` (append two tests)

**Interfaces:**

- Consumes: `DataLoaderConfig::single_frame(batch_size, seed)`, `DataLoaderConfig::temporal_pair(batch_size, seed, stride)`, `TrainingDataLoader::new(dataset, config)`, and the existing `pub(crate) fn sample_count`.
- Produces: `DataLoaderConfig::sample_count(&self, frame_count: u64) -> Result<u64, TrainingError>` becomes `pub`, and `TrainingDataLoader::dataset(&self) -> &D` is new. Task 7 needs `sample_count` for the ETA denominator; Tasks 6 and 9 need `dataset()` to reach the project dataset through the runner.

**Why now:** Both accessors are one-line visibility changes on already-tested logic, and every later task in the new crate depends on them. Doing this before the crate exists keeps the two existing-crate touches isolated in their own reviewable commit.

- [ ] **Step 1: Write the failing test**

Append to `rust/crates/feathertalk-training/tests/data_loader.rs`, after the final test `dataset_failure_during_prepare_leaves_state_unchanged`. The stub `PlanDataset { frames: u64 }` and every name used here is already imported at the top of that file:

```rust
#[test]
fn sample_count_is_public_for_each_sampling_kind() {
    assert_eq!(
        DataLoaderConfig::single_frame(4, 7)
            .sample_count(10)
            .unwrap(),
        10
    );
    assert_eq!(
        DataLoaderConfig::temporal_pair(4, 7, 3)
            .sample_count(10)
            .unwrap(),
        7
    );
}

#[test]
fn the_loader_lends_out_its_dataset() {
    let loader = TrainingDataLoader::new(
        PlanDataset { frames: 6 },
        DataLoaderConfig::single_frame(2, 7),
    )
    .unwrap();
    assert_eq!(loader.dataset().frame_count(), 6);
}
```

`10` and `7` are not arbitrary: single-frame sampling yields one sample per frame, temporal-pair sampling yields `frame_count - stride`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-training --test data_loader`
Expected: FAIL to compile — `error[E0624]: method 'sample_count' is private` and `error[E0599]: no method named 'dataset'`.

- [ ] **Step 3: Write minimal implementation**

In `rust/crates/feathertalk-training/src/data.rs`, replace the signature line at line 67:

```rust
    pub(crate) fn sample_count(&self, frame_count: u64) -> Result<u64, TrainingError> {
```

with a documented public one:

```rust
    /// Samples per epoch: `frame_count` for single frames, `frame_count - stride` for pairs.
    pub fn sample_count(&self, frame_count: u64) -> Result<u64, TrainingError> {
```

Then, in the same file, immediately after the existing `state` accessor (lines 261-263), add:

```rust
    /// Lends out the dataset this loader samples from.
    pub fn dataset(&self) -> &D {
        &self.dataset
    }
```

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-training --test data_loader
cargo fmt --all -- --check
cargo clippy -p feathertalk-training --all-targets -- -D warnings
```

Expected: every `data_loader` test passes, fmt clean, clippy clean. Watch for a `dead_code` warning if the `pub(crate)` call site vanished — it did not, `TrainingDataLoader::new` still calls `sample_count`.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-training/src/data.rs rust/crates/feathertalk-training/tests/data_loader.rs
git commit -m "feat(training): expose the sample count and the loader dataset"
```

---

### Task 3: The new crate and the mode-to-loader-config mapping

**Files:**

- Create: `rust/crates/feathertalk-training-run/Cargo.toml`
- Create: `rust/crates/feathertalk-training-run/src/lib.rs`
- Create: `rust/crates/feathertalk-training-run/src/step.rs`
- Create: `rust/crates/feathertalk-training-run/tests/support/mod.rs`
- Create: `rust/crates/feathertalk-training-run/tests/mode_mapping.rs`
- Modify: `rust/Cargo.toml` (workspace `members`)
- Modify: `rust/Cargo.lock` (regenerated by cargo)
- Test: `rust/crates/feathertalk-training-run/tests/mode_mapping.rs`

**Interfaces:**

- Consumes: `TrainingConfig { mode, batch_size, learning_rate, total_epochs, temporal_stride, mouth_weight, temporal_weight, temporal_mouth_weight, perceptual_weight }` and its `validate()`, `TrainingMode::{Baseline, MouthRoi, MouthRoiTemporal}`, `DataLoaderConfig::single_frame`/`temporal_pair`, `TrainingError`, `PerceptualFeatureExtractor<B>`, and Task 1's nothing-yet (the trait is used from Task 4 onward).
- Produces: `pub fn data_loader_config_for(config: &TrainingConfig, seed: u64) -> Result<DataLoaderConfig, TrainingError>`, and the whole `tests/support` module that Tasks 4-9 reuse: `CpuBackend`, `CpuAutodiffBackend`, `CpuDevice`, `IdentityExtractor`, `NanExtractor`, `model(&CpuDevice) -> OriginalUnet<CpuAutodiffBackend>`, `training_config(mode, batch_size, total_epochs, temporal_stride) -> TrainingConfig`, `assert_close(f64, f64)`.

**Why now:** This is the smallest slice that makes the crate exist and compile, and the mode mapping is the one decision every later task branches on. The full `tests/support/mod.rs` lands here rather than being grown task by task, because a test-support module that changes under every task is a merge hazard; `#![allow(dead_code)]` at its top is what lets items sit unused until Task 4 picks them up.

- [ ] **Step 1: Write the failing test**

First create the crate skeleton so there is something to test. `rust/crates/feathertalk-training-run/Cargo.toml` — field style copied from `feathertalk-training-data/Cargo.toml`:

```toml
[package]
name = "feathertalk-training-run"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[dependencies]
burn.workspace = true
feathertalk-models = { path = "../feathertalk-models", default-features = false }
feathertalk-training = { path = "../feathertalk-training" }
feathertalk-training-data = { path = "../feathertalk-training-data" }

[dev-dependencies]
feathertalk-audio = { path = "../feathertalk-audio" }
feathertalk-inference = { path = "../feathertalk-inference" }
feathertalk-preprocess = { path = "../feathertalk-preprocess" }
feathertalk-project = { path = "../feathertalk-project" }
tempfile.workspace = true
```

`default-features = false` on `feathertalk-models` drops its default `wgpu` feature; the new crate only ever needs the NdArray path.

`rust/crates/feathertalk-training-run/src/lib.rs`:

```rust
//! Drives a FeatherTalk training run: batches in, weights and telemetry out.

mod step;

pub use step::data_loader_config_for;
```

`rust/crates/feathertalk-training-run/src/step.rs` — leave the body unimplemented for now so the test genuinely fails:

```rust
use feathertalk_training::{DataLoaderConfig, TrainingConfig, TrainingError};

/// Derives the data-loader config a training mode needs.
pub fn data_loader_config_for(
    _config: &TrainingConfig,
    _seed: u64,
) -> Result<DataLoaderConfig, TrainingError> {
    Err(TrainingError::InvalidConfig("not implemented".to_owned()))
}
```

Register the crate in `rust/Cargo.toml`. The `members` list is alphabetical; insert one line between `"crates/feathertalk-training-data",` (line 24) and `"tools/pfld-artifact",` (line 25):

```toml
    "crates/feathertalk-training-data",
    "crates/feathertalk-training-run",
    "tools/pfld-artifact",
```

Now `rust/crates/feathertalk-training-run/tests/support/mod.rs`, the shared harness for every later task:

```rust
#![allow(dead_code)]

use burn::tensor::{Tensor, backend::Backend};
use feathertalk_models::unet::{OriginalUnet, OriginalUnetConfig};
use feathertalk_training::{PerceptualFeatureExtractor, TrainingConfig, TrainingMode};

pub type CpuBackend = burn::backend::NdArray<f32>;
pub type CpuAutodiffBackend = burn::backend::Autodiff<CpuBackend>;
pub type CpuDevice = burn::backend::ndarray::NdArrayDevice;

#[derive(Debug, Clone, Copy)]
pub struct IdentityExtractor;

impl<B: Backend> PerceptualFeatureExtractor<B> for IdentityExtractor {
    fn forward(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        image
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NanExtractor;

impl<B: Backend> PerceptualFeatureExtractor<B> for NanExtractor {
    fn forward(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        image.mul_scalar(f32::NAN)
    }
}

pub fn model(device: &CpuDevice) -> OriginalUnet<CpuAutodiffBackend> {
    OriginalUnetConfig::parity_micro().init::<CpuAutodiffBackend>(device)
}

pub fn training_config(
    mode: TrainingMode,
    batch_size: u64,
    total_epochs: u64,
    temporal_stride: u64,
) -> TrainingConfig {
    TrainingConfig {
        mode,
        batch_size,
        learning_rate: 1e-4,
        total_epochs,
        temporal_stride,
        mouth_weight: 4.0,
        temporal_weight: 0.5,
        temporal_mouth_weight: 4.0,
        perceptual_weight: 0.01,
    }
}

pub fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-5,
        "expected {expected}, got {actual}"
    );
}
```

`IdentityExtractor` makes `perceptual_mse` collapse to `mean((pred - target)^2)` — a real number, not zero, which is what lets Task 4 assert the weighted sum. `NanExtractor` is the poisoning lever for Tasks 4 and 6.

Finally `rust/crates/feathertalk-training-run/tests/mode_mapping.rs`:

```rust
mod support;

use feathertalk_training::{
    DataLoaderConfig, SamplingKind, TrainingConfig, TrainingError, TrainingMode,
};
use feathertalk_training_run::data_loader_config_for;
use support::training_config;

fn rejection(mode: TrainingMode, temporal_stride: u64) -> String {
    let config = training_config(mode, 4, 1, temporal_stride);
    let error = data_loader_config_for(&config, 7).unwrap_err();
    let TrainingError::InvalidCheckpoint(message) = error else {
        panic!("expected an invalid-checkpoint rejection, got {error:?}");
    };
    message
}

#[test]
fn baseline_maps_to_single_frame_sampling() {
    let config = training_config(TrainingMode::Baseline, 4, 1, 0);
    let loader_config = data_loader_config_for(&config, 7).unwrap();
    assert_eq!(loader_config, DataLoaderConfig::single_frame(4, 7));
    assert_eq!(loader_config.sampling.kind, SamplingKind::SingleFrame);
    assert_eq!(loader_config.sampling.temporal_stride, 0);
}

#[test]
fn mouth_roi_maps_to_single_frame_sampling() {
    let config = training_config(TrainingMode::MouthRoi, 4, 1, 0);
    let loader_config = data_loader_config_for(&config, 42).unwrap();
    assert_eq!(loader_config, DataLoaderConfig::single_frame(4, 42));
}

#[test]
fn temporal_mode_maps_to_temporal_pair_sampling() {
    let config = training_config(TrainingMode::MouthRoiTemporal, 4, 1, 2);
    let loader_config = data_loader_config_for(&config, 42).unwrap();
    assert_eq!(loader_config, DataLoaderConfig::temporal_pair(4, 42, 2));
    assert_eq!(loader_config.sampling.kind, SamplingKind::TemporalPair);
}

#[test]
fn a_non_temporal_mode_rejects_a_temporal_stride() {
    assert_eq!(
        rejection(TrainingMode::Baseline, 3),
        "training_config.temporal_stride must be zero for non-temporal modes"
    );
}

#[test]
fn the_temporal_mode_rejects_a_zero_stride() {
    assert_eq!(
        rejection(TrainingMode::MouthRoiTemporal, 0),
        "training_config.temporal_stride must be greater than zero for temporal mode"
    );
}
```

The two rejection strings are `TrainingConfig::validate`'s own wording, copied verbatim from `feathertalk-training/src/checkpoint.rs` lines 68 and 72. `TrainingConfig` is imported because `training_config` returns one and the `let`-else binding names the error type; if the compiler reports it unused, drop just that one name from the import list.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-training-run --test mode_mapping`
Expected: 5 tests run, all 5 FAIL — the three mapping tests on `called 'Result::unwrap()' on an 'Err' value: InvalidConfig("not implemented")`, the two rejection tests on `expected an invalid-checkpoint rejection, got InvalidConfig("not implemented")`.

- [ ] **Step 3: Write minimal implementation**

Replace the whole body of `rust/crates/feathertalk-training-run/src/step.rs`:

```rust
use feathertalk_training::{DataLoaderConfig, TrainingConfig, TrainingError, TrainingMode};

/// Derives the data-loader config a training mode needs.
pub fn data_loader_config_for(
    config: &TrainingConfig,
    seed: u64,
) -> Result<DataLoaderConfig, TrainingError> {
    config.validate()?;
    Ok(match config.mode {
        TrainingMode::Baseline | TrainingMode::MouthRoi => {
            DataLoaderConfig::single_frame(config.batch_size, seed)
        }
        TrainingMode::MouthRoiTemporal => {
            DataLoaderConfig::temporal_pair(config.batch_size, seed, config.temporal_stride)
        }
    })
}
```

`config.validate()?` is the only validation this function performs — it already enforces the stride/mode agreement, so the mapping below it cannot produce an invalid pairing.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-training-run --test mode_mapping
cargo fmt --all -- --check
cargo clippy -p feathertalk-training-run --all-targets -- -D warnings
```

Expected: 5 passed. The first build here is cold for the new crate and pulls the whole dependency graph; allow a few minutes.

- [ ] **Step 5: Commit**

`Cargo.lock` gains the new crate's entry during Step 2's build, so stage it too.

```bash
git add rust/Cargo.toml rust/Cargo.lock rust/crates/feathertalk-training-run/Cargo.toml rust/crates/feathertalk-training-run/src/lib.rs rust/crates/feathertalk-training-run/src/step.rs rust/crates/feathertalk-training-run/tests/support/mod.rs rust/crates/feathertalk-training-run/tests/mode_mapping.rs
git commit -m "feat(training-run): map a training mode to a loader config"
```

---

### Task 4: One single-frame training step

**Files:**

- Create: `rust/crates/feathertalk-training-run/src/loss.rs`
- Modify: `rust/crates/feathertalk-training-run/src/step.rs`
- Modify: `rust/crates/feathertalk-training-run/src/lib.rs`
- Test: `rust/crates/feathertalk-training-run/tests/single_frame_step.rs`

**Interfaces:**

- Consumes: Task 1's `TrainableTalkingHead<B>`, Task 3's `tests/support` module, `LossBreakdown<B>` (six public `Tensor<B, 1>` fields: `total`, `full`, `perceptual`, `mouth`, `temporal`, `temporal_mouth`, the last three `Option`), `baseline_loss(extractor, prediction, target, &BaselineLossConfig)`, `mouth_roi_loss(extractor, prediction, target, mask, &MouthRoiLossConfig)`, `SingleFrameBatch<B> { image, audio, target, mouth_mask }`, burn's `GradientsParams::from_grads` / `Optimizer::step`.
- Produces: `pub struct LossValues { total: f64, full: f64, perceptual: f64, mouth: Option<f64>, temporal: Option<f64>, temporal_mouth: Option<f64> }` (all fields public, `Debug + Clone + Copy + PartialEq`) with `LossValues::from_breakdown::<B>(&LossBreakdown<B>) -> Self` and `require_finite(&self) -> Result<(), TrainingError>`; `pub fn train_single_frame_step<B, M, O, E>(model: M, optimizer: &mut O, extractor: &E, batch: SingleFrameBatch<B>, config: &TrainingConfig) -> Result<(M, LossValues), TrainingError>`; and the private `commit_gradients` that Task 5 also calls.

**Why now:** The step function is the smallest unit that actually trains. `LossValues` lands with it because a `LossBreakdown` is tensors on the autodiff graph and everything downstream — the finiteness gate, telemetry, checkpoint decisions — needs plain `f64`s detached from that graph. Doing the detach once, here, is what stops later tasks from holding graph nodes alive.

- [ ] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-training-run/tests/single_frame_step.rs`:

```rust
mod support;

use burn::optim::AdamConfig;
use burn::tensor::Tensor;
use feathertalk_training::{TrainingError, TrainingMode};
use feathertalk_training_data::SingleFrameBatch;
use feathertalk_training_run::train_single_frame_step;
use support::{
    CpuAutodiffBackend, CpuDevice, IdentityExtractor, NanExtractor, assert_close, model,
    training_config,
};

fn batch(device: &CpuDevice) -> SingleFrameBatch<CpuAutodiffBackend> {
    SingleFrameBatch {
        image: Tensor::ones([2, 6, 160, 160], device),
        audio: Tensor::ones([2, 16, 32, 32], device),
        target: Tensor::zeros([2, 3, 160, 160], device),
        mouth_mask: Tensor::ones([2, 1, 160, 160], device),
    }
}

#[test]
fn a_baseline_step_reports_only_the_required_losses() {
    let device = CpuDevice::default();
    let mut optimizer = AdamConfig::new().init();
    let (_model, values) = train_single_frame_step(
        model(&device),
        &mut optimizer,
        &IdentityExtractor,
        batch(&device),
        &training_config(TrainingMode::Baseline, 2, 1, 0),
    )
    .unwrap();
    assert_eq!(values.mouth, None);
    assert_eq!(values.temporal, None);
    assert_eq!(values.temporal_mouth, None);
    assert_close(values.total, values.full + 0.01 * values.perceptual);
}

#[test]
fn a_mouth_roi_step_adds_the_weighted_mouth_loss() {
    let device = CpuDevice::default();
    let mut optimizer = AdamConfig::new().init();
    let (_model, values) = train_single_frame_step(
        model(&device),
        &mut optimizer,
        &IdentityExtractor,
        batch(&device),
        &training_config(TrainingMode::MouthRoi, 2, 1, 0),
    )
    .unwrap();
    assert_eq!(values.temporal, None);
    assert_eq!(values.temporal_mouth, None);
    let mouth = values.mouth.unwrap();
    assert_close(mouth, values.full);
    assert_close(
        values.total,
        values.full + 4.0 * mouth + 0.01 * values.perceptual,
    );
}

#[test]
fn the_mouth_roi_total_exceeds_the_baseline_total() {
    let device = CpuDevice::default();
    let start = model(&device);
    let mut baseline_optimizer = AdamConfig::new().init();
    let (_model, baseline) = train_single_frame_step(
        start.clone(),
        &mut baseline_optimizer,
        &IdentityExtractor,
        batch(&device),
        &training_config(TrainingMode::Baseline, 2, 1, 0),
    )
    .unwrap();
    let mut mouth_optimizer = AdamConfig::new().init();
    let (_model, mouth) = train_single_frame_step(
        start,
        &mut mouth_optimizer,
        &IdentityExtractor,
        batch(&device),
        &training_config(TrainingMode::MouthRoi, 2, 1, 0),
    )
    .unwrap();
    assert_close(mouth.full, baseline.full);
    assert!(
        mouth.total > baseline.total,
        "expected {} > {}",
        mouth.total,
        baseline.total
    );
}

#[test]
fn a_zero_learning_rate_leaves_the_weights_untouched() {
    let device = CpuDevice::default();
    let mut config = training_config(TrainingMode::Baseline, 2, 1, 0);
    config.learning_rate = 0.0;
    let mut optimizer = AdamConfig::new().init();
    let start = model(&device);
    let before = start.outc.conv.weight.val().into_data();
    let (trained, _values) = train_single_frame_step(
        start,
        &mut optimizer,
        &IdentityExtractor,
        batch(&device),
        &config,
    )
    .unwrap();
    let after = trained.outc.conv.weight.val().into_data();
    assert_eq!(before, after);
}

#[test]
fn the_temporal_mode_is_rejected() {
    let device = CpuDevice::default();
    let mut optimizer = AdamConfig::new().init();
    let error = train_single_frame_step(
        model(&device),
        &mut optimizer,
        &IdentityExtractor,
        batch(&device),
        &training_config(TrainingMode::MouthRoiTemporal, 2, 1, 1),
    )
    .unwrap_err();
    let TrainingError::InvalidConfig(message) = error else {
        panic!("expected an invalid-config rejection, got {error:?}");
    };
    assert_eq!(message, "the temporal mode needs train_temporal_step");
}

#[test]
fn a_non_finite_loss_is_rejected_and_the_optimizer_survives() {
    let device = CpuDevice::default();
    let mut optimizer = AdamConfig::new().init();
    let start = model(&device);
    let error = train_single_frame_step(
        start.clone(),
        &mut optimizer,
        &NanExtractor,
        batch(&device),
        &training_config(TrainingMode::Baseline, 2, 1, 0),
    )
    .unwrap_err();
    let TrainingError::InvalidInput(message) = error else {
        panic!("expected an invalid-input rejection, got {error:?}");
    };
    assert!(
        message.contains("total") && message.contains("is not finite"),
        "unexpected message: {message}"
    );

    let (_model, values) = train_single_frame_step(
        start,
        &mut optimizer,
        &IdentityExtractor,
        batch(&device),
        &training_config(TrainingMode::Baseline, 2, 1, 0),
    )
    .unwrap();
    assert!(values.total.is_finite());
}
```

Three of these assertions rest on arithmetic that must be understood, not guessed:

- `mouth == full` for an all-ones mask. `mouth_l1_loss` is `(|pred - target| * mask).sum() / (mask.sum().clamp_min(1.0) * channels)`. With a `[N, 1, H, W]` mask of ones, the denominator is `N*H*W*3` and the numerator sums all `N*3*H*W` absolute differences — exactly `full`, which is `(pred - target).abs().mean()`. `validate_mask` requires the single-channel shape, so this is the only mask shape available.
- `total == full + 4*mouth + 0.01*perceptual`. `total` is `full + mouth*mouth_weight + temporal*temporal_weight + temporal_mouth*temporal_mouth_weight + perceptual*perceptual_weight`, and `training_config` fixes the weights at 4.0 / 0.5 / 4.0 / 0.01.
- `mouth.full == baseline.full` in the third test. Both steps start from `start.clone()`, so the forward pass is identical; only the weighting differs. This is what makes `mouth.total > baseline.total` a statement about the loss composition rather than about drifted weights.

The zero-learning-rate test copies the existing idiom in `feathertalk-models/tests/train_step.rs:46-55` — read one real parameter tensor before and after and compare `TensorData` directly. Adam's update is `p - lr * m_hat / (sqrt(v_hat) + eps)`, so `lr = 0.0` is exactly a no-op, and `AdamConfig::new()` has no weight decay to muddy that.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-training-run --test single_frame_step`
Expected: FAIL to compile — `error[E0432]: unresolved import feathertalk_training_run::train_single_frame_step`.

- [ ] **Step 3: Write minimal implementation**

Create `rust/crates/feathertalk-training-run/src/loss.rs`:

```rust
use burn::tensor::{ElementConversion, Tensor, backend::Backend};
use feathertalk_training::{LossBreakdown, TrainingError};

/// Scalar view of a `LossBreakdown`, detached from the autodiff graph.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossValues {
    pub total: f64,
    pub full: f64,
    pub perceptual: f64,
    pub mouth: Option<f64>,
    pub temporal: Option<f64>,
    pub temporal_mouth: Option<f64>,
}

impl LossValues {
    pub fn from_breakdown<B: Backend>(breakdown: &LossBreakdown<B>) -> Self {
        Self {
            total: scalar(&breakdown.total),
            full: scalar(&breakdown.full),
            perceptual: scalar(&breakdown.perceptual),
            mouth: breakdown.mouth.as_ref().map(scalar),
            temporal: breakdown.temporal.as_ref().map(scalar),
            temporal_mouth: breakdown.temporal_mouth.as_ref().map(scalar),
        }
    }

    pub fn require_finite(&self) -> Result<(), TrainingError> {
        check("total", Some(self.total))?;
        check("full", Some(self.full))?;
        check("perceptual", Some(self.perceptual))?;
        check("mouth", self.mouth)?;
        check("temporal", self.temporal)?;
        check("temporal_mouth", self.temporal_mouth)
    }
}

fn scalar<B: Backend>(value: &Tensor<B, 1>) -> f64 {
    value.clone().into_scalar().elem::<f64>()
}

fn check(field: &str, value: Option<f64>) -> Result<(), TrainingError> {
    match value {
        Some(value) if !value.is_finite() => {
            let message = format!("training loss {field} is not finite: {value}");
            Err(TrainingError::InvalidInput(message))
        }
        _ => Ok(()),
    }
}
```

`total` is checked first on purpose: it is the only field that is always poisoned when any component is, so its name is the one the operator sees.

Rewrite `rust/crates/feathertalk-training-run/src/step.rs` — the import block is greedy-packed exactly as rustfmt emits it:

```rust
use burn::{
    module::AutodiffModule,
    optim::{GradientsParams, Optimizer},
    tensor::backend::AutodiffBackend,
};
use feathertalk_models::unet::TrainableTalkingHead;
use feathertalk_training::{
    BaselineLossConfig, DataLoaderConfig, LossBreakdown, MouthRoiLossConfig,
    PerceptualFeatureExtractor, TrainingConfig, TrainingError, TrainingMode, baseline_loss,
    mouth_roi_loss,
};
use feathertalk_training_data::SingleFrameBatch;

use crate::LossValues;

/// Derives the data-loader config a training mode needs.
pub fn data_loader_config_for(
    config: &TrainingConfig,
    seed: u64,
) -> Result<DataLoaderConfig, TrainingError> {
    config.validate()?;
    Ok(match config.mode {
        TrainingMode::Baseline | TrainingMode::MouthRoi => {
            DataLoaderConfig::single_frame(config.batch_size, seed)
        }
        TrainingMode::MouthRoiTemporal => {
            DataLoaderConfig::temporal_pair(config.batch_size, seed, config.temporal_stride)
        }
    })
}

fn commit_gradients<B, M, O>(
    model: M,
    optimizer: &mut O,
    breakdown: LossBreakdown<B>,
    learning_rate: f64,
) -> Result<(M, LossValues), TrainingError>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: Optimizer<M, B>,
{
    let values = LossValues::from_breakdown(&breakdown);
    values.require_finite()?;
    let gradients = GradientsParams::from_grads(breakdown.total.backward(), &model);
    Ok((optimizer.step(learning_rate, model, gradients), values))
}

/// Runs one optimizer step over a single-frame batch.
pub fn train_single_frame_step<B, M, O, E>(
    model: M,
    optimizer: &mut O,
    extractor: &E,
    batch: SingleFrameBatch<B>,
    config: &TrainingConfig,
) -> Result<(M, LossValues), TrainingError>
where
    B: AutodiffBackend,
    M: TrainableTalkingHead<B> + AutodiffModule<B>,
    O: Optimizer<M, B>,
    E: PerceptualFeatureExtractor<B>,
{
    let prediction = model.forward_training(batch.image, batch.audio);
    let breakdown = match config.mode {
        TrainingMode::Baseline => {
            let loss_config = BaselineLossConfig {
                perceptual_weight: config.perceptual_weight,
            };
            baseline_loss(extractor, prediction, batch.target, &loss_config)?
        }
        TrainingMode::MouthRoi => {
            let loss_config = MouthRoiLossConfig {
                mouth_weight: config.mouth_weight,
                perceptual_weight: config.perceptual_weight,
            };
            mouth_roi_loss(
                extractor,
                prediction,
                batch.target,
                batch.mouth_mask,
                &loss_config,
            )?
        }
        TrainingMode::MouthRoiTemporal => {
            return Err(TrainingError::InvalidConfig(
                "the temporal mode needs train_temporal_step".to_owned(),
            ));
        }
    };
    commit_gradients(model, optimizer, breakdown, config.learning_rate)
}
```

`commit_gradients` computes and checks the scalars *before* calling `backward()` and `optimizer.step`, which is what makes a poisoned loss leave the optimizer state clean — the sixth test above is the proof.

Update `rust/crates/feathertalk-training-run/src/lib.rs`:

```rust
//! Drives a FeatherTalk training run: batches in, weights and telemetry out.

mod loss;
mod step;

pub use loss::LossValues;
pub use step::{data_loader_config_for, train_single_frame_step};
```

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-training-run --test single_frame_step
cargo test -p feathertalk-training-run --test mode_mapping
cargo fmt --all -- --check
cargo clippy -p feathertalk-training-run --all-targets -- -D warnings
```

Expected: 6 passed in `single_frame_step`, 5 still passing in `mode_mapping`, fmt clean, clippy clean. Each test runs a real micro-UNet forward and backward at 160x160 on NdArray, so budget one to two seconds per step.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-training-run/src/loss.rs rust/crates/feathertalk-training-run/src/step.rs rust/crates/feathertalk-training-run/src/lib.rs rust/crates/feathertalk-training-run/tests/single_frame_step.rs
git commit -m "feat(training-run): run one single-frame training step"
```

---

### Task 5: One temporal training step

**Files:**

- Modify: `rust/crates/feathertalk-training-run/src/step.rs`
- Modify: `rust/crates/feathertalk-training-run/src/lib.rs`
- Test: `rust/crates/feathertalk-training-run/tests/temporal_step.rs`

**Interfaces:**

- Consumes: Task 4's `commit_gradients` and `LossValues`, `temporal_loss(extractor, prediction, target, mask, &TemporalLossConfig)`, `TemporalBatch<B> { image: Tensor<B, 4>, audio: Tensor<B, 4>, target: Tensor<B, 5>, mouth_mask: Tensor<B, 5> }`.
- Produces: `pub fn train_temporal_step<B, M, O, E>(model: M, optimizer: &mut O, extractor: &E, batch: TemporalBatch<B>, config: &TrainingConfig) -> Result<(M, LossValues), TrainingError>`. Task 6's runner dispatches to it.

**Why now:** The temporal path is the only one where the model's output shape and the loss's input shape disagree — the UNet emits `[pairs*2, 3, 160, 160]` while `temporal_loss` wants `[pairs, 2, 3, 160, 160]`. That reshape is a real piece of logic and deserves its own test cycle rather than being smuggled into Task 4.

**Deviation from the design, deliberate:** design section 7 says a row-count mismatch is caught by `temporal_loss`'s own dimension validation. It is not — burn's `reshape` panics on an element-count mismatch before `temporal_loss` ever sees the tensor, and a panic is not an acceptable failure mode here. So this task adds an explicit row guard that returns `TrainingError::InvalidInput` before the reshape, and the third test below pins its message.

- [ ] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-training-run/tests/temporal_step.rs`:

```rust
mod support;

use burn::optim::AdamConfig;
use burn::tensor::Tensor;
use feathertalk_training::{TrainingError, TrainingMode};
use feathertalk_training_data::TemporalBatch;
use feathertalk_training_run::train_temporal_step;
use support::{
    CpuAutodiffBackend, CpuDevice, IdentityExtractor, assert_close, model, training_config,
};

fn stacked(
    values: &[f32],
    channels: usize,
    size: usize,
    device: &CpuDevice,
) -> Tensor<CpuAutodiffBackend, 4> {
    let rows = values
        .iter()
        .map(|value| Tensor::full([1, channels, size, size], *value, device))
        .collect::<Vec<_>>();
    Tensor::cat(rows, 0)
}

fn pairs(values: &[f32], device: &CpuDevice) -> Tensor<CpuAutodiffBackend, 5> {
    stacked(values, 3, 160, device).reshape([2, 2, 3, 160, 160])
}

fn batch(device: &CpuDevice) -> TemporalBatch<CpuAutodiffBackend> {
    TemporalBatch {
        image: stacked(&[0.25, 0.25, 0.75, 0.75], 6, 160, device),
        audio: stacked(&[0.5, 0.5, 0.1, 0.1], 16, 32, device),
        target: pairs(&[0.0, 0.5, 0.25, 1.0], device),
        mouth_mask: Tensor::ones([2, 2, 1, 160, 160], device),
    }
}

#[test]
fn a_temporal_step_reports_every_loss_component() {
    let device = CpuDevice::default();
    let mut optimizer = AdamConfig::new().init();
    let (_model, values) = train_temporal_step(
        model(&device),
        &mut optimizer,
        &IdentityExtractor,
        batch(&device),
        &training_config(TrainingMode::MouthRoiTemporal, 2, 1, 1),
    )
    .unwrap();
    let mouth = values.mouth.unwrap();
    let temporal = values.temporal.unwrap();
    let temporal_mouth = values.temporal_mouth.unwrap();
    assert_close(mouth, values.full);
    assert_close(temporal, 0.625);
    assert_close(temporal_mouth, 0.625);
    assert_close(
        values.total,
        values.full
            + 4.0 * mouth
            + 0.5 * temporal
            + 4.0 * temporal_mouth
            + 0.01 * values.perceptual,
    );
}

#[test]
fn a_non_temporal_mode_is_rejected() {
    let device = CpuDevice::default();
    let mut optimizer = AdamConfig::new().init();
    let error = train_temporal_step(
        model(&device),
        &mut optimizer,
        &IdentityExtractor,
        batch(&device),
        &training_config(TrainingMode::Baseline, 2, 1, 0),
    )
    .unwrap_err();
    let TrainingError::InvalidConfig(message) = error else {
        panic!("expected an invalid-config rejection, got {error:?}");
    };
    assert_eq!(
        message,
        "the non-temporal modes need train_single_frame_step"
    );
}

#[test]
fn a_row_count_that_does_not_fill_the_pairs_is_rejected() {
    let device = CpuDevice::default();
    let mut optimizer = AdamConfig::new().init();
    let mut wrong = batch(&device);
    wrong.image = stacked(&[0.25, 0.25, 0.75], 6, 160, &device);
    wrong.audio = stacked(&[0.5, 0.5, 0.1], 16, 32, &device);
    let error = train_temporal_step(
        model(&device),
        &mut optimizer,
        &IdentityExtractor,
        wrong,
        &training_config(TrainingMode::MouthRoiTemporal, 2, 1, 1),
    )
    .unwrap_err();
    let TrainingError::InvalidInput(message) = error else {
        panic!("expected an invalid-input rejection, got {error:?}");
    };
    assert_eq!(message, "temporal rows 3 do not match 2x2");
}
```

The two `0.625` values are not magic. Within each pair the two image rows and the two audio rows are identical, so the model produces two identical predictions and `pred[1] - pred[0]` is exactly zero. `temporal` is therefore `mean(|0 - (target[1] - target[0])|)`, and the target deltas are `0.5 - 0.0 = 0.5` for the first pair and `1.0 - 0.25 = 0.75` for the second: `(0.5 + 0.75) / 2 = 0.625`. `temporal_mouth` is the same quantity through `mouth_l1_loss` with an all-ones mask, which — as in Task 4 — collapses to the same mean.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-training-run --test temporal_step`
Expected: FAIL to compile — `error[E0432]: unresolved import feathertalk_training_run::train_temporal_step`.

- [ ] **Step 3: Write minimal implementation**

In `rust/crates/feathertalk-training-run/src/step.rs`, widen the two `feathertalk_training*` imports (the rest of the import block is unchanged):

```rust
use feathertalk_training::{
    BaselineLossConfig, DataLoaderConfig, LossBreakdown, MouthRoiLossConfig,
    PerceptualFeatureExtractor, TemporalLossConfig, TrainingConfig, TrainingError, TrainingMode,
    baseline_loss, mouth_roi_loss, temporal_loss,
};
use feathertalk_training_data::{SingleFrameBatch, TemporalBatch};
```

Then append to the same file:

```rust
/// Runs one optimizer step over a temporal pair batch.
pub fn train_temporal_step<B, M, O, E>(
    model: M,
    optimizer: &mut O,
    extractor: &E,
    batch: TemporalBatch<B>,
    config: &TrainingConfig,
) -> Result<(M, LossValues), TrainingError>
where
    B: AutodiffBackend,
    M: TrainableTalkingHead<B> + AutodiffModule<B>,
    O: Optimizer<M, B>,
    E: PerceptualFeatureExtractor<B>,
{
    if config.mode != TrainingMode::MouthRoiTemporal {
        return Err(TrainingError::InvalidConfig(
            "the non-temporal modes need train_single_frame_step".to_owned(),
        ));
    }
    let [pairs, pair_len, ..] = batch.target.dims();
    let flat = model.forward_training(batch.image, batch.audio);
    let [rows, channels, height, width] = flat.dims();
    if rows != pairs.saturating_mul(pair_len) {
        return Err(TrainingError::InvalidInput(format!(
            "temporal rows {rows} do not match {pairs}x{pair_len}"
        )));
    }
    let prediction = flat.reshape([pairs, pair_len, channels, height, width]);
    let loss_config = TemporalLossConfig {
        mouth_weight: config.mouth_weight,
        temporal_weight: config.temporal_weight,
        temporal_mouth_weight: config.temporal_mouth_weight,
        perceptual_weight: config.perceptual_weight,
    };
    let breakdown = temporal_loss(
        extractor,
        prediction,
        batch.target,
        batch.mouth_mask,
        &loss_config,
    )?;
    commit_gradients(model, optimizer, breakdown, config.learning_rate)
}
```

The pair geometry is read from the *target*, not from the model output, because the target is the shape the loss will validate against. The row guard then makes the model output agree with it before `reshape` can panic.

Update `rust/crates/feathertalk-training-run/src/lib.rs`:

```rust
pub use step::{data_loader_config_for, train_single_frame_step, train_temporal_step};
```

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-training-run --test temporal_step
cargo test -p feathertalk-training-run --test single_frame_step
cargo fmt --all -- --check
cargo clippy -p feathertalk-training-run --all-targets -- -D warnings
```

Expected: 3 passed in `temporal_step`, 6 still passing in `single_frame_step`, fmt clean, clippy clean.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-training-run/src/step.rs rust/crates/feathertalk-training-run/src/lib.rs rust/crates/feathertalk-training-run/tests/temporal_step.rs
git commit -m "feat(training-run): run one temporal training step"
```

---

### Task 6: The runner steps across epoch boundaries

**Files:**

- Create: `rust/crates/feathertalk-training-run/src/runner.rs`
- Create: `rust/crates/feathertalk-training-run/tests/fixture/mod.rs`
- Modify: `rust/crates/feathertalk-training-run/src/lib.rs`
- Test: `rust/crates/feathertalk-training-run/tests/runner_progress.rs`
- Test: `rust/crates/feathertalk-training-run/tests/non_finite_loss.rs`

**Interfaces:**

- Consumes: Tasks 4 and 5's step functions, Task 3's `data_loader_config_for`, Task 2's `TrainingDataLoader::dataset()`, `TrainingDataLoader::{new, state, prepare_next_batch, commit_batch}`, `PreparedBatch::{epoch, items}`, `TrainingDataset<Item = TrainingItem>`, `stack_single_frame_batch::<B>(&[TrainingItem], &B::Device)`, `stack_temporal_batch::<B>(...)`, `ProjectTrainingDataset::open_with_reader`.
- Produces: `pub struct StepReport { epoch: u64, global_step: u64, samples_in_batch: u64, losses: LossValues }` (all fields public, `Debug + Clone + Copy + PartialEq`) and `pub struct TrainingRunner<B, M, O, D>` with `new(dataset, model, optimizer, config, seed, device)`, `step(&mut self, extractor) -> Result<StepReport, TrainingError>`, and the accessors `epoch()`, `global_step()`, `samples_seen()`, `is_finished()`, `training_config()`, `dataset()`, `model()`. Tasks 7, 8, and 9 extend this same `impl` block. Also produces `tests/fixture/mod.rs`, whose `locked_project(frame_count)` and `dataset(&Path)` every remaining test uses.

**Why now:** The two step functions are useless without something that feeds them. This task is where the runner's ownership model gets settled: the model lives in an `Option<M>` so that a failed step can `take()` it and leave the runner poisoned rather than handing back a half-updated graph. The poisoning test ships in the same commit as the mechanism, because a poisoning rule without a test is just a comment.

- [ ] **Step 1: Write the failing test**

First build the fixture. It is a trimmed copy of the training-data test support module, so copy rather than retype:

```powershell
cd E:\workspace\github\FeatherTalk\rust
New-Item -ItemType Directory -Path crates\feathertalk-training-run\tests\fixture -Force
Copy-Item crates\feathertalk-training-data\tests\support\mod.rs crates\feathertalk-training-run\tests\fixture\mod.rs
```

Then edit `rust/crates/feathertalk-training-run/tests/fixture/mod.rs` three times.

First, delete these six items outright — they exercise preprocess and render paths this crate never touches: `downgrade_to_preparing`, `landmarks_for`, `face_crop`, `inner_planes`, `mouth_rect`, `preparing_manifest`. Everything else stays: the four `pub const`s, `GradientFrameReader`, `FixtureSpec` with `gradient` and `manifest`, `locked_project`, `build_locked_project`, `write_features`, `write_landmarks`, and `valid_project`.

Second, replace the whole import block with exactly this — the deletions above orphaned most of it:

```rust
#![allow(dead_code)]

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use feathertalk_audio::{FeatureMatrix, write_feature_file};
use feathertalk_inference::{BgrFrame, FrameReader, InferenceError};
use feathertalk_preprocess::PFLD_LANDMARK_COUNT;
use feathertalk_project::{
    AssetManifest, AssetPackageState, FeatureType, ModelSelection, ProjectManifest,
    TaskHistoryEntry, TaskHistoryStatus, lock_asset_package, write_project_manifest_atomic,
};
use feathertalk_training_data::ProjectTrainingDataset;
use tempfile::TempDir;
```

Third, append the one helper the copy does not have:

```rust
/// Opens the fixture project as a training dataset backed by the gradient reader.
pub fn dataset(project_dir: &Path) -> ProjectTrainingDataset<GradientFrameReader> {
    ProjectTrainingDataset::open_with_reader(project_dir, GradientFrameReader).unwrap()
}
```

`GradientFrameReader` is what keeps this offline: it synthesises a deterministic 256x256 gradient instead of decoding the five-byte placeholder JPEGs on disk, while still rejecting a wrong filename or a missing file.

Now create `rust/crates/feathertalk-training-run/tests/runner_progress.rs`:

```rust
mod fixture;
mod support;

use burn::optim::AdamConfig;
use feathertalk_training::{TrainingDataset, TrainingMode};
use feathertalk_training_run::TrainingRunner;
use fixture::{dataset, locked_project};
use support::{CpuAutodiffBackend, CpuDevice, IdentityExtractor, model, training_config};

#[test]
fn a_full_batch_advances_the_epoch_without_reporting_it() {
    let device = CpuDevice::default();
    let (_temp, project_dir) = locked_project(4);
    let mut runner = TrainingRunner::<CpuAutodiffBackend, _, _, _>::new(
        dataset(&project_dir),
        model(&device),
        AdamConfig::new().init(),
        training_config(TrainingMode::Baseline, 4, 2, 0),
        7,
        device,
    )
    .unwrap();

    let report = runner.step(&IdentityExtractor).unwrap();
    assert_eq!(report.epoch, 0);
    assert_eq!(report.global_step, 1);
    assert_eq!(report.samples_in_batch, 4);
    assert!(report.losses.total.is_finite());
    assert_eq!(runner.epoch(), 1);
    assert_eq!(runner.global_step(), 1);
    assert_eq!(runner.samples_seen(), 4);
    assert!(!runner.is_finished());
}

#[test]
fn a_short_final_batch_still_counts_every_sample() {
    let device = CpuDevice::default();
    let (_temp, project_dir) = locked_project(5);
    let mut runner = TrainingRunner::<CpuAutodiffBackend, _, _, _>::new(
        dataset(&project_dir),
        model(&device),
        AdamConfig::new().init(),
        training_config(TrainingMode::Baseline, 2, 2, 0),
        7,
        device,
    )
    .unwrap();

    let sizes = [
        runner.step(&IdentityExtractor).unwrap().samples_in_batch,
        runner.step(&IdentityExtractor).unwrap().samples_in_batch,
        runner.step(&IdentityExtractor).unwrap().samples_in_batch,
    ];
    assert_eq!(sizes, [2, 2, 1]);
    assert_eq!(runner.samples_seen(), 5);
    assert_eq!(runner.global_step(), 3);
    assert_eq!(runner.epoch(), 1);
    assert_eq!(runner.dataset().frame_count(), 5);
    assert_eq!(runner.training_config().batch_size, 2);
    assert!(runner.model().is_ok());
}

#[test]
fn the_runner_finishes_after_its_last_epoch() {
    let device = CpuDevice::default();
    let (_temp, project_dir) = locked_project(4);
    let mut runner = TrainingRunner::<CpuAutodiffBackend, _, _, _>::new(
        dataset(&project_dir),
        model(&device),
        AdamConfig::new().init(),
        training_config(TrainingMode::Baseline, 4, 2, 0),
        7,
        device,
    )
    .unwrap();

    runner.step(&IdentityExtractor).unwrap();
    let second = runner.step(&IdentityExtractor).unwrap();
    assert_eq!(second.epoch, 1);
    assert_eq!(runner.epoch(), 2);
    assert!(runner.is_finished());
}

#[test]
fn a_temporal_run_steps_through_its_pairs() {
    let device = CpuDevice::default();
    let (_temp, project_dir) = locked_project(5);
    let mut runner = TrainingRunner::<CpuAutodiffBackend, _, _, _>::new(
        dataset(&project_dir),
        model(&device),
        AdamConfig::new().init(),
        training_config(TrainingMode::MouthRoiTemporal, 2, 1, 1),
        7,
        device,
    )
    .unwrap();

    let first = runner.step(&IdentityExtractor).unwrap();
    assert!(first.losses.temporal.is_some());
    assert!(first.losses.temporal_mouth.is_some());
    runner.step(&IdentityExtractor).unwrap();
    assert_eq!(runner.samples_seen(), 4);
    assert_eq!(runner.epoch(), 1);
    assert!(runner.is_finished());
}
```

The turbofish on `TrainingRunner::<CpuAutodiffBackend, _, _, _>::new` is required: `B` appears only in `device: B::Device`, and `NdArrayDevice` is the device of both `NdArray<f32>` and `Autodiff<NdArray<f32>>`, so inference cannot pick one. `device` is `Copy`, so it is passed by value and still usable afterwards — never `.clone()` it, clippy's `clone_on_copy` is an error here.

The sample arithmetic behind the numbers: single-frame sampling yields one sample per frame, so 5 frames at batch 2 gives 2 + 2 + 1. Temporal-pair sampling yields `frame_count - stride` samples, so 5 frames at stride 1 gives 4, which is two full batches of 2. And `report.epoch` is the epoch the batch *came from* while `runner.epoch()` is where the loader now sits — that off-by-one-looking pair is the contract, and the first test pins it.

Also create `rust/crates/feathertalk-training-run/tests/non_finite_loss.rs`:

```rust
mod fixture;
mod support;

use burn::optim::AdamConfig;
use feathertalk_training::{TrainingError, TrainingMode};
use feathertalk_training_run::TrainingRunner;
use fixture::{dataset, locked_project};
use support::{
    CpuAutodiffBackend, CpuDevice, IdentityExtractor, NanExtractor, model, training_config,
};

fn message(error: TrainingError) -> String {
    let TrainingError::InvalidInput(message) = error else {
        panic!("expected an invalid-input rejection, got {error:?}");
    };
    message
}

#[test]
fn a_non_finite_loss_poisons_the_runner() {
    let device = CpuDevice::default();
    let (_temp, project_dir) = locked_project(4);
    let mut runner = TrainingRunner::<CpuAutodiffBackend, _, _, _>::new(
        dataset(&project_dir),
        model(&device),
        AdamConfig::new().init(),
        training_config(TrainingMode::Baseline, 2, 2, 0),
        7,
        device,
    )
    .unwrap();

    let first = message(runner.step(&NanExtractor).unwrap_err());
    assert!(
        first.contains("is not finite"),
        "unexpected message: {first}"
    );

    let second = message(runner.step(&IdentityExtractor).unwrap_err());
    assert_eq!(second, "training runner was poisoned by a failed step");
    let third = message(runner.model().unwrap_err());
    assert_eq!(third, "training runner was poisoned by a failed step");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-training-run --test runner_progress`
Expected: FAIL to compile — `error[E0432]: unresolved import feathertalk_training_run::TrainingRunner`.

- [ ] **Step 3: Write minimal implementation**

Create `rust/crates/feathertalk-training-run/src/runner.rs`:

```rust
use burn::{module::AutodiffModule, optim::Optimizer, tensor::backend::AutodiffBackend};
use feathertalk_models::unet::TrainableTalkingHead;
use feathertalk_training::{
    PerceptualFeatureExtractor, TrainingConfig, TrainingDataLoader, TrainingDataset, TrainingError,
    TrainingMode,
};
use feathertalk_training_data::{TrainingItem, stack_single_frame_batch, stack_temporal_batch};

use crate::{LossValues, data_loader_config_for, train_single_frame_step, train_temporal_step};

const POISONED: &str = "training runner was poisoned by a failed step";

fn poisoned() -> TrainingError {
    TrainingError::InvalidInput(POISONED.to_owned())
}

fn overflow(operation: &'static str) -> TrainingError {
    TrainingError::DataLoaderOverflow { operation }
}

/// What one committed optimizer step did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepReport {
    pub epoch: u64,
    pub global_step: u64,
    pub samples_in_batch: u64,
    pub losses: LossValues,
}

/// Owns a training run: the loader, the model, the optimizer, and the progress counters.
pub struct TrainingRunner<B, M, O, D>
where
    B: AutodiffBackend,
    D: TrainingDataset<Item = TrainingItem>,
{
    model: Option<M>,
    optimizer: O,
    loader: TrainingDataLoader<D>,
    config: TrainingConfig,
    device: B::Device,
    global_step: u64,
    samples_seen: u64,
}

impl<B, M, O, D> TrainingRunner<B, M, O, D>
where
    B: AutodiffBackend,
    M: TrainableTalkingHead<B> + AutodiffModule<B> + Clone,
    O: Optimizer<M, B> + Clone,
    D: TrainingDataset<Item = TrainingItem>,
{
    pub fn new(
        dataset: D,
        model: M,
        optimizer: O,
        config: TrainingConfig,
        seed: u64,
        device: B::Device,
    ) -> Result<Self, TrainingError> {
        let loader_config = data_loader_config_for(&config, seed)?;
        let loader = TrainingDataLoader::new(dataset, loader_config)?;
        Ok(Self {
            model: Some(model),
            optimizer,
            loader,
            config,
            device,
            global_step: 0,
            samples_seen: 0,
        })
    }

    fn run_step<E>(
        &mut self,
        model: M,
        items: &[TrainingItem],
        extractor: &E,
    ) -> Result<(M, LossValues), TrainingError>
    where
        E: PerceptualFeatureExtractor<B>,
    {
        let config = &self.config;
        match config.mode {
            TrainingMode::Baseline | TrainingMode::MouthRoi => {
                let batch = stack_single_frame_batch::<B>(items, &self.device)?;
                train_single_frame_step(model, &mut self.optimizer, extractor, batch, config)
            }
            TrainingMode::MouthRoiTemporal => {
                let batch = stack_temporal_batch::<B>(items, &self.device)?;
                train_temporal_step(model, &mut self.optimizer, extractor, batch, config)
            }
        }
    }

    /// Prepares one batch, trains on it, and commits the loader position.
    pub fn step<E>(&mut self, extractor: &E) -> Result<StepReport, TrainingError>
    where
        E: PerceptualFeatureExtractor<B>,
    {
        let prepared = self.loader.prepare_next_batch()?;
        let epoch = prepared.epoch();
        let samples_in_batch = u64::try_from(prepared.items().len())
            .map_err(|_| overflow("counting batch items"))?;
        let model = self.model.take().ok_or_else(poisoned)?;
        let (model, losses) = self.run_step(model, prepared.items(), extractor)?;
        self.loader.commit_batch(prepared)?;
        self.model = Some(model);
        self.global_step = self
            .global_step
            .checked_add(1)
            .ok_or_else(|| overflow("counting training steps"))?;
        self.samples_seen = self
            .samples_seen
            .checked_add(samples_in_batch)
            .ok_or_else(|| overflow("counting seen samples"))?;
        Ok(StepReport {
            epoch,
            global_step: self.global_step,
            samples_in_batch,
            losses,
        })
    }

    pub fn epoch(&self) -> u64 {
        self.loader.state().epoch
    }

    pub fn global_step(&self) -> u64 {
        self.global_step
    }

    pub fn samples_seen(&self) -> u64 {
        self.samples_seen
    }

    pub fn is_finished(&self) -> bool {
        self.epoch() >= self.config.total_epochs
    }

    pub fn training_config(&self) -> &TrainingConfig {
        &self.config
    }

    pub fn dataset(&self) -> &D {
        self.loader.dataset()
    }

    pub fn model(&self) -> Result<&M, TrainingError> {
        self.model.as_ref().ok_or_else(poisoned)
    }
}
```

Four details that are load-bearing:

- `self.model.take()` happens *before* `run_step`. If the step returns `Err`, the `?` propagates and the `None` stays — that is the poisoning, and it is why the second `step` call in `non_finite_loss.rs` reports `POISONED` rather than retrying.
- `commit_batch` runs only after a successful step, so a failed step leaves the loader position untouched.
- `let config = &self.config;` next to `&mut self.optimizer` and `&self.device` compiles because these are disjoint field borrows through `self`, not three borrows of `self`.
- `TrainingDataLoader` is not `Debug`, so `TrainingRunner` cannot derive `Debug`. Do not add one.

The `+ Clone` on `M` and `O` is not needed by this task; it is what `save_training_checkpoint` requires in Task 8, and declaring it now keeps this `impl` header stable across the remaining tasks.

Update `rust/crates/feathertalk-training-run/src/lib.rs`:

```rust
//! Drives a FeatherTalk training run: batches in, weights and telemetry out.

mod loss;
mod runner;
mod step;

pub use loss::LossValues;
pub use runner::{StepReport, TrainingRunner};
pub use step::{data_loader_config_for, train_single_frame_step, train_temporal_step};
```

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-training-run --test runner_progress
cargo test -p feathertalk-training-run --test non_finite_loss
cargo fmt --all -- --check
cargo clippy -p feathertalk-training-run --all-targets -- -D warnings
```

Expected: 4 passed in `runner_progress`, 1 passed in `non_finite_loss`, fmt clean, clippy clean. These tests build real fixture projects on disk and run several micro-UNet steps, so they are the slowest in the crate — tens of seconds is normal.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-training-run/src/runner.rs rust/crates/feathertalk-training-run/src/lib.rs rust/crates/feathertalk-training-run/tests/fixture/mod.rs rust/crates/feathertalk-training-run/tests/runner_progress.rs rust/crates/feathertalk-training-run/tests/non_finite_loss.rs
git commit -m "feat(training-run): step the runner across epoch boundaries"
```

---

### Task 7: Metrics for the wire

**Files:**

- Modify: `rust/crates/feathertalk-training-run/src/runner.rs`
- Test: `rust/crates/feathertalk-training-run/tests/metrics.rs`

**Interfaces:**

- Consumes: Task 6's `StepReport` and the runner's own `global_step`/`samples_seen` counters, Task 2's `DataLoaderConfig::sample_count`, `TrainingDataLoader::state()`, and `TrainingMetrics::new` (14 arguments, `telemetry.rs:37`).
- Produces: `TrainingRunner::metrics(&self, report: &StepReport, elapsed: Duration, gpu_memory_bytes: Option<u64>, worker_state: &str) -> Result<TrainingMetrics, TrainingError>`. Nothing else in this slice consumes it; the worker will send the value as-is.

**Why now:** The runner already holds every number the wire type wants, and `TrainingMetrics` is what the worker actually publishes. Rate and ETA are the only derived quantities in the whole slice, so they get pinned here — including both division-by-zero paths, which is why the first test passes `Duration::ZERO` instead of pretending time always moves.

- [ ] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-training-run/tests/metrics.rs`:

```rust
mod fixture;
mod support;

use std::time::Duration;

use burn::optim::AdamConfig;
use feathertalk_training::TrainingMode;
use feathertalk_training_run::TrainingRunner;
use fixture::{dataset, locked_project};
use support::{
    CpuAutodiffBackend, CpuDevice, IdentityExtractor, assert_close, model, training_config,
};

#[test]
fn zero_elapsed_time_reports_no_rate_and_no_eta() {
    let device = CpuDevice::default();
    let (_temp, project_dir) = locked_project(4);
    let mut runner = TrainingRunner::<CpuAutodiffBackend, _, _, _>::new(
        dataset(&project_dir),
        model(&device),
        AdamConfig::new().init(),
        training_config(TrainingMode::Baseline, 2, 1, 0),
        7,
        device,
    )
    .unwrap();

    let report = runner.step(&IdentityExtractor).unwrap();
    let metrics = runner
        .metrics(&report, Duration::ZERO, None, "training")
        .unwrap();

    assert_eq!(metrics.mode, TrainingMode::Baseline);
    assert_eq!(metrics.epoch, 0);
    assert_eq!(metrics.global_step, 1);
    assert_eq!(metrics.samples_seen, 2);
    assert_close(metrics.samples_per_second, 0.0);
    assert_close(metrics.estimated_remaining_seconds, 0.0);
    assert_eq!(metrics.gpu_memory_bytes, None);
    assert_eq!(metrics.worker_state, "training");
    assert!(metrics.mouth_loss.is_none());
}

#[test]
fn the_metrics_copy_every_loss_component() {
    let device = CpuDevice::default();
    let (_temp, project_dir) = locked_project(4);
    let mut runner = TrainingRunner::<CpuAutodiffBackend, _, _, _>::new(
        dataset(&project_dir),
        model(&device),
        AdamConfig::new().init(),
        training_config(TrainingMode::MouthRoi, 2, 1, 0),
        7,
        device,
    )
    .unwrap();

    let report = runner.step(&IdentityExtractor).unwrap();
    let metrics = runner
        .metrics(&report, Duration::from_secs(1), Some(4096), "training")
        .unwrap();

    assert_eq!(metrics.total_loss, report.losses.total);
    assert_eq!(metrics.full_loss, report.losses.full);
    assert_eq!(metrics.perceptual_loss, report.losses.perceptual);
    assert_eq!(metrics.mouth_loss, report.losses.mouth);
    assert!(metrics.mouth_loss.is_some());
    assert_eq!(metrics.temporal_loss, None);
    assert_eq!(metrics.temporal_mouth_loss, None);
    assert_eq!(metrics.gpu_memory_bytes, Some(4096));
}

#[test]
fn the_eta_shrinks_as_the_run_progresses() {
    let device = CpuDevice::default();
    let (_temp, project_dir) = locked_project(4);
    let mut runner = TrainingRunner::<CpuAutodiffBackend, _, _, _>::new(
        dataset(&project_dir),
        model(&device),
        AdamConfig::new().init(),
        training_config(TrainingMode::Baseline, 2, 2, 0),
        7,
        device,
    )
    .unwrap();

    let first = runner.step(&IdentityExtractor).unwrap();
    let after_one = runner
        .metrics(&first, Duration::from_secs(1), None, "training")
        .unwrap();
    assert_close(after_one.samples_per_second, 2.0);
    assert_close(after_one.estimated_remaining_seconds, 3.0);

    let second = runner.step(&IdentityExtractor).unwrap();
    let after_two = runner
        .metrics(&second, Duration::from_secs(2), None, "training")
        .unwrap();
    assert_close(after_two.samples_per_second, 2.0);
    assert_close(after_two.estimated_remaining_seconds, 2.0);
    assert_eq!(after_two.epoch, 0);
    assert_eq!(after_two.samples_seen, 4);
}
```

The third test's numbers come straight from the sampling contract. Four frames under single-frame sampling is 4 samples per epoch, and `total_epochs` 2 makes 8 samples of total work. After the first step the loader sits at epoch 0 position 2, so `done` is 2 and `remaining` is 6; 2 samples in 1 second is a rate of 2.0 and an ETA of 3.0. After the second step the loader has rolled to epoch 1 position 0, so `done` is `1 * 4 + 0 = 4`, `remaining` is 4, the rate is still `4 / 2 = 2.0`, and the ETA drops to 2.0. Note that `after_two.epoch` is 0: it is the epoch the batch came from, the same contract Task 6 pinned.

`Duration::ZERO` is a legal input, not an error case — `validate_metric` rejects only non-finite and negative values, so a rate and ETA of 0.0 pass validation.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-training-run --test metrics`
Expected: FAIL to compile — `error[E0599]: no method named metrics found for struct TrainingRunner`.

- [ ] **Step 3: Write minimal implementation**

In `rust/crates/feathertalk-training-run/src/runner.rs`, add a `std` import group above the existing `burn` line:

```rust
use std::time::Duration;

use burn::{module::AutodiffModule, optim::Optimizer, tensor::backend::AutodiffBackend};
```

and grow the `feathertalk_training` import to exactly:

```rust
use feathertalk_training::{
    PerceptualFeatureExtractor, TrainingConfig, TrainingDataLoader, TrainingDataset, TrainingError,
    TrainingMetrics, TrainingMode,
};
```

Then append one method to the existing `impl` block, after `model()`:

```rust
    /// Turns one `StepReport` plus wall-clock elapsed time into wire metrics.
    pub fn metrics(
        &self,
        report: &StepReport,
        elapsed: Duration,
        gpu_memory_bytes: Option<u64>,
        worker_state: &str,
    ) -> Result<TrainingMetrics, TrainingError> {
        let state = self.loader.state();
        let sample_count = state.config.sample_count(state.frame_count)?;
        let total = self.config.total_epochs.saturating_mul(sample_count);
        let done = state
            .epoch
            .saturating_mul(sample_count)
            .saturating_add(state.next_position);
        let remaining = total.saturating_sub(done);
        let seconds = elapsed.as_secs_f64();
        let samples_per_second = if seconds > 0.0 {
            self.samples_seen as f64 / seconds
        } else {
            0.0
        };
        let estimated_remaining_seconds = if samples_per_second > 0.0 {
            remaining as f64 / samples_per_second
        } else {
            0.0
        };
        TrainingMetrics::new(
            self.config.mode,
            report.epoch,
            report.global_step,
            report.losses.total,
            report.losses.full,
            report.losses.perceptual,
            report.losses.mouth,
            report.losses.temporal,
            report.losses.temporal_mouth,
            self.samples_seen,
            samples_per_second,
            estimated_remaining_seconds,
            gpu_memory_bytes,
            worker_state,
        )
    }
```

`state.config.sample_count(...)` is the accessor Task 2 made public; without it the runner would have to duplicate the SingleFrame/TemporalPair branch and the two would drift. Every arithmetic operation on the progress path is `saturating_*` on purpose: a metrics call must never be the thing that fails a training run, and a clamped ETA is strictly more useful than an error. The mode is read from `self.config.mode` rather than carried on `StepReport`, so the mode-to-optional-losses invariant that `TrainingMetrics::validate` enforces (`telemetry.rs:95-128`) is checked against the same config the step functions used.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-training-run --test metrics
cargo fmt --all -- --check
cargo clippy -p feathertalk-training-run --all-targets -- -D warnings
```

Expected: 3 passed, fmt clean, clippy clean.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-training-run/src/runner.rs rust/crates/feathertalk-training-run/tests/metrics.rs
git commit -m "feat(training-run): report training metrics"
```

---

### Task 8: Checkpoint and resume a run

**Files:**

- Modify: `rust/crates/feathertalk-training-run/src/runner.rs`
- Test: `rust/crates/feathertalk-training-run/tests/checkpoint_round_trip.rs`

**Interfaces:**

- Consumes: `save_training_checkpoint` (`checkpoint.rs:342`), `load_training_checkpoint` (`checkpoint.rs:444`), `TrainingCheckpointState` (`checkpoint.rs:81`), `Provenance` (`checkpoint.rs:22`), `RestoredTrainingState` (`checkpoint.rs:334`), `TRAINING_STATE_SCHEMA_VERSION`, and `TrainingDataLoader::restore` (`data.rs:246`).
- Produces: `TrainingRunner::checkpoint_state(&self) -> TrainingCheckpointState`, `TrainingRunner::save_checkpoint(&self, destination: impl AsRef<Path>, descriptor: CheckpointDescriptor) -> Result<TrainingCheckpointManifest, TrainingError>`, and the associated function `TrainingRunner::restore(dataset: D, restored: RestoredTrainingState<M, O>, device: B::Device) -> Result<Self, TrainingError>`.

**Why now:** A checkpoint needs four things that only Task 6's runner holds together: the loader position, the training config, the model, and the optimizer. Putting the three methods on the runner keeps the worker out of the business of assembling a `TrainingCheckpointState` by hand, which is where the cross-field invariants in `TrainingCheckpointState::validate` would get violated.

This is also the first task whose test can prove the optimizer record is doing something. A restored run replays *two* steps: the first is satisfied by weights plus sampling order alone, while the second one only matches if Adam's moments came back from disk.

**Deviation from the design (§12):** `restore` does *not* re-derive the loader config with `data_loader_config_for` and compare it against the saved `state.data_loader.config`. That branch is unreachable. `DataLoaderConfig::sample_count` (`data.rs:85-107`) rejects `SingleFrame` with a non-zero stride and `TemporalPair` with a stride of 0 or `>= frame_count`, `TrainingConfig::validate` (`checkpoint.rs:65-76`) forces a stride of 0 exactly for the non-temporal modes, and `TrainingCheckpointState::validate` (`checkpoint.rs:102-118`) pins the seed, the batch size and the stride to the training config. A state that validates therefore determines `sampling.kind` uniquely, so no test could ever reach the mismatch arm. Code no test can reach is not written.

- [ ] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-training-run/tests/checkpoint_round_trip.rs`:

```rust
mod fixture;
mod support;

use burn::optim::AdamConfig;
use feathertalk_training::{
    CheckpointCompatibility, CheckpointDescriptor, TRAINING_STATE_SCHEMA_VERSION, TrainingMode,
    load_training_checkpoint,
};
use feathertalk_training_run::{TrainingRunner, data_loader_config_for};
use fixture::{dataset, locked_project};
use support::{
    CpuAutodiffBackend, CpuDevice, IdentityExtractor, NanExtractor, assert_close, model,
    training_config,
};

fn descriptor() -> CheckpointDescriptor {
    CheckpointDescriptor::new("original-unet", "original-unet-v1", "0".repeat(64))
}

#[test]
fn a_restored_runner_reproduces_the_next_steps() {
    let device = CpuDevice::default();
    let (_temp, project_dir) = locked_project(4);
    let mut runner = TrainingRunner::<CpuAutodiffBackend, _, _, _>::new(
        dataset(&project_dir),
        model(&device),
        AdamConfig::new().init(),
        training_config(TrainingMode::Baseline, 2, 2, 0),
        7,
        device,
    )
    .unwrap();

    runner.step(&IdentityExtractor).unwrap();
    runner.step(&IdentityExtractor).unwrap();

    let root = tempfile::tempdir().unwrap();
    let checkpoint = root.path().join("checkpoint-000002");
    let manifest = runner.save_checkpoint(&checkpoint, descriptor()).unwrap();
    assert_eq!(manifest.model_kind, "original-unet");

    let third = runner.step(&IdentityExtractor).unwrap();
    let state_after_third = runner.checkpoint_state();
    let fourth = runner.step(&IdentityExtractor).unwrap();

    let (_temp_b, project_b) = locked_project(4);
    let replay = dataset(&project_b);
    let expected = CheckpointCompatibility::new(
        descriptor(),
        training_config(TrainingMode::Baseline, 2, 2, 0),
        4,
    );
    let template_model = model(&device);
    let template_optimizer = AdamConfig::new().init();
    let restored = load_training_checkpoint::<CpuAutodiffBackend, _, _>(
        &checkpoint,
        &template_model,
        &template_optimizer,
        &device,
        &expected,
    )
    .unwrap();
    let mut replayed =
        TrainingRunner::<CpuAutodiffBackend, _, _, _>::restore(replay, restored, device).unwrap();

    let replayed_third = replayed.step(&IdentityExtractor).unwrap();
    assert_eq!(replayed_third.global_step, 3);
    assert_eq!(replayed_third.epoch, third.epoch);
    assert_eq!(replayed_third.samples_in_batch, third.samples_in_batch);
    assert_close(replayed_third.losses.total, third.losses.total);
    assert_close(replayed_third.losses.full, third.losses.full);
    assert_eq!(replayed.checkpoint_state(), state_after_third);

    let replayed_fourth = replayed.step(&IdentityExtractor).unwrap();
    assert_close(replayed_fourth.losses.total, fourth.losses.total);
}

#[test]
fn the_checkpoint_state_matches_the_runner() {
    let device = CpuDevice::default();
    let (_temp, project_dir) = locked_project(4);
    let config = training_config(TrainingMode::Baseline, 2, 2, 0);
    let mut runner = TrainingRunner::<CpuAutodiffBackend, _, _, _>::new(
        dataset(&project_dir),
        model(&device),
        AdamConfig::new().init(),
        config.clone(),
        7,
        device,
    )
    .unwrap();

    runner.step(&IdentityExtractor).unwrap();
    runner.step(&IdentityExtractor).unwrap();

    let state = runner.checkpoint_state();
    let expected_loader = data_loader_config_for(&config, 7).unwrap();

    assert!(state.validate().is_ok());
    assert_eq!(state.schema_version, TRAINING_STATE_SCHEMA_VERSION);
    assert_eq!(state.epoch, 1);
    assert_eq!(state.epoch, runner.epoch());
    assert_eq!(state.global_step, 2);
    assert_eq!(state.random_seed, 7);
    assert_eq!(state.data_loader.config, expected_loader);
    assert_eq!(state.data_loader.next_position, 0);
    assert_eq!(state.training_config, config);
    assert!(state.asset_provenance.entries.is_empty());
    assert!(state.model_provenance.entries.is_empty());
}

#[test]
fn a_mismatched_dataset_refuses_to_restore() {
    let device = CpuDevice::default();
    let (_temp, project_dir) = locked_project(4);
    let mut runner = TrainingRunner::<CpuAutodiffBackend, _, _, _>::new(
        dataset(&project_dir),
        model(&device),
        AdamConfig::new().init(),
        training_config(TrainingMode::Baseline, 2, 2, 0),
        7,
        device,
    )
    .unwrap();

    runner.step(&IdentityExtractor).unwrap();

    let root = tempfile::tempdir().unwrap();
    let checkpoint = root.path().join("checkpoint-000001");
    runner.save_checkpoint(&checkpoint, descriptor()).unwrap();

    let (_temp_b, project_b) = locked_project(6);
    let replay = dataset(&project_b);
    let expected = CheckpointCompatibility::new(
        descriptor(),
        training_config(TrainingMode::Baseline, 2, 2, 0),
        4,
    );
    let template_model = model(&device);
    let template_optimizer = AdamConfig::new().init();
    let restored = load_training_checkpoint::<CpuAutodiffBackend, _, _>(
        &checkpoint,
        &template_model,
        &template_optimizer,
        &device,
        &expected,
    )
    .unwrap();

    let error = TrainingRunner::<CpuAutodiffBackend, _, _, _>::restore(replay, restored, device)
        .map(|_| ())
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "invalid data loader state: dataset frame_count does not match saved state"
    );
}

#[test]
fn a_poisoned_runner_refuses_to_save() {
    let device = CpuDevice::default();
    let (_temp, project_dir) = locked_project(4);
    let mut runner = TrainingRunner::<CpuAutodiffBackend, _, _, _>::new(
        dataset(&project_dir),
        model(&device),
        AdamConfig::new().init(),
        training_config(TrainingMode::Baseline, 2, 2, 0),
        7,
        device,
    )
    .unwrap();

    runner.step(&NanExtractor).unwrap_err();

    let root = tempfile::tempdir().unwrap();
    let checkpoint = root.path().join("checkpoint-000000");
    let error = runner
        .save_checkpoint(&checkpoint, descriptor())
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "invalid training input: training runner was poisoned by a failed step"
    );
    assert!(!checkpoint.exists());
}
```

Five things about this file that are deliberate:

- `assert_eq!(replayed.checkpoint_state(), state_after_third)` is how this file discharges the design's closing assertion (§15: *the next batch's `TrainingSample` list must be identical to the uninterrupted path*). The runner keeps its loader private, so a test cannot read the next batch directly — but it does not need to. `DataLoaderState` carries the seed, the epoch and `next_position`, and `epoch_permutation(sample_count, seed, epoch)` is a pure function of those three, so two runners with equal state necessarily produce the same sample list. Comparing the whole `TrainingCheckpointState` also pins `global_step` and the training config in the same line.
- The replay dataset is a *second* synthetic project, not the first one reopened. `GradientFrameReader` is deterministic, so two `locked_project(4)` fixtures produce byte-identical frames, and using a fresh one proves the runner is restoring from the checkpoint rather than from live loader state.
- `CheckpointCompatibility::new` starts with empty provenances (`checkpoint.rs:267-283`), and `checkpoint_state` writes empty provenances, so `validate_manifest_state` passes without the field overrides that `checkpoint_recovery.rs:157-158` needs.
- `.map(|_| ())` before `unwrap_err()` is not decoration: `TrainingRunner` has no `Debug` impl, so `Result<TrainingRunner<…>, _>::unwrap_err` does not compile without erasing the `Ok` type first.
- Losses are compared with `assert_close`, not `==`. The repo already treats a replayed loss as a 1e-6 quantity (`checkpoint_recovery.rs:216`); Adam's moments make the exact bit pattern depend on the order the records were written in.

`TrainingError` is intentionally not imported. Every error assertion goes through `error.to_string()`, and the `Display` prefixes (`error.rs`) identify the variant unambiguously — `invalid data loader state:` can only be `InvalidDataLoaderState`, `invalid training input:` can only be `InvalidInput`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-training-run --test checkpoint_round_trip`
Expected: FAIL to compile — `error[E0599]: no method named save_checkpoint found for struct TrainingRunner`.

- [ ] **Step 3: Write minimal implementation**

In `rust/crates/feathertalk-training-run/src/runner.rs`, replace Task 7's single `use std::time::Duration;` line with the crate's usual nested form (`checkpoint.rs:1` writes its `std` group the same way):

```rust
use std::{collections::BTreeMap, path::Path, time::Duration};
```

and grow the `feathertalk_training` import to exactly:

```rust
use feathertalk_training::{
    CheckpointDescriptor, PerceptualFeatureExtractor, Provenance, RestoredTrainingState,
    TRAINING_STATE_SCHEMA_VERSION, TrainingCheckpointManifest, TrainingCheckpointState,
    TrainingConfig, TrainingDataLoader, TrainingDataset, TrainingError, TrainingMetrics,
    TrainingMode, save_training_checkpoint,
};
```

Then append three methods to the existing `impl` block, after `metrics()`:

```rust
    /// Snapshots everything a checkpoint needs besides the weights themselves.
    pub fn checkpoint_state(&self) -> TrainingCheckpointState {
        let state = self.loader.state();
        TrainingCheckpointState {
            schema_version: TRAINING_STATE_SCHEMA_VERSION,
            epoch: state.epoch,
            global_step: self.global_step,
            random_seed: state.config.seed,
            data_loader: state.clone(),
            training_config: self.config.clone(),
            asset_provenance: Provenance {
                entries: BTreeMap::new(),
            },
            model_provenance: Provenance {
                entries: BTreeMap::new(),
            },
        }
    }

    /// Writes a complete checkpoint directory: weights, optimizer, and state.
    pub fn save_checkpoint(
        &self,
        destination: impl AsRef<Path>,
        descriptor: CheckpointDescriptor,
    ) -> Result<TrainingCheckpointManifest, TrainingError> {
        let model = self.model()?;
        save_training_checkpoint::<B, M, O>(
            destination,
            model,
            &self.optimizer,
            descriptor,
            self.checkpoint_state(),
        )
    }

    /// Rebuilds a runner from a loaded checkpoint and a freshly opened dataset.
    pub fn restore(
        dataset: D,
        restored: RestoredTrainingState<M, O>,
        device: B::Device,
    ) -> Result<Self, TrainingError> {
        restored.state.validate()?;
        let global_step = restored.state.global_step;
        let loader = TrainingDataLoader::restore(dataset, restored.state.data_loader)?;
        Ok(Self {
            model: Some(restored.model),
            optimizer: restored.optimizer,
            loader,
            config: restored.state.training_config,
            device,
            global_step,
            samples_seen: 0,
        })
    }
```

Why the code looks like this:

- `DataLoaderState` and `TrainingConfig` derive `Clone` but not `Copy` (`data.rs:115`, `checkpoint.rs:34`), so `state.clone()` and `self.config.clone()` are required; `state` is a `&DataLoaderState` handed out by `loader.state()`.
- `Provenance` has no `Default` impl (`checkpoint.rs:20-24`), hence the explicit `BTreeMap::new()`. Both maps are empty because this slice owns no asset or model provenance — the worker fills them when it knows which package produced the run.
- `save_checkpoint` goes through `self.model()?`, which is what makes a poisoned runner refuse to write. It fails before `save_training_checkpoint` touches the filesystem, so no partial directory is left behind.
- The three moves out of `restored` (`restored.model`, `restored.optimizer`, `restored.state.data_loader`, `restored.state.training_config`) are disjoint field paths, and neither struct implements `Drop`, so the partial moves compile. `restored.manifest` is simply dropped: `load_training_checkpoint` has already validated it against `expected`.
- `samples_seen: 0` is deliberate, not an oversight. It is a per-process throughput counter with no checkpoint field, so after a restore the worker must measure `elapsed` from the restore point for `metrics` to report a truthful rate.
- `restored.state.validate()?` is redundant when the value came from `load_training_checkpoint` (which validates at `checkpoint.rs:473`), but `RestoredTrainingState` is a public struct with public fields, so `restore` cannot assume its input took that path. No test claims to cover this line — fabricating a `TrainingCheckpointManifest` by hand to reach it would cost more than the line is worth.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-training-run --test checkpoint_round_trip
cargo fmt --all -- --check
cargo clippy -p feathertalk-training-run --all-targets -- -D warnings
```

Expected: 4 passed, fmt clean, clippy clean. This is the slowest binary in the crate — six micro-UNet steps plus two Burn record round-trips.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-training-run/src/runner.rs rust/crates/feathertalk-training-run/tests/checkpoint_round_trip.rs
git commit -m "feat(training-run): save and restore a training run"
```

---

### Task 9: Build a preview artifact

**Files:**

- Create: `rust/crates/feathertalk-training-run/src/preview.rs`
- Modify: `rust/crates/feathertalk-training-run/src/lib.rs`
- Test: `rust/crates/feathertalk-training-run/tests/preview.rs`

**Interfaces:**

- Consumes: `TrainableTalkingHead::forward_training` (Task 1), `TrainingDataset::load_sample` (`data.rs:171`), `stack_single_frame_batch` (`batch.rs:106`), `PreviewArtifact::new` (`telemetry.rs:150`, 10 arguments).
- Produces: `build_preview_artifact(model, dataset, device, sample, epoch, global_step, model_kind, model_config_sha256, worker_state) -> Result<PreviewArtifact, TrainingError>`. The worker will pair it with `write_preview_artifact`; nothing else in this slice calls it.

**Why now:** Previews are the last thing the worker needs from this crate, and they are the only consumer of a forward pass without a backward pass. Keeping the function free of the runner means the worker can render a preview from any model it holds — including a restored one — without borrowing a mutable runner or advancing the loader position.

The mask arithmetic is the part worth pinning with a test: `mouth_roi` is the prediction multiplied by the `[1, 1, 160, 160]` mask, which relies on Burn broadcasting the single mask channel across the three colour planes. If that broadcast ever changed shape semantics, the preview would silently show a black frame.

- [ ] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-training-run/tests/preview.rs`:

```rust
mod fixture;
mod support;

use feathertalk_training::{
    PREVIEW_TENSOR_ELEMENTS, PREVIEW_TENSOR_SHAPE, TrainingDataset, TrainingSample,
    read_preview_artifact, write_preview_artifact,
};
use feathertalk_training_data::TrainingItem;
use feathertalk_training_run::build_preview_artifact;
use fixture::{dataset, locked_project};
use support::{CpuAutodiffBackend, CpuDevice, model};

const PLANE: usize = 160 * 160;

fn sha256() -> String {
    "a".repeat(64)
}

fn single_frame() -> TrainingSample {
    TrainingSample::SingleFrame {
        target_index: 1,
        reference_index: 0,
    }
}

#[test]
fn the_preview_masks_the_prediction_with_the_mouth_roi() {
    let device = CpuDevice::default();
    let (_temp, project_dir) = locked_project(4);
    let data = dataset(&project_dir);
    let unet = model(&device);
    let sample = single_frame();

    let artifact = build_preview_artifact::<CpuAutodiffBackend, _, _>(
        &unet,
        &data,
        &device,
        &sample,
        3,
        12,
        "original-unet",
        &sha256(),
        "training",
    )
    .unwrap();

    assert_eq!(artifact.sample_index(), 1);
    assert_eq!(artifact.reference_index(), 0);
    assert_eq!(artifact.epoch(), 3);
    assert_eq!(artifact.global_step(), 12);
    assert_eq!(artifact.model_kind(), "original-unet");
    assert_eq!(artifact.model_config_sha256(), sha256());
    assert_eq!(artifact.worker_state(), "training");
    assert_eq!(artifact.shape(), PREVIEW_TENSOR_SHAPE);
    assert_eq!(artifact.prediction().len(), PREVIEW_TENSOR_ELEMENTS);
    assert_eq!(artifact.mouth_roi().len(), PREVIEW_TENSOR_ELEMENTS);

    let TrainingItem::SingleFrame(frame) = data.load_sample(&sample).unwrap() else {
        panic!("a single-frame sample must load a single frame");
    };
    assert_eq!(artifact.target(), frame.target());

    let mask = frame.mouth_mask();
    let mut inside = 0_usize;
    let mut outside = 0_usize;
    for channel in 0..3 {
        for index in 0..PLANE {
            let masked = artifact.mouth_roi()[channel * PLANE + index];
            if mask[index] == 0.0 {
                assert_eq!(masked, 0.0);
                outside += 1;
            } else {
                assert_eq!(masked, artifact.prediction()[channel * PLANE + index]);
                inside += 1;
            }
        }
    }
    assert!(inside > 0, "the mouth mask must cover some pixels");
    assert!(outside > 0, "the mouth mask must exclude some pixels");
}

#[test]
fn a_temporal_sample_has_no_preview() {
    let device = CpuDevice::default();
    let (_temp, project_dir) = locked_project(4);
    let data = dataset(&project_dir);
    let unet = model(&device);
    let sample = TrainingSample::TemporalPair {
        first_target_index: 1,
        second_target_index: 2,
        reference_index: 0,
    };

    let error = build_preview_artifact::<CpuAutodiffBackend, _, _>(
        &unet,
        &data,
        &device,
        &sample,
        0,
        1,
        "original-unet",
        &sha256(),
        "training",
    )
    .unwrap_err();

    assert_eq!(
        error.to_string(),
        "invalid training input: a preview needs a single-frame sample"
    );
}

#[test]
fn the_preview_round_trips_through_disk() {
    let device = CpuDevice::default();
    let (_temp, project_dir) = locked_project(4);
    let data = dataset(&project_dir);
    let unet = model(&device);

    let artifact = build_preview_artifact::<CpuAutodiffBackend, _, _>(
        &unet,
        &data,
        &device,
        &single_frame(),
        0,
        1,
        "original-unet",
        &sha256(),
        "training",
    )
    .unwrap();

    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("preview-000001");
    let manifest = write_preview_artifact(&destination, &artifact).unwrap();
    assert_eq!(manifest.shape, PREVIEW_TENSOR_SHAPE);

    let (loaded, loaded_manifest) =
        read_preview_artifact(&destination, "original-unet", &sha256()).unwrap();
    assert_eq!(loaded, artifact);
    assert_eq!(loaded_manifest, manifest);
}
```

Three notes on the test:

- The mask comparison reloads the sample instead of reaching into the batch, because `stack_single_frame_batch` consumes the item. `FrameSample::mouth_mask` is documented as `[1, 160, 160]` with one inside the ROI and zero outside (`dataset.rs:47-50`), so `mask[index]` indexes the single plane while the artifact indexes three.
- `assert_eq!(artifact.target(), frame.target())` is an exact float comparison on purpose: the target makes a round trip through an f32 `NdArray` tensor and back, with no arithmetic, so any difference at all would mean the wrong planes were stacked.
- `inside > 0` and `outside > 0` are what keep the mask assertion honest. Without them, an all-zero mask or an all-ones mask would pass the loop.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-training-run --test preview`
Expected: FAIL to compile — `error[E0432]: unresolved import feathertalk_training_run::build_preview_artifact`.

- [ ] **Step 3: Write minimal implementation**

Create `rust/crates/feathertalk-training-run/src/preview.rs`:

```rust
use burn::tensor::{Tensor, backend::Backend};
use feathertalk_models::unet::TrainableTalkingHead;
use feathertalk_training::{PreviewArtifact, TrainingDataset, TrainingError, TrainingSample};
use feathertalk_training_data::{TrainingItem, stack_single_frame_batch};

/// Renders one single-frame sample into a wire-ready preview artifact.
#[allow(clippy::too_many_arguments)]
pub fn build_preview_artifact<B, M, D>(
    model: &M,
    dataset: &D,
    device: &B::Device,
    sample: &TrainingSample,
    epoch: u64,
    global_step: u64,
    model_kind: &str,
    model_config_sha256: &str,
    worker_state: &str,
) -> Result<PreviewArtifact, TrainingError>
where
    B: Backend,
    M: TrainableTalkingHead<B>,
    D: TrainingDataset<Item = TrainingItem>,
{
    let TrainingSample::SingleFrame {
        target_index,
        reference_index,
    } = sample
    else {
        return Err(TrainingError::InvalidInput(
            "a preview needs a single-frame sample".to_owned(),
        ));
    };

    let item = dataset.load_sample(sample)?;
    let batch = stack_single_frame_batch::<B>(&[item], device)?;
    let prediction = model.forward_training(batch.image, batch.audio).detach();
    let mouth_roi = prediction.clone() * batch.mouth_mask;

    PreviewArtifact::new(
        *target_index,
        *reference_index,
        epoch,
        global_step,
        model_kind,
        model_config_sha256,
        worker_state,
        preview_values(prediction, "prediction")?,
        preview_values(batch.target, "target")?,
        preview_values(mouth_roi, "mouth_roi")?,
    )
}

fn preview_values<B: Backend>(
    tensor: Tensor<B, 4>,
    context: &str,
) -> Result<Vec<f32>, TrainingError> {
    tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| TrainingError::InvalidInput(format!("preview {context}: {error}")))
}
```

Details that matter:

- `B: Backend`, not `AutodiffBackend`. A preview is a forward pass, so the function works on an inference backend as well; the tests still pass `Autodiff<NdArray<f32>>` because that is the backend the runner holds.
- `.detach()` drops the autodiff graph before the tensor is read back, which is what `perceptual.rs:22` does for the same reason. Without it the preview would keep the whole forward graph alive until `into_data`.
- `prediction.clone() * batch.mouth_mask` broadcasts `[1, 1, 160, 160]` over `[1, 3, 160, 160]`. The clone is required because the multiplication consumes the tensor and the unmasked prediction is still needed.
- The three `batch` field moves (`image`, `audio`, `mouth_mask`, `target`) are disjoint, so no clone of the batch is needed.
- `PreviewArtifact::new` validates the identifier, the 64-hex digest, the worker state charset and all three tensor lengths (`telemetry.rs:222-230`), so this function never has to check `PREVIEW_TENSOR_ELEMENTS` itself.
- `preview_values` maps Burn's `DataError` into `InvalidInput` with `error.to_string()` interpolated, matching `feathertalk-inference/src/burn.rs:125-132`. The signature has to break across lines: on one line it is 102 characters.

Update `rust/crates/feathertalk-training-run/src/lib.rs`:

```rust
//! Drives a FeatherTalk training run: batches in, weights and telemetry out.

mod loss;
mod preview;
mod runner;
mod step;

pub use loss::LossValues;
pub use preview::build_preview_artifact;
pub use runner::{StepReport, TrainingRunner};
pub use step::{data_loader_config_for, train_single_frame_step, train_temporal_step};
```

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-training-run --test preview
cargo fmt --all -- --check
cargo clippy -p feathertalk-training-run --all-targets -- -D warnings
```

Expected: 3 passed, fmt clean, clippy clean.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-training-run/src/preview.rs rust/crates/feathertalk-training-run/src/lib.rs rust/crates/feathertalk-training-run/tests/preview.rs
git commit -m "feat(training-run): build a preview artifact"
```

---

### Task 10: Verify the whole workspace

**Files:** none. This task adds no code and produces no commit.

**Interfaces:** none.

**Why now:** Tasks 1 to 9 each gate only the crate they touch, which is fast but blind to two things: a new workspace member changing resolved feature unification for everyone, and the doctest in Task 1's `compile_fail` block, which `--all-targets` does not run. This task is the only place where the full suite and the real-worker E2E suite run, so it is also the only place that can catch a regression in the 197 test binaries this slice did not touch.

- [ ] **Step 1: Run every workspace gate**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo fmt --all -- --check *> "$env:TEMP\ft_g1.log"; "gate1=$LASTEXITCODE"
cargo clippy --workspace --all-targets -- -D warnings *> "$env:TEMP\ft_g2.log"; "gate2=$LASTEXITCODE"
cargo test --workspace --all-targets *> "$env:TEMP\ft_g3.log"; "gate3=$LASTEXITCODE"
cargo test -p feathertalk-models --doc *> "$env:TEMP\ft_g3b.log"; "gate3b=$LASTEXITCODE"
```

Budget, from this session's measurements: fmt seconds, workspace clippy 95-115 s, the workspace test suite 36-48 minutes plus 2-5 minutes for the new crate, doctests around 45 s. Each command writes to its own log because a cargo run piped through `Select-String` loses the exit code; read the logs afterwards with

```powershell
Select-String -Path "$env:TEMP\ft_g3.log" -Pattern 'test result:|error\[E|panicked'
```

Expected: every gate exits 0, every `test result:` line reports `0 failed`. A `NativeCommandError` line in a redirected log is PowerShell noise from cargo writing to stderr, not a failure — judge by the exit code.

- [ ] **Step 2: Run the real-worker E2E suite**

```powershell
cd E:\workspace\github\FeatherTalk\rust
$env:FEATHERTALK_REQUIRE_E2E = "1"
$env:FEATHERTALK_WORKER_FFMPEG = "D:\environment\ffmpeg\bin\ffmpeg.exe"
$env:FEATHERTALK_WORKER_HUBERT_DIR = "C:\Users\Administrator\AppData\Local\Temp\ft_hubert_e2e\package"
cargo test --release -p feathertalk-cli --test real_worker -- --nocapture *> "$env:TEMP\ft_g4.log"; "gate4=$LASTEXITCODE"
```

Expected: 9 passed in roughly 8 seconds. `FEATHERTALK_REQUIRE_E2E` turns the suite's skip paths into hard failures, so this run also proves the ffmpeg binary and the extracted FeatherHuBERT package are still where the previous slices left them. This slice does not touch the worker, so a failure here means the environment moved, not the code.

- [ ] **Step 3: Confirm the working tree**

```powershell
cd E:\workspace\github\FeatherTalk
git status -sb
git diff --check
git log --oneline -10
```

Expected: `git status -sb` lists only the untracked `demo/kanghui_training_video_featherhubert_188_latest/` directory, `git diff --check` is silent, and `git log --oneline` shows the nine commits from Tasks 1 to 9 on top of `166b8e4`. Nothing is pushed.

---
