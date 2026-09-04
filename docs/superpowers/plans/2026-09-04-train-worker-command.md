# Train Worker Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve `Request::Train` in `feathertalk-worker` and expose it as `feathertalk train`, so a locked project directory turns into checkpoints under `<project>/models/unet/`, metrics and previews under `<project>/outputs/`, and a run that can be resumed or cancelled without losing a complete checkpoint.

**Architecture:** `feathertalk-training-run` already owns the loop mechanics: `TrainingRunner` pulls batches, runs the right loss per mode, counts steps, projects metrics, saves and restores checkpoints, and builds preview artifacts. It has no caller. This slice writes the layer above it. `training.rs` turns `TrainParams` plus worker-side hyperparameter constants into a `TrainingPlan` (config, checkpoint descriptor, paths), publishes checkpoints through a staging rename because `save_training_checkpoint` refuses an existing destination, and writes telemetry that skips a name collision instead of ending a long run. `train.rs` is the command body: admission, VGG19 loading, variant dispatch, the loop, progress, cancellation, error mapping. The wire protocol does not change -- `TaskKind::Train`, `TrainParams`, `Request::Train` and `TaskStage::Training` were committed slices ago.

**Tech Stack:** Rust edition 2024 (rust-version 1.94), burn `=0.21.0` on `Autodiff<NdArray<f32>>` (CPU only, see design section 4), clap 4 for the CLI, `sha2` plus `hex` for the model-configuration digest, `tempfile` for fixtures.

**Design:** `docs/superpowers/specs/2026-09-04-train-worker-command-design.md`

## Global Constraints

- Run every `cargo`, `rustfmt`, and `clippy` command from `E:\workspace\github\FeatherTalk\rust`. Run every `git` command from `E:\workspace\github\FeatherTalk`.
- The wire protocol is frozen in this slice. `feathertalk-domain` is not edited at all: no new params field, no new stage, no new error code.
- Exactly one change lands outside `feathertalk-worker` and `feathertalk-cli`: `FrameSample::new` in `feathertalk-training-data` (Task 1). `feathertalk-training`, `feathertalk-training-run`, `feathertalk-models`, and `feathertalk-export` are consumed as they are.
- CPU only. `type TrainBackend = CpuAutodiffBackend` (`Autodiff<NdArray<f32>>`), `capabilities.wgpu_training` stays `false`, `backends` stays `[Cpu]`, `adapters` stays `[cpu-0]`.
- One new environment variable, `FEATHERTALK_WORKER_VGG19_DIR`. Training needs no ffmpeg, no SCRFD, no PFLD and no FeatherHuBERT directory: the frames, landmarks and audio features are already inside the locked project.
- Hyperparameters the request cannot carry are worker constants, all at the top of `training.rs`: `DEFAULT_BATCH_SIZE = 1`, `DEFAULT_LEARNING_RATE = 1e-4`, `TRAINING_SEED = 1`, mouth 4.0, temporal 0.5, temporal-mouth 4.0, perceptual 0.01, `MAX_EPOCHS = 10_000`, `WORKER_STATE = "training"`, one checkpoint per epoch boundary.
- Mode mapping, copied from design section 3: domain `Baseline` -> training `Baseline`, stride 0, `frame_count` samples; `MouthRoi` -> `MouthRoi`, stride 0, `frame_count` samples; `Temporal` -> `MouthRoiTemporal`, stride 1, `frame_count - 1` samples. Import the domain enum as `DomainTrainingMode` wherever both are in scope.
- Checkpoint directories are `checkpoint-{global_step:08}`, metrics files `step-{global_step:08}.json`, preview directories `step-{global_step:08}`. Never epoch-numbered.
- Chinese only inside user-facing string literals (task-error summaries, CLI help, rejection reasons). Identifiers, comments, doc comments, and error `detail` text are English.
- No `unwrap`, `expect`, `panic!`, panicking index, or panicking arithmetic outside `#[cfg(test)]` and `tests/`. Diagnostic counters use `saturating_add`; progress totals use `checked_mul` and degrade to `None`; `div_ceil` guards its divisor with `.max(1)`.
- Never `.clone()` a `Copy` device -- clippy's `clone_on_copy` is an error under `-D warnings`. Prefer `ok_or_else` over `ok_or`. `#[allow(clippy::too_many_arguments)]` is the established repo idiom for the two 8-argument functions this slice adds.
- rustfmt defaults apply (`max_width` 100, `fn_call_width` 60, `chain_width` 60). Every code block below is written in rustfmt's output shape; rustfmt owns the order of names inside `use` braces, so run `cargo fmt --all` after editing an import list rather than hand-sorting it.
- Every new unit test runs offline: `Autodiff<NdArray<f32>>`, `parity_micro` model configs, a stub dataset, a constant perceptual extractor. No test reads an environment variable, shells out to ffmpeg, touches a GPU, or loads real VGG19 weights. Only `feathertalk-cli/tests/real_worker.rs` (Task 13) does, and it skips unless its variables are set.
- A 160x160 forward plus backward overflows the default 2 MiB libtest stack in a debug build. Every test that steps a model runs inside `on_step_stack` (64 MiB), mirroring `feathertalk-training-run/tests/support/mod.rs`.
- Stage explicit paths only. Never stage anything under `demo/`. One commit per task with the exact message given. Do not push.

## File Structure

```
rust/crates/feathertalk-training-data/src/dataset.rs          + FrameSample::new and the four plane lengths
rust/crates/feathertalk-training-data/tests/dataset.rs        + constructor coverage
rust/crates/feathertalk-worker/Cargo.toml                     + training crates, burn; hex and sha2 promoted
rust/crates/feathertalk-worker/src/lib.rs                     + train modules and their public surface
rust/crates/feathertalk-worker/src/config.rs                  + ENV_VGG19_DIR, TrainingToolchain
rust/crates/feathertalk-worker/src/handshake.rs               + TaskKind::Train, capabilities.training
rust/crates/feathertalk-worker/src/runtime.rs                 + training_reason, 64 MiB execution thread
rust/crates/feathertalk-worker/src/error_map.rs               + training_task_error, training_data_task_error
rust/crates/feathertalk-worker/src/training.rs                new: backend alias, constants, plan, paths, publish, telemetry
rust/crates/feathertalk-worker/src/train.rs                   new: run_training and execute_train
rust/crates/feathertalk-worker/src/train_result.rs            new: the result payload
rust/crates/feathertalk-worker/src/commands.rs                + the Request::Train arm
rust/crates/feathertalk-worker/tests/support/mod.rs           new: big-stack helper, stub dataset, recorder
rust/crates/feathertalk-worker/tests/training.rs              new: plan, paths, publish, telemetry
rust/crates/feathertalk-worker/tests/train.rs                 new: the loop, resume, cancellation, admission
rust/crates/feathertalk-worker/tests/train_result.rs          new: payload shape
rust/crates/feathertalk-worker/tests/config.rs                + the training toolchain
rust/crates/feathertalk-worker/tests/handshake.rs             + train in the handshake
rust/crates/feathertalk-worker/tests/runtime.rs               + rejection text, execution thread name
rust/crates/feathertalk-worker/tests/error_mapping.rs         + both new mappers
rust/crates/feathertalk-worker/tests/commands.rs              unchanged: its train dispatch test already covers a missing toolchain
rust/crates/feathertalk-cli/src/cli.rs                        + the train subcommand and two mirror enums
rust/crates/feathertalk-cli/src/run.rs                        + build_request arm, enum mapping, inline tests
rust/crates/feathertalk-cli/src/render.rs                     + ENV_WORKER_VGG19_DIR and the train hint
rust/crates/feathertalk-cli/tests/cli.rs                      + train argument handling
rust/crates/feathertalk-cli/tests/real_worker.rs              + the gated end-to-end training run
```

Read `docs/superpowers/specs/2026-09-04-train-worker-command-design.md` once before Task 1. Every "why" below is a pointer back into it.

---

### Task 1: A frame sample the worker can synthesise

**Files:**

- Modify: `rust/crates/feathertalk-training-data/src/dataset.rs`
- Test: `rust/crates/feathertalk-training-data/tests/dataset.rs` (append two tests)

**Interfaces:**

- Produces: `FrameSample::new(image: Vec<f32>, audio: Vec<f32>, target: Vec<f32>, mouth_mask: Vec<f32>) -> Result<FrameSample, TrainingDataError>`, validating `153600 / 16384 / 76800 / 25600` values.
- Consumes: the existing private `sample_error(index, message)` helper and `INNER_SIZE`.

**Why now:** `FrameSample`'s four fields are private and only `ProjectTrainingDataset` fills them, so the worker's loop tests (Task 10) could not build a single `TrainingItem`. Without this constructor they would have to copy the 180-line locked-project fixture from `feathertalk-training-run/tests/fixture/mod.rs`, which binds the worker's orchestration tests to another crate's on-disk format. Design section 1 settled this as the one change outside the worker and the CLI.

- [ ] **Step 1: Write the failing test**

Append to `rust/crates/feathertalk-training-data/tests/dataset.rs` (add `FrameSample` to the existing `feathertalk_training_data::{...}` import):

```rust
#[test]
fn a_synthesised_frame_sample_keeps_the_four_planes() {
    let sample = FrameSample::new(
        vec![0.25; 6 * 160 * 160],
        vec![0.5; 16 * 32 * 32],
        vec![0.75; 3 * 160 * 160],
        vec![1.0; 160 * 160],
    )
    .expect("the four planes match the tensor contract");

    assert_eq!(sample.image().len(), 153_600);
    assert_eq!(sample.audio().len(), 16_384);
    assert_eq!(sample.target().len(), 76_800);
    assert_eq!(sample.mouth_mask().len(), 25_600);
    assert_eq!(sample.image().first().copied(), Some(0.25));
    assert_eq!(sample.mouth_mask().last().copied(), Some(1.0));
}

#[test]
fn a_plane_of_the_wrong_length_is_refused_by_name() {
    let error = FrameSample::new(
        vec![0.0; 6 * 160 * 160],
        vec![0.0; 16 * 32 * 32],
        vec![0.0; 3 * 160 * 160],
        vec![0.0; 160],
    )
    .expect_err("a truncated mouth mask cannot be stacked into a batch");

    let message = error.to_string();
    assert!(message.contains("mouth_mask"), "{message}");
    assert!(message.contains("25600"), "{message}");
    assert!(message.contains("160"), "{message}");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-training-data --test dataset`
Expected: FAIL to compile with `error[E0599]: no function or associated item named new found for struct FrameSample`.

- [ ] **Step 3: Write minimal implementation**

In `rust/crates/feathertalk-training-data/src/dataset.rs`, add the plane lengths next to the existing `INNER_SIZE` constant:

```rust
const INNER_SIZE: usize = 160;
/// The four plane lengths the batch stackers assume, fixed by the tensor
/// contract: `[6, 160, 160]`, `[16, 32, 32]`, `[3, 160, 160]`, `[1, 160, 160]`.
const IMAGE_ELEMENTS: usize = 6 * INNER_SIZE * INNER_SIZE;
const AUDIO_ELEMENTS: usize = 16 * 32 * 32;
const TARGET_ELEMENTS: usize = 3 * INNER_SIZE * INNER_SIZE;
const MOUTH_MASK_ELEMENTS: usize = INNER_SIZE * INNER_SIZE;
```

Then add the constructor as the first item of `impl FrameSample`, before `image()`:

```rust
    /// Assembles a frame sample from four already-flattened planes.
    ///
    /// `ProjectTrainingDataset` fills the fields directly from disk. This is for
    /// callers that synthesise a sample instead -- the worker's training tests
    /// drive the loop with a stub dataset, and every plane length the stackers
    /// rely on is checked here so a wrong one fails at construction rather than
    /// inside a tensor reshape.
    pub fn new(
        image: Vec<f32>,
        audio: Vec<f32>,
        target: Vec<f32>,
        mouth_mask: Vec<f32>,
    ) -> Result<Self, TrainingDataError> {
        check_plane("image", image.len(), IMAGE_ELEMENTS)?;
        check_plane("audio", audio.len(), AUDIO_ELEMENTS)?;
        check_plane("target", target.len(), TARGET_ELEMENTS)?;
        check_plane("mouth_mask", mouth_mask.len(), MOUTH_MASK_ELEMENTS)?;
        Ok(Self {
            image,
            audio,
            target,
            mouth_mask,
        })
    }
```

And the checker as a free function next to the existing `sample_error` helper:

```rust
/// A synthesised sample has no frame index, so the error reports index 0 and
/// names the plane instead.
fn check_plane(plane: &str, actual: usize, expected: usize) -> Result<(), TrainingDataError> {
    if actual == expected {
        return Ok(());
    }
    Err(sample_error(
        0,
        format!("{plane} plane must hold {expected} values, got {actual}"),
    ))
}
```

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-training-data --all-targets
cargo fmt --all -- --check
cargo clippy -p feathertalk-training-data --all-targets -- -D warnings
```

Expected: every test in the package passes including the two new ones, fmt clean, clippy clean.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-training-data/src/dataset.rs rust/crates/feathertalk-training-data/tests/dataset.rs
git commit -m "feat(training-data): expose a frame sample constructor"
```

---

### Task 2: Configure the training toolchain

**Files:**

- Modify: `rust/crates/feathertalk-worker/src/config.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs` (export list)
- Test: `rust/crates/feathertalk-worker/tests/config.rs` (append)

**Interfaces:**

- Produces: `ENV_VGG19_DIR`, `TrainingToolchain::vgg19_dir()`, `WorkerConfig::training()`, `WorkerConfig::training_rejection()`, `WorkerConfig::from_values_with_training(ffprobe, ffmpeg, timeout_ms, scrfd_dir, pfld_dir, hubert_dir, vgg19_dir)`.
- Unchanged: `from_env`, `from_values`, `from_values_with_models`, `from_values_with_toolchains` keep their signatures; the last one delegates with `vgg19_dir = None`.

**Why now:** Every later task asks `config.training()` -- the handshake to announce the command, the runtime to explain a rejection, `commands.rs` to find the VGG19 directory. Design section 6 fixes the shape: same as `FeatureToolchain`, resolved independently of the media and model toolchains, because a worker configured only for training must still be able to announce `train`.

- [ ] **Step 1: Write the failing test**

Append to `rust/crates/feathertalk-worker/tests/config.rs`, and extend the import to `use feathertalk_worker::{ENV_HUBERT_DIR, ENV_VGG19_DIR, WorkerConfig};`:

```rust
fn with_vgg19(vgg19_dir: Option<String>) -> WorkerConfig {
    WorkerConfig::from_values_with_training(None, None, None, None, None, None, vgg19_dir)
}

#[test]
fn an_absolute_directory_resolves_the_training_toolchain() {
    let config = with_vgg19(Some(absolute("vgg19")));

    let training = config.training().expect("an absolute directory resolves");
    assert_eq!(training.vgg19_dir(), PathBuf::from(absolute("vgg19")));
    assert_eq!(config.training_rejection(), None);
    // Training shares nothing with the other toolchains.
    assert!(config.media().is_none());
    assert!(config.models().is_none());
    assert!(config.features().is_none());
}

#[test]
fn a_missing_vgg19_directory_rejects_the_training_toolchain() {
    let config = with_vgg19(None);

    assert!(config.training().is_none());
    let rejection = config.training_rejection().expect("a reason is kept");
    assert!(rejection.contains(ENV_VGG19_DIR), "{rejection}");
}

#[test]
fn a_relative_vgg19_directory_is_rejected_with_the_variable_name() {
    let config = with_vgg19(Some("artifacts/vgg19".to_owned()));

    assert!(config.training().is_none());
    let rejection = config.training_rejection().expect("a reason is kept");
    assert!(rejection.contains(ENV_VGG19_DIR), "{rejection}");
    assert!(rejection.contains("absolute"), "{rejection}");
}

#[test]
fn an_empty_vgg19_directory_is_rejected() {
    let config = with_vgg19(Some("   ".to_owned()));

    assert!(
        config
            .training_rejection()
            .is_some_and(|reason| reason.contains(ENV_VGG19_DIR))
    );
}

#[test]
fn the_toolchain_constructor_leaves_training_unconfigured() {
    let config = WorkerConfig::from_values_with_toolchains(
        None,
        None,
        None,
        None,
        None,
        Some(absolute("feather_hubert")),
    );

    assert!(config.features().is_some());
    assert!(config.training().is_none());
    assert_eq!(ENV_VGG19_DIR, "FEATHERTALK_WORKER_VGG19_DIR");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-worker --test config`
Expected: FAIL to compile, unresolved import `ENV_VGG19_DIR` and no method `from_values_with_training`.

- [ ] **Step 3: Write minimal implementation**

In `config.rs`, add the variable next to `ENV_HUBERT_DIR`:

```rust
pub const ENV_VGG19_DIR: &str = "FEATHERTALK_WORKER_VGG19_DIR";
```

Add the toolchain after `FeatureToolchain`:

```rust
/// Where the worker finds the VGG19 perceptual-loss package.
///
/// Only the shape of the path is checked here, for the same reason as
/// `FeatureToolchain`: the manifest, the licence bundle and the safetensors
/// weights are validated when a training job loads them, because a directory
/// can disappear between startup and the first job.
#[derive(Debug, Clone)]
pub struct TrainingToolchain {
    vgg19_dir: PathBuf,
}

impl TrainingToolchain {
    pub fn vgg19_dir(&self) -> &Path {
        &self.vgg19_dir
    }
}
```

Add the two fields to `WorkerConfig` after `feature_rejection`, read the variable in `from_env`, turn `from_values_with_toolchains` into a delegation, and add the widest constructor:

```rust
    /// The training form: the VGG19 package the perceptual loss reads. Training
    /// needs no media tools and no frame models, so this is orthogonal to every
    /// other toolchain.
    pub fn from_values_with_training(
        ffprobe: Option<String>,
        ffmpeg: Option<String>,
        timeout_ms: Option<String>,
        scrfd_dir: Option<String>,
        pfld_dir: Option<String>,
        hubert_dir: Option<String>,
        vgg19_dir: Option<String>,
    ) -> Self {
```

Move the existing body of `from_values_with_toolchains` into it, add

```rust
        let (training, training_rejection) = match training_toolchain(vgg19_dir) {
            Ok(toolchain) => (Some(toolchain), None),
            Err(reason) => (None, Some(reason)),
        };
```

and the two readers next to `feature_rejection()`:

```rust
    pub fn training(&self) -> Option<&TrainingToolchain> {
        self.training.as_ref()
    }

    pub fn training_rejection(&self) -> Option<&str> {
        self.training_rejection.as_deref()
    }
```

plus the private resolver next to `feature_toolchain`:

```rust
fn training_toolchain(vgg19_dir: Option<String>) -> Result<TrainingToolchain, String> {
    let vgg19_dir = required_path(vgg19_dir, ENV_VGG19_DIR)?;
    Ok(TrainingToolchain { vgg19_dir })
}
```

Then add `ENV_VGG19_DIR` and `TrainingToolchain` to the `pub use config::{...}` list in `lib.rs` and let `cargo fmt` order it.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-worker --test config
cargo fmt --all -- --check
cargo clippy -p feathertalk-worker --all-targets -- -D warnings
```

Expected: the whole `config` test binary passes, fmt clean, clippy clean.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/src/config.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/tests/config.rs
git commit -m "feat(worker): configure the training toolchain"
```

---

### Task 3: Announce the train command

**Files:**

- Modify: `rust/crates/feathertalk-worker/src/handshake.rs`
- Modify: `rust/crates/feathertalk-worker/src/runtime.rs` (the `unsupported_reason` match and a new reason function)
- Modify: `rust/crates/feathertalk-worker/src/lib.rs` (crate doc comment)
- Test: `rust/crates/feathertalk-worker/tests/handshake.rs`, `rust/crates/feathertalk-worker/tests/runtime.rs`

**Interfaces:**

- Produces: `supported_commands` appends `TaskKind::Train` when `config.training().is_some()`; `ready_frame` sets `capabilities.training` from the same predicate; `unsupported_reason` gains a `TaskKind::Train` arm backed by `training_reason(slug, config)`.
- Unchanged: `backends`, `adapters`, `wgpu_training`, `onnx_validation`. Training runs on the same `cpu-0` adapter, so `AdapterLocks` already serialises it against every other job.

**Why now:** The runtime rejects any command outside `supported_commands` before dispatch, so nothing later in this plan is reachable until the handshake admits `train`. Doing it before the command body exists is safe: `commands.rs` still answers `Failed(unsupported(...))` for `Request::Train` until Task 11, and a worker without `FEATHERTALK_WORKER_VGG19_DIR` -- which is every existing test -- announces nothing new.

- [ ] **Step 1: Write the failing test**

Append to `rust/crates/feathertalk-worker/tests/handshake.rs`:

```rust
/// Training is orthogonal to media and models: the VGG19 package alone is
/// enough to offer `train`.
fn training_only() -> WorkerConfig {
    WorkerConfig::from_values_with_training(
        None,
        None,
        None,
        None,
        None,
        None,
        Some(absolute("vgg19-test")),
    )
}

#[test]
fn a_worker_with_a_vgg19_package_offers_train() {
    let config = training_only();
    assert_eq!(config.training_rejection(), None);
    let frame = ready_frame(&config);
    frame.validate().unwrap();
    assert_eq!(
        frame.supported_commands,
        vec![TaskKind::ValidateProject, TaskKind::Train]
    );
    assert!(frame.capabilities.training);
    // Design section 4: the worker never promises GPU training in this slice.
    assert!(!frame.capabilities.wgpu_training);
    assert_eq!(frame.backends, vec![Backend::Cpu]);
    assert_eq!(frame.adapters.len(), 1);
    assert_eq!(frame.adapters[0].id, CPU_ADAPTER_ID);
}

#[test]
fn every_toolchain_plus_vgg19_offers_every_command() {
    let config = WorkerConfig::from_values_with_training(
        Some(absolute("ffprobe-test")),
        Some(absolute("ffmpeg-test")),
        None,
        Some(absolute("scrfd-test")),
        Some(absolute("pfld-test")),
        Some(absolute("hubert-test")),
        Some(absolute("vgg19-test")),
    );
    let frame = ready_frame(&config);
    frame.validate().unwrap();
    assert_eq!(
        frame.supported_commands,
        vec![
            TaskKind::ValidateProject,
            TaskKind::ProbeMedia,
            TaskKind::NormalizeMedia,
            TaskKind::ExtractFrames,
            TaskKind::ExtractFeatures,
            TaskKind::LockAssetPackage,
            TaskKind::Train
        ]
    );
    assert!(frame.capabilities.training);
    assert!(frame.capabilities.ffmpeg);
}

#[test]
fn a_worker_without_a_vgg19_package_leaves_train_out() {
    let config = every_toolchain();
    assert!(config.training().is_none());
    assert!(
        config
            .training_rejection()
            .is_some_and(|reason| reason.contains(ENV_VGG19_DIR))
    );
    assert!(!supported_commands(&config).contains(&TaskKind::Train));
    assert!(!ready_frame(&config).capabilities.training);
}
```

Extend that file's import to include `ENV_VGG19_DIR`. Then in `tests/runtime.rs`, replace the body of `an_unsupported_command_is_rejected_without_creating_a_task` so it also pins the reason text, and add a second case for a rejected training configuration:

```rust
#[test]
fn an_unsupported_command_is_rejected_without_creating_a_task() {
    let harness = Harness::start(media_config(), instant_executor());
    harness.send(&start(&task("0000000a"), train_request()));
    let frames = harness.finish();

    let reasons = rejections(&frames);
    assert_eq!(reasons.len(), 1, "{frames:?}");
    assert!(reasons[0].contains("train"), "{}", reasons[0]);
    // The reason has to name the variable an operator can fix.
    assert!(
        reasons[0].contains("FEATHERTALK_WORKER_VGG19_DIR"),
        "{}",
        reasons[0]
    );
    assert!(
        events(&frames).is_empty(),
        "a rejected start creates no task"
    );
}

#[test]
fn a_rejected_training_configuration_explains_itself() {
    let config = WorkerConfig::from_values_with_training(
        None,
        None,
        None,
        None,
        None,
        None,
        Some("vgg19".to_owned()),
    );
    let harness = Harness::start(config, instant_executor());
    harness.send(&start(&task("0000000a"), train_request()));
    let frames = harness.finish();

    let reasons = rejections(&frames);
    assert_eq!(reasons.len(), 1, "{frames:?}");
    assert!(reasons[0].contains("train"), "{}", reasons[0]);
    // A relative path is a rejection, not an absence, so the reason quotes it.
    assert!(reasons[0].contains("absolute"), "{}", reasons[0]);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-worker --test handshake --test runtime`
Expected: FAIL -- `supported_commands` omits `TaskKind::Train`, `capabilities.training` is `false`, and the rejection reason is the generic list instead of naming `FEATHERTALK_WORKER_VGG19_DIR`.

- [ ] **Step 3: Write minimal implementation**

In `handshake.rs`, append to `supported_commands` after the feature block:

```rust
    // Training needs no media tools and no frame models: the frames, landmarks
    // and audio features are already inside the locked project, so the
    // perceptual-loss package is its only requirement.
    if config.training().is_some() {
        commands.push(TaskKind::Train);
    }
```

and in `ready_frame` replace the hard-coded flag:

```rust
            training: config.training().is_some(),
```

In `runtime.rs`, add the arm before the catch-all in `unsupported_reason`:

```rust
        // Training reads a locked project off disk, so the perceptual-loss
        // package is its only wall.
        TaskKind::Train => training_reason(slug, config),
```

add the reason function after `feature_reason`:

```rust
fn training_reason(slug: &str, config: &WorkerConfig) -> String {
    match config.training_rejection() {
        Some(rejection) => format!(
            "命令 {slug} 需要可用的感知损失模型目录，当前配置被拒绝：{rejection}。修正后重启 worker。"
        ),
        None => format!("命令 {slug} 需要 VGG19 感知损失模型，请设置 {ENV_VGG19_DIR} 后重启 worker。"),
    }
}
```

and add `ENV_VGG19_DIR` to the `use crate::{...}` list at the top of `runtime.rs`. Finally extend the crate doc comment in `lib.rs` to name `train` among the served commands.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-worker --all-targets
cargo fmt --all -- --check
cargo clippy -p feathertalk-worker --all-targets -- -D warnings
```

Expected: every worker test passes. The existing `!frame.capabilities.training` assertions still hold -- none of those configurations sets a VGG19 directory.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/src/handshake.rs rust/crates/feathertalk-worker/src/runtime.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/tests/handshake.rs rust/crates/feathertalk-worker/tests/runtime.rs
git commit -m "feat(worker): announce the train command"
```

---

### Task 4: Give the execution thread a bigger stack

**Files:**

- Modify: `rust/crates/feathertalk-worker/src/runtime.rs` (the executor `thread::spawn`)
- Test: `rust/crates/feathertalk-worker/tests/runtime.rs` (append)

**Interfaces:**

- Produces: the executor thread is created by `thread::Builder::new().name("execution").stack_size(EXECUTION_STACK_BYTES).spawn(...)`, with `EXECUTION_STACK_BYTES = 64 * 1024 * 1024`.
- Unchanged: the input thread stays a detached `thread::spawn`; it only reads lines.

**Why now:** A 160x160 UNet forward plus backward overruns the 2 MiB default stack and takes the whole process down with `STATUS_STACK_OVERFLOW` (`0xc00000fd`), which would make Task 10 unrunnable in-process and Task 13 unrunnable at all. The established precedents are `feathertalk-pfld/tests/runtime.rs` and `feathertalk-training-run/tests/support/mod.rs`, both 64 MiB. Design section 15 rejected a per-command thread inside `train.rs`: handing `&dyn TaskReporter` to another thread means adding `Sync` to the trait and touching every existing implementation and test double.

- [ ] **Step 1: Write the failing test**

Append to `rust/crates/feathertalk-worker/tests/runtime.rs`:

```rust
/// The executor thread must be the named, big-stack one: burn's autodiff graph
/// for a 160x160 step does not fit in the 2 MiB a bare `thread::spawn` gives.
fn thread_name_executor(name: Sender<Option<String>>) -> JobExecutor {
    Box::new(move |_request, _config, _token, _reporter| {
        let _ = name.send(thread::current().name().map(str::to_owned));
        CommandOutcome::Completed(None)
    })
}

#[test]
fn commands_run_on_the_named_execution_thread() {
    let (name_tx, name_rx) = mpsc::channel();
    let harness = Harness::start(bare_config(), thread_name_executor(name_tx));
    harness.send(&start(
        &task("0000000a"),
        Request::ValidateProject(ProjectDirParams {
            project_dir: PathBuf::from("C:/tmp/project"),
        }),
    ));
    let observed = name_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("the executor ran");
    harness.finish();

    assert_eq!(observed.as_deref(), Some("execution"));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-worker --test runtime commands_run_on_the_named_execution_thread`
Expected: FAIL with `assertion `left == right` failed: left: None, right: Some("execution")` -- a bare `thread::spawn` leaves the thread unnamed.

- [ ] **Step 3: Write minimal implementation**

In `runtime.rs`, add the constant next to `NORMALIZE_STEPS`-style module constants at the top of the file:

```rust
/// The executor thread's stack. A 160x160 training step builds a deep autodiff
/// graph and overflows the 2 MiB default in a debug build; 64 MiB is the size
/// `feathertalk-pfld` and `feathertalk-training-run` already settled on.
const EXECUTION_STACK_BYTES: usize = 64 * 1024 * 1024;
```

and replace the executor spawn:

```rust
    let execution = thread::Builder::new()
        .name("execution".to_owned())
        .stack_size(EXECUTION_STACK_BYTES)
        .spawn(move || run_jobs(&job_rx, &execution_tx, execution_config, executor))
        .map_err(|error| DomainError::MalformedFrame {
            reason: format!("cannot start the execution thread: {error}"),
        })?;
```

No `DomainError` variant describes a thread failure, and adding one would widen wire-protocol surface for a case an operator cannot act on. The file already routes an I/O failure (the final `flush`) through `MalformedFrame`, so the spawn failure follows that precedent.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-worker --test runtime
cargo fmt --all -- --check
cargo clippy -p feathertalk-worker --all-targets -- -D warnings
```

Expected: the whole `runtime` binary passes, including the cancellation and shutdown tests that join this thread.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/src/runtime.rs rust/crates/feathertalk-worker/tests/runtime.rs
git commit -m "fix(worker): give the execution thread a bigger stack"
```

---

### Task 5: Map training errors to task errors

**Files:**

- Modify: `rust/crates/feathertalk-worker/src/error_map.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs` (export list)
- Modify: `rust/crates/feathertalk-worker/Cargo.toml` (the two training crates are needed to name the error types)
- Test: `rust/crates/feathertalk-worker/tests/error_mapping.rs` (append)

**Interfaces:**

- Produces: `training_task_error(&TrainingError, stage: TaskStage) -> TaskError` and `training_data_task_error(&TrainingDataError) -> TaskError`.
- Consumes: the private `io_error_code` / `io_summary` helpers and `clamp`, all already in the file.

**Why now:** Every fallible call in Tasks 6 to 11 returns one of these two errors, so the mapper has to exist before the first of them. `training_task_error` is the first mapper in this file that does not use the file-level `FAILURE_STAGE`: design section 13 requires the stage to be a parameter, because a run that fails at step 3000 after forty minutes must not tell the event stream it was still preparing. `Cargo.toml` grows the dependency here rather than in Task 6 so the crate compiles at every commit.

- [ ] **Step 1: Write the failing test**

Append to `rust/crates/feathertalk-worker/tests/error_mapping.rs`, extending the imports with `feathertalk_training::TrainingError`, `feathertalk_training_data::TrainingDataError`, and the two new functions:

```rust
fn try_reserve_error() -> std::collections::TryReserveError {
    Vec::<u64>::new()
        .try_reserve_exact(usize::MAX)
        .expect_err("an impossible reservation fails")
}

#[test]
fn every_training_error_maps_to_a_code_and_a_valid_payload() {
    let cases = vec![
        (
            TrainingError::Io(io_error(io::ErrorKind::StorageFull)),
            ErrorCode::DiskSpaceLow,
        ),
        (
            TrainingError::Io(io_error(io::ErrorKind::PermissionDenied)),
            ErrorCode::WorkerCrashed,
        ),
        (
            TrainingError::InvalidInput("loss is not finite".to_owned()),
            ErrorCode::MediaInvalid,
        ),
        (
            TrainingError::InvalidConfig("batch_size".to_owned()),
            ErrorCode::WorkerCrashed,
        ),
        (
            TrainingError::InvalidDataLoaderConfig("stride".to_owned()),
            ErrorCode::WorkerCrashed,
        ),
        (
            TrainingError::InvalidDataLoaderState("epoch".to_owned()),
            ErrorCode::WorkerCrashed,
        ),
        (
            TrainingError::DataLoaderOverflow {
                operation: "counting steps",
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            TrainingError::PermutationAllocation {
                samples: 8,
                source: try_reserve_error(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            TrainingError::BatchAllocation {
                items: 2,
                source: try_reserve_error(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (TrainingError::StalePreparedBatch, ErrorCode::WorkerCrashed),
        (
            TrainingError::InvalidPackage("manifest".to_owned()),
            ErrorCode::ModelIncompatible,
        ),
        (
            TrainingError::HashMismatch {
                file: "model.safetensors".to_owned(),
                expected: "a".repeat(64),
                actual: "b".repeat(64),
            },
            ErrorCode::ModelIncompatible,
        ),
        (
            TrainingError::Store("record write failed".to_owned()),
            ErrorCode::WorkerCrashed,
        ),
        (
            TrainingError::InvalidCheckpoint("manifest".to_owned()),
            ErrorCode::ModelIncompatible,
        ),
        (
            TrainingError::CheckpointCompatibility("frame_count".to_owned()),
            ErrorCode::ModelIncompatible,
        ),
        (
            TrainingError::CheckpointDirectory("already exists".to_owned()),
            ErrorCode::MediaInvalid,
        ),
    ];

    for (error, expected) in cases {
        let mapped = training_task_error(&error, TaskStage::Preparing);
        assert_eq!(mapped.code, expected, "{error:?}");
        assert_eq!(mapped.stage, TaskStage::Preparing, "{error:?}");
        assert!(!mapped.summary.trim().is_empty(), "{error:?}");
        assert!(!mapped.detail.is_empty(), "{error:?}");
        mapped.validate().unwrap();
    }
}

#[test]
fn a_mid_run_training_failure_keeps_the_stage_it_failed_in() {
    let stage = TaskStage::Training {
        epoch: 3,
        step: 3000,
        loss: 0.125,
    };
    let mapped = training_task_error(&TrainingError::StalePreparedBatch, stage.clone());

    assert_eq!(mapped.stage, stage);
    mapped.validate().unwrap();
}

#[test]
fn a_long_training_detail_is_clamped() {
    let error = TrainingError::InvalidInput("x".repeat(MAX_DETAIL_CHARS * 2));
    let mapped = training_task_error(&error, TaskStage::Preparing);

    assert_eq!(mapped.detail.chars().count(), MAX_DETAIL_CHARS);
    mapped.validate().unwrap();
}

#[test]
fn every_training_data_error_maps_to_a_code_and_a_valid_payload() {
    let cases = vec![
        (
            TrainingDataError::Project {
                path: path(),
                message: "not locked".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            TrainingDataError::Features {
                path: path(),
                message: "truncated".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            TrainingDataError::FeatureShape {
                path: path(),
                expected_tokens: 8,
                actual_tokens: 98,
                dims: 1024,
            },
            ErrorCode::FeatureShapeMismatch,
        ),
        (
            TrainingDataError::FrameIndexOutOfRange {
                index: 9,
                frame_count: 4,
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            TrainingDataError::Frame {
                index: 0,
                path: path(),
                message: "not a jpeg".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            TrainingDataError::Landmarks {
                index: 0,
                path: path(),
                message: "short line".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            TrainingDataError::Sample {
                index: 0,
                message: "image plane".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            TrainingDataError::Batch {
                message: "shape".to_owned(),
            },
            ErrorCode::WorkerCrashed,
        ),
    ];

    for (error, expected) in cases {
        let mapped = training_data_task_error(&error);
        assert_eq!(mapped.code, expected, "{error:?}");
        assert_eq!(mapped.stage, TaskStage::Preparing, "{error:?}");
        assert!(!mapped.summary.trim().is_empty(), "{error:?}");
        mapped.validate().unwrap();
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-worker --test error_mapping`
Expected: FAIL to compile -- unresolved crates `feathertalk_training` / `feathertalk_training_data` and unresolved imports `training_task_error` / `training_data_task_error`.

- [ ] **Step 3: Write minimal implementation**

Add the two path dependencies to `rust/crates/feathertalk-worker/Cargo.toml`, in the alphabetical `[dependencies]` block after `feathertalk-project`:

```toml
feathertalk-training = { path = "../feathertalk-training" }
feathertalk-training-data = { path = "../feathertalk-training-data" }
```

Then in `error_map.rs`, add the imports and the four functions after `package_task_error`:

```rust
/// Maps a training failure onto the wire.
///
/// The stage is a parameter, unlike every other mapper in this file: a run that
/// fails at step 3000 has to report the step it failed in, not `Preparing`. The
/// command body passes `TaskStage::Preparing` while it is assembling the run and
/// the last reported `Training { .. }` once the loop has started.
pub fn training_task_error(error: &TrainingError, stage: TaskStage) -> TaskError {
    TaskError::new(
        training_error_code(error),
        training_summary(error),
        &clamp(&error.to_string()),
        stage,
    )
}

/// Maps a dataset failure. Opening the dataset is the only place one can
/// surface, and that happens during admission, so the stage is fixed.
pub fn training_data_task_error(error: &TrainingDataError) -> TaskError {
    TaskError::new(
        training_data_error_code(error),
        training_data_summary(error),
        &clamp(&error.to_string()),
        FAILURE_STAGE,
    )
}

fn training_error_code(error: &TrainingError) -> ErrorCode {
    match error {
        TrainingError::Io(source) => io_error_code(source),
        // `InvalidInput` is wide: a corrupt sample and a diverged loss both land
        // here, and both are the user's data or hyperparameters.
        TrainingError::InvalidInput(_) | TrainingError::CheckpointDirectory(_) => {
            ErrorCode::MediaInvalid
        }
        TrainingError::InvalidPackage(_)
        | TrainingError::HashMismatch { .. }
        | TrainingError::InvalidCheckpoint(_)
        | TrainingError::CheckpointCompatibility(_) => ErrorCode::ModelIncompatible,
        // Every input to the config variants is a worker constant, so they can
        // only be worker bugs; `Store` is left with storage-layer faults once
        // the checkpoint preflight has rejected the incompatible cases.
        TrainingError::InvalidConfig(_)
        | TrainingError::InvalidDataLoaderConfig(_)
        | TrainingError::InvalidDataLoaderState(_)
        | TrainingError::DataLoaderOverflow { .. }
        | TrainingError::PermutationAllocation { .. }
        | TrainingError::BatchAllocation { .. }
        | TrainingError::StalePreparedBatch
        | TrainingError::Store(_) => ErrorCode::WorkerCrashed,
    }
}

fn training_summary(error: &TrainingError) -> &'static str {
    match error {
        TrainingError::Io(source) => io_summary(source),
        TrainingError::InvalidInput(_) => "训练输入无效",
        TrainingError::InvalidConfig(_) | TrainingError::InvalidDataLoaderConfig(_) => {
            "训练配置无效"
        }
        TrainingError::InvalidDataLoaderState(_)
        | TrainingError::DataLoaderOverflow { .. }
        | TrainingError::PermutationAllocation { .. }
        | TrainingError::BatchAllocation { .. }
        | TrainingError::StalePreparedBatch => "训练运行状态异常",
        TrainingError::InvalidPackage(_) | TrainingError::HashMismatch { .. } => {
            "感知损失模型加载失败"
        }
        TrainingError::Store(_) => "检查点读写失败",
        TrainingError::InvalidCheckpoint(_) | TrainingError::CheckpointCompatibility(_) => {
            "检查点与当前训练不兼容"
        }
        TrainingError::CheckpointDirectory(_) => "检查点目录无效",
    }
}

fn training_data_error_code(error: &TrainingDataError) -> ErrorCode {
    match error {
        // The name lines up with the domain code: the token count and the frame
        // count disagree, which deserves better than a generic media error.
        TrainingDataError::FeatureShape { .. } => ErrorCode::FeatureShapeMismatch,
        // Index and stacking invariants a user cannot reach from a request.
        TrainingDataError::FrameIndexOutOfRange { .. } | TrainingDataError::Batch { .. } => {
            ErrorCode::WorkerCrashed
        }
        TrainingDataError::Project { .. }
        | TrainingDataError::Features { .. }
        | TrainingDataError::Frame { .. }
        | TrainingDataError::Landmarks { .. }
        | TrainingDataError::Sample { .. } => ErrorCode::MediaInvalid,
    }
}

fn training_data_summary(error: &TrainingDataError) -> &'static str {
    match error {
        TrainingDataError::Project { .. } => "工程目录不可用于训练",
        TrainingDataError::Features { .. } => "音频特征文件不可读",
        TrainingDataError::FeatureShape { .. } => "特征长度与帧数不匹配",
        TrainingDataError::FrameIndexOutOfRange { .. } => "样本帧号越界",
        TrainingDataError::Frame { .. } => "训练帧不可读",
        TrainingDataError::Landmarks { .. } => "关键点文件不可读",
        TrainingDataError::Sample { .. } => "训练样本构造失败",
        TrainingDataError::Batch { .. } => "训练批次堆叠失败",
    }
}
```

Export both from `lib.rs` in the `pub use error_map::{...}` list.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-worker --test error_mapping
cargo fmt --all -- --check
cargo clippy -p feathertalk-worker --all-targets -- -D warnings
```

Expected: all four new tests pass, fmt clean, clippy clean. The first build after the new dependencies is slow -- `feathertalk-training` pulls in burn.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/Cargo.toml rust/crates/feathertalk-worker/src/error_map.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/tests/error_mapping.rs rust/Cargo.lock
git commit -m "feat(worker): map training errors to task errors"
```

---

### Task 6: Assemble a training plan

**Files:**

- Modify: `rust/crates/feathertalk-worker/Cargo.toml` (burn, `feathertalk-training-run`; `hex` and `sha2` promoted out of dev-dependencies)
- Create: `rust/crates/feathertalk-worker/src/training.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs`
- Test: `rust/crates/feathertalk-worker/tests/training.rs` (new)

**Interfaces:**

- Produces: `TrainBackend`, `TrainDevice`, `TRAIN_BACKEND_NAME`, the hyperparameter constants, `training_mode`, `sample_count`, `training_config`, `checkpoint_descriptor`, `TrainingPaths`, `TrainingPlan`.
- Consumes: `feathertalk_domain::{TrainParams, TrainingMode as DomainTrainingMode, UnetVariant}`, `feathertalk_export::ModelConfiguration`, `feathertalk_models::backend::CpuAutodiffBackend`, `feathertalk_training::{CheckpointDescriptor, TrainingConfig, TrainingError, TrainingMode}`.

**Why now:** Tasks 7 to 11 all take a `&TrainingPlan`, and the checkpoint publisher (Task 7) needs the path scheme. Everything here is pure -- no filesystem, no tensors -- so it is the cheapest place to pin the three decisions that are easy to get wrong: the mode mapping including `temporal_stride` (design section 3), the five hyperparameters the wire protocol cannot carry (section 5), and the digest that becomes `model_config_sha256` (section 7).

Note on visibility: design section 14 sketches `run_training` as `pub(crate)`, and section 16 puts its tests in `tests/`, which is a separate crate. The tests win: `training.rs` and `train.rs` expose their seams as `pub` and `lib.rs` re-exports them, the same way `execute_extract_features` is public. Nothing else about the layering changes.

- [ ] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-worker/tests/training.rs`:

```rust
use std::path::{Path, PathBuf};

use feathertalk_domain::{TrainParams, TrainingMode as DomainTrainingMode, UnetVariant};
use feathertalk_export::ModelConfiguration;
use feathertalk_models::unet::{MobileOneUnetConfig, OriginalUnetConfig};
use feathertalk_training::TrainingMode;
use feathertalk_worker::{
    DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, MAX_EPOCHS, TRAINING_SEED, TRAIN_BACKEND_NAME,
    TrainingPaths, WORKER_STATE, checkpoint_descriptor, sample_count, training_config,
    training_mode,
};

fn params(mode: DomainTrainingMode, epochs: u32) -> TrainParams {
    TrainParams {
        project_dir: PathBuf::from("C:/tmp/project"),
        mode,
        variant: UnetVariant::OriginalUnet,
        epochs,
        resume: false,
    }
}

#[test]
fn the_three_request_modes_map_onto_the_training_crate() {
    assert_eq!(
        training_mode(DomainTrainingMode::Baseline),
        TrainingMode::Baseline
    );
    assert_eq!(
        training_mode(DomainTrainingMode::MouthRoi),
        TrainingMode::MouthRoi
    );
    // The request has three modes, the training crate three too, but the third
    // pair does not share a name.
    assert_eq!(
        training_mode(DomainTrainingMode::Temporal),
        TrainingMode::MouthRoiTemporal
    );
}

#[test]
fn only_the_temporal_mode_takes_a_stride_and_loses_a_sample() {
    for mode in [DomainTrainingMode::Baseline, DomainTrainingMode::MouthRoi] {
        let config = training_config(&params(mode, 2));
        assert_eq!(config.temporal_stride, 0);
        assert_eq!(sample_count(mode, 188), 188);
    }

    let config = training_config(&params(DomainTrainingMode::Temporal, 2));
    assert_eq!(config.temporal_stride, 1);
    assert_eq!(sample_count(DomainTrainingMode::Temporal, 188), 187);
    // A one-frame project starts no temporal sample at all.
    assert_eq!(sample_count(DomainTrainingMode::Temporal, 1), 0);
    assert_eq!(sample_count(DomainTrainingMode::Temporal, 0), 0);
}

#[test]
fn the_config_takes_five_fields_from_worker_constants() {
    let config = training_config(&params(DomainTrainingMode::MouthRoi, 7));

    assert_eq!(config.total_epochs, 7);
    assert_eq!(config.batch_size, DEFAULT_BATCH_SIZE);
    assert_eq!(config.learning_rate, DEFAULT_LEARNING_RATE);
    assert_eq!(config.mouth_weight, 4.0);
    assert_eq!(config.temporal_weight, 0.5);
    assert_eq!(config.temporal_mouth_weight, 4.0);
    assert_eq!(config.perceptual_weight, 0.01);
    config
        .validate()
        .expect("the assembled config is valid on its own");

    assert_eq!(DEFAULT_BATCH_SIZE, 1);
    assert_eq!(TRAINING_SEED, 1);
    assert_eq!(MAX_EPOCHS, 10_000);
    assert_eq!(WORKER_STATE, "training");
    assert_eq!(TRAIN_BACKEND_NAME, "ndarray-cpu");
}

#[test]
fn the_descriptor_digests_the_model_configuration() {
    let configuration = ModelConfiguration::original_unet(&OriginalUnetConfig::production());
    let descriptor = checkpoint_descriptor(&configuration).expect("the digest is computable");

    descriptor.validate().expect("64 lowercase hex characters");
    assert_eq!(descriptor.model_kind, configuration.model_type());
    assert_eq!(
        descriptor.architecture_version,
        configuration.architecture_version()
    );
    assert_eq!(descriptor.model_config_sha256.len(), 64);
    assert!(
        descriptor
            .model_config_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{}",
        descriptor.model_config_sha256
    );
    // Same configuration, same digest: this value is what a later resume has to
    // match, so it must not depend on anything but the configuration.
    let again = checkpoint_descriptor(&configuration).unwrap();
    assert_eq!(again, descriptor);

    // The training graph is the multi-branch one; a reparameterized descriptor
    // would claim a structure training never builds.
    let mobile = ModelConfiguration::mobileone_unet(&MobileOneUnetConfig::production(), false);
    let mobile_descriptor = checkpoint_descriptor(&mobile).unwrap();
    assert_eq!(mobile_descriptor.model_kind, "mobileone_unet");
    assert_ne!(
        mobile_descriptor.model_config_sha256,
        descriptor.model_config_sha256
    );

    let reparameterized =
        ModelConfiguration::mobileone_unet(&MobileOneUnetConfig::production(), true);
    assert_ne!(
        checkpoint_descriptor(&reparameterized)
            .unwrap()
            .model_config_sha256,
        mobile_descriptor.model_config_sha256
    );

    // A micro configuration digests differently again, which is what makes the
    // offline tests' descriptors distinct from a production run's.
    assert_ne!(
        checkpoint_descriptor(&ModelConfiguration::original_unet(
            &OriginalUnetConfig::parity_micro()
        ))
        .unwrap()
        .model_config_sha256,
        descriptor.model_config_sha256
    );
}

#[test]
fn the_artifact_paths_are_step_numbered() {
    let paths = TrainingPaths::new(Path::new("C:/tmp/project"));

    assert_eq!(
        paths.checkpoints(),
        Path::new("C:/tmp/project").join("models").join("unet")
    );
    assert!(
        paths
            .checkpoint(188)
            .ends_with("models/unet/checkpoint-00000188")
    );
    assert!(
        paths
            .metrics(188)
            .ends_with("outputs/metrics/step-00000188.json")
    );
    assert!(paths.preview(188).ends_with("outputs/preview/step-00000188"));
    // Eight digits, so a nine-digit run does not silently truncate.
    assert!(
        paths
            .checkpoint(123_456_789)
            .ends_with("checkpoint-123456789")
    );
}

#[test]
fn only_step_numbered_checkpoint_names_are_recognised() {
    assert_eq!(TrainingPaths::checkpoint_step("checkpoint-00000188"), Some(188));
    assert_eq!(TrainingPaths::checkpoint_step("checkpoint-00000000"), Some(0));
    // `{:08}` pads but never truncates, so a run past eight digits must still
    // find its own checkpoints.
    assert_eq!(
        TrainingPaths::checkpoint_step("checkpoint-123456789"),
        Some(123_456_789)
    );
    for name in [
        "checkpoint-188",
        "checkpoint-0000018x",
        "checkpoint-",
        "checkpoint",
        ".publish-1234-0",
        ".retired-1234-0",
        "last",
    ] {
        assert_eq!(TrainingPaths::checkpoint_step(name), None, "{name}");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-worker --test training`
Expected: FAIL to compile -- `feathertalk_worker` exports none of these names and `feathertalk_models`/`feathertalk_export` are not yet imported by the test.

- [ ] **Step 3: Write minimal implementation**

In `rust/crates/feathertalk-worker/Cargo.toml`, add to `[dependencies]` (alphabetical, `burn` first):

```toml
burn = { workspace = true }
```

plus `feathertalk-training-run = { path = "../feathertalk-training-run" }` next to the two training crates from Task 5, and move `hex` and `sha2` out of `[dev-dependencies]` into `[dependencies]`:

```toml
hex = { workspace = true }
sha2 = { workspace = true }
```

`[dev-dependencies]` keeps only `tempfile`.

Create `rust/crates/feathertalk-worker/src/training.rs`:

```rust
use std::path::{Path, PathBuf};

use burn::tensor::backend::Backend;
use feathertalk_domain::{TrainParams, TrainingMode as DomainTrainingMode, UnetVariant};
use feathertalk_export::ModelConfiguration;
use feathertalk_models::backend::CpuAutodiffBackend;
use feathertalk_training::{CheckpointDescriptor, TrainingConfig, TrainingError, TrainingMode};
use sha2::{Digest, Sha256};

/// The backend every training run in this slice uses.
///
/// Design section 4: the handshake keeps `wgpu_training` false, so the worker
/// never promises GPU training and there is no silent fallback to explain. The
/// runner, the losses and the perceptual extractor are all generic over
/// `AutodiffBackend`, so a later GPU slice replaces this alias and adds one
/// dispatch, not a rewrite.
pub type TrainBackend = CpuAutodiffBackend;

/// The device that goes with it. `NdArrayDevice` is `Copy`, so it is passed by
/// reference and copied rather than cloned.
pub type TrainDevice = <TrainBackend as Backend>::Device;

/// What the result payload calls the backend, so the artefact records what it
/// actually ran on (design section 12).
pub const TRAIN_BACKEND_NAME: &str = "ndarray-cpu";

/// One sample per step: the wire protocol carries no batch size, so any value is
/// a placeholder. One keeps peak memory smallest and makes a step equal a
/// sample, which is the finest possible granularity for progress and loss.
pub const DEFAULT_BATCH_SIZE: u64 = 1;

/// The migration design's learning rate (section 8.2).
pub const DEFAULT_LEARNING_RATE: f64 = 1e-4;

/// The sampling seed. Fixed, so two runs of the same project see the same order.
pub const TRAINING_SEED: u64 = 1;

/// The largest accepted `epochs`. `TrainingConfig::validate` would reject zero
/// anyway; the ceiling exists so a typo cannot ask for a run no one can finish.
pub const MAX_EPOCHS: u32 = 10_000;

/// The `worker_state` every metrics file and preview manifest records.
/// `validate_worker_state` accepts 1 to 128 lowercase letters, digits, hyphens
/// and underscores.
pub const WORKER_STATE: &str = "training";

/// Loss weights, all from the migration design section 8.2.
const MOUTH_WEIGHT: f64 = 4.0;
const TEMPORAL_WEIGHT: f64 = 0.5;
const TEMPORAL_MOUTH_WEIGHT: f64 = 4.0;
const PERCEPTUAL_WEIGHT: f64 = 0.01;

const MODELS_DIR: &str = "models";
const UNET_DIR: &str = "unet";
const OUTPUTS_DIR: &str = "outputs";
const METRICS_DIR: &str = "metrics";
const PREVIEW_DIR: &str = "preview";
const CHECKPOINT_PREFIX: &str = "checkpoint-";
const STEP_PREFIX: &str = "step-";

/// How many digits a step number is padded to in an artefact name.
const STEP_DIGITS: usize = 8;

/// Maps the request's mode onto the training crate's mode. The two enums have
/// three variants each and only two names in common.
pub fn training_mode(mode: DomainTrainingMode) -> TrainingMode {
    match mode {
        DomainTrainingMode::Baseline => TrainingMode::Baseline,
        DomainTrainingMode::MouthRoi => TrainingMode::MouthRoi,
        DomainTrainingMode::Temporal => TrainingMode::MouthRoiTemporal,
    }
}

/// `TrainingConfig::validate` demands a zero stride outside the temporal mode
/// and a positive one inside it, and `DataLoaderConfig::sample_count` subtracts
/// the stride from the frame count.
fn temporal_stride(mode: DomainTrainingMode) -> u64 {
    match mode {
        DomainTrainingMode::Baseline | DomainTrainingMode::MouthRoi => 0,
        DomainTrainingMode::Temporal => 1,
    }
}

/// How many samples one epoch holds. A temporal pair needs a successor, so the
/// last frame starts no sample.
pub fn sample_count(mode: DomainTrainingMode, frame_count: u64) -> u64 {
    match mode {
        DomainTrainingMode::Baseline | DomainTrainingMode::MouthRoi => frame_count,
        DomainTrainingMode::Temporal => frame_count.saturating_sub(1),
    }
}

/// The nine-field training config: four fields from the request, five from the
/// constants above (design section 5).
pub fn training_config(params: &TrainParams) -> TrainingConfig {
    TrainingConfig {
        mode: training_mode(params.mode),
        batch_size: DEFAULT_BATCH_SIZE,
        learning_rate: DEFAULT_LEARNING_RATE,
        total_epochs: u64::from(params.epochs),
        temporal_stride: temporal_stride(params.mode),
        mouth_weight: MOUTH_WEIGHT,
        temporal_weight: TEMPORAL_WEIGHT,
        temporal_mouth_weight: TEMPORAL_MOUTH_WEIGHT,
        perceptual_weight: PERCEPTUAL_WEIGHT,
    }
}

/// Derives the checkpoint descriptor from the model configuration instead of
/// hand-writing its three fields.
///
/// `ModelConfiguration` is a fixed-field, map-free structure, so its serialised
/// bytes are stable and their digest is the natural canonical form of "this
/// model configuration". `CheckpointDescriptor::validate` requires 64 lowercase
/// hex characters, which is exactly what `hex::encode` produces. This is the
/// first place in the workspace that computes the value; every existing test
/// uses a repeated-digit placeholder.
pub fn checkpoint_descriptor(
    configuration: &ModelConfiguration,
) -> Result<CheckpointDescriptor, TrainingError> {
    let bytes = serde_json::to_vec(configuration).map_err(|error| {
        TrainingError::InvalidConfig(format!("serialize model configuration: {error}"))
    })?;
    let descriptor = CheckpointDescriptor::new(
        configuration.model_type(),
        configuration.architecture_version(),
        hex::encode(Sha256::digest(&bytes)),
    );
    descriptor.validate()?;
    Ok(descriptor)
}

/// Where a training run writes. Every directory is created by its writer, so
/// nothing here touches the filesystem.
#[derive(Debug, Clone)]
pub struct TrainingPaths {
    checkpoints: PathBuf,
    metrics: PathBuf,
    previews: PathBuf,
}

impl TrainingPaths {
    pub fn new(project_dir: &Path) -> Self {
        Self {
            checkpoints: project_dir.join(MODELS_DIR).join(UNET_DIR),
            metrics: project_dir.join(OUTPUTS_DIR).join(METRICS_DIR),
            previews: project_dir.join(OUTPUTS_DIR).join(PREVIEW_DIR),
        }
    }

    /// The directory every checkpoint of this project lives in.
    pub fn checkpoints(&self) -> &Path {
        &self.checkpoints
    }

    /// `models/unet/checkpoint-00000188`.
    ///
    /// Named by step, never by epoch: a cancellation lands mid-epoch, and
    /// `DataLoaderState.next_position` already carries the position inside the
    /// epoch, so one naming scheme covers both save points and resume only has
    /// to take the largest number.
    pub fn checkpoint(&self, global_step: u64) -> PathBuf {
        self.checkpoints
            .join(format!("{CHECKPOINT_PREFIX}{global_step:08}"))
    }

    /// `outputs/metrics/step-00000188.json`.
    pub fn metrics(&self, global_step: u64) -> PathBuf {
        self.metrics
            .join(format!("{STEP_PREFIX}{global_step:08}.json"))
    }

    /// `outputs/preview/step-00000188`.
    pub fn preview(&self, global_step: u64) -> PathBuf {
        self.previews.join(format!("{STEP_PREFIX}{global_step:08}"))
    }

    /// The step a checkpoint directory name encodes, or `None` if the name is
    /// not one of ours.
    ///
    /// At least eight ASCII digits are required, so a hand-made
    /// `checkpoint-188` or a stray `.publish-*` never becomes a resume
    /// candidate. A step past eight digits still round-trips because `{:08}`
    /// only pads.
    pub fn checkpoint_step(name: &str) -> Option<u64> {
        let digits = name.strip_prefix(CHECKPOINT_PREFIX)?;
        if digits.len() < STEP_DIGITS || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            return None;
        }
        digits.parse::<u64>().ok()
    }
}

/// Everything the command settled before the first optimizer step.
#[derive(Debug, Clone)]
pub struct TrainingPlan {
    pub mode: DomainTrainingMode,
    pub variant: UnetVariant,
    pub epochs_requested: u32,
    pub frame_count: u64,
    pub config: TrainingConfig,
    pub descriptor: CheckpointDescriptor,
    pub paths: TrainingPaths,
    /// The checkpoint this run resumes from, `None` for a fresh run. It is also
    /// what the result payload reports as `resumed_from`.
    pub resume_from: Option<PathBuf>,
}
```

Add `mod training;` and the `pub use training::{...}` list to `lib.rs`, then let `cargo fmt` order both.

Note the deliberate looseness in `checkpoint_step`: a name is accepted when it has *at least* eight digits, because `{global_step:08}` pads but does not truncate, so a run past 99,999,999 steps must still find its own checkpoints. Nine digits with a leading zero (`checkpoint-000001888`) is therefore also accepted and parses to 1888 -- an unreachable name that no writer produces. The test above asserts the rejections that matter: too few digits and a non-digit.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-worker --test training
cargo fmt --all -- --check
cargo clippy -p feathertalk-worker --all-targets -- -D warnings
```

Expected: six tests pass, fmt clean, clippy clean.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/Cargo.toml rust/crates/feathertalk-worker/src/training.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/tests/training.rs rust/Cargo.lock
git commit -m "feat(worker): assemble a training plan"
```

---

### Task 7: Publish training checkpoints

**Files:**

- Modify: `rust/crates/feathertalk-worker/src/training.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs` (export list)
- Test: `rust/crates/feathertalk-worker/tests/training.rs` (append)

**Interfaces:**

- Produces: `publish_checkpoint(&TrainingPaths, global_step, save: impl FnOnce(&Path) -> Result<(), TrainingError>) -> Result<PathBuf, TrainingError>` and `latest_checkpoint(&TrainingPaths) -> Result<Option<PathBuf>, TrainingError>`.
- The `save` closure is the seam: the command passes `|staged| runner.save_checkpoint(staged, descriptor.clone()).map(|_| ())`, and the tests pass a closure that writes a marker file, so the publish routine is testable without burn.

**Why now:** `save_training_checkpoint` refuses a destination that already exists, and a resumed run can legitimately arrive at a step number it already saved -- resume from step 188, get cancelled at 188 again. Without the staging rename the first save point of a resumed run fails. Design section 8 fixes the sequence: stage, retire the old name, rename into place, best-effort delete the retired copy, so the disk always holds at least one complete checkpoint.

- [ ] **Step 1: Write the failing test**

Append to `rust/crates/feathertalk-worker/tests/training.rs`, extending the imports with `feathertalk_training::TrainingError` and `feathertalk_worker::{latest_checkpoint, publish_checkpoint}`:

```rust
/// Behaves like `save_training_checkpoint`: it creates the destination itself
/// and refuses one that already exists.
fn fake_save(marker: &'static str) -> impl FnOnce(&Path) -> Result<(), TrainingError> {
    move |staged: &Path| {
        if staged.exists() {
            return Err(TrainingError::CheckpointDirectory(format!(
                "checkpoint destination already exists: {}",
                staged.display()
            )));
        }
        std::fs::create_dir_all(staged)?;
        std::fs::write(staged.join("manifest.json"), marker)?;
        Ok(())
    }
}

fn names(directory: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(directory)
        .expect("the directory is readable")
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

fn marker_of(checkpoint: &Path) -> String {
    std::fs::read_to_string(checkpoint.join("manifest.json")).expect("the marker is readable")
}

#[test]
fn a_published_checkpoint_lands_under_its_step_name() {
    let root = tempfile::tempdir().unwrap();
    let paths = TrainingPaths::new(root.path());

    let published = publish_checkpoint(&paths, 4, fake_save("first")).expect("the publish succeeds");

    assert_eq!(published, paths.checkpoint(4));
    assert_eq!(marker_of(&published), "first");
    // Nothing but the checkpoint is left behind.
    assert_eq!(names(paths.checkpoints()), vec!["checkpoint-00000004"]);
}

#[test]
fn publishing_over_an_existing_step_retires_the_old_directory() {
    let root = tempfile::tempdir().unwrap();
    let paths = TrainingPaths::new(root.path());
    let occupied = paths.checkpoint(8);
    std::fs::create_dir_all(&occupied).unwrap();
    std::fs::write(occupied.join("manifest.json"), "stale").unwrap();
    std::fs::write(occupied.join("leftover.bin"), "junk").unwrap();

    let published = publish_checkpoint(&paths, 8, fake_save("fresh")).expect("the publish succeeds");

    assert_eq!(published, occupied);
    assert_eq!(marker_of(&published), "fresh");
    // The staged directory replaced the old one wholesale rather than merging
    // into it, so the stale file is gone.
    assert!(!published.join("leftover.bin").exists());
    assert_eq!(names(paths.checkpoints()), vec!["checkpoint-00000008"]);
}

#[test]
fn a_failed_save_leaves_neither_a_destination_nor_a_staging_directory() {
    let root = tempfile::tempdir().unwrap();
    let paths = TrainingPaths::new(root.path());

    let error = publish_checkpoint(&paths, 4, |staged: &Path| {
        std::fs::create_dir_all(staged)?;
        std::fs::write(staged.join("partial.bin"), "half").unwrap();
        Err(TrainingError::Store("record write failed".to_owned()))
    })
    .expect_err("a failing save fails the publish");

    assert!(matches!(error, TrainingError::Store(_)), "{error:?}");
    assert!(!paths.checkpoint(4).exists());
    assert!(names(paths.checkpoints()).is_empty());
}

#[test]
fn the_latest_checkpoint_is_the_largest_step() {
    let root = tempfile::tempdir().unwrap();
    let paths = TrainingPaths::new(root.path());
    for step in [4_u64, 12, 8] {
        publish_checkpoint(&paths, step, fake_save("body")).unwrap();
    }
    // Decoys: a hand-made short name, a file wearing a checkpoint name, and a
    // staging directory a crashed run left behind.
    std::fs::create_dir_all(paths.checkpoints().join("checkpoint-16")).unwrap();
    std::fs::write(paths.checkpoints().join("checkpoint-00000020"), "not a dir").unwrap();
    std::fs::create_dir_all(paths.checkpoints().join(".publish-1-0")).unwrap();

    let latest = latest_checkpoint(&paths)
        .expect("the directory is readable")
        .expect("three checkpoints exist");

    assert_eq!(latest, paths.checkpoint(12));
}

#[test]
fn a_project_that_never_trained_has_no_latest_checkpoint() {
    let root = tempfile::tempdir().unwrap();
    let paths = TrainingPaths::new(root.path());

    assert_eq!(latest_checkpoint(&paths).expect("a missing directory is not an error"), None);

    // An empty directory is the same answer.
    std::fs::create_dir_all(paths.checkpoints()).unwrap();
    assert_eq!(latest_checkpoint(&paths).unwrap(), None);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-worker --test training`
Expected: FAIL to compile -- unresolved imports `publish_checkpoint` and `latest_checkpoint`.

- [ ] **Step 3: Write minimal implementation**

Extend the imports of `rust/crates/feathertalk-worker/src/training.rs` with `use std::{fs, io};` and append:

```rust
/// Prefix of the directory a checkpoint is written into before it takes its
/// step name.
const PUBLISH_PREFIX: &str = ".publish-";
/// Prefix of the directory an overwritten checkpoint is moved aside into.
const RETIRED_PREFIX: &str = ".retired-";
/// How many names to try before giving up on finding a free one.
const MAX_PUBLISH_ATTEMPTS: u32 = 1024;

/// Writes the checkpoint for `global_step` through a staging directory so the
/// step's own name only ever holds a complete checkpoint.
///
/// `save` is handed a directory that does not exist yet -- the contract
/// `save_training_checkpoint` already expects -- and only has to write. The
/// rename into place happens here.
pub fn publish_checkpoint<F>(
    paths: &TrainingPaths,
    global_step: u64,
    save: F,
) -> Result<PathBuf, TrainingError>
where
    F: FnOnce(&Path) -> Result<(), TrainingError>,
{
    let root = paths.checkpoints();
    fs::create_dir_all(root)?;
    let staged = reserve_name(root, PUBLISH_PREFIX)?;
    if let Err(error) = save(&staged) {
        // The staging name is ours alone, so removing it cannot take a real
        // checkpoint down with it.
        let _ = fs::remove_dir_all(&staged);
        return Err(error);
    }

    let destination = paths.checkpoint(global_step);
    let retired = match fs::symlink_metadata(&destination) {
        Ok(_) => {
            let retired = reserve_name(root, RETIRED_PREFIX)?;
            fs::rename(&destination, &retired)?;
            Some(retired)
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(TrainingError::from(error)),
    };

    fs::rename(&staged, &destination)?;
    if let Some(retired) = retired {
        // The destination is already correct, so a leftover copy is untidy
        // rather than broken and is not worth failing the step over.
        let _ = fs::remove_dir_all(&retired);
    }
    Ok(destination)
}

/// Reserves a free name under `root` by creating the directory and removing it
/// again, which proves the name is both free and creatable.
///
/// The name comes back unoccupied because `save_training_checkpoint` refuses a
/// destination that exists. Two workers on one project would race here, which
/// the pid in the name makes harmless in practice and which the
/// one-worker-per-project rule rules out anyway.
fn reserve_name(root: &Path, prefix: &str) -> Result<PathBuf, TrainingError> {
    let pid = std::process::id();
    for attempt in 0..MAX_PUBLISH_ATTEMPTS {
        let candidate = root.join(format!("{prefix}{pid}-{attempt}"));
        match fs::create_dir(&candidate) {
            Ok(()) => {
                fs::remove_dir(&candidate)?;
                return Ok(candidate);
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(TrainingError::from(error)),
        }
    }
    Err(TrainingError::CheckpointDirectory(format!(
        "no free staging name under {} after {MAX_PUBLISH_ATTEMPTS} attempts",
        root.display()
    )))
}

/// The checkpoint with the largest step, or `None` when the project has never
/// trained.
pub fn latest_checkpoint(paths: &TrainingPaths) -> Result<Option<PathBuf>, TrainingError> {
    let entries = match fs::read_dir(paths.checkpoints()) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(TrainingError::from(error)),
    };

    let mut best: Option<(u64, PathBuf)> = None;
    for entry in entries {
        let entry = entry?;
        // Bind the name before borrowing from it: `file_name` returns an owned
        // `OsString`, and `entry.file_name().to_str()` would not live long
        // enough (E0716).
        let name = entry.file_name();
        let Some(step) = name.to_str().and_then(TrainingPaths::checkpoint_step) else {
            continue;
        };
        if !entry.file_type()?.is_dir() {
            continue;
        }
        if best.as_ref().is_none_or(|(seen, _)| step > *seen) {
            best = Some((step, entry.path()));
        }
    }
    Ok(best.map(|(_, path)| path))
}
```

Add `latest_checkpoint` and `publish_checkpoint` to the `pub use training::{...}` list in `lib.rs`.

The ordering inside `publish_checkpoint` is the one property worth defending: the destination is unlinked only once a complete replacement sits on disk, and the retired copy is deleted only once the rename has succeeded. If the process dies at the worst moment -- after the retire, before the rename -- the data is still on disk under `.retired-<pid>-<n>` and the next run resumes from the previous step instead. A crash during `save` leaves a `.publish-*` directory, which `latest_checkpoint` ignores because the name carries no `checkpoint-` prefix.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-worker --test training
cargo fmt --all -- --check
cargo clippy -p feathertalk-worker --all-targets -- -D warnings
```

Expected: eleven tests pass, fmt clean, clippy clean.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/src/training.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/tests/training.rs
git commit -m "feat(worker): publish training checkpoints"
```

---

### Task 8: Write training telemetry

**Files:**

- Modify: `rust/crates/feathertalk-worker/src/training.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs` (export list)
- Test: `rust/crates/feathertalk-worker/tests/training.rs` (append)

**Interfaces:**

- Produces: `write_metrics_unless_present(&TrainingPaths, u64, &TrainingMetrics) -> Result<bool, TrainingError>`, `write_preview_unless_present(&TrainingPaths, u64, &PreviewArtifact) -> Result<bool, TrainingError>` and `preview_sample(frame_count: u64) -> TrainingSample`.
- `true` means written, `false` means something was already there and nothing was touched. The counters in the result payload count the `true`s.

**Why now:** both writers call `reject_existing_destination`, and a resumed run walks over step numbers it may already have written telemetry for -- resume from step 188, finish the next epoch at step 188 again when the epoch length is unchanged. A checkpoint is state, so Task 7 goes to the trouble of replacing it atomically; metrics and previews are diagnostics, so losing a two-hour run over a duplicate preview would be absurd. The collision is a skip, and the skip is counted in the payload rather than hidden.

- [ ] **Step 1: Write the failing test**

Append to `rust/crates/feathertalk-worker/tests/training.rs`, extending the imports with `feathertalk_training::{PREVIEW_TENSOR_ELEMENTS, PreviewArtifact, TrainingMetrics, TrainingSample, read_preview_artifact, read_training_metrics}` and `feathertalk_worker::{preview_sample, write_metrics_unless_present, write_preview_unless_present}`:

```rust
const MODEL_KIND: &str = "original_unet";

fn metrics_fixture(global_step: u64, total_loss: f64) -> TrainingMetrics {
    TrainingMetrics::new(
        TrainingMode::Baseline,
        1,
        global_step,
        total_loss,
        total_loss,
        0.01,
        None,
        None,
        None,
        global_step,
        2.5,
        12.0,
        None,
        WORKER_STATE,
    )
    .expect("the fixture is valid")
}

fn preview_fixture(global_step: u64, prediction: f32) -> PreviewArtifact {
    PreviewArtifact::new(
        0,
        2,
        1,
        global_step,
        MODEL_KIND,
        "a".repeat(64),
        WORKER_STATE,
        vec![prediction; PREVIEW_TENSOR_ELEMENTS],
        vec![0.5; PREVIEW_TENSOR_ELEMENTS],
        vec![1.0; PREVIEW_TENSOR_ELEMENTS],
    )
    .expect("the fixture is valid")
}

#[test]
fn metrics_are_written_once_per_step() {
    let root = tempfile::tempdir().unwrap();
    let paths = TrainingPaths::new(root.path());

    assert!(write_metrics_unless_present(&paths, 4, &metrics_fixture(4, 0.5)).unwrap());
    // A resumed run reaches step 4 again with a different loss.
    assert!(!write_metrics_unless_present(&paths, 4, &metrics_fixture(4, 0.25)).unwrap());

    let written = read_training_metrics(paths.metrics(4)).expect("the file reads back");
    // The first write is the one kept.
    assert_eq!(written.total_loss, 0.5);
    assert_eq!(written.worker_state, WORKER_STATE);
}

#[test]
fn previews_are_written_once_per_step() {
    let root = tempfile::tempdir().unwrap();
    let paths = TrainingPaths::new(root.path());
    let sha = "a".repeat(64);

    assert!(write_preview_unless_present(&paths, 8, &preview_fixture(8, 0.25)).unwrap());
    assert!(!write_preview_unless_present(&paths, 8, &preview_fixture(8, 0.75)).unwrap());

    let (artifact, manifest) =
        read_preview_artifact(paths.preview(8), MODEL_KIND, &sha).expect("the artifact reads back");
    assert_eq!(manifest.global_step, 8);
    assert_eq!(artifact.prediction()[0], 0.25);
}

#[test]
fn the_preview_pairs_the_first_frame_with_the_middle_one() {
    assert_eq!(
        preview_sample(8),
        TrainingSample::SingleFrame {
            target_index: 0,
            reference_index: 4,
        }
    );
    // A one-frame project has to reference itself, which the dataset allows.
    assert_eq!(
        preview_sample(1),
        TrainingSample::SingleFrame {
            target_index: 0,
            reference_index: 0,
        }
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-worker --test training`
Expected: FAIL to compile -- unresolved imports `preview_sample`, `write_metrics_unless_present` and `write_preview_unless_present`.

- [ ] **Step 3: Write minimal implementation**

Extend the `feathertalk_training` import in `rust/crates/feathertalk-worker/src/training.rs` with `PreviewArtifact, TrainingMetrics, TrainingSample, write_preview_artifact, write_training_metrics` and append:

```rust
/// Writes the metrics file for `global_step`, or reports that one was already
/// there.
pub fn write_metrics_unless_present(
    paths: &TrainingPaths,
    global_step: u64,
    metrics: &TrainingMetrics,
) -> Result<bool, TrainingError> {
    let path = paths.metrics(global_step);
    if exists(&path)? {
        return Ok(false);
    }
    write_training_metrics(&path, metrics)?;
    Ok(true)
}

/// Writes the preview directory for `global_step`, or reports that one was
/// already there.
pub fn write_preview_unless_present(
    paths: &TrainingPaths,
    global_step: u64,
    artifact: &PreviewArtifact,
) -> Result<bool, TrainingError> {
    let destination = paths.preview(global_step);
    if exists(&destination)? {
        return Ok(false);
    }
    write_preview_artifact(&destination, artifact)?;
    Ok(true)
}

/// The pair every preview renders: frame zero, driven by the audio and judged
/// against the ground truth of frame zero, wearing the middle frame of the clip
/// as its appearance reference.
///
/// Fixed rather than sampled, because the whole point of the artifact is to
/// watch one frame improve from epoch to epoch. A one-frame project references
/// itself, which is in range and therefore not an error.
pub fn preview_sample(frame_count: u64) -> TrainingSample {
    TrainingSample::SingleFrame {
        target_index: 0,
        reference_index: frame_count / 2,
    }
}

/// Whether `path` is already taken, without following a symlink.
fn exists(path: &Path) -> Result<bool, TrainingError> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(TrainingError::from(error)),
    }
}
```

Add `preview_sample`, `write_metrics_unless_present` and `write_preview_unless_present` to the `pub use training::{...}` list in `lib.rs`.

`symlink_metadata` rather than `Path::exists` on purpose: a dangling symlink standing where the metrics file should be is *not* a free name, and following it would write outside the project. The check is deliberately not atomic -- two workers on one project would still race -- but the underlying writers refuse an occupied destination themselves, so the worst outcome of a lost race is a hard error instead of a skip, and never a half-written file.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-worker --test training
cargo fmt --all -- --check
cargo clippy -p feathertalk-worker --all-targets -- -D warnings
```

Expected: fourteen tests pass, fmt clean, clippy clean.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/src/training.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/tests/training.rs
git commit -m "feat(worker): write training telemetry"
```

---

### Task 9: Report a training result

**Files:**

- Create: `rust/crates/feathertalk-worker/src/train_result.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs` (module and export list)
- Test: `rust/crates/feathertalk-worker/tests/train_result.rs`

**Interfaces:**

- Produces: `TrainSummary<'a>` and `train_to_json(&TrainSummary<'_>) -> serde_json::Value`, the seventeen fields of design section 12.

**Why now:** the loop of Task 10 has to hand its numbers somewhere, and shaping them in the loop would make the payload untestable without a training run. Shaping it first means the loop only has to fill a struct, and the payload gets asserted field by field in a test that finishes in milliseconds.

- [ ] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-worker/tests/train_result.rs`:

```rust
use std::path::PathBuf;

use feathertalk_domain::{TrainingMode as DomainTrainingMode, UnetVariant};
use feathertalk_export::ModelConfiguration;
use feathertalk_models::unet::{MobileOneUnetConfig, OriginalUnetConfig};
use feathertalk_training::CheckpointDescriptor;
use feathertalk_worker::{TRAIN_BACKEND_NAME, TrainSummary, checkpoint_descriptor, train_to_json};
use serde_json::{Value, json};

fn descriptor() -> CheckpointDescriptor {
    let configuration = ModelConfiguration::original_unet(&OriginalUnetConfig::parity_micro());
    checkpoint_descriptor(&configuration).expect("the configuration serialises")
}

/// A minimal summary, for the tests that only care about one field.
///
/// The descriptor stays `original_unet` even when the variant does not: what is
/// under test here is the enum mapping, not the model.
fn payload_of(mode: DomainTrainingMode, variant: UnetVariant) -> Value {
    let descriptor = descriptor();
    train_to_json(&TrainSummary {
        mode,
        variant,
        descriptor: &descriptor,
        frame_count: 4,
        epochs_requested: 1,
        epochs_completed: 1,
        global_step: 4,
        samples_seen: 4,
        total_loss: Some(1.0),
        resumed_from: None,
        checkpoint_dir: None,
        checkpoints_written: 1,
        metrics_written: 1,
        previews_written: 1,
    })
}

#[test]
fn a_finished_run_reports_every_field_the_design_lists() {
    let descriptor = descriptor();
    let checkpoint = PathBuf::from("C:/tmp/project/models/unet/checkpoint-00000376");

    let payload = train_to_json(&TrainSummary {
        mode: DomainTrainingMode::MouthRoi,
        variant: UnetVariant::OriginalUnet,
        descriptor: &descriptor,
        frame_count: 188,
        epochs_requested: 2,
        epochs_completed: 2,
        global_step: 376,
        samples_seen: 376,
        total_loss: Some(0.0412),
        resumed_from: None,
        checkpoint_dir: Some(&checkpoint),
        checkpoints_written: 2,
        metrics_written: 2,
        previews_written: 2,
    });

    assert_eq!(
        payload,
        json!({
            "mode": "mouth_roi",
            "variant": "original_unet",
            "backend": TRAIN_BACKEND_NAME,
            "model_kind": "original_unet",
            "architecture_version": descriptor.architecture_version,
            "model_config_sha256": descriptor.model_config_sha256,
            "frame_count": 188,
            "epochs_requested": 2,
            "epochs_completed": 2,
            "global_step": 376,
            "samples_seen": 376,
            "total_loss": 0.0412,
            "resumed_from": null,
            "checkpoint_dir": "C:/tmp/project/models/unet/checkpoint-00000376",
            "checkpoints_written": 2,
            "metrics_written": 2,
            "previews_written": 2,
        })
    );
}

#[test]
fn a_resume_with_nothing_left_to_do_invents_nothing() {
    let descriptor = descriptor();
    let resumed = PathBuf::from("C:/tmp/project/models/unet/checkpoint-00000748");

    let payload = train_to_json(&TrainSummary {
        mode: DomainTrainingMode::Temporal,
        variant: UnetVariant::MobileOneUnet,
        descriptor: &descriptor,
        frame_count: 188,
        epochs_requested: 4,
        epochs_completed: 4,
        global_step: 748,
        samples_seen: 0,
        total_loss: None,
        resumed_from: Some(&resumed),
        checkpoint_dir: None,
        checkpoints_written: 0,
        metrics_written: 0,
        previews_written: 0,
    });

    // The checkpoint had already finished all four epochs, so the loop never
    // ran: no loss was observed and no checkpoint was published. Both stay null
    // rather than being filled with a plausible zero.
    assert_eq!(payload["total_loss"], json!(null));
    assert_eq!(payload["checkpoint_dir"], json!(null));
    assert_eq!(payload["samples_seen"], json!(0));
    // The step the run was already at is still reported.
    assert_eq!(payload["global_step"], json!(748));
    assert_eq!(payload["resumed_from"], json!(resumed.display().to_string()));
}

#[test]
fn the_reported_mode_and_variant_use_the_command_line_slugs() {
    for (mode, slug) in [
        (DomainTrainingMode::Baseline, "baseline"),
        (DomainTrainingMode::MouthRoi, "mouth_roi"),
        (DomainTrainingMode::Temporal, "temporal"),
    ] {
        assert_eq!(payload_of(mode, UnetVariant::OriginalUnet)["mode"], json!(slug));
        // For the mode, the request's own spelling is the same string.
        assert_eq!(serde_json::to_value(mode).unwrap(), json!(slug));
    }

    // For the variant it is not: the protocol splits `MobileOneUnet` into
    // `mobile_one_unet`, while the checkpoint manifest, the ONNX export and the
    // `--variant mobileone-unet` flag all say `mobileone`. The payload follows
    // the model, so it cannot contradict the `model_kind` beside it.
    for (variant, slug) in [
        (UnetVariant::OriginalUnet, "original_unet"),
        (UnetVariant::MobileOneUnet, "mobileone_unet"),
    ] {
        let payload = payload_of(DomainTrainingMode::Baseline, variant);
        assert_eq!(payload["variant"], json!(slug));
    }

    let mobileone = MobileOneUnetConfig::parity_micro();
    let configuration = ModelConfiguration::mobileone_unet(&mobileone, false);
    assert_eq!(configuration.model_type(), "mobileone_unet");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-worker --test train_result`
Expected: FAIL to compile -- unresolved imports `TrainSummary` and `train_to_json`.

- [ ] **Step 3: Write minimal implementation**

Create `rust/crates/feathertalk-worker/src/train_result.rs`:

```rust
//! The JSON payload a finished training run returns.

use std::path::Path;

use feathertalk_domain::{TrainingMode, UnetVariant};
use feathertalk_training::CheckpointDescriptor;
use serde_json::{Value, json};

use crate::TRAIN_BACKEND_NAME;

/// What a finished run has to say for itself.
///
/// A struct rather than seventeen positional arguments: eight of them are `u64`
/// counters, so a wrong order would type-check and report the wrong numbers
/// without a word.
#[derive(Debug)]
pub struct TrainSummary<'a> {
    /// The mode the request asked for, echoed in the request's own words.
    pub mode: TrainingMode,
    pub variant: UnetVariant,
    /// Supplies `model_kind`, `architecture_version` and `model_config_sha256`
    /// -- the three values a later resume has to match.
    pub descriptor: &'a CheckpointDescriptor,
    pub frame_count: u64,
    pub epochs_requested: u32,
    /// Epochs this run finished, which is below `epochs_requested` when the
    /// task was cancelled.
    pub epochs_completed: u64,
    pub global_step: u64,
    /// Samples this run saw; a resume starts the count again (design section 8).
    pub samples_seen: u64,
    /// The total loss of the last step, `None` when this run never stepped.
    pub total_loss: Option<f64>,
    pub resumed_from: Option<&'a Path>,
    /// The newest checkpoint this run published, `None` when it published none.
    pub checkpoint_dir: Option<&'a Path>,
    pub checkpoints_written: u64,
    pub metrics_written: u64,
    pub previews_written: u64,
}

/// Shapes the payload the `completed` event of a training task carries.
///
/// The three fields beyond the minimum are deliberate (design section 12):
/// `backend` puts the backend that actually ran into the artifact,
/// `model_config_sha256` is both the audit trail and the value the next resume
/// must match, and `resumed_from` says which checkpoint this run continued.
/// Loss curves stay out -- they would blow up a single-line JSON event, and
/// `outputs/metrics/` holds every step.
pub fn train_to_json(summary: &TrainSummary<'_>) -> Value {
    json!({
        "mode": mode_slug(summary.mode),
        "variant": variant_slug(summary.variant),
        "backend": TRAIN_BACKEND_NAME,
        "model_kind": summary.descriptor.model_kind.as_str(),
        "architecture_version": summary.descriptor.architecture_version.as_str(),
        "model_config_sha256": summary.descriptor.model_config_sha256.as_str(),
        "frame_count": summary.frame_count,
        "epochs_requested": summary.epochs_requested,
        "epochs_completed": summary.epochs_completed,
        "global_step": summary.global_step,
        "samples_seen": summary.samples_seen,
        "total_loss": summary.total_loss,
        "resumed_from": summary.resumed_from.map(path_text),
        "checkpoint_dir": summary.checkpoint_dir.map(path_text),
        "checkpoints_written": summary.checkpoints_written,
        "metrics_written": summary.metrics_written,
        "previews_written": summary.previews_written,
    })
}

/// The request's spelling of the mode.
///
/// Matched exhaustively rather than serialised, so a fourth mode is a compile
/// error here instead of a surprise string in the payload.
fn mode_slug(mode: TrainingMode) -> &'static str {
    match mode {
        TrainingMode::Baseline => "baseline",
        TrainingMode::MouthRoi => "mouth_roi",
        TrainingMode::Temporal => "temporal",
    }
}

/// The request's spelling of the variant.
fn variant_slug(variant: UnetVariant) -> &'static str {
    match variant {
        UnetVariant::OriginalUnet => "original_unet",
        UnetVariant::MobileOneUnet => "mobileone_unet",
    }
}

fn path_text(path: &Path) -> String {
    path.display().to_string()
}
```

Add `mod train_result;` and `pub use train_result::{TrainSummary, train_to_json};` to `lib.rs`.

The MobileOne variant has three spellings in this workspace, so `variant_slug` has to pick one on purpose. The protocol says `mobile_one_unet`, because `feathertalk-domain` derives `rename_all = "snake_case"` and snake_case splits `MobileOneUnet`. The model says `mobileone_unet`, which is what `ModelConfiguration::model_type()` returns and therefore what the checkpoint manifest and the ONNX export carry. The command line says `mobileone-unet` (design section 14). The payload follows the model, so `variant` and the `model_kind` printed directly beneath it agree, and so the field matches the flag the operator typed; the protocol spelling stays where it belongs, in the request frame. The test pins all of it, including `model_type()` itself, so a rename on either side fails here rather than in an artifact.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-worker --test train_result
cargo fmt --all -- --check
cargo clippy -p feathertalk-worker --all-targets -- -D warnings
```

Expected: three tests pass, fmt clean, clippy clean.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/src/train_result.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/tests/train_result.rs
git commit -m "feat(worker): report a training result"
```

---

### Task 10: Run the training loop

**Files:**

- Create: `rust/crates/feathertalk-worker/src/train.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs` (module and export list)
- Test: `rust/crates/feathertalk-worker/tests/support/mod.rs` (create)
- Test: `rust/crates/feathertalk-worker/tests/train.rs` (create)

**Interfaces:**

- Produces: `run_training<M, O, D, E>(plan, dataset, model, optimizer, extractor, device, token, reporter) -> CommandOutcome`.
- Consumes: everything Tasks 5 to 9 built -- the plan, the publish routine, the telemetry writers, the payload and the error mappers.

**Why now:** this is the slice's centre, and it is the part that has to be testable without a real project, real weights or an environment variable. Splitting it from `execute_train` (Task 11) is what makes that possible: the loop takes a plan, a dataset, a model, an optimizer and an extractor as parameters, so a test drives it with a stub dataset, the `parity_micro` model and a constant extractor, and a two-epoch run finishes in seconds.

Design section 14 writes `run_training` as `pub(crate)`. It is `pub` here for the reason Task 6 already recorded: integration tests live in `tests/`, which sees the crate from outside.

- [ ] **Step 1: Write the failing test**

First the shared fixtures. Create `rust/crates/feathertalk-worker/tests/support/mod.rs`:

```rust
#![allow(dead_code)]

use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use burn::{
    module::Module,
    tensor::{Tensor, backend::Backend},
};
use feathertalk_domain::{
    Progress, TaskStage, TrainParams, TrainingMode as DomainTrainingMode, UnetVariant,
};
use feathertalk_export::ModelConfiguration;
use feathertalk_media::CancellationToken;
use feathertalk_models::unet::{OriginalUnet, OriginalUnetConfig};
use feathertalk_training::{
    PerceptualFeatureExtractor, TrainingDataset, TrainingError, TrainingSample,
};
use feathertalk_training_data::{FrameSample, TrainingItem};
use feathertalk_worker::{
    TaskReporter, TrainBackend, TrainDevice, TrainingPaths, TrainingPlan, checkpoint_descriptor,
    training_config,
};

/// A 160x160 forward plus backward through burn's autodiff graph overruns the
/// default 2 MiB libtest stack in a debug build and takes the whole binary down
/// with `STATUS_STACK_OVERFLOW`. `feathertalk-training-run/tests/support/mod.rs`
/// solves it the same way, and Task 4 gives the worker's own execution thread
/// the same stack.
const STEP_STACK_BYTES: usize = 64 * 1024 * 1024;

/// Runs `body` on a thread whose stack is large enough for a training step.
/// Panics travel back through `join`, so failed assertions still fail the test.
pub fn on_step_stack(name: &str, body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name(name.to_owned())
        .stack_size(STEP_STACK_BYTES)
        .spawn(body)
        .expect("the step thread starts")
        .join()
        .expect("the step thread does not panic");
}

/// The perceptual term with the weights taken out: it compares the images
/// themselves, which keeps the loss finite and the test independent of VGG19.
#[derive(Debug, Clone, Copy)]
pub struct IdentityExtractor;

impl<B: Backend> PerceptualFeatureExtractor<B> for IdentityExtractor {
    fn forward(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        image
    }
}

/// Behaves like `IdentityExtractor` until the given call, then poisons the loss.
///
/// One step calls the extractor more than once, so the threshold is counted in
/// calls rather than steps; the tests only need "not the first step".
#[derive(Debug)]
pub struct PoisonedExtractor {
    calls: Cell<usize>,
    poison_from: usize,
}

impl PoisonedExtractor {
    pub fn after(calls: usize) -> Self {
        Self {
            calls: Cell::new(0),
            poison_from: calls,
        }
    }
}

impl<B: Backend> PerceptualFeatureExtractor<B> for PoisonedExtractor {
    fn forward(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        let seen = self.calls.get().saturating_add(1);
        self.calls.set(seen);
        if seen > self.poison_from {
            return image.mul_scalar(f32::NAN);
        }
        image
    }
}

/// The micro model with every parameter already materialised.
///
/// `fork` pushes each `Param` through `val()`; without it a clone would copy the
/// lazy initialiser instead of the weights, and two clones would draw different
/// numbers (burn-core 0.21 `module/param/base.rs`).
pub fn model(device: &TrainDevice) -> OriginalUnet<TrainBackend> {
    OriginalUnetConfig::parity_micro()
        .init::<TrainBackend>(device)
        .fork(device)
}

/// A dataset that synthesises every sample, so the loop can be driven without a
/// locked project on disk. This is what Task 1 opened `FrameSample::new` for.
pub struct StubDataset {
    frame_count: u64,
}

impl StubDataset {
    pub fn new(frame_count: u64) -> Self {
        Self { frame_count }
    }
}

impl TrainingDataset for StubDataset {
    type Item = TrainingItem;

    fn frame_count(&self) -> u64 {
        self.frame_count
    }

    fn load_sample(&self, sample: &TrainingSample) -> Result<Self::Item, TrainingError> {
        Ok(match sample {
            TrainingSample::SingleFrame { target_index, .. } => {
                TrainingItem::SingleFrame(frame(*target_index)?)
            }
            TrainingSample::TemporalPair {
                first_target_index,
                second_target_index,
                ..
            } => TrainingItem::TemporalPair {
                first: frame(*first_target_index)?,
                second: frame(*second_target_index)?,
            },
        })
    }
}

/// Flat planes whose value follows the frame index: enough for a finite loss and
/// for two frames to differ, cheap enough to allocate per sample.
fn frame(index: u64) -> Result<FrameSample, TrainingError> {
    let value = (index % 7) as f32 / 7.0;
    Ok(FrameSample::new(
        vec![value; 6 * 160 * 160],
        vec![value; 16 * 32 * 32],
        vec![value; 3 * 160 * 160],
        vec![1.0; 160 * 160],
    )?)
}

/// Records every event, and can cancel a token once enough have arrived.
pub struct Recorder {
    events: Mutex<Vec<(TaskStage, Option<Progress>)>>,
    cancel_after: Option<(usize, CancellationToken)>,
}

impl Recorder {
    pub fn new() -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            cancel_after: None,
        }
    }

    /// Cancels `token` once `events` events have been reported, which is how a
    /// test interrupts a run at a known step instead of at a known time.
    pub fn cancelling_after(events: usize, token: CancellationToken) -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            cancel_after: Some((events, token)),
        }
    }

    pub fn events(&self) -> Vec<(TaskStage, Option<Progress>)> {
        self.events.lock().expect("the recorder is intact").clone()
    }
}

impl TaskReporter for Recorder {
    fn report(&self, stage: TaskStage, progress: Option<Progress>) {
        let mut events = self.events.lock().expect("the recorder is intact");
        events.push((stage, progress));
        if let Some((limit, token)) = &self.cancel_after {
            if events.len() >= *limit {
                token.cancel();
            }
        }
    }
}

/// The plan a micro run trains under: `parity_micro`, batch size 1, whatever
/// mode and epoch count the test asks for.
pub fn micro_plan(
    project_dir: &Path,
    mode: DomainTrainingMode,
    epochs: u32,
    frame_count: u64,
    resume_from: Option<PathBuf>,
) -> TrainingPlan {
    let params = TrainParams {
        project_dir: project_dir.to_path_buf(),
        mode,
        variant: UnetVariant::OriginalUnet,
        epochs,
        resume: resume_from.is_some(),
    };
    let configuration = ModelConfiguration::original_unet(&OriginalUnetConfig::parity_micro());
    TrainingPlan {
        mode,
        variant: UnetVariant::OriginalUnet,
        epochs_requested: epochs,
        frame_count,
        config: training_config(&params),
        descriptor: checkpoint_descriptor(&configuration).expect("the configuration serialises"),
        paths: TrainingPaths::new(project_dir),
        resume_from,
    }
}
```

Then the loop's own tests. Create `rust/crates/feathertalk-worker/tests/train.rs`:

```rust
mod support;

use std::fs;
use std::path::Path;

use burn::optim::AdamConfig;
use feathertalk_domain::{ErrorCode, Progress, TaskStage, TrainingMode as DomainTrainingMode};
use feathertalk_media::CancellationToken;
use feathertalk_worker::{
    CommandOutcome, TrainDevice, TrainingPaths, latest_checkpoint, run_training,
};
use serde_json::{Value, json};

use support::{
    IdentityExtractor, PoisonedExtractor, Recorder, StubDataset, micro_plan, model, on_step_stack,
};

fn completed(outcome: CommandOutcome) -> Value {
    match outcome {
        CommandOutcome::Completed(Some(payload)) => payload,
        other => panic!("expected a completed outcome, got {other:?}"),
    }
}

fn failed(outcome: CommandOutcome) -> feathertalk_domain::TaskError {
    match outcome {
        CommandOutcome::Failed(error) => error,
        other => panic!("expected a failed outcome, got {other:?}"),
    }
}

fn names(directory: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(directory)
        .expect("the directory is readable")
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// The step number of the first training event.
fn first_step(events: &[(TaskStage, Option<Progress>)]) -> u64 {
    match events.first().expect("at least one event") {
        (TaskStage::Training { step, .. }, _) => *step,
        (other, _) => panic!("expected a training stage, got {other:?}"),
    }
}

#[test]
fn a_baseline_run_trains_publishes_and_reports() {
    on_step_stack("baseline-run", || {
        let root = tempfile::tempdir().unwrap();
        let project = root.path().join("project");
        let device = TrainDevice::default();
        let plan = micro_plan(&project, DomainTrainingMode::Baseline, 2, 4, None);
        let token = CancellationToken::new();
        let reporter = Recorder::new();

        let outcome = run_training(
            &plan,
            StubDataset::new(4),
            model(&device),
            AdamConfig::new().init(),
            &IdentityExtractor,
            &device,
            &token,
            &reporter,
        );

        let payload = completed(outcome);
        assert_eq!(payload["mode"], json!("baseline"));
        assert_eq!(payload["frame_count"], json!(4));
        assert_eq!(payload["epochs_requested"], json!(2));
        assert_eq!(payload["epochs_completed"], json!(2));
        assert_eq!(payload["global_step"], json!(8));
        assert_eq!(payload["samples_seen"], json!(8));
        assert_eq!(payload["resumed_from"], json!(null));
        assert_eq!(payload["checkpoints_written"], json!(2));
        assert_eq!(payload["metrics_written"], json!(2));
        assert_eq!(payload["previews_written"], json!(2));
        let loss = payload["total_loss"].as_f64().expect("a loss was observed");
        assert!(loss.is_finite(), "{loss}");

        // One checkpoint per epoch boundary, named by step, with nothing staged
        // or retired left behind.
        let paths = TrainingPaths::new(&project);
        assert_eq!(
            names(paths.checkpoints()),
            vec!["checkpoint-00000004", "checkpoint-00000008"]
        );
        let latest = paths.checkpoint(8).display().to_string();
        assert_eq!(payload["checkpoint_dir"], json!(latest));
        assert!(paths.metrics(4).is_file());
        assert!(paths.metrics(8).is_file());
        assert!(paths.preview(4).join("manifest.json").is_file());
        assert!(paths.preview(8).join("manifest.json").is_file());

        // One event per step, every one of them a training stage, the last one
        // complete. Epochs are zero-based inside the loader, so the final step
        // belongs to epoch 1 while `epochs_completed` counts 2.
        let events = reporter.events();
        assert_eq!(events.len(), 8);
        assert_eq!(first_step(&events), 1);
        let (stage, progress) = events.last().expect("eight events").clone();
        let expected = Progress {
            completed: 8,
            total: Some(8),
        };
        assert_eq!(progress, Some(expected));
        match stage {
            TaskStage::Training { epoch, step, loss } => {
                assert_eq!((epoch, step), (1, 8));
                assert!(loss.is_finite(), "{loss}");
            }
            other => panic!("expected a training stage, got {other:?}"),
        }
    });
}

#[test]
fn a_cancelled_run_leaves_a_checkpoint_the_resume_continues_from() {
    on_step_stack("cancel-resume", || {
        let root = tempfile::tempdir().unwrap();
        let project = root.path().join("project");
        let device = TrainDevice::default();
        let paths = TrainingPaths::new(&project);
        let plan = micro_plan(&project, DomainTrainingMode::Baseline, 2, 4, None);
        let token = CancellationToken::new();
        // The fourth step ends the first epoch, so the cancellation lands right
        // after a checkpoint was published for that step.
        let reporter = Recorder::cancelling_after(4, token.clone());

        let outcome = run_training(
            &plan,
            StubDataset::new(4),
            model(&device),
            AdamConfig::new().init(),
            &IdentityExtractor,
            &device,
            &token,
            &reporter,
        );

        assert!(matches!(outcome, CommandOutcome::Cancelled), "{outcome:?}");
        // The cancellation republished the same step, so the retire path ran and
        // still left exactly one checkpoint and no staging directories.
        assert_eq!(names(paths.checkpoints()), vec!["checkpoint-00000004"]);

        let resumed_from = latest_checkpoint(&paths)
            .expect("the directory is readable")
            .expect("the cancelled run saved one");
        let plan = micro_plan(
            &project,
            DomainTrainingMode::Baseline,
            2,
            4,
            Some(resumed_from.clone()),
        );
        let token = CancellationToken::new();
        let reporter = Recorder::new();

        let outcome = run_training(
            &plan,
            StubDataset::new(4),
            model(&device),
            AdamConfig::new().init(),
            &IdentityExtractor,
            &device,
            &token,
            &reporter,
        );

        let payload = completed(outcome);
        assert_eq!(payload["global_step"], json!(8));
        assert_eq!(payload["epochs_completed"], json!(2));
        // `restore` zeroes the sample counter, so the payload reports what this
        // run saw rather than the lineage total.
        assert_eq!(payload["samples_seen"], json!(4));
        let resumed = resumed_from.display().to_string();
        assert_eq!(payload["resumed_from"], json!(resumed));
        assert_eq!(payload["checkpoints_written"], json!(1));
        // Only the second epoch ran, and it started at step five.
        let events = reporter.events();
        assert_eq!(events.len(), 4);
        assert_eq!(first_step(&events), 5);
    });
}

#[test]
fn a_checkpoint_from_another_configuration_is_refused() {
    on_step_stack("descriptor-mismatch", || {
        let root = tempfile::tempdir().unwrap();
        let project = root.path().join("project");
        let device = TrainDevice::default();
        let paths = TrainingPaths::new(&project);
        let token = CancellationToken::new();
        let reporter = Recorder::new();
        let plan = micro_plan(&project, DomainTrainingMode::Baseline, 1, 2, None);

        let outcome = run_training(
            &plan,
            StubDataset::new(2),
            model(&device),
            AdamConfig::new().init(),
            &IdentityExtractor,
            &device,
            &token,
            &reporter,
        );
        completed(outcome);

        let checkpoint = latest_checkpoint(&paths)
            .expect("the directory is readable")
            .expect("the first run saved one");
        // Everything else about the plan is identical -- the epoch count too,
        // because `training_config` is compared field by field and a different
        // one would fail the same way for a different reason.
        let mut plan = micro_plan(&project, DomainTrainingMode::Baseline, 1, 2, Some(checkpoint));
        plan.descriptor.model_config_sha256 = "b".repeat(64);

        let outcome = run_training(
            &plan,
            StubDataset::new(2),
            model(&device),
            AdamConfig::new().init(),
            &IdentityExtractor,
            &device,
            &token,
            &reporter,
        );

        let error = failed(outcome);
        assert_eq!(error.code, ErrorCode::ModelIncompatible);
        // The run never started, so the stage is still the preparing one.
        assert_eq!(error.stage, TaskStage::Preparing);
    });
}

#[test]
fn telemetry_that_is_already_there_is_skipped_and_counted() {
    on_step_stack("telemetry-skip", || {
        let root = tempfile::tempdir().unwrap();
        let project = root.path().join("project");
        let device = TrainDevice::default();
        let paths = TrainingPaths::new(&project);
        // What a previous run of the same lineage leaves behind.
        fs::create_dir_all(paths.preview(4)).unwrap();
        fs::create_dir_all(paths.metrics(4).parent().unwrap()).unwrap();
        fs::write(paths.metrics(4), "not even json").unwrap();
        let plan = micro_plan(&project, DomainTrainingMode::Baseline, 1, 4, None);
        let token = CancellationToken::new();
        let reporter = Recorder::new();

        let outcome = run_training(
            &plan,
            StubDataset::new(4),
            model(&device),
            AdamConfig::new().init(),
            &IdentityExtractor,
            &device,
            &token,
            &reporter,
        );

        let payload = completed(outcome);
        // The weights are the product, so the checkpoint still lands; the two
        // diagnostics are skipped and the payload says so.
        assert_eq!(payload["checkpoints_written"], json!(1));
        assert_eq!(payload["metrics_written"], json!(0));
        assert_eq!(payload["previews_written"], json!(0));
        assert!(paths.checkpoint(4).is_dir());
        // Neither leftover was touched.
        assert_eq!(fs::read_to_string(paths.metrics(4)).unwrap(), "not even json");
        assert!(names(&paths.preview(4)).is_empty());
    });
}

#[test]
fn a_step_that_fails_reports_the_step_it_reached() {
    on_step_stack("failing-step", || {
        let root = tempfile::tempdir().unwrap();
        let project = root.path().join("project");
        let device = TrainDevice::default();
        let paths = TrainingPaths::new(&project);
        let plan = micro_plan(&project, DomainTrainingMode::Baseline, 1, 4, None);
        let token = CancellationToken::new();
        let reporter = Recorder::new();
        // A baseline step calls the extractor twice, so the first step goes
        // through and the second one produces a non-finite loss.
        let extractor = PoisonedExtractor::after(2);

        let outcome = run_training(
            &plan,
            StubDataset::new(4),
            model(&device),
            AdamConfig::new().init(),
            &extractor,
            &device,
            &token,
            &reporter,
        );

        let error = failed(outcome);
        assert_eq!(error.code, ErrorCode::MediaInvalid);
        // The whole point of threading the stage through: a run that dies at
        // step two says step two, not "preparing".
        match error.stage {
            TaskStage::Training { epoch, step, .. } => assert_eq!((epoch, step), (0, 1)),
            other => panic!("expected the last training stage, got {other:?}"),
        }
        // The failed run reached no epoch boundary, so it published nothing.
        assert!(!paths.checkpoints().exists());
        assert_eq!(reporter.events().len(), 1);
    });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-worker --test train`
Expected: FAIL to compile -- unresolved import `run_training`.

- [ ] **Step 3: Write minimal implementation**

Create `rust/crates/feathertalk-worker/src/train.rs`:

```rust
//! The training loop and the command that drives it.

use std::path::PathBuf;
use std::time::Instant;

use burn::{module::AutodiffModule, optim::Optimizer};
use feathertalk_domain::{Progress, TaskStage};
use feathertalk_media::CancellationToken;
use feathertalk_models::unet::TrainableTalkingHead;
use feathertalk_training::{
    CheckpointCompatibility, PerceptualFeatureExtractor, TrainingDataset, TrainingError,
    load_training_checkpoint,
};
use feathertalk_training_data::TrainingItem;
use feathertalk_training_run::{StepReport, TrainingRunner, build_preview_artifact};

use crate::{
    CommandOutcome, TRAINING_SEED, TaskReporter, TrainBackend, TrainDevice, TrainSummary,
    TrainingPlan, WORKER_STATE, preview_sample, publish_checkpoint, sample_count, train_to_json,
    training_task_error, write_metrics_unless_present, write_preview_unless_present,
};

/// Trains until the plan's epoch count is reached, the task is cancelled, or a
/// step fails.
///
/// Everything the loop needs arrives as a parameter, which is what lets the
/// tests drive it with a stub dataset and a constant extractor instead of a
/// locked project and half a gigabyte of VGG19 weights.
#[allow(clippy::too_many_arguments)]
pub fn run_training<M, O, D, E>(
    plan: &TrainingPlan,
    dataset: D,
    model: M,
    optimizer: O,
    extractor: &E,
    device: &TrainDevice,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
) -> CommandOutcome
where
    M: TrainableTalkingHead<TrainBackend> + AutodiffModule<TrainBackend> + Clone,
    O: Optimizer<M, TrainBackend> + Clone,
    D: TrainingDataset<Item = TrainingItem>,
    E: PerceptualFeatureExtractor<TrainBackend>,
{
    let mut runner = match build_runner(plan, dataset, model, optimizer, device) {
        Ok(runner) => runner,
        Err(error) => {
            return CommandOutcome::Failed(training_task_error(&error, TaskStage::Preparing));
        }
    };

    // The clock starts here, not at admission: `restore` zeroes `samples_seen`,
    // and the throughput this feeds has to describe the run that is starting.
    let started = Instant::now();
    let total = total_steps(plan);
    let mut published = Published::default();
    let mut stage = TaskStage::Preparing;
    let mut steps: u64 = 0;
    let mut total_loss = None;

    while !runner.is_finished() {
        if token.is_cancelled() {
            // With no step behind it, the checkpoint already on disk is the best
            // one there is and republishing would only copy it.
            if steps > 0 {
                if let Err(error) = publish(&runner, plan, runner.global_step(), &mut published) {
                    return CommandOutcome::Failed(training_task_error(&error, stage));
                }
            }
            return CommandOutcome::Cancelled;
        }

        let report = match runner.step(extractor) {
            Ok(report) => report,
            Err(error) => return CommandOutcome::Failed(training_task_error(&error, stage)),
        };
        steps = steps.saturating_add(1);
        total_loss = Some(report.losses.total);
        stage = TaskStage::Training {
            epoch: u32::try_from(report.epoch).unwrap_or(u32::MAX),
            step: report.global_step,
            loss: report.losses.total,
        };
        reporter.report(stage.clone(), Some(progress(report.global_step, total)));

        // `report.epoch` is the epoch the batch came from, `runner.epoch()` is
        // where the loader stands now, so they differ exactly once per epoch.
        if runner.epoch() > report.epoch {
            let closed = close_epoch(&runner, plan, &report, started, device, &mut published);
            if let Err(error) = closed {
                return CommandOutcome::Failed(training_task_error(&error, stage));
            }
        }
    }

    let summary = TrainSummary {
        mode: plan.mode,
        variant: plan.variant,
        descriptor: &plan.descriptor,
        frame_count: plan.frame_count,
        epochs_requested: plan.epochs_requested,
        epochs_completed: runner.epoch(),
        global_step: runner.global_step(),
        samples_seen: runner.samples_seen(),
        total_loss,
        resumed_from: plan.resume_from.as_deref(),
        checkpoint_dir: published.latest.as_deref(),
        checkpoints_written: published.checkpoints,
        metrics_written: published.metrics,
        previews_written: published.previews,
    };
    CommandOutcome::Completed(Some(train_to_json(&summary)))
}

/// What this run has put on disk.
#[derive(Debug, Default)]
struct Published {
    checkpoints: u64,
    metrics: u64,
    previews: u64,
    latest: Option<PathBuf>,
}

/// Starts a fresh run, or continues from the checkpoint the plan names.
fn build_runner<M, O, D>(
    plan: &TrainingPlan,
    dataset: D,
    model: M,
    optimizer: O,
    device: &TrainDevice,
) -> Result<TrainingRunner<TrainBackend, M, O, D>, TrainingError>
where
    M: TrainableTalkingHead<TrainBackend> + AutodiffModule<TrainBackend> + Clone,
    O: Optimizer<M, TrainBackend> + Clone,
    D: TrainingDataset<Item = TrainingItem>,
{
    let Some(directory) = plan.resume_from.as_deref() else {
        return TrainingRunner::new(
            dataset,
            model,
            optimizer,
            plan.config.clone(),
            TRAINING_SEED,
            *device,
        );
    };

    // The model and the optimizer go in as templates: the loader reads the
    // records into their shapes and hands the restored pair back.
    let expected = CheckpointCompatibility::new(
        plan.descriptor.clone(),
        plan.config.clone(),
        plan.frame_count,
    );
    let restored = load_training_checkpoint::<TrainBackend, M, O>(
        directory,
        &model,
        &optimizer,
        device,
        &expected,
    )?;
    TrainingRunner::restore(dataset, restored, *device)
}

/// Publishes a checkpoint for `global_step` and remembers it as the newest one.
fn publish<M, O, D>(
    runner: &TrainingRunner<TrainBackend, M, O, D>,
    plan: &TrainingPlan,
    global_step: u64,
    published: &mut Published,
) -> Result<(), TrainingError>
where
    M: TrainableTalkingHead<TrainBackend> + AutodiffModule<TrainBackend> + Clone,
    O: Optimizer<M, TrainBackend> + Clone,
    D: TrainingDataset<Item = TrainingItem>,
{
    let checkpoint = publish_checkpoint(&plan.paths, global_step, |staged| {
        runner
            .save_checkpoint(staged, plan.descriptor.clone())
            .map(|_| ())
    })?;
    published.checkpoints = published.checkpoints.saturating_add(1);
    published.latest = Some(checkpoint);
    Ok(())
}

/// What an epoch boundary owes: the checkpoint first, then the two diagnostics.
fn close_epoch<M, O, D>(
    runner: &TrainingRunner<TrainBackend, M, O, D>,
    plan: &TrainingPlan,
    report: &StepReport,
    started: Instant,
    device: &TrainDevice,
    published: &mut Published,
) -> Result<(), TrainingError>
where
    M: TrainableTalkingHead<TrainBackend> + AutodiffModule<TrainBackend> + Clone,
    O: Optimizer<M, TrainBackend> + Clone,
    D: TrainingDataset<Item = TrainingItem>,
{
    publish(runner, plan, report.global_step, published)?;

    // `None` for GPU memory: the CPU backend has none to report (design 4).
    let metrics = runner.metrics(report, started.elapsed(), None, WORKER_STATE)?;
    if write_metrics_unless_present(&plan.paths, report.global_step, &metrics)? {
        published.metrics = published.metrics.saturating_add(1);
    }

    let artifact = build_preview_artifact::<TrainBackend, M, D>(
        runner.model()?,
        runner.dataset(),
        device,
        &preview_sample(plan.frame_count),
        report.epoch,
        report.global_step,
        &plan.descriptor.model_kind,
        &plan.descriptor.model_config_sha256,
        WORKER_STATE,
    )?;
    if write_preview_unless_present(&plan.paths, report.global_step, &artifact)? {
        published.previews = published.previews.saturating_add(1);
    }
    Ok(())
}

/// The number of steps the whole lineage will reach, or `None` when it does not
/// fit in a `u64` -- in which case progress reports what it has done and no
/// total, which is what `Progress.total` being optional is for.
fn total_steps(plan: &TrainingPlan) -> Option<u64> {
    let samples = sample_count(plan.mode, plan.frame_count);
    // `TrainingConfig::validate` rejects a zero batch size, but a divide here
    // must not be the place that finds out.
    let steps_per_epoch = samples.div_ceil(plan.config.batch_size.max(1));
    plan.config.total_epochs.checked_mul(steps_per_epoch)
}

/// A resumed run keeps the lineage's step numbers, so `completed` is clamped
/// rather than trusted: if the loader and the total ever disagree, a total that
/// is reached early beats a progress bar past one hundred percent.
fn progress(global_step: u64, total: Option<u64>) -> Progress {
    Progress {
        completed: match total {
            Some(total) => global_step.min(total),
            None => global_step,
        },
        total,
    }
}
```

Add `mod train;` and `pub use train::run_training;` to `lib.rs`.

Two decisions in there go beyond what the design spells out. A publish that fails on the cancellation path returns `Failed`, not `Cancelled`: the user asked to stop, and reporting a clean stop while the hours of work that were meant to be saved went nowhere would be the one lie this command must not tell. And cancellation before this run's first step publishes nothing at all, because the checkpoint it would write is byte-for-byte the one it just restored.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-worker --test train
cargo fmt --all -- --check
cargo clippy -p feathertalk-worker --all-targets -- -D warnings
```

Expected: five tests pass, fmt clean, clippy clean.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/src/train.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/tests/support/mod.rs rust/crates/feathertalk-worker/tests/train.rs
git commit -m "feat(worker): run the training loop"
```

---

### Task 11: Execute the train command

**Files:**

- Modify: `rust/crates/feathertalk-worker/src/train.rs` (append)
- Modify: `rust/crates/feathertalk-worker/src/commands.rs` (the `Request::Train` arm)
- Modify: `rust/crates/feathertalk-worker/src/lib.rs` (export list)
- Test: `rust/crates/feathertalk-worker/tests/train.rs` (append)

**Interfaces:**

- Produces: `execute_train(&TrainParams, &CancellationToken, &dyn TaskReporter, &TrainingToolchain) -> CommandOutcome` and `check_frame_count(mode, frame_count) -> Result<(), TaskError>`.
- Consumes: `admission::check_project_dir`, `ProjectTrainingDataset::open`, `load_vgg19_package`, `latest_checkpoint`, `checkpoint_descriptor`, `run_training`.

**Why now:** the loop exists and is tested; what is missing is everything that turns a request into its arguments. This is also where the two model variants are chosen, and that choice is a monomorphisation rather than a value, which is why it has to happen in a generic function that takes the model as a type parameter.

The order of admission is the design's, cheapest check first (design section 11): the project directory, the epoch range, the dataset, the frame count, the resume target, then half a gigabyte of VGG19 weights, then one last look at the token.

- [ ] **Step 1: Write the failing test**

Append to `rust/crates/feathertalk-worker/tests/train.rs`, extending the imports with `std::path::PathBuf`, `feathertalk_domain::{TrainParams, UnetVariant}` and `feathertalk_worker::{MAX_EPOCHS, WorkerConfig, check_frame_count, execute_train}`:

```rust
/// A config whose training toolchain points at an empty directory. Every
/// admission test below fails long before the weights would be read.
fn training_config(vgg19_dir: &Path) -> WorkerConfig {
    WorkerConfig::from_values_with_training(
        None,
        None,
        None,
        None,
        None,
        None,
        Some(vgg19_dir.display().to_string()),
    )
}

fn train_params(project_dir: &Path, mode: DomainTrainingMode, epochs: u32) -> TrainParams {
    TrainParams {
        project_dir: project_dir.to_path_buf(),
        mode,
        variant: UnetVariant::OriginalUnet,
        epochs,
        resume: false,
    }
}

/// A directory that gets past `check_project_dir`: absolute, a real directory,
/// with a regular `project.json` inside. Nothing in it is locked.
fn project_shell(root: &Path) -> PathBuf {
    let project = root.join("project");
    fs::create_dir_all(&project).unwrap();
    fs::write(project.join("project.json"), "{}").unwrap();
    project
}

#[test]
fn a_relative_project_directory_is_refused() {
    let root = tempfile::tempdir().unwrap();
    let config = training_config(root.path());
    let training = config.training().expect("the directory exists");
    let params = train_params(Path::new("project"), DomainTrainingMode::Baseline, 1);

    let outcome = execute_train(
        &params,
        &CancellationToken::new(),
        &Recorder::new(),
        training,
    );

    let error = failed(outcome);
    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.summary, "工程目录必须是绝对路径");
}

#[test]
fn the_epoch_count_has_to_be_in_range() {
    let root = tempfile::tempdir().unwrap();
    let project = project_shell(root.path());
    let config = training_config(root.path());
    let training = config.training().expect("the directory exists");

    for epochs in [0, MAX_EPOCHS + 1] {
        let params = train_params(&project, DomainTrainingMode::Baseline, epochs);
        let outcome = execute_train(
            &params,
            &CancellationToken::new(),
            &Recorder::new(),
            training,
        );
        let error = failed(outcome);
        assert_eq!(error.summary, "训练轮数无效", "{epochs}");
        assert_eq!(error.code, ErrorCode::MediaInvalid, "{epochs}");
    }
}

#[test]
fn a_project_without_a_locked_package_is_refused_by_the_dataset() {
    let root = tempfile::tempdir().unwrap();
    let project = project_shell(root.path());
    let config = training_config(root.path());
    let training = config.training().expect("the directory exists");
    let reporter = Recorder::new();
    let params = train_params(&project, DomainTrainingMode::Baseline, 1);

    let outcome = execute_train(&params, &CancellationToken::new(), &reporter, training);

    // `ProjectTrainingDataset::open` is the single place that enforces "extract,
    // extract features, then lock"; the worker does not re-check it.
    let error = failed(outcome);
    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.stage, TaskStage::Preparing);
    // The stage went out before the expensive part of admission started.
    assert_eq!(reporter.events().len(), 1);
    assert_eq!(reporter.events()[0].0, TaskStage::Preparing);
}

#[test]
fn the_temporal_mode_needs_two_frames() {
    // Checked through its own function: reaching it through `execute_train`
    // needs a locked one-frame project, which would mean copying
    // feathertalk-training-run's fixture into this crate for one assertion.
    assert!(check_frame_count(DomainTrainingMode::Baseline, 1).is_ok());
    assert!(check_frame_count(DomainTrainingMode::Temporal, 2).is_ok());

    let error = check_frame_count(DomainTrainingMode::Temporal, 1)
        .expect_err("one frame yields no temporal pair");
    assert_eq!(error.summary, "帧数不足，无法做时序训练");
    assert_eq!(error.code, ErrorCode::MediaInvalid);
}
```

`tests/commands.rs` needs nothing: `an_unsupported_command_is_refused_with_its_slug` already sends a `Request::Train` through `execute_with_runner` with a bare config, and after this task it still fails the same way -- through the new arm's missing-toolchain branch instead of the catch-all.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-worker --test train`
Expected: FAIL to compile -- unresolved imports `check_frame_count` and `execute_train`.

- [ ] **Step 3: Write minimal implementation**

Extend the imports of `rust/crates/feathertalk-worker/src/train.rs` with `burn::optim::AdamConfig`, `feathertalk_domain::{TaskError, TrainParams, TrainingMode as DomainTrainingMode, UnetVariant}`, `feathertalk_export::ModelConfiguration`, `feathertalk_models::unet::{MobileOneUnetConfig, OriginalUnetConfig}`, `feathertalk_training::load_vgg19_package`, `feathertalk_training_data::ProjectTrainingDataset` and, from the crate, `MAX_EPOCHS, TrainingPaths, TrainingToolchain, admission::{check_project_dir, invalid_request}, checkpoint_descriptor, latest_checkpoint, training_config, training_data_task_error`. Then append:

```rust
/// Trains the U-Net of a locked project.
///
/// The toolchain arrives by reference instead of being read from the
/// environment here: `commands.rs` already holds the validated config, and a
/// command that reads its own environment cannot be tested.
pub fn execute_train(
    params: &TrainParams,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    toolchain: &TrainingToolchain,
) -> CommandOutcome {
    // Admission reads the asset manifest, the whole feature file and then half a
    // gigabyte of VGG19 weights, so the stage goes out before any of it.
    reporter.report(TaskStage::Preparing, None);
    if let Err(error) = check_project_dir(&params.project_dir) {
        return CommandOutcome::Failed(error);
    }
    if let Err(error) = check_epochs(params.epochs) {
        return CommandOutcome::Failed(error);
    }

    // The variant is a type rather than a value, so each arm monomorphises the
    // whole run for one model. This is the only place that branches on it.
    match params.variant {
        UnetVariant::OriginalUnet => {
            let configuration = OriginalUnetConfig::production();
            let described = ModelConfiguration::original_unet(&configuration);
            start(params, token, reporter, toolchain, described, |device| {
                configuration.init::<TrainBackend>(device)
            })
        }
        UnetVariant::MobileOneUnet => {
            let configuration = MobileOneUnetConfig::production();
            // Not reparameterized: training needs the multi-branch graph, and
            // fusing the branches is an export-time step.
            let described = ModelConfiguration::mobileone_unet(&configuration, false);
            start(params, token, reporter, toolchain, described, |device| {
                configuration.init::<TrainBackend>(device)
            })
        }
    }
}

/// The rest of the command, once the model type is known.
fn start<M, F>(
    params: &TrainParams,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    toolchain: &TrainingToolchain,
    configuration: ModelConfiguration,
    init: F,
) -> CommandOutcome
where
    M: TrainableTalkingHead<TrainBackend> + AutodiffModule<TrainBackend> + Clone,
    F: FnOnce(&TrainDevice) -> M,
{
    // Deliberately never named: writing out `ProjectTrainingDataset<JpegFrameReader>`
    // would pull `feathertalk-inference` into this crate's dependencies for the
    // sake of one type annotation.
    let dataset = match ProjectTrainingDataset::open(&params.project_dir) {
        Ok(dataset) => dataset,
        Err(error) => return CommandOutcome::Failed(training_data_task_error(&error)),
    };
    let frame_count = dataset.frame_count();
    if let Err(error) = check_frame_count(params.mode, frame_count) {
        return CommandOutcome::Failed(error);
    }

    let paths = TrainingPaths::new(&params.project_dir);
    let found = match latest_checkpoint(&paths) {
        Ok(found) => found,
        Err(error) => return failed_preparing(&error),
    };
    if params.resume && found.is_none() {
        return CommandOutcome::Failed(invalid_request(
            "未找到可续训的检查点",
            format!("no checkpoint under {}", paths.checkpoints().display()),
        ));
    }
    let descriptor = match checkpoint_descriptor(&configuration) {
        Ok(descriptor) => descriptor,
        Err(error) => return failed_preparing(&error),
    };
    let plan = TrainingPlan {
        mode: params.mode,
        variant: params.variant,
        epochs_requested: params.epochs,
        frame_count,
        config: training_config(params),
        descriptor,
        paths,
        // Without `--resume`, a checkpoint on disk is not continued: the run
        // starts from fresh weights and republishes over the old names.
        resume_from: if params.resume { found } else { None },
    };

    let device = TrainDevice::default();
    let extractor = match load_vgg19_package::<TrainBackend>(toolchain.vgg19_dir(), &device) {
        Ok(extractor) => extractor,
        Err(error) => return failed_preparing(&error),
    };
    // Loading the weights took seconds; the caller may have given up meanwhile.
    if token.is_cancelled() {
        return CommandOutcome::Cancelled;
    }

    let optimizer = AdamConfig::new().init::<TrainBackend, M>();
    run_training(
        &plan,
        dataset,
        init(&device),
        optimizer,
        &extractor,
        &device,
        token,
        reporter,
    )
}

/// `TrainingConfig::validate` also refuses zero, later and with a message about
/// a config the operator never wrote.
fn check_epochs(epochs: u32) -> Result<(), TaskError> {
    if epochs == 0 || epochs > MAX_EPOCHS {
        return Err(invalid_request(
            "训练轮数无效",
            format!("epochs must be between 1 and {MAX_EPOCHS}, got {epochs}"),
        ));
    }
    Ok(())
}

/// The temporal mode pairs each frame with its successor, so one frame yields no
/// samples at all. `DataLoaderConfig::validate` refuses it too, in terms of a
/// sample count rather than of frames.
pub fn check_frame_count(mode: DomainTrainingMode, frame_count: u64) -> Result<(), TaskError> {
    if matches!(mode, DomainTrainingMode::Temporal) && frame_count < 2 {
        return Err(invalid_request(
            "帧数不足，无法做时序训练",
            format!("temporal training needs at least 2 frames, got {frame_count}"),
        ));
    }
    Ok(())
}

/// Everything that goes wrong before the first step is a preparing failure.
fn failed_preparing(error: &TrainingError) -> CommandOutcome {
    CommandOutcome::Failed(training_task_error(error, TaskStage::Preparing))
}
```

Then the dispatch, inserted in `commands.rs` before the surviving `other =>` arm:

```rust
        Request::Train(params) => {
            let Some(training) = config.training() else {
                return CommandOutcome::Failed(unsupported(request.kind()));
            };
            execute_train(params, token, reporter, training)
        }
```

Export `check_frame_count` and `execute_train` from `lib.rs`.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-worker
cargo fmt --all -- --check
cargo clippy -p feathertalk-worker --all-targets -- -D warnings
```

Expected: the whole crate passes -- nine tests in `train`, and `commands.rs`'s existing train dispatch test still failing the request for the same reason through the new arm.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/src/train.rs rust/crates/feathertalk-worker/src/commands.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/tests/train.rs
git commit -m "feat(worker): execute the train command"
```

---

### Task 12: Add the train subcommand

**Files:**

- Modify: `rust/crates/feathertalk-cli/src/cli.rs`
- Modify: `rust/crates/feathertalk-cli/src/run.rs`
- Modify: `rust/crates/feathertalk-cli/src/render.rs`
- Test: `rust/crates/feathertalk-cli/tests/cli.rs` (append)
- Test: `rust/crates/feathertalk-cli/src/run.rs` (the inline `mod tests`)

**Interfaces:**

- Produces: `feathertalk train <PROJECT_DIR> --mode <MODE> --variant <VARIANT> --epochs <N> [--resume]`.
- Produces: `Command::Train`, the mirror enums `TrainMode` and `TrainVariant`, and their two mapping functions.

**Why now:** the worker can train; nothing can ask it to. The CLI is also the layer that turns an unsupported command into an actionable sentence, which for training means naming `FEATHERTALK_WORKER_VGG19_DIR`.

The two enums are mirrored rather than reused because `ValueEnum` has to be derived on a local type, and adding clap to `feathertalk-domain` to parse arguments would be the wrong trade (design section 14). The mirror also buys `--help` text and clap's own "invalid value, possible values are" message for free. Clap's kebab values and the protocol's snake slugs differ in shape, and `run.rs` is the single conversion point.

- [ ] **Step 1: Write the failing test**

Append to the inline `mod tests` of `rust/crates/feathertalk-cli/src/run.rs`:

```rust
    #[test]
    fn train_refuses_an_empty_project_directory() {
        let error = build_request(&Command::Train {
            project_dir: PathBuf::new(),
            mode: TrainMode::Baseline,
            variant: TrainVariant::OriginalUnet,
            epochs: 1,
            resume: false,
        })
        .expect_err("an empty project directory is refused");
        assert_eq!(error, "工程目录不能为空。");
    }

    #[test]
    fn train_carries_every_flag_into_the_request() {
        let request = build_request(&Command::Train {
            project_dir: PathBuf::from("project"),
            mode: TrainMode::Temporal,
            variant: TrainVariant::MobileOneUnet,
            epochs: 3,
            resume: true,
        })
        .expect("the arguments are accepted")
        .expect("train needs a task");
        let Request::Train(params) = request else {
            panic!("train must build a Train request");
        };
        assert_eq!(params.project_dir, PathBuf::from("project"));
        assert_eq!(params.mode, TrainingMode::Temporal);
        assert_eq!(params.variant, UnetVariant::MobileOneUnet);
        assert_eq!(params.epochs, 3);
        assert!(params.resume);
    }

    #[test]
    fn an_out_of_range_epoch_count_is_left_to_the_worker() {
        // The CLI does not know `MAX_EPOCHS`, and two answers that can disagree
        // are worse than one; the worker rejects it with a Chinese summary.
        let request = build_request(&Command::Train {
            project_dir: PathBuf::from("project"),
            mode: TrainMode::Baseline,
            variant: TrainVariant::OriginalUnet,
            epochs: 0,
            resume: false,
        })
        .expect("zero epochs still builds a request");
        assert!(request.is_some());
    }

    #[test]
    fn every_mirrored_value_maps_onto_the_domain() {
        assert_eq!(training_mode(TrainMode::Baseline), TrainingMode::Baseline);
        assert_eq!(training_mode(TrainMode::MouthRoi), TrainingMode::MouthRoi);
        assert_eq!(training_mode(TrainMode::Temporal), TrainingMode::Temporal);
        assert_eq!(
            unet_variant(TrainVariant::OriginalUnet),
            UnetVariant::OriginalUnet
        );
        assert_eq!(
            unet_variant(TrainVariant::MobileOneUnet),
            UnetVariant::MobileOneUnet
        );
    }
```

And append to `rust/crates/feathertalk-cli/tests/cli.rs`:

```rust
#[test]
fn an_unsupported_train_names_the_vgg19_variable() {
    // The fake worker advertises `validate_project` alone, so the client's
    // capability gate answers before any task starts.
    let output = run("only-validate", &["train", "p", "--epochs", "1"]);
    assert_eq!(code(&output), 3);
    let text = stderr(&output);
    assert!(text.contains("train"), "{text}");
    assert!(text.contains("FEATHERTALK_WORKER_VGG19_DIR"), "{text}");
}

#[test]
fn an_unknown_training_mode_is_refused_with_the_choices() {
    let output = run(
        "ready-complete",
        &["train", "p", "--epochs", "1", "--mode", "mouth"],
    );
    assert_eq!(code(&output), 3, "a usage error is a session error");
    // Clap owns this message, which is exactly why the enums are mirrored.
    let text = stderr(&output);
    assert!(text.contains("mouth-roi"), "{text}");
    assert!(text.contains("temporal"), "{text}");
}

#[test]
fn train_needs_an_epoch_count() {
    let output = run("ready-complete", &["train", "p"]);
    assert_eq!(code(&output), 3);
    assert!(stderr(&output).contains("--epochs"), "{}", stderr(&output));
}
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-cli --lib
```

Expected: FAIL to compile -- no variant `Train` on `Command`, no `TrainMode`, no `TrainVariant`, no `training_mode`, no `unet_variant`.

- [ ] **Step 3: Write minimal implementation**

In `rust/crates/feathertalk-cli/src/cli.rs`, extend the clap import to `use clap::{Parser, Subcommand, ValueEnum};`, add `train` to the doc comment that lists the kebab-cased commands, add the subcommand after `LockAssetPackage`:

```rust
    /// 训练 U-Net：读取已加锁的工程，按轮数训练并写出检查点与诊断产物
    Train {
        /// 工程目录
        project_dir: PathBuf,
        /// 训练模式
        #[arg(long, value_enum, default_value_t = TrainMode::Baseline)]
        mode: TrainMode,
        /// 模型变体
        #[arg(long, value_enum, default_value_t = TrainVariant::OriginalUnet)]
        variant: TrainVariant,
        /// 训练轮数
        #[arg(long)]
        epochs: u32,
        /// 从最新检查点继续训练，没有检查点时报错
        #[arg(long)]
        resume: bool,
    },
```

and the two mirror enums after `Command`:

```rust
/// The training modes, mirrored from `feathertalk-domain` because `ValueEnum`
/// has to be derived on a local type. `run.rs` maps them onto the domain enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum TrainMode {
    /// 基线：整幅 L1 加感知损失
    Baseline,
    /// 基线之上加嘴部 ROI 权重
    MouthRoi,
    /// 嘴部 ROI 之上加相邻帧的时序一致性
    Temporal,
}

/// The U-Net variants, mirrored for the same reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum TrainVariant {
    /// 原版 U-Net
    OriginalUnet,
    /// MobileOne U-Net
    ///
    /// Spelled the way the model is spelled everywhere else -- the checkpoint
    /// manifest and the ONNX export both say `mobileone_unet` -- rather than the
    /// `mobile-one-unet` clap would derive from the variant name.
    #[value(name = "mobileone-unet")]
    MobileOneUnet,
}
```

In `rust/crates/feathertalk-cli/src/run.rs`, extend the `feathertalk_domain` import with `TrainParams, TrainingMode, UnetVariant`, the `crate::cli` import with `TrainMode, TrainVariant`, add the `build_request` arm:

```rust
        Command::Train {
            project_dir,
            mode,
            variant,
            epochs,
            resume,
        } => {
            reject_empty(project_dir, "工程目录")?;
            // The epoch range is the worker's judgement, like every path here.
            Ok(Some(Request::Train(TrainParams {
                project_dir: project_dir.clone(),
                mode: training_mode(*mode),
                variant: unet_variant(*variant),
                epochs: *epochs,
                resume: *resume,
            })))
        }
```

and the two mappings next to `reject_empty`:

```rust
/// The only place where clap's kebab-cased values meet the domain enums.
fn training_mode(mode: TrainMode) -> TrainingMode {
    match mode {
        TrainMode::Baseline => TrainingMode::Baseline,
        TrainMode::MouthRoi => TrainingMode::MouthRoi,
        TrainMode::Temporal => TrainingMode::Temporal,
    }
}

fn unet_variant(variant: TrainVariant) -> UnetVariant {
    match variant {
        TrainVariant::OriginalUnet => UnetVariant::OriginalUnet,
        TrainVariant::MobileOneUnet => UnetVariant::MobileOneUnet,
    }
}
```

In `rust/crates/feathertalk-cli/src/render.rs`, add the constant next to the other worker variables:

```rust
/// The worker's variable for the VGG19 package directory, a literal for the same
/// reason as the others: `feathertalk-worker`'s `ENV_VGG19_DIR` is the source of
/// truth for this name.
const ENV_WORKER_VGG19_DIR: &str = "FEATHERTALK_WORKER_VGG19_DIR";
```

and the branch at the end of `render_client_error`'s `UnsupportedCommand` chain:

```rust
            } else if *requested == "train" {
                text.push_str(&format!(
                    "\n{requested} 需要 VGG19 感知损失模型包。请用环境变量 \
                     {ENV_WORKER_VGG19_DIR} 指定模型包目录的完整路径。"
                ));
            }
```

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-cli --lib
cargo test -p feathertalk-cli --test cli
cargo fmt --all -- --check
cargo clippy -p feathertalk-cli --all-targets -- -D warnings
```

Expected: four new inline tests and three new CLI tests pass, fmt clean, clippy clean.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-cli/src/cli.rs rust/crates/feathertalk-cli/src/run.rs rust/crates/feathertalk-cli/src/render.rs rust/crates/feathertalk-cli/tests/cli.rs
git commit -m "feat(cli): add the train subcommand"
```

---

### Task 13: Train a locked project end to end

**Files:**

- Test: `rust/crates/feathertalk-cli/tests/real_worker.rs` (append one test, change two helpers)

**Interfaces:**

- Produces: `a_real_project_is_trained_end_to_end`, the gated release test for the whole slice.
- Changes: `cut_audio` takes the duration to cut, and `write_frame_fixtures` writes the landmarks the frame fixture actually records.

**Why now:** every task before this one proved a piece against a stub -- a stub dataset, `parity_micro` weights, a constant extractor. This is the only test that puts the real dataset, the production U-Net, real VGG19 weights and a real feature file into one run, so it is the only one that can catch what a unit test cannot see: a path assembled from the wrong root, an environment variable spelled two ways, a plan whose sample count does not match the loader the dataset opens.

It is gated like the four real-worker tests before it, and for their reason: this repository ships neither ffmpeg nor any model package. Without `FEATHERTALK_WORKER_FFMPEG`, `FEATHERTALK_WORKER_HUBERT_DIR`, `FEATHERTALK_WORKER_VGG19_DIR` and `demo/feathertalk_demo_latest_188.mp4` it prints why it skipped and returns, and `FEATHERTALK_REQUIRE_E2E=1` is what turns a missing worker binary into a failure instead of a skip.

Four frames, one epoch, and both numbers are forced. Every step is a 160x160 forward and backward through the production U-Net plus two VGG19 passes on a CPU backend, so four steps are minutes where 49 would be an hour. Four is also the fewest frames the lock accepts here: `MAX_TOKEN_FIT_DELTA` lets a package sit 50 tokens away from `2 * frame_count`, and the two seconds of audio the other tests cut extract 98 tokens, 90 away from the 8 that four frames need. The audio is cut to the frame count instead.

Two helpers move before the test can pass:

- `cut_audio` cuts a fixed two seconds. The duration becomes a parameter: `"2"` at the two existing callers, `TRAINED_AUDIO_SECONDS` here.
- `write_frame_fixtures` writes `i i` for every landmark. The lock never reads a coordinate, but the dataset reads all 110 of them: points 1 and 31 are the face box's x range, point 52 is its top, and 90..110 are the mouth. A diagonal puts the face box in a 30x30 corner of the frame and projects the mouth outside the inner crop, so the run would train on a corner of the wall. The committed `demo_frame_v1` fixture already records the real 110 points for the frame this helper copies, so it reads them out of `fixture.json` instead of inventing them.

- [ ] **Step 1: Write the failing test**

Append to `rust/crates/feathertalk-cli/tests/real_worker.rs`:

```rust
/// The frames the training test locks, and half the tokens
/// `TRAINED_AUDIO_SECONDS` extracts: `2 * frame_count` exactly, so the lock fits
/// the feature file without trimming a token.
const TRAINED_FRAME_COUNT: u64 = 4;

/// 0.18 s at 16 kHz is 2 880 samples, which the 400-sample kernel and the
/// 320-sample stride turn into `(2 880 - 80) / 320` = 8 FeatherHuBERT frames and
/// so into 8 tokens. Every cut from 2 640 to 3 279 samples yields the same 8,
/// which is the slack this leaves a resampler that hands over a few samples more
/// or fewer than the arithmetic above.
const TRAINED_AUDIO_SECONDS: &str = "0.18";

/// A manifest `validate_project_dir` accepts. `extract-features` and
/// `lock-asset-package` only need the file to exist, which is why the tests
/// above write `{}`; training opens the project properly and reads every field.
const PROJECT_MANIFEST: &str = r#"{
  "schema_version": 1,
  "project_id": "demo",
  "display_name": "Demo",
  "asset_package": "assets/assets.json",
  "default_model": "original_unet",
  "task_history": []
}"#;

/// The whole training command, over a package this test locks itself, and only
/// when the real ffmpeg, a built FeatherHuBERT package, a built VGG19 package
/// and the demo clip are all present. Anything missing is a skip rather than a
/// failure, for the reason the tests above give.
#[test]
fn a_real_project_is_trained_end_to_end() {
    let Some(worker) = worker_or_skip("a_real_project_is_trained_end_to_end") else {
        return;
    };
    let (Some(ffmpeg), Some(hubert), Some(vgg19), Some(demo)) = (
        real_tool("FFMPEG"),
        real_dir("HUBERT_DIR"),
        real_dir("VGG19_DIR"),
        demo_clip(),
    ) else {
        println!(
            "skipping a_real_project_is_trained_end_to_end: it needs \
             FEATHERTALK_WORKER_FFMPEG, FEATHERTALK_WORKER_HUBERT_DIR, \
             FEATHERTALK_WORKER_VGG19_DIR, and \
             demo/feathertalk_demo_latest_188.mp4"
        );
        return;
    };
    let project = TempDir::new().expect("a temporary directory is available");
    let assets = project.path().join("assets");
    for directory in ["frames", "landmarks"] {
        std::fs::create_dir_all(assets.join(directory)).expect("the assets tree is writable");
    }
    std::fs::write(project.path().join("project.json"), PROJECT_MANIFEST)
        .expect("the temporary manifest is writable");
    let audio = assets.join("audio_16k_mono.wav");
    cut_audio(&ffmpeg, &demo, &audio, TRAINED_AUDIO_SECONDS);
    // Nothing reads the video: the lock stats it and so does the project
    // validation the dataset runs. It is cut with the helper the tests above use.
    cut_one_second(&ffmpeg, &demo, &assets.join("video_25fps.mp4"));

    let project_arg = project.path().to_string_lossy().into_owned();
    let audio_arg = audio.to_string_lossy().into_owned();
    let hubert_arg = hubert.to_string_lossy().into_owned();
    let vgg19_arg = vgg19.to_string_lossy().into_owned();
    let hubert_env = [("FEATHERTALK_WORKER_HUBERT_DIR", hubert_arg.as_str())];
    let output = run(
        &worker,
        &["extract-features", &project_arg, &audio_arg],
        &hubert_env,
    );
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));
    let extracted: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("stdout is exactly one JSON document");
    // The audio cut is pinned here rather than left to a comment: if a resampler
    // ever moves the sample count out of the window above, this is the assertion
    // that names the number.
    assert_eq!(extracted["tokens"], 8);
    assert_eq!(extracted["frame_count"], TRAINED_FRAME_COUNT);

    write_frame_fixtures(&assets, TRAINED_FRAME_COUNT);
    let output = run(&worker, &["lock-asset-package", &project_arg], &hubert_env);
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));
    let locked: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("stdout is exactly one JSON document");
    assert_eq!(locked["frame_count"], TRAINED_FRAME_COUNT);
    assert_eq!(locked["token_adjustment"], 0);

    let output = run(
        &worker,
        &["train", &project_arg, "--mode", "baseline", "--epochs", "1"],
        &[("FEATHERTALK_WORKER_VGG19_DIR", vgg19_arg.as_str())],
    );
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));
    let result: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("stdout is exactly one JSON document");
    assert_eq!(result["mode"], "baseline");
    assert_eq!(result["variant"], "original_unet");
    assert_eq!(result["model_kind"], "original_unet");
    assert_eq!(result["backend"], "ndarray-cpu");
    assert_eq!(result["model_config_sha256"].as_str().unwrap().len(), 64);
    assert_eq!(result["frame_count"], TRAINED_FRAME_COUNT);
    assert_eq!(result["epochs_requested"], 1);
    assert_eq!(result["epochs_completed"], 1);
    // One step per frame, and the one epoch boundary is the only publish point.
    assert_eq!(result["global_step"], TRAINED_FRAME_COUNT);
    assert_eq!(result["samples_seen"], TRAINED_FRAME_COUNT);
    assert_eq!(result["resumed_from"], serde_json::Value::Null);
    assert_eq!(result["checkpoints_written"], 1);
    assert_eq!(result["metrics_written"], 1);
    assert_eq!(result["previews_written"], 1);
    let loss = result["total_loss"]
        .as_f64()
        .expect("a finished run reports the loss it last saw");
    assert!(loss.is_finite() && loss >= 0.0, "{loss}");

    // The three artifacts a publish writes, every one of them named by the same
    // global step.
    let checkpoint = project.path().join("models/unet/checkpoint-00000004");
    assert_eq!(result["checkpoint_dir"], checkpoint.display().to_string());
    assert!(checkpoint.join("manifest.json").is_file());
    let outputs = project.path().join("outputs");
    assert!(outputs.join("metrics/step-00000004.json").is_file());
    assert!(outputs.join("preview/step-00000004/manifest.json").is_file());

    let narration = stderr(&output);
    assert!(narration.contains("正在训练"), "{narration}");
    assert!(narration.contains("进度 4/4"), "{narration}");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-cli --test real_worker`
Expected: FAIL to compile -- `cut_audio` takes 3 arguments but 4 were supplied.

- [ ] **Step 3: Write minimal implementation**

Give `cut_audio` its duration, and update its doc comment and its two existing callers, which pass `"2"`:

```rust
/// `seconds` of the demo video's audio, decoded into the one shape the reader
/// admits -- 16 kHz, mono, 16-bit PCM -- and written under the name the pipeline
/// expects. No offset is needed: unlike a video frame, whose face score decides
/// whether it is usable, any cut of this clip's audio extracts alike.
fn cut_audio(ffmpeg: &Path, demo: &Path, audio: &Path, seconds: &str) {
    let output = Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-y"])
        .arg("-i")
        .arg(demo)
        .args(["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-t"])
        .arg(seconds)
        .arg(audio)
        .output()
        .expect("ffmpeg runs");
    assert!(
        output.status.success(),
        "ffmpeg could not cut the audio: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
```

Then read the landmarks out of the fixture instead of drawing a diagonal. In `write_frame_fixtures`, the directory is now named once and the landmark loop is replaced by one call; everything below it, including the placeholder digests its doc comment already explains, stays as it is:

```rust
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../feathertalk-frame-adapters/tests/fixtures/demo_frame_v1");
    let frame_bytes =
        std::fs::read(fixture.join("frame.jpg")).expect("the committed frame fixture is readable");
    let landmarks = fixture_landmarks(&fixture);
```

```rust
/// The 110 landmarks the fixture records for `frame.jpg`, in the `.lms` shape the
/// frame pipeline writes: one `x y` line per point, in detection order. The
/// coordinates have to be the real ones -- points 1, 31 and 52 are the face box a
/// training run crops to, and 90..110 are the mouth it projects into the inner
/// crop.
fn fixture_landmarks(fixture: &Path) -> String {
    let manifest = std::fs::read_to_string(fixture.join("fixture.json"))
        .expect("the fixture manifest is readable");
    let manifest: serde_json::Value =
        serde_json::from_str(&manifest).expect("the fixture manifest is JSON");
    let points = manifest["frames"][0]["sharp"]["landmarks"]
        .as_array()
        .expect("the sharp frame records its landmarks");
    let mut text = String::new();
    for point in points {
        let x = point[0].as_f64().expect("a landmark x is a number");
        let y = point[1].as_f64().expect("a landmark y is a number");
        text.push_str(&format!("{x} {y}\n"));
    }
    text
}
```

- [ ] **Step 4: Run test to verify it passes**

The VGG19 package is built once, from the official source file, the way `docs/WEIGHTS.md` prescribes: `cargo run -p feathertalk-vgg19-package -- --source <vgg19-dcbb9e9d.pth> --licenses <reviewed LICENSES.json> --destination <package dir>`.

```powershell
cd E:\workspace\github\FeatherTalk\rust
$env:FEATHERTALK_REQUIRE_E2E = "1"
$env:FEATHERTALK_WORKER_FFMPEG = "D:\environment\ffmpeg\bin\ffmpeg.exe"
$env:FEATHERTALK_WORKER_HUBERT_DIR = "$env:TEMP\ft_hubert_e2e\package"
$env:FEATHERTALK_WORKER_VGG19_DIR = "$env:TEMP\ft_vgg19_e2e\package"
cargo test --release -p feathertalk-cli --test real_worker
cargo fmt --all -- --check
cargo clippy -p feathertalk-cli --all-targets -- -D warnings
```

Expected: all ten real-worker tests pass, the new one included, with a finite loss, one checkpoint, one metrics file and one preview on disk. Release matters: a debug build of the production U-Net turns four steps into an afternoon.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-cli/tests/real_worker.rs
git commit -m "test(cli): train a locked project end to end"
```

---
