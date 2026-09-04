# Render Worker Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve `Request::Render` in `feathertalk-worker` and expose it as `feathertalk render`, so a locked project plus one of its own training checkpoints becomes an mp4 that FFmpeg muxed with the requested audio track -- with a progress event per output frame, a cancellation that leaves no file behind, and a result payload that names the weights the video came from.

**Architecture:** `feathertalk-inference::execute_offline_render` already owns the frame loop: it reads the features, builds a `RenderPlan`, runs one forward pass per output frame, writes BGR24 into FFmpeg's stdin and publishes the result with a same-directory rename. `feathertalk-training` owns checkpoints but only as a training pair. This slice writes the layer between them and the wire. `feathertalk-training` gains a model-only reader, because inference has no optimizer to restore. `rendering.rs` turns `RenderParams` plus the locked manifest into a `RenderJob`; `render.rs` is the command body -- admission, variant dispatch, weight loading, the observing sink that reports progress and honours cancellation, error mapping; `render_result.rs` shapes the payload. The wire protocol does not change: `TaskKind::Render`, `RenderParams`, `Request::Render` and `TaskStage::Rendering` were committed slices ago, and the CLI already knows the Chinese label for the stage.

**Tech Stack:** Rust edition 2024 (rust-version 1.94), burn `=0.21.0` -- `NdArray<f32>` for the forward passes and `Autodiff<NdArray<f32>>` to read a record written by those exact types -- clap 4 for the CLI, `tempfile` for fixtures. Real FFmpeg appears only in the gated end-to-end test.

**Design:** `docs/superpowers/specs/2026-09-04-render-worker-command-design.md`

## Global Constraints

- Run every `cargo`, `rustfmt` and `clippy` command from `E:\workspace\github\FeatherTalk\rust`. Run every `git` command from `E:\workspace\github\FeatherTalk`.
- The wire protocol is frozen. `feathertalk-domain` is not edited at all: no new params field, no new stage, no new error code, no new capability flag.
- Exactly four changes land outside `feathertalk-worker` and `feathertalk-cli`: `InferenceError::Cancelled` and the sink traits' lifetime and auto-trait bounds in `feathertalk-inference` (Task 1), and `read_training_checkpoint` plus `load_training_checkpoint_model` in `feathertalk-training` (Task 2). Everything else is consumed as it is.
- No new environment variable. Rendering shells out to ffmpeg and nothing else; the frames, landmarks and audio features are already inside the locked project, so SCRFD, PFLD, FeatherHuBERT and VGG19 are all irrelevant here. `MediaToolchain::new` requires both tools, so the user-facing hint names `FEATHERTALK_WORKER_FFPROBE` and `FEATHERTALK_WORKER_FFMPEG`.
- CPU only. `type RenderBackend = CpuBackend` (`NdArray<f32>`), `RENDER_BACKEND_NAME = "ndarray-cpu"`, `capabilities.wgpu_training` stays `false`, `backends` stays `[Cpu]`.
- Features come from the project's own `assets/features/feather_hubert.f32`; the request's `audio` is only the track FFmpeg muxes into the output (design section 2). Rendering against arbitrary driving audio is out of scope.
- The three asset locations are worker-side constants in `rendering.rs`, relative to the project root: `assets/frames`, `assets/landmarks`, `assets/features/feather_hubert.f32`. They are written as component arrays and joined one component at a time, so a Windows path comes out with native separators -- `Path::join("assets/frames")` keeps the forward slash and the payload would then disagree with a path the test builds.
- `max_output_frames` is `Option<u64>` on the wire and `Option<usize>` in inference. `Some(0)` is rejected and a value that does not fit in `usize` is rejected. Never truncate: the protocol design says so in writing.
- Progress total is `min(frame_count, max_output_frames)` from the locked manifest, never a re-read of the feature file. `Progress.completed` is clamped to the total the way `train.rs::progress` clamps it.
- Chinese only inside user-facing string literals (task-error summaries, CLI help, rejection reasons). Identifiers, comments, doc comments and error `detail` text are English.
- No `unwrap`, `expect`, `panic!`, panicking index or panicking arithmetic outside `#[cfg(test)]` and `tests/`. Frame counters use `saturating_add`; every `u64` to `usize` conversion is a `try_from`.
- Never `.clone()` a `Copy` device -- clippy's `clone_on_copy` is an error under `-D warnings`. Prefer `ok_or_else` over `ok_or`. `TaskStage` is not `Copy`, so a stage that is reported and then reused is cloned.
- rustfmt defaults apply (`max_width` 100, `fn_call_width` 60, `chain_width` 60). Every code block below is written in rustfmt's output shape; rustfmt owns the order of names inside `use` braces, so run `cargo fmt --all` after editing an import list rather than hand-sorting it.
- Every new unit test runs offline: `NdArray<f32>`, a `parity_micro` model, a stub frame reader, an in-memory sink. No test spawns ffmpeg, reads an environment variable or loads real weights. Only `feathertalk-cli/tests/real_worker.rs` (Task 10) does, and it skips unless its variables are set.
- Stage explicit paths only. Never stage anything under `demo/`. One commit per task with the exact message given. Do not push.

## Two Deviations From The Design

Both are recorded here because the design text does not mention them, and both are forced by the type system rather than by taste.

1. `RawVideoSinkFactory::start` returns `Box<dyn RawVideoSink + '_>` instead of `Box<dyn RawVideoSink>`, and both sink traits lose their auto-trait supertraits -- `RawVideoSink: Send` and `RawVideoSinkFactory: Send + Sync` become plain traits (Task 1). Design section 6 has the observing sink borrow the reporter and the cancellation token, and both halves of the current declaration forbid that: `Box<dyn Trait>` in return position means `Box<dyn Trait + 'static>`, and a `TaskReporter` is neither `Send` nor `Sync`, so a sink holding `&dyn TaskReporter` can never be `Send`. Nothing needs the bounds: `execute_offline_render` creates the sink, writes every frame and finishes it on the calling thread, and outside `feathertalk-inference` nothing mentions either trait. Removing a supertrait cannot break an implementor either. The alternative -- rendering on a scoped thread and pumping progress back through a channel -- keeps the bounds but buys threading, a join, and a panic payload to translate, for a loop that is single-threaded by nature.
2. The three asset-path constants are component arrays rather than the slash-joined literals the design writes, for the Windows reason in the constraints above.

## File Structure

```
rust/crates/feathertalk-inference/src/error.rs                 + InferenceError::Cancelled
rust/crates/feathertalk-inference/src/raw_sink.rs              + the borrowing sink lifetime, minus the auto-trait bounds
rust/crates/feathertalk-inference/tests/executor.rs            + cancellation at frame N, a borrowing sink
rust/crates/feathertalk-training/src/checkpoint.rs             + read_training_checkpoint, load_training_checkpoint_model
rust/crates/feathertalk-training/src/lib.rs                    + their exports
rust/crates/feathertalk-training/tests/checkpoint_recovery.rs  + model-only loader coverage
rust/crates/feathertalk-worker/Cargo.toml                      + feathertalk-inference
rust/crates/feathertalk-worker/src/lib.rs                       + the render modules and their public surface
rust/crates/feathertalk-worker/src/handshake.rs                + TaskKind::Render
rust/crates/feathertalk-worker/src/runtime.rs                  + the render rejection reason
rust/crates/feathertalk-worker/src/error_map.rs                + render_task_error, is_inference_cancellation
rust/crates/feathertalk-worker/src/rendering.rs                new: backend alias, asset layout, variant dispatch, RenderJob
rust/crates/feathertalk-worker/src/render_result.rs            new: the result payload
rust/crates/feathertalk-worker/src/render.rs                   new: the observing sink, run_render, execute_render
rust/crates/feathertalk-worker/src/commands.rs                 + the Request::Render arm
rust/crates/feathertalk-worker/tests/support/mod.rs            + render fixtures: project tree, stub reader, memory sink
rust/crates/feathertalk-worker/tests/rendering.rs              new: job assembly, admission, dispatch, staging task id
rust/crates/feathertalk-worker/tests/render_result.rs          new: payload shape
rust/crates/feathertalk-worker/tests/render.rs                 new: the loop, progress, cancellation, admission
rust/crates/feathertalk-worker/tests/handshake.rs              + render in the handshake
rust/crates/feathertalk-worker/tests/runtime.rs                + the handshake vectors and the rejection text
rust/crates/feathertalk-worker/tests/error_mapping.rs          + the inference mapper
rust/crates/feathertalk-cli/src/cli.rs                         + the render subcommand
rust/crates/feathertalk-cli/src/run.rs                         + the build_request arm and its inline tests
rust/crates/feathertalk-cli/src/render.rs                      + the unsupported-command hint
rust/crates/feathertalk-cli/tests/real_worker.rs               + the gated end-to-end render
```

Read `docs/superpowers/specs/2026-09-04-render-worker-command-design.md` once before Task 1. Every "why" below is a pointer back into it.

---

### Task 1: Report a cancelled render

**Files:**

- Modify: `rust/crates/feathertalk-inference/src/error.rs` (append one variant)
- Modify: `rust/crates/feathertalk-inference/src/raw_sink.rs:17-18` and `:30-31`
- Test: `rust/crates/feathertalk-inference/tests/executor.rs` (append two tests and their fixtures)

**Interfaces:**

- Produces: `InferenceError::Cancelled { operation: &'static str }`, whose `Display` is `cancelled during {operation}`.
- Produces: `RawVideoSinkFactory::start(&self, command: &CommandSpec) -> Result<Box<dyn RawVideoSink + '_>, InferenceError>`, with `Send` dropped from `RawVideoSink` and `Send + Sync` dropped from `RawVideoSinkFactory`.
- Consumes: `execute_offline_render`, and the `OutputModel`, `RecordingReader`, `RecordingSinkState` and `artifact_tree` fixtures already in `tests/executor.rs`.

**Why now:** the worker turns a cancelled token into a stopped render by failing the sink (design section 6), and `InferenceError` has no variant for that. Faking it with a `SinkWrite` carrying a sentinel message would make a string into a protocol. The lifetime on `start` is the other half: the observing sink has to reach the reporter, and a `'static` box cannot hold a borrow.

- [ ] **Step 1: Write the failing test**

Append to `rust/crates/feathertalk-inference/tests/executor.rs`, and change the existing `RecordingSinkFactory::start` (around line 233) to return `Result<Box<dyn RawVideoSink + '_>, InferenceError>` so the file matches the new trait:

```rust
/// A sink that writes `cancel_at` frames and then reports the render cancelled,
/// which is what the worker's observing sink does once its token is set.
struct CancellingSink {
    state: Arc<Mutex<RecordingSinkState>>,
    cancel_at: usize,
}

impl RawVideoSink for CancellingSink {
    fn write_frame(&mut self, frame: &BgrFrame) -> Result<(), InferenceError> {
        let mut state = self.state.lock().unwrap();
        if state.frames.len() >= self.cancel_at {
            return Err(InferenceError::Cancelled {
                operation: "render",
            });
        }
        state.frames.push(frame.as_bytes().to_vec());
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<(), InferenceError> {
        Ok(())
    }
}

struct CancellingSinkFactory {
    state: Arc<Mutex<RecordingSinkState>>,
    cancel_at: usize,
}

impl RawVideoSinkFactory for CancellingSinkFactory {
    fn start(
        &self,
        command: &feathertalk_inference::CommandSpec,
    ) -> Result<Box<dyn RawVideoSink + '_>, InferenceError> {
        self.state.lock().unwrap().staging = Some(PathBuf::from(
            command.arguments().last().unwrap().to_string_lossy().into_owned(),
        ));
        Ok(Box::new(CancellingSink {
            state: Arc::clone(&self.state),
            cancel_at: self.cancel_at,
        }))
    }
}

/// A factory whose sink borrows the factory. This is the shape the worker needs
/// for progress reporting, and it compiles only because `start` ties the boxed
/// sink to the `&self` borrow.
struct BorrowingSinkFactory {
    written: Mutex<usize>,
}

struct BorrowingSink<'a> {
    factory: &'a BorrowingSinkFactory,
    staging: PathBuf,
}

impl RawVideoSink for BorrowingSink<'_> {
    fn write_frame(&mut self, _frame: &BgrFrame) -> Result<(), InferenceError> {
        let mut written = self.factory.written.lock().unwrap();
        *written += 1;
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<(), InferenceError> {
        std::fs::write(&self.staging, b"rendered-video").unwrap();
        Ok(())
    }
}

impl RawVideoSinkFactory for BorrowingSinkFactory {
    fn start(
        &self,
        command: &feathertalk_inference::CommandSpec,
    ) -> Result<Box<dyn RawVideoSink + '_>, InferenceError> {
        Ok(Box::new(BorrowingSink {
            factory: self,
            staging: PathBuf::from(
                command.arguments().last().unwrap().to_string_lossy().into_owned(),
            ),
        }))
    }
}

#[test]
fn a_cancelled_sink_stops_the_render_and_leaves_no_output() {
    let (_root, request) = artifact_tree(2, 4, None);
    let reader = RecordingReader {
        frames: Arc::new(Mutex::new(Vec::new())),
        fail_at: None,
        alternate_dimensions_at: None,
    };
    let sink_state = Arc::new(Mutex::new(RecordingSinkState::default()));
    let sink_factory = CancellingSinkFactory {
        state: Arc::clone(&sink_state),
        cancel_at: 2,
    };
    let device = Default::default();

    let error = execute_offline_render::<CpuBackend, _, _, _>(
        &OutputModel { value: 1.0 },
        &device,
        &request,
        &reader,
        &sink_factory,
    )
    .expect_err("a cancelled sink fails the render");

    assert!(
        matches!(error, InferenceError::Cancelled { operation } if operation == "render"),
        "{error:?}"
    );
    assert_eq!(error.to_string(), "cancelled during render");
    // The two frames before the cancellation were written; the staging file was
    // cleaned up by the guard and the destination was never created.
    let staging = {
        let state = sink_state.lock().unwrap();
        assert_eq!(state.frames.len(), 2);
        state.staging.clone().expect("the sink was started")
    };
    assert!(!staging.exists());
    assert!(!request.output_path().exists());
}

#[test]
fn a_sink_that_borrows_its_factory_renders_every_frame() {
    let (_root, request) = artifact_tree(2, 2, None);
    let reader = RecordingReader {
        frames: Arc::new(Mutex::new(Vec::new())),
        fail_at: None,
        alternate_dimensions_at: None,
    };
    let sink_factory = BorrowingSinkFactory {
        written: Mutex::new(0),
    };
    let device = Default::default();

    let result = execute_offline_render::<CpuBackend, _, _, _>(
        &OutputModel { value: 1.0 },
        &device,
        &request,
        &reader,
        &sink_factory,
    )
    .expect("the borrowing sink renders");

    assert_eq!(result.frame_count(), 2);
    assert_eq!(*sink_factory.written.lock().unwrap(), 2);
    assert!(request.output_path().is_file());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-inference --test executor`
Expected: FAIL to compile with `error[E0599]: no variant or associated item named Cancelled found for enum InferenceError`, plus a signature mismatch on `BorrowingSinkFactory::start`.

- [ ] **Step 3: Write minimal implementation**

In `rust/crates/feathertalk-inference/src/error.rs`, append the variant after `AllocationFailure`:

```rust
    #[error("cancelled during {operation}")]
    Cancelled { operation: &'static str },
```

In `rust/crates/feathertalk-inference/src/raw_sink.rs`, widen both sink contracts and the one implementation:

```rust
/// The render loop writes every frame on the thread that started it, so neither
/// half of this pair needs an auto-trait bound. Requiring `Send` here would rule
/// out a sink that borrows a `TaskReporter`, which is not `Sync`.
pub trait RawVideoSink {
    fn write_frame(&mut self, frame: &BgrFrame) -> Result<(), InferenceError>;
    fn finish(self: Box<Self>) -> Result<(), InferenceError>;
}

pub trait RawVideoSinkFactory {
    /// The returned sink may borrow the factory. `Box<dyn RawVideoSink>` would
    /// mean `+ 'static` and rule out a sink that observes something the caller
    /// owns -- a progress reporter, a cancellation token -- which is exactly
    /// what a worker wraps around a system sink.
    fn start(&self, command: &CommandSpec) -> Result<Box<dyn RawVideoSink + '_>, InferenceError>;
}
```

```rust
impl RawVideoSinkFactory for SystemRawVideoSinkFactory {
    fn start(&self, command: &CommandSpec) -> Result<Box<dyn RawVideoSink + '_>, InferenceError> {
```

Nothing inside `SystemRawVideoSinkFactory::start` changes: its sink owns everything it uses, and a `'static` box coerces into a shorter-lived one.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-inference --all-targets
cargo fmt --all -- --check
cargo clippy -p feathertalk-inference --all-targets -- -D warnings
```

Expected: every test in the package passes including the two new ones, fmt clean, clippy clean.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-inference/src/error.rs rust/crates/feathertalk-inference/src/raw_sink.rs rust/crates/feathertalk-inference/tests/executor.rs
git commit -m "feat(inference): report a cancelled render"
```

---

### Task 2: Read a checkpoint without its optimizer

**Files:**

- Modify: `rust/crates/feathertalk-training/src/checkpoint.rs` (append after `load_training_checkpoint`, which ends at line 508)
- Modify: `rust/crates/feathertalk-training/src/lib.rs` (the `checkpoint::{...}` export list)
- Test: `rust/crates/feathertalk-training/tests/checkpoint_recovery.rs` (append four tests)

**Interfaces:**

- Produces: `TrainingCheckpointMetadata { manifest: TrainingCheckpointManifest, state: TrainingCheckpointState }`.
- Produces: `RestoredCheckpointModel<M> { model: M, metadata: TrainingCheckpointMetadata }`.
- Produces: `read_training_checkpoint(directory: impl AsRef<Path>) -> Result<TrainingCheckpointMetadata, TrainingError>`.
- Produces: `load_training_checkpoint_model<B, M>(directory: impl AsRef<Path>, model_template: &M, device: &B::Device, expected: &CheckpointDescriptor) -> Result<RestoredCheckpointModel<M>, TrainingError>` where `B: AutodiffBackend, M: AutodiffModule<B> + Clone`.
- Consumes: the private `checkpoint_io` helpers (`reject_symlink_components`, `validate_checkpoint_directory`, `validate_declared_file`, `load_model_record`, `MANIFEST_MAX_BYTES`, `STATE_MAX_BYTES`) and the private `read_checkpoint_json`. `checkpoint_io` is a private module, so both functions must live in `checkpoint.rs`.

**Why now:** rendering needs the weights and nothing else. `load_training_checkpoint` restores the optimizer too, which means building an `AdamConfig` template for a run that will never step, and it demands a full `CheckpointCompatibility` -- a training config and a frame count the render request does not carry. Two entry points rather than one because the dependency runs both ways: a model template can only be built once the variant is known, and the variant is only written in the manifest (design section 4). Repeating the preflight costs two `symlink_metadata` calls and two small JSON reads.

- [ ] **Step 1: Write the failing test**

Append to `rust/crates/feathertalk-training/tests/checkpoint_recovery.rs`, adding `RestoredCheckpointModel`, `load_training_checkpoint_model` and `read_training_checkpoint` to the existing `feathertalk_training::{...}` import:

```rust
/// A checkpoint of a model that has taken one step, so its parameters differ
/// from a fresh `deterministic` template and a restore is visible.
fn saved_checkpoint(
    device: &<CpuAutodiffBackend as Backend>::Device,
    directory: &std::path::Path,
) -> (CheckpointDescriptor, TinyModel<CpuAutodiffBackend>) {
    let model = TinyModel::<CpuAutodiffBackend>::deterministic(device);
    let mut optimizer = AdamConfig::new().init();
    let (model, _) = train_step(model, &mut optimizer, [[1.0, -2.0]], [[0.5]], device);
    let descriptor = CheckpointDescriptor::new("tiny", "tiny-v1", "0".repeat(64));
    save_training_checkpoint::<CpuAutodiffBackend, _, _>(
        directory,
        &model,
        &optimizer,
        descriptor.clone(),
        state(),
    )
    .unwrap();
    (descriptor, model)
}

#[test]
fn a_checkpoint_reports_its_manifest_and_state_without_a_record() {
    let device = Default::default();
    let root = tempfile::tempdir().unwrap();
    let checkpoint = root.path().join("checkpoint-000001");
    let (descriptor, _) = saved_checkpoint(&device, &checkpoint);

    let metadata = read_training_checkpoint(&checkpoint).unwrap();

    assert_eq!(metadata.manifest.descriptor(), descriptor);
    assert_eq!(metadata.manifest.model.file_name, "model.bin");
    assert_eq!(metadata.state, state());
    assert_eq!(metadata.state.global_step, 1);
}

#[test]
fn a_model_only_load_restores_the_weights_and_leaves_the_template_alone() {
    let device = Default::default();
    let root = tempfile::tempdir().unwrap();
    let checkpoint = root.path().join("checkpoint-000001");
    let (descriptor, saved) = saved_checkpoint(&device, &checkpoint);
    let template = TinyModel::<CpuAutodiffBackend>::deterministic(&device);
    let fresh_values = model_parameter_values(&template);

    let restored = load_training_checkpoint_model::<CpuAutodiffBackend, _>(
        &checkpoint,
        &template,
        &device,
        &descriptor,
    )
    .unwrap();

    assert_eq!(
        model_parameter_values(&restored.model),
        model_parameter_values(&saved)
    );
    assert_eq!(model_parameter_values(&template), fresh_values);
    assert_eq!(restored.metadata.state.global_step, 1);
    assert_eq!(restored.metadata.manifest.descriptor(), descriptor);
}

#[test]
fn a_model_only_load_refuses_a_descriptor_that_does_not_match() {
    let device = Default::default();
    let root = tempfile::tempdir().unwrap();
    let checkpoint = root.path().join("checkpoint-000001");
    let (_, _) = saved_checkpoint(&device, &checkpoint);
    let template = TinyModel::<CpuAutodiffBackend>::deterministic(&device);
    let other = CheckpointDescriptor::new("other", "tiny-v1", "0".repeat(64));

    let error = load_training_checkpoint_model::<CpuAutodiffBackend, _>(
        &checkpoint,
        &template,
        &device,
        &other,
    )
    .expect_err("a checkpoint of another model is refused");

    assert!(
        matches!(
            error,
            feathertalk_training::TrainingError::CheckpointCompatibility(_)
        ),
        "{error:?}"
    );
}

#[test]
fn a_model_only_load_refuses_a_checkpoint_without_its_record() {
    let device = Default::default();
    let root = tempfile::tempdir().unwrap();
    let checkpoint = root.path().join("checkpoint-000001");
    let (descriptor, _) = saved_checkpoint(&device, &checkpoint);
    std::fs::remove_file(checkpoint.join("model.bin")).unwrap();
    let template = TinyModel::<CpuAutodiffBackend>::deterministic(&device);

    // The metadata still reads: the manifest and the state are intact, and the
    // missing record is only discovered when it is about to be read.
    read_training_checkpoint(&checkpoint).expect("the metadata is still readable");
    let error = load_training_checkpoint_model::<CpuAutodiffBackend, _>(
        &checkpoint,
        &template,
        &device,
        &descriptor,
    )
    .expect_err("a missing model record is refused");

    let message = error.to_string();
    assert!(message.contains("model.bin"), "{message}");
}

#[test]
fn a_model_only_load_of_a_restored_checkpoint_carries_the_metadata_type() {
    let device = Default::default();
    let root = tempfile::tempdir().unwrap();
    let checkpoint = root.path().join("checkpoint-000001");
    let (descriptor, _) = saved_checkpoint(&device, &checkpoint);
    let template = TinyModel::<CpuAutodiffBackend>::deterministic(&device);

    let restored: RestoredCheckpointModel<TinyModel<CpuAutodiffBackend>> =
        load_training_checkpoint_model::<CpuAutodiffBackend, _>(
            &checkpoint,
            &template,
            &device,
            &descriptor,
        )
        .unwrap();

    assert_eq!(restored.metadata, read_training_checkpoint(&checkpoint).unwrap());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-training --test checkpoint_recovery`
Expected: FAIL to compile with `error[E0432]: unresolved imports feathertalk_training::RestoredCheckpointModel, feathertalk_training::load_training_checkpoint_model, feathertalk_training::read_training_checkpoint`.

- [ ] **Step 3: Write minimal implementation**

In `rust/crates/feathertalk-training/src/checkpoint.rs`, after `load_training_checkpoint` (line 508) and before `read_checkpoint_json`:

```rust
/// Everything a checkpoint says about itself, with no Burn record read.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingCheckpointMetadata {
    pub manifest: TrainingCheckpointManifest,
    pub state: TrainingCheckpointState,
}

/// A model restored from a checkpoint, next to the metadata that described it.
#[derive(Debug, Clone)]
pub struct RestoredCheckpointModel<M> {
    pub model: M,
    pub metadata: TrainingCheckpointMetadata,
}

/// Reads a checkpoint's manifest and training state, and nothing else.
///
/// The preflight is `load_training_checkpoint`'s, in the same order and with the
/// same bounds: no symbolic link on the path, a validated checkpoint directory,
/// then the two JSON documents read under their size caps and validated.
///
/// This exists for a caller that cannot build a model template yet, because the
/// template depends on the model variant and the variant is only written in the
/// manifest. Rendering reads this first, picks the configuration it names, and
/// then calls [`load_training_checkpoint_model`].
pub fn read_training_checkpoint(
    directory: impl AsRef<Path>,
) -> Result<TrainingCheckpointMetadata, TrainingError> {
    let directory = directory.as_ref();
    crate::checkpoint_io::reject_symlink_components(directory)?;
    crate::checkpoint_io::validate_checkpoint_directory(directory)?;
    let manifest: TrainingCheckpointManifest = read_checkpoint_json(
        &directory.join(CHECKPOINT_MANIFEST_FILE_NAME),
        crate::checkpoint_io::MANIFEST_MAX_BYTES,
        "checkpoint manifest",
    )?;
    manifest.validate()?;
    let state: TrainingCheckpointState = read_checkpoint_json(
        &directory.join(CHECKPOINT_STATE_FILE_NAME),
        crate::checkpoint_io::STATE_MAX_BYTES,
        "training checkpoint state",
    )?;
    state.validate()?;
    Ok(TrainingCheckpointMetadata { manifest, state })
}

/// Restores only the model record of a checkpoint.
///
/// Inference has no optimizer to continue and no training configuration to
/// match, so the compatibility gate is the descriptor alone: the model kind, the
/// architecture version and the digest of the model configuration all have to be
/// the ones the caller expects, or the weights would be poured into the wrong
/// shapes and the failure would be a bad video rather than an error.
///
/// The `AutodiffBackend` bound is not decoration: the record was written by a
/// module on `Autodiff<_>`, so reading it back with the same types is what makes
/// it certainly compatible instead of probably compatible. The caller drops the
/// autodiff shell afterwards with `AutodiffModule::valid`.
///
/// The template is only ever cloned. A failed load leaves the caller's template
/// untouched, the same rule `load_training_checkpoint` follows.
pub fn load_training_checkpoint_model<B, M>(
    directory: impl AsRef<Path>,
    model_template: &M,
    device: &B::Device,
    expected: &CheckpointDescriptor,
) -> Result<RestoredCheckpointModel<M>, TrainingError>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Clone,
{
    let directory = directory.as_ref();
    let metadata = read_training_checkpoint(directory)?;
    if metadata.manifest.descriptor() != *expected {
        return Err(TrainingError::CheckpointCompatibility(
            "checkpoint descriptor does not match the expected model".to_owned(),
        ));
    }
    let model_path = directory.join(CHECKPOINT_MODEL_FILE_NAME);
    crate::checkpoint_io::validate_declared_file(&model_path, &metadata.manifest.model)?;
    let model = crate::checkpoint_io::load_model_record::<B, M>(
        model_template.clone(),
        &model_path,
        device,
    )?;
    Ok(RestoredCheckpointModel { model, metadata })
}
```

In `rust/crates/feathertalk-training/src/lib.rs`, add `RestoredCheckpointModel`, `TrainingCheckpointMetadata`, `load_training_checkpoint_model` and `read_training_checkpoint` to the `pub use checkpoint::{...}` list. Do not hand-sort the braces; `cargo fmt --all` orders them.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo fmt --all
cargo test -p feathertalk-training --test checkpoint_recovery
cargo test -p feathertalk-training --all-targets
cargo fmt --all -- --check
cargo clippy -p feathertalk-training --all-targets -- -D warnings
```

Expected: the five new tests pass, the rest of the package is unchanged, fmt clean, clippy clean.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-training/src/checkpoint.rs rust/crates/feathertalk-training/src/lib.rs rust/crates/feathertalk-training/tests/checkpoint_recovery.rs
git commit -m "feat(training): read a checkpoint without its optimizer"
```

---

### Task 3: Announce the render command

**Files:**

- Modify: `rust/crates/feathertalk-worker/src/handshake.rs:26-37` (the media block of `supported_commands`)
- Modify: `rust/crates/feathertalk-worker/src/runtime.rs:404-420` (the `unsupported_reason` match)
- Test: `rust/crates/feathertalk-worker/tests/handshake.rs` (four existing vectors, two new tests)
- Test: `rust/crates/feathertalk-worker/tests/runtime.rs` (two existing vectors, one new test)

**Interfaces:**

- Produces: `TaskKind::Render` inside `supported_commands` whenever `config.media().is_some()`, positioned directly after `TaskKind::NormalizeMedia`.
- Produces: the rejection reason for `render`, which is `media_reason` -- the same sentence `probe_media` and `normalize_media` get.
- Consumes: `WorkerConfig::media`, `media_reason`, and the `Harness` fixtures in `tests/runtime.rs`.

**Why now:** the client refuses a command the handshake does not list, so nothing below can be tested through the wire until the handshake says `render`. Announcing it before the command body exists is safe in the other direction too: `commands.rs` still drops `Request::Render` into its `other =>` arm, which fails the task with a Chinese summary rather than panicking.

- [ ] **Step 1: Write the failing test**

In `rust/crates/feathertalk-worker/tests/handshake.rs`, add `TaskKind::Render` after `TaskKind::NormalizeMedia` in all four expected vectors (lines 41-45, 146-151, 204-211, 295-303), then append:

```rust
#[test]
fn a_media_toolchain_alone_offers_render() {
    let config = configured();
    // Rendering needs ffmpeg and the locked project, so it is offered without
    // any model directory at all.
    assert!(config.models().is_none());
    assert!(config.features().is_none());
    assert!(config.training().is_none());
    let commands = supported_commands(&config);
    assert!(commands.contains(&TaskKind::Render), "{commands:?}");

    let frame = ready_frame(&config);
    frame.validate().unwrap();
    // No new capability flag: `ffmpeg` already reports the same fact.
    assert!(frame.capabilities.ffmpeg);
    assert!(!frame.capabilities.training);
    assert!(!frame.capabilities.wgpu_training);
}

#[test]
fn a_worker_without_a_media_toolchain_leaves_render_out() {
    let config = WorkerConfig::from_values(None, None, None);
    assert!(!supported_commands(&config).contains(&TaskKind::Render));
}
```

In `rust/crates/feathertalk-worker/tests/runtime.rs`, add `TaskKind::Render` after `TaskKind::NormalizeMedia` in the two expected vectors (lines 451-455 and 1008-1013), add `RenderParams` to the `feathertalk_domain::{...}` import, and append the request helper next to `train_request` plus the new test:

```rust
fn render_request() -> Request {
    Request::Render(RenderParams {
        project_dir: PathBuf::from("C:/tmp/project"),
        checkpoint: PathBuf::from("C:/tmp/project/models/unet/checkpoint-00000004"),
        audio: PathBuf::from("C:/tmp/project/assets/audio_16k_mono.wav"),
        output: PathBuf::from("C:/tmp/preview.mp4"),
        max_output_frames: None,
    })
}
```

```rust
#[test]
fn a_render_request_without_media_tools_names_both_variables() {
    let harness = Harness::start(WorkerConfig::from_values(None, None, None), instant_executor());
    harness.send(&start(&task("0000000b"), render_request()));
    let frames = harness.finish();

    let reasons = rejections(&frames);
    assert_eq!(reasons.len(), 1, "{frames:?}");
    assert!(reasons[0].contains("render"), "{}", reasons[0]);
    // `MediaToolchain::new` wants both tools, so both are named.
    assert!(
        reasons[0].contains("FEATHERTALK_WORKER_FFPROBE"),
        "{}",
        reasons[0]
    );
    assert!(
        reasons[0].contains("FEATHERTALK_WORKER_FFMPEG"),
        "{}",
        reasons[0]
    );
    assert!(
        events(&frames).is_empty(),
        "a rejected start creates no task"
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-worker --test handshake --test runtime`
Expected: FAIL. The vector assertions report a left-hand side without `render`, and `a_render_request_without_media_tools_names_both_variables` gets the generic "this worker does not support" sentence, which lists the supported commands instead of the two variables.

- [ ] **Step 3: Write minimal implementation**

In `rust/crates/feathertalk-worker/src/handshake.rs`, inside the existing `config.media().is_some()` block and before the models check:

```rust
        // Rendering shells out to ffmpeg and to nothing else: the frames, the
        // landmarks and the audio features are already inside the locked
        // project, and inference computes no perceptual loss, so neither the
        // frame models nor a model package are preconditions.
        commands.push(TaskKind::Render);
```

In `rust/crates/feathertalk-worker/src/runtime.rs`, add the arm to `unsupported_reason` after the `TaskKind::Train` arm:

```rust
        // Rendering needs the media toolchain and nothing else, so it shares the
        // media commands' reason: both tools, both variable names.
        TaskKind::Render => media_reason(slug, config),
```

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-worker --test handshake --test runtime
cargo test -p feathertalk-worker --all-targets
cargo fmt --all -- --check
cargo clippy -p feathertalk-worker --all-targets -- -D warnings
```

Expected: every worker test passes. `tests/process_boundary.rs` asserts the no-toolchain handshake, so it is unaffected; if it fails, the push landed outside the media block.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/src/handshake.rs rust/crates/feathertalk-worker/src/runtime.rs rust/crates/feathertalk-worker/tests/handshake.rs rust/crates/feathertalk-worker/tests/runtime.rs
git commit -m "feat(worker): announce the render command"
```

---

### Task 4: Map inference errors to task errors

**Files:**

- Modify: `rust/crates/feathertalk-worker/Cargo.toml` (add the `feathertalk-inference` dependency)
- Modify: `rust/crates/feathertalk-worker/src/error_map.rs` (append the inference mapper)
- Modify: `rust/crates/feathertalk-worker/src/lib.rs` (export `render_task_error` and `is_inference_cancellation`)
- Test: `rust/crates/feathertalk-worker/tests/error_mapping.rs` (append the variant table)

**Interfaces:**

- Produces: `pub fn render_task_error(error: &InferenceError, stage: TaskStage) -> TaskError`, which is `TaskError::new(code, summary, &clamp(&error.to_string()), stage)` -- the same shape `training_task_error` has at lines 121-128.
- Produces: `pub fn is_inference_cancellation(error: &InferenceError) -> bool`.
- Produces, private to the module: `inference_error_code` and `inference_summary`, both matching exhaustively with no `_` arm.
- Consumes: `clamp` (already `pub(crate)` here), `TaskError::new`, `ErrorCode`, `TaskStage`.

**Why now:** every task below reports its failures through this one function, and the mapping is the part of design section 7 that is easiest to get wrong. Doing it first means the later tasks only ever call it. The exhaustive match without a `_` arm is deliberate: a new `InferenceError` variant must not silently become a `MediaInvalid`, it must break the build right here.

- [ ] **Step 1: Write the failing test**

Append to `rust/crates/feathertalk-worker/tests/error_mapping.rs`, adding `render_task_error` and `is_inference_cancellation` to the `feathertalk_worker::{...}` import and `feathertalk_inference::InferenceError` next to it. Write one row per variant, in the declaration order of `feathertalk-inference/src/error.rs` -- all forty of them, because a `Vec` gives no exhaustiveness check and a forgotten row is a mapping nobody ever ran. The rows below are the shape; the rest come from the table in Step 3.

```rust
#[test]
fn every_inference_error_maps_to_a_render_task_error() {
    let stage = TaskStage::Rendering { frame: 3, total: 8 };
    let cases: Vec<(InferenceError, ErrorCode)> = vec![
        (
            InferenceError::InvalidInputDirectory {
                field: "frame_dir",
                path: path(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            InferenceError::InvalidInputArtifact {
                field: "landmark_path",
                path: path(),
                message: "unreadable".to_owned(),
            },
            ErrorCode::LandmarkInvalid,
        ),
        (
            InferenceError::InvalidInputArtifact {
                field: "feature_path",
                path: path(),
                message: "unreadable".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            InferenceError::InvalidFeatureShape {
                tokens: 3,
                dims: 1024,
            },
            ErrorCode::FeatureShapeMismatch,
        ),
        (
            InferenceError::NonFinitePrediction { index: 2 },
            ErrorCode::ModelIncompatible,
        ),
        (
            InferenceError::SinkWrite {
                message: "broken pipe".to_owned(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            InferenceError::Cancelled {
                operation: "render",
            },
            ErrorCode::TaskCancelled,
        ),
        // ... one row for every remaining variant
    ];

    for (error, expected) in cases {
        let mapped = render_task_error(&error, stage.clone());
        assert_eq!(mapped.code, expected, "{error:?}");
        mapped.validate().unwrap();
        assert!(!mapped.summary.is_empty(), "{error:?}");
        // Summaries are user-facing, so they are Chinese, never English.
        assert!(!mapped.summary.is_ascii(), "{}", mapped.summary);
        assert!(!mapped.detail.is_empty(), "{error:?}");
        assert_eq!(mapped.stage, stage, "{error:?}");
    }
}

#[test]
fn only_a_cancelled_render_counts_as_a_cancellation() {
    assert!(is_inference_cancellation(&InferenceError::Cancelled {
        operation: "render",
    }));
    assert!(!is_inference_cancellation(&InferenceError::EmptyFeatures));
    assert!(!is_inference_cancellation(
        &InferenceError::ArithmeticOverflow
    ));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-worker --test error_mapping`
Expected: FAIL to compile -- `unresolved import feathertalk_worker::render_task_error`, and `use of unresolved crate feathertalk_inference` until the dependency is added.

- [ ] **Step 3: Write minimal implementation**

In `rust/crates/feathertalk-worker/Cargo.toml`, next to the existing path dependencies:

```toml
feathertalk-inference = { path = "../feathertalk-inference" }
```

Append to `rust/crates/feathertalk-worker/src/error_map.rs`:

```rust
/// The `field` name inference uses for the landmark artifact. It is the one
/// input whose failure the protocol reports as a landmark problem rather than a
/// generic media problem.
const LANDMARK_FIELD: &str = "landmark_path";

/// Maps an inference failure onto the wire, keeping the stage the render was in
/// when it happened (design section 7).
pub fn render_task_error(error: &InferenceError, stage: TaskStage) -> TaskError {
    TaskError::new(
        inference_error_code(error),
        inference_summary(error),
        &clamp(&error.to_string()),
        stage,
    )
}

/// True when the render stopped because the task was cancelled, which the
/// command reports as `CommandOutcome::Cancelled` rather than as a failure.
pub fn is_inference_cancellation(error: &InferenceError) -> bool {
    matches!(error, InferenceError::Cancelled { .. })
}

/// Exhaustive on purpose: a new inference failure has to be classified here
/// rather than defaulting to `MediaInvalid` through a `_` arm.
fn inference_error_code(error: &InferenceError) -> ErrorCode {
    match error {
        // The guard comes first: a landmark file that will not parse is the
        // client's data problem, and the protocol has a code that says so.
        InferenceError::InvalidInputArtifact { field, .. } if *field == LANDMARK_FIELD => {
            ErrorCode::LandmarkInvalid
        }
        InferenceError::InvalidBbox { .. } => ErrorCode::LandmarkInvalid,
        InferenceError::InvalidFeatureShape { .. } => ErrorCode::FeatureShapeMismatch,
        InferenceError::InvalidInputDirectory { .. }
        | InferenceError::InvalidInputArtifact { .. }
        | InferenceError::InvalidField { .. }
        | InferenceError::ArithmeticOverflow
        | InferenceError::OutputExists { .. }
        | InferenceError::OutputNotRegular { .. }
        | InferenceError::OutputSymlink { .. }
        | InferenceError::OutputParentInvalid { .. }
        | InferenceError::InvalidTaskId { .. }
        | InferenceError::FfmpegPathNotAbsolute { .. }
        | InferenceError::EmptyFfmpegPath
        | InferenceError::FrameCountTooSmall { .. }
        | InferenceError::EmptyFeatures
        | InferenceError::FrameIndexOutOfRange { .. }
        | InferenceError::OutputFrameOutOfRange { .. }
        | InferenceError::InvalidAudioWindowIndex { .. }
        | InferenceError::FrameDimensionsMismatch { .. }
        | InferenceError::InvalidFrameDimensions { .. }
        | InferenceError::InvalidResizeTarget { .. }
        | InferenceError::PixelOutOfRange { .. }
        | InferenceError::PasteOutOfBounds { .. }
        | InferenceError::FrameBufferLengthMismatch { .. } => ErrorCode::MediaInvalid,
        InferenceError::TensorShapeMismatch { .. }
        | InferenceError::NonFiniteModelInput { .. }
        | InferenceError::ModelTensorData { .. }
        | InferenceError::NonFiniteModelOutput { .. }
        | InferenceError::ModelOutputOutOfRange { .. }
        | InferenceError::NonFinitePrediction { .. } => ErrorCode::ModelIncompatible,
        InferenceError::SinkStart { .. }
        | InferenceError::SinkWrite { .. }
        | InferenceError::SinkFinish { .. }
        | InferenceError::ToolFailed { .. }
        | InferenceError::StagingCollision { .. }
        | InferenceError::StagingOutputInvalid { .. }
        | InferenceError::AtomicPublishFailed { .. }
        | InferenceError::FrameReader { .. }
        | InferenceError::AllocationFailure { .. } => ErrorCode::WorkerCrashed,
        InferenceError::Cancelled { .. } => ErrorCode::TaskCancelled,
    }
}
```

```rust
/// The user-facing half of the mapping. Grouped by what the operator can do
/// about it, which is why the groups do not match the code groups one for one.
fn inference_summary(error: &InferenceError) -> &'static str {
    match error {
        InferenceError::InvalidInputArtifact { field, .. } if *field == LANDMARK_FIELD => {
            "关键点数据无效"
        }
        InferenceError::InvalidBbox { .. } => "关键点数据无效",
        InferenceError::InvalidFeatureShape { .. } => "音频特征形状不符",
        InferenceError::InvalidField { .. }
        | InferenceError::ArithmeticOverflow
        | InferenceError::OutputExists { .. }
        | InferenceError::OutputNotRegular { .. }
        | InferenceError::OutputSymlink { .. }
        | InferenceError::OutputParentInvalid { .. }
        | InferenceError::InvalidTaskId { .. }
        | InferenceError::FfmpegPathNotAbsolute { .. }
        | InferenceError::EmptyFfmpegPath
        | InferenceError::InvalidResizeTarget { .. } => "渲染请求无效",
        InferenceError::InvalidInputDirectory { .. }
        | InferenceError::InvalidInputArtifact { .. }
        | InferenceError::FrameCountTooSmall { .. }
        | InferenceError::EmptyFeatures
        | InferenceError::FrameIndexOutOfRange { .. }
        | InferenceError::OutputFrameOutOfRange { .. }
        | InferenceError::InvalidAudioWindowIndex { .. }
        | InferenceError::FrameDimensionsMismatch { .. }
        | InferenceError::InvalidFrameDimensions { .. }
        | InferenceError::FrameBufferLengthMismatch { .. } => "渲染素材不可用",
        InferenceError::PixelOutOfRange { .. } | InferenceError::PasteOutOfBounds { .. } => {
            "渲染几何越界"
        }
        InferenceError::TensorShapeMismatch { .. }
        | InferenceError::NonFiniteModelInput { .. }
        | InferenceError::ModelTensorData { .. }
        | InferenceError::NonFiniteModelOutput { .. }
        | InferenceError::ModelOutputOutOfRange { .. }
        | InferenceError::NonFinitePrediction { .. } => "模型推理结果异常",
        InferenceError::SinkStart { .. }
        | InferenceError::SinkWrite { .. }
        | InferenceError::SinkFinish { .. }
        | InferenceError::ToolFailed { .. } => "视频编码进程失败",
        InferenceError::StagingCollision { .. }
        | InferenceError::StagingOutputInvalid { .. }
        | InferenceError::AtomicPublishFailed { .. } => "产物发布失败",
        InferenceError::FrameReader { .. } => "视频帧解码失败",
        InferenceError::AllocationFailure { .. } => "内存不足",
        InferenceError::Cancelled { .. } => "任务已取消",
    }
}
```

Add `use feathertalk_inference::InferenceError;` to the imports at the top of the module, and re-export both public functions from `rust/crates/feathertalk-worker/src/lib.rs` next to `training_task_error`.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-worker --test error_mapping
cargo fmt --all
cargo fmt --all -- --check
cargo clippy -p feathertalk-worker --all-targets -- -D warnings
```

Expected: the table passes and clippy is silent. If clippy reports `match_like_matches_macro` on `is_inference_cancellation`, the `matches!` form is already what it wants; if it reports an unreachable arm, the landmark guard moved below the general `InvalidInputArtifact` arm.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/Cargo.toml rust/crates/feathertalk-worker/src/error_map.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/tests/error_mapping.rs
git commit -m "feat(worker): map inference errors to task errors"
```

---

### Task 5: Assemble a render job

**Files:**

- Create: `rust/crates/feathertalk-worker/src/rendering.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs` (declare the module and re-export its surface)
- Test: `rust/crates/feathertalk-worker/tests/rendering.rs` (new)

**Interfaces:**

- Produces the backend aliases: `pub type RenderBackend = CpuBackend`, `pub type RenderDevice = Device<RenderBackend>`, `pub const RENDER_BACKEND_NAME: &str = "ndarray-cpu"`, `pub const RENDER_FPS: u32 = 25`.
- Produces the asset layout: `pub struct ProjectAssets { pub frame_dir, pub landmark_dir, pub feature_path }` and `pub fn project_assets(project_dir: &Path) -> ProjectAssets`, built from the three private component arrays.
- Produces the variant dispatch: `pub enum RenderVariant { OriginalUnet(OriginalUnetConfig), MobileOneUnet(MobileOneUnetConfig) }`, `RenderVariant::configuration(&self) -> ModelConfiguration`, and `pub fn render_variant(model_kind: &str) -> Option<RenderVariant>`.
- Produces the admission helpers: `pub fn check_render_paths(params: &RenderParams) -> Result<(), TaskError>` and `pub fn check_max_output_frames(max: Option<u64>) -> Result<Option<usize>, TaskError>`.
- Produces `pub fn staging_task_id() -> String` and `pub fn progress_total(frame_count: u64, max_output_frames: Option<u64>) -> u64`.
- Produces `pub struct RenderJob` and `pub fn render_job(...) -> Result<RenderJob, TaskError>`.
- Consumes: `RenderParams`, `OfflineRenderRequest::new`, `CheckpointDescriptor`, `render_task_error` from Task 4, `invalid_request` from `admission.rs`, `ModelConfiguration::{original_unet, mobileone_unet, model_type}`.

**Why now:** `render.rs` is the command body and it should not also be the place where paths, limits and variant names are decided. Splitting the pure decisions out means the interesting half -- the loop, progress, cancellation -- can be tested with an in-memory sink, and the boring half can be tested without a model at all. `render_variant` compares `ModelConfiguration::model_type()` rather than a string literal, so the worker cannot drift from the name the checkpoint manifest was written with.

- [ ] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-worker/tests/rendering.rs`. Every path in a `RenderParams` fixture is absolute, because `check_render_paths` is what rejects the relative ones.

```rust
use std::path::{Path, PathBuf};

use feathertalk_domain::{ErrorCode, RenderParams};
use feathertalk_worker::{
    ProjectAssets, RENDER_BACKEND_NAME, RENDER_FPS, RenderVariant, check_max_output_frames,
    check_render_paths, progress_total, project_assets, render_variant, staging_task_id,
};

/// A `TempDir` root keeps the fixture absolute on every platform without
/// hard-coding a drive letter.
fn params(root: &Path) -> RenderParams {
    RenderParams {
        project_dir: root.join("project"),
        checkpoint: root.join("checkpoint"),
        audio: root.join("voice.wav"),
        output: root.join("preview.mp4"),
        max_output_frames: None,
    }
}
```

Cover, one test each:

- `project_assets` puts the three locations under the project root with native separators: assert every component with `Path::components`, and assert the feature file name is `feather_hubert.f32`.
- `render_variant("original_unet")` is `Some(RenderVariant::OriginalUnet(..))`, `render_variant("mobileone_unet")` is `Some(..MobileOneUnet..)`, `render_variant("unet")` and `render_variant("")` are `None`; and for both hits, `variant.configuration().model_type()` equals the string that was passed in.
- `check_render_paths` accepts the absolute fixture and rejects a relative `checkpoint`, `audio` and `output` with `ErrorCode::MediaInvalid`, a Chinese summary and a detail naming the field.
- `check_max_output_frames`: `None` stays `None`, `Some(3)` becomes `Some(3)`, `Some(0)` is rejected, and `Some(u64::MAX)` is rejected on a 64-bit host only if it does not fit -- assert the `Some(0)` rejection is `MediaInvalid` and leave the overflow case to a `usize::try_from` check so the test is portable.
- `progress_total`: `(4, None) == 4`, `(4, Some(2)) == 2`, `(4, Some(9)) == 4`.
- `staging_task_id` returns two different ids on two calls, both starting with `render-`, and neither containing a path separator.
- `RENDER_BACKEND_NAME == "ndarray-cpu"` and `RENDER_FPS == 25`, so the payload contract in Task 6 is pinned by a test that does not need a video.

Then the job itself, which is the only test here that needs a `CheckpointDescriptor`:

```rust
#[test]
fn a_render_job_carries_the_project_layout_and_the_checkpoint_identity() {
    let root = tempdir().unwrap();
    let params = params(root.path());
    let variant = render_variant("original_unet").expect("original_unet is a known kind");
    let descriptor = checkpoint_descriptor(&variant.configuration());
    let ffmpeg = root.path().join("ffmpeg.exe");

    let job = render_job(&params, 4, &ffmpeg, descriptor.clone(), 2, 8)
        .expect("an absolute request with four frames is a job");

    let assets = project_assets(&params.project_dir);
    assert_eq!(job.request.frame_dir(), assets.frame_dir);
    assert_eq!(job.request.landmark_dir(), assets.landmark_dir);
    assert_eq!(job.request.feature_path(), assets.feature_path);
    assert_eq!(job.request.audio_path(), params.audio);
    assert_eq!(job.request.output_path(), params.output);
    assert!(job.request.task_id().starts_with("render-"));
    // The total comes from the locked manifest, never from the feature file.
    assert_eq!(job.progress_total, 4);
    assert_eq!(job.source_frame_count, 4);
    assert_eq!(job.max_output_frames, None);
    assert_eq!(job.checkpoint_dir, params.checkpoint);
    assert_eq!(job.checkpoint_epoch, 2);
    assert_eq!(job.checkpoint_global_step, 8);
    assert_eq!(job.descriptor, descriptor);
}
```

Use the accessor names `OfflineRenderRequest` actually exposes -- read `rust/crates/feathertalk-inference/src/request.rs` before writing this test rather than guessing -- and add two more cases: a `frame_count` of 1 is rejected with `ErrorCode::MediaInvalid` (inference needs at least two source frames for the ping-pong walk), and `max_output_frames: Some(2)` gives `progress_total == 2` while `source_frame_count` stays 4.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-worker --test rendering`
Expected: FAIL to compile -- `unresolved import feathertalk_worker::render_variant` and the rest of the surface, because `src/rendering.rs` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create `rust/crates/feathertalk-worker/src/rendering.rs`. First the backend and the layout:

```rust
/// Rendering runs one forward pass per output frame on the CPU. There is no
/// autodiff here: the weights are read once and never updated.
pub type RenderBackend = CpuBackend;

/// The device type the render command hands to the model.
pub type RenderDevice = Device<RenderBackend>;

/// The backend name the result payload reports.
pub const RENDER_BACKEND_NAME: &str = "ndarray-cpu";

/// The frame rate inference writes into the container.
pub const RENDER_FPS: u32 = 25;

/// Where a locked project keeps the cropped frames, the landmarks and the audio
/// features, relative to the project root. These are component arrays rather
/// than slash-joined literals so a Windows join produces native separators.
const FRAME_DIR: [&str; 2] = ["assets", "frames"];
const LANDMARK_DIR: [&str; 2] = ["assets", "landmarks"];
const FEATURE_PATH: [&str; 3] = ["assets", "features", "feather_hubert.f32"];

/// Joins one component at a time, which is the only way to keep the separator
/// native on Windows.
fn project_path(root: &Path, components: &[&str]) -> PathBuf {
    let mut path = root.to_path_buf();
    for component in components {
        path.push(component);
    }
    path
}

/// The three inference inputs a locked project already contains.
pub struct ProjectAssets {
    pub frame_dir: PathBuf,
    pub landmark_dir: PathBuf,
    pub feature_path: PathBuf,
}

pub fn project_assets(project_dir: &Path) -> ProjectAssets {
    ProjectAssets {
        frame_dir: project_path(project_dir, &FRAME_DIR),
        landmark_dir: project_path(project_dir, &LANDMARK_DIR),
        feature_path: project_path(project_dir, &FEATURE_PATH),
    }
}
```

Then the variant dispatch, the staging id and the numbers:

```rust
/// The two architectures a training checkpoint can hold.
pub enum RenderVariant {
    OriginalUnet(OriginalUnetConfig),
    MobileOneUnet(MobileOneUnetConfig),
}

impl RenderVariant {
    /// The descriptor identity of this variant. `mobileone_unet` is described in
    /// its training shape (`false`), because that is how the checkpoint was
    /// written; reparameterization happens after the record is restored.
    pub fn configuration(&self) -> ModelConfiguration {
        match self {
            Self::OriginalUnet(config) => ModelConfiguration::original_unet(config),
            Self::MobileOneUnet(config) => ModelConfiguration::mobileone_unet(config, false),
        }
    }
}

/// Resolves the `model_kind` a checkpoint manifest recorded. The comparison is
/// against `ModelConfiguration::model_type` rather than a literal, so the worker
/// cannot drift from the name the checkpoint was written with.
pub fn render_variant(model_kind: &str) -> Option<RenderVariant> {
    let original = RenderVariant::OriginalUnet(OriginalUnetConfig::default());
    if original.configuration().model_type() == model_kind {
        return Some(original);
    }
    let mobileone = RenderVariant::MobileOneUnet(MobileOneUnetConfig::default());
    if mobileone.configuration().model_type() == model_kind {
        return Some(mobileone);
    }
    None
}

/// Inference stages its output next to the destination under a name built from
/// the task id, so the id has to be unique within the process. The pid keeps two
/// workers apart, the counter keeps two renders in one worker apart.
pub fn staging_task_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let ordinal = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("render-{}-{ordinal}", std::process::id())
}

/// The number of frames the render will write, from the locked manifest and the
/// caller's cap. This mirrors `RenderPlan::new`, which takes the same minimum.
pub fn progress_total(frame_count: u64, max_output_frames: Option<u64>) -> u64 {
    match max_output_frames {
        Some(max) => max.min(frame_count),
        None => frame_count,
    }
}
```

Then admission. Both helpers return `invalid_request`, which is already `MediaInvalid` at `TaskStage::Preparing`:

```rust
/// Every path in a render request is absolute, because the worker resolves
/// nothing against its own working directory.
pub fn check_render_paths(params: &RenderParams) -> Result<(), TaskError> {
    if !params.checkpoint.is_absolute() {
        return Err(invalid_request(
            "检查点目录必须是绝对路径",
            format!("checkpoint is not absolute: {}", params.checkpoint.display()),
        ));
    }
    if !params.audio.is_absolute() {
        return Err(invalid_request(
            "音频文件必须是绝对路径",
            format!("audio is not absolute: {}", params.audio.display()),
        ));
    }
    if !params.output.is_absolute() {
        return Err(invalid_request(
            "输出文件必须是绝对路径",
            format!("output is not absolute: {}", params.output.display()),
        ));
    }
    Ok(())
}

/// `max_output_frames` is `Option<u64>` on the wire and `Option<usize>` in
/// inference. Zero is refused rather than clamped, and a value that does not fit
/// the host word size is refused rather than truncated.
pub fn check_max_output_frames(max: Option<u64>) -> Result<Option<usize>, TaskError> {
    let Some(max) = max else {
        return Ok(None);
    };
    if max == 0 {
        return Err(invalid_request(
            "最大输出帧数必须大于 0",
            "max_output_frames is zero".to_owned(),
        ));
    }
    let max = usize::try_from(max).map_err(|_| {
        invalid_request(
            "最大输出帧数超出本机可表示范围",
            format!("max_output_frames does not fit in usize: {max}"),
        )
    })?;
    Ok(Some(max))
}
```

And the job:

```rust
/// Everything the render loop needs once admission is done: the inference
/// request, the progress denominator, and the checkpoint identity the result
/// payload reports.
pub struct RenderJob {
    pub request: OfflineRenderRequest,
    pub progress_total: u64,
    pub descriptor: CheckpointDescriptor,
    pub checkpoint_dir: PathBuf,
    pub checkpoint_epoch: u64,
    pub checkpoint_global_step: u64,
    pub source_frame_count: u64,
    pub max_output_frames: Option<u64>,
}

/// Turns an admitted request plus the locked manifest's frame count into a job.
/// The frame count comes from the manifest and not from the feature file, so a
/// project whose features were regenerated cannot silently change the total.
pub fn render_job(
    params: &RenderParams,
    frame_count: u64,
    ffmpeg: &Path,
    descriptor: CheckpointDescriptor,
    checkpoint_epoch: u64,
    checkpoint_global_step: u64,
) -> Result<RenderJob, TaskError> {
    /// Inference walks the source frames forwards and back, which needs two.
    const MINIMUM_FRAMES: u64 = 2;

    if frame_count < MINIMUM_FRAMES {
        return Err(invalid_request(
            "工程帧数不足，无法渲染",
            format!("frame_count is {frame_count}, the minimum is {MINIMUM_FRAMES}"),
        ));
    }
    let source_frames = usize::try_from(frame_count).map_err(|_| {
        invalid_request(
            "工程帧数超出本机可表示范围",
            format!("frame_count does not fit in usize: {frame_count}"),
        )
    })?;
    let max_output_frames = check_max_output_frames(params.max_output_frames)?;
    let assets = project_assets(&params.project_dir);
    let request = OfflineRenderRequest::new(
        assets.frame_dir,
        assets.landmark_dir,
        assets.feature_path,
        params.audio.clone(),
        ffmpeg.to_path_buf(),
        params.output.clone(),
        staging_task_id(),
        source_frames,
        max_output_frames,
    )
    .map_err(|error| render_task_error(&error, TaskStage::Preparing))?;

    Ok(RenderJob {
        request,
        progress_total: progress_total(frame_count, params.max_output_frames),
        descriptor,
        checkpoint_dir: params.checkpoint.clone(),
        checkpoint_epoch,
        checkpoint_global_step,
        source_frame_count: frame_count,
        max_output_frames: params.max_output_frames,
    })
}
```

Imports: `std::path::{Path, PathBuf}`, `std::sync::atomic::{AtomicU64, Ordering}`, `burn::tensor::Device`, `feathertalk_domain::{RenderParams, TaskError, TaskStage}`, `feathertalk_inference::OfflineRenderRequest`, `feathertalk_models::backend::CpuBackend`, the two model configs, `feathertalk_models::ModelConfiguration`, `feathertalk_training::CheckpointDescriptor`, and `crate::{admission::invalid_request, error_map::render_task_error}`. Declare `pub mod rendering;` in `src/lib.rs` and re-export the public surface the way the training modules are re-exported. Check the argument order and the exact accessor names of `OfflineRenderRequest::new` in `rust/crates/feathertalk-inference/src/request.rs` before writing this.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-worker --test rendering
cargo fmt --all
cargo fmt --all -- --check
cargo clippy -p feathertalk-worker --all-targets -- -D warnings
```

Expected: all of `tests/rendering.rs` passes. If clippy asks for `Option::map_or` in `progress_total`, keep the `match`: it reads as the minimum it is.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/src/rendering.rs rust/crates/feathertalk-worker/tests/rendering.rs
git commit -m "feat(worker): assemble a render job"
```

---

### Task 6: Report a render result

**Files:**

- Create: `rust/crates/feathertalk-worker/src/render_result.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs` (declare the module, export `RenderSummary` and `render_to_json`)
- Test: `rust/crates/feathertalk-worker/tests/render_result.rs` (new)

**Interfaces:**

- Produces: `pub struct RenderSummary<'a>` borrowing the `OfflineRenderResult`, the `CheckpointDescriptor` and the checkpoint directory, plus the four scalars the request and the manifest contributed.
- Produces: `pub fn render_to_json(summary: &RenderSummary<'_>) -> Value` with exactly fourteen fields, in this order: `output_path`, `frame_count`, `width`, `height`, `fps`, `backend`, `checkpoint_dir`, `model_kind`, `architecture_version`, `model_config_sha256`, `checkpoint_epoch`, `checkpoint_global_step`, `source_frame_count`, `max_output_frames`.
- Consumes: `OfflineRenderResult::{output_path, frame_count, width, height}`, `RENDER_FPS`, `RENDER_BACKEND_NAME`, `CheckpointDescriptor::{model_kind, architecture_version, model_config_sha256}`.

**Why now:** the payload is the contract Task 10 asserts against, and it is the one part of this slice that can be written and tested without a render at all. Building it before `render.rs` means the loop has somewhere to hand its result, and the end-to-end test has a documented shape to check.

- [ ] **Step 1: Write the failing test**

`OfflineRenderResult` has private fields and no public constructor (`feathertalk-inference/src/executor.rs:120-143`), so the payload test has to render something. This task therefore also adds the render fixtures to `rust/crates/feathertalk-worker/tests/support/mod.rs`, which Task 7 reuses:

- `pub fn render_tree(frame_count: usize, feature_frames: usize) -> (TempDir, PathBuf)` -- a project root with `assets/frames/{index:06}.jpg`, `assets/landmarks/{index:06}.lms` (110 points, point 31 at `x = 168`), `assets/features/feather_hubert.f32` written with `feathertalk_audio::write_feature_file`, and an `audio.wav` placeholder. Port it from `feathertalk-inference/tests/executor.rs:252-292`, but lay the files out at the locked project's own paths so `project_assets` finds them.
- `pub struct StubFrameReader { pub frames: Mutex<Vec<usize>> }` implementing `FrameReader` and returning a 168x168 BGR frame, so the render geometry's paste stays in bounds.
- `pub struct MemorySinkFactory { pub frames: Mutex<Vec<usize>>, pub staging: Mutex<Option<PathBuf>> }` implementing `RawVideoSinkFactory`, whose sink counts frames and writes `b"rendered-video"` to the staging path on `finish` so the atomic publish has something to rename.
- `pub fn render_model(device: &RenderDevice) -> OriginalUnet<RenderBackend>` built from `OriginalUnetConfig::parity_micro()`, which is small enough to run inside a unit test.

Then create `rust/crates/feathertalk-worker/tests/render_result.rs`, mirroring `tests/train_result.rs`, with `rendered(frame_count)` running `execute_offline_render` over those fixtures once and returning the `OfflineRenderResult`.

```rust
#[test]
fn a_render_payload_names_the_weights_the_video_came_from() {
    let descriptor = checkpoint_descriptor(&ModelConfiguration::original_unet(
        &OriginalUnetConfig::default(),
    ));
    let checkpoint_dir = PathBuf::from("C:/tmp/project/models/unet/checkpoint-00000004");
    let result = rendered(2);
    let payload = render_to_json(&RenderSummary {
        result: &result,
        descriptor: &descriptor,
        checkpoint_dir: &checkpoint_dir,
        checkpoint_epoch: 1,
        checkpoint_global_step: 4,
        source_frame_count: 2,
        max_output_frames: None,
    });

    let object = payload.as_object().expect("the payload is an object");
    assert_eq!(object.len(), 14, "{payload}");
    assert_eq!(payload["frame_count"], 2);
    // The stub reader hands back 168x168 frames, and inference reports the size
    // it actually wrote rather than the size the manifest advertises.
    assert_eq!(payload["width"], 168);
    assert_eq!(payload["height"], 168);
    // The container's frame rate is fixed by inference, not by the request.
    assert_eq!(payload["fps"], 25);
    assert_eq!(payload["backend"], "ndarray-cpu");
    assert_eq!(payload["model_kind"], "original_unet");
    assert_eq!(payload["checkpoint_epoch"], 1);
    assert_eq!(payload["checkpoint_global_step"], 4);
    assert_eq!(payload["source_frame_count"], 2);
    // A request without a cap reports the absence, rather than repeating the
    // frame count and pretending the client asked for it.
    assert!(payload["max_output_frames"].is_null(), "{payload}");
    assert_eq!(
        payload["architecture_version"],
        descriptor.architecture_version.as_str()
    );
    assert_eq!(
        payload["model_config_sha256"],
        descriptor.model_config_sha256.as_str()
    );
}
```

Add a second test for `max_output_frames: Some(2)` reporting `2`, and a third asserting `output_path` and `checkpoint_dir` come back as the same text `Path::display` produces, so a Windows payload keeps its backslashes.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-worker --test render_result`
Expected: FAIL to compile -- `unresolved import feathertalk_worker::render_to_json`.

- [ ] **Step 3: Write minimal implementation**

Create `rust/crates/feathertalk-worker/src/render_result.rs`:

```rust
/// Everything the completed event of a render task reports. Borrowed rather than
/// cloned: the caller owns the result and the job while the payload is built.
pub struct RenderSummary<'a> {
    pub result: &'a OfflineRenderResult,
    /// Supplies `model_kind`, `architecture_version` and `model_config_sha256`
    /// -- the identity of the weights this video came from.
    pub descriptor: &'a CheckpointDescriptor,
    pub checkpoint_dir: &'a Path,
    pub checkpoint_epoch: u64,
    pub checkpoint_global_step: u64,
    /// The locked manifest's frame count, which is the render's upper bound.
    pub source_frame_count: u64,
    /// The request's cap, echoed as `null` when it did not set one.
    pub max_output_frames: Option<u64>,
}

/// Shapes the payload the `completed` event of a render task carries.
///
/// `fps` and `backend` are constants rather than measurements: inference fixes
/// the container's frame rate and the render always runs on the CPU. The four
/// checkpoint fields are the audit trail -- given this payload, the exact
/// weights that produced the video can be found again (design section 12).
pub fn render_to_json(summary: &RenderSummary<'_>) -> Value {
    json!({
        "output_path": path_text(summary.result.output_path()),
        "frame_count": summary.result.frame_count(),
        "width": summary.result.width(),
        "height": summary.result.height(),
        "fps": RENDER_FPS,
        "backend": RENDER_BACKEND_NAME,
        "checkpoint_dir": path_text(summary.checkpoint_dir),
        "model_kind": summary.descriptor.model_kind.as_str(),
        "architecture_version": summary.descriptor.architecture_version.as_str(),
        "model_config_sha256": summary.descriptor.model_config_sha256.as_str(),
        "checkpoint_epoch": summary.checkpoint_epoch,
        "checkpoint_global_step": summary.checkpoint_global_step,
        "source_frame_count": summary.source_frame_count,
        "max_output_frames": summary.max_output_frames,
    })
}

fn path_text(path: &Path) -> String {
    path.display().to_string()
}
```

Declare `pub mod render_result;` in `src/lib.rs` and re-export both names next to `train_to_json`. Each result module keeps its own private `path_text`; that repetition is the existing convention in this crate, not an oversight.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-worker --test render_result
cargo fmt --all
cargo fmt --all -- --check
cargo clippy -p feathertalk-worker --all-targets -- -D warnings
```

Expected: the payload tests pass. `serde_json` serialises `Option<u64>` as `null`, so the absent cap needs no special case.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/src/render_result.rs rust/crates/feathertalk-worker/tests/render_result.rs rust/crates/feathertalk-worker/tests/support/mod.rs
git commit -m "feat(worker): report a render result"
```

---

### Task 7: Run the render loop

**Files:**

- Create: `rust/crates/feathertalk-worker/src/render.rs` (the observing sink and `run_render`; `execute_render` arrives in Task 8)
- Modify: `rust/crates/feathertalk-worker/src/lib.rs` (declare the module, export `run_render`)
- Test: `rust/crates/feathertalk-worker/tests/render.rs` (new)

**Interfaces:**

- Produces, private to the module: `ObservedSinkFactory` and `ObservedSink`, which wrap the caller's factory, count the frames written, report `TaskStage::Rendering { frame, total }` with a `Progress` for each one, and refuse the next frame once the token is cancelled.
- Produces: `pub fn run_render<M, R, F>(job: &RenderJob, model: &M, device: &RenderDevice, token: &CancellationToken, reporter: &dyn TaskReporter, frame_reader: &R, sink_factory: &F) -> CommandOutcome` with `M: TalkingHeadModel<RenderBackend>`, `R: FrameReader + ?Sized`, `F: RawVideoSinkFactory + ?Sized`.
- Consumes: `execute_offline_render`, `render_to_json`, `render_task_error`, `is_inference_cancellation`, `RenderJob`.

**Why now:** this is the part of the slice that can go wrong quietly. Progress that skips a frame, a cancellation that leaves a half-written mp4 behind, a failure reported at the wrong stage -- none of that shows up in a payload test. Building the loop against an in-memory sink, before any checkpoint or ffmpeg is involved, means the end-to-end test in Task 10 confirms a mechanism that is already known to work rather than discovering it.

`run_render` takes seven arguments, which is exactly clippy's limit, so no `allow` is needed. Keep it that way: anything more belongs in the `RenderJob`.

- [ ] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-worker/tests/render.rs`, reusing the Task 6 fixtures from `tests/support/mod.rs` plus the `Recorder` reporter that is already there.

```rust
#[test]
fn a_render_reports_one_progress_event_per_frame() {
    let (root, project_dir) = render_tree(2, 2);
    let job = job(&project_dir, root.path(), 2, None);
    let device = RenderDevice::default();
    let model = render_model(&device);
    let token = CancellationToken::new();
    let reporter = Recorder::default();
    let reader = StubFrameReader::default();
    let sinks = MemorySinkFactory::default();

    let outcome = run_render(&job, &model, &device, &token, &reporter, &reader, &sinks);

    let Some(payload) = completed(&outcome) else {
        panic!("{outcome:?}");
    };
    assert_eq!(payload["frame_count"], 2);
    // One event per written frame, in order, with the total from the manifest.
    let stages = reporter.stages();
    assert_eq!(
        stages,
        vec![
            TaskStage::Rendering { frame: 1, total: 2 },
            TaskStage::Rendering { frame: 2, total: 2 },
        ],
        "{stages:?}"
    );
    assert!(job.request.output_path().is_file());
}
```

Then, one test each:

- **Cancellation mid-render.** A token cancelled after the first frame -- the `Recorder` already supports running a closure on a step, so cancel from inside the reporter's second call, or use a sink that cancels the token once one frame is in. Assert the outcome is `CommandOutcome::Cancelled`, that the destination file does not exist, and that the staging path the factory recorded does not exist either: inference's guard removes it, and this is the assertion that proves it.
- **A failure keeps the stage.** Create the destination file first so `OfflineRenderRequest`'s output check fails as `OutputExists`; assert the outcome is `Failed` with `ErrorCode::MediaInvalid` and `TaskStage::Preparing`, because no frame was written.
- **A failure mid-loop reports the render stage.** A `StubFrameReader` that fails at frame 1 gives `ErrorCode::WorkerCrashed` (`FrameReader`) and `TaskStage::Rendering { frame: 1, total: 2 }` -- the stage of the last frame that got through, not `Preparing`.
- **A token cancelled before the first frame.** `run_render` returns `Cancelled` without touching the sink: assert the factory recorded no staging path at all.
- **A cap below the frame count.** `max_output_frames: Some(1)` over a two-frame project reports one event with `total: 1` and a payload `frame_count` of 1.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-worker --test render`
Expected: FAIL to compile -- `unresolved import feathertalk_worker::run_render`.

- [ ] **Step 3: Write minimal implementation**

Create `rust/crates/feathertalk-worker/src/render.rs`. The observing sink first:

```rust
/// Wraps the caller's sink factory so the render reports one progress event per
/// written frame and stops at the next frame once the task is cancelled.
///
/// The sink borrows the reporter, which is why `RawVideoSinkFactory` no longer
/// requires `Send + Sync` (Task 1): a `TaskReporter` is not `Sync`, and the
/// render loop runs on this thread anyway.
struct ObservedSinkFactory<'a, F: ?Sized> {
    inner: &'a F,
    reporter: &'a dyn TaskReporter,
    token: &'a CancellationToken,
    /// The frames the render will write, from the locked manifest.
    total: u64,
    /// The frames it has written so far. Interior mutability because `start` and
    /// `write_frame` only ever get a shared borrow of the factory.
    frames: AtomicU64,
}

impl<F: ?Sized> ObservedSinkFactory<'_, F> {
    /// Counts one written frame and reports it.
    fn observe(&self) {
        let frame = self.frames.fetch_add(1, Ordering::Relaxed).saturating_add(1);
        self.reporter.report(
            TaskStage::Rendering {
                frame,
                total: self.total,
            },
            Some(Progress {
                completed: frame.min(self.total),
                total: Some(self.total),
            }),
        );
    }

    /// The stage a failure happened at: `Preparing` while nothing has been
    /// written, the last reported frame once the loop is running.
    fn stage(&self) -> TaskStage {
        let frame = self.frames.load(Ordering::Relaxed);
        if frame == 0 {
            TaskStage::Preparing
        } else {
            TaskStage::Rendering {
                frame,
                total: self.total,
            }
        }
    }
}

struct ObservedSink<'a, F: ?Sized> {
    inner: Box<dyn RawVideoSink + 'a>,
    observer: &'a ObservedSinkFactory<'a, F>,
}

impl<F: RawVideoSinkFactory + ?Sized> RawVideoSinkFactory for ObservedSinkFactory<'_, F> {
    fn start(&self, command: &CommandSpec) -> Result<Box<dyn RawVideoSink + '_>, InferenceError> {
        Ok(Box::new(ObservedSink {
            inner: self.inner.start(command)?,
            observer: self,
        }))
    }
}

impl<F: RawVideoSinkFactory + ?Sized> RawVideoSink for ObservedSink<'_, F> {
    fn write_frame(&mut self, frame: &BgrFrame) -> Result<(), InferenceError> {
        // Checked before the write, so a cancelled task stops at a frame
        // boundary rather than half a frame into the encoder's stdin.
        if self.observer.token.is_cancelled() {
            return Err(InferenceError::Cancelled {
                operation: "render",
            });
        }
        self.inner.write_frame(frame)?;
        self.observer.observe();
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<(), InferenceError> {
        // Destructured rather than moved out field by field: `finish` consumes
        // the boxed inner sink, and `Self` is not `Copy`.
        let ObservedSink { inner, .. } = *self;
        inner.finish()
    }
}
```

Then the loop itself:

```rust
/// Renders every planned frame, reporting one progress event per frame.
///
/// Inference owns the loop, so the wrapper sink is the only place cancellation
/// can act: a failing `write_frame` is how a caller stops it, and inference's
/// own guard then removes the staging file (design section 6).
pub fn run_render<M, R, F>(
    job: &RenderJob,
    model: &M,
    device: &RenderDevice,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    frame_reader: &R,
    sink_factory: &F,
) -> CommandOutcome
where
    M: TalkingHeadModel<RenderBackend>,
    R: FrameReader + ?Sized,
    F: RawVideoSinkFactory + ?Sized,
{
    // A task cancelled before the first frame never starts the encoder.
    if token.is_cancelled() {
        return CommandOutcome::Cancelled;
    }
    let observed = ObservedSinkFactory {
        inner: sink_factory,
        reporter,
        token,
        total: job.progress_total,
        frames: AtomicU64::new(0),
    };
    match execute_offline_render::<RenderBackend, M, R, ObservedSinkFactory<'_, F>>(
        model,
        device,
        &job.request,
        frame_reader,
        &observed,
    ) {
        Ok(result) => CommandOutcome::Completed(Some(render_to_json(&RenderSummary {
            result: &result,
            descriptor: &job.descriptor,
            checkpoint_dir: &job.checkpoint_dir,
            checkpoint_epoch: job.checkpoint_epoch,
            checkpoint_global_step: job.checkpoint_global_step,
            source_frame_count: job.source_frame_count,
            max_output_frames: job.max_output_frames,
        }))),
        // A cancelled render is not a failure: no error event, no output file.
        Err(error) if is_inference_cancellation(&error) => CommandOutcome::Cancelled,
        Err(error) => CommandOutcome::Failed(render_task_error(&error, observed.stage())),
    }
}
```

Imports: `std::sync::atomic::{AtomicU64, Ordering}`, `feathertalk_domain::{Progress, TaskStage}`, `feathertalk_inference::{BgrFrame, CommandSpec, FrameReader, InferenceError, RawVideoSink, RawVideoSinkFactory, execute_offline_render}`, `feathertalk_media::CancellationToken`, `feathertalk_models::unet::TalkingHeadModel`, and the crate's own `CommandOutcome`, `RenderJob`, `RenderSummary`, `render_to_json`, `render_task_error`, `is_inference_cancellation`, `RenderBackend`, `RenderDevice`, `TaskReporter`.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-worker --test render
cargo test -p feathertalk-worker --all-targets
cargo fmt --all
cargo fmt --all -- --check
cargo clippy -p feathertalk-worker --all-targets -- -D warnings
```

Expected: every render test passes. If the borrow checker rejects `observer: self`, the two lifetimes on `ObservedSink` were spelled apart: one lifetime parameter shared by the box and the back-reference is what makes the variance work out.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/src/render.rs rust/crates/feathertalk-worker/tests/render.rs
git commit -m "feat(worker): run the render loop"
```

---

### Task 8: Execute the render command

**Files:**

- Modify: `rust/crates/feathertalk-worker/src/render.rs` (append `execute_render`)
- Modify: `rust/crates/feathertalk-worker/src/commands.rs` (the `Request::Render` arm)
- Modify: `rust/crates/feathertalk-worker/src/lib.rs` (export `execute_render`)
- Test: `rust/crates/feathertalk-worker/tests/render.rs` (append admission and dispatch tests)

**Interfaces:**

- Produces: `pub fn execute_render<R, F>(params: &RenderParams, token: &CancellationToken, reporter: &dyn TaskReporter, toolchain: &MediaToolchain, frame_reader: &R, sink_factory: &F) -> CommandOutcome`.
- Produces: the `Request::Render(params)` arm of `execute_with_runner`, placed before `other =>` and guarded the way `Request::Train` is.
- Consumes: `check_project_dir`, `check_render_paths`, `validate_project_dir`, `read_training_checkpoint`, `render_variant`, `checkpoint_descriptor`, `render_job`, `load_training_checkpoint_model`, `run_render`, `project_task_error`, `training_task_error`.

**Why now:** everything it calls exists and is tested, so this task is the wiring and the order of the wiring. The order matters: the cheap rejections come before the expensive reads, and the checkpoint is read before the model template is built because the template's shape is written in the checkpoint's own manifest.

The command is generic over the reader and the sink so a test can render without ffmpeg; `commands.rs` passes the real pair. That is the same reason `execute_train` takes its toolchain by reference instead of reading the environment.

- [ ] **Step 1: Write the failing test**

Append to `rust/crates/feathertalk-worker/tests/render.rs`. A `MediaToolchain` needs two absolute paths and a timeout, and it never runs here because the sink is in memory:

```rust
fn toolchain(root: &Path) -> MediaToolchain {
    MediaToolchain::new(
        root.join("ffmpeg.exe"),
        root.join("ffprobe.exe"),
        Duration::from_secs(30),
    )
    .expect("two absolute paths make a toolchain")
}
```

Cover:

- **A relative checkpoint is refused.** `ErrorCode::MediaInvalid`, `TaskStage::Preparing`, and no `Rendering` stage was ever reported.
- **A project that is not locked is refused.** An empty temp directory gives the project error's own code, and the failure comes from `project_task_error` rather than from inference.
- **A checkpoint whose `model_kind` is unknown is refused** with `ErrorCode::ModelIncompatible` and a detail containing the kind that was read, so the operator learns which name failed. Write the manifest by hand for this one.
- **`max_output_frames: Some(0)` is refused** before anything is read.
- **A cancelled token returns `Cancelled`** without reading the checkpoint.

A full success path through `execute_render` needs a real locked project and a real checkpoint, which is the gated end-to-end test in Task 10; these five are the admission gates, and they need neither.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-worker --test render`
Expected: FAIL to compile -- `unresolved import feathertalk_worker::execute_render`.

- [ ] **Step 3: Write minimal implementation**

Append to `rust/crates/feathertalk-worker/src/render.rs`:

```rust
/// Renders a locked project with one of its own training checkpoints.
///
/// The order is deliberate (design section 10): the cheap rejections first, then
/// the project manifest, then the checkpoint's own manifest -- which is where the
/// model variant is written, so the template cannot be built any earlier.
pub fn execute_render<R, F>(
    params: &RenderParams,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    toolchain: &MediaToolchain,
    frame_reader: &R,
    sink_factory: &F,
) -> CommandOutcome
where
    R: FrameReader + ?Sized,
    F: RawVideoSinkFactory + ?Sized,
{
    // Reading the checkpoint means reading a Burn record off the disk, so the
    // stage goes out before any of it.
    reporter.report(TaskStage::Preparing, None);
    if let Err(error) = check_project_dir(&params.project_dir) {
        return CommandOutcome::Failed(error);
    }
    if let Err(error) = check_render_paths(params) {
        return CommandOutcome::Failed(error);
    }
    if let Err(error) = check_max_output_frames(params.max_output_frames) {
        return CommandOutcome::Failed(error);
    }
    let project = match validate_project_dir(&params.project_dir) {
        Ok(project) => project,
        Err(error) => return CommandOutcome::Failed(project_task_error(&error)),
    };
    // The locked manifest owns the frame count. The feature file is read by
    // inference, but it never decides how many frames the client was promised.
    let frame_count = project.asset_package().manifest().frame_count;
    let metadata = match read_training_checkpoint(&params.checkpoint) {
        Ok(metadata) => metadata,
        Err(error) => {
            return CommandOutcome::Failed(training_task_error(&error, TaskStage::Preparing));
        }
    };
    let Some(variant) = render_variant(&metadata.manifest.model_kind) else {
        return CommandOutcome::Failed(TaskError::new(
            ErrorCode::ModelIncompatible,
            "检查点的模型类型不受支持",
            &format!("unsupported model_kind: {}", metadata.manifest.model_kind),
            TaskStage::Preparing,
        ));
    };
    let descriptor = checkpoint_descriptor(&variant.configuration());
    let job = match render_job(
        params,
        frame_count,
        toolchain.ffmpeg(),
        descriptor,
        metadata.state.epoch,
        metadata.state.global_step,
    ) {
        Ok(job) => job,
        Err(error) => return CommandOutcome::Failed(error),
    };

    // The record was written by modules on `Autodiff<NdArray>`, so it is read
    // back by the same types and the autodiff shell is dropped with `valid`.
    let load_device = TrainDevice::default();
    let device = RenderDevice::default();
    match variant {
        RenderVariant::OriginalUnet(configuration) => {
            let template = configuration.init::<TrainBackend>(&load_device);
            let restored = match load_training_checkpoint_model::<TrainBackend, _>(
                &params.checkpoint,
                &template,
                &load_device,
                &job.descriptor,
            ) {
                Ok(restored) => restored,
                Err(error) => {
                    return CommandOutcome::Failed(training_task_error(
                        &error,
                        TaskStage::Preparing,
                    ));
                }
            };
            // Loading half a gigabyte of weights is the longest step before the
            // first frame, so the token is checked again on the far side of it.
            if token.is_cancelled() {
                return CommandOutcome::Cancelled;
            }
            let model = restored.model.valid();
            run_render(
                &job, &model, &device, token, reporter, frame_reader, sink_factory,
            )
        }
        RenderVariant::MobileOneUnet(configuration) => {
            let template = configuration.init::<TrainBackend>(&load_device);
            let restored = match load_training_checkpoint_model::<TrainBackend, _>(
                &params.checkpoint,
                &template,
                &load_device,
                &job.descriptor,
            ) {
                Ok(restored) => restored,
                Err(error) => {
                    return CommandOutcome::Failed(training_task_error(
                        &error,
                        TaskStage::Preparing,
                    ));
                }
            };
            if token.is_cancelled() {
                return CommandOutcome::Cancelled;
            }
            // Inference fuses the multi-branch blocks; training needed them
            // separate, which is why the descriptor above describes the
            // unfused shape.
            let model = restored.model.valid().reparameterize();
            run_render(
                &job, &model, &device, token, reporter, frame_reader, sink_factory,
            )
        }
    }
}
```

In `rust/crates/feathertalk-worker/src/commands.rs`, before the `other =>` arm:

```rust
        Request::Render(params) => {
            let Some(toolchain) = config.media() else {
                // Unreachable through the runtime, which rejects `render` when no
                // toolchain is configured; kept so a direct caller gets an error
                // rather than a panic.
                return CommandOutcome::Failed(unsupported(request.kind()));
            };
            execute_render(
                params,
                token,
                reporter,
                toolchain,
                &JpegFrameReader::default(),
                &SystemRawVideoSinkFactory,
            )
        }
```

The two variant arms differ by three lines, and factoring them into one generic
helper would need a `FnOnce(&TrainDevice) -> M` plus an `M: AutodiffModule` bound
whose `InnerModule` is a `TalkingHeadModel` -- a bound that cannot be written
without naming both models. `execute_train::start` gets away with it because
training keeps the autodiff module; rendering has to leave it. Repeat the arms.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-worker --all-targets
cargo fmt --all
cargo fmt --all -- --check
cargo clippy -p feathertalk-worker --all-targets -- -D warnings
```

Expected: the whole worker package passes. If `restored.model.valid()` does not resolve, `AutodiffModule` is not in scope: import it from `burn::module`.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/src/commands.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/src/render.rs rust/crates/feathertalk-worker/tests/render.rs
git commit -m "feat(worker): execute the render command"
```

---

### Task 9: Add the render subcommand

**Files:**

- Modify: `rust/crates/feathertalk-cli/src/cli.rs` (the `Command` enum, after `Train`)
- Modify: `rust/crates/feathertalk-cli/src/run.rs` (`build_request` and its inline tests)
- Modify: `rust/crates/feathertalk-cli/src/render.rs` (the unsupported-command hint)
- Test: the inline `#[cfg(test)]` module in `run.rs`

**Interfaces:**

- Produces: `feathertalk render <PROJECT_DIR> <CHECKPOINT> <AUDIO> <OUTPUT> [--max-output-frames N]`.
- Produces: the `Command::Render` arm of `build_request`, rejecting an empty value for each of the four paths.
- Produces: the `render` branch of the unsupported-command hint, naming `FEATHERTALK_WORKER_FFPROBE` and `FEATHERTALK_WORKER_FFMPEG`.
- Consumes: `RenderParams`, `reject_empty`, and the stage label for `TaskStage::Rendering`, which `src/render.rs:38` already prints as 「正在渲染 第 {frame}/{total} 帧」.

**Why now:** the worker serves the command, so this is the last piece of the loop. Nothing about the CLI's progress printing changes: the label existed before this slice, and Task 10 is what finally proves it prints.

- [ ] **Step 1: Write the failing test**

In the inline test module of `rust/crates/feathertalk-cli/src/run.rs`, mirroring `train_refuses_an_empty_project_directory` and `train_carries_every_flag_into_the_request`:

```rust
#[test]
fn render_refuses_an_empty_output_path() {
    let error = build_request(&Command::Render {
        project_dir: PathBuf::from("project"),
        checkpoint: PathBuf::from("checkpoint"),
        audio: PathBuf::from("voice.wav"),
        output: PathBuf::new(),
        max_output_frames: None,
    })
    .expect_err("an empty output path is refused");

    assert!(error.contains("输出文件"), "{error}");
}

#[test]
fn render_carries_every_flag_into_the_request() {
    let request = build_request(&Command::Render {
        project_dir: PathBuf::from("project"),
        checkpoint: PathBuf::from("project/models/unet/checkpoint-00000004"),
        audio: PathBuf::from("voice.wav"),
        output: PathBuf::from("preview.mp4"),
        max_output_frames: Some(2),
    })
    .expect("a complete render command is a request");

    let Some(Request::Render(params)) = request else {
        panic!("{request:?}");
    };
    assert_eq!(params.project_dir, PathBuf::from("project"));
    assert_eq!(params.audio, PathBuf::from("voice.wav"));
    assert_eq!(params.output, PathBuf::from("preview.mp4"));
    // The CLI passes the cap through untouched: whether zero or a huge value is
    // acceptable is the worker's judgement, like every path here.
    assert_eq!(params.max_output_frames, Some(2));
}
```

Add one more test for each of the other three empty paths, and extend the existing help-text test if the file has one that lists the subcommands.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p feathertalk-cli --lib`
Expected: FAIL to compile -- `error[E0599]: no variant named Render found for enum Command`.

- [ ] **Step 3: Write minimal implementation**

In `rust/crates/feathertalk-cli/src/cli.rs`, after the `Train` variant:

```rust
    /// 渲染视频：用检查点权重逐帧推理，并混入指定音轨
    Render {
        /// 工程目录
        project_dir: PathBuf,
        /// 检查点目录，例如 models/unet/checkpoint-00000004
        checkpoint: PathBuf,
        /// 混入输出视频的音频文件
        audio: PathBuf,
        /// 输出的 mp4 文件，不能已存在
        output: PathBuf,
        /// 最多渲染多少帧，默认渲染整个工程
        #[arg(long, value_name = "N")]
        max_output_frames: Option<u64>,
    },
```

In `rust/crates/feathertalk-cli/src/run.rs`, after the `Command::Train` arm:

```rust
        Command::Render {
            project_dir,
            checkpoint,
            audio,
            output,
            max_output_frames,
        } => {
            reject_empty(project_dir, "工程目录")?;
            reject_empty(checkpoint, "检查点目录")?;
            reject_empty(audio, "音频文件")?;
            reject_empty(output, "输出文件")?;
            Ok(Some(Request::Render(RenderParams {
                project_dir: project_dir.clone(),
                checkpoint: checkpoint.clone(),
                audio: audio.clone(),
                output: output.clone(),
                max_output_frames: *max_output_frames,
            })))
        }
```

In `rust/crates/feathertalk-cli/src/render.rs`, extend the `UnsupportedCommand` chain (lines 279-315) with a branch for `render` that names both media variables, the way the media commands' branch does:

```rust
    } else if *requested == "render" {
        format!(
            "工作进程不支持 render：请设置 {ENV_WORKER_FFPROBE} 与 {ENV_WORKER_FFMPEG} 后重试。"
        )
```

Match the surrounding branch's exact wording rather than this sketch; read them first.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-cli
cargo fmt --all
cargo fmt --all -- --check
cargo clippy -p feathertalk-cli --all-targets -- -D warnings
```

Expected: the CLI package passes, including the gated end-to-end tests skipping.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-cli/src/cli.rs rust/crates/feathertalk-cli/src/render.rs rust/crates/feathertalk-cli/src/run.rs
git commit -m "feat(cli): add the render subcommand"
```

---

### Task 10: Render a locked project end to end

**Files:**

- Test: `rust/crates/feathertalk-cli/tests/real_worker.rs` (append one gated test and two constants)

**Interfaces:**

- Consumes the helpers already in the file: `worker_or_skip`, `real_tool`, `real_dir`, `demo_clip`, `cut_audio`, `cut_one_second`, `write_frame_fixtures`, `PROJECT_MANIFEST`, `run`, `code`, `stdout`, `stderr`.
- Produces no library code. This task proves the slice, it does not add to it.

**Why now:** everything above is tested against stubs. Nothing so far has run a real ffmpeg, decoded a real JPEG, read a real Burn record or produced a file a person can play. This test is the only place where the four commands meet: extract features, lock, train one epoch, render the checkpoint that training just wrote.

It skips unless `FEATHERTALK_REQUIRE_E2E`, both ffmpeg tools, the FeatherHuBERT package, the VGG19 package and `demo/feathertalk_demo_latest_188.mp4` are all present -- the same contract the training test has.

- [ ] **Step 1: Write the failing test**

Append to `rust/crates/feathertalk-cli/tests/real_worker.rs`:

```rust
/// The frames the render test locks, and half the tokens
/// `RENDERED_AUDIO_SECONDS` extracts, so the lock trims nothing.
const RENDERED_FRAME_COUNT: u64 = 2;

/// 0.09 s at 16 kHz is 1 440 samples, which the 400-sample kernel and the
/// 320-sample stride turn into `(1 440 - 80) / 320` = 4 FeatherHuBERT frames and
/// so into 4 tokens, which is two video frames. Every cut from 1 360 to 1 679
/// samples yields the same 4, which is the slack a resampler gets.
const RENDERED_AUDIO_SECONDS: &str = "0.09";
```

Then the test, following `a_real_project_is_trained_end_to_end` step for step through `extract-features`, `write_frame_fixtures`, `lock-asset-package` and `train --mode baseline --epochs 1`, with `RENDERED_FRAME_COUNT` in place of `TRAINED_FRAME_COUNT` and `extracted["tokens"] == 4`. Training one epoch over two frames takes two steps, so the checkpoint is `checkpoint-00000002`:

```rust
    let checkpoint = project
        .path()
        .join("models")
        .join("unet")
        .join("checkpoint-00000002");
    assert!(checkpoint.join("manifest.json").is_file());

    let output_file = project.path().join("preview.mp4");
    let checkpoint_arg = checkpoint.to_string_lossy().into_owned();
    let output_arg = output_file.to_string_lossy().into_owned();
    let ffmpeg_arg = ffmpeg.to_string_lossy().into_owned();
    let ffprobe_arg = ffprobe.to_string_lossy().into_owned();
    let output = run(
        &worker,
        &[
            "render",
            &project_arg,
            &checkpoint_arg,
            &audio_arg,
            &output_arg,
        ],
        &[
            ("FEATHERTALK_WORKER_FFMPEG", ffmpeg_arg.as_str()),
            ("FEATHERTALK_WORKER_FFPROBE", ffprobe_arg.as_str()),
        ],
    );
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));
    let rendered: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("stdout is exactly one JSON document");

    assert_eq!(rendered["output_path"], output_file.display().to_string());
    assert_eq!(rendered["frame_count"], RENDERED_FRAME_COUNT);
    // The committed frame fixture is 1280x720, and the render pastes the
    // generated mouth back into the full frame rather than cropping it.
    assert_eq!(rendered["width"], 1280);
    assert_eq!(rendered["height"], 720);
    assert_eq!(rendered["fps"], 25);
    assert_eq!(rendered["backend"], "ndarray-cpu");
    assert_eq!(rendered["checkpoint_dir"], checkpoint.display().to_string());
    assert_eq!(rendered["model_kind"], "original_unet");
    assert_eq!(rendered["model_config_sha256"].as_str().unwrap().len(), 64);
    assert_eq!(rendered["checkpoint_epoch"], 1);
    assert_eq!(rendered["checkpoint_global_step"], RENDERED_FRAME_COUNT);
    assert_eq!(rendered["source_frame_count"], RENDERED_FRAME_COUNT);
    assert_eq!(rendered["max_output_frames"], serde_json::Value::Null);

    // A file a person can play: the atomic publish renamed the staging file into
    // place, so this path is the mp4 and not a leftover fragment.
    let published = std::fs::metadata(&output_file).expect("the render published its output");
    assert!(published.is_file());
    assert!(published.len() > 0, "{}", published.len());

    let narration = stderr(&output);
    assert!(narration.contains("正在渲染"), "{narration}");
    assert!(narration.contains("进度 2/2"), "{narration}");
```

Add `real_tool("FFPROBE")` to the tuple the test destructures at the top, and name `FEATHERTALK_WORKER_FFPROBE` in the skip message.

- [ ] **Step 2: Run test to verify it fails**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-cli --test real_worker
```

Expected: PASS by skipping, because `FEATHERTALK_REQUIRE_E2E` is not set in a plain run. That is the point of the gate; the real verification is Step 4.

- [ ] **Step 3: Write minimal implementation**

No implementation. If this test fails against the real tools, the fix belongs in whichever earlier task owns the behaviour, not here.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo build --release -p feathertalk-worker -p feathertalk-cli
$env:FEATHERTALK_REQUIRE_E2E = "1"
$env:FEATHERTALK_WORKER_FFMPEG = "D:\environment\ffmpeg\bin\ffmpeg.exe"
$env:FEATHERTALK_WORKER_FFPROBE = "D:\environment\ffmpeg\bin\ffprobe.exe"
$env:FEATHERTALK_WORKER_HUBERT_DIR = "$env:TEMP\ft_hubert_e2e\package"
$env:FEATHERTALK_WORKER_VGG19_DIR = "$env:TEMP\ft_vgg19_e2e\package"
cargo test --release -p feathertalk-cli --test real_worker -- --nocapture *> "$env:TEMP\ft_render_e2e.log"
Select-String -Path "$env:TEMP\ft_render_e2e.log" -Pattern "test result:|panicked at|skipping"
```

Expected: `a_real_project_is_rendered_end_to_end` passes; the two tests that need SCRFD and PFLD skip, which they already do. Then the full sweep:

```powershell
cargo test --workspace --all-targets
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-cli/tests/real_worker.rs
git commit -m "test(cli): render a locked project end to end"
```

---

## Done When

- `feathertalk render <project> <checkpoint> <audio> <output>` writes a playable mp4 from a locked project and one of its own checkpoints, printing 「正在渲染 第 n/N 帧」 as it goes.
- The handshake lists `render` whenever the media toolchain is configured, and a worker without it rejects the command by naming both environment variables.
- A cancelled render leaves no output file and reports `cancelled`, not an error.
- The completed payload names the weights the video came from: the checkpoint directory, its epoch and global step, and the model kind, architecture version and configuration digest.
- `feathertalk-domain` is untouched.
