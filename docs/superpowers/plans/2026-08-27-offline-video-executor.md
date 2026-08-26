# Offline Video Executor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a complete, offline, failure-atomic video executor to `feathertalk-inference` that reads published frame/landmark/feature artifacts, renders every planned frame through the existing Burn and BGR kernels, streams raw BGR24 to FFmpeg stdin, and atomically publishes the output.

**Architecture:** Keep the executor in `feathertalk-inference`. `executor.rs` owns request validation, artifact lookup, `RenderPlan` consumption, and staging publication; `frame_reader.rs` owns JPEG-to-BGR decoding behind an injectable `FrameReader`; `raw_sink.rs` owns an injectable raw-video sink plus the production FFmpeg process adapter. Existing `RenderPlan`, `read_feature_file`, `read_landmarks`, `compute_face_bbox`, `render_planned_frame`, and `raw_video_command` remain the sole sources of media/model semantics.

**Tech Stack:** Rust 1.92 edition 2024, Burn 0.21, `jpeg-decoder` 0.3 for production JPEG decoding, standard-library `std::process`, `std::fs`, `Path`, `thiserror`, `tempfile`, and existing FeatherTalk crates.

## Global Constraints

- Work in isolated worktree `.worktrees/offline-video-executor` on branch `offline-video-executor`; do not modify protected worktrees `frame-face-pipeline`, `media-normalization-execution`, or `pfld-burn-inference`.
- Preserve the untracked user-owned directory `demo/kanghui_training_video_featherhubert_188_latest/`; never stage, modify, or read its `kanghui_training_video.MOV`.
- The model crate must not depend on the weights crate; the executor must receive an already constructed `TalkingHeadModel` and Burn device.
- All new production functions require a failing test before implementation; use focused red/green/refactor cycles.
- Use `apply_patch` for source edits, stage explicit paths only, and run `git diff --check` after every task.
- No shell command construction, PATH lookup, silent WGPU-to-CPU fallback, broad directory scans, or deletion of files not created by the current invocation.
- Input frames are fixed `{index:06}.jpg`, landmarks `{index:06}.lms`; feature dimensions are exactly `1024`, token count is positive and even, and output FPS is exactly `25`.
- Output publication is same-directory staging plus atomic rename. Existing destination files are rejected and remain byte-for-byte unchanged on every failure.

---

## File Map

Create or modify only these implementation files and their focused tests:

- Modify: `rust/Cargo.toml` (workspace member) and generated `rust/Cargo.lock`.
- Modify: `rust/crates/feathertalk-inference/Cargo.toml` (JPEG decoder dependency).
- Create: `rust/crates/feathertalk-inference/src/frame_reader.rs` (reader trait and JPEG adapter).
- Create: `rust/crates/feathertalk-inference/src/raw_sink.rs` (sink traits, FFmpeg process adapter).
- Create: `rust/crates/feathertalk-inference/src/executor.rs` (request/result and orchestration).
- Modify: `rust/crates/feathertalk-inference/src/error.rs` and `src/lib.rs` (errors and exports).
- Create: `rust/crates/feathertalk-inference/tests/frame_reader.rs`.
- Create: `rust/crates/feathertalk-inference/tests/raw_sink.rs`.
- Create: `rust/crates/feathertalk-inference/tests/executor.rs`.
- Modify: `rust/crates/feathertalk-inference/tests/public_api.rs`.
- Create: `rust/crates/feathertalk-inference/tests/support/mod.rs` helpers only if existing support cannot be reused.

No files under `demo/` are in scope.

---

### Task 1: Define executor request/result and structured errors

**Files:**
- Modify: `rust/crates/feathertalk-inference/src/error.rs`
- Create: `rust/crates/feathertalk-inference/src/executor.rs`
- Modify: `rust/crates/feathertalk-inference/src/lib.rs`
- Test: `rust/crates/feathertalk-inference/tests/executor.rs`

**Interfaces:**
- Produces `OfflineRenderRequest::new(...)`, path/count accessors, `OfflineRenderResult` accessors, and `validate()` used by later tasks.
- Produces error variants `InvalidInputArtifact`, `InvalidInputDirectory`, `FrameIndexOutOfRange`, `FrameDimensionsMismatch`, `FrameReader`, `SinkStart`, `SinkWrite`, `SinkFinish`, `StagingCollision`, `StagingOutputInvalid`, `AtomicPublishFailed`, and `ToolFailed` with structured paths/indexes and bounded messages.

- [ ] **Step 1: Write the failing request tests.**

```rust
use std::path::PathBuf;
use feathertalk_inference::{InferenceError, OfflineRenderRequest};

fn valid(root: &std::path::Path) -> OfflineRenderRequest {
    OfflineRenderRequest::new(
        root.join("frames"), root.join("landmarks"), root.join("features.f32"),
        root.join("audio.wav"), root.join("ffmpeg.exe"), root.join("result.mp4"),
        "task-01", 2, None,
    ).unwrap()
}

#[test]
fn request_rejects_relative_paths_and_invalid_counts() {
    let root = tempfile::tempdir().unwrap();
    let request = valid(root.path());
    assert_eq!(request.task_id(), "task-01");
    assert_eq!(request.source_frame_count(), 2);
    assert!(matches!(
        OfflineRenderRequest::new(
            PathBuf::from("frames"), root.path().into(), root.path().join("f"),
            root.path().join("a"), root.path().join("ffmpeg"), root.path().join("o"),
            "x", 2, None
        ),
        Err(InferenceError::InvalidField { field: "frame_dir", .. })
    ));
    assert!(matches!(
        OfflineRenderRequest::new(
            root.path().join("f"), root.path().join("l"), root.path().join("f"),
            root.path().join("a"), root.path().join("ffmpeg"), root.path().join("o"),
            "x", 1, None
        ),
        Err(InferenceError::FrameCountTooSmall { .. })
    ));
    assert!(matches!(
        OfflineRenderRequest::new(
            root.path().join("f"), root.path().join("l"), root.path().join("f"),
            root.path().join("a"), root.path().join("ffmpeg"), root.path().join("o"),
            "x", 2, Some(0)
        ),
        Err(InferenceError::InvalidField { field: "max_output_frames", .. })
    ));
}

```

- [ ] **Step 2: Run the focused test and verify the expected missing-symbol failure.**

Run from `rust/`:

```powershell
cargo test -p feathertalk-inference --test executor request_rejects_relative_paths_and_invalid_counts
```

Expected: compilation fails because `OfflineRenderRequest` and the new error variants do not yet exist.

- [ ] **Step 3: Implement the minimal request/result types and errors.**

`OfflineRenderRequest::new` must require non-empty absolute paths, reject `source_frame_count < 2`, reject `Some(0)`, validate the task id by calling the existing staging helper against the requested output, and store no derived mutable state. Add accessors for every path, task id, count, and limit. Define `OfflineRenderResult` now with a `pub(crate)` constructor and public accessors; Task 4 verifies those accessors through a result returned by `execute_offline_render`. Keep error messages bounded to 512 characters; preserve source paths and frame indexes as fields.

- [ ] **Step 4: Run focused tests and lint.**

```powershell
cargo test -p feathertalk-inference --test executor
cargo clippy -p feathertalk-inference --all-targets -- -D warnings
git diff --check
```

- [ ] **Step 5: Commit.**

```powershell
git add rust/crates/feathertalk-inference/src/error.rs rust/crates/feathertalk-inference/src/executor.rs rust/crates/feathertalk-inference/src/lib.rs rust/crates/feathertalk-inference/tests/executor.rs
git commit -m "feat: define offline render executor contract"
```

---

### Task 2: Add bounded JPEG frame reader and BGR conversion

**Files:**
- Modify: `rust/crates/feathertalk-inference/Cargo.toml`
- Modify: `rust/Cargo.toml` and `rust/Cargo.lock`
- Create: `rust/crates/feathertalk-inference/src/frame_reader.rs`
- Modify: `rust/crates/feathertalk-inference/src/error.rs` and `src/lib.rs`
- Test: `rust/crates/feathertalk-inference/tests/frame_reader.rs`

**Interfaces:**
- `pub trait FrameReader: Send + Sync { fn read(&self, index: usize, path: &Path) -> Result<BgrFrame, InferenceError>; }`
- `pub struct JpegFrameReader { max_pixels: u64 }` with `new(max_pixels: u64)`, `Default`, and `read`.

- [ ] **Step 1: Write the failing reader tests.**

Embed a deterministic tiny JPEG fixture in the test source (base64-decoded bytes or a literal byte array); do not read any demo asset:

```rust
#[test]
fn jpeg_reader_decodes_rgb_as_bgr() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("000000.jpg");
    std::fs::write(&path, KNOWN_1X1_JPEG).unwrap();
    let frame = JpegFrameReader::default().read(0, &path).unwrap();
    assert_eq!((frame.width(), frame.height()), (1, 1));
    assert_eq!(frame.as_bytes(), &[EXPECTED_B, EXPECTED_G, EXPECTED_R]);
}
```

Also cover corrupt bytes, zero max-pixel configuration, compressed input over 16 MiB, and a symlink when the platform permits creating one.

- [ ] **Step 2: Run the focused tests and confirm they fail because the reader is absent.**

```powershell
cargo test -p feathertalk-inference --test frame_reader
```

- [ ] **Step 3: Implement `JpegFrameReader`.**

Open with `symlink_metadata`; require a regular non-symlink file, reject files larger than a fixed 16 MiB compressed-input limit, read bytes with a bounded buffer, decode using `jpeg_decoder::Decoder`, require RGB output and non-zero dimensions, check `width * height <= max_pixels` using checked arithmetic, and convert each RGB triple to BGR in row-major order. Return `FrameReader` errors with index/path and no panic text. Never follow a symlink.

- [ ] **Step 4: Run tests, format, lint, and diff check.**

```powershell
cargo fmt --all -- --check
cargo test -p feathertalk-inference --test frame_reader
cargo clippy -p feathertalk-inference --all-targets -- -D warnings
git diff --check
```

- [ ] **Step 5: Commit.**

```powershell
git add rust/Cargo.toml rust/Cargo.lock rust/crates/feathertalk-inference/Cargo.toml rust/crates/feathertalk-inference/src/frame_reader.rs rust/crates/feathertalk-inference/src/error.rs rust/crates/feathertalk-inference/src/lib.rs rust/crates/feathertalk-inference/tests/frame_reader.rs
git commit -m "feat: add bounded JPEG BGR frame reader"
```

---

### Task 3: Implement injectable raw-video sinks and production FFmpeg process

**Files:**
- Create: `rust/crates/feathertalk-inference/src/raw_sink.rs`
- Modify: `rust/crates/feathertalk-inference/src/error.rs` and `src/lib.rs`
- Test: `rust/crates/feathertalk-inference/tests/raw_sink.rs`

**Interfaces:**
- `pub trait RawVideoSink { fn write_frame(&mut self, frame: &BgrFrame) -> Result<(), InferenceError>; fn finish(self: Box<Self>) -> Result<(), InferenceError>; }`
- `pub trait RawVideoSinkFactory: Send + Sync { fn start(&self, command: &CommandSpec) -> Result<Box<dyn RawVideoSink>, InferenceError>; }`
- `pub struct SystemRawVideoSinkFactory` with `new()` and `Default`.

- [ ] **Step 1: Write failing fake-sink and helper-process tests.**

Test a fake sink that records bytes and a helper executable (the current test binary with an ignored helper test) that consumes stdin and writes a marker file. Assert that `write_frame` emits exact `BgrFrame::as_bytes()`, `finish` closes stdin, and a non-zero child returns `InferenceError::ToolFailed` with bounded stderr.

```rust
#[test]
fn system_sink_streams_exact_bgr_bytes_and_waits_for_success() {
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("marker.bin");
    let command = helper_command("sink_success", &marker);
    let factory = SystemRawVideoSinkFactory::default();
    let mut sink = factory.start(&command).unwrap();
    let frame = BgrFrame::new(2, 1, vec![1, 2, 3, 4, 5, 6]).unwrap();
    sink.write_frame(&frame).unwrap();
    sink.finish().unwrap();
    assert_eq!(std::fs::read(marker).unwrap(), frame.as_bytes());
}
```

- [ ] **Step 2: Run tests and verify missing-symbol/behavior failure.**

```powershell
cargo test -p feathertalk-inference --test raw_sink
```

- [ ] **Step 3: Implement `SystemRawVideoSinkFactory` and process lifecycle.**

Validate the command executable as an absolute regular non-symlink file before spawn. Spawn with piped stdin/stderr and null stdout; drain stderr in a thread capped at `MAX_CAPTURE_BYTES` (reuse or mirror the existing media limit). `write_frame` uses `write_all`; on error kill and wait, then return `SinkWrite`. `finish` drops stdin, joins stderr, waits for the child, rejects oversized stderr and non-zero exit with `ToolFailed`, and never leaves a child running. The sink owns only its child and does not create output paths.

- [ ] **Step 4: Run focused verification.**

```powershell
cargo fmt --all -- --check
cargo test -p feathertalk-inference --test raw_sink
cargo clippy -p feathertalk-inference --all-targets -- -D warnings
git diff --check
```

- [ ] **Step 5: Commit.**

```powershell
git add rust/crates/feathertalk-inference/src/raw_sink.rs rust/crates/feathertalk-inference/src/error.rs rust/crates/feathertalk-inference/src/lib.rs rust/crates/feathertalk-inference/tests/raw_sink.rs
git commit -m "feat: stream rendered frames to ffmpeg stdin"
```

---

### Task 4: Implement artifact-driven render orchestration and atomic publication

**Files:**
- Modify: `rust/crates/feathertalk-inference/src/executor.rs`
- Modify: `rust/crates/feathertalk-inference/src/error.rs` and `src/lib.rs`
- Test: `rust/crates/feathertalk-inference/tests/executor.rs`

**Interfaces:**
- `pub fn execute_offline_render<B, M, R, F>(model: &M, device: &B::Device, request: &OfflineRenderRequest, frame_reader: &R, sink_factory: &F) -> Result<OfflineRenderResult, InferenceError>` with `B: Backend`, `M: TalkingHeadModel<B>`, `R: FrameReader + ?Sized`, `F: RawVideoSinkFactory + ?Sized`.
- Uses `RenderPlan`, `RawFrameRenderSpec`, `raw_video_command`, `render_planned_frame`, `read_feature_file`, `read_landmarks`, and `compute_face_bbox` exactly as already exported.

- [ ] **Step 1: Write failing orchestration tests with real feature/landmark artifacts and fake reader/sink.**

Create a temporary absolute artifact tree containing `frames/000000.jpg`, `frames/000001.jpg`, `landmarks/000000.lms`, `landmarks/000001.lms`, `features.f32`, `audio.wav`, and a regular helper executable path. Use `write_feature_file` for a `FeatureMatrix` with two frames (`4 x 1024` values), 110 valid landmark points with a bbox inside a test frame, a fake `FrameReader` returning deterministic BGR frames, an `OutputModel` returning all ones, and a fake sink collecting frames and creating a non-empty staging output on `finish`. Assert:

```rust
let result = execute_offline_render(...).unwrap();
assert_eq!(result.frame_count(), 2);
assert_eq!(sink.frames(), &[frame0_bytes, frame1_bytes]);
assert_eq!(result.output_path(), request.output_path());
assert!(request.output_path().is_file());
```

Add a capped five-output case asserting source read order `[0,1,2,1,0]`. Add failures at reader frame 1, model NaN, sink write, sink finish, and publish rename; each must leave a pre-existing destination sentinel unchanged and remove only the invocation staging path.

- [ ] **Step 2: Run focused tests and verify they fail before orchestration exists.**

```powershell
cargo test -p feathertalk-inference --test executor
```

- [ ] **Step 3: Implement orchestration in small helpers.**

Implement these private helpers:

```rust
fn frame_path(request: &OfflineRenderRequest, index: usize) -> PathBuf;
fn landmark_path(request: &OfflineRenderRequest, index: usize) -> PathBuf;
fn validate_input_file(path: &Path, field: &'static str) -> Result<(), InferenceError>;
fn reserve_staging(path: &Path) -> Result<StagingGuard, InferenceError>;
fn verify_staging_output(path: &Path) -> Result<(), InferenceError>;
fn publish_staging(staging: &Path, destination: &Path) -> Result<(), InferenceError>;
```

The public function must validate request and input artifacts, read the feature matrix, construct `RenderPlan`, read frame 0 for dimensions, reserve staging with `create_new(true)`, construct the existing raw command, start the sink, then for each output frame read the selected frame and landmark, compute the bbox, call `render_planned_frame`, and call `sink.write_frame` exactly once. Finish the sink before touching the destination; verify and sync staging; atomically rename it; disarm the guard; return the summary. On every error, drop the sink before the guard and remove only the guard's reserved path. Do not scan or clean unrelated files.

- [ ] **Step 4: Run complete executor verification.**

```powershell
cargo fmt --all -- --check
cargo test -p feathertalk-inference --test executor
cargo clippy -p feathertalk-inference --all-targets -- -D warnings
git diff --check
```

- [ ] **Step 5: Commit.**

```powershell
git add rust/crates/feathertalk-inference/src/executor.rs rust/crates/feathertalk-inference/src/error.rs rust/crates/feathertalk-inference/src/lib.rs rust/crates/feathertalk-inference/tests/executor.rs
git commit -m "feat: execute offline render plans atomically"
```

---

### Task 5: Public API, workspace acceptance, review, and integration

**Files:**
- Modify: `rust/crates/feathertalk-inference/tests/public_api.rs`
- Modify: `docs/superpowers/specs/2026-08-27-offline-video-executor-design.md` only if review finds a documented mismatch.

- [ ] **Step 1: Add crate-root API coverage.**

Import every public executor, reader, sink, request/result, and error type only from `feathertalk_inference`; assert constructors/accessors compile without private-module paths.

- [ ] **Step 2: Run fresh full verification on the isolated branch.**

```powershell
cargo test -p feathertalk-inference --all-targets
cargo test --workspace --all-targets
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
git diff --check
```

Expected: all non-ignored tests pass; no command reads or writes the user demo directory; certified WGPU tests retain their existing explicit ignore behavior.

- [ ] **Step 3: Review the diff against the spec and migration design.**

Check every requirement: fixed frame/audio semantics are delegated to `RenderPlan`; JPEG conversion is explicit; FFmpeg receives raw BGR24 over stdin; no shell/PATH fallback; staging is same-parent and failure-cleaned; existing destinations are unchanged; no model/MOV dependency was introduced. Fix any mismatch with a new failing regression test first.

- [ ] **Step 4: Commit final API/test changes and record evidence.**

```powershell
git add rust/crates/feathertalk-inference/tests/public_api.rs
git commit -m "test: verify offline video executor public API"
```

- [ ] **Step 5: Complete the branch.**

Use `finishing-a-development-branch`: rerun the full suite, verify branch/worktree state, then merge locally into `main` (the branch forked from `main`) and rerun the merged full suite. Remove only `.worktrees/offline-video-executor` and its branch after the merged result is green. Preserve the user-owned untracked demo directory.

- [ ] **Step 6: Continue the migration automatically.**

Re-read `docs/superpowers/specs/2026-08-17-rust-desktop-migration-design.md`. If milestone four still has unmet work, start the next independent brainstorming/spec/plan cycle for standard model packages, ONNX opset 17 validation, or legacy model/feature migration CLI, in that order, without redefining this executor contract.

## Plan Self-Review

- Spec coverage: Tasks 1–4 cover request validation, bounded JPEG decoding, injected and production FFmpeg sinks, complete plan execution, artifact reads, model rendering, staging, atomic publication, and failure cleanup. Task 5 covers public API, full verification, review, merge, and automatic continuation.
- Placeholder scan: every task contains concrete interfaces, commands, assertions, and expected outcomes; no placeholder instruction or undefined neighboring interface remains. The JPEG fixture is explicitly required to be embedded in test source, and every helper signature used by later tasks is defined.
- Type consistency: `FrameReader`, `RawVideoSink`, `RawVideoSinkFactory`, `OfflineRenderRequest`, and `OfflineRenderResult` are introduced before executor use; existing `CommandSpec` and `RenderPlan` accessors match the current crate API.
- Scope: model loading, audio decoding, standard model packages, ONNX export, legacy migration, worker, GPUI, and demo assets remain outside this plan as required by the spec.
