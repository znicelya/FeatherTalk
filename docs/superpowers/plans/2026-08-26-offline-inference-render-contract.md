# Offline Inference Render Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox ( - [ ] ) syntax for tracking.

**Goal:** Add a dependency-light feathertalk-inference crate that makes FeatherTalk offline frame selection, audio windows, render geometry, output destinations, and raw-video FFmpeg command deterministic and testable.

**Architecture:** Keep this first inference slice pure and side-effect free. sequence.rs owns reflected frame-index math, plan.rs maps output frames to source/reference/audio indices, render.rs validates fixed geometry and output paths, and command.rs builds an argv vector without shell quoting. Burn execution, image decoding, pixel operations, and atomic installation consume these contracts in subsequent slices.

**Tech Stack:** Rust 1.92 edition 2024, thiserror, feathertalk-preprocess, feathertalk-media, tempfile for integration tests, and standard-library Path/OsString filesystem inspection.

## Global Constraints

- Add only rust/crates/feathertalk-inference and its workspace entry; never touch the protected demo/kanghui_training_video_featherhubert_188_latest directory.
- Runtime dependencies are limited to thiserror, feathertalk-preprocess, and feathertalk-media. Do not add Burn, WGPU, image/OpenCV, FFmpeg bindings, GPUI, ONNX Runtime, or shell execution.
- Output FPS is exactly 25; the audio window is exactly 8 slots centered at frame - 4; crop/inner/border are exactly 168/160/4.
- The frame sequence is 0,1,...,N-1,N-2,...,1,0,1,...; every frame plan uses the selected source frame as its reference frame.
- No function creates, truncates, renames, or deletes files. Existing symlinks and non-regular output destinations are rejected.
- Public wrappers expose immutable accessors. Integration tests import only from the crate root.
- Use apply_patch for edits, stage explicit paths, and finish each task with focused tests plus git diff --check.

---

## File Map

Create:

~~~text
rust/crates/feathertalk-inference/Cargo.toml
rust/crates/feathertalk-inference/src/lib.rs
rust/crates/feathertalk-inference/src/error.rs
rust/crates/feathertalk-inference/src/sequence.rs
rust/crates/feathertalk-inference/src/plan.rs
rust/crates/feathertalk-inference/src/render.rs
rust/crates/feathertalk-inference/src/command.rs
rust/crates/feathertalk-inference/tests/sequence.rs
rust/crates/feathertalk-inference/tests/plan.rs
rust/crates/feathertalk-inference/tests/render.rs
rust/crates/feathertalk-inference/tests/command.rs
rust/crates/feathertalk-inference/tests/public_api.rs
rust/crates/feathertalk-inference/tests/support/mod.rs
~~~

Modify rust/Cargo.toml and generated rust/Cargo.lock only to register the crate.

### Task 1: Deterministic frame sequence and render plan

Files:
- Modify rust/Cargo.toml and generated rust/Cargo.lock.
- Create the crate manifest, src/lib.rs, src/error.rs, src/sequence.rs, src/plan.rs.
- Create tests/sequence.rs and tests/plan.rs.

Interfaces:
- PingPongFrames::new(frame_count: usize) -> Result<Self, InferenceError>; frame_count(), position(), and next() are immutable/constant-time queries.
- RenderPlan::new(source_frame_count: usize, feature_frame_count: usize, max_output_frames: Option<usize>) -> Result<Self, InferenceError>.
- RenderPlan::output_frame_count() -> usize and frame(output_index: usize) -> Result<InferenceFramePlan, InferenceError>.
- InferenceFramePlan has public output_index, source_frame_index, reference_frame_index, and audio_window: [Option<usize>; 8].

- [ ] Step 1: Register the crate and write failing sequence tests.

Use this manifest:

~~~toml
[package]
name = "feathertalk-inference"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[dependencies]
feathertalk-media = { path = "../feathertalk-media" }
feathertalk-preprocess = { path = "../feathertalk-preprocess" }
thiserror.workspace = true

[dev-dependencies]
tempfile.workspace = true
~~~

Create tests/sequence.rs with:

~~~rust
use feathertalk_inference::{InferenceError, PingPongFrames};

#[test]
fn two_frames_repeat_without_duplicate_endpoints() {
    let mut picker = PingPongFrames::new(2).unwrap();
    let values: Vec<_> = (0..7).map(|_| picker.next()).collect();
    assert_eq!(values, vec![0, 1, 0, 1, 0, 1, 0]);
}

#[test]
fn three_frames_reflect_at_both_boundaries() {
    let mut picker = PingPongFrames::new(3).unwrap();
    let values: Vec<_> = (0..9).map(|_| picker.next()).collect();
    assert_eq!(values, vec![0, 1, 2, 1, 0, 1, 2, 1, 0]);
}

#[test]
fn rejects_fewer_than_two_source_frames() {
    assert!(matches!(PingPongFrames::new(0), Err(InferenceError::FrameCountTooSmall { .. })));
    assert!(matches!(PingPongFrames::new(1), Err(InferenceError::FrameCountTooSmall { .. })));
}
~~~

- [ ] Step 2: Run cargo test -p feathertalk-inference --test sequence from rust/ and confirm compilation fails because the package/API is absent.

- [ ] Step 3: Implement InferenceError and PingPongFrames.

InferenceError must include FrameCountTooSmall { actual, minimum }, EmptyFeatures, OutputFrameOutOfRange { index, count }, InvalidField { field, message }, ArithmeticOverflow, OutputExists { path }, OutputNotRegular { path }, OutputSymlink { path }, OutputParentInvalid { path }, InvalidTaskId { task_id }, FfmpegPathNotAbsolute { path }, and EmptyFfmpegPath.

Implement private frame_count, next_index, and direction. Initialize next_index=0 and direction=1; next() returns the current index and then advances. Use checked 2*(frame_count-1) arithmetic and reflect phase after the forward endpoint. position() returns the next index that will be returned.

- [ ] Step 4: Write failing tests/plan.rs.

~~~rust
use feathertalk_inference::{InferenceError, RenderPlan};

#[test]
fn plan_maps_ping_pong_source_and_current_frame_reference() {
    let plan = RenderPlan::new(3, 6, None).unwrap();
    let frames: Vec<_> = (0..6).map(|i| plan.frame(i).unwrap()).collect();
    assert_eq!(frames.iter().map(|f| f.source_frame_index).collect::<Vec<_>>(), vec![0, 1, 2, 1, 0, 1]);
    assert!(frames.iter().all(|f| f.source_frame_index == f.reference_frame_index));
    assert_eq!(frames[0].audio_window, [None, None, None, None, Some(0), Some(1), Some(2), Some(3)]);
    assert_eq!(frames[3].audio_window, [None, Some(0), Some(1), Some(2), Some(3), Some(4), Some(5), None]);
}

#[test]
fn plan_caps_preview_and_rejects_invalid_requests() {
    let plan = RenderPlan::new(2, 10, Some(4)).unwrap();
    assert_eq!(plan.output_frame_count(), 4);
    assert!(matches!(plan.frame(4), Err(InferenceError::OutputFrameOutOfRange { index: 4, count: 4 })));
    assert!(matches!(RenderPlan::new(2, 0, None), Err(InferenceError::EmptyFeatures)));
    assert!(matches!(RenderPlan::new(1, 2, None), Err(InferenceError::FrameCountTooSmall { .. })));
    assert!(matches!(RenderPlan::new(2, 2, Some(0)), Err(InferenceError::InvalidField { field: "max_output_frames", .. })));
}
~~~

- [ ] Step 5: Run cargo test -p feathertalk-inference --test plan and confirm it fails before implementation.

- [ ] Step 6: Implement RenderPlan with private counts. Compute output count as min(feature_frame_count, max_output_frames.unwrap_or(feature_frame_count)); reject Some(0). In frame(), reject output_index >= count, derive the same reflected source index as PingPongFrames, set reference_frame_index to source index, and call feathertalk_preprocess::audio_window_indices(output_index, feature_frame_count). Convert a preprocess error to InvalidField { field: "audio_window", ... }.

Export modules and types from src/lib.rs. Derive Debug, Clone, PartialEq, and Eq for InferenceFramePlan.

- [ ] Step 7: Run cargo fmt --all, cargo test -p feathertalk-inference --test sequence --test plan, cargo clippy -p feathertalk-inference --all-targets --all-features -- -D warnings, and git diff --check. Stage only explicit crate/Cargo files and commit with feat: add deterministic inference frame plan.

### Task 2: Fixed render geometry and safe output destination contract

Files:
- Create src/render.rs, tests/render.rs, tests/support/mod.rs.
- Modify src/lib.rs and src/error.rs.

Interfaces:
- RenderGeometry::standard() returns crop_size=168, inner_size=160, border=4, replacement_offset=(4,4).
- RawFrameRenderSpec::new(width, height, audio_path, output_path) validates dimensions and non-empty paths; accessors expose width, height, audio_path, output_path, and fixed fps()=25.
- validate_output_destination(path: &Path) -> Result<(), InferenceError> rejects existing destinations, symlinks, directories, and invalid parents without mutation.
- staging_output_path(path: &Path, task_id: &str) -> Result<PathBuf, InferenceError> returns a same-parent, same-extension staging name.

- [ ] Step 1: Write tests/render.rs before implementation. Cover standard geometry equality with feathertalk_preprocess::default_crop_spec(), native paths/fixed FPS, zero dimensions/empty paths, missing output with existing parent, existing file and directory, invalid task IDs, a symlinked parent, and a symlinked destination. The test should assert sentinel contents stay unchanged.

Use this representative test:

~~~rust
use std::path::Path;
use feathertalk_inference::{InferenceError, RawFrameRenderSpec, RenderGeometry, staging_output_path, validate_output_destination};

#[test]
fn standard_geometry_matches_preprocess_contract() {
    let geometry = RenderGeometry::standard();
    assert_eq!((geometry.crop_size(), geometry.inner_size(), geometry.border()), (168, 160, 4));
    assert_eq!(geometry.replacement_offset(), (4, 4));
    let crop = feathertalk_preprocess::default_crop_spec();
    assert_eq!(geometry.crop_size(), crop.crop_size);
    assert_eq!(geometry.inner_size(), crop.inner_size);
    assert_eq!(geometry.border(), crop.border);
}

#[test]
fn raw_spec_keeps_native_paths_and_fixed_fps() {
    let spec = RawFrameRenderSpec::new(1280, 720, Path::new("drive audio.wav"), Path::new("result.mp4")).unwrap();
    assert_eq!(spec.width(), 1280);
    assert_eq!(spec.height(), 720);
    assert_eq!(spec.fps(), 25);
    assert_eq!(spec.audio_path(), Path::new("drive audio.wav"));
    assert_eq!(spec.output_path(), Path::new("result.mp4"));
}
~~~

Add platform symlink helpers in tests/support/mod.rs and skip symlink assertions when creation is unavailable.

- [ ] Step 2: Run cargo test -p feathertalk-inference --test render and confirm missing-symbol failures.

- [ ] Step 3: Implement RenderGeometry and RawFrameRenderSpec with private fields and immutable accessors. standard() uses the constants from default_crop_spec and asserts crop_size == inner_size + 2*border. RawFrameRenderSpec stores no configurable FPS and rejects zero dimensions or empty OsStr values.

- [ ] Step 4: Implement component-wise path validation. Walk existing prefixes with symlink_metadata and return OutputSymlink before following links. Require the parent to be an existing real directory. For an existing final regular file return OutputExists; for a directory/device return OutputNotRegular; for a missing final path return Ok. Never create directories or alter files.

staging_output_path first validates the destination, accepts only 1–64 ASCII alphanumeric/underscore/hyphen/dot task IDs while rejecting dot and dot-dot, derives .{stem}.{task_id}.staging{extension} in the same parent, and returns OutputExists if the staging name already exists.

- [ ] Step 5: Add crate-root public_api.rs. Construct geometry/spec/path functions from root exports, bind accessors to &Path, and avoid private modules.

- [ ] Step 6: Run cargo fmt --all, cargo test -p feathertalk-inference --test render --test public_api, cargo clippy -p feathertalk-inference --all-targets --all-features -- -D warnings, and git diff --check. Commit with feat: add inference render geometry contract.

### Task 3: Raw-frame FFmpeg command and complete crate acceptance

Files:
- Create src/command.rs and tests/command.rs.
- Modify src/lib.rs and src/error.rs.

Interfaces:
- CommandSpec has immutable executable(), arguments(), and operation() accessors.
- raw_video_command(ffmpeg: &Path, spec: &RawFrameRenderSpec) -> Result<CommandSpec, InferenceError>.

- [ ] Step 1: Write tests/command.rs before implementation.

~~~rust
use std::{ffi::OsString, path::Path};
use feathertalk_inference::{RawFrameRenderSpec, raw_video_command};

#[test]
fn raw_video_command_has_stable_argument_order_and_native_paths() {
    let spec = RawFrameRenderSpec::new(640, 480, Path::new("audio file.wav"), Path::new("result file.mp4")).unwrap();
    let command = raw_video_command(Path::new("C:/tools/ffmpeg.exe"), &spec).unwrap();
    assert_eq!(command.executable(), Path::new("C:/tools/ffmpeg.exe"));
    assert_eq!(command.operation(), "render_raw_video");
    assert_eq!(command.arguments(), &[
        OsString::from("-hide_banner"), OsString::from("-nostdin"), OsString::from("-y"), OsString::from("-v"), OsString::from("error"),
        OsString::from("-f"), OsString::from("rawvideo"), OsString::from("-pix_fmt"), OsString::from("bgr24"),
        OsString::from("-video_size"), OsString::from("640x480"), OsString::from("-framerate"), OsString::from("25"),
        OsString::from("-i"), OsString::from("-"), OsString::from("-i"), OsString::from("audio file.wav"),
        OsString::from("-c:v"), OsString::from("libx264"), OsString::from("-pix_fmt"), OsString::from("yuv420p"),
        OsString::from("-c:a"), OsString::from("aac"), OsString::from("-shortest"), OsString::from("result file.mp4"),
    ]);
}

#[test]
fn command_rejects_empty_or_relative_ffmpeg_paths() {
    let spec = RawFrameRenderSpec::new(1, 1, Path::new("a.wav"), Path::new("o.mp4")).unwrap();
    assert!(raw_video_command(Path::new(""), &spec).is_err());
    assert!(raw_video_command(Path::new("ffmpeg"), &spec).is_err());
}
~~~

- [ ] Step 2: Run cargo test -p feathertalk-inference --test command and confirm missing-symbol failures.

- [ ] Step 3: Implement CommandSpec and argv-only raw_video_command. Reject empty and relative FFmpeg paths; retain executable exactly as supplied; emit the tested argument order; append audio/output paths as one OsString each; do not quote, split, or normalize paths.

- [ ] Step 4: Extend public_api.rs to build a RenderPlan, geometry, spec, and command from crate-root exports. Assert plan audio_window equals feathertalk_preprocess::audio_window_indices(0, 4).unwrap() and command FPS is 25.

- [ ] Step 5: Run crate verification:
~~~powershell
cargo fmt --all -- --check
cargo test -p feathertalk-inference --all-targets
cargo clippy -p feathertalk-inference --all-targets --all-features -- -D warnings
git diff --check
~~~

Then run workspace verification:
~~~powershell
cargo test --workspace --all-targets
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
git diff --check
~~~

- [ ] Step 6: Stage explicit files and commit with feat: add raw video inference command contract. Re-read the migration design; the next slice implements actual image crop/resize/paste-back and Burn adapters using RenderPlan, without redefining frame order or audio windows.

## Plan Self-Review

- Spec coverage: Tasks 1–3 cover ping-pong sequence, current-frame reference, eight-slot audio window, 25 FPS, 168/160/4 geometry, output path/symlink rules, staging naming, argv-only FFmpeg command, immutable public API, and focused/workspace verification.
- Placeholder scan: no TODO, TBD, implement later, or undefined task dependency appears in this plan.
- Type consistency: every type and error variant is introduced before consumption; accessor names and signatures match the tests.
- Scope: no model loading, image decoding, filesystem mutation, FFmpeg execution, ONNX export, model packaging, CLI, worker, or GPUI work is included.
