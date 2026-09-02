# extract_frames Worker Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `extract_frames` to the worker and the CLI: cut every frame of a normalised 25 fps video, run SCRFD and PFLD over each frame, publish `frames/`, `landmarks/`, and `quality.json`, and report frame-accurate progress that a cancel can interrupt.

**Architecture:** `feathertalk-frame-pipeline` gets three corrections and one new seam. The corrections: seek timestamps carry milliseconds (today 25 indices per second collapse onto 7 distinct images), one ffmpeg invocation writes a chunk of 250 frames instead of one frame, and the spec's containment rule stops rejecting a video that sits directly under the output root. The seam: a `PipelineObserver` that extraction and evaluation call to publish frame counts and to ask whether the caller wants to stop. The worker composes the three pipeline stages itself, so a quality failure can name the offending frames; it resolves SCRFD and PFLD artifact directories from two new environment variables, loads the three adapters once per task, and turns the pipeline's phases into `ExtractingFrames` and `DetectingFaces` progress events over its existing `TaskReporter`. `JobExecutor` starts carrying the whole `WorkerConfig` because this command needs two toolchains and the next slice will need a third.

**Tech Stack:** Rust 2024, 1.94.0 toolchain, `serde_json`, `sha2`, `clap 4.5` (derive), `tempfile`, burn 0.21 through `feathertalk-frame-adapters`, std threads and channels. No async runtime.

**Design:** `docs/superpowers/specs/2026-09-03-extract-frames-worker-command-design.md`

## Global Constraints

- Run every cargo command from `E:/workspace/github/FeatherTalk/rust`; run git from `E:/workspace/github/FeatherTalk`.
- Extraction targets are fixed: 25 fps, `-q:v 2` JPEG, six-digit file names, `frames/` and `landmarks/` under the project's `assets/`. No new tuning knob and no `force` flag.
- `FRAME_CHUNK` is a compile-time constant, never configuration. It is a throughput and cancellation-latency decision, not a user preference.
- User-facing strings are Chinese; code comments, doc comments, and diagnostics are English.
- Every source file stays free of a BOM and uses LF endings.
- `serde_json` is built without `preserve_order`; never re-serialise a frame the worker or CLI received.
- Progress events carry no metrics: `Metrics::empty()` stays untouched. `completed` always counts frames, never chunks or percentages.
- Cancellation is cooperative. The observer is polled at every chunk boundary and before every evaluated frame; nothing in the frame pipeline kills a child process.
- No new binary fixture enters git. The end-to-end test cuts its clip at runtime from the already-tracked `demo/feathertalk_demo_latest_188.mp4` and reads the already-tracked SCRFD and PFLD artifacts.
- Do not touch `demo/kanghui_training_video_featherhubert_188_latest/`; it must stay untracked.
- Commit after each task. Stage explicit paths, never `git add .`. Never push to `origin`.
- Every task leaves the tree green: the task's own test command plus `cargo check` must pass before its commit.
- The final gate for the whole slice: `cargo check`, `cargo test --workspace --all-targets`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --all -- --check`, `git diff --check`.

## File Structure

- `rust/crates/feathertalk-frame-pipeline/src/commands.rs` — millisecond seek timestamps and the chunked `frame_command`.
- `rust/crates/feathertalk-frame-pipeline/src/extraction.rs` — `FRAME_CHUNK`, the chunk loop, `extract_frames_observed`.
- `rust/crates/feathertalk-frame-pipeline/src/observe.rs` (new) — `PipelinePhase`, `PipelineObserver`, `NoObserver`. One responsibility: the progress and cancellation seam.
- `rust/crates/feathertalk-frame-pipeline/src/evaluate.rs` — `evaluate_frames_observed`.
- `rust/crates/feathertalk-frame-pipeline/src/error.rs` — `PipelineError::Cancelled`.
- `rust/crates/feathertalk-frame-pipeline/src/model.rs` — the narrowed containment rule and the two directory getters.
- `rust/crates/feathertalk-frame-pipeline/src/lib.rs` — module declarations and re-exports.
- `rust/crates/feathertalk-frame-pipeline/tests/support/mod.rs` (new) — argument helpers shared by the crate's test binaries.
- `rust/crates/feathertalk-frame-pipeline/tests/{commands,contracts,extraction,evaluation,pipeline,publish}.rs` — updated fake runners and new coverage.
- `rust/crates/feathertalk-frame-adapters/src/lib.rs` — re-export `ScrfdArtifactPaths`.
- `rust/crates/feathertalk-frame-adapters/tests/{support/mod.rs,pipeline.rs}` — the fixture runner follows the chunk contract.
- `rust/crates/feathertalk-worker/Cargo.toml` — the frame crates as dependencies.
- `rust/crates/feathertalk-worker/src/config.rs` — `ModelToolchain` and the two new environment variables.
- `rust/crates/feathertalk-worker/src/models.rs` (new) — load the three adapters, PFLD on a big-stack thread.
- `rust/crates/feathertalk-worker/src/quality_result.rs` (new) — the completed-result payload.
- `rust/crates/feathertalk-worker/src/error_map.rs` — pipeline and quality-anomaly error mapping.
- `rust/crates/feathertalk-worker/src/extract_frames.rs` (new) — the command body and its observer bridge.
- `rust/crates/feathertalk-worker/src/commands.rs` — `&WorkerConfig` plumbing and the new request arm.
- `rust/crates/feathertalk-worker/src/runtime.rs` — the `JobExecutor` shape and the rejection text.
- `rust/crates/feathertalk-worker/src/handshake.rs` — advertise the command once both toolchains resolve.
- `rust/crates/feathertalk-worker/src/lib.rs` — module declarations and re-exports.
- `rust/crates/feathertalk-worker/tests/{config,quality_result,extract_frames,commands,handshake,runtime}.rs` — configuration, result, command, capability, and wire coverage.
- `rust/crates/feathertalk-cli/src/{cli,run,render}.rs` — the subcommand, its request, and the unsupported-command advice.
- `rust/crates/feathertalk-cli/tests/{cli,real_worker}.rs` — CLI behaviour and end-to-end coverage.

---

### Task 1: Millisecond seek timestamps

**Files:**
- Modify: `rust/crates/feathertalk-frame-pipeline/src/commands.rs`
- Create: `rust/crates/feathertalk-frame-pipeline/tests/support/mod.rs`
- Test: `rust/crates/feathertalk-frame-pipeline/tests/commands.rs`

**Interfaces:**
- Consumes: existing `frame_command(&FrameExtractor, &Path, u64, &Path) -> CommandSpec` and `CommandSpec::{executable, arguments, operation}`.
- Produces: `format_timestamp` with millisecond precision, private `FRAME_RATE: u64 = 25` and `MILLIS_PER_FRAME: u64 = 40`, and the test helpers `support::flag_value(&CommandSpec, &str) -> String` and `support::flag_number(&CommandSpec, &str) -> u64`. Tasks 2 and 4 use both helpers.

**Why first:** `format_timestamp(index)` currently prints `{seconds}.{remainder:02}`, where `remainder` is a frame count, not a fraction. Frame 27 becomes `-ss 1.02`, i.e. 1.02 s, which ffmpeg rounds back to frame 26. Verified against `demo/feathertalk_demo_latest_188.mp4`: the 25 indices of one second produce only 7 distinct images, every one a valid JPEG of a face. The pipeline cannot detect this, so it must not be able to happen.

- [ ] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-frame-pipeline/tests/support/mod.rs`. `#![allow(dead_code)]` matches `feathertalk-frame-adapters/tests/support/mod.rs`: each test binary uses a subset of the helpers.

```rust
#![allow(dead_code)]

use std::ffi::OsStr;

use feathertalk_frame_pipeline::CommandSpec;

/// The value ffmpeg receives after `flag`, as text.
pub fn flag_value(command: &CommandSpec, flag: &str) -> String {
    let arguments = command.arguments();
    let position = arguments
        .iter()
        .position(|argument| argument.as_os_str() == OsStr::new(flag))
        .unwrap_or_else(|| panic!("{flag} is missing from the frame command"));
    arguments
        .get(position + 1)
        .unwrap_or_else(|| panic!("{flag} carries no value"))
        .to_str()
        .unwrap_or_else(|| panic!("{flag} carries non-UTF-8 text"))
        .to_owned()
}

/// The same value parsed as a frame counter.
pub fn flag_number(command: &CommandSpec, flag: &str) -> u64 {
    flag_value(command, flag)
        .parse()
        .unwrap_or_else(|_| panic!("{flag} must carry a number"))
}
```

In `rust/crates/feathertalk-frame-pipeline/tests/commands.rs`, add `mod support;` and `use support::flag_value;` below the existing imports, replace the last assertion of `frame_command_uses_fixed_flags_and_native_path_arguments` (`assert!(command.arguments().contains(&"1.01".into()));`) with the line below, and append the new test.

```rust
    assert_eq!(flag_value(&command, "-ss"), "1.040");
}

#[test]
fn frame_timestamps_keep_millisecond_precision() {
    let value = spec();
    let tool = extractor();
    // 25 frames per second divide 1000 ms exactly, so every index lands on a
    // whole millisecond and no case needs rounding.
    for (index, expected) in [
        (0_u64, "0.000"),
        (24, "0.960"),
        (25, "1.000"),
        (26, "1.040"),
        (49, "1.960"),
        (50, "2.000"),
        (1510, "60.400"),
    ] {
        let command = frame_command(&tool, value.video_path(), index, &value.frame_path(index));
        assert_eq!(flag_value(&command, "-ss"), expected, "frame {index}");
    }
}
```

- [ ] **Step 2: Run the test and watch it fail**

Run: `cargo test -p feathertalk-frame-pipeline --test commands`

Expected: FAIL. `frame_timestamps_keep_millisecond_precision` reports `assertion left == right` with `left: "1.02"`, `right: "1.040"` for frame 26, and `frame_command_uses_fixed_flags_and_native_path_arguments` fails on the same value.

- [ ] **Step 3: Implement**

In `rust/crates/feathertalk-frame-pipeline/src/commands.rs`, add the two constants above `frame_command` and replace `format_timestamp`.

```rust
/// The frame rate the extraction pipeline pins ffmpeg to.
///
/// `-vf fps=25` stays a literal in the argument list: it is filter syntax, and
/// spelling it out keeps the command readable next to ffmpeg documentation.
const FRAME_RATE: u64 = 25;

/// Milliseconds one frame occupies. 25 divides 1000 exactly, so every frame
/// index maps onto a whole number of milliseconds and `-ss` never rounds.
const MILLIS_PER_FRAME: u64 = 1_000 / FRAME_RATE;
```

```rust
fn format_timestamp(index: u64) -> OsString {
    let seconds = index / FRAME_RATE;
    let millis = (index % FRAME_RATE) * MILLIS_PER_FRAME;
    format!("{seconds}.{millis:03}").into()
}
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `cargo test -p feathertalk-frame-pipeline --test commands`, then `cargo test -p feathertalk-frame-pipeline` for the rest of the crate. No other test asserts a timestamp, so both must be green.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-frame-pipeline/src/commands.rs rust/crates/feathertalk-frame-pipeline/tests/commands.rs rust/crates/feathertalk-frame-pipeline/tests/support/mod.rs
git commit -m "fix(frame-pipeline): seek frames with millisecond precision"
```

---

### Task 2: Extract frames in chunks of 250

**Files:**
- Modify: `rust/crates/feathertalk-frame-pipeline/src/commands.rs`
- Modify: `rust/crates/feathertalk-frame-pipeline/src/extraction.rs`
- Modify: `rust/crates/feathertalk-frame-pipeline/src/lib.rs`
- Modify: `rust/crates/feathertalk-frame-pipeline/tests/support/mod.rs`
- Test: `rust/crates/feathertalk-frame-pipeline/tests/{commands,extraction,evaluation,pipeline,publish}.rs`
- Modify: `rust/crates/feathertalk-frame-adapters/tests/support/mod.rs`
- Test: `rust/crates/feathertalk-frame-adapters/tests/pipeline.rs`

**Interfaces:**
- Consumes: `format_timestamp`, `FRAME_RATE` from Task 1.
- Produces: `frame_command(extractor: &FrameExtractor, source: &Path, first_index: u64, count: u64, output_pattern: &Path) -> CommandSpec` with `operation() == "extract_frames"`; `pub const FRAME_CHUNK: u64 = 250` exported from the crate root; `support::chunk_outputs(&CommandSpec) -> Vec<(u64, PathBuf)>`. Task 4 wraps the loop this task writes; Task 11 relies on the throughput.

**Why:** one ffmpeg process per frame costs 129–193 ms of process and decoder start-up, about 255 s for the 1511-frame demo clip. Chunks of 250 finish the same clip in 3.2 s. Both runs were compared file by file: byte-identical output.

- [ ] **Step 1: Write the failing tests**

Append to `rust/crates/feathertalk-frame-pipeline/tests/support/mod.rs` and extend its `std` import to `use std::{ffi::OsStr, path::{Path, PathBuf}};`.

```rust
/// The frames one chunk command is expected to write, as `(index, path)` pairs.
///
/// Fake runners use this to stand in for ffmpeg's `image2` muxer: the command
/// ends with a `%06d.jpg` pattern, and `-start_number` plus `-frames:v` say
/// which indices that pattern expands to.
pub fn chunk_outputs(command: &CommandSpec) -> Vec<(u64, PathBuf)> {
    let pattern = Path::new(
        command
            .arguments()
            .last()
            .expect("the frame command ends with the output pattern"),
    );
    let directory = pattern
        .parent()
        .expect("the output pattern sits inside the frames directory")
        .to_owned();
    let first = flag_number(command, "-start_number");
    let count = flag_number(command, "-frames:v");
    (first..first + count)
        .map(|index| (index, directory.join(format!("{index:06}.jpg"))))
        .collect()
}
```

In `tests/commands.rs`, both existing calls move to the five-argument form. Add `use support::flag_number;`, add the pattern helper, and rewrite the flag test.

```rust
fn pattern() -> PathBuf {
    PathBuf::from(r"C:\project\assets\.feathertalk-frame-build-1-1\frames\%06d.jpg")
}

#[test]
fn frame_command_uses_fixed_flags_and_native_path_arguments() {
    let value = spec();
    let command = frame_command(&extractor(), value.video_path(), 26, 250, &pattern());
    assert_eq!(command.operation(), "extract_frames");
    assert_eq!(command.executable(), PathBuf::from(r"C:\bundle\ffmpeg.exe"));
    assert!(
        command
            .arguments()
            .windows(2)
            .any(|pair| pair == ["-vf", "fps=25"])
    );
    assert_eq!(flag_number(&command, "-frames:v"), 250);
    assert_eq!(flag_number(&command, "-start_number"), 26);
    assert_eq!(flag_value(&command, "-ss"), "1.040");
    assert_eq!(command.arguments().last(), Some(&pattern().into_os_string()));
    assert!(
        command
            .arguments()
            .contains(&value.video_path().as_os_str().to_owned())
    );
}
```

In the same file, the timestamp test's call becomes `frame_command(&tool, value.video_path(), index, 1, &pattern())`.

In `tests/extraction.rs`: add `mod support;` plus `use support::{chunk_outputs, flag_number, flag_value};`, add `FRAME_CHUNK` to the `feathertalk_frame_pipeline` import list, and replace the write section of `FakeRunner::run` (the `let path = Path::new(...)` line and the `match self.writes` block) with a loop over the chunk's frames.

```rust
        self.commands.lock().unwrap().push(command.clone());
        let output = self.outputs.lock().unwrap().pop_front().unwrap()?;
        for (index, path) in chunk_outputs(command) {
            match self.writes {
                WriteMode::Bytes => fs::write(&path, format!("frame:{index}")).unwrap(),
                WriteMode::Missing => {}
                WriteMode::Empty => fs::write(&path, []).unwrap(),
                WriteMode::Oversized => {
                    fs::write(&path, b"x").unwrap();
                    let file = fs::OpenOptions::new().write(true).open(&path).unwrap();
                    file.set_len(16 * 1024 * 1024 + 1).unwrap();
                }
            }
        }
        Ok(output)
```

`SymlinkRunner::run` in the same file gets the same treatment:

```rust
        for (_, path) in chunk_outputs(command) {
            std::os::windows::fs::symlink_file(&self.target, &path).unwrap();
        }
        Ok(ProcessOutput::new(Some(0), vec![], vec![]))
```

Three frames are now one invocation, so `extracts_exact_frame_count_and_records_hashes` takes `ok_outputs(1)` and asserts `runner.commands.lock().unwrap().len() == 1`. In `injected_timeout_is_preserved`, the two `operation: "extract_frame"` patterns (lines 165 and 173) become `"extract_frames"`. Then append the chunk-boundary test:

```rust
#[test]
fn frames_are_extracted_in_chunks_with_a_short_tail() {
    let (_root, spec, extractor) = setup(FRAME_CHUNK * 6 + 11);
    let runner = FakeRunner::new(ok_outputs(7), WriteMode::Bytes);
    let batch = extract_frames_with_runner(&spec, &extractor, &runner).unwrap();
    assert_eq!(batch.frames().len() as u64, FRAME_CHUNK * 6 + 11);
    let commands = runner.commands.lock().unwrap();
    assert_eq!(commands.len(), 7);
    for (position, command) in commands.iter().enumerate() {
        let first = FRAME_CHUNK * position as u64;
        assert_eq!(flag_number(command, "-start_number"), first);
        assert_eq!(
            flag_number(command, "-frames:v"),
            if position == 6 { 11 } else { FRAME_CHUNK }
        );
        // 250 frames are exactly 10 s, so every chunk starts on a whole second.
        assert_eq!(flag_value(command, "-ss"), format!("{}.000", position * 10));
    }
}
```

In `tests/evaluation.rs`, `tests/pipeline.rs`, and `tests/publish.rs`, add `mod support;` plus `use support::chunk_outputs;` and replace the single `fs::write(command.arguments().last().unwrap(), b"jpeg-frame").unwrap();` line in each fake runner with:

```rust
        for (_, path) in chunk_outputs(command) {
            fs::write(path, b"jpeg-frame").unwrap();
        }
```

In `rust/crates/feathertalk-frame-adapters/tests/support/mod.rs`, append the same `flag_value`, `flag_number`, and `chunk_outputs` helpers (that crate's test binaries cannot reach the other crate's test module), importing `feathertalk_frame_pipeline::CommandSpec`. Then rewrite `FixtureRunner::run` in `rust/crates/feathertalk-frame-adapters/tests/pipeline.rs`:

```rust
    fn run(
        &self,
        command: &CommandSpec,
        _timeout: Duration,
    ) -> Result<ProcessOutput, PipelineError> {
        for (index, path) in support::chunk_outputs(command) {
            let payload = self
                .payloads
                .get(index as usize)
                .unwrap_or_else(|| panic!("no payload for frame {index}"));
            fs::write(&path, payload).expect("the staging directory is writable");
        }
        Ok(ProcessOutput::new(Some(0), vec![], vec![]))
    }
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `cargo test -p feathertalk-frame-pipeline`

Expected: FAIL at compile time. `frame_command` takes four arguments, `FRAME_CHUNK` does not exist, and `chunk_outputs` finds no `-frames:v` value it can trust.

- [ ] **Step 3: Implement**

Replace `frame_command` in `src/commands.rs`. The flag order matches the sequence validated by hand against the demo clip.

```rust
/// One ffmpeg invocation that writes `count` frames starting at `first_index`.
///
/// `output_pattern` must be an `image2` pattern such as `frames/%06d.jpg`;
/// ffmpeg expands it with `-start_number`, so the file names match
/// [`FramePipelineSpec::frame_path`].
pub fn frame_command(
    extractor: &FrameExtractor,
    source: &Path,
    first_index: u64,
    count: u64,
    output_pattern: &Path,
) -> CommandSpec {
    let timestamp = format_timestamp(first_index);
    let mut arguments = args(["-hide_banner", "-nostdin", "-y", "-v", "error", "-ss"]);
    arguments.push(timestamp);
    arguments.push("-i".into());
    arguments.push(source.as_os_str().to_owned());
    arguments.extend(args([
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-vf",
        "fps=25",
    ]));
    arguments.push("-frames:v".into());
    arguments.push(count.to_string().into());
    arguments.push("-start_number".into());
    arguments.push(first_index.to_string().into());
    arguments.extend(args(["-q:v", "2", "-f", "image2"]));
    arguments.push(output_pattern.as_os_str().to_owned());
    CommandSpec::new(extractor.ffmpeg().to_owned(), arguments, "extract_frames")
}
```

In `src/extraction.rs`, add the constant above `ExtractedFrame`:

```rust
/// How many frames one ffmpeg invocation writes.
///
/// Measured against `demo/feathertalk_demo_latest_188.mp4` (1511 frames,
/// 1280x720, 25 fps): one process per frame costs 129-193 ms of process and
/// decoder start-up, roughly 255 s for the clip, while chunks of 250 finish in
/// 3.2 s with byte-identical JPEG output. The chunk also bounds how long a
/// cancellation waits, measured at about 1.1 s for a chunk of 250 frames at
/// this resolution.
pub const FRAME_CHUNK: u64 = 250;
```

Replace the extraction loop in `extract_frames_with_runner` (from `let mut frames = Vec::with_capacity(...)` up to the `Ok(FrameBatch { ... })` line, which stays):

```rust
    let mut frames = Vec::with_capacity(spec.frame_count() as usize);
    let pattern = frames_dir.join("%06d.jpg");
    let mut first_index = 0;
    while first_index < spec.frame_count() {
        let count = FRAME_CHUNK.min(spec.frame_count() - first_index);
        let command = frame_command(extractor, spec.video_path(), first_index, count, &pattern);
        if let Err(error) = run_frame(runner, &command, extractor.timeout()) {
            let _ = fs::remove_dir_all(&staging);
            return Err(error);
        }
        // ffmpeg writing fewer files than asked is a hard failure, not
        // something to compensate for: `inspect_frame` reports the first gap.
        for index in first_index..first_index + count {
            let output = frames_dir.join(format!("{index:06}.jpg"));
            match inspect_frame(index, output) {
                Ok(frame) => frames.push(frame),
                Err(error) => {
                    let _ = fs::remove_dir_all(&staging);
                    return Err(error);
                }
            }
        }
        first_index += count;
    }
```

In `src/lib.rs`, extend the extraction re-export:

```rust
pub use extraction::{
    ExtractedFrame, FRAME_CHUNK, FrameBatch, extract_frames, extract_frames_with_runner,
};
```

- [ ] **Step 4: Run the tests and watch them pass**

Run: `cargo test -p feathertalk-frame-pipeline`, then `cargo test -p feathertalk-frame-adapters --test pipeline`. If `Path` or another import falls out of use in a rewritten test file, delete it from that file's `use` list; `cargo clippy -p feathertalk-frame-pipeline --all-targets -- -D warnings` is the check.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-frame-pipeline/src/commands.rs rust/crates/feathertalk-frame-pipeline/src/extraction.rs rust/crates/feathertalk-frame-pipeline/src/lib.rs rust/crates/feathertalk-frame-pipeline/tests/support/mod.rs rust/crates/feathertalk-frame-pipeline/tests/commands.rs rust/crates/feathertalk-frame-pipeline/tests/extraction.rs rust/crates/feathertalk-frame-pipeline/tests/evaluation.rs rust/crates/feathertalk-frame-pipeline/tests/pipeline.rs rust/crates/feathertalk-frame-pipeline/tests/publish.rs rust/crates/feathertalk-frame-adapters/tests/support/mod.rs rust/crates/feathertalk-frame-adapters/tests/pipeline.rs
git commit -m "feat(frame-pipeline): extract frames in chunks of 250"
```

---

### Task 3: Allow a video that sits directly under the output root

**Files:**
- Modify: `rust/crates/feathertalk-frame-pipeline/src/model.rs`
- Test: `rust/crates/feathertalk-frame-pipeline/tests/contracts.rs`

**Interfaces:**
- Consumes: `FramePipelineSpec::new`.
- Produces: the narrowed containment rule. Task 11 depends on it: the worker's video is `<project>/assets/video_25fps.mp4` and its output root is `<project>/assets`, which today's `starts_with` check rejects.

**Why:** `feathertalk-project` fixes the normalised video at `assets/video_25fps.mp4` and this slice writes `assets/frames` and `assets/landmarks`. The real hazard is a video the pipeline would overwrite or read while writing, which means the three paths the pipeline owns, not every path under the root.

- [ ] **Step 1: Write the failing tests**

Append to `rust/crates/feathertalk-frame-pipeline/tests/contracts.rs`.

```rust
#[test]
fn a_video_directly_under_the_output_root_is_accepted() {
    // This is the real project layout: assets/video_25fps.mp4 next to the
    // frames/ and landmarks/ directories the pipeline writes.
    let value = FramePipelineSpec::new(
        PathBuf::from(r"C:\project\assets\video_25fps.mp4"),
        PathBuf::from(r"C:\project\assets"),
        3,
        640,
        480,
    )
    .unwrap();
    assert_eq!(
        value.frame_path(0),
        PathBuf::from(r"C:\project\assets\frames\000000.jpg")
    );
}

#[test]
fn output_root_rejects_only_the_paths_the_pipeline_owns() {
    for source in [
        r"C:\project\assets",
        r"C:\project\assets\frames\000000.jpg",
        r"C:\project\assets\landmarks\000000.lms",
        r"C:\project\assets\quality.json",
    ] {
        let result = FramePipelineSpec::new(
            PathBuf::from(source),
            PathBuf::from(r"C:\project\assets"),
            3,
            640,
            480,
        );
        assert!(
            matches!(
                result,
                Err(PipelineError::InvalidField {
                    field: "output_root",
                    ..
                })
            ),
            "{source} must be rejected"
        );
    }
}
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `cargo test -p feathertalk-frame-pipeline --test contracts`

Expected: FAIL. `a_video_directly_under_the_output_root_is_accepted` panics on `unwrap()` with `InvalidField { field: "output_root", message: "must not equal or contain the source video path" }`.

- [ ] **Step 3: Implement**

In `src/model.rs`, replace the containment check (the `if video_path == output_root || video_path.starts_with(&output_root)` block) with three rules.

```rust
    // The source video may live beside the outputs -- that is the project
    // layout -- but it must not be one of the three paths extraction and
    // publication write.
    if video_path == output_root {
        return Err(invalid("output_root", "must not equal the source video path"));
    }
    if video_path.starts_with(&output_root) && video_path.parent() != Some(output_root.as_path()) {
        return Err(invalid(
            "output_root",
            "must not contain the source video path in a nested directory",
        ));
    }
    if video_path == output_root.join("quality.json") {
        return Err(invalid(
            "output_root",
            "must not equal the quality report path",
        ));
    }
```

- [ ] **Step 4: Run the tests and watch them pass**

Run: `cargo test -p feathertalk-frame-pipeline --test contracts`, then `cargo test -p feathertalk-frame-pipeline`.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-frame-pipeline/src/model.rs rust/crates/feathertalk-frame-pipeline/tests/contracts.rs
git commit -m "fix(frame-pipeline): allow a video that sits directly under the output root"
```

---

### Task 4: Report extraction progress and honour cancellation

**Files:**
- Create: `rust/crates/feathertalk-frame-pipeline/src/observe.rs`
- Modify: `rust/crates/feathertalk-frame-pipeline/src/error.rs`
- Modify: `rust/crates/feathertalk-frame-pipeline/src/extraction.rs`
- Modify: `rust/crates/feathertalk-frame-pipeline/src/lib.rs`
- Modify: `rust/crates/feathertalk-frame-pipeline/tests/support/mod.rs`
- Test: `rust/crates/feathertalk-frame-pipeline/tests/extraction.rs`

**Interfaces:**
- Consumes: `FRAME_CHUNK` and the chunk loop from Task 2.
- Produces: `PipelinePhase`, `PipelineObserver`, `NoObserver`, `PipelineError::Cancelled { operation: &'static str }`, the test helper `support::Recorder`, and

```rust
pub fn extract_frames_observed<R: ProcessRunner + ?Sized>(
    spec: &FramePipelineSpec,
    extractor: &FrameExtractor,
    runner: &R,
    observer: &dyn PipelineObserver,
) -> Result<FrameBatch, PipelineError>
```

  Task 5 reuses the trait and the recorder, Task 8 maps `Cancelled` onto `ErrorCode::TaskCancelled`, and Task 11 implements the trait over the worker's reporter.

**Why:** extraction of a 60-second clip takes seconds and evaluation takes minutes, so the worker needs both a progress stream and a way to stop. The observer is generic over `R: ProcessRunner + ?Sized` rather than taking `&dyn ProcessRunner` because the worker holds a `&R` with exactly that bound and `&R where R: ?Sized` cannot coerce to `&dyn`. Cancellation is observer-only: no child process is killed, so cancel lands at the next chunk boundary (one ffmpeg invocation, measured at ~1.1 s for 250 frames on the demo clip). Killing ffmpeg would mean bridging this crate's `CommandSpec`/`ProcessOutput` onto `feathertalk-media`'s, and `CommandSpec::new` is `pub(crate)`; the design document rules that out.

- [ ] **Step 1: Write the failing tests**

Append to `rust/crates/feathertalk-frame-pipeline/tests/support/mod.rs`, extend its `std` import with `sync::Mutex`, and extend its crate import to `use feathertalk_frame_pipeline::{CommandSpec, PipelineObserver, PipelinePhase};`.

```rust
/// Records every phase a pipeline stage reports, and flips to cancelled once
/// `cancel_after` phases have arrived.
pub struct Recorder {
    phases: Mutex<Vec<PipelinePhase>>,
    cancel_after: Option<usize>,
}

impl Recorder {
    pub fn new(cancel_after: Option<usize>) -> Self {
        Self {
            phases: Mutex::new(Vec::new()),
            cancel_after,
        }
    }

    pub fn phases(&self) -> Vec<PipelinePhase> {
        self.phases.lock().unwrap().clone()
    }
}

impl PipelineObserver for Recorder {
    fn phase(&self, phase: PipelinePhase) {
        self.phases.lock().unwrap().push(phase);
    }

    fn is_cancelled(&self) -> bool {
        match self.cancel_after {
            Some(limit) => self.phases.lock().unwrap().len() >= limit,
            None => false,
        }
    }
}
```

In `tests/extraction.rs`, add `PipelinePhase` and `extract_frames_observed` to the `feathertalk_frame_pipeline` import list, add `Recorder` to the `use support::{…};` list, and append both tests.

```rust
#[test]
fn extraction_reports_one_phase_per_finished_chunk() {
    let total = FRAME_CHUNK * 2 + 5;
    let (_root, spec, extractor) = setup(total);
    let runner = FakeRunner::new(ok_outputs(3), WriteMode::Bytes);
    let observer = Recorder::new(None);

    let batch = extract_frames_observed(&spec, &extractor, &runner, &observer).unwrap();

    assert_eq!(batch.frames().len() as u64, total);
    // The last phase carries the exact frame count, so the worker never has
    // to invent a final 100 % event.
    assert_eq!(
        observer.phases(),
        vec![
            PipelinePhase::Extracting {
                completed: FRAME_CHUNK,
                total
            },
            PipelinePhase::Extracting {
                completed: FRAME_CHUNK * 2,
                total
            },
            PipelinePhase::Extracting { completed: total, total },
        ]
    );
}

#[test]
fn cancellation_stops_extraction_at_the_next_chunk_boundary() {
    let (_root, spec, extractor) = setup(FRAME_CHUNK * 2);
    let runner = FakeRunner::new(ok_outputs(2), WriteMode::Bytes);
    // Cancelled as soon as the first chunk has been reported.
    let observer = Recorder::new(Some(1));

    let error = extract_frames_observed(&spec, &extractor, &runner, &observer).unwrap_err();

    assert!(
        matches!(
            error,
            PipelineError::Cancelled {
                operation: "extract_frames"
            }
        ),
        "{error:?}"
    );
    assert_eq!(runner.commands.lock().unwrap().len(), 1);
    assert!(staging_dirs(spec.output_root()).is_empty());
}
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `cargo test -p feathertalk-frame-pipeline --test extraction`

Expected: FAIL to compile. `error[E0432]: unresolved imports feathertalk_frame_pipeline::PipelinePhase, feathertalk_frame_pipeline::extract_frames_observed`, and the same unresolved names in `tests/support/mod.rs`.

- [ ] **Step 3: Implement**

Create `rust/crates/feathertalk-frame-pipeline/src/observe.rs`.

```rust
//! Progress and cancellation seam for the long-running pipeline stages.
//!
//! This crate reports frame counts and asks whether it should stop. It knows
//! nothing about the worker protocol, threads, or task identifiers.

/// A progress point a pipeline stage reached.
///
/// `completed` and `total` count frames -- never chunks, never percentages --
/// so a caller can forward them into its own progress type unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelinePhase {
    /// `completed` frames of `total` have been written and inspected.
    Extracting { completed: u64, total: u64 },
    /// Evaluation is about to run on frame number `completed` of `total`.
    Evaluating { completed: u64, total: u64 },
}

/// Receives progress points and answers cancellation questions.
///
/// Both methods run on the thread that drives the pipeline, so an
/// implementation must not block. There is deliberately no `Send + Sync`
/// bound: the worker's reporter owns an `mpsc::Sender`, which is `Send` but
/// not `Sync`, and the pipeline never moves the observer across threads.
pub trait PipelineObserver {
    /// Called once per progress point. The default drops the phase.
    fn phase(&self, _phase: PipelinePhase) {}

    /// Called before each chunk and before each evaluated frame. The default
    /// never cancels.
    fn is_cancelled(&self) -> bool {
        false
    }
}

/// The observer for callers that want neither progress nor cancellation.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoObserver;

impl PipelineObserver for NoObserver {}
```

In `src/error.rs`, append the variant before the closing brace of `PipelineError`.

```rust
    /// The observer asked the pipeline to stop.
    #[error("frame pipeline cancelled during {operation}")]
    Cancelled { operation: &'static str },
```

In `src/extraction.rs`, add `NoObserver`, `PipelineObserver`, and `PipelinePhase` to the `use crate::{…}` list, rename the existing `extract_frames_with_runner` body to `extract_frames_observed`, and leave a delegating wrapper behind.

```rust
/// Extracts every frame with the given runner and no observer.
pub fn extract_frames_with_runner<R: ProcessRunner + ?Sized>(
    spec: &FramePipelineSpec,
    extractor: &FrameExtractor,
    runner: &R,
) -> Result<FrameBatch, PipelineError> {
    extract_frames_observed(spec, extractor, runner, &NoObserver)
}

/// Extracts every frame, reporting one phase per finished chunk and stopping
/// at the next chunk boundary once the observer reports cancellation.
pub fn extract_frames_observed<R: ProcessRunner + ?Sized>(
    spec: &FramePipelineSpec,
    extractor: &FrameExtractor,
    runner: &R,
    observer: &dyn PipelineObserver,
) -> Result<FrameBatch, PipelineError> {
```

Everything from `reject_final_destinations(spec)?` down to the `frames_dir` creation stays byte-identical. Inside the chunk loop Task 2 wrote, add the cancellation check at the top and the phase report at the bottom.

```rust
    while first_index < spec.frame_count() {
        if observer.is_cancelled() {
            // Staging is disposable: a cancelled run leaves the previous
            // outputs, if any, exactly as they were.
            let _ = fs::remove_dir_all(&staging);
            return Err(PipelineError::Cancelled {
                operation: "extract_frames",
            });
        }
        let count = FRAME_CHUNK.min(spec.frame_count() - first_index);
        let command = frame_command(extractor, spec.video_path(), first_index, count, &pattern);
        if let Err(error) = run_frame(runner, &command, extractor.timeout()) {
            let _ = fs::remove_dir_all(&staging);
            return Err(error);
        }
        for index in first_index..first_index + count {
            match inspect_frame(index, frames_dir.join(format!("{index:06}.jpg"))) {
                Ok(frame) => frames.push(frame),
                Err(error) => {
                    let _ = fs::remove_dir_all(&staging);
                    return Err(error);
                }
            }
        }
        first_index += count;
        observer.phase(PipelinePhase::Extracting {
            completed: first_index,
            total: spec.frame_count(),
        });
    }
```

Keep whatever error plumbing Task 2 left in place; only the `is_cancelled` block and the `observer.phase` call are new. In `src/lib.rs`, add `mod observe;` after `mod model;`, add `extract_frames_observed` to the extraction re-export, and re-export the observer types after the `model` block.

```rust
pub use extraction::{
    ExtractedFrame, FRAME_CHUNK, FrameBatch, extract_frames, extract_frames_observed,
    extract_frames_with_runner,
};
```

```rust
pub use observe::{NoObserver, PipelineObserver, PipelinePhase};
```

- [ ] **Step 4: Run the tests and watch them pass**

Run: `cargo test -p feathertalk-frame-pipeline`, then `cargo clippy -p feathertalk-frame-pipeline --all-targets -- -D warnings`.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-frame-pipeline/src/error.rs rust/crates/feathertalk-frame-pipeline/src/extraction.rs rust/crates/feathertalk-frame-pipeline/src/lib.rs rust/crates/feathertalk-frame-pipeline/src/observe.rs rust/crates/feathertalk-frame-pipeline/tests/support/mod.rs rust/crates/feathertalk-frame-pipeline/tests/extraction.rs
git commit -m "feat(frame-pipeline): report extraction progress and honour cancellation"
```

---

### Task 5: Report evaluation progress and honour cancellation

**Files:**
- Modify: `rust/crates/feathertalk-frame-pipeline/src/evaluate.rs`
- Modify: `rust/crates/feathertalk-frame-pipeline/src/lib.rs`
- Test: `rust/crates/feathertalk-frame-pipeline/tests/evaluation.rs`

**Interfaces:**
- Consumes: `PipelineObserver`, `PipelinePhase`, `NoObserver`, `PipelineError::Cancelled`, and `support::Recorder` from Task 4.
- Produces:

```rust
pub fn evaluate_frames_observed<D, F, L>(
    batch: &FrameBatch,
    decoder: &D,
    detector: &F,
    predictor: &L,
    observer: &dyn PipelineObserver,
) -> Result<FrameEvaluation, PipelineError>
where
    D: FrameDecoder + ?Sized,
    F: FaceDetector + ?Sized,
    L: LandmarkPredictor + ?Sized,
```

  Task 11 calls it directly.

**Why:** evaluation is the slow stage -- roughly 190 ms per frame on CPU, so 1511 frames take about five minutes -- and it is where a cancel request most needs to land. Unlike extraction, the phase is reported *before* the frame runs: the loop body is a chain of `continue` arms for anomalies, and reporting up front means the counter advances without touching any of them. `completed` therefore runs `0..total-1` here while extraction reports `FRAME_CHUNK..=total`. The asymmetry is intentional and both satisfy `completed <= total`, which is all `Event::validate` checks.

- [ ] **Step 1: Write the failing tests**

In `tests/evaluation.rs`, add `PipelinePhase` and `evaluate_frames_observed` to the `feathertalk_frame_pipeline` import list, add `mod support;` with `use support::Recorder;` (Task 2 already added the module declaration for `chunk_outputs`), generalise the batch helper, and append both tests.

```rust
fn batch() -> (tempfile::TempDir, FrameBatch) {
    batch_of(1)
}

fn batch_of(frame_count: u64) -> (tempfile::TempDir, FrameBatch) {
    let root = tempfile::tempdir().unwrap();
    let video = root.path().join("video_25fps.mp4");
    fs::write(&video, b"video").unwrap();
    let spec =
        FramePipelineSpec::new(video, root.path().join("assets"), frame_count, 640, 480).unwrap();
    let extractor =
        FrameExtractor::new(root.path().join("ffmpeg"), Duration::from_secs(1)).unwrap();
    let batch = extract_frames_with_runner(&spec, &extractor, &OneFrameRunner).unwrap();
    (root, batch)
}
```

```rust
#[test]
fn evaluation_reports_one_phase_before_each_frame() {
    let (_root, batch) = batch_of(3);
    // A failing decoder turns every frame into an anomaly, which keeps the
    // test focused on the counter instead of on model plumbing.
    let decoder = Decoder {
        blur: 0.0,
        fail: true,
    };
    let detector = Detector {
        detections: Vec::new(),
        fail: false,
    };
    let observer = Recorder::new(None);

    let evaluation =
        evaluate_frames_observed(&batch, &decoder, &detector, &predictor(0.5), &observer).unwrap();

    assert_eq!(evaluation.anomalies().len(), 3);
    assert_eq!(
        observer.phases(),
        vec![
            PipelinePhase::Evaluating {
                completed: 0,
                total: 3
            },
            PipelinePhase::Evaluating {
                completed: 1,
                total: 3
            },
            PipelinePhase::Evaluating {
                completed: 2,
                total: 3
            },
        ]
    );
}

#[test]
fn cancellation_stops_evaluation_at_the_next_frame() {
    let (_root, batch) = batch_of(4);
    let decoder = Decoder {
        blur: 0.0,
        fail: true,
    };
    let detector = Detector {
        detections: Vec::new(),
        fail: false,
    };
    // Cancelled once two frames have been reported, so the third frame is
    // where the run stops.
    let observer = Recorder::new(Some(2));

    let error = evaluate_frames_observed(&batch, &decoder, &detector, &predictor(0.5), &observer)
        .unwrap_err();

    assert!(
        matches!(
            error,
            PipelineError::Cancelled {
                operation: "evaluate_frames"
            }
        ),
        "{error:?}"
    );
    assert_eq!(observer.phases().len(), 2);
}
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `cargo test -p feathertalk-frame-pipeline --test evaluation`

Expected: FAIL to compile. `error[E0432]: unresolved import feathertalk_frame_pipeline::evaluate_frames_observed`.

- [ ] **Step 3: Implement**

In `src/evaluate.rs`, add `NoObserver`, `PipelineObserver`, and `PipelinePhase` to the file's `use crate::{…}` list, then replace the head of `evaluate_frames_with_models` with a delegating wrapper plus the observed entry point.

```rust
/// Evaluates every extracted frame with no observer.
pub fn evaluate_frames_with_models<D, F, L>(
    batch: &FrameBatch,
    decoder: &D,
    detector: &F,
    predictor: &L,
) -> Result<FrameEvaluation, PipelineError>
where
    D: FrameDecoder + ?Sized,
    F: FaceDetector + ?Sized,
    L: LandmarkPredictor + ?Sized,
{
    evaluate_frames_observed(batch, decoder, detector, predictor, &NoObserver)
}

/// Evaluates every extracted frame, reporting one phase before each frame and
/// stopping before the next frame once the observer reports cancellation.
pub fn evaluate_frames_observed<D, F, L>(
    batch: &FrameBatch,
    decoder: &D,
    detector: &F,
    predictor: &L,
    observer: &dyn PipelineObserver,
) -> Result<FrameEvaluation, PipelineError>
where
    D: FrameDecoder + ?Sized,
    F: FaceDetector + ?Sized,
    L: LandmarkPredictor + ?Sized,
{
    let mut accepted = Vec::new();
    let mut anomalies = Vec::new();
    let total = batch.frames().len() as u64;
    for (completed, extracted) in batch.frames().iter().enumerate() {
        if observer.is_cancelled() {
            return Err(PipelineError::Cancelled {
                operation: "evaluate_frames",
            });
        }
        observer.phase(PipelinePhase::Evaluating {
            completed: completed as u64,
            total,
        });
        let index = extracted.index();
```

Everything from the `let frame = match decoder.decode(index, extracted.path())` line to the closing `Ok(FrameEvaluation { accepted, anomalies })` stays byte-identical. Nothing is cleaned up on cancellation: evaluation only reads the staging directory, and the `FrameBatch` destructor still disarms it.

In `src/lib.rs`, extend the `evaluate` re-export with `evaluate_frames_observed` (alphabetically before `evaluate_frames_with_models`).

- [ ] **Step 4: Run the tests and watch them pass**

Run: `cargo test -p feathertalk-frame-pipeline`, then `cargo clippy -p feathertalk-frame-pipeline --all-targets -- -D warnings`.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-frame-pipeline/src/evaluate.rs rust/crates/feathertalk-frame-pipeline/src/lib.rs rust/crates/feathertalk-frame-pipeline/tests/evaluation.rs
git commit -m "feat(frame-pipeline): report evaluation progress and honour cancellation"
```

---

### Task 6: Resolve the SCRFD and PFLD artifact directories

**Files:**
- Modify: `rust/crates/feathertalk-worker/src/config.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs`
- Test: `rust/crates/feathertalk-worker/tests/config.rs` (new)

**Interfaces:**
- Consumes: the existing `required_path` validator.
- Produces: `ENV_SCRFD_DIR = "FEATHERTALK_WORKER_SCRFD_DIR"`, `ENV_PFLD_DIR = "FEATHERTALK_WORKER_PFLD_DIR"`, `ModelToolchain` with `scrfd_dir() -> &Path` and `pfld_dir() -> &Path`, `WorkerConfig::from_values_with_models(Option<String>, Option<String>, Option<String>, Option<String>, Option<String>) -> Self`, and the getters `WorkerConfig::models() -> Option<&ModelToolchain>` and `WorkerConfig::model_rejection() -> Option<&str>`. Task 10 loads from these directories, Task 12 gates the handshake on `models()`, and Task 13 names both variables in the CLI's advice.

**Why:** the worker cannot hard-code model paths and must not fail to start when they are missing, which is exactly how the media toolchain already behaves. `from_values` keeps its three-argument shape so the existing media tests stay untouched; only `from_env` and the new tests use the five-argument form. Existence is not checked here: `required_path` only demands a non-empty absolute path, and a directory that exists at startup but is deleted before the first job would make the check a lie anyway. Task 10 surfaces a bad directory as a load failure.

- [ ] **Step 1: Write the failing tests**

Create `rust/crates/feathertalk-worker/tests/config.rs`.

```rust
use std::path::PathBuf;

use feathertalk_worker::WorkerConfig;

fn absolute(name: &str) -> String {
    std::env::current_dir()
        .unwrap()
        .join(name)
        .display()
        .to_string()
}

#[test]
fn two_absolute_directories_resolve_the_model_toolchain() {
    let config = WorkerConfig::from_values_with_models(
        None,
        None,
        None,
        Some(absolute("scrfd_2_5g")),
        Some(absolute("pfld_ghost_one")),
    );

    let models = config.models().expect("both directories are absolute");
    assert_eq!(models.scrfd_dir(), PathBuf::from(absolute("scrfd_2_5g")));
    assert_eq!(models.pfld_dir(), PathBuf::from(absolute("pfld_ghost_one")));
    assert_eq!(config.model_rejection(), None);
}

#[test]
fn a_missing_pfld_directory_rejects_the_model_toolchain() {
    let config =
        WorkerConfig::from_values_with_models(None, None, None, Some(absolute("scrfd_2_5g")), None);

    assert!(config.models().is_none());
    let rejection = config.model_rejection().expect("a reason is kept");
    assert!(
        rejection.contains("FEATHERTALK_WORKER_PFLD_DIR"),
        "{rejection}"
    );
}

#[test]
fn a_relative_model_directory_is_rejected() {
    let config = WorkerConfig::from_values_with_models(
        None,
        None,
        None,
        Some("artifacts/scrfd_2_5g".to_owned()),
        Some(absolute("pfld_ghost_one")),
    );

    let rejection = config.model_rejection().expect("a reason is kept");
    assert!(rejection.contains("must be an absolute path"), "{rejection}");
}

#[test]
fn the_two_toolchains_are_resolved_independently() {
    // A usable media toolchain must not imply usable models, and the
    // three-argument constructor must keep working for the media tests.
    let config = WorkerConfig::from_values(
        Some(absolute("ffprobe-test")),
        Some(absolute("ffmpeg-test")),
        None,
    );

    assert!(config.media().is_some());
    assert_eq!(config.media_rejection(), None);
    assert!(config.models().is_none());
    assert!(config.model_rejection().is_some());
}
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `cargo test -p feathertalk-worker --test config`

Expected: FAIL to compile. `error[E0599]: no function or associated item named from_values_with_models found for struct WorkerConfig`, plus `no method named models` and `no method named model_rejection`.

- [ ] **Step 3: Implement**

In `src/config.rs`, extend the `std` import to `use std::{path::{Path, PathBuf}, time::Duration};`, add the two variables below the existing constants, and add the toolchain type.

```rust
pub const ENV_SCRFD_DIR: &str = "FEATHERTALK_WORKER_SCRFD_DIR";
pub const ENV_PFLD_DIR: &str = "FEATHERTALK_WORKER_PFLD_DIR";
```

```rust
/// Where the worker finds the two model artifact directories.
///
/// Only the shape of the paths is checked here. Whether the directories hold a
/// loadable manifest and weights is discovered when the first job loads them,
/// because a directory can disappear between startup and the first job.
#[derive(Debug, Clone)]
pub struct ModelToolchain {
    scrfd_dir: PathBuf,
    pfld_dir: PathBuf,
}

impl ModelToolchain {
    pub fn scrfd_dir(&self) -> &Path {
        &self.scrfd_dir
    }

    pub fn pfld_dir(&self) -> &Path {
        &self.pfld_dir
    }
}
```

Add the two fields to `WorkerConfig`, then rework the constructors and add the getters.

```rust
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    worker_version: String,
    media: Option<MediaToolchain>,
    media_rejection: Option<String>,
    models: Option<ModelToolchain>,
    model_rejection: Option<String>,
}

impl WorkerConfig {
    pub fn from_env() -> Self {
        Self::from_values_with_models(
            std::env::var(ENV_FFPROBE).ok(),
            std::env::var(ENV_FFMPEG).ok(),
            std::env::var(ENV_MEDIA_TIMEOUT_MS).ok(),
            std::env::var(ENV_SCRFD_DIR).ok(),
            std::env::var(ENV_PFLD_DIR).ok(),
        )
    }

    /// The media-only form: no model directories, so `extract_frames` stays
    /// unsupported.
    pub fn from_values(
        ffprobe: Option<String>,
        ffmpeg: Option<String>,
        timeout_ms: Option<String>,
    ) -> Self {
        Self::from_values_with_models(ffprobe, ffmpeg, timeout_ms, None, None)
    }

    pub fn from_values_with_models(
        ffprobe: Option<String>,
        ffmpeg: Option<String>,
        timeout_ms: Option<String>,
        scrfd_dir: Option<String>,
        pfld_dir: Option<String>,
    ) -> Self {
        let (media, media_rejection) = match media_toolchain(ffprobe, ffmpeg, timeout_ms) {
            Ok(toolchain) => (Some(toolchain), None),
            Err(reason) => (None, Some(reason)),
        };
        let (models, model_rejection) = match model_toolchain(scrfd_dir, pfld_dir) {
            Ok(toolchain) => (Some(toolchain), None),
            Err(reason) => (None, Some(reason)),
        };
        Self {
            worker_version: env!("CARGO_PKG_VERSION").to_owned(),
            media,
            media_rejection,
            models,
            model_rejection,
        }
    }
```

```rust
    pub fn models(&self) -> Option<&ModelToolchain> {
        self.models.as_ref()
    }

    pub fn model_rejection(&self) -> Option<&str> {
        self.model_rejection.as_deref()
    }
```

Add the resolver next to `media_toolchain`.

```rust
fn model_toolchain(
    scrfd_dir: Option<String>,
    pfld_dir: Option<String>,
) -> Result<ModelToolchain, String> {
    let scrfd_dir = required_path(scrfd_dir, ENV_SCRFD_DIR)?;
    let pfld_dir = required_path(pfld_dir, ENV_PFLD_DIR)?;
    Ok(ModelToolchain {
        scrfd_dir,
        pfld_dir,
    })
}
```

In `src/lib.rs`, extend the config re-export.

```rust
pub use config::{
    DEFAULT_MEDIA_TIMEOUT_MS, ENV_FFMPEG, ENV_FFPROBE, ENV_MEDIA_TIMEOUT_MS, ENV_PFLD_DIR,
    ENV_SCRFD_DIR, ModelToolchain, WorkerConfig,
};
```

- [ ] **Step 4: Run the tests and watch them pass**

Run: `cargo test -p feathertalk-worker --test config`, then `cargo test -p feathertalk-worker`.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/src/config.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/tests/config.rs
git commit -m "feat(worker): resolve the SCRFD and PFLD artifact directories"
```

---

### Task 7: Pass the whole config into command execution

**Files:**
- Modify: `rust/crates/feathertalk-worker/src/commands.rs`
- Modify: `rust/crates/feathertalk-worker/src/runtime.rs`
- Test: `rust/crates/feathertalk-worker/tests/commands.rs`
- Test: `rust/crates/feathertalk-worker/tests/runtime.rs`

**Interfaces:**
- Consumes: `WorkerConfig` with the model getters from Task 6.
- Produces the three reshaped signatures:

```rust
pub type JobExecutor = Box<
    dyn Fn(&Request, &WorkerConfig, &CancellationToken, &dyn TaskReporter) -> CommandOutcome
        + Send
        + 'static,
>;

pub fn execute(
    request: &Request,
    config: &WorkerConfig,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
) -> CommandOutcome

pub fn execute_with_runner<R: ProcessRunner + ?Sized>(
    request: &Request,
    config: &WorkerConfig,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    runner: &R,
) -> CommandOutcome
```

  Task 11 adds the `ExtractFrames` arm that needs `config.media()` and `config.models()` in the same body.

**Why:** `Option<&MediaToolchain>` was enough while media was the only environment-dependent thing. Adding a second option would mean a fifth parameter on `execute` and a sixth on `execute_with_runner`, and every future command would add another. `WorkerConfig` is `Clone` and holds three small fields, so the job thread can own a clone. This task changes no behaviour, so its "failing test" is the mechanical rewrite of the call sites: they stop compiling until the signatures change.

- [ ] **Step 1: Write the failing tests**

In `tests/commands.rs`, drop `MediaToolchain` from the `feathertalk_media` import, add `WorkerConfig` to the `feathertalk_worker` import, and replace `fn toolchain()` with two configuration builders.

```rust
fn bare_config() -> WorkerConfig {
    WorkerConfig::from_values(None, None, None)
}

fn media_config() -> WorkerConfig {
    let root = std::env::current_dir().unwrap();
    WorkerConfig::from_values(
        Some(root.join("ffprobe-test").display().to_string()),
        Some(root.join("ffmpeg-test").display().to_string()),
        Some("10000".to_owned()),
    )
}
```

Then rewrite the second argument of all thirteen `execute_with_runner` calls: every `None` becomes `&bare_config()`, every inline `Some(&toolchain())` becomes `&media_config()`, and each of the four `let toolchain = toolchain();` bindings becomes `let config = media_config();` with `Some(&toolchain)` becoming `&config`. The bindings are needed where the test also asserts on the recorded commands, because the temporary would otherwise be dropped mid-expression.

In `tests/runtime.rs`, rename the `_media` closure parameter to `_config` in the six executors that ignore it (`instant_executor`, `blocking_executor`, `gated_executor`, `reporting_executor`, and the two inline `Box::new` executors), and thread the configuration through the one that does not.

```rust
fn blocking_probe_executor(started: Sender<()>) -> JobExecutor {
    Box::new(move |request, config, token, reporter| {
        let runner = BlockingRunner {
            started: Mutex::new(started.clone()),
            token: token.clone(),
        };
        execute_with_runner(request, config, token, reporter, &runner)
    })
}
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `cargo test -p feathertalk-worker --test commands`

Expected: FAIL to compile. `error[E0308]: mismatched types -- expected Option<&MediaToolchain>, found &WorkerConfig` at every rewritten call site.

- [ ] **Step 3: Implement**

In `src/commands.rs`, replace the `media: Option<&MediaToolchain>` parameter of `execute` and `execute_with_runner` with `config: &WorkerConfig`, add `WorkerConfig` to the `crate::{…}` import, and keep `MediaToolchain` in the `feathertalk_media` import only if the file still names the type. The guard inside the two media arms becomes:

```rust
        let Some(toolchain) = config.media() else {
            return CommandOutcome::Failed(unsupported(request.kind()));
        };
```

In `src/runtime.rs`, reshape the executor type, hand the job thread a clone of the configuration, and drop the now-unused `MediaToolchain` import.

```rust
pub type JobExecutor = Box<
    dyn Fn(&Request, &WorkerConfig, &CancellationToken, &dyn TaskReporter) -> CommandOutcome
        + Send
        + 'static,
>;
```

```rust
    let execution_tx = control_tx;
    // The executor thread needs the whole configuration, and `WorkerConfig` is
    // three small fields, so it gets its own clone instead of a shared borrow.
    let execution_config = config.clone();
    let execution = thread::spawn(move || run_jobs(&job_rx, &execution_tx, execution_config, executor));
```

```rust
fn run_jobs(
    job_rx: &Receiver<Job>,
    control_tx: &Sender<ControlMessage>,
    config: WorkerConfig,
    executor: JobExecutor,
) {
```

and inside it, `executor(&job.request, &config, &job.token, &reporter)`.

- [ ] **Step 4: Run the tests and watch them pass**

Run: `cargo test -p feathertalk-worker`, then `cargo clippy -p feathertalk-worker --all-targets -- -D warnings` to catch the unused `MediaToolchain` import.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/src/commands.rs rust/crates/feathertalk-worker/src/runtime.rs rust/crates/feathertalk-worker/tests/commands.rs rust/crates/feathertalk-worker/tests/runtime.rs
git commit -m "chore(worker): pass the whole config into command execution"
```

---

### Task 8: Map pipeline errors onto protocol error codes

**Files:**
- Modify: `rust/crates/feathertalk-worker/Cargo.toml`
- Modify: `rust/crates/feathertalk-worker/src/error_map.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs`
- Test: `rust/crates/feathertalk-worker/tests/error_mapping.rs`

**Interfaces:**
- Consumes: `PipelineError` including the `Cancelled` variant from Task 4, plus `AnomalyCode` and `FrameAnomaly`.
- Produces: `pipeline_task_error(&PipelineError) -> TaskError`, `is_pipeline_cancellation(&PipelineError) -> bool`, and `quality_task_error(&[FrameAnomaly]) -> TaskError`, all re-exported from `lib.rs`. Task 11 calls all three.

**Why:** this task also adds the `feathertalk-frame-pipeline` dependency to the worker, because it is the first worker file that names `PipelineError`. Every match is exhaustive with no `_` arm, matching `project_error_code` and `media_error_code`: adding a pipeline error variant must break this file rather than silently degrade to `WorkerCrashed`. A rejected quality report is not a `PipelineError` at all -- `evaluate_frames_observed` returns `Ok` with anomalies -- so it needs its own mapping that reports the *first* anomaly's code, which is the one the user has to act on.

- [ ] **Step 1: Write the failing tests**

Append to `rust/crates/feathertalk-worker/tests/error_mapping.rs`, adding `feathertalk_frame_pipeline::{AnomalyCode, FrameAnomaly, PipelineError, RecoveryAction}` and the three new worker functions to the imports.

```rust
fn anomaly(index: u64, code: AnomalyCode) -> FrameAnomaly {
    FrameAnomaly::new(index, code, "摘要", "detail", RecoveryAction::ExcludeFrame).unwrap()
}

#[test]
fn every_pipeline_error_maps_to_a_code_and_a_valid_payload() {
    let cases = vec![
        (
            PipelineError::InvalidField {
                field: "frame_count",
                message: "must be greater than zero".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            PipelineError::OutputDestinationExists { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            PipelineError::FrameMissing { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            PipelineError::Io {
                operation: "create_dir",
                path: path(),
                source: io_error(io::ErrorKind::StorageFull),
            },
            ErrorCode::DiskSpaceLow,
        ),
        (
            PipelineError::Io {
                operation: "create_dir",
                path: path(),
                source: io_error(io::ErrorKind::PermissionDenied),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            PipelineError::Adapter {
                component: "scrfd",
                message: "device lost".to_owned(),
            },
            ErrorCode::ModelIncompatible,
        ),
        (
            PipelineError::Cancelled {
                operation: "extract_frames",
            },
            ErrorCode::TaskCancelled,
        ),
        (
            PipelineError::ToolTimedOut {
                operation: "extract_frames",
                timeout_ms: 300_000,
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            PipelineError::QualityRejected { count: 4 },
            ErrorCode::WorkerCrashed,
        ),
    ];

    for (error, expected) in cases {
        let mapped = pipeline_task_error(&error);
        assert_eq!(mapped.code, expected, "{error:?}");
        assert_eq!(mapped.stage, TaskStage::Preparing, "{error:?}");
        assert_eq!(mapped.recovery, expected.default_recovery(), "{error:?}");
        assert!(!mapped.summary.trim().is_empty(), "{error:?}");
        assert!(!mapped.detail.is_empty(), "{error:?}");
        mapped.validate().unwrap();
    }
}

#[test]
fn only_cancellation_is_reported_as_cancellation() {
    assert!(is_pipeline_cancellation(&PipelineError::Cancelled {
        operation: "evaluate_frames"
    }));
    assert!(!is_pipeline_cancellation(&PipelineError::QualityRejected {
        count: 1
    }));
}

#[test]
fn a_rejected_quality_report_reports_the_first_anomaly() {
    let anomalies = vec![
        anomaly(7, AnomalyCode::LandmarkInvalid),
        anomaly(8, AnomalyCode::FaceNotFound),
        anomaly(9, AnomalyCode::BlurredFrame),
        anomaly(10, AnomalyCode::ModelFailed),
    ];

    let mapped = quality_task_error(&anomalies);

    assert_eq!(mapped.code, ErrorCode::LandmarkInvalid);
    assert_eq!(mapped.stage, TaskStage::Preparing);
    assert!(mapped.detail.contains("4 frame(s) rejected"), "{}", mapped.detail);
    // Only the first three are named, so a run with thousands of bad frames
    // still produces a readable detail.
    assert!(mapped.detail.contains("frame 7"), "{}", mapped.detail);
    assert!(mapped.detail.contains("frame 9"), "{}", mapped.detail);
    assert!(!mapped.detail.contains("frame 10"), "{}", mapped.detail);
    mapped.validate().unwrap();
}

#[test]
fn an_empty_anomaly_list_still_produces_a_valid_error() {
    let mapped = quality_task_error(&[]);

    assert_eq!(mapped.code, ErrorCode::MediaInvalid);
    mapped.validate().unwrap();
}
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `cargo test -p feathertalk-worker --test error_mapping`

Expected: FAIL to compile. `error[E0432]: unresolved import feathertalk_frame_pipeline` (the dependency is not declared yet), followed by the three unresolved worker functions once it is.

- [ ] **Step 3: Implement**

In `rust/crates/feathertalk-worker/Cargo.toml`, add the dependency in alphabetical order.

```toml
feathertalk-frame-pipeline = { path = "../feathertalk-frame-pipeline" }
```

In `src/error_map.rs`, add `use feathertalk_frame_pipeline::{AnomalyCode, FrameAnomaly, PipelineError};` and append the mapping. The comment above `FAILURE_STAGE` is still accurate: the pipeline runs inside one job, and a stage that has already reached `ExtractingFrames` still reports its failure as a preparation failure so the terminal stage stays `Failed`.

```rust
/// How many rejected frames the detail names before it stops. Enough to see a
/// pattern, short enough to stay inside `MAX_DETAIL_CHARS`.
const MAX_REPORTED_ANOMALIES: usize = 3;

pub fn pipeline_task_error(error: &PipelineError) -> TaskError {
    let code = pipeline_error_code(error);
    TaskError::new(
        code,
        pipeline_summary(error),
        &clamp(&error.to_string()),
        FAILURE_STAGE,
    )
}

pub fn is_pipeline_cancellation(error: &PipelineError) -> bool {
    matches!(error, PipelineError::Cancelled { .. })
}

/// Maps a quality report the pipeline accepted but the caller must reject.
///
/// The first anomaly decides the code and the summary: it is the earliest frame
/// the user has to fix, and mixing codes would leave the CLI without a single
/// recovery hint.
pub fn quality_task_error(anomalies: &[FrameAnomaly]) -> TaskError {
    let first = anomalies.first();
    let code = first.map_or(ErrorCode::MediaInvalid, |anomaly| {
        anomaly_error_code(anomaly.code())
    });
    let summary = first.map_or("抽帧质检未通过", |anomaly| {
        anomaly_summary(anomaly.code())
    });
    let mut detail = format!("{} frame(s) rejected", anomalies.len());
    for anomaly in anomalies.iter().take(MAX_REPORTED_ANOMALIES) {
        detail.push_str(&format!(
            "; frame {} {:?}: {}",
            anomaly.frame_index(),
            anomaly.code(),
            anomaly.summary()
        ));
    }
    TaskError::new(code, summary, &clamp(&detail), FAILURE_STAGE)
}

fn pipeline_error_code(error: &PipelineError) -> ErrorCode {
    match error {
        // Bad input or an output directory the user has to clear: the media,
        // not the worker, is what has to change.
        PipelineError::InvalidField { .. }
        | PipelineError::InvalidReport { .. }
        | PipelineError::OutputDestinationExists { .. }
        | PipelineError::FrameMissing { .. }
        | PipelineError::FrameNotRegular { .. }
        | PipelineError::FrameEmpty { .. }
        | PipelineError::FrameTooLarge { .. } => ErrorCode::MediaInvalid,
        PipelineError::Io { source, .. } => io_error_code(source),
        PipelineError::Adapter { .. } => ErrorCode::ModelIncompatible,
        PipelineError::Cancelled { .. } => ErrorCode::TaskCancelled,
        // Everything left is the worker's own machinery misbehaving.
        PipelineError::ToolFailed { .. }
        | PipelineError::ToolTimedOut { .. }
        | PipelineError::ToolOutputTooLarge { .. }
        | PipelineError::ToolSpawn { .. }
        | PipelineError::ReportJson { .. }
        | PipelineError::ReportNotRegular { .. }
        | PipelineError::ReportTooLarge { .. }
        | PipelineError::PublishFailed { .. }
        | PipelineError::PublishRollbackFailed { .. }
        | PipelineError::QualityRejected { .. } => ErrorCode::WorkerCrashed,
    }
}

fn pipeline_summary(error: &PipelineError) -> &'static str {
    match error {
        PipelineError::InvalidField { .. } | PipelineError::InvalidReport { .. } => {
            "抽帧参数不合法"
        }
        PipelineError::OutputDestinationExists { .. } => "素材目录已存在抽帧结果",
        PipelineError::FrameMissing { .. }
        | PipelineError::FrameNotRegular { .. }
        | PipelineError::FrameEmpty { .. }
        | PipelineError::FrameTooLarge { .. } => "抽出的帧不可用",
        PipelineError::Io { source, .. } => io_summary(source),
        PipelineError::Adapter { .. } => "模型推理失败",
        PipelineError::Cancelled { .. } => "任务已取消",
        PipelineError::ToolFailed { .. } | PipelineError::ToolSpawn { .. } => "ffmpeg 抽帧失败",
        PipelineError::ToolTimedOut { .. } => "ffmpeg 抽帧超时",
        PipelineError::ToolOutputTooLarge { .. } => "ffmpeg 输出过大",
        PipelineError::ReportJson { .. }
        | PipelineError::ReportNotRegular { .. }
        | PipelineError::ReportTooLarge { .. } => "质检报告写入失败",
        PipelineError::PublishFailed { .. } | PipelineError::PublishRollbackFailed { .. } => {
            "抽帧结果发布失败"
        }
        PipelineError::QualityRejected { .. } => "抽帧质检未通过",
    }
}

fn anomaly_error_code(code: AnomalyCode) -> ErrorCode {
    match code {
        AnomalyCode::FaceNotFound | AnomalyCode::MultipleFaces | AnomalyCode::BboxOutOfBounds => {
            ErrorCode::FaceNotFound
        }
        AnomalyCode::LandmarkInvalid => ErrorCode::LandmarkInvalid,
        AnomalyCode::BlurredFrame
        | AnomalyCode::FrameDecodeFailed
        | AnomalyCode::FrameWriteFailed => ErrorCode::MediaInvalid,
        AnomalyCode::ModelFailed => ErrorCode::ModelIncompatible,
    }
}

fn anomaly_summary(code: AnomalyCode) -> &'static str {
    match code {
        AnomalyCode::FaceNotFound => "有帧未检测到人脸",
        AnomalyCode::MultipleFaces => "有帧检测到多张人脸",
        AnomalyCode::BboxOutOfBounds => "人脸框超出画面范围",
        AnomalyCode::LandmarkInvalid => "关键点不合法",
        AnomalyCode::BlurredFrame => "有帧过于模糊",
        AnomalyCode::FrameDecodeFailed => "有帧无法解码",
        AnomalyCode::FrameWriteFailed => "有帧写入失败",
        AnomalyCode::ModelFailed => "模型推理失败",
    }
}
```

In `src/lib.rs`, extend the `error_map` re-export.

```rust
pub use error_map::{
    is_media_cancellation, is_pipeline_cancellation, media_task_error, pipeline_task_error,
    project_task_error, quality_task_error,
};
```

- [ ] **Step 4: Run the tests and watch them pass**

Run: `cargo test -p feathertalk-worker --test error_mapping`, then `cargo clippy -p feathertalk-worker --all-targets -- -D warnings`.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/Cargo.toml rust/crates/feathertalk-worker/src/error_map.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/tests/error_mapping.rs
git commit -m "feat(worker): map pipeline errors onto protocol error codes"
```

---

### Task 9: Serialise the quality report for the extract_frames result

**Files:**
- Modify: `rust/crates/feathertalk-frame-pipeline/src/model.rs`
- Create: `rust/crates/feathertalk-worker/src/quality_result.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs`
- Test: `rust/crates/feathertalk-frame-pipeline/tests/contracts.rs`
- Test: `rust/crates/feathertalk-worker/tests/quality_result.rs` (new)

**Interfaces:**
- Consumes: `FramePipelineSpec` with the relaxed containment rule from Task 3, `QualityReport`, and the `feathertalk-frame-pipeline` dependency Task 8 added to the worker.
- Produces: `FramePipelineSpec::frames_dir() -> PathBuf`, `FramePipelineSpec::landmarks_dir() -> PathBuf`, and

```rust
pub fn quality_to_json(spec: &FramePipelineSpec, report: &QualityReport) -> serde_json::Value
```

  re-exported from the worker's `lib.rs`. Task 11 calls it on the success path.

**Why:** the payload names all three directories and the report file because the CLI and the later asset-locking slice both need them, and re-deriving the layout from `project_dir` would copy layout knowledge into every caller. Three things are left out on purpose: the per-frame array (1511 records would push a single-line JSON event past hundreds of kilobytes; the detail is in `quality.json`), `accepted_count` (publish only succeeds when every frame is accepted, so on this path it equals `frame_count`), and `anomalies` (empty on success, and no result payload is produced on failure). The two directory getters land on the spec rather than in the worker because `frame_path`/`landmark_path` already own that layout; a second copy in the worker would be a second source of truth.

- [ ] **Step 1: Write the failing tests**

Append to `rust/crates/feathertalk-frame-pipeline/tests/contracts.rs`, which already has the `spec()` helper over `C:\project\assets`.

```rust
#[test]
fn valid_spec_exposes_the_two_artifact_directories() {
    let value = spec();
    assert_eq!(
        value.frames_dir(),
        PathBuf::from(r"C:\project\assets\frames")
    );
    assert_eq!(
        value.landmarks_dir(),
        PathBuf::from(r"C:\project\assets\landmarks")
    );
    assert_eq!(value.frames_dir().join("000000.jpg"), value.frame_path(0));
    assert_eq!(
        value.landmarks_dir().join("000002.lms"),
        value.landmark_path(2)
    );
}
```

Create `rust/crates/feathertalk-worker/tests/quality_result.rs`.

```rust
use std::path::PathBuf;

use feathertalk_frame_pipeline::{FramePipelineSpec, FrameQuality, QualityReport};
use feathertalk_worker::quality_to_json;

/// The layout the command actually produces: the video is the direct child of
/// the output root that Task 3 made legal.
fn spec() -> FramePipelineSpec {
    FramePipelineSpec::new(
        PathBuf::from(r"C:\project\assets\video_25fps.mp4"),
        PathBuf::from(r"C:\project\assets"),
        2,
        1280,
        720,
    )
    .unwrap()
}

fn frame(index: u64) -> FrameQuality {
    FrameQuality::new(
        index,
        format!("frames/{index:06}.jpg"),
        format!("landmarks/{index:06}.lms"),
        1024,
        "a".repeat(64),
        "b".repeat(64),
        0.9,
        [0.0, 0.0, 100.0, 100.0],
        30.0,
    )
    .unwrap()
}

fn report() -> QualityReport {
    QualityReport::new(2, vec![frame(0), frame(1)], Vec::new()).unwrap()
}

#[test]
fn the_payload_names_every_published_location() {
    let value = quality_to_json(&spec(), &report());

    assert_eq!(value["output_dir"], r"C:\project\assets");
    assert_eq!(value["frames_dir"], r"C:\project\assets\frames");
    assert_eq!(value["landmarks_dir"], r"C:\project\assets\landmarks");
    assert_eq!(value["quality_report"], r"C:\project\assets\quality.json");
    assert_eq!(value["frame_count"], 2);
    assert_eq!(value["frame_width"], 1280);
    assert_eq!(value["frame_height"], 720);
}

#[test]
fn the_payload_omits_the_per_frame_detail() {
    let value = quality_to_json(&spec(), &report());

    let object = value.as_object().expect("the payload is an object");
    assert_eq!(object.len(), 7);
    assert!(object.get("frames").is_none());
    assert!(object.get("anomalies").is_none());
    assert!(object.get("accepted_count").is_none());
}
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `cargo test -p feathertalk-frame-pipeline --test contracts`, then `cargo test -p feathertalk-worker --test quality_result`.

Expected: `error[E0599]: no method named 'frames_dir' found for struct 'FramePipelineSpec'` from the first, and `error[E0432]: unresolved import 'feathertalk_worker::quality_to_json'` from the second.

- [ ] **Step 3: Implement**

In `frame-pipeline/src/model.rs`, add the two directory getters above `frame_path` and rebuild the two file getters on top of them, so the two directory names appear once each.

```rust
    pub fn frames_dir(&self) -> PathBuf {
        self.output_root.join("frames")
    }

    pub fn landmarks_dir(&self) -> PathBuf {
        self.output_root.join("landmarks")
    }

    pub fn frame_path(&self, index: u64) -> PathBuf {
        self.frames_dir().join(format!("{index:06}.jpg"))
    }

    pub fn landmark_path(&self, index: u64) -> PathBuf {
        self.landmarks_dir().join(format!("{index:06}.lms"))
    }
```

Create `rust/crates/feathertalk-worker/src/quality_result.rs`.

```rust
use std::path::Path;

use feathertalk_frame_pipeline::{FramePipelineSpec, QualityReport};
use serde_json::{Value, json};

/// Shapes a published frame set as the JSON object a `completed` event carries.
///
/// Like a normalization and unlike a probe, the payload names the locations:
/// the caller asked for a project directory and the worker chose the layout
/// inside it, so a later task would otherwise have to guess. The per-frame
/// array is deliberately absent -- one JSON line per task must stay small, and
/// `quality.json` at the reported path holds every record.
pub fn quality_to_json(spec: &FramePipelineSpec, report: &QualityReport) -> Value {
    json!({
        "output_dir": path_text(spec.output_root()),
        "frames_dir": path_text(&spec.frames_dir()),
        "landmarks_dir": path_text(&spec.landmarks_dir()),
        "quality_report": path_text(&spec.quality_path()),
        "frame_count": report.frame_count(),
        "frame_width": spec.image_width(),
        "frame_height": spec.image_height(),
    })
}

fn path_text(path: &Path) -> String {
    path.display().to_string()
}
```

In `worker/src/lib.rs`, declare `mod quality_result;` between `probe_result` and `reporter`, and add `pub use quality_result::quality_to_json;` in the matching position among the re-exports.

- [ ] **Step 4: Run the tests and watch them pass**

Run: `cargo test -p feathertalk-frame-pipeline --test contracts`, `cargo test -p feathertalk-worker --test quality_result`, then `cargo test -p feathertalk-frame-pipeline` to confirm the `frame_path`/`landmark_path` refactor kept every publish and pipeline assertion green.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-frame-pipeline/src/model.rs rust/crates/feathertalk-frame-pipeline/tests/contracts.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/src/quality_result.rs rust/crates/feathertalk-worker/tests/quality_result.rs
git commit -m "feat(worker): serialise the quality report for the extract_frames result"
```

---

### Task 10: Load the SCRFD and PFLD models for the frame pipeline

**Files:**
- Modify: `rust/crates/feathertalk-frame-adapters/src/lib.rs`
- Modify: `rust/crates/feathertalk-worker/Cargo.toml`
- Create: `rust/crates/feathertalk-worker/src/models.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs`
- Test: `rust/crates/feathertalk-worker/tests/models.rs` (new)

**Interfaces:**
- Consumes: `ModelToolchain` from Task 6; `FrameImageCache::new()`, `JpegFrameDecoder::new(Arc<FrameImageCache>)`, `ScrfdFaceDetector::<B>::load(&ScrfdArtifactPaths, B::Device, Arc<FrameImageCache>)`, `PfldLandmarkPredictor::<B>::load(&Path, B::Device, Arc<FrameImageCache>)`; `feathertalk_models::backend::CpuBackend`.
- Produces: `pub use feathertalk_scrfd::ScrfdArtifactPaths;` from `feathertalk-frame-adapters`, and

```rust
pub struct FrameModels { /* decoder, detector, predictor */ }

impl FrameModels {
    pub fn load(models: &ModelToolchain) -> Result<Self, PipelineError>
    pub fn decoder(&self) -> &dyn FrameDecoder
    pub fn detector(&self) -> &dyn FaceDetector
    pub fn predictor(&self) -> &dyn LandmarkPredictor
}
```

  re-exported from the worker's `lib.rs`. Task 11 loads it once per job and hands the three trait objects to `execute_extract_frames`.

**Why:** `ScrfdArtifactPaths` appears in `ScrfdFaceDetector::load`'s public signature but is not re-exported, so that function is currently uncallable from outside `feathertalk-frame-adapters`; exporting it is a prerequisite, not a convenience. The three adapters share one `Arc<FrameImageCache>` so that the detector and the predictor reuse the pixels the decoder already decoded, which is the arrangement the adapter tests certify. PFLD gets its own 64 MiB-stack thread: loading the GhostOne graph moves a 125 768-byte module struct through several frames and overruns a default thread stack, and both `feathertalk-frame-adapters/tests/pfld_model.rs` and `feathertalk-weights/src/pfld/mod.rs` already solve it this way. The predictor stays boxed so the value crosses the thread boundary without another large stack copy. Loading lives in its own module rather than in `extract_frames.rs` because it is the one part of the command that touches burn, and keeping it separate means the command tests never need weights.

- [ ] **Step 1: Write the failing tests**

Create `rust/crates/feathertalk-worker/tests/models.rs`. The committed artifacts are hermetic and small (3.3 MiB for SCRFD, 3.8 MiB for PFLD), so this test is not gated.

```rust
use std::path::{Path, PathBuf};

use feathertalk_frame_pipeline::PipelineError;
use feathertalk_worker::{FrameModels, WorkerConfig};

/// The artifact directories committed one and two crates over. `FrameModels`
/// names `manifest.json` and `model.safetensors` inside the SCRFD one, and
/// hands the PFLD one to `PfldRuntime::load` whole.
fn scrfd_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../feathertalk-scrfd/artifacts/scrfd_2_5g")
}

fn pfld_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../feathertalk-pfld/artifacts/pfld_ghost_one")
}

fn config(scrfd: &Path, pfld: &Path) -> WorkerConfig {
    WorkerConfig::from_values_with_models(
        None,
        None,
        None,
        Some(scrfd.display().to_string()),
        Some(pfld.display().to_string()),
    )
}

#[test]
fn the_committed_artifacts_load_into_three_live_adapters() {
    let config = config(&scrfd_dir(), &pfld_dir());
    let models = FrameModels::load(config.models().expect("both directories are absolute"))
        .expect("the committed artifacts load");

    // The three accessors hand out live trait objects. Decoding a path that
    // does not exist proves the decoder is wired without needing pixels.
    let error = models
        .decoder()
        .decode(0, Path::new(r"C:\missing\000000.jpg"))
        .expect_err("a missing frame cannot decode");
    assert!(matches!(error, PipelineError::Io { .. }), "{error:?}");
}

#[test]
fn a_directory_without_scrfd_artifacts_reports_an_adapter_failure() {
    let empty = tempfile::tempdir().unwrap();
    let config = config(empty.path(), &pfld_dir());

    let error = FrameModels::load(config.models().expect("both directories are absolute"))
        .expect_err("an empty directory has no manifest");

    match error {
        PipelineError::Adapter { component, message } => {
            assert_eq!(component, "scrfd");
            assert!(!message.is_empty());
        }
        other => panic!("expected an adapter failure, got {other:?}"),
    }
}

#[test]
fn a_directory_without_pfld_artifacts_reports_an_adapter_failure() {
    let empty = tempfile::tempdir().unwrap();
    let config = config(&scrfd_dir(), empty.path());

    let error = FrameModels::load(config.models().expect("both directories are absolute"))
        .expect_err("an empty directory has no manifest");

    match error {
        PipelineError::Adapter { component, .. } => assert_eq!(component, "pfld"),
        other => panic!("expected an adapter failure, got {other:?}"),
    }
}
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `cargo test -p feathertalk-worker --test models`.

Expected: `error[E0432]: unresolved import 'feathertalk_frame_pipeline::PipelineError'` is already satisfied by Task 8, so the failure is `error[E0432]: unresolved import 'feathertalk_worker::FrameModels'`.

- [ ] **Step 3: Implement**

In `feathertalk-frame-adapters/src/lib.rs`, add the re-export as its own line after `pub use scrfd::{...}`. It has to be spelled against the upstream crate rather than folded into the `scrfd` line: `src/scrfd.rs` imports the type with a private `use`, and re-exporting a private import does not compile.

```rust
/// Re-exported because it appears in [`ScrfdFaceDetector::load`]'s signature.
/// Without it the function is public but uncallable from another crate.
pub use feathertalk_scrfd::ScrfdArtifactPaths;
```

In `worker/Cargo.toml`, add the two remaining dependencies, keeping the list alphabetical alongside the `feathertalk-frame-pipeline` entry Task 8 added.

```toml
feathertalk-frame-adapters = { path = "../feathertalk-frame-adapters" }
feathertalk-models = { path = "../feathertalk-models" }
```

Create `rust/crates/feathertalk-worker/src/models.rs`.

```rust
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use feathertalk_frame_adapters::{
    FrameImageCache, JpegFrameDecoder, PfldLandmarkPredictor, ScrfdArtifactPaths,
    ScrfdFaceDetector,
};
use feathertalk_frame_pipeline::{FaceDetector, FrameDecoder, LandmarkPredictor, PipelineError};
use feathertalk_models::backend::CpuBackend;

use crate::ModelToolchain;

/// Loading the GhostOne graph moves a 125 768-byte module struct through
/// several frames, which overruns a default thread stack. The precedents are
/// `feathertalk-frame-adapters/tests/pfld_model.rs` and
/// `feathertalk-weights/src/pfld/mod.rs`: a dedicated big-stack thread and a
/// boxed return slot.
const PREDICTOR_LOAD_STACK_BYTES: usize = 64 * 1024 * 1024;

/// The three adapters one `extract_frames` job needs, loaded together.
///
/// They share one image cache so the detector and the predictor reuse the
/// pixels the decoder already produced, which is the arrangement the adapter
/// parity tests certify. Only the CPU backend is loaded: this slice does not
/// offer a GPU path.
pub struct FrameModels {
    decoder: JpegFrameDecoder,
    detector: ScrfdFaceDetector<CpuBackend>,
    predictor: Box<PfldLandmarkPredictor<CpuBackend>>,
}

impl FrameModels {
    pub fn load(models: &ModelToolchain) -> Result<Self, PipelineError> {
        let cache = Arc::new(FrameImageCache::new());
        let decoder = JpegFrameDecoder::new(Arc::clone(&cache));
        let detector = ScrfdFaceDetector::<CpuBackend>::load(
            &scrfd_paths(models.scrfd_dir()),
            Default::default(),
            Arc::clone(&cache),
        )?;
        let predictor = load_predictor(models.pfld_dir().to_owned(), cache)?;
        Ok(Self {
            decoder,
            detector,
            predictor,
        })
    }

    pub fn decoder(&self) -> &dyn FrameDecoder {
        &self.decoder
    }

    pub fn detector(&self) -> &dyn FaceDetector {
        &self.detector
    }

    pub fn predictor(&self) -> &dyn LandmarkPredictor {
        self.predictor.as_ref()
    }
}

/// SCRFD takes the manifest and the weights separately; PFLD takes the
/// directory. The two file names are fixed by the importer that wrote them.
fn scrfd_paths(dir: &Path) -> ScrfdArtifactPaths {
    ScrfdArtifactPaths {
        manifest: dir.join("manifest.json"),
        weights: dir.join("model.safetensors"),
    }
}

fn load_predictor(
    artifacts: PathBuf,
    cache: Arc<FrameImageCache>,
) -> Result<Box<PfldLandmarkPredictor<CpuBackend>>, PipelineError> {
    std::thread::Builder::new()
        .name("pfld-predictor-load".to_owned())
        .stack_size(PREDICTOR_LOAD_STACK_BYTES)
        .spawn(move || {
            PfldLandmarkPredictor::<CpuBackend>::load(&artifacts, Default::default(), cache)
                .map(Box::new)
        })
        .map_err(|error| adapter_failure(format!("spawning the loader thread failed: {error}")))?
        .join()
        .map_err(|_| adapter_failure("the loader thread panicked".to_owned()))?
}

fn adapter_failure(message: String) -> PipelineError {
    PipelineError::Adapter {
        component: "pfld",
        message,
    }
}
```

In `worker/src/lib.rs`, declare `mod models;` between `handshake` and `normalize_result`, and add `pub use models::FrameModels;` in the matching position among the re-exports.

- [ ] **Step 4: Run the tests and watch them pass**

Run: `cargo test -p feathertalk-worker --test models`, then `cargo clippy -p feathertalk-worker -p feathertalk-frame-adapters --all-targets -- -D warnings`. The first run also compiles burn into the worker for the first time, so expect several minutes.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-frame-adapters/src/lib.rs rust/crates/feathertalk-worker/Cargo.toml rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/src/models.rs rust/crates/feathertalk-worker/tests/models.rs
git commit -m "feat(worker): load the SCRFD and PFLD models for the frame pipeline"
```

---

### Task 11: Run the extract_frames command

**Files:**
- Modify: `rust/crates/feathertalk-worker/Cargo.toml`
- Create: `rust/crates/feathertalk-worker/src/extract_frames.rs`
- Modify: `rust/crates/feathertalk-worker/src/commands.rs`
- Modify: `rust/crates/feathertalk-worker/src/error_map.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs`
- Test: `rust/crates/feathertalk-worker/tests/extract_frames.rs` (new)

**Interfaces:**
- Consumes: `FramePipelineSpec::new` with Task 3's containment rule; `extract_frames_observed` and `PipelineObserver`/`PipelinePhase` from Task 4; `evaluate_frames_observed` from Task 5; `WorkerConfig::models()` from Task 6; `pipeline_task_error`, `quality_task_error`, and `is_pipeline_cancellation` from Task 8; `quality_to_json` from Task 9; `FrameModels::load` and its three getters from Task 10; and from the existing crates `FrameExtractor::new`, `publish_frame_artifacts`, `validate_input`, `probe_media_with_runner`, `commands::media_failure`, `commands::unsupported`, `error_map::clamp`.
- Produces: the `Request::ExtractFrames` arm of `execute_with_runner`, `pub(crate)` visibility on `media_failure`, `unsupported`, and `clamp`, and

```rust
#[allow(clippy::too_many_arguments)]
pub fn execute_extract_frames<M, F>(
    params: &ExtractFramesParams,
    config: &WorkerConfig,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    media_runner: &M,
    frame_runner: &F,
    decoder: &dyn FrameDecoder,
    detector: &dyn FaceDetector,
    predictor: &dyn LandmarkPredictor,
) -> CommandOutcome
where
    M: feathertalk_media::ProcessRunner + ?Sized,
    F: feathertalk_frame_pipeline::ProcessRunner + ?Sized,
```

  re-exported from the crate root. Task 12 gates the handshake on the same two halves of the configuration, Task 13 drives it from the CLI, and Task 14 runs it against the real toolchain.

**Why:** every piece of the pipeline is already tested on its own, so what is left here is the worker's own share of the work: admission, progress bridging, and error translation. Admission is a bespoke check rather than `feathertalk_project::validate_project_dir`, which demands the finished asset set -- including the `assets/frames` and `assets/landmarks` directories this command is about to create -- and would therefore reject every project that has not been extracted yet.

The nine parameters are all injection points. Both runners are separate because the two crates define their own `ProcessRunner`, `CommandSpec`, and `ProcessOutput`, and the models arrive as three trait objects instead of one `FrameModels` so a test can drive the whole command without loading 7 MB of weights; `FrameQuality::new` in `frame-pipeline/src/model.rs:180` sets the precedent for the `allow`. Both are generic over `?Sized` because `execute_with_runner` holds a `&R` with exactly that bound, which cannot coerce to `&dyn`.

The `config.media()` guard sits inside this function even though the arm in `execute_with_runner` already checks `config.models()`: the function is public and re-exported, so a direct caller has to get the same `unsupported` answer that a request would.

- [ ] **Step 1: Write the failing tests**

Create `rust/crates/feathertalk-worker/tests/extract_frames.rs`. The two runners stand in for the two toolchains: `MediaRunner` answers every ffprobe call with the same JSON, and `FfmpegRunner` stands in for the `image2` muxer the way the pipeline's own tests do.

```rust
use std::{ffi::OsStr, fs, path::Path, sync::Mutex, time::Duration};

use feathertalk_domain::{ErrorCode, ExtractFramesParams, Progress, TaskError, TaskStage};
use feathertalk_frame_pipeline::{
    self as pipeline, DecodedFrame, FaceDetection, FaceDetector, FrameDecoder, LandmarkPredictor,
    PipelineError,
};
use feathertalk_media::{CancellationToken, CommandSpec, MediaError, ProcessOutput, ProcessRunner};
use feathertalk_pfld::{CropGeometry, PFLDLandmarks, decode_landmarks};
use feathertalk_worker::{
    CommandOutcome, NoReporter, TaskReporter, WorkerConfig, execute_extract_frames,
};
use tempfile::TempDir;

/// Answers every ffprobe call with the same report, so a test only has to say
/// how many frames the video has and what frame rate it claims.
struct MediaRunner {
    probe: Vec<u8>,
    calls: Mutex<usize>,
}

impl MediaRunner {
    fn new(frame_count: u64, frame_rate: &str) -> Self {
        let probe = serde_json::json!({
            "format": { "format_name": "mov,mp4", "duration": "2.0" },
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "pix_fmt": "yuv420p",
                    "width": 640,
                    "height": 480,
                    "avg_frame_rate": frame_rate,
                    "nb_read_frames": frame_count.to_string(),
                    "duration": "2.0"
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "sample_fmt": "fltp",
                    "sample_rate": "48000",
                    "channels": 2,
                    "duration": "2.0"
                }
            ]
        });
        Self {
            probe: probe.to_string().into_bytes(),
            calls: Mutex::new(0),
        }
    }

    fn call_count(&self) -> usize {
        *self.calls.lock().unwrap()
    }
}

impl ProcessRunner for MediaRunner {
    fn run(&self, _command: &CommandSpec, _timeout: Duration) -> Result<ProcessOutput, MediaError> {
        *self.calls.lock().unwrap() += 1;
        Ok(ProcessOutput::new(Some(0), self.probe.clone(), Vec::new()))
    }
}

/// Stands in for ffmpeg's `image2` muxer: the chunk command ends with a
/// `%06d.jpg` pattern, and `-start_number` plus `-frames:v` say which indices
/// that pattern expands to.
struct FfmpegRunner {
    chunks: Mutex<Vec<(u64, u64)>>,
}

impl FfmpegRunner {
    fn new() -> Self {
        Self {
            chunks: Mutex::new(Vec::new()),
        }
    }

    /// One `(first_index, count)` pair per invocation.
    fn chunks(&self) -> Vec<(u64, u64)> {
        self.chunks.lock().unwrap().clone()
    }
}

impl pipeline::ProcessRunner for FfmpegRunner {
    fn run(
        &self,
        command: &pipeline::CommandSpec,
        _timeout: Duration,
    ) -> Result<pipeline::ProcessOutput, PipelineError> {
        let pattern = Path::new(command.arguments().last().unwrap());
        let directory = pattern.parent().unwrap().to_owned();
        let first = flag_number(command, "-start_number");
        let count = flag_number(command, "-frames:v");
        for index in first..first + count {
            fs::write(directory.join(format!("{index:06}.jpg")), b"jpeg").unwrap();
        }
        self.chunks.lock().unwrap().push((first, count));
        Ok(pipeline::ProcessOutput::new(Some(0), Vec::new(), Vec::new()))
    }
}

/// The same helper `frame-pipeline/tests/support/mod.rs` owns; a test binary in
/// this crate cannot reach the other crate's test module.
fn flag_number(command: &pipeline::CommandSpec, flag: &str) -> u64 {
    let arguments = command.arguments();
    let position = arguments
        .iter()
        .position(|argument| argument.as_os_str() == OsStr::new(flag))
        .unwrap_or_else(|| panic!("{flag} is missing from the frame command"));
    arguments[position + 1]
        .to_str()
        .unwrap_or_else(|| panic!("{flag} carries non-UTF-8 text"))
        .parse()
        .unwrap_or_else(|_| panic!("{flag} must carry a number"))
}
```

The three models are the fakes `frame-pipeline/tests/evaluation.rs` already owns, with one change: this predictor answers every call instead of handing out a single stored `PFLDLandmarks`, because a batch here holds three frames. `Recorder` keeps every reported event in order, so one test can assert the whole sequence rather than a single stage.

```rust
struct Decoder {
    blur: f64,
}

impl FrameDecoder for Decoder {
    fn decode(&self, _index: u64, path: &Path) -> Result<DecodedFrame, PipelineError> {
        Ok(DecodedFrame::new(path.to_owned(), 640, 480, self.blur).unwrap())
    }
}

struct Detector {
    detections: Vec<FaceDetection>,
}

impl FaceDetector for Detector {
    fn detect(&self, _frame: &DecodedFrame) -> Result<Vec<FaceDetection>, PipelineError> {
        Ok(self.detections.clone())
    }
}

struct Predictor {
    landmarks: PFLDLandmarks,
}

impl LandmarkPredictor for Predictor {
    fn predict(
        &self,
        _frame: &DecodedFrame,
        _face: &FaceDetection,
    ) -> Result<PFLDLandmarks, PipelineError> {
        Ok(self.landmarks.clone())
    }
}

/// A blur variance well above `BLUR_VARIANCE_THRESHOLD`.
fn decoder() -> Decoder {
    Decoder { blur: 30.0 }
}

/// One face, scored above `FACE_CONFIDENCE_THRESHOLD`, well inside a 640x480
/// frame.
fn detector() -> Detector {
    Detector {
        detections: vec![FaceDetection {
            bbox: [12.0, 12.0, 400.0, 350.0],
            score: 0.9,
            keypoints: [[0.0, 0.0]; 5],
        }],
    }
}

fn predictor() -> Predictor {
    Predictor {
        landmarks: decode_landmarks(
            &vec![0.5; 220],
            &vec![0.0; 220],
            CropGeometry {
                width: 640,
                height: 480,
                offset_x: 0,
                offset_y: 0,
            },
        )
        .unwrap(),
    }
}

struct Recorder {
    events: Mutex<Vec<(TaskStage, Option<Progress>)>>,
}

impl Recorder {
    fn new() -> Self {
        Self {
            events: Mutex::new(Vec::new()),
        }
    }

    fn events(&self) -> Vec<(TaskStage, Option<Progress>)> {
        self.events.lock().unwrap().clone()
    }
}

impl TaskReporter for Recorder {
    fn report(&self, stage: TaskStage, progress: Option<Progress>) {
        self.events.lock().unwrap().push((stage, progress));
    }
}
```

The fixtures leave each test with only the thing it varies. `Case` carries a working default for every injection point, so a test that is about admission does not have to describe the models.

```rust
/// A project directory that passes admission: a real directory holding
/// `project.json`, with the normalised video inside `assets/`.
fn project() -> (TempDir, ExtractFramesParams) {
    let root = tempfile::tempdir().unwrap();
    let project_dir = root.path().join("project");
    let assets = project_dir.join("assets");
    fs::create_dir_all(&assets).unwrap();
    // Only the file's presence is checked; nothing here parses the manifest.
    fs::write(project_dir.join("project.json"), b"{}").unwrap();
    let video = assets.join("video_25fps.mp4");
    fs::write(&video, b"video").unwrap();
    (root, ExtractFramesParams { project_dir, video })
}

/// The media half of the configuration. Neither binary is ever executed --
/// both runners are fakes -- and the models arrive as arguments, so the model
/// directories play no part here.
fn config() -> WorkerConfig {
    let root = std::env::current_dir().unwrap();
    WorkerConfig::from_values(
        Some(root.join("ffprobe-test").display().to_string()),
        Some(root.join("ffmpeg-test").display().to_string()),
        Some("10000".to_owned()),
    )
}

/// Everything a test varies, with a working default for each.
struct Case {
    config: WorkerConfig,
    token: CancellationToken,
    media: MediaRunner,
    frames: FfmpegRunner,
    detector: Detector,
}

impl Case {
    fn new() -> Self {
        Self {
            config: config(),
            token: CancellationToken::new(),
            media: MediaRunner::new(3, "25/1"),
            frames: FfmpegRunner::new(),
            detector: detector(),
        }
    }

    fn run(&self, params: &ExtractFramesParams, reporter: &dyn TaskReporter) -> CommandOutcome {
        execute_extract_frames(
            params,
            &self.config,
            &self.token,
            reporter,
            &self.media,
            &self.frames,
            &decoder(),
            &self.detector,
            &predictor(),
        )
    }
}

fn progress(completed: u64, total: u64) -> Option<Progress> {
    Some(Progress {
        completed,
        total: Some(total),
    })
}

fn file_count(directory: &Path) -> usize {
    fs::read_dir(directory).unwrap().count()
}

fn expect_failure(outcome: CommandOutcome) -> TaskError {
    match outcome {
        CommandOutcome::Failed(error) => error,
        other => panic!("expected a failure, got {other:?}"),
    }
}
```

The eight tests cover the published result and the event sequence, the four admission rejections, cancellation, a quality rejection, and a build without a media toolchain.

```rust
#[test]
fn three_frames_are_published_and_every_stage_is_reported() {
    let (_root, params) = project();
    let case = Case::new();
    let recorder = Recorder::new();

    let result = match case.run(&params, &recorder) {
        CommandOutcome::Completed(Some(result)) => result,
        other => panic!("expected a completed command, got {other:?}"),
    };

    let assets = params.project_dir.join("assets");
    let frames = assets.join("frames");
    let landmarks = assets.join("landmarks");
    assert_eq!(result["output_dir"], assets.display().to_string());
    assert_eq!(result["frames_dir"], frames.display().to_string());
    assert_eq!(result["landmarks_dir"], landmarks.display().to_string());
    assert_eq!(
        result["quality_report"],
        assets.join("quality.json").display().to_string()
    );
    assert_eq!(result["frame_count"], 3);
    assert_eq!(result["frame_width"], 640);
    assert_eq!(result["frame_height"], 480);
    assert_eq!(file_count(&frames), 3);
    assert_eq!(file_count(&landmarks), 3);
    assert!(assets.join("quality.json").is_file());
    // `FRAME_CHUNK` is 250, so three frames are a single invocation.
    assert_eq!(case.frames.chunks(), vec![(0, 3)]);
    assert_eq!(
        recorder.events(),
        vec![
            (TaskStage::Preparing, None),
            (TaskStage::ExtractingFrames, progress(3, 3)),
            (TaskStage::DetectingFaces, progress(0, 3)),
            (TaskStage::DetectingFaces, progress(1, 3)),
            (TaskStage::DetectingFaces, progress(2, 3)),
        ]
    );
}

#[test]
fn a_video_that_is_not_25fps_is_rejected_before_ffmpeg_runs() {
    let (_root, params) = project();
    let mut case = Case::new();
    case.media = MediaRunner::new(3, "30/1");

    let error = expect_failure(case.run(&params, &NoReporter));

    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.summary, "抽帧要求 25fps 的归一化视频");
    assert!(error.detail.contains("30/1"), "detail was {}", error.detail);
    assert_eq!(case.media.call_count(), 1);
    assert!(case.frames.chunks().is_empty());
}

#[test]
fn a_missing_project_directory_is_rejected_before_ffprobe_runs() {
    let (root, params) = project();
    let params = ExtractFramesParams {
        project_dir: root.path().join("absent"),
        ..params
    };
    let case = Case::new();

    let error = expect_failure(case.run(&params, &NoReporter));

    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.summary, "工程目录不可用");
    assert_eq!(case.media.call_count(), 0);
}

#[test]
fn a_project_directory_without_a_manifest_is_rejected() {
    let (_root, params) = project();
    fs::remove_file(params.project_dir.join("project.json")).unwrap();
    let case = Case::new();

    let error = expect_failure(case.run(&params, &NoReporter));

    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.summary, "工程目录缺少 project.json");
    assert_eq!(case.media.call_count(), 0);
}

#[test]
fn an_existing_frames_directory_stops_the_run_and_is_left_untouched() {
    let (_root, params) = project();
    let frames = params.project_dir.join("assets").join("frames");
    fs::create_dir_all(&frames).unwrap();
    fs::write(frames.join("000000.jpg"), b"old").unwrap();
    let case = Case::new();

    let error = expect_failure(case.run(&params, &NoReporter));

    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.summary, "素材目录已存在抽帧结果");
    assert_eq!(fs::read(frames.join("000000.jpg")).unwrap(), b"old".to_vec());
    assert!(case.frames.chunks().is_empty());
}

#[test]
fn a_cancelled_token_stops_the_run_before_the_first_chunk() {
    let (_root, params) = project();
    let case = Case::new();
    case.token.cancel();

    let outcome = case.run(&params, &NoReporter);

    assert!(
        matches!(outcome, CommandOutcome::Cancelled),
        "expected a cancelled run, got {outcome:?}"
    );
    assert!(case.frames.chunks().is_empty());
}

#[test]
fn a_frame_without_a_face_fails_the_run_and_publishes_nothing() {
    let (_root, params) = project();
    let mut case = Case::new();
    case.detector = Detector {
        detections: Vec::new(),
    };

    let error = expect_failure(case.run(&params, &NoReporter));

    assert_eq!(error.code, ErrorCode::FaceNotFound);
    assert_eq!(error.summary, "有帧未检测到人脸");
    assert!(
        error.detail.contains("3 frame(s) rejected"),
        "detail was {}",
        error.detail
    );
    // The batch destructor removes the staging directory, so the project keeps
    // whatever it had before the run.
    assert!(!params.project_dir.join("assets").join("frames").exists());
}

#[test]
fn a_worker_without_a_media_toolchain_reports_the_command_as_unsupported() {
    let (_root, params) = project();
    let mut case = Case::new();
    case.config = WorkerConfig::from_values(None, None, None);

    let error = expect_failure(case.run(&params, &NoReporter));

    assert_eq!(error.code, ErrorCode::WorkerCrashed);
    assert_eq!(error.summary, "当前 worker 不支持该命令");
    assert_eq!(case.media.call_count(), 0);
}
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `cargo test -p feathertalk-worker --test extract_frames`.

Expected: FAIL to compile with `error[E0433]: failed to resolve: use of unresolved module or unlinked crate 'feathertalk_pfld'` until the dev-dependency lands, and then `error[E0432]: unresolved import 'feathertalk_worker::execute_extract_frames'`.

- [ ] **Step 3: Implement**

In `worker/Cargo.toml`, add the crate the test decodes its fake landmarks with. It belongs in the existing `[dev-dependencies]`, before `tempfile`; the worker itself reaches PFLD only through `feathertalk-frame-adapters`.

```toml
feathertalk-pfld = { path = "../feathertalk-pfld" }
```

Create `rust/crates/feathertalk-worker/src/extract_frames.rs`.

```rust
use std::{fs, path::Path};

use feathertalk_domain::{
    ErrorCode, ExtractFramesParams, Progress, TaskError, TaskKind, TaskStage,
};
use feathertalk_frame_pipeline::{
    FaceDetector, FrameDecoder, FrameExtractor, FramePipelineSpec, LandmarkPredictor,
    PipelineError, PipelineObserver, PipelinePhase, evaluate_frames_observed,
    extract_frames_observed, publish_frame_artifacts,
};
use feathertalk_media::{
    CancellationToken, MediaInput, MediaToolchain, probe_media_with_runner, validate_input,
};

use crate::{
    CommandOutcome, TaskReporter, WorkerConfig,
    commands::{media_failure, unsupported},
    error_map::clamp,
    is_pipeline_cancellation, pipeline_task_error, quality_task_error, quality_to_json,
};

/// The frame rate `normalize_media` fixes for a project video.
const TARGET_FRAME_RATE: (u32, u32) = (25, 1);

/// The manifest every project directory carries. `feathertalk-project` owns the
/// name but exports no constant for it (`src/package.rs:66`), so the literal is
/// duplicated the way `cli/src/render.rs` duplicates the worker's environment
/// variable names.
const PROJECT_MANIFEST: &str = "project.json";

/// Extract every frame of a normalised video into a project's asset directory.
///
/// The three models arrive as trait objects so a caller can drive the command
/// without loading weights; `FrameModels` in this crate supplies the real ones.
#[allow(clippy::too_many_arguments)]
pub fn execute_extract_frames<M, F>(
    params: &ExtractFramesParams,
    config: &WorkerConfig,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    media_runner: &M,
    frame_runner: &F,
    decoder: &dyn FrameDecoder,
    detector: &dyn FaceDetector,
    predictor: &dyn LandmarkPredictor,
) -> CommandOutcome
where
    M: feathertalk_media::ProcessRunner + ?Sized,
    F: feathertalk_frame_pipeline::ProcessRunner + ?Sized,
{
    let Some(media) = config.media() else {
        return CommandOutcome::Failed(unsupported(TaskKind::ExtractFrames));
    };
    // One stage before the probe: probing and model loading take seconds, and
    // the CLI would otherwise print nothing until the first chunk lands.
    reporter.report(TaskStage::Preparing, None);
    let spec = match frame_spec(params, media, media_runner) {
        Ok(spec) => spec,
        Err(outcome) => return outcome,
    };
    let extractor = match FrameExtractor::new(media.ffmpeg().to_owned(), media.timeout()) {
        Ok(extractor) => extractor,
        Err(error) => return pipeline_failure(&error),
    };
    let observer = FrameProgress { reporter, token };
    let mut batch = match extract_frames_observed(&spec, &extractor, frame_runner, &observer) {
        Ok(batch) => batch,
        Err(error) => return pipeline_failure(&error),
    };
    let evaluation = match evaluate_frames_observed(&batch, decoder, detector, predictor, &observer)
    {
        Ok(evaluation) => evaluation,
        Err(error) => return pipeline_failure(&error),
    };
    if !evaluation.is_success() {
        // Staging is still armed, so the batch destructor removes the frames
        // this run wrote and the project keeps whatever it had before.
        return CommandOutcome::Failed(quality_task_error(evaluation.anomalies()));
    }
    match publish_frame_artifacts(&spec, &mut batch, &evaluation) {
        Ok(report) => CommandOutcome::Completed(Some(quality_to_json(&spec, &report))),
        Err(error) => pipeline_failure(&error),
    }
}

/// Bridges the pipeline's observer onto the worker's reporter and token.
struct FrameProgress<'a> {
    reporter: &'a dyn TaskReporter,
    token: &'a CancellationToken,
}

impl PipelineObserver for FrameProgress<'_> {
    fn phase(&self, phase: PipelinePhase) {
        let (stage, completed, total) = match phase {
            PipelinePhase::Extracting { completed, total } => {
                (TaskStage::ExtractingFrames, completed, total)
            }
            PipelinePhase::Evaluating { completed, total } => {
                (TaskStage::DetectingFaces, completed, total)
            }
        };
        self.reporter.report(
            stage,
            Some(Progress {
                completed,
                total: Some(total),
            }),
        );
    }

    fn is_cancelled(&self) -> bool {
        self.token.is_cancelled()
    }
}

/// Admission plus the probe: everything that has to hold before the first
/// ffmpeg invocation.
fn frame_spec<M>(
    params: &ExtractFramesParams,
    media: &MediaToolchain,
    runner: &M,
) -> Result<FramePipelineSpec, CommandOutcome>
where
    M: feathertalk_media::ProcessRunner + ?Sized,
{
    check_project_dir(&params.project_dir).map_err(CommandOutcome::Failed)?;
    if !params.video.is_absolute() {
        return Err(CommandOutcome::Failed(invalid_request(
            "输入文件必须是绝对路径",
            format!("video {} is not absolute", params.video.display()),
        )));
    }
    let input = validate_input(&MediaInput {
        source: params.video.clone(),
    })
    .map_err(|error| media_failure(&error))?;
    let probe =
        probe_media_with_runner(&input, media, runner).map_err(|error| media_failure(&error))?;
    let Some(video) = probe.video() else {
        return Err(CommandOutcome::Failed(invalid_request(
            "输入文件不含视频流",
            format!("{} has no video stream", params.video.display()),
        )));
    };
    let rate = video.frame_rate();
    if (rate.numerator(), rate.denominator()) != TARGET_FRAME_RATE {
        return Err(CommandOutcome::Failed(invalid_request(
            "抽帧要求 25fps 的归一化视频",
            format!(
                "video frame rate is {}/{}, expected {}/{}",
                rate.numerator(),
                rate.denominator(),
                TARGET_FRAME_RATE.0,
                TARGET_FRAME_RATE.1
            ),
        )));
    }
    FramePipelineSpec::new(
        params.video.clone(),
        params.project_dir.join("assets"),
        video.frame_count(),
        video.width(),
        video.height(),
    )
    .map_err(|error| CommandOutcome::Failed(pipeline_task_error(&error)))
}

/// `feathertalk_project::validate_project_dir` cannot be reused here: it
/// requires the finished asset set, including the two directories this command
/// is about to create. What has to hold before extraction is narrower -- a real
/// directory carrying a manifest.
fn check_project_dir(project_dir: &Path) -> Result<(), TaskError> {
    if !project_dir.is_absolute() {
        return Err(invalid_request(
            "工程目录必须是绝对路径",
            format!("project_dir {} is not absolute", project_dir.display()),
        ));
    }
    // `symlink_metadata` does not follow links, so a symlinked directory is
    // rejected here the way `feathertalk-project` rejects one.
    let metadata = fs::symlink_metadata(project_dir).map_err(|error| {
        invalid_request(
            "工程目录不可用",
            format!("{}: {error}", project_dir.display()),
        )
    })?;
    if !metadata.is_dir() {
        return Err(invalid_request(
            "工程目录不可用",
            format!("{} is not a directory", project_dir.display()),
        ));
    }
    let manifest = project_dir.join(PROJECT_MANIFEST);
    let found = fs::symlink_metadata(&manifest)
        .map(|metadata| metadata.is_file())
        .unwrap_or(false);
    if !found {
        return Err(invalid_request(
            "工程目录缺少 project.json",
            format!("{} is missing or not a regular file", manifest.display()),
        ));
    }
    Ok(())
}

/// Every admission failure reports `MediaInvalid`: the request named a
/// directory or a video the worker cannot work with.
fn invalid_request(summary: &'static str, detail: String) -> TaskError {
    TaskError::new(
        ErrorCode::MediaInvalid,
        summary,
        &clamp(&detail),
        TaskStage::Preparing,
    )
}

/// Cancellation is not a failure: the pipeline reports it as an error and the
/// runtime needs it back as `Cancelled`.
fn pipeline_failure(error: &PipelineError) -> CommandOutcome {
    if is_pipeline_cancellation(error) {
        CommandOutcome::Cancelled
    } else {
        CommandOutcome::Failed(pipeline_task_error(error))
    }
}
```

In `worker/src/commands.rs`, widen `media_failure` and `unsupported` to `pub(crate)` so the new module reuses both, add `FrameModels`, `execute_extract_frames`, and `pipeline_task_error` to the `crate::{…}` import, and bring in the pipeline's runner under a name that cannot be mistaken for the media one this file already uses.

```rust
use feathertalk_frame_pipeline::SystemProcessRunner as FrameProcessRunner;
```

It does not have to be cancellable the way `execute`'s `CancellableProcessRunner` is: `extract_frames_observed` asks the observer between chunks, and `FrameExtractor`'s timeout bounds each single call.

Then add the arm ahead of the `other =>` fallback, so a build without models still answers `unsupported`.

```rust
        Request::ExtractFrames(params) => {
            let Some(models) = config.models() else {
                return CommandOutcome::Failed(unsupported(request.kind()));
            };
            let models = match FrameModels::load(models) {
                Ok(models) => models,
                Err(error) => return CommandOutcome::Failed(pipeline_task_error(&error)),
            };
            execute_extract_frames(
                params,
                config,
                token,
                reporter,
                runner,
                &FrameProcessRunner,
                models.decoder(),
                models.detector(),
                models.predictor(),
            )
        }
```

In `worker/src/error_map.rs`, widen `fn clamp` to `pub(crate) fn clamp`. The admission errors are built in `extract_frames.rs` but still have to respect `MAX_DETAIL_CHARS`, and a path the caller controls is exactly the kind of detail that can be long.

In `worker/src/lib.rs`, declare `mod extract_frames;` between `error_map` and `handshake`, add `pub use extract_frames::execute_extract_frames;` in the matching position among the re-exports, and name `extract_frames` in the crate doc comment's list of served commands.

- [ ] **Step 4: Run the tests and watch them pass**

Run: `cargo test -p feathertalk-worker --test extract_frames`, then `cargo test -p feathertalk-worker`, then `cargo clippy -p feathertalk-worker --all-targets -- -D warnings`.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/Cargo.toml rust/crates/feathertalk-worker/src/commands.rs rust/crates/feathertalk-worker/src/error_map.rs rust/crates/feathertalk-worker/src/extract_frames.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/tests/extract_frames.rs
git commit -m "feat(worker): run the extract_frames command"
```

---

### Task 12: Advertise extract_frames when models resolve

**Files:**
- Modify: `rust/crates/feathertalk-worker/src/handshake.rs`
- Modify: `rust/crates/feathertalk-worker/src/runtime.rs`
- Test: `rust/crates/feathertalk-worker/tests/handshake.rs`
- Test: `rust/crates/feathertalk-worker/tests/runtime.rs`

**Interfaces:**
- Consumes: `WorkerConfig::models()` and `WorkerConfig::model_rejection()` from Task 6, and `ENV_SCRFD_DIR`/`ENV_PFLD_DIR`.
- Produces: `TaskKind::ExtractFrames` in `supported_commands` when both halves of the configuration resolve, and a rejection reason that names the missing half.

**Why:** the runtime refuses every command outside `supported_commands` before it creates a task (`runtime.rs:213`), so Task 11's arm is unreachable until the handshake advertises the command. `Capabilities` stays untouched: its four flags describe training and ffmpeg, and protocol version 2 has no model flag, so `supported_commands` is the only place the answer belongs.

The reason names media before models because extraction probes the video before it touches a model: an operator who fixed the model directories but not ffmpeg would hit the same wall one step later.

- [ ] **Step 1: Write the failing tests**

Append to `rust/crates/feathertalk-worker/tests/handshake.rs`, and extend the import with `ENV_SCRFD_DIR`. The model directories never have to exist -- `required_path` only demands a non-empty absolute path.

```rust
/// Media and models both resolve, so every command in this slice is offered.
fn fully_configured() -> WorkerConfig {
    WorkerConfig::from_values_with_models(
        Some(absolute("ffprobe-test")),
        Some(absolute("ffmpeg-test")),
        None,
        Some(absolute("scrfd-test")),
        Some(absolute("pfld-test")),
    )
}

#[test]
fn a_fully_configured_worker_offers_extract_frames() {
    let config = fully_configured();
    assert_eq!(config.model_rejection(), None);
    let frame = ready_frame(&config);
    frame.validate().unwrap();
    assert_eq!(
        frame.supported_commands,
        vec![
            TaskKind::ValidateProject,
            TaskKind::ProbeMedia,
            TaskKind::NormalizeMedia,
            TaskKind::ExtractFrames
        ]
    );
    // Protocol version 2 has no model capability flag, so nothing here moves.
    assert!(frame.capabilities.ffmpeg);
    assert!(!frame.capabilities.training);
}

#[test]
fn a_media_only_worker_leaves_extract_frames_out() {
    let config = configured();
    assert!(config.models().is_none());
    assert!(
        config
            .model_rejection()
            .is_some_and(|reason| reason.contains(ENV_SCRFD_DIR))
    );
    assert!(!supported_commands(&config).contains(&TaskKind::ExtractFrames));
}

#[test]
fn models_without_a_media_toolchain_offer_nothing_new() {
    let config = WorkerConfig::from_values_with_models(
        None,
        None,
        None,
        Some(absolute("scrfd-test")),
        Some(absolute("pfld-test")),
    );
    assert!(config.models().is_some());
    assert_eq!(supported_commands(&config), vec![TaskKind::ValidateProject]);
}
```

Append to `rust/crates/feathertalk-worker/tests/runtime.rs`, keeping the two helpers next to the existing configuration helpers, and add `ExtractFramesParams` to the `feathertalk_domain` import.

```rust
/// Media and models both resolve, so `extract_frames` reaches the executor.
fn full_config() -> WorkerConfig {
    WorkerConfig::from_values_with_models(
        Some(absolute("ffprobe-test")),
        Some(absolute("ffmpeg-test")),
        None,
        Some(absolute("scrfd-test")),
        Some(absolute("pfld-test")),
    )
}

fn extract_frames_request() -> Request {
    Request::ExtractFrames(ExtractFramesParams {
        project_dir: PathBuf::from("C:/tmp/project"),
        video: PathBuf::from("C:/tmp/project/assets/video_25fps.mp4"),
    })
}

#[test]
fn a_fully_configured_worker_enables_extract_frames_in_the_handshake() {
    let frames = Harness::start(full_config(), instant_executor()).finish();

    let ServerFrame::Ready(ready) = &frames[0] else {
        panic!("the first frame must be ready: {frames:?}");
    };
    assert_eq!(
        ready.supported_commands,
        vec![
            TaskKind::ValidateProject,
            TaskKind::ProbeMedia,
            TaskKind::NormalizeMedia,
            TaskKind::ExtractFrames
        ]
    );
}

#[test]
fn extract_frames_reaches_the_executor_once_the_models_resolve() {
    let harness = Harness::start(full_config(), instant_executor());
    harness.send(&start(&task("0000002c"), extract_frames_request()));
    let frames = harness.finish();

    assert!(rejections(&frames).is_empty(), "{frames:?}");
    assert_eq!(
        stages(&frames),
        vec![
            ("1787900000000-0000002c", "preparing"),
            ("1787900000000-0000002c", "completed"),
        ]
    );
}

#[test]
fn extract_frames_is_rejected_when_the_models_are_unavailable() {
    let harness = Harness::start(media_config(), instant_executor());
    harness.send(&start(&task("0000002d"), extract_frames_request()));
    let frames = harness.finish();

    let reasons = rejections(&frames);
    assert_eq!(reasons.len(), 1, "{frames:?}");
    assert!(reasons[0].contains("extract_frames"), "{}", reasons[0]);
    // The reason has to name the variable an operator can fix.
    assert!(
        reasons[0].contains("FEATHERTALK_WORKER_SCRFD_DIR"),
        "{}",
        reasons[0]
    );
    assert!(events(&frames).is_empty());
}

#[test]
fn extract_frames_names_the_media_toolchain_before_the_models() {
    let harness = Harness::start(bare_config(), instant_executor());
    harness.send(&start(&task("0000002e"), extract_frames_request()));
    let frames = harness.finish();

    let reasons = rejections(&frames);
    assert_eq!(reasons.len(), 1, "{frames:?}");
    assert!(
        reasons[0].contains("FEATHERTALK_WORKER_FFPROBE"),
        "{}",
        reasons[0]
    );
}
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `cargo test -p feathertalk-worker --test handshake`, then `cargo test -p feathertalk-worker --test runtime`.

Expected: `assertion 'left == right' failed` on the four-element command vector in both handshake and runtime, and the three rejection tests failing because the fallback arm answers `此 worker 不支持命令 extract_frames，当前支持：…` instead of naming a variable.

- [ ] **Step 3: Implement**

In `src/handshake.rs`, nest the model gate inside the media one: extraction probes the video with ffprobe, cuts it with ffmpeg, and only then runs SCRFD and PFLD, so it needs both halves.

```rust
pub fn supported_commands(config: &WorkerConfig) -> Vec<TaskKind> {
    let mut commands = vec![TaskKind::ValidateProject];
    // Both media commands shell out to the same two binaries, so they are
    // available together or not at all.
    if config.media().is_some() {
        commands.push(TaskKind::ProbeMedia);
        commands.push(TaskKind::NormalizeMedia);
        // Extraction needs the media toolchain *and* both model directories.
        if config.models().is_some() {
            commands.push(TaskKind::ExtractFrames);
        }
    }
    commands
}
```

In `src/runtime.rs`, split the media message out of `unsupported_reason` so both commands can share it, add the `ExtractFrames` arms, and extend the `crate::{…}` import with `ENV_PFLD_DIR` and `ENV_SCRFD_DIR`.

```rust
fn unsupported_reason(request: &Request, config: &WorkerConfig) -> String {
    let slug = request.kind().as_slug();
    match request.kind() {
        // Both media commands need the same two binaries, so they share the
        // reason that names what to fix.
        TaskKind::ProbeMedia | TaskKind::NormalizeMedia => media_reason(slug, config),
        // Extraction needs both halves. Media first: it probes the video before
        // it loads a model, so that is the wall an operator would hit next.
        TaskKind::ExtractFrames if config.media().is_none() => media_reason(slug, config),
        TaskKind::ExtractFrames => model_reason(slug, config),
        // Listing `supported_commands` instead of a hard-coded set keeps this
        // message correct as later commands land.
        _ => format!(
            "此 worker 不支持命令 {slug}，当前支持：{}。",
            supported_commands(config)
                .iter()
                .copied()
                .map(TaskKind::as_slug)
                .collect::<Vec<_>>()
                .join("、")
        ),
    }
}

fn media_reason(slug: &str, config: &WorkerConfig) -> String {
    match config.media_rejection() {
        Some(rejection) => format!(
            "命令 {slug} 需要可用的媒体工具链，当前配置被拒绝：{rejection}。修正后重启 worker。"
        ),
        None => format!(
            "命令 {slug} 需要媒体工具链，请设置 {ENV_FFPROBE} 与 {ENV_FFMPEG} 后重启 worker。"
        ),
    }
}

fn model_reason(slug: &str, config: &WorkerConfig) -> String {
    match config.model_rejection() {
        Some(rejection) => format!(
            "命令 {slug} 需要可用的模型目录，当前配置被拒绝：{rejection}。修正后重启 worker。"
        ),
        None => format!(
            "命令 {slug} 需要人脸与关键点模型，请设置 {ENV_SCRFD_DIR} 与 {ENV_PFLD_DIR} 后重启 worker。"
        ),
    }
}
```

- [ ] **Step 4: Run the tests and watch them pass**

Run: `cargo test -p feathertalk-worker`, then `cargo clippy -p feathertalk-worker --all-targets -- -D warnings`.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/src/handshake.rs rust/crates/feathertalk-worker/src/runtime.rs rust/crates/feathertalk-worker/tests/handshake.rs rust/crates/feathertalk-worker/tests/runtime.rs
git commit -m "feat(worker): advertise extract_frames when models resolve"
```

---

### Task 13: Add the extract-frames subcommand

**Files:**
- Modify: `rust/crates/feathertalk-cli/src/cli.rs`
- Modify: `rust/crates/feathertalk-cli/src/run.rs`
- Modify: `rust/crates/feathertalk-cli/src/render.rs`
- Test: `rust/crates/feathertalk-cli/src/run.rs` (its inline `mod tests`)
- Test: `rust/crates/feathertalk-cli/tests/cli.rs`

**Interfaces:**
- Consumes: `ExtractFramesParams` and `Request::ExtractFrames` from `feathertalk-domain`, and the handshake Task 12 changed -- a worker without models leaves `extract_frames` out of `supported_commands`, and the client turns that into `ClientError::UnsupportedCommand` before any task starts.
- Produces: `Command::ExtractFrames { project_dir: PathBuf, video: PathBuf }`, its `build_request` arm, and the branch of `render_client_error` that names all four variables. Task 14 drives the subcommand against the real worker.

**Why:** the CLI's share is deliberately small: two positional paths, the same empty-string guard the other commands use, and one sentence of advice when the worker cannot take the job. Whether a path exists, is a project, or is decodable stays the worker's judgement -- the comment above `build_request` says so -- and a second opinion in the CLI could only disagree with the first.

The advice branch is the part that earns its keep. `extract_frames` vanishes from `supported_commands` when either half of the configuration is missing, and the message the client builds on its own names no variable at all. Listing the media pair first and the two model directories second matches the order Task 12's `media_reason` and `model_reason` answer in, so the worker and the CLI tell the same story. Both new constants are literals for the reason the existing pair already is: the CLI must not link the worker crate.

`stage_label` already names `ExtractingFrames` and `DetectingFaces`, so the progress narration needs no change, and `--json` needs none either because it forwards the worker's own frames verbatim.

- [ ] **Step 1: Write the failing tests**

Add two unit tests to the inline module in `rust/crates/feathertalk-cli/src/run.rs`, after `normalize_media_refuses_empty_arguments`.

```rust
    #[test]
    fn extract_frames_refuses_empty_arguments() {
        let error = build_request(&Command::ExtractFrames {
            project_dir: PathBuf::new(),
            video: PathBuf::from("project/assets/video_25fps.mp4"),
        })
        .expect_err("an empty project directory is refused");
        assert_eq!(error, "工程目录不能为空。");

        let error = build_request(&Command::ExtractFrames {
            project_dir: PathBuf::from("project"),
            video: PathBuf::new(),
        })
        .expect_err("an empty video is refused");
        assert_eq!(error, "输入文件不能为空。");
    }

    #[test]
    fn extract_frames_carries_both_paths() {
        let request = build_request(&Command::ExtractFrames {
            project_dir: PathBuf::from("project"),
            video: PathBuf::from("project/assets/video_25fps.mp4"),
        })
        .expect("both paths are accepted")
        .expect("extract-frames needs a task");
        let Request::ExtractFrames(params) = request else {
            panic!("extract-frames must build an ExtractFrames request");
        };
        assert_eq!(params.project_dir, PathBuf::from("project"));
        assert_eq!(
            params.video,
            PathBuf::from("project/assets/video_25fps.mp4")
        );
    }
```

Then append one process-level test to `rust/crates/feathertalk-cli/tests/cli.rs`. The fake worker needs no new scenario: `only-validate` advertises `validate_project` alone, which is exactly the shape of a worker that resolved neither toolchain.

```rust
#[test]
fn an_unsupported_extract_frames_names_the_model_variables() {
    // The fake worker advertises `validate_project` alone, so the client's
    // capability gate answers before any task starts.
    let output = run("only-validate", &["extract-frames", "project", "clip.mp4"]);
    assert_eq!(code(&output), 3);
    let text = stderr(&output);
    assert!(text.contains("extract_frames"), "{text}");
    assert!(text.contains("FEATHERTALK_WORKER_SCRFD_DIR"), "{text}");
    assert!(text.contains("FEATHERTALK_WORKER_PFLD_DIR"), "{text}");
}
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `cargo test -p feathertalk-cli`.

Expected: FAIL to compile. Each of the three `Command::ExtractFrames` literals in `run.rs` reports `error[E0599]: no variant or associated item named 'ExtractFrames' found for enum 'cli::Command'`, so no test in the crate runs -- including the one in `tests/cli.rs`, which would otherwise fail on `assert!(text.contains("extract_frames"))` because clap's own "unrecognized subcommand 'extract-frames'" spells it with a hyphen.

- [ ] **Step 3: Implement**

In `rust/crates/feathertalk-cli/src/cli.rs`, extend the enum's doc comment with the new name.

```rust
/// The task commands, kebab-cased by clap: `validate-project`, `probe-media`,
/// `normalize-media`, `extract-frames`, `capabilities`.
```

Then add the variant between `NormalizeMedia` and `Capabilities`, so clap's declaration order -- which is also `--help`'s order -- matches the domain's `TaskKind`.

```rust
    /// 抽取视频帧并检测人脸关键点
    ExtractFrames {
        /// 工程目录
        project_dir: PathBuf,
        /// 已归一化的 25fps 视频，位于工程目录的 assets 下
        video: PathBuf,
    },
```

In `src/run.rs`, add `ExtractFramesParams` to the domain import, which crosses 100 columns and therefore wraps.

```rust
use feathertalk_domain::{
    ExtractFramesParams, NormalizeMediaParams, ProbeMediaParams, ProjectDirParams, Request, TaskId,
};
```

Then add the arm to `build_request` after `NormalizeMedia`. 「输入文件」 is reused for the video so the CLI has one name for "the file you named".

```rust
        Command::ExtractFrames { project_dir, video } => {
            reject_empty(project_dir, "工程目录")?;
            reject_empty(video, "输入文件")?;
            Ok(Some(Request::ExtractFrames(ExtractFramesParams {
                project_dir: project_dir.clone(),
                video: video.clone(),
            })))
        }
```

In `src/render.rs`, add the two model directories beside the media pair.

```rust
/// The worker's variables for the two model directories, literals for the same
/// reason: `feathertalk-worker`'s `ENV_SCRFD_DIR` and `ENV_PFLD_DIR` are the
/// source of truth for these names.
const ENV_WORKER_SCRFD_DIR: &str = "FEATHERTALK_WORKER_SCRFD_DIR";
const ENV_WORKER_PFLD_DIR: &str = "FEATHERTALK_WORKER_PFLD_DIR";
```

Then give the `ClientError::UnsupportedCommand` arm a second branch. `requested` is a `&'static str` field, so `*requested` compares against a literal directly.

```rust
            if matches!(*requested, "probe_media" | "normalize_media") {
                text.push_str(&format!(
                    "\n{requested} 需要可用的 ffprobe 与 ffmpeg。请安装 ffmpeg，或用环境变量 \
                     {ENV_WORKER_FFPROBE} 与 {ENV_WORKER_FFMPEG} 指定它们的完整路径。"
                ));
            } else if *requested == "extract_frames" {
                text.push_str(&format!(
                    "\n{requested} 需要媒体工具与人脸模型。请用环境变量 {ENV_WORKER_FFPROBE}、\
                     {ENV_WORKER_FFMPEG}、{ENV_WORKER_SCRFD_DIR}、{ENV_WORKER_PFLD_DIR} \
                     指定它们的完整路径。"
                ));
            }
```

- [ ] **Step 4: Run the tests and watch them pass**

Run: `cargo test -p feathertalk-cli`, then `cargo clippy -p feathertalk-cli --all-targets -- -D warnings`.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-cli/src/cli.rs rust/crates/feathertalk-cli/src/render.rs rust/crates/feathertalk-cli/src/run.rs rust/crates/feathertalk-cli/tests/cli.rs
git commit -m "feat(cli): add the extract-frames subcommand"
```

---

### Task 14: Extract a real second end to end

**Files:**
- Test: `rust/crates/feathertalk-cli/tests/real_worker.rs`

**Interfaces:**
- Consumes: the `extract-frames` subcommand from Task 13, the handshake gate from Task 12, the result payload `quality_to_json` builds in Task 9, and the four worker variables from Task 6.
- Produces: nothing a later task consumes -- this is the slice's only end-to-end proof.

**Why:** every unit in this slice is tested against a fake -- a fake process runner, a fake detector, a fake worker binary. That is what keeps the suite fast and hermetic, and it is also what leaves one question open: whether the real ffmpeg, the real SCRFD and PFLD artifacts, the real project layout, and the CLI's own argument order agree with one another. One test answers it, and it has to be this one, because nothing smaller spans all four.

It skips instead of failing when anything it needs is absent, exactly as `a_real_clip_is_normalized_end_to_end` does. A test that fails because ffmpeg is not installed reports on the developer's machine rather than on the code, and a suite that cries wolf stops being read.

One second is the whole budget. Evaluation costs about 190 ms per frame, so 25 frames spend roughly 5 s in detection and landmarks on top of a 0.8 s cut and a sub-second extraction; the full 1511-frame demo would take four minutes and prove nothing further. The 30 s offset is not arbitrary either: the committed `demo_frame_v1` fixture measures the frame at that offset -- frame 750 -- at a 0.8108 face score and a 776.03 blur variance, far above the 0.50 and 20.0 thresholds, so the 24 frames that follow are near-certain to be accepted as well. If one is not, `quality.json` names the rejected index and the offset can move; asserting `accepted_count` is what makes that visible instead of silent.

- [ ] **Step 1: Write the failing tests**

Append one test to `rust/crates/feathertalk-cli/tests/real_worker.rs`, after `real_tool`.

```rust
/// The whole extraction, and only when the real toolchain, the imported models,
/// and the demo clip are all present. Neither this repository nor CI ships
/// ffmpeg or the model artifacts, so anything missing is a skip rather than a
/// failure: a test that fails for reasons unrelated to the code under test
/// teaches nothing.
#[test]
fn a_real_second_is_extracted_end_to_end() {
    let Some(worker) = worker_or_skip("a_real_second_is_extracted_end_to_end") else {
        return;
    };
    let (Some(ffmpeg), Some(ffprobe), Some(scrfd), Some(pfld), Some(demo)) = (
        real_tool("FFMPEG"),
        real_tool("FFPROBE"),
        real_dir("SCRFD_DIR"),
        real_dir("PFLD_DIR"),
        demo_clip(),
    ) else {
        println!(
            "skipping a_real_second_is_extracted_end_to_end: it needs \
             FEATHERTALK_WORKER_FFMPEG, FEATHERTALK_WORKER_FFPROBE, \
             FEATHERTALK_WORKER_SCRFD_DIR, FEATHERTALK_WORKER_PFLD_DIR, and \
             demo/feathertalk_demo_latest_188.mp4"
        );
        return;
    };
    let project = TempDir::new().expect("a temporary directory is available");
    let assets = project.path().join("assets");
    std::fs::create_dir_all(&assets).expect("the assets directory is writable");
    // Admission only asks that the manifest exists; reading it is
    // `validate-project`'s job, and this command runs before a project has the
    // assets that validation demands.
    std::fs::write(project.path().join("project.json"), "{}")
        .expect("the temporary manifest is writable");
    let video = assets.join("video_25fps.mp4");
    cut_one_second(&ffmpeg, &demo, &video);

    let project_arg = project.path().to_string_lossy().into_owned();
    let video_arg = video.to_string_lossy().into_owned();
    let ffmpeg_arg = ffmpeg.to_string_lossy().into_owned();
    let ffprobe_arg = ffprobe.to_string_lossy().into_owned();
    let scrfd_arg = scrfd.to_string_lossy().into_owned();
    let pfld_arg = pfld.to_string_lossy().into_owned();
    let output = run(
        &worker,
        &["extract-frames", &project_arg, &video_arg],
        &[
            ("FEATHERTALK_WORKER_FFMPEG", ffmpeg_arg.as_str()),
            ("FEATHERTALK_WORKER_FFPROBE", ffprobe_arg.as_str()),
            ("FEATHERTALK_WORKER_SCRFD_DIR", scrfd_arg.as_str()),
            ("FEATHERTALK_WORKER_PFLD_DIR", pfld_arg.as_str()),
        ],
    );
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));

    let result: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("stdout is exactly one JSON document");
    assert_eq!(result["output_dir"], assets.display().to_string());
    assert_eq!(result["frame_count"], 25);
    assert_eq!(result["frame_width"], 1280);
    assert_eq!(result["frame_height"], 720);

    assert_eq!(file_count(&assets.join("frames")), 25);
    assert_eq!(file_count(&assets.join("landmarks")), 25);
    assert!(assets.join("frames").join("000000.jpg").is_file());
    assert!(assets.join("landmarks").join("000000.lms").is_file());

    let report =
        std::fs::read_to_string(assets.join("quality.json")).expect("the report is readable");
    let report: serde_json::Value = serde_json::from_str(&report).expect("the report is JSON");
    assert_eq!(report["frame_count"], 25);
    assert_eq!(report["accepted_count"], 25);
    assert!(
        report["anomalies"]
            .as_array()
            .expect("anomalies is an array")
            .is_empty(),
        "{report}"
    );

    let narration = stderr(&output);
    assert!(narration.contains("正在提取视频帧"), "{narration}");
    assert!(narration.contains("正在检测人脸"), "{narration}");
    assert!(narration.contains("进度 25/25"), "{narration}");
}
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `cargo test -p feathertalk-cli --test real_worker`.

Expected: FAIL to compile, with `error[E0425]: cannot find function 'real_dir' in this scope` and the same error for 'demo_clip', 'cut_one_second', and 'file_count'.

- [ ] **Step 3: Implement**

Add the four helpers at the bottom of `rust/crates/feathertalk-cli/tests/real_worker.rs`, next to `real_tool`. No `use` line changes: `Path`, `PathBuf`, `Command`, `Output`, and `TempDir` are already imported.

```rust
/// A model directory from the environment, only if it is an existing directory.
fn real_dir(suffix: &str) -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var(format!("FEATHERTALK_WORKER_{suffix}")).ok()?);
    path.is_dir().then_some(path)
}

/// The demo video this repository ships, resolved from this crate's directory.
fn demo_clip() -> Option<PathBuf> {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../demo/feathertalk_demo_latest_188.mp4");
    path.is_file().then_some(path)
}

/// One second of the demo video, re-encoded at 25 fps into the project's
/// `assets/` under the name the pipeline expects. The offset is 30 s because the
/// committed `demo_frame_v1` fixture records a 0.8108 face score and a 776.03
/// blur variance for that frame, well clear of the 0.50 and 20.0 thresholds; if
/// a neighbouring frame is ever rejected, `quality.json` names its index and the
/// offset can move.
fn cut_one_second(ffmpeg: &Path, demo: &Path, video: &Path) {
    let output = Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-y", "-ss", "30.000"])
        .arg("-i")
        .arg(demo)
        .args([
            "-frames:v",
            "25",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "25",
        ])
        .arg(video)
        .output()
        .expect("ffmpeg runs");
    assert!(
        output.status.success(),
        "ffmpeg could not cut the clip: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// The number of regular files directly inside `dir`.
fn file_count(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("{} is readable: {error}", dir.display()))
        .filter_map(Result::ok)
        .filter(|entry| entry.path().is_file())
        .count()
}
```

- [ ] **Step 4: Run the tests and watch them pass**

Run: `cargo build -p feathertalk-worker` so the binary sits beside the CLI's in the shared target directory, then `cargo test -p feathertalk-cli --test real_worker -- --nocapture` with `FEATHERTALK_REQUIRE_E2E=1`, `FEATHERTALK_WORKER_FFMPEG=D:\environment\ffmpeg\bin\ffmpeg.exe`, `FEATHERTALK_WORKER_FFPROBE=D:\environment\ffmpeg\bin\ffprobe.exe`, `FEATHERTALK_WORKER_SCRFD_DIR=E:\workspace\github\FeatherTalk\rust\crates\feathertalk-scrfd\artifacts\scrfd_2_5g` and `FEATHERTALK_WORKER_PFLD_DIR=E:\workspace\github\FeatherTalk\rust\crates\feathertalk-pfld\artifacts\pfld_ghost_one` in the environment. The two model paths must be absolute: cargo runs a test binary with the package directory as its working directory, so a workspace-relative path would fail `is_dir` and the test would quietly skip. Confirm the output carries `test a_real_second_is_extracted_end_to_end ... ok` and no `skipping` line -- `--nocapture` is what makes a silent skip visible.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-cli/tests/real_worker.rs
git commit -m "test(cli): extract a one-second clip end to end"
```

---

### Task 15: Close the slice with the workspace gates

**Files:**
- No file changes, unless a gate demands one.

**Interfaces:**
- Consumes: every task in this slice.
- Produces: the evidence that the slice is finished. Nothing depends on it.

**Why:** each of the fourteen tasks ran only its own crate's tests, which is what keeps the loop short -- and that is exactly why the workspace has to be run once at the end. `feathertalk-frame-pipeline` gained a module and changed the shape of a command its neighbours call, and those neighbours are compiled by test binaries no per-task command touched. Clippy over `--all-targets` also lints test code with `-D warnings`, which `cargo test` compiles but never judges.

This task carries no test-first cycle, because it adds no behaviour: the gates are its steps. Any edit a gate forces is mechanical -- a reformat, a lint fix -- and belongs in a single commit at the end rather than folded back into a finished task.

The list mirrors the final gate in the Global Constraints above; `cargo check` is absent only because Gate 2 compiles strictly more than it does.

- [ ] **Gate 1: `cargo fmt --all -- --check`**

Expected: no output and exit 0. Every code fence in this plan was shaped by `rustfmt --edition 2024`, so a failure means a hand-edit slipped past it.

- [ ] **Gate 2: `cargo clippy --workspace --all-targets -- -D warnings`**

Expected: no warnings. This slice adds exactly one suppression, the `#[allow(clippy::too_many_arguments)]` on `execute_extract_frames` in Task 11; confirm clippy still needs it, because an `allow` that suppresses nothing is dead weight.

- [ ] **Gate 3: `cargo test --workspace --all-targets`**

Expected: 0 failures, in roughly 40 minutes. The last full run before this slice was 181 test binaries, 865 passed, 0 failed, 13 ignored; the new counts must be higher, never lower. `feathertalk-frame-adapters` alone accounts for about 11 of those minutes, so a long silence is the fixture-backed pipeline test, not a hang.

- [ ] **Gate 4: the gated end-to-end**

Run Task 14's Step 4 command again, with `FEATHERTALK_REQUIRE_E2E=1` and the four absolute paths in the environment. Gate 3 cannot stand in for it: without those variables the test skips and still reports success, so this is the only gate that proves the real toolchain and the imported models agree with the code.

- [ ] **Gate 5: `git diff --check`, then `git status -sb`**

Expected: no whitespace errors, a clean tree, and `demo/kanghui_training_video_featherhubert_188_latest/` still untracked. Confirm no `.jpg`, `.mp4`, `.wav`, or `.npy` was staged along the way -- `.gitignore` re-includes `demo/*.jpg` and `demo/*.mp4`, so a stray artefact written under `demo/` would not have been ignored.

If a gate forces an edit, stage the touched paths and commit them as `chore: satisfy the workspace lints for the extract-frames slice`. If every gate passes untouched, the slice ends at Task 14's commit and this task adds none.
