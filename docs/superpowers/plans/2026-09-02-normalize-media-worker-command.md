# normalize_media Worker Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `normalize_media` to the worker and the CLI, together with the progress-reporting boundary a long-running command needs.

**Architecture:** `feathertalk-media` grows a phase observer around its existing normalization pipeline. The worker grows a `TaskReporter` trait; the runtime implements it by sending events through the control channel it already owns, so progress events keep the same lifecycle and validation guarantees as terminal events. The command arm maps three of the five phases onto protocol stages with a 3-step progress count and returns the artifact paths, sizes, and hashes as the completed result.

**Tech Stack:** Rust 2024, edition 1.92 toolchain, `serde_json`, `clap 4.5` (derive), `tempfile`, std threads and channels. No async runtime.

**Design:** `docs/superpowers/specs/2026-09-02-normalize-media-worker-command-design.md`

## Global Constraints

- Run every cargo command from `E:/workspace/github/FeatherTalk/rust`; run git from `E:/workspace/github/FeatherTalk`.
- Normalization targets are fixed: 25 fps video, 16000 Hz audio, 1 channel. No new environment variable.
- User-facing strings are Chinese; code comments, doc comments, and diagnostics are English.
- Every source file stays free of a BOM and uses LF endings.
- `serde_json` is built without `preserve_order`; never re-serialise a frame the worker or CLI received.
- Progress events carry no metrics: `Metrics::empty()` stays untouched.
- Do not touch `demo/kanghui_training_video_featherhubert_188_latest/`; it must stay untracked.
- Commit after each task. Stage explicit paths, never `git add .`.
- The final gate for the whole slice: `cargo check`, `cargo test --workspace --all-targets`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --all -- --check`, `git diff --check`.

## File Structure

- `rust/crates/feathertalk-media/src/normalize.rs` — add `NormalizePhase` and `normalize_media_observed`; the two existing entry points delegate to it.
- `rust/crates/feathertalk-media/src/lib.rs` — export the new phase enum and function.
- `rust/crates/feathertalk-media/tests/normalization_execution.rs` — phase-order coverage next to the existing scripted-runner tests.
- `rust/crates/feathertalk-worker/src/reporter.rs` (new) — `TaskReporter` and `NoReporter`. One responsibility: the progress channel's shape.
- `rust/crates/feathertalk-worker/src/runtime.rs` — `ChannelReporter`, the new `JobExecutor` signature, and the rejection text.
- `rust/crates/feathertalk-worker/src/commands.rs` — the `normalize_media` arm and the phase-to-stage mapping.
- `rust/crates/feathertalk-worker/src/normalize_result.rs` (new) — `normalize_to_json`, mirroring `probe_result.rs`.
- `rust/crates/feathertalk-worker/src/handshake.rs` — advertise the command when a toolchain resolved.
- `rust/crates/feathertalk-worker/src/lib.rs` — module declarations and re-exports.
- `rust/crates/feathertalk-worker/tests/{commands,handshake,runtime}.rs` — command, capability, and wire coverage.
- `rust/crates/feathertalk-cli/src/{cli,run,render}.rs` — the subcommand, its request, and the unsupported-command advice.
- `rust/crates/feathertalk-client/tests/support/fake_worker.rs` — the advertised command list and a normalization scenario.
- `rust/crates/feathertalk-client/tests/handshake.rs` — the advertised command list assertion.
- `rust/crates/feathertalk-cli/tests/{cli,real_worker}.rs` — CLI behaviour and end-to-end coverage.

---

### Task 1: Phase observer in feathertalk-media

**Files:**
- Modify: `rust/crates/feathertalk-media/src/normalize.rs`
- Modify: `rust/crates/feathertalk-media/src/lib.rs`
- Test: `rust/crates/feathertalk-media/tests/normalization_execution.rs`

**Interfaces:**
- Consumes: existing `normalize_media_with_runner(&ValidatedInput, &NormalizationSpec, &MediaToolchain, &R) -> Result<NormalizedMedia, MediaError>`.
- Produces: `NormalizePhase::{Probing, NormalizingVideo, NormalizingAudio, Verifying, Committing}` and `normalize_media_observed<R: ProcessRunner + ?Sized>(input: &ValidatedInput, spec: &NormalizationSpec, toolchain: &MediaToolchain, runner: &R, observer: &dyn Fn(NormalizePhase)) -> Result<NormalizedMedia, MediaError>`. Task 3 depends on both.

- [ ] **Step 1: Write the failing tests**

Append to `rust/crates/feathertalk-media/tests/normalization_execution.rs`. The file already has `setup()`, `tools()`, `FakeRunner::new`, `source_probe()`, `video_probe()`, `audio_probe()`, and `validate_normalization` in scope; add `NormalizePhase` and `normalize_media_observed` to its `feathertalk_media` import list, and `std::sync::Mutex` is already imported.

```rust
fn record_phases() -> (Mutex<Vec<NormalizePhase>>,) {
    (Mutex::new(Vec::new()),)
}

#[test]
fn a_successful_run_reports_every_phase_in_order() {
    let (root, input, spec) = setup();
    let runner = FakeRunner::new(vec![
        Ok(ProcessOutput::new(Some(0), source_probe(), Vec::new())),
        Ok(ProcessOutput::new(Some(0), Vec::new(), Vec::new())),
        Ok(ProcessOutput::new(Some(0), Vec::new(), Vec::new())),
        Ok(ProcessOutput::new(Some(0), video_probe(), Vec::new())),
        Ok(ProcessOutput::new(Some(0), audio_probe(), Vec::new())),
    ]);
    let (phases,) = record_phases();

    normalize_media_observed(&input, &spec, &tools(root.path()), &runner, &|phase| {
        phases.lock().unwrap().push(phase)
    })
    .unwrap();

    assert_eq!(
        *phases.lock().unwrap(),
        vec![
            NormalizePhase::Probing,
            NormalizePhase::NormalizingVideo,
            NormalizePhase::NormalizingAudio,
            NormalizePhase::Verifying,
            NormalizePhase::Committing,
        ]
    );
}

#[test]
fn a_failing_video_pass_reports_no_phase_after_the_one_that_failed() {
    let (root, input, spec) = setup();
    let runner = FakeRunner::new(vec![
        Ok(ProcessOutput::new(Some(0), source_probe(), Vec::new())),
        Ok(ProcessOutput::new(
            Some(1),
            Vec::new(),
            b"encode failed".to_vec(),
        )),
    ]);
    let (phases,) = record_phases();

    let error = normalize_media_observed(&input, &spec, &tools(root.path()), &runner, &|phase| {
        phases.lock().unwrap().push(phase)
    })
    .expect_err("the video pass fails");

    assert!(matches!(
        error,
        MediaError::ToolFailed {
            operation: "normalize_video",
            ..
        }
    ));
    assert_eq!(
        *phases.lock().unwrap(),
        vec![NormalizePhase::Probing, NormalizePhase::NormalizingVideo]
    );
}

#[test]
fn an_unusable_output_directory_reports_no_phase_at_all() {
    // A file where the output directory should be: layout validation fails
    // before any work is announced.
    let (root, input, mut spec) = setup();
    let blocked = root.path().join("blocked");
    fs::write(&blocked, b"not-a-directory").unwrap();
    spec.output_dir = blocked;
    let runner = FakeRunner::new(vec![]);
    let (phases,) = record_phases();

    let error = normalize_media_observed(&input, &spec, &tools(root.path()), &runner, &|phase| {
        phases.lock().unwrap().push(phase)
    })
    .expect_err("an output directory that is a file is refused");

    assert!(matches!(error, MediaError::OutputDirectoryInvalid { .. }));
    assert!(phases.lock().unwrap().is_empty());
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p feathertalk-media --test normalization_execution`
Expected: FAIL to compile — `NormalizePhase` and `normalize_media_observed` are not found in `feathertalk_media`.

- [ ] **Step 3: Add the phase enum and the observing entry point**

In `rust/crates/feathertalk-media/src/normalize.rs`, add above `normalize_media_with_runner`:

```rust
/// The phases of one normalization, in the order they run.
///
/// Reported to an observer immediately *before* each phase starts, so a caller
/// that displays the phase describes what is running. The observer is an output
/// channel only: it cannot fail and cannot stop the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizePhase {
    Probing,
    NormalizingVideo,
    NormalizingAudio,
    Verifying,
    Committing,
}
```

Rename the body of `normalize_media_with_runner` into the observing function and add the `observer` calls:

```rust
pub fn normalize_media_observed<R: ProcessRunner + ?Sized>(
    input: &crate::ValidatedInput,
    spec: &NormalizationSpec,
    toolchain: &MediaToolchain,
    runner: &R,
    observer: &dyn Fn(NormalizePhase),
) -> Result<NormalizedMedia, MediaError> {
    // Layout validation runs before the first phase is announced: a spec that
    // names an unusable output directory never claims work it did not start.
    let layout = validate_normalization(input, spec)?;

    observer(NormalizePhase::Probing);
    let source = probe_media_with_runner(input, toolchain, runner)?;
    require_source_streams(&source)?;

    let mut video_temp = TempOutput::create(layout.output_dir(), "video", "mp4")?;
    let mut audio_temp = TempOutput::create(layout.output_dir(), "audio", "wav")?;

    observer(NormalizePhase::NormalizingVideo);
    run_tool(
        runner,
        &video_normalization_command(toolchain, input.source(), video_temp.path()),
        toolchain,
    )?;

    observer(NormalizePhase::NormalizingAudio);
    run_tool(
        runner,
        &audio_normalization_command(toolchain, input.source(), audio_temp.path()),
        toolchain,
    )?;

    observer(NormalizePhase::Verifying);
    let video = verify_video_output(video_temp.path(), toolchain, runner)?;
    let audio = verify_audio_output(audio_temp.path(), toolchain, runner)?;
    let delta = (video.duration_seconds() - audio.duration_seconds()).abs();
    if delta > 0.020 {
        return Err(MediaError::NormalizationVerificationFailed {
            field: "duration_delta",
            expected: "<= 0.020 seconds".to_owned(),
            actual: format!("{delta:.6} seconds"),
        });
    }

    observer(NormalizePhase::Committing);
    let video_artifact = hash_file(video_temp.path())?;
    let audio_artifact = hash_file(audio_temp.path())?;
    commit_output_pair(
        video_temp.path(),
        audio_temp.path(),
        layout.video_path(),
        layout.audio_path(),
        &SystemFileOps,
    )?;
    video_temp.disarm();
    audio_temp.disarm();
    Ok(NormalizedMedia::new(
        layout,
        source,
        Some(video),
        Some(audio),
        video_artifact,
        audio_artifact,
    ))
}

pub fn normalize_media_with_runner<R: ProcessRunner + ?Sized>(
    input: &crate::ValidatedInput,
    spec: &NormalizationSpec,
    toolchain: &MediaToolchain,
    runner: &R,
) -> Result<NormalizedMedia, MediaError> {
    normalize_media_observed(input, spec, toolchain, runner, &|_phase| {})
}
```

In `rust/crates/feathertalk-media/src/lib.rs`, replace the normalize export line with:

```rust
pub use normalize::{NormalizePhase, normalize_media, normalize_media_observed, normalize_media_with_runner};
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p feathertalk-media`
Expected: PASS, including every pre-existing normalization test — `normalize_media_with_runner` must still behave identically.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-media/src/normalize.rs rust/crates/feathertalk-media/src/lib.rs rust/crates/feathertalk-media/tests/normalization_execution.rs
git commit -m "feat(media): report normalization phases to an observer"
```

---

### Task 2: Progress reporting through the worker runtime

**Files:**
- Create: `rust/crates/feathertalk-worker/src/reporter.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs`
- Modify: `rust/crates/feathertalk-worker/src/commands.rs`
- Modify: `rust/crates/feathertalk-worker/src/runtime.rs`
- Test: `rust/crates/feathertalk-worker/tests/runtime.rs`, `rust/crates/feathertalk-worker/tests/commands.rs`

**Interfaces:**
- Consumes: `feathertalk_domain::{Event, Progress, TaskStage}`, the runtime's private `ControlMessage::Emit(Event)` and `now_rfc3339()`.
- Produces: `pub trait TaskReporter { fn report(&self, stage: TaskStage, progress: Option<Progress>); }`, `pub struct NoReporter`, and the four-argument executor shape `execute(&Request, Option<&MediaToolchain>, &CancellationToken, &dyn TaskReporter)` / `execute_with_runner(&Request, Option<&MediaToolchain>, &CancellationToken, &dyn TaskReporter, &R)`. Task 3 reports through it; Task 4 observes the events it produces.

- [ ] **Step 1: Write the failing tests**

Add to `rust/crates/feathertalk-worker/tests/runtime.rs`. Add `Progress` to the `feathertalk_domain` import list and `NoReporter` to the `feathertalk_worker` one.

```rust
/// Reports two progress events, then completes.
fn reporting_executor() -> JobExecutor {
    Box::new(|_request, _media, _token, reporter| {
        reporter.report(
            TaskStage::ExtractingFrames,
            Some(Progress {
                completed: 1,
                total: Some(2),
            }),
        );
        reporter.report(
            TaskStage::ExtractingAudio,
            Some(Progress {
                completed: 2,
                total: Some(2),
            }),
        );
        CommandOutcome::Completed(None)
    })
}

#[test]
fn reported_progress_reaches_the_wire_in_order() {
    let harness = Harness::start(bare_config(), reporting_executor());
    let id = task("00000021");
    harness.send(&start(&id, validate_project("C:/tmp/project")));
    let frames = harness.wait_for("the task to complete", |frames| {
        stages(frames).iter().any(|(_, stage)| *stage == "completed")
    });

    assert_eq!(
        stages(&frames)
            .into_iter()
            .map(|(_, stage)| stage)
            .collect::<Vec<_>>(),
        vec![
            "preparing",
            "extracting_frames",
            "extracting_audio",
            "completed"
        ]
    );
    let progressed = events(&frames)
        .into_iter()
        .filter_map(|event| event.progress)
        .collect::<Vec<_>>();
    assert_eq!(
        progressed,
        vec![
            Progress {
                completed: 1,
                total: Some(2)
            },
            Progress {
                completed: 2,
                total: Some(2)
            }
        ]
    );
    // Progress events carry no metrics.
    for event in events(&frames) {
        assert_eq!(event.metrics, feathertalk_domain::Metrics::empty());
    }
    harness.finish();
}

#[test]
fn a_cancelled_task_keeps_the_progress_it_already_reported() {
    let (started, started_rx) = mpsc::channel::<()>();
    // Reports one progress event, then waits to be cancelled.
    let executor: JobExecutor = Box::new(move |_request, _media, token, reporter| {
        reporter.report(
            TaskStage::ExtractingFrames,
            Some(Progress {
                completed: 1,
                total: Some(2),
            }),
        );
        started.send(()).unwrap();
        while !token.is_cancelled() {
            thread::sleep(Duration::from_millis(5));
        }
        CommandOutcome::Cancelled
    });
    let harness = Harness::start(bare_config(), executor);
    let id = task("00000022");
    harness.send(&start(&id, validate_project("C:/tmp/project")));
    started_rx.recv().unwrap();
    harness.send(&cancel(&id));
    let frames = harness.finish();

    assert_eq!(
        stages(&frames)
            .into_iter()
            .map(|(_, stage)| stage)
            .collect::<Vec<_>>(),
        vec!["preparing", "extracting_frames", "cancelled"]
    );
}

#[test]
fn the_noop_reporter_emits_nothing() {
    // `NoReporter` is what a direct caller of `execute` passes; it must be safe
    // to call and must not panic.
    NoReporter.report(TaskStage::Preparing, None);
    NoReporter.report(
        TaskStage::ExtractingAudio,
        Some(Progress {
            completed: 1,
            total: None,
        }),
    );
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p feathertalk-worker --test runtime`
Expected: FAIL to compile — `NoReporter` is unresolved and the closures take four arguments while `JobExecutor` takes three.

- [ ] **Step 3: Add the reporter and thread it through**

Create `rust/crates/feathertalk-worker/src/reporter.rs`:

```rust
use feathertalk_domain::{Progress, TaskStage};

/// How a running command reports intermediate stages.
///
/// A command never writes to stdout itself: the runtime's control loop is the
/// only owner of the writer and of task lifecycle, so a report is a message to
/// it, not an event. Terminal stages are not reported here; they are the
/// command's return value.
pub trait TaskReporter {
    fn report(&self, stage: TaskStage, progress: Option<Progress>);
}

/// The reporter for callers that do not observe progress: direct library users
/// and tests that only assert the outcome.
pub struct NoReporter;

impl TaskReporter for NoReporter {
    fn report(&self, _stage: TaskStage, _progress: Option<Progress>) {}
}
```

In `rust/crates/feathertalk-worker/src/lib.rs` add `mod reporter;` with the other module declarations and `pub use reporter::{NoReporter, TaskReporter};` with the other re-exports.

In `rust/crates/feathertalk-worker/src/commands.rs`, add `TaskReporter` to the `crate` import list and take it in both entry points:

```rust
pub fn execute(
    request: &Request,
    media: Option<&MediaToolchain>,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
) -> CommandOutcome {
    let runner = CancellableProcessRunner::new(token.clone());
    execute_with_runner(request, media, token, reporter, &runner)
}

pub fn execute_with_runner<R: ProcessRunner + ?Sized>(
    request: &Request,
    media: Option<&MediaToolchain>,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    runner: &R,
) -> CommandOutcome {
```

Silence the unused parameter for now with `let _ = reporter;` immediately after the token check; Task 3 uses it.

In `rust/crates/feathertalk-worker/src/runtime.rs`, add `Progress` to the `feathertalk_domain` imports and `TaskReporter` to the `crate` imports, widen the executor type, and add the channel-backed reporter:

```rust
pub type JobExecutor = Box<
    dyn Fn(&Request, Option<&MediaToolchain>, &CancellationToken, &dyn TaskReporter)
        -> CommandOutcome
        + Send
        + 'static,
>;

/// The execution thread's reporter: one per job, holding that job's id and a
/// clone of the control channel.
///
/// A closed channel is ignored. Losing a progress event while the runtime is
/// already shutting down is not a reason to fail a task.
struct ChannelReporter {
    task_id: TaskId,
    control_tx: Sender<ControlMessage>,
}

impl TaskReporter for ChannelReporter {
    fn report(&self, stage: TaskStage, progress: Option<Progress>) {
        let mut event = Event::new(self.task_id.clone(), &now_rfc3339(), stage);
        event.progress = progress;
        let _ = self.control_tx.send(ControlMessage::Emit(event));
    }
}
```

In `run_jobs`, build the reporter before executing:

```rust
    while let Ok(job) = job_rx.recv() {
        let reporter = ChannelReporter {
            task_id: job.task_id.clone(),
            control_tx: control_tx.clone(),
        };
        let event = match executor(&job.request, media.as_ref(), &job.token, &reporter) {
```

- [ ] **Step 4: Update the existing executors and direct callers**

In `rust/crates/feathertalk-worker/tests/runtime.rs`, add the fourth parameter to `instant_executor`, `blocking_executor`, `gated_executor`, and `blocking_probe_executor` (`|request, media, token, reporter|`, passing `reporter` through in `blocking_probe_executor`'s `execute_with_runner` call and `let _ = reporter;` in the others).

In `rust/crates/feathertalk-worker/tests/commands.rs`, add `NoReporter` to the `feathertalk_worker` import list and `&NoReporter` as the fourth argument of every `execute_with_runner` call.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p feathertalk-worker`
Expected: PASS, all tests.

- [ ] **Step 6: Commit**

```bash
git add rust/crates/feathertalk-worker/src/reporter.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/src/commands.rs rust/crates/feathertalk-worker/src/runtime.rs rust/crates/feathertalk-worker/tests/runtime.rs rust/crates/feathertalk-worker/tests/commands.rs
git commit -m "feat(worker): let a running command report progress"
```

---

### Task 3: The normalize_media command

**Files:**
- Create: `rust/crates/feathertalk-worker/src/normalize_result.rs`
- Modify: `rust/crates/feathertalk-worker/src/commands.rs`
- Modify: `rust/crates/feathertalk-worker/src/handshake.rs`
- Modify: `rust/crates/feathertalk-worker/src/runtime.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs`
- Test: `rust/crates/feathertalk-worker/tests/commands.rs`, `rust/crates/feathertalk-worker/tests/handshake.rs`, `rust/crates/feathertalk-worker/tests/runtime.rs`

**Interfaces:**
- Consumes: `NormalizePhase`, `normalize_media_observed` (Task 1); `TaskReporter` (Task 2); `feathertalk_domain::{NormalizeMediaParams, Progress, TaskKind, TaskStage}`.
- Produces: `normalize_to_json(&NormalizedMedia) -> serde_json::Value`, `TaskKind::NormalizeMedia` in `supported_commands`, and the result payload Task 4 prints.

- [ ] **Step 1: Write the failing tests**

In `rust/crates/feathertalk-worker/tests/handshake.rs`, extend the configured-worker assertion:

```rust
    assert_eq!(
        frame.supported_commands,
        vec![
            TaskKind::ValidateProject,
            TaskKind::ProbeMedia,
            TaskKind::NormalizeMedia
        ]
    );
```

In `rust/crates/feathertalk-worker/tests/commands.rs`, add `NormalizeMediaParams` to the domain imports and `Progress`/`TaskStage` too, plus `std::sync::Mutex` (already imported) and these helpers and tests. The scripted runner must stage the output files the way the media crate's own tests do, because normalization hashes and commits real bytes.

```rust
/// A runner that scripts probe output and writes the bytes `ffmpeg` would have
/// written, so the normalization pipeline can verify and commit them.
struct NormalizeRunner {
    outputs: Mutex<VecDeque<Result<ProcessOutput, MediaError>>>,
    commands: Mutex<Vec<CommandSpec>>,
}

impl NormalizeRunner {
    fn new(outputs: Vec<Result<ProcessOutput, MediaError>>) -> Self {
        Self {
            outputs: Mutex::new(outputs.into_iter().collect()),
            commands: Mutex::new(Vec::new()),
        }
    }
}

impl ProcessRunner for NormalizeRunner {
    fn run(&self, command: &CommandSpec, _timeout: Duration) -> Result<ProcessOutput, MediaError> {
        self.commands.lock().unwrap().push(command.clone());
        let output = self.outputs.lock().unwrap().pop_front().unwrap()?;
        if matches!(command.operation(), "normalize_video" | "normalize_audio") {
            let path = PathBuf::from(command.arguments().last().unwrap());
            std::fs::write(path, b"normalized-bytes").unwrap();
        }
        Ok(output)
    }
}

/// Records everything a command reports.
#[derive(Default)]
struct RecordingReporter {
    reports: Mutex<Vec<(String, Option<Progress>)>>,
}

impl TaskReporter for RecordingReporter {
    fn report(&self, stage: TaskStage, progress: Option<Progress>) {
        self.reports
            .lock()
            .unwrap()
            .push((stage.as_slug().to_owned(), progress));
    }
}

fn normalized_video_probe() -> Vec<u8> {
    br#"{"format":{"format_name":"mp4","duration":"2.0"},"streams":[{"codec_type":"video","codec_name":"mpeg4","pix_fmt":"yuv420p","width":640,"height":480,"avg_frame_rate":"25/1","nb_read_frames":"50","duration":"2.0"}]}"#.to_vec()
}

fn normalized_audio_probe() -> Vec<u8> {
    br#"{"format":{"format_name":"wav","duration":"2.0"},"streams":[{"codec_type":"audio","codec_name":"pcm_s16le","sample_fmt":"s16","sample_rate":"16000","channels":1,"duration":"2.0"}]}"#.to_vec()
}

fn normalize_request(input: PathBuf, output_dir: PathBuf) -> Request {
    Request::NormalizeMedia(NormalizeMediaParams { input, output_dir })
}

fn normalize_outputs() -> Vec<Result<ProcessOutput, MediaError>> {
    vec![
        Ok(ProcessOutput::new(Some(0), valid_probe(), Vec::new())),
        Ok(ProcessOutput::new(Some(0), Vec::new(), Vec::new())),
        Ok(ProcessOutput::new(Some(0), Vec::new(), Vec::new())),
        Ok(ProcessOutput::new(
            Some(0),
            normalized_video_probe(),
            Vec::new(),
        )),
        Ok(ProcessOutput::new(
            Some(0),
            normalized_audio_probe(),
            Vec::new(),
        )),
    ]
}

#[test]
fn normalizing_media_reports_paths_sizes_and_hashes() {
    let (temp, source) = media_file();
    let output_dir = temp.path().join("assets");
    let runner = NormalizeRunner::new(normalize_outputs());
    let reporter = RecordingReporter::default();

    let CommandOutcome::Completed(Some(result)) = execute_with_runner(
        &normalize_request(source, output_dir.clone()),
        Some(&toolchain()),
        &CancellationToken::new(),
        &reporter,
        &runner,
    ) else {
        panic!("a scripted normalization completes with a result");
    };

    assert_eq!(result["video"]["codec_name"], "mpeg4");
    assert_eq!(result["video"]["frame_rate"]["numerator"], 25);
    assert_eq!(result["audio"]["sample_rate"], 16_000);
    assert_eq!(result["audio"]["channels"], 1);
    assert_eq!(result["video"]["bytes"], b"normalized-bytes".len());
    assert_eq!(
        result["video"]["sha256"].as_str().unwrap().len(),
        64,
        "{result}"
    );
    // The source probe is reported under `source`, in the probe payload shape.
    assert_eq!(result["source"]["video"]["codec_name"], "h264");
    // The committed paths are what a later task has to open.
    let video_path = PathBuf::from(result["video"]["path"].as_str().unwrap());
    assert!(video_path.is_file(), "{}", video_path.display());
    assert_eq!(video_path.file_name().unwrap(), "video_25fps.mp4");
    let audio_path = PathBuf::from(result["audio"]["path"].as_str().unwrap());
    assert_eq!(audio_path.file_name().unwrap(), "audio_16k_mono.wav");
    assert!(
        result["output_dir"]
            .as_str()
            .unwrap()
            .ends_with("assets"),
        "{result}"
    );
}

#[test]
fn normalizing_media_reports_three_progress_steps() {
    let (temp, source) = media_file();
    let output_dir = temp.path().join("assets");
    let runner = NormalizeRunner::new(normalize_outputs());
    let reporter = RecordingReporter::default();

    execute_with_runner(
        &normalize_request(source, output_dir),
        Some(&toolchain()),
        &CancellationToken::new(),
        &reporter,
        &runner,
    );

    assert_eq!(
        *reporter.reports.lock().unwrap(),
        vec![
            (
                "preparing".to_owned(),
                Some(Progress {
                    completed: 1,
                    total: Some(3)
                })
            ),
            (
                "extracting_frames".to_owned(),
                Some(Progress {
                    completed: 2,
                    total: Some(3)
                })
            ),
            (
                "extracting_audio".to_owned(),
                Some(Progress {
                    completed: 3,
                    total: Some(3)
                })
            ),
        ]
    );
}

#[test]
fn normalizing_without_a_toolchain_is_unsupported() {
    let (temp, source) = media_file();
    let runner = NormalizeRunner::new(vec![]);
    let CommandOutcome::Failed(error) = execute_with_runner(
        &normalize_request(source, temp.path().join("assets")),
        None,
        &CancellationToken::new(),
        &NoReporter,
        &runner,
    ) else {
        panic!("no toolchain means the command cannot run");
    };
    assert_eq!(error.code, ErrorCode::WorkerCrashed);
    assert!(error.detail.contains("normalize_media"), "{}", error.detail);
}

#[test]
fn a_source_without_audio_fails_before_any_output_is_written() {
    let (temp, source) = media_file();
    let output_dir = temp.path().join("assets");
    let video_only = br#"{"format":{"format_name":"mov,mp4","duration":"2.0"},"streams":[{"codec_type":"video","codec_name":"h264","pix_fmt":"yuv420p","width":640,"height":480,"avg_frame_rate":"25/1","nb_read_frames":"50","duration":"2.0"}]}"#.to_vec();
    let runner = NormalizeRunner::new(vec![Ok(ProcessOutput::new(
        Some(0),
        video_only,
        Vec::new(),
    ))]);
    let CommandOutcome::Failed(error) = execute_with_runner(
        &normalize_request(source, output_dir.clone()),
        Some(&toolchain()),
        &CancellationToken::new(),
        &NoReporter,
        &runner,
    ) else {
        panic!("a source without audio cannot be normalized");
    };
    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert!(!output_dir.join("video_25fps.mp4").exists());
}

#[test]
fn a_cancelled_normalization_reports_cancelled() {
    let (temp, source) = media_file();
    let runner = NormalizeRunner::new(vec![Err(MediaError::ToolCancelled {
        operation: "ffprobe",
    })]);
    let outcome = execute_with_runner(
        &normalize_request(source, temp.path().join("assets")),
        Some(&toolchain()),
        &CancellationToken::new(),
        &NoReporter,
        &runner,
    );
    assert!(matches!(outcome, CommandOutcome::Cancelled), "{outcome:?}");
}
```

In `rust/crates/feathertalk-worker/tests/runtime.rs`, replace the assertion in `probe_media_is_rejected_when_the_media_toolchain_is_unavailable`'s sibling coverage by adding:

```rust
#[test]
fn normalize_media_is_rejected_when_the_media_toolchain_is_unavailable() {
    let harness = Harness::start(broken_config(), instant_executor());
    harness.send(&start(
        &task("00000023"),
        Request::NormalizeMedia(feathertalk_domain::NormalizeMediaParams {
            input: PathBuf::from("C:/tmp/clip.mp4"),
            output_dir: PathBuf::from("C:/tmp/assets"),
        }),
    ));
    let frames = harness.finish();
    let reasons = frames
        .iter()
        .filter_map(|frame| match frame {
            ServerFrame::Rejected(rejected) => Some(rejected.reason.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(reasons.len(), 1, "{frames:?}");
    assert!(reasons[0].contains("normalize_media"), "{}", reasons[0]);
    assert!(reasons[0].contains("FEATHERTALK_WORKER_FFPROBE"), "{}", reasons[0]);
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p feathertalk-worker`
Expected: FAIL — `normalize_to_json` does not exist, the handshake omits `NormalizeMedia`, and the runtime rejects `normalize_media` with the generic reason.

- [ ] **Step 3: Write the result serialiser**

Create `rust/crates/feathertalk-worker/src/normalize_result.rs`:

```rust
use std::path::Path;

use feathertalk_media::{AudioMetadata, MediaArtifact, NormalizedMedia, VideoMetadata};
use serde_json::{Value, json};

use crate::probe_to_json;

/// Shapes a normalization as the JSON object a `completed` event carries.
///
/// Unlike a probe, the payload names the files: the caller asked for a
/// directory and the worker chose the file names, so a later task would
/// otherwise have to guess them. The paths are the canonical ones the media
/// crate committed to, reported as produced rather than prettified.
pub fn normalize_to_json(media: &NormalizedMedia) -> Value {
    json!({
        "output_dir": path_text(media.layout().output_dir()),
        "video": media
            .video()
            .map(|video| video_json(media.layout().video_path(), video, media.video_artifact())),
        "audio": media
            .audio()
            .map(|audio| audio_json(media.layout().audio_path(), audio, media.audio_artifact())),
        "source": probe_to_json(media.source()),
    })
}

fn video_json(path: &Path, video: &VideoMetadata, artifact: &MediaArtifact) -> Value {
    json!({
        "path": path_text(path),
        "bytes": artifact.bytes(),
        "sha256": artifact.sha256(),
        "codec_name": video.codec_name(),
        "pixel_format": video.pixel_format(),
        "width": video.width(),
        "height": video.height(),
        "frame_rate": {
            "numerator": video.frame_rate().numerator(),
            "denominator": video.frame_rate().denominator(),
        },
        "frame_count": video.frame_count(),
        "duration_seconds": video.duration_seconds(),
    })
}

fn audio_json(path: &Path, audio: &AudioMetadata, artifact: &MediaArtifact) -> Value {
    json!({
        "path": path_text(path),
        "bytes": artifact.bytes(),
        "sha256": artifact.sha256(),
        "codec_name": audio.codec_name(),
        "sample_format": audio.sample_format(),
        "sample_rate": audio.sample_rate(),
        "channels": audio.channels(),
        "sample_count": audio.sample_count(),
        "duration_seconds": audio.duration_seconds(),
    })
}

fn path_text(path: &Path) -> String {
    path.display().to_string()
}
```

In `rust/crates/feathertalk-worker/src/lib.rs` add `mod normalize_result;` and `pub use normalize_result::normalize_to_json;`.

- [ ] **Step 4: Add the command arm**

In `rust/crates/feathertalk-worker/src/commands.rs`, extend the imports with `NormalizePhase`, `NormalizationSpec`, `normalize_media_observed`, `Progress`, and `normalize_to_json`, drop the `let _ = reporter;` line, and add the arm before the `other =>` arm:

```rust
/// How many progress steps `normalize_media` reports. Verification and the
/// commit are bounded and short, so they end the count rather than extend it.
const NORMALIZE_STEPS: u64 = 3;
```

```rust
        Request::NormalizeMedia(params) => {
            let Some(toolchain) = media else {
                return CommandOutcome::Failed(unsupported(request.kind()));
            };
            let input = match validate_input(&MediaInput {
                source: params.input.clone(),
            }) {
                Ok(input) => input,
                Err(error) => return media_failure(&error),
            };
            // The targets are fixed by the asset contract, and
            // `validate_normalization` rejects anything else, so there is
            // nothing here for a caller to configure.
            let spec = NormalizationSpec {
                target_video_fps: 25,
                target_audio_sample_rate: 16_000,
                target_audio_channels: 1,
                output_dir: params.output_dir.clone(),
            };
            match normalize_media_observed(&input, &spec, toolchain, runner, &|phase| {
                report_phase(reporter, phase)
            }) {
                Ok(normalized) => CommandOutcome::Completed(Some(normalize_to_json(&normalized))),
                Err(error) => media_failure(&error),
            }
        }
```

```rust
/// Map a normalization phase onto the protocol stage that names it.
///
/// Protocol version 2 has no stage for media normalization, so the two passes
/// that dominate wall time borrow the stages that describe their output.
/// Verification and the commit report nothing: giving them a stage would mean
/// moving the label backwards to `preparing`, which reads as a bug.
fn report_phase(reporter: &dyn TaskReporter, phase: NormalizePhase) {
    let (stage, completed) = match phase {
        NormalizePhase::Probing => (TaskStage::Preparing, 1),
        NormalizePhase::NormalizingVideo => (TaskStage::ExtractingFrames, 2),
        NormalizePhase::NormalizingAudio => (TaskStage::ExtractingAudio, 3),
        NormalizePhase::Verifying | NormalizePhase::Committing => return,
    };
    reporter.report(
        stage,
        Some(Progress {
            completed,
            total: Some(NORMALIZE_STEPS),
        }),
    );
}
```

- [ ] **Step 5: Advertise the command and fix the rejection text**

In `rust/crates/feathertalk-worker/src/handshake.rs`:

```rust
pub fn supported_commands(config: &WorkerConfig) -> Vec<TaskKind> {
    let mut commands = vec![TaskKind::ValidateProject];
    // Both media commands shell out to the same two binaries, so they are
    // available together or not at all.
    if config.media().is_some() {
        commands.push(TaskKind::ProbeMedia);
        commands.push(TaskKind::NormalizeMedia);
    }
    commands
}
```

In `rust/crates/feathertalk-worker/src/runtime.rs`, rewrite `unsupported_reason` so the media commands share the actionable message and the fallback lists what is actually supported:

```rust
fn unsupported_reason(request: &Request, config: &WorkerConfig) -> String {
    let slug = request.kind().as_slug();
    match request.kind() {
        TaskKind::ProbeMedia | TaskKind::NormalizeMedia => match config.media_rejection() {
            Some(rejection) => format!(
                "命令 {slug} 需要可用的媒体工具链，当前配置被拒绝：{rejection}。修正后重启 worker。"
            ),
            None => format!(
                "命令 {slug} 需要媒体工具链，请设置 {ENV_FFPROBE} 与 {ENV_FFMPEG} 后重启 worker。"
            ),
        },
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
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test -p feathertalk-worker`
Expected: PASS. If a pre-existing test asserted the old two-command rejection sentence, update that assertion to the new text — the message is now derived from `supported_commands`.

- [ ] **Step 7: Commit**

```bash
git add rust/crates/feathertalk-worker/src rust/crates/feathertalk-worker/tests
git commit -m "feat(worker): normalize media into 25fps video and 16k mono audio"
```

---

### Task 4: normalize-media in the CLI

**Files:**
- Modify: `rust/crates/feathertalk-cli/src/cli.rs`
- Modify: `rust/crates/feathertalk-cli/src/run.rs`
- Modify: `rust/crates/feathertalk-cli/src/render.rs`
- Modify: `rust/crates/feathertalk-client/tests/support/fake_worker.rs`
- Modify: `rust/crates/feathertalk-client/tests/handshake.rs`
- Test: `rust/crates/feathertalk-cli/tests/cli.rs`

**Interfaces:**
- Consumes: `feathertalk_domain::NormalizeMediaParams`, the result payload from Task 3, the fake worker's scenario selector `FT_FAKE_WORKER_SCENARIO`.
- Produces: the `normalize-media <INPUT> <OUTPUT_DIR>` subcommand and the `normalize-progress` fake-worker scenario.

- [ ] **Step 1: Write the failing tests**

In `rust/crates/feathertalk-cli/tests/cli.rs`, use the file's existing helpers: `run(scenario, args)` spawns the CLI against the fake worker with `FT_FAKE_WORKER_SCENARIO` set, and `code`/`stdout`/`stderr` read the result.

```rust
#[test]
fn normalize_media_prints_the_result_and_narrates_progress() {
    let output = run(
        "normalize-progress",
        &["normalize-media", "clip.mp4", "assets"],
    );
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));
    let text = stdout(&output);
    assert!(text.contains("video_25fps.mp4"), "{text}");
    assert!(text.contains("audio_16k_mono.wav"), "{text}");
    let narration = stderr(&output);
    assert!(narration.contains("正在提取视频帧"), "{narration}");
    assert!(narration.contains("正在提取音频"), "{narration}");
    assert!(narration.contains("进度 2/3 (66.7%)"), "{narration}");
}

#[test]
fn normalize_media_refuses_empty_arguments() {
    let error = build_request(&Command::NormalizeMedia {
        input: PathBuf::new(),
        output_dir: PathBuf::from("assets"),
    })
    .expect_err("an empty input is refused");
    assert_eq!(error, "输入文件不能为空。");

    let error = build_request(&Command::NormalizeMedia {
        input: PathBuf::from("clip.mp4"),
        output_dir: PathBuf::new(),
    })
    .expect_err("an empty output directory is refused");
    assert_eq!(error, "输出目录不能为空。");
}
```

`build_request` is private to `run.rs`; put the second test in `run.rs`'s own `#[cfg(test)] mod tests` alongside `an_empty_path_is_rejected_in_chinese` instead of in `tests/cli.rs`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p feathertalk-cli`
Expected: FAIL — `Command::NormalizeMedia` does not exist and the scenario is unknown (the fake worker exits 97).

- [ ] **Step 3: Add the subcommand and its request**

In `rust/crates/feathertalk-cli/src/cli.rs`, add to `Command`:

```rust
    /// 归一化媒体文件：输出 25fps 视频与 16kHz 单声道音频
    NormalizeMedia {
        /// 输入的音视频文件
        input: PathBuf,
        /// 输出目录，归一化后的视频与音频写入其中
        output_dir: PathBuf,
    },
```

In `rust/crates/feathertalk-cli/src/run.rs`, add `NormalizeMediaParams` to the domain imports and the arm to `build_request`:

```rust
        Command::NormalizeMedia { input, output_dir } => {
            reject_empty(input, "输入文件")?;
            reject_empty(output_dir, "输出目录")?;
            Ok(Some(Request::NormalizeMedia(NormalizeMediaParams {
                input: input.clone(),
                output_dir: output_dir.clone(),
            })))
        }
```

In `rust/crates/feathertalk-cli/src/render.rs`, add the ffmpeg variable next to the existing one and widen the hint:

```rust
/// The worker's own variables for locating its media tools. Written as
/// literals because the CLI must not link the worker crate.
const ENV_WORKER_FFPROBE: &str = "FEATHERTALK_WORKER_FFPROBE";
const ENV_WORKER_FFMPEG: &str = "FEATHERTALK_WORKER_FFMPEG";
```

```rust
            if matches!(requested.as_str(), "probe_media" | "normalize_media") {
                text.push_str(&format!(
                    "\n{requested} 需要可用的 ffprobe 与 ffmpeg。请安装 ffmpeg，或用环境变量 \
                     {ENV_WORKER_FFPROBE} 与 {ENV_WORKER_FFMPEG} 指定它们的完整路径。"
                ));
            }
```

- [ ] **Step 4: Teach the fake worker the new command**

In `rust/crates/feathertalk-client/tests/support/fake_worker.rs`, add `TaskKind::NormalizeMedia` to `default_commands()`, add the scenario arm next to `ready-complete`:

```rust
        // Three progress steps and a normalization result, like the real
        // worker's `normalize_media`.
        "normalize-progress" => {
            write_frame(&ready(default_commands()));
            serve_normalize_task(&mut reader);
        }
```

and the scripted task:

```rust
fn serve_normalize_task(reader: &mut Reader) {
    let Some(task_id) = wait_for_start(reader) else {
        return;
    };
    for (stage, completed) in [
        (TaskStage::Preparing, 1),
        (TaskStage::ExtractingFrames, 2),
        (TaskStage::ExtractingAudio, 3),
    ] {
        let mut event = stage_event(&task_id, stage);
        event.progress = Some(Progress {
            completed,
            total: Some(3),
        });
        write_frame(&ServerFrame::Event(event));
    }
    let mut event = stage_event(&task_id, TaskStage::Completed);
    event.result = Some(serde_json::json!({
        "output_dir": "C:/tmp/assets",
        "video": {
            "path": "C:/tmp/assets/video_25fps.mp4",
            "bytes": 16,
            "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "codec_name": "mpeg4",
            "pixel_format": "yuv420p",
            "width": 640,
            "height": 480,
            "frame_rate": { "numerator": 25, "denominator": 1 },
            "frame_count": 50,
            "duration_seconds": 2.0
        },
        "audio": {
            "path": "C:/tmp/assets/audio_16k_mono.wav",
            "bytes": 16,
            "sha256": "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
            "codec_name": "pcm_s16le",
            "sample_format": "s16",
            "sample_rate": 16000,
            "channels": 1,
            "sample_count": 32000,
            "duration_seconds": 2.0
        }
    }));
    write_frame(&ServerFrame::Event(event));
}
```

In `rust/crates/feathertalk-client/tests/handshake.rs`, extend the advertised-command assertion:

```rust
        vec![
            TaskKind::ValidateProject,
            TaskKind::ProbeMedia,
            TaskKind::NormalizeMedia
        ]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p feathertalk-cli -p feathertalk-client`
Expected: PASS. `--help` output changes, so update any test that pins the subcommand list.

- [ ] **Step 6: Commit**

```bash
git add rust/crates/feathertalk-cli/src rust/crates/feathertalk-cli/tests rust/crates/feathertalk-client/tests
git commit -m "feat(cli): add the normalize-media subcommand"
```

---

### Task 5: End-to-end coverage against the real worker

**Files:**
- Modify: `rust/crates/feathertalk-cli/tests/real_worker.rs`

**Interfaces:**
- Consumes: the existing `worker_or_skip`, `run`, `code`, `stdout`, `stderr` helpers and the `FEATHERTALK_REQUIRE_E2E` gate.
- Produces: nothing other crates use.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn a_missing_toolchain_makes_normalize_media_unsupported() {
    let Some(worker) = worker_or_skip("a_missing_toolchain_makes_normalize_media_unsupported")
    else {
        return;
    };
    let output = run(
        &worker,
        &["normalize-media", "clip.mp4", "assets"],
        &[("FEATHERTALK_WORKER_FFPROBE", "relative-ffprobe")],
    );
    assert_eq!(code(&output), 3, "stdout was: {}", stdout(&output));
    let text = stderr(&output);
    assert!(text.contains("normalize_media"), "{text}");
    assert!(text.contains("FEATHERTALK_WORKER_FFMPEG"), "{text}");
}

#[test]
fn a_missing_input_is_a_normalize_task_failure() {
    let Some(worker) = worker_or_skip("a_missing_input_is_a_normalize_task_failure") else {
        return;
    };
    // Absolute tool paths are all the worker's configuration requires, so the
    // command is accepted and fails where it should: on the input. This is the
    // one end-to-end normalization path that needs no real ffmpeg.
    let temp = TempDir::new().expect("a temporary directory is available");
    let missing = temp.path().join("absent.mp4");
    let assets = temp.path().join("assets");
    let fake_tool = temp.path().join("not-a-real-ffmpeg");
    let output = run(
        &worker,
        &[
            "normalize-media",
            &missing.to_string_lossy(),
            &assets.to_string_lossy(),
        ],
        &[
            (
                "FEATHERTALK_WORKER_FFPROBE",
                &fake_tool.to_string_lossy().into_owned(),
            ),
            (
                "FEATHERTALK_WORKER_FFMPEG",
                &fake_tool.to_string_lossy().into_owned(),
            ),
        ],
    );
    assert_eq!(code(&output), 1, "stdout was: {}", stdout(&output));
    let text = stderr(&output);
    assert!(text.contains("MEDIA_INVALID"), "{text}");
    // The task failed before the output directory was needed.
    assert!(!assets.join("video_25fps.mp4").exists());
}

/// A full normalization, only when real tools are pointed at by the
/// environment. Neither this repository nor CI ships ffmpeg, so an absent tool
/// is a skip, not a failure: the alternative is a test that fails for reasons
/// that have nothing to do with the code under test.
#[test]
fn a_real_clip_is_normalized_end_to_end() {
    let Some(worker) = worker_or_skip("a_real_clip_is_normalized_end_to_end") else {
        return;
    };
    let (Some(ffmpeg), Some(ffprobe)) = (real_tool("FFMPEG"), real_tool("FFPROBE")) else {
        println!(
            "skipping a_real_clip_is_normalized_end_to_end: set FEATHERTALK_WORKER_FFMPEG and \
             FEATHERTALK_WORKER_FFPROBE to real binaries to run it"
        );
        return;
    };
    let temp = TempDir::new().expect("a temporary directory is available");
    let clip = temp.path().join("clip.mp4");
    // One second of colour bars and a tone: the smallest input with both
    // streams the pipeline requires.
    let generated = Command::new(&ffmpeg)
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=30:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-shortest",
        ])
        .arg(&clip)
        .output()
        .expect("ffmpeg runs");
    assert!(
        generated.status.success(),
        "ffmpeg could not generate the clip: {}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let assets = temp.path().join("assets");
    let output = run(
        &worker,
        &[
            "normalize-media",
            &clip.to_string_lossy(),
            &assets.to_string_lossy(),
        ],
        &[
            ("FEATHERTALK_WORKER_FFMPEG", &ffmpeg.to_string_lossy().into_owned()),
            ("FEATHERTALK_WORKER_FFPROBE", &ffprobe.to_string_lossy().into_owned()),
        ],
    );
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));
    assert!(assets.join("video_25fps.mp4").is_file());
    assert!(assets.join("audio_16k_mono.wav").is_file());
    let text = stdout(&output);
    assert!(text.contains("pcm_s16le"), "{text}");
    let narration = stderr(&output);
    assert!(narration.contains("进度 3/3"), "{narration}");
}

/// A media tool from the environment, only if it is an existing file.
fn real_tool(suffix: &str) -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var(format!("FEATHERTALK_WORKER_{suffix}")).ok()?);
    path.is_file().then_some(path)
}
```

- [ ] **Step 2: Run the tests**

Run: `cargo build -p feathertalk-worker` then `cargo test -p feathertalk-cli --test real_worker -- --nocapture`
Expected: the first two tests PASS; the third prints its skip reason unless ffmpeg and ffprobe are configured.

- [ ] **Step 3: Commit**

```bash
git add rust/crates/feathertalk-cli/tests/real_worker.rs
git commit -m "test(cli): cover normalize-media against the real worker"
```

---

## Final Gate

- [ ] `cargo check` exits 0.
- [ ] `cargo test --workspace --all-targets` — every test passes; the count is at least the 756 the previous slice left behind.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` exits 0.
- [ ] `cargo fmt --all -- --check` exits 0.
- [ ] `git diff --check` exits 0 and `demo/kanghui_training_video_featherhubert_188_latest/` is still untracked.
- [ ] Update `docs/superpowers/specs/2026-08-17-rust-desktop-migration-design.md` §16 so the CLI-parity note records `normalize-media`, and commit that separately.
- [ ] Run the `finishing-a-development-branch` skill.
