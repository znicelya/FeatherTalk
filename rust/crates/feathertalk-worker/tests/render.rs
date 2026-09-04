use std::path::{Path, PathBuf};

use feathertalk_domain::{ErrorCode, Progress, RenderParams, TaskError, TaskStage};
use feathertalk_export::ModelConfiguration;
use feathertalk_media::CancellationToken;
use feathertalk_models::unet::OriginalUnetConfig;
use feathertalk_worker::{
    CommandOutcome, RenderDevice, RenderJob, checkpoint_descriptor, render_job, run_render,
};
use serde_json::Value;

#[path = "support/mod.rs"]
mod support;

use support::{
    MemorySinkFactory, Recorder, StubFrameReader, render_audio, render_model, render_tree,
};

/// A job over a fixture project, carrying the checkpoint identity a real
/// training run would have written.
///
/// The test binary stands in for ffmpeg: the sink is a stub, so the path only has
/// to be an absolute file that exists.
fn job(
    project_dir: &Path,
    output: PathBuf,
    frame_count: u64,
    max_output_frames: Option<u64>,
) -> RenderJob {
    let params = RenderParams {
        project_dir: project_dir.to_path_buf(),
        checkpoint: project_dir
            .join("models")
            .join("unet")
            .join("checkpoint-00000002"),
        audio: render_audio(project_dir),
        output,
        max_output_frames,
    };
    let configuration = ModelConfiguration::original_unet(&OriginalUnetConfig::production());
    let descriptor = checkpoint_descriptor(&configuration).expect("the configuration serialises");
    let ffmpeg = std::env::current_exe().expect("the test binary knows its own path");
    render_job(&params, frame_count, &ffmpeg, descriptor, 1, 2).expect("the job is valid")
}

fn completed(outcome: CommandOutcome) -> Value {
    match outcome {
        CommandOutcome::Completed(Some(payload)) => payload,
        other => panic!("expected a completed outcome, got {other:?}"),
    }
}

fn failed(outcome: CommandOutcome) -> TaskError {
    match outcome {
        CommandOutcome::Failed(error) => error,
        other => panic!("expected a failed outcome, got {other:?}"),
    }
}

/// One reported frame, with the progress that goes with it.
fn frame_event(frame: u64, total: u64) -> (TaskStage, Option<Progress>) {
    (
        TaskStage::Rendering { frame, total },
        Some(Progress {
            completed: frame,
            total: Some(total),
        }),
    )
}

/// The staging file the factory was asked to write, once the render has ended.
fn staging(sinks: &MemorySinkFactory) -> Option<PathBuf> {
    sinks.staging.lock().expect("the factory is intact").clone()
}

fn written_frames(sinks: &MemorySinkFactory) -> usize {
    sinks.frames.lock().expect("the factory is intact").len()
}

#[test]
fn a_render_reports_one_progress_event_per_frame() {
    let (root, project) = render_tree(2, 2);
    let job = job(&project, root.path().join("render.mp4"), 2, None);
    let device = RenderDevice::default();
    let model = render_model(&device);
    let token = CancellationToken::new();
    let reporter = Recorder::new();
    let reader = StubFrameReader::default();
    let sinks = MemorySinkFactory::default();

    let outcome = run_render(&job, &model, &device, &token, &reporter, &reader, &sinks);

    let payload = completed(outcome);
    assert_eq!(payload["frame_count"], 2);
    // One event per written frame, in order, with the total the job carries.
    let events = reporter.events();
    assert_eq!(
        events,
        vec![frame_event(1, 2), frame_event(2, 2)],
        "{events:?}"
    );
    assert!(job.request.output_path().is_file());
    assert_eq!(written_frames(&sinks), 2);
}

#[test]
fn a_cancelled_render_leaves_neither_a_video_nor_a_staging_file() {
    let (root, project) = render_tree(2, 2);
    let job = job(&project, root.path().join("render.mp4"), 2, None);
    let device = RenderDevice::default();
    let model = render_model(&device);
    let token = CancellationToken::new();
    // Cancelled once the first frame has been reported, so the second write is
    // the one that sees it.
    let reporter = Recorder::cancelling_after(1, token.clone());
    let reader = StubFrameReader::default();
    let sinks = MemorySinkFactory::default();

    let outcome = run_render(&job, &model, &device, &token, &reporter, &reader, &sinks);

    assert!(matches!(outcome, CommandOutcome::Cancelled), "{outcome:?}");
    assert_eq!(written_frames(&sinks), 1);
    assert!(!job.request.output_path().exists());
    // Inference's own guard removes what it staged; this is the assertion that
    // proves a cancelled render leaves no half-written file behind.
    let staged = staging(&sinks).expect("the encoder was started");
    assert!(!staged.exists(), "{}", staged.display());
}

#[test]
fn a_render_that_cannot_start_fails_before_the_first_frame() {
    let (root, project) = render_tree(2, 2);
    let output = root.path().join("render.mp4");
    let job = job(&project, output.clone(), 2, None);
    // Written after the job was assembled, so the executor's own destination
    // check is the one that rejects it.
    std::fs::write(&output, b"sentinel").expect("the destination is written");
    let device = RenderDevice::default();
    let model = render_model(&device);
    let token = CancellationToken::new();
    let reporter = Recorder::new();
    let reader = StubFrameReader::default();
    let sinks = MemorySinkFactory::default();

    let outcome = run_render(&job, &model, &device, &token, &reporter, &reader, &sinks);

    let error = failed(outcome);
    assert_eq!(error.code, ErrorCode::MediaInvalid);
    // Nothing was written, so the failure belongs to preparation.
    assert_eq!(error.stage, TaskStage::Preparing);
    assert!(reporter.events().is_empty(), "{:?}", reporter.events());
    // The file that was already there is untouched.
    assert_eq!(
        std::fs::read(&output).expect("the destination is readable"),
        b"sentinel"
    );
}

#[test]
fn a_failure_mid_render_reports_the_frame_it_reached() {
    let (root, project) = render_tree(2, 2);
    let job = job(&project, root.path().join("render.mp4"), 2, None);
    let device = RenderDevice::default();
    let model = render_model(&device);
    let token = CancellationToken::new();
    let reporter = Recorder::new();
    // The first output frame reuses the frame the executor prefetched, so the
    // second source read is the first one that can fail.
    let reader = StubFrameReader::failing_at(1);
    let sinks = MemorySinkFactory::default();

    let outcome = run_render(&job, &model, &device, &token, &reporter, &reader, &sinks);

    let error = failed(outcome);
    assert_eq!(error.code, ErrorCode::WorkerCrashed);
    // The stage of the last frame that got through, rather than `Preparing`.
    assert_eq!(error.stage, TaskStage::Rendering { frame: 1, total: 2 });
    assert!(!job.request.output_path().exists());
    let staged = staging(&sinks).expect("the encoder was started");
    assert!(!staged.exists(), "{}", staged.display());
}

#[test]
fn a_render_cancelled_before_it_starts_never_opens_the_encoder() {
    let (root, project) = render_tree(2, 2);
    let job = job(&project, root.path().join("render.mp4"), 2, None);
    let device = RenderDevice::default();
    let model = render_model(&device);
    let token = CancellationToken::new();
    token.cancel();
    let reporter = Recorder::new();
    let reader = StubFrameReader::default();
    let sinks = MemorySinkFactory::default();

    let outcome = run_render(&job, &model, &device, &token, &reporter, &reader, &sinks);

    assert!(matches!(outcome, CommandOutcome::Cancelled), "{outcome:?}");
    // The sink was never started, so there is nothing to clean up and nothing to
    // report.
    assert!(staging(&sinks).is_none());
    assert_eq!(written_frames(&sinks), 0);
    assert!(reporter.events().is_empty(), "{:?}", reporter.events());
    assert!(!job.request.output_path().exists());
}

#[test]
fn a_capped_render_counts_up_to_the_cap() {
    let (root, project) = render_tree(2, 2);
    // Two frames in the project, one of them asked for.
    let job = job(&project, root.path().join("render.mp4"), 2, Some(1));
    let device = RenderDevice::default();
    let model = render_model(&device);
    let token = CancellationToken::new();
    let reporter = Recorder::new();
    let reader = StubFrameReader::default();
    let sinks = MemorySinkFactory::default();

    let outcome = run_render(&job, &model, &device, &token, &reporter, &reader, &sinks);

    let payload = completed(outcome);
    assert_eq!(payload["frame_count"], 1);
    assert_eq!(payload["source_frame_count"], 2);
    assert_eq!(payload["max_output_frames"], 1);
    // The denominator is the cap, not the project: a progress bar that ended at
    // one half would be a lie.
    let events = reporter.events();
    assert_eq!(events, vec![frame_event(1, 1)], "{events:?}");
    assert_eq!(written_frames(&sinks), 1);
}
