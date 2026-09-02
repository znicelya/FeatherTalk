use feathertalk_domain::{ErrorCode, Progress, Request, TaskError, TaskKind, TaskStage};
use feathertalk_media::{
    CancellableProcessRunner, CancellationToken, MediaError, MediaInput, NormalizationSpec,
    NormalizePhase, ProcessRunner, normalize_media_observed, probe_media_with_runner,
    validate_input,
};
use feathertalk_project::validate_project_dir;

use crate::{
    TaskReporter, WorkerConfig, is_media_cancellation, media_task_error, normalize_to_json,
    probe_to_json, project_task_error,
};

/// How many progress steps `normalize_media` reports. Verification and the
/// commit are bounded and short, so they end the count rather than extend it.
const NORMALIZE_STEPS: u64 = 3;

#[derive(Debug)]
pub enum CommandOutcome {
    /// The command finished. `Some` carries the JSON object a `completed` event
    /// reports; `None` means the command has no result payload.
    Completed(Option<serde_json::Value>),
    Cancelled,
    Failed(TaskError),
}

pub fn execute(
    request: &Request,
    config: &WorkerConfig,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
) -> CommandOutcome {
    let runner = CancellableProcessRunner::new(token.clone());
    execute_with_runner(request, config, token, reporter, &runner)
}

pub fn execute_with_runner<R: ProcessRunner + ?Sized>(
    request: &Request,
    config: &WorkerConfig,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    runner: &R,
) -> CommandOutcome {
    if token.is_cancelled() {
        return CommandOutcome::Cancelled;
    }
    match request {
        Request::ValidateProject(params) => match validate_project_dir(&params.project_dir) {
            // Project validation is filesystem-bound and has no interrupt hook,
            // so cancellation is honoured at this boundary: the work is thrown
            // away rather than reported as a completed task.
            Ok(_) if token.is_cancelled() => CommandOutcome::Cancelled,
            Ok(_) => CommandOutcome::Completed(None),
            Err(error) => CommandOutcome::Failed(project_task_error(&error)),
        },
        Request::ProbeMedia(params) => {
            let Some(toolchain) = config.media() else {
                // Unreachable through the runtime, which rejects `probe_media`
                // when no toolchain is configured. Kept so a direct caller
                // cannot get a panic instead of an error.
                return CommandOutcome::Failed(unsupported(request.kind()));
            };
            let input = match validate_input(&MediaInput {
                source: params.input.clone(),
            }) {
                Ok(input) => input,
                Err(error) => return media_failure(&error),
            };
            match probe_media_with_runner(&input, toolchain, runner) {
                Ok(probe) => CommandOutcome::Completed(Some(probe_to_json(&probe))),
                Err(error) => media_failure(&error),
            }
        }
        Request::NormalizeMedia(params) => {
            let Some(toolchain) = config.media() else {
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
        other => CommandOutcome::Failed(unsupported(other.kind())),
    }
}

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

fn media_failure(error: &MediaError) -> CommandOutcome {
    if is_media_cancellation(error) {
        CommandOutcome::Cancelled
    } else {
        CommandOutcome::Failed(media_task_error(error))
    }
}

fn unsupported(kind: TaskKind) -> TaskError {
    TaskError::new(
        ErrorCode::WorkerCrashed,
        "当前 worker 不支持该命令",
        &format!(
            "command {} is not supported by this worker build",
            kind.as_slug()
        ),
        TaskStage::Preparing,
    )
}
