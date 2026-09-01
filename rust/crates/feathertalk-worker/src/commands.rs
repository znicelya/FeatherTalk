use feathertalk_domain::{ErrorCode, Request, TaskError, TaskKind, TaskStage};
use feathertalk_media::{
    CancellableProcessRunner, CancellationToken, MediaError, MediaInput, MediaToolchain,
    ProcessRunner, probe_media_with_runner, validate_input,
};
use feathertalk_project::validate_project_dir;

use crate::{is_media_cancellation, media_task_error, probe_to_json, project_task_error};

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
    media: Option<&MediaToolchain>,
    token: &CancellationToken,
) -> CommandOutcome {
    let runner = CancellableProcessRunner::new(token.clone());
    execute_with_runner(request, media, token, &runner)
}

pub fn execute_with_runner<R: ProcessRunner + ?Sized>(
    request: &Request,
    media: Option<&MediaToolchain>,
    token: &CancellationToken,
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
            let Some(toolchain) = media else {
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
        other => CommandOutcome::Failed(unsupported(other.kind())),
    }
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
