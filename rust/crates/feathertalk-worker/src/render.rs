//! The render loop and the progress it reports.

use std::sync::atomic::{AtomicU64, Ordering};

use feathertalk_domain::{Progress, TaskStage};
use feathertalk_inference::{
    BgrFrame, CommandSpec, FrameReader, InferenceError, RawVideoSink, RawVideoSinkFactory,
    execute_offline_render,
};
use feathertalk_media::CancellationToken;
use feathertalk_models::unet::TalkingHeadModel;

use crate::{
    CommandOutcome, RenderBackend, RenderDevice, RenderJob, RenderSummary, TaskReporter,
    is_inference_cancellation, render_task_error, render_to_json,
};

/// Wraps the caller's sink factory so the render reports one progress event per
/// written frame and stops at the next frame once the task is cancelled.
///
/// The sink borrows the reporter, which is why `RawVideoSinkFactory` no longer
/// requires `Send + Sync`: a `TaskReporter` is not `Sync`, and the render loop
/// runs on this thread anyway.
struct ObservedSinkFactory<'a, F: ?Sized> {
    inner: &'a F,
    reporter: &'a dyn TaskReporter,
    token: &'a CancellationToken,
    /// The frames the render will write, from the locked manifest and the cap.
    total: u64,
    /// The frames it has written so far. Interior mutability because `start` and
    /// `write_frame` only ever get a shared borrow of the factory.
    frames: AtomicU64,
}

impl<F: ?Sized> ObservedSinkFactory<'_, F> {
    /// Counts one written frame and reports it.
    fn observe(&self) {
        let frame = self
            .frames
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1);
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
