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
