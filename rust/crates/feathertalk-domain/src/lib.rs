mod error;
mod lifecycle;
mod stage;
mod task;
mod task_error;

pub const PROTOCOL_VERSION: u32 = 1;

pub use error::DomainError;
pub use lifecycle::TaskLifecycle;
pub use stage::TaskStage;
pub use task::{
    TASK_ID_LEN, TASK_ID_MILLIS_DIGITS, TASK_ID_SUFFIX_DIGITS, TaskId, TaskKind, TaskStatus,
};
pub use task_error::{ErrorCode, MAX_DETAIL_CHARS, MAX_SUMMARY_CHARS, Recovery, TaskError};
