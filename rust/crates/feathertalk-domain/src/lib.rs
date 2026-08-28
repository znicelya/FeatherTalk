mod error;
mod task;

pub const PROTOCOL_VERSION: u32 = 1;

pub use error::DomainError;
pub use task::{
    TASK_ID_LEN, TASK_ID_MILLIS_DIGITS, TASK_ID_SUFFIX_DIGITS, TaskId, TaskKind, TaskStatus,
};
