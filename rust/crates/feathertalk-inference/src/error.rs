use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("source frame count {actual} is too small; expected at least {minimum}")]
    FrameCountTooSmall { actual: usize, minimum: usize },
    #[error("feature frame count must be greater than zero")]
    EmptyFeatures,
    #[error("output frame index {index} is outside output count {count}")]
    OutputFrameOutOfRange { index: usize, count: usize },
    #[error("invalid inference field {field}: {message}")]
    InvalidField {
        field: &'static str,
        message: String,
    },
    #[error("arithmetic overflow while building inference plan")]
    ArithmeticOverflow,
    #[error("output destination already exists: {path}")]
    OutputExists { path: PathBuf },
    #[error("output destination is not a regular non-symlink file: {path}")]
    OutputNotRegular { path: PathBuf },
    #[error("symbolic link encountered in output path: {path}")]
    OutputSymlink { path: PathBuf },
    #[error("output parent directory is missing or invalid: {path}")]
    OutputParentInvalid { path: PathBuf },
    #[error("task id is invalid: {task_id}")]
    InvalidTaskId { task_id: String },
    #[error("FFmpeg path must be absolute: {path}")]
    FfmpegPathNotAbsolute { path: PathBuf },
    #[error("FFmpeg path must not be empty")]
    EmptyFfmpegPath,
}
