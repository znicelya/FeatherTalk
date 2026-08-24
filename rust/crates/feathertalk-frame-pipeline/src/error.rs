use thiserror::Error;

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("invalid frame pipeline field {field}: {message}")]
    InvalidField {
        field: &'static str,
        message: String,
    },
    #[error("invalid quality report field {field}: {message}")]
    InvalidReport { field: String, message: String },
}
