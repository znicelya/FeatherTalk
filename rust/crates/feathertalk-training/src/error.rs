#[derive(Debug, thiserror::Error)]
pub enum TrainingError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid training input: {0}")]
    InvalidInput(String),
    #[error("invalid training configuration: {0}")]
    InvalidConfig(String),
    #[error("invalid VGG19 package: {0}")]
    InvalidPackage(String),
    #[error("VGG19 package hash mismatch for {file}: expected {expected}, got {actual}")]
    HashMismatch {
        file: String,
        expected: String,
        actual: String,
    },
    #[error("Burn store error: {0}")]
    Store(String),
}
