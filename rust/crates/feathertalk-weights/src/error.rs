use thiserror::Error;

#[derive(Debug, Error)]
pub enum WeightImportError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("unsafe checkpoint limit: {0}")]
    UnsafeLimit(String),
    #[error("unsupported checkpoint structure: {0}")]
    UnsupportedStructure(String),
    #[error("missing tensor: {0}")]
    MissingTensor(String),
    #[error("unexpected tensor: {0}")]
    UnexpectedTensor(String),
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("dtype mismatch: {0}")]
    DTypeMismatch(String),
    #[error("duplicate remapped tensor key: {0}")]
    DuplicateKey(String),
    #[error("Burn store error: {0}")]
    Store(String),
}
