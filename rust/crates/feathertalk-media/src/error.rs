use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MediaError {
    #[error("I/O error during {operation} at {path}: {source}")]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("media input does not exist: {path}")]
    InputMissing { path: PathBuf },
    #[error("media input is not a regular file: {path}")]
    InputNotRegularFile { path: PathBuf },
    #[error("symbolic link is not allowed: {path}")]
    SymlinkNotAllowed { path: PathBuf },
    #[error("output directory is invalid: {path}")]
    OutputDirectoryInvalid { path: PathBuf },
    #[error("output directory contains the input: input={input}, output={output}")]
    OutputInsideInput { input: PathBuf, output: PathBuf },
    #[error("normalization output conflicts with input: {path}")]
    OutputConflictsWithInput { path: PathBuf },
    #[error("normalization output destination is invalid: {path}")]
    OutputDestinationInvalid { path: PathBuf },
    #[error("unsupported normalization target for {field}: expected {expected}, got {actual}")]
    UnsupportedTarget {
        field: &'static str,
        expected: &'static str,
        actual: String,
    },
}
