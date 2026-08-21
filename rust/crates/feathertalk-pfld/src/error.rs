use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PfldError {
    #[error("I/O error during {operation} at {path}: {source}")]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("mean face file is not valid UTF-8: {path}")]
    InvalidUtf8 { path: PathBuf },
    #[error("invalid mean face token at index {index} in {path}")]
    InvalidMeanFaceToken { path: PathBuf, index: usize },
    #[error("invalid mean face count in {path}: expected {expected}, got {actual}")]
    InvalidMeanFaceCount {
        path: PathBuf,
        expected: usize,
        actual: usize,
    },
    #[error("invalid vector length for {field}: expected {expected}, got {actual}")]
    InvalidVectorLength {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("non-finite value for {field} at index {index}")]
    NonFiniteValue { field: &'static str, index: usize },
    #[error("crop width and height must be non-zero")]
    InvalidCropGeometry,
    #[error("decoded coordinate {axis} at landmark index {index} is outside i32 range")]
    CoordinateOutOfRange { index: usize, axis: &'static str },
}
