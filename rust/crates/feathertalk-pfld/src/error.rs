use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PfldError {
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
