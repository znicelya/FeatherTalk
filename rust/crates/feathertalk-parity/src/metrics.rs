use ndarray::ArrayViewD;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParityMetrics {
    pub max_abs: f32,
    pub mean_abs: f32,
    pub max_relative: f32,
}

#[derive(Debug, Error)]
pub enum ParityError {
    #[error("array shapes differ: actual {actual:?}, expected {expected:?}")]
    ShapeMismatch {
        actual: Vec<usize>,
        expected: Vec<usize>,
    },
    #[error("cannot compare empty arrays")]
    EmptyArray,
    #[error("non-finite value at element {index}: actual {actual}, expected {expected}")]
    NonFinite {
        index: usize,
        actual: f32,
        expected: f32,
    },
    #[error("fixture error: {0}")]
    Fixture(#[from] crate::archive::FixtureError),
    #[error("weight import error: {0}")]
    WeightImport(#[from] feathertalk_weights::WeightImportError),
    #[error("tensor data error: {0}")]
    TensorData(String),
    #[error("array construction error: {0}")]
    Array(String),
    #[error("fixture array is missing: {0}")]
    MissingArray(String),
}

pub fn compare_f32(
    actual: ArrayViewD<'_, f32>,
    expected: ArrayViewD<'_, f32>,
) -> Result<ParityMetrics, ParityError> {
    if actual.shape() != expected.shape() {
        return Err(ParityError::ShapeMismatch {
            actual: actual.shape().to_vec(),
            expected: expected.shape().to_vec(),
        });
    }
    if actual.is_empty() {
        return Err(ParityError::EmptyArray);
    }

    let mut max_abs = 0.0_f32;
    let mut mean_abs = 0.0_f32;
    let mut max_relative = 0.0_f32;
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        if !actual.is_finite() || !expected.is_finite() {
            return Err(ParityError::NonFinite {
                index,
                actual,
                expected,
            });
        }
        let absolute = (actual - expected).abs();
        max_abs = max_abs.max(absolute);
        mean_abs += absolute;
        max_relative = max_relative.max(absolute / expected.abs().max(1e-7));
    }

    Ok(ParityMetrics {
        max_abs,
        mean_abs: mean_abs / actual.len() as f32,
        max_relative,
    })
}
