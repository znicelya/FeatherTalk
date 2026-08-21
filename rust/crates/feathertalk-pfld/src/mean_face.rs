use std::{fs, path::Path};

use crate::{PFLD_OUTPUT_VALUE_COUNT, PfldError};

#[derive(Debug, Clone, PartialEq)]
pub struct MeanFace {
    values: [f32; PFLD_OUTPUT_VALUE_COUNT],
}

impl MeanFace {
    pub fn values(&self) -> &[f32; PFLD_OUTPUT_VALUE_COUNT] {
        &self.values
    }
}

pub fn read_mean_face(path: &Path) -> Result<MeanFace, PfldError> {
    let bytes = fs::read(path).map_err(|source| PfldError::Io {
        operation: "read_mean_face",
        path: path.to_path_buf(),
        source,
    })?;
    let text = String::from_utf8(bytes).map_err(|_| PfldError::InvalidUtf8 {
        path: path.to_path_buf(),
    })?;
    let mut values = Vec::new();
    for (index, token) in text.split_whitespace().enumerate() {
        let value = token
            .parse::<f32>()
            .map_err(|_| PfldError::InvalidMeanFaceToken {
                path: path.to_path_buf(),
                index,
            })?;
        if !value.is_finite() {
            return Err(PfldError::NonFiniteValue {
                field: "mean_face",
                index,
            });
        }
        values.push(value);
    }
    if values.len() != PFLD_OUTPUT_VALUE_COUNT {
        return Err(PfldError::InvalidMeanFaceCount {
            path: path.to_path_buf(),
            expected: PFLD_OUTPUT_VALUE_COUNT,
            actual: values.len(),
        });
    }
    let values = values
        .try_into()
        .map_err(|values: Vec<f32>| PfldError::InvalidMeanFaceCount {
            path: path.to_path_buf(),
            expected: PFLD_OUTPUT_VALUE_COUNT,
            actual: values.len(),
        })?;
    Ok(MeanFace { values })
}
