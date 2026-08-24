use crate::{AudioError, FeatureMatrix, plan_chunks};

pub trait ChunkEncoder {
    fn output_dim(&self) -> usize;
    fn encode(&mut self, chunk_index: usize, samples: &[f32]) -> Result<Vec<f32>, AudioError>;
}

pub fn extract_long_audio<E: ChunkEncoder>(
    samples: &[f32],
    encoder: &mut E,
    chunk_samples: usize,
) -> Result<FeatureMatrix, AudioError> {
    for (index, value) in samples.iter().enumerate() {
        if !value.is_finite() {
            return Err(AudioError::NonFiniteWaveform { index });
        }
    }
    let dimension = encoder.output_dim();
    if dimension == 0 {
        return Err(AudioError::InvalidFeatureDimension);
    }
    let plan = plan_chunks(samples.len(), chunk_samples)?;
    let mut values = Vec::new();
    for range in plan.ranges() {
        let output = encoder.encode(range.index(), &samples[range.start()..range.end()])?;
        if !output.len().is_multiple_of(dimension) {
            return Err(AudioError::FeatureLengthMismatch {
                actual: output.len(),
                dimension,
            });
        }
        for (index, value) in output.iter().enumerate() {
            if !value.is_finite() {
                return Err(AudioError::NonFiniteFeature { index });
            }
        }
        values.extend(output);
    }
    let target_values = plan
        .target_tokens()
        .checked_mul(dimension)
        .ok_or(AudioError::FeatureSizeOverflow)?;
    if values.len() < target_values {
        values.resize(target_values, 0.0);
    } else {
        values.truncate(target_values);
    }
    FeatureMatrix::new(plan.target_tokens(), dimension, values)
}

pub fn drop_odd_token(matrix: FeatureMatrix) -> FeatureMatrix {
    if matrix.tokens().is_multiple_of(2) {
        matrix
    } else {
        let tokens = matrix.tokens() - 1;
        let values = matrix.values()[..tokens * matrix.dims()].to_vec();
        FeatureMatrix::new(tokens, matrix.dims(), values).expect("validated feature matrix")
    }
}
