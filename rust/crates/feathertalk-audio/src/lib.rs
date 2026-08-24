mod chunk;
mod error;
mod format;
mod normalize;
mod stitch;

pub use chunk::{
    ChunkPlan, ChunkRange, DEFAULT_CHUNK_SAMPLES, HUBERT_KERNEL, HUBERT_STRIDE,
    expected_hubert_frames, plan_chunks,
};
pub use error::AudioError;
pub use format::{FeatureArtifact, MAX_FEATURE_FILE_BYTES, read_feature_file, write_feature_file};
pub use normalize::normalize_waveform;
pub use stitch::{ChunkEncoder, drop_odd_token, extract_long_audio};

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureMatrix {
    tokens: usize,
    dims: usize,
    values: Vec<f32>,
}

impl FeatureMatrix {
    pub fn new(tokens: usize, dims: usize, values: Vec<f32>) -> Result<Self, AudioError> {
        if dims == 0 {
            return Err(AudioError::InvalidFeatureDimension);
        }
        let expected = tokens
            .checked_mul(dims)
            .ok_or(AudioError::FeatureSizeOverflow)?;
        if values.len() != expected {
            return Err(AudioError::FeatureLengthMismatch {
                actual: values.len(),
                dimension: dims,
            });
        }
        for (index, value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(AudioError::NonFiniteFeature { index });
            }
        }
        Ok(Self {
            tokens,
            dims,
            values,
        })
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }
    pub fn dims(&self) -> usize {
        self.dims
    }
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}
