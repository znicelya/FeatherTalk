use serde::{Deserialize, Serialize};

use crate::TrainingError;

pub const DATA_LOADER_STATE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RandomAlgorithm {
    Splitmix64FisherYatesV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingKind {
    SingleFrame,
    TemporalPair,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingConfig {
    pub kind: SamplingKind,
    pub temporal_stride: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataLoaderConfig {
    pub batch_size: u64,
    pub seed: u64,
    pub sampling: SamplingConfig,
}

impl DataLoaderConfig {
    pub const fn single_frame(batch_size: u64, seed: u64) -> Self {
        Self {
            batch_size,
            seed,
            sampling: SamplingConfig {
                kind: SamplingKind::SingleFrame,
                temporal_stride: 0,
            },
        }
    }

    pub const fn temporal_pair(batch_size: u64, seed: u64, temporal_stride: u64) -> Self {
        Self {
            batch_size,
            seed,
            sampling: SamplingConfig {
                kind: SamplingKind::TemporalPair,
                temporal_stride,
            },
        }
    }

    pub fn validate(&self, frame_count: u64) -> Result<(), TrainingError> {
        self.sample_count(frame_count).map(|_| ())
    }

    pub(crate) fn sample_count(&self, frame_count: u64) -> Result<u64, TrainingError> {
        if self.batch_size == 0 {
            return Err(TrainingError::InvalidDataLoaderConfig(
                "batch_size must be greater than zero".into(),
            ));
        }
        usize::try_from(self.batch_size).map_err(|_| TrainingError::DataLoaderOverflow {
            operation: "converting batch size",
        })?;
        if frame_count == 0 {
            return Err(TrainingError::InvalidDataLoaderConfig(
                "frame_count must be greater than zero".into(),
            ));
        }
        usize::try_from(frame_count).map_err(|_| TrainingError::DataLoaderOverflow {
            operation: "converting frame count",
        })?;

        let sample_count = match self.sampling.kind {
            SamplingKind::SingleFrame => {
                if self.sampling.temporal_stride != 0 {
                    return Err(TrainingError::InvalidDataLoaderConfig(
                        "single_frame requires temporal_stride zero".into(),
                    ));
                }
                frame_count
            }
            SamplingKind::TemporalPair => {
                let stride = self.sampling.temporal_stride;
                if stride == 0 || stride >= frame_count {
                    return Err(TrainingError::InvalidDataLoaderConfig(
                        "temporal_pair requires 1 <= temporal_stride < frame_count".into(),
                    ));
                }
                frame_count
                    .checked_sub(stride)
                    .ok_or(TrainingError::DataLoaderOverflow {
                        operation: "computing temporal sample count",
                    })?
            }
        };
        usize::try_from(sample_count).map_err(|_| TrainingError::DataLoaderOverflow {
            operation: "converting sample count",
        })?;
        Ok(sample_count)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataLoaderState {
    pub schema_version: u32,
    pub random_algorithm: RandomAlgorithm,
    pub config: DataLoaderConfig,
    pub frame_count: u64,
    pub epoch: u64,
    pub next_position: u64,
}

impl DataLoaderState {
    pub fn validate(&self, dataset_frame_count: u64) -> Result<(), TrainingError> {
        if self.schema_version != DATA_LOADER_STATE_SCHEMA_VERSION {
            return Err(TrainingError::InvalidDataLoaderState(
                "unsupported schema_version".into(),
            ));
        }
        if self.random_algorithm != RandomAlgorithm::Splitmix64FisherYatesV1 {
            return Err(TrainingError::InvalidDataLoaderState(
                "unsupported random_algorithm".into(),
            ));
        }
        if self.frame_count != dataset_frame_count {
            return Err(TrainingError::InvalidDataLoaderState(
                "dataset frame_count does not match saved state".into(),
            ));
        }
        let sample_count = self.config.sample_count(self.frame_count)?;
        if self.next_position >= sample_count {
            return Err(TrainingError::InvalidDataLoaderState(
                "next_position must be inside the current epoch".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainingSample {
    SingleFrame {
        target_index: u64,
        reference_index: u64,
    },
    TemporalPair {
        first_target_index: u64,
        second_target_index: u64,
        reference_index: u64,
    },
}
