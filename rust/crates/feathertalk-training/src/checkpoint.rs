use std::{collections::BTreeMap, path::Path};

use burn::{
    module::AutodiffModule,
    optim::Optimizer,
    tensor::backend::AutodiffBackend,
};
use serde::{Deserialize, Serialize};

use crate::{DataLoaderState, TrainingError};

pub const TRAINING_CHECKPOINT_MANIFEST_SCHEMA_VERSION: u32 = 1;
pub const TRAINING_STATE_SCHEMA_VERSION: u32 = 1;
pub const TRAINING_CHECKPOINT_RECORD_FORMAT: &str = "burn-bin-full-precision-v1";
pub const TRAINING_CHECKPOINT_OPTIMIZER_KIND: &str = "adam";
pub const TRAINING_CHECKPOINT_OPTIMIZER_SCHEMA_VERSION: u32 = 1;

pub const CHECKPOINT_MANIFEST_FILE_NAME: &str = "manifest.json";
pub const CHECKPOINT_MODEL_FILE_NAME: &str = "model.bin";
pub const CHECKPOINT_OPTIMIZER_FILE_NAME: &str = "optimizer.bin";
pub const CHECKPOINT_STATE_FILE_NAME: &str = "training-state.json";

pub type Provenance = BTreeMap<String, String>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingMode {
    Baseline,
    MouthRoi,
    MouthRoiTemporal,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingConfig {
    pub mode: TrainingMode,
    pub batch_size: u64,
    pub learning_rate: f64,
    pub total_epochs: u64,
    pub temporal_stride: u64,
    pub mouth_weight: f64,
    pub temporal_weight: f64,
    pub temporal_mouth_weight: f64,
    pub perceptual_weight: f64,
}

impl TrainingConfig {
    pub fn validate(&self) -> Result<(), TrainingError> {
        if self.batch_size == 0 {
            return invalid_checkpoint("training_config.batch_size must be greater than zero");
        }
        if self.total_epochs == 0 {
            return invalid_checkpoint("training_config.total_epochs must be greater than zero");
        }
        validate_finite_non_negative("training_config.learning_rate", self.learning_rate)?;
        validate_finite_non_negative("training_config.mouth_weight", self.mouth_weight)?;
        validate_finite_non_negative("training_config.temporal_weight", self.temporal_weight)?;
        validate_finite_non_negative(
            "training_config.temporal_mouth_weight",
            self.temporal_mouth_weight,
        )?;
        validate_finite_non_negative(
            "training_config.perceptual_weight",
            self.perceptual_weight,
        )?;

        match self.mode {
            TrainingMode::Baseline | TrainingMode::MouthRoi if self.temporal_stride != 0 => {
                invalid_checkpoint(
                    "training_config.temporal_stride must be zero for non-temporal modes",
                )
            }
            TrainingMode::MouthRoiTemporal if self.temporal_stride == 0 => invalid_checkpoint(
                "training_config.temporal_stride must be greater than zero for temporal mode",
            ),
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingCheckpointState {
    pub schema_version: u32,
    pub epoch: u64,
    pub global_step: u64,
    pub random_seed: u64,
    pub data_loader: DataLoaderState,
    pub training_config: TrainingConfig,
    pub asset_provenance: Provenance,
    pub model_provenance: Provenance,
}

impl TrainingCheckpointState {
    pub fn validate(&self) -> Result<(), TrainingError> {
        if self.schema_version != TRAINING_STATE_SCHEMA_VERSION {
            return invalid_checkpoint(format!(
                "unsupported training state schema_version {}, expected {}",
                self.schema_version, TRAINING_STATE_SCHEMA_VERSION
            ));
        }
        self.training_config.validate()?;
        self.data_loader.validate(self.data_loader.frame_count)?;
        if self.epoch != self.data_loader.epoch {
            return invalid_checkpoint("training state epoch must equal data_loader.epoch");
        }
        if self.random_seed != self.data_loader.config.seed {
            return invalid_checkpoint("random_seed must equal data_loader.config.seed");
        }
        if self.training_config.batch_size != self.data_loader.config.batch_size {
            return invalid_checkpoint(
                "training_config.batch_size must equal data_loader.config.batch_size",
            );
        }
        if self.training_config.temporal_stride
            != self.data_loader.config.sampling.temporal_stride
        {
            return invalid_checkpoint(
                "training_config.temporal_stride must equal data_loader sampling stride",
            );
        }
        validate_provenance("asset_provenance", &self.asset_provenance)?;
        validate_provenance("model_provenance", &self.model_provenance)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointFileManifest {
    pub file_name: String,
    pub bytes: u64,
    pub sha256: String,
}

impl CheckpointFileManifest {
    pub fn validate(&self, expected_file_name: &str) -> Result<(), TrainingError> {
        if self.file_name != expected_file_name {
            return invalid_checkpoint(format!(
                "file_name must be {expected_file_name}, got {}",
                self.file_name
            ));
        }
        if self.bytes == 0 {
            return invalid_checkpoint(format!(
                "{} must contain at least one byte",
                self.file_name
            ));
        }
        validate_sha256(&format!("{}.sha256", self.file_name), &self.sha256)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointDescriptor {
    pub model_kind: String,
    pub architecture_version: String,
    pub model_config_sha256: String,
    pub optimizer_kind: String,
    pub optimizer_schema_version: u32,
}

impl CheckpointDescriptor {
    pub fn new(
        model_kind: impl Into<String>,
        architecture_version: impl Into<String>,
        model_config_sha256: impl Into<String>,
    ) -> Self {
        Self {
            model_kind: model_kind.into(),
            architecture_version: architecture_version.into(),
            model_config_sha256: model_config_sha256.into(),
            optimizer_kind: TRAINING_CHECKPOINT_OPTIMIZER_KIND.to_owned(),
            optimizer_schema_version: TRAINING_CHECKPOINT_OPTIMIZER_SCHEMA_VERSION,
        }
    }

    pub fn validate(&self) -> Result<(), TrainingError> {
        validate_non_empty("model_kind", &self.model_kind)?;
        validate_non_empty("architecture_version", &self.architecture_version)?;
        validate_sha256("model_config_sha256", &self.model_config_sha256)?;
        if self.optimizer_kind != TRAINING_CHECKPOINT_OPTIMIZER_KIND {
            return invalid_checkpoint(format!(
                "optimizer_kind must be {}, got {}",
                TRAINING_CHECKPOINT_OPTIMIZER_KIND, self.optimizer_kind
            ));
        }
        if self.optimizer_schema_version != TRAINING_CHECKPOINT_OPTIMIZER_SCHEMA_VERSION {
            return invalid_checkpoint(format!(
                "unsupported optimizer_schema_version {}, expected {}",
                self.optimizer_schema_version, TRAINING_CHECKPOINT_OPTIMIZER_SCHEMA_VERSION
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingCheckpointManifest {
    pub schema_version: u32,
    pub record_format: String,
    pub model_kind: String,
    pub architecture_version: String,
    pub model_config_sha256: String,
    pub optimizer_kind: String,
    pub optimizer_schema_version: u32,
    pub model: CheckpointFileManifest,
    pub optimizer: CheckpointFileManifest,
    pub training_state: CheckpointFileManifest,
    pub training_state_sha256: String,
    pub burn_version: String,
    pub rust_version: String,
}

impl TrainingCheckpointManifest {
    pub fn validate(&self) -> Result<(), TrainingError> {
        if self.schema_version != TRAINING_CHECKPOINT_MANIFEST_SCHEMA_VERSION {
            return invalid_checkpoint(format!(
                "unsupported manifest schema_version {}, expected {}",
                self.schema_version, TRAINING_CHECKPOINT_MANIFEST_SCHEMA_VERSION
            ));
        }
        if self.record_format != TRAINING_CHECKPOINT_RECORD_FORMAT {
            return invalid_checkpoint(format!(
                "unsupported record_format {}, expected {}",
                self.record_format, TRAINING_CHECKPOINT_RECORD_FORMAT
            ));
        }
        CheckpointDescriptor {
            model_kind: self.model_kind.clone(),
            architecture_version: self.architecture_version.clone(),
            model_config_sha256: self.model_config_sha256.clone(),
            optimizer_kind: self.optimizer_kind.clone(),
            optimizer_schema_version: self.optimizer_schema_version,
        }
        .validate()?;
        self.model.validate(CHECKPOINT_MODEL_FILE_NAME)?;
        self.optimizer.validate(CHECKPOINT_OPTIMIZER_FILE_NAME)?;
        self.training_state.validate(CHECKPOINT_STATE_FILE_NAME)?;
        validate_sha256("training_state_sha256", &self.training_state_sha256)?;
        if self.training_state_sha256 != self.training_state.sha256 {
            return invalid_checkpoint("training_state_sha256 must equal training_state.sha256");
        }
        validate_non_empty("burn_version", &self.burn_version)?;
        validate_non_empty("rust_version", &self.rust_version)
    }

    pub fn descriptor(&self) -> CheckpointDescriptor {
        CheckpointDescriptor {
            model_kind: self.model_kind.clone(),
            architecture_version: self.architecture_version.clone(),
            model_config_sha256: self.model_config_sha256.clone(),
            optimizer_kind: self.optimizer_kind.clone(),
            optimizer_schema_version: self.optimizer_schema_version,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointCompatibility {
    pub descriptor: CheckpointDescriptor,
    pub training_config: TrainingConfig,
    pub frame_count: u64,
    pub asset_provenance: Provenance,
    pub model_provenance: Provenance,
}

impl CheckpointCompatibility {
    pub fn new(
        descriptor: CheckpointDescriptor,
        training_config: TrainingConfig,
        frame_count: u64,
    ) -> Self {
        Self {
            descriptor,
            training_config,
            frame_count,
            asset_provenance: BTreeMap::new(),
            model_provenance: BTreeMap::new(),
        }
    }

    pub fn validate(&self) -> Result<(), TrainingError> {
        self.descriptor.validate()?;
        self.training_config.validate()?;
        if self.frame_count == 0 {
            return Err(TrainingError::CheckpointCompatibility(
                "frame_count must be greater than zero".to_owned(),
            ));
        }
        validate_provenance("asset_provenance", &self.asset_provenance)?;
        validate_provenance("model_provenance", &self.model_provenance)
    }

    pub fn validate_manifest_state(
        &self,
        manifest: &TrainingCheckpointManifest,
        state: &TrainingCheckpointState,
    ) -> Result<(), TrainingError> {
        self.validate()?;
        manifest.validate()?;
        state.validate()?;
        if manifest.descriptor() != self.descriptor {
            return Err(TrainingError::CheckpointCompatibility(
                "model or optimizer descriptor does not match expected compatibility".to_owned(),
            ));
        }
        if state.data_loader.frame_count != self.frame_count {
            return Err(TrainingError::CheckpointCompatibility(
                "data loader frame_count does not match expected compatibility".to_owned(),
            ));
        }
        if state.training_config != self.training_config {
            return Err(TrainingError::CheckpointCompatibility(
                "training configuration does not match expected compatibility".to_owned(),
            ));
        }
        if state.asset_provenance != self.asset_provenance {
            return Err(TrainingError::CheckpointCompatibility(
                "asset provenance does not match expected compatibility".to_owned(),
            ));
        }
        if state.model_provenance != self.model_provenance {
            return Err(TrainingError::CheckpointCompatibility(
                "model provenance does not match expected compatibility".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct RestoredTrainingState<M, O> {
    pub model: M,
    pub optimizer: O,
    pub state: TrainingCheckpointState,
    pub manifest: TrainingCheckpointManifest,
}

pub fn save_training_checkpoint<B, M, O>(
    _destination: impl AsRef<Path>,
    _model: &M,
    _optimizer: &O,
    _descriptor: CheckpointDescriptor,
    _state: TrainingCheckpointState,
) -> Result<TrainingCheckpointManifest, TrainingError>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Clone,
    O: Optimizer<M, B> + Clone,
{
    Err(TrainingError::CheckpointDirectory(
        "checkpoint saving is not implemented yet".to_owned(),
    ))
}

pub fn load_training_checkpoint<B, M, O>(
    _directory: impl AsRef<Path>,
    _model_template: &M,
    _optimizer_template: &O,
    _device: &B::Device,
    _expected: &CheckpointCompatibility,
) -> Result<RestoredTrainingState<M, O>, TrainingError>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Clone,
    O: Optimizer<M, B> + Clone,
{
    Err(TrainingError::CheckpointDirectory(
        "checkpoint loading is not implemented yet".to_owned(),
    ))
}

fn validate_provenance(name: &str, values: &Provenance) -> Result<(), TrainingError> {
    for (key, value) in values {
        validate_non_empty(&format!("{name} key"), key)?;
        validate_sha256(&format!("{name}.{key}"), value)?;
    }
    Ok(())
}

fn validate_sha256(name: &str, value: &str) -> Result<(), TrainingError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return invalid_checkpoint(format!(
            "{name} must be 64 lowercase hexadecimal characters"
        ));
    }
    Ok(())
}

fn validate_non_empty(name: &str, value: &str) -> Result<(), TrainingError> {
    if value.trim().is_empty() {
        return invalid_checkpoint(format!("{name} must be non-empty"));
    }
    Ok(())
}

fn validate_finite_non_negative(name: &str, value: f64) -> Result<(), TrainingError> {
    if !value.is_finite() || value < 0.0 {
        return invalid_checkpoint(format!(
            "{name} must be finite and non-negative, got {value}"
        ));
    }
    Ok(())
}

fn invalid_checkpoint<T>(message: impl Into<String>) -> Result<T, TrainingError> {
    Err(TrainingError::InvalidCheckpoint(message.into()))
}
