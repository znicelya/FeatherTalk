use std::path::{Path, PathBuf};

use burn::tensor::Device;
use feathertalk_domain::{TrainParams, TrainingMode as DomainTrainingMode, UnetVariant};
use feathertalk_export::ModelConfiguration;
use feathertalk_models::backend::CpuAutodiffBackend;
use feathertalk_training::{CheckpointDescriptor, TrainingConfig, TrainingError, TrainingMode};
use sha2::{Digest, Sha256};

/// The backend every training run in this slice uses.
///
/// Design section 4: the handshake keeps `wgpu_training` false, so the worker
/// never promises GPU training and there is no silent fallback to explain. The
/// runner, the losses and the perceptual extractor are all generic over
/// `AutodiffBackend`, so a later GPU slice replaces this alias and adds one
/// dispatch, not a rewrite.
pub type TrainBackend = CpuAutodiffBackend;

/// The device that goes with it. `NdArrayDevice` is `Copy`, so it is passed by
/// reference and copied rather than cloned. Burn 0.21 hangs `Device` off
/// `BackendTypes`, not `Backend`, so the alias goes through `burn::tensor::Device`
/// the way the rest of the workspace does.
pub type TrainDevice = Device<TrainBackend>;

/// What the result payload calls the backend, so the artefact records what it
/// actually ran on (design section 12).
pub const TRAIN_BACKEND_NAME: &str = "ndarray-cpu";

/// One sample per step: the wire protocol carries no batch size, so any value is
/// a placeholder. One keeps peak memory smallest and makes a step equal a
/// sample, which is the finest possible granularity for progress and loss.
pub const DEFAULT_BATCH_SIZE: u64 = 1;

/// The migration design's learning rate (section 8.2).
pub const DEFAULT_LEARNING_RATE: f64 = 1e-4;

/// The sampling seed. Fixed, so two runs of the same project see the same order.
pub const TRAINING_SEED: u64 = 1;

/// The largest accepted `epochs`. `TrainingConfig::validate` would reject zero
/// anyway; the ceiling exists so a typo cannot ask for a run no one can finish.
pub const MAX_EPOCHS: u32 = 10_000;

/// The `worker_state` every metrics file and preview manifest records.
/// `validate_worker_state` accepts 1 to 128 lowercase letters, digits, hyphens
/// and underscores.
pub const WORKER_STATE: &str = "training";

/// Loss weights, all from the migration design section 8.2.
const MOUTH_WEIGHT: f64 = 4.0;
const TEMPORAL_WEIGHT: f64 = 0.5;
const TEMPORAL_MOUTH_WEIGHT: f64 = 4.0;
const PERCEPTUAL_WEIGHT: f64 = 0.01;

const MODELS_DIR: &str = "models";
const UNET_DIR: &str = "unet";
const OUTPUTS_DIR: &str = "outputs";
const METRICS_DIR: &str = "metrics";
const PREVIEW_DIR: &str = "preview";
const CHECKPOINT_PREFIX: &str = "checkpoint-";
const STEP_PREFIX: &str = "step-";

/// How many digits a step number is padded to in an artefact name.
const STEP_DIGITS: usize = 8;

/// Maps the request's mode onto the training crate's mode. The two enums have
/// three variants each and only two names in common.
pub fn training_mode(mode: DomainTrainingMode) -> TrainingMode {
    match mode {
        DomainTrainingMode::Baseline => TrainingMode::Baseline,
        DomainTrainingMode::MouthRoi => TrainingMode::MouthRoi,
        DomainTrainingMode::Temporal => TrainingMode::MouthRoiTemporal,
    }
}

/// `TrainingConfig::validate` demands a zero stride outside the temporal mode
/// and a positive one inside it, and `DataLoaderConfig::sample_count` subtracts
/// the stride from the frame count.
fn temporal_stride(mode: DomainTrainingMode) -> u64 {
    match mode {
        DomainTrainingMode::Baseline | DomainTrainingMode::MouthRoi => 0,
        DomainTrainingMode::Temporal => 1,
    }
}

/// How many samples one epoch holds. A temporal pair needs a successor, so the
/// last frame starts no sample.
pub fn sample_count(mode: DomainTrainingMode, frame_count: u64) -> u64 {
    match mode {
        DomainTrainingMode::Baseline | DomainTrainingMode::MouthRoi => frame_count,
        DomainTrainingMode::Temporal => frame_count.saturating_sub(1),
    }
}

/// The nine-field training config: four fields from the request, five from the
/// constants above (design section 5).
pub fn training_config(params: &TrainParams) -> TrainingConfig {
    TrainingConfig {
        mode: training_mode(params.mode),
        batch_size: DEFAULT_BATCH_SIZE,
        learning_rate: DEFAULT_LEARNING_RATE,
        total_epochs: u64::from(params.epochs),
        temporal_stride: temporal_stride(params.mode),
        mouth_weight: MOUTH_WEIGHT,
        temporal_weight: TEMPORAL_WEIGHT,
        temporal_mouth_weight: TEMPORAL_MOUTH_WEIGHT,
        perceptual_weight: PERCEPTUAL_WEIGHT,
    }
}

/// Derives the checkpoint descriptor from the model configuration instead of
/// hand-writing its three fields.
///
/// `ModelConfiguration` is a fixed-field, map-free structure, so its serialised
/// bytes are stable and their digest is the natural canonical form of "this
/// model configuration". `CheckpointDescriptor::validate` requires 64 lowercase
/// hex characters, which is exactly what `hex::encode` produces. This is the
/// first place in the workspace that computes the value; every existing test
/// uses a repeated-digit placeholder.
pub fn checkpoint_descriptor(
    configuration: &ModelConfiguration,
) -> Result<CheckpointDescriptor, TrainingError> {
    let bytes = serde_json::to_vec(configuration).map_err(|error| {
        TrainingError::InvalidConfig(format!("serialize model configuration: {error}"))
    })?;
    let descriptor = CheckpointDescriptor::new(
        configuration.model_type(),
        configuration.architecture_version(),
        hex::encode(Sha256::digest(&bytes)),
    );
    descriptor.validate()?;
    Ok(descriptor)
}

/// Where a training run writes. Every directory is created by its writer, so
/// nothing here touches the filesystem.
#[derive(Debug, Clone)]
pub struct TrainingPaths {
    checkpoints: PathBuf,
    metrics: PathBuf,
    previews: PathBuf,
}

impl TrainingPaths {
    pub fn new(project_dir: &Path) -> Self {
        Self {
            checkpoints: project_dir.join(MODELS_DIR).join(UNET_DIR),
            metrics: project_dir.join(OUTPUTS_DIR).join(METRICS_DIR),
            previews: project_dir.join(OUTPUTS_DIR).join(PREVIEW_DIR),
        }
    }

    /// The directory every checkpoint of this project lives in.
    pub fn checkpoints(&self) -> &Path {
        &self.checkpoints
    }

    /// `models/unet/checkpoint-00000188`.
    ///
    /// Named by step, never by epoch: a cancellation lands mid-epoch, and
    /// `DataLoaderState.next_position` already carries the position inside the
    /// epoch, so one naming scheme covers both save points and resume only has
    /// to take the largest number.
    pub fn checkpoint(&self, global_step: u64) -> PathBuf {
        self.checkpoints
            .join(format!("{CHECKPOINT_PREFIX}{global_step:08}"))
    }

    /// `outputs/metrics/step-00000188.json`.
    pub fn metrics(&self, global_step: u64) -> PathBuf {
        self.metrics
            .join(format!("{STEP_PREFIX}{global_step:08}.json"))
    }

    /// `outputs/preview/step-00000188`.
    pub fn preview(&self, global_step: u64) -> PathBuf {
        self.previews.join(format!("{STEP_PREFIX}{global_step:08}"))
    }

    /// The step a checkpoint directory name encodes, or `None` if the name is
    /// not one of ours.
    ///
    /// At least eight ASCII digits are required, so a hand-made
    /// `checkpoint-188` or a stray `.publish-*` never becomes a resume
    /// candidate. A step past eight digits still round-trips because `{:08}`
    /// only pads.
    pub fn checkpoint_step(name: &str) -> Option<u64> {
        let digits = name.strip_prefix(CHECKPOINT_PREFIX)?;
        if digits.len() < STEP_DIGITS || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            return None;
        }
        digits.parse::<u64>().ok()
    }
}

/// Everything the command settled before the first optimizer step.
#[derive(Debug, Clone)]
pub struct TrainingPlan {
    pub mode: DomainTrainingMode,
    pub variant: UnetVariant,
    pub epochs_requested: u32,
    pub frame_count: u64,
    pub config: TrainingConfig,
    pub descriptor: CheckpointDescriptor,
    pub paths: TrainingPaths,
    /// The checkpoint this run resumes from, `None` for a fresh run. It is also
    /// what the result payload reports as `resumed_from`.
    pub resume_from: Option<PathBuf>,
}
