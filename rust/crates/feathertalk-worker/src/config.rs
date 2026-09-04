use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use feathertalk_media::MediaToolchain;

pub const ENV_FFPROBE: &str = "FEATHERTALK_WORKER_FFPROBE";
pub const ENV_FFMPEG: &str = "FEATHERTALK_WORKER_FFMPEG";
pub const ENV_MEDIA_TIMEOUT_MS: &str = "FEATHERTALK_WORKER_MEDIA_TIMEOUT_MS";
pub const DEFAULT_MEDIA_TIMEOUT_MS: u64 = 300_000;
pub const ENV_SCRFD_DIR: &str = "FEATHERTALK_WORKER_SCRFD_DIR";
pub const ENV_PFLD_DIR: &str = "FEATHERTALK_WORKER_PFLD_DIR";
pub const ENV_HUBERT_DIR: &str = "FEATHERTALK_WORKER_HUBERT_DIR";
pub const ENV_VGG19_DIR: &str = "FEATHERTALK_WORKER_VGG19_DIR";

/// Where the worker finds the two model artifact directories.
///
/// Only the shape of the paths is checked here. Whether the directories hold a
/// loadable manifest and weights is discovered when the first job loads them,
/// because a directory can disappear between startup and the first job.
#[derive(Debug, Clone)]
pub struct ModelToolchain {
    scrfd_dir: PathBuf,
    pfld_dir: PathBuf,
}

impl ModelToolchain {
    pub fn scrfd_dir(&self) -> &Path {
        &self.scrfd_dir
    }

    pub fn pfld_dir(&self) -> &Path {
        &self.pfld_dir
    }
}

/// Where the worker finds the FeatherHuBERT model package.
///
/// Only the shape of the path is checked here, for the same reason as
/// `ModelToolchain`: a directory can disappear between startup and the first
/// job, so the manifest and the weights are validated when a job loads them.
#[derive(Debug, Clone)]
pub struct FeatureToolchain {
    hubert_dir: PathBuf,
}

impl FeatureToolchain {
    pub fn hubert_dir(&self) -> &Path {
        &self.hubert_dir
    }
}

/// Where the worker finds the VGG19 perceptual-loss package.
///
/// Only the shape of the path is checked here, for the same reason as
/// `FeatureToolchain`: the manifest, the licence bundle and the safetensors
/// weights are validated when a training job loads them, because a directory
/// can disappear between startup and the first job.
#[derive(Debug, Clone)]
pub struct TrainingToolchain {
    vgg19_dir: PathBuf,
}

impl TrainingToolchain {
    pub fn vgg19_dir(&self) -> &Path {
        &self.vgg19_dir
    }
}

/// Everything the worker learns from its environment at startup.
///
/// A missing or unusable media toolchain is not a startup failure: the worker
/// still serves `validate_project` and simply reports `probe_media` as
/// unsupported, with the reason kept for the rejection message.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    worker_version: String,
    media: Option<MediaToolchain>,
    media_rejection: Option<String>,
    models: Option<ModelToolchain>,
    model_rejection: Option<String>,
    features: Option<FeatureToolchain>,
    feature_rejection: Option<String>,
    training: Option<TrainingToolchain>,
    training_rejection: Option<String>,
}

impl WorkerConfig {
    pub fn from_env() -> Self {
        Self::from_values_with_training(
            std::env::var(ENV_FFPROBE).ok(),
            std::env::var(ENV_FFMPEG).ok(),
            std::env::var(ENV_MEDIA_TIMEOUT_MS).ok(),
            std::env::var(ENV_SCRFD_DIR).ok(),
            std::env::var(ENV_PFLD_DIR).ok(),
            std::env::var(ENV_HUBERT_DIR).ok(),
            std::env::var(ENV_VGG19_DIR).ok(),
        )
    }

    /// The media-only form: no model directories, so `extract_frames` stays
    /// unsupported.
    pub fn from_values(
        ffprobe: Option<String>,
        ffmpeg: Option<String>,
        timeout_ms: Option<String>,
    ) -> Self {
        Self::from_values_with_models(ffprobe, ffmpeg, timeout_ms, None, None)
    }

    /// The frame form: no FeatherHuBERT directory, so `extract_features` stays
    /// unsupported.
    pub fn from_values_with_models(
        ffprobe: Option<String>,
        ffmpeg: Option<String>,
        timeout_ms: Option<String>,
        scrfd_dir: Option<String>,
        pfld_dir: Option<String>,
    ) -> Self {
        Self::from_values_with_toolchains(ffprobe, ffmpeg, timeout_ms, scrfd_dir, pfld_dir, None)
    }

    /// The toolchain form: no VGG19 directory, so `train` stays unsupported.
    pub fn from_values_with_toolchains(
        ffprobe: Option<String>,
        ffmpeg: Option<String>,
        timeout_ms: Option<String>,
        scrfd_dir: Option<String>,
        pfld_dir: Option<String>,
        hubert_dir: Option<String>,
    ) -> Self {
        Self::from_values_with_training(
            ffprobe, ffmpeg, timeout_ms, scrfd_dir, pfld_dir, hubert_dir, None,
        )
    }

    /// The training form: the VGG19 package the perceptual loss reads. Training
    /// needs no media tools and no frame models, so this is orthogonal to every
    /// other toolchain.
    pub fn from_values_with_training(
        ffprobe: Option<String>,
        ffmpeg: Option<String>,
        timeout_ms: Option<String>,
        scrfd_dir: Option<String>,
        pfld_dir: Option<String>,
        hubert_dir: Option<String>,
        vgg19_dir: Option<String>,
    ) -> Self {
        let (media, media_rejection) = match media_toolchain(ffprobe, ffmpeg, timeout_ms) {
            Ok(toolchain) => (Some(toolchain), None),
            Err(reason) => (None, Some(reason)),
        };
        let (models, model_rejection) = match model_toolchain(scrfd_dir, pfld_dir) {
            Ok(toolchain) => (Some(toolchain), None),
            Err(reason) => (None, Some(reason)),
        };
        let (features, feature_rejection) = match feature_toolchain(hubert_dir) {
            Ok(toolchain) => (Some(toolchain), None),
            Err(reason) => (None, Some(reason)),
        };
        let (training, training_rejection) = match training_toolchain(vgg19_dir) {
            Ok(toolchain) => (Some(toolchain), None),
            Err(reason) => (None, Some(reason)),
        };
        Self {
            worker_version: env!("CARGO_PKG_VERSION").to_owned(),
            media,
            media_rejection,
            models,
            model_rejection,
            features,
            feature_rejection,
            training,
            training_rejection,
        }
    }

    pub fn worker_version(&self) -> &str {
        &self.worker_version
    }

    pub fn media(&self) -> Option<&MediaToolchain> {
        self.media.as_ref()
    }

    pub fn media_rejection(&self) -> Option<&str> {
        self.media_rejection.as_deref()
    }

    pub fn models(&self) -> Option<&ModelToolchain> {
        self.models.as_ref()
    }

    pub fn model_rejection(&self) -> Option<&str> {
        self.model_rejection.as_deref()
    }

    pub fn features(&self) -> Option<&FeatureToolchain> {
        self.features.as_ref()
    }

    pub fn feature_rejection(&self) -> Option<&str> {
        self.feature_rejection.as_deref()
    }

    pub fn training(&self) -> Option<&TrainingToolchain> {
        self.training.as_ref()
    }

    pub fn training_rejection(&self) -> Option<&str> {
        self.training_rejection.as_deref()
    }
}

fn media_toolchain(
    ffprobe: Option<String>,
    ffmpeg: Option<String>,
    timeout_ms: Option<String>,
) -> Result<MediaToolchain, String> {
    let ffprobe = required_path(ffprobe, ENV_FFPROBE)?;
    let ffmpeg = required_path(ffmpeg, ENV_FFMPEG)?;
    let timeout_ms = match timeout_ms {
        None => DEFAULT_MEDIA_TIMEOUT_MS,
        Some(value) => value.trim().parse::<u64>().map_err(|_| {
            format!("{ENV_MEDIA_TIMEOUT_MS} must be a whole number of milliseconds, got {value:?}")
        })?,
    };
    if timeout_ms == 0 {
        return Err(format!("{ENV_MEDIA_TIMEOUT_MS} must be greater than zero"));
    }
    MediaToolchain::new(ffmpeg, ffprobe, Duration::from_millis(timeout_ms))
        .map_err(|error| error.to_string())
}

fn model_toolchain(
    scrfd_dir: Option<String>,
    pfld_dir: Option<String>,
) -> Result<ModelToolchain, String> {
    let scrfd_dir = required_path(scrfd_dir, ENV_SCRFD_DIR)?;
    let pfld_dir = required_path(pfld_dir, ENV_PFLD_DIR)?;
    Ok(ModelToolchain {
        scrfd_dir,
        pfld_dir,
    })
}

fn feature_toolchain(hubert_dir: Option<String>) -> Result<FeatureToolchain, String> {
    let hubert_dir = required_path(hubert_dir, ENV_HUBERT_DIR)?;
    Ok(FeatureToolchain { hubert_dir })
}

fn training_toolchain(vgg19_dir: Option<String>) -> Result<TrainingToolchain, String> {
    let vgg19_dir = required_path(vgg19_dir, ENV_VGG19_DIR)?;
    Ok(TrainingToolchain { vgg19_dir })
}

fn required_path(value: Option<String>, variable: &str) -> Result<PathBuf, String> {
    let value = value.ok_or_else(|| format!("{variable} is not set"))?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(format!("{variable} must not be empty"));
    }
    let path = PathBuf::from(trimmed);
    if !path.is_absolute() {
        return Err(format!(
            "{variable} must be an absolute path, got {trimmed:?}"
        ));
    }
    Ok(path)
}
