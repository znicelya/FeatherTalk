use std::{path::PathBuf, time::Duration};

use feathertalk_media::MediaToolchain;

pub const ENV_FFPROBE: &str = "FEATHERTALK_WORKER_FFPROBE";
pub const ENV_FFMPEG: &str = "FEATHERTALK_WORKER_FFMPEG";
pub const ENV_MEDIA_TIMEOUT_MS: &str = "FEATHERTALK_WORKER_MEDIA_TIMEOUT_MS";
pub const DEFAULT_MEDIA_TIMEOUT_MS: u64 = 300_000;

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
}

impl WorkerConfig {
    pub fn from_env() -> Self {
        Self::from_values(
            std::env::var(ENV_FFPROBE).ok(),
            std::env::var(ENV_FFMPEG).ok(),
            std::env::var(ENV_MEDIA_TIMEOUT_MS).ok(),
        )
    }

    pub fn from_values(
        ffprobe: Option<String>,
        ffmpeg: Option<String>,
        timeout_ms: Option<String>,
    ) -> Self {
        let (media, media_rejection) = match media_toolchain(ffprobe, ffmpeg, timeout_ms) {
            Ok(toolchain) => (Some(toolchain), None),
            Err(reason) => (None, Some(reason)),
        };
        Self {
            worker_version: env!("CARGO_PKG_VERSION").to_owned(),
            media,
            media_rejection,
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
