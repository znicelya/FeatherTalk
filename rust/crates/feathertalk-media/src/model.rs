use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaInput {
    pub source: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizationSpec {
    pub target_video_fps: u32,
    pub target_audio_sample_rate: u32,
    pub target_audio_channels: u16,
    pub output_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedInput {
    source: PathBuf,
}

impl ValidatedInput {
    pub(crate) fn new(source: PathBuf) -> Self {
        Self { source }
    }

    pub fn source(&self) -> &Path {
        &self.source
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedMediaLayout {
    output_dir: PathBuf,
    video_path: PathBuf,
    audio_path: PathBuf,
}

impl NormalizedMediaLayout {
    pub(crate) fn new(output_dir: PathBuf, video_path: PathBuf, audio_path: PathBuf) -> Self {
        Self {
            output_dir,
            video_path,
            audio_path,
        }
    }

    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }
    pub fn video_path(&self) -> &Path {
        &self.video_path
    }
    pub fn audio_path(&self) -> &Path {
        &self.audio_path
    }
}
