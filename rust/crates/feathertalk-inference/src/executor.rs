use std::path::{Path, PathBuf};

use crate::{InferenceError, staging_output_path};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OfflineRenderRequest {
    frame_dir: PathBuf,
    landmark_dir: PathBuf,
    feature_path: PathBuf,
    audio_path: PathBuf,
    ffmpeg_path: PathBuf,
    output_path: PathBuf,
    task_id: String,
    source_frame_count: usize,
    max_output_frames: Option<usize>,
}

impl OfflineRenderRequest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frame_dir: PathBuf,
        landmark_dir: PathBuf,
        feature_path: PathBuf,
        audio_path: PathBuf,
        ffmpeg_path: PathBuf,
        output_path: PathBuf,
        task_id: impl Into<String>,
        source_frame_count: usize,
        max_output_frames: Option<usize>,
    ) -> Result<Self, InferenceError> {
        let task_id = task_id.into();
        for (field, path) in [
            ("frame_dir", &frame_dir),
            ("landmark_dir", &landmark_dir),
            ("feature_path", &feature_path),
            ("audio_path", &audio_path),
            ("ffmpeg_path", &ffmpeg_path),
            ("output_path", &output_path),
        ] {
            validate_absolute_non_empty(field, path)?;
        }
        if source_frame_count < 2 {
            return Err(InferenceError::FrameCountTooSmall {
                actual: source_frame_count,
                minimum: 2,
            });
        }
        if max_output_frames == Some(0) {
            return Err(InferenceError::InvalidField {
                field: "max_output_frames",
                message: "must be greater than zero when provided".into(),
            });
        }
        // Reuse the established destination and task-id contract without creating a file.
        staging_output_path(&output_path, &task_id)?;
        Ok(Self {
            frame_dir,
            landmark_dir,
            feature_path,
            audio_path,
            ffmpeg_path,
            output_path,
            task_id,
            source_frame_count,
            max_output_frames,
        })
    }

    pub fn frame_dir(&self) -> &Path {
        &self.frame_dir
    }

    pub fn landmark_dir(&self) -> &Path {
        &self.landmark_dir
    }

    pub fn feature_path(&self) -> &Path {
        &self.feature_path
    }

    pub fn audio_path(&self) -> &Path {
        &self.audio_path
    }

    pub fn ffmpeg_path(&self) -> &Path {
        &self.ffmpeg_path
    }

    pub fn output_path(&self) -> &Path {
        &self.output_path
    }

    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    pub fn source_frame_count(&self) -> usize {
        self.source_frame_count
    }

    pub fn max_output_frames(&self) -> Option<usize> {
        self.max_output_frames
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OfflineRenderResult {
    output_path: PathBuf,
    frame_count: usize,
    width: u32,
    height: u32,
}

impl OfflineRenderResult {
    pub fn output_path(&self) -> &Path {
        &self.output_path
    }

    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}

fn validate_absolute_non_empty(field: &'static str, path: &Path) -> Result<(), InferenceError> {
    if path.as_os_str().is_empty() || !path.is_absolute() {
        return Err(InferenceError::InvalidField {
            field,
            message: "must be a non-empty absolute path".into(),
        });
    }
    Ok(())
}
