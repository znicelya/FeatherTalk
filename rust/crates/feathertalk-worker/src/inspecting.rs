//! What a model directory is, and whether this build can use it.

use std::{fs, path::Path};

use feathertalk_domain::{ErrorCode, TaskError, TaskStage};
use feathertalk_export::MODEL_FILE_NAME;
use feathertalk_training::CHECKPOINT_MODEL_FILE_NAME;

use crate::{admission::check_model_source, error_map::clamp};

/// The two directory layouts `inspect_model` accepts (design section 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSourceKind {
    ModelPackage,
    TrainingCheckpoint,
}

impl ModelSourceKind {
    /// The wire spelling. It goes into the payload's `source_kind`, so it is
    /// part of the protocol and must not drift.
    pub fn as_slug(self) -> &'static str {
        match self {
            Self::ModelPackage => "model_package",
            Self::TrainingCheckpoint => "training_checkpoint",
        }
    }
}

/// Decide the layout from the one file only that layout has. An exported package
/// carries `model.safetensors`; a checkpoint carries `model.bin`. Neither file is
/// opened -- the digests in the manifests are the source of truth for content,
/// and reading weights is out of scope (design section 3).
pub fn model_source_kind(source: &Path) -> Result<ModelSourceKind, TaskError> {
    check_model_source(source)?;
    let package = is_regular_file(&source.join(MODEL_FILE_NAME));
    let checkpoint = is_regular_file(&source.join(CHECKPOINT_MODEL_FILE_NAME));
    match (package, checkpoint) {
        (true, false) => Ok(ModelSourceKind::ModelPackage),
        (false, true) => Ok(ModelSourceKind::TrainingCheckpoint),
        // Both means the directory is two things at once and neither reader
        // would be right; neither means it is not a model directory at all.
        _ => Err(unrecognized(source)),
    }
}

fn is_regular_file(path: &Path) -> bool {
    fs::symlink_metadata(path)
        .map(|metadata| metadata.is_file())
        .unwrap_or(false)
}

fn unrecognized(source: &Path) -> TaskError {
    TaskError::new(
        ErrorCode::ModelIncompatible,
        "无法识别的模型目录",
        &clamp(&format!(
            "{} holds neither exactly one {MODEL_FILE_NAME} nor exactly one \
             {CHECKPOINT_MODEL_FILE_NAME}",
            source.display()
        )),
        TaskStage::Preparing,
    )
}
