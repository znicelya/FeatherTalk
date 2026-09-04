use std::{
    fs,
    path::{Path, PathBuf},
};

use feathertalk_domain::{ErrorCode, TaskStage};
use feathertalk_worker::{ModelSourceKind, model_source_kind};

/// Classification never opens a file, so every fixture is a one-byte placeholder.
fn touch(path: &Path) {
    fs::write(path, b"x").expect("the placeholder is written");
}

fn directory(root: &Path, name: &str) -> PathBuf {
    let dir = root.join(name);
    fs::create_dir_all(&dir).expect("the directory is created");
    dir
}

#[test]
fn a_directory_with_a_safetensors_model_is_a_package() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let dir = directory(root.path(), "package");
    touch(&dir.join("model.safetensors"));
    touch(&dir.join("manifest.json"));
    assert_eq!(
        model_source_kind(&dir).expect("the layout is recognized"),
        ModelSourceKind::ModelPackage
    );
}

#[test]
fn a_directory_with_a_binary_model_is_a_checkpoint() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let dir = directory(root.path(), "checkpoint");
    touch(&dir.join("model.bin"));
    touch(&dir.join("manifest.json"));
    assert_eq!(
        model_source_kind(&dir).expect("the layout is recognized"),
        ModelSourceKind::TrainingCheckpoint
    );
}

#[test]
fn a_directory_holding_both_model_files_is_refused() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let dir = directory(root.path(), "both");
    touch(&dir.join("model.safetensors"));
    touch(&dir.join("model.bin"));
    let error = model_source_kind(&dir).expect_err("two layouts at once is no layout");
    assert_eq!(error.code, ErrorCode::ModelIncompatible);
    assert_eq!(error.summary, "无法识别的模型目录");
    assert_eq!(error.stage, TaskStage::Preparing);
}

#[test]
fn a_directory_holding_neither_model_file_is_refused() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let dir = directory(root.path(), "empty");
    touch(&dir.join("manifest.json"));
    let error = model_source_kind(&dir).expect_err("a manifest alone is no layout");
    assert_eq!(error.code, ErrorCode::ModelIncompatible);
}

#[test]
fn a_relative_source_is_refused_before_any_probe() {
    let error = model_source_kind(Path::new("model")).expect_err("a relative source is refused");
    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.summary, "模型目录必须是绝对路径");
}

#[test]
fn a_file_instead_of_a_directory_is_refused() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let file = root.path().join("model.safetensors");
    touch(&file);
    let error = model_source_kind(&file).expect_err("a file is not a model directory");
    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.summary, "模型目录不可用");
}

#[test]
fn a_missing_source_is_refused() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let error =
        model_source_kind(&root.path().join("absent")).expect_err("a missing source is refused");
    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.summary, "模型目录不可用");
}

#[test]
fn both_kinds_have_a_stable_slug() {
    assert_eq!(ModelSourceKind::ModelPackage.as_slug(), "model_package");
    assert_eq!(
        ModelSourceKind::TrainingCheckpoint.as_slug(),
        "training_checkpoint"
    );
}
