use std::{fs, path::PathBuf};

use feathertalk_domain::{ImportLegacyModelParams, LegacyModelKind, TaskStage};
use feathertalk_media::CancellationToken;
use feathertalk_worker::{NoReporter, WorkerConfig, execute_import_legacy_model};

fn params(source: PathBuf, kind: LegacyModelKind, destination: PathBuf) -> ImportLegacyModelParams {
    ImportLegacyModelParams {
        source,
        kind,
        destination,
    }
}

#[test]
fn import_rejects_relative_source_before_reading_it() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("package");
    let error = execute_import_legacy_model(
        &params(
            PathBuf::from("model.pth"),
            LegacyModelKind::FeatherHubert,
            destination,
        ),
        &WorkerConfig::from_values(None, None, None),
        &CancellationToken::new(),
        &NoReporter,
    )
    .unwrap_err();
    assert!(error.to_string().contains("absolute"));
    assert!(!root.path().join("package").exists());
}

#[test]
fn import_rejects_unsupported_kind_without_creating_destination() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("model.pth");
    fs::write(&source, b"not-read").unwrap();
    let destination = root.path().join("package");
    let error = execute_import_legacy_model(
        &params(source, LegacyModelKind::Pfld, destination.clone()),
        &WorkerConfig::from_values(None, None, None),
        &CancellationToken::new(),
        &NoReporter,
    )
    .unwrap_err();
    assert!(error.to_string().contains("not supported"));
    assert!(!destination.exists());
}

#[test]
fn import_honours_cancellation_before_import() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("model.pth");
    fs::write(&source, b"not-read").unwrap();
    let token = CancellationToken::new();
    token.cancel();
    let error = execute_import_legacy_model(
        &params(
            source,
            LegacyModelKind::FeatherHubert,
            root.path().join("package"),
        ),
        &WorkerConfig::from_values(None, None, None),
        &token,
        &NoReporter,
    )
    .unwrap_err();
    assert!(matches!(error.stage(), TaskStage::Preparing));
    assert!(error.is_cancelled());
}
