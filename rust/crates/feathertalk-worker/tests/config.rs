use std::path::PathBuf;

use feathertalk_worker::{ENV_HUBERT_DIR, WorkerConfig};

fn absolute(name: &str) -> String {
    std::env::current_dir()
        .unwrap()
        .join(name)
        .display()
        .to_string()
}

fn with_hubert(hubert_dir: Option<String>) -> WorkerConfig {
    WorkerConfig::from_values_with_toolchains(None, None, None, None, None, hubert_dir)
}

#[test]
fn two_absolute_directories_resolve_the_model_toolchain() {
    let config = WorkerConfig::from_values_with_models(
        None,
        None,
        None,
        Some(absolute("scrfd_2_5g")),
        Some(absolute("pfld_ghost_one")),
    );

    let models = config.models().expect("both directories are absolute");
    assert_eq!(models.scrfd_dir(), PathBuf::from(absolute("scrfd_2_5g")));
    assert_eq!(models.pfld_dir(), PathBuf::from(absolute("pfld_ghost_one")));
    assert_eq!(config.model_rejection(), None);
}

#[test]
fn a_missing_pfld_directory_rejects_the_model_toolchain() {
    let config =
        WorkerConfig::from_values_with_models(None, None, None, Some(absolute("scrfd_2_5g")), None);

    assert!(config.models().is_none());
    let rejection = config.model_rejection().expect("a reason is kept");
    assert!(
        rejection.contains("FEATHERTALK_WORKER_PFLD_DIR"),
        "{rejection}"
    );
}

#[test]
fn a_relative_model_directory_is_rejected() {
    let config = WorkerConfig::from_values_with_models(
        None,
        None,
        None,
        Some("artifacts/scrfd_2_5g".to_owned()),
        Some(absolute("pfld_ghost_one")),
    );

    let rejection = config.model_rejection().expect("a reason is kept");
    assert!(
        rejection.contains("must be an absolute path"),
        "{rejection}"
    );
}

#[test]
fn the_two_toolchains_are_resolved_independently() {
    // A usable media toolchain must not imply usable models, and the
    // three-argument constructor must keep working for the media tests.
    let config = WorkerConfig::from_values(
        Some(absolute("ffprobe-test")),
        Some(absolute("ffmpeg-test")),
        None,
    );

    assert!(config.media().is_some());
    assert_eq!(config.media_rejection(), None);
    assert!(config.models().is_none());
    assert!(config.model_rejection().is_some());
}

#[test]
fn an_absolute_directory_resolves_the_feature_toolchain() {
    let hubert = absolute("feather_hubert_188");
    let config = with_hubert(Some(hubert.clone()));

    let features = config.features().expect("the directory is absolute");
    assert_eq!(features.hubert_dir(), PathBuf::from(&hubert));
    assert_eq!(config.feature_rejection(), None);
    // An unresolved media or model toolchain must not block the feature one:
    // extract_features shells out to nothing and loads neither SCRFD nor PFLD.
    assert!(config.media().is_none());
    assert!(config.models().is_none());
}

#[test]
fn a_relative_hubert_directory_is_rejected_with_the_variable_name() {
    assert_eq!(ENV_HUBERT_DIR, "FEATHERTALK_WORKER_HUBERT_DIR");
    let config = with_hubert(Some("models/hubert".to_owned()));

    assert!(config.features().is_none());
    let rejection = config.feature_rejection().expect("a reason is kept");
    assert!(rejection.contains(ENV_HUBERT_DIR), "{rejection}");
    assert!(
        rejection.contains("must be an absolute path"),
        "{rejection}"
    );
}

#[test]
fn the_feature_toolchain_is_resolved_independently_of_the_models() {
    // The five-argument constructor keeps its meaning: it configures media and
    // models and leaves extract_features unsupported.
    let config = WorkerConfig::from_values_with_models(
        Some(absolute("ffprobe-test")),
        Some(absolute("ffmpeg-test")),
        None,
        Some(absolute("scrfd_2_5g")),
        Some(absolute("pfld_ghost_one")),
    );

    assert!(config.media().is_some());
    assert!(config.models().is_some());
    assert!(config.features().is_none());
    let rejection = config.feature_rejection().expect("a reason is kept");
    assert!(rejection.contains("is not set"), "{rejection}");
}
