use std::path::PathBuf;

use feathertalk_worker::WorkerConfig;

fn absolute(name: &str) -> String {
    std::env::current_dir()
        .unwrap()
        .join(name)
        .display()
        .to_string()
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
