use std::time::Duration;

use feathertalk_domain::{AdapterKind, Backend, TaskKind};
use feathertalk_worker::{
    CPU_ADAPTER_ID, DEFAULT_MEDIA_TIMEOUT_MS, ENV_FFPROBE, ENV_HUBERT_DIR, ENV_MEDIA_TIMEOUT_MS,
    ENV_SCRFD_DIR, WorkerConfig, ready_frame, supported_commands,
};

fn absolute(name: &str) -> String {
    std::env::current_dir()
        .unwrap()
        .join(name)
        .display()
        .to_string()
}

fn configured() -> WorkerConfig {
    WorkerConfig::from_values(
        Some(absolute("ffprobe-test")),
        Some(absolute("ffmpeg-test")),
        None,
    )
}

#[test]
fn a_configured_worker_reports_a_cpu_adapter_and_both_commands() {
    let config = configured();
    assert_eq!(config.media_rejection(), None);
    let frame = ready_frame(&config);
    frame.validate().unwrap();
    assert_eq!(frame.worker_version, env!("CARGO_PKG_VERSION"));
    assert_eq!(frame.backends, vec![Backend::Cpu]);
    assert_eq!(frame.adapters.len(), 1);
    assert_eq!(frame.adapters[0].id, CPU_ADAPTER_ID);
    assert_eq!(frame.adapters[0].backend, Backend::Cpu);
    assert_eq!(frame.adapters[0].kind, AdapterKind::Cpu);
    assert!(frame.adapters[0].certified);
    assert_eq!(frame.adapters[0].vram_bytes, None);
    assert_eq!(
        frame.supported_commands,
        vec![
            TaskKind::ValidateProject,
            TaskKind::ProbeMedia,
            TaskKind::NormalizeMedia
        ]
    );
    assert!(frame.capabilities.ffmpeg);
    assert!(!frame.capabilities.training);
    assert!(!frame.capabilities.wgpu_training);
    assert!(!frame.capabilities.onnx_validation);
}

#[test]
fn a_worker_without_a_media_toolchain_only_offers_project_validation() {
    let config = WorkerConfig::from_values(None, None, None);
    assert!(config.media().is_none());
    assert!(
        config
            .media_rejection()
            .is_some_and(|reason| reason.contains(ENV_FFPROBE))
    );
    let frame = ready_frame(&config);
    frame.validate().unwrap();
    assert_eq!(frame.supported_commands, vec![TaskKind::ValidateProject]);
    assert!(!frame.capabilities.ffmpeg);
    assert_eq!(supported_commands(&config).len(), 1);
}

#[test]
fn a_relative_tool_path_is_rejected_with_the_variable_name() {
    let config = WorkerConfig::from_values(
        Some("ffprobe.exe".to_owned()),
        Some(absolute("ffmpeg-test")),
        None,
    );
    let reason = config.media_rejection().expect("relative path must reject");
    assert!(reason.contains(ENV_FFPROBE), "{reason}");
    assert!(reason.contains("absolute"), "{reason}");
}

#[test]
fn an_empty_tool_path_is_rejected() {
    let config =
        WorkerConfig::from_values(Some("   ".to_owned()), Some(absolute("ffmpeg-test")), None);
    assert!(
        config
            .media_rejection()
            .is_some_and(|reason| reason.contains(ENV_FFPROBE))
    );
}

#[test]
fn an_unusable_timeout_is_rejected_with_the_variable_name() {
    for bad in ["0", "abc", "-1", ""] {
        let config = WorkerConfig::from_values(
            Some(absolute("ffprobe-test")),
            Some(absolute("ffmpeg-test")),
            Some(bad.to_owned()),
        );
        let reason = config
            .media_rejection()
            .unwrap_or_else(|| panic!("expected rejection for {bad:?}"));
        assert!(reason.contains(ENV_MEDIA_TIMEOUT_MS), "{bad:?}: {reason}");
    }
}

#[test]
fn the_default_media_timeout_is_five_minutes() {
    assert_eq!(DEFAULT_MEDIA_TIMEOUT_MS, 300_000);
    let config = configured();
    assert_eq!(
        config.media().unwrap().timeout(),
        Duration::from_millis(DEFAULT_MEDIA_TIMEOUT_MS)
    );

    let explicit = WorkerConfig::from_values(
        Some(absolute("ffprobe-test")),
        Some(absolute("ffmpeg-test")),
        Some("1500".to_owned()),
    );
    assert_eq!(
        explicit.media().unwrap().timeout(),
        Duration::from_millis(1500)
    );
}

/// Media and models both resolve, so every command in this slice is offered.
fn fully_configured() -> WorkerConfig {
    WorkerConfig::from_values_with_models(
        Some(absolute("ffprobe-test")),
        Some(absolute("ffmpeg-test")),
        None,
        Some(absolute("scrfd-test")),
        Some(absolute("pfld-test")),
    )
}

#[test]
fn a_fully_configured_worker_offers_extract_frames() {
    let config = fully_configured();
    assert_eq!(config.model_rejection(), None);
    let frame = ready_frame(&config);
    frame.validate().unwrap();
    assert_eq!(
        frame.supported_commands,
        vec![
            TaskKind::ValidateProject,
            TaskKind::ProbeMedia,
            TaskKind::NormalizeMedia,
            TaskKind::ExtractFrames
        ]
    );
    // Protocol version 2 has no model capability flag, so nothing here moves.
    assert!(frame.capabilities.ffmpeg);
    assert!(!frame.capabilities.training);
}

#[test]
fn a_media_only_worker_leaves_extract_frames_out() {
    let config = configured();
    assert!(config.models().is_none());
    assert!(
        config
            .model_rejection()
            .is_some_and(|reason| reason.contains(ENV_SCRFD_DIR))
    );
    assert!(!supported_commands(&config).contains(&TaskKind::ExtractFrames));
}

#[test]
fn models_without_a_media_toolchain_offer_nothing_new() {
    let config = WorkerConfig::from_values_with_models(
        None,
        None,
        None,
        Some(absolute("scrfd-test")),
        Some(absolute("pfld-test")),
    );
    assert!(config.models().is_some());
    assert_eq!(supported_commands(&config), vec![TaskKind::ValidateProject]);
}

/// Media, models, and the FeatherHuBERT directory all resolve, so the handshake
/// offers every command in this slice.
fn every_toolchain() -> WorkerConfig {
    WorkerConfig::from_values_with_toolchains(
        Some(absolute("ffprobe-test")),
        Some(absolute("ffmpeg-test")),
        None,
        Some(absolute("scrfd-test")),
        Some(absolute("pfld-test")),
        Some(absolute("hubert-test")),
    )
}

#[test]
fn a_worker_with_a_feature_model_offers_extract_features() {
    let config = every_toolchain();
    assert_eq!(config.feature_rejection(), None);
    let frame = ready_frame(&config);
    frame.validate().unwrap();
    assert_eq!(
        frame.supported_commands,
        vec![
            TaskKind::ValidateProject,
            TaskKind::ProbeMedia,
            TaskKind::NormalizeMedia,
            TaskKind::ExtractFrames,
            TaskKind::ExtractFeatures,
            TaskKind::LockAssetPackage
        ]
    );
}

#[test]
fn a_worker_without_a_feature_model_leaves_extract_features_out() {
    let config = fully_configured();
    assert!(config.features().is_none());
    assert!(
        config
            .feature_rejection()
            .is_some_and(|reason| reason.contains(ENV_HUBERT_DIR))
    );
    assert!(!supported_commands(&config).contains(&TaskKind::ExtractFeatures));
}

#[test]
fn a_feature_model_without_a_media_toolchain_still_offers_extract_features() {
    let config = WorkerConfig::from_values_with_toolchains(
        None,
        None,
        None,
        None,
        None,
        Some(absolute("hubert-test")),
    );
    assert!(config.media().is_none());
    assert_eq!(
        supported_commands(&config),
        vec![
            TaskKind::ValidateProject,
            TaskKind::ExtractFeatures,
            TaskKind::LockAssetPackage
        ]
    );
}
