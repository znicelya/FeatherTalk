use std::time::Duration;

use feathertalk_domain::{AdapterKind, Backend, TaskKind};
use feathertalk_worker::{
    CPU_ADAPTER_ID, DEFAULT_MEDIA_TIMEOUT_MS, ENV_FFPROBE, ENV_MEDIA_TIMEOUT_MS, WorkerConfig,
    ready_frame, supported_commands,
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
