use feathertalk_domain::{
    AdapterInfo, AdapterKind, Backend, Capabilities, PROTOCOL_VERSION, ReadyFrame, TaskKind,
};

use crate::WorkerConfig;

/// Stable identity of the single CPU adapter this slice exposes. The adapter
/// lock table keys on it, so it must not change between worker restarts.
pub const CPU_ADAPTER_ID: &str = "cpu-0";

pub fn cpu_adapter() -> AdapterInfo {
    AdapterInfo {
        id: CPU_ADAPTER_ID.to_owned(),
        name: "CPU".to_owned(),
        backend: Backend::Cpu,
        kind: AdapterKind::Cpu,
        certified: true,
        vram_bytes: None,
    }
}

pub fn supported_commands(config: &WorkerConfig) -> Vec<TaskKind> {
    let mut commands = vec![TaskKind::ValidateProject];
    if config.media().is_some() {
        commands.push(TaskKind::ProbeMedia);
    }
    commands
}

pub fn ready_frame(config: &WorkerConfig) -> ReadyFrame {
    ReadyFrame {
        protocol_version: PROTOCOL_VERSION,
        worker_version: config.worker_version().to_owned(),
        backends: vec![Backend::Cpu],
        adapters: vec![cpu_adapter()],
        supported_commands: supported_commands(config),
        capabilities: Capabilities {
            training: false,
            wgpu_training: false,
            onnx_validation: false,
            ffmpeg: config.media().is_some(),
        },
    }
}
