//! The FeatherTalk worker: a JSON Lines command server over stdin/stdout.
//!
//! This slice serves `validate_project` and `probe_media` on the CPU. Every
//! other command in [`feathertalk_domain::TaskKind`] is reported as unsupported
//! in the handshake and rejected if a client asks for it anyway.

mod config;
mod handshake;

pub use config::{
    DEFAULT_MEDIA_TIMEOUT_MS, ENV_FFMPEG, ENV_FFPROBE, ENV_MEDIA_TIMEOUT_MS, WorkerConfig,
};
pub use handshake::{CPU_ADAPTER_ID, cpu_adapter, ready_frame, supported_commands};
