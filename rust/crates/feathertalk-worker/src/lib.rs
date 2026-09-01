//! The FeatherTalk worker: a JSON Lines command server over stdin/stdout.
//!
//! This slice serves `validate_project` and `probe_media` on the CPU. Every
//! other command in [`feathertalk_domain::TaskKind`] is reported as unsupported
//! in the handshake and rejected if a client asks for it anyway.

mod adapters;
mod commands;
mod config;
mod error_map;
mod handshake;
mod probe_result;
mod runtime;

pub use adapters::{AdapterLockError, AdapterLocks};
pub use commands::{CommandOutcome, execute, execute_with_runner};
pub use config::{
    DEFAULT_MEDIA_TIMEOUT_MS, ENV_FFMPEG, ENV_FFPROBE, ENV_MEDIA_TIMEOUT_MS, WorkerConfig,
};
pub use error_map::{is_media_cancellation, media_task_error, project_task_error};
pub use handshake::{CPU_ADAPTER_ID, cpu_adapter, ready_frame, supported_commands};
pub use probe_result::probe_to_json;
pub use runtime::{JobExecutor, serve, serve_with_executor};
