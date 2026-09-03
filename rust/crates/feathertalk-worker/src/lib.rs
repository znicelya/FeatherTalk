//! The FeatherTalk worker: a JSON Lines command server over stdin/stdout.
//!
//! This slice serves `validate_project`, `probe_media`, `normalize_media`, and
//! `extract_frames` on the CPU. Every other command in
//! [`feathertalk_domain::TaskKind`] is reported as unsupported in the handshake
//! and rejected if a client asks for it anyway.

mod adapters;
mod commands;
mod config;
mod error_map;
mod extract_frames;
mod handshake;
mod models;
mod normalize_result;
mod probe_result;
mod quality_result;
mod reporter;
mod runtime;

pub use adapters::{AdapterLockError, AdapterLocks};
pub use commands::{CommandOutcome, execute, execute_with_runner};
pub use config::{
    DEFAULT_MEDIA_TIMEOUT_MS, ENV_FFMPEG, ENV_FFPROBE, ENV_HUBERT_DIR, ENV_MEDIA_TIMEOUT_MS,
    ENV_PFLD_DIR, ENV_SCRFD_DIR, FeatureToolchain, ModelToolchain, WorkerConfig,
};
pub use error_map::{
    is_media_cancellation, is_pipeline_cancellation, media_task_error, pipeline_task_error,
    project_task_error, quality_task_error,
};
pub use extract_frames::execute_extract_frames;
pub use handshake::{CPU_ADAPTER_ID, cpu_adapter, ready_frame, supported_commands};
pub use models::FrameModels;
pub use normalize_result::normalize_to_json;
pub use probe_result::probe_to_json;
pub use quality_result::quality_to_json;
pub use reporter::{NoReporter, TaskReporter};
pub use runtime::{JobExecutor, serve, serve_with_executor};
