//! The FeatherTalk worker: a JSON Lines command server over stdin/stdout.
//!
//! This slice serves `validate_project`, `probe_media`, `normalize_media`,
//! `extract_frames`, `extract_features`, `lock_asset_package`, and `train` on
//! the CPU.
//! Every other command in [`feathertalk_domain::TaskKind`] is reported as
//! unsupported in the handshake and rejected if a client asks for it anyway.

mod adapters;
mod admission;
mod asset_scan;
mod commands;
mod config;
mod error_map;
mod extract_features;
mod extract_frames;
mod feature_result;
mod features;
mod handshake;
mod lock_asset_package;
mod lock_result;
mod models;
mod normalize_result;
mod probe_result;
mod quality_result;
mod reporter;
mod runtime;
mod train;
mod train_result;
mod training;

pub use adapters::{AdapterLockError, AdapterLocks};
pub use commands::{CommandOutcome, execute, execute_with_runner};
pub use config::{
    DEFAULT_MEDIA_TIMEOUT_MS, ENV_FFMPEG, ENV_FFPROBE, ENV_HUBERT_DIR, ENV_MEDIA_TIMEOUT_MS,
    ENV_PFLD_DIR, ENV_SCRFD_DIR, ENV_VGG19_DIR, FeatureToolchain, ModelToolchain,
    TrainingToolchain, WorkerConfig,
};
pub use error_map::{
    audio_task_error, is_audio_cancellation, is_inference_cancellation, is_media_cancellation,
    is_pipeline_cancellation, media_task_error, package_task_error, pipeline_task_error,
    project_task_error, quality_task_error, render_task_error, training_data_task_error,
    training_task_error,
};
pub use extract_features::execute_extract_features;
pub use extract_frames::execute_extract_frames;
pub use feature_result::feature_to_json;
pub use features::FeatureModel;
pub use handshake::{CPU_ADAPTER_ID, cpu_adapter, ready_frame, supported_commands};
pub use lock_asset_package::execute_lock_asset_package;
pub use lock_result::lock_to_json;
pub use models::FrameModels;
pub use normalize_result::normalize_to_json;
pub use probe_result::probe_to_json;
pub use quality_result::quality_to_json;
pub use reporter::{NoReporter, TaskReporter};
pub use runtime::{JobExecutor, serve, serve_with_executor};
pub use train::{check_frame_count, execute_train, run_training};
pub use train_result::{TrainSummary, train_to_json};
pub use training::{
    DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, MAX_EPOCHS, TRAIN_BACKEND_NAME, TRAINING_SEED,
    TrainBackend, TrainDevice, TrainingPaths, TrainingPlan, WORKER_STATE, checkpoint_descriptor,
    latest_checkpoint, preview_sample, publish_checkpoint, sample_count, training_config,
    training_mode, write_metrics_unless_present, write_preview_unless_present,
};
