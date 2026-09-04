//! Drives a FeatherTalk training run: batches in, weights and telemetry out.

mod loss;
mod runner;
mod step;

pub use loss::LossValues;
pub use runner::{StepReport, TrainingRunner};
pub use step::{data_loader_config_for, train_single_frame_step, train_temporal_step};
