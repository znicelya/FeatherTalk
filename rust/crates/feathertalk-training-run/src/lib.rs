//! Drives a FeatherTalk training run: batches in, weights and telemetry out.

mod loss;
mod step;

pub use loss::LossValues;
pub use step::{data_loader_config_for, train_single_frame_step};
