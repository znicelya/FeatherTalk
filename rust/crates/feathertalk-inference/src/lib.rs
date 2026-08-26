mod command;
mod error;
mod frame;
mod plan;
mod render;
mod sequence;

pub use command::{CommandSpec, raw_video_command};
pub use error::InferenceError;
pub use frame::{BgrFrame, crop_bgr, resize_bilinear};
pub use plan::{InferenceFramePlan, RenderPlan};
pub use render::{
    RawFrameRenderSpec, RenderGeometry, staging_output_path, validate_output_destination,
};
pub use sequence::PingPongFrames;
