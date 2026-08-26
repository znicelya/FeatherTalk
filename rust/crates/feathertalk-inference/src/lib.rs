mod burn;
mod command;
mod error;
mod executor;
mod frame;
mod frame_reader;
mod plan;
mod render;
mod sequence;

pub use burn::{UnetAudioInput, build_unet_audio_input, render_planned_frame, run_unet_prediction};
pub use command::{CommandSpec, raw_video_command};
pub use error::InferenceError;
pub use executor::{OfflineRenderRequest, OfflineRenderResult};
pub use frame::{
    BgrFrame, UnetImageInput, apply_unet_prediction, build_unet_image_input, crop_bgr, paste_bgr,
    render_frame, resize_bilinear,
};
pub use frame_reader::{DEFAULT_MAX_FRAME_PIXELS, FrameReader, JpegFrameReader};
pub use plan::{InferenceFramePlan, RenderPlan};
pub use render::{
    RawFrameRenderSpec, RenderGeometry, staging_output_path, validate_output_destination,
};
pub use sequence::PingPongFrames;
