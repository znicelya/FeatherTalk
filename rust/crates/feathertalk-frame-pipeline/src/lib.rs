mod commands;
mod error;
mod extraction;
mod model;
mod process;

pub use commands::{CommandSpec, frame_command};
pub use error::PipelineError;
pub use extraction::{ExtractedFrame, FrameBatch, extract_frames, extract_frames_with_runner};
pub use model::{
    AnomalyCode, FaceDetection, FrameAnomaly, FramePipelineSpec, FrameQuality, QualityReport,
    RecoveryAction,
};
pub use process::{
    FrameExtractor, MAX_CAPTURE_BYTES, MAX_FRAME_BYTES, ProcessOutput, ProcessRunner,
    SystemProcessRunner,
};
