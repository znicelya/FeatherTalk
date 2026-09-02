//! Dependency-light pixel kernels that reproduce OpenCV's CPU results.
//!
//! Every kernel here is pinned by fixtures generated with OpenCV on the CPU;
//! see `tests/fixtures/opencv_cpu_v1/` and `rust/tools/image-parity/`.

mod error;
mod image;
mod jpeg;

pub use error::ImageError;
pub use image::BgrImage;
pub use jpeg::decode_jpeg;
