#![allow(dead_code)]

use burn::tensor::{Tensor, backend::Backend};
use feathertalk_models::unet::{OriginalUnet, OriginalUnetConfig};
use feathertalk_training::{PerceptualFeatureExtractor, TrainingConfig, TrainingMode};

pub type CpuBackend = burn::backend::NdArray<f32>;
pub type CpuAutodiffBackend = burn::backend::Autodiff<CpuBackend>;
pub type CpuDevice = burn::backend::ndarray::NdArrayDevice;

#[derive(Debug, Clone, Copy)]
pub struct IdentityExtractor;

impl<B: Backend> PerceptualFeatureExtractor<B> for IdentityExtractor {
    fn forward(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        image
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NanExtractor;

impl<B: Backend> PerceptualFeatureExtractor<B> for NanExtractor {
    fn forward(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        image.mul_scalar(f32::NAN)
    }
}

pub fn model(device: &CpuDevice) -> OriginalUnet<CpuAutodiffBackend> {
    OriginalUnetConfig::parity_micro().init::<CpuAutodiffBackend>(device)
}

pub fn training_config(
    mode: TrainingMode,
    batch_size: u64,
    total_epochs: u64,
    temporal_stride: u64,
) -> TrainingConfig {
    TrainingConfig {
        mode,
        batch_size,
        learning_rate: 1e-4,
        total_epochs,
        temporal_stride,
        mouth_weight: 4.0,
        temporal_weight: 0.5,
        temporal_mouth_weight: 4.0,
        perceptual_weight: 0.01,
    }
}

pub fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-5,
        "expected {expected}, got {actual}"
    );
}
