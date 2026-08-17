use super::{AudioConvHubert, Down, InvertedResidual, OriginalUnet};
use burn::nn::{BatchNormConfig, PaddingConfig2d, conv::Conv2dConfig};
use burn::tensor::backend::Backend;

#[derive(Debug, Clone)]
pub struct InvertedResidualConfig {
    pub inp: usize,
    pub oup: usize,
    pub expansion: usize,
    pub stride: usize,
}

impl InvertedResidualConfig {
    pub const fn new(inp: usize, oup: usize) -> Self {
        Self {
            inp,
            oup,
            expansion: 6,
            stride: 1,
        }
    }

    pub const fn with_expansion(mut self, expansion: usize) -> Self {
        self.expansion = expansion;
        self
    }

    pub const fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> InvertedResidual<B> {
        InvertedResidual::new(self, device)
    }
}

#[derive(Debug, Clone)]
pub struct DownConfig {
    pub inp: usize,
    pub oup: usize,
}

impl DownConfig {
    pub const fn new(inp: usize, oup: usize) -> Self {
        Self { inp, oup }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Down<B> {
        Down::new(self.inp, self.oup, device)
    }
}

#[derive(Debug, Clone)]
pub struct AudioConvHubertConfig {
    pub channels: [usize; 5],
}

impl AudioConvHubertConfig {
    pub const fn new(channels: [usize; 5]) -> Self {
        Self { channels }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> AudioConvHubert<B> {
        AudioConvHubert::new(self.channels, device)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OriginalUnetConfig {
    pub channels: [usize; 5],
}

impl OriginalUnetConfig {
    pub const fn production() -> Self {
        Self {
            channels: [32, 64, 128, 256, 512],
        }
    }

    pub const fn parity_micro() -> Self {
        Self {
            channels: [2, 4, 8, 16, 32],
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> OriginalUnet<B> {
        OriginalUnet::new(self.channels, device)
    }
}

pub(crate) fn conv2d<B: Backend>(
    channels: [usize; 2],
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: PaddingConfig2d,
    bias: bool,
    device: &B::Device,
) -> burn::nn::conv::Conv2d<B> {
    Conv2dConfig::new(channels, kernel_size)
        .with_stride(stride)
        .with_padding(padding)
        .with_bias(bias)
        .init(device)
}

pub(crate) fn batch_norm<B: Backend>(
    channels: usize,
    device: &B::Device,
) -> burn::nn::BatchNorm<B> {
    BatchNormConfig::new(channels).init(device)
}
