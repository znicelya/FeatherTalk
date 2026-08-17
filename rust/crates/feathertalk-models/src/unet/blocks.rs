use super::config::{InvertedResidualConfig, batch_norm, conv2d};
use burn::nn::{
    BatchNorm, Relu,
    conv::{Conv2d, Conv2dConfig},
    interpolate::{Interpolate2d, Interpolate2dConfig, InterpolateMode},
};
use burn::tensor::{Tensor, backend::Backend, ops::PadMode};

#[derive(burn::module::Module, Debug)]
pub struct InvertedResidual<B: Backend> {
    pub expand_conv: Conv2d<B>,
    pub expand_bn: BatchNorm<B>,
    pub depthwise_conv: Conv2d<B>,
    pub depthwise_bn: BatchNorm<B>,
    pub project_conv: Conv2d<B>,
    pub project_bn: BatchNorm<B>,
    #[module(skip)]
    pub use_residual: bool,
}

impl<B: Backend> InvertedResidual<B> {
    pub(crate) fn new(config: &InvertedResidualConfig, device: &B::Device) -> Self {
        assert!(matches!(config.stride, 1 | 2));
        let hidden = config.inp * config.expansion;
        Self {
            expand_conv: conv2d(
                [config.inp, hidden],
                [1, 1],
                [1, 1],
                burn::nn::PaddingConfig2d::Valid,
                false,
                device,
            ),
            expand_bn: batch_norm(hidden, device),
            depthwise_conv: Conv2dConfig::new([hidden, hidden], [3, 3])
                .with_stride([config.stride, config.stride])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_groups(hidden)
                .with_bias(false)
                .init(device),
            depthwise_bn: batch_norm(hidden, device),
            project_conv: conv2d(
                [hidden, config.oup],
                [1, 1],
                [1, 1],
                burn::nn::PaddingConfig2d::Valid,
                false,
                device,
            ),
            project_bn: batch_norm(config.oup, device),
            use_residual: config.stride == 1 && config.inp == config.oup,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let relu = Relu;
        let output = relu.forward(
            self.expand_bn
                .forward(self.expand_conv.forward(input.clone())),
        );
        let output = relu.forward(
            self.depthwise_bn
                .forward(self.depthwise_conv.forward(output)),
        );
        let output = self.project_bn.forward(self.project_conv.forward(output));
        if self.use_residual {
            input + output
        } else {
            output
        }
    }
}

#[derive(burn::module::Module, Debug)]
pub struct DoubleConvDw<B: Backend> {
    pub first: InvertedResidual<B>,
    pub second: InvertedResidual<B>,
}

impl<B: Backend> DoubleConvDw<B> {
    pub(crate) fn new(inp: usize, oup: usize, stride: usize, device: &B::Device) -> Self {
        Self {
            first: InvertedResidualConfig::new(inp, oup)
                .with_expansion(2)
                .with_stride(stride)
                .init(device),
            second: InvertedResidualConfig::new(oup, oup)
                .with_expansion(2)
                .with_stride(1)
                .init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        self.second.forward(self.first.forward(input))
    }
}

#[derive(burn::module::Module, Debug)]
pub struct InConvDw<B: Backend> {
    pub inconv: InvertedResidual<B>,
}

impl<B: Backend> InConvDw<B> {
    pub(crate) fn new(inp: usize, oup: usize, device: &B::Device) -> Self {
        Self {
            inconv: InvertedResidualConfig::new(inp, oup)
                .with_expansion(2)
                .init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        self.inconv.forward(input)
    }
}

#[derive(burn::module::Module, Debug)]
pub struct Down<B: Backend> {
    pub maxpool_conv: DoubleConvDw<B>,
}

impl<B: Backend> Down<B> {
    pub(crate) fn new(inp: usize, oup: usize, device: &B::Device) -> Self {
        Self {
            maxpool_conv: DoubleConvDw::new(inp, oup, 2, device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        self.maxpool_conv.forward(input)
    }
}

#[derive(burn::module::Module, Debug)]
pub struct Up<B: Backend> {
    pub up: Interpolate2d,
    pub conv: DoubleConvDw<B>,
}

impl<B: Backend> Up<B> {
    pub(crate) fn new(inp: usize, oup: usize, device: &B::Device) -> Self {
        let up = Interpolate2dConfig::new()
            .with_mode(InterpolateMode::Linear)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_align_corners(true)
            .init();
        Self {
            up,
            conv: DoubleConvDw::new(inp, oup, 1, device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>, skip: Tensor<B, 4>) -> Tensor<B, 4> {
        let input = self.up.forward(input);
        let [_, _, input_h, input_w] = input.dims();
        let [_, _, skip_h, skip_w] = skip.dims();
        assert!(
            skip_h >= input_h,
            "skip height is smaller than upsampled input"
        );
        assert!(
            skip_w >= input_w,
            "skip width is smaller than upsampled input"
        );
        let diff_h = skip_h - input_h;
        let diff_w = skip_w - input_w;
        let input = input.pad(
            [
                (0, 0),
                (0, 0),
                (diff_h / 2, diff_h - diff_h / 2),
                (diff_w / 2, diff_w - diff_w / 2),
            ],
            PadMode::Constant(0.0),
        );
        self.conv.forward(Tensor::cat(vec![input, skip], 1))
    }
}

#[derive(burn::module::Module, Debug)]
pub struct OutConv<B: Backend> {
    pub conv: Conv2d<B>,
}

impl<B: Backend> OutConv<B> {
    pub(crate) fn new(inp: usize, device: &B::Device) -> Self {
        Self {
            conv: conv2d(
                [inp, 3],
                [1, 1],
                [1, 1],
                burn::nn::PaddingConfig2d::Valid,
                true,
                device,
            ),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        self.conv.forward(input)
    }
}
