use burn::nn::{BatchNorm, BatchNormConfig, Relu, conv::Conv2d};
use burn::tensor::{Tensor, backend::Backend};

#[derive(Debug)]
pub struct MobileOneBlock<B: Backend> {
    branches: Vec<(Conv2d<B>, BatchNorm<B>)>,
    scale: Option<(Conv2d<B>, BatchNorm<B>)>,
    skip: Option<BatchNorm<B>>,
    activation: bool,
}

impl<B: Backend> MobileOneBlock<B> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        groups: usize,
        num_conv_branches: usize,
        is_linear: bool,
        device: &B::Device,
    ) -> Self {
        assert!(num_conv_branches > 0);
        assert!(in_channels.is_multiple_of(groups));
        assert!(out_channels.is_multiple_of(groups));
        let branches = (0..num_conv_branches)
            .map(|_| {
                let conv = burn::nn::conv::Conv2dConfig::new(
                    [in_channels, out_channels],
                    [kernel_size, kernel_size],
                )
                .with_stride([stride, stride])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(
                    padding, padding, padding, padding,
                ))
                .with_groups(groups)
                .with_bias(false)
                .init(device);
                let bn = BatchNormConfig::new(out_channels).init(device);
                (conv, bn)
            })
            .collect();
        let scale = if kernel_size > 1 {
            let conv = burn::nn::conv::Conv2dConfig::new([in_channels, out_channels], [1, 1])
                .with_stride([stride, stride])
                .with_padding(burn::nn::PaddingConfig2d::Valid)
                .with_groups(groups)
                .with_bias(false)
                .init(device);
            Some((conv, BatchNormConfig::new(out_channels).init(device)))
        } else {
            None
        };
        let skip = if stride == 1 && in_channels == out_channels {
            Some(BatchNormConfig::new(in_channels).init(device))
        } else {
            None
        };
        Self {
            branches,
            scale,
            skip,
            activation: !is_linear,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut output: Option<Tensor<B, 4>> = None;
        for (conv, bn) in &self.branches {
            let branch = bn.forward(conv.forward(input.clone()));
            output = Some(match output {
                Some(current) => current + branch,
                None => branch,
            });
        }
        if let Some((conv, bn)) = &self.scale {
            let branch = bn.forward(conv.forward(input.clone()));
            output = Some(match output {
                Some(current) => current + branch,
                None => branch,
            });
        }
        if let Some(skip) = &self.skip {
            let branch = skip.forward(input);
            output = Some(match output {
                Some(current) => current + branch,
                None => branch,
            });
        }
        let output = output.expect("MobileOneBlock always has a convolution branch");
        if self.activation {
            Relu.forward(output)
        } else {
            output
        }
    }
}
