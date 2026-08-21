use burn::tensor::{Tensor, backend::Backend};

use super::config::{PFLD_INPUT_CHANNELS, PFLD_OUTPUT_VALUES, PfldConfig};

#[allow(non_camel_case_types)]
#[derive(Debug, Clone)]
pub struct PFLD_GhostOne {
    config: PfldConfig,
}

impl PFLD_GhostOne {
    pub fn new(config: PfldConfig, _device: &impl Sized) -> Self {
        assert_eq!(config.width_factor, 0.5);
        assert_eq!(config.input_size, 192);
        assert_eq!(config.landmark_count * 2, PFLD_OUTPUT_VALUES);
        assert_eq!(PFLD_INPUT_CHANNELS, 3);
        Self { config }
    }

    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch, channels, height, width] = input.dims();
        assert_eq!(channels, PFLD_INPUT_CHANNELS);
        assert_eq!(height, self.config.input_size);
        assert_eq!(width, self.config.input_size);
        Tensor::zeros([batch, self.config.output_values()], &input.device())
    }
}
