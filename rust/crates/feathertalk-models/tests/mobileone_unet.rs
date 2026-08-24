use burn::tensor::Tensor;
use feathertalk_models::{
    backend::CpuBackend,
    unet::{
        MobileOneAudioConvHubertConfig, MobileOneDownConfig, MobileOneUnetConfig, MobileOneUpConfig,
    },
};

#[test]
fn production_mobileone_config_is_fixed() {
    let config = MobileOneUnetConfig::production();
    assert_eq!(config.channels, [32, 64, 128, 256, 512]);
    assert_eq!(config.num_conv_branches, 2);
}

#[test]
fn mobileone_down_halves_the_spatial_size() {
    let device = Default::default();
    let down = MobileOneDownConfig::new(2, 4, 2).init::<CpuBackend>(&device);
    let input = Tensor::zeros([1, 2, 160, 160], &device);
    assert_eq!(down.forward(input).dims(), [1, 4, 80, 80]);
}

#[test]
fn mobileone_up_restores_the_skip_spatial_size() {
    let device = Default::default();
    let up = MobileOneUpConfig::new(16, 4, 2).init::<CpuBackend>(&device);
    let input = Tensor::zeros([1, 8, 20, 20], &device);
    let skip = Tensor::zeros([1, 8, 40, 40], &device);
    assert_eq!(up.forward(input, skip).dims(), [1, 4, 40, 40]);
}

#[test]
fn mobileone_hubert_audio_branch_matches_the_micro_bottleneck() {
    let device = Default::default();
    let branch =
        MobileOneAudioConvHubertConfig::new([2, 4, 8, 16, 32], 2).init::<CpuBackend>(&device);
    let audio = Tensor::zeros([1, 16, 32, 32], &device);
    assert_eq!(branch.forward(audio).dims(), [1, 32, 10, 10]);
}
