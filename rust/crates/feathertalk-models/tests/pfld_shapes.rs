use burn::tensor::Tensor;
use feathertalk_models::backend::CpuBackend;
use feathertalk_models::{PFLD_GhostOne, PFLD_OUTPUT_VALUES, PfldConfig};

#[test]
fn production_config_is_fixed() {
    let config = PfldConfig::production();
    assert_eq!(config.width_factor, 0.5);
    assert_eq!(config.input_size, 192);
    assert_eq!(config.landmark_count, 110);
    assert_eq!(config.num_conv_branches, 6);
    assert_eq!(PFLD_OUTPUT_VALUES, 220);
}

#[test]
fn production_model_shape_is_declared() {
    let device = Default::default();
    let model = PFLD_GhostOne::new(PfldConfig::production(), &device);
    let input = Tensor::<CpuBackend, 4>::zeros([1, 3, 192, 192], &device);
    assert_eq!(model.forward(input).dims(), [1, PFLD_OUTPUT_VALUES]);
}
