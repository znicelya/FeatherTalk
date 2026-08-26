use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{Tensor, backend::{AutodiffBackend, Backend}},
};
use feathertalk_training::{
    CheckpointCompatibility, CheckpointDescriptor, DataLoaderConfig, DataLoaderState,
    RandomAlgorithm, SamplingConfig, SamplingKind, TrainingCheckpointState, TrainingConfig,
    TrainingMode, DATA_LOADER_STATE_SCHEMA_VERSION, TRAINING_STATE_SCHEMA_VERSION,
    load_training_checkpoint, save_training_checkpoint,
};
use std::collections::BTreeMap;

type CpuBackend = burn::backend::NdArray<f32>;
type CpuAutodiffBackend = burn::backend::Autodiff<CpuBackend>;

#[derive(Module, Debug)]
struct TinyModel<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> TinyModel<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            linear: LinearConfig::new(2, 1).init(device),
        }
    }

    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(input)
    }
}

fn train_one_step<B: AutodiffBackend>(
    model: TinyModel<B>,
    optimizer: &mut impl Optimizer<TinyModel<B>, B>,
    device: &B::Device,
) -> TinyModel<B> {
    let input = Tensor::<B, 2>::from_floats([[1.0, -2.0]], device);
    let target = Tensor::<B, 2>::from_floats([[0.5]], device);
    let loss = (model.forward(input) - target).abs().mean();
    let gradients = GradientsParams::from_grads(loss.backward(), &model);
    optimizer.step(1e-2, model, gradients)
}

fn training_config() -> TrainingConfig {
    TrainingConfig {
        mode: TrainingMode::Baseline,
        batch_size: 1,
        learning_rate: 1e-2,
        total_epochs: 2,
        temporal_stride: 0,
        mouth_weight: 0.0,
        temporal_weight: 0.0,
        temporal_mouth_weight: 0.0,
        perceptual_weight: 0.01,
    }
}

fn state() -> TrainingCheckpointState {
    TrainingCheckpointState {
        schema_version: TRAINING_STATE_SCHEMA_VERSION,
        epoch: 0,
        global_step: 1,
        random_seed: 7,
        data_loader: DataLoaderState {
            schema_version: DATA_LOADER_STATE_SCHEMA_VERSION,
            random_algorithm: RandomAlgorithm::Splitmix64FisherYatesV1,
            config: DataLoaderConfig {
                batch_size: 1,
                seed: 7,
                sampling: SamplingConfig {
                    kind: SamplingKind::SingleFrame,
                    temporal_stride: 0,
                },
            },
            frame_count: 2,
            epoch: 0,
            next_position: 0,
        },
        training_config: training_config(),
        asset_provenance: BTreeMap::new(),
        model_provenance: BTreeMap::new(),
    }
}

#[test]
fn checkpoint_round_trip_loads_new_model_and_optimizer_instances() {
    let device = Default::default();
    <CpuAutodiffBackend as burn::tensor::backend::Backend>::seed(&device, 123);
    let model = TinyModel::<CpuAutodiffBackend>::new(&device);
    let mut optimizer = AdamConfig::new().init();
    let model = train_one_step(model, &mut optimizer, &device);
    let descriptor = CheckpointDescriptor::new("tiny", "tiny-v1", "0".repeat(64));
    let checkpoint = tempfile::tempdir().unwrap().path().join("checkpoint-000001");
    let state = state();

    let manifest = save_training_checkpoint::<CpuAutodiffBackend, _, _>(
        &checkpoint,
        &model,
        &optimizer,
        descriptor.clone(),
        state.clone(),
    )
    .unwrap();

    let mut expected = CheckpointCompatibility::new(descriptor, training_config(), 2);
    expected.asset_provenance = state.asset_provenance.clone();
    expected.model_provenance = state.model_provenance.clone();
    let fresh_model = TinyModel::<CpuAutodiffBackend>::new(&device);
    let fresh_optimizer = AdamConfig::new().init();
    let restored = load_training_checkpoint::<CpuAutodiffBackend, _, _>(
        &checkpoint,
        &fresh_model,
        &fresh_optimizer,
        &device,
        &expected,
    )
    .unwrap();

    assert_eq!(restored.state, state);
    assert_eq!(restored.manifest, manifest);
}
