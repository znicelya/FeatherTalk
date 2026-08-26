use burn::{
    module::{Initializer, Module, list_param_ids},
    nn::{Linear, LinearConfig},
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
    tensor::{
        Tensor,
        backend::{AutodiffBackend, Backend},
    },
};
use feathertalk_training::{
    CheckpointCompatibility, CheckpointDescriptor, DATA_LOADER_STATE_SCHEMA_VERSION,
    DataLoaderConfig, DataLoaderState, Provenance, RandomAlgorithm, RestoredTrainingState,
    SamplingConfig, SamplingKind, TRAINING_STATE_SCHEMA_VERSION, TrainingCheckpointState,
    TrainingConfig, TrainingMode, load_training_checkpoint, save_training_checkpoint,
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

    fn deterministic(device: &B::Device) -> Self {
        Self {
            linear: LinearConfig::new(2, 1)
                .with_initializer(Initializer::Constant { value: 0.125 })
                .init(device),
        }
    }

    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(input)
    }
}

fn train_step<B: AutodiffBackend>(
    model: TinyModel<B>,
    optimizer: &mut impl Optimizer<TinyModel<B>, B>,
    input_values: [[f32; 2]; 1],
    target_values: [[f32; 1]; 1],
    device: &B::Device,
) -> (TinyModel<B>, f32) {
    let input = Tensor::<B, 2>::from_floats(input_values, device);
    let target = Tensor::<B, 2>::from_floats(target_values, device);
    let loss = (model.forward(input) - target).abs().mean();
    let loss_value = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
    let gradients = GradientsParams::from_grads(loss.backward(), &model);
    (optimizer.step(1e-2, model, gradients), loss_value)
}

fn model_parameter_values(model: &TinyModel<CpuAutodiffBackend>) -> Vec<f32> {
    let mut values = model
        .linear
        .weight
        .val()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    if let Some(bias) = &model.linear.bias {
        values.extend(bias.val().into_data().to_vec::<f32>().unwrap());
    }
    values
}

fn model_record_bytes(model: &TinyModel<CpuAutodiffBackend>) -> Vec<u8> {
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    recorder.record(model.clone().into_record(), ()).unwrap()
}

fn model_parameter_ids(model: &TinyModel<CpuAutodiffBackend>) -> Vec<u64> {
    list_param_ids::<TinyModel<CpuAutodiffBackend>, CpuAutodiffBackend>(model)
        .into_iter()
        .map(|id| id.val())
        .collect()
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
        asset_provenance: Provenance {
            entries: BTreeMap::new(),
        },
        model_provenance: Provenance {
            entries: BTreeMap::new(),
        },
    }
}

#[test]
fn checkpoint_round_trip_loads_new_model_and_optimizer_instances() {
    let device = Default::default();
    <CpuAutodiffBackend as burn::tensor::backend::Backend>::seed(&device, 123);
    let model = TinyModel::<CpuAutodiffBackend>::new(&device);
    let mut optimizer = AdamConfig::new().init();
    let (model, _) = train_step(model, &mut optimizer, [[1.0, -2.0]], [[0.5]], &device);
    let descriptor = CheckpointDescriptor::new("tiny", "tiny-v1", "0".repeat(64));
    let checkpoint = tempfile::tempdir()
        .unwrap()
        .path()
        .join("checkpoint-000001");
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

#[test]
fn restored_adam_and_model_match_uninterrupted_next_step() {
    let device = Default::default();
    let input0 = [[1.0, -2.0]];
    let target0 = [[0.5]];
    let input1 = [[-0.25, 3.0]];
    let target1 = [[-0.75]];

    // The uninterrupted reference path.
    let (continuous_model, mut continuous_optimizer) = (
        TinyModel::<CpuAutodiffBackend>::deterministic(&device),
        AdamConfig::new().init(),
    );
    let (continuous_model, first_loss) = train_step(
        continuous_model,
        &mut continuous_optimizer,
        input0,
        target0,
        &device,
    );
    let (continuous_model, continuous_second_loss) = train_step(
        continuous_model,
        &mut continuous_optimizer,
        input1,
        target1,
        &device,
    );

    // The interrupted path uses the same initial values, then persists both
    // the model record and Adam's parameter-keyed momentum record.
    let (interrupted_model, mut interrupted_optimizer) = (
        TinyModel::<CpuAutodiffBackend>::deterministic(&device),
        AdamConfig::new().init(),
    );
    let (interrupted_model, interrupted_first_loss) = train_step(
        interrupted_model,
        &mut interrupted_optimizer,
        input0,
        target0,
        &device,
    );
    assert!(!interrupted_optimizer.to_record().is_empty());
    assert!((first_loss - interrupted_first_loss).abs() <= 1e-6);

    let interrupted_ids = model_parameter_ids(&interrupted_model);
    let interrupted_record = model_record_bytes(&interrupted_model);
    let root = tempfile::tempdir().unwrap();
    let checkpoint = root.path().join("checkpoint-000001");
    let state = progress_state();
    let descriptor = CheckpointDescriptor::new("tiny", "tiny-v1", "0".repeat(64));
    save_training_checkpoint::<CpuAutodiffBackend, _, _>(
        &checkpoint,
        &interrupted_model,
        &interrupted_optimizer,
        descriptor.clone(),
        state.clone(),
    )
    .unwrap();

    <CpuAutodiffBackend as burn::tensor::backend::Backend>::seed(&device, 999);
    let fresh_model = TinyModel::<CpuAutodiffBackend>::new(&device);
    let fresh_optimizer = AdamConfig::new().init();
    let mut expected =
        CheckpointCompatibility::new(descriptor, training_config_for_state(&state), 5);
    expected.asset_provenance = state.asset_provenance.clone();
    expected.model_provenance = state.model_provenance.clone();
    let restored = load_training_checkpoint::<CpuAutodiffBackend, _, _>(
        &checkpoint,
        &fresh_model,
        &fresh_optimizer,
        &device,
        &expected,
    )
    .unwrap();
    assert_eq!(restored.state, state);
    assert_eq!(model_parameter_ids(&restored.model), interrupted_ids);
    assert_eq!(model_record_bytes(&restored.model), interrupted_record);

    let RestoredTrainingState {
        model: restored_model,
        optimizer: mut restored_optimizer,
        ..
    } = restored;
    let (restored_model, restored_second_loss) = train_step(
        restored_model,
        &mut restored_optimizer,
        input1,
        target1,
        &device,
    );

    let continuous_values = model_parameter_values(&continuous_model);
    let restored_values = model_parameter_values(&restored_model);
    assert_eq!(continuous_values.len(), restored_values.len());
    let max_abs_error = continuous_values
        .iter()
        .zip(restored_values.iter())
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs_error <= 1e-4,
        "restored Adam update diverged: max_abs_error={max_abs_error}"
    );
    assert!((continuous_second_loss - restored_second_loss).abs() <= 1e-4);
}

fn training_config_for_state(state: &TrainingCheckpointState) -> TrainingConfig {
    state.training_config.clone()
}

fn progress_state() -> TrainingCheckpointState {
    TrainingCheckpointState {
        schema_version: TRAINING_STATE_SCHEMA_VERSION,
        epoch: 3,
        global_step: 1,
        random_seed: 17,
        data_loader: DataLoaderState {
            schema_version: DATA_LOADER_STATE_SCHEMA_VERSION,
            random_algorithm: RandomAlgorithm::Splitmix64FisherYatesV1,
            config: DataLoaderConfig {
                batch_size: 2,
                seed: 17,
                sampling: SamplingConfig {
                    kind: SamplingKind::SingleFrame,
                    temporal_stride: 0,
                },
            },
            frame_count: 5,
            epoch: 3,
            next_position: 4,
        },
        training_config: TrainingConfig {
            mode: TrainingMode::Baseline,
            batch_size: 2,
            learning_rate: 1e-2,
            total_epochs: 10,
            temporal_stride: 0,
            mouth_weight: 0.0,
            temporal_weight: 0.0,
            temporal_mouth_weight: 0.0,
            perceptual_weight: 0.01,
        },
        asset_provenance: Provenance {
            entries: BTreeMap::new(),
        },
        model_provenance: Provenance {
            entries: BTreeMap::new(),
        },
    }
}
