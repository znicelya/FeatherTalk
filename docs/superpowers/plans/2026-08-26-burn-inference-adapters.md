# FeatherTalk Burn Inference Adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Connect validated FeatherHuBERT features and BGR frame values to Original/MobileOne Burn inference, and load a restricted FeatherHuBERT PyTorch checkpoint without Python at runtime.

**Architecture:** feathertalk-models owns only model-level inference traits and model-backed audio adapters. feathertalk-inference owns domain-value validation, tensor construction, model-output validation, and single-frame rendering. feathertalk-weights depends one-way on feathertalk-models and owns restricted checkpoint inspection/loading.

**Tech Stack:** Rust 1.92, edition 2024, Burn 0.21.0, burn-store 0.21.0, NdArray CPU backend, existing FeatherTalk audio/preprocess/inference crates.

## Global Constraints

- Execute inline in one visible session; do not use subagents.
- Work on branch burn-inference-adapters in .worktrees/burn-inference-adapters, created from the committed plan on main.
- Follow strict RED-GREEN-REFACTOR TDD. Every production behavior gets a focused test that is observed failing for the intended reason before implementation.
- Keep the Cargo dependency direction feathertalk-weights -> feathertalk-models. feathertalk-models must not depend on feathertalk-weights.
- Keep checkpoint parsing/loading out of feathertalk-inference and all frame/bbox/render values out of feathertalk-models.
- Product UNet audio input is exactly [1,16,32,32], image input is exactly [1,6,160,160], and output is exactly [1,3,160,160].
- FeatureMatrix must have dims=1024, a positive even token count, and two consecutive tokens per video frame.
- Generic FeatherHuBERT checkpoint loading permits any positive output_dim so the committed parity micro fixture with output_dim=64 remains valid. Product wiring and the user checkpoint acceptance require output_dim=1024.
- Checkpoint metadata dropout is validated when present, but the loaded Rust inference config always stores dropout=0.0.
- Rust implementation and tests must not launch Python. Python remains only an external migration oracle and was used before implementation for read-only checkpoint inspection.
- Never read demo/kanghui_training_video_featherhubert_188_latest/kanghui_training_video.MOV in this slice.
- Never modify, move, rename, stage, commit, or delete demo/kanghui_training_video_featherhubert_188_latest/.
- The real checkpoint test is enabled only by FEATHERTALK_FEATHER_HUBERT_CHECKPOINT and must read the model file without writing to its directory.
- Do not silently fall back from WGPU to CPU. Existing certified-adapter WGPU tests remain explicitly ignored unless invoked in their established environment.
- This slice does not add JPEG/WAV decoding, resampling, FFmpeg process control, a full video loop, model packaging, ONNX, CLI, worker RPC, or GPUI.

---

## File Map

Create or modify only these implementation paths:

- Create rust/crates/feathertalk-models/src/unet/inference.rs — TalkingHeadModel trait and the two production inference implementations.
- Modify rust/crates/feathertalk-models/src/unet/mod.rs — export TalkingHeadModel.
- Modify rust/crates/feathertalk-models/src/feather_hubert/adapter.rs — add BurnFeatherHubertEncoder::from_model.
- Create rust/crates/feathertalk-models/tests/inference_adapters.rs — model-level trait and adapter tests.
- Modify rust/crates/feathertalk-models/tests/feather_hubert_long_audio.rs — loaded-model adapter behavior.
- Modify rust/crates/feathertalk-inference/Cargo.toml — add Burn/audio/models dependencies.
- Create rust/crates/feathertalk-inference/src/burn.rs — audio-window value, tensor bridges, output validation, and planned-frame rendering.
- Modify rust/crates/feathertalk-inference/src/error.rs — structured adapter errors.
- Modify rust/crates/feathertalk-inference/src/lib.rs — public adapter exports.
- Create rust/crates/feathertalk-inference/tests/burn_audio.rs — audio-window contract tests.
- Create rust/crates/feathertalk-inference/tests/burn_prediction.rs — model execution/output validation tests.
- Create rust/crates/feathertalk-inference/tests/burn_render.rs — single-frame composition and transactional tests.
- Modify rust/crates/feathertalk-weights/Cargo.toml — promote feathertalk-models to a normal dependency.
- Create rust/crates/feathertalk-weights/src/feather_hubert.rs — checkpoint facts, config inference, metadata validation, inspect/load API.
- Modify rust/crates/feathertalk-weights/src/legacy.rs — expose the existing top-level state-dict selector inside the crate.
- Modify rust/crates/feathertalk-weights/src/lib.rs — public FeatherHuBERT checkpoint exports.
- Modify rust/crates/feathertalk-weights/src/error.rs only if an existing structured variant cannot express a discovered failure; prefer MissingTensor, UnexpectedTensor, ShapeMismatch, DTypeMismatch, UnsupportedStructure, UnsafeLimit, and Store.
- Create rust/crates/feathertalk-weights/tests/feather_hubert_checkpoint.rs — committed golden checkpoint inspect/load tests.
- Create rust/crates/feathertalk-weights/tests/feather_hubert_real_checkpoint.rs — environment-gated real checkpoint CPU test.
- Modify rust/crates/feathertalk-parity/src/fixture.rs — route FeatherMicro CPU parity through the new checkpoint loader.

The design and this plan are committed on main before creating the implementation worktree.

### Task 1: Model-level talking-head interface and loaded FeatherHuBERT adapter

**Files:**

- Create: rust/crates/feathertalk-models/src/unet/inference.rs
- Modify: rust/crates/feathertalk-models/src/unet/mod.rs
- Modify: rust/crates/feathertalk-models/src/feather_hubert/adapter.rs
- Create: rust/crates/feathertalk-models/tests/inference_adapters.rs
- Modify: rust/crates/feathertalk-models/tests/feather_hubert_long_audio.rs

**Interfaces:**

- Produces TalkingHeadModel<B>::forward_talking_head(image, audio) -> Tensor<B,4>.
- Implements the trait only for OriginalUnet<B> and MobileOneUnetInference<B>.
- Produces BurnFeatherHubertEncoder::from_model(model, device).
- Keeps BurnFeatherHubertEncoder::from_config and model unchanged.

- [ ] **Step 1: Write the failing public-interface tests.**

Create inference_adapters.rs with:

~~~rust
use burn::tensor::Tensor;
use feathertalk_models::{
    backend::CpuBackend,
    unet::{
        MobileOneUnetConfig, MobileOneUnetInference, OriginalUnet, OriginalUnetConfig,
        TalkingHeadModel,
    },
};

fn assert_talking_head_model<M: TalkingHeadModel<CpuBackend>>() {}

#[test]
fn original_and_reparameterized_mobileone_implement_the_public_inference_trait() {
    assert_talking_head_model::<OriginalUnet<CpuBackend>>();
    assert_talking_head_model::<MobileOneUnetInference<CpuBackend>>();
}

#[test]
fn trait_forward_preserves_the_fixed_unet_contract() {
    let device = Default::default();
    let image = Tensor::<CpuBackend, 4>::zeros([1, 6, 160, 160], &device);
    let audio = Tensor::<CpuBackend, 4>::zeros([1, 16, 32, 32], &device);

    let original = OriginalUnetConfig::parity_micro().init::<CpuBackend>(&device);
    assert_eq!(
        original.forward_talking_head(image.clone(), audio.clone()).dims(),
        [1, 3, 160, 160]
    );

    let mobile = MobileOneUnetConfig::parity_micro()
        .init::<CpuBackend>(&device)
        .reparameterize();
    assert_eq!(
        mobile.forward_talking_head(image, audio).dims(),
        [1, 3, 160, 160]
    );
}
~~~

Add this test to feather_hubert_long_audio.rs:

~~~rust
#[test]
fn cpu_adapter_can_take_ownership_of_an_imported_model() {
    let device = Default::default();
    let model = FeatherHubertConfig::parity_micro().init::<CpuBackend>(&device);
    let mut encoder = BurnFeatherHubertEncoder::from_model(model, &device);

    assert_eq!(encoder.output_dim(), 64);
    assert_eq!(encoder.model().config.output_dim, 64);
    let rows = encoder.encode(0, &[0.0; 1360]).unwrap();
    assert_eq!(rows.len(), 4 * 64);
    assert!(rows.iter().all(|value| value.is_finite()));
}
~~~

- [ ] **Step 2: Run the focused tests and confirm RED.**

From rust/:

~~~powershell
cargo test -p feathertalk-models --test inference_adapters --test feather_hubert_long_audio
~~~

Expected: compilation fails because TalkingHeadModel and BurnFeatherHubertEncoder::from_model do not exist.

- [ ] **Step 3: Implement the minimal model trait.**

Create unet/inference.rs:

~~~rust
use burn::tensor::{Tensor, backend::Backend};

use super::{MobileOneUnetInference, OriginalUnet};

pub trait TalkingHeadModel<B: Backend> {
    fn forward_talking_head(
        &self,
        image: Tensor<B, 4>,
        audio: Tensor<B, 4>,
    ) -> Tensor<B, 4>;
}

impl<B: Backend> TalkingHeadModel<B> for OriginalUnet<B> {
    fn forward_talking_head(
        &self,
        image: Tensor<B, 4>,
        audio: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        self.forward(image, audio)
    }
}

impl<B: Backend> TalkingHeadModel<B> for MobileOneUnetInference<B> {
    fn forward_talking_head(
        &self,
        image: Tensor<B, 4>,
        audio: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        self.forward(image, audio)
    }
}
~~~

Export the trait from unet/mod.rs. Do not implement it for MobileOneUnet<B>.

- [ ] **Step 4: Add the loaded-model constructor.**

Add to the existing BurnFeatherHubertEncoder implementation:

~~~rust
pub fn from_model(model: FeatherHubertEncoder<B>, device: &B::Device) -> Self {
    let output_dim = model.config.output_dim;
    Self {
        model,
        device: device.clone(),
        output_dim,
    }
}
~~~

Refactor from_config to call from_model(config.init(device), device) so both constructors have one initialization path.

- [ ] **Step 5: Verify GREEN, formatting, and the negative type boundary.**

Add a compile_fail documentation example to TalkingHeadModel showing that MobileOneUnet<CpuBackend> cannot be passed to a function requiring TalkingHeadModel<CpuBackend>. Then run:

~~~powershell
cargo test -p feathertalk-models --test inference_adapters --test feather_hubert_long_audio
cargo test -p feathertalk-models --doc
cargo fmt --all -- --check
cargo clippy -p feathertalk-models --all-targets -- -D warnings
~~~

Expected: all commands exit 0, while the compile_fail example succeeds by proving the training graph does not satisfy the trait.

- [ ] **Step 6: Commit the model boundary.**

~~~powershell
git add rust/crates/feathertalk-models/src/unet/inference.rs rust/crates/feathertalk-models/src/unet/mod.rs rust/crates/feathertalk-models/src/feather_hubert/adapter.rs rust/crates/feathertalk-models/tests/inference_adapters.rs rust/crates/feathertalk-models/tests/feather_hubert_long_audio.rs
git commit -m "feat: add talking-head model adapters"
~~~

### Task 2: Deterministic FeatherHuBERT feature-window input

**Files:**

- Modify: rust/crates/feathertalk-inference/Cargo.toml
- Create: rust/crates/feathertalk-inference/src/burn.rs
- Modify: rust/crates/feathertalk-inference/src/error.rs
- Modify: rust/crates/feathertalk-inference/src/lib.rs
- Create: rust/crates/feathertalk-inference/tests/burn_audio.rs

**Interfaces:**

- Produces UnetAudioInput::shape() -> [1,16,32,32].
- Produces UnetAudioInput::as_slice() -> &[f32].
- Produces build_unet_audio_input(&FeatureMatrix, &InferenceFramePlan).
- Produces InvalidFeatureShape and InvalidAudioWindowIndex errors.
- Reuses existing OutputFrameOutOfRange for plan.output_index >= tokens/2.

- [ ] **Step 1: Add dependencies and write failing audio-window tests.**

Add normal dependencies on burn.workspace, feathertalk-audio, and feathertalk-models with default-features=false.

Create burn_audio.rs:

~~~rust
use feathertalk_audio::FeatureMatrix;
use feathertalk_inference::{
    InferenceError, InferenceFramePlan, build_unet_audio_input,
};

fn features(frame_count: usize) -> FeatureMatrix {
    let tokens = frame_count * 2;
    let values = (0..tokens)
        .flat_map(|token| {
            (0..1024).map(move |dimension| (token * 10_000 + dimension) as f32)
        })
        .collect();
    FeatureMatrix::new(tokens, 1024, values).unwrap()
}

#[test]
fn audio_window_flattens_two_tokens_per_slot_without_transpose() {
    let plan = InferenceFramePlan {
        output_index: 1,
        source_frame_index: 0,
        reference_frame_index: 0,
        audio_window: [
            None,
            None,
            Some(0),
            Some(1),
            Some(2),
            None,
            None,
            None,
        ],
    };
    let input = build_unet_audio_input(&features(3), &plan).unwrap();

    assert_eq!(input.shape(), [1, 16, 32, 32]);
    assert_eq!(input.as_slice().len(), 16 * 32 * 32);
    assert!(input.as_slice()[..2 * 2048].iter().all(|value| *value == 0.0));
    let first = 2 * 2048;
    assert_eq!(&input.as_slice()[first..first + 3], &[0.0, 1.0, 2.0]);
    assert_eq!(input.as_slice()[first + 1024], 10_000.0);
    let second = 3 * 2048;
    assert_eq!(input.as_slice()[second], 20_000.0);
    assert_eq!(input.as_slice()[second + 1024], 30_000.0);
}

#[test]
fn audio_window_rejects_invalid_feature_matrix_contracts() {
    for matrix in [
        FeatureMatrix::new(0, 1024, vec![]).unwrap(),
        FeatureMatrix::new(3, 1024, vec![0.0; 3 * 1024]).unwrap(),
        FeatureMatrix::new(2, 64, vec![0.0; 2 * 64]).unwrap(),
    ] {
        let plan = InferenceFramePlan {
            output_index: 0,
            source_frame_index: 0,
            reference_frame_index: 0,
            audio_window: [None; 8],
        };
        assert!(matches!(
            build_unet_audio_input(&matrix, &plan),
            Err(InferenceError::InvalidFeatureShape { .. })
        ));
    }
}

#[test]
fn audio_window_rejects_output_and_slot_indices_beyond_feature_frames() {
    let matrix = features(2);
    let output_plan = InferenceFramePlan {
        output_index: 2,
        source_frame_index: 0,
        reference_frame_index: 0,
        audio_window: [None; 8],
    };
    assert!(matches!(
        build_unet_audio_input(&matrix, &output_plan),
        Err(InferenceError::OutputFrameOutOfRange { index: 2, count: 2 })
    ));

    let slot_plan = InferenceFramePlan {
        output_index: 0,
        source_frame_index: 0,
        reference_frame_index: 0,
        audio_window: [Some(2), None, None, None, None, None, None, None],
    };
    assert!(matches!(
        build_unet_audio_input(&matrix, &slot_plan),
        Err(InferenceError::InvalidAudioWindowIndex {
            slot: 0,
            index: 2,
            frame_count: 2
        })
    ));
}
~~~

- [ ] **Step 2: Run the focused test and confirm RED.**

~~~powershell
cargo test -p feathertalk-inference --test burn_audio
~~~

Expected: compilation fails because burn.rs, UnetAudioInput, the builder, and the new errors do not exist.

- [ ] **Step 3: Implement the exact window expansion.**

In burn.rs define constants:

~~~rust
const FEATURE_DIMS: usize = 1024;
const TOKENS_PER_FRAME: usize = 2;
const AUDIO_WINDOW_SLOTS: usize = 8;
const AUDIO_VALUES_PER_SLOT: usize = TOKENS_PER_FRAME * FEATURE_DIMS;
const UNET_AUDIO_VALUES: usize = 16 * 32 * 32;
~~~

Implement the builder in this order:

1. Reject dims != 1024, tokens == 0, or odd tokens with InvalidFeatureShape.
2. Set frame_count=tokens/2 and reject plan.output_index >= frame_count with OutputFrameOutOfRange.
3. Allocate exactly 16*32*32 zero f32 values using try_reserve_exact and AllocationFailure on failure.
4. For each of the eight slots, leave None as zeros.
5. For Some(frame_index), reject frame_index >= frame_count with InvalidAudioWindowIndex.
6. Compute source_start=frame_index*2*1024 and destination_start=slot*2048 with checked arithmetic.
7. Copy exactly 2048 consecutive values with copy_from_slice.
8. Return a private-values UnetAudioInput exposing only shape and as_slice.

Export UnetAudioInput and build_unet_audio_input from the crate root.

- [ ] **Step 4: Verify GREEN and crate hygiene.**

~~~powershell
cargo test -p feathertalk-inference --test burn_audio
cargo fmt --all -- --check
cargo clippy -p feathertalk-inference --all-targets -- -D warnings
git diff --check
~~~

Expected: all audio-window tests pass and static checks emit no warnings.

- [ ] **Step 5: Commit the audio bridge.**

~~~powershell
git add rust/crates/feathertalk-inference/Cargo.toml rust/crates/feathertalk-inference/src/burn.rs rust/crates/feathertalk-inference/src/error.rs rust/crates/feathertalk-inference/src/lib.rs rust/crates/feathertalk-inference/tests/burn_audio.rs
git commit -m "feat: build deterministic UNet audio windows"
~~~

### Task 3: Burn tensor execution, model-output validation, and planned-frame rendering

**Files:**

- Modify: rust/crates/feathertalk-inference/src/burn.rs
- Modify: rust/crates/feathertalk-inference/src/error.rs
- Modify: rust/crates/feathertalk-inference/src/lib.rs
- Create: rust/crates/feathertalk-inference/tests/burn_prediction.rs
- Create: rust/crates/feathertalk-inference/tests/burn_render.rs

**Interfaces:**

- Produces run_unet_prediction<B,M>(&M, &UnetImageInput, &UnetAudioInput, &B::Device).
- Produces render_planned_frame<B,M>(&M, &BgrFrame, &FaceBoundingBox, &FeatureMatrix, &InferenceFramePlan, &RenderGeometry, &B::Device).
- Produces NonFiniteModelInput, ModelTensorData, NonFiniteModelOutput, and ModelOutputOutOfRange errors.
- Reuses TensorShapeMismatch for image/audio/output shapes and existing crop/resize/render_frame behavior.

- [ ] **Step 1: Write failing prediction tests with a real local model seam.**

Create burn_prediction.rs. Use this local model to generate exact output contracts:

~~~rust
use burn::tensor::{Tensor, TensorData};
use feathertalk_audio::FeatureMatrix;
use feathertalk_inference::{
    BgrFrame, InferenceError, InferenceFramePlan, RenderGeometry,
    build_unet_audio_input, build_unet_image_input, run_unet_prediction,
};
use feathertalk_models::{
    backend::CpuBackend,
    unet::{MobileOneUnetConfig, OriginalUnetConfig, TalkingHeadModel},
};

struct OutputModel {
    shape: [usize; 4],
    value: f32,
}

impl TalkingHeadModel<CpuBackend> for OutputModel {
    fn forward_talking_head(
        &self,
        image: Tensor<CpuBackend, 4>,
        _audio: Tensor<CpuBackend, 4>,
    ) -> Tensor<CpuBackend, 4> {
        let device = image.device();
        let elements = self.shape.into_iter().product();
        Tensor::from_data(
            TensorData::new(vec![self.value; elements], self.shape),
            &device,
        )
    }
}

fn valid_inputs() -> (
    feathertalk_inference::UnetImageInput,
    feathertalk_inference::UnetAudioInput,
) {
    let crop = BgrFrame::new(168, 168, vec![64; 168 * 168 * 3]).unwrap();
    let image = build_unet_image_input(&crop, &RenderGeometry::standard()).unwrap();
    let features = FeatureMatrix::new(2, 1024, vec![0.0; 2 * 1024]).unwrap();
    let plan = InferenceFramePlan {
        output_index: 0,
        source_frame_index: 0,
        reference_frame_index: 0,
        audio_window: [None, None, None, None, Some(0), None, None, None],
    };
    let audio = build_unet_audio_input(&features, &plan).unwrap();
    (image, audio)
}

#[test]
fn prediction_returns_validated_channel_first_values() {
    let device = Default::default();
    let (image, audio) = valid_inputs();
    let values = run_unet_prediction::<CpuBackend, _>(
        &OutputModel {
            shape: [1, 3, 160, 160],
            value: 0.25,
        },
        &image,
        &audio,
        &device,
    )
    .unwrap();
    assert_eq!(values.len(), 3 * 160 * 160);
    assert!(values.iter().all(|value| *value == 0.25));
}

#[test]
fn prediction_rejects_wrong_shape_non_finite_and_out_of_range_outputs() {
    let device = Default::default();
    let (image, audio) = valid_inputs();
    for (model, expected) in [
        (
            OutputModel { shape: [1, 3, 80, 80], value: 0.5 },
            "shape",
        ),
        (
            OutputModel { shape: [1, 3, 160, 160], value: f32::NAN },
            "finite",
        ),
        (
            OutputModel { shape: [1, 3, 160, 160], value: 1.01 },
            "range",
        ),
    ] {
        let error =
            run_unet_prediction::<CpuBackend, _>(&model, &image, &audio, &device).unwrap_err();
        match expected {
            "shape" => assert!(matches!(error, InferenceError::TensorShapeMismatch { .. })),
            "finite" => assert!(matches!(error, InferenceError::NonFiniteModelOutput { .. })),
            "range" => assert!(matches!(error, InferenceError::ModelOutputOutOfRange { .. })),
            _ => unreachable!(),
        }
    }
}

#[test]
fn original_and_reparameterized_mobileone_run_through_the_same_adapter() {
    let device = Default::default();
    let (image, audio) = valid_inputs();
    let original = OriginalUnetConfig::parity_micro().init::<CpuBackend>(&device);
    let original_values =
        run_unet_prediction::<CpuBackend, _>(&original, &image, &audio, &device).unwrap();
    assert!(original_values.iter().all(|value| (0.0..=1.0).contains(value)));

    let mobile = MobileOneUnetConfig::parity_micro()
        .init::<CpuBackend>(&device)
        .reparameterize();
    let mobile_values =
        run_unet_prediction::<CpuBackend, _>(&mobile, &image, &audio, &device).unwrap();
    assert!(mobile_values.iter().all(|value| (0.0..=1.0).contains(value)));
}
~~~

- [ ] **Step 2: Write failing planned-frame tests.**

Create burn_render.rs with a local constant model and these assertions:

~~~rust
#[test]
fn planned_frame_reuses_the_existing_crop_prediction_and_paste_kernel() {
    let device = Default::default();
    let model = OutputModel {
        shape: [1, 3, 160, 160],
        value: 1.0,
    };
    let frame = BgrFrame::new(2, 2, vec![10; 12]).unwrap();
    let original = frame.clone();
    let bbox = FaceBoundingBox { xmin: 0, ymin: 0, xmax: 2, ymax: 2 };
    let features = FeatureMatrix::new(2, 1024, vec![0.0; 2048]).unwrap();
    let plan = InferenceFramePlan {
        output_index: 0,
        source_frame_index: 0,
        reference_frame_index: 0,
        audio_window: [None, None, None, None, Some(0), None, None, None],
    };

    let rendered = render_planned_frame::<CpuBackend, _>(
        &model,
        &frame,
        &bbox,
        &features,
        &plan,
        &RenderGeometry::standard(),
        &device,
    )
    .unwrap();

    assert_eq!(frame, original);
    assert_eq!(rendered.as_bytes(), &[255; 12]);
}

#[test]
fn invalid_model_output_returns_before_any_frame_can_be_published() {
    let device = Default::default();
    let model = OutputModel {
        shape: [1, 3, 160, 160],
        value: f32::NAN,
    };
    let frame = BgrFrame::new(2, 2, vec![10; 12]).unwrap();
    let original = frame.clone();
    let bbox = FaceBoundingBox { xmin: 0, ymin: 0, xmax: 2, ymax: 2 };
    let features = FeatureMatrix::new(2, 1024, vec![0.0; 2048]).unwrap();
    let plan = InferenceFramePlan {
        output_index: 0,
        source_frame_index: 0,
        reference_frame_index: 0,
        audio_window: [None, None, None, None, Some(0), None, None, None],
    };

    assert!(matches!(
        render_planned_frame::<CpuBackend, _>(
            &model,
            &frame,
            &bbox,
            &features,
            &plan,
            &RenderGeometry::standard(),
            &device,
        ),
        Err(InferenceError::NonFiniteModelOutput { .. })
    ));
    assert_eq!(frame, original);
}
~~~

The file may share its local OutputModel definition through tests/support only if that support module remains test-only and contains no production behavior.

- [ ] **Step 3: Run both test binaries and confirm RED.**

~~~powershell
cargo test -p feathertalk-inference --test burn_prediction --test burn_render
~~~

Expected: compilation fails because the execution and planned-render functions and errors do not exist.

- [ ] **Step 4: Implement tensor construction and strict output validation.**

run_unet_prediction must:

1. Compare image.shape() with [1,6,160,160] and audio.shape() with [1,16,32,32].
2. Scan both slices before tensor creation; return NonFiniteModelInput { context, index } for the first invalid value.
3. Construct tensors with TensorData::new(slice.to_vec(), fixed_shape) on the supplied device.
4. Call model.forward_talking_head.
5. Check output.dims() equals [1,3,160,160] before copying data.
6. Convert output.into_data().to_vec::<f32>(); map conversion failure to ModelTensorData { context: "unet_output", message }.
7. Check length equals 3*160*160.
8. Reject the first non-finite value with NonFiniteModelOutput.
9. Reject the first value outside 0.0..=1.0 with ModelOutputOutOfRange.
10. Return the validated Vec<f32> without clamping.

Add private unit tests inside burn.rs for the non-finite input scanner because the public UnetImageInput and UnetAudioInput constructors already prevent callers from creating invalid values.

- [ ] **Step 5: Implement planned-frame composition in the approved order.**

render_planned_frame must:

~~~rust
pub fn render_planned_frame<B, M>(
    model: &M,
    frame: &BgrFrame,
    bbox: &feathertalk_preprocess::FaceBoundingBox,
    features: &feathertalk_audio::FeatureMatrix,
    plan: &InferenceFramePlan,
    geometry: &RenderGeometry,
    device: &B::Device,
) -> Result<BgrFrame, InferenceError>
where
    B: Backend,
    M: TalkingHeadModel<B>,
~~~

Implementation order is exact:

1. Validate the feature contract and plan.output_index by calling build_unet_audio_input before model execution.
2. crop_bgr(frame,bbox).
3. resize_bilinear(crop,geometry.crop_size(),geometry.crop_size()).
4. build_unet_image_input on that 168x168 crop.
5. run_unet_prediction.
6. Call existing render_frame(frame,bbox,&prediction,geometry).

Do not add a second paste implementation or mutate frame.

- [ ] **Step 6: Verify GREEN and all inference tests.**

~~~powershell
cargo test -p feathertalk-inference --test burn_prediction --test burn_render
cargo test -p feathertalk-inference --all-targets
cargo fmt --all -- --check
cargo clippy -p feathertalk-inference --all-targets -- -D warnings
git diff --check
~~~

Expected: all commands exit 0.

- [ ] **Step 7: Commit the Burn execution adapter.**

~~~powershell
git add rust/crates/feathertalk-inference/src/burn.rs rust/crates/feathertalk-inference/src/error.rs rust/crates/feathertalk-inference/src/lib.rs rust/crates/feathertalk-inference/tests/burn_prediction.rs rust/crates/feathertalk-inference/tests/burn_render.rs
git commit -m "feat: run Burn talking-head inference"
~~~

### Task 4: Restricted FeatherHuBERT checkpoint inspection and loading

**Files:**

- Modify: rust/crates/feathertalk-weights/Cargo.toml
- Create: rust/crates/feathertalk-weights/src/feather_hubert.rs
- Modify: rust/crates/feathertalk-weights/src/legacy.rs
- Modify: rust/crates/feathertalk-weights/src/lib.rs
- Modify: rust/crates/feathertalk-weights/src/error.rs only if necessary
- Create: rust/crates/feathertalk-weights/tests/feather_hubert_checkpoint.rs

**Interfaces:**

- Produces FeatherHubertCheckpoint and its four getters.
- Produces inspect_feather_hubert_checkpoint(path).
- Produces load_feather_hubert_checkpoint<B>(path,device).
- Uses the existing LegacyImportRequest/import_into path as the only tensor-application implementation.
- Uses model, state_dict, or a direct state dictionary through the existing selector.

- [ ] **Step 1: Write the failing committed-golden integration test.**

Create feather_hubert_checkpoint.rs. Reuse the existing ZipArchive extraction pattern from legacy_import.rs to extract weights/feather_micro.pth into a process-specific temp directory.

~~~rust
use burn::tensor::{Tensor, TensorData};
use feathertalk_models::backend::CpuBackend;
use feathertalk_weights::{
    inspect_feather_hubert_checkpoint, load_feather_hubert_checkpoint,
};

#[test]
fn golden_micro_checkpoint_is_inferred_loaded_and_executed() {
    let path = extract_fixture("weights/feather_micro.pth");
    let inspection = inspect_feather_hubert_checkpoint(&path).unwrap();
    assert_eq!(inspection.config().channels, 32);
    assert_eq!(inspection.config().expansion, 2);
    assert_eq!(inspection.config().num_blocks, 2);
    assert_eq!(inspection.config().output_dim, 64);
    assert_eq!(inspection.config().dropout, 0.0);
    assert_eq!(inspection.tensor_count(), 35);
    assert_eq!(inspection.total_elements(), 472_384);
    assert_eq!(inspection.source_sha256().len(), 64);

    let device = Default::default();
    let (model, loaded) =
        load_feather_hubert_checkpoint::<CpuBackend>(&path, &device).unwrap();
    assert_eq!(loaded.source_sha256(), inspection.source_sha256());
    let waveform = Tensor::from_data(
        TensorData::new(vec![0.0_f32; 1360], [1, 1360]),
        &device,
    );
    let output = model.forward(waveform);
    assert_eq!(output.dims(), [1, 4, 64]);
    assert!(
        output
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );
}
~~~

- [ ] **Step 2: Add failing pure inspection tests before file parsing code.**

Inside feather_hubert.rs add cfg(test) tests over a private BTreeMap<String, TensorFact>. The valid fact builder must generate the exact model tensor set:

- frontend layers 0..6: conv.weight plus norm.weight and norm.bias.
- encoder blocks 0..num_blocks-1: norm.weight, norm.bias, pw_expand.weight, dw_conv.weight, pw_project.weight.
- final_norm.weight, final_norm.bias, proj.weight, proj.bias.

Assert these mutations fail with the named structured errors:

~~~rust
#[test]
fn facts_infer_the_micro_config() {
    let facts = valid_facts(32, 2, 2, 64);
    let config = infer_config(&facts, None).unwrap();
    assert_eq!((config.channels, config.expansion), (32, 2));
    assert_eq!((config.num_blocks, config.output_dim), (2, 64));
    assert_eq!(config.dropout, 0.0);
}

#[test]
fn facts_reject_missing_block_tensor_wrong_shape_dtype_and_extra_key() {
    let mut missing = valid_facts(32, 2, 2, 64);
    missing.remove("encoder.1.dw_conv.weight");
    assert!(matches!(infer_config(&missing, None), Err(WeightImportError::MissingTensor(_))));

    let mut shape = valid_facts(32, 2, 2, 64);
    shape.get_mut("proj.weight").unwrap().shape = vec![64, 31, 1];
    assert!(matches!(infer_config(&shape, None), Err(WeightImportError::ShapeMismatch(_))));

    let mut dtype = valid_facts(32, 2, 2, 64);
    dtype.get_mut("final_norm.weight").unwrap().dtype = burn::tensor::DType::I64;
    assert!(matches!(infer_config(&dtype, None), Err(WeightImportError::DTypeMismatch(_))));

    let mut extra = valid_facts(32, 2, 2, 64);
    extra.insert("unexpected.weight".into(), TensorFact::f32(vec![1]));
    assert!(matches!(infer_config(&extra, None), Err(WeightImportError::UnexpectedTensor(_))));
}
~~~

Also test:

- encoder indices 0 and 2 without 1 -> UnsupportedStructure.
- pw_expand output not divisible by channels -> ShapeMismatch.
- metadata config/args structural value differs from derived facts -> UnsupportedStructure.
- metadata dropout is NaN, negative, or >=1 -> UnsupportedStructure.
- metadata dropout=0.05 is accepted but the returned inference config has dropout=0.0.
- checked tensor-count and total-element arithmetic maps overflow/limit failures to UnsafeLimit.

- [ ] **Step 3: Run the new test and confirm RED.**

~~~powershell
cargo test -p feathertalk-weights --test feather_hubert_checkpoint
~~~

Expected: compilation fails because the public inspection/loading API does not exist.

- [ ] **Step 4: Implement exact structural config inference.**

Promote feathertalk-models from dev-dependencies to dependencies with default-features=false.

For selected raw PyTorch tensor facts require DType::F32 and these exact shapes:

~~~text
frontend.layers.0.conv.weight = [64,1,10]
frontend.layers.1.conv.weight = [128,64,3]
frontend.layers.2.conv.weight = [256,128,3]
frontend.layers.3.conv.weight = [384,256,3]
frontend.layers.4.conv.weight = [channels,384,3]
frontend.layers.5.conv.weight = [channels,channels,2]
frontend.layers.6.conv.weight = [channels,channels,2]
each frontend.layers.N.norm.{weight,bias} = [that layer output channels]

encoder.N.norm.{weight,bias} = [channels]
encoder.N.pw_expand.weight = [channels*expansion,channels,1]
encoder.N.dw_conv.weight = [channels*expansion,1,5]
encoder.N.pw_project.weight = [channels,channels*expansion,1]

final_norm.{weight,bias} = [channels]
proj.weight = [output_dim,channels,1]
proj.bias = [output_dim]
~~~

Derive channels and output_dim from proj.weight, expansion from encoder.0.pw_expand.weight, and num_blocks from the contiguous encoder index set. Build the expected key set from those values and reject its symmetric difference with the actual selected tensor set.

- [ ] **Step 5: Implement restricted metadata and state-dict selection.**

Make legacy::select_top_level_key pub(crate) without changing its existing selection order:

1. explicitly requested key;
2. model;
3. state_dict;
4. direct dictionary.

Use burn_store::pytorch::PytorchReader only on the temporary SnapshotFile copy. For optional root config and args dictionaries:

- Accept channels, expansion, num_blocks, and output_dim only as positive Int values fitting usize.
- Accept dropout as Int or Float only when finite and 0.0 <= value < 1.0.
- If config or args exists, require all five fields in that dictionary.
- If both exist, require all five values to match.
- Require the four structural values to equal the tensor-derived values.
- Set the returned FeatherHubertConfig.dropout to 0.0 after validation.
- Ignore unrelated primitive training metadata such as epoch/train_loss/val_loss.
- Never deserialize or invoke arbitrary Python callables.

- [ ] **Step 6: Implement inspect and load using the existing importer.**

The public values are:

~~~rust
#[derive(Debug, Clone)]
pub struct FeatherHubertCheckpoint {
    config: FeatherHubertConfig,
    source_sha256: String,
    tensor_count: usize,
    total_elements: u64,
}
~~~

inspect_feather_hubert_checkpoint must snapshot/hash the source, select only the state dict, materialize tensor metadata, infer/validate config, count tensors/elements with checked arithmetic, and return the value.

load_feather_hubert_checkpoint must:

1. Call inspect_feather_hubert_checkpoint.
2. Initialize checkpoint.config().clone().init(device).
3. Call import_into with LegacyModelKind::FeatherHubert and default limits.
4. Require ImportReport source_sha256, tensor_count, and total_elements to equal inspection.
5. Return the loaded model and inspection.

Do not add a second tensor-application loop.

- [ ] **Step 7: Verify GREEN and all weight-import tests.**

~~~powershell
cargo test -p feathertalk-weights --test feather_hubert_checkpoint
cargo test -p feathertalk-weights --all-targets
cargo fmt --all -- --check
cargo clippy -p feathertalk-weights --all-targets -- -D warnings
git diff --check
~~~

Expected: all commands exit 0, including existing strict legacy/PFLD tests.

- [ ] **Step 8: Commit the checkpoint loader.**

~~~powershell
git add rust/crates/feathertalk-weights/Cargo.toml rust/crates/feathertalk-weights/src/feather_hubert.rs rust/crates/feathertalk-weights/src/legacy.rs rust/crates/feathertalk-weights/src/lib.rs rust/crates/feathertalk-weights/src/error.rs rust/crates/feathertalk-weights/tests/feather_hubert_checkpoint.rs
git commit -m "feat: load FeatherHuBERT checkpoints"
~~~

If error.rs did not change, omit it from git add.

### Task 5: Golden parity routing and user checkpoint read-only CPU acceptance

**Files:**

- Modify: rust/crates/feathertalk-parity/src/fixture.rs
- Create: rust/crates/feathertalk-weights/tests/feather_hubert_real_checkpoint.rs

**Interfaces:**

- Routes ForwardCase::FeatherMicro CPU parity through load_feather_hubert_checkpoint.
- Keeps Original UNet parity on LegacyImportRequest/import_into.
- Adds an environment-gated test for the exact user checkpoint baseline.

- [ ] **Step 1: Write the failing real-checkpoint test.**

Create feather_hubert_real_checkpoint.rs:

~~~rust
use std::path::PathBuf;

use burn::tensor::{Tensor, TensorData};
use feathertalk_models::backend::CpuBackend;
use feathertalk_weights::load_feather_hubert_checkpoint;

const EXPECTED_BYTES: u64 = 40_436_613;
const EXPECTED_SHA256: &str =
    "58df96af118d75d7f69da441e1f3960096f28dda637a4e8f4265f108d27aeb52";

#[test]
fn configured_user_checkpoint_loads_and_runs_on_cpu_without_writes() {
    let Some(path) = std::env::var_os("FEATHERTALK_FEATHER_HUBERT_CHECKPOINT") else {
        eprintln!("FEATHERTALK_FEATHER_HUBERT_CHECKPOINT is not set; skipping local model");
        return;
    };
    let path = PathBuf::from(path);
    assert!(path.is_absolute());
    let before = std::fs::metadata(&path).unwrap();
    assert_eq!(before.len(), EXPECTED_BYTES);

    let device = Default::default();
    let (model, checkpoint) =
        load_feather_hubert_checkpoint::<CpuBackend>(&path, &device).unwrap();
    assert_eq!(checkpoint.source_sha256(), EXPECTED_SHA256);
    assert_eq!(checkpoint.config().channels, 256);
    assert_eq!(checkpoint.config().expansion, 2);
    assert_eq!(checkpoint.config().num_blocks, 8);
    assert_eq!(checkpoint.config().output_dim, 1024);
    assert_eq!(checkpoint.config().dropout, 0.0);
    assert_eq!(checkpoint.tensor_count(), 65);
    assert_eq!(checkpoint.total_elements(), 3_364_096);

    let samples = (0..1360)
        .map(|index| (index as f32 - 680.0) / 680.0)
        .collect::<Vec<_>>();
    let waveform = Tensor::from_data(TensorData::new(samples, [1, 1360]), &device);
    let output = model.forward(waveform);
    assert_eq!(output.dims(), [1, 4, 1024]);
    let values = output.into_data().to_vec::<f32>().unwrap();
    assert!(values.iter().all(|value| value.is_finite()));

    let after = std::fs::metadata(&path).unwrap();
    assert_eq!(after.len(), before.len());
    assert_eq!(checkpoint.source_sha256(), EXPECTED_SHA256);
}
~~~

The test must not enumerate or open any sibling file.

- [ ] **Step 2: Route committed golden CPU parity through the new loader.**

In run_cpu_forward, keep validation/extraction unchanged. In the FeatherMicro arm replace manual parity_micro initialization plus import_into with:

~~~rust
let (model, checkpoint) =
    load_feather_hubert_checkpoint::<CpuBackend>(
        &extracted.join(case.weights_entry()),
        &device,
    )?;
if checkpoint.config().output_dim != FEATHER_OUTPUT_SHAPE[2] {
    return Err(config_mismatch(
        case,
        &FEATHER_OUTPUT_SHAPE[2],
        &checkpoint.config().output_dim,
    ));
}
~~~

Keep the Original UNet arm using LegacyImportRequest. Adjust imports and request construction so clippy reports no unused values.

- [ ] **Step 3: Run the parity test first and confirm the new path remains numerically green.**

~~~powershell
cargo test -p feathertalk-parity --test cpu_parity feather_micro_matches_python_on_cpu -- --nocapture
~~~

Expected: exits 0 and reports FeatherMicro max_abs within the existing 1e-4 gate.

- [ ] **Step 4: Explicitly run the user checkpoint test.**

From rust/:

~~~powershell
$env:FEATHERTALK_FEATHER_HUBERT_CHECKPOINT = (Resolve-Path -LiteralPath '../demo/kanghui_training_video_featherhubert_188_latest/feather_hubert_188_latest_99.pth').Path
cargo test -p feathertalk-weights --test feather_hubert_real_checkpoint -- --nocapture
Remove-Item Env:FEATHERTALK_FEATHER_HUBERT_CHECKPOINT
~~~

When running inside .worktrees/burn-inference-adapters/rust, set the variable to the absolute main-workspace path E:/workspace/github/FeatherTalk/demo/kanghui_training_video_featherhubert_188_latest/feather_hubert_188_latest_99.pth instead of using ../demo.

Expected:

- byte count 40,436,613;
- SHA-256 58df96af118d75d7f69da441e1f3960096f28dda637a4e8f4265f108d27aeb52;
- inferred config 256/2/8/1024 with inference dropout 0.0;
- 65 tensors and 3,364,096 elements;
- CPU output [1,4,1024], all finite;
- no file is created in the user model directory.

- [ ] **Step 5: Run focused static checks and inspect protected paths.**

~~~powershell
cargo test -p feathertalk-parity --test cpu_parity feather_micro_matches_python_on_cpu
cargo test -p feathertalk-weights --test feather_hubert_real_checkpoint
cargo fmt --all -- --check
cargo clippy -p feathertalk-parity -p feathertalk-weights --all-targets -- -D warnings
git diff --check
git status --short
~~~

With the environment variable absent, the real-checkpoint test exits successfully after printing its explicit local-skip message. Confirm no demo path is staged or modified.

- [ ] **Step 6: Commit acceptance routing and test.**

~~~powershell
git add rust/crates/feathertalk-parity/src/fixture.rs rust/crates/feathertalk-weights/tests/feather_hubert_real_checkpoint.rs
git commit -m "test: verify FeatherHuBERT checkpoint loading"
~~~

### Task 6: Full verification, merge, cleanup, and automatic milestone continuation

**Files:**

- No new production files.
- Inspect the implementation commits and protected demo status.

- [ ] **Step 1: Run the complete slice verification from the implementation worktree rust/ directory.**

~~~powershell
cargo test -p feathertalk-models --all-targets
cargo test -p feathertalk-models --doc
cargo test -p feathertalk-inference --all-targets
cargo test -p feathertalk-weights --all-targets
cargo test -p feathertalk-parity --all-targets
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
git diff --check
~~~

Every command must exit 0. Do not infer workspace health from a partial crate run.

- [ ] **Step 2: Re-run the user checkpoint test with the explicit absolute path after all code is final.**

~~~powershell
$env:FEATHERTALK_FEATHER_HUBERT_CHECKPOINT = 'E:/workspace/github/FeatherTalk/demo/kanghui_training_video_featherhubert_188_latest/feather_hubert_188_latest_99.pth'
cargo test -p feathertalk-weights --test feather_hubert_real_checkpoint -- --nocapture
Remove-Item Env:FEATHERTALK_FEATHER_HUBERT_CHECKPOINT
~~~

Record the fresh exit code and the asserted config/count/shape evidence.

- [ ] **Step 3: Inspect commits and protected-directory state.**

~~~powershell
git log --oneline --decorate main..HEAD
git status --short --branch
git diff --stat main...HEAD
git -C E:/workspace/github/FeatherTalk status --short --branch
~~~

Confirm:

- implementation worktree is clean;
- main workspace has only the untracked user demo directory outside committed work;
- no commit contains the demo directory;
- the dependency graph has no models -> weights edge.

- [ ] **Step 4: Use finishing-a-development-branch and merge with the standing recommended choice.**

Run the finishing-a-development-branch skill. Because the user has already authorized the recommended option and automatic continuation, select local fast-forward merge into main, subject to its fresh verification gate.

After fast-forward merge, run from E:/workspace/github/FeatherTalk/rust:

~~~powershell
cargo test --workspace --all-targets
git status --short --branch
~~~

Only claim the slice complete if the merged workspace command exits 0.

- [ ] **Step 5: Remove only this implementation worktree and branch after verifying their exact paths.**

Verify the resolved target is E:/workspace/github/FeatherTalk/.worktrees/burn-inference-adapters, then remove the linked worktree and delete branch burn-inference-adapters. Do not touch the three pre-existing worktrees:

- .worktrees/frame-face-pipeline
- .worktrees/media-normalization-execution
- .worktrees/pfld-burn-inference

- [ ] **Step 6: Continue automatically to the next milestone-four slice.**

Re-read docs/superpowers/specs/2026-08-17-rust-desktop-migration-design.md and the Burn adapter design. Start a new brainstorming/spec/plan cycle for the complete offline video executor:

- controlled reads of frame, landmark, bbox, and FeatherHuBERT feature artifacts;
- loop over RenderPlan;
- select source frame and bbox;
- call render_planned_frame;
- write raw BGR frames to FFmpeg stdin;
- stage output and atomically publish only after successful process completion;
- leave standard model packages, ONNX opset 17, and legacy model/feature migration CLI as later independent milestone-four slices.

Do not stop merely because the Burn adapter slice merged successfully.

## Plan Self-Review

- Spec coverage: Tasks 1-5 cover TalkingHeadModel, loaded FeatherHuBERT adapters, exact two-token audio windows, Burn image/audio tensors, strict output shape/finite/range validation, side-effect-free planned rendering, restricted checkpoint inference/loading, committed golden parity, and the exact user checkpoint CPU test.
- Dependency consistency: models has no weights dependency; weights owns checkpoint files and depends on models; inference depends on audio/models but never weights.
- Type consistency: all signatures use existing BgrFrame, UnetImageInput, RenderGeometry, InferenceFramePlan, FeatureMatrix, FaceBoundingBox, FeatherHubertEncoder, and Burn Backend types.
- Fixture consistency: the generic loader accepts the committed 32/2/2/64 micro checkpoint, while product wiring and the user checkpoint assert 256/2/8/1024.
- TDD coverage: each production group has an explicit failing command before implementation and focused green commands afterward.
- Placeholder scan: no TBD, TODO, “similar to”, unspecified error handling, or deferred implementation step remains.
- Protected data: the model is accessed only through an explicit absolute environment variable; the sibling MOV is never opened; the demo directory is never staged.
- Scope: full video execution, FFmpeg process lifecycle, packaging, ONNX, CLI, worker, and GPUI remain outside this plan.
