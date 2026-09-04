#![allow(dead_code)]

use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use burn::{
    module::Module,
    tensor::{Tensor, backend::Backend},
};
use feathertalk_domain::{
    Progress, TaskStage, TrainParams, TrainingMode as DomainTrainingMode, UnetVariant,
};
use feathertalk_export::ModelConfiguration;
use feathertalk_media::CancellationToken;
use feathertalk_models::unet::{OriginalUnet, OriginalUnetConfig};
use feathertalk_training::{
    PerceptualFeatureExtractor, TrainingDataset, TrainingError, TrainingSample,
};
use feathertalk_training_data::{FrameSample, TrainingItem};
use feathertalk_worker::{
    TaskReporter, TrainBackend, TrainDevice, TrainingPaths, TrainingPlan, checkpoint_descriptor,
    training_config,
};

/// A 160x160 forward plus backward through burn's autodiff graph overruns the
/// default 2 MiB libtest stack in a debug build and takes the whole binary down
/// with `STATUS_STACK_OVERFLOW`. `feathertalk-training-run/tests/support/mod.rs`
/// solves it the same way, and Task 4 gives the worker's own execution thread
/// the same stack.
const STEP_STACK_BYTES: usize = 64 * 1024 * 1024;

/// Runs `body` on a thread whose stack is large enough for a training step.
/// Panics travel back through `join`, so failed assertions still fail the test.
pub fn on_step_stack(name: &str, body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name(name.to_owned())
        .stack_size(STEP_STACK_BYTES)
        .spawn(body)
        .expect("the step thread starts")
        .join()
        .expect("the step thread does not panic");
}

/// The perceptual term with the weights taken out: it compares the images
/// themselves, which keeps the loss finite and the test independent of VGG19.
#[derive(Debug, Clone, Copy)]
pub struct IdentityExtractor;

impl<B: Backend> PerceptualFeatureExtractor<B> for IdentityExtractor {
    fn forward(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        image
    }
}

/// Behaves like `IdentityExtractor` until the given call, then poisons the loss.
///
/// One step calls the extractor more than once, so the threshold is counted in
/// calls rather than steps; the tests only need "not the first step".
#[derive(Debug)]
pub struct PoisonedExtractor {
    calls: Cell<usize>,
    poison_from: usize,
}

impl PoisonedExtractor {
    pub fn after(calls: usize) -> Self {
        Self {
            calls: Cell::new(0),
            poison_from: calls,
        }
    }
}

impl<B: Backend> PerceptualFeatureExtractor<B> for PoisonedExtractor {
    fn forward(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        let seen = self.calls.get().saturating_add(1);
        self.calls.set(seen);
        if seen > self.poison_from {
            return image.mul_scalar(f32::NAN);
        }
        image
    }
}

/// The micro model with every parameter already materialised.
///
/// `fork` pushes each `Param` through `val()`; without it a clone would copy the
/// lazy initialiser instead of the weights, and two clones would draw different
/// numbers (burn-core 0.21 `module/param/base.rs`).
pub fn model(device: &TrainDevice) -> OriginalUnet<TrainBackend> {
    OriginalUnetConfig::parity_micro()
        .init::<TrainBackend>(device)
        .fork(device)
}

/// A dataset that synthesises every sample, so the loop can be driven without a
/// locked project on disk. This is what Task 1 opened `FrameSample::new` for.
pub struct StubDataset {
    frame_count: u64,
}

impl StubDataset {
    pub fn new(frame_count: u64) -> Self {
        Self { frame_count }
    }
}

impl TrainingDataset for StubDataset {
    type Item = TrainingItem;

    fn frame_count(&self) -> u64 {
        self.frame_count
    }

    fn load_sample(&self, sample: &TrainingSample) -> Result<Self::Item, TrainingError> {
        Ok(match sample {
            TrainingSample::SingleFrame { target_index, .. } => {
                TrainingItem::SingleFrame(frame(*target_index)?)
            }
            TrainingSample::TemporalPair {
                first_target_index,
                second_target_index,
                ..
            } => TrainingItem::TemporalPair {
                first: frame(*first_target_index)?,
                second: frame(*second_target_index)?,
            },
        })
    }
}

/// Flat planes whose value follows the frame index: enough for a finite loss and
/// for two frames to differ, cheap enough to allocate per sample.
fn frame(index: u64) -> Result<FrameSample, TrainingError> {
    let value = (index % 7) as f32 / 7.0;
    Ok(FrameSample::new(
        vec![value; 6 * 160 * 160],
        vec![value; 16 * 32 * 32],
        vec![value; 3 * 160 * 160],
        vec![1.0; 160 * 160],
    )?)
}

/// Records every event, and can cancel a token once enough have arrived.
pub struct Recorder {
    events: Mutex<Vec<(TaskStage, Option<Progress>)>>,
    cancel_after: Option<(usize, CancellationToken)>,
}

impl Recorder {
    pub fn new() -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            cancel_after: None,
        }
    }

    /// Cancels `token` once `events` events have been reported, which is how a
    /// test interrupts a run at a known step instead of at a known time.
    pub fn cancelling_after(events: usize, token: CancellationToken) -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            cancel_after: Some((events, token)),
        }
    }

    pub fn events(&self) -> Vec<(TaskStage, Option<Progress>)> {
        self.events.lock().expect("the recorder is intact").clone()
    }
}

impl TaskReporter for Recorder {
    fn report(&self, stage: TaskStage, progress: Option<Progress>) {
        let mut events = self.events.lock().expect("the recorder is intact");
        events.push((stage, progress));
        if let Some((limit, token)) = &self.cancel_after
            && events.len() >= *limit
        {
            token.cancel();
        }
    }
}

/// The plan a micro run trains under: `parity_micro`, batch size 1, whatever
/// mode and epoch count the test asks for.
pub fn micro_plan(
    project_dir: &Path,
    mode: DomainTrainingMode,
    epochs: u32,
    frame_count: u64,
    resume_from: Option<PathBuf>,
) -> TrainingPlan {
    let params = TrainParams {
        project_dir: project_dir.to_path_buf(),
        mode,
        variant: UnetVariant::OriginalUnet,
        epochs,
        resume: resume_from.is_some(),
    };
    let configuration = ModelConfiguration::original_unet(&OriginalUnetConfig::parity_micro());
    TrainingPlan {
        mode,
        variant: UnetVariant::OriginalUnet,
        epochs_requested: epochs,
        frame_count,
        config: training_config(&params),
        descriptor: checkpoint_descriptor(&configuration).expect("the configuration serialises"),
        paths: TrainingPaths::new(project_dir),
        resume_from,
    }
}
