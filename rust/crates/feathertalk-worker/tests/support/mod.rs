#![allow(dead_code)]

use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use burn::{
    module::Module,
    tensor::{Tensor, backend::Backend},
};
use feathertalk_audio::{FeatureMatrix, write_feature_file};
use feathertalk_domain::{
    Progress, TaskStage, TrainParams, TrainingMode as DomainTrainingMode, UnetVariant,
};
use feathertalk_export::ModelConfiguration;
use feathertalk_inference::{
    BgrFrame, CommandSpec, FrameReader, InferenceError, RawVideoSink, RawVideoSinkFactory,
};
use feathertalk_media::CancellationToken;
use feathertalk_models::unet::{OriginalUnet, OriginalUnetConfig};
use feathertalk_training::{
    PerceptualFeatureExtractor, TrainingDataset, TrainingError, TrainingSample,
};
use feathertalk_training_data::{FrameSample, TrainingItem};
use feathertalk_worker::{
    RenderBackend, RenderDevice, TaskReporter, TrainBackend, TrainDevice, TrainingPaths,
    TrainingPlan, checkpoint_descriptor, project_assets, training_config,
};
use tempfile::TempDir;

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

/// The three inference inputs a locked project holds, plus a placeholder for the
/// audio a render request names.
///
/// Ported from `feathertalk-inference/tests/executor.rs`, but laid out at the
/// paths `project_assets` resolves, so the worker's own assembly finds them.
/// Returns the temporary root and the project directory inside it, which lets a
/// test put its output next to the project instead of inside it.
pub fn render_tree(frame_count: usize, feature_frames: usize) -> (TempDir, PathBuf) {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let project = root.path().join("project");
    let assets = project_assets(&project);
    std::fs::create_dir_all(&assets.frame_dir).expect("the frame directory is created");
    std::fs::create_dir_all(&assets.landmark_dir).expect("the landmark directory is created");
    for index in 0..frame_count {
        std::fs::write(assets.frame_dir.join(format!("{index:06}.jpg")), b"fixture")
            .expect("the frame is written");
        let mut landmarks = String::new();
        for point in 0..110 {
            // Point 31 is the one the crop geometry measures; 168 keeps the crop
            // inside the frame `StubFrameReader` hands back.
            let x = if point == 31 { 168 } else { 0 };
            landmarks.push_str(&format!("{x} 0\n"));
        }
        std::fs::write(
            assets.landmark_dir.join(format!("{index:06}.lms")),
            landmarks,
        )
        .expect("the landmarks are written");
    }
    if let Some(parent) = assets.feature_path.parent() {
        std::fs::create_dir_all(parent).expect("the feature directory is created");
    }
    // Two tokens per frame, which is the ratio the feature extractor writes.
    let tokens = feature_frames * 2;
    let features =
        FeatureMatrix::new(tokens, 1024, vec![0.0; tokens * 1024]).expect("the features are valid");
    write_feature_file(&assets.feature_path, &features).expect("the features are written");
    std::fs::write(render_audio(&project), b"audio").expect("the audio placeholder is written");
    (root, project)
}

/// The audio file `render_tree` writes. The sink is a stub, so nothing reads it;
/// the request only needs a path that exists.
pub fn render_audio(project_dir: &Path) -> PathBuf {
    project_dir.join("audio.wav")
}

/// Hands back a 168x168 frame for every index and remembers which indexes the
/// render asked for.
///
/// The size is not arbitrary: the crop the fixture's landmarks describe has to
/// fit inside the frame, or the paste would fall outside it.
#[derive(Debug, Default)]
pub struct StubFrameReader {
    pub frames: Mutex<Vec<usize>>,
}

impl FrameReader for StubFrameReader {
    fn read(&self, index: usize, _path: &Path) -> Result<BgrFrame, InferenceError> {
        const SIDE: u32 = 168;

        self.frames
            .lock()
            .expect("the reader is intact")
            .push(index);
        // A value that follows the index, so an all-zero frame would be visible.
        let value = (index as u8).wrapping_add(1);
        BgrFrame::new(SIDE, SIDE, vec![value; (SIDE * SIDE * 3) as usize])
    }
}

/// A sink that keeps the frame sizes it was handed and writes a placeholder to
/// the staging path, so the executor's atomic publish has a file to rename.
///
/// This is what makes a real `execute_offline_render` runnable in a unit test
/// without ffmpeg on the machine.
#[derive(Debug, Default)]
pub struct MemorySinkFactory {
    pub frames: Mutex<Vec<usize>>,
    pub staging: Mutex<Option<PathBuf>>,
}

struct MemorySink<'a> {
    factory: &'a MemorySinkFactory,
}

impl RawVideoSink for MemorySink<'_> {
    fn write_frame(&mut self, frame: &BgrFrame) -> Result<(), InferenceError> {
        self.factory
            .frames
            .lock()
            .expect("the sink is intact")
            .push(frame.as_bytes().len());
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<(), InferenceError> {
        let staging = self.factory.staging.lock().expect("the sink is intact");
        let path = staging.as_ref().expect("the sink knows its staging path");
        std::fs::write(path, b"rendered-video").expect("the staging file is written");
        Ok(())
    }
}

impl RawVideoSinkFactory for MemorySinkFactory {
    fn start(&self, command: &CommandSpec) -> Result<Box<dyn RawVideoSink + '_>, InferenceError> {
        // ffmpeg's last argument is the file it writes, which is the staging
        // path the executor renames once the render succeeds.
        let staging = command
            .arguments()
            .last()
            .expect("the command names an output file")
            .clone();
        *self.staging.lock().expect("the factory is intact") = Some(PathBuf::from(staging));
        Ok(Box::new(MemorySink { factory: self }))
    }
}

/// The micro model a render test runs, with every parameter materialised.
///
/// `parity_micro` rather than `production`: the shapes are the ones inference
/// requires, and the parameter count is small enough for a unit test.
pub fn render_model(device: &RenderDevice) -> OriginalUnet<RenderBackend> {
    OriginalUnetConfig::parity_micro()
        .init::<RenderBackend>(device)
        .fork(device)
}
