//! The training loop and the command that drives it.

use std::path::PathBuf;
use std::time::Instant;

use burn::{
    module::AutodiffModule,
    optim::{AdamConfig, Optimizer},
};
use feathertalk_domain::{
    Progress, TaskError, TaskStage, TrainParams, TrainingMode as DomainTrainingMode, UnetVariant,
};
use feathertalk_export::ModelConfiguration;
use feathertalk_media::CancellationToken;
use feathertalk_models::unet::{MobileOneUnetConfig, OriginalUnetConfig, TrainableTalkingHead};
use feathertalk_training::{
    CheckpointCompatibility, PerceptualFeatureExtractor, TrainingDataset, TrainingError,
    load_training_checkpoint, load_vgg19_package,
};
use feathertalk_training_data::{ProjectTrainingDataset, TrainingItem};
use feathertalk_training_run::{StepReport, TrainingRunner, build_preview_artifact};

use crate::admission::{check_project_dir, invalid_request};
use crate::{
    CommandOutcome, MAX_EPOCHS, TRAINING_SEED, TaskReporter, TrainBackend, TrainDevice,
    TrainSummary, TrainingPaths, TrainingPlan, TrainingToolchain, WORKER_STATE,
    checkpoint_descriptor, latest_checkpoint, preview_sample, publish_checkpoint, sample_count,
    train_to_json, training_config, training_data_task_error, training_task_error,
    write_metrics_unless_present, write_preview_unless_present,
};

/// Trains until the plan's epoch count is reached, the task is cancelled, or a
/// step fails.
///
/// Everything the loop needs arrives as a parameter, which is what lets the
/// tests drive it with a stub dataset and a constant extractor instead of a
/// locked project and half a gigabyte of VGG19 weights.
#[allow(clippy::too_many_arguments)]
pub fn run_training<M, O, D, E>(
    plan: &TrainingPlan,
    dataset: D,
    model: M,
    optimizer: O,
    extractor: &E,
    device: &TrainDevice,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
) -> CommandOutcome
where
    M: TrainableTalkingHead<TrainBackend> + AutodiffModule<TrainBackend> + Clone,
    O: Optimizer<M, TrainBackend> + Clone,
    D: TrainingDataset<Item = TrainingItem>,
    E: PerceptualFeatureExtractor<TrainBackend>,
{
    let mut runner = match build_runner(plan, dataset, model, optimizer, device) {
        Ok(runner) => runner,
        Err(error) => {
            return CommandOutcome::Failed(training_task_error(&error, TaskStage::Preparing));
        }
    };

    // The clock starts here, not at admission: `restore` zeroes `samples_seen`,
    // and the throughput this feeds has to describe the run that is starting.
    let started = Instant::now();
    let total = total_steps(plan);
    let mut published = Published::default();
    let mut stage = TaskStage::Preparing;
    let mut steps: u64 = 0;
    let mut total_loss = None;

    while !runner.is_finished() {
        if token.is_cancelled() {
            // With no step behind it, the checkpoint already on disk is the best
            // one there is and republishing would only copy it.
            if steps > 0
                && let Err(error) = publish(&runner, plan, runner.global_step(), &mut published)
            {
                return CommandOutcome::Failed(training_task_error(&error, stage));
            }
            return CommandOutcome::Cancelled;
        }

        let report = match runner.step(extractor) {
            Ok(report) => report,
            Err(error) => return CommandOutcome::Failed(training_task_error(&error, stage)),
        };
        steps = steps.saturating_add(1);
        total_loss = Some(report.losses.total);
        stage = TaskStage::Training {
            epoch: u32::try_from(report.epoch).unwrap_or(u32::MAX),
            step: report.global_step,
            loss: report.losses.total,
        };
        reporter.report(stage.clone(), Some(progress(report.global_step, total)));

        // `report.epoch` is the epoch the batch came from, `runner.epoch()` is
        // where the loader stands now, so they differ exactly once per epoch.
        if runner.epoch() > report.epoch {
            let closed = close_epoch(&runner, plan, &report, started, device, &mut published);
            if let Err(error) = closed {
                return CommandOutcome::Failed(training_task_error(&error, stage));
            }
        }
    }

    let summary = TrainSummary {
        mode: plan.mode,
        variant: plan.variant,
        descriptor: &plan.descriptor,
        frame_count: plan.frame_count,
        epochs_requested: plan.epochs_requested,
        epochs_completed: runner.epoch(),
        global_step: runner.global_step(),
        samples_seen: runner.samples_seen(),
        total_loss,
        resumed_from: plan.resume_from.as_deref(),
        checkpoint_dir: published.latest.as_deref(),
        checkpoints_written: published.checkpoints,
        metrics_written: published.metrics,
        previews_written: published.previews,
    };
    CommandOutcome::Completed(Some(train_to_json(&summary)))
}

/// What this run has put on disk.
#[derive(Debug, Default)]
struct Published {
    checkpoints: u64,
    metrics: u64,
    previews: u64,
    latest: Option<PathBuf>,
}

/// Starts a fresh run, or continues from the checkpoint the plan names.
fn build_runner<M, O, D>(
    plan: &TrainingPlan,
    dataset: D,
    model: M,
    optimizer: O,
    device: &TrainDevice,
) -> Result<TrainingRunner<TrainBackend, M, O, D>, TrainingError>
where
    M: TrainableTalkingHead<TrainBackend> + AutodiffModule<TrainBackend> + Clone,
    O: Optimizer<M, TrainBackend> + Clone,
    D: TrainingDataset<Item = TrainingItem>,
{
    let Some(directory) = plan.resume_from.as_deref() else {
        return TrainingRunner::new(
            dataset,
            model,
            optimizer,
            plan.config.clone(),
            TRAINING_SEED,
            *device,
        );
    };

    // The model and the optimizer go in as templates: the loader reads the
    // records into their shapes and hands the restored pair back.
    let expected = CheckpointCompatibility::new(
        plan.descriptor.clone(),
        plan.config.clone(),
        plan.frame_count,
    );
    let restored = load_training_checkpoint::<TrainBackend, M, O>(
        directory, &model, &optimizer, device, &expected,
    )?;
    TrainingRunner::restore(dataset, restored, *device)
}

/// Publishes a checkpoint for `global_step` and remembers it as the newest one.
fn publish<M, O, D>(
    runner: &TrainingRunner<TrainBackend, M, O, D>,
    plan: &TrainingPlan,
    global_step: u64,
    published: &mut Published,
) -> Result<(), TrainingError>
where
    M: TrainableTalkingHead<TrainBackend> + AutodiffModule<TrainBackend> + Clone,
    O: Optimizer<M, TrainBackend> + Clone,
    D: TrainingDataset<Item = TrainingItem>,
{
    let checkpoint = publish_checkpoint(&plan.paths, global_step, |staged| {
        runner
            .save_checkpoint(staged, plan.descriptor.clone())
            .map(|_| ())
    })?;
    published.checkpoints = published.checkpoints.saturating_add(1);
    published.latest = Some(checkpoint);
    Ok(())
}

/// What an epoch boundary owes: the checkpoint first, then the two diagnostics.
fn close_epoch<M, O, D>(
    runner: &TrainingRunner<TrainBackend, M, O, D>,
    plan: &TrainingPlan,
    report: &StepReport,
    started: Instant,
    device: &TrainDevice,
    published: &mut Published,
) -> Result<(), TrainingError>
where
    M: TrainableTalkingHead<TrainBackend> + AutodiffModule<TrainBackend> + Clone,
    O: Optimizer<M, TrainBackend> + Clone,
    D: TrainingDataset<Item = TrainingItem>,
{
    publish(runner, plan, report.global_step, published)?;

    // `None` for GPU memory: the CPU backend has none to report (design 4).
    let metrics = runner.metrics(report, started.elapsed(), None, WORKER_STATE)?;
    if write_metrics_unless_present(&plan.paths, report.global_step, &metrics)? {
        published.metrics = published.metrics.saturating_add(1);
    }

    let artifact = build_preview_artifact::<TrainBackend, M, D>(
        runner.model()?,
        runner.dataset(),
        device,
        &preview_sample(plan.frame_count),
        report.epoch,
        report.global_step,
        &plan.descriptor.model_kind,
        &plan.descriptor.model_config_sha256,
        WORKER_STATE,
    )?;
    if write_preview_unless_present(&plan.paths, report.global_step, &artifact)? {
        published.previews = published.previews.saturating_add(1);
    }
    Ok(())
}

/// The number of steps the whole lineage will reach, or `None` when it does not
/// fit in a `u64` -- in which case progress reports what it has done and no
/// total, which is what `Progress.total` being optional is for.
fn total_steps(plan: &TrainingPlan) -> Option<u64> {
    let samples = sample_count(plan.mode, plan.frame_count);
    // `TrainingConfig::validate` rejects a zero batch size, but a divide here
    // must not be the place that finds out.
    let steps_per_epoch = samples.div_ceil(plan.config.batch_size.max(1));
    plan.config.total_epochs.checked_mul(steps_per_epoch)
}

/// A resumed run keeps the lineage's step numbers, so `completed` is clamped
/// rather than trusted: if the loader and the total ever disagree, a total that
/// is reached early beats a progress bar past one hundred percent.
fn progress(global_step: u64, total: Option<u64>) -> Progress {
    Progress {
        completed: match total {
            Some(total) => global_step.min(total),
            None => global_step,
        },
        total,
    }
}

/// Trains the U-Net of a locked project.
///
/// The toolchain arrives by reference instead of being read from the
/// environment here: `commands.rs` already holds the validated config, and a
/// command that reads its own environment cannot be tested.
pub fn execute_train(
    params: &TrainParams,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    toolchain: &TrainingToolchain,
) -> CommandOutcome {
    // Admission reads the asset manifest, the whole feature file and then half a
    // gigabyte of VGG19 weights, so the stage goes out before any of it.
    reporter.report(TaskStage::Preparing, None);
    if let Err(error) = check_project_dir(&params.project_dir) {
        return CommandOutcome::Failed(error);
    }
    if let Err(error) = check_epochs(params.epochs) {
        return CommandOutcome::Failed(error);
    }

    // The variant is a type rather than a value, so each arm monomorphises the
    // whole run for one model. This is the only place that branches on it.
    match params.variant {
        UnetVariant::OriginalUnet => {
            let configuration = OriginalUnetConfig::production();
            let described = ModelConfiguration::original_unet(&configuration);
            start(params, token, reporter, toolchain, described, |device| {
                configuration.init::<TrainBackend>(device)
            })
        }
        UnetVariant::MobileOneUnet => {
            let configuration = MobileOneUnetConfig::production();
            // Not reparameterized: training needs the multi-branch graph, and
            // fusing the branches is an export-time step.
            let described = ModelConfiguration::mobileone_unet(&configuration, false);
            start(params, token, reporter, toolchain, described, |device| {
                configuration.init::<TrainBackend>(device)
            })
        }
    }
}

/// The rest of the command, once the model type is known.
fn start<M, F>(
    params: &TrainParams,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    toolchain: &TrainingToolchain,
    configuration: ModelConfiguration,
    init: F,
) -> CommandOutcome
where
    M: TrainableTalkingHead<TrainBackend> + AutodiffModule<TrainBackend> + Clone,
    F: FnOnce(&TrainDevice) -> M,
{
    // Deliberately never named: writing out `ProjectTrainingDataset<JpegFrameReader>`
    // would pull `feathertalk-inference` into this crate's dependencies for the
    // sake of one type annotation.
    let dataset = match ProjectTrainingDataset::open(&params.project_dir) {
        Ok(dataset) => dataset,
        Err(error) => return CommandOutcome::Failed(training_data_task_error(&error)),
    };
    let frame_count = dataset.frame_count();
    if let Err(error) = check_frame_count(params.mode, frame_count) {
        return CommandOutcome::Failed(error);
    }

    let paths = TrainingPaths::new(&params.project_dir);
    let found = match latest_checkpoint(&paths) {
        Ok(found) => found,
        Err(error) => return failed_preparing(&error),
    };
    if params.resume && found.is_none() {
        return CommandOutcome::Failed(invalid_request(
            "未找到可续训的检查点",
            format!("no checkpoint under {}", paths.checkpoints().display()),
        ));
    }
    let descriptor = match checkpoint_descriptor(&configuration) {
        Ok(descriptor) => descriptor,
        Err(error) => return failed_preparing(&error),
    };
    let plan = TrainingPlan {
        mode: params.mode,
        variant: params.variant,
        epochs_requested: params.epochs,
        frame_count,
        config: training_config(params),
        descriptor,
        paths,
        // Without `--resume`, a checkpoint on disk is not continued: the run
        // starts from fresh weights and republishes over the old names.
        resume_from: if params.resume { found } else { None },
    };

    let device = TrainDevice::default();
    let extractor = match load_vgg19_package::<TrainBackend>(toolchain.vgg19_dir(), &device) {
        Ok(extractor) => extractor,
        Err(error) => return failed_preparing(&error),
    };
    // Loading the weights took seconds; the caller may have given up meanwhile.
    if token.is_cancelled() {
        return CommandOutcome::Cancelled;
    }

    let optimizer = AdamConfig::new().init::<TrainBackend, M>();
    run_training(
        &plan,
        dataset,
        init(&device),
        optimizer,
        &extractor,
        &device,
        token,
        reporter,
    )
}

/// `TrainingConfig::validate` also refuses zero, later and with a message about
/// a config the operator never wrote.
fn check_epochs(epochs: u32) -> Result<(), TaskError> {
    if epochs == 0 || epochs > MAX_EPOCHS {
        return Err(invalid_request(
            "训练轮数无效",
            format!("epochs must be between 1 and {MAX_EPOCHS}, got {epochs}"),
        ));
    }
    Ok(())
}

/// The temporal mode pairs each frame with its successor, so one frame yields no
/// samples at all. `DataLoaderConfig::validate` refuses it too, in terms of a
/// sample count rather than of frames.
pub fn check_frame_count(mode: DomainTrainingMode, frame_count: u64) -> Result<(), TaskError> {
    if matches!(mode, DomainTrainingMode::Temporal) && frame_count < 2 {
        return Err(invalid_request(
            "帧数不足，无法做时序训练",
            format!("temporal training needs at least 2 frames, got {frame_count}"),
        ));
    }
    Ok(())
}

/// Everything that goes wrong before the first step is a preparing failure.
fn failed_preparing(error: &TrainingError) -> CommandOutcome {
    CommandOutcome::Failed(training_task_error(error, TaskStage::Preparing))
}
