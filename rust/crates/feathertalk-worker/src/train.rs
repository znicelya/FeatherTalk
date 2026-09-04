//! The training loop and the command that drives it.

use std::path::PathBuf;
use std::time::Instant;

use burn::{module::AutodiffModule, optim::Optimizer};
use feathertalk_domain::{Progress, TaskStage};
use feathertalk_media::CancellationToken;
use feathertalk_models::unet::TrainableTalkingHead;
use feathertalk_training::{
    CheckpointCompatibility, PerceptualFeatureExtractor, TrainingDataset, TrainingError,
    load_training_checkpoint,
};
use feathertalk_training_data::TrainingItem;
use feathertalk_training_run::{StepReport, TrainingRunner, build_preview_artifact};

use crate::{
    CommandOutcome, TRAINING_SEED, TaskReporter, TrainBackend, TrainDevice, TrainSummary,
    TrainingPlan, WORKER_STATE, preview_sample, publish_checkpoint, sample_count, train_to_json,
    training_task_error, write_metrics_unless_present, write_preview_unless_present,
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
