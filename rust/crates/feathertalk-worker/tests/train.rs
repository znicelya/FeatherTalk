mod support;

use std::fs;
use std::path::Path;

use burn::optim::AdamConfig;
use feathertalk_domain::{ErrorCode, Progress, TaskStage, TrainingMode as DomainTrainingMode};
use feathertalk_media::CancellationToken;
use feathertalk_worker::{
    CommandOutcome, TrainDevice, TrainingPaths, latest_checkpoint, run_training,
};
use serde_json::{Value, json};

use support::{
    IdentityExtractor, PoisonedExtractor, Recorder, StubDataset, micro_plan, model, on_step_stack,
};

fn completed(outcome: CommandOutcome) -> Value {
    match outcome {
        CommandOutcome::Completed(Some(payload)) => payload,
        other => panic!("expected a completed outcome, got {other:?}"),
    }
}

fn failed(outcome: CommandOutcome) -> feathertalk_domain::TaskError {
    match outcome {
        CommandOutcome::Failed(error) => error,
        other => panic!("expected a failed outcome, got {other:?}"),
    }
}

fn names(directory: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(directory)
        .expect("the directory is readable")
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// The step number of the first training event.
fn first_step(events: &[(TaskStage, Option<Progress>)]) -> u64 {
    match events.first().expect("at least one event") {
        (TaskStage::Training { step, .. }, _) => *step,
        (other, _) => panic!("expected a training stage, got {other:?}"),
    }
}

#[test]
fn a_baseline_run_trains_publishes_and_reports() {
    on_step_stack("baseline-run", || {
        let root = tempfile::tempdir().unwrap();
        let project = root.path().join("project");
        let device = TrainDevice::default();
        let plan = micro_plan(&project, DomainTrainingMode::Baseline, 2, 4, None);
        let token = CancellationToken::new();
        let reporter = Recorder::new();

        let outcome = run_training(
            &plan,
            StubDataset::new(4),
            model(&device),
            AdamConfig::new().init(),
            &IdentityExtractor,
            &device,
            &token,
            &reporter,
        );

        let payload = completed(outcome);
        assert_eq!(payload["mode"], json!("baseline"));
        assert_eq!(payload["frame_count"], json!(4));
        assert_eq!(payload["epochs_requested"], json!(2));
        assert_eq!(payload["epochs_completed"], json!(2));
        assert_eq!(payload["global_step"], json!(8));
        assert_eq!(payload["samples_seen"], json!(8));
        assert_eq!(payload["resumed_from"], json!(null));
        assert_eq!(payload["checkpoints_written"], json!(2));
        assert_eq!(payload["metrics_written"], json!(2));
        assert_eq!(payload["previews_written"], json!(2));
        let loss = payload["total_loss"].as_f64().expect("a loss was observed");
        assert!(loss.is_finite(), "{loss}");

        // One checkpoint per epoch boundary, named by step, with nothing staged
        // or retired left behind.
        let paths = TrainingPaths::new(&project);
        assert_eq!(
            names(paths.checkpoints()),
            vec!["checkpoint-00000004", "checkpoint-00000008"]
        );
        let latest = paths.checkpoint(8).display().to_string();
        assert_eq!(payload["checkpoint_dir"], json!(latest));
        assert!(paths.metrics(4).is_file());
        assert!(paths.metrics(8).is_file());
        assert!(paths.preview(4).join("manifest.json").is_file());
        assert!(paths.preview(8).join("manifest.json").is_file());

        // One event per step, every one of them a training stage, the last one
        // complete. Epochs are zero-based inside the loader, so the final step
        // belongs to epoch 1 while `epochs_completed` counts 2.
        let events = reporter.events();
        assert_eq!(events.len(), 8);
        assert_eq!(first_step(&events), 1);
        let (stage, progress) = events.last().expect("eight events").clone();
        let expected = Progress {
            completed: 8,
            total: Some(8),
        };
        assert_eq!(progress, Some(expected));
        match stage {
            TaskStage::Training { epoch, step, loss } => {
                assert_eq!((epoch, step), (1, 8));
                assert!(loss.is_finite(), "{loss}");
            }
            other => panic!("expected a training stage, got {other:?}"),
        }
    });
}

#[test]
fn a_cancelled_run_leaves_a_checkpoint_the_resume_continues_from() {
    on_step_stack("cancel-resume", || {
        let root = tempfile::tempdir().unwrap();
        let project = root.path().join("project");
        let device = TrainDevice::default();
        let paths = TrainingPaths::new(&project);
        let plan = micro_plan(&project, DomainTrainingMode::Baseline, 2, 4, None);
        let token = CancellationToken::new();
        // The fourth step ends the first epoch, so the cancellation lands right
        // after a checkpoint was published for that step.
        let reporter = Recorder::cancelling_after(4, token.clone());

        let outcome = run_training(
            &plan,
            StubDataset::new(4),
            model(&device),
            AdamConfig::new().init(),
            &IdentityExtractor,
            &device,
            &token,
            &reporter,
        );

        assert!(matches!(outcome, CommandOutcome::Cancelled), "{outcome:?}");
        // The cancellation republished the same step, so the retire path ran and
        // still left exactly one checkpoint and no staging directories.
        assert_eq!(names(paths.checkpoints()), vec!["checkpoint-00000004"]);

        let resumed_from = latest_checkpoint(&paths)
            .expect("the directory is readable")
            .expect("the cancelled run saved one");
        let plan = micro_plan(
            &project,
            DomainTrainingMode::Baseline,
            2,
            4,
            Some(resumed_from.clone()),
        );
        let token = CancellationToken::new();
        let reporter = Recorder::new();

        let outcome = run_training(
            &plan,
            StubDataset::new(4),
            model(&device),
            AdamConfig::new().init(),
            &IdentityExtractor,
            &device,
            &token,
            &reporter,
        );

        let payload = completed(outcome);
        assert_eq!(payload["global_step"], json!(8));
        assert_eq!(payload["epochs_completed"], json!(2));
        // `restore` zeroes the sample counter, so the payload reports what this
        // run saw rather than the lineage total.
        assert_eq!(payload["samples_seen"], json!(4));
        let resumed = resumed_from.display().to_string();
        assert_eq!(payload["resumed_from"], json!(resumed));
        assert_eq!(payload["checkpoints_written"], json!(1));
        // Only the second epoch ran, and it started at step five.
        let events = reporter.events();
        assert_eq!(events.len(), 4);
        assert_eq!(first_step(&events), 5);
    });
}

#[test]
fn a_checkpoint_from_another_configuration_is_refused() {
    on_step_stack("descriptor-mismatch", || {
        let root = tempfile::tempdir().unwrap();
        let project = root.path().join("project");
        let device = TrainDevice::default();
        let paths = TrainingPaths::new(&project);
        let token = CancellationToken::new();
        let reporter = Recorder::new();
        let plan = micro_plan(&project, DomainTrainingMode::Baseline, 1, 2, None);

        let outcome = run_training(
            &plan,
            StubDataset::new(2),
            model(&device),
            AdamConfig::new().init(),
            &IdentityExtractor,
            &device,
            &token,
            &reporter,
        );
        completed(outcome);

        let checkpoint = latest_checkpoint(&paths)
            .expect("the directory is readable")
            .expect("the first run saved one");
        // Everything else about the plan is identical -- the epoch count too,
        // because `training_config` is compared field by field and a different
        // one would fail the same way for a different reason.
        let mut plan = micro_plan(
            &project,
            DomainTrainingMode::Baseline,
            1,
            2,
            Some(checkpoint),
        );
        plan.descriptor.model_config_sha256 = "b".repeat(64);

        let outcome = run_training(
            &plan,
            StubDataset::new(2),
            model(&device),
            AdamConfig::new().init(),
            &IdentityExtractor,
            &device,
            &token,
            &reporter,
        );

        let error = failed(outcome);
        assert_eq!(error.code, ErrorCode::ModelIncompatible);
        // The run never started, so the stage is still the preparing one.
        assert_eq!(error.stage, TaskStage::Preparing);
    });
}

#[test]
fn telemetry_that_is_already_there_is_skipped_and_counted() {
    on_step_stack("telemetry-skip", || {
        let root = tempfile::tempdir().unwrap();
        let project = root.path().join("project");
        let device = TrainDevice::default();
        let paths = TrainingPaths::new(&project);
        // What a previous run of the same lineage leaves behind.
        fs::create_dir_all(paths.preview(4)).unwrap();
        fs::create_dir_all(paths.metrics(4).parent().unwrap()).unwrap();
        fs::write(paths.metrics(4), "not even json").unwrap();
        let plan = micro_plan(&project, DomainTrainingMode::Baseline, 1, 4, None);
        let token = CancellationToken::new();
        let reporter = Recorder::new();

        let outcome = run_training(
            &plan,
            StubDataset::new(4),
            model(&device),
            AdamConfig::new().init(),
            &IdentityExtractor,
            &device,
            &token,
            &reporter,
        );

        let payload = completed(outcome);
        // The weights are the product, so the checkpoint still lands; the two
        // diagnostics are skipped and the payload says so.
        assert_eq!(payload["checkpoints_written"], json!(1));
        assert_eq!(payload["metrics_written"], json!(0));
        assert_eq!(payload["previews_written"], json!(0));
        assert!(paths.checkpoint(4).is_dir());
        // Neither leftover was touched.
        assert_eq!(
            fs::read_to_string(paths.metrics(4)).unwrap(),
            "not even json"
        );
        assert!(names(&paths.preview(4)).is_empty());
    });
}

#[test]
fn a_step_that_fails_reports_the_step_it_reached() {
    on_step_stack("failing-step", || {
        let root = tempfile::tempdir().unwrap();
        let project = root.path().join("project");
        let device = TrainDevice::default();
        let paths = TrainingPaths::new(&project);
        let plan = micro_plan(&project, DomainTrainingMode::Baseline, 1, 4, None);
        let token = CancellationToken::new();
        let reporter = Recorder::new();
        // A baseline step calls the extractor twice, so the first step goes
        // through and the second one produces a non-finite loss.
        let extractor = PoisonedExtractor::after(2);

        let outcome = run_training(
            &plan,
            StubDataset::new(4),
            model(&device),
            AdamConfig::new().init(),
            &extractor,
            &device,
            &token,
            &reporter,
        );

        let error = failed(outcome);
        assert_eq!(error.code, ErrorCode::MediaInvalid);
        // The whole point of threading the stage through: a run that dies at
        // step two says step two, not "preparing".
        match error.stage {
            TaskStage::Training { epoch, step, .. } => assert_eq!((epoch, step), (0, 1)),
            other => panic!("expected the last training stage, got {other:?}"),
        }
        // The failed run reached no epoch boundary, so it published nothing.
        assert!(!paths.checkpoints().exists());
        assert_eq!(reporter.events().len(), 1);
    });
}
