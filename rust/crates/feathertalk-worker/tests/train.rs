mod support;

use std::fs;
use std::path::{Path, PathBuf};

use burn::optim::AdamConfig;
use feathertalk_domain::{
    ErrorCode, Progress, TaskStage, TrainParams, TrainingMode as DomainTrainingMode, UnetVariant,
};
use feathertalk_media::CancellationToken;
use feathertalk_worker::{
    CommandOutcome, MAX_EPOCHS, TrainDevice, TrainingPaths, WorkerConfig, check_frame_count,
    execute_train, latest_checkpoint, run_training,
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

/// A config whose training toolchain points at an empty directory. Every
/// admission test below fails long before the weights would be read.
fn training_config(vgg19_dir: &Path) -> WorkerConfig {
    WorkerConfig::from_values_with_training(
        None,
        None,
        None,
        None,
        None,
        None,
        Some(vgg19_dir.display().to_string()),
    )
}

fn train_params(project_dir: &Path, mode: DomainTrainingMode, epochs: u32) -> TrainParams {
    TrainParams {
        project_dir: project_dir.to_path_buf(),
        mode,
        variant: UnetVariant::OriginalUnet,
        epochs,
        resume: false,
    }
}

/// A directory that gets past `check_project_dir`: absolute, a real directory,
/// with a regular `project.json` inside. Nothing in it is locked.
fn project_shell(root: &Path) -> PathBuf {
    let project = root.join("project");
    fs::create_dir_all(&project).unwrap();
    fs::write(project.join("project.json"), "{}").unwrap();
    project
}

#[test]
fn a_relative_project_directory_is_refused() {
    let root = tempfile::tempdir().unwrap();
    let config = training_config(root.path());
    let training = config.training().expect("the directory exists");
    let params = train_params(Path::new("project"), DomainTrainingMode::Baseline, 1);

    let outcome = execute_train(
        &params,
        &CancellationToken::new(),
        &Recorder::new(),
        training,
    );

    let error = failed(outcome);
    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.summary, "工程目录必须是绝对路径");
}

#[test]
fn the_epoch_count_has_to_be_in_range() {
    let root = tempfile::tempdir().unwrap();
    let project = project_shell(root.path());
    let config = training_config(root.path());
    let training = config.training().expect("the directory exists");

    for epochs in [0, MAX_EPOCHS + 1] {
        let params = train_params(&project, DomainTrainingMode::Baseline, epochs);
        let outcome = execute_train(
            &params,
            &CancellationToken::new(),
            &Recorder::new(),
            training,
        );
        let error = failed(outcome);
        assert_eq!(error.summary, "训练轮数无效", "{epochs}");
        assert_eq!(error.code, ErrorCode::MediaInvalid, "{epochs}");
    }
}

#[test]
fn a_project_without_a_locked_package_is_refused_by_the_dataset() {
    let root = tempfile::tempdir().unwrap();
    let project = project_shell(root.path());
    let config = training_config(root.path());
    let training = config.training().expect("the directory exists");
    let reporter = Recorder::new();
    let params = train_params(&project, DomainTrainingMode::Baseline, 1);

    let outcome = execute_train(&params, &CancellationToken::new(), &reporter, training);

    // `ProjectTrainingDataset::open` is the single place that enforces "extract,
    // extract features, then lock"; the worker does not re-check it.
    let error = failed(outcome);
    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.stage, TaskStage::Preparing);
    // The stage went out before the expensive part of admission started.
    assert_eq!(reporter.events().len(), 1);
    assert_eq!(reporter.events()[0].0, TaskStage::Preparing);
}

#[test]
fn the_temporal_mode_needs_two_frames() {
    // Checked through its own function: reaching it through `execute_train`
    // needs a locked one-frame project, which would mean copying
    // feathertalk-training-run's fixture into this crate for one assertion.
    assert!(check_frame_count(DomainTrainingMode::Baseline, 1).is_ok());
    assert!(check_frame_count(DomainTrainingMode::Temporal, 2).is_ok());

    let error = check_frame_count(DomainTrainingMode::Temporal, 1)
        .expect_err("one frame yields no temporal pair");
    assert_eq!(error.summary, "帧数不足，无法做时序训练");
    assert_eq!(error.code, ErrorCode::MediaInvalid);
}
