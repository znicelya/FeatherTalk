use std::{collections::VecDeque, path::PathBuf, sync::Mutex, time::Duration};

use feathertalk_domain::{
    ErrorCode, ProbeMediaParams, ProjectDirParams, Request, TrainParams, TrainingMode, UnetVariant,
};
use feathertalk_media::{
    CancellationToken, CommandSpec, MediaError, MediaToolchain, ProcessOutput, ProcessRunner,
};
use feathertalk_project::{
    AssetManifest, AssetPackageState, FeatureType, ModelSelection, ProjectManifest,
    TaskHistoryEntry, TaskHistoryStatus, lock_asset_package, write_project_manifest_atomic,
};
use feathertalk_worker::{CommandOutcome, execute_with_runner};

struct FakeRunner {
    outputs: Mutex<VecDeque<Result<ProcessOutput, MediaError>>>,
    commands: Mutex<Vec<CommandSpec>>,
}

impl FakeRunner {
    fn new(outputs: Vec<Result<ProcessOutput, MediaError>>) -> Self {
        Self {
            outputs: Mutex::new(outputs.into_iter().collect()),
            commands: Mutex::new(Vec::new()),
        }
    }

    fn call_count(&self) -> usize {
        self.commands.lock().unwrap().len()
    }
}

impl ProcessRunner for FakeRunner {
    fn run(&self, command: &CommandSpec, _timeout: Duration) -> Result<ProcessOutput, MediaError> {
        self.commands.lock().unwrap().push(command.clone());
        self.outputs.lock().unwrap().pop_front().unwrap()
    }
}

fn toolchain() -> MediaToolchain {
    let root = std::env::current_dir().unwrap();
    MediaToolchain::new(
        root.join("ffmpeg-test"),
        root.join("ffprobe-test"),
        Duration::from_secs(10),
    )
    .unwrap()
}

fn valid_probe() -> Vec<u8> {
    br#"{
      "format":{"format_name":"mov,mp4","duration":"2.0"},
      "streams":[
        {"codec_type":"video","codec_name":"h264","pix_fmt":"yuv420p","width":640,"height":480,"avg_frame_rate":"25/1","nb_read_frames":"50","duration":"2.0"},
        {"codec_type":"audio","codec_name":"aac","sample_fmt":"fltp","sample_rate":"48000","channels":2,"duration":"2.0"}
      ]
    }"#
    .to_vec()
}

fn probe_request(input: PathBuf) -> Request {
    Request::ProbeMedia(ProbeMediaParams { input })
}

fn media_file() -> (tempfile::TempDir, PathBuf) {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("input.mov");
    std::fs::write(&source, b"media").unwrap();
    (temp, source)
}

fn valid_project() -> ProjectManifest {
    ProjectManifest {
        schema_version: 1,
        project_id: "demo".to_owned(),
        display_name: "Demo".to_owned(),
        asset_package: "assets/assets.json".to_owned(),
        default_model: ModelSelection::OriginalUnet,
        task_history: vec![TaskHistoryEntry {
            task_id: "task-1".to_owned(),
            kind: "preprocess".to_owned(),
            status: TaskHistoryStatus::Completed,
            updated_at: "2026-08-20T10:00:00Z".to_owned(),
        }],
    }
}

fn locked_manifest() -> AssetManifest {
    AssetManifest {
        schema_version: 1,
        state: AssetPackageState::Locked,
        video_fps: 25,
        audio_sample_rate: 16_000,
        audio_channels: 1,
        frame_count: 12,
        frame_width: 160,
        frame_height: 160,
        feature_type: FeatureType::FeatherHubert,
        feature_shape: [12, 2, 1024],
        landmark_model_sha256: "a".repeat(64),
        feature_model_sha256: "b".repeat(64),
    }
}

fn complete_project() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("assets/frames")).unwrap();
    std::fs::create_dir_all(dir.path().join("assets/landmarks")).unwrap();
    std::fs::create_dir_all(dir.path().join("assets/features")).unwrap();
    for file in [
        "assets/video_25fps.mp4",
        "assets/audio_16k_mono.wav",
        "assets/features/feather_hubert.f32",
    ] {
        std::fs::write(dir.path().join(file), b"x").unwrap();
    }
    write_project_manifest_atomic(&dir.path().join("project.json"), &valid_project()).unwrap();
    lock_asset_package(dir.path(), locked_manifest()).unwrap();
    dir
}

#[test]
fn validating_a_complete_project_completes_without_a_result() {
    let dir = complete_project();
    let request = Request::ValidateProject(ProjectDirParams {
        project_dir: dir.path().to_path_buf(),
    });
    let runner = FakeRunner::new(vec![]);
    let outcome = execute_with_runner(&request, None, &CancellationToken::new(), &runner);
    assert!(
        matches!(outcome, CommandOutcome::Completed(None)),
        "{outcome:?}"
    );
    assert_eq!(runner.call_count(), 0);
}

#[test]
fn validating_a_missing_project_fails_with_a_wire_error() {
    let dir = tempfile::tempdir().unwrap();
    let request = Request::ValidateProject(ProjectDirParams {
        project_dir: dir.path().join("nope"),
    });
    let runner = FakeRunner::new(vec![]);
    let CommandOutcome::Failed(error) =
        execute_with_runner(&request, None, &CancellationToken::new(), &runner)
    else {
        panic!("a missing project must fail");
    };
    error.validate().unwrap();
    assert!(!error.summary.is_empty());
}

#[test]
fn probing_media_completes_with_the_probe_result() {
    let (_temp, source) = media_file();
    let runner = FakeRunner::new(vec![Ok(ProcessOutput::new(
        Some(0),
        valid_probe(),
        Vec::new(),
    ))]);
    let toolchain = toolchain();
    let CommandOutcome::Completed(Some(result)) = execute_with_runner(
        &probe_request(source),
        Some(&toolchain),
        &CancellationToken::new(),
        &runner,
    ) else {
        panic!("a successful probe must carry a result");
    };
    assert!(result.is_object());
    assert_eq!(result["format"]["format_name"], "mov,mp4");
    assert_eq!(result["format"]["duration_seconds"], 2.0);
    assert_eq!(result["video"]["codec_name"], "h264");
    assert_eq!(result["video"]["pixel_format"], "yuv420p");
    assert_eq!(result["video"]["width"], 640);
    assert_eq!(result["video"]["height"], 480);
    assert_eq!(result["video"]["frame_rate"]["numerator"], 25);
    assert_eq!(result["video"]["frame_rate"]["denominator"], 1);
    assert_eq!(result["video"]["frame_count"], 50);
    assert_eq!(result["audio"]["codec_name"], "aac");
    assert_eq!(result["audio"]["sample_format"], "fltp");
    assert_eq!(result["audio"]["sample_rate"], 48_000);
    assert_eq!(result["audio"]["channels"], 2);
    assert_eq!(runner.call_count(), 1);
    assert_eq!(result.get("input"), None, "the result must not leak paths");
}

#[test]
fn probing_a_missing_file_fails_before_the_tool_runs() {
    let temp = tempfile::tempdir().unwrap();
    let runner = FakeRunner::new(vec![]);
    let toolchain = toolchain();
    let CommandOutcome::Failed(error) = execute_with_runner(
        &probe_request(temp.path().join("absent.mov")),
        Some(&toolchain),
        &CancellationToken::new(),
        &runner,
    ) else {
        panic!("a missing input must fail");
    };
    assert_eq!(error.code, ErrorCode::MediaInvalid);
    error.validate().unwrap();
    assert_eq!(runner.call_count(), 0);
}

#[test]
fn a_cancelled_tool_reports_cancellation_not_failure() {
    let (_temp, source) = media_file();
    let runner = FakeRunner::new(vec![Err(MediaError::ToolCancelled { operation: "probe" })]);
    let toolchain = toolchain();
    let outcome = execute_with_runner(
        &probe_request(source),
        Some(&toolchain),
        &CancellationToken::new(),
        &runner,
    );
    assert!(matches!(outcome, CommandOutcome::Cancelled), "{outcome:?}");
}

#[test]
fn an_already_cancelled_token_runs_nothing() {
    let (_temp, source) = media_file();
    let runner = FakeRunner::new(vec![]);
    let toolchain = toolchain();
    let token = CancellationToken::new();
    token.cancel();
    let outcome = execute_with_runner(&probe_request(source), Some(&toolchain), &token, &runner);
    assert!(matches!(outcome, CommandOutcome::Cancelled), "{outcome:?}");
    assert_eq!(runner.call_count(), 0);
}

#[test]
fn probing_without_a_toolchain_is_refused() {
    let (_temp, source) = media_file();
    let runner = FakeRunner::new(vec![]);
    let CommandOutcome::Failed(error) = execute_with_runner(
        &probe_request(source),
        None,
        &CancellationToken::new(),
        &runner,
    ) else {
        panic!("probing without a toolchain must fail");
    };
    assert_eq!(error.code, ErrorCode::WorkerCrashed);
    error.validate().unwrap();
}

#[test]
fn an_unsupported_command_is_refused_with_its_slug() {
    let request = Request::Train(TrainParams {
        project_dir: PathBuf::from("C:/tmp/project"),
        mode: TrainingMode::Baseline,
        variant: UnetVariant::OriginalUnet,
        epochs: 1,
        resume: false,
    });
    let runner = FakeRunner::new(vec![]);
    let CommandOutcome::Failed(error) =
        execute_with_runner(&request, None, &CancellationToken::new(), &runner)
    else {
        panic!("an unsupported command must fail");
    };
    assert_eq!(error.code, ErrorCode::WorkerCrashed);
    assert!(error.detail.contains("train"), "{}", error.detail);
    error.validate().unwrap();
}
