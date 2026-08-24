use std::{
    collections::VecDeque,
    fs,
    path::{Path, PathBuf},
    sync::Mutex,
    time::Duration,
};

use feathertalk_frame_pipeline::{
    CommandSpec, FrameExtractor, FramePipelineSpec, PipelineError, ProcessOutput, ProcessRunner,
    extract_frames_with_runner,
};

struct FakeRunner {
    outputs: Mutex<VecDeque<Result<ProcessOutput, PipelineError>>>,
    commands: Mutex<Vec<CommandSpec>>,
    writes: WriteMode,
}

#[derive(Clone, Copy)]
enum WriteMode {
    Bytes,
    Missing,
    Empty,
    Oversized,
}

impl FakeRunner {
    fn new(outputs: Vec<Result<ProcessOutput, PipelineError>>, writes: WriteMode) -> Self {
        Self {
            outputs: Mutex::new(outputs.into()),
            commands: Mutex::new(Vec::new()),
            writes,
        }
    }
}

impl ProcessRunner for FakeRunner {
    fn run(
        &self,
        command: &CommandSpec,
        _timeout: Duration,
    ) -> Result<ProcessOutput, PipelineError> {
        let command_number = {
            let mut commands = self.commands.lock().unwrap();
            commands.push(command.clone());
            commands.len()
        };
        let output = self.outputs.lock().unwrap().pop_front().unwrap()?;
        let path = Path::new(command.arguments().last().unwrap());
        match self.writes {
            WriteMode::Bytes => fs::write(path, format!("frame:{command_number}")).unwrap(),
            WriteMode::Missing => {}
            WriteMode::Empty => {
                fs::write(path, []).unwrap();
            }
            WriteMode::Oversized => {
                fs::write(path, b"x").unwrap();
                let file = fs::OpenOptions::new().write(true).open(path).unwrap();
                file.set_len(16 * 1024 * 1024 + 1).unwrap();
            }
        }
        Ok(output)
    }
}

fn setup(frame_count: u64) -> (tempfile::TempDir, FramePipelineSpec, FrameExtractor) {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("video_25fps.mp4");
    fs::write(&source, b"video").unwrap();
    let output = root.path().join("assets");
    let spec = FramePipelineSpec::new(source, output, frame_count, 640, 480).unwrap();
    let extractor =
        FrameExtractor::new(root.path().join("ffmpeg"), Duration::from_secs(10)).unwrap();
    (root, spec, extractor)
}

fn ok_outputs(count: usize) -> Vec<Result<ProcessOutput, PipelineError>> {
    (0..count)
        .map(|_| Ok(ProcessOutput::new(Some(0), vec![], vec![])))
        .collect()
}

fn staging_dirs(root: &Path) -> Vec<PathBuf> {
    fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(".feathertalk-frame-build-"))
        })
        .collect()
}

#[test]
fn extracts_exact_frame_count_and_records_hashes() {
    let (root, spec, extractor) = setup(3);
    let runner = FakeRunner::new(ok_outputs(3), WriteMode::Bytes);
    let batch = extract_frames_with_runner(&spec, &extractor, &runner).unwrap();
    assert_eq!(batch.frames().len(), 3);
    assert_eq!(batch.frames()[0].index(), 0);
    assert_eq!(batch.frames()[2].path().file_name().unwrap(), "000002.jpg");
    assert!(
        batch
            .frames()
            .iter()
            .all(|frame| frame.bytes() > 0 && frame.sha256().len() == 64)
    );
    assert_eq!(runner.commands.lock().unwrap().len(), 3);
    assert!(batch.staging_dir().starts_with(root.path()));
}

#[test]
fn nonzero_tool_exit_cleans_staging_and_preserves_old_outputs() {
    let (root, spec, extractor) = setup(2);
    fs::create_dir_all(spec.output_root().join("frames")).unwrap();
    fs::write(spec.frame_path(0), b"old").unwrap();
    let runner = FakeRunner::new(
        vec![
            Ok(ProcessOutput::new(Some(0), vec![], vec![])),
            Ok(ProcessOutput::new(Some(2), vec![], b"bad input".to_vec())),
        ],
        WriteMode::Bytes,
    );
    assert!(matches!(
        extract_frames_with_runner(&spec, &extractor, &runner),
        Err(PipelineError::OutputDestinationExists { .. })
    ));
    assert_eq!(fs::read(spec.frame_path(0)).unwrap(), b"old");
    assert!(staging_dirs(root.path()).is_empty());
}

#[test]
fn missing_output_is_rejected_and_staging_is_removed() {
    let (root, spec, extractor) = setup(1);
    let runner = FakeRunner::new(ok_outputs(1), WriteMode::Missing);
    assert!(matches!(
        extract_frames_with_runner(&spec, &extractor, &runner),
        Err(PipelineError::FrameMissing { .. })
    ));
    assert!(staging_dirs(root.path()).is_empty());
}

#[test]
fn empty_and_oversized_outputs_are_rejected() {
    for (mode, expected) in [(WriteMode::Empty, "empty"), (WriteMode::Oversized, "large")] {
        let (root, spec, extractor) = setup(1);
        let runner = FakeRunner::new(ok_outputs(1), mode);
        let result = extract_frames_with_runner(&spec, &extractor, &runner);
        match (expected, result) {
            ("empty", Err(PipelineError::FrameEmpty { .. })) => {}
            ("large", Err(PipelineError::FrameTooLarge { .. })) => {}
            other => panic!("unexpected result: {other:?}"),
        }
        assert!(staging_dirs(root.path()).is_empty());
    }
}

#[test]
fn injected_timeout_is_preserved() {
    let (_root, spec, extractor) = setup(1);
    let runner = FakeRunner::new(
        vec![Err(PipelineError::ToolTimedOut {
            operation: "extract_frame",
            timeout_ms: 10,
        })],
        WriteMode::Missing,
    );
    assert!(matches!(
        extract_frames_with_runner(&spec, &extractor, &runner),
        Err(PipelineError::ToolTimedOut {
            operation: "extract_frame",
            ..
        })
    ));
}

#[cfg(windows)]
#[test]
fn symlink_output_is_rejected() {
    let (root, spec, extractor) = setup(1);
    let target = root.path().join("target.jpg");
    fs::write(&target, b"target").unwrap();
    let runner = SymlinkRunner { target };
    assert!(matches!(
        extract_frames_with_runner(&spec, &extractor, &runner),
        Err(PipelineError::FrameNotRegular { .. })
    ));
}

#[cfg(windows)]
struct SymlinkRunner {
    target: PathBuf,
}

#[cfg(windows)]
impl ProcessRunner for SymlinkRunner {
    fn run(
        &self,
        command: &CommandSpec,
        _timeout: Duration,
    ) -> Result<ProcessOutput, PipelineError> {
        std::os::windows::fs::symlink_file(&self.target, command.arguments().last().unwrap())
            .unwrap();
        Ok(ProcessOutput::new(Some(0), vec![], vec![]))
    }
}
