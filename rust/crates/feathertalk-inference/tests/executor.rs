use std::path::PathBuf;

use feathertalk_inference::{InferenceError, OfflineRenderRequest};

fn request(root: &std::path::Path) -> OfflineRenderRequest {
    OfflineRenderRequest::new(
        root.join("frames"),
        root.join("landmarks"),
        root.join("features.f32"),
        root.join("audio.wav"),
        root.join("ffmpeg.exe"),
        root.join("result.mp4"),
        "task-01",
        2,
        None,
    )
    .unwrap()
}

#[test]
fn request_keeps_native_paths_and_limits() {
    let root = tempfile::tempdir().unwrap();
    let request = request(root.path());

    assert_eq!(request.frame_dir(), root.path().join("frames"));
    assert_eq!(request.landmark_dir(), root.path().join("landmarks"));
    assert_eq!(request.feature_path(), root.path().join("features.f32"));
    assert_eq!(request.audio_path(), root.path().join("audio.wav"));
    assert_eq!(request.ffmpeg_path(), root.path().join("ffmpeg.exe"));
    assert_eq!(request.output_path(), root.path().join("result.mp4"));
    assert_eq!(request.task_id(), "task-01");
    assert_eq!(request.source_frame_count(), 2);
    assert_eq!(request.max_output_frames(), None);
}

#[test]
fn request_rejects_relative_paths_and_invalid_counts() {
    let root = tempfile::tempdir().unwrap();

    assert!(matches!(
        OfflineRenderRequest::new(
            PathBuf::from("frames"),
            root.path().join("landmarks"),
            root.path().join("features.f32"),
            root.path().join("audio.wav"),
            root.path().join("ffmpeg.exe"),
            root.path().join("result.mp4"),
            "task-01",
            2,
            None,
        ),
        Err(InferenceError::InvalidField {
            field: "frame_dir",
            ..
        })
    ));
    assert!(matches!(
        OfflineRenderRequest::new(
            root.path().join("frames"),
            root.path().join("landmarks"),
            root.path().join("features.f32"),
            root.path().join("audio.wav"),
            root.path().join("ffmpeg.exe"),
            root.path().join("result.mp4"),
            "task-01",
            1,
            None,
        ),
        Err(InferenceError::FrameCountTooSmall {
            actual: 1,
            minimum: 2
        })
    ));
    assert!(matches!(
        OfflineRenderRequest::new(
            root.path().join("frames"),
            root.path().join("landmarks"),
            root.path().join("features.f32"),
            root.path().join("audio.wav"),
            root.path().join("ffmpeg.exe"),
            root.path().join("result.mp4"),
            "task-01",
            2,
            Some(0),
        ),
        Err(InferenceError::InvalidField {
            field: "max_output_frames",
            ..
        })
    ));
}

#[test]
fn request_reuses_output_and_task_id_validation() {
    let root = tempfile::tempdir().unwrap();
    let output = root.path().join("result.mp4");
    std::fs::write(&output, b"sentinel").unwrap();

    assert!(matches!(
        OfflineRenderRequest::new(
            root.path().join("frames"),
            root.path().join("landmarks"),
            root.path().join("features.f32"),
            root.path().join("audio.wav"),
            root.path().join("ffmpeg.exe"),
            output.clone(),
            "task-01",
            2,
            None,
        ),
        Err(InferenceError::OutputExists { path }) if path == output
    ));
    assert_eq!(std::fs::read(&output).unwrap(), b"sentinel");

    std::fs::remove_file(&output).unwrap();
    assert!(matches!(
        OfflineRenderRequest::new(
            root.path().join("frames"),
            root.path().join("landmarks"),
            root.path().join("features.f32"),
            root.path().join("audio.wav"),
            root.path().join("ffmpeg.exe"),
            output,
            "bad/task",
            2,
            None,
        ),
        Err(InferenceError::InvalidTaskId { .. })
    ));
}
