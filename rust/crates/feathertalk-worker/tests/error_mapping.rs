use std::{io, path::PathBuf};

use feathertalk_domain::{ErrorCode, MAX_DETAIL_CHARS, TaskStage};
use feathertalk_frame_pipeline::{AnomalyCode, FrameAnomaly, PipelineError, RecoveryAction};
use feathertalk_media::MediaError;
use feathertalk_project::ProjectError;
use feathertalk_worker::{
    is_media_cancellation, is_pipeline_cancellation, media_task_error, pipeline_task_error,
    project_task_error, quality_task_error,
};

fn io_error(kind: io::ErrorKind) -> io::Error {
    io::Error::new(kind, "synthetic")
}

fn json_error() -> serde_json::Error {
    serde_json::from_str::<serde_json::Value>("{").unwrap_err()
}

fn path() -> PathBuf {
    PathBuf::from("C:/tmp/x")
}

#[test]
fn every_project_error_maps_to_a_code_and_a_valid_payload() {
    let cases = vec![
        (
            ProjectError::Io {
                operation: "read",
                path: path(),
                source: io_error(io::ErrorKind::PermissionDenied),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            ProjectError::Io {
                operation: "write",
                path: path(),
                source: io_error(io::ErrorKind::StorageFull),
            },
            ErrorCode::DiskSpaceLow,
        ),
        (
            ProjectError::Io {
                operation: "write",
                path: path(),
                source: io_error(io::ErrorKind::QuotaExceeded),
            },
            ErrorCode::DiskSpaceLow,
        ),
        (
            ProjectError::ManifestTooLarge {
                path: path(),
                limit: 1024,
            },
            ErrorCode::MediaInvalid,
        ),
        (
            ProjectError::InvalidUtf8 { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            ProjectError::InvalidJson {
                path: path(),
                source: json_error(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            ProjectError::UnsupportedSchemaVersion {
                path: path(),
                version: 9,
            },
            ErrorCode::MediaInvalid,
        ),
        (
            ProjectError::InvalidField {
                field: "project_id".to_owned(),
                message: "empty".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            ProjectError::UnsafeRelativePath {
                path: "../x".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            ProjectError::Symlink { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            ProjectError::InvalidFilesystemEntry { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            ProjectError::EmptyArtifact { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            ProjectError::LockedAssetMutation { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            ProjectError::AtomicReplacementUnsupported { path: path() },
            ErrorCode::WorkerCrashed,
        ),
    ];

    for (error, expected) in cases {
        let mapped = project_task_error(&error);
        assert_eq!(mapped.code, expected, "{error:?}");
        assert_eq!(mapped.stage, TaskStage::Preparing, "{error:?}");
        assert_eq!(mapped.recovery, expected.default_recovery(), "{error:?}");
        assert!(!mapped.summary.trim().is_empty(), "{error:?}");
        assert!(!mapped.detail.is_empty(), "{error:?}");
        mapped.validate().unwrap();
    }
}

#[test]
fn every_media_error_maps_to_a_code_and_a_valid_payload() {
    let cases = vec![
        (
            MediaError::Io {
                operation: "read",
                path: path(),
                source: io_error(io::ErrorKind::NotFound),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::Io {
                operation: "write",
                path: path(),
                source: io_error(io::ErrorKind::StorageFull),
            },
            ErrorCode::DiskSpaceLow,
        ),
        (
            MediaError::InputMissing { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            MediaError::InputNotRegularFile { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            MediaError::SymlinkNotAllowed { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            MediaError::OutputDirectoryInvalid { path: path() },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::OutputInsideInput {
                input: path(),
                output: path(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::OutputConflictsWithInput { path: path() },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::OutputDestinationInvalid { path: path() },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::UnsupportedTarget {
                field: "fps",
                expected: "25",
                actual: "30".to_owned(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::InvalidToolchain {
                field: "ffprobe",
                message: "relative".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            MediaError::ProbeTooLarge {
                limit: 16,
                actual: 32,
            },
            ErrorCode::MediaInvalid,
        ),
        (
            MediaError::ProbeJson {
                message: "bad".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            MediaError::ProbeContract {
                field: "width".to_owned(),
                message: "missing".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            MediaError::MissingStream { stream: "video" },
            ErrorCode::MediaInvalid,
        ),
        (
            MediaError::DuplicateStream { stream: "audio" },
            ErrorCode::MediaInvalid,
        ),
        (
            MediaError::ToolFailed {
                operation: "probe",
                exit_code: Some(1),
                stderr: "boom".to_owned(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::ToolTimedOut {
                operation: "probe",
                timeout_ms: 10,
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::ToolOutputTooLarge {
                operation: "probe",
                stream: "stdout",
                limit: 16,
                actual: 32,
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::ToolSpawn {
                operation: "probe",
                message: "not found".to_owned(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::ToolCancelled { operation: "probe" },
            ErrorCode::TaskCancelled,
        ),
        (
            MediaError::NormalizationVerificationFailed {
                field: "fps",
                expected: "25".to_owned(),
                actual: "30".to_owned(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::OutputCommitFailed {
                operation: "commit",
                message: "busy".to_owned(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::OutputRollbackFailed {
                operation: "rollback",
                primary: "a".to_owned(),
                rollback: "b".to_owned(),
            },
            ErrorCode::WorkerCrashed,
        ),
    ];

    for (error, expected) in cases {
        let mapped = media_task_error(&error);
        assert_eq!(mapped.code, expected, "{error:?}");
        assert_eq!(mapped.stage, TaskStage::Preparing, "{error:?}");
        assert_eq!(mapped.recovery, expected.default_recovery(), "{error:?}");
        assert!(!mapped.summary.trim().is_empty(), "{error:?}");
        mapped.validate().unwrap();
    }
}

#[test]
fn an_oversized_detail_is_clamped_to_the_wire_limit() {
    let mapped = media_task_error(&MediaError::ProbeJson {
        message: "x".repeat(MAX_DETAIL_CHARS * 2),
    });
    assert_eq!(mapped.detail.chars().count(), MAX_DETAIL_CHARS);
    mapped.validate().unwrap();
}

#[test]
fn only_tool_cancelled_counts_as_cancellation() {
    assert!(is_media_cancellation(&MediaError::ToolCancelled {
        operation: "probe"
    }));
    assert!(!is_media_cancellation(&MediaError::ToolTimedOut {
        operation: "probe",
        timeout_ms: 10
    }));
    assert!(!is_media_cancellation(&MediaError::InputMissing {
        path: path()
    }));
}

fn anomaly(index: u64, code: AnomalyCode) -> FrameAnomaly {
    FrameAnomaly::new(index, code, "摘要", "detail", RecoveryAction::ExcludeFrame).unwrap()
}

#[test]
fn every_pipeline_error_maps_to_a_code_and_a_valid_payload() {
    let cases = vec![
        (
            PipelineError::InvalidField {
                field: "frame_count",
                message: "must be greater than zero".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            PipelineError::OutputDestinationExists { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            PipelineError::FrameMissing { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            PipelineError::Io {
                operation: "create_dir",
                path: path(),
                source: io_error(io::ErrorKind::StorageFull),
            },
            ErrorCode::DiskSpaceLow,
        ),
        (
            PipelineError::Io {
                operation: "create_dir",
                path: path(),
                source: io_error(io::ErrorKind::PermissionDenied),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            PipelineError::Adapter {
                component: "scrfd",
                message: "device lost".to_owned(),
            },
            ErrorCode::ModelIncompatible,
        ),
        (
            PipelineError::Cancelled {
                operation: "extract_frames",
            },
            ErrorCode::TaskCancelled,
        ),
        (
            PipelineError::ToolTimedOut {
                operation: "extract_frames",
                timeout_ms: 300_000,
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            PipelineError::QualityRejected { count: 4 },
            ErrorCode::WorkerCrashed,
        ),
    ];

    for (error, expected) in cases {
        let mapped = pipeline_task_error(&error);
        assert_eq!(mapped.code, expected, "{error:?}");
        assert_eq!(mapped.stage, TaskStage::Preparing, "{error:?}");
        assert_eq!(mapped.recovery, expected.default_recovery(), "{error:?}");
        assert!(!mapped.summary.trim().is_empty(), "{error:?}");
        assert!(!mapped.detail.is_empty(), "{error:?}");
        mapped.validate().unwrap();
    }
}

#[test]
fn only_cancellation_is_reported_as_cancellation() {
    assert!(is_pipeline_cancellation(&PipelineError::Cancelled {
        operation: "evaluate_frames"
    }));
    assert!(!is_pipeline_cancellation(&PipelineError::QualityRejected {
        count: 1
    }));
}

#[test]
fn a_rejected_quality_report_reports_the_first_anomaly() {
    let anomalies = vec![
        anomaly(7, AnomalyCode::LandmarkInvalid),
        anomaly(8, AnomalyCode::FaceNotFound),
        anomaly(9, AnomalyCode::BlurredFrame),
        anomaly(10, AnomalyCode::ModelFailed),
    ];

    let mapped = quality_task_error(&anomalies);

    assert_eq!(mapped.code, ErrorCode::LandmarkInvalid);
    assert_eq!(mapped.stage, TaskStage::Preparing);
    assert!(
        mapped.detail.contains("4 frame(s) rejected"),
        "{}",
        mapped.detail
    );
    // Only the first three are named, so a run with thousands of bad frames
    // still produces a readable detail.
    assert!(mapped.detail.contains("frame 7"), "{}", mapped.detail);
    assert!(mapped.detail.contains("frame 9"), "{}", mapped.detail);
    assert!(!mapped.detail.contains("frame 10"), "{}", mapped.detail);
    mapped.validate().unwrap();
}

#[test]
fn an_empty_anomaly_list_still_produces_a_valid_error() {
    let mapped = quality_task_error(&[]);

    assert_eq!(mapped.code, ErrorCode::MediaInvalid);
    mapped.validate().unwrap();
}
