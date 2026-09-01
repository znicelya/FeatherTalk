use std::io;

use feathertalk_domain::{ErrorCode, MAX_DETAIL_CHARS, TaskError, TaskStage};
use feathertalk_media::MediaError;
use feathertalk_project::ProjectError;

/// Both commands in this slice are single-shot checks that run before any long
/// pipeline stage exists, so every failure is reported as happening while the
/// task was being prepared. `TaskError.stage` must never be terminal.
const FAILURE_STAGE: TaskStage = TaskStage::Preparing;

pub fn project_task_error(error: &ProjectError) -> TaskError {
    let code = project_error_code(error);
    TaskError::new(
        code,
        project_summary(error),
        &clamp(&error.to_string()),
        FAILURE_STAGE,
    )
}

pub fn media_task_error(error: &MediaError) -> TaskError {
    let code = media_error_code(error);
    TaskError::new(
        code,
        media_summary(error),
        &clamp(&error.to_string()),
        FAILURE_STAGE,
    )
}

pub fn is_media_cancellation(error: &MediaError) -> bool {
    matches!(error, MediaError::ToolCancelled { .. })
}

fn project_error_code(error: &ProjectError) -> ErrorCode {
    match error {
        ProjectError::Io { source, .. } => io_error_code(source),
        ProjectError::ManifestTooLarge { .. }
        | ProjectError::InvalidUtf8 { .. }
        | ProjectError::InvalidJson { .. }
        | ProjectError::UnsupportedSchemaVersion { .. }
        | ProjectError::InvalidField { .. }
        | ProjectError::UnsafeRelativePath { .. }
        | ProjectError::Symlink { .. }
        | ProjectError::InvalidFilesystemEntry { .. }
        | ProjectError::EmptyArtifact { .. }
        | ProjectError::LockedAssetMutation { .. } => ErrorCode::MediaInvalid,
        ProjectError::AtomicReplacementUnsupported { .. } => ErrorCode::WorkerCrashed,
    }
}

fn project_summary(error: &ProjectError) -> &'static str {
    match error {
        ProjectError::Io { source, .. } => io_summary(source),
        ProjectError::ManifestTooLarge { .. } => "项目清单过大",
        ProjectError::InvalidUtf8 { .. } => "项目清单不是有效的 UTF-8 文本",
        ProjectError::InvalidJson { .. } => "项目清单 JSON 格式错误",
        ProjectError::UnsupportedSchemaVersion { .. } => "项目清单版本不受支持",
        ProjectError::InvalidField { .. } => "项目清单字段无效",
        ProjectError::UnsafeRelativePath { .. } => "项目清单包含不安全的相对路径",
        ProjectError::Symlink { .. } => "项目目录包含符号链接",
        ProjectError::InvalidFilesystemEntry { .. } => "项目目录结构不符合要求",
        ProjectError::EmptyArtifact { .. } => "项目素材文件为空",
        ProjectError::LockedAssetMutation { .. } => "素材包已锁定，无法修改",
        ProjectError::AtomicReplacementUnsupported { .. } => "当前文件系统不支持原子替换",
    }
}

fn media_error_code(error: &MediaError) -> ErrorCode {
    match error {
        MediaError::Io { source, .. } => io_error_code(source),
        MediaError::InputMissing { .. }
        | MediaError::InputNotRegularFile { .. }
        | MediaError::SymlinkNotAllowed { .. }
        | MediaError::InvalidToolchain { .. }
        | MediaError::ProbeTooLarge { .. }
        | MediaError::ProbeJson { .. }
        | MediaError::ProbeContract { .. }
        | MediaError::MissingStream { .. }
        | MediaError::DuplicateStream { .. } => ErrorCode::MediaInvalid,
        // The runtime intercepts cancellation before it reaches this mapper.
        // The arm exists so the mapping stays total and so a cancellation that
        // somehow arrives here is not mislabelled as a crash.
        MediaError::ToolCancelled { .. } => ErrorCode::TaskCancelled,
        MediaError::OutputDirectoryInvalid { .. }
        | MediaError::OutputInsideInput { .. }
        | MediaError::OutputConflictsWithInput { .. }
        | MediaError::OutputDestinationInvalid { .. }
        | MediaError::UnsupportedTarget { .. }
        | MediaError::ToolFailed { .. }
        | MediaError::ToolTimedOut { .. }
        | MediaError::ToolOutputTooLarge { .. }
        | MediaError::ToolSpawn { .. }
        | MediaError::NormalizationVerificationFailed { .. }
        | MediaError::OutputCommitFailed { .. }
        | MediaError::OutputRollbackFailed { .. } => ErrorCode::WorkerCrashed,
    }
}

fn media_summary(error: &MediaError) -> &'static str {
    match error {
        MediaError::Io { source, .. } => io_summary(source),
        MediaError::InputMissing { .. } => "找不到输入文件",
        MediaError::InputNotRegularFile { .. } => "输入不是常规文件",
        MediaError::SymlinkNotAllowed { .. } => "输入路径包含符号链接",
        MediaError::InvalidToolchain { .. } => "媒体工具链配置无效",
        MediaError::ProbeTooLarge { .. } => "媒体探测输出过大",
        MediaError::ProbeJson { .. } => "媒体探测输出不是有效 JSON",
        MediaError::ProbeContract { .. } => "媒体探测结果缺少必需字段",
        MediaError::MissingStream { .. } => "媒体文件缺少必需的音视频流",
        MediaError::DuplicateStream { .. } => "媒体文件包含重复的音视频流",
        MediaError::ToolCancelled { .. } => "任务已取消",
        MediaError::OutputDirectoryInvalid { .. }
        | MediaError::OutputInsideInput { .. }
        | MediaError::OutputConflictsWithInput { .. }
        | MediaError::OutputDestinationInvalid { .. } => "输出路径无效",
        MediaError::UnsupportedTarget { .. } => "不支持的媒体转换目标",
        MediaError::ToolFailed { .. } => "媒体工具执行失败",
        MediaError::ToolTimedOut { .. } => "媒体工具执行超时",
        MediaError::ToolOutputTooLarge { .. } => "媒体工具输出过大",
        MediaError::ToolSpawn { .. } => "无法启动媒体工具",
        MediaError::NormalizationVerificationFailed { .. } => "媒体规范化结果校验失败",
        MediaError::OutputCommitFailed { .. } => "写入输出文件失败",
        MediaError::OutputRollbackFailed { .. } => "写入失败后回滚也失败",
    }
}

fn io_error_code(source: &io::Error) -> ErrorCode {
    match source.kind() {
        io::ErrorKind::StorageFull | io::ErrorKind::QuotaExceeded => ErrorCode::DiskSpaceLow,
        _ => ErrorCode::WorkerCrashed,
    }
}

fn io_summary(source: &io::Error) -> &'static str {
    match source.kind() {
        io::ErrorKind::StorageFull | io::ErrorKind::QuotaExceeded => "磁盘空间不足",
        _ => "文件读写失败",
    }
}

/// `TaskError::validate` counts characters, not bytes, so the detail is clamped
/// on a character boundary.
fn clamp(detail: &str) -> String {
    detail.chars().take(MAX_DETAIL_CHARS).collect()
}
