# Worker Protocol and Task Domain Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `feathertalk-domain` crate so the desktop process, the worker process, and the CLI share one versioned vocabulary for tasks, commands, stage events, progress, metrics, errors, and JSON Lines framing.

**Architecture:** One dependency-light crate at the bottom of the workspace. Closed serde enums for every wire vocabulary so a missing command or stage fails at compile time in all three consumers. Pure types plus a line codec and generic `std::io` adapters; no process spawning, no command dispatch, no GPUI.

**Tech Stack:** Rust 2024 edition, rust-version 1.92, `serde`, `serde_json`, `thiserror`, `time`. Test-only dev dependency on `feathertalk-project`.

**Spec:** `docs/superpowers/specs/2026-08-28-worker-protocol-task-domain-design.md`

## Global Constraints

- Production dependencies are exactly `serde`, `serde_json`, `thiserror`, `time`, all taken from `[workspace.dependencies]` with `workspace = true`. Adding any other production dependency is a plan violation.
- No dependency on `burn`, `feathertalk-media`, `feathertalk-preprocess`, `feathertalk-audio`, `feathertalk-models`, `feathertalk-training`, `feathertalk-inference`, `feathertalk-export`, `feathertalk-weights`, `feathertalk-scrfd`, `feathertalk-pfld`, or `feathertalk-frame-pipeline`. This is what keeps model code out of the UI process at compile time.
- `feathertalk-project` may appear only under `[dev-dependencies]`.
- `feathertalk-project` source files must not be modified by any task in this plan.
- `PROTOCOL_VERSION: u32 = 1`. Version comparison is exact equality; no range negotiation.
- `MAX_FRAME_BYTES: usize = 1_048_576`.
- Every enum that crosses the wire is closed, uses `#[serde(rename_all = "snake_case")]` (except `ErrorCode`, whose explicit Serde names are uppercase by contract), and every payload struct carries `#[serde(deny_unknown_fields)]`.
- Enums with data-carrying variants use adjacent tagging (`#[serde(tag = "...", content = "...")]`), never internal tagging. serde rejects `deny_unknown_fields` combined with an internal tag, so internal tagging would silently drop the strictness this contract depends on.
- Task IDs are exactly 22 characters: 13 decimal digits, `-`, 8 lowercase hex digits.
- Verification for every task: `cargo test -p feathertalk-domain --all-targets`. Verification for the final task adds `cargo test --workspace --all-targets`, `cargo check --workspace --all-targets`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --all -- --check`, and `git diff --check`, all requiring exit code 0.
- Run every command from `E:/workspace/github/FeatherTalk/rust` unless a step says otherwise.
- Commit after every task. Never commit `demo/kanghui_training_video_featherhubert_188_latest/`.

---

## File Structure

```text
rust/Cargo.toml                                  Modify: add workspace member
rust/crates/feathertalk-domain/
  Cargo.toml                                     Create
  src/lib.rs                                     Module declarations and public re-exports
  src/error.rs                                   DomainError — this crate's own error type
  src/task.rs                                    TaskId, TaskKind, TaskStatus
  src/task_error.rs                              ErrorCode, Recovery, TaskError
  src/stage.rs                                   TaskStage, projection to TaskStatus, is_terminal
  src/lifecycle.rs                               TaskLifecycle transition validator
  src/request.rs                                 Request, 12 params structs, wire-level mirrors
  src/event.rs                                   Event envelope, Progress, Metrics
  src/frame.rs                                   ClientFrame, ServerFrame, Ready capability types
  src/codec.rs                                   encode_line / decode_line, MAX_FRAME_BYTES
  src/stream.rs                                  FrameReader<R: BufRead>, FrameWriter<W: Write>
  tests/public_api.rs                            Task 1
  tests/error_model.rs                           Task 2
  tests/stage_status.rs                          Task 3
  tests/lifecycle.rs                             Task 4
  tests/request.rs                               Task 5
  tests/event.rs                                 Task 6
  tests/handshake.rs                             Task 7
  tests/frame_codec.rs                           Task 8
  tests/stream_io.rs                             Task 9
  tests/project_compatibility.rs                 Task 10
  tests/golden_frames.rs                         Task 11
docs/superpowers/specs/2026-08-17-rust-desktop-migration-design.md   Modify: Task 12
```

## Two Refinements To The Approved Spec

Planning surfaced two points the spec left underdetermined. Both are recorded here and applied by the tasks below.

**One — the spec's test inventory gains `tests/request.rs` and `tests/event.rs`.** §9 listed nine test files. `Request` and `Event` are payloads that travel inside frames, and TDD builds them before frames exist, so each needs its own file. Coverage is added, never reduced.

**Two — the `Event` envelope gains `error: Option<TaskError>`.** §6 gave the envelope `stage`, `progress`, and `metrics`. §7 requires every error to carry a user-readable summary, technical detail, stage, and recovery hint. `TaskStage::Failed { code, message }` is the compact stream marker §11 specifies verbatim, and it has no room for summary, detail, or recovery — so without this field those three fields have no transport and §7 cannot be delivered. The envelope therefore carries `error`, required to be `Some` exactly when `stage` is `Failed` and `None` otherwise, validated in Task 6. `TaskError.stage` records the stage at which the failure happened and is never itself `Failed`, `Cancelled`, or `Completed`, which also keeps the type from recursing.

## What This Plan Deliberately Does Not Build

The spec's §4 table mapping each command to the stages it emits has no task here, and that is correct: the table describes which stages a running command is expected to produce, and only the worker knows that. Slice 1 supplies the vocabulary and the transition validator; slice 2 enforces the per-command sequences and tests them against that table. The same applies to mapping the ten error codes onto `MediaError`, `PipelineError`, `TrainingError`, and their siblings — §7 assigns that to slice 2, and doing it here would require exactly the dependencies the Global Constraints forbid.

---

### Task 1: Crate skeleton, DomainError, and task identity

**Files:**
- Modify: `rust/Cargo.toml` (workspace `members` list)
- Create: `rust/crates/feathertalk-domain/Cargo.toml`
- Create: `rust/crates/feathertalk-domain/src/lib.rs`
- Create: `rust/crates/feathertalk-domain/src/error.rs`
- Create: `rust/crates/feathertalk-domain/src/task.rs`
- Test: `rust/crates/feathertalk-domain/tests/public_api.rs`

**Interfaces:**
- Consumes: nothing.
- Produces: `PROTOCOL_VERSION: u32`, `DomainError`, `TaskId::parse(&str) -> Result<TaskId, DomainError>`, `TaskId::as_str(&self) -> &str`, `TaskKind` with `as_slug(self) -> &'static str` / `from_slug(&str) -> Option<TaskKind>` / `ALL: [TaskKind; 13]`, `TaskStatus` with `is_incomplete(self) -> bool` / `ALL: [TaskStatus; 5]`.

- [x] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-domain/tests/public_api.rs`:

```rust
use feathertalk_domain::{DomainError, PROTOCOL_VERSION, TaskId, TaskKind, TaskStatus};

#[test]
fn protocol_version_is_one() {
    assert_eq!(PROTOCOL_VERSION, 1);
}

#[test]
fn task_id_accepts_the_canonical_shape_and_orders_by_time() {
    let older = TaskId::parse("1787900000000-0000000a").unwrap();
    let newer = TaskId::parse("1787900000001-0000000a").unwrap();
    assert_eq!(older.as_str(), "1787900000000-0000000a");
    assert!(older < newer);
}

#[test]
fn task_id_rejects_every_off_contract_shape() {
    for bad in [
        "",
        "1787900000000",
        "178790000000-0000000a",
        "17879000000000-0000000a",
        "1787900000000-0000000A",
        "1787900000000-0000000",
        "1787900000000_0000000a",
        "abcdefghijklm-0000000a",
    ] {
        assert!(
            matches!(TaskId::parse(bad), Err(DomainError::InvalidTaskId { .. })),
            "expected rejection for {bad:?}"
        );
    }
}
```

```rust
#[test]
fn task_kind_slugs_match_their_serde_form_and_are_all_distinct() {
    let mut seen = std::collections::BTreeSet::new();
    for kind in TaskKind::ALL {
        let slug = kind.as_slug();
        assert_eq!(serde_json::to_string(&kind).unwrap(), format!("\"{slug}\""));
        assert_eq!(TaskKind::from_slug(slug), Some(kind));
        assert!(seen.insert(slug), "duplicate slug {slug}");
    }
    assert_eq!(seen.len(), 13);
    assert_eq!(TaskKind::from_slug("no_such_command"), None);
}

#[test]
fn only_queued_and_running_are_incomplete() {
    assert_eq!(TaskStatus::ALL.len(), 5);
    for status in TaskStatus::ALL {
        let expected = matches!(status, TaskStatus::Queued | TaskStatus::Running);
        assert_eq!(status.is_incomplete(), expected, "{status:?}");
    }
}
```

- [x] **Step 2: Run the test to verify it fails**

Run: `cargo test -p feathertalk-domain --all-targets`

Expected: FAIL. Cargo reports `error: package ID specification 'feathertalk-domain' did not match any packages` because the crate does not exist yet.

- [x] **Step 3: Register the crate in the workspace**

In `rust/Cargo.toml`, add `"crates/feathertalk-domain",` to `members` as the first entry, before `"crates/feathertalk-face"`. Keep the existing entries and the `exclude` line untouched.

Create `rust/crates/feathertalk-domain/Cargo.toml`:

```toml
[package]
name = "feathertalk-domain"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[dependencies]
serde = { workspace = true }
serde_json = { workspace = true }
thiserror = { workspace = true }
time = { workspace = true }
```

- [x] **Step 4: Implement DomainError**

Create `rust/crates/feathertalk-domain/src/error.rs`:

```rust
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DomainError {
    #[error("invalid task id: {reason}")]
    InvalidTaskId { reason: String },
    #[error("invalid task transition from {from} to {to}")]
    InvalidTransition { from: &'static str, to: &'static str },
    #[error("frame exceeds the {limit} byte limit")]
    FrameTooLong { limit: usize },
    #[error("malformed frame: {reason}")]
    MalformedFrame { reason: String },
    #[error("protocol version mismatch: expected {expected}, got {actual}")]
    ProtocolVersion { expected: u32, actual: u32 },
    #[error("invalid {field}: {reason}")]
    InvalidField {
        field: &'static str,
        reason: String,
    },
}
```

- [x] **Step 5: Implement task identity**

Create `rust/crates/feathertalk-domain/src/task.rs`:

```rust
use serde::{Deserialize, Serialize};

use crate::DomainError;

pub const TASK_ID_MILLIS_DIGITS: usize = 13;
pub const TASK_ID_SUFFIX_DIGITS: usize = 8;
pub const TASK_ID_LEN: usize = TASK_ID_MILLIS_DIGITS + 1 + TASK_ID_SUFFIX_DIGITS;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TaskId(String);

impl TaskId {
    pub fn parse(value: &str) -> Result<Self, DomainError> {
        let invalid = |reason: &str| DomainError::InvalidTaskId {
            reason: reason.to_owned(),
        };
        if value.len() != TASK_ID_LEN {
            return Err(invalid("must be exactly 22 characters"));
        }
        let (millis, suffix) = value.split_at(TASK_ID_MILLIS_DIGITS);
        let Some(suffix) = suffix.strip_prefix('-') else {
            return Err(invalid("must separate millis and suffix with '-'"));
        };
        if !millis.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(invalid("millis must be 13 decimal digits"));
        }
        if !suffix
            .bytes()
            .all(|byte| byte.is_ascii_digit() || byte.is_ascii_lowercase() && byte <= b'f')
        {
            return Err(invalid("suffix must be 8 lowercase hex digits"));
        }
        Ok(Self(value.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}
```

Ordering comes free: `Ord` on the inner `String` is lexicographic, and a fixed-width zero-padded millisecond prefix makes lexicographic order equal to time order. Do not hand-write `Ord`.

Append to the same file:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    ProbeMedia,
    NormalizeMedia,
    ValidateProject,
    LockAssetPackage,
    ExtractFrames,
    ExtractFeatures,
    Train,
    Render,
    InspectModel,
    ImportLegacyModel,
    ExportModelPackage,
    ExportOnnx,
    MigrateLegacyFeatures,
}

impl TaskKind {
    pub const ALL: [Self; 13] = [
        Self::ProbeMedia,
        Self::NormalizeMedia,
        Self::ValidateProject,
        Self::LockAssetPackage,
        Self::ExtractFrames,
        Self::ExtractFeatures,
        Self::Train,
        Self::Render,
        Self::InspectModel,
        Self::ImportLegacyModel,
        Self::ExportModelPackage,
        Self::ExportOnnx,
        Self::MigrateLegacyFeatures,
    ];

    pub fn as_slug(self) -> &'static str {
        match self {
            Self::ProbeMedia => "probe_media",
            Self::NormalizeMedia => "normalize_media",
            Self::ValidateProject => "validate_project",
            Self::LockAssetPackage => "lock_asset_package",
            Self::ExtractFrames => "extract_frames",
            Self::ExtractFeatures => "extract_features",
            Self::Train => "train",
            Self::Render => "render",
            Self::InspectModel => "inspect_model",
            Self::ImportLegacyModel => "import_legacy_model",
            Self::ExportModelPackage => "export_model_package",
            Self::ExportOnnx => "export_onnx",
            Self::MigrateLegacyFeatures => "migrate_legacy_features",
        }
    }

    pub fn from_slug(slug: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_slug() == slug)
    }
}
```

Append to the same file:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl TaskStatus {
    pub const ALL: [Self; 5] = [
        Self::Queued,
        Self::Running,
        Self::Completed,
        Self::Failed,
        Self::Cancelled,
    ];

    pub fn is_incomplete(self) -> bool {
        matches!(self, Self::Queued | Self::Running)
    }
}
```

Create `rust/crates/feathertalk-domain/src/lib.rs`:

```rust
mod error;
mod task;

pub const PROTOCOL_VERSION: u32 = 1;

pub use error::DomainError;
pub use task::{
    TASK_ID_LEN, TASK_ID_MILLIS_DIGITS, TASK_ID_SUFFIX_DIGITS, TaskId, TaskKind, TaskStatus,
};
```

- [x] **Step 6: Run the tests to verify they pass**

Run: `cargo test -p feathertalk-domain --all-targets`

Expected: PASS, 5 passed, 0 failed.

- [x] **Step 7: Format and commit**

```bash
cargo fmt --all
git add rust/Cargo.toml rust/crates/feathertalk-domain
git commit -m "feat: add task domain identity types"
```

---

### Task 2: Protocol error model

**Files:**
- Create: `rust/crates/feathertalk-domain/src/task_error.rs`
- Modify: `rust/crates/feathertalk-domain/src/lib.rs`
- Test: `rust/crates/feathertalk-domain/tests/error_model.rs`

**Interfaces:**
- Consumes: `DomainError` from Task 1.
- Produces: `ErrorCode` with `ALL: [ErrorCode; 10]` / `as_wire(self) -> &'static str` / `default_recovery(self) -> Recovery`, `Recovery`, `TaskError { code, summary, detail, recovery }` built by `TaskError::new(code, summary, detail) -> TaskError`, `TaskError::validate(&self) -> Result<(), DomainError>`, `MAX_SUMMARY_CHARS: usize`, `MAX_DETAIL_CHARS: usize`. Task 3 adds the `stage` field.

**Ordering note for the implementer:** `TaskError.stage` must hold a `TaskStage`, and `TaskStage::Failed` must hold an `ErrorCode`. Task 2 therefore defines `ErrorCode`, `Recovery`, and a `TaskError` **without** the `stage` field; Task 3 defines `TaskStage` and adds `stage` to `TaskError` along with the validation rule that rejects terminal stages. Do not try to write both halves in one task — the compiler cannot accept a forward reference.

- [x] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-domain/tests/error_model.rs`:

```rust
use feathertalk_domain::{DomainError, ErrorCode, Recovery, TaskError};

#[test]
fn every_error_code_has_the_wire_form_from_the_design() {
    let expected = [
        "MEDIA_INVALID",
        "FACE_NOT_FOUND",
        "LANDMARK_INVALID",
        "FEATURE_SHAPE_MISMATCH",
        "MODEL_INCOMPATIBLE",
        "GPU_OUT_OF_MEMORY",
        "GPU_DEVICE_LOST",
        "DISK_SPACE_LOW",
        "WORKER_CRASHED",
        "TASK_CANCELLED",
    ];
    assert_eq!(ErrorCode::ALL.len(), 10);
    for (code, wire) in ErrorCode::ALL.into_iter().zip(expected) {
        assert_eq!(code.as_wire(), wire);
        assert_eq!(serde_json::to_string(&code).unwrap(), format!("\"{wire}\""));
    }
}

#[test]
fn every_error_code_maps_to_an_actionable_recovery() {
    for code in ErrorCode::ALL {
        let recovery = code.default_recovery();
        if matches!(code, ErrorCode::TaskCancelled) {
            assert_eq!(recovery, Recovery::NotRecoverable);
        } else {
            assert_ne!(recovery, Recovery::NotRecoverable, "{code:?}");
        }
    }
}
```

```rust
#[test]
fn validate_rejects_an_empty_summary_and_oversized_fields() {
    let ok = TaskError::new(ErrorCode::MediaInvalid, "无法读取视频", "ffprobe exit 1");
    ok.validate().unwrap();

    let empty = TaskError::new(ErrorCode::MediaInvalid, "  ", "detail");
    assert!(matches!(
        empty.validate(),
        Err(DomainError::InvalidField { field: "summary", .. })
    ));

    let long_summary = "字".repeat(feathertalk_domain::MAX_SUMMARY_CHARS + 1);
    let too_long = TaskError::new(ErrorCode::MediaInvalid, &long_summary, "detail");
    assert!(matches!(
        too_long.validate(),
        Err(DomainError::InvalidField { field: "summary", .. })
    ));

    let long_detail = "x".repeat(feathertalk_domain::MAX_DETAIL_CHARS + 1);
    let too_long = TaskError::new(ErrorCode::MediaInvalid, "摘要", &long_detail);
    assert!(matches!(
        too_long.validate(),
        Err(DomainError::InvalidField { field: "detail", .. })
    ));
}

#[test]
fn task_error_round_trips_and_rejects_unknown_fields() {
    let error = TaskError::new(ErrorCode::GpuDeviceLost, "显卡连接中断", "device lost");
    let json = serde_json::to_string(&error).unwrap();
    assert_eq!(serde_json::from_str::<TaskError>(&json).unwrap(), error);
    assert!(serde_json::from_str::<TaskError>(r#"{"code":"GPU_DEVICE_LOST","summary":"a","detail":"b","recovery":"resume_from_checkpoint","extra":1}"#).is_err());
}
```

- [x] **Step 2: Run the test to verify it fails**

Run: `cargo test -p feathertalk-domain --test error_model`

Expected: FAIL with `unresolved import` for `ErrorCode`, `Recovery`, and `TaskError`.

- [x] **Step 3: Implement the error model**

Create `rust/crates/feathertalk-domain/src/task_error.rs`:

```rust
use serde::{Deserialize, Serialize};

use crate::DomainError;

pub const MAX_SUMMARY_CHARS: usize = 200;
pub const MAX_DETAIL_CHARS: usize = 4_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCode {
    #[serde(rename = "MEDIA_INVALID")]
    MediaInvalid,
    #[serde(rename = "FACE_NOT_FOUND")]
    FaceNotFound,
    #[serde(rename = "LANDMARK_INVALID")]
    LandmarkInvalid,
    #[serde(rename = "FEATURE_SHAPE_MISMATCH")]
    FeatureShapeMismatch,
    #[serde(rename = "MODEL_INCOMPATIBLE")]
    ModelIncompatible,
    #[serde(rename = "GPU_OUT_OF_MEMORY")]
    GpuOutOfMemory,
    #[serde(rename = "GPU_DEVICE_LOST")]
    GpuDeviceLost,
    #[serde(rename = "DISK_SPACE_LOW")]
    DiskSpaceLow,
    #[serde(rename = "WORKER_CRASHED")]
    WorkerCrashed,
    #[serde(rename = "TASK_CANCELLED")]
    TaskCancelled,
}
```

Append to the same file:

```rust
impl ErrorCode {
    pub const ALL: [Self; 10] = [
        Self::MediaInvalid,
        Self::FaceNotFound,
        Self::LandmarkInvalid,
        Self::FeatureShapeMismatch,
        Self::ModelIncompatible,
        Self::GpuOutOfMemory,
        Self::GpuDeviceLost,
        Self::DiskSpaceLow,
        Self::WorkerCrashed,
        Self::TaskCancelled,
    ];

    pub fn as_wire(self) -> &'static str {
        match self {
            Self::MediaInvalid => "MEDIA_INVALID",
            Self::FaceNotFound => "FACE_NOT_FOUND",
            Self::LandmarkInvalid => "LANDMARK_INVALID",
            Self::FeatureShapeMismatch => "FEATURE_SHAPE_MISMATCH",
            Self::ModelIncompatible => "MODEL_INCOMPATIBLE",
            Self::GpuOutOfMemory => "GPU_OUT_OF_MEMORY",
            Self::GpuDeviceLost => "GPU_DEVICE_LOST",
            Self::DiskSpaceLow => "DISK_SPACE_LOW",
            Self::WorkerCrashed => "WORKER_CRASHED",
            Self::TaskCancelled => "TASK_CANCELLED",
        }
    }

    pub fn default_recovery(self) -> Recovery {
        match self {
            Self::MediaInvalid => Recovery::Retry,
            Self::FaceNotFound | Self::LandmarkInvalid => Recovery::ExcludeBadFrames,
            Self::FeatureShapeMismatch => Recovery::Retry,
            Self::ModelIncompatible => Recovery::ReimportModel,
            Self::GpuOutOfMemory => Recovery::SelectDifferentAdapter,
            Self::GpuDeviceLost | Self::WorkerCrashed => Recovery::ResumeFromCheckpoint,
            Self::DiskSpaceLow => Recovery::FreeDiskSpace,
            Self::TaskCancelled => Recovery::NotRecoverable,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Recovery {
    Retry,
    ResumeFromCheckpoint,
    FreeDiskSpace,
    SelectDifferentAdapter,
    ExcludeBadFrames,
    ReimportModel,
    NotRecoverable,
}
```

Append to the same file:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskError {
    pub code: ErrorCode,
    pub summary: String,
    pub detail: String,
    pub recovery: Recovery,
}

impl TaskError {
    pub fn new(code: ErrorCode, summary: &str, detail: &str) -> Self {
        Self {
            code,
            summary: summary.to_owned(),
            detail: detail.to_owned(),
            recovery: code.default_recovery(),
        }
    }

    pub fn validate(&self) -> Result<(), DomainError> {
        let summary_chars = self.summary.trim().chars().count();
        if summary_chars == 0 || self.summary.chars().count() > MAX_SUMMARY_CHARS {
            return Err(DomainError::InvalidField {
                field: "summary",
                reason: format!("must be 1-{MAX_SUMMARY_CHARS} characters after trimming"),
            });
        }
        if self.detail.chars().count() > MAX_DETAIL_CHARS {
            return Err(DomainError::InvalidField {
                field: "detail",
                reason: format!("must be at most {MAX_DETAIL_CHARS} characters"),
            });
        }
        Ok(())
    }
}
```

In `src/lib.rs`, add `mod task_error;` and extend the re-exports:

```rust
pub use task_error::{ErrorCode, MAX_DETAIL_CHARS, MAX_SUMMARY_CHARS, Recovery, TaskError};
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p feathertalk-domain --all-targets`

Expected: PASS, 9 passed, 0 failed.

- [x] **Step 5: Format and commit**

```bash
cargo fmt --all
git add rust/crates/feathertalk-domain
git commit -m "feat: add protocol error codes and recovery hints"
```

---

### Task 3: Stage vocabulary and projection to status

**Files:**
- Create: `rust/crates/feathertalk-domain/src/stage.rs`
- Modify: `rust/crates/feathertalk-domain/src/task_error.rs` (add the `stage` field and its rule)
- Modify: `rust/crates/feathertalk-domain/src/lib.rs`
- Test: `rust/crates/feathertalk-domain/tests/stage_status.rs`
- Test: `rust/crates/feathertalk-domain/tests/error_model.rs` (update calls to `TaskError::new`)

**Interfaces:**
- Consumes: `ErrorCode`, `TaskStatus`, `DomainError`.
- Produces: `TaskStage` with `ALL_UNIT_SAMPLES: [TaskStage; 13]` / `status(&self) -> TaskStatus` / `is_terminal(&self) -> bool` / `as_slug(&self) -> &'static str`, and `TaskError::new(code, summary, detail, stage)` now taking a fourth argument.

- [x] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-domain/tests/stage_status.rs`:

```rust
use feathertalk_domain::{ErrorCode, TaskStage, TaskStatus};

#[test]
fn the_stage_vocabulary_has_thirteen_variants() {
    assert_eq!(TaskStage::ALL_UNIT_SAMPLES.len(), 13);
}

#[test]
fn every_stage_projects_to_exactly_one_status() {
    for stage in TaskStage::ALL_UNIT_SAMPLES {
        let expected = match &stage {
            TaskStage::Queued => TaskStatus::Queued,
            TaskStage::Completed => TaskStatus::Completed,
            TaskStage::Failed { .. } => TaskStatus::Failed,
            TaskStage::Cancelled => TaskStatus::Cancelled,
            _ => TaskStatus::Running,
        };
        assert_eq!(stage.status(), expected, "{stage:?}");
    }
}

#[test]
fn only_completed_failed_and_cancelled_are_terminal() {
    for stage in TaskStage::ALL_UNIT_SAMPLES {
        let expected = matches!(
            stage,
            TaskStage::Completed | TaskStage::Failed { .. } | TaskStage::Cancelled
        );
        assert_eq!(stage.is_terminal(), expected, "{stage:?}");
    }
}
```

```rust
#[test]
fn data_carrying_stages_use_adjacent_tagging_on_the_wire() {
    let training = TaskStage::Training {
        epoch: 3,
        step: 1200,
        loss: 0.0425,
    };
    assert_eq!(
        serde_json::to_string(&training).unwrap(),
        r#"{"stage":"training","data":{"epoch":3,"step":1200,"loss":0.0425}}"#
    );
    assert_eq!(
        serde_json::to_string(&TaskStage::Preparing).unwrap(),
        r#"{"stage":"preparing"}"#
    );
    let failed = TaskStage::Failed {
        code: ErrorCode::DiskSpaceLow,
        message: "磁盘空间不足".to_owned(),
    };
    let json = serde_json::to_string(&failed).unwrap();
    assert_eq!(serde_json::from_str::<TaskStage>(&json).unwrap(), failed);
}

#[test]
fn task_error_stage_must_not_be_terminal() {
    use feathertalk_domain::{DomainError, TaskError};

    let ok = TaskError::new(
        ErrorCode::DiskSpaceLow,
        "磁盘空间不足",
        "needed 4 GiB",
        TaskStage::Rendering {
            frame: 10,
            total: 900,
        },
    );
    ok.validate().unwrap();

    let bad = TaskError::new(
        ErrorCode::DiskSpaceLow,
        "磁盘空间不足",
        "needed 4 GiB",
        TaskStage::Completed,
    );
    assert!(matches!(
        bad.validate(),
        Err(DomainError::InvalidField { field: "stage", .. })
    ));
}
```

- [x] **Step 2: Run the test to verify it fails**

Run: `cargo test -p feathertalk-domain --test stage_status`

Expected: FAIL with `unresolved import` for `TaskStage`.

- [x] **Step 3: Implement the stage vocabulary**

Create `rust/crates/feathertalk-domain/src/stage.rs`:

```rust
use serde::{Deserialize, Serialize};

use crate::{ErrorCode, TaskStatus};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "stage", content = "data", rename_all = "snake_case")]
pub enum TaskStage {
    Queued,
    Preparing,
    ExtractingAudio,
    ExtractingFrames,
    DetectingFaces,
    ExtractingFeatures,
    Training {
        epoch: u32,
        step: u64,
        loss: f64,
    },
    Importing,
    Exporting,
    Rendering {
        frame: u64,
        total: u64,
    },
    Completed,
    Failed {
        code: ErrorCode,
        message: String,
    },
    Cancelled,
}

impl TaskStage {
    /// One sample per variant, for exhaustive tests. Data-carrying variants use
    /// arbitrary but fixed payloads.
    pub const ALL_UNIT_SAMPLES: [Self; 13] = [
        Self::Queued,
        Self::Preparing,
        Self::ExtractingAudio,
        Self::ExtractingFrames,
        Self::DetectingFaces,
        Self::ExtractingFeatures,
        Self::Training {
            epoch: 0,
            step: 0,
            loss: 0.0,
        },
        Self::Importing,
        Self::Exporting,
        Self::Rendering { frame: 0, total: 1 },
        Self::Completed,
        Self::Failed {
            code: ErrorCode::WorkerCrashed,
            message: String::new(),
        },
        Self::Cancelled,
    ];
}
```

`ALL_UNIT_SAMPLES` can be a `const` because `String::new()` is const. Do not replace it with a function.

Append to `src/stage.rs`:

```rust
impl TaskStage {
    pub fn status(&self) -> TaskStatus {
        match self {
            Self::Queued => TaskStatus::Queued,
            Self::Completed => TaskStatus::Completed,
            Self::Failed { .. } => TaskStatus::Failed,
            Self::Cancelled => TaskStatus::Cancelled,
            Self::Preparing
            | Self::ExtractingAudio
            | Self::ExtractingFrames
            | Self::DetectingFaces
            | Self::ExtractingFeatures
            | Self::Training { .. }
            | Self::Importing
            | Self::Exporting
            | Self::Rendering { .. } => TaskStatus::Running,
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed { .. } | Self::Cancelled
        )
    }

    pub fn as_slug(&self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Preparing => "preparing",
            Self::ExtractingAudio => "extracting_audio",
            Self::ExtractingFrames => "extracting_frames",
            Self::DetectingFaces => "detecting_faces",
            Self::ExtractingFeatures => "extracting_features",
            Self::Training { .. } => "training",
            Self::Importing => "importing",
            Self::Exporting => "exporting",
            Self::Rendering { .. } => "rendering",
            Self::Completed => "completed",
            Self::Failed { .. } => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}
```

Write `status` with every non-terminal variant spelled out rather than a `_ =>` arm. That is the whole point: adding a stage later must break this match.

- [x] **Step 4: Add the stage field to TaskError**

In `src/task_error.rs`, add `use crate::TaskStage;`, add the field, and extend `new` and `validate`. **Also drop `Eq` from `TaskError`'s derive list**, leaving `#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]`: `TaskStage::Training` and the metrics carry `f64`, so nothing containing a `TaskStage` can be `Eq`. Leaving `Eq` on will not compile.

```rust
pub struct TaskError {
    pub code: ErrorCode,
    pub summary: String,
    pub detail: String,
    pub stage: TaskStage,
    pub recovery: Recovery,
}

impl TaskError {
    pub fn new(code: ErrorCode, summary: &str, detail: &str, stage: TaskStage) -> Self {
        Self {
            code,
            summary: summary.to_owned(),
            detail: detail.to_owned(),
            stage,
            recovery: code.default_recovery(),
        }
    }
}
```

Insert this check at the end of `validate`, before `Ok(())`:

```rust
        if self.stage.is_terminal() {
            return Err(DomainError::InvalidField {
                field: "stage",
                reason: "must be the stage the failure occurred in, not a terminal stage".into(),
            });
        }
```

In `src/lib.rs`, add `mod stage;` and `pub use stage::TaskStage;`.

In `tests/error_model.rs`, add a fourth argument `TaskStage::Preparing` to every `TaskError::new` call, import `TaskStage`, and update the unknown-field JSON literal to include `"stage":{"stage":"preparing"}` before `"recovery"`.

- [x] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p feathertalk-domain --all-targets`

Expected: PASS, 14 passed, 0 failed.

- [x] **Step 6: Format and commit**

```bash
cargo fmt --all
git add rust/crates/feathertalk-domain
git commit -m "feat: add task stage vocabulary and status projection"
```

---

### Task 4: Lifecycle transition validator

**Files:**
- Create: `rust/crates/feathertalk-domain/src/lifecycle.rs`
- Modify: `rust/crates/feathertalk-domain/src/lib.rs`
- Test: `rust/crates/feathertalk-domain/tests/lifecycle.rs`

**Interfaces:**
- Consumes: `TaskStage`, `DomainError`.
- Produces: `TaskLifecycle::new() -> TaskLifecycle`, `TaskLifecycle::current(&self) -> &TaskStage`, `TaskLifecycle::advance(&mut self, TaskStage) -> Result<(), DomainError>`, `TaskLifecycle::request_cancel(&mut self) -> Result<bool, DomainError>`, `TaskLifecycle::is_terminal(&self) -> bool`.

`request_cancel` returns `Ok(true)` when this call moved the task into `Cancelled`, and `Ok(false)` when the task was already terminal. It never returns `Err`. That signature is what makes cancellation idempotent for callers that do not know the current state, and it guarantees at most one `Cancelled` per task.

- [x] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-domain/tests/lifecycle.rs`:

```rust
use feathertalk_domain::{DomainError, ErrorCode, TaskLifecycle, TaskStage};

#[test]
fn a_new_lifecycle_starts_queued_and_is_not_terminal() {
    let lifecycle = TaskLifecycle::new();
    assert_eq!(lifecycle.current(), &TaskStage::Queued);
    assert!(!lifecycle.is_terminal());
}

#[test]
fn a_normal_render_run_advances_through_every_stage() {
    let mut lifecycle = TaskLifecycle::new();
    lifecycle.advance(TaskStage::Preparing).unwrap();
    lifecycle
        .advance(TaskStage::Rendering {
            frame: 1,
            total: 900,
        })
        .unwrap();
    lifecycle
        .advance(TaskStage::Rendering {
            frame: 2,
            total: 900,
        })
        .unwrap();
    lifecycle.advance(TaskStage::Completed).unwrap();
    assert!(lifecycle.is_terminal());
}

#[test]
fn advancing_out_of_a_terminal_stage_is_rejected() {
    let mut lifecycle = TaskLifecycle::new();
    lifecycle.advance(TaskStage::Completed).unwrap();
    assert!(matches!(
        lifecycle.advance(TaskStage::Cancelled),
        Err(DomainError::InvalidTransition {
            from: "completed",
            to: "cancelled"
        })
    ));
    assert_eq!(lifecycle.current(), &TaskStage::Completed);
}
```

```rust
#[test]
fn repeated_cancel_is_idempotent_and_yields_one_cancelled() {
    let mut lifecycle = TaskLifecycle::new();
    lifecycle.advance(TaskStage::Preparing).unwrap();
    assert_eq!(lifecycle.request_cancel().unwrap(), true);
    assert_eq!(lifecycle.current(), &TaskStage::Cancelled);
    for _ in 0..5 {
        assert_eq!(lifecycle.request_cancel().unwrap(), false);
        assert_eq!(lifecycle.current(), &TaskStage::Cancelled);
    }
}

#[test]
fn cancel_after_completion_does_not_overwrite_the_outcome() {
    let mut lifecycle = TaskLifecycle::new();
    lifecycle.advance(TaskStage::Completed).unwrap();
    assert_eq!(lifecycle.request_cancel().unwrap(), false);
    assert_eq!(lifecycle.current(), &TaskStage::Completed);

    let mut failed = TaskLifecycle::new();
    failed
        .advance(TaskStage::Failed {
            code: ErrorCode::DiskSpaceLow,
            message: "磁盘空间不足".to_owned(),
        })
        .unwrap();
    assert_eq!(failed.request_cancel().unwrap(), false);
    assert!(matches!(failed.current(), TaskStage::Failed { .. }));
}

#[test]
fn queued_cannot_be_re_entered() {
    let mut lifecycle = TaskLifecycle::new();
    lifecycle.advance(TaskStage::Preparing).unwrap();
    assert!(matches!(
        lifecycle.advance(TaskStage::Queued),
        Err(DomainError::InvalidTransition {
            from: "preparing",
            to: "queued"
        })
    ));
}
```

- [x] **Step 2: Run the test to verify it fails**

Run: `cargo test -p feathertalk-domain --test lifecycle`

Expected: FAIL with `unresolved import` for `TaskLifecycle`.

- [x] **Step 3: Implement the validator**

Create `rust/crates/feathertalk-domain/src/lifecycle.rs`:

```rust
use crate::{DomainError, TaskStage};

#[derive(Debug, Clone, PartialEq)]
pub struct TaskLifecycle {
    current: TaskStage,
}

impl Default for TaskLifecycle {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskLifecycle {
    pub fn new() -> Self {
        Self {
            current: TaskStage::Queued,
        }
    }

    pub fn current(&self) -> &TaskStage {
        &self.current
    }

    pub fn is_terminal(&self) -> bool {
        self.current.is_terminal()
    }

    pub fn advance(&mut self, next: TaskStage) -> Result<(), DomainError> {
        if self.current.is_terminal() || matches!(next, TaskStage::Queued) {
            return Err(DomainError::InvalidTransition {
                from: self.current.as_slug(),
                to: next.as_slug(),
            });
        }
        self.current = next;
        Ok(())
    }

    pub fn request_cancel(&mut self) -> Result<bool, DomainError> {
        if self.current.is_terminal() {
            return Ok(false);
        }
        self.current = TaskStage::Cancelled;
        Ok(true)
    }
}
```

`advance` rejects two things: leaving a terminal stage, and re-entering `Queued`. Everything else is allowed, because the legal order of intermediate stages depends on which command is running and that knowledge lives in the worker, not here.

In `src/lib.rs`, add `mod lifecycle;` and `pub use lifecycle::TaskLifecycle;`.

- [x] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p feathertalk-domain --all-targets`

Expected: PASS, 20 passed, 0 failed.

- [x] **Step 5: Format, lint, and commit**

```bash
cargo fmt --all
cargo clippy -p feathertalk-domain --all-targets -- -D warnings
git add rust/crates/feathertalk-domain
git commit -m "feat: add task lifecycle transition validator"
```

---

### Task 5: Command vocabulary

**Files:**
- Create: `rust/crates/feathertalk-domain/src/request.rs`
- Modify: `rust/crates/feathertalk-domain/src/lib.rs`
- Test: `rust/crates/feathertalk-domain/tests/request.rs`

**Interfaces:**
- Consumes: `TaskKind`.
- Produces: `Request` (13 variants, each wrapping a params struct), the params structs `ProbeMediaParams`, `NormalizeMediaParams`, `ProjectDirParams`, `ExtractFramesParams`, `ExtractFeaturesParams`, `TrainParams`, `RenderParams`, `InspectModelParams`, `ImportLegacyModelParams`, `ExportModelPackageParams`, `ExportOnnxParams`, `MigrateLegacyFeaturesParams`, the wire mirrors `TrainingMode`, `UnetVariant`, `LegacyModelKind`, `OnnxExportKind`, and `Request::kind(&self) -> TaskKind`.

`ValidateProject` and `LockAssetPackage` share `ProjectDirParams` because both take only a project directory. That is 12 params structs for 13 task commands. Together with the control-plane `Cancel` operation, the protocol exposes 14 operations; `Cancel` is a frame-level control operation, not a `Request` variant.

`RenderParams.max_output_frames` is a fixed `Option<u64>` on the JSON wire. The
inference crate's `RenderPlan::new` accepts a local `Option<usize>` instead;
that `usize` is not the wire contract. Slice 2's worker mapping must perform a
checked conversion (for example, `usize::try_from`) and reject values that do
not fit rather than truncating them.

The `ErrorCode` enum intentionally uses explicit uppercase Serde names (`MEDIA_INVALID`, `GPU_DEVICE_LOST`, and so on), an exception to the general `snake_case` enum convention. These names are wire-contract literals and must not be normalized.

**Why the wire mirrors exist:** `TrainingMode`, `UnetVariant`, `LegacyModelKind`, and `OnnxExportKind` duplicate enums that live in `feathertalk-training`, `feathertalk-models`, `feathertalk-weights`, and `feathertalk-export`. `domain` cannot depend on those crates without breaking the Global Constraints, so these are protocol-level types and slice 2 owns the mapping in both directions and tests it there. Do not add those crates as dependencies to make the duplication go away.

- [x] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-domain/tests/request.rs`:

```rust
use std::path::PathBuf;

use feathertalk_domain::{
    LegacyModelKind, OnnxExportKind, ProbeMediaParams, Request, TaskKind, TrainParams,
    TrainingMode, UnetVariant,
};

#[test]
fn every_task_kind_has_exactly_one_request_variant() {
    let requests = sample_requests();
    assert_eq!(requests.len(), 13);
    let mut kinds: Vec<TaskKind> = requests.iter().map(Request::kind).collect();
    kinds.sort();
    kinds.dedup();
    assert_eq!(kinds.len(), 13, "two requests reported the same TaskKind");
    for kind in TaskKind::ALL {
        assert!(kinds.contains(&kind), "no request maps to {kind:?}");
    }
}
```

```rust
#[test]
fn requests_use_adjacent_tagging_and_round_trip() {
    for request in sample_requests() {
        let json = serde_json::to_string(&request).unwrap();
        assert!(
            json.starts_with(r#"{"command":""#),
            "unexpected wire shape: {json}"
        );
        assert_eq!(serde_json::from_str::<Request>(&json).unwrap(), request);
    }
}

#[test]
fn probe_media_has_the_exact_wire_form() {
    let request = Request::ProbeMedia(ProbeMediaParams {
        input: PathBuf::from("a.mov"),
    });
    assert_eq!(
        serde_json::to_string(&request).unwrap(),
        r#"{"command":"probe_media","params":{"input":"a.mov"}}"#
    );
}

#[test]
fn params_reject_unknown_fields() {
    let bad = r#"{"command":"probe_media","params":{"input":"a.mov","extra":1}}"#;
    assert!(serde_json::from_str::<Request>(bad).is_err());
}

#[test]
fn render_treats_preview_as_a_frame_cap_not_a_separate_command() {
    let json = r#"{"command":"render","params":{"project_dir":"p","checkpoint":"c","audio":"a.wav","output":"o.mp4","max_output_frames":120}}"#;
    let request: Request = serde_json::from_str(json).unwrap();
    let Request::Render(params) = &request else {
        panic!("expected Render");
    };
    assert_eq!(params.max_output_frames, Some(120));
    assert_eq!(request.kind(), TaskKind::Render);
}

#[test]
fn training_mode_and_variant_are_independent_dimensions() {
    let params = TrainParams {
        project_dir: PathBuf::from("p"),
        mode: TrainingMode::Temporal,
        variant: UnetVariant::MobileOneUnet,
        epochs: 4,
        resume: true,
    };
    let json = serde_json::to_string(&params).unwrap();
    assert!(json.contains(r#""mode":"temporal""#));
    assert!(json.contains(r#""variant":"mobile_one_unet""#));
    assert_eq!(serde_json::from_str::<TrainParams>(&json).unwrap(), params);
}
```

Add this helper at the bottom of the same test file:

```rust
fn sample_requests() -> Vec<Request> {
    use feathertalk_domain::{
        ExportModelPackageParams, ExportOnnxParams, ExtractFeaturesParams, ExtractFramesParams,
        ImportLegacyModelParams, InspectModelParams, MigrateLegacyFeaturesParams,
        NormalizeMediaParams, ProjectDirParams, RenderParams,
    };

    let p = || PathBuf::from("p");
    vec![
        Request::ProbeMedia(ProbeMediaParams { input: p() }),
        Request::NormalizeMedia(NormalizeMediaParams {
            input: p(),
            output_dir: p(),
        }),
        Request::ValidateProject(ProjectDirParams { project_dir: p() }),
        Request::LockAssetPackage(ProjectDirParams { project_dir: p() }),
        Request::ExtractFrames(ExtractFramesParams {
            project_dir: p(),
            video: p(),
        }),
        Request::ExtractFeatures(ExtractFeaturesParams {
            project_dir: p(),
            audio: p(),
        }),
        Request::Train(TrainParams {
            project_dir: p(),
            mode: TrainingMode::Baseline,
            variant: UnetVariant::OriginalUnet,
            epochs: 1,
            resume: false,
        }),
        Request::Render(RenderParams {
            project_dir: p(),
            checkpoint: p(),
            audio: p(),
            output: p(),
            max_output_frames: None,
        }),
        Request::InspectModel(InspectModelParams { source: p() }),
        Request::ImportLegacyModel(ImportLegacyModelParams {
            source: p(),
            kind: LegacyModelKind::FeatherHubert,
            destination: p(),
        }),
        Request::ExportModelPackage(ExportModelPackageParams {
            source: p(),
            destination: p(),
        }),
        Request::ExportOnnx(ExportOnnxParams {
            source: p(),
            kind: OnnxExportKind::FeatherHubert,
            destination: p(),
        }),
        Request::MigrateLegacyFeatures(MigrateLegacyFeaturesParams {
            source: p(),
            destination: p(),
        }),
    ]
}
```

- [x] **Step 2: Run the test to verify it fails**

Run: `cargo test -p feathertalk-domain --test request`

Expected: FAIL with `unresolved import` for `Request` and the params structs.

- [x] **Step 3: Implement the wire mirrors and params structs**

Create `rust/crates/feathertalk-domain/src/request.rs`:

```rust
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::TaskKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingMode {
    Baseline,
    MouthRoi,
    Temporal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnetVariant {
    OriginalUnet,
    MobileOneUnet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LegacyModelKind {
    FeatherHubert,
    Pfld,
    OriginalUnet,
    MobileOneUnet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnnxExportKind {
    FeatherHubert,
    OriginalUnet,
    MobileOneUnet,
}

macro_rules! params {
    ($name:ident { $($field:ident : $ty:ty),* $(,)? }) => {
        #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct $name {
            $(pub $field: $ty),*
        }
    };
}

params!(ProbeMediaParams { input: PathBuf });
params!(NormalizeMediaParams { input: PathBuf, output_dir: PathBuf });
params!(ProjectDirParams { project_dir: PathBuf });
params!(ExtractFramesParams { project_dir: PathBuf, video: PathBuf });
params!(ExtractFeaturesParams { project_dir: PathBuf, audio: PathBuf });
params!(InspectModelParams { source: PathBuf });
params!(ExportModelPackageParams { source: PathBuf, destination: PathBuf });
params!(MigrateLegacyFeaturesParams { source: PathBuf, destination: PathBuf });
params!(ImportLegacyModelParams { source: PathBuf, kind: LegacyModelKind, destination: PathBuf });
params!(ExportOnnxParams { source: PathBuf, kind: OnnxExportKind, destination: PathBuf });
```

The `params!` macro exists only to keep twelve near-identical struct definitions readable. `TrainParams` and `RenderParams` are written out longhand below because they carry more fields and are the two a reader will look up most often.

- [x] **Step 4: Implement TrainParams, RenderParams, and the Request enum**

Append to `src/request.rs`:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainParams {
    pub project_dir: PathBuf,
    pub mode: TrainingMode,
    pub variant: UnetVariant,
    pub epochs: u32,
    pub resume: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RenderParams {
    pub project_dir: PathBuf,
    pub checkpoint: PathBuf,
    pub audio: PathBuf,
    pub output: PathBuf,
    /// `None` renders the full sequence; `Some(n)` caps output frames and is how
    /// a short preview is requested. Preview and full render share this one path.
    pub max_output_frames: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "command", content = "params", rename_all = "snake_case")]
pub enum Request {
    ProbeMedia(ProbeMediaParams),
    NormalizeMedia(NormalizeMediaParams),
    ValidateProject(ProjectDirParams),
    LockAssetPackage(ProjectDirParams),
    ExtractFrames(ExtractFramesParams),
    ExtractFeatures(ExtractFeaturesParams),
    Train(TrainParams),
    Render(RenderParams),
    InspectModel(InspectModelParams),
    ImportLegacyModel(ImportLegacyModelParams),
    ExportModelPackage(ExportModelPackageParams),
    ExportOnnx(ExportOnnxParams),
    MigrateLegacyFeatures(MigrateLegacyFeaturesParams),
}

impl Request {
    pub fn kind(&self) -> TaskKind {
        match self {
            Self::ProbeMedia(_) => TaskKind::ProbeMedia,
            Self::NormalizeMedia(_) => TaskKind::NormalizeMedia,
            Self::ValidateProject(_) => TaskKind::ValidateProject,
            Self::LockAssetPackage(_) => TaskKind::LockAssetPackage,
            Self::ExtractFrames(_) => TaskKind::ExtractFrames,
            Self::ExtractFeatures(_) => TaskKind::ExtractFeatures,
            Self::Train(_) => TaskKind::Train,
            Self::Render(_) => TaskKind::Render,
            Self::InspectModel(_) => TaskKind::InspectModel,
            Self::ImportLegacyModel(_) => TaskKind::ImportLegacyModel,
            Self::ExportModelPackage(_) => TaskKind::ExportModelPackage,
            Self::ExportOnnx(_) => TaskKind::ExportOnnx,
            Self::MigrateLegacyFeatures(_) => TaskKind::MigrateLegacyFeatures,
        }
    }
}
```

In `src/lib.rs`, add `mod request;` and re-export every public item from `request`.

- [x] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p feathertalk-domain --all-targets`

Expected: PASS, 26 passed, 0 failed.

- [x] **Step 6: Format, lint, and commit**

```bash
cargo fmt --all
cargo clippy -p feathertalk-domain --all-targets -- -D warnings
git add rust/crates/feathertalk-domain
git commit -m "feat: add worker command vocabulary"
```

---

### Task 6: Event envelope, progress, and metrics

**Files:**
- Create: `rust/crates/feathertalk-domain/src/event.rs`
- Modify: `rust/crates/feathertalk-domain/src/lib.rs`
- Test: `rust/crates/feathertalk-domain/tests/event.rs`

**Interfaces:**
- Consumes: `TaskId`, `TaskStage`, `TaskError`, `DomainError`, `PROTOCOL_VERSION`.
- Produces: `Progress { completed: u64, total: Option<u64> }`, `Metrics { samples_per_second, eta_seconds, vram_bytes }` with `Metrics::empty()`, `Event { protocol_version, task_id, emitted_at, stage, progress, metrics, error }` with `Event::new(TaskId, &str, TaskStage) -> Event` and `Event::validate(&self) -> Result<(), DomainError>`.

`emitted_at` is an RFC 3339 `String`, matching how `feathertalk-project` stores `updated_at`. `domain` does not read the clock; the caller supplies the timestamp.

The event validator intentionally covers only the protocol invariants listed
here (protocol version, timestamp syntax, progress ordering, and error-payload
relationships). `serde_json` rejects non-finite floating-point values while
encoding JSON. Loss/metrics finiteness beyond that encoding behavior and
rendering rules such as `frame > total` are producer/worker business
responsibilities for a later slice, not additional behavior to add here.

- [x] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-domain/tests/event.rs`:

```rust
use feathertalk_domain::{
    DomainError, ErrorCode, Event, Metrics, PROTOCOL_VERSION, Progress, TaskError, TaskId,
    TaskStage,
};

fn task_id() -> TaskId {
    TaskId::parse("1787900000000-0000000a").unwrap()
}

const NOW: &str = "2026-08-28T09:00:00Z";

#[test]
fn a_new_event_carries_the_protocol_version_and_empty_metrics() {
    let event = Event::new(task_id(), NOW, TaskStage::Preparing);
    assert_eq!(event.protocol_version, PROTOCOL_VERSION);
    assert_eq!(event.metrics, Metrics::empty());
    assert_eq!(event.progress, None);
    assert_eq!(event.error, None);
    event.validate().unwrap();
}

#[test]
fn a_failed_stage_requires_the_error_payload() {
    let mut event = Event::new(
        task_id(),
        NOW,
        TaskStage::Failed {
            code: ErrorCode::DiskSpaceLow,
            message: "磁盘空间不足".to_owned(),
        },
    );
    assert!(matches!(
        event.validate(),
        Err(DomainError::InvalidField { field: "error", .. })
    ));

    event.error = Some(TaskError::new(
        ErrorCode::DiskSpaceLow,
        "磁盘空间不足",
        "needed 4 GiB",
        TaskStage::Exporting,
    ));
    event.validate().unwrap();
}
```

```rust
#[test]
fn a_non_failed_stage_must_not_carry_an_error_payload() {
    let mut event = Event::new(task_id(), NOW, TaskStage::Exporting);
    event.error = Some(TaskError::new(
        ErrorCode::DiskSpaceLow,
        "磁盘空间不足",
        "needed 4 GiB",
        TaskStage::Exporting,
    ));
    assert!(matches!(
        event.validate(),
        Err(DomainError::InvalidField { field: "error", .. })
    ));
}

#[test]
fn progress_rejects_a_completed_count_beyond_the_total() {
    let mut event = Event::new(task_id(), NOW, TaskStage::ExtractingFrames);
    event.progress = Some(Progress {
        completed: 5,
        total: Some(4),
    });
    assert!(matches!(
        event.validate(),
        Err(DomainError::InvalidField { field: "progress", .. })
    ));

    event.progress = Some(Progress {
        completed: 5,
        total: None,
    });
    event.validate().unwrap();
}

#[test]
fn validate_rejects_a_non_rfc3339_timestamp_and_a_foreign_protocol_version() {
    let mut event = Event::new(task_id(), "yesterday", TaskStage::Preparing);
    assert!(matches!(
        event.validate(),
        Err(DomainError::InvalidField { field: "emitted_at", .. })
    ));

    event = Event::new(task_id(), NOW, TaskStage::Preparing);
    event.protocol_version = PROTOCOL_VERSION + 1;
    assert!(matches!(
        event.validate(),
        Err(DomainError::ProtocolVersion { .. })
    ));
}

#[test]
fn events_round_trip_and_reject_unknown_fields() {
    let mut event = Event::new(task_id(), NOW, TaskStage::Training { epoch: 2, step: 40, loss: 0.1 });
    event.metrics = Metrics {
        samples_per_second: Some(12.5),
        eta_seconds: Some(90.0),
        vram_bytes: Some(3_221_225_472),
    };
    let json = serde_json::to_string(&event).unwrap();
    assert_eq!(serde_json::from_str::<Event>(&json).unwrap(), event);

    let injected = json.replace(r#""metrics":{"#, r#""surprise":1,"metrics":{"#);
    assert!(serde_json::from_str::<Event>(&injected).is_err());
}
```

- [x] **Step 2: Run the test to verify it fails**

Run: `cargo test -p feathertalk-domain --test event`

Expected: FAIL with `unresolved import` for `Event`, `Metrics`, and `Progress`.

- [x] **Step 3: Implement the envelope**

Create `rust/crates/feathertalk-domain/src/event.rs`:

```rust
use serde::{Deserialize, Serialize};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

use crate::{DomainError, PROTOCOL_VERSION, TaskError, TaskId, TaskStage};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Progress {
    pub completed: u64,
    pub total: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metrics {
    pub samples_per_second: Option<f64>,
    pub eta_seconds: Option<f64>,
    pub vram_bytes: Option<u64>,
}

impl Metrics {
    pub fn empty() -> Self {
        Self::default()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Event {
    pub protocol_version: u32,
    pub task_id: TaskId,
    pub emitted_at: String,
    pub stage: TaskStage,
    pub progress: Option<Progress>,
    pub metrics: Metrics,
    pub error: Option<TaskError>,
}

impl Event {
    pub fn new(task_id: TaskId, emitted_at: &str, stage: TaskStage) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            task_id,
            emitted_at: emitted_at.to_owned(),
            stage,
            progress: None,
            metrics: Metrics::empty(),
            error: None,
        }
    }
}
```

- [x] **Step 4: Implement validation**

Append to `src/event.rs`:

```rust
impl Event {
    pub fn validate(&self) -> Result<(), DomainError> {
        if self.protocol_version != PROTOCOL_VERSION {
            return Err(DomainError::ProtocolVersion {
                expected: PROTOCOL_VERSION,
                actual: self.protocol_version,
            });
        }
        OffsetDateTime::parse(&self.emitted_at, &Rfc3339).map_err(|_| {
            DomainError::InvalidField {
                field: "emitted_at",
                reason: "must be RFC 3339".into(),
            }
        })?;
        if let Some(progress) = self.progress {
            if let Some(total) = progress.total {
                if progress.completed > total {
                    return Err(DomainError::InvalidField {
                        field: "progress",
                        reason: "completed must not exceed total".into(),
                    });
                }
            }
        }
        let is_failed = matches!(self.stage, TaskStage::Failed { .. });
        match (&self.error, is_failed) {
            (Some(error), true) => error.validate()?,
            (None, false) => {}
            (None, true) => {
                return Err(DomainError::InvalidField {
                    field: "error",
                    reason: "a failed stage must carry the error payload".into(),
                });
            }
            (Some(_), false) => {
                return Err(DomainError::InvalidField {
                    field: "error",
                    reason: "only a failed stage may carry an error payload".into(),
                });
            }
        }
        Ok(())
    }
}
```

In `src/lib.rs`, add `mod event;` and `pub use event::{Event, Metrics, Progress};`.

- [x] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p feathertalk-domain --all-targets`

Expected: PASS, 32 passed, 0 failed.

- [x] **Step 6: Format, lint, and commit**

```bash
cargo fmt --all
cargo clippy -p feathertalk-domain --all-targets -- -D warnings
git add rust/crates/feathertalk-domain
git commit -m "feat: add task event envelope with progress and metrics"
```

---

### Task 7: Frames and the handshake capability report

**Files:**
- Create: `rust/crates/feathertalk-domain/src/frame.rs`
- Modify: `rust/crates/feathertalk-domain/src/lib.rs`
- Test: `rust/crates/feathertalk-domain/tests/handshake.rs`

**Interfaces:**
- Consumes: `Event`, `Request`, `TaskId`, `DomainError`, `PROTOCOL_VERSION`.
- Produces: `Backend`, `AdapterKind`, `AdapterInfo`, `Capabilities`, `ReadyFrame`, `StartFrame`, `CancelFrame`, `ShutdownFrame`, `RejectedFrame`, `ClientFrame` (3 variants), `ServerFrame` (3 variants), `ClientFrame::protocol_version(&self) -> u32`, `ServerFrame::protocol_version(&self) -> u32`, `ReadyFrame::validate(&self) -> Result<(), DomainError>`.

Each frame struct carries `protocol_version` as its own first field, flat — there is no separate params layer. `ServerFrame::Event` wraps `Event` directly, because `Event` already carries `protocol_version`.

- [x] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-domain/tests/handshake.rs`:

```rust
use feathertalk_domain::{
    AdapterInfo, AdapterKind, Backend, CancelFrame, Capabilities, ClientFrame, DomainError,
    PROTOCOL_VERSION, ReadyFrame, ServerFrame, TaskId,
};

fn adapters() -> Vec<AdapterInfo> {
    vec![
        AdapterInfo {
            id: "dx12:nvidia:0".to_owned(),
            name: "NVIDIA GeForce RTX 4090".to_owned(),
            backend: Backend::Wgpu,
            kind: AdapterKind::Discrete,
            certified: true,
            vram_bytes: Some(25_769_803_776),
        },
        AdapterInfo {
            id: "dx12:intel:0".to_owned(),
            name: "Intel UHD Graphics".to_owned(),
            backend: Backend::Wgpu,
            kind: AdapterKind::Integrated,
            certified: false,
            vram_bytes: None,
        },
    ]
}

fn ready() -> ReadyFrame {
    ReadyFrame {
        protocol_version: PROTOCOL_VERSION,
        worker_version: "0.1.0".to_owned(),
        backends: vec![Backend::Cpu, Backend::Wgpu],
        adapters: adapters(),
        capabilities: Capabilities {
            training: true,
            wgpu_training: true,
            onnx_validation: false,
            ffmpeg: true,
        },
    }
}

#[test]
fn uncertified_adapters_are_still_reported() {
    let params = ready();
    params.validate().unwrap();
    assert_eq!(params.adapters.len(), 2);
    assert!(params.adapters.iter().any(|adapter| !adapter.certified));
}
```

```rust
#[test]
fn adapter_ids_must_be_unique_and_non_empty() {
    let mut params = ready();
    params.adapters[1].id = params.adapters[0].id.clone();
    assert!(matches!(
        params.validate(),
        Err(DomainError::InvalidField { field: "adapters", .. })
    ));

    let mut params = ready();
    params.adapters[0].id = String::new();
    assert!(matches!(
        params.validate(),
        Err(DomainError::InvalidField { field: "adapters", .. })
    ));
}

#[test]
fn a_worker_reporting_no_backend_is_rejected() {
    let mut params = ready();
    params.backends.clear();
    assert!(matches!(
        params.validate(),
        Err(DomainError::InvalidField { field: "backends", .. })
    ));
}

#[test]
fn both_frame_directions_expose_the_protocol_version() {
    let cancel = ClientFrame::Cancel(CancelFrame {
        protocol_version: PROTOCOL_VERSION,
        task_id: TaskId::parse("1787900000000-0000000a").unwrap(),
    });
    assert_eq!(cancel.protocol_version(), PROTOCOL_VERSION);
    assert_eq!(
        ServerFrame::Ready(ready()).protocol_version(),
        PROTOCOL_VERSION
    );
}

#[test]
fn frames_use_adjacent_tagging_and_round_trip() {
    let frame = ServerFrame::Ready(ready());
    let json = serde_json::to_string(&frame).unwrap();
    assert!(json.starts_with(r#"{"frame":"ready","data":{"#), "{json}");
    assert_eq!(serde_json::from_str::<ServerFrame>(&json).unwrap(), frame);
}
```

- [x] **Step 2: Run the test to verify it fails**

Run: `cargo test -p feathertalk-domain --test handshake`

Expected: FAIL with `unresolved import` for `ClientFrame`, `ServerFrame`, and the capability types.

- [x] **Step 3: Implement the capability types**

Create `rust/crates/feathertalk-domain/src/frame.rs`:

```rust
use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{DomainError, Event, Request, TaskId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Cpu,
    Wgpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterKind {
    Discrete,
    Integrated,
    Cpu,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdapterInfo {
    /// Stable identity. Slice 2 keys the "one training or inference task per
    /// adapter" rule on this value, so it must survive a worker restart.
    pub id: String,
    pub name: String,
    pub backend: Backend,
    pub kind: AdapterKind,
    /// False for adapters shown for experimental detection only. Launch support
    /// is promised for the certified set alone.
    pub certified: bool,
    pub vram_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Capabilities {
    pub training: bool,
    pub wgpu_training: bool,
    pub onnx_validation: bool,
    pub ffmpeg: bool,
}
```

- [x] **Step 4: Implement the frame structs and enums**

Append to `src/frame.rs`:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReadyFrame {
    pub protocol_version: u32,
    pub worker_version: String,
    pub backends: Vec<Backend>,
    pub adapters: Vec<AdapterInfo>,
    pub capabilities: Capabilities,
}

impl ReadyFrame {
    pub fn validate(&self) -> Result<(), DomainError> {
        if self.backends.is_empty() {
            return Err(DomainError::InvalidField {
                field: "backends",
                reason: "a worker must report at least one backend".into(),
            });
        }
        let mut seen = BTreeSet::new();
        for adapter in &self.adapters {
            if adapter.id.is_empty() {
                return Err(DomainError::InvalidField {
                    field: "adapters",
                    reason: "adapter id must not be empty".into(),
                });
            }
            if !seen.insert(adapter.id.as_str()) {
                return Err(DomainError::InvalidField {
                    field: "adapters",
                    reason: format!("duplicate adapter id {}", adapter.id),
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StartFrame {
    pub protocol_version: u32,
    pub task_id: TaskId,
    pub request: Request,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CancelFrame {
    pub protocol_version: u32,
    pub task_id: TaskId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShutdownFrame {
    pub protocol_version: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RejectedFrame {
    pub protocol_version: u32,
    pub reason: String,
}
```

- [x] **Step 5: Implement the two direction enums**

Append to `src/frame.rs`:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "frame", content = "data", rename_all = "snake_case")]
pub enum ClientFrame {
    Start(StartFrame),
    Cancel(CancelFrame),
    Shutdown(ShutdownFrame),
}

impl ClientFrame {
    pub fn protocol_version(&self) -> u32 {
        match self {
            Self::Start(frame) => frame.protocol_version,
            Self::Cancel(frame) => frame.protocol_version,
            Self::Shutdown(frame) => frame.protocol_version,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "frame", content = "data", rename_all = "snake_case")]
pub enum ServerFrame {
    Ready(ReadyFrame),
    Event(Event),
    Rejected(RejectedFrame),
}

impl ServerFrame {
    pub fn protocol_version(&self) -> u32 {
        match self {
            Self::Ready(frame) => frame.protocol_version,
            Self::Event(event) => event.protocol_version,
            Self::Rejected(frame) => frame.protocol_version,
        }
    }
}
```

`ServerFrame` derives `PartialEq` but not `Eq`, because `Event` contains `f64` fields through `TaskStage::Training` and `Metrics`. Do not add `Eq` — it will not compile.

In `src/lib.rs`, add `mod frame;` and re-export every public item from `frame`.

- [x] **Step 6: Run the tests to verify they pass**

Run: `cargo test -p feathertalk-domain --all-targets`

Expected: PASS, 37 passed, 0 failed.

- [x] **Step 7: Format, lint, and commit**

```bash
cargo fmt --all
cargo clippy -p feathertalk-domain --all-targets -- -D warnings
git add rust/crates/feathertalk-domain
git commit -m "feat: add protocol frames and worker capability report"
```

---

### Task 8: Line codec and frame length limit

**Files:**
- Create: `rust/crates/feathertalk-domain/src/codec.rs`
- Modify: `rust/crates/feathertalk-domain/src/lib.rs`
- Test: `rust/crates/feathertalk-domain/tests/frame_codec.rs`

**Interfaces:**
- Consumes: `ClientFrame`, `ServerFrame`, `DomainError`, `PROTOCOL_VERSION`.
- Produces: `MAX_FRAME_BYTES: usize`, `encode_line<T: Serialize>(&T) -> Result<String, DomainError>`, `decode_line<T: DeserializeOwned>(&str) -> Result<T, DomainError>`, `check_protocol_version(u32) -> Result<(), DomainError>`.

`encode_line` returns the JSON without a trailing newline; the writer in Task 9 appends it. `encode_line` fails with `FrameTooLong` when the encoded form exceeds `MAX_FRAME_BYTES`, so an oversized frame can never be put on the wire in the first place. `decode_line` is syntax-only: it checks the length, strips an optional delimiter, rejects blank input, and runs serde JSON deserialization, but it does not call a frame's semantic `validate()` method. Callers must invoke `ClientFrame::validate()` or `ServerFrame::validate()` after decoding and before dispatching a frame.

- [x] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-domain/tests/frame_codec.rs`:

```rust
use feathertalk_domain::{
    CancelFrame, ClientFrame, DomainError, MAX_FRAME_BYTES, PROTOCOL_VERSION, RejectedFrame,
    ServerFrame, ShutdownFrame, StartFrame, TaskId, check_protocol_version, decode_line,
    encode_line,
};

fn task_id() -> TaskId {
    TaskId::parse("1787900000000-0000000a").unwrap()
}

#[test]
fn every_client_frame_round_trips_on_one_line() {
    use feathertalk_domain::{ProbeMediaParams, Request};

    let frames = vec![
        ClientFrame::Start(StartFrame {
            protocol_version: PROTOCOL_VERSION,
            task_id: task_id(),
            request: Request::ProbeMedia(ProbeMediaParams {
                input: std::path::PathBuf::from("a.mov"),
            }),
        }),
        ClientFrame::Cancel(CancelFrame {
            protocol_version: PROTOCOL_VERSION,
            task_id: task_id(),
        }),
        ClientFrame::Shutdown(ShutdownFrame {
            protocol_version: PROTOCOL_VERSION,
        }),
    ];
    for frame in frames {
        let line = encode_line(&frame).unwrap();
        assert!(!line.contains('\n'), "encoded frame contains a newline");
        assert_eq!(decode_line::<ClientFrame>(&line).unwrap(), frame);
    }
}
```

```rust
#[test]
fn a_multiline_string_payload_still_encodes_to_one_line() {
    let frame = ServerFrame::Rejected(RejectedFrame {
        protocol_version: PROTOCOL_VERSION,
        reason: "line one\nline two".to_owned(),
    });
    let line = encode_line(&frame).unwrap();
    assert!(!line.contains('\n'));
    assert!(line.contains(r"line one\nline two"));
    assert_eq!(decode_line::<ServerFrame>(&line).unwrap(), frame);
}

#[test]
fn encoding_an_oversized_frame_is_refused() {
    let frame = ServerFrame::Rejected(RejectedFrame {
        protocol_version: PROTOCOL_VERSION,
        reason: "x".repeat(MAX_FRAME_BYTES),
    });
    assert!(matches!(
        encode_line(&frame),
        Err(DomainError::FrameTooLong {
            limit: MAX_FRAME_BYTES
        })
    ));
}

#[test]
fn decoding_an_oversized_line_is_refused_before_parsing() {
    let line = format!("{{\"frame\":\"{}\"}}", "x".repeat(MAX_FRAME_BYTES));
    assert!(matches!(
        decode_line::<ServerFrame>(&line),
        Err(DomainError::FrameTooLong { .. })
    ));
}

#[test]
fn malformed_and_unknown_frames_are_refused() {
    for bad in [
        "",
        "   ",
        "not json",
        r#"{"frame":"greetings","data":{}}"#,
        r#"{"frame":"shutdown","data":{"protocol_version":1,"extra":true}}"#,
    ] {
        assert!(
            matches!(
                decode_line::<ClientFrame>(bad),
                Err(DomainError::MalformedFrame { .. })
            ),
            "expected rejection for {bad:?}"
        );
    }
}

#[test]
fn protocol_version_comparison_is_exact() {
    check_protocol_version(PROTOCOL_VERSION).unwrap();
    for wrong in [0, PROTOCOL_VERSION + 1, u32::MAX] {
        assert!(matches!(
            check_protocol_version(wrong),
            Err(DomainError::ProtocolVersion {
                expected: PROTOCOL_VERSION,
                ..
            })
        ));
    }
}
```

- [x] **Step 2: Run the test to verify it fails**

Run: `cargo test -p feathertalk-domain --test frame_codec`

Expected: FAIL with `unresolved import` for `encode_line`, `decode_line`, and `MAX_FRAME_BYTES`.

- [x] **Step 3: Implement the codec**

Create `rust/crates/feathertalk-domain/src/codec.rs`:

```rust
use serde::{Serialize, de::DeserializeOwned};

use crate::{DomainError, PROTOCOL_VERSION};

pub const MAX_FRAME_BYTES: usize = 1_048_576;

pub fn encode_line<T: Serialize>(value: &T) -> Result<String, DomainError> {
    let line = serde_json::to_string(value).map_err(|error| DomainError::MalformedFrame {
        reason: error.to_string(),
    })?;
    if line.len() > MAX_FRAME_BYTES {
        return Err(DomainError::FrameTooLong {
            limit: MAX_FRAME_BYTES,
        });
    }
    Ok(line)
}

pub fn decode_line<T: DeserializeOwned>(line: &str) -> Result<T, DomainError> {
    if line.len() > MAX_FRAME_BYTES {
        return Err(DomainError::FrameTooLong {
            limit: MAX_FRAME_BYTES,
        });
    }
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Err(DomainError::MalformedFrame {
            reason: "empty line".into(),
        });
    }
    serde_json::from_str(trimmed).map_err(|error| DomainError::MalformedFrame {
        reason: error.to_string(),
    })
}

pub fn check_protocol_version(actual: u32) -> Result<(), DomainError> {
    if actual == PROTOCOL_VERSION {
        Ok(())
    } else {
        Err(DomainError::ProtocolVersion {
            expected: PROTOCOL_VERSION,
            actual,
        })
    }
}
```

The length check in `decode_line` runs before `serde_json::from_str`, so an oversized line is rejected without ever being parsed. Keep that order.

`serde_json::to_string` escapes newlines inside strings as `\n`, which is why a reason containing a line break still encodes to a single physical line. This property is what makes line-delimited framing safe; do not switch to `to_string_pretty`.

In `src/lib.rs`, add `mod codec;` and `pub use codec::{MAX_FRAME_BYTES, check_protocol_version, decode_line, encode_line};`.

- [x] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p feathertalk-domain --all-targets`

Expected: PASS, 43 passed, 0 failed.

- [x] **Step 5: Format, lint, and commit**

```bash
cargo fmt --all
cargo clippy -p feathertalk-domain --all-targets -- -D warnings
git add rust/crates/feathertalk-domain
git commit -m "feat: add JSON Lines frame codec with a length limit"
```

---

### Task 9: Stream reader and writer

**Files:**
- Create: `rust/crates/feathertalk-domain/src/stream.rs`
- Modify: `rust/crates/feathertalk-domain/src/lib.rs`
- Test: `rust/crates/feathertalk-domain/tests/stream_io.rs`

**Interfaces:**
- Consumes: `encode_line`, `decode_line`, `MAX_FRAME_BYTES`, `DomainError`.
- Produces: `FrameReader<R: BufRead>` with `new(R)` and `read_frame<T: DeserializeOwned>(&mut self) -> Option<Result<T, DomainError>>`, `FrameWriter<W: Write>` with `new(W)`, `write_frame<T: Serialize>(&mut self, &T) -> Result<(), DomainError>`, and `into_inner(self) -> W`.

`read_frame` returns `None` at clean end of stream, `Some(Ok(frame))` for a good line, and `Some(Err(_))` for a bad one. Blank lines are skipped rather than reported, so a worker that flushes a stray newline does not look like a protocol violation. Like `decode_line`, it performs only framing/UTF-8/serde syntax checks; a successful result still requires the direction-specific `ClientFrame::validate()` or `ServerFrame::validate()` call.

- [x] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-domain/tests/stream_io.rs`:

```rust
use std::io::Cursor;

use feathertalk_domain::{
    DomainError, FrameReader, FrameWriter, MAX_FRAME_BYTES, PROTOCOL_VERSION, RejectedFrame,
    ServerFrame,
};

fn rejected(reason: &str) -> ServerFrame {
    ServerFrame::Rejected(RejectedFrame {
        protocol_version: PROTOCOL_VERSION,
        reason: reason.to_owned(),
    })
}

#[test]
fn frames_written_then_read_back_survive_the_trip() {
    let mut writer = FrameWriter::new(Vec::new());
    writer.write_frame(&rejected("first")).unwrap();
    writer.write_frame(&rejected("second")).unwrap();
    let bytes = writer.into_inner();
    assert_eq!(bytes.iter().filter(|byte| **byte == b'\n').count(), 2);

    let mut reader = FrameReader::new(Cursor::new(bytes));
    assert_eq!(
        reader.read_frame::<ServerFrame>().unwrap().unwrap(),
        rejected("first")
    );
    assert_eq!(
        reader.read_frame::<ServerFrame>().unwrap().unwrap(),
        rejected("second")
    );
    assert!(reader.read_frame::<ServerFrame>().is_none());
}

#[test]
fn blank_lines_between_frames_are_skipped() {
    let line = feathertalk_domain::encode_line(&rejected("only")).unwrap();
    let input = format!("\n\n{line}\n   \n");
    let mut reader = FrameReader::new(Cursor::new(input));
    assert_eq!(
        reader.read_frame::<ServerFrame>().unwrap().unwrap(),
        rejected("only")
    );
    assert!(reader.read_frame::<ServerFrame>().is_none());
}
```

```rust
#[test]
fn a_final_line_without_a_newline_is_still_delivered() {
    let line = feathertalk_domain::encode_line(&rejected("unterminated")).unwrap();
    let mut reader = FrameReader::new(Cursor::new(line));
    assert_eq!(
        reader.read_frame::<ServerFrame>().unwrap().unwrap(),
        rejected("unterminated")
    );
    assert!(reader.read_frame::<ServerFrame>().is_none());
}

#[test]
fn a_bad_line_is_reported_and_the_reader_keeps_going() {
    let good = feathertalk_domain::encode_line(&rejected("after")).unwrap();
    let input = format!("not json\n{good}\n");
    let mut reader = FrameReader::new(Cursor::new(input));
    assert!(matches!(
        reader.read_frame::<ServerFrame>().unwrap(),
        Err(DomainError::MalformedFrame { .. })
    ));
    assert_eq!(
        reader.read_frame::<ServerFrame>().unwrap().unwrap(),
        rejected("after")
    );
}

#[test]
fn an_oversized_line_is_refused() {
    let input = format!("{}\n", "x".repeat(MAX_FRAME_BYTES + 1_024));
    let mut reader = FrameReader::new(Cursor::new(input));
    assert!(matches!(
        reader.read_frame::<ServerFrame>().unwrap(),
        Err(DomainError::FrameTooLong { .. })
    ));
}

#[test]
fn invalid_utf8_is_reported_as_a_malformed_frame() {
    let input: Vec<u8> = vec![0xff, 0xfe, b'\n'];
    let mut reader = FrameReader::new(Cursor::new(input));
    assert!(matches!(
        reader.read_frame::<ServerFrame>().unwrap(),
        Err(DomainError::MalformedFrame { .. })
    ));
}

#[test]
fn writing_an_oversized_frame_leaves_the_stream_untouched() {
    let mut writer = FrameWriter::new(Vec::new());
    let huge = ServerFrame::Rejected(RejectedFrame {
        protocol_version: PROTOCOL_VERSION,
        reason: "x".repeat(MAX_FRAME_BYTES),
    });
    assert!(matches!(
        writer.write_frame(&huge),
        Err(DomainError::FrameTooLong { .. })
    ));
    writer
        .write_frame(&ServerFrame::Rejected(RejectedFrame {
            protocol_version: PROTOCOL_VERSION,
            reason: "small".to_owned(),
        }))
        .unwrap();
    let bytes = writer.into_inner();
    assert_eq!(bytes.iter().filter(|byte| **byte == b'\n').count(), 1);
}
```

- [x] **Step 2: Run the test to verify it fails**

Run: `cargo test -p feathertalk-domain --test stream_io`

Expected: FAIL with `unresolved import` for `FrameReader` and `FrameWriter`.

- [x] **Step 3: Implement the reader**

Create `rust/crates/feathertalk-domain/src/stream.rs`:

```rust
use std::io::{BufRead, Write};

use serde::{Serialize, de::DeserializeOwned};

use crate::{DomainError, MAX_FRAME_BYTES, decode_line, encode_line};

pub struct FrameReader<R: BufRead> {
    inner: R,
    buffer: Vec<u8>,
}

impl<R: BufRead> FrameReader<R> {
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            buffer: Vec::with_capacity(MAX_FRAME_BYTES),
        }
    }

    pub fn read_frame<T: DeserializeOwned>(&mut self) -> Option<Result<T, DomainError>> {
        loop {
            self.buffer.clear();
            let mut content_len = 0usize;
            let mut saw_input = false;
            let mut too_long = false;
            let mut terminated = false;

            while !terminated {
                let (content_chunk_len, has_newline) = {
                    let available = match self.inner.fill_buf() {
                        Ok(available) => available,
                        Err(error) => {
                            return Some(Err(DomainError::MalformedFrame {
                                reason: error.to_string(),
                            }));
                        }
                    };
                    if available.is_empty() {
                        break;
                    }
                    saw_input = true;
                    let newline = available.iter().position(|byte| *byte == b'\n');
                    let content_chunk_len = newline.unwrap_or(available.len());
                    if !too_long {
                        let remaining = MAX_FRAME_BYTES - content_len;
                        if content_chunk_len > remaining {
                            self.buffer.extend_from_slice(&available[..remaining]);
                            too_long = true;
                        } else {
                            self.buffer
                                .extend_from_slice(&available[..content_chunk_len]);
                            content_len += content_chunk_len;
                        }
                    }
                    let consumed = content_chunk_len + usize::from(newline.is_some());
                    self.inner.consume(consumed);
                    (content_chunk_len, newline.is_some())
                };
                if has_newline {
                    terminated = true;
                } else if !too_long {
                    debug_assert!(content_len >= content_chunk_len);
                }
            }

            if !saw_input {
                return None;
            }
            if too_long {
                return Some(Err(DomainError::FrameTooLong {
                    limit: MAX_FRAME_BYTES,
                }));
            }
            let text = match std::str::from_utf8(&self.buffer) {
                Ok(text) => text,
                Err(error) => {
                    return Some(Err(DomainError::MalformedFrame {
                        reason: error.to_string(),
                    }));
                }
            };
            if text.trim().is_empty() {
                continue;
            }
            return Some(decode_line(text));
        }
    }
}
```

The reader uses `fill_buf`/`consume` in chunks. The retained frame content never exceeds
`MAX_FRAME_BYTES`; the terminating `\n` delimiter is consumed but is not counted toward that limit. Once
a line exceeds the limit, the reader retains only the bounded prefix and continues consuming/discarding
bytes through the next newline, so the following frame remains synchronized. An unterminated oversized
line is reported as `FrameTooLong` when EOF is reached. A non-oversized final line does not need a trailing
newline: EOF after any input still delivers that line, while a clean EOF with no input returns `None`.

- [x] **Step 4: Implement the writer**

Append to `src/stream.rs`:

```rust
pub struct FrameWriter<W: Write> {
    inner: W,
}

impl<W: Write> FrameWriter<W> {
    pub fn new(inner: W) -> Self {
        Self { inner }
    }

    pub fn write_frame<T: Serialize>(&mut self, value: &T) -> Result<(), DomainError> {
        let line = encode_line(value)?;
        self.inner
            .write_all(line.as_bytes())
            .and_then(|()| self.inner.write_all(b"\n"))
            .and_then(|()| self.inner.flush())
            .map_err(|error| DomainError::MalformedFrame {
                reason: error.to_string(),
            })
    }

    pub fn into_inner(self) -> W {
        self.inner
    }
}
```

`encode_line` runs before any byte is written, so a refused oversized frame leaves the stream byte-for-byte untouched — that is the property the last test in Step 1 pins down. Flushing on every frame is deliberate: the desktop needs progress events promptly, and a buffered worker would look hung.

In `src/lib.rs`, add `mod stream;` and `pub use stream::{FrameReader, FrameWriter};`.

- [x] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p feathertalk-domain --all-targets`

Expected: PASS, 50 passed, 0 failed.

- [x] **Step 6: Format, lint, and commit**

```bash
cargo fmt --all
cargo clippy -p feathertalk-domain --all-targets -- -D warnings
git add rust/crates/feathertalk-domain
git commit -m "feat: add bounded frame reader and writer"
```

---

### Task 10: Guard tests against the persisted project format

**Files:**
- Modify: `rust/crates/feathertalk-domain/Cargo.toml` (add `[dev-dependencies]`)
- Test: `rust/crates/feathertalk-domain/tests/project_compatibility.rs`

**Interfaces:**
- Consumes: `TaskId`, `TaskKind`, `TaskStatus`, `TaskStage`, and `feathertalk_project::{ProjectManifest, ModelSelection, TaskHistoryEntry, TaskHistoryStatus}`.
- Produces: no new public API. This task adds guards only.

**Do not modify any file under `rust/crates/feathertalk-project/`.** The entire point of this task is that the persisted format stays untouched while the two definitions of the five-state vocabulary are held in step by tests. The dev dependency does not propagate to consumers, so `feathertalk-app` will not pull `feathertalk-project` through `feathertalk-domain`.

- [x] **Step 1: Add the dev dependency**

Append to `rust/crates/feathertalk-domain/Cargo.toml`:

```toml
[dev-dependencies]
feathertalk-project = { path = "../feathertalk-project" }
```

- [x] **Step 2: Write the failing test**

Create `rust/crates/feathertalk-domain/tests/project_compatibility.rs`:

```rust
use feathertalk_domain::{TaskId, TaskKind, TaskStage, TaskStatus};
use feathertalk_project::{
    ModelSelection, ProjectManifest, TaskHistoryEntry, TaskHistoryStatus,
};

const SAMPLE_TASK_ID: &str = "1787900000000-0000000a";

fn manifest(entries: Vec<TaskHistoryEntry>) -> ProjectManifest {
    ProjectManifest {
        schema_version: 1,
        project_id: "demo-project".to_owned(),
        display_name: "Demo".to_owned(),
        asset_package: "assets/assets.json".to_owned(),
        default_model: ModelSelection::OriginalUnet,
        task_history: entries,
    }
}

fn entry(task_id: &str, kind: &str) -> TaskHistoryEntry {
    TaskHistoryEntry {
        task_id: task_id.to_owned(),
        kind: kind.to_owned(),
        status: TaskHistoryStatus::Running,
        updated_at: "2026-08-28T09:00:00Z".to_owned(),
    }
}

#[test]
fn the_five_state_vocabulary_has_not_drifted() {
    let domain: Vec<String> = TaskStatus::ALL
        .into_iter()
        .map(|status| serde_json::to_string(&status).unwrap())
        .collect();
    let persisted: Vec<String> = [
        TaskHistoryStatus::Queued,
        TaskHistoryStatus::Running,
        TaskHistoryStatus::Completed,
        TaskHistoryStatus::Failed,
        TaskHistoryStatus::Cancelled,
    ]
    .into_iter()
    .map(|status| serde_json::to_string(&status).unwrap())
    .collect();
    assert_eq!(domain, persisted);
    assert_eq!(domain.len(), 5);
}
```

```rust
#[test]
fn every_task_kind_slug_is_accepted_by_the_real_project_validator() {
    for (index, kind) in TaskKind::ALL.into_iter().enumerate() {
        let task_id = format!("178790000{:04}-0000000a", index);
        let manifest = manifest(vec![entry(&task_id, kind.as_slug())]);
        manifest.validate().unwrap_or_else(|error| {
            panic!("slug {:?} rejected by ProjectManifest: {error}", kind.as_slug())
        });
    }
}

#[test]
fn the_canonical_task_id_shape_is_accepted_by_the_real_project_validator() {
    let task_id = TaskId::parse(SAMPLE_TASK_ID).unwrap();
    let manifest = manifest(vec![entry(task_id.as_str(), TaskKind::Train.as_slug())]);
    manifest.validate().unwrap();
}

#[test]
fn a_stage_projection_reaches_every_persisted_status() {
    let mut reached: Vec<TaskStatus> = TaskStage::ALL_UNIT_SAMPLES
        .into_iter()
        .map(|stage| stage.status())
        .collect();
    reached.sort_by_key(|status| serde_json::to_string(status).unwrap());
    reached.dedup();
    assert_eq!(reached.len(), 5, "projection does not cover all five states");
}
```

**A note on why these guards use `ProjectManifest::validate` rather than restating rules:** `feathertalk-project` enforces the `kind` character class and the 128-byte identifier limit inside private functions. Copying those rules into this test would create a second place to update. Building a real manifest and validating it exercises the actual enforcement, so a future tightening in `feathertalk-project` surfaces here as a failure instead of silently diverging.

- [x] **Step 3: Run the test to verify it fails**

Run: `cargo test -p feathertalk-domain --test project_compatibility`

Expected: FAIL. Before Step 1 is applied the failure is `unresolved import feathertalk_project`; if Step 1 is already applied the four tests compile and must pass, in which case re-check that `Cargo.toml` truly lists `feathertalk-project` under `[dev-dependencies]` and not `[dependencies]`.

- [x] **Step 4: Verify the dependency direction**

Run: `cargo tree -p feathertalk-domain --edges normal`

Expected: the output lists only `serde`, `serde_json`, `thiserror`, and `time` with their transitive crates. `feathertalk-project` must not appear. If it does, it is in the wrong section of `Cargo.toml`.

- [x] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p feathertalk-domain --all-targets`

Expected: PASS, 54 passed, 0 failed.

- [x] **Step 6: Format, lint, and commit**

```bash
cargo fmt --all
cargo clippy -p feathertalk-domain --all-targets -- -D warnings
git add rust/crates/feathertalk-domain
git commit -m "test: guard task vocabulary against the persisted project format"
```

---

### Task 11: Golden wire format

**Files:**
- Test: `rust/crates/feathertalk-domain/tests/golden_frames.rs`

**Interfaces:**
- Consumes: every public type from Tasks 1 through 9.
- Produces: no new public API.

**Why this task exists:** a round-trip test passes even when both the serializer and the deserializer are renamed together, because it only ever compares the crate against itself. The desktop and the worker are separate processes that can be built from different commits during development, so a silent rename is a real incompatibility. Golden lines are literal text, so they fail the moment the wire format moves.

- [x] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-domain/tests/golden_frames.rs`:

```rust
use std::path::PathBuf;

use feathertalk_domain::{
    CancelFrame, ClientFrame, ErrorCode, Event, Metrics, PROTOCOL_VERSION, ProbeMediaParams,
    Progress, Recovery, Request, ServerFrame, ShutdownFrame, StartFrame, TaskError, TaskId,
    TaskStage, decode_line, encode_line,
};

const START_PROBE: &str = r#"{"frame":"start","data":{"protocol_version":1,"task_id":"1787900000000-0000000a","request":{"command":"probe_media","params":{"input":"a.mov"}}}}"#;

const CANCEL: &str = r#"{"frame":"cancel","data":{"protocol_version":1,"task_id":"1787900000000-0000000a"}}"#;

const SHUTDOWN: &str = r#"{"frame":"shutdown","data":{"protocol_version":1}}"#;

fn task_id() -> TaskId {
    TaskId::parse("1787900000000-0000000a").unwrap()
}

#[test]
fn client_frames_match_their_golden_lines_byte_for_byte() {
    let start = ClientFrame::Start(StartFrame {
        protocol_version: PROTOCOL_VERSION,
        task_id: task_id(),
        request: Request::ProbeMedia(ProbeMediaParams {
            input: PathBuf::from("a.mov"),
        }),
    });
    assert_eq!(encode_line(&start).unwrap(), START_PROBE);

    let cancel = ClientFrame::Cancel(CancelFrame {
        protocol_version: PROTOCOL_VERSION,
        task_id: task_id(),
    });
    assert_eq!(encode_line(&cancel).unwrap(), CANCEL);

    let shutdown = ClientFrame::Shutdown(ShutdownFrame {
        protocol_version: PROTOCOL_VERSION,
    });
    assert_eq!(encode_line(&shutdown).unwrap(), SHUTDOWN);
}

#[test]
fn golden_client_lines_still_decode() {
    for line in [START_PROBE, CANCEL, SHUTDOWN] {
        decode_line::<ClientFrame>(line).unwrap_or_else(|error| panic!("{line}: {error}"));
    }
}
```

- [x] **Step 2: Add the server-side golden lines**

Append to the same file:

```rust
const TRAINING_EVENT: &str = r#"{"frame":"event","data":{"protocol_version":1,"task_id":"1787900000000-0000000a","emitted_at":"2026-08-28T09:00:00Z","stage":{"stage":"training","data":{"epoch":3,"step":1200,"loss":0.0425}},"progress":{"completed":1200,"total":4000},"metrics":{"samples_per_second":12.5,"eta_seconds":90.0,"vram_bytes":3221225472},"error":null}}"#;

const FAILED_EVENT: &str = r#"{"frame":"event","data":{"protocol_version":1,"task_id":"1787900000000-0000000a","emitted_at":"2026-08-28T09:00:00Z","stage":{"stage":"failed","data":{"code":"DISK_SPACE_LOW","message":"磁盘空间不足"}},"progress":null,"metrics":{"samples_per_second":null,"eta_seconds":null,"vram_bytes":null},"error":{"code":"DISK_SPACE_LOW","summary":"磁盘空间不足","detail":"needed 4 GiB","stage":{"stage":"exporting"},"recovery":"free_disk_space"}}}"#;

#[test]
fn a_training_event_matches_its_golden_line() {
    let mut event = Event::new(
        task_id(),
        "2026-08-28T09:00:00Z",
        TaskStage::Training {
            epoch: 3,
            step: 1200,
            loss: 0.0425,
        },
    );
    event.progress = Some(Progress {
        completed: 1200,
        total: Some(4000),
    });
    event.metrics = Metrics {
        samples_per_second: Some(12.5),
        eta_seconds: Some(90.0),
        vram_bytes: Some(3_221_225_472),
    };
    event.validate().unwrap();
    assert_eq!(encode_line(&ServerFrame::Event(event)).unwrap(), TRAINING_EVENT);
}

#[test]
fn a_failed_event_carries_summary_detail_stage_and_recovery() {
    let mut event = Event::new(
        task_id(),
        "2026-08-28T09:00:00Z",
        TaskStage::Failed {
            code: ErrorCode::DiskSpaceLow,
            message: "磁盘空间不足".to_owned(),
        },
    );
    event.error = Some(TaskError::new(
        ErrorCode::DiskSpaceLow,
        "磁盘空间不足",
        "needed 4 GiB",
        TaskStage::Exporting,
    ));
    event.validate().unwrap();
    assert_eq!(
        event.error.as_ref().unwrap().recovery,
        Recovery::FreeDiskSpace
    );
    assert_eq!(encode_line(&ServerFrame::Event(event)).unwrap(), FAILED_EVENT);
}

#[test]
fn golden_server_lines_still_decode() {
    for line in [TRAINING_EVENT, FAILED_EVENT] {
        let frame = decode_line::<ServerFrame>(line).unwrap_or_else(|error| panic!("{line}: {error}"));
        let ServerFrame::Event(event) = frame else {
            panic!("expected an event frame");
        };
        event.validate().unwrap();
    }
}
```

- [x] **Step 3: Run the tests**

Run: `cargo test -p feathertalk-domain --test golden_frames`

Expected: PASS, 5 passed, 0 failed.

If a golden comparison fails on field order, do not reorder the golden string to match the code. serde emits fields in declaration order, so the fix is to confirm the struct field order matches the spec's declared order and only then adjust. Field order is part of this contract precisely so it cannot drift unnoticed.

If a golden comparison fails on float formatting — for example `90.0` serialized as `90` — adjust the golden string to whatever `serde_json` actually produces on this toolchain and leave a one-line comment recording that the value is a `f64`. Do not change the field types to make the text prettier.

- [x] **Step 4: Format, lint, and commit**

```bash
cargo fmt --all
cargo clippy -p feathertalk-domain --all-targets -- -D warnings
git add rust/crates/feathertalk-domain
git commit -m "test: pin the wire format with golden frame lines"
```

---

### Task 12: Amend the master design doc and verify the slice

**Files:**
- Modify: `docs/superpowers/specs/2026-08-17-rust-desktop-migration-design.md` (§4.3 crate names, §11 stage list)
- Modify: `docs/superpowers/plans/2026-08-28-worker-protocol-task-domain.md` (tick every checkbox)

**Interfaces:**
- Consumes: everything above.
- Produces: nothing new. This task closes the slice.

- [x] **Step 1: Add the Importing stage to §11**

In `docs/superpowers/specs/2026-08-17-rust-desktop-migration-design.md`, inside the任务事件 code block in §11, insert `Importing` on its own line immediately before `Exporting`:

```text
ExtractingFeatures
Training { epoch, step, loss }
Importing
Exporting
Rendering { frame, total }
```

- [x] **Step 2: Reconcile the crate names in §4.3**

In the same file, in the §4.3 directory tree, rename the crate entries to the prefixed form actually used in the workspace, and add the new crate. Replace the `crates/` block so it reads:

```text
  crates/
    feathertalk-app/          GPUI 桌面端
    feathertalk-worker/       后台任务进程和 RPC 服务
    feathertalk-domain/       项目、任务、模型、错误、进度类型
    feathertalk-media/        FFmpeg、WAV、视频帧和图像读写
    feathertalk-preprocess/   抽帧、人脸检测、关键点和素材包验证
    feathertalk-audio/        FeatherHuBERT、波形处理和特征窗口
    feathertalk-models/       Burn 模型定义
    feathertalk-training/     数据集、损失、优化器和 checkpoint
    feathertalk-inference/    UNet 推理、图像贴回和视频合成
    feathertalk-weights/      PyTorch 权重导入和 safetensors
    feathertalk-export/       部署包和 ONNX opset 17 导出
    feathertalk-cli/          与 worker 能力一致的命令行入口
```

Leave the sentence after the tree unchanged; it already says every crate exchanges data through `domain` types, which is what this slice delivers.

- [x] **Step 3: Run the full workspace verification**

Run each command from `E:/workspace/github/FeatherTalk/rust` and record the exit code. All five must be 0.

```bash
cargo test --workspace --all-targets
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cd .. && git diff --check
```

The workspace suite takes roughly 30 minutes on a warm target directory and includes long CPU parity tests. Do not shorten it with `-p feathertalk-domain`; the point of this step is confirming the new crate did not disturb the existing 133 test binaries. Expect the 13 pre-existing ignored tests — six subprocess helpers, six gated on a certified WGPU adapter, one gated on a licensed VGG19 package — to stay ignored.

- [x] **Step 4: Tick every checkbox in this plan**

Go back through Tasks 1 through 12 and change each `- [ ]` to `- [x]`. The other 26 plans in `docs/superpowers/plans/` were left unticked, which is why their checkboxes cannot be used to read project progress. Do not repeat that.

- [x] **Step 5: Commit the slice close-out**

```bash
git add docs/superpowers/specs/2026-08-17-rust-desktop-migration-design.md
git add docs/superpowers/plans/2026-08-28-worker-protocol-task-domain.md
git commit -m "docs: record worker protocol slice completion"
```

- [x] **Step 6: Finish the branch**

Use the `superpowers:finishing-a-development-branch` skill. Base branch is `main`. Do not merge until Step 3 shows five exit codes of 0 on this branch.

One local hazard worth knowing before you get there: `git worktree remove` on a worktree whose `rust/target` is populated can take longer than a two-minute command timeout, and an interrupted removal leaves the worktree marked `prunable` with files still on disk. If that happens, finish with `git worktree remove --force` rather than a raw recursive delete, then `git worktree prune`.

---

## Definition Of Done

- `feathertalk-domain` exists as a workspace member with exactly four production dependencies.
- No file under `rust/crates/feathertalk-project/` was modified.
- `cargo tree -p feathertalk-domain --edges normal` does not list `feathertalk-project`.
- All tests in the 11 checked-in test files pass; the suite count is determined by the checked-in tests.
- The five workspace verification commands each exit 0 on the branch that will be merged.
- §11 of the master design lists `Importing`; §4.3 lists the `feathertalk-*` crate names.
- Every checkbox in this plan is ticked.
