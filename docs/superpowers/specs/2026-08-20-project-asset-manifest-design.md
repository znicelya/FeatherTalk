# FeatherTalk Project and Asset Manifest Design

Date: 2026-08-20
Status: Approved for implementation planning
Parent specification: `docs/superpowers/specs/2026-08-17-rust-desktop-migration-design.md`

## 1. Purpose

This specification defines the first independently deliverable part of migration milestone two: versioned project and asset-package manifests, strict validation, safe project-relative paths, and atomic JSON persistence.

The deliverable establishes the data contract used later by media preprocessing, face analysis, FeatherHuBERT feature extraction, worker task recovery, training, inference, and the desktop application. It does not execute FFmpeg, SCRFD, PFLD, Burn models, or GPUI code.

## 2. Crate Boundary

Create a new workspace crate:

```text
rust/crates/feathertalk-project/
```

The crate owns:

- `project.json` and `assets/assets.json` schema types.
- Schema-version validation.
- Project-relative path validation.
- Asset package lifecycle validation.
- Bounded JSON reads.
- Atomic JSON writes.
- Project-directory and locked-package validation.

The crate must not depend on Burn, WGPU, GPUI, FFmpeg, model crates, or parity fixtures. Later crates consume these types without reimplementing the persistence rules.

The implementation dependencies are limited to `serde`, `serde_json`, `thiserror`, and `time` for RFC 3339 parsing. A Windows-only `windows-sys` dependency is permitted for atomic replacement. Temporary names use the process ID plus an atomic counter and do not require a random-number dependency.

## 3. Canonical Project Layout

The canonical paths are:

```text
project/
  project.json
  source/
    input.mp4
  assets/
    assets.json
    video_25fps.mp4
    audio_16k_mono.wav
    frames/
    landmarks/
    features/
      feather_hubert.f32
  models/
  outputs/
```

Manifest paths are stored with `/` separators and are relative to the project root. Absolute paths, Windows drive prefixes, backslashes, empty path components, `.` components, and `..` components are rejected.

## 4. Project Manifest

`ProjectManifest` maps to `project.json` and rejects unknown JSON fields.

```rust
pub struct ProjectManifest {
    pub schema_version: u32,
    pub project_id: String,
    pub display_name: String,
    pub asset_package: String,
    pub default_model: ModelSelection,
    pub task_history: Vec<TaskHistoryEntry>,
}

pub enum ModelSelection {
    OriginalUnet,
    MobileOneUnet,
}

pub struct TaskHistoryEntry {
    pub task_id: String,
    pub kind: String,
    pub status: TaskHistoryStatus,
    pub updated_at: String,
}

pub enum TaskHistoryStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}
```

JSON enum values use `snake_case`.

Validation requirements:

- `schema_version` must equal `1`.
- `project_id` must contain 1 to 128 ASCII characters from `A-Z`, `a-z`, `0-9`, `.`, `_`, and `-`.
- `display_name` must equal its trimmed form and contain 1 to 256 Unicode scalar values.
- `asset_package` must be a valid project-relative manifest path and must resolve to `assets/assets.json` for schema version 1.
- `default_model` must be one of the declared enum values.
- `task_history` may contain at most 10,000 entries.
- Each `task_id` follows the same identifier rule as `project_id`.
- `kind` must contain 1 to 64 ASCII lowercase letters, digits, `_`, or `-`.
- `updated_at` must be RFC 3339 with an explicit UTC `Z` or numeric offset.
- Duplicate task IDs are rejected.

## 5. Asset Manifest

`AssetManifest` maps to `assets/assets.json` and rejects unknown JSON fields.

```rust
pub struct AssetManifest {
    pub schema_version: u32,
    pub state: AssetPackageState,
    pub video_fps: u32,
    pub audio_sample_rate: u32,
    pub audio_channels: u16,
    pub frame_count: u64,
    pub frame_width: u32,
    pub frame_height: u32,
    pub feature_type: FeatureType,
    pub feature_shape: [u64; 3],
    pub landmark_model_sha256: String,
    pub feature_model_sha256: String,
}

pub enum AssetPackageState {
    Preparing,
    Locked,
}

pub enum FeatureType {
    FeatherHubert,
}
```

JSON enum values use `snake_case`.

General validation limits:

- `schema_version` must equal `1`.
- `video_fps` must be `25` once locked. A preparing manifest may use `0` or `25`.
- `audio_sample_rate` must be `16000` once locked. A preparing manifest may use `0` or `16000`.
- `audio_channels` must be `1` once locked. A preparing manifest may use `0` or `1`.
- `frame_count` must not exceed `100_000_000`.
- `frame_width` and `frame_height` must not exceed `32_768`.
- For `feather_hubert`, `feature_shape[1]` must be `2` and `feature_shape[2]` must be `1024` when the shape is populated.
- A SHA-256 value is either empty when permitted by lifecycle state or exactly 64 lowercase hexadecimal characters.

### 5.1 Preparing State

`Preparing` represents an incomplete, recoverable preprocessing stage.

- Zero media metadata is permitted.
- `feature_shape` must be either `[0, 0, 0]` or `[tokens, 2, 1024]`.
- When both `tokens` and `frame_count` are non-zero, they must be equal.
- Model hashes may be empty.
- Atomic updates may replace the existing preparing manifest.
- A preparing manifest is not considered a valid asset package for training or inference.

### 5.2 Locked State

`Locked` represents an immutable, complete asset package.

- `video_fps == 25`.
- `audio_sample_rate == 16000`.
- `audio_channels == 1`.
- `frame_count`, `frame_width`, and `frame_height` are non-zero.
- `feature_shape == [frame_count, 2, 1024]`.
- Both model SHA-256 values are valid and non-empty.
- The locked manifest cannot be replaced through the ordinary asset-manifest write API.
- Reprocessing creates a new asset-package version in a later milestone instead of mutating the locked package.

## 6. Public API

The crate exposes:

```rust
pub fn read_project_manifest(path: &Path) -> Result<ProjectManifest, ProjectError>;

pub fn read_asset_manifest(path: &Path) -> Result<AssetManifest, ProjectError>;

pub fn write_project_manifest_atomic(
    path: &Path,
    manifest: &ProjectManifest,
) -> Result<(), ProjectError>;

pub fn write_asset_manifest_atomic(
    path: &Path,
    manifest: &AssetManifest,
) -> Result<(), ProjectError>;

pub fn lock_asset_package(
    project_root: &Path,
    manifest: AssetManifest,
) -> Result<AssetPackage, ProjectError>;

pub fn validate_project_dir(
    project_root: &Path,
) -> Result<ValidatedProject, ProjectError>;
```

`AssetPackage` contains the validated project root and locked asset manifest. `ValidatedProject` contains the validated project manifest and locked asset package. These types expose read-only accessors rather than public mutable fields.

## 7. Locked-Package Filesystem Contract

Before committing a locked manifest, `lock_asset_package` verifies that the following entries exist below the supplied project root and do not escape through symbolic links:

```text
assets/video_25fps.mp4
assets/audio_16k_mono.wav
assets/frames/
assets/landmarks/
assets/features/feather_hubert.f32
```

Files must be regular, non-empty files. Directories must be real directories. This milestone does not parse media containers, JPEG data, landmark rows, feature headers, or individual frame counts. Those checks belong to later milestone-two subprojects.

`validate_project_dir` additionally verifies:

- The root is a directory and not a symbolic link.
- `project.json` exists and is a regular file.
- `asset_package` resolves exactly to `assets/assets.json`.
- The referenced asset manifest is locked.
- The locked-package filesystem contract passes.

## 8. Bounded Reads and Strict JSON

- Each manifest is limited to 1 MiB before JSON parsing.
- Reads stop after `1 MiB + 1 byte`; oversized input returns `ProjectError::ManifestTooLarge`.
- UTF-8 and JSON failures are returned as structured errors.
- Both manifest structs use `serde(deny_unknown_fields)`.
- Unsupported schema versions are reported distinctly from malformed JSON.

## 9. Atomic Write Protocol

Both manifest writers use the same internal protocol:

```text
validate manifest
create parent directory when absent
reject symbolic-link parent or destination
serialize pretty JSON with one trailing newline
create a unique sibling temporary file with create_new
write_all
flush
sync_all temporary file
rename temporary file to destination
sync parent directory where the platform supports directory syncing
```

The temporary file is always in the destination directory so rename remains filesystem-local. The implementation retries temporary-name collisions with a bounded attempt count.

Failure requirements:

- The existing destination is not truncated before the replacement is ready.
- A validation, serialization, write, flush, sync, or rename failure leaves the old destination intact.
- Temporary cleanup is best-effort and never replaces the primary error.
- No panic text is used as a public user-facing error.
- `write_asset_manifest_atomic` rejects replacement when the existing destination contains a valid locked manifest.
- `lock_asset_package` writes `assets/assets.json` last, after validating every required artifact.

On Unix, replacement uses same-directory `rename`. On Windows, replacement uses `ReplaceFileW` when the destination exists and `MoveFileExW` with write-through semantics for the first install. The platform-specific calls are isolated in the persistence module. If the target filesystem rejects the atomic operation, the function returns an explicit unsupported-operation error rather than silently using in-place truncation.

## 10. Errors

`ProjectError` is a `thiserror` enum with stable categories for:

- I/O operation and path.
- Manifest too large.
- Invalid UTF-8 or JSON.
- Unsupported schema version.
- Field validation failure.
- Unsafe relative path.
- Symbolic link encountered.
- Missing or wrong filesystem entry type.
- Empty required artifact.
- Locked asset package mutation.
- Atomic replacement unsupported.

Errors retain technical source details for logs while presenting a stable category and affected field/path to callers.

## 11. Tests

Unit and integration tests cover:

1. Project and asset manifest JSON round trips.
2. Unknown JSON field rejection.
3. Unsupported schema versions.
4. Identifier, display name, task history, timestamp, and duplicate-task validation.
5. Absolute paths, drive prefixes, backslashes, `.`, `..`, and empty path components.
6. Preparing-state partial metadata.
7. Locked-state required metadata, shape, and hashes.
8. Oversized manifest rejection before unbounded allocation.
9. New atomic writes and replacement of preparing manifests.
10. Existing locked asset manifest mutation rejection.
11. Failed writes leaving the old destination unchanged.
12. Temporary-file cleanup after failure.
13. Project root, manifest, artifact, and directory symbolic-link rejection.
14. Missing, empty, and wrong-type locked artifacts.
15. Successful `lock_asset_package` and `validate_project_dir` round trips.

All tests use temporary directories and do not require Python, FFmpeg, GPU hardware, or network access.

## 12. Acceptance Criteria

- `feathertalk-project` is a workspace member and has no Burn, WGPU, GPUI, FFmpeg, model, or parity dependency.
- Strict manifest validation matches this specification.
- Manifest paths cannot escape the project root on Windows, macOS, or Linux.
- Interrupted or failed writes do not corrupt an existing manifest.
- A locked package cannot be committed without every required artifact.
- A locked package cannot be mutated through the ordinary manifest writer.
- `cargo fmt --check` passes.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` passes.
- `cargo test --workspace --all-features` passes.
- `git diff --check` passes.

## 13. Excluded from This Subproject

- FFmpeg discovery, bundling, media probing, transcoding, audio extraction, and frame extraction.
- SCRFD and PFLD model ports or inference.
- Frame-level landmark, blur, face-count, and anomaly validation.
- FeatherHuBERT long-audio chunking and feature-file encoding.
- Worker RPC, task scheduling, cancellation, and recovery UI.
- Project creation UI and GPUI pages.
- Versioned multiple asset-package directories and garbage collection.
- Parsing the future versioned `.f32` feature header.
