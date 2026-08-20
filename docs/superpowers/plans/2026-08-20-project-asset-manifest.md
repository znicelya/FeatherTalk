# Project and Asset Manifest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independent `feathertalk-project` crate that defines, validates, atomically persists, and locks FeatherTalk project and asset manifests without Burn, WGPU, GPUI, FFmpeg, or model dependencies.

**Architecture:** Add a small workspace crate with strict Serde schema types, lifecycle-aware validation, bounded JSON reads, and platform-specific atomic replacement isolated behind one persistence module. Filesystem validation operates from a project root and returns read-only validated wrapper types; media/model content parsing remains outside this subproject.

**Tech Stack:** Rust 1.92 edition 2024, `serde`, `serde_json`, `thiserror`, `time` 0.3.55, `tempfile` for tests, and Windows-only `windows-sys` APIs for `ReplaceFileW`/`MoveFileExW`.

## Global Constraints

- Schema version is exactly `1`.
- Manifest JSON rejects unknown fields with `serde(deny_unknown_fields)`.
- Manifest reads are bounded to `1 MiB`; oversized input fails before unbounded parsing.
- Project-relative paths use `/` separators and reject absolute paths, drive prefixes, backslashes, `.`, `..`, and empty components.
- Locked assets require `25 FPS`, mono `16 kHz` audio, `[frame_count, 2, 1024]` FeatherHuBERT features, non-empty regular artifacts, and two 64-character lowercase SHA-256 values.
- A locked asset manifest cannot be replaced by the ordinary asset writer.
- Existing manifests are never truncated before replacement is ready.
- No task may add Burn, WGPU, GPUI, FFmpeg, model, or parity dependencies to `feathertalk-project`.
- Every task follows red-green-refactor TDD and ends with a focused verification command.

---

### Task 1: Bootstrap the crate and schema types

**Files:**
- Modify: `rust/Cargo.toml`
- Modify: `rust/Cargo.lock` (Cargo-generated)
- Create: `rust/crates/feathertalk-project/Cargo.toml`
- Create: `rust/crates/feathertalk-project/src/lib.rs`
- Create: `rust/crates/feathertalk-project/src/model.rs`
- Create: `rust/crates/feathertalk-project/src/error.rs`
- Create: `rust/crates/feathertalk-project/tests/support/mod.rs`
- Test: `rust/crates/feathertalk-project/tests/manifest_round_trip.rs`

**Interfaces:**
- Produces `ProjectManifest`, `AssetManifest`, `ModelSelection`, `TaskHistoryEntry`, `TaskHistoryStatus`, `AssetPackageState`, `FeatureType`, and `ProjectError`.
- Consumes only `serde`, `serde_json`, `thiserror`, and `time`.

- [ ] **Step 1: Add the failing schema round-trip tests**

Create `tests/manifest_round_trip.rs`:

```rust
use feathertalk_project::{
    AssetManifest, AssetPackageState, FeatureType, ModelSelection, ProjectManifest,
    TaskHistoryEntry, TaskHistoryStatus,
};

#[test]
fn project_manifest_round_trips_with_snake_case_enums() {
    let manifest = ProjectManifest {
        schema_version: 1,
        project_id: "demo_01".to_owned(),
        display_name: "Demo".to_owned(),
        asset_package: "assets/assets.json".to_owned(),
        default_model: ModelSelection::OriginalUnet,
        task_history: vec![TaskHistoryEntry {
            task_id: "task-1".to_owned(),
            kind: "preprocess".to_owned(),
            status: TaskHistoryStatus::Completed,
            updated_at: "2026-08-20T10:00:00Z".to_owned(),
        }],
    };
    let json = serde_json::to_string(&manifest).unwrap();
    assert!(json.contains("original_unet"));
    assert!(json.contains("completed"));
    let decoded: ProjectManifest = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, manifest);
}

#[test]
fn asset_manifest_round_trips_lifecycle_and_feature_type() {
    let manifest = AssetManifest {
        schema_version: 1,
        state: AssetPackageState::Preparing,
        video_fps: 0,
        audio_sample_rate: 0,
        audio_channels: 0,
        frame_count: 0,
        frame_width: 0,
        frame_height: 0,
        feature_type: FeatureType::FeatherHubert,
        feature_shape: [0, 0, 0],
        landmark_model_sha256: String::new(),
        feature_model_sha256: String::new(),
    };
    let json = serde_json::to_string(&manifest).unwrap();
    assert!(json.contains("preparing"));
    assert!(json.contains("feather_hubert"));
    assert_eq!(serde_json::from_str::<AssetManifest>(&json).unwrap(), manifest);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p feathertalk-project --test manifest_round_trip`

Expected: FAIL because the workspace member, schema types, and `ProjectError` do not exist.

- [ ] **Step 3: Add the workspace member and minimal dependencies**

Add `"crates/feathertalk-project"` to `rust/Cargo.toml`. Create the crate manifest with:

```toml
[package]
name = "feathertalk-project"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[dependencies]
serde.workspace = true
serde_json.workspace = true
thiserror.workspace = true
time = { version = "0.3.55", features = ["parsing", "formatting", "macros"] }

[dev-dependencies]
tempfile.workspace = true

[target.'cfg(windows)'.dependencies]
windows-sys = { version = "0.61.2", features = ["Win32_Foundation", "Win32_Storage_FileSystem"] }
```

Do not add any Burn, WGPU, GPUI, FFmpeg, model, or parity dependency.

- [ ] **Step 4: Implement the schema types and error skeleton**

Use `#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]` and `#[serde(deny_unknown_fields)]` on all externally deserialized structs. Use `#[serde(rename_all = "snake_case")]` on all enums. Export the types from `src/lib.rs`; define `ProjectError` with the stable categories required by the spec, even if later tasks add variants.

- [ ] **Step 5: Run the focused test to verify it passes**

Run: `cargo test -p feathertalk-project --test manifest_round_trip`

Expected: 2 tests pass.

- [ ] **Step 6: Commit the bootstrap**

```powershell
git add rust/Cargo.toml rust/Cargo.lock rust/crates/feathertalk-project
git commit -m "feat: add project manifest crate"
```

### Task 2: Implement strict manifest validation

**Files:**
- Modify: `rust/crates/feathertalk-project/src/model.rs`
- Modify: `rust/crates/feathertalk-project/src/error.rs`
- Modify: `rust/crates/feathertalk-project/src/lib.rs`
- Create: `rust/crates/feathertalk-project/tests/manifest_validation.rs`
- Modify: `rust/crates/feathertalk-project/tests/support/mod.rs`

**Interfaces:**
- Produces `ProjectManifest::validate`, `AssetManifest::validate_preparing`, `AssetManifest::validate_locked`, and a shared `validate_relative_manifest_path` helper.
- Consumes the schema types from Task 1.

- [ ] **Step 1: Write failing validation tests**

Cover these concrete cases in `manifest_validation.rs`:

```rust
fn preparing_manifest() -> AssetManifest { AssetManifest { schema_version: 1, state: AssetPackageState::Preparing, video_fps: 0, audio_sample_rate: 0, audio_channels: 0, frame_count: 0, frame_width: 0, frame_height: 0, feature_type: FeatureType::FeatherHubert, feature_shape: [0, 0, 0], landmark_model_sha256: String::new(), feature_model_sha256: String::new() } }

fn locked_manifest() -> AssetManifest { AssetManifest { schema_version: 1, state: AssetPackageState::Locked, video_fps: 25, audio_sample_rate: 16_000, audio_channels: 1, frame_count: 12, frame_width: 160, frame_height: 160, feature_type: FeatureType::FeatherHubert, feature_shape: [12, 2, 1024], landmark_model_sha256: "a".repeat(64), feature_model_sha256: "b".repeat(64) } }

#[test] fn project_rejects_unknown_fields() { let json = r#"{"schema_version":1,"project_id":"demo","display_name":"Demo","asset_package":"assets/assets.json","default_model":"original_unet","task_history":[],"extra":true}"#; assert!(serde_json::from_str::<ProjectManifest>(json).is_err()); }
#[test] fn project_rejects_bad_identifier_and_duplicate_task_ids() { let mut project = valid_project(); project.project_id = "bad/id".into(); assert!(project.validate().is_err()); project.project_id = "demo".into(); project.task_history.push(project.task_history[0].clone()); assert!(project.validate().is_err()); }
#[test] fn project_rejects_non_rfc3339_timestamp_and_unsafe_asset_path() { let mut project = valid_project(); project.task_history[0].updated_at = "tomorrow".into(); assert!(project.validate().is_err()); project.task_history[0].updated_at = "2026-08-20T10:00:00Z".into(); project.asset_package = "../assets.json".into(); assert!(project.validate().is_err()); }
#[test] fn preparing_manifest_accepts_empty_progress_metadata() { assert!(preparing_manifest().validate_preparing().is_ok()); }
#[test] fn preparing_manifest_rejects_partial_feature_shape() { let mut manifest = preparing_manifest(); manifest.feature_shape = [12, 0, 1024]; assert!(manifest.validate_preparing().is_err()); }
#[test] fn locked_manifest_requires_exact_media_and_feature_contract() { let mut manifest = locked_manifest(); manifest.video_fps = 24; assert!(manifest.validate_locked().is_err()); }
#[test] fn locked_manifest_rejects_uppercase_or_short_sha256() { let mut manifest = locked_manifest(); manifest.landmark_model_sha256 = "A".repeat(64); assert!(manifest.validate_locked().is_err()); manifest.landmark_model_sha256 = "a".repeat(63); assert!(manifest.validate_locked().is_err()); }
#[test] fn locked_manifest_rejects_frame_count_shape_mismatch() { let mut manifest = locked_manifest(); manifest.frame_count = 11; assert!(manifest.validate_locked().is_err()); }
```

Put `preparing_manifest()`, `locked_manifest()`, and `valid_project()` in `tests/support/mod.rs`; each integration test imports them with `#[path = "support/mod.rs"] mod support; use support::*;`.

- [ ] **Step 2: Run the validation tests to verify they fail**

Run: `cargo test -p feathertalk-project --test manifest_validation`

Expected: FAIL because validation methods and strict schema behavior are not implemented.

- [ ] **Step 3: Implement shared validators**

Implement bounded validators with explicit constants:

```rust
const CURRENT_SCHEMA_VERSION: u32 = 1;
const MAX_IDENTIFIER_BYTES: usize = 128;
const MAX_DISPLAY_NAME_CHARS: usize = 256;
const MAX_TASK_HISTORY: usize = 10_000;
const MAX_FRAME_COUNT: u64 = 100_000_000;
const MAX_FRAME_DIMENSION: u32 = 32_768;
```

Use `time::OffsetDateTime::parse(value, &Rfc3339)` for timestamps. Reject a display name whose trimmed value differs from the stored value. Validate paths by inspecting slash-separated components before constructing a platform `Path`.

- [ ] **Step 4: Implement lifecycle-specific asset validation**

`validate_preparing` accepts only `[0,0,0]` or `[tokens,2,1024]` with non-zero `tokens`; zero media metadata and empty hashes are allowed. `validate_locked` requires `25`, `16000`, `1`, non-zero dimensions/count, shape `[frame_count,2,1024]`, and two lowercase 64-character hashes.

- [ ] **Step 5: Run validation tests to verify they pass**

Run: `cargo test -p feathertalk-project --test manifest_validation`

Expected: all validation tests pass.

- [ ] **Step 6: Commit validation**

```powershell
git add rust/crates/feathertalk-project/src rust/crates/feathertalk-project/tests/manifest_validation.rs
git commit -m "feat: validate project and asset manifests"
```

### Task 3: Add bounded reads and atomic manifest persistence

**Files:**
- Create: `rust/crates/feathertalk-project/src/persistence.rs`
- Modify: `rust/crates/feathertalk-project/src/lib.rs`
- Modify: `rust/crates/feathertalk-project/src/error.rs`
- Create: `rust/crates/feathertalk-project/tests/manifest_persistence.rs`
- Modify: `rust/crates/feathertalk-project/tests/support/mod.rs`

**Interfaces:**
- Produces `read_project_manifest`, `read_asset_manifest`, `write_project_manifest_atomic`, and `write_asset_manifest_atomic`.
- Consumes manifest validators from Task 2 and returns `ProjectError` without panics.

- [ ] **Step 1: Write failing persistence tests**

Create tests for:

```rust
#[test] fn reads_and_writes_preparing_manifest_with_one_newline() { let dir = tempfile::tempdir().unwrap(); let path = dir.path().join("assets.json"); let manifest = preparing_manifest(); write_asset_manifest_atomic(&path, &manifest).unwrap(); assert!(std::fs::read(&path).unwrap().ends_with(b"\n")); assert_eq!(read_asset_manifest(&path).unwrap(), manifest); }
#[test] fn rejects_manifest_larger_than_one_mib() { let dir = tempfile::tempdir().unwrap(); let path = dir.path().join("project.json"); std::fs::write(&path, vec![b' '; 1_048_577]).unwrap(); assert!(matches!(read_project_manifest(&path), Err(ProjectError::ManifestTooLarge { .. }))); }
#[test] fn replaces_existing_preparing_manifest_without_truncating_on_success() { let dir = tempfile::tempdir().unwrap(); let path = dir.path().join("assets.json"); let mut first = preparing_manifest(); write_asset_manifest_atomic(&path, &first).unwrap(); first.video_fps = 25; write_asset_manifest_atomic(&path, &first).unwrap(); assert_eq!(read_asset_manifest(&path).unwrap().video_fps, 25); }
#[test] fn failed_validation_leaves_existing_manifest_unchanged() { let dir = tempfile::tempdir().unwrap(); let path = dir.path().join("assets.json"); let first = preparing_manifest(); write_asset_manifest_atomic(&path, &first).unwrap(); let mut invalid = first.clone(); invalid.feature_shape = [1, 0, 1024]; assert!(write_asset_manifest_atomic(&path, &invalid).is_err()); assert_eq!(read_asset_manifest(&path).unwrap(), first); }
#[test] fn rejects_symlink_destination_or_parent() { let dir = tempfile::tempdir().unwrap(); let target = dir.path().join("target"); let link = dir.path().join("link"); std::fs::create_dir(&target).unwrap(); if support::create_dir_symlink(&target, &link).is_ok() { assert!(write_asset_manifest_atomic(&link.join("assets.json"), &preparing_manifest()).is_err()); } }
#[test] fn asset_writer_rejects_existing_locked_manifest() { let dir = tempfile::tempdir().unwrap(); let path = dir.path().join("assets.json"); let locked = locked_manifest(); write_asset_manifest_atomic(&path, &locked).unwrap(); assert!(matches!(write_asset_manifest_atomic(&path, &preparing_manifest()), Err(ProjectError::LockedAssetMutation { .. }))); }
```

- [ ] **Step 2: Run persistence tests to verify they fail**

Run: `cargo test -p feathertalk-project --test manifest_persistence`

Expected: FAIL because persistence functions do not exist.

- [ ] **Step 3: Implement bounded JSON reads**

Read with `File::open`, `take(MAX_MANIFEST_BYTES + 1)`, and `read_to_end`. Return `ManifestTooLarge` before `serde_json::from_slice` when the byte count exceeds `1 MiB`. Map UTF-8/JSON failures to structured `ProjectError` variants.

- [ ] **Step 4: Implement deterministic JSON serialization**

Serialize with `serde_json::to_vec_pretty`, append exactly one `b'\n'`, and validate the manifest before serializing. Use a 1 MiB serialized-size guard as well as the read guard.

- [ ] **Step 5: Implement same-directory temporary-file creation**

Create a bounded sequence of sibling names using process ID and an atomic counter. Open with `create_new(true)`, write, flush, and `sync_all`. Ensure the parent exists and reject a symlink destination or parent component before creating the temp file.

- [ ] **Step 6: Implement platform replacement**

Create `src/platform.rs` with:

```rust
pub fn replace_file_atomic(temp: &Path, destination: &Path) -> Result<(), ProjectError>;
pub fn sync_parent_directory(parent: &Path) -> Result<(), ProjectError>;
```

On Unix, call `std::fs::rename`. On Windows, use `windows-sys` `ReplaceFileW` when the destination exists and `MoveFileExW` with `MOVEFILE_WRITE_THROUGH` for first install. Keep all unsafe FFI in this module and convert nonzero Win32 errors to `ProjectError::Io` with the destination path.

- [ ] **Step 7: Implement locked-manifest mutation protection**

Before replacing an existing asset manifest, read it through the bounded reader. If it parses and validates as locked, return `ProjectError::LockedAssetMutation`; if it is absent, malformed, or preparing, continue according to the specified persistence semantics while preserving the old file on every failure.

- [ ] **Step 8: Run persistence tests to verify they pass**

Run: `cargo test -p feathertalk-project --test manifest_persistence`

Expected: all persistence tests pass on Windows. Unix-specific tests must be conditionally compiled and pass on Unix CI.

- [ ] **Step 9: Commit persistence**

```powershell
git add rust/crates/feathertalk-project/src rust/crates/feathertalk-project/tests/manifest_persistence.rs
git commit -m "feat: persist manifests atomically"
```

### Task 4: Implement locked asset-package and project-directory validation

**Files:**
- Create: `rust/crates/feathertalk-project/src/package.rs`
- Modify: `rust/crates/feathertalk-project/src/lib.rs`
- Modify: `rust/crates/feathertalk-project/src/error.rs`
- Create: `rust/crates/feathertalk-project/tests/package_validation.rs`
- Modify: `rust/crates/feathertalk-project/tests/support/mod.rs`

**Interfaces:**
- Produces `AssetPackage`, `ValidatedProject`, `lock_asset_package`, and `validate_project_dir`.
- Consumes `read_*_manifest`, `AssetManifest::validate_locked`, and safe relative-path helpers.

- [ ] **Step 1: Write failing package validation tests**

Create a temporary project fixture helper that writes valid manifests and creates required entries. Add tests:

```rust
fn create_complete_project() -> tempfile::TempDir { let dir = tempfile::tempdir().unwrap(); std::fs::create_dir_all(dir.path().join("assets/frames")).unwrap(); std::fs::create_dir_all(dir.path().join("assets/landmarks")).unwrap(); std::fs::create_dir_all(dir.path().join("assets/features")).unwrap(); for file in ["assets/video_25fps.mp4", "assets/audio_16k_mono.wav", "assets/features/feather_hubert.f32"] { std::fs::write(dir.path().join(file), b"x").unwrap(); } dir }
#[test] fn locks_complete_non_empty_asset_package() { let dir = create_complete_project(); let package = lock_asset_package(dir.path(), locked_manifest()).unwrap(); assert_eq!(package.manifest().state, AssetPackageState::Locked); }
#[test] fn lock_rejects_missing_empty_and_wrong_type_artifacts() { let dir = tempfile::tempdir().unwrap(); assert!(lock_asset_package(dir.path(), locked_manifest()).is_err()); }
#[test] fn lock_rejects_symbolic_link_root_and_artifact() { let dir = tempfile::tempdir().unwrap(); let target = create_complete_project(); if support::create_file_symlink(&target.path().join("assets/video_25fps.mp4"), &dir.path().join("assets_video")).is_ok() { assert!(validate_project_dir(dir.path()).is_err()); } }
#[test] fn lock_writes_assets_manifest_last_and_validate_project_dir_round_trips() { let dir = create_complete_project(); let project = valid_project(); write_project_manifest_atomic(&dir.path().join("project.json"), &project).unwrap(); lock_asset_package(dir.path(), locked_manifest()).unwrap(); assert!(validate_project_dir(dir.path()).is_ok()); }
#[test] fn validate_project_rejects_unlocked_assets_manifest() { let dir = create_complete_project(); write_asset_manifest_atomic(&dir.path().join("assets/assets.json"), &preparing_manifest()).unwrap(); assert!(validate_project_dir(dir.path()).is_err()); }
```

- [ ] **Step 2: Run package tests to verify they fail**

Run: `cargo test -p feathertalk-project --test package_validation`

Expected: FAIL because package wrappers and filesystem validation do not exist.

- [ ] **Step 3: Implement safe filesystem entry checks**

Reject a root symlink, inspect each required path with `symlink_metadata`, require regular non-empty files for media/features, and require real directories for frames/landmarks. Do not follow symlinked artifacts. Keep all required paths as fixed crate constants.

- [ ] **Step 4: Implement `lock_asset_package`**

Validate the supplied manifest as locked, validate required artifacts, then write `assets/assets.json` with the atomic asset writer. Return an immutable `AssetPackage` containing the canonical root and manifest.

- [ ] **Step 5: Implement `validate_project_dir`**

Reject a root symlink, read and validate `project.json`, require `asset_package == "assets/assets.json"`, read the referenced asset manifest, require locked state, validate the filesystem contract, and return an immutable `ValidatedProject`.

- [ ] **Step 6: Run package tests to verify they pass**

Run: `cargo test -p feathertalk-project --test package_validation`

Expected: all package tests pass.

- [ ] **Step 7: Commit package validation**

```powershell
git add rust/crates/feathertalk-project/src rust/crates/feathertalk-project/tests/package_validation.rs
git commit -m "feat: validate locked asset packages"
```

### Task 5: Add workspace-wide acceptance coverage and documentation handoff

**Files:**
- Create: `rust/crates/feathertalk-project/tests/public_api.rs`
- Modify: `docs/superpowers/specs/2026-08-20-project-asset-manifest-design.md` only if implementation decisions require a clarified invariant
- Modify: `.superpowers/sdd/2026-08-17-burn-feasibility-loop/progress.md` or create a new SDD ledger for milestone two

**Interfaces:**
- Consumes the complete public API from Tasks 1-4.
- Produces a workspace-level acceptance record and a migration milestone-two checkpoint.

- [ ] **Step 1: Write public API integration tests**

Create `tests/public_api.rs` using only crate-root imports. Construct a preparing `AssetManifest`, call `write_asset_manifest_atomic` and `read_asset_manifest`, construct a complete temporary fixture tree, call `lock_asset_package`, then call `validate_project_dir`. Assert `package.root()`, `package.manifest()`, `project.root()`, and `project.asset_package()` are readable and that no method accepts `&mut AssetManifest` or exposes a mutable manifest reference.

- [ ] **Step 2: Run the focused API tests to verify any missing exports fail**

Run: `cargo test -p feathertalk-project --test public_api`

Expected: FAIL only if a public export or accessor is missing; fix the API surface rather than reaching into private modules.

- [ ] **Step 3: Run the complete verification suite**

Run:

```powershell
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 4: Update the milestone ledger**

Record the new crate, tests, platform coverage, and any intentionally deferred media/model checks. Do not claim milestone two complete; this plan covers only the project/asset manifest subproject.

- [ ] **Step 5: Commit the acceptance checkpoint**

```powershell
git add rust/crates/feathertalk-project/tests/public_api.rs .superpowers/sdd docs/superpowers/specs/2026-08-20-project-asset-manifest-design.md
git commit -m "test: accept project asset manifest contract"
```

## Plan Self-Review

- Spec coverage: schema, strict JSON, bounded reads, path safety, lifecycle states, atomic persistence, Windows/Unix replacement, filesystem artifact checks, public API, tests, and exclusions are covered by Tasks 1-5.
- Completeness scan: every implementation step names concrete files, APIs, validation rules, commands, and expected outcomes.
- Type consistency: Task 1 defines all model types; Task 2 adds validators; Task 3 consumes them for persistence; Task 4 consumes persistence and validators for wrappers; Task 5 consumes the complete public API.
- Scope: this plan does not introduce media processing, face models, worker RPC, GPUI, training, or inference, matching the approved subproject boundary.
