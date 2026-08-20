# Media Normalization Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent `feathertalk-media` crate that validates source media paths and produces a safe, fixed `25 FPS / 16 kHz mono` normalization layout without decoding media or invoking FFmpeg.

**Architecture:** The crate separates unvalidated request structs from read-only validated wrappers. Filesystem validation rejects symbolic links and non-regular inputs before canonicalization; normalization validation creates and canonicalizes the output directory, enforces fixed target values, and derives non-configurable output filenames.

**Tech Stack:** Rust 1.92 edition 2024, standard library filesystem APIs, `thiserror`, and `tempfile` for integration tests.

## Global Constraints

- Add `rust/crates/feathertalk-media` as an independent workspace member.
- The only runtime dependency is `thiserror`.
- Do not add FFmpeg bindings, Burn, WGPU, GPUI, image, model, parity, or `feathertalk-project` dependencies.
- Supported targets are exactly `25 FPS`, `16_000 Hz`, and `1` audio channel.
- Fixed output filenames are exactly `video_25fps.mp4` and `audio_16k_mono.wav`.
- Existing path components must not be symbolic links.
- Validated wrapper fields remain private and expose only immutable path accessors.
- The crate never creates, truncates, renames, or deletes media files; it may only create the output directory.
- Every task follows red-green-refactor TDD and ends with a focused verification command.

---

### Task 1: Bootstrap the crate and public contract types

**Files:**
- Modify: `rust/Cargo.toml`
- Modify: `rust/Cargo.lock` (Cargo-generated)
- Create: `rust/crates/feathertalk-media/Cargo.toml`
- Create: `rust/crates/feathertalk-media/src/lib.rs`
- Create: `rust/crates/feathertalk-media/src/model.rs`
- Create: `rust/crates/feathertalk-media/src/error.rs`
- Create: `rust/crates/feathertalk-media/tests/public_types.rs`

**Interfaces:**
- Produces `MediaInput`, `NormalizationSpec`, `ValidatedInput`, `NormalizedMediaLayout`, and `MediaError`.
- Later tasks construct validated wrappers only through crate-private constructors.

- [ ] **Step 1: Write the failing public type test**

Create `tests/public_types.rs`:

```rust
use std::path::{Path, PathBuf};

use feathertalk_media::{MediaInput, NormalizationSpec};

#[test]
fn request_types_hold_native_paths_and_fixed_target_values() {
    let input = MediaInput { source: PathBuf::from("source/input.mp4") };
    let spec = NormalizationSpec {
        target_video_fps: 25,
        target_audio_sample_rate: 16_000,
        target_audio_channels: 1,
        output_dir: PathBuf::from("assets"),
    };
    assert_eq!(input.source, Path::new("source/input.mp4"));
    assert_eq!(spec.target_video_fps, 25);
    assert_eq!(spec.target_audio_sample_rate, 16_000);
    assert_eq!(spec.target_audio_channels, 1);
    assert_eq!(spec.output_dir, Path::new("assets"));
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p feathertalk-media --test public_types`

Expected: FAIL because the workspace package and public request types do not exist.

- [ ] **Step 3: Add the workspace member and dependency manifest**

Add `"crates/feathertalk-media"` to `rust/Cargo.toml`. Create the crate manifest:

```toml
[package]
name = "feathertalk-media"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[dependencies]
thiserror.workspace = true

[dev-dependencies]
tempfile.workspace = true
```

- [ ] **Step 4: Implement the model and error skeleton**

Define the request structs with `Debug`, `Clone`, `PartialEq`, and `Eq`. Define wrapper structs with private fields, crate-private constructors, and these accessors:

```rust
impl ValidatedInput {
    pub(crate) fn new(source: PathBuf) -> Self;
    pub fn source(&self) -> &Path;
}

impl NormalizedMediaLayout {
    pub(crate) fn new(output_dir: PathBuf, video_path: PathBuf, audio_path: PathBuf) -> Self;
    pub fn output_dir(&self) -> &Path;
    pub fn video_path(&self) -> &Path;
    pub fn audio_path(&self) -> &Path;
}
```

Define every stable `MediaError` category from the design, retaining `std::io::Error` as the source of `Io`.

- [ ] **Step 5: Export the public types and run the test**

Run: `cargo test -p feathertalk-media --test public_types`

Expected: 1 test passes.

- [ ] **Step 6: Commit the crate bootstrap**

```powershell
git add rust/Cargo.toml rust/Cargo.lock rust/crates/feathertalk-media
git commit -m "feat: add media normalization contract crate"
```

### Task 2: Validate source media inputs safely

**Files:**
- Create: `rust/crates/feathertalk-media/src/validate.rs`
- Modify: `rust/crates/feathertalk-media/src/lib.rs`
- Create: `rust/crates/feathertalk-media/tests/input_validation.rs`
- Create: `rust/crates/feathertalk-media/tests/support/mod.rs`

**Interfaces:**
- Consumes `MediaInput`, `ValidatedInput`, and `MediaError` from Task 1.
- Produces `pub fn validate_input(input: &MediaInput) -> Result<ValidatedInput, MediaError>`.

- [ ] **Step 1: Write failing input validation tests**

Cover these concrete cases:

```rust
#[test]
fn validates_and_canonicalizes_a_regular_source_file() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("input.mp4");
    std::fs::write(&source, b"media").unwrap();
    let validated = validate_input(&MediaInput { source: source.clone() }).unwrap();
    assert_eq!(validated.source(), std::fs::canonicalize(source).unwrap());
}

#[test]
fn rejects_missing_input() {
    let dir = tempfile::tempdir().unwrap();
    let error = validate_input(&MediaInput { source: dir.path().join("missing.mp4") }).unwrap_err();
    assert!(matches!(error, MediaError::InputMissing { .. }));
}

#[test]
fn rejects_directory_input() {
    let dir = tempfile::tempdir().unwrap();
    let error = validate_input(&MediaInput { source: dir.path().to_path_buf() }).unwrap_err();
    assert!(matches!(error, MediaError::InputNotRegularFile { .. }));
}

#[test]
fn rejects_a_source_symlink() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("target.mp4");
    std::fs::write(&target, b"media").unwrap();
    let link = dir.path().join("link.mp4");
    if support::create_file_symlink(&target, &link).is_ok() {
        assert!(matches!(validate_input(&MediaInput { source: link }), Err(MediaError::SymlinkNotAllowed { .. })));
    }
}
```

Add a separate test for a symlinked parent component. Implement platform-specific file and directory symlink helpers in `tests/support/mod.rs` behind `cfg(windows)` and `cfg(unix)`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p feathertalk-media --test input_validation`

Expected: FAIL because `validate_input` does not exist.

- [ ] **Step 3: Implement component-wise symlink rejection**

Walk `Path::components()` from the filesystem root. For each existing prefix, call `symlink_metadata`; return `SymlinkNotAllowed` before following a symlink. Preserve missing-leaf handling for `InputMissing`.

- [ ] **Step 4: Implement input type checks and canonicalization**

Use `symlink_metadata` to distinguish missing input from non-regular input. Canonicalize only after the source has passed the symlink and regular-file checks. Map canonicalization failures to `MediaError::Io { operation: "canonicalize_input", .. }`.

- [ ] **Step 5: Run focused tests**

Run: `cargo test -p feathertalk-media --test input_validation`

Expected: all input validation tests pass; symlink assertions run when the platform permits symlink creation.

- [ ] **Step 6: Commit input validation**

```powershell
git add rust/crates/feathertalk-media/src rust/crates/feathertalk-media/tests/input_validation.rs rust/crates/feathertalk-media/tests/support
git commit -m "feat: validate media source paths"
```

### Task 3: Validate normalization targets and fixed output layout

**Files:**
- Modify: `rust/crates/feathertalk-media/src/validate.rs`
- Modify: `rust/crates/feathertalk-media/src/lib.rs`
- Create: `rust/crates/feathertalk-media/tests/normalization_validation.rs`
- Create: `rust/crates/feathertalk-media/tests/public_api.rs`
- Modify: `rust/crates/feathertalk-media/tests/support/mod.rs`

**Interfaces:**
- Consumes `ValidatedInput`, `NormalizationSpec`, `NormalizedMediaLayout`, and `MediaError`.
- Produces `pub fn validate_normalization(input: &ValidatedInput, spec: &NormalizationSpec) -> Result<NormalizedMediaLayout, MediaError>`.

- [ ] **Step 1: Write failing target and layout tests**

Cover:

```rust
#[test]
fn accepts_fixed_target_and_produces_fixed_output_names() {
    let (dir, input) = support::validated_source();
    let output = dir.path().join("assets");
    let layout = validate_normalization(&input, &support::normalization_spec(output.clone())).unwrap();
    assert_eq!(layout.output_dir(), std::fs::canonicalize(output).unwrap());
    assert_eq!(layout.video_path(), layout.output_dir().join("video_25fps.mp4"));
    assert_eq!(layout.audio_path(), layout.output_dir().join("audio_16k_mono.wav"));
}

#[test]
fn rejects_each_unsupported_target_value() {
    let (dir, input) = support::validated_source();
    let mut spec = support::normalization_spec(dir.path().join("assets"));
    spec.target_video_fps = 24;
    assert!(matches!(validate_normalization(&input, &spec), Err(MediaError::UnsupportedTarget { .. })));
    spec.target_video_fps = 25;
    spec.target_audio_sample_rate = 48_000;
    assert!(matches!(validate_normalization(&input, &spec), Err(MediaError::UnsupportedTarget { .. })));
    spec.target_audio_sample_rate = 16_000;
    spec.target_audio_channels = 2;
    assert!(matches!(validate_normalization(&input, &spec), Err(MediaError::UnsupportedTarget { .. })));
}

#[test]
fn creates_a_missing_output_directory() {
    let (dir, input) = support::validated_source();
    let output = dir.path().join("nested/assets");
    validate_normalization(&input, &support::normalization_spec(output.clone())).unwrap();
    assert!(output.is_dir());
}

#[test]
fn rejects_source_inside_output_directory() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("assets");
    std::fs::create_dir(&output).unwrap();
    let source = output.join("input.mp4");
    std::fs::write(&source, b"media").unwrap();
    let input = validate_input(&MediaInput { source }).unwrap();
    assert!(matches!(validate_normalization(&input, &support::normalization_spec(output)), Err(MediaError::OutputInsideInput { .. })));
}
```

Also test an output path that is a file, a symlinked output component, an existing regular destination, and symlink/non-regular fixed destinations.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p feathertalk-media --test normalization_validation`

Expected: FAIL because `validate_normalization` does not exist.

- [ ] **Step 3: Implement exact target validation**

Return `UnsupportedTarget` with exact field names `target_video_fps`, `target_audio_sample_rate`, or `target_audio_channels`, expected values `25`, `16000`, or `1`, and the actual value converted to a string.

- [ ] **Step 4: Implement safe output directory preparation**

Reject symlink components before `create_dir_all`, create missing directories, repeat the symlink-component scan after creation, require a real directory, and canonicalize it. Map failures to the matching stable error category.

- [ ] **Step 5: Implement conflict and destination checks**

Use `Path::starts_with` on canonical paths to reject a source at or below the output directory. Derive the two fixed destination paths. Reject equality with the source. For existing destinations, use `symlink_metadata`; permit only regular non-symlink files.

- [ ] **Step 6: Add crate-root public API acceptance coverage**

Create `tests/public_api.rs` using only crate-root imports. Validate a source and layout, assert canonical accessors, and bind accessors to `&Path` values. Do not access private modules or fields.

- [ ] **Step 7: Run complete verification**

Run:

```powershell
cargo fmt --check
cargo clippy -p feathertalk-media --all-targets --all-features -- -D warnings
cargo test -p feathertalk-media --all-targets
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 8: Commit the completed contract**

```powershell
git add rust/crates/feathertalk-media
git commit -m "feat: validate media normalization layout"
```

## Plan Self-Review

- Spec coverage: crate boundary, dependency exclusions, request models, read-only wrappers, exact target values, component-wise symlink checks, canonicalization, directory creation, source/output conflicts, fixed filenames, existing destination rules, stable errors, and public acceptance are assigned to Tasks 1-3.
- Placeholder scan: no TODO, TBD, deferred implementation instruction, or undefined interface remains.
- Type consistency: Task 1 defines every type used by Tasks 2 and 3; function signatures and accessor names match the approved design.
- Scope: no media parsing, metadata probing, FFmpeg execution, frame processing, models, hashes, or manifest writes are introduced.
