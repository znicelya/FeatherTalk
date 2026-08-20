# Media Normalization Contract Design

Date: 2026-08-20  
Status: Approved for implementation planning

## 1. Purpose

Create a small Rust crate that defines and validates the inputs and fixed output contract for FeatherTalk media normalization. This is the next bounded part of migration milestone two after project and asset manifest support.

The crate validates whether a source media file and normalization destination are safe to hand to a later FFmpeg adapter. It does not decode media, inspect streams, launch FFmpeg, or claim that the source already satisfies the target media properties.

## 2. Scope

Included:

- Source media path validation.
- Output directory validation and creation.
- Fixed normalization target validation.
- Canonical path comparison and conflict prevention.
- Fixed normalized video and audio output paths.
- Read-only validated wrapper types.
- Stable structured errors.

Excluded:

- Video or audio decoding.
- WAV parsing.
- Media metadata probing.
- FFmpeg discovery, command construction, execution, cancellation, or stderr parsing.
- Frame extraction and image encoding.
- SCRFD, PFLD, FeatherHuBERT, hashing, and manifest updates.

## 3. Crate Boundary

Add an independent workspace crate:

```text
rust/crates/feathertalk-media/
  Cargo.toml
  src/
    lib.rs
    model.rs
    validate.rs
    error.rs
  tests/
```

The crate depends only on the Rust standard library and `thiserror`. It must not depend on FFmpeg bindings, Burn, WGPU, GPUI, image libraries, model crates, parity crates, or `feathertalk-project`.

The later FFmpeg adapter will consume this crate's validated types. Keeping execution out of this crate allows path and configuration rules to run on every CPU CI platform without native media dependencies.

## 4. Public Data Model

### 4.1 Media Input

```rust
pub struct MediaInput {
    pub source: PathBuf,
}
```

`MediaInput` is an unvalidated request. Its public path does not imply that the file exists or is safe to open.

### 4.2 Normalization Specification

```rust
pub struct NormalizationSpec {
    pub target_video_fps: u32,
    pub target_audio_sample_rate: u32,
    pub target_audio_channels: u16,
    pub output_dir: PathBuf,
}
```

Schema version one supports exactly:

- `target_video_fps == 25`
- `target_audio_sample_rate == 16_000`
- `target_audio_channels == 1`

No alternate target is accepted in this subproject. Adding another target requires an explicit versioned contract change.

### 4.3 Validated Input

`ValidatedInput` contains the canonical source path and exposes it through a read-only accessor:

```rust
pub fn source(&self) -> &Path;
```

The type has no public fields or mutable path accessor. Construction is only possible through `validate_input`.

### 4.4 Normalized Media Layout

`NormalizedMediaLayout` contains:

- Canonical output directory.
- Normalized video path: `video_25fps.mp4`.
- Normalized audio path: `audio_16k_mono.wav`.

It exposes read-only accessors:

```rust
pub fn output_dir(&self) -> &Path;
pub fn video_path(&self) -> &Path;
pub fn audio_path(&self) -> &Path;
```

Callers cannot choose the output filenames or inject relative path components.

## 5. Public API

```rust
pub fn validate_input(input: &MediaInput) -> Result<ValidatedInput, MediaError>;

pub fn validate_normalization(
    input: &ValidatedInput,
    spec: &NormalizationSpec,
) -> Result<NormalizedMediaLayout, MediaError>;
```

Validation is intentionally split into two steps so later execution code can accept only a `ValidatedInput`. Configuration validation and output directory preparation happen in `validate_normalization`.

## 6. Input Validation

`validate_input` applies these rules in order:

1. The source path must exist.
2. Every existing path component from the filesystem root through the source must not be a symbolic link.
3. The source must be a regular file.
4. The source is canonicalized only after the preceding checks pass.

The crate does not enforce a source extension. Containers and codecs are determined by the later media probe and FFmpeg adapter, not by filenames.

The crate does not impose a source file-size limit. A future execution layer may impose resource limits based on available disk space, duration, and media metadata.

## 7. Normalization Validation

`validate_normalization` applies these rules:

1. All three target values must match the fixed schema-one target.
2. If the output directory is absent, it is created recursively.
3. Every existing component of the output path must not be a symbolic link.
4. The resulting output path must be a real directory.
5. The directory is canonicalized after validation and creation.
6. The source must not be located at or below the output directory.
7. Neither fixed output path may equal the source path.
8. If a fixed output destination already exists, it must be a regular file and must not be a symbolic link. Existing regular files are permitted because the later atomic execution layer may replace them.

The source-outside-output rule prevents the standardization task from recursively consuming or overwriting a file stored in its own destination tree. The output directory may be located below the source file's parent directory, provided the source itself is not inside that output directory.

## 8. Path Semantics

All comparisons use canonical filesystem paths after symbolic-link checks. This prevents bypasses using `..`, Windows short paths, mixed path spellings, or case normalization performed by the filesystem.

The API accepts native platform paths. It does not store these paths in manifests and therefore does not apply the slash-separated project-relative path rules from `feathertalk-project`.

The output filenames are constants:

```text
video_25fps.mp4
audio_16k_mono.wav
```

The crate does not create either output file. It only creates and validates the containing directory and computes the fixed paths.

## 9. Error Model

`MediaError` is a `thiserror` enum with stable categories:

```rust
pub enum MediaError {
    Io { operation, path, source },
    InputMissing { path },
    InputNotRegularFile { path },
    SymlinkNotAllowed { path },
    OutputDirectoryInvalid { path },
    OutputInsideInput { input, output },
    OutputConflictsWithInput { path },
    OutputDestinationInvalid { path },
    UnsupportedTarget { field, expected, actual },
}
```

`UnsupportedTarget` stores values as strings so the same stable category can report integer fields of different widths. I/O errors retain the technical source error and operation for diagnostics.

Errors do not expose FFmpeg terminology because no FFmpeg operation occurs in this crate.

## 10. Filesystem Mutation and Atomicity

The only permitted mutation is creating a missing output directory. The crate never creates, truncates, renames, or deletes media files.

Directory creation may leave newly created directories behind if a later validation rule fails. Empty-directory rollback is deliberately excluded because it introduces destructive behavior and races without protecting any completed artifact. The later execution layer owns temporary file cleanup and atomic replacement.

## 11. Tests

### 11.1 Configuration Tests

- Accept exactly `25 / 16000 / 1`.
- Reject an unsupported frame rate.
- Reject an unsupported sample rate.
- Reject an unsupported channel count.
- Produce the two exact fixed filenames.

### 11.2 Filesystem Tests

- Reject a missing input.
- Reject a directory as input.
- Reject a symbolic-link source or source component where the platform permits symlink creation.
- Create a missing output directory.
- Reject an output path that resolves to a file.
- Reject a symbolic-link output directory or output component where supported.
- Reject a source located inside the output directory.
- Reject a fixed output destination that resolves to the source.
- Permit an existing regular output destination.
- Reject a symbolic-link or non-regular output destination.

### 11.3 Public API Tests

- Import all public types and functions from the crate root.
- Validate a real source and inspect the canonical source accessor.
- Validate the normalization layout and inspect all read-only accessors.
- Confirm the wrapper types do not expose public mutable fields through normal compilation usage.

## 12. Acceptance Criteria

- `feathertalk-media` is a workspace member.
- The crate has no forbidden dependency.
- All path and fixed-target rules above are covered by focused tests.
- The API returns canonical, read-only validated types.
- `cargo fmt --check` passes.
- `cargo clippy -p feathertalk-media --all-targets --all-features -- -D warnings` passes.
- `cargo test -p feathertalk-media --all-targets` passes on Windows and is written for Windows, macOS, and Linux behavior.
- `git diff --check` passes.

This completes only the media normalization contract and input-validation slice. It does not complete milestone two or media normalization execution.
