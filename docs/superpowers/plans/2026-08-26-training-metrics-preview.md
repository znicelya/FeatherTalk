# Training Metrics and Preview Artifacts Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

Goal: Add strict Rust training-metrics snapshots and fixed-sample preview artifacts with bounded, failure-atomic local persistence for milestone three.

Architecture: Keep public value types and validation in rust/crates/feathertalk-training/src/telemetry.rs. Keep JSON/binary encoding, hashing, symlink checks, staging, and atomic publication in rust/crates/feathertalk-training/src/telemetry_io.rs. Metrics are immutable schema-one JSON snapshots; previews are immutable four-entry directories containing three fixed-shape float32 files and a manifest. No training loop, Burn execution, FFmpeg, RPC, or GPUI code is introduced.

Tech Stack: Rust 1.92, serde/serde_json, SHA-256, standard filesystem primitives, existing feathertalk-training crate, tempfile integration tests.

## Global Constraints

- Use Rust 1.92 and the existing workspace dependencies; do not add random, image, FFmpeg, Burn, RPC, or UI dependencies.
- TRAINING_METRICS_SCHEMA_VERSION = 1 and PREVIEW_ARTIFACT_SCHEMA_VERSION = 1.
- Every persisted struct uses serde deny_unknown_fields; JSON is bounded at 64 KiB.
- Metrics floating-point values are finite and non-negative; mode determines which optional loss components are present.
- Preview arrays are contiguous channel-major float32 values with shape [3,160,160] and exactly 76,800 elements.
- Preview files use the exact 32-byte little-endian header and exact directory entries defined in the approved spec.
- SHA-256 values are exactly 64 lower-case hexadecimal characters; identifiers are trimmed non-empty ASCII and bounded.
- Reject missing/extra entries, symlinks, malformed headers, wrong byte counts, hash mismatches, non-finite payloads, and metadata mismatches before returning a value.
- Writes use same-parent staging, create_new, flush/sync_all, manifest-last publication, and atomic rename; an existing destination is never overwritten.
- Do not read, modify, stage, commit, or delete demo/kanghui_training_video_featherhubert_188_latest/.
- Never use git add .; stage only the paths named by each task.
- Do not use subagents; execute this plan inline with superpowers:executing-plans.

## File Map

- Create rust/crates/feathertalk-training/src/telemetry.rs for metrics, preview value objects, manifests, constants, constructors, getters, and validation.
- Create rust/crates/feathertalk-training/src/telemetry_io.rs for bounded reads, JSON persistence, preview binary codec, hashing, directory validation, staging guard, and atomic rename.
- Modify rust/crates/feathertalk-training/src/lib.rs for module declarations and public exports.
- Create rust/crates/feathertalk-training/tests/telemetry_schema.rs for metrics/manifest JSON and semantic validation contracts.
- Create rust/crates/feathertalk-training/tests/preview_artifact.rs for binary round-trip and corruption/compatibility checks.
- Create rust/crates/feathertalk-training/tests/telemetry_atomicity.rs for staging, symlink, destination-preservation, and bounded-I/O checks.

---

### Task 1: Define strict training metrics and preview schemas

Files:
- Create rust/crates/feathertalk-training/src/telemetry.rs
- Modify rust/crates/feathertalk-training/src/lib.rs
- Test rust/crates/feathertalk-training/tests/telemetry_schema.rs

Interfaces:
- Export TRAINING_METRICS_SCHEMA_VERSION, PREVIEW_ARTIFACT_SCHEMA_VERSION, PREVIEW_ARTIFACT_FORMAT, PREVIEW_TENSOR_SHAPE, and the four preview file-name constants.
- Export TrainingMetrics, PreviewArtifact, PreviewFileManifest, and PreviewArtifactManifest.
- TrainingMetrics::validate() returns Result<(), TrainingError> and enforces mode/component, finite-number, counter, worker-state, and schema rules.
- PreviewArtifact::new(...), getters, and validate() create/read an immutable fixed-shape sample.
- PreviewArtifactManifest::validate() and validate_against(&PreviewArtifact) enforce exact metadata and file contracts.

- [ ] Step 1: Write the failing schema tests.

Create telemetry_schema.rs with tests that construct all three TrainingMode variants, round-trip strict JSON, add an unknown field and expect deserialization failure, inject a mode/component mismatch and NaN, and construct a valid PreviewArtifact with three vectors of length 76,800. The tests must assert shape [3,160,160], read-only slice lengths, schema constants equal one, and invalid vector lengths return TrainingError::InvalidCheckpoint.

- [ ] Step 2: Run the focused test and verify RED.

Run from rust:

~~~powershell
cargo test -p feathertalk-training --test telemetry_schema
~~~

Expected: compilation fails because the telemetry module, types, and constants do not exist.

- [ ] Step 3: Implement the metrics schema.

Define this strict value type:

~~~rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingMetrics {
    pub schema_version: u32,
    pub mode: TrainingMode,
    pub epoch: u64,
    pub global_step: u64,
    pub total_loss: f64,
    pub full_loss: f64,
    pub perceptual_loss: f64,
    pub mouth_loss: Option<f64>,
    pub temporal_loss: Option<f64>,
    pub temporal_mouth_loss: Option<f64>,
    pub samples_seen: u64,
    pub samples_per_second: f64,
    pub estimated_remaining_seconds: f64,
    pub gpu_memory_bytes: Option<u64>,
    pub worker_state: String,
}
~~~

Use TrainingMode from the existing checkpoint schema. Validate schema one, all five required floating-point values, every present optional component, and a lower-case ASCII worker state matching 1-128 characters from a-z, 0-9, underscore, or hyphen. Enforce exactly these component sets: Baseline has no optional losses; Mouth ROI has only mouth_loss; Mouth ROI + Temporal has all three optional losses. Return TrainingError::InvalidCheckpoint with a field path for every violation.

- [ ] Step 4: Implement the preview value and manifest schemas.

Define fixed constants and private Vec<f32> fields for PreviewArtifact. Its constructor accepts sample_index, reference_index, epoch, global_step, model_kind, model_config_sha256, worker_state, prediction, target, and mouth_roi. Validate all metadata, require each vector length to equal 3*160*160, and reject non-finite values. Expose read-only getters for all metadata, shape, and the three slices. Do not expose mutable production slices.

Define strict serializable PreviewFileManifest with file_name, bytes, sha256 and PreviewArtifactManifest with schema_version, format, sample/reference/epoch/step, model metadata, worker_state, shape, and the three file manifests. Validate exact schema/format/shape/file names, positive bytes, lower-case hashes, and metadata equality with an artifact.

- [ ] Step 5: Export and run GREEN.

Add mod telemetry; mod telemetry_io will be added with Task 2. Re-export schema values from lib.rs. Run cargo test -p feathertalk-training --test telemetry_schema and expect all schema tests to pass.

- [ ] Step 6: Commit the schema slice.

~~~powershell
git add rust/crates/feathertalk-training/src/telemetry.rs rust/crates/feathertalk-training/src/lib.rs rust/crates/feathertalk-training/tests/telemetry_schema.rs
git commit -m "feat: define training metrics and preview schemas"
~~~

---

### Task 2: Add bounded metrics JSON and preview binary I/O

Files:
- Create rust/crates/feathertalk-training/src/telemetry_io.rs
- Modify rust/crates/feathertalk-training/src/telemetry.rs
- Modify rust/crates/feathertalk-training/src/lib.rs
- Test rust/crates/feathertalk-training/tests/preview_artifact.rs

Interfaces:
- Export write_training_metrics and read_training_metrics.
- Export write_preview_artifact and read_preview_artifact with the exact signatures in the approved spec.
- Keep file names, header parsing, hashing, symlink checks, and platform-specific directory syncing private to telemetry_io.rs.

- [ ] Step 1: Write failing round-trip and corruption tests.

Create a valid preview with deterministic arrays and assert write returns schema one, the destination has exactly manifest.json, prediction.f32, target.f32, and mouth-roi.f32, and read returns values and manifest equal to the originals. Add tests that flip one payload byte, replace a binary file with malformed bytes, alter manifest shape or model hash, add/remove entries, and provide a symlink where supported. Each must fail before returning an artifact.

- [ ] Step 2: Run the test and verify RED.

~~~powershell
cargo test -p feathertalk-training --test preview_artifact
~~~

Expected: compilation fails because the I/O functions do not exist.

- [ ] Step 3: Implement the exact binary codec.

Use these constants:

~~~rust
const PREVIEW_MAGIC: [u8; 8] = *b"FTPV32\0\0";
const PREVIEW_HEADER_BYTES: usize = 32;
const PREVIEW_PAYLOAD_BYTES: usize = 76_800 * 4;
const PREVIEW_MAX_FILE_BYTES: u64 = 1 * 1024 * 1024;
const JSON_MAX_BYTES: u64 = 64 * 1024;
~~~

Write fields in little-endian order: magic, version 1, rank 3, dimensions 3,160,160, payload bytes 307200, followed by each f32::to_le_bytes value. On read, reject mismatch, truncation/trailing bytes, wrong dimensions, wrong payload count, and non-finite values. Hash with SHA-256 in 64 KiB chunks and return lower-case hex::encode.

- [ ] Step 4: Implement strict directory and manifest preflight.

validate_preview_directory must use symlink_metadata, reject a directory symlink, reject file symlinks, and require exactly manifest.json, prediction.f32, target.f32, and mouth-roi.f32. read_preview_artifact must perform this sequence before decoding payloads:

~~~text
validate directory and entries
bounded-read and strict-parse manifest
validate manifest schema/format/shape/file declarations
compare expected model kind and config hash
validate declared byte counts and SHA-256 for all three files
decode all three headers and finite float32 payloads
construct and validate PreviewArtifact against manifest
return cloned value and manifest
~~~

Map expected-model mismatches to TrainingError::CheckpointCompatibility; malformed values to InvalidCheckpoint; filesystem shape errors to CheckpointDirectory; and hashes to HashMismatch.

- [ ] Step 5: Implement metrics JSON persistence.

write_training_metrics validates the value, serializes bounded JSON, rejects an existing/symlink destination, writes a unique sibling temporary file with create_new, flushes and syncs it, atomically renames it to the destination, and syncs the parent where supported. read_training_metrics rejects symlinks, bounded-reads JSON, strictly deserializes it, and calls validate().

- [ ] Step 6: Implement preview staging and atomic publication.

write_preview_artifact validates the value, rejects symlink parent components, creates the destination parent, rejects an existing destination, creates a unique same-parent .preview-<pid>-<counter>.staging directory, writes and syncs the three binary files, computes actual manifests, constructs and validates the manifest, writes/syncs manifest.json last, syncs staging, renames staging to destination, disarms cleanup, and syncs the parent. Any error removes only the owned staging directory.

- [ ] Step 7: Run focused tests and commit.

~~~powershell
cargo test -p feathertalk-training --test preview_artifact
cargo fmt --all -- --check
cargo clippy -p feathertalk-training --all-targets -- -D warnings
~~~

~~~powershell
git add rust/crates/feathertalk-training/src/telemetry.rs rust/crates/feathertalk-training/src/telemetry_io.rs rust/crates/feathertalk-training/src/lib.rs rust/crates/feathertalk-training/tests/preview_artifact.rs
git commit -m "feat: persist fixed training preview artifacts"
~~~

---

### Task 3: Prove failure atomicity, bounds, and compatibility

Files:
- Modify rust/crates/feathertalk-training/tests/telemetry_atomicity.rs
- Modify rust/crates/feathertalk-training/tests/preview_artifact.rs
- Modify rust/crates/feathertalk-training/src/telemetry_io.rs only for verified defects

Interfaces:
- No new public API; tests exercise coordinator functions from Tasks 1-2.

- [ ] Step 1: Add atomicity and bounded-I/O tests.

Cover existing destination preservation, invalid preview with no staging directory, oversized metrics JSON and unknown-field rejection, symlinked parent or entry rejection where supported, exact four-entry directory, no staging sibling after success, actual manifest byte counts, and fresh SHA-256 matches.

- [ ] Step 2: Run focused tests and inspect files.

~~~powershell
cargo test -p feathertalk-training --test telemetry_schema
cargo test -p feathertalk-training --test preview_artifact
cargo test -p feathertalk-training --test telemetry_atomicity
~~~

Inspect a temporary parent and expect no staging directory remains and no existing destination changes.

- [ ] Step 3: Run package-level verification and commit any verified fix explicitly.

~~~powershell
cargo check -p feathertalk-training --all-targets
cargo clippy -p feathertalk-training --all-targets -- -D warnings
cargo fmt --all -- --check
git diff --check
~~~

If a concrete defect is found, stage only telemetry paths and commit message fix: harden training telemetry persistence.

---

### Task 4: Full verification and integration

- [ ] Step 1: Run cargo test --workspace --all-targets, cargo check --workspace --all-targets, cargo clippy --workspace --all-targets -- -D warnings, cargo fmt --all -- --check, and git diff --check. Record exit codes and ignored hardware/license tests.
- [ ] Step 2: Review requirements line by line: schema-one metrics, mode-aware components, finite values, fixed [3,160,160] arrays, exact 32-byte header, strict four-entry directory, manifest-last atomic write, bounded reads, hashes/byte checks, symlink rejection, expected-model compatibility, destination preservation, staging cleanup, and exact round-trip values. Confirm no protected demo path appears in status or diff.
- [ ] Step 3: Fast-forward training-metrics-preview into main, rerun focused telemetry tests and workspace check on merged main, then remove only its worktree and branch. Leave all other worktrees and the protected demo untouched.
- [ ] Step 4: Read the migration design again. If milestone three has no remaining contract, begin the next unmet milestone-four slice with a new design/plan cycle and continue automatically.

## Plan Self-Review

- Spec coverage: metrics schema and mode rules are Task 1; binary/header and strict load order are Task 2; atomicity and bounds are Task 3; full integration and milestone continuation are Task 4.
- Placeholder scan: all implementation steps name concrete files, constants, field names, commands, and expected outcomes; no TODO/TBD or vague edge-case instruction remains.
- Type consistency: TrainingMetrics, PreviewArtifact, PreviewFileManifest, PreviewArtifactManifest, write_*, and read_* names and argument order are identical across tasks and the approved design.
- Scope: no training loop, Burn model execution, RPC, GPUI, FFmpeg, or protected demo access is introduced.

