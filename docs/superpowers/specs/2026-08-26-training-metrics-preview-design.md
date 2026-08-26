# Rust Training Metrics and Preview Artifacts Design

Date: 2026-08-26  
Status: Approved for implementation under the project's default-continuation authorization

## 1. Goal and scope

This slice completes the milestone-three contract for training metrics and fixed-sample preview artifacts. It provides strict, serializable Rust values and failure-atomic local persistence that can later be consumed by the worker, RPC layer, and GPUI without importing Burn or FFmpeg into the persistence boundary.

This slice includes:

- A schema-one `TrainingMetrics` snapshot for progress displays and checkpoint-adjacent diagnostics.
- Mode-aware loss component values for Baseline, Mouth ROI, and Mouth ROI + Temporal training.
- A fixed-sample `PreviewArtifact` containing prediction, target, and mouth-ROI tensors for one training sample.
- A bounded little-endian float32 file format and a strict manifest for preview artifacts.
- Atomic write/read helpers with hash, shape, finite-value, directory, and symlink validation.
- Unit and integration tests for round trips, rejection paths, and preservation of existing outputs.

This slice does not implement the training loop, task queue, JSON-lines RPC, GPUI screens, video encoding, or FFmpeg rendering. Those consumers receive the public values and file APIs defined here.

## 2. Design decisions

### 2.1 Metrics are immutable snapshots

`TrainingMetrics` is a value object. A producer constructs a complete snapshot after a successful optimizer update and publishes it as an event or embeds it in a later RPC message. The type does not own a timer, GPU handle, model, data loader, or mutable global state.

The snapshot contains:

- `schema_version: 1`;
- `mode`, `epoch`, and `global_step`;
- `total_loss`, `full_loss`, `perceptual_loss`;
- optional `mouth_loss`, `temporal_loss`, and `temporal_mouth_loss`;
- `samples_seen`, `samples_per_second`, and `estimated_remaining_seconds`;
- `gpu_memory_bytes` (optional because some backends cannot report it);
- `worker_state` as a bounded, lower-case identifier.

All floating-point values must be finite and non-negative. Counters are checked for overflow only when derived by helper constructors. `global_step` and `samples_seen` are ordinary persisted observations; checkpoint sequencing remains the responsibility of the checkpoint coordinator.

The optional loss components are structurally tied to `TrainingMode`:

- Baseline requires mouth, temporal, and temporal-mouth values to be absent.
- Mouth ROI requires mouth and forbids both temporal values.
- Mouth ROI + Temporal requires all three optional values.

This prevents a UI or RPC consumer from displaying a misleading component set.

### 2.2 Preview artifacts use one fixed sample and three arrays

`PreviewArtifact` represents one deterministic sample selected by the training executor. It stores the sample index, reference index, epoch, global step, model kind, model configuration hash, worker state, and three arrays:

- `prediction`: `[3, 160, 160]` float32;
- `target`: `[3, 160, 160]` float32;
- `mouth_roi`: `[3, 160, 160]` float32.

The arrays are stored in channel-major contiguous order. Values must be finite and each array must contain exactly `3 * 160 * 160` elements. The type exposes read-only slices and never exposes a mutable view after construction.

### 2.3 On-disk format and atomic publication

Each preview artifact is a versioned directory with exactly:

```text
manifest.json
prediction.f32
target.f32
mouth-roi.f32
```

Each `.f32` file has a 32-byte little-endian header:

```text
bytes 0..8    magic `FTPV32\\0\\0` (8 bytes)
bytes 8..12   format version u32 = 1
bytes 12..16  rank u32 = 3
bytes 16..20  dimension 0 u32 = 3
bytes 20..24  dimension 1 u32 = 160
bytes 24..28  dimension 2 u32 = 160
bytes 28..32  payload byte count u32 = 307200
```

The payload contains 76,800 little-endian IEEE-754 float32 values. The manifest records schema version, sample metadata, model provenance, each file's exact name/byte count/SHA-256, and worker state. JSON uses `serde(deny_unknown_fields)` and deterministic field order from the Rust structs.

Writes use a unique sibling staging directory, `create_new` files, `flush`/`sync_all`, hash the staged bytes, write and sync the manifest last, sync the parent where supported, and atomically rename staging to the destination. An existing destination is rejected and never modified. Any error removes only the staging directory owned by the current call.

Reads reject directory symlinks, file symlinks, extra/missing entries, oversized files, malformed headers, non-finite payload values, invalid JSON, invalid hashes, and metadata mismatches before returning an artifact.

### 2.4 Bounds and compatibility

- Metrics JSON is bounded at 64 KiB when persisted by a helper.
- Preview manifest is bounded at 64 KiB.
- Each preview tensor file is bounded at 1 MiB (header plus payload).
- Text identifiers are trimmed, non-empty ASCII and capped at 128 bytes.
- SHA-256 values are exactly 64 lower-case hexadecimal characters.
- `model_config_sha256` is required for preview compatibility and is not optional metadata.

## 3. Public API

The `feathertalk-training` crate adds a `telemetry` module and re-exports these concepts:

```rust
pub const TRAINING_METRICS_SCHEMA_VERSION: u32 = 1;
pub const PREVIEW_ARTIFACT_SCHEMA_VERSION: u32 = 1;

pub struct TrainingMetrics { /* strict schema-one snapshot */ }
pub struct PreviewArtifact { /* immutable sample values */ }
pub struct PreviewArtifactManifest { /* strict file manifest */ }
pub struct PreviewFileManifest { /* name, bytes, sha256 */ }

pub fn write_training_metrics(path: impl AsRef<Path>, metrics: &TrainingMetrics)
    -> Result<(), TrainingError>;
pub fn read_training_metrics(path: impl AsRef<Path>)
    -> Result<TrainingMetrics, TrainingError>;

pub fn write_preview_artifact(
    destination: impl AsRef<Path>,
    artifact: &PreviewArtifact,
) -> Result<PreviewArtifactManifest, TrainingError>;
pub fn read_preview_artifact(
    directory: impl AsRef<Path>,
    expected_model_kind: &str,
    expected_model_config_sha256: &str,
) -> Result<(PreviewArtifact, PreviewArtifactManifest), TrainingError>;
```

Constructors and `validate` methods return existing structured `TrainingError` variants. The API never follows symlinks and never silently repairs a malformed artifact.

## 4. Data flow

```text
successful train step
        |
        +--> TrainingMetrics::new(...) --> event/RPC/checkpoint-adjacent consumer
        |
        +--> PreviewArtifact::new(...) --> write_preview_artifact (staging)
                                                  |
                                      manifest-last atomic rename
                                                  |
                                      read_preview_artifact (strict preflight)
```

The preview writer receives already materialized arrays; it does not call the UNet, VGG19, DataLoader, or image decoder. The training executor can therefore decide when a sample is safe to publish without introducing a second definition of model execution.

## 5. Error handling

Invalid values, schema mismatches, and compatibility mismatches use `TrainingError::InvalidCheckpoint` for JSON/value violations and `TrainingError::CheckpointCompatibility` for expected-model mismatches. Filesystem and persistence failures use `Io`, `CheckpointDirectory`, `HashMismatch`, or `Store` with the path and phase in the message. No partial destination is returned as valid, and a failed write cannot alter an existing destination.

## 6. Verification requirements

Tests must prove:

1. Metrics and manifest JSON round-trip exactly and reject unknown fields.
2. Each training mode accepts exactly its required loss components.
3. NaN, infinity, negative values, invalid counters, malformed identifiers, and bad hashes are rejected.
4. Preview headers, dimensions, payload lengths, and finite values are checked.
5. Hash and byte-count tampering is rejected before an artifact is returned.
6. Missing/extra entries, symlinks, oversized files, and existing destinations are rejected.
7. A failure after staged tensor writes leaves no staging directory and preserves any prior destination byte-for-byte.
8. A valid artifact round-trips with exact metadata and array values.

The implementation must pass focused tests, workspace tests, `cargo check`, clippy with `-D warnings`, rustfmt check, and `git diff --check`.

## 7. Non-goals and follow-up

The following remain future slices:

- Wiring metrics into the worker's versioned JSON-lines RPC event stream.
- A full training executor that calls the three loss functions and checkpoint coordinator.
- GPU adapter memory sampling and platform-specific telemetry providers.
- GPUI training and preview screens.
- Video preview rendering and final FFmpeg composition.
