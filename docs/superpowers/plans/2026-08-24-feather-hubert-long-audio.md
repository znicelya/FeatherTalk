# FeatherHuBERT Long Audio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce Python FeatherHuBERT long-audio chunking and publish a validated, versioned `.f32` feature artifact plus locked `assets.json` atomically.

**Architecture:** `feathertalk-audio` owns pure chunk planning, normalization, stitching, feature encoding, and artifact persistence. `feathertalk-models` depends on it only to expose a Burn encoder adapter; the audio crate never depends on Burn or model crates. Existing `feathertalk-project` remains the authority for locked manifest validation and atomic JSON replacement.

**Tech Stack:** Rust 1.92 edition 2024, serde/serde_json, sha2, thiserror, Burn 0.21, tempfile.

## Global Constraints

- Kernel 400, stride 320, chunk 320000, overlap extension 80.
- Target token count is `0` below 400 samples, otherwise `(samples - 80) / 320`.
- Tail starts at `chunk_samples * floor(samples / chunk_samples)`.
- Short output is zero padded; long output is cropped; odd token is dropped.
- No shell, Python runtime, image/OpenCV dependency, or silent CPU fallback.
- All arithmetic and file sizes are bounded and checked.
- Existing locked assets and any previous final output remain unchanged on failure.
- The user demo directory remains untracked and untouched.

---

### Task 1: Pure long-audio contracts and chunk planner

**Files:**
- Create: `rust/crates/feathertalk-audio/Cargo.toml`
- Modify: `rust/Cargo.toml`, `rust/Cargo.lock`
- Create: `rust/crates/feathertalk-audio/src/lib.rs`, `src/error.rs`, `src/chunk.rs`, `src/normalize.rs`
- Test: `rust/crates/feathertalk-audio/tests/chunking.rs`, `tests/normalization.rs`

**Interfaces:**

```rust
pub const SAMPLE_RATE: usize = 16_000;
pub const HUBERT_KERNEL: usize = 400;
pub const HUBERT_STRIDE: usize = 320;
pub const DEFAULT_CHUNK_SAMPLES: usize = 320_000;

pub struct ChunkRange { pub index: usize, pub start: usize, pub end: usize }
pub struct ChunkPlan { pub total_samples: usize, pub target_tokens: usize, pub chunks: Vec<ChunkRange> }
pub fn expected_hubert_frames(samples: usize) -> usize;
pub fn plan_chunks(samples: usize, chunk_samples: usize) -> Result<ChunkPlan, AudioError>;
pub fn normalize_waveform(samples: &[f32]) -> Result<Vec<f32>, AudioError>;
```

- [ ] Write tests for all boundary lengths, exact Python ranges, checked overflow, empty/short input, and normalization finite/constant behavior.
- [ ] Run `cargo test -p feathertalk-audio --test chunking --test normalization`; confirm red because the crate/API do not exist.
- [ ] Implement minimal bounded planner and normalization.
- [ ] Run the focused tests green and commit `feat: add long audio chunk contracts`.

### Task 2: Encoder seam, stitching, even-token output, and `.f32` format

**Files:**
- Modify: `rust/crates/feathertalk-audio/src/lib.rs`, `src/error.rs`
- Create: `rust/crates/feathertalk-audio/src/stitch.rs`, `src/format.rs`
- Test: `tests/stitching.rs`, `tests/format.rs`

**Interfaces:**

```rust
pub trait ChunkEncoder {
    fn output_dim(&self) -> usize;
    fn encode(&mut self, chunk_index: usize, samples: &[f32]) -> Result<Vec<f32>, AudioError>;
}
pub struct FeatureMatrix { pub tokens: usize, pub dims: usize, pub values: Vec<f32> }
pub fn extract_long_audio<E: ChunkEncoder>(samples: &[f32], encoder: &mut E, chunk_samples: usize) -> Result<FeatureMatrix, AudioError>;
pub fn drop_odd_token(matrix: FeatureMatrix) -> FeatureMatrix;
pub fn write_feature_file(path: &Path, matrix: &FeatureMatrix) -> Result<FeatureArtifact, AudioError>;
pub fn read_feature_file(path: &Path) -> Result<FeatureMatrix, AudioError>;
```

- [ ] Write a fake encoder that emits chunk-index rows; test ordering, exact slice lengths, crop/pad, odd-token removal, output-dimension mismatch, and non-finite rejection.
- [ ] Run focused stitching tests red.
- [ ] Implement extraction and strict feature header/payload codec with a fixed 16 MiB per-file bound for tests and a checked production bound.
- [ ] Run focused tests green and commit `feat: stitch long audio features`.

### Task 3: Atomic feature artifact and manifest commit

**Files:**
- Modify: `rust/crates/feathertalk-audio/src/lib.rs`, `src/error.rs`
- Create: `rust/crates/feathertalk-audio/src/commit.rs`
- Test: `tests/commit.rs`

**Interfaces:**

```rust
pub struct FeatureCommitSpec { pub project_root: PathBuf, pub frame_count: u64, pub frame_width: u32, pub frame_height: u32, pub landmark_model_sha256: String, pub feature_model_sha256: String }
pub fn commit_feature_artifact(spec: &FeatureCommitSpec, matrix: &FeatureMatrix) -> Result<FeatureArtifact, AudioError>;
```

- [ ] Write tests for successful commit, hash/byte/shape checks, existing preparing replacement, locked-manifest rejection, staging collision, late manifest failure, rollback, and old-output preservation.
- [ ] Run `cargo test -p feathertalk-audio --test commit`; confirm red.
- [ ] Implement sibling staging, fsync, atomic feature rename, manifest write, rollback, and cleanup ownership.
- [ ] Run focused tests green and commit `feat: commit audio features atomically`.

### Task 4: Burn FeatherHuBERT adapter

**Files:**
- Modify: `rust/crates/feathertalk-models/Cargo.toml`
- Create: `rust/crates/feathertalk-models/src/feather_hubert/adapter.rs`
- Modify: `rust/crates/feathertalk-models/src/feather_hubert/mod.rs`
- Test: `rust/crates/feathertalk-models/tests/feather_hubert_long_audio.rs`

**Interfaces:**

```rust
pub struct BurnFeatherHubertEncoder<B: Backend> { model: FeatherHubertEncoder<B>, device: B::Device }
impl<B: Backend> ChunkEncoder for BurnFeatherHubertEncoder<B> { ... }
```

- [ ] Write CPU micro-model test for chunk encoding shape and finite output; test long-audio extraction through the adapter and odd-token rule.
- [ ] Run test red.
- [ ] Implement tensor conversion, eval forward, data extraction, and finite/shape checks.
- [ ] Add WGPU smoke test with explicit adapter check and ignored behavior when unavailable.
- [ ] Run model-focused tests and commit `feat: adapt burn feather hubert to long audio`.

### Task 5: Integration, docs, and full verification

- [ ] Add a public API test proving project manifest and feature artifact agree on `[frame_count,2,1024]`.
- [ ] Run `cargo fmt --all -- --check`.
- [ ] Run `cargo clippy --workspace --all-targets --all-features -- -D warnings`.
- [ ] Run `cargo test --workspace --all-targets`.
- [ ] Run `cargo check --workspace --all-targets`.
- [ ] Run `git diff --check` and verify the demo directory is still untracked only.
- [ ] Commit `feat: add feather hubert feature artifact pipeline`, fast-forward merge to `main`, and repeat fresh targeted/workspace verification.

## Plan Self-Review

- Python chunk boundaries, target token count, padding/cropping, and odd-token behavior each have explicit tasks/tests.
- The feature header and manifest shape/hash contract are covered before worker/UI work.
- Burn coupling is one-way (`models -> audio`), so the pure seam remains testable without GPU.
- No placeholder tasks or unbounded reads/writes are present.
