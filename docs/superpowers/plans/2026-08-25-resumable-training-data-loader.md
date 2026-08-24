# Resumable Training DataLoader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, serializable and explicitly committed Rust training DataLoader whose shuffle and reference choices resume exactly across process restarts.

**Architecture:** Keep the public sampling contract and loader lifecycle in `data.rs`, with a private fixed SplitMix64/Fisher–Yates implementation in `random.rs`. The loader owns a generic `TrainingDataset`, derives all random choices before materialization, returns an opaque `PreparedBatch`, and advances state only when the caller commits that exact batch after a successful optimizer step.

**Tech Stack:** Rust 1.92, serde/serde_json, thiserror, standard-library checked arithmetic and allocation APIs, existing `feathertalk-training` crate.

## Global Constraints

- Run implementation in `.worktrees/resumable-training-data-loader` on branch `resumable-training-data-loader`; run Rust commands from that worktree's `rust/` directory.
- Do not use subagents; execute inline with `superpowers:executing-plans`.
- Every production behavior follows RED, then minimal GREEN, then focused regression.
- Do not add a direct `rand`, image, OpenCV, filesystem, `feathertalk-project`, or `feathertalk-preprocess` dependency to `feathertalk-training`.
- Keep `drop_last = false`; preserve the final partial batch.
- Single-frame reference selection may equal the target, matching Python's `random.randint(0, frame_count - 1)` behavior.
- Temporal samples contain one shared reference for targets `i` and `i + temporal_stride`.
- The persisted state never contains the full epoch permutation.
- Unknown schema fields, unsupported versions/algorithms, invalid config, dataset frame-count mismatch and invalid cursors fail before sample loading.
- `prepare_next_batch` never advances persistent state; only a valid `commit_batch` advances it.
- Do not read, modify, stage, commit or delete `demo/kanghui_training_video_featherhubert_188_latest/`.
- Never use `git add .`; stage explicit paths only.

---

## File Map

- Create `rust/crates/feathertalk-training/src/random.rs`: fixed constants, SplitMix64 stream derivation, unbiased bounded integers and Fisher–Yates permutation construction.
- Create `rust/crates/feathertalk-training/src/data.rs`: public config/state/sample types, validation, dataset trait, opaque prepared batch and generic loader lifecycle.
- Modify `rust/crates/feathertalk-training/src/error.rs`: structured DataLoader config/state/overflow/allocation/stale-batch errors.
- Modify `rust/crates/feathertalk-training/src/lib.rs`: crate-root exports for the complete public data contract.
- Create `rust/crates/feathertalk-training/tests/data_loader.rs`: literal sampling, batch boundaries and commit failure-atomicity.
- Create `rust/crates/feathertalk-training/tests/data_loader_recovery.rs`: serde contract, strict restore validation and uninterrupted/resumed equivalence.

---

### Task 1: Define the strict public configuration and state schema

**Files:**

- Create: `rust/crates/feathertalk-training/src/data.rs`
- Modify: `rust/crates/feathertalk-training/src/error.rs`
- Modify: `rust/crates/feathertalk-training/src/lib.rs`
- Create: `rust/crates/feathertalk-training/tests/data_loader.rs`

**Interfaces:**

- Produces `DATA_LOADER_STATE_SCHEMA_VERSION: u32 = 1`.
- Produces `RandomAlgorithm::Splitmix64FisherYatesV1` serialized as `splitmix64_fisher_yates_v1`.
- Produces `SamplingKind`, `SamplingConfig`, `DataLoaderConfig`, `DataLoaderState` and `TrainingSample`.
- Produces `DataLoaderConfig::single_frame(batch_size, seed)` and `DataLoaderConfig::temporal_pair(batch_size, seed, temporal_stride)`.
- Produces internal `DataLoaderConfig::sample_count(frame_count) -> Result<u64, TrainingError>` for later tasks.

- [ ] **Step 1: Write failing public-schema and validation tests**

Create `tests/data_loader.rs` with crate-root imports and these literal assertions:

```rust
use feathertalk_training::{
    DATA_LOADER_STATE_SCHEMA_VERSION, DataLoaderConfig, DataLoaderState, RandomAlgorithm,
    SamplingConfig, SamplingKind, TrainingError,
};

#[test]
fn single_and_temporal_configs_expose_the_fixed_contract() {
    assert_eq!(DATA_LOADER_STATE_SCHEMA_VERSION, 1);
    assert_eq!(
        DataLoaderConfig::single_frame(4, 7),
        DataLoaderConfig {
            batch_size: 4,
            seed: 7,
            sampling: SamplingConfig {
                kind: SamplingKind::SingleFrame,
                temporal_stride: 0,
            },
        }
    );
    assert_eq!(
        DataLoaderConfig::temporal_pair(3, 42, 2),
        DataLoaderConfig {
            batch_size: 3,
            seed: 42,
            sampling: SamplingConfig {
                kind: SamplingKind::TemporalPair,
                temporal_stride: 2,
            },
        }
    );
}

#[test]
fn invalid_config_and_state_are_rejected_before_loading() {
    let invalid = [
        DataLoaderConfig::single_frame(0, 7),
        DataLoaderConfig {
            batch_size: 2,
            seed: 7,
            sampling: SamplingConfig {
                kind: SamplingKind::SingleFrame,
                temporal_stride: 1,
            },
        },
        DataLoaderConfig::temporal_pair(2, 7, 0),
    ];
    for config in invalid {
        assert!(matches!(
            config.validate(5),
            Err(TrainingError::InvalidDataLoaderConfig(_))
        ));
    }

    assert!(matches!(
        DataLoaderConfig::single_frame(2, 7).validate(0),
        Err(TrainingError::InvalidDataLoaderConfig(_))
    ));
    assert!(matches!(
        DataLoaderConfig::temporal_pair(2, 7, 5).validate(5),
        Err(TrainingError::InvalidDataLoaderConfig(_))
    ));

    let state = DataLoaderState {
        schema_version: 99,
        random_algorithm: RandomAlgorithm::Splitmix64FisherYatesV1,
        config: DataLoaderConfig::single_frame(2, 7),
        frame_count: 5,
        epoch: 0,
        next_position: 0,
    };
    assert!(matches!(
        state.validate(5),
        Err(TrainingError::InvalidDataLoaderState(_))
    ));
}
```

Also test invalid canonical cursors using schema one: `next_position == sample_count`, and temporal `frame_count <= temporal_stride`. The tests must call only crate-root public APIs.

- [ ] **Step 2: Run the focused test to verify RED**

Run:

```powershell
cargo test -p feathertalk-training --test data_loader
```

Expected: compilation fails because the public DataLoader types, constants and error variants do not exist.

- [ ] **Step 3: Add structured DataLoader errors**

Extend `TrainingError` without changing existing variants:

```rust
#[error("invalid data loader configuration: {0}")]
InvalidDataLoaderConfig(String),
#[error("invalid data loader state: {0}")]
InvalidDataLoaderState(String),
#[error("data loader arithmetic overflow while {operation}")]
DataLoaderOverflow { operation: &'static str },
```

- [ ] **Step 4: Implement the schema and validation**

Create `data.rs` with serde deny-unknown-fields and snake-case enums:

```rust
use serde::{Deserialize, Serialize};

use crate::TrainingError;

pub const DATA_LOADER_STATE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RandomAlgorithm {
    Splitmix64FisherYatesV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingKind {
    SingleFrame,
    TemporalPair,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingConfig {
    pub kind: SamplingKind,
    pub temporal_stride: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataLoaderConfig {
    pub batch_size: u64,
    pub seed: u64,
    pub sampling: SamplingConfig,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataLoaderState {
    pub schema_version: u32,
    pub random_algorithm: RandomAlgorithm,
    pub config: DataLoaderConfig,
    pub frame_count: u64,
    pub epoch: u64,
    pub next_position: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainingSample {
    SingleFrame {
        target_index: u64,
        reference_index: u64,
    },
    TemporalPair {
        first_target_index: u64,
        second_target_index: u64,
        reference_index: u64,
    },
}
```

Validation must use `usize::try_from` for `frame_count`, sample count and batch size, checked subtraction for temporal sample count, require `next_position < sample_count`, and return the new structured variants. `DataLoaderState::validate(dataset_frame_count)` must check schema one, the fixed algorithm, exact dataset/state frame-count equality, config validity and canonical cursor validity.

Implement the methods with this exact control flow:

```rust
impl DataLoaderConfig {
    pub const fn single_frame(batch_size: u64, seed: u64) -> Self {
        Self {
            batch_size,
            seed,
            sampling: SamplingConfig {
                kind: SamplingKind::SingleFrame,
                temporal_stride: 0,
            },
        }
    }

    pub const fn temporal_pair(batch_size: u64, seed: u64, temporal_stride: u64) -> Self {
        Self {
            batch_size,
            seed,
            sampling: SamplingConfig {
                kind: SamplingKind::TemporalPair,
                temporal_stride,
            },
        }
    }

    pub fn validate(&self, frame_count: u64) -> Result<(), TrainingError> {
        self.sample_count(frame_count).map(|_| ())
    }

    pub(crate) fn sample_count(&self, frame_count: u64) -> Result<u64, TrainingError> {
        if self.batch_size == 0 {
            return Err(TrainingError::InvalidDataLoaderConfig(
                "batch_size must be greater than zero".into(),
            ));
        }
        usize::try_from(self.batch_size).map_err(|_| TrainingError::DataLoaderOverflow {
            operation: "converting batch size",
        })?;
        if frame_count == 0 {
            return Err(TrainingError::InvalidDataLoaderConfig(
                "frame_count must be greater than zero".into(),
            ));
        }
        usize::try_from(frame_count).map_err(|_| TrainingError::DataLoaderOverflow {
            operation: "converting frame count",
        })?;

        let sample_count = match self.sampling.kind {
            SamplingKind::SingleFrame => {
                if self.sampling.temporal_stride != 0 {
                    return Err(TrainingError::InvalidDataLoaderConfig(
                        "single_frame requires temporal_stride zero".into(),
                    ));
                }
                frame_count
            }
            SamplingKind::TemporalPair => {
                let stride = self.sampling.temporal_stride;
                if stride == 0 || stride >= frame_count {
                    return Err(TrainingError::InvalidDataLoaderConfig(
                        "temporal_pair requires 1 <= temporal_stride < frame_count".into(),
                    ));
                }
                frame_count.checked_sub(stride).ok_or(
                    TrainingError::DataLoaderOverflow {
                        operation: "computing temporal sample count",
                    },
                )?
            }
        };
        usize::try_from(sample_count).map_err(|_| TrainingError::DataLoaderOverflow {
            operation: "converting sample count",
        })?;
        Ok(sample_count)
    }
}

impl DataLoaderState {
    pub fn validate(&self, dataset_frame_count: u64) -> Result<(), TrainingError> {
        if self.schema_version != DATA_LOADER_STATE_SCHEMA_VERSION {
            return Err(TrainingError::InvalidDataLoaderState(
                "unsupported schema_version".into(),
            ));
        }
        if self.random_algorithm != RandomAlgorithm::Splitmix64FisherYatesV1 {
            return Err(TrainingError::InvalidDataLoaderState(
                "unsupported random_algorithm".into(),
            ));
        }
        if self.frame_count != dataset_frame_count {
            return Err(TrainingError::InvalidDataLoaderState(
                "dataset frame_count does not match saved state".into(),
            ));
        }
        let sample_count = self.config.sample_count(self.frame_count)?;
        if self.next_position >= sample_count {
            return Err(TrainingError::InvalidDataLoaderState(
                "next_position must be inside the current epoch".into(),
            ));
        }
        Ok(())
    }
}
```

- [ ] **Step 5: Export the public schema and verify GREEN**

Add `mod data;` and explicit crate-root exports in `lib.rs`. Run:

```powershell
cargo test -p feathertalk-training --test data_loader
cargo test -p feathertalk-training --all-targets
```

Expected: schema/validation tests pass and all existing training tests remain green.

- [ ] **Step 6: Commit Task 1**

```powershell
git add rust/crates/feathertalk-training/src/data.rs rust/crates/feathertalk-training/src/error.rs rust/crates/feathertalk-training/src/lib.rs rust/crates/feathertalk-training/tests/data_loader.rs
git commit -m "feat: define resumable data loader state"
```

---

### Task 2: Implement the fixed version-one random algorithm

**Files:**

- Create: `rust/crates/feathertalk-training/src/random.rs`
- Modify: `rust/crates/feathertalk-training/src/error.rs`
- Modify: `rust/crates/feathertalk-training/src/lib.rs`

**Interfaces:**

- Consumes `TrainingError` from Task 1.
- Produces private `epoch_permutation(sample_count, seed, epoch) -> Result<Vec<u64>, TrainingError>`.
- Produces private `reference_index(frame_count, seed, epoch, position) -> Result<u64, TrainingError>`.

- [ ] **Step 1: Create private literal tests before the implementation**

Create `random.rs` containing only imports and this test module first, then add `mod random;` to `lib.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::{epoch_permutation, reference_index};
    use crate::TrainingError;

    #[test]
    fn version_one_shuffle_and_reference_literals_are_stable() {
        assert_eq!(epoch_permutation(5, 7, 0).unwrap(), vec![0, 4, 2, 1, 3]);
        assert_eq!(epoch_permutation(5, 7, 1).unwrap(), vec![0, 3, 1, 4, 2]);
        assert_eq!(epoch_permutation(6, 42, 0).unwrap(), vec![2, 0, 4, 1, 3, 5]);
        assert_eq!(
            (0..5)
                .map(|position| reference_index(5, 7, 0, position).unwrap())
                .collect::<Vec<_>>(),
            vec![1, 2, 4, 2, 2],
        );
        assert_eq!(
            (0..5)
                .map(|position| reference_index(5, 7, 1, position).unwrap())
                .collect::<Vec<_>>(),
            vec![0, 4, 1, 0, 0],
        );
        assert_eq!(
            (0..6)
                .map(|position| reference_index(8, 42, 0, position).unwrap())
                .collect::<Vec<_>>(),
            vec![4, 1, 6, 6, 4, 5],
        );
    }

    #[test]
    fn reference_selection_preserves_python_self_reference_behavior() {
        assert_eq!(epoch_permutation(5, 0, 0).unwrap(), vec![2, 0, 3, 1, 4]);
        assert_eq!(reference_index(5, 0, 0, 1).unwrap(), 0);
    }

    #[test]
    fn impossible_permutation_lengths_fail_without_allocating() {
        assert!(matches!(
            epoch_permutation(u64::MAX, 7, 0),
            Err(TrainingError::DataLoaderOverflow { .. })
                | Err(TrainingError::PermutationAllocation { .. })
        ));
    }
}
```

- [ ] **Step 2: Run the private tests to verify RED**

Run:

```powershell
cargo test -p feathertalk-training random::tests -- --nocapture
```

Expected: compilation fails because `epoch_permutation` and `reference_index` do not exist.

- [ ] **Step 3: Implement the private version-one random algorithm**

First add the allocation error exercised by permutation construction:

```rust
#[error("unable to allocate epoch permutation for {samples} samples")]
PermutationAllocation {
    samples: u64,
    #[source]
    source: std::collections::TryReserveError,
},
```

Create `random.rs` with these exact constants and wrapping derivation; changing any literal is an algorithm-version change:

```rust
const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const MIX_MULTIPLIER_ONE: u64 = 0xBF58_476D_1CE4_E5B9;
const MIX_MULTIPLIER_TWO: u64 = 0x94D0_49BB_1331_11EB;
const POSITION_MULTIPLIER: u64 = 0xD1B5_4A32_D192_ED03;
const SHUFFLE_DOMAIN: u64 = 0x5348_5546_464C_455F;
const REFERENCE_DOMAIN: u64 = 0x5245_4645_5245_4E43;
```

Use this exact derivation:

```rust
fn derive_seed(base: u64, epoch: u64, position: u64, domain: u64) -> u64 {
    mix64(
        base
            ^ domain
            ^ epoch.wrapping_mul(SPLITMIX_GAMMA)
            ^ position.wrapping_mul(POSITION_MULTIPLIER),
    )
}
```

`SplitMix64::next_u64` first adds `SPLITMIX_GAMMA`, then applies the standard two multiply/xor rounds. `bounded(upper)` requires non-zero `upper`, computes `threshold = upper.wrapping_neg() % upper`, rejects values below the threshold and returns `value % upper`.

`epoch_permutation` must:

1. Convert sample count to `usize` with a structured overflow error.
2. Create an empty vector and call `try_reserve_exact` before pushing indices.
3. Fill `0..sample_count`.
4. Seed a shuffle stream with `derive_seed(seed, epoch, 0, SHUFFLE_DOMAIN)`.
5. Apply descending Fisher–Yates using `bounded(i + 1)`.

`reference_index` creates its own stream from `derive_seed(seed, epoch, position, REFERENCE_DOMAIN)` and samples below `frame_count`. Keep both functions `pub(super)` so only the sibling data module can call them.

Use this complete implementation shape:

```rust
use crate::TrainingError;

pub(super) fn epoch_permutation(
    sample_count: u64,
    seed: u64,
    epoch: u64,
) -> Result<Vec<u64>, TrainingError> {
    let length = usize::try_from(sample_count).map_err(|_| TrainingError::DataLoaderOverflow {
        operation: "converting sample count",
    })?;
    let mut permutation = Vec::new();
    permutation
        .try_reserve_exact(length)
        .map_err(|source| TrainingError::PermutationAllocation {
            samples: sample_count,
            source,
        })?;
    permutation.extend(0..sample_count);

    let mut random = SplitMix64::new(derive_seed(seed, epoch, 0, SHUFFLE_DOMAIN));
    for index in (1..length).rev() {
        let upper = u64::try_from(index)
            .map_err(|_| TrainingError::DataLoaderOverflow {
                operation: "converting shuffle index",
            })?
            .checked_add(1)
            .ok_or(TrainingError::DataLoaderOverflow {
                operation: "computing shuffle bound",
            })?;
        let swap_index = usize::try_from(random.bounded(upper)?).map_err(|_| {
            TrainingError::DataLoaderOverflow {
                operation: "converting bounded shuffle index",
            }
        })?;
        permutation.swap(index, swap_index);
    }
    Ok(permutation)
}

pub(super) fn reference_index(
    frame_count: u64,
    seed: u64,
    epoch: u64,
    position: u64,
) -> Result<u64, TrainingError> {
    let mut random = SplitMix64::new(derive_seed(seed, epoch, position, REFERENCE_DOMAIN));
    random.bounded(frame_count)
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(state: u64) -> Self {
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(SPLITMIX_GAMMA);
        mix64(self.state)
    }

    fn bounded(&mut self, upper: u64) -> Result<u64, TrainingError> {
        if upper == 0 {
            return Err(TrainingError::InvalidDataLoaderConfig(
                "random upper bound must be non-zero".into(),
            ));
        }
        let threshold = upper.wrapping_neg() % upper;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return Ok(value % upper);
            }
        }
    }
}

fn derive_seed(base: u64, epoch: u64, position: u64, domain: u64) -> u64 {
    mix64(
        base
            ^ domain
            ^ epoch.wrapping_mul(SPLITMIX_GAMMA)
            ^ position.wrapping_mul(POSITION_MULTIPLIER),
    )
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(MIX_MULTIPLIER_ONE);
    value = (value ^ (value >> 27)).wrapping_mul(MIX_MULTIPLIER_TWO);
    value ^ (value >> 31)
}
```

- [ ] **Step 4: Make the fixed-algorithm tests GREEN**

Run:

```powershell
cargo test -p feathertalk-training random::tests -- --nocapture
cargo test -p feathertalk-training --all-targets
```

Expected: all fixed literals pass and no existing VGG/loss regression fails.

- [ ] **Step 5: Commit Task 2**

```powershell
git add rust/crates/feathertalk-training/src/random.rs rust/crates/feathertalk-training/src/error.rs rust/crates/feathertalk-training/src/lib.rs
git commit -m "feat: add deterministic training sampling"
```

---

### Task 3: Add failure-atomic prepare and explicit batch commit

**Files:**

- Modify: `rust/crates/feathertalk-training/src/data.rs`
- Modify: `rust/crates/feathertalk-training/src/error.rs`
- Modify: `rust/crates/feathertalk-training/src/lib.rs`
- Modify: `rust/crates/feathertalk-training/tests/data_loader.rs`

**Interfaces:**

- Consumes public schema from Task 1 and fixed random functions from Task 2.
- Produces public `TrainingDataset` and `TrainingDataLoader::new`.
- Produces `TrainingDataLoader::state` and private position-to-sample construction.
- Produces opaque `PreparedBatch<T>` with `epoch`, `start_position`, `samples` and `items` getters.
- Produces `TrainingDataLoader::prepare_next_batch(&self)`.
- Produces `TrainingDataLoader::commit_batch(&mut self, PreparedBatch<D::Item>)`.
- Guarantees final-batch next-epoch allocation happens during prepare, before the caller performs model/optimizer work.

- [ ] **Step 1: Add RED tests for batch boundaries and safe commit**

Extend `tests/data_loader.rs` with a dataset that returns the plan unchanged:

```rust
#[derive(Debug)]
struct PlanDataset {
    frames: u64,
}

impl TrainingDataset for PlanDataset {
    type Item = TrainingSample;

    fn frame_count(&self) -> u64 {
        self.frames
    }

    fn load_sample(&self, sample: &TrainingSample) -> Result<Self::Item, TrainingError> {
        Ok(sample.clone())
    }
}
```

Add helpers that prepare and explicitly commit batches. Assert the complete epoch-zero literals for frames 5, batch size 2, seed 7 are:

```rust
let expected_epoch_zero = vec![
    TrainingSample::SingleFrame { target_index: 0, reference_index: 1 },
    TrainingSample::SingleFrame { target_index: 4, reference_index: 2 },
    TrainingSample::SingleFrame { target_index: 2, reference_index: 4 },
    TrainingSample::SingleFrame { target_index: 1, reference_index: 2 },
    TrainingSample::SingleFrame { target_index: 3, reference_index: 2 },
];
```

After the tail commit, collect epoch one and assert:

```rust
let expected_epoch_one = vec![
    TrainingSample::SingleFrame { target_index: 0, reference_index: 0 },
    TrainingSample::SingleFrame { target_index: 3, reference_index: 4 },
    TrainingSample::SingleFrame { target_index: 1, reference_index: 1 },
    TrainingSample::SingleFrame { target_index: 4, reference_index: 0 },
    TrainingSample::SingleFrame { target_index: 2, reference_index: 0 },
];
```

For Temporal with frames 8, batch size 4, seed 42 and stride 2, assert the complete first epoch:

```rust
let expected_temporal = vec![
    TrainingSample::TemporalPair { first_target_index: 2, second_target_index: 4, reference_index: 4 },
    TrainingSample::TemporalPair { first_target_index: 0, second_target_index: 2, reference_index: 1 },
    TrainingSample::TemporalPair { first_target_index: 4, second_target_index: 6, reference_index: 6 },
    TrainingSample::TemporalPair { first_target_index: 1, second_target_index: 3, reference_index: 6 },
    TrainingSample::TemporalPair { first_target_index: 3, second_target_index: 5, reference_index: 4 },
    TrainingSample::TemporalPair { first_target_index: 5, second_target_index: 7, reference_index: 5 },
];
```

Also assert the Python-compatible self-reference case using frames 5, batch size 5 and seed 0: epoch-zero sample position one is `SingleFrame { target_index: 0, reference_index: 0 }`.

Test these exact transitions:

```rust
let mut loader = TrainingDataLoader::new(
    PlanDataset { frames: 5 },
    DataLoaderConfig::single_frame(2, 7),
).unwrap();

let first = loader.prepare_next_batch().unwrap();
assert_eq!(first.epoch(), 0);
assert_eq!(first.start_position(), 0);
assert_eq!(first.samples().len(), 2);
assert_eq!(loader.state().next_position, 0);
loader.commit_batch(first).unwrap();
assert_eq!(loader.state().next_position, 2);

let second = loader.prepare_next_batch().unwrap();
loader.commit_batch(second).unwrap();
let tail = loader.prepare_next_batch().unwrap();
assert_eq!(tail.start_position(), 4);
assert_eq!(tail.samples().len(), 1);
loader.commit_batch(tail).unwrap();
assert_eq!((loader.state().epoch, loader.state().next_position), (1, 0));
```

Also add:

- batch size 8 over five samples returns one five-item batch;
- preparing twice before commit returns identical plans and leaves state unchanged;
- committing one of those batches makes the second stale;
- a batch prepared by another loader with identical serialized state is foreign;
- all stale/foreign errors leave state unchanged.

Use a `FailingDataset` with `Cell<usize>` that returns `TrainingError::InvalidInput("injected dataset failure".into())` on the second load. Assert `prepare_next_batch` returns that error and the loader state remains byte-for-byte equal to the pre-call clone.

- [ ] **Step 2: Run focused tests to verify RED**

Run:

```powershell
cargo test -p feathertalk-training --test data_loader commit -- --nocapture
cargo test -p feathertalk-training --test data_loader failure -- --nocapture
```

Expected: compilation fails because `TrainingDataset`, `TrainingDataLoader`, `PreparedBatch`, `prepare_next_batch` and `commit_batch` are absent.

- [ ] **Step 3: Implement loader construction and position-to-sample planning**

Add the two error variants exercised by the RED tests:

```rust
#[error("unable to allocate prepared batch buffers for {items} items")]
BatchAllocation {
    items: u64,
    #[source]
    source: std::collections::TryReserveError,
},
#[error("prepared batch is stale or belongs to another data loader")]
StalePreparedBatch,
```

In `data.rs`, define:

```rust
pub trait TrainingDataset {
    type Item;

    fn frame_count(&self) -> u64;

    fn load_sample(&self, sample: &TrainingSample) -> Result<Self::Item, TrainingError>;
}

pub struct TrainingDataLoader<D: TrainingDataset> {
    dataset: D,
    state: DataLoaderState,
    sample_count: u64,
    permutation: Vec<u64>,
    loader_id: u64,
}
```

Use a private `AtomicU64` starting at one and `fetch_update` with `checked_add` for an in-process loader identifier. If the counter is exhausted, return `DataLoaderOverflow { operation: "allocating loader identifier" }`. The identifier is only an opaque prepared-batch ownership token and never influences sampling or serialized state.

`TrainingDataLoader::new(dataset, config)` must:

1. Read `dataset.frame_count()` without loading a sample.
2. Validate the config and compute sample count.
3. Build canonical schema-one epoch-zero state.
4. Allocate the epoch-zero permutation.
5. Allocate the loader identifier.
6. Return the loader; no partially initialized value may escape on failure.

Add `state(&self) -> &DataLoaderState` and a private `sample_at_position(position)` that maps the current permutation value to `SingleFrame` or `TemporalPair`, checked-adds the temporal second target and derives the position-specific reference before dataset loading.

Export `PreparedBatch`, `TrainingDataset` and `TrainingDataLoader` explicitly from the crate root; keep all implementation fields private.

Use this implementation shape:

```rust
static NEXT_LOADER_ID: AtomicU64 = AtomicU64::new(1);

fn allocate_loader_id() -> Result<u64, TrainingError> {
    NEXT_LOADER_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .map_err(|_| TrainingError::DataLoaderOverflow {
            operation: "allocating loader identifier",
        })
}

impl<D: TrainingDataset> TrainingDataLoader<D> {
    pub fn new(dataset: D, config: DataLoaderConfig) -> Result<Self, TrainingError> {
        let frame_count = dataset.frame_count();
        let sample_count = config.sample_count(frame_count)?;
        let state = DataLoaderState {
            schema_version: DATA_LOADER_STATE_SCHEMA_VERSION,
            random_algorithm: RandomAlgorithm::Splitmix64FisherYatesV1,
            config,
            frame_count,
            epoch: 0,
            next_position: 0,
        };
        let permutation = epoch_permutation(sample_count, config.seed, 0)?;
        let loader_id = allocate_loader_id()?;
        Ok(Self {
            dataset,
            state,
            sample_count,
            permutation,
            loader_id,
        })
    }

    pub fn state(&self) -> &DataLoaderState {
        &self.state
    }

    fn sample_at_position(&self, position: u64) -> Result<TrainingSample, TrainingError> {
        let index = usize::try_from(position).map_err(|_| TrainingError::DataLoaderOverflow {
            operation: "converting sample position",
        })?;
        let target_index = *self.permutation.get(index).ok_or_else(|| {
            TrainingError::InvalidDataLoaderState(
                "sample position is outside the epoch permutation".into(),
            )
        })?;
        let reference_index = reference_index(
            self.state.frame_count,
            self.state.config.seed,
            self.state.epoch,
            position,
        )?;
        match self.state.config.sampling.kind {
            SamplingKind::SingleFrame => Ok(TrainingSample::SingleFrame {
                target_index,
                reference_index,
            }),
            SamplingKind::TemporalPair => {
                let second_target_index = target_index
                    .checked_add(self.state.config.sampling.temporal_stride)
                    .ok_or(TrainingError::DataLoaderOverflow {
                        operation: "computing temporal second target",
                    })?;
                if second_target_index >= self.state.frame_count {
                    return Err(TrainingError::InvalidDataLoaderState(
                        "temporal target is outside frame_count".into(),
                    ));
                }
                Ok(TrainingSample::TemporalPair {
                    first_target_index: target_index,
                    second_target_index,
                    reference_index,
                })
            }
        }
    }
}
```

- [ ] **Step 4: Implement opaque prepared batches**

Add private ownership/state fields and public read-only getters:

```rust
pub struct PreparedBatch<T> {
    loader_id: u64,
    epoch: u64,
    start_position: u64,
    end_position: u64,
    samples: Vec<TrainingSample>,
    items: Vec<T>,
    next_epoch: Option<u64>,
    next_permutation: Option<Vec<u64>>,
}

impl<T> PreparedBatch<T> {
    pub fn epoch(&self) -> u64 { self.epoch }
    pub fn start_position(&self) -> u64 { self.start_position }
    pub fn samples(&self) -> &[TrainingSample] { &self.samples }
    pub fn items(&self) -> &[T] { &self.items }
}
```

Do not implement `Clone` for `PreparedBatch<T>`.

- [ ] **Step 5: Implement failure-atomic prepare**

`prepare_next_batch(&self)` must:

1. Checked-subtract the remaining sample count, take `min(batch_size, remaining)`, then checked-add that count to `next_position` for `end`.
2. Allocate sample and item vectors with `try_reserve_exact`; map allocation failure to `BatchAllocation` using the requested item count.
3. Build every sample plan before calling the dataset.
4. If this is the final batch, checked-add the next epoch and build its complete permutation before loading any item; store both privately in the prepared batch.
5. Load every plan through `dataset.load_sample`.
6. Return only after all items succeed; never modify `state` or `permutation`.

The final-batch preallocation ensures every failure that could prevent a canonical epoch transition occurs before the caller executes model forward/backward/optimizer work.

Use this complete method:

```rust
pub fn prepare_next_batch(&self) -> Result<PreparedBatch<D::Item>, TrainingError> {
    let start_position = self.state.next_position;
    let remaining = self.sample_count.checked_sub(start_position).ok_or_else(|| {
        TrainingError::InvalidDataLoaderState(
            "next_position exceeds sample_count".into(),
        )
    })?;
    let batch_items = self.state.config.batch_size.min(remaining);
    let end_position = start_position.checked_add(batch_items).ok_or(
        TrainingError::DataLoaderOverflow {
            operation: "computing batch end",
        },
    )?;
    let batch_length = usize::try_from(batch_items).map_err(|_| {
        TrainingError::DataLoaderOverflow {
            operation: "converting batch length",
        }
    })?;

    let mut samples = Vec::new();
    samples
        .try_reserve_exact(batch_length)
        .map_err(|source| TrainingError::BatchAllocation {
            items: batch_items,
            source,
        })?;
    for position in start_position..end_position {
        samples.push(self.sample_at_position(position)?);
    }

    let (next_epoch, next_permutation) = if end_position == self.sample_count {
        let next_epoch = self.state.epoch.checked_add(1).ok_or(
            TrainingError::DataLoaderOverflow {
                operation: "advancing epoch",
            },
        )?;
        let next_permutation = epoch_permutation(
            self.sample_count,
            self.state.config.seed,
            next_epoch,
        )?;
        (Some(next_epoch), Some(next_permutation))
    } else {
        (None, None)
    };

    let mut items = Vec::new();
    items
        .try_reserve_exact(batch_length)
        .map_err(|source| TrainingError::BatchAllocation {
            items: batch_items,
            source,
        })?;
    for sample in &samples {
        items.push(self.dataset.load_sample(sample)?);
    }

    Ok(PreparedBatch {
        loader_id: self.loader_id,
        epoch: self.state.epoch,
        start_position,
        end_position,
        samples,
        items,
        next_epoch,
        next_permutation,
    })
}
```

- [ ] **Step 6: Implement strict commit**

`commit_batch` consumes the batch. Before mutation, verify:

- loader IDs match;
- batch epoch equals current state epoch;
- start equals current next position;
- end equals the checked expected end;
- sample/item lengths equal `end - start`;
- final batches contain `next_epoch == epoch + 1` and a next permutation of exact sample-count length;
- non-final batches contain neither a next epoch nor a next permutation.

Any mismatch returns `StalePreparedBatch` with no mutation. A non-final valid commit sets only `next_position = end`. A final valid commit takes the prebuilt permutation and then updates `epoch`, `next_position = 0` and `permutation` as one mutation section.

Implement the validation before any mutation:

```rust
pub fn commit_batch(
    &mut self,
    prepared: PreparedBatch<D::Item>,
) -> Result<(), TrainingError> {
    let remaining = self
        .sample_count
        .checked_sub(self.state.next_position)
        .ok_or(TrainingError::StalePreparedBatch)?;
    let expected_items = self.state.config.batch_size.min(remaining);
    let expected_end = self
        .state
        .next_position
        .checked_add(expected_items)
        .ok_or(TrainingError::StalePreparedBatch)?;
    let expected_length = usize::try_from(expected_items)
        .map_err(|_| TrainingError::StalePreparedBatch)?;
    let is_final = expected_end == self.sample_count;

    if prepared.loader_id != self.loader_id
        || prepared.epoch != self.state.epoch
        || prepared.start_position != self.state.next_position
        || prepared.end_position != expected_end
        || prepared.samples.len() != expected_length
        || prepared.items.len() != expected_length
    {
        return Err(TrainingError::StalePreparedBatch);
    }

    if is_final {
        let expected_next_epoch = self
            .state
            .epoch
            .checked_add(1)
            .ok_or(TrainingError::StalePreparedBatch)?;
        let Some(next_epoch) = prepared.next_epoch else {
            return Err(TrainingError::StalePreparedBatch);
        };
        let Some(next_permutation) = prepared.next_permutation else {
            return Err(TrainingError::StalePreparedBatch);
        };
        if next_epoch != expected_next_epoch
            || next_permutation.len() != self.permutation.len()
        {
            return Err(TrainingError::StalePreparedBatch);
        }
        self.state.epoch = next_epoch;
        self.state.next_position = 0;
        self.permutation = next_permutation;
    } else {
        if prepared.next_epoch.is_some() || prepared.next_permutation.is_some() {
            return Err(TrainingError::StalePreparedBatch);
        }
        self.state.next_position = expected_end;
    }
    Ok(())
}
```

- [ ] **Step 7: Verify GREEN and regressions**

Run:

```powershell
cargo test -p feathertalk-training --test data_loader
cargo test -p feathertalk-training --all-targets
cargo check -p feathertalk-training --all-targets
```

Expected: literal sampling, batch/tail, retry, stale/foreign and dataset failure tests pass; existing training tests remain green.

- [ ] **Step 8: Commit Task 3**

```powershell
git add rust/crates/feathertalk-training/src/data.rs rust/crates/feathertalk-training/src/error.rs rust/crates/feathertalk-training/src/lib.rs rust/crates/feathertalk-training/tests/data_loader.rs
git commit -m "feat: commit training batches explicitly"
```

---

### Task 4: Restore state exactly across JSON and epoch boundaries

**Files:**

- Modify: `rust/crates/feathertalk-training/src/data.rs`
- Create: `rust/crates/feathertalk-training/tests/data_loader_recovery.rs`

**Interfaces:**

- Consumes schema, fixed random algorithm and batch commit behavior from Tasks 1-3.
- Produces `TrainingDataLoader::restore(dataset, state)`.
- Produces exact uninterrupted/resumed equivalence for both sampling kinds.

- [ ] **Step 1: Write RED serde and recovery tests**

Create `tests/data_loader_recovery.rs` with its own small `PlanDataset`. First assert exact schema-one JSON shape and no permutation payload:

```rust
let state = DataLoaderState {
    schema_version: 1,
    random_algorithm: RandomAlgorithm::Splitmix64FisherYatesV1,
    config: DataLoaderConfig::single_frame(2, 7),
    frame_count: 5,
    epoch: 3,
    next_position: 2,
};
let json = serde_json::to_string(&state).unwrap();
assert_eq!(
    json,
    r#"{"schema_version":1,"random_algorithm":"splitmix64_fisher_yates_v1","config":{"batch_size":2,"seed":7,"sampling":{"kind":"single_frame","temporal_stride":0}},"frame_count":5,"epoch":3,"next_position":2}"#
);
assert!(!json.contains("permutation"));
assert_eq!(serde_json::from_str::<DataLoaderState>(&json).unwrap(), state);
```

Unknown root/config/sampling fields and `"future_rng"` must fail serde decoding.

Add `collect_committed(loader, batch_count)` that repeatedly prepares, clones only `samples()` into a result, and commits. For SingleFrame:

```rust
fn collect_committed<D>(
    loader: &mut TrainingDataLoader<D>,
    batch_count: usize,
) -> Vec<Vec<TrainingSample>>
where
    D: TrainingDataset,
{
    let mut batches = Vec::with_capacity(batch_count);
    for _ in 0..batch_count {
        let prepared = loader.prepare_next_batch().unwrap();
        batches.push(prepared.samples().to_vec());
        loader.commit_batch(prepared).unwrap();
    }
    batches
}
```

1. Create uninterrupted loader A with frames 5, batch size 2, seed 7.
2. Collect eight committed batches, crossing multiple epochs.
3. Create loader B with the same inputs, commit two batches, JSON round-trip `state()`.
4. Restore loader C from the decoded state and collect six batches.
5. Assert B's two-batch prefix plus C's six batches exactly equals A's eight batches.

Use this comparison shape so the checkpoint prefix is not discarded:

```rust
let mut uninterrupted = TrainingDataLoader::new(
    PlanDataset { frames: 5 },
    DataLoaderConfig::single_frame(2, 7),
).unwrap();
let expected = collect_committed(&mut uninterrupted, 8);

let mut interrupted = TrainingDataLoader::new(
    PlanDataset { frames: 5 },
    DataLoaderConfig::single_frame(2, 7),
).unwrap();
let mut actual = collect_committed(&mut interrupted, 2);
let json = serde_json::to_string(interrupted.state()).unwrap();
let state = serde_json::from_str::<DataLoaderState>(&json).unwrap();
let mut resumed = TrainingDataLoader::restore(PlanDataset { frames: 5 }, state).unwrap();
actual.extend(collect_committed(&mut resumed, 6));
assert_eq!(actual, expected);
```

Repeat for TemporalPair with frames 8, batch size 4, seed 42, stride 2, choosing a checkpoint after the first batch so the comparison crosses both a tail batch and the next epoch.

- [ ] **Step 2: Add strict restore rejection tests**

Use a counting dataset whose `load_sample` increments a `Cell`. Assert every failure occurs with count zero:

Use `Rc<Cell<usize>>` so the test can observe the counter after the dataset is moved into a loader:

```rust
struct CountingDataset {
    frames: u64,
    loads: Rc<Cell<usize>>,
}

impl TrainingDataset for CountingDataset {
    type Item = TrainingSample;

    fn frame_count(&self) -> u64 {
        self.frames
    }

    fn load_sample(&self, sample: &TrainingSample) -> Result<Self::Item, TrainingError> {
        self.loads.set(self.loads.get() + 1);
        Ok(sample.clone())
    }
}
```

- dataset frame count differs from state;
- schema version is two;
- `batch_size = 0`;
- SingleFrame has stride one;
- TemporalPair has stride zero;
- TemporalPair stride equals frame count;
- `next_position == sample_count`;
- `frame_count = u64::MAX` fails before loading: `DataLoaderOverflow` on narrower `usize` targets or `PermutationAllocation` when the target can represent the length but the vector cannot reserve it.

Add an epoch-overflow state at `epoch = u64::MAX`, with a valid cursor positioned at the final batch. Restore must succeed because the current epoch is representable; `prepare_next_batch` must return `DataLoaderOverflow { operation: "advancing epoch" }` before loading any item and leave the state unchanged.

Unknown random algorithm is rejected by serde before `restore`:

```rust
let json = valid_json.replace(
    "splitmix64_fisher_yates_v1",
    "future_rng",
);
assert!(serde_json::from_str::<DataLoaderState>(&json).is_err());
```

- [ ] **Step 3: Run recovery tests to verify RED**

Run:

```powershell
cargo test -p feathertalk-training --test data_loader_recovery -- --nocapture
```

Expected: compilation fails because `TrainingDataLoader::restore` does not exist.

- [ ] **Step 4: Implement strict restore**

Add:

```rust
pub fn restore(dataset: D, state: DataLoaderState) -> Result<Self, TrainingError>
```

The order is mandatory:

1. Read `dataset.frame_count()` without loading samples.
2. Call `state.validate(dataset_frame_count)`.
3. Compute sample count from the validated config.
4. Rebuild only `state.epoch`'s permutation from the fixed algorithm.
5. Allocate the loader ID.
6. Return the loader with the supplied canonical state unchanged.

Do not accept an optional fallback config, repair invalid state, reset to epoch zero, skip to the next epoch or consume any random values during restore.

Use this implementation:

```rust
pub fn restore(dataset: D, state: DataLoaderState) -> Result<Self, TrainingError> {
    let dataset_frame_count = dataset.frame_count();
    state.validate(dataset_frame_count)?;
    let sample_count = state.config.sample_count(state.frame_count)?;
    let permutation = epoch_permutation(
        sample_count,
        state.config.seed,
        state.epoch,
    )?;
    let loader_id = allocate_loader_id()?;
    Ok(Self {
        dataset,
        state,
        sample_count,
        permutation,
        loader_id,
    })
}
```

- [ ] **Step 5: Verify exact resumed equivalence**

Run:

```powershell
cargo test -p feathertalk-training --test data_loader_recovery -- --nocapture
cargo test -p feathertalk-training --all-targets
```

Expected: exact JSON, unknown-field/algorithm rejection, zero-load failure paths and both uninterrupted/resumed comparisons pass.

- [ ] **Step 6: Commit Task 4**

```powershell
git add rust/crates/feathertalk-training/src/data.rs rust/crates/feathertalk-training/tests/data_loader_recovery.rs
git commit -m "feat: restore deterministic training order"
```

---

### Task 5: Review, verify and integrate the completed slice

**Files:**

- Modify only files already listed if review finds a tested defect.
- Verify: `docs/superpowers/specs/2026-08-25-resumable-training-data-loader-design.md`
- Verify: `docs/superpowers/plans/2026-08-25-resumable-training-data-loader.md`

**Interfaces:**

- Consumes all Task 1-4 public APIs and guarantees.
- Produces a green merged `main` and removes only the completed worktree/branch.

- [ ] **Step 1: Run formatting and focused quality gates**

From the implementation worktree's `rust/` directory:

```powershell
cargo fmt --all
cargo test -p feathertalk-training --all-targets
cargo check -p feathertalk-training --all-targets
cargo clippy -p feathertalk-training --all-targets --all-features -- -D warnings
```

Expected: every command exits zero; no warning is accepted.

- [ ] **Step 2: Run full fresh workspace verification**

```powershell
cargo test --workspace --all-targets
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
```

From the worktree root:

```powershell
git diff --check
git status --short --branch
git status --short -- demo/kanghui_training_video_featherhubert_188_latest
```

Expected: all Rust commands exit zero, diff check is empty, branch changes contain only this slice, and the protected demo path has no worktree change.

- [ ] **Step 3: Perform inline specification/code review**

Compare every design section to the implementation and tests. Explicitly verify:

- exact version-one constants and literal sequences;
- no direct random/system entropy dependency;
- no full permutation in persisted JSON;
- reference may equal target;
- Temporal has one shared reference field;
- final partial batch is retained;
- prepare performs no state mutation;
- final-batch next-permutation/epoch failure occurs before model work;
- stale/foreign commit is failure-atomic;
- restore rejects invalid state before dataset loading;
- resumed output matches uninterrupted output across epoch boundaries;
- no unrelated crate, Python or protected demo change.

Fix every Critical or Important finding with a new failing test, verify RED, implement minimal GREEN, rerun focused gates and commit the fix explicitly.

- [ ] **Step 4: Commit final formatting/review changes if needed**

Stage only named changed files. If no files changed after the last feature commit, do not create an empty commit.

- [ ] **Step 5: Finish the branch**

Use `superpowers:finishing-a-development-branch`. The standing user choice is local fast-forward merge to `main` after green verification. Re-run:

```powershell
cargo test --workspace --all-targets
```

on merged `main`. Then remove only `.worktrees/resumable-training-data-loader`, prune worktree metadata and delete only the merged `resumable-training-data-loader` branch with `git branch -d`.

- [ ] **Step 6: Continue milestone three automatically**

After cleanup, re-read the migration design and current training APIs. Start the next independent specification for optimizer/checkpoint/epoch/global-step recovery using `superpowers:brainstorming`, followed by `superpowers:writing-plans`; do not stop at this slice unless an external blocker requires user authority.

## Plan Self-Review

- Spec coverage: Tasks 1-4 cover the schema, fixed RNG, literal shuffle/reference contracts, SingleFrame/Temporal semantics, tail batches, opaque prepared batches, explicit commit, failure atomicity, strict restore and cross-epoch equivalence. Task 5 covers full verification, protected-path checks, local integration and automatic milestone continuation.
- Placeholder scan: no forbidden placeholder marker, deferred implementation phrase, unspecified error handling or undefined neighboring API remains.
- Type consistency: `DataLoaderConfig`, `DataLoaderState`, `TrainingSample`, `TrainingDataset`, `TrainingDataLoader`, `PreparedBatch`, `RandomAlgorithm` and every method name are stable across tasks.
- Scope check: JPEG/crop/mask/feature materialization, optimizer/checkpoint publication, global step, training metrics and previews remain separate follow-up slices as required by the approved design.
