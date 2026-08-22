# PFLD Checkpoint Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strictly import the tracked epoch-335 PFLD PyTorch checkpoint into the existing Burn training-form graph, publish a verified temporary safetensors artifact and audit manifest, and leave the caller and destination unchanged on every failure.

**Architecture:** Keep the public PFLD contract and orchestration in a focused `pfld` module inside `feathertalk-weights`, with separate internal modules for envelope inspection, reviewed key remapping, and artifact verification/publication. Reuse an immutable hashed checkpoint snapshot for both legacy and PFLD imports, apply the checkpoint only to a cloned generic `ModuleSnapshot<B>` target, then reload and compare the staged safetensors before a same-filesystem directory rename and final caller replacement.

**Tech Stack:** Rust 1.92 edition 2024, Burn and `burn-store =0.21.0`, `serde`/`serde_json`, SHA-256 via `sha2`/`hex`, `tempfile`, the existing `feathertalk-models` CPU backend as a dev-only integration dependency, and Rust tests only.

## Global Constraints

- Accept only a root dictionary with exactly `{ epoch, pfld_backbone }` or `{ epoch, pfld_backbone, auxiliarynet }`.
- Require integer `epoch == 335`; never fall back to a direct state dict, `model`, or `state_dict`.
- Require `pfld_backbone` to be a non-empty tensor mapping; optional `auxiliarynet` is audited but never applied.
- Ignore exactly `localization.0.weight`, `localization.0.bias`, `localization.3.weight`, and `localization.3.bias`; a missing or additional `localization.*` key is an error.
- Ignore a `.num_batches_tracked` tensor only when its reviewed mapped BatchNorm parent also has mapped `running_mean` and `running_var` siblings.
- Count every backbone and auxiliary tensor toward `max_tensor_count` and `max_total_elements`, including tensors that will be ignored.
- Apply only `float32` non-ignored backbone tensors and reject duplicate mapped destinations, missing Burn tensors, unused non-ignored tensors, shape mismatches, and dtype mismatches.
- Keep `feathertalk-models` out of normal `feathertalk-weights` dependencies; it is permitted only under `[dev-dependencies]`.
- Write only `model.safetensors` and `manifest.json` beneath a caller-selected destination that did not previously exist.
- Stage under the destination parent, re-read and compare every safetensors tensor, re-read the manifest, revalidate hashes/counts, then rename the complete directory and replace the caller module.
- Keep the source checkpoint immutable; generated weights remain under `tempfile::TempDir` in tests and are never committed.
- Do not add Python execution, Python/Burn numerical parity, image preprocessing, SCRFD, AuxiliaryNet implementation, localization/STN execution, MobileOne reparameterization, a CLI, worker, or GPUI integration.
- Run every Rust command from `rust/` in `E:\workspace\github\FeatherTalk\.worktrees\pfld-checkpoint-import`.
- The user selected inline execution; use `superpowers:executing-plans` and do not dispatch subagents.

## File Map

- Modify `rust/crates/feathertalk-weights/Cargo.toml`: add `serde_json` normally and `feathertalk-models` only as a dev-dependency.
- Modify `rust/crates/feathertalk-weights/src/error.rs`: add stable PFLD envelope, epoch, ignored-set, artifact destination, artifact validation, and manifest errors.
- Create `rust/crates/feathertalk-weights/src/source.rs`: own shared safety defaults, immutable checkpoint copying, source hashing, and tensor element counting.
- Modify `rust/crates/feathertalk-weights/src/legacy.rs`: consume `source.rs` without changing generic legacy-import behavior.
- Create `rust/crates/feathertalk-weights/src/pfld/mod.rs`: define the public request/report/manifest contract and orchestrate the generic import.
- Create `rust/crates/feathertalk-weights/src/pfld/key_map.rs`: own only reviewed PFLD transformations, duplicate detection, BatchNorm-parent recognition, and the localization constants.
- Create `rust/crates/feathertalk-weights/src/pfld/envelope.rs`: validate `PickleValue` structure, inspect tensor metadata, classify ignored tensors, and enforce global safety limits.
- Create `rust/crates/feathertalk-weights/src/pfld/artifact.rs`: stage, save, hash, reload, compare, serialize, revalidate, and atomically publish artifacts.
- Modify `rust/crates/feathertalk-weights/src/lib.rs`: register internal modules and export the complete PFLD public API.
- Create `rust/crates/feathertalk-weights/tests/pfld_contract.rs`: verify public defaults and manifest serialization.
- Create `rust/crates/feathertalk-weights/tests/pfld_import_failures.rs`: verify destination rejection and model/destination failure atomicity.
- Create `rust/crates/feathertalk-weights/tests/pfld_checkpoint.rs`: exercise the tracked real checkpoint, safetensors round trip, manifest, and zero-input shape smoke test.
- Modify `rust/Cargo.lock` only through Cargo after dependency changes.

---

### Task 1: Define the public contract and share immutable source snapshots

**Files:**
- Modify: `rust/crates/feathertalk-weights/Cargo.toml`
- Modify: `rust/crates/feathertalk-weights/src/error.rs`
- Create: `rust/crates/feathertalk-weights/src/source.rs`
- Modify: `rust/crates/feathertalk-weights/src/legacy.rs`
- Create: `rust/crates/feathertalk-weights/src/pfld/mod.rs`
- Modify: `rust/crates/feathertalk-weights/src/lib.rs`
- Create: `rust/crates/feathertalk-weights/tests/pfld_contract.rs`
- Modify: `rust/Cargo.lock`

**Interfaces:**
- Produces `PFLD_CHECKPOINT_EPOCH`, `PFLD_ARCHITECTURE_VERSION`, `PfldImportRequest`, `TensorAudit`, `TensorSummary`, `PfldSourceManifest`, `PfldModelArtifact`, `PfldIgnoredTensors`, `PfldImportManifest`, and `PfldImportReport`.
- Produces crate-private `SnapshotFile::copy_from`, `SnapshotFile::path`, `SnapshotFile::sha256`, `sha256_file`, `tensor_elements`, and the three existing safety defaults.
- Preserves `LegacyImportRequest::default()` and all existing `import_into` behavior.

- [ ] **Step 1: Add failing public-contract tests**

Create `tests/pfld_contract.rs` with these exact cases:

```rust
use std::path::PathBuf;

use feathertalk_weights::{
    PFLD_ARCHITECTURE_VERSION, PFLD_CHECKPOINT_EPOCH, PfldIgnoredTensors,
    PfldImportManifest, PfldImportRequest, PfldModelArtifact, PfldSourceManifest, TensorAudit,
    TensorSummary,
};

#[test]
fn pfld_request_defaults_match_existing_import_limits() {
    let request = PfldImportRequest::default();
    assert_eq!(request.checkpoint, PathBuf::new());
    assert_eq!(request.destination_dir, PathBuf::new());
    assert_eq!(request.max_file_bytes, 4 * 1024 * 1024 * 1024);
    assert_eq!(request.max_tensor_count, 10_000);
    assert_eq!(request.max_total_elements, 2_000_000_000);
    assert_eq!(PFLD_CHECKPOINT_EPOCH, 335);
    assert_eq!(PFLD_ARCHITECTURE_VERSION, "burn-pfld-structure-v1");
}

#[test]
fn manifest_round_trips_without_absolute_paths_or_timestamps() {
    let manifest = PfldImportManifest {
        schema_version: 1,
        model_type: "pfld_ghost_one".to_owned(),
        architecture_version: PFLD_ARCHITECTURE_VERSION.to_owned(),
        source: PfldSourceManifest {
            file_name: "checkpoint_epoch_335.pth.tar".to_owned(),
            sha256: "a".repeat(64),
        },
        epoch: PFLD_CHECKPOINT_EPOCH,
        backbone: TensorSummary {
            tensor_count: 2_090,
            total_elements: 913_663,
        },
        model: PfldModelArtifact {
            format: "safetensors".to_owned(),
            file_name: "model.safetensors".to_owned(),
            sha256: "b".repeat(64),
            tensor_count: 1_735,
            total_elements: 910_902,
        },
        ignored: PfldIgnoredTensors {
            batch_norm_counters: TensorAudit {
                tensor_count: 1,
                total_elements: 1,
                keys: vec!["conv1.rbr_conv.0.bn.num_batches_tracked".to_owned()],
            },
            localization: TensorAudit {
                tensor_count: 4,
                total_elements: 2_410,
                keys: vec![
                    "localization.0.bias".to_owned(),
                    "localization.0.weight".to_owned(),
                    "localization.3.bias".to_owned(),
                    "localization.3.weight".to_owned(),
                ],
            },
            auxiliarynet: None,
        },
    };

    let json = serde_json::to_string_pretty(&manifest).unwrap();
    assert!(!json.contains(r#""destination_dir""#));
    assert!(!json.contains(r#""timestamp""#));
    assert_eq!(
        serde_json::from_str::<PfldImportManifest>(&json).unwrap(),
        manifest
    );
}
```

- [ ] **Step 2: Run the contract test and observe the red result**

Run: `cargo test -p feathertalk-weights --test pfld_contract`

Expected: compilation fails because the PFLD types and constants are not exported.

- [ ] **Step 3: Add the dependency and exact public value types**

Add `serde_json.workspace = true` under normal dependencies. In `src/pfld/mod.rs` define:

```rust
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::source::{
    DEFAULT_MAX_FILE_BYTES, DEFAULT_MAX_TENSOR_COUNT, DEFAULT_MAX_TOTAL_ELEMENTS,
};

pub const PFLD_CHECKPOINT_EPOCH: u64 = 335;
pub const PFLD_ARCHITECTURE_VERSION: &str = "burn-pfld-structure-v1";

#[derive(Debug, Clone)]
pub struct PfldImportRequest {
    pub checkpoint: PathBuf,
    pub destination_dir: PathBuf,
    pub max_file_bytes: u64,
    pub max_tensor_count: usize,
    pub max_total_elements: u64,
}

impl Default for PfldImportRequest {
    fn default() -> Self {
        Self {
            checkpoint: PathBuf::new(),
            destination_dir: PathBuf::new(),
            max_file_bytes: DEFAULT_MAX_FILE_BYTES,
            max_tensor_count: DEFAULT_MAX_TENSOR_COUNT,
            max_total_elements: DEFAULT_MAX_TOTAL_ELEMENTS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorAudit {
    pub tensor_count: usize,
    pub total_elements: u64,
    pub keys: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorSummary {
    pub tensor_count: usize,
    pub total_elements: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PfldSourceManifest {
    pub file_name: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PfldModelArtifact {
    pub format: String,
    pub file_name: String,
    pub sha256: String,
    pub tensor_count: usize,
    pub total_elements: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PfldIgnoredTensors {
    pub batch_norm_counters: TensorAudit,
    pub localization: TensorAudit,
    pub auxiliarynet: Option<TensorAudit>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PfldImportManifest {
    pub schema_version: u32,
    pub model_type: String,
    pub architecture_version: String,
    pub source: PfldSourceManifest,
    pub epoch: u64,
    pub backbone: TensorSummary,
    pub model: PfldModelArtifact,
    pub ignored: PfldIgnoredTensors,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PfldImportReport {
    pub destination_dir: PathBuf,
    pub manifest: PfldImportManifest,
    pub applied: Vec<String>,
}
```

Register `mod pfld; mod source;` in `src/lib.rs` and re-export every public item listed under Interfaces. Do not export internal envelope, key-map, artifact, or snapshot helpers.

- [ ] **Step 4: Add stable errors before any importer uses them**

Append these variants to `WeightImportError`:

```rust
#[error("invalid PFLD checkpoint envelope: {0}")]
InvalidPfldEnvelope(String),
#[error("invalid PFLD epoch: expected {expected}, got {actual}")]
InvalidPfldEpoch { expected: u64, actual: String },
#[error("invalid PFLD ignored tensor set: {0}")]
InvalidPfldIgnoredSet(String),
#[error("artifact destination already exists: {}", .0.display())]
ArtifactDestinationExists(std::path::PathBuf),
#[error("artifact validation failed: {0}")]
ArtifactValidation(String),
#[error("manifest error: {0}")]
Manifest(String),
```

- [ ] **Step 5: Extract the immutable snapshot implementation without behavior changes**

Move the existing `SnapshotFile` implementation and safety constants from `legacy.rs` into `source.rs`. Keep create-new copying, the 64 KiB buffer, pre-copy metadata limit, during-copy limit, `sync_all`, lowercase SHA-256, and the owning `TempDir`/file handle. Add these exact crate-private signatures:

```rust
pub(crate) const DEFAULT_MAX_FILE_BYTES: u64 = 4 * 1024 * 1024 * 1024;
pub(crate) const DEFAULT_MAX_TENSOR_COUNT: usize = 10_000;
pub(crate) const DEFAULT_MAX_TOTAL_ELEMENTS: u64 = 2_000_000_000;

pub(crate) struct SnapshotFile {
    _directory: tempfile::TempDir,
    _handle: std::fs::File,
    path: std::path::PathBuf,
    sha256: String,
}

impl SnapshotFile {
    pub(crate) fn copy_from(
        path: &std::path::Path,
        max_file_bytes: u64,
    ) -> Result<Self, WeightImportError>;

    pub(crate) fn path(&self) -> &std::path::Path;

    pub(crate) fn sha256(&self) -> &str;
}

pub(crate) fn sha256_file(path: &std::path::Path) -> Result<String, WeightImportError>;

pub(crate) fn tensor_elements(
    snapshot: &burn_store::TensorSnapshot,
) -> Result<u64, WeightImportError>;
```

Implement `sha256_file` with the same 64 KiB streaming loop rather than `fs::read`. Change `legacy.rs` imports and calls to use these helpers; do not change top-level fallback behavior or which legacy tensors count toward its existing report.

- [ ] **Step 6: Run contract and legacy regression tests**

Run:

```powershell
cargo test -p feathertalk-weights --test pfld_contract
cargo test -p feathertalk-weights --test legacy_import
```

Expected: the two PFLD contract tests and all existing legacy-import tests pass.

- [ ] **Step 7: Commit the public contract**

```powershell
git add rust/Cargo.toml rust/Cargo.lock rust/crates/feathertalk-weights
git commit -m "feat: define PFLD import contract"
```

---

### Task 2: Implement the reviewed PFLD key map and ignored-key recognition

**Files:**
- Modify: `rust/crates/feathertalk-weights/src/pfld/mod.rs`
- Create: `rust/crates/feathertalk-weights/src/pfld/key_map.rs`

**Interfaces:**
- Produces `pfld_remapper() -> KeyRemapper` and `map_pfld_key(&str) -> String` with identical transformations.
- Produces `reject_duplicate_destinations` and `is_valid_batch_norm_counter`.
- Produces the sorted `LOCALIZATION_KEYS` constant consumed by envelope auditing.

- [ ] **Step 1: Add focused failing unit tests in `key_map.rs`**

Add tests for all representative paths, transformation boundaries, collisions, and BatchNorm-counter siblings:

```rust
#[test]
fn reviewed_pfld_paths_map_exactly() {
    let cases = [
        (
            "conv1.rbr_conv.0.conv.weight",
            "conv1.branches.0.0.weight",
        ),
        (
            "conv1.rbr_conv.0.bn.weight",
            "conv1.branches.0.1.gamma",
        ),
        (
            "conv3_1.ghost_conv.0.primary_conv.rbr_conv.0.bn.weight",
            "conv3_1.ghost.primary.branches.0.1.gamma",
        ),
        (
            "conv3_1.ghost_conv.1.rbr_conv.0.conv.weight",
            "conv3_1.depthwise.branches.0.0.weight",
        ),
        (
            "conv3_1.ghost_conv.2.cheap_operation.rbr_scale.bn.bias",
            "conv3_1.linear.cheap.scale.1.beta",
        ),
        ("conv8.0.weight", "conv8.weight"),
        ("conv_out.bias", "head.bias"),
    ];

    for (source, expected) in cases {
        assert_eq!(map_pfld_key(source), expected);
    }
}

#[test]
fn unapproved_near_matches_are_unchanged() {
    for key in [
        "prefix.conv8.0.weight",
        "conv80.weight",
        "conv1.rbr_conv.named.conv.weight",
        "conv1.rbr_scale.extra.bn.weight",
        "conv3_1.ghost_conv.3.primary_conv.weight",
        "conv3_1.primary_convolution.weight",
        "conv1.rbr_skip_extra.weight",
    ] {
        assert_eq!(map_pfld_key(key), key);
    }
}

#[test]
fn colliding_pfld_destinations_are_rejected() {
    let error = reject_duplicate_destinations([
        "conv8.0.weight".to_owned(),
        "conv8.weight".to_owned(),
    ])
    .unwrap_err();
    assert!(matches!(
        error,
        WeightImportError::DuplicateKey(key) if key == "conv8.weight"
    ));
}

#[test]
fn batch_norm_counter_requires_reviewed_sibling_buffers() {
    let keys = [
        "conv1.rbr_conv.0.bn.running_mean",
        "conv1.rbr_conv.0.bn.running_var",
        "conv1.rbr_conv.0.bn.num_batches_tracked",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<BTreeSet<_>>();
    assert!(is_valid_batch_norm_counter(
        "conv1.rbr_conv.0.bn.num_batches_tracked",
        &keys
    ));

    let missing_var = [
        "conv1.rbr_conv.0.bn.running_mean",
        "conv1.rbr_conv.0.bn.num_batches_tracked",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<BTreeSet<_>>();
    assert!(!is_valid_batch_norm_counter(
        "conv1.rbr_conv.0.bn.num_batches_tracked",
        &missing_var
    ));

    let arbitrary = [
        "unknown.running_mean",
        "unknown.running_var",
        "unknown.num_batches_tracked",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<BTreeSet<_>>();
    assert!(!is_valid_batch_norm_counter(
        "unknown.num_batches_tracked",
        &arbitrary
    ));

    let burn_looking_counterfeit = [
        "unknown.branches.0.1.running_mean",
        "unknown.branches.0.1.running_var",
        "unknown.branches.0.1.num_batches_tracked",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<BTreeSet<_>>();
    assert!(!is_valid_batch_norm_counter(
        "unknown.branches.0.1.num_batches_tracked",
        &burn_looking_counterfeit
    ));
}
```

- [ ] **Step 2: Run the key-map unit tests and observe the red result**

Run: `cargo test -p feathertalk-weights pfld::key_map::tests`

Expected: compilation fails because `key_map` and its functions do not exist.

- [ ] **Step 3: Implement one remapper used by both preflight and `PytorchStore`**

Define `LOCALIZATION_KEYS` in lexicographic order:

```rust
pub(super) const LOCALIZATION_KEYS: [&str; 4] = [
    "localization.0.bias",
    "localization.0.weight",
    "localization.3.bias",
    "localization.3.weight",
];
```

Build `pfld_remapper` with these ordered, segment-anchored patterns:

```rust
pub(super) fn pfld_remapper() -> KeyRemapper {
    KeyRemapper::new()
        .add_pattern(
            r"(^|\.)rbr_conv\.([0-9]+)\.conv\.",
            "${1}branches.${2}.0.",
        )
        .expect("reviewed literal regex")
        .add_pattern(
            r"(^|\.)rbr_conv\.([0-9]+)\.bn\.",
            "${1}branches.${2}.1.",
        )
        .expect("reviewed literal regex")
        .add_pattern(r"(^|\.)rbr_scale\.conv\.", "${1}scale.0.")
        .expect("reviewed literal regex")
        .add_pattern(r"(^|\.)rbr_scale\.bn\.", "${1}scale.1.")
        .expect("reviewed literal regex")
        .add_pattern(r"(^|\.)rbr_skip\.", "${1}skip.")
        .expect("reviewed literal regex")
        .add_pattern(r"(^|\.)ghost_conv\.0\.", "${1}ghost.")
        .expect("reviewed literal regex")
        .add_pattern(r"(^|\.)ghost_conv\.1\.", "${1}depthwise.")
        .expect("reviewed literal regex")
        .add_pattern(r"(^|\.)ghost_conv\.2\.", "${1}linear.")
        .expect("reviewed literal regex")
        .add_pattern(r"(^|\.)primary_conv\.", "${1}primary.")
        .expect("reviewed literal regex")
        .add_pattern(r"(^|\.)cheap_operation\.", "${1}cheap.")
        .expect("reviewed literal regex")
        .add_pattern(r"^conv8\.0\.", "conv8.")
        .expect("reviewed literal regex")
        .add_pattern(r"^conv_out\.", "head.")
        .expect("reviewed literal regex")
        .add_pattern(
            r"^(.+\.(?:branches\.[0-9]+\.1|scale\.1|skip))\.weight$",
            "${1}.gamma",
        )
        .expect("reviewed literal regex")
        .add_pattern(
            r"^(.+\.(?:branches\.[0-9]+\.1|scale\.1|skip))\.bias$",
            "${1}.beta",
        )
        .expect("reviewed literal regex")
}
```

Implement `map_pfld_key` by applying `pfld_remapper().patterns` in order, matching the established generic key-map implementation. Implement duplicate detection with a `BTreeSet<String>` so the first duplicated destination returns `WeightImportError::DuplicateKey`.

- [ ] **Step 4: Implement strict BatchNorm-parent recognition**

Require the source parent itself to end in exactly one of `rbr_conv.<decimal-index>.bn`, `rbr_scale.bn`, or `rbr_skip`. Map the counter and sibling source keys only after this source-form check, then require the mapped parent to end in one of:

```text
branches.<decimal-index>.1
scale.1
skip
```

Use this exact signature and sibling check:

```rust
pub(super) fn is_valid_batch_norm_counter(
    source_key: &str,
    source_keys: &BTreeSet<String>,
) -> bool {
    let Some(source_parent) = source_key.strip_suffix(".num_batches_tracked") else {
        return false;
    };
    if !is_reviewed_source_batch_norm_parent(source_parent) {
        return false;
    }
    let running_mean = format!("{source_parent}.running_mean");
    let running_var = format!("{source_parent}.running_var");
    if !source_keys.contains(&running_mean) || !source_keys.contains(&running_var) {
        return false;
    }
    let mapped_mean = map_pfld_key(&running_mean);
    let mapped_var = map_pfld_key(&running_var);
    let Some(mapped_parent) = mapped_mean.strip_suffix(".running_mean") else {
        return false;
    };
    is_mapped_batch_norm_parent(mapped_parent)
        && mapped_var == format!("{mapped_parent}.running_var")
}
```

`is_reviewed_source_batch_norm_parent` and `is_mapped_batch_norm_parent` must split on `.`, require a decimal branch index for `rbr_conv`/`branches`, and reject arbitrary or already-Burn-shaped parents that merely use the counter suffix.

- [ ] **Step 5: Run the focused unit tests and commit**

Run:

```powershell
cargo test -p feathertalk-weights pfld::key_map::tests
cargo test -p feathertalk-weights --test legacy_import
```

Expected: all key-map and legacy regression tests pass.

```powershell
git add rust/crates/feathertalk-weights/src/pfld
git commit -m "feat: map PFLD checkpoint keys"
```

---

### Task 3: Validate the fixed envelope and build complete tensor audits

**Files:**
- Modify: `rust/crates/feathertalk-weights/src/pfld/mod.rs`
- Create: `rust/crates/feathertalk-weights/src/pfld/envelope.rs`

**Interfaces:**
- Produces `validate_envelope(PickleValue) -> PfldEnvelope`.
- Produces `inspect_checkpoint(path, envelope, request) -> PfldInspection`.
- `PfldInspection` contains the backbone summary, applied summary, ignored audits, sorted expected applied destinations, and sorted expected unused mapped paths.
- Consumes `SnapshotFile::path()` later; it never reads the mutable caller path.

- [ ] **Step 1: Add failing envelope tests over synthetic `PickleValue`**

Inside `envelope.rs` add a helper that creates a flat tensor mapping with `PickleValue::None` values, then cover both accepted key sets and every rejected root condition:

```rust
fn tensor_dict(keys: &[&str]) -> PickleValue {
    PickleValue::Dict(
        keys.iter()
            .map(|key| ((*key).to_owned(), PickleValue::None))
            .collect(),
    )
}

fn root(entries: impl IntoIterator<Item = (&'static str, PickleValue)>) -> PickleValue {
    PickleValue::Dict(
        entries
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value))
            .collect(),
    )
}

#[test]
fn accepts_only_the_two_approved_envelopes() {
    let minimal = root([
        ("epoch", PickleValue::Int(335)),
        ("pfld_backbone", tensor_dict(&["conv8.0.weight"])),
    ]);
    assert_eq!(
        validate_envelope(minimal).unwrap(),
        PfldEnvelope {
            has_auxiliarynet: false
        }
    );

    let with_auxiliary = root([
        ("epoch", PickleValue::Int(335)),
        ("pfld_backbone", tensor_dict(&["conv8.0.weight"])),
        ("auxiliarynet", tensor_dict(&["conv1.weight"])),
    ]);
    assert_eq!(
        validate_envelope(with_auxiliary).unwrap(),
        PfldEnvelope {
            has_auxiliarynet: true
        }
    );
}

#[test]
fn rejects_bad_roots_keys_epoch_and_tensor_mappings() {
    assert!(matches!(
        validate_envelope(PickleValue::List(Vec::new())),
        Err(WeightImportError::InvalidPfldEnvelope(_))
    ));
    assert!(matches!(
        validate_envelope(root([(
            "pfld_backbone",
            tensor_dict(&["conv8.0.weight"])
        )])),
        Err(WeightImportError::InvalidPfldEnvelope(_))
    ));
    assert!(matches!(
        validate_envelope(root([
            ("epoch", PickleValue::String("335".to_owned())),
            ("pfld_backbone", tensor_dict(&["conv8.0.weight"])),
        ])),
        Err(WeightImportError::InvalidPfldEpoch { expected: 335, .. })
    ));
    assert!(matches!(
        validate_envelope(root([
            ("epoch", PickleValue::Int(334)),
            ("pfld_backbone", tensor_dict(&["conv8.0.weight"])),
        ])),
        Err(WeightImportError::InvalidPfldEpoch {
            expected: 335,
            actual
        }) if actual == "334"
    ));
    assert!(matches!(
        validate_envelope(root([
            ("epoch", PickleValue::Int(335)),
            ("pfld_backbone", tensor_dict(&[])),
        ])),
        Err(WeightImportError::InvalidPfldEnvelope(_))
    ));
    assert!(matches!(
        validate_envelope(root([
            ("epoch", PickleValue::Int(335)),
            ("pfld_backbone", tensor_dict(&["conv8.0.weight"])),
            ("state_dict", tensor_dict(&["conv8.0.weight"])),
        ])),
        Err(WeightImportError::InvalidPfldEnvelope(_))
    ));
}
```

- [ ] **Step 2: Add failing pure-audit tests**

Define the test-only constructor for internal `TensorFact { key, dtype, elements }`:

```rust
fn fact(key: &str, dtype: DType, elements: u64) -> TensorFact {
    TensorFact {
        key: key.to_owned(),
        dtype,
        elements,
    }
}
```

Then assert:

```rust
#[test]
fn audit_requires_exact_localization_and_counts_auxiliary_tensors() {
    let backbone = vec![
        fact("conv8.0.weight", DType::F32, 64),
        fact("localization.0.weight", DType::F32, 2_304),
        fact("localization.0.bias", DType::F32, 32),
        fact("localization.3.weight", DType::F32, 64),
        fact("localization.3.bias", DType::F32, 10),
    ];
    let auxiliary = vec![fact("conv1.weight", DType::F32, 7)];
    let inspection = audit_tensor_facts(
        backbone,
        Some(auxiliary),
        SafetyLimits {
            max_tensor_count: 6,
            max_total_elements: 2_481,
        },
    )
    .unwrap();

    assert_eq!(
        inspection.backbone,
        TensorSummary {
            tensor_count: 5,
            total_elements: 2_474
        }
    );
    assert_eq!(
        inspection.applied,
        TensorSummary {
            tensor_count: 1,
            total_elements: 64
        }
    );
    assert_eq!(
        inspection.ignored.localization.keys,
        LOCALIZATION_KEYS.map(str::to_owned)
    );
    assert_eq!(
        inspection.ignored.auxiliarynet.unwrap(),
        TensorAudit {
            tensor_count: 1,
            total_elements: 7,
            keys: vec!["auxiliarynet.conv1.weight".to_owned()],
        }
    );
}

#[test]
fn audit_rejects_partial_or_extra_localization_sets() {
    let partial = vec![
        fact("localization.0.weight", DType::F32, 1),
        fact("localization.0.bias", DType::F32, 1),
        fact("localization.3.weight", DType::F32, 1),
    ];
    assert!(matches!(
        audit_tensor_facts(partial, None, SafetyLimits::unbounded()),
        Err(WeightImportError::InvalidPfldIgnoredSet(_))
    ));

    let extra = vec![
        fact("localization.0.weight", DType::F32, 1),
        fact("localization.0.bias", DType::F32, 1),
        fact("localization.3.weight", DType::F32, 1),
        fact("localization.3.bias", DType::F32, 1),
        fact("localization.6.weight", DType::F32, 1),
    ];
    assert!(matches!(
        audit_tensor_facts(extra, None, SafetyLimits::unbounded()),
        Err(WeightImportError::InvalidPfldIgnoredSet(_))
    ));
}

#[test]
fn global_limits_include_ignored_backbone_and_auxiliary_tensors() {
    let backbone = LOCALIZATION_KEYS
        .into_iter()
        .map(|key| fact(key, DType::F32, 1))
        .collect::<Vec<_>>();
    let auxiliary = vec![fact("weight", DType::F32, 1)];
    assert!(matches!(
        audit_tensor_facts(
            backbone.clone(),
            Some(auxiliary.clone()),
            SafetyLimits {
                max_tensor_count: 4,
                max_total_elements: 10,
            }
        ),
        Err(WeightImportError::UnsafeLimit(_))
    ));
    assert!(matches!(
        audit_tensor_facts(
            backbone,
            Some(auxiliary),
            SafetyLimits {
                max_tensor_count: 10,
                max_total_elements: 4,
            }
        ),
        Err(WeightImportError::UnsafeLimit(_))
    ));
}

#[test]
fn only_non_ignored_float32_tensors_can_be_applied() {
    let mut backbone = LOCALIZATION_KEYS
        .into_iter()
        .map(|key| fact(key, DType::F32, 1))
        .collect::<Vec<_>>();
    backbone.push(fact("conv8.0.weight", DType::F64, 64));
    assert!(matches!(
        audit_tensor_facts(backbone, None, SafetyLimits::unbounded()),
        Err(WeightImportError::DTypeMismatch(path)) if path == "conv8.weight"
    ));
}
```

- [ ] **Step 3: Run envelope tests and observe the red result**

Run: `cargo test -p feathertalk-weights pfld::envelope::tests`

Expected: compilation fails because envelope validation and tensor auditing are absent.

- [ ] **Step 4: Implement fixed-envelope validation**

Import `burn_store::pytorch::reader::PickleValue`. `validate_envelope` must:

1. Match only `PickleValue::Dict` at the root.
2. Sort root keys and compare them to `["epoch", "pfld_backbone"]` or `["auxiliarynet", "epoch", "pfld_backbone"]`.
3. Match `epoch` as `PickleValue::Int(335)`; use `actual.to_string()` for integers and `format!("{actual:?}")` for other variants.
4. Require `pfld_backbone` to be a non-empty `Dict` whose values are all `PickleValue::None`.
5. Require optional `auxiliarynet` to be a `Dict` whose values are all `PickleValue::None`.
6. Return only `PfldEnvelope { has_auxiliarynet }`; do not carry or trust mutable source data after validation.

Use `InvalidPfldEnvelope` for structure/key/mapping errors and `InvalidPfldEpoch` only for epoch type/value errors.

- [ ] **Step 5: Implement metadata inspection and pure classification**

Use these internal types:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PfldEnvelope {
    pub(super) has_auxiliarynet: bool,
}

#[derive(Debug, Clone, Copy)]
struct SafetyLimits {
    max_tensor_count: usize,
    max_total_elements: u64,
}

#[cfg(test)]
impl SafetyLimits {
    fn unbounded() -> Self {
        Self {
            max_tensor_count: usize::MAX,
            max_total_elements: u64::MAX,
        }
    }
}

struct TensorFact {
    key: String,
    dtype: burn::tensor::DType,
    elements: u64,
}

#[derive(Debug)]
pub(super) struct PfldInspection {
    pub(super) backbone: TensorSummary,
    pub(super) applied: TensorSummary,
    pub(super) ignored: PfldIgnoredTensors,
    pub(super) expected_applied: std::collections::BTreeSet<String>,
    pub(super) expected_unused: std::collections::BTreeSet<String>,
}
```

`inspect_checkpoint` opens `PytorchReader::with_top_level_key(path, "pfld_backbone")`, converts every snapshot to a `TensorFact` with `tensor_elements`, and does the same for `auxiliarynet` only when the validated envelope says it exists. It then calls one pure `audit_tensor_facts` implementation.

`audit_tensor_facts` must:

- reject a count or element overflow with `UnsafeLimit`;
- enforce limits over backbone plus auxiliary facts before classifying ignored tensors;
- collect all `localization.*` source keys and compare their sorted vector exactly to `LOCALIZATION_KEYS`;
- classify reviewed BatchNorm counters with `is_valid_batch_norm_counter`;
- prefix auxiliary audit keys with `auxiliarynet.` and sort them;
- require `DType::F32` only for facts classified as applied;
- map every applied key with `map_pfld_key` and call `reject_duplicate_destinations` before returning;
- produce `expected_unused` from mapped BatchNorm-counter keys plus unchanged localization keys;
- sort every public key list lexicographically.

- [ ] **Step 6: Run the focused tests and commit**

Run:

```powershell
cargo test -p feathertalk-weights pfld::envelope::tests
cargo test -p feathertalk-weights --all-targets
```

Expected: all PFLD unit tests and the existing 18+ legacy targets pass.

```powershell
git add rust/crates/feathertalk-weights/src
git commit -m "feat: inspect PFLD checkpoint envelope"
```

---

### Task 4: Load a strict candidate without mutating the caller

**Files:**
- Modify: `rust/crates/feathertalk-weights/Cargo.toml`
- Modify: `rust/crates/feathertalk-weights/src/pfld/mod.rs`
- Modify: `rust/Cargo.lock`

**Interfaces:**
- Produces crate-private `prepare_pfld_import<B, M>(&M, &SnapshotFile, &PfldImportRequest) -> Result<PreparedPfldImport<M>, WeightImportError>`.
- Produces `validate_target_contract<B, M>(&M, &PfldInspection) -> Result<(), WeightImportError>` so the Burn graph key set and target dtypes are checked before mutation.
- Produces `validate_pfld_apply_result(&ApplyResult, &PfldInspection) -> Result<Vec<String>, WeightImportError>`.
- `PreparedPfldImport<M>` owns the fully loaded candidate, source file name/hash, inspection, and sorted applied paths; the caller remains borrowed immutably throughout preparation.

- [ ] **Step 1: Add the dev-only model dependency and failing candidate tests**

Add only this dev-dependency:

```toml
[dev-dependencies]
feathertalk-models = { path = "../feathertalk-models", default-features = false }
zip.workspace = true
```

Inside `pfld/mod.rs` add unit-test helpers that resolve the tracked fixture as:

```rust
fn checkpoint_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../data_utils/checkpoint_epoch_335.pth.tar")
}

fn request_for(checkpoint: PathBuf, destination_dir: PathBuf) -> PfldImportRequest {
    PfldImportRequest {
        checkpoint,
        destination_dir,
        ..PfldImportRequest::default()
    }
}

fn assert_module_snapshots_equal<B: Backend, M: ModuleSnapshot<B>>(left: &M, right: &M) {
    let left = left.collect(None, None, false);
    let right = right.collect(None, None, false);
    assert_eq!(left.len(), right.len());
    for (left, right) in left.iter().zip(right.iter()) {
        assert_eq!(left.full_path(), right.full_path());
        assert_eq!(left.shape, right.shape);
        assert_eq!(left.dtype, right.dtype);
        assert_eq!(left.to_data().unwrap(), right.to_data().unwrap());
    }
}
```

Add these tests:

```rust
#[test]
fn real_checkpoint_prepares_a_complete_candidate_without_mutating_source_model() {
    let device = Default::default();
    let model = PFLD_GhostOne::<CpuBackend>::new(PfldConfig::production(), &device);
    let before = model.clone();
    let snapshot = SnapshotFile::copy_from(
        &checkpoint_path(),
        PfldImportRequest::default().max_file_bytes,
    )
    .unwrap();
    let request = request_for(checkpoint_path(), PathBuf::from("unused"));

    let prepared = prepare_pfld_import::<CpuBackend, _>(&model, &snapshot, &request).unwrap();

    assert_module_snapshots_equal(&before, &model);
    assert_eq!(prepared.applied.len(), 1_735);
    assert_eq!(
        prepared.inspection.applied,
        TensorSummary {
            tensor_count: 1_735,
            total_elements: 910_902
        }
    );
    assert!(prepared.applied.windows(2).all(|pair| pair[0] < pair[1]));
}

#[test]
fn incompatible_module_fails_before_any_caller_mutation() {
    let device = Default::default();
    let model = LinearConfig::new(2, 2).init::<CpuBackend>(&device);
    let before = model.clone();
    let snapshot = SnapshotFile::copy_from(
        &checkpoint_path(),
        PfldImportRequest::default().max_file_bytes,
    )
    .unwrap();
    let request = request_for(checkpoint_path(), PathBuf::from("unused"));

    let error =
        prepare_pfld_import::<CpuBackend, _>(&model, &snapshot, &request).unwrap_err();

    assert!(matches!(
        error,
        WeightImportError::MissingTensor(_) | WeightImportError::UnexpectedTensor(_)
    ));
    assert_module_snapshots_equal(&before, &model);
}
```

- [ ] **Step 2: Run the candidate tests and observe the red result**

Run:

```powershell
cargo test -p feathertalk-weights pfld::tests::real_checkpoint_prepares_a_complete_candidate_without_mutating_source_model
cargo test -p feathertalk-weights pfld::tests::incompatible_module_fails_before_any_caller_mutation
```

Expected: compilation fails because `prepare_pfld_import` and `PreparedPfldImport` do not exist.

- [ ] **Step 3: Configure a PFLD-only store and prepare the candidate**

Use these internal definitions:

```rust
#[derive(Debug)]
struct PreparedPfldImport<M> {
    candidate: M,
    source_file_name: String,
    source_sha256: String,
    inspection: PfldInspection,
    applied: Vec<String>,
}

fn configure_pfld_store(path: &Path) -> PytorchStore {
    PytorchStore::from_file(path)
        .with_top_level_key("pfld_backbone")
        .allow_partial(true)
        .validate(false)
        .map_indices_contiguous(false)
        .remap(pfld_remapper())
}

fn prepare_pfld_import<B, M>(
    module: &M,
    snapshot: &SnapshotFile,
    request: &PfldImportRequest,
) -> Result<PreparedPfldImport<M>, WeightImportError>
where
    B: Backend,
    M: ModuleSnapshot<B> + Clone,
{
    let pickle = PytorchReader::read_pickle_data(snapshot.path(), None)
        .map_err(|error| WeightImportError::InvalidPfldEnvelope(error.to_string()))?;
    let envelope = validate_envelope(pickle)?;
    let inspection = inspect_checkpoint(snapshot.path(), envelope, request)?;
    validate_target_contract::<B, M>(module, &inspection)?;
    let mut store = configure_pfld_store(snapshot.path());
    let mut candidate = module.clone();
    let result = candidate
        .load_from(&mut store)
        .map_err(|error| WeightImportError::Store(error.to_string()))?;
    let applied = validate_pfld_apply_result(&result, &inspection)?;
    let source_file_name = request
        .checkpoint
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .ok_or_else(|| {
            WeightImportError::Manifest(
                "checkpoint path must end in a UTF-8 file name".to_owned(),
            )
        })?
        .to_owned();

    Ok(PreparedPfldImport {
        candidate,
        source_file_name,
        source_sha256: snapshot.sha256().to_owned(),
        inspection,
        applied,
    })
}
```

The function must inspect and import only `snapshot.path()`. It must not reopen `request.checkpoint` after `SnapshotFile::copy_from` succeeds.

- [ ] **Step 4: Validate the target graph and complete `ApplyResult`**

`validate_target_contract` must collect the original module with `collect(None, None, false)` into a `BTreeMap` keyed by `full_path()`. Compare its key set exactly to `inspection.expected_applied`: a source path with no target returns `UnexpectedTensor`, while a target path with no source returns `MissingTensor`. Require `DType::F32` for every target snapshot and return `DTypeMismatch(path)` for the first non-f32 path. This closes the Burn 0.21 gap where `ApplyResult` preserves source dtype rather than comparing it to the target parameter dtype.

Implement `validate_pfld_apply_result` in this order:

1. Return `MissingTensor` for the first sorted `result.missing` path.
2. Return `ShapeMismatch` or `DTypeMismatch` for those `ApplyError` variants; convert adapter/load errors to `Store(error.to_string())`.
3. Reject any `result.skipped` entry with `Store("PFLD import unexpectedly skipped tensor: <path>")`.
4. Convert `result.unused` to a `BTreeSet`. An entry outside `inspection.expected_unused` returns `UnexpectedTensor`; an expected ignored path absent from `result.unused` returns `InvalidPfldIgnoredSet` because an ignored tensor was unexpectedly consumed.
5. Convert `result.applied` to a `BTreeSet` and compare it exactly to `inspection.expected_applied`. A missing expected path returns `MissingTensor`; an extra applied path returns `UnexpectedTensor`.
6. Return a lexicographically sorted copy of `result.applied`.

Use set differences rather than length-only checks. This makes an unapproved source key fail even when tensor counts happen to match.

- [ ] **Step 5: Run candidate, legacy, and model tests**

Run:

```powershell
cargo test -p feathertalk-weights pfld::tests
cargo test -p feathertalk-weights --all-targets
cargo test -p feathertalk-models --test pfld_shapes
```

Expected: candidate preparation reports exactly 1,735 applied tensors / 910,902 elements; the incompatible model fails; all existing weight and PFLD shape tests pass.

- [ ] **Step 6: Commit strict candidate loading**

```powershell
git add rust/Cargo.lock rust/crates/feathertalk-weights
git commit -m "feat: load strict PFLD checkpoint candidate"
```

---

### Task 5: Publish verified artifacts and accept the real checkpoint end to end

**Files:**
- Create: `rust/crates/feathertalk-weights/src/pfld/artifact.rs`
- Modify: `rust/crates/feathertalk-weights/src/pfld/mod.rs`
- Modify: `rust/crates/feathertalk-weights/src/lib.rs`
- Create: `rust/crates/feathertalk-weights/tests/pfld_import_failures.rs`
- Create: `rust/crates/feathertalk-weights/tests/pfld_checkpoint.rs`

**Interfaces:**
- Produces public `import_pfld_checkpoint<B, M>(&mut M, &PfldImportRequest) -> Result<PfldImportReport, WeightImportError> where B: Backend, M: ModuleSnapshot<B> + Clone`.
- Produces internal `write_staged_artifacts`, `verify_staged_artifacts`, `publish_staged_artifacts`, `module_summary`, and `compare_module_snapshots`.
- The only successful destination entries are `manifest.json` and `model.safetensors`.

- [ ] **Step 1: Add failing public failure-atomicity tests**

Create `tests/pfld_import_failures.rs`:

```rust
use std::path::{Path, PathBuf};

use burn::{
    nn::{LinearConfig},
    tensor::backend::Backend,
};
use burn_store::ModuleSnapshot;
use feathertalk_models::{PFLD_GhostOne, PfldConfig, backend::CpuBackend};
use feathertalk_weights::{
    PfldImportRequest, WeightImportError, import_pfld_checkpoint,
};

fn checkpoint_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../data_utils/checkpoint_epoch_335.pth.tar")
}

fn request(destination_dir: PathBuf) -> PfldImportRequest {
    PfldImportRequest {
        checkpoint: checkpoint_path(),
        destination_dir,
        ..PfldImportRequest::default()
    }
}

fn assert_module_snapshots_equal<B: Backend, M: ModuleSnapshot<B>>(left: &M, right: &M) {
    let left = left.collect(None, None, false);
    let right = right.collect(None, None, false);
    assert_eq!(left.len(), right.len());
    for (left, right) in left.iter().zip(right.iter()) {
        assert_eq!(left.full_path(), right.full_path());
        assert_eq!(left.shape, right.shape);
        assert_eq!(left.dtype, right.dtype);
        assert_eq!(left.to_data().unwrap(), right.to_data().unwrap());
    }
}

#[test]
fn existing_destination_is_rejected_without_overwrite_or_model_mutation() {
    let temp = tempfile::tempdir().unwrap();
    let destination = temp.path().join("published");
    std::fs::create_dir(&destination).unwrap();
    std::fs::write(destination.join("sentinel.txt"), b"keep").unwrap();
    let device = Default::default();
    let mut model = PFLD_GhostOne::<CpuBackend>::new(PfldConfig::production(), &device);
    let before = model.clone();

    let error =
        import_pfld_checkpoint::<CpuBackend, _>(&mut model, &request(destination.clone()))
            .unwrap_err();

    assert!(matches!(
        error,
        WeightImportError::ArtifactDestinationExists(path) if path == destination
    ));
    assert_eq!(
        std::fs::read(destination.join("sentinel.txt")).unwrap(),
        b"keep"
    );
    assert_module_snapshots_equal(&before, &model);
}

#[test]
fn incompatible_module_leaves_destination_absent_and_module_unchanged() {
    let temp = tempfile::tempdir().unwrap();
    let destination = temp.path().join("published");
    let device = Default::default();
    let mut model = LinearConfig::new(2, 2).init::<CpuBackend>(&device);
    let before = model.clone();

    let error =
        import_pfld_checkpoint::<CpuBackend, _>(&mut model, &request(destination.clone()))
            .unwrap_err();

    assert!(matches!(
        error,
        WeightImportError::MissingTensor(_) | WeightImportError::UnexpectedTensor(_)
    ));
    assert!(!destination.exists());
    assert_module_snapshots_equal(&before, &model);
}
```

- [ ] **Step 2: Add failing staged-corruption unit tests in `artifact.rs`**

Use a small `Linear<CpuBackend>` candidate and a temporary source file so these tests do not load the real checkpoint. Build a `PfldImportManifest` through `write_staged_artifacts`, then corrupt one staged file before verification:

```rust
#[test]
fn corrupt_safetensors_never_publishes_destination() {
    let fixture = artifact_fixture();
    let staged = write_staged_artifacts::<CpuBackend, _>(
        &fixture.candidate,
        &fixture.source_path,
        &fixture.source_file_name,
        &fixture.source_sha256,
        &fixture.inspection,
        fixture.parent.path(),
    )
    .unwrap();
    std::fs::write(staged.path().join("model.safetensors"), b"broken").unwrap();

    assert!(matches!(
        verify_staged_artifacts::<CpuBackend, _>(
            &fixture.original,
            &fixture.candidate,
            &fixture.source_path,
            &fixture.inspection,
            &staged,
        ),
        Err(WeightImportError::ArtifactValidation(_))
            | Err(WeightImportError::Store(_))
    ));
    assert!(!fixture.destination.exists());
}

#[test]
fn corrupt_manifest_never_publishes_destination() {
    let fixture = artifact_fixture();
    let staged = write_staged_artifacts::<CpuBackend, _>(
        &fixture.candidate,
        &fixture.source_path,
        &fixture.source_file_name,
        &fixture.source_sha256,
        &fixture.inspection,
        fixture.parent.path(),
    )
    .unwrap();
    std::fs::write(staged.path().join("manifest.json"), b"{not-json").unwrap();

    assert!(matches!(
        verify_staged_artifacts::<CpuBackend, _>(
            &fixture.original,
            &fixture.candidate,
            &fixture.source_path,
            &fixture.inspection,
            &staged,
        ),
        Err(WeightImportError::Manifest(_))
    ));
    assert!(!fixture.destination.exists());
}
```

Define the fixture exactly as:

```rust
struct ArtifactFixture {
    parent: tempfile::TempDir,
    destination: PathBuf,
    source_path: PathBuf,
    source_file_name: String,
    source_sha256: String,
    original: Linear<CpuBackend>,
    candidate: Linear<CpuBackend>,
    inspection: PfldInspection,
}

fn artifact_fixture() -> ArtifactFixture {
    let parent = tempfile::tempdir().unwrap();
    let destination = parent.path().join("published");
    let source_path = parent.path().join("source.pth");
    std::fs::write(&source_path, b"immutable-source").unwrap();
    let source_sha256 = sha256_file(&source_path).unwrap();
    let device = Default::default();
    let original = LinearConfig::new(2, 2).init::<CpuBackend>(&device);
    let candidate = original.clone();
    let summary = TensorSummary {
        tensor_count: 2,
        total_elements: 6,
    };
    let ignored = PfldIgnoredTensors {
        batch_norm_counters: TensorAudit {
            tensor_count: 0,
            total_elements: 0,
            keys: Vec::new(),
        },
        localization: TensorAudit {
            tensor_count: 0,
            total_elements: 0,
            keys: Vec::new(),
        },
        auxiliarynet: None,
    };
    let inspection = PfldInspection {
        backbone: summary.clone(),
        applied: summary,
        ignored,
        expected_applied: ["bias", "weight"]
            .into_iter()
            .map(str::to_owned)
            .collect(),
        expected_unused: BTreeSet::new(),
    };

    ArtifactFixture {
        parent,
        destination,
        source_path,
        source_file_name: "source.pth".to_owned(),
        source_sha256,
        original,
        candidate,
        inspection,
    }
}
```

The tests call no publish function after verification fails; dropping the owning `TempDir` removes staging.

- [ ] **Step 3: Run the new tests and observe the red result**

Run:

```powershell
cargo test -p feathertalk-weights --test pfld_import_failures
cargo test -p feathertalk-weights pfld::artifact::tests
```

Expected: compilation fails because the public importer and artifact module are absent.

- [ ] **Step 4: Implement staging, deterministic manifest writing, and strict reload**

Use:

```rust
const MODEL_FILE_NAME: &str = "model.safetensors";
const MANIFEST_FILE_NAME: &str = "manifest.json";

pub(super) struct StagedArtifacts {
    directory: tempfile::TempDir,
    manifest: PfldImportManifest,
}

impl StagedArtifacts {
    fn path(&self) -> &Path {
        self.directory.path()
    }
}
```

Implement these exact helper signatures:

```text
pub(super) fn write_staged_artifacts<B, M>(
    candidate: &M,
    source_path: &Path,
    source_file_name: &str,
    source_sha256: &str,
    inspection: &PfldInspection,
    parent: &Path,
) -> Result<StagedArtifacts, WeightImportError>
where
    B: Backend,
    M: ModuleSnapshot<B>

pub(super) fn verify_staged_artifacts<B, M>(
    original: &M,
    candidate: &M,
    source_path: &Path,
    inspection: &PfldInspection,
    staged: &StagedArtifacts,
) -> Result<(), WeightImportError>
where
    B: Backend,
    M: ModuleSnapshot<B> + Clone

fn module_summary<B, M>(module: &M) -> Result<TensorSummary, WeightImportError>
where
    B: Backend,
    M: ModuleSnapshot<B>

fn compare_module_snapshots<B, M>(
    expected: &M,
    actual: &M,
) -> Result<(), WeightImportError>
where
    B: Backend,
    M: ModuleSnapshot<B>
```

`write_staged_artifacts` must:

1. Call `tempfile::Builder::new().prefix(".feathertalk-pfld-").tempdir_in(parent)`.
2. Recompute `sha256_file(source_path)` and require it to equal `source_sha256`.
3. Save the candidate to `model.safetensors` with `SafetensorsStore::from_file(path).overwrite(false)`.
4. Compute the staged model SHA-256 with `sha256_file`.
5. Compute `module_summary(candidate)` from `candidate.collect(None, None, false)` with checked count/element arithmetic.
6. Require the candidate summary to equal `inspection.applied`.
7. Construct the exact schema-v1 manifest values:

```rust
PfldImportManifest {
    schema_version: 1,
    model_type: "pfld_ghost_one".to_owned(),
    architecture_version: PFLD_ARCHITECTURE_VERSION.to_owned(),
    source: PfldSourceManifest {
        file_name: source_file_name.to_owned(),
        sha256: source_sha256.to_owned(),
    },
    epoch: PFLD_CHECKPOINT_EPOCH,
    backbone: inspection.backbone.clone(),
    model: PfldModelArtifact {
        format: "safetensors".to_owned(),
        file_name: MODEL_FILE_NAME.to_owned(),
        sha256: model_sha256,
        tensor_count: inspection.applied.tensor_count,
        total_elements: inspection.applied.total_elements,
    },
    ignored: inspection.ignored.clone(),
}
```

8. Serialize with `serde_json::to_vec_pretty`, append exactly one `b'\n'`, and write with `OpenOptions::new().write(true).create_new(true)` followed by `sync_all`.

`verify_staged_artifacts` must:

1. Clone the original module structure.
2. Load staged safetensors with `allow_partial(true).validate(false)`.
3. Require no missing, unused, skipped, or apply errors; map shape/dtype failures to the existing stable variants.
4. Compare candidate and reloaded snapshots by a `BTreeMap` keyed by `full_path()`, checking exact key, shape, dtype, and `TensorData` equality.
5. Read and deserialize `manifest.json` into `PfldImportManifest` and require equality with `staged.manifest`.
6. Recompute the source hash from the immutable snapshot path and the model hash from staged safetensors.
7. Recompute the reloaded module tensor count/elements and compare them to `manifest.model` and `inspection.applied`.
8. Compare `manifest.backbone` and every ignored audit to `inspection`; for each audit require `tensor_count == keys.len()` and lexicographically sorted, duplicate-free keys.
9. Require both hashes to be exactly 64 lowercase hexadecimal characters.
10. Sort `read_dir(staging)` file names and require exactly `["manifest.json", "model.safetensors"]`.

Parser/serialization failures return `Manifest`. Hash, content, count, unexpected-file, and tensor-comparison failures return `ArtifactValidation`. Safetensors store failures retain `Store`, `MissingTensor`, `UnexpectedTensor`, `ShapeMismatch`, or `DTypeMismatch` as appropriate.

- [ ] **Step 5: Implement same-filesystem publication and the public orchestration**

Implement publication:

```rust
pub(super) fn publish_staged_artifacts(
    staged: StagedArtifacts,
    destination: &Path,
) -> Result<PfldImportManifest, WeightImportError> {
    ensure_destination_absent(destination)?;
    std::fs::rename(staged.path(), destination)?;
    let manifest = staged.manifest.clone();
    let _old_staging_path = staged.directory.keep();
    Ok(manifest)
}
```

Use `std::fs::symlink_metadata` rather than `Path::exists` so a broken symlink, file, or directory all count as an existing destination:

```rust
pub(super) fn ensure_destination_absent(path: &Path) -> Result<(), WeightImportError> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Err(WeightImportError::ArtifactDestinationExists(
            path.to_owned(),
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(WeightImportError::Io(error)),
    }
}
```

Implement the public function in `pfld/mod.rs` and export it from `lib.rs`:

```rust
pub fn import_pfld_checkpoint<B, M>(
    module: &mut M,
    request: &PfldImportRequest,
) -> Result<PfldImportReport, WeightImportError>
where
    B: Backend,
    M: ModuleSnapshot<B> + Clone,
{
    if request.destination_dir.as_os_str().is_empty() {
        return Err(WeightImportError::ArtifactValidation(
            "destination directory must not be empty".to_owned(),
        ));
    }
    ensure_destination_absent(&request.destination_dir)?;
    let parent = request
        .destination_dir
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(WeightImportError::ArtifactValidation(
            "destination directory parent must exist and be a directory".to_owned(),
        ));
    }

    let snapshot = SnapshotFile::copy_from(&request.checkpoint, request.max_file_bytes)?;
    let prepared = prepare_pfld_import::<B, M>(module, &snapshot, request)?;
    let staged = write_staged_artifacts::<B, M>(
        &prepared.candidate,
        snapshot.path(),
        &prepared.source_file_name,
        &prepared.source_sha256,
        &prepared.inspection,
        parent,
    )?;
    verify_staged_artifacts::<B, M>(
        module,
        &prepared.candidate,
        snapshot.path(),
        &prepared.inspection,
        &staged,
    )?;
    let manifest = publish_staged_artifacts(staged, &request.destination_dir)?;
    *module = prepared.candidate;

    Ok(PfldImportReport {
        destination_dir: request.destination_dir.clone(),
        manifest,
        applied: prepared.applied,
    })
}
```

There must be no fallible operation after the successful directory rename and before `*module = prepared.candidate`. If the final rename fails, `StagedArtifacts` still owns and removes staging; the destination remains absent and the caller remains unchanged.

- [ ] **Step 6: Add the real-checkpoint integration test**

Create `tests/pfld_checkpoint.rs`. The single test must fail clearly when the tracked fixture is absent and assert every approved baseline:

```rust
use std::{
    fs::File,
    io::Read,
    path::Path,
};

use burn::tensor::{Tensor, backend::Backend};
use burn_store::{ModuleSnapshot, SafetensorsStore};
use feathertalk_models::{PFLD_GhostOne, PfldConfig, backend::CpuBackend};
use feathertalk_weights::{
    PfldImportManifest, PfldImportRequest, TensorAudit, TensorSummary,
    import_pfld_checkpoint,
};
use sha2::{Digest, Sha256};

fn sha256(path: &Path) -> String {
    let mut file = File::open(path).unwrap();
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).unwrap();
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    hex::encode(hasher.finalize())
}

fn assert_module_snapshots_equal<B: Backend, M: ModuleSnapshot<B>>(left: &M, right: &M) {
    let left = left.collect(None, None, false);
    let right = right.collect(None, None, false);
    assert_eq!(left.len(), right.len());
    for (left, right) in left.iter().zip(right.iter()) {
        assert_eq!(left.full_path(), right.full_path());
        assert_eq!(left.shape, right.shape);
        assert_eq!(left.dtype, right.dtype);
        assert_eq!(left.to_data().unwrap(), right.to_data().unwrap());
    }
}

fn assert_sorted_unique(keys: &[String]) {
    assert!(keys.windows(2).all(|pair| pair[0] < pair[1]));
}

#[test]
fn tracked_epoch_335_checkpoint_imports_and_round_trips() {
    let checkpoint = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../data_utils/checkpoint_epoch_335.pth.tar");
    assert!(
        checkpoint.is_file(),
        "tracked PFLD checkpoint is missing: {}",
        checkpoint.display()
    );
    assert_eq!(std::fs::metadata(&checkpoint).unwrap().len(), 5_039_598);
    assert_eq!(
        sha256(&checkpoint),
        "bada866661ad5fa1080a085f51fe9c016c69958c406951afa4afc7840f856de0"
    );

    let temp = tempfile::tempdir().unwrap();
    let destination = temp.path().join("pfld-import");
    let device = Default::default();
    let mut model = PFLD_GhostOne::<CpuBackend>::new(PfldConfig::production(), &device);
    let report = import_pfld_checkpoint::<CpuBackend, _>(
        &mut model,
        &PfldImportRequest {
            checkpoint: checkpoint.clone(),
            destination_dir: destination.clone(),
            ..PfldImportRequest::default()
        },
    )
    .unwrap();

    assert_eq!(report.destination_dir, destination);
    assert_eq!(report.manifest.schema_version, 1);
    assert_eq!(report.manifest.model_type, "pfld_ghost_one");
    assert_eq!(
        report.manifest.architecture_version,
        "burn-pfld-structure-v1"
    );
    assert_eq!(report.manifest.epoch, 335);
    assert_eq!(
        report.manifest.source.sha256,
        "bada866661ad5fa1080a085f51fe9c016c69958c406951afa4afc7840f856de0"
    );
    assert_eq!(
        report.manifest.backbone,
        TensorSummary {
            tensor_count: 2_090,
            total_elements: 913_663
        }
    );
    assert_eq!(report.applied.len(), 1_735);
    assert_sorted_unique(&report.applied);
    assert_eq!(report.manifest.model.tensor_count, 1_735);
    assert_eq!(report.manifest.model.total_elements, 910_902);
    assert_eq!(
        report.manifest.ignored.batch_norm_counters.tensor_count,
        351
    );
    assert_eq!(
        report.manifest.ignored.batch_norm_counters.total_elements,
        351
    );
    assert_sorted_unique(&report.manifest.ignored.batch_norm_counters.keys);
    assert_eq!(
        report.manifest.ignored.localization,
        TensorAudit {
            tensor_count: 4,
            total_elements: 2_410,
            keys: vec![
                "localization.0.bias".to_owned(),
                "localization.0.weight".to_owned(),
                "localization.3.bias".to_owned(),
                "localization.3.weight".to_owned(),
            ],
        }
    );
    assert_sorted_unique(&report.manifest.ignored.localization.keys);
    let auxiliary = report.manifest.ignored.auxiliarynet.as_ref().unwrap();
    assert_eq!(auxiliary.tensor_count, 48);
    assert_eq!(auxiliary.total_elements, 137_036);
    assert!(auxiliary.keys.iter().all(|key| key.starts_with("auxiliarynet.")));
    assert_sorted_unique(&auxiliary.keys);

    let mut entries = std::fs::read_dir(&destination)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    entries.sort();
    assert_eq!(
        entries,
        vec!["manifest.json".to_owned(), "model.safetensors".to_owned()]
    );

    let manifest: PfldImportManifest = serde_json::from_slice(
        &std::fs::read(destination.join("manifest.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(manifest, report.manifest);
    assert_eq!(
        sha256(&destination.join("model.safetensors")),
        report.manifest.model.sha256
    );

    let mut reloaded =
        PFLD_GhostOne::<CpuBackend>::new(PfldConfig::production(), &device);
    let mut store = SafetensorsStore::from_file(destination.join("model.safetensors"));
    let result = reloaded.load_from(&mut store).unwrap();
    assert!(result.missing.is_empty());
    assert!(result.unused.is_empty());
    assert!(result.errors.is_empty());
    assert_module_snapshots_equal(&model, &reloaded);

    let input = Tensor::<CpuBackend, 4>::zeros([1, 3, 192, 192], &device);
    assert_eq!(reloaded.forward(input).dims(), [1, 220]);
}
```

The helpers stream hashes, compare every collected tensor path/shape/dtype/value, and enforce strict ascending key order. Call `assert_sorted_unique` for `report.applied` and every ignored key list, including the fixed localization list.

- [ ] **Step 7: Run focused red/green verification**

Run:

```powershell
cargo test -p feathertalk-weights pfld::artifact::tests
cargo test -p feathertalk-weights --test pfld_import_failures
cargo test -p feathertalk-weights --test pfld_checkpoint -- --nocapture
```

Expected: corruption tests fail without publishing, failure-atomicity tests preserve modules and destinations, and the real checkpoint produces the fixed counts and output shape.

- [ ] **Step 8: Run full acceptance**

Run from `rust/`:

```powershell
cargo fmt --check
cargo clippy -p feathertalk-weights --all-targets --all-features -- -D warnings
cargo test -p feathertalk-weights --all-targets
cargo test -p feathertalk-models --all-targets
git diff --check
```

Expected: every command exits 0; no generated `model.safetensors` or `manifest.json` appears in `git status --short`.

- [ ] **Step 9: Commit the verified importer**

```powershell
git add rust/Cargo.lock rust/crates/feathertalk-weights
git commit -m "feat: import PFLD checkpoint artifacts"
```

## Plan Self-Review

- Spec coverage: Tasks 1-5 cover the exact epoch/envelope, immutable source snapshot, global limits, reviewed remapping, exact localization set, conditional BatchNorm counters, optional auxiliary audit, strict cloned-module application, safetensors reload/equality, deterministic manifest, same-filesystem publication, failure atomicity, real checkpoint statistics, and zero-input Burn shape smoke test.
- Dependency boundary: `feathertalk-weights` adds only workspace `serde_json` normally; `feathertalk-models` is dev-only and no Python/runtime component is introduced.
- Exclusions: no Python/Burn numerical parity, image path, STN/localization execution, AuxiliaryNet implementation, MobileOne reparameterization, permanent weights, final model-package schema, CLI, worker, or GPUI work appears in any task.
- Type consistency: `PfldInspection` feeds `PreparedPfldImport` and artifact verification; the public manifest/report names and field types match the approved specification; `import_pfld_checkpoint` retains `M: ModuleSnapshot<B> + Clone`.
- Atomicity: the caller is immutable through preparation and verification, staging is owned by `TempDir` until rename, the destination must be absent, and no fallible step follows publication.
- Placeholder scan: the plan contains no deferred-work markers, undefined public type, or unspecified test command.
