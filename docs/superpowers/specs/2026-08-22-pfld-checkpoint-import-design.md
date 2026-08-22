# PFLD Checkpoint Import Design

Date: 2026-08-22
Status: Approved for implementation planning

## Purpose

Add a strict Rust import path for the repository's legacy PFLD checkpoint format. The importer loads the `pfld_backbone` tensor mapping into the existing Burn `PFLD_GhostOne` training-form graph, writes a temporary safetensors artifact plus an auditable manifest, and proves that the safetensors artifact can be read back without changing tensor values.

This slice establishes checkpoint compatibility and conversion integrity. It does not claim numerical parity between Python and Burn model execution.

## Approved Decisions

- Import only the fixed PFLD checkpoint envelope used by `data_utils/checkpoint_epoch_335.pth.tar`.
- Require top-level `epoch` and `pfld_backbone`; allow optional `auxiliarynet`.
- Do not fall back to direct state dictionaries or generic `model` / `state_dict` keys.
- Require integer epoch `335`.
- Treat the four unused `localization.*` tensors as an exact PFLD-only ignored set.
- Record an optional `auxiliarynet` tensor mapping as known but not imported.
- Reject every other unknown top-level or backbone tensor.
- Generate `model.safetensors` and `manifest.json` only in a caller-selected temporary destination. Do not commit generated weights in this slice.
- Re-read the generated safetensors and compare every module snapshot tensor before reporting success.
- Keep Python out of the Rust import and test path.

## Scope

Included:

- A PFLD-specific import request, report, manifest, and import function in `feathertalk-weights`.
- Restricted inspection of the PyTorch checkpoint envelope through Burn's existing pickle reader.
- Exact PFLD tensor-key remapping into the existing Burn module snapshot paths.
- File-size, tensor-count, and total-element safety limits over all checkpoint tensor mappings, including ignored mappings.
- Failure-atomic model replacement and artifact publication.
- Safetensors serialization, hashing, reloading, and tensor-by-tensor equality verification.
- A real-checkpoint integration test against `data_utils/checkpoint_epoch_335.pth.tar`.

Excluded:

- Python-versus-Burn output parity or tolerance selection.
- Image decoding, BGR handling, resize, normalization, crop creation, SCRFD, or landmark postprocessing.
- `AuxiliaryNet`, STN, or the unused localization graph in the Burn model.
- MobileOne reparameterization or an inference-form PFLD graph.
- Committing generated safetensors or replacing the source `.pth.tar`.
- The final standard `model-package/manifest.json`, license bundle, CLI, worker, or GPUI integration.
- Generic support for arbitrary PFLD checkpoints, direct state dictionaries, `model`, or `state_dict` envelopes.

## Source Checkpoint Baseline

The repository source checkpoint is:

```text
path:   data_utils/checkpoint_epoch_335.pth.tar
size:   5,039,598 bytes
sha256: bada866661ad5fa1080a085f51fe9c016c69958c406951afa4afc7840f856de0
```

Its approved structural baseline is:

```text
top-level keys: epoch, pfld_backbone, auxiliarynet
epoch:          335

pfld_backbone:       2090 tensors / 913663 elements
applied to Burn:     1735 tensors / 910902 elements
BN counters ignored:  351 tensors / 351 elements
localization ignored:   4 tensors / 2410 elements
auxiliarynet ignored:   48 tensors / 137036 elements
```

The source SHA-256 and these statistics are asserted by the repository integration test. The public importer records the source hash instead of hard-coding it as an acceptance condition, so a byte-distinct checkpoint with the same approved envelope is still identifiable and auditable. The fixed epoch and architecture contract remain mandatory.

## Crate Boundaries

The implementation belongs in `feathertalk-weights` because that crate already owns:

- restricted PyTorch tensor loading;
- import safety limits;
- Burn `ModuleSnapshot` application;
- safetensors writing; and
- source SHA-256 reporting.

`feathertalk-weights` must not gain a normal dependency on `feathertalk-models`. The PFLD importer accepts a generic Burn `ModuleSnapshot` target but applies only the PFLD envelope and key contract. The real-checkpoint integration test may add `feathertalk-models` as a dev-dependency so it can construct `PFLD_GhostOne<CpuBackend>`.

This avoids a runtime dependency between the weight-format crate and a concrete model crate while still proving the production PFLD graph imports successfully.

The implementation adds `serde_json` from the existing workspace dependency table and a dev-only path dependency on `feathertalk-models`. It introduces no new third-party crate or network-fetched runtime component beyond the workspace lockfile.

## Public API

The exact field layout is:

```rust
pub const PFLD_CHECKPOINT_EPOCH: u64 = 335;
pub const PFLD_ARCHITECTURE_VERSION: &str = "burn-pfld-structure-v1";

pub struct PfldImportRequest {
    pub checkpoint: PathBuf,
    pub destination_dir: PathBuf,
    pub max_file_bytes: u64,
    pub max_tensor_count: usize,
    pub max_total_elements: u64,
}

pub struct TensorAudit {
    pub tensor_count: usize,
    pub total_elements: u64,
    pub keys: Vec<String>,
}

pub struct PfldIgnoredTensors {
    pub batch_norm_counters: TensorAudit,
    pub localization: TensorAudit,
    pub auxiliarynet: Option<TensorAudit>,
}

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

pub struct PfldImportReport {
    pub destination_dir: PathBuf,
    pub manifest: PfldImportManifest,
    pub applied: Vec<String>,
}

pub fn import_pfld_checkpoint<B, M>(
    module: &mut M,
    request: &PfldImportRequest,
) -> Result<PfldImportReport, WeightImportError>
where
    B: Backend,
    M: ModuleSnapshot<B> + Clone;
```

`PfldSourceManifest`, `TensorSummary`, and `PfldModelArtifact` are serializable value types containing only the fields shown in the manifest schema below. Audit key lists are sorted lexicographically before they enter either the report or manifest.

`PfldImportRequest::default()` uses the existing legacy-import safety defaults:

```text
max_file_bytes:    4 GiB
max_tensor_count:  10,000
max_total_elements: 2,000,000,000
```

Callers must supply `checkpoint` and `destination_dir`. The destination directory must not already exist. Its parent must exist and be writable.

## Restricted Envelope Inspection

The importer first copies the source checkpoint into an internal temporary snapshot while hashing it and enforcing `max_file_bytes`. Parsing always uses the immutable snapshot so a concurrent source-file change cannot create a mismatch between validation and import.

Burn 0.21 exposes `PytorchReader::read_pickle_data`, which returns restricted `PickleValue` data without invoking arbitrary Python globals or reduce callables. The importer uses it to validate the top-level dictionary before opening tensor stores.

The only accepted top-level key sets are:

```text
{ epoch, pfld_backbone }
{ epoch, pfld_backbone, auxiliarynet }
```

Validation rules:

1. The root value must be a dictionary.
2. `epoch` must exist and be an integer equal to `335`.
3. `pfld_backbone` must exist and be readable as a non-empty tensor mapping.
4. `auxiliarynet`, when present, must be readable as a tensor mapping.
5. Any other top-level key is an error.
6. No fallback to `model`, `state_dict`, or a direct tensor mapping is attempted.

All tensors in `pfld_backbone` and optional `auxiliarynet`, including tensors later classified as ignored, count toward `max_tensor_count` and `max_total_elements`.

## PFLD Key Mapping

The key remapper is PFLD-specific and uses anchored, reviewed transformations. It does not share the looser model-selection behavior of `LegacyImportRequest`.

MobileOne mappings:

```text
rbr_conv.N.conv       -> branches.N.0
rbr_conv.N.bn         -> branches.N.1
rbr_scale.conv        -> scale.0
rbr_scale.bn          -> scale.1
rbr_skip              -> skip
```

GhostOne mappings:

```text
ghost_conv.0          -> ghost
ghost_conv.1          -> depthwise
ghost_conv.2          -> linear
primary_conv          -> primary
cheap_operation       -> cheap
```

Head mappings:

```text
conv8.0               -> conv8
conv_out              -> head
```

BatchNorm parameters map as follows only on paths known to refer to BatchNorm modules:

```text
weight        -> gamma
bias          -> beta
running_mean  -> running_mean
running_var   -> running_var
```

`PytorchStore` continues to own tensor-layout adaptation. The PFLD remapper does not manually transpose convolution weights.

Representative exact examples:

```text
conv1.rbr_conv.0.conv.weight
-> conv1.branches.0.0.weight

conv1.rbr_conv.0.bn.weight
-> conv1.branches.0.1.gamma

conv3_1.ghost_conv.0.primary_conv.rbr_conv.0.bn.weight
-> conv3_1.ghost.primary.branches.0.1.gamma

conv3_1.ghost_conv.1.rbr_conv.0.conv.weight
-> conv3_1.depthwise.branches.0.0.weight

conv3_1.ghost_conv.2.cheap_operation.rbr_scale.bn.bias
-> conv3_1.linear.cheap.scale.1.beta

conv8.0.weight
-> conv8.weight

conv_out.bias
-> head.bias
```

The implementation rejects duplicate destination keys after remapping and before applying any tensor to the candidate model.

## Ignored Tensor Contract

Ignored tensors are explicit audit categories, never silent wildcard exclusions.

### BatchNorm Counters

A key ending in `.num_batches_tracked` is ignored only when its parent path also contains successfully mapped BatchNorm `running_mean` and `running_var` tensors. An arbitrary unknown key cannot become accepted merely by using this suffix.

The real checkpoint contains 351 such scalar `int64` tensors.

### Localization Tensors

The `pfld_backbone` mapping must contain exactly this four-key set:

```text
localization.0.weight
localization.0.bias
localization.3.weight
localization.3.bias
```

These tensors are unused by the Python `PFLD_GhostOne.forward` path and have no counterpart in the approved Burn graph. All four are ignored and reported. A missing member, additional `localization.*` member, or different localization name is an architecture-contract error.

### Auxiliary Network

The optional top-level `auxiliarynet` mapping is not loaded into the PFLD backbone. When present, the importer:

- enumerates every tensor key;
- includes its tensors in safety-limit accounting;
- records its sorted, fully qualified keys as `auxiliarynet.<key>`;
- records tensor count and total elements; and
- does not validate AuxiliaryNet-specific shapes because AuxiliaryNet is outside this slice.

Unknown top-level siblings remain errors.

## Strict Module Application

The importer creates a clone of the caller's module and applies tensors only to that candidate. The candidate replaces the caller's module only after every validation and artifact-publication step succeeds.

The applied backbone contract is strict:

- every non-ignored source tensor must map to exactly one Burn snapshot;
- every required Burn snapshot must receive a tensor;
- only `float32` tensors may be applied;
- shape and dtype mismatches are errors;
- unused non-ignored source tensors are errors; and
- missing Burn tensors are errors.

The report's `applied` list contains sorted Burn destination paths. The manifest records the applied tensor count and total elements but does not duplicate the full applied-key list. All ignored groups include complete sorted key lists.

## Failure-Atomic Artifact Publication

`destination_dir` must not exist. The importer creates a staging directory in the same parent directory so the final publication can use a same-filesystem directory rename.

The staging sequence is:

1. Save the fully imported candidate as `model.safetensors`.
2. Compute the lowercase SHA-256 of `model.safetensors`.
3. Clone the original target structure, load the staged safetensors into it, and require a strict apply result.
4. Compare every candidate and reloaded module snapshot for identical key, shape, dtype, and value data.
5. Serialize `manifest.json` using deterministic struct field order and lexicographically sorted key lists.
6. Re-read and deserialize `manifest.json`.
7. Recompute source and model hashes and revalidate all manifest counts.
8. Rename the complete staging directory to `destination_dir`.
9. Replace the caller's module with the verified candidate.

Any error before step 8 removes the staging directory automatically and leaves both the caller's module and destination absent. The final directory contains exactly:

```text
destination_dir/
  manifest.json
  model.safetensors
```

No file is overwritten. Absolute paths and timestamps are excluded from the manifest so identical inputs produce stable metadata apart from the safetensors hash, which is itself derived from the generated file.

## Temporary Manifest Schema

The manifest is an import-audit record, not the final model-package schema. In the schema illustration below, descriptive strings state the required value constraint; produced manifests contain the actual hash and key values rather than those descriptions.

```json
{
  "schema_version": 1,
  "model_type": "pfld_ghost_one",
  "architecture_version": "burn-pfld-structure-v1",
  "source": {
    "file_name": "checkpoint_epoch_335.pth.tar",
    "sha256": "bada866661ad5fa1080a085f51fe9c016c69958c406951afa4afc7840f856de0"
  },
  "epoch": 335,
  "backbone": {
    "tensor_count": 2090,
    "total_elements": 913663
  },
  "model": {
    "format": "safetensors",
    "file_name": "model.safetensors",
    "sha256": "exactly 64 lowercase hexadecimal characters",
    "tensor_count": 1735,
    "total_elements": 910902
  },
  "ignored": {
    "batch_norm_counters": {
      "tensor_count": 351,
      "total_elements": 351,
      "keys": ["every ignored BatchNorm counter key in lexicographic order"]
    },
    "localization": {
      "tensor_count": 4,
      "total_elements": 2410,
      "keys": [
        "localization.0.bias",
        "localization.0.weight",
        "localization.3.bias",
        "localization.3.weight"
      ]
    },
    "auxiliarynet": {
      "tensor_count": 48,
      "total_elements": 137036,
      "keys": ["every fully qualified auxiliarynet key in lexicographic order"]
    }
  }
}
```

When `auxiliarynet` is absent, its manifest value is `null`. Empty or partial localization groups are never valid.

## Error Model

`WeightImportError` gains stable PFLD/artifact categories while retaining existing store, shape, dtype, duplicate-key, I/O, and safety-limit variants. Required new categories are:

```rust
InvalidPfldEnvelope(String)
InvalidPfldEpoch { expected: u64, actual: String }
InvalidPfldIgnoredSet(String)
ArtifactDestinationExists(PathBuf)
ArtifactValidation(String)
Manifest(String)
```

Technical parser/store sources remain visible in error text, but tests match stable variants and fields rather than full display strings.

## Testing Strategy

All committed tests run with Rust only, require no network access, and never modify the source checkpoint.

### Key Mapping Unit Tests

- Assert representative MobileOne, GhostOne, conv8, head, and BatchNorm mappings.
- Assert similar but unapproved names do not match reviewed transformations.
- Reject two source keys that collide after remapping.
- Verify BatchNorm-counter recognition requires mapped sibling buffers.

### Envelope and Audit Unit Tests

- Accept the two allowed top-level key sets.
- Reject missing `epoch`, missing `pfld_backbone`, non-integer epoch, epoch other than `335`, and every unknown top-level key.
- Require the exact four localization keys.
- Verify optional auxiliary keys are fully qualified, sorted, and counted.
- Verify applied and ignored tensors all contribute to global safety limits.

Envelope helpers operate on synthetic `PickleValue` and key collections, so failure cases do not require creating unsafe or duplicated multi-megabyte checkpoint files.

### Real Checkpoint Integration Test

The integration test resolves the tracked fixture relative to `CARGO_MANIFEST_DIR`:

```text
../../../data_utils/checkpoint_epoch_335.pth.tar
```

It must fail clearly if the tracked file is missing; it must not silently skip. The test:

1. Creates `PFLD_GhostOne<CpuBackend>` with `PfldConfig::production()`.
2. Imports the real checkpoint into a non-existent destination under `tempfile::TempDir`.
3. Asserts epoch, source SHA-256, all fixed counts, and the exact localization set.
4. Asserts the destination contains only `manifest.json` and `model.safetensors`.
5. Recomputes both hashes and deserializes the manifest.
6. Loads safetensors into a fresh production PFLD model and compares all module snapshots.
7. Runs a zero-input smoke forward and asserts output shape `[1, 220]` without comparing numerical values to Python.

### Failure Atomicity Tests

- Importing into an incompatible Burn module must fail without mutating it.
- A pre-existing destination directory must be rejected without overwrite.
- Invalid envelope, ignored-set, key-map, shape, or dtype helpers must not publish a destination.
- A staged manifest or safetensors verification failure must leave the destination absent.

Existing generic legacy-import tests continue to prove missing, unexpected, shape-mismatched, dtype-mismatched, and oversized inputs are rejected.

## Acceptance

Run from `rust/` in an isolated worktree:

```powershell
cargo fmt --check
cargo clippy -p feathertalk-weights --all-targets --all-features -- -D warnings
cargo test -p feathertalk-weights --all-targets
cargo test -p feathertalk-models --all-targets
git diff --check
```

Success for this slice means the tracked legacy checkpoint is strictly imported, converted to a temporary safetensors artifact, described by a verified temporary manifest, reloaded tensor-for-tensor, and usable for a Burn shape smoke test. Numerical output parity remains a separate follow-up contract.
