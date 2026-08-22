# SCRFD Burn Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the tracked SCRFD 2.5G ONNX model into committed Burn Rust source and safetensors weights, expose strict batch-one raw inference through `feathertalk-scrfd`, and prove all nine outputs match Python/OpenCV DNN on a deterministic fixture.

**Architecture:** Keep the runtime crate independent from ONNX, Python, image processing, and `feathertalk-face`. A standalone pinned development tool validates the source ONNX, invokes `burn-onnx 0.21.0`, converts the temporary burnpack state to safetensors, and publishes reproducible source/artifact files. The runtime crate validates a strict manifest and immutable weight-byte snapshot, loads the private generated Burn model, checks the fixed input/output shapes, and returns three structured feature levels.

**Tech Stack:** Rust 1.92 edition 2024, Burn and `burn-store =0.21.0`, `burn-onnx =0.21.0`, `prost =0.14.4` for a minimal read-only ONNX contract parser, `serde`/`serde_json`, SHA-256 via `sha2`/`hex`, `ndarray`/`ndarray-npy`, Python 3.11, `numpy =2.2.6`, and `opencv-python-headless =4.12.0.88`.

## Global Constraints

- Execute implementation inline with `superpowers:executing-plans`; the user explicitly declined subagents, so do not dispatch any.
- Start execution through `superpowers:using-git-worktrees` in an isolated branch/worktree named `scrfd-burn-inference`, suggested path `E:\workspace\github\FeatherTalk\.worktrees\scrfd-burn-inference`.
- Run Rust workspace commands from that worktree's `rust/` directory unless a step explicitly targets the standalone importer manifest.
- Preserve the user-owned untracked file `demo/kanghui_training_video_featherhubert_188_latest/README.txt`; never add, delete, overwrite, or move it.
- Source ONNX is exactly `data_utils/scrfd_2.5g_kps.onnx`, 3,291,017 bytes, lowercase SHA-256 `32d20c77b9e2dc1d07e94c2ab9d25bdd5cd05eddbe0b46e7b38e7a1eca22e99a`, default-domain opset 12, input `images [1,3,640,640]`, and outputs `out0..out8`.
- Raw ONNX output shapes are `out0 [1,12800,1]`, `out1 [1,3200,1]`, `out2 [1,800,1]`, `out3 [1,12800,4]`, `out4 [1,3200,4]`, `out5 [1,800,4]`, `out6 [1,12800,10]`, `out7 [1,3200,10]`, and `out8 [1,800,10]`.
- Public score tensors remove only the final singleton dimension; bbox and keypoint tensors preserve their raw values and shapes.
- The public model accepts only `Tensor<B, 4>` shaped `[1,3,640,640]`; do not add dynamic batches or pixel/image preprocessing.
- The runtime crate must not depend on `burn-onnx`, ONNX protobuf code, Python, OpenCV, or `feathertalk-face`.
- `rust/tools/scrfd-import` is explicitly excluded from the main workspace and pins its own dependency versions and lockfile.
- Burn ONNX generation uses simplification and partitioning enabled with `LoadStrategy::None`. Burn ONNX still writes a temporary `.bpk`; generated runtime source contains no burnpack loading constructor.
- The converter loads the temporary `.bpk` into `generated::Model::new` through `BurnpackStore`, saves safetensors, reloads it into a fresh model, and compares every module snapshot before publishing.
- Ordinary `cargo build/test --workspace` must not parse ONNX, compile the importer, execute Python, read OpenCV, or access the network.
- Runtime loading rejects manifests over 64 KiB and weights over 16 MiB, hashes the exact weight bytes later supplied to `SafetensorsStore::from_bytes`, and requires strict tensor application.
- Manifest license fields remain `license_id = "NOASSERTION"` and `redistribution_approved = false`; do not imply release or commercial redistribution approval.
- CPU/NdArray comparison is mandatory for all nine outputs with `max_abs <= 1e-3` and `mean_abs <= 1e-4`. WGPU is an ignored manual load/forward/shape smoke test only.
- Use test-driven development: each behavior starts with a failing focused test, then minimal implementation, focused green verification, and a commit.
- Do not claim completion until fresh formatting, clippy, importer verification, fixture verification, crate tests, workspace tests, and `git diff --check` all pass.

## File Map

- Modify `rust/Cargo.toml`: add `crates/feathertalk-scrfd` as a member and `tools/scrfd-import` under `exclude`.
- Modify `rust/Cargo.lock`: update only through Cargo for the new runtime crate.
- Create `rust/crates/feathertalk-scrfd/Cargo.toml`: runtime and test dependencies, mirroring existing Burn backend features.
- Create `rust/crates/feathertalk-scrfd/src/lib.rs`: export the manifest, artifact path, model, output, and error APIs.
- Create `rust/crates/feathertalk-scrfd/src/error.rs`: stable manifest, artifact, store, input-shape, and output-shape errors.
- Create `rust/crates/feathertalk-scrfd/src/manifest.rs`: schema-one types, fixed SCRFD constants, and structural validation.
- Create `rust/crates/feathertalk-scrfd/src/artifact.rs`: bounded immutable reads, hash checks, compiled-artifact checks, and strict safetensors application.
- Create `rust/crates/feathertalk-scrfd/src/model.rs`: private generated model ownership, `load`, `forward`, and `manifest`.
- Create `rust/crates/feathertalk-scrfd/src/output.rs`: raw nine-tensor shape validation and three-level structured output.
- Create `rust/crates/feathertalk-scrfd/src/generated/mod.rs`: keep generated implementation private.
- Generate `rust/crates/feathertalk-scrfd/src/generated/scrfd_2_5g.rs`: committed Burn ONNX graph.
- Generate `rust/crates/feathertalk-scrfd/src/generated/artifact_contract.rs`: committed source/weight byte-count and SHA-256 constants.
- Generate `rust/crates/feathertalk-scrfd/artifacts/scrfd_2_5g/model.safetensors`: committed standard weights.
- Generate `rust/crates/feathertalk-scrfd/artifacts/scrfd_2_5g/manifest.json`: committed strict artifact metadata.
- Create `rust/crates/feathertalk-scrfd/tests/manifest_contract.rs`: schema and public value-type contract.
- Create `rust/crates/feathertalk-scrfd/tests/artifact.rs`: successful load, immutable hash verification, corruption, and input-shape failures.
- Create `rust/crates/feathertalk-scrfd/tests/fixture_contract.rs`: fixture metadata, hashes, shapes, and synthetic-input reconstruction.
- Create `rust/crates/feathertalk-scrfd/tests/parity.rs`: all-nine-output NdArray/OpenCV comparison.
- Create `rust/crates/feathertalk-scrfd/tests/wgpu_smoke.rs`: ignored WGPU load/forward/shape test.
- Create `rust/crates/feathertalk-scrfd/tests/support/mod.rs`: test-only paths, NPY loading, hashing, and metrics helpers.
- Generate `rust/crates/feathertalk-scrfd/tests/fixtures/opencv_cpu_v1/{fixture.json,input.npy,out0.npy,...,out8.npy}`.
- Create `rust/tools/scrfd-import/Cargo.toml` and `Cargo.lock`: standalone pinned Rust tool.
- Create `rust/tools/scrfd-import/build.rs`: conditionally compile the converter against a staged generated model.
- Create `rust/tools/scrfd-import/src/lib.rs`: shared source inspection, hashing, generation, publication, and tree comparison helpers.
- Create `rust/tools/scrfd-import/src/onnx.rs`: minimal prost messages and the fixed ONNX contract parser.
- Create `rust/tools/scrfd-import/src/artifact.rs`: strict burnpack/safetensors conversion and deterministic manifest generation.
- Create `rust/tools/scrfd-import/src/bin/generate.rs`: one-command generation or byte-for-byte verification orchestration.
- Create `rust/tools/scrfd-import/src/bin/convert.rs`: staged generated-source compilation and artifact conversion.
- Create `rust/tools/scrfd-import/tests/source_contract.rs`: real ONNX metadata and raw generation smoke tests.
- Create `rust/tools/scrfd-import/python/requirements-fixture.txt`: exact NumPy/OpenCV versions.
- Create `rust/tools/scrfd-import/python/generate_fixture.py`: deterministic OpenCV fixture generation and verification.
- Create `rust/tools/scrfd-import/.gitignore`: ignore only the local `.venv/`.
- Create `rust/tools/scrfd-import/README.md`: exact regeneration, verification, and licensing commands.

---

### Task 1: Define the runtime manifest and error contract

**Files:**
- Modify: `rust/Cargo.toml`
- Create: `rust/crates/feathertalk-scrfd/Cargo.toml`
- Create: `rust/crates/feathertalk-scrfd/src/lib.rs`
- Create: `rust/crates/feathertalk-scrfd/src/error.rs`
- Create: `rust/crates/feathertalk-scrfd/src/manifest.rs`
- Create: `rust/crates/feathertalk-scrfd/tests/manifest_contract.rs`
- Modify: `rust/Cargo.lock`

**Interfaces:**
- Produces constants `SCRFD_SCHEMA_VERSION`, `SCRFD_ARCHITECTURE_VERSION`, `SCRFD_MODEL_KIND`, `SCRFD_SOURCE_ONNX_BYTES`, `SCRFD_SOURCE_ONNX_SHA256`, `SCRFD_SOURCE_OPSET`, `SCRFD_INPUT_SHAPE`, `SCRFD_STRIDES`, and `SCRFD_ANCHORS`.
- Produces public serializable types `ScrfdArtifactManifest`, `ScrfdSourceManifest`, `ScrfdGeneratorManifest`, `ScrfdInputManifest`, `ScrfdLevelManifest`, `ScrfdOutputManifest`, `ScrfdFileManifest`, `ScrfdWeightManifest`, and `ScrfdLicenseManifest`.
- Produces `ScrfdArtifactManifest::validate(&self) -> Result<(), ScrfdError>` for structural/fixed-contract validation; compiled artifact hashes are checked later by `artifact.rs`.
- Produces the complete `ScrfdError` enum used by all later tasks.

- [ ] **Step 1: Add the workspace member and failing public-contract test**

Add `crates/feathertalk-scrfd` to `members` and `tools/scrfd-import` to a new workspace `exclude` list. Create `tests/manifest_contract.rs` with this fixture and assertions:

```rust
use feathertalk_scrfd::{
    SCRFD_ANCHORS, SCRFD_ARCHITECTURE_VERSION, SCRFD_INPUT_SHAPE, SCRFD_MODEL_KIND,
    SCRFD_SCHEMA_VERSION, SCRFD_SOURCE_ONNX_BYTES, SCRFD_SOURCE_ONNX_SHA256,
    SCRFD_SOURCE_OPSET, SCRFD_STRIDES, ScrfdArtifactManifest, ScrfdFileManifest,
    ScrfdGeneratorManifest, ScrfdInputManifest, ScrfdLevelManifest, ScrfdLicenseManifest,
    ScrfdError, ScrfdOutputManifest, ScrfdSourceManifest, ScrfdWeightManifest,
};

fn output(name: &str, source: &[usize], public: &[usize]) -> ScrfdOutputManifest {
    ScrfdOutputManifest {
        onnx_name: name.to_owned(),
        source_shape: source.to_vec(),
        public_shape: public.to_vec(),
    }
}

fn valid_manifest() -> ScrfdArtifactManifest {
    ScrfdArtifactManifest {
        schema_version: 1,
        model_kind: "scrfd_2.5g_kps".to_owned(),
        architecture_version: 1,
        source: ScrfdSourceManifest {
            format: "onnx".to_owned(),
            file_name: "scrfd_2.5g_kps.onnx".to_owned(),
            file_bytes: 3_291_017,
            sha256: "32d20c77b9e2dc1d07e94c2ab9d25bdd5cd05eddbe0b46e7b38e7a1eca22e99a"
                .to_owned(),
            opset: 12,
            input_name: "images".to_owned(),
            output_names: [
                "out0", "out1", "out2", "out3", "out4", "out5", "out6", "out7",
                "out8",
            ]
            .map(str::to_owned),
        },
        generator: ScrfdGeneratorManifest {
            burn: "0.21.0".to_owned(),
            burn_onnx: "0.21.0".to_owned(),
            burn_store: "0.21.0".to_owned(),
            simplify: true,
            load_strategy: "none".to_owned(),
        },
        input: ScrfdInputManifest {
            dtype: "float32".to_owned(),
            shape: [1, 3, 640, 640],
            scale: 1.0 / 128.0,
            mean: [127.5; 3],
            swap_rb: true,
        },
        levels: [
            ScrfdLevelManifest {
                stride: 8,
                anchors: 12_800,
                score: output("out0", &[1, 12_800, 1], &[1, 12_800]),
                bbox: output("out3", &[1, 12_800, 4], &[1, 12_800, 4]),
                keypoints: output("out6", &[1, 12_800, 10], &[1, 12_800, 10]),
            },
            ScrfdLevelManifest {
                stride: 16,
                anchors: 3_200,
                score: output("out1", &[1, 3_200, 1], &[1, 3_200]),
                bbox: output("out4", &[1, 3_200, 4], &[1, 3_200, 4]),
                keypoints: output("out7", &[1, 3_200, 10], &[1, 3_200, 10]),
            },
            ScrfdLevelManifest {
                stride: 32,
                anchors: 800,
                score: output("out2", &[1, 800, 1], &[1, 800]),
                bbox: output("out5", &[1, 800, 4], &[1, 800, 4]),
                keypoints: output("out8", &[1, 800, 10], &[1, 800, 10]),
            },
        ],
        generated_source: ScrfdFileManifest {
            file_name: "scrfd_2_5g.rs".to_owned(),
            file_bytes: 123,
            sha256: "a".repeat(64),
        },
        weights: ScrfdWeightManifest {
            format: "safetensors".to_owned(),
            file_name: "model.safetensors".to_owned(),
            file_bytes: 456,
            sha256: "b".repeat(64),
        },
        license: ScrfdLicenseManifest {
            license_id: "NOASSERTION".to_owned(),
            redistribution_approved: false,
            evidence: "repository does not provide a verifiable model-weight license".to_owned(),
        },
    }
}

#[test]
fn fixed_constants_match_the_approved_source() {
    assert_eq!(SCRFD_SCHEMA_VERSION, 1);
    assert_eq!(SCRFD_ARCHITECTURE_VERSION, 1);
    assert_eq!(SCRFD_MODEL_KIND, "scrfd_2.5g_kps");
    assert_eq!(SCRFD_SOURCE_ONNX_BYTES, 3_291_017);
    assert_eq!(SCRFD_SOURCE_ONNX_SHA256.len(), 64);
    assert_eq!(SCRFD_SOURCE_OPSET, 12);
    assert_eq!(SCRFD_INPUT_SHAPE, [1, 3, 640, 640]);
    assert_eq!(SCRFD_STRIDES, [8, 16, 32]);
    assert_eq!(SCRFD_ANCHORS, [12_800, 3_200, 800]);
}

#[test]
fn schema_one_manifest_round_trips_and_validates() {
    let manifest = valid_manifest();
    manifest.validate().unwrap();
    let json = serde_json::to_string_pretty(&manifest).unwrap();
    assert_eq!(serde_json::from_str::<ScrfdArtifactManifest>(&json).unwrap(), manifest);
}

fn assert_invalid_field(manifest: ScrfdArtifactManifest, field: &str) {
    match manifest.validate().unwrap_err() {
        ScrfdError::InvalidManifest { field: actual, .. } => assert_eq!(actual, field),
        error => panic!("expected InvalidManifest for {field}, got {error}"),
    }
}

#[test]
fn unknown_fields_and_changed_output_mapping_are_rejected() {
    let manifest = valid_manifest();
    let mut value = serde_json::to_value(&manifest).unwrap();
    value["future_field"] = serde_json::json!(true);
    assert!(serde_json::from_value::<ScrfdArtifactManifest>(value).is_err());

    let mut changed = manifest;
    changed.levels[1].bbox.onnx_name = "out5".to_owned();
    assert!(changed.validate().is_err());
}

#[test]
fn unsupported_manifest_and_architecture_versions_are_distinct() {
    let mut manifest = valid_manifest();
    manifest.schema_version = 2;
    assert!(matches!(
        manifest.validate(),
        Err(ScrfdError::UnsupportedSchemaVersion { expected: 1, actual: 2 })
    ));

    let mut manifest = valid_manifest();
    manifest.architecture_version = 2;
    assert!(matches!(
        manifest.validate(),
        Err(ScrfdError::UnsupportedArchitectureVersion { expected: 1, actual: 2 })
    ));
}

#[test]
fn zero_sizes_dimensions_and_malformed_hashes_are_rejected() {
    let mut manifest = valid_manifest();
    manifest.generated_source.file_bytes = 0;
    assert_invalid_field(manifest, "generated_source.file_bytes");

    let mut manifest = valid_manifest();
    manifest.weights.sha256 = "A".repeat(64);
    assert_invalid_field(manifest, "weights.sha256");

    let mut manifest = valid_manifest();
    manifest.levels[0].score.source_shape[1] = 0;
    assert_invalid_field(manifest, "levels[0].score.source_shape");

    let mut manifest = valid_manifest();
    manifest.source.opset = 0;
    assert_invalid_field(manifest, "source.opset");
}
```

- [ ] **Step 2: Run the contract test and verify the red result**

Run: `cargo test -p feathertalk-scrfd --test manifest_contract`

Expected: Cargo reports that the `feathertalk-scrfd` package/manifest does not yet exist.

- [ ] **Step 3: Create the crate manifest and complete error enum**

Use this crate manifest:

```toml
[package]
name = "feathertalk-scrfd"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[features]
default = ["wgpu"]
wgpu = []
metal = ["burn/metal"]
vulkan = ["burn/vulkan"]

[dependencies]
burn.workspace = true
burn-store.workspace = true
hex.workspace = true
serde.workspace = true
serde_json.workspace = true
sha2.workspace = true
thiserror.workspace = true

[dev-dependencies]
ndarray.workspace = true
ndarray-npy.workspace = true
tempfile.workspace = true
```

Define these stable error categories in `src/error.rs`:

```rust
use std::{io, path::PathBuf};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ScrfdError {
    #[error("I/O error during {operation} at {}: {source}", path.display())]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("manifest exceeds {limit} bytes: {actual}")]
    ManifestTooLarge { limit: u64, actual: u64 },
    #[error("weights exceed {limit} bytes: {actual}")]
    WeightsTooLarge { limit: u64, actual: u64 },
    #[error("manifest JSON error: {0}")]
    ManifestJson(String),
    #[error("unsupported manifest schema version: expected {expected}, got {actual}")]
    UnsupportedSchemaVersion { expected: u32, actual: u32 },
    #[error("unsupported SCRFD architecture version: expected {expected}, got {actual}")]
    UnsupportedArchitectureVersion { expected: u32, actual: u32 },
    #[error("invalid manifest field {field}: {message}")]
    InvalidManifest {
        field: String,
        message: String,
    },
    #[error("artifact contract mismatch for {field}: expected {expected}, got {actual}")]
    ContractMismatch {
        field: &'static str,
        expected: String,
        actual: String,
    },
    #[error("weight byte count mismatch: expected {expected}, got {actual}")]
    WeightSizeMismatch { expected: u64, actual: u64 },
    #[error("SHA-256 mismatch for {artifact}: expected {expected}, got {actual}")]
    HashMismatch {
        artifact: &'static str,
        expected: String,
        actual: String,
    },
    #[error("Burn store error: {0}")]
    Store(String),
    #[error("missing tensor: {0}")]
    MissingTensor(String),
    #[error("unexpected tensor: {0}")]
    UnexpectedTensor(String),
    #[error("tensor shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("tensor dtype mismatch: {0}")]
    DTypeMismatch(String),
    #[error("invalid SCRFD input shape: expected [1, 3, 640, 640], got {actual:?}")]
    InvalidInputShape { actual: [usize; 4] },
    #[error("invalid SCRFD output shape for {name}: expected {expected:?}, got {actual:?}")]
    InvalidOutputShape {
        name: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
}
```

- [ ] **Step 4: Implement exact manifest value types and validation**

All manifest structs derive `Debug, Clone, PartialEq, Serialize, Deserialize` and carry `#[serde(deny_unknown_fields)]`. Use `[String; 9]` for output names, `[ScrfdLevelManifest; 3]` for levels, `[usize; 4]` for input shape, and `Vec<usize>` for source/public output shapes.

Define their fields exactly:

```rust
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScrfdArtifactManifest {
    pub schema_version: u32,
    pub model_kind: String,
    pub architecture_version: u32,
    pub source: ScrfdSourceManifest,
    pub generator: ScrfdGeneratorManifest,
    pub input: ScrfdInputManifest,
    pub levels: [ScrfdLevelManifest; 3],
    pub generated_source: ScrfdFileManifest,
    pub weights: ScrfdWeightManifest,
    pub license: ScrfdLicenseManifest,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScrfdSourceManifest {
    pub format: String,
    pub file_name: String,
    pub file_bytes: u64,
    pub sha256: String,
    pub opset: u64,
    pub input_name: String,
    pub output_names: [String; 9],
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScrfdGeneratorManifest {
    pub burn: String,
    pub burn_onnx: String,
    pub burn_store: String,
    pub simplify: bool,
    pub load_strategy: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScrfdInputManifest {
    pub dtype: String,
    pub shape: [usize; 4],
    pub scale: f32,
    pub mean: [f32; 3],
    pub swap_rb: bool,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScrfdLevelManifest {
    pub stride: u32,
    pub anchors: usize,
    pub score: ScrfdOutputManifest,
    pub bbox: ScrfdOutputManifest,
    pub keypoints: ScrfdOutputManifest,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScrfdOutputManifest {
    pub onnx_name: String,
    pub source_shape: Vec<usize>,
    pub public_shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScrfdFileManifest {
    pub file_name: String,
    pub file_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScrfdWeightManifest {
    pub format: String,
    pub file_name: String,
    pub file_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScrfdLicenseManifest {
    pub license_id: String,
    pub redistribution_approved: bool,
    pub evidence: String,
}
```

Define the fixed expected level table and validation helpers:

```rust
pub const SCRFD_SCHEMA_VERSION: u32 = 1;
pub const SCRFD_ARCHITECTURE_VERSION: u32 = 1;
pub const SCRFD_MODEL_KIND: &str = "scrfd_2.5g_kps";
pub const SCRFD_SOURCE_ONNX_BYTES: u64 = 3_291_017;
pub const SCRFD_SOURCE_ONNX_SHA256: &str =
    "32d20c77b9e2dc1d07e94c2ab9d25bdd5cd05eddbe0b46e7b38e7a1eca22e99a";
pub const SCRFD_SOURCE_OPSET: u64 = 12;
pub const SCRFD_INPUT_SHAPE: [usize; 4] = [1, 3, 640, 640];
pub const SCRFD_STRIDES: [u32; 3] = [8, 16, 32];
pub const SCRFD_ANCHORS: [usize; 3] = [12_800, 3_200, 800];

const OUTPUT_NAMES: [&str; 9] = [
    "out0", "out1", "out2", "out3", "out4", "out5", "out6", "out7", "out8",
];

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
```

`ScrfdArtifactManifest::validate` must return `UnsupportedSchemaVersion` or `UnsupportedArchitectureVersion` for the two version fields before checking the remaining literals; it then checks every literal in the test fixture, requires positive source/generated/weight sizes, requires lowercase 64-character hashes, requires a nonempty license evidence string, and compares the three levels against this exact mapping. For `input.scale`, compare with `1.0 / 128.0` using `to_bits()` so the serialized binary32 value is exact; for `input.mean`, compare each element with `127.5` using `to_bits()`.

```rust
let expected = [
    (8, 12_800, "out0", "out3", "out6"),
    (16, 3_200, "out1", "out4", "out7"),
    (32, 800, "out2", "out5", "out8"),
];
```

For each `(stride, anchors, score, bbox, keypoints)`, require score source/public shapes `[1, anchors, 1]`/`[1, anchors]`, bbox `[1, anchors, 4]` for both, and keypoints `[1, anchors, 10]` for both. Return `ScrfdError::InvalidManifest { field: field_path.to_owned(), message }` with the precise field path on the first mismatch.

Validation must also reject zero dimensions in every `source_shape` and `public_shape`, reject a source opset of zero, reject empty names/evidence, and require `license.redistribution_approved == false`.

- [ ] **Step 5: Export the contract and run focused green verification**

`src/lib.rs` initially contains only:

```rust
//! SCRFD Burn model artifact and raw inference contract.

mod error;
mod manifest;

pub use error::ScrfdError;
pub use manifest::{
    SCRFD_ANCHORS, SCRFD_ARCHITECTURE_VERSION, SCRFD_INPUT_SHAPE, SCRFD_MODEL_KIND,
    SCRFD_SCHEMA_VERSION, SCRFD_SOURCE_ONNX_BYTES, SCRFD_SOURCE_ONNX_SHA256,
    SCRFD_SOURCE_OPSET, SCRFD_STRIDES, ScrfdArtifactManifest, ScrfdFileManifest,
    ScrfdGeneratorManifest, ScrfdInputManifest, ScrfdLevelManifest, ScrfdLicenseManifest,
    ScrfdOutputManifest, ScrfdSourceManifest, ScrfdWeightManifest,
};
```

Run:

```powershell
cargo fmt --check
cargo test -p feathertalk-scrfd --test manifest_contract
cargo test -p feathertalk-scrfd --all-targets
```

Expected: all manifest tests pass and no model source/artifact is required yet.

- [ ] **Step 6: Commit the runtime contract**

```powershell
git add rust/Cargo.toml rust/Cargo.lock rust/crates/feathertalk-scrfd
git commit -m "feat: define SCRFD artifact contract"
```

---

### Task 2: Validate the ONNX source and generate deterministic Burn files

**Files:**
- Create: `rust/tools/scrfd-import/Cargo.toml`
- Create: `rust/tools/scrfd-import/src/lib.rs`
- Create: `rust/tools/scrfd-import/src/onnx.rs`
- Create: `rust/tools/scrfd-import/tests/source_contract.rs`
- Create: `rust/tools/scrfd-import/Cargo.lock`

**Interfaces:**
- Produces `inspect_source(repo_root: &Path) -> Result<OnnxContract, ToolError>`.
- Produces `generate_burn_files(repo_root: &Path, destination: &Path) -> Result<GeneratedBurnFiles, ToolError>`.
- `OnnxContract` contains the fixed opset, input name/shape/dtype, and ordered output names/shapes.
- `GeneratedBurnFiles` returns exact published paths to the normalized `scrfd_2_5g.rs` source and temporary `scrfd_2.5g_kps.bpk` burnpack.

- [ ] **Step 1: Create the standalone manifest and failing source-contract test**

Use a standalone package with an empty `[workspace]` table and exact dependencies:

```toml
[package]
name = "feathertalk-scrfd-import"
version = "0.1.0"
edition = "2024"
rust-version = "1.92"
license = "Apache-2.0"

[workspace]

[dependencies]
burn-onnx = { version = "=0.21.0", default-features = false, features = ["mmap"] }
clap = { version = "=4.6.6", features = ["derive"] }
feathertalk-scrfd = { path = "../../crates/feathertalk-scrfd", default-features = false }
hex = "=0.4.3"
prost = "=0.14.4"
serde_json = "=1.0.151"
sha2 = "=0.11.0"
tempfile = "=3.27.0"
thiserror = "=2.0.20"
```

Create `tests/source_contract.rs`:

```rust
use std::path::{Path, PathBuf};

use feathertalk_scrfd_import::{generate_burn_files, inspect_source};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

#[test]
fn tracked_onnx_has_the_approved_graph_boundary() {
    let contract = inspect_source(&repo_root()).unwrap();
    assert_eq!(contract.opset, 12);
    assert_eq!(contract.input_name, "images");
    assert_eq!(contract.input_shape, vec![1, 3, 640, 640]);
    assert_eq!(contract.input_elem_type, 1);
    assert_eq!(
        contract.output_names.iter().map(String::as_str).collect::<Vec<_>>(),
        vec!["out0", "out1", "out2", "out3", "out4", "out5", "out6", "out7", "out8"]
    );
    assert_eq!(contract.output_shapes, vec![
        vec![1, 12_800, 1],
        vec![1, 3_200, 1],
        vec![1, 800, 1],
        vec![1, 12_800, 4],
        vec![1, 3_200, 4],
        vec![1, 800, 4],
        vec![1, 12_800, 10],
        vec![1, 3_200, 10],
        vec![1, 800, 10],
    ]);
}

#[test]
#[ignore = "runs pinned Burn ONNX code generation"]
fn burn_generation_writes_only_reviewable_source_and_temporary_burnpack() {
    let temp = tempfile::tempdir().unwrap();
    let destination = temp.path().join("raw");
    let generated = generate_burn_files(&repo_root(), &destination).unwrap();
    assert!(generated.source.is_file());
    assert!(generated.burnpack.is_file());
    let mut names = std::fs::read_dir(&destination)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    names.sort();
    assert_eq!(names, vec!["scrfd_2.5g_kps.bpk", "scrfd_2_5g.rs"]);
    let source = std::fs::read_to_string(generated.source).unwrap();
    assert!(source.contains("pub struct Model<B: Backend>"));
    assert!(source.contains("pub fn forward"));
    assert!(!source.contains("from_file"));
    assert!(!source.contains("from_bytes"));
}
```

- [ ] **Step 2: Run the source-contract test and verify the red result**

Run:

```powershell
cargo test --manifest-path tools/scrfd-import/Cargo.toml --test source_contract
```

Expected: compilation fails because the importer library and ONNX parser do not exist. If Cargo needs uncached crates, rerun with the required network approval; do not loosen any pinned version.

- [ ] **Step 3: Implement a minimal read-only ONNX protobuf boundary**

In `src/onnx.rs`, define only the protobuf fields needed for source validation:

```rust
use prost::Message;

#[derive(Clone, PartialEq, prost::Message)]
struct ModelProto {
    #[prost(message, optional, tag = "7")]
    graph: Option<GraphProto>,
    #[prost(message, repeated, tag = "8")]
    opset_import: Vec<OperatorSetIdProto>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct OperatorSetIdProto {
    #[prost(string, tag = "1")]
    domain: String,
    #[prost(int64, tag = "2")]
    version: i64,
}

#[derive(Clone, PartialEq, prost::Message)]
struct GraphProto {
    #[prost(message, repeated, tag = "11")]
    input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    output: Vec<ValueInfoProto>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct ValueInfoProto {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(message, optional, tag = "2")]
    r#type: Option<TypeProto>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct TypeProto {
    #[prost(message, optional, tag = "1")]
    tensor_type: Option<TensorTypeProto>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct TensorTypeProto {
    #[prost(int32, tag = "1")]
    elem_type: i32,
    #[prost(message, optional, tag = "2")]
    shape: Option<TensorShapeProto>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    dim: Vec<DimensionProto>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct DimensionProto {
    #[prost(int64, optional, tag = "1")]
    dim_value: Option<i64>,
    #[prost(string, optional, tag = "2")]
    dim_param: Option<String>,
}
```

Decode with `prost::Message::decode`, accept exactly one empty/default-domain opset import with version 12, require exactly one input named `images`, reject symbolic/non-positive dimensions, require ONNX tensor element type `1` (`FLOAT`) for the input and every output, and compare all nine ordered outputs against the Global Constraints.

- [ ] **Step 4: Implement bounded hashing, source inspection, and deterministic ModelGen invocation**

In `src/lib.rs`, define:

```rust
pub const SOURCE_RELATIVE_PATH: &str = "data_utils/scrfd_2.5g_kps.onnx";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnnxContract {
    pub opset: u64,
    pub input_name: String,
    pub input_elem_type: i32,
    pub input_shape: Vec<usize>,
    pub output_names: [String; 9],
    pub output_shapes: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedBurnFiles {
    pub source: std::path::PathBuf,
    pub burnpack: std::path::PathBuf,
}
```

Define the shared error type in the same file so every later reference is concrete:

```rust
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("I/O error during {operation} at {}: {source}", path.display())]
    Io {
        operation: &'static str,
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("source contract error: {0}")]
    SourceContract(String),
    #[error("ONNX protobuf decode error: {0}")]
    OnnxDecode(String),
    #[error("destination already exists: {}", .0.display())]
    DestinationExists(std::path::PathBuf),
    #[error("path is not valid UTF-8: {}", .0.display())]
    NonUtf8Path(std::path::PathBuf),
    #[error("Burn ONNX generation failed: {0}")]
    Generation(String),
    #[error("Burn store error: {0}")]
    Store(String),
    #[error("snapshot comparison failed: {0}")]
    Snapshot(String),
    #[error("manifest error: {0}")]
    Manifest(String),
    #[error("converter process failed with status {status:?}: {stderr}")]
    ConversionProcess { status: Option<i32>, stderr: String },
    #[error("generated tree differs at {}", .0.display())]
    TreeMismatch(std::path::PathBuf),
}
```

`inspect_source` must use one opened file, enforce the exact byte count before and after reading, hash the same bytes, compare the fixed SHA-256, then pass those bytes to the prost parser.

`generate_burn_files` must resolve an absolute destination before changing the current directory, reject an existing destination with `symlink_metadata`, create `staged_output` beneath a `TempDir` in the destination parent, temporarily set the current directory to `repo_root`, and invoke:

```rust
let staged_utf8 = staged_output
    .to_str()
    .ok_or_else(|| ToolError::NonUtf8Path(staged_output.clone()))?;
let mut generator = burn_onnx::ModelGen::new();
generator
    .input(SOURCE_RELATIVE_PATH)
    .out_dir(staged_utf8)
    .development(false)
    .simplify(true)
    .partition(true)
    .load_strategy(burn_onnx::LoadStrategy::None);
generator.run_from_cli();
```

Use an RAII current-directory guard so the original directory is restored after both success and panic. Wrap only `generator.run_from_cli()` in `catch_unwind(AssertUnwindSafe(...))`, map a panic to `ToolError::Generation`, and let the staging `TempDir` clean up on failure. Because Burn's `Path::file_stem` uses the last dot as the extension separator, the successful generator emits exactly `scrfd_2.rs` and `scrfd_2.bpk`; rename those inside staging to `scrfd_2_5g.rs` and `scrfd_2.5g_kps.bpk`, require exactly those two final files, rename `staged_output` to the still-absent destination, drop the now-empty `TempDir`, and return paths rooted at the published destination.

- [ ] **Step 5: Run source inspection and explicit generation smoke tests**

Run:

```powershell
cargo fmt --manifest-path tools/scrfd-import/Cargo.toml --check
cargo test --manifest-path tools/scrfd-import/Cargo.toml --test source_contract
cargo test --manifest-path tools/scrfd-import/Cargo.toml --test source_contract -- --ignored --nocapture
```

Expected: the real ONNX metadata test passes; the ignored generation test creates source and burnpack only under `TempDir` and passes.

- [ ] **Step 6: Commit the source validator and generator**

```powershell
git add rust/Cargo.toml rust/tools/scrfd-import
git commit -m "feat: add SCRFD ONNX generator"
```

---

### Task 3: Convert burnpack state and commit reproducible model artifacts

**Files:**
- Create: `rust/tools/scrfd-import/build.rs`
- Create: `rust/tools/scrfd-import/src/artifact.rs`
- Create: `rust/tools/scrfd-import/src/bin/generate.rs`
- Create: `rust/tools/scrfd-import/src/bin/convert.rs`
- Modify: `rust/tools/scrfd-import/src/lib.rs`
- Modify: `rust/tools/scrfd-import/Cargo.toml`
- Create: `rust/tools/scrfd-import/README.md`
- Generate: `rust/crates/feathertalk-scrfd/src/generated/scrfd_2_5g.rs`
- Generate: `rust/crates/feathertalk-scrfd/src/generated/artifact_contract.rs`
- Generate: `rust/crates/feathertalk-scrfd/artifacts/scrfd_2_5g/model.safetensors`
- Generate: `rust/crates/feathertalk-scrfd/artifacts/scrfd_2_5g/manifest.json`
- Modify: `rust/tools/scrfd-import/Cargo.lock`

**Interfaces:**
- Produces `convert_burnpack(burnpack: &Path, safetensors: &Path) -> Result<(), ToolError>` for the generated NdArray model, plus generic snapshot-comparison helpers.
- Produces a `generate` CLI with mutually exclusive `--destination` and `--verify-against` modes.
- Produces generated constants `GENERATED_SOURCE_BYTES`, `GENERATED_SOURCE_SHA256`, `MODEL_SAFETENSORS_BYTES`, and `MODEL_SAFETENSORS_SHA256`.
- Produces the committed private Burn model source and artifact directory consumed by Task 4.

- [ ] **Step 1: Add failing strict-apply and publication tests**

In `src/artifact.rs`, first add unit tests that construct `burn_store::ApplyResult` values and require stable errors for every strict failure:

```rust
#[test]
fn strict_apply_rejects_every_non_applied_entry() {
    use burn::tensor::{DType, Shape};
    use burn_store::{ApplyError, ApplyResult};

    let empty = || ApplyResult {
        applied: Vec::new(),
        skipped: Vec::new(),
        missing: Vec::new(),
        unused: Vec::new(),
        errors: Vec::new(),
    };
    let mut missing = empty();
    missing.missing.push(("conv.weight".to_owned(), "Struct:Model".to_owned()));
    let mut unused = empty();
    unused.unused.push("extra.weight".to_owned());
    let mut skipped = empty();
    skipped.skipped.push("head.bias".to_owned());
    let mut shape = empty();
    shape.errors.push(ApplyError::ShapeMismatch {
        path: "neck.weight".to_owned(),
        expected: Shape::new([1, 2]),
        found: Shape::new([2, 1]),
    });
    let mut dtype = empty();
    dtype.errors.push(ApplyError::DTypeMismatch {
        path: "score.bias".to_owned(),
        expected: DType::F32,
        found: DType::I32,
    });
    let mut adapter = empty();
    adapter.errors.push(ApplyError::AdapterError {
        path: "neck.weight".to_owned(),
        message: "adapter failed".to_owned(),
    });
    let mut load = empty();
    load.errors.push(ApplyError::LoadError {
        path: "head.weight".to_owned(),
        message: "load failed".to_owned(),
    });

    let cases = [missing, unused, skipped, shape, dtype, adapter, load];
    for result in cases {
        assert!(validate_apply_result(&result).is_err());
    }
}

#[test]
fn publishing_rejects_an_existing_file_or_directory() {
    let temp = tempfile::tempdir().unwrap();
    for name in ["file", "directory"] {
        let path = temp.path().join(name);
        if name == "file" {
            std::fs::write(&path, b"occupied").unwrap();
        } else {
            std::fs::create_dir(&path).unwrap();
        }
        assert!(ensure_destination_absent(&path).is_err());
    }
}
```

Rename the second test to `publishing_rejects_an_existing_file_or_directory`; its two mandatory cases are sufficient to prove `symlink_metadata` treats an occupied path as existing without adding platform-privileged test setup.

- [ ] **Step 2: Run importer tests and verify the red result**

Run: `cargo test --manifest-path tools/scrfd-import/Cargo.toml artifact::tests`

Expected: compilation fails because the artifact module, Burn dependencies, and validation helpers are absent.

- [ ] **Step 3: Add exact conversion dependencies and conditional generated-source build**

Add:

```toml
burn = { version = "=0.21.0", default-features = false, features = ["std", "ndarray", "store"] }
burn-store = { version = "=0.21.0", default-features = false, features = ["std", "burnpack", "safetensors"] }
```

`build.rs` must allow normal tests to compile without a staged model while enabling the real converter only when requested:

```rust
fn main() {
    println!("cargo:rustc-check-cfg=cfg(scrfd_generated)");
    println!("cargo:rerun-if-env-changed=SCRFD_GENERATED_RS");
    if let Ok(source) = std::env::var("SCRFD_GENERATED_RS") {
        println!("cargo:rerun-if-changed={source}");
        let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap())
            .join("scrfd_generated.rs");
        std::fs::copy(source, out).unwrap();
        println!("cargo:rustc-cfg=scrfd_generated");
    }
}
```

In `convert.rs`, keep every import and real function under `#[cfg(scrfd_generated)]`. The no-model `main` prints `SCRFD_GENERATED_RS is required` to stderr and exits with code 2. The real module is:

```rust
#[cfg(scrfd_generated)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/scrfd_generated.rs"));
}

#[cfg(scrfd_generated)]
#[derive(clap::Parser)]
struct Args {
    #[arg(long)]
    burnpack: std::path::PathBuf,
    #[arg(long)]
    safetensors: std::path::PathBuf,
}
```

- [ ] **Step 4: Implement strict burnpack-to-safetensors conversion**

Use `type CpuBackend = burn::backend::NdArray<f32>`. The real converter must:

1. Construct `generated::Model::<CpuBackend>::new(&device)`.
2. Load the staged `.bpk` with `BurnpackStore::from_file` and call `model.load_from(&mut store)`.
3. Validate no missing, unused, skipped, shape, dtype, adapter, or load errors.
4. Save with `SafetensorsStore::from_file(path).overwrite(false)`.
5. Read the resulting bytes once, load a fresh generated model with `SafetensorsStore::from_bytes(Some(bytes.clone())).allow_partial(true).validate(false)`, and validate strictly. The store owns `bytes`; do not read the output file again during this check.
6. Compare a `BTreeMap<String, TensorData>` keyed by `snapshot.full_path()` for the burnpack-loaded and safetensors-reloaded models. Require identical keys, shapes, dtypes, and tensor data.

The generated `main` parses `Args`, rejects an existing safetensors destination with `symlink_metadata`, calls `convert_burnpack(&args.burnpack, &args.safetensors)`, prints a path-specific error and exits 1 on failure, and exits 0 only after the reloaded snapshot comparison passes.

Use these helper signatures:

```rust
pub fn convert_burnpack(
    burnpack: &std::path::Path,
    safetensors: &std::path::Path,
) -> Result<(), ToolError>;

pub fn validate_apply_result(result: &burn_store::ApplyResult) -> Result<(), ToolError>;

pub fn snapshot_map<B, M>(
    module: &M,
) -> Result<std::collections::BTreeMap<String, burn_store::TensorSnapshot>, ToolError>
where
    B: burn::tensor::backend::Backend,
    M: burn_store::ModuleSnapshot<B>;

pub fn compare_snapshots<B, M>(expected: &M, actual: &M) -> Result<(), ToolError>
where
    B: burn::tensor::backend::Backend,
    M: burn_store::ModuleSnapshot<B>;
```

Implement `snapshot_map` by calling `module.collect(None, None, false)`, inserting `snapshot.full_path()` keys, and returning `ToolError::Snapshot` for a duplicate key. `compare_snapshots` compares sorted key sets first, then each `TensorSnapshot` shape and dtype, materializes both through `to_data()`, and requires exact `TensorData` equality. It must not use floating-point tolerances because this is a serialization round-trip check.

Define the shared publication guard and strict apply-result mapping exactly once in `src/artifact.rs`:

```rust
pub fn ensure_destination_absent(path: &std::path::Path) -> Result<(), ToolError> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Err(ToolError::DestinationExists(path.to_owned())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(ToolError::Io {
            operation: "inspect destination",
            path: path.to_owned(),
            source,
        }),
    }
}

pub fn validate_apply_result(result: &burn_store::ApplyResult) -> Result<(), ToolError> {
    if let Some(path) = result.missing.iter().map(|(path, _)| path).min() {
        return Err(ToolError::Store(format!("missing tensor: {path}")));
    }
    if let Some(error) = result.errors.first() {
        return Err(ToolError::Store(error.to_string()));
    }
    if let Some(path) = result.skipped.iter().min() {
        return Err(ToolError::Store(format!("skipped tensor: {path}")));
    }
    if let Some(path) = result.unused.iter().min() {
        return Err(ToolError::Store(format!("unexpected tensor: {path}")));
    }
    Ok(())
}
```

Use `ensure_destination_absent` for generated directories and the converter's safetensors output. Both helpers must be covered by the Step 1 unit tests.

- [ ] **Step 5: Implement deterministic final tree and generated constants**

The converter writes into a `TempDir` under the requested destination parent and then renames the complete directory. The staged tree is exactly:

```text
src/generated/scrfd_2_5g.rs
src/generated/artifact_contract.rs
artifacts/scrfd_2_5g/model.safetensors
artifacts/scrfd_2_5g/manifest.json
```

`artifact_contract.rs` is generated with this exact formatter and a trailing newline:

```rust
let constants = format!(
    "pub(crate) const GENERATED_SOURCE_BYTES: u64 = {source_bytes};\n\
     pub(crate) const GENERATED_SOURCE_SHA256: &str = \"{source_sha256}\";\n\
     pub(crate) const MODEL_SAFETENSORS_BYTES: u64 = {weight_bytes};\n\
     pub(crate) const MODEL_SAFETENSORS_SHA256: &str = \"{weight_sha256}\";\n"
);
```

Build `ScrfdArtifactManifest` using the exact source/generator/input/level/license values from Task 1, the real generated-source/weight sizes and hashes, and `source.opset = 12`. Call `manifest.validate()` before serialization. Serialize with `serde_json::to_vec_pretty`, append one newline, use create-new writes, sync each file, require exactly the four listed files, and then publish by same-filesystem rename.

- [ ] **Step 6: Implement one-command generation and verification orchestration**

`generate.rs` uses this exact argument model:

```rust
#[derive(clap::Parser)]
struct Args {
    #[arg(long)]
    repo_root: std::path::PathBuf,
    #[arg(long, conflicts_with = "verify_against", required_unless_present = "verify_against")]
    destination: Option<std::path::PathBuf>,
    #[arg(long, conflicts_with = "destination", required_unless_present = "destination")]
    verify_against: Option<std::path::PathBuf>,
}
```

`--destination` and `--verify-against` are mutually exclusive and one is required. The orchestration must:

1. Call `inspect_source` before creating output.
2. Generate raw source/burnpack under a temporary directory.
3. Spawn the same pinned tool manifest's `convert` binary through Cargo with `SCRFD_GENERATED_RS` set to the staged source and `CARGO_TARGET_DIR` set beneath the temporary directory.
4. In destination mode, rename the completed four-file tree to the absent requested path.
5. In verification mode, compare the four generated files byte-for-byte with the corresponding files under the supplied crate root and return a nonzero exit on the first difference.

Use `std::process::Command`, pass arguments without shell interpolation, and invoke the converter with this exact shape:

```rust
let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
let output = std::process::Command::new(cargo)
    .arg("run")
    .arg("--manifest-path")
    .arg(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"))
    .arg("--bin")
    .arg("convert")
    .arg("--")
    .arg("--burnpack")
    .arg(&generated.burnpack)
    .arg("--safetensors")
    .arg(&staged_safetensors)
    .env("SCRFD_GENERATED_RS", &generated.source)
    .env("CARGO_TARGET_DIR", temp.path().join("converter-target"))
    .output()
    .map_err(|source| ToolError::Io {
        operation: "spawn SCRFD converter",
        path: std::path::PathBuf::from("cargo"),
        source,
    })?;
```

Require `output.status.success()` and include the child exit status and UTF-8-lossy stderr in `ToolError::ConversionProcess` otherwise.

- [ ] **Step 7: Generate and install the first committed artifact set**

From `rust/`, use a known absent candidate directory:

```powershell
cargo run --manifest-path tools/scrfd-import/Cargo.toml --bin generate -- --repo-root .. --destination crates/feathertalk-scrfd/.generated-candidate
```

Expected: the candidate contains exactly the four-file tree. Inspect the generated model header and forward signature:

```powershell
rg -n "pub struct Model|pub fn forward|from_file|from_bytes|Tensor<B, 3>" crates/feathertalk-scrfd/.generated-candidate/src/generated/scrfd_2_5g.rs
```

Expected: one public generic `Model`, a nine-`Tensor<B, 3>` tuple forward result, and no `from_file`/`from_bytes` constructor.

Copy the four generated files mechanically into the matching crate paths, verify every copy, then remove only the validated candidate directory:

```powershell
New-Item -ItemType Directory -Force crates\feathertalk-scrfd\src\generated | Out-Null
New-Item -ItemType Directory -Force crates\feathertalk-scrfd\artifacts\scrfd_2_5g | Out-Null
$copies = @(
  @{ Source = 'crates\feathertalk-scrfd\.generated-candidate\src\generated\scrfd_2_5g.rs'; Destination = 'crates\feathertalk-scrfd\src\generated\scrfd_2_5g.rs' },
  @{ Source = 'crates\feathertalk-scrfd\.generated-candidate\src\generated\artifact_contract.rs'; Destination = 'crates\feathertalk-scrfd\src\generated\artifact_contract.rs' },
  @{ Source = 'crates\feathertalk-scrfd\.generated-candidate\artifacts\scrfd_2_5g\model.safetensors'; Destination = 'crates\feathertalk-scrfd\artifacts\scrfd_2_5g\model.safetensors' },
  @{ Source = 'crates\feathertalk-scrfd\.generated-candidate\artifacts\scrfd_2_5g\manifest.json'; Destination = 'crates\feathertalk-scrfd\artifacts\scrfd_2_5g\manifest.json' }
)
foreach ($copy in $copies) {
  Copy-Item -LiteralPath $copy.Source -Destination $copy.Destination
  if ((Get-FileHash -Algorithm SHA256 -LiteralPath $copy.Source).Hash -ne
      (Get-FileHash -Algorithm SHA256 -LiteralPath $copy.Destination).Hash) {
    throw "generated artifact copy mismatch: $($copy.Destination)"
  }
}
$crateRoot = (Resolve-Path -LiteralPath 'crates\feathertalk-scrfd').Path.TrimEnd('\') + '\'
$candidate = (Resolve-Path -LiteralPath 'crates\feathertalk-scrfd\.generated-candidate').Path
if (-not $candidate.StartsWith($crateRoot, [StringComparison]::OrdinalIgnoreCase)) {
  throw "candidate escaped crate root: $candidate"
}
Remove-Item -LiteralPath $candidate -Recurse -Force
```

- [ ] **Step 8: Run round-trip and reproducibility verification**

Run:

```powershell
cargo fmt --manifest-path tools/scrfd-import/Cargo.toml --check
cargo test --manifest-path tools/scrfd-import/Cargo.toml
cargo run --manifest-path tools/scrfd-import/Cargo.toml --bin generate -- --repo-root .. --verify-against crates/feathertalk-scrfd
cargo test -p feathertalk-scrfd --test manifest_contract
git diff --check
```

Expected: tool tests pass, regeneration is byte-identical, the committed manifest validates, and normal workspace tests still do not compile the importer.

- [ ] **Step 9: Document the exact commands and commit artifacts**

The README must document source identity, exact dependency versions, generation mode, verification mode, the temporary `.bpk` lifecycle, Python fixture commands reserved for Task 5, and the `NOASSERTION` license status.

```powershell
git add rust/tools/scrfd-import rust/crates/feathertalk-scrfd/src/generated rust/crates/feathertalk-scrfd/artifacts
git commit -m "feat: generate SCRFD Burn artifacts"
```

---

### Task 4: Load the safetensors artifact and expose structured raw inference

**Files:**
- Modify: `rust/crates/feathertalk-scrfd/src/lib.rs`
- Create: `rust/crates/feathertalk-scrfd/src/generated/mod.rs`
- Create: `rust/crates/feathertalk-scrfd/src/artifact.rs`
- Create: `rust/crates/feathertalk-scrfd/src/model.rs`
- Create: `rust/crates/feathertalk-scrfd/src/output.rs`
- Create: `rust/crates/feathertalk-scrfd/tests/artifact.rs`

**Interfaces:**
- Produces `ScrfdArtifactPaths { manifest: PathBuf, weights: PathBuf }`.
- Produces `ScrfdModel<B>::load`, `ScrfdModel<B>::forward`, and `ScrfdModel<B>::manifest` with the approved signatures.
- Produces `ScrfdRawOutput<B> { levels: [ScrfdLevelOutput<B>; 3] }` and per-level `stride`, `scores`, `bbox_deltas`, and `keypoint_deltas`.
- Keeps `generated::scrfd_2_5g::Model` and all generated artifact constants crate-private.

- [ ] **Step 1: Write failing public artifact and shape tests**

Create `tests/artifact.rs`:

```rust
use std::path::Path;

use burn::backend::NdArray;
use burn::tensor::Tensor;
use feathertalk_scrfd::{ScrfdArtifactPaths, ScrfdError, ScrfdModel};

type CpuBackend = NdArray<f32>;

fn artifact_paths() -> ScrfdArtifactPaths {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/scrfd_2_5g");
    ScrfdArtifactPaths {
        manifest: root.join("manifest.json"),
        weights: root.join("model.safetensors"),
    }
}

#[test]
fn committed_artifact_loads_and_exposes_the_validated_manifest() {
    let device = Default::default();
    let model = ScrfdModel::<CpuBackend>::load(&artifact_paths(), &device).unwrap();
    assert_eq!(model.manifest().schema_version, 1);
    assert_eq!(model.manifest().levels[0].anchors, 12_800);
}

#[test]
fn forward_rejects_every_non_contract_input_shape_before_graph_execution() {
    let device = Default::default();
    let model = ScrfdModel::<CpuBackend>::load(&artifact_paths(), &device).unwrap();
    for shape in [
        [2, 3, 640, 640],
        [1, 1, 640, 640],
        [1, 3, 639, 640],
        [1, 3, 640, 639],
    ] {
        let input = Tensor::<CpuBackend, 4>::zeros(shape, &device);
        assert!(matches!(
            model.forward(input),
            Err(ScrfdError::InvalidInputShape { actual }) if actual == shape
        ));
    }
}

#[test]
fn committed_model_returns_the_three_fixed_level_shapes() {
    let device = Default::default();
    let model = ScrfdModel::<CpuBackend>::load(&artifact_paths(), &device).unwrap();
    let output = model
        .forward(Tensor::<CpuBackend, 4>::zeros([1, 3, 640, 640], &device))
        .unwrap();
    for (level, stride, anchors) in output
        .levels
        .into_iter()
        .zip([8, 16, 32])
        .zip([12_800, 3_200, 800])
        .map(|((level, stride), anchors)| (level, stride, anchors))
    {
        assert_eq!(level.stride, stride);
        assert_eq!(level.scores.dims(), [1, anchors]);
        assert_eq!(level.bbox_deltas.dims(), [1, anchors, 4]);
        assert_eq!(level.keypoint_deltas.dims(), [1, anchors, 10]);
    }
}

```

Add corruption tests that copy the two committed artifact files into `TempDir`, then separately: insert a top-level unknown JSON field with `serde_json::Value` (`ManifestJson`), replace `weights.sha256` with another lowercase 64-character digest (`ContractMismatch { field: "weights.sha256", .. }`), write a 65,537-byte manifest (`ManifestTooLarge`), use `File::set_len(16 * 1024 * 1024 + 1)` for an oversized weight file (`WeightsTooLarge`), flip one weight byte without changing length (`HashMismatch { artifact: "weights", .. }`), truncate the weights (`WeightSizeMismatch`), and point each path at a missing file (`Io`).

- [ ] **Step 2: Run artifact tests and verify the red result**

Run: `cargo test -p feathertalk-scrfd --test artifact -- --nocapture`

Expected: compilation fails because artifact paths, model loading, output types, and generated module registration are absent.

- [ ] **Step 3: Register private generated modules and artifact constants**

`src/generated/mod.rs` contains:

```rust
pub(crate) mod artifact_contract;
pub(crate) mod scrfd_2_5g;
```

Do not re-export either module from the crate root. `artifact.rs` imports the four generated constants and compares them with `manifest.generated_source` and `manifest.weights` after `manifest.validate()` succeeds. Add this repository-integrity unit test inside `src/artifact.rs`, where the generated constant remains crate-private:

```rust
#[test]
fn committed_generated_source_matches_its_compiled_contract() {
    use sha2::{Digest, Sha256};

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/generated/scrfd_2_5g.rs");
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(bytes.len() as u64, super::GENERATED_SOURCE_BYTES);
    assert_eq!(
        hex::encode(Sha256::digest(bytes)),
        super::GENERATED_SOURCE_SHA256,
    );
}
```

- [ ] **Step 4: Implement bounded immutable artifact reads and strict application**

Define:

```rust
pub const MAX_MANIFEST_BYTES: u64 = 64 * 1024;
pub const MAX_WEIGHT_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScrfdArtifactPaths {
    pub manifest: std::path::PathBuf,
    pub weights: std::path::PathBuf,
}
```

Use one opened handle per file. For the manifest, reject metadata length over the limit, read through `take(MAX_MANIFEST_BYTES + 1)`, and reject if the resulting bytes exceed the limit. For weights, reject metadata over 16 MiB, read through `take(MAX_WEIGHT_BYTES + 1)`, require the bytes length to equal both the manifest and generated contract byte counts, hash those exact bytes, and pass the same `Vec<u8>` to `SafetensorsStore::from_bytes`.

Construct a fresh private generated model and load with:

```rust
let mut store = burn_store::SafetensorsStore::from_bytes(Some(weight_bytes))
    .allow_partial(true)
    .validate(false);
let result = model
    .load_from(&mut store)
    .map_err(|error| ScrfdError::Store(error.to_string()))?;
validate_apply_result(&result)?;
```

`validate_apply_result` rejects `missing`, `unused`, `skipped`, and every `ApplyError`, mapping shape and dtype failures to their stable variants and other apply failures to `Store`. Unit-test this helper with manually constructed `ApplyResult` values so every mapping is covered without mutating the committed weight hash.

Use this exact decision order and mapping:

```rust
pub fn validate_apply_result(result: &burn_store::ApplyResult) -> Result<(), ScrfdError> {
    if let Some(path) = result.missing.iter().map(|(path, _)| path).min() {
        return Err(ScrfdError::MissingTensor(path.clone()));
    }
    if let Some(error) = result.errors.first() {
        return Err(match error {
            burn_store::ApplyError::ShapeMismatch { path, expected, found } => {
                ScrfdError::ShapeMismatch(format!("{path}: expected {expected:?}, got {found:?}"))
            }
            burn_store::ApplyError::DTypeMismatch { path, expected, found } => {
                ScrfdError::DTypeMismatch(format!("{path}: expected {expected:?}, got {found:?}"))
            }
            burn_store::ApplyError::AdapterError { path, message }
            | burn_store::ApplyError::LoadError { path, message } => {
                ScrfdError::Store(format!("{path}: {message}"))
            }
        });
    }
    if let Some(path) = result.skipped.iter().min() {
        return Err(ScrfdError::Store(format!("skipped tensor: {path}")));
    }
    if let Some(path) = result.unused.iter().min() {
        return Err(ScrfdError::UnexpectedTensor(path.clone()));
    }
    Ok(())
}
```

- [ ] **Step 5: Implement raw-output assembly with exact rank changes**

Define:

```rust
pub struct ScrfdRawOutput<B: burn::tensor::backend::Backend> {
    pub levels: [ScrfdLevelOutput<B>; 3],
}

pub struct ScrfdLevelOutput<B: burn::tensor::backend::Backend> {
    pub stride: u32,
    pub scores: burn::tensor::Tensor<B, 2>,
    pub bbox_deltas: burn::tensor::Tensor<B, 3>,
    pub keypoint_deltas: burn::tensor::Tensor<B, 3>,
}
```

The crate-private assembler accepts the generated tuple in exact ONNX order:

```rust
pub(crate) type GeneratedOutput<B> = (
    Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>,
    Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>,
    Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>,
);
```

Validate all nine raw dimensions before reshaping. Convert each score with `reshape([1, anchors])`; leave bbox/keypoint values unchanged. Add unit tests in `output.rs` using zero tensors for one correct tuple and one wrong shape per output class.

- [ ] **Step 6: Implement the public model wrapper**

```rust
pub struct ScrfdModel<B: burn::tensor::backend::Backend> {
    model: crate::generated::scrfd_2_5g::Model<B>,
    manifest: crate::ScrfdArtifactManifest,
}

impl<B: burn::tensor::backend::Backend> ScrfdModel<B> {
    pub fn load(
        paths: &crate::ScrfdArtifactPaths,
        device: &B::Device,
    ) -> Result<Self, crate::ScrfdError>;

    pub fn forward(
        &self,
        input: burn::tensor::Tensor<B, 4>,
    ) -> Result<crate::ScrfdRawOutput<B>, crate::ScrfdError>;

    pub fn manifest(&self) -> &crate::ScrfdArtifactManifest;
}
```

`forward` checks `input.dims()` before calling the generated graph, destructures the nine-tensor tuple, and passes it to the output assembler. It performs no numerical preprocessing or postprocessing.

- [ ] **Step 7: Run focused runtime verification**

Run:

```powershell
cargo fmt --check
cargo clippy -p feathertalk-scrfd --all-targets --all-features -- -D warnings
cargo test -p feathertalk-scrfd --test artifact -- --nocapture
cargo test -p feathertalk-scrfd --all-targets
git diff --check
```

Expected: artifact load, corruption, input validation, output assembler, and zero-input forward shape tests pass.

- [ ] **Step 8: Commit runtime loading and raw inference**

```powershell
git add rust/crates/feathertalk-scrfd rust/Cargo.lock
git commit -m "feat: run SCRFD Burn inference"
```

---

### Task 5: Generate and validate deterministic OpenCV fixtures

**Files:**
- Create: `rust/tools/scrfd-import/.gitignore`
- Create: `rust/tools/scrfd-import/python/requirements-fixture.txt`
- Create: `rust/tools/scrfd-import/python/generate_fixture.py`
- Modify: `rust/tools/scrfd-import/README.md`
- Create: `rust/crates/feathertalk-scrfd/tests/support/mod.rs`
- Create: `rust/crates/feathertalk-scrfd/tests/fixture_contract.rs`
- Generate: `rust/crates/feathertalk-scrfd/tests/fixtures/opencv_cpu_v1/fixture.json`
- Generate: `rust/crates/feathertalk-scrfd/tests/fixtures/opencv_cpu_v1/input.npy`
- Generate: `rust/crates/feathertalk-scrfd/tests/fixtures/opencv_cpu_v1/out0.npy`
- Generate: `rust/crates/feathertalk-scrfd/tests/fixtures/opencv_cpu_v1/out1.npy`
- Generate: `rust/crates/feathertalk-scrfd/tests/fixtures/opencv_cpu_v1/out2.npy`
- Generate: `rust/crates/feathertalk-scrfd/tests/fixtures/opencv_cpu_v1/out3.npy`
- Generate: `rust/crates/feathertalk-scrfd/tests/fixtures/opencv_cpu_v1/out4.npy`
- Generate: `rust/crates/feathertalk-scrfd/tests/fixtures/opencv_cpu_v1/out5.npy`
- Generate: `rust/crates/feathertalk-scrfd/tests/fixtures/opencv_cpu_v1/out6.npy`
- Generate: `rust/crates/feathertalk-scrfd/tests/fixtures/opencv_cpu_v1/out7.npy`
- Generate: `rust/crates/feathertalk-scrfd/tests/fixtures/opencv_cpu_v1/out8.npy`

**Interfaces:**
- Produces a pinned Python command that generates or verifies `opencv_cpu_v1` without modifying existing destinations.
- Produces the exact test-only fixture schema and `VerifiedFixture`, `artifact_paths`, `fixture_dir`, `read_array`, `load_and_verify_fixture`, and injectable `load_and_verify_fixture_at` support APIs.
- Fixes the synthetic BGR formula and OpenCV backend/target/thread/OpenCL settings.

- [ ] **Step 1: Write the failing fixture-contract test**

Create `tests/support/mod.rs` with these exact test-only value types:

```rust
#![allow(dead_code)] // Each integration-test crate consumes a different helper subset.

use std::{
    collections::BTreeMap,
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
};

use ndarray::ArrayD;
use ndarray_npy::ReadNpyExt;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use feathertalk_scrfd::ScrfdArtifactPaths;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixtureManifest {
    pub schema_version: u32,
    pub case: String,
    pub source: FixtureSource,
    pub generator: FixtureGenerator,
    pub files: BTreeMap<String, FixtureFile>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixtureSource {
    pub file_name: String,
    pub file_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixtureGenerator {
    pub python_version: String,
    pub numpy_version: String,
    pub opencv_version: String,
    pub backend: String,
    pub target: String,
    pub threads: u32,
    pub opencl: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixtureFile {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug)]
pub struct VerifiedFixture {
    pub root: PathBuf,
    pub manifest: FixtureManifest,
}

pub fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/opencv_cpu_v1")
}

pub fn artifact_paths() -> ScrfdArtifactPaths {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/scrfd_2_5g");
    ScrfdArtifactPaths {
        manifest: root.join("manifest.json"),
        weights: root.join("model.safetensors"),
    }
}

pub fn read_array(path: &Path) -> Result<ArrayD<f32>, String> {
    let file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    ArrayD::<f32>::read_npy(BufReader::new(file))
        .map_err(|error| format!("{}: {error}", path.display()))
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
```

Implement these two entry points, with the first delegating to the second:

```rust
pub fn load_and_verify_fixture() -> Result<VerifiedFixture, String> {
    load_and_verify_fixture_at(&fixture_dir())
}

pub fn load_and_verify_fixture_at(root: &Path) -> Result<VerifiedFixture, String>;
```

`load_and_verify_fixture_at` deserializes `root/fixture.json`, then enforces all of the following before returning:

1. `schema_version == 1`, `case == "opencv_cpu_v1"`, and source name/size/hash exactly match the approved ONNX.
2. Generator fields equal `python_version = "3.11"`, `numpy_version = "2.2.6"`, `opencv_version = "4.12.0"`, backend `opencv`, target `cpu`, threads `1`, and OpenCL `false`.
3. The `BTreeMap` key set is exactly `input.npy`, followed by `out0.npy` through `out8.npy`; every dtype is `float32` and every shape matches the Global Constraints.
4. A streaming SHA-256 helper reads each NPY through a 64 KiB buffer, compares its actual byte count and lowercase digest with the descriptor, then calls `read_array` and independently compares the decoded shape.
5. Every decoded input and output value is finite; any metadata, hashing, NPY, shape, or finite-value failure returns a path-specific `Err(String)`.

Use this exact descriptor table so validation and generation share one ordering:

```rust
const FIXTURE_FILES: [(&str, &[usize]); 10] = [
    ("input.npy", &[1, 3, 640, 640]),
    ("out0.npy", &[1, 12_800, 1]),
    ("out1.npy", &[1, 3_200, 1]),
    ("out2.npy", &[1, 800, 1]),
    ("out3.npy", &[1, 12_800, 4]),
    ("out4.npy", &[1, 3_200, 4]),
    ("out5.npy", &[1, 800, 4]),
    ("out6.npy", &[1, 12_800, 10]),
    ("out7.npy", &[1, 3_200, 10]),
    ("out8.npy", &[1, 800, 10]),
];
```

Compare each descriptor and decoded array against the referenced slice exactly; no zero tensor dimension is permitted.

Create `tests/fixture_contract.rs` that calls `load_and_verify_fixture()`, uses `fixture.manifest` to assert the schema/generator values, and reconstructs every normalized input element from the exact BGR formula after RGB swap with exact `f32` equality.

The Rust reconstruction is:

```rust
mod support;

fn source_channel(channel: usize, x: usize, y: usize) -> u8 {
    let value = match channel {
        0 => 3 * x + 5 * y + 17,
        1 => 7 * x + 11 * y + 29,
        2 => 13 * x + 17 * y + 43,
        _ => unreachable!(),
    };
    (value % 256) as u8
}

fn expected_nchw(channel: usize, x: usize, y: usize) -> f32 {
    let bgr_channel = [2, 1, 0][channel];
    (f32::from(source_channel(bgr_channel, x, y)) - 127.5) / 128.0
}

#[test]
fn committed_opencv_fixture_has_the_fixed_contract_and_input() {
    let fixture = support::load_and_verify_fixture().unwrap();
    assert_eq!(fixture.manifest.schema_version, 1);
    assert_eq!(fixture.manifest.generator.python_version, "3.11");
    assert_eq!(fixture.manifest.generator.numpy_version, "2.2.6");
    assert_eq!(fixture.manifest.generator.opencv_version, "4.12.0");

    let input = support::read_array(&fixture.root.join("input.npy")).unwrap();
    assert_eq!(input.shape(), &[1, 3, 640, 640]);
    for channel in 0..3 {
        for y in 0..640 {
            for x in 0..640 {
                assert_eq!(
                    input[ndarray::IxDyn(&[0, channel, y, x])],
                    expected_nchw(channel, x, y),
                    "channel={channel}, x={x}, y={y}",
                );
            }
        }
    }
}

#[test]
fn fixture_schema_rejects_unknown_fields() {
    let path = support::fixture_dir().join("fixture.json");
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    value["future_field"] = serde_json::json!(true);
    assert!(serde_json::from_value::<support::FixtureManifest>(value).is_err());
}

#[test]
fn fixture_loader_rejects_corrupt_metadata_and_non_finite_arrays() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    std::fs::create_dir_all(root).unwrap();
    std::fs::copy(
        support::fixture_dir().join("fixture.json"),
        root.join("fixture.json"),
    )
    .unwrap();
    for name in ["input.npy", "out0.npy", "out1.npy", "out2.npy", "out3.npy", "out4.npy", "out5.npy", "out6.npy", "out7.npy", "out8.npy"] {
        std::fs::copy(support::fixture_dir().join(name), root.join(name)).unwrap();
    }
    let mut bytes = std::fs::read(root.join("out0.npy")).unwrap();
    let payload_start = bytes.len() - 4 * 12_800;
    bytes[payload_start..payload_start + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    std::fs::write(root.join("out0.npy"), bytes).unwrap();
    let mut json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("fixture.json")).unwrap()).unwrap();
    let bytes = std::fs::read(root.join("out0.npy")).unwrap();
    json["files"]["out0.npy"]["bytes"] = serde_json::json!(bytes.len());
    json["files"]["out0.npy"]["sha256"] =
        serde_json::Value::String(support::sha256_bytes(&bytes));
    std::fs::write(root.join("fixture.json"), serde_json::to_vec(&json).unwrap()).unwrap();
    assert!(support::load_and_verify_fixture_at(root).is_err());

    let mut json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("fixture.json")).unwrap()).unwrap();
    json["files"]["out0.npy"]["sha256"] = serde_json::Value::String("0".repeat(64));
    std::fs::write(root.join("fixture.json"), serde_json::to_vec(&json).unwrap()).unwrap();
    assert!(support::load_and_verify_fixture_at(root).is_err());
}
```

- [ ] **Step 2: Run the fixture test and verify the red result**

Run: `cargo test -p feathertalk-scrfd --test fixture_contract -- --nocapture`

Expected: failure clearly reports that `tests/fixtures/opencv_cpu_v1/fixture.json` is missing.

- [ ] **Step 3: Pin Python dependencies and implement the generator**

`requirements-fixture.txt` contains exactly:

```text
numpy==2.2.6
opencv-python-headless==4.12.0.88
```

`.gitignore` contains only:

```text
.venv/
```

The Python script uses this exact argument contract and publication policy:

```python
import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--repo-root", type=Path, required=True)
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--destination", type=Path)
mode.add_argument("--verify-against", type=Path)
args = parser.parse_args()
repo_root = args.repo_root.resolve()
onnx_path = repo_root / "data_utils" / "scrfd_2.5g_kps.onnx"
onnx_bytes = onnx_path.read_bytes()
assert len(onnx_bytes) == 3_291_017
assert hashlib.sha256(onnx_bytes).hexdigest() == (
    "32d20c77b9e2dc1d07e94c2ab9d25bdd5cd05eddbe0b46e7b38e7a1eca22e99a"
)
if args.destination is not None:
    destination = args.destination.resolve()
    if destination.exists() or destination.is_symlink():
        raise SystemExit(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = destination.parent
else:
    destination = None
    staging_parent = Path(tempfile.gettempdir()).resolve()

with tempfile.TemporaryDirectory(prefix="scrfd-fixture-", dir=staging_parent) as temp:
    staging = Path(temp)
    # Generate all eleven files below, then either rename this absent tree or compare it.
```

The script validates the ONNX size/hash, then constructs:

```python
x = np.arange(640, dtype=np.uint16)[None, :]
y = np.arange(640, dtype=np.uint16)[:, None]
image = np.empty((640, 640, 3), dtype=np.uint8)
image[..., 0] = ((3 * x + 5 * y + 17) % 256).astype(np.uint8)
image[..., 1] = ((7 * x + 11 * y + 29) % 256).astype(np.uint8)
image[..., 2] = ((13 * x + 17 * y + 43) % 256).astype(np.uint8)
```

Configure and run OpenCV exactly:

```python
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
net = cv2.dnn.readNetFromONNX(str(onnx_path))
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
names = tuple(net.getUnconnectedOutLayersNames())
assert names == tuple(f"out{i}" for i in range(9))
blob = cv2.dnn.blobFromImage(
    image,
    scalefactor=1.0 / 128.0,
    size=(640, 640),
    mean=(127.5, 127.5, 127.5),
    swapRB=True,
    crop=False,
)
net.setInput(blob)
outputs = net.forward(names)
```

Before inference, require `sys.version_info[:2] == (3, 11)`, `np.__version__ == "2.2.6"`, and `cv2.__version__ == "4.12.0"`. Build the exact JSON schema from Step 1 with this data flow after inference:

```python
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(64 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
```

```python
arrays = {"input.npy": blob}
arrays.update({f"out{index}.npy": output for index, output in enumerate(outputs)})
expected_shapes = {
    "input.npy": [1, 3, 640, 640],
    "out0.npy": [1, 12800, 1],
    "out1.npy": [1, 3200, 1],
    "out2.npy": [1, 800, 1],
    "out3.npy": [1, 12800, 4],
    "out4.npy": [1, 3200, 4],
    "out5.npy": [1, 800, 4],
    "out6.npy": [1, 12800, 10],
    "out7.npy": [1, 3200, 10],
    "out8.npy": [1, 800, 10],
}
files = {}
for name, array in arrays.items():
    assert array.dtype == np.float32
    assert list(array.shape) == expected_shapes[name]
    path = staging / name
    np.save(path, array, allow_pickle=False)
    files[name] = {
        "dtype": "float32",
        "shape": expected_shapes[name],
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }

manifest = {
    "schema_version": 1,
    "case": "opencv_cpu_v1",
    "source": {
        "file_name": "scrfd_2.5g_kps.onnx",
        "file_bytes": 3291017,
        "sha256": "32d20c77b9e2dc1d07e94c2ab9d25bdd5cd05eddbe0b46e7b38e7a1eca22e99a",
    },
    "generator": {
        "python_version": "3.11",
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "backend": "opencv",
        "target": "cpu",
        "threads": cv2.getNumThreads(),
        "opencl": cv2.ocl.useOpenCL(),
    },
    "files": files,
}
(staging / "fixture.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
    newline="\n",
)

expected_names = {"fixture.json", *arrays}
assert {path.name for path in staging.iterdir()} == expected_names
if destination is not None:
    os.rename(staging, destination)
else:
    committed = args.verify_against.resolve()
    expected_paths = [committed / name for name in sorted(expected_names)]
    actual_paths = [staging / name for name in sorted(expected_names)]
    for actual, expected in zip(actual_paths, expected_paths):
        if actual.read_bytes() != expected.read_bytes():
            raise SystemExit(f"fixture differs: {expected}")
```

`sha256_file` streams 64 KiB chunks. Require `cv2.getNumThreads() == 1` and `not cv2.ocl.useOpenCL()` after configuration. Destination mode publishes an absent directory by same-filesystem rename; verification mode generates into a temporary directory and byte-compares the exact set of all eleven files with the committed fixture.

- [ ] **Step 4: Create the isolated Python environment and generate fixtures**

From `rust/tools/scrfd-import`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --disable-pip-version-check -r python\requirements-fixture.txt
.\.venv\Scripts\python.exe python\generate_fixture.py --repo-root ..\..\.. --destination ..\..\crates\feathertalk-scrfd\tests\fixtures\opencv_cpu_v1
```

The dependency installation is the only Python/network setup step and requires explicit network approval if packages are not cached. The script aborts unless the interpreter is Python 3.11 and records the normalized major/minor value `3.11`, NumPy `2.2.6`, and OpenCV `4.12.0` so Python patch releases do not perturb fixture bytes.

- [ ] **Step 5: Run fixture validation and regeneration verification**

Run:

```powershell
cargo test -p feathertalk-scrfd --test fixture_contract -- --nocapture
tools\scrfd-import\.venv\Scripts\python.exe tools\scrfd-import\python\generate_fixture.py --repo-root .. --verify-against crates\feathertalk-scrfd\tests\fixtures\opencv_cpu_v1
git diff --check
```

Expected: all file hashes/shapes and every synthetic input value validate, and regeneration is byte-identical.

- [ ] **Step 6: Commit the fixture generator and committed arrays**

```powershell
git add rust/tools/scrfd-import rust/crates/feathertalk-scrfd/tests
git commit -m "test: add OpenCV SCRFD fixtures"
```

---

### Task 6: Prove CPU parity and add the ignored WGPU smoke test

**Files:**
- Modify: `rust/crates/feathertalk-scrfd/tests/support/mod.rs`
- Create: `rust/crates/feathertalk-scrfd/tests/parity.rs`
- Create: `rust/crates/feathertalk-scrfd/tests/wgpu_smoke.rs`
- Modify: `rust/tools/scrfd-import/README.md`

**Interfaces:**
- Produces test-only `ParityMetrics { max_abs, mean_abs, max_relative }` and finite-value comparison helpers.
- Proves all nine NdArray outputs meet the fixed OpenCV tolerances.
- Provides one ignored WGPU load/forward/shape smoke test using the same committed input.

- [ ] **Step 1: Write the failing CPU parity test**

Create `tests/parity.rs` with a local CPU backend and one model execution:

```rust
mod support;

use burn::{
    backend::NdArray,
    tensor::{Tensor, TensorData},
};
use feathertalk_scrfd::ScrfdModel;

type CpuBackend = NdArray<f32>;

#[test]
fn all_nine_outputs_match_opencv_cpu() {
    let fixture = support::load_and_verify_fixture().unwrap();
    let input = support::read_array(&fixture.root.join("input.npy")).unwrap();
    assert_eq!(input.shape(), &[1, 3, 640, 640]);

    let device = Default::default();
    let tensor = Tensor::<CpuBackend, 4>::from_data(
        TensorData::new(
            input.iter().copied().collect::<Vec<_>>(),
            input.shape().to_vec(),
        ),
        &device,
    );
    let model = ScrfdModel::<CpuBackend>::load(&support::artifact_paths(), &device).unwrap();
    let output = model.forward(tensor).unwrap();

    for (level_index, level) in output.levels.into_iter().enumerate() {
        let score_name = format!("out{level_index}.npy");
        let bbox_name = format!("out{}.npy", level_index + 3);
        let keypoint_name = format!("out{}.npy", level_index + 6);
        support::assert_cpu_tensor_matches_fixture(level.scores, &fixture.root.join(score_name));
        support::assert_cpu_tensor_matches_fixture(
            level.bbox_deltas,
            &fixture.root.join(bbox_name),
        );
        support::assert_cpu_tensor_matches_fixture(
            level.keypoint_deltas,
            &fixture.root.join(keypoint_name),
        );
    }
}

#[test]
fn parity_metric_rejects_non_finite_values() {
    assert!(support::compare_f32(&[f32::NAN], &[0.0]).is_err());
    assert!(support::compare_f32(&[0.0], &[f32::INFINITY]).is_err());
}
```

`assert_cpu_tensor_matches_fixture` compares flattened values after independently checking the public tensor dimensions against the fixture shape; for score fixtures it requires raw `[1,A,1]` and public `[1,A]`, while bbox/keypoint ranks must match exactly.

- [ ] **Step 2: Run CPU parity and observe the red result**

Run: `cargo test -p feathertalk-scrfd --test parity -- --nocapture`

Expected: compilation fails because parity metric and tensor comparison helpers are absent.

- [ ] **Step 3: Implement finite all-element metrics and fixed thresholds**

Add to test support:

```rust
#[derive(Debug, Clone, Copy)]
pub struct ParityMetrics {
    pub max_abs: f32,
    pub mean_abs: f32,
    pub max_relative: f32,
}

pub fn compare_f32(actual: &[f32], expected: &[f32]) -> Result<ParityMetrics, String> {
    if actual.len() != expected.len() || actual.is_empty() {
        return Err("arrays must have the same nonzero length".to_owned());
    }
    let mut max_abs = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    let mut max_relative = 0.0_f64;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if !actual.is_finite() || !expected.is_finite() {
            return Err(format!("non-finite value at index {index}"));
        }
        let actual = f64::from(actual);
        let expected = f64::from(expected);
        let absolute = (actual - expected).abs();
        max_abs = max_abs.max(absolute);
        sum_abs += absolute;
        max_relative = max_relative.max(absolute / expected.abs().max(1e-7));
    }
    Ok(ParityMetrics {
        max_abs: max_abs as f32,
        mean_abs: (sum_abs / actual.len() as f64) as f32,
        max_relative: max_relative as f32,
    })
}

pub fn assert_cpu_tensor_matches_fixture<const D: usize>(
    tensor: burn::tensor::Tensor<burn::backend::NdArray<f32>, D>,
    path: &std::path::Path,
) {
    let expected = read_array(path).unwrap();
    let raw_shape = expected.shape().to_vec();
    let public_shape = match (D, raw_shape.as_slice()) {
        (2, [1, anchors, 1]) => vec![1, *anchors],
        (3, [1, anchors, width]) => vec![1, *anchors, *width],
        _ => panic!(
            "{}: fixture shape {raw_shape:?} is incompatible with public rank {D}",
            path.display(),
        ),
    };
    assert_eq!(
        tensor.dims().to_vec(),
        public_shape,
        "{}: public tensor shape mismatch",
        path.display(),
    );

    let actual = tensor.into_data().to_vec::<f32>().unwrap();
    let expected = expected.iter().copied().collect::<Vec<_>>();
    let metrics = compare_f32(&actual, &expected).unwrap();
    assert!(metrics.max_abs <= 1e-3, "{}: {metrics:?}", path.display());
    assert!(metrics.mean_abs <= 1e-4, "{}: {metrics:?}", path.display());
    assert!(metrics.max_relative.is_finite(), "{}: {metrics:?}", path.display());
}
```

Do not sample outputs; compare every element in all nine arrays.

- [ ] **Step 4: Add the ignored WGPU smoke test**

Create `tests/wgpu_smoke.rs`:

```rust
mod support;

use burn::{
    backend::Wgpu,
    tensor::{Tensor, TensorData},
};
use feathertalk_scrfd::ScrfdModel;

type GpuBackend = Wgpu<f32, i32, u32>;

#[test]
#[ignore = "requires a compatible WGPU adapter"]
fn committed_scrfd_artifact_runs_on_wgpu() {
    let fixture = support::load_and_verify_fixture().unwrap();
    let input = support::read_array(&fixture.root.join("input.npy")).unwrap();
    let device = Default::default();
    let tensor = Tensor::<GpuBackend, 4>::from_data(
        TensorData::new(
            input.iter().copied().collect::<Vec<_>>(),
            input.shape().to_vec(),
        ),
        &device,
    );
    let model = ScrfdModel::<GpuBackend>::load(&support::artifact_paths(), &device).unwrap();
    let output = model.forward(tensor).unwrap();
    for (level, anchors) in output.levels.into_iter().zip([12_800, 3_200, 800]) {
        assert_eq!(level.scores.dims(), [1, anchors]);
        assert_eq!(level.bbox_deltas.dims(), [1, anchors, 4]);
        assert_eq!(level.keypoint_deltas.dims(), [1, anchors, 10]);
    }
}
```

Do not add CPU fallback logic or make this ignored test part of normal CI acceptance.

- [ ] **Step 5: Run focused parity and manual WGPU verification**

Run:

```powershell
cargo test -p feathertalk-scrfd --test parity -- --nocapture
cargo test -p feathertalk-scrfd --test wgpu_smoke -- --ignored --nocapture
```

Expected: CPU parity passes all nine arrays. The WGPU command passes on a compatible adapter; if no adapter is available, retain the ignored test and report the environment limitation without weakening CPU acceptance.

- [ ] **Step 6: Run complete regeneration and repository acceptance**

Run from `rust/`:

```powershell
cargo fmt --check
cargo clippy -p feathertalk-scrfd --all-targets --all-features -- -D warnings
cargo test -p feathertalk-scrfd --all-targets
cargo fmt --manifest-path tools/scrfd-import/Cargo.toml --check
cargo clippy --manifest-path tools/scrfd-import/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path tools/scrfd-import/Cargo.toml
cargo run --manifest-path tools/scrfd-import/Cargo.toml --bin generate -- --repo-root .. --verify-against crates/feathertalk-scrfd
tools\scrfd-import\.venv\Scripts\python.exe tools\scrfd-import\python\generate_fixture.py --repo-root .. --verify-against crates\feathertalk-scrfd\tests\fixtures\opencv_cpu_v1
cargo test --workspace --all-targets
git diff --check
git status --short
```

Expected: every non-ignored command exits 0; three existing repository WGPU tests plus the new SCRFD WGPU smoke remain ignored under normal workspace tests; no ONNX/Python generation occurs during workspace tests; `git status` contains only intentional task files and the preserved user-owned untracked demo directory.

- [ ] **Step 7: Commit final parity coverage**

```powershell
git add rust/crates/feathertalk-scrfd/tests rust/tools/scrfd-import/README.md
git commit -m "test: verify SCRFD inference parity"
```

## Plan Self-Review

- Spec coverage: Tasks 1-4 cover the independent crate, strict manifest/hash contract, caller-supplied paths, immutable byte loading, generated private graph, fixed batch-one input, structured three-level outputs, and no preprocessing/postprocessing coupling. Tasks 2-3 cover pinned ONNX generation, temporary burnpack conversion, safetensors round trip, deterministic metadata, source/weight constants, absent-destination publication, and byte-identical regeneration. Tasks 5-6 cover pinned OpenCV fixture generation, the exact synthetic input, all-nine CPU parity, and ignored WGPU smoke execution.
- Dependency boundary: `feathertalk-scrfd` uses only workspace Burn/store/serialization/hash dependencies; `burn-onnx`, prost, Python, NumPy, and OpenCV remain under the excluded development tool.
- Runtime isolation: normal workspace builds compile committed Rust source and load caller-provided safetensors only; they never read the source ONNX or invoke development tools.
- Type consistency: the manifest names and field types are fixed in Task 1 and reused by the converter and loader; generated raw outputs are nine rank-three tensors; public score tensors alone become rank two; public model and output signatures match the approved spec.
- Security and failure handling: bounded reads precede allocation, hashes cover the same bytes supplied to the store, strict `ApplyResult` validation rejects every partial application, and generation publishes only complete absent directories.
- Licensing: every generated manifest retains `NOASSERTION` and `redistribution_approved: false`; no task changes release eligibility.
- Test completeness: source metadata, source generation, burnpack/safetensors equality, manifest corruption, weight corruption, input/output shapes, fixture hashes/formula, all-element CPU parity, WGPU smoke, importer reproducibility, fixture reproducibility, crate acceptance, and workspace regression are each tied to an explicit command.
- Placeholder scan: angle-bracket notation appears only in documentation of converter-emitted derived values and CLI syntax; no committed implementation value is left for manual substitution, and no task contains a deferred behavior marker.
- User preference: execution is explicitly inline through `executing-plans`; no subagent workflow is part of this plan.
