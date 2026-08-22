# SCRFD Burn Inference Design

Date: 2026-08-22
Status: Approved for implementation planning

## 1. Purpose

Complete the SCRFD model-execution slice that follows the existing pure-Rust
postprocessing contract. The repository's `data_utils/scrfd_2.5g_kps.onnx`
remains the source of truth for the model graph. A fixed Burn ONNX 0.21 import
run will produce reviewable Rust model source and a standard safetensors
artifact. The runtime will execute the generated Burn graph on a caller-selected
Burn backend and expose the nine raw model outputs in a structured form.

The acceptance target is numerical agreement with the current
Python/OpenCV-DNN implementation on a deterministic synthetic input. Rust
tests consume committed fixtures and never start Python or parse ONNX.

This slice does not implement image decoding, image resizing, BGR/RGB
conversion, normalization, anchor generation, bbox/keypoint decoding, score
filtering, NMS, PFLD inference, frame classification, workers, or GPUI
integration.

## 2. Decisions and fixed constraints

The following decisions are part of the contract:

- Reference engine: Python/OpenCV DNN, preserving the behavior in
  `data_utils/detect_face.py`.
- Model crate: new independent `rust/crates/feathertalk-scrfd` crate.
- Public scope: raw SCRFD inference only; the crate does not depend on
  `feathertalk-face`.
- Conversion tool: retained in the repository as a development-only tool;
  ordinary workspace builds and runtime execution do not depend on Python,
  ONNX, or the tool.
- Import route: `burn-onnx = 0.21.0`, matching the workspace's exact Burn
  version. The tool may use a temporary `.bpk` file as an intermediate, but
  `.bpk` is not a runtime or committed artifact.
- Backend: the model is generic over `burn::tensor::backend::Backend`.
  NdArray/CPU is the mandatory parity backend. WGPU has an ignored manual
  smoke test for loading, execution, and output shapes.
- Input: `Tensor<B, 4>` with exact shape `[1, 3, 640, 640]`. It is already
  normalized according to the OpenCV blob contract. The crate does not own
  pixels or preprocessing and only supports batch size one in this slice.
- Output: a structured value with three levels and retained batch dimensions.
- Weight delivery: the caller supplies a manifest path and a safetensors path;
  weights are not embedded in the binary.
- Reference input: a deterministic synthetic tensor with channel and spatial
  variation. No demo video frame or real-person image is committed.

## 3. Crate and artifact layout

The workspace gains:

```text
rust/crates/feathertalk-scrfd/
  Cargo.toml
  src/
    lib.rs
    artifact.rs
    error.rs
    output.rs
    generated/
      scrfd_2_5g.rs
  artifacts/scrfd_2_5g/
    model.safetensors
    manifest.json
  tests/
    public_api.rs
    artifact.rs
    parity.rs
    wgpu_smoke.rs
    fixtures/
      opencv_cpu_v1/
        fixture.json
        input.npy
        out0.npy
        out1.npy
        out2.npy
        out3.npy
        out4.npy
        out5.npy
        out6.npy
        out7.npy
        out8.npy
```

`generated/scrfd_2_5g.rs` is generated once by the pinned tool and then
reviewed and committed. It is not generated from `build.rs`; normal Cargo
builds compile the committed source directly.

The development-only converter lives outside the workspace member list:

```text
rust/tools/scrfd-import/
  Cargo.toml
  build.rs
  src/bin/generate.rs
  src/bin/convert.rs
  python/generate_fixture.py
  python/requirements-fixture.txt
  README.md
```

`rust/Cargo.toml` lists `tools/scrfd-import` under `workspace.exclude`, and the
tool manifest pins its own exact dependencies. Consequently
`cargo build/test --workspace` never compiles the importer.

The tool's conditional `build.rs` only copies a caller-selected staged Rust
source file into Cargo's `OUT_DIR` so `convert.rs` can compile that exact
module. It does not parse ONNX, run model generation, or execute during normal
workspace builds.

The converter may create temporary source, debug, and `.bpk` files. It writes
the final source and artifacts to a caller-selected destination only after all
validation and round-trip checks succeed. It never overwrites an existing
destination. Temporary files are not committed.

The source ONNX is not copied into the new crate. The tool reads the existing
`data_utils/scrfd_2.5g_kps.onnx` and validates:

```text
file name: scrfd_2.5g_kps.onnx
size:     3,291,017 bytes
sha256:   32d20c77b9e2dc1d07e94c2ab9d25bdd5cd05edbe0b46e7b38e7a1eca22e99a
```

The manifest records the source hash rather than making the runtime depend on
the ONNX file.

## 4. Runtime architecture

The crate has four focused responsibilities:

1. `generated`: the Burn graph and its generated tensor-level forward method.
2. `artifact`: manifest parsing, source/model contract validation, SHA-256
   verification, and strict safetensors loading.
3. `output`: stable mapping from generated `out0..out8` tensors to the public
   three-level result.
4. `error`: structured, non-panicking errors for caller-controlled files and
   tensor shapes.

The crate does not import or call `feathertalk-face`; a later pipeline layer
will pass the raw output tensors to the already implemented postprocessor.

Loading uses one immutable byte snapshot of the safetensors file: the loader
reads and hashes the bytes first, compares the hash with the manifest, and
loads from those same bytes. This prevents a file replacement between hash
verification and tensor application. The safetensors store is configured for
strict application, so missing, unexpected, dtype-mismatched, or shape-
mismatched tensors fail before the model is returned.

The generated model is private. `ScrfdModel::load` constructs it on the
caller's device, applies the verified weights, and stores the validated
manifest. `forward` validates the input shape before invoking the graph. The
model does not mutate between calls.

Burn ONNX's generated `LoadStrategy::None` source contains no built-in
burnpack or safetensors loading constructor. The development converter must
construct the generated `Model` with `Model::new`, apply the temporary
`.bpk` through `BurnpackStore`, and then save a standard safetensors file.
The generated module is private, and the runtime loader accepts only the
verified safetensors artifact.

## 5. Public API and tensor contract

The public API is intentionally small and uses value/configuration types for
artifact paths and manifest metadata:

```rust
pub struct ScrfdArtifactPaths {
    pub manifest: std::path::PathBuf,
    pub weights: std::path::PathBuf,
}

pub struct ScrfdModel<B: burn::tensor::backend::Backend> {
    // private generated model and validated manifest
}

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

The core methods are:

```rust
impl<B: burn::tensor::backend::Backend> ScrfdModel<B> {
    pub fn load(
        paths: &ScrfdArtifactPaths,
        device: &B::Device,
    ) -> Result<Self, ScrfdError>;

    pub fn forward(
        &self,
        input: burn::tensor::Tensor<B, 4>,
    ) -> Result<ScrfdRawOutput<B>, ScrfdError>;

    pub fn manifest(&self) -> &ScrfdArtifactManifest;
}
```

The generated ONNX output order is fixed and explicitly mapped:

```text
out0, out1, out2 -> scores       for strides 8, 16, 32
out3, out4, out5 -> bbox deltas  for strides 8, 16, 32
out6, out7, out8 -> keypoint deltas for strides 8, 16, 32
```

The public shapes are:

```text
stride 8:  scores [1, 12800], bbox [1, 12800, 4], kps [1, 12800, 10]
stride 16: scores [1,  3200], bbox [1,  3200, 4], kps [1,  3200, 10]
stride 32: scores [1,   800], bbox [1,   800, 4], kps [1,   800, 10]
```

The wrapper preserves model output values. It does not multiply deltas by
stride, generate anchor centers, clip coordinates, filter confidence scores,
or perform NMS; those operations remain in `feathertalk-face`.

## 6. Manifest and licensing metadata

`manifest.json` is a versioned, reviewable artifact. Its schema-one fields are:

- manifest schema and SCRFD architecture versions;
- source ONNX file name, size, SHA-256, opset, and output names;
- exact Burn, burn-onnx, and burn-store versions;
- graph simplification setting;
- input shape, dtype, and normalization description;
- each level's stride, anchor count, and output shapes;
- generated Rust source file name, size, and SHA-256;
- safetensors file name, size, and SHA-256;
- source and licensing metadata.

The nested schema is fixed rather than accepting arbitrary metadata:

```text
ScrfdArtifactManifest
  schema_version: u32                         # exactly 1
  model_kind: string                          # exactly "scrfd_2.5g_kps"
  architecture_version: u32                   # exactly 1
  source: ScrfdSourceManifest
  generator: ScrfdGeneratorManifest
  input: ScrfdInputManifest
  levels: [ScrfdLevelManifest; 3]
  generated_source: ScrfdFileManifest
  weights: ScrfdWeightManifest
  license: ScrfdLicenseManifest

ScrfdSourceManifest
  format: string                              # exactly "onnx"
  file_name: string                           # exactly "scrfd_2.5g_kps.onnx"
  file_bytes: u64                             # exactly 3291017
  sha256: 64-character lowercase hex string   # fixed source hash
  opset: u64                                  # parsed nonzero source opset
  input_name: string                          # exactly "images"
  output_names: [string; 9]                   # exactly "out0" ... "out8"

ScrfdGeneratorManifest
  burn: string                                # exactly "0.21.0"
  burn_onnx: string                           # exactly "0.21.0"
  burn_store: string                          # exactly "0.21.0"
  simplify: bool                              # exactly true
  load_strategy: string                       # exactly "none"

ScrfdInputManifest
  dtype: string                               # exactly "float32"
  shape: [usize; 4]                           # exactly [1,3,640,640]
  scale: f32                                  # exactly 1/128
  mean: [f32; 3]                              # exactly [127.5; 3]
  swap_rb: bool                               # exactly true

ScrfdLevelManifest
  stride: u32
  anchors: usize
  score: ScrfdOutputManifest
  bbox: ScrfdOutputManifest
  keypoints: ScrfdOutputManifest

ScrfdOutputManifest
  onnx_name: string
  source_shape: array of positive dimensions
  public_shape: array of positive dimensions

ScrfdFileManifest
  file_name: string                           # exactly "scrfd_2_5g.rs"
  file_bytes: positive u64
  sha256: 64-character lowercase hex string

ScrfdWeightManifest
  format: string                              # exactly "safetensors"
  file_name: string                           # exactly "model.safetensors"
  file_bytes: positive u64
  sha256: 64-character lowercase hex string

ScrfdLicenseManifest
  license_id: string                          # exactly "NOASSERTION"
  redistribution_approved: bool               # exactly false
  evidence: nonempty string
```

The three level values and their nine output mappings are exactly those in
the following table:

| Stride | Anchors | Score source/public | Bbox source/public | Keypoint source/public |
| --- | ---: | --- | --- | --- |
| 8 | 12,800 | `out0 [1,12800,1]` / `[1,12800]` | `out3 [1,12800,4]` / same | `out6 [1,12800,10]` / same |
| 16 | 3,200 | `out1 [1,3200,1]` / `[1,3200]` | `out4 [1,3200,4]` / same | `out7 [1,3200,10]` / same |
| 32 | 800 | `out2 [1,800,1]` / `[1,800]` | `out5 [1,800,4]` / same | `out8 [1,800,10]` / same |

Unknown JSON fields are rejected. No numeric field that represents a version,
file size, opset, stride, anchor count, or tensor dimension may be zero.

The crate contains compile-time constants for the model kind, architecture
version, source ONNX hash/opset, generated-source hash, weight byte count/hash,
and output mapping. Runtime loading compares the manifest with those constants.
A repository test hashes the committed generated source and proves that it
still matches the compiled generated-source constant.

The repository currently has no independently verified redistribution license
for this SCRFD weight. Therefore the generated manifest must state:

```text
license_id: NOASSERTION
redistribution_approved: false
```

This permits local migration and parity work but blocks treating the artifact
as cleared for a commercial installer. A later release process must perform a
separate model-license audit.

## 7. Development conversion workflow

The pinned tool performs the following deterministic steps:

1. Validate the source ONNX name, byte length, SHA-256, graph output names
   `out0..out8`, and expected input contract.
2. Invoke `burn-onnx 0.21.0` with the workspace's exact Burn version,
   simplification enabled, and `LoadStrategy::None` to generate readable Rust
   source and a temporary `.bpk` state file.
3. Compile `convert.rs` with `SCRFD_GENERATED_RS` pointing at that staged source,
   include the exact staged module, construct `Model::new`, call
   `model.load_from(&mut BurnpackStore::from_file(...))`, and save its state to
   safetensors. The source
   published to the crate is byte-for-byte the source compiled by this
   conversion step.
4. Load the safetensors into a fresh model and compare every tensor snapshot
   before publishing anything.
5. Hash the generated source and safetensors, write the manifest, and publish
   the final files atomically into a previously absent destination.
6. A verification mode regenerates into a temporary directory and requires
   byte-identical Rust source, safetensors, and manifest before reporting that
   committed artifacts are current.

The converter's normal invocation and its exact dependency versions are
documented in its README. It is an explicit developer command, not a Cargo
build script. Python/OpenCV is used only by the separate fixture generator;
the converter itself does not execute Python.

## 8. Error semantics

`ScrfdError` contains stable categories for:

- manifest I/O and JSON errors;
- unsupported schema or architecture version;
- manifest/model contract mismatch;
- missing, unreadable, truncated, or hash-mismatched weights;
- safetensors application errors, including missing/unexpected tensors and
  dtype/shape mismatch;
- invalid input shape (including batch sizes other than one);
- invalid generated output shape or output-name mapping;
- ordinary filesystem errors.

Before allocating buffers, the loader rejects manifests larger than 64 KiB
and weight files larger than 16 MiB. It also requires the manifest's exact
weight byte count before hashing and loading the file.

All caller-controlled inputs return `Result`; no public path, manifest, or
tensor validation path may panic. Internal generated-model assumptions may use
compile-time constants and assertions only where they cannot be influenced by
runtime input.

## 9. OpenCV reference fixtures

The fixture generator is a development-only Python script with pinned
requirements. For integer coordinates `x,y` in `[0,639]`, it creates the
following exact synthetic `uint8` BGR image:

```text
B(x,y) = (3*x  + 5*y  + 17) mod 256
G(x,y) = (7*x  + 11*y + 29) mod 256
R(x,y) = (13*x + 17*y + 43) mod 256
```

It then reproduces the existing code's normalization exactly:

```text
blobFromImage(image,
  scalefactor=1/128,
  size=(640,640),
  mean=(127.5,127.5,127.5),
  swapRB=true)
```

OpenCV DNN is configured with `DNN_BACKEND_OPENCV`, `DNN_TARGET_CPU`,
`cv2.setNumThreads(1)`, and OpenCL disabled. The generator asserts that
`getUnconnectedOutLayersNames()` is exactly `out0..out8` and records Python,
NumPy, and OpenCV versions.

The committed fixture directory contains `input.npy`, `out0.npy` through
`out8.npy`, and a JSON descriptor with dtype, shape, byte size, SHA-256, source
ONNX hash, and generator metadata. The input shape is `[1,3,640,640]`. OpenCV's
raw score outputs `out0..out2` have shapes `[1,A,1]`; bbox outputs have
`[1,A,4]`; keypoint outputs have `[1,A,10]`, where `A` is `12800`, `3200`, or
`800`. The Rust wrapper removes only the trailing singleton score dimension,
producing the public score shapes in Section 5 without changing values.

Rust parity tests reconstruct the Section 9 synthetic BGR formula and verify
that `input.npy` equals its normalized NCHW representation. They also verify
every fixture file hash, load the arrays without invoking Python, OpenCV, or
ONNX, and compare the three score arrays after the documented singleton-
dimension removal. Every value must be finite. The required tolerances for
each of the nine arrays are:

```text
max_abs  <= 1e-3
mean_abs <= 1e-4
```

## 10. Tests and acceptance

Tests in `feathertalk-scrfd` cover:

- public API construction and manifest access;
- exact input and output shape contracts;
- successful artifact loading and strict safetensors round-trip;
- missing, malformed, or hash-mismatched manifests and weights;
- truncated and tensor-incompatible safetensors;
- CPU parity against all nine committed OpenCV arrays;
- non-finite and malformed fixture rejection;
- an ignored WGPU smoke test that loads the same artifact, executes the same
  input, and checks all output shapes.

The fixture tests never access the source ONNX and never launch Python. The
development conversion and fixture generation commands are separate from
ordinary test commands.

Acceptance commands for the implemented slice are:

```powershell
cargo fmt --check
cargo clippy -p feathertalk-scrfd --all-targets --all-features -- -D warnings
cargo test -p feathertalk-scrfd --all-targets
cargo test --workspace --all-targets
git diff --check
```

The WGPU smoke test is intentionally ignored under normal CI and is run
explicitly when a compatible graphics adapter is available.

## 11. Non-goals and follow-up

This design completes only the SCRFD Burn model import and raw inference
boundary. Follow-up work may add an adapter that feeds these outputs into
`feathertalk-face`, image preprocessing, multi-batch execution, model-package
installation, license-cleared distribution metadata, worker orchestration,
and desktop UI integration. None of those changes are implied by this spec.
