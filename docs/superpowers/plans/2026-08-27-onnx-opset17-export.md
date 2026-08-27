# ONNX Opset 17 Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an auditable Rust-only ONNX opset 17 exporter and validator for FeatherHuBERT, Original UNet, and reparameterized MobileOne UNet, with an optional `ort` compatibility check kept outside product runtime dependencies.

**Architecture:** `feathertalk-export` owns a small ONNX protobuf builder, deterministic initializer encoding, graph-contract validation, and model-specific graph emitters. Model snapshots supply f32 weights; the exporter emits a self-contained protobuf model with fixed public names and dynamic only where the contract permits. A separate opt-in validation tool loads the generated file with ONNX Runtime and compares a reference input/output; no runtime crate depends on ONNX.

**Tech Stack:** Rust 1.92, Burn 0.21, `onnx-protobuf` 0.2.3, Safetensors/Burn snapshots, optional `ort` 2.0.0-rc.13 validation tool, clap CLI.

## Global Constraints

- ONNX is a compatibility output, not an application runtime dependency.
- Every exported model uses default-domain ONNX opset 17 and f32 tensors.
- FeatherHuBERT interface is `waveform [1, samples] -> hidden [1, tokens, 1024]`.
- UNet interface is `input [1, 6, 160, 160]`, `audio [1, 16, 32, 32]` -> `output [1, 3, 160, 160]`.
- MobileOne exports the reparameterized inference graph by default; train-time branches are never emitted unless an explicit compatibility option is added later.
- Exporters reject missing, duplicate, unexpected, non-f32, or shape-inconsistent tensors; they never use partial loading or random fallback weights.
- The protected demo `.MOV` and the three protected `.worktrees` paths must not be read or modified.
- Tests must be written before production implementation and must observe the failing behavior first.

---

### Task 1: Add the ONNX protobuf foundation and contract types

**Files:**
- Modify: `rust/Cargo.toml`
- Modify: `rust/crates/feathertalk-export/Cargo.toml`
- Modify: `rust/crates/feathertalk-export/src/lib.rs`
- Create: `rust/crates/feathertalk-export/src/onnx.rs`
- Create: `rust/crates/feathertalk-export/tests/onnx_contract.rs`

**Interfaces:**
- Produces `ONNX_OPSET_VERSION`, `OnnxModelKind`, `OnnxTensorContract`, `OnnxExportError`, `serialize_model`, and `validate_model_contract`.
- `validate_model_contract(bytes, expected)` decodes protobuf and checks IR version, one default-domain opset 17 import, graph presence, exact public input/output names, f32 element types, and allowed dynamic dimensions.

- [ ] Write tests for valid FeatherHuBERT and UNet contracts, wrong opset, wrong names, wrong dtype, missing graph, and symbolic dimensions in forbidden positions.
- [ ] Run `cargo test -p feathertalk-export --test onnx_contract`; confirm failure because the ONNX module and dependency are absent.
- [ ] Add the dependency and minimal protobuf helpers using `onnx-protobuf::Message`.
- [ ] Implement deterministic model serialization and strict structural validation.
- [ ] Run the focused tests and `cargo fmt --all`.
- [ ] Commit `feat: add Rust ONNX protobuf contracts`.

### Task 2: Encode Burn snapshots as deterministic ONNX initializers

**Files:**
- Modify: `rust/crates/feathertalk-export/src/onnx.rs`
- Create: `rust/crates/feathertalk-export/tests/onnx_initializers.rs`

**Interfaces:**
- Produces `InitializerSet`, `initializer_from_snapshot`, `add_snapshot_initializers`, and helpers for ONNX `TensorProto` f32 raw little-endian data.
- Initializer names are sorted, unique, shape-preserving, and use `data_type = FLOAT`; `raw_data` is the only emitted payload field.

- [ ] Write tests proving f32 byte encoding, shape/count checks, sorted names, duplicate rejection, and rejection of non-f32 snapshots.
- [ ] Run focused tests and verify the expected failures.
- [ ] Implement snapshot conversion without lossy casts or partial tensors.
- [ ] Re-run focused tests and inspect a decoded initializer round trip.
- [ ] Commit `feat: encode Burn weights as ONNX initializers`.

### Task 3: Emit the FeatherHuBERT opset 17 graph

**Files:**
- Modify: `rust/crates/feathertalk-export/src/onnx.rs`
- Create: `rust/crates/feathertalk-export/src/onnx_feather_hubert.rs`
- Create: `rust/crates/feathertalk-export/tests/onnx_feather_hubert.rs`

**Interfaces:**
- Produces `export_feather_hubert_onnx(model, config) -> Result<Vec<u8>, OnnxExportError>`.
- Graph contains the seven strided Conv1D + GroupNorm + GELU frontend layers, TCN residual blocks, final GroupNorm/GELU/projection, transpose to `[B,T,C]`, and explicit dynamic sample/token shape metadata.

- [ ] Write a graph test that checks public I/O, opset, initializer coverage, Conv/GroupNorm/GELU/residual node presence, and no training-only Dropout nodes.
- [ ] Run it to observe missing emitter failure.
- [ ] Implement graph naming and node construction; encode GroupNorm as reshape/InstanceNormalization/reshape with affine tensors where necessary, keeping the graph portable.
- [ ] Validate the emitted model with the Rust structural checker and a small zero-input shape smoke test when an ONNX backend is available.
- [ ] Commit `feat: export FeatherHuBERT to ONNX opset 17`.

### Task 4: Emit Original UNet and reparameterized MobileOne graphs

**Files:**
- Create: `rust/crates/feathertalk-export/src/onnx_unet.rs`
- Create: `rust/crates/feathertalk-export/tests/onnx_unet.rs`
- Modify: `rust/crates/feathertalk-export/src/lib.rs`

**Interfaces:**
- Produces `export_original_unet_onnx(model, config)` and `export_mobileone_unet_onnx(model, config)`.
- Original UNet emits depthwise inverted-residual blocks, BatchNorm/Relu, audio bottleneck, align-corners bilinear resize, skip concat, and final Sigmoid.
- MobileOne accepts `MobileOneUnetInference` only and emits fused Conv2D(+bias), optional Relu, residual Add, resize, concat, and Sigmoid; no branch BatchNorm tensors remain in the emitted graph.

- [ ] Write tests for exact I/O, spatial shapes, depthwise group attributes, resize attributes, final sigmoid, and MobileOne absence of train-time branch names.
- [ ] Run tests to observe missing emitters.
- [ ] Implement shared ONNX Conv/BN/Relu/resize/concat helpers and model traversal.
- [ ] Add graph checks that every initializer is consumed and every node input is defined.
- [ ] Commit `feat: export Original and MobileOne UNet graphs`.

### Task 5: Add export and structural-validation CLI

**Files:**
- Modify: `rust/tools/model-package/Cargo.toml`
- Modify: `rust/tools/model-package/src/main.rs`
- Create: `rust/tools/model-package/tests/onnx_cli.rs`

**Interfaces:**
- Adds commands:
  - `feathertalk-model-package onnx feather-hubert --source CHECKPOINT --destination MODEL.onnx`
  - `feathertalk-model-package onnx unet --source PACKAGE_OR_CHECKPOINT --variant original|mobileone --destination MODEL.onnx`
  - `feathertalk-model-package onnx validate --source MODEL.onnx --kind feather-hubert|original-unet|mobileone-unet`
- CLI writes atomically, refuses an existing destination, prints JSON with model kind, opset, byte length, and SHA-256, and never touches protected paths.

- [ ] Write command/help and destination no-clobber tests first.
- [ ] Run them to observe missing subcommands.
- [ ] Implement checkpoint/package loading through existing strict import APIs and route to the exporters.
- [ ] Implement atomic output and JSON report.
- [ ] Run CLI tests and a synthetic model export.
- [ ] Commit `feat: add ONNX export CLI`.

### Task 6: Add an opt-in Rust `ort` compatibility validator

**Files:**
- Create: `rust/tools/onnx-validate/Cargo.toml`
- Create: `rust/tools/onnx-validate/src/main.rs`
- Create: `rust/tools/onnx-validate/tests/cli.rs`
- Modify: `rust/Cargo.toml`

**Interfaces:**
- Separate tool binary `feathertalk-onnx-validate` with `--model`, `--input`, `--expected-output`, and `--kind`.
- Uses `ort` only in this tool; emits one JSON result containing provider, input/output metadata, max absolute error, mean absolute error, and pass/fail threshold.

- [ ] Write CLI contract tests that do not require a provider and verify malformed model/input errors are nonzero.
- [ ] Run tests to observe missing tool failure.
- [ ] Implement optional `ort` session setup and ndarray f32 input/output comparison; keep download/network features disabled where possible.
- [ ] Add a structural-only mode for CI environments without ONNX Runtime binaries.
- [ ] Commit `feat: add opt-in ONNX Runtime compatibility validator`.

### Task 7: Integrate legacy model and `.npy` migration commands

**Files:**
- Create: `rust/tools/model-package/src/migrate.rs`
- Create: `rust/tools/model-package/tests/migrate_cli.rs`
- Modify: `rust/tools/model-package/src/main.rs`
- Modify: `rust/crates/feathertalk-audio/src/lib.rs`
- Modify: `rust/crates/feathertalk-audio/src/format.rs`

**Interfaces:**
- Adds `migrate model` for supported `.pth/.pth.tar` kinds and `migrate features --source AUD_HU.NPY --destination FEATURES.F32`.
- Feature files use the existing versioned little-endian f32 format; `.npy` is accepted only at the migration boundary and is never used by runtime code.

- [ ] Write tests for valid `.npy` conversion, wrong dtype/rank, truncated input, and no-clobber output.
- [ ] Run focused tests to observe missing migration command/API.
- [ ] Implement bounded `ndarray-npy` reading, shape/dtype validation, versioned feature output, and strict model-package routing.
- [ ] Run migration tests and verify the produced header/hash.
- [ ] Commit `feat: add legacy model and feature migration CLI`.

### Task 8: Milestone-wide verification and evidence

**Files:**
- Create: `docs/migration/onnx-export-report.md`
- Modify: `docs/WEIGHTS.md` if model/export commands need documenting

- [ ] Run focused exporter, CLI, and migration tests.
- [ ] Run `cargo test --workspace --all-targets`, `cargo check --workspace --all-targets`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --all -- --check`, and `git diff --check`.
- [ ] Export the supplied `demo/kanghui_training_video_featherhubert_188_latest/feather_hubert_188_latest_99.pth` to a temporary ONNX file, validate its SHA/structure, and record exact evidence without reading the `.MOV`.
- [ ] Run `ort` compatibility checks where the local runtime is available; otherwise record the explicit structural-only limitation.
- [ ] Write the report with exact interfaces, opset, tensor counts, hashes, commands, and any failed threshold; do not claim `GO` unless all required checks pass.
- [ ] Commit `docs: record ONNX export milestone evidence`.
