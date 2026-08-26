# Standard Model Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strict, atomic FeatherTalk standard model package crate and use the supplied FeatherHuBERT `.pth` checkpoint to produce and verify a loadable safetensors package.

**Architecture:** Create `feathertalk-export` as the reusable package-format boundary: schema and license types live in `manifest.rs`, filesystem/hashing/staging mechanics in `io.rs`, generic Burn module packing/loading in `package.rs`, and FeatherHuBERT legacy conversion in `feather_hubert.rs`. A small CLI depends on the crate and accepts explicit source, licenses, destination, creation time, and minimum app version; runtime consumers never parse `.pth`.

**Tech Stack:** Rust 1.92, Burn/burn-store 0.21.0, NdArray CPU backend, serde/serde_json, SHA-256, tempfile, clap, existing `feathertalk-models` and `feathertalk-weights` crates.

## Global Constraints

- Use Rust 1.92 and Burn/burn-store exactly 0.21.0.
- `MODEL_PACKAGE_SCHEMA_VERSION = 1` and `MODEL_LICENSE_SCHEMA_VERSION = 1`.
- Inference packages contain exactly `manifest.json`, `model.safetensors`, and `LICENSES.json` as regular non-symlink files.
- Training-only `optimizer.safetensors` and `training-state.json` are optional but must appear together; this plan does not generate them.
- Manifest and license schemas reject unknown fields.
- Source size is limited to 512 MiB, manifest to 64 KiB, licenses to 1 MiB, and model weights to 2 GiB.
- All file and source hashes are 64 lowercase hexadecimal characters.
- A tensor contract records every sorted full tensor path, exact shape, `f32` dtype, count, and total elements.
- Package loading validates directory shape, JSON, byte lengths, hashes, model description, and tensor audit before applying safetensors.
- Package writing uses a same-parent staging directory, syncs completed files, writes `manifest.json` last, and atomically renames to an absent destination.
- Existing destinations are never overwritten; failures clean only staging owned by the current operation.
- Runtime never downloads, searches caches, accepts `.pth`, falls back to random weights, or silently changes WGPU requests to CPU.
- The FeatherHuBERT package contract is `waveform [1,-1] f32 -> hidden [1,-1,output_dim] f32`, with `architecture_version = "feather-hubert-burn-v1"`.
- Do not read, modify, stage, commit, or delete `demo/kanghui_training_video_featherhubert_188_latest/kanghui_training_video.MOV`.
- The supplied `.pth` may be read only as the explicitly requested model input and must remain unchanged.
- Never use `git add .`; stage only named files.
- Do not use subagents; execute inline and continue automatically to ONNX after fresh verification.

---

## File Map

- Modify `rust/Cargo.toml`: add the export crate first, then add the CLI member when its files exist; add the reusable `time` dependency.
- Modify `rust/Cargo.lock`: record the new local packages.
- Create `rust/crates/feathertalk-export/Cargo.toml`: package-format dependencies.
- Create `rust/crates/feathertalk-export/src/lib.rs`: crate-root public API.
- Create `rust/crates/feathertalk-export/src/error.rs`: structured package errors.
- Create `rust/crates/feathertalk-export/src/manifest.rs`: strict manifest, model description, tensor, file, source, training, and license schemas.
- Create `rust/crates/feathertalk-export/src/io.rs`: bounded reads, SHA-256, symlink checks, sync, staging, exact directory entries, and no-clobber publication.
- Create `rust/crates/feathertalk-export/src/package.rs`: generic module audit, write, strict load, and round-trip comparison.
- Create `rust/crates/feathertalk-export/src/feather_hubert.rs`: FeatherHuBERT `.pth` conversion request/report and expected model contract.
- Create `rust/crates/feathertalk-export/tests/manifest.rs`: schema validation.
- Create `rust/crates/feathertalk-export/tests/package.rs`: generic package round-trip, corruption, atomicity, and preflight tests.
- Create `rust/crates/feathertalk-export/tests/feather_hubert.rs`: micro fixture conversion.
- Create `rust/crates/feathertalk-export/tests/feather_hubert_real.rs`: explicit real checkpoint package smoke test.
- Create `rust/tools/model-package/Cargo.toml`: CLI package.
- Create `rust/tools/model-package/src/main.rs`: `feathertalk-model-package feather-hubert` command.
- Modify `docs/WEIGHTS.md`: document standard package creation and honest license requirements.

---

### Task 1: Define strict package and license schemas

**Files:**
- Modify: `rust/Cargo.toml`
- Modify: `rust/Cargo.lock`
- Create: `rust/crates/feathertalk-export/Cargo.toml`
- Create: `rust/crates/feathertalk-export/src/lib.rs`
- Create: `rust/crates/feathertalk-export/src/error.rs`
- Create: `rust/crates/feathertalk-export/src/manifest.rs`
- Test: `rust/crates/feathertalk-export/tests/manifest.rs`

**Interfaces:**
- Produces `ModelPackageManifest`, `ModelDescription`, `ModelConfiguration`, `TensorContract`, `TensorSpec`, `SourceManifest`, `FileManifest`, `TrainingManifest`, `LicenseBundle`, and `LicenseEntry`.
- Produces `ModelDescription::feather_hubert(config)` and `validate()` methods.
- Produces constants for schema versions, architecture versions, fixed file names, and size limits.
- `ModelPackageManifest` stores `model_type`, `architecture_version`, `configuration`, `inputs`, and `outputs` as top-level JSON/Rust fields; `ModelDescription` contains the same five fields for build/load compatibility checks.
- `TensorContract` stores `tensor_count`, `total_elements`, and sorted `entries: Vec<TensorSpec>`.

- [ ] **Step 1: Write the failing schema tests**

Create tests that construct:

```rust
let description = ModelDescription::feather_hubert(FeatherHubertConfig {
    channels: 32,
    expansion: 2,
    num_blocks: 2,
    output_dim: 64,
    dropout: 0.0,
});
assert_eq!(description.inputs[0].name, "waveform");
assert_eq!(description.inputs[0].shape, vec![1, -1]);
assert_eq!(description.outputs[0].shape, vec![1, -1, 64]);
description.validate().unwrap();
```

Construct a schema-one manifest with sorted `TensorSpec` values and `FileManifest` values, call `validate`, serialize/deserialize, and assert equality. Add separate tests that reject an unknown JSON field, uppercase/short hashes, empty license entries, duplicate tensor names, unsorted tensors, zero fixed dimensions, invalid dynamic dimensions other than `-1`, non-finite/negative loss values, and a single training-only file without its pair.

- [ ] **Step 2: Run RED**

From `rust/`:

```powershell
cargo test -p feathertalk-export --test manifest
```

Expected: Cargo fails because the package and schema types do not exist.

- [ ] **Step 3: Add the crate and schema implementation**

Add this workspace member:

```toml
"crates/feathertalk-export",
```

Add `time = { version = "0.3.55", features = ["parsing", "formatting", "macros"] }` to workspace dependencies and change `feathertalk-project` to `time.workspace = true`.

Use exact serde tagging:

```rust
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelConfiguration {
    FeatherHubert { channels: usize, expansion: usize, num_blocks: usize, output_dim: usize, dropout: f64 },
    OriginalUnet { channels: [usize; 5] },
    MobileOneUnet { channels: [usize; 5], num_conv_branches: usize, reparameterized: bool },
}
```

Use `i64` tensor dimensions so only `-1` represents a dynamic dimension. Validate `created_at` as RFC 3339 with `time::OffsetDateTime::parse`, validate `minimum_app_version` as three dot-separated `u64` components, and require all package tensor dtypes to equal `"f32"` in schema one.

- [ ] **Step 4: Run GREEN and format**

```powershell
cargo test -p feathertalk-export --test manifest
cargo fmt --all -- --check
```

- [ ] **Step 5: Commit**

```powershell
git add rust/Cargo.toml rust/Cargo.lock rust/crates/feathertalk-project/Cargo.toml rust/crates/feathertalk-export/Cargo.toml rust/crates/feathertalk-export/src/lib.rs rust/crates/feathertalk-export/src/error.rs rust/crates/feathertalk-export/src/manifest.rs rust/crates/feathertalk-export/tests/manifest.rs
git commit -m "feat: define standard model package schema"
```

---

### Task 2: Add generic atomic package writing and strict loading

**Files:**
- Create: `rust/crates/feathertalk-export/src/io.rs`
- Create: `rust/crates/feathertalk-export/src/package.rs`
- Modify: `rust/crates/feathertalk-export/src/lib.rs`
- Test: `rust/crates/feathertalk-export/tests/package.rs`

**Interfaces:**
- Produces `PackageBuildRequest`, `PackageBuildReport`, `write_model_package<B, M, F>()`, and `load_model_package<B, M, F>()`.
- `write_model_package` consumes a model reference, a factory that creates a fresh empty model, description, source, license path, creation time, minimum app version, training metadata, and absent destination.
- `load_model_package` consumes an expected description, directory, device, and fresh-model factory, returning `(M, ModelPackageManifest)`.
- The factory boundary avoids Burn `Clone` sharing BatchNorm `RunningState`; no caller-owned model can be mutated on load failure.

- [ ] **Step 1: Write the failing generic package tests**

Use `burn::nn::Linear<CpuBackend>` as a small real Burn module. Build a `ModelDescription` with a test-only Original UNet configuration and a matching source/license record, pass `|device| LinearConfig::new(2, 2).init(device)` as the fresh-model factory, then assert the published directory contains exactly:

```rust
["LICENSES.json", "manifest.json", "model.safetensors"]
```

Load through the fresh-model factory and compare every `ModuleSnapshot` path, shape, dtype, and `TensorData`. Add tests for existing destination preservation, invalid license leaving destination absent, tampered model hash, extra directory entry, corrupt safetensors, wrong expected description, and a late validation hook failure that cleans staging.

- [ ] **Step 2: Run RED**

```powershell
cargo test -p feathertalk-export --test package
```

Expected: compilation fails because package I/O APIs do not exist.

- [ ] **Step 3: Implement filesystem and hashing primitives**

In `io.rs`, implement:

```text
reject_symlink_components
validate_existing_directory
ensure_destination_absent
read_bounded_regular
write_synced_create_new
sha256_file
file_manifest
exact_directory_entries
sync_directory
StagingDirectory drop cleanup
publish_no_clobber
```

Use a process ID plus `AtomicU64` staging suffix in the destination parent. Never call recursive deletion on any path not created by `StagingDirectory`.

- [ ] **Step 4: Implement generic package writer/loader**

`module_tensor_contract` collects `module.collect(None, None, false)`, maps `DType::name()`, requires `F32`, checks unique sorted full paths, and calculates total elements with checked arithmetic.

`write_model_package` writes safetensors with `overwrite(false)`, copies validated license bytes, constructs file manifests, writes manifest last, re-parses all JSON, invokes the supplied factory to create a new staged-load target, compares tensor data, validates exact entries, then publishes.

`load_model_package` validates all directory/JSON/hash/tensor contract data before invoking the factory and before `SafetensorsStore::load_from`. Reject every missing, skipped, unused, shape, and dtype result. Do not clone or mutate a caller-owned module.

- [ ] **Step 5: Run GREEN**

```powershell
cargo test -p feathertalk-export --test package -- --nocapture
cargo clippy -p feathertalk-export --all-targets -- -D warnings
```

- [ ] **Step 6: Commit**

```powershell
git add rust/crates/feathertalk-export/src/io.rs rust/crates/feathertalk-export/src/package.rs rust/crates/feathertalk-export/src/lib.rs rust/crates/feathertalk-export/tests/package.rs
git commit -m "feat: publish standard model packages atomically"
```

---

### Task 3: Convert FeatherHuBERT checkpoints and add the CLI

**Files:**
- Create: `rust/crates/feathertalk-export/src/feather_hubert.rs`
- Modify: `rust/crates/feathertalk-export/src/lib.rs`
- Create: `rust/crates/feathertalk-export/tests/feather_hubert.rs`
- Modify: `rust/Cargo.toml`
- Modify: `rust/Cargo.lock`
- Create: `rust/tools/model-package/Cargo.toml`
- Create: `rust/tools/model-package/src/main.rs`
- Modify: `docs/WEIGHTS.md`

**Interfaces:**
- Produces `FeatherHubertPackageRequest`, `FeatherHubertPackageReport`, and `build_feather_hubert_package()`.
- CLI syntax:

```powershell
cargo run -p feathertalk-model-package -- feather-hubert `
  --source <checkpoint.pth> `
  --licenses <reviewed-LICENSES.json> `
  --destination <new-package-directory> `
  --created-at <RFC3339> `
  --minimum-app-version 0.1.0
```

- [ ] **Step 1: Write the failing micro conversion test**

Extract `weights/feather_micro.pth` from `rust/tests/golden/burn-feasibility-v1.zip`, create a synthetic test-only license file, invoke `build_feather_hubert_package`, and assert:

```rust
assert_eq!(report.manifest.model_type, "feather_hubert");
assert_eq!(report.manifest.architecture_version, "feather-hubert-burn-v1");
assert_eq!(report.manifest.source.sha256.len(), 64);
assert_eq!(report.manifest.tensors.tensor_count, 35);
assert_eq!(report.manifest.tensors.total_elements, 472_384);
assert_eq!(report.manifest.outputs[0].shape, vec![1, -1, 64]);
```

Reload through `load_model_package` into `FeatherHubertConfig::parity_micro().init`, execute `[1,1360]`, and assert `[1,4,64]` finite output.

- [ ] **Step 2: Run RED**

```powershell
cargo test -p feathertalk-export --test feather_hubert
```

- [ ] **Step 3: Implement immutable-source conversion**

Validate and snapshot the `.pth` before import. Call `load_feather_hubert_checkpoint::<CpuBackend>` on the snapshot, require its hash/count/elements to match the snapshot audit, derive `ModelDescription::feather_hubert`, and pass the loaded module to `write_model_package`. Set source fields to:

```text
format = "pytorch-pickle-restricted"
identifier = "feathertalk-feather-hubert"
version = source file stem
file_name = source file name
sha256 = snapshot SHA-256
```

Use the package writer's internal pre-publish validation hook to check the original source length and SHA-256 again after staged round-trip validation but before the atomic rename. If it changed, return an error while destination is still absent. Hash it once more after success in acceptance tests to prove the tool itself made no modification.

- [ ] **Step 4: Implement CLI and documentation**

Add `"tools/model-package"` to the workspace only after creating its `Cargo.toml` and `src/main.rs`. Use clap subcommands. Print destination, source SHA-256, model SHA-256, tensor count, total elements, and FeatherHuBERT configuration. Do not accept a license-free flag. Document that the demo checkpoint can be the source but the caller must provide an honestly reviewed `LICENSES.json`; a synthetic test license is never redistribution approval.

- [ ] **Step 5: Run GREEN**

```powershell
cargo test -p feathertalk-export --test feather_hubert
cargo test -p feathertalk-model-package --all-targets
cargo clippy -p feathertalk-model-package --all-targets -- -D warnings
```

- [ ] **Step 6: Commit**

```powershell
git add rust/Cargo.toml rust/Cargo.lock rust/crates/feathertalk-export/src/feather_hubert.rs rust/crates/feathertalk-export/src/lib.rs rust/crates/feathertalk-export/tests/feather_hubert.rs rust/tools/model-package/Cargo.toml rust/tools/model-package/src/main.rs docs/WEIGHTS.md
git commit -m "feat: package FeatherHuBERT checkpoints"
```

---

### Task 4: Verify the supplied real FeatherHuBERT checkpoint

**Files:**
- Create: `rust/crates/feathertalk-export/tests/feather_hubert_real.rs`
- Create locally but do not commit: a temporary test-only `LICENSES.json` and package destination outside the protected demo directory.

**Interfaces:**
- Uses `FEATHERTALK_FEATHER_HUBERT_CHECKPOINT`; the test creates an explicit temporary local-only license record.
- Does not read any other file in the demo directory.

- [ ] **Step 1: Write the environment-gated real test**

The test skips when the variable is absent. When set, assert source length `40_436_613` and source SHA-256 `58df96af118d75d7f69da441e1f3960096f28dda637a4e8f4265f108d27aeb52`, write a temporary `LicenseRef-User-Supplied-Unreviewed` bundle outside the demo directory, build into a `tempfile` directory, and assert config `256/2/8/1024/0`, tensor count `65`, total elements `3_364_096`, exact package entries, and finite `[1,4,1024]` CPU output after strict reload. Hash the source before and after.

- [ ] **Step 2: Run the test without variables**

```powershell
cargo test -p feathertalk-export --test feather_hubert_real -- --nocapture
```

Expected: exits 0 with an explicit skip message and no demo directory access.

- [ ] **Step 3: Confirm the temporary license remains local-only**

Inspect the test to ensure its `LicenseRef-User-Supplied-Unreviewed` file is created below `tempfile::TempDir`, never beneath the repository or demo directory, and the notice states that it is a local conversion record rather than redistribution approval.

- [ ] **Step 4: Run the real checkpoint package verification**

```powershell
$env:FEATHERTALK_FEATHER_HUBERT_CHECKPOINT = (Resolve-Path 'demo/kanghui_training_video_featherhubert_188_latest/feather_hubert_188_latest_99.pth')
cargo test -p feathertalk-export --test feather_hubert_real -- --nocapture
```

Expected: one test passes, the source hash remains unchanged, and no file is created under the demo directory.

- [ ] **Step 5: Commit the real acceptance test**

```powershell
git add rust/crates/feathertalk-export/tests/feather_hubert_real.rs
git commit -m "test: verify real FeatherHuBERT model package"
```

---

### Task 5: Full verification and continuation

**Files:**
- Modify only files required by fresh verification findings.

- [ ] **Step 1: Run focused verification**

```powershell
cargo fmt --all -- --check
cargo test -p feathertalk-export --all-targets
cargo test -p feathertalk-model-package --all-targets
cargo clippy -p feathertalk-export -p feathertalk-model-package --all-targets -- -D warnings
```

- [ ] **Step 2: Run full workspace verification**

```powershell
cargo test --workspace --all-targets
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
git diff --check
```

- [ ] **Step 3: Audit requirements and protected paths**

Confirm schema coverage, license honesty, strict preflight, tensor audit, source immutability, staging cleanup, no-clobber publication, real checkpoint output, and absence of `.MOV` from command lines/diffs. Run:

```powershell
git status --short
git diff --name-only HEAD~5..HEAD
git log -5 --oneline
```

- [ ] **Step 4: Commit any verification-only fixes**

Stage exact changed paths and use:

```powershell
git commit -m "chore: verify standard model packages"
```

- [ ] **Step 5: Continue automatically**

After fresh evidence confirms this slice, reread migration design section 5.6 and begin the independent ONNX opset 17 design/plan/implementation cycle. Do not pause for an execution-option prompt and do not dispatch subagents.

---

## Plan Self-Review

- **Spec coverage:** Tasks 1-2 define the reusable schema, strict loader, hashes, tensor audit, licenses, staging, and no-clobber publication; Task 3 provides FeatherHuBERT conversion and CLI; Task 4 proves the supplied real model; Task 5 runs fresh workspace validation and continuation.
- **Placeholder scan:** All tasks name exact paths, public types, commands, expected failures, and acceptance assertions; there are no deferred implementation placeholders.
- **Type consistency:** `ModelPackageManifest`, `ModelDescription`, `PackageBuildRequest`, `write_model_package`, `load_model_package`, `FeatherHubertPackageRequest`, and `build_feather_hubert_package` retain the same names and roles throughout.
- **Scope:** ONNX, ORT, `.npy` migration, worker, GPUI, and commercial license approval stay outside this slice.
- **Safety:** No command reads the protected `.MOV`; the only demo input is the explicitly supplied `.pth`, and generated artifacts remain outside its directory.
