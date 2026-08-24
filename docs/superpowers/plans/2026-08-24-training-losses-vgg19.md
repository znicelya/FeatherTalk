# VGG19 Training Losses Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an offline, strictly packaged VGG19 `features.14` extractor and exact Rust implementations of FeatherTalk's Baseline, Mouth ROI, and Temporal training losses.

**Architecture:** Create a focused `feathertalk-training` crate for the frozen VGG graph, package loading, perceptual feature API, and loss formulas. Extend the generic `feathertalk-weights` importer with a kind-scoped torchvision key map, and add a CPU-only packaging tool that converts an explicitly supplied `.pth` plus reviewed license bundle into a three-file safetensors package. Runtime code never downloads or parses pickle.

**Tech Stack:** Rust 1.92, Burn 0.21.0, burn-store 0.21.0, NdArray CPU tests, WGPU-compatible generic tensors, serde/serde_json, SHA-256, tempfile, clap, Python 3 with torch only for golden generation.

## Global Constraints

- VGG implements torchvision `features[0..=14]` exactly and returns the raw `features.14` convolution output before `features.15` ReLU.
- VGG input remains BGR float32 `[0,1]` with no channel swap and no ImageNet normalization.
- Production `[B,3,160,160]` features are `[B,256,40,40]`; generic test inputs require non-zero batch and spatial dimensions at least 4.
- Loaded VGG parameters are frozen with `Module::no_grad`; prediction gradients remain enabled and target features are detached.
- Runtime accepts only `manifest.json`, `model.safetensors`, and `LICENSES.json`; it never downloads, searches caches, accepts `.pth`, or falls back to random weights.
- Baseline defaults to perceptual weight `0.01`; Mouth ROI defaults to mouth `4.0`, perceptual `0.01`; Temporal defaults to mouth `4.0`, temporal `0.5`, temporal-mouth `4.0`, perceptual `0.01`.
- Temporal inputs have exact pair length 2. Mouth masks are single-channel and use `sum(mask).clamp_min(1) * image_channels` as the denominator.
- Official CPU parity thresholds are `max_abs_error <= 1e-4` and `mean_abs_error <= 1e-5`.
- All public external-input failures return structured errors; tensor value-range validation must not synchronize GPU tensors back to CPU.
- WGPU never silently falls back to CPU.
- Use TDD for every production behavior, explicitly stage files, and never use `git add .`.
- Do not modify, delete, inspect for implementation input, or commit `demo/kanghui_training_video_featherhubert_188_latest/`.

---

### Task 1: Add the training crate and exact VGG19 cutoff graph

**Files:**
- Modify: `rust/Cargo.toml`
- Modify: `rust/Cargo.lock`
- Create: `rust/crates/feathertalk-training/Cargo.toml`
- Create: `rust/crates/feathertalk-training/src/lib.rs`
- Create: `rust/crates/feathertalk-training/src/error.rs`
- Create: `rust/crates/feathertalk-training/src/vgg19.rs`
- Create: `rust/crates/feathertalk-training/tests/vgg19_graph.rs`

**Interfaces:**
- Produces: `TrainingError`, `Vgg19Conv3_3<B>::new_for_import(device)`, and `Vgg19Conv3_3::forward(Tensor<B,4>) -> Tensor<B,4>`.
- Produces public convolution fields `conv1_1` through `conv3_3` so burn-store paths are stable and gradient tests can address exact parameters.
- The constructor initializes import targets only; production callers later use `load_vgg19_package` from Task 3.

- [ ] **Step 1: Write the failing graph tests**

Create `tests/vgg19_graph.rs` with a module assertion and fixed topology checks:

```rust
use burn::{module::Module, tensor::Tensor};
use feathertalk_training::Vgg19Conv3_3;

type CpuBackend = burn::backend::NdArray<f32>;

fn assert_module<M: Module<CpuBackend>>() {}

#[test]
fn vgg19_conv3_3_is_a_burn_module() {
    assert_module::<Vgg19Conv3_3<CpuBackend>>();
}

#[test]
fn vgg19_conv3_3_maps_sixteen_to_four() {
    let device = Default::default();
    let model = Vgg19Conv3_3::<CpuBackend>::new_for_import(&device);
    let output = model.forward(Tensor::zeros([1, 3, 16, 16], &device));
    assert_eq!(output.dims(), [1, 256, 4, 4]);
}
```

Add `conv3_3_output_is_not_post_relu`. Initialize the module, replace every convolution weight and bias with zero tensors, replace only `conv3_3.bias` with `-1`, run zeros, and assert every output is `-1`. This test must fail if a ReLU is added after `conv3_3`.

- [ ] **Step 2: Run the focused test and verify RED**

Run from `rust/`:

```powershell
cargo test -p feathertalk-training --test vgg19_graph
```

Expected: Cargo fails because `feathertalk-training` and `Vgg19Conv3_3` do not exist.

- [ ] **Step 3: Add the crate and minimal VGG graph**

Add only the training crate workspace member in this task:

```toml
"crates/feathertalk-training",
```

Task 4 adds `"tools/vgg19-package"` together with the tool crate files, so every workspace member exists when Cargo first resolves it.

Create the training crate dependencies:

```toml
[dependencies]
burn.workspace = true
burn-store.workspace = true
hex.workspace = true
serde.workspace = true
serde_json.workspace = true
sha2.workspace = true
thiserror.workspace = true

[dev-dependencies]
feathertalk-weights = { path = "../feathertalk-weights" }
tempfile.workspace = true
zip.workspace = true
```

Define `TrainingError` initially with the variants needed by this task and reserve exact later variants without placeholder text:

```rust
#[derive(Debug, thiserror::Error)]
pub enum TrainingError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid training input: {0}")]
    InvalidInput(String),
    #[error("invalid training configuration: {0}")]
    InvalidConfig(String),
    #[error("invalid VGG19 package: {0}")]
    InvalidPackage(String),
    #[error("VGG19 package hash mismatch for {file}: expected {expected}, got {actual}")]
    HashMismatch { file: String, expected: String, actual: String },
    #[error("Burn store error: {0}")]
    Store(String),
}
```

Implement seven `Conv2d<B>` fields with bias, explicit padding 1, and the exact channel schedule. Use `burn::tensor::module::max_pool2d` after `conv1_2` and `conv2_2`. Apply ReLU after the first six convolutions only:

```rust
let x = relu(self.conv1_1.forward(input));
let x = relu(self.conv1_2.forward(x));
let x = max_pool2d(x, [2, 2], [2, 2], [0, 0], [1, 1], false);
// conv2_1, conv2_2, pool, conv3_1, conv3_2
self.conv3_3.forward(x)
```

Assert channel 3, non-zero batch, and spatial dimensions at least 4 at the start of `forward` with messages naming the VGG input contract.

- [ ] **Step 4: Run focused and formatting tests**

```powershell
cargo test -p feathertalk-training --test vgg19_graph -- --nocapture
cargo fmt --all -- --check
```

Expected: all VGG graph tests pass; the negative bias remains negative after forward.

- [ ] **Step 5: Commit the graph**

```powershell
git add rust/Cargo.toml rust/Cargo.lock rust/crates/feathertalk-training/Cargo.toml rust/crates/feathertalk-training/src/lib.rs rust/crates/feathertalk-training/src/error.rs rust/crates/feathertalk-training/src/vgg19.rs rust/crates/feathertalk-training/tests/vgg19_graph.rs
git commit -m "feat: add VGG19 conv3_3 graph"
```

### Task 2: Extend strict PyTorch import for the truncated VGG graph

**Files:**
- Modify: `rust/crates/feathertalk-weights/src/key_map.rs`
- Modify: `rust/crates/feathertalk-weights/src/legacy.rs`
- Modify: `rust/crates/feathertalk-weights/src/lib.rs`
- Create: `rust/crates/feathertalk-training/tests/vgg19_import.rs`
- Create: `rust/tools/parity/generate_vgg19_import_fixture.py`
- Create: `rust/tests/golden/vgg19-import-v1.zip`
- Create: `rust/tests/golden/vgg19-import-v1.sha256`

**Interfaces:**
- Produces: `LegacyModelKind::Vgg19Conv3_3`.
- Preserves: `is_known_ignored_key(key)` for generic BatchNorm counter callers.
- Produces: `is_known_ignored_key_for(kind, key)` for kind-specific validation.
- `import_into` must report exactly 14 applied tensors, 24 ignored tensors, and 1,735,488 applied elements for the fixture.

- [ ] **Step 1: Create the deterministic official-shape fixture generator**

Create a Python script using only `torch`, `json`, `hashlib`, `tempfile`, and `zipfile`. It must construct a direct state dict with these relevant shapes:

```python
RELEVANT = {
    0: (64, 3),
    2: (64, 64),
    5: (128, 64),
    7: (128, 128),
    10: (256, 128),
    12: (256, 256),
    14: (256, 256),
}

state = {}
for ordinal, (index, (out_channels, in_channels)) in enumerate(RELEVANT.items(), 1):
    state[f"features.{index}.weight"] = torch.full(
        (out_channels, in_channels, 3, 3), ordinal / 1000.0, dtype=torch.float32
    )
    state[f"features.{index}.bias"] = torch.full(
        (out_channels,), -ordinal / 100.0, dtype=torch.float32
    )
```

Add scalar float32 tensors for all 24 allowed ignored keys:

```python
for index in [16, 19, 21, 23, 25, 28, 30, 32, 34]:
    state[f"features.{index}.weight"] = torch.tensor([1.0])
    state[f"features.{index}.bias"] = torch.tensor([1.0])
for index in [0, 3, 6]:
    state[f"classifier.{index}.weight"] = torch.tensor([1.0])
    state[f"classifier.{index}.bias"] = torch.tensor([1.0])
```

Save `vgg19-direct.pth`, add a second `vgg19-unexpected.pth` with `unexpected.weight`, write a manifest containing SHA-256 values, and zip them with DEFLATE. Write the zip SHA-256 as one lowercase hash plus two spaces and the zip file name.

- [ ] **Step 2: Generate and inspect the fixture**

Run from repository root:

```powershell
python rust/tools/parity/generate_vgg19_import_fixture.py
```

Expected: creates the zip and sidecar under `rust/tests/golden/`; `vgg19-direct.pth` is bounded and contains 38 tensor keys.

- [ ] **Step 3: Write failing key-map and import tests**

Add these key-map cases to the `key_map.rs` unit tests:

```rust
assert_eq!(
    map_key(LegacyModelKind::Vgg19Conv3_3, "features.14.weight"),
    "conv3_3.weight"
);
assert!(is_known_ignored_key_for(
    LegacyModelKind::Vgg19Conv3_3,
    "features.16.weight"
));
assert!(!is_known_ignored_key_for(
    LegacyModelKind::OriginalUnet,
    "features.16.weight"
));
```

In `feathertalk-training/tests/vgg19_import.rs`, extract `vgg19-direct.pth`, construct `Vgg19Conv3_3::<CpuBackend>::new_for_import`, call `import_into`, and assert:

```rust
assert_eq!(report.applied.len(), 14);
assert_eq!(report.ignored.len(), 24);
assert_eq!(report.tensor_count, 14);
assert_eq!(report.total_elements, 1_735_488);
```

Also assert the first `conv1_1` weight and `conv3_3` bias match the literal constants from the generator. Add an unexpected-key test and a shape-mismatch test that both preserve the caller module snapshot. Keep these cross-crate importer tests in `feathertalk-training`, which already has a development dependency on `feathertalk-weights`; do not add a reverse `feathertalk-weights -> feathertalk-training` development dependency.

- [ ] **Step 4: Run the focused tests and verify RED**

```powershell
cargo test -p feathertalk-training --test vgg19_import -- --nocapture
```

Expected: compilation fails because the new model kind and kind-specific ignore function do not exist.

- [ ] **Step 5: Implement exact remapping and kind-scoped ignores**

Add the enum variant and these exact remap patterns:

```rust
.add_pattern(r"^features\.0\.", "conv1_1.")
.add_pattern(r"^features\.2\.", "conv1_2.")
.add_pattern(r"^features\.5\.", "conv2_1.")
.add_pattern(r"^features\.7\.", "conv2_2.")
.add_pattern(r"^features\.10\.", "conv3_1.")
.add_pattern(r"^features\.12\.", "conv3_2.")
.add_pattern(r"^features\.14\.", "conv3_3.")
```

Keep:

```rust
pub fn is_known_ignored_key(key: &str) -> bool {
    key.ends_with(".num_batches_tracked")
}
```

Add:

```rust
pub fn is_known_ignored_key_for(kind: LegacyModelKind, key: &str) -> bool {
    if is_known_ignored_key(key) {
        return true;
    }
    matches!(kind, LegacyModelKind::Vgg19Conv3_3) && is_vgg19_truncated_key(key)
}
```

`is_vgg19_truncated_key` must parse only the exact literal paths listed in the spec; do not accept broad `features.*` or `classifier.*` patterns.

Thread `request.kind` through ignored collection and `validate_apply_result`. Existing FeatherHuBERT and Original UNet behavior must remain unchanged.

- [ ] **Step 6: Run import and regression tests**

```powershell
cargo test -p feathertalk-training --test vgg19_import -- --nocapture
cargo test -p feathertalk-weights
cargo test -p feathertalk-parity --test unet_checkpoint_import
```

Expected: VGG counts are exact; existing import tests pass.

- [ ] **Step 7: Commit the importer**

The zip is globally ignored, so force-add only the reviewed generated fixture:

```powershell
git add rust/crates/feathertalk-weights/src/key_map.rs rust/crates/feathertalk-weights/src/legacy.rs rust/crates/feathertalk-weights/src/lib.rs rust/crates/feathertalk-training/tests/vgg19_import.rs rust/tools/parity/generate_vgg19_import_fixture.py rust/tests/golden/vgg19-import-v1.sha256
git add -f rust/tests/golden/vgg19-import-v1.zip
git commit -m "feat: import truncated VGG19 weights"
```

### Task 3: Add the strict three-file VGG model package loader

**Files:**
- Modify: `rust/crates/feathertalk-training/src/lib.rs`
- Modify: `rust/crates/feathertalk-training/src/error.rs`
- Create: `rust/crates/feathertalk-training/src/artifact.rs`
- Create: `rust/crates/feathertalk-training/tests/vgg19_package.rs`

**Interfaces:**
- Produces constants `VGG19_PACKAGE_SCHEMA_VERSION`, `VGG19_ARCHITECTURE_VERSION`, `VGG19_MODEL_KIND`, and `VGG19_SOURCE_URL`.
- Produces manifest structs `Vgg19PackageManifest`, `Vgg19SourceManifest`, `Vgg19InputManifest`, `Vgg19FileManifest`, `Vgg19LicenseBundle`, and `Vgg19LicenseEntry`.
- Produces `load_vgg19_package<B: Backend>(directory, device) -> Result<Vgg19Conv3_3<B>, TrainingError>`.
- Produces validation methods used by the packaging tool in Task 4.

- [ ] **Step 1: Write failing valid-package and corruption tests**

Build a test helper that:

1. creates `Vgg19Conv3_3::<CpuBackend>::new_for_import`;
2. saves it with `feathertalk_weights::save_safetensors`;
3. writes a license bundle with one literal local-test entry;
4. hashes both files;
5. serializes the exact manifest;
6. returns the temp directory and original model snapshot.

The success test must load the package and compare all 14 tensors. Add separate tests for:

- unknown manifest field;
- wrong model hash;
- wrong declared byte length;
- missing and extra directory entry;
- symlink model file where supported;
- model file over 16 MiB rejected before store load;
- missing safetensors tensor;
- extra safetensors tensor;
- invalid or empty license bundle;
- wrong BGR/no-normalization input contract.

- [ ] **Step 2: Run the package tests and verify RED**

```powershell
cargo test -p feathertalk-training --test vgg19_package -- --nocapture
```

Expected: compilation fails because artifact types and loader do not exist.

- [ ] **Step 3: Implement bounded file and manifest validation**

Use `symlink_metadata` before opening every path. Sort directory entry names and require exact equality with:

```rust
["LICENSES.json", "manifest.json", "model.safetensors"]
```

Implement `read_bounded(path, limit)` by reading at most `limit + 1` bytes and rejecting overflow. Hash model and license files with a 64 KiB streaming buffer. Validate every literal schema value and these exact counts:

```rust
tensor_count == 14
total_elements == 1_735_488
model.file_name == "model.safetensors"
licenses.file_name == "LICENSES.json"
```

Use `#[serde(deny_unknown_fields)]` on every manifest and license struct. A license bundle requires `schema_version == 1`, at least one entry, and trimmed non-empty `component`, `license_id`, `source_url`, and `notice`.

- [ ] **Step 4: Implement strict safetensors load and freeze**

Initialize `Vgg19Conv3_3::new_for_import`, load through `SafetensorsStore`, and reject the first item in `missing`, `errors`, or `unused`. Collect the loaded module snapshot and require exact tensor paths, float32 dtypes, shapes, count 14, and total elements 1,735,488. Return `model.no_grad()` only after every check passes.

Expose the parsed manifest through `read_vgg19_manifest(directory)` so the official parity test can bind the fixture source hash to the package.

- [ ] **Step 5: Run package, graph, and clippy tests**

```powershell
cargo test -p feathertalk-training --test vgg19_package -- --nocapture
cargo test -p feathertalk-training --test vgg19_graph
cargo clippy -p feathertalk-training --all-targets --all-features -- -D warnings
```

Expected: all pass with no warning or random fallback path.

- [ ] **Step 6: Commit the runtime package loader**

```powershell
git add rust/crates/feathertalk-training/src/lib.rs rust/crates/feathertalk-training/src/error.rs rust/crates/feathertalk-training/src/artifact.rs rust/crates/feathertalk-training/tests/vgg19_package.rs
git commit -m "feat: load strict VGG19 packages"
```

### Task 4: Build the atomic VGG19 packaging tool

**Files:**
- Modify: `rust/Cargo.toml`
- Modify: `rust/Cargo.lock`
- Create: `rust/tools/vgg19-package/Cargo.toml`
- Create: `rust/tools/vgg19-package/src/lib.rs`
- Create: `rust/tools/vgg19-package/src/main.rs`
- Create: `rust/tools/vgg19-package/tests/package.rs`

**Interfaces:**
- Produces `Vgg19PackageRequest { source, licenses, destination }`.
- Produces `Vgg19PackageReport { manifest }`.
- Produces `build_vgg19_package(&request) -> Result<Vgg19PackageReport, PackageError>` using `NdArray<f32>` only.
- CLI flags are exactly `--source`, `--licenses`, and `--destination`.

- [ ] **Step 1: Write failing end-to-end tool tests**

Extract the direct VGG fixture from Task 2, create a valid local-test license JSON, choose a non-existent destination, and call `build_vgg19_package`. Assert:

```rust
assert_eq!(report.manifest.tensor_count, 14);
assert_eq!(report.manifest.total_elements, 1_735_488);
assert!(destination.join("manifest.json").is_file());
assert!(destination.join("model.safetensors").is_file());
assert!(destination.join("LICENSES.json").is_file());
```

Load the destination with `load_vgg19_package` and compare all tensors to a separately imported candidate. Add failure tests for existing destination, invalid licenses, source with unexpected key, and a late staged manifest corruption hook. Every failure must leave destination absent.

- [ ] **Step 2: Run the tool test and verify RED**

```powershell
cargo test -p feathertalk-vgg19-package --test package -- --nocapture
```

Expected: Cargo fails because the tool crate and package function do not exist.

- [ ] **Step 3: Implement immutable import and staged publication**

The tool library must:

1. reject an existing destination and missing/non-directory parent;
2. parse and validate licenses before importing the large source;
3. initialize `Vgg19Conv3_3<CpuBackend>`;
4. call `import_into` with `LegacyModelKind::Vgg19Conv3_3`, direct state dict, a 1 GiB file limit, 64 tensor limit, and 200 million element limit;
5. require exact report counts 14/24/1,735,488;
6. create a `tempfile::TempDir` in the destination parent;
7. save safetensors and strictly reload it;
8. compare candidate and reload snapshots path/shape/dtype/data;
9. copy the already validated license bytes with `create_new`;
10. calculate file lengths and SHA-256 values;
11. write, read, and validate the manifest;
12. call `load_vgg19_package` on staging;
13. rename staging to destination, then call `TempDir::keep()` only after the rename succeeds.

Return structured `PackageError` variants for training, weight import, I/O, invalid request, and publication errors. Do not print and continue after any failed apply result.

- [ ] **Step 4: Implement the CLI wrapper**

Use a clap parser with native `PathBuf` fields. On success print the destination, source hash, model hash, and tensor count. On failure return a non-zero process exit through `Result<(), PackageError>`; do not catch errors and exit zero.

- [ ] **Step 5: Run tool and workspace regressions**

```powershell
cargo test -p feathertalk-vgg19-package -- --nocapture
cargo test -p feathertalk-training --test vgg19_package
cargo test -p feathertalk-weights --test legacy_import
```

- [ ] **Step 6: Commit the packaging tool**

```powershell
git add rust/Cargo.toml rust/Cargo.lock rust/tools/vgg19-package/Cargo.toml rust/tools/vgg19-package/src/lib.rs rust/tools/vgg19-package/src/main.rs rust/tools/vgg19-package/tests/package.rs
git commit -m "feat: package VGG19 perceptual weights"
```

### Task 5: Add the frozen perceptual feature and MSE API

**Files:**
- Modify: `rust/crates/feathertalk-training/src/lib.rs`
- Create: `rust/crates/feathertalk-training/src/perceptual.rs`
- Create: `rust/crates/feathertalk-training/tests/perceptual_loss.rs`

**Interfaces:**
- Produces `PerceptualFeatureExtractor<B>` with `forward(Tensor<B,4>) -> Tensor<B,4>`.
- Implements the trait for `Vgg19Conv3_3<B>`.
- Produces `perceptual_mse(extractor, prediction, target) -> Result<Tensor<B,1>, TrainingError>`.

- [ ] **Step 1: Write failing formula and gradient tests**

Define a test-only identity extractor. Verify two literal tensors produce the hand-computed mean squared error. Verify identical inputs produce exactly zero.

Add an autodiff test using a small `Vgg19Conv3_3<CpuAutodiffBackend>::new_for_import(&device).no_grad()`:

```rust
let prediction = Tensor::ones([1, 3, 8, 8], &device).require_grad();
let target = Tensor::zeros([1, 3, 8, 8], &device);
let loss = perceptual_mse(&vgg, prediction.clone(), target).unwrap();
let gradients = loss.backward();
assert!(prediction.grad(&gradients).is_some());
assert!(vgg.conv1_1.weight.val().grad(&gradients).is_none());
```

Add shape mismatch, wrong channel, empty batch, and spatial-smaller-than-four errors.

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
cargo test -p feathertalk-training --test perceptual_loss -- --nocapture
```

Expected: compilation fails because the trait and function do not exist.

- [ ] **Step 3: Implement validation, target detach, and MSE**

Validate both input shapes before invoking the extractor. Compute:

```rust
let predicted = extractor.forward(prediction);
let expected = extractor.forward(target).detach();
(predicted - expected).square().mean()
```

Do not normalize, swap channels, call `into_data`, or create a non-autodiff backend tensor.

- [ ] **Step 4: Run perceptual and package tests**

```powershell
cargo test -p feathertalk-training --test perceptual_loss -- --nocapture
cargo test -p feathertalk-training --test vgg19_package
```

- [ ] **Step 5: Commit the perceptual API**

```powershell
git add rust/crates/feathertalk-training/src/lib.rs rust/crates/feathertalk-training/src/perceptual.rs rust/crates/feathertalk-training/tests/perceptual_loss.rs
git commit -m "feat: add frozen perceptual loss"
```

### Task 6: Implement Baseline, Mouth ROI, and Temporal losses

**Files:**
- Modify: `rust/crates/feathertalk-training/src/lib.rs`
- Create: `rust/crates/feathertalk-training/src/losses.rs`
- Create: `rust/crates/feathertalk-training/tests/training_losses.rs`

**Interfaces:**
- Produces `BaselineLossConfig`, `MouthRoiLossConfig`, `TemporalLossConfig`, all serde serializable/deserializable with exact defaults.
- Produces `LossBreakdown<B>` with `total`, `full`, `perceptual`, `mouth`, `temporal`, and `temporal_mouth`.
- Produces `baseline_loss`, `mouth_roi_loss`, `temporal_loss`, and public `mouth_l1_loss`.

- [ ] **Step 1: Write failing config and hand-computed loss tests**

Use an identity feature extractor so perceptual MSE is computed directly on images.

Baseline fixture:

```rust
prediction = [1.0, 0.0]
target     = [0.0, 0.0]
full       = 0.5
perceptual = 0.5
total      = 0.505
```

Mouth fixture uses a three-channel `[1,3,1,2]` prediction with mask `[1,1,1,2] = [1,0]`. Derive literal full, mouth, perceptual, and total values by hand; assert every component within `1e-6`.

Temporal fixture uses `[1,2,3,1,2]` with different first and second frame errors. Assert full, mouth, temporal, temporal-mouth, perceptual, and total literals independently. The expected union mask must be a literal, not computed with the production helper.

Add config tests for exact defaults and rejection of negative, NaN, and infinity weights.

- [ ] **Step 2: Add failing edge and gradient tests**

Cover:

- zero mouth mask returns mouth 0 and finite total;
- mask channel other than 1 fails;
- mismatched mask batch/spatial fails;
- Temporal pair length 1 or 3 fails;
- target shape mismatch fails before extractor invocation;
- all three totals give an autodiff prediction tensor a gradient;
- unused `LossBreakdown` optional fields are `None` for Baseline and the expected fields are `Some` for other modes.

- [ ] **Step 3: Run tests and verify RED**

```powershell
cargo test -p feathertalk-training --test training_losses -- --nocapture
```

Expected: compilation fails because configs, breakdown, and loss functions do not exist.

- [ ] **Step 4: Implement validated configs and common helpers**

Implement defaults exactly from the spec. Each config has `validate() -> Result<(), TrainingError>` checking `value.is_finite() && value >= 0.0`.

Implement:

```rust
pub fn mouth_l1_loss<B: Backend>(
    prediction: Tensor<B, 4>,
    target: Tensor<B, 4>,
    mask: Tensor<B, 4>,
) -> Result<Tensor<B, 1>, TrainingError>
```

After shape validation:

```rust
let channels = prediction.dims()[1] as f64;
let denominator = mask.clone().sum().clamp_min(1.0).mul_scalar(channels);
Ok(((prediction - target).abs() * mask).sum() / denominator)
```

Confirm Burn broadcasting is `[B,3,H,W] * [B,1,H,W]`; do not manually repeat masks unless the backend rejects broadcast in a focused RED test.

- [ ] **Step 5: Implement the three public loss functions**

Baseline and Mouth ROI operate on rank-4 tensors and call `perceptual_mse` once.

Temporal accepts rank-5 tensors. Validate `[B,2,3,H,W]` and `[B,2,1,H,W]`, flatten pair into batch for full/mouth/perceptual, and use literal slices for the two deltas:

```rust
let first_prediction = prediction.clone().slice([0..batch, 0..1, 0..3, 0..height, 0..width]).squeeze_dim(1);
let second_prediction = prediction.clone().slice([0..batch, 1..2, 0..3, 0..height, 0..width]).squeeze_dim(1);
```

Build union mask with an elementwise max across pair dimension, squeeze it to `[B,1,H,W]`, then call `mouth_l1_loss` on deltas. Combine cloned component tensors with config weights and populate every `LossBreakdown` field.

- [ ] **Step 6: Run loss, perceptual, and model-gradient regressions**

```powershell
cargo test -p feathertalk-training --test training_losses -- --nocapture
cargo test -p feathertalk-training --test perceptual_loss
cargo test -p feathertalk-models --test train_step
cargo test -p feathertalk-models --test mobileone_unet mobileone_output_weight_receives_gradient -- --exact
```

Expected: all pass and all scalar components are finite.

- [ ] **Step 7: Commit the three loss modes**

```powershell
git add rust/crates/feathertalk-training/src/lib.rs rust/crates/feathertalk-training/src/losses.rs rust/crates/feathertalk-training/tests/training_losses.rs
git commit -m "feat: add FeatherTalk training losses"
```

### Task 7: Add official VGG19 Python/Burn parity and weight documentation

**Files:**
- Modify: `rust/crates/feathertalk-training/Cargo.toml`
- Create: `rust/tools/parity/generate_vgg19_golden.py`
- Create: `rust/crates/feathertalk-training/tests/vgg19_official_parity.rs`
- Create: `rust/tests/golden/vgg19-conv3-3-v1.zip`
- Create: `rust/tests/golden/vgg19-conv3-3-v1.sha256`
- Create: `rust/tests/fixtures/vgg19/LICENSES.local-parity.json`
- Modify: `docs/WEIGHTS.md`

**Interfaces:**
- Golden generator accepts `--source` and writes the fixed repository zip plus sidecar.
- Ignored external-package test reads `FEATHERTALK_VGG19_PACKAGE` and compares its source hash and output to the committed fixture.
- Normal `cargo test` never downloads and leaves the external test ignored.

- [ ] **Step 1: Write the golden generator**

Use `argparse`, `hashlib`, `io`, `json`, `numpy`, `torch`, and `zipfile`; do not import torchvision. Load the direct state dict on CPU with `weights_only=True`. Generate a deterministic BGR input:

```python
input_tensor = torch.linspace(
    0.0, 1.0, steps=1 * 3 * 16 * 16, dtype=torch.float32
).reshape(1, 3, 16, 16)
```

Execute exactly:

```python
x = F.relu(F.conv2d(x, state["features.0.weight"], state["features.0.bias"], padding=1))
x = F.relu(F.conv2d(x, state["features.2.weight"], state["features.2.bias"], padding=1))
x = F.max_pool2d(x, 2, 2)
x = F.relu(F.conv2d(x, state["features.5.weight"], state["features.5.bias"], padding=1))
x = F.relu(F.conv2d(x, state["features.7.weight"], state["features.7.bias"], padding=1))
x = F.max_pool2d(x, 2, 2)
x = F.relu(F.conv2d(x, state["features.10.weight"], state["features.10.bias"], padding=1))
x = F.relu(F.conv2d(x, state["features.12.weight"], state["features.12.bias"], padding=1))
x = F.conv2d(x, state["features.14.weight"], state["features.14.bias"], padding=1)
```

Write `input.npy`, `expected.npy`, and `manifest.json` into a deterministic DEFLATE zip. Manifest records schema, source SHA-256, shapes, dtype, BGR/range/normalization, and each member hash. Write the zip sidecar.

- [ ] **Step 2: Add the ignored external-package parity test**

Add dev dependencies `ndarray.workspace = true` and `ndarray-npy.workspace = true`. The test must:

1. require `FEATHERTALK_VGG19_PACKAGE` with a clear panic message when explicitly run;
2. verify the zip sidecar before reading members;
3. deserialize the fixture manifest with unknown fields denied;
4. compare fixture source SHA-256 to `read_vgg19_manifest(package).source.sha256`;
5. load input and expected arrays;
6. run `load_vgg19_package::<CpuBackend>` and forward;
7. compute max and mean absolute errors;
8. assert `max <= 1e-4` and `mean <= 1e-5`.

Mark only this test `#[ignore = "requires an explicitly supplied licensed VGG19 package"]`.

- [ ] **Step 3: Add an honest local-parity license record**

Create:

```json
{
  "schema_version": 1,
  "entries": [
    {
      "component": "torchvision VGG19 IMAGENET1K_V1 pretrained weights",
      "license_id": "LicenseRef-Local-Parity-Only",
      "source_url": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
      "notice": "Local numerical-parity testing only. Redistribution requires separate license review and an approved release license bundle."
    }
  ]
}
```

This file is for local validation only and must not be described as release redistribution approval.

- [ ] **Step 4: Download the official source with explicit network approval**

Use an explicit temporary file, not a repository path:

```powershell
$vggSource = Join-Path $env:TEMP 'feathertalk-vgg19-dcbb9e9d.pth'
curl.exe -L --fail --output $vggSource https://download.pytorch.org/models/vgg19-dcbb9e9d.pth
Get-FileHash -Algorithm SHA256 $vggSource
```

Expected: a non-empty official source file and a 64-character SHA-256. If network access is sandbox-blocked, rerun with the required approval rather than adding automatic downloader code.

- [ ] **Step 5: Generate the committed official golden fixture**

```powershell
python rust/tools/parity/generate_vgg19_golden.py --source $vggSource
```

Inspect the zip manifest and sidecar. The zip contains only small input/output evidence, never model weights.

- [ ] **Step 6: Build a local package and run official parity**

Choose a new unique destination under the system temp directory:

```powershell
$vggPackage = Join-Path $env:TEMP ("feathertalk-vgg19-package-" + [guid]::NewGuid())
cargo run -p feathertalk-vgg19-package -- --source $vggSource --licenses tests\fixtures\vgg19\LICENSES.local-parity.json --destination $vggPackage
$env:FEATHERTALK_VGG19_PACKAGE = $vggPackage
cargo test -p feathertalk-training --test vgg19_official_parity -- --ignored --exact --nocapture
```

Run all three commands from `rust/`; the license argument above is relative to that directory.

Expected: package creation succeeds and parity reports both metrics below their thresholds. Do not commit the package or the 500+ MiB source file.

- [ ] **Step 7: Document offline weight handling**

Update `docs/WEIGHTS.md` with:

- the fixed official URL and weight id;
- the fact that VGG weights are not bundled;
- the exact `vgg19-package` command;
- the three-file runtime directory contract;
- no automatic downloads at runtime;
- local-parity license data is not release redistribution approval;
- release packaging must supply an independently reviewed `LICENSES.json` and record hashes.

- [ ] **Step 8: Run normal and official tests, then commit evidence**

```powershell
cargo test -p feathertalk-training
$env:FEATHERTALK_VGG19_PACKAGE = $vggPackage
cargo test -p feathertalk-training --test vgg19_official_parity -- --ignored --exact --nocapture
```

Then explicitly stage:

```powershell
git add rust/crates/feathertalk-training/Cargo.toml rust/tools/parity/generate_vgg19_golden.py rust/crates/feathertalk-training/tests/vgg19_official_parity.rs rust/tests/golden/vgg19-conv3-3-v1.sha256 rust/tests/fixtures/vgg19/LICENSES.local-parity.json docs/WEIGHTS.md
git add -f rust/tests/golden/vgg19-conv3-3-v1.zip
git commit -m "test: verify official VGG19 parity"
```

### Task 8: Run slice acceptance and prepare branch integration

**Files:**
- No intended production changes.

**Interfaces:**
- Proves the complete slice matches the design and does not regress existing models, imports, or protected data.

- [ ] **Step 1: Run all focused suites**

From `rust/`:

```powershell
cargo test -p feathertalk-training -- --nocapture
cargo test -p feathertalk-vgg19-package -- --nocapture
cargo test -p feathertalk-weights -- --nocapture
cargo test -p feathertalk-models --test train_step
cargo test -p feathertalk-models --test mobileone_unet
```

- [ ] **Step 2: Re-run explicit official parity**

```powershell
$env:FEATHERTALK_VGG19_PACKAGE = $vggPackage
cargo test -p feathertalk-training --test vgg19_official_parity -- --ignored --exact --nocapture
```

Record the printed max and mean errors in the handoff; both must meet the fixed thresholds.

- [ ] **Step 3: Run full fresh workspace verification**

```powershell
cargo test --workspace --all-targets
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
```

From repository worktree root:

```powershell
git diff --check
git status --short
git status --short -- demo/kanghui_training_video_featherhubert_188_latest
```

Expected: every command exits 0; CPU tests have 0 failures; only established environment-gated WGPU/media helpers and the explicit external VGG test are ignored in ordinary runs; protected demo status is empty inside the worktree.

- [ ] **Step 4: Perform inline plan/code review**

Compare the complete diff against every design section. In particular inspect:

- cutoff is pre-ReLU `features.14`;
- no normalization/channel swap;
- exact 14/24/1,735,488 importer contract;
- no automatic network/cache/random fallback;
- `no_grad` and target detach behavior;
- exact loss denominators and pair flattening;
- package failure atomicity and three-entry directory rule;
- license wording does not claim redistribution approval;
- no path references or changes under the protected demo.

Fix all Critical or Important findings with a new RED/GREEN test and a focused commit before proceeding.

- [ ] **Step 5: Finish the branch**

Use `superpowers:finishing-a-development-branch`. The standing user choice is local fast-forward merge to `main` after green verification. Re-run `cargo test --workspace --all-targets` on merged `main`, then remove only `.worktrees/training-losses` and delete only the merged `training-losses` branch.

## Plan Self-Review

- Spec coverage: crate boundaries, exact VGG cutoff, BGR/no-normalization compatibility, frozen gradients, model package schema, kind-scoped importer, atomic tool, three loss formulas, structured errors, official parity, license record, exclusions, and full verification map to Tasks 1-8.
- Placeholder scan: no `TBD`, `TODO`, “implement later”, unspecified error handling, or undefined neighboring interface remains.
- Type consistency: `Vgg19Conv3_3`, `TrainingError`, `LegacyModelKind::Vgg19Conv3_3`, `load_vgg19_package`, `PerceptualFeatureExtractor`, `perceptual_mse`, three config types, `LossBreakdown`, and all three loss function names are stable throughout.
- Scope check: this plan ends at feature/loss readiness. DataLoader, resumable randomness, optimizer/checkpoint state, metrics, and previews remain explicitly deferred to the next milestone-three plans.
