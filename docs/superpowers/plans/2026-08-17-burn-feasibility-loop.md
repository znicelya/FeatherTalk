# Burn Feasibility Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove that stable Burn/WGPU can load FeatherTalk PyTorch weights, reproduce FeatherHuBERT and Original UNet numerically, and execute one Adam training step without Python at runtime.

**Architecture:** Create a self-contained `rust/` workspace with three focused crates: model definitions, strict legacy-weight import, and parity tooling. Python is used once to generate immutable golden fixtures from the current implementation; all Rust tests and acceptance commands consume those fixtures directly and do not launch Python.

**Tech Stack:** Rust 1.92.0, edition 2024, Burn 0.21.0, burn-store 0.21.0, Burn NdArray CPU backend, Burn WGPU backend, Safetensors, NumPy fixture files, Clap, Serde, SHA-256.

## Global Constraints

- Follow [the approved design](../specs/2026-08-17-rust-desktop-migration-design.md).
- Keep every Rust source file, Cargo manifest, Rust-only tool, golden fixture, and feasibility report under the repository-level `rust/` directory.
- Run every Cargo, Python fixture-generation, and task-local `git add` command in this plan from `rust/` unless a step explicitly says otherwise.
- Pin `rust-toolchain.toml` to Rust `1.92.0`; Burn 0.21.0 requires Rust 1.92 or newer.
- Pin every Burn crate to exactly `=0.21.0`; do not use Burn 0.22 prereleases.
- Use `burn-store` for PyTorch and safetensors I/O; do not add Candle, LibTorch, `tch-rs`, or a second tensor runtime.
- Scope is FeatherHuBERT and Original UNet only. Do not add Wenet, original HuBERT, MobileOne, VGG19, GPUI, FFmpeg, preprocessing, or production training loops.
- Python may only generate committed golden fixtures. Rust tests and binaries must not spawn Python.
- Legacy import is strict: missing tensors, unexpected tensors, shape errors, unsupported dtype, and duplicate remapped keys fail the operation. PyTorch `num_batches_tracked` buffers are the only known ignored keys.
- CPU float32 forward tolerance is `max_abs_error <= 1e-4`.
- WGPU float32 forward tolerance is `max_abs_error <= 1e-3`.
- Loss and selected gradient/parameter tolerance is `relative_error <= 1e-3`.
- No silent CPU fallback is allowed in a command that requested WGPU.
- Keep the existing Python/C++ files unchanged during this milestone; they remain the migration oracle.
- Run `cargo fmt --check`, `cargo clippy --workspace --all-targets --all-features -- -D warnings`, and `cargo test --workspace --all-features` before declaring the milestone complete.

## Repository Map

Files created or modified by this milestone:

```text
rust/
  Cargo.toml
  Cargo.lock
  rust-toolchain.toml
  .gitignore

  crates/feathertalk-models/
    Cargo.toml
    src/lib.rs
    src/backend.rs
    src/feather_hubert/mod.rs
    src/feather_hubert/config.rs
    src/feather_hubert/frontend.rs
    src/feather_hubert/tcn.rs
    src/feather_hubert/model.rs
    src/unet/mod.rs
    src/unet/config.rs
    src/unet/blocks.rs
    src/unet/audio.rs
    src/unet/model.rs
    src/train_step.rs
    tests/backend_contract.rs
    tests/feather_hubert_shapes.rs
    tests/unet_shapes.rs
    tests/train_step.rs

  crates/feathertalk-weights/
    Cargo.toml
    src/lib.rs
    src/error.rs
    src/key_map.rs
    src/legacy.rs
    src/safe.rs
    tests/legacy_import.rs

  crates/feathertalk-parity/
    Cargo.toml
    src/lib.rs
    src/archive.rs
    src/fixture.rs
    src/metrics.rs
    src/probe.rs
    src/main.rs
    tests/archive_contract.rs
    tests/cli_contract.rs
    tests/cpu_parity.rs
    tests/wgpu_parity.rs

  tools/parity/generate_golden.py
  tests/golden/burn-feasibility-v1.zip
  tests/golden/burn-feasibility-v1.sha256
  docs/migration/burn-feasibility-report.md
```

All paths below are relative to `rust/`. The existing Python sources referenced by the golden generator remain relative to the repository root.

Crate ownership:

- `feathertalk-models`: Burn modules and a single-step training primitive; no file-format knowledge.
- `feathertalk-weights`: strict PyTorch-to-Burn loading and safetensors writing; generic over Burn modules.
- `feathertalk-parity`: fixture loading, metrics, backend probes, acceptance CLI, and cross-crate tests.

---

### Task 1: Bootstrap the Rust workspace and backend contract

**Files:**

- Create: `Cargo.toml`
- Create: `Cargo.lock`
- Create: `rust-toolchain.toml`
- Create: `.gitignore`
- Create: `crates/feathertalk-models/Cargo.toml`
- Create: `crates/feathertalk-models/src/lib.rs`
- Create: `crates/feathertalk-models/src/backend.rs`
- Create: `crates/feathertalk-models/tests/backend_contract.rs`
- Create: `crates/feathertalk-weights/Cargo.toml`
- Create: `crates/feathertalk-weights/src/lib.rs`
- Create: `crates/feathertalk-parity/Cargo.toml`
- Create: `crates/feathertalk-parity/src/lib.rs`

**Interfaces:**

- Produces: `CpuBackend`, `CpuAutodiffBackend`, `GpuBackend`, and `GpuAutodiffBackend` aliases.
- Produces: workspace feature flags `wgpu`, `metal`, and `vulkan` for later tasks.

- [ ] **Step 1: Write the backend compile-contract test**

From the repository root, create and enter the dedicated workspace before creating the test:

```powershell
New-Item -ItemType Directory -Force rust | Out-Null
Set-Location rust
```

Create `crates/feathertalk-models/tests/backend_contract.rs`:

```rust
use burn::tensor::{Tensor, backend::Backend};
use feathertalk_models::backend::{CpuAutodiffBackend, CpuBackend};

fn assert_backend<B: Backend>() {}

#[test]
fn cpu_backend_aliases_compile_and_execute() {
    assert_backend::<CpuBackend>();
    assert_backend::<CpuAutodiffBackend>();

    let device = Default::default();
    let tensor = Tensor::<CpuBackend, 2>::ones([2, 3], &device);
    assert_eq!(tensor.dims(), [2, 3]);
}
```

- [ ] **Step 2: Run the test and verify the workspace does not exist yet**

Run:

```powershell
cargo test -p feathertalk-models --test backend_contract
```

Expected: FAIL because `rust/Cargo.toml` and the crate do not exist.

- [ ] **Step 3: Create the pinned workspace manifest**

Create `rust/Cargo.toml` with this dependency policy:

```toml
[workspace]
resolver = "2"
members = [
  "crates/feathertalk-models",
  "crates/feathertalk-weights",
  "crates/feathertalk-parity",
]

[workspace.package]
version = "0.1.0"
edition = "2024"
rust-version = "1.92"
license = "Apache-2.0"

[workspace.dependencies]
burn = { version = "=0.21.0", default-features = false, features = ["std", "autodiff", "ndarray", "wgpu", "store"] }
burn-store = { version = "=0.21.0", default-features = false, features = ["std", "pytorch", "safetensors"] }
clap = { version = "4.5", features = ["derive"] }
hex = "0.4"
ndarray = "0.17.1"
ndarray-npy = "0.10.0"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
sha2 = "0.11.0"
tempfile = "3.20"
thiserror = "2.0"
zip = { version = "6", default-features = false, features = ["deflate"] }
```

Create `rust-toolchain.toml`:

```toml
[toolchain]
channel = "1.92.0"
components = ["clippy", "rustfmt"]
profile = "minimal"
```

Create `rust/.gitignore`:

```gitignore

# Rust
/target/
/.cache/parity/
```

- [ ] **Step 4: Create the three crate manifests**

`crates/feathertalk-models/Cargo.toml`:

```toml
[package]
name = "feathertalk-models"
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
serde.workspace = true
thiserror.workspace = true
```

`crates/feathertalk-weights/Cargo.toml`:

```toml
[package]
name = "feathertalk-weights"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[dependencies]
burn.workspace = true
burn-store.workspace = true
hex.workspace = true
serde.workspace = true
sha2.workspace = true
thiserror.workspace = true
```

`crates/feathertalk-parity/Cargo.toml`:

```toml
[package]
name = "feathertalk-parity"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[features]
default = ["wgpu"]
wgpu = ["feathertalk-models/wgpu"]
metal = ["feathertalk-models/metal"]
vulkan = ["feathertalk-models/vulkan"]

[dependencies]
burn.workspace = true
clap.workspace = true
feathertalk-models = { path = "../feathertalk-models" }
feathertalk-weights = { path = "../feathertalk-weights" }
hex.workspace = true
ndarray.workspace = true
ndarray-npy.workspace = true
serde.workspace = true
serde_json.workspace = true
sha2.workspace = true
tempfile.workspace = true
thiserror.workspace = true
zip.workspace = true
```

- [ ] **Step 5: Implement the backend aliases**

Create `crates/feathertalk-models/src/backend.rs`:

```rust
use burn::backend::{Autodiff, NdArray, Wgpu};

pub type CpuBackend = NdArray<f32>;
pub type CpuAutodiffBackend = Autodiff<CpuBackend>;
pub type GpuBackend = Wgpu<f32, i32, u32>;
pub type GpuAutodiffBackend = Autodiff<GpuBackend>;
```

Create `crates/feathertalk-models/src/lib.rs`:

```rust
pub mod backend;
pub mod feather_hubert;
pub mod train_step;
pub mod unet;
```

Create empty module roots so the crate compiles:

```rust
// crates/feathertalk-models/src/feather_hubert/mod.rs
```

```rust
// crates/feathertalk-models/src/unet/mod.rs
```

```rust
// crates/feathertalk-models/src/train_step.rs
```

Create `crates/feathertalk-weights/src/lib.rs` and `crates/feathertalk-parity/src/lib.rs` as empty library roots.

- [ ] **Step 6: Run the workspace checks**

Run:

```powershell
cargo fmt --all
cargo test -p feathertalk-models --test backend_contract
cargo check --workspace --all-targets
```

Expected: PASS and `Cargo.lock` is generated with Burn 0.21.0.

- [ ] **Step 7: Commit the workspace foundation**

```powershell
git add Cargo.toml Cargo.lock rust-toolchain.toml .gitignore crates
git commit -m "build: bootstrap Burn feasibility workspace"
```

---

### Task 2: Freeze deterministic Python golden fixtures

**Files:**

- Create: `tools/parity/generate_golden.py`
- Create: `tests/golden/burn-feasibility-v1.zip`
- Create: `tests/golden/burn-feasibility-v1.sha256`
- Create: `crates/feathertalk-parity/src/archive.rs`
- Create: `crates/feathertalk-parity/src/fixture.rs`
- Modify: `crates/feathertalk-parity/src/lib.rs`
- Create: `crates/feathertalk-parity/tests/archive_contract.rs`

**Interfaces:**

- Produces: `GoldenArchive::open(path)` and `GoldenFixture::load(name)`.
- Produces immutable fixture IDs `feather_micro_eval`, `unet_production_eval`, and `unet_micro_train_step`.
- Consumes current Python `FeatherHuBERTEncoder` and `unet.Model` only while generating the archive.

- [ ] **Step 1: Write the archive contract test**

Create `crates/feathertalk-parity/tests/archive_contract.rs`:

```rust
use feathertalk_parity::archive::GoldenArchive;

#[test]
fn golden_archive_has_required_entries_and_valid_hash() {
    let root = env!("CARGO_MANIFEST_DIR");
    let archive = GoldenArchive::open(format!(
        "{root}/../../tests/golden/burn-feasibility-v1.zip"
    ))
    .expect("golden archive should open");

    archive.verify_sidecar_sha256().expect("archive hash");
    for entry in [
        "manifest.json",
        "weights/tiny_direct.pth",
        "weights/tiny_nested.pth",
        "weights/tiny_missing.pth",
        "weights/tiny_unexpected.pth",
        "weights/feather_micro.pth",
        "weights/unet_production.pth",
        "weights/unet_micro_train.pth",
        "arrays/feather_input.npy",
        "arrays/feather_output.npy",
        "arrays/unet_image.npy",
        "arrays/unet_audio.npy",
        "arrays/unet_output.npy",
        "arrays/train_target.npy",
        "arrays/train_expected.json",
    ] {
        assert!(archive.contains(entry), "missing {entry}");
    }
}
```

- [ ] **Step 2: Run the test and verify the fixture is absent**

Run:

```powershell
cargo test -p feathertalk-parity --test archive_contract
```

Expected: FAIL because `GoldenArchive` and the fixture archive do not exist.

- [ ] **Step 3: Implement deterministic parameter filling in the generator**

Create `tools/parity/generate_golden.py`. Resolve the existing Python oracle from the repository root without copying it into `rust/`:

```python
RUST_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = RUST_ROOT.parent
sys.path.insert(0, str(REPOSITORY_ROOT))
```

Use the following exact rules for every state-dict tensor:

```python
def deterministic_values(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_floating_point():
        return torch.zeros_like(tensor)
    offset = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:2], "little") % 7
    flat = torch.arange(tensor.numel(), dtype=torch.float32)
    if name.endswith("running_var"):
        values = 0.75 + (flat.remainder(17) / 64.0)
    elif name.endswith("running_mean"):
        values = (flat.remainder(11) - 5.0) / 512.0
    else:
        values = ((flat + offset).remainder(17) - 8.0) / 512.0
    return values.reshape(tensor.shape).to(dtype=tensor.dtype)


def fill_state_dict(module: torch.nn.Module) -> None:
    state = module.state_dict()
    for name, tensor in state.items():
        tensor.copy_(deterministic_values(name, tensor))
    module.load_state_dict(state)
```

The generator must call `torch.manual_seed(20260817)`, set every inference model to `eval()`, and set FeatherHuBERT dropout to `0.0`.

- [ ] **Step 4: Generate the three fixture cases**

The script must create these cases:

```text
feather_micro_eval:
  config = channels=32, expansion=2, num_blocks=2, output_dim=64, dropout=0.0
  waveform = linspace(-0.75, 0.75, 1360).reshape(1, 1360)

unet_production_eval:
  channels = [32, 64, 128, 256, 512]
  image = repeating float32 pattern shaped [1, 6, 160, 160]
  audio = repeating float32 pattern shaped [1, 16, 32, 32]

unet_micro_train_step:
  channels = [2, 4, 8, 16, 32]
  batch = 1
  image/audio/target use deterministic repeating float32 patterns
  optimizer = Adam(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
  loss = mean(abs(prediction - target))
```

It must also create four small 2x2 linear checkpoints used by strict import tests:

```python
tiny_state = {
    "weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
    "bias": torch.tensor([0.25, -0.5], dtype=torch.float32),
}
torch.save(tiny_state, work / "weights/tiny_direct.pth")
torch.save({"model": tiny_state}, work / "weights/tiny_nested.pth")
torch.save({"model": {"weight": tiny_state["weight"]}}, work / "weights/tiny_missing.pth")
torch.save(
    {"model": {**tiny_state, "unexpected": torch.ones(1)}},
    work / "weights/tiny_unexpected.pth",
)
```

For the micro UNet, define the same Python blocks as `unet.py` but accept the channel array in the constructor. Do not alter production Python source files.

Save checkpoints with top-level keys `epoch`, `model`, and `config`. Save input/output arrays as float32 NPY. Save initial loss, post-step loss, and these selected post-step tensors in `arrays/train_expected.json` plus NPY payloads:

```text
inc.inconv.conv.0.weight
audio_model.conv1.conv.0.weight
outc.conv.weight
```

- [ ] **Step 5: Package and size-check the fixture archive**

Use `zipfile.ZipFile(..., compression=zipfile.ZIP_DEFLATED, compresslevel=9)`. The generator must fail if the final archive exceeds 20 MiB:

```python
archive_size = output_path.stat().st_size
if archive_size > 20 * 1024 * 1024:
    raise RuntimeError(f"golden archive is too large: {archive_size} bytes")
```

Write lowercase SHA-256 plus newline to `tests/golden/burn-feasibility-v1.sha256`.

- [ ] **Step 6: Generate the archive once**

Run from the existing Python environment:

```powershell
python tools/parity/generate_golden.py
```

Expected: creates both golden files, reports all fixture shapes, and exits zero. This is the only step in the milestone that requires Python.

- [ ] **Step 7: Implement the Rust archive and fixture readers**

Create these public types:

```rust
pub struct GoldenArchive {
    path: std::path::PathBuf,
    entries: std::collections::BTreeSet<String>,
}

pub struct GoldenFixture {
    pub id: String,
    pub inputs: std::collections::BTreeMap<String, ndarray::ArrayD<f32>>,
    pub expected: std::collections::BTreeMap<String, ndarray::ArrayD<f32>>,
}

impl GoldenArchive {
    pub fn open(path: impl Into<std::path::PathBuf>) -> Result<Self, FixtureError>;
    pub fn contains(&self, entry: &str) -> bool;
    pub fn verify_sidecar_sha256(&self) -> Result<(), FixtureError>;
    pub fn extract_to(&self, directory: &std::path::Path) -> Result<(), FixtureError>;
    pub fn load_fixture(&self, id: &str) -> Result<GoldenFixture, FixtureError>;
}
```

ZIP extraction must reject absolute paths, parent traversal, entries over 512 MiB, and total expanded size over 2 GiB.

- [ ] **Step 8: Run archive tests and commit**

Run:

```powershell
cargo fmt --all
cargo test -p feathertalk-parity --test archive_contract
```

Expected: PASS without invoking Python.

Commit:

```powershell
git add tools/parity tests/golden crates/feathertalk-parity
git commit -m "test: freeze Burn parity fixtures"
```

---

### Task 3: Implement strict PyTorch checkpoint import

**Files:**

- Modify: `crates/feathertalk-weights/src/lib.rs`
- Create: `crates/feathertalk-weights/src/error.rs`
- Create: `crates/feathertalk-weights/src/key_map.rs`
- Create: `crates/feathertalk-weights/src/legacy.rs`
- Create: `crates/feathertalk-weights/src/safe.rs`
- Create: `crates/feathertalk-weights/tests/legacy_import.rs`

**Interfaces:**

- Produces: `LegacyImportRequest`, `LegacyModelKind`, `ImportReport`, and `import_into`.
- Produces: `save_safetensors` for any Burn module implementing `ModuleSnapshot`.
- Consumes: Burn `PytorchStore`, `SafetensorsStore`, and model-specific key remapping rules.

- [ ] **Step 1: Write strict import tests**

Define a local two-input/two-output Burn `Linear` fixture in the test file. Tests must cover:

```rust
#[test]
fn nested_model_checkpoint_loads_all_expected_tensors() {
    let fixture = extract_fixture("weights/tiny_nested.pth");
    let device = Default::default();
    let mut model = burn::nn::LinearConfig::new(2, 2).init::<CpuBackend>(&device);
    let request = request_for(fixture, Some("model"));
    let report = import_into::<CpuBackend, _>(&mut model, &request).unwrap();
    assert_eq!(report.applied.len(), 2);
    assert!(report.ignored.is_empty());
}

#[test]
fn direct_state_dict_is_detected() {
    let fixture = extract_fixture("weights/tiny_direct.pth");
    let device = Default::default();
    let mut model = burn::nn::LinearConfig::new(2, 2).init::<CpuBackend>(&device);
    let report = import_into::<CpuBackend, _>(&mut model, &request_for(fixture, None)).unwrap();
    assert_eq!(report.applied.len(), 2);
}

#[test]
fn missing_tensor_is_rejected() {
    let fixture = extract_fixture("weights/tiny_missing.pth");
    let device = Default::default();
    let mut model = burn::nn::LinearConfig::new(2, 2).init::<CpuBackend>(&device);
    let error = import_into::<CpuBackend, _>(&mut model, &request_for(fixture, Some("model")))
        .unwrap_err();
    assert!(matches!(error, WeightImportError::MissingTensor(_)));
}

#[test]
fn unexpected_tensor_is_rejected() {
    let fixture = extract_fixture("weights/tiny_unexpected.pth");
    let device = Default::default();
    let mut model = burn::nn::LinearConfig::new(2, 2).init::<CpuBackend>(&device);
    let error = import_into::<CpuBackend, _>(&mut model, &request_for(fixture, Some("model")))
        .unwrap_err();
    assert!(matches!(error, WeightImportError::UnexpectedTensor(_)));
}

#[test]
fn num_batches_tracked_is_ignored_as_a_known_buffer() {
    assert!(is_known_ignored_key("encoder.0.bn.num_batches_tracked"));
    assert!(!is_known_ignored_key("encoder.0.bn.running_mean"));
}

#[test]
fn imported_module_round_trips_through_safetensors() {
    let fixture = extract_fixture("weights/tiny_nested.pth");
    let device = Default::default();
    let mut first = burn::nn::LinearConfig::new(2, 2).init::<CpuBackend>(&device);
    import_into::<CpuBackend, _>(&mut first, &request_for(fixture, Some("model"))).unwrap();

    let temp = tempfile::tempdir().unwrap();
    let safe = temp.path().join("tiny.safetensors");
    save_safetensors::<CpuBackend, _>(&first, &safe).unwrap();
    let second = load_linear_safetensors::<CpuBackend>(&safe, &device).unwrap();
    assert_module_snapshots_equal(&first, &second);
}
```

Implement `extract_fixture`, `request_for`, `load_linear_safetensors`, and `assert_module_snapshots_equal` inside `legacy_import.rs`. Add `tempfile` and `zip` as dev-dependencies of `feathertalk-weights`; do not add a dependency on `feathertalk-parity`, because that would create a crate cycle.

- [ ] **Step 2: Run tests and verify import APIs are missing**

```powershell
cargo test -p feathertalk-weights --test legacy_import
```

Expected: FAIL because the importer types do not exist.

- [ ] **Step 3: Define strict request and report types**

Create:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegacyModelKind {
    FeatherHubert,
    OriginalUnet,
}

#[derive(Debug, Clone)]
pub struct LegacyImportRequest {
    pub path: std::path::PathBuf,
    pub kind: LegacyModelKind,
    pub top_level_key: Option<String>,
    pub max_file_bytes: u64,
    pub max_tensor_count: usize,
    pub max_total_elements: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ImportReport {
    pub source_sha256: String,
    pub applied: Vec<String>,
    pub ignored: Vec<String>,
    pub tensor_count: usize,
    pub total_elements: u64,
}
```

Default limits are 4 GiB source size, 10,000 tensors, and 2,000,000,000 elements.

- [ ] **Step 4: Implement model-specific key remapping**

FeatherHuBERT keeps these prefixes unchanged:

```text
frontend.layers.<n>.conv
frontend.layers.<n>.norm
encoder.<n>.norm
encoder.<n>.pw_expand
encoder.<n>.dw_conv
encoder.<n>.pw_project
final_norm
proj
```

Original UNet uses these regex remaps before Burn's `PyTorchToBurnAdapter`:

```text
\.double_conv\.0\. -> .double_conv.first.
\.double_conv\.1\. -> .double_conv.second.
\.conv\.0\.        -> .expand_conv.
\.conv\.1\.        -> .expand_bn.
\.conv\.3\.        -> .depthwise_conv.
\.conv\.4\.        -> .depthwise_bn.
\.conv\.6\.        -> .project_conv.
\.conv\.7\.        -> .project_bn.
^fuse_conv\.0\.     -> fuse_first.
^fuse_conv\.1\.     -> fuse_second.
```

Keep `map_indices_contiguous(false)` so index changes only occur through reviewed remaps.

- [ ] **Step 5: Implement bounded checkpoint inspection**

Before applying tensors:

1. Check source file length.
2. Open with `PytorchStore::from_file`.
3. Try the requested top-level key; when absent, try `model`, `state_dict`, then direct state dict in that order.
4. Read `keys()` and snapshots.
5. Exclude only keys ending in `.num_batches_tracked`.
6. Check tensor count and total element count.
7. Reject duplicate keys after remapping.

Return typed `WeightImportError` variants for I/O, unsafe limits, unsupported structure, missing tensor, unexpected tensor, shape mismatch, dtype mismatch, duplicate key, and store errors.

- [ ] **Step 6: Apply tensors and reject partial results**

Expose this generic function:

```rust
pub fn import_into<B, M>(
    module: &mut M,
    request: &LegacyImportRequest,
) -> Result<ImportReport, WeightImportError>
where
    B: burn::tensor::backend::Backend,
    M: burn_store::ModuleSnapshot<B>,
{
    let mut store = build_strict_store(request)?;
    let result = module.load_from(&mut store)?;
    validate_apply_result(&result)?;
    build_report(request, &mut store, result)
}
```

`validate_apply_result` must require no missing tensors, no errors, and no unused tensors except the known ignored BatchNorm counter.

- [ ] **Step 7: Implement safetensors output and round-trip validation**

Expose:

```rust
pub fn save_safetensors<B, M>(
    module: &M,
    path: impl Into<std::path::PathBuf>,
) -> Result<(), WeightImportError>
where
    B: burn::tensor::backend::Backend,
    M: burn_store::ModuleSnapshot<B>,
{
    let path = path.into();
    let mut store = burn_store::SafetensorsStore::from_file(&path).overwrite(true);
    module.save_into(&mut store)?;
    Ok(())
}
```

The test must load the saved file into a freshly initialized module and compare every tensor name, shape, dtype, and value.

- [ ] **Step 8: Run strict import tests and commit**

```powershell
cargo fmt --all
cargo test -p feathertalk-weights --test legacy_import
cargo clippy -p feathertalk-weights --all-targets -- -D warnings
git add crates/feathertalk-weights
git commit -m "feat: import PyTorch checkpoints with burn-store"
```

---

### Task 4: Port FeatherHuBERT to Burn

**Files:**

- Create: `crates/feathertalk-models/src/feather_hubert/config.rs`
- Create: `crates/feathertalk-models/src/feather_hubert/frontend.rs`
- Create: `crates/feathertalk-models/src/feather_hubert/tcn.rs`
- Create: `crates/feathertalk-models/src/feather_hubert/model.rs`
- Modify: `crates/feathertalk-models/src/feather_hubert/mod.rs`
- Create: `crates/feathertalk-models/tests/feather_hubert_shapes.rs`

**Interfaces:**

- Produces: `FeatherHubertConfig`, `FeatherHubertEncoder<B>`, `expected_hubert_frames`, `normalize_waveform`, and `make_even_tokens`.
- Consumes: `[batch, samples]` float32 waveform.
- Produces: `[batch, tokens, output_dim]` float32 hidden features.

- [ ] **Step 1: Write frame-count, normalization, and shape tests**

Required assertions:

```rust
use burn::tensor::{Tensor, TensorData};
use feathertalk_models::{
    backend::CpuBackend,
    feather_hubert::{
        FeatherHubertConfig, expected_hubert_frames, make_even_tokens, normalize_waveform,
    },
};

#[test]
fn hubert_frame_count_matches_python_contract() {
    assert_eq!(expected_hubert_frames(399), 0);
    assert_eq!(expected_hubert_frames(400), 1);
    assert_eq!(expected_hubert_frames(720), 2);
    assert_eq!(expected_hubert_frames(1360), 4);
}

#[test]
fn waveform_normalization_has_zero_mean_and_unit_variance() {
    let device = Default::default();
    let waveform = Tensor::<CpuBackend, 2>::from_data(
        TensorData::from([[1.0_f32, 2.0, 3.0, 4.0]]),
        &device,
    );
    let values = normalize_waveform(waveform)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f32>()
        / values.len() as f32;

    assert!(mean.abs() <= 1e-6, "mean={mean}");
    assert!((variance - 1.0).abs() <= 1e-5, "variance={variance}");
}

#[test]
fn micro_encoder_returns_four_tokens() {
    let device = Default::default();
    let model = FeatherHubertConfig::parity_micro().init::<CpuBackend>(&device);
    let waveform = Tensor::<CpuBackend, 2>::zeros([1, 1360], &device);
    assert_eq!(model.forward(waveform).dims(), [1, 4, 64]);
}

#[test]
fn production_encoder_returns_1024_features() {
    let device = Default::default();
    let model = FeatherHubertConfig::default().init::<CpuBackend>(&device);
    let waveform = Tensor::<CpuBackend, 2>::zeros([1, 1360], &device);
    assert_eq!(model.forward(waveform).dims(), [1, 4, 1024]);
}

#[test]
fn odd_token_count_drops_the_last_token() {
    let device = Default::default();
    let tokens = Tensor::<CpuBackend, 3>::zeros([1, 5, 64], &device);
    assert_eq!(make_even_tokens(tokens).dims(), [1, 4, 64]);
}
```

- [ ] **Step 2: Run tests and verify model APIs are missing**

```powershell
cargo test -p feathertalk-models --test feather_hubert_shapes
```

Expected: FAIL because FeatherHuBERT types do not exist.

- [ ] **Step 3: Implement configuration and pure helpers**

Create:

```rust
pub const SAMPLE_RATE: usize = 16_000;
pub const HUBERT_KERNEL: usize = 400;
pub const HUBERT_STRIDE: usize = 320;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FeatherHubertConfig {
    pub channels: usize,
    pub expansion: usize,
    pub num_blocks: usize,
    pub output_dim: usize,
    pub dropout: f64,
}

impl Default for FeatherHubertConfig {
    fn default() -> Self {
        Self {
            channels: 512,
            expansion: 2,
            num_blocks: 12,
            output_dim: 1024,
            dropout: 0.05,
        }
    }
}

impl FeatherHubertConfig {
    pub const fn parity_micro() -> Self {
        Self {
            channels: 32,
            expansion: 2,
            num_blocks: 2,
            output_dim: 64,
            dropout: 0.0,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> FeatherHubertEncoder<B>;
}

pub fn expected_hubert_frames(samples: usize) -> usize {
    if samples < HUBERT_KERNEL {
        0
    } else {
        (samples - (HUBERT_KERNEL - HUBERT_STRIDE)) / HUBERT_STRIDE
    }
}

pub fn normalize_waveform<B: Backend>(waveform: Tensor<B, 2>) -> Tensor<B, 2>;

pub fn make_even_tokens<B: Backend>(tokens: Tensor<B, 3>) -> Tensor<B, 3>;
```

`normalize_waveform` must compute population variance with epsilon `1e-7`, matching NumPy `speech.var()`.

- [ ] **Step 4: Implement `ConvNormAct1d` and the valid-convolution frontend**

Use Burn `Conv1dConfig`, `PaddingConfig1d::Valid`, `GroupNormConfig`, and exact GELU. Build seven layers with:

```rust
const KERNELS: [usize; 7] = [10, 3, 3, 3, 3, 2, 2];
const STRIDES: [usize; 7] = [5, 2, 2, 2, 2, 2, 2];
```

Channel sequence is `[64, 128, 256, 384, config.channels, config.channels, config.channels]`. Every convolution has `bias=false`. Group count selects the first divisor from `[32, 16, 8, 4, 2]`, else 1.

- [ ] **Step 5: Implement `DepthwiseTcnBlock`**

The forward order is exact:

```rust
pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
    let residual = input.clone();
    let x = self.norm.forward(input);
    let x = self.pw_expand.forward(x);
    let x = burn::tensor::activation::gelu(self.dw_conv.forward(x));
    let x = self.pw_project.forward(x);
    residual + self.dropout.forward(x)
}
```

Use kernel 5, expansion from config, depthwise groups equal to expanded channels, symmetric padding `2 * dilation`, no convolution bias, and dilation cycle `[1, 2, 4, 8]`.

- [ ] **Step 6: Implement `FeatherHubertEncoder`**

The module fields are named to match Python checkpoint prefixes:

```rust
#[derive(burn::module::Module, Debug)]
pub struct FeatherHubertEncoder<B: Backend> {
    pub frontend: HubertStrideFrontend<B>,
    pub encoder: Vec<DepthwiseTcnBlock<B>>,
    pub final_norm: GroupNorm<B>,
    pub proj: Conv1d<B>,
    #[module(skip)]
    pub config: FeatherHubertConfig,
}
```

Forward accepts rank-2 waveform, inserts the channel dimension, executes frontend and TCN, applies final norm, GELU, 1x1 projection, swaps channel/time axes, and crops or zero-pads the token axis to `expected_hubert_frames`.

- [ ] **Step 7: Run shape tests on CPU**

```powershell
cargo fmt --all
cargo test -p feathertalk-models --test feather_hubert_shapes
```

Expected: PASS for micro and production configurations.

- [ ] **Step 8: Commit FeatherHuBERT**

```powershell
git add crates/feathertalk-models/src/feather_hubert crates/feathertalk-models/tests/feather_hubert_shapes.rs
git commit -m "feat: port FeatherHuBERT model to Burn"
```

---

### Task 5: Port Original UNet to Burn

**Files:**

- Create: `crates/feathertalk-models/src/unet/config.rs`
- Create: `crates/feathertalk-models/src/unet/blocks.rs`
- Create: `crates/feathertalk-models/src/unet/audio.rs`
- Create: `crates/feathertalk-models/src/unet/model.rs`
- Modify: `crates/feathertalk-models/src/unet/mod.rs`
- Create: `crates/feathertalk-models/tests/unet_shapes.rs`

**Interfaces:**

- Produces: `OriginalUnetConfig`, `OriginalUnet<B>`, `InvertedResidualConfig`, `DownConfig`, and `AudioConvHubertConfig`.
- Consumes: image `[batch, 6, 160, 160]` and audio `[batch, 16, 32, 32]`.
- Produces: sigmoid image `[batch, 3, 160, 160]`.

- [ ] **Step 1: Write block and full-model shape tests**

Tests must assert:

```rust
use burn::tensor::Tensor;
use feathertalk_models::{
    backend::CpuBackend,
    unet::{AudioConvHubertConfig, DownConfig, InvertedResidualConfig, OriginalUnetConfig},
};

#[test]
fn inverted_residual_preserves_shape_when_residual_is_enabled() {
    let device = Default::default();
    let block = InvertedResidualConfig::new(8, 8)
        .with_expansion(2)
        .with_stride(1)
        .init::<CpuBackend>(&device);
    let input = Tensor::<CpuBackend, 4>::ones([1, 8, 20, 20], &device);
    assert_eq!(block.forward(input).dims(), [1, 8, 20, 20]);
}

#[test]
fn down_blocks_produce_80_40_20_10_spatial_sizes() {
    let device = Default::default();
    let input = Tensor::<CpuBackend, 4>::ones([1, 32, 160, 160], &device);
    let down1 = DownConfig::new(32, 64).init::<CpuBackend>(&device);
    let down2 = DownConfig::new(64, 128).init::<CpuBackend>(&device);
    let down3 = DownConfig::new(128, 256).init::<CpuBackend>(&device);
    let down4 = DownConfig::new(256, 512).init::<CpuBackend>(&device);

    let x1 = down1.forward(input);
    assert_eq!(x1.dims(), [1, 64, 80, 80]);
    let x2 = down2.forward(x1);
    assert_eq!(x2.dims(), [1, 128, 40, 40]);
    let x3 = down3.forward(x2);
    assert_eq!(x3.dims(), [1, 256, 20, 20]);
    assert_eq!(down4.forward(x3).dims(), [1, 512, 10, 10]);
}

#[test]
fn hubert_audio_branch_matches_image_bottleneck_shape() {
    let device = Default::default();
    let branch = AudioConvHubertConfig::new([32, 64, 128, 256, 512])
        .init::<CpuBackend>(&device);
    let audio = Tensor::<CpuBackend, 4>::ones([1, 16, 32, 32], &device);
    assert_eq!(branch.forward(audio).dims(), [1, 512, 10, 10]);
}

#[test]
fn production_unet_returns_three_by_160_by_160() {
    let device = Default::default();
    let model = OriginalUnetConfig::production().init::<CpuBackend>(&device);
    let image = Tensor::<CpuBackend, 4>::zeros([1, 6, 160, 160], &device);
    let audio = Tensor::<CpuBackend, 4>::zeros([1, 16, 32, 32], &device);
    assert_eq!(model.forward(image, audio).dims(), [1, 3, 160, 160]);
}

#[test]
fn output_is_bounded_by_sigmoid() {
    let device = Default::default();
    let model = OriginalUnetConfig::parity_micro().init::<CpuBackend>(&device);
    let image = Tensor::<CpuBackend, 4>::ones([1, 6, 160, 160], &device);
    let audio = Tensor::<CpuBackend, 4>::ones([1, 16, 32, 32], &device);
    let values = model
        .forward(image, audio)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert!(values.iter().all(|value| value.is_finite()));
    assert!(values.iter().all(|value| (0.0..=1.0).contains(value)));
}
```

- [ ] **Step 2: Run tests and verify UNet APIs are missing**

```powershell
cargo test -p feathertalk-models --test unet_shapes
```

Expected: FAIL because UNet types do not exist.

- [ ] **Step 3: Implement production and micro configurations**

```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OriginalUnetConfig {
    pub channels: [usize; 5],
}

impl OriginalUnetConfig {
    pub const fn production() -> Self {
        Self { channels: [32, 64, 128, 256, 512] }
    }

    pub const fn parity_micro() -> Self {
        Self { channels: [2, 4, 8, 16, 32] }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> OriginalUnet<B>;
}
```

The micro configuration is test-only by convention; production callers must use `production()`. `InvertedResidualConfig`, `DownConfig`, and `AudioConvHubertConfig` must expose the constructors used by the tests above and initialize only the named Burn modules described in Steps 4-6.

- [ ] **Step 4: Implement `InvertedResidual`**

Use explicit named fields so key remapping is reviewable:

```rust
#[derive(burn::module::Module, Debug)]
pub struct InvertedResidual<B: Backend> {
    pub expand_conv: Conv2d<B>,
    pub expand_bn: BatchNorm<B, 2>,
    pub depthwise_conv: Conv2d<B>,
    pub depthwise_bn: BatchNorm<B, 2>,
    pub project_conv: Conv2d<B>,
    pub project_bn: BatchNorm<B, 2>,
    #[module(skip)]
    pub use_residual: bool,
}
```

Forward order is Conv1x1, BatchNorm, ReLU, depthwise Conv3x3, BatchNorm, ReLU, project Conv1x1, BatchNorm, optional residual add. All convolutions have `bias=false`; depthwise groups equal hidden channels.

- [ ] **Step 5: Implement double-conv, input, down, and up blocks**

`DoubleConvDw` fields must be named `first` and `second`. `Up` must use Burn `Interpolate2dConfig` with linear mode, scale factor `[2.0, 2.0]`, and `align_corners=true`.

Before concatenation, compare both spatial dimensions. If they differ, zero-pad symmetrically using the same left/right and top/bottom split as Python `F.pad`; reject a negative difference.

- [ ] **Step 6: Implement the FeatherHuBERT audio branch**

Do not implement the Wenet branch. Preserve the Python Hubert branch:

```text
16 -> channels[1] inverted residual, stride 1
channels[1] -> channels[2] inverted residual, stride 1
channels[2] -> channels[3] Conv3x3, stride 2, padding 1 + BN + ReLU
channels[3] -> channels[3] residual inverted block
channels[3] -> channels[4] Conv3x3, stride 2, padding 3 + BN + ReLU
two channels[4] residual inverted blocks
```

- [ ] **Step 7: Implement the full UNet**

Use field names matching model concepts:

```rust
#[derive(burn::module::Module, Debug)]
pub struct OriginalUnet<B: Backend> {
    pub audio_model: AudioConvHubert<B>,
    pub fuse_first: DoubleConvDw<B>,
    pub fuse_second: DoubleConvDw<B>,
    pub inc: InConvDw<B>,
    pub down1: Down<B>,
    pub down2: Down<B>,
    pub down3: Down<B>,
    pub down4: Down<B>,
    pub up1: Up<B>,
    pub up2: Up<B>,
    pub up3: Up<B>,
    pub up4: Up<B>,
    pub outc: OutConv<B>,
}
```

Forward must reproduce the Python skip connections, concatenate image and audio bottlenecks on channel axis 1, and apply sigmoid after the final 1x1 convolution.

- [ ] **Step 8: Run CPU shape tests and commit**

```powershell
cargo fmt --all
cargo test -p feathertalk-models --test unet_shapes
cargo clippy -p feathertalk-models --all-targets -- -D warnings
git add crates/feathertalk-models/src/unet crates/feathertalk-models/tests/unet_shapes.rs
git commit -m "feat: port Original UNet model to Burn"
```

---

### Task 6: Establish CPU numerical parity

**Files:**

- Create: `crates/feathertalk-parity/src/metrics.rs`
- Modify: `crates/feathertalk-parity/src/fixture.rs`
- Modify: `crates/feathertalk-parity/src/lib.rs`
- Create: `crates/feathertalk-parity/tests/cpu_parity.rs`
- Modify: `crates/feathertalk-weights/src/key_map.rs`

**Interfaces:**

- Produces: `ParityMetrics { max_abs, mean_abs, max_relative }`, `compare_f32`, `ForwardCase`, and `run_cpu_forward`.
- Consumes: golden PTH weights and NPY tensors.
- Proves: FeatherHuBERT and production Original UNet CPU forward parity.

- [ ] **Step 1: Write metric unit tests**

```rust
use feathertalk_parity::metrics::{ParityError, compare_f32};
use ndarray::array;

#[test]
fn exact_arrays_have_zero_error() {
    let values = array![[1.0_f32, -2.0], [3.0, 4.0]].into_dyn();
    let metrics = compare_f32(values.view(), values.view()).unwrap();
    assert_eq!(metrics.max_abs, 0.0);
    assert_eq!(metrics.mean_abs, 0.0);
    assert_eq!(metrics.max_relative, 0.0);
}

#[test]
fn metrics_report_max_mean_and_relative_error() {
    let actual = array![1.0_f32, 3.0].into_dyn();
    let expected = array![1.0_f32, 1.0].into_dyn();
    let metrics = compare_f32(actual.view(), expected.view()).unwrap();
    assert_eq!(metrics.max_abs, 2.0);
    assert_eq!(metrics.mean_abs, 1.0);
    assert_eq!(metrics.max_relative, 2.0);
}

#[test]
fn shape_mismatch_is_an_error() {
    let actual = array![1.0_f32, 2.0].into_dyn();
    let expected = array![[1.0_f32, 2.0]].into_dyn();
    assert!(matches!(
        compare_f32(actual.view(), expected.view()),
        Err(ParityError::ShapeMismatch { .. })
    ));
}
```

Metric definitions:

```rust
max_abs = max(abs(actual - expected))
mean_abs = mean(abs(actual - expected))
max_relative = max(abs(actual - expected) / max(abs(expected), 1e-7))
```

- [ ] **Step 2: Write failing CPU model parity tests**

```rust
use feathertalk_parity::{
    archive::GoldenArchive,
    fixture::{ForwardCase, run_cpu_forward},
};

fn golden_archive() -> GoldenArchive {
    let root = env!("CARGO_MANIFEST_DIR");
    GoldenArchive::open(format!(
        "{root}/../../tests/golden/burn-feasibility-v1.zip"
    ))
    .unwrap()
}

#[test]
fn feather_micro_matches_python_on_cpu() {
    let metrics = run_cpu_forward(&golden_archive(), ForwardCase::FeatherMicro).unwrap();
    assert!(metrics.max_abs <= 1e-4, "{metrics:?}");
}

#[test]
fn unet_production_matches_python_on_cpu() {
    let metrics = run_cpu_forward(&golden_archive(), ForwardCase::UnetProduction).unwrap();
    assert!(metrics.max_abs <= 1e-4, "{metrics:?}");
}
```

Each test extracts the archive, initializes the exact model configuration, imports the corresponding `.pth`, loads NPY input, runs Burn `CpuBackend`, and compares output with `max_abs <= 1e-4`.

Expose the exact runner contract used by these tests and later CLI commands:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForwardCase {
    FeatherMicro,
    UnetProduction,
}

pub fn run_cpu_forward(
    archive: &GoldenArchive,
    case: ForwardCase,
) -> Result<ParityMetrics, ParityError>;
```

- [ ] **Step 3: Run tests and capture the first mismatch**

```powershell
cargo test -p feathertalk-parity --test cpu_parity -- --nocapture
```

Expected: FAIL with a concrete missing key, shape difference, or numerical mismatch.

- [ ] **Step 4: Complete checkpoint key mappings from real apply reports**

Update only reviewed regex mappings. Do not enable partial loading. For every mapping, add a unit test that shows its Python source key and Burn destination key.

The final import report for each model must satisfy:

```text
missing = 0
unexpected = 0
errors = 0
ignored = only *.num_batches_tracked
```

- [ ] **Step 5: Diagnose operator-level mismatches in dependency order**

When full output exceeds tolerance, add temporary assertions and permanent focused tests in this order:

1. Conv1d/Conv2d padding and stride.
2. GroupNorm epsilon and grouping.
3. BatchNorm epsilon, momentum, and running statistics in inference.
4. Exact GELU versus approximate GELU.
5. Bilinear interpolation with `align_corners=true`.
6. Tensor concatenation axes and final sigmoid.

Fix the first divergent operation before checking later layers.

- [ ] **Step 6: Run CPU parity to the approved threshold**

```powershell
cargo test -p feathertalk-parity --test cpu_parity --release -- --nocapture
```

Expected:

```text
feather_micro max_abs <= 1e-4
unet_production max_abs <= 1e-4
```

- [ ] **Step 7: Commit CPU parity**

```powershell
git add crates/feathertalk-parity crates/feathertalk-weights/src/key_map.rs
git commit -m "test: prove CPU parity for FeatherHuBERT and UNet"
```

---

### Task 7: Prove backward and one Adam step

**Files:**

- Modify: `crates/feathertalk-models/src/train_step.rs`
- Create: `crates/feathertalk-models/tests/train_step.rs`
- Modify: `crates/feathertalk-parity/src/fixture.rs`
- Modify: `crates/feathertalk-parity/tests/cpu_parity.rs`

**Interfaces:**

- Produces: `l1_loss`, `adam_train_step`, `TrainStepParity`, and `run_cpu_train_step`.
- Consumes: `OriginalUnet<AutodiffBackend>`, image, audio, target, and mutable Adam optimizer.
- Produces: updated model and scalar loss.

- [ ] **Step 1: Write L1 and gradient tests**

Create `crates/feathertalk-models/tests/train_step.rs` with these assertions. A local `micro_batch` helper returns deterministic image, audio, and target tensors for `OriginalUnetConfig::parity_micro()`.

```rust
use burn::{
    optim::{AdamConfig, GradientsParams},
    tensor::{Tensor, TensorData},
};
use feathertalk_models::{
    backend::{CpuAutodiffBackend, CpuBackend},
    train_step::{adam_train_step, l1_loss},
    unet::OriginalUnetConfig,
};

#[test]
fn l1_loss_matches_hand_computed_value() {
    let device = Default::default();
    let prediction = Tensor::<CpuBackend, 2>::from_data(
        TensorData::from([[1.0_f32, -2.0, 3.0]]),
        &device,
    );
    let target = Tensor::<CpuBackend, 2>::zeros([1, 3], &device);
    let actual = l1_loss(prediction, target).into_scalar();
    assert!((actual - 2.0).abs() <= f32::EPSILON);
}

#[test]
fn backward_registers_output_weight_gradient() {
    let device = Default::default();
    let model = OriginalUnetConfig::parity_micro().init::<CpuAutodiffBackend>(&device);
    let (image, audio, target) = micro_batch(&device);
    let loss = l1_loss(model.forward(image, audio), target);
    let gradients = GradientsParams::from_grads(loss.backward(), &model);
    assert!(model.outc.conv.weight.grad(&gradients).is_some());
}

#[test]
fn zero_learning_rate_leaves_output_weight_unchanged() {
    let device = Default::default();
    let model = OriginalUnetConfig::parity_micro().init::<CpuAutodiffBackend>(&device);
    let before = model.outc.conv.weight.val().into_data();
    let (image, audio, target) = micro_batch(&device);
    let mut optimizer = AdamConfig::new().init();
    let (model, loss) = adam_train_step(model, &mut optimizer, image, audio, target, 0.0);
    let after = model.outc.conv.weight.val().into_data();
    assert!(loss.is_finite());
    assert_eq!(before, after);
}
```

- [ ] **Step 2: Write the golden one-step parity test**

The test loads `unet_micro_train.pth`, executes one training step with batch size 1 and the fixture Adam configuration, then compares:

- Initial loss.
- Post-step loss.
- `inc.inconv.expand_conv.weight`.
- `audio_model.conv1.expand_conv.weight`.
- `outc.conv.weight`.

Require `relative_error <= 1e-3` for every scalar and selected tensor.

Append this concrete acceptance test to `cpu_parity.rs`:

```rust
#[test]
fn unet_micro_train_step_matches_python_on_cpu() {
    let result = run_cpu_train_step(&golden_archive()).unwrap();
    assert!(result.initial_loss_relative <= 1e-3, "{result:?}");
    assert!(result.post_step_loss_relative <= 1e-3, "{result:?}");
    for (name, relative_error) in &result.selected_parameter_relative {
        assert!(*relative_error <= 1e-3, "{name}: {relative_error}");
    }
    for (name, relative_error) in &result.batch_norm_state_relative {
        assert!(*relative_error <= 1e-3, "{name}: {relative_error}");
    }
}
```

The parity crate exposes the result without hiding any compared value:

```rust
#[derive(Debug)]
pub struct TrainStepParity {
    pub initial_loss_relative: f32,
    pub post_step_loss_relative: f32,
    pub selected_parameter_relative: std::collections::BTreeMap<String, f32>,
    pub batch_norm_state_relative: std::collections::BTreeMap<String, f32>,
}

pub fn run_cpu_train_step(
    archive: &GoldenArchive,
) -> Result<TrainStepParity, ParityError>;
```

- [ ] **Step 3: Run tests and verify the training API is missing**

```powershell
cargo test -p feathertalk-models --test train_step
cargo test -p feathertalk-parity --test cpu_parity unet_micro_train_step -- --nocapture
```

Expected: FAIL because `l1_loss` and `adam_train_step` do not exist.

- [ ] **Step 4: Implement L1 loss**

```rust
pub fn l1_loss<B: Backend, const D: usize>(
    prediction: Tensor<B, D>,
    target: Tensor<B, D>,
) -> Tensor<B, 1> {
    (prediction - target).abs().mean()
}
```

- [ ] **Step 5: Implement the single-step primitive**

```rust
pub fn adam_train_step<B>(
    model: OriginalUnet<B>,
    optimizer: &mut impl burn::optim::Optimizer<OriginalUnet<B>, B>,
    image: Tensor<B, 4>,
    audio: Tensor<B, 4>,
    target: Tensor<B, 4>,
    learning_rate: f64,
) -> (OriginalUnet<B>, f32)
where
    B: burn::tensor::backend::AutodiffBackend,
{
    let prediction = model.forward(image, audio);
    let loss = l1_loss(prediction, target);
    let loss_value = loss.clone().into_scalar().elem::<f32>();
    let gradients = burn::optim::GradientsParams::from_grads(loss.backward(), &model);
    let model = optimizer.step(learning_rate, model, gradients);
    (model, loss_value)
}
```

Initialize Adam with beta1 `0.9`, beta2 `0.999`, epsilon `1e-8`, and no weight decay.

- [ ] **Step 6: Resolve training-mode parity differences**

Compare BatchNorm running mean/variance after the step in addition to selected trainable tensors. Any momentum convention difference must be handled explicitly in the Burn model configuration; do not loosen tolerance to hide it.

- [ ] **Step 7: Run training parity and commit**

```powershell
cargo fmt --all
cargo test -p feathertalk-models --test train_step
cargo test -p feathertalk-parity --test cpu_parity unet_micro_train_step --release -- --nocapture
git add crates/feathertalk-models/src/train_step.rs crates/feathertalk-models/tests/train_step.rs crates/feathertalk-parity/src/fixture.rs crates/feathertalk-parity/tests/cpu_parity.rs
git commit -m "test: prove Burn autodiff and Adam parity"
```

---

### Task 8: Prove WGPU execution and record the go/no-go result

**Files:**

- Create: `crates/feathertalk-parity/src/probe.rs`
- Create: `crates/feathertalk-parity/src/main.rs`
- Modify: `crates/feathertalk-parity/src/fixture.rs`
- Create: `crates/feathertalk-parity/tests/cli_contract.rs`
- Create: `crates/feathertalk-parity/tests/wgpu_parity.rs`
- Create: `docs/migration/burn-feasibility-report.md`

**Interfaces:**

- Produces CLI commands `probe`, `forward`, and `train-step` plus `GraphicsSelection`, `ExecutionEvidence`, `run_wgpu_forward`, and `run_wgpu_train_step`.
- Produces machine-readable JSON evidence and a reviewed Markdown decision report.
- Proves that a requested WGPU graphics API executes rather than falling back to CPU.

- [ ] **Step 1: Write the CLI contract test**

The binary must expose:

```text
feathertalk-parity probe --graphics auto|dx12|metal|vulkan
feathertalk-parity forward --model feather|unet --backend cpu|wgpu --fixture PATH
feathertalk-parity train-step --backend cpu|wgpu --fixture PATH --full
```

Each command prints one JSON object and exits nonzero when the requested backend cannot initialize or parity exceeds tolerance.

Create `tests/cli_contract.rs` before the binary implementation:

```rust
use std::process::Command;

#[test]
fn help_exposes_probe_forward_and_train_step() {
    let output = Command::new(env!("CARGO_BIN_EXE_feathertalk-parity"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    for command in ["probe", "forward", "train-step"] {
        assert!(stdout.contains(command), "missing command {command}: {stdout}");
    }
}
```

Run `cargo test -p feathertalk-parity --test cli_contract`; expected: FAIL because the binary does not exist.

- [ ] **Step 2: Implement explicit WGPU initialization**

Map graphics selection by platform:

```text
Windows: auto or dx12
macOS:   auto or metal
Linux:   auto or vulkan
```

Call the matching `burn::backend::wgpu::init_setup::<GraphicsApi>` before creating tensors. Reject graphics APIs unavailable on the current target at argument-validation time.

The JSON probe record contains:

```json
{
  "backend": "wgpu",
  "graphics": "dx12",
  "device": "default",
  "burn_version": "0.21.0",
  "rust_version": "1.92.0",
  "status": "passed"
}
```

Use the actual selected graphics string; never print `wgpu` alone as evidence of the active API.

Expose these shared records from `probe.rs` and `fixture.rs`; the CLI serializes the same values used by tests:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum GraphicsSelection {
    Auto,
    Dx12,
    Metal,
    Vulkan,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ExecutionEvidence {
    pub backend: String,
    pub graphics: String,
    pub device: String,
    pub used_cpu_fallback: bool,
}

#[derive(Debug)]
pub struct WgpuForwardResult {
    pub execution: ExecutionEvidence,
    pub metrics: ParityMetrics,
}

#[derive(Debug)]
pub struct WgpuTrainStepResult {
    pub execution: ExecutionEvidence,
    pub initial_loss: f32,
    pub gradient_norm: f32,
    pub output_weight_changed: bool,
}

pub fn run_wgpu_forward(
    archive: &GoldenArchive,
    case: ForwardCase,
    graphics: GraphicsSelection,
) -> Result<WgpuForwardResult, ParityError>;

pub fn run_wgpu_train_step(
    archive: &GoldenArchive,
    graphics: GraphicsSelection,
    full_production_model: bool,
) -> Result<WgpuTrainStepResult, ParityError>;
```

- [ ] **Step 3: Write ignored WGPU parity tests**

Mark hardware tests ignored for ordinary CPU CI:

```rust
use feathertalk_parity::{
    archive::GoldenArchive,
    fixture::{ForwardCase, run_wgpu_forward, run_wgpu_train_step},
    probe::GraphicsSelection,
};

fn golden_archive() -> GoldenArchive {
    let root = env!("CARGO_MANIFEST_DIR");
    GoldenArchive::open(format!(
        "{root}/../../tests/golden/burn-feasibility-v1.zip"
    ))
    .unwrap()
}

fn certified_graphics() -> GraphicsSelection {
    #[cfg(target_os = "windows")]
    return GraphicsSelection::Dx12;
    #[cfg(target_os = "macos")]
    return GraphicsSelection::Metal;
    #[cfg(target_os = "linux")]
    return GraphicsSelection::Vulkan;
}

#[test]
#[ignore = "requires a certified WGPU adapter"]
fn feather_matches_python_on_wgpu() {
    let result = run_wgpu_forward(
        &golden_archive(),
        ForwardCase::FeatherMicro,
        certified_graphics(),
    )
    .unwrap();
    assert_eq!(result.execution.backend, "wgpu");
    assert!(!result.execution.used_cpu_fallback);
    assert!(result.metrics.max_abs <= 1e-3, "{result:?}");
}

#[test]
#[ignore = "requires a certified WGPU adapter"]
fn production_unet_matches_python_on_wgpu() {
    let result = run_wgpu_forward(
        &golden_archive(),
        ForwardCase::UnetProduction,
        certified_graphics(),
    )
    .unwrap();
    assert_eq!(result.execution.backend, "wgpu");
    assert!(!result.execution.used_cpu_fallback);
    assert!(result.metrics.max_abs <= 1e-3, "{result:?}");
}

#[test]
#[ignore = "requires a certified WGPU adapter with training capacity"]
fn production_unet_completes_one_adam_step_on_wgpu() {
    let result = run_wgpu_train_step(&golden_archive(), certified_graphics(), true).unwrap();
    assert_eq!(result.execution.backend, "wgpu");
    assert!(!result.execution.used_cpu_fallback);
    assert!(result.initial_loss.is_finite());
    assert!(result.gradient_norm.is_finite());
    assert!(result.gradient_norm > 0.0);
    assert!(result.output_weight_changed);
}
```

Forward tests require `max_abs <= 1e-3`. The training test requires finite loss, finite gradients, a changed output-layer weight, and no backend fallback.

- [ ] **Step 4: Run the current platform probe**

On the current Windows workspace:

```powershell
cargo run -p feathertalk-parity --release -- probe --graphics dx12
```

Expected: JSON reports `backend=wgpu`, `graphics=dx12`, and `status=passed`.

- [ ] **Step 5: Run WGPU forward parity**

```powershell
cargo test -p feathertalk-parity --test wgpu_parity --release -- --ignored --nocapture
```

Expected: FeatherHuBERT and production UNet forward pass within `1e-3` max absolute error.

- [ ] **Step 6: Run the production single-step WGPU acceptance command**

```powershell
cargo run -p feathertalk-parity --release -- train-step --backend wgpu --fixture tests/golden/burn-feasibility-v1.zip --full
```

Expected: finite initial loss, finite gradient norm, changed output-layer weight, and zero backend fallback indicators. This command may take longer than normal unit tests and is required once per certified GPU/platform combination.

- [ ] **Step 7: Write the feasibility report**

Create `docs/migration/burn-feasibility-report.md` with exact command output for:

- Rust and Burn versions.
- CPU FeatherHuBERT max/mean error.
- CPU production UNet max/mean error.
- CPU micro training-step relative error.
- WGPU graphics API and device identity available from the runtime.
- WGPU FeatherHuBERT max/mean error.
- WGPU production UNet max/mean error.
- WGPU production train-step result and peak memory when available.
- A final decision line exactly `Decision: GO` only when every required threshold passes; otherwise `Decision: NO-GO` followed by the failed threshold and evidence.

- [ ] **Step 8: Run milestone-wide verification**

```powershell
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo test -p feathertalk-parity --test cpu_parity --release -- --nocapture
cargo test -p feathertalk-parity --test wgpu_parity --release -- --ignored --nocapture
git diff --check
```

Expected: all commands pass and the report says `Decision: GO`. If the report says `Decision: NO-GO`, stop after committing the evidence; do not begin milestone two or GPUI work.

- [ ] **Step 9: Commit WGPU evidence and milestone decision**

```powershell
git add crates/feathertalk-parity docs/migration/burn-feasibility-report.md
git commit -m "test: validate Burn WGPU feasibility"
```

## Milestone Exit Criteria

Milestone one is complete only when:

- The deterministic archive is reproducible and at most 20 MiB.
- Rust tests consume golden data without launching Python.
- FeatherHuBERT CPU output is within `1e-4` max absolute error.
- Production Original UNet CPU output is within `1e-4` max absolute error.
- Micro UNet Adam-step selected tensors are within `1e-3` relative error.
- FeatherHuBERT and production Original UNet execute on explicit WGPU within `1e-3` max absolute error.
- Production Original UNet completes one WGPU backward and Adam step.
- No import uses partial loading or silently ignores unknown keys.
- `docs/migration/burn-feasibility-report.md` contains `Decision: GO`.

Milestone two must not start when any exit criterion fails.
