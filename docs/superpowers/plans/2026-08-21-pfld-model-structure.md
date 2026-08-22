# PFLD Burn Model Structure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the fixed Burn `PFLD_GhostOne` structure with production shape `[N,3,192,192] -> [N,220]`, without checkpoint import or numerical parity in this subproject.

**Architecture:** Add a focused `pfld` module under `feathertalk-models`. Keep MobileOne branches, GhostOne composition, pooling, concatenation, and the output head in separate files with the existing Burn model conventions. The public config owns fixed production dimensions; tests verify tensor shapes on the CPU backend.

**Tech Stack:** Rust 1.92 edition 2024, Burn `=0.21.0`, existing `feathertalk-models` CPU backend, and standard Rust tests.

## Global Constraints

- Production configuration is exactly `width_factor = 0.5`, `input_size = 192`, `landmark_count = 110`, `num_conv_branches = 6`.
- The model accepts `[batch, 3, 192, 192]` and returns `[batch, 220]`.
- The pooled tensors concatenate to `256` channels before the final `1x1` head.
- MobileOne uses six convolution branches, a 1x1 scale branch only for kernels larger than 1, and a BatchNorm skip branch only when stride is 1 and channels match.
- Linear blocks omit ReLU; non-linear blocks use ReLU.
- This plan does not add checkpoint parsing, safetensors conversion, image processing, SCRFD, PFLD postprocessing, STN, AuxiliaryNet, FFmpeg, or new runtime dependencies.
- All Rust commands run from `rust/` in the isolated worktree.

---

### Task 1: Add PFLD configuration and module skeleton

**Files:**
- Modify: `rust/crates/feathertalk-models/src/lib.rs`
- Create: `rust/crates/feathertalk-models/src/pfld/mod.rs`
- Create: `rust/crates/feathertalk-models/src/pfld/config.rs`
- Create: `rust/crates/feathertalk-models/tests/pfld_shapes.rs`

**Interfaces:**
- Produces `PfldConfig`, `PfldConfig::production()`, `PFLD_INPUT_CHANNELS`, `PFLD_OUTPUT_VALUES`, and a crate-root `PFLD_GhostOne` export placeholder used by later tasks.
- Consumes `feathertalk_models::backend::CpuBackend`.

- [ ] **Step 1: Write the failing configuration and shape test**

```rust
use burn::tensor::Tensor;
use feathertalk_models::{PFLD_GhostOne, PFLD_OUTPUT_VALUES, PfldConfig};
use feathertalk_models::backend::CpuBackend;

#[test]
fn production_config_is_fixed() {
    let config = PfldConfig::production();
    assert_eq!(config.width_factor, 0.5);
    assert_eq!(config.input_size, 192);
    assert_eq!(config.landmark_count, 110);
    assert_eq!(config.num_conv_branches, 6);
    assert_eq!(PFLD_OUTPUT_VALUES, 220);
}

#[test]
fn production_model_shape_is_declared() {
    let device = Default::default();
    let model = PFLD_GhostOne::new(PfldConfig::production(), &device);
    let input = Tensor::<CpuBackend, 4>::zeros([1, 3, 192, 192], &device);
    assert_eq!(model.forward(input).dims(), [1, PFLD_OUTPUT_VALUES]);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p feathertalk-models --test pfld_shapes`

Expected: FAIL because the PFLD module, config, and model do not exist.

- [ ] **Step 3: Add the module skeleton and constants**

Add `pub mod pfld;` to `src/lib.rs`, define `PfldConfig`, constants, and a crate-private model constructor signature:

```rust
pub const PFLD_INPUT_CHANNELS: usize = 3;
pub const PFLD_OUTPUT_VALUES: usize = 220;

impl PfldConfig {
    pub const fn production() -> Self { /* exact fixed values */ }
}
```

The temporary model may return a zero tensor only until Task 3; keep the public `forward` signature stable.

- [ ] **Step 4: Run the focused test and commit the skeleton**

Run: `cargo test -p feathertalk-models --test pfld_shapes`

Expected: configuration test passes; shape test remains explicitly marked or scoped to the later graph implementation. Commit with `feat: add PFLD model configuration`.

### Task 2: Implement MobileOne and GhostOne building blocks

**Files:**
- Create: `rust/crates/feathertalk-models/src/pfld/mobileone.rs`
- Create: `rust/crates/feathertalk-models/src/pfld/ghost.rs`
- Modify: `rust/crates/feathertalk-models/src/pfld/mod.rs`
- Modify: `rust/crates/feathertalk-models/tests/pfld_shapes.rs`

**Interfaces:**
- Produces generic `MobileOneBlock<B>`, `GhostOneModule<B>`, and `GhostOneBottleneck<B>` with `new(...)` and `forward(...)` methods.
- Consumes `PfldConfig` and existing Burn `Backend`, `Conv2d`, `BatchNorm`, and `Relu` patterns.

- [ ] **Step 1: Write failing block shape tests**

```rust
#[test]
fn ghost_one_bottleneck_preserves_expected_stride_shapes() {
    let device = Default::default();
    let down = GhostOneBottleneck::<CpuBackend>::new(32, 40, 40, 2, 6, &device);
    let same = GhostOneBottleneck::<CpuBackend>::new(40, 40, 40, 1, 6, &device);
    let input = Tensor::<CpuBackend, 4>::zeros([1, 32, 96, 96], &device);
    assert_eq!(down.forward(input).dims(), [1, 40, 48, 48]);
    let input = Tensor::<CpuBackend, 4>::zeros([1, 40, 48, 48], &device);
    assert_eq!(same.forward(input).dims(), [1, 40, 48, 48]);
}
```

- [ ] **Step 2: Run the block test to verify it fails**

Run: `cargo test -p feathertalk-models --test pfld_shapes ghost_one_bottleneck_preserves_expected_stride_shapes`

Expected: FAIL because the block types do not exist.

- [ ] **Step 3: Implement MobileOne branch composition**

Use `Conv2dConfig` with explicit stride/padding/groups and `BatchNormConfig`. For training-form blocks, sum six Conv-BN branches, the optional 1x1 Conv-BN scale branch, and the optional BatchNorm skip branch before applying ReLU when `is_linear == false`. Keep branch fields as Burn modules so future checkpoint import can address them deterministically.

- [ ] **Step 4: Implement GhostOne composition**

Implement `GhostOneModule` as a primary MobileOne 1x1 path plus a depthwise MobileOne 3x3 path, concatenate both outputs, and truncate to `out_channels`. Implement `GhostOneBottleneck` as GhostOne module, optional depthwise MobileOne when stride is 2, and a linear GhostOne module.

- [ ] **Step 5: Run focused block tests and commit**

Run: `cargo test -p feathertalk-models --test pfld_shapes ghost_one_bottleneck_preserves_expected_stride_shapes`

Expected: PASS. Commit with `feat: add PFLD MobileOne ghost blocks`.

### Task 3: Implement the production PFLD graph and head

**Files:**
- Modify: `rust/crates/feathertalk-models/src/pfld/config.rs`
- Modify: `rust/crates/feathertalk-models/src/pfld/mod.rs`
- Create: `rust/crates/feathertalk-models/src/pfld/model.rs`
- Modify: `rust/crates/feathertalk-models/tests/pfld_shapes.rs`

**Interfaces:**
- Produces `PFLD_GhostOne<B>::new(config, device)` and `forward(Tensor<B,4>) -> Tensor<B,2>`.
- Consumes the MobileOne/GhostOne blocks from Task 2.

- [ ] **Step 1: Extend tests for batch shape and graph dimensions**

```rust
#[test]
fn production_model_supports_multiple_batch_items() {
    let device = Default::default();
    let model = PFLD_GhostOne::new(PfldConfig::production(), &device);
    let input = Tensor::<CpuBackend, 4>::zeros([2, 3, 192, 192], &device);
    assert_eq!(model.forward(input).dims(), [2, 220]);
}
```

Also assert the graph's pooled channel sum is `32 + 40 + 48 + 72 + 64 == 256` in a config-level test.

- [ ] **Step 2: Run the new tests to verify the missing graph behavior**

Run: `cargo test -p feathertalk-models --test pfld_shapes`

Expected: the batch/graph tests fail until the complete model is implemented.

- [ ] **Step 3: Implement the exact production graph**

At width factor `0.5`, use these stages:

```text
conv1: 3 -> 32, 3x3, stride 2
conv2: 32 -> 32, depthwise 3x3, stride 1
conv3: 32 -> 40 (hidden 48), then 40 -> 40 (hidden 60), twice
conv4: 40 -> 48 (hidden 100), then 48 -> 48 (hidden 120), twice
conv5: 48 -> 72 (hidden 168), then 72 -> 72 (hidden 252), three times
conv6: 72 -> 8 (hidden 108), stride 1
conv7: 8 -> 16, 3x3, stride 1
conv8: 16 -> 64, `input_size / 16` (12x12), stride 1, no BatchNorm, with ReLU
head: 256 -> 220, 1x1
```

Use average pooling factors `96`, `48`, `24`, and `12` on the first four feature maps, concatenate pooled maps with `x5`, apply the output head, and flatten `[N,220,1,1]` to `[N,220]`.

- [ ] **Step 4: Run all PFLD shape tests and commit**

Run: `cargo test -p feathertalk-models --test pfld_shapes`

Expected: all shape tests pass. Commit with `feat: add PFLD GhostOne model graph`.

### Task 4: Public API and focused acceptance

**Files:**
- Modify: `rust/crates/feathertalk-models/src/lib.rs`
- Modify: `rust/crates/feathertalk-models/tests/pfld_shapes.rs`
- Create: `rust/crates/feathertalk-models/tests/pfld_public_api.rs`

**Interfaces:**
- Consumes the complete `PfldConfig`, `PFLD_GhostOne`, constants, and `forward` API.
- Produces crate-root-only acceptance coverage.

- [ ] **Step 1: Write the crate-root public API test**

```rust
use burn::tensor::Tensor;
use feathertalk_models::{PFLD_GhostOne, PfldConfig, PFLD_INPUT_CHANNELS, PFLD_OUTPUT_VALUES};
use feathertalk_models::backend::CpuBackend;

#[test]
fn pfld_public_api_is_crate_root_only() {
    let device = Default::default();
    let model = PFLD_GhostOne::new(PfldConfig::production(), &device);
    let input = Tensor::<CpuBackend, 4>::zeros([1, PFLD_INPUT_CHANNELS, 192, 192], &device);
    assert_eq!(model.forward(input).dims(), [1, PFLD_OUTPUT_VALUES]);
}
```

- [ ] **Step 2: Run focused acceptance commands**

```powershell
cargo fmt --check
cargo clippy -p feathertalk-models --all-targets --all-features -- -D warnings
cargo test -p feathertalk-models --all-targets
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 3: Commit acceptance**

```powershell
git add rust/crates/feathertalk-models docs/superpowers/plans/2026-08-21-pfld-model-structure.md
git commit -m "test: accept PFLD model structure"
```

## Plan Self-Review

- Spec coverage: production config, graph channels/spatial sizes, six-branch MobileOne semantics, GhostOne composition, multi-scale pooling, output shape, public API, and dependency exclusions map to Tasks 1-4.
- No checkpoint import, numerical parity, image processing, or STN behavior is introduced.
- All later task signatures consume types defined in earlier tasks.
- No placeholder or undefined step remains.
