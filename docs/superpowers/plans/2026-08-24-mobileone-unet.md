# MobileOne UNet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Burn MobileOne UNet training graph and a separately typed, numerically equivalent reparameterized inference graph.

**Architecture:** Extract the existing PFLD MobileOne block into a crate-level shared module, extend it with two-axis stride and explicit Conv+BN fusion, then build parallel MobileOne UNet training and inference component trees. Conversion reads frozen BatchNorm running statistics, creates new fused convolution parameters, and never mutates the source training model.

**Tech Stack:** Rust 1.92, Burn 0.21.0, NdArray CPU backend, WGPU-compatible generic modules, Cargo test/clippy/fmt.

## Global Constraints

- Production image input is `[B, 6, 160, 160]`; FeatherHuBERT input is `[B, 16, 32, 32]`; output is `[B, 3, 160, 160]` after sigmoid.
- Production channels are `[32, 64, 128, 256, 512]`; every UNet MobileOne block uses two convolution branches.
- CPU float32 reparameterization must satisfy `max_abs_error <= 1e-4` at block and complete micro-model levels.
- Training and inference graphs are different Rust types; conversion must not mutate the source model.
- Do not add checkpoint import, VGG19, DataLoader, loss presets, checkpoint recovery, ONNX, worker, GPUI, Wenet runtime, or video synthesis in this slice.
- WGPU must never silently fall back to CPU.
- Use TDD for every production behavior and explicitly stage files; never use `git add .`.

---

### Task 1: Share and extend the MobileOne training block

**Files:**
- Create: `rust/crates/feathertalk-models/src/mobileone.rs`
- Modify: `rust/crates/feathertalk-models/src/lib.rs`
- Modify: `rust/crates/feathertalk-models/src/pfld/mobileone.rs`
- Test: `rust/crates/feathertalk-models/tests/mobileone_reparameterization.rs`

**Interfaces:**
- Consumes: existing `MobileOneBlock::new(in_channels, out_channels, kernel_size, stride, padding, groups, num_conv_branches, is_linear, device)` used by PFLD.
- Produces: the same constructor plus `MobileOneBlock::new_with_stride(..., stride: [usize; 2], ...)`, preserving `forward(Tensor<B,4>) -> Tensor<B,4>`.

- [ ] **Step 1: Write the failing shared-block tests**

Create the test file with a compile-time `Module` assertion, the existing square-stride shape contract, and an anisotropic stride case:

```rust
#[test]
fn anisotropic_stride_halves_only_width() {
    let device = Default::default();
    let block = MobileOneBlock::<CpuBackend>::new_with_stride(
        4, 8, 3, [1, 2], 1, 1, 2, false, &device,
    );
    let input = Tensor::zeros([1, 4, 12, 14], &device);
    assert_eq!(block.forward(input).dims(), [1, 8, 12, 7]);
}
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `cargo test -p feathertalk-models --test mobileone_reparameterization anisotropic_stride_halves_only_width -- --exact`

Expected: compilation fails because `new_with_stride` and the crate-level shared implementation do not exist.

- [ ] **Step 3: Extract the existing implementation and add two-axis stride**

Move the current PFLD implementation to `src/mobileone.rs`. Keep `new(...)` as a compatibility wrapper that passes `[stride, stride]`; implement `new_with_stride(...)` using the two-axis value for every main and scale branch. Keep skip only for `[1,1]` with equal channels. Re-export the shared type through both the crate root and `pfld::mobileone` compatibility module.

- [ ] **Step 4: Run focused and PFLD regression tests**

Run:

```powershell
cargo test -p feathertalk-models --test mobileone_reparameterization
cargo test -p feathertalk-models --test pfld_shapes
```

Expected: both pass with no changed PFLD shapes.

- [ ] **Step 5: Commit the shared block**

```powershell
git add rust/crates/feathertalk-models/src/mobileone.rs rust/crates/feathertalk-models/src/lib.rs rust/crates/feathertalk-models/src/pfld/mobileone.rs rust/crates/feathertalk-models/tests/mobileone_reparameterization.rs
git commit -m "refactor: share MobileOne training block"
```

### Task 2: Fuse MobileOne branches into one inference convolution

**Files:**
- Modify: `rust/crates/feathertalk-models/src/mobileone.rs`
- Modify: `rust/crates/feathertalk-models/src/lib.rs`
- Modify: `rust/crates/feathertalk-models/tests/mobileone_reparameterization.rs`

**Interfaces:**
- Consumes: `MobileOneBlock<B>` branch parameters and BatchNorm running state.
- Produces: `ReparameterizedMobileOneBlock<B>` with `forward(Tensor<B,4>) -> Tensor<B,4>` and `MobileOneBlock::reparameterize(&self) -> ReparameterizedMobileOneBlock<B>`.

- [ ] **Step 1: Add failing numerical-equivalence tests**

Cover these independent cases with deterministic CPU tensors:

```rust
fn assert_reparameterized_close(block: MobileOneBlock<CpuBackend>, input: Tensor<CpuBackend, 4>) {
    let expected = block.forward(input.clone()).into_data().to_vec::<f32>().unwrap();
    let actual = block.reparameterize().forward(input).into_data().to_vec::<f32>().unwrap();
    let max_abs = expected.iter().zip(actual.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_abs <= 1.0e-4, "max_abs={max_abs}");
}
```

Add separate tests for `1x1` without scale, `3x3` with scale, equal-channel skip, depthwise groups, and anisotropic stride without skip.

- [ ] **Step 2: Run the equivalence tests and verify RED**

Run: `cargo test -p feathertalk-models --test mobileone_reparameterization reparameterized -- --nocapture`

Expected: compilation fails because the inference block and `reparameterize` method do not exist.

- [ ] **Step 3: Implement Conv+BN and branch fusion**

Implement the fixed formula from the spec:

```text
scale = gamma / sqrt(running_var + epsilon)
kernel = conv.weight * scale.reshape([out,1,1,1])
bias = beta - running_mean * scale
```

Pad the `1x1` scale kernel into the center of the main odd kernel. Build grouped identity weights for skip BN with `identity[o, o % (in_channels/groups), center, center] = 1`. Sum all fused kernels and biases, then construct one `Conv2d` preserving stride, dilation, groups, and padding. Apply the same optional ReLU in `ReparameterizedMobileOneBlock::forward`.

- [ ] **Step 4: Run equivalence and PFLD tests**

Run:

```powershell
cargo test -p feathertalk-models --test mobileone_reparameterization
cargo test -p feathertalk-models --test pfld_shapes
```

Expected: all tests pass and every measured max error is at most `1e-4`.

- [ ] **Step 5: Commit fusion support**

```powershell
git add rust/crates/feathertalk-models/src/mobileone.rs rust/crates/feathertalk-models/src/lib.rs rust/crates/feathertalk-models/tests/mobileone_reparameterization.rs
git commit -m "feat: reparameterize MobileOne blocks"
```

### Task 3: Build MobileOne UNet training components

**Files:**
- Create: `rust/crates/feathertalk-models/src/unet/mobileone_blocks.rs`
- Modify: `rust/crates/feathertalk-models/src/unet/blocks.rs`
- Modify: `rust/crates/feathertalk-models/src/unet/config.rs`
- Modify: `rust/crates/feathertalk-models/src/unet/mod.rs`
- Test: `rust/crates/feathertalk-models/tests/mobileone_unet.rs`

**Interfaces:**
- Produces: `MobileOneUnetConfig { channels: [usize;5], num_conv_branches: usize }`, `production()`, `parity_micro()`, and training components used by the full model.
- Reuses: a crate-private `upsample_and_concat(input, skip)` helper extracted from the existing Original UNet `Up` implementation.

- [ ] **Step 1: Add failing config and component shape tests**

Test exact production constants and component behavior:

```rust
#[test]
fn production_mobileone_config_is_fixed() {
    let config = MobileOneUnetConfig::production();
    assert_eq!(config.channels, [32, 64, 128, 256, 512]);
    assert_eq!(config.num_conv_branches, 2);
}
```

Add tests that a micro down block maps `160 -> 80`, an up block restores its skip size, and the MobileOne Hubert audio branch maps `[1,16,32,32]` to `[1,ch[4],10,10]`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `cargo test -p feathertalk-models --test mobileone_unet production_mobileone_config_is_fixed -- --exact`

Expected: compilation fails because `MobileOneUnetConfig` and MobileOne UNet components do not exist.

- [ ] **Step 3: Implement the training components**

Implement training-form separable, double-conv, in, down, up, and Hubert audio components exactly as `unet_mobileone.py`. Extract the existing bilinear `align_corners=true` upsample/pad/concat logic into one crate-private helper and make Original UNet continue using it.

- [ ] **Step 4: Run MobileOne and Original UNet component tests**

Run:

```powershell
cargo test -p feathertalk-models --test mobileone_unet
cargo test -p feathertalk-models --test unet_shapes
```

Expected: new component tests pass and Original UNet behavior is unchanged.

- [ ] **Step 5: Commit training components**

```powershell
git add rust/crates/feathertalk-models/src/unet/mobileone_blocks.rs rust/crates/feathertalk-models/src/unet/blocks.rs rust/crates/feathertalk-models/src/unet/config.rs rust/crates/feathertalk-models/src/unet/mod.rs rust/crates/feathertalk-models/tests/mobileone_unet.rs
git commit -m "feat: add MobileOne UNet components"
```

### Task 4: Add complete training and inference model graphs

**Files:**
- Create: `rust/crates/feathertalk-models/src/unet/mobileone_model.rs`
- Modify: `rust/crates/feathertalk-models/src/unet/mobileone_blocks.rs`
- Modify: `rust/crates/feathertalk-models/src/unet/config.rs`
- Modify: `rust/crates/feathertalk-models/src/unet/mod.rs`
- Modify: `rust/crates/feathertalk-models/src/lib.rs`
- Modify: `rust/crates/feathertalk-models/tests/mobileone_unet.rs`

**Interfaces:**
- Produces: `MobileOneUnet<B>::forward(image, audio)`, `MobileOneUnet::reparameterize(&self)`, and `MobileOneUnetInference<B>::forward(image, audio)`.
- Both forwards enforce image `[B,6,160,160]`, audio `[B,16,32,32]`, equal batch size, and output `[B,3,160,160]`.

- [ ] **Step 1: Add failing complete-model tests**

Add tests for public Burn module types, micro output shape/range, production output shape, complete micro-model numerical equivalence, source-model immutability, and the autodiff output gradient:

```rust
#[test]
fn micro_training_and_inference_graphs_are_equivalent() {
    let device = Default::default();
    let model = MobileOneUnetConfig::parity_micro().init::<CpuBackend>(&device);
    let image = Tensor::ones([1, 6, 160, 160], &device);
    let audio = Tensor::ones([1, 16, 32, 32], &device);
    let expected = model.forward(image.clone(), audio.clone());
    let inference = model.reparameterize();
    let actual = inference.forward(image, audio);
    assert_max_abs(expected, actual, 1.0e-4);
}

#[test]
fn mobileone_output_weight_receives_gradient() {
    let device = Default::default();
    let model = MobileOneUnetConfig::parity_micro().init::<CpuAutodiffBackend>(&device);
    let image = Tensor::ones([1, 6, 160, 160], &device);
    let audio = Tensor::ones([1, 16, 32, 32], &device);
    let target = Tensor::zeros([1, 3, 160, 160], &device);
    let gradients = l1_loss(model.forward(image, audio), target).backward();
    assert!(model.outc.conv.weight.grad(&gradients).is_some());
}
```

- [ ] **Step 2: Run the complete-model tests and verify RED**

Run: `cargo test -p feathertalk-models --test mobileone_unet micro_training_and_inference_graphs_are_equivalent -- --exact --nocapture`

Expected: compilation fails because complete model types and conversion do not exist.

- [ ] **Step 3: Implement parallel model trees and conversion**

Build the training model with image encoder, Hubert audio encoder, bottleneck concat/fuse, four up blocks, output convolution, and sigmoid. Build matching inference component types whose constructors take references to training components and reparameterize every MobileOne block while cloning ordinary conv/BN/output modules. Validate fixed input contracts before graph execution.

- [ ] **Step 4: Run complete-model and regression tests**

Run:

```powershell
cargo test -p feathertalk-models --test mobileone_unet -- --nocapture
cargo test -p feathertalk-models --test mobileone_reparameterization
cargo test -p feathertalk-models --test unet_shapes
cargo test -p feathertalk-models --test pfld_shapes
```

Expected: all pass, including production shape and full micro-model `max_abs_error <= 1e-4`.

- [ ] **Step 5: Commit complete model graphs**

```powershell
git add rust/crates/feathertalk-models/src/unet/mobileone_model.rs rust/crates/feathertalk-models/src/unet/mobileone_blocks.rs rust/crates/feathertalk-models/src/unet/config.rs rust/crates/feathertalk-models/src/unet/mod.rs rust/crates/feathertalk-models/src/lib.rs rust/crates/feathertalk-models/tests/mobileone_unet.rs
git commit -m "feat: add MobileOne UNet graphs"
```

### Task 5: Run milestone acceptance verification

**Files:**
- No production changes.

**Interfaces:**
- Consumes: all tests and public APIs completed in Tasks 1-4.
- Proves: the isolated branch satisfies the slice acceptance commands with no uncommitted implementation changes.

- [ ] **Step 1: Re-run the focused model suite**

Run:

```powershell
cargo test -p feathertalk-models --test mobileone_unet -- --nocapture
cargo test -p feathertalk-models --test mobileone_reparameterization
cargo test -p feathertalk-models --test train_step
```

- [ ] **Step 2: Run full fresh verification**

Run:

```powershell
cargo test --workspace --all-targets
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
git diff --check
```

Expected: every command exits 0; CPU tests report 0 failed; only environment-gated WGPU/media tests remain explicitly ignored.

- [ ] **Step 3: Inspect branch status and commit only if formatting changed tracked files**

```powershell
git status --short
```

Expected: clean status. If `cargo fmt` changed tracked Rust files, inspect the diff, explicitly stage only those files, and commit them as `chore: format MobileOne UNet` before repeating verification.

## Plan Self-Review

- Spec coverage: shared MobileOne semantics, anisotropic stride, fixed UNet graph, separate inference type, exact fusion formula, shape/range checks, source immutability, gradient proof, numerical threshold, exclusions, and full verification map to Tasks 1-5.
- Placeholder scan: no `TBD`, `TODO`, “similar to”, or unspecified implementation/error-handling steps remain.
- Type consistency: `MobileOneBlock::new_with_stride`, `ReparameterizedMobileOneBlock`, `MobileOneUnetConfig`, `MobileOneUnet`, and `MobileOneUnetInference` use the same names and signatures throughout.
