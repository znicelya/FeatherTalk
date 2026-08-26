# Pure Rust BGR Frame Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dependency-light, side-effect-free BGR24 frame kernel to `feathertalk-inference` for crop, C++-compatible bilinear resize, UNet image preparation, prediction replacement, and paste-back.

**Architecture:** `frame.rs` owns the validated `BgrFrame` buffer and all pixel operations. It exposes small pure functions that consume the existing `RenderGeometry` and `feathertalk_preprocess::FaceBoundingBox`; no model, decoder, filesystem, process, or global state is involved. Tensor values remain flat channel-first `Vec<f32>` values so a later Burn adapter can copy them without redefining pixel semantics.

**Tech Stack:** Rust 1.92 edition 2024, standard library, `thiserror`, `feathertalk-preprocess`, existing `feathertalk-inference` render types, and `tempfile` only where the existing integration-test harness already uses it.

## Global Constraints

- Runtime dependencies remain limited to `thiserror`, `feathertalk-preprocess`, and `feathertalk-media`; do not add Burn, WGPU, image/OpenCV, ndarray, FFmpeg bindings, or unsafe code.
- The BGR layout is interleaved row-major BGR24, with exact `width × height × 3` storage and no implicit stride or padding.
- Crop uses left/top inclusive and right/bottom exclusive `FaceBoundingBox` coordinates.
- Resize uses the C++ half-pixel formula and edge clamp from `FeatherTalk-CPP/src/main.cc`; channel values are rounded with Rust `f32::round` after clamping to `[0,255]`.
- Standard geometry is exactly crop `168`, inner `160`, border `4`; mouth mask comes from `feathertalk_preprocess::default_crop_spec().mouth_mask` (`5,5,150,145`).
- UNet image input is `[1,6,160,160]` and prediction is `[1,3,160,160]`, both flat channel-first buffers; prediction values must be finite before any mutation.
- All public wrappers use private fields and immutable accessors; all arithmetic that can overflow is checked.
- Every new production behavior gets a failing integration test first; tests import only from the crate root and use hand-derived expected pixels.
- Never read, modify, stage, commit, rename, or delete `demo/kanghui_training_video_featherhubert_188_latest/`.

---

## File Map

Create or modify only these paths in the implementation worktree:

- Create `rust/crates/feathertalk-inference/src/frame.rs` — BGR frame value type, tensor wrapper, crop/resize/input/prediction/paste/render functions.
- Modify `rust/crates/feathertalk-inference/src/error.rs` — structured frame and tensor errors.
- Modify `rust/crates/feathertalk-inference/src/lib.rs` — crate-root exports.
- Create `rust/crates/feathertalk-inference/tests/frame.rs` — all frame-kernel integration tests.
- Create `rust/crates/feathertalk-inference/tests/frame_public_api.rs` — crate-root-only API smoke test.
- Create `docs/superpowers/specs/2026-08-26-bgr-frame-kernel-design.md` and this plan document (already committed before implementation).

No existing model, media, preprocess, or protected demo files are modified.

### Task 1: Validated BGR frame value type and error variants

**Files:**
- Modify: `rust/crates/feathertalk-inference/src/error.rs`
- Modify: `rust/crates/feathertalk-inference/src/lib.rs`
- Create: `rust/crates/feathertalk-inference/src/frame.rs`
- Create: `rust/crates/feathertalk-inference/tests/frame.rs`

**Interfaces:**
- Produces `BgrFrame::new(width, height, bytes)`, `width()`, `height()`, `as_bytes()`, `into_bytes()`, and `pixel(x, y)`.
- Produces `InferenceError` variants `InvalidFrameDimensions`, `FrameBufferLengthMismatch`, `PixelOutOfRange`, and `AllocationFailure`.

- [ ] **Step 1: Write failing tests for frame invariants.**

Add these tests before implementing `BgrFrame`:

```rust
use feathertalk_inference::{BgrFrame, InferenceError};

#[test]
fn bgr_frame_keeps_dimensions_and_interleaved_bytes() {
    let bytes = vec![10, 20, 30, 40, 50, 60];
    let frame = BgrFrame::new(2, 1, bytes.clone()).unwrap();
    assert_eq!((frame.width(), frame.height()), (2, 1));
    assert_eq!(frame.as_bytes(), bytes.as_slice());
    assert_eq!(frame.pixel(1, 0).unwrap(), [40, 50, 60]);
    assert_eq!(frame.clone().into_bytes(), bytes);
}

#[test]
fn bgr_frame_rejects_zero_dimensions_and_wrong_length() {
    assert!(matches!(
        BgrFrame::new(0, 1, Vec::new()),
        Err(InferenceError::InvalidFrameDimensions { .. })
    ));
    assert!(matches!(
        BgrFrame::new(2, 1, vec![0; 5]),
        Err(InferenceError::FrameBufferLengthMismatch { expected: 6, actual: 5 })
    ));
}

#[test]
fn bgr_frame_rejects_out_of_range_pixels() {
    let frame = BgrFrame::new(2, 2, vec![0; 12]).unwrap();
    assert!(matches!(
        frame.pixel(2, 0),
        Err(InferenceError::PixelOutOfRange { x: 2, y: 0, .. })
    ));
    assert!(matches!(
        frame.pixel(0, 2),
        Err(InferenceError::PixelOutOfRange { x: 0, y: 2, .. })
    ));
}
```

- [ ] **Step 2: Run the focused test and confirm the intended RED failure.**

Run from `E:\workspace\github\FeatherTalk\rust`:

```powershell
cargo test -p feathertalk-inference --test frame bgr_frame -- --nocapture
```

Expected: compilation fails because `BgrFrame` and the new error variants do not exist.

- [ ] **Step 3: Implement the minimal value type and checked allocation helper.**

In `frame.rs`, use private `width: u32`, `height: u32`, and `bgr: Vec<u8>`. Compute expected length with checked `usize::try_from(width)`, `checked_mul(height)`, and `checked_mul(3)`; return existing `ArithmeticOverflow` on arithmetic failure. Reject zero dimensions before checking length. `pixel` checks coordinates, computes `(y * width + x) * 3` with checked arithmetic, and returns a copied `[u8; 3]`. Use `Vec::try_reserve_exact` in the internal allocator and map failure to `AllocationFailure`.

- [ ] **Step 4: Run the focused tests and verify GREEN.**

```powershell
cargo test -p feathertalk-inference --test frame bgr_frame -- --nocapture
```

Expected: all three frame invariant tests pass with no warnings.

- [ ] **Step 5: Commit the isolated value-type change.**

```powershell
git add rust/crates/feathertalk-inference/src/error.rs rust/crates/feathertalk-inference/src/lib.rs rust/crates/feathertalk-inference/src/frame.rs rust/crates/feathertalk-inference/tests/frame.rs
git commit -m "feat: add validated BGR frame buffer"
```

### Task 2: Crop and C++-compatible bilinear resize

**Files:**
- Modify: `rust/crates/feathertalk-inference/src/frame.rs`
- Modify: `rust/crates/feathertalk-inference/src/error.rs`
- Modify: `rust/crates/feathertalk-inference/tests/frame.rs`

**Interfaces:**
- Produces `crop_bgr(&BgrFrame, &FaceBoundingBox) -> Result<BgrFrame, InferenceError>`.
- Produces `resize_bilinear(&BgrFrame, width, height) -> Result<BgrFrame, InferenceError>`.
- Produces `InvalidBbox` and `InvalidResizeTarget` errors.

- [ ] **Step 1: Add failing crop and resize tests.**

Append tests with literals derived independently from the source fixture:

```rust
use feathertalk_inference::{crop_bgr, resize_bilinear};
use feathertalk_preprocess::FaceBoundingBox;

#[test]
fn crop_copies_a_left_top_inclusive_region() {
    let frame = BgrFrame::new(
        3,
        2,
        vec![
            0, 1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16, 17,
        ],
    ).unwrap();
    let crop = crop_bgr(&frame, &FaceBoundingBox { xmin: 1, ymin: 0, xmax: 3, ymax: 2 }).unwrap();
    assert_eq!((crop.width(), crop.height()), (2, 2));
    assert_eq!(crop.as_bytes(), &[3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17]);
}

#[test]
fn crop_rejects_negative_or_outside_bbox_without_panicking() {
    let frame = BgrFrame::new(3, 2, vec![0; 18]).unwrap();
    for bbox in [
        FaceBoundingBox { xmin: -1, ymin: 0, xmax: 2, ymax: 2 },
        FaceBoundingBox { xmin: 1, ymin: 0, xmax: 4, ymax: 2 },
        FaceBoundingBox { xmin: 2, ymin: 1, xmax: 2, ymax: 2 },
    ] {
        assert!(matches!(crop_bgr(&frame, &bbox), Err(InferenceError::InvalidBbox { .. })));
    }
}

#[test]
fn resize_bilinear_matches_half_pixel_average_and_edges() {
    let source = BgrFrame::new(2, 2, vec![0,0,0, 10,10,10, 20,20,20, 30,30,30]).unwrap();
    let one = resize_bilinear(&source, 1, 1).unwrap();
    assert_eq!(one.pixel(0, 0).unwrap(), [15, 15, 15]);
    let enlarged = resize_bilinear(&source, 3, 3).unwrap();
    assert_eq!(enlarged.pixel(0, 0).unwrap(), [0, 0, 0]);
    assert_eq!(enlarged.pixel(1, 1).unwrap(), [15, 15, 15]);
    assert_eq!(enlarged.pixel(2, 2).unwrap(), [30, 30, 30]);
}

#[test]
fn resize_rejects_zero_target_dimensions() {
    let source = BgrFrame::new(1, 1, vec![1, 2, 3]).unwrap();
    assert!(matches!(
        resize_bilinear(&source, 0, 1),
        Err(InferenceError::InvalidResizeTarget { .. })
    ));
}
```

- [ ] **Step 2: Run the new tests and confirm RED.**

```powershell
cargo test -p feathertalk-inference --test frame crop -- --nocapture
cargo test -p feathertalk-inference --test frame resize -- --nocapture
```

Expected: compilation fails because the two functions and errors are absent.

- [ ] **Step 3: Implement bbox validation and row-copy crop.**

Use checked `FaceBoundingBox::xmax.checked_sub(xmin)` and the equivalent y calculation. Reject negative coordinates, non-positive extents, and any right/bottom coordinate greater than the frame dimensions using `i64` comparisons. Allocate exactly `crop_width × crop_height × 3`, then copy each source row with `copy_from_slice`.

- [ ] **Step 4: Implement the half-pixel resize formula.**

For each target coordinate use `scale = source_size as f32 / target_size as f32`, `source = (target + 0.5) * scale - 0.5`, floor before clamping the integer sample coordinate, and derive the weight from the unclamped floor exactly as the C++ reference does. Interpolate each BGR channel, call `.round()`, clamp to `[0,255]`, and store the byte. Use checked output allocation and no external image code.

- [ ] **Step 5: Run crop/resize tests and the formatter/linter for the crate.**

```powershell
cargo test -p feathertalk-inference --test frame
cargo fmt --all -- --check
cargo clippy -p feathertalk-inference --all-targets --all-features -- -D warnings
```

Expected: all focused tests pass and clippy emits zero warnings.

- [ ] **Step 6: Commit crop and resize.**

```powershell
git add rust/crates/feathertalk-inference/src/error.rs rust/crates/feathertalk-inference/src/frame.rs rust/crates/feathertalk-inference/tests/frame.rs
git commit -m "feat: add deterministic BGR crop and resize"
```

### Task 3: UNet image input and prediction replacement

**Files:**
- Modify: `rust/crates/feathertalk-inference/src/frame.rs`
- Modify: `rust/crates/feathertalk-inference/src/error.rs`
- Modify: `rust/crates/feathertalk-inference/tests/frame.rs`

**Interfaces:**
- Produces `UnetImageInput { shape(), as_slice() }`.
- Produces `build_unet_image_input(&BgrFrame, &RenderGeometry) -> Result<UnetImageInput, InferenceError>`.
- Produces `apply_unet_prediction(&mut BgrFrame, &[f32], &RenderGeometry) -> Result<(), InferenceError>`.
- Produces `TensorShapeMismatch` and `NonFinitePrediction` errors.

- [ ] **Step 1: Add failing input/prediction tests.**

Use a standard-size synthetic crop with every byte set to 64, then change one non-mask inner pixel and one mask pixel. Add:

```rust
use feathertalk_inference::{apply_unet_prediction, build_unet_image_input, RenderGeometry};

#[test]
fn image_input_is_bgr_channel_first_and_masks_only_the_mouth_rectangle() {
    let mut bytes = vec![64; 168 * 168 * 3];
    let pixel = (4 + 6) * 168 * 3 + (4 + 6) * 3;
    bytes[pixel..pixel + 3].copy_from_slice(&[128, 0, 255]);
    let crop = BgrFrame::new(168, 168, bytes).unwrap();
    let input = build_unet_image_input(&crop, &RenderGeometry::standard()).unwrap();
    assert_eq!(input.shape(), [1, 6, 160, 160]);
    let plane = 160 * 160;
    let offset = 6 * 160 + 6;
    assert_eq!(&input.as_slice()[offset..offset + 1], &[128.0 / 255.0]);
    assert_eq!(input.as_slice()[plane + offset], 0.0);
    assert_eq!(input.as_slice()[2 * plane + offset], 1.0);
    assert_eq!(input.as_slice()[3 * plane + offset], 0.0);
    assert_eq!(input.as_slice()[4 * plane + offset], 0.0);
    assert_eq!(input.as_slice()[5 * plane + offset], 0.0);
}

#[test]
fn prediction_is_clamped_rounded_and_keeps_the_four_pixel_border() {
    let mut crop = BgrFrame::new(168, 168, vec![7; 168 * 168 * 3]).unwrap();
    let mut prediction = vec![0.0; 3 * 160 * 160];
    prediction[0] = -1.0;
    prediction[160 * 160] = 0.5;
    prediction[2 * 160 * 160] = 2.0;
    apply_unet_prediction(&mut crop, &prediction, &RenderGeometry::standard()).unwrap();
    assert_eq!(crop.pixel(0, 0).unwrap(), [7, 7, 7]);
    assert_eq!(crop.pixel(4, 4).unwrap(), [0, 128, 255]);
}

#[test]
fn prediction_rejects_wrong_length_and_non_finite_values_before_mutation() {
    let geometry = RenderGeometry::standard();
    let mut crop = BgrFrame::new(168, 168, vec![9; 168 * 168 * 3]).unwrap();
    assert!(matches!(
        apply_unet_prediction(&mut crop, &[0.0; 3], &geometry),
        Err(InferenceError::TensorShapeMismatch { .. })
    ));
    let mut prediction = vec![0.0; 3 * 160 * 160];
    prediction[42] = f32::NAN;
    assert!(matches!(
        apply_unet_prediction(&mut crop, &prediction, &geometry),
        Err(InferenceError::NonFinitePrediction { index: 42 })
    ));
    assert_eq!(crop.pixel(4, 4).unwrap(), [9, 9, 9]);
}
```

The mask assertion intentionally uses inner coordinate `(6,6)`, which lies inside the fixed `x=5..154`, `y=5..149` mask rectangle.

- [ ] **Step 2: Run the tests and confirm RED.**

```powershell
cargo test -p feathertalk-inference --test frame image_input -- --nocapture
cargo test -p feathertalk-inference --test frame prediction -- --nocapture
```

Expected: compilation fails because `UnetImageInput` and the two functions are absent.

- [ ] **Step 3: Implement the tensor wrapper and standard-geometry validation.**

Validate that the crop is exactly `geometry.crop_size() × geometry.crop_size()` and that the geometry equals `RenderGeometry::standard()`. Read `default_crop_spec().mouth_mask`, verify it fits inside the inner square, allocate `6 × 160 × 160` values, and write normalized BGR channel planes. Expose only `shape()` and `as_slice()`.

- [ ] **Step 4: Implement transactional prediction replacement.**

Validate crop geometry, expected `3 × 160 × 160` length, and every value’s finiteness before touching the crop. Then write channel-first BGR values into the `(border,border)` inner region with `clamp(value * 255.0, 0.0, 255.0).round() as u8`.

- [ ] **Step 5: Run focused tests and clippy.**

```powershell
cargo test -p feathertalk-inference --test frame
cargo clippy -p feathertalk-inference --all-targets --all-features -- -D warnings
```

Expected: all input/prediction tests pass with no warnings.

- [ ] **Step 6: Commit the tensor boundary.**

```powershell
git add rust/crates/feathertalk-inference/src/error.rs rust/crates/feathertalk-inference/src/frame.rs rust/crates/feathertalk-inference/tests/frame.rs
git commit -m "feat: add UNet image tensor frame bridge"
```

### Task 4: Paste-back, complete render function, and public API acceptance

**Files:**
- Modify: `rust/crates/feathertalk-inference/src/frame.rs`
- Modify: `rust/crates/feathertalk-inference/src/error.rs`
- Modify: `rust/crates/feathertalk-inference/src/lib.rs`
- Modify: `rust/crates/feathertalk-inference/tests/frame.rs`
- Create: `rust/crates/feathertalk-inference/tests/frame_public_api.rs`

**Interfaces:**
- Produces `paste_bgr(&mut BgrFrame, &BgrFrame, x: i32, y: i32) -> Result<(), InferenceError>`.
- Produces `render_frame(&BgrFrame, &FaceBoundingBox, &[f32], &RenderGeometry) -> Result<BgrFrame, InferenceError>`.
- Exports all frame types/functions from `feathertalk_inference` crate root.
- Produces `PasteOutOfBounds` error.

- [ ] **Step 1: Add failing paste/render tests and root smoke test.**

```rust
use feathertalk_inference::{paste_bgr, render_frame};

#[test]
fn paste_copies_rows_and_rejects_negative_or_outside_origins() {
    let mut destination = BgrFrame::new(3, 2, vec![0; 18]).unwrap();
    let source = BgrFrame::new(2, 1, vec![1, 2, 3, 4, 5, 6]).unwrap();
    paste_bgr(&mut destination, &source, 1, 1).unwrap();
    assert_eq!(destination.as_bytes(), &[0,0,0, 0,0,0, 0,0,0, 0,0,0, 1,2,3, 4,5,6]);
    assert!(matches!(paste_bgr(&mut destination, &source, -1, 0), Err(InferenceError::PasteOutOfBounds { .. })));
    assert!(matches!(paste_bgr(&mut destination, &source, 2, 1), Err(InferenceError::PasteOutOfBounds { .. })));
}

#[test]
fn render_frame_returns_new_frame_and_leaves_input_unchanged() {
    let geometry = RenderGeometry::standard();
    let frame = BgrFrame::new(2, 2, vec![10; 12]).unwrap();
    let original = frame.clone();
    let bbox = FaceBoundingBox { xmin: 0, ymin: 0, xmax: 2, ymax: 2 };
    let prediction = vec![1.0; 3 * 160 * 160];
    let rendered = render_frame(&frame, &bbox, &prediction, &geometry).unwrap();
    assert_eq!(frame, original);
    assert_eq!(rendered.as_bytes(), &[255; 12]);
}
```

Create `tests/frame_public_api.rs` that imports `BgrFrame`, `UnetImageInput`, `RenderGeometry`, `crop_bgr`, `resize_bilinear`, `build_unet_image_input`, `apply_unet_prediction`, `paste_bgr`, and `render_frame` only from `feathertalk_inference`, then compiles a small call to each. Assert `build_unet_image_input(...).unwrap().shape() == [1,6,160,160]`.

- [ ] **Step 2: Run paste/render tests and confirm RED.**

```powershell
cargo test -p feathertalk-inference --test frame paste -- --nocapture
cargo test -p feathertalk-inference --test frame render_frame -- --nocapture
```

Expected: compilation fails because paste/render functions and `PasteOutOfBounds` are absent.

- [ ] **Step 3: Implement bounds-checked paste.**

Reject negative origins. For non-negative origins, require `x + source.width <= destination.width` and the equivalent y condition using subtraction checks rather than overflowing addition. Copy each source row’s exact BGR byte range into the destination.

- [ ] **Step 4: Implement the complete side-effect-free render composition.**

Validate the bbox with the same helper used by crop, crop the source, resize to `geometry.crop_size()` in both dimensions, apply the prediction, resize the processed crop back to the bbox width/height, clone the original frame, paste the resized crop at `(xmin,ymin)`, and return the clone. Do not mutate or allocate any file/process resources.

- [ ] **Step 5: Export and run focused acceptance tests.**

```powershell
cargo test -p feathertalk-inference --test frame --test frame_public_api
cargo fmt --all -- --check
cargo clippy -p feathertalk-inference --all-targets --all-features -- -D warnings
git diff --check
```

Expected: all frame and public API tests pass, formatting and clippy are clean, and `git diff --check` reports no whitespace errors.

- [ ] **Step 6: Commit the complete frame kernel.**

```powershell
git add rust/crates/feathertalk-inference/src/error.rs rust/crates/feathertalk-inference/src/lib.rs rust/crates/feathertalk-inference/src/frame.rs rust/crates/feathertalk-inference/tests/frame.rs rust/crates/feathertalk-inference/tests/frame_public_api.rs
git commit -m "feat: add pure Rust inference frame kernel"
```

### Task 5: Crate and workspace verification

**Files:**
- No new production files; inspect only the explicit files above.

- [ ] **Step 1: Run the full crate verification from `rust/`.**

```powershell
cargo fmt --all -- --check
cargo test -p feathertalk-inference --all-targets
cargo check -p feathertalk-inference --all-targets
cargo clippy -p feathertalk-inference --all-targets --all-features -- -D warnings
git diff --check
```

- [ ] **Step 2: Run the full workspace verification.**

```powershell
cargo test --workspace --all-targets
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
git diff --check
```

Expected: every command exits with code 0; ignored tests remain ignored only where the existing workspace already marks them ignored.

- [ ] **Step 3: Inspect the diff and protected-directory status.**

```powershell
git diff --stat HEAD~4..HEAD
git status --short --branch
```

Confirm the only untracked path is the user-owned `demo/kanghui_training_video_featherhubert_188_latest/` directory, and it is not staged or included in any commit.

- [ ] **Step 4: Mark the slice complete and immediately prepare the next design.**

Re-read `docs/superpowers/specs/2026-08-26-bgr-frame-kernel-design.md` and the migration design. The next slice is the Burn inference adapter (FeatherHuBERT and Original/MobileOne UNet tensor boundary); it must consume `RenderPlan`, `BgrFrame`, `UnetImageInput`, and `apply_unet_prediction` without redefining pixel or frame rules.

## Plan Self-Review

- Spec coverage: the tasks cover every API and behavior in the design: validated BGR storage, crop, C++ resize, fixed image input/mask, prediction validation and write-back, paste-back, side-effect-free composition, root exports, and full verification.
- TDD coverage: each production group has an explicit failing test command before implementation and a focused green command afterward.
- Placeholder scan: no unfinished marker or undefined implementation step is used.
- Type consistency: all functions consume the existing `FaceBoundingBox` and `RenderGeometry`; `UnetImageInput::as_slice()` is the exact later-adapter boundary; error variant names match every test.
- Scope: no model loading, image decoding, real demo assets, FFmpeg execution, atomic output installation, or CLI work is included.
