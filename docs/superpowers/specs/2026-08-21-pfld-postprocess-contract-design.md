# PFLD Numeric Postprocess Contract Design

Date: 2026-08-21  
Status: Approved for implementation

## Purpose

Define the pure-Rust numerical boundary between PFLD model output and FeatherTalk's 110-point `.lms` landmarks. The contract reproduces the existing Python postprocess without executing a model or manipulating image buffers.

## Scope

Included:

- Validate model output and `mean_face` vectors of exactly 220 values.
- Add corresponding normalized values element-wise.
- Reshape pairs into 110 points.
- Scale normalized coordinates by the detector crop width and height.
- Truncate each scaled coordinate toward zero like Python's `astype(np.int32)` for finite in-range values.
- Add the crop's original-image integer offset.
- Return read-only 110-point landmark values.

Excluded:

- PFLD model execution, Burn integration, or checkpoint conversion.
- Image decoding, resizing, crop creation, or padding.
- 110-to-68 landmark mapping.
- `.lms` file persistence; the existing `feathertalk-preprocess` parser owns file validation.

## Public API

```rust
pub const PFLD_OUTPUT_VALUE_COUNT: usize = 220;
pub const PFLD_LANDMARK_COUNT: usize = 110;

pub struct CropGeometry {
    pub width: u32,
    pub height: u32,
    pub offset_x: i32,
    pub offset_y: i32,
}

pub struct LandmarkPoint {
    pub x: i32,
    pub y: i32,
}

pub struct PFLDLandmarks {
    points: Vec<LandmarkPoint>,
}

pub fn decode_landmarks(
    model_output: &[f32],
    mean_face: &[f32],
    crop: CropGeometry,
) -> Result<PFLDLandmarks, PfldError>;
```

`PFLDLandmarks::points()` returns `&[LandmarkPoint]`. No public mutable point vector is exposed.

## Numerical Rules

For each point pair `i`:

```text
normalized_x = model_output[2*i] + mean_face[2*i]
normalized_y = model_output[2*i+1] + mean_face[2*i+1]
crop_x = normalized_x * crop.width
crop_y = normalized_y * crop.height
x = trunc_toward_zero(crop_x) + crop.offset_x
y = trunc_toward_zero(crop_y) + crop.offset_y
```

All model and mean-face values must be finite. Crop width and height must be non-zero. Scaled coordinates must be finite and within the representable `i32` truncation range; invalid values return structured errors instead of relying on casts.

## Errors and Tests

`PfldError` reports invalid vector lengths, non-finite values, invalid crop dimensions, and coordinates that cannot be represented as `i32`. Tests cover exact scaling, addition, negative offsets, truncation toward zero, all-zero vectors, length mismatches, non-finite inputs, zero crop dimensions, and out-of-range coordinates.

Acceptance commands:

```powershell
cargo fmt --check
cargo clippy -p feathertalk-pfld --all-targets --all-features -- -D warnings
cargo test -p feathertalk-pfld --all-targets
git diff --check
```
