# SCRFD Postprocess Contract Design

Date: 2026-08-20  
Status: Approved for implementation planning

## 1. Purpose

Create an independent pure-Rust crate for the deterministic numerical postprocessing used by FeatherTalk's SCRFD face detector. The crate computes the model resize transform, anchor centers, bbox and five-point keypoint decoding, source-image coordinate mapping, score filtering, and deterministic non-maximum suppression.

The crate does not load or execute a model and does not manipulate image pixels. A later Burn/ONNX adapter supplies flat numerical outputs to this contract.

## 2. Scope

Included:

- Aspect-preserving resize and zero-padding geometry for a fixed `640x640` model input.
- Feature strides `[8, 16, 32]`.
- Two anchors per feature location.
- SCRFD distance-to-bbox and distance-to-keypoint decoding.
- Mapping decoded values from model coordinates to original-image coordinates.
- Source-image boundary clipping.
- Confidence filtering at `0.1`.
- Deterministic NMS at IoU threshold `0.5`.
- Strict slice-length and finite-value validation.

Excluded:

- Image loading, resizing, normalization, channel order, or blob creation.
- SCRFD ONNX parsing or Burn model execution.
- GPU/WGPU devices.
- PFLD inference and 68-point landmarks.
- Frame anomaly classification and asset manifest updates.

## 3. Crate Boundary

Add:

```text
rust/crates/feathertalk-face/
  Cargo.toml
  src/
    lib.rs
    error.rs
    preprocess.rs
    decode.rs
    nms.rs
  tests/
```

Runtime dependencies are only the standard library and `thiserror`. The crate does not depend on OpenCV, image crates, ONNX Runtime, Burn, WGPU, GPUI, `feathertalk-preprocess`, or `feathertalk-models`.

## 4. Fixed Detector Contract

Schema-one SCRFD postprocessing uses:

```text
model width:             640
model height:            640
feature strides:         [8, 16, 32]
anchors per location:    2
confidence threshold:    0.1
NMS IoU threshold:       0.5
keypoints per detection: 5
```

These values are exposed through constructors/defaults but alternate model layouts are not accepted by the schema-one high-level path.

## 5. Public Types

```rust
pub struct ImageSize {
    pub width: u32,
    pub height: u32,
}

pub struct ResizeTransform {
    pub input: ImageSize,
    pub model: ImageSize,
    pub new_width: u32,
    pub new_height: u32,
    pub pad_x: u32,
    pub pad_y: u32,
    pub scale_x: f32,
    pub scale_y: f32,
}

pub struct Detection {
    pub bbox: [f32; 4],
    pub score: f32,
    pub keypoints: [[f32; 2]; 5],
}

pub struct DetectionConfig {
    pub confidence_threshold: f32,
    pub nms_iou_threshold: f32,
}
```

`Detection::bbox` is `[x, y, width, height]` in original-image coordinates. All types are value types with no internal mutable state.

`DetectionConfig::default()` returns exactly `0.1` and `0.5`. Non-finite thresholds or thresholds outside `[0,1]` are rejected by NMS.

## 6. Resize and Padding

```rust
pub fn resize_with_padding(input: ImageSize) -> Result<ResizeTransform, FaceError>;
```

Zero width or height is rejected. The model input is fixed to `640x640`.

The calculation preserves the current Python behavior:

- Square input becomes `640x640` with zero padding.
- Portrait input uses `new_height = 640`, `new_width = floor(640 / (height / width))`.
- Landscape input uses `new_width = 640`, `new_height = floor(640 * (height / width)) + 1`.
- Horizontal or vertical total padding is split with `floor(total / 2)` on the left/top and the remainder on the right/bottom.
- `pad_x` and `pad_y` store only the left/top padding used when mapping coordinates back.
- `scale_x = input.width / new_width` and `scale_y = input.height / new_height`.

Every computed dimension and scale must be finite and positive.

## 7. Anchor Centers

```rust
pub fn generate_anchor_centers(
    model: ImageSize,
    stride: u32,
    anchors_per_location: u32,
) -> Result<Vec<[f32; 2]>, FaceError>;
```

The schema-one caller supplies `640x640`, one of strides `8`, `16`, or `32`, and `2` anchors. Invalid values are rejected.

Ordering matches:

```python
np.stack(np.mgrid[:height, :width][::-1], axis=-1)
```

That is row-major order by `y`, then `x`, with each `[x * stride, y * stride]` center repeated twice consecutively.

Expected anchor counts are:

```text
stride 8:  80 * 80 * 2 = 12,800
stride 16: 40 * 40 * 2 = 3,200
stride 32: 20 * 20 * 2 = 800
```

## 8. Level Decoding

```rust
pub fn decode_level(
    level: usize,
    stride: u32,
    anchors: &[[f32; 2]],
    scores: &[f32],
    bbox_distances: &[[f32; 4]],
    keypoint_distances: &[[f32; 10]],
    transform: &ResizeTransform,
) -> Result<Vec<Detection>, FaceError>;
```

All four input slices must have identical lengths. Every score and distance must be finite. Distances are multiplied by `stride` during decoding, matching the Python output path.

For anchor `[cx, cy]` and bbox distance `[left, top, right, bottom]`:

```text
x1 = cx - left * stride
y1 = cy - top * stride
x2 = cx + right * stride
y2 = cy + bottom * stride
```

For each keypoint pair `[dx, dy]`:

```text
kx = cx + dx * stride
ky = cy + dy * stride
```

Model coordinates are mapped back with:

```text
source_x = (model_x - pad_x) * scale_x
source_y = (model_y - pad_y) * scale_y
```

Coordinates are clipped to source boundaries. The bbox is returned as `[x1, y1, x2 - x1, y2 - y1]`. Non-positive area after clipping is rejected. `decode_level` returns every decoded detection; confidence filtering belongs to NMS so callers retain one deterministic filtering location.

## 9. Non-Maximum Suppression

```rust
pub fn non_max_suppression(
    detections: &[Detection],
    config: &DetectionConfig,
) -> Result<Vec<usize>, FaceError>;
```

NMS applies these rules:

1. Reject invalid config values.
2. Reject non-finite scores or bbox values and non-positive bbox area.
3. Remove detections with `score < confidence_threshold`; equality is retained.
4. Sort candidates by score descending, then original input index ascending.
5. Keep the highest-priority candidate and suppress later candidates whose IoU is strictly greater than `nms_iou_threshold`.
6. Return original input indices in deterministic keep order.

IoU uses continuous floating-point rectangle areas with `[x, y, width, height]`; no inclusive `+1` pixel convention is applied.

## 10. Errors

```rust
pub enum FaceError {
    InvalidImageSize,
    InvalidConfiguration { field, message },
    InvalidTensorLength { level, field, expected, actual },
    NonFiniteValue { level, field, index },
    InvalidDetectionGeometry { index },
}
```

Configuration errors cover invalid model size, stride, anchor count, or NMS thresholds. Tensor-length errors identify the level and field. No public API panics for caller-provided numerical input.

## 11. Tests and Acceptance

Tests cover:

- Square, portrait, and landscape transforms, including odd-padding allocation.
- Invalid zero image sizes.
- Anchor count and exact first/last ordering for strides `8`, `16`, and `32`.
- Invalid model size, stride, and anchor count.
- Exact bbox and five-keypoint decoding.
- Padding removal, scale mapping, and boundary clipping.
- Tensor-length mismatch and non-finite numerical values.
- Invalid detection geometry.
- Confidence filtering and overlapping/non-overlapping NMS cases.
- Threshold equality and stable index ordering for equal scores.
- Invalid NMS configuration and numerical input.
- Crate-root public API usage.

Acceptance commands:

```powershell
cargo fmt --check
cargo clippy -p feathertalk-face --all-targets --all-features -- -D warnings
cargo test -p feathertalk-face --all-targets
git diff --check
```

This completes only the SCRFD numerical postprocess contract. It does not complete SCRFD model inference, PFLD, frame anomaly detection, or milestone two.
