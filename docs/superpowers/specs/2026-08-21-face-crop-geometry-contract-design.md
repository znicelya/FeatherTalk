# Face Crop Geometry Contract Design

Date: 2026-08-21  
Status: Approved for implementation

## Purpose

Define the pure-Rust integer geometry used between SCRFD detections and PFLD crop processing. This contract reproduces the existing Python `face_det` square crop calculations without reading images, adding borders to pixel buffers, or executing a model.

## Scope

Included:

- Convert a finite SCRFD `[x, y, width, height]` bbox into Python-compatible integer edges.
- Expand the larger bbox dimension by `1.05` and construct a centered square.
- Intersect the requested square with the source image.
- Compute left/top/right/bottom padding needed to restore the requested square.
- Expose the requested square, source intersection, padding, crop size, and original-image origin.

Excluded:

- Image decoding, resizing, pixel copying, or OpenCV border operations.
- SCRFD model execution and PFLD model execution.
- Landmark decoding and `.lms` persistence.

## Public API

```rust
pub struct RectI {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

pub struct Padding {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
}

pub struct FaceCropGeometry {
    pub requested: RectI,
    pub source: RectI,
    pub padding: Padding,
    pub size: u32,
    pub origin_x: i32,
    pub origin_y: i32,
}

pub fn compute_face_crop_geometry(
    image: ImageSize,
    bbox: [f32; 4],
) -> Result<FaceCropGeometry, FaceError>;
```

`requested` is the centered square before clipping. `source` is the in-image rectangle copied before padding. `origin_x` and `origin_y` are the requested square's top-left coordinates, matching the offset used to map PFLD crop landmarks back to the source image.

## Numerical Rules

The implementation follows Python:

```text
x1 = int(bbox.x)
y1 = int(bbox.y)
x2 = int(bbox.x + bbox.width)
y2 = int(bbox.y + bbox.height)
w = x2 - x1
h = y2 - y1
cx = (x1 + x2) // 2
cy = (y1 + y2) // 2
size = int(max(w, h) * 1.05)
requested = [cx - size//2, cy - size//2, size, size]
```

The source rectangle is clipped to `[0, image.width) x [0, image.height)`. Padding is the exact missing amount on each side, and the final padded crop is always `size x size`.

## Errors and Tests

`FaceError` gains stable categories for invalid image size, non-finite bbox, invalid bbox geometry, integer overflow, and crop geometry overflow. Tests cover normal, wide, tall, left/top overflow, right/bottom overflow, all-side overflow, zero image, non-finite bbox, and non-positive bbox cases.

Acceptance commands:

```powershell
cargo fmt --check
cargo clippy -p feathertalk-face --all-targets --all-features -- -D warnings
cargo test -p feathertalk-face --all-targets
git diff --check
```
