# Preprocess Geometry Contract Design

Date: 2026-08-20  
Status: Approved for implementation planning

## 1. Purpose

Create an independent Rust crate for deterministic preprocessing geometry and landmark/audio-window contracts used by FeatherTalk. This slice ports the parts of the existing Python preprocessing behavior that do not require media decoding, image libraries, model inference, or tensor allocation.

The crate provides strict `.lms` parsing, face bounding-box calculation, fixed crop/mask constants, and bounded audio-window index calculation. Later SCRFD, PFLD, image, and FeatherHuBERT adapters consume these validated values.

## 2. Scope

Included:

- Strict UTF-8 `.lms` parsing.
- Exactly 110 non-empty landmark points, matching the Python PFLD output.
- Finite, non-negative coordinate validation.
- Face bbox calculation using the existing point indices.
- Fixed crop, inner-region, border, and mouth-mask geometry.
- Eight-frame audio-window index calculation with explicit boundary slots.
- Read-only validated types and stable structured errors.

Excluded:

- Video/audio decoding or FFmpeg.
- OpenCV, image encoding, resizing, masking, or pixel buffers.
- SCRFD or PFLD model execution.
- FeatherHuBERT feature extraction or tensor reshape.
- Project/asset manifest writes.
- GPU, Burn, WGPU, GPUI, and worker RPC.

## 3. Crate Boundary

Add:

```text
rust/crates/feathertalk-preprocess/
  Cargo.toml
  src/
    lib.rs
    error.rs
    landmarks.rs
    geometry.rs
    audio_window.rs
  tests/
```

Runtime dependencies are only the Rust standard library and `thiserror`; `tempfile` is test-only. The crate must not depend on `feathertalk-media`, `feathertalk-project`, model crates, image/OpenCV crates, Burn, WGPU, GPUI, or FFmpeg.

## 4. Landmark Model and Parsing

```rust
pub struct Point {
    pub x: f32,
    pub y: f32,
}

pub struct Landmarks {
    points: Vec<Point>,
}

impl Landmarks {
    pub fn points(&self) -> &[Point];
}

pub fn read_landmarks(path: &Path) -> Result<Landmarks, PreprocessError>;
```

Parsing rules:

- Read the file as UTF-8.
- Ignore blank or whitespace-only lines.
- Every non-empty line must contain exactly two float tokens.
- Reject malformed or extra tokens.
- Reject `NaN`, positive infinity, and negative infinity.
- Reject negative coordinates.
- Require exactly 110 points.
- Preserve source line numbers in line-specific errors.

`Landmarks` cannot be created with a public mutable point vector. A later model adapter may use `points()` for read-only access.

## 5. Face Geometry

```rust
pub struct FaceBoundingBox {
    pub xmin: i32,
    pub ymin: i32,
    pub xmax: i32,
    pub ymax: i32,
}

pub struct MaskRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

pub struct CropSpec {
    pub crop_size: u32,
    pub inner_size: u32,
    pub border: u32,
    pub mouth_mask: MaskRect,
}

pub fn compute_face_bbox(landmarks: &Landmarks) -> Result<FaceBoundingBox, PreprocessError>;
pub fn default_crop_spec() -> CropSpec;
```

`compute_face_bbox` preserves the existing Python rule:

```text
xmin = int(point[1].x)
ymin = int(point[52].y)
xmax = int(point[31].x)
width = xmax - xmin
ymax = ymin + width
```

The integer conversion truncates toward zero, matching Rust's finite `f32 as i32` behavior for the validated non-negative coordinates. The function rejects `xmax <= xmin` and any resulting non-positive square.

`default_crop_spec()` returns exactly:

```text
crop_size = 168
inner_size = 160
border = 4
mouth_mask = { x: 5, y: 5, width: 150, height: 145 }
```

It also enforces the invariant `crop_size == inner_size + 2 * border`.

All geometry structs expose read-only accessors or public scalar fields only where construction is harmless; no pixel data is owned by this crate.

## 6. Audio Window Contract

```rust
pub fn audio_window_indices(
    frame_index: usize,
    frame_count: usize,
) -> Result<[Option<usize>; 8], PreprocessError>;
```

Rules:

- `frame_count == 0` rejects every request.
- `frame_index >= frame_count` returns `FrameIndexOutOfRange`.
- Slot `0` corresponds to `frame_index - 4`.
- Slot `7` corresponds to `frame_index + 3`.
- An in-range source frame returns `Some(index)`.
- A boundary position outside `0..frame_count` returns `None`.

This function does not allocate feature tensors, copy data, pad values, or reshape `[2, 1024]` features. Consumers apply the returned slots to their own feature representation.

## 7. Error Model

```rust
pub enum PreprocessError {
    Io { operation, path, source },
    InvalidUtf8 { path },
    InvalidLine { path, line, message },
    WrongLandmarkCount { path, expected, actual },
    NonFiniteCoordinate { path, line },
    NegativeCoordinate { path, line },
    InvalidGeometry { field, message },
    FrameIndexOutOfRange { frame_index, frame_count },
}
```

Errors retain paths and source I/O errors for diagnostics while exposing stable categories for callers. No parser panic or raw library error is part of the public contract.

## 8. Tests and Acceptance

Tests use temporary files and crate-root imports. They cover:

- Valid 110-point parsing and whitespace handling.
- Missing file, invalid UTF-8, malformed line, extra token, wrong point count, non-finite coordinate, and negative coordinate errors.
- Exact bbox calculation and invalid geometry rejection.
- Exact default crop constants and border invariant.
- First, middle, and last audio windows, including `None` boundary slots.
- Empty frame count and out-of-range frame index errors.
- Read-only public accessors.

Acceptance commands:

```powershell
cargo fmt --check
cargo clippy -p feathertalk-preprocess --all-targets --all-features -- -D warnings
cargo test -p feathertalk-preprocess --all-targets
git diff --check
```

This crate completes only the deterministic preprocessing geometry contract. It does not complete SCRFD, PFLD, exception-frame handling, or milestone two.
