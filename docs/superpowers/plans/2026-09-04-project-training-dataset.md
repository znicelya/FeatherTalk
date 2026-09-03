# Project Training Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a locked FeatherTalk project directory (frames, landmarks, FeatherHuBERT features) into the exact tensors the existing `feathertalk-training` losses already consume: `[N,6,160,160]` image input, `[N,16,32,32]` audio window, `[N,3,160,160]` target and `[N,1,160,160]` mouth mask, plus the temporal-pair shapes `[2N,6,160,160]`, `[2N,16,32,32]`, `[N,2,3,160,160]` and `[N,2,1,160,160]`.

**Architecture:** A new leaf crate `feathertalk-training-data` sits between the on-disk asset crates (`feathertalk-project`, `feathertalk-audio`) and `feathertalk-training`. It implements the existing `TrainingDataset` trait and reuses the inference tensor bridges instead of duplicating pixel arithmetic. Tasks 1-4 expose the four reusable pieces the dataset needs (mouth ROI rectangle, a single inner plane set, a face-crop helper, a plan-free audio window); tasks 5-9 build the crate on top of them; task 10 runs the workspace gates.

**Tech Stack:** Rust edition 2024 (rust-version 1.94), burn `=0.21.0` with the NdArray backend in tests, thiserror 2.0, tempfile 3.20 for fixtures.

**Design:** docs/superpowers/specs/2026-09-04-project-training-dataset-design.md

## Global Constraints

- Run every `cargo`, `rustfmt` and `clippy` command from `E:\workspace\github\FeatherTalk\rust`; run every `git` command from `E:\workspace\github\FeatherTalk`.
- Do not change `feathertalk-training`, the worker, the CLI or the wire protocol: this slice only adds one crate plus additive helpers in `feathertalk-preprocess` and `feathertalk-inference`.
- The new crate's dependencies are exactly `burn`, `feathertalk-audio`, `feathertalk-inference`, `feathertalk-preprocess`, `feathertalk-project`, `feathertalk-training`, `thiserror`, with `tempfile` as the only dev-dependency.
- Fixed numbers: face crop 168, inner crop 160, border 4, 6 input channels, 3 output channels, `FEATURE_DIMS` 1024, `TOKENS_PER_FRAME` 2, audio window `16 * 32 * 32`, mouth ROI defaults `start 90`, `end 110`, `expand_x 1.45`, `expand_y 1.75`, `min_w 52`, `min_h 36`, `pad 2`.
- Every float-to-integer conversion of an ROI extent uses `f32::round_ties_even`; `f32::round` is wrong and one test pins the difference.
- No `unwrap`, `expect`, `panic!` or panicking indexing outside `#[cfg(test)]` code and `tests/`; every fallible step returns a typed error.
- Chinese text only in user-facing string literals. Everything added here reaches the user through `TrainingError`, whose messages are English, so identifiers, comments and error strings stay English.
- rustfmt defaults apply (`max_width` 100, `fn_call_width` 60 measured over the argument list alone, `struct_lit_width` 18, `chain_width` 60). Run `rustfmt --edition 2024 --check` on every file a step touched and apply the reported diff verbatim.
- clippy runs with `-D warnings` on 1.94: inline format args, `.enumerate()` instead of manual counters, `Option::is_none_or` instead of `map_or(true, ..)`.
- Stage explicit paths only. Never stage binary media and never stage `demo/`.
- Commit at the end of every task with the message the task gives. Do not push.

## File Structure

```
rust/Cargo.toml                                               workspace members gain feathertalk-training-data
rust/Cargo.lock                                               lock entry for the new crate
rust/crates/feathertalk-preprocess/src/geometry.rs            + MouthRoiSpec, default_mouth_roi_spec, mouth_roi_rect
rust/crates/feathertalk-preprocess/src/lib.rs                 + the three new geometry exports
rust/crates/feathertalk-preprocess/tests/geometry.rs          + mouth ROI vectors and spec rejections
rust/crates/feathertalk-inference/src/frame.rs                + MouthMasking, InnerImagePlanes, build_inner_image_planes, build_face_crop
rust/crates/feathertalk-inference/src/burn.rs                 + build_unet_audio_window, feature_frame_count; uses build_face_crop
rust/crates/feathertalk-inference/src/lib.rs                  + the new frame and burn exports
rust/crates/feathertalk-inference/tests/frame_tensor.rs       + keep/blackout plane behaviour
rust/crates/feathertalk-inference/tests/frame_crop_resize.rs  + build_face_crop equivalence
rust/crates/feathertalk-inference/tests/burn_audio.rs         + plan-free audio window behaviour
rust/crates/feathertalk-training-data/Cargo.toml              new crate manifest
rust/crates/feathertalk-training-data/src/lib.rs              module wiring and the public surface
rust/crates/feathertalk-training-data/src/error.rs            TrainingDataError and its mapping into TrainingError
rust/crates/feathertalk-training-data/src/dataset.rs          ProjectTrainingDataset, FrameSample, TrainingItem
rust/crates/feathertalk-training-data/src/batch.rs            SingleFrameBatch, TemporalBatch and the stacking functions
rust/crates/feathertalk-training-data/tests/support/mod.rs    project fixtures and the deterministic frame reader
rust/crates/feathertalk-training-data/tests/error_mapping.rs  error messages and TrainingError mapping
rust/crates/feathertalk-training-data/tests/dataset.rs        single-frame and temporal sample loading
rust/crates/feathertalk-training-data/tests/batch.rs          batch stacking shapes and ordering
rust/crates/feathertalk-training-data/tests/real_frames.rs    a project whose frames are real JPEG files
```

---

### Task 1: Mouth ROI rectangle in `feathertalk-preprocess`

**Files:**

- Modify: `rust/crates/feathertalk-preprocess/src/geometry.rs`
- Modify: `rust/crates/feathertalk-preprocess/src/lib.rs`
- Test: `rust/crates/feathertalk-preprocess/tests/geometry.rs`

**Interfaces:**

- Consumes: `Landmarks::points() -> &[Point]`, `compute_face_bbox(&Landmarks) -> Result<FaceBoundingBox, PreprocessError>`, `CropSpec`, `PreprocessError::InvalidGeometry { field: &'static str, message: String }`.
- Produces: `pub struct MouthRoiSpec { pub start: usize, pub end: usize, pub expand_x: f32, pub expand_y: f32, pub min_w: u32, pub min_h: u32, pub pad: u32 }`, `pub fn default_mouth_roi_spec() -> MouthRoiSpec`, `pub fn mouth_roi_rect(landmarks: &Landmarks, crop: &CropSpec, spec: &MouthRoiSpec) -> Result<MaskRect, PreprocessError>`.

**Why first:** The mouth mask plane is the only piece of dataset geometry that exists nowhere in the Rust tree yet, and it is pure arithmetic over data the preprocess crate already owns. Landing it first means the new crate never has to reach for landmark internals, and the rounding rule (`round_ties_even`) gets pinned by a test in the crate that owns it rather than in the consumer.

- [ ] **Step 1: Write the failing test**

Rewrite the import block at the top of `rust/crates/feathertalk-preprocess/tests/geometry.rs`:

```rust
use feathertalk_preprocess::{
    CropSpec, FaceBoundingBox, Landmarks, MaskRect, MouthRoiSpec, PFLD_LANDMARK_COUNT,
    PreprocessError, compute_face_bbox, default_crop_spec, default_mouth_roi_spec, mouth_roi_rect,
    read_landmarks,
};
```

Keep `landmarks_file` as it is and append these helpers and tests to the same file:

```rust
fn mouth_landmarks_file(
    x1: f32,
    x31: f32,
    y52: f32,
    mouth: &[(f32, f32)],
) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("face.lms");
    let mut lines = (0..PFLD_LANDMARK_COUNT)
        .map(|_| "0 0".to_owned())
        .collect::<Vec<_>>();
    lines[1] = format!("{x1} 0");
    lines[31] = format!("{x31} 0");
    lines[52] = format!("0 {y52}");
    for (offset, (x, y)) in mouth.iter().enumerate() {
        lines[90 + offset] = format!("{x} {y}");
    }
    std::fs::write(&path, lines.join("\n")).unwrap();
    (dir, path)
}

fn roi(path: &std::path::Path) -> MaskRect {
    let landmarks = read_landmarks(path).unwrap();
    mouth_roi_rect(&landmarks, &default_crop_spec(), &default_mouth_roi_spec()).unwrap()
}

fn rejection_field(landmarks: &Landmarks, crop: &CropSpec, spec: &MouthRoiSpec) -> &'static str {
    match mouth_roi_rect(landmarks, crop, spec) {
        Err(PreprocessError::InvalidGeometry { field, .. }) => field,
        other => panic!("expected a geometry rejection, got {other:?}"),
    }
}

#[test]
fn default_mouth_roi_spec_matches_python_constants() {
    let spec = default_mouth_roi_spec();
    assert_eq!((spec.start, spec.end), (90, 110));
    assert_eq!((spec.expand_x, spec.expand_y), (1.45, 1.75));
    assert_eq!((spec.min_w, spec.min_h, spec.pad), (52, 36, 2));
}

#[test]
fn mouth_roi_projects_landmarks_into_the_inner_crop() {
    let mut mouth = vec![(120.0, 170.0); 20];
    mouth[0] = (100.0, 160.0);
    mouth[19] = (140.0, 180.0);
    let (_dir, path) = mouth_landmarks_file(40.0, 200.0, 60.0, &mouth);
    assert_eq!(
        roi(&path),
        MaskRect {
            x: 47,
            y: 90,
            width: 66,
            height: 43
        }
    );
}

#[test]
fn mouth_roi_truncates_landmark_coordinates_like_python() {
    let mut mouth = vec![(120.9, 170.9); 20];
    mouth[0] = (100.9, 160.9);
    mouth[19] = (140.9, 180.9);
    let (_dir, path) = mouth_landmarks_file(40.9, 200.9, 60.9, &mouth);
    assert_eq!(
        roi(&path),
        MaskRect {
            x: 47,
            y: 90,
            width: 66,
            height: 43
        }
    );
}

#[test]
fn mouth_roi_rounds_half_to_even_instead_of_half_away_from_zero() {
    let mut mouth = vec![(120.0, 160.0); 20];
    mouth[0] = (114.0, 154.0);
    mouth[19] = (135.0, 174.0);
    let (_dir, path) = mouth_landmarks_file(40.0, 208.0, 60.0, &mouth);
    assert_eq!(
        roi(&path),
        MaskRect {
            x: 54,
            y: 79,
            width: 52,
            height: 42
        }
    );
}

#[test]
fn mouth_roi_grows_a_degenerate_span_to_the_minimum_extents() {
    let (_dir, path) = mouth_landmarks_file(40.0, 208.0, 60.0, &[(114.0, 154.0); 20]);
    assert_eq!(
        roi(&path),
        MaskRect {
            x: 44,
            y: 72,
            width: 52,
            height: 36
        }
    );
}

#[test]
fn mouth_roi_clamps_to_the_left_and_right_edges() {
    let mut left = vec![(49.0, 164.0); 20];
    left[0] = (44.0, 164.0);
    let (_left_dir, left_path) = mouth_landmarks_file(40.0, 208.0, 60.0, &left);
    assert_eq!(
        roi(&left_path),
        MaskRect {
            x: 0,
            y: 82,
            width: 28,
            height: 36
        }
    );
    let mut right = vec![(199.0, 164.0); 20];
    right[19] = (203.0, 164.0);
    let (_right_dir, right_path) = mouth_landmarks_file(40.0, 208.0, 60.0, &right);
    assert_eq!(
        roi(&right_path),
        MaskRect {
            x: 131,
            y: 82,
            width: 29,
            height: 36
        }
    );
}

#[test]
fn mouth_roi_keeps_a_one_pixel_rectangle_for_landmarks_outside_the_crop() {
    let (_dir, path) = mouth_landmarks_file(40.0, 208.0, 60.0, &[(344.0, 344.0); 20]);
    assert_eq!(
        roi(&path),
        MaskRect {
            x: 159,
            y: 159,
            width: 1,
            height: 1
        }
    );
}

#[test]
fn mouth_roi_rejects_inconsistent_specs() {
    let (_dir, path) = mouth_landmarks_file(40.0, 208.0, 60.0, &[(120.0, 160.0); 20]);
    let landmarks = read_landmarks(&path).unwrap();
    let crop = default_crop_spec();
    let spec = default_mouth_roi_spec();
    assert_eq!(
        rejection_field(&landmarks, &crop, &MouthRoiSpec { start: 110, ..spec }),
        "mouth_roi_range"
    );
    assert_eq!(
        rejection_field(&landmarks, &crop, &MouthRoiSpec { end: 111, ..spec }),
        "mouth_roi_range"
    );
    assert_eq!(
        rejection_field(
            &landmarks,
            &crop,
            &MouthRoiSpec {
                expand_x: 0.0,
                ..spec
            }
        ),
        "mouth_roi_expand"
    );
    assert_eq!(
        rejection_field(
            &landmarks,
            &crop,
            &MouthRoiSpec {
                expand_y: f32::NAN,
                ..spec
            }
        ),
        "mouth_roi_expand"
    );
    assert_eq!(
        rejection_field(&landmarks, &crop, &MouthRoiSpec { min_w: 0, ..spec }),
        "mouth_roi_min_size"
    );
    assert_eq!(
        rejection_field(&landmarks, &crop, &MouthRoiSpec { min_h: 161, ..spec }),
        "mouth_roi_min_size"
    );
    assert_eq!(
        rejection_field(
            &landmarks,
            &CropSpec {
                inner_size: 0,
                ..crop
            },
            &spec
        ),
        "inner_size"
    );
    assert_eq!(
        rejection_field(
            &landmarks,
            &CropSpec {
                crop_size: 0,
                ..crop
            },
            &spec
        ),
        "mouth_roi_projection"
    );
}
```

The expected rectangles come from the design's projection formula. For the first two tests the bbox is `xmin 40`, `xmax 200`, so `scale = 168 / 160 = 1.05`; the mouth x span projects to `59..101`, giving `center 80`, `size = (42 + 4) * 1.45 = 66.7`, hence `x 47` and `width 66`, and the y span `101..122` gives `center 111.5`, `size = 25 * 1.75 = 43.75`, hence `y 90` and `height 43`. The rounding test uses `xmax 208` so `scale` is exactly `1.0`; the x extent lands on `54.5` and `106.5`, which `round_ties_even` maps to `54` and `106` (`f32::round` would give `55` and `107`).

- [ ] **Step 2: Run test to verify it fails**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-preprocess --test geometry
```

Expected failure: `error[E0432]: unresolved imports feathertalk_preprocess::MouthRoiSpec, feathertalk_preprocess::default_mouth_roi_spec, feathertalk_preprocess::mouth_roi_rect`. The test binary must not compile.

- [ ] **Step 3: Write minimal implementation**

Append to `rust/crates/feathertalk-preprocess/src/geometry.rs` (leave `compute_face_bbox` and `default_crop_spec` untouched):

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MouthRoiSpec {
    pub start: usize,
    pub end: usize,
    pub expand_x: f32,
    pub expand_y: f32,
    pub min_w: u32,
    pub min_h: u32,
    pub pad: u32,
}

pub fn default_mouth_roi_spec() -> MouthRoiSpec {
    MouthRoiSpec {
        start: 90,
        end: 110,
        expand_x: 1.45,
        expand_y: 1.75,
        min_w: 52,
        min_h: 36,
        pad: 2,
    }
}

pub fn mouth_roi_rect(
    landmarks: &Landmarks,
    crop: &CropSpec,
    spec: &MouthRoiSpec,
) -> Result<MaskRect, PreprocessError> {
    if spec.start >= spec.end {
        return Err(invalid_geometry(
            "mouth_roi_range",
            "start must be smaller than end",
        ));
    }
    let points = landmarks.points();
    if spec.end > points.len() {
        return Err(invalid_geometry(
            "mouth_roi_range",
            "end exceeds the landmark count",
        ));
    }
    if !spec.expand_x.is_finite()
        || spec.expand_x <= 0.0
        || !spec.expand_y.is_finite()
        || spec.expand_y <= 0.0
    {
        return Err(invalid_geometry(
            "mouth_roi_expand",
            "expansion factors must be finite and positive",
        ));
    }
    if crop.inner_size == 0 {
        return Err(invalid_geometry(
            "inner_size",
            "inner crop size must be positive",
        ));
    }
    if spec.min_w == 0
        || spec.min_h == 0
        || spec.min_w > crop.inner_size
        || spec.min_h > crop.inner_size
    {
        return Err(invalid_geometry(
            "mouth_roi_min_size",
            "minimum extents must fit inside the inner crop",
        ));
    }
    let bbox = compute_face_bbox(landmarks)?;
    let scale = crop.crop_size as f32 / (bbox.xmax - bbox.xmin) as f32;
    if !scale.is_finite() || scale <= 0.0 {
        return Err(invalid_geometry(
            "mouth_roi_projection",
            "crop scale must be finite and positive",
        ));
    }
    let border = crop.border as f32;
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for point in &points[spec.start..spec.end] {
        let x = (point.x.trunc() - bbox.xmin as f32) * scale - border;
        let y = (point.y.trunc() - bbox.ymin as f32) * scale - border;
        if !x.is_finite() || !y.is_finite() {
            return Err(invalid_geometry(
                "mouth_roi_projection",
                "projected landmark is not finite",
            ));
        }
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    let (x, width) = extent(
        min_x,
        max_x,
        spec.pad,
        spec.expand_x,
        spec.min_w,
        crop.inner_size,
    );
    let (y, height) = extent(
        min_y,
        max_y,
        spec.pad,
        spec.expand_y,
        spec.min_h,
        crop.inner_size,
    );
    Ok(MaskRect {
        x,
        y,
        width,
        height,
    })
}

fn extent(
    low: f32,
    high: f32,
    pad: u32,
    expand: f32,
    min_extent: u32,
    inner_size: u32,
) -> (u32, u32) {
    let center = (low + high) / 2.0;
    let size = ((high - low + 2.0 * pad as f32) * expand).max(min_extent as f32);
    let start = (center - size / 2.0).round_ties_even() as i64;
    let end = (center + size / 2.0).round_ties_even() as i64;
    let inner = i64::from(inner_size);
    let start = start.clamp(0, inner - 1);
    let end = (start + 1).max(end.min(inner));
    (start as u32, (end - start) as u32)
}

fn invalid_geometry(field: &'static str, message: impl Into<String>) -> PreprocessError {
    PreprocessError::InvalidGeometry {
        field,
        message: message.into(),
    }
}
```

Replace the geometry export line in `rust/crates/feathertalk-preprocess/src/lib.rs`:

```rust
pub use geometry::{
    CropSpec, FaceBoundingBox, MaskRect, MouthRoiSpec, compute_face_bbox, default_crop_spec,
    default_mouth_roi_spec, mouth_roi_rect,
};
```

Note that the validation order is load-bearing: `inner_size` is checked before the minimum extents (so a zero inner crop reports `inner_size`, not `mouth_roi_min_size`), and a zero `crop_size` falls through to the scale check and reports `mouth_roi_projection`. `MouthRoiSpec` derives `PartialEq` but not `Eq` because two fields are `f32`.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-preprocess --test geometry
rustfmt --edition 2024 --check crates/feathertalk-preprocess/src/geometry.rs crates/feathertalk-preprocess/src/lib.rs crates/feathertalk-preprocess/tests/geometry.rs
cargo clippy -p feathertalk-preprocess --all-targets -- -D warnings
```

All ten tests in the binary pass, rustfmt reports no diff, clippy is clean.

- [ ] **Step 5: Commit**

```powershell
cd E:\workspace\github\FeatherTalk
git add rust/crates/feathertalk-preprocess/src/geometry.rs rust/crates/feathertalk-preprocess/src/lib.rs rust/crates/feathertalk-preprocess/tests/geometry.rs
git commit -m "feat(preprocess): compute the mouth ROI rectangle"
```

---

### Task 2: One inner image plane set in `feathertalk-inference`

**Files:**

- Modify: `rust/crates/feathertalk-inference/src/frame.rs`
- Modify: `rust/crates/feathertalk-inference/src/lib.rs`
- Test: `rust/crates/feathertalk-inference/tests/frame_tensor.rs`

**Interfaces:**

- Consumes: the private helpers `validate_geometry_and_crop`, `checked_elements`, `linear_offset`, `allocate_f32`, `BgrFrame::pixel_offset_checked`, and `feathertalk_preprocess::default_crop_spec`.
- Produces: `pub enum MouthMasking { Keep, Blackout }`, `pub struct InnerImagePlanes` with `shape() -> [usize; 4]`, `as_slice() -> &[f32]` and `into_values() -> Vec<f32>`, and `pub fn build_inner_image_planes(face_crop: &BgrFrame, geometry: &RenderGeometry, masking: MouthMasking) -> Result<InnerImagePlanes, InferenceError>`.
- Preserves: `build_unet_image_input(face_crop, geometry)` keeps its signature and output bytes; it becomes `Keep` planes followed by `Blackout` planes.

**Why now:** Training needs the three plane sets separately (reference keep, target blackout, target keep as the loss target), while inference only ever needed the concatenated six. Factoring the loop out first means the dataset composes plane sets instead of re-deriving normalisation and border arithmetic, and the existing inference test keeps guarding the byte layout.

- [ ] **Step 1: Write the failing test**

Replace the import block at the top of `rust/crates/feathertalk-inference/tests/frame_tensor.rs`:

```rust
use feathertalk_inference::{
    BgrFrame, InferenceError, MouthMasking, RenderGeometry, apply_unet_prediction,
    build_inner_image_planes, build_unet_image_input,
};
```

Append to the same file, leaving the four existing tests untouched:

```rust
#[test]
fn inner_planes_blackout_only_the_mouth_rectangle() {
    let mut bytes = vec![64; 168 * 168 * 3];
    let pixel = (4 + 6) * 168 * 3 + (4 + 6) * 3;
    bytes[pixel..pixel + 3].copy_from_slice(&[128, 0, 255]);
    let unmasked_pixel = (4_u32 * 168 + 4) as usize * 3;
    bytes[unmasked_pixel..unmasked_pixel + 3].copy_from_slice(&[32, 64, 96]);
    let crop = BgrFrame::new(168, 168, bytes).unwrap();
    let geometry = RenderGeometry::standard();
    let keep = build_inner_image_planes(&crop, &geometry, MouthMasking::Keep).unwrap();
    let blackout = build_inner_image_planes(&crop, &geometry, MouthMasking::Blackout).unwrap();
    assert_eq!(keep.shape(), [1, 3, 160, 160]);
    assert_eq!(blackout.shape(), [1, 3, 160, 160]);
    assert_eq!(keep.as_slice().len(), 3 * 160 * 160);
    let plane = 160 * 160;
    let offset = 6 * 160 + 6;
    assert_eq!(keep.as_slice()[offset], 128.0 / 255.0);
    assert_eq!(keep.as_slice()[plane + offset], 0.0);
    assert_eq!(keep.as_slice()[2 * plane + offset], 1.0);
    assert_eq!(blackout.as_slice()[offset], 0.0);
    assert_eq!(blackout.as_slice()[plane + offset], 0.0);
    assert_eq!(blackout.as_slice()[2 * plane + offset], 0.0);
    assert_eq!(keep.as_slice()[0], 32.0 / 255.0);
    assert_eq!(keep.as_slice()[plane], 64.0 / 255.0);
    assert_eq!(keep.as_slice()[2 * plane], 96.0 / 255.0);
    assert_eq!(blackout.as_slice()[0], 32.0 / 255.0);
    assert_eq!(blackout.as_slice()[plane], 64.0 / 255.0);
    assert_eq!(blackout.as_slice()[2 * plane], 96.0 / 255.0);
}

#[test]
fn image_input_is_keep_planes_followed_by_blackout_planes() {
    let bytes: Vec<u8> = (0..168 * 168 * 3).map(|index| (index % 251) as u8).collect();
    let crop = BgrFrame::new(168, 168, bytes).unwrap();
    let geometry = RenderGeometry::standard();
    let input = build_unet_image_input(&crop, &geometry).unwrap();
    let keep = build_inner_image_planes(&crop, &geometry, MouthMasking::Keep).unwrap();
    let blackout = build_inner_image_planes(&crop, &geometry, MouthMasking::Blackout).unwrap();
    let half = 3 * 160 * 160;
    assert_eq!(input.as_slice().len(), 2 * half);
    assert_eq!(&input.as_slice()[..half], keep.as_slice());
    assert_eq!(&input.as_slice()[half..], blackout.as_slice());
}

#[test]
fn inner_planes_reject_wrong_crop_dimensions() {
    let crop = BgrFrame::new(10, 10, vec![0; 300]).unwrap();
    let geometry = RenderGeometry::standard();
    for masking in [MouthMasking::Keep, MouthMasking::Blackout] {
        assert!(matches!(
            build_inner_image_planes(&crop, &geometry, masking),
            Err(InferenceError::TensorShapeMismatch {
                context: "face_crop",
                ..
            })
        ));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-inference --test frame_tensor
```

Expected failure: `error[E0432]: unresolved imports feathertalk_inference::MouthMasking, feathertalk_inference::build_inner_image_planes`.

- [ ] **Step 3: Write minimal implementation**

In `rust/crates/feathertalk-inference/src/frame.rs`, insert the new type and builder before `build_unet_image_input` and replace that function's body:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouthMasking {
    Keep,
    Blackout,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InnerImagePlanes {
    values: Vec<f32>,
}

impl InnerImagePlanes {
    pub fn shape(&self) -> [usize; 4] {
        [1, UNET_OUTPUT_CHANNELS, 160, 160]
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    pub fn into_values(self) -> Vec<f32> {
        self.values
    }
}

pub fn build_inner_image_planes(
    face_crop: &BgrFrame,
    geometry: &crate::RenderGeometry,
    masking: MouthMasking,
) -> Result<InnerImagePlanes, InferenceError> {
    let (crop_size, inner_size, border) = validate_geometry_and_crop(face_crop, geometry)?;
    let crop_spec = feathertalk_preprocess::default_crop_spec();
    let mask_right = crop_spec
        .mouth_mask
        .x
        .checked_add(crop_spec.mouth_mask.width)
        .ok_or(InferenceError::ArithmeticOverflow)?;
    let mask_bottom = crop_spec
        .mouth_mask
        .y
        .checked_add(crop_spec.mouth_mask.height)
        .ok_or(InferenceError::ArithmeticOverflow)?;
    if mask_right > inner_size || mask_bottom > inner_size {
        return Err(InferenceError::InvalidField {
            field: "mouth_mask",
            message: "mask rectangle exceeds the inner crop".into(),
        });
    }
    let plane = checked_elements(inner_size, inner_size)?;
    let elements = plane
        .checked_mul(UNET_OUTPUT_CHANNELS)
        .ok_or(InferenceError::ArithmeticOverflow)?;
    let mut values = allocate_f32(elements)?;
    for y in 0..inner_size {
        for x in 0..inner_size {
            let source_x = x
                .checked_add(border)
                .ok_or(InferenceError::ArithmeticOverflow)?;
            let source_y = y
                .checked_add(border)
                .ok_or(InferenceError::ArithmeticOverflow)?;
            let source_offset = face_crop.pixel_offset_checked(source_x, source_y)?;
            let offset = linear_offset(inner_size, x, y)?;
            let masked = masking == MouthMasking::Blackout
                && x >= crop_spec.mouth_mask.x
                && x < mask_right
                && y >= crop_spec.mouth_mask.y
                && y < mask_bottom;
            for channel in 0..UNET_OUTPUT_CHANNELS {
                let normalized = f32::from(face_crop.bgr[source_offset + channel]) / 255.0;
                values[channel * plane + offset] = if masked { 0.0 } else { normalized };
            }
        }
    }
    debug_assert_eq!(crop_size, inner_size + 2 * border);
    Ok(InnerImagePlanes { values })
}

pub fn build_unet_image_input(
    face_crop: &BgrFrame,
    geometry: &crate::RenderGeometry,
) -> Result<UnetImageInput, InferenceError> {
    let keep = build_inner_image_planes(face_crop, geometry, MouthMasking::Keep)?;
    let blackout = build_inner_image_planes(face_crop, geometry, MouthMasking::Blackout)?;
    let mut values = keep.into_values();
    let mut tail = blackout.into_values();
    let total = values
        .len()
        .checked_add(tail.len())
        .ok_or(InferenceError::ArithmeticOverflow)?;
    values
        .try_reserve_exact(tail.len())
        .map_err(|_| InferenceError::AllocationFailure {
            bytes: total.saturating_mul(std::mem::size_of::<f32>()),
        })?;
    values.append(&mut tail);
    Ok(UnetImageInput { values })
}
```

Update the `frame` export line in `rust/crates/feathertalk-inference/src/lib.rs`:

```rust
pub use frame::{
    BgrFrame, InnerImagePlanes, MouthMasking, UnetImageInput, apply_unet_prediction,
    build_inner_image_planes, build_unet_image_input, crop_bgr, paste_bgr, render_frame,
    resize_bilinear,
};
```

The mask-bounds check stays in the shared builder so `MouthMasking::Keep` rejects an oversized mask exactly like `Blackout` does; the design treats that rectangle as a geometry invariant, not a masking detail.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-inference --test frame_tensor --test frame_public_api
rustfmt --edition 2024 --check crates/feathertalk-inference/src/frame.rs crates/feathertalk-inference/src/lib.rs crates/feathertalk-inference/tests/frame_tensor.rs
cargo clippy -p feathertalk-inference --all-targets -- -D warnings
```

`image_input_is_bgr_channel_first_and_masks_only_the_mouth_rectangle` must still pass unchanged: it is the regression net proving the concatenated layout did not move.

- [ ] **Step 5: Commit**

```powershell
cd E:\workspace\github\FeatherTalk
git add rust/crates/feathertalk-inference/src/frame.rs rust/crates/feathertalk-inference/src/lib.rs rust/crates/feathertalk-inference/tests/frame_tensor.rs
git commit -m "feat(inference): build one inner image plane set"
```

---

### Task 3: A single `build_face_crop` helper

**Files:**

- Modify: `rust/crates/feathertalk-inference/src/frame.rs`
- Modify: `rust/crates/feathertalk-inference/src/burn.rs`
- Modify: `rust/crates/feathertalk-inference/src/lib.rs`
- Test: `rust/crates/feathertalk-inference/tests/frame_crop_resize.rs`

**Interfaces:**

- Consumes: `crop_bgr(&BgrFrame, &FaceBoundingBox)`, `resize_bilinear(&BgrFrame, u32, u32)`, `RenderGeometry::crop_size()`.
- Produces: `pub fn build_face_crop(frame: &BgrFrame, bbox: &FaceBoundingBox, geometry: &RenderGeometry) -> Result<BgrFrame, InferenceError>`.
- Replaces: the crop-then-resize pair inside `render_frame` (frame.rs) and inside `render_planned_frame` (burn.rs).

**Why now:** The dataset needs the same 168x168 crop for every frame it loads, and the sequence "crop to the bbox, then bilinear-resize to `crop_size`" is already duplicated in two places. Naming it once removes the chance that the training path resizes differently from the inference path, which would silently shift every pixel the model sees.

- [ ] **Step 1: Write the failing test**

Replace the import block at the top of `rust/crates/feathertalk-inference/tests/frame_crop_resize.rs`:

```rust
use feathertalk_inference::{
    BgrFrame, InferenceError, RenderGeometry, build_face_crop, crop_bgr, resize_bilinear,
};
use feathertalk_preprocess::FaceBoundingBox;
```

Append to the same file:

```rust
#[test]
fn face_crop_is_the_bbox_crop_resized_to_the_geometry() {
    let bytes: Vec<u8> = (0..64 * 64 * 3).map(|index| (index % 253) as u8).collect();
    let frame = BgrFrame::new(64, 64, bytes).unwrap();
    let bbox = FaceBoundingBox {
        xmin: 4,
        ymin: 6,
        xmax: 44,
        ymax: 46,
    };
    let geometry = RenderGeometry::standard();
    let face_crop = build_face_crop(&frame, &bbox, &geometry).unwrap();
    let source = crop_bgr(&frame, &bbox).unwrap();
    let expected = resize_bilinear(&source, 168, 168).unwrap();
    assert_eq!((face_crop.width(), face_crop.height()), (168, 168));
    assert_eq!(face_crop, expected);
}

#[test]
fn face_crop_rejects_a_bbox_outside_the_frame() {
    let frame = BgrFrame::new(8, 8, vec![0; 8 * 8 * 3]).unwrap();
    let bbox = FaceBoundingBox {
        xmin: 0,
        ymin: 0,
        xmax: 9,
        ymax: 8,
    };
    assert!(matches!(
        build_face_crop(&frame, &bbox, &RenderGeometry::standard()),
        Err(InferenceError::InvalidBbox { .. })
    ));
}
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-inference --test frame_crop_resize
```

Expected failure: `error[E0432]: unresolved import feathertalk_inference::build_face_crop`.

- [ ] **Step 3: Write minimal implementation**

Add to `rust/crates/feathertalk-inference/src/frame.rs`, next to `render_frame`:

```rust
pub fn build_face_crop(
    frame: &BgrFrame,
    bbox: &FaceBoundingBox,
    geometry: &crate::RenderGeometry,
) -> Result<BgrFrame, InferenceError> {
    let source_crop = crop_bgr(frame, bbox)?;
    resize_bilinear(&source_crop, geometry.crop_size(), geometry.crop_size())
}
```

Replace the first two lines of `render_frame`'s body with:

```rust
    let mut face_crop = build_face_crop(frame, bbox, geometry)?;
```

In `rust/crates/feathertalk-inference/src/burn.rs`, replace the import block and the crop lines of `render_planned_frame`:

```rust
use crate::{
    BgrFrame, InferenceError, InferenceFramePlan, RenderGeometry, build_face_crop,
    build_unet_image_input, render_frame,
};
```

```rust
    let audio = build_unet_audio_input(features, plan)?;
    let face_crop = build_face_crop(frame, bbox, geometry)?;
    let image = build_unet_image_input(&face_crop, geometry)?;
```

Add `build_face_crop` to the `frame` export list in `rust/crates/feathertalk-inference/src/lib.rs`:

```rust
pub use frame::{
    BgrFrame, InnerImagePlanes, MouthMasking, UnetImageInput, apply_unet_prediction,
    build_face_crop, build_inner_image_planes, build_unet_image_input, crop_bgr, paste_bgr,
    render_frame, resize_bilinear,
};
```

`burn.rs` no longer uses `crop_bgr` or `crate::resize_bilinear`, so both must leave its import block or the unused-import lint fails the clippy gate.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-inference --test frame_crop_resize --test frame_public_api
rustfmt --edition 2024 --check crates/feathertalk-inference/src/frame.rs crates/feathertalk-inference/src/burn.rs crates/feathertalk-inference/src/lib.rs crates/feathertalk-inference/tests/frame_crop_resize.rs
cargo clippy -p feathertalk-inference --all-targets -- -D warnings
```

- [ ] **Step 5: Commit**

```powershell
cd E:\workspace\github\FeatherTalk
git add rust/crates/feathertalk-inference/src/frame.rs rust/crates/feathertalk-inference/src/burn.rs rust/crates/feathertalk-inference/src/lib.rs rust/crates/feathertalk-inference/tests/frame_crop_resize.rs
git commit -m "refactor(inference): build the face crop in one place"
```

---

### Task 4: An audio window without a render plan

**Files:**

- Modify: `rust/crates/feathertalk-inference/src/burn.rs`
- Modify: `rust/crates/feathertalk-inference/src/lib.rs`
- Test: `rust/crates/feathertalk-inference/tests/burn_audio.rs`

**Interfaces:**

- Consumes: `FeatureMatrix::{tokens, dims, values}`, the module constants `FEATURE_DIMS`, `TOKENS_PER_FRAME`, `AUDIO_VALUES_PER_SLOT`, `UNET_AUDIO_VALUES`.
- Produces: `pub fn build_unet_audio_window(features: &FeatureMatrix, audio_window: &[Option<usize>; 8]) -> Result<UnetAudioInput, InferenceError>` and the private `fn feature_frame_count(features: &FeatureMatrix) -> Result<usize, InferenceError>`.
- Preserves: `build_unet_audio_input(features, plan)` keeps its signature, its `OutputFrameOutOfRange` check and its output bytes.

**Why now:** Training samples have no `InferenceFramePlan`: there is no output frame, no source frame and no reference frame index to fake. The dataset only has a frame index and the window that `audio_window_indices` returns, so the plan-free entry point is the last inference-side piece the new crate needs. Keeping `build_unet_audio_input` as a thin wrapper means the inference path's contract is unchanged.

- [ ] **Step 1: Write the failing test**

Replace the import block at the top of `rust/crates/feathertalk-inference/tests/burn_audio.rs`:

```rust
use feathertalk_audio::FeatureMatrix;
use feathertalk_inference::{
    InferenceError, InferenceFramePlan, build_unet_audio_input, build_unet_audio_window,
};
```

Append to the same file:

```rust
#[test]
fn plan_free_audio_window_matches_the_planned_window() {
    let matrix = features(3);
    let audio_window = [None, None, Some(0), Some(1), Some(2), None, None, None];
    let plan = InferenceFramePlan {
        output_index: 1,
        source_frame_index: 0,
        reference_frame_index: 0,
        audio_window,
    };
    let planned = build_unet_audio_input(&matrix, &plan).unwrap();
    let direct = build_unet_audio_window(&matrix, &audio_window).unwrap();
    assert_eq!(direct.shape(), [1, 16, 32, 32]);
    assert_eq!(direct.as_slice(), planned.as_slice());
}

#[test]
fn plan_free_audio_window_rejects_slots_and_feature_shapes() {
    let matrix = features(2);
    let window = [None, None, None, None, Some(2), None, None, None];
    assert!(matches!(
        build_unet_audio_window(&matrix, &window),
        Err(InferenceError::InvalidAudioWindowIndex {
            slot: 4,
            index: 2,
            frame_count: 2
        })
    ));
    let odd = FeatureMatrix::new(3, 1024, vec![0.0; 3 * 1024]).unwrap();
    assert!(matches!(
        build_unet_audio_window(&odd, &[None; 8]),
        Err(InferenceError::InvalidFeatureShape {
            tokens: 3,
            dims: 1024
        })
    ));
}
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-inference --test burn_audio
```

Expected failure: `error[E0432]: unresolved import feathertalk_inference::build_unet_audio_window`.

- [ ] **Step 3: Write minimal implementation**

In `rust/crates/feathertalk-inference/src/burn.rs`, replace `build_unet_audio_input` with these three items (the slot loop moves verbatim into the new function):

```rust
pub fn build_unet_audio_window(
    features: &FeatureMatrix,
    audio_window: &[Option<usize>; 8],
) -> Result<UnetAudioInput, InferenceError> {
    let frame_count = feature_frame_count(features)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(UNET_AUDIO_VALUES)
        .map_err(|_| InferenceError::AllocationFailure {
            bytes: UNET_AUDIO_VALUES * std::mem::size_of::<f32>(),
        })?;
    values.resize(UNET_AUDIO_VALUES, 0.0);

    for (slot, frame_index) in audio_window.iter().copied().enumerate() {
        let Some(frame_index) = frame_index else {
            continue;
        };
        if frame_index >= frame_count {
            return Err(InferenceError::InvalidAudioWindowIndex {
                slot,
                index: frame_index,
                frame_count,
            });
        }

        let source_start = frame_index
            .checked_mul(AUDIO_VALUES_PER_SLOT)
            .ok_or(InferenceError::ArithmeticOverflow)?;
        let destination_start = slot
            .checked_mul(AUDIO_VALUES_PER_SLOT)
            .ok_or(InferenceError::ArithmeticOverflow)?;
        let source_end = source_start
            .checked_add(AUDIO_VALUES_PER_SLOT)
            .ok_or(InferenceError::ArithmeticOverflow)?;
        let destination_end = destination_start
            .checked_add(AUDIO_VALUES_PER_SLOT)
            .ok_or(InferenceError::ArithmeticOverflow)?;
        values[destination_start..destination_end]
            .copy_from_slice(&features.values()[source_start..source_end]);
    }

    Ok(UnetAudioInput { values })
}

pub fn build_unet_audio_input(
    features: &FeatureMatrix,
    plan: &InferenceFramePlan,
) -> Result<UnetAudioInput, InferenceError> {
    let frame_count = feature_frame_count(features)?;
    if plan.output_index >= frame_count {
        return Err(InferenceError::OutputFrameOutOfRange {
            index: plan.output_index,
            count: frame_count,
        });
    }
    build_unet_audio_window(features, &plan.audio_window)
}

fn feature_frame_count(features: &FeatureMatrix) -> Result<usize, InferenceError> {
    let tokens = features.tokens();
    let dims = features.dims();
    if dims != FEATURE_DIMS || tokens == 0 || !tokens.is_multiple_of(TOKENS_PER_FRAME) {
        return Err(InferenceError::InvalidFeatureShape { tokens, dims });
    }
    Ok(tokens / TOKENS_PER_FRAME)
}
```

Update the `burn` export line in `rust/crates/feathertalk-inference/src/lib.rs`:

```rust
pub use burn::{
    UnetAudioInput, build_unet_audio_input, build_unet_audio_window, render_planned_frame,
    run_unet_prediction,
};
```

The feature-shape check must run before the `output_index` check so an invalid matrix still reports `InvalidFeatureShape` rather than `OutputFrameOutOfRange`; the existing test `audio_window_rejects_invalid_feature_matrix_contracts` pins that order.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-inference --test burn_audio --test frame_public_api
rustfmt --edition 2024 --check crates/feathertalk-inference/src/burn.rs crates/feathertalk-inference/src/lib.rs crates/feathertalk-inference/tests/burn_audio.rs
cargo clippy -p feathertalk-inference --all-targets -- -D warnings
```

- [ ] **Step 5: Commit**

```powershell
cd E:\workspace\github\FeatherTalk
git add rust/crates/feathertalk-inference/src/burn.rs rust/crates/feathertalk-inference/src/lib.rs rust/crates/feathertalk-inference/tests/burn_audio.rs
git commit -m "feat(inference): build an audio window without a render plan"
```

---

### Task 5: The `feathertalk-training-data` crate and its error surface

**Files:**

- Modify: `rust/Cargo.toml`
- Modify: `rust/Cargo.lock`
- Create: `rust/crates/feathertalk-training-data/Cargo.toml`
- Create: `rust/crates/feathertalk-training-data/src/lib.rs`
- Create: `rust/crates/feathertalk-training-data/src/error.rs`
- Test: `rust/crates/feathertalk-training-data/tests/error_mapping.rs`

**Interfaces:**

- Consumes: `feathertalk_training::TrainingError::InvalidInput(String)`.
- Produces: `pub enum TrainingDataError` with the variants `Project`, `Features`, `FeatureShape`, `FrameIndexOutOfRange`, `Frame`, `Landmarks`, `Sample`, `Batch`, plus `impl From<TrainingDataError> for TrainingError`.

**Why now:** Every later task returns one of these variants, so the error surface has to exist before the dataset does. Defining the mapping into `TrainingError` first also settles the crate's boundary: the trait `feathertalk-training` exposes only knows `TrainingError`, so the dataset's typed errors must degrade into `InvalidInput` with a message that still names the file and the frame.

- [ ] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-training-data/tests/error_mapping.rs`:

```rust
use std::path::PathBuf;

use feathertalk_training::TrainingError;
use feathertalk_training_data::TrainingDataError;

fn cases() -> Vec<TrainingDataError> {
    vec![
        TrainingDataError::Project {
            path: PathBuf::from("project"),
            message: "asset package is not locked".to_owned(),
        },
        TrainingDataError::Features {
            path: PathBuf::from("project/assets/features/feather_hubert.f32"),
            message: "unexpected end of file".to_owned(),
        },
        TrainingDataError::FeatureShape {
            path: PathBuf::from("project/assets/features/feather_hubert.f32"),
            expected_tokens: 24,
            actual_tokens: 22,
            dims: 1024,
        },
        TrainingDataError::FrameIndexOutOfRange {
            index: 12,
            frame_count: 12,
        },
        TrainingDataError::Frame {
            index: 3,
            path: PathBuf::from("project/assets/frames/000003.jpg"),
            message: "not a file".to_owned(),
        },
        TrainingDataError::Landmarks {
            index: 3,
            path: PathBuf::from("project/assets/landmarks/000003.lms"),
            message: "wrong landmark count".to_owned(),
        },
        TrainingDataError::Sample {
            index: 3,
            message: "inner planes rejected the crop".to_owned(),
        },
        TrainingDataError::Batch {
            message: "batch is empty".to_owned(),
        },
    ]
}

#[test]
fn feature_shape_names_the_file_and_both_token_counts() {
    let message = cases()
        .into_iter()
        .map(|case| case.to_string())
        .find(|message| message.contains("feather_hubert.f32") && message.contains("token"))
        .unwrap();
    assert!(message.contains("24"), "{message}");
    assert!(message.contains("22"), "{message}");
    assert!(message.contains("1024"), "{message}");
}

#[test]
fn frame_and_landmark_errors_name_the_frame_index_and_path() {
    for case in cases() {
        let message = case.to_string();
        match case {
            TrainingDataError::Frame { index, path, .. }
            | TrainingDataError::Landmarks { index, path, .. } => {
                assert!(message.contains(&index.to_string()), "{message}");
                let name = path.file_name().unwrap().to_string_lossy().into_owned();
                assert!(message.contains(&name), "{message}");
            }
            _ => {}
        }
    }
}

#[test]
fn every_variant_maps_to_invalid_training_input() {
    let mut count = 0;
    for case in cases() {
        let expected = case.to_string();
        match TrainingError::from(case) {
            TrainingError::InvalidInput(message) => assert_eq!(message, expected),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
        count += 1;
    }
    assert_eq!(count, 8);
}
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-training-data --test error_mapping
```

Expected failure: `error: package ID specification 'feathertalk-training-data' did not match any packages`, because the crate is not a workspace member yet.

- [ ] **Step 3: Write minimal implementation**

Insert the new member in `rust/Cargo.toml` immediately after `"crates/feathertalk-export",`:

```toml
  "crates/feathertalk-training-data",
```

Create `rust/crates/feathertalk-training-data/Cargo.toml`:

```toml
[package]
name = "feathertalk-training-data"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[dependencies]
burn.workspace = true
feathertalk-audio = { path = "../feathertalk-audio" }
feathertalk-inference = { path = "../feathertalk-inference" }
feathertalk-preprocess = { path = "../feathertalk-preprocess" }
feathertalk-project = { path = "../feathertalk-project" }
feathertalk-training = { path = "../feathertalk-training" }
thiserror.workspace = true

[dev-dependencies]
tempfile.workspace = true
```

Create `rust/crates/feathertalk-training-data/src/lib.rs`:

```rust
mod error;

pub use error::TrainingDataError;
```

Create `rust/crates/feathertalk-training-data/src/error.rs`:

```rust
use std::path::PathBuf;

use feathertalk_training::TrainingError;

#[derive(Debug, thiserror::Error)]
pub enum TrainingDataError {
    #[error("invalid training project at {path}: {message}")]
    Project { path: PathBuf, message: String },
    #[error("unable to read audio features from {path}: {message}")]
    Features { path: PathBuf, message: String },
    #[error(
        "feature file {path} holds {actual_tokens} tokens of {dims} dims but the asset package declares {expected_tokens} tokens"
    )]
    FeatureShape {
        path: PathBuf,
        expected_tokens: usize,
        actual_tokens: usize,
        dims: usize,
    },
    #[error("frame index {index} is out of range for {frame_count} frames")]
    FrameIndexOutOfRange { index: u64, frame_count: u64 },
    #[error("unable to read frame {index} from {path}: {message}")]
    Frame {
        index: usize,
        path: PathBuf,
        message: String,
    },
    #[error("unable to read landmarks for frame {index} from {path}: {message}")]
    Landmarks {
        index: usize,
        path: PathBuf,
        message: String,
    },
    #[error("unable to build the training sample for frame {index}: {message}")]
    Sample { index: usize, message: String },
    #[error("unable to stack a training batch: {message}")]
    Batch { message: String },
}

impl From<TrainingDataError> for TrainingError {
    fn from(error: TrainingDataError) -> Self {
        TrainingError::InvalidInput(error.to_string())
    }
}
```

Running any cargo command from `rust/` after the member insert rewrites `rust/Cargo.lock` with the new package entry; all dependencies already resolve, so nothing is downloaded.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-training-data --test error_mapping
rustfmt --edition 2024 --check crates/feathertalk-training-data/src/lib.rs crates/feathertalk-training-data/src/error.rs crates/feathertalk-training-data/tests/error_mapping.rs
cargo clippy -p feathertalk-training-data --all-targets -- -D warnings
```

The long `#[error(..)]` string for `FeatureShape` stays on its own line inside the attribute; that is how `feathertalk-inference` formats `InvalidBbox`, and rustfmt leaves it alone.

- [ ] **Step 5: Commit**

```powershell
cd E:\workspace\github\FeatherTalk
git add rust/Cargo.toml rust/Cargo.lock rust/crates/feathertalk-training-data/Cargo.toml rust/crates/feathertalk-training-data/src/lib.rs rust/crates/feathertalk-training-data/src/error.rs rust/crates/feathertalk-training-data/tests/error_mapping.rs
git commit -m "feat(training-data): name the dataset failures"
```

---

### Task 6: Open a project and load a single-frame sample

**Files:**

- Create: `rust/crates/feathertalk-training-data/src/dataset.rs`
- Modify: `rust/crates/feathertalk-training-data/src/lib.rs`
- Test: `rust/crates/feathertalk-training-data/tests/support/mod.rs`
- Test: `rust/crates/feathertalk-training-data/tests/dataset.rs`

**Interfaces:**

- Consumes: `validate_project_dir`, `read_feature_file`, `FrameReader::read`, `read_landmarks`, `compute_face_bbox`, `audio_window_indices`, plus the four helpers tasks 1-4 added (`mouth_roi_rect`, `build_face_crop`, `build_inner_image_planes`, `build_unet_audio_window`).
- Produces: `pub struct FrameSample` with `image()`, `audio()`, `target()`, `mouth_mask()`; `pub enum TrainingItem { SingleFrame(FrameSample) }`; `pub struct ProjectTrainingDataset<R: FrameReader>` with `open`, `open_with_reader`, `root`; and `impl<R: FrameReader> TrainingDataset for ProjectTrainingDataset<R>`.
- Preserves: `feathertalk-training` is untouched. The dataset implements the trait exactly as declared, so `TrainingDataError` degrades into `TrainingError::InvalidInput` at the trait boundary and nowhere else.

**Why now:** This is the task the whole slice exists for, and the single-frame path exercises every helper the earlier tasks added. Doing it before temporal pairs and batching keeps the first version small: one sample kind, one reference frame, one target frame. The fixture module written here is also what tasks 7-9 reuse, so it has to appear with the first real test.

- [ ] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-training-data/tests/support/mod.rs`:

```rust
#![allow(dead_code)]

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use feathertalk_audio::{FeatureMatrix, write_feature_file};
use feathertalk_inference::{
    BgrFrame, FrameReader, InferenceError, MouthMasking, RenderGeometry, build_face_crop,
    build_inner_image_planes,
};
use feathertalk_preprocess::{
    Landmarks, MaskRect, PFLD_LANDMARK_COUNT, compute_face_bbox, default_crop_spec,
    default_mouth_roi_spec, mouth_roi_rect, read_landmarks,
};
use feathertalk_project::{
    AssetManifest, AssetPackageState, FeatureType, ModelSelection, ProjectManifest,
    TaskHistoryEntry, TaskHistoryStatus, lock_asset_package, write_asset_manifest_atomic,
    write_project_manifest_atomic,
};
use tempfile::TempDir;

pub const FRAME_WIDTH: u32 = 256;
pub const FRAME_HEIGHT: u32 = 256;
pub const INNER_SIZE: usize = 160;
pub const FEATURE_DIMS: usize = 1024;

/// A frame reader that synthesises a deterministic gradient instead of decoding JPEG bytes.
///
/// It still rejects a wrong file name and a missing file, so the dataset's frame-level error
/// paths stay testable while the fixture frames on disk remain five-byte placeholders.
#[derive(Debug, Clone, Copy, Default)]
pub struct GradientFrameReader;

impl FrameReader for GradientFrameReader {
    fn read(&self, index: usize, path: &Path) -> Result<BgrFrame, InferenceError> {
        let expected = format!("{index:06}.jpg");
        if path.file_name() != Some(OsStr::new(&expected)) {
            return Err(InferenceError::FrameReader {
                index,
                path: path.to_path_buf(),
                message: format!("expected a file named {expected}"),
            });
        }
        if !path.is_file() {
            return Err(InferenceError::FrameReader {
                index,
                path: path.to_path_buf(),
                message: "not a file".to_owned(),
            });
        }
        let width = FRAME_WIDTH as usize;
        let height = FRAME_HEIGHT as usize;
        let mut bgr = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                for channel in 0..3 {
                    let seed = x + y * 2 + channel * 7 + index * 3;
                    bgr.push((seed % 251) as u8);
                }
            }
        }
        BgrFrame::new(FRAME_WIDTH, FRAME_HEIGHT, bgr)
    }
}

/// Everything the fixtures vary between tests.
pub struct FixtureSpec {
    pub frame_count: usize,
    pub manifest_width: u32,
    pub manifest_height: u32,
    pub frame_bytes: Vec<u8>,
    pub face_xmin: u32,
    pub face_xmax: u32,
    pub face_ymin: u32,
    pub mouth_x: u32,
    pub mouth_y: u32,
}

impl FixtureSpec {
    /// The 256x256 gradient project the dataset tests start from.
    pub fn gradient(frame_count: usize) -> Self {
        Self {
            frame_count,
            manifest_width: FRAME_WIDTH,
            manifest_height: FRAME_HEIGHT,
            frame_bytes: b"stub\n".to_vec(),
            face_xmin: 40,
            face_xmax: 200,
            face_ymin: 60,
            mouth_x: 100,
            mouth_y: 160,
        }
    }

    pub fn manifest(&self) -> AssetManifest {
        AssetManifest {
            schema_version: 1,
            state: AssetPackageState::Locked,
            video_fps: 25,
            audio_sample_rate: 16_000,
            audio_channels: 1,
            frame_count: self.frame_count as u64,
            frame_width: self.manifest_width,
            frame_height: self.manifest_height,
            feature_type: FeatureType::FeatherHubert,
            feature_shape: [self.frame_count as u64, 2, 1024],
            landmark_model_sha256: "a".repeat(64),
            feature_model_sha256: "b".repeat(64),
        }
    }
}

pub fn locked_project(frame_count: usize) -> (TempDir, PathBuf) {
    build_locked_project(&FixtureSpec::gradient(frame_count))
}

/// Writes every required artifact and then locks the asset package.
///
/// The order matters: `write_asset_manifest_atomic` refuses to overwrite a manifest that already
/// validates as locked, so `lock_asset_package` has to run last.
pub fn build_locked_project(spec: &FixtureSpec) -> (TempDir, PathBuf) {
    let temp = tempfile::tempdir().unwrap();
    let project_dir = temp.path().join("project");
    fs::create_dir_all(project_dir.join("assets/frames")).unwrap();
    fs::create_dir_all(project_dir.join("assets/landmarks")).unwrap();
    fs::create_dir_all(project_dir.join("assets/features")).unwrap();
    fs::write(project_dir.join("assets/video_25fps.mp4"), b"video").unwrap();
    fs::write(project_dir.join("assets/audio_16k_mono.wav"), b"audio").unwrap();
    write_features(&project_dir, spec.frame_count);
    for index in 0..spec.frame_count {
        let frame_path = project_dir.join(format!("assets/frames/{index:06}.jpg"));
        fs::write(&frame_path, &spec.frame_bytes).unwrap();
        write_landmarks(&project_dir, spec, index);
    }
    let project_path = project_dir.join("project.json");
    write_project_manifest_atomic(&project_path, &valid_project()).unwrap();
    lock_asset_package(&project_dir, spec.manifest()).unwrap();
    (temp, project_dir)
}

/// Overwrites the feature file with `2 * frame_count` tokens of deterministic values.
pub fn write_features(project_dir: &Path, frame_count: usize) {
    let tokens = 2 * frame_count;
    let mut values = Vec::with_capacity(tokens * FEATURE_DIMS);
    for offset in 0..tokens * FEATURE_DIMS {
        values.push((offset % 97) as f32 / 97.0);
    }
    let matrix = FeatureMatrix::new(tokens, FEATURE_DIMS, values).unwrap();
    let path = project_dir.join("assets/features/feather_hubert.f32");
    write_feature_file(&path, &matrix).unwrap();
}

/// Writes 110 landmark lines: three of them fix the face box, twenty carry the mouth.
pub fn write_landmarks(project_dir: &Path, spec: &FixtureSpec, index: usize) {
    let mut lines = vec![String::from("0 0"); PFLD_LANDMARK_COUNT];
    lines[1] = format!("{} 0", spec.face_xmin);
    lines[31] = format!("{} 0", spec.face_xmax);
    lines[52] = format!("0 {}", spec.face_ymin);
    for (offset, line) in lines.iter_mut().skip(90).take(20).enumerate() {
        let x = spec.mouth_x + offset as u32;
        let y = spec.mouth_y + (offset + index) as u32;
        *line = format!("{x} {y}");
    }
    let path = project_dir.join(format!("assets/landmarks/{index:06}.lms"));
    fs::write(path, lines.join("\n")).unwrap();
}

/// Replaces the locked asset manifest with a preparing one.
pub fn downgrade_to_preparing(project_dir: &Path) {
    let manifest_path = project_dir.join("assets/assets.json");
    fs::remove_file(&manifest_path).unwrap();
    write_asset_manifest_atomic(&manifest_path, &preparing_manifest()).unwrap();
}

pub fn landmarks_for(project_dir: &Path, index: usize) -> Landmarks {
    let path = project_dir.join(format!("assets/landmarks/{index:06}.lms"));
    read_landmarks(&path).unwrap()
}

pub fn face_crop(project_dir: &Path, index: usize) -> BgrFrame {
    let frame_path = project_dir.join(format!("assets/frames/{index:06}.jpg"));
    let frame = GradientFrameReader.read(index, &frame_path).unwrap();
    let landmarks = landmarks_for(project_dir, index);
    let bbox = compute_face_bbox(&landmarks).unwrap();
    let geometry = RenderGeometry::standard();
    build_face_crop(&frame, &bbox, &geometry).unwrap()
}

pub fn inner_planes(project_dir: &Path, index: usize, masking: MouthMasking) -> Vec<f32> {
    let crop = face_crop(project_dir, index);
    let geometry = RenderGeometry::standard();
    let planes = build_inner_image_planes(&crop, &geometry, masking).unwrap();
    planes.into_values()
}

pub fn mouth_rect(project_dir: &Path, index: usize) -> MaskRect {
    let landmarks = landmarks_for(project_dir, index);
    let crop = default_crop_spec();
    let spec = default_mouth_roi_spec();
    mouth_roi_rect(&landmarks, &crop, &spec).unwrap()
}

pub fn preparing_manifest() -> AssetManifest {
    AssetManifest {
        schema_version: 1,
        state: AssetPackageState::Preparing,
        video_fps: 0,
        audio_sample_rate: 0,
        audio_channels: 0,
        frame_count: 0,
        frame_width: 0,
        frame_height: 0,
        feature_type: FeatureType::FeatherHubert,
        feature_shape: [0, 0, 0],
        landmark_model_sha256: String::new(),
        feature_model_sha256: String::new(),
    }
}

pub fn valid_project() -> ProjectManifest {
    ProjectManifest {
        schema_version: 1,
        project_id: "demo".into(),
        display_name: "Demo".into(),
        asset_package: "assets/assets.json".into(),
        default_model: ModelSelection::OriginalUnet,
        task_history: vec![TaskHistoryEntry {
            task_id: "task-1".into(),
            kind: "preprocess".into(),
            status: TaskHistoryStatus::Completed,
            updated_at: "2026-08-20T10:00:00Z".into(),
        }],
    }
}
```

The fixture geometry is fixed on purpose. With `face_xmin 40`, `face_xmax 200`, `face_ymin 60` the face box is `40..200` by `60..220`, entirely inside 256x256, and the crop scale is `168 / 160 = 1.05`. The mouth ROI then lands on `MaskRect { x: 43, y: 90 + index, width: 52, height: 42 }`: never clamped, and its centre sits inside the fixed blackout rectangle `{5, 5, 150, 145}`, so a blacked-out plane really differs from a kept one. Shifting the mouth y by one pixel per frame is what makes two frames' masks and targets distinguishable.

Create `rust/crates/feathertalk-training-data/tests/dataset.rs`:

```rust
mod support;

use std::fs;
use std::path::Path;

use feathertalk_inference::MouthMasking;
use feathertalk_training::{TrainingDataset, TrainingSample};
use feathertalk_training_data::{
    FrameSample, ProjectTrainingDataset, TrainingDataError, TrainingItem,
};
use support::{
    GradientFrameReader, INNER_SIZE, downgrade_to_preparing, inner_planes, locked_project,
    mouth_rect, write_features,
};

fn open_dataset(project_dir: &Path) -> ProjectTrainingDataset<GradientFrameReader> {
    ProjectTrainingDataset::open_with_reader(project_dir, GradientFrameReader).unwrap()
}

fn single_frame(item: &TrainingItem) -> &FrameSample {
    match item {
        TrainingItem::SingleFrame(sample) => sample,
    }
}

fn single_frame_sample(target_index: u64, reference_index: u64) -> TrainingSample {
    TrainingSample::SingleFrame {
        target_index,
        reference_index,
    }
}

#[test]
fn frame_count_and_root_come_from_the_locked_project() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let canonical = project_dir.canonicalize().unwrap();
    assert_eq!(dataset.frame_count(), 4);
    assert_eq!(dataset.root(), canonical.as_path());
}

#[test]
fn single_frame_image_is_the_reference_then_the_masked_target() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let item = dataset.load_sample(&single_frame_sample(2, 0)).unwrap();
    let image = single_frame(&item).image();
    let plane = INNER_SIZE * INNER_SIZE;
    let reference = inner_planes(&project_dir, 0, MouthMasking::Keep);
    let masked = inner_planes(&project_dir, 2, MouthMasking::Blackout);
    let kept = inner_planes(&project_dir, 2, MouthMasking::Keep);
    assert_eq!(&image[..3 * plane], reference.as_slice());
    assert_eq!(&image[3 * plane..], masked.as_slice());
    assert_ne!(&image[3 * plane..], kept.as_slice());
}

#[test]
fn the_masked_half_is_black_where_the_mouth_is() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let item = dataset.load_sample(&single_frame_sample(2, 0)).unwrap();
    let image = single_frame(&item).image();
    let plane = INNER_SIZE * INNER_SIZE;
    let mask = mouth_rect(&project_dir, 2);
    let centre_x = (mask.x + mask.width / 2) as usize;
    let centre_y = (mask.y + mask.height / 2) as usize;
    assert_eq!(image[3 * plane + centre_y * INNER_SIZE + centre_x], 0.0);
}

#[test]
fn the_target_is_the_unmasked_target_frame() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let item = dataset.load_sample(&single_frame_sample(2, 0)).unwrap();
    let expected = inner_planes(&project_dir, 2, MouthMasking::Keep);
    assert_eq!(single_frame(&item).target(), expected.as_slice());
}

#[test]
fn every_tensor_has_the_length_the_losses_expect() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let item = dataset.load_sample(&single_frame_sample(2, 0)).unwrap();
    let sample = single_frame(&item);
    let plane = INNER_SIZE * INNER_SIZE;
    assert_eq!(sample.image().len(), 6 * plane);
    assert_eq!(sample.audio().len(), 16 * 32 * 32);
    assert_eq!(sample.target().len(), 3 * plane);
    assert_eq!(sample.mouth_mask().len(), plane);
}

#[test]
fn the_mouth_mask_is_one_inside_the_roi_and_zero_outside() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let item = dataset.load_sample(&single_frame_sample(2, 0)).unwrap();
    let plane = single_frame(&item).mouth_mask();
    let mask = mouth_rect(&project_dir, 2);
    let ones = plane.iter().filter(|value| **value == 1.0).count();
    let inside = (mask.y as usize) * INNER_SIZE + mask.x as usize;
    assert_eq!(ones, (mask.width * mask.height) as usize);
    assert_eq!(plane[inside], 1.0);
    assert_eq!(plane[0], 0.0);
    assert!(plane.iter().all(|value| *value == 0.0 || *value == 1.0));
}

#[test]
fn the_feature_token_count_must_match_the_frame_count() {
    let (_temp, project_dir) = locked_project(4);
    write_features(&project_dir, 3);
    let error =
        ProjectTrainingDataset::open_with_reader(&project_dir, GradientFrameReader).unwrap_err();
    assert!(matches!(
        error,
        TrainingDataError::FeatureShape {
            expected_tokens: 8,
            actual_tokens: 6,
            dims: 1024,
            ..
        }
    ));
}

#[test]
fn a_project_whose_assets_are_not_locked_is_rejected() {
    let (_temp, project_dir) = locked_project(4);
    downgrade_to_preparing(&project_dir);
    let error =
        ProjectTrainingDataset::open_with_reader(&project_dir, GradientFrameReader).unwrap_err();
    assert!(matches!(error, TrainingDataError::Project { .. }));
}

#[test]
fn a_missing_frame_file_names_its_index() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    fs::remove_file(project_dir.join("assets/frames/000002.jpg")).unwrap();
    let error = dataset.load_sample(&single_frame_sample(2, 0)).unwrap_err();
    assert!(error.to_string().contains("unable to read frame 2"));
}

#[test]
fn a_missing_landmark_file_names_its_index() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    fs::remove_file(project_dir.join("assets/landmarks/000002.lms")).unwrap();
    let error = dataset.load_sample(&single_frame_sample(2, 0)).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("unable to read landmarks for frame 2"));
}

#[test]
fn a_corrupt_landmark_file_names_its_index() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let lines = vec![String::from("1 2 3"); 110];
    let path = project_dir.join("assets/landmarks/000002.lms");
    fs::write(path, lines.join("\n")).unwrap();
    let error = dataset.load_sample(&single_frame_sample(2, 0)).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("unable to read landmarks for frame 2"));
}

#[test]
fn a_frame_index_past_the_end_is_rejected() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let error = dataset.load_sample(&single_frame_sample(4, 0)).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("frame index 4 is out of range for 4 frames"));
}
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-training-data --test dataset
```

Expected failure: `error[E0432]: unresolved imports feathertalk_training_data::FrameSample, feathertalk_training_data::ProjectTrainingDataset, feathertalk_training_data::TrainingItem`. The test binary must not compile.

- [ ] **Step 3: Write minimal implementation**

Create `rust/crates/feathertalk-training-data/src/dataset.rs`:

```rust
use std::path::{Path, PathBuf};

use feathertalk_audio::{FeatureMatrix, read_feature_file};
use feathertalk_inference::{
    BgrFrame, FrameReader, JpegFrameReader, MouthMasking, RenderGeometry, build_face_crop,
    build_inner_image_planes, build_unet_audio_window,
};
use feathertalk_preprocess::{
    CropSpec, MaskRect, MouthRoiSpec, audio_window_indices, compute_face_bbox, default_crop_spec,
    default_mouth_roi_spec, mouth_roi_rect, read_landmarks,
};
use feathertalk_project::validate_project_dir;
use feathertalk_training::{TrainingDataset, TrainingError, TrainingSample};

use crate::TrainingDataError;

const FEATURE_FILE: &str = "assets/features/feather_hubert.f32";
const FEATURE_DIMS: usize = 1024;
const TOKENS_PER_FRAME: usize = 2;
const INNER_SIZE: usize = 160;

/// One training frame, flattened into the planes the losses consume.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameSample {
    image: Vec<f32>,
    audio: Vec<f32>,
    target: Vec<f32>,
    mouth_mask: Vec<f32>,
}

impl FrameSample {
    /// `[6, 160, 160]`: the reference frame's planes followed by the mouth-masked target planes.
    pub fn image(&self) -> &[f32] {
        &self.image
    }

    /// `[16, 32, 32]`: the eight-slot audio window centred on the target frame.
    pub fn audio(&self) -> &[f32] {
        &self.audio
    }

    /// `[3, 160, 160]`: the unmasked target planes.
    pub fn target(&self) -> &[f32] {
        &self.target
    }

    /// `[1, 160, 160]`: one inside the mouth ROI, zero outside it.
    pub fn mouth_mask(&self) -> &[f32] {
        &self.mouth_mask
    }
}

/// What one training sample loads into.
#[derive(Debug, Clone, PartialEq)]
pub enum TrainingItem {
    SingleFrame(FrameSample),
}

/// A locked project directory presented as a training dataset.
#[derive(Debug)]
pub struct ProjectTrainingDataset<R: FrameReader> {
    root: PathBuf,
    frame_count: usize,
    frame_width: u32,
    frame_height: u32,
    features: FeatureMatrix,
    reader: R,
    crop: CropSpec,
    mouth_roi: MouthRoiSpec,
    geometry: RenderGeometry,
}

struct LoadedFrame {
    crop: BgrFrame,
    mouth: MaskRect,
}

fn project_error(path: &Path, message: String) -> TrainingDataError {
    TrainingDataError::Project {
        path: path.to_path_buf(),
        message,
    }
}

fn features_error(path: &Path, message: String) -> TrainingDataError {
    TrainingDataError::Features {
        path: path.to_path_buf(),
        message,
    }
}

fn frame_error(index: usize, path: &Path, message: String) -> TrainingDataError {
    TrainingDataError::Frame {
        index,
        path: path.to_path_buf(),
        message,
    }
}

fn landmark_error(index: usize, path: &Path, message: String) -> TrainingDataError {
    TrainingDataError::Landmarks {
        index,
        path: path.to_path_buf(),
        message,
    }
}

fn sample_error(index: usize, message: String) -> TrainingDataError {
    TrainingDataError::Sample { index, message }
}

impl ProjectTrainingDataset<JpegFrameReader> {
    /// Opens a locked project directory and decodes its frames as JPEG files.
    pub fn open(project_dir: &Path) -> Result<Self, TrainingDataError> {
        Self::open_with_reader(project_dir, JpegFrameReader::default())
    }
}

impl<R: FrameReader> ProjectTrainingDataset<R> {
    /// Opens a locked project directory with a caller-supplied frame reader.
    pub fn open_with_reader(project_dir: &Path, reader: R) -> Result<Self, TrainingDataError> {
        let project = validate_project_dir(project_dir)
            .map_err(|error| project_error(project_dir, error.to_string()))?;
        let manifest = project.asset_package().manifest();
        if manifest.frame_count == 0 {
            return Err(project_error(
                project_dir,
                "the asset package declares zero frames".to_owned(),
            ));
        }
        let Ok(frame_count) = usize::try_from(manifest.frame_count) else {
            return Err(project_error(
                project_dir,
                format!("{} frames do not fit in memory", manifest.frame_count),
            ));
        };
        let Some(expected_tokens) = frame_count.checked_mul(TOKENS_PER_FRAME) else {
            return Err(project_error(
                project_dir,
                format!("{frame_count} frames overflow the feature token count"),
            ));
        };
        let root = project.root().to_path_buf();
        let feature_path = root.join(FEATURE_FILE);
        let features = read_feature_file(&feature_path)
            .map_err(|error| features_error(&feature_path, error.to_string()))?;
        if features.dims() != FEATURE_DIMS || features.tokens() != expected_tokens {
            return Err(TrainingDataError::FeatureShape {
                path: feature_path,
                expected_tokens,
                actual_tokens: features.tokens(),
                dims: features.dims(),
            });
        }
        Ok(Self {
            root,
            frame_count,
            frame_width: manifest.frame_width,
            frame_height: manifest.frame_height,
            features,
            reader,
            crop: default_crop_spec(),
            mouth_roi: default_mouth_roi_spec(),
            geometry: RenderGeometry::standard(),
        })
    }

    /// The canonical project root the dataset reads from.
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn resolve_index(&self, index: u64) -> Result<usize, TrainingDataError> {
        match usize::try_from(index) {
            Ok(resolved) if resolved < self.frame_count => Ok(resolved),
            _ => Err(TrainingDataError::FrameIndexOutOfRange {
                index,
                frame_count: self.frame_count as u64,
            }),
        }
    }

    fn load_frame(&self, index: usize) -> Result<LoadedFrame, TrainingDataError> {
        let frame_path = self.root.join(format!("assets/frames/{index:06}.jpg"));
        let frame = self
            .reader
            .read(index, &frame_path)
            .map_err(|error| frame_error(index, &frame_path, error.to_string()))?;
        if frame.width() != self.frame_width || frame.height() != self.frame_height {
            return Err(frame_error(
                index,
                &frame_path,
                format!(
                    "frame is {}x{} but the asset package declares {}x{}",
                    frame.width(),
                    frame.height(),
                    self.frame_width,
                    self.frame_height
                ),
            ));
        }
        let landmark_path = self.root.join(format!("assets/landmarks/{index:06}.lms"));
        let landmarks = read_landmarks(&landmark_path)
            .map_err(|error| landmark_error(index, &landmark_path, error.to_string()))?;
        let bbox = compute_face_bbox(&landmarks)
            .map_err(|error| landmark_error(index, &landmark_path, error.to_string()))?;
        let mouth = mouth_roi_rect(&landmarks, &self.crop, &self.mouth_roi)
            .map_err(|error| landmark_error(index, &landmark_path, error.to_string()))?;
        let crop = build_face_crop(&frame, &bbox, &self.geometry)
            .map_err(|error| frame_error(index, &frame_path, error.to_string()))?;
        Ok(LoadedFrame { crop, mouth })
    }

    fn inner_planes(
        &self,
        index: usize,
        face_crop: &BgrFrame,
        masking: MouthMasking,
    ) -> Result<Vec<f32>, TrainingDataError> {
        let planes = build_inner_image_planes(face_crop, &self.geometry, masking)
            .map_err(|error| sample_error(index, error.to_string()))?;
        Ok(planes.into_values())
    }

    fn audio_window(&self, index: usize) -> Result<Vec<f32>, TrainingDataError> {
        let window = audio_window_indices(index, self.frame_count)
            .map_err(|error| sample_error(index, error.to_string()))?;
        let audio = build_unet_audio_window(&self.features, &window)
            .map_err(|error| sample_error(index, error.to_string()))?;
        Ok(audio.as_slice().to_vec())
    }

    fn mouth_mask_plane(rect: &MaskRect) -> Vec<f32> {
        let mut plane = vec![0.0; INNER_SIZE * INNER_SIZE];
        let x_start = (rect.x as usize).min(INNER_SIZE);
        let y_start = (rect.y as usize).min(INNER_SIZE);
        let x_end = x_start.saturating_add(rect.width as usize).min(INNER_SIZE);
        let y_end = y_start.saturating_add(rect.height as usize).min(INNER_SIZE);
        let rows = plane
            .chunks_exact_mut(INNER_SIZE)
            .skip(y_start)
            .take(y_end - y_start);
        for row in rows {
            for value in row.iter_mut().skip(x_start).take(x_end - x_start) {
                *value = 1.0;
            }
        }
        plane
    }

    fn build_frame_sample(
        &self,
        index: usize,
        target: &LoadedFrame,
        reference: &[f32],
    ) -> Result<FrameSample, TrainingDataError> {
        let blackout = self.inner_planes(index, &target.crop, MouthMasking::Blackout)?;
        let keep = self.inner_planes(index, &target.crop, MouthMasking::Keep)?;
        let Some(elements) = reference.len().checked_add(blackout.len()) else {
            return Err(sample_error(index, "image plane count overflows".to_owned()));
        };
        let mut image = Vec::new();
        image
            .try_reserve_exact(elements)
            .map_err(|_| sample_error(index, format!("cannot allocate {elements} floats")))?;
        image.extend_from_slice(reference);
        image.extend_from_slice(&blackout);
        Ok(FrameSample {
            image,
            audio: self.audio_window(index)?,
            target: keep,
            mouth_mask: Self::mouth_mask_plane(&target.mouth),
        })
    }

    fn load_item(&self, sample: &TrainingSample) -> Result<TrainingItem, TrainingDataError> {
        match sample {
            TrainingSample::SingleFrame {
                target_index,
                reference_index,
            } => {
                let target_index = self.resolve_index(*target_index)?;
                let reference_index = self.resolve_index(*reference_index)?;
                let reference = self.load_frame(reference_index)?;
                let planes =
                    self.inner_planes(reference_index, &reference.crop, MouthMasking::Keep)?;
                let target = self.load_frame(target_index)?;
                let frame = self.build_frame_sample(target_index, &target, &planes)?;
                Ok(TrainingItem::SingleFrame(frame))
            }
            TrainingSample::TemporalPair { .. } => Err(TrainingDataError::Sample {
                index: 0,
                message: "only single-frame samples are implemented".to_owned(),
            }),
        }
    }
}

impl<R: FrameReader> TrainingDataset for ProjectTrainingDataset<R> {
    type Item = TrainingItem;

    fn frame_count(&self) -> u64 {
        self.frame_count as u64
    }

    fn load_sample(&self, sample: &TrainingSample) -> Result<Self::Item, TrainingError> {
        Ok(self.load_item(sample)?)
    }
}
```

Replace `rust/crates/feathertalk-training-data/src/lib.rs` with:

```rust
mod dataset;
mod error;

pub use dataset::{FrameSample, ProjectTrainingDataset, TrainingItem};
pub use error::TrainingDataError;
```

Three details are deliberate. The temporal arm of `load_item` returns an error instead of being absent, so `TrainingSample` stays exhaustively matched and nothing is dead code; task 7 replaces it. `mouth_mask_plane` is an associated function without an `index` parameter, because an unused parameter is a `-D warnings` failure. The mask is filled through `chunks_exact_mut` plus `skip`/`take` rather than slice indexing, which keeps the panic-free rule and avoids `clippy::needless_range_loop`.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-training-data --test dataset
rustfmt --edition 2024 --check crates/feathertalk-training-data/src/lib.rs crates/feathertalk-training-data/src/dataset.rs crates/feathertalk-training-data/tests/support/mod.rs crates/feathertalk-training-data/tests/dataset.rs
cargo clippy -p feathertalk-training-data --all-targets -- -D warnings
```

All twelve tests must pass. If a mask assertion fails, print the rectangle before changing any constant: the fixture arithmetic above is exact, so a mismatch means `mouth_roi_rect` or `build_face_crop` deviates from task 1 or task 3, not that the expectation is wrong.

- [ ] **Step 5: Commit**

```powershell
cd E:\workspace\github\FeatherTalk
git add rust/crates/feathertalk-training-data/src/lib.rs rust/crates/feathertalk-training-data/src/dataset.rs rust/crates/feathertalk-training-data/tests/support/mod.rs rust/crates/feathertalk-training-data/tests/dataset.rs
git commit -m "feat(training-data): load a single-frame training sample"
```

---

### Task 7: Load a temporal pair against one reference

**Files:**

- Modify: `rust/crates/feathertalk-training-data/src/dataset.rs`
- Test: `rust/crates/feathertalk-training-data/tests/dataset.rs`

**Interfaces:**

- Consumes: the private `resolve_index`, `load_frame`, `inner_planes` and `build_frame_sample` from task 6.
- Produces: `TrainingItem::TemporalPair { first: FrameSample, second: FrameSample }`.
- Preserves: `TrainingItem::SingleFrame` and every accessor on `FrameSample` stay exactly as task 6 left them.

**Why now:** The temporal sampler already exists in `feathertalk-training`, and `temporal_loss` needs both halves of a pair to share one reference frame. Loading the pair is a small addition on top of task 6 and it removes the stub arm, so the dataset stops lying about what it supports. Batching (task 8) needs both item kinds to exist before it can stack either.

- [ ] **Step 1: Write the failing test**

In `rust/crates/feathertalk-training-data/tests/dataset.rs`, extend the `single_frame` helper with the new variant and add the pair helpers next to it:

```rust
fn single_frame(item: &TrainingItem) -> &FrameSample {
    match item {
        TrainingItem::SingleFrame(sample) => sample,
        TrainingItem::TemporalPair { .. } => panic!("expected a single-frame item"),
    }
}

fn temporal_pair(item: &TrainingItem) -> (&FrameSample, &FrameSample) {
    match item {
        TrainingItem::TemporalPair { first, second } => (first, second),
        TrainingItem::SingleFrame(_) => panic!("expected a temporal-pair item"),
    }
}

fn temporal_sample(first: u64, second: u64, reference_index: u64) -> TrainingSample {
    TrainingSample::TemporalPair {
        first_target_index: first,
        second_target_index: second,
        reference_index,
    }
}
```

Then append these tests:

```rust
#[test]
fn a_temporal_pair_shares_one_reference_frame() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let item = dataset.load_sample(&temporal_sample(1, 3, 0)).unwrap();
    let (first, second) = temporal_pair(&item);
    let plane = INNER_SIZE * INNER_SIZE;
    let reference = inner_planes(&project_dir, 0, MouthMasking::Keep);
    assert_eq!(&first.image()[..3 * plane], reference.as_slice());
    assert_eq!(&second.image()[..3 * plane], reference.as_slice());
}

#[test]
fn a_temporal_pair_masks_each_target_separately() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let item = dataset.load_sample(&temporal_sample(1, 3, 0)).unwrap();
    let (first, second) = temporal_pair(&item);
    let plane = INNER_SIZE * INNER_SIZE;
    let first_masked = inner_planes(&project_dir, 1, MouthMasking::Blackout);
    let second_masked = inner_planes(&project_dir, 3, MouthMasking::Blackout);
    assert_eq!(&first.image()[3 * plane..], first_masked.as_slice());
    assert_eq!(&second.image()[3 * plane..], second_masked.as_slice());
}

#[test]
fn a_temporal_pair_carries_two_targets_masks_and_windows() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let item = dataset.load_sample(&temporal_sample(1, 3, 0)).unwrap();
    let (first, second) = temporal_pair(&item);
    let first_target = inner_planes(&project_dir, 1, MouthMasking::Keep);
    let second_target = inner_planes(&project_dir, 3, MouthMasking::Keep);
    assert_eq!(first.target(), first_target.as_slice());
    assert_eq!(second.target(), second_target.as_slice());
    assert_ne!(first.mouth_mask(), second.mouth_mask());
    assert_ne!(first.audio(), second.audio());
}

#[test]
fn a_temporal_pair_rejects_an_index_past_the_end() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let error = dataset.load_sample(&temporal_sample(1, 9, 0)).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("frame index 9 is out of range for 4 frames"));
}
```

The mask and window assertions are the point of the third test. The fixture shifts the mouth y by one pixel per frame, so frame 1's ROI is `y 91` and frame 3's is `y 93`; the audio windows differ because they are centred on different frames.

- [ ] **Step 2: Run test to verify it fails**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-training-data --test dataset
```

Expected failure: `error[E0599]: no variant named TemporalPair found for enum TrainingItem`. The test binary must not compile.

- [ ] **Step 3: Write minimal implementation**

In `rust/crates/feathertalk-training-data/src/dataset.rs`, extend the enum:

```rust
/// What one training sample loads into.
#[derive(Debug, Clone, PartialEq)]
pub enum TrainingItem {
    SingleFrame(FrameSample),
    TemporalPair {
        first: FrameSample,
        second: FrameSample,
    },
}
```

Then replace the stub arm of `load_item` with the real one:

```rust
            TrainingSample::TemporalPair {
                first_target_index,
                second_target_index,
                reference_index,
            } => {
                let first_index = self.resolve_index(*first_target_index)?;
                let second_index = self.resolve_index(*second_target_index)?;
                let reference_index = self.resolve_index(*reference_index)?;
                let reference = self.load_frame(reference_index)?;
                let planes =
                    self.inner_planes(reference_index, &reference.crop, MouthMasking::Keep)?;
                let first_target = self.load_frame(first_index)?;
                let first = self.build_frame_sample(first_index, &first_target, &planes)?;
                let second_target = self.load_frame(second_index)?;
                let second = self.build_frame_sample(second_index, &second_target, &planes)?;
                Ok(TrainingItem::TemporalPair { first, second })
            }
```

The reference planes are built once and passed to both halves. That is not only cheaper, it is what the loss assumes: a pair is two consecutive predictions driven from the same identity reference.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-training-data --test dataset
rustfmt --edition 2024 --check crates/feathertalk-training-data/src/dataset.rs crates/feathertalk-training-data/tests/dataset.rs
cargo clippy -p feathertalk-training-data --all-targets -- -D warnings
```

All sixteen tests must pass.

- [ ] **Step 5: Commit**

```powershell
cd E:\workspace\github\FeatherTalk
git add rust/crates/feathertalk-training-data/src/dataset.rs rust/crates/feathertalk-training-data/tests/dataset.rs
git commit -m "feat(training-data): load a temporal training pair"
```

---

### Task 8: Stack loaded items into batched tensors

**Files:**

- Create: `rust/crates/feathertalk-training-data/src/batch.rs`
- Modify: `rust/crates/feathertalk-training-data/src/lib.rs`
- Test: `rust/crates/feathertalk-training-data/tests/batch.rs`

**Interfaces:**

- Consumes: `TrainingItem` and the `FrameSample` accessors from tasks 6 and 7.
- Produces: `SingleFrameBatch<B>`, `TemporalBatch<B>`, `stack_single_frame_batch` and `stack_temporal_batch`.
- Preserves: the dataset itself stays backend-free; `burn` only appears in this module.

**Why now:** A training step feeds tensors to the U-Net, not per-sample float vectors, so something has to stack the loaded items. The layout is not free either: `temporal_loss` reshapes `[batch * pair_len, ...]` into `[batch, pair_len, ...]`, so the temporal batch must be sample-major — pair 0's first half, pair 0's second half, pair 1's first half, pair 1's second half. That rule belongs next to the code that knows the item layout, which is this crate. Both item kinds exist after task 7, so stacking can cover both in one task instead of being split.

- [ ] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-training-data/tests/batch.rs`:

```rust
mod support;

use std::path::Path;

use feathertalk_training::{TrainingDataset, TrainingSample};
use feathertalk_training_data::{
    FrameSample, ProjectTrainingDataset, TrainingDataError, TrainingItem, stack_single_frame_batch,
    stack_temporal_batch,
};
use support::{GradientFrameReader, INNER_SIZE, locked_project};

type CpuBackend = burn::backend::NdArray<f32>;

fn open_dataset(project_dir: &Path) -> ProjectTrainingDataset<GradientFrameReader> {
    ProjectTrainingDataset::open_with_reader(project_dir, GradientFrameReader).unwrap()
}

fn single_frame(item: &TrainingItem) -> &FrameSample {
    match item {
        TrainingItem::SingleFrame(sample) => sample,
        TrainingItem::TemporalPair { .. } => panic!("expected a single-frame item"),
    }
}

fn temporal_pair(item: &TrainingItem) -> (&FrameSample, &FrameSample) {
    match item {
        TrainingItem::TemporalPair { first, second } => (first, second),
        TrainingItem::SingleFrame(_) => panic!("expected a temporal-pair item"),
    }
}

fn single_frame_sample(target_index: u64, reference_index: u64) -> TrainingSample {
    TrainingSample::SingleFrame {
        target_index,
        reference_index,
    }
}

fn temporal_sample(first: u64, second: u64, reference_index: u64) -> TrainingSample {
    TrainingSample::TemporalPair {
        first_target_index: first,
        second_target_index: second,
        reference_index,
    }
}

fn single_frame_items(project_dir: &Path) -> Vec<TrainingItem> {
    let dataset = open_dataset(project_dir);
    vec![
        dataset.load_sample(&single_frame_sample(1, 0)).unwrap(),
        dataset.load_sample(&single_frame_sample(2, 0)).unwrap(),
    ]
}

fn temporal_items(project_dir: &Path) -> Vec<TrainingItem> {
    let dataset = open_dataset(project_dir);
    vec![
        dataset.load_sample(&temporal_sample(1, 2, 0)).unwrap(),
        dataset.load_sample(&temporal_sample(3, 1, 0)).unwrap(),
    ]
}

#[test]
fn a_single_frame_batch_has_one_row_per_item() {
    let (_temp, project_dir) = locked_project(4);
    let items = single_frame_items(&project_dir);
    let device = Default::default();
    let batch = stack_single_frame_batch::<CpuBackend>(&items, &device).unwrap();
    assert_eq!(batch.image.dims(), [2, 6, INNER_SIZE, INNER_SIZE]);
    assert_eq!(batch.audio.dims(), [2, 16, 32, 32]);
    assert_eq!(batch.target.dims(), [2, 3, INNER_SIZE, INNER_SIZE]);
    assert_eq!(batch.mouth_mask.dims(), [2, 1, INNER_SIZE, INNER_SIZE]);
}

#[test]
fn a_single_frame_batch_keeps_the_item_order() {
    let (_temp, project_dir) = locked_project(4);
    let items = single_frame_items(&project_dir);
    let device = Default::default();
    let batch = stack_single_frame_batch::<CpuBackend>(&items, &device).unwrap();
    let values = batch.image.into_data().to_vec::<f32>().unwrap();
    let stride = 6 * INNER_SIZE * INNER_SIZE;
    assert_eq!(&values[..stride], single_frame(&items[0]).image());
    assert_eq!(&values[stride..], single_frame(&items[1]).image());
}

#[test]
fn a_temporal_batch_flattens_both_halves() {
    let (_temp, project_dir) = locked_project(4);
    let items = temporal_items(&project_dir);
    let device = Default::default();
    let batch = stack_temporal_batch::<CpuBackend>(&items, &device).unwrap();
    assert_eq!(batch.image.dims(), [4, 6, INNER_SIZE, INNER_SIZE]);
    assert_eq!(batch.audio.dims(), [4, 16, 32, 32]);
    assert_eq!(batch.target.dims(), [2, 2, 3, INNER_SIZE, INNER_SIZE]);
    assert_eq!(batch.mouth_mask.dims(), [2, 2, 1, INNER_SIZE, INNER_SIZE]);
}

#[test]
fn a_temporal_batch_is_sample_major() {
    let (_temp, project_dir) = locked_project(4);
    let items = temporal_items(&project_dir);
    let device = Default::default();
    let batch = stack_temporal_batch::<CpuBackend>(&items, &device).unwrap();
    let values = batch.target.into_data().to_vec::<f32>().unwrap();
    let stride = 3 * INNER_SIZE * INNER_SIZE;
    let (first, second) = temporal_pair(&items[0]);
    let (third, fourth) = temporal_pair(&items[1]);
    assert_eq!(&values[..stride], first.target());
    assert_eq!(&values[stride..2 * stride], second.target());
    assert_eq!(&values[2 * stride..3 * stride], third.target());
    assert_eq!(&values[3 * stride..], fourth.target());
}

#[test]
fn an_empty_batch_is_rejected() {
    let device = Default::default();
    let error = stack_single_frame_batch::<CpuBackend>(&[], &device).unwrap_err();
    let message = error.to_string();
    assert!(matches!(error, TrainingDataError::Batch { .. }));
    assert!(message.contains("a batch needs at least one item"));
}

#[test]
fn a_batch_of_mixed_item_kinds_is_rejected() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let items = vec![
        dataset.load_sample(&single_frame_sample(1, 0)).unwrap(),
        dataset.load_sample(&temporal_sample(2, 3, 0)).unwrap(),
    ];
    let device = Default::default();
    let single = stack_single_frame_batch::<CpuBackend>(&items, &device).unwrap_err();
    assert!(single.to_string().contains("item 1 is a temporal pair"));
    let temporal = stack_temporal_batch::<CpuBackend>(&items, &device).unwrap_err();
    assert!(temporal.to_string().contains("item 0 is a single frame"));
}
```

The two pairs deliberately target frames 1, 2, 3 and 1. A sample-major layout therefore reads `1, 2, 3, 1` while a half-major layout would read `1, 3, 2, 1`, so the fourth test can tell the two apart instead of merely checking a shape.

- [ ] **Step 2: Run test to verify it fails**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-training-data --test batch
```

Expect a compile failure, not an assertion failure:

```
error[E0432]: unresolved imports `feathertalk_training_data::stack_single_frame_batch`, `feathertalk_training_data::stack_temporal_batch`
```

- [ ] **Step 3: Write minimal implementation**

Create `rust/crates/feathertalk-training-data/src/batch.rs`:

```rust
use burn::tensor::{Tensor, TensorData, backend::Backend};

use crate::{FrameSample, TrainingDataError, TrainingItem};

const INNER_SIZE: usize = 160;
const IMAGE_CHANNELS: usize = 6;
const TARGET_CHANNELS: usize = 3;
const AUDIO_CHANNELS: usize = 16;
const AUDIO_SIZE: usize = 32;

/// One batch of single-frame items, ready for a U-Net forward pass.
#[derive(Debug, Clone)]
pub struct SingleFrameBatch<B: Backend> {
    pub image: Tensor<B, 4>,
    pub audio: Tensor<B, 4>,
    pub target: Tensor<B, 4>,
    pub mouth_mask: Tensor<B, 4>,
}

/// One batch of temporal pairs: the inputs are flattened, the targets keep the pair axis.
#[derive(Debug, Clone)]
pub struct TemporalBatch<B: Backend> {
    pub image: Tensor<B, 4>,
    pub audio: Tensor<B, 4>,
    pub target: Tensor<B, 5>,
    pub mouth_mask: Tensor<B, 5>,
}

fn batch_error(message: String) -> TrainingDataError {
    TrainingDataError::Batch { message }
}

fn single_frame_samples(items: &[TrainingItem]) -> Result<Vec<&FrameSample>, TrainingDataError> {
    if items.is_empty() {
        return Err(batch_error("a batch needs at least one item".to_owned()));
    }
    let mut samples = Vec::with_capacity(items.len());
    for (position, item) in items.iter().enumerate() {
        match item {
            TrainingItem::SingleFrame(sample) => samples.push(sample),
            TrainingItem::TemporalPair { .. } => {
                return Err(batch_error(format!(
                    "item {position} is a temporal pair but the batch is single-frame"
                )));
            }
        }
    }
    Ok(samples)
}

fn temporal_samples(items: &[TrainingItem]) -> Result<Vec<&FrameSample>, TrainingDataError> {
    if items.is_empty() {
        return Err(batch_error("a batch needs at least one item".to_owned()));
    }
    let mut samples = Vec::with_capacity(items.len().saturating_mul(2));
    for (position, item) in items.iter().enumerate() {
        match item {
            TrainingItem::TemporalPair { first, second } => {
                samples.push(first);
                samples.push(second);
            }
            TrainingItem::SingleFrame(_) => {
                return Err(batch_error(format!(
                    "item {position} is a single frame but the batch is temporal"
                )));
            }
        }
    }
    Ok(samples)
}

fn gather(
    samples: &[&FrameSample],
    field: fn(&FrameSample) -> &[f32],
    expected: usize,
) -> Result<Vec<f32>, TrainingDataError> {
    let Some(elements) = samples.len().checked_mul(expected) else {
        return Err(batch_error("the batch element count overflows".to_owned()));
    };
    let mut values: Vec<f32> = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| batch_error(format!("cannot allocate {elements} floats")))?;
    for (position, sample) in samples.iter().enumerate() {
        let plane = field(sample);
        if plane.len() != expected {
            return Err(batch_error(format!(
                "item {position} has {} values but {expected} were expected",
                plane.len()
            )));
        }
        values.extend_from_slice(plane);
    }
    Ok(values)
}

fn tensor4<B: Backend>(values: Vec<f32>, shape: [usize; 4], device: &B::Device) -> Tensor<B, 4> {
    Tensor::<B, 4>::from_data(TensorData::new(values, shape), device)
}

fn tensor5<B: Backend>(values: Vec<f32>, shape: [usize; 5], device: &B::Device) -> Tensor<B, 5> {
    Tensor::<B, 5>::from_data(TensorData::new(values, shape), device)
}

/// Stacks single-frame items in the order they were given.
pub fn stack_single_frame_batch<B: Backend>(
    items: &[TrainingItem],
    device: &B::Device,
) -> Result<SingleFrameBatch<B>, TrainingDataError> {
    let samples = single_frame_samples(items)?;
    let count = samples.len();
    let plane = INNER_SIZE * INNER_SIZE;
    let audio = AUDIO_CHANNELS * AUDIO_SIZE * AUDIO_SIZE;
    let image_values = gather(&samples, FrameSample::image, IMAGE_CHANNELS * plane)?;
    let audio_values = gather(&samples, FrameSample::audio, audio)?;
    let target_values = gather(&samples, FrameSample::target, TARGET_CHANNELS * plane)?;
    let mask_values = gather(&samples, FrameSample::mouth_mask, plane)?;
    let image_shape = [count, IMAGE_CHANNELS, INNER_SIZE, INNER_SIZE];
    let audio_shape = [count, AUDIO_CHANNELS, AUDIO_SIZE, AUDIO_SIZE];
    let target_shape = [count, TARGET_CHANNELS, INNER_SIZE, INNER_SIZE];
    let mask_shape = [count, 1, INNER_SIZE, INNER_SIZE];
    Ok(SingleFrameBatch {
        image: tensor4(image_values, image_shape, device),
        audio: tensor4(audio_values, audio_shape, device),
        target: tensor4(target_values, target_shape, device),
        mouth_mask: tensor4(mask_values, mask_shape, device),
    })
}

/// Stacks temporal pairs sample-major, so `temporal_loss` can reshape the flattened rows back.
pub fn stack_temporal_batch<B: Backend>(
    items: &[TrainingItem],
    device: &B::Device,
) -> Result<TemporalBatch<B>, TrainingDataError> {
    let samples = temporal_samples(items)?;
    let pairs = items.len();
    let halves = samples.len();
    let plane = INNER_SIZE * INNER_SIZE;
    let audio = AUDIO_CHANNELS * AUDIO_SIZE * AUDIO_SIZE;
    let image_values = gather(&samples, FrameSample::image, IMAGE_CHANNELS * plane)?;
    let audio_values = gather(&samples, FrameSample::audio, audio)?;
    let target_values = gather(&samples, FrameSample::target, TARGET_CHANNELS * plane)?;
    let mask_values = gather(&samples, FrameSample::mouth_mask, plane)?;
    let image_shape = [halves, IMAGE_CHANNELS, INNER_SIZE, INNER_SIZE];
    let audio_shape = [halves, AUDIO_CHANNELS, AUDIO_SIZE, AUDIO_SIZE];
    let target_shape = [pairs, 2, TARGET_CHANNELS, INNER_SIZE, INNER_SIZE];
    let mask_shape = [pairs, 2, 1, INNER_SIZE, INNER_SIZE];
    Ok(TemporalBatch {
        image: tensor4(image_values, image_shape, device),
        audio: tensor4(audio_values, audio_shape, device),
        target: tensor5(target_values, target_shape, device),
        mouth_mask: tensor5(mask_values, mask_shape, device),
    })
}
```

Then extend `rust/crates/feathertalk-training-data/src/lib.rs` to:

```rust
mod batch;
mod dataset;
mod error;

pub use batch::{SingleFrameBatch, TemporalBatch, stack_single_frame_batch, stack_temporal_batch};
pub use dataset::{FrameSample, ProjectTrainingDataset, TrainingItem};
pub use error::TrainingDataError;
```

`try_reserve_exact` is deliberate: the element count is derived from caller data, so a bad batch size has to surface as a `TrainingDataError::Batch` instead of aborting the process on allocation failure.

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-training-data
rustfmt --edition 2024 --check crates/feathertalk-training-data/src/batch.rs crates/feathertalk-training-data/src/lib.rs crates/feathertalk-training-data/tests/batch.rs
cargo clippy -p feathertalk-training-data --all-targets -- -D warnings
```

All six batch tests plus the sixteen dataset tests must pass.

- [ ] **Step 5: Commit**

```powershell
cd E:\workspace\github\FeatherTalk
git add rust/crates/feathertalk-training-data/src/batch.rs rust/crates/feathertalk-training-data/src/lib.rs rust/crates/feathertalk-training-data/tests/batch.rs
git commit -m "feat(training-data): stack training batches"
```

---

### Task 9: Prove the real JPEG path end to end

**Files:**

- Test: `rust/crates/feathertalk-training-data/tests/real_frames.rs`

**Interfaces:**

- Consumes: `ProjectTrainingDataset::open` (the `JpegFrameReader` default) and `support::build_locked_project`.
- Produces: nothing public; this task adds coverage only.
- Preserves: every earlier test keeps using `GradientFrameReader`, so the fast suite stays fast.

**Why now:** Tasks 6 to 8 all run through `GradientFrameReader`, which synthesises pixels instead of decoding them. That keeps those tests deterministic, but it also means nothing has yet exercised `ProjectTrainingDataset::open`, the real decoder, or a real face box against real frame dimensions. This task closes that gap with the JPEG fixture the frame adapters already ship, and it pins the manifest-versus-frame disagreement to a message that names both sizes — the single most likely production misconfiguration. It is test-only, so it belongs after the code is complete.

- [ ] **Step 1: Write the test**

Create `rust/crates/feathertalk-training-data/tests/real_frames.rs`:

```rust
mod support;

use std::fs;
use std::path::{Path, PathBuf};

use feathertalk_training::{TrainingDataset, TrainingError, TrainingSample};
use feathertalk_training_data::{ProjectTrainingDataset, TrainingItem};
use support::{FixtureSpec, INNER_SIZE, build_locked_project};

fn demo_frame() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../feathertalk-frame-adapters/tests/fixtures/demo_frame_v1/frame.jpg")
}

fn real_frame_spec(manifest_width: u32, manifest_height: u32) -> FixtureSpec {
    FixtureSpec {
        frame_count: 2,
        manifest_width,
        manifest_height,
        frame_bytes: fs::read(demo_frame()).unwrap(),
        face_xmin: 551,
        face_xmax: 710,
        face_ymin: 194,
        mouth_x: 600,
        mouth_y: 230,
    }
}

fn target_one() -> TrainingSample {
    TrainingSample::SingleFrame {
        target_index: 1,
        reference_index: 0,
    }
}

#[test]
fn opens_a_project_whose_frames_are_real_jpeg_files() {
    let (_temp, project_dir) = build_locked_project(&real_frame_spec(1280, 720));
    let dataset = ProjectTrainingDataset::open(&project_dir).unwrap();
    assert_eq!(dataset.frame_count(), 2);
    let item = dataset.load_sample(&target_one()).unwrap();
    let TrainingItem::SingleFrame(frame) = item else {
        panic!("expected a single-frame item");
    };
    assert_eq!(frame.image().len(), 6 * INNER_SIZE * INNER_SIZE);
    assert_eq!(frame.target().len(), 3 * INNER_SIZE * INNER_SIZE);
    assert_eq!(frame.mouth_mask().len(), INNER_SIZE * INNER_SIZE);
    assert!(frame.image().iter().all(|value| (0.0..=1.0).contains(value)));
}

#[test]
fn a_frame_that_contradicts_the_manifest_is_rejected() {
    let (_temp, project_dir) = build_locked_project(&real_frame_spec(640, 480));
    let dataset = ProjectTrainingDataset::open(&project_dir).unwrap();
    let error = dataset.load_sample(&target_one()).unwrap_err();
    let message = error.to_string();
    assert!(matches!(error, TrainingError::InvalidInput(_)));
    assert!(message.contains("frame is 1280x720 but the asset package declares 640x480"));
}
```

The fixture is `mjpeg`, `1280x720`, three components, so `JpegFrameReader` decodes it as `RGB24`. The face box `551..710` by `194..353` sits inside that frame, the crop scale is `168 / 159`, and the mouth ROI stays well inside the `160x160` inner image, so nothing is clamped. The second test still opens successfully because `validate_project_dir` only reads manifests; the disagreement can only be caught when a frame is actually decoded. `load_frame` checks the decoded dimensions before it reads landmarks, which is why the out-of-range landmark coordinates in the `640x480` case never matter.

- [ ] **Step 2: Run the test**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test -p feathertalk-training-data --test real_frames
```

Both tests are expected to pass on the first run. That is the point: they assert that tasks 1 to 8 already handle real bytes, so passing immediately is a valid outcome for a coverage task.

- [ ] **Step 3: If a test fails, fix the owning task's code**

A failure here is a real defect, not a wrong expectation. Map it before editing anything:

- `unsupported JPEG pixel format` or a decode error: the fixture choice is wrong, pick another real frame.
- a landmark or bbox error: task 1's `mouth_roi_rect` or `compute_face_bbox` mishandles a non-square face box.
- a crop or plane length error: task 2 or task 3 assumes the `256x256` fixture geometry.
- a value outside `0.0..=1.0`: task 2's normalisation is wrong.

Fix the crate that owns the defect, then re-run its own tests before coming back here.

- [ ] **Step 4: Run the crate gates**

```powershell
cd E:\workspace\github\FeatherTalk\rust
rustfmt --edition 2024 --check crates/feathertalk-training-data/tests/real_frames.rs
cargo clippy -p feathertalk-training-data --all-targets -- -D warnings
cargo test -p feathertalk-training-data
```

- [ ] **Step 5: Commit**

```powershell
cd E:\workspace\github\FeatherTalk
git add rust/crates/feathertalk-training-data/tests/real_frames.rs
git commit -m "test(training-data): open a project with real JPEG frames"
```

---

### Task 10: Run the workspace gates

**Files:**

- None: this task runs gates and commits nothing.

**Interfaces:**

- Consumes: the whole workspace as tasks 1 to 9 left it.
- Produces: a clean format, clippy, test and end-to-end result.
- Preserves: the untracked `demo/kanghui_training_video_featherhubert_188_latest/` directory stays untracked and unstaged.

**Why now:** This slice touched five crates and added one. Per-task gates only covered the crate being edited, so a signature change that broke a downstream caller would still be invisible. Running the full set once at the end is the cheapest way to catch that. A gate failure is fixed inside the owning task's scope; it is never fixed by loosening the gate, skipping a test or adding an `allow` attribute.

- [ ] **Step 1: Format gate**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo fmt --all -- --check *> "$env:TEMP\ft_g1.log"; "gate1=$LASTEXITCODE"
```

- [ ] **Step 2: Clippy gate**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo clippy --workspace --all-targets -- -D warnings *> "$env:TEMP\ft_g2.log"; "gate2=$LASTEXITCODE"
Select-String -Path "$env:TEMP\ft_g2.log" -Pattern 'warning:|error'
```

Takes about 95 seconds warm.

- [ ] **Step 3: Test gate**

```powershell
cd E:\workspace\github\FeatherTalk\rust
cargo test --workspace --all-targets *> "$env:TEMP\ft_g3.log"; "gate3=$LASTEXITCODE"
Select-String -Path "$env:TEMP\ft_g3.log" -Pattern 'test result:' | Select-Object -Last 5
Select-String -Path "$env:TEMP\ft_g3.log" -Pattern 'FAILED|panicked' -CaseSensitive
```

This is the long one, roughly 48 minutes. The baseline before this slice was 193 test binaries with 995 passed, 0 failed and 13 ignored. Afterwards expect about 198 binaries and a higher pass count; 0 failed is the only number that must hold.

- [ ] **Step 4: Real-worker end-to-end gate**

```powershell
cd E:\workspace\github\FeatherTalk\rust
$env:FEATHERTALK_REQUIRE_E2E = "1"
$env:FEATHERTALK_WORKER_FFMPEG = "D:\environment\ffmpeg\bin\ffmpeg.exe"
$env:FEATHERTALK_WORKER_HUBERT_DIR = "C:\Users\Administrator\AppData\Local\Temp\ft_hubert_e2e\package"
cargo test --release -p feathertalk-cli --test real_worker -- --nocapture *> "$env:TEMP\ft_g4.log"; "gate4=$LASTEXITCODE"
Select-String -Path "$env:TEMP\ft_g4.log" -Pattern 'test result:'
```

Nine tests must pass, not be skipped: `FEATHERTALK_REQUIRE_E2E=1` turns a missing prerequisite into a failure instead of an ignore. The HuBERT package directory must still exist; if it does not, restore it before running rather than dropping the variable.

- [ ] **Step 5: Final tree check**

```powershell
cd E:\workspace\github\FeatherTalk
git status -sb
git diff --check
git log --oneline -11
```

Expect a clean tree apart from the untracked demo directory, no whitespace complaints, and the nine task commits sitting on top of the design and plan commits. Nothing is pushed.

---
