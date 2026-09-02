use feathertalk_face::{
    Detection, DetectionConfig, FaceError, ImageSize, ResizeTransform, decode_level,
    generate_anchor_centers, non_max_suppression, resize_with_padding,
};
use feathertalk_frame_pipeline::{FaceDetection, PipelineError};
use feathertalk_image::{BgrImage, resize_area};

/// SCRFD's fixed square input edge, the canvas `resize_with_padding` targets.
const SCRFD_EDGE: u32 = 640;

/// `(0.0 - 127.5) / 128.0`: what the zero-filled letterbox border normalizes
/// to. `blobFromImage` applies the mean and scale to the padding as well, so
/// the border is not zero in the blob.
const PADDED_VALUE: f32 = -0.99609375;

/// A ready-to-upload SCRFD input plus the letterbox that produced it.
#[derive(Debug, Clone)]
pub struct ScrfdInput {
    /// The mapping Task 11 inverts to return detections in source pixels.
    pub transform: ResizeTransform,
    /// NCHW `[1, 3, 640, 640]`, RGB, normalized to `(v - 127.5) / 128`.
    pub data: Vec<f32>,
}

/// Letterbox `image` into SCRFD's 640x640 canvas and build the input blob.
///
/// This is the port of `resize_image` followed by `cv2.dnn.blobFromImage` in
/// `data_utils/detect_face.py`. It performs no inference and loads no weights,
/// so its parity is testable on its own.
pub fn scrfd_input(image: &BgrImage) -> Result<ScrfdInput, PipelineError> {
    let transform = resize_with_padding(ImageSize {
        width: image.width(),
        height: image.height(),
    })
    .map_err(|error| PipelineError::Adapter {
        component: "scrfd",
        message: format!("letterbox failed: {error}"),
    })?;

    let resized =
        resize_area(image, transform.new_width, transform.new_height).map_err(|error| {
            PipelineError::Adapter {
                component: "scrfd",
                message: format!(
                    "resize to {}x{} failed: {error}",
                    transform.new_width, transform.new_height
                ),
            }
        })?;

    let edge = SCRFD_EDGE as usize;
    let plane = edge * edge;
    let width = resized.width() as usize;
    let height = resized.height() as usize;
    let pad_x = transform.pad_x as usize;
    let pad_y = transform.pad_y as usize;

    // Both branches of `resize_with_padding` cap the resized edge at 640, so
    // this cannot fire today. It stays because the loop below indexes a `Vec`
    // at a computed offset, and an out-of-bounds panic in an adapter kills the
    // worker instead of producing a frame anomaly.
    if pad_x + width > edge || pad_y + height > edge {
        return Err(PipelineError::Adapter {
            component: "scrfd",
            message: format!(
                "letterbox does not fit the {edge}x{edge} canvas: {width}x{height} at ({pad_x}, {pad_y})"
            ),
        });
    }

    let mut data = vec![PADDED_VALUE; 3 * plane];
    let bytes = resized.as_bytes();
    for y in 0..height {
        for x in 0..width {
            let source = (y * width + x) * 3;
            let target = (y + pad_y) * edge + x + pad_x;
            // SCRFD consumes RGB and `BgrImage` stores B, G, R, so channel c
            // reads byte 2 - c. Task 13's `pfld_input` keeps BGR; the two are
            // not interchangeable.
            for channel in 0..3 {
                data[channel * plane + target] =
                    (f32::from(bytes[source + 2 - channel]) - 127.5) * (1.0 / 128.0);
            }
        }
    }

    Ok(ScrfdInput { transform, data })
}

/// SCRFD emits two anchors per feature-map location at every stride.
const ANCHORS_PER_LOCATION: u32 = 2;

/// One SCRFD output level copied back to host memory.
///
/// `bbox_distances` and `keypoint_distances` are the raw regression outputs, in
/// stride units; `decode_level` applies the stride and the letterbox.
#[derive(Debug, Clone)]
pub struct LevelHostData {
    /// Index into `SCRFD_STRIDES`, used only in error messages.
    pub level: usize,
    pub stride: u32,
    /// One score per anchor: 12 800, 3 200 and 800 for strides 8, 16 and 32.
    pub scores: Vec<f32>,
    pub bbox_distances: Vec<[f32; 4]>,
    pub keypoint_distances: Vec<[f32; 10]>,
}

/// Decode three SCRFD levels and reduce them with non-maximum suppression.
///
/// The port of `detect_face.py`'s postprocessing loop. Anchors below
/// `config.confidence_threshold` are skipped before decoding, exactly as the
/// reference does, and boxes that clamp to nothing are dropped rather than
/// failing the frame.
pub fn scrfd_detections(
    levels: &[LevelHostData; 3],
    transform: &ResizeTransform,
    config: &DetectionConfig,
) -> Result<Vec<FaceDetection>, PipelineError> {
    let mut candidates: Vec<Detection> = Vec::new();

    for level in levels {
        let anchors = generate_anchor_centers(transform.model, level.stride, ANCHORS_PER_LOCATION)
            .map_err(|error| level_error(level.level, None, &error))?;

        // `decode_level` checks these too, but it is called with one-anchor
        // slices below, so its check can never see a truncated buffer.
        for (field, actual) in [
            ("scores", level.scores.len()),
            ("bbox_distances", level.bbox_distances.len()),
            ("keypoint_distances", level.keypoint_distances.len()),
        ] {
            if actual != anchors.len() {
                return Err(PipelineError::Adapter {
                    component: "scrfd",
                    message: format!(
                        "level {} {field} holds {actual} entries, expected {}",
                        level.level,
                        anchors.len()
                    ),
                });
            }
        }

        for (index, score) in level.scores.iter().enumerate() {
            // The reference filters before decoding, so a sub-threshold anchor
            // is allowed to hold geometry that would be rejected below.
            if *score < config.confidence_threshold {
                continue;
            }
            let window = index..index + 1;
            match decode_level(
                level.level,
                level.stride,
                &anchors[window.clone()],
                &level.scores[window.clone()],
                &level.bbox_distances[window.clone()],
                &level.keypoint_distances[window],
                transform,
            ) {
                Ok(decoded) => candidates.extend(decoded),
                // A box that clamps to zero area is not an error: upstream
                // never emits it, and the frame may still hold a real face.
                Err(FaceError::InvalidDetectionGeometry { .. }) => continue,
                Err(error) => return Err(level_error(level.level, Some(index), &error)),
            }
        }
    }

    let kept =
        non_max_suppression(&candidates, config).map_err(|error| PipelineError::Adapter {
            component: "scrfd",
            message: error.to_string(),
        })?;

    Ok(kept
        .into_iter()
        .map(|index| {
            let candidate = candidates[index];
            FaceDetection {
                bbox: candidate.bbox,
                score: candidate.score,
                keypoints: candidate.keypoints,
            }
        })
        .collect())
}

/// Attach the level, and where known the anchor, to a `feathertalk-face`
/// failure. `decode_level` reports index 0 for a one-anchor slice, so its own
/// message cannot identify the anchor.
fn level_error(level: usize, anchor: Option<usize>, error: &FaceError) -> PipelineError {
    let message = match anchor {
        Some(anchor) => format!("level {level} anchor {anchor}: {error}"),
        None => format!("level {level}: {error}"),
    };
    PipelineError::Adapter {
        component: "scrfd",
        message,
    }
}
