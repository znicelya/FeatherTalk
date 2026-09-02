use feathertalk_face::{ImageSize, ResizeTransform, resize_with_padding};
use feathertalk_frame_pipeline::PipelineError;
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
