use feathertalk_face::FaceCropGeometry;
use feathertalk_frame_pipeline::PipelineError;
use feathertalk_image::{BgrImage, resize_linear};

/// PFLD consumes a 192x192 crop.
const PFLD_EDGE: u32 = 192;

/// Build the PFLD input blob for one face.
///
/// The port of the crop, `cv2.copyMakeBorder`, `cv2.resize` and
/// `cv2.dnn.blobFromImage` sequence in `data_utils/detect_face.py`. It loads no
/// weights, so its parity is testable on its own.
pub fn pfld_input(
    image: &BgrImage,
    geometry: &FaceCropGeometry,
) -> Result<Vec<f32>, PipelineError> {
    let size = geometry.size as usize;
    if size == 0 {
        return Err(PipelineError::Adapter {
            component: "pfld",
            message: "crop size must be non-zero".to_owned(),
        });
    }

    // `source` is already clipped to the image, so both coordinates are
    // non-negative; `RectI` stores them signed because `requested` is not.
    let (Ok(source_x), Ok(source_y)) = (
        usize::try_from(geometry.source.x),
        usize::try_from(geometry.source.y),
    ) else {
        return Err(PipelineError::Adapter {
            component: "pfld",
            message: format!(
                "clipped crop origin is negative: ({}, {})",
                geometry.source.x, geometry.source.y
            ),
        });
    };

    let source_width = geometry.source.width as usize;
    let source_height = geometry.source.height as usize;
    let pad_left = geometry.padding.left as usize;
    let pad_top = geometry.padding.top as usize;
    let image_width = image.width() as usize;
    let image_height = image.height() as usize;

    // Neither check can fire for a geometry `compute_face_crop_geometry`
    // produced: it clips `source` to the image and derives `padding` from the
    // same integers. They stay because the copy below indexes `Vec`s at
    // computed offsets, and an out-of-bounds panic in an adapter kills the
    // worker instead of producing a frame anomaly.
    if source_x + source_width > image_width || source_y + source_height > image_height {
        return Err(PipelineError::Adapter {
            component: "pfld",
            message: format!(
                "source rectangle {source_width}x{source_height} at ({source_x}, {source_y}) exceeds the {image_width}x{image_height} frame"
            ),
        });
    }
    if pad_left + source_width > size || pad_top + source_height > size {
        return Err(PipelineError::Adapter {
            component: "pfld",
            message: format!(
                "source rectangle {source_width}x{source_height} at ({pad_left}, {pad_top}) does not fit the {size}x{size} canvas"
            ),
        });
    }

    // `copyMakeBorder(..., BORDER_CONSTANT, 0)`: the canvas starts black and
    // the clipped rectangle lands at the padding offset.
    let mut canvas = vec![0_u8; size * size * 3];
    let source_bytes = image.as_bytes();
    let row_bytes = source_width * 3;
    for row in 0..source_height {
        let from = ((source_y + row) * image_width + source_x) * 3;
        let to = ((pad_top + row) * size + pad_left) * 3;
        canvas[to..to + row_bytes].copy_from_slice(&source_bytes[from..from + row_bytes]);
    }

    let square = BgrImage::new(geometry.size, geometry.size, canvas).map_err(|error| {
        PipelineError::Adapter {
            component: "pfld",
            message: format!("crop canvas {size}x{size} is invalid: {error}"),
        }
    })?;
    let resized =
        resize_linear(&square, PFLD_EDGE, PFLD_EDGE).map_err(|error| PipelineError::Adapter {
            component: "pfld",
            message: format!("resize to {PFLD_EDGE}x{PFLD_EDGE} failed: {error}"),
        })?;

    let plane = PFLD_EDGE as usize * PFLD_EDGE as usize;
    let resized_bytes = resized.as_bytes();
    let mut data = vec![0.0_f32; 3 * plane];
    for pixel in 0..plane {
        // PFLD was trained on BGR, so channel c reads byte c. `scrfd_input`
        // reads 2 - c; the two are not interchangeable. The division must stay
        // a division: `1.0 / 255.0` is inexact in f32, and the reference
        // divides.
        for channel in 0..3 {
            data[channel * plane + pixel] = f32::from(resized_bytes[pixel * 3 + channel]) / 255.0;
        }
    }

    Ok(data)
}
