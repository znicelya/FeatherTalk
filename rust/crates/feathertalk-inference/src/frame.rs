use crate::InferenceError;
use feathertalk_preprocess::FaceBoundingBox;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BgrFrame {
    width: u32,
    height: u32,
    bgr: Vec<u8>,
}

impl BgrFrame {
    pub fn new(width: u32, height: u32, bgr: Vec<u8>) -> Result<Self, InferenceError> {
        if width == 0 || height == 0 {
            return Err(InferenceError::InvalidFrameDimensions { width, height });
        }
        let expected = checked_byte_len(width, height)?;
        if bgr.len() != expected {
            return Err(InferenceError::FrameBufferLengthMismatch {
                expected,
                actual: bgr.len(),
            });
        }
        Ok(Self { width, height, bgr })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bgr
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bgr
    }

    pub fn pixel(&self, x: u32, y: u32) -> Result<[u8; 3], InferenceError> {
        if x >= self.width || y >= self.height {
            return Err(InferenceError::PixelOutOfRange {
                x,
                y,
                width: self.width,
                height: self.height,
            });
        }
        let offset = pixel_offset(self.width, x, y)?;
        Ok([self.bgr[offset], self.bgr[offset + 1], self.bgr[offset + 2]])
    }

    fn zeroed(width: u32, height: u32) -> Result<Self, InferenceError> {
        if width == 0 || height == 0 {
            return Err(InferenceError::InvalidFrameDimensions { width, height });
        }
        let expected = checked_byte_len(width, height)?;
        let mut bgr = Vec::new();
        bgr.try_reserve_exact(expected)
            .map_err(|_| InferenceError::AllocationFailure { bytes: expected })?;
        bgr.resize(expected, 0);
        Ok(Self { width, height, bgr })
    }

    fn pixel_offset_checked(&self, x: u32, y: u32) -> Result<usize, InferenceError> {
        pixel_offset(self.width, x, y)
    }
}

pub fn crop_bgr(frame: &BgrFrame, bbox: &FaceBoundingBox) -> Result<BgrFrame, InferenceError> {
    if bbox.xmin < 0
        || bbox.ymin < 0
        || bbox.xmax <= bbox.xmin
        || bbox.ymax <= bbox.ymin
        || i64::from(bbox.xmax) > i64::from(frame.width)
        || i64::from(bbox.ymax) > i64::from(frame.height)
    {
        return Err(InferenceError::InvalidBbox {
            xmin: bbox.xmin,
            ymin: bbox.ymin,
            xmax: bbox.xmax,
            ymax: bbox.ymax,
            frame_width: frame.width,
            frame_height: frame.height,
        });
    }
    let width =
        u32::try_from(bbox.xmax - bbox.xmin).map_err(|_| InferenceError::ArithmeticOverflow)?;
    let height =
        u32::try_from(bbox.ymax - bbox.ymin).map_err(|_| InferenceError::ArithmeticOverflow)?;
    let mut crop = BgrFrame::zeroed(width, height)?;
    let source_x = u32::try_from(bbox.xmin).map_err(|_| InferenceError::ArithmeticOverflow)?;
    let source_y = u32::try_from(bbox.ymin).map_err(|_| InferenceError::ArithmeticOverflow)?;
    let row_bytes = checked_byte_len(width, 1)?;
    for y in 0..height {
        let source_offset = frame.pixel_offset_checked(source_x, source_y + y)?;
        let destination_offset = crop.pixel_offset_checked(0, y)?;
        crop.bgr[destination_offset..destination_offset + row_bytes]
            .copy_from_slice(&frame.bgr[source_offset..source_offset + row_bytes]);
    }
    Ok(crop)
}

pub fn resize_bilinear(
    frame: &BgrFrame,
    width: u32,
    height: u32,
) -> Result<BgrFrame, InferenceError> {
    if width == 0 || height == 0 {
        return Err(InferenceError::InvalidResizeTarget { width, height });
    }
    let mut result = BgrFrame::zeroed(width, height)?;
    let scale_x = frame.width as f32 / width as f32;
    let scale_y = frame.height as f32 / height as f32;
    let max_x = frame.width - 1;
    let max_y = frame.height - 1;
    for y in 0..height {
        let source_y = (y as f32 + 0.5) * scale_y - 0.5;
        let floor_y = source_y.floor();
        let y0 = floor_y.clamp(0.0, max_y as f32) as u32;
        let y1 = y0.saturating_add(1).min(max_y);
        let wy = (source_y - floor_y).clamp(0.0, 1.0);
        for x in 0..width {
            let source_x = (x as f32 + 0.5) * scale_x - 0.5;
            let floor_x = source_x.floor();
            let x0 = floor_x.clamp(0.0, max_x as f32) as u32;
            let x1 = x0.saturating_add(1).min(max_x);
            let wx = (source_x - floor_x).clamp(0.0, 1.0);
            let p00 = frame.pixel(x0, y0)?;
            let p01 = frame.pixel(x1, y0)?;
            let p10 = frame.pixel(x0, y1)?;
            let p11 = frame.pixel(x1, y1)?;
            let destination_offset = result.pixel_offset_checked(x, y)?;
            for channel in 0..3 {
                let top = p00[channel] as f32 * (1.0 - wx) + p01[channel] as f32 * wx;
                let bottom = p10[channel] as f32 * (1.0 - wx) + p11[channel] as f32 * wx;
                let value = (top * (1.0 - wy) + bottom * wy).clamp(0.0, 255.0).round();
                result.bgr[destination_offset + channel] = value as u8;
            }
        }
    }
    Ok(result)
}

fn checked_byte_len(width: u32, height: u32) -> Result<usize, InferenceError> {
    let width = usize::try_from(width).map_err(|_| InferenceError::ArithmeticOverflow)?;
    let height = usize::try_from(height).map_err(|_| InferenceError::ArithmeticOverflow)?;
    width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or(InferenceError::ArithmeticOverflow)
}

fn pixel_offset(width: u32, x: u32, y: u32) -> Result<usize, InferenceError> {
    let width = usize::try_from(width).map_err(|_| InferenceError::ArithmeticOverflow)?;
    let x = usize::try_from(x).map_err(|_| InferenceError::ArithmeticOverflow)?;
    let y = usize::try_from(y).map_err(|_| InferenceError::ArithmeticOverflow)?;
    y.checked_mul(width)
        .and_then(|row| row.checked_add(x))
        .and_then(|pixel| pixel.checked_mul(3))
        .ok_or(InferenceError::ArithmeticOverflow)
}
