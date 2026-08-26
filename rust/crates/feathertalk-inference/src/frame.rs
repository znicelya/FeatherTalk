use crate::InferenceError;

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
