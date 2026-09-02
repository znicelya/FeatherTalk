use crate::error::{ImageError, expected_len};

/// Row-major, pixel-interleaved BGR24 image.
///
/// The same memory layout as `feathertalk_inference::BgrFrame`. The two are kept
/// separate on purpose: this crate must not depend on the inference crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BgrImage {
    width: u32,
    height: u32,
    bgr: Vec<u8>,
}

impl BgrImage {
    /// Take ownership of a buffer that is exactly `width * height * 3` bytes.
    pub fn new(width: u32, height: u32, bgr: Vec<u8>) -> Result<Self, ImageError> {
        let expected = expected_len(width, height, 3)?;
        if bgr.len() != expected {
            return Err(ImageError::BufferLengthMismatch {
                width,
                height,
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

    /// The three bytes at `(x, y)` in B, G, R order.
    pub fn pixel(&self, x: u32, y: u32) -> Result<[u8; 3], ImageError> {
        if x >= self.width || y >= self.height {
            return Err(ImageError::PixelOutOfBounds {
                x,
                y,
                width: self.width,
                height: self.height,
            });
        }
        let base = (y as usize * self.width as usize + x as usize) * 3;
        Ok([self.bgr[base], self.bgr[base + 1], self.bgr[base + 2]])
    }
}
