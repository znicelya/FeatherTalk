use std::io::Cursor;

use jpeg_decoder::{Decoder, PixelFormat};

use crate::{BgrImage, error::ImageError};

/// Decode a JPEG into BGR24.
///
/// The header is parsed first, so an image larger than `max_pixels` is rejected
/// before any pixel buffer is allocated. `max_pixels` counts pixels, not bytes;
/// the decoder's own buffer limit is derived from it.
pub fn decode_jpeg(bytes: &[u8], max_pixels: u64) -> Result<BgrImage, ImageError> {
    let mut decoder = Decoder::new(Cursor::new(bytes));
    decoder
        .read_info()
        .map_err(|error| ImageError::JpegDecode {
            message: error.to_string(),
        })?;
    let info = decoder.info().ok_or_else(|| ImageError::JpegDecode {
        message: "JPEG header carried no image information".to_owned(),
    })?;
    let width = u32::from(info.width);
    let height = u32::from(info.height);
    if width == 0 || height == 0 {
        return Err(ImageError::InvalidDimensions { width, height });
    }
    let pixels = u64::from(width) * u64::from(height);
    if pixels > max_pixels {
        return Err(ImageError::FrameTooLarge { pixels, max_pixels });
    }
    let bgr_len = pixels
        .checked_mul(3)
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or(ImageError::FrameTooLarge { pixels, max_pixels })?;
    decoder.set_max_decoding_buffer_size(bgr_len);
    let pixel_format = info.pixel_format;
    let data = decoder.decode().map_err(|error| ImageError::JpegDecode {
        message: error.to_string(),
    })?;
    let pixel_count = bgr_len / 3;
    let mut bgr = vec![0u8; bgr_len];
    match pixel_format {
        PixelFormat::RGB24 => {
            if data.len() != bgr_len {
                return Err(ImageError::JpegDecode {
                    message: format!(
                        "decoder returned {} bytes for a {width}x{height} RGB image, expected {bgr_len}",
                        data.len()
                    ),
                });
            }
            for (destination, source) in bgr.chunks_exact_mut(3).zip(data.chunks_exact(3)) {
                destination[0] = source[2];
                destination[1] = source[1];
                destination[2] = source[0];
            }
        }
        PixelFormat::L8 => {
            if data.len() != pixel_count {
                return Err(ImageError::JpegDecode {
                    message: format!(
                        "decoder returned {} bytes for a {width}x{height} grayscale image, expected {pixel_count}",
                        data.len()
                    ),
                });
            }
            for (destination, &luma) in bgr.chunks_exact_mut(3).zip(data.iter()) {
                destination.fill(luma);
            }
        }
        other => {
            return Err(ImageError::UnsupportedPixelFormat {
                format: format!("{other:?}"),
            });
        }
    }
    BgrImage::new(width, height, bgr)
}
