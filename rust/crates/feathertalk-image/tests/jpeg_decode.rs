use feathertalk_image::{BgrImage, ImageError, decode_jpeg};

/// SOI followed by a baseline 4:4:4 three-component SOF0 and nothing else.
fn jpeg_header(width: u16, height: u16) -> Vec<u8> {
    let mut bytes = vec![0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x11, 0x08];
    bytes.extend_from_slice(&height.to_be_bytes());
    bytes.extend_from_slice(&width.to_be_bytes());
    bytes.extend_from_slice(&[0x03, 0x01, 0x11, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01]);
    bytes
}

#[test]
fn an_oversized_header_is_rejected_before_decoding() {
    let error = decode_jpeg(&jpeg_header(20_000, 20_000), 64 * 1024 * 1024).unwrap_err();
    assert!(
        matches!(
            error,
            ImageError::FrameTooLarge {
                pixels: 400_000_000,
                max_pixels: 67_108_864
            }
        ),
        "{error:?}"
    );
}

#[test]
fn a_header_within_budget_reaches_the_scan_decoder() {
    // The header carries no scan data, so decoding must fail after the budget check.
    let error = decode_jpeg(&jpeg_header(8, 8), 64 * 1024 * 1024).unwrap_err();
    assert!(matches!(error, ImageError::JpegDecode { .. }), "{error:?}");
}

#[test]
fn empty_input_is_a_decode_error() {
    let error = decode_jpeg(&[], 64 * 1024 * 1024).unwrap_err();
    let ImageError::JpegDecode { message } = error else {
        panic!("expected a decode error, got {error:?}");
    };
    assert!(message.contains("failed to fill whole buffer"), "{message}");
}

#[test]
fn non_jpeg_input_is_a_decode_error() {
    let error = decode_jpeg(b"not a jpeg at all", 64 * 1024 * 1024).unwrap_err();
    let ImageError::JpegDecode { message } = error else {
        panic!("expected a decode error, got {error:?}");
    };
    assert!(
        message.contains("first two bytes are not an SOI marker"),
        "{message}"
    );
}

#[test]
fn a_zero_pixel_budget_rejects_every_image() {
    let error = decode_jpeg(&jpeg_header(8, 8), 0).unwrap_err();
    assert!(
        matches!(
            error,
            ImageError::FrameTooLarge {
                pixels: 64,
                max_pixels: 0
            }
        ),
        "{error:?}"
    );
}

#[test]
fn bgr_image_validates_its_buffer_length() {
    let error = BgrImage::new(2, 2, vec![0; 11]).unwrap_err();
    assert!(
        matches!(
            error,
            ImageError::BufferLengthMismatch {
                width: 2,
                height: 2,
                expected: 12,
                actual: 11
            }
        ),
        "{error:?}"
    );
    assert!(matches!(
        BgrImage::new(0, 2, vec![]).unwrap_err(),
        ImageError::InvalidDimensions {
            width: 0,
            height: 2
        }
    ));
}

#[test]
fn bgr_image_reads_interleaved_pixels() {
    let image = BgrImage::new(2, 1, vec![1, 2, 3, 4, 5, 6]).unwrap();
    assert_eq!(image.width(), 2);
    assert_eq!(image.height(), 1);
    assert_eq!(image.as_bytes().len(), 6);
    assert_eq!(image.pixel(0, 0).unwrap(), [1, 2, 3]);
    assert_eq!(image.pixel(1, 0).unwrap(), [4, 5, 6]);
    assert!(matches!(
        image.pixel(2, 0).unwrap_err(),
        ImageError::PixelOutOfBounds {
            x: 2,
            y: 0,
            width: 2,
            height: 1
        }
    ));
}
