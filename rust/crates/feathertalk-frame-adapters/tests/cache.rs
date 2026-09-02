use std::{path::Path, sync::Arc};

use feathertalk_frame_adapters::{DEFAULT_MAX_FRAME_PIXELS, FrameImageCache, JpegFrameDecoder};
use feathertalk_frame_pipeline::{FrameDecoder, PipelineError};

/// SOI followed by a baseline 4:4:4 three-component SOF0 and nothing else.
fn jpeg_header(width: u16, height: u16) -> Vec<u8> {
    let mut bytes = vec![0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x11, 0x08];
    bytes.extend_from_slice(&height.to_be_bytes());
    bytes.extend_from_slice(&width.to_be_bytes());
    bytes.extend_from_slice(&[0x03, 0x01, 0x11, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01]);
    bytes
}

#[test]
fn the_default_budget_matches_the_inference_frame_reader() {
    assert_eq!(DEFAULT_MAX_FRAME_PIXELS, 64 * 1024 * 1024);
}

#[test]
fn a_missing_frame_file_is_an_io_error() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("absent.jpg");
    let error = FrameImageCache::new().load(&path).unwrap_err();
    match error {
        PipelineError::Io {
            operation,
            path: reported,
            ..
        } => {
            assert_eq!(operation, "decode_frame");
            assert_eq!(reported, path);
        }
        other => panic!("expected an Io error, got {other}"),
    }
}

#[test]
fn a_file_that_is_not_a_jpeg_is_an_adapter_error() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("frame.jpg");
    std::fs::write(&path, b"this is not a JPEG").unwrap();
    let error = FrameImageCache::new().load(&path).unwrap_err();
    match error {
        PipelineError::Adapter { component, message } => {
            assert_eq!(component, "jpeg");
            assert!(message.contains("SOI marker"), "{message}");
        }
        other => panic!("expected an Adapter error, got {other}"),
    }
}

#[test]
fn a_frame_beyond_the_pixel_budget_is_an_adapter_error() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("huge.jpg");
    std::fs::write(&path, jpeg_header(4_096, 4_096)).unwrap();
    let error = FrameImageCache::with_max_pixels(1_024)
        .load(&path)
        .unwrap_err();
    match error {
        PipelineError::Adapter { component, message } => {
            assert_eq!(component, "jpeg");
            // ImageError::FrameTooLarge, carried across as a string.
            assert!(message.contains("16777216 pixels"), "{message}");
            assert!(message.contains("1024 pixel budget"), "{message}");
        }
        other => panic!("expected an Adapter error, got {other}"),
    }
}

#[test]
fn default_and_new_build_the_same_cache() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("frame.jpg");
    std::fs::write(&path, jpeg_header(20_000, 20_000)).unwrap();
    // 400 million pixels is over the default budget but not over a raised one,
    // so the two constructors are distinguished by the same input.
    let error = FrameImageCache::default().load(&path).unwrap_err();
    assert!(
        matches!(
            error,
            PipelineError::Adapter {
                component: "jpeg",
                ..
            }
        ),
        "{error}"
    );
    let raised = FrameImageCache::with_max_pixels(500_000_000)
        .load(&path)
        .unwrap_err();
    match raised {
        PipelineError::Adapter { component, message } => {
            assert_eq!(component, "jpeg");
            // Past the budget check, the truncated header fails in the decoder.
            assert!(!message.contains("pixel budget"), "{message}");
        }
        other => panic!("expected an Adapter error, got {other}"),
    }
}

#[test]
fn the_decoder_propagates_the_caches_io_error() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("absent.jpg");
    let decoder = JpegFrameDecoder::new(Arc::new(FrameImageCache::new()));
    let error = decoder.decode(7, &path).unwrap_err();
    assert!(
        matches!(
            error,
            PipelineError::Io {
                operation: "decode_frame",
                ..
            }
        ),
        "{error}"
    );
}

#[test]
fn the_decoder_is_a_frame_decoder_trait_object() {
    let decoder: Box<dyn FrameDecoder> =
        Box::new(JpegFrameDecoder::new(Arc::new(FrameImageCache::new())));
    let error = decoder.decode(0, Path::new("absent.jpg")).unwrap_err();
    assert!(
        matches!(
            error,
            PipelineError::Io {
                operation: "decode_frame",
                ..
            }
        ),
        "{error}"
    );
}
