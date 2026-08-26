use feathertalk_inference::{BgrFrame, InferenceError, crop_bgr, resize_bilinear};
use feathertalk_preprocess::FaceBoundingBox;

#[test]
fn crop_copies_a_left_top_inclusive_region() {
    let frame = BgrFrame::new(
        3,
        2,
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    )
    .unwrap();
    let crop = crop_bgr(
        &frame,
        &FaceBoundingBox {
            xmin: 1,
            ymin: 0,
            xmax: 3,
            ymax: 2,
        },
    )
    .unwrap();
    assert_eq!((crop.width(), crop.height()), (2, 2));
    assert_eq!(crop.as_bytes(), &[3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17]);
}

#[test]
fn crop_rejects_negative_or_outside_bbox_without_panicking() {
    let frame = BgrFrame::new(3, 2, vec![0; 18]).unwrap();
    for bbox in [
        FaceBoundingBox {
            xmin: -1,
            ymin: 0,
            xmax: 2,
            ymax: 2,
        },
        FaceBoundingBox {
            xmin: 1,
            ymin: 0,
            xmax: 4,
            ymax: 2,
        },
        FaceBoundingBox {
            xmin: 2,
            ymin: 1,
            xmax: 2,
            ymax: 2,
        },
    ] {
        assert!(matches!(
            crop_bgr(&frame, &bbox),
            Err(InferenceError::InvalidBbox { .. })
        ));
    }
}

#[test]
fn resize_bilinear_matches_half_pixel_average_and_edges() {
    let source = BgrFrame::new(2, 2, vec![0, 0, 0, 10, 10, 10, 20, 20, 20, 30, 30, 30]).unwrap();
    let one = resize_bilinear(&source, 1, 1).unwrap();
    assert_eq!(one.pixel(0, 0).unwrap(), [15, 15, 15]);
    let enlarged = resize_bilinear(&source, 3, 3).unwrap();
    assert_eq!(enlarged.pixel(0, 0).unwrap(), [25, 25, 25]);
    assert_eq!(enlarged.pixel(1, 1).unwrap(), [15, 15, 15]);
    assert_eq!(enlarged.pixel(2, 2).unwrap(), [30, 30, 30]);
}

#[test]
fn resize_rejects_zero_target_dimensions() {
    let source = BgrFrame::new(1, 1, vec![1, 2, 3]).unwrap();
    assert!(matches!(
        resize_bilinear(&source, 0, 1),
        Err(InferenceError::InvalidResizeTarget { .. })
    ));
}
