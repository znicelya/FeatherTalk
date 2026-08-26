use feathertalk_inference::{
    BgrFrame, InferenceError, RenderGeometry, apply_unet_prediction, build_unet_image_input,
};

#[test]
fn image_input_is_bgr_channel_first_and_masks_only_the_mouth_rectangle() {
    let mut bytes = vec![64; 168 * 168 * 3];
    let pixel = (4 + 6) * 168 * 3 + (4 + 6) * 3;
    bytes[pixel..pixel + 3].copy_from_slice(&[128, 0, 255]);
    let unmasked_pixel = (4_u32 * 168 + 4) as usize * 3;
    bytes[unmasked_pixel..unmasked_pixel + 3].copy_from_slice(&[32, 64, 96]);
    let crop = BgrFrame::new(168, 168, bytes).unwrap();
    let input = build_unet_image_input(&crop, &RenderGeometry::standard()).unwrap();
    assert_eq!(input.shape(), [1, 6, 160, 160]);
    let plane = 160 * 160;
    let offset = 6 * 160 + 6;
    assert_eq!(input.as_slice()[offset], 128.0 / 255.0);
    assert_eq!(input.as_slice()[plane + offset], 0.0);
    assert_eq!(input.as_slice()[2 * plane + offset], 1.0);
    assert_eq!(input.as_slice()[3 * plane + offset], 0.0);
    assert_eq!(input.as_slice()[4 * plane + offset], 0.0);
    assert_eq!(input.as_slice()[5 * plane + offset], 0.0);
    let corner = 0;
    assert_eq!(input.as_slice()[corner], 32.0 / 255.0);
    assert_eq!(input.as_slice()[plane + corner], 64.0 / 255.0);
    assert_eq!(input.as_slice()[2 * plane + corner], 96.0 / 255.0);
    assert_eq!(input.as_slice()[3 * plane + corner], 32.0 / 255.0);
    assert_eq!(input.as_slice()[4 * plane + corner], 64.0 / 255.0);
    assert_eq!(input.as_slice()[5 * plane + corner], 96.0 / 255.0);
}

#[test]
fn prediction_is_clamped_rounded_and_keeps_the_four_pixel_border() {
    let mut crop = BgrFrame::new(168, 168, vec![7; 168 * 168 * 3]).unwrap();
    let mut prediction = vec![0.0; 3 * 160 * 160];
    prediction[0] = -1.0;
    prediction[160 * 160] = 0.5;
    prediction[2 * 160 * 160] = 2.0;
    apply_unet_prediction(&mut crop, &prediction, &RenderGeometry::standard()).unwrap();
    assert_eq!(crop.pixel(0, 0).unwrap(), [7, 7, 7]);
    assert_eq!(crop.pixel(4, 4).unwrap(), [0, 128, 255]);
}

#[test]
fn prediction_rejects_wrong_length_and_non_finite_values_before_mutation() {
    let geometry = RenderGeometry::standard();
    let mut crop = BgrFrame::new(168, 168, vec![9; 168 * 168 * 3]).unwrap();
    assert!(matches!(
        apply_unet_prediction(&mut crop, &[0.0; 3], &geometry),
        Err(InferenceError::TensorShapeMismatch { .. })
    ));
    let mut prediction = vec![0.0; 3 * 160 * 160];
    prediction[42] = f32::NAN;
    assert!(matches!(
        apply_unet_prediction(&mut crop, &prediction, &geometry),
        Err(InferenceError::NonFinitePrediction { index: 42 })
    ));
    assert_eq!(crop.pixel(4, 4).unwrap(), [9, 9, 9]);
}
