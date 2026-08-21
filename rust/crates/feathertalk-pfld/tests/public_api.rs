use feathertalk_pfld::{
    CropGeometry, LandmarkPoint, PFLD_LANDMARK_COUNT, PFLD_OUTPUT_VALUE_COUNT, PFLDLandmarks,
    PfldError, decode_landmarks,
};

#[test]
fn crate_root_exposes_pfld_contract() {
    let _: usize = PFLD_OUTPUT_VALUE_COUNT;
    let _: usize = PFLD_LANDMARK_COUNT;
    let _: LandmarkPoint = LandmarkPoint { x: 0, y: 0 };
    let output = decode_landmarks(
        &[0.0; PFLD_OUTPUT_VALUE_COUNT],
        &[0.0; PFLD_OUTPUT_VALUE_COUNT],
        CropGeometry {
            width: 1,
            height: 1,
            offset_x: 0,
            offset_y: 0,
        },
    )
    .unwrap();
    let _: &PFLDLandmarks = &output;
    let _: Result<PFLDLandmarks, PfldError> = Ok(output);
}
