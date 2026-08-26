use std::path::{Path, PathBuf};

use feathertalk_inference::{
    InferenceFramePlan, InferenceError, PingPongFrames, RawFrameRenderSpec, RenderGeometry,
    RenderPlan, staging_output_path, validate_output_destination,
};

#[test]
fn crate_root_exposes_read_only_inference_contract() {
    let mut picker = PingPongFrames::new(2).unwrap();
    let _: usize = picker.next();
    let plan = RenderPlan::new(2, 4, Some(2)).unwrap();
    let frame: InferenceFramePlan = plan.frame(0).unwrap();
    assert_eq!(frame.reference_frame_index, frame.source_frame_index);

    let geometry = RenderGeometry::standard();
    assert_eq!(geometry.replacement_offset(), (4, 4));

    let audio = Path::new("audio.wav");
    let output = Path::new("output.mp4");
    let spec = RawFrameRenderSpec::new(640, 480, audio, output).unwrap();
    let _: &Path = spec.audio_path();
    let _: &Path = spec.output_path();
    assert_eq!(spec.fps(), 25);

    let _ = PathBuf::from(spec.output_path());
    let _ = validate_output_destination;
    let _ = staging_output_path;
    let _ = InferenceError::EmptyFeatures;
}
