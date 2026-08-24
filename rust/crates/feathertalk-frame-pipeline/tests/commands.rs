use std::{path::PathBuf, time::Duration};

use feathertalk_frame_pipeline::{FrameExtractor, FramePipelineSpec, frame_command};

fn extractor() -> FrameExtractor {
    FrameExtractor::new(
        PathBuf::from(r"C:\bundle\ffmpeg.exe"),
        Duration::from_secs(10),
    )
    .unwrap()
}

fn spec() -> FramePipelineSpec {
    FramePipelineSpec::new(
        PathBuf::from(r"C:\media\hostile name & $() ;.mp4"),
        PathBuf::from(r"C:\project\assets"),
        2,
        640,
        480,
    )
    .unwrap()
}

#[test]
fn frame_command_uses_fixed_flags_and_native_path_arguments() {
    let value = spec();
    let command = frame_command(&extractor(), value.video_path(), 26, &value.frame_path(26));
    assert_eq!(command.operation(), "extract_frame");
    assert_eq!(command.executable(), PathBuf::from(r"C:\bundle\ffmpeg.exe"));
    assert!(
        command
            .arguments()
            .windows(2)
            .any(|pair| pair == ["-vf", "fps=25"])
    );
    assert!(
        command
            .arguments()
            .windows(2)
            .any(|pair| pair == ["-frames:v", "1"])
    );
    assert!(
        command
            .arguments()
            .windows(2)
            .any(|pair| pair == ["-start_number", "0"])
    );
    assert_eq!(
        command.arguments().last(),
        Some(&value.frame_path(26).into_os_string())
    );
    assert!(
        command
            .arguments()
            .contains(&value.video_path().as_os_str().to_owned())
    );
    assert!(command.arguments().contains(&"1.01".into()));
}
