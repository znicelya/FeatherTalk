use crate::{
    MediaError, MediaProbe, MediaToolchain, ProcessRunner, SystemProcessRunner,
    commands::probe_command,
    probe::{StreamExpectation, parse_probe_json_for},
    process::MAX_CAPTURE_BYTES,
};

pub fn probe_media_with_runner<R: ProcessRunner + ?Sized>(
    input: &crate::ValidatedInput,
    toolchain: &MediaToolchain,
    runner: &R,
) -> Result<MediaProbe, MediaError> {
    let command = probe_command(toolchain, input.source());
    let output = runner.run(&command, toolchain.timeout())?;
    if output.stdout().len() > MAX_CAPTURE_BYTES {
        return Err(MediaError::ToolOutputTooLarge {
            operation: "probe",
            stream: "stdout",
            limit: MAX_CAPTURE_BYTES,
            actual: output.stdout().len(),
        });
    }
    if output.stderr().len() > MAX_CAPTURE_BYTES {
        return Err(MediaError::ToolOutputTooLarge {
            operation: "probe",
            stream: "stderr",
            limit: MAX_CAPTURE_BYTES,
            actual: output.stderr().len(),
        });
    }
    if output.exit_code() != Some(0) {
        return Err(MediaError::ToolFailed {
            operation: "probe",
            exit_code: output.exit_code(),
            stderr: String::from_utf8_lossy(output.stderr()).into_owned(),
        });
    }
    parse_probe_json_for(output.stdout(), StreamExpectation::AudioVideo)
}

pub fn probe_media(
    input: &crate::ValidatedInput,
    toolchain: &MediaToolchain,
) -> Result<MediaProbe, MediaError> {
    probe_media_with_runner(input, toolchain, &SystemProcessRunner)
}
