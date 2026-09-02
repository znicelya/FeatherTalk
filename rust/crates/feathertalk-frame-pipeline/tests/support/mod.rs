#![allow(dead_code)]

use std::ffi::OsStr;

use feathertalk_frame_pipeline::CommandSpec;

/// The value ffmpeg receives after `flag`, as text.
pub fn flag_value(command: &CommandSpec, flag: &str) -> String {
    let arguments = command.arguments();
    let position = arguments
        .iter()
        .position(|argument| argument.as_os_str() == OsStr::new(flag))
        .unwrap_or_else(|| panic!("{flag} is missing from the frame command"));
    arguments
        .get(position + 1)
        .unwrap_or_else(|| panic!("{flag} carries no value"))
        .to_str()
        .unwrap_or_else(|| panic!("{flag} carries non-UTF-8 text"))
        .to_owned()
}

/// The same value parsed as a frame counter.
pub fn flag_number(command: &CommandSpec, flag: &str) -> u64 {
    flag_value(command, flag)
        .parse()
        .unwrap_or_else(|_| panic!("{flag} must carry a number"))
}
