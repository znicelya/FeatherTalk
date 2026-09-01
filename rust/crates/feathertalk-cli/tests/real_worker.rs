//! End-to-end coverage against the real worker binary.
//!
//! `cargo test -p feathertalk-cli` does not build `feathertalk-worker`, so there
//! is no `CARGO_BIN_EXE_feathertalk-worker`. The worker is found as a sibling of
//! this crate's own binary, which is where cargo puts every binary in the
//! workspace's shared target directory. When it is absent — a fresh clone that
//! only built this crate — each test prints why it skipped and passes, unless
//! `FEATHERTALK_REQUIRE_E2E=1` demands the real thing. CI sets that variable;
//! a developer running one crate's tests is not blocked by it.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use tempfile::TempDir;

const CLI: &str = env!("CARGO_BIN_EXE_feathertalk");

/// The variable that makes a missing worker a failure instead of a skip.
const REQUIRE_E2E: &str = "FEATHERTALK_REQUIRE_E2E";

/// Locate `feathertalk-worker` next to the CLI binary under test.
fn worker_path() -> Option<PathBuf> {
    let cli = PathBuf::from(CLI);
    let path = cli.parent()?.join(format!(
        "feathertalk-worker{}",
        std::env::consts::EXE_SUFFIX
    ));
    path.is_file().then_some(path)
}

/// The worker, or `None` after explaining the skip.
fn worker_or_skip(test: &str) -> Option<PathBuf> {
    if let Some(path) = worker_path() {
        return Some(path);
    }
    let required = std::env::var(REQUIRE_E2E).as_deref() == Ok("1");
    assert!(
        !required,
        "{REQUIRE_E2E}=1 but feathertalk-worker was not found next to {CLI}; \
         build it with `cargo build -p feathertalk-worker`"
    );
    println!(
        "skipping {test}: feathertalk-worker is not built; run `cargo build -p feathertalk-worker`"
    );
    None
}

/// Run the CLI against the real worker. `env` is applied to the CLI process,
/// which the worker inherits — that is how the worker's own configuration is
/// reached without the CLI knowing anything about it.
fn run(worker: &Path, args: &[&str], env: &[(&str, &str)]) -> Output {
    let mut command = Command::new(CLI);
    command.arg("--worker").arg(worker).args(args);
    for (key, value) in env {
        command.env(key, value);
    }
    command.output().expect("the CLI binary runs")
}

fn code(output: &Output) -> i32 {
    output.status.code().expect("the CLI exits normally")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn capabilities_reports_the_real_handshake() {
    let Some(worker) = worker_or_skip("capabilities_reports_the_real_handshake") else {
        return;
    };
    let output = run(&worker, &["capabilities"], &[]);
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));
    let text = stdout(&output);
    // `CPU_ADAPTER_ID` and the one command the worker always advertises.
    assert!(text.contains("cpu-0"), "{text}");
    assert!(text.contains("validate_project"), "{text}");
}

#[test]
fn an_invalid_manifest_is_a_task_failure() {
    let Some(worker) = worker_or_skip("an_invalid_manifest_is_a_task_failure") else {
        return;
    };
    let project = TempDir::new().expect("a temporary directory is available");
    // A manifest that parses as JSON but is not a manifest. An *absent*
    // `project.json` would be `ProjectError::Io`, which `io_error_code` maps to
    // `WORKER_CRASHED`; only a manifest the worker managed to read and reject
    // reaches `MEDIA_INVALID`, which is the mapping worth pinning here.
    std::fs::write(project.path().join("project.json"), "{}")
        .expect("the temporary manifest is writable");
    let path = project.path().to_string_lossy().into_owned();
    let output = run(&worker, &["validate-project", &path], &[]);
    assert_eq!(code(&output), 1, "stdout was: {}", stdout(&output));
    assert!(
        stderr(&output).contains("MEDIA_INVALID"),
        "the wire error code is shown verbatim: {}",
        stderr(&output)
    );
}

#[test]
fn a_missing_ffprobe_makes_probe_media_unsupported() {
    let Some(worker) = worker_or_skip("a_missing_ffprobe_makes_probe_media_unsupported") else {
        return;
    };
    // With no resolvable ffprobe the worker drops `probe_media` from
    // `supported_commands`, so the client's capability gate refuses the request
    // before any task starts.
    let output = run(
        &worker,
        &["probe-media", "clip.mp4"],
        &[(
            "FEATHERTALK_WORKER_FFPROBE",
            "this-path-does-not-exist-ffprobe",
        )],
    );
    assert_eq!(code(&output), 3, "stdout was: {}", stdout(&output));
    let text = stderr(&output);
    assert!(
        text.contains("FEATHERTALK_WORKER_FFPROBE"),
        "the advice names the variable that would fix it: {text}"
    );
}
