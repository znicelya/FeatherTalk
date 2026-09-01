# FeatherTalk CLI Worker Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a `feathertalk` command-line binary that drives the slice 2 worker over the version 2 stdio protocol — it finds the worker executable, performs the handshake, gates commands on the worker's reported `supported_commands`, streams one task's events as Chinese lines on stderr while keeping stdout machine-readable, cancels on Ctrl-C, and maps the outcome onto four exit codes.

**Architecture:** Two new workspace members above `feathertalk-domain`. `feathertalk-client` is a library that owns the child process and the protocol: worker discovery, spawn, handshake, one task per session, cancellation, and reaping. It has no argument parsing and no terminal output. `feathertalk-cli` is a lib plus a `feathertalk` binary that owns everything user-facing: clap parsing, the Chinese stage dictionary, the two output modes, the Ctrl-C handler, and exit codes. The split is what lets milestone 5 reuse the same RPC path from the desktop shell with a different `EventSink`. Frames are read on a dedicated thread that forwards decoded results over an `mpsc` channel, so the main loop can poll a cancel flag while a blocking read is in flight.

**Tech Stack:** Rust 2024 edition, rust-version 1.92, `std::thread` and `std::sync::mpsc` (no Tokio), `clap` with `derive`, `ctrlc`, `serde`, `serde_json`, `thiserror`, `time`, `tempfile` for tests.

**Spec:** `docs/superpowers/specs/2026-09-01-feathertalk-cli-worker-client-design.md`

## Global Constraints

- Run every cargo command from `E:/workspace/github/FeatherTalk/rust` unless a step says otherwise. Run `git` from the repository root, `E:/workspace/github/FeatherTalk`.
- Per-task verification is `cargo test -p <crate> --all-targets`, `cargo clippy -p <crate> --all-targets -- -D warnings`, and `cargo fmt --all -- --check`, for each crate the task touched. If `fmt --check` reports a diff, run `cargo fmt --all` and re-run the check. Task 6 adds the full workspace gate.
- Commit after every task, with the message the task gives. Never stage or commit `demo/kanghui_training_video_featherhubert_188_latest/` — it is an untracked input fixture that must be left byte-for-byte as it is. `git status --porcelain` must still show it as `??` after every commit.
- No Tokio, no async runtime, no `burn`, no GPUI in either new crate. Threading is `std::thread` plus `std::sync::mpsc`. The one piece of shared mutable state permitted in `feathertalk-client` is the stderr tail buffer, an `Arc<Mutex<VecDeque<String>>>`; frame flow stays on channels.
- Exactly one new third-party dependency: `ctrlc`, pinned with `=` at the workspace level, used only by `feathertalk-cli`. Everything else comes from the existing `[workspace.dependencies]` table.
- Every frame is `validate()`d before it is written and after it is decoded. `FrameReader`, `FrameWriter`, `encode_line`, and `decode_line` are syntax-only by contract; semantic checking is the caller's job.
- User-facing CLI copy is Chinese. `ClientError`'s `Display` text is English; the CLI owns the Chinese rendering. Inside a `TaskError`, `summary` is Chinese and `detail` is English — the CLI prints both verbatim and translates neither.
- `serde_json` is compiled without `preserve_order` anywhere in this workspace, so `serde_json::Value::Object` is a `BTreeMap` and any round trip through `Value` reorders object keys alphabetically. `--json` must therefore print the worker's original line, never a re-serialization.
- The CLI never links an execution crate, never validates a path the worker will validate, and never injects a `FEATHERTALK_WORKER_*` variable into the child. The worker reads its own configuration through `WorkerConfig::from_env()` and inherits the CLI's environment.
- No file under `rust/crates/feathertalk-project/` is modified by this plan.
- Exit codes are fixed: 0 completed, 1 the worker reported a task failure, 2 cancelled, 3 session-level error. Clap's own parse failures must exit 3, not clap's default 2.
- The workspace baseline is 707 passing, 0 failing, 13 ignored, and the full suite takes roughly 30 minutes on a warm target directory. This slice may only increase the passing count. The 13 pre-existing ignored tests stay ignored.

---

## File Structure

```text
rust/Cargo.toml                                       Modify: members (Task 1), ctrlc (Task 5)
rust/crates/feathertalk-client/
  Cargo.toml                                          Create (Task 1), Modify: fake-worker bin (Task 2)
  src/lib.rs                                          Create (Task 1), Modify (Tasks 2, 3)
  src/error.rs                                        Create (Task 1)   ClientError and the probed-path record
  src/locator.rs                                      Create (Task 1)   worker executable discovery
  src/task_id.rs                                      Create (Task 1)   protocol-valid task id generation
  src/options.rs                                      Create (Task 2)   the three deadlines and the tail bound
  src/session.rs                                      Create (Task 2), Modify (Tasks 3, 4)
                                                                        transport, handshake, run loop, cancellation
  tests/support/fake_worker.rs                        Create (Task 2), Modify (Tasks 3, 4)
                                                                        scripted worker stand-in, one scenario per case
  tests/support/harness.rs                            Create (Task 2)   shared test helpers
  tests/discovery.rs                                  Create (Task 1)
  tests/task_id.rs                                    Create (Task 1)
  tests/handshake.rs                                  Create (Task 2)
  tests/run.rs                                        Create (Task 3)
  tests/cancel.rs                                     Create (Task 4)
rust/crates/feathertalk-cli/
  Cargo.toml                                          Create (Task 5)
  src/lib.rs                                          Create (Task 5)   crate root, re-exports, `run`
  src/main.rs                                         Create (Task 5)   argument parsing and process exit only
  src/cli.rs                                          Create (Task 5)   clap types, Chinese help
  src/render.rs                                       Create (Task 5)   stage dictionary, sinks, error copy
  src/run.rs                                          Create (Task 5)   locate, spawn, dispatch, exit code
  tests/support/fake_worker_bin.rs                    Create (Task 5)   includes the client's fake worker source
  tests/cli.rs                                        Create (Task 5)
  tests/real_worker.rs                                Create (Task 6)
docs/superpowers/specs/2026-08-17-rust-desktop-migration-design.md   Modify: section 16 (Task 6)
docs/superpowers/plans/2026-09-01-feathertalk-cli-worker-client.md   Modify: tick the boxes (every task)
```

Why these boundaries: `error.rs`, `locator.rs`, and `task_id.rs` have no I/O and no threads, so they are unit-testable on their own and land first. `session.rs` is the only file that owns a `Child`, and it grows in three passes — transport plus handshake, then the run loop, then the cancel state machine — because each pass is independently testable. In the CLI, `render.rs` is separated from `run.rs` so the entire Chinese output surface can be asserted without spawning a process, which is why the CLI is a lib plus a bin rather than a bin alone (`tools/onnx-validate` is the existing precedent for that shape).

---

### Task 1: The client crate, worker discovery, and task id generation

The CLI cannot spawn a worker it cannot find, and the error it prints when discovery fails is the single most likely thing a new user will hit. This task builds the crate skeleton, the three-source locator, and the task id generator — everything that needs no child process.

The locator rule that matters: **the first source that is *set* wins.** A `--worker` path or a `FEATHERTALK_WORKER_BIN` value that does not point at a file is an error, never a silent fall-through to the next source. Silent fallback would run a different binary than the operator asked for.

**Files:**
- Modify: `rust/Cargo.toml`
- Create: `rust/crates/feathertalk-client/Cargo.toml`
- Create: `rust/crates/feathertalk-client/src/lib.rs`
- Create: `rust/crates/feathertalk-client/src/error.rs`
- Create: `rust/crates/feathertalk-client/src/locator.rs`
- Create: `rust/crates/feathertalk-client/src/task_id.rs`
- Test: `rust/crates/feathertalk-client/tests/discovery.rs`
- Test: `rust/crates/feathertalk-client/tests/task_id.rs`

**Interfaces:**
- Consumes: `feathertalk_domain::{DomainError, TaskId}`; `TaskId::parse(&str) -> Result<TaskId, DomainError>` and `TaskId::as_str(&self) -> &str` (`TaskId` is neither `Copy` nor `Display`); `time::OffsetDateTime`.
- Produces:
  - `ClientError` with variants `WorkerNotFound { probed: Vec<ProbedPath> }`, `Spawn { path: PathBuf, source: std::io::Error }`, `Handshake { reason: String, stderr_tail: Vec<String> }`, `ProtocolVersion { expected: u32, actual: u32 }`, `Rejected { reason: String }`, `UnsupportedCommand { requested: &'static str, supported: Vec<&'static str> }`, `Protocol(DomainError)`, `Io(std::io::Error)`, `WorkerGone { status: Option<i32>, stderr_tail: Vec<String> }`, plus `ClientError::stderr_tail(&self) -> &[String]`.
  - `WorkerPathSource { CliOption, EnvVar, SiblingOfCurrentExe }` with `as_label(self) -> &'static str`.
  - `ProbedPath { source: WorkerPathSource, path: Option<PathBuf> }`.
  - `WorkerLocator::from_env(cli_option: Option<PathBuf>) -> Self`, `WorkerLocator::from_parts(cli_option: Option<PathBuf>, env_var: Option<PathBuf>, sibling: Option<PathBuf>) -> Self`, `WorkerLocator::sibling_of(exe: &Path) -> Option<PathBuf>`, `WorkerLocator::candidates(&self) -> Vec<ProbedPath>`, `WorkerLocator::resolve(&self) -> Result<PathBuf, ClientError>`.
  - `ENV_WORKER_BIN: &str = "FEATHERTALK_WORKER_BIN"`, `WORKER_FILE_STEM: &str = "feathertalk-worker"`.
  - `generate_task_id() -> Result<TaskId, DomainError>`.

- [ ] **Step 1: Register the crate in the workspace**

In `rust/Cargo.toml`, add the member immediately after `crates/feathertalk-worker` so the list stays roughly in dependency order:

```toml
  "crates/feathertalk-domain",
  "crates/feathertalk-worker",
  "crates/feathertalk-client",
  "crates/feathertalk-face",
```

- [ ] **Step 2: Create the crate manifest**

Create `rust/crates/feathertalk-client/Cargo.toml`:

```toml
[package]
name = "feathertalk-client"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[dependencies]
feathertalk-domain = { path = "../feathertalk-domain" }
serde_json = { workspace = true }
thiserror = { workspace = true }
time = { workspace = true }

[dev-dependencies]
tempfile = { workspace = true }
```

- [ ] **Step 3: Write the failing discovery tests**

Create `rust/crates/feathertalk-client/tests/discovery.rs`:

```rust
use std::path::PathBuf;

use feathertalk_client::{ClientError, WORKER_FILE_STEM, WorkerLocator, WorkerPathSource};
use tempfile::TempDir;

fn touch(dir: &TempDir, name: &str) -> PathBuf {
    let path = dir.path().join(name);
    std::fs::write(&path, b"stand-in for an executable").unwrap();
    path
}

#[test]
fn the_cli_option_outranks_the_environment_and_the_sibling() {
    let dir = TempDir::new().unwrap();
    let chosen = touch(&dir, "chosen");
    let other = touch(&dir, "other");
    let locator = WorkerLocator::from_parts(
        Some(chosen.clone()),
        Some(other.clone()),
        Some(other),
    );
    assert_eq!(locator.resolve().unwrap(), chosen);
}

#[test]
fn the_environment_variable_outranks_the_sibling() {
    let dir = TempDir::new().unwrap();
    let chosen = touch(&dir, "chosen");
    let sibling = touch(&dir, "sibling");
    let locator = WorkerLocator::from_parts(None, Some(chosen.clone()), Some(sibling));
    assert_eq!(locator.resolve().unwrap(), chosen);
}

#[test]
fn the_sibling_is_used_when_nothing_is_configured() {
    let dir = TempDir::new().unwrap();
    let sibling = touch(&dir, "sibling");
    let locator = WorkerLocator::from_parts(None, None, Some(sibling.clone()));
    assert_eq!(locator.resolve().unwrap(), sibling);
}

#[test]
fn a_configured_path_that_is_missing_is_an_error_not_a_fallback() {
    let dir = TempDir::new().unwrap();
    let sibling = touch(&dir, "sibling");
    let missing = dir.path().join("missing-worker");
    let locator = WorkerLocator::from_parts(Some(missing.clone()), None, Some(sibling));
    let error = locator.resolve().unwrap_err();
    let ClientError::WorkerNotFound { probed } = error else {
        panic!("expected WorkerNotFound, got {error:?}");
    };
    assert_eq!(probed.len(), 3);
    assert_eq!(probed[0].source, WorkerPathSource::CliOption);
    assert_eq!(probed[0].path, Some(missing));
    assert_eq!(probed[1].source, WorkerPathSource::EnvVar);
    assert_eq!(probed[1].path, None);
    assert_eq!(probed[2].source, WorkerPathSource::SiblingOfCurrentExe);
    assert!(probed[2].path.is_some());
}

#[test]
fn every_probed_source_is_reported_when_none_is_set() {
    let locator = WorkerLocator::from_parts(None, None, None);
    let error = locator.resolve().unwrap_err();
    let ClientError::WorkerNotFound { probed } = error else {
        panic!("expected WorkerNotFound, got {error:?}");
    };
    let labels: Vec<&str> = probed
        .iter()
        .map(|candidate| candidate.source.as_label())
        .collect();
    assert_eq!(
        labels,
        vec![
            "--worker",
            "FEATHERTALK_WORKER_BIN",
            "sibling of the current executable",
        ]
    );
    assert!(probed.iter().all(|candidate| candidate.path.is_none()));
}

#[test]
fn the_sibling_name_carries_the_platform_executable_suffix() {
    let exe = PathBuf::from("some").join("dir").join("feathertalk.exe");
    let sibling = WorkerLocator::sibling_of(&exe).unwrap();
    assert_eq!(sibling.parent().unwrap(), exe.parent().unwrap());
    assert_eq!(
        sibling.file_name().unwrap().to_str().unwrap(),
        format!("{WORKER_FILE_STEM}{}", std::env::consts::EXE_SUFFIX)
    );
}
```

A note on why `from_parts` exists at all: `std::env::set_var` is `unsafe` in edition 2024, so tests must not mutate the process environment. `from_env` reads the environment once and hands the three candidates to `from_parts`, which is the seam the tests drive.

- [ ] **Step 4: Write the failing task id test**

Create `rust/crates/feathertalk-client/tests/task_id.rs`:

```rust
use feathertalk_client::generate_task_id;
use feathertalk_domain::TaskId;

#[test]
fn a_generated_task_id_parses_as_a_domain_task_id() {
    let generated = generate_task_id().unwrap();
    let reparsed = TaskId::parse(generated.as_str()).unwrap();
    assert_eq!(reparsed.as_str(), generated.as_str());
    assert_eq!(generated.as_str().len(), 22);
    let (millis, suffix) = generated.as_str().split_once('-').unwrap();
    assert_eq!(millis.len(), 13);
    assert!(millis.bytes().all(|byte| byte.is_ascii_digit()));
    assert_eq!(suffix.len(), 8);
    assert!(
        suffix
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    );
}

#[test]
fn two_task_ids_generated_in_the_same_millisecond_differ() {
    let first = generate_task_id().unwrap();
    let second = generate_task_id().unwrap();
    assert_ne!(first.as_str(), second.as_str());
}
```

- [ ] **Step 5: Run both test files to verify they fail**

Run: `cargo test -p feathertalk-client --all-targets`

Expected: FAIL. Cargo cannot find the crate root, reporting `error: failed to load manifest` or `couldn't read src/lib.rs`, because `src/lib.rs` does not exist yet.

- [ ] **Step 6: Write the error types**

Create `rust/crates/feathertalk-client/src/error.rs`:

```rust
use std::path::PathBuf;

use feathertalk_domain::DomainError;

/// Where a candidate worker path came from, so a failure can name the knob the
/// operator has to turn instead of just saying "not found".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerPathSource {
    CliOption,
    EnvVar,
    SiblingOfCurrentExe,
}

impl WorkerPathSource {
    pub fn as_label(self) -> &'static str {
        match self {
            Self::CliOption => "--worker",
            Self::EnvVar => crate::ENV_WORKER_BIN,
            Self::SiblingOfCurrentExe => "sibling of the current executable",
        }
    }
}

/// One discovery attempt. `path` is `None` when the source was not set at all,
/// which reads differently to the operator than a path that was set and missed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbedPath {
    pub source: WorkerPathSource,
    pub path: Option<PathBuf>,
}

/// Everything that can go wrong between the caller and the worker process.
///
/// `Display` is English on purpose: this crate is also the desktop shell's
/// transport, and each front end renders its own user-facing copy. The CLI
/// translates these into Chinese.
#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("no worker executable was found")]
    WorkerNotFound { probed: Vec<ProbedPath> },
    #[error("failed to spawn the worker at {path}")]
    Spawn {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("the handshake failed: {reason}")]
    Handshake {
        reason: String,
        stderr_tail: Vec<String>,
    },
    #[error("protocol version mismatch: this client speaks {expected}, the worker reported {actual}")]
    ProtocolVersion { expected: u32, actual: u32 },
    #[error("the worker rejected the request: {reason}")]
    Rejected { reason: String },
    #[error("the worker does not support {requested}")]
    UnsupportedCommand {
        requested: &'static str,
        supported: Vec<&'static str>,
    },
    #[error("protocol error: {0}")]
    Protocol(#[from] DomainError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("the worker exited without reporting a terminal stage")]
    WorkerGone {
        status: Option<i32>,
        stderr_tail: Vec<String>,
    },
}

impl ClientError {
    /// The last lines the worker wrote to stderr, when they were captured.
    ///
    /// A worker that dies during startup usually explains itself here and
    /// nowhere else, so every front end wants to print this.
    pub fn stderr_tail(&self) -> &[String] {
        match self {
            Self::Handshake { stderr_tail, .. } | Self::WorkerGone { stderr_tail, .. } => {
                stderr_tail
            }
            Self::WorkerNotFound { .. }
            | Self::Spawn { .. }
            | Self::ProtocolVersion { .. }
            | Self::Rejected { .. }
            | Self::UnsupportedCommand { .. }
            | Self::Protocol(_)
            | Self::Io(_) => &[],
        }
    }
}
```

- [ ] **Step 7: Write the locator**

Create `rust/crates/feathertalk-client/src/locator.rs`:

```rust
use std::path::{Path, PathBuf};

use crate::{ClientError, ProbedPath, WorkerPathSource};

/// Environment variable naming the worker executable.
pub const ENV_WORKER_BIN: &str = "FEATHERTALK_WORKER_BIN";

/// File stem of the worker executable, without a platform suffix.
pub const WORKER_FILE_STEM: &str = "feathertalk-worker";

/// The three places a worker executable is looked for, in priority order.
#[derive(Debug, Clone, Default)]
pub struct WorkerLocator {
    cli_option: Option<PathBuf>,
    env_var: Option<PathBuf>,
    sibling: Option<PathBuf>,
}

impl WorkerLocator {
    /// Read the environment once and build the candidate list.
    pub fn from_env(cli_option: Option<PathBuf>) -> Self {
        let env_var = std::env::var_os(ENV_WORKER_BIN)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from);
        let sibling = std::env::current_exe()
            .ok()
            .as_deref()
            .and_then(Self::sibling_of);
        Self::from_parts(cli_option, env_var, sibling)
    }

    /// Test seam: the same logic with the environment supplied by the caller.
    pub fn from_parts(
        cli_option: Option<PathBuf>,
        env_var: Option<PathBuf>,
        sibling: Option<PathBuf>,
    ) -> Self {
        Self {
            cli_option,
            env_var,
            sibling,
        }
    }

    /// The worker that would sit next to `exe` in the same directory.
    pub fn sibling_of(exe: &Path) -> Option<PathBuf> {
        let directory = exe.parent()?;
        Some(directory.join(format!(
            "{WORKER_FILE_STEM}{}",
            std::env::consts::EXE_SUFFIX
        )))
    }

    /// Every source in priority order, whether or not it was set.
    pub fn candidates(&self) -> Vec<ProbedPath> {
        vec![
            ProbedPath {
                source: WorkerPathSource::CliOption,
                path: self.cli_option.clone(),
            },
            ProbedPath {
                source: WorkerPathSource::EnvVar,
                path: self.env_var.clone(),
            },
            ProbedPath {
                source: WorkerPathSource::SiblingOfCurrentExe,
                path: self.sibling.clone(),
            },
        ]
    }

    /// Resolve the worker executable.
    ///
    /// The highest-priority source that is *set* decides the outcome. A path
    /// that was configured but does not exist is an error rather than a
    /// fall-through, because silently running a different binary than the
    /// operator named is worse than failing.
    pub fn resolve(&self) -> Result<PathBuf, ClientError> {
        let candidates = self.candidates();
        let configured = candidates
            .iter()
            .find_map(|candidate| candidate.path.clone());
        match configured {
            Some(path) if path.is_file() => Ok(path),
            _ => Err(ClientError::WorkerNotFound { probed: candidates }),
        }
    }
}
```

- [ ] **Step 8: Write the task id generator**

Create `rust/crates/feathertalk-client/src/task_id.rs`:

```rust
use std::sync::atomic::{AtomicU32, Ordering};

use feathertalk_domain::{DomainError, TaskId};
use time::OffsetDateTime;

/// Bumped once per generated id so two ids minted in the same millisecond
/// cannot collide.
static COUNTER: AtomicU32 = AtomicU32::new(0);

/// 10^13, the exclusive bound of the 13-digit millisecond field.
const MILLIS_MODULUS: i128 = 10_000_000_000_000;

/// Mint a task id in the domain's wire format: thirteen decimal digits of Unix
/// milliseconds, `-`, then eight lowercase hex digits.
///
/// The generator lives here rather than in `feathertalk-domain` because the
/// format is domain-owned but the *policy* — which clock, how uniqueness is
/// obtained — is a client concern.
pub fn generate_task_id() -> Result<TaskId, DomainError> {
    let millis = (OffsetDateTime::now_utc().unix_timestamp_nanos() / 1_000_000)
        .rem_euclid(MILLIS_MODULUS);
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    let suffix = std::process::id() ^ counter.rotate_left(16);
    TaskId::parse(&format!("{millis:013}-{suffix:08x}"))
}
```

`rem_euclid` keeps the value non-negative, and `{millis:013}` pads it, so the millisecond field is always exactly thirteen digits. `suffix` is a `u32`, so `{suffix:08x}` is always exactly eight lowercase hex digits. `TaskId::parse` is still called, because the format is the domain's to enforce.

- [ ] **Step 9: Write the crate root**

Create `rust/crates/feathertalk-client/src/lib.rs`:

```rust
//! Worker protocol client.
//!
//! This crate owns the worker child process and the version 2 stdio protocol:
//! discovery, spawn, handshake, one task per session, cancellation, and
//! reaping. It performs no argument parsing and writes nothing to the
//! terminal, so both the CLI and the future desktop shell can drive the same
//! transport with their own presentation.

mod error;
mod locator;
mod task_id;

pub use error::{ClientError, ProbedPath, WorkerPathSource};
pub use locator::{ENV_WORKER_BIN, WORKER_FILE_STEM, WorkerLocator};
pub use task_id::generate_task_id;
```

- [ ] **Step 10: Run the tests to verify they pass**

Run: `cargo test -p feathertalk-client --all-targets`

Expected: PASS, 8 tests across the two integration test files.

- [ ] **Step 11: Run the linters**

Run: `cargo clippy -p feathertalk-client --all-targets -- -D warnings`
Run: `cargo fmt --all -- --check`

Expected: both exit 0. If `fmt --check` prints a diff, run `cargo fmt --all` and re-run it.

- [ ] **Step 12: Commit**

```bash
git add rust/Cargo.toml rust/crates/feathertalk-client
git commit -m "feat(client): add the client crate with worker discovery"
```

Check `git status --porcelain` first: the only untracked entry left must be `?? demo/kanghui_training_video_featherhubert_188_latest/`.

---

### Task 2: The fake worker, the transport, and the handshake

This task makes the client able to start a real child process over real pipes and complete the version 2 handshake. It also builds the test double every later task depends on: a scripted stand-in worker compiled as a `[[bin]]` of this crate, selected by a scenario name in the environment. Tasks 3 and 4 add scenarios to it; nobody mocks `Child`.

Two design points that are easy to get wrong:

- The frame reader thread **decodes but never validates**. Validation is semantic and its errors have to be attributed to a specific protocol phase, which only the main thread knows.
- `FrameReader` cannot be used for the child's stdout, even though it exists and is tested. It returns a decoded frame and drops the original text, and `--json` has to echo the worker's exact bytes. So this task writes `read_line_bounded`, which keeps the raw line and still enforces `MAX_FRAME_BYTES`.

**Files:**
- Modify: `rust/crates/feathertalk-client/Cargo.toml`
- Create: `rust/crates/feathertalk-client/src/options.rs`
- Create: `rust/crates/feathertalk-client/src/session.rs`
- Modify: `rust/crates/feathertalk-client/src/lib.rs`
- Test support: `rust/crates/feathertalk-client/tests/support/fake_worker.rs`
- Test support: `rust/crates/feathertalk-client/tests/support/harness.rs`
- Test: `rust/crates/feathertalk-client/tests/handshake.rs`

**Interfaces:**
- Consumes: `crate::{ClientError, SessionOptions}`; `feathertalk_domain::{ClientFrame, DomainError, FrameReader, FrameWriter, MAX_FRAME_BYTES, PROTOCOL_VERSION, ReadyFrame, ServerFrame, decode_line, encode_line}`; `std::process::{Child, Command, Stdio}`; `std::sync::mpsc`.
- Produces:
  - `SessionOptions { handshake_timeout, cancel_grace, shutdown_grace, stderr_tail_lines }` with `Default` (30 s / 10 s / 5 s / 20 lines).
  - `FrameLine { raw: String, frame: ServerFrame }` — a decoded frame paired with the bytes it came from.
  - `WorkerSession::spawn(&Path, SessionOptions) -> Result<Self, ClientError>`, `WorkerSession::spawn_with_env(&Path, SessionOptions, &[(String, String)])`, `WorkerSession::ready(&self) -> &ReadyFrame`, `WorkerSession::ready_raw(&self) -> &str`.
  - Private: `Transport`, `FrameEvent { Frame(FrameLine), Timeout, Eof }`, `read_line_bounded`, `spawn_reader`, `spawn_stderr_pump`.
  - Test-only: the `feathertalk-fake-worker` binary and the `harness` module.

- [ ] **Step 1: Declare the fake worker binary**

Append to `rust/crates/feathertalk-client/Cargo.toml`:

```toml
[[bin]]
name = "feathertalk-fake-worker"
path = "tests/support/fake_worker.rs"
```

No `required-features`: `CARGO_BIN_EXE_feathertalk-fake-worker` is only defined for integration tests if the binary is always built. The binary lives under `tests/support/` so `cargo` does not treat it as a test target, and `#[path]` inclusion (Step 3) keeps it out of the library.

A `[[bin]]` target cannot use dev-dependencies, so the fake worker may only use `feathertalk-domain`, `serde_json`, and std. That is why the next step hard-codes timestamps instead of reading a clock.

- [ ] **Step 2: Write the fake worker**

Create `rust/crates/feathertalk-client/tests/support/fake_worker.rs`:

```rust
//! A scripted stand-in for `feathertalk-worker`.
//!
//! Compiled as a `[[bin]]` of `feathertalk-client` so integration tests spawn a
//! real process and talk to it over real pipes. The behaviour is chosen by
//! `FT_FAKE_WORKER_SCENARIO`; one scenario per test case, each one a straight
//! line of writes with no branching, so a failing test names its own script.
//!
//! Only `feathertalk-domain`, `serde_json`, and std are available here, because
//! a `[[bin]]` target cannot use dev-dependencies. Hence the fixed timestamp.

use std::io::{BufReader, StdinLock, Write};
use std::time::Duration;

use feathertalk_domain::{
    AdapterInfo, AdapterKind, Backend, Capabilities, ClientFrame, Event, FrameReader,
    PROTOCOL_VERSION, Progress, ReadyFrame, RejectedFrame, ServerFrame, TaskId, TaskKind,
    TaskStage, encode_line,
};

/// The scenario selector. Tests set it; there is no command line.
const SCENARIO_ENV: &str = "FT_FAKE_WORKER_SCENARIO";

/// A fixed RFC 3339 instant. `Event::validate` only checks the format.
const EMITTED_AT: &str = "2026-09-01T00:00:00Z";

/// A well-formed task id that is never the one the client sends.
const FOREIGN_TASK_ID: &str = "1787900000000-0000beef";

type Reader = FrameReader<BufReader<StdinLock<'static>>>;

fn main() {
    let scenario =
        std::env::var(SCENARIO_ENV).unwrap_or_else(|_| "ready-complete".to_string());
    let mut reader = FrameReader::new(BufReader::new(std::io::stdin().lock()));
    match scenario.as_str() {
        // Never writes anything, so the handshake has to time out.
        "silent" => park(),
        // A syntactically valid frame that is not `ready`.
        "no-ready" => {
            let task_id = TaskId::parse(FOREIGN_TASK_ID).expect("the constant id is valid");
            write_frame(&ServerFrame::Event(stage_event(&task_id, TaskStage::Queued)));
            park();
        }
        // Truncated JSON: decodable as a line, not as a frame.
        "invalid-line" => {
            write_line("{\"frame\":\"ready\"");
            park();
        }
        // A structurally valid `ready` frame from a future protocol.
        "bad-version" => {
            let mut value =
                serde_json::to_value(ready(default_commands())).expect("ready serializes");
            value["data"]["protocol_version"] = serde_json::json!(99);
            write_line(&serde_json::to_string(&value).expect("the patched value serializes"));
            park();
        }
        // A worker that advertises no commands at all: `ReadyFrame::validate` rejects it.
        "empty-commands" => {
            let mut value =
                serde_json::to_value(ready(default_commands())).expect("ready serializes");
            value["data"]["supported_commands"] = serde_json::json!([]);
            write_line(&serde_json::to_string(&value).expect("the patched value serializes"));
            park();
        }
        // Refuses the session outright instead of going ready.
        "rejected-handshake" => {
            write_frame(&ServerFrame::Rejected(RejectedFrame {
                protocol_version: PROTOCOL_VERSION,
                reason: "工作进程当前无法接受新会话".to_string(),
            }));
        }
        // Goes ready and then stops reading, so only a kill reaps it.
        "hang-after-ready" => {
            write_frame(&ready(default_commands()));
            park();
        }
        // Floods stderr before going ready, to exercise the tail bound.
        "noisy-stderr" => {
            for index in 0..200 {
                eprintln!("stderr line {index}");
            }
            write_frame(&ready(default_commands()));
            serve_one_task(&mut reader);
        }
        // The happy path: ready, one progress event, then completed.
        "ready-complete" => {
            write_frame(&ready(default_commands()));
            serve_one_task(&mut reader);
        }
        other => {
            eprintln!("unknown fake worker scenario: {other}");
            std::process::exit(97);
        }
    }
}

/// Both task commands the real worker offers when a media toolchain resolved.
fn default_commands() -> Vec<TaskKind> {
    vec![TaskKind::ValidateProject, TaskKind::ProbeMedia]
}

fn ready(commands: Vec<TaskKind>) -> ServerFrame {
    ServerFrame::Ready(ReadyFrame {
        protocol_version: PROTOCOL_VERSION,
        worker_version: "fake-0".to_string(),
        backends: vec![Backend::Cpu],
        adapters: vec![AdapterInfo {
            id: "cpu-0".to_string(),
            name: "Fake CPU".to_string(),
            backend: Backend::Cpu,
            kind: AdapterKind::Cpu,
            certified: true,
            vram_bytes: None,
        }],
        supported_commands: commands,
        capabilities: Capabilities {
            training: false,
            wgpu_training: false,
            onnx_validation: false,
            ffmpeg: true,
        },
    })
}

/// Read frames until a `start` arrives, then run the scripted happy path.
fn serve_one_task(reader: &mut Reader) {
    let Some(task_id) = wait_for_start(reader) else {
        return;
    };
    let mut preparing = stage_event(&task_id, TaskStage::Preparing);
    preparing.progress = Some(Progress {
        completed: 1,
        total: Some(2),
    });
    write_frame(&ServerFrame::Event(preparing));
    write_frame(&ServerFrame::Event(completed(&task_id)));
}

/// Block until the client sends `start`. Returns `None` on shutdown or EOF.
fn wait_for_start(reader: &mut Reader) -> Option<TaskId> {
    loop {
        match reader.read_frame::<ClientFrame>()? {
            Ok(ClientFrame::Start(start)) => return Some(start.task_id),
            Ok(ClientFrame::Cancel(_)) => continue,
            Ok(ClientFrame::Shutdown(_)) => return None,
            Err(error) => {
                eprintln!("fake worker could not decode a client frame: {error}");
                return None;
            }
        }
    }
}

fn stage_event(task_id: &TaskId, stage: TaskStage) -> Event {
    Event::new(task_id.clone(), EMITTED_AT, stage)
}

fn completed(task_id: &TaskId) -> Event {
    let mut event = stage_event(task_id, TaskStage::Completed);
    event.result = Some(serde_json::json!({ "checked": true }));
    event
}

fn write_frame(frame: &ServerFrame) {
    let line = encode_line(frame).expect("the scripted frame serializes");
    write_line(line.trim_end());
}

fn write_line(line: &str) {
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(line.as_bytes()).expect("stdout accepts a line");
    stdout.write_all(b"\n").expect("stdout accepts a newline");
    stdout.flush().expect("stdout flushes");
}

/// Stay alive until the parent kills us. Several scenarios exist to prove the
/// client's deadlines and reaping actually work.
fn park() -> ! {
    loop {
        std::thread::sleep(Duration::from_millis(50));
    }
}
```

`encode_line` already appends the newline; `trim_end()` makes `write_line` the single place a line ending is chosen, so the two write paths cannot disagree.

- [ ] **Step 3: Write the shared test harness**

Create `rust/crates/feathertalk-client/tests/support/harness.rs`:

```rust
//! Helpers shared by this crate's integration tests.
//!
//! Included with `#[path = "support/harness.rs"] mod harness;` rather than
//! compiled as its own test target, so each test binary gets its own copy.
//! That is why the whole module allows dead code: no single test uses all of it.

#![allow(dead_code)]

use std::path::PathBuf;
use std::time::Duration;

use feathertalk_client::SessionOptions;

/// Cargo builds the fake worker before the test binary and hands us its path.
pub const FAKE_WORKER: &str = env!("CARGO_BIN_EXE_feathertalk-fake-worker");

pub fn fake_worker() -> PathBuf {
    PathBuf::from(FAKE_WORKER)
}

/// The environment that selects one fake worker scenario.
pub fn scenario(name: &str) -> Vec<(String, String)> {
    vec![("FT_FAKE_WORKER_SCENARIO".to_string(), name.to_string())]
}

/// Production deadlines are seconds long. Tests use these instead so a case
/// that is *supposed* to hit a deadline finishes in well under a second.
pub fn fast_options() -> SessionOptions {
    SessionOptions {
        handshake_timeout: Duration::from_millis(800),
        cancel_grace: Duration::from_millis(200),
        shutdown_grace: Duration::from_millis(200),
        stderr_tail_lines: 20,
    }
}
```

- [ ] **Step 4: Write the failing handshake tests**

Create `rust/crates/feathertalk-client/tests/handshake.rs`:

```rust
#[path = "support/harness.rs"]
mod harness;

use std::time::{Duration, Instant};

use feathertalk_client::{ClientError, WorkerSession};
use feathertalk_domain::TaskKind;

use harness::{fake_worker, fast_options, scenario};

fn spawn(name: &str) -> Result<WorkerSession, ClientError> {
    WorkerSession::spawn_with_env(&fake_worker(), fast_options(), &scenario(name))
}

#[test]
fn a_healthy_worker_completes_the_handshake() {
    let session = spawn("ready-complete").expect("the handshake succeeds");
    assert_eq!(session.ready().worker_version, "fake-0");
    assert_eq!(
        session.ready().supported_commands,
        vec![TaskKind::ValidateProject, TaskKind::ProbeMedia]
    );
    assert_eq!(session.ready().adapters.len(), 1);
    assert!(
        session.ready_raw().contains("\"frame\":\"ready\""),
        "the raw line is kept verbatim for --json: {}",
        session.ready_raw()
    );
}

#[test]
fn an_event_before_ready_fails_the_handshake() {
    let error = spawn("no-ready").expect_err("an event is not a handshake");
    let ClientError::Handshake { reason, .. } = error else {
        panic!("expected a handshake error, got {error:?}");
    };
    assert!(
        reason.contains("1787900000000-0000beef"),
        "the reason names the offending task: {reason}"
    );
}

#[test]
fn an_undecodable_first_line_fails_the_handshake() {
    let error = spawn("invalid-line").expect_err("a truncated frame is not a handshake");
    assert!(
        matches!(error, ClientError::Handshake { .. }),
        "expected a handshake error, got {error:?}"
    );
}

#[test]
fn a_rejected_frame_surfaces_the_worker_reason() {
    let error = spawn("rejected-handshake").expect_err("a rejection is not a handshake");
    let ClientError::Rejected { reason } = error else {
        panic!("expected a rejection, got {error:?}");
    };
    assert_eq!(reason, "工作进程当前无法接受新会话");
}

#[test]
fn a_future_protocol_version_is_reported_precisely() {
    let error = spawn("bad-version").expect_err("version 99 is not supported");
    assert!(
        matches!(
            error,
            ClientError::ProtocolVersion {
                expected: 2,
                actual: 99
            }
        ),
        "expected a protocol version mismatch, got {error:?}"
    );
}

#[test]
fn a_worker_with_no_commands_fails_the_handshake() {
    let error = spawn("empty-commands").expect_err("an empty command list is invalid");
    assert!(
        matches!(error, ClientError::Handshake { .. }),
        "expected a handshake error, got {error:?}"
    );
}

#[test]
fn a_silent_worker_times_out_instead_of_hanging() {
    let started = Instant::now();
    let error = spawn("silent").expect_err("silence is not a handshake");
    assert!(
        matches!(error, ClientError::Handshake { .. }),
        "expected a handshake error, got {error:?}"
    );
    assert!(
        started.elapsed() < Duration::from_secs(4),
        "the 800 ms handshake deadline was not honoured: {:?}",
        started.elapsed()
    );
}

#[test]
fn the_stderr_tail_is_bounded() {
    let session = spawn("noisy-stderr").expect("the handshake succeeds");
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut tail = session.stderr_tail();
    while tail.len() < 20 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
        tail = session.stderr_tail();
    }
    assert_eq!(tail.len(), 20, "the tail keeps exactly the configured bound");
    assert_eq!(tail.first().map(String::as_str), Some("stderr line 180"));
    assert_eq!(tail.last().map(String::as_str), Some("stderr line 199"));
}

#[test]
fn dropping_a_session_reaps_a_hung_worker() {
    let started = Instant::now();
    let session = spawn("hang-after-ready").expect("the handshake succeeds");
    drop(session);
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "drop must kill and join rather than wait for a worker that never exits: {:?}",
        started.elapsed()
    );
}
```

`stderr_tail()` on `WorkerSession` is needed by the noisy test, so Step 6 exposes it as a thin delegate to the transport.

Run: `cargo test -p feathertalk-client --test handshake`

Expected: FAIL to compile — `WorkerSession` does not exist yet.

- [ ] **Step 5: Write the session options**

Create `rust/crates/feathertalk-client/src/options.rs`:

```rust
use std::time::Duration;

/// The deadlines and bounds of one worker session.
///
/// Every wait in this crate is bounded by one of these, so a misbehaving worker
/// can never hang the caller. Tests shorten them; the CLI uses the defaults.
#[derive(Debug, Clone)]
pub struct SessionOptions {
    /// How long the worker has to send `ready` after it is spawned.
    pub handshake_timeout: Duration,
    /// How long the worker has to react to `cancel` before it is killed.
    pub cancel_grace: Duration,
    /// How long the worker has to exit after `shutdown` before it is killed.
    pub shutdown_grace: Duration,
    /// How many of the worker's most recent stderr lines to keep for reports.
    pub stderr_tail_lines: usize,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            handshake_timeout: Duration::from_secs(30),
            cancel_grace: Duration::from_secs(10),
            shutdown_grace: Duration::from_secs(5),
            stderr_tail_lines: 20,
        }
    }
}
```

- [ ] **Step 6: Write the transport and the handshake**

Create `rust/crates/feathertalk-client/src/session.rs`:

```rust
//! The worker child process and the version 2 protocol.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use feathertalk_domain::{
    ClientFrame, DomainError, FrameWriter, MAX_FRAME_BYTES, ReadyFrame, ServerFrame, decode_line,
};

use crate::{ClientError, SessionOptions};

/// A decoded frame together with the exact line it was decoded from.
///
/// `--json` must reprint the worker's own bytes: this workspace compiles
/// `serde_json` without `preserve_order`, so any round trip through `Value`
/// would silently reorder object keys.
#[derive(Debug, Clone)]
pub struct FrameLine {
    pub raw: String,
    pub frame: ServerFrame,
}

/// What one bounded read of the frame channel produced.
enum FrameEvent {
    Frame(FrameLine),
    Timeout,
    Eof,
}

/// Read one newline-terminated line, refusing to buffer past `MAX_FRAME_BYTES`.
///
/// `FrameReader` is not usable here: it hands back a decoded frame and discards
/// the text. An over-long line is drained to its newline before the error is
/// reported, so the stream stays framed.
fn read_line_bounded<R: BufRead>(reader: &mut R) -> Option<Result<String, ClientError>> {
    let mut buffer: Vec<u8> = Vec::new();
    let mut discarded = false;
    loop {
        // The `fill_buf` borrow must end before `consume`, so copy inside.
        let (consumed, finished) = {
            let available = match reader.fill_buf() {
                Ok(available) => available,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Some(Err(ClientError::Io(error))),
            };
            if available.is_empty() {
                break;
            }
            let (chunk, consumed, finished) =
                match available.iter().position(|byte| *byte == b'\n') {
                    Some(index) => (&available[..index], index + 1, true),
                    None => (available, available.len(), false),
                };
            if buffer.len() + chunk.len() > MAX_FRAME_BYTES {
                discarded = true;
            }
            if !discarded {
                buffer.extend_from_slice(chunk);
            }
            (consumed, finished)
        };
        reader.consume(consumed);
        if finished {
            return Some(finish_line(buffer, discarded));
        }
    }
    if buffer.is_empty() && !discarded {
        return None;
    }
    Some(finish_line(buffer, discarded))
}

fn finish_line(buffer: Vec<u8>, discarded: bool) -> Result<String, ClientError> {
    if discarded {
        return Err(ClientError::Protocol(DomainError::FrameTooLong {
            limit: MAX_FRAME_BYTES,
        }));
    }
    Ok(String::from_utf8_lossy(&buffer)
        .trim_end_matches('\r')
        .to_string())
}

/// Move the worker's stdout onto its own thread.
///
/// The thread decodes but deliberately does **not** validate: validation errors
/// have to be attributed to a protocol phase, and only the main loop knows which
/// phase it is in. The thread stops after forwarding an error, because a stream
/// that has lost framing cannot be resynchronised.
fn spawn_reader(
    stdout: ChildStdout,
) -> (Receiver<Result<FrameLine, ClientError>>, JoinHandle<()>) {
    let (sender, receiver) = channel();
    let handle = std::thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        while let Some(line) = read_line_bounded(&mut reader) {
            let message = match line {
                Ok(raw) if raw.trim().is_empty() => continue,
                Ok(raw) => match decode_line::<ServerFrame>(&raw) {
                    Ok(frame) => Ok(FrameLine { raw, frame }),
                    Err(error) => Err(ClientError::Protocol(error)),
                },
                Err(error) => Err(error),
            };
            let fatal = message.is_err();
            if sender.send(message).is_err() || fatal {
                break;
            }
        }
    });
    (receiver, handle)
}

/// Drain the worker's stderr into a bounded ring so a failure report can quote
/// the last few lines. Without this pump a chatty worker would block on a full
/// pipe while the client waited for a frame that could never arrive.
fn spawn_stderr_pump(
    stderr: ChildStderr,
    tail: Arc<Mutex<VecDeque<String>>>,
    limit: usize,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            let Ok(line) = line else {
                break;
            };
            if limit == 0 {
                continue;
            }
            let mut guard = tail.lock().expect("the stderr tail mutex is not poisoned");
            while guard.len() >= limit {
                guard.pop_front();
            }
            guard.push_back(line);
        }
    })
}

/// The child process and its three pipes.
struct Transport {
    child: Child,
    /// `None` once stdin has been closed or has failed; both mean the same to
    /// the worker, which exits on EOF.
    writer: Option<FrameWriter<ChildStdin>>,
    frames: Receiver<Result<FrameLine, ClientError>>,
    reader_thread: Option<JoinHandle<()>>,
    stderr_thread: Option<JoinHandle<()>>,
    stderr_tail: Arc<Mutex<VecDeque<String>>>,
    options: SessionOptions,
}

impl Transport {
    fn spawn(
        path: &Path,
        options: SessionOptions,
        env: &[(String, String)],
    ) -> Result<Self, ClientError> {
        let mut command = Command::new(path);
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (key, value) in env {
            command.env(key, value);
        }
        let mut child = command.spawn().map_err(|source| ClientError::Spawn {
            path: path.to_path_buf(),
            source,
        })?;
        let stdin = child.stdin.take().expect("stdin was piped");
        let stdout = child.stdout.take().expect("stdout was piped");
        let stderr = child.stderr.take().expect("stderr was piped");
        let stderr_tail = Arc::new(Mutex::new(VecDeque::new()));
        let stderr_thread =
            spawn_stderr_pump(stderr, Arc::clone(&stderr_tail), options.stderr_tail_lines);
        let (frames, reader_thread) = spawn_reader(stdout);
        Ok(Self {
            child,
            writer: Some(FrameWriter::new(stdin)),
            frames,
            reader_thread: Some(reader_thread),
            stderr_thread: Some(stderr_thread),
            stderr_tail,
            options,
        })
    }

    fn stderr_tail(&self) -> Vec<String> {
        self.stderr_tail
            .lock()
            .map(|guard| guard.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Validate, then write. A write failure means the worker is gone, which is
    /// a more useful diagnosis than the raw broken-pipe error.
    fn write_frame(&mut self, frame: &ClientFrame) -> Result<(), ClientError> {
        frame.validate().map_err(ClientError::Protocol)?;
        if self.writer.is_none() {
            return Err(self.worker_gone());
        }
        let outcome = self
            .writer
            .as_mut()
            .expect("checked immediately above")
            .write_frame(frame);
        if outcome.is_err() {
            self.writer = None;
            return Err(self.worker_gone());
        }
        Ok(())
    }

    /// Build the `WorkerGone` report, giving the child a short grace period so
    /// the exit status is usually available.
    fn worker_gone(&mut self) -> ClientError {
        // Copy the deadline out first: `wait_for_exit` needs `&mut self`.
        let grace = self.options.shutdown_grace;
        let status = self.wait_for_exit(grace);
        ClientError::WorkerGone {
            status,
            stderr_tail: self.stderr_tail(),
        }
    }

    /// Poll for exit up to `timeout`. `None` means still running or unknown.
    fn wait_for_exit(&mut self, timeout: Duration) -> Option<i32> {
        let deadline = Instant::now() + timeout;
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => return status.code(),
                Ok(None) => {}
                Err(_) => return None,
            }
            if Instant::now() >= deadline {
                return None;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Close stdin. The worker treats EOF as `shutdown`.
    fn close_stdin(&mut self) {
        self.writer = None;
    }

    fn kill_and_reap(&mut self) -> Option<i32> {
        self.writer = None;
        let _ = self.child.kill();
        self.child.wait().ok().and_then(|status| status.code())
    }

    fn next_frame(&self, timeout: Duration) -> Result<FrameEvent, ClientError> {
        match self.frames.recv_timeout(timeout) {
            Ok(Ok(line)) => Ok(FrameEvent::Frame(line)),
            Ok(Err(error)) => Err(error),
            Err(RecvTimeoutError::Timeout) => Ok(FrameEvent::Timeout),
            // The sender is dropped when the reader thread sees EOF.
            Err(RecvTimeoutError::Disconnected) => Ok(FrameEvent::Eof),
        }
    }
}

impl Drop for Transport {
    /// Never wait on a worker that may never exit: close stdin, kill, reap, then
    /// join the two threads, which end as soon as their pipes close.
    fn drop(&mut self) {
        self.writer = None;
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(handle) = self.reader_thread.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.stderr_thread.take() {
            let _ = handle.join();
        }
    }
}

/// One worker process that has completed the handshake.
///
/// A session runs at most one task. That is the protocol's rule, not a
/// limitation of this type.
pub struct WorkerSession {
    transport: Transport,
    ready: ReadyFrame,
    ready_raw: String,
    foreign_events: usize,
}

impl WorkerSession {
    pub fn spawn(path: &Path, options: SessionOptions) -> Result<Self, ClientError> {
        Self::spawn_with_env(path, options, &[])
    }

    /// Spawn with extra environment variables. Only tests use this: the CLI
    /// never injects configuration into the worker, which reads its own.
    pub fn spawn_with_env(
        path: &Path,
        options: SessionOptions,
        env: &[(String, String)],
    ) -> Result<Self, ClientError> {
        let handshake_timeout = options.handshake_timeout;
        let transport = Transport::spawn(path, options, env)?;
        let line = match transport.next_frame(handshake_timeout) {
            Ok(FrameEvent::Frame(line)) => line,
            Ok(FrameEvent::Timeout) => {
                return Err(ClientError::Handshake {
                    reason: format!(
                        "no ready frame within {} ms",
                        handshake_timeout.as_millis()
                    ),
                    stderr_tail: transport.stderr_tail(),
                });
            }
            Ok(FrameEvent::Eof) => {
                return Err(ClientError::Handshake {
                    reason: "the worker closed its output before sending a ready frame"
                        .to_string(),
                    stderr_tail: transport.stderr_tail(),
                });
            }
            Err(ClientError::Protocol(error)) => {
                return Err(ClientError::Handshake {
                    reason: format!("the first line was not a decodable frame: {error}"),
                    stderr_tail: transport.stderr_tail(),
                });
            }
            Err(error) => return Err(error),
        };
        // A partial move of `line.frame` is fine: `FrameLine` has no `Drop`.
        match line.frame {
            ServerFrame::Ready(ready) => {
                if let Err(error) = ready.validate() {
                    return Err(match error {
                        DomainError::ProtocolVersion { expected, actual } => {
                            ClientError::ProtocolVersion { expected, actual }
                        }
                        other => ClientError::Handshake {
                            reason: format!("the ready frame is not usable: {other}"),
                            stderr_tail: transport.stderr_tail(),
                        },
                    });
                }
                Ok(Self {
                    transport,
                    ready,
                    ready_raw: line.raw,
                    foreign_events: 0,
                })
            }
            ServerFrame::Rejected(rejected) => Err(ClientError::Rejected {
                reason: rejected.reason,
            }),
            ServerFrame::Event(event) => Err(ClientError::Handshake {
                reason: format!(
                    "the worker sent an event for task {} before its ready frame",
                    event.task_id.as_str()
                ),
                stderr_tail: transport.stderr_tail(),
            }),
        }
    }

    /// The validated handshake frame: backends, adapters, capabilities, and the
    /// command list the capability gate consults.
    pub fn ready(&self) -> &ReadyFrame {
        &self.ready
    }

    /// The handshake line exactly as the worker wrote it, for `--json`.
    pub fn ready_raw(&self) -> &str {
        &self.ready_raw
    }

    /// The worker's most recent stderr lines, oldest first.
    pub fn stderr_tail(&self) -> Vec<String> {
        self.transport.stderr_tail()
    }
}
```

`foreign_events` is written here and read in Task 3. Until then `cargo clippy -D warnings` will flag it as never read, so keep Steps 6 and 7 in the same commit and expect the field's first real use in Task 3; if clippy complains before then, the fix is to finish Task 3, not to add an allow.

- [ ] **Step 7: Wire the new modules into the crate root**

Rewrite the module block of `rust/crates/feathertalk-client/src/lib.rs`:

```rust
mod error;
mod locator;
mod options;
mod session;
mod task_id;

pub use error::{ClientError, ProbedPath, WorkerPathSource};
pub use locator::{ENV_WORKER_BIN, WORKER_FILE_STEM, WorkerLocator};
pub use options::SessionOptions;
pub use session::{FrameLine, WorkerSession};
pub use task_id::generate_task_id;
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `cargo test -p feathertalk-client --all-targets`

Expected: PASS, 17 tests (8 from Task 1, 9 handshake tests). The whole file should finish in a few seconds: only `a_silent_worker_times_out_instead_of_hanging` waits on a deadline, and that one is 800 ms.

If a test hangs instead of failing, the bug is a missing bound, not a slow machine. Check that `next_frame` is called with a timeout and that `Drop` kills before it waits.

- [ ] **Step 9: Run the linters**

Run: `cargo clippy -p feathertalk-client --all-targets -- -D warnings`
Run: `cargo fmt --all -- --check`

Expected: both exit 0.

- [ ] **Step 10: Commit**

```bash
git add rust/crates/feathertalk-client
git commit -m "feat(client): spawn the worker and complete the handshake"
```

---

### Task 3: The capability gate and the run loop

A session now runs exactly one task and reports one of four outcomes. Two rules carry most of the weight:

- **The gate is the worker's own answer.** `supported_commands` from the handshake decides what may be sent. The client never guesses from the platform, an environment variable, or a probe of its own; if `probe_media` is missing it is because the worker could not resolve a media toolchain, and that is the worker's fact to report.
- **A frame for another task is ignored, not fatal.** The protocol allows a worker to be chatty. Counting foreign events instead of dropping them silently keeps the ignore rule observable in a test.

`SessionOutcome` deliberately carries a `SessionError` variant instead of the whole method returning `Result`. A task that failed (exit 1) and a session that broke (exit 3) are different events for the user, and modelling them as one enum makes the CLI's exit-code mapping total.

**Files:**
- Modify: `rust/crates/feathertalk-client/src/session.rs`
- Modify: `rust/crates/feathertalk-client/src/lib.rs`
- Modify: `rust/crates/feathertalk-client/tests/support/fake_worker.rs`
- Test: `rust/crates/feathertalk-client/tests/run.rs`

**Interfaces:**
- Consumes: `feathertalk_domain::{Event, PROTOCOL_VERSION, RejectedFrame, Request, ShutdownFrame, StartFrame, TaskError, TaskId, TaskKind, TaskStage}`; `serde_json::Value`; `std::sync::atomic::{AtomicUsize, Ordering}`.
- Produces:
  - `trait EventSink { fn on_event(&mut self, event: &Event, raw: &str); fn on_rejected(&mut self, rejected: &RejectedFrame, raw: &str) { .. } }`.
  - `enum SessionOutcome { Completed { result: Option<Value> }, Failed(TaskError), Cancelled, SessionError(ClientError) }`.
  - `CancelToken` with `new()`, `request()`, `count()` — defined here, honoured in Task 4.
  - `WorkerSession::run(&mut self, TaskId, Request, &CancelToken, &mut dyn EventSink) -> SessionOutcome`, `WorkerSession::foreign_event_count(&self) -> usize`, `WorkerSession::shutdown(self) -> Option<i32>`.
  - New fake worker scenarios: `only-validate`, `only-probe`, `fail`, `self-cancel`, `foreign-event`, `die-after-ready`, `oversized-line`.

- [ ] **Step 1: Extend the fake worker**

In `rust/crates/feathertalk-client/tests/support/fake_worker.rs`, add to the import list: `ErrorCode`, `MAX_FRAME_BYTES`, `TaskError`.

Add these arms immediately before the `other =>` arm:

```rust
        // Advertises one command each, to exercise the capability gate both ways.
        "only-validate" => {
            write_frame(&ready(vec![TaskKind::ValidateProject]));
            serve_one_task(&mut reader);
        }
        "only-probe" => {
            write_frame(&ready(vec![TaskKind::ProbeMedia]));
            serve_one_task(&mut reader);
        }
        // Reports a task failure, which is exit 1 rather than a broken session.
        "fail" => {
            write_frame(&ready(default_commands()));
            if let Some(task_id) = wait_for_start(&mut reader) {
                write_frame(&ServerFrame::Event(failed(&task_id)));
            }
        }
        // Reports itself cancelled without being asked, so the terminal-stage
        // mapping can be tested without involving a signal.
        "self-cancel" => {
            write_frame(&ready(default_commands()));
            if let Some(task_id) = wait_for_start(&mut reader) {
                write_frame(&ServerFrame::Event(stage_event(
                    &task_id,
                    TaskStage::Cancelled,
                )));
            }
        }
        // Emits an event for a different task before the real one.
        "foreign-event" => {
            write_frame(&ready(default_commands()));
            if let Some(task_id) = wait_for_start(&mut reader) {
                let foreign = TaskId::parse(FOREIGN_TASK_ID).expect("the constant id is valid");
                write_frame(&ServerFrame::Event(stage_event(
                    &foreign,
                    TaskStage::Preparing,
                )));
                write_frame(&ServerFrame::Event(completed(&task_id)));
            }
        }
        // Exits immediately after the handshake: the client must diagnose a lost
        // worker whether the `start` write succeeds or fails.
        "die-after-ready" => {
            write_frame(&ready(default_commands()));
            std::process::exit(0);
        }
        // Writes a line past the protocol's frame bound.
        "oversized-line" => {
            write_frame(&ready(default_commands()));
            if wait_for_start(&mut reader).is_some() {
                write_line(&"x".repeat(MAX_FRAME_BYTES + 16));
                park();
            }
        }
```

And add the failure builder next to `completed`:

```rust
fn failed(task_id: &TaskId) -> Event {
    // `TaskError::stage` records where the task was when it broke, so it must be
    // the non-terminal stage; the event's own stage is the terminal `Failed`.
    let error = TaskError::new(
        ErrorCode::MediaInvalid,
        "输入文件无法解析，请确认它是完整的视频",
        "ffprobe exited with status 1",
        TaskStage::Preparing,
    );
    let mut event = stage_event(
        task_id,
        TaskStage::Failed {
            code: error.code,
            message: error.summary.clone(),
        },
    );
    event.error = Some(error);
    event
}
```

- [ ] **Step 2: Write the failing run tests**

Create `rust/crates/feathertalk-client/tests/run.rs`:

```rust
#[path = "support/harness.rs"]
mod harness;

use std::path::PathBuf;

use feathertalk_client::{
    CancelToken, ClientError, EventSink, SessionOutcome, WorkerSession, generate_task_id,
};
use feathertalk_domain::{DomainError, ErrorCode, Event, ProbeMediaParams, ProjectDirParams, Request};

use harness::{fake_worker, fast_options, scenario};

/// Records what the session reported, in order.
#[derive(Default)]
struct Collected {
    stages: Vec<String>,
    raw: Vec<String>,
}

impl EventSink for Collected {
    fn on_event(&mut self, event: &Event, raw: &str) {
        self.stages.push(event.stage.as_slug().to_string());
        self.raw.push(raw.to_string());
    }
}

fn validate_project() -> Request {
    Request::ValidateProject(ProjectDirParams {
        project_dir: PathBuf::from("project-dir-the-fake-worker-never-reads"),
    })
}

fn probe_media() -> Request {
    Request::ProbeMedia(ProbeMediaParams {
        input: PathBuf::from("input-the-fake-worker-never-reads.mp4"),
    })
}

/// Spawn, run one task, and report the outcome, the sink, and the foreign count.
fn run_scenario(name: &str, request: Request) -> (SessionOutcome, Collected, usize) {
    let mut session = WorkerSession::spawn_with_env(&fake_worker(), fast_options(), &scenario(name))
        .expect("the handshake succeeds");
    let mut sink = Collected::default();
    let outcome = session.run(
        generate_task_id().expect("the generated id is valid"),
        request,
        &CancelToken::new(),
        &mut sink,
    );
    let foreign = session.foreign_event_count();
    (outcome, sink, foreign)
}

#[test]
fn a_completed_task_carries_the_workers_result() {
    let (outcome, sink, foreign) = run_scenario("ready-complete", validate_project());
    let SessionOutcome::Completed { result } = outcome else {
        panic!("expected completion, got {outcome:?}");
    };
    assert_eq!(result, Some(serde_json::json!({ "checked": true })));
    assert_eq!(sink.stages, vec!["preparing", "completed"]);
    assert_eq!(foreign, 0);
    assert!(
        sink.raw.iter().all(|line| line.contains("\"frame\":\"event\"")),
        "the sink receives the worker's own bytes: {:?}",
        sink.raw
    );
}

#[test]
fn a_failed_task_carries_the_workers_error() {
    let (outcome, sink, _) = run_scenario("fail", validate_project());
    let SessionOutcome::Failed(error) = outcome else {
        panic!("expected a task failure, got {outcome:?}");
    };
    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.summary, "输入文件无法解析，请确认它是完整的视频");
    assert_eq!(error.detail, "ffprobe exited with status 1");
    assert_eq!(sink.stages, vec!["failed"]);
}

#[test]
fn a_cancelled_stage_is_a_cancelled_outcome() {
    let (outcome, sink, _) = run_scenario("self-cancel", validate_project());
    assert!(
        matches!(outcome, SessionOutcome::Cancelled),
        "expected cancellation, got {outcome:?}"
    );
    assert_eq!(sink.stages, vec!["cancelled"]);
}

#[test]
fn an_unsupported_command_is_refused_before_it_is_sent() {
    let (outcome, sink, _) = run_scenario("only-validate", probe_media());
    let SessionOutcome::SessionError(ClientError::UnsupportedCommand {
        requested,
        supported,
    }) = outcome
    else {
        panic!("expected the capability gate to refuse, got {outcome:?}");
    };
    assert_eq!(requested, "probe_media");
    assert_eq!(supported, vec!["validate_project"]);
    assert!(sink.stages.is_empty(), "nothing was sent, so nothing ran");
}

#[test]
fn a_supported_command_passes_the_gate() {
    let (outcome, _, _) = run_scenario("only-probe", probe_media());
    assert!(
        matches!(outcome, SessionOutcome::Completed { .. }),
        "expected completion, got {outcome:?}"
    );
}

#[test]
fn an_event_for_another_task_is_ignored_and_counted() {
    let (outcome, sink, foreign) = run_scenario("foreign-event", validate_project());
    assert!(
        matches!(outcome, SessionOutcome::Completed { .. }),
        "expected completion, got {outcome:?}"
    );
    assert_eq!(sink.stages, vec!["completed"]);
    assert_eq!(foreign, 1);
}

#[test]
fn a_worker_that_exits_mid_task_is_reported_as_gone() {
    let (outcome, _, _) = run_scenario("die-after-ready", validate_project());
    assert!(
        matches!(
            outcome,
            SessionOutcome::SessionError(ClientError::WorkerGone { .. })
        ),
        "expected a lost worker, got {outcome:?}"
    );
}

#[test]
fn an_oversized_line_is_a_protocol_error() {
    let (outcome, _, _) = run_scenario("oversized-line", validate_project());
    assert!(
        matches!(
            outcome,
            SessionOutcome::SessionError(ClientError::Protocol(DomainError::FrameTooLong { .. }))
        ),
        "expected a frame bound violation, got {outcome:?}"
    );
}
```

Run: `cargo test -p feathertalk-client --test run`

Expected: FAIL to compile — `run`, `EventSink`, `SessionOutcome`, and `CancelToken` do not exist yet.

- [ ] **Step 3: Add the sink, the outcome, and the cancel token**

Extend the imports at the top of `rust/crates/feathertalk-client/src/session.rs`:

```rust
use std::sync::atomic::{AtomicUsize, Ordering};

use feathertalk_domain::{
    ClientFrame, DomainError, Event, FrameWriter, MAX_FRAME_BYTES, PROTOCOL_VERSION, ReadyFrame,
    RejectedFrame, Request, ServerFrame, ShutdownFrame, StartFrame, TaskError, TaskId, TaskKind,
    TaskStage, decode_line,
};
use serde_json::Value;
```

Then append to the same file:

```rust
/// How long one wait on the frame channel blocks. The loop has to wake up
/// regularly even when the worker is quiet, because that is when the cancel flag
/// is checked; 100 ms keeps Ctrl-C responsive without busy-waiting.
const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Where a session's events go. The CLI has two implementations, human and
/// JSON; the desktop shell will add a third that forwards to the UI.
///
/// `raw` is the worker's original line. Presenters that echo JSON must use it
/// rather than re-serialising `event`.
pub trait EventSink {
    fn on_event(&mut self, event: &Event, raw: &str);

    /// A mid-session rejection. Default: ignore it — the returned
    /// `ClientError::Rejected` already carries the reason.
    fn on_rejected(&mut self, rejected: &RejectedFrame, raw: &str) {
        let _ = (rejected, raw);
    }
}

/// How a session ended. The four variants are the four CLI exit codes: 0, 1, 2,
/// and 3, in this order.
#[derive(Debug)]
pub enum SessionOutcome {
    Completed { result: Option<Value> },
    Failed(TaskError),
    Cancelled,
    SessionError(ClientError),
}

/// A cancel request counter shared with a signal handler.
///
/// A counter rather than a flag: the first request asks politely, the second
/// kills. It is `Clone` so a handler can own one, and only ever counts up, so
/// there is no lost-wakeup case to reason about.
#[derive(Debug, Clone, Default)]
pub struct CancelToken(Arc<AtomicUsize>);

impl CancelToken {
    pub fn new() -> Self {
        Self(Arc::new(AtomicUsize::new(0)))
    }

    /// Signal-handler safe: one atomic add, no allocation, no locks.
    pub fn request(&self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }

    pub fn count(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }
}

/// Map a terminal stage onto an outcome. The three arms are exactly the stages
/// `TaskStage::is_terminal` reports; every other stage is progress.
fn terminal_outcome(event: Event) -> Option<SessionOutcome> {
    match event.stage {
        TaskStage::Completed => Some(SessionOutcome::Completed {
            result: event.result,
        }),
        TaskStage::Failed { .. } => Some(SessionOutcome::Failed(
            event
                .error
                .expect("Event::validate guarantees a failed stage carries its error"),
        )),
        TaskStage::Cancelled => Some(SessionOutcome::Cancelled),
        _ => None,
    }
}
```

- [ ] **Step 4: Add the run loop to `WorkerSession`**

Add to the `impl WorkerSession` block in `rust/crates/feathertalk-client/src/session.rs`:

```rust
    /// Run one task to completion. A session runs at most one.
    ///
    /// Session-level failures are returned as `SessionOutcome::SessionError`
    /// rather than `Err`, so the caller has a single total match over the four
    /// things that can happen.
    pub fn run(
        &mut self,
        task_id: TaskId,
        request: Request,
        cancel: &CancelToken,
        sink: &mut dyn EventSink,
    ) -> SessionOutcome {
        match self.run_inner(task_id, request, cancel, sink) {
            Ok(outcome) => outcome,
            Err(error) => SessionOutcome::SessionError(error),
        }
    }

    /// Refuse a command the worker did not advertise.
    ///
    /// The handshake is the only authority here. Sending a command the worker
    /// disclaimed would earn a rejection frame at best and confuse the user
    /// about whose fault it is at worst.
    fn ensure_supported(&self, request: &Request) -> Result<(), ClientError> {
        let requested = request.kind();
        if self.ready.supported_commands.contains(&requested) {
            return Ok(());
        }
        Err(ClientError::UnsupportedCommand {
            // `TaskKind::as_slug` takes `self` by value, so copy out of the slice.
            requested: requested.as_slug(),
            supported: self
                .ready
                .supported_commands
                .iter()
                .copied()
                .map(TaskKind::as_slug)
                .collect(),
        })
    }

    fn run_inner(
        &mut self,
        task_id: TaskId,
        request: Request,
        cancel: &CancelToken,
        sink: &mut dyn EventSink,
    ) -> Result<SessionOutcome, ClientError> {
        // Task 4 replaces this with the cancel state machine.
        let _ = cancel;
        self.ensure_supported(&request)?;
        self.transport.write_frame(&ClientFrame::Start(StartFrame {
            protocol_version: PROTOCOL_VERSION,
            task_id: task_id.clone(),
            request,
        }))?;
        loop {
            match self.transport.next_frame(POLL_INTERVAL)? {
                // Quiet worker: keep waiting. The task has no deadline of its
                // own — training legitimately runs for hours.
                FrameEvent::Timeout => continue,
                FrameEvent::Eof => return Err(self.transport.worker_gone()),
                FrameEvent::Frame(line) => {
                    // Validation happens here, not on the reader thread, so the
                    // error can be attributed to this phase of the protocol.
                    line.frame.validate().map_err(ClientError::Protocol)?;
                    match line.frame {
                        ServerFrame::Event(event) => {
                            if event.task_id.as_str() != task_id.as_str() {
                                // A chatty worker is allowed; count it so the
                                // ignore rule stays observable.
                                self.foreign_events += 1;
                                continue;
                            }
                            sink.on_event(&event, &line.raw);
                            if let Some(outcome) = terminal_outcome(event) {
                                return Ok(outcome);
                            }
                        }
                        ServerFrame::Rejected(rejected) => {
                            sink.on_rejected(&rejected, &line.raw);
                            return Err(ClientError::Rejected {
                                reason: rejected.reason,
                            });
                        }
                        ServerFrame::Ready(_) => {
                            return Err(ClientError::Protocol(DomainError::MalformedFrame {
                                reason: "the worker sent a second ready frame".to_string(),
                            }));
                        }
                    }
                }
            }
        }
    }

    /// How many events for other tasks were ignored.
    pub fn foreign_event_count(&self) -> usize {
        self.foreign_events
    }

    /// Ask the worker to exit and wait out the shutdown grace.
    ///
    /// A best-effort write: if the worker is already gone the frame cannot be
    /// delivered and there is nothing to report. `Drop` kills any survivor, so
    /// this never leaks a process even when it returns `None`.
    pub fn shutdown(mut self) -> Option<i32> {
        let grace = self.transport.options.shutdown_grace;
        let _ = self
            .transport
            .write_frame(&ClientFrame::Shutdown(ShutdownFrame {
                protocol_version: PROTOCOL_VERSION,
            }));
        self.transport.close_stdin();
        self.transport.wait_for_exit(grace)
    }
```

- [ ] **Step 5: Export the new types**

In `rust/crates/feathertalk-client/src/lib.rs`, replace the session re-export:

```rust
pub use session::{CancelToken, EventSink, FrameLine, SessionOutcome, WorkerSession};
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test -p feathertalk-client --all-targets`

Expected: PASS, 25 tests (17 from Tasks 1–2, 8 run tests).

`an_oversized_line_is_a_protocol_error` allocates just over a mebibyte in the child and streams it; if it is slow rather than failing, that is the copy, not a bug.

- [ ] **Step 7: Run the linters**

Run: `cargo clippy -p feathertalk-client --all-targets -- -D warnings`
Run: `cargo fmt --all -- --check`

Expected: both exit 0. `foreign_events` now has a reader, so the dead-code warning from Task 2 is gone.

- [ ] **Step 8: Commit**

```bash
git add rust/crates/feathertalk-client
git commit -m "feat(client): drive one task per session"
```

---

### Task 4: Cancellation with a bounded grace

Ctrl-C has to stop the task without corrupting anything and without ever hanging. The escalation is three steps, each bounded:

1. First request: send `cancel` and start the grace clock. The worker gets a chance to stop cleanly, flush a checkpoint, and report `cancelled`.
2. Grace expired: send `shutdown`, close stdin, wait the shutdown grace, then kill.
3. Second request: kill immediately. A user who presses Ctrl-C twice is telling us the polite path is not working.

The race rule falls out of the loop's ordering rather than being special-cased: a terminal event is handled the moment it arrives, so a `completed` frame already in flight when the cancel was sent still wins. Reporting "cancelled" for work that actually finished would be a lie the user could not detect.

**Files:**
- Modify: `rust/crates/feathertalk-client/src/session.rs`
- Modify: `rust/crates/feathertalk-client/tests/support/fake_worker.rs`
- Test: `rust/crates/feathertalk-client/tests/cancel.rs`

**Interfaces:**
- Consumes: `feathertalk_domain::CancelFrame`; `crate::CancelToken` from Task 3.
- Produces:
  - Private `enum CancelState { Idle, Requested { deadline: Instant } }`, `#[derive(Copy, Clone)]`.
  - Private `WorkerSession::service_cancel(&mut self, &CancelToken, &mut CancelState, &TaskId) -> Result<Option<SessionOutcome>, ClientError>`, called once per loop iteration before the bounded read.
  - New fake worker scenarios: `cancel-acks`, `cancel-completes`, `cancel-ignored`, `die-on-cancel`.
- No public API changes: `CancelToken` was already threaded through `run` in Task 3.

- [ ] **Step 1: Teach the fake worker to receive cancels**

In `rust/crates/feathertalk-client/tests/support/fake_worker.rs`, add a cancel reader next to `wait_for_start`:

```rust
/// Block until the client sends `cancel`. Returns `None` on shutdown or EOF.
fn wait_for_cancel(reader: &mut Reader) -> Option<TaskId> {
    loop {
        match reader.read_frame::<ClientFrame>()? {
            Ok(ClientFrame::Cancel(cancel)) => return Some(cancel.task_id),
            Ok(ClientFrame::Start(_)) => continue,
            Ok(ClientFrame::Shutdown(_)) => return None,
            Err(error) => {
                eprintln!("fake worker could not decode a client frame: {error}");
                return None;
            }
        }
    }
}
```

And add these arms before the `other =>` arm:

```rust
        // The cooperative path: acknowledges the cancel with a terminal event.
        "cancel-acks" => {
            write_frame(&ready(default_commands()));
            if let Some(task_id) = wait_for_start(&mut reader) {
                if wait_for_cancel(&mut reader).is_some() {
                    write_frame(&ServerFrame::Event(stage_event(
                        &task_id,
                        TaskStage::Cancelled,
                    )));
                }
            }
        }
        // Finishes anyway: the completion must win over the pending cancel.
        "cancel-completes" => {
            write_frame(&ready(default_commands()));
            if let Some(task_id) = wait_for_start(&mut reader) {
                if wait_for_cancel(&mut reader).is_some() {
                    write_frame(&ServerFrame::Event(completed(&task_id)));
                }
            }
        }
        // Reads nothing and answers nothing, so only the grace deadline ends it.
        "cancel-ignored" => {
            write_frame(&ready(default_commands()));
            if wait_for_start(&mut reader).is_some() {
                park();
            }
        }
        // Dies without acknowledging: EOF after a cancel is still a cancellation.
        "die-on-cancel" => {
            write_frame(&ready(default_commands()));
            if wait_for_start(&mut reader).is_some() && wait_for_cancel(&mut reader).is_some() {
                std::process::exit(0);
            }
        }
```

- [ ] **Step 2: Write the failing cancellation tests**

Create `rust/crates/feathertalk-client/tests/cancel.rs`:

```rust
#[path = "support/harness.rs"]
mod harness;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use feathertalk_client::{
    CancelToken, EventSink, SessionOutcome, WorkerSession, generate_task_id,
};
use feathertalk_domain::{Event, ProjectDirParams, Request};

use harness::{fake_worker, fast_options, scenario};

/// The cancel tests only care about outcomes, not about event text.
struct Ignore;

impl EventSink for Ignore {
    fn on_event(&mut self, event: &Event, raw: &str) {
        let _ = (event, raw);
    }
}

fn validate_project() -> Request {
    Request::ValidateProject(ProjectDirParams {
        project_dir: PathBuf::from("project-dir-the-fake-worker-never-reads"),
    })
}

/// Run one task with `requests` cancel requests already registered.
///
/// Requesting before `run` rather than from a thread keeps these tests
/// deterministic: the token is checked at the top of every loop iteration, so a
/// pre-registered request is seen on the first one.
fn run_cancelled(name: &str, requests: usize) -> (SessionOutcome, Duration) {
    let mut session = WorkerSession::spawn_with_env(&fake_worker(), fast_options(), &scenario(name))
        .expect("the handshake succeeds");
    let cancel = CancelToken::new();
    for _ in 0..requests {
        cancel.request();
    }
    let started = Instant::now();
    let outcome = session.run(
        generate_task_id().expect("the generated id is valid"),
        validate_project(),
        &cancel,
        &mut Ignore,
    );
    (outcome, started.elapsed())
}

#[test]
fn a_worker_that_acknowledges_is_reported_cancelled() {
    let (outcome, elapsed) = run_cancelled("cancel-acks", 1);
    assert!(
        matches!(outcome, SessionOutcome::Cancelled),
        "expected cancellation, got {outcome:?}"
    );
    assert!(
        elapsed < Duration::from_secs(2),
        "a cooperative worker should not need the grace period: {elapsed:?}"
    );
}

#[test]
fn a_completion_in_flight_beats_a_cancel() {
    let (outcome, _) = run_cancelled("cancel-completes", 1);
    assert!(
        matches!(outcome, SessionOutcome::Completed { .. }),
        "work that finished must not be reported as cancelled, got {outcome:?}"
    );
}

#[test]
fn an_unresponsive_worker_is_stopped_when_the_grace_expires() {
    let (outcome, elapsed) = run_cancelled("cancel-ignored", 1);
    assert!(
        matches!(outcome, SessionOutcome::Cancelled),
        "expected cancellation, got {outcome:?}"
    );
    assert!(
        elapsed >= Duration::from_millis(200),
        "the worker must be given the full grace period first: {elapsed:?}"
    );
    assert!(
        elapsed < Duration::from_secs(5),
        "the escalation must be bounded: {elapsed:?}"
    );
}

#[test]
fn a_worker_that_exits_after_a_cancel_is_reported_cancelled() {
    let (outcome, _) = run_cancelled("die-on-cancel", 1);
    assert!(
        matches!(outcome, SessionOutcome::Cancelled),
        "EOF after a cancel is a cancellation, not a lost worker: got {outcome:?}"
    );
}

#[test]
fn a_second_request_kills_without_waiting() {
    let (outcome, elapsed) = run_cancelled("cancel-ignored", 2);
    assert!(
        matches!(outcome, SessionOutcome::Cancelled),
        "expected cancellation, got {outcome:?}"
    );
    assert!(
        elapsed < Duration::from_millis(200),
        "a second request must skip the grace period entirely: {elapsed:?}"
    );
}
```

Run: `cargo test -p feathertalk-client --test cancel`

Expected: FAIL — the token is still ignored, so `cancel-acks`, `cancel-ignored`, and the double request hang until the test harness gives up rather than returning `Cancelled`.

- [ ] **Step 3: Add the cancel state machine**

Add `CancelFrame` to the `feathertalk_domain` import list in `rust/crates/feathertalk-client/src/session.rs`, and add the state type next to `terminal_outcome`:

```rust
/// Where the cancel escalation currently is.
///
/// Two states are enough: every kill path returns an outcome immediately, so
/// there is no "killed" state anyone could observe.
#[derive(Debug, Copy, Clone)]
enum CancelState {
    Idle,
    Requested { deadline: Instant },
}
```

Then add the escalation to the `impl WorkerSession` block:

```rust
    /// Act on the cancel token, and on the grace deadline if one is running.
    ///
    /// `Some(Cancelled)` means the session is over. Every branch here is
    /// bounded: the polite path has a deadline, and the deadline has a kill.
    fn service_cancel(
        &mut self,
        cancel: &CancelToken,
        state: &mut CancelState,
        task_id: &TaskId,
    ) -> Result<Option<SessionOutcome>, ClientError> {
        match (cancel.count(), *state) {
            // Nothing asked for, or already asked and still within the grace.
            (0, _) | (1, CancelState::Requested { .. }) => {}
            (1, CancelState::Idle) => {
                let grace = self.transport.options.cancel_grace;
                let frame = ClientFrame::Cancel(CancelFrame {
                    protocol_version: PROTOCOL_VERSION,
                    task_id: task_id.clone(),
                });
                if self.transport.write_frame(&frame).is_err() {
                    // The worker is already unreachable. The user asked for the
                    // task to stop, and it has: report that, not a transport
                    // failure they can do nothing about.
                    self.transport.kill_and_reap();
                    return Ok(Some(SessionOutcome::Cancelled));
                }
                *state = CancelState::Requested {
                    deadline: Instant::now() + grace,
                };
            }
            // Two or more requests: the user has asked twice, stop now.
            _ => {
                self.transport.kill_and_reap();
                return Ok(Some(SessionOutcome::Cancelled));
            }
        }
        if let CancelState::Requested { deadline } = *state
            && Instant::now() >= deadline
        {
            // The worker had its chance. Escalate: shutdown, then EOF, then kill.
            let grace = self.transport.options.shutdown_grace;
            let _ = self
                .transport
                .write_frame(&ClientFrame::Shutdown(ShutdownFrame {
                    protocol_version: PROTOCOL_VERSION,
                }));
            self.transport.close_stdin();
            if self.transport.wait_for_exit(grace).is_none() {
                self.transport.kill_and_reap();
            }
            return Ok(Some(SessionOutcome::Cancelled));
        }
        Ok(None)
    }
```

- [ ] **Step 4: Wire the escalation into the run loop**

In `run_inner`, delete the `let _ = cancel;` placeholder, declare the state before the loop, service it at the top of every iteration, and reinterpret EOF:

```rust
        let mut cancel_state = CancelState::Idle;
        loop {
            // Checked before the bounded read, so a request registered while the
            // worker was quiet is acted on within one POLL_INTERVAL.
            if let Some(outcome) = self.service_cancel(cancel, &mut cancel_state, &task_id)? {
                return Ok(outcome);
            }
            match self.transport.next_frame(POLL_INTERVAL)? {
                FrameEvent::Timeout => continue,
                FrameEvent::Eof => {
                    // A worker that exits after being asked to stop has stopped.
                    if !matches!(cancel_state, CancelState::Idle) {
                        return Ok(SessionOutcome::Cancelled);
                    }
                    return Err(self.transport.worker_gone());
                }
                FrameEvent::Frame(line) => {
                    // Keep Task 3's body for this arm exactly as it is.
                }
            }
        }
```

The frame arm is untouched. Terminal events are still handled the instant they arrive, which is exactly what makes `a_completion_in_flight_beats_a_cancel` pass: the `completed` frame is processed before the next `service_cancel` call ever runs.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p feathertalk-client --all-targets`

Expected: PASS, 30 tests (25 from Tasks 1–3, 5 cancel tests).

If `a_second_request_kills_without_waiting` is flaky above 200 ms, the cause is a `write_frame` call before the count check; the count must be read first.

- [ ] **Step 6: Run the linters**

Run: `cargo clippy -p feathertalk-client --all-targets -- -D warnings`
Run: `cargo fmt --all -- --check`

Expected: both exit 0.

- [ ] **Step 7: Commit**

```bash
git add rust/crates/feathertalk-client
git commit -m "feat(client): honour cancellation with a bounded grace"
```

---

### Task 5: The `feathertalk` command line binary

Everything the user sees lands here, and the whole crate exists in the shape lib-plus-bin so that it can be asserted. `main.rs` parses arguments and exits; every string is produced by a function in `render.rs` that a test can call directly.

The output contract is the part to get right: **stdout is the result channel, stderr is the narration channel.** Human progress goes to stderr so `feathertalk probe-media a.mp4 > info.json` produces a usable file. Under `--json`, stdout carries the worker's own frame lines verbatim — one object per line, the handshake first — and the pretty result summary is suppressed, because a machine consumer already has the completed event.

**Files:**
- Modify: `rust/Cargo.toml`
- Create: `rust/crates/feathertalk-cli/Cargo.toml`
- Create: `rust/crates/feathertalk-cli/src/lib.rs`
- Create: `rust/crates/feathertalk-cli/src/main.rs`
- Create: `rust/crates/feathertalk-cli/src/cli.rs`
- Create: `rust/crates/feathertalk-cli/src/render.rs`
- Create: `rust/crates/feathertalk-cli/src/run.rs`
- Test support: `rust/crates/feathertalk-cli/tests/support/fake_worker_bin.rs`
- Test: `rust/crates/feathertalk-cli/tests/cli.rs`

**Interfaces:**
- Consumes: `feathertalk_client::{CancelToken, ClientError, EventSink, ENV_WORKER_BIN, ProbedPath, SessionOptions, SessionOutcome, WorkerLocator, WorkerPathSource, WorkerSession, generate_task_id}`; `feathertalk_domain::{Event, ProbeMediaParams, ProjectDirParams, ReadyFrame, Recovery, RejectedFrame, Request, TaskError, TaskId, TaskKind, TaskStage}`; `clap::{Parser, Subcommand}`; `ctrlc::set_handler`.
- Produces:
  - `Cli { worker, json, quiet, task_id, command }` and `Command { ValidateProject { project_dir }, ProbeMedia { input }, Capabilities }`.
  - `stage_label(&TaskStage) -> String`, `event_line(&Event) -> String`, `recovery_label(&Recovery) -> &'static str`, `failure_block(&TaskError) -> String`, `capabilities_report(&ReadyFrame) -> String`, `render_client_error(&ClientError) -> String`, `slug<T: serde::Serialize>(&T) -> String`.
  - `HumanSink::new(quiet: bool)`, `JsonSink`.
  - `run(cli: Cli) -> i32` plus the four `EXIT_*` constants.
  - The `feathertalk` binary and, for tests, `feathertalk-cli-fake-worker`.

- [ ] **Step 1: Add the dependency and the member**

In `rust/Cargo.toml`, add the member after `crates/feathertalk-client`:

```toml
  "crates/feathertalk-client",
  "crates/feathertalk-cli",
```

And add the one new third-party dependency to `[workspace.dependencies]`, keeping the table's alphabetical order:

```toml
ctrlc = "=3.4.5"
```

Pinned with `=` because it installs a signal handler: a patch release that changes handler semantics should be an explicit decision, not something a `cargo update` does silently. If that exact version does not resolve, pin the newest 3.4.x and note the change in the commit body.

- [ ] **Step 2: Create the crate manifest**

Create `rust/crates/feathertalk-cli/Cargo.toml`:

```toml
[package]
name = "feathertalk-cli"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[lib]
name = "feathertalk_cli"
path = "src/lib.rs"

[[bin]]
name = "feathertalk"
path = "src/main.rs"

# The client's fake worker, rebuilt under this crate so the CLI tests have a
# worker to talk to without depending on a sibling crate's test targets.
[[bin]]
name = "feathertalk-cli-fake-worker"
path = "tests/support/fake_worker_bin.rs"

[dependencies]
clap = { workspace = true }
ctrlc = { workspace = true }
feathertalk-client = { path = "../feathertalk-client" }
feathertalk-domain = { path = "../feathertalk-domain" }
serde = { workspace = true }
serde_json = { workspace = true }

[dev-dependencies]
tempfile = { workspace = true }
```

The binary is `feathertalk`, not `feathertalk-cli`: that is the command users type.

- [ ] **Step 3: Re-use the fake worker**

Create `rust/crates/feathertalk-cli/tests/support/fake_worker_bin.rs`:

```rust
//! The client's fake worker, compiled again as a binary of this crate.
//!
//! `include!` rather than a copy: one source of truth for the scenarios, and a
//! distinct binary name so the two crates cannot collide in the target
//! directory. A `[[bin]]` cannot use dev-dependencies, which is why the included
//! file only needs `feathertalk-domain` and `serde_json` — both are ordinary
//! dependencies of this crate.

include!("../../../feathertalk-client/tests/support/fake_worker.rs");
```

- [ ] **Step 4: Write the argument surface**

Create `rust/crates/feathertalk-cli/src/cli.rs`:

```rust
use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// The command line. Help text is Chinese, because the user is.
#[derive(Debug, Parser)]
#[command(
    name = "feathertalk",
    version,
    about = "FeatherTalk 命令行客户端",
    long_about = "通过标准输入输出驱动 feathertalk-worker 执行单个任务。\n\n\
                  标准输出只有结果，进度输出在标准错误，因此可以安全重定向。\n\
                  退出码：0 完成，1 任务失败，2 已取消，3 会话错误。"
)]
pub struct Cli {
    /// 工作进程可执行文件路径，默认依次查找环境变量与本程序同目录
    #[arg(long, global = true, value_name = "PATH")]
    pub worker: Option<PathBuf>,

    /// 按行输出原始协议帧，供程序解析
    #[arg(long, global = true)]
    pub json: bool,

    /// 不输出进度，只保留结果与错误
    #[arg(long, global = true, conflicts_with = "json")]
    pub quiet: bool,

    /// 指定任务 ID：13 位毫秒时间戳、连字符、8 位小写十六进制
    #[arg(long, global = true, value_name = "ID")]
    pub task_id: Option<String>,

    #[command(subcommand)]
    pub command: Command,
}

/// The task commands, kebab-cased by clap: `validate-project`, `probe-media`,
/// `capabilities`.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// 校验工程目录
    ValidateProject {
        /// 工程目录
        project_dir: PathBuf,
    },
    /// 探测媒体文件信息
    ProbeMedia {
        /// 输入的音视频文件
        input: PathBuf,
    },
    /// 打印工作进程的握手信息：后端、设备、支持的命令
    Capabilities,
}
```

`--quiet` conflicts with `--json` rather than silently losing to it: the two ask for opposite things, and clap can say so better than a precedence rule nobody reads.

- [ ] **Step 5: Write the presentation layer**

Create `rust/crates/feathertalk-cli/src/render.rs`:

```rust
//! Every string the user sees.
//!
//! Separated from `run.rs` so the whole output surface can be asserted without
//! spawning a process.

use feathertalk_client::{ClientError, ENV_WORKER_BIN, EventSink, WorkerPathSource};
use feathertalk_domain::{Event, ReadyFrame, Recovery, RejectedFrame, TaskError, TaskKind, TaskStage};

/// The worker's own variable for locating `ffprobe`. Written as a literal
/// because the CLI must not link the worker crate; `feathertalk-worker`'s
/// `ENV_FFPROBE` is the source of truth for the name.
const ENV_WORKER_FFPROBE: &str = "FEATHERTALK_WORKER_FFPROBE";

/// The Chinese name of every stage.
///
/// No `_` arm on purpose: adding a stage to the protocol must break this match.
/// A stage the CLI cannot name is a stage the user cannot understand.
pub fn stage_label(stage: &TaskStage) -> String {
    match stage {
        TaskStage::Queued => "排队中".to_string(),
        TaskStage::Preparing => "准备中".to_string(),
        TaskStage::ExtractingAudio => "正在提取音频".to_string(),
        TaskStage::ExtractingFrames => "正在提取视频帧".to_string(),
        TaskStage::DetectingFaces => "正在检测人脸".to_string(),
        TaskStage::ExtractingFeatures => "正在提取特征".to_string(),
        TaskStage::Training { epoch, step, loss } => {
            format!("正在训练 轮次 {epoch} 步 {step} 损失 {loss:.4}")
        }
        TaskStage::Importing => "正在导入".to_string(),
        TaskStage::Exporting => "正在导出".to_string(),
        TaskStage::Rendering { frame, total } => format!("正在渲染 第 {frame}/{total} 帧"),
        TaskStage::Completed => "已完成".to_string(),
        TaskStage::Failed { code, message } => format!("已失败 {} {message}", code.as_wire()),
        TaskStage::Cancelled => "已取消".to_string(),
    }
}

/// One line per event. The slug is kept alongside the Chinese label so a user
/// can search the logs or a spec for the same token the protocol uses.
pub fn event_line(event: &Event) -> String {
    let mut line = format!("[{}] {}", event.stage.as_slug(), stage_label(&event.stage));
    if let Some(text) = progress_text(event) {
        line.push(' ');
        line.push_str(&text);
    }
    if let Some(text) = metrics_text(event) {
        line.push(' ');
        line.push_str(&text);
    }
    line
}

fn progress_text(event: &Event) -> Option<String> {
    let progress = event.progress.as_ref()?;
    match progress.total {
        // A percentage needs a denominator, and zero is not one.
        Some(total) if total > 0 => Some(format!(
            "进度 {}/{} ({:.1}%)",
            progress.completed,
            total,
            progress.completed as f64 * 100.0 / total as f64
        )),
        Some(total) => Some(format!("进度 {}/{}", progress.completed, total)),
        None => Some(format!("已处理 {}", progress.completed)),
    }
}

fn metrics_text(event: &Event) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(rate) = event.metrics.samples_per_second {
        parts.push(format!("速率 {rate:.2}/秒"));
    }
    if let Some(eta) = event.metrics.eta_seconds {
        parts.push(format!("预计剩余 {eta} 秒"));
    }
    if let Some(vram) = event.metrics.vram_bytes {
        parts.push(format!("显存 {}", mebibytes(vram)));
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}

fn mebibytes(bytes: u64) -> String {
    format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
}

/// What the user can do about a failure. Also exhaustive by design.
pub fn recovery_label(recovery: &Recovery) -> &'static str {
    match recovery {
        Recovery::Retry => "可以直接重试该任务",
        Recovery::ResumeFromCheckpoint => "可以从最近的检查点继续",
        Recovery::FreeDiskSpace => "请清理磁盘空间后重试",
        Recovery::SelectDifferentAdapter => "请改用其他计算设备后重试",
        Recovery::ExcludeBadFrames => "请排除有问题的视频帧后重试",
        Recovery::ReimportModel => "请重新导入模型文件",
        Recovery::NotRecoverable => "该错误无法自动恢复，请检查输入与环境",
    }
}

/// The failure report.
///
/// `summary` is the worker's Chinese sentence and `detail` is its English
/// diagnostic. Both are printed verbatim: translating either would put a second
/// author between the operator and what actually happened.
pub fn failure_block(error: &TaskError) -> String {
    [
        format!("任务失败：{}", error.summary),
        format!("错误码: {}", error.code.as_wire()),
        format!("阶段: {}", error.stage.as_slug()),
        format!("建议: {}", recovery_label(&error.recovery)),
        format!("详情: {}", error.detail),
    ]
    .join("\n")
}

/// The handshake, in Chinese. Built from `ready` alone — the CLI probes nothing
/// itself, so what it prints is exactly what the worker claims.
pub fn capabilities_report(ready: &ReadyFrame) -> String {
    let mut lines = vec![
        format!("工作进程版本: {}", ready.worker_version),
        format!("协议版本: {}", ready.protocol_version),
        format!(
            "后端: {}",
            ready.backends.iter().map(slug).collect::<Vec<_>>().join(", ")
        ),
        "计算设备:".to_string(),
    ];
    for adapter in &ready.adapters {
        let vram = match adapter.vram_bytes {
            Some(bytes) => format!(" 显存 {}", mebibytes(bytes)),
            None => String::new(),
        };
        lines.push(format!(
            "  {} {} 类型 {} 后端 {} 认证 {}{vram}",
            adapter.id,
            adapter.name,
            slug(&adapter.kind),
            slug(&adapter.backend),
            yes_no(adapter.certified)
        ));
    }
    lines.push(format!(
        "支持的命令: {}",
        ready
            .supported_commands
            .iter()
            .copied()
            .map(TaskKind::as_slug)
            .collect::<Vec<_>>()
            .join(", ")
    ));
    lines.push(format!(
        "能力: 训练 {} wgpu 训练 {} onnx 校验 {} ffmpeg {}",
        yes_no(ready.capabilities.training),
        yes_no(ready.capabilities.wgpu_training),
        yes_no(ready.capabilities.onnx_validation),
        yes_no(ready.capabilities.ffmpeg)
    ));
    lines.join("\n")
}

fn yes_no(value: bool) -> &'static str {
    if value { "是" } else { "否" }
}

/// The wire spelling of a type that has no `as_slug` — `Backend`, `AdapterKind`,
/// `ErrorCode`. Taken from serde so the CLI can never drift from the protocol.
pub fn slug<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string(value)
        .unwrap_or_default()
        .trim_matches('"')
        .to_string()
}

/// Human output. Events go to stderr so stdout stays a clean result channel and
/// `feathertalk probe-media a.mp4 > info.json` produces a usable file.
pub struct HumanSink {
    quiet: bool,
}

impl HumanSink {
    pub fn new(quiet: bool) -> Self {
        Self { quiet }
    }
}

impl EventSink for HumanSink {
    fn on_event(&mut self, event: &Event, raw: &str) {
        let _ = raw;
        // `--quiet` drops progress, never the result or the failure report.
        if self.quiet {
            return;
        }
        eprintln!("{}", event_line(event));
    }

    fn on_rejected(&mut self, rejected: &RejectedFrame, raw: &str) {
        let _ = raw;
        eprintln!("工作进程拒绝了请求：{}", rejected.reason);
    }
}

/// Machine output: every frame exactly as the worker wrote it, one per line, on
/// stdout. Never a re-serialisation — this workspace builds `serde_json` without
/// `preserve_order`, so a round trip would silently reorder object keys.
pub struct JsonSink;

impl EventSink for JsonSink {
    fn on_event(&mut self, event: &Event, raw: &str) {
        let _ = event;
        println!("{raw}");
    }

    fn on_rejected(&mut self, rejected: &RejectedFrame, raw: &str) {
        let _ = rejected;
        println!("{raw}");
    }
}

/// Session-level errors, in Chinese, each one ending in something to try.
///
/// `ClientError`'s own `Display` is English and stays that way: it is a
/// developer-facing diagnostic, and this is the user-facing translation of it.
pub fn render_client_error(error: &ClientError) -> String {
    let mut text = match error {
        ClientError::WorkerNotFound { probed } => {
            let mut lines = vec!["找不到工作进程可执行文件。已按以下顺序查找：".to_string()];
            for candidate in probed {
                let shown = match candidate.path.as_ref() {
                    Some(path) => path.display().to_string(),
                    None => "未设置".to_string(),
                };
                lines.push(format!("  {}: {shown}", source_label(candidate.source)));
            }
            lines.push(format!(
                "请用 --worker 指定路径，或设置环境变量 {ENV_WORKER_BIN}。"
            ));
            lines.join("\n")
        }
        ClientError::Spawn { path, source } => format!(
            "无法启动工作进程 {}：{source}\n请确认该文件存在并且可以执行。",
            path.display()
        ),
        ClientError::Handshake { reason, .. } => {
            format!("工作进程握手失败：{reason}\n请确认 --worker 指向的是 feathertalk-worker。")
        }
        ClientError::ProtocolVersion { expected, actual } => format!(
            "协议版本不匹配：本客户端支持 {expected}，工作进程使用 {actual}。\n请让两者来自同一次构建。"
        ),
        ClientError::Rejected { reason } => format!("工作进程拒绝了本次请求：{reason}"),
        ClientError::UnsupportedCommand {
            requested,
            supported,
        } => {
            let mut text = format!(
                "工作进程不支持命令 {requested}。它声明支持：{}。",
                supported.join(", ")
            );
            if *requested == "probe_media" {
                text.push_str(&format!(
                    "\nprobe_media 需要可用的 ffprobe。请安装 ffmpeg，或用环境变量 \
                     {ENV_WORKER_FFPROBE} 指定 ffprobe 的完整路径。"
                ));
            }
            text
        }
        ClientError::Protocol(source) => {
            format!("协议错误：{source}\n工作进程与客户端的版本可能不一致。")
        }
        ClientError::Io(source) => format!("读写工作进程时出错：{source}"),
        ClientError::WorkerGone { status, .. } => match status {
            Some(code) => format!("工作进程已退出（退出码 {code}），任务没有完成。"),
            None => "工作进程已不可用，任务没有完成。".to_string(),
        },
    };
    let tail = error.stderr_tail();
    if !tail.is_empty() {
        text.push_str("\n工作进程最后的输出：");
        for line in tail {
            text.push_str(&format!("\n  {line}"));
        }
    }
    text
}

fn source_label(source: WorkerPathSource) -> &'static str {
    match source {
        WorkerPathSource::CliOption => "--worker 选项",
        WorkerPathSource::EnvVar => "环境变量 FEATHERTALK_WORKER_BIN",
        WorkerPathSource::SiblingOfCurrentExe => "与本程序同目录",
    }
}

#[cfg(test)]
mod tests {
    use feathertalk_domain::ErrorCode;

    use super::*;

    #[test]
    fn every_stage_has_a_chinese_label() {
        for stage in TaskStage::ALL_UNIT_SAMPLES {
            let label = stage_label(&stage);
            assert!(
                !label.is_ascii(),
                "{stage:?} must have a Chinese label, got {label:?}"
            );
        }
    }

    #[test]
    fn every_recovery_has_advice() {
        for recovery in [
            Recovery::Retry,
            Recovery::ResumeFromCheckpoint,
            Recovery::FreeDiskSpace,
            Recovery::SelectDifferentAdapter,
            Recovery::ExcludeBadFrames,
            Recovery::ReimportModel,
            Recovery::NotRecoverable,
        ] {
            assert!(!recovery_label(&recovery).is_empty(), "{recovery:?}");
        }
    }

    #[test]
    fn error_codes_are_shown_in_their_wire_form() {
        // `as_wire` and serde must agree; this is the guard against drift.
        for code in ErrorCode::ALL {
            assert_eq!(slug(&code), code.as_wire(), "{code:?}");
        }
    }
}
```

- [ ] **Step 6: Write the driver**

Create `rust/crates/feathertalk-cli/src/run.rs`:

```rust
//! Locate the worker, run one task, choose the exit code.

use std::path::Path;

use feathertalk_client::{
    CancelToken, EventSink, SessionOptions, SessionOutcome, WorkerLocator, WorkerSession,
    generate_task_id,
};
use feathertalk_domain::{ProbeMediaParams, ProjectDirParams, Request, TaskId};

use crate::cli::{Cli, Command};
use crate::render::{
    HumanSink, JsonSink, capabilities_report, failure_block, render_client_error,
};

/// The four exit codes, fixed by the spec. Nothing else is ever returned.
pub const EXIT_COMPLETED: i32 = 0;
pub const EXIT_TASK_FAILED: i32 = 1;
pub const EXIT_CANCELLED: i32 = 2;
pub const EXIT_SESSION_ERROR: i32 = 3;

pub fn run(cli: Cli) -> i32 {
    let request = match build_request(&cli.command) {
        Ok(request) => request,
        Err(message) => {
            eprintln!("{message}");
            return EXIT_SESSION_ERROR;
        }
    };
    let path = match WorkerLocator::from_env(cli.worker.clone()).resolve() {
        Ok(path) => path,
        Err(error) => {
            eprintln!("{}", render_client_error(&error));
            return EXIT_SESSION_ERROR;
        }
    };
    // The worker inherits this process's environment and reads its own
    // configuration; the CLI injects nothing.
    let mut session = match WorkerSession::spawn(&path, SessionOptions::default()) {
        Ok(session) => session,
        Err(error) => {
            eprintln!("{}", render_client_error(&error));
            return EXIT_SESSION_ERROR;
        }
    };
    if cli.json {
        // The handshake is a protocol frame too, so machine consumers get it.
        println!("{}", session.ready_raw());
    }
    let code = match request {
        // `capabilities` needs no task: the handshake already answered it.
        None => {
            if !cli.json {
                println!("{}", capabilities_report(session.ready()));
            }
            EXIT_COMPLETED
        }
        Some(request) => run_task(&mut session, &cli, request),
    };
    let _ = session.shutdown();
    code
}

/// Build the request, or `None` for `capabilities`.
///
/// Only empty arguments are rejected here. Whether a path exists, is a project,
/// or is decodable media is the worker's judgement, and duplicating it in the
/// CLI would produce two answers that can disagree.
fn build_request(command: &Command) -> Result<Option<Request>, String> {
    match command {
        Command::Capabilities => Ok(None),
        Command::ValidateProject { project_dir } => {
            reject_empty(project_dir, "工程目录")?;
            Ok(Some(Request::ValidateProject(ProjectDirParams {
                project_dir: project_dir.clone(),
            })))
        }
        Command::ProbeMedia { input } => {
            reject_empty(input, "输入文件")?;
            Ok(Some(Request::ProbeMedia(ProbeMediaParams {
                input: input.clone(),
            })))
        }
    }
}

fn reject_empty(path: &Path, label: &str) -> Result<(), String> {
    if path.as_os_str().is_empty() {
        return Err(format!("{label}不能为空。"));
    }
    Ok(())
}

fn run_task(session: &mut WorkerSession, cli: &Cli, request: Request) -> i32 {
    let task_id = match resolve_task_id(cli.task_id.as_deref()) {
        Ok(task_id) => task_id,
        Err(message) => {
            eprintln!("{message}");
            return EXIT_SESSION_ERROR;
        }
    };
    let cancel = CancelToken::new();
    install_cancel_handler(&cancel);
    let mut human = HumanSink::new(cli.quiet);
    let mut json = JsonSink;
    let sink: &mut dyn EventSink = if cli.json { &mut json } else { &mut human };
    match session.run(task_id, request, &cancel, sink) {
        SessionOutcome::Completed { result } => {
            // Under --json the completed frame already carried the result.
            if !cli.json {
                let value = result.unwrap_or_else(|| serde_json::json!({}));
                let text = serde_json::to_string_pretty(&value)
                    .unwrap_or_else(|_| value.to_string());
                println!("{text}");
            }
            EXIT_COMPLETED
        }
        SessionOutcome::Failed(error) => {
            eprintln!("{}", failure_block(&error));
            EXIT_TASK_FAILED
        }
        SessionOutcome::Cancelled => {
            eprintln!("任务已取消。");
            EXIT_CANCELLED
        }
        SessionOutcome::SessionError(error) => {
            eprintln!("{}", render_client_error(&error));
            EXIT_SESSION_ERROR
        }
    }
}

fn resolve_task_id(requested: Option<&str>) -> Result<TaskId, String> {
    match requested {
        Some(text) => TaskId::parse(text).map_err(|error| {
            format!("任务 ID 无效：{error}\n格式为 13 位毫秒时间戳、连字符、8 位小写十六进制。")
        }),
        None => generate_task_id().map_err(|error| format!("无法生成任务 ID：{error}")),
    }
}

/// Ctrl-C bumps the token and does nothing else: one atomic add, no allocation,
/// no locks. All the escalation lives in the client's run loop.
///
/// Failing to install is reported and ignored — the task can still run, it just
/// cannot be interrupted politely, and refusing to work would be worse.
fn install_cancel_handler(cancel: &CancelToken) {
    let token = cancel.clone();
    if let Err(error) = ctrlc::set_handler(move || token.request()) {
        eprintln!("无法注册 Ctrl-C 处理器：{error}");
    }
}
```

- [ ] **Step 7: Write the crate root and the binary**

Create `rust/crates/feathertalk-cli/src/lib.rs`:

```rust
//! The FeatherTalk command line client.
//!
//! A library plus a thin binary, following `tools/onnx-validate`: `main.rs` only
//! parses arguments and exits, so every line of user-facing text is reachable
//! from a test without spawning a process.

mod cli;
mod render;
mod run;

pub use cli::{Cli, Command};
pub use render::{
    HumanSink, JsonSink, capabilities_report, event_line, failure_block, recovery_label,
    render_client_error, slug, stage_label,
};
pub use run::{EXIT_CANCELLED, EXIT_COMPLETED, EXIT_SESSION_ERROR, EXIT_TASK_FAILED, run};
```

Create `rust/crates/feathertalk-cli/src/main.rs`:

```rust
use clap::Parser;
use clap::error::ErrorKind;

use feathertalk_cli::{Cli, EXIT_SESSION_ERROR};

fn main() {
    match Cli::try_parse() {
        Ok(cli) => std::process::exit(feathertalk_cli::run(cli)),
        Err(error) => {
            // `--help` and `--version` are requests that succeeded.
            if matches!(
                error.kind(),
                ErrorKind::DisplayHelp | ErrorKind::DisplayVersion
            ) {
                print!("{error}");
                std::process::exit(EXIT_COMPLETED_ON_HELP);
            }
            // Clap's own default for a usage error is 2, which this CLI has
            // already spent on "cancelled". A misused command line is a session
            // error, so it exits 3.
            eprint!("{error}");
            std::process::exit(EXIT_SESSION_ERROR);
        }
    }
}

/// Help and version are output, not failure.
const EXIT_COMPLETED_ON_HELP: i32 = 0;
```

- [ ] **Step 8: Write the end-to-end CLI tests**

Create `rust/crates/feathertalk-cli/tests/cli.rs`:

```rust
use std::process::{Command, Output};

const CLI: &str = env!("CARGO_BIN_EXE_feathertalk");
const FAKE_WORKER: &str = env!("CARGO_BIN_EXE_feathertalk-cli-fake-worker");

/// Run the CLI against the fake worker.
///
/// The worker path and the scenario are set on the child's environment rather
/// than this process's: `std::env::set_var` is `unsafe` in edition 2024 and
/// would race between tests that run in parallel.
fn run(scenario: &str, args: &[&str]) -> Output {
    Command::new(CLI)
        .args(args)
        .env("FEATHERTALK_WORKER_BIN", FAKE_WORKER)
        .env("FT_FAKE_WORKER_SCENARIO", scenario)
        .output()
        .expect("the CLI binary runs")
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
fn stdout_carries_only_the_result() {
    let output = run("ready-complete", &["validate-project", "some-project"]);
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));
    let value: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("stdout is exactly one JSON document");
    assert_eq!(value, serde_json::json!({ "checked": true }));
    // Progress narration belongs on stderr so stdout stays redirectable.
    assert!(stderr(&output).contains("[preparing]"), "{}", stderr(&output));
}

#[test]
fn json_mode_streams_the_workers_own_frames() {
    let output = run("ready-complete", &["--json", "validate-project", "p"]);
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));
    let lines: Vec<&str> = stdout(&output).lines().filter(|line| !line.is_empty()).collect();
    assert_eq!(lines.len(), 3, "ready plus two events: {lines:?}");
    assert!(lines[0].contains("\"frame\":\"ready\""), "{:?}", lines[0]);
    for line in &lines {
        serde_json::from_str::<serde_json::Value>(line)
            .unwrap_or_else(|error| panic!("{line:?} is not a JSON frame: {error}"));
    }
    assert!(
        lines[2].contains("completed"),
        "the last frame is the terminal event: {:?}",
        lines[2]
    );
}

#[test]
fn json_and_quiet_are_refused() {
    let output = run("ready-complete", &["--json", "--quiet", "capabilities"]);
    assert_eq!(code(&output), 3, "a usage error is a session error, not 2");
}

#[test]
fn an_invalid_task_id_is_refused_before_the_task_starts() {
    let output = run("ready-complete", &["--task-id", "nope", "validate-project", "p"]);
    assert_eq!(code(&output), 3);
    assert!(stderr(&output).contains("任务 ID 无效"), "{}", stderr(&output));
}

#[test]
fn a_task_failure_exits_one_and_prints_the_wire_code() {
    let output = run("fail", &["validate-project", "p"]);
    assert_eq!(code(&output), 1);
    let text = stderr(&output);
    assert!(text.contains("MEDIA_INVALID"), "{text}");
    assert!(text.contains("输入文件无法解析"), "{text}");
    assert!(text.contains("详情: ffprobe exited with status 1"), "{text}");
}

#[test]
fn a_cancelled_task_exits_two() {
    // The fake worker reports itself cancelled, so no signal is involved.
    let output = run("self-cancel", &["validate-project", "p"]);
    assert_eq!(code(&output), 2);
    assert!(stderr(&output).contains("任务已取消"), "{}", stderr(&output));
}

#[test]
fn a_protocol_mismatch_exits_three() {
    let output = run("bad-version", &["validate-project", "p"]);
    assert_eq!(code(&output), 3);
    assert!(stderr(&output).contains("协议版本不匹配"), "{}", stderr(&output));
}

#[test]
fn an_unsupported_command_names_the_variable_that_would_fix_it() {
    let output = run("only-validate", &["probe-media", "clip.mp4"]);
    assert_eq!(code(&output), 3);
    let text = stderr(&output);
    assert!(text.contains("FEATHERTALK_WORKER_FFPROBE"), "{text}");
    assert!(text.contains("validate_project"), "{text}");
}

#[test]
fn capabilities_reports_the_handshake() {
    let output = run("ready-complete", &["capabilities"]);
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));
    let text = stdout(&output);
    assert!(text.contains("cpu-0"), "{text}");
    assert!(text.contains("validate_project"), "{text}");
}

#[test]
fn an_empty_path_argument_is_refused() {
    let output = run("ready-complete", &["validate-project", ""]);
    assert_eq!(code(&output), 3);
    assert!(stderr(&output).contains("不能为空"), "{}", stderr(&output));
}
```

- [ ] **Step 9: Run the tests to verify they pass**

Run: `cargo test -p feathertalk-cli --all-targets`

Expected: PASS, 13 tests (10 CLI tests, 3 render unit tests).

If `stdout_carries_only_the_result` fails because stdout has more than one document, something is printing narration to stdout. The rule is absolute: only the result, only under the human mode.

- [ ] **Step 10: Run the linters**

Run: `cargo clippy -p feathertalk-cli --all-targets -- -D warnings`
Run: `cargo fmt --all -- --check`

Expected: both exit 0.

- [ ] **Step 11: Commit**

```bash
git add rust/Cargo.toml rust/Cargo.lock rust/crates/feathertalk-cli
git commit -m "feat(cli): add the feathertalk command line client"
```

`Cargo.lock` changes because `ctrlc` is new; stage it deliberately rather than leaving the tree dirty.

---

### Task 6: The real worker, the docs, and the full gate

Everything up to here was verified against a fake worker, which proves the client's own logic but not that the two crates agree on the protocol. This task adds an end-to-end test that drives the *real* `feathertalk-worker` binary, records the slice in the migration design, and runs the whole workspace gate.

The end-to-end test is deliberately narrow: three cases, one per exit code that a real worker can actually produce today. It does not re-test rendering or cancellation, which the fake worker covers deterministically. Its only job is to catch a divergence between the client's expectations and the worker's real behaviour.

**Files:**
- Create: `rust/crates/feathertalk-cli/tests/real_worker.rs`
- Modify: `docs/superpowers/specs/2026-08-17-rust-desktop-migration-design.md`
- Modify: `docs/superpowers/plans/2026-09-01-feathertalk-cli-worker-client.md` (tick every step)

**Interfaces:**
- Consumes: the `feathertalk` binary via `env!("CARGO_BIN_EXE_feathertalk")`; the `feathertalk-worker` binary as a sibling file on disk; `tempfile::TempDir`; `feathertalk_worker::ENV_FFPROBE` as a string literal, not a dependency.
- Produces: no library code. Only a test target and documentation.

- [ ] **Step 1: Write the real-worker end-to-end test**

Create `rust/crates/feathertalk-cli/tests/real_worker.rs`:

```rust
//! End-to-end coverage against the real worker binary.
//!
//! `cargo test -p feathertalk-cli` does not build `feathertalk-worker`, so there
//! is no `CARGO_BIN_EXE_feathertalk-worker`. The worker is found as a sibling of
//! this crate's own binary, which is where cargo puts every binary in the
//! workspace's shared target directory. When it is absent — a fresh clone that
//! only built this crate — each test prints why it skipped and passes, unless
//! `FEATHERTALK_REQUIRE_E2E=1` demands the real thing. CI sets that variable;
//! a developer running one crate's tests is not blocked by it.

use std::path::PathBuf;
use std::process::{Command, Output};

use tempfile::TempDir;

const CLI: &str = env!("CARGO_BIN_EXE_feathertalk");

/// The variable that makes a missing worker a failure instead of a skip.
const REQUIRE_E2E: &str = "FEATHERTALK_REQUIRE_E2E";

/// Locate `feathertalk-worker` next to the CLI binary under test.
fn worker_path() -> Option<PathBuf> {
    let cli = PathBuf::from(CLI);
    let path = cli
        .parent()?
        .join(format!("feathertalk-worker{}", std::env::consts::EXE_SUFFIX));
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
    println!("skipping {test}: feathertalk-worker is not built; run `cargo build -p feathertalk-worker`");
    None
}

/// Run the CLI against the real worker. `env` is applied to the CLI process,
/// which the worker inherits — that is how the worker's own configuration is
/// reached without the CLI knowing anything about it.
fn run(worker: &PathBuf, args: &[&str], env: &[(&str, &str)]) -> Output {
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
fn an_empty_directory_is_not_a_project() {
    let Some(worker) = worker_or_skip("an_empty_directory_is_not_a_project") else {
        return;
    };
    let project = TempDir::new().expect("a temporary directory is available");
    let path = project.path().to_string_lossy().into_owned();
    let output = run(&worker, &["validate-project", &path], &[]);
    // The worker maps every `ProjectError` except `AtomicReplacementUnsupported`
    // to `ErrorCode::MediaInvalid`, so a missing manifest is a task failure.
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
```

`FEATHERTALK_WORKER_FFPROBE` is written as a literal rather than imported from `feathertalk_worker::ENV_FFPROBE`, because taking a dev-dependency on the worker crate to reach one `&str` would make the CLI's test build depend on the whole worker.

- [ ] **Step 2: Run the end-to-end test against a built worker**

Run: `cargo build -p feathertalk-worker`
Run: `cargo test -p feathertalk-cli --test real_worker -- --nocapture`

Expected: PASS, 3 tests, and no "skipping" lines in the output. If a skip is printed, the worker binary did not land next to the CLI binary; check that both were built into the same target directory.

If `an_empty_directory_is_not_a_project` reports exit 0 instead of 1, the worker accepted an empty directory as a project. Do not weaken the assertion — that would be a worker bug worth reporting.

- [ ] **Step 3: Record the slice in the migration design**

In `docs/superpowers/specs/2026-08-17-rust-desktop-migration-design.md`, section 16 「完成定义」, replace this line:

```markdown
- CLI 覆盖所有 worker 能力，便于自动化测试和无界面运行。
```

with:

```markdown
- CLI 覆盖所有 worker 能力，便于自动化测试和无界面运行。当前 `feathertalk` 已覆盖 worker 的全部命令（`capabilities`、`validate-project`、`probe-media`），后续每新增一个 worker 命令都必须同步新增子命令。
```

The wording keeps the definition of done unchanged and states where the CLI stands against it today, so the next slice that adds a worker command knows the CLI is part of its scope.

- [ ] **Step 4: Tick every step in this plan**

Change every `- [ ]` in `docs/superpowers/plans/2026-09-01-feathertalk-cli-worker-client.md` to `- [x]`. The plan is the record of what was done; leaving the boxes empty makes it look abandoned.

- [ ] **Step 5: Run the full workspace gate**

From `rust/`:

Run: `cargo test --workspace --all-targets`
Run: `cargo check --workspace --all-targets`
Run: `cargo clippy --workspace --all-targets -- -D warnings`
Run: `cargo fmt --all -- --check`

Expected: the test count rises from the 707 passed / 0 failed / 13 ignored baseline and nothing regresses; the other three exit 0. The full suite takes roughly half an hour, so run it once, here, rather than after every task.

From the repo root:

Run: `git diff --check`

Expected: no output. This catches trailing whitespace and conflict markers that `cargo fmt` does not look at, including in the Markdown edited above.

- [ ] **Step 6: Commit**

```bash
git add rust/crates/feathertalk-cli/tests/real_worker.rs docs/superpowers
git commit -m "feat(cli): verify the CLI against the real worker"
```

Check `git status --porcelain` first: the only untracked entry left must be `?? demo/kanghui_training_video_featherhubert_188_latest/`.
