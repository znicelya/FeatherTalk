# FeatherTalk Worker Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `feathertalk-worker` crate so a real worker process speaks the slice 1 protocol over stdin/stdout: it announces its capabilities, accepts `validate_project` and `probe_media`, rejects every command it cannot run, streams stage events, honours hard cancellation of a running external process, and shuts down cleanly.

**Architecture:** One binary crate above `feathertalk-domain`, `feathertalk-media`, and `feathertalk-project`. Three threads, no shared mutexes: an input thread that owns the reader and decodes client frames, a control loop on the calling thread that is the sole owner of all task state and the frame writer, and one execution thread that runs commands. All coordination happens through `std::sync::mpsc` channels. Cancellation is a shared atomic flag threaded into the media process runner so a running `ffprobe` is killed rather than awaited. CPU-only: one adapter with the stable id `cpu-0`, guarded by an adapter lock table that is already general enough for the GPU adapters a later slice adds.

**Tech Stack:** Rust 2024 edition, rust-version 1.92, `std::thread` and `std::sync::mpsc` (no Tokio), `serde_json`, `thiserror`, `time`, `tempfile` for tests.

**Spec:** `docs/superpowers/specs/2026-09-01-feathertalk-worker-runtime-design.md`

## Global Constraints

- Run every cargo command from `E:/workspace/github/FeatherTalk/rust` unless a step says otherwise.
- Per-task verification is `cargo test -p <crate> --all-targets` for the crate that task touched. The final task adds `cargo test --workspace --all-targets`, `cargo check --workspace --all-targets`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --all -- --check`, and `git diff --check` (that last one from the repository root), all requiring exit code 0.
- Commit after every task. Never commit `demo/kanghui_training_video_featherhubert_188_latest/` — it is an untracked input fixture that must be left exactly as it is.
- `feathertalk-worker` production dependencies are exactly `feathertalk-domain`, `feathertalk-media`, `feathertalk-project`, `serde_json`, `thiserror`, `time`. No Tokio, no async runtime, no `burn`, no GPUI, no new third-party crate.
- The worker never spawns a subprocess per task. Task execution happens on a thread inside the worker process; the only subprocesses are the media toolchain binaries (`ffprobe`, `ffmpeg`).
- `serde_json` is compiled without the `preserve_order` feature anywhere in this workspace, so `serde_json::Value::Object` is a `BTreeMap` and any JSON built through `json!` or `to_value` serializes its keys in alphabetical order. Byte-exact golden strings in `feathertalk-domain` must be written in that order. Worker-side probe results are asserted structurally (key by key), never byte-exact, because their key order is an implementation detail of `serde_json`.
- Every frame is `validate()`d before it is written and after it is decoded. `FrameReader` and `FrameWriter` are syntax-only by contract; semantic checking is the caller's job.
- User-facing `summary` strings are Chinese; `detail` strings are English. This matches the `FAILED_EVENT` golden already checked in.
- No file under `rust/crates/feathertalk-project/` is modified by this plan.
- The full workspace suite takes roughly 30 minutes on a warm target directory. The 13 pre-existing ignored tests — six subprocess helpers, six gated on a certified WGPU adapter, one gated on a licensed VGG19 package — stay ignored.

---

## File Structure

```text
rust/Cargo.toml                                        Modify: add workspace member (Task 4)
rust/crates/feathertalk-domain/
  src/lib.rs                                           Modify: PROTOCOL_VERSION = 2 (Task 1)
  src/frame.rs                                         Modify: ReadyFrame.supported_commands (Task 1)
  src/event.rs                                         Modify: Event.result (Task 2)
  tests/public_api.rs                                  Modify: version assertion (Task 1)
  tests/handshake.rs                                   Modify: supported_commands tests (Task 1)
  tests/frame_codec.rs                                 Modify: literals and fixtures (Tasks 1, 2)
  tests/golden_frames.rs                               Modify: goldens (Tasks 1, 2)
  tests/event.rs                                       Modify: result validation tests (Task 2)
rust/crates/feathertalk-media/
  src/error.rs                                         Modify: MediaError::ToolCancelled (Task 3)
  src/process.rs                                       Modify: CancellationToken, CancellableProcessRunner (Task 3)
  src/lib.rs                                           Modify: re-exports (Task 3)
  tests/probe_execution.rs                             Modify: cancellation surfaces through probe (Task 3)
rust/crates/feathertalk-worker/
  Cargo.toml                                           Create (Task 4)
  src/lib.rs                                           Create (Task 4), modified by Tasks 5-8
  src/config.rs                                        Create (Task 4)
  src/handshake.rs                                     Create (Task 4)
  src/error_map.rs                                     Create (Task 5)
  src/probe_result.rs                                  Create (Task 6)
  src/commands.rs                                      Create (Task 6)
  src/adapters.rs                                      Create (Task 7)
  src/runtime.rs                                       Create (Task 8)
  src/main.rs                                          Create (Task 8)
  tests/handshake.rs                                   Create (Task 4)
  tests/error_mapping.rs                               Create (Task 5)
  tests/commands.rs                                    Create (Task 6)
  tests/adapter_locks.rs                               Create (Task 7)
  tests/runtime.rs                                     Create (Task 8)
  tests/process_boundary.rs                            Create (Task 8)
docs/superpowers/specs/2026-08-17-rust-desktop-migration-design.md   Modify: Task 9
```

---

### Task 1: Protocol version 2 and the supported-command handshake

The worker can only run two of the thirteen commands. Rather than discovering that by failing a task, the desktop learns it from the handshake: `ReadyFrame` gains a required `supported_commands: Vec<TaskKind>`. That is a breaking wire change, so `PROTOCOL_VERSION` moves to 2 in the same task.

**Files:**
- Modify: `rust/crates/feathertalk-domain/src/lib.rs`
- Modify: `rust/crates/feathertalk-domain/src/frame.rs`
- Test: `rust/crates/feathertalk-domain/tests/handshake.rs`
- Test: `rust/crates/feathertalk-domain/tests/public_api.rs`
- Test: `rust/crates/feathertalk-domain/tests/frame_codec.rs`
- Test: `rust/crates/feathertalk-domain/tests/golden_frames.rs`

**Interfaces:**
- Consumes: `TaskKind` (already `Copy + Ord`), `DomainError::InvalidField`, `check_protocol_version`.
- Produces: `PROTOCOL_VERSION: u32 = 2`; `ReadyFrame.supported_commands: Vec<TaskKind>` positioned between `adapters` and `capabilities`; `ReadyFrame::validate` rejecting an empty or duplicated command list.

- [ ] **Step 1: Write the failing tests**

In `rust/crates/feathertalk-domain/tests/handshake.rs`, add `TaskKind` to the import list, add the new field to the `ready()` fixture, and append four tests.

The import becomes:

```rust
use feathertalk_domain::{
    AdapterInfo, AdapterKind, Backend, CancelFrame, Capabilities, ClientFrame, DomainError,
    PROTOCOL_VERSION, ReadyFrame, ServerFrame, TaskId, TaskKind,
};
```

In `ready()`, insert the field between `adapters` and `capabilities`:

```rust
        adapters: adapters(),
        supported_commands: vec![TaskKind::ValidateProject, TaskKind::ProbeMedia],
        capabilities: Capabilities {
```

Append:

```rust
#[test]
fn the_protocol_version_is_two() {
    assert_eq!(PROTOCOL_VERSION, 2);
    assert_eq!(ready().protocol_version, 2);
}

#[test]
fn a_worker_reporting_no_supported_command_is_rejected() {
    let mut params = ready();
    params.supported_commands.clear();
    assert!(matches!(
        params.validate(),
        Err(DomainError::InvalidField {
            field: "supported_commands",
            ..
        })
    ));
}

#[test]
fn duplicate_supported_commands_are_rejected() {
    let mut params = ready();
    params.supported_commands = vec![TaskKind::ProbeMedia, TaskKind::ProbeMedia];
    assert!(matches!(
        params.validate(),
        Err(DomainError::InvalidField {
            field: "supported_commands",
            ..
        })
    ));
}

#[test]
fn supported_commands_travel_as_task_slugs() {
    let json = serde_json::to_string(&ready()).unwrap();
    assert!(
        json.contains(r#""supported_commands":["validate_project","probe_media"]"#),
        "{json}"
    );
    let restored: ReadyFrame = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, ready());
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p feathertalk-domain --all-targets`

Expected: FAIL to compile, with `error[E0560]: struct `ReadyFrame` has no field named `supported_commands`` from the `ready()` fixture and `error[E0609]` from the two mutation tests.

- [ ] **Step 3: Bump the protocol version**

In `rust/crates/feathertalk-domain/src/lib.rs`, change the constant:

```rust
pub const PROTOCOL_VERSION: u32 = 2;
```

Leave the doc comment above it, but make sure it does not claim version 1. If it names a version, say: version 2 added `supported_commands` to the handshake and `result` to completed events.

- [ ] **Step 4: Add the field and its validation**

In `rust/crates/feathertalk-domain/src/frame.rs`, add `TaskKind` to the crate import:

```rust
use crate::{DomainError, Event, Request, TaskId, TaskKind, check_protocol_version};
```

Add the field to `ReadyFrame`, between `adapters` and `capabilities`:

```rust
    pub adapters: Vec<AdapterInfo>,
    /// Commands this worker will actually accept. A `start` frame naming any
    /// other command is rejected, so the desktop can grey out unsupported
    /// actions instead of discovering them through a failed task.
    pub supported_commands: Vec<TaskKind>,
    pub capabilities: Capabilities,
```

Append this to the end of `ReadyFrame::validate`, immediately before the closing `Ok(())`:

```rust
        if self.supported_commands.is_empty() {
            return Err(DomainError::InvalidField {
                field: "supported_commands",
                reason: "a worker must report at least one supported command".into(),
            });
        }
        let mut commands = BTreeSet::new();
        for command in &self.supported_commands {
            if !commands.insert(*command) {
                return Err(DomainError::InvalidField {
                    field: "supported_commands",
                    reason: format!("duplicate supported command {}", command.as_slug()),
                });
            }
        }
```

`BTreeSet` is already imported at the top of the file, and the `seen` set above uses `&str` keys, so this second set with `TaskKind` keys needs its own binding name.

- [ ] **Step 5: Update the tests that pinned version 1**

`rust/crates/feathertalk-domain/tests/public_api.rs` line 5 — rename the test and change the constant:

```rust
#[test]
fn protocol_version_is_two() {
    assert_eq!(PROTOCOL_VERSION, 2);
}
```

`rust/crates/feathertalk-domain/tests/handshake.rs` line 109 — the literal must carry the new version and the new field, otherwise the test would pass for the wrong reason (a missing required field rather than the rejected `extra` key):

```rust
    let json = r#"{"frame":"ready","data":{"protocol_version":2,"worker_version":"0.1.0","backends":["cpu"],"adapters":[],"supported_commands":["probe_media"],"capabilities":{"training":false,"wgpu_training":false,"onnx_validation":false,"ffmpeg":true}},"extra":1}"#;
```

`rust/crates/feathertalk-domain/tests/frame_codec.rs` — change `"protocol_version":1` to `"protocol_version":2` in the three string literals at lines 80, 94, and 97, and add the field to `crate_ready()` (near line 179):

```rust
        adapters: vec![],
        supported_commands: vec![feathertalk_domain::TaskKind::ProbeMedia],
        capabilities: feathertalk_domain::Capabilities {
```

`rust/crates/feathertalk-domain/tests/golden_frames.rs` — change `"protocol_version":1` to `"protocol_version":2` in all five constants (lines 9, 12, 14, 50, 52). Do not touch anything else in those strings; Task 2 changes the two event goldens again.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test -p feathertalk-domain --all-targets`

Expected: PASS, exit code 0. If a golden string still says version 1 the failure is a byte-for-byte string comparison, and the diff points at the exact constant.

- [ ] **Step 7: Commit**

```bash
git add rust/crates/feathertalk-domain
git commit -m "feat(domain): report supported commands in the handshake at protocol version 2"
```

---

### Task 2: Command results on completed events

`probe_media` produces data the desktop needs. The event envelope has nowhere to put it, so `Event` gains `result: Option<serde_json::Value>`, permitted only on `TaskStage::Completed` and required to be a JSON object. `validate_project` returns `None`; a JSON object keeps room for future fields without another wire break.

**Files:**
- Modify: `rust/crates/feathertalk-domain/src/event.rs`
- Test: `rust/crates/feathertalk-domain/tests/event.rs`
- Test: `rust/crates/feathertalk-domain/tests/frame_codec.rs`
- Test: `rust/crates/feathertalk-domain/tests/golden_frames.rs`

**Interfaces:**
- Consumes: `serde_json::Value`, `TaskStage::Completed`, `DomainError::InvalidField`.
- Produces: `Event.result: Option<serde_json::Value>` as the last field of the struct and of the wire object; `Event::new` initialising it to `None`; `Event::validate` rejecting a result on a non-completed stage and a non-object result.

- [ ] **Step 1: Write the failing tests**

In `rust/crates/feathertalk-domain/tests/event.rs`, extend the first test with one assertion and append three tests.

In `a_new_event_carries_the_protocol_version_and_empty_metrics`, after `assert_eq!(event.error, None);`:

```rust
    assert_eq!(event.result, None);
```

Append:

```rust
#[test]
fn a_completed_stage_may_carry_a_result_object() {
    let mut event = Event::new(task_id(), NOW, TaskStage::Completed);
    event.validate().unwrap();
    event.result = Some(serde_json::json!({"format_name": "mov,mp4"}));
    event.validate().unwrap();
}

#[test]
fn only_a_completed_stage_may_carry_a_result() {
    for stage in [
        TaskStage::Preparing,
        TaskStage::Rendering {
            frame: 1,
            total: 10,
        },
        TaskStage::Cancelled,
    ] {
        let mut event = Event::new(task_id(), NOW, stage);
        event.result = Some(serde_json::json!({}));
        assert!(matches!(
            event.validate(),
            Err(DomainError::InvalidField {
                field: "result",
                ..
            })
        ));
    }
}

#[test]
fn a_result_must_be_a_json_object() {
    for value in [
        serde_json::json!(7),
        serde_json::json!("mov,mp4"),
        serde_json::json!([1, 2]),
        serde_json::Value::Null,
    ] {
        let mut event = Event::new(task_id(), NOW, TaskStage::Completed);
        event.result = Some(value);
        assert!(matches!(
            event.validate(),
            Err(DomainError::InvalidField {
                field: "result",
                ..
            })
        ));
    }
}
```

In `rust/crates/feathertalk-domain/tests/golden_frames.rs`, add the new golden next to the existing two (a single flat key keeps the assertion independent of `serde_json` map ordering):

```rust
const COMPLETED_EVENT: &str = r#"{"frame":"event","data":{"protocol_version":2,"task_id":"1787900000000-0000000a","emitted_at":"2026-09-01T09:00:00Z","stage":{"stage":"completed"},"progress":null,"metrics":{"samples_per_second":null,"eta_seconds":null,"vram_bytes":null},"error":null,"result":{"format_name":"mov,mp4"}}}"#;
```

and the test that pins it:

```rust
#[test]
fn a_completed_event_carries_its_command_result() {
    let mut event = Event::new(task_id(), "2026-09-01T09:00:00Z", TaskStage::Completed);
    event.result = Some(serde_json::json!({"format_name": "mov,mp4"}));
    event.validate().unwrap();
    assert_eq!(
        encode_line(&ServerFrame::Event(event)).unwrap(),
        COMPLETED_EVENT
    );
}
```

Add `COMPLETED_EVENT` to the array in `golden_server_lines_still_decode`:

```rust
    for line in [TRAINING_EVENT, FAILED_EVENT, COMPLETED_EVENT] {
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p feathertalk-domain --all-targets`

Expected: FAIL to compile with `error[E0609]: no field `result` on type `Event``.

- [ ] **Step 3: Add the field and its validation**

In `rust/crates/feathertalk-domain/src/event.rs`, add the field as the last member of `Event` — the wire order follows the declaration order, and the goldens depend on `result` coming last:

```rust
    pub error: Option<TaskError>,
    /// Command output for a successful task. Only a `Completed` stage may set
    /// it, and it is always a JSON object so a command can add fields later
    /// without another protocol break.
    pub result: Option<serde_json::Value>,
```

In `Event::new`, add `result: None,` after `error: None,`.

In `Event::validate`, after the existing `match (&self.error, &self.stage)` block and before `Ok(())`:

```rust
        match (&self.result, &self.stage) {
            (None, _) => {}
            (Some(result), TaskStage::Completed) => {
                if !result.is_object() {
                    return Err(DomainError::InvalidField {
                        field: "result",
                        reason: "a command result must be a JSON object".into(),
                    });
                }
            }
            (Some(_), _) => {
                return Err(DomainError::InvalidField {
                    field: "result",
                    reason: "only a completed stage may carry a command result".into(),
                });
            }
        }
```

`Event` derives `PartialEq` but not `Eq`, and `serde_json::Value` is `PartialEq` only, so no derive changes are needed. `serde_json` is already a production dependency of this crate.

- [ ] **Step 4: Update the fixtures and goldens the new field breaks**

`rust/crates/feathertalk-domain/tests/frame_codec.rs` — the `Event` struct literal near line 164 is exhaustive, so add the field after `error: None,`:

```rust
        error: None,
        result: None,
```

`rust/crates/feathertalk-domain/tests/golden_frames.rs` — `TRAINING_EVENT` and `FAILED_EVENT` now end with `,"error":null,"result":null}}` and `...,"recovery":"free_disk_space"},"result":null}}` respectively. Change only the tail of each string.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p feathertalk-domain --all-targets`

Expected: PASS, exit code 0.

- [ ] **Step 6: Commit**

```bash
git add rust/crates/feathertalk-domain
git commit -m "feat(domain): carry command results on completed events"
```

---

### Task 3: Cancellable process execution in feathertalk-media

Hard cancellation means a running `ffprobe` is killed, not awaited. The media crate owns process spawning, so it gains a cancellation token and a runner that polls it. `SystemProcessRunner` keeps its exact current behaviour by passing `None`.

**Files:**
- Modify: `rust/crates/feathertalk-media/src/error.rs`
- Modify: `rust/crates/feathertalk-media/src/process.rs` (including its in-file `mod tests`)
- Modify: `rust/crates/feathertalk-media/src/lib.rs`
- Test: `rust/crates/feathertalk-media/tests/probe_execution.rs`

**Interfaces:**
- Consumes: `CommandSpec`, `MediaError`, the private `read_limited` / `validate_executable` / `ReadResult` helpers in `process.rs`.
- Produces: `MediaError::ToolCancelled { operation: &'static str }`; `CancellationToken` with `new()`, `cancel(&self)`, `is_cancelled(&self) -> bool`, `Clone`, `Default`; `CancellableProcessRunner::new(CancellationToken)` implementing `ProcessRunner`.

- [ ] **Step 1: Write the failing tests**

Append to the `#[cfg(test)] mod tests` block in `rust/crates/feathertalk-media/src/process.rs`, above the `#[ignore]` helpers so the file keeps its "tests first, helpers last" shape:

```rust
    #[test]
    fn cancelling_mid_run_kills_the_child() {
        let token = CancellationToken::new();
        let runner = CancellableProcessRunner::new(token.clone());
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            token.cancel();
        });
        let started = Instant::now();
        let error = runner
            .run(&helper_command("helper_sleep"), Duration::from_secs(30))
            .expect_err("cancellation must surface as an error");
        assert!(
            matches!(
                error,
                MediaError::ToolCancelled {
                    operation: "test_process"
                }
            ),
            "{error:?}"
        );
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "cancellation waited {:?} for a child that sleeps 5 s",
            started.elapsed()
        );
    }

    #[test]
    fn an_already_cancelled_token_never_spawns() {
        let token = CancellationToken::new();
        token.cancel();
        let runner = CancellableProcessRunner::new(token);
        let started = Instant::now();
        let error = runner
            .run(&helper_command("helper_sleep"), Duration::from_secs(30))
            .expect_err("a cancelled token must refuse to start work");
        assert!(matches!(error, MediaError::ToolCancelled { .. }), "{error:?}");
        assert!(started.elapsed() < Duration::from_millis(500));
    }

    #[test]
    fn an_uncancelled_token_behaves_like_the_system_runner() {
        let runner = CancellableProcessRunner::new(CancellationToken::new());
        let output = runner
            .run(&helper_command("helper_success"), Duration::from_secs(10))
            .unwrap();
        assert_eq!(output.exit_code(), Some(0));
        assert!(String::from_utf8_lossy(output.stdout()).contains("helper-output"));
    }
```

`use super::*` already brings `Instant` into the test module, and `thread` and `Duration` are already imported there.

Append to `rust/crates/feathertalk-media/tests/probe_execution.rs` a test proving the error survives the probe layer unchanged, so the worker can recognise it:

```rust
#[test]
fn a_cancelled_probe_surfaces_as_tool_cancelled() {
    let toolchain = toolchain();
    let (_temp, input) = input();
    let runner = FakeRunner::new(vec![Err(MediaError::ToolCancelled {
        operation: "probe",
    })]);
    let error = probe_media_with_runner(&input, &toolchain, &runner)
        .expect_err("a cancelled probe must not report success");
    assert!(
        matches!(
            error,
            MediaError::ToolCancelled {
                operation: "probe"
            }
        ),
        "{error:?}"
    );
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p feathertalk-media --all-targets`

Expected: FAIL to compile with `error[E0433]: failed to resolve: use of undeclared type `CancellationToken`` and `error[E0599]: no variant or associated item named `ToolCancelled` found for enum `MediaError``.

- [ ] **Step 3: Add the error variant**

In `rust/crates/feathertalk-media/src/error.rs`, add the variant next to `ToolTimedOut`:

```rust
    #[error("media tool was cancelled during {operation}")]
    ToolCancelled { operation: &'static str },
```

- [ ] **Step 4: Extract the shared run body and add the cancellable runner**

In `rust/crates/feathertalk-media/src/process.rs`, extend the std import:

```rust
use std::{
    io::Read,
    process::{Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
```

Replace the body of `impl ProcessRunner for SystemProcessRunner` with a delegation:

```rust
impl ProcessRunner for SystemProcessRunner {
    fn run(&self, command: &CommandSpec, timeout: Duration) -> Result<ProcessOutput, MediaError> {
        run_child(command, timeout, None)
    }
}

/// Cooperative cancellation flag shared between the thread that requests a stop
/// and the thread waiting on a child process.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

/// A [`SystemProcessRunner`] that kills its child as soon as its token is
/// cancelled instead of waiting for the timeout.
#[derive(Debug, Clone)]
pub struct CancellableProcessRunner {
    token: CancellationToken,
}

impl CancellableProcessRunner {
    pub fn new(token: CancellationToken) -> Self {
        Self { token }
    }
}

impl ProcessRunner for CancellableProcessRunner {
    fn run(&self, command: &CommandSpec, timeout: Duration) -> Result<ProcessOutput, MediaError> {
        run_child(command, timeout, Some(&self.token))
    }
}

fn run_child(
    command: &CommandSpec,
    timeout: Duration,
    token: Option<&CancellationToken>,
) -> Result<ProcessOutput, MediaError> {
    validate_executable(command)?;
    if token.is_some_and(CancellationToken::is_cancelled) {
        return Err(MediaError::ToolCancelled {
            operation: command.operation(),
        });
    }
    let mut child = Command::new(command.executable())
        .args(command.arguments())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| MediaError::ToolSpawn {
            operation: command.operation(),
            message: error.to_string(),
        })?;
    let mut stdout = child.stdout.take().expect("stdout was piped");
    let mut stderr = child.stderr.take().expect("stderr was piped");
    let stdout_thread = thread::spawn(move || read_limited(&mut stdout));
    let stderr_thread = thread::spawn(move || read_limited(&mut stderr));
    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if token.is_some_and(CancellationToken::is_cancelled) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdout_thread.join();
                let _ = stderr_thread.join();
                return Err(MediaError::ToolCancelled {
                    operation: command.operation(),
                });
            }
            Ok(None) if started.elapsed() >= timeout => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdout_thread.join();
                let _ = stderr_thread.join();
                return Err(MediaError::ToolTimedOut {
                    operation: command.operation(),
                    timeout_ms: timeout.as_millis().min(u128::from(u64::MAX)) as u64,
                });
            }
            Ok(None) => thread::sleep(Duration::from_millis(5)),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdout_thread.join();
                let _ = stderr_thread.join();
                return Err(MediaError::ToolSpawn {
                    operation: command.operation(),
                    message: error.to_string(),
                });
            }
        }
    };
    let stdout = stdout_thread
        .join()
        .unwrap_or_else(|_| ReadResult::Error("stdout reader panicked".to_owned()));
    let stderr = stderr_thread
        .join()
        .unwrap_or_else(|_| ReadResult::Error("stderr reader panicked".to_owned()));
    let stdout = stdout.into_result(command.operation(), "stdout")?;
    let stderr = stderr.into_result(command.operation(), "stderr")?;
    Ok(ProcessOutput::new(status.code(), stdout, stderr))
}
```

The cancellation arm goes **before** the timeout arm: match arms are tried in order, and a cancelled task that has also run past its timeout should report cancellation, which is what the operator asked for.

- [ ] **Step 5: Export the new types**

In `rust/crates/feathertalk-media/src/lib.rs`, replace the `process` re-export line (rustfmt sorts the braces alphabetically):

```rust
pub use process::{
    CancellableProcessRunner, CancellationToken, MAX_CAPTURE_BYTES, ProcessOutput, ProcessRunner,
    SystemProcessRunner,
};
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test -p feathertalk-media --all-targets`

Expected: PASS, exit code 0. `cancelling_mid_run_kills_the_child` should finish in well under a second; if it takes five, the child was awaited rather than killed.

- [ ] **Step 7: Commit**

```bash
git add rust/crates/feathertalk-media
git commit -m "feat(media): kill running tools on cancellation"
```

---

### Task 4: Worker crate, configuration, and the handshake it reports

**Files:**
- Modify: `rust/Cargo.toml` (workspace `members`)
- Create: `rust/crates/feathertalk-worker/Cargo.toml`
- Create: `rust/crates/feathertalk-worker/src/lib.rs`
- Create: `rust/crates/feathertalk-worker/src/config.rs`
- Create: `rust/crates/feathertalk-worker/src/handshake.rs`
- Test: `rust/crates/feathertalk-worker/tests/handshake.rs`

**Interfaces:**
- Consumes: `MediaToolchain::new(ffmpeg: PathBuf, ffprobe: PathBuf, timeout: Duration) -> Result<MediaToolchain, MediaError>`; `ReadyFrame`, `AdapterInfo`, `Backend`, `AdapterKind`, `Capabilities`, `TaskKind`, `PROTOCOL_VERSION`.
- Produces: `WorkerConfig::from_env() -> WorkerConfig`, `WorkerConfig::from_values(ffprobe: Option<String>, ffmpeg: Option<String>, timeout_ms: Option<String>) -> WorkerConfig`, `WorkerConfig::worker_version(&self) -> &str`, `WorkerConfig::media(&self) -> Option<&MediaToolchain>`, `WorkerConfig::media_rejection(&self) -> Option<&str>`; `CPU_ADAPTER_ID: &str`, `cpu_adapter() -> AdapterInfo`, `supported_commands(&WorkerConfig) -> Vec<TaskKind>`, `ready_frame(&WorkerConfig) -> ReadyFrame`; the env var names `ENV_FFPROBE`, `ENV_FFMPEG`, `ENV_MEDIA_TIMEOUT_MS` and `DEFAULT_MEDIA_TIMEOUT_MS`.

`src/main.rs` arrives in Task 8; until then this crate is a library only.

- [ ] **Step 1: Write the failing tests**

Create `rust/crates/feathertalk-worker/tests/handshake.rs`:

```rust
use std::time::Duration;

use feathertalk_domain::{AdapterKind, Backend, TaskKind};
use feathertalk_worker::{
    CPU_ADAPTER_ID, DEFAULT_MEDIA_TIMEOUT_MS, ENV_FFPROBE, ENV_MEDIA_TIMEOUT_MS, WorkerConfig,
    ready_frame, supported_commands,
};

fn absolute(name: &str) -> String {
    std::env::current_dir()
        .unwrap()
        .join(name)
        .display()
        .to_string()
}

fn configured() -> WorkerConfig {
    WorkerConfig::from_values(
        Some(absolute("ffprobe-test")),
        Some(absolute("ffmpeg-test")),
        None,
    )
}

#[test]
fn a_configured_worker_reports_a_cpu_adapter_and_both_commands() {
    let config = configured();
    assert_eq!(config.media_rejection(), None);
    let frame = ready_frame(&config);
    frame.validate().unwrap();
    assert_eq!(frame.worker_version, env!("CARGO_PKG_VERSION"));
    assert_eq!(frame.backends, vec![Backend::Cpu]);
    assert_eq!(frame.adapters.len(), 1);
    assert_eq!(frame.adapters[0].id, CPU_ADAPTER_ID);
    assert_eq!(frame.adapters[0].backend, Backend::Cpu);
    assert_eq!(frame.adapters[0].kind, AdapterKind::Cpu);
    assert!(frame.adapters[0].certified);
    assert_eq!(frame.adapters[0].vram_bytes, None);
    assert_eq!(
        frame.supported_commands,
        vec![TaskKind::ValidateProject, TaskKind::ProbeMedia]
    );
    assert!(frame.capabilities.ffmpeg);
    assert!(!frame.capabilities.training);
    assert!(!frame.capabilities.wgpu_training);
    assert!(!frame.capabilities.onnx_validation);
}

#[test]
fn a_worker_without_a_media_toolchain_only_offers_project_validation() {
    let config = WorkerConfig::from_values(None, None, None);
    assert!(config.media().is_none());
    assert!(
        config
            .media_rejection()
            .is_some_and(|reason| reason.contains(ENV_FFPROBE))
    );
    let frame = ready_frame(&config);
    frame.validate().unwrap();
    assert_eq!(frame.supported_commands, vec![TaskKind::ValidateProject]);
    assert!(!frame.capabilities.ffmpeg);
    assert_eq!(supported_commands(&config).len(), 1);
}

#[test]
fn a_relative_tool_path_is_rejected_with_the_variable_name() {
    let config = WorkerConfig::from_values(
        Some("ffprobe.exe".to_owned()),
        Some(absolute("ffmpeg-test")),
        None,
    );
    let reason = config.media_rejection().expect("relative path must reject");
    assert!(reason.contains(ENV_FFPROBE), "{reason}");
    assert!(reason.contains("absolute"), "{reason}");
}

#[test]
fn an_empty_tool_path_is_rejected() {
    let config = WorkerConfig::from_values(
        Some("   ".to_owned()),
        Some(absolute("ffmpeg-test")),
        None,
    );
    assert!(
        config
            .media_rejection()
            .is_some_and(|reason| reason.contains(ENV_FFPROBE))
    );
}

#[test]
fn an_unusable_timeout_is_rejected_with_the_variable_name() {
    for bad in ["0", "abc", "-1", ""] {
        let config = WorkerConfig::from_values(
            Some(absolute("ffprobe-test")),
            Some(absolute("ffmpeg-test")),
            Some(bad.to_owned()),
        );
        let reason = config
            .media_rejection()
            .unwrap_or_else(|| panic!("expected rejection for {bad:?}"));
        assert!(reason.contains(ENV_MEDIA_TIMEOUT_MS), "{bad:?}: {reason}");
    }
}

#[test]
fn the_default_media_timeout_is_five_minutes() {
    assert_eq!(DEFAULT_MEDIA_TIMEOUT_MS, 300_000);
    let config = configured();
    assert_eq!(
        config.media().unwrap().timeout(),
        Duration::from_millis(DEFAULT_MEDIA_TIMEOUT_MS)
    );

    let explicit = WorkerConfig::from_values(
        Some(absolute("ffprobe-test")),
        Some(absolute("ffmpeg-test")),
        Some("1500".to_owned()),
    );
    assert_eq!(
        explicit.media().unwrap().timeout(),
        Duration::from_millis(1500)
    );
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p feathertalk-worker --all-targets`

Expected: FAIL with `error: package ID specification 'feathertalk-worker' did not match any packages`, because the crate does not exist yet.

- [ ] **Step 3: Register the crate**

In `rust/Cargo.toml`, add `"crates/feathertalk-worker",` to `members`, immediately after `"crates/feathertalk-domain",`. Leave `exclude` and every other entry untouched.

Create `rust/crates/feathertalk-worker/Cargo.toml`:

```toml
[package]
name = "feathertalk-worker"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[dependencies]
feathertalk-domain = { path = "../feathertalk-domain" }
feathertalk-media = { path = "../feathertalk-media" }
feathertalk-project = { path = "../feathertalk-project" }
serde_json = { workspace = true }
thiserror = { workspace = true }
time = { workspace = true }

[dev-dependencies]
tempfile = { workspace = true }
```

- [ ] **Step 4: Implement the configuration**

Create `rust/crates/feathertalk-worker/src/config.rs`:

```rust
use std::{path::PathBuf, time::Duration};

use feathertalk_media::MediaToolchain;

pub const ENV_FFPROBE: &str = "FEATHERTALK_WORKER_FFPROBE";
pub const ENV_FFMPEG: &str = "FEATHERTALK_WORKER_FFMPEG";
pub const ENV_MEDIA_TIMEOUT_MS: &str = "FEATHERTALK_WORKER_MEDIA_TIMEOUT_MS";
pub const DEFAULT_MEDIA_TIMEOUT_MS: u64 = 300_000;

/// Everything the worker learns from its environment at startup.
///
/// A missing or unusable media toolchain is not a startup failure: the worker
/// still serves `validate_project` and simply reports `probe_media` as
/// unsupported, with the reason kept for the rejection message.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    worker_version: String,
    media: Option<MediaToolchain>,
    media_rejection: Option<String>,
}

impl WorkerConfig {
    pub fn from_env() -> Self {
        Self::from_values(
            std::env::var(ENV_FFPROBE).ok(),
            std::env::var(ENV_FFMPEG).ok(),
            std::env::var(ENV_MEDIA_TIMEOUT_MS).ok(),
        )
    }

    pub fn from_values(
        ffprobe: Option<String>,
        ffmpeg: Option<String>,
        timeout_ms: Option<String>,
    ) -> Self {
        let (media, media_rejection) = match media_toolchain(ffprobe, ffmpeg, timeout_ms) {
            Ok(toolchain) => (Some(toolchain), None),
            Err(reason) => (None, Some(reason)),
        };
        Self {
            worker_version: env!("CARGO_PKG_VERSION").to_owned(),
            media,
            media_rejection,
        }
    }

    pub fn worker_version(&self) -> &str {
        &self.worker_version
    }

    pub fn media(&self) -> Option<&MediaToolchain> {
        self.media.as_ref()
    }

    pub fn media_rejection(&self) -> Option<&str> {
        self.media_rejection.as_deref()
    }
}

fn media_toolchain(
    ffprobe: Option<String>,
    ffmpeg: Option<String>,
    timeout_ms: Option<String>,
) -> Result<MediaToolchain, String> {
    let ffprobe = required_path(ffprobe, ENV_FFPROBE)?;
    let ffmpeg = required_path(ffmpeg, ENV_FFMPEG)?;
    let timeout_ms = match timeout_ms {
        None => DEFAULT_MEDIA_TIMEOUT_MS,
        Some(value) => value.trim().parse::<u64>().map_err(|_| {
            format!("{ENV_MEDIA_TIMEOUT_MS} must be a whole number of milliseconds, got {value:?}")
        })?,
    };
    if timeout_ms == 0 {
        return Err(format!("{ENV_MEDIA_TIMEOUT_MS} must be greater than zero"));
    }
    MediaToolchain::new(ffmpeg, ffprobe, Duration::from_millis(timeout_ms))
        .map_err(|error| error.to_string())
}

fn required_path(value: Option<String>, variable: &str) -> Result<PathBuf, String> {
    let value = value.ok_or_else(|| format!("{variable} is not set"))?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(format!("{variable} must not be empty"));
    }
    let path = PathBuf::from(trimmed);
    if !path.is_absolute() {
        return Err(format!(
            "{variable} must be an absolute path, got {trimmed:?}"
        ));
    }
    Ok(path)
}
```

`MediaToolchain::new` takes `ffmpeg` first and `ffprobe` second; the argument order above is deliberate. It also checks absoluteness itself, but its message does not name the environment variable, which is the whole point of `required_path`. Neither binary needs to exist at configuration time — the media crate reports a missing tool when it tries to spawn it.

- [ ] **Step 5: Implement the handshake**

Create `rust/crates/feathertalk-worker/src/handshake.rs`:

```rust
use feathertalk_domain::{
    AdapterInfo, AdapterKind, Backend, Capabilities, PROTOCOL_VERSION, ReadyFrame, TaskKind,
};

use crate::WorkerConfig;

/// Stable identity of the single CPU adapter this slice exposes. The adapter
/// lock table keys on it, so it must not change between worker restarts.
pub const CPU_ADAPTER_ID: &str = "cpu-0";

pub fn cpu_adapter() -> AdapterInfo {
    AdapterInfo {
        id: CPU_ADAPTER_ID.to_owned(),
        name: "CPU".to_owned(),
        backend: Backend::Cpu,
        kind: AdapterKind::Cpu,
        certified: true,
        vram_bytes: None,
    }
}

pub fn supported_commands(config: &WorkerConfig) -> Vec<TaskKind> {
    let mut commands = vec![TaskKind::ValidateProject];
    if config.media().is_some() {
        commands.push(TaskKind::ProbeMedia);
    }
    commands
}

pub fn ready_frame(config: &WorkerConfig) -> ReadyFrame {
    ReadyFrame {
        protocol_version: PROTOCOL_VERSION,
        worker_version: config.worker_version().to_owned(),
        backends: vec![Backend::Cpu],
        adapters: vec![cpu_adapter()],
        supported_commands: supported_commands(config),
        capabilities: Capabilities {
            training: false,
            wgpu_training: false,
            onnx_validation: false,
            ffmpeg: config.media().is_some(),
        },
    }
}
```

Create `rust/crates/feathertalk-worker/src/lib.rs`:

```rust
//! The FeatherTalk worker: a JSON Lines command server over stdin/stdout.
//!
//! This slice serves `validate_project` and `probe_media` on the CPU. Every
//! other command in [`feathertalk_domain::TaskKind`] is reported as unsupported
//! in the handshake and rejected if a client asks for it anyway.

mod config;
mod handshake;

pub use config::{
    DEFAULT_MEDIA_TIMEOUT_MS, ENV_FFMPEG, ENV_FFPROBE, ENV_MEDIA_TIMEOUT_MS, WorkerConfig,
};
pub use handshake::{CPU_ADAPTER_ID, cpu_adapter, ready_frame, supported_commands};
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `cargo test -p feathertalk-worker --all-targets`

Expected: PASS, exit code 0.

- [ ] **Step 7: Commit**

```bash
git add rust/Cargo.toml rust/crates/feathertalk-worker
git commit -m "feat(worker): add the worker crate with configuration and handshake"
```

---

### Task 5: Mapping library errors onto the ten wire error codes

`ProjectError` has 12 variants and `MediaError` has 22; the wire has 10 codes, a Chinese summary, an English detail, a stage, and a recovery hint. This task is the total function between them, written with exhaustive matches so a new library variant breaks the build instead of silently becoming `WorkerCrashed`.

**Files:**
- Create: `rust/crates/feathertalk-worker/src/error_map.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs`
- Test: `rust/crates/feathertalk-worker/tests/error_mapping.rs`

**Interfaces:**
- Consumes: `ProjectError`, `MediaError`, `TaskError::new(code, summary: &str, detail: &str, stage) -> TaskError`, `ErrorCode`, `MAX_DETAIL_CHARS`.
- Produces: `project_task_error(&ProjectError) -> TaskError`, `media_task_error(&MediaError) -> TaskError`, `is_media_cancellation(&MediaError) -> bool`.

- [ ] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-worker/tests/error_mapping.rs`:

```rust
use std::{io, path::PathBuf};

use feathertalk_domain::{ErrorCode, MAX_DETAIL_CHARS, TaskStage};
use feathertalk_media::MediaError;
use feathertalk_project::ProjectError;
use feathertalk_worker::{is_media_cancellation, media_task_error, project_task_error};

fn io_error(kind: io::ErrorKind) -> io::Error {
    io::Error::new(kind, "synthetic")
}

fn json_error() -> serde_json::Error {
    serde_json::from_str::<serde_json::Value>("{").unwrap_err()
}

fn path() -> PathBuf {
    PathBuf::from("C:/tmp/x")
}

#[test]
fn every_project_error_maps_to_a_code_and_a_valid_payload() {
    let cases = vec![
        (
            ProjectError::Io {
                operation: "read",
                path: path(),
                source: io_error(io::ErrorKind::PermissionDenied),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            ProjectError::Io {
                operation: "write",
                path: path(),
                source: io_error(io::ErrorKind::StorageFull),
            },
            ErrorCode::DiskSpaceLow,
        ),
        (
            ProjectError::Io {
                operation: "write",
                path: path(),
                source: io_error(io::ErrorKind::QuotaExceeded),
            },
            ErrorCode::DiskSpaceLow,
        ),
        (
            ProjectError::ManifestTooLarge {
                path: path(),
                limit: 1024,
            },
            ErrorCode::MediaInvalid,
        ),
        (ProjectError::InvalidUtf8 { path: path() }, ErrorCode::MediaInvalid),
        (
            ProjectError::InvalidJson {
                path: path(),
                source: json_error(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            ProjectError::UnsupportedSchemaVersion {
                path: path(),
                version: 9,
            },
            ErrorCode::MediaInvalid,
        ),
        (
            ProjectError::InvalidField {
                field: "project_id".to_owned(),
                message: "empty".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            ProjectError::UnsafeRelativePath {
                path: "../x".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (ProjectError::Symlink { path: path() }, ErrorCode::MediaInvalid),
        (
            ProjectError::InvalidFilesystemEntry { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (ProjectError::EmptyArtifact { path: path() }, ErrorCode::MediaInvalid),
        (
            ProjectError::LockedAssetMutation { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            ProjectError::AtomicReplacementUnsupported { path: path() },
            ErrorCode::WorkerCrashed,
        ),
    ];

    for (error, expected) in cases {
        let mapped = project_task_error(&error);
        assert_eq!(mapped.code, expected, "{error:?}");
        assert_eq!(mapped.stage, TaskStage::Preparing, "{error:?}");
        assert_eq!(mapped.recovery, expected.default_recovery(), "{error:?}");
        assert!(!mapped.summary.trim().is_empty(), "{error:?}");
        assert!(!mapped.detail.is_empty(), "{error:?}");
        mapped.validate().unwrap();
    }
}

#[test]
fn every_media_error_maps_to_a_code_and_a_valid_payload() {
    let cases = vec![
        (
            MediaError::Io {
                operation: "read",
                path: path(),
                source: io_error(io::ErrorKind::NotFound),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::Io {
                operation: "write",
                path: path(),
                source: io_error(io::ErrorKind::StorageFull),
            },
            ErrorCode::DiskSpaceLow,
        ),
        (MediaError::InputMissing { path: path() }, ErrorCode::MediaInvalid),
        (
            MediaError::InputNotRegularFile { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            MediaError::SymlinkNotAllowed { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            MediaError::OutputDirectoryInvalid { path: path() },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::OutputInsideInput {
                input: path(),
                output: path(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::OutputConflictsWithInput { path: path() },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::OutputDestinationInvalid { path: path() },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::UnsupportedTarget {
                field: "fps",
                expected: "25",
                actual: "30".to_owned(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::InvalidToolchain {
                field: "ffprobe",
                message: "relative".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            MediaError::ProbeTooLarge {
                limit: 16,
                actual: 32,
            },
            ErrorCode::MediaInvalid,
        ),
        (
            MediaError::ProbeJson {
                message: "bad".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            MediaError::ProbeContract {
                field: "width".to_owned(),
                message: "missing".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            MediaError::MissingStream { stream: "video" },
            ErrorCode::MediaInvalid,
        ),
        (
            MediaError::DuplicateStream { stream: "audio" },
            ErrorCode::MediaInvalid,
        ),
        (
            MediaError::ToolFailed {
                operation: "probe",
                exit_code: Some(1),
                stderr: "boom".to_owned(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::ToolTimedOut {
                operation: "probe",
                timeout_ms: 10,
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::ToolOutputTooLarge {
                operation: "probe",
                stream: "stdout",
                limit: 16,
                actual: 32,
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::ToolSpawn {
                operation: "probe",
                message: "not found".to_owned(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::ToolCancelled { operation: "probe" },
            ErrorCode::TaskCancelled,
        ),
        (
            MediaError::NormalizationVerificationFailed {
                field: "fps",
                expected: "25".to_owned(),
                actual: "30".to_owned(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::OutputCommitFailed {
                operation: "commit",
                message: "busy".to_owned(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::OutputRollbackFailed {
                operation: "rollback",
                primary: "a".to_owned(),
                rollback: "b".to_owned(),
            },
            ErrorCode::WorkerCrashed,
        ),
    ];

    for (error, expected) in cases {
        let mapped = media_task_error(&error);
        assert_eq!(mapped.code, expected, "{error:?}");
        assert_eq!(mapped.stage, TaskStage::Preparing, "{error:?}");
        assert_eq!(mapped.recovery, expected.default_recovery(), "{error:?}");
        assert!(!mapped.summary.trim().is_empty(), "{error:?}");
        mapped.validate().unwrap();
    }
}

#[test]
fn an_oversized_detail_is_clamped_to_the_wire_limit() {
    let mapped = media_task_error(&MediaError::ProbeJson {
        message: "x".repeat(MAX_DETAIL_CHARS * 2),
    });
    assert_eq!(mapped.detail.chars().count(), MAX_DETAIL_CHARS);
    mapped.validate().unwrap();
}

#[test]
fn only_tool_cancelled_counts_as_cancellation() {
    assert!(is_media_cancellation(&MediaError::ToolCancelled {
        operation: "probe"
    }));
    assert!(!is_media_cancellation(&MediaError::ToolTimedOut {
        operation: "probe",
        timeout_ms: 10
    }));
    assert!(!is_media_cancellation(&MediaError::InputMissing {
        path: path()
    }));
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p feathertalk-worker --all-targets`

Expected: FAIL to compile with `error[E0432]: unresolved imports `feathertalk_worker::is_media_cancellation`, `feathertalk_worker::media_task_error`, `feathertalk_worker::project_task_error``.

- [ ] **Step 3: Implement the mapping**

Create `rust/crates/feathertalk-worker/src/error_map.rs`:

```rust
use std::io;

use feathertalk_domain::{ErrorCode, MAX_DETAIL_CHARS, TaskError, TaskStage};
use feathertalk_media::MediaError;
use feathertalk_project::ProjectError;

/// Both commands in this slice are single-shot checks that run before any long
/// pipeline stage exists, so every failure is reported as happening while the
/// task was being prepared. `TaskError.stage` must never be terminal.
const FAILURE_STAGE: TaskStage = TaskStage::Preparing;

pub fn project_task_error(error: &ProjectError) -> TaskError {
    let code = project_error_code(error);
    TaskError::new(
        code,
        project_summary(error),
        &clamp(&error.to_string()),
        FAILURE_STAGE,
    )
}

pub fn media_task_error(error: &MediaError) -> TaskError {
    let code = media_error_code(error);
    TaskError::new(
        code,
        media_summary(error),
        &clamp(&error.to_string()),
        FAILURE_STAGE,
    )
}

pub fn is_media_cancellation(error: &MediaError) -> bool {
    matches!(error, MediaError::ToolCancelled { .. })
}

fn project_error_code(error: &ProjectError) -> ErrorCode {
    match error {
        ProjectError::Io { source, .. } => io_error_code(source),
        ProjectError::ManifestTooLarge { .. }
        | ProjectError::InvalidUtf8 { .. }
        | ProjectError::InvalidJson { .. }
        | ProjectError::UnsupportedSchemaVersion { .. }
        | ProjectError::InvalidField { .. }
        | ProjectError::UnsafeRelativePath { .. }
        | ProjectError::Symlink { .. }
        | ProjectError::InvalidFilesystemEntry { .. }
        | ProjectError::EmptyArtifact { .. }
        | ProjectError::LockedAssetMutation { .. } => ErrorCode::MediaInvalid,
        ProjectError::AtomicReplacementUnsupported { .. } => ErrorCode::WorkerCrashed,
    }
}

fn project_summary(error: &ProjectError) -> &'static str {
    match error {
        ProjectError::Io { source, .. } => io_summary(source),
        ProjectError::ManifestTooLarge { .. } => "项目清单过大",
        ProjectError::InvalidUtf8 { .. } => "项目清单不是有效的 UTF-8 文本",
        ProjectError::InvalidJson { .. } => "项目清单 JSON 格式错误",
        ProjectError::UnsupportedSchemaVersion { .. } => "项目清单版本不受支持",
        ProjectError::InvalidField { .. } => "项目清单字段无效",
        ProjectError::UnsafeRelativePath { .. } => "项目清单包含不安全的相对路径",
        ProjectError::Symlink { .. } => "项目目录包含符号链接",
        ProjectError::InvalidFilesystemEntry { .. } => "项目目录结构不符合要求",
        ProjectError::EmptyArtifact { .. } => "项目素材文件为空",
        ProjectError::LockedAssetMutation { .. } => "素材包已锁定，无法修改",
        ProjectError::AtomicReplacementUnsupported { .. } => "当前文件系统不支持原子替换",
    }
}

fn media_error_code(error: &MediaError) -> ErrorCode {
    match error {
        MediaError::Io { source, .. } => io_error_code(source),
        MediaError::InputMissing { .. }
        | MediaError::InputNotRegularFile { .. }
        | MediaError::SymlinkNotAllowed { .. }
        | MediaError::InvalidToolchain { .. }
        | MediaError::ProbeTooLarge { .. }
        | MediaError::ProbeJson { .. }
        | MediaError::ProbeContract { .. }
        | MediaError::MissingStream { .. }
        | MediaError::DuplicateStream { .. } => ErrorCode::MediaInvalid,
        // The runtime intercepts cancellation before it reaches this mapper.
        // The arm exists so the mapping stays total and so a cancellation that
        // somehow arrives here is not mislabelled as a crash.
        MediaError::ToolCancelled { .. } => ErrorCode::TaskCancelled,
        MediaError::OutputDirectoryInvalid { .. }
        | MediaError::OutputInsideInput { .. }
        | MediaError::OutputConflictsWithInput { .. }
        | MediaError::OutputDestinationInvalid { .. }
        | MediaError::UnsupportedTarget { .. }
        | MediaError::ToolFailed { .. }
        | MediaError::ToolTimedOut { .. }
        | MediaError::ToolOutputTooLarge { .. }
        | MediaError::ToolSpawn { .. }
        | MediaError::NormalizationVerificationFailed { .. }
        | MediaError::OutputCommitFailed { .. }
        | MediaError::OutputRollbackFailed { .. } => ErrorCode::WorkerCrashed,
    }
}

fn media_summary(error: &MediaError) -> &'static str {
    match error {
        MediaError::Io { source, .. } => io_summary(source),
        MediaError::InputMissing { .. } => "找不到输入文件",
        MediaError::InputNotRegularFile { .. } => "输入不是常规文件",
        MediaError::SymlinkNotAllowed { .. } => "输入路径包含符号链接",
        MediaError::InvalidToolchain { .. } => "媒体工具链配置无效",
        MediaError::ProbeTooLarge { .. } => "媒体探测输出过大",
        MediaError::ProbeJson { .. } => "媒体探测输出不是有效 JSON",
        MediaError::ProbeContract { .. } => "媒体探测结果缺少必需字段",
        MediaError::MissingStream { .. } => "媒体文件缺少必需的音视频流",
        MediaError::DuplicateStream { .. } => "媒体文件包含重复的音视频流",
        MediaError::ToolCancelled { .. } => "任务已取消",
        MediaError::OutputDirectoryInvalid { .. }
        | MediaError::OutputInsideInput { .. }
        | MediaError::OutputConflictsWithInput { .. }
        | MediaError::OutputDestinationInvalid { .. } => "输出路径无效",
        MediaError::UnsupportedTarget { .. } => "不支持的媒体转换目标",
        MediaError::ToolFailed { .. } => "媒体工具执行失败",
        MediaError::ToolTimedOut { .. } => "媒体工具执行超时",
        MediaError::ToolOutputTooLarge { .. } => "媒体工具输出过大",
        MediaError::ToolSpawn { .. } => "无法启动媒体工具",
        MediaError::NormalizationVerificationFailed { .. } => "媒体规范化结果校验失败",
        MediaError::OutputCommitFailed { .. } => "写入输出文件失败",
        MediaError::OutputRollbackFailed { .. } => "写入失败后回滚也失败",
    }
}

fn io_error_code(source: &io::Error) -> ErrorCode {
    match source.kind() {
        io::ErrorKind::StorageFull | io::ErrorKind::QuotaExceeded => ErrorCode::DiskSpaceLow,
        _ => ErrorCode::WorkerCrashed,
    }
}

fn io_summary(source: &io::Error) -> &'static str {
    match source.kind() {
        io::ErrorKind::StorageFull | io::ErrorKind::QuotaExceeded => "磁盘空间不足",
        _ => "文件读写失败",
    }
}

/// `TaskError::validate` counts characters, not bytes, so the detail is clamped
/// on a character boundary.
fn clamp(detail: &str) -> String {
    detail.chars().take(MAX_DETAIL_CHARS).collect()
}
```

`io::ErrorKind::StorageFull` and `io::ErrorKind::QuotaExceeded` are both stable on the toolchain in use (rustc 1.95.0). `ErrorKind` is `#[non_exhaustive]`, so the `_` arm there is required; the `ProjectError` and `MediaError` matches must stay exhaustive with no `_` arm, which is what turns a future library variant into a compile error instead of a silent mislabel.

- [ ] **Step 4: Export the mapping**

In `rust/crates/feathertalk-worker/src/lib.rs`, add `mod error_map;` after `mod config;` and the re-export after the `config` block:

```rust
pub use error_map::{is_media_cancellation, media_task_error, project_task_error};
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cargo test -p feathertalk-worker --all-targets`

Expected: PASS, exit code 0.

- [ ] **Step 6: Commit**

```bash
git add rust/crates/feathertalk-worker
git commit -m "feat(worker): map project and media errors onto wire error codes"
```

---

### Task 6: Command execution and the probe result payload

This is the layer that actually runs work: one function that takes a `Request` and returns completed, cancelled, or failed. It is deliberately synchronous and runner-injectable so the runtime tests in Task 8 never need a real `ffprobe`.

**Files:**
- Create: `rust/crates/feathertalk-worker/src/probe_result.rs`
- Create: `rust/crates/feathertalk-worker/src/commands.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs`
- Test: `rust/crates/feathertalk-worker/tests/commands.rs`

**Interfaces:**
- Consumes: `Request`, `ProbeMediaParams.input`, `ProjectDirParams.project_dir`, `validate_project_dir`, `validate_input`, `probe_media_with_runner`, `MediaProbe` accessors, `CancellationToken`, `CancellableProcessRunner`, `project_task_error`, `media_task_error`, `is_media_cancellation`.
- Produces: `probe_to_json(&MediaProbe) -> serde_json::Value`; `CommandOutcome::{Completed(Option<serde_json::Value>), Cancelled, Failed(TaskError)}`; `execute(&Request, Option<&MediaToolchain>, &CancellationToken) -> CommandOutcome`; `execute_with_runner<R: ProcessRunner + ?Sized>(&Request, Option<&MediaToolchain>, &CancellationToken, &R) -> CommandOutcome`.

- [ ] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-worker/tests/commands.rs`:

```rust
use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::Mutex,
    time::Duration,
};

use feathertalk_domain::{
    ErrorCode, ProbeMediaParams, ProjectDirParams, Request, TrainParams, TrainingMode,
    UnetVariant,
};
use feathertalk_media::{
    CancellationToken, CommandSpec, MediaError, MediaToolchain, ProcessOutput, ProcessRunner,
};
use feathertalk_project::{
    AssetManifest, AssetPackageState, FeatureType, ModelSelection, ProjectManifest,
    TaskHistoryEntry, TaskHistoryStatus, lock_asset_package, write_project_manifest_atomic,
};
use feathertalk_worker::{CommandOutcome, execute_with_runner};

struct FakeRunner {
    outputs: Mutex<VecDeque<Result<ProcessOutput, MediaError>>>,
    commands: Mutex<Vec<CommandSpec>>,
}

impl FakeRunner {
    fn new(outputs: Vec<Result<ProcessOutput, MediaError>>) -> Self {
        Self {
            outputs: Mutex::new(outputs.into_iter().collect()),
            commands: Mutex::new(Vec::new()),
        }
    }

    fn call_count(&self) -> usize {
        self.commands.lock().unwrap().len()
    }
}

impl ProcessRunner for FakeRunner {
    fn run(&self, command: &CommandSpec, _timeout: Duration) -> Result<ProcessOutput, MediaError> {
        self.commands.lock().unwrap().push(command.clone());
        self.outputs.lock().unwrap().pop_front().unwrap()
    }
}

fn toolchain() -> MediaToolchain {
    let root = std::env::current_dir().unwrap();
    MediaToolchain::new(
        root.join("ffmpeg-test"),
        root.join("ffprobe-test"),
        Duration::from_secs(10),
    )
    .unwrap()
}

fn valid_probe() -> Vec<u8> {
    br#"{
      "format":{"format_name":"mov,mp4","duration":"2.0"},
      "streams":[
        {"codec_type":"video","codec_name":"h264","pix_fmt":"yuv420p","width":640,"height":480,"avg_frame_rate":"25/1","nb_read_frames":"50","duration":"2.0"},
        {"codec_type":"audio","codec_name":"aac","sample_fmt":"fltp","sample_rate":"48000","channels":2,"duration":"2.0"}
      ]
    }"#
    .to_vec()
}

fn probe_request(input: PathBuf) -> Request {
    Request::ProbeMedia(ProbeMediaParams { input })
}

fn media_file() -> (tempfile::TempDir, PathBuf) {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("input.mov");
    std::fs::write(&source, b"media").unwrap();
    (temp, source)
}

fn valid_project() -> ProjectManifest {
    ProjectManifest {
        schema_version: 1,
        project_id: "demo".to_owned(),
        display_name: "Demo".to_owned(),
        asset_package: "assets/assets.json".to_owned(),
        default_model: ModelSelection::OriginalUnet,
        task_history: vec![TaskHistoryEntry {
            task_id: "task-1".to_owned(),
            kind: "preprocess".to_owned(),
            status: TaskHistoryStatus::Completed,
            updated_at: "2026-08-20T10:00:00Z".to_owned(),
        }],
    }
}

fn locked_manifest() -> AssetManifest {
    AssetManifest {
        schema_version: 1,
        state: AssetPackageState::Locked,
        video_fps: 25,
        audio_sample_rate: 16_000,
        audio_channels: 1,
        frame_count: 12,
        frame_width: 160,
        frame_height: 160,
        feature_type: FeatureType::FeatherHubert,
        feature_shape: [12, 2, 1024],
        landmark_model_sha256: "a".repeat(64),
        feature_model_sha256: "b".repeat(64),
    }
}

fn complete_project() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("assets/frames")).unwrap();
    std::fs::create_dir_all(dir.path().join("assets/landmarks")).unwrap();
    std::fs::create_dir_all(dir.path().join("assets/features")).unwrap();
    for file in [
        "assets/video_25fps.mp4",
        "assets/audio_16k_mono.wav",
        "assets/features/feather_hubert.f32",
    ] {
        std::fs::write(dir.path().join(file), b"x").unwrap();
    }
    write_project_manifest_atomic(&dir.path().join("project.json"), &valid_project()).unwrap();
    lock_asset_package(dir.path(), locked_manifest()).unwrap();
    dir
}

#[test]
fn validating_a_complete_project_completes_without_a_result() {
    let dir = complete_project();
    let request = Request::ValidateProject(ProjectDirParams {
        project_dir: dir.path().to_path_buf(),
    });
    let runner = FakeRunner::new(vec![]);
    let outcome = execute_with_runner(&request, None, &CancellationToken::new(), &runner);
    assert!(matches!(outcome, CommandOutcome::Completed(None)), "{outcome:?}");
    assert_eq!(runner.call_count(), 0);
}

#[test]
fn validating_a_missing_project_fails_with_a_wire_error() {
    let dir = tempfile::tempdir().unwrap();
    let request = Request::ValidateProject(ProjectDirParams {
        project_dir: dir.path().join("nope"),
    });
    let runner = FakeRunner::new(vec![]);
    let CommandOutcome::Failed(error) =
        execute_with_runner(&request, None, &CancellationToken::new(), &runner)
    else {
        panic!("a missing project must fail");
    };
    error.validate().unwrap();
    assert!(!error.summary.is_empty());
}

#[test]
fn probing_media_completes_with_the_probe_result() {
    let (_temp, source) = media_file();
    let runner = FakeRunner::new(vec![Ok(ProcessOutput::new(
        Some(0),
        valid_probe(),
        Vec::new(),
    ))]);
    let toolchain = toolchain();
    let CommandOutcome::Completed(Some(result)) = execute_with_runner(
        &probe_request(source),
        Some(&toolchain),
        &CancellationToken::new(),
        &runner,
    ) else {
        panic!("a successful probe must carry a result");
    };
    assert!(result.is_object());
    assert_eq!(result["format"]["format_name"], "mov,mp4");
    assert_eq!(result["format"]["duration_seconds"], 2.0);
    assert_eq!(result["video"]["codec_name"], "h264");
    assert_eq!(result["video"]["pixel_format"], "yuv420p");
    assert_eq!(result["video"]["width"], 640);
    assert_eq!(result["video"]["height"], 480);
    assert_eq!(result["video"]["frame_rate"]["numerator"], 25);
    assert_eq!(result["video"]["frame_rate"]["denominator"], 1);
    assert_eq!(result["video"]["frame_count"], 50);
    assert_eq!(result["audio"]["codec_name"], "aac");
    assert_eq!(result["audio"]["sample_format"], "fltp");
    assert_eq!(result["audio"]["sample_rate"], 48_000);
    assert_eq!(result["audio"]["channels"], 2);
    assert_eq!(runner.call_count(), 1);
    assert_eq!(result.get("input"), None, "the result must not leak paths");
}

#[test]
fn probing_a_missing_file_fails_before_the_tool_runs() {
    let temp = tempfile::tempdir().unwrap();
    let runner = FakeRunner::new(vec![]);
    let toolchain = toolchain();
    let CommandOutcome::Failed(error) = execute_with_runner(
        &probe_request(temp.path().join("absent.mov")),
        Some(&toolchain),
        &CancellationToken::new(),
        &runner,
    ) else {
        panic!("a missing input must fail");
    };
    assert_eq!(error.code, ErrorCode::MediaInvalid);
    error.validate().unwrap();
    assert_eq!(runner.call_count(), 0);
}

#[test]
fn a_cancelled_tool_reports_cancellation_not_failure() {
    let (_temp, source) = media_file();
    let runner = FakeRunner::new(vec![Err(MediaError::ToolCancelled {
        operation: "probe",
    })]);
    let toolchain = toolchain();
    let outcome = execute_with_runner(
        &probe_request(source),
        Some(&toolchain),
        &CancellationToken::new(),
        &runner,
    );
    assert!(matches!(outcome, CommandOutcome::Cancelled), "{outcome:?}");
}

#[test]
fn an_already_cancelled_token_runs_nothing() {
    let (_temp, source) = media_file();
    let runner = FakeRunner::new(vec![]);
    let toolchain = toolchain();
    let token = CancellationToken::new();
    token.cancel();
    let outcome = execute_with_runner(&probe_request(source), Some(&toolchain), &token, &runner);
    assert!(matches!(outcome, CommandOutcome::Cancelled), "{outcome:?}");
    assert_eq!(runner.call_count(), 0);
}

#[test]
fn probing_without_a_toolchain_is_refused() {
    let (_temp, source) = media_file();
    let runner = FakeRunner::new(vec![]);
    let CommandOutcome::Failed(error) =
        execute_with_runner(&probe_request(source), None, &CancellationToken::new(), &runner)
    else {
        panic!("probing without a toolchain must fail");
    };
    assert_eq!(error.code, ErrorCode::WorkerCrashed);
    error.validate().unwrap();
}

#[test]
fn an_unsupported_command_is_refused_with_its_slug() {
    let request = Request::Train(TrainParams {
        project_dir: PathBuf::from("C:/tmp/project"),
        mode: TrainingMode::Full,
        variant: UnetVariant::Original,
        adapter_id: "cpu-0".to_owned(),
        epochs: 1,
        batch_size: 1,
        learning_rate: 0.0001,
        resume_from_checkpoint: None,
    });
    let runner = FakeRunner::new(vec![]);
    let CommandOutcome::Failed(error) =
        execute_with_runner(&request, None, &CancellationToken::new(), &runner)
    else {
        panic!("an unsupported command must fail");
    };
    assert_eq!(error.code, ErrorCode::WorkerCrashed);
    assert!(error.detail.contains("train"), "{}", error.detail);
    error.validate().unwrap();
}
```

Read `rust/crates/feathertalk-domain/src/request.rs` before writing the `TrainParams` literal and use its exact fields; the runtime never accepts this command, the literal exists only to prove the defensive arm rejects it. If a field name or enum variant differs, fix the literal, not the params struct.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p feathertalk-worker --all-targets`

Expected: FAIL to compile with `error[E0432]: unresolved imports `feathertalk_worker::CommandOutcome`, `feathertalk_worker::execute_with_runner``.

- [ ] **Step 3: Implement the probe result payload**

Create `rust/crates/feathertalk-worker/src/probe_result.rs`:

```rust
use feathertalk_media::MediaProbe;
use serde_json::{Value, json};

/// Shapes a probe as the JSON object a `completed` event carries.
///
/// The input path is deliberately absent: the desktop already knows which file
/// it asked about, and the event stream is written to logs.
pub fn probe_to_json(probe: &MediaProbe) -> Value {
    json!({
        "format": {
            "format_name": probe.format().format_name(),
            "duration_seconds": probe.format().duration_seconds(),
        },
        "video": probe.video().map(|video| {
            json!({
                "codec_name": video.codec_name(),
                "pixel_format": video.pixel_format(),
                "width": video.width(),
                "height": video.height(),
                "frame_rate": {
                    "numerator": video.frame_rate().numerator(),
                    "denominator": video.frame_rate().denominator(),
                },
                "frame_count": video.frame_count(),
                "duration_seconds": video.duration_seconds(),
            })
        }),
        "audio": probe.audio().map(|audio| {
            json!({
                "codec_name": audio.codec_name(),
                "sample_format": audio.sample_format(),
                "sample_rate": audio.sample_rate(),
                "channels": audio.channels(),
                "sample_count": audio.sample_count(),
                "duration_seconds": audio.duration_seconds(),
            })
        }),
    })
}
```

`Option<Value>` serializes as `null`, so an audio-only or video-only file still produces the same three top-level keys. Frame rate stays an exact rational — `25/1`, not `25.0` — because a later slice checks it against the 25 fps requirement and a float would make that check approximate.

- [ ] **Step 4: Implement command execution**

Create `rust/crates/feathertalk-worker/src/commands.rs`:

```rust
use feathertalk_domain::{ErrorCode, Request, TaskError, TaskKind, TaskStage};
use feathertalk_media::{
    CancellableProcessRunner, CancellationToken, MediaError, MediaInput, MediaToolchain,
    ProcessRunner, probe_media_with_runner, validate_input,
};
use feathertalk_project::validate_project_dir;

use crate::{is_media_cancellation, media_task_error, probe_to_json, project_task_error};

#[derive(Debug)]
pub enum CommandOutcome {
    /// The command finished. `Some` carries the JSON object a `completed` event
    /// reports; `None` means the command has no result payload.
    Completed(Option<serde_json::Value>),
    Cancelled,
    Failed(TaskError),
}

pub fn execute(
    request: &Request,
    media: Option<&MediaToolchain>,
    token: &CancellationToken,
) -> CommandOutcome {
    let runner = CancellableProcessRunner::new(token.clone());
    execute_with_runner(request, media, token, &runner)
}

pub fn execute_with_runner<R: ProcessRunner + ?Sized>(
    request: &Request,
    media: Option<&MediaToolchain>,
    token: &CancellationToken,
    runner: &R,
) -> CommandOutcome {
    if token.is_cancelled() {
        return CommandOutcome::Cancelled;
    }
    match request {
        Request::ValidateProject(params) => match validate_project_dir(&params.project_dir) {
            // Project validation is filesystem-bound and has no interrupt hook,
            // so cancellation is honoured at this boundary: the work is thrown
            // away rather than reported as a completed task.
            Ok(_) if token.is_cancelled() => CommandOutcome::Cancelled,
            Ok(_) => CommandOutcome::Completed(None),
            Err(error) => CommandOutcome::Failed(project_task_error(&error)),
        },
        Request::ProbeMedia(params) => {
            let Some(toolchain) = media else {
                // Unreachable through the runtime, which rejects `probe_media`
                // when no toolchain is configured. Kept so a direct caller
                // cannot get a panic instead of an error.
                return CommandOutcome::Failed(unsupported(request.kind()));
            };
            let input = match validate_input(&MediaInput {
                source: params.input.clone(),
            }) {
                Ok(input) => input,
                Err(error) => return media_failure(&error),
            };
            match probe_media_with_runner(&input, toolchain, runner) {
                Ok(probe) => CommandOutcome::Completed(Some(probe_to_json(&probe))),
                Err(error) => media_failure(&error),
            }
        }
        other => CommandOutcome::Failed(unsupported(other.kind())),
    }
}

fn media_failure(error: &MediaError) -> CommandOutcome {
    if is_media_cancellation(error) {
        CommandOutcome::Cancelled
    } else {
        CommandOutcome::Failed(media_task_error(error))
    }
}

fn unsupported(kind: TaskKind) -> TaskError {
    TaskError::new(
        ErrorCode::WorkerCrashed,
        "当前 worker 不支持该命令",
        &format!(
            "command {} is not supported by this worker build",
            kind.as_slug()
        ),
        TaskStage::Preparing,
    )
}
```

- [ ] **Step 5: Export the new modules**

In `rust/crates/feathertalk-worker/src/lib.rs`, add `mod commands;` and `mod probe_result;` in alphabetical position among the module declarations, and the re-exports:

```rust
pub use commands::{CommandOutcome, execute, execute_with_runner};
pub use probe_result::probe_to_json;
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `cargo test -p feathertalk-worker --all-targets`

Expected: PASS, exit code 0.

- [ ] **Step 7: Commit**

```bash
git add rust/crates/feathertalk-worker
git commit -m "feat(worker): execute validate_project and probe_media"
```

---

### Task 7: Adapter lock table

One task per adapter is the rule that keeps a later GPU slice from running two trainings on one card. The table is built and tested now, with synthetic adapter ids, so the rule is enforced by a tested component rather than by whatever the runtime happens to do.

**Files:**
- Create: `rust/crates/feathertalk-worker/src/adapters.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs`
- Test: `rust/crates/feathertalk-worker/tests/adapter_locks.rs`

**Interfaces:**
- Consumes: `TaskId` (`Clone + Eq + Ord + Hash`, not `Copy`).
- Produces: `AdapterLocks::new(impl IntoIterator<Item = String>) -> AdapterLocks`, `acquire(&mut self, &str, TaskId) -> Result<(), AdapterLockError>`, `release(&mut self, &str) -> Result<(), AdapterLockError>`, `holder(&self, &str) -> Option<&TaskId>`, `is_free(&self, &str) -> bool`; `AdapterLockError::{Unknown, Occupied, NotHeld}`.

- [ ] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-worker/tests/adapter_locks.rs`:

```rust
use feathertalk_domain::TaskId;
use feathertalk_worker::{AdapterLockError, AdapterLocks, CPU_ADAPTER_ID};

fn task(suffix: &str) -> TaskId {
    TaskId::parse(&format!("1787900000000-{suffix}")).unwrap()
}

fn cpu_locks() -> AdapterLocks {
    AdapterLocks::new([CPU_ADAPTER_ID.to_owned()])
}

#[test]
fn a_fresh_table_reports_every_known_adapter_as_free() {
    let locks = cpu_locks();
    assert!(locks.is_free(CPU_ADAPTER_ID));
    assert_eq!(locks.holder(CPU_ADAPTER_ID), None);
}

#[test]
fn an_unknown_adapter_cannot_be_locked_or_released() {
    let mut locks = cpu_locks();
    assert!(matches!(
        locks.acquire("gpu-9", task("0000000a")),
        Err(AdapterLockError::Unknown(_))
    ));
    assert!(matches!(
        locks.release("gpu-9"),
        Err(AdapterLockError::Unknown(_))
    ));
    assert!(!locks.is_free("gpu-9"));
}

#[test]
fn a_locked_adapter_refuses_a_second_task_and_names_the_holder() {
    let mut locks = cpu_locks();
    let first = task("0000000a");
    locks.acquire(CPU_ADAPTER_ID, first.clone()).unwrap();
    assert!(!locks.is_free(CPU_ADAPTER_ID));
    assert_eq!(locks.holder(CPU_ADAPTER_ID), Some(&first));

    match locks.acquire(CPU_ADAPTER_ID, task("0000000b")) {
        Err(AdapterLockError::Occupied { adapter_id, holder }) => {
            assert_eq!(adapter_id, CPU_ADAPTER_ID);
            assert_eq!(holder, first);
        }
        other => panic!("expected an occupied adapter, got {other:?}"),
    }
}

#[test]
fn releasing_frees_the_adapter_for_the_next_task() {
    let mut locks = cpu_locks();
    locks.acquire(CPU_ADAPTER_ID, task("0000000a")).unwrap();
    locks.release(CPU_ADAPTER_ID).unwrap();
    assert!(locks.is_free(CPU_ADAPTER_ID));
    locks.acquire(CPU_ADAPTER_ID, task("0000000b")).unwrap();
    assert_eq!(locks.holder(CPU_ADAPTER_ID), Some(&task("0000000b")));
}

#[test]
fn releasing_a_free_adapter_is_an_error() {
    let mut locks = cpu_locks();
    assert!(matches!(
        locks.release(CPU_ADAPTER_ID),
        Err(AdapterLockError::NotHeld(_))
    ));
}

#[test]
fn adapters_are_locked_independently() {
    let mut locks = AdapterLocks::new(["gpu-a".to_owned(), "gpu-b".to_owned()]);
    let first = task("0000000a");
    let second = task("0000000b");
    locks.acquire("gpu-a", first.clone()).unwrap();
    locks.acquire("gpu-b", second.clone()).unwrap();
    assert_eq!(locks.holder("gpu-a"), Some(&first));
    assert_eq!(locks.holder("gpu-b"), Some(&second));
    locks.release("gpu-a").unwrap();
    assert!(locks.is_free("gpu-a"));
    assert_eq!(locks.holder("gpu-b"), Some(&second));
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p feathertalk-worker --all-targets`

Expected: FAIL to compile with `error[E0432]: unresolved imports `feathertalk_worker::AdapterLockError`, `feathertalk_worker::AdapterLocks``.

- [ ] **Step 3: Implement the table**

Create `rust/crates/feathertalk-worker/src/adapters.rs`:

```rust
use std::collections::{BTreeMap, BTreeSet};

use feathertalk_domain::TaskId;

#[derive(Debug, thiserror::Error)]
pub enum AdapterLockError {
    #[error("unknown adapter {0}")]
    Unknown(String),
    #[error("adapter {adapter_id} is already running task {}", holder.as_str())]
    Occupied { adapter_id: String, holder: TaskId },
    #[error("adapter {0} is not locked")]
    NotHeld(String),
}

/// Enforces "at most one task per adapter".
///
/// This slice registers only the CPU adapter, but the table is keyed by adapter
/// id so the GPU slice adds cards without changing the rule or its tests.
#[derive(Debug)]
pub struct AdapterLocks {
    known: BTreeSet<String>,
    occupied: BTreeMap<String, TaskId>,
}

impl AdapterLocks {
    pub fn new(adapter_ids: impl IntoIterator<Item = String>) -> Self {
        Self {
            known: adapter_ids.into_iter().collect(),
            occupied: BTreeMap::new(),
        }
    }

    pub fn acquire(&mut self, adapter_id: &str, task_id: TaskId) -> Result<(), AdapterLockError> {
        self.check_known(adapter_id)?;
        if let Some(holder) = self.occupied.get(adapter_id) {
            return Err(AdapterLockError::Occupied {
                adapter_id: adapter_id.to_owned(),
                holder: holder.clone(),
            });
        }
        self.occupied.insert(adapter_id.to_owned(), task_id);
        Ok(())
    }

    pub fn release(&mut self, adapter_id: &str) -> Result<(), AdapterLockError> {
        self.check_known(adapter_id)?;
        if self.occupied.remove(adapter_id).is_none() {
            return Err(AdapterLockError::NotHeld(adapter_id.to_owned()));
        }
        Ok(())
    }

    pub fn holder(&self, adapter_id: &str) -> Option<&TaskId> {
        self.occupied.get(adapter_id)
    }

    pub fn is_free(&self, adapter_id: &str) -> bool {
        self.known.contains(adapter_id) && !self.occupied.contains_key(adapter_id)
    }

    fn check_known(&self, adapter_id: &str) -> Result<(), AdapterLockError> {
        if self.known.contains(adapter_id) {
            Ok(())
        } else {
            Err(AdapterLockError::Unknown(adapter_id.to_owned()))
        }
    }
}
```

`TaskId` is not `Copy`, so `holder` is cloned into the error and `acquire` takes the id by value.

- [ ] **Step 4: Export the table**

In `rust/crates/feathertalk-worker/src/lib.rs`, add `mod adapters;` as the first module declaration and the re-export:

```rust
pub use adapters::{AdapterLockError, AdapterLocks};
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cargo test -p feathertalk-worker --all-targets`

Expected: PASS, exit code 0.

- [ ] **Step 6: Commit**

```bash
git add rust/crates/feathertalk-worker
git commit -m "feat(worker): enforce one task per adapter"
```

---

### Task 8: The runtime, the queue, cancellation, shutdown, and the binary

Everything above is a pure function or a plain data structure. This task adds the only stateful component: a control loop that owns all task state and the frame writer, an input thread that owns the reader, and one execution thread that runs commands. No mutex is shared between them; every hand-off is an `mpsc` message.

Command execution is reached through one injectable seam, `JobExecutor`. Production code passes `execute`; the tests pass closures. Without that seam a runtime test would need a real `ffprobe` and a real long-running process to observe queueing, cancellation, and shutdown, and those tests would be timing-dependent.

**Files:**
- Create: `rust/crates/feathertalk-worker/src/runtime.rs`
- Create: `rust/crates/feathertalk-worker/src/main.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs`
- Test: `rust/crates/feathertalk-worker/tests/runtime.rs`
- Test: `rust/crates/feathertalk-worker/tests/process_boundary.rs`

**Interfaces:**
- Consumes: `FrameReader::read_frame::<ClientFrame>()`, `FrameWriter::{write_frame, into_inner}`, `ClientFrame::validate` (which already includes the protocol-version check, so a version mismatch surfaces as `DomainError::ProtocolVersion` at decode time), `ServerFrame::validate`, `Event::new`, `TaskLifecycle::advance`, `TaskStage`, `TaskId::as_str`, `CancellationToken`, `MediaToolchain` (`Clone`), `WorkerConfig`, `ready_frame`, `supported_commands`, `AdapterLocks`, `CPU_ADAPTER_ID`, `execute`, `CommandOutcome`, `ENV_FFPROBE`, `ENV_FFMPEG`.
- Produces: `JobExecutor = Box<dyn Fn(&Request, Option<&MediaToolchain>, &CancellationToken) -> CommandOutcome + Send + 'static>`; `serve<R: BufRead + Send + 'static, W: Write>(input: R, output: W, config: &WorkerConfig) -> Result<(), DomainError>`; `serve_with_executor<R: BufRead + Send + 'static, W: Write>(input: R, output: W, config: &WorkerConfig, executor: JobExecutor) -> Result<(), DomainError>`; the `feathertalk-worker` binary.

- [ ] **Step 1: Write the failing runtime test**

Create `rust/crates/feathertalk-worker/tests/runtime.rs`:

```rust
use std::{
    fs,
    io::{self, BufRead, Read, Write},
    path::PathBuf,
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver, Sender},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use feathertalk_domain::{
    CancelFrame, ClientFrame, DomainError, ErrorCode, Event, PROTOCOL_VERSION, ProbeMediaParams,
    ProjectDirParams, Request, ServerFrame, ShutdownFrame, StartFrame, TaskId, TaskKind, TaskStage,
    TrainParams, TrainingMode, UnetVariant, decode_line, encode_line,
};
use feathertalk_media::{
    CancellationToken, CommandSpec, MediaError, MediaToolchain, ProcessOutput, ProcessRunner,
};
use feathertalk_worker::{
    CPU_ADAPTER_ID, CommandOutcome, JobExecutor, WorkerConfig, execute_with_runner,
    serve_with_executor,
};

/// An output sink the test can read while the worker is still writing to it.
#[derive(Clone, Default)]
struct SharedSink(Arc<Mutex<Vec<u8>>>);

impl Write for SharedSink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl SharedSink {
    /// Decode every frame that has already been fully written.
    ///
    /// A trailing fragment without its `\n` is skipped rather than decoded, so
    /// polling never races a half-written line. Every decoded frame is
    /// validated here, which is what asserts the runtime never emits a frame
    /// that fails `ServerFrame::validate`.
    fn frames(&self) -> Vec<ServerFrame> {
        let bytes = self.0.lock().unwrap().clone();
        let text = String::from_utf8(bytes).unwrap();
        let complete = match text.rfind('\n') {
            Some(index) => &text[..=index],
            None => "",
        };
        complete
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                let frame: ServerFrame = decode_line(line).unwrap();
                frame.validate().unwrap();
                frame
            })
            .collect()
    }
}

/// A `BufRead` whose bytes arrive from the test thread. Dropping the sender is
/// end-of-stream, which is how the tests model a closed stdin.
struct ChannelReader {
    receiver: Receiver<Vec<u8>>,
    buffer: Vec<u8>,
    cursor: usize,
    closed: bool,
}

impl ChannelReader {
    fn new(receiver: Receiver<Vec<u8>>) -> Self {
        Self {
            receiver,
            buffer: Vec::new(),
            cursor: 0,
            closed: false,
        }
    }
}

impl BufRead for ChannelReader {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        while self.cursor >= self.buffer.len() {
            if self.closed {
                return Ok(&[]);
            }
            match self.receiver.recv() {
                Ok(chunk) => {
                    self.buffer = chunk;
                    self.cursor = 0;
                }
                Err(_) => {
                    self.closed = true;
                    return Ok(&[]);
                }
            }
        }
        Ok(&self.buffer[self.cursor..])
    }

    fn consume(&mut self, amount: usize) {
        self.cursor += amount;
    }
}

impl Read for ChannelReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let available = self.fill_buf()?;
        let count = available.len().min(out.len());
        out[..count].copy_from_slice(&available[..count]);
        self.consume(count);
        Ok(count)
    }
}

struct Harness {
    input: Option<Sender<Vec<u8>>>,
    sink: SharedSink,
    worker: Option<JoinHandle<Result<(), DomainError>>>,
}

impl Harness {
    fn start(config: WorkerConfig, executor: JobExecutor) -> Self {
        let (input, receiver) = mpsc::channel::<Vec<u8>>();
        let sink = SharedSink::default();
        let worker_sink = sink.clone();
        let worker = thread::spawn(move || {
            serve_with_executor(
                ChannelReader::new(receiver),
                worker_sink,
                &config,
                executor,
            )
        });
        Self {
            input: Some(input),
            sink,
            worker: Some(worker),
        }
    }

    fn send(&self, frame: &ClientFrame) {
        let mut line = encode_line(frame).unwrap().into_bytes();
        line.push(b'\n');
        self.input.as_ref().unwrap().send(line).unwrap();
    }

    fn send_raw(&self, line: &str) {
        self.input
            .as_ref()
            .unwrap()
            .send(format!("{line}\n").into_bytes())
            .unwrap();
    }

    fn frames(&self) -> Vec<ServerFrame> {
        self.sink.frames()
    }

    fn wait_for(
        &self,
        description: &str,
        predicate: impl Fn(&[ServerFrame]) -> bool,
    ) -> Vec<ServerFrame> {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let frames = self.frames();
            if predicate(&frames) {
                return frames;
            }
            assert!(
                Instant::now() < deadline,
                "timed out waiting for {description}; frames so far: {frames:?}"
            );
            thread::sleep(Duration::from_millis(10));
        }
    }

    /// Close the input stream, wait for the worker to exit, and return every
    /// frame it wrote.
    fn finish(mut self) -> Vec<ServerFrame> {
        self.input = None;
        self.worker.take().unwrap().join().unwrap().unwrap();
        self.frames()
    }
}

fn task(suffix: &str) -> TaskId {
    TaskId::parse(&format!("1787900000000-{suffix}")).unwrap()
}

fn start(task_id: &TaskId, request: Request) -> ClientFrame {
    ClientFrame::Start(StartFrame {
        protocol_version: PROTOCOL_VERSION,
        task_id: task_id.clone(),
        request,
    })
}

fn cancel(task_id: &TaskId) -> ClientFrame {
    ClientFrame::Cancel(CancelFrame {
        protocol_version: PROTOCOL_VERSION,
        task_id: task_id.clone(),
    })
}

fn shutdown() -> ClientFrame {
    ClientFrame::Shutdown(ShutdownFrame {
        protocol_version: PROTOCOL_VERSION,
    })
}

fn validate_project(dir: &str) -> Request {
    Request::ValidateProject(ProjectDirParams {
        project_dir: PathBuf::from(dir),
    })
}

fn train_request() -> Request {
    Request::Train(TrainParams {
        project_dir: PathBuf::from("C:/tmp/project"),
        mode: TrainingMode::Full,
        variant: UnetVariant::Original,
        adapter_id: CPU_ADAPTER_ID.to_owned(),
        epochs: 1,
        batch_size: 1,
        learning_rate: 0.0001,
        resume_from_checkpoint: None,
    })
}

fn absolute(name: &str) -> String {
    std::env::current_dir()
        .unwrap()
        .join(name)
        .display()
        .to_string()
}

/// A configuration whose media toolchain is accepted, so `probe_media` is
/// supported. The paths never have to exist: `MediaToolchain::new` only
/// requires absolute paths.
fn media_config() -> WorkerConfig {
    WorkerConfig::from_values(
        Some(absolute("ffprobe-test")),
        Some(absolute("ffmpeg-test")),
        None,
    )
}

/// No media environment at all: `probe_media` is unsupported and there is no
/// rejection reason to report.
fn bare_config() -> WorkerConfig {
    WorkerConfig::from_values(None, None, None)
}

/// Relative paths are rejected by `MediaToolchain::new`, so this configuration
/// carries a rejection reason.
fn broken_config() -> WorkerConfig {
    WorkerConfig::from_values(Some("ffprobe".to_owned()), Some("ffmpeg".to_owned()), None)
}

fn instant_executor() -> JobExecutor {
    Box::new(|_request, _media, _token| CommandOutcome::Completed(None))
}

/// Reports that the job started, then runs until it is cancelled.
fn blocking_executor(started: Sender<TaskId>) -> JobExecutor {
    Box::new(move |request, _media, token| {
        let _ = request;
        started.send(task("0000000f")).unwrap();
        while !token.is_cancelled() {
            thread::sleep(Duration::from_millis(5));
        }
        CommandOutcome::Cancelled
    })
}

/// Reports that the job started, then waits for the test to release it. A
/// dropped release channel or a cancelled token ends the job as cancelled.
fn gated_executor(started: Sender<()>, release: Receiver<()>) -> JobExecutor {
    Box::new(move |_request, _media, token| {
        started.send(()).unwrap();
        if release.recv().is_err() || token.is_cancelled() {
            return CommandOutcome::Cancelled;
        }
        CommandOutcome::Completed(None)
    })
}

/// A process runner that behaves like an external tool killed by the
/// cancellation token: it blocks while the token is clear and then reports the
/// cancellation the real `CancellableProcessRunner` reports after a kill.
struct BlockingRunner {
    started: Mutex<Sender<()>>,
    token: CancellationToken,
}

impl ProcessRunner for BlockingRunner {
    fn run(
        &self,
        _spec: &CommandSpec,
        _timeout: Duration,
    ) -> Result<ProcessOutput, MediaError> {
        self.started.lock().unwrap().send(()).unwrap();
        while !self.token.is_cancelled() {
            thread::sleep(Duration::from_millis(5));
        }
        Err(MediaError::ToolCancelled {
            operation: "ffprobe",
        })
    }
}

fn blocking_probe_executor(started: Sender<()>) -> JobExecutor {
    Box::new(move |request, media, token| {
        let runner = BlockingRunner {
            started: Mutex::new(started.clone()),
            token: token.clone(),
        };
        execute_with_runner(request, media, token, &runner)
    })
}

fn events(frames: &[ServerFrame]) -> Vec<&Event> {
    frames
        .iter()
        .filter_map(|frame| match frame {
            ServerFrame::Event(event) => Some(event),
            _ => None,
        })
        .collect()
}

fn stages(frames: &[ServerFrame]) -> Vec<(&str, &str)> {
    events(frames)
        .into_iter()
        .map(|event| (event.task_id.as_str(), event.stage.as_slug()))
        .collect()
}

fn rejections(frames: &[ServerFrame]) -> Vec<&str> {
    frames
        .iter()
        .filter_map(|frame| match frame {
            ServerFrame::Rejected(rejected) => Some(rejected.reason.as_str()),
            _ => None,
        })
        .collect()
}

/// One task at a time: no task may report a non-terminal stage while another
/// task is still in flight. A task that is cancelled while queued terminates
/// without ever being in flight, which is the one allowed exception.
fn assert_serialized(frames: &[ServerFrame]) {
    let mut in_flight: Option<TaskId> = None;
    for event in events(frames) {
        match (in_flight.as_ref(), event.stage.is_terminal()) {
            (Some(active), true) if *active == event.task_id => in_flight = None,
            (_, true) => assert_eq!(
                event.stage,
                TaskStage::Cancelled,
                "only a queued task may terminate without running: {frames:?}"
            ),
            (None, false) => in_flight = Some(event.task_id.clone()),
            (Some(_), false) => panic!("two tasks were in flight at once: {frames:?}"),
        }
    }
}

#[test]
fn ready_is_the_first_frame_and_reports_the_cpu_adapter() {
    let frames = Harness::start(bare_config(), instant_executor()).finish();

    let ServerFrame::Ready(ready) = &frames[0] else {
        panic!("the first frame must be ready: {frames:?}");
    };
    assert_eq!(ready.protocol_version, PROTOCOL_VERSION);
    assert_eq!(ready.adapters.len(), 1);
    assert_eq!(ready.adapters[0].adapter_id, CPU_ADAPTER_ID);
    assert_eq!(frames.len(), 1, "an idle worker emits nothing else");
}

#[test]
fn a_usable_media_toolchain_enables_probe_media_in_the_handshake() {
    let frames = Harness::start(media_config(), instant_executor()).finish();

    let ServerFrame::Ready(ready) = &frames[0] else {
        panic!("the first frame must be ready: {frames:?}");
    };
    assert_eq!(
        ready.supported_commands,
        vec![TaskKind::ValidateProject, TaskKind::ProbeMedia]
    );
}

#[test]
fn a_rejected_media_configuration_leaves_probe_media_out_of_the_handshake() {
    let frames = Harness::start(broken_config(), instant_executor()).finish();

    let ServerFrame::Ready(ready) = &frames[0] else {
        panic!("the first frame must be ready: {frames:?}");
    };
    assert_eq!(ready.supported_commands, vec![TaskKind::ValidateProject]);
}

#[test]
fn an_unsupported_command_is_rejected_without_creating_a_task() {
    let harness = Harness::start(media_config(), instant_executor());
    harness.send(&start(&task("0000000a"), train_request()));
    let frames = harness.finish();

    let reasons = rejections(&frames);
    assert_eq!(reasons.len(), 1, "{frames:?}");
    assert!(reasons[0].contains("train"), "{}", reasons[0]);
    assert!(events(&frames).is_empty(), "a rejected start creates no task");
}

#[test]
fn probe_media_is_rejected_when_the_media_toolchain_is_unavailable() {
    let harness = Harness::start(broken_config(), instant_executor());
    let request = Request::ProbeMedia(ProbeMediaParams {
        input: PathBuf::from("C:/tmp/input.mp4"),
    });
    harness.send(&start(&task("0000000a"), request));
    let frames = harness.finish();

    let reasons = rejections(&frames);
    assert_eq!(reasons.len(), 1, "{frames:?}");
    assert!(reasons[0].contains("probe_media"), "{}", reasons[0]);
    assert!(events(&frames).is_empty());
}

#[test]
fn a_protocol_version_mismatch_is_rejected() {
    let harness = Harness::start(bare_config(), instant_executor());
    harness.send(&ClientFrame::Start(StartFrame {
        protocol_version: PROTOCOL_VERSION - 1,
        task_id: task("0000000a"),
        request: validate_project("C:/tmp/project"),
    }));
    let frames = harness.finish();

    let reasons = rejections(&frames);
    assert_eq!(reasons.len(), 1, "{frames:?}");
    assert!(
        reasons[0].contains(&PROTOCOL_VERSION.to_string()),
        "{}",
        reasons[0]
    );
    assert!(events(&frames).is_empty());
}

#[test]
fn an_undecodable_line_is_rejected_without_ending_the_session() {
    let harness = Harness::start(bare_config(), instant_executor());
    harness.send_raw("{ not json");
    harness.send(&start(&task("0000000a"), validate_project("C:/tmp/project")));
    let frames = harness.finish();

    assert_eq!(rejections(&frames).len(), 1, "{frames:?}");
    assert_eq!(
        stages(&frames),
        vec![
            ("1787900000000-0000000a", "preparing"),
            ("1787900000000-0000000a", "completed"),
        ]
    );
}

#[test]
fn queued_tasks_run_one_after_another_in_arrival_order() {
    let (started_tx, started_rx) = mpsc::channel::<()>();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let harness = Harness::start(bare_config(), gated_executor(started_tx, release_rx));

    let first = task("0000000a");
    let second = task("0000000b");
    harness.send(&start(&first, validate_project("C:/tmp/first")));
    harness.send(&start(&second, validate_project("C:/tmp/second")));

    started_rx.recv().unwrap();
    let frames = harness.wait_for("the first task to start", |frames| {
        !stages(frames).is_empty()
    });
    assert_eq!(
        stages(&frames),
        vec![("1787900000000-0000000a", "preparing")],
        "the second task must stay queued while the first runs"
    );

    release_tx.send(()).unwrap();
    started_rx.recv().unwrap();
    release_tx.send(()).unwrap();
    let frames = harness.finish();

    assert_eq!(
        stages(&frames),
        vec![
            ("1787900000000-0000000a", "preparing"),
            ("1787900000000-0000000a", "completed"),
            ("1787900000000-0000000b", "preparing"),
            ("1787900000000-0000000b", "completed"),
        ]
    );
    assert_serialized(&frames);
}

#[test]
fn only_one_task_holds_the_cpu_adapter_at_a_time() {
    let (started_tx, started_rx) = mpsc::channel::<()>();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let harness = Harness::start(bare_config(), gated_executor(started_tx, release_rx));

    for suffix in ["0000000a", "0000000b", "0000000c"] {
        harness.send(&start(&task(suffix), validate_project("C:/tmp/project")));
    }
    for _ in 0..3 {
        started_rx.recv().unwrap();
        release_tx.send(()).unwrap();
    }
    let frames = harness.finish();

    assert_eq!(events(&frames).len(), 6, "{frames:?}");
    assert_serialized(&frames);
}

#[test]
fn a_duplicate_task_id_is_rejected() {
    let (started_tx, started_rx) = mpsc::channel::<()>();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let harness = Harness::start(bare_config(), gated_executor(started_tx, release_rx));

    let task_id = task("0000000a");
    harness.send(&start(&task_id, validate_project("C:/tmp/project")));
    started_rx.recv().unwrap();
    harness.send(&start(&task_id, validate_project("C:/tmp/project")));
    harness.wait_for("the duplicate to be rejected", |frames| {
        !rejections(frames).is_empty()
    });
    release_tx.send(()).unwrap();
    let frames = harness.finish();

    let reasons = rejections(&frames);
    assert_eq!(reasons.len(), 1, "{frames:?}");
    assert!(reasons[0].contains(task_id.as_str()), "{}", reasons[0]);
    assert_eq!(
        stages(&frames),
        vec![
            ("1787900000000-0000000a", "preparing"),
            ("1787900000000-0000000a", "completed"),
        ],
        "the duplicate must not affect the running task"
    );
}

#[test]
fn cancelling_a_queued_task_ends_it_before_it_ever_runs() {
    let (started_tx, started_rx) = mpsc::channel::<()>();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let harness = Harness::start(bare_config(), gated_executor(started_tx, release_rx));

    let running = task("0000000a");
    let queued = task("0000000b");
    harness.send(&start(&running, validate_project("C:/tmp/first")));
    started_rx.recv().unwrap();
    harness.send(&start(&queued, validate_project("C:/tmp/second")));
    harness.send(&cancel(&queued));
    harness.wait_for("the queued task to be cancelled", |frames| {
        stages(frames).contains(&("1787900000000-0000000b", "cancelled"))
    });
    release_tx.send(()).unwrap();
    let frames = harness.finish();

    assert_eq!(
        stages(&frames),
        vec![
            ("1787900000000-0000000a", "preparing"),
            ("1787900000000-0000000b", "cancelled"),
            ("1787900000000-0000000a", "completed"),
        ],
        "a cancelled queued task never reports preparing"
    );
    assert!(
        started_rx.try_recv().is_err(),
        "a cancelled queued task must never be handed to the executor"
    );
}

#[test]
fn cancelling_a_running_task_emits_exactly_one_cancelled_event() {
    let (started_tx, started_rx) = mpsc::channel::<TaskId>();
    let harness = Harness::start(bare_config(), blocking_executor(started_tx));

    let task_id = task("0000000a");
    harness.send(&start(&task_id, validate_project("C:/tmp/project")));
    started_rx.recv().unwrap();
    harness.send(&cancel(&task_id));
    harness.send(&cancel(&task_id));
    let frames = harness.finish();

    assert_eq!(
        stages(&frames),
        vec![
            ("1787900000000-0000000a", "preparing"),
            ("1787900000000-0000000a", "cancelled"),
        ]
    );
    assert!(rejections(&frames).is_empty(), "cancel is idempotent");
}

#[test]
fn a_cancelled_external_process_becomes_one_cancelled_event() {
    let input_dir = tempfile::tempdir().unwrap();
    let input = input_dir.path().join("input.mp4");
    fs::write(&input, b"not a real video").unwrap();

    let (started_tx, started_rx) = mpsc::channel::<()>();
    let harness = Harness::start(media_config(), blocking_probe_executor(started_tx));

    let task_id = task("0000000a");
    harness.send(&start(
        &task_id,
        Request::ProbeMedia(ProbeMediaParams { input }),
    ));
    started_rx.recv().unwrap();
    harness.send(&cancel(&task_id));
    let frames = harness.finish();

    assert_eq!(
        stages(&frames),
        vec![
            ("1787900000000-0000000a", "preparing"),
            ("1787900000000-0000000a", "cancelled"),
        ]
    );
    let cancelled = events(&frames)[1];
    assert!(
        cancelled.error.is_none() && cancelled.result.is_none(),
        "{cancelled:?}"
    );
}

#[test]
fn cancelling_an_unknown_or_finished_task_is_silently_accepted() {
    let harness = Harness::start(bare_config(), instant_executor());
    let task_id = task("0000000a");

    harness.send(&cancel(&task("0000000f")));
    harness.send(&start(&task_id, validate_project("C:/tmp/project")));
    harness.wait_for("the task to complete", |frames| {
        stages(frames).contains(&("1787900000000-0000000a", "completed"))
    });
    harness.send(&cancel(&task_id));
    let frames = harness.finish();

    assert!(rejections(&frames).is_empty(), "{frames:?}");
    assert_eq!(
        stages(&frames),
        vec![
            ("1787900000000-0000000a", "preparing"),
            ("1787900000000-0000000a", "completed"),
        ],
        "a late cancel must not add a second terminal event"
    );
}

#[test]
fn a_failing_command_reports_a_failed_event_with_its_error() {
    let harness = Harness::start(
        bare_config(),
        Box::new(|_request, _media, _token| {
            CommandOutcome::Failed(
                feathertalk_domain::TaskError::new(
                    ErrorCode::ProjectInvalid,
                    "项目目录缺少必需文件",
                    "project directory is missing assets/assets.json",
                    TaskStage::Preparing,
                )
                .unwrap(),
            )
        }),
    );
    harness.send(&start(&task("0000000a"), validate_project("C:/tmp/project")));
    let frames = harness.finish();

    let failed = events(&frames)[1];
    assert_eq!(failed.stage.as_slug(), "failed");
    assert_eq!(
        failed.error.as_ref().map(|error| error.code),
        Some(ErrorCode::ProjectInvalid)
    );
}

#[test]
fn shutdown_cancels_queued_tasks_and_waits_for_the_running_one() {
    let (started_tx, started_rx) = mpsc::channel::<TaskId>();
    let harness = Harness::start(bare_config(), blocking_executor(started_tx));

    let running = task("0000000a");
    let queued = task("0000000b");
    harness.send(&start(&running, validate_project("C:/tmp/first")));
    started_rx.recv().unwrap();
    harness.send(&start(&queued, validate_project("C:/tmp/second")));
    harness.send(&shutdown());
    harness.send(&start(&task("0000000c"), validate_project("C:/tmp/third")));
    let frames = harness.finish();

    let observed = stages(&frames);
    assert!(
        observed.contains(&("1787900000000-0000000b", "cancelled")),
        "{frames:?}"
    );
    assert!(
        observed.contains(&("1787900000000-0000000a", "cancelled")),
        "{frames:?}"
    );
    assert!(
        !observed
            .iter()
            .any(|(task_id, _)| *task_id == "1787900000000-0000000c"),
        "a start after shutdown must not create a task: {frames:?}"
    );
    assert_serialized(&frames);
}

#[test]
fn closing_the_input_stream_shuts_the_worker_down() {
    let (started_tx, started_rx) = mpsc::channel::<TaskId>();
    let harness = Harness::start(bare_config(), blocking_executor(started_tx));

    let task_id = task("0000000a");
    harness.send(&start(&task_id, validate_project("C:/tmp/project")));
    started_rx.recv().unwrap();
    let frames = harness.finish();

    assert_eq!(
        stages(&frames),
        vec![
            ("1787900000000-0000000a", "preparing"),
            ("1787900000000-0000000a", "cancelled"),
        ],
        "a closed stdin cancels the running task and exits"
    );
}
```

`blocking_executor` reports a fixed `TaskId` because the executor closure receives the request, not the id; the tests only need the "a job reached the executor" signal, and `assert_serialized` verifies the real ids from the event stream.

- [ ] **Step 2: Run the test to verify it fails for the right reason**

Run: `cargo test -p feathertalk-worker --all-targets`

Expected: FAIL to compile with `error[E0432]: unresolved imports \`feathertalk_worker::JobExecutor\`, \`feathertalk_worker::serve_with_executor\``.

- [ ] **Step 3: Implement the runtime**

Create `rust/crates/feathertalk-worker/src/runtime.rs`:

```rust
use std::{
    collections::{BTreeMap, VecDeque},
    io::{BufRead, Write},
    sync::mpsc::{self, Receiver, Sender},
    thread,
};

use feathertalk_domain::{
    ClientFrame, DomainError, Event, FrameReader, FrameWriter, PROTOCOL_VERSION, RejectedFrame,
    Request, ServerFrame, TaskId, TaskKind, TaskLifecycle, TaskStage,
};
use feathertalk_media::{CancellationToken, MediaToolchain};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

use crate::{
    AdapterLockError, AdapterLocks, CPU_ADAPTER_ID, CommandOutcome, ENV_FFMPEG, ENV_FFPROBE,
    WorkerConfig, execute, ready_frame, supported_commands,
};

/// How the runtime reaches command execution.
///
/// Production callers use [`serve`], which passes [`crate::execute`]. Tests
/// pass a closure so queueing, cancellation, and shutdown can be observed
/// without a real external tool.
pub type JobExecutor = Box<
    dyn Fn(&Request, Option<&MediaToolchain>, &CancellationToken) -> CommandOutcome + Send + 'static,
>;

/// Everything the control loop receives. It is the only owner of task state,
/// so every other thread talks to it through this one channel.
enum ControlMessage {
    Client(ClientFrame),
    ClientError(DomainError),
    InputClosed,
    Emit(Event),
    Finished { task_id: TaskId, adapter_id: String },
}

/// One unit of work handed to the execution thread.
struct Job {
    task_id: TaskId,
    request: Request,
    token: CancellationToken,
    adapter_id: String,
}

struct TaskState {
    lifecycle: TaskLifecycle,
    token: CancellationToken,
    /// True until the job is handed to the execution thread. A queued task can
    /// be cancelled without ever running.
    queued: bool,
}

/// Serve one client session over `input`/`output` until shutdown or EOF.
pub fn serve<R, W>(input: R, output: W, config: &WorkerConfig) -> Result<(), DomainError>
where
    R: BufRead + Send + 'static,
    W: Write,
{
    serve_with_executor(input, output, config, Box::new(execute))
}

/// [`serve`] with an injected executor.
pub fn serve_with_executor<R, W>(
    input: R,
    output: W,
    config: &WorkerConfig,
    executor: JobExecutor,
) -> Result<(), DomainError>
where
    R: BufRead + Send + 'static,
    W: Write,
{
    let mut writer = FrameWriter::new(output);
    // The handshake goes out before a single byte is read, so a desktop that
    // sees an incompatible version never has to send a request first.
    write_frame(&mut writer, &ServerFrame::Ready(ready_frame(config)))?;

    let (control_tx, control_rx) = mpsc::channel::<ControlMessage>();
    let (job_tx, job_rx) = mpsc::channel::<Job>();

    let input_tx = control_tx.clone();
    // The input thread is deliberately detached. After `shutdown` the control
    // loop stops reading, and a thread blocked on a still-open stdin must not
    // keep the process alive; it dies with the process instead.
    let _ = thread::spawn(move || read_input(input, &input_tx));

    let execution_tx = control_tx;
    let media = config.media().cloned();
    let execution = thread::spawn(move || run_jobs(&job_rx, &execution_tx, media, executor));

    let result = control_loop(&control_rx, &mut writer, &job_tx, config);

    // Dropping the sender ends the execution thread's receive loop.
    drop(job_tx);
    let _ = execution.join();
    let mut output = writer.into_inner();
    output.flush().map_err(|error| DomainError::MalformedFrame {
        reason: error.to_string(),
    })?;
    result
}

fn read_input<R: BufRead>(input: R, control_tx: &Sender<ControlMessage>) {
    let mut reader = FrameReader::new(input);
    while let Some(decoded) = reader.read_frame::<ClientFrame>() {
        // `FrameReader` is syntax-only, so semantic validation happens here.
        // `ClientFrame::validate` includes the protocol-version check.
        let message = match decoded {
            Ok(frame) => match frame.validate() {
                Ok(()) => ControlMessage::Client(frame),
                Err(error) => ControlMessage::ClientError(error),
            },
            Err(error) => ControlMessage::ClientError(error),
        };
        if control_tx.send(message).is_err() {
            return;
        }
    }
    let _ = control_tx.send(ControlMessage::InputClosed);
}

fn run_jobs(
    job_rx: &Receiver<Job>,
    control_tx: &Sender<ControlMessage>,
    media: Option<MediaToolchain>,
    executor: JobExecutor,
) {
    while let Ok(job) = job_rx.recv() {
        let event = match executor(&job.request, media.as_ref(), &job.token) {
            CommandOutcome::Completed(result) => {
                let mut event = Event::new(job.task_id.clone(), &now_rfc3339(), TaskStage::Completed);
                event.result = result;
                event
            }
            CommandOutcome::Cancelled => {
                Event::new(job.task_id.clone(), &now_rfc3339(), TaskStage::Cancelled)
            }
            CommandOutcome::Failed(error) => {
                let stage = TaskStage::Failed {
                    code: error.code,
                    message: error.summary.clone(),
                };
                let mut event = Event::new(job.task_id.clone(), &now_rfc3339(), stage);
                event.error = Some(error);
                event
            }
        };
        let _ = control_tx.send(ControlMessage::Emit(event));
        // The adapter is released only after the event, so the next task never
        // starts before the previous one is reported.
        let _ = control_tx.send(ControlMessage::Finished {
            task_id: job.task_id,
            adapter_id: job.adapter_id,
        });
    }
}

fn control_loop<W: Write>(
    control_rx: &Receiver<ControlMessage>,
    writer: &mut FrameWriter<W>,
    job_tx: &Sender<Job>,
    config: &WorkerConfig,
) -> Result<(), DomainError> {
    let supported = supported_commands(config);
    let mut tasks: BTreeMap<TaskId, TaskState> = BTreeMap::new();
    let mut pending: VecDeque<Job> = VecDeque::new();
    let mut locks = AdapterLocks::new([CPU_ADAPTER_ID.to_owned()]);
    let mut active: Option<TaskId> = None;
    let mut draining = false;

    while let Ok(message) = control_rx.recv() {
        match message {
            ControlMessage::Client(ClientFrame::Start(frame)) => {
                if draining {
                    reject(writer, "worker 正在关闭，请等待进程退出后重新启动任务。".to_owned())?;
                } else if !supported.contains(&frame.request.kind()) {
                    reject(writer, unsupported_reason(&frame.request, config))?;
                } else if tasks.contains_key(&frame.task_id) {
                    reject(
                        writer,
                        format!(
                            "task_id {} 已存在，请为新任务生成新的 task_id。",
                            frame.task_id.as_str()
                        ),
                    )?;
                } else {
                    let token = CancellationToken::new();
                    tasks.insert(
                        frame.task_id.clone(),
                        TaskState {
                            lifecycle: TaskLifecycle::new(),
                            token: token.clone(),
                            queued: true,
                        },
                    );
                    pending.push_back(Job {
                        task_id: frame.task_id,
                        request: frame.request,
                        token,
                        adapter_id: CPU_ADAPTER_ID.to_owned(),
                    });
                }
            }
            ControlMessage::Client(ClientFrame::Cancel(frame)) => {
                // Cancel is idempotent: an unknown or already terminal task is
                // accepted silently.
                let cancel_queued = match tasks.get_mut(&frame.task_id) {
                    Some(state) if !state.lifecycle.is_terminal() => {
                        state.token.cancel();
                        let queued = state.queued;
                        state.queued = false;
                        queued
                    }
                    _ => false,
                };
                if cancel_queued {
                    pending.retain(|job| job.task_id != frame.task_id);
                    let event =
                        Event::new(frame.task_id, &now_rfc3339(), TaskStage::Cancelled);
                    emit(writer, &mut tasks, event)?;
                }
            }
            ControlMessage::Client(ClientFrame::Shutdown(_)) | ControlMessage::InputClosed => {
                draining = true;
                begin_drain(writer, &mut tasks, &mut pending, active.as_ref())?;
            }
            ControlMessage::ClientError(error) => reject(writer, client_error_reason(&error))?,
            ControlMessage::Emit(event) => emit(writer, &mut tasks, event)?,
            ControlMessage::Finished {
                task_id,
                adapter_id,
            } => {
                locks.release(&adapter_id).map_err(lock_failure)?;
                if active.as_ref() == Some(&task_id) {
                    active = None;
                }
            }
        }

        if draining {
            if active.is_none() {
                break;
            }
        } else {
            dispatch(writer, &mut tasks, &mut pending, &mut locks, &mut active, job_tx)?;
        }
    }
    Ok(())
}

/// Hand the next queued job to the execution thread if the runtime is idle and
/// its adapter is free.
fn dispatch<W: Write>(
    writer: &mut FrameWriter<W>,
    tasks: &mut BTreeMap<TaskId, TaskState>,
    pending: &mut VecDeque<Job>,
    locks: &mut AdapterLocks,
    active: &mut Option<TaskId>,
    job_tx: &Sender<Job>,
) -> Result<(), DomainError> {
    if active.is_some() {
        return Ok(());
    }
    let Some(job) = pending.pop_front() else {
        return Ok(());
    };
    if !locks.is_free(&job.adapter_id) {
        pending.push_front(job);
        return Ok(());
    }
    locks
        .acquire(&job.adapter_id, job.task_id.clone())
        .map_err(lock_failure)?;
    if let Some(state) = tasks.get_mut(&job.task_id) {
        state.queued = false;
    }
    *active = Some(job.task_id.clone());
    let event = Event::new(job.task_id.clone(), &now_rfc3339(), TaskStage::Preparing);
    emit(writer, tasks, event)?;
    job_tx.send(job).map_err(|_| DomainError::MalformedFrame {
        reason: "execution thread stopped before the task was dispatched".to_owned(),
    })
}

/// Stop accepting work: cancel every queued task with its own `cancelled`
/// event and ask the running task to stop.
fn begin_drain<W: Write>(
    writer: &mut FrameWriter<W>,
    tasks: &mut BTreeMap<TaskId, TaskState>,
    pending: &mut VecDeque<Job>,
    active: Option<&TaskId>,
) -> Result<(), DomainError> {
    for job in std::mem::take(pending) {
        if let Some(state) = tasks.get_mut(&job.task_id) {
            state.token.cancel();
            state.queued = false;
        }
        let event = Event::new(job.task_id, &now_rfc3339(), TaskStage::Cancelled);
        emit(writer, tasks, event)?;
    }
    if let Some(task_id) = active
        && let Some(state) = tasks.get(task_id)
    {
        state.token.cancel();
    }
    Ok(())
}

/// Advance the task lifecycle and write the event.
///
/// A task whose lifecycle is already terminal silently drops the event. That is
/// what guarantees at most one terminal event per task even when a cancel and a
/// natural completion race each other.
fn emit<W: Write>(
    writer: &mut FrameWriter<W>,
    tasks: &mut BTreeMap<TaskId, TaskState>,
    event: Event,
) -> Result<(), DomainError> {
    let Some(state) = tasks.get_mut(&event.task_id) else {
        return Ok(());
    };
    if state.lifecycle.advance(event.stage.clone()).is_err() {
        return Ok(());
    }
    write_frame(writer, &ServerFrame::Event(event))
}

fn reject<W: Write>(writer: &mut FrameWriter<W>, reason: String) -> Result<(), DomainError> {
    write_frame(
        writer,
        &ServerFrame::Rejected(RejectedFrame {
            protocol_version: PROTOCOL_VERSION,
            reason,
        }),
    )
}

fn write_frame<W: Write>(
    writer: &mut FrameWriter<W>,
    frame: &ServerFrame,
) -> Result<(), DomainError> {
    frame.validate()?;
    writer.write_frame(frame)
}

fn unsupported_reason(request: &Request, config: &WorkerConfig) -> String {
    let slug = request.kind().as_slug();
    match request.kind() {
        TaskKind::ProbeMedia => match config.media_rejection() {
            Some(rejection) => format!(
                "命令 {slug} 需要可用的媒体工具链，当前配置被拒绝：{rejection}。修正后重启 worker。"
            ),
            None => format!(
                "命令 {slug} 需要媒体工具链，请设置 {ENV_FFPROBE} 与 {ENV_FFMPEG} 后重启 worker。"
            ),
        },
        _ => format!("此 worker 不支持命令 {slug}，当前仅支持 validate_project 与 probe_media。"),
    }
}

fn client_error_reason(error: &DomainError) -> String {
    match error {
        DomainError::ProtocolVersion { expected, actual } => format!(
            "协议版本不兼容：worker 使用 {expected}，收到 {actual}。请升级桌面端后重试。"
        ),
        other => format!("无法解析请求帧：{other}。请检查帧格式后重试。"),
    }
}

/// An adapter lock error at this point is an internal invariant violation: the
/// control loop is the only owner of the table.
fn lock_failure(error: AdapterLockError) -> DomainError {
    DomainError::InvalidField {
        field: "adapter_id",
        reason: error.to_string(),
    }
}

fn now_rfc3339() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .expect("formatting a UTC timestamp as RFC 3339 cannot fail")
}
```

- [ ] **Step 4: Export the runtime**

In `rust/crates/feathertalk-worker/src/lib.rs`, add `mod runtime;` after `mod probe_result;` and the re-export:

```rust
pub use runtime::{JobExecutor, serve, serve_with_executor};
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cargo test -p feathertalk-worker --all-targets`

Expected: PASS, exit code 0. If a test times out inside `wait_for`, the panic message prints every frame written so far.

- [ ] **Step 6: Write the failing process-boundary test**

Create `rust/crates/feathertalk-worker/tests/process_boundary.rs`:

```rust
use std::{
    io::{BufRead, BufReader, Write},
    process::{Command, Stdio},
};

use feathertalk_domain::{
    ClientFrame, PROTOCOL_VERSION, ServerFrame, ShutdownFrame, TaskKind, decode_line, encode_line,
};
use feathertalk_worker::{ENV_FFMPEG, ENV_FFPROBE, ENV_MEDIA_TIMEOUT_MS};

/// The real binary with a cleared media environment, so the handshake it
/// reports does not depend on the developer's machine.
fn worker_command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_feathertalk-worker"));
    command
        .env_remove(ENV_FFPROBE)
        .env_remove(ENV_FFMPEG)
        .env_remove(ENV_MEDIA_TIMEOUT_MS)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command
}

#[test]
fn the_binary_announces_itself_and_exits_zero_on_shutdown() {
    let mut child = worker_command().spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();

    let ready_line = lines.next().expect("the worker must write ready").unwrap();
    let frame: ServerFrame = decode_line(&ready_line).unwrap();
    frame.validate().unwrap();
    let ServerFrame::Ready(ready) = frame else {
        panic!("the first frame must be ready: {ready_line}");
    };
    assert_eq!(ready.protocol_version, PROTOCOL_VERSION);
    assert_eq!(ready.supported_commands, vec![TaskKind::ValidateProject]);

    let shutdown = ClientFrame::Shutdown(ShutdownFrame {
        protocol_version: PROTOCOL_VERSION,
    });
    writeln!(stdin, "{}", encode_line(&shutdown).unwrap()).unwrap();
    stdin.flush().unwrap();

    // stdin stays open on purpose: shutdown alone must end the process.
    assert!(
        lines.next().is_none(),
        "the worker must write nothing after shutdown"
    );
    let status = child.wait().unwrap();
    assert!(status.success(), "{status:?}");
    drop(stdin);
}

#[test]
fn closing_stdin_exits_the_binary_cleanly() {
    let mut child = worker_command().spawn().unwrap();
    drop(child.stdin.take().unwrap());

    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{:?}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    let frames: Vec<ServerFrame> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| decode_line(line).unwrap())
        .collect();
    assert_eq!(frames.len(), 1, "only the handshake is expected: {text}");
}
```

Run: `cargo test -p feathertalk-worker --all-targets`

Expected: FAIL to compile with `error: environment variable \`CARGO_BIN_EXE_feathertalk-worker\` not defined at compile time`, because the crate has no binary target yet.

- [ ] **Step 7: Add the binary**

Create `rust/crates/feathertalk-worker/src/main.rs`:

```rust
use std::io::{self, BufReader};

use feathertalk_worker::{WorkerConfig, serve};

fn main() {
    let config = WorkerConfig::from_env();
    // stdout carries the protocol; every diagnostic goes to stderr.
    if let Err(error) = serve(BufReader::new(io::stdin()), io::stdout(), &config) {
        eprintln!("feathertalk-worker: {error}");
        std::process::exit(1);
    }
}
```

Cargo discovers `src/main.rs` as the `feathertalk-worker` binary alongside the library; no `[[bin]]` section is needed.

- [ ] **Step 8: Run the test to verify it passes**

Run: `cargo test -p feathertalk-worker --all-targets`

Expected: PASS, exit code 0.

- [ ] **Step 9: Commit**

```bash
git add rust/crates/feathertalk-worker
git commit -m "feat(worker): serve the task protocol over stdio"
```

---

### Task 9: Record the protocol in the migration design and close the slice

**Files:**

- Modify: `docs/superpowers/specs/2026-08-17-rust-desktop-migration-design.md`
- Modify: `docs/superpowers/plans/2026-09-01-feathertalk-worker-runtime.md` (tick every checkbox)

**Interfaces:** none. This task changes documentation and verifies the whole workspace.

- [ ] **Step 1: Record the version-2 protocol properties in the migration design**

In `docs/superpowers/specs/2026-08-17-rust-desktop-migration-design.md`, section `### 4.2`, the bullet list under `协议具备以下属性：` currently reads:

```markdown
- 每条请求包含 `protocol_version`、`task_id`、命令和参数。
- 每条事件包含 `task_id`、阶段、进度、时间和可选指标。
- 取消请求是幂等操作。
- worker 启动时报告版本、支持的 backend、adapter 和功能列表。
- 协议版本不兼容时，桌面端拒绝启动任务并显示可操作错误。
```

Replace that list with:

```markdown
- 每条请求包含 `protocol_version`、`task_id`、命令和参数。当前协议版本为 `2`。
- 每条事件包含 `task_id`、阶段、进度、时间和可选指标。
- `completed` 事件可携带 `result` JSON 对象：`probe_media` 返回探测结果，`validate_project` 返回 `null`。
- 取消请求是幂等操作。
- worker 启动时报告版本、支持的 backend、adapter、功能列表和 `supported_commands`。
- 桌面端请求 `supported_commands` 之外的命令时，worker 在 `start` 阶段即返回 `rejected`，不产生任务。
- 协议版本不兼容时，桌面端拒绝启动任务并显示可操作错误。
```

Do not touch any other section of the spec. The design document describes the whole migration; this slice only settles the protocol properties listed above.

- [ ] **Step 2: Verify the whole workspace**

Run each command from `E:/workspace/github/FeatherTalk/rust`. Every command must exit 0.

```bash
cargo test --workspace --all-targets
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

Expected: all pass. `cargo test --workspace --all-targets` takes roughly 30 minutes; do not interrupt it. The 13 pre-existing ignored tests stay ignored — an ignored count of 13 is the expected result, not a failure.

Then run from the repository root:

```bash
git diff --check
```

Expected: no output, exit code 0 (no trailing whitespace and no conflict markers).

- [ ] **Step 3: Tick every checkbox in this plan**

Go through `docs/superpowers/plans/2026-09-01-feathertalk-worker-runtime.md` and change every `- [ ]` to `- [x]`. Verify with:

```bash
rg -c '^- \[ \]' docs/superpowers/plans/2026-09-01-feathertalk-worker-runtime.md
```

Expected: no match, exit code 1.

This step is easy to skip and it has been skipped before. Of the 28 plans already in `docs/superpowers/plans/`, only three (`2026-08-17-burn-feasibility-loop.md`, `2026-08-27-onnx-opset17-export.md`, `2026-08-28-worker-protocol-task-domain.md`) have their boxes ticked; the other 25 still show unticked boxes for work that shipped. A stale plan is worse than no plan, because the next reader cannot tell which steps actually ran.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-08-17-rust-desktop-migration-design.md docs/superpowers/plans/2026-09-01-feathertalk-worker-runtime.md
git commit -m "docs: record the version 2 task protocol"
```

Never stage `demo/kanghui_training_video_featherhubert_188_latest/`. Check with `git status --short` before committing that the directory is still listed as untracked (`??`).

- [ ] **Step 5: Finish the branch**

Use the `superpowers:finishing-a-development-branch` skill with base branch `main`.

Hazard: if the work happened in a git worktree, `git worktree remove` can hang for minutes on a populated `rust/target` directory. Delete `rust/target` first, or pass the removal enough time instead of interrupting it.

---

## Definition Of Done

- `rust/crates/feathertalk-worker` exists as a workspace member with a library and a `feathertalk-worker` binary, and its production dependencies are exactly `feathertalk-domain`, `feathertalk-media`, `feathertalk-project`, `serde_json`, `thiserror`, `time`, with `tempfile` as the only dev dependency.
- `PROTOCOL_VERSION` is `2`, `ReadyFrame` carries a required `supported_commands: Vec<TaskKind>`, and `Event` carries `result: Option<serde_json::Value>` that validation permits only on `TaskStage::Completed` and only as a JSON object.
- The worker writes its `ready` frame before reading any input, reports `Backend::Cpu` with the single adapter `cpu-0`, and lists `validate_project` plus `probe_media` when a media toolchain is configured, or `validate_project` alone when it is not.
- `start` for an unsupported command, a duplicate `task_id`, an undecodable frame, an incompatible protocol version, or a session already draining produces a `rejected` frame with an actionable Chinese reason and no task.
- `validate_project` and `probe_media` run on a worker thread, emit `preparing` before execution and exactly one terminal event afterwards, and `probe_media` reports its probe result as a JSON object on the `completed` event.
- Every library error reaching the wire is mapped onto one of the ten `ErrorCode` values with a Chinese `summary`, an English `detail`, and a `recovery` hint.
- `cancel` is idempotent, cancels a queued task without running it, kills a running `ffprobe` through `CancellationToken`, and produces at most one terminal event per task even when cancellation races completion.
- `shutdown` and stdin EOF both drain the queue, cancel the running task, and exit the process with status 0.
- The adapter lock table serialises tasks that need the same adapter and releases the lock only after the terminal event has been written.
- Every frame is `validate()`d before it is written and after it is decoded.
- `cargo test --workspace --all-targets`, `cargo check --workspace --all-targets`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --all -- --check` and `git diff --check` all exit 0.
- No file under `rust/crates/feathertalk-project/` is modified, and `demo/kanghui_training_video_featherhubert_188_latest/` is never staged.
- Section 4.2 of `docs/superpowers/specs/2026-08-17-rust-desktop-migration-design.md` records protocol version 2, `supported_commands`, and the `result` field, and every checkbox in this plan is ticked.
