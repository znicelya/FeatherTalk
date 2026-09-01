# feathertalk CLI Worker Client Design

## 1. Scope

This is milestone slice 3: CLI parity over the worker protocol. It depends on
`feathertalk-domain` for the JSON Lines protocol and on the slice 2 worker
binary for execution.

The slice delivers a command-line front end that is a protocol client, not a
second execution path:

- A reusable library that spawns the worker, performs the handshake, drives one
  task, cancels it, and reaps the child process.
- A binary that parses arguments, renders human-readable Chinese output,
  and maps session outcomes onto process exit codes.
- Dynamic capability negotiation, so the command surface follows the worker's
  reported `supported_commands` instead of a compile-time assumption.

The enabled commands are exactly the ones the worker enables: `validate-project`
and `probe-media`, plus a `capabilities` command that only performs the
handshake. Any other request variant is out of scope until the worker enables
it.

The CLI never links the execution crates and never reimplements validation. All
domain judgement stays in the worker, so the CLI and the future GPUI workbench
observe identical behaviour.

## 2. Architecture

Two new workspace members:

```text
feathertalk-cli (bin: feathertalk)
  |
  | depends on
  v
feathertalk-client (lib)
  |
  | depends on
  v
feathertalk-domain
```

`feathertalk-client` owns the protocol and the process lifetime. It has no
argument parsing and no terminal rendering:

- `WorkerLocator` resolves the worker executable.
- `WorkerSession` spawns the child, performs the handshake, exposes the
  `ReadyFrame`, runs one task, and cancels or shuts down.
- `SessionOptions` carries the two cancellation deadlines, `cancel_grace`
  (default 10 s) and `shutdown_grace` (default 5 s), so tests can inject short
  values instead of waiting out the production timeouts.
- `CancelToken` is an `Arc<AtomicUsize>` request counter that callers bump from
  any thread, including a signal handler.
- `EventSink` is the callback trait the caller implements to observe frames.
- `SessionOutcome` reports the authoritative end state of a task.

`feathertalk-cli` owns everything user-facing: clap parsing, the stage
dictionary, the two output modes, the Ctrl-C handler, and exit codes.

The split exists so milestone 5 can reuse the same RPC path from the desktop
workbench by supplying a different `EventSink`, without duplicating handshake,
cancellation, or child-reaping logic.

Threading follows the worker's constraints: `std::thread` and
`std::sync::mpsc`, no async runtime, no burn or GPUI dependency.

### 2.1 Worker Discovery

`WorkerLocator` probes, in order:

1. An explicit `--worker <PATH>`.
2. The `FEATHERTALK_WORKER_BIN` environment variable.
3. A sibling of the CLI executable named `feathertalk-worker`, with the
   platform executable suffix on Windows.

If none resolves to an existing file, the error names all three probed
locations so the user can fix the environment without reading source. The CLI
injects no `FEATHERTALK_WORKER_*` configuration into the child; the worker
reads its own configuration through `WorkerConfig::from_env()`.

## 3. Session Lifecycle and Capability Negotiation

```text
spawn --> handshake --> capability gate --> run --> teardown
```

Spawn uses piped stdin, stdout, and stderr. A spawn failure is a session-level
error.

Handshake reads exactly one line, decodes a `ServerFrame`, requires
`ServerFrame::Ready`, and calls `validate()`, which enforces the protocol
version. Three handshake failures are distinguished in the message:

- The first line is absent, malformed, or not a `ready` frame, including the
  case where the child dies immediately. The tail of the worker's stderr is
  attached.
- `DomainError::ProtocolVersion { expected, actual }` reports both versions and
  advises rebuilding both binaries from the same revision.
- A `rejected` frame in the handshake position prints its reason verbatim,
  because the worker's rejection reasons are already actionable Chinese.

The capability gate compares the requested `TaskKind`, obtained from
`Request::kind()`, against `ready.supported_commands`. There is no second
hard-coded mapping table. If the command is absent, the CLI prints the
requested slug and the supported slugs. For a missing `probe_media` it also
names `FEATHERTALK_WORKER_FFPROBE`, because an unresolved ffprobe path is
precisely what removes that command from the list without failing worker
startup.

Run generates a `TaskId`, writes a validated `start` frame, and consumes
`ServerFrame`s until one of these ends the task:

- an `Event` whose `TaskStage::is_terminal()` is true,
- a `rejected` frame,
- stdout EOF.

Events carrying a different `task_id` are ignored and counted; the count is
reported as a diagnostic because it indicates a worker defect.

`SessionOutcome` has four variants: `Completed { result: Option<Value> }`,
`Failed(TaskError)`, `Cancelled`, and `SessionError(ClientError)`.

Teardown writes `shutdown`, closes stdin, and waits for the child. The worker
already exits zero on either `shutdown` or stdin EOF, so the normal path never
kills the child. A non-zero child exit after a terminal event does not change
the CLI exit code; it only adds one diagnostic line on stderr.

The `capabilities` command stops after the handshake, prints the reported
`protocol_version`, `worker_version`, backends, adapters, `supported_commands`,
and `capabilities` fields, then performs the same teardown and exits zero.
Its output goes to stdout, because those fields are the command's product
rather than progress commentary: a Chinese human-readable listing by default,
and the verbatim `ready` line under `--json`. `--quiet` does not change it,
since the command emits no stage or progress lines.

## 4. Command Surface

The binary is `feathertalk`. All clap help text is Chinese.

Global options:

| Option | Type | Purpose |
| --- | --- | --- |
| `--worker <PATH>` | `Option<PathBuf>` | Highest-priority worker executable |
| `--json` | flag | Pass worker frames through to stdout verbatim |
| `--quiet` | flag | Suppress stage and progress lines |
| `--task-id <ID>` | `Option<String>` | Override the generated task ID |

`--json` and `--quiet` are mutually exclusive through clap's `conflicts_with`,
because their output semantics would otherwise be ambiguous.

Subcommands:

- `feathertalk validate-project <PROJECT_DIR>` builds
  `Request::ValidateProject(ProjectDirParams { project_dir })`.
- `feathertalk probe-media <INPUT>` builds
  `Request::ProbeMedia(ProbeMediaParams { input })`.
- `feathertalk capabilities` performs the handshake only.

Path arguments are checked for emptiness and passed through unchanged. The CLI
does not canonicalize them: the worker is the authority on existence, entry
type, and manifest state, and canonicalization on Windows would inject a
`\\?\` prefix into error messages. Duplicating validation would create two
divergent sets of error wording.

Task IDs are generated locally as `<command-slug>-<UTC timestamp>-<pid>`, for
example `validate-project-20260901T083012Z-12345`, and then passed through
`TaskId::parse` before use. The `time` crate is already a workspace dependency,
so this needs no new dependency and keeps log correlation readable. An invalid
`--task-id` is a session-level error.

## 5. Output Contract

Stream roles in the default mode, for the two task commands:

- stdout carries only machine-consumable output: the `result` object of the
  `completed` event, pretty-printed with a trailing newline. A `None` result
  prints `{}`, so a task command's stdout is always exactly one JSON object.
- stderr carries every human-facing Chinese line: stages, progress, error
  summaries, and diagnostics.

This makes `feathertalk probe-media input.mp4 > probe.json` correct without
`--json`.

`capabilities` is the one exception, defined at the end of section 3.

Default stderr shape:

```text
[preparing] 正在准备任务
[extracting-frames] 抽帧 128/512 (25%)
[failed] 媒体文件无效
  错误码: media_invalid
  阶段: preparing
  建议: 重试
  详情: ffprobe exited with status 1
```

The extraction line is illustrative of the general renderer; the two commands
enabled in this slice only report `Preparing` and a terminal stage.

The bracketed name is `TaskStage::as_slug()`. The Chinese description comes
from a CLI-side table that matches all thirteen `TaskStage` variants with no
`_` arm, so adding a stage breaks the build instead of silently rendering
nothing. A percentage is printed only when `progress.total` is `Some`;
otherwise the completed count is printed alone. Non-`None` `Metrics` fields are
appended to the same line.

`--quiet` suppresses stage and progress lines but keeps terminal and error
output.

`--json` writes every worker line to stdout byte-for-byte, including `ready`,
while CLI diagnostics stay on stderr. Frames are still decoded and validated to
drive the state machine, but the original line is what gets printed:
`serde_json` is built without `preserve_order`, so a serialization round trip
through `Value` would reorder object keys.

## 6. Exit Codes

| Code | Condition | Output |
| --- | --- | --- |
| 0 | `Completed` event | `result` on stdout |
| 1 | `Failed { code, .. }` event | code, stage, recovery, summary, detail on stderr |
| 2 | `Cancelled` event, or a forced local interrupt | one Chinese line on stderr |
| 3 | Session-level error | reason plus the worker stderr tail |

Session-level errors are: worker not found, spawn failure, handshake failure,
protocol version mismatch, a `rejected` frame, an unsupported command, an
invalid `--task-id`, a malformed or oversized frame, a broken pipe while
writing, and stdout EOF without a terminal event.

The boundary between 1 and 3 is whether the worker produced a structured task
failure. If it did, the failure is a business outcome and the code is 1. If it
did not, the problem is environmental or a protocol violation and the code is
3, which lets scripts decide whether retrying can help.

An `ErrorCode::TaskCancelled` delivered inside a `Failed` event still exits 1.
Exit code 2 is reserved for the `Cancelled` stage and for forced local
interrupts.

## 7. Cancellation

A blocking read on the child's stdout cannot observe a signal, so
`WorkerSession::run` reads frames on a dedicated thread that forwards decoded
results over an `mpsc` channel. The main loop uses `recv_timeout(100ms)` and
checks the cancel token on every timeout, which bounds interrupt latency at
roughly 100 ms.

`feathertalk-client` does not depend on `ctrlc`. It exposes `CancelToken` with
`request()` and `count()`. The CLI installs the `ctrlc` handler, and the
handler body only bumps the counter: no allocation and no I/O, which is what a
signal handler may safely do. Tests drive `CancelToken` directly instead of
raising real signals.

Sequence, keyed on the request count:

```text
count 0 -> 1 : write cancel, wait up to 10s for a terminal event
   |- Cancelled received          -> exit 2
   |- Completed or Failed received -> honour it (exit 0 or 1)
   |- 10s timeout                 -> write shutdown, wait 5s, then kill -> exit 2
   \- stdout EOF while cancelling -> exit 2
count 1 -> 2 : kill the child immediately and reap it            -> exit 2
```

The two waits are `SessionOptions::cancel_grace` and
`SessionOptions::shutdown_grace`; the CLI uses the defaults shown above.

Honouring a `Completed` event that arrives after a cancel request is
deliberate. Cancellation races task completion, and the worker's terminal event
is the only authoritative outcome; the CLI does not overwrite it.

A console Ctrl-C reaches the whole process group on both Windows and Unix-like
systems, so the worker, which installs no signal handler, will usually die
before it can read our `cancel` frame. That appears as stdout EOF with no
terminal event, which the fourth branch above maps to exit 2. This keeps the
cancellation exit code stable under either race.

Isolating the child with `CommandExt::process_group(0)` or
`CREATE_NEW_PROCESS_GROUP` so that only protocol frames can cancel it is
rejected for this slice. It requires two platform-specific code paths that are
hard to verify reliably in CI, and it changes no observable exit code or
message. Milestone 5 can revisit it if the desktop shell needs stricter
cancellation semantics.

## 8. Failure Handling

| Situation | Detection point | Handling |
| --- | --- | --- |
| Worker not found | all three probes fail | list every probed path |
| Spawn failure | `Command::spawn` | attach the OS error |
| First frame is not `ready`, or no output | handshake | attach the worker stderr tail |
| Protocol version mismatch | `DomainError::ProtocolVersion` | print expected and actual |
| Malformed frame or a line over `MAX_FRAME_BYTES` | `decode_line` / `FrameReader` | print the `DomainError`, then kill the child |
| Broken pipe writing `start` or `cancel` | `FrameWriter` | treat the worker as gone |
| stdout EOF with no terminal event and no cancel request | main loop | attach the child exit status and stderr tail |

The worker's stderr is drained continuously by its own thread into a ring
buffer holding the last twenty lines. Draining prevents a full pipe from
blocking the child, and bounding it prevents unbounded memory growth if the
worker becomes noisy.

`WorkerSession` implements `Drop` with an unconditional `kill()` followed by
`wait()`, ignoring the error from killing an already-exited process. No early
return, `?` propagation, or panic can leave a zombie child.

## 9. Dependencies

One new external dependency: `ctrlc`, declared with an exact version at the
workspace level and used only by `feathertalk-cli`. It is the established
cross-platform crate for this and handles the Windows console Ctrl-C event,
which a bare Unix-signal approach does not.

`clap` with the `derive` feature, `serde_json`, `thiserror`, `time`, and
`tempfile` are already workspace dependencies and are reused as-is.

## 10. Testing

`CARGO_BIN_EXE_*` is only injected for integration tests of the package that
defines the binary, so the CLI's tests cannot receive an official worker path.
Testing is therefore two-tiered.

### 10.1 Scripted Fake Worker

`feathertalk-client` declares a test-support binary:

```toml
[[bin]]
name = "feathertalk-fake-worker"
path = "tests/support/fake_worker.rs"
```

It reads `FT_FAKE_WORKER_SCENARIO` and emits a fixed frame sequence, covering:
a normal completion carrying a `result`; a `failed` event with every field
populated; a `cancelled` event; a first frame that is not `ready`; a `ready`
frame declaring protocol version 99; a `rejected` frame in the handshake
position; an empty `supported_commands`; an invalid JSON line; an oversized
line; an exit immediately after `ready`; ignoring `cancel` and hanging, to
exercise the cancel timeout with an injected shorter deadline; and answering
`cancel` with `completed`, to exercise the race rule.

The client's integration tests use `env!("CARGO_BIN_EXE_feathertalk-fake-worker")`,
so they need no skip logic and no platform branches.

The cost is that this binary is built with the workspace. It deliberately does
not use `required-features`, because that would stop `--all-targets` from
building it and break the `env!` lookup at compile time. The `fake` prefix in
the name carries the intent.

### 10.2 Real Worker End to End

`feathertalk-cli/tests/real_worker.rs` resolves both the real worker and the
fake worker as siblings of `env!("CARGO_BIN_EXE_feathertalk")`, since all
workspace binaries land in the same target directory. If a binary is missing,
the test skips with an explanatory note; setting `FEATHERTALK_REQUIRE_E2E=1`
turns a skip into a failure.

The authoritative gate is `cargo test --workspace --all-targets`, which builds
every binary, so the siblings always exist there and nothing is skipped. A skip
can only occur while running `-p feathertalk-cli` alone, which keeps per-task
single-crate verification self-contained.

### 10.3 Test Inventory

`feathertalk-client`:

- Worker discovery precedence across all three sources and the failure message
  content.
- Handshake success and each of the five handshake failures.
- The capability gate for a supported and an unsupported command.
- All four `SessionOutcome` variants.
- The three cancellation branches plus the cancel/complete race.
- Events carrying a foreign `task_id` being ignored and counted.
- The stderr ring buffer keeping only the last twenty lines.
- `Drop` leaving no live child, asserted by an immediately returning `wait`.
- Every outbound frame validated before it is written.

`feathertalk-cli`:

- Argument parsing, including the `--json` and `--quiet` conflict.
- An invalid `--task-id` exiting 3.
- The stage dictionary covering all thirteen `TaskStage` variants.
- Default-mode stdout containing only the `result` JSON.
- `--json` stdout matching the worker's bytes line for line.
- One case per exit code.
- Real `validate-project` against a temporary project directory, and
  `capabilities`.

No test requires a real FFmpeg installation. The `probe-media` path is covered
by pointing `FEATHERTALK_WORKER_FFPROBE` at a nonexistent path and asserting
that the command disappears from `supported_commands` and that the CLI exits 3
while naming that variable.

## 11. Verification

Per plan task, from `rust/`:

```text
cargo test -p feathertalk-client --all-targets
cargo test -p feathertalk-cli --all-targets
cargo clippy -p feathertalk-client --all-targets -- -D warnings
cargo clippy -p feathertalk-cli --all-targets -- -D warnings
cargo fmt --all -- --check
```

Only the crates a task actually touches need the per-crate commands.

At the end of the slice, the repository's five standard commands must pass:

```text
cargo test --workspace --all-targets
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
git diff --check
```

The workspace baseline is 707 passing, 0 failing, 13 ignored. This slice may
only increase the passing count.

## 12. Recorded Tradeoffs

- The CLI is a protocol client rather than a direct caller of the execution
  crates. It costs one process boundary per invocation and buys a single source
  of truth for behaviour shared with the future desktop shell.
- Path validation is not duplicated in the CLI, so an invalid path produces a
  worker error code instead of a fast local message.
- Process-group isolation of the child is deferred; the EOF-while-cancelling
  rule covers the observable behaviour instead.
- A test-support binary ships in the workspace build to keep protocol failure
  tests deterministic and free of external tools.

## 13. Out of Scope

- Every command the worker does not enable, including training, rendering,
  frame and feature extraction, normalization, model import and export,
  inspection, legacy feature migration, and `LockAssetPackage`.
- WGPU enumeration and GPU execution.
- Concurrent or batched tasks in one CLI invocation.
- Worker supervision, restart policy, and task-history recovery.
- Shell completion generation and man page output.
- GPUI integration; that remains milestone 5.
