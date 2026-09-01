# feathertalk-worker Runtime Design

## 1. Scope

This is milestone slice 2: the worker process and task runtime. It depends on
`feathertalk-domain` for the JSON Lines protocol, task vocabulary, lifecycle,
and event/error shapes.

The slice implements the full runtime boundary even though the command set is
intentionally small:

- One long-lived worker process communicating over stdin/stdout.
- Frame validation before dispatch.
- A task queue and one-at-a-time execution for the reported CPU adapter.
- Cancellation of queued and running tasks.
- Hard cancellation of the external `ffprobe` process.
- Adapter mutual exclusion keyed by stable adapter IDs.
- Graceful shutdown.
- Mapping existing execution errors to the ten protocol error codes.

The only enabled commands are `ValidateProject` and `ProbeMedia`. Every other
request variant is reported as unsupported and rejected without executing a
partial fallback.

Real WGPU enumeration and GPU execution remain later slices. This slice reports
`Backend::Cpu` and one CPU adapter, but implements the adapter lock table so
later GPU work can reuse the scheduling boundary.

## 2. Architecture

`feathertalk-worker` is a standalone binary. It uses standard library threads
and channels rather than an async runtime.

```text
stdin
  |
  v
[input thread] --validated ClientFrame--> [control thread]
                                             |
                                             v
                                    [task queue / state / adapter locks]
                                             |
                                             v
                                    [execution worker]
                                             |
                                             v
                                  [validated ServerFrame writer] --> stdout
```

Responsibilities are deliberately separated:

- The input thread only reads bounded frames. It decodes a `ClientFrame`, then
  calls `ClientFrame::validate()` before sending it onward.
- The control thread is the sole owner of queue state, task lifecycle, cancel
  tokens, adapter occupancy, and the output writer. It never performs a task's
  computation.
- The execution worker dequeues tasks, emits `Preparing`, acquires the task's
  adapter, runs the command, and reports the terminal event. It does not decide
  whether an incoming command is supported; that is fixed by the `Ready` frame.
- Event writing serializes a `ServerFrame`, calls `ServerFrame::validate()`,
  and writes exactly one newline-terminated compact JSON object to stdout.

The process writes `Ready` before reading any client frame. The first client
frame with an unsupported protocol version receives a `Rejected` frame with an
actionable reason and is never executed.

## 3. Handshake and Command Capability

Slice 1 reported only broad product capabilities. That cannot distinguish a
worker that supports `validate_project` and `probe_media` from one that supports
`lock_asset_package`, so slice 2 extends the handshake.

`ReadyFrame` gains a required `supported_commands: Vec<TaskKind>` field. Its
JSON values are the existing task slugs:

```json
"supported_commands": ["validate_project", "probe_media"]
```

`ReadyFrame::validate()` requires the list to be non-empty and contain no
duplicate task kinds.

The protocol version becomes `2`. This is a breaking wire-format change because
`deny_unknown_fields` causes old readers to reject the new required field.
Golden frame tests must be updated with the exact new Ready and Completed Event
text.

Broad product capabilities remain:

```json
"capabilities": {
  "training": false,
  "wgpu_training": false,
  "onnx_validation": false,
  "ffmpeg": true
}
```

`ffmpeg` is true only when a valid media toolchain is configured. It does not
imply that normalization or rendering is enabled.

For this worker, `supported_commands` is:

```text
ValidateProject
ProbeMedia
```

An unsupported `Start` frame receives `Rejected`. The reason names the task kind
and states that the worker does not support the command in this build. No task
is created, no queued event is emitted, and the command is not partially
executed.

`LockAssetPackage` remains disabled because the wire request carries only a
project directory, while `feathertalk_project::lock_asset_package` also requires
a complete `AssetManifest`. Enabling it without a command-level API or protocol
extension would either create a partial fallback or redefine locking as an
idempotent rewrite of an already locked package.

## 4. Event Results

Slice 1's `Event` could report progress and terminal status, but had no way to
return command output. `ProbeMedia` needs to return duration, frame rate,
resolution, and audio stream metadata to the desktop.

`Event` gains a `result` field. Like `progress` and `error`, it is present in
the serialized object and encoded as JSON `null` when there is no result:

```text
result: Option<serde_json::Value>
```

Validation rules:

- `result` is allowed only when `stage` is `Completed`.
- `result` must be `None` for every non-completed stage.
- When present, `result` must be a JSON object.
- A command defines its own result shape; `supported_commands` restricts which
  shapes can occur in this worker.

`ValidateProject` reports `result = None`; its `Completed` event is the result.

`ProbeMedia` reports a JSON object mirroring `feathertalk_media::MediaProbe`:

```text
MediaProbeResult {
  format: { format_name, duration_seconds },
  video: Option<{
    codec_name, pixel_format, width, height,
    frame_rate: { numerator, denominator },
    frame_count, duration_seconds
  }>,
  audio: Option<{
    codec_name, sample_format, sample_rate,
    channels, sample_count, duration_seconds
  }>
}
```

The conversion is implemented in the worker, not in `feathertalk-media`. The
existing media type is a validated internal value, not a wire contract; keeping
the mapping in the worker lets the JSON shape evolve with the protocol without
forcing a broad media crate release.

The result object does not contain the local input path. The client already
sent the path in `ProbeMediaParams.input`.

## 5. Task Queue, Cancellation, and Adapter Locks

Task IDs are unique per worker process. A second `Start` with an existing ID is
rejected and does not replace or mutate the original task.

Lifecycle events are:

```text
Queued -> Preparing -> Completed | Failed | Cancelled
```

- `Start` creates a queued task and emits `Queued`.
- The execution worker emits `Preparing` immediately before executing the
  command.
- Success emits `Completed`, including the command result if one is defined.
- Failure emits `Failed { code, message }` with a complete `TaskError`.
- User cancellation emits `Cancelled`.

Every task owns a cancellation token.

When a queued task is cancelled, it transitions directly to `Cancelled` and is
never started.

When `ValidateProject` is already running, the worker checks its token before
starting and immediately after the existing project API returns. If cancellation
was requested before the terminal event is emitted, the worker reports
`Cancelled` and discards the validation result. The existing project API has no
internal cancellation hook, so this is a safe-boundary check rather than an
interrupt.

When `ProbeMedia` is already running, the process execution layer polls the
token and kills the child `ffprobe` process. Cancellation is therefore hard for
the external tool, not merely advisory.

`Cancel` is idempotent:

- Cancelling an unknown task is silently accepted.
- Cancelling a terminal task is silently accepted.
- Repeated cancellation has the same effect as a single request.
- Each task emits at most one `Cancelled` event.
- User cancellation never emits a `Failed` event with `TASK_CANCELLED`;
  `TASK_CANCELLED` is reserved for command-line adapters that need an error
  result instead of a protocol event.

The runtime keeps an adapter occupancy table keyed by adapter ID. A task
acquires its adapter immediately before execution and releases it after the
terminal event is queued. The CPU adapter uses the stable ID `cpu-0`. The
runtime must reject internal attempts to acquire an occupied adapter rather
than allowing a second concurrent task.

The queue is serial for this slice because there is one CPU adapter. Tests also
exercise two synthetic adapters to prove that the lock table independently
enforces one task per adapter ID.

Later GPU slices will reuse the same runtime and lock table. They will add real
WGPU enumeration and map a request to a reported adapter ID; they will not need
to redesign queueing or cancellation.

## 6. Shutdown

`Shutdown` means stop and save:

1. Stop accepting new `Start` frames.
2. Cancel tasks that have not started.
3. Request the currently running task to stop at its next safe boundary.
4. Allow the task to save recoverable state.
5. Emit any remaining terminal event, flush stdout, and exit with status zero.

`ValidateProject` and `ProbeMedia` have no persistent training checkpoint in
this slice. Their safe boundary is command completion or cancellation. The
runtime still follows the full shutdown sequence so later training commands can
plug in checkpoint saving without changing the process semantics.

When stdin reaches EOF, the worker performs the same shutdown sequence. It does
not abort the process while a task is running.

## 7. Configuration

Configuration is provided by environment variables:

```text
FEATHERTALK_WORKER_FFPROBE
FEATHERTALK_WORKER_FFMPEG
FEATHERTALK_WORKER_MEDIA_TIMEOUT_MS
```

- Both media tool paths must be absolute and non-empty.
- The timeout must be a positive integer; the default is five minutes.
- The values must construct a valid `feathertalk_media::MediaToolchain`.

If media configuration is invalid or incomplete, the worker still starts and
reports:

```text
capabilities.ffmpeg = false
supported_commands = ["validate_project"]
```

It does not fail the whole process, because project validation remains useful
and a desktop supervisor can show the reduced capability state.

`ProbeMedia` is excluded from `supported_commands` whenever the media
configuration is rejected; it is never advertised as supported and then failed
at request time.

The worker version is taken from `CARGO_PKG_VERSION`.

## 8. Cancellable Process Execution

`feathertalk-media` currently provides a strict process runner with timeout,
bounded output capture, and regular-file executable validation, but no
cancellation token. Slice 2 extends its public execution boundary with a
cancellable variant while preserving the existing APIs.

The media crate gains a cancellation flag and a cancellable system runner. The
runner is constructed with the flag, implements the existing `ProcessRunner`
trait, and can be passed to the existing `probe_media_with_runner` entry point.
The current `SystemProcessRunner` and non-cancellable APIs remain unchanged.

The cancellable runner:

- Reuses the existing absolute-path and regular-file executable validation.
- Reuses the existing bounded stdout/stderr capture.
- Reuses the existing timeout behavior.
- Polls a shared cancellation token while waiting for the child process.
- Kills and waits for the child when cancellation is observed.
- Joins output reader threads and releases their resources.
- Returns a distinct cancellation result to the command boundary.

The worker treats that result as user cancellation and emits `Cancelled`; it
does not map a killed process to `WORKER_CRASHED`.

Tests use a fake runner for command-level behavior and a test helper process for
actual child termination. They do not depend on a real FFmpeg installation.

## 9. Error Mapping

`ValidateProject` maps:

- Invalid manifest or asset state, unsupported schema version, invalid field,
  unsafe relative path, symlink, invalid filesystem entry, empty artifact, and
  locked-state mismatch to `MEDIA_INVALID`.
- Disk-space-related I/O errors to `DISK_SPACE_LOW`.
- Unexpected failures to `WORKER_CRASHED`.

`ProbeMedia` maps:

- Missing input, non-regular input, symlink, invalid toolchain field, probe
  output too large, invalid probe JSON, probe contract violation, missing or
  duplicate stream to `MEDIA_INVALID`.
- Tool spawn failure, tool failure, timeout, and unexpected failures to
  `WORKER_CRASHED`.
- Cancellation to the `Cancelled` terminal stage, never to `TASK_CANCELLED`.

Failed events in this slice use `TaskError::stage = TaskStage::Preparing`
because the enabled commands fail during preparation or command execution, and
the error model forbids terminal stages in `TaskError`.

The mapper must:

- Provide a non-empty user-readable summary.
- Bound the summary and detail to the protocol limits.
- Preserve useful technical detail without exposing a raw command line as the
  user-facing summary.
- Use the recovery action already associated with each `ErrorCode`.

## 10. Testing

Tests are split by boundary and do not depend on a real FFmpeg executable.

`feathertalk-domain` tests:

- Protocol version 2.
- `supported_commands` required, non-empty, and duplicate-free.
- `Event.result` allowed only on `Completed`.
- `Event.result` must be an object when present.
- Golden wire text for the updated `Ready` and completed `Event` frames.

`feathertalk-worker` tests:

- `Ready` is the first output frame.
- Correct CPU backend and adapter reporting.
- The exact command support matrix for valid and invalid media configuration.
- Unsupported command rejection without task creation.
- Unsupported protocol version rejection.
- Queue ordering.
- Duplicate task ID rejection.
- Cancellation of a queued task before execution.
- Cancellation of a running project validation task.
- Hard cancellation of a running external media process.
- Unknown and terminal-task cancellation idempotency.
- One `Cancelled` event per task.
- Adapter mutual exclusion with the CPU adapter and two synthetic adapters.
- Shutdown cancels queued work and waits for the active task.
- stdin EOF triggers the same shutdown path.
- Error mapping for all `ProjectError` and `MediaError` variants used by the
  enabled commands.
- Result conversion from `MediaProbe` to the JSON wire object.
- Every emitted event passes `ServerFrame::validate()`.

Process tests use the existing test-helper pattern: spawn the test binary in an
ignored helper mode, then exercise normal exit, non-zero exit, timeout, bounded
output, and cancellation.

## 11. Verification

The repository's five standard commands must pass from `rust/`:

```text
cargo test --workspace --all-targets
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
git diff --check
```

## 12. Out of Scope

- WGPU enumeration or GPU execution.
- Training, rendering, frame extraction, feature extraction, normalization,
  model import/export, inspection, or legacy feature migration.
- `LockAssetPackage` until either the wire request carries its manifest inputs
  or a command-level API computes them safely.
- Desktop supervision, process restart policy, and task-history recovery.
- GPUI integration.
- CLI parity; that is slice 3.
