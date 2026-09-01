# normalize_media Worker Command Design

## 1. Scope

This slice adds the third worker command, `normalize_media`, and the CLI
subcommand that drives it. It is the next step of the preprocessing pipeline
after `probe_media`: one source file in, a 25 fps MP4 and a 16 kHz mono WAV out,
both verified and committed atomically.

The normalization work itself already exists and is fully tested in
`feathertalk-media` (`normalize_media_with_runner`). This slice is therefore not
about media processing; it is about the missing runtime boundary that a
long-running command needs:

- A way for a command to report intermediate stages while it runs. Today
  `execute` returns only a terminal `CommandOutcome`, so the worker can emit
  nothing between `preparing` and the terminal event. Every command after this
  one (frame extraction, feature extraction, training, rendering) needs the same
  channel, so it is built here rather than improvised later.
- A phase hook in `feathertalk-media`'s normalization so the worker can observe
  the pipeline it delegates to instead of re-implementing its orchestration.
- The result payload, capability handshake entry, rejection text, and CLI
  surface for the new command.

`lock_asset_package` stays out of scope: the wire request does not carry the
manifest inputs it needs. GPU work, frame extraction, feature extraction,
training, rendering, and model import/export remain later slices.

## 2. Progress Reporting Through the Runtime

A new trait in `feathertalk-worker` is the only way a command reports progress:

```rust
pub trait TaskReporter {
    fn report(&self, stage: TaskStage, progress: Option<Progress>);
}

/// For direct callers and tests that do not observe progress.
pub struct NoReporter;
```

`execute` and `execute_with_runner` gain a `&dyn TaskReporter` parameter, and
`JobExecutor` becomes:

```rust
pub type JobExecutor = Box<
    dyn Fn(&Request, Option<&MediaToolchain>, &CancellationToken, &dyn TaskReporter)
        -> CommandOutcome
        + Send
        + 'static,
>;
```

The execution thread builds one reporter per job. It holds the job's `TaskId`
and a clone of the control channel sender, so `report` does exactly what the
terminal event already does: build an `Event`, attach the progress, and send
`ControlMessage::Emit` to the control loop. The control loop stays the sole
owner of the writer and of task lifecycle, and `emit` keeps its existing
guarantees:

- A task whose lifecycle is already terminal silently drops the event, so a
  progress event that races a cancel cannot appear after `cancelled`.
- `TaskLifecycle::advance` rejects only a return to `Queued` or a transition out
  of a terminal stage, so the stage sequence this command emits is legal.

A dropped control channel is ignored by `report`. Losing a progress event when
the runtime is already shutting down is not worth failing a task over.

## 3. Phase Hook in feathertalk-media

`normalize_media_with_runner` is one blocking call around five internal phases.
The crate gains an observing variant; the existing two functions stay and
delegate to it with a no-op observer, so no current caller changes.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizePhase {
    Probing,
    NormalizingVideo,
    NormalizingAudio,
    Verifying,
    Committing,
}

pub fn normalize_media_observed<R: ProcessRunner + ?Sized>(
    input: &ValidatedInput,
    spec: &NormalizationSpec,
    toolchain: &MediaToolchain,
    runner: &R,
    observer: &dyn Fn(NormalizePhase),
) -> Result<NormalizedMedia, MediaError>;
```

Each phase is reported immediately before its work starts, so a caller that
prints the phase names describes what is running, not what has finished. The
observer is infallible and returns nothing: it is an output channel, never a
control point, so it cannot make normalization fail or stop.

Layout validation happens before `Probing` is reported. A spec that names an
unusable output directory fails before any phase is announced, which keeps a
failed task from claiming it started work it never did.

## 4. Phase to Stage Mapping

`TaskStage` is a closed protocol enum, and protocol version 2 has no variant
that names media normalization. The worker maps phases onto the stages that are
accurate for the two passes which dominate wall time, and reports a
three-step progress count:

| `NormalizePhase`   | Emitted stage       | Progress |
| ------------------ | ------------------- | -------- |
| `Probing`          | `Preparing`         | 1/3      |
| `NormalizingVideo` | `ExtractingFrames`  | 2/3      |
| `NormalizingAudio` | `ExtractingAudio`   | 3/3      |
| `Verifying`        | no event            | —        |
| `Committing`       | no event            | —        |

The last two phases are two `ffprobe` calls on the freshly written outputs plus
a rename. They are bounded and short, and giving them a stage would mean either
inventing a protocol variant or moving the stage label backwards to `preparing`,
which reads as a bug in a progress display. They stay observable in the media
crate for callers with a richer UI; the wire simply ends at 3/3 and then reports
the terminal event.

The runtime's own `preparing` event at dispatch is unchanged, so a client sees
`preparing` (no progress), then `preparing 1/3`, `extracting_frames 2/3`,
`extracting_audio 3/3`, then `completed`.

## 5. The Normalization Spec Is Fixed, Not Configurable

`NormalizeMediaParams` carries only `input` and `output_dir`.
`validate_normalization` rejects any target other than 25 fps, 16 kHz, and one
channel, so the worker builds the spec from those constants:

```rust
NormalizationSpec {
    target_video_fps: 25,
    target_audio_sample_rate: 16_000,
    target_audio_channels: 1,
    output_dir: params.output_dir.clone(),
}
```

No new environment variable is introduced. The asset manifest contract pins the
same three values, so a configurable target would only create a way to produce
assets the rest of the pipeline rejects.

## 6. Execution and Cancellation

The command arm mirrors `probe_media`: it requires a media toolchain, calls
`validate_input` on the source, then runs the observing normalization with the
cancellable runner the runtime supplies.

Cancellation keeps working exactly as it does for `probe_media`, because it is
enforced where the time is actually spent: `CancellableProcessRunner` kills the
running `ffmpeg`/`ffprobe` child and returns `MediaError::ToolCancelled`, which
the existing `media_failure` helper turns into `CommandOutcome::Cancelled`. The
temporary outputs are removed by `TempOutput`'s drop guard, so a cancelled task
leaves the output directory as it was.

A cancel that arrives after the commit succeeded is reported as `completed`, not
`cancelled`. The artifacts exist and are valid at that point, and claiming
otherwise would leave a client deleting files it was told were never written.
The client already handles a completion that follows a cancel request.

## 7. Result Payload

`normalize_to_json(&NormalizedMedia) -> serde_json::Value`, alongside the
existing `probe_to_json`:

```json
{
  "output_dir": "<canonical directory>",
  "video": {
    "path": "<canonical file>",
    "bytes": 123456,
    "sha256": "<64 hex>",
    "codec_name": "mpeg4",
    "pixel_format": "yuv420p",
    "width": 640,
    "height": 480,
    "frame_rate": { "numerator": 25, "denominator": 1 },
    "frame_count": 50,
    "duration_seconds": 2.0
  },
  "audio": {
    "path": "<canonical file>",
    "bytes": 64000,
    "sha256": "<64 hex>",
    "codec_name": "pcm_s16le",
    "sample_format": "s16",
    "sample_rate": 16000,
    "channels": 1,
    "sample_count": 32000,
    "duration_seconds": 2.0
  },
  "source": { "format": {}, "video": {}, "audio": {} }
}
```

Unlike `probe_media`, the payload carries paths: the client asked for a
directory and the worker decides the file names, so without them the caller has
to guess. Paths are the canonical ones the media crate committed to, which on
Windows means the verbatim `\\?\` form. They are reported as produced rather
than prettified, because they are what a later task must open.

`source` reuses the `probe_to_json` shape for the original file, so a caller
that already parses a probe result parses this too.

`video` and `audio` are objects for every success the media crate can produce.
The serialiser still tolerates absence and writes `null` instead of panicking,
so a future library change degrades the payload rather than crashing a worker.

## 8. Handshake and Rejection Text

`supported_commands` reports `NormalizeMedia` under the same condition as
`ProbeMedia`: a resolved media toolchain. Both commands shell out to the same
two binaries, so a worker that cannot probe cannot normalize either.

`unsupported_reason` grows to cover the new command. The toolchain-specific
message (which names `FEATHERTALK_WORKER_FFPROBE` and
`FEATHERTALK_WORKER_FFMPEG`) applies to `normalize_media` as well, and the
generic fallback stops hard-coding the old two-command list, so the text cannot
drift from `supported_commands` as later commands land.

## 9. CLI Surface

A new subcommand:

```text
feathertalk normalize-media <INPUT> <OUTPUT_DIR>
```

It follows the existing pattern exactly: empty paths are refused in Chinese
before the worker is spawned, everything else is the worker's judgement, the
result goes to stdout (pretty JSON, or raw frames under `--json`), progress goes
to stderr, and the exit code is the existing four-way mapping.

The `UnsupportedCommand` hint in `render.rs` currently fires only for
`probe_media`; it applies to `normalize_media` too, since the cause and the fix
are identical.

## 10. Error Mapping

No new mapping. `media_task_error` already covers every `MediaError` variant
normalization can return, including `MissingStream`,
`NormalizationVerificationFailed`, `OutputDirectoryInvalid`,
`OutputConflictsWithInput`, `OutputInsideInput`, and the IO variants that select
`DISK_SPACE_LOW`. A source without both a video and an audio stream fails with
the media crate's own `MissingStream`, which is the honest reason.

## 11. Testing

- `feathertalk-media`: the observer reports the five phases in order for a
  successful run, reports nothing after the phase that fails, and reports no
  phase at all when layout validation fails. The existing scripted
  `FakeRunner` supplies the probe outputs and stages the output files.
- `feathertalk-worker` commands: a scripted five-call run produces
  `Completed` with the documented payload; the reporter receives exactly the
  three mapped events in order; a missing toolchain, a failing pass, and a
  cancelled pass produce the expected outcomes; the request is rejected when the
  source lacks a stream.
- `feathertalk-worker` handshake: `normalize_media` is present with a toolchain
  and absent without one.
- `feathertalk-worker` runtime: an executor that reports two progress events
  yields those events on the wire, in order, between `preparing` and the
  terminal event; a task cancelled mid-run does not emit progress after
  `cancelled`.
- `feathertalk-cli`: `normalize-media` builds the right request, refuses empty
  arguments in Chinese, renders a progress event with its Chinese stage label,
  and prints the result on stdout. The fake worker gains a scenario that emits
  the three progress events followed by a completed frame with a normalization
  result.
- End to end against the real worker, gated the same way as the existing e2e
  tests, on a short generated clip.

## 12. Verification

`cargo check`, `cargo test --workspace --all-targets`,
`cargo clippy --workspace --all-targets -- -D warnings`,
`cargo fmt --all -- --check`, and `git diff --check` all clean.

## 13. Out of Scope

GPU adapters and WGPU, frame extraction, face detection, feature extraction,
training, rendering, model import/export/inspection, legacy feature migration,
`lock_asset_package`, desktop supervision, and GPUI. Metrics on progress events
(`samples_per_second`, `eta_seconds`, `vram_bytes`) stay empty: normalization
has no meaningful sample rate, and a fabricated ETA is worse than none.
