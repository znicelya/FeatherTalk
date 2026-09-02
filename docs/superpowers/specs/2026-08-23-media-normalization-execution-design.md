# Media Normalization Execution and Metadata Design

Date: 2026-08-23
Status: Approved for implementation

## 1. Goal

Extend `feathertalk-media` with a safe execution layer that probes source media
with bundled `ffprobe`, normalizes it with bundled `ffmpeg`, and returns verified
metadata for the fixed FeatherTalk media outputs.

This bounded milestone-two slice completes source probing and creation of
`video_25fps.mp4` plus `audio_16k_mono.wav`; it does not extract frames or run
face/audio models.

## 2. Scope

Included:

- Strict, bounded parsing of selected `ffprobe` JSON fields.
- Exactly one source video stream and exactly one source audio stream.
- Safe argv construction without a shell.
- Bounded subprocess output and timeout handling.
- Fixed 25 FPS video normalization using FFmpeg's built-in `mpeg4` encoder.
- Fixed 16 kHz mono PCM S16LE WAV extraction.
- Temporary outputs, file synchronization, post-write probing, SHA-256, and
  rollback-capable two-file commit.
- Injectable process and filesystem operations for deterministic tests.

Excluded:

- Frame/JPEG extraction, SCRFD, PFLD, blur analysis, FeatherHuBERT, manifest
  locking, worker RPC, cancellation protocol, rendering, and package bundling.
- Shell command strings, PATH lookup, arbitrary codec/filter arguments, or
  user-provided FFmpeg flags.
- Synthetic silence for videos without audio. Missing audio is an explicit
  source-contract failure.

## 3. Public Value Types

```rust
pub struct MediaToolchain { /* absolute bundled executable paths + timeout */ }
pub struct FrameRate { /* numerator + denominator */ }
pub struct ProbeFormat { /* format name + finite duration */ }
pub struct VideoMetadata { /* codec, pixel format, dimensions, rate, counts */ }
pub struct AudioMetadata { /* codec, sample format, rate, channels, count */ }
pub struct MediaProbe { /* format + exactly one video and audio */ }
pub struct MediaArtifact { /* byte count + SHA-256 */ }
pub struct NormalizedMedia { /* layout, probes, and artifact hashes */ }
```

All fields are private. Public immutable accessors expose values. Types that
contain `f64` derive `PartialEq` but not `Eq`.

`MediaToolchain::new` requires absolute non-empty paths and
`0 < timeout <= 24 hours`. The real runner additionally requires each path to
resolve to a regular non-symlink file, preventing PATH ambiguity and bundle
tampering.

## 4. Public Operations

```rust
pub fn probe_media(
    input: &ValidatedInput,
    toolchain: &MediaToolchain,
) -> Result<MediaProbe, MediaError>;

pub fn normalize_media(
    input: &ValidatedInput,
    spec: &NormalizationSpec,
    toolchain: &MediaToolchain,
) -> Result<NormalizedMedia, MediaError>;
```

The crate also exposes `CommandSpec`, `ProcessOutput`, `ProcessRunner`, and
`SystemProcessRunner`. `probe_media_with_runner`, `probe_video_with_runner`, and
`normalize_media_with_runner` accept a caller-supplied runner. These are test
and future worker seams; they cannot change fixed command arguments.
`probe_video_with_runner` runs the same argv but expects one video stream and no
audio, which is the shape of the `video_25fps.mp4` this crate writes; frame
extraction consumes that artifact and cannot use the audio/video entry point.

## 5. Probe Contract

The fixed `ffprobe` argv is equivalent to:

```text
-v error -count_frames
-show_entries format=format_name,duration:stream=codec_type,codec_name,width,height,pix_fmt,avg_frame_rate,r_frame_rate,nb_read_frames,duration,sample_fmt,sample_rate,channels,duration_ts,time_base
-of json <canonical input>
```

The `-show_entries` value is one argv element. JSON is limited to 1 MiB before
parsing. Unknown fields outside the selected contract are ignored because
FFprobe versions emit additional metadata; every selected value is validated.

Source probing requires exactly one video and exactly one audio stream. Internal
post-normalization probing uses the same parser with expectations for video-only
or audio-only output.

Validation rejects invalid UTF-8/JSON, missing or duplicate streams, missing or
zero dimensions/rates/channels, malformed ratios, `N/A`, fractional integers,
non-finite values, durations outside `0..=86_400` seconds, dimensions above
16,384, and frame/sample counts above `100_000_000_000`. Counts use FFprobe's
`nb_read_frames` when present, otherwise checked rounded duration-derived values.

## 6. Normalization Commands

Video invocation:

```text
-hide_banner -nostdin -y -v error -i <source>
-map 0:v:0 -an -sn -dn -map_metadata -1 -map_chapters -1
-vf fps=25 -fps_mode cfr -c:v mpeg4 -q:v 2 -pix_fmt yuv420p -f mp4 <temp video>
```

Audio invocation:

```text
-hide_banner -nostdin -y -v error -i <source>
-map 0:a:0 -vn -sn -dn -map_metadata -1 -map_chapters -1
-ac 1 -ar 16000 -c:a pcm_s16le -f wav <temp audio>
```

The built-in `mpeg4` encoder avoids GPL `libx264` in the standard commercial
FFmpeg build. Codec patent/license review remains a packaging milestone.
Paths are single `OsString` argv elements; no shell or quoting is used.

## 7. Process Execution

`SystemProcessRunner` validates absolute regular non-symlink executables, spawns
with null stdin and piped stdout/stderr, drains both pipes concurrently, retains
at most 1 MiB per stream, polls `try_wait`, and kills/reaps on timeout. Non-zero
exit, timeout, spawn failure, and output overflow map to stable error variants.

## 8. Output Verification and Atomicity

Temporary files are unique paths in the canonical output directory. After each
FFmpeg run, the file is required to be regular, non-symlink, and non-empty;
`sync_all` is called; the output is probed and must match:

- video: codec `mpeg4`, pixel format `yuv420p`, exact 25/1 FPS, positive size,
  no audio;
- audio: codec `pcm_s16le`, sample format `s16`, 16,000 Hz, one channel, no video.

Video/audio duration difference must be <=20 ms. Both files are streamed for
SHA-256 and byte counts. Existing destinations remain untouched until both pass.
The commit state machine backs up existing regular files, renames both new files,
then deletes backups. Any failure reverses completed renames and reports primary
and rollback errors. Guards remove only invocation-owned temporary/backup paths.

## 9. Errors, Limits, and Acceptance

Extend `MediaError` with toolchain, tool execution, probe, verification, commit,
and rollback categories. Captured tool text is bounded and never exposed as a
constructed command line. Timeout and media duration are at most 24 hours.

Tests must cover parser boundaries, exact argv, hostile native paths, process
failure/timeout/overflow, output verification, hash/byte reporting, temporary
cleanup, and second-rename rollback. Real FFmpeg is optional in CI; deterministic
fake-runner tests are mandatory. Formatting, clippy, focused tests, workspace
tests, and diff checks must pass before integration.
