# extract_features Worker Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `extract_features` to the worker and the CLI: read the project's normalised 16 kHz mono wav, run FeatherHuBERT over it on the CPU in chunks, publish an even-token `assets/features/feather_hubert.f32`, and report per-chunk progress that a cancel can interrupt.

**Architecture:** `feathertalk-audio` is missing exactly one piece, a reader for the single audio shape the encoder accepts; chunk planning, stitching, the odd-token trim, and the no-clobber writer already exist and are already tested. The worker composes them: it resolves the FeatherHuBERT package directory from one new environment variable, loads the encoder through `feathertalk-export`'s package loader, wraps it in a `ChunkEncoder` that reports progress and polls the cancellation token at every chunk boundary, and writes exactly one file. Nothing touches `assets.json`, the asset lock, or `quality.json`: a feature commit needs a frame count, a frame size, and a landmark model hash, none of which this command's two parameters can supply.

**Tech Stack:** Rust 2024, 1.94.0 toolchain, `serde_json`, `sha2`, `clap 4.5` (derive), `tempfile`, burn 0.21 on the NdArray CPU backend through `feathertalk-models`, `thiserror`. No async runtime.

**Design:** `docs/superpowers/specs/2026-09-03-extract-features-worker-command-design.md`

## Global Constraints

- Run every cargo command from `E:/workspace/github/FeatherTalk/rust`; run git from `E:/workspace/github/FeatherTalk`.
- The wav reader admits exactly one shape: RIFF/WAVE, format code 1, 1 channel, 16 000 Hz, 16 bits. No resampling, no channel mixing, no fallback decoder. A file that needs any of those did not come from our own `normalize_media`.
- The token contract is fixed and its order is not negotiable: `read_wav_16k_mono` then `normalize_waveform` then `extract_long_audio(&normalized, &mut encoder, DEFAULT_CHUNK_SAMPLES)` then `drop_odd_token` then `write_feature_file_no_clobber`.
- No `force` flag. `ExtractFeaturesParams` is `deny_unknown_fields`, so a new field is a wire-protocol change and belongs to a protocol slice, not this one.
- Never call `FeatherHubertConfig::default()`. Its 512/2/12/1024/0.05 does not match the shipped 256/2/8/1024/0.0; all five hyperparameters come from the package manifest's `configuration`.
- The command writes exactly one file, `assets/features/feather_hubert.f32`, and touches neither `assets.json`, nor the asset lock, nor `quality.json`.
- User-facing strings are Chinese; code comments, doc comments, and diagnostics are English.
- Every source file stays free of a BOM and uses LF endings.
- `serde_json` is built without `preserve_order`; never re-serialise a frame the worker or CLI received.
- Progress events carry no metrics: `Metrics::empty()` stays untouched. `completed` counts encoded chunks, never samples, tokens, or percentages.
- Cancellation is cooperative. The token is polled once after admission and once before every chunk; there is no child process to kill.
- No new binary fixture enters git. The end-to-end test cuts its audio at runtime from the already-tracked `demo/feathertalk_demo_latest_188.mp4` and builds its model package into a `TempDir`.
- Do not touch `demo/kanghui_training_video_featherhubert_188_latest/`; it must stay untracked.
- Commit after each task. Stage explicit paths, never `git add .`. Never push to `origin`.
- Every task leaves the tree green: the task's own test command plus `cargo check` must pass before its commit.
- The final gate for the whole slice: `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace --all-targets`, the gated release end-to-end, `git diff --check`.

## File Structure

- `rust/crates/feathertalk-audio/src/wav.rs` (new) — the RIFF chunk walk and the `i16` to `f32` scaling. One responsibility: turn a path into samples or into a precise refusal.
- `rust/crates/feathertalk-audio/src/error.rs` — twelve wav variants, and later the cancellation variant.
- `rust/crates/feathertalk-audio/src/lib.rs` — module declaration and re-exports.
- `rust/crates/feathertalk-audio/tests/wav.rs` (new) — byte-built fixtures for the reader.
- `rust/crates/feathertalk-export/src/{lib,package}.rs` — `read_package_manifest`, factored out of `load_model_package` so a caller can read a package's shipped hyperparameters instead of hardcoding them.
- `rust/crates/feathertalk-export/tests/package.rs` — coverage for the extracted reader.
- `rust/crates/feathertalk-worker/Cargo.toml` — `feathertalk-audio` and `feathertalk-export` as dependencies, `hex` and `sha2` as dev-dependencies for the package fixture.
- `rust/crates/feathertalk-worker/src/config.rs` — `FeatureToolchain` and the one new environment variable.
- `rust/crates/feathertalk-worker/src/error_map.rs` — audio and package error mapping.
- `rust/crates/feathertalk-worker/src/feature_result.rs` (new) — the completed-result payload.
- `rust/crates/feathertalk-worker/src/admission.rs` (new) — the shared project-directory check and the `MediaInvalid` constructor, moved out of `extract_frames.rs`.
- `rust/crates/feathertalk-worker/src/extract_frames.rs` — imports those two helpers instead of owning them. No behaviour change.
- `rust/crates/feathertalk-worker/src/extract_features.rs` (new) — admission, the progress bridge, and the token contract.
- `rust/crates/feathertalk-worker/src/features.rs` (new) — load the FeatherHuBERT package into a burn encoder.
- `rust/crates/feathertalk-worker/src/commands.rs` — the new request arm.
- `rust/crates/feathertalk-worker/src/handshake.rs` — advertise the command once the model directory resolves.
- `rust/crates/feathertalk-worker/src/runtime.rs` — the rejection text for the unsupported case.
- `rust/crates/feathertalk-worker/src/lib.rs` — module declarations and re-exports.
- `rust/crates/feathertalk-worker/tests/{config,error_mapping,feature_result,extract_features,features,commands,handshake,runtime}.rs` — configuration, mapping, result, command, capability, and wire coverage.
- `rust/crates/feathertalk-cli/src/{cli,run,render}.rs` — the subcommand, its request, and the unsupported-command advice.
- `rust/crates/feathertalk-cli/tests/{cli,real_worker}.rs` — CLI behaviour and end-to-end coverage.

---

### Task 1: Read 16 kHz mono wav files

**Files:**
- Modify: `rust/crates/feathertalk-audio/src/error.rs`
- Modify: `rust/crates/feathertalk-audio/src/lib.rs`
- Create: `rust/crates/feathertalk-audio/src/wav.rs`
- Test: `rust/crates/feathertalk-audio/tests/wav.rs`

**Interfaces:**
- Consumes: `AudioError` from `crates/feathertalk-audio/src/error.rs`, and the `symlink_metadata` then size-cap then `read_to_end` template already used by `read_feature_file` in `crates/feathertalk-audio/src/format.rs`.
- Produces: `read_wav_16k_mono(path: impl AsRef<Path>) -> Result<Vec<f32>, AudioError>`, `pub const MAX_WAV_FILE_BYTES: u64`, `pub const WAV_SAMPLE_RATE: u32`, and twelve new `AudioError` variants (`WavIo`, `WavNotRegular`, `WavTooLarge`, `InvalidRiffHeader`, `InvalidWavHeader`, `MissingWavChunk`, `UnsupportedWavFormat`, `UnsupportedWavChannels`, `UnsupportedWavSampleRate`, `UnsupportedWavBitDepth`, `WavPayloadTruncated`, `EmptyWav`). Task 3 maps every one of those variants onto a wire code; Task 6 calls `read_wav_16k_mono`.

**Why first:** nothing in the workspace can read a wav file. `feathertalk-audio` plans chunks, stitches encoder output, and writes the feature format, but the samples have always arrived from a test fixture. The Python reference (`data_utils/feather_hubert/feather_hubert.py`) leans on `soundfile`, which decodes dozens of formats; we need exactly one, because `normalize_media` already wrote it: `-ac 1 -ar 16000 -c:a pcm_s16le`. So this is a validating reader, not a decoder — about 150 lines of header walk with no new dependency — and it comes first because every later task consumes either its samples or its error variants.

- [ ] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-audio/tests/wav.rs`. `tempfile` is already a regular dependency of this crate, so its `Cargo.toml` does not change. The crate's test style is `.unwrap()` over `expect`, no per-test doc comments, and few fat tests rather than many thin ones.

```rust
//! The reader admits exactly one shape of file: 16 kHz, mono, 16-bit PCM.
//!
//! Every fixture here is assembled byte by byte, because what is under test is
//! the header walk, not a library's idea of a well-formed file.

use std::path::PathBuf;

use feathertalk_audio::{AudioError, MAX_WAV_FILE_BYTES, WAV_SAMPLE_RATE, read_wav_16k_mono};

fn fmt_body() -> Vec<u8> {
    format_body(1, 1, 16_000, 32_000, 2, 16)
}

fn format_body(
    code: u16,
    channels: u16,
    sample_rate: u32,
    byte_rate: u32,
    block_align: u16,
    bits: u16,
) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&code.to_le_bytes());
    body.extend_from_slice(&channels.to_le_bytes());
    body.extend_from_slice(&sample_rate.to_le_bytes());
    body.extend_from_slice(&byte_rate.to_le_bytes());
    body.extend_from_slice(&block_align.to_le_bytes());
    body.extend_from_slice(&bits.to_le_bytes());
    body
}

fn pcm(samples: &[i16]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect()
}

fn riff(chunks: &[(&str, Vec<u8>)]) -> Vec<u8> {
    let mut payload = b"WAVE".to_vec();
    for (id, body) in chunks {
        payload.extend_from_slice(id.as_bytes());
        payload.extend_from_slice(&(body.len() as u32).to_le_bytes());
        payload.extend_from_slice(body);
        if !body.len().is_multiple_of(2) {
            payload.push(0);
        }
    }
    let mut bytes = b"RIFF".to_vec();
    bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&payload);
    bytes
}

fn wav_file(bytes: &[u8]) -> (tempfile::TempDir, PathBuf) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("audio_16k_mono.wav");
    std::fs::write(&path, bytes).unwrap();
    (directory, path)
}

#[test]
fn canonical_wav_scales_samples_and_skips_unknown_chunks() {
    assert_eq!(MAX_WAV_FILE_BYTES, 256 * 1024 * 1024);
    assert_eq!(WAV_SAMPLE_RATE, 16_000);

    // The odd-length LIST body exercises the pad byte every RIFF chunk carries
    // when its size is odd.
    let bytes = riff(&[
        ("LIST", b"abc".to_vec()),
        ("fmt ", fmt_body()),
        ("data", pcm(&[0, 32_767, -32_768, -16_384])),
    ]);
    let (_directory, path) = wav_file(&bytes);

    let samples = read_wav_16k_mono(&path).unwrap();

    assert_eq!(samples, vec![0.0, 32_767.0 / 32_768.0, -1.0, -0.5]);
}

#[test]
fn a_longer_format_chunk_keeps_its_extension_out_of_the_way() {
    // An 18-byte fmt chunk -- the 16 PCM fields plus a zero `cbSize` -- is what
    // several encoders write, so the reader must read the fields it knows and
    // step over the rest.
    let mut body = fmt_body();
    body.extend_from_slice(&0_u16.to_le_bytes());
    let bytes = riff(&[("fmt ", body), ("data", pcm(&[1_024]))]);
    let (_directory, path) = wav_file(&bytes);

    let samples = read_wav_16k_mono(&path).unwrap();

    assert_eq!(samples, vec![0.031_25]);
}

#[test]
fn a_broken_container_is_rejected_before_any_chunk() {
    let valid = riff(&[("fmt ", fmt_body()), ("data", pcm(&[1]))]);

    let (_short_directory, short) = wav_file(b"RIFF");
    let error = read_wav_16k_mono(&short).unwrap_err();
    assert!(matches!(error, AudioError::InvalidRiffHeader), "{error:?}");

    let mut wrong_magic = valid.clone();
    wrong_magic[..4].copy_from_slice(b"RIFX");
    let (_magic_directory, magic) = wav_file(&wrong_magic);
    let error = read_wav_16k_mono(&magic).unwrap_err();
    assert!(matches!(error, AudioError::InvalidRiffHeader), "{error:?}");

    let mut wrong_form = valid.clone();
    wrong_form[8..12].copy_from_slice(b"AVI ");
    let (_form_directory, form) = wav_file(&wrong_form);
    let error = read_wav_16k_mono(&form).unwrap_err();
    assert!(matches!(error, AudioError::InvalidRiffHeader), "{error:?}");

    let mut trailing = valid.clone();
    trailing.extend_from_slice(b"data");
    let (_trailing_directory, path) = wav_file(&trailing);
    let error = read_wav_16k_mono(&path).unwrap_err();
    let AudioError::InvalidWavHeader { reason } = error else {
        panic!("expected an invalid header, got {error:?}");
    };
    assert!(reason.contains("truncated"), "{reason}");
}

#[test]
fn both_chunks_are_required_and_the_format_must_come_first() {
    let (_format_directory, format_only) = wav_file(&riff(&[("fmt ", fmt_body())]));
    let error = read_wav_16k_mono(&format_only).unwrap_err();
    assert!(
        matches!(error, AudioError::MissingWavChunk { chunk: "data" }),
        "{error:?}"
    );

    let (_data_directory, data_only) = wav_file(&riff(&[("data", pcm(&[1]))]));
    let error = read_wav_16k_mono(&data_only).unwrap_err();
    assert!(
        matches!(error, AudioError::MissingWavChunk { chunk: "fmt " }),
        "{error:?}"
    );

    let late = riff(&[("data", pcm(&[1])), ("fmt ", fmt_body())]);
    let (_late_directory, path) = wav_file(&late);
    let error = read_wav_16k_mono(&path).unwrap_err();
    let AudioError::InvalidWavHeader { reason } = error else {
        panic!("expected an invalid header, got {error:?}");
    };
    assert!(reason.contains("follows the data chunk"), "{reason}");
}

#[test]
fn every_unsupported_format_field_names_itself() {
    // `AudioError` has no `PartialEq`, so the expected variant is compared
    // through its `Display` text.
    let cases = [
        (
            format_body(3, 1, 16_000, 64_000, 4, 32),
            AudioError::UnsupportedWavFormat { code: 3 },
        ),
        (
            format_body(1, 2, 16_000, 64_000, 4, 16),
            AudioError::UnsupportedWavChannels { actual: 2 },
        ),
        (
            format_body(1, 1, 44_100, 88_200, 2, 16),
            AudioError::UnsupportedWavSampleRate {
                actual: 44_100,
                expected: 16_000,
            },
        ),
        (
            format_body(1, 1, 16_000, 48_000, 3, 24),
            AudioError::UnsupportedWavBitDepth { actual: 24 },
        ),
    ];

    for (body, expected) in cases {
        let (_directory, path) = wav_file(&riff(&[("fmt ", body), ("data", pcm(&[1]))]));
        let error = read_wav_16k_mono(&path).unwrap_err();
        assert_eq!(error.to_string(), expected.to_string());
    }
}

#[test]
fn an_inconsistent_format_chunk_is_rejected_with_a_reason() {
    let cases = [
        (format_body(1, 1, 16_000, 32_000, 4, 16), "block align 4"),
        (format_body(1, 1, 16_000, 16_000, 2, 16), "byte rate 16000"),
        (fmt_body()[..14].to_vec(), "14 bytes"),
    ];

    for (body, fragment) in cases {
        let (_directory, path) = wav_file(&riff(&[("fmt ", body), ("data", pcm(&[1]))]));
        let error = read_wav_16k_mono(&path).unwrap_err();
        let AudioError::InvalidWavHeader { reason } = error else {
            panic!("expected an invalid header, got {error:?}");
        };
        assert!(reason.contains(fragment), "{reason}");
    }
}

#[test]
fn an_unusable_data_chunk_is_rejected() {
    let empty = riff(&[("fmt ", fmt_body()), ("data", Vec::new())]);
    let (_empty_directory, path) = wav_file(&empty);
    let error = read_wav_16k_mono(&path).unwrap_err();
    assert!(matches!(error, AudioError::EmptyWav), "{error:?}");

    let odd = riff(&[("fmt ", fmt_body()), ("data", vec![1, 2, 3])]);
    let (_odd_directory, path) = wav_file(&odd);
    let error = read_wav_16k_mono(&path).unwrap_err();
    let AudioError::InvalidWavHeader { reason } = error else {
        panic!("expected an invalid header, got {error:?}");
    };
    assert!(reason.contains("3 bytes"), "{reason}");

    let mut truncated = riff(&[("fmt ", fmt_body()), ("data", pcm(&[1, 2, 3, 4]))]);
    truncated.truncate(truncated.len() - 4);
    let (_truncated_directory, path) = wav_file(&truncated);
    let error = read_wav_16k_mono(&path).unwrap_err();
    let AudioError::WavPayloadTruncated { expected, actual } = error else {
        panic!("expected a truncated payload, got {error:?}");
    };
    assert_eq!((expected, actual), (8, 4));
}

#[test]
fn a_path_that_is_not_a_regular_file_is_rejected_before_the_read() {
    let directory = tempfile::tempdir().unwrap();

    let error = read_wav_16k_mono(directory.path()).unwrap_err();
    assert!(
        matches!(error, AudioError::WavNotRegular { .. }),
        "{error:?}"
    );

    let error = read_wav_16k_mono(directory.path().join("missing.wav")).unwrap_err();
    let AudioError::WavIo { operation, .. } = error else {
        panic!("expected an I/O failure, got {error:?}");
    };
    assert_eq!(operation, "metadata");
}
```

- [ ] **Step 2: Run the test and watch it fail**

Run: `cargo test -p feathertalk-audio --test wav`

Expected: FAIL, and not a single assertion runs. `error[E0432]: unresolved imports` names `feathertalk_audio::MAX_WAV_FILE_BYTES`, `feathertalk_audio::WAV_SAMPLE_RATE`, and `feathertalk_audio::read_wav_16k_mono`, followed by `error[E0599]` for each `AudioError` variant the tests pattern-match.

- [ ] **Step 3: Implement**

In `rust/crates/feathertalk-audio/src/error.rs`, insert the twelve variants at the top of `pub enum AudioError`, immediately above `#[error("waveform is empty")] EmptyWaveform`. The wav variants come first because they are the earliest failure a task can hit; the existing order otherwise follows the pipeline.

```rust
    #[error("wav I/O error during {operation} at {path}: {source}")]
    WavIo {
        operation: &'static str,
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("wav file is not a regular non-symlink file: {path}")]
    WavNotRegular { path: std::path::PathBuf },
    #[error("wav file exceeds {limit} bytes: {actual}")]
    WavTooLarge { limit: u64, actual: u64 },
    #[error("wav file is not a RIFF/WAVE container")]
    InvalidRiffHeader,
    #[error("wav header is invalid: {reason}")]
    InvalidWavHeader { reason: String },
    #[error("wav file is missing the {chunk:?} chunk")]
    MissingWavChunk { chunk: &'static str },
    #[error("unsupported wav format code {code}, expected 16-bit PCM")]
    UnsupportedWavFormat { code: u16 },
    #[error("unsupported wav channel count {actual}, expected mono")]
    UnsupportedWavChannels { actual: u16 },
    #[error("unsupported wav sample rate {actual}, expected {expected}")]
    UnsupportedWavSampleRate { actual: u32, expected: u32 },
    #[error("unsupported wav bit depth {actual}, expected 16")]
    UnsupportedWavBitDepth { actual: u16 },
    #[error("wav payload is truncated: expected {expected} bytes, got {actual}")]
    WavPayloadTruncated { expected: u64, actual: u64 },
    #[error("wav file has no samples")]
    EmptyWav,
```

Create `rust/crates/feathertalk-audio/src/wav.rs`.

```rust
//! A reader for the one audio shape the feature extractor accepts.
//!
//! FeatherHuBERT consumes 16 kHz mono waveforms, and `normalize_media` already
//! writes exactly that file with `-ac 1 -ar 16000 -c:a pcm_s16le`. So this
//! module validates rather than converts: it walks the RIFF chunk list, refuses
//! anything that is not 16-bit PCM mono at 16 kHz, and scales the samples into
//! `f32`. There is no resampler and no channel mixer, because a file that needs
//! one did not come from our own normalisation step.

use std::{
    fs::{self, File},
    io::Read,
    path::Path,
};

use crate::AudioError;

/// The largest wav file the reader will open, 256 MiB.
///
/// That is roughly two hours of 16 kHz mono 16-bit audio. The whole file is
/// read into memory before the header walk, so this bound is what caps the
/// allocation a truncated or hostile file can ask for.
pub const MAX_WAV_FILE_BYTES: u64 = 256 * 1024 * 1024;

/// The only sample rate the reader accepts.
pub const WAV_SAMPLE_RATE: u32 = 16_000;

/// `RIFF`, the declared size, and `WAVE`.
const RIFF_HEADER_BYTES: usize = 12;

/// A chunk identifier plus its declared size.
const CHUNK_HEADER_BYTES: usize = 8;

/// The 16 bytes every PCM `fmt ` chunk carries. Encoders may append more.
const MIN_FMT_BODY_BYTES: usize = 16;

/// The `wFormatTag` value for uncompressed PCM.
const WAVE_FORMAT_PCM: u16 = 1;

/// One 16-bit sample per frame.
const BLOCK_ALIGN: u16 = 2;

/// 16 kHz times two bytes per sample.
const BYTE_RATE: u32 = WAV_SAMPLE_RATE * 2;

/// Read a 16 kHz mono 16-bit wav file into `f32` samples in `[-1.0, 1.0]`.
///
/// The scaling divides by `32_768.0`, which is what the Python reference in
/// `data_utils/feather_hubert/feather_hubert.py` gets from `soundfile`, so
/// features extracted here and there start from identical numbers.
pub fn read_wav_16k_mono(path: impl AsRef<Path>) -> Result<Vec<f32>, AudioError> {
    let path = path.as_ref();
    let bytes = read_bounded(path)?;
    parse(&bytes)
}

/// Read the whole file once it is known to be a regular file within the bound.
fn read_bounded(path: &Path) -> Result<Vec<u8>, AudioError> {
    let metadata = fs::symlink_metadata(path).map_err(|source| io("metadata", path, source))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(AudioError::WavNotRegular {
            path: path.to_owned(),
        });
    }
    if metadata.len() > MAX_WAV_FILE_BYTES {
        return Err(AudioError::WavTooLarge {
            limit: MAX_WAV_FILE_BYTES,
            actual: metadata.len(),
        });
    }

    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    File::open(path)
        .map_err(|source| io("open", path, source))?
        .read_to_end(&mut bytes)
        .map_err(|source| io("read", path, source))?;
    Ok(bytes)
}

/// Walk the chunk list, then decode the payload the walk found.
fn parse(bytes: &[u8]) -> Result<Vec<f32>, AudioError> {
    if bytes.len() < RIFF_HEADER_BYTES
        || &bytes[..4] != b"RIFF"
        || &bytes[8..RIFF_HEADER_BYTES] != b"WAVE"
    {
        return Err(AudioError::InvalidRiffHeader);
    }

    let mut offset = RIFF_HEADER_BYTES;
    let mut format_seen = false;
    let mut data: Option<&[u8]> = None;
    while offset < bytes.len() {
        if bytes.len() - offset < CHUNK_HEADER_BYTES {
            return Err(AudioError::InvalidWavHeader {
                reason: format!("the chunk header at offset {offset} is truncated"),
            });
        }
        let id = &bytes[offset..offset + 4];
        let size = u32::from_le_bytes(
            bytes[offset + 4..offset + CHUNK_HEADER_BYTES]
                .try_into()
                .unwrap(),
        ) as usize;
        let body = offset + CHUNK_HEADER_BYTES;
        let available = bytes.len() - body;
        match id {
            b"fmt " => {
                if data.is_some() {
                    return Err(AudioError::InvalidWavHeader {
                        reason: "the fmt chunk follows the data chunk".to_owned(),
                    });
                }
                if available < size {
                    return Err(AudioError::InvalidWavHeader {
                        reason: format!(
                            "the fmt chunk declares {size} bytes, {available} are present"
                        ),
                    });
                }
                check_format(&bytes[body..body + size])?;
                format_seen = true;
            }
            b"data" => {
                if !size.is_multiple_of(2) {
                    return Err(AudioError::InvalidWavHeader {
                        reason: format!(
                            "the data chunk holds {size} bytes, which is not a whole number of 16-bit samples"
                        ),
                    });
                }
                if available < size {
                    return Err(AudioError::WavPayloadTruncated {
                        expected: size as u64,
                        actual: available as u64,
                    });
                }
                data = Some(&bytes[body..body + size]);
            }
            _ => {}
        }
        // Every RIFF chunk is padded to an even length, and the pad byte is not
        // counted by the declared size.
        offset = body + size + size % 2;
    }

    if !format_seen {
        return Err(AudioError::MissingWavChunk { chunk: "fmt " });
    }
    let Some(data) = data else {
        return Err(AudioError::MissingWavChunk { chunk: "data" });
    };
    if data.is_empty() {
        return Err(AudioError::EmptyWav);
    }

    Ok(data
        .chunks_exact(2)
        .map(|pair| f32::from(i16::from_le_bytes(pair.try_into().unwrap())) / 32_768.0)
        .collect())
}

/// Refuse every `fmt ` chunk that is not 16 kHz mono 16-bit PCM.
fn check_format(body: &[u8]) -> Result<(), AudioError> {
    if body.len() < MIN_FMT_BODY_BYTES {
        return Err(AudioError::InvalidWavHeader {
            reason: format!(
                "the fmt chunk is {} bytes, expected at least {MIN_FMT_BODY_BYTES}",
                body.len()
            ),
        });
    }

    let code = u16::from_le_bytes(body[..2].try_into().unwrap());
    if code != WAVE_FORMAT_PCM {
        return Err(AudioError::UnsupportedWavFormat { code });
    }
    let channels = u16::from_le_bytes(body[2..4].try_into().unwrap());
    if channels != 1 {
        return Err(AudioError::UnsupportedWavChannels { actual: channels });
    }
    let sample_rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
    if sample_rate != WAV_SAMPLE_RATE {
        return Err(AudioError::UnsupportedWavSampleRate {
            actual: sample_rate,
            expected: WAV_SAMPLE_RATE,
        });
    }

    let byte_rate = u32::from_le_bytes(body[8..12].try_into().unwrap());
    let block_align = u16::from_le_bytes(body[12..14].try_into().unwrap());
    let bits = u16::from_le_bytes(body[14..16].try_into().unwrap());
    // The bit depth is checked before the block align: a 24-bit file carries a
    // block align that is consistent with its own depth, and naming the depth
    // is the more useful diagnostic.
    if bits != 16 {
        return Err(AudioError::UnsupportedWavBitDepth { actual: bits });
    }
    if block_align != BLOCK_ALIGN {
        return Err(AudioError::InvalidWavHeader {
            reason: format!("block align {block_align} does not match 16-bit mono"),
        });
    }
    if byte_rate != BYTE_RATE {
        return Err(AudioError::InvalidWavHeader {
            reason: format!("byte rate {byte_rate} does not match 16 kHz 16-bit mono"),
        });
    }

    Ok(())
}

/// Name the operation an I/O failure came from.
fn io(operation: &'static str, path: &Path, source: std::io::Error) -> AudioError {
    AudioError::WavIo {
        operation,
        path: path.to_owned(),
        source,
    }
}
```

In `rust/crates/feathertalk-audio/src/lib.rs`, declare the module after `mod stitch;` and re-export after the `stitch` re-export, keeping both lists alphabetical.

```rust
mod wav;
```

```rust
pub use wav::{MAX_WAV_FILE_BYTES, WAV_SAMPLE_RATE, read_wav_16k_mono};
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `cargo test -p feathertalk-audio --test wav`, expecting 8 passed. Then `cargo test -p feathertalk-audio` for the rest of the crate: the new variants sit at the top of a non-`PartialEq` enum that nothing matches exhaustively yet, so chunking, commit, format, normalisation, and stitching must all stay green.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-audio/src/error.rs rust/crates/feathertalk-audio/src/lib.rs rust/crates/feathertalk-audio/src/wav.rs rust/crates/feathertalk-audio/tests/wav.rs
git commit -m "feat(audio): read 16 kHz mono wav files"
```

---

### Task 2: Resolve the FeatherHuBERT model directory

**Files:**
- Modify: `rust/crates/feathertalk-worker/src/config.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs`
- Test: `rust/crates/feathertalk-worker/tests/config.rs`

**Interfaces:**
- Consumes: `WorkerConfig`, the private `required_path`, and the `ENV_*` constants that already live in `config.rs`. Nothing from Task 1.
- Produces: `pub const ENV_HUBERT_DIR: &str`, `FeatureToolchain::hubert_dir(&self) -> &Path`, `WorkerConfig::features(&self) -> Option<&FeatureToolchain>`, `WorkerConfig::feature_rejection(&self) -> Option<&str>`, and `WorkerConfig::from_values_with_toolchains(ffprobe: Option<String>, ffmpeg: Option<String>, timeout_ms: Option<String>, scrfd_dir: Option<String>, pfld_dir: Option<String>, hubert_dir: Option<String>) -> Self`. Task 8 consumes `FeatureToolchain`, Tasks 9 and 10 consume `features()`, Task 10 consumes `feature_rejection()`.

**Why:** The command needs exactly one new piece of configuration, and it has to resolve independently of the other two toolchains. A worker with no ffmpeg and no SCRFD directory must still advertise `extract_features`, because feature extraction never shells out and never loads a face model -- it reads a wav that an earlier `normalize_media` already wrote. That independence is the property Task 10's handshake depends on, so it is pinned by a test before the field exists. This task is first among the worker tasks because every later worker task needs a way to say "the FeatherHuBERT directory is configured".

- [ ] **Step 1: Write the failing test**

In `rust/crates/feathertalk-worker/tests/config.rs`, widen the import and add a helper directly after `absolute` so the six-argument constructor is never spelled out inline (rustfmt wraps a six-argument call differently depending on the argument lengths, and the helper keeps every call site one line):

```rust
use feathertalk_worker::{ENV_HUBERT_DIR, WorkerConfig};
```

```rust
fn with_hubert(hubert_dir: Option<String>) -> WorkerConfig {
    WorkerConfig::from_values_with_toolchains(None, None, None, None, None, hubert_dir)
}
```

Then append the three tests. The existing four tests stay untouched.

```rust
#[test]
fn an_absolute_directory_resolves_the_feature_toolchain() {
    let hubert = absolute("feather_hubert_188");
    let config = with_hubert(Some(hubert.clone()));

    let features = config.features().expect("the directory is absolute");
    assert_eq!(features.hubert_dir(), PathBuf::from(&hubert));
    assert_eq!(config.feature_rejection(), None);
    // An unresolved media or model toolchain must not block the feature one:
    // extract_features shells out to nothing and loads neither SCRFD nor PFLD.
    assert!(config.media().is_none());
    assert!(config.models().is_none());
}

#[test]
fn a_relative_hubert_directory_is_rejected_with_the_variable_name() {
    assert_eq!(ENV_HUBERT_DIR, "FEATHERTALK_WORKER_HUBERT_DIR");
    let config = with_hubert(Some("models/hubert".to_owned()));

    assert!(config.features().is_none());
    let rejection = config.feature_rejection().expect("a reason is kept");
    assert!(rejection.contains(ENV_HUBERT_DIR), "{rejection}");
    assert!(
        rejection.contains("must be an absolute path"),
        "{rejection}"
    );
}

#[test]
fn the_feature_toolchain_is_resolved_independently_of_the_models() {
    // The five-argument constructor keeps its meaning: it configures media and
    // models and leaves extract_features unsupported.
    let config = WorkerConfig::from_values_with_models(
        Some(absolute("ffprobe-test")),
        Some(absolute("ffmpeg-test")),
        None,
        Some(absolute("scrfd_2_5g")),
        Some(absolute("pfld_ghost_one")),
    );

    assert!(config.media().is_some());
    assert!(config.models().is_some());
    assert!(config.features().is_none());
    let rejection = config.feature_rejection().expect("a reason is kept");
    assert!(rejection.contains("is not set"), "{rejection}");
}
```

- [ ] **Step 2: Run the test and watch it fail**

Run: `cargo test -p feathertalk-worker --test config`

Expected: FAIL to compile, with `error[E0432]: unresolved import feathertalk_worker::ENV_HUBERT_DIR`, `error[E0599]: no function or associated item named from_values_with_toolchains found for struct WorkerConfig`, and `error[E0599]: no method named features found for struct WorkerConfig`.

- [ ] **Step 3: Implement**

In `rust/crates/feathertalk-worker/src/config.rs`, add the constant after `ENV_PFLD_DIR`:

```rust
pub const ENV_HUBERT_DIR: &str = "FEATHERTALK_WORKER_HUBERT_DIR";
```

Add the toolchain type after the `ModelToolchain` impl block. One directory, one getter; the doc comment says why the contents are not validated here, mirroring `ModelToolchain`:

```rust
/// Where the worker finds the FeatherHuBERT model package.
///
/// Only the shape of the path is checked here, for the same reason as
/// `ModelToolchain`: a directory can disappear between startup and the first
/// job, so the manifest and the weights are validated when a job loads them.
#[derive(Debug, Clone)]
pub struct FeatureToolchain {
    hubert_dir: PathBuf,
}

impl FeatureToolchain {
    pub fn hubert_dir(&self) -> &Path {
        &self.hubert_dir
    }
}
```

Give `WorkerConfig` the two new fields, after `model_rejection`:

```rust
    features: Option<FeatureToolchain>,
    feature_rejection: Option<String>,
```

Point `from_env` at the new constructor:

```rust
    pub fn from_env() -> Self {
        Self::from_values_with_toolchains(
            std::env::var(ENV_FFPROBE).ok(),
            std::env::var(ENV_FFMPEG).ok(),
            std::env::var(ENV_MEDIA_TIMEOUT_MS).ok(),
            std::env::var(ENV_SCRFD_DIR).ok(),
            std::env::var(ENV_PFLD_DIR).ok(),
            std::env::var(ENV_HUBERT_DIR).ok(),
        )
    }
```

Demote `from_values_with_models` to a delegating form and document what it leaves out, the way `from_values` already documents the models it leaves out:

```rust
    /// The frame form: no FeatherHuBERT directory, so `extract_features` stays
    /// unsupported.
    pub fn from_values_with_models(
        ffprobe: Option<String>,
        ffmpeg: Option<String>,
        timeout_ms: Option<String>,
        scrfd_dir: Option<String>,
        pfld_dir: Option<String>,
    ) -> Self {
        Self::from_values_with_toolchains(ffprobe, ffmpeg, timeout_ms, scrfd_dir, pfld_dir, None)
    }
```

Add the widest constructor after it. The three `match` blocks are deliberately uniform: each toolchain either resolves or leaves a reason behind, and no failure short-circuits another.

```rust
    pub fn from_values_with_toolchains(
        ffprobe: Option<String>,
        ffmpeg: Option<String>,
        timeout_ms: Option<String>,
        scrfd_dir: Option<String>,
        pfld_dir: Option<String>,
        hubert_dir: Option<String>,
    ) -> Self {
        let (media, media_rejection) = match media_toolchain(ffprobe, ffmpeg, timeout_ms) {
            Ok(toolchain) => (Some(toolchain), None),
            Err(reason) => (None, Some(reason)),
        };
        let (models, model_rejection) = match model_toolchain(scrfd_dir, pfld_dir) {
            Ok(toolchain) => (Some(toolchain), None),
            Err(reason) => (None, Some(reason)),
        };
        let (features, feature_rejection) = match feature_toolchain(hubert_dir) {
            Ok(toolchain) => (Some(toolchain), None),
            Err(reason) => (None, Some(reason)),
        };
        Self {
            worker_version: env!("CARGO_PKG_VERSION").to_owned(),
            media,
            media_rejection,
            models,
            model_rejection,
            features,
            feature_rejection,
        }
    }
```

Add the two getters after `model_rejection`, mirroring it exactly:

```rust
    pub fn features(&self) -> Option<&FeatureToolchain> {
        self.features.as_ref()
    }

    pub fn feature_rejection(&self) -> Option<&str> {
        self.feature_rejection.as_deref()
    }
```

Add the free function after `model_toolchain`:

```rust
fn feature_toolchain(hubert_dir: Option<String>) -> Result<FeatureToolchain, String> {
    let hubert_dir = required_path(hubert_dir, ENV_HUBERT_DIR)?;
    Ok(FeatureToolchain { hubert_dir })
}
```

In `rust/crates/feathertalk-worker/src/lib.rs`, the `config` re-export becomes:

```rust
pub use config::{
    DEFAULT_MEDIA_TIMEOUT_MS, ENV_FFMPEG, ENV_FFPROBE, ENV_HUBERT_DIR, ENV_MEDIA_TIMEOUT_MS,
    ENV_PFLD_DIR, ENV_SCRFD_DIR, FeatureToolchain, ModelToolchain, WorkerConfig,
};
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `cargo test -p feathertalk-worker --test config`, expecting 7 passed. Then `cargo test -p feathertalk-worker` (roughly 35 to 50 seconds): `WorkerConfig` is only ever built through its constructors, so the two new fields must not disturb the command, handshake, or runtime suites.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/src/config.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/tests/config.rs
git commit -m "feat(worker): resolve the FeatherHuBERT model directory"
```

---

### Task 3: Map audio and package failures onto wire codes

**Files:**
- Modify: `rust/crates/feathertalk-worker/Cargo.toml`
- Modify: `rust/crates/feathertalk-audio/src/error.rs`
- Modify: `rust/crates/feathertalk-worker/src/error_map.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs`
- Test: `rust/crates/feathertalk-worker/tests/error_mapping.rs`

**Interfaces:**
- Consumes: `AudioError` including the twelve wav variants from Task 1, `PackageError` from `feathertalk-export`, `ENV_HUBERT_DIR` from Task 2, and the private `io_error_code`, `io_summary`, `clamp`, and `FAILURE_STAGE` already in `error_map.rs`.
- Produces: `audio_task_error(error: &AudioError) -> TaskError`, `is_audio_cancellation(error: &AudioError) -> bool`, `package_task_error(error: &PackageError) -> TaskError`, and one more `AudioError` variant, `Cancelled { operation: &'static str }`. Task 6 calls `audio_task_error` and `is_audio_cancellation` and constructs `AudioError::Cancelled`; Task 9 calls `package_task_error`.

**Why:** The two crates the command composes are not yet visible from the worker, and their errors have to reach the wire as codes a client can act on. Doing the mapping in its own task keeps Task 6 about the token contract instead of about 38 match arms, and it settles two decisions that would otherwise be made twice: a corrupt or wrong-shaped wav is `MediaInvalid` because the file is what has to change, while an unloadable model package is always `ModelIncompatible` -- even when the underlying failure is I/O -- because the request named no path, so the only thing a user can fix is the directory `FEATHERTALK_WORKER_HUBERT_DIR` points at. The cancellation variant belongs here too: `ChunkEncoder::encode` returns `Result<_, AudioError>`, so a cancelled chunk has nowhere else to go, and the mapper is what keeps it from being reported as a crash.

- [ ] **Step 1: Write the failing test**

In `rust/crates/feathertalk-worker/tests/error_mapping.rs`, add two imports and widen the worker one:

```rust
use feathertalk_audio::AudioError;
use feathertalk_export::PackageError;
```

```rust
use feathertalk_worker::{
    audio_task_error, is_audio_cancellation, is_media_cancellation, is_pipeline_cancellation,
    media_task_error, package_task_error, pipeline_task_error, project_task_error,
    quality_task_error,
};
```

Then append three tests. The table keeps the file's existing shape: one representative per code, and the loop asserts the whole payload rather than only the code, so a mapper that produces an invalid `TaskError` fails here rather than on the wire.

```rust
#[test]
fn audio_errors_map_onto_wire_codes() {
    let cases = vec![
        (AudioError::InvalidRiffHeader, ErrorCode::MediaInvalid),
        (
            AudioError::UnsupportedWavSampleRate {
                actual: 44_100,
                expected: 16_000,
            },
            ErrorCode::MediaInvalid,
        ),
        (AudioError::EmptyWav, ErrorCode::MediaInvalid),
        (AudioError::ConstantWaveform, ErrorCode::MediaInvalid),
        (
            AudioError::WavIo {
                operation: "read",
                path: path(),
                source: io_error(io::ErrorKind::StorageFull),
            },
            ErrorCode::DiskSpaceLow,
        ),
        (
            AudioError::WavIo {
                operation: "read",
                path: path(),
                source: io_error(io::ErrorKind::PermissionDenied),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            AudioError::InvalidFeatureDimension,
            ErrorCode::ModelIncompatible,
        ),
        (
            AudioError::FeatureShapeMismatch {
                frame_count: 4,
                tokens: 7,
                dims: 1024,
            },
            ErrorCode::FeatureShapeMismatch,
        ),
        (
            AudioError::TooManyChunks {
                actual: 2_000_000,
                limit: 1_000_000,
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            AudioError::Cancelled {
                operation: "extract_features",
            },
            ErrorCode::TaskCancelled,
        ),
    ];

    for (error, expected) in cases {
        let mapped = audio_task_error(&error);
        assert_eq!(mapped.code, expected, "{error:?}");
        assert_eq!(mapped.stage, TaskStage::Preparing, "{error:?}");
        assert_eq!(mapped.recovery, expected.default_recovery(), "{error:?}");
        assert!(!mapped.summary.trim().is_empty(), "{error:?}");
        mapped.validate().unwrap();
    }
}

#[test]
fn only_cancellation_is_audio_cancellation() {
    let cancelled = AudioError::Cancelled {
        operation: "extract_features",
    };

    assert!(is_audio_cancellation(&cancelled));
    assert!(!is_audio_cancellation(&AudioError::EmptyWav));
}

#[test]
fn a_package_failure_names_the_hubert_variable() {
    // Not an I/O failure, and still ModelIncompatible: the request carried no
    // path, so the directory is the only thing a user can act on.
    let error = PackageError::InvalidRequest("no manifest".to_owned());

    let mapped = package_task_error(&error);

    assert_eq!(mapped.code, ErrorCode::ModelIncompatible);
    assert_eq!(mapped.summary, "特征模型加载失败");
    assert_eq!(mapped.stage, TaskStage::Preparing);
    assert!(
        mapped.detail.contains("FEATHERTALK_WORKER_HUBERT_DIR"),
        "{}",
        mapped.detail
    );
    mapped.validate().unwrap();
}
```

- [ ] **Step 2: Run the test and watch it fail**

Run: `cargo test -p feathertalk-worker --test error_mapping`

Expected: FAIL to compile with `error[E0432]: unresolved import feathertalk_audio` and the same for `feathertalk_export` -- neither crate is a dependency of the worker yet -- plus `error[E0432]` for the three unresolved names in the `feathertalk_worker` import list.

- [ ] **Step 3: Implement**

In `rust/crates/feathertalk-worker/Cargo.toml`, add the two path dependencies in alphabetical order, so `feathertalk-audio` goes above `feathertalk-domain` and `feathertalk-export` below it. Neither introduces a cycle: `feathertalk-audio` depends only on `feathertalk-project` plus hashing and temp-file crates, and `feathertalk-export` depends only on burn, `feathertalk-models`, and `feathertalk-weights`, both of which the worker already pulls in.

```toml
feathertalk-audio = { path = "../feathertalk-audio" }
feathertalk-domain = { path = "../feathertalk-domain" }
feathertalk-export = { path = "../feathertalk-export" }
```

In `rust/crates/feathertalk-audio/src/error.rs`, add the cancellation variant at the very end of `pub enum AudioError`, after `StagingCollision`. It goes last because it is not a stage failure: it is the encoder reporting that the caller asked to stop.

```rust
    #[error("{operation} was cancelled")]
    Cancelled { operation: &'static str },
```

In `rust/crates/feathertalk-worker/src/error_map.rs`, extend the imports. `ENV_HUBERT_DIR` is a crate-local item, so it gets its own group after the external ones.

```rust
use std::io;

use feathertalk_audio::AudioError;
use feathertalk_domain::{ErrorCode, MAX_DETAIL_CHARS, TaskError, TaskStage};
use feathertalk_export::PackageError;
use feathertalk_frame_pipeline::{AnomalyCode, FrameAnomaly, PipelineError};
use feathertalk_media::MediaError;
use feathertalk_project::ProjectError;

use crate::ENV_HUBERT_DIR;
```

Add the three public functions after `quality_task_error`, keeping the file's convention that every public mapper is a thin wrapper over a code function and a summary function:

```rust
pub fn audio_task_error(error: &AudioError) -> TaskError {
    let code = audio_error_code(error);
    TaskError::new(
        code,
        audio_summary(error),
        &clamp(&error.to_string()),
        FAILURE_STAGE,
    )
}

pub fn is_audio_cancellation(error: &AudioError) -> bool {
    matches!(error, AudioError::Cancelled { .. })
}

/// Maps a model-package failure the feature command could not recover from.
///
/// Every variant reports `ModelIncompatible`, including `Io`: the request named
/// no path, so a missing or unreadable file under the package directory is a
/// misconfigured model directory, not a broken disk. The detail names the
/// variable that points at it.
pub fn package_task_error(error: &PackageError) -> TaskError {
    TaskError::new(
        ErrorCode::ModelIncompatible,
        "特征模型加载失败",
        &clamp(&package_detail(error)),
        FAILURE_STAGE,
    )
}
```

Add the private functions after `anomaly_summary` and before `io_error_code`, which keeps the file's order of public wrappers, then per-error private helpers, then the shared I/O helpers.

```rust
fn audio_error_code(error: &AudioError) -> ErrorCode {
    match error {
        // The audio or the feature file on disk is what has to change.
        AudioError::WavNotRegular { .. }
        | AudioError::WavTooLarge { .. }
        | AudioError::InvalidRiffHeader
        | AudioError::InvalidWavHeader { .. }
        | AudioError::MissingWavChunk { .. }
        | AudioError::UnsupportedWavFormat { .. }
        | AudioError::UnsupportedWavChannels { .. }
        | AudioError::UnsupportedWavSampleRate { .. }
        | AudioError::UnsupportedWavBitDepth { .. }
        | AudioError::WavPayloadTruncated { .. }
        | AudioError::EmptyWav
        | AudioError::EmptyWaveform
        | AudioError::NonFiniteWaveform { .. }
        | AudioError::ConstantWaveform
        | AudioError::FeatureNotRegular { .. }
        | AudioError::FeatureTooLarge { .. }
        | AudioError::InvalidFeatureMagic
        | AudioError::UnsupportedFeatureVersion { .. }
        | AudioError::FeatureHeaderTruncated { .. }
        | AudioError::FeaturePayloadTruncated { .. }
        | AudioError::FeatureTrailingBytes { .. }
        | AudioError::InvalidFeaturePayloadSize
        | AudioError::InvalidFeaturePairWidth { .. }
        | AudioError::LockedAssetMutation { .. } => ErrorCode::MediaInvalid,
        AudioError::WavIo { source, .. } | AudioError::FeatureIo { source, .. } => {
            io_error_code(source)
        }
        // A feature file whose width or values disagree with the encoder is a
        // model mismatch, not a bad request.
        AudioError::InvalidFeatureDimension
        | AudioError::FeatureLengthMismatch { .. }
        | AudioError::NonFiniteFeature { .. } => ErrorCode::ModelIncompatible,
        AudioError::FeatureShapeMismatch { .. } => ErrorCode::FeatureShapeMismatch,
        // The runtime intercepts cancellation before it reaches this mapper, so
        // this arm exists for the caller that maps an error without asking
        // `is_audio_cancellation` first.
        AudioError::Cancelled { .. } => ErrorCode::TaskCancelled,
        // Everything left is the worker's own machinery misbehaving.
        AudioError::InvalidChunkSize
        | AudioError::ArithmeticOverflow
        | AudioError::TooManyChunks { .. }
        | AudioError::FeatureSizeOverflow
        | AudioError::CommitFailed { .. }
        | AudioError::CommitRollbackFailed { .. }
        | AudioError::StagingCollision { .. } => ErrorCode::WorkerCrashed,
    }
}

fn audio_summary(error: &AudioError) -> &'static str {
    match error {
        AudioError::WavIo { source, .. } | AudioError::FeatureIo { source, .. } => {
            io_summary(source)
        }
        AudioError::WavNotRegular { .. } => "音频文件不是常规文件",
        AudioError::WavTooLarge { .. } => "音频文件过大",
        AudioError::InvalidRiffHeader => "音频文件不是有效的 WAV",
        AudioError::InvalidWavHeader { .. } => "WAV 头部字段无效",
        AudioError::MissingWavChunk { .. } => "WAV 缺少必需的数据块",
        AudioError::UnsupportedWavFormat { .. } => "WAV 编码格式不受支持，需要 16 位 PCM",
        AudioError::UnsupportedWavChannels { .. } => "音频必须是单声道",
        AudioError::UnsupportedWavSampleRate { .. } => "音频采样率必须是 16kHz",
        AudioError::UnsupportedWavBitDepth { .. } => "音频位深必须是 16 位",
        AudioError::WavPayloadTruncated { .. } => "WAV 数据被截断",
        AudioError::EmptyWav | AudioError::EmptyWaveform => "音频没有采样点",
        AudioError::NonFiniteWaveform { .. } => "音频包含非有限采样值",
        AudioError::ConstantWaveform => "音频是恒定值，无法归一化",
        AudioError::InvalidChunkSize
        | AudioError::ArithmeticOverflow
        | AudioError::TooManyChunks { .. } => "音频分块规划失败",
        AudioError::InvalidFeatureDimension | AudioError::FeatureLengthMismatch { .. } => {
            "特征维度与模型不一致"
        }
        AudioError::FeatureShapeMismatch { .. } => "特征长度与帧数不匹配",
        AudioError::NonFiniteFeature { .. } => "特征包含非有限值",
        AudioError::FeatureSizeOverflow => "特征文件尺寸溢出",
        AudioError::FeatureNotRegular { .. } => "特征文件不是常规文件",
        AudioError::FeatureTooLarge { .. } => "特征文件过大",
        AudioError::InvalidFeatureMagic | AudioError::UnsupportedFeatureVersion { .. } => {
            "特征文件格式不受支持"
        }
        AudioError::FeatureHeaderTruncated { .. }
        | AudioError::FeaturePayloadTruncated { .. }
        | AudioError::FeatureTrailingBytes { .. }
        | AudioError::InvalidFeaturePayloadSize
        | AudioError::InvalidFeaturePairWidth { .. } => "特征文件内容损坏",
        AudioError::LockedAssetMutation { .. } => "素材包已锁定，无法修改",
        AudioError::CommitFailed { .. } => "特征文件写入失败",
        AudioError::CommitRollbackFailed { .. } => "写入失败后回滚也失败",
        AudioError::StagingCollision { .. } => "暂存文件已存在",
        AudioError::Cancelled { .. } => "任务已取消",
    }
}

/// The package loader's message plus the variable a user has to fix.
fn package_detail(error: &PackageError) -> String {
    format!("{error} (check {ENV_HUBERT_DIR})")
}
```

Both matches are exhaustive with no `_` arm, on purpose: a new `AudioError` variant must not silently become `WorkerCrashed`. In `rust/crates/feathertalk-worker/src/lib.rs`, the `error_map` re-export becomes:

```rust
pub use error_map::{
    audio_task_error, is_audio_cancellation, is_media_cancellation, is_pipeline_cancellation,
    media_task_error, package_task_error, pipeline_task_error, project_task_error,
    quality_task_error,
};
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `cargo test -p feathertalk-worker --test error_mapping`, then `cargo test -p feathertalk-worker`, then `cargo test -p feathertalk-audio` for the new enum variant. `AudioError` is matched exhaustively nowhere inside `feathertalk-audio`, so the audio suite must stay green without a change.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-audio/src/error.rs rust/crates/feathertalk-worker/Cargo.toml rust/crates/feathertalk-worker/src/error_map.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/tests/error_mapping.rs
git commit -m "feat(worker): map audio and package failures onto wire codes"
```

---

### Task 4: Shape the feature result payload

**Files:**
- Create: `rust/crates/feathertalk-worker/src/feature_result.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs`
- Test: `rust/crates/feathertalk-worker/tests/feature_result.rs`

**Interfaces:**
- Consumes: `FeatureArtifact` from `feathertalk-audio`, which the worker can see since Task 3.
- Produces: `feature_to_json(output_dir: &Path, artifact: &FeatureArtifact, model_sha256: &str) -> serde_json::Value`. Task 6 calls it once, on the only success path.

**Why:** The wire payload is the part of this command a client actually consumes, and it is worth pinning on its own before the command that produces it exists. Two decisions live here. The payload reports `frame_count` as `tokens / 2` rather than carrying the token count alone, because every downstream consumer counts video frames, not FeatherHuBERT tokens, and the division is only correct because the odd token was already dropped. And it reports two digests: `sha256` identifies the file that was written, `model_sha256` identifies the encoder that produced it, which is the only way a later run can tell whether a cached feature file is still valid. The per-token values never appear: one JSON line per task has to stay small.

- [ ] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-worker/tests/feature_result.rs`. `FeatureArtifact` has no public constructor, so the fixture writes a real feature file into a `TempDir` -- which is the better fixture anyway, because the byte count and the digest in the payload are exactly the fields that must not drift.

```rust
use std::path::Path;

use feathertalk_audio::{FeatureArtifact, FeatureMatrix, write_feature_file_no_clobber};
use feathertalk_worker::feature_to_json;

fn artifact(directory: &Path) -> FeatureArtifact {
    let path = directory.join("features").join("feather_hubert.f32");
    let matrix = FeatureMatrix::new(2, 4, vec![0.5; 8]).unwrap();
    write_feature_file_no_clobber(&path, &matrix).unwrap()
}

#[test]
fn the_payload_names_every_published_location() {
    let directory = tempfile::tempdir().unwrap();
    let output_dir = directory.path().join("features");
    let artifact = artifact(directory.path());

    let value = feature_to_json(&output_dir, &artifact, &"c".repeat(64));

    assert_eq!(value["output_dir"], output_dir.display().to_string());
    assert_eq!(
        value["feature_file"],
        output_dir.join("feather_hubert.f32").display().to_string()
    );
    assert_eq!(value["tokens"], 2);
    assert_eq!(value["dims"], 4);
    // Two tokens per video frame, and the odd one was already dropped.
    assert_eq!(value["frame_count"], 1);
    // The 44-byte header plus 2 * 4 f32 values.
    assert_eq!(value["bytes"], 76);
    assert_eq!(artifact.sha256().len(), 64);
    assert_eq!(value["sha256"], artifact.sha256());
    assert_eq!(value["model_sha256"], "c".repeat(64));
}

#[test]
fn the_payload_omits_the_per_token_detail() {
    let directory = tempfile::tempdir().unwrap();
    let artifact = artifact(directory.path());

    let value = feature_to_json(&directory.path().join("features"), &artifact, "d");

    let object = value.as_object().expect("the payload is an object");
    assert_eq!(object.len(), 8);
    assert!(object.get("values").is_none());
    assert!(object.get("waveform").is_none());
}
```

- [ ] **Step 2: Run the test and watch it fail**

Run: `cargo test -p feathertalk-worker --test feature_result`

Expected: FAIL to compile with `error[E0432]: unresolved import feathertalk_worker::feature_to_json`.

- [ ] **Step 3: Implement**

Create `rust/crates/feathertalk-worker/src/feature_result.rs`, modelled on `quality_result.rs` down to the `path_text` helper, because both modules answer the same question in the same voice.

```rust
use std::path::Path;

use feathertalk_audio::FeatureArtifact;
use serde_json::{Value, json};

/// Shapes a published feature file as the JSON object a `completed` event
/// carries.
///
/// Like a published frame set and unlike a probe, the payload names the
/// locations: the caller asked for a project directory and the worker chose the
/// layout inside it. The tokens themselves are deliberately absent -- one JSON
/// line per task must stay small, the file at the reported path holds every
/// number, and the digest is here to say which file was meant. `model_sha256`
/// comes from the package manifest, not from the artifact: it is what lets a
/// later run decide whether these features still match the encoder.
pub fn feature_to_json(output_dir: &Path, artifact: &FeatureArtifact, model_sha256: &str) -> Value {
    json!({
        "output_dir": path_text(output_dir),
        "feature_file": path_text(artifact.path()),
        "tokens": artifact.tokens(),
        "dims": artifact.dims(),
        "frame_count": artifact.tokens() / 2,
        "bytes": artifact.bytes(),
        "sha256": artifact.sha256(),
        "model_sha256": model_sha256,
    })
}

fn path_text(path: &Path) -> String {
    path.display().to_string()
}
```

In `rust/crates/feathertalk-worker/src/lib.rs`, declare `mod feature_result;` between `mod extract_frames;` and `mod handshake;`, and re-export `pub use feature_result::feature_to_json;` immediately after `pub use extract_frames::execute_extract_frames;`.

- [ ] **Step 4: Run the test and watch it pass**

Run: `cargo test -p feathertalk-worker --test feature_result`, expecting 2 passed. Then `cargo test -p feathertalk-worker`.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/src/feature_result.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/tests/feature_result.rs
git commit -m "feat(worker): shape the feature result payload"
```

---

### Task 5: Share the project directory admission

**Files:**
- Create: `rust/crates/feathertalk-worker/src/admission.rs`
- Modify: `rust/crates/feathertalk-worker/src/extract_frames.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs`
- Test: `rust/crates/feathertalk-worker/tests/extract_frames.rs`, unchanged, which is the whole point

**Interfaces:**
- Consumes: the private `PROJECT_MANIFEST`, `check_project_dir`, and `invalid_request` that live in `extract_frames.rs` today.
- Produces: `pub(crate) fn check_project_dir(project_dir: &Path) -> Result<(), TaskError>` and `pub(crate) fn invalid_request(summary: &'static str, detail: String) -> TaskError`, both in a new private `crate::admission` module. Task 6 calls both, and `extract_frames.rs` keeps calling both through an import.

**Why:** Task 6 needs exactly these two helpers with exactly this behaviour: the same absolute-path rule, the same refusal to follow a symlinked directory, the same `project.json` requirement. Copying them would leave two bodies that are byte-identical apart from one doc line, and would put four user-facing Chinese summaries in two places at once -- the shape of a bug where two commands slowly start describing the same bad directory differently. Sharing a small helper is not a new pattern in this crate: `extract_frames.rs` already imports `media_failure` and `unsupported` from `commands.rs` for the same reason.

The move gets its own task because it changes no behaviour, and a behaviour-preserving diff is worth keeping out of the diff that introduces a command. The 8 tests in `tests/extract_frames.rs` are the proof: they have to pass before and after without a single edit.

This task has no failing test to write first. A pure move has nothing to fail, and asserting the same thing again from a new module path would test the compiler rather than the code. The steps below replace the red-green pair with a recorded baseline and a comparison against it.

- [ ] **Step 1: Record the baseline**

Run: `cargo test -p feathertalk-worker --test extract_frames`

Expected: PASS, 8 passed. Step 4 has to reproduce that number and those names exactly; anything else means something moved that should not have.

- [ ] **Step 2: Move the three items**

Create `rust/crates/feathertalk-worker/src/admission.rs` with the constant and both functions cut verbatim out of `extract_frames.rs`. Only the doc comments change, and only to stop naming one command: `check_project_dir` now says "the commands that call this", and it gains a paragraph saying why a single copy matters. The module gets no `//!` header, because no other module in `src/` has one -- `lib.rs` carries the crate's.

```rust
use std::{fs, path::Path};

use feathertalk_domain::{ErrorCode, TaskError, TaskStage};

use crate::error_map::clamp;

/// The manifest every project directory carries. `feathertalk-project` owns the
/// name but exports no constant for it (`src/package.rs:66`), so the literal is
/// duplicated the way `cli/src/render.rs` duplicates the worker's environment
/// variable names.
const PROJECT_MANIFEST: &str = "project.json";

/// `feathertalk_project::validate_project_dir` cannot be reused here: it
/// requires the finished asset set, including directories the commands that
/// call this are about to create. What has to hold before a command runs is
/// narrower -- a real directory carrying a manifest.
///
/// Two commands need this answer, and the answer is four user-facing summaries.
/// Keeping one copy is what stops the wording from drifting per command.
pub(crate) fn check_project_dir(project_dir: &Path) -> Result<(), TaskError> {
    if !project_dir.is_absolute() {
        return Err(invalid_request(
            "工程目录必须是绝对路径",
            format!("project_dir {} is not absolute", project_dir.display()),
        ));
    }
    // `symlink_metadata` does not follow links, so a symlinked directory is
    // rejected here the way `feathertalk-project` rejects one.
    let metadata = fs::symlink_metadata(project_dir).map_err(|error| {
        invalid_request(
            "工程目录不可用",
            format!("{}: {error}", project_dir.display()),
        )
    })?;
    if !metadata.is_dir() {
        return Err(invalid_request(
            "工程目录不可用",
            format!("{} is not a directory", project_dir.display()),
        ));
    }
    let manifest = project_dir.join(PROJECT_MANIFEST);
    let found = fs::symlink_metadata(&manifest)
        .map(|metadata| metadata.is_file())
        .unwrap_or(false);
    if !found {
        return Err(invalid_request(
            "工程目录缺少 project.json",
            format!("{} is missing or not a regular file", manifest.display()),
        ));
    }
    Ok(())
}

/// Every admission failure reports `MediaInvalid`: the request named a
/// directory or an input file the worker cannot work with.
pub(crate) fn invalid_request(summary: &'static str, detail: String) -> TaskError {
    TaskError::new(
        ErrorCode::MediaInvalid,
        summary,
        &clamp(&detail),
        TaskStage::Preparing,
    )
}
```

- [ ] **Step 3: Point `extract_frames.rs` at the module**

Delete the constant and both functions from `rust/crates/feathertalk-worker/src/extract_frames.rs`, then fix the imports. Four things leave with them: `use std::{fs, path::Path};` in full, `ErrorCode` and `TaskError` from the `feathertalk_domain` group, and `error_map::clamp` from the `crate` group. The domain group then fits on one line:

```rust
use feathertalk_domain::{ExtractFramesParams, Progress, TaskKind, TaskStage};
```

And the `crate` group gains one nested entry, which rustfmt sorts before `commands`:

```rust
use crate::{
    CommandOutcome, TaskReporter, WorkerConfig,
    admission::{check_project_dir, invalid_request},
    commands::{media_failure, unsupported},
    is_pipeline_cancellation, pipeline_task_error, quality_task_error, quality_to_json,
};
```

In `rust/crates/feathertalk-worker/src/lib.rs`, declare `mod admission;` between `mod adapters;` and `mod commands;`. Nothing is re-exported: both helpers are `pub(crate)`, and no test outside the crate has a reason to call them.

- [ ] **Step 4: Confirm nothing changed**

Run: `cargo test -p feathertalk-worker --test extract_frames`

Expected: PASS, 8 passed, the same 8 names as Step 1. Then `cargo test -p feathertalk-worker`, then `cargo clippy -p feathertalk-worker --all-targets -- -D warnings` -- clippy earns its place here because a move is exactly how an unused import survives.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/src/admission.rs rust/crates/feathertalk-worker/src/extract_frames.rs rust/crates/feathertalk-worker/src/lib.rs
git commit -m "refactor(worker): share the project directory admission"
```

---

### Task 6: Extract features on the CPU

**Files:**
- Create: `rust/crates/feathertalk-worker/src/extract_features.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs`
- Test: `rust/crates/feathertalk-worker/tests/extract_features.rs` (new)

**Interfaces:**
- Consumes: `read_wav_16k_mono` and `MAX_WAV_FILE_BYTES` behaviour from Task 1; `check_project_dir` and `invalid_request` from Task 5; `audio_task_error`, `is_audio_cancellation`, and `AudioError::Cancelled` from Task 3; `feature_to_json` from Task 4; and the parts of `feathertalk-audio` that already shipped -- `ChunkEncoder`, `ChunkPlan`, `DEFAULT_CHUNK_SAMPLES`, `MAX_FEATURE_FILE_BYTES`, `normalize_waveform`, `plan_chunks`, `expected_hubert_frames`, `extract_long_audio`, `drop_odd_token`, `write_feature_file_no_clobber`.
- Produces: `pub fn execute_extract_features<E: ChunkEncoder>(params: &ExtractFeaturesParams, token: &CancellationToken, reporter: &dyn TaskReporter, encoder: &mut E, model_sha256: &str) -> CommandOutcome`. Task 9 is the only caller, and it supplies the encoder Task 8 loads.

**Why:** This task is the command. Three decisions are settled here, and they are why it is one task and not three.

The encoder arrives as a `&mut E` type parameter rather than as a loaded model. `extract_long_audio` needs a sized `ChunkEncoder`, so a trait object is not available, and taking the encoder from the caller is what lets every test here run without weights: a fake that answers with the token count the plan predicts covers the whole contract in milliseconds instead of loading 40 MB of parameters. Loading the real encoder is Task 8's job and wiring the two together is Task 9's.

Admission projects the output size before the first forward pass. `write_feature_file_no_clobber` does refuse an oversized file and `plan_chunks` does refuse an absurd chunk count, but both answer after the CPU has already spent the minutes. Planning twice costs nothing -- `plan_chunks` walks no samples, only counts -- and it buys a refusal that arrives at once. The same reasoning covers the existing-file check: without a `force` flag, a collision is going to be refused either way, so it should be refused before the work rather than after it.

Cancellation is polled between chunks and nowhere else. One chunk is a single forward pass over twenty seconds of audio with no seam inside it, so a finer check would have to reach into burn. `ChunkEncoder::encode` can only refuse with an `AudioError`, which is the whole reason `AudioError::Cancelled` exists, and `audio_failure` is what turns that refusal back into `CommandOutcome::Cancelled` instead of reporting a crash.

The odd-token trim is not a decision but the contract: two FeatherHuBERT tokens per 25 fps video frame, so a token without a partner has no frame to belong to.

- [ ] **Step 1: Write the failing test**

Create `rust/crates/feathertalk-worker/tests/extract_features.rs`, shaped like `tests/extract_frames.rs`: a `Recorder` that collects stage events behind a `Mutex`, an `expect_failure` helper that panics with the outcome it did not want, and a fixture that builds a real project directory inside a `TempDir`.

Two pieces are new. `FakeEncoder` answers each chunk with `expected_hubert_frames(samples.len()) * dims` values, which is exactly the shape the real encoder returns, and it can cancel the run from inside the first chunk -- the only place a cancel can land. And `write_wav` writes the canonical 44-byte header by hand: the worker's test suite depends on no media tools, and building the one wav shape the reader admits takes fifteen lines.

`CommandOutcome` derives only `Debug`, so the success and cancellation paths are read with `match` and `matches!` rather than compared.

```rust
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Mutex,
};

use feathertalk_audio::{AudioError, ChunkEncoder, expected_hubert_frames};
use feathertalk_domain::{ErrorCode, ExtractFeaturesParams, Progress, TaskError, TaskStage};
use feathertalk_media::CancellationToken;
use feathertalk_worker::{CommandOutcome, NoReporter, TaskReporter, execute_extract_features};
use tempfile::TempDir;

/// Stands in for the digest `FeatureModel` reads out of the package manifest.
const MODEL_SHA256: &str = "a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4";

/// An encoder that records the chunks it was asked for and answers with the
/// token count the plan expects, so a test can pin the plumbing without
/// weights. `cancel_on_first_chunk` cancels from inside the first call, which
/// is the only way to reach the seam between two chunks.
struct FakeEncoder {
    dims: usize,
    chunks: Vec<usize>,
    cancel_on_first_chunk: Option<CancellationToken>,
}

impl FakeEncoder {
    fn new(dims: usize) -> Self {
        Self {
            dims,
            chunks: Vec::new(),
            cancel_on_first_chunk: None,
        }
    }
}

impl ChunkEncoder for FakeEncoder {
    fn output_dim(&self) -> usize {
        self.dims
    }

    fn encode(&mut self, chunk_index: usize, samples: &[f32]) -> Result<Vec<f32>, AudioError> {
        self.chunks.push(chunk_index);
        if chunk_index == 0
            && let Some(token) = &self.cancel_on_first_chunk
        {
            token.cancel();
        }
        Ok(vec![0.5; expected_hubert_frames(samples.len()) * self.dims])
    }
}

struct Recorder {
    events: Mutex<Vec<(TaskStage, Option<Progress>)>>,
}

impl Recorder {
    fn new() -> Self {
        Self {
            events: Mutex::new(Vec::new()),
        }
    }

    fn events(&self) -> Vec<(TaskStage, Option<Progress>)> {
        self.events.lock().unwrap().clone()
    }
}

impl TaskReporter for Recorder {
    fn report(&self, stage: TaskStage, progress: Option<Progress>) {
        self.events.lock().unwrap().push((stage, progress));
    }
}

/// A project directory that passes admission: a real directory holding
/// `project.json`, with the normalised wav inside `assets/`.
fn project(samples: &[i16]) -> (TempDir, ExtractFeaturesParams) {
    let root = tempfile::tempdir().unwrap();
    let project_dir = root.path().join("project");
    let assets = project_dir.join("assets");
    fs::create_dir_all(&assets).unwrap();
    // Only the file's presence is checked; nothing here parses the manifest.
    fs::write(project_dir.join("project.json"), b"{}").unwrap();
    let audio = assets.join("audio_16k_mono.wav");
    write_wav(&audio, samples);
    (root, ExtractFeaturesParams { project_dir, audio })
}

/// The canonical 44-byte header `normalize_media` produces, followed by the
/// samples: 16 kHz, mono, 16-bit PCM.
fn write_wav(path: &Path, samples: &[i16]) {
    let payload = samples.len() as u32 * 2;
    let mut bytes = Vec::with_capacity(44 + payload as usize);
    bytes.extend(b"RIFF");
    bytes.extend((36 + payload).to_le_bytes());
    bytes.extend(b"WAVEfmt ");
    bytes.extend(16u32.to_le_bytes());
    bytes.extend(1u16.to_le_bytes());
    bytes.extend(1u16.to_le_bytes());
    bytes.extend(16_000u32.to_le_bytes());
    bytes.extend(32_000u32.to_le_bytes());
    bytes.extend(2u16.to_le_bytes());
    bytes.extend(16u16.to_le_bytes());
    bytes.extend(b"data");
    bytes.extend(payload.to_le_bytes());
    bytes.extend(samples.iter().copied().flat_map(i16::to_le_bytes));
    fs::write(path, bytes).unwrap();
}

/// A ramp rather than a constant, because `normalize_waveform` refuses a
/// waveform with no dynamic range.
fn ramp(count: usize) -> Vec<i16> {
    (0..count)
        .map(|index| (index % 2_000) as i16 - 1_000)
        .collect()
}

fn run(
    params: &ExtractFeaturesParams,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    encoder: &mut FakeEncoder,
) -> CommandOutcome {
    execute_extract_features(params, token, reporter, encoder, MODEL_SHA256)
}

fn progress(completed: u64, total: u64) -> Option<Progress> {
    Some(Progress {
        completed,
        total: Some(total),
    })
}

fn expect_failure(outcome: CommandOutcome) -> TaskError {
    match outcome {
        CommandOutcome::Failed(error) => error,
        other => panic!("expected a failure, got {other:?}"),
    }
}

#[test]
fn a_two_second_wav_becomes_an_even_token_feature_file() {
    // 32 000 samples at 16 kHz: (32 000 - 80) / 320 is 99 frames.
    let (_root, params) = project(&ramp(32_000));
    let token = CancellationToken::new();
    let recorder = Recorder::new();
    let mut encoder = FakeEncoder::new(4);

    let result = match run(&params, &token, &recorder, &mut encoder) {
        CommandOutcome::Completed(Some(result)) => result,
        other => panic!("expected a completed command, got {other:?}"),
    };

    let features = params.project_dir.join("assets").join("features");
    let file = features.join("feather_hubert.f32");
    assert_eq!(result["output_dir"], features.display().to_string());
    assert_eq!(result["feature_file"], file.display().to_string());
    // 99 is odd, so the last token has no video frame to pair with.
    assert_eq!(result["tokens"], 98);
    assert_eq!(result["dims"], 4);
    assert_eq!(result["frame_count"], 49);
    // The 44-byte header plus 98 * 4 f32 values.
    assert_eq!(result["bytes"], 1_612);
    assert_eq!(result["model_sha256"], MODEL_SHA256);
    assert_eq!(fs::metadata(&file).unwrap().len(), 1_612);
    // `DEFAULT_CHUNK_SAMPLES` is 320 000, so two seconds are one chunk.
    assert_eq!(encoder.chunks, vec![0]);
    assert_eq!(
        recorder.events(),
        vec![
            (TaskStage::Preparing, None),
            (TaskStage::ExtractingFeatures, progress(1, 1)),
        ]
    );
}

#[test]
fn a_long_wav_reports_progress_for_every_chunk() {
    // 640 000 samples are two whole chunks and 1 999 frames.
    let (_root, params) = project(&ramp(640_000));
    let token = CancellationToken::new();
    let recorder = Recorder::new();
    let mut encoder = FakeEncoder::new(4);

    let result = match run(&params, &token, &recorder, &mut encoder) {
        CommandOutcome::Completed(Some(result)) => result,
        other => panic!("expected a completed command, got {other:?}"),
    };

    assert_eq!(result["tokens"], 1_998);
    assert_eq!(result["frame_count"], 999);
    assert_eq!(result["bytes"], 32_012);
    // The chunks overlap by `HUBERT_KERNEL - HUBERT_STRIDE` samples, so the
    // first one is 320 080 samples long and the second one 320 000.
    assert_eq!(encoder.chunks, vec![0, 1]);
    assert_eq!(
        recorder.events(),
        vec![
            (TaskStage::Preparing, None),
            (TaskStage::ExtractingFeatures, progress(1, 2)),
            (TaskStage::ExtractingFeatures, progress(2, 2)),
        ]
    );
}

#[test]
fn a_cancel_between_chunks_leaves_no_feature_file() {
    let (_root, params) = project(&ramp(640_000));
    let token = CancellationToken::new();
    let mut encoder = FakeEncoder::new(4);
    encoder.cancel_on_first_chunk = Some(token.clone());

    let outcome = run(&params, &token, &NoReporter, &mut encoder);

    assert!(
        matches!(outcome, CommandOutcome::Cancelled),
        "expected a cancelled run, got {outcome:?}"
    );
    assert_eq!(encoder.chunks, vec![0]);
    assert!(!params.project_dir.join("assets").join("features").exists());
}

#[test]
fn relative_paths_are_rejected_before_anything_is_touched() {
    let (_root, params) = project(&ramp(32_000));
    let token = CancellationToken::new();
    let mut encoder = FakeEncoder::new(4);
    let relative_dir = ExtractFeaturesParams {
        project_dir: PathBuf::from("project"),
        audio: params.audio.clone(),
    };
    let relative_audio = ExtractFeaturesParams {
        project_dir: params.project_dir.clone(),
        audio: PathBuf::from("assets/audio_16k_mono.wav"),
    };

    let first = expect_failure(run(&relative_dir, &token, &NoReporter, &mut encoder));
    let second = expect_failure(run(&relative_audio, &token, &NoReporter, &mut encoder));

    assert_eq!(first.code, ErrorCode::MediaInvalid);
    assert_eq!(first.stage, TaskStage::Preparing);
    assert_eq!(first.summary, "工程目录必须是绝对路径");
    assert_eq!(second.code, ErrorCode::MediaInvalid);
    assert_eq!(second.stage, TaskStage::Preparing);
    assert_eq!(second.summary, "音频文件必须是绝对路径");
    assert!(encoder.chunks.is_empty());
}

#[test]
fn a_project_without_a_manifest_is_rejected_before_the_audio_is_read() {
    let (_root, params) = project(&ramp(32_000));
    fs::remove_file(params.project_dir.join("project.json")).unwrap();
    let token = CancellationToken::new();
    let mut encoder = FakeEncoder::new(4);

    let error = expect_failure(run(&params, &token, &NoReporter, &mut encoder));

    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.stage, TaskStage::Preparing);
    assert_eq!(error.summary, "工程目录缺少 project.json");
    assert!(encoder.chunks.is_empty());
}

#[test]
fn a_short_audio_file_is_rejected_with_its_frame_count() {
    // 400 samples are exactly one frame, and one token has no pair.
    let (_root, params) = project(&ramp(400));
    let token = CancellationToken::new();
    let mut encoder = FakeEncoder::new(4);

    let error = expect_failure(run(&params, &token, &NoReporter, &mut encoder));

    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.stage, TaskStage::Preparing);
    assert_eq!(error.summary, "音频太短，无法提取特征");
    assert!(
        error
            .detail
            .contains("400 samples yield 1 FeatherHuBERT frame"),
        "detail was {}",
        error.detail
    );
    assert!(encoder.chunks.is_empty());
}

#[test]
fn an_existing_feature_file_is_never_overwritten() {
    let (_root, params) = project(&ramp(32_000));
    let features = params.project_dir.join("assets").join("features");
    fs::create_dir_all(&features).unwrap();
    let existing = features.join("feather_hubert.f32");
    fs::write(&existing, b"old").unwrap();
    let token = CancellationToken::new();
    let mut encoder = FakeEncoder::new(4);

    let error = expect_failure(run(&params, &token, &NoReporter, &mut encoder));

    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.stage, TaskStage::Preparing);
    assert_eq!(error.summary, "特征文件已存在");
    assert_eq!(fs::read(&existing).unwrap(), b"old".to_vec());
    assert!(encoder.chunks.is_empty());
}

#[test]
fn a_file_that_is_not_a_wav_is_rejected_as_invalid_media() {
    let (_root, params) = project(&ramp(32_000));
    fs::write(&params.audio, b"this is not a wav file").unwrap();
    let token = CancellationToken::new();
    let mut encoder = FakeEncoder::new(4);

    let error = expect_failure(run(&params, &token, &NoReporter, &mut encoder));

    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.stage, TaskStage::Preparing);
    assert_eq!(error.summary, "音频文件不是有效的 WAV");
    assert!(encoder.chunks.is_empty());
}
```

- [ ] **Step 2: Run the test and watch it fail**

Run: `cargo test -p feathertalk-worker --test extract_features`

Expected: FAIL to compile with `error[E0432]: unresolved import feathertalk_worker::execute_extract_features`.

- [ ] **Step 3: Implement**

Create `rust/crates/feathertalk-worker/src/extract_features.rs`. The layout follows `extract_frames.rs`: the duplicated path literals as consts at the top, the public entry point, an `admit` function that hands the body everything it established, a progress bridge, and a failure mapper. `Admitted` exists so the body never re-reads the request and never recomputes a path.

```rust
use std::path::PathBuf;

use feathertalk_audio::{
    AudioError, ChunkEncoder, ChunkPlan, DEFAULT_CHUNK_SAMPLES, MAX_FEATURE_FILE_BYTES,
    drop_odd_token, expected_hubert_frames, extract_long_audio, normalize_waveform, plan_chunks,
    read_wav_16k_mono, write_feature_file_no_clobber,
};
use feathertalk_domain::{ExtractFeaturesParams, Progress, TaskStage};
use feathertalk_media::CancellationToken;

use crate::{
    CommandOutcome, TaskReporter,
    admission::{check_project_dir, invalid_request},
    audio_task_error, feature_to_json, is_audio_cancellation,
};

/// The asset directory `normalize_media` writes into.
const ASSETS_DIR: &str = "assets";

/// The feature subdirectory. `feathertalk-audio` owns the file name but nothing
/// owns the directory, so the worker decides the layout.
const FEATURES_DIR: &str = "features";

/// The published feature file. `feathertalk-audio` keeps its own copy of this
/// name private (`src/commit.rs:15`), so the literal is duplicated here.
const FEATURE_FILE_NAME: &str = "feather_hubert.f32";

/// The fixed feature header. `feathertalk-audio` computes the same number as a
/// `pub(crate) usize` (`src/format.rs`), so admission keeps a `u64` copy to
/// project a file size without a cast.
const FEATURE_HEADER_BYTES: u64 = 44;

/// Extract the FeatherHuBERT features of a normalised wav into a project's
/// asset directory.
///
/// The encoder arrives by mutable reference so a caller can drive the command
/// without loading weights; `FeatureModel` in this crate supplies the real one.
/// It is a type parameter rather than a trait object because
/// `feathertalk_audio::extract_long_audio` needs a sized encoder.
pub fn execute_extract_features<E: ChunkEncoder>(
    params: &ExtractFeaturesParams,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    encoder: &mut E,
    model_sha256: &str,
) -> CommandOutcome {
    // One stage before the first chunk: the caller has already spent seconds
    // loading the model and admission reads the whole wav, and the CLI would
    // otherwise print nothing until the first chunk lands.
    reporter.report(TaskStage::Preparing, None);
    let admitted = match admit(params, encoder.output_dim()) {
        Ok(admitted) => admitted,
        Err(outcome) => return outcome,
    };
    // The runtime checks the token before dispatch; this second check covers
    // the seconds admission spent reading a large wav file.
    if token.is_cancelled() {
        return CommandOutcome::Cancelled;
    }
    let normalized = match normalize_waveform(&admitted.samples) {
        Ok(normalized) => normalized,
        Err(error) => return audio_failure(&error),
    };
    let total = admitted.plan.ranges().len() as u64;
    let mut progress = ChunkProgress {
        inner: encoder,
        reporter,
        token,
        total,
        completed: 0,
    };
    let matrix = match extract_long_audio(&normalized, &mut progress, DEFAULT_CHUNK_SAMPLES) {
        Ok(matrix) => matrix,
        Err(error) => return audio_failure(&error),
    };
    // Two tokens per video frame, so an odd one has no frame to belong to.
    let matrix = drop_odd_token(matrix);
    match write_feature_file_no_clobber(&admitted.destination, &matrix) {
        Ok(artifact) => {
            let payload = feature_to_json(&admitted.output_dir, &artifact, model_sha256);
            CommandOutcome::Completed(Some(payload))
        }
        Err(error) => audio_failure(&error),
    }
}

/// Everything admission established, so the command body never re-reads the
/// request.
struct Admitted {
    samples: Vec<f32>,
    plan: ChunkPlan,
    output_dir: PathBuf,
    destination: PathBuf,
}

/// Everything that has to hold before the encoder runs, ordered so that the
/// cheapest refusal happens first.
///
/// The plan is computed here and again inside `extract_long_audio`, which costs
/// nothing -- it walks no samples -- and it is what lets admission refuse an
/// oversized feature file before a forward pass rather than after all of them.
fn admit(params: &ExtractFeaturesParams, dims: usize) -> Result<Admitted, CommandOutcome> {
    check_project_dir(&params.project_dir).map_err(CommandOutcome::Failed)?;
    if !params.audio.is_absolute() {
        return Err(CommandOutcome::Failed(invalid_request(
            "音频文件必须是绝对路径",
            format!("audio {} is not absolute", params.audio.display()),
        )));
    }
    let output_dir = params.project_dir.join(ASSETS_DIR).join(FEATURES_DIR);
    let destination = output_dir.join(FEATURE_FILE_NAME);
    // `write_feature_file_no_clobber` refuses the collision anyway, but only
    // after the encoder has run. This slice has no `force` flag, so the
    // cheapest correct answer is to refuse now.
    if destination.exists() {
        return Err(CommandOutcome::Failed(invalid_request(
            "特征文件已存在",
            format!("{} already exists", destination.display()),
        )));
    }
    let samples = read_wav_16k_mono(&params.audio)
        .map_err(|error| CommandOutcome::Failed(audio_task_error(&error)))?;
    let frames = expected_hubert_frames(samples.len());
    if frames < 2 {
        return Err(CommandOutcome::Failed(invalid_request(
            "音频太短，无法提取特征",
            format!(
                "{} samples yield {frames} FeatherHuBERT frame(s), at least 2 are required",
                samples.len()
            ),
        )));
    }
    let plan = plan_chunks(samples.len(), DEFAULT_CHUNK_SAMPLES)
        .map_err(|error| CommandOutcome::Failed(audio_task_error(&error)))?;
    // Overflow takes the rejection path too: a token count that cannot be
    // turned into a byte count is over the limit by definition.
    let projected = (plan.target_tokens() as u64)
        .checked_mul(dims as u64)
        .and_then(|values| values.checked_mul(4))
        .and_then(|bytes| bytes.checked_add(FEATURE_HEADER_BYTES))
        .unwrap_or(u64::MAX);
    if projected > MAX_FEATURE_FILE_BYTES {
        return Err(CommandOutcome::Failed(invalid_request(
            "音频过长，特征文件会超出上限",
            format!(
                "{} tokens at {dims} dims need {projected} bytes, over {MAX_FEATURE_FILE_BYTES}",
                plan.target_tokens()
            ),
        )));
    }
    Ok(Admitted {
        samples,
        plan,
        output_dir,
        destination,
    })
}

/// Bridges the encoder onto the worker's reporter and token.
///
/// One chunk is the granularity, which `DEFAULT_CHUNK_SAMPLES` fixes at just
/// over twenty seconds of audio: fine enough for a progress bar, coarse enough
/// that reporting is never the bottleneck.
struct ChunkProgress<'a, E: ChunkEncoder> {
    inner: &'a mut E,
    reporter: &'a dyn TaskReporter,
    token: &'a CancellationToken,
    total: u64,
    completed: u64,
}

impl<E: ChunkEncoder> ChunkEncoder for ChunkProgress<'_, E> {
    fn output_dim(&self) -> usize {
        self.inner.output_dim()
    }

    fn encode(&mut self, chunk_index: usize, samples: &[f32]) -> Result<Vec<f32>, AudioError> {
        // Between chunks is the only place this command can be interrupted: a
        // chunk is a single forward pass with no seam inside it.
        if self.token.is_cancelled() {
            return Err(AudioError::Cancelled {
                operation: "extract_features",
            });
        }
        let output = self.inner.encode(chunk_index, samples)?;
        self.completed += 1;
        self.reporter.report(
            TaskStage::ExtractingFeatures,
            Some(Progress {
                completed: self.completed,
                total: Some(self.total),
            }),
        );
        Ok(output)
    }
}

/// Cancellation is not a failure: the encoder reports it as an error and the
/// runtime needs it back as `Cancelled`.
fn audio_failure(error: &AudioError) -> CommandOutcome {
    if is_audio_cancellation(error) {
        CommandOutcome::Cancelled
    } else {
        CommandOutcome::Failed(audio_task_error(error))
    }
}
```

In `rust/crates/feathertalk-worker/src/lib.rs`, declare `mod extract_features;` between `mod error_map;` and `mod extract_frames;`, and add `pub use extract_features::execute_extract_features;` immediately before `pub use extract_frames::execute_extract_frames;`.

- [ ] **Step 4: Run the test and watch it pass**

Run: `cargo test -p feathertalk-worker --test extract_features`, expecting 8 passed. Then `cargo test -p feathertalk-worker`, which must still report the same counts as before for every other test binary.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/src/extract_features.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/tests/extract_features.rs
git commit -m "feat(worker): extract features on the CPU"
```

---

### Task 7: Expose the package manifest reader

**Files:**
- Modify: `rust/crates/feathertalk-export/src/package.rs`, `rust/crates/feathertalk-export/src/lib.rs`
- Test: `rust/crates/feathertalk-export/tests/package.rs`

**Interfaces:**
- Consumes: the `pub(crate)` `io::validate_package_directory` and `io::read_bounded_regular`, plus `MANIFEST_FILE_NAME` and `MAX_MANIFEST_BYTES` -- exactly what the first three steps of `load_model_package` already use.
- Produces: `pub fn read_package_manifest(directory: impl AsRef<Path>) -> Result<ModelPackageManifest, PackageError>`. `load_model_package` becomes its first caller; Task 8 is its only caller outside this crate.

**Why:** Task 8 has to know the five FeatherHuBERT hyperparameters before it can build the `ModelDescription` that `load_model_package` demands as `expected`, and the only honest source is the package's own `manifest.json`. The shipped model is 256/2/8/1024/0.0 while `FeatherHubertConfig::default()` is 512/2/12/1024/0.05, so a hardcoded configuration would load exactly one package and then reject every future one with a description mismatch. The worker therefore has to read the manifest before it loads any weights.

Reading it from the worker means one of two things: duplicating the directory contract, the symlink rejection, and the bounded read, or making the loader's own first step callable. Duplication is the worse option, and not marginally. `io::read_bounded_regular` and `io::validate_package_directory` are `pub(crate)`, so a worker-side copy could not call them; it would restate them, and the two statements would drift the first time either side is touched -- different error text for the same broken package, and eventually two different opinions about which files a package directory may contain. Extracting the step leaves one implementation, one set of messages, and one contract.

The extraction is behaviour-preserving by construction. `load_model_package` currently runs `expected.validate()`, then `io::validate_package_directory`, then the bounded manifest read, then the JSON parse, then `manifest.validate()`, then the description comparison. Afterwards it runs `expected.validate()`, then `read_package_manifest` -- which is the middle four steps verbatim -- then the same comparison. Same checks, same order, same errors.

The consequence is that the worker's success path reads `manifest.json` twice: once for the configuration, once inside the loader when it compares `manifest.description()` against the expected description. That is one 64 KiB-bounded read next to a 40 MB weight load, and it is what buys the property that the package, not the worker's source, decides the model's shape.

- [ ] **Step 1: Write the failing test**

Add one test to `rust/crates/feathertalk-export/tests/package.rs`, immediately after `package_round_trip_contains_exact_three_files_and_preserves_tensor_data`. It reuses the file's existing `fixture()` helper, so it needs no fixture of its own. Extend the `feathertalk_export` import list with `MAX_MANIFEST_BYTES` and `read_package_manifest`.

```rust
#[test]
fn manifest_reader_returns_the_published_manifest_and_enforces_the_directory_contract() {
    let (_root, request, model) = fixture();
    let device = Default::default();
    let report = write_model_package::<CpuBackend, _, _>(&request, &model, &device, |device| {
        LinearConfig::new(2, 2).init::<CpuBackend>(device)
    })
    .unwrap();

    let manifest = read_package_manifest(&request.destination).unwrap();
    assert_eq!(manifest, report.manifest);
    assert_eq!(manifest.configuration, request.description.configuration);

    fs::write(request.destination.join("notes.txt"), b"unexpected").unwrap();
    let error = read_package_manifest(&request.destination).unwrap_err();
    assert!(matches!(error, PackageError::InvalidRequest(_)));
    fs::remove_file(request.destination.join("notes.txt")).unwrap();

    let oversized = vec![b' '; usize::try_from(MAX_MANIFEST_BYTES).unwrap() + 1];
    fs::write(request.destination.join("manifest.json"), oversized).unwrap();
    let error = read_package_manifest(&request.destination).unwrap_err();
    assert!(matches!(error, PackageError::InvalidRequest(_)));
    assert!(error.to_string().contains("manifest exceeds 65536 bytes"));
}
```

- [ ] **Step 2: Run the test and watch it fail**

Run: `cargo test -p feathertalk-export --test package`

Expected: FAIL to compile, `error[E0432]: unresolved imports feathertalk_export::MAX_MANIFEST_BYTES, feathertalk_export::read_package_manifest` -- `MAX_MANIFEST_BYTES` is re-exported already, so the unresolved name is `read_package_manifest`.

- [ ] **Step 3: Implement**

In `rust/crates/feathertalk-export/src/package.rs`, add the function immediately above `load_model_package`, copying the four steps out of the loader's body verbatim.

```rust
/// Reads and validates the manifest of an inference model package.
///
/// `load_model_package` runs this first. A caller that has to know the shipped
/// hyperparameters before it can name the expected description -- the worker's
/// FeatherHuBERT loader, for one -- runs it on its own.
pub fn read_package_manifest(
    directory: impl AsRef<Path>,
) -> Result<ModelPackageManifest, PackageError> {
    let directory = directory.as_ref();
    io::validate_package_directory(directory, false)?;
    let manifest_bytes = io::read_bounded_regular(
        &directory.join(crate::MANIFEST_FILE_NAME),
        crate::MAX_MANIFEST_BYTES,
        "manifest",
    )?;
    let manifest: ModelPackageManifest =
        serde_json::from_slice(&manifest_bytes).map_err(|error| {
            PackageError::InvalidManifest(format!("invalid manifest JSON: {error}"))
        })?;
    manifest.validate()?;
    Ok(manifest)
}
```

Then replace the head of `load_model_package` -- everything from `let directory = directory.as_ref();` down to the `manifest.validate()?;` line -- with three lines, leaving the description comparison and everything after it untouched.

```rust
    let directory = directory.as_ref();
    expected.validate()?;
    let manifest = read_package_manifest(directory)?;
```

In `rust/crates/feathertalk-export/src/lib.rs`, add the name to the `package` re-export; it crosses 100 columns, so rustfmt wraps it:

```rust
pub use package::{
    PackageBuildReport, PackageBuildRequest, load_model_package, read_package_manifest,
    write_model_package,
};
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `cargo test -p feathertalk-export --test package`, expecting 10 passed. Then `cargo test -p feathertalk-export` to prove the extraction changed nothing else in the crate.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-export/src/lib.rs rust/crates/feathertalk-export/src/package.rs rust/crates/feathertalk-export/tests/package.rs
git commit -m "refactor(export): expose the package manifest reader"
```

---

### Task 8: Load the FeatherHuBERT package

**Files:**
- Create: `rust/crates/feathertalk-worker/src/features.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs`, `rust/crates/feathertalk-worker/Cargo.toml`
- Test: `rust/crates/feathertalk-worker/tests/features.rs` (new)

**Interfaces:**
- Consumes: `WorkerConfig::features(&self) -> Option<&FeatureToolchain>` and `FeatureToolchain::hubert_dir(&self) -> &Path` from Task 2; `read_package_manifest(directory: impl AsRef<Path>) -> Result<ModelPackageManifest, PackageError>` from Task 7; from `feathertalk-export`, `load_model_package::<B, M, F>(directory: impl AsRef<Path>, expected: &ModelDescription, device: &B::Device, factory: F) -> Result<(M, ModelPackageManifest), PackageError>` and `ModelDescription::feather_hubert(config: FeatherHubertConfig) -> ModelDescription`; from `feathertalk-models`, `FeatherHubertConfig`, `FeatherHubertEncoder<B>`, and `BurnFeatherHubertEncoder::from_model(model: FeatherHubertEncoder<B>, device: &B::Device) -> Self`.
- Produces: `FeatureModel::load(features: &FeatureToolchain) -> Result<FeatureModel, PackageError>` and `FeatureModel::into_parts(self) -> (BurnFeatherHubertEncoder<CpuBackend>, String)`. Task 9 is the only caller of either, and it calls them in that order: load, split, then hand the encoder and the hash to `execute_extract_features`.

**Why:** This is the seam between a directory on disk and the `&mut E: ChunkEncoder` that Task 6 demands. It is a module of its own rather than four statements inside the `commands.rs` arm because the loading rules -- which manifest field decides the model shape, which failure the user must be told about -- are worth reading in one place, and because `commands.rs` stays a dispatch table that way.

The load happens on the calling thread, with no oversized stack and no `Box`. That is a deliberate departure from `models.rs`, which loads the PFLD predictor through a named `pfld-predictor-load` thread with `PREDICTOR_LOAD_STACK_BYTES = 64 * 1024 * 1024` and then keeps it as `Box<PfldLandmarkPredictor<CpuBackend>>`. Both of those exist for one reason: `PfldLandmarkPredictor` inlines a 125_768-byte `GhostOne` struct, so `factory(device)` builds a value on the stack that the default 2 MiB thread stack cannot hold. `FeatherHubertEncoder` is a `Vec<DepthwiseTcnBlock<B>>` plus four small fields; the struct is a few hundred bytes and every weight lives in a heap tensor. Copying the thread would add a thread, a join, and an error path for a failed join, all to protect against a problem this model does not have. If a future configuration ever makes that false, the failure is a stack overflow under `cargo test`, which is impossible to miss.

Because nothing is boxed, `into_parts` can hand the encoder out by value, which is what lets Task 9 write `let (mut encoder, model_sha256) = model.into_parts();` and pass `&mut encoder` straight into the generic `execute_extract_features`. The pair is returned as a tuple instead of exposing two getters because both halves are consumed exactly once, immediately, by the same caller.

The five hyperparameters come from the package manifest. Task 7's Why carries the argument; the short version is that the shipped model is 256/2/8/1024/0.0 while `FeatherHubertConfig::default()` is 512/2/12/1024/0.05, so a hardcoded configuration would fail the loader's description comparison on every package but one. The visible consequence is that `load` reads `manifest.json` twice: once here for the configuration, once inside `load_model_package` when it compares the manifest's description against the expected one. That is a 64 KiB-bounded read and a JSON parse next to a 40 MB weight load.

`feather_hubert_config` refuses a package of another kind by name, before a single weight byte is read, and Task 3's `package_task_error` maps that `PackageError::InvalidManifest` onto `ModelIncompatible` with the detail that names `FEATHERTALK_WORKER_HUBERT_DIR`. A user who points the variable at a mobileone-unet package therefore learns which package it is, rather than reading a tensor-contract mismatch after a 40 MB read. The alternative -- letting `load_model_package` catch it through the description comparison -- reports "expected model description does not match package manifest", which is true and useless, because in this code path the expected description was itself derived from that manifest.

The hash reported by `into_parts` is the manifest's `model.sha256`, which is what Task 4's `feature_to_json` publishes as `model_sha256`. It is a verified hash, not a claim: `load_model_package` runs `io::validate_declared_file(&model_path, &manifest.model)` before it opens the safetensors store, so the bytes that were loaded are the bytes that hash to that string.

The three tests cover exactly what this module adds: the manifest drives the configuration, a directory holding no package is refused before any weight is read, and a package of another kind is refused by name. They deliberately do not restate the loader's own guards -- tampered weights, symlinked entries, extra directory entries, a broken license bundle -- because those live in `rust/crates/feathertalk-export/tests/package.rs` and duplicating them here would double the cost of every future change to the loader. The happy-path assertion is `output_dim() == 64`, which is the cheapest available proof that the configuration came from the manifest: the fixture publishes `FeatherHubertConfig::parity_micro()`, 32/2/2/64/0.0, so a code path that had used `FeatherHubertConfig::default()` would either report 1024 or die in the description comparison.

The fixture publishes a real package with `write_model_package` instead of hand-writing a `manifest.json`. A hand-written manifest would have to carry a matching `TensorContract` -- `tensor_count`, `total_elements`, and one `TensorSpec` per parameter -- and computing that by hand is exactly the job the writer already does; the loader compares it against the module it built, so an approximate contract fails. `parity_micro()` keeps the real package cheap: 35 tensors and 472_384 elements, the same shape `rust/crates/feathertalk-export/tests/feather_hubert.rs` already asserts. `hex` and `sha2` join `[dev-dependencies]` for the one source-file hash the build request declares.

- [ ] **Step 1: Write the failing test**

Add two entries to `[dev-dependencies]` in `rust/crates/feathertalk-worker/Cargo.toml`, between `feathertalk-pfld` and `tempfile`, matching the crate's existing `{ workspace = true }` spelling.

```toml
hex = { workspace = true }
sha2 = { workspace = true }
```

Create `rust/crates/feathertalk-worker/tests/features.rs`. `published_package` is the fixture from `feathertalk-export/tests/package.rs` retargeted at FeatherHuBERT: same synthetic source file, same synthetic license bundle, same `created_at` and `minimum_app_version`, `parity_micro()` in place of `LinearConfig::new(2, 2)`.

```rust
use std::{
    fs,
    path::{Path, PathBuf},
};

use feathertalk_audio::ChunkEncoder;
use feathertalk_export::{
    LicenseBundle, LicenseEntry, ModelConfiguration, ModelDescription, ModelPackageManifest,
    PackageBuildRequest, PackageError, SourceManifest, TrainingManifest, write_model_package,
};
use feathertalk_models::{
    backend::CpuBackend,
    feather_hubert::{FeatherHubertConfig, FeatherHubertEncoder},
};
use feathertalk_worker::{FeatureModel, WorkerConfig};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

/// Publishes a micro FeatherHuBERT package under `root` and returns its directory.
///
/// The configuration is `parity_micro()`, 32/2/2/64/0.0, so an assertion on
/// `output_dim` can tell a manifest-driven load from one that silently fell back
/// to `FeatherHubertConfig::default()`, which is 512/2/12/1024/0.05.
fn published_package(root: &Path) -> PathBuf {
    let source_path = root.join("source.pth");
    fs::write(&source_path, b"source-fixture").unwrap();
    let source_sha256 = hex::encode(Sha256::digest(b"source-fixture"));
    let licenses_path = root.join("LICENSES.input.json");
    let licenses = LicenseBundle {
        schema_version: 1,
        entries: vec![LicenseEntry {
            component: "synthetic FeatherHuBERT fixture".to_owned(),
            license_id: "LicenseRef-Test".to_owned(),
            source_url: "https://example.invalid/feather-hubert".to_owned(),
            notice: "test-only local record".to_owned(),
        }],
    };
    fs::write(&licenses_path, serde_json::to_vec(&licenses).unwrap()).unwrap();
    let config = FeatherHubertConfig::parity_micro();
    let device = Default::default();
    let model = config.init::<CpuBackend>(&device);
    let request = PackageBuildRequest {
        destination: root.join("hubert"),
        description: ModelDescription::feather_hubert(config.clone()),
        source_path,
        source: SourceManifest {
            format: "test".to_owned(),
            identifier: "feather-hubert-fixture".to_owned(),
            version: "1".to_owned(),
            file_name: "source.pth".to_owned(),
            sha256: source_sha256,
            url: None,
        },
        licenses_path,
        created_at: "2026-08-27T00:00:00Z".to_owned(),
        minimum_app_version: "0.1.0".to_owned(),
        training: TrainingManifest::default(),
    };
    write_model_package::<CpuBackend, FeatherHubertEncoder<CpuBackend>, _>(
        &request,
        &model,
        &device,
        |device| config.init::<CpuBackend>(device),
    )
    .unwrap();
    request.destination
}

fn config_for(hubert_dir: &Path) -> WorkerConfig {
    WorkerConfig::from_values_with_toolchains(
        None,
        None,
        None,
        None,
        None,
        Some(hubert_dir.display().to_string()),
    )
}

#[test]
fn a_published_package_loads_with_the_configuration_the_manifest_declares() {
    let root = TempDir::new().unwrap();
    let directory = published_package(root.path());
    let config = config_for(&directory);

    let (encoder, model_sha256) = FeatureModel::load(config.features().unwrap())
        .unwrap()
        .into_parts();

    assert_eq!(encoder.output_dim(), 64);
    let manifest: ModelPackageManifest =
        serde_json::from_slice(&fs::read(directory.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(model_sha256, manifest.model.sha256);
    assert_eq!(
        manifest.configuration,
        ModelConfiguration::FeatherHubert {
            channels: 32,
            expansion: 2,
            num_blocks: 2,
            output_dim: 64,
            dropout: 0.0,
        }
    );
}

#[test]
fn a_directory_without_a_package_is_refused_before_any_weight_is_read() {
    let root = TempDir::new().unwrap();
    let config = config_for(root.path());

    let error = FeatureModel::load(config.features().unwrap()).unwrap_err();

    match error {
        PackageError::InvalidRequest(message) => assert!(
            message.contains("package directory entries must be exactly"),
            "unexpected message: {message}"
        ),
        other => panic!("expected an invalid request, got {other:?}"),
    }
}

#[test]
fn a_package_of_another_kind_is_refused_by_name() {
    let root = TempDir::new().unwrap();
    let directory = published_package(root.path());
    let manifest_path = directory.join("manifest.json");
    let mut manifest: ModelPackageManifest =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    let other = ModelDescription::from_configuration(ModelConfiguration::MobileOneUnet {
        channels: [2, 4, 8, 16, 32],
        num_conv_branches: 1,
        reparameterized: false,
    });
    manifest.model_type = other.model_type;
    manifest.architecture_version = other.architecture_version;
    manifest.configuration = other.configuration;
    manifest.inputs = other.inputs;
    manifest.outputs = other.outputs;
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let config = config_for(&directory);

    let error = FeatureModel::load(config.features().unwrap()).unwrap_err();

    match error {
        PackageError::InvalidManifest(message) => assert_eq!(
            message,
            "expected a feather_hubert configuration, got mobileone_unet"
        ),
        other => panic!("expected an invalid manifest, got {other:?}"),
    }
}
```

The third test rewrites all five description fields together. Rewriting only `configuration` would trip `ModelDescription::validate` first, which compares `model_type`, `architecture_version`, and the input/output contract against the configuration and reports "expected mobileone_unet" instead of the message under test. `ModelDescription::from_configuration` computes the five consistently, so the tampered manifest is a valid manifest for a different model -- which is the case worth covering.

- [ ] **Step 2: Run the test and watch it fail**

Run: `cargo test -p feathertalk-worker --test features`

Expected: FAIL to compile with `error[E0432]: unresolved import feathertalk_worker::FeatureModel`, pointing at the `use feathertalk_worker::{FeatureModel, WorkerConfig};` line with the note `no FeatureModel in the root`. Every other import in the file already resolves: `feathertalk-audio` and `feathertalk-export` became dependencies in Task 3, `hex` and `sha2` were just added, and `WorkerConfig::from_values_with_toolchains` came from Task 2.

- [ ] **Step 3: Implement**

Create `rust/crates/feathertalk-worker/src/features.rs`.

```rust
use feathertalk_export::{
    ModelConfiguration, ModelDescription, PackageError, load_model_package, read_package_manifest,
};
use feathertalk_models::{
    backend::CpuBackend,
    feather_hubert::{BurnFeatherHubertEncoder, FeatherHubertConfig, FeatherHubertEncoder},
};

use crate::FeatureToolchain;

/// A FeatherHuBERT encoder loaded from a strict model package, paired with the
/// weight hash the package declares.
#[derive(Debug)]
pub struct FeatureModel {
    encoder: BurnFeatherHubertEncoder<CpuBackend>,
    model_sha256: String,
}

impl FeatureModel {
    /// Loads the encoder from the directory the FeatherHuBERT toolchain resolved.
    ///
    /// The five hyperparameters come from the package manifest, never from
    /// `FeatherHubertConfig::default()`: the shipped model is 256/2/8/1024/0.0
    /// while the default is 512/2/12/1024/0.05.
    pub fn load(features: &FeatureToolchain) -> Result<Self, PackageError> {
        let directory = features.hubert_dir();
        let manifest = read_package_manifest(directory)?;
        let config = feather_hubert_config(&manifest.configuration)?;
        let device = Default::default();
        let (model, _) = load_model_package::<CpuBackend, FeatherHubertEncoder<CpuBackend>, _>(
            directory,
            &ModelDescription::feather_hubert(config.clone()),
            &device,
            |device| config.init::<CpuBackend>(device),
        )?;
        Ok(Self {
            encoder: BurnFeatherHubertEncoder::from_model(model, &device),
            model_sha256: manifest.model.sha256,
        })
    }

    /// Splits the loaded model into the two pieces the command needs.
    pub fn into_parts(self) -> (BurnFeatherHubertEncoder<CpuBackend>, String) {
        (self.encoder, self.model_sha256)
    }
}

/// Copies the five FeatherHuBERT hyperparameters out of a package configuration.
fn feather_hubert_config(
    configuration: &ModelConfiguration,
) -> Result<FeatherHubertConfig, PackageError> {
    match configuration {
        ModelConfiguration::FeatherHubert {
            channels,
            expansion,
            num_blocks,
            output_dim,
            dropout,
        } => Ok(FeatherHubertConfig {
            channels: *channels,
            expansion: *expansion,
            num_blocks: *num_blocks,
            output_dim: *output_dim,
            dropout: *dropout,
        }),
        other => Err(PackageError::InvalidManifest(format!(
            "expected a feather_hubert configuration, got {}",
            other.model_type()
        ))),
    }
}
```

The `load_model_package` call discards the manifest it returns, because the configuration and the hash were already taken from the first read. `BurnFeatherHubertEncoder::from_model` reads `output_dim` back out of the loaded module's own config, so the `ChunkEncoder` implementation reports the shape of the weights that are actually in memory.

In `rust/crates/feathertalk-worker/src/lib.rs`, declare `mod features;` between `mod feature_result;` and `mod handshake;`, and re-export `pub use features::FeatureModel;` immediately after `pub use feature_result::feature_to_json;`.

- [ ] **Step 4: Run the test and watch it pass**

Run: `cargo test -p feathertalk-worker --test features`, expecting 3 passed. Then `cargo test -p feathertalk-worker` to prove the new module and the new dev-dependencies broke nothing else in the crate.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/Cargo.toml rust/crates/feathertalk-worker/src/features.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/tests/features.rs
git commit -m "feat(worker): load the FeatherHuBERT package"
```

---

### Task 9: Serve extract_features

**Files:**
- Modify: `rust/crates/feathertalk-worker/src/commands.rs`, `rust/crates/feathertalk-worker/src/lib.rs`
- Test: `rust/crates/feathertalk-worker/tests/commands.rs`

**Interfaces:**
- Consumes: `FeatureModel::load(features: &FeatureToolchain) -> Result<FeatureModel, PackageError>` and `FeatureModel::into_parts(self) -> (BurnFeatherHubertEncoder<CpuBackend>, String)` from Task 8; `execute_extract_features(params, token, reporter, encoder, model_sha256) -> CommandOutcome` from Task 6; `package_task_error(&PackageError) -> TaskError` from Task 3; `WorkerConfig::features(&self) -> Option<&FeatureToolchain>` from Task 2; and the existing `commands::unsupported(kind: TaskKind) -> TaskError`.
- Produces: no new name. What it produces is a behaviour -- `Request::ExtractFeatures` stops falling through to the catch-all arm. Task 10 advertises the command on the strength of it, and Task 12 drives it end to end.

**Why:** the arm is shaped exactly like the `ExtractFrames` arm above it: refuse when the toolchain is missing, load the model, delegate. The worker should have one way of serving a command that needs artefacts on disk, not two, so a reader who has understood the frame arm has already understood this one.

The model is loaded per request, the way `FrameModels::load` is, and not cached in `WorkerConfig` or behind a `OnceCell`. The worker handles one command at a time, and this command's wall time is dominated by the encoder passes, not by the read that precedes them: the shipped weight file is 40_436_613 bytes. A cache would buy that read back at the price of deciding when to evict it, and a resident encoder inflates the worker's resident set for every later command that has no use for it. If a future slice ever encodes several audio files inside one command, the cache belongs to that loop, not here, and this arm's contract does not change when it arrives.

The `else` branch returns `unsupported(request.kind())` -- `request.kind()`, not `other.kind()` -- because inside the arm the payload is already bound to `params`, so `request` is the only binding left that still knows the command's slug; the three toolchain guards above spell it the same way. That branch is reachable only by a caller that skipped the handshake or ignored what it said, which makes it an error rather than a panic: the worker owes such a caller a `WorkerCrashed` failure and then its next request, not a dead process.

A load failure goes through `package_task_error(&error)` in one call and nothing else. Task 3 already decided that every `PackageError` on this path is `ModelIncompatible`, with one summary and a detail that names the environment variable, so the arm never inspects an error kind and there is exactly one place to edit if that advice ever changes.

The tests here cover the wiring and nothing else. The command's own behaviour -- the token contract, the progress events, the file it writes -- is Task 6's `tests/extract_features.rs`, and the loader's behaviour is Task 8's `tests/features.rs`. What is new is that a request reaches them at all, and that the two admission failures are reported with the right code: no model directory configured, and a configured directory that holds no loadable package. A test that produced a real feature file from here would need a published package and a wav on disk, which is Task 12's job and runs at release speed.

One of the two tests is honest about being a guard rather than a driver. `extract_features_reports_a_package_failure_as_a_model_incompatibility` is the failing one: before the arm exists the catch-all answers `WorkerCrashed` with the unsupported-command summary, so the code assertion fails. `extract_features_without_a_model_directory_is_refused_with_its_slug` passes before and after by construction, because the catch-all satisfies it by accident today. It is written down anyway, because from this task on it is the only test that pins the `else` branch, and a refactor that dropped that guard would otherwise leave every test in the crate green.

- [ ] **Step 1: Write the failing test**

Add both tests to `rust/crates/feathertalk-worker/tests/commands.rs`, immediately after `an_unsupported_command_is_refused_with_its_slug`. They reuse the file's `FakeRunner` and `bare_config` helpers and need none of their own. Extend the `feathertalk_domain` import list with `ExtractFeaturesParams`; rustfmt rewraps the block as

```rust
use feathertalk_domain::{
    ErrorCode, ExtractFeaturesParams, NormalizeMediaParams, ProbeMediaParams, Progress,
    ProjectDirParams, Request, TaskStage, TrainParams, TrainingMode, UnetVariant,
};
```

```rust
#[test]
fn extract_features_reports_a_package_failure_as_a_model_incompatibility() {
    let temp = tempfile::tempdir().unwrap();
    let request = Request::ExtractFeatures(ExtractFeaturesParams {
        project_dir: temp.path().join("project"),
        audio: temp.path().join("project/assets/audio_16k_mono.wav"),
    });
    let config = WorkerConfig::from_values_with_toolchains(
        None,
        None,
        None,
        None,
        None,
        Some(temp.path().display().to_string()),
    );
    let runner = FakeRunner::new(vec![]);
    let CommandOutcome::Failed(error) = execute_with_runner(
        &request,
        &config,
        &CancellationToken::new(),
        &NoReporter,
        &runner,
    ) else {
        panic!("loading a package out of an empty directory must fail");
    };
    assert_eq!(error.code, ErrorCode::ModelIncompatible);
    assert_eq!(error.summary, "特征模型加载失败");
    assert!(
        error.detail.contains("FEATHERTALK_WORKER_HUBERT_DIR"),
        "{}",
        error.detail
    );
    error.validate().unwrap();
}

#[test]
fn extract_features_without_a_model_directory_is_refused_with_its_slug() {
    let request = Request::ExtractFeatures(ExtractFeaturesParams {
        project_dir: PathBuf::from("C:/tmp/project"),
        audio: PathBuf::from("C:/tmp/project/assets/audio_16k_mono.wav"),
    });
    let runner = FakeRunner::new(vec![]);
    let CommandOutcome::Failed(error) = execute_with_runner(
        &request,
        &bare_config(),
        &CancellationToken::new(),
        &NoReporter,
        &runner,
    ) else {
        panic!("extract_features without a model directory must fail");
    };
    assert_eq!(error.code, ErrorCode::WorkerCrashed);
    assert_eq!(error.summary, "当前 worker 不支持该命令");
    assert!(
        error.detail.contains("extract_features"),
        "{}",
        error.detail
    );
    error.validate().unwrap();
}
```

- [ ] **Step 2: Run the test and watch it fail**

Run: `cargo test -p feathertalk-worker --test commands`

Expected: FAIL in `extract_features_reports_a_package_failure_as_a_model_incompatibility` on the code assertion, `left: WorkerCrashed, right: ModelIncompatible` -- the catch-all arm answered instead of the new one. `extract_features_without_a_model_directory_is_refused_with_its_slug` passes already, for the reason its Why gives.

- [ ] **Step 3: Implement**

In `rust/crates/feathertalk-worker/src/commands.rs`, add the arm after the closing brace of the `Request::ExtractFrames` arm and before `other => CommandOutcome::Failed(unsupported(other.kind())),`.

```rust
        Request::ExtractFeatures(params) => {
            let Some(features) = config.features() else {
                return CommandOutcome::Failed(unsupported(request.kind()));
            };
            let model = match FeatureModel::load(features) {
                Ok(model) => model,
                Err(error) => return CommandOutcome::Failed(package_task_error(&error)),
            };
            let (mut encoder, model_sha256) = model.into_parts();
            execute_extract_features(params, token, reporter, &mut encoder, &model_sha256)
        }
```

The file's `use crate::{...}` block gains `FeatureModel`, `execute_extract_features`, and `package_task_error`; rustfmt sorts the uppercase names ahead of the lowercase ones and rewraps the whole block:

```rust
use crate::{
    FeatureModel, FrameModels, TaskReporter, WorkerConfig, execute_extract_features,
    execute_extract_frames, is_media_cancellation, media_task_error, normalize_to_json,
    package_task_error, pipeline_task_error, probe_to_json, project_task_error,
};
```

Then extend the crate's module doc in `rust/crates/feathertalk-worker/src/lib.rs`. These two lines

```rust
//! This slice serves `validate_project`, `probe_media`, `normalize_media`, and
//! `extract_frames` on the CPU. Every other command in
```

become

```rust
//! This slice serves `validate_project`, `probe_media`, `normalize_media`,
//! `extract_frames`, and `extract_features` on the CPU. Every other command in
```

That sentence is edited here and not in Task 8 because a module that can load the package is not yet a worker that serves the command; the arm above is what makes the claim true.

- [ ] **Step 4: Run the test and watch it pass**

Run: `cargo test -p feathertalk-worker --test commands`, expecting 15 passed. Then `cargo test -p feathertalk-worker` to prove the new arm disturbed none of the crate's other suites.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/src/commands.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/tests/commands.rs
git commit -m "feat(worker): serve extract_features"
```

---

### Task 10: Advertise extract_features

**Files:**
- Modify: `rust/crates/feathertalk-worker/src/handshake.rs`, `rust/crates/feathertalk-worker/src/runtime.rs`
- Test: `rust/crates/feathertalk-worker/tests/handshake.rs`, `rust/crates/feathertalk-worker/tests/runtime.rs`

**Interfaces:**
- Consumes: `WorkerConfig::features(&self) -> Option<&FeatureToolchain>`, `WorkerConfig::feature_rejection(&self) -> Option<&str>`, and `ENV_HUBERT_DIR` from Task 2; the existing `supported_commands`, `ready_frame`, and the `media_reason`/`model_reason` pair in `runtime.rs`.
- Produces: `TaskKind::ExtractFeatures` inside `supported_commands(&WorkerConfig)`, and therefore inside every `ReadyFrame`; plus a private `feature_reason(slug: &str, config: &WorkerConfig) -> String` in `runtime.rs`. Task 11's CLI advice names the same variable this reason names, and Task 12 cannot run at all until the command is advertised, because the client refuses to send a command the handshake left out.

**Why:** the handshake is the contract, and until this task the worker can serve `extract_features` but will never be asked to. The client gates every request on the `supported_commands` list it received, and `runtime.rs` rejects an unlisted command before a task is even created. So Task 9's arm is unreachable in production until this list grows, which is why the two tasks are adjacent and why the end-to-end test cannot come earlier.

The guard sits outside the `if config.media().is_some()` block, and that placement is the one design decision in this task. `ExtractFrames` is nested inside it because extraction shells out to ffmpeg before it loads a model, so without the media toolchain it cannot start. This command shells out to nothing: it reads a wav that `normalize_media` already wrote, feeds it to a burn encoder, and writes one file. Nesting it would mean a worker with a FeatherHuBERT directory and no ffmpeg silently refuses a command it is fully able to run -- and that configuration is not hypothetical, because features can be re-extracted long after the media step, on a machine that never needs ffmpeg again.

The command is appended last, after `ExtractFrames`. The order in the vector is observable: it crosses the wire in the ready frame, the desktop client renders it, and three tests in this crate assert the whole vector element by element. Appending keeps the list append-only across slices, so a new command never invalidates an older expectation's prefix; inserting one in the middle would churn every one of those assertions for no gain.

`Capabilities` does not move. Protocol version 2 has exactly one tool flag, `ffmpeg`, and this command does not use ffmpeg; setting it because a model directory resolved would be a lie, and adding a feature flag is a protocol change that belongs to a protocol slice.

`unsupported_reason` gets an explicit arm instead of falling into the `_` fallback. The fallback is honest but useless here: it lists what the worker does support, which tells an operator that `extract_features` is missing and nothing about how to get it. `feature_reason` mirrors `model_reason` exactly -- a rejected configuration is quoted back, an absent one is named by its variable -- because an operator who mistyped a path and an operator who never set one need different sentences. What it deliberately does not mirror is the `ExtractFrames` precedence rule: there is no media-first branch, because a missing media toolchain is not a reason this command is unavailable.

The third handshake test is the one that pins the placement. `a_feature_model_without_a_media_toolchain_still_offers_extract_features` fails the moment someone tidies the guard into the media block, which is exactly the kind of edit that looks like symmetry and silently removes a capability. The other two cover the ordinary directions: the command appears once the directory resolves, and it stays out when it does not.

- [ ] **Step 1: Write the failing test**

In `rust/crates/feathertalk-worker/tests/handshake.rs`, add `ENV_HUBERT_DIR` to the `feathertalk_worker` import list -- rustfmt rewraps the block as

```rust
use feathertalk_worker::{
    CPU_ADAPTER_ID, DEFAULT_MEDIA_TIMEOUT_MS, ENV_FFPROBE, ENV_HUBERT_DIR, ENV_MEDIA_TIMEOUT_MS,
    ENV_SCRFD_DIR, WorkerConfig, ready_frame, supported_commands,
};
```

-- then append the helper and the three tests at the end of the file, after `models_without_a_media_toolchain_offer_nothing_new`.

```rust
/// Media, models, and the FeatherHuBERT directory all resolve, so the handshake
/// offers every command in this slice.
fn every_toolchain() -> WorkerConfig {
    WorkerConfig::from_values_with_toolchains(
        Some(absolute("ffprobe-test")),
        Some(absolute("ffmpeg-test")),
        None,
        Some(absolute("scrfd-test")),
        Some(absolute("pfld-test")),
        Some(absolute("hubert-test")),
    )
}

#[test]
fn a_worker_with_a_feature_model_offers_extract_features() {
    let config = every_toolchain();
    assert_eq!(config.feature_rejection(), None);
    let frame = ready_frame(&config);
    frame.validate().unwrap();
    assert_eq!(
        frame.supported_commands,
        vec![
            TaskKind::ValidateProject,
            TaskKind::ProbeMedia,
            TaskKind::NormalizeMedia,
            TaskKind::ExtractFrames,
            TaskKind::ExtractFeatures
        ]
    );
}

#[test]
fn a_worker_without_a_feature_model_leaves_extract_features_out() {
    let config = fully_configured();
    assert!(config.features().is_none());
    assert!(
        config
            .feature_rejection()
            .is_some_and(|reason| reason.contains(ENV_HUBERT_DIR))
    );
    assert!(!supported_commands(&config).contains(&TaskKind::ExtractFeatures));
}

#[test]
fn a_feature_model_without_a_media_toolchain_still_offers_extract_features() {
    let config = WorkerConfig::from_values_with_toolchains(
        None,
        None,
        None,
        None,
        None,
        Some(absolute("hubert-test")),
    );
    assert!(config.media().is_none());
    assert_eq!(
        supported_commands(&config),
        vec![TaskKind::ValidateProject, TaskKind::ExtractFeatures]
    );
}
```

In `rust/crates/feathertalk-worker/tests/runtime.rs`, add `ExtractFeaturesParams` to the `feathertalk_domain` import list:

```rust
use feathertalk_domain::{
    CancelFrame, ClientFrame, DomainError, ErrorCode, Event, ExtractFeaturesParams,
    ExtractFramesParams, NormalizeMediaParams, PROTOCOL_VERSION, ProbeMediaParams, Progress,
    ProjectDirParams, Request, ServerFrame, ShutdownFrame, StartFrame, TaskId, TaskKind, TaskStage,
    TrainParams, TrainingMode, UnetVariant, decode_line, encode_line,
};
```

Add the request builder after `extract_frames_request` and the configuration after `full_config`, then the three tests at the end of the file, after `extract_frames_names_the_media_toolchain_before_the_models`. The task id suffixes continue the file's sequence, which ends at `0000002e`.

```rust
fn extract_features_request() -> Request {
    Request::ExtractFeatures(ExtractFeaturesParams {
        project_dir: PathBuf::from("C:/tmp/project"),
        audio: PathBuf::from("C:/tmp/project/assets/audio_16k_mono.wav"),
    })
}

/// Every toolchain resolves, so `extract_features` reaches the executor as well.
fn every_toolchain_config() -> WorkerConfig {
    WorkerConfig::from_values_with_toolchains(
        Some(absolute("ffprobe-test")),
        Some(absolute("ffmpeg-test")),
        None,
        Some(absolute("scrfd-test")),
        Some(absolute("pfld-test")),
        Some(absolute("hubert-test")),
    )
}
```

```rust
#[test]
fn extract_features_reaches_the_executor_once_the_model_directory_resolves() {
    let harness = Harness::start(every_toolchain_config(), instant_executor());
    harness.send(&start(&task("0000002f"), extract_features_request()));
    let frames = harness.finish();

    assert!(rejections(&frames).is_empty(), "{frames:?}");
    assert_eq!(
        stages(&frames),
        vec![
            ("1787900000000-0000002f", "preparing"),
            ("1787900000000-0000002f", "completed"),
        ]
    );
}

#[test]
fn extract_features_is_rejected_with_the_hubert_variable() {
    let harness = Harness::start(full_config(), instant_executor());
    harness.send(&start(&task("00000030"), extract_features_request()));
    let frames = harness.finish();

    let reasons = rejections(&frames);
    assert_eq!(reasons.len(), 1, "{frames:?}");
    assert!(reasons[0].contains("extract_features"), "{}", reasons[0]);
    assert!(
        reasons[0].contains("FEATHERTALK_WORKER_HUBERT_DIR"),
        "{}",
        reasons[0]
    );
    assert!(events(&frames).is_empty());
}

#[test]
fn extract_features_never_asks_for_the_media_toolchain() {
    let harness = Harness::start(bare_config(), instant_executor());
    harness.send(&start(&task("00000031"), extract_features_request()));
    let frames = harness.finish();

    let reasons = rejections(&frames);
    assert_eq!(reasons.len(), 1, "{frames:?}");
    assert!(
        reasons[0].contains("FEATHERTALK_WORKER_HUBERT_DIR"),
        "{}",
        reasons[0]
    );
    assert!(
        !reasons[0].contains("FEATHERTALK_WORKER_FFPROBE"),
        "{}",
        reasons[0]
    );
}
```

- [ ] **Step 2: Run the test and watch it fail**

Run: `cargo test -p feathertalk-worker --test handshake`, then `cargo test -p feathertalk-worker --test runtime`

Expected: five of the six new tests FAIL, all on assertions rather than on compilation, because every name they use already exists.

- `a_worker_with_a_feature_model_offers_extract_features`: the vector comparison, four elements on the left against five on the right.
- `a_feature_model_without_a_media_toolchain_still_offers_extract_features`: `left: [ValidateProject], right: [ValidateProject, ExtractFeatures]`.
- `extract_features_reaches_the_executor_once_the_model_directory_resolves`: the runtime rejected the command instead of queueing it, so `rejections(&frames)` is not empty.
- `extract_features_is_rejected_with_the_hubert_variable`: the fallback arm answered, so the reason lists the supported commands and never names the variable.
- `extract_features_never_asks_for_the_media_toolchain`: same fallback, failing on the `HUBERT_DIR` assertion first.

`a_worker_without_a_feature_model_leaves_extract_features_out` passes already, the same way Task 9's second test does: what it asserts is a hole the fallback fills by accident today, and only this task makes it a hole on purpose.

- [ ] **Step 3: Implement**

In `rust/crates/feathertalk-worker/src/handshake.rs`, add the guard to `supported_commands`, after the media block closes and immediately before `commands`.

```rust
    // Feature extraction needs no media tools: it reads the wav the media
    // commands already wrote, so its only requirement is the model directory.
    if config.features().is_some() {
        commands.push(TaskKind::ExtractFeatures);
    }
```

In `rust/crates/feathertalk-worker/src/runtime.rs`, add `ENV_HUBERT_DIR` to the `use crate::{...}` block:

```rust
use crate::{
    AdapterLockError, AdapterLocks, CPU_ADAPTER_ID, CommandOutcome, ENV_FFMPEG, ENV_FFPROBE,
    ENV_HUBERT_DIR, ENV_PFLD_DIR, ENV_SCRFD_DIR, TaskReporter, WorkerConfig, execute, ready_frame,
    supported_commands,
};
```

Add the arm to `unsupported_reason`, between the `ExtractFrames` arms and the `_` fallback:

```rust
        // Feature extraction needs no media tools, so its only wall is the
        // FeatherHuBERT directory.
        TaskKind::ExtractFeatures => feature_reason(slug, config),
```

Add the function immediately after `model_reason`. Both branches sit inside a wrapped `format!` because the Chinese text counts two columns per character:

```rust
fn feature_reason(slug: &str, config: &WorkerConfig) -> String {
    match config.feature_rejection() {
        Some(rejection) => format!(
            "命令 {slug} 需要可用的特征模型目录，当前配置被拒绝：{rejection}。修正后重启 worker。"
        ),
        None => format!(
            "命令 {slug} 需要 FeatherHuBERT 特征模型，请设置 {ENV_HUBERT_DIR} 后重启 worker。"
        ),
    }
}
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `cargo test -p feathertalk-worker --test handshake`, expecting 12 passed, then `cargo test -p feathertalk-worker --test runtime`, expecting 28 passed. Then `cargo test -p feathertalk-worker`, which now covers the whole worker side of the slice: the two suites above plus `commands`, `extract_features`, `features`, `feature_result`, `config`, and `error_mapping`.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-worker/src/handshake.rs rust/crates/feathertalk-worker/src/runtime.rs rust/crates/feathertalk-worker/tests/handshake.rs rust/crates/feathertalk-worker/tests/runtime.rs
git commit -m "feat(worker): advertise extract_features"
```

---

### Task 11: Add the extract-features subcommand

**Files:**
- Modify: `rust/crates/feathertalk-cli/src/cli.rs`
- Modify: `rust/crates/feathertalk-cli/src/run.rs`
- Modify: `rust/crates/feathertalk-cli/src/render.rs`
- Test: `rust/crates/feathertalk-cli/src/run.rs` (its inline `mod tests`)
- Test: `rust/crates/feathertalk-cli/tests/cli.rs`

**Interfaces:**
- Consumes: `ExtractFeaturesParams` and `Request::ExtractFeatures` from `feathertalk-domain`, and the handshake gate Task 10 added -- a worker without a FeatherHuBERT package leaves `extract_features` out of `supported_commands`, and the client turns that into `ClientError::UnsupportedCommand` before any task starts.
- Produces: `Command::ExtractFeatures { project_dir: PathBuf, audio: PathBuf }`, its `build_request` arm, and the branch of `render_client_error` that names `FEATHERTALK_WORKER_HUBERT_DIR`. Task 12 drives the subcommand against the real worker.

**Why:** the CLI's share is as small here as it was for `extract-frames`: two positional paths, the same empty-string guard every other command uses, and one sentence of advice when the worker cannot take the job. Whether the audio exists, sits inside the project, or decodes at all stays the worker's judgement -- the comment above `build_request` says so -- and Task 6 already refuses each of those cases with a wire error code the renderer knows how to print. A second opinion in the CLI could only disagree with the first.

The advice branch is the part that earns its keep. `extract_features` vanishes from `supported_commands` whenever the package directory is missing or was rejected, and the message the client builds on its own names no variable at all, so an operator learns that the command is unavailable and nothing about how to fix it. Only one variable is named, unlike the four `extract_frames` lists: this command shells out to nothing, so ffmpeg has no bearing on whether it can run.

`stage_label` already maps `TaskStage::ExtractingFeatures` to its Chinese label, so the progress narration needs no change, and `--json` needs none either because it forwards the worker's own frames verbatim.

- [ ] **Step 1: Write the failing tests**

Add two unit tests to the inline module in `rust/crates/feathertalk-cli/src/run.rs`, after `extract_frames_carries_both_paths`.

```rust
    #[test]
    fn extract_features_refuses_empty_arguments() {
        let error = build_request(&Command::ExtractFeatures {
            project_dir: PathBuf::new(),
            audio: PathBuf::from("project/assets/audio_16k_mono.wav"),
        })
        .expect_err("an empty project directory is refused");
        assert_eq!(error, "工程目录不能为空。");

        let error = build_request(&Command::ExtractFeatures {
            project_dir: PathBuf::from("project"),
            audio: PathBuf::new(),
        })
        .expect_err("an empty audio file is refused");
        assert_eq!(error, "音频文件不能为空。");
    }

    #[test]
    fn extract_features_carries_both_paths() {
        let request = build_request(&Command::ExtractFeatures {
            project_dir: PathBuf::from("project"),
            audio: PathBuf::from("project/assets/audio_16k_mono.wav"),
        })
        .expect("both paths are accepted")
        .expect("extract-features needs a task");
        let Request::ExtractFeatures(params) = request else {
            panic!("extract-features must build an ExtractFeatures request");
        };
        assert_eq!(params.project_dir, PathBuf::from("project"));
        assert_eq!(
            params.audio,
            PathBuf::from("project/assets/audio_16k_mono.wav")
        );
    }
```

Then append one test to `rust/crates/feathertalk-cli/tests/cli.rs`, after `an_unsupported_extract_frames_names_the_model_variables`.

```rust
#[test]
fn an_unsupported_extract_features_names_the_hubert_variable() {
    // The fake worker advertises `validate_project` alone, so the client's
    // capability gate answers before any task starts.
    let output = run("only-validate", &["extract-features", "p", "audio.wav"]);
    assert_eq!(code(&output), 3);
    let text = stderr(&output);
    assert!(text.contains("extract_features"), "{text}");
    assert!(text.contains("FEATHERTALK_WORKER_HUBERT_DIR"), "{text}");
}
```

The project directory is the single letter `p` because nothing ever reads it: the client refuses the request from the ready frame alone. A longer name pushes the `run` call past rustfmt's 60-column call width and splits it over four lines, which buys nothing.

- [ ] **Step 2: Run the test and watch it fail**

Run: `cargo test -p feathertalk-cli`

Expected: FAIL to compile with `error[E0599]: no variant or associated item named ExtractFeatures found for enum cli::Command`, once for each of the three `Command::ExtractFeatures` literals in the two unit tests. Only the library's test target is broken -- the `#[cfg(test)]` module is not compiled into the binary -- but cargo stops at the first compile error, so no test in the crate runs.

Were `tests/cli.rs` run on its own, it would fail for a subtler reason worth knowing: `main.rs` turns every clap error except help and version into exit 3, so the exit-code assertion would pass, and the failure would land on `assert!(text.contains("extract_features"))`, because clap's unrecognised-subcommand message spells the name with a hyphen.

- [ ] **Step 3: Implement**

In `rust/crates/feathertalk-cli/src/cli.rs`, extend the doc comment above `Command` so the list stays complete:

```rust
/// The task commands, kebab-cased by clap: `validate-project`, `probe-media`,
/// `normalize-media`, `extract-frames`, `extract-features`, `capabilities`.
```

then add the variant after `ExtractFrames` and before `Capabilities`, so the order the help text prints follows the order of the pipeline:

```rust
    /// 提取音频的 FeatherHuBERT 特征
    ExtractFeatures {
        /// 工程目录
        project_dir: PathBuf,
        /// 已归一化的 16kHz 单声道音频，位于工程目录的 assets 下
        audio: PathBuf,
    },
```

In `rust/crates/feathertalk-cli/src/run.rs`, add `ExtractFeaturesParams` to the `feathertalk_domain` import -- rustfmt refills the block as

```rust
use feathertalk_domain::{
    ExtractFeaturesParams, ExtractFramesParams, NormalizeMediaParams, ProbeMediaParams,
    ProjectDirParams, Request, TaskId,
};
```

-- and add the arm to `build_request`, after `Command::ExtractFrames`:

```rust
        Command::ExtractFeatures { project_dir, audio } => {
            reject_empty(project_dir, "工程目录")?;
            reject_empty(audio, "音频文件")?;
            Ok(Some(Request::ExtractFeatures(ExtractFeaturesParams {
                project_dir: project_dir.clone(),
                audio: audio.clone(),
            })))
        }
```

In `rust/crates/feathertalk-cli/src/render.rs`, add the constant beside `ENV_WORKER_SCRFD_DIR` and `ENV_WORKER_PFLD_DIR`:

```rust
/// The worker's variable for the FeatherHuBERT package directory, a literal for
/// the same reason: `feathertalk-worker`'s `ENV_HUBERT_DIR` is the source of
/// truth for this name.
const ENV_WORKER_HUBERT_DIR: &str = "FEATHERTALK_WORKER_HUBERT_DIR";
```

and a third branch on the `UnsupportedCommand` arm of `render_client_error`, after the `extract_frames` one:

```rust
            } else if *requested == "extract_features" {
                text.push_str(&format!(
                    "\n{requested} 需要 FeatherHuBERT 特征模型。请用环境变量 \
                     {ENV_WORKER_HUBERT_DIR} 指定模型包目录的完整路径。"
                ));
            }
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `cargo test -p feathertalk-cli`, expecting the inline suite in `run.rs` to grow from 6 tests to 8 and `tests/cli.rs` from 12 to 13. Then `cargo clippy -p feathertalk-cli --all-targets -- -D warnings`.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-cli/src/cli.rs rust/crates/feathertalk-cli/src/render.rs rust/crates/feathertalk-cli/src/run.rs rust/crates/feathertalk-cli/tests/cli.rs
git commit -m "feat(cli): add the extract-features subcommand"
```

---

### Task 12: Extract features end to end

**Files:**
- Test: `rust/crates/feathertalk-cli/tests/real_worker.rs`

**Interfaces:**
- Consumes: the `extract-features` subcommand from Task 11, the handshake gate from Task 10, the payload `feature_to_json` builds in Task 4, and `FEATHERTALK_WORKER_HUBERT_DIR` from Task 2.
- Produces: nothing a later task consumes -- this is the slice's only end-to-end proof.

**Why:** every unit in this slice is tested against a fake -- a fake chunk encoder, a fake worker binary, a wav written byte by byte inside the test. That is what keeps the suite fast and hermetic, and it is also what leaves one question open: whether the real ffmpeg, a package built by `feathertalk-model-package` from the real checkpoint, the 256/2/8/1024/0.0 configuration inside it, and the CLI's own argument order agree with one another. One test answers it, and it has to be this one, because nothing smaller spans all four.

It skips instead of failing when anything it needs is absent, exactly as `a_real_clip_is_normalized_end_to_end` and `a_real_second_is_extracted_end_to_end` do. A test that fails because ffmpeg is not installed reports on the developer's machine rather than on the code, and a suite that cries wolf stops being read.

Two seconds is the whole budget. The demo clip runs a full minute, and a minute of audio would spend minutes inside the encoder to exercise the same code path the chunk plan already covers with synthetic samples. Two seconds is also arithmetically interesting in a way one second is not: 32 000 samples make 99 frames, an odd count, so the trim Task 6 relies on actually fires and the file that lands holds 98 tokens rather than 99. Asserting `tokens`, `dims`, `frame_count`, `bytes`, and the file's own length pins the header layout, the trim, and the write against numbers derived from the kernel and stride rather than copied from a previous run.

`--release` is not optional. The encoder runs on burn's NdArray backend, and in a debug build two seconds of audio is slow enough to read as a hang. The profile also decides which binary the test finds: `CARGO_BIN_EXE_feathertalk` resolves per profile and the worker is looked up beside it, so a release test needs a release worker.

- [ ] **Step 1: Write the failing test**

Append one test to the end of `rust/crates/feathertalk-cli/tests/real_worker.rs`, after `file_count`.

```rust
/// The whole feature extraction, and only when the real ffmpeg, a built
/// FeatherHuBERT package, and the demo clip are all present. Neither this
/// repository nor CI ships ffmpeg or the model package, so anything missing is a
/// skip rather than a failure, for the reason the two tests above give.
#[test]
fn real_audio_becomes_features_end_to_end() {
    let Some(worker) = worker_or_skip("real_audio_becomes_features_end_to_end") else {
        return;
    };
    let (Some(ffmpeg), Some(hubert), Some(demo)) =
        (real_tool("FFMPEG"), real_dir("HUBERT_DIR"), demo_clip())
    else {
        println!(
            "skipping real_audio_becomes_features_end_to_end: it needs \
             FEATHERTALK_WORKER_FFMPEG, FEATHERTALK_WORKER_HUBERT_DIR, and \
             demo/feathertalk_demo_latest_188.mp4"
        );
        return;
    };
    let project = TempDir::new().expect("a temporary directory is available");
    let assets = project.path().join("assets");
    std::fs::create_dir_all(&assets).expect("the assets directory is writable");
    // Admission only asks that the manifest exists; reading it is
    // `validate-project`'s job, and this command runs before a project has the
    // assets that validation demands.
    std::fs::write(project.path().join("project.json"), "{}")
        .expect("the temporary manifest is writable");
    let audio = assets.join("audio_16k_mono.wav");
    cut_audio(&ffmpeg, &demo, &audio);

    let project_arg = project.path().to_string_lossy().into_owned();
    let audio_arg = audio.to_string_lossy().into_owned();
    let hubert_arg = hubert.to_string_lossy().into_owned();
    let output = run(
        &worker,
        &["extract-features", &project_arg, &audio_arg],
        &[("FEATHERTALK_WORKER_HUBERT_DIR", hubert_arg.as_str())],
    );
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));

    // Two seconds at 16 kHz is 32 000 samples, which one chunk covers whole. The
    // 400-sample kernel and the 320-sample stride turn them into
    // `(32_000 - 80) / 320` = 99 frames, the odd-token trim drops one, and 98
    // tokens of 1024 dimensions behind a 44-byte header make
    // `44 + 98 * 1024 * 4` = 401_452 bytes. If a future resampler hands over a
    // different sample count, recompute the four numbers the same way rather
    // than adjusting them by hand.
    let features_dir = assets.join("features");
    let features = features_dir.join("feather_hubert.f32");
    let result: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("stdout is exactly one JSON document");
    assert_eq!(result["output_dir"], features_dir.display().to_string());
    assert_eq!(result["feature_file"], features.display().to_string());
    assert_eq!(result["tokens"], 98);
    assert_eq!(result["dims"], 1024);
    assert_eq!(result["frame_count"], 49);
    assert_eq!(result["bytes"], 401_452);
    assert_eq!(
        std::fs::metadata(&features)
            .expect("the feature file is readable")
            .len(),
        401_452
    );
    // Exactly one file, and none of the bookkeeping this command stays out of.
    assert_eq!(file_count(&features_dir), 1);
    assert!(!assets.join("assets.json").exists());
    assert!(!assets.join("quality.json").exists());

    // The digest in the payload is the package's own, which is what lets a later
    // run decide whether these features still match the encoder.
    let manifest = std::fs::read_to_string(hubert.join("manifest.json"))
        .expect("the package manifest is readable");
    let manifest: serde_json::Value =
        serde_json::from_str(&manifest).expect("the manifest is JSON");
    assert_eq!(result["model_sha256"], manifest["model"]["sha256"]);

    let narration = stderr(&output);
    assert!(narration.contains("正在提取特征"), "{narration}");
    assert!(narration.contains("进度 1/1"), "{narration}");
}
```

Every helper it calls already exists: `worker_or_skip`, `run`, `code`, `stdout`, `stderr`, `real_tool`, `real_dir`, `demo_clip`, and `file_count`. `cut_audio` does not, which is what makes the test fail.

- [ ] **Step 2: Run the test and watch it fail**

Run: `cargo test --release -p feathertalk-cli --test real_worker`

Expected: FAIL to compile with `error[E0425]: cannot find function cut_audio in this scope`, pointing at the one call site. No other name in the test is new.

- [ ] **Step 3: Implement**

Add the helper after the test. No import changes: `Path`, `Command`, and `TempDir` are already in scope.

```rust
/// Two seconds of the demo video's audio, decoded into the one shape the reader
/// admits -- 16 kHz, mono, 16-bit PCM -- and written under the name the pipeline
/// expects. No offset is needed: unlike a video frame, whose face score decides
/// whether it is usable, any two seconds of this clip's audio extract alike.
fn cut_audio(ffmpeg: &Path, demo: &Path, audio: &Path) {
    let output = Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-y"])
        .arg("-i")
        .arg(demo)
        .args([
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-t",
            "2",
        ])
        .arg(audio)
        .output()
        .expect("ffmpeg runs");
    assert!(
        output.status.success(),
        "ffmpeg could not cut the audio: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
```

The test also needs a package to point at, and the repository ships none. Build one from the checkpoint the user placed under `demo/`, which is 40 436 613 bytes and hashes to `58df96af118d75d7f69da441e1f3960096f28dda637a4e8f4265f108d27aeb52`:

```bash
cargo run -p feathertalk-model-package -- feather-hubert \
  --source demo/kanghui_training_video_featherhubert_188_latest/feather_hubert_188_latest_99.pth \
  --licenses <path to the bundle below> \
  --destination <a directory outside the repository, e.g. under the temp directory> \
  --created-at 2026-09-03T00:00:00Z \
  --minimum-app-version 0.1.0
```

A synthetic license bundle is acceptable for a local package, and there is precedent for both the shape and the wording: `rust/tests/fixtures/vgg19/LICENSES.local-parity.json` and the bundle `crates/feathertalk-export/tests/feather_hubert_real.rs` writes for the same checkpoint.

```json
{
  "schema_version": 1,
  "entries": [
    {
      "component": "user-supplied FeatherHuBERT checkpoint",
      "license_id": "LicenseRef-User-Supplied-Unreviewed",
      "source_url": "https://example.invalid/local-conversion-record",
      "notice": "Local end-to-end testing only; not redistribution approval."
    }
  ]
}
```

Write the bundle outside the repository as well, next to the destination. The finished package directory holds exactly `LICENSES.json`, `manifest.json`, and `model.safetensors`, which is the entry set `validate_package_directory` insists on -- an extra file in that directory is a load failure, so nothing else may be dropped there.

- [ ] **Step 4: Run the test and watch it pass**

First `cargo build --release -p feathertalk-worker`, because the test looks for the worker beside the release CLI. Then, with `FEATHERTALK_REQUIRE_E2E=1`, `FEATHERTALK_WORKER_FFMPEG` pointing at `D:\environment\ffmpeg\bin\ffmpeg.exe`, and `FEATHERTALK_WORKER_HUBERT_DIR` pointing at the package directory:

Run: `cargo test --release -p feathertalk-cli --test real_worker -- --nocapture`, expecting 8 passed and the line `test real_audio_becomes_features_end_to_end ... ok`.

Both paths must be absolute. `real_dir` accepts a directory only if `is_dir` holds from the process's own working directory, and `real_tool` behaves the same way, so a relative path makes the test skip while still reporting success -- which is why `--nocapture` matters here: confirm no `skipping` line was printed. If the assertions on `tokens` or `bytes` fail because ffmpeg's resampler produced a different sample count, recompute the numbers from the kernel and stride rather than pasting in whatever the run reported.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/feathertalk-cli/tests/real_worker.rs
git commit -m "test(cli): extract features end to end"
```

---

### Task 13: Close the slice with the workspace gates

**Files:**
- No file changes, unless a gate demands one.

**Interfaces:**
- Consumes: every task in this slice.
- Produces: the evidence that the slice is finished. Nothing depends on it.

**Why:** each of the twelve tasks ran only its own crate's tests, which is what keeps the loop short -- and that is exactly why the workspace has to be run once at the end. Two crates outside the worker changed shape: `feathertalk-audio` gained a reader and a dozen error variants, and `feathertalk-export` made a manifest reader public. Their neighbours are compiled by test binaries no per-task command touched. Clippy over `--all-targets` also lints test code with `-D warnings`, which `cargo test` compiles but never judges.

This task carries no test-first cycle, because it adds no behaviour: the gates are its steps. Any edit a gate forces is mechanical -- a reformat, a lint fix -- and belongs in a single commit at the end rather than folded back into a finished task.

The list mirrors the final gate in the Global Constraints above; `cargo check` is absent only because Gate 2 compiles strictly more than it does.

- [ ] **Gate 1: `cargo fmt --all -- --check`**

Expected: no output and exit 0. Every code fence in this plan was shaped by `rustfmt --edition 2024`, so a failure means a hand-edit slipped past it.

- [ ] **Gate 2: `cargo clippy --workspace --all-targets -- -D warnings`**

Expected: no warnings. This slice adds no suppression at all, unlike the `extract_frames` one; if clippy demands an `#[allow]` here, that is a finding worth reporting rather than a formality, because it means a function this plan judged simple is not.

- [ ] **Gate 3: `cargo test --workspace --all-targets`**

Expected: 0 failures, in roughly 45 minutes. The last full run before this slice was 185 test binaries, 909 passed, 0 failed, 13 ignored; the new counts must be higher, never lower. `feathertalk-frame-adapters` alone accounts for about 11 of those minutes, so a long silence is the fixture-backed pipeline test, not a hang.

- [ ] **Gate 4: the gated end-to-end**

Run Task 12's Step 4 command again, with `FEATHERTALK_REQUIRE_E2E=1` and the two absolute paths in the environment. Gate 3 cannot stand in for it: without those variables the test skips and still reports success, so this is the only gate that proves the real ffmpeg and a real package agree with the code.

- [ ] **Gate 5: `git diff --check`, then `git status -sb`**

Expected: no whitespace errors, a clean tree, and `demo/kanghui_training_video_featherhubert_188_latest/` still untracked. Confirm no `.jpg`, `.mp4`, `.wav`, `.f32`, or `.safetensors` was staged along the way -- `.gitignore` re-includes `demo/*.jpg` and `demo/*.mp4`, so a stray artefact written under `demo/` would not have been ignored, and the package built in Task 12 belongs outside the repository entirely.

If a gate forces an edit, stage the touched paths and commit them as `chore: satisfy the workspace lints for the extract-features slice`. If every gate passes untouched, the slice ends at Task 12's commit and this task adds none.
