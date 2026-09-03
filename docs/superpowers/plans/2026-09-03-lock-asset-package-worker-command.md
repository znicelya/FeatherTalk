# Lock Asset Package Worker Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `lock_asset_package` worker command, plus the CLI subcommand that drives it, that verifies a finished asset package and writes the locked `assets/assets.json` manifest.

**Architecture:** The command lives entirely in `feathertalk-worker`. An admission phase reads `assets/quality.json` and the feature file and refuses anything that is not ready; a verification phase walks every frame and landmark file to learn the frame geometry and prove the package is structurally whole; a commit phase hands the fitted feature matrix to `feathertalk_audio::commit_feature_artifact`, which rewrites the feature file and writes the locked manifest atomically. Four leaf helpers are added below the worker — JPEG header dimensions in `feathertalk-image`, a geometry probe in `feathertalk-frame-adapters`, a landmark-file reader in `feathertalk-frame-pipeline`, and token fitting in `feathertalk-audio` — so the worker only orchestrates and never reimplements file parsing.

**Tech Stack:** Rust 2024, `jpeg-decoder`, `serde_json`, `thiserror`, `clap`, `tempfile`, and the existing FeatherTalk worker/CLI JSON-lines protocol.

**Design:** docs/superpowers/specs/2026-09-03-lock-asset-package-worker-command-design.md

## Global Constraints

- Run every `cargo`, `rustfmt`, and `clippy` command from `E:\workspace\github\FeatherTalk\rust`; run every `git` command from the repository root `E:\workspace\github\FeatherTalk`.
- The command is gated on `config.features().is_some()`. No new environment variable is introduced; `FEATHERTALK_WORKER_HUBERT_DIR` is the only knob involved.
- `TaskStage::Preparing` is the only stage this command reports, and it is also the `FAILURE_STAGE` already used by `error_map.rs`. No new `TaskStage` variant, no new `Capabilities` field.
- Progress is reported once per verified frame with `total = Some(report.frame_count())`, with no throttling; the initial `reporter.report(TaskStage::Preparing, None)` still comes first.
- Cancellation is checked once after admission and once before each frame. The commit phase is deliberately not cancellable.
- Verification is structural, not byte-identical: no frame or landmark file is re-hashed and `frame_bytes` is never compared, because operators are allowed to hand-edit frames before locking.
- `landmark_model_sha256` is `feathertalk_pfld::PFLD_MODEL_SHA256`; `feature_model_sha256` is `read_package_manifest(features.hubert_dir())?.model.sha256` and means only "the encoder installed in the locking worker".
- `MAX_TOKEN_FIT_DELTA: i64 = 50` (one second at 25 fps). A larger gap between `2 * frame_count` and the feature file's token count is a rejected request, never a silent fit.
- Fixed limits: `LANDMARK_POINTS: usize = 110`, `MAX_LANDMARK_FILE_BYTES: u64 = 8 * 1024`, `JPEG_HEADER_PROBE_BYTES: u64 = 64 * 1024`; frame size stays capped by `feathertalk_frame_pipeline::MAX_FRAME_BYTES` (16 MiB).
- All three new `PipelineError` variants map to `ErrorCode::MediaInvalid`. `AudioError::CommitRollbackFailed` keeps mapping to `WORKER_CRASHED`.
- The locked manifest's fixed fields (schema version, 25 fps, 16 000 Hz, mono, `feather_hubert`, `[frame_count, 2, 1024]`) are produced by `feathertalk_audio::commit.rs::locked_manifest`. This slice never hand-writes them.
- Chinese text appears only in user-facing string literals (error summaries, CLI help). Code, comments, commit messages, and this plan are English.
- No `unwrap`, `expect`, or `panic!` outside test code.
- Run `rustfmt --edition 2024 --check <files>` after each edit and `cargo clippy -p <crate> --all-targets -- -D warnings` before each commit. rustfmt defaults apply: `max_width` 100, `fn_call_width` 60, `struct_lit_width` 18, `array_width` 60, `chain_width` 60.
- Never stage binary media (`.jpg`, `.mp4`, `.wav`, `.f32`, `.safetensors`), and never stage the untracked `demo/kanghui_training_video_featherhubert_188_latest/` directory.
- Every task leaves the tree green: the task's own test command plus `cargo check` must pass before its commit.
- Commit after each task. Stage explicit paths, never `git add .`. Never push to `origin`.
- Task 13 is the final gate: `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace --all-targets`, the gated release end-to-end test, and `git diff --check`.

## File Structure

- `rust/crates/feathertalk-image/src/jpeg.rs` — JPEG decoding; gains the shared header reader and the dimensions-only entry point.
- `rust/crates/feathertalk-image/src/lib.rs` — crate exports.
- `rust/crates/feathertalk-image/tests/jpeg_decode.rs` — header and decode behaviour.
- `rust/crates/feathertalk-frame-pipeline/src/error.rs` — `PipelineError`; gains the three asset-lock failures.
- `rust/crates/feathertalk-frame-pipeline/src/landmark.rs` *(new)* — read one `.lms` file back and validate it against the frame geometry.
- `rust/crates/feathertalk-frame-pipeline/src/evaluate.rs` — landmark serialization; switches its literal `110`s to the shared constant.
- `rust/crates/feathertalk-frame-pipeline/src/lib.rs` — module list and exports.
- `rust/crates/feathertalk-frame-pipeline/tests/landmarks.rs` *(new)* — landmark reader behaviour.
- `rust/crates/feathertalk-frame-adapters/src/geometry.rs` *(new)* — translate a JPEG header read into a `PipelineError`.
- `rust/crates/feathertalk-frame-adapters/src/lib.rs` — module list and exports.
- `rust/crates/feathertalk-frame-adapters/tests/geometry.rs` *(new)* — geometry probe against the checked-in fixtures.
- `rust/crates/feathertalk-audio/src/stitch.rs` — token fitting shared with `extract_long_audio`.
- `rust/crates/feathertalk-audio/src/lib.rs` — `FeatureMatrix::into_values` and the new export.
- `rust/crates/feathertalk-audio/tests/stitching.rs` — fitting behaviour.
- `rust/crates/feathertalk-worker/src/lock_result.rs` *(new)* — the command's JSON result payload.
- `rust/crates/feathertalk-worker/tests/lock_result.rs` *(new)* — payload shape.
- `rust/crates/feathertalk-worker/src/asset_scan.rs` *(new)* — frame verification, geometry agreement, landmark validation, and directory file counts, with inline unit tests.
- `rust/crates/feathertalk-worker/src/lock_asset_package.rs` *(new)* — admission checks and command orchestration.
- `rust/crates/feathertalk-worker/tests/lock_asset_package.rs` *(new)* — command behaviour end to end inside the worker.
- `rust/crates/feathertalk-worker/src/error_map.rs` — error code and summary mapping.
- `rust/crates/feathertalk-worker/tests/error_mapping.rs` — exhaustive mapping table.
- `rust/crates/feathertalk-worker/src/commands.rs` — request dispatch arm.
- `rust/crates/feathertalk-worker/tests/commands.rs` — dispatch behaviour.
- `rust/crates/feathertalk-worker/src/handshake.rs` — advertise the command in `supported_commands`.
- `rust/crates/feathertalk-worker/src/runtime.rs` — the rejection reason for an unconfigured worker.
- `rust/crates/feathertalk-worker/tests/handshake.rs`, `rust/crates/feathertalk-worker/tests/runtime.rs` — advertisement and rejection.
- `rust/crates/feathertalk-worker/src/lib.rs` — module list, exports, module documentation.
- `rust/crates/feathertalk-worker/Cargo.toml`, `rust/Cargo.lock` — promote `feathertalk-pfld` from a dev dependency to a dependency.
- `rust/crates/feathertalk-cli/src/cli.rs` — the `lock-asset-package` subcommand surface.
- `rust/crates/feathertalk-cli/src/run.rs` — argument validation and request construction.
- `rust/crates/feathertalk-cli/src/render.rs` — the unsupported-command hint.
- `rust/crates/feathertalk-cli/tests/cli.rs` — CLI behaviour against the fake worker.
- `rust/crates/feathertalk-cli/tests/real_worker.rs` — the gated end-to-end lock.

---

### Task 1: JPEG Header Dimensions

**Files:**
- Modify: `rust/crates/feathertalk-image/src/jpeg.rs`
- Modify: `rust/crates/feathertalk-image/src/lib.rs`
- Test: `rust/crates/feathertalk-image/tests/jpeg_decode.rs`

**Interfaces:**
- Consumes: nothing. This is the deepest leaf in the slice.
- Produces: `pub fn jpeg_dimensions(bytes: &[u8]) -> Result<(u32, u32), ImageError>`, re-exported as `feathertalk_image::jpeg_dimensions`. Task 3 is its only caller.

**Why first:** The lock has to write `frame_width` and `frame_height` into `assets.json`, and the quality report does not record them. Decoding every frame to learn two numbers would cost a full baseline JPEG decode per frame — for a 49-frame clip at 1280x720 that is 34 million pixels of pointless work. `jpeg-decoder` can stop after the SOF header, so the geometry costs a header parse. The risk in adding a second entry point is that it drifts from `decode_jpeg`: the two could disagree about which sizes are legal. Extracting the header read into one private function both callers share removes that risk structurally instead of by convention, and it must happen before anything above it can depend on the geometry.

- [ ] **Step 1: Write the failing test**

  Append three tests to `rust/crates/feathertalk-image/tests/jpeg_decode.rs`. The file already owns a `jpeg_header(width: u16, height: u16) -> Vec<u8>` helper that builds a minimal SOI + baseline SOF0 byte string; reuse it, do not add a second builder. Extend the existing `use feathertalk_image::{...}` import to include `jpeg_dimensions`, keeping the file's ASCII ordering (CamelCase names first, then snake_case): `use feathertalk_image::{ImageError, decode_jpeg, jpeg_dimensions};`.

  Do not add a zero-dimension case. `jpeg-decoder` rejects a SOF with a zero extent itself, so such a test would assert the dependency's behaviour rather than ours.

  ```rust
  #[test]
  fn the_header_reader_agrees_with_the_decoder() {
      let bytes = jpeg_header(640, 480);
      assert_eq!(jpeg_dimensions(&bytes).unwrap(), (640, 480));
      // `decode_jpeg` rejects the image on the pixel count it read from the same
      // shared header, so that count is the product of the dimensions
      // `jpeg_dimensions` returned. This is what pins the two paths together.
      let error = decode_jpeg(&bytes, 0).unwrap_err();
      let ImageError::FrameTooLarge { pixels, max_pixels } = error else {
          panic!("a zero budget must reject the image: {error:?}");
      };
      assert_eq!(pixels, 640 * 480);
      assert_eq!(max_pixels, 0);
  }

  #[test]
  fn a_header_prefix_is_enough_and_a_truncated_one_is_not() {
      let bytes = jpeg_header(1280, 720);
      let mut padded = bytes.clone();
      padded.extend_from_slice(&[0x00; 4096]);
      assert_eq!(jpeg_dimensions(&padded).unwrap(), (1280, 720));
      let error = jpeg_dimensions(&bytes[..6]).unwrap_err();
      assert!(matches!(error, ImageError::JpegDecode { .. }), "{error:?}");
  }

  #[test]
  fn garbage_bytes_are_not_a_jpeg_header() {
      let error = jpeg_dimensions(b"not a jpeg at all").unwrap_err();
      let ImageError::JpegDecode { message } = error else {
          panic!("garbage must be a decode error: {error:?}");
      };
      assert!(message.contains("SOI"), "{message}");
  }
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `cargo test -p feathertalk-image --test jpeg_decode`

  Expected: FAIL to compile with `error[E0432]: unresolved import` naming `feathertalk_image::jpeg_dimensions` (and `cannot find function jpeg_dimensions in this scope` at each call site).

- [ ] **Step 3: Write minimal implementation**

  In `rust/crates/feathertalk-image/src/jpeg.rs`, widen the `std::io` import to `use std::io::{Cursor, Read};` — `Read` is needed for the generic bound — and insert both functions above `decode_jpeg`:

  ```rust
  /// Read the SOF header and reject the degenerate sizes.
  ///
  /// Split out of `decode_jpeg` so that `jpeg_dimensions` cannot drift from it.
  /// The decoder is borrowed rather than owned because `decode_jpeg` keeps using
  /// it afterwards, and the pixel format comes back for the same reason.
  fn read_header<R: Read>(decoder: &mut Decoder<R>) -> Result<(u32, u32, PixelFormat), ImageError> {
      decoder
          .read_info()
          .map_err(|error| ImageError::JpegDecode {
              message: error.to_string(),
          })?;
      let info = decoder.info().ok_or_else(|| ImageError::JpegDecode {
          message: "JPEG header carried no image information".to_owned(),
      })?;
      let width = u32::from(info.width);
      let height = u32::from(info.height);
      if width == 0 || height == 0 {
          return Err(ImageError::InvalidDimensions { width, height });
      }
      Ok((width, height, info.pixel_format))
  }

  /// Read a JPEG's pixel dimensions without decoding a single scan.
  ///
  /// Allocates nothing beyond the decoder's header state, so it takes no pixel
  /// budget: a caller that cares about size limits owns that policy. The asset
  /// lock uses it to learn the frame geometry a quality report does not record.
  pub fn jpeg_dimensions(bytes: &[u8]) -> Result<(u32, u32), ImageError> {
      let mut decoder = Decoder::new(Cursor::new(bytes));
      let (width, height, _) = read_header(&mut decoder)?;
      Ok((width, height))
  }
  ```

  Then rewrite the head of `decode_jpeg` to use the shared reader. Delete its `read_info()` call, its `info()` lookup, its zero-extent check, its `let width = ...; let height = ...;` pair, and its `let pixel_format = info.pixel_format;` line, and replace all of them with:

  ```rust
      let mut decoder = Decoder::new(Cursor::new(bytes));
      let (width, height, pixel_format) = read_header(&mut decoder)?;
  ```

  Everything below that line — the `pixels` product, the `max_pixels` comparison producing `ImageError::FrameTooLarge`, the `bgr_len` computation, `set_max_decoding_buffer_size`, the `decode()` call, and the pixel-format match — stays exactly as it is. `info` must no longer be referenced after this change; if the compiler reports it as unused, the deletion was incomplete.

  In `rust/crates/feathertalk-image/src/lib.rs`, the JPEG re-export currently names one function. Make it name both: `pub use jpeg::{decode_jpeg, jpeg_dimensions};`.

- [ ] **Step 4: Run test to verify it passes**

  Run: `cargo test -p feathertalk-image --test jpeg_decode`

  Expected: PASS — 10 tests (the 7 that existed plus the 3 added here), 0 failed. Then `rustfmt --edition 2024 --check crates/feathertalk-image/src/jpeg.rs crates/feathertalk-image/src/lib.rs crates/feathertalk-image/tests/jpeg_decode.rs` and `cargo clippy -p feathertalk-image --all-targets -- -D warnings`, both clean.

- [ ] **Step 5: Commit**

  ```powershell
  git add rust/crates/feathertalk-image/src/jpeg.rs rust/crates/feathertalk-image/src/lib.rs rust/crates/feathertalk-image/tests/jpeg_decode.rs
  git commit -m "feat(image): read JPEG dimensions from the header"
  ```

---

### Task 2: Name the Asset-Lock Frame Failures

**Files:**
- Modify: `rust/crates/feathertalk-frame-pipeline/src/error.rs`
- Modify: `rust/crates/feathertalk-worker/src/error_map.rs`
- Test: `rust/crates/feathertalk-worker/tests/error_mapping.rs`

**Interfaces:**
- Consumes: nothing.
- Produces: three `PipelineError` variants that Tasks 3, 4, and 7 construct:
  - `PipelineError::FrameUndecodable { path: PathBuf, message: String }`
  - `PipelineError::LandmarkNotRegular { path: PathBuf }`
  - `PipelineError::InvalidLandmark { path: PathBuf, message: String }`

  All three resolve to `ErrorCode::MediaInvalid` through the existing `pipeline_task_error`.

**Why now:** `pipeline_error_code` and `pipeline_summary` in `error_map.rs` match `PipelineError` exhaustively with no wildcard arm. Adding a variant therefore breaks the worker's build until the mapping is extended, which means any task that introduces a variant and its producer at the same time cannot compile its own test. Landing the vocabulary first — variants plus mapping plus mapping test — keeps every later task to a single concern and keeps the tree green at each commit.

- [ ] **Step 1: Write the failing test**

  `rust/crates/feathertalk-worker/tests/error_mapping.rs` already contains `every_pipeline_error_maps_to_a_code_and_a_valid_payload`, whose body starts with a `let cases = vec![(PipelineError::..., ErrorCode::...), ...];`. Append these three entries at the end of that vector, using the file's existing `path()` helper:

  ```rust
          (
              PipelineError::FrameUndecodable {
                  path: path(),
                  message: "no SOI marker".to_owned(),
              },
              ErrorCode::MediaInvalid,
          ),
          (
              PipelineError::LandmarkNotRegular { path: path() },
              ErrorCode::MediaInvalid,
          ),
          (
              PipelineError::InvalidLandmark {
                  path: path(),
                  message: "expected 110 lines, found 109".to_owned(),
              },
              ErrorCode::MediaInvalid,
          ),
  ```

  Then add a new test that pins the two operator-facing summaries, so a later refactor cannot quietly reword them:

  ```rust
  #[test]
  fn the_asset_lock_failures_read_as_media_problems() {
      let undecodable = pipeline_task_error(&PipelineError::FrameUndecodable {
          path: path(),
          message: "no SOI marker".to_owned(),
      });
      assert_eq!(undecodable.summary, "素材帧无法解码");
      assert!(undecodable.detail.contains("no SOI marker"), "{}", undecodable.detail);

      let not_regular = pipeline_task_error(&PipelineError::LandmarkNotRegular { path: path() });
      assert_eq!(not_regular.summary, "关键点文件不可用");

      let malformed = pipeline_task_error(&PipelineError::InvalidLandmark {
          path: path(),
          message: "expected 110 lines, found 109".to_owned(),
      });
      assert_eq!(malformed.summary, "关键点文件不可用");
      malformed.validate().unwrap();
  }
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `cargo test -p feathertalk-worker --test error_mapping`

  Expected: FAIL to compile with `error[E0599]: no variant or associated item named FrameUndecodable found for enum PipelineError` (and the same for `LandmarkNotRegular` and `InvalidLandmark`).

- [ ] **Step 3: Write minimal implementation**

  In `rust/crates/feathertalk-frame-pipeline/src/error.rs`, insert the three variants directly after `FrameTooLarge`, so the frame-related variants stay grouped. The file spells the type inline as `std::path::PathBuf` rather than importing it; match that.

  ```rust
      /// A frame file exists but its JPEG header cannot be read.
      #[error("frame is not a decodable JPEG: {path} ({message})")]
      FrameUndecodable {
          path: std::path::PathBuf,
          message: String,
      },
      /// A landmark file is a symlink, a directory, or another non-regular entry.
      #[error("landmark file is not a regular non-symlink file: {path}")]
      LandmarkNotRegular { path: std::path::PathBuf },
      /// A landmark file's bytes are not what `serialize_landmarks` writes.
      #[error("landmark file is malformed: {path} ({message})")]
      InvalidLandmark {
          path: std::path::PathBuf,
          message: String,
      },
  ```

  In `rust/crates/feathertalk-worker/src/error_map.rs`, extend `pipeline_error_code`. Its `ErrorCode::MediaInvalid` group is a run of `|`-joined patterns; append the three new ones to that run:

  ```rust
          | PipelineError::FrameUndecodable { .. }
          | PipelineError::LandmarkNotRegular { .. }
          | PipelineError::InvalidLandmark { .. }
  ```

  In `pipeline_summary`, add two arms immediately after the existing `PipelineError::FrameMissing { .. } | ... | PipelineError::FrameTooLarge { .. } => "抽出的帧不可用"` arm. A broken JPEG and a broken landmark file are different repairs for the operator, so they do not share the frame wording:

  ```rust
          PipelineError::FrameUndecodable { .. } => "素材帧无法解码",
          PipelineError::LandmarkNotRegular { .. } | PipelineError::InvalidLandmark { .. } => {
              "关键点文件不可用"
          }
  ```

  Nothing else changes: `pipeline_task_error` already fills `detail` from `error.to_string()`, `stage` from `FAILURE_STAGE`, and the recovery hint from the code.

- [ ] **Step 4: Run test to verify it passes**

  Run: `cargo test -p feathertalk-worker --test error_mapping`

  Expected: PASS, 0 failed. Then `cargo check -p feathertalk-frame-pipeline -p feathertalk-worker` (proves no other exhaustive match over `PipelineError` was missed), `rustfmt --edition 2024 --check crates/feathertalk-frame-pipeline/src/error.rs crates/feathertalk-worker/src/error_map.rs crates/feathertalk-worker/tests/error_mapping.rs`, and `cargo clippy -p feathertalk-worker --all-targets -- -D warnings`.

- [ ] **Step 5: Commit**

  ```powershell
  git add rust/crates/feathertalk-frame-pipeline/src/error.rs rust/crates/feathertalk-worker/src/error_map.rs rust/crates/feathertalk-worker/tests/error_mapping.rs
  git commit -m "feat(frame-pipeline): name the asset-lock frame failures"
  ```

---
  git add rust/crates/feathertalk-frame-pipeline/src/error.rs rust/crates/feathertalk-worker/src/error_map.rs rust/crates/feathertalk-worker/tests/error_mapping.rs
  git commit -m "feat(frame-pipeline): name the asset-lock frame failures"
  ```

---

### Task 3: Frame Geometry Probe

**Files:**
- Create: `rust/crates/feathertalk-frame-adapters/src/geometry.rs`
- Modify: `rust/crates/feathertalk-frame-adapters/src/lib.rs`
- Test: `rust/crates/feathertalk-frame-adapters/tests/geometry.rs`

**Interfaces:**
- Consumes: `feathertalk_image::jpeg_dimensions` (Task 1) and `PipelineError::FrameUndecodable` (Task 2).
- Produces: `pub fn probe_jpeg_geometry(path: &Path, bytes: &[u8]) -> Result<(u32, u32), PipelineError>`, re-exported as `feathertalk_frame_adapters::probe_jpeg_geometry`. Task 7 is its only caller.

**Why now:** `feathertalk-image` must not depend on `feathertalk-frame-pipeline` — it is the lower crate, and the image layer has its own error type on purpose. `feathertalk-frame-adapters` is the crate that already owns exactly this translation: `cache.rs::load` turns an `ImageError` into `PipelineError::Adapter { component: "jpeg" }`. Putting the geometry bridge here keeps the dependency direction intact and gives the worker a single function to call with no error mapping of its own. `FrameUndecodable` is used instead of the generic `Adapter` variant because the operator's repair is specific: replace that frame file.

- [ ] **Step 1: Write the failing test**

  Create `rust/crates/feathertalk-frame-adapters/tests/geometry.rs`. The crate already ships the fixture used here; `demo_frame_v1/frame.jpg` is 1280x720 and 157 768 bytes.

  ```rust
  //! Frame geometry read from the checked-in JPEG fixtures.

  use std::fs;
  use std::path::{Path, PathBuf};

  use feathertalk_frame_adapters::probe_jpeg_geometry;
  use feathertalk_frame_pipeline::PipelineError;

  fn fixture_frame() -> PathBuf {
      Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/demo_frame_v1/frame.jpg")
  }

  #[test]
  fn a_real_frame_reports_its_pixel_dimensions() {
      let path = fixture_frame();
      let bytes = fs::read(&path).expect("the demo frame fixture must be readable");
      assert_eq!(probe_jpeg_geometry(&path, &bytes).unwrap(), (1280, 720));
  }

  #[test]
  fn garbage_bytes_name_the_frame_that_is_broken() {
      let path = Path::new("assets/frames/000007.jpg");
      let error = probe_jpeg_geometry(path, b"not a jpeg at all").unwrap_err();
      let PipelineError::FrameUndecodable {
          path: reported,
          message,
      } = error
      else {
          panic!("garbage must be an undecodable frame: {error:?}");
      };
      assert_eq!(reported, path);
      assert!(!message.is_empty(), "the decoder's own message must survive");
  }

  #[test]
  fn a_truncated_frame_is_undecodable() {
      let path = fixture_frame();
      let bytes = fs::read(&path).expect("the demo frame fixture must be readable");
      let error = probe_jpeg_geometry(&path, &bytes[..4]).unwrap_err();
      assert!(
          matches!(error, PipelineError::FrameUndecodable { .. }),
          "{error:?}"
      );
  }
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `cargo test -p feathertalk-frame-adapters --test geometry`

  Expected: FAIL to compile with `error[E0432]: unresolved import feathertalk_frame_adapters::probe_jpeg_geometry`. Note that a cold build of this crate compiles `burn`; allow several minutes and do not mistake build time for a hang.

- [ ] **Step 3: Write minimal implementation**

  Create `rust/crates/feathertalk-frame-adapters/src/geometry.rs`:

  ```rust
  //! Frame geometry read from a JPEG header.

  use std::path::Path;

  use feathertalk_frame_pipeline::PipelineError;
  use feathertalk_image::jpeg_dimensions;

  /// Read a frame's pixel dimensions from its JPEG header.
  ///
  /// Pure over `bytes`: the caller owns the file read and its size cap, which is
  /// what lets the tests here run without a temporary directory. `path` is
  /// carried only so the error names the frame that is broken.
  pub fn probe_jpeg_geometry(path: &Path, bytes: &[u8]) -> Result<(u32, u32), PipelineError> {
      jpeg_dimensions(bytes).map_err(|error| PipelineError::FrameUndecodable {
          path: path.to_path_buf(),
          message: error.to_string(),
      })
  }
  ```

  In `rust/crates/feathertalk-frame-adapters/src/lib.rs`, add `mod geometry;` to the module list (after `decoder`, keeping the list alphabetical) and `pub use geometry::probe_jpeg_geometry;` after the existing `pub use decoder::JpegFrameDecoder;`.

  No `Cargo.toml` change is needed: `feathertalk-frame-pipeline` and `feathertalk-image` are both already dependencies of this crate.

- [ ] **Step 4: Run test to verify it passes**

  Run: `cargo test -p feathertalk-frame-adapters --test geometry`

  Expected: PASS — 3 tests, 0 failed. Then `rustfmt --edition 2024 --check crates/feathertalk-frame-adapters/src/geometry.rs crates/feathertalk-frame-adapters/src/lib.rs crates/feathertalk-frame-adapters/tests/geometry.rs` and `cargo clippy -p feathertalk-frame-adapters --all-targets -- -D warnings`.

- [ ] **Step 5: Commit**

  ```powershell
  git add rust/crates/feathertalk-frame-adapters/src/geometry.rs rust/crates/feathertalk-frame-adapters/src/lib.rs rust/crates/feathertalk-frame-adapters/tests/geometry.rs
  git commit -m "feat(frame-adapters): probe frame geometry from the JPEG header"
  ```

---

### Task 4: Read the Landmark Files Back

**Files:**
- Create: `rust/crates/feathertalk-frame-pipeline/src/landmark.rs`
- Modify: `rust/crates/feathertalk-frame-pipeline/src/lib.rs`
- Modify: `rust/crates/feathertalk-frame-pipeline/src/evaluate.rs`
- Test: `rust/crates/feathertalk-frame-pipeline/tests/landmarks.rs`

**Interfaces:**
- Consumes: `PipelineError::LandmarkNotRegular` and `PipelineError::InvalidLandmark` (Task 2).
- Produces, all re-exported from the crate root:
  - `pub const LANDMARK_POINTS: usize = 110;`
  - `pub const MAX_LANDMARK_FILE_BYTES: u64 = 8 * 1024;`
  - `pub fn read_landmark_file(path: &Path, frame_width: u32, frame_height: u32) -> Result<Vec<(i32, i32)>, PipelineError>`

  Task 7 calls `read_landmark_file` once per frame with the geometry it just probed.

**Why now:** The lock has to prove the landmark files are usable, and the only existing code that knows their format is the writer, `evaluate.rs::serialize_landmarks`. A reader placed next to it in the same crate can share the point-count constant, so the writer and reader cannot disagree about 110. It is a leaf like Tasks 1 and 3, so it lands before the worker code that composes them. The reader is deliberately strict — exact line count, exact single-space separator, mandatory trailing newline, no CR, every point inside the frame — because a lock is the last chance to catch a hand-edited or half-written file, and a permissive reader would let a broken package become an immutable one.

- [ ] **Step 1: Write the failing test**

  Create `rust/crates/feathertalk-frame-pipeline/tests/landmarks.rs`. `tempfile` is already a dev dependency of this crate (`tests/report.rs` uses it); if `TempDir` does not resolve, add `tempfile` to `[dev-dependencies]` and stage `rust/Cargo.lock` with this task's commit.

  ```rust
  //! Reading landmark files back out of an asset package.

  use std::fs;
  use std::path::PathBuf;

  use feathertalk_frame_pipeline::{
      LANDMARK_POINTS, MAX_LANDMARK_FILE_BYTES, PipelineError, read_landmark_file,
  };
  use tempfile::TempDir;

  const FRAME_WIDTH: u32 = 512;
  const FRAME_HEIGHT: u32 = 512;

  /// The exact shape `serialize_landmarks` writes: one `"{x} {y}"` per line,
  /// every line terminated. The largest point is (109, 218), well inside the
  /// 512x512 frame these tests declare.
  fn valid_text() -> String {
      let mut text = String::new();
      for index in 0..LANDMARK_POINTS {
          text.push_str(&format!("{} {}\n", index, index * 2));
      }
      text
  }

  fn write(dir: &TempDir, name: &str, text: &str) -> PathBuf {
      let path = dir.path().join(name);
      fs::write(&path, text).expect("the fixture must be writable");
      path
  }

  #[test]
  fn a_well_formed_file_reads_back_every_point() {
      let dir = TempDir::new().unwrap();
      let path = write(&dir, "000000.lms", &valid_text());
      let points = read_landmark_file(&path, FRAME_WIDTH, FRAME_HEIGHT).unwrap();
      assert_eq!(points.len(), LANDMARK_POINTS);
      assert_eq!(points[0], (0, 0));
      assert_eq!(points[109], (109, 218));
  }

  #[test]
  fn malformed_bodies_are_refused_one_by_one() {
      let valid = valid_text();
      let one_line_short: String = valid
          .lines()
          .take(LANDMARK_POINTS - 1)
          .map(|line| format!("{line}\n"))
          .collect();
      let cases = vec![
          ("one line short", one_line_short),
          ("one line long", format!("{valid}110 220\n")),
          ("no trailing newline", valid.trim_end().to_owned()),
          ("windows line endings", valid.replace('\n', "\r\n")),
          ("two separators", valid.replacen("0 0", "0  0", 1)),
          ("fractional coordinate", valid.replacen("0 0", "0.5 0", 1)),
          ("negative coordinate", valid.replacen("0 0", "-1 0", 1)),
          ("point outside the frame", valid.replacen("0 0", "512 0", 1)),
          ("empty file", String::new()),
      ];
      for (label, text) in cases {
          let dir = TempDir::new().unwrap();
          let path = write(&dir, "000000.lms", &text);
          let error = read_landmark_file(&path, FRAME_WIDTH, FRAME_HEIGHT)
              .expect_err(&format!("{label} must be refused"));
          assert!(
              matches!(error, PipelineError::InvalidLandmark { .. }),
              "{label}: {error:?}"
          );
      }
  }

  #[test]
  fn a_non_utf8_file_is_not_a_landmark_file() {
      let dir = TempDir::new().unwrap();
      let path = dir.path().join("000000.lms");
      fs::write(&path, [0xFF, 0xFE, 0x00, b'\n']).unwrap();
      let error = read_landmark_file(&path, FRAME_WIDTH, FRAME_HEIGHT).unwrap_err();
      let PipelineError::InvalidLandmark { message, .. } = error else {
          panic!("non-UTF-8 bytes must be a landmark problem: {error:?}");
      };
      assert!(message.contains("not UTF-8"), "{message}");
  }

  #[test]
  fn an_oversized_file_is_refused_with_its_limit() {
      let dir = TempDir::new().unwrap();
      let text = "0 0\n".repeat(4096);
      assert!(text.len() as u64 > MAX_LANDMARK_FILE_BYTES);
      let path = write(&dir, "000000.lms", &text);
      let error = read_landmark_file(&path, FRAME_WIDTH, FRAME_HEIGHT).unwrap_err();
      let PipelineError::InvalidLandmark { message, .. } = error else {
          panic!("an oversized file must be a landmark problem: {error:?}");
      };
      assert!(
          message.contains(&MAX_LANDMARK_FILE_BYTES.to_string()),
          "{message}"
      );
  }

  #[test]
  fn a_directory_is_not_a_regular_landmark_file() {
      let dir = TempDir::new().unwrap();
      let path = dir.path().join("000000.lms");
      fs::create_dir(&path).unwrap();
      let error = read_landmark_file(&path, FRAME_WIDTH, FRAME_HEIGHT).unwrap_err();
      assert!(
          matches!(error, PipelineError::LandmarkNotRegular { .. }),
          "{error:?}"
      );
  }

  #[test]
  fn a_symlink_is_not_a_regular_landmark_file() {
      let dir = TempDir::new().unwrap();
      let target = write(&dir, "target.lms", &valid_text());
      let link = dir.path().join("000000.lms");
      #[cfg(windows)]
      let result = std::os::windows::fs::symlink_file(&target, &link);
      #[cfg(unix)]
      let result = std::os::unix::fs::symlink(&target, &link);
      if let Err(error) = result {
          // 1314 is ERROR_PRIVILEGE_NOT_HELD: an unprivileged Windows account
          // cannot create symlinks, so the case is skipped rather than failed.
          if error.raw_os_error() == Some(1314) {
              eprintln!("skipping: this account may not create symlinks");
              return;
          }
          panic!("the symlink must be creatable: {error:?}");
      }
      let error = read_landmark_file(&link, FRAME_WIDTH, FRAME_HEIGHT).unwrap_err();
      assert!(
          matches!(error, PipelineError::LandmarkNotRegular { .. }),
          "{error:?}"
      );
  }

  #[test]
  fn a_missing_file_names_the_failed_operation() {
      let dir = TempDir::new().unwrap();
      let path = dir.path().join("000000.lms");
      let error = read_landmark_file(&path, FRAME_WIDTH, FRAME_HEIGHT).unwrap_err();
      let PipelineError::Io {
          operation,
          path: reported,
          ..
      } = error
      else {
          panic!("a missing file must be an IO failure: {error:?}");
      };
      assert_eq!(operation, "stat_landmarks");
      assert_eq!(reported, path);
  }
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `cargo test -p feathertalk-frame-pipeline --test landmarks`

  Expected: FAIL to compile with `error[E0432]: unresolved imports feathertalk_frame_pipeline::LANDMARK_POINTS, feathertalk_frame_pipeline::MAX_LANDMARK_FILE_BYTES, feathertalk_frame_pipeline::read_landmark_file`.

- [ ] **Step 3: Write minimal implementation**

  Create `rust/crates/feathertalk-frame-pipeline/src/landmark.rs`:

  ```rust
  //! Landmark files read back out of a finished asset package.

  use std::fs::{self, File};
  use std::io::Read;
  use std::path::Path;

  use crate::error::PipelineError;

  /// The number of landmark points PFLD produces for one frame.
  ///
  /// `serialize_landmarks` writes exactly this many lines and the reader below
  /// demands exactly this many, so the writer and the reader cannot drift.
  pub const LANDMARK_POINTS: usize = 110;

  /// The largest landmark file this reader will accept.
  ///
  /// The longest line the writer can emit is `"32767 32767\n"`, twelve bytes,
  /// so a complete file is at most 1 320 bytes. Eight KiB leaves six times that
  /// headroom while still refusing a file that has been replaced by something
  /// else entirely, before any of it is read into memory.
  pub const MAX_LANDMARK_FILE_BYTES: u64 = 8 * 1024;

  /// Read one landmark file and validate it against the frame it belongs to.
  ///
  /// Accepts only what `serialize_landmarks` writes: `LANDMARK_POINTS` lines of
  /// `"{x} {y}"`, each terminated by a single `\n`, every point inside the
  /// frame. The geometry is passed in because the file does not record it.
  pub fn read_landmark_file(
      path: &Path,
      frame_width: u32,
      frame_height: u32,
  ) -> Result<Vec<(i32, i32)>, PipelineError> {
      let metadata =
          fs::symlink_metadata(path).map_err(|source| io("stat_landmarks", path, source))?;
      if metadata.file_type().is_symlink() || !metadata.is_file() {
          return Err(PipelineError::LandmarkNotRegular {
              path: path.to_owned(),
          });
      }
      let size = metadata.len();
      if size > MAX_LANDMARK_FILE_BYTES {
          return Err(invalid_landmark(
              path,
              format!("file is {size} bytes, over the {MAX_LANDMARK_FILE_BYTES} byte limit"),
          ));
      }
      let mut file = File::open(path).map_err(|source| io("open_landmarks", path, source))?;
      let mut bytes = Vec::with_capacity(size as usize);
      file.read_to_end(&mut bytes)
          .map_err(|source| io("read_landmarks", path, source))?;
      parse_landmarks(path, &bytes, frame_width, frame_height)
  }

  /// Parse the file body, kept separate from the IO so the shape rules read in
  /// one place.
  fn parse_landmarks(
      path: &Path,
      bytes: &[u8],
      frame_width: u32,
      frame_height: u32,
  ) -> Result<Vec<(i32, i32)>, PipelineError> {
      let text = std::str::from_utf8(bytes)
          .map_err(|error| invalid_landmark(path, format!("file is not UTF-8: {error}")))?;
      let body = text
          .strip_suffix('\n')
          .ok_or_else(|| invalid_landmark(path, "file does not end with a newline".to_owned()))?;
      // `split` rather than `lines`: `lines` tolerates a missing final
      // terminator and silently strips a trailing `\r`, and both of those are
      // files this reader must refuse rather than quietly repair.
      let lines: Vec<&str> = body.split('\n').collect();
      if lines.len() != LANDMARK_POINTS {
          return Err(invalid_landmark(
              path,
              format!("expected {LANDMARK_POINTS} lines, found {}", lines.len()),
          ));
      }
      let mut points = Vec::with_capacity(LANDMARK_POINTS);
      for (index, line) in lines.iter().enumerate() {
          points.push(parse_point(path, index, line, frame_width, frame_height)?);
      }
      Ok(points)
  }

  fn parse_point(
      path: &Path,
      index: usize,
      line: &str,
      frame_width: u32,
      frame_height: u32,
  ) -> Result<(i32, i32), PipelineError> {
      let (x_text, y_text) = line.split_once(' ').ok_or_else(|| {
          invalid_landmark(
              path,
              format!("line {index} is not two integers separated by one space: {line:?}"),
          )
      })?;
      let x = parse_coordinate(path, index, "x", x_text)?;
      let y = parse_coordinate(path, index, "y", y_text)?;
      if x < 0 || y < 0 || x >= frame_width as i32 || y >= frame_height as i32 {
          return Err(invalid_landmark(
              path,
              format!("line {index} point ({x}, {y}) is outside {frame_width}x{frame_height}"),
          ));
      }
      Ok((x, y))
  }

  fn parse_coordinate(
      path: &Path,
      index: usize,
      axis: &'static str,
      text: &str,
  ) -> Result<i32, PipelineError> {
      text.parse::<i32>().map_err(|error| {
          invalid_landmark(
              path,
              format!("line {index} has a bad {axis} coordinate {text:?}: {error}"),
          )
      })
  }

  /// `publish.rs` keeps its own private copy of this helper rather than sharing
  /// one, so this module follows the same local pattern.
  fn io(operation: &'static str, path: &Path, source: std::io::Error) -> PipelineError {
      PipelineError::Io {
          operation,
          path: path.to_owned(),
          source,
      }
  }

  fn invalid_landmark(path: &Path, message: String) -> PipelineError {
      PipelineError::InvalidLandmark {
          path: path.to_owned(),
          message,
      }
  }
  ```

  In `rust/crates/feathertalk-frame-pipeline/src/lib.rs`, add `mod landmark;` to the module list between `extraction` and `model`, and add the export after the existing `pub use extraction::{...};` block:

  ```rust
  pub use landmark::{LANDMARK_POINTS, MAX_LANDMARK_FILE_BYTES, read_landmark_file};
  ```

  In `rust/crates/feathertalk-frame-pipeline/src/evaluate.rs`, make `serialize_landmarks` use the shared constant instead of its two literal `110`s. Add `LANDMARK_POINTS` to the existing `use crate::{...}` list between `FrameBatch` and `NoObserver`, then change:

  - `if landmarks.points().len() != 110` to `if landmarks.points().len() != LANDMARK_POINTS`
  - the message `"expected 110 points, got {}"` to `"expected {LANDMARK_POINTS} points, got {}"` (it renders identically, so no test changes)
  - `Vec::with_capacity(110 * 16)` to `Vec::with_capacity(LANDMARK_POINTS * 16)`

  Leave the rest of `serialize_landmarks` alone; the writer's bounds check and `format!("{} {}\n", point.x, point.y)` line are what this reader is written against.

- [ ] **Step 4: Run test to verify it passes**

  Run: `cargo test -p feathertalk-frame-pipeline --test landmarks`

  Expected: PASS — 7 tests, 0 failed (the symlink test may print `skipping` on an unprivileged Windows account, which still counts as passed). Then `cargo test -p feathertalk-frame-pipeline` to prove the `evaluate.rs` edit changed no existing expectation, `rustfmt --edition 2024 --check crates/feathertalk-frame-pipeline/src/landmark.rs crates/feathertalk-frame-pipeline/src/lib.rs crates/feathertalk-frame-pipeline/src/evaluate.rs crates/feathertalk-frame-pipeline/tests/landmarks.rs`, and `cargo clippy -p feathertalk-frame-pipeline --all-targets -- -D warnings`.

- [ ] **Step 5: Commit**

  ```powershell
  git add rust/crates/feathertalk-frame-pipeline/src/landmark.rs rust/crates/feathertalk-frame-pipeline/src/lib.rs rust/crates/feathertalk-frame-pipeline/src/evaluate.rs rust/crates/feathertalk-frame-pipeline/tests/landmarks.rs
  git commit -m "feat(frame-pipeline): read back the landmark files"
  ```

---
  git add rust/crates/feathertalk-frame-pipeline/src/landmark.rs rust/crates/feathertalk-frame-pipeline/src/lib.rs rust/crates/feathertalk-frame-pipeline/src/evaluate.rs rust/crates/feathertalk-frame-pipeline/tests/landmarks.rs
  git commit -m "feat(frame-pipeline): read back the landmark files"
  ```

---

### Task 5: Fit a Feature Matrix to a Token Count

**Files:**
- Modify: `rust/crates/feathertalk-audio/src/stitch.rs`
- Modify: `rust/crates/feathertalk-audio/src/lib.rs`
- Test: `rust/crates/feathertalk-audio/tests/stitching.rs`

**Interfaces:**
- Consumes: nothing.
- Produces: `pub fn fit_feature_tokens(matrix: FeatureMatrix, tokens: usize) -> Result<FeatureMatrix, AudioError>`, re-exported as `feathertalk_audio::fit_feature_tokens`. Task 8 calls it once, with `tokens = 2 * frame_count`.
- Also produces `pub(crate) fn FeatureMatrix::into_values(self) -> Vec<f32>`, used only by `fit_feature_tokens`.

**Why now:** `commit_feature_artifact` refuses any matrix whose token count is not exactly `2 * frame_count`, so the lock has to reconcile the feature file against the frame count before committing. The reconciliation rule already exists — the tail of `extract_long_audio` pads short output with zero vectors and truncates long output — but it is welded to the waveform path, where the token count is derived from the sample count. The lock derives it from the frame count instead. Extracting the rule into `fit_values` and exposing it as `fit_feature_tokens` means both paths share one definition of "fit", and the worker gets a total function instead of an ad-hoc `resize`/`truncate` pair of its own. This is the last leaf; Task 6 starts on the worker.

- [ ] **Step 1: Write the failing test**

  `rust/crates/feathertalk-audio/tests/stitching.rs` opens with exactly this import block:

  ```rust
  use std::sync::{Arc, Mutex};

  use feathertalk_audio::{
      AudioError, ChunkEncoder, DEFAULT_CHUNK_SAMPLES, FeatureMatrix, drop_odd_token,
      extract_long_audio,
  };
  ```

  Add `fit_feature_tokens` after `extract_long_audio`, then append three tests:

  ```rust
  #[test]
  fn fitting_pads_truncates_and_leaves_an_exact_matrix_alone() {
      let matrix = FeatureMatrix::new(2, 4, vec![1.0; 8]).unwrap();

      let padded = fit_feature_tokens(matrix.clone(), 3).unwrap();
      assert_eq!(padded.tokens(), 3);
      assert_eq!(padded.dims(), 4);
      assert_eq!(&padded.values()[..8], &[1.0; 8]);
      assert_eq!(&padded.values()[8..], &[0.0; 4]);

      let truncated = fit_feature_tokens(matrix.clone(), 1).unwrap();
      assert_eq!(truncated.tokens(), 1);
      assert_eq!(truncated.values(), &[1.0; 4]);

      let unchanged = fit_feature_tokens(matrix.clone(), 2).unwrap();
      assert_eq!(unchanged, matrix);
  }

  #[test]
  fn an_impossible_token_count_overflows_instead_of_allocating() {
      let matrix = FeatureMatrix::new(1, 1024, vec![0.5; 1024]).unwrap();
      let error = fit_feature_tokens(matrix, usize::MAX).unwrap_err();
      assert!(matches!(error, AudioError::FeatureSizeOverflow), "{error:?}");
  }

  #[test]
  fn fitting_to_zero_tokens_empties_the_matrix() {
      let matrix = FeatureMatrix::new(2, 4, vec![1.0; 8]).unwrap();
      let empty = fit_feature_tokens(matrix, 0).unwrap();
      assert_eq!(empty.tokens(), 0);
      assert_eq!(empty.dims(), 4);
      assert!(empty.values().is_empty());
  }
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `cargo test -p feathertalk-audio --test stitching`

  Expected: FAIL to compile with `error[E0432]: unresolved import feathertalk_audio::fit_feature_tokens`.

- [ ] **Step 3: Write minimal implementation**

  In `rust/crates/feathertalk-audio/src/stitch.rs`, add both functions. `fit_values` goes next to the other private helpers; `fit_feature_tokens` goes after `extract_long_audio`.

  ```rust
  /// Pad with zeros or truncate so that `values` holds exactly `target_values`.
  fn fit_values(values: &mut Vec<f32>, target_values: usize) {
      if values.len() < target_values {
          values.resize(target_values, 0.0);
      } else {
          values.truncate(target_values);
      }
  }

  /// Pad or truncate a feature matrix to exactly `tokens` tokens.
  ///
  /// Same rule as the tail of `extract_long_audio` — short output gains zero
  /// vectors, long output loses its tail — exposed for callers that learn the
  /// token count from somewhere other than the waveform. The asset lock learns
  /// it from the frame count.
  pub fn fit_feature_tokens(
      matrix: FeatureMatrix,
      tokens: usize,
  ) -> Result<FeatureMatrix, AudioError> {
      let dims = matrix.dims();
      let target_values = tokens
          .checked_mul(dims)
          .ok_or(AudioError::FeatureSizeOverflow)?;
      let mut values = matrix.into_values();
      fit_values(&mut values, target_values);
      FeatureMatrix::new(tokens, dims, values)
  }
  ```

  Then replace the padding/truncation tail of `extract_long_audio` with `fit_values(&mut values, target_values);`, keeping the `checked_mul` that computes `target_values` above it. The two paths must now share one body; if `extract_long_audio` still contains its own `resize`/`truncate`, the extraction is incomplete.

  In `rust/crates/feathertalk-audio/src/lib.rs`, add the consuming accessor to `impl FeatureMatrix`, next to `values()`:

  ```rust
      /// Take the backing storage. `pub(crate)` because it is a stepping stone
      /// for in-crate transforms, not part of the crate's public surface.
      pub(crate) fn into_values(self) -> Vec<f32> {
          self.values
      }
  ```

  and extend the stitch re-export to `pub use stitch::{ChunkEncoder, drop_odd_token, extract_long_audio, fit_feature_tokens};`.

- [ ] **Step 4: Run test to verify it passes**

  Run: `cargo test -p feathertalk-audio --test stitching`

  Expected: PASS, 0 failed. Then `cargo test -p feathertalk-audio` — the whole crate, because `extract_long_audio` was edited and its own fitting tests must still hold. Then `rustfmt --edition 2024 --check crates/feathertalk-audio/src/stitch.rs crates/feathertalk-audio/src/lib.rs crates/feathertalk-audio/tests/stitching.rs` and `cargo clippy -p feathertalk-audio --all-targets -- -D warnings`.

- [ ] **Step 5: Commit**

  ```powershell
  git add rust/crates/feathertalk-audio/src/stitch.rs rust/crates/feathertalk-audio/src/lib.rs rust/crates/feathertalk-audio/tests/stitching.rs
  git commit -m "feat(audio): fit a feature matrix to a token count"
  ```

---

### Task 6: Shape the Lock Result Payload

**Files:**
- Create: `rust/crates/feathertalk-worker/src/lock_result.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs`
- Test: `rust/crates/feathertalk-worker/tests/lock_result.rs`

**Interfaces:**
- Consumes: `feathertalk_audio::{FeatureArtifact, FeatureCommitSpec}` (both already public).
- Produces: `pub fn lock_to_json(project_dir: &Path, spec: &FeatureCommitSpec, artifact: &FeatureArtifact, token_adjustment: i64) -> Value`, re-exported as `feathertalk_worker::lock_to_json`. Task 8 calls it once, on success.

**Why now:** The payload is the command's contract with the desktop UI and the CLI's JSON mode, and it is the one piece of the command that can be pinned without any filesystem work. Writing it first means Task 8 has a finished return value to build toward instead of inventing keys while it also handles admission and commit. It mirrors `feature_result.rs`, which does the same job for `extract_features`, including the `path_text` helper — paths are rendered once, in one place, so the JSON never carries a `Path` debug form.

- [ ] **Step 1: Write the failing test**

  Create `rust/crates/feathertalk-worker/tests/lock_result.rs`, mirroring the existing `tests/feature_result.rs`. A 2x4 matrix produces a 76-byte file: the 44-byte header plus eight `f32`s.

  ```rust
  //! The shape of the asset lock's result payload.

  use feathertalk_audio::{FeatureCommitSpec, FeatureMatrix, write_feature_file_no_clobber};
  use feathertalk_worker::lock_to_json;
  use tempfile::TempDir;

  const LANDMARK_SHA256: &str =
      "e131dd764236fde54a27b2f7084906119f06c28b140bf127b459ec967e92915b";
  const FEATURE_SHA256: &str =
      "1111111111111111111111111111111111111111111111111111111111111111";

  #[test]
  fn the_payload_carries_every_field_the_desktop_needs() {
      let dir = TempDir::new().unwrap();
      let project_dir = dir.path().join("project");
      let feature_path = dir.path().join("feather_hubert.f32");
      let matrix = FeatureMatrix::new(2, 4, vec![0.5; 8]).unwrap();
      let artifact = write_feature_file_no_clobber(&feature_path, &matrix).unwrap();
      let spec = FeatureCommitSpec {
          project_root: project_dir.clone(),
          frame_count: 1,
          frame_width: 1280,
          frame_height: 720,
          landmark_model_sha256: LANDMARK_SHA256.to_owned(),
          feature_model_sha256: FEATURE_SHA256.to_owned(),
      };

      let value = lock_to_json(&project_dir, &spec, &artifact, -3);
      let object = value.as_object().expect("the payload must be an object");

      assert_eq!(object["project_dir"], project_dir.display().to_string());
      let manifest = project_dir.join("assets").join("assets.json");
      assert_eq!(object["manifest_file"], manifest.display().to_string());
      assert_eq!(object["frame_count"], 1);
      assert_eq!(object["frame_width"], 1280);
      assert_eq!(object["frame_height"], 720);
      assert_eq!(object["feature_file"], feature_path.display().to_string());
      assert_eq!(object["tokens"], 2);
      assert_eq!(object["dims"], 4);
      assert_eq!(object["bytes"], 76);
      assert_eq!(object["sha256"], artifact.sha256());
      assert_eq!(object["token_adjustment"], -3);
      assert_eq!(object["landmark_model_sha256"], LANDMARK_SHA256);
      assert_eq!(object["feature_model_sha256"], FEATURE_SHA256);
      // Every key is asserted above, so the count keeps a future field from
      // slipping into the protocol untested.
      assert_eq!(object.len(), 13);
  }
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `cargo test -p feathertalk-worker --test lock_result`

  Expected: FAIL to compile with `error[E0432]: unresolved import feathertalk_worker::lock_to_json`.

- [ ] **Step 3: Write minimal implementation**

  Create `rust/crates/feathertalk-worker/src/lock_result.rs`:

  ```rust
  //! The JSON payload the asset lock returns on success.

  use std::path::Path;

  use feathertalk_audio::{FeatureArtifact, FeatureCommitSpec};
  use serde_json::{Value, json};

  fn path_text(path: &Path) -> String {
      path.display().to_string()
  }

  /// Shape the result of a successful lock.
  ///
  /// Reports what the caller cannot see for itself: where the manifest landed,
  /// the geometry that was verified, the feature file that was committed, and
  /// how far the feature stream had to move to match the frame count.
  /// `token_adjustment` is signed on purpose — a negative value means tokens
  /// were dropped, which is the case an operator may want to look at.
  pub fn lock_to_json(
      project_dir: &Path,
      spec: &FeatureCommitSpec,
      artifact: &FeatureArtifact,
      token_adjustment: i64,
  ) -> Value {
      let manifest_file = project_dir.join("assets").join("assets.json");
      json!({
          "project_dir": path_text(project_dir),
          "manifest_file": path_text(&manifest_file),
          "frame_count": spec.frame_count,
          "frame_width": spec.frame_width,
          "frame_height": spec.frame_height,
          "feature_file": path_text(artifact.path()),
          "tokens": artifact.tokens(),
          "dims": artifact.dims(),
          "bytes": artifact.bytes(),
          "sha256": artifact.sha256(),
          "token_adjustment": token_adjustment,
          "landmark_model_sha256": spec.landmark_model_sha256.as_str(),
          "feature_model_sha256": spec.feature_model_sha256.as_str(),
      })
  }
  ```

  In `rust/crates/feathertalk-worker/src/lib.rs`, add `mod lock_result;` to the module list between `handshake` and `models`, and `pub use lock_result::lock_to_json;` to the `pub use` block in the same relative position.

- [ ] **Step 4: Run test to verify it passes**

  Run: `cargo test -p feathertalk-worker --test lock_result`

  Expected: PASS — 1 test, 0 failed. Then `rustfmt --edition 2024 --check crates/feathertalk-worker/src/lock_result.rs crates/feathertalk-worker/src/lib.rs crates/feathertalk-worker/tests/lock_result.rs` and `cargo clippy -p feathertalk-worker --all-targets -- -D warnings`.

- [ ] **Step 5: Commit**

  ```powershell
  git add rust/crates/feathertalk-worker/src/lock_result.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/tests/lock_result.rs
  git commit -m "feat(worker): shape the lock result payload"
  ```

---
  git add rust/crates/feathertalk-worker/src/lock_result.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/tests/lock_result.rs
  git commit -m "feat(worker): shape the lock result payload"
  ```

---

### Task 7: Verify an Asset Package

**Files:**
- Create: `rust/crates/feathertalk-worker/src/asset_scan.rs` (implementation and its inline `#[cfg(test)] mod tests`)
- Modify: `rust/crates/feathertalk-worker/src/lib.rs`

**Interfaces:**
- Consumes: `feathertalk_frame_adapters::probe_jpeg_geometry` (Task 3); `feathertalk_frame_pipeline::{MAX_FRAME_BYTES, PipelineError, QualityReport, read_landmark_file}` (Task 4); `crate::admission::invalid_request` and `crate::pipeline_task_error` (Task 2 extended the latter's mapping).
- Produces, both `pub(crate)`:
  - `pub(crate) fn verify_frames(assets: &Path, report: &QualityReport, token: &CancellationToken, reporter: &dyn TaskReporter) -> Result<(u32, u32), CommandOutcome>`
  - `pub(crate) fn count_asset_files(assets: &Path, frame_count: u64) -> Result<(), CommandOutcome>`

  Task 8 calls both, in that order, and uses the returned `(width, height)` to fill `FeatureCommitSpec`.

**Why now:** Every leaf the scan needs now exists, and the scan is the only part of the command with real algorithmic content: it decides what "the package is whole" means. Keeping it in its own module, returning `CommandOutcome` rather than a bespoke error type, lets Task 8 read as a straight sequence of `?`s. The tests are inline rather than in `tests/` because both functions are `pub(crate)` — the precedent in this workspace is `cli/src/run.rs`, which unit-tests its private helpers in a `#[cfg(test)] mod tests`. Exposing them publicly just to test them would widen the crate's surface for no caller's benefit.

  Two design points the tests pin down. First, the scan reads only a 64 KiB prefix of each frame; a JPEG whose SOF marker sits beyond that gets one full re-read, and the second attempt's error is the one reported, so the diagnostic always describes the complete file. Second, the directory file counts are checked separately from the report walk: the report is a list of what should be there, and a leftover frame from an earlier, longer extraction is invisible to a walk that only follows the list.

- [ ] **Step 1: Write the failing test**

  Create `rust/crates/feathertalk-worker/src/asset_scan.rs` containing only the module documentation and this test module, so the test names the functions before they exist. `tempfile` is already a dev dependency of `feathertalk-worker`. The two fixtures come from the sibling crate and were measured with `ffprobe`: `demo_frame_v1/frame.jpg` is 1280x720, `opencv_cpu_v1/frame.jpg` is 640x640.

  ```rust
  #[cfg(test)]
  mod tests {
      use std::path::PathBuf;
      use std::sync::Mutex;

      use feathertalk_domain::ErrorCode;
      use feathertalk_frame_pipeline::FrameQuality;
      use tempfile::TempDir;

      use super::*;
      use crate::NoReporter;

      const SHA256: &str = "1111111111111111111111111111111111111111111111111111111111111111";

      /// 1280x720.
      fn wide_frame() -> PathBuf {
          Path::new(env!("CARGO_MANIFEST_DIR"))
              .join("../feathertalk-frame-adapters/tests/fixtures/demo_frame_v1/frame.jpg")
      }

      /// 640x640, used to force a geometry disagreement.
      fn square_frame() -> PathBuf {
          Path::new(env!("CARGO_MANIFEST_DIR"))
              .join("../feathertalk-frame-adapters/tests/fixtures/opencv_cpu_v1/frame.jpg")
      }

      #[derive(Default)]
      struct Recorder {
          events: Mutex<Vec<(TaskStage, Option<Progress>)>>,
      }

      impl TaskReporter for Recorder {
          fn report(&self, stage: TaskStage, progress: Option<Progress>) {
              self.events
                  .lock()
                  .expect("the recorder must not be poisoned")
                  .push((stage, progress));
          }
      }

      /// 110 points, all inside every frame these tests use.
      fn landmark_text() -> String {
          let mut text = String::new();
          for index in 0..110 {
              text.push_str(&format!("{} {}\n", index, index * 2));
          }
          text
      }

      fn frame_quality(index: u64, frame_bytes: u64) -> FrameQuality {
          FrameQuality::new(
              index,
              format!("frames/{index:06}.jpg"),
              format!("landmarks/{index:06}.lms"),
              frame_bytes,
              SHA256,
              SHA256,
              0.9,
              [0.0, 0.0, 64.0, 64.0],
              12.5,
          )
          .expect("the frame quality fixture must be valid")
      }

      /// An asset directory of `count` copies of the wide fixture frame, with a
      /// matching quality report. The `TempDir` is returned so it outlives the
      /// test body.
      fn package(count: u64) -> (TempDir, PathBuf, QualityReport) {
          let dir = TempDir::new().expect("a temporary directory must be available");
          let assets = dir.path().join("assets");
          fs::create_dir_all(assets.join("frames")).unwrap();
          fs::create_dir_all(assets.join("landmarks")).unwrap();
          let source = fs::read(wide_frame()).expect("the wide fixture must be readable");
          let mut frames = Vec::new();
          for index in 0..count {
              fs::write(assets.join(format!("frames/{index:06}.jpg")), &source).unwrap();
              fs::write(assets.join(format!("landmarks/{index:06}.lms")), landmark_text()).unwrap();
              frames.push(frame_quality(index, source.len() as u64));
          }
          let report =
              QualityReport::new(count, frames, Vec::new()).expect("the report must be valid");
          (dir, assets, report)
      }

      #[test]
      fn a_consistent_package_reports_its_geometry_and_progress() {
          let (_dir, assets, report) = package(3);
          let reporter = Recorder::default();
          let geometry =
              verify_frames(&assets, &report, &CancellationToken::new(), &reporter).unwrap();
          assert_eq!(geometry, (1280, 720));
          count_asset_files(&assets, report.frame_count()).unwrap();

          let events = reporter.events.lock().unwrap().clone();
          assert_eq!(events.len(), 3);
          for (position, (stage, progress)) in events.iter().enumerate() {
              assert_eq!(*stage, TaskStage::Preparing);
              let completed = position as u64 + 1;
              assert_eq!(
                  *progress,
                  Some(Progress {
                      completed,
                      total: Some(3)
                  })
              );
          }
      }

      #[test]
      fn a_missing_frame_is_a_media_failure() {
          let (_dir, assets, report) = package(2);
          fs::remove_file(assets.join("frames/000001.jpg")).unwrap();
          let outcome = verify_frames(&assets, &report, &CancellationToken::new(), &NoReporter)
              .expect_err("a missing frame must fail");
          let CommandOutcome::Failed(error) = outcome else {
              panic!("a missing frame must be a failure");
          };
          assert_eq!(error.code, ErrorCode::MediaInvalid);
          assert_eq!(error.summary, "抽出的帧不可用");
          error.validate().unwrap();
      }

      #[test]
      fn a_frame_of_a_different_size_is_refused() {
          let (_dir, assets, report) = package(3);
          let square = fs::read(square_frame()).expect("the square fixture must be readable");
          fs::write(assets.join("frames/000001.jpg"), &square).unwrap();
          let outcome = verify_frames(&assets, &report, &CancellationToken::new(), &NoReporter)
              .expect_err("mismatched geometry must fail");
          let CommandOutcome::Failed(error) = outcome else {
              panic!("mismatched geometry must be a failure");
          };
          assert_eq!(error.summary, "素材帧尺寸不一致");
          assert!(error.detail.contains("640x640"), "{}", error.detail);
          assert!(error.detail.contains("1280x720"), "{}", error.detail);
      }

      #[test]
      fn a_malformed_landmark_file_is_refused() {
          let (_dir, assets, report) = package(1);
          fs::write(assets.join("landmarks/000000.lms"), "0 0\n").unwrap();
          let outcome = verify_frames(&assets, &report, &CancellationToken::new(), &NoReporter)
              .expect_err("a malformed landmark file must fail");
          let CommandOutcome::Failed(error) = outcome else {
              panic!("a malformed landmark file must be a failure");
          };
          assert_eq!(error.summary, "关键点文件不可用");
          assert_eq!(error.code, ErrorCode::MediaInvalid);
      }

      #[test]
      fn a_report_with_no_frames_has_no_geometry() {
          let dir = TempDir::new().unwrap();
          let assets = dir.path().join("assets");
          fs::create_dir_all(assets.join("frames")).unwrap();
          let report = QualityReport::new(1, Vec::new(), Vec::new()).unwrap();
          let outcome = verify_frames(&assets, &report, &CancellationToken::new(), &NoReporter)
              .expect_err("an empty report must fail");
          let CommandOutcome::Failed(error) = outcome else {
              panic!("an empty report must be a failure");
          };
          assert_eq!(error.summary, "素材包没有可用的帧");
      }

      #[test]
      fn a_leftover_file_breaks_the_count() {
          let (_dir, assets, report) = package(3);
          fs::copy(
              assets.join("frames/000000.jpg"),
              assets.join("frames/000009.jpg"),
          )
          .unwrap();
          let outcome = count_asset_files(&assets, report.frame_count())
              .expect_err("a leftover frame must fail");
          let CommandOutcome::Failed(error) = outcome else {
              panic!("a leftover frame must be a failure");
          };
          assert_eq!(error.summary, "素材目录的文件数与质检报告不一致");
          assert!(error.detail.contains("4 .jpg files"), "{}", error.detail);
      }

      #[test]
      fn files_that_are_not_indexed_assets_do_not_count() {
          let (_dir, assets, report) = package(2);
          fs::write(assets.join("frames/notes.txt"), "scratch").unwrap();
          fs::write(assets.join("frames/00001.jpg"), "a five digit stem").unwrap();
          fs::create_dir(assets.join("frames/nested")).unwrap();
          count_asset_files(&assets, report.frame_count()).unwrap();
      }

      #[test]
      fn a_cancelled_token_stops_before_the_first_frame() {
          let (_dir, assets, report) = package(2);
          let token = CancellationToken::new();
          token.cancel();
          let reporter = Recorder::default();
          let outcome = verify_frames(&assets, &report, &token, &reporter)
              .expect_err("a cancelled token must stop the scan");
          assert!(matches!(outcome, CommandOutcome::Cancelled));
          assert!(reporter.events.lock().unwrap().is_empty());
      }
  }
  ```

  Add `mod asset_scan;` to `rust/crates/feathertalk-worker/src/lib.rs` between `admission` and `commands`. No `pub use` line: nothing here is public.

- [ ] **Step 2: Run test to verify it fails**

  Run: `cargo test -p feathertalk-worker --lib`

  Expected: FAIL to compile with `error[E0425]: cannot find function verify_frames in this scope` and the same for `count_asset_files`.

- [ ] **Step 3: Write minimal implementation**

  Prepend the implementation to `rust/crates/feathertalk-worker/src/asset_scan.rs`, above the test module:

  ```rust
  //! Verification of a finished asset package: frames, landmarks, and counts.

  use std::fs::{self, File};
  use std::io::Read;
  use std::path::Path;

  use feathertalk_domain::{Progress, TaskStage};
  use feathertalk_frame_adapters::probe_jpeg_geometry;
  use feathertalk_frame_pipeline::{
      MAX_FRAME_BYTES, PipelineError, QualityReport, read_landmark_file,
  };
  use feathertalk_media::CancellationToken;

  use crate::{CommandOutcome, TaskReporter, admission::invalid_request, pipeline_task_error};

  /// How much of a frame is read before its header is parsed.
  ///
  /// A baseline JPEG's SOF marker sits within the first few hundred bytes and a
  /// progressive one is not much further in, so 64 KiB is generous. Reading a
  /// prefix rather than whole files keeps a 49-frame verification from moving
  /// tens of megabytes for two numbers per frame.
  const JPEG_HEADER_PROBE_BYTES: u64 = 64 * 1024;

  /// Verify every frame the quality report lists and return the shared geometry.
  ///
  /// Structural verification only: nothing is re-hashed and `frame_bytes` is not
  /// compared, because operators are allowed to hand-edit frames before locking.
  /// What must hold is that every frame exists, is a regular file within the
  /// size limit, decodes, has the same dimensions as its siblings, and has a
  /// landmark file whose points fall inside it.
  pub(crate) fn verify_frames(
      assets: &Path,
      report: &QualityReport,
      token: &CancellationToken,
      reporter: &dyn TaskReporter,
  ) -> Result<(u32, u32), CommandOutcome> {
      let total = Some(report.frame_count());
      let mut geometry: Option<(u64, u32, u32)> = None;
      let mut completed = 0u64;
      for frame in report.frames() {
          if token.is_cancelled() {
              return Err(CommandOutcome::Cancelled);
          }
          let frame_path = assets.join(frame.frame_file());
          let (width, height) = verify_frame(&frame_path).map_err(failed)?;
          match geometry {
              None => geometry = Some((frame.index(), width, height)),
              Some((reference_index, reference_width, reference_height)) => {
                  if width != reference_width || height != reference_height {
                      return Err(CommandOutcome::Failed(invalid_request(
                          "素材帧尺寸不一致",
                          format!(
                              "frame {} is {width}x{height} but frame {reference_index} is \
                               {reference_width}x{reference_height}",
                              frame.index()
                          ),
                      )));
                  }
              }
          }
          let landmark_path = assets.join(frame.landmark_file());
          read_landmark_file(&landmark_path, width, height).map_err(failed)?;
          completed += 1;
          reporter.report(TaskStage::Preparing, Some(Progress { completed, total }));
      }
      match geometry {
          Some((_, width, height)) => Ok((width, height)),
          None => Err(CommandOutcome::Failed(invalid_request(
              "素材包没有可用的帧",
              "the quality report lists no frames".to_owned(),
          ))),
      }
  }

  /// Prove the asset directories hold exactly the files the report declares.
  ///
  /// The report is a list of what should be there; the directories are what is
  /// there. A leftover frame from an earlier, longer extraction is invisible to
  /// a walk that follows the list, and would desynchronise every consumer that
  /// reads the directory instead.
  pub(crate) fn count_asset_files(assets: &Path, frame_count: u64) -> Result<(), CommandOutcome> {
      for (directory, extension) in [("frames", "jpg"), ("landmarks", "lms")] {
          let path = assets.join(directory);
          let counted = count_matching(&path, extension).map_err(failed)?;
          if counted != frame_count {
              return Err(CommandOutcome::Failed(invalid_request(
                  "素材目录的文件数与质检报告不一致",
                  format!(
                      "{} holds {counted} .{extension} files, the quality report declares \
                       {frame_count}",
                      path.display()
                  ),
              )));
          }
      }
      Ok(())
  }

  /// Check one frame file and read its geometry.
  fn verify_frame(path: &Path) -> Result<(u32, u32), PipelineError> {
      let metadata = match fs::symlink_metadata(path) {
          Ok(metadata) => metadata,
          Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
              return Err(PipelineError::FrameMissing {
                  path: path.to_owned(),
              });
          }
          Err(source) => return Err(io("stat_frame", path, source)),
      };
      if metadata.file_type().is_symlink() || !metadata.is_file() {
          return Err(PipelineError::FrameNotRegular {
              path: path.to_owned(),
          });
      }
      let size = metadata.len();
      if size == 0 {
          return Err(PipelineError::FrameEmpty {
              path: path.to_owned(),
          });
      }
      if size > MAX_FRAME_BYTES {
          return Err(PipelineError::FrameTooLarge {
              path: path.to_owned(),
              limit: MAX_FRAME_BYTES,
              actual: size,
          });
      }
      let prefix = read_prefix(path, JPEG_HEADER_PROBE_BYTES)?;
      match probe_jpeg_geometry(path, &prefix) {
          Ok(geometry) => Ok(geometry),
          // A progressive or heavily commented JPEG can push its SOF marker
          // past the prefix. Re-read the whole file once and let the second
          // attempt's diagnostic stand, so the error describes the whole file.
          Err(_) if size > prefix.len() as u64 => {
              let whole = read_prefix(path, size)?;
              probe_jpeg_geometry(path, &whole)
          }
          Err(error) => Err(error),
      }
  }

  /// Read at most `limit` bytes from `path`.
  fn read_prefix(path: &Path, limit: u64) -> Result<Vec<u8>, PipelineError> {
      let file = File::open(path).map_err(|source| io("open_frame", path, source))?;
      let mut bytes = Vec::new();
      file.take(limit)
          .read_to_end(&mut bytes)
          .map_err(|source| io("read_frame", path, source))?;
      Ok(bytes)
  }

  fn count_matching(path: &Path, extension: &str) -> Result<u64, PipelineError> {
      let entries = fs::read_dir(path).map_err(|source| io("read_assets_dir", path, source))?;
      let mut counted = 0u64;
      for entry in entries {
          let entry = entry.map_err(|source| io("read_assets_dir", path, source))?;
          let name = entry.file_name();
          let Some(name) = name.to_str() else {
              continue;
          };
          if is_indexed_name(name, extension) {
              counted += 1;
          }
      }
      Ok(counted)
  }

  /// `000123.jpg` and nothing else. Hand-written because the workspace carries
  /// no regex dependency and does not need one for six digits and a suffix.
  fn is_indexed_name(name: &str, extension: &str) -> bool {
      let Some((stem, suffix)) = name.rsplit_once('.') else {
          return false;
      };
      suffix == extension && stem.len() == 6 && stem.bytes().all(|byte| byte.is_ascii_digit())
  }

  /// `publish.rs` and `landmark.rs` keep their own copies of this helper, so
  /// this module follows the same local pattern.
  fn io(operation: &'static str, path: &Path, source: std::io::Error) -> PipelineError {
      PipelineError::Io {
          operation,
          path: path.to_owned(),
          source,
      }
  }

  /// Nothing in this module can produce `PipelineError::Cancelled`, so there is
  /// no cancellation bridge here; cancellation is the caller's own outcome.
  fn failed(error: PipelineError) -> CommandOutcome {
      CommandOutcome::Failed(pipeline_task_error(&error))
  }
  ```

  `feathertalk-frame-adapters` is already a dependency of `feathertalk-worker` (the frame extraction command uses it), so no `Cargo.toml` change is needed here.

- [ ] **Step 4: Run test to verify it passes**

  Run: `cargo test -p feathertalk-worker --lib`

  Expected: PASS — 8 tests in `asset_scan::tests`, 0 failed. Then `rustfmt --edition 2024 --check crates/feathertalk-worker/src/asset_scan.rs crates/feathertalk-worker/src/lib.rs` and `cargo clippy -p feathertalk-worker --all-targets -- -D warnings`.

- [ ] **Step 5: Commit**

  ```powershell
  git add rust/crates/feathertalk-worker/src/asset_scan.rs rust/crates/feathertalk-worker/src/lib.rs
  git commit -m "feat(worker): verify an asset package before locking it"
  ```

---

### Task 8: Lock the Asset Package

**Files:**
- Create: `rust/crates/feathertalk-worker/src/lock_asset_package.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs`
- Modify: `rust/crates/feathertalk-worker/Cargo.toml`, `rust/Cargo.lock`
- Test: `rust/crates/feathertalk-worker/tests/lock_asset_package.rs`

**Interfaces:**
- Consumes: `crate::asset_scan::{count_asset_files, verify_frames}` (Task 7); `crate::lock_to_json` (Task 6); `feathertalk_audio::fit_feature_tokens` (Task 5); and, all pre-existing, `crate::admission::{check_project_dir, invalid_request}`, `crate::{audio_task_error, pipeline_task_error, project_task_error}`, `feathertalk_audio::{FeatureCommitSpec, FeatureMatrix, commit_feature_artifact, read_feature_file}`, `feathertalk_frame_pipeline::{QualityReport, read_quality_report}`, `feathertalk_pfld::PFLD_MODEL_SHA256`, `feathertalk_project::{AssetPackageState, read_asset_manifest}`.
- Produces:

  ```rust
  pub fn execute_lock_asset_package(
      params: &ProjectDirParams,
      token: &CancellationToken,
      reporter: &dyn TaskReporter,
      feature_model_sha256: &str,
  ) -> CommandOutcome
  ```

  Re-exported as `feathertalk_worker::execute_lock_asset_package`. Task 9 dispatches to it, Task 10 advertises it, Task 11 reaches it from the CLI.

**Why now:** Every leaf the command needs is in place, so what is left is the command itself: admit, scan, fit, commit. Admission lives in a private `admit` that returns `Result<Admitted, CommandOutcome>`, the same shape `extract_features.rs` uses, so the body reads as a straight sequence and every refusal has exactly one home. The order inside `admit` is the design: the already-locked check comes first, so re-running the command on a finished package is one `stat` and a manifest read; the cheap `stat`s for the required files come before the feature file is read; the feature file is read before anything walks the frames. The encoder digest arrives as a `&str` because this command runs no inference — Task 9 reads it out of the package manifest and passes it down.

The one number that needs justifying is `MAX_TOKEN_FIT_DELTA`. Fitting exists to absorb the rounding between a waveform and a frame stream, which is under four tokens for a clip cut from one source, not to reconcile a feature file belonging to a different take. 50 tokens is one second of audio at 25 fps: generous enough that no honest package is refused, tight enough that a mismatched pair is caught before it becomes immutable.

- [ ] **Step 1: Write the failing test**

  Create `rust/crates/feathertalk-worker/tests/lock_asset_package.rs`. The fixture frame is the same 1280x720 JPEG Task 7 uses, reached through `CARGO_MANIFEST_DIR` because the sibling crate's fixtures are not copied anywhere. `serde_json` is a regular dependency of this crate, so the test can read the manifest back without a new dev dependency; `feathertalk-pfld` becomes one in Step 3.

  ```rust
  //! The asset lock, driven through the worker's public entry point.

  use std::{
      fs,
      path::{Path, PathBuf},
      sync::Mutex,
  };

  use feathertalk_audio::{FeatureMatrix, read_feature_file, write_feature_file_no_clobber};
  use feathertalk_domain::{ErrorCode, Progress, ProjectDirParams, TaskError, TaskStage};
  use feathertalk_frame_pipeline::{
      AnomalyCode, FrameAnomaly, FrameQuality, QualityReport, RecoveryAction,
  };
  use feathertalk_media::CancellationToken;
  use feathertalk_pfld::PFLD_MODEL_SHA256;
  use feathertalk_worker::{CommandOutcome, NoReporter, TaskReporter, execute_lock_asset_package};
  use serde_json::{Value, json};
  use tempfile::TempDir;

  /// Stands in for the digest Task 9 reads out of the FeatherHuBERT package
  /// manifest.
  const MODEL_SHA256: &str = "a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4";

  /// The per-frame digests the quality report carries. The lock verifies
  /// structure and never re-hashes a frame, so any 64-hex string does.
  const SHA256: &str = "1111111111111111111111111111111111111111111111111111111111111111";

  /// Three frames is the smallest fixture that still proves the walk iterates.
  const FRAME_COUNT: u64 = 3;

  /// FeatherHuBERT's output width; `commit_feature_artifact` accepts no other.
  const DIMS: usize = 1_024;

  #[derive(Default)]
  struct Recorder {
      events: Mutex<Vec<(TaskStage, Option<Progress>)>>,
  }

  impl Recorder {
      fn events(&self) -> Vec<(TaskStage, Option<Progress>)> {
          self.events
              .lock()
              .expect("the recorder must not be poisoned")
              .clone()
      }
  }

  impl TaskReporter for Recorder {
      fn report(&self, stage: TaskStage, progress: Option<Progress>) {
          self.events
              .lock()
              .expect("the recorder must not be poisoned")
              .push((stage, progress));
      }
  }

  /// The checked-in 1280x720 frame, borrowed from the adapters crate.
  fn fixture() -> PathBuf {
      Path::new(env!("CARGO_MANIFEST_DIR"))
          .join("../feathertalk-frame-adapters/tests/fixtures/demo_frame_v1/frame.jpg")
  }

  /// 110 points, all inside the fixture frame.
  fn landmark_text() -> String {
      let mut text = String::new();
      for index in 0..110 {
          text.push_str(&format!("{} {}\n", index, index * 2));
      }
      text
  }

  /// A complete, unlocked package: a project manifest, the two normalised media
  /// files, `FRAME_COUNT` frames with landmarks, a feature file of exactly
  /// `2 * FRAME_COUNT` tokens, and a clean quality report. The `TempDir` is
  /// returned so it outlives the test body.
  fn project() -> (TempDir, ProjectDirParams) {
      let root = TempDir::new().expect("a temporary directory must be available");
      let project_dir = root.path().join("project");
      let assets = project_dir.join("assets");
      fs::create_dir_all(assets.join("frames")).unwrap();
      fs::create_dir_all(assets.join("landmarks")).unwrap();
      fs::create_dir_all(assets.join("features")).unwrap();
      // Only its presence is checked; nothing here parses the project manifest.
      fs::write(project_dir.join("project.json"), b"{}").unwrap();
      fs::write(assets.join("video_25fps.mp4"), b"video").unwrap();
      fs::write(assets.join("audio_16k_mono.wav"), b"audio").unwrap();
      write_features(&assets, 2 * FRAME_COUNT as usize);
      let source = fs::read(fixture()).expect("the fixture frame must be readable");
      for index in 0..FRAME_COUNT {
          fs::write(assets.join(format!("frames/{index:06}.jpg")), &source).unwrap();
          fs::write(
              assets.join(format!("landmarks/{index:06}.lms")),
              landmark_text(),
          )
          .unwrap();
      }
      let frames = frame_qualities(&assets);
      let report = QualityReport::new(FRAME_COUNT, frames, Vec::new()).unwrap();
      write_report(&assets, &report);
      (root, ProjectDirParams { project_dir })
  }

  /// The report entries for the frames `project` wrote.
  fn frame_qualities(assets: &Path) -> Vec<FrameQuality> {
      (0..FRAME_COUNT)
          .map(|index| {
              let path = assets.join(format!("frames/{index:06}.jpg"));
              let frame_bytes = fs::metadata(&path).unwrap().len();
              FrameQuality::new(
                  index,
                  format!("frames/{index:06}.jpg"),
                  format!("landmarks/{index:06}.lms"),
                  frame_bytes,
                  SHA256,
                  SHA256,
                  0.9,
                  [0.0, 0.0, 64.0, 64.0],
                  12.5,
              )
              .expect("the frame quality fixture must be valid")
          })
          .collect()
  }

  fn write_report(assets: &Path, report: &QualityReport) {
      let bytes = serde_json::to_vec_pretty(report).expect("the report must serialise");
      fs::write(assets.join("quality.json"), bytes).unwrap();
  }

  /// Replace the feature file with one of `tokens` tokens.
  /// `write_feature_file_no_clobber` refuses to overwrite, so the old file goes
  /// first.
  fn write_features(assets: &Path, tokens: usize) {
      let path = assets.join("features").join("feather_hubert.f32");
      if path.exists() {
          fs::remove_file(&path).unwrap();
      }
      let matrix = FeatureMatrix::new(tokens, DIMS, vec![0.25; tokens * DIMS]).unwrap();
      write_feature_file_no_clobber(&path, &matrix).expect("the feature fixture must be writable");
  }

  fn run(
      params: &ProjectDirParams,
      token: &CancellationToken,
      reporter: &dyn TaskReporter,
  ) -> CommandOutcome {
      execute_lock_asset_package(params, token, reporter, MODEL_SHA256)
  }

  fn progress(completed: u64) -> Option<Progress> {
      Some(Progress {
          completed,
          total: Some(FRAME_COUNT),
      })
  }

  fn expect_completed(outcome: CommandOutcome) -> Value {
      match outcome {
          CommandOutcome::Completed(Some(result)) => result,
          other => panic!("expected a completed command, got {other:?}"),
      }
  }

  fn expect_failure(outcome: CommandOutcome) -> TaskError {
      match outcome {
          CommandOutcome::Failed(error) => error,
          other => panic!("expected a failure, got {other:?}"),
      }
  }

  fn manifest_path(params: &ProjectDirParams) -> PathBuf {
      params.project_dir.join("assets").join("assets.json")
  }
  ```

  Then the twelve tests, in the same file:

  ```rust
  #[test]
  fn a_complete_package_is_locked_and_reports_its_manifest() {
      let (_root, params) = project();

      let result = expect_completed(run(&params, &CancellationToken::new(), &NoReporter));

      let manifest = manifest_path(&params);
      let feature_file = params
          .project_dir
          .join("assets")
          .join("features")
          .join("feather_hubert.f32");
      assert_eq!(result["project_dir"], params.project_dir.display().to_string());
      assert_eq!(result["manifest_file"], manifest.display().to_string());
      assert_eq!(result["feature_file"], feature_file.display().to_string());
      assert_eq!(result["frame_count"], 3);
      assert_eq!(result["frame_width"], 1_280);
      assert_eq!(result["frame_height"], 720);
      assert_eq!(result["tokens"], 6);
      assert_eq!(result["dims"], 1_024);
      // The 44-byte header plus 6 * 1024 f32 values.
      assert_eq!(result["bytes"], 24_620);
      assert_eq!(result["token_adjustment"], 0);
      assert_eq!(result["landmark_model_sha256"], PFLD_MODEL_SHA256);
      assert_eq!(result["feature_model_sha256"], MODEL_SHA256);
      assert_eq!(result["sha256"].as_str().unwrap().len(), 64);

      let written: Value = serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
      assert_eq!(written["schema_version"], 1);
      assert_eq!(written["state"], "locked");
      assert_eq!(written["video_fps"], 25);
      assert_eq!(written["audio_sample_rate"], 16_000);
      assert_eq!(written["audio_channels"], 1);
      assert_eq!(written["frame_count"], 3);
      assert_eq!(written["frame_width"], 1_280);
      assert_eq!(written["frame_height"], 720);
      assert_eq!(written["feature_type"], "feather_hubert");
      assert_eq!(written["feature_shape"], json!([3, 2, 1_024]));
      assert_eq!(written["landmark_model_sha256"], PFLD_MODEL_SHA256);
      assert_eq!(written["feature_model_sha256"], MODEL_SHA256);
  }

  #[test]
  fn every_frame_reports_progress_under_one_stage() {
      let (_root, params) = project();
      let recorder = Recorder::default();

      expect_completed(run(&params, &CancellationToken::new(), &recorder));

      // One `Preparing` with no progress while admission reads the report and
      // the feature file, then one per frame. Nothing else: the commit is a
      // rename, not a stage.
      assert_eq!(
          recorder.events(),
          vec![
              (TaskStage::Preparing, None),
              (TaskStage::Preparing, progress(1)),
              (TaskStage::Preparing, progress(2)),
              (TaskStage::Preparing, progress(3)),
          ]
      );
  }

  #[test]
  fn a_relative_project_dir_is_rejected() {
      let relative = ProjectDirParams {
          project_dir: PathBuf::from("project"),
      };

      let error = expect_failure(run(&relative, &CancellationToken::new(), &NoReporter));

      assert_eq!(error.code, ErrorCode::MediaInvalid);
      assert_eq!(error.stage, TaskStage::Preparing);
      assert_eq!(error.summary, "工程目录必须是绝对路径");
  }

  #[test]
  fn an_already_locked_package_is_refused() {
      let (_root, params) = project();
      expect_completed(run(&params, &CancellationToken::new(), &NoReporter));

      let error = expect_failure(run(&params, &CancellationToken::new(), &NoReporter));

      assert_eq!(error.code, ErrorCode::MediaInvalid);
      assert_eq!(error.stage, TaskStage::Preparing);
      assert_eq!(error.summary, "素材包已加锁");
      assert!(error.detail.contains("assets.json"), "{}", error.detail);
      error.validate().unwrap();
  }

  #[test]
  fn a_corrupt_asset_manifest_is_not_a_crash() {
      let (_root, params) = project();
      fs::write(manifest_path(&params), b"{ not json").unwrap();

      let error = expect_failure(run(&params, &CancellationToken::new(), &NoReporter));

      // `feathertalk-project` owns the wording for a broken manifest. What this
      // pins is that a hand-edited file is a task failure, never a crash.
      assert_ne!(error.code, ErrorCode::WorkerCrashed);
      assert_eq!(error.stage, TaskStage::Preparing);
      error.validate().unwrap();
  }

  #[test]
  fn a_report_with_anomalies_is_refused() {
      let (_root, params) = project();
      let assets = params.project_dir.join("assets");
      let mut frames = frame_qualities(&assets);
      // An anomaly and an accepted frame cannot share an index, so the excluded
      // frame leaves the accepted list.
      let excluded = frames.pop().unwrap();
      let anomaly = FrameAnomaly::new(
          excluded.index(),
          AnomalyCode::BlurredFrame,
          "画面模糊",
          "blur variance 3.1 is below the threshold",
          RecoveryAction::ExcludeFrame,
      )
      .unwrap();
      let report = QualityReport::new(FRAME_COUNT, frames, vec![anomaly]).unwrap();
      write_report(&assets, &report);

      let error = expect_failure(run(&params, &CancellationToken::new(), &NoReporter));

      assert_eq!(error.code, ErrorCode::MediaInvalid);
      assert_eq!(error.summary, "素材包仍有异常帧");
      assert!(!manifest_path(&params).exists());
  }

  #[test]
  fn a_report_that_did_not_accept_every_frame_is_refused() {
      let (_root, params) = project();
      let assets = params.project_dir.join("assets");
      let mut frames = frame_qualities(&assets);
      frames.pop();
      // `QualityReport::new` derives `accepted_count` from the entries it is
      // given, so two entries against a frame count of three is exactly the
      // "a frame was dropped and never recovered" state.
      let report = QualityReport::new(FRAME_COUNT, frames, Vec::new()).unwrap();
      write_report(&assets, &report);

      let error = expect_failure(run(&params, &CancellationToken::new(), &NoReporter));

      assert_eq!(error.code, ErrorCode::MediaInvalid);
      assert_eq!(error.summary, "仍有帧未被接受");
      assert!(error.detail.contains("2 of 3"), "{}", error.detail);
  }

  #[test]
  fn a_missing_media_file_is_refused_by_name() {
      let (_root, params) = project();
      let audio = params.project_dir.join("assets").join("audio_16k_mono.wav");
      fs::remove_file(&audio).unwrap();

      let error = expect_failure(run(&params, &CancellationToken::new(), &NoReporter));

      assert_eq!(error.code, ErrorCode::MediaInvalid);
      assert_eq!(error.summary, "素材包缺少必需文件");
      assert!(
          error.detail.contains("audio_16k_mono.wav"),
          "{}",
          error.detail
      );
  }

  #[test]
  fn a_feature_file_from_another_take_is_refused() {
      let (_root, params) = project();
      // Six tokens are wanted; 57 is 51 away, one token past the fitting limit.
      write_features(&params.project_dir.join("assets"), 57);

      let error = expect_failure(run(&params, &CancellationToken::new(), &NoReporter));

      assert_eq!(error.code, ErrorCode::MediaInvalid);
      assert_eq!(error.summary, "特征令牌数与帧数不匹配");
      assert!(error.detail.contains("57 tokens"), "{}", error.detail);
      assert!(error.detail.contains("need 6"), "{}", error.detail);
  }

  #[test]
  fn a_missing_frame_reaches_the_scan() {
      let (_root, params) = project();
      let frame = params.project_dir.join("assets").join("frames/000001.jpg");
      fs::remove_file(&frame).unwrap();

      let error = expect_failure(run(&params, &CancellationToken::new(), &NoReporter));

      // Task 7 owns this wording; seeing it here proves the command runs the
      // scan rather than trusting the report.
      assert_eq!(error.summary, "抽出的帧不可用");
      assert!(!manifest_path(&params).exists());
  }

  #[test]
  fn a_cancelled_token_writes_no_manifest() {
      let (_root, params) = project();
      let token = CancellationToken::new();
      token.cancel();

      let outcome = run(&params, &token, &NoReporter);

      assert!(
          matches!(outcome, CommandOutcome::Cancelled),
          "expected a cancelled run, got {outcome:?}"
      );
      assert!(!manifest_path(&params).exists());
  }

  #[test]
  fn a_feature_file_two_tokens_short_is_padded_before_the_commit() {
      let (_root, params) = project();
      let assets = params.project_dir.join("assets");
      write_features(&assets, 4);

      let result = expect_completed(run(&params, &CancellationToken::new(), &NoReporter));

      assert_eq!(result["tokens"], 6);
      assert_eq!(result["token_adjustment"], 2);
      let path = assets.join("features").join("feather_hubert.f32");
      let matrix = read_feature_file(&path).unwrap();
      assert_eq!(matrix.tokens(), 6);
      assert_eq!(matrix.dims(), DIMS);
      // Padding is zero vectors, so the tail is distinguishable from data.
      assert!(matrix.values()[4 * DIMS..].iter().all(|value| *value == 0.0));
  }
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `cargo test -p feathertalk-worker --test lock_asset_package`

  Expected: FAIL to compile with `error[E0432]: unresolved import feathertalk_worker::execute_lock_asset_package` and `error[E0433]: failed to resolve: use of unresolved module or unlinked crate feathertalk_pfld`.

- [ ] **Step 3: Write minimal implementation**

  Create `rust/crates/feathertalk-worker/src/lock_asset_package.rs`:

  ```rust
  //! Verify a prepared asset package, then make it immutable.

  use std::path::PathBuf;

  use feathertalk_audio::{
      FeatureCommitSpec, FeatureMatrix, commit_feature_artifact, fit_feature_tokens,
      read_feature_file,
  };
  use feathertalk_domain::{ProjectDirParams, TaskStage};
  use feathertalk_frame_pipeline::{QualityReport, read_quality_report};
  use feathertalk_media::CancellationToken;
  use feathertalk_pfld::PFLD_MODEL_SHA256;
  use feathertalk_project::{AssetPackageState, read_asset_manifest};

  use crate::{
      CommandOutcome, TaskReporter,
      admission::{check_project_dir, invalid_request},
      asset_scan::{count_asset_files, verify_frames},
      audio_task_error, lock_to_json, pipeline_task_error, project_task_error,
  };

  /// The asset directory the earlier commands write into.
  const ASSETS_DIR: &str = "assets";

  /// The package manifest. `feathertalk-project` keeps its own private copy of
  /// this name (`src/package.rs:56`), so the literal is duplicated the way
  /// `admission.rs` duplicates `project.json`.
  const MANIFEST_FILE: &str = "assets.json";

  /// The report `extract_frames` publishes, relative to the asset directory.
  const QUALITY_FILE: &str = "quality.json";

  /// The feature file `extract_features` publishes, relative to the project
  /// root.
  const FEATURE_FILE: &str = "assets/features/feather_hubert.f32";

  /// What a lockable package must already contain, relative to the project root.
  /// Frames and landmarks are absent on purpose: they are checked against the
  /// quality report, one by one, rather than by name.
  const REQUIRED_FILES: [&str; 3] = [
      "assets/video_25fps.mp4",
      "assets/audio_16k_mono.wav",
      FEATURE_FILE,
  ];

  /// How far the feature stream may sit from the frame count and still be
  /// fitted.
  ///
  /// 50 tokens is one second of audio at 25 fps. The real drift between a wav
  /// and the frames cut from the same clip is under four tokens; anything past a
  /// second means the two inputs do not belong together, and the lock says so
  /// instead of quietly padding a mismatch into an immutable package.
  const MAX_TOKEN_FIT_DELTA: i64 = 50;

  /// Verify a prepared asset package, then commit it as locked.
  ///
  /// `feature_model_sha256` is the digest of the FeatherHuBERT package installed
  /// in this worker, which the caller reads out of the package manifest. It
  /// records which encoder was present when the package was locked; the feature
  /// file carries no provenance of its own, so this is a claim about the worker,
  /// not a proof about the file.
  pub fn execute_lock_asset_package(
      params: &ProjectDirParams,
      token: &CancellationToken,
      reporter: &dyn TaskReporter,
      feature_model_sha256: &str,
  ) -> CommandOutcome {
      // Admission reads a quality report and a feature file that can both be
      // large, so the stage lands before the first byte.
      reporter.report(TaskStage::Preparing, None);
      let admitted = match admit(params) {
          Ok(admitted) => admitted,
          Err(outcome) => return outcome,
      };
      // The runtime checks the token before dispatch; this covers the seconds
      // admission just spent reading.
      if token.is_cancelled() {
          return CommandOutcome::Cancelled;
      }
      let (frame_width, frame_height) =
          match verify_frames(&admitted.assets, &admitted.report, token, reporter) {
              Ok(geometry) => geometry,
              Err(outcome) => return outcome,
          };
      if let Err(outcome) = count_asset_files(&admitted.assets, admitted.report.frame_count()) {
          return outcome;
      }
      let spec = FeatureCommitSpec {
          project_root: params.project_dir.clone(),
          frame_count: admitted.report.frame_count(),
          frame_width,
          frame_height,
          landmark_model_sha256: PFLD_MODEL_SHA256.to_owned(),
          feature_model_sha256: feature_model_sha256.to_owned(),
      };
      // The commit sits past the last cancellation point on purpose: it stages,
      // backs up, renames and rolls back on failure, and interrupting it halfway
      // is exactly what would leave a package neither prepared nor locked.
      match commit_feature_artifact(&spec, &admitted.matrix) {
          Ok(artifact) => {
              let payload = lock_to_json(
                  &params.project_dir,
                  &spec,
                  &artifact,
                  admitted.token_adjustment,
              );
              CommandOutcome::Completed(Some(payload))
          }
          Err(error) => CommandOutcome::Failed(audio_task_error(&error)),
      }
  }

  /// Everything admission established, so the command body never re-reads the
  /// package.
  struct Admitted {
      assets: PathBuf,
      report: QualityReport,
      matrix: FeatureMatrix,
      token_adjustment: i64,
  }

  /// Everything that has to hold before the package is walked, ordered so that
  /// the cheapest refusal happens first.
  fn admit(params: &ProjectDirParams) -> Result<Admitted, CommandOutcome> {
      check_project_dir(&params.project_dir).map_err(CommandOutcome::Failed)?;
      let assets = params.project_dir.join(ASSETS_DIR);
      let manifest_path = assets.join(MANIFEST_FILE);
      // `commit_feature_artifact` refuses to mutate a locked package as well,
      // but only after the whole package has been walked, and it reports the
      // refusal as a mutation rather than as the state the operator asked about.
      if manifest_path.exists() {
          match read_asset_manifest(&manifest_path) {
              Ok(manifest) if manifest.state == AssetPackageState::Locked => {
                  return Err(CommandOutcome::Failed(invalid_request(
                      "素材包已加锁",
                      format!("{} is already locked", manifest_path.display()),
                  )));
              }
              Ok(_) => {}
              Err(error) => return Err(CommandOutcome::Failed(project_task_error(&error))),
          }
      }
      let report = read_quality_report(&assets.join(QUALITY_FILE))
          .map_err(|error| CommandOutcome::Failed(pipeline_task_error(&error)))?;
      if !report.anomalies().is_empty() {
          return Err(CommandOutcome::Failed(invalid_request(
              "素材包仍有异常帧",
              format!(
                  "the report carries {} anomalies",
                  report.anomalies().len()
              ),
          )));
      }
      if report.accepted_count() != report.frame_count() {
          return Err(CommandOutcome::Failed(invalid_request(
              "仍有帧未被接受",
              format!(
                  "the report accepted {} of {} frames",
                  report.accepted_count(),
                  report.frame_count()
              ),
          )));
      }
      for relative in REQUIRED_FILES {
          let path = params.project_dir.join(relative);
          if !path.is_file() {
              return Err(CommandOutcome::Failed(invalid_request(
                  "素材包缺少必需文件",
                  format!("{} is missing or not a regular file", path.display()),
              )));
          }
      }
      let matrix = read_feature_file(&params.project_dir.join(FEATURE_FILE))
          .map_err(|error| CommandOutcome::Failed(audio_task_error(&error)))?;
      // Two tokens per frame is the shape `commit_feature_artifact` enforces.
      let target_tokens = report
          .frame_count()
          .checked_mul(2)
          .and_then(|tokens| usize::try_from(tokens).ok())
          .ok_or_else(|| {
              CommandOutcome::Failed(invalid_request(
                  "素材帧数过多",
                  format!(
                      "{} frames need more tokens than this platform can address",
                      report.frame_count()
                  ),
              ))
          })?;
      let token_adjustment = target_tokens as i64 - matrix.tokens() as i64;
      if token_adjustment.abs() > MAX_TOKEN_FIT_DELTA {
          return Err(CommandOutcome::Failed(invalid_request(
              "特征令牌数与帧数不匹配",
              format!(
                  "the feature file holds {} tokens, {} frames need {target_tokens}",
                  matrix.tokens(),
                  report.frame_count()
              ),
          )));
      }
      let matrix = fit_feature_tokens(matrix, target_tokens)
          .map_err(|error| CommandOutcome::Failed(audio_task_error(&error)))?;
      Ok(Admitted {
          assets,
          report,
          matrix,
          token_adjustment,
      })
  }
  ```

  In `rust/crates/feathertalk-worker/src/lib.rs`, add `mod lock_asset_package;` immediately before the `mod lock_result;` Task 6 added, and `pub use lock_asset_package::execute_lock_asset_package;` immediately before `pub use lock_result::lock_to_json;`.

  In `rust/crates/feathertalk-worker/Cargo.toml`, move `feathertalk-pfld = { path = "../feathertalk-pfld" }` out of `[dev-dependencies]` and into `[dependencies]`, between `feathertalk-models` and `feathertalk-project`. The crate is already built for this workspace, so the change is a graph edge, not a new download; check whether it moved the lock file with `git status --short rust/Cargo.lock` and stage the lock file only if it did.

- [ ] **Step 4: Run test to verify it passes**

  Run: `cargo test -p feathertalk-worker --test lock_asset_package`

  Expected: PASS — 12 tests, 0 failed. Then `rustfmt --edition 2024 --check crates/feathertalk-worker/src/lock_asset_package.rs crates/feathertalk-worker/src/lib.rs crates/feathertalk-worker/tests/lock_asset_package.rs` and `cargo clippy -p feathertalk-worker --all-targets -- -D warnings`.

- [ ] **Step 5: Commit**

  ```powershell
  git add rust/crates/feathertalk-worker/src/lock_asset_package.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/Cargo.toml rust/crates/feathertalk-worker/tests/lock_asset_package.rs
  git commit -m "feat(worker): lock the asset package"
  ```

---

### Task 9: Dispatch the Command

**Files:**
- Modify: `rust/crates/feathertalk-worker/src/commands.rs`
- Modify: `rust/crates/feathertalk-worker/src/lib.rs`
- Test: `rust/crates/feathertalk-worker/tests/commands.rs`

**Interfaces:**
- Consumes: `crate::execute_lock_asset_package` (Task 8); `feathertalk_export::read_package_manifest` and `crate::package_task_error`, both pre-existing.
- Produces: `Request::LockAssetPackage` reaches `execute_lock_asset_package` from `execute_with_runner`, with the encoder digest already resolved. Nothing later in this plan calls the dispatcher directly; Task 10 makes the handshake agree with what this arm can now serve.

**Why now:** Task 8 finished the command but left it unreachable — `execute_with_runner` still falls through to `other => unsupported(...)`. This task is where the encoder digest comes from, and it is the one design decision left: the arm reads only `manifest.json` out of the package instead of loading the model. `FeatureModel::load`, which `extract_features` uses, maps the safetensors weights into memory; the lock runs no inference, so loading a 13 MB tensor file to read one hex string would be waste with no upside. `read_package_manifest` still runs `validate_package_directory` and `manifest.validate()`, so a broken or absent package is refused exactly as loudly, and through the same `package_task_error` mapping.

Dispatch comes before the handshake work in Task 10 so that the two never disagree in the direction that hurts: a command that is served but not advertised is invisible, whereas a command that is advertised but not served fails after a client has already started a task.

- [ ] **Step 1: Write the failing test**

  Append both tests to `rust/crates/feathertalk-worker/tests/commands.rs`, after `extract_features_without_a_model_directory_is_refused_with_its_slug`. They mirror that pair one for one; `ProjectDirParams` is already imported by the file.

  ```rust
  #[test]
  fn lock_asset_package_reports_a_package_failure_as_a_model_incompatibility() {
      let temp = tempfile::tempdir().unwrap();
      let request = Request::LockAssetPackage(ProjectDirParams {
          project_dir: temp.path().join("project"),
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
          panic!("locking with a broken package must fail");
      };
      // The package is read before the project is admitted, so an empty model
      // directory is reported as a model problem even though the project
      // directory does not exist either.
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
  fn lock_asset_package_without_a_model_directory_is_refused_with_its_slug() {
      let request = Request::LockAssetPackage(ProjectDirParams {
          project_dir: PathBuf::from("C:/tmp/project"),
      });
      let runner = FakeRunner::new(vec![]);
      let CommandOutcome::Failed(error) = execute_with_runner(
          &request,
          &bare_config(),
          &CancellationToken::new(),
          &NoReporter,
          &runner,
      ) else {
          panic!("lock_asset_package without a model directory must fail");
      };
      assert_eq!(error.code, ErrorCode::WorkerCrashed);
      assert_eq!(error.summary, "当前 worker 不支持该命令");
      assert!(
          error.detail.contains("lock_asset_package"),
          "{}",
          error.detail
      );
      error.validate().unwrap();
  }
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `cargo test -p feathertalk-worker --test commands`

  Expected: FAIL. `lock_asset_package_reports_a_package_failure_as_a_model_incompatibility` fails with `assertion \`left == right\` failed: left: WorkerCrashed, right: ModelIncompatible`, because the request still lands in the `other` arm. The second test passes before the change — the fall-through already produces that refusal — and is written anyway: once Step 3 adds the arm, it is the only test that proves an unconfigured worker still refuses by slug rather than panicking on a missing `features()`.

- [ ] **Step 3: Write minimal implementation**

  In `rust/crates/feathertalk-worker/src/commands.rs`, add the import between the `feathertalk_domain` and `feathertalk_frame_pipeline` lines, keeping the crate order:

  ```rust
  use feathertalk_export::read_package_manifest;
  ```

  Add `execute_lock_asset_package` to the `use crate::{...}` list, after `execute_extract_frames`:

  ```rust
  use crate::{
      FeatureModel, FrameModels, TaskReporter, WorkerConfig, execute_extract_features,
      execute_extract_frames, execute_lock_asset_package, is_media_cancellation, media_task_error,
      normalize_to_json, package_task_error, pipeline_task_error, probe_to_json, project_task_error,
  };
  ```

  Then add the arm to `execute_with_runner`, after `Request::ExtractFeatures` and before `other =>`:

  ```rust
          Request::LockAssetPackage(params) => {
              let Some(features) = config.features() else {
                  return CommandOutcome::Failed(unsupported(request.kind()));
              };
              // Only the package manifest is needed: the command runs no
              // inference, so mapping the safetensors weights into memory to
              // read one string would be pure waste. `read_package_manifest`
              // still runs `validate_package_directory` and
              // `manifest.validate()`, so a broken package is caught here.
              let manifest = match read_package_manifest(features.hubert_dir()) {
                  Ok(manifest) => manifest,
                  Err(error) => return CommandOutcome::Failed(package_task_error(&error)),
              };
              execute_lock_asset_package(params, token, reporter, &manifest.model.sha256)
          }
  ```

  In `rust/crates/feathertalk-worker/src/lib.rs`, extend the module documentation's served-command sentence:

  ```rust
  //! This slice serves `validate_project`, `probe_media`, `normalize_media`,
  //! `extract_frames`, `extract_features`, and `lock_asset_package` on the CPU.
  //! Every other command in [`feathertalk_domain::TaskKind`] is reported as
  //! unsupported in the handshake and rejected if a client asks for it anyway.
  ```

- [ ] **Step 4: Run test to verify it passes**

  Run: `cargo test -p feathertalk-worker --test commands`

  Expected: PASS — the two new tests plus every test the file already had, 0 failed. Then `rustfmt --edition 2024 --check crates/feathertalk-worker/src/commands.rs crates/feathertalk-worker/src/lib.rs crates/feathertalk-worker/tests/commands.rs` and `cargo clippy -p feathertalk-worker --all-targets -- -D warnings`.

- [ ] **Step 5: Commit**

  ```powershell
  git add rust/crates/feathertalk-worker/src/commands.rs rust/crates/feathertalk-worker/src/lib.rs rust/crates/feathertalk-worker/tests/commands.rs
  git commit -m "feat(worker): serve lock_asset_package"
  ```

---

### Task 10: Advertise the Command

**Files:**
- Modify: `rust/crates/feathertalk-worker/src/handshake.rs`
- Modify: `rust/crates/feathertalk-worker/src/runtime.rs`
- Test: `rust/crates/feathertalk-worker/tests/handshake.rs`, `rust/crates/feathertalk-worker/tests/runtime.rs`

**Interfaces:**
- Consumes: `feathertalk_domain::TaskKind::LockAssetPackage` (already in `TaskKind::ALL`) and `runtime::feature_reason`, both pre-existing.
- Produces: `supported_commands` lists `TaskKind::LockAssetPackage` whenever `config.features().is_some()`, and `unsupported_reason` names `FEATHERTALK_WORKER_HUBERT_DIR` when it does not. Task 11's CLI hint assumes both.

**Why now:** Task 9 made the command servable; until the handshake says so, a client has no way to know. The two edits belong together because they answer the same question from opposite sides: `supported_commands` answers "can this worker do it", `unsupported_reason` answers "what would make it possible". Leaving the second one out would fall through to the generic `_ =>` arm, which lists the supported commands but never names the variable to set — the one piece of information the operator actually needs.

The gate is `config.features().is_some()` and nothing else. The lock reads frames, landmarks, the quality report, and the feature file, all of which are already on disk by the time it runs, so it needs neither ffmpeg nor the face models. Requiring the media toolchain here would refuse a worker that is perfectly able to finish the job.

- [ ] **Step 1: Write the failing test**

  In `rust/crates/feathertalk-worker/tests/handshake.rs`, extend exactly two expectations. In `a_worker_with_a_feature_model_offers_extract_features`:

  ```rust
      assert_eq!(
          frame.supported_commands,
          vec![
              TaskKind::ValidateProject,
              TaskKind::ProbeMedia,
              TaskKind::NormalizeMedia,
              TaskKind::ExtractFrames,
              TaskKind::ExtractFeatures,
              TaskKind::LockAssetPackage
          ]
      );
  ```

  And in `a_feature_model_without_a_media_toolchain_still_offers_extract_features`:

  ```rust
      assert_eq!(
          supported_commands(&config),
          vec![
              TaskKind::ValidateProject,
              TaskKind::ExtractFeatures,
              TaskKind::LockAssetPackage
          ]
      );
  ```

  No other expectation in that file changes: `configured()` and `fully_configured()` carry no FeatherHuBERT directory, so their lists must stay exactly as they are.

  In `rust/crates/feathertalk-worker/tests/runtime.rs`, add the request helper next to `extract_features_request`:

  ```rust
  fn lock_asset_package_request() -> Request {
      Request::LockAssetPackage(ProjectDirParams {
          project_dir: PathBuf::from("C:/tmp/project"),
      })
  }
  ```

  `ProjectDirParams` is not yet imported by that file — add it to the `feathertalk_domain` import list, which already brings in `ExtractFeaturesParams`. Widen the `every_toolchain_config` doc comment, which currently claims only one command reaches the executor:

  ```rust
  /// Every toolchain resolves, so `extract_features` and `lock_asset_package`
  /// reach the executor as well.
  ```

  Then append both tests at the end of the file, after `extract_features_never_asks_for_the_media_toolchain`. Task ids continue the file's sequence, which currently ends at `00000031`:

  ```rust
  #[test]
  fn lock_asset_package_reaches_the_executor_once_the_model_directory_resolves() {
      let harness = Harness::start(every_toolchain_config(), instant_executor());
      harness.send(&start(&task("00000032"), lock_asset_package_request()));
      let frames = harness.finish();

      assert!(rejections(&frames).is_empty(), "{frames:?}");
      assert_eq!(
          stages(&frames),
          vec![
              ("1787900000000-00000032", "preparing"),
              ("1787900000000-00000032", "completed"),
          ]
      );
  }

  #[test]
  fn lock_asset_package_is_rejected_with_the_hubert_variable() {
      let harness = Harness::start(full_config(), instant_executor());
      harness.send(&start(&task("00000033"), lock_asset_package_request()));
      let frames = harness.finish();

      let reasons = rejections(&frames);
      assert_eq!(reasons.len(), 1, "{frames:?}");
      assert!(reasons[0].contains("lock_asset_package"), "{}", reasons[0]);
      assert!(
          reasons[0].contains("FEATHERTALK_WORKER_HUBERT_DIR"),
          "{}",
          reasons[0]
      );
      assert!(events(&frames).is_empty());
  }
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `cargo test -p feathertalk-worker --test handshake --test runtime`

  Expected: FAIL, four times. Both handshake tests fail on the missing `LockAssetPackage` element; `lock_asset_package_reaches_the_executor_once_the_model_directory_resolves` fails because the request is rejected instead of started, so `rejections(&frames)` is not empty; `lock_asset_package_is_rejected_with_the_hubert_variable` fails on the reason text, which lists the supported commands without naming `FEATHERTALK_WORKER_HUBERT_DIR`.

- [ ] **Step 3: Write minimal implementation**

  In `rust/crates/feathertalk-worker/src/handshake.rs`, extend the feature branch of `supported_commands`:

  ```rust
      // Feature extraction needs no media tools: it reads the wav the media
      // commands already wrote, so its only requirement is the model directory.
      if config.features().is_some() {
          commands.push(TaskKind::ExtractFeatures);
          // The lock needs the same package for a different reason: it reads the
          // encoder's digest out of the package manifest and writes it into
          // `assets.json`, which is what later runs compare against.
          commands.push(TaskKind::LockAssetPackage);
      }
  ```

  In `rust/crates/feathertalk-worker/src/runtime.rs`, add one arm to `unsupported_reason`, after the `ExtractFeatures` arm:

  ```rust
          // Feature extraction needs no media tools, so its only wall is the
          // FeatherHuBERT directory.
          TaskKind::ExtractFeatures => feature_reason(slug, config),
          // The lock reads files the earlier commands already wrote, so the
          // package directory is its only wall too.
          TaskKind::LockAssetPackage => feature_reason(slug, config),
  ```

  The two arms stay separate rather than merging into `TaskKind::ExtractFeatures | TaskKind::LockAssetPackage`: their reasons happen to coincide today, and the comment above each records why, which is the part that would be lost.

- [ ] **Step 4: Run test to verify it passes**

  Run: `cargo test -p feathertalk-worker --test handshake --test runtime`

  Expected: PASS — every test in both binaries, 0 failed. Then `rustfmt --edition 2024 --check crates/feathertalk-worker/src/handshake.rs crates/feathertalk-worker/src/runtime.rs crates/feathertalk-worker/tests/handshake.rs crates/feathertalk-worker/tests/runtime.rs` and `cargo clippy -p feathertalk-worker --all-targets -- -D warnings`.

- [ ] **Step 5: Commit**

  ```powershell
  git add rust/crates/feathertalk-worker/src/handshake.rs rust/crates/feathertalk-worker/src/runtime.rs rust/crates/feathertalk-worker/tests/handshake.rs rust/crates/feathertalk-worker/tests/runtime.rs
  git commit -m "feat(worker): advertise lock_asset_package"
  ```

---

### Task 11: The CLI Subcommand

**Files:**
- Modify: `rust/crates/feathertalk-cli/src/cli.rs`
- Modify: `rust/crates/feathertalk-cli/src/run.rs`
- Modify: `rust/crates/feathertalk-cli/src/render.rs`
- Test: `rust/crates/feathertalk-cli/tests/cli.rs`, and the inline `mod tests` in `rust/crates/feathertalk-cli/src/run.rs`

**Interfaces:**
- Consumes: `feathertalk_domain::{ProjectDirParams, Request::LockAssetPackage}` (already imported by `run.rs`) and the worker advertisement from Task 10.
- Produces: `feathertalk lock-asset-package <PROJECT_DIR>`, which builds `Request::LockAssetPackage` and inherits the existing exit codes: 0 completed, 1 task failed, 2 cancelled, 3 session error. Task 12 drives this subcommand against the real worker.

**Why now:** The worker half is finished and reachable over the protocol, so this is the last piece needed before an end-to-end run is possible. One positional argument is the whole surface: every other path the command touches — `assets/quality.json`, `assets/features/feather_hubert.f32`, `assets/assets.json` — is derived from the project directory by the worker, and letting the CLI pass any of them separately would invite a caller to name a file the manifest does not describe.

The `render.rs` hint is part of this task rather than a follow-up because the CLI's capability gate refuses the command locally, before a task starts, whenever the worker did not advertise it. Without the hint the operator gets "工作进程不支持命令 lock_asset_package" and a list of what is supported, with nothing saying that `FEATHERTALK_WORKER_HUBERT_DIR` is the fix — the same dead end Task 10 avoided on the worker side.

- [ ] **Step 1: Write the failing test**

  In `rust/crates/feathertalk-cli/tests/cli.rs`, append the unsupported-command test, mirroring `an_unsupported_extract_features_names_the_hubert_variable`:

  ```rust
  #[test]
  fn an_unsupported_lock_asset_package_names_the_hubert_variable() {
      // The fake worker advertises `validate_project` alone, so the client's
      // capability gate answers before any task starts.
      let output = run("only-validate", &["lock-asset-package", "p"]);
      assert_eq!(code(&output), 3);
      let text = stderr(&output);
      assert!(text.contains("lock_asset_package"), "{text}");
      assert!(text.contains("FEATHERTALK_WORKER_HUBERT_DIR"), "{text}");
  }
  ```

  In the inline `mod tests` of `rust/crates/feathertalk-cli/src/run.rs`, add both cases after `extract_features_carries_both_paths` and before `a_malformed_task_id_explains_the_format`:

  ```rust
      #[test]
      fn lock_asset_package_refuses_an_empty_project_directory() {
          let error = build_request(&Command::LockAssetPackage {
              project_dir: PathBuf::new(),
          })
          .expect_err("an empty project directory is refused");
          assert_eq!(error, "工程目录不能为空。");
      }

      #[test]
      fn lock_asset_package_carries_the_project_directory() {
          let request = build_request(&Command::LockAssetPackage {
              project_dir: PathBuf::from("project"),
          })
          .expect("the path is accepted")
          .expect("lock-asset-package needs a task");
          let Request::LockAssetPackage(params) = request else {
              panic!("lock-asset-package must build a LockAssetPackage request");
          };
          assert_eq!(params.project_dir, PathBuf::from("project"));
      }
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `cargo test -p feathertalk-cli`

  Expected: FAIL to compile — `no variant named \`LockAssetPackage\` found for enum \`Command\``, from both `run.rs` tests. The integration test cannot fail any earlier than that, because the crate does not build.

- [ ] **Step 3: Write minimal implementation**

  In `rust/crates/feathertalk-cli/src/cli.rs`, add the variant between `ExtractFeatures` and `Capabilities`, and name it in the enum's doc comment:

  ```rust
  /// The task commands, kebab-cased by clap: `validate-project`, `probe-media`,
  /// `normalize-media`, `extract-frames`, `extract-features`,
  /// `lock-asset-package`, `capabilities`.
  ```

  ```rust
      /// 写入素材清单并加锁素材包
      LockAssetPackage {
          /// 工程目录
          project_dir: PathBuf,
      },
  ```

  In `rust/crates/feathertalk-cli/src/run.rs`, add the arm to `build_request`, after `Command::ExtractFeatures`:

  ```rust
          Command::LockAssetPackage { project_dir } => {
              reject_empty(project_dir, "工程目录")?;
              Ok(Some(Request::LockAssetPackage(ProjectDirParams {
                  project_dir: project_dir.clone(),
              })))
          }
  ```

  There is only one argument to validate, and no `--force` or `--resume`: re-running the command on a locked package is a refusal from the worker, which the spec put out of scope for this slice.

  In `rust/crates/feathertalk-cli/src/render.rs`, extend the `UnsupportedCommand` advice chain with a fourth branch, after the `extract_features` one:

  ```rust
              } else if *requested == "lock_asset_package" {
                  text.push_str(&format!(
                      "\n{requested} 需要 FeatherHuBERT 模型包来记录编码器摘要。\
                       请用环境变量 {ENV_WORKER_HUBERT_DIR} 指定模型包目录的完整路径。"
                  ));
              }
  ```

  The wording differs from the `extract_features` branch on purpose: the model package is not used to run the encoder here, only to record which encoder produced the features, and an operator who has just been told the command "需要特征模型" would reasonably wonder why a lock needs to run inference.

- [ ] **Step 4: Run test to verify it passes**

  Run: `cargo test -p feathertalk-cli`

  Expected: PASS — every unit and integration test in the crate, 0 failed; the gated tests in `tests/real_worker.rs` still print their skip lines. Then `rustfmt --edition 2024 --check crates/feathertalk-cli/src/cli.rs crates/feathertalk-cli/src/run.rs crates/feathertalk-cli/src/render.rs crates/feathertalk-cli/tests/cli.rs` and `cargo clippy -p feathertalk-cli --all-targets -- -D warnings`.

  Also check the help text renders, since clap derives it from the doc comments: `cargo run -p feathertalk-cli --bin feathertalk -- lock-asset-package --help`. Expect `写入素材清单并加锁素材包` and one positional `<PROJECT_DIR>`.

- [ ] **Step 5: Commit**

  ```powershell
  git add rust/crates/feathertalk-cli/src/cli.rs rust/crates/feathertalk-cli/src/run.rs rust/crates/feathertalk-cli/src/render.rs rust/crates/feathertalk-cli/tests/cli.rs
  git commit -m "feat(cli): add the lock-asset-package subcommand"
  ```

---

### Task 12: Lock a Real Package End to End

**Files:**
- Test: `rust/crates/feathertalk-cli/tests/real_worker.rs`

**Interfaces:**
- Consumes: the `lock-asset-package` subcommand from Task 11, the real worker binary built by the same crate's harness, the real `ffmpeg` named by `FEATHERTALK_WORKER_FFMPEG`, and the real FeatherHuBERT package named by `FEATHERTALK_WORKER_HUBERT_DIR`.
- Produces: no new API. A gated regression test, `a_real_package_is_locked_end_to_end`, plus one file-local helper `write_frame_fixtures(assets: &Path, count: u64)`.

**Why now:** Every earlier task proved one layer against fakes. This is the only place where the whole chain runs against real inputs at once: real audio cut out of the demo clip, the real encoder producing a real `feather_hubert.f32`, and the real lock reading that file back, fitting it, committing it, and writing `assets/assets.json`. The unit tests cannot catch a mismatch between what the extractor writes and what the lock expects, because in a unit test the same test wrote both sides.

The frame stage is stood in for rather than run. `extract-frames` needs SCRFD and PFLD weights, which this repository does not ship and this machine does not have installed, so a test that shelled out to it would skip on every machine and prove nothing. The substitution is sound because of the spec decision that the lock performs *structural* verification: it stats each frame, reads its JPEG header for geometry, and parses the landmark file, but it never re-hashes a frame and never compares `frame_bytes` against the report. Nothing in the lock can observe how a frame was produced, so copying a committed 1280x720 fixture 49 times and hand-writing the matching `quality.json` exercises exactly the code paths a real extraction would.

That substitution is temporary by design: once `extract-frames` has a gated test of its own, the synthesized frames here can be replaced by its real output and the rest of the test stays as written.

The audio arrives the same way the extraction test gets it — `cut_audio` shelling out to the real ffmpeg — rather than through `normalize-media`. That is the precedent in this file, it is one process instead of two, and the lock never looks at the audio file, so routing it through the normalizer would add runtime without adding coverage.

No dependency changes: `feathertalk-cli` depends on `clap`, `ctrlc`, `feathertalk-client`, `feathertalk-domain`, `serde`, and `serde_json`, with `tempfile` as its only dev dependency. It has no path to `feathertalk-frame-pipeline` or `feathertalk-pfld`, which is why the quality report is hand-written JSON instead of `QualityReport::new` and why PFLD's digest is a literal instead of `feathertalk_pfld::PFLD_MODEL_SHA256`. Do not add a dev dependency to make either of those reachable — a test binary in the CLI crate that links the pipeline would be the only place in the workspace where the client depends on the worker's internals.

The token arithmetic is what ties the two halves together and is worth stating before the code. `cut_audio` produces two seconds of 16 kHz mono audio; `expected_hubert_frames` turns 32 000 samples into 98 tokens, and the extraction test in this same file already asserts that number. The lock demands `tokens == 2 * frame_count`, so 49 frames is the only count for which the fit is a no-op and `token_adjustment` is 0. Choosing 49 rather than a count that forces a trim keeps this test focused on the seam; Task 5 and Task 8 already cover trimming and padding against fakes.

- [ ] **Step 1: Write the gated end-to-end test**

  Append to `rust/crates/feathertalk-cli/tests/real_worker.rs`. The file already carries every helper this test needs — `worker_or_skip`, `run`, `code`, `stdout`, `stderr`, `real_tool`, `real_dir`, `demo_clip`, `cut_one_second`, `cut_audio`, `file_count` — and already imports `std::path::{Path, PathBuf}`, `std::process::{Command, Output}`, and `tempfile::TempDir`. `serde_json` is a normal dependency of this crate and the file already reaches it through its full path, so no new `use` line is needed.

  First the three constants, at the top of the file next to the existing ones:

  ```rust
  /// How many frames the locked package holds. Two seconds of audio become 98
  /// tokens (the extraction test above derives the number) and the lock demands
  /// two tokens per frame, so 49 is the only frame count that makes the fit a
  /// no-op.
  const LOCKED_FRAME_COUNT: u64 = 49;

  /// PFLD's digest, a compile-time constant in `feathertalk-pfld`, which this
  /// crate does not depend on.
  const PFLD_SHA256: &str = "e131dd764236fde54a27b2f7084906119f06c28b140bf127b459ec967e92915b";

  /// The per-frame digests the report carries. The lock verifies structure and
  /// never re-hashes a frame, so any 64-hex string does.
  const SHA256: &str = "1111111111111111111111111111111111111111111111111111111111111111";
  ```

  Then the test itself, modelled on `real_audio_becomes_features_end_to_end`:

  ```rust
  #[test]
  fn a_real_package_is_locked_end_to_end() {
      let Some(worker) = worker_or_skip("a_real_package_is_locked_end_to_end") else {
          return;
      };
      let (Some(ffmpeg), Some(hubert), Some(demo)) =
          (real_tool("FFMPEG"), real_dir("HUBERT_DIR"), demo_clip())
      else {
          println!(
              "skipping a_real_package_is_locked_end_to_end: it needs \
               FEATHERTALK_WORKER_FFMPEG, FEATHERTALK_WORKER_HUBERT_DIR, and \
               demo/feathertalk_demo_latest_188.mp4"
          );
          return;
      };
      let project = TempDir::new().expect("a temporary directory is available");
      let assets = project.path().join("assets");
      // `extract-features` creates `assets/features` itself, exactly as the
      // extraction test above relies on; only the frame directories are ours.
      for directory in ["frames", "landmarks"] {
          std::fs::create_dir_all(assets.join(directory)).expect("the assets tree is writable");
      }
      std::fs::write(project.path().join("project.json"), "{}")
          .expect("the temporary manifest is writable");
      let audio = assets.join("audio_16k_mono.wav");
      cut_audio(&ffmpeg, &demo, &audio);
      // The lock only stats the video; nothing reads it. It is cut with the same
      // helper the extraction test uses so the package keeps its real shape.
      cut_one_second(&ffmpeg, &demo, &assets.join("video_25fps.mp4"));

      let project_arg = project.path().to_string_lossy().into_owned();
      let audio_arg = audio.to_string_lossy().into_owned();
      let hubert_arg = hubert.to_string_lossy().into_owned();
      let env = [("FEATHERTALK_WORKER_HUBERT_DIR", hubert_arg.as_str())];
      let output = run(
          &worker,
          &["extract-features", &project_arg, &audio_arg],
          &env,
      );
      assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));

      write_frame_fixtures(&assets, LOCKED_FRAME_COUNT);

      let output = run(&worker, &["lock-asset-package", &project_arg], &env);
      assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));

      let features_dir = assets.join("features");
      let features = features_dir.join("feather_hubert.f32");
      let manifest_file = assets.join("assets.json");
      let result: serde_json::Value =
          serde_json::from_str(&stdout(&output)).expect("stdout is exactly one JSON document");
      assert_eq!(result["project_dir"], project.path().display().to_string());
      assert_eq!(result["manifest_file"], manifest_file.display().to_string());
      assert_eq!(result["feature_file"], features.display().to_string());
      assert_eq!(result["frame_count"], 49);
      assert_eq!(result["frame_width"], 1280);
      assert_eq!(result["frame_height"], 720);
      assert_eq!(result["tokens"], 98);
      assert_eq!(result["dims"], 1024);
      // The same 44 + 98 * 1024 * 4 bytes the extraction wrote: 49 frames need
      // exactly the 98 tokens already in the file, so the fit changes nothing.
      assert_eq!(result["bytes"], 401_452);
      assert_eq!(result["token_adjustment"], 0);
      assert_eq!(result["landmark_model_sha256"], PFLD_SHA256);
      assert_eq!(result["sha256"].as_str().unwrap().len(), 64);
      let package_manifest = std::fs::read_to_string(hubert.join("manifest.json"))
          .expect("the package manifest is readable");
      let package_manifest: serde_json::Value =
          serde_json::from_str(&package_manifest).expect("the manifest is JSON");
      assert_eq!(
          result["feature_model_sha256"],
          package_manifest["model"]["sha256"]
      );

      let written = std::fs::read(&manifest_file).expect("the locked manifest is readable");
      let written: serde_json::Value =
          serde_json::from_slice(&written).expect("the locked manifest is JSON");
      assert_eq!(written["schema_version"], 1);
      assert_eq!(written["state"], "locked");
      assert_eq!(written["video_fps"], 25);
      assert_eq!(written["audio_sample_rate"], 16_000);
      assert_eq!(written["audio_channels"], 1);
      assert_eq!(written["frame_count"], 49);
      assert_eq!(written["frame_width"], 1280);
      assert_eq!(written["frame_height"], 720);
      assert_eq!(written["feature_type"], "feather_hubert");
      assert_eq!(written["feature_shape"], serde_json::json!([49, 2, 1024]));
      assert_eq!(written["landmark_model_sha256"], PFLD_SHA256);
      // The commit rewrote the file in place at its original size.
      assert_eq!(
          std::fs::metadata(&features)
              .expect("the feature file is readable")
              .len(),
          401_452
      );
      assert_eq!(file_count(&features_dir), 1);

      let narration = stderr(&output);
      assert!(narration.contains("准备中"), "{narration}");
      assert!(narration.contains("进度 49/49"), "{narration}");
  }
  ```

  And the helper, placed directly below the test next to the other file-local helpers:

  ```rust
  /// Stand in for `extract-frames`, which needs SCRFD and PFLD this repository
  /// does not ship. Copies the committed 1280x720 fixture `count` times, writes a
  /// matching landmark file next to each frame, and hand-writes the quality report
  /// the lock reads. The digests are placeholders: the lock verifies structure and
  /// never re-hashes a frame.
  fn write_frame_fixtures(assets: &Path, count: u64) {
      let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
          .join("../feathertalk-frame-adapters/tests/fixtures/demo_frame_v1/frame.jpg");
      let frame_bytes = std::fs::read(&fixture).expect("the committed frame fixture is readable");
      let mut landmarks = String::new();
      for point in 0..110 {
          landmarks.push_str(&format!("{point} {point}\n"));
      }
      let mut frames = Vec::new();
      for index in 0..count {
          let frame_file = format!("frames/{index:06}.jpg");
          let landmark_file = format!("landmarks/{index:06}.lms");
          std::fs::write(assets.join(&frame_file), &frame_bytes).expect("the frame is writable");
          std::fs::write(assets.join(&landmark_file), &landmarks)
              .expect("the landmark file is writable");
          frames.push(serde_json::json!({
              "index": index,
              "frame_file": frame_file,
              "landmark_file": landmark_file,
              "frame_bytes": frame_bytes.len(),
              "frame_sha256": SHA256,
              "landmark_sha256": SHA256,
              "face_score": 0.9,
              "bbox": [0.0, 0.0, 64.0, 64.0],
              "blur_variance": 120.0
          }));
      }
      let report = serde_json::json!({
          "schema_version": 1,
          "frame_count": count,
          "accepted_count": count,
          "frames": frames,
          "anomalies": []
      });
      let text = serde_json::to_string_pretty(&report).expect("the report serializes");
      std::fs::write(assets.join("quality.json"), text).expect("the report is writable");
  }
  ```

  Two details in the helper are load-bearing. The landmark points are `0 0` through `109 109`, all inside a 1280x720 frame, because `read_landmark_file` from Task 4 rejects a point outside the geometry it was handed. And `accepted_count` equals `frame_count` with an empty `anomalies` array, because `QualityReport::validate` requires `accepted_count == frames.len()` and the lock refuses a report that still lists an anomaly.

  Because the report is hand-written rather than built through `QualityReport::new`, every field the deserializer and `validate()` demand has to be right by hand. `QualityReport` and `FrameQuality` are both `#[serde(deny_unknown_fields)]` with no optional fields, so a missing or misspelled key is a parse failure, not a default. The values above satisfy each constraint deliberately: the digests are 64 lowercase hex characters, `frame_bytes` is the fixture's real length and therefore non-zero, `face_score` is inside `[0, 1]`, the bbox has positive width and height, `blur_variance` is finite, and the file names are exactly `frames/{index:06}.jpg` and `landmarks/{index:06}.lms`, which is what `validate_artifact_path` accepts and what `publish.rs` writes.

- [ ] **Step 2: Run the test to verify it is neither skipped by accident nor vacuous**

  This task adds a regression test after the code it covers is already green, so there is no red phase in the usual sense. Two runs replace it.

  First, confirm the guard, from `E:\workspace\github\FeatherTalk\rust` in a shell with none of the e2e variables set:

  ```powershell
  cargo test -p feathertalk-cli --test real_worker -- --nocapture *> "$env:TEMP\ft_t12_skip.log"; "exit=$LASTEXITCODE"
  ```

  Expected: exit 0, and the log contains `skipping a_real_package_is_locked_end_to_end` alongside the three skip lines the file already prints. A test that silently passes because it never ran is the failure mode this check exists to rule out.

  Second, prove the assertions are live. Temporarily change `LOCKED_FRAME_COUNT` to `48` and run the gated command from Step 4. Expected: FAIL at `assert_eq!(result["frame_count"], 49)` with `left: 48, right: 49`. This is the honest red signal — it shows the lock really consumed the fixtures and really produced the payload the assertions read, rather than the test passing on a skip or an empty JSON document.

- [ ] **Step 3: Restore the constant**

  Set `LOCKED_FRAME_COUNT` back to `49`. There is no production code in this task: every line the test exercises was written in Tasks 1 through 11, and if the gated run needs a source change to pass, that change belongs to whichever earlier task owns the file — the plan is wrong there, not here.

- [ ] **Step 4: Run the test to verify it passes**

  From `E:\workspace\github\FeatherTalk\rust`, with the real tools named explicitly:

  ```powershell
  $env:FEATHERTALK_REQUIRE_E2E = "1"
  $env:FEATHERTALK_WORKER_FFMPEG = "D:\environment\ffmpeg\bin\ffmpeg.exe"
  $env:FEATHERTALK_WORKER_HUBERT_DIR = "C:\Users\Administrator\AppData\Local\Temp\ft_hubert_e2e\package"
  cargo test --release -p feathertalk-cli --test real_worker -- --nocapture *> "$env:TEMP\ft_t12_e2e.log"; "exit=$LASTEXITCODE"
  ```

  Expected: PASS — `test result: ok`, with `a_real_package_is_locked_end_to_end` and `real_audio_becomes_features_end_to_end` both running and the two SCRFD/PFLD tests printing their skip lines. `FEATHERTALK_REQUIRE_E2E` is what turns a missing worker binary into a failure instead of a skip, so a typo in a path shows up as a red test rather than a quiet pass. The release profile matters: a debug FeatherHuBERT forward pass over two seconds of audio is minutes, the release one is seconds.

  If `FEATHERTALK_WORKER_HUBERT_DIR` does not exist on the machine, build the package first with `rust/tools/model-package` — `feather-hubert --source <weights> --licenses <file> --destination <dir> --created-at <rfc3339> --minimum-app-version <version>` — which takes a few minutes and produces the `LICENSES.json`, `manifest.json`, `model.safetensors` trio the worker expects.

  Then `rustfmt --edition 2024 --check crates/feathertalk-cli/tests/real_worker.rs` and `cargo clippy -p feathertalk-cli --all-targets -- -D warnings`.

- [ ] **Step 5: Commit**

  ```powershell
  git add rust/crates/feathertalk-cli/tests/real_worker.rs
  git commit -m "test(cli): lock an asset package end to end"
  ```

---

### Task 13: Workspace Gates

**Files:** none. This task runs the workspace-wide checks and fixes only what they report.

**Interfaces:**
- Consumes: everything Tasks 1 through 12 committed.
- Produces: nothing. The slice is finished when all five gates are green.

**Why now:** Each earlier task ran the checks for the crate it touched, which is fast but narrow. Three things only a workspace run catches: a formatting or lint regression in a crate that was edited early and never revisited, a downstream crate that broke because `feathertalk-audio` or `feathertalk-frame-pipeline` gained a public item or an error variant, and a binary artifact that slipped into the index. Run the gates from `E:\workspace\github\FeatherTalk\rust` unless a gate says otherwise, and route cargo output to a log file rather than piping it into `Select-String`, because PowerShell wraps cargo's stderr in a `NativeCommandError` that garbles a pipeline.

- [ ] **Gate 1: Formatting**

  ```powershell
  cargo fmt --all -- --check *> "$env:TEMP\ft_g1.log"; "gate1=$LASTEXITCODE"
  ```

  Expected: `gate1=0` and an empty log. There is no `rustfmt.toml`, so the defaults apply: `max_width` 100, `fn_call_width` 60 — macro calls included — `struct_lit_width` 18, `array_width` 60, `chain_width` 60. Chinese characters count as two columns each, which is the usual reason a line with a Chinese literal is reflowed.

- [ ] **Gate 2: Lints**

  ```powershell
  cargo clippy --workspace --all-targets -- -D warnings *> "$env:TEMP\ft_g2.log"; "gate2=$LASTEXITCODE"
  ```

  Expected: `gate2=0`. Takes roughly 95 seconds warm. `--all-targets` is required: it is what puts the test files through clippy, and the new fixtures in Task 12 are the most likely source of a `needless_borrow` or `useless_vec`.

- [ ] **Gate 3: The Whole Test Suite**

  ```powershell
  cargo test --workspace --all-targets *> "$env:TEMP\ft_g3.log"; "gate3=$LASTEXITCODE"
  Select-String -Path "$env:TEMP\ft_g3.log" -Pattern '[1-9][0-9]* failed'
  ```

  Expected: `gate3=0` and no match from the `Select-String`. Budget about 48 minutes; this is the long pole of the slice, so start it and let it run rather than polling it in short waits. The pre-slice baseline is 189 test binaries, 949 passed, 0 failed, 13 ignored. Afterwards the passed count must be strictly higher — Tasks 1 through 12 add tests to `feathertalk-image`, `feathertalk-frame-adapters`, `feathertalk-frame-pipeline`, `feathertalk-audio`, `feathertalk-worker`, and `feathertalk-cli` — with `0 failed` and the ignored count still 13. A dropped total means a test file stopped compiling into the run, which passes a naive "no failures" check.

- [ ] **Gate 4: The Gated End-to-End Run**

  ```powershell
  $env:FEATHERTALK_REQUIRE_E2E = "1"
  $env:FEATHERTALK_WORKER_FFMPEG = "D:\environment\ffmpeg\bin\ffmpeg.exe"
  $env:FEATHERTALK_WORKER_HUBERT_DIR = "C:\Users\Administrator\AppData\Local\Temp\ft_hubert_e2e\package"
  cargo test --release -p feathertalk-cli --test real_worker -- --nocapture *> "$env:TEMP\ft_g4.log"; "gate4=$LASTEXITCODE"
  ```

  Expected: `gate4=0`, `test result: ok`, `a_real_package_is_locked_end_to_end` and `real_audio_becomes_features_end_to_end` both executed, and the two SCRFD/PFLD tests printing their skip lines. Gate 3 does not cover this: without the environment variables those tests skip, so the only way the real chain is exercised is a separate run. Keep the FeatherHuBERT package directory around after the run — rebuilding it costs about four minutes.

- [ ] **Gate 5: A Clean Index**

  ```powershell
  cd E:\workspace\github\FeatherTalk
  git diff --check
  git status -sb
  ```

  Expected: `git diff --check` silent — no trailing whitespace and no conflict markers — and `git status -sb` showing a clean tree apart from `?? demo/kanghui_training_video_featherhubert_188_latest/`, which stays untracked. Confirm no `.jpg`, `.mp4`, `.wav`, `.f32`, or `.safetensors` file was committed anywhere in the slice: Task 12 reads a fixture that is already tracked and writes everything else into a `TempDir`, so a media file in the history means a temporary path leaked into a commit. Also confirm `rust/Cargo.lock` is committed, since Task 8 moved `feathertalk-pfld` from `[dev-dependencies]` to `[dependencies]` in the worker's manifest.

If a gate forces an edit, stage the touched paths and commit them as `chore: satisfy the workspace lints for the lock-asset-package slice`.
