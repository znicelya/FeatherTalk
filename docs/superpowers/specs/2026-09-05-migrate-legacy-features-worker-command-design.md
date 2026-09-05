# Legacy Feature Migration Worker Command Design

## Goal

Implement `Request::MigrateLegacyFeatures` in `feathertalk-worker` and expose `feathertalk migrate-legacy-features <SOURCE> <DESTINATION>`. The command converts a legacy NumPy `.npy` feature matrix into FeatherTalk's versioned `.f32` artifact without overwriting existing files.

## Scope and constraints

- The wire contract in `feathertalk-domain` is frozen: use `MigrateLegacyFeaturesParams`, `TaskKind::MigrateLegacyFeatures`, `TaskStage::Preparing` and `TaskStage::Importing` as already defined.
- The source must be an absolute, regular non-symlink file, no larger than `MAX_FEATURE_FILE_BYTES`.
- The source must decode as finite `f32` values with rank three and shape `[video_frames, 2, 1024]`, where `video_frames > 0`.
- The destination must be an absolute path that does not exist. Its parent must already exist and must not be a symlink. The existing audio writer performs no-clobber staging and atomic publication.
- No new dependency, environment variable, or domain change is allowed. The command is CPU-only and never shells out to ffmpeg.
- User-visible text is Chinese; identifiers, comments, docs, and technical details are English. Production code has no `unwrap`, `expect`, `panic!`, unchecked conversion, or unchecked multiplication.
- Never read, stage, modify, or delete anything under `demo/`.

## Execution

The handshake always advertises `migrate_legacy_features`, after `import_legacy_model`. Execution reports `Preparing` before admission and `Importing` before decoding/conversion. Cancellation is checked before reading, after decoding, and before publication. A successful result contains the source shape, token count, dimensions, output bytes, and output SHA-256. Any source, NPY, matrix, or write error maps to `MODEL_INCOMPATIBLE`, summary `特征迁移失败`, stage `Importing`, and recovery `ReimportModel`; cancellation remains a cancelled task.

The implementation reuses `ndarray_npy::ReadNpyExt`, `FeatureMatrix::new`, and `write_feature_file_no_clobber`. The worker module owns path admission and JSON shaping so CLI and direct callers share exactly one behavior.

## CLI

The clap command has two positional paths. It rejects only empty paths locally and preserves the paths exactly in the request. Absolute-path, filesystem, extension, NPY dtype/rank/shape, and destination checks remain worker decisions. Normal human, JSON, progress, cancellation, and exit-code handling uses the existing CLI runner.

## Tests

Unit tests cover handshake ordering, request construction, source/destination admission, invalid rank/shape/dtype, cancellation before read and before publication, successful conversion and payload, and no-clobber behavior. A gated real-worker test uses only an explicit `FEATHERTALK_WORKER_LEGACY_FEATURES` source path and skips when absent; it never scans or opens `demo/`.
