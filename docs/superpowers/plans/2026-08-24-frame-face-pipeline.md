# 抽帧与 SCRFD/PFLD 质量管线实施计划

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

Goal: Add a deterministic, atomic frame extraction and face/landmark quality pipeline over normalized video.

Architecture: feathertalk-frame-pipeline owns fixed FFmpeg frame extraction, bounded output validation, injected decoder/detector/predictor seams, strict anomaly classification, artifact hashing, and an atomic staging/backup publisher. It consumes existing media/preprocess/face/PFLD value contracts without adding image runtime dependencies.

Tech Stack: Rust 1.92 edition 2024, serde, serde_json, sha2, thiserror, tempfile for tests.

## Global Constraints

- Source is the verified video_25fps.mp4; frame indices are exactly 0..frame_count.
- FFmpeg argv is fixed, shell-free, and paths are separate native arguments.
- Frame files are six-digit JPEG names; temporary files and backups are invocation-owned.
- JSON is bounded at 16 MiB and uses deny-unknown-fields.
- No image/OpenCV runtime dependency is introduced in this slice.
- Any anomaly fails the build and leaves previous outputs untouched.
- Every code change follows red-green-refactor.

---

### Task 1: Bootstrap crate and immutable contracts

Files:
- Modify: rust/Cargo.toml, rust/Cargo.lock
- Create: rust/crates/feathertalk-frame-pipeline/Cargo.toml
- Create: rust/crates/feathertalk-frame-pipeline/src/lib.rs, src/model.rs, src/error.rs
- Test: tests/contracts.rs

Interfaces:
- FramePipelineSpec { video_path, output_root, frame_count, image_width, image_height }
- FramePipelineSpec::validate()
- FrameQuality, QualityReport, FrameAnomaly, AnomalyCode, RecoveryAction
- read-only accessors and strict serde schemas

- [ ] Write tests for valid fixed names, rejected zero/count overflow/dimensions, anomaly serialization, and unknown fields.
- [ ] Run focused test and confirm RED.
- [ ] Implement checked constructors and serde validation.
- [ ] Run focused tests green and commit feat: define frame pipeline contracts.

### Task 2: Fixed frame extraction command and bounded runner

Files:
- Create: src/commands.rs, src/process.rs, tests/commands.rs, tests/extraction.rs

Interfaces:
- CommandSpec, FrameExtractor, ProcessRunner, ProcessOutput
- extract_frames_with_runner(spec, runner)
- six-digit paths and hostile native path preservation

- [ ] Write fake-runner tests for exact argv, command count, non-zero exit, timeout, missing/empty/symlink/oversized frame, and old-output preservation.
- [ ] Run RED.
- [ ] Implement fixed -vf fps=25, -start_number 0, -frames:v 1 per frame plus bounded capture.
- [ ] Run extraction tests green and commit feat: extract frames with bounded commands.

### Task 3: Compose SCRFD/PFLD quality evaluation

Files:
- Create: src/evaluate.rs, tests/evaluation.rs
- Modify: src/model.rs, src/error.rs, src/lib.rs
- Dependencies: feathertalk-face, feathertalk-pfld

Interfaces:
- DecodedFrame { width, height, laplacian_variance }
- FrameDecoder, FaceDetector, LandmarkPredictor
- evaluate_frames_with_models

- [ ] Write fake model tests for no/multiple face, bbox ratio, invalid landmarks, blur, model errors, score ordering, and valid .lms serialization.
- [ ] Run RED.
- [ ] Implement deterministic classification and stable summaries/actions.
- [ ] Run tests green and commit feat: evaluate frame face quality.

### Task 4: Atomic artifact publisher and strict report reader

Files:
- Create: src/publish.rs, tests/publish.rs, tests/report.rs
- Modify: src/lib.rs, src/error.rs
- Interfaces: publish_frame_artifacts, read_quality_report, rollback-capable FileOps.

- [ ] Write tests for staging collision, backup collision, late rename failure, rollback failure, hash/byte recording, malformed/oversized/unknown report fields.
- [ ] Run RED.
- [ ] Implement sibling staging, fsync, backup/rename/rollback, report validation.
- [ ] Run tests green and commit feat: publish frame quality artifacts atomically.

### Task 5: End-to-end orchestration and integration

Files:
- Modify: src/lib.rs, tests/pipeline.rs
- Add docs: docs/superpowers/specs/... and docs/superpowers/plans/...
- [ ] Write end-to-end fake runner/model test for successful N-frame build and every failure preserving old outputs.
- [ ] Run RED.
- [ ] Integrate extraction -> evaluation -> .lms write -> report -> atomic publish.
- [ ] Run focused/all-target tests.
- [ ] Run cargo fmt, cargo clippy -p feathertalk-frame-pipeline --all-targets --all-features -- -D warnings, cargo test -p feathertalk-frame-pipeline --all-targets, cargo check --workspace --all-targets, git diff --check.
- [ ] Commit feat: add frame face quality pipeline.

## Plan Self-Review

- Contract, extraction, model evaluation, report validation, atomicity, and orchestration each map to tasks.
- No image runtime or Python dependency is introduced.
- Every design limit has a test target.

