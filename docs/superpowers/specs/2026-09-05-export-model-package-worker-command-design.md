# Model Package Export Worker Command Design

## Goal

Implement `Request::ExportModelPackage` in `feathertalk-worker` and expose `feathertalk export-model-package <SOURCE> <DESTINATION>`. The command publishes a standard, auditable model package from a training checkpoint, which is what milestone four still owes for model tooling (migration design sections 5.5 and 15.4).

## Scope and constraints

- The wire contract in `feathertalk-domain` is frozen: `ExportModelPackageParams { source, destination }`, `TaskKind::ExportModelPackage`, `TaskStage::Preparing` and `TaskStage::Exporting`.
- The source must be an absolute path that `model_source_kind` recognises as a training checkpoint: a directory holding exactly `manifest.json`, `model.bin`, `optimizer.bin` and `training-state.json`. A published package as source is refused, because export would have nothing left to do.
- Only the two architectures a checkpoint can hold are exportable. The kind is resolved through `render_variant`, so an unknown `model_kind` is refused instead of guessed, and the configuration digest the checkpoint recorded is the gate the record is applied under. FeatherHuBERT packages come from `import-legacy-model`.
- The destination must be an absolute path that does not exist, with an existing non-symlink parent. `write_model_package` stages, validates a full save/load round trip, and publishes without clobbering.
- The license bundle is read from `LICENSES.json` beside the checkpoint directory. A checkpoint directory is allowed exactly four entries, so the bundle cannot live inside it; beside it is the same rule `import-legacy-model` already follows for its `.pth`.
- No new dependency, no new environment variable, no domain change. CPU only, no ffmpeg, and nothing under `demo/` is read.
- User-visible text is Chinese; identifiers, comments, docs and technical detail are English. Production code has no `unwrap`, `expect`, `panic!`, unchecked conversion or unchecked arithmetic.

## What the published manifest says

- `source` names the checkpoint's own weight file: `format` `feathertalk-training-checkpoint`, `identifier` the checkpoint's `model_kind`, `version` `epoch-{epoch}-step-{global_step}`, `file_name` `model.bin`, and the digest the checkpoint manifest declares. `write_model_package` re-hashes the file, so a tampered checkpoint fails admission instead of publishing.
- `training` carries the mode and the four loss weights out of `training-state.json` rather than `TrainingManifest::default()`. Section 5.5 asks the manifest to record the training mode and loss parameters, and these weights were produced by exactly that recipe.
- `created_at` is the current UTC time in RFC 3339 and `minimum_app_version` is the worker version, as in `import-legacy-model`.
- MobileOne is published reparameterised (`MobileOneUnetInference`, `reparameterized: true`): a package is an inference artifact, and both the renderer and the ONNX exporter fuse the branches before use. Original UNet is published in the shape it is trained in.

## Execution

`execute_export_model_package` admits the request, reads the checkpoint manifest and state, resolves the variant, assembles an `ExportPlan`, and hands it to `publish_checkpoint_package`, which restores the record on the autodiff backend, drops the autodiff shell with `valid`, fuses MobileOne, and calls `write_model_package`. The split mirrors `execute_render` / `run_render`: because the seam takes an already-resolved variant, tests drive a real publication with the micro configuration instead of a production-sized checkpoint.

Progress reports `Preparing` before admission, then `Exporting` `0/1` before the load and `1/1` once the package is published. Cancellation is checked before the load, after the load, and before publication, and a cancelled export leaves no destination behind. Every source, license, architecture, record or publication failure maps to `MODEL_INCOMPATIBLE`, summary `模型导出失败`, recovery `ReimportModel`, and the stage it happened in.

The completed payload is `kind`, `model_kind`, `architecture_version`, `source`, `destination`, `epoch`, `global_step`, `training_mode`, `source_sha256`, `model_sha256`, `tensor_count` and `total_elements`, with `training_mode` spelled the way `inspect_model` spells it.

## CLI

`export-model-package` takes two positional paths, rejects only empty paths locally, and preserves both exactly. Absolute paths, directory layout, licenses, architecture and destination collisions stay worker decisions. Human and JSON rendering, progress, cancellation and exit codes come from the existing runner.

## Tests

Worker tests cover handshake membership, admission (relative source, a package as source, a missing license bundle, an existing destination, an unknown model kind), cancellation before and after the load, one full publication from a micro checkpoint with manifest and payload assertions, and the command-level error mapping. The real-worker test asserts the handshake lists the command and that exporting a directory which is not a checkpoint exits 1 and publishes nothing; a production-sized happy path stays out of the automated suite for the same reason `import-legacy-model` keeps its real `.pth` path gated.
