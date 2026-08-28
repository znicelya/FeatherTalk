# Worker protocol/task-domain documentation reconciliation

## Scope

Reconciled the approved design spec and implementation plan with the checked-in
`feathertalk-domain` implementation. Only these two documentation files and this
report are in scope; the master migration design, Rust sources, `Cargo.lock`, and
the demo artifact were left untouched.

## Reconciliations

- The design now describes serde adjacent tagging, matching the wire enums and
  strict unknown-field handling.
- The command inventory distinguishes 13 task commands from the control-plane
  `Cancel`, for 14 protocol operations total.
- The design's `Event` shape now includes `error: Option<TaskError>` and documents
  the required failed-stage presence/code consistency semantics.
- The error model describes all five `TaskError` fields and records the deliberate
  uppercase Serde-name exception for `ErrorCode` (`MEDIA_INVALID`, etc.). The plan
  carries the same exception in its global constraints and command-task notes.
- The plan's file tree says there are 12 parameter structs; its command task
  explains that `ValidateProject` and `LockAssetPackage` share `ProjectDirParams`.
- The plan documents the bounded `FrameReader` implementation using
  `fill_buf`/`consume`: retained content is capped at `MAX_FRAME_BYTES`, the
  newline delimiter is excluded from the limit, oversized lines are discarded
  through their delimiter to preserve synchronization, and unterminated oversized
  lines are reported at EOF.
- The stale hardcoded test total in the Definition of Done was replaced with a
  statement tied to the checked-in test files.

## Verification

From the repository root:

```text
git diff --check                         exit 0
stale-phrase search (rg)                 no matches
plan checkbox audit                       line 3 remains `[ ]`; formal task steps are `[x]`
```

No workspace-wide test run was needed for these documentation-only changes; the
implementation and domain verification were already recorded in the preceding
reports/commits.
