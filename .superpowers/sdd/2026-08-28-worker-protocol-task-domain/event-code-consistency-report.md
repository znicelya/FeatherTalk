# Event error-code consistency fix

## Status

Implemented and committed on `worker-protocol-task-domain`.

## TDD record

1. Added `a_failed_stage_and_error_payload_must_use_the_same_code` to
   `rust/crates/feathertalk-domain/tests/event.rs`. The test constructs a
   `TaskStage::Failed { code: DiskSpaceLow, .. }` with an embedded
   `TaskError { code: GpuDeviceLost, .. }` and requires validation to reject
   the frame.
2. Ran the red test before changing production code:

   ```text
   cargo test -p feathertalk-domain --test event a_failed_stage_and_error_payload_must_use_the_same_code
   ```

   Result: failed as expected. The test panicked at `event.rs:63` because the
   existing implementation returned `Ok(())` for the mismatched codes.
3. Updated `Event::validate` to compare the failed stage code with
   `Event.error.code` before validating the `TaskError`; mismatch returns
   `DomainError::InvalidField { field: "error", .. }`.

## Verification

All commands were run from `E:\workspace\github\FeatherTalk\rust` after the
implementation change:

```text
cargo test -p feathertalk-domain --test event
```

Result: exit 0; 7 tests passed, 0 failed.

```text
cargo test -p feathertalk-domain --all-targets
```

Result: exit 0; all domain test binaries passed (including 7 event tests), 0
failures.

```text
cargo fmt --all -- --check
```

Result: exit 0.

```text
cargo clippy -p feathertalk-domain --all-targets -- -D warnings
```

Result: exit 0.

## Commit

The code and regression test are committed in `d4f0904`. This report is
committed in a follow-up commit; both SHAs are reported in the handoff
message.

## Concerns

None for this scoped change. The check applies only when the event stage is
`Failed`; non-failed/error-presence rules remain unchanged.
