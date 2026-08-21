# PFLD Mean Face Contract Design

Date: 2026-08-21  
Status: Approved for specification review

## Purpose

Extend `feathertalk-pfld` with strict loading for the Python baseline's `mean_face.txt`. The resulting immutable value supplies the 220 normalized coordinates added to PFLD model output before scaling to the detector crop.

## Scope

Included:

- Read a mean-face text file from a caller-provided path.
- Require valid UTF-8.
- Parse ASCII whitespace-separated `f32` tokens.
- Require exactly 220 finite values.
- Expose the parsed values through a read-only fixed-size array accessor.
- Add a typed convenience decoder accepting `&MeanFace` while preserving the existing slice-based decoder.
- Verify the repository's current `data_utils/mean_face.txt` parses successfully.

Excluded:

- PFLD checkpoint loading or model execution.
- Embedding or copying the current mean-face values into Rust source.
- Image decoding, crop construction, and pixel normalization.
- Writing or modifying `mean_face.txt`.

## Public API

```rust
pub struct MeanFace {
    values: [f32; PFLD_OUTPUT_VALUE_COUNT],
}

impl MeanFace {
    pub fn values(&self) -> &[f32; PFLD_OUTPUT_VALUE_COUNT];
}

pub fn read_mean_face(path: &Path) -> Result<MeanFace, PfldError>;

pub fn decode_landmarks_with_mean_face(
    model_output: &[f32],
    mean_face: &MeanFace,
    crop: CropGeometry,
) -> Result<PFLDLandmarks, PfldError>;
```

The existing function remains unchanged:

```rust
pub fn decode_landmarks(
    model_output: &[f32],
    mean_face: &[f32],
    crop: CropGeometry,
) -> Result<PFLDLandmarks, PfldError>;
```

The typed convenience decoder delegates to the existing numerical implementation using `mean_face.values()` so there is one coordinate-mapping path.

## Parsing Rules

`read_mean_face` applies these rules in order:

1. Read the complete file with `std::fs::read`.
2. Reject invalid UTF-8.
3. Split using `str::split_whitespace`, accepting spaces, tabs, CRLF, and newlines.
4. Parse every token as `f32`; report the zero-based token index for malformed tokens.
5. Reject `NaN`, positive infinity, and negative infinity with the token index.
6. Require exactly `PFLD_OUTPUT_VALUE_COUNT == 220` values.
7. Convert the validated vector into `[f32; 220]` without truncation or padding.

Empty files therefore fail with an invalid value count. Extra values are parsed and finite-checked before the final count error, making malformed input diagnostics deterministic.

## Error Model

Extend `PfldError` with:

```rust
Io {
    operation: &'static str,
    path: PathBuf,
    source: std::io::Error,
}
InvalidUtf8 { path: PathBuf }
InvalidMeanFaceToken { path: PathBuf, index: usize }
InvalidMeanFaceCount { path: PathBuf, expected: usize, actual: usize }
```

Existing `NonFiniteValue { field, index }` is reused with `field == "mean_face"` after token parsing. I/O errors retain their technical source, so `PfldError` will no longer derive `PartialEq` or `Eq`; tests use pattern matching on stable variants.

## Tests

Focused tests cover:

- The repository's current `data_utils/mean_face.txt`: exactly 220 values and known first/last values.
- Spaces, tabs, CRLF, and newlines.
- Missing file and invalid UTF-8.
- Malformed token with exact zero-based index.
- `NaN`, positive infinity, and negative infinity.
- Empty, 219-value, and 221-value files.
- Read-only `MeanFace::values()`.
- Exact parity between `decode_landmarks_with_mean_face` and `decode_landmarks`.
- Crate-root public imports only.

## Acceptance

```powershell
cargo fmt --check
cargo clippy -p feathertalk-pfld --all-targets --all-features -- -D warnings
cargo test -p feathertalk-pfld --all-targets
git diff --check
```

Runtime dependencies remain only the Rust standard library and `thiserror`. No Burn, image, FFmpeg, checkpoint, or GPU dependency is introduced.
