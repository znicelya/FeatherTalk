# ONNX Opset 17 Export Report

Date: 2026-08-27

This report records the milestone-wide ONNX export and compatibility evidence for the Rust
exporter. The exported model was written only to a temporary directory under `rust/target`; no
model artifact is added to the repository.

## Scope and Contract

- Exporter: `feathertalk-export` and `feathertalk-model-package`.
- Validator: `feathertalk-onnx-validate`.
- ONNX IR: 8, default-domain opset 17, f32 tensors.
- Model: FeatherHuBERT, `waveform [1, samples] -> hidden [1, tokens, 1024]`.
- Checkpoint configuration: channels 256, expansion 2, 8 TCN blocks, output dimension 1024,
  dropout 0.0.
- Checkpoint tensors: 65 f32 tensors, 3,364,096 total elements.
- Reference input: deterministic f32 waveform `[1, 1360]`,
  `sample[i] = (i - 680) / 680`.
- Reference output: CPU Burn output `[1, 4, 1024]`.

The source checkpoint was read from the explicitly supplied absolute path:

```text
E:\workspace\github\FeatherTalk\demo\kanghui_training_video_featherhubert_188_latest\feather_hubert_188_latest_99.pth
```

Source bytes: `40,436,613`

Source SHA-256: `58df96af118d75d7f69da441e1f3960096f28dda637a4e8f4265f108d27aeb52`

The protected demo video `kanghui_training_video.MOV` was not opened, read, or modified.

## Final Export Evidence

Command:

```powershell
Set-Location rust
cargo run -p feathertalk-model-package -- onnx feather-hubert `
  --source E:\workspace\github\FeatherTalk\demo\kanghui_training_video_featherhubert_188_latest\feather_hubert_188_latest_99.pth `
  --destination target\task8-real\feather_hubert_188_latest_99-final.onnx
```

CLI report:

```json
{"model_kind":"feather_hubert","opset":17,"bytes":13501147,"sha256":"0e0425cd8a1433ac74f19c28e068b265a49e65cf913e98291e3c9f41a8d184a1"}
```

Decoded graph statistics:

| Item | Value |
| --- | ---: |
| Graph name | `feathertalk.feather_hubert` |
| Nodes | 234 |
| Initializers | 194 |
| Initializer elements | 3,365,315 |
| Initializer raw bytes | 13,461,848 |
| Input | `waveform [1, 1360]` |
| Output | `hidden [1, 4, 1024]` |

Operator counts:

```text
Add 40
Conv 32
Erf 16
InstanceNormalization 16
Mul 64
Reshape 65
Transpose 1
```

The Rust structural checker and the standalone validator both accepted the final file:

```powershell
cargo run -p feathertalk-model-package -- onnx validate `
  --source target\task8-real\feather_hubert_188_latest_99-final.onnx `
  --kind feather-hubert
cargo run -p feathertalk-onnx-validate -- `
  --model target\task8-real\feather_hubert_188_latest_99-final.onnx `
  --kind feather-hubert --structural-only
```

Both commands returned exit code 0 and reported 13,501,147 bytes and the SHA-256 above;
the standalone validator reported `provider: structural-only` and `passed: true`.

## ONNX Runtime Compatibility

The optional Rust `ort-runtime` validator used the local official ONNX Runtime 1.23.2 CPU DLL:

```text
C:\Users\Administrator\AppData\Local\Temp\feathertalk-onnxruntime-1.23.2\onnxruntime-win-x64-1.23.2\lib\onnxruntime.dll
```

DLL bytes: `14,186,016`.

Command:

```powershell
$env:ORT_DYLIB_PATH = 'C:\Users\Administrator\AppData\Local\Temp\feathertalk-onnxruntime-1.23.2\onnxruntime-win-x64-1.23.2\lib\onnxruntime.dll'
cargo run -p feathertalk-onnx-validate --features ort-runtime -- `
  --model target\task8-real\feather_hubert_188_latest_99-final.onnx `
  --kind feather-hubert `
  --input target\task8-real\waveform.npy `
  --expected-output target\task8-real\hidden-burn.npy `
  --threshold 0.0001
```

Final report:

```json
{"provider":"CPUExecutionProvider","model_bytes":13501147,"model_sha256":"0e0425cd8a1433ac74f19c28e068b265a49e65cf913e98291e3c9f41a8d184a1","input_metadata":[{"name":"waveform","shape":[1,1360],"elements":1360}],"output_metadata":{"name":"hidden","shape":[1,4,1024],"elements":4096},"max_absolute_error":0.0000014901161,"mean_absolute_error":2.1074055e-7,"threshold":0.0001,"passed":true}
```

The deterministic fixture hashes are:

```text
waveform.npy     E746802565BC5EF297AC5B580A3C5B854C46F9B9F3ADBF575D40F17D3A9B49FB
hidden-burn.npy  CE2A895B77CAFBAC8F2FA16A7B3AD299FDEBB7034986FC82087A64E94795266F
```

## Corrective Finding

The first real-model ORT run exposed two exporter bugs that structural protobuf checks did not
catch: TCN depthwise Conv omitted the Burn model's symmetric padding, and GroupNorm reshaped by
`channels/group` instead of by `groups`. Regression tests now lock the TCN pads and the correct
`[batch, groups, channels/group * spatial]` normalization layout. After those fixes, the final ORT
run above completed successfully with error well below the `1e-4` threshold.

## Verification Matrix

After the corrective changes, the focused exporter, model-package CLI, migration CLI, validator,
and audio tests passed. The complete workspace test command exited with code 0. On this Windows
host, the checkpoint symlink helpers classify Win32 `ERROR_PRIVILEGE_NOT_HELD (1314)` as an
unavailable symlink capability and skip those assertions; all other tests passed. No ONNX or
model export test failed. The verification commands exited with code 0:

```text
cargo test --workspace --all-targets
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
git diff --check
```

The checkpoint and all generated fixtures remained outside the commit; only this evidence report,
the plan checkbox updates, documentation, the symlink capability test guard, and the exporter
regression fix are tracked.
