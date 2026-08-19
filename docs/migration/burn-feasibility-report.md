# Burn Feasibility Report

Generated for the Burn feasibility loop on 2026-08-19.

## Environment

- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`
- Burn: `0.21.0`
- Fixture archive: `8,719,476` bytes
- Fixture SHA-256: `9b5d341d44c336739744fe2d121b3bd0f443a90d968626e9af6ad3867a7dff27`

## CPU Evidence

Commands:

```text
cargo run -p feathertalk-parity --release -- forward --model feather --backend cpu --fixture tests/golden/burn-feasibility-v1.zip
cargo run -p feathertalk-parity --release -- forward --model unet --backend cpu --fixture tests/golden/burn-feasibility-v1.zip
cargo run -p feathertalk-parity --release -- train-step --backend cpu --fixture tests/golden/burn-feasibility-v1.zip --full
```

- FeatherHuBERT: max abs `1.0550022e-5`, mean abs `2.4652254e-6`.
- Production Original UNet: max abs `1.4901161e-7`, mean abs `1.7011772e-8`.
- Micro Adam step: initial loss relative `4.178404e-7`, post-step loss relative `3.5359878e-7`.
- Selected parameter maximum relative error: `7.203512e-6`.
- BatchNorm state maximum relative error: `8.699233e-5`.

## WGPU Evidence

The requested Windows graphics API is DX12. The runtime selected the following adapter without CPU fallback:

```json
{
  "backend": "wgpu",
  "graphics": "dx12",
  "device": "NVIDIA GeForce RTX 2060 (DiscreteGpu)",
  "used_cpu_fallback": false
}
```

Forward commands:

```text
cargo run -p feathertalk-parity --release -- probe --graphics dx12
cargo test -p feathertalk-parity --test wgpu_parity --release -- --ignored --nocapture
```

- FeatherHuBERT: max abs `7.4133277e-6`, mean abs `1.5631676e-6`.
- Production Original UNet: max abs `2.0861626e-7`, mean abs `3.8406192e-8`.
- The ignored WGPU suite passed `3/3`, including production backward and Adam.

Production training acceptance command:

```text
cargo run -p feathertalk-parity --release -- train-step --backend wgpu --fixture tests/golden/burn-feasibility-v1.zip --full
```

Result: initial loss `0.5041679`, gradient norm `0.0016970835`, output weight changed `true`, and `used_cpu_fallback=false`.

## Decision

Decision: GO
