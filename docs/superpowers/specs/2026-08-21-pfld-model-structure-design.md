# PFLD Burn Model Structure Design

Date: 2026-08-21  
Status: Approved for implementation planning

## Purpose

Define the Burn model-structure boundary for the Python `PFLD_GhostOne` backbone used by FeatherTalk. This step fixes the model graph and tensor shapes before checkpoint import and numerical parity work.

## Scope

Included:

- A generic Burn `PFLD_GhostOne<B>` model in `feathertalk-models`.
- Fixed production configuration: `width_factor = 0.5`, input size `192`, and `110` landmarks (`220` output values).
- MobileOne training-form blocks with six convolution branches, optional 1x1 scale branch, conditional BatchNorm skip branch, and optional ReLU.
- GhostOne modules and GhostOne bottlenecks matching `data_utils/base_module.py`.
- Multi-scale average pooling and concatenation before the final `1x1` output convolution.
- Shape tests for the production input and output contract.

Excluded:

- `.pth`/`.pth.tar` parsing or checkpoint import.
- Weight remapping, safetensors conversion, or numerical parity against Python.
- Image decoding, BGR/RGB conversion, resizing, normalization, SCRFD, or PFLD postprocessing.
- STN variant `PFLD_GhostOne_WithSTN` and `AuxiliaryNet`.
- Training loops, optimizer state, GPU policy, and worker integration.

## Model Contract

The public model constructor is configuration-based:

```rust
pub struct PfldConfig {
    pub width_factor: f32,
    pub input_size: usize,
    pub landmark_count: usize,
    pub num_conv_branches: usize,
}

impl PfldConfig {
    pub const fn production() -> Self;
}
```

`PfldConfig::production()` returns `0.5`, `192`, `110`, and `6`. The initialized model accepts `Tensor<B, 4>` shaped `[batch, 3, 192, 192]` and returns `Tensor<B, 2>` shaped `[batch, 220]`.

## Graph

At width factor `0.5`, the channel and spatial contract is:

| Stage | Operation | Channels | Spatial size |
|---|---|---:|---:|
| input | input tensor | 3 | 192x192 |
| conv1 | MobileOne, 3x3, stride 2 | 32 | 96x96 |
| conv2 | depthwise MobileOne, 3x3, stride 1 | 32 | 96x96 |
| x1 | average pool `input_size / 2` | 32 | 1x1 |
| conv3_1 | GhostOne bottleneck, stride 2 | 40 | 48x48 |
| conv3_2 | GhostOne bottleneck, stride 1 | 40 | 48x48 |
| conv3_3 | GhostOne bottleneck, stride 1 | 40 | 48x48 |
| x2 | average pool `input_size / 4` | 40 | 1x1 |
| conv4_1 | GhostOne bottleneck, stride 2 | 48 | 24x24 |
| conv4_2 | GhostOne bottleneck, stride 1 | 48 | 24x24 |
| conv4_3 | GhostOne bottleneck, stride 1 | 48 | 24x24 |
| x3 | average pool `input_size / 8` | 48 | 1x1 |
| conv5_1 | GhostOne bottleneck, stride 2 | 72 | 12x12 |
| conv5_2 | GhostOne bottleneck, stride 1 | 72 | 12x12 |
| conv5_3 | GhostOne bottleneck, stride 1 | 72 | 12x12 |
| conv5_4 | GhostOne bottleneck, stride 1 | 72 | 12x12 |
| x4 | average pool `input_size / 16` | 72 | 1x1 |
| conv6 | GhostOne bottleneck, stride 1 | 8 | 12x12 |
| conv7 | MobileOne, 3x3, stride 1 | 16 | 12x12 |
| conv8 | Conv block, 1x1, no BatchNorm | 64 | 1x1 |
| x5 | conv8 output | 64 | 1x1 |
| head | concatenate x1..x5, 1x1 convolution | 220 | 1x1 |
| output | flatten | 220 | N/A |

The pooled tensors are concatenated along channels, producing `256` channels before the output head.

## MobileOne Semantics

For `inference_mode = false`, each block computes:

```text
activation(scale_branch(x) + skip_branch(x) + sum(conv_branch_i(x)))
```

The scale branch exists only for kernels larger than 1. The skip branch is BatchNorm only when stride is 1 and input/output channels match. Linear blocks omit activation. The current structure does not implement reparameterization; the six-branch graph is the explicit checkpoint-compatible form for this step.

## Errors and Invariants

This structure step uses constructor assertions for invalid internal configuration, matching the existing model crate pattern. Runtime input shape validation is tested at the public forward boundary and must reject channel or spatial mismatches before model execution if the chosen Burn API permits structured validation. No checkpoint or filesystem errors are introduced here.

## Tests and Acceptance

Focused tests must verify:

- `PfldConfig::production()` exact values.
- Production model input `[1, 3, 192, 192]` returns `[1, 220]` on `CpuBackend`.
- A non-unit batch returns `[batch, 220]`.
- The public crate-root export exposes the config, model, and forward API.
- No checkpoint, image, FFmpeg, or forbidden runtime dependency is added.

Acceptance commands:

```powershell
cargo fmt --check
cargo clippy -p feathertalk-models --all-targets --all-features -- -D warnings
cargo test -p feathertalk-models --test pfld_shapes
git diff --check
```

This design does not claim PFLD numerical parity or model execution from checkpoint; those are separate follow-up contracts.
