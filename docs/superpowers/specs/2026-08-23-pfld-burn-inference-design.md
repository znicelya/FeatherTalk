# PFLD Burn 推理与数值一致性设计

日期：2026-08-23
状态：已确认（执行中）

## 1. 目标

将现有 `data_utils/checkpoint_epoch_335.pth.tar` 的 PFLD 推理路径迁移为可发布的 Rust/Burn runtime，并以固定 Python/PyTorch 基线验证数值一致性。产品运行时只读取受版本控制的 safetensors artifact，不加载 pickle，不依赖 Python 或开发期导入器。

本切片完成后应具备：

- 固定生产模型：输入 `[batch, 3, 192, 192]`、输出 `[batch, 220]`；
- 严格、有限内存的 artifact manifest 与 safetensors 校验；
- CPU 全 220 个输出元素的 Python parity，`max_abs <= 1e-4`；
- 可选 WGPU parity/smoke，`max_abs <= 1e-3`，无 certified adapter 时 ignored；
- artifact 生成可重复，manifest、权重哈希和字节内容可审计；
- 继续保留 `.pth.tar` 仅供 `feathertalk-weights` 的开发期受限导入。

## 2. 非目标与安全边界

- 不在产品 runtime 中执行 pickle、任意 Python global、反序列化 callable 或 `.pth.tar` 读取。
- 不迁移 PFLD localization 分支、`auxiliarynet`、BatchNorm `num_batches_tracked` 等未用于主干推理的张量；这些必须记录在导入审计中。
- 不改变既有 110 点 `mean_face` 解码契约；mean face 继续以 crate 内固定常量发布。
- 不把未经许可的源 checkpoint 复制到发布 artifact；许可证字段保持 `NOASSERTION`，`redistribution_approved=false`。

所有输入文件、manifest 和 safetensors 均使用有界读取；长度、计数、元素总量、shape、dtype、哈希和未知字段在分配大块内存或执行图前校验。发布文件写入临时目录，完成 fsync、重新读取、逐 tensor apply 和 byte/hash 校验后原子发布，目标已存在时拒绝覆盖。

## 3. Runtime artifact 契约

发布目录仅包含：

```text
manifest.json
model.safetensors
```

manifest 使用 `serde(deny_unknown_fields)`，固定为 schema `1`：

```json
{
  "schema_version": 1,
  "model_type": "pfld_ghost_one",
  "architecture_version": "burn-pfld-inference-v1",
  "source": {
    "file_name": "checkpoint_epoch_335.pth.tar",
    "sha256": "<64 lowercase hex>"
  },
  "epoch": 335,
  "input": {"name":"input","shape":[1,3,192,192],"dtype":"f32"},
  "output": {"name":"landmarks","shape":[1,220],"dtype":"f32"},
  "model": {
    "format":"safetensors",
    "file_name":"model.safetensors",
    "sha256":"<64 lowercase hex>",
    "tensor_count": 1735,
    "total_elements": 910902
  },
  "license": {"spdx":"NOASSERTION","redistribution_approved":false}
}
```

`source` 是 provenance，不是 runtime 的读取输入；runtime 只允许打开 manifest 指定的固定相对文件名，并拒绝绝对路径、父目录遍历、符号链接、额外目录项、超大文件、错误哈希、错误 tensor 集合和错误架构/shape/dtype。artifact 中所有应用 tensor 必须是 PFLD 主干映射后的 float32，且与 `PfldGhostOne<P>` 的完整 snapshot 一一对应。

## 4. Runtime API

`feathertalk-pfld` 暴露一个不可变权重的 runtime wrapper（CPU 与 WGPU 使用相同接口）：

```rust
pub struct PfldRuntime<B: Backend> { /* manifest + loaded model */ }

impl<B: Backend> PfldRuntime<B> {
    pub fn load(dir: &Path, device: &B::Device) -> Result<Self, PfldRuntimeError>;
    pub fn forward(&self, input: Tensor<B, 4>) -> Result<Tensor<B, 2>, PfldRuntimeError>;
    pub fn manifest(&self) -> &PfldRuntimeManifest;
}
```

`load` 先读取并验证 manifest，再以固定上限读取 safetensors，最后在 detached model 上严格 apply；任何错误都不得改变 caller 状态。`forward` 在图执行前验证 rank、batch（必须为 1 的发布契约）、channel、height、width、dtype 和有限值，输出必须严格为 `[1,220]` 且有限。runtime 不暴露可变模型或任意路径加载入口。

## 5. Artifact 生成

新增受版本控制的 generator/tool：

1. 从仓库内固定 checkpoint 建立 `PfldGhostOne<CpuBackend>`；
2. 通过现有受限 importer 生成候选 safetensors；
3. 对候选文件执行 strict apply、完整 tensor snapshot、manifest schema 和哈希验证；
4. 以稳定 JSON（UTF-8、LF、末尾单换行）写入 `rust/crates/feathertalk-pfld/artifacts/pfld_ghost_one/`；
5. 生成结果必须与已提交 artifact 字节一致；源 checkpoint 哈希变化时生成/验证失败。

生成器不得把临时路径、时间戳、主机信息写入 manifest。Windows 下根 `.gitattributes` 对 manifest 和 safetensors 使用 LF byte contract，避免 `core.autocrlf` 改写提交内容。

## 6. Python parity fixture

固定 fixture 使用仓库脚本和 pinned `torch`/`numpy` 版本，直接构造确定性的 float32 输入（无需 OpenCV）：

- 每个 channel/像素由整数索引的有限公式生成；
- 保存输入 `[1,3,192,192]` 与 PyTorch 输出 `[1,220]`，另存 fixture manifest、Python 版本和模型/源 checkpoint 哈希；
- Python 预处理语义保持 `float32 / 255.0`、BGR、HWC→CHW、batch 维；
- 生成脚本在 checkpoint、模型源码或依赖版本变化时显式失败或更新 fixture id。

Rust CPU 测试读取 fixture，在 NdArray backend 上运行发布 artifact，对 220 个元素逐一计算 `max_abs`、`mean_abs` 和有限性；门槛为 `max_abs <= 1e-4`。测试还必须覆盖错误 shape、非有限输入、篡改 manifest/权重和额外文件。

WGPU 测试标为 ignored，只有检测到 certified adapter 才运行；使用相同输入和 artifact，门槛 `max_abs <= 1e-3`，并至少完成一次 forward smoke。

## 7. 验收标准

- `cargo fmt --all -- --check`、`cargo test --workspace --all-targets` 通过；
- PFLD artifact contract、strict loader、forward shape/error 和 CPU parity 全部通过；
- WGPU 测试在无适配器环境下只 ignored，不得静默回退 CPU；
- generator 二次运行结果 byte-for-byte 相同；
- `git diff --check` 通过；保留用户明确要求不提交的 demo README 及目录内容；
- 合并后从干净 worktree 重跑 PFLD 专项与 workspace 验收。
