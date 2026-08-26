# Rust 训练 Checkpoint 恢复设计

日期：2026-08-26  
状态：已确认设计，进入实现计划

## 1. 目标与范围

本切片为里程碑三补齐可恢复训练状态。一次已发布的 checkpoint 必须能够在新的进程、新的模型实例和新的 Adam 实例中恢复，并使恢复后的下一次训练更新与不中断训练在同一数值容差内等价。

本切片包含：

- Burn 模型参数记录；
- Adam optimizer 状态记录；
- `epoch`、`global_step` 和随机种子；
- 已完成的 `DataLoaderState`；
- 完整、可比较的训练配置；
- 素材包和模型输入 provenance 哈希；
- 版本化 manifest、文件哈希和严格恢复校验；
- staging 目录、fsync、manifest 最后写入和原子目录发布。

本切片不包含训练循环、指标数据库、预览图生成、GPU OOM 重试策略或 GPUI/RPC；这些消费者只依赖本切片提供的 coordinator API。

## 2. Burn 持久化决策

Burn 0.21 的 `Optimizer` record 是 `HashMap<ParamId, AdaptorRecord<...>>`。`ParamId` 由 Burn 的模型 record 和 optimizer record 各自保存；模型 record 恢复后，新的模型实例会拥有 checkpoint 中的参数 ID，随后加载 optimizer record 即可重新绑定 Adam 动量。

训练 checkpoint 使用：

```text
burn::record::BinFileRecorder<burn::record::FullPrecisionSettings>
```

模型和 optimizer 分别写入 `model.bin` 与 `optimizer.bin`。该格式保留 Burn 类型元数据、float32 精度、Adam 状态和 `ParamId`，不将 optimizer 当作 SafeTensors module 伪装。正式发布的推理模型仍按主设计使用 `model.safetensors`；两种格式的用途明确分离。

## 3. 目录和文件契约

每个 checkpoint 是一个不可变的版本目录：

```text
checkpoints/
  checkpoint-000042/
    manifest.json
    model.bin
    optimizer.bin
    training-state.json
```

目录中的条目必须恰好是上述四个文件；禁止符号链接、子目录、额外文件和半成品 manifest。写入时使用同一父目录下的唯一 staging 目录，例如 `.checkpoint-000042.<nonce>.staging`，成功后将 staging 目录原子重命名为目标目录。目标目录已存在时拒绝写入，不覆盖旧 checkpoint；调用方通过新的版本目录发布下一份 checkpoint。

`manifest.json` 使用 `TRAINING_CHECKPOINT_MANIFEST_SCHEMA_VERSION = 1`，并包含：

- `schema_version`；
- `record_format = "burn-bin-full-precision-v1"`；
- `model_kind`、`architecture_version` 和稳定的 `model_config_sha256`；
- `optimizer_kind = "adam"`、`optimizer_schema_version = 1`；
- `model`、`optimizer`、`training_state` 三个文件的文件名、字节数和 SHA-256；
- `training_state_sha256` 的重复索引（用于在不读取 tensor 前确认 state）；
- 生成时的 `burn_version` 和 `rust_version`。

`training-state.json` 使用 `TRAINING_STATE_SCHEMA_VERSION = 1`，并包含：

- `epoch`：当前 DataLoader epoch；
- `global_step`：成功提交的 optimizer 更新次数；
- `random_seed`：训练级随机种子，必须与 DataLoader 配置中的 seed 一致；
- `data_loader`：完整的 `DataLoaderState`；
- `training_config`：模式、batch size、学习率、总 epoch、temporal stride 及四个 loss 权重；
- `asset_provenance`：按稳定键排序的素材包/特征/landmark 哈希集合；
- `model_provenance`：输入模型包、VGG19 包等来源哈希集合。

所有结构体使用 `serde(deny_unknown_fields)`。哈希必须是 64 个小写十六进制字符；数值字段执行有限性、非负性和溢出校验。JSON 使用稳定字段顺序生成，便于审计和测试。

## 4. 公共 coordinator API

`feathertalk-training` 新增 `checkpoint` 模块，公开以下概念：

```rust
pub struct CheckpointFileManifest {
    pub file_name: String,
    pub bytes: u64,
    pub sha256: String,
}
pub struct Provenance {
    pub entries: BTreeMap<String, String>,
}
pub struct TrainingConfig {
    pub mode: TrainingMode,
    pub batch_size: u64,
    pub learning_rate: f64,
    pub total_epochs: u64,
    pub temporal_stride: u64,
    pub mouth_weight: f64,
    pub temporal_weight: f64,
    pub temporal_mouth_weight: f64,
    pub perceptual_weight: f64,
}
pub enum TrainingMode { Baseline, MouthRoi, MouthRoiTemporal }
pub struct TrainingCheckpointState {
    pub schema_version: u32,
    pub epoch: u64,
    pub global_step: u64,
    pub random_seed: u64,
    pub data_loader: DataLoaderState,
    pub training_config: TrainingConfig,
    pub asset_provenance: Provenance,
    pub model_provenance: Provenance,
}
pub struct TrainingCheckpointManifest {
    pub schema_version: u32,
    pub record_format: String,
    pub model_kind: String,
    pub architecture_version: String,
    pub model_config_sha256: String,
    pub optimizer_kind: String,
    pub optimizer_schema_version: u32,
    pub model: CheckpointFileManifest,
    pub optimizer: CheckpointFileManifest,
    pub training_state: CheckpointFileManifest,
    pub training_state_sha256: String,
    pub burn_version: String,
    pub rust_version: String,
}
pub struct RestoredTrainingState<M, O> {
    pub model: M,
    pub optimizer: O,
    pub state: TrainingCheckpointState,
    pub manifest: TrainingCheckpointManifest,
}

pub fn save_training_checkpoint<B, M, O>(
    destination: impl AsRef<Path>,
    model: &M,
    optimizer: &O,
    manifest: TrainingCheckpointManifest,
    state: TrainingCheckpointState,
) -> Result<(), TrainingError>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Clone,
    O: Optimizer<M, B> + Clone;

pub fn load_training_checkpoint<B, M, O>(
    directory: impl AsRef<Path>,
    model_template: &M,
    optimizer_template: &O,
    device: &B::Device,
    expected: &CheckpointCompatibility,
) -> Result<RestoredTrainingState<M, O>, TrainingError>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Clone,
    O: Optimizer<M, B> + Clone;
```

`CheckpointCompatibility` carries the expected model kind/architecture/config hash, optimizer kind/schema, training config, DataLoader frame count and provenance hashes. `load_training_checkpoint` 先完成目录枚举、JSON 严格解析、兼容性检查、文件大小和 SHA-256 校验，再调用 Burn recorder；因此模型 tensor 尚未载入时就能发现架构、配置、素材和损坏文件不匹配。

函数只返回新实例，不就地修改调用方传入的 template。模型加载使用 `template.clone().load_record(...)`，optimizer 加载使用 `template.clone().load_record(...)`；任一步骤失败都丢弃候选值并保留调用方现有状态。

## 5. 保存数据流和原子性

1. 校验 destination 是目标父目录下的非符号链接路径，目标目录不存在。
2. 校验 manifest/state 彼此一致：epoch、global step、seed、DataLoader seed、配置和 provenance 必须匹配。
3. 创建 staging 目录，并在其中写 `model.bin`、`optimizer.bin`、`training-state.json`。
4. 每个文件写完后 flush、sync data、读取并计算 SHA-256；字节数和哈希写入内存中的 manifest。
5. 对 staging 中的文件执行 `sync_all`；父目录同步使用平台适配实现（支持的平台调用 `sync_all`，不支持的平台完成文件同步后继续原子 rename）。
6. 最后写 `manifest.json`，再次 flush/sync。
7. 同步 staging 父目录后，将 staging 原子 rename 为 destination，再按同一平台适配实现同步父目录。
8. 任一步骤失败时删除当前进程拥有的 staging 目录；既有 checkpoint 不会被触碰。

manifest 只有在所有数据文件完成并校验后才出现，加载器只接受完整四文件集合。这保证取消、磁盘不足、进程崩溃或 GPU 错误不会产生“看似可恢复”的目录。

## 6. 恢复顺序和错误语义

恢复严格按以下顺序执行：

1. 检查目录、符号链接和精确文件集合；
2. 读取并验证 manifest schema、record format、模型/optimizer 标识和未知字段；
3. 读取 training state，验证 epoch/global-step、seed、DataLoader 状态和配置；
4. 对三个数据文件执行大小与 SHA-256 校验；
5. 对照 `CheckpointCompatibility` 验证模型架构、模型配置哈希、optimizer schema、素材包/模型 provenance；
6. 使用 full-precision recorder 加载模型 record；
7. 使用同一 recorder 加载 optimizer record；
8. 返回新的模型、optimizer、state 和 manifest。

错误使用现有 `TrainingError` 的 `Io`、`InvalidConfig`、`InvalidDataLoaderState`、`HashMismatch`、`Store`，并新增带字段路径和阶段的 checkpoint 错误。未知 schema、额外文件、缺文件、哈希不匹配、record 类型不匹配和 optimizer 参数集合不匹配均为硬错误；不自动修复、重置 epoch、跳过 optimizer 或降级到随机初始化。

## 7. 训练调用方契约

训练循环必须在一次 forward/backward/optimizer 更新成功后先提交 `PreparedBatch`，再递增 `global_step`，最后调用保存 coordinator。`stop-and-save` 只能发生在这个安全边界；未提交 batch 不得写入新的 DataLoader cursor。GPU OOM、取消或 worker 崩溃只保留最近一个完整 checkpoint，不能修改 batch size 或伪造 step。

## 8. 验证要求

必须包含以下测试：

1. schema-one manifest/state 的精确 JSON、未知字段、错误哈希和额外文件拒绝；
2. 模型与 Adam 记录可在新的模型/optimizer 实例中恢复，恢复后的下一步与不中断路径逐元素比较；
3. epoch、global step、DataLoader cursor 和随机 seed 精确保持；
4. 模型架构、配置、素材哈希或 optimizer schema 不匹配时，在 Burn tensor 加载前失败；
5. 缺文件、损坏文件、符号链接和 staging 目录均被拒绝或清理；
6. 已存在目标 checkpoint 不被覆盖，保存失败不改变旧目录；
7. final partial batch 在保存/恢复后仍按 DataLoader 已提交顺序继续；
8. CPU `NdArray` float32 作为强制测试后端；可用时追加 WGPU smoke test。

断点恢复的数值验收门槛沿用主设计：恢复后的下一训练 step 与连续训练在同一容差内等价；CPU float32 权重比较 `max_abs_error <= 1e-4`，损失/梯度比较 `relative_error <= 1e-3`。

## 9. 非目标与后续衔接

- 不改变现有 DataLoader 的确定性算法和 JSON schema；checkpoint 直接嵌入其 `DataLoaderState`。
- 不把训练 record 改成部署 SafeTensors，也不在本切片实现 ONNX 导出。
- 不在本切片维护 `last` 指针、周期保留策略或 GPUI 页面；后续训练执行器可在版本目录发布成功后原子更新指针。
- 不读取、修改、暂存、提交或删除 `demo/kanghui_training_video_featherhubert_188_latest/`。
