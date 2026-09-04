# 训练执行器设计

日期：2026-09-04
状态：已定稿

## 1. 目标与范围

`2026-09-04-project-training-dataset-design.md` §2 把完整的 `train` 命令切成 A/B 两半：A 是「加锁项目目录 → 训练张量批次」，已在 `7d3e303` 收口；B 是 `train` worker 命令、VGG19 包握手宣告、CLI 子命令、门控端到端，以及两套 `TrainingMode` 枚举的映射。

B 里还藏着一件 A 与 B 的描述都没点名的东西：**把批次喂进模型、算出损失、更新权重、推进 epoch 的那个循环**。仓库今天只有一个原型：

```rust
pub fn adam_train_step<B>(
    model: OriginalUnet<B>,
    optimizer: &mut impl Optimizer<OriginalUnet<B>, B>,
    image: Tensor<B, 4>, audio: Tensor<B, 4>, target: Tensor<B, 4>,
    learning_rate: f64,
) -> (OriginalUnet<B>, f32)
```

它固定 `OriginalUnet`、固定纯 L1、不看 `TrainingMode`、不碰 DataLoader、不出指标、不写 checkpoint。迁移设计 §15.3 要求的两种 UNet、三种模式、可恢复采样、指标与预览产物，一件都没接上。

所以 B 再切一刀。本文件是 **B1：训练执行器**，产出一个新 crate `feathertalk-training-run`，把已有零件连成一台可单步驱动、可存档、可恢复的状态机。B2 是 worker 命令与 CLI 那一层。

已有零件，全部带单元测试：

- 采样：`TrainingDataLoader<D>` 的 `prepare_next_batch` / `commit_batch` / `state` / `restore`（`feathertalk-training::data`）。
- 装配：`ProjectTrainingDataset`、`TrainingItem`、`stack_single_frame_batch`、`stack_temporal_batch`（`feathertalk-training-data`）。
- 损失：`baseline_loss`、`mouth_roi_loss`、`temporal_loss` → `LossBreakdown<B>`（`feathertalk-training::losses`）。
- 存档：`save_training_checkpoint` / `load_training_checkpoint` / `TrainingCheckpointState`。
- 遥测：`TrainingMetrics`、`PreviewArtifact` 与它们的原子写入。
- 图：`OriginalUnet<B>`、`MobileOneUnet<B>` 及 `parity_micro()` 微型配置。

缺的只有中间那一层。

## 2. 切点：为什么执行器先于 worker 命令

两个理由。

第一，依赖方向单向。worker 的 `train` 命令要做的是阶段划分、进度上报、取消、按周期落盘——它需要一个「跑一步」的能力可调用。反过来执行器不需要 worker 的任何东西。迁移设计 §16 还要求每新增一个 worker 命令必须同步新增 CLI 子命令，B2 因此天然是「命令 + 子命令 + 门控端到端」的一整包，先把被调用方做出来，B2 才只剩编排。

第二，B1 可以完全离线测试。损失函数对感知特征提取器是泛型的：

```rust
pub fn baseline_loss<B: Backend, E: PerceptualFeatureExtractor<B>>(...)
```

`feathertalk-training` 的损失测试已经用 `IdentityExtractor` 这个桩注入。执行器沿用同一个泛型参数，于是 B1 的测试不需要 VGG19 权重包、不需要 ffmpeg、不需要 GPU：微型 UNet（`channels = [2, 4, 8, 16, 32]`）+ 桩提取器 + 合成加锁项目夹具，跑在 `NdArray` 的 autodiff 后端上。B2 的端到端必须有真实 VGG19 包与 HuBERT 特征，属于另一个量级的门控成本，混在一起会让 B1 的回归跑不快。

## 3. 新 crate `feathertalk-training-run`

执行器同时需要模型、损失和数据集，三者分属三个 crate，所以它必须落在一个新的上层 crate 里。三个候选位置里只有这一个成立：

- 放进 `feathertalk-training`：要给它加 `feathertalk-models` 依赖。`feathertalk-models` 带 `wgpu` / `metal` / `vulkan` 特性开关（默认 `wgpu`），把后端特性拖进当前这个只依赖 `burn` 的纯算法 crate，会让损失与 checkpoint 的测试从此绑上后端选择。否决。
- 放进 `feathertalk-training-data`：那个 crate 的职责是「项目目录 → 张量」，不依赖模型，名字也对不上。否决。
- 新 crate，依赖三方：与 `feathertalk-inference` 完全同构——那个 crate 坐在 `feathertalk-models` + `feathertalk-preprocess` + `feathertalk-media` 之上，用 `executor.rs` 里的 `execute_offline_render` 把它们连起来。本切片照抄这个形状。

```toml
[dependencies]
burn.workspace = true
feathertalk-models = { path = "../feathertalk-models", default-features = false }
feathertalk-training = { path = "../feathertalk-training" }
feathertalk-training-data = { path = "../feathertalk-training-data" }

[dev-dependencies]
feathertalk-audio = { path = "../feathertalk-audio" }
feathertalk-inference = { path = "../feathertalk-inference" }
feathertalk-preprocess = { path = "../feathertalk-preprocess" }
feathertalk-project = { path = "../feathertalk-project" }
tempfile.workspace = true
```

`default-features = false` 与 `feathertalk-inference` 一致：后端由最终二进制选，库不替它决定。dev-dependencies 那五个只为夹具服务（§15）：`feathertalk-project` 写 manifest 与加锁包，`feathertalk-preprocess` 写 landmarks，`feathertalk-audio` 写特征文件，`feathertalk-inference` 提供 `FrameReader` 桩要实现的 trait，`tempfile` 给临时项目目录。

模块划分，每个文件一件事：

```text
src/lib.rs      仅 mod + pub use
src/loss.rs     LossValues：把 LossBreakdown<B> 的张量收成 f64
src/step.rs     data_loader_config_for、train_single_frame_step、train_temporal_step
src/runner.rs   TrainingRunner、StepReport
src/preview.rs  build_preview_artifact
```

错误类型不新增。执行器的失败面全部落在既有 `TrainingError` 的变体里（`InvalidInput`、`InvalidConfig`、`InvalidCheckpoint`、`StalePreparedBatch` 等），而 `feathertalk-training-data` 已经提供 `impl From<TrainingDataError> for TrainingError`，装配失败可以直接 `?` 上来。再套一层 enum 只会让 worker 侧多写一次映射。

## 4. 可训练模型边界 `TrainableTalkingHead`

执行器必须同时支持两种 UNet，不能像 `adam_train_step` 那样写死一种。现成的 `TalkingHeadModel` 不能用——它是**推理**边界，并且故意把 MobileOne 的训练图排除在外，`inference.rs` 里那段 `compile_fail` 文档测试就是在钉这条规矩：MobileOne 必须先重参数化才能过推理边界。

于是在 `feathertalk-models::unet` 里加一个对称的训练边界，新文件 `training_graph.rs`：

```rust
pub trait TrainableTalkingHead<B: Backend> {
    fn forward_training(&self, image: Tensor<B, 4>, audio: Tensor<B, 4>) -> Tensor<B, 4>;
}

impl<B: Backend> TrainableTalkingHead<B> for OriginalUnet<B> { ... }
impl<B: Backend> TrainableTalkingHead<B> for MobileOneUnet<B> { ... }
```

`MobileOneUnetInference` 不实现它：那是重参数化之后的图，多分支已经塌缩，再训练就没有意义。两个 trait 因此把两个方向分得干净——`MobileOneUnet` 只在训练边界内，`MobileOneUnetInference` 只在推理边界内，`OriginalUnet` 两边都在。

`adam_train_step` 保持原样不动：它是里程碑一的数值门槛证据，`crates/feathertalk-models/tests/train_step.rs` 三个测试仍然在守它。执行器不复用它，因为它固定纯 L1。

## 5. 模式 → 采样 → 损失

这张表不是设计选择，是既有校验逼出来的唯一解。`TrainingConfig::validate()` 已经规定非时序模式的 `temporal_stride` 必须为 0、时序模式必须大于 0；`DataLoaderConfig::single_frame` 造出来的 `temporal_stride` 恰好是 0；`TrainingCheckpointState::validate()` 又要求 `training_config.batch_size` 与 `temporal_stride` 逐一等于 `data_loader.config` 里的对应字段。三处校验合起来只允许一种映射：

| `TrainingMode` | `SamplingKind` | `temporal_stride` | 批次 | 损失 | `LossBreakdown` 可选分量 |
| --- | --- | --- | --- | --- | --- |
| `Baseline` | `SingleFrame` | 0 | `SingleFrameBatch` | `baseline_loss` | 全部 `None` |
| `MouthRoi` | `SingleFrame` | 0 | `SingleFrameBatch` | `mouth_roi_loss` | `mouth` 有值 |
| `MouthRoiTemporal` | `TemporalPair` | `config.temporal_stride` | `TemporalBatch` | `temporal_loss` | 三者全有值 |

右边两列同时也是 `TrainingMetrics::validate()` 对三种模式的硬要求，所以损失分量到指标字段是恒等搬运，不需要任何模式判断。

映射写成一个函数，供 `TrainingRunner::new` 和 worker 侧共用：

```rust
pub fn data_loader_config_for(config: &TrainingConfig, seed: u64) -> Result<DataLoaderConfig, TrainingError>
```

它先 `config.validate()?`，再按 mode 选 `single_frame` 或 `temporal_pair`。`seed` 独立传入是因为 `TrainingConfig` 里没有 seed 字段，而 `TrainingCheckpointState::validate()` 要求 `random_seed == data_loader.config.seed`。

损失权重全部从 `TrainingConfig` 取，不用 `Default`：`BaselineLossConfig { perceptual_weight }`、`MouthRoiLossConfig { mouth_weight, perceptual_weight }`、`TemporalLossConfig` 四个权重。默认值（4.0 / 0.5 / 4.0 / 0.01）是 B2 构造 `TrainingConfig` 时的事，执行器只负责照抄。

## 6. 单帧模式的一步

```rust
pub fn train_single_frame_step<B, M, O, E>(
    model: M,
    optimizer: &mut O,
    extractor: &E,
    batch: SingleFrameBatch<B>,
    config: &TrainingConfig,
) -> Result<(M, LossValues), TrainingError>
where
    B: AutodiffBackend,
    M: TrainableTalkingHead<B> + AutodiffModule<B>,
    O: Optimizer<M, B>,
    E: PerceptualFeatureExtractor<B>,
```

顺序：

1. `let prediction = model.forward_training(batch.image, batch.audio);`
2. 按 mode 选 `baseline_loss(extractor, prediction, batch.target, &cfg)` 或 `mouth_roi_loss(extractor, prediction, batch.target, batch.mouth_mask, &cfg)`；`MouthRoiTemporal` 走到这里是编程错误，返回 `InvalidConfig`。
3. `let values = LossValues::from_breakdown(&breakdown);`，随即 `values.require_finite()?`（§8）。
4. `let gradients = GradientsParams::from_grads(breakdown.total.backward(), &model);`
5. `Ok((optimizer.step(config.learning_rate, model, gradients), values))`

第 3 步在第 5 步之前，是为了**永远不把一次 NaN 更新写进权重**。检查通过后才生成梯度、才 `step`。

批次尺寸不做校验：`prepare_next_batch` 在 epoch 末尾会给出短批（`batch_size.min(remaining)`），损失全部以 `.mean()` 收尾，短批是合法输入。

## 7. 时序模式的一步

`TemporalBatch` 的四个张量形状不对称，这是 A 切片刻意的：

```text
image      [pairs * 2, 6, 160, 160]   sample-major
audio      [pairs * 2, 16, 32, 32]    sample-major
target     [pairs, 2, 3, 160, 160]
mouth_mask [pairs, 2, 1, 160, 160]
```

UNet 的 forward 只吃 4 维，所以两个半帧摊平成行喂进去；`temporal_loss` 要 5 维才能算帧间差分，所以 target 与 mask 保留 pair 轴。衔接靠一次 reshape：

```rust
let flat = model.forward_training(batch.image, batch.audio);   // [pairs * 2, 3, 160, 160]
let prediction = flat.reshape([pairs, 2, 3, 160, 160]);
let breakdown = temporal_loss(extractor, prediction, batch.target, batch.mouth_mask, &cfg)?;
```

这一步成立的前提是行序恰好是 `(pair0.first, pair0.second, pair1.first, pair1.second, ...)`，与 target 的 pair 轴同序。`stack_temporal_batch` 的文档注释写明了这一点（“Stacks temporal pairs sample-major, so `temporal_loss` can reshape the flattened rows back”），`a_temporal_batch_is_sample_major` 在守它。行主序 reshape 把相邻两行折成 pair 轴，因此两侧对齐。

`pairs` 不从 `batch.image.dims()[0] / 2` 反推，直接取 `batch.target.dims()[0]`——少一次除法假设，且如果两者不一致，`temporal_loss` 的形状校验会当场报出来。

其余与 §6 同构：非有限检查在前，`backward` 与 `step` 在后。

## 8. 损失标量与非有限值

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossValues {
    pub total: f64,
    pub full: f64,
    pub perceptual: f64,
    pub mouth: Option<f64>,
    pub temporal: Option<f64>,
    pub temporal_mouth: Option<f64>,
}
```

`from_breakdown` 对每个分量做一次 `into_scalar()`，`Option` 结构原样保留——这样它就直接匹配 `TrainingMetrics` 的六个字段与三种模式的校验规则。

`require_finite()` 逐字段检查，第一个非有限值变成 `TrainingError::InvalidInput`，消息里带字段名和实际值。理由：`TrainingMetrics::validate()` 本来就会拒绝非有限指标，如果不在这里拦住，NaN 会先被写进权重、再在几十步之后于写指标时才爆出来，那时最近的 checkpoint 已经被污染。提前一步拦，最近的 checkpoint 就一定是干净的——这与迁移设计 §8.4「OOM 时保存最近已完成的 checkpoint，不自动修改 batch size 后继续」是同一种态度：不掩盖，交给恢复路径。

每步一次 `into_scalar()` 会同步等待设备。这是有意的：损失值本来就要立即上报，而且 §6 的顺序依赖它。

## 9. `TrainingRunner`：状态机与失败语义

```rust
pub struct TrainingRunner<B: AutodiffBackend, M, O, D: TrainingDataset<Item = TrainingItem>> {
    model: Option<M>,
    optimizer: O,
    loader: TrainingDataLoader<D>,
    config: TrainingConfig,
    device: B::Device,
    global_step: u64,
    samples_seen: u64,
}
```

感知提取器不是字段，而是 `step` 的参数。它是一份冻结的大模型，由上层（B2 的 worker）持有并在多个 runner 之间共享，塞进 runner 只会多一个泛型参数。

```rust
pub fn new(dataset: D, model: M, optimizer: O, config: TrainingConfig, seed: u64, device: B::Device) -> Result<Self, TrainingError>
pub fn restore(dataset: D, restored: RestoredTrainingState<M, O>, device: B::Device) -> Result<Self, TrainingError>
pub fn step<E: PerceptualFeatureExtractor<B>>(&mut self, extractor: &E) -> Result<StepReport, TrainingError>
pub fn epoch(&self) -> u64
pub fn global_step(&self) -> u64
pub fn samples_seen(&self) -> u64
pub fn is_finished(&self) -> bool
pub fn training_config(&self) -> &TrainingConfig
pub fn dataset(&self) -> &D
pub fn model(&self) -> Result<&M, TrainingError>
pub fn checkpoint_state(&self, asset: Provenance, model: Provenance) -> Result<TrainingCheckpointState, TrainingError>
pub fn save_checkpoint(&self, destination: &Path, descriptor: CheckpointDescriptor, asset: Provenance, model: Provenance) -> Result<TrainingCheckpointManifest, TrainingError>
pub fn metrics(&self, report: &StepReport, elapsed: Duration, gpu_memory_bytes: Option<u64>, worker_state: &str) -> Result<TrainingMetrics, TrainingError>
```

`new` 用 `data_loader_config_for` 造 `DataLoaderConfig`，再 `TrainingDataLoader::new`；`restore` 直接 `TrainingDataLoader::restore(dataset, restored.state.data_loader.clone())`，并校验 `restored.state.training_config` 与自身一致（`load_training_checkpoint` 已经比过 `CheckpointCompatibility`，这里只是不信任调用方硬塞进来的 `RestoredTrainingState`）。`global_step` 从 `restored.state.global_step` 接续，`samples_seen` 从 0 起（§11）。

`step` 的顺序：

1. `let prepared = self.loader.prepare_next_batch()?;`
2. 按 mode 堆叠批次（`stack_single_frame_batch::<B>` / `stack_temporal_batch::<B>`）。
3. 调 §6 / §7 的 step 函数，拿回新模型与 `LossValues`。
4. `self.loader.commit_batch(prepared)?;`
5. `global_step += 1; samples_seen += batch_items;`

第 4 步在第 3 步之后：**采样位置只在权重更新落地之后才前进**。反过来会在失败时留下一个「样本被消费但没被学习」的空洞。

失败即毒化。`step` 返回 `Err` 时模型已被消耗，`model` 字段留在 `None`，之后任何 `step` / `model` / `save_checkpoint` 都返回 `InvalidInput("training runner was poisoned by a failed step")`。这是刻意的：一步失败意味着这一轮不可信，恢复路径是从最近的 checkpoint 重新 `restore`，而不是在内存里带伤继续。§8 保证了那个 checkpoint 是干净的。

`is_finished()` 是 `self.loader.state().epoch >= self.config.total_epochs`。epoch 从 0 起、在一个 epoch 的最后一批 commit 之后自增，所以 `total_epochs = 3` 意味着跑到 `epoch == 3` 为止。

## 10. epoch 计数的两个口径

```rust
pub struct StepReport {
    pub epoch: u64,
    pub global_step: u64,
    pub samples_in_batch: u64,
    pub losses: LossValues,
}
```

`StepReport::epoch` 取 `prepared.epoch()`，即**这一批所属的 epoch**；`checkpoint_state().epoch` 取 `loader.state().epoch`，即 commit 之后的位置。在一个 epoch 的最后一步上，两者相差 1：报告说「第 0 轮的最后一步刚跑完」，checkpoint 说「下次从第 1 轮第 0 个样本开始」。两个口径都对，混用会让 UI 的进度条在边界上跳一格或让恢复点错一轮，所以各留一个具名测试钉住。

`TrainingCheckpointState` 的 `epoch` 字段必须等于 `data_loader.epoch`（它自己的 `validate()` 在查），这也证明 checkpoint 侧只能用后者。

## 11. 指标

`metrics()` 把 `StepReport` 与外部时间拼成 `TrainingMetrics`：

- 六个损失字段：从 `report.losses` 恒等搬运。
- `epoch` / `global_step`：来自 `report`（§10 的前一个口径）。
- `samples_seen`：本次 runner 存活期间处理的样本数，不跨恢复累计——`TrainingCheckpointState` 里没有这个字段，跨恢复的累计值无从还原，硬凑只会给出一个假数。
- `samples_per_second`：`samples_seen / elapsed.as_secs_f64()`，`elapsed` 为零时取 `0.0`。不这么防会得到 `inf`，而 `TrainingMetrics::validate()` 要求有限值，第一次上报就会失败。
- `estimated_remaining_seconds`：`remaining / rate`，`rate` 为零时取 `0.0`。`remaining = total_epochs * sample_count - (epoch * sample_count + next_position)`，用 `saturating_sub` 收口。
- `gpu_memory_bytes`、`worker_state`：调用方传入。runner 不认识设备内存，也不认识 worker 状态机。

`elapsed` 由参数注入，runner 内部不读时钟——否则速率与 ETA 都无法写确定性测试。

`sample_count` 是每个 epoch 的样本数（单帧模式为 `frame_count`，时序模式为 `frame_count - temporal_stride`）。这个公式已经存在于 `DataLoaderConfig::sample_count`，只是 `pub(crate)`。把它放宽为 `pub` 而不是在新 crate 里重算一遍——重算会在时序模式下悄悄漂移（§14）。

## 12. checkpoint 保存与恢复

`checkpoint_state()` 组装：

```rust
TrainingCheckpointState {
    schema_version: TRAINING_STATE_SCHEMA_VERSION,
    epoch: loader.state().epoch,
    global_step: self.global_step,
    random_seed: loader.state().config.seed,
    data_loader: loader.state().clone(),
    training_config: self.config.clone(),
    asset_provenance, model_provenance,
}
```

`random_seed` 从 loader 的配置取而不是另存一份，`batch_size` 与 `temporal_stride` 由 `data_loader_config_for` 保证一致——`TrainingCheckpointState::validate()` 会把这四条交叉校验全查一遍，任何一处不一致都存不进去。这是把三个校验点串起来的收益：runner 不需要自己写这些不变式，构造正确即可通过。

`descriptor`、两份 `Provenance` 都由调用方给。模型种类、架构版本、模型配置的 sha256、素材包与模型来源，都是 B2 才知道的信息；runner 硬编码任何一项都会说谎。

`save_checkpoint` 直接转调 `save_training_checkpoint::<B, M, O>`，因此 `M: AutodiffModule<B> + Clone`、`O: Optimizer<M, B> + Clone`。`AdamConfig::new().init()` 满足后者，`checkpoint_recovery.rs` 已经在这么用。

恢复的完整链路：`load_training_checkpoint` → `RestoredTrainingState` → `TrainingRunner::restore`。恢复后的 runner 必须给出与「未中断时的同一位置」逐字节相同的下一批样本；这是 §15 的核心断言。

## 13. 预览产物

```rust
pub fn build_preview_artifact<B, M, D>(
    model: &M,
    dataset: &D,
    device: &B::Device,
    sample: &TrainingSample,
    epoch: u64,
    global_step: u64,
    model_kind: &str,
    model_config_sha256: &str,
    worker_state: &str,
) -> Result<PreviewArtifact, TrainingError>
```

`PREVIEW_TENSOR_SHAPE` 是 `[3, 160, 160]`，三份张量都是这个形状。所以：

- `prediction`：把 `sample` 装成一条单帧批次，`forward_training` 之后取第 0 行。
- `target`：批次里的 target 第 0 行。
- `mouth_roi`：`prediction * mouth_mask`，掩膜从 1 通道广播到 3 通道，ROI 之外为 0。

第三份的定义是本切片的选择：迁移设计 §10.2 要求界面显示「固定样本预测、target 和嘴部 ROI」，而 mask 本身是 1 通道二值图，塞不进 `[3, 160, 160]`。显示预测帧的嘴部区域比显示掩膜本身有用——它正好是 `mouth_roi_loss` 在盯的那块像素。

`sample` 必须是 `TrainingSample::SingleFrame`；传时序样本返回 `InvalidInput`。固定样本由调用方选定并在整个训练过程中不变，这是「固定样本」的含义所在，runner 不替它挑。

前向在 autodiff 后端上跑一次然后 `detach`，不建反向图。预览按周期触发，频率远低于训练步，不值得为它引入 `valid()` 的第二套内部模块类型约束。

## 14. 对既有 crate 的最小改动

三个改动，各一行量级：

- `feathertalk-models`：新增 `unet/training_graph.rs`（`TrainableTalkingHead` 与两个 impl），`unet/mod.rs` 加 `mod` 与 `pub use`。
- `feathertalk-training`：`DataLoaderConfig::sample_count` 由 `pub(crate)` 改 `pub`（§11）。它是纯查询、已有校验、已被 `validate` 间接测过，放宽可见性不引入新行为。
- `feathertalk-training`：`TrainingDataLoader` 新增 `pub fn dataset(&self) -> &D`。`dataset` 字段是私有的，而 §13 的 `build_preview_artifact` 要拿 `&D` 去 `load_sample` 固定样本；runner 必须自己持有 loader 才能推进采样，没有这个访问器就得把数据集在 runner 里再存一份或让调用方在外面另开一个。与已有的 `state()` 完全对称，只读借用，不引入新行为。

`rust/Cargo.toml` 的 `members` 加一行。`feathertalk-training-data` 不动，worker、CLI、线协议不动。

## 15. 测试

全部离线：`NdArray` 的 autodiff 后端、`OriginalUnetConfig::parity_micro()`、桩感知提取器、合成加锁项目夹具。夹具沿用 `feathertalk-training-data/tests/support` 的做法（`GradientFrameReader` + 写出 manifest / landmarks / features），只保留驱动执行器所需的部分。

按文件：

- `tests/mode_mapping.rs`：三种模式各自映射到正确的 `SamplingKind` 与 stride；非时序模式带非零 stride 被 `TrainingConfig::validate()` 拒绝；时序模式 stride 为 0 被拒绝。
- `tests/single_frame_step.rs`：Baseline 与 MouthRoi 各跑一步；`LossValues` 的 `Option` 结构符合 §5 的表；`learning_rate = 0` 时权重不变（沿用 `train_step.rs` 的手法）；MouthRoi 的 total 大于 Baseline 的 total（mouth 项非负且权重为正）。
- `tests/temporal_step.rs`：时序一步跑通；`reshape` 后的 pair 轴与 target 同序（构造两个可区分的 pair，断言 `temporal_loss` 的 `temporal` 分量与手算的帧间差分一致）；三个可选分量都有值。
- `tests/non_finite_loss.rs`：注入一个返回 NaN 的桩提取器，断言 `step` 返回 `InvalidInput` 且消息含字段名；断言此时权重与 checkpoint 都未被改写；再断言第二次 `step` 报毒化。
- `tests/runner_progress.rs`：连续步进跨过 epoch 边界，断言 §10 的两个口径分别是 `n` 和 `n + 1`；`is_finished` 在 `epoch == total_epochs` 时为真；短批（`batch_size` 不整除 `sample_count`）能跑完且 `samples_in_batch` 正确。
- `tests/metrics.rs`：六个损失字段的搬运；`elapsed` 为零时速率与 ETA 均为 0 且 `validate()` 通过；ETA 在半程时约等于已耗时（合成数据下速率恒定）。
- `tests/checkpoint_round_trip.rs`：跑 N 步 → 存档 → `load_training_checkpoint` → `restore` → 断言下一批 `TrainingSample` 与未中断路径完全相同，且 `global_step` 接续。这是本切片的收口断言。
- `tests/preview.rs`：三份张量长度均为 `PREVIEW_TENSOR_ELEMENTS`；`mouth_roi` 在 ROI 外恒为 0、在 ROI 内等于 prediction；`write_preview_artifact` 之后 `read_preview_artifact` 能读回；传时序样本被拒绝。
- `feathertalk-models/tests/` 内补 `TrainableTalkingHead` 对两种 UNet 的形状测试，并补一段 `compile_fail` 文档测试钉住 `MobileOneUnetInference` 不在训练边界内。

不做的测试：真实 VGG19 权重下的数值门槛（属于 B2 的门控端到端）、GPU 后端（后端选择在二进制层，库测试固定 CPU）、长时间收敛性（合成夹具上「loss 下降」只能证明梯度接通，不能证明收敛）。

## 16. 验证

`cargo fmt --all -- --check`、`cargo clippy --workspace --all-targets -- -D warnings`、`cargo test --workspace --all-targets`，全部在 `rust/` 下跑。新 crate 的测试全部离线，不需要 `FEATHERTALK_REQUIRE_E2E`。

## 17. 范围外

- `train` worker 命令：阶段、进度、取消、按周期落盘与保留策略、`feathertalk_domain::TrainingMode` 与 `feathertalk_training::TrainingMode` 的显式映射。
- `feathertalk-cli` 的 `train` 子命令与门控端到端。
- VGG19 包的配置项与握手宣告。
- GPU OOM 的识别与降级（迁移设计 §8.4 的那半条）。
- MobileOne 重参数化、ONNX 导出、模型包。
- GPUI 的训练页面。

## 18. 残余风险

- **毒化语义把恢复成本推给调用方**。一次瞬时失败会让整个 runner 作废，B2 必须真的实现「从最近 checkpoint 重建」这条路径，否则一次 NaN 就是一次训练中断。取舍是明确的：让失败可见且可恢复，胜过在内存里带伤继续。
- **`samples_seen` 不跨恢复累计**，因此恢复后的速率读数在最初几步会偏低（分母是新起的耗时，分子是新起的样本数，两者同步，其实自洽，但与恢复前的曲线不连续）。真正的修法是给 `TrainingCheckpointState` 加字段，那会动 schema 版本，不属于本切片。
- **每步一次 `into_scalar()` 同步等待设备**，在 GPU 后端上会限制流水线深度。当前没有异步指标通道，而 §6 的非有限检查依赖这个同步点。等 B2 有了真实吞吐数据再决定是否值得改。
- **预览的 `mouth_roi` 定义是本切片的选择**，GPUI 页面实现时可能想要掩膜本身或叠加图。改定义要动 `PREVIEW_ARTIFACT_FORMAT` 的语义（格式与形状不变，含义变），届时需要一次显式的版本决策。
