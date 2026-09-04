# 训练 worker 命令与 CLI 子命令设计

日期：2026-09-04
状态：已定稿

## 1. 目标与范围

上一切片交付了 `feathertalk-training-run`：`TrainingRunner` 能把数据集、模型、优化器装在一起，逐步训练、跨 epoch 前进、存取检查点、产出指标与预览产物。它至今没有调用方。本切片把它接到线协议上：worker 实现 `Request::Train`，CLI 增加 `train` 子命令，训练产物落进工程目录。

领域契约早已提交，本切片一个字节都不改：`TaskKind::Train`（slug `train`）、`TrainParams { project_dir, mode, variant, epochs, resume }`（带 `deny_unknown_fields`）、`Request::Train`、`TaskStage::Training { epoch, step, loss }`，`render.rs` 的中文标签「正在训练」也已就位。`commands.rs` 目前把 `Request::Train` 落到 `other => Failed(unsupported(...))` 分支。

本切片范围内的改动：

- `feathertalk-worker`：新增 `training.rs`（后端别名、模型与描述符装配、检查点发布、遥测落盘）、`train.rs`（命令编排）、`train_result.rs`（结果载荷）；`config.rs` 新增 `ENV_VGG19_DIR` 与 `TrainingToolchain`；`handshake.rs` 宣告 `Train` 与 `Capabilities.training`；`runtime.rs` 增加拒绝文案并把执行线程的栈提到 64 MiB；`error_map.rs` 新增 `training_task_error` 与 `training_data_task_error`；`Cargo.toml` 新增 `feathertalk-training`、`feathertalk-training-data`、`feathertalk-training-run`、`burn` 依赖，并把 `hex`、`sha2` 从 dev-dependencies 提为正式依赖。
- `feathertalk-training-data`：新增 `FrameSample::new`，校验四个平面的长度（`[6, 160, 160]`、`[16, 32, 32]`、`[3, 160, 160]`、`[1, 160, 160]`）。worker 的单元测试要构造 `TrainingItem`，而 `FrameSample` 的四个字段是私有的、只有 `ProjectTrainingDataset` 能填；没有这个构造函数，worker 就得把 `feathertalk-training-run/tests/fixture/mod.rs` 那套 180 行加锁工程夹具复制一遍，把编排层的测试绑死在另一个 crate 的磁盘格式上。
- `feathertalk-cli`：新增 `train` 子命令、对应的 `build_request` 分支与 `UnsupportedCommand` 提示分支。

一个直接结论：本命令只需要 `FEATHERTALK_WORKER_VGG19_DIR` 一个新环境变量，不需要 ffmpeg、SCRFD、PFLD 或 HuBERT——帧、关键点与音频特征都已经在加锁后的工程目录里（见 §11）。

范围外的内容集中在 §18。

## 2. 一次训练运行的装配

链路全部由已提交的 API 组成，worker 只负责装配与编排：

```rust
type TrainBackend = CpuAutodiffBackend; // feathertalk_models::backend，即 Autodiff<NdArray<f32>>

let dataset = ProjectTrainingDataset::open(&params.project_dir)?; // 要求工程已加锁
let frame_count = dataset.frame_count();
let extractor = load_vgg19_package::<TrainBackend>(toolchain.vgg19_dir(), &device)?;
let model = OriginalUnetConfig::production().init::<TrainBackend>(&device);
let optimizer = AdamConfig::new().init::<TrainBackend, OriginalUnet<TrainBackend>>();
let mut runner = TrainingRunner::new(dataset, model, optimizer, config, TRAINING_SEED, device)?;
while !runner.is_finished() {
    let report = runner.step(&extractor)?;
}
```

三点值得点明。

第一，VGG19 必须加载在**自动微分**后端上。`TrainingRunner::step` 的约束是 `E: PerceptualFeatureExtractor<B>`，而这里的 `B` 就是 `AutodiffBackend`；把感知特征提取器放在裸后端上，感知损失的梯度到不了生成器，`perceptual_weight` 会变成一个不起作用的数字。

第二，`runner.step` 内部已经按模式选择单帧或时序步骤、做 `require_finite` 检查、用 `checked_add` 计数。worker 不重复这些判断。

第三，一步失败后 runner 被毒化（`model` 已被 take 走），所以命令体在 `step` 返回 `Err` 时直接走错误映射，不做补救性保存——那会把一个已知损坏的状态写进检查点。

## 3. 模式、变体与每 epoch 样本数

领域枚举与训练 crate 的枚举同名但不同集合，必须显式映射（导入时别名 `DomainTrainingMode`）：

| `TrainParams.mode` | `TrainingConfig.mode` | `temporal_stride` | 每 epoch 样本数 |
| --- | --- | --- | --- |
| `baseline` | `Baseline` | 0 | `frame_count` |
| `mouth_roi` | `MouthRoi` | 0 | `frame_count` |
| `temporal` | `MouthRoiTemporal` | 1 | `frame_count - 1` |

`temporal_stride` 不能随便填：`TrainingConfig::validate` 要求非时序模式必须为 0、时序模式必须大于 0，而 `DataLoaderConfig::sample_count` 按 `frame_count - stride` 算时序样本数。

`TrainParams.variant` 决定模型类型，两者都实现了 `TrainableTalkingHead`：`original_unet` 对应 `OriginalUnet`，`mobileone_unet` 对应 `MobileOneUnet`。

## 4. 为什么本切片只跑 CPU

workspace 的 burn 特性里有 `wgpu`，`feathertalk_models::backend` 也已经有 `GpuAutodiffBackend` 别名，`Autodiff<Wgpu>` 是能编出来的。本切片仍然只跑 CPU，理由有三条。

迁移设计 §4.1 要求「应用必须显示实际 adapter、backend 和显存信息，不允许将 GPU 请求静默回退到 CPU」。worker 现在的握手是硬编码的 `backends: [Cpu]`、`adapters: [cpu-0]`、`wgpu_training: false`；要诚实地宣告 GPU，得先做 adapter 枚举、显存读取、`GPU_OUT_OF_MEMORY` 与 `GPU_DEVICE_LOST` 的映射，还要让 `AdapterLocks` 管住真实设备。这是一条独立切片。本切片把 `wgpu_training` 保持 false，等于 worker 从不承诺 GPU，也就不存在静默回退。

runner、损失与感知提取器全部对 `B: AutodiffBackend` 泛型，换后端是替换一个类型参数，不是重写编排。先把「协议 → CLI → 磁盘产物 → 续训」这条纵向链路打通，GPU 切片只需要在 §7 的分发处多一层。

代价要说清楚：`production()` 通道 `[32, 64, 128, 256, 512]` 在 160×160 上做前向加反向，CPU 单步是秒级。本命令在真实数据集上「正确但慢」，它现在的用途是把产物格式、续训与取消语义钉死，不是产出可用权重。

否决方案：本切片同时支持两个后端，用环境变量选。第一，变体（2）× 后端（2）会产生 4 份单态化的 runner，编译期与二进制体积翻倍，其中两份没有任何测试能覆盖（本机与 CI 的单元测试都不碰 GPU）。第二，没有 adapter 枚举与显存读取，`Metrics.vram_bytes` 与 §4.1 的显示要求依然满足不了，等于付出成本却拿不到那条规则要的东西。

## 5. 超参数的来源

`TrainParams` 只带四个字段，`TrainingConfig` 有九个。缺的五个，加上随机种子与保存频率，只能是 worker 侧常量——给请求加字段是线协议变更（§18）。

| 字段 | 取值 | 来源 |
| --- | --- | --- |
| `mode` | 见 §3 | `params.mode` |
| `total_epochs` | `u64::from(params.epochs)` | `params.epochs` |
| `temporal_stride` | 0 或 1 | 见 §3 |
| `batch_size` | `DEFAULT_BATCH_SIZE = 1` | worker 常量 |
| `learning_rate` | `DEFAULT_LEARNING_RATE = 1e-4` | worker 常量 |
| `mouth_weight` | 4.0 | 迁移设计 §8.2 |
| `temporal_weight` | 0.5 | 迁移设计 §8.2 |
| `temporal_mouth_weight` | 4.0 | 迁移设计 §8.2 |
| `perceptual_weight` | 0.01 | 迁移设计 §8.2 |
| 随机种子 | `TRAINING_SEED = 1` | worker 常量 |
| 保存频率 | 每个 epoch 边界 | worker 常量 |

`batch_size = 1` 的理由：线协议没有这个字段，任何取值都只是占位；CPU 上单步耗时由样本数主导，批大小只是拿内存换吞吐；取 1 让峰值内存最小，也让「一步等于一个样本」把进度与损失的粒度做到最细。这些常量集中在 `training.rs` 顶部，等协议加上字段时只有一处要改。

还有一条由检查点契约推出的硬约束：`CheckpointCompatibility::validate_manifest_state` 会逐字段比较 `training_config`，所以**续训必须用与首次运行完全相同的 mode、variant 与 epochs**，上表的常量也不能在两次运行之间改动，否则续训被拒（见 §8）。

## 6. 配置与握手

`config.rs` 新增：

```rust
pub const ENV_VGG19_DIR: &str = FEATHERTALK_WORKER_VGG19_DIR;

pub struct TrainingToolchain {
    vgg19_dir: PathBuf,
}
```

`WorkerConfig` 新增 `training: Option<TrainingToolchain>` 与 `training_rejection: Option<String>`，读取器 `training()`、`training_rejection()`。路径校验复用 `required_path`（非空且绝对，启动时不碰文件系统），与 `FeatureToolchain` 完全同形。训练工具链独立解析：训练与 SCRFD/PFLD/HuBERT 没有关系，只配了 VGG19 的 worker 应该照样能宣告 `train`。

构造函数再加一个兄弟函数，既有签名不动：

```rust
pub fn from_values_with_training(ffprobe, ffmpeg, timeout_ms, scrfd_dir, pfld_dir, hubert_dir, vgg19_dir) -> Self;
pub fn from_values_with_toolchains(...) -> Self; // 委托，vgg19 传 None
```

握手两处改动：`supported_commands` 在 `config.training().is_some()` 时追加 `TaskKind::Train`；`ready_frame` 的 `Capabilities.training` 从硬编码 `false` 改为 `config.training().is_some()`，`wgpu_training` 保持 false（§4）。`backends` 与 `adapters` 不动——训练跑在同一个 `cpu-0` 上，`AdapterLocks` 因此天然保证训练不与其他任务并发。

`runtime::unsupported_reason` 新增 `TaskKind::Train => training_reason(slug, config)`，形状照既有的 `feature_reason`：有 `training_rejection()` 就用它，否则给出「未配置 `FEATHERTALK_WORKER_VGG19_DIR`」的说明。

## 7. 变体分发与检查点描述符

`TrainingRunner<B, M, O, D>` 对模型泛型，两个变体就是两次单态化。命令体因此写成泛型函数，只在入口分发一次：

```rust
match params.variant {
    UnetVariant::OriginalUnet => {
        let config = OriginalUnetConfig::production();
        let configuration = ModelConfiguration::original_unet(&config);
        start::<OriginalUnet<TrainBackend>, _>(configuration, |device| config.init(device), ...)
    }
    UnetVariant::MobileOneUnet => {
        let config = MobileOneUnetConfig::production();
        let configuration = ModelConfiguration::mobileone_unet(&config, false);
        start::<MobileOneUnet<TrainBackend>, _>(configuration, |device| config.init(device), ...)
    }
}
```

`mobileone_unet(&config, /* reparameterized */ false)`：训练用多分支形态，重参数化是导出阶段的事，把 true 写进描述符会让检查点声称一个训练时并不存在的结构。

描述符三要素全部从 `ModelConfiguration` 推出，不手写字面量：

```rust
let model_kind = configuration.model_type();                     // original_unet / mobileone_unet
let architecture_version = configuration.architecture_version();  // original-unet-burn-v1 / ...
let model_config_sha256 = hex::encode(Sha256::digest(serde_json::to_vec(&configuration)?));
let descriptor = CheckpointDescriptor::new(model_kind, architecture_version, model_config_sha256);
```

`ModelConfiguration` 是 `#[serde(tag = kind)]` 的固定字段结构，没有映射类型，序列化字节序稳定，所以它的 sha256 就是「模型配置」这个概念的自然规范形式；`descriptor.validate()` 要求 64 位小写十六进制，`hex::encode` 正好给出小写。这是 workspace 里第一处真正计算 `model_config_sha256` 的代码——既有测试都用 `1.repeat(64)` 占位。`sha2` 与 `hex` 因此从 dev-dependencies 提为正式依赖。

否决方案：给两个变体做一个 `enum TrainableUnet` 包装再实现 `TrainableTalkingHead`。`TrainingRunner` 还要求 `AutodiffModule<B> + Clone`，手写 `AutodiffModule`（`InnerModule`、record 关联类型、`valid`）的代价远超再单态化一份编排函数。

## 8. 检查点布局、发布与续训

目录布局（迁移设计 §6.1）：

```
<project>/models/unet/checkpoint-00000188/{manifest.json,model.bin,optimizer.bin,training-state.json}
```

名字用 8 位补零的 `global_step`，不用 epoch。取消可能发生在 epoch 中间，而 `DataLoaderState.next_position` 完全能表达 epoch 内的位置（`TrainingCheckpointState::validate` 只要求 `next_position` 落在当前 epoch 内），用步号命名可以让「epoch 边界保存」与「取消时保存」共用一套命名，续训只需取最大步号。epoch 在 `training-state.json` 里，不必出现在目录名上。

`save_training_checkpoint` 会拒绝已存在的目标（`checkpoint.rs:365`）。续训一旦重复走过同一个步号（例如从第 188 步续训、又在同一位置被取消），命名就会撞上，训练会在第一个保存点直接失败。因此 worker 侧加一个发布例程：

1. `staged = models/unet/.publish-{pid}-{n}`，`n` 是进程内递增计数，`create_dir` 撞名就换号；
2. `runner.save_checkpoint(&staged, descriptor)`，写完即是一个完整检查点；
3. 目标已存在则先 `rename` 到 `.retired-{pid}-{n}`；
4. `rename(staged, final)`；
5. 尽力删除 `.retired-*`，失败只记录、不致命。

任何时刻磁盘上都至少有一个完整检查点；第 3 与第 4 步之间崩溃会留下 `.retired-*`，内容完整、可人工恢复。这满足迁移设计 §12 的「取消保留最新的完整检查点、删除部分写入的临时文件」。

续训发现：扫 `models/unet`，取名字为 `checkpoint-` 加恰好 8 位 ASCII 数字、且确实是目录的项，选步号最大者，然后

```rust
let expected = CheckpointCompatibility::new(descriptor.clone(), config.clone(), frame_count);
let restored = load_training_checkpoint::<TrainBackend, M, O>(&dir, &model, &optimizer, &device, &expected)?;
let runner = TrainingRunner::restore(dataset, restored, device)?;
```

`CheckpointCompatibility::new` 写的是空 provenance，`TrainingRunner::checkpoint_state()` 保存时写的也是空 provenance，两边对齐；等 provenance 真正开始记录资产与模型哈希时，两处要一起改。

三条由既有契约推出的行为，必须在提示文案与交付说明里讲清楚：

- `training_config` 逐字段比较，所以续训的 `--mode`、`--variant`、`--epochs` 必须与首次运行一致，否则报 `MODEL_INCOMPATIBLE`。「续训时放宽 `total_epochs`」在 §18。
- `data_loader.frame_count` 必须一致，所以把工程重新加锁成不同帧数会让旧检查点失效。这是正确行为：样本编号变了，恢复出来的 loader 位置没有意义。
- `TrainingRunner::restore` 把 `samples_seen` 归零、保留 `global_step`，所以续训后的吞吐与 ETA 是本次运行的，不是全程累计的（见 §9、§12）。

`resume = true` 却找不到检查点时拒绝（`MEDIA_INVALID`，摘要「未找到可续训的检查点」），不静默从零开始——那是与用户意图相反的静默偏离。`resume = false` 时允许旧检查点存在并从第 0 步重新开始；新运行走到同一步号时，发布例程的第 3 步会把同名旧目录退休掉，也就是重复运行会覆盖同名步号的检查点。

## 9. 遥测与预览产物

每个 epoch 边界，在发布检查点之后写两份诊断产物：

| 产物 | 路径 | 写入函数 |
| --- | --- | --- |
| 指标 | `<project>/outputs/metrics/step-{global_step:08}.json` | `write_training_metrics` |
| 预览 | `<project>/outputs/preview/step-{global_step:08}/` | `write_preview_artifact` |

指标由 `runner.metrics(&report, started.elapsed(), None, WORKER_STATE)` 生成，`WORKER_STATE = training`（`validate_worker_state` 只接受 1 到 128 个小写字母、数字、下划线与连字符）；`gpu_memory_bytes` 传 `None`，因为 CPU 后端没有显存可报（§4）。`elapsed` 从本次运行开始计时，与被归零的 `samples_seen` 配套，算出的 `samples_per_second` 才是本次运行的真实速率。

预览由 `build_preview_artifact` 生成。它要求一个 `TrainingSample::SingleFrame`，这与训练模式无关——预览样本是 worker 自己选的：`target_index = 0`、`reference_index = frame_count / 2`。参考帧刻意取中间帧而不是第 0 帧：参考帧等于目标帧时预览等于把答案抄给模型，看不出训练效果。

两个写入函数都拒绝已存在的目标。worker 在调用前自己判断存在性：已存在就跳过这一份诊断产物并计数，其他任何错误都是致命的。理由是这两份产物是诊断信息、权重才是产品，不能让上次运行留下的一个目录把一次长训练在第一个 epoch 边界打断；而把「跳过」限制在「目标已存在」这一种情况，就不会顺手吞掉磁盘满之类的真错误。

## 10. 阶段、进度与取消

进度总量在开跑前算一次：

```
sample_count    = frame_count 或 frame_count - 1        （§3）
steps_per_epoch = sample_count.div_ceil(batch_size)      （末批被 batch_size.min(remaining) 截短）
total           = total_epochs.checked_mul(steps_per_epoch)
```

`checked_mul` 溢出时 `total` 取 `None`，进度退化成只报已完成步数。188 帧、baseline、2 个 epoch、批大小 1 的例子：`steps_per_epoch = 188`，`total = 376`；同样参数下时序模式是 187 与 374。

| 时点 | 上报阶段 | completed / total |
| --- | --- | --- |
| 命令开始（打开数据集、加载 VGG19、恢复检查点之前） | `Preparing` | 无 |
| 每步提交之后 | `Training { epoch, step, loss }` | `min(global_step, total) / total` |
| 发布检查点、写指标与预览 | 不上报 | 无 |

`epoch` 字段是 `u32` 而训练侧是 `u64`，用 `u32::try_from(...).unwrap_or(u32::MAX)` 收窄；`step` 取 `report.global_step`，`loss` 取 `report.losses.total`。不做节流：CPU 单步是秒级，每步一帧事件的开销可以忽略；等 GPU 切片把单步压到毫秒级再谈节流。`min(global_step, total)` 是防御性夹取——续训时 `global_step` 从检查点接着走，lineage 一致就不会超过 `total`，夹取只为避免 `total` 与 loader 出现不一致时把一个荒谬的进度写上线。

取消在每步之前检查一次（准入之后、第一步之前还有一次）。命中时先发布一个检查点（此刻 loader 位置刚提交完，状态一致），再返回 `CommandOutcome::Cancelled`。已经进入 `runner.step` 的那一步不可中断，CPU 上是秒级，这是已知的残余窗口。`CommandOutcome::Cancelled` 不带载荷，所以那个检查点只出现在磁盘上，靠续训发现（§8）。

`Metrics` 事件的三个字段仍然留空。`TaskReporter::report(&self, stage, progress)` 是唯一的接缝，要把 `samples_per_second`、`eta_seconds`、`vram_bytes` 送上线就得改这个 trait 的签名，牵动每个既有命令与它们的测试假件。这些数字没有丢，它们在 `outputs/metrics/` 里（§9）；扩接缝的事留给 §18。

## 11. 准入检查

顺序按代价从低到高，失败全部经本文件的 `invalid_request` 加 `error_map::clamp`，照 `extract_features.rs` 的形状：

1. `check_project_dir(&params.project_dir)`：绝对路径、`symlink_metadata` 判定为目录、`project.json` 是常规文件。复用 `admission.rs` 的既有函数。
2. `params.epochs` 必须落在 1..=`MAX_EPOCHS`（10 000）。`TrainingConfig::validate` 也会拒绝 0，先在这里拒绝是为了给出「训练轮数无效」这样的中文摘要，而不是一句配置错误。
3. `ProjectTrainingDataset::open`：内部走 `validate_project_dir`（要求 `assets.json` 已加锁）、拒绝零帧、读 `assets/features/feather_hubert.f32` 并要求 `dims == 1024` 且 `tokens == 2 × frame_count`。失败经 `training_data_task_error`。这一步是「工程必须先抽帧、提特征、加锁」这条前置条件的唯一执行点，worker 不另写一遍。
4. 时序模式要求 `frame_count >= 2`（`temporal_stride = 1`，样本数 `frame_count - 1` 必须大于 0）。`DataLoaderConfig::validate` 也会拒绝，先拒是为了给出「帧数不足，无法做时序训练」这样的摘要。
5. 续训发现（§8）：`resume = true` 且没有检查点则拒绝。放在加载 VGG19 之前，省掉一次几十兆的读取。
6. `load_vgg19_package`：校验 `LICENSES.json` 与 `manifest.json` 的哈希，再严格加载 safetensors。这是准入里最贵的一步，因此排在最后。
7. `token.is_cancelled()`：一切就绪、第一步之前查一次。

## 12. 结果载荷

新增 `worker/src/train_result.rs`，形状照 `feature_result.rs`（`serde_json::json!` 加 `path.display().to_string()`）：

```json
{
  mode: mouth_roi,
  variant: original_unet,
  backend: ndarray-cpu,
  model_kind: original_unet,
  architecture_version: original-unet-burn-v1,
  model_config_sha256: …,
  frame_count: 188,
  epochs_requested: 2,
  epochs_completed: 2,
  global_step: 376,
  samples_seen: 376,
  total_loss: 0.0412,
  resumed_from: null,
  checkpoint_dir: <project>/models/unet/checkpoint-00000376,
  checkpoints_written: 2,
  metrics_written: 2,
  previews_written: 2
}
```

三个超出最小集的字段是刻意加的。`backend` 让「这次训练到底跑在什么后端上」出现在产物里，这是迁移设计 §4.1 那条显示要求在没有 GPU 时的最小落实。`model_config_sha256` 是审计依据，也是下一次续训必须匹配的值。`resumed_from` 给出这次运行从哪个检查点接上，未续训为 `null`。`samples_seen` 是本次运行的口径（§8 第三条）。不放损失曲线与逐步指标，理由同抽帧设计 §10：几百到几万个数字会把单行 JSON 事件撑爆，要曲线就读 `outputs/metrics/`。

## 13. 错误映射

`error_map.rs` 新增两个映射器并从 `lib.rs` 导出：

```rust
pub fn training_task_error(error: &TrainingError, stage: TaskStage) -> TaskError;
pub fn training_data_task_error(error: &TrainingDataError) -> TaskError;
```

`training_task_error` 收一个 `stage` 参数，这是本文件里第一次不用固定的 `FAILURE_STAGE`。理由很直接：一次跑了几十分钟、在第 3000 步失败的训练，若把阶段报成 `Preparing`，事件流就在撒谎。命令体在装配期传 `TaskStage::Preparing`，进入循环后传 `Training { epoch, step, loss }`，取最后一次成功上报的三个值。`training_data_task_error` 只在打开数据集时使用，固定 `Preparing`。

`TrainingError` 到错误码：

| 变体 | 错误码 | 摘要 |
| --- | --- | --- |
| `Io` | `io_error_code`（沿用） | `io_summary`（沿用） |
| `InvalidInput` | `MEDIA_INVALID` | 训练输入无效 |
| `InvalidConfig`、`InvalidDataLoaderConfig` | `WORKER_CRASHED` | 训练配置无效 |
| `InvalidDataLoaderState`、`StalePreparedBatch`、`DataLoaderOverflow`、`PermutationAllocation`、`BatchAllocation` | `WORKER_CRASHED` | 训练运行状态异常 |
| `InvalidPackage`、`HashMismatch` | `MODEL_INCOMPATIBLE` | 感知损失模型加载失败 |
| `Store` | `WORKER_CRASHED` | 检查点读写失败 |
| `InvalidCheckpoint`、`CheckpointCompatibility` | `MODEL_INCOMPATIBLE` | 检查点与当前训练不兼容 |
| `CheckpointDirectory` | `MEDIA_INVALID` | 检查点目录无效 |

三行需要解释。`InvalidConfig` 与 `InvalidDataLoaderConfig` 的输入全是 worker 自己的常量（§5），用户改不了，所以它们只可能是 worker 的 bug，走 `WORKER_CRASHED`。`InvalidInput` 是个宽变体：数据集样本损坏与「损失不是有限值」都落在这里，细节文本会带上具体字段名，摘要统一用「训练输入无效」——损失发散在这条链路上确实是数据或超参的问题。`Store` 同时覆盖 burn 记录的写与读，消息前缀能区分但按文本匹配太脆；真正的兼容性问题在 `load_training_checkpoint` 的预检阶段就被 `InvalidCheckpoint` 与 `CheckpointCompatibility` 拦住了，走到 `Store` 的只剩存储层故障。

`TrainingDataError` 到错误码，摘要按变体给中文：

| 变体 | 错误码 |
| --- | --- |
| `Project`、`Features`、`Frame`、`Landmarks`、`Sample` | `MEDIA_INVALID` |
| `FeatureShape` | `FEATURE_SHAPE_MISMATCH` |
| `FrameIndexOutOfRange`、`Batch` | `WORKER_CRASHED` |

`FeatureShape` 与领域错误码的名字正好对齐，它表达的正是「特征令牌数与工程帧数不匹配」，值得一个专门的码而不是笼统的 `MEDIA_INVALID`。`FrameIndexOutOfRange` 与 `Batch` 是索引与堆叠的内部不变量，用户造不出来，走 `WORKER_CRASHED`。

## 14. 命令签名与 CLI 形态

worker 侧分两层，分层的目的是让循环可以离线测试：

```rust
pub fn execute_train(
    params: &TrainParams,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    toolchain: &TrainingToolchain,
) -> CommandOutcome;

pub(crate) fn run_training<M, O, D, E>(
    plan: &TrainingPlan, // 工程根、descriptor、config、frame_count、续训目标
    dataset: D,
    model: M,
    optimizer: O,
    extractor: &E,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
) -> CommandOutcome;
```

`execute_train` 负责准入、VGG19 加载与变体分发；`run_training` 负责循环、检查点发布、遥测、进度与取消。单元测试直接驱动 `run_training`，用 `parity_micro` 通道的模型、常量感知提取器和桩数据集，不碰环境变量、真实权重或磁盘上的真实工程。`commands.rs` 的分支照 `extract_features` 的判例：先取 `config.training()`，缺失则 `Failed(unsupported(request.kind()))`。

CLI：

```
feathertalk train <PROJECT_DIR> --mode <MODE> --variant <VARIANT> --epochs <N> [--resume]
```

`--mode` 取 `baseline|mouth-roi|temporal`（缺省 `baseline`），`--variant` 取 `original-unet|mobileone-unet`（缺省 `original-unet`），`--epochs` 必填。两个枚举在 CLI 侧用本地的 `#[derive(ValueEnum)]` 镜像枚举，再在 `run.rs` 映射到领域枚举：为了参数解析给 `feathertalk-domain` 加 clap 依赖不划算，而镜像枚举顺带让 `--help` 与非法取值的报错由 clap 生成。clap 的连字符取值与线协议的下划线 slug 不同形，转换点只有 `run.rs` 一处。

`render.rs` 新增 `const ENV_WORKER_VGG19_DIR: &str = FEATHERTALK_WORKER_VGG19_DIR;`，沿用该文件「把 worker 常量复制一份并注明 worker 是唯一来源」的约定；`render_client_error` 的 `UnsupportedCommand` 分支加 `else if *requested == train`。阶段中文标签与 `capabilities.training` 的打印已经存在，不改。`cli.rs` 顶部列举 kebab 命令的文档注释同步更新。

## 15. 栈与线程

`runtime.rs` 的执行线程现在是裸 `thread::spawn`，拿的是 Rust 默认 2 MiB 栈。160×160 的前向加反向在 debug 构建下会把它撑爆（`0xc00000fd`；判例是 `feathertalk-pfld/tests/runtime.rs` 与 `feathertalk-training-run/tests/support/mod.rs` 的 64 MiB 线程）。因此把那次 spawn 换成 `thread::Builder::new().name(execution).stack_size(EXECUTION_STACK_BYTES).spawn(...)`，`EXECUTION_STACK_BYTES = 64 * 1024 * 1024`，spawn 失败按现有的启动错误通道上报。

否决方案：在 `train.rs` 里为训练循环单独起一个大栈线程。那需要把 `&dyn TaskReporter` 跨线程借出去，也就是给 `TaskReporter` 加 `Sync` 约束并改所有实现与测试假件；而把栈提到执行线程上是一行改动，顺带让既有的推理命令也不再贴着 2 MiB 跑。

## 16. 测试

`feathertalk-worker`（新增 `tests/train.rs`、`tests/train_result.rs`）：

- 成功路径：桩数据集（8 帧，样本由 `FrameSample::new` 合成）、`parity_micro` 模型、常量提取器、`epochs = 2`，断言检查点目录名、指标文件数、预览目录数与结果载荷各字段；
- 续训：先跑 1 个 epoch，再用 `resume = true` 跑同样的配置，断言 `global_step` 从检查点接上、`resumed_from` 指向那个目录；
- 描述符不匹配：改掉 `model_config_sha256` 后续训，断言 `MODEL_INCOMPATIBLE`；
- 名字撞车：预置一个同名 `checkpoint-*` 目录，断言发布例程把它退休并写入新内容；
- 诊断产物撞车：预置同名指标文件与预览目录，断言训练照样完成且对应计数为 0；
- 进度：假 reporter 记录 `preparing` 到 `training 1/N … N/N`，并断言 epoch 字段跨边界递增；
- 取消：token 在第二步之前置位，断言 `Cancelled` 且磁盘上留下一个能被 `load_training_checkpoint` 读回的检查点；
- 准入七项各一个失败用例；
- 扩展 `tests/config.rs`（新环境变量与新构造函数）、`tests/handshake.rs`（只配 VGG19、只配媒体、全配三种组合，以及 `capabilities.training`）、`tests/error_mapping.rs`（`TrainingError` 与 `TrainingDataError` 全变体，含 stage 参数）、`tests/runtime.rs`（拒绝文案）、`tests/commands.rs`（未配置工具链时返回 `Failed`）。

所有单元测试跑在 `Autodiff<NdArray<f32>>` 上、由 64 MiB 栈线程包住，不用环境变量、不碰 GPU、不加载 VGG19。

`feathertalk-cli`：`tests/cli.rs` 与 `run.rs` 的内联测试覆盖空参数拒绝、非法枚举取值与请求构造。

端到端（门控，`--release`）：加在 `tests/real_worker.rs`，沿用 `REQUIRE_E2E` 加 `worker_or_skip` 加 `real_dir(...)` 的组合，新增 `real_dir(VGG19_DIR)`。先用既有的归一化 → 抽帧 → 提特征 → 加锁链路造一个短片段工程（1 秒、25 帧），再跑 `train --mode baseline --epochs 1`，断言检查点、指标与预览产物齐备且检查点能被读回。必须跑 release：debug 下的 burn 慢约三个数量级。`production()` 通道是硬编码的，E2E 的耗时只能靠帧数压——25 帧 1 个 epoch 在 CPU 上是分钟级，可接受。

不新增二进制夹具入库，理由同抽帧设计 §14。

## 17. 验证

在 `rust/` 下执行，要求零告警零失败：

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace --all-targets`
- `cargo test --release -p feathertalk-cli --test real_worker`（门控变量齐备时）
- `git diff --check`

## 18. 范围外

- GPU 训练。`Autodiff<Wgpu>`、adapter 枚举、显存上报、`GPU_OUT_OF_MEMORY` 与 `GPU_DEVICE_LOST` 的映射、OOM 之后的批大小策略（迁移设计 §8.4）是一条独立切片，理由见 §4。
- 线协议扩展：批大小、学习率、种子、保存频率、损失权重、嘴部 ROI 与时序步长（迁移设计 §8.3 的高级面板）都要给 `TrainParams` 加字段；`Metrics` 的 `samples_per_second`、`eta_seconds`、`vram_bytes` 要扩 `TaskReporter` 的接缝。
- `models/unet/last/`（迁移设计 §6.1、§8.4）。`save_training_checkpoint` 拒绝已存在目标，Windows 上又没有原子的目录替换，正确的 `last` 需要一套发布交换协议加一次额外的整份写入或文件级复制；步号命名加「取最大值」已经让续训拿到它需要的一切。
- 多 lineage 的检查点管理：按运行分目录、保留最近 N 个、清理旧检查点。
- 续训时放宽 `total_epochs`。`CheckpointCompatibility` 逐字段比较 `training_config`，放宽要动那个契约与它的测试。
- 把预览产物转成可看的图片。`outputs/preview/` 现在是 f32 裸张量加清单。
- 训练完成后的模型导出、ONNX 校验、渲染与推理。
- 桌面端与 GPUI 的训练界面、进度图与「停止并保存」按钮。
- provenance 记录（资产与模型哈希）。现在保存与校验两侧都写空 map，见 §8。
