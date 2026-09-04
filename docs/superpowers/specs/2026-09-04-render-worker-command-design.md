# 渲染 worker 命令与 CLI 子命令设计

日期：2026-09-04
状态：已定稿

## 1. 目标与范围

里程碑四第四切片交付了 `feathertalk-inference::execute_offline_render`：给它一个模型、一个 `OfflineRenderRequest`、一个帧读取器和一个 sink 工厂，它就按 `RenderPlan` 逐帧推理、把 BGR24 写进 FFmpeg stdin，并在全部成功后用同目录 rename 原子发布视频。上一切片交付了 `train`，工程目录里现在会出现 `models/unet/checkpoint-XXXXXXXX/`。两者之间还没有连线。本切片把它们接到线协议上：worker 实现 `Request::Render`，CLI 增加 `render` 子命令。

领域契约早已提交，本切片一个字节都不改：`TaskKind::Render`（slug `render`）、`RenderParams { project_dir, checkpoint, audio, output, max_output_frames }`（带 `deny_unknown_fields`）、`Request::Render`、`TaskStage::Rendering { frame: u64, total: u64 }`，CLI `render.rs` 的中文标签「正在渲染 第 {frame}/{total} 帧」也已就位。`commands.rs` 目前把 `Request::Render` 落到 `other => Failed(unsupported(...))` 分支。

本切片范围内的改动：

- `feathertalk-inference`：`InferenceError` 新增 `Cancelled { operation }` 变体，字段是静态字符串（见 §7）。
- `feathertalk-training`：新增两个公开入口 `read_training_checkpoint` 与 `load_training_checkpoint_model`，以及共用的 `TrainingCheckpointMetadata` 与 `RestoredCheckpointModel<M>`（见 §4）。
- `feathertalk-worker`：新增 `rendering.rs`（后端别名、素材路径、变体分发、进度总数）、`render.rs`（命令编排与观测 sink）、`render_result.rs`（结果载荷）；`handshake.rs` 宣告 `Render`；`error_map.rs` 新增 `render_task_error` 与 `is_inference_cancellation`；`Cargo.toml` 新增 `feathertalk-inference` 依赖。
- `feathertalk-cli`：新增 `render` 子命令、对应的 `build_request` 分支与 `UnsupportedCommand` 提示分支。

一个直接结论：本命令不需要任何新环境变量。唯一的外部工具是 ffmpeg，`FEATHERTALK_WORKER_FFMPEG` 与 `FEATHERTALK_WORKER_FFPROBE` 已经由 `MediaToolchain` 承载；VGG19、HuBERT、SCRFD、PFLD 都用不上，因为推理不算感知损失，帧、关键点与音频特征都已在加锁后的工程目录里。

范围外的内容集中在 §14。

## 2. 特征来自工程，音频只是音轨

`RenderParams` 只有五个字段，没有特征文件路径。这不是遗漏：协议设计（2026-08-28 §5）把 `Render` 定为 `inference::execute_offline_render` 的唯一入口，字段表当时就已封闭。因此本切片只有一种自洽读法：

**特征取自工程自己锁定的 `assets/features/feather_hubert.f32`；`audio` 只是混进输出视频的那条音轨。**

这条读法与已交付的执行器完全对齐：`OfflineRenderRequest` 同时收 `feature_path` 和 `audio_path`，执行器不检查两者是否同源，音频只经 `-i AUDIO ... -shortest` 交给 FFmpeg。加锁工程还额外送了一份保证：锁要求 `tokens == 2 * frame_count`，于是 `RenderPlan` 的 `feature_frame_count`（即 `tokens / 2`）必然等于素材帧数，不会出现计划长度与素材长度对不上的情况。

代价要写明：本切片渲染出的口型对应的是工程自己的音频。用任意驱动音频生成视频，需要先把驱动音频标准化再抽特征，而 `extract-features` 只往加锁工程的固定路径写、且拒绝覆盖，没有临时特征位置可用。那是独立的一个切片（§14），不是本切片悄悄扩大 `RenderParams` 能解决的事。

## 3. 一次渲染的装配

链路全部由已提交的 API 组成，worker 只负责装配与编排：

```text
RenderParams
  -> check_project_dir               准入：绝对路径、真目录、带 project.json
  -> validate_project_dir            加锁工程：frame_count 与帧宽高
  -> read_training_checkpoint        检查点元数据：descriptor 与 state，不碰 record
  -> 变体分发                        model_kind 选出 OriginalUnet 或 MobileOneUnet
  -> load_training_checkpoint_model  只读模型 record，要求 descriptor 相等
  -> AutodiffModule::valid()         去掉自动微分外壳，得到推理图
  -> OfflineRenderRequest::new       素材目录、特征、音频、ffmpeg、输出、task id
  -> execute_offline_render          逐帧推理、写 FFmpeg、原子发布
  -> render_to_json                  结果载荷
```

素材路径由 worker 决定，与 `extract-frames`、`extract-features` 写出的布局一一对应，全部相对工程根：

```text
frame_dir     = assets/frames
landmark_dir  = assets/landmarks
feature_path  = assets/features/feather_hubert.f32
```

这三个字面量在 `rendering.rs` 里各自一个 `const`。`feathertalk-training-data` 的 `FEATURE_FILE` 是 crate 私有的，`extract_features.rs` 已经为同一个理由复制过 `assets`、`features`、`feather_hubert.f32` 三个名字；再复制一次比把三个 crate 的私有常量互相导出更省。帧与关键点的文件名由执行器按 `{index:06}.jpg` 与 `{index:06}.lms` 拼出，worker 不参与。

## 4. 从检查点取回模型

已有的 `load_training_checkpoint` 用不了：它要一个完整的 `CheckpointCompatibility`（descriptor 加 `TrainingConfig` 加 `frame_count` 加两组 provenance）逐字段比对，还要一个优化器模板。渲染既不知道训练时的 mode、epochs、lr、batch，也不需要优化器状态；为了拿一份权重去重建整套训练配置，是把巧合当契约。

因此在 `feathertalk-training` 里新增两个公开入口，与 `save_training_checkpoint`、`load_training_checkpoint` 并列：

```rust
pub struct TrainingCheckpointMetadata {
    pub manifest: TrainingCheckpointManifest,
    pub state: TrainingCheckpointState,
}

pub struct RestoredCheckpointModel<M> {
    pub model: M,
    pub metadata: TrainingCheckpointMetadata,
}

pub fn read_training_checkpoint(
    directory: impl AsRef<Path>,
) -> Result<TrainingCheckpointMetadata, TrainingError>;

pub fn load_training_checkpoint_model<B, M>(
    directory: impl AsRef<Path>,
    model_template: &M,
    device: &B::Device,
    expected: &CheckpointDescriptor,
) -> Result<RestoredCheckpointModel<M>, TrainingError>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Clone;
```

两者都走 `load_training_checkpoint` 已有的那套前置检查，顺序不变：拒绝路径上的符号链接、校验检查点目录、读并校验 `manifest.json` 与 `state.json`。`load_training_checkpoint_model` 随后只校验 manifest 声明的模型文件，要求 `manifest.descriptor()` 与 `expected` 相等，再把模型 record 读进 `model_template.clone()`。模板永远只被克隆，失败时调用方手里的模板不变，与既有实现同一条规则。优化器与训练状态的 record 不读，因为推理用不到。

需要两个入口而不是一个，是因为存在先后依赖：模板必须先知道变体，而变体只写在 manifest 里。先读元数据、再按变体建模板、再带着期望 descriptor 去装载，是这条依赖唯一的拆法。两次前置检查的重复成本是两次 `symlink_metadata` 加两个小 JSON，可以忽略。

`AutodiffBackend` 这个界不是随手写的：record 是 `Autodiff<NdArray>` 上的模块写出来的，用同一组类型读回来才是肯定兼容，而不是大概兼容。装载后立刻调用 `AutodiffModule::valid()` 拿到内层后端上的推理图，自动微分外壳随即释放，逐帧前向不会记录任何计算图。

## 5. 后端、变体分发与描述符校验

推理只跑 CPU，与训练切片同一个理由：握手里的 `wgpu_training` 仍是 false，worker 不承诺 GPU，也就没有静默回退需要解释。

```rust
pub type RenderBackend = CpuBackend;
pub type RenderDevice = Device<RenderBackend>;
pub const RENDER_BACKEND_NAME: &str = "ndarray-cpu";
```

装载用的是训练那套类型（`TrainBackend` 即 `Autodiff<NdArray>`），`valid()` 之后落到 `RenderBackend`。

变体不由请求给出，而是从检查点读出来：`manifest.descriptor().model_kind`。worker 把两个候选配置都建出来，用 `ModelConfiguration::model_type()` 去比对，命中的那个就是变体：

```text
ModelConfiguration::original_unet(&OriginalUnetConfig::production())    -> "original_unet"
ModelConfiguration::mobileone_unet(&MobileOneUnetConfig::production(), false) -> "mobileone_unet"
```

这样 `render.rs` 里不出现任何模型类型字面量，`model_kind` 的拼写只有 `feathertalk-export` 一处来源。命中之后，worker 用既有的 `checkpoint_descriptor(&configuration)` 算出期望 descriptor 交给装载器，于是 `model_kind`、`architecture_version` 和 `model_config_sha256` 三项一起被校验：配置漂移过的检查点会被 `ModelIncompatible` 挡住，而不是渲染出一段无声无息错掉的视频。都不命中时同样以 `ModelIncompatible` 拒绝，并在 detail 里带上读到的 `model_kind`。

MobileOne 多一步：`valid()` 得到的是多分支训练图 `MobileOneUnet`，它不实现 `TalkingHeadModel`；必须调用 `reparameterize()` 融合分支后，才拿到实现了推理边界的 `MobileOneUnetInference`。这一步由类型系统强制，`feathertalk-models` 里有一条 `compile_fail` doctest 守着它。

## 6. 进度与取消：一个观测 sink

`execute_offline_render` 没有观察者回调，逐帧循环整个在执行器内部。但它有一个更好的接缝：每渲染完一帧，恰好调用 `sink.write_frame` 一次，顺序与 `output_index` 一致。于是 worker 不改执行器签名，而是包一层 sink 工厂：

```rust
struct ObservedSinkFactory<'a, F> {
    inner: &'a F,
    reporter: &'a dyn TaskReporter,
    token: &'a CancellationToken,
    total: u64,
}
```

它把 `start` 转给内层工厂，把返回的 sink 包成一个计数 sink。每次 `write_frame`：先问取消，再写内层，再把帧号加一并上报

```text
TaskStage::Rendering { frame, total }
Progress { completed: frame, total: Some(total) }
```

`Rendering` 的 `total` 是 `u64` 而非 `Option`，所以总数必须是个确定值。它取自加锁清单：`min(frame_count, max_output_frames)`。这与执行器自己算的 `plan.output_frame_count()` 在加锁工程上必然一致（§2 的 token 保证），因此不必为了一个进度分母把可能上百 MB 的特征文件多读一遍。万一某个工程的特征比清单短，执行器的计划会更短，进度条只是提前停在不足百分之百；`completed` 也照 `train.rs` 的做法夹紧到 `total`，绝不越过分母。真正权威的帧数由结果载荷里 `OfflineRenderResult::frame_count()` 给出。

取消就落在同一个接缝上：token 已取消时 `write_frame` 直接返回 `InferenceError::Cancelled`，执行器把错误向上抛，途中照它原本的失败路径终止并回收 FFmpeg 子进程、清理 staging 文件，目标文件不会出现。worker 把这个错误认回 `CommandOutcome::Cancelled`。一帧是一次前向加一次写入，中间没有更细的接缝，与 `extract_features` 在 chunk 之间取消是同一种粒度。

## 7. 错误模型

`InferenceError` 现在没有取消变体，新增一个：

```rust
Cancelled { operation: &'static str }
```

这与 `AudioError::Cancelled`、`MediaError::ToolCancelled`、`PipelineError::Cancelled` 是同一套做法，worker 侧也照抄同一套判定：`is_inference_cancellation` 与既有的 `is_audio_cancellation`、`is_media_cancellation` 并列。用一个带哨兵文本的 `SinkWrite` 冒充取消，是把字符串当协议，不做。

`error_map.rs` 新增 `render_task_error(error: &InferenceError, stage: TaskStage) -> TaskError`，穷举匹配、不留 `_` 分支，映射如下：

```text
MediaInvalid           请求字段、输入目录与产物、输出已存在或非普通文件、
                       符号链接、非法 task id、ffmpeg 路径非绝对、帧数不足、
                       特征为空、几何与索引越界、算术溢出
FeatureShapeMismatch   InvalidFeatureShape
LandmarkInvalid        关键点文件读取与 bbox 计算失败
ModelIncompatible      模型输入输出非有限、越界、张量形状不符、预测非有限
WorkerCrashed          sink 启动 / 写入 / 结束、FFmpeg 非零退出、staging 冲突、
                       staging 产物无效、原子发布失败、帧解码失败、内存分配失败
TaskCancelled          Cancelled
```

`stage` 由调用方给：第一帧写出之前的一切失败报 `Preparing`，进入逐帧循环之后报当时的 `Rendering`，与 `training_task_error(error, stage)` 的形状一致。detail 一律经既有的 `clamp` 截断，中文摘要只说人能懂的那一句，路径与索引留在 detail 里。

## 8. staging task id 与原子发布

发布规则全部在执行器里，worker 一个字节都不重写：目标必须不存在，staging 文件与目标同目录同扩展名，全部帧写完、FFmpeg 正常退出、staging 校验通过之后才 rename。

worker 只需要给出 staging 用的 task id。协议里的 task id 在运行时的事件信封上，而命令函数拿不到它：`TaskReporter` 只有 `report`，分发器的函数类型也不带 task id，为一个文件名去改这两处不值得。因此 worker 自己造一个：

```text
render-{进程号}-{进程内自增计数}
```

字符集落在 `validate_task_id` 允许的字母数字与连字符里，与 `feathertalk-training` 的 `.checkpoint-{pid}-{id}.staging` 是同一套命名习惯。唯一性由执行器的 `create_new` 兜底：真撞上了就报 `StagingCollision` 失败，绝不覆盖别人的文件。

## 9. 准入检查

按最便宜的拒绝排在最前的顺序：

1. `check_project_dir(&params.project_dir)`：绝对路径、真目录、带 `project.json`，与 `train` 共用同一份中文文案。
2. `checkpoint`、`audio`、`output` 三条路径都必须是绝对路径，各自一句中文摘要。
3. `max_output_frames`：`Some(0)` 拒绝（渲染零帧不是预览，是请求写错了）；`usize::try_from` 失败也拒绝，绝不截断，这是协议设计 §5 的明文要求。
4. `validate_project_dir(&params.project_dir)`：拿加锁清单里的 `frame_count` 与帧宽高。这一步也是「工程还没加锁」的唯一判定处，错误经既有的 `project_task_error` 映射。
5. `frame_count` 必须至少为 2 并且能装进 `usize`：`PingPongFrames::new` 拒绝少于两帧，与其让执行器在读完检查点之后才报，不如在装载半个模型之前就说清楚。

其余的一切交给执行器，因为它本来就做得比 worker 细：素材目录与关键点文件是否存在、是否普通文件、特征维度与 token 数、输出父目录、逐帧尺寸一致性。worker 不重复这些检查，重复就意味着两处判定可以不一致。

装载检查点之后、进入循环之前再问一次取消：读几十 MB 的 record 要花时间，调用方可能已经放弃。这与 `train.rs` 在装载 VGG19 之后补问一次是同一个理由。

## 10. 结果载荷

```json
{
  "output_path": "...",
  "frame_count": 4,
  "width": 1280,
  "height": 720,
  "fps": 25,
  "backend": "ndarray-cpu",
  "checkpoint_dir": "...",
  "model_kind": "original_unet",
  "architecture_version": "...",
  "model_config_sha256": "...",
  "checkpoint_epoch": 1,
  "checkpoint_global_step": 4,
  "source_frame_count": 4,
  "max_output_frames": null
}
```

`frame_count`、`width`、`height` 与 `output_path` 直接来自 `OfflineRenderResult`，是执行器实际写出的事实。`fps` 是常数 25，写进载荷是因为它是产物的属性而不是代码的属性。三项模型身份来自检查点 descriptor，和训练载荷同名同义，两条命令的产物因此可以直接对照。`checkpoint_epoch` 与 `checkpoint_global_step` 来自 `state`，回答的是「这段视频是哪一步的权重渲染的」。`max_output_frames` 原样回显请求，`null` 表示整段渲染，`Some(n)` 表示这是一段预览，这样从产物就能看出它是预览还是成片。

载荷里不放耗时与速率：`metrics` 信封已经有 `samples_per_second` 与 `eta_seconds` 的位置，把它们再抄进结果 JSON 只会多出一处可以对不上的数字。

## 11. 命令签名与 CLI 形态

worker 侧分成两层，与训练切片同一个形状：`run_render` 是可测的核心，`execute_render` 是薄薄的编排。模型、帧读取器与 sink 工厂都是参数，于是单元测试可以直接塞一个 `OriginalUnetConfig::parity_micro()` 初始化的小模型和一个内存 sink，既不依赖 FFmpeg，也不必先造一份 production 尺寸的检查点。

```rust
pub fn run_render<M, R, F>(
    job: &RenderJob,
    model: &M,
    device: &RenderDevice,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    frame_reader: &R,
    sink_factory: &F,
) -> CommandOutcome
where
    M: TalkingHeadModel<RenderBackend>,
    R: FrameReader + ?Sized,
    F: RawVideoSinkFactory + ?Sized;
```

`RenderJob` 住在 `rendering.rs`，装的是准入之后就已经定下来的东西：`OfflineRenderRequest`、进度总数、检查点 descriptor、检查点的 epoch 与 global step、素材帧数、`max_output_frames` 与检查点目录。名字刻意不叫 `RenderPlan`，那个名字属于 `feathertalk-inference` 的逐帧计划。

外层负责从请求走到模型：准入、读检查点元数据、分发变体、装载权重、组装 `RenderJob`，再调用 `run_render`。

```rust
pub fn execute_render<R, F>(
    params: &RenderParams,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    toolchain: &MediaToolchain,
    frame_reader: &R,
    sink_factory: &F,
) -> CommandOutcome
where
    R: FrameReader + ?Sized,
    F: RawVideoSinkFactory + ?Sized;
```

`commands.rs` 的新分支在 `other =>` 之前，条件是 `config.media()`，传入 `JpegFrameReader::default()` 与 `SystemRawVideoSinkFactory`。与 `execute_train` 一样，toolchain 以引用传入而不是在这里读环境变量：`commands.rs` 手里已经有校验过的配置。

CLI 子命令：

```text
feathertalk render <project_dir> <checkpoint> <audio> <output> [--max-output-frames <N>]
```

四个位置参数按「从哪来、用什么、配什么音、写到哪」排列。`--max-output-frames` 不给就是整段渲染；帮助文本点明它就是预览的表达方式，不另设 `--preview`，因为协议里没有 `Preview` 命令，多一个开关等于给将来的管线分叉留门。`build_request` 只做与其他命令相同的空路径拒绝，其余判断留给 worker，避免两处判定各说一套。

## 12. 握手与不支持提示

`supported_commands` 在 `config.media().is_some()` 时追加 `TaskKind::Render`，与 `ProbeMedia`、`NormalizeMedia` 同一个条件块。理由写在代码注释里：渲染只需要 ffmpeg，帧、关键点与特征都已经在加锁工程里，所以它既不依赖 SCRFD 与 PFLD，也不依赖 HuBERT 与 VGG19。`Capabilities` 不加字段，`ffmpeg` 那一项已经是同一个事实。

CLI 的 `UnsupportedCommand` 链新增一条 `render` 分支，文案与媒体命令那条同源：请用 `FEATHERTALK_WORKER_FFPROBE` 与 `FEATHERTALK_WORKER_FFMPEG` 指定它们的完整路径。`MediaToolchain::new` 要求两者都在，因此提示两个变量而不是只提 ffmpeg。

## 13. 测试

新增与改动的测试，按 crate 分：

- `feathertalk-inference`：`InferenceError::Cancelled` 的 `Display` 与判定；执行器在 sink 于第 N 帧返回取消时，目标文件不存在、staging 被清理。
- `feathertalk-training`：`read_training_checkpoint` 读回 descriptor 与 state；`load_training_checkpoint_model` 在 descriptor 相等时装回权重、在 `model_kind` 或 `model_config_sha256` 不符时报 `CheckpointCompatibility`；缺少 `model` 文件、目录是符号链接、`manifest.json` 损坏时逐条拒绝；模板在失败后不变。
- `feathertalk-worker` `tests/rendering.rs`：三条素材路径的拼装、`min(frame_count, max)` 的总数、`Some(0)` 与超出 `usize` 的拒绝、非绝对路径的四条中文摘要、变体分发命中与未命中、staging task id 的字符集。
- `feathertalk-worker` `tests/render_result.rs`：载荷十四个字段齐全，`max_output_frames` 为 `None` 时是 JSON `null`。
- `feathertalk-worker` `tests/render.rs`：用 `OriginalUnetConfig::parity_micro()` 的小模型、内存 sink 与桩帧读取器驱动 `run_render`，断言每帧一次 `Rendering { frame, total }` 事件、每帧一次完整的 BGR24 写入、取消在第二帧生效且返回 `Cancelled`、输出已存在时报 `MediaInvalid`、失败后目标文件不存在。`execute_render` 另测准入与变体分发的拒绝路径：非绝对路径、`Some(0)`、未加锁工程、`model_kind` 不认识时报 `ModelIncompatible`。装载真权重的完整路径由 `feathertalk-training` 的装载器测试与下面那条端到端用例覆盖。
- `feathertalk-cli` `tests/cli.rs`：`build_request` 携带四条路径与 `max_output_frames`，空路径逐条拒绝。
- `feathertalk-cli` `tests/real_worker.rs`：新增 gated 端到端用例 `a_real_project_is_rendered_end_to_end`。它锁一个两帧工程、训练一轮拿到检查点、再渲染两帧，断言 mp4 存在且非空、载荷帧数与宽高正确、stderr 出现「正在渲染」。两帧而不是四帧，是因为这条用例的时间几乎全花在训练那一步。

两帧的算术要记在代码注释里：两帧需要四个 token，`(样本数 - 80) / 320 = 4` 于是样本数落在 1360 到 1679 之间，0.09 秒的 16 kHz 音频是 1440 个样本，正在窗口中间。

## 14. 范围外

- 任意驱动音频：标准化加特征提取加渲染的串联，需要一个不在加锁工程里的临时特征位置，那是独立切片。
- GPU 推理：`RenderBackend` 换成 WGPU 加一处分发即可，但握手承诺、显存报告与认证 adapter 都要一起改。
- 输出尺寸、编码器与质量选项：`RenderParams` 没有这些字段，`raw_video_command` 也把参数表固定成了 libx264 加 aac。桌面工作台 §10.3 要的那几个下拉框需要先扩协议。
- 渲染中途的 metrics 信封（速率与预计剩余）：本切片只报阶段与进度。
- 模型包与 ONNX 导出路径的渲染：本命令只读训练检查点。
