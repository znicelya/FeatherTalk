# FeatherTalk worker 协议与任务域契约设计

日期：2026-08-28  
状态：已确认，按推荐方案实施

## 1. 目标与边界

本切片是里程碑五（GPUI 工作台）九个切片中的第一个，为桌面进程、worker 进程和 CLI 建立
唯一的共享词汇：任务标识、命令词表、阶段事件、进度、指标、错误码，以及 JSON Lines 帧
格式、握手和取消语义。里程碑五其余八个切片全部单向依赖本切片的类型，不再各自定义任务
概念。

本切片包含：

- `feathertalk-domain` crate，提供版本化的 `TaskId`、`TaskKind`、`TaskStage`、`TaskStatus`、
  `TaskLifecycle`、`Request`、`Event`、`Progress`、`Metrics`、`ErrorCode`、`TaskError`；
- `ClientFrame` / `ServerFrame` 两个方向的封闭帧枚举，含 `Ready` 握手与能力报告；
- 基于 `std::io` 的 `FrameReader` / `FrameWriter`，含帧长上限；
- `TaskStage -> TaskStatus` 全函数投影与 `TaskLifecycle` 转移校验；
- 与 `feathertalk-project` 既有持久化格式的守卫测试。

本切片不包含 worker 进程本体、真实 stdin/stdout 句柄的接线与子进程启动、命令派发、任务队列、
GPU adapter 互斥、既有 crate 错误类型到十个错误码的映射，以及任何 GPUI 代码；这些属于切片 2
至 5。本切片提供的 `FrameReader` / `FrameWriter` 是泛型 `std::io` 适配器，由切片 2 递入真实句柄。

本切片对主设计文档提出两处修正：

- §11 的阶段列表新增 `Importing` 变体，理由见第 4 节；
- §4.3 的 crate 命名沿用仓库既有的 `feathertalk-*` 前缀，而非文档中的 `app`、`worker`、
  `domain`、`cli` 裸名。

## 2. 方案选择

### 方案 A（推荐）：单一版本化信封 + 封闭命令枚举和封闭事件枚举

一个 `Request` 枚举每命令一个变体，一个 `TaskStage` 枚举对应 §11 的阶段，外层信封携带
`protocol_version`、`task_id` 和时间戳，serde 内部标签配 `deny_unknown_fields`，协议版本精确
相等。

三条理由。其一，它是 §4.2 的字面实现，不增不减。其二，封闭枚举把遗漏推到编译期：worker、
CLI 和桌面端任何一处漏掉某条命令都编译不过，这在切片 2 与切片 3 并行开工时价值最大。
其三，与仓库既有风格一致——`feathertalk-project` 的清单类型全部是封闭 serde 枚举加
`deny_unknown_fields`。桌面端与 worker 由同一安装包一起分发（§13），不存在需要前向兼容的
版本错配场景。

### 方案 B：每命令独立请求/响应类型 + trait 注册表

每条命令各自一对类型，worker 侧用 trait 对象分发。单命令类型安全更强，新增命令不必改动
中心枚举。代价是机械代码显著增多，且失去跨三处的穷尽性检查——漏实现一条命令从编译错误
退化为运行时错误。命令集基本已知且封闭，这层机器不划算。

### 方案 C：瘦信封 + 不透明载荷

信封只认 `protocol_version`、`task_id` 和 `kind: String`，载荷由各命令模块自行解析。传输层可
独立演化。代价是穷尽性检查全部丢失，且 `kind` 拼写错误要到运行时才暴露——这正是
`feathertalk-project::TaskHistoryEntry.kind` 已有的隐患，不应扩大。

## 3. crate 边界与依赖

`rust/crates/feathertalk-domain`，生产依赖仅 `serde`、`serde_json`、`thiserror`、`time`，四者均已
在 workspace 依赖表中，不引入新第三方。不依赖 `burn`，不依赖任何 media、preprocess、audio、
models、training、inference、export crate。

这条约束是 §4.3 中"`app` 不依赖 `models`、`training` 或模型计算使用的 WGPU crate"能够成立的
物理基础：`app` 只依赖 `domain`，编译期就无法把模型代码拖入 UI 进程。

依赖方向为 `worker`、`cli`、`app`、supervisor 四个上层 crate 单向依赖 `domain`。

`feathertalk-project` 保持一行不改。`TaskStatus` 在 `domain` 内独立定义五态，
`feathertalk-project::TaskHistoryStatus` 原样保留，两份定义靠守卫测试对齐：

```text
domain/Cargo.toml
  [dev-dependencies]
  feathertalk-project = { path = "../feathertalk-project" }
```

dev 依赖对下游不传递，`app` 的真实依赖图中不会因此出现 `project`。守卫测试不复述规则，而是
调用 `project` 的真实公开校验器，详见第 9 节。

需如实记录代价：五态词表自此有两份定义，依靠守卫测试在漂移时失败而非静默分叉；底层类型层
向上 dev 依赖 `project` 在结构上略显别扭。这是不改动既有已验证代码所付的代价。

## 4. 任务标识与生命周期

`TaskId` 只做校验与排序，不负责生成。形态为 `{13 位毫秒}-{8 位小写 hex}`，`Ord` 按字符串
比较即等于时间序，满足 §12 启动扫描未完成任务与 §10.5 历史排序的需要。生成放在切片 2 与
切片 4——时钟与计数器在那里，`domain` 因此不引入随机数或时钟依赖。该形态同时满足
`feathertalk-project` 现有的 128 字节标识符约束，由守卫测试锁定。

两个轴分开定义。`TaskStage` 是带载荷的实时事件流，`TaskStatus` 是持久化到 `project.json` 的
五态粗粒度状态。`domain` 提供 `TaskStage -> TaskStatus` 的全函数投影，穷尽匹配：`Queued` 对应
`Queued`，`Completed`、`Failed`、`Cancelled` 各自对应，其余全部折叠为 `Running`。新增阶段而
遗漏映射将无法编译。

`TaskKind` 是封闭枚举，但不改变磁盘格式。`feathertalk-project::validate_kind` 当前只约束字符类
（1-64 位小写 ASCII、数字、`_`、`-`），并非封闭词表。因此 `TaskKind` 的 serde 形态取满足该
字符类的 slug，`TaskHistoryEntry.kind` 在磁盘上仍为 `String`，无需 bump `schema_version`，现存
清单继续通过校验。写入方一律使用 `TaskKind::as_slug()`；读取方遇到无法识别的 slug 按未知类型
展示，而不是让整份清单校验失败。

阶段列表在 §11 十一个变体基础上新增 `Importing`。命令到阶段的映射为：

```text
NormalizeMedia            Preparing -> ExtractingAudio -> Completed
ValidateProject           Preparing -> Completed
LockAssetPackage          Preparing -> Completed
ExtractFrames             Preparing -> ExtractingFrames -> DetectingFaces -> Completed
ExtractFeatures           Preparing -> ExtractingFeatures -> Completed
Train                     Preparing -> Training{epoch,step,loss} -> Completed
Render                    Preparing -> Rendering{frame,total} -> Completed
ExportModelPackage        Preparing -> Exporting -> Completed
ExportOnnx                Preparing -> Exporting -> Completed
ImportLegacyModel         Preparing -> Importing -> Completed
MigrateLegacyFeatures     Preparing -> Importing -> Completed
ProbeMedia / InspectModel Preparing -> Completed
任意命令                   -> Failed{code,message} | Cancelled
```

新增 `Importing` 而非把导入映射到 `Exporting`，是因为后者会让界面在导入旧 `.pth` 时显示
"导出中"，与 §11"每个错误包含用户可读摘要"所体现的用户可读性取向相悖。一个变体的成本低于
长期让界面文案说反话。

`DetectingFaces` 是 `ExtractFrames` 的内部阶段而非独立命令：
`frame_pipeline::run_frame_pipeline_with_runner` 一次完成抽帧、SCRFD、PFLD、质量评估和原子
发布。命令词表与阶段词表因此并非一对一。

## 5. 命令词表

`Request` 是封闭枚举。每条命令对应现有 crate 已交付的入口，本切片不新增任何计算能力：

```text
ProbeMedia            -> media::probe_media
NormalizeMedia        -> media::normalize_media
ValidateProject       -> project::validate_project_dir
LockAssetPackage      -> project::lock_asset_package
ExtractFrames         -> frame_pipeline::run_frame_pipeline_with_runner
ExtractFeatures       -> audio::extract_long_audio + audio::commit_feature_artifact
Train                 -> training  { mode: Baseline | MouthRoi | Temporal,
                                     variant: OriginalUnet | MobileOneUnet }
Render                -> inference::execute_offline_render
InspectModel          -> weights / export 的清单与哈希检查
ImportLegacyModel     -> weights::import_into  { kind: LegacyModelKind }
ExportModelPackage    -> export::package
ExportOnnx            -> export::onnx_feather_hubert | export::onnx_unet
MigrateLegacyFeatures -> model-package 的 migrate 路径
```

加上控制面的 `Cancel { task_id }`，共十三条任务命令。

不设独立的 `Preview` 命令。`inference::RenderPlan::new` 的第三个参数即
`max_output_frames: Option<usize>`，预览是 `Some(n)`、完整渲染是 `None`，两者共用同一条
`frame()` 路径。§9 中"预览和完整渲染使用同一管线"在本切片通过不给 `Preview` 单独变体来兑现；
多一个变体等于为将来的管线分叉留门。

`Train` 的 `mode` 覆盖 §8 的三种训练模式，`variant` 覆盖 §5.2 的两种 UNet。两者是独立维度，
不合并为六个变体。

## 6. 事件、进度与指标

事件信封按 §4.2 定义，包含 `task_id`、阶段、进度、时间和可选指标：

```text
Event {
  protocol_version, task_id, emitted_at,
  stage: TaskStage,
  progress: Option<Progress>,
  metrics: Metrics,
}
Progress { completed: u64, total: Option<u64> }
Metrics  { samples_per_second: Option<f64>, eta_seconds: Option<f64>, vram_bytes: Option<u64> }
```

阶段自带的载荷（`Training{epoch,step,loss}`、`Rendering{frame,total}`）保留，另立统一的
`Progress` 让界面进度条有单一来源，无需为每个阶段各写一套换算。`total` 为 `Option` 是因为长
音频特征提取在开始时拿不到总量。

`Metrics` 采用封闭结构而非 `map<String, f64>`，三个字段正好覆盖 §10.2 要求显示的每秒样本数、
预计剩余时间和当前显存。开放 map 会把字段名拼写错误推到运行时，与第 2 节选择封闭枚举的
理由相同。`Metrics` 本身不是 `Option`：不携带指标的阶段发送三个字段全为 `None` 的实例，使
界面无需区分"没有指标"与"指标为空"两种情形。

## 7. 错误模型

§11 要求每个错误包含用户可读摘要、技术详情、任务阶段和可恢复建议，对应四个字段：

```text
TaskError { code: ErrorCode, summary: String, detail: String, stage: TaskStage, recovery: Recovery }

ErrorCode = MEDIA_INVALID | FACE_NOT_FOUND | LANDMARK_INVALID | FEATURE_SHAPE_MISMATCH
          | MODEL_INCOMPATIBLE | GPU_OUT_OF_MEMORY | GPU_DEVICE_LOST | DISK_SPACE_LOW
          | WORKER_CRASHED | TASK_CANCELLED

Recovery  = Retry | ResumeFromCheckpoint | FreeDiskSpace | SelectDifferentAdapter
          | ExcludeBadFrames | ReimportModel | NotRecoverable
```

`recovery` 是封闭枚举而非建议文本，目的是让界面能渲染出一个可点击的动作。例如
`GPU_DEVICE_LOST` 配 `ResumeFromCheckpoint`，界面直接给出"从最近 checkpoint 恢复"，这正是
§11 中"GPU device lost 后重启 worker，并允许从最近 checkpoint 恢复"落到交互层的形态。自由
文本只能印出一段话。

既有错误类型到十个错误码的映射不属于本切片。`MediaError`、`PreprocessError`、`AudioError`、
`PipelineError`、`TrainingError`、`InferenceError`、`PackageError`、`WeightImportError`、
`ScrfdError`、`PfldError` 分散在各计算 crate，`domain` 若要映射它们就必须依赖它们，第 3 节
那条"`app` 编译期拖不进模型代码"当即失效。因此本切片只定义码值与形状，映射函数放在切片 2
的 worker——那里这些 crate 本来就是依赖。

`TaskError::validate()` 只检查摘要非空与各字段长度上限。§11 中"底层 Rust panic、WGPU debug
文本和 FFmpeg 命令行不直接作为用户提示"是构造方的义务，在切片 2 连同映射一并测试。

§11 同时列出了 `Cancelled` 阶段和 `TASK_CANCELLED` 错误码，两者分工需明确：用户主动取消走
正常路径，只发 `TaskStage::Cancelled`，不构造 `TaskError`；`TASK_CANCELLED` 仅用于调用方需要
错误形状结果的场合，例如 CLI 把取消映射为非零退出码。同一次取消不会既发 `Cancelled` 阶段又
发一个带 `TASK_CANCELLED` 的 `Failed` 阶段，这条由 `TaskLifecycle` 的终态唯一性保证。

## 8. 帧格式、握手、版本与取消

一行一个紧凑 JSON 对象，UTF-8，`\n` 结尾。serde_json 的紧凑输出不含内嵌换行，因此行边界即
帧边界。两个方向各一个封闭枚举：

```text
ClientFrame = Start    { protocol_version, task_id, request }
            | Cancel   { protocol_version, task_id }
            | Shutdown { protocol_version }

ServerFrame = Ready    { protocol_version, worker_version, backends, adapters, capabilities }
            | Event    { ... }
            | Rejected { protocol_version, reason }
```

不设 `Hello` 帧。§4.2 写的是"worker 启动时报告版本、支持的 backend、adapter 和功能列表"，因此
worker 主动把 `Ready` 作为第一行输出，桌面端在校验通过前不发送任何帧，省掉一次往返和一个
变体。worker 侧保留 `Rejected`：桌面端发来版本不符的帧时回以 `Rejected` 且不执行，两端各自
设防。

版本策略是单个 `PROTOCOL_VERSION: u32` 精确相等，不做区间协商。按 §13，标准安装包不要求
用户自备任何运行环境，桌面端与 worker 由同一个包一起分发，不存在版本错配场景。不相等时
桌面端拒绝启动任务并给出可操作错误，即 §4.2 最后一条。

能力报告的结构：

```text
Ready { protocol_version, worker_version, backends, adapters, capabilities }

Backend      = Cpu | Wgpu
AdapterInfo  { id, name, backend, kind, certified: bool, vram_bytes: Option<u64> }
AdapterKind  = Discrete | Integrated | Cpu | Other
Capabilities { training: bool, wgpu_training: bool, onnx_validation: bool, ffmpeg: bool }
```

其中两个字段由上游需求决定。`certified` 对应 §4.4——AMD 和 Intel adapter 可以在界面中显示并
进入实验性检测，但首发版不将其列为发布承诺，界面需要能表达这个区别。`id` 是稳定标识，因为
§10.5 要求"一个 GPU adapter 同时只运行一个训练或推理任务"，切片 2 的队列以它作互斥键。

取消的幂等性由类型保证而非约定。§4.2 只声明取消是幂等操作，落到实现需要三条规则：重复
`Cancel` 效果相同；对未知 `task_id` 的取消静默接受而不报错，因为幂等意味着调用方不必先知道
当前状态；已进入终态的任务不再产生新事件。因此本切片提供 `TaskLifecycle` 转移校验器，拒绝
`Completed -> Cancelled` 一类非法转移，并保证每个任务最多产生一个 `Cancelled`。这是纯逻辑、
可穷尽测试，且 worker 与 app 需要的是同一套规则，放在 `domain` 只写一遍。

配套两个谓词：`TaskStage::is_terminal()`，以及 `TaskStatus::is_incomplete()`（`Queued | Running`）
供 §12 中"应用启动时扫描并提供恢复或清理未完成任务"复用。

`Shutdown` 不带参数，语义固定为停止并保存——§10.5 要求应用退出时训练任务先停后存。worker
卡死属于监督进程应当终止的情况，不由协议表达。

帧长设上限常量 `MAX_FRAME_BYTES`，解码器拒绝超长行。仓库既有此纪律（`MAX_CAPTURE_BYTES`、
`read_bounded_regular`、`MAX_LICENSE_BYTES`、`DEFAULT_MAX_FRAME_PIXELS`），协议层不例外。

配一对基于 `std::io` 的 `FrameReader<R: BufRead>` 和 `FrameWriter<W: Write>`，只用 std，不触及
进程。上限检查因此有唯一落点，切片 2 只需递入真实的 stdin 与 stdout 句柄。

## 9. 测试与验收

沿用仓库的 TDD 节奏与 `tests/*.rs` 布局：

```text
public_api.rs             公开类型可从 crate 根构造与匹配，沿用既有 public_api.rs 惯例
frame_codec.rs            六个帧变体逐个 round-trip；紧凑单行；deny_unknown_fields 拒绝
                          多余字段；超过 MAX_FRAME_BYTES 的行被拒
stream_io.rs              FrameReader / FrameWriter over Cursor 与内存管道；半行、空行、
                          超长行、非 UTF-8 的处理
handshake.rs              Ready 作为首帧解析；protocol_version 不等即拒绝并带可操作原因；
                          certified=false 的 adapter 仍可枚举
lifecycle.rs              合法与非法转移穷尽；重复 Cancel 幂等；终态后不再产生事件；
                          每个任务最多一个 Cancelled
stage_status.rs           TaskStage -> TaskStatus 投影穷尽；is_terminal 与 is_incomplete
error_model.rs            十个 ErrorCode 的 serde 形态；validate 拒绝空摘要与超长字段；
                          每个 ErrorCode 有默认 Recovery 且穷尽
project_compatibility.rs  四条守卫，dev 依赖 feathertalk-project
golden_frames.rs          固定 JSON 文本行作为 golden
```

第 3 节提到的四条守卫具体为：`TaskStatus` 与 `TaskHistoryStatus` 的 serde slug 集合及变体数量
必须相等；把每个 `TaskKind::as_slug()` 放入 `TaskHistoryEntry.kind` 后真实
`ProjectManifest::validate()` 必须通过；把规范形态的 `TaskId` 放入 `TaskHistoryEntry.task_id` 后
`validate()` 必须通过；`TaskStage -> TaskStatus` 投影穷尽。第二与第三条调用 `project` 的真实公开
校验器，而不在测试中重抄那两条规则。

`golden_frames.rs` 单独存在是为了堵住 round-trip 的盲区：若编码与解码两侧被同时改名，
round-trip 依然通过，线格式却已变化——而 worker 与桌面端是两个独立进程，跨版本时这就是静默
不兼容。固定若干条 JSON 文本行做 golden 可以捕获这类改动，与 §14.1 的 golden 基准取向一致。

验收命令为仓库标准五条，全部要求退出码 0：

```text
cargo test --workspace --all-targets
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
git diff --check
```

## 10. 后续衔接

里程碑五拆为九个切片，本文是第一个：

```text
1  协议与任务域契约        feathertalk-domain（本切片）
2  worker 进程与任务执行    feathertalk-worker：命令派发、任务队列、GPU adapter 互斥、
                           取消与停止保存、既有错误类型到十个错误码的映射
3  CLI 与 worker 能力对等   feathertalk-cli：§16 完成定义的硬要求，同时为后续 UI 切片
                           提供无界面驱动路径
4  桌面端 worker 监督与恢复 拉起、监控、重启 worker；崩溃与 device lost 隔离；保存最后
                           日志；启动扫描未完成任务后恢复或清理
5  GPUI 应用外壳与任务页    导航骨架、项目工作台、任务页，端到端跑通事件管道
6  素材页                   导入、指标展示、时间轴坏帧标记、关键点 overlay、排除与重检、
                           锁定素材包版本
7  训练页                   预设与高级面板、实时 epoch/step/loss/样本速率/ETA、固定样本
                           预测与嘴部 ROI、显存与 worker 状态
8  生成页                   选素材包、checkpoint、驱动音频；输出设置；预览与完整渲染；
                           播放或打开输出目录
9  模型页                   导入旧 .pth/.pth.tar；展示类型、参数量、shape、哈希、兼容状态；
                           导出标准包或 ONNX；拒绝架构不匹配
```

切片 1 至 4 无界面，可按现有 TDD 节奏推进；5 至 9 依赖 GPUI。

一个前置风险需在切片 5 开工前处理：仓库当前没有任何 `gpui` 依赖。能否锁定版本并在 §4.4 的三
个首发平台上构建通过，建议在切片 4 结束后用一次独立 spike 验证，而不是在编写页面时才发现。

本切片交付后，切片 2 与切片 3 可并行开工——两者都只依赖本文定义的 `Request`、`Event` 和帧
格式，互不阻塞。

