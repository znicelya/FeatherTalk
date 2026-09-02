# 抽帧 worker 命令与 CLI 子命令设计

日期：2026-09-03
状态：已定稿

## 1. 目标与范围

本切片把三块已经各自可用的代码拼成一条端到端可调用的命令：`feathertalk-frame-pipeline`（抽帧、评估、发布）、`feathertalk-frame-adapters`（SCRFD 人脸检测、PFLD 关键点、JPEG 解码）、`feathertalk-worker` 与 `feathertalk-cli`（协议、进度、错误映射、命令行）。领域契约早已提交：`TaskKind::ExtractFrames`（slug `extract_frames`）、`ExtractFramesParams { project_dir, video }`、`Request::ExtractFrames`、阶段 `ExtractingFrames` 与 `DetectingFaces`。缺的只是 worker 侧的实现与 CLI 入口。

拼装之前必须修两个阻断性缺陷，它们都只在真实视频上才暴露。第一，抽帧时间戳格式错误，导致 25 fps 视频里每秒 25 个请求帧塌缩成 7 张不同的图，而且失败是静默的（见 §2）。第二，`FramePipelineSpec::new` 的包含规则会拒绝契约规定的输入路径 `<project>/assets/video_25fps.mp4`，因为它落在 `output_root = <project>/assets` 之内（见 §4）。

本切片范围内的改动：

- `feathertalk-frame-pipeline`：修正时间戳；抽帧从「一次 ffmpeg 一帧」改为分块（`FRAME_CHUNK = 250`）；收窄包含规则；新增观察者接缝 `PipelineObserver`/`PipelinePhase` 与 `PipelineError::Cancelled`，配套 `extract_frames_observed`/`evaluate_frames_observed`。
- `feathertalk-frame-adapters`：从 `lib.rs` 重新导出 `ScrfdArtifactPaths`。它出现在本 crate 公开的 `load` 签名里，不导出等于该公开 API 外部不可调用。
- `feathertalk-worker`：新增 `ModelToolchain` 与两个模型工件环境变量；`JobExecutor` 与 `execute` 改为传 `&WorkerConfig`；新增 `execute_extract_frames` 注入入口；新增 `quality_result.rs`；`error_map.rs` 新增 `pipeline_task_error`/`is_pipeline_cancellation`；握手把 `ExtractFrames` 加入 `supported_commands`。
- `feathertalk-cli`：新增 `extract-frames <PROJECT_DIR> <VIDEO>` 子命令，以及对应的 `build_request` 分支和 `UnsupportedCommand` 提示分支。

范围外的内容集中在 §16。

## 2. 抽帧时间戳缺陷与分块抽帧

### 2.1 缺陷

`crates/feathertalk-frame-pipeline/src/commands.rs:75-79` 现在这样构造 `-ss` 参数：

```rust
fn format_timestamp(index: u64) -> String {
    let seconds = index / FRAME_RATE;
    let remainder = index % FRAME_RATE;
    format!("{seconds}.{remainder:02}")
}
```

`remainder` 是「这一秒里的第几帧」（0..=24），却被写进了「百分之几秒」的位置。于是帧 27 请求的是 1.02 秒而不是 1.08 秒。25 fps 下一帧是 40 ms，而这个格式每秒只能表达 0.00..0.24 秒——真实取到的只有 7 张不同的图，其余 18 个请求命中重复帧。

在 `demo/feathertalk_demo_latest_188.mp4`（h264 1280×720，25/1 fps，1511 帧，60.48 s）上逐帧提取，并与一次性 `-vf fps=25` 的基准做 SHA-256 比对，实测：

| 请求帧号 | 缺陷 `-ss` | 实际取到 | 修正 `-ss` | 实际取到 |
| --- | --- | --- | --- | --- |
| 25 | 1.00 | 25 | 1.000 | 25 |
| 26 | 1.01 | 26 | 1.040 | 26 |
| 27 | 1.02 | 26 | 1.080 | 27 |
| 28 | 1.03 | 26 | 1.120 | 28 |
| 30 | 1.05 | 27 | 1.200 | 30 |
| 34 | 1.09 | 28 | 1.360 | 34 |
| 48 | 1.23 | 31 | 1.920 | 48 |
| 49 | 1.24 | 31 | 1.960 | 49 |
| 50 | 2.00 | 50 | 2.000 | 50 |

每个输出文件都是合法 JPEG、都是一张人脸，`inspect_frame` 全部通过，`quality.json` 也全部 accepted。这正是 `2026-08-24-frame-face-pipeline-design.md` §4「不静默排除坏帧」要防的那类失败，只不过这里是静默地把坏数据当好数据。现有测试 `crates/feathertalk-frame-pipeline/tests/commands.rs:57` 断言帧 26 的时间戳是 `"1.01"`，把缺陷固化成了期望，必须改成 `"1.040"`。

### 2.2 修正

时间戳改为毫秒三位：

```rust
fn format_timestamp(index: u64) -> String {
    let seconds = index / FRAME_RATE;
    let millis = (index % FRAME_RATE) * 40;
    format!("{seconds}.{millis:03}")
}
```

25 fps 下 `(index % 25) * 40` 恒为 40 的整数倍且小于 1000，三位十进制精确表示，不存在舍入。

同时把「一次 ffmpeg 一帧」改成「一次 ffmpeg 一块」。`frame_command` 变为块命令，参数为 `(extractor, source, first_index, count, output_pattern)`，其中 `output_pattern` 是 `<staging>/frames/%06d.jpg`：

```
ffmpeg -hide_banner -loglevel error -ss <ts(first_index)> -i <video>
  -map 0:v:0 -an -sn -dn -map_metadata -1 -map_chapters -1
  -vf fps=25 -frames:v <count> -start_number <first_index>
  -q:v 2 -f image2 <staging>/frames/%06d.jpg
```

`-start_number <first_index>` 配合 `%06d` 让 ffmpeg 直接写出最终文件名，与 `FramePipelineSpec::frame_path` 的 `frames/NNNNNN.jpg` 完全一致，所以 `publish` 阶段不需要任何改动。`-vf fps=25` 在 `-ss` 之后按输出帧率重采样，块内帧序号连续。

块大小 `FRAME_CHUNK: u64 = 250`，公开导出（测试与 worker 都要按它算期望值）。抽帧循环变成：按 `FRAME_CHUNK` 切分 `[0, frame_count)`，每块跑一次 ffmpeg，块返回后对该块的每一帧仍然逐帧调用 `inspect_frame`（拒绝缺失、符号链接、非常规文件、空文件、超过 16 MiB，`sync_all` 后算 SHA-256）。ffmpeg 少写文件时 `inspect_frame` 报 `FrameMissing`，直接失败，不做任何补偿性重试——`frame_count` 来自 ffprobe（§6.2），比容器元数据更可信；真对不上时，宁可响亮失败也不要偷偷少几帧。

### 2.3 实测依据

同一台机器、缓存预热后，1511 帧三种结构的耗时：

| 结构 | ffmpeg 调用次数 | 总耗时 | 每帧 |
| --- | --- | --- | --- |
| 一次性 `-vf fps=25` | 1 | 2907 ms | 1.9 ms |
| 分块，chunk=250 | 7 | 3199 ms | 2.1 ms |
| 逐帧（现状） | 1511 | 约 255 s（推算） | 129 / 185 / 193 ms（帧 0 / 700 / 1486） |

逐帧方案的每帧耗时随 `-ss` 增大而上升，因为每次都要重新解析并 seek。分块方案相对一次性方案只多 10% 开销，而 1511 帧从 4 分钟降到 3.2 秒，约 80 倍。

分块结果与一次性基准逐字节一致，四组校验零不匹配：

- 块起点 250、250 帧，写出 `000250..000499`，1149 ms，0 不匹配。
- 块起点 263（时间戳 `10.520`，非整秒边界）、25 帧，写出 `000263..000287`，180 ms，0 不匹配。
- 尾块起点 1500、11 帧，写出 `001500..001510`，126 ms，0 不匹配。
- 完整 7 块（250×6 + 11），1511 个文件全部写出，3199 ms，0 不匹配。

非整秒起点单独验证，是因为块起点通常不落在整秒上，而 `-ss` 加 `-vf fps` 的组合在这种情况下最容易出现相位偏移。

### 2.4 为什么是 250 帧

250 帧等于 10 秒素材、约 0.5 秒 ffmpeg 时间，是三个尺度的折中：

- 进度粒度：抽帧阶段每块报一次进度，1511 帧共 7 次，客户端不会长时间无输出，也不会被事件淹没。
- 取消粒度：取消只在块边界与逐帧评估边界生效（§3），块越大响应越慢，0.5 秒是可接受的残余窗口。
- seek 开销：块越小，重复 seek 越多，那 10% 的额外开销会重新长回来。

块大小不做成参数。它不是用户关心的量，暴露出来只会变成一个需要文档和校验的旋钮。

## 3. frame-pipeline 的观察者与取消

`feathertalk-frame-pipeline` 现在完全不知道进度和取消的存在，而 worker 必须在长达数分钟的推理过程中报进度并响应取消。新增一个最小接缝：

```rust
pub enum PipelinePhase {
    Extracting { completed: u64, total: u64 },
    Evaluating { completed: u64, total: u64 },
}

pub trait PipelineObserver {
    fn phase(&self, _phase: PipelinePhase) {}
    fn is_cancelled(&self) -> bool {
        false
    }
}

pub struct NoObserver;

impl PipelineObserver for NoObserver {}

pub fn extract_frames_observed(
    spec: &FramePipelineSpec,
    extractor: &FrameExtractor,
    runner: &dyn ProcessRunner,
    observer: &dyn PipelineObserver,
) -> Result<FrameBatch, PipelineError>;

pub fn evaluate_frames_observed(
    batch: &FrameBatch,
    decoder: &dyn FrameDecoder,
    detector: &dyn FaceDetector,
    predictor: &dyn LandmarkPredictor,
    observer: &dyn PipelineObserver,
) -> Result<FrameEvaluation, PipelineError>;
```

没有 `Publishing` 阶段。发布阶段不报进度（§9），worker 自己组合三段（§5.1），所以枚举里放一个没人发送的变体只是负担。

观察者上不加 `Send`/`Sync` 约束。worker 的 `ChannelReporter` 持有 `mpsc::Sender<ControlMessage>`，`Sender` 不是 `Sync`；一旦要求 `Sync`，worker 就无法把自己的 reporter 包成观察者，接缝也就白设计了。`&dyn PipelineObserver` 只在单线程调用链上传递，不需要任何线程约束。

用 trait 而不是两个闭包，是因为这个接缝是双向的：一个方向报出阶段，另一个方向问「要不要停」。两个闭包参数会让签名更长，也无法给出 `NoObserver` 这种一次性默认实现，而带默认方法的 trait 让现有的 `extract_frames_with_runner`/`evaluate_frames_with_models` 直接用 `NoObserver` 转调，老调用点零改动。

`completed` 语义：抽帧阶段在每块的 `inspect_frame` 全部通过后报一次，值为已确认落盘的帧数；评估阶段在每帧处理完后报一次，值为已评估帧数。评估阶段逐帧上报，因为单帧推理是几十毫秒级，1511 个事件对通道不构成压力；抽帧阶段一块 0.5 秒，逐帧上报只会毫无意义地放大事件量。

`evaluate_frames_with_models<D, F, L>` 的泛型边界已经是 `?Sized`（`evaluate.rs:143-146`），而 `FrameDecoder`/`FaceDetector`/`LandmarkPredictor` 三个 trait 本身对象安全，所以传 `&dyn` 不需要改任何现有签名。

新增错误变体：

```rust
Cancelled { operation: &'static str }
```

已确认 `PipelineError` 在 `feathertalk-frame-pipeline` 之外没有穷尽 `match`（`feathertalk-frame-adapters` 只构造 `Adapter` 与 `Io`），新增变体不会破坏下游编译。

不在 frame-pipeline 里引入「能杀子进程的 runner」。normalize-media 那样做是合理的：它是一次可能跑很久的 ffmpeg，不杀进程就只能等超时。抽帧不同——一块 ffmpeg 只有 0.5 秒，在块边界检查取消已经足够快。要跨 crate 复用 `feathertalk-media` 的 `CancellableProcessRunner`，得先桥接两套互不相同的 `CommandSpec`/`ProcessOutput` 类型，而 frame-pipeline 的 `CommandSpec::new` 是 `pub(crate)`，外部无法构造；这个代价与 0.5 秒的收益不成比例。frame-pipeline 也因此不需要新增对 `feathertalk-media` 的依赖（目前确实没有这条依赖）。

## 4. FramePipelineSpec 包含规则收窄

`crates/feathertalk-frame-pipeline/src/model.rs:33` 现在是：

```rust
if video_path == output_root || video_path.starts_with(&output_root) {
```

契约规定的输入是 `<project>/assets/video_25fps.mp4`（`feathertalk-project` 的 `REQUIRED_FILES` 第一项），而 `output_root` 是 `<project>/assets`（§6.1 说明为什么固定成它）。这条规则会把唯一合法的输入直接拒掉。

收窄为四条：

1. `video_path == output_root`，拒绝。
2. `video_path` 在 `output_root` 之内，但其父目录不是 `output_root`，拒绝。
3. `video_path == output_root/quality.json`，拒绝。
4. 其余允许。

第 2 条是关键。它用一句话同时挡住 `frames/`、`landmarks/`，以及形如 `.feathertalk-frame-build-{pid}-{counter}`、`.feathertalk-frame-backup-{pid}-{counter}` 的暂存与备份目录。后者名字里带进程号和计数器，无法枚举；而 `extract_frames_with_runner` 在任何错误路径和 `Drop` 里都会 `remove_dir_all` 暂存目录——如果有人把视频放进去，输入文件会被删掉。「必须是 `output_root` 的直接子文件」比逐个列举安全，也不会随暂存命名策略变化而失效。

第 3 条单列，因为 `quality.json` 是 `output_root` 的直接子文件，逃得过第 2 条。

`image_width`/`image_height` 仍然只做校验（上限 `MAX_IMAGE_DIMENSION = 32_768`），管线内部不读取它们：ffmpeg 输出多大就是多大，尺寸只用于早期发现明显不合理的探测结果。本切片不改这个行为，只是记录下来，免得下次有人误以为它们参与了缩放。

## 5. Worker 命令编排

### 5.1 自行组合三段

不调用 `run_frame_pipeline_with_runner`。它把 extract、evaluate、publish 三段包成一个调用，`FrameEvaluation` 留在函数内部，失败时 worker 只能拿到 `QualityRejected { count }`，报不出「哪几帧坏了、为什么」。而主线路线图 `2026-08-17-rust-desktop-migration-design.md` §11 明确要求坏帧定位到具体帧。所以 worker 依次调用：

```
extract_frames_observed(&spec, &extractor, &runner, &observer)?
evaluate_frames_observed(&batch, decoder, detector, predictor, &observer)?
publish_frame_artifacts(...)
```

拿到 `FrameEvaluation` 后，若 `is_success()` 为假，直接从 `anomalies()` 里挑第一条决定错误码与摘要（§13），不走 `QualityRejected`。

### 5.2 注入接缝

新增可注入入口，便于用假 runner 与假模型做单元测试：

```rust
pub fn execute_extract_frames<M, F>(
    params: &ExtractFramesParams,
    config: &WorkerConfig,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    media_runner: &M,
    frame_runner: &F,
    decoder: &dyn FrameDecoder,
    detector: &dyn FaceDetector,
    predictor: &dyn LandmarkPredictor,
) -> CommandOutcome
where
    M: feathertalk_media::ProcessRunner + ?Sized,
    F: feathertalk_frame_pipeline::ProcessRunner + ?Sized;
```

两个 runner 是两个 crate 的不同 trait：ffprobe 探测走 `feathertalk-media`，抽帧走 `feathertalk-frame-pipeline`。`?Sized` 加上三个模型 trait 的对象安全性，让测试传 `&dyn` 不产生额外成本。

`execute_with_runner` 的 `Request::ExtractFrames` 分支负责默认装配：用 media runner 跑一次 `probe_media_with_runner`，构造 `SystemProcessRunner`，加载三个适配器，然后转调 `execute_extract_frames`。现有 worker 测试大多注入假件、不加载权重，这一层拆分让它们不受影响。

### 5.3 JobExecutor 改为传 &WorkerConfig

```rust
pub type JobExecutor = Box<
    dyn Fn(&Request, &WorkerConfig, &CancellationToken, &dyn TaskReporter) -> CommandOutcome
        + Send
        + 'static,
>;
```

`execute` 与 `execute_with_runner` 的第二个参数同步从 `Option<&MediaToolchain>` 改为 `&WorkerConfig`。理由：`extract_frames` 需要两套工具链（ffmpeg/ffprobe 与 SCRFD/PFLD），下一切片的 FeatherHuBERT 会再加一套；每加一套就改一次签名，不如一次改成传只读配置快照。`WorkerConfig` 本来就是 `Clone`、字段全 `Send`、启动后不再变化，正是为此形状准备的。代价是大约 20 处机械的测试改动（`worker/tests/runtime.rs` 里约 8 个闭包，加上 `runtime.rs` 内部的调用点）。

## 6. 准入检查与帧元数据来源

### 6.1 不调用 validate_project_dir

`feathertalk-project::validate_project_dir` 要求包状态为 `Locked`，且 `assets/frames`、`assets/landmarks`、`assets/features/feather_hubert.f32` 都已存在。抽帧命令正是要创建 `assets/frames` 和 `assets/landmarks`，而且此时 FeatherHuBERT 特征还不存在——它与 `extract_frames_with_runner` 的 `reject_final_destinations`（存在即拒）直接矛盾。所以准入检查在 worker 里本地做：

- `project_dir`：绝对路径、存在、是目录、不是符号链接。
- `project_dir/project.json`：存在且是常规文件。只查存在性，不解析、不要求 `Locked`。这是一道便宜的护栏——目录敲错时立刻失败，而不是先跑几分钟推理。
- `video`：走 `feathertalk_media::validate_input`，与 probe-media、normalize-media 保持同一套输入校验。
- 帧率：ffprobe 报告的 `frame_rate` 必须是 25/1。管线里 `FRAME_RATE` 是硬编码常量，非 25 fps 的输入会静默错位。

`output_root` 固定为 `<project_dir>/assets`，不可配置。理由同 normalize-media 的「规格固定而非可配」：产物布局是契约的一部分，可配只会制造出下游无法识别的目录结构。

### 6.2 帧数与尺寸来自 ffprobe

`frame_count`、`image_width`、`image_height` 全部取自同一次 `probe_media_with_runner` 返回的 `MediaProbe.video`，绝不从请求里取。请求里只有 `project_dir` 和 `video` 两个字段，本来也没有地方放它们；更重要的是，帧数决定了要抽多少帧，让调用方提供等于把一致性责任推给客户端。

### 6.3 quality.json 与既有产物重跑

`quality.json` 落在 `<project_dir>/assets/quality.json`，与 `assets.json` 同级。`2026-08-17` §6.1 的工程目录树没有列出这个文件，这里显式补上。

已有抽帧结果时不覆盖、不合并、不续跑：`reject_final_destinations` 抛 `OutputDestinationExists`，映射为 `MEDIA_INVALID`，摘要「素材目录已存在抽帧结果」。不加 `force` 参数——`ExtractFramesParams` 已经提交且带 `deny_unknown_fields`，加字段是线协议变更，应该和「重跑与修复流程」一起单独做（§16）。

## 7. 模型工件发现

SCRFD 与 PFLD 的工件目录通过环境变量给出：

- `FEATHERTALK_WORKER_SCRFD_DIR`
- `FEATHERTALK_WORKER_PFLD_DIR`

各自是绝对目录路径，内含 `manifest.json` 与 `model.safetensors`。仓库内已提交可用工件：`rust/crates/feathertalk-scrfd/artifacts/scrfd_2_5g/`、`rust/crates/feathertalk-pfld/artifacts/pfld_ghost_one/`。

`worker/src/config.rs` 新增：

```rust
pub struct ModelToolchain {
    scrfd_dir: PathBuf,
    pfld_dir: PathBuf,
}
```

`WorkerConfig` 新增 `models: Option<ModelToolchain>` 与 `model_rejection: Option<String>`，与现有 `media`/`media_rejection` 对称，复用 `required_path`：只校验非空加绝对路径，不检查存在性。这一点刻意与 `MediaToolchain` 一致——启动期不碰文件系统，配置错误在命令执行时以正常的任务失败呈现，而不是让 worker 起不来。

依赖变化：worker 新增 `feathertalk-frame-pipeline`、`feathertalk-frame-adapters`、`feathertalk-models`（`CpuBackend = NdArray<f32>`；它目前只是 frame-adapters 的 dev-dependency，worker 需要正式依赖）。同时把 `ScrfdArtifactPaths` 从 `feathertalk-frame-adapters/src/lib.rs` 重新导出——它出现在该 crate 公开的 `ScrfdFaceDetector::load` 签名里，不导出的话外部根本无法调用。

## 8. PFLD 大栈加载

PFLD GhostOne 的权重加载会深度递归，默认栈会溢出。既有先例是 `feathertalk-frame-adapters/tests/pfld_model.rs:28-57` 与 `feathertalk-weights/src/pfld/mod.rs:191-215`（`PFLD_DETACHED_CLONE_STACK_BYTES`）。worker 沿用同样的做法：

```rust
let handle = std::thread::Builder::new()
    .stack_size(PFLD_LOAD_STACK_BYTES) // 64 MiB
    .spawn(move || PfldLandmarkPredictor::<CpuBackend>::load(&pfld_dir, device, cache))?;
let predictor = handle.join().map_err(...)??;
```

只有加载在大栈线程上，推理留在正常线程。把整条管线搬到大栈的 scoped thread 会要求 `TaskReporter: Sync`（观察者要跨线程借用 reporter），而 `ChannelReporter` 持有 `Sender`、不是 `Sync`。这条路与 §3 不加 `Sync` 约束是同一个约束的两面，已排除。

SCRFD 与 JPEG 解码器在正常线程加载。三者共享同一个 `Arc<FrameImageCache>`（单槽共享缓存，`DEFAULT_MAX_FRAME_PIXELS = 64 MiB`），避免同一帧被解码多次。

`FrameExtractor` 复用媒体超时：`FrameExtractor::new(toolchain.ffmpeg().to_owned(), toolchain.timeout())`。默认 300 000 ms 对 0.5 秒一块的 ffmpeg 绰绰有余，不新增超时环境变量。

## 9. 阶段与进度映射

| 来源 | 上报阶段 | completed / total |
| --- | --- | --- |
| 探测与模型加载 | 不上报 | 无 |
| `Extracting { completed }` | `ExtractingFrames` | `completed / frame_count` |
| `Evaluating { completed }` | `DetectingFaces` | `completed / frame_count` |
| 发布 | 不上报 | 无 |

探测和模型加载期间不上报，因为运行时在受理任务时已经发过一次 `preparing`，再发一条同样的事件不携带新信息。发布阶段不上报，与 normalize-media 把 `Verifying | Committing` 直接 `return` 同理：它是一次原子的目录改名，没有可分级的进度。

两个阶段的计数都从 0 重新开始。这是合法的——`feathertalk-domain/src/event.rs:70-78` 的 `Event::validate` 只校验单条事件内 `progress.completed <= total`，不存在跨事件的单调性约束，也没有跨阶段的计数器连续性要求。

客户端看到的序列：`preparing` → `extracting_frames 0/1511 … 1511/1511` → `detecting_faces 0/1511 … 1511/1511` → 终态。

`samples_per_second`、`eta_seconds`、`vram_bytes` 留空。前两个需要训练那样的稳定吞吐模型，第三个只在 GPU 后端有意义，而本切片只跑 CPU。

## 10. 结果载荷

新增 `worker/src/quality_result.rs`，形状照 `normalize_result.rs`（`serde_json::json!` 加 `path.display().to_string()`）：

```json
{
  "output_dir": "<project>/assets",
  "frames_dir": "<project>/assets/frames",
  "landmarks_dir": "<project>/assets/landmarks",
  "quality_report": "<project>/assets/quality.json",
  "frame_count": 1511,
  "frame_width": 1280,
  "frame_height": 720
}
```

三个目录与文件路径都给出，是因为 CLI 与后续的加锁步骤都需要它们，而从 `project_dir` 反推布局会把布局知识复制到调用方。

三处刻意不放：

- 不放逐帧数组。1511 条记录会把单行 JSON 事件撑到几百 KB；需要细节就读 `quality.json`。
- 不放 `accepted_count`。`publish` 只在所有帧都被接受时才成功，成功路径上它恒等于 `frame_count`。
- 不放 `anomalies`。成功时它必然为空，失败时结果载荷根本不会产生。

## 11. 握手与拒绝文本

`supported_commands` 在 `config.media().is_some() && config.models().is_some()` 时追加 `ExtractFrames`。两套工具链缺任何一套都不宣告支持——缺 ffmpeg 抽不出帧，缺模型评估不了。

`unsupported_reason` 保持媒体优先：媒体缺失时沿用现有文案，媒体齐备而模型缺失时给出模型相关的说明。这样用户一次只看到最靠前的那个缺失项，而不是一串。

`Capabilities` 结构不动。`supported_commands` 已经承载了「能不能抽帧」这个信息，加字段是协议变更，收益为零。

## 12. CLI 形态

```
feathertalk extract-frames <PROJECT_DIR> <VIDEO>
```

`Command` 枚举新增 `ExtractFrames { project_dir, video }`，`run.rs::build_request` 增加对应分支，复用现有中文拒绝文案「工程目录不能为空。」与「输入文件不能为空。」，单元测试加在 `run.rs` 末尾。整体照 `f04cd0f feat(cli): add the normalize-media subcommand` 与 `faed5a2 test(cli): cover normalize-media against the real worker` 的形状。

`render.rs` 的阶段中文标签已经齐备：`ExtractingFrames → "正在提取视频帧"`、`DetectingFaces → "正在检测人脸"`，不需要新增。`UnsupportedCommand` 提示新增 `extract_frames` 分支，把四个环境变量都列出来；该文件已有「把 worker 侧常量以字面量复制一份并注明 worker 常量是唯一来源」的约定，两个新变量名沿用这个约定。

## 13. 错误映射

`error_map.rs` 新增 `pipeline_task_error` 与 `is_pipeline_cancellation`，从 `lib.rs` 导出。

`PipelineError` 到 `ErrorCode`：

| 变体 | 错误码 |
| --- | --- |
| `InvalidField`、`InvalidReport`、`OutputDestinationExists`、`FrameMissing`、`FrameNotRegular`、`FrameEmpty`、`FrameTooLarge` | `MEDIA_INVALID` |
| `Io` | `io_error_code`，沿用现有映射 |
| `Adapter` | `MODEL_INCOMPATIBLE` |
| `Cancelled` | `TASK_CANCELLED` |
| `ToolFailed`、`ToolTimedOut`、`ToolOutputTooLarge`、`ToolSpawn`、`ReportJson`、`ReportNotRegular`、`ReportTooLarge`、`PublishFailed`、`PublishRollbackFailed`、`QualityRejected` | `WORKER_CRASHED` |

`Adapter` 映射到 `MODEL_INCOMPATIBLE` 而不是 `WORKER_CRASHED`，因为它的实际来源是权重加载失败和张量形状不匹配——问题在工件而不在 worker 进程。`technical_detail` 里带上对应的环境变量名，让用户知道该改哪个路径。

`Cancelled` 这一分支是完备性需要：运行时会先拦下取消并产生 `Cancelled` 结局，正常不会走到这里。

质量失败不走 `QualityRejected`（那条只在 `run_frame_pipeline_with_runner` 里产生，本切片不用它）。worker 从第一条异常挑码：

| `AnomalyCode` | 错误码 |
| --- | --- |
| `FaceNotFound`、`MultipleFaces`、`BboxOutOfBounds` | `FACE_NOT_FOUND` |
| `LandmarkInvalid` | `LANDMARK_INVALID` |
| `BlurredFrame`、`FrameDecodeFailed`、`FrameWriteFailed` | `MEDIA_INVALID` |
| `ModelFailed` | `MODEL_INCOMPATIBLE` |

摘要用中文 `&'static str`，与该文件既有风格一致。`technical_detail` 列出前若干条异常的「帧号加代码加摘要」，用现有 `clamp` 截到 `MAX_DETAIL_CHARS`。所有失败沿用 `FAILURE_STAGE = TaskStage::Preparing`。

## 14. 测试

组合逻辑全部由便宜的假 runner 单元测试承担，只保留一个昂贵的门控端到端测试。

frame-pipeline：

- `format_timestamp` 的边界：帧 0、24、25、26、49、50、1510，覆盖整秒与秒内相位；`tests/commands.rs:57` 的 `"1.01"` 改为 `"1.040"`。
- 块切分：1511 帧切成 250×6 + 11，断言每次命令的 `-ss`、`-frames:v`、`-start_number`。
- 观察者：`Extracting` 事件 7 次且 `completed` 单调递增到 1511；`Evaluating` 事件 1511 次。
- 取消：观察者在第 2 块、第 3 帧返回 `is_cancelled() == true`，断言得到 `PipelineError::Cancelled` 且暂存目录已被清理。
- `FrameMissing`：假 runner 少写一个文件。
- 包含规则四条拒绝路径各一个用例，外加合法的 `assets/video_25fps.mp4` 必须通过。

worker：

- 每个 `AnomalyCode` 分支的错误码映射各一个用例。
- `PipelineError` 各变体到错误码的映射。
- 准入检查失败路径：`project_dir` 不存在、不是目录、`project.json` 缺失、帧率非 25/1。
- 进度事件序列 `preparing` → `extracting_frames` → `detecting_faces`，用假观察者驱动。
- 握手：仅媒体、仅模型、两者齐备三种配置下 `supported_commands` 的内容。

CLI：`build_request` 的空参数拒绝与正常构造。

端到端（门控）：沿用 `cli/tests/real_worker.rs` 的 `FEATHERTALK_REQUIRE_E2E` 加 `worker_or_skip` 加 `real_tool` 模式。测试时从已入库的 `demo/feathertalk_demo_latest_188.mp4` 裁出自帧 750 起的 25 帧（1 秒）片段，指向仓库内的 SCRFD/PFLD 工件，断言 `frames/`、`landmarks/` 各 25 个文件，`quality.json` 的 `frame_count == 25` 且无异常。

选帧 750 是因为它是唯一有已记录基线的帧：`crates/feathertalk-frame-adapters/tests/fixtures/demo_frame_v1/fixture.json` 记录 `frame_index: 750`、人脸分 0.8108（阈值 0.50）、拉普拉斯方差 776.03（阈值 20.0），两项都远高于阈值。若窗口内出现异常，缩小或平移窗口，并把最终窗口写进测试注释。

不新增二进制夹具。片段在测试运行时生成，避免 `.gitignore` 的 `*.mp4` 规则与 `git add -f`。测试套件里不跑完整 1511 帧。

## 15. 验证

- `cargo check`
- `cargo test --workspace --all-targets`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo fmt --all -- --check`
- `git diff --check`

全部零告警、零失败。cargo 命令在 `rust/` 下执行。

## 16. 范围外

- 重跑与修复流程（`force` 覆盖、单帧重抽、部分续跑）。`ExtractFramesParams` 带 `deny_unknown_fields`，加字段是线协议变更，应与 CLI 标志和幂等语义一起单独设计。
- `assets.json` 写入与资产包加锁。清单字段属于加锁切片。
- 取消的残余窗口：一块 ffmpeg（约 0.5 秒）、一帧推理，或 ffmpeg 挂死时的 300 秒超时；发布阶段不可取消。收紧需要跨 crate 的进程组管理。
- GPU 后端。本切片只跑 `CpuBackend`，`Capabilities.wgpu_training` 保持 false。
- FeatherHuBERT 特征提取、训练、渲染、模型导入导出、桌面端与 GPUI。
- 进度指标字段 `samples_per_second`、`eta_seconds`、`vram_bytes`。
