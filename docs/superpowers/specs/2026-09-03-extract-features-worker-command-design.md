# 提取特征 worker 命令与 CLI 子命令设计

日期：2026-09-03
状态：已定稿

## 1. 目标与范围

本切片把四块已经各自可用的代码拼成一条端到端可调用的命令：`feathertalk-audio`（分块规划、长音频拼接、特征文件读写与提交）、`feathertalk-models::feather_hubert`（burn 编码器，已实现 `ChunkEncoder`）、`feathertalk-export`（标准模型包加载）、`feathertalk-worker` 与 `feathertalk-cli`（协议、进度、错误映射、命令行）。领域契约早已提交：`TaskKind::ExtractFeatures`（slug `extract_features`）、`ExtractFeaturesParams { project_dir, audio }`、`Request::ExtractFeatures`、阶段 `ExtractingFeatures`，`render.rs` 的中文标签「正在提取特征」也已就位。

缺口有两处。第一，worker 侧没有实现，也没有工件发现与握手宣告。第二，整个 workspace 没有任何代码能读 WAV：`feathertalk-media` 只调 ffmpeg/ffprobe 处理容器，`feathertalk-audio` 的入口类型是 `&[f32]`，两者之间缺一段解码。规范化步骤产出的 `assets/audio_16k_mono.wav` 因此无法进入特征提取。

本切片范围内的改动：

- `feathertalk-audio`：新增 `wav.rs`（`read_wav_16k_mono`、`MAX_WAV_FILE_BYTES`、`WAV_SAMPLE_RATE`）；`AudioError` 新增 WAV 相关变体与 `Cancelled`。
- `feathertalk-worker`：新增 `features.rs`（模型加载）、`extract_features.rs`（命令编排）、`feature_result.rs`（结果载荷）；`config.rs` 新增 `ENV_HUBERT_DIR` 与 `FeatureToolchain`；`handshake.rs` 与 `runtime.rs` 增加宣告与拒绝文案；`error_map.rs` 新增 `audio_task_error`、`package_task_error`、`is_audio_cancellation`；`Cargo.toml` 新增 `feathertalk-audio`、`feathertalk-export` 依赖。
- `feathertalk-cli`：新增 `extract-features <PROJECT_DIR> <AUDIO>` 子命令、对应的 `build_request` 分支与 `UnsupportedCommand` 提示分支。

一个直接结论：本命令不依赖 ffmpeg/ffprobe（见 §4），只需要 `FEATHERTALK_WORKER_HUBERT_DIR` 一个环境变量。

范围外的内容集中在 §14。

## 2. 令牌契约

链路全部由既有 API 组成，只有第一行是新增的：

```rust
let samples = read_wav_16k_mono(&params.audio)?;
let normalized = normalize_waveform(&samples)?;
let matrix = extract_long_audio(&normalized, &mut encoder, DEFAULT_CHUNK_SAMPLES)?;
let matrix = drop_odd_token(matrix);
let artifact = write_feature_file_no_clobber(&destination, &matrix)?;
```

令牌数完全由音频长度决定：

- `expected_hubert_frames(n) = if n < 400 { 0 } else { (n - 80) / 320 }`，对应 HuBERT 前端的 400 样本核与 320 样本步长。
- `plan_chunks` 以 `DEFAULT_CHUNK_SAMPLES = 320_000`（20.000 秒）为步长切块，每块窗口多读 80 个样本（`end = start + chunk + 80`）补齐边界；尾块从 `chunk * complete` 起，剩余不足 400 样本时丢弃。
- `extract_long_audio` 逐块调 `ChunkEncoder::encode`，拼接后按 `expected_hubert_frames(samples.len())` 补 `0.0` 或截断，因此块边界不影响总长度。
- `drop_odd_token` 在令牌数为奇数时丢掉最后一个，保证可按 `pair_width = 2` 成对读出。

`ExtractFeaturesParams` 里没有帧数，命令也不去读 `quality.json` 反推帧数。理由有两层：第一，令牌契约照搬 Python 的 `get_feather_hubert_from_16k_speech`，那里同样只看音频；第二，推理阶段的音频与工程帧数无关，把帧数耦合进来会让同一条命令在推理路径上不可用。「令牌数 = 2 × 帧数」这条约束属于加锁切片（见 §3）。

一个可对账的例子，2.000 秒 16 kHz 音频：32 000 样本 → `(32000 - 80) / 320 = 99` 个令牌 → `drop_odd_token` → 98 个令牌；dims 为 1024 时文件是 `44 + 98 × 1024 × 4 = 401_452` 字节。

写入用 `write_feature_file_no_clobber`，目标已存在即失败，不提供 `force`：`ExtractFeaturesParams` 带 `deny_unknown_fields`，加标志是线协议变更。

## 3. 为什么不写清单、不加锁

`commit_feature_artifact` 需要一个 `FeatureCommitSpec`，其字段包含 `tokens == 2 * frame_count`、`frame_width`、`frame_height`、`landmark_model_sha256`、`feature_model_sha256`。前三项与关键点模型哈希都无法从 `{ project_dir, audio }` 推出：`QualityReport` 只有 `schema_version`、`frame_count`、`accepted_count`、`frames`、`anomalies`，既没有几何尺寸，也没有关键点模型标识。硬造一个几何尺寸写进资产清单，等于把错误数据固化到加锁产物里。

因此本命令只写 `assets/features/feather_hubert.f32`，不写 `assets.json`，不加锁。这与既有两处判例一致：抽帧设计 §16 把「`assets.json` 写入与资产包加锁」列为范围外；worker 运行时设计 §114 同样保留 `LockAssetPackage` 未启用。

被否决的替代方案：在本切片顺带调用 `commit_feature_artifact`，用 ffprobe 补几何尺寸、用工件目录哈希补关键点模型标识。否决理由是它把两个独立职责绑死——特征提取会因为「读不到关键点模型」而失败，而这跟音频编码毫无关系；而且刚补出的几何尺寸未必与抽帧时真正写入的帧一致。

给加锁切片留的接口是现成的：`read_feature_file` 读回矩阵 → 按 `2 * frame_count` 补齐或截断 → `commit_feature_artifact`。另外 `write_feature_file_no_clobber` 会对目标父目录做 `create_dir_all`，这正好满足 `commit_feature_artifact` 要求 `assets` 与 `assets/features` 都已存在（`validate_real_directory`）的前置条件。

## 4. WAV 读取器

新增 `crates/feathertalk-audio/src/wav.rs`：

```rust
pub const MAX_WAV_FILE_BYTES: u64 = 256 * 1024 * 1024;
pub const WAV_SAMPLE_RATE: u32 = 16_000;

pub fn read_wav_16k_mono(path: impl AsRef<Path>) -> Result<Vec<f32>, AudioError>;
```

严格而不宽容：只接受规范化步骤自己产出的那一种文件。校验顺序——

1. `symlink_metadata` 拒绝非常规文件（与 `read_feature_file` 一致）；字节数超过 `MAX_WAV_FILE_BYTES` 直接拒绝；随后整文件 `fs::read`。
2. `RIFF` 魔数与 `WAVE` 类型。
3. 遍历 chunk：4 字节 id 加 4 字节长度，奇数长度按 WAV 规范跳过一字节补位；未知 chunk（`LIST`、`fact` 等）跳过。
4. `fmt ` 长度至少 16，超出部分（`cbSize` 扩展）忽略；要求 `audio_format == 1`（PCM）、`channels == 1`、`sample_rate == 16_000`、`bits_per_sample == 16`，并校验 `block_align == 2`、`byte_rate == 32_000` 自洽。
5. `data` 必须出现在 `fmt ` 之后；长度非零且为偶数；声明长度超过剩余字节视为截断。

采样值按 `i16 as f32 / 32768.0` 缩放，与 Python 侧 `soundfile` 的定标一致。这个常数对最终输出其实不敏感——下一步 `normalize_waveform` 会做零均值单位方差归一化，任何正的统一缩放都会被消掉；选它只是为了让中间值能与 Python 逐位对照。

否决的两个替代方案：

- 用 ffprobe 加 ffmpeg 解码到 stdout。`feathertalk-media` 的进程封装有输出上限，而 f32 裸流是每分钟 3.84 MB（16000 × 4 × 60），长音频必然撞上限；而且会让本命令凭空多依赖两个外部工具。
- 引入 `hound` 之类的 WAV crate。要处理的只是自家产出的单一格式，读取逻辑不到 200 行，不值得增加一条供应链依赖。

`AudioError` 新增变体，粒度沿用该文件既有的特征文件系列风格：

| 变体 | 触发条件 |
| --- | --- |
| `WavIo { operation, path, source }` | 取元数据或读文件失败 |
| `WavNotRegular { path }` | 非常规文件 |
| `WavTooLarge { limit, actual }` | 超出字节上限 |
| `InvalidRiffHeader` | `RIFF`/`WAVE` 不匹配 |
| `InvalidWavHeader { reason }` | chunk 结构非法、`fmt ` 过短、`fmt ` 晚于 `data`、`block_align`/`byte_rate` 不自洽 |
| `MissingWavChunk { chunk }` | 缺 `fmt ` 或 `data` |
| `UnsupportedWavFormat { code }` | 非 PCM |
| `UnsupportedWavChannels { actual }` | 非单声道 |
| `UnsupportedWavSampleRate { actual, expected }` | 非 16 kHz |
| `UnsupportedWavBitDepth { actual }` | 非 16 位 |
| `WavPayloadTruncated { expected, actual }` | `data` 声明长度大于实际 |
| `EmptyWav` | `data` 为空 |
| `Cancelled { operation }` | 协作式取消（见 §7） |

`Cancelled` 与 WAV 无关，放在同一批新增里是因为 §7 的进度装饰器需要一个能从 `ChunkEncoder::encode` 返回的取消信号。照 `PipelineError::Cancelled` 的形状把取消做成一等变体，比借用 `CommitFailed` 之类的变体夹带语义更清楚，也让 `is_audio_cancellation` 成为一次简单匹配。

## 5. 配置与握手

`config.rs` 新增：

```rust
pub const ENV_HUBERT_DIR: &str = "FEATHERTALK_WORKER_HUBERT_DIR";

pub struct FeatureToolchain {
    hubert_dir: PathBuf,
}
```

`WorkerConfig` 新增 `features: Option<FeatureToolchain>` 与 `feature_rejection: Option<String>`，读取器 `features()`、`feature_rejection()`。路径校验复用 `required_path`（非空且绝对，启动时不碰文件系统），与 `ModelToolchain` 的处理一致。

特征工具链独立解析。`ModelToolchain` 要求 scrfd 与 pfld 同时存在（缺一就评估不了帧），而 FeatherHuBERT 与它们没有任何关系；只配了 HuBERT 的 worker 应该照样能宣告 `extract_features`。

构造函数加一个兄弟函数，而不是改既有签名：

```rust
pub fn from_values_with_toolchains(ffprobe, ffmpeg, timeout_ms, scrfd_dir, pfld_dir, hubert_dir) -> Self;
pub fn from_values_with_models(ffprobe, ffmpeg, timeout_ms, scrfd_dir, pfld_dir) -> Self; // 委托，hubert 传 None
```

保留五参数版本让既有测试与调用点零改动；`from_env` 读取新环境变量。

握手：`supported_commands` 在 `config.features().is_some()` 时追加 `TaskKind::ExtractFeatures`。这段判断放在 `config.media().is_some()` 分支之外——特征提取不需要媒体工具链，嵌进去会让「只配 HuBERT」的 worker 什么都不宣告。追加位置在末尾，既有命令的顺序不变。

`Capabilities` 结构不动，理由同抽帧设计 §11：`supported_commands` 已经承载了这个信息，加字段是协议变更。

`runtime::unsupported_reason` 新增 `TaskKind::ExtractFeatures => feature_reason(slug, config)`，形状照既有 `model_reason`：有 `feature_rejection()` 就用它，否则给出「未配置 `FEATHERTALK_WORKER_HUBERT_DIR`」的说明。

## 6. 模型加载

新增 `crates/feathertalk-worker/src/features.rs`，与 `models.rs` 平级：

```rust
pub struct FeatureModel {
    encoder: BurnFeatherHubertEncoder<CpuBackend>,
    model_sha256: String,
}

impl FeatureModel {
    pub fn load(toolchain: &FeatureToolchain) -> Result<Self, PackageError>;
    pub fn into_parts(self) -> (BurnFeatherHubertEncoder<CpuBackend>, String);
}
```

加载分两步，因为 `load_model_package` 要求调用方先给出期望描述和模型工厂：

1. worker 自己读 `<hubert_dir>/manifest.json`：`symlink_metadata` 加字节上限 `MAX_MANIFEST_BYTES`（64 KiB）加 `serde_json::from_slice::<ModelPackageManifest>`。`feathertalk-export` 没有公开的清单读取函数，这段只能写在 worker 侧。
2. 从 `manifest.configuration` 取出 `ModelConfiguration::FeatherHubert { channels, expansion, num_blocks, output_dim, dropout }` 构造 `FeatherHubertConfig`，调 `load_model_package::<CpuBackend, FeatherHubertEncoder<CpuBackend>, _>(dir, &manifest.description(), &device, |d| config.init(d))`，再 `BurnFeatherHubertEncoder::from_model(model, &device)`。

`manifest.model.sha256` 保留下来用于结果载荷（见 §9）。

否决：直接用 `FeatherHubertConfig::default()`（512/2/12/1024/0.05）。用户手上的检查点是 256/2/8/1024/0.0，默认配置会因张量形状不匹配而加载失败。配置必须来自清单。

两次读清单不是重复劳动：`load_model_package` 会自己重读并重新校验，还会拒绝 `manifest.description() != *expected`；worker 的第一次读取是为了知道该构造什么形状的模型。

若加载期出现栈溢出，沿用 `models.rs` 的 `PREDICTOR_LOAD_STACK_BYTES = 64 * 1024 * 1024` 大栈线程模式。FeatherHuBERT 是 8 到 12 个残差块的浅结构，预期不需要。

## 7. 阶段、进度与取消

`extract_long_audio` 没有观察者接缝，而为一个调用方给 `feathertalk-audio` 加一套公共观察者 API 收益不足。改为在 worker 内部套一层装饰器：

```rust
struct ChunkProgress<'a, E: ChunkEncoder> {
    inner: E,
    reporter: &'a dyn TaskReporter,
    token: &'a CancellationToken,
    total: u64,
    completed: u64,
}
```

`encode` 先查 `token.is_cancelled()`，已取消则返回 `AudioError::Cancelled { operation: "extract_features" }`；否则委托给 `inner.encode`，成功后 `completed += 1` 并上报 `TaskStage::ExtractingFeatures` 与 `Progress { completed, total: Some(total) }`。

`total` 由命令自己调 `plan_chunks(samples.len(), DEFAULT_CHUNK_SAMPLES)` 得到（同一次调用也用于 §8 的字节预算预检）。`extract_long_audio` 内部会再规划一次，那是个纯函数，重复调用的代价可以忽略。

进度粒度是一块，约 20.005 秒音频（320 000 样本步长加 80 样本对齐窗口）。块大小不可配置，与 Python 侧的 `320 * 1000` 保持一致。

| 来源 | 上报阶段 | completed / total |
| --- | --- | --- |
| 模型加载、WAV 读取、归一化 | 不上报 | 无 |
| 每块编码完成 | `ExtractingFeatures` | `completed / chunk_count` |
| 补齐、丢奇数令牌、写文件 | 不上报 | 无 |

命令体开头上报一次 `TaskStage::Preparing`（无进度）。模型加载发生在更早的 `commands.rs` 里，不上报——与 `extract_frames` 相同的已知空档。

`samples_per_second`、`eta_seconds`、`vram_bytes` 留空，理由同抽帧设计 §9。

## 8. 准入检查

失败全部映射到 `MEDIA_INVALID`，经本文件内的 `invalid_request` 辅助函数加 `error_map::clamp`，照 `extract_frames.rs` 的形状。顺序有意义：便宜的检查在前，昂贵的在后。

1. `check_project_dir`：`project_dir` 必须是绝对路径；`fs::symlink_metadata` 判定为目录；`project.json` 存在且是常规文件。这段从 `extract_frames.rs` 复制——`feathertalk-project` 没有导出该文件名常量（字面量在其 `src/package.rs:66`），所以 `PROJECT_MANIFEST` 在 worker 侧第二次出现。
2. `params.audio` 必须是绝对路径。
3. 目标预检：`<project_dir>/assets/features/feather_hubert.f32` 不得已存在。让 `persist_noclobber` 在几分钟编码之后才报错，给出的中文摘要不如这里明确。常量 `ASSETS_DIR`、`FEATURES_DIR`、`FEATURE_FILE_NAME` 定义在 worker 侧（audio crate 的 `commit.rs:15` 有同名私有常量）。
4. `read_wav_16k_mono`，失败走 `audio_task_error`。
5. 音频过短：要求 `expected_hubert_frames(samples.len()) >= 2`。低于 400 样本时块计划为空，恰好 1 个令牌又会被 `drop_odd_token` 丢掉，两种情况都会产出 0 令牌的文件。摘要「音频太短，无法提取特征」。
6. 字节预算预检：`44 + target_tokens * dims * 4 > MAX_FEATURE_FILE_BYTES` 时在编码之前拒绝。推导：512 MiB 减 44 字节头，除以 1024 维令牌的 4096 字节，得 131 071 个令牌的上限；反解样本数 `131_071 × 320 + 80 = 41_942_800`，即约 43.7 分钟音频。256 MiB 的 WAV 上限对应约 2.33 小时，永远不会先触发——两个上限里 512 MiB 那个才是实际约束。
7. `token.is_cancelled()`：准入通过、编码开始之前查一次。

## 9. 结果载荷

新增 `worker/src/feature_result.rs`，形状照 `quality_result.rs`（`serde_json::json!` 加 `path.display().to_string()`）：

```json
{
  "output_dir": "<project>/assets/features",
  "feature_file": "<project>/assets/features/feather_hubert.f32",
  "tokens": 98,
  "dims": 1024,
  "frame_count": 49,
  "bytes": 401452,
  "sha256": "…",
  "model_sha256": "…"
}
```

`tokens`、`dims`、`bytes`、`sha256` 直接来自 `FeatureArtifact` 的读取器。`frame_count = tokens / 2` 是给加锁切片与 CLI 的换算结果，省得调用方各自复制 `pair_width` 知识。

`model_sha256` 是超出最小集的一项，刻意加入：请求参数里不带模型标识，缺了它就无法从事件流回答「这份特征是哪个权重产出的」，产物不可审计。

不放特征数值本身，理由同抽帧设计 §10 不放逐帧数组：几十万个 f32 会把单行 JSON 事件撑爆，需要数据就读文件。

## 10. 错误映射

`error_map.rs` 新增 `audio_task_error(&AudioError) -> TaskError`、`package_task_error(&PackageError) -> TaskError`、`is_audio_cancellation(&AudioError) -> bool`，并从 `lib.rs` 导出。

`AudioError` 到 `ErrorCode`：

| 变体 | 错误码 |
| --- | --- |
| WAV 系列（除 `WavIo`）、`EmptyWaveform`、`NonFiniteWaveform`、`ConstantWaveform`、`FeatureTooLarge`、`FeatureNotRegular`、特征文件解析系列（`InvalidFeatureMagic`、`UnsupportedFeatureVersion`、`FeatureHeaderTruncated`、`FeaturePayloadTruncated`、`FeatureTrailingBytes`、`InvalidFeaturePayloadSize`、`InvalidFeaturePairWidth`）、`LockedAssetMutation` | `MEDIA_INVALID` |
| `WavIo`、`FeatureIo` | `io_error_code`，沿用现有映射 |
| `InvalidFeatureDimension`、`FeatureLengthMismatch`、`NonFiniteFeature` | `MODEL_INCOMPATIBLE` |
| `FeatureShapeMismatch` | `FEATURE_SHAPE_MISMATCH` |
| `InvalidChunkSize`、`ArithmeticOverflow`、`TooManyChunks`、`FeatureSizeOverflow`、`CommitFailed`、`CommitRollbackFailed`、`StagingCollision` | `WORKER_CRASHED` |
| `Cancelled` | `TASK_CANCELLED` |

三行需要解释。维度与长度类错误说明权重产出的形状不对，恢复动作 `ReimportModel` 比重试有用，所以走 `MODEL_INCOMPATIBLE`。`FeatureShapeMismatch` 两侧名字正好对齐，它由提交校验产生，在本命令路径上不可达，列出只为 match 完备。`Cancelled` 同样是完备性兜底——运行时会先拦下取消并产生 `Cancelled` 结局，沿用该文件既有注释风格标注这一点。

`PackageError` 的所有变体（`InvalidRequest`、`InvalidManifest`、`InvalidLicense`、`Io`、`HashMismatch`、`Store`、`WeightImport`、`Publication`）统一映射到 `MODEL_INCOMPATIBLE`，摘要「特征模型加载失败」，`technical_detail` 里点名 `FEATHERTALK_WORKER_HUBERT_DIR`。把 `Io` 也归到这里是刻意的：加载只读不写，让它走 `io_error_code` 会对「模型目录不存在」这种最常见的情况回答 `WORKER_CRASHED` 加 `ResumeFromCheckpoint`，那不是可执行的建议。

所有失败沿用 `FAILURE_STAGE = TaskStage::Preparing`。

## 11. 命令签名与 CLI 形态

worker 侧入口：

```rust
pub fn execute_extract_features<E: ChunkEncoder>(
    params: &ExtractFeaturesParams,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    encoder: E,
    model_sha256: &str,
) -> CommandOutcome;
```

不像 `extract_frames`，这里不收 `&WorkerConfig`：工具链已经在调用方被 `FeatureModel::load` 完全消费掉了，再传一遍只会给测试增加构造负担。泛型参数 `E` 是测试接缝——单元测试传假编码器，不加载任何权重。`commands.rs` 的分支先取 `config.features()`，缺失则 `CommandOutcome::Failed(unsupported(request.kind()))`；`FeatureModel::load` 失败则 `CommandOutcome::Failed(package_task_error(&error))`；成功则 `into_parts()` 后调用上面的函数。

CLI：

```
feathertalk extract-features <PROJECT_DIR> <AUDIO>
```

`Command` 枚举在 `ExtractFrames` 之后新增：

```rust
/// 提取音频的 FeatherHuBERT 特征
ExtractFeatures {
    /// 工程目录
    project_dir: PathBuf,
    /// 已归一化的 16kHz 单声道音频，位于工程目录的 assets 下
    audio: PathBuf,
},
```

`run.rs::build_request` 增加分支，用 `reject_empty(project_dir, "工程目录")` 与 `reject_empty(audio, "音频文件")`，单元测试加在该文件末尾的 `mod tests`。`cli.rs` 里列举 kebab 命令的文档注释、以及 `lib.rs` 与相关模块文档中枚举「已服务命令」的地方一并更新。

`render.rs` 新增 `const ENV_WORKER_HUBERT_DIR: &str = "FEATHERTALK_WORKER_HUBERT_DIR";`，沿用该文件「把 worker 侧常量以字面量复制一份并注明 worker 常量是唯一来源」的约定；`render_client_error` 的 `UnsupportedCommand` 分支加 `else if *requested == "extract_features"`，只提示这一个变量。阶段中文标签已存在，不改。

## 12. 测试

组合逻辑用便宜的假编码器覆盖，只保留一个门控的真实端到端测试。

`feathertalk-audio`（新增 `tests/wav.rs`，用本地 WAV 写入辅助函数造夹具）：

- 往返：写 16 kHz 单声道 16 位 PCM，读回样本数与定标；
- 格式拒绝：44.1 kHz、双声道、24 位、非 PCM 格式码；
- 结构拒绝：缺 `fmt `、缺 `data`、`fmt ` 晚于 `data`、声明长度超出实际、`data` 为空、`data` 长度为奇数；
- 兼容：未知 chunk（`LIST`）被跳过，奇数长度 chunk 的补位字节被正确跳过；
- 非常规文件与符号链接拒绝；
- `MAX_WAV_FILE_BYTES` 用常量断言，不真的造 256 MiB 文件。

`feathertalk-worker`（新增 `tests/extract_features.rs`、`tests/feature_result.rs`）：

- 成功路径：假编码器加临时工程目录，断言输出文件存在、令牌数为偶数、结果载荷各字段；
- 准入七项各一个失败用例；
- 进度：假 reporter 记录 `preparing` → `extracting_features 1/N … N/N`；
- 取消：token 在第二块置位，断言 `CommandOutcome::Cancelled` 且目标文件未生成；
- 扩展 `tests/config.rs`（新环境变量与新构造函数）、`tests/handshake.rs`（仅 HuBERT、仅媒体、全配三种组合）、`tests/error_mapping.rs`（`AudioError` 与 `PackageError` 全变体）、`tests/runtime.rs`（拒绝文案）、`tests/commands.rs`（未配置工具链时返回 `Failed`）。

`feathertalk-cli`：`tests/cli.rs` 与 `run.rs` 内联测试覆盖空参数拒绝与请求构造。

端到端（门控，`--release`）：加在 `tests/real_worker.rs`，沿用现成的 `REQUIRE_E2E` 加 `worker_or_skip` 加 `real_tool("FFMPEG")` 加 `real_dir("HUBERT_DIR")` 组合；环境变量必须给绝对路径，否则辅助函数会静默跳过。必须跑 release：debug 下的 burn 慢约三个数量级，而 `CARGO_BIN_EXE_*` 是按 profile 解析的。

音频片段用真实 ffmpeg 从 `demo/feathertalk_demo_latest_188.mp4` 裁 2 秒（`-vn -ac 1 -ar 16000 -c:a pcm_s16le -t 2`），预期 32 000 样本 → 99 → 98 个令牌、dims 1024、`44 + 98 × 1024 × 4 = 401_452` 字节。推导写进测试注释，便于在重采样器行为不同时重算。

模型包由 `cargo run -p feathertalk-model-package -- feather-hubert` 从 `demo/kanghui_training_video_featherhubert_188_latest/feather_hubert_188_latest_99.pth`（40 436 613 字节，配置 256/2/8/1024/0.0）现场构建，许可清单可用本地合成文件（判例：`rust/tests/fixtures/vgg19/LICENSES.local-parity.json` 与 `crates/feathertalk-export/tests/feather_hubert_real.rs` 的 `LicenseRef-User-Supplied-Unreviewed`）。成品目录只含 `LICENSES.json`、`manifest.json`、`model.safetensors`。

不新增二进制夹具入库，理由同抽帧设计 §14。

## 13. 验证

在 `rust/` 下执行，要求零告警零失败：

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace --all-targets`
- `cargo test --release -p feathertalk-cli --test real_worker`（门控变量齐备时）
- `git diff --check`

## 14. 范围外

- `assets.json` 写入与资产包加锁，以及「令牌数 = 2 × 帧数」的强制。见 §3，接口已备好。
- 重跑与覆盖（`force`、部分续跑）。同抽帧设计，属线协议变更，应与幂等语义一起单独设计。
- 非 16 kHz、非单声道、非 16 位 WAV 的自动重采样与混音。规范化步骤已经保证格式，读取器就该严格。
- GPU 后端。本切片只跑 `CpuBackend`，`Capabilities.wgpu_training` 保持 false。
- 取消的残余窗口：一块编码（CPU 上数秒）不可中断；写文件阶段不可取消。
- 训练、渲染、模型导入导出的 UI、桌面端与 GPUI。
- 进度指标字段 `samples_per_second`、`eta_seconds`、`vram_bytes`。
