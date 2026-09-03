# 加锁素材包 worker 命令与 CLI 子命令设计

日期：2026-09-03
状态：已定稿

## 1. 目标与范围

本切片实现迁移设计 §7.1 的第 9 步：在完整清单与哈希写入之后原子地提交素材包。这是里程碑二的最后一块——归一化、抽帧、提取特征三条命令都已落地，但它们各自只写自己的产物，`assets/assets.json` 至今没有任何代码路径会写出，`AssetManifest::state` 也从未变成 `Locked`。领域契约同样早已提交：`TaskKind::LockAssetPackage`（slug `lock_asset_package`，已在 `TaskKind::ALL` 内）、`Request::LockAssetPackage(ProjectDirParams)`。

缺口分布在四层。最上层是 worker 没有实现、握手不宣告；下面三层是提交所需的事实无处可取：整个 workspace 没有代码能只读 JPEG 头拿到帧尺寸（`decode_jpeg` 会把整帧解成 BGR），没有代码能读回 `.lms` 关键点文件（只有写出的 `serialize_landmarks`），也没有函数能把已有的特征矩阵对齐到指定令牌数（`extract_long_audio` 内部做了这件事，但没有暴露）。

本切片范围内的改动：

- `feathertalk-image`：`jpeg.rs` 新增 `jpeg_dimensions`，并把 `decode_jpeg` 的头部解析抽成私有 `read_header`，避免两条路径漂移。
- `feathertalk-frame-adapters`：新增 `probe_jpeg_geometry`，把 `ImageError` 桥接到 `PipelineError`。
- `feathertalk-frame-pipeline`：新增 `LANDMARK_POINTS`、`MAX_LANDMARK_FILE_BYTES`、`read_landmark_file`；`PipelineError` 新增 `FrameUndecodable`、`LandmarkNotRegular`、`InvalidLandmark` 三个变体；`serialize_landmarks` 改用新常量。
- `feathertalk-audio`：`stitch.rs` 新增 `fit_feature_tokens`，与 `extract_long_audio` 共用私有的 `fit_values`；`FeatureMatrix` 新增 `pub(crate) fn into_values`。
- `feathertalk-worker`：新增 `lock_asset_package.rs`（命令编排）、`lock_result.rs`（结果载荷）；`commands.rs`、`handshake.rs`、`runtime.rs`、`error_map.rs`、`lib.rs` 增加分支与导出；`Cargo.toml` 把 `feathertalk-pfld` 从 dev 依赖提为正式依赖。
- `feathertalk-cli`：新增 `lock-asset-package <PROJECT_DIR>` 子命令、`build_request` 分支与 `UnsupportedCommand` 提示分支。

`feathertalk-pfld` 的提升不增加构建图成本：它已经通过 `feathertalk-frame-adapters` 传递进来了，本切片只是把它变成直接依赖以便引用一个常量。worker 不新增 `feathertalk-image` 依赖，JPEG 能力一律经 adapters 转手，与既有分层一致。

一个直接结论：本命令既不需要 ffmpeg/ffprobe，也不需要 SCRFD 与 PFLD 权重目录，只需要 `FEATHERTALK_WORKER_HUBERT_DIR` 一个环境变量（理由见 §8 与 §9）。

迁移设计 §7.1 的第 7 步（异常帧的人工编辑）属于桌面端 UI，不在本切片内；但它的存在直接决定了 §7 的校验策略。范围外的内容集中在 §17。

## 2. 清单字段的来源

`FeatureCommitSpec` 的五个字段加上矩阵令牌数，就是本切片全部的信息收集工作：

| 字段 | 来源 |
| --- | --- |
| `frame_count` | `assets/quality.json`，经 `read_quality_report` |
| `frame_width` / `frame_height` | 逐帧解析 JPEG 的 SOF 头，要求全部一致 |
| `landmark_model_sha256` | 常量 `feathertalk_pfld::PFLD_MODEL_SHA256` |
| `feature_model_sha256` | `feathertalk_export::read_package_manifest(hubert_dir)` 的 `manifest.model.sha256` |
| 矩阵令牌数 | `read_feature_file` 读回后经 `fit_feature_tokens(2 * frame_count)` 对齐 |

其余字段不需要收集：`locked_manifest`（`audio/src/commit.rs:246`）把 schema 版本 1、状态 `Locked`、25 fps、16 000 Hz、单声道、特征类型 FeatherHubert、形状 `[frame_count, 2, 1024]` 全部写死，与 `AssetManifest::validate_locked`（`project/src/model.rs:145`）的期望一一对应。本命令不重复这些常量。

几何尺寸只能从帧文件本身取。`QualityReport` 有 `schema_version`、`frame_count`、`accepted_count`、`frames`、`anomalies`，其中 `FrameQuality` 记录了文件名、字节数、两个哈希、人脸分数、bbox 与模糊方差——没有一处是帧的像素尺寸。bbox 是检测框，不是画幅。所以必须读文件，这就是 §4 存在的原因。

## 3. 复用提交协议

`commit_feature_artifact`（`audio/src/commit.rs:29`）已经把提交做完了：校验规格、预检清单、写入 `assets/.feathertalk-feature-build-*` 暂存目录、备份特征文件与清单两个目标、调 `feathertalk_project::lock_asset_package` 写出 `assets.json`（内部再跑 `validate_locked` 与 `validate_artifacts`）、失败时回滚。本命令的职责因此收缩为「把 `FeatureCommitSpec` 凑齐并保证前置条件成立」，然后调用它一次。

被否决的替代方案：当磁盘上的令牌数已经等于 `2 * frame_count` 时，跳过特征文件重写，直接调用 `feathertalk_project::lock_asset_package` 只写清单。否决理由是它把一条命令变成两条代码路径，而省下的只是重写约 1.5 MB（188 帧 × 2 × 1024 × 4）文件的 I/O；代价则是丢掉 `commit_feature_artifact` 的原子安装与回滚——只写清单的路径一旦在中途失败，工程会留下一个「清单已加锁但特征文件是旧的」的状态。始终走同一条路径，成本可忽略，且失败语义只需推理一次。

`commit_feature_artifact` 要求 `assets` 与 `assets/features` 都已存在（`validate_real_directory`），这由抽帧与提取特征两条命令保证：`publish_frame_artifacts` 写入 `output_root = project_dir/assets`，`write_feature_file_no_clobber` 对目标父目录做 `create_dir_all`。提取特征设计 §3 明确把这个接口留给了本切片。

## 4. 帧几何：JPEG 头解析

`feathertalk-image` 新增：

```rust
pub fn jpeg_dimensions(bytes: &[u8]) -> Result<(u32, u32), ImageError>;
```

实现方式是把 `decode_jpeg` 的前半段——`read_info` 加 `info()` 加 u16 转 u32 加零值拒绝——抽成一个接受解码器的私有函数：

```rust
fn read_header<R: Read>(decoder: &mut Decoder<R>) -> Result<(u32, u32, PixelFormat), ImageError>;
```

它收 `&mut Decoder` 而不是字节切片，因为 `decode_jpeg` 之后还要在同一个解码器上调 `set_max_decoding_buffer_size` 与 `decode()`；返回 `PixelFormat` 是同样的原因。`jpeg_dimensions` 自己 `Decoder::new(Cursor::new(bytes))` 后调它，丢掉像素格式。`jpeg_dimensions` 不接 `max_pixels`：它不分配任何像素缓冲，尺寸上限该由调用方按自己的语义判断（本命令的上限是 `MAX_FRAME_DIMENSION`，由 `AssetManifest` 校验负责）。抽取而非复制，是因为两处若各自解析头部，`InvalidDimensions` 的判定条件迟早会不一致。

`feathertalk-frame-adapters` 新增：

```rust
pub fn probe_jpeg_geometry(path: &Path, bytes: &[u8]) -> Result<(u32, u32), PipelineError>;
```

任何 `ImageError` 都映射到新的 `PipelineError::FrameUndecodable { path, message }`。路径要作为参数传进来，是因为错误里必须点名是哪一帧坏了，而字节切片不携带来源。这种「image 层错误翻译成 pipeline 层错误」的桥接在本 crate 已有判例：`cache.rs::load` 与 `decoder.rs` 都把 `ImageError` 转成 `PipelineError`。区别是它们转成 `Adapter { component: "jpeg" }`，而 `Adapter` 在 worker 的映射里是 `ModelIncompatible` 加「模型推理失败」——对一张坏掉的素材帧给出「重新导入模型」的建议是误导，这正是要新增 `FrameUndecodable` 的原因（见 §13）。

`probe_jpeg_geometry` 对字节是纯函数，不碰文件系统；文件读取由调用方（`lock_asset_package.rs`）负责，并且是有界的。常量定义在 worker 侧：

```rust
const JPEG_HEADER_PROBE_BYTES: u64 = 64 * 1024;
```

先读前 64 KiB 交给它解析。SOF 段位于所有扫描数据之前，正常 JPEG 里离文件头只有几百字节，64 KiB 已经覆盖了带大段 EXIF 缩略图的情况。若前缀解析失败而文件本身更大，则整文件重读一次（此时它已经通过了 `MAX_FRAME_BYTES` 检查，最多 16 MiB）再报错——这样错误消息描述的永远是文件的真实问题，而不是我们自己截断出来的假象。把读取留在 worker 侧，也让 adapters 的新函数在单元测试里不需要临时目录。

## 5. 关键点文件校验

`feathertalk-frame-pipeline` 新增：

```rust
pub const LANDMARK_POINTS: usize = 110;
pub const MAX_LANDMARK_FILE_BYTES: u64 = 8 * 1024;

pub fn read_landmark_file(
    path: &Path,
    frame_width: u32,
    frame_height: u32,
) -> Result<Vec<(i32, i32)>, PipelineError>;
```

格式与写出端严格对称。`serialize_landmarks`（`evaluate.rs:410`）写的是 110 行 `"{x} {y}\n"`，并保证 `0 <= x < width`、`0 <= y < height`。读取端因此要求：恰好 `LANDMARK_POINTS` 行；每行两个以单个空格分隔的十进制整数；坐标落在给定几何内。多余空格、CRLF 行尾、缺末尾换行、非整数、负数、越界、行数不等于 110，全部拒绝为 `InvalidLandmark { path, message }`；非常规文件与符号链接拒绝为 `LandmarkNotRegular { path }`；超过 `MAX_LANDMARK_FILE_BYTES` 也归入 `InvalidLandmark`（消息点明字节上限）。

CRLF 被拒绝是有意的：写出端只写 `\n`，一个带 CRLF 的 `.lms` 说明文件被文本工具处理过，此时坐标是否还可信已无法判断，宁可让加锁失败。上限的推导：坐标最大 32 767（`MAX_FRAME_DIMENSION`），单行最长 `"32767 32767\n"` 共 12 字节，110 行合计 1 320 字节；8 KiB 留了六倍余量，同时把「有人把整张图塞进 `.lms`」挡在读取之前。

`serialize_landmarks` 里两处硬编码的 `110` 改用 `LANDMARK_POINTS`。这不是顺手清理：读写两端共用一个常量，是「格式对称」这条性质唯一可靠的保证方式。

本命令不重跑 PFLD、不比对坐标是否合理，只确认文件在结构上可用。理由见 §7。

## 6. 特征令牌对齐

`feathertalk-audio` 新增：

```rust
pub fn fit_feature_tokens(matrix: FeatureMatrix, tokens: usize) -> Result<FeatureMatrix, AudioError>;
```

语义与 `extract_long_audio` 结尾那几行完全相同——不足补 `0.0`，超出截断——所以把它抽成私有的 `fit_values(&mut Vec<f32>, target_values)`，两处共用。同时给 `FeatureMatrix` 加 `pub(crate) fn into_values(self) -> Vec<f32>`，让 `fit_feature_tokens` 能拿走底层缓冲原地改，避免多复制一份约 1.5 MB 的载荷。`FeatureMatrix::new` 会重新扫描有限性，这层校验保留。

策略留在 worker 侧，机制留在库里。库函数无条件对齐；worker 在调用之前要求

```rust
const MAX_TOKEN_FIT_DELTA: i64 = 50;
```

即 `|磁盘令牌数 - 2 * frame_count| <= MAX_TOKEN_FIT_DELTA`，否则拒绝为 `invalid_request`。50 个令牌是 1 秒（25 fps 下 1 帧等于 2 个令牌）。同一次规范化产出的音视频只差容器时长的舍入与 HuBERT 前端 400 样本核带来的边界损耗，合计不到 4 个令牌；50 留了一个数量级的余量，同时能挡住「拿另一段音频的特征来加锁」这类真正的错配。

一个可对账的例子：188 帧目标 376 个令牌；7.52 秒 wav 是 120 320 个样本，`expected_hubert_frames` 取整除得 `(120320 - 80) / 320 = 375`（120 240 除以 320 商 375 余 240），`drop_odd_token` 后 374，补 2 个零向量。带符号的差值以 `token_adjustment` 上报（见 §12），让调用方能看见发生了对齐而不是猜。

不声称与 Python 侧对齐：Python 实现里帧数与令牌数的关系由训练脚本各自处理，没有一个可引用的规范化差值。这里的容差是本实现自己的约定。

## 7. 校验遍历

指导原则一句话：加锁校验的是素材包**结构上自洽且可用**，不是与抽帧时刻**逐字节相同**。

按 `report.frames()` 给出的索引逐帧检查：

1. 帧文件存在、是常规文件且非符号链接、非空、不超过 `MAX_FRAME_BYTES`（16 MiB，`process.rs:13`）——直接复用既有的 `FrameMissing`、`FrameNotRegular`、`FrameEmpty`、`FrameTooLarge` 四个变体。
2. `probe_jpeg_geometry` 取几何。第一帧的几何是基准，之后任何不一致都拒绝为 `invalid_request`，摘要「素材帧尺寸不一致」，详情同时点名基准索引与冲突索引。
3. `read_landmark_file` 用该几何做边界检查。

随后对 `assets/frames` 与 `assets/landmarks` 各做一次 `read_dir`：文件名匹配 `^\d{6}\.jpg$` 与 `^\d{6}\.lms$` 的条目数必须恰好等于 `frame_count`。这一步抓的是清单查不到的情况——上一次抽帧留下的多余帧文件。不匹配命名规则的条目一律忽略，这样 `desktop.ini` 或编辑器备份文件不会让加锁失败。

**不重新计算 SHA-256，也不比对 `frame_bytes`。** 迁移设计 §7.1 的第 7 步明确允许用户在 `quality.json` 写出之后编辑异常帧，强制字节一致等于把一个自相矛盾的约束写进设计：要么加锁必然失败，要么人工修帧这个功能不能存在。`quality.json` 里的哈希是抽帧时刻的证据，不是加锁时刻的不变量。顺带的好处是加锁不必读满 188 个 16 MiB 上限的文件两遍。

被否决的替代方案：逐帧重新哈希并与 `frame_sha256` 比对（理由如上）；只比 `frame_bytes` 不比哈希（同样的矛盾，只是更弱的版本，还会给出「字节数变了但内容可能没变」这种无法解释的失败）。

## 8. 两个模型哈希的语义

两个哈希的可信度不同，设计上必须区别对待。

`landmark_model_sha256` 取常量 `feathertalk_pfld::PFLD_MODEL_SHA256`（`"e131dd764236fde54a27b2f7084906119f06c28b140bf127b459ec967e92915b"`）。这不是猜测：`PfldRuntimeManifest::validate()`（`manifest.rs:168`）要求 `self.model.sha256 == PFLD_MODEL_SHA256`，任何一次 PFLD 加载都必须通过这道校验，因此凡是本工程产出的关键点，其模型哈希必然是这个值。用常量比读目录更强——它在没有 PFLD 目录的机器上也成立，而事实不变。

`feature_model_sha256` 取 `read_package_manifest(hubert_dir)` 的 `manifest.model.sha256`，语义是「执行加锁的这个 worker 里装的编码器」。它**不能**证明磁盘上那份 `feather_hubert.f32` 就是这个权重产出的：特征文件格式（44 字节头加 f32 载荷）没有模型标识位。这一点在文档里直说，不粉饰。加锁与提取特征通常在同一台机器同一个配置下先后执行，这个哈希在实践中是对的；要做到可证明，需要线协议层面的改动，属范围外。

被否决的替代方案：写一个特征旁文件记录编码器哈希（新增一个没人校验的元数据文件，还得考虑它与特征文件的原子性）；在 `.f32` 头里加模型哈希槽位（v1 到 v2 会让已有文件全部读不出来）；加锁时重跑 PFLD 以获得可证明的来源（需要 SCRFD 与 PFLD 权重，把一个纯本地校验命令变成需要两套模型的重活，而结论仍然只是「现在这个模型的哈希」）；用 `read_package_manifest(pfld_dir)` 读关键点模型清单——这条会直接失败：PFLD 用的是 `PfldRuntimeManifest`，与 `ModelPackageManifest` 不是同一套 schema，且 `validate_package_directory` 要求目录恰好含 `LICENSES.json`、`manifest.json`、`model.safetensors` 三个文件。

## 9. 配置与握手

不新增环境变量。命令需要的唯一外部事实是 HuBERT 包的清单，`WorkerConfig::features()` 已经指向它。

`handshake.rs::supported_commands` 在既有的 `config.features().is_some()` 分支里追加 `TaskKind::LockAssetPackage`——与 `ExtractFeatures` 同一个条件，因为二者依赖同一份配置。`runtime.rs::unsupported_reason`（行 389）新增 `TaskKind::LockAssetPackage => feature_reason(slug, config)`，复用现成的拒绝文案。

被否决的替代方案：让本命令无条件可用（`feature_model_sha256` 无处可取，只能写空串或假值，等于把坏数据固化进加锁产物）；给 `Capabilities` 加字段说明「加锁可用但哈希不可证」（线协议变更，收益是一句免责声明）。

## 10. 阶段、进度与取消

阶段用 `TaskStage::Preparing`，不新增阶段变体。现有 13 个变体（`Queued`、`Preparing`、`ExtractingAudio`、`ExtractingFrames`、`DetectingFaces`、`ExtractingFeatures`、`Training`、`Importing`、`Exporting`、`Rendering`、`Completed`、`Failed`、`Cancelled`）是线协议的一部分，为一条本地校验命令加一个 `Locking` 需要同步桌面端与中文标签，收益是一个更精确的名字。

进度按帧上报：`Progress { completed, total: Some(frame_count) }`，每完成一帧的检查递增一次，不做节流——判例是 `extraction.rs:152` 与 `evaluate.rs:178`，两者都是逐帧上报。校验遍历是本命令唯一耗时可观的部分（188 次 JPEG 头解析加 188 次小文件读取），进度覆盖它就够了。

取消在每帧检查之前查一次 `token.is_cancelled()`，命中则返回 `CommandOutcome::Cancelled`，此时尚未有任何写入。提交本身不可取消：`commit_feature_artifact` 内部有暂存、备份、回滚三段，从中间打断只会制造它自己要处理的残局。

| 来源 | 上报阶段 | completed / total |
| --- | --- | --- |
| 准入检查、读清单、读特征文件 | 不上报 | 无 |
| 每帧校验完成 | `Preparing` | `completed / frame_count` |
| 令牌对齐、目录扫描、提交 | 不上报 | 无 |

## 11. 准入检查

顺序按「便宜且用户能自己修的排前面」排列，形状照 `extract_frames.rs`：

1. `check_project_dir`：绝对路径、`symlink_metadata` 判定为目录、`project.json` 存在且为常规文件。
2. 若 `assets/assets.json` 已存在，先自己调 `read_asset_manifest` 预检：状态为 `Locked` 则拒绝，摘要「素材包已加锁」；解析失败走 `project_task_error`。这一步是刻意前置的——`commit_feature_artifact` 的 `preflight_manifest`（`commit.rs:224`）会把不可解析的清单报成 `CommitFailed { operation: "read_manifest" }`，进而映射成 `WORKER_CRASHED`，而真实原因是工程文件坏了，用户该去修文件而不是重试。
3. `read_quality_report(project_dir/assets/quality.json)`：抽帧命令写在这里（`FramePipelineSpec::quality_report_path()` 是 `output_root.join("quality.json")`，而 `output_root` 就是 `project_dir/assets`）。函数自带符号链接与常规文件检查、`MAX_REPORT_BYTES` 上限与 `validate()`。
4. 要求 `anomalies.is_empty()` 且 `accepted_count == frame_count`。`QualityReport::validate` **不**检查这两条，只有 `publish_frame_artifacts` 在发布时检查过；加锁必须自己判，否则会把一个含异常帧的素材包锁死。摘要分别为「素材包仍有异常帧」与「仍有帧未被接受」。
5. `REQUIRED_FILES` 的存在性预检：`assets/video_25fps.mp4`、`assets/audio_16k_mono.wav`、`assets/features/feather_hubert.f32`（与 `project/src/package.rs:10` 同一份清单）。缺文件在这里报，比在提交回滚路径里报清楚得多。
6. `read_feature_file` 读回矩阵，失败走 `audio_task_error`。
7. 令牌容差检查，随后 `fit_feature_tokens(2 * frame_count)`（见 §6）。
8. §7 的校验遍历。
9. `commit_feature_artifact`。

被否决的替代方案：把 `REQUIRED_FILES` 检查交给 `commit_feature_artifact` 内部的 `validate_artifacts`（它确实会查，但那时特征文件已经写进暂存目录，失败要走回滚，日志里出现一次无谓的回滚记录）；改动 `ProjectDirParams` 增加 `force` 或 `frame_count` 之类的字段（线协议变更，且两者都没有需求）。

## 12. 结果载荷

新增 `worker/src/lock_result.rs`，形状照 `feature_result.rs`（`serde_json::json!` 加 `path.display().to_string()`）：

```rust
pub fn lock_to_json(
    project_dir: &Path,
    spec: &FeatureCommitSpec,
    artifact: &FeatureArtifact,
    token_adjustment: i64,
) -> Value;
```

```json
{
  "project_dir": "<project>",
  "manifest_file": "<project>/assets/assets.json",
  "frame_count": 188,
  "frame_width": 512,
  "frame_height": 512,
  "feature_file": "<project>/assets/features/feather_hubert.f32",
  "tokens": 376,
  "dims": 1024,
  "bytes": 1540140,
  "sha256": "…",
  "token_adjustment": 2,
  "landmark_model_sha256": "…",
  "feature_model_sha256": "…"
}
```

`token_adjustment` 是签名值（正为补齐，负为截断，0 为原样），它是本命令唯一会静默修改用户数据的地方，必须在事件流里留痕。两个模型哈希放进载荷，是因为它们已经被写进 `assets.json`，事件流里出现同一份值让审计不必额外读文件。不放帧级明细，理由同抽帧设计 §10：需要逐帧数据就读 `quality.json`。

## 13. 错误映射

三个新增的 `PipelineError` 变体加进 `pipeline_error_code`（`error_map.rs:206`）与 `pipeline_summary`（行 234），两个 match 都是穷尽的，漏一个编译就会失败。错误码统一为 `ErrorCode::MediaInvalid`——三者说的都是「素材文件本身有问题，worker 没坏」。

| 变体 | 错误码 | 中文摘要 |
| --- | --- | --- |
| `FrameUndecodable` | `MEDIA_INVALID` | 素材帧无法解码 |
| `LandmarkNotRegular` | `MEDIA_INVALID` | 关键点文件不可用 |
| `InvalidLandmark` | `MEDIA_INVALID` | 关键点文件不可用 |

两个关键点变体共用一条摘要：对用户而言「这个 `.lms` 用不了」就是全部可执行信息，区别在 `technical_detail` 里说。

其余错误复用现成的辅助函数：`project_task_error`（清单读取）、`audio_task_error`（特征读取与提交）、`pipeline_task_error`（校验遍历）、`invalid_request`（几何不一致、令牌超差、质量报告不满足加锁条件）。`AudioError` 里有两个变体值得点名：`LockedAssetMutation` 与 `FeatureShapeMismatch` 在本路径上不可达——前者被第 2 步准入拦下，后者被 §6 的对齐消除——列出只为完备；`CommitRollbackFailed` 映射到 `WORKER_CRASHED` 是正确的，它意味着工程目录处于需要人工修复的状态。

所有失败沿用 `FAILURE_STAGE = TaskStage::Preparing`，与 §10 的阶段选择一致。

## 14. 命令签名与 CLI 形态

worker 侧入口：

```rust
pub fn execute_lock_asset_package(
    params: &ProjectDirParams,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    feature_model_sha256: &str,
) -> CommandOutcome;
```

哈希作为参数传入，与 `execute_extract_features(params, token, reporter, encoder, model_sha256)` 同构：单元测试因此不需要准备一个 13 MB 的模型包。`commands.rs` 的分支负责解析——取 `config.features()`，缺失则 `CommandOutcome::Failed(unsupported(request.kind()))`；调 `read_package_manifest(features.hubert_dir())`，`PackageError` 走 `package_task_error`；成功则取 `manifest.model.sha256`。

这里刻意不用 `FeatureModel::load`：本命令不做任何推理，把整份 safetensors 权重读进内存只为取一个字符串是纯浪费，而 `read_package_manifest` 已经包含 `validate_package_directory` 与 `manifest.validate()`，该有的校验一样不少。

CLI：

```
feathertalk lock-asset-package <PROJECT_DIR>
```

`Command` 枚举在 `ExtractFeatures` 之后新增：

```rust
/// 写入素材清单并加锁素材包
LockAssetPackage {
    /// 工程目录
    project_dir: PathBuf,
},
```

`run.rs::build_request` 增加分支，用 `reject_empty(project_dir, "工程目录")` 构造 `Request::LockAssetPackage(ProjectDirParams { .. })`，单元测试加在该文件末尾的 `mod tests`。`render.rs` 的 `UnsupportedCommand` 分支追加 `else if *requested == "lock_asset_package"`，只点名 `ENV_WORKER_HUBERT_DIR`（该常量已存在）。`cli.rs` 里列举 kebab 命令的文档注释，以及 `worker/src/lib.rs` 模块文档与其他枚举「已服务命令」的地方一并更新。

## 15. 测试

夹具全部合成，唯一的例外是 JPEG 字节。

`feathertalk-image`（扩展 `tests/jpeg_decode.rs`，复用文件里现成的 `jpeg_header(width, height)` SOF0 构造器）：`jpeg_dimensions` 与 `decode_jpeg` 给出相同尺寸；只给头部前缀也能解析；SOF 之前被截断则失败；纯垃圾字节则失败。

`feathertalk-frame-adapters`：解析失败映射为 `FrameUndecodable`，且 `path` 被保留。

`feathertalk-frame-pipeline`：`read_landmark_file` 的正常路径，以及 109 行、111 行、非整数、负数、越界、多余空格、缺末尾换行、CRLF、超过字节上限（`InvalidLandmark`）、符号链接与目录（`LandmarkNotRegular`）。

`feathertalk-audio`：`fit_feature_tokens` 的补齐、截断、原样三种情形，以及 `tokens * dims` 溢出。

`feathertalk-worker`：新增 `tests/lock_asset_package.rs` 与 `tests/lock_result.rs`；扩展 `tests/handshake.rs`（宣告随 HuBERT 配置出现）、`tests/runtime.rs`（拒绝文案）、`tests/commands.rs`（未配置工具链时返回 `Failed`——注意该文件行 122 已经在调 `lock_asset_package(dir.path(), locked_manifest())`，改之前先读）、`tests/error_mapping.rs`（三个新变体）。覆盖成功路径、§11 每项准入各一个失败用例、进度序列、取消。

合成夹具的约束是硬的：`QualityReport::new` 与 `FrameQuality::new` 都是公开的，但要求 64 位小写十六进制哈希、非零 `frame_bytes`、`face_score ∈ [0,1]`、正尺寸 bbox、有限模糊方差，且文件名必须恰好是 `frames/{index:06}.jpg` 与 `landmarks/{index:06}.lms`（`validate_artifact_path`）；报告用 `serde_json::to_vec_pretty` 写出。JPEG 字节取现成夹具 `crates/feathertalk-frame-adapters/tests/fixtures/demo_frame_v1/frame.jpg` 复制而来——workspace 里没有 JPEG 编码器，凭空造一张能被 `jpeg_decoder` 解出尺寸的图不现实。worker 侧用 `Path::new(env!("CARGO_MANIFEST_DIR")).join("../feathertalk-frame-adapters/tests/fixtures/demo_frame_v1/frame.jpg")` 跨 crate 引用，判例是 `worker/tests/models.rs:10`。不新增二进制夹具入库。

端到端（门控，`--release`，加在 `crates/feathertalk-cli/tests/real_worker.rs`）：真实 ffmpeg 归一化 → 真实 HuBERT 提取特征 → 从上述夹具 JPEG 合成帧、关键点与 `quality.json` → 加锁。帧数取 **49**：2 秒片段产出 98 个令牌，`2 * 49 = 98` 恰好零对齐，测试断言 `token_adjustment == 0`。只需 `FFMPEG` 与 `HUBERT_DIR` 两个门控变量。后续等 `extract-frames` 的门控测试就绪，这里的合成帧可以换成真实抽帧输出。

## 16. 验证

在 `rust/` 下执行，要求零告警零失败：

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace --all-targets`
- `cargo test --release -p feathertalk-cli --test real_worker`（门控变量齐备时）
- `git diff --check`

## 17. 范围外

- 异常帧的人工编辑与重新评估（迁移设计 §7.1 第 7 步）。属桌面端 UI，本命令只要求加锁时 `anomalies` 为空。
- 解锁与重新加锁。`AssetManifest` 没有从 `Locked` 回到 `Preparing` 的迁移，补上它需要先定义「解锁后哪些产物仍然有效」，是独立设计。
- 特征文件与产出权重的可证明绑定。见 §8，需要文件格式或线协议变更。
- 帧内容与 `quality.json` 哈希的一致性强制。见 §7，与人工修帧直接冲突。
- 重跑、覆盖与部分续跑（`force`、`resume`）。`ProjectDirParams` 带 `deny_unknown_fields`，加标志是线协议变更。
- 新的 `TaskStage` 变体与 `Capabilities` 字段。见 §9、§10。
- 提交阶段的取消。见 §10。
- 训练、渲染、模型导入导出的 UI、桌面端与 GPUI。
- 进度指标字段 `samples_per_second`、`eta_seconds`、`vram_bytes`。
