# FeatherTalk 离线视频执行器设计

日期：2026-08-27  
状态：已确认（里程碑四第四切片）

## 1. 目标

在已经完成的 `RenderPlan`、BGR 帧内核、Burn UNet 适配器和 FeatherHuBERT `.f32` artifact 契约之上，提供一个完整的、可离线运行的单视频执行器。执行器读取已发布的素材帧、landmark 和特征文件，逐帧运行 talking-head 模型，把 BGR24 帧写入 FFmpeg stdin，并在成功后以同目录 staging 文件原子发布最终视频。

本切片不实现模型权重选择或加载、标准模型包、ONNX 导出、旧 `.npy`/`.pth` 迁移 CLI、worker、GPUI、音频解码或媒体标准化。模型实例和 Burn device 由调用方提供；用户目录 `demo/kanghui_training_video_featherhubert_188_latest/` 不属于测试输入，且不读取其中的 `.MOV`。

## 2. 设计边界与方案

采用 `feathertalk-inference` 内的 executor 层，不新建重复的模型 crate：

1. `executor` 负责请求校验、artifact 路径映射、特征/landmark 读取、计划消费、单帧渲染、sink 生命周期和原子发布。
2. `frame_reader` 负责把 JPEG artifact 解码为现有 `BgrFrame`。它通过 trait 注入，因此执行器测试不依赖图片编解码器；生产实现使用受限的 Rust JPEG 解码依赖，并把 RGB 像素明确转换为 BGR。
3. `raw_sink` 负责把 `CommandSpec` 变成带 stdin 的 FFmpeg 子进程。stderr 在独立线程中持续排空并有大小上限，避免子进程因管道堵塞而死锁；测试使用内存 sink。
4. 现有 `RenderPlan`、`read_feature_file`、`read_landmarks`、`compute_face_bbox` 和 `render_planned_frame` 是唯一的语义来源。executor 不重新实现 ping-pong、音频窗口、crop、mask、resize 或 paste 规则。

推荐方案是“注入 reader/sink + 生产适配器”：它同时给离线生产路径和无模型/无 FFmpeg 的确定性单元测试提供了清晰边界，且不把外部进程或图像解码细节泄漏到模型 API。

## 3. 输入与输出契约

### 3.1 请求

新增 `OfflineRenderRequest`，字段使用私有存储和只读 accessor：

```rust
pub struct OfflineRenderRequest {
    frame_dir: PathBuf,
    landmark_dir: PathBuf,
    feature_path: PathBuf,
    audio_path: PathBuf,
    ffmpeg_path: PathBuf,
    output_path: PathBuf,
    task_id: String,
    source_frame_count: usize,
    max_output_frames: Option<usize>,
}
```

构造函数签名为：

```rust
pub fn new(
    frame_dir: PathBuf,
    landmark_dir: PathBuf,
    feature_path: PathBuf,
    audio_path: PathBuf,
    ffmpeg_path: PathBuf,
    output_path: PathBuf,
    task_id: impl Into<String>,
    source_frame_count: usize,
    max_output_frames: Option<usize>,
) -> Result<Self, InferenceError>;
```

`OfflineRenderResult` 提供 `output_path()`, `frame_count()`, `width()` 和 `height()` accessor；它只在目标已经成功发布后返回。

构造时要求所有路径非空且为绝对路径；目录必须是现有的普通非 symlink 目录，输入 artifact 必须是普通非 symlink 文件。`source_frame_count` 至少为 2，`max_output_frames` 若提供必须大于零。帧和 landmark 文件名固定为 `{index:06}.jpg` 与 `{index:06}.lms`，不接受用户提供的任意相对路径。

输出目标使用已有 `validate_output_destination`：父目录必须已存在且没有 symlink，目标必须不存在；因此执行器不会覆盖既有文件。请求开始后若发现目标或 staging 路径竞争，操作失败并保留既有目标不变。

### 3.2 reader/sink seam

```rust
pub trait FrameReader: Send + Sync {
    fn read(&self, index: usize, path: &Path) -> Result<BgrFrame, InferenceError>;
}

pub trait RawVideoSink {
    fn write_frame(&mut self, frame: &BgrFrame) -> Result<(), InferenceError>;
    fn finish(self: Box<Self>) -> Result<(), InferenceError>;
}

pub trait RawVideoSinkFactory: Send + Sync {
    fn start(
        &self,
        command: &CommandSpec,
    ) -> Result<Box<dyn RawVideoSink>, InferenceError>;
}
```

提供 `JpegFrameReader` 和 `SystemRawVideoSinkFactory` 作为生产适配器。reader 必须拒绝零尺寸、非 JPEG/无法解码、超过受控像素上限或非 finite 的结果；解码后的内存布局是行主序 BGR24。sink 必须写出完整帧字节，不允许截断、重排或静默丢帧。

### 3.3 执行 API

```rust
pub struct OfflineRenderResult {
    output_path: PathBuf,
    frame_count: usize,
    width: u32,
    height: u32,
}

pub fn execute_offline_render<B, M, R, F>(
    model: &M,
    device: &B::Device,
    request: &OfflineRenderRequest,
    frame_reader: &R,
    sink_factory: &F,
) -> Result<OfflineRenderResult, InferenceError>
where
    B: burn::tensor::backend::Backend,
    M: feathertalk_models::unet::TalkingHeadModel<B>,
    R: FrameReader + ?Sized,
    F: RawVideoSinkFactory + ?Sized;
```

执行流程固定为：

1. 校验请求和输出目标；读取 `.f32`，要求 `dims == 1024`、token 数为正偶数；建立 `RenderPlan`。
2. 读取 source frame `0` 确定输出宽高，并检查每个计划帧取得的 BGR frame 都与该尺寸一致；读取对应 `.lms`，调用 `read_landmarks` 和 `compute_face_bbox`。
3. 生成同目录、同扩展名的 staging 路径并以 `create_new` 预留；用 staging 路径构造已有 `raw_video_command`，启动 sink。
4. 按 `output_index = 0..plan.output_frame_count()` 顺序调用 `plan.frame(output_index)` 和 `render_planned_frame`，每次把返回的 BGR24 完整写入 sink。输入帧和特征均保持不变。
5. 关闭 stdin，等待 sink/FFmpeg 成功结束；验证 staging 是普通非 symlink、非空文件并 `sync_all`。
6. 仅在上述所有步骤成功后执行 staging 到目标的同目录 `rename`，再同步父目录（平台不支持目录同步时保留文件 rename 结果）。返回结果摘要。

## 4. 失败和原子性

- 任何请求、artifact、模型、reader、sink、FFmpeg、输出验证或 rename 错误都会返回结构化 `InferenceError`。
- staging guard 只删除本次调用成功 `create_new` 预留的路径；不会扫描目录、删除旧文件或触碰用户模型目录。
- 目标文件在执行开始前必须不存在；失败路径始终不会创建或修改目标文件。
- sink 写入失败时先尝试终止并回收子进程，再清理 staging；FFmpeg 非零退出码包含受限 stderr 摘要，不暴露 shell 命令拼接。
- 每个计划帧完成后才写入 sink；模型输出验证失败发生在 `render_planned_frame` 返回之前，因此不会向输出流写入对应的半成品帧。
- executor 不使用 shell，不接受相对 FFmpeg 可执行文件路径，也不静默把 WGPU backend 回退到 CPU。

错误模型补充以下变体（保留已有变体不变）：请求/输入 artifact 无效、reader 解码失败、sink 启动/写入/完成失败、FFmpeg 非零退出、staging 冲突、staging 输出无效、原子发布失败和帧索引/尺寸不一致。路径和 frame index 作为结构化字段保存；底层 panic 不作为正常错误路径。

## 5. 测试策略

### 5.1 单元和集成测试

- `OfflineRenderRequest`：相对/空路径、缺失目录、非普通文件、少于两帧、零 max-output、非法 task id 和既有目标均拒绝且不改变 sentinel。
- `JpegFrameReader`：确定性 RGB→BGR 转换、尺寸、损坏 JPEG、超限输入和 symlink 拒绝。
- executor fake reader/sink：验证 feature 读取、landmark/bbox 读取、计划顺序、ping-pong source index、每帧字节写入、输入帧不变和结果摘要。
- 失败原子性：reader 在中途失败、模型返回 NaN、sink 短写/失败、FFmpeg 非零完成、staging 无效和 rename 失败时，目标文件不存在、预留 staging 被清理，既有目录内容不被扫描或删除。
- 生产 sink：通过受控 helper executable 验证 argv 顺序、stdin 原始 BGR24 字节、stdin EOF 后完成，以及非零退出错误；不依赖系统 PATH 中的 FFmpeg。
- crate-root public API 测试只从 `feathertalk_inference` 导入公开类型。

### 5.2 验收命令

```powershell
cargo test -p feathertalk-inference --all-targets
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
git diff --check
```

WGPU smoke 仍只在已认证 adapter 上运行；本切片的默认测试使用 CPU 和 fake sink，不会静默回退 GPU。用户提供的 `feather_hubert_188_latest_99.pth` 继续由既有显式测试覆盖，本执行器测试不读取该目录的 `.MOV`。

## 6. 后续切片

本切片完成后，里程碑四剩余独立工作为标准模型包、ONNX opset 17 导出/校验和旧模型/特征迁移 CLI；这些工作不得把 executor 的 artifact 或原子发布规则重新定义一遍。
