# FeatherTalk 离线推理与视频合成执行契约设计

日期：2026-08-26  
状态：已确认（里程碑四第一切片）

## 1. 目标

为 Rust 离线推理和视频合成建立一个独立、无副作用、可单元测试的执行契约。该契约把迁移设计第 9 节中容易被不同实现重复解释的规则固化为类型和验证函数，供后续 Burn 推理、像素合成、模型包和 CLI 复用。

本切片不加载模型、不解码图片、不启动 FFmpeg，也不读取 demo 训练资源。`feather_hubert_188_latest_99.pth` 仅作为后续真实模型验证的外部输入路径，不进入本 crate 的测试或发布产物。

## 2. 设计边界

新增 `rust/crates/feathertalk-inference`，只依赖 `feathertalk-preprocess`、`feathertalk-media` 和 `thiserror`。crate 分成四个职责单元：

- `sequence`：生成确定性的素材帧 ping-pong 顺序。
- `plan`：把输出帧映射到素材帧、reference 帧和八槽音频窗口。
- `render`：描述固定 crop/inner 几何与 raw-frame 渲染请求。
- `command`：构造不经过 shell 的 FFmpeg 参数列表。

所有公开包装类型使用私有字段和只读 accessor；路径只接受原生 `Path`/`PathBuf`，不拼接 shell 字符串。

## 3. 固定数据契约

```text
OUTPUT_FPS              = 25
AUDIO_WINDOW_FRAMES    = 8
AUDIO_HALF_WINDOW      = 4
FACE_CROP_SIZE         = 168
FACE_INNER_SIZE        = 160
FACE_BORDER            = 4
```

对 `frame_count >= 2` 的素材，选择序列从帧 `0` 开始，随后按：

```text
0, 1, 2, ..., N-1, N-2, ..., 1, 0, 1, ...
```

循环。每个输出帧的 `reference_frame_index` 等于当前选择的素材帧；不允许隐式选择邻帧作为 reference。

对视频帧 `i`，音频窗口槽位对应 `i-4 .. i+3`。越界槽位表示为 `None`，后续执行器必须将其填为全零特征。窗口长度恒为 8，不因边界缩短。

## 4. 公开 API

### 4.1 帧选择

```rust
pub struct PingPongFrames { /* private state */ }

impl PingPongFrames {
    pub fn new(frame_count: usize) -> Result<Self, InferenceError>;
    pub fn frame_count(&self) -> usize;
    pub fn position(&self) -> usize;
    pub fn next(&mut self) -> usize;
}
```

`new` 拒绝少于两帧；`next` 不分配内存，并按上述序列返回素材索引。

### 4.2 输出计划

```rust
pub struct InferenceFramePlan {
    pub output_index: usize,
    pub source_frame_index: usize,
    pub reference_frame_index: usize,
    pub audio_window: [Option<usize>; 8],
}

pub struct RenderPlan { /* immutable validated plan */ }

impl RenderPlan {
    pub fn new(
        source_frame_count: usize,
        feature_frame_count: usize,
        max_output_frames: Option<usize>,
    ) -> Result<Self, InferenceError>;
    pub fn output_frame_count(&self) -> usize;
    pub fn frame(&self, output_index: usize) -> Result<InferenceFramePlan, InferenceError>;
}
```

输出帧数为 `min(feature_frame_count, max_output_frames.unwrap_or(usize::MAX))`。计划要求特征帧数大于零，且素材至少两帧；`frame` 越界返回结构化错误。计划在构造时验证所有算术不会溢出。

### 4.3 几何和渲染请求

```rust
pub struct RenderGeometry { /* validated constants */ }

impl RenderGeometry {
    pub fn standard() -> Self;
    pub fn crop_size(&self) -> u32;
    pub fn inner_size(&self) -> u32;
    pub fn border(&self) -> u32;
    pub fn replacement_offset(&self) -> (u32, u32);
}

pub struct RawFrameRenderSpec { /* private fields */ }
```

`RenderGeometry::standard()` 必须与 `feathertalk-preprocess::default_crop_spec()` 的 `168/160/4` 保持一致。`replacement_offset` 固定为 `(4, 4)`，表示网络输出替换 crop 内部区域；本切片不执行 resize 或贴回，只输出几何约束。

`RawFrameRenderSpec` 固定保存宽、高、25 FPS、音频路径和输出路径，并拒绝零尺寸、相对 FFmpeg 路径、空音频路径或空输出路径。

### 4.4 FFmpeg 命令

```rust
pub struct CommandSpec { /* executable, arguments, operation */ }

pub fn raw_video_command(
    ffmpeg: &Path,
    spec: &RawFrameRenderSpec,
) -> Result<CommandSpec, InferenceError>;
```

命令参数必须按以下顺序生成，输入/输出路径作为独立 `OsString` 参数，禁止 shell quoting：

```text
ffmpeg -hide_banner -nostdin -y -v error
  -f rawvideo -pix_fmt bgr24 -video_size WxH -framerate 25 -i -
  -i AUDIO
  -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest OUTPUT
```

### 4.5 输出路径契约

提供 `validate_output_destination(path)` 和 `staging_output_path(path, task_id)`：

- 既有目标必须是普通非 symlink 文件，且默认渲染入口拒绝覆盖；
- 目标及其已有父级路径中的 symlink 一律拒绝；
- staging 文件必须与目标位于同一父目录，扩展名保持一致，并包含受限 ASCII task id；
- 不创建、删除或重命名文件；真正的 manifest-last 和 atomic rename 由后续执行器实现。

## 5. 错误模型

错误至少区分：无效字段、帧数不足、特征为空、输出帧越界、输出已存在、symlink、非普通文件、路径为空/非绝对、算术溢出和无效 task id。错误消息不得泄露 shell 命令或底层 panic 文本；底层路径作为结构化字段保留。

## 6. 测试策略

- 帧选择：两帧、三帧、边界反转和长序列循环。
- 计划：首帧/中间帧/末帧音频窗口、零填充槽位、最大输出帧截断、越界和空特征拒绝。
- 几何：标准常量与 preprocess crate 一致，替换偏移为 `(4,4)`。
- 命令：逐项断言 executable、参数顺序、路径不被拆分、25 FPS、rawvideo、`-shortest`。
- 输出路径：已有目标、目录、symlink 父级、非法 task id、同目录 staging。
- 公共 API：集成测试只从 crate root 导入，不依赖私有模块。

## 7. 后续切片接口

后续实现将按此契约接入：

1. raw frame 解码/编码和 BGR 图像 crop、resize、inner 替换、paste-back；
2. Burn FeatherHuBERT 与 Original/MobileOne UNet 推理适配器；
3. 标准模型包 manifest、safetensors 校验和 ONNX opset 17 导出；
4. 旧 `.pth`、`.pth.tar` 和 `.npy` 迁移 CLI；
5. worker/GPUI 任务封装。

每个后续切片都必须复用 `RenderPlan`，不得重新实现帧顺序或音频窗口规则。
