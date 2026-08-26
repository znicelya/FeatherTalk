# FeatherTalk Burn 推理适配器设计

日期：2026-08-26
状态：已确认（里程碑四第三切片）

## 1. 目标

为已完成的 FeatherHuBERT、Original UNet、MobileOne UNet、离线推理计划和纯 Rust BGR 帧内核建立一个稳定的 Burn 推理边界。该边界负责把经过验证的 Rust 值转换为 Burn tensor、运行模型、校验输出，并把 prediction 交回现有像素内核；它不重新解释帧顺序、音频窗口、crop、resize、mask 或 paste 规则。

本切片覆盖：

- 统一的 Original/MobileOne UNet 推理 trait；
- `[tokens, 1024]` FeatherHuBERT 特征到 `[1, 16, 32, 32]` UNet 音频 tensor 的确定性转换；
- `UnetImageInput` 到 Burn image tensor 的转换；
- UNet 输出 shape、finite 和 `[0,1]` 范围校验；
- 按 `RenderPlan` 执行单帧模型前向和 BGR 写回；
- 从现有模型构造 FeatherHuBERT 长音频 adapter；
- 受限 `.pth` FeatherHuBERT checkpoint 配置推导和 Rust 导入；
- 使用用户提供的 `feather_hubert_188_latest_99.pth` 做显式只读导入与 CPU forward 验证。

本切片不实现图片/JPEG 解码、WAV 解码或重采样、FFmpeg 子进程管理、完整视频循环、标准模型包、ONNX 导出、worker、CLI 或 GPUI。

## 2. 方案选择

采用分层适配方案：

1. `feathertalk-models` 定义模型级 trait，以及从已构造模型创建长音频 adapter 的入口；
2. `feathertalk-weights` 单向依赖 `feathertalk-models`，负责 checkpoint 检查、配置推导和权重应用；
3. `feathertalk-inference` 定义领域值到 Burn tensor 的桥接和单帧渲染组合；
4. 已有 `feathertalk-audio`、`RenderPlan` 和 `frame.rs` 继续拥有各自的数据语义。

不采用把全部离线推理流程移入 `feathertalk-models` 的方案，因为模型 crate 不应依赖图像、bbox 或帧计划。不采用运行时 enum 或闭包统一 UNet 的方案，因为类型系统应保证 MobileOne 产品推理只接受重参数化后的 `MobileOneUnetInference`，而不是训练图。

## 3. crate 边界与依赖

### 3.1 `feathertalk-models`

新增：

- `TalkingHeadModel<B>`：统一 Original UNet 和 MobileOne inference graph；
- `BurnFeatherHubertEncoder::from_model`：复用已经导入权重的模型。

`TalkingHeadModel` 只由以下类型实现：

- `OriginalUnet<B>`；
- `MobileOneUnetInference<B>`。

`MobileOneUnet<B>` 训练图不实现该 trait。调用方必须显式调用 `reparameterize()`，从而在编译期避免把训练态多分支图误用于产品推理。

### 3.2 `feathertalk-inference`

增加对 `burn`、`feathertalk-audio` 和 `feathertalk-models` 的依赖，新增 `burn.rs`。该模块只负责：

- 校验并转换 image/audio 输入；
- 调用 `TalkingHeadModel`；
- 把模型输出复制回受控 `Vec<f32>`；
- 组合 `RenderPlan`、bbox 和帧内核执行单帧渲染。

它不加载文件、不生成权重、不执行 Python 或 FFmpeg。

### 3.3 `feathertalk-weights`

把现有对 `feathertalk-models` 的 dev-dependency 提升为普通生产依赖，并在该 crate 新增 FeatherHuBERT checkpoint 配置检查/推导和受限导入函数。现有 `LegacyImportRequest` 和 `import_into` 保持唯一 `.pth` tensor 应用实现；`feathertalk-models` 不得反向依赖 `feathertalk-weights`，从而维持 `feathertalk-weights -> feathertalk-models` 的单向依赖并避免循环。

## 4. 公共模型接口

```rust
pub trait TalkingHeadModel<B: burn::tensor::backend::Backend> {
    fn forward_talking_head(
        &self,
        image: burn::tensor::Tensor<B, 4>,
        audio: burn::tensor::Tensor<B, 4>,
    ) -> burn::tensor::Tensor<B, 4>;
}
```

trait 不负责 shape 或 finite 校验；这些校验集中在 `feathertalk-inference` 的适配层。模型已有的内部断言作为防御性检查保留，但公共推理调用在进入模型前返回结构化 `InferenceError`，不依赖 panic 作为正常错误路径。

```rust
impl<B: Backend> BurnFeatherHubertEncoder<B> {
    pub fn from_model(model: FeatherHubertEncoder<B>, device: &B::Device) -> Self;
    pub fn model(&self) -> &FeatherHubertEncoder<B>;
}
```

`from_model` 从 `model.config.output_dim` 取得输出维度，使 checkpoint 导入后的模型可直接接入现有 `extract_long_audio`。

## 5. FeatherHuBERT checkpoint 导入

### 5.1 配置推导

用户提供的 `.pth` 可能在顶层 `config` 或 `args` 中记录配置，也可能只有 state dict。Rust 不执行任意 pickle callable。配置通过受限 tensor snapshot 推导并与可选 metadata 交叉验证：

- `channels`：`proj.weight` 的输入通道；
- `output_dim`：`proj.weight` 的输出通道；
- `expansion`：`encoder.0.pw_expand.weight` 的输出通道除以 `channels`；
- `num_blocks`：连续 `encoder.0..N-1` block 的完整 tensor 集；
- `dropout`：优先读取可选 metadata；它不属于 state dict。Rust 产品推理实例固定使用 `0.0`（eval 语义），因此训练 checkpoint 中的 `0.05` 等 dropout 值不会进入推理图；该值仍须在 metadata 存在时做合法性和交叉一致性检查。

推导过程要求：

- frontend 恰有 7 层，固定 kernel/stride 图不变；
- encoder block 索引从 0 连续且每层 tensor 集完整；
- `channels > 0`、`expansion > 0`、`num_blocks > 0`、`output_dim > 0`；生产 FeatherHuBERT/UNet 接线另行要求 `output_dim == 1024`。
- 所有相关 tensor 为 float32，shape 相互一致；
- 配置算术使用 checked 运算。

### 5.2 导入 API

以下 API 由 `feathertalk-weights` 导出；返回的模型类型和配置类型来自其单向依赖 `feathertalk-models`：

```rust
pub struct FeatherHubertCheckpoint {
    config: FeatherHubertConfig,
    source_sha256: String,
    tensor_count: usize,
    total_elements: u64,
}

impl FeatherHubertCheckpoint {
    pub fn config(&self) -> &FeatherHubertConfig;
    pub fn source_sha256(&self) -> &str;
    pub fn tensor_count(&self) -> usize;
    pub fn total_elements(&self) -> u64;
}

pub fn inspect_feather_hubert_checkpoint(
    path: &std::path::Path,
) -> Result<FeatherHubertCheckpoint, WeightImportError>;

pub fn load_feather_hubert_checkpoint<B: Backend>(
    path: &std::path::Path,
    device: &B::Device,
) -> Result<(FeatherHubertEncoder<B>, FeatherHubertCheckpoint), WeightImportError>;
```

加载顺序为 inspect → 用推导配置初始化候选模型 → 现有 `import_into` 严格应用 → 比较报告中的 SHA-256/tensor 数/element 数 → 返回模型。任何失败都不修改外部模型或源文件。

## 6. 推理数据契约

### 6.1 音频特征

资产特征继续使用 `FeatureMatrix { tokens, dims, values }`，要求：

```text
dims = 1024
tokens > 0
tokens % 2 = 0
feature_frame_count = tokens / 2
```

每个视频帧是连续两个 token。`InferenceFramePlan::audio_window` 的 8 个槽位按顺序展开，每个槽位提供 `[2,1024]`；`None` 槽位写入 2048 个零。最终 flat buffer 长度恒为：

```text
8 * 2 * 1024 = 16384 = 16 * 32 * 32
```

该 buffer 不转置，直接按 Python `reshape(16,32,32)` 的连续内存语义构造 `[1,16,32,32]`。

```rust
pub struct UnetAudioInput { /* private */ }

impl UnetAudioInput {
    pub fn shape(&self) -> [usize; 4]; // [1,16,32,32]
    pub fn as_slice(&self) -> &[f32];
}

pub fn build_unet_audio_input(
    features: &feathertalk_audio::FeatureMatrix,
    plan: &InferenceFramePlan,
) -> Result<UnetAudioInput, InferenceError>;
```

### 6.2 image 与 prediction

`run_unet_prediction` 只接受现有 `UnetImageInput` 和 `UnetAudioInput`：

```rust
pub fn run_unet_prediction<B, M>(
    model: &M,
    image: &UnetImageInput,
    audio: &UnetAudioInput,
    device: &B::Device,
) -> Result<Vec<f32>, InferenceError>
where
    B: Backend,
    M: TalkingHeadModel<B>;
```

进入模型前验证 input shape 和所有值 finite。输出必须：

- shape 恰为 `[1,3,160,160]`；
- flat 长度恰为 `3*160*160`；
- 每个值 finite；
- 每个值位于闭区间 `[0,1]`。

输出范围超界作为模型契约错误返回，不在适配层静默 clamp。像素写回时现有 `apply_unet_prediction` 仍保留 clamp/round，作为最后一道防御。

## 7. 单帧组合

```rust
pub fn render_planned_frame<B, M>(
    model: &M,
    frame: &BgrFrame,
    bbox: &feathertalk_preprocess::FaceBoundingBox,
    features: &feathertalk_audio::FeatureMatrix,
    plan: &InferenceFramePlan,
    geometry: &RenderGeometry,
    device: &B::Device,
) -> Result<BgrFrame, InferenceError>
where
    B: Backend,
    M: TalkingHeadModel<B>;
```

固定顺序：

1. 验证 `plan.output_index < feature_frame_count`；
2. 按现有 bbox crop 并 resize 到 `168×168`；
3. 调用 `build_unet_image_input`；
4. 调用 `build_unet_audio_input`；
5. 调用 `run_unet_prediction`；
6. 调用现有 `render_frame` 完成 inner 替换、resize-back 和 paste-back。

函数不修改输入帧或特征。reference 仍由现有 `build_unet_image_input` 使用当前 crop 实现；不新增邻帧 reference 语义。

## 8. 错误模型

在 `InferenceError` 增加可区分错误：

- `InvalidFeatureShape { tokens, dims }`；
- `InvalidAudioWindowIndex { slot, index, frame_count }`；
- `NonFiniteModelInput { context, index }`；
- `ModelTensorData { context, message }`；
- `NonFiniteModelOutput { index }`；
- `ModelOutputOutOfRange { index, value }`。

既有 `TensorShapeMismatch` 用于 image/audio/output shape。所有错误发生在 `render_frame` 修改返回值之前，因此失败不会产生半修改帧。

## 9. 测试与验收

### 9.1 单元/集成测试

- `TalkingHeadModel`：Original 和 `MobileOneUnetInference` 可编译使用；训练态 `MobileOneUnet` 不通过公共推理 helper 的类型边界。
- audio input：首帧/中间帧/末帧窗口、零填充、连续两个 token 的字节顺序、奇数 token、错误维度和越界索引。
- prediction：CPU micro Original 和重参数化 MobileOne 输出固定 shape/range/finite。
- 事务性：错误 output 不修改输入帧。
- 单帧组合：小型 BGR 帧、标准 bbox、确定性零/一模型输出，验证 `RenderPlan` 窗口与现有 pixel kernel 被复用。
- checkpoint：golden `feather_micro.pth` 可推导 `32/2/2/64` 并与现有 parity 输出一致；坏 block 集、shape、dtype、额外 tensor 和限制均拒绝。

### 9.2 用户提供模型

显式测试路径：

```text
demo/kanghui_training_video_featherhubert_188_latest/feather_hubert_188_latest_99.pth
```

已记录只读基线：

```text
bytes  = 40436613
sha256 = 58df96af118d75d7f69da441e1f3960096f28dda637a4e8f4265f108d27aeb52
```

验收测试通过环境变量显式启用，避免普通 CI 依赖未跟踪文件：

```text
FEATHERTALK_FEATHER_HUBERT_CHECKPOINT=<absolute path>
```

测试要求 SHA-256 与记录值一致，受限 Rust 导入成功，推导 `output_dim == 1024`，对 1360 个有限 samples 完成 CPU forward，输出 shape 为 `[1,4,1024]` 且全部 finite。通用 golden micro checkpoint 仍允许其明确声明的 `output_dim == 64`，用于导入/形状/parity 测试。测试不读取同目录视频，不写入模型目录。

### 9.3 完成门槛

运行：

```powershell
cargo test -p feathertalk-models --all-targets
cargo test -p feathertalk-weights --all-targets
cargo test -p feathertalk-inference --all-targets
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
git diff --check
```

随后以环境变量显式运行用户模型测试。WGPU smoke 继续只在 certified adapter 上运行，不能静默回退 CPU。

## 10. 后续切片

本切片完成后，下一步是完整离线视频执行器：受控读取素材帧/landmark/feature artifact，循环消费 `RenderPlan`，把 BGR raw frames 写入 FFmpeg stdin，并以 staging + atomic publish 安装输出。标准模型包、ONNX opset 17 和旧模型/`.npy` 迁移 CLI 继续作为独立里程碑四切片。
