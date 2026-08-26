# FeatherTalk 纯 Rust BGR 帧处理内核设计

日期：2026-08-26  
状态：已确认（里程碑四第二切片）

## 1. 目标

在 `feathertalk-inference` 中加入一个不依赖图像解码库、OpenCV、Burn 或 FFmpeg 的纯 Rust 像素处理内核。内核把当前 C++/Python 推理路径中容易被重复实现的 BGR24 帧操作固化为可单元测试的值类型和函数，供后续 Burn UNet 适配器直接调用。

本切片覆盖：

- 受控的交错 BGR24 帧缓冲区；
- 人脸 bbox crop；
- 与 `FeatherTalk-CPP/src/main.cc` 一致的 half-pixel bilinear resize；
- 从 `168×168` crop 生成 `[1, 6, 160, 160]` 的 UNet image input；
- 将 `[1, 3, 160, 160]` prediction 写回 crop inner；
- 将 crop resize 回 bbox 尺寸并 paste-back 到原始帧；
- 尺寸、缓冲区长度、bbox、tensor shape 和非有限 prediction 的结构化校验。

本切片不读取或修改 `demo/kanghui_training_video_featherhubert_188_latest/`，不解码图片，不加载模型，不启动 FFmpeg，也不改变既有 `RenderPlan` 的帧顺序和音频窗口规则。

## 2. 设计边界与依赖

新增 `src/frame.rs`，继续使用现有 `feathertalk-inference` crate。运行时依赖保持为 `thiserror`、`feathertalk-preprocess` 和 `feathertalk-media`；不引入 `image`、OpenCV、Burn、ndarray 或 unsafe 代码。

`BgrFrame` 采用行主序、像素交错的 BGR24 布局：

```text
row 0: B0 G0 R0 B1 G1 R1 ...
row 1: ...
```

帧宽高使用 `u32`，内部索引和长度计算使用 checked `usize` 运算。构造函数只接受 `width × height × 3` 与缓冲区长度完全相等的输入；所有公开方法都通过借用或拥有值返回，不暴露可绕过不变量的字段。

## 3. 公共 API

```rust
pub struct BgrFrame { /* private fields */ }

impl BgrFrame {
    pub fn new(width: u32, height: u32, bgr: Vec<u8>)
        -> Result<Self, InferenceError>;
    pub fn width(&self) -> u32;
    pub fn height(&self) -> u32;
    pub fn as_bytes(&self) -> &[u8];
    pub fn into_bytes(self) -> Vec<u8>;
    pub fn pixel(&self, x: u32, y: u32) -> Result<[u8; 3], InferenceError>;
}

pub struct UnetImageInput { /* private fields */ }

impl UnetImageInput {
    pub fn shape(&self) -> [usize; 4];
    pub fn as_slice(&self) -> &[f32];
}

pub fn crop_bgr(
    frame: &BgrFrame,
    bbox: &feathertalk_preprocess::FaceBoundingBox,
) -> Result<BgrFrame, InferenceError>;

pub fn resize_bilinear(
    frame: &BgrFrame,
    width: u32,
    height: u32,
) -> Result<BgrFrame, InferenceError>;

pub fn build_unet_image_input(
    face_crop: &BgrFrame,
    geometry: &RenderGeometry,
) -> Result<UnetImageInput, InferenceError>;

pub fn apply_unet_prediction(
    face_crop: &mut BgrFrame,
    prediction: &[f32],
    geometry: &RenderGeometry,
) -> Result<(), InferenceError>;

pub fn paste_bgr(
    destination: &mut BgrFrame,
    source: &BgrFrame,
    x: i32,
    y: i32,
) -> Result<(), InferenceError>;

pub fn render_frame(
    frame: &BgrFrame,
    bbox: &feathertalk_preprocess::FaceBoundingBox,
    prediction: &[f32],
    geometry: &RenderGeometry,
) -> Result<BgrFrame, InferenceError>;
```

`UnetImageInput::shape()` 恒为 `[1, 6, 160, 160]`（由传入的标准 `RenderGeometry` 验证得出），其数据布局为 channel-first，每个通道平面长度为 `160×160`。前三个平面是原始 BGR crop inner，后三个平面是嘴部 mask 后的 BGR crop inner，所有像素先转换为 `value / 255.0`。

`apply_unet_prediction` 要求 prediction 长度为 `3×160×160`，按 BGR channel-first 解释；每个值必须是 finite，随后执行 `clamp(value * 255.0, 0, 255).round()` 并写入 crop 的 `(4,4)..(164,164)` inner 区域。crop 的 4px border 保持不变。

## 4. 像素与几何语义

### 4.1 Crop

`FaceBoundingBox` 使用左上闭、右下开的 `xmin/ymin/xmax/ymax` 语义，与现有 `compute_face_bbox` 和 C++ `Bbox` 一致。crop 要求坐标非负、宽高为正且完全位于源帧内；复制每一行的连续 BGR24 字节，不做隐式 padding。

### 4.2 Bilinear resize

目标像素 `(x,y)` 使用 C++ 参考公式：

```text
scale_x = source_width / target_width
source_x = (x + 0.5) * scale_x - 0.5
x0 = clamp(floor(source_x), 0, source_width - 1)
x1 = clamp(x0 + 1, 0, source_width - 1)
wx = clamp(source_x - floor(source_x), 0, 1)
```

`y` 方向同理。每个 BGR 通道先做双线性插值，再用 nearest-half-up（Rust `f32::round`，输入已被 clamp 为非负）转为 `u8`。这保留 C++ `std::lround` 的边界行为，包括边缘坐标的 clamp 方式。

### 4.3 Image input 与 prediction

只有 `RenderGeometry::standard()` 的 `168/160/4` 几何被接受。mask 矩形从 `feathertalk_preprocess::default_crop_spec().mouth_mask` 读取（`x=5,y=5,width=150,height=145`），并在生成 input 时验证它落在 inner 区域内。这样后续模型适配器不会重新定义嘴部范围。

### 4.4 Paste-back

`paste_bgr` 要求源图完整落在目标帧内，使用行级 `copy_from_slice`，不创建目录、不写文件。`render_frame` 只在返回的新帧中修改内容：它依次执行 crop、resize 到标准 crop、写回 prediction、resize 回 bbox 宽高和 paste-back，输入 `frame` 保持不变。

## 5. 错误模型

在现有 `InferenceError` 中增加以下可区分分支：

- `InvalidFrameDimensions`：宽或高为零；
- `FrameBufferLengthMismatch`：实际 BGR 字节数与期望值不同；
- `PixelOutOfRange`：像素访问越界；
- `InvalidBbox`：bbox 非法或超出帧边界；
- `InvalidResizeTarget`：resize 目标尺寸为零；
- `TensorShapeMismatch`：UNet 输入或输出长度不符合固定 shape；
- `NonFinitePrediction`：prediction 包含 NaN 或无穷值；
- `PasteOutOfBounds`：paste 源区域超出目标帧；
- `AllocationFailure`：checked reserve 失败。

错误中只携带尺寸、索引、字段和结构化路径等信息，不暴露 panic 文本或执行命令。所有算术（像素长度、行偏移、bbox 宽高）在溢出时返回 `ArithmeticOverflow` 或上述结构化错误。

## 6. 测试策略

集成测试只从 crate root 导入公共 API，并按 TDD 先写出失败测试：

- `BgrFrame`：零尺寸、长度不匹配、字节所有权和像素越界；
- crop：合法 2×2 子区域、负坐标、右下越界；
- resize：1×1 平均值、2×2→3×3 half-pixel 边缘值、零目标；
- image input：固定 shape、BGR channel-first、mask 区域为零且非 mask 区域保留；
- prediction：正确写回、clamp/round、border 不变、长度错误和非 finite 拒绝；
- paste/render：合法贴回、越界拒绝、原始输入不被修改、完整组合顺序；
- 公共 API：所有类型和函数可从 crate root 使用，且不依赖私有模块路径。

测试 fixture 使用小型人工 BGR 缓冲区和标准几何，不读取真实视频、图片或 demo 模型。重点断言手工推导的像素值，避免用被测 resize/helper 计算期望值。

## 7. 后续切片接口

后续 Burn 适配器直接消费 `UnetImageInput::as_slice()`，将模型输出复制为 `&[f32]` 后交给 `apply_unet_prediction`；视频循环继续消费现有 `RenderPlan`。任何模型适配器不得重新实现 crop、resize、mask、paste 或帧顺序。
