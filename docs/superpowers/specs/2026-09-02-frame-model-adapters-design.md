# 帧管线生产模型适配器设计

日期：2026-09-02
状态：已确认（抽帧 worker 命令的前置切片）

## 1. 目标

`feathertalk-frame-pipeline` 已经实现固定编号抽帧、质量策略编排、异常分类和原子发布，但三个模型接缝在 `2026-08-24-frame-face-pipeline-design.md` §8 中被显式推迟，目前只有 fake decoder 覆盖组合契约。本切片补齐生产实现，使 `evaluate_frames_with_models` 能在真实 JPEG 帧上执行 SCRFD 人脸筛选、PFLD 关键点解码和 Laplacian 模糊判定。

新增两个 crate，`feathertalk-frame-pipeline` 不做任何修改（trait 签名、阈值常量、异常代码、原子发布语义全部保持现状）：

- `feathertalk-image`：不依赖 Burn 的像素内核，提供 JPEG 解码，以及与 OpenCV 数值语义一致的 resize、灰度转换和 Laplacian 方差；
- `feathertalk-frame-adapters`：把像素内核与既有 `feathertalk-scrfd` / `feathertalk-pfld` Burn runtime 和 `feathertalk-face` 几何函数组合成 `JpegFrameDecoder`、`ScrfdFaceDetector<B>`、`PfldLandmarkPredictor<B>`。

数值目标是与 `data_utils/detect_face.py`、`data_utils/get_landmark.py` 的 OpenCV 路径对齐：模型输入张量逐字节相同，检测框与关键点落在本文档实测并固化的容差内。

本切片不包含抽帧 worker 命令与 CLI 子命令、模型工件路径解析、GPU 后端选择。

## 2. crate 划分与依赖

拆成两个 crate 的理由：像素内核必须能在没有 Burn 的环境中编译和测试，否则每次调试 resize 舍入都要付编译 Burn 的代价，且内核会被迫携带 backend 泛型。

`feathertalk-image`（`rust/crates/feathertalk-image`）运行时依赖只有 `jpeg-decoder` 和 `thiserror`；dev 依赖为 `ndarray`、`ndarray-npy`、`serde_json`、`sha2`、`hex`（读取 npy fixture 和校验 manifest）。不引入 `image`、OpenCV、ndarray 运行时依赖或 unsafe 代码。

`feathertalk-frame-adapters`（`rust/crates/feathertalk-frame-adapters`）依赖 `feathertalk-image`、`feathertalk-frame-pipeline`、`feathertalk-face`、`feathertalk-scrfd`、`feathertalk-pfld` 和 `burn`。feature：`default = []`，`metal` 和 `vulkan` 只做转发到 `feathertalk-scrfd` 的同名 feature；dev 依赖包含 `feathertalk-models`（取 `backend::CpuBackend`，即 `NdArray<f32>`）、`ndarray`、`ndarray-npy`、`serde_json`、`tempfile`。

两个 crate 都加入 `rust/Cargo.toml` 的 workspace members。

接受的重复：`feathertalk-image::BgrImage` 与 `feathertalk-inference::BgrFrame` 是两个独立类型。复用后者会把 `feathertalk-inference` 的依赖（含 Burn 相关链路）拖进像素内核层，代价高于维护一个 200 行的值类型。本切片不做合并重构，也不改动 `feathertalk-inference`。

## 3. feathertalk-image：像素内核

### 3.1 公共 API

```rust
pub struct BgrImage { /* private fields */ }

impl BgrImage {
    pub fn new(width: u32, height: u32, bgr: Vec<u8>) -> Result<Self, ImageError>;
    pub fn width(&self) -> u32;
    pub fn height(&self) -> u32;
    pub fn as_bytes(&self) -> &[u8];
    pub fn pixel(&self, x: u32, y: u32) -> Result<[u8; 3], ImageError>;
}

pub struct GrayImage { /* private fields */ }

impl GrayImage {
    pub fn width(&self) -> u32;
    pub fn height(&self) -> u32;
    pub fn as_bytes(&self) -> &[u8];
}

pub fn decode_jpeg(bytes: &[u8], max_pixels: u64) -> Result<BgrImage, ImageError>;
pub fn resize_area(image: &BgrImage, width: u32, height: u32) -> Result<BgrImage, ImageError>;
pub fn resize_linear(image: &BgrImage, width: u32, height: u32) -> Result<BgrImage, ImageError>;
pub fn to_gray(image: &BgrImage) -> GrayImage;
pub fn laplacian_variance(image: &GrayImage) -> f64;
```

`BgrImage` 采用行主序、像素交错的 BGR24 布局，与 `BgrFrame` 相同；构造函数只接受长度恰好为 `width × height × 3` 的缓冲区，宽高为零一律拒绝。`GrayImage` 存在的唯一目的是避免在调用点传递 `(&[u8], width, height)` 三元组而错配尺寸。

### 3.2 decode_jpeg

使用 `jpeg-decoder`：先读 header 拿到尺寸，`width × height` 超过 `max_pixels` 时在解码前返回 `FrameTooLarge`。像素格式处理：

- `PixelFormat::RGB24`：按 `(r,g,b) -> (b,g,r)` 重排；
- `PixelFormat::L8`：灰度值复制到三个通道；
- 其他格式（含 CMYK32、L16）：返回 `UnsupportedPixelFormat`。

默认上限沿用 `feathertalk-inference::frame_reader::DEFAULT_MAX_FRAME_PIXELS`（`64 * 1024 * 1024`）的数值，由调用方传入，本 crate 不自己定义环境变量或全局配置。

### 3.3 resize_area（INTER_AREA）

`resize_area` 复刻 OpenCV `cv2.resize(..., interpolation=cv2.INTER_AREA)`。数值语义以本机 **OpenCV 5.0.0** 为准，已逐分支实测确认，实现时不需要重新推导。缩放比定义为 `scale = src / dst`，四个分支：

| 条件 | 算法 | 舍入 |
| --- | --- | --- |
| 两轴 scale 均为精确整数，且 kernel 为 2×2 | 块内整数求和 | `(sum + 2) / 4`（进位式四舍五入） |
| 两轴 scale 均为精确整数，kernel 非 2×2 | 块内整数求和 | `round_ties_even(sum as f32 * (1f32 / area))` |
| 两轴 scale 均 ≥ 1 但非整数 | 两遍 f32 累加，镜像 `ResizeArea_Invoker`：权重取自 `computeResizeAreaTab`，先水平累加进 f32 行缓冲，再 `acc += beta * buf` | `round_ties_even` |
| 任一轴放大（含一轴放大一轴缩小） | 通用 2-tap 11 位定点路径，taps 用 area 模式公式 | `clamp((acc + 2) >> 2, 0, 255)` |

第四个分支必须实现：`detect_face.py` 无条件使用 `INTER_AREA`，小于 640 的输入会走放大路径。

`computeResizeAreaTab` 的移植（`d` 为目标索引，`src_n` 为该轴源长度，alpha 为 f32）：

```text
fs1 = d * scale
fs2 = fs1 + scale
cell = min(scale, src_n - fs1)
s2 = min(floor(fs2), src_n - 1)
s1 = min(ceil(fs1), s2)
if s1 - fs1 > 1e-3:  产出 (s1 - 1, (s1 - fs1) / cell)
for s in s1..s2:     产出 (s, 1 / cell)
if fs2 - s2 > 1e-3:  产出 (s2, min(min(fs2 - s2, 1), cell) / cell)
```

2-tap 定点分支的 tap 计算（`INTER_RESIZE_COEF_SCALE = 2048`，`cvRound` 为 round-half-to-even）：

```text
# INTER_LINEAR（half-pixel），f 用 f32 计算：
f = f32((d + 0.5) * scale - 0.5); s = floor(f); f -= s

# INTER_AREA 放大（area 模式）：
s = floor(d * scale); f = f32((d + 1) - (s + 1) / scale)
f = if f <= 0 { 0 } else { f - floor(f) }

# 边界钳制：s < 0 -> s = 0, f = 0；s >= src_n - 1 -> s = src_n - 1, f = 0
a1 = cvRound(f * 2048); a0 = 2048 - a1
```

定点输出流程（已网格搜索过其他舍入组合，只有这一种逐字节一致）：

```text
r   = src[s] * a0 + src[s + 1] * a1      # 水平方向，i32；s + 1 钳制到 src_n - 1
acc = ((b0 * (r0 >> 4)) >> 16) + ((b1 * (r1 >> 4)) >> 16)
out = clamp((acc + 2) >> 2, 0, 255)
```

### 3.4 resize_linear（INTER_LINEAR）

`resize_linear` 复刻 `cv2.resize` 默认插值，taps 用 §3.3 的 half-pixel 公式，输出流程相同。实测结论：

- 缩小与等尺寸：逐字节一致；
- 放大：残留 ±1 偏差，比例 ≤ 0.3%。实测 `150×150 -> 192×192` 为 151/110592，`61×47 -> 192×192` 为 289/110592，`5×4 -> 9×7` 为 3/189。

已排除的成因：IPP 分派、系数独立舍入、以及所有其他垂直方向舍入公式组合。因此放大路径接受一个固化的 ≤1 容差，而不是继续追平：PFLD 预处理随后除以 255，1/255 ≈ 0.004 的输入扰动远小于该模型对输入的敏感度（§8 给出实测的端到端影响）。缩小路径不给容差，必须逐字节一致。

### 3.5 to_gray 与 laplacian_variance

`to_gray` 使用 OpenCV 定点公式，与 `cv2.COLOR_BGR2GRAY` 实测完全一致：

```text
gray = (B * 3735 + G * 19235 + R * 9798 + 16384) >> 15
```

`laplacian_variance` 复刻 `cv2.Laplacian(gray, cv2.CV_64F)` 后取 `.var()`：4 邻域 kernel `[[0,1,0],[1,-4,1],[0,1,0]]`，边界模式 `BORDER_REFLECT_101`，f64 累加，总体方差（`ddof = 0`）。实测最大差值为 0.0。

该函数是 `BLUR_VARIANCE_THRESHOLD = 20.0` 判定的唯一输入来源，因此舍入必须与参考实现一致，不接受容差。

### 3.6 归一化等价关系

SCRFD 预处理的 OpenCV 调用等价于纯张量运算，已实测逐字节一致：

```text
cv2.dnn.blobFromImage(img, 1/128, size, (127.5, 127.5, 127.5), swapRB=True)
  == ((rgb.astype(f32) - 127.5f) * (1f32 / 128f32)).transpose(2, 0, 1)[None]
```

关键点：标量 `127.5` 和 `1/128` 以 f32 参与运算，先减后乘，通道顺序在归一化之前完成 BGR→RGB 交换。

### 3.7 fixture 与生成器

fixture 目录 `rust/crates/feathertalk-image/tests/fixtures/opencv_cpu_v1/`，生成器 `rust/tools/image-parity/python/generate_fixture.py` 加 `requirements-fixture.txt`（钉住 `opencv-python-headless==5.0.0`）。沿用既有约定：`fixture.json` 记录每个文件的字节数、dtype、shape、SHA-256 和生成环境（OpenCV / numpy / Python 版本、线程数），生成器先写 `.staging` 目录再 rename，crate 内提供 `tests/fixture_contract.rs` 和小型 `tests/support/mod.rs` 加载器（复制 scrfd/pfld 的模式）。

七个用例，全部为 KB 级：

| 用例 | 断言 |
| --- | --- |
| area `8×8 -> 4×4`（整数 scale，2×2 kernel） | 逐字节一致 |
| area `8×8 -> 2×2`（整数 scale，4×4 kernel） | 逐字节一致 |
| area `13×9 -> 7×5`（非整数缩小） | 逐字节一致 |
| area `5×5 -> 8×8`（放大） | 逐字节一致 |
| linear `200×200 -> 192×192`（缩小） | 逐字节一致 |
| linear `61×47 -> 192×192`（放大） | `max_abs ≤ 1` 且不一致像素比例 ≤ 1% |
| `64×64` 灰度加 Laplacian 方差 | 灰度逐字节一致，方差 `max_abs = 0` |

`f32::round_ties_even()` 在当前 edition 2024 / rust-version 1.92 下可直接使用，不需要自己实现。

## 4. feathertalk-frame-adapters：适配器

### 4.1 构造与依赖注入

三个类型都对 `burn::tensor::backend::Backend` 泛型，测试用 `CpuBackend`。构造函数只接受显式工件路径和 `B::Device`：不做路径搜索、不读环境变量、不选择后端，这些属于下一切片的 worker 命令。

```rust
impl FrameImageCache {
    pub fn new() -> Self;                                  // 使用默认像素上限
    pub fn with_max_pixels(max_pixels: u64) -> Self;
    pub fn load(&self, path: &Path) -> Result<Arc<BgrImage>, PipelineError>;
}

impl JpegFrameDecoder {
    pub fn new(cache: Arc<FrameImageCache>) -> Self;
}

impl<B: Backend> ScrfdFaceDetector<B> {
    pub fn load(
        paths: &ScrfdArtifactPaths,
        device: B::Device,
        cache: Arc<FrameImageCache>,
    ) -> Result<Self, PipelineError>;

    pub fn from_model(model: ScrfdModel<B>, device: B::Device, cache: Arc<FrameImageCache>) -> Self;
}

impl<B: Backend> PfldLandmarkPredictor<B> {
    pub fn load(
        artifact_directory: &Path,
        device: B::Device,
        cache: Arc<FrameImageCache>,
    ) -> Result<Self, PipelineError>;

    pub fn from_runtime(runtime: PfldRuntime<B>, device: B::Device, cache: Arc<FrameImageCache>) -> Self;
}
```

像素上限归 `FrameImageCache`，因为解码由缓存执行，解码器只做灰度和方差。`FrameImageCache` 同时实现 `Default`（等价于 `new()`），避免 clippy 的 `new_without_default`。`from_model` / `from_runtime` 让测试可以复用一次加载的权重。`ScrfdModel::load` 和 `PfldRuntime::load` 接受 `&B::Device`，适配器持有 `B::Device` 值并在建张量时借用。

### 4.2 FrameImageCache

`Mutex<Option<(PathBuf, Arc<BgrImage>)>>`，容量 1。三个适配器共享同一个 `Arc<FrameImageCache>`：`load` 命中同一路径时克隆 `Arc`，否则读文件、`decode_jpeg`、替换缓存条目。

这个缓存是必要的，因为 `DecodedFrame` 只携带 `path/width/height/laplacian_variance`，不带像素，而 `detect` 和 `predict` 都需要原始像素。`evaluate_frames_with_models` 是逐帧顺序循环（decode → detect → choose_primary → 相交比校验 → predict → 序列化 → 模糊判定），因此容量 1 即可达到 100% 命中率，每帧只解码一次。

已知局限并需要在代码注释中写明：如果外部在一次评估过程中改写了同一路径的帧文件，缓存会返回旧像素。管线自己先抽帧再评估，不会触发这种情况。

### 4.3 JpegFrameDecoder

```text
cache.load（读文件 + decode_jpeg）-> to_gray -> laplacian_variance
       -> DecodedFrame::new(path, width, height, variance)
```

`decode` 的 `index` 参数不参与计算：`PipelineError` 的分支都不携带帧号，帧号由管线在生成 `FrameAnomaly` 时补上。`DecodedFrame::new` 自身校验非零尺寸与有限非负方差，适配器不重复校验。

### 4.4 ScrfdFaceDetector

预处理：

```text
resize_with_padding(ImageSize { width, height })      # feathertalk-face
  -> resize_area(image, new_width, new_height)        # Python 无条件用 INTER_AREA
  -> 居中零填充到 640×640（pad_x / pad_y）
  -> BGR→RGB，(v - 127.5) / 128，NCHW
```

后处理：

```text
ScrfdModel::forward -> 每个 level：
  generate_anchor_centers(transform.model, stride, 2)
  scores / bbox_deltas / keypoint_deltas 通过 .to_data().to_vec() 取回主机
  按 score >= config.confidence_threshold 预筛选下标
  对每个存活下标用长度 1 的切片调用 decode_level(level, stride, ...)：
    Ok(detection)                        -> 收集
    Err(InvalidDetectionGeometry)        -> 丢弃该 anchor
    其他 Err                             -> 传播，message 中带真实 anchor 下标
-> non_max_suppression(&detections, &DetectionConfig {
       confidence_threshold: FACE_CONFIDENCE_THRESHOLD,   // 0.50
       nms_iou_threshold: NMS_IOU_THRESHOLD,              // 0.40
   })
```

`decode_level` 内部已经把 delta 乘以 stride，适配器传入模型原始输出，不预乘。

预筛选和逐 anchor 丢弃都是必需的，不是可选优化。`decode_level` 把坐标钳制到 `[0, W] × [0, H]` 后，对 `x2 > x1 && y2 > y1` 不成立的框返回 `Err(InvalidDetectionGeometry)`（不是丢弃），而 `non_max_suppression` 对任何非正宽高的 detection 也直接返回 `Err`，且该校验发生在 score 过滤之前。实测：demo 帧第 750 帧整批解码会命中 6621 个退化 anchor（stride 8/16/32 分别为 5147 / 1015 / 459），既有 `feathertalk-scrfd` fixture 的 stride 32 也有 398 个；其中一部分（demo stride 32 的 399 个）在钳制之前就已退化，说明网络本身会对背景 anchor 输出负距离。因此不预筛选的实现会在真实帧上稳定失败。按 0.50 预筛选后，这两份样本的退化数都是 0，但贴边人脸仍可能被钳制成零宽框，所以对存活 anchor 采用逐个解码加丢弃，而不是让整帧变成 `model_failed`。

逐 anchor 调用与整批调用数值等价：`decode_level` 对每个下标独立计算，无跨下标状态。预筛选阈值与 NMS、`choose_primary` 用的是同一个常量，重复应用是幂等的，不会引入第二套规则。

`non_max_suppression` 返回存活下标，适配器按该顺序把 `feathertalk_face::Detection` 逐字段搬进 `FaceDetection`：`bbox`、`score`、`keypoints` 的语义和字段序完全相同，无需换算。

`detect` 的契约是「NMS 存活者」，顺序即 `non_max_suppression` 返回的顺序。适配器不再排序、不截断、不做数量判定：`choose_primary` 会自己按 score 排序、再应用 0.50 阈值并判定「恰好一张脸」。

`FaceDetection.bbox` 是 xywh（与 `Detection.bbox` 一致），不是 xyxy。

### 4.5 PfldLandmarkPredictor

```text
compute_face_crop_geometry(ImageSize, face.bbox)      # size = trunc(max(w,h) * 1.05)
  -> 按 geometry.source 裁剪，粘贴到 size×size 全零画布的 (padding.left, padding.top)
  -> resize_linear(canvas, 192, 192)
  -> v / 255.0，NCHW
  -> PfldRuntime::forward
  -> decode_landmarks_with_default_mean_face(output, CropGeometry {
         width: size, height: size,
         offset_x: geometry.origin_x, offset_y: geometry.origin_y,
     })
```

零填充画布对应 Python 的 `copyMakeBorder(..., BORDER_CONSTANT, 0)`；`origin_x/origin_y` 是未钳制的裁剪原点，与 `get_landmark.py` 返回的 `x1/y1` 一致，因此解码出的点可以落在图像外，由管线的 landmark 校验处理。

### 4.6 通道序

两个模型的通道序不同，必须在代码注释中写明，否则是一个静默的精度回归来源：

- PFLD 消费 **BGR**：`get_landmark.py` 没有 `cvtColor`，直接把 `cv2.imread` 的 BGR crop 归一化后送入网络；
- SCRFD 消费 **RGB**：`blobFromImage(..., swapRB=True)`。

## 5. 错误模型

不新增错误类型，不修改 `PipelineError`：

- 读帧文件失败 -> `PipelineError::Io { operation: "decode_frame", path, source }`；
- 其他一切失败（JPEG 解码、resize、张量建构、`forward`、解码后处理）-> `PipelineError::Adapter { component, message: error.to_string() }`。

`component` 取值为 `"jpeg"`、`"scrfd"`、`"pfld"`，与 `evaluate.rs` 中既有的 `"scrfd"` 用法一致。管线把这两类错误分别归类为 `frame_decode_failed` 和 `model_failed`，适配器不需要参与分类。

不重复做非有限值校验：`decode_level`、`non_max_suppression`、`decode_landmarks`、两个 `forward` 都已经拒绝 NaN 和无穷值。

## 6. 可测试接缝

把预处理和后处理暴露为不需要权重的纯函数，使数值一致性可以在毫秒级测试中验证：

```rust
pub struct ScrfdInput {
    pub transform: ResizeTransform,
    pub data: Vec<f32>,          // [1, 3, 640, 640]
}

pub struct LevelHostData {
    pub level: usize,
    pub stride: u32,
    pub scores: Vec<f32>,                       // 长度 N
    pub bbox_distances: Vec<[f32; 4]>,          // 长度 N
    pub keypoint_distances: Vec<[f32; 10]>,     // 长度 N
}

pub fn scrfd_input(image: &BgrImage) -> Result<ScrfdInput, PipelineError>;

pub fn scrfd_detections(
    levels: &[LevelHostData; 3],
    transform: &ResizeTransform,
    config: &DetectionConfig,
) -> Result<Vec<FaceDetection>, PipelineError>;

pub fn pfld_input(
    image: &BgrImage,
    geometry: &FaceCropGeometry,
) -> Result<Vec<f32>, PipelineError>;   // [1, 3, 192, 192]
```

`LevelHostData` 只是把一个 level 的输出从张量搬回主机后的普通 `Vec`，`N` 按 stride 依次为 12800 / 3200 / 800。`scrfd_detections` 承担 §4.4 的全部后处理规则：anchor 生成、按 `config.confidence_threshold` 预筛选、逐 anchor 解码与退化框丢弃、NMS、类型搬运。`detect` 和 `predict` 因此只负责建张量、调 `forward`、搬回主机，数值逻辑全部落在这三个纯函数里。

## 7. fixture 策略

### 7.1 合成 fixture：opencv_cpu_v1

目录 `rust/crates/feathertalk-frame-adapters/tests/fixtures/opencv_cpu_v1/`，生成器 `rust/tools/frame-adapters-parity/python/generate_fixture.py`。

| 文件 | 来源 | 断言 |
| --- | --- | --- |
| `frame_bgr.npy` | 由 `feathertalk-scrfd` 已有 fixture 的 `input.npy` 反演得到（已验证可精确还原） | 其他用例的输入图像 |
| `scrfd_blob.npy` | 对该图像执行 Python 预处理 | `scrfd_input` 逐字节一致 |
| `detections_thr002.json` | 对已提交的 `out0..out8.npy` 以 0.02 / 0.40 做后处理 | 数量相同，bbox 与 keypoints `max_abs ≤ 0.05` 像素，score `≤ 1e-3` |
| `frame.jpg` 加 `frame_decode.npy` | 同一像素经 `cv2.imwrite` / `cv2.imread` | 解码一致性；宽高与 Laplacian 方差精确一致 |
| `crop_blob.npy` | 生成器写死的 bbox（记录在 `fixture.json`）做 crop + pad + resize192 + `/255` | `pfld_input` 逐字节一致 |
| `crop_blob_padded.npy` | 越界 bbox（触发 `copyMakeBorder` 分支） | `pfld_input` 逐字节一致 |
| `landmarks.json` | torch 加载 `checkpoint_epoch_335.pth.tar` 前向 + mean_face 解码 | `predict()` 输出 110 点，最大绝对差 ≤ 1 像素 |

阈值取 0.02 是为了让低分框也进入比较，覆盖 `decode_level` 的坐标映射而不是只覆盖最高分框。

容差理由：SCRFD 用 0.05 像素而不是 scrfd crate 现有的 `1e-3`，因为 level 输出的 delta 会乘以 stride（最大 32）再映射回源坐标，输入端 1e-3 级差异被放大；landmark 是 `i32`，`trunc` 边界会整体翻 ±1。既有精度先例见 `feathertalk-scrfd/tests/support/mod.rs`（`max_abs ≤ 1e-3`，`mean_abs ≤ 1e-4`）。

生成器必须复刻 Rust 侧的顺序和钳制规则：先按阈值筛选 anchor，再把 bbox 和 keypoints 钳制到 `[0, W] × [0, H]`，再丢弃 `x2 > x1 && y2 > y1` 不成立的框，最后做 NMS。Python 参考实现传入 `max_shape=None`、完全不钳制，因此直接照抄 `detect_face.py` 会在贴边框上产生假失败。0.02 阈值下这条规则是实测会触发的：既有 fixture 的 stride 32 在该阈值上有 33 个退化 anchor。

### 7.2 真实帧 fixture：demo_frame_v1

合成 fixture 证明数值一致，但不能证明一张真实人脸能通过 0.50 阈值。本地已安装 ffmpeg（`D:\environment\ffmpeg\bin`，`2026-06-15-git-44d082edc8-full_build`），仓库内已跟踪 `demo/feathertalk_demo_latest_188.mp4`（7,442,868 字节，h264 1280×720，25 fps，1511 帧，SHA-256 `9353ad796089aa104765d651ca99f158349cfd203644923b2fa72f68b44e9ac1`），因此可以固化一个真实帧。

抽帧命令（已验证与 `-ss 30` 逐字节一致，PNG 为无损中间产物）：

```text
ffmpeg -v error -y -i demo/feathertalk_demo_latest_188.mp4 \
  -vf "select=eq(n\,750)" -fps_mode passthrough -frames:v 1 frame_750.png
```

生成流程：ffmpeg 抽出第 750 帧为 PNG，再由 cv2 读 PNG 并以 q=90 写出两个 JPEG。JPEG 字节因此只依赖钉住的 cv2 版本，与 ffmpeg 的 mjpeg 编码器无关。

目录 `rust/crates/feathertalk-frame-adapters/tests/fixtures/demo_frame_v1/` 包含：`frame.jpg`（约 158 KB）、`frame_blurred.jpg`（`GaussianBlur(k=19, σ=3)` 后编码，约 99 KB）、`expected.json`、`fixture.json`（记录视频 SHA-256、帧号 750、ffmpeg / cv2 / torch 版本）。

第 750 帧的实测值（cv2 5.0.0 `dnn.readNetFromONNX` 加 torch 2.13 CPU，conf 0.50 / NMS 0.40）：

| 量 | 值 |
| --- | --- |
| 各 level 最高分 | 0.1047 / 0.8111 / 0.0341 |
| NMS 存活数 | 1，score `0.81110` |
| bbox（xyxy） | `[551.902, 79.089, 706.897, 283.154]` |
| bbox（xywh） | `[551.902, 79.089, 154.995, 204.065]` |
| resize 几何 | `new_height = 361, new_width = 640, pad_y = 139, pad_x = 0` |
| crop 几何 | `size = 214`，`origin = (521, 74)`，完全在图像内 |
| PFLD 110 点范围 | x ∈ [550, 710]，y ∈ [131, 284] |
| Laplacian 方差 | 756.684（阈值 20） |
| 模糊版本方差 | 3.991（经 q90 JPEG 后 5.122），score 仍为 0.8064 |

断言用行为加宽容差，逐字节一致性留在 §7.1 的 npy 路径上：

- 恰好 1 个检测；score ≥ 0.50 且 `|Δ| ≤ 0.01`；
- bbox 与 keypoints `|Δ| ≤ 1.0` 像素；
- landmark `|Δ| ≤ 2` 像素（整数坐标，吸收 crop `size` ±1 翻转）；
- Laplacian 方差相对误差 `|Δ| ≤ 1%`，并硬断言清晰帧 `> 20`、模糊帧 `< 20`。

容差依据实测的扰动敏感度：q90 重编码使最大像素差达 47，但 Δscore 0.0003、Δbbox 0.122 像素、Δkps 0.119 像素、Δlandmark 1 像素，crop `size` 由 214 翻到 215；q95 的 Δbbox 为 0.098；ffmpeg `-q:v 2` 的 Δbbox 为 0.344；±1 均匀噪声的 Δscore 0.00014、Δbbox 0.058 像素、Δlandmark 1 像素。

### 7.3 测试分层

1. `feathertalk-image` 的 OpenCV fixture（§3.7），不加载模型；
2. 纯函数 fixture（§7.1 的 blob 与 detections），不加载权重；
3. 真实权重加 `CpuBackend` 的 `detect()` / `predict()`；
4. 管线集成：在 tempdir 中调用 `evaluate_frames_with_models`，三条全真实链路——`demo_frame_v1/frame.jpg` 判定为 accepted，`frame_blurred.jpg` 判定为 `blurred_frame`，`opencv_cpu_v1/frame.jpg`（合成图，无人脸）判定为 `face_not_found`；
5. `FrameImageCache` 单元测试：同路径只解码一次，路径变化使缓存失效。

`multiple_faces` 和 `bbox_out_of_bounds` 继续用 stub 检测器覆盖——真实帧无法自然产生这两种异常。其余异常路径不再依赖 stub。

成本：fixture 增加约 257 KB；测试增加 2 次 SCRFD 与 2 次 PFLD CPU 前向，约 10 秒。

## 8. 已知偏差与容差理由

`resize_linear` 放大路径：≤ 0.3% 像素存在 ±1 偏差（§3.4）。影响链路是 PFLD 输入除以 255，即 ≤ 0.004 的输入扰动；实测 ±1 均匀噪声只带来 1 像素的 landmark 变化，因此 §7.2 的 2 像素容差已覆盖。

JPEG 解码一致性尚未验证：`jpeg-decoder` 与 libjpeg-turbo 的 IDCT 实现不同，可能存在 ±1 级差异。风险已被隔离——所有 blob 断言都从 `frame_bgr.npy` 出发，不经过 JPEG；解码一致性只有一个独立用例。处理规则在实现第一步就地确定，不后延：先按逐字节一致写该用例，如果实测不一致，把实测的 `max_abs` 与不一致比例写入 fixture 并在测试注释中说明来源。

OpenCV 版本分歧：新 fixture 用 cv2 5.0.0 生成并在 `fixture.json` 中记录；`feathertalk-scrfd` 的既有 fixture 由 cv2 4.12.0 生成，其生成器会硬失败于版本不匹配，本切片不改动它。5.0.0 的 dnn 使用新 graph engine（会打印 "Targets are not supported by the new graph engine for now"），由此产生的差异被 §7.2 的宽容差吸收。

钳制语义与 Python 参考不同：Rust 侧把框和关键点钳制到图像范围内，Python 不钳制。对通过 0.50 的真实人脸框实测无差异（demo 帧第 750 帧的存活框未被钳制改动）；差异只出现在低分背景 anchor 上，由 §7.1 的生成器规则对齐。

## 9. 排除项

- 抽帧 worker 命令与 CLI 子命令（下一切片）；
- 模型工件路径解析与模型发现；
- GPU / 后端选择策略；
- `evaluate_frames_with_models` 的并行化；
- 合并 `BgrFrame` 与 `BgrImage`；
- 统一旧 fixture 的 OpenCV 版本；
- 修改 `feathertalk-frame-pipeline`、`feathertalk-face`、`feathertalk-scrfd`、`feathertalk-pfld` 的任何公开行为。
