# 项目训练数据集装配设计

日期：2026-09-04
状态：已定稿

## 1. 目标与范围

里程碑二在 `b4cb3fd` 收口：归一化、抽帧、提取特征、加锁四条 worker 命令与四个 CLI 子命令连成一条链，`assets/assets.json` 的 `state` 已经能走到 `Locked`。下一站是迁移设计 §15.3 的里程碑三——训练。`TaskKind::Train` 是 13 个任务类型里尚未实现的 7 个之一。

训练的算法层其实早已落地并各自带着单元测试：损失函数（迁移设计 §8.2）、可恢复采样器 `TrainingDataLoader`（§8.1）、检查点与恢复、指标与预览产物都在 `feathertalk-training` 内。真正的缺口只有一处——`TrainingDataset` 至今没有任何实现：

```rust
pub trait TrainingDataset {
    type Item;
    fn frame_count(&self) -> u64;
    fn load_sample(&self, sample: &TrainingSample) -> Result<Self::Item, TrainingError>;
}
```

`TrainingDataLoader` 已经把全部随机性（帧排列、参考帧、时序步长）收在自己手里，`TrainingSample` 是它交出来的纯数据：`SingleFrame { target_index, reference_index }` 与 `TemporalPair { first_target_index, second_target_index, reference_index }`，字段全是 `u64`。因此 `load_sample` 必须是「文件 + 样本索引」的纯函数：不允许内部随机、不允许隐藏状态、同一个 `TrainingSample` 在任何时刻都得给出同一个 `Item`。这一条约束决定了本切片全部接口的形状。

本切片实现「加锁项目目录 → 训练张量批次」这一段，改动分布在四处：

- `feathertalk-preprocess`：`geometry.rs` 新增 `MouthRoiSpec`、`default_mouth_roi_spec`、`mouth_roi_rect`（§3）。
- `feathertalk-inference`：`frame.rs` 新增 `MouthMasking`、`InnerImagePlanes`、`build_inner_image_planes` 并用它重写 `build_unet_image_input`（§4），新增 `build_face_crop`（§5）；`burn.rs` 抽出 `build_unet_audio_window`（§6）。
- 新 crate `feathertalk-training-data`：`ProjectTrainingDataset`、`TrainingItem` / `FrameSample`、批次堆叠、错误类型（§7–§12）。
- `rust/Cargo.toml`：`members` 增加一行。

不改 worker、不改 CLI、不动线协议，也不改 `feathertalk-training` 一行代码——`impl From<TrainingDataError> for TrainingError` 写在新 crate 内即可，孤儿规则允许（`From` 的类型参数里有本地类型）。

## 2. 为什么先做数据装配

完整的 `train` 命令粗算 14 个提交：数据集装配、worker 命令编排、VGG19 感知损失的模型包配置与握手门控、CLI 子命令、门控端到端测试、两套 `TrainingMode` 枚举的映射。一个切片塞不下，按 crate 边界切成 A/B 两半，本文件是 A。

切点选在这里，是因为依赖方向是单向的：B 需要 A 的 `TrainingItem` 与批次张量才能把数据喂进 `TalkingHeadModel`，A 不需要 B 的任何东西。A 还有一个独立优点——它可以完全离线测试：不需要 ffmpeg、不需要模型权重、不需要 GPU，夹具是若干合成文件加一个桩帧读取器（§13）。B 的端到端测试则必须有真实 HuBERT 包与 VGG19 权重，属于另一个量级的门控成本，混在一起会让 A 的回归无法快速跑。

B 的内容（不在本文件内）：`train` worker 命令与阶段/进度/取消、VGG19 包的配置与握手宣告、`feathertalk-cli` 的 `train` 子命令、门控端到端，以及 `feathertalk_training::TrainingMode{Baseline, MouthRoi, MouthRoiTemporal}` 与 `feathertalk_domain::TrainingMode{Baseline, MouthRoi, Temporal}` 的映射——两套枚举名字不同、判别值也不保证一致，映射必须显式写死并逐项测试。

## 3. 嘴部 ROI 几何

`mouth_roi_loss` 与 `temporal_loss` 都要一张 `[B,1,H,W]` 的嘴部掩码，但 workspace 里现在只有静态矩形：`default_crop_spec().mouth_mask` = `{x:5, y:5, width:150, height:145}`，那是 UNet 的重绘区域，不是嘴。Python 侧 `dataset_mouth_roi.py::mouth_roi_from_landmarks` 从关键点算出逐帧的嘴部矩形，这条几何必须原样搬过来。落点是 `feathertalk-preprocess/src/geometry.rs`——`compute_face_bbox` 与 `CropSpec` 已经在那里，ROI 是同一坐标系里的下一步。

```rust
pub struct MouthRoiSpec {
    pub start: usize,
    pub end: usize,
    pub expand_x: f32,
    pub expand_y: f32,
    pub min_w: u32,
    pub min_h: u32,
    pub pad: u32,
}

pub fn default_mouth_roi_spec() -> MouthRoiSpec;
pub fn mouth_roi_rect(
    landmarks: &Landmarks,
    crop: &CropSpec,
    spec: &MouthRoiSpec,
) -> Result<MaskRect, PreprocessError>;
```

默认值对齐 `MouthRoiConfig`：`start 90`、`end 110`、`expand_x 1.45`、`expand_y 1.75`、`min_w 52`、`min_h 36`、`pad 2`。导出列表变成 `pub use geometry::{CropSpec, FaceBoundingBox, MaskRect, MouthRoiSpec, compute_face_bbox, default_crop_spec, default_mouth_roi_spec, mouth_roi_rect};`。

算法逐步照抄，全部中间量走 `f32`：

1. `compute_face_bbox(landmarks)`，`scale = crop.crop_size as f32 / (xmax - xmin) as f32`。
2. 取 `[start..end)` 的点，**先截断到整数**（`p.x.trunc()`）。Python 的 `read_landmarks` 返回 int32，`astype(np.float32)` 之前坐标已经是整数；我们的 `Point` 是 `f32`，`.lms` 文件里写的又都是整数，所以这一步在实际数据上是恒等变换——但写出来能让契约精确，也能防住将来有人塞进小数坐标时两边悄悄分叉。
3. `px = (p.x.trunc() - xmin as f32) * scale - crop.border as f32`，`py` 同理用 `ymin`。
4. 取各轴 min/max 得 `x1,x2,y1,y2`；`cx = (x1 + x2) / 2.0`；`width = ((x2 - x1 + 2 * pad) * expand_x).max(min_w as f32)`，纵向同理。
5. `rx1 = (cx - width / 2.0).round_ties_even() as i64`，四个边界都用 `round_ties_even`。**不能用 `f32::round`**：Rust 的 `round` 是「远离零」，Python 内建 `round` 与 numpy 都是「就近取偶」。半整数上差 1 像素会让 ROI 整体平移，而 mask-L1 的分母就是 ROI 面积，损失量级会跟 Python 基线错开一个可观的比例。
6. 夹紧顺序照抄：`rx1 = rx1.clamp(0, inner_size - 1)`，然后 `rx2 = max(rx1 + 1, min(inner_size, rx2))`。顺序反了会在退化情形（嘴部点全部落在内圈外）产出零宽矩形。
7. 返回 `MaskRect { x: rx1, y: ry1, width: rx2 - rx1, height: ry2 - ry1 }`（`u32` 字段），宽高恒 ≥ 1。

拒绝条件全部走 `PreprocessError::InvalidGeometry { field, message }`（现成变体，`field: &'static str`）：

| 条件 | `field` |
| --- | --- |
| `start >= end` | `mouth_roi_range` |
| `end > PFLD_LANDMARK_COUNT` | `mouth_roi_range` |
| `expand_x` / `expand_y` 非有限或 ≤ 0 | `mouth_roi_expand` |
| `min_w` / `min_h` 为 0 或 > `inner_size` | `mouth_roi_min_size` |
| `crop.inner_size == 0` | `inner_size` |
| `scale` 或投影坐标非有限 | `mouth_roi_projection` |

被否决的替代方案：为几何单独开一个 crate。ROI 与 `compute_face_bbox` 共享 bbox → 内圈的坐标变换，拆开会立刻出现「谁拥有 `scale` 定义」的问题；而且预览与桌面端标注也要用这条几何，放在已有的叶子 crate 里传播成本最低。

被否决的替代方案：返回 `(u32, u32, u32, u32)` 元组。四个同类型数字无名字，调用点写反了编译器不会吭声，`MaskRect` 是同一文件里现成的类型。

被否决的替代方案：直接返回 `Vec<f32>` 掩码平面。那会把缓冲区布局（`[1,160,160]`、行主序）绑进一个几何函数里，而 `feathertalk-preprocess` 依赖只有 `thiserror`，不该开始承担张量布局。

## 4. 内圈图像平面构造

现在的 `build_unet_image_input(face_crop, geometry)` 从**一张** crop 造出 6 通道：`ch0..3` 是未遮挡的内圈，`ch3..6` 是同一张图挖掉 `mouth_mask` 的版本。推理这么做是有意的——`inference.py:127` 直接承认「参考帧就用当前帧」，因为推理时没有别的帧可用。

训练不一样。`datasetsss.py` 的 `img_concat_T = cat([ref_T, masked_T])`：前 3 通道来自**随机参考帧**，后 3 通道来自**当前帧**的遮挡版本。两半来自不同图像，现有函数无法表达。同时 `2026-08-26-bgr-frame-kernel-design.md` §7 明确禁止在别处重写 crop/resize/mask 三件套。因此把「造一张 3 通道内圈平面」抽成公开的窄函数，两个调用方各自组合：

```rust
pub enum MouthMasking { Keep, Blackout }

pub struct InnerImagePlanes { /* 私有 values: Vec<f32> */ }
impl InnerImagePlanes {
    pub fn shape(&self) -> [usize; 4];       // [1, 3, 160, 160]
    pub fn as_slice(&self) -> &[f32];
    pub fn into_values(self) -> Vec<f32>;
}

pub fn build_inner_image_planes(
    face_crop: &BgrFrame,
    geometry: &RenderGeometry,
    masking: MouthMasking,
) -> Result<InnerImagePlanes, InferenceError>;
```

`build_unet_image_input` 重写为 `Keep` ⧺ `Blackout` 的拼接，输出必须逐位不变：通道序 BGR、`/255.0`、源像素取 `(x + border, y + border)`、`ch0..3` 未遮挡、`ch3..6` 遮挡。现有 `tests/frame_tensor.rs` 与 `tests/frame_public_api.rs` 就是这次重写的回归网，一个断言都不改。

代价是每次推理多走一遍 76 800 个元素的循环、多一次 0.3 MB 分配，发生在一次 UNet 前向之前——可忽略。`mask_right`/`mask_bottom` 越界检查留在 `build_inner_image_planes` 内部，对两种 masking 都执行：`Keep` 用不到矩形，但让两条路径的准入完全一致比省一个分支更值。`into_values` 是给训练侧准备的，样本装配要把两块缓冲拼进一个 `Vec<f32>`，没有它就得多复制一次。

## 5. 单一 face crop 入口

`render_frame`（`frame.rs:338`）与 `render_planned_frame`（`burn.rs:157`）里各有一段一模一样的 `crop_bgr` + `resize_bilinear`。训练会是第三份。这类重复的失败模式很难查：训练与推理的 crop 一旦漂移，症状是「训练收敛正常但渲染发虚」，没人会先去看 crop。

```rust
pub fn build_face_crop(
    frame: &BgrFrame,
    bbox: &FaceBoundingBox,
    geometry: &RenderGeometry,
) -> Result<BgrFrame, InferenceError>;
```

实现就是 `resize_bilinear(&crop_bgr(frame, bbox)?, geometry.crop_size(), geometry.crop_size())`，两个调用点改为调它。

## 6. 音频窗口窄入口

`build_unet_audio_input(features, plan)` 只需要 `plan` 里的两个字段：`output_index`（用于越界检查）与 `audio_window`。训练侧没有 `InferenceFramePlan`——它是渲染计划的产物，带 `source_frame_index`/`reference_frame_index`，训练的参考帧由 `TrainingDataLoader` 决定，造一个假 plan 只为过一个检查是错的抽象方向。

```rust
pub fn build_unet_audio_window(
    features: &FeatureMatrix,
    audio_window: &[Option<usize>; 8],
) -> Result<UnetAudioInput, InferenceError>;
```

`build_unet_audio_input` 保留自己的 `plan.output_index < frame_count` 检查后转调新函数，两者共用抽出来的私有 `feature_frame_count(features)`（`dims == 1024`、`tokens > 0`、`tokens` 为偶数，否则 `InvalidFeatureShape`）。窗口本身仍由 `feathertalk_preprocess::audio_window_indices(index, frame_count)` 生成——迁移设计 §7.3 把它定为训练/推理/预览共用的公开数据契约，训练侧不得自己算 `i-4..i+3`。

## 7. 新 crate `feathertalk-training-data`

`rust/Cargo.toml` 的 `members` 在 `crates/feathertalk-export` 之后加入 `"crates/feathertalk-training-data"`。依赖：`burn.workspace`、`thiserror.workspace`，以及对 `feathertalk-training`、`feathertalk-inference`、`feathertalk-preprocess`、`feathertalk-project`、`feathertalk-audio` 的 path 依赖；dev 依赖 `tempfile.workspace`。workspace 的 `burn` 已经开了 `ndarray` 与 `autodiff` 特性，测试里用 `burn::backend::NdArray` 不需要额外后端依赖。模块划分 `error.rs`、`dataset.rs`、`batch.rs`。

为什么不直接扩 `feathertalk-training`：它现在是纯算法 crate，依赖只有 `burn`、`burn-store`、`hex`、`serde`、`serde_json`、`sha2`、`thiserror`。加上 project/inference/audio/preprocess 之后，改一行损失函数就要连带编译 JPEG 解码器与文件系统层，`cargo test -p feathertalk-training` 的反馈时间会成倍变长。判例是 `feathertalk-frame-pipeline`（契约）与 `feathertalk-frame-adapters`（实现）的既有分层。

## 8. 打开数据集与校验

```rust
impl ProjectTrainingDataset<JpegFrameReader> {
    pub fn open(project_dir: &Path) -> Result<Self, TrainingDataError>;
}

impl<R: FrameReader> ProjectTrainingDataset<R> {
    pub fn open_with_reader(project_dir: &Path, reader: R) -> Result<Self, TrainingDataError>;
    pub fn root(&self) -> &Path;
}
```

`FrameReader` 泛型化只为测试注入桩读取器（trait 已是 `Send + Sync`，签名 `read(&self, index: usize, path: &Path)`），生产路径用 `open` 即得 `JpegFrameReader`。不提供 inherent 的 `frame_count`——那会遮蔽 trait 方法，调用点得写 `TrainingDataset::frame_count(&ds)` 才能确定拿到哪个。

`open_with_reader` 的步骤：

1. `feathertalk_project::validate_project_dir(root)`，它已经要求 `state == Locked`，并检查 `REQUIRED_FILES`（`assets/video_25fps.mp4`、`assets/audio_16k_mono.wav`、`assets/features/feather_hubert.f32`，均非空）与 `REQUIRED_DIRS`（`assets/frames`、`assets/landmarks`），且每一段路径都拒绝符号链接。训练不做第二套目录校验。
2. `frame_count = manifest.frame_count`，要求 ≥ 1，`usize::try_from` 转换，并对 `2 * frame_count` 做溢出检查。
3. `read_feature_file(root/assets/features/feather_hubert.f32)`，要求 `dims == 1024 && tokens == 2 * frame_count`。清单里的 `feature_shape` 是 `[frame_count, 2, 1024]`，加锁命令已经保证过一致，这里再查一遍是防手工改文件——训练是长任务，第 3 个 epoch 才发现令牌数不对的成本远高于开工前查一次。

状态里存 root、`frame_count`、`FeatureMatrix`、reader、`CropSpec`、`MouthRoiSpec`、`RenderGeometry::standard()`。

逐样本的读取：帧路径 `assets/frames/{index:06}.jpg` 交给 `reader.read(index, &path)`，读回后按清单的 `frame_width`/`frame_height` 校验尺寸（与 `inference/src/executor.rs` 的做法一致）；关键点 `assets/landmarks/{index:06}.lms` 走 `read_landmarks` → `compute_face_bbox` → `build_face_crop`。路径格式与 `executor.rs:333-339`、`frame-pipeline/src/model.rs:94,98` 相同，不新增常量。

特征矩阵常驻内存：188 帧约 1.5 MB，`MAX_FEATURE_FILE_BYTES = 512 MB` 给出了最坏情形的上界，而每个样本要随机访问 8 个不连续的令牌槽，mmap 或窗口式读取只会把随机 I/O 放大。范围外。

## 9. 样本装配

`TrainingDataset` 只有一个 `Item` 关联类型，而两种采样方式产出的东西结构不同，所以 `Item` 是枚举：

```rust
pub enum TrainingItem {
    SingleFrame(FrameSample),
    TemporalPair { first: FrameSample, second: FrameSample },
}

pub struct FrameSample { /* 私有 Vec<f32>：image 6*160*160、audio 16*32*32、target 3*160*160、mouth_mask 1*160*160 */ }
```

访问器 `image()` / `audio()` / `target()` / `mouth_mask()` 返回 `&[f32]`。私有字段加访问器与 `BgrFrame`、`FeatureMatrix`、`UnetImageInput` 同构：缓冲区长度是不变量，公开字段等于允许调用方破坏它。

每个 `FrameSample` 的四块数据：

| 字段 | 来源 |
| --- | --- |
| `image` | `Keep(参考帧 crop)` ⧺ `Blackout(当前帧 crop)`，共 6 通道 |
| `target` | `Keep(当前帧 crop)`，3 通道未遮挡内圈 |
| `mouth_mask` | `mouth_roi_rect(当前帧关键点, …)` 矩形内 1.0，其余 0.0 |
| `audio` | `build_unet_audio_window(features, audio_window_indices(target_index, frame_count))` |

`TemporalPair` 的两个样本共用同一个参考帧（Python 的 `_build_frame(idx, ref_T)` 就是把 `ref_T` 传进去复用，迁移设计 §8.1 同款）：参考帧的 `Keep` 平面只构造一次，第二个样本克隆缓冲区。省下的是一次 JPEG 解码加一次 crop/resize，而不只是一次内存拷贝。

单个 `FrameSample` 约 1.09 MB f32。不做帧缓存——Python 侧每个 item 都 `cv2.imread`，我们先对齐这个行为，缓存策略要连同 batch 大小与磁盘吞吐一起量，属范围外。

## 10. 缩放滤波器的已知偏差

face crop 的缩放用 `feathertalk-inference::resize_bilinear`（C++ 参考实现的两抽样双线性：`source_x = (x + 0.5) * scale - 0.5`、夹紧取整、`f32::round` 到 u8 等价于 C++ `std::lround`，见 `2026-08-26-bgr-frame-kernel-design.md` §4.2），**不用** `feathertalk-image::resize_area`（与 OpenCV `INTER_AREA` 逐字节一致，SCRFD/PFLD 适配器在用）。

这是一处需要显式记账的偏差。Python 的 `face_utils.crop_face` 用 `cv2.INTER_AREA` 生成训练输入，而我们的渲染路径用双线性做推理——上游本身就带着训练/服务不一致。两害相权：让训练与我们自己的渲染路径一致，消除的是我们自己引入的偏差；照抄 `INTER_AREA` 则是把上游的偏差复制进来。

被否决的替代方案：训练用 `INTER_AREA`、推理用双线性。它精确复现了 Python 的偏差，还要给训练 crate 加 `feathertalk-image` 依赖，并让同一条流水线里同时存在两个缩放实现。

被否决的替代方案：把渲染路径改成 `INTER_AREA`。那会破坏与 C++ 参考实现的逐位对齐（现有校验依赖它），远超本切片范围。

残余风险记录在此：微调一个 Python 训练出的检查点时，输入分布与它原本见过的略有差异。迁移设计 §14.1 的黄金基线覆盖「固定 face crop、landmarks、mask 与 audio window」，不覆盖训练缩放滤波器，所以这条偏差不会被现有基线捕获，只能靠本节的文字与后续的收敛曲线观察。

## 11. 批次堆叠

```rust
pub struct SingleFrameBatch<B: Backend> {
    pub image: Tensor<B, 4>,       // [N, 6, 160, 160]
    pub audio: Tensor<B, 4>,       // [N, 16, 32, 32]
    pub target: Tensor<B, 4>,      // [N, 3, 160, 160]
    pub mouth_mask: Tensor<B, 4>,  // [N, 1, 160, 160]
}

pub struct TemporalBatch<B: Backend> {
    pub image: Tensor<B, 4>,       // [2N, 6, 160, 160]
    pub audio: Tensor<B, 4>,       // [2N, 16, 32, 32]
    pub target: Tensor<B, 5>,      // [N, 2, 3, 160, 160]
    pub mouth_mask: Tensor<B, 5>,  // [N, 2, 1, 160, 160]
}

pub fn stack_single_frame_batch<B: Backend>(
    items: &[TrainingItem],
    device: &B::Device,
) -> Result<SingleFrameBatch<B>, TrainingDataError>;

pub fn stack_temporal_batch<B: Backend>(
    items: &[TrainingItem],
    device: &B::Device,
) -> Result<TemporalBatch<B>, TrainingDataError>;
```

张量字段公开，判例是 `LossBreakdown`：批次是纯数据载体，没有跨字段不变量需要保护（形状由构造函数保证，`Tensor` 自身不可变）。

每个字段先拼一个连续的 `Vec<f32>`，再一次 `Tensor::from_data(TensorData::new(values, shape), device)`。不逐样本建张量后 `Tensor::cat`：那是 N 次分配加一次拷贝，且 `cat` 在 autodiff 后端上会进计算图。

扁平化顺序是**样本主序**（`s0_first, s0_second, s1_first, …`），不是「所有 first 再所有 second」。理由是 `temporal_loss` 内部对 5-D 张量做 `reshape([batch * pair_len, channels, height, width])`（`losses.rs:196-202`），行主序下这个展开等价于索引 `b * pair_len + p`。图像与音频保持 4-D，因为 `TalkingHeadModel::forward_talking_head(Tensor<B,4>, Tensor<B,4>) -> Tensor<B,4>`；把预测 `[2N,3,160,160]` 变回 `[N,2,3,160,160]` 是切片 B 里 worker 的一行 `reshape`，放在这里等于让数据层猜模型的调用方式。

`target` 与 `mouth_mask` 直接堆成 5-D，是因为 `temporal_loss` 的签名要 5-D，而它们不经过模型。

错误：空切片、`items` 里出现不匹配的变体（单帧堆叠器收到 `TemporalPair` 或反之）、形状乘法溢出，全部走 `TrainingDataError::Batch`。

## 12. 错误分类

`TrainingDataError` 用 `thiserror`，8 个变体。凡是上游错误不透明的地方就带一个预渲染的 `message: String`，这是 `PipelineError::FrameUndecodable` 已经确立的做法——保留 `#[source]` 会把 `feathertalk-project`/`audio`/`inference` 的错误类型全部漏进公开签名。

| 变体 | 携带 |
| --- | --- |
| `Project` | `path`, `message` |
| `Features` | `path`, `message` |
| `FeatureShape` | `path`, `expected_tokens`, `actual_tokens`, `dims` |
| `FrameIndexOutOfRange` | `index`, `frame_count` |
| `Frame` | `index`, `path`, `message` |
| `Landmarks` | `index`, `path`, `message` |
| `Sample` | `index`, `message` |
| `Batch` | `message` |

`FeatureShape` 是唯一带结构化数字的变体：令牌数不匹配是最容易发生的人为错误（改了帧数没重跑特征），期望值与实际值必须能直接读出来，不该埋在字符串里。

`impl From<TrainingDataError> for TrainingError` 映射到 `TrainingError::InvalidInput(String)`（经 `to_string()`），这样 `load_sample` 的实现体是一行 `?`。为什么不加新变体：`TrainingError::Io` 是 `Io(#[from] std::io::Error)`，无法承载字符串化的错误链；而所有数据集失败在训练循环眼里都是同一件事——「第 N 个样本造不出来，去修素材」，这正是 `TrainingDataLoader` 已经用 `InvalidInput` 表达的语义。切片 B 的 worker 侧可以在错误映射时把它细化成更具体的错误码。

## 13. 测试

`feathertalk-preprocess`：`mouth_roi_rect` 的正常路径（手算矩形，逐值断言）；`min_w`/`min_h` 下限生效；左上与右下两侧的夹紧；退化输入下 `rx2 ≥ rx1 + 1`；一个恰好落在半整数上的取整用例，专门锁住 `round_ties_even`（用 `f32::round` 时必须失败）；§3 表格里每一条拒绝条件各一个用例。

`feathertalk-inference`：`build_inner_image_planes` 的 `Keep`/`Blackout` 取值与形状；非 168×168 crop 被拒；`build_unet_image_input` 的既有断言全部保持通过（这就是重写的验收标准）；`build_face_crop` 与两个调用点的输出一致；`build_unet_audio_window` 的零填充槽与越界槽。

`feathertalk-training-data`：单元测试用桩 `FrameReader`——它忽略路径、按索引生成确定的 BGR 渐变，于是**不需要任何 JPEG 夹具**，只有 `.lms` 文件必须真实存在（`read_landmarks` 要读文件）。集成测试在 `tempfile` 目录里合成一个加锁项目，覆盖：未加锁的清单被拒、令牌数不匹配、缺帧、缺关键点、`.lms` 内容损坏、帧索引越界。批次测试在 `burn::backend::NdArray` 上验证两种批次的形状与样本主序。

夹具配方（照 `crates/feathertalk-project/tests/support/mod.rs` 的形状，它已有 `locked_manifest()`/`preparing_manifest()`/`valid_project()`，在新 crate 里复制一份自己的 `tests/support/mod.rs`）：`create_dir_all assets/{frames,landmarks,features}`；给 `assets/video_25fps.mp4` 与 `assets/audio_16k_mono.wav` 写**非空**占位字节（`validate_artifacts` 拒绝零长度文件）；用 `write_feature_file` 写真实载荷，`tokens = 2 * frame_count`、`dims = 1024`；`write_project_manifest_atomic` 之后 `lock_asset_package(dir, manifest)`。清单取 `AssetManifest { schema_version: 1, state: Locked, video_fps: 25, audio_sample_rate: 16_000, audio_channels: 1, frame_count, frame_width, frame_height, feature_type: FeatherHubert, feature_shape: [frame_count, 2, 1024], landmark_model_sha256: "a".repeat(64), feature_model_sha256: "b".repeat(64) }`。关键点文件是 110 行 `"{x} {y}"`，坐标要让 bbox（`pt[1].x`、`pt[52].y`、`pt[31].x`，正方形）稳稳落在合成帧内部，例如 256×256 帧配 xmin 40 / xmax 200 / ymin 60。

如果将来某个用例确实需要真实 JPEG，复用 `crates/feathertalk-frame-adapters/tests/fixtures/demo_frame_v1/frame.jpg`，跨 crate 引用的判例是 `worker/tests/models.rs:10`。本切片不新增二进制夹具入库。

## 14. 验证

在 `rust/` 下执行，要求零告警零失败：

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace --all-targets`
- `cargo test --release -p feathertalk-cli --test real_worker`（门控变量齐备时；本切片不碰 worker，此门只作回归确认）
- `git diff --check`

## 15. 范围外

- 帧缓存与预取。见 §9，需要连同 batch 大小与磁盘吞吐一起测量。
- 特征文件的 mmap 或窗口式读取。见 §8。
- `train` worker 命令、阶段/进度/取消、CLI 子命令、门控端到端。切片 B。
- VGG19 模型包的配置与握手宣告。切片 B。
- 优化器循环、检查点写出、指标与预览产物。`feathertalk-training` 已有这些能力，把它们串起来属于切片 B。
- 两套 `TrainingMode` 枚举的映射。切片 B。
- 改动渲染路径的缩放滤波器。见 §10。
- wenet 特征（`128,16,32`）。当前只支持 FeatherHubert 的 `[2,1024]`。
- 数据增强（翻转、色彩抖动）。Python 侧也没有，加进来会改变数值基线。
- 多项目合并数据集。`TrainingSample` 只有单一索引空间，跨项目需要先定义索引映射。
