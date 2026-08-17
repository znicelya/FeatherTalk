# FeatherTalk Rust 桌面产品迁移设计

日期：2026-08-17  
状态：已确认，等待书面规格复核  
目标平台：Windows、macOS、Linux  

## 1. 目标

将 FeatherTalk 从研究型 Python/C++ 仓库迁移为可离线交付的商业桌面产品。最终产品使用 Rust + GPUI，覆盖素材包制作、预处理、FeatherHuBERT 特征提取、UNet 训练、模型管理、离线推理和视频合成。

最终发布物满足以下条件：

- 产品运行、训练和命令行工具均不依赖 Python。
- 所有 FeatherTalk 业务逻辑和模型逻辑均使用 Rust 实现。
- 模型训练和推理统一使用 Burn，GPU 后端统一使用 WGPU。
- 允许由 Rust 封装和分发经过许可审计的原生依赖，包括 FFmpeg 和操作系统 GPU 驱动。
- 支持导入现有 `.pth` 和 `.pth.tar` 权重，但新模型以 `safetensors` 为标准权重格式。
- 桌面端面向不熟悉机器学习的内容制作人员，高级训练参数仍对专业用户开放。

## 2. 范围

### 2.1 纳入范围

- GPUI 桌面工作台。
- 项目、素材包、模型、任务和输出管理。
- 视频标准化、音频提取、抽帧和视频合成。
- SCRFD 人脸检测。
- PFLD 人脸关键点检测。
- FeatherHuBERT 固定特征提取。
- Original UNet 和 MobileOne UNet。
- Baseline、Mouth ROI、Mouth ROI + Temporal 三种训练模式。
- Adam 优化器、checkpoint、停止并保存、断点恢复。
- 旧 PyTorch 权重的受限安全导入。
- 标准 FeatherTalk 模型包和 ONNX 兼容导出。
- Windows、macOS、Linux 安装包和离线运行。

### 2.2 明确排除

- Wenet 及 `data_utils/wenet/` 下的训练、识别、对齐和导出工具。
- 原始 HuBERT 模型和 Hugging Face 下载链路。
- FeatherHuBERT 蒸馏训练。
- SaaS、云端训练、账号、订阅和在线素材同步。
- 首版对 AMD/Intel GPU 训练性能作出保证。
- 对任意 PyTorch pickle 文件提供通用执行或反序列化能力。

## 3. 已有实现基线

迁移基线来自当前仓库：

- `data_utils/process.py`：视频预处理入口。
- `data_utils/detect_face.py`：SCRFD 人脸检测。
- `data_utils/get_landmark.py`、`data_utils/pfld_mobileone.py`：PFLD 关键点检测。
- `data_utils/feather_hubert/feather_hubert.py`：FeatherHuBERT 前向和特征切片。
- `face_utils.py`：人脸裁剪、嘴部遮挡和音频窗口。
- `unet.py`、`unet_mobileone.py`：两种 UNet。
- `train.py`、`train_mouth_roi_loss.py`、`train_mouth_roi_temporal_loss.py`：三种训练流程。
- `inference.py`、`dihuman_run.py`：离线和流式推理行为参考。
- `pth2onnx.py`、`FeatherTalk-CPP/tools/export_models.py`：ONNX 接口参考。
- `FeatherTalk-CPP/src/main.cc`：独立推理、图像贴回和 FFmpeg 管线参考。

Python 实现只在迁移期间作为数值基准保留。所有验收通过后，从产品源码和发布链路中删除 Python 依赖及旧 C++ 运行器。

## 4. 核心技术决策

### 4.1 模型栈

- 模型定义：Burn。
- GPU 训练与推理：Burn `Wgpu` backend。
- 自动微分训练：Burn `Autodiff<Wgpu>`。
- CPU 验证和无 GPU 回退：Burn `NdArray` backend。
- 权重标准格式：Safetensors。
- 模型交换格式：ONNX opset 17。
- 桌面 UI：GPUI。

WGPU 根据平台选择 DX12、Metal 或 Vulkan。应用必须显示实际 adapter、backend 和显存信息，不允许将 GPU 请求静默回退到 CPU。

### 4.2 进程隔离

桌面 UI 和计算任务使用两个 Rust 进程：

```text
feathertalk-desktop
  GPUI、项目状态、任务队列、进度、日志、错误展示
            |
            | versioned JSON Lines RPC over stdio
            v
feathertalk-worker
  预处理、特征提取、训练、推理、导入、导出
```

UI 进程不持有 Burn 模型、WGPU device 或 FFmpeg 管线。worker 崩溃、GPU device lost 或显存不足不能导致桌面进程退出。

协议具备以下属性：

- 每条请求包含 `protocol_version`、`task_id`、命令和参数。
- 每条事件包含 `task_id`、阶段、进度、时间和可选指标。
- 取消请求是幂等操作。
- worker 启动时报告版本、支持的 backend、adapter 和功能列表。
- 协议版本不兼容时，桌面端拒绝启动任务并显示可操作错误。

### 4.3 Rust workspace

```text
crates/
  app/          GPUI 桌面端
  worker/       后台任务进程和 RPC 服务
  domain/       项目、任务、模型、错误、进度类型
  media/        FFmpeg、WAV、视频帧和图像读写
  preprocess/   抽帧、人脸检测、关键点和素材包验证
  audio/        FeatherHuBERT、波形处理和特征窗口
  models/       Burn 模型定义
  training/     数据集、损失、优化器和 checkpoint
  inference/    UNet 推理、图像贴回和视频合成
  weights/      PyTorch 权重导入和 safetensors
  export/       部署包和 ONNX opset 17 导出
  cli/          与 worker 能力一致的命令行入口
```

每个 crate 只通过 `domain` 中的版本化类型交换数据。`app` 不依赖 `models`、`training` 或模型计算使用的 WGPU crate。

### 4.4 首发平台矩阵

```text
Windows 11 23H2+ x86_64:
  CPU；NVIDIA GPU 使用 DX12

macOS 14+ Apple Silicon:
  CPU；Apple GPU 使用 Metal

Ubuntu 22.04/24.04 x86_64:
  CPU；NVIDIA GPU 使用 Vulkan
```

AMD 和 Intel GPU adapter 可以在界面中显示并进入实验性检测，但首发版不将其训练性能或兼容性列为发布承诺。Intel Mac 不在首发支持矩阵内。

## 5. 模型和权重契约

### 5.1 FeatherHuBERT

输入和输出：

```text
waveform: [batch, samples], 16 kHz float32
hidden:   [batch, tokens, 1024]
```

Rust 实现必须复现以下行为：

- 波形转为单声道并重采样到 16 kHz。
- 使用 `(x - mean) / sqrt(var + 1e-7)` 归一化。
- HuBERT 风格七层 valid Conv1d frontend。
- frontend kernel：`[10, 3, 3, 3, 3, 2, 2]`。
- frontend stride：`[5, 2, 2, 2, 2, 2, 2]`。
- GroupNorm、GELU、Depthwise TCN、dilation 循环 `[1, 2, 4, 8]`。
- 默认输出维度为 1024，具体 channels、blocks、expansion 和 dropout 从模型 manifest 读取。
- token 数与 400 sample kernel、320 sample stride 的 HuBERT 帧数规则一致。
- 长音频按当前 Python 边界规则分块，拼接后裁剪或填充至精确 token 数。
- 奇数 token 删除，最终特征 reshape 为 `[video_frames, 2, 1024]`。

FeatherHuBERT 在产品中始终处于 eval 模式，不提供蒸馏训练入口。

### 5.2 UNet

固定接口：

```text
image:  [batch, 6, 160, 160]
audio:  [batch, 16, 32, 32]
output: [batch, 3, 160, 160]
```

`models` 提供 Original 和 MobileOne 两个实现，并通过同一 trait 暴露：

```rust
pub trait TalkingHeadModel<B: burn::tensor::backend::Backend> {
    fn forward(
        &self,
        image: burn::tensor::Tensor<B, 4>,
        audio: burn::tensor::Tensor<B, 4>,
    ) -> burn::tensor::Tensor<B, 4>;
}
```

Original UNet 必须保持当前通道 `[32, 64, 128, 256, 512]`、depthwise inverted residual、bilinear resize、skip connection、音频 bottleneck 和 sigmoid 输出。

MobileOne UNet 必须同时支持训练图和重参数化推理图。重参数化结果需通过独立数值测试，不允许只比较最终视频主观效果。

### 5.3 人脸模型

- SCRFD 使用当前 `scrfd_2.5g_kps.onnx` 作为来源，通过 Burn ONNX 导入链路生成受版本控制的模型定义和标准权重。
- PFLD 使用当前 `checkpoint_epoch_335.pth.tar` 作为来源，由受限 Rust 导入器转换为 safetensors，再由 Burn 模型执行。
- 发布模型必须带来源、版本、许可证标识和 SHA-256。

### 5.4 旧权重导入

`weights` crate 只解析 FeatherTalk 已知 checkpoint 结构，不执行 pickle 中的任意 Python global 或 reduce callable。

支持的旧结构：

- 直接 state dict。
- 包含 `model`、`epoch`、`config` 或 `args` 的映射。
- 当前 PFLD `.pth.tar`。

导入结果：

- 模型权重转为 safetensors。
- epoch 和模型配置写入 manifest。
- 旧优化器状态不迁移。
- 未知 key、重复 tensor、shape 不匹配、非预期 dtype 或架构不匹配均视为失败。

新 Rust checkpoint 完整保存模型、优化器、epoch、global step、随机种子和训练配置，保证后续 Rust 训练可准确恢复。

### 5.5 标准模型包

```text
model-package/
  manifest.json
  model.safetensors
  optimizer.safetensors      # 仅训练 checkpoint
  training-state.json        # 仅训练 checkpoint
  LICENSES.json
```

`manifest.json` 至少包含：

- schema version。
- 模型类型和架构版本。
- FeatherHuBERT 配置或 UNet variant。
- 输入输出名称、shape 和 dtype。
- 训练模式及 loss 参数。
- 来源和创建时间。
- 每个文件的 SHA-256。
- 所需最低应用版本。

### 5.6 ONNX 导出

ONNX 是兼容输出，不是应用内部运行依赖。`export` crate 使用 Rust protobuf 类型生成固定模型图和 initializer，输出 opset 17：

```text
FeatherHuBERT:
  waveform [1, samples] -> hidden [1, tokens, 1024]

UNet:
  input [1, 6, 160, 160]
  audio [1, 16, 32, 32]
  -> output [1, 3, 160, 160]
```

导出后必须在 Rust 中完成 protobuf 结构检查，并通过 Rust `ort` 兼容验证工具执行一次参考输入推理对比。`ort` 只用于 ONNX 兼容验证，不参与产品的日常训练或推理。MobileOne 默认导出重参数化推理图。

## 6. 项目和素材包格式

### 6.1 项目目录

```text
project/
  project.json
  source/
    input.mp4
  assets/
    video_25fps.mp4
    audio_16k_mono.wav
    frames/
      000000.jpg
    landmarks/
      000000.lms
    features/
      feather_hubert.f32
    assets.json
  models/
    feather_hubert/
    unet/
      checkpoint-000120/
      last/
  outputs/
    preview/
    renders/
```

`project.json` 记录项目 ID、显示名称、schema version、当前素材包、默认模型和任务历史索引。

`assets.json` 记录：

```json
{
  "schema_version": 1,
  "video_fps": 25,
  "audio_sample_rate": 16000,
  "audio_channels": 1,
  "frame_count": 0,
  "frame_width": 0,
  "frame_height": 0,
  "feature_type": "feather_hubert",
  "feature_shape": [0, 2, 1024],
  "landmark_model_sha256": "",
  "feature_model_sha256": ""
}
```

字段中的 `0` 由实际素材值替换。空哈希只允许出现在尚未完成的临时 manifest 中；已锁定素材包必须拥有完整哈希。

### 6.2 兼容格式

内部特征使用带版本 header 的 little-endian float32 文件，避免将 NPY 作为运行时依赖。CLI 和模型页提供 `.npy` 导入导出，用于迁移现有 `aud_hu.npy` 数据。

关键点继续支持每行 `x y` 的 `.lms`，同时由 `assets.json` 校验帧数和 landmark 点数。

## 7. 数据处理契约

### 7.1 预处理

1. 校验输入媒体、磁盘空间和输出权限。
2. 使用随应用分发的 FFmpeg 将视频标准化为恒定 25 FPS。
3. 提取 16 kHz 单声道 PCM WAV。
4. 按六位数字文件名提取 JPEG 帧。
5. 对每帧执行 SCRFD 和 PFLD。
6. 标记无人脸、多张脸、bbox 越界、关键点异常和模糊帧。
7. 用户处理异常帧后锁定素材包。
8. 执行 FeatherHuBERT 并写入特征文件。
9. 写入完整 manifest 和哈希后原子提交素材包。

源视频不要求预先为 25 FPS；产品负责标准化。所有视频帧和音频时间戳以标准化产物为准。

### 7.2 人脸 crop

保持当前行为：

- `xmin = landmark[1].x`。
- `ymin = landmark[52].y`。
- `xmax = landmark[31].x`。
- bbox 高度等于 `xmax - xmin`。
- face crop resize 为 `168 x 168`。
- 去除四周 4 像素，网络区域为 `160 x 160`。
- 嘴部遮挡矩形为 `x=5, y=5, width=150, height=145`。
- 图像张量为 CHW float32，范围 `[0, 1]`，通道顺序与当前 BGR 基准保持一致。

### 7.3 音频窗口

单帧特征为 `[2, 1024]`。训练和推理以当前视频帧为中心，取 `[i-4, i+3]` 共 8 帧，越界位置填零：

```text
[8, 2, 1024] -> [16, 32, 32]
```

该窗口规则是公开数据契约，训练、离线推理和预览必须复用同一实现。

## 8. 训练设计

### 8.1 数据采样

- Baseline 和 Mouth ROI：当前帧作为 target，随机另一帧作为 reference。
- Temporal：相邻帧共享同一个随机 reference，避免 reference 差异污染 temporal loss。
- 随机种子保存在训练状态中。
- DataLoader 的 shuffle 顺序可从 checkpoint 恢复。

### 8.2 损失

Baseline：

```text
L = L1(full) + 0.01 * MSE(VGG19 conv3_3)
```

Mouth ROI：

```text
L = L1(full)
  + mouth_weight * L1(mouth_roi)
  + perceptual_weight * MSE(VGG19 conv3_3)
```

Mouth ROI + Temporal：

```text
L = L1(full)
  + mouth_weight * L1(mouth_roi)
  + temporal_weight * L1(frame_delta)
  + temporal_mouth_weight * L1(mouth_delta)
  + perceptual_weight * MSE(VGG19 conv3_3)
```

默认值保持当前 Python 行为：

- `mouth_weight = 4.0`。
- `temporal_weight = 0.5`。
- `temporal_mouth_weight = 4.0`。
- `perceptual_weight = 0.01`。
- `temporal_stride = 1`。

VGG19 仅作冻结特征提取器。其权重必须作为独立受许可模型包分发并记录哈希。

### 8.3 训练预设

- 快速：Baseline。
- 嘴部增强：Mouth ROI。
- 时序稳定：Mouth ROI + Temporal。

高级面板暴露 batch size、learning rate、epoch、保存频率、loss 权重、mouth ROI 参数、temporal stride 和随机种子。

### 8.4 checkpoint

- 每个 epoch 更新 `last`。
- 按配置保留周期 checkpoint。
- “停止并保存”只在完成当前安全边界后退出。
- GPU OOM 时保存最近已完成的 checkpoint，不自动修改 batch size 后继续。
- 恢复时校验素材包哈希、模型架构、optimizer schema 和训练配置。

## 9. 推理和视频合成

- 驱动音频标准化并提取 FeatherHuBERT 特征。
- 素材帧按 `0,1,2,...,N-1,N-2,...,1,0,...` 往返选择。
- 推理 reference 使用当前帧。
- UNet 输出替换 `168 x 168` face crop 内部的 `160 x 160` 区域。
- face crop resize 回原 bbox 后贴入原始帧。
- FFmpeg 接收 raw frame 流，编码视频并与驱动音频合成。
- 输出帧率固定为 25 FPS，音频采用 `-shortest` 等价行为。
- 完整渲染前支持短时长预览，预览和完整渲染使用同一管线。

## 10. 桌面工作台

首屏为项目工作台，主导航固定为：

```text
素材 | 训练 | 生成 | 模型 | 任务
```

### 10.1 素材

- 导入人物视频。
- 显示时长、帧率、分辨率、音轨、响度和人脸可见率。
- 启动视频标准化、抽帧、关键点和特征提取。
- 时间轴标记异常帧。
- 预览关键点 overlay，排除坏帧或重新检测。
- 验证通过后锁定素材包版本。

### 10.2 训练

默认显示 UNet variant、训练预设、设备、预计显存、epoch 和保存路径。高级面板显示完整参数。

训练期间显示：

- 当前 epoch 和 step。
- 总 loss 和各分量。
- 每秒样本数和预计剩余时间。
- 固定样本预测、target 和嘴部 ROI。
- 当前显存和 worker 状态。

### 10.3 生成

- 选择素材包、checkpoint 和驱动音频。
- 选择输出尺寸、编码器、质量和路径。
- 生成短预览或完整视频。
- 完成后播放视频或打开输出目录。

### 10.4 模型

- 导入旧 `.pth/.pth.tar`。
- 显示模型类型、参数量、shape、哈希和兼容状态。
- 导出标准模型包或 ONNX。
- 拒绝错误架构和配置不匹配的权重。

### 10.5 任务

- 显示排队、运行、完成、失败和取消历史。
- 一个 GPU adapter 同时只运行一个训练或推理任务。
- 应用退出时，训练任务先执行停止并保存。

## 11. 任务状态和错误

任务事件：

```text
Queued
Preparing
ExtractingAudio
ExtractingFrames
DetectingFaces
ExtractingFeatures
Training { epoch, step, loss }
Exporting
Rendering { frame, total }
Completed
Failed { code, message }
Cancelled
```

稳定错误码：

```text
MEDIA_INVALID
FACE_NOT_FOUND
LANDMARK_INVALID
FEATURE_SHAPE_MISMATCH
MODEL_INCOMPATIBLE
GPU_OUT_OF_MEMORY
GPU_DEVICE_LOST
DISK_SPACE_LOW
WORKER_CRASHED
TASK_CANCELLED
```

错误处理原则：

- 底层 Rust panic、WGPU debug 文本和 FFmpeg 命令行不直接作为用户提示。
- 每个错误包含用户可读摘要、技术详情、任务阶段和可恢复建议。
- 坏帧定位到具体帧，允许排除或重跑。
- GPU device lost 后重启 worker，并允许从最近 checkpoint 恢复。
- worker 崩溃时 GPUI 保持运行并保存最后日志。
- 任务开始前估算磁盘空间，运行中持续检查。
- 日志、诊断和崩溃报告默认仅保存本地。

## 12. 原子性和恢复

- 每个阶段写入项目内临时目录。
- 文件写完后执行 fsync、哈希校验和原子重命名。
- manifest 最后写入；存在完整 manifest 才表示阶段完成。
- 重跑任务时按输入哈希、模型哈希和配置判断是否复用产物。
- 取消任务保留最近完整 checkpoint，删除未完成临时文件。
- 应用启动时扫描并提供恢复或清理未完成任务的操作。

## 13. 商业分发约束

- 标准安装包不要求用户安装 Python、Rust、FFmpeg 或模型运行环境。
- FFmpeg 使用经过审计的 LGPL 兼容构建；标准构建不包含未获商业许可的 GPL codec。
- 每个平台生成机器可读的第三方许可证清单和人工可读的 Notices。
- 应用许可证和模型许可证分开记录。
- 外部模型权重必须记录来源、许可、版本和哈希，未确认商业使用权的权重不能进入正式安装包。
- worker 和模型包均进行代码签名或哈希签名验证。
- `.pth` 导入器按不可信输入处理，限制文件大小、tensor 数量、总分配和嵌套深度。
- 产品默认离线，无遥测和网络请求。

Apache-2.0 仓库许可证允许商业使用，但不自动覆盖外部模型、数据集、人物肖像、字体、FFmpeg codec 或操作系统 SDK 的授权。

## 14. 验证策略

### 14.1 Golden 基准

迁移开始前固定一组最小基准：

- 短 WAV 和完整 FeatherHuBERT 中间输出。
- 固定 face crop、landmarks、mask 和 audio window。
- Original UNet 和 MobileOne UNet 固定输入输出。
- 三种 loss 的输入、标量输出和关键梯度。
- 两秒端到端测试视频。

Python 只用于生成这些不可变基准。最终 Rust 测试读取固化数据，不在 CI 中启动 Python。

### 14.2 数值门槛

```text
权重导入：
  tensor 名称、shape、dtype 和数量全部一致

CPU float32 前向：
  max_abs_error <= 1e-4

WGPU 前向：
  max_abs_error <= 1e-3
  图像 SSIM >= 0.999

损失与梯度：
  relative_error <= 1e-3

端到端视频：
  帧数一致
  音画偏差 <= 20 ms
  无缺帧、花帧和静默 CPU 回退

断点恢复：
  恢复后的下一训练 step 与连续训练在同一容差内等价
```

### 14.3 测试层级

- 单元测试：音频归一化、token 数、分块边界、零填充、crop、mask、ROI 和帧选择。
- 模型测试：FeatherHuBERT、Original UNet、MobileOne UNet 的前向、反向和重参数化。
- 流程测试：固定两秒视频执行素材制作、训练 smoke test、推理和合成。
- CPU CI：Windows、macOS、Linux 每次提交执行。
- GPU 测试：Windows DX12、macOS Metal、Linux Vulkan 定期执行。
- 安装测试：三平台在无 Python、无系统 FFmpeg 的干净环境中运行完整 smoke test。

## 15. 实施分解

该迁移不使用一份巨型实现计划。每个里程碑拥有独立规格、实现计划和验收。

### 15.1 里程碑一：Burn 可行性闭环

- 建立 Rust workspace 和测试夹具。
- 实现受限 `.pth` tensor 读取。
- 导入一份 FeatherHuBERT 和 UNet 权重。
- 实现 FeatherHuBERT 前向。
- 实现 Original UNet 前向。
- 完成 Original UNet 单步反向和 Adam 更新。
- 在 CPU 和至少一个 WGPU backend 达到数值门槛。

该里程碑是路线 3 的继续条件。未达到数值门槛时，不进入 GPUI 开发。

### 15.2 里程碑二：素材包与预处理

- 媒体标准化和项目格式。
- SCRFD 和 PFLD。
- 异常帧检测和素材包校验。
- FeatherHuBERT 完整长音频特征提取。
- 特征和 manifest 的原子写入。

### 15.3 里程碑三：完整训练

- Original 和 MobileOne UNet。
- VGG19 感知损失。
- Baseline、Mouth ROI、Temporal。
- DataLoader、随机状态、checkpoint 和恢复。
- 训练指标和预览产物。

### 15.4 里程碑四：推理与模型工具

- 离线推理和视频合成。
- MobileOne 重参数化。
- 标准模型包。
- ONNX opset 17 导出和校验。
- 旧模型和旧特征迁移 CLI。

### 15.5 里程碑五：GPUI 工作台

- 项目、素材、训练、生成、模型和任务页面。
- worker RPC、进度、取消、恢复和日志。
- 新手预设和高级参数。
- 本地播放器、文件选择和输出管理。

### 15.6 里程碑六：商业发布

- 三平台安装包、签名和升级。
- 原生依赖和模型许可证审计。
- 离线安装和干净环境 smoke test。
- worker 崩溃、断电、磁盘不足和 GPU device lost 恢复测试。
- 删除仓库中的 Python 源码、`requirements.txt`、Wenet vendored 目录和旧 C++ 运行器；历史实现仅保留在 Git 历史中。

## 16. 完成定义

只有同时满足以下条件，才视为“全部迁移到 Rust”：

- FeatherHuBERT 特征提取与当前基准一致。
- 两种 UNet 的训练、恢复、推理和重参数化达到数值门槛。
- 素材包制作、三种训练模式、生成和模型管理均可从 GPUI 完成。
- CLI 覆盖所有 worker 能力，便于自动化测试和无界面运行。
- 三平台安装包无需 Python 和系统 FFmpeg。
- 产品运行路径不引用 Wenet、原始 HuBERT 或 FeatherHuBERT 蒸馏训练。
- 仓库默认分支不再包含 Python 源码、Wenet vendored 目录或旧 C++ 运行器。
- 所有发布依赖、模型和 codec 均有许可证清单和来源记录。
- 端到端测试、GPU 测试和安装测试通过。
