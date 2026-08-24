# MobileOne UNet 训练图与重参数化推理图设计

日期：2026-08-24  
状态：已确认

## 1. 目标

在 `feathertalk-models` 中实现与 Python `unet_mobileone.py` 对齐的 Burn MobileOne UNet，并提供彼此独立的训练态与推理态类型：

- `MobileOneUnet<B>` 保留多分支 MobileOne 训练图，可参与反向传播和 checkpoint 保存；
- `MobileOneUnetInference<B>` 只保留融合后的带 bias 单卷积，可用于后续离线推理、模型包与 ONNX 导出；
- `MobileOneUnet::reparameterize()` 从训练态模型显式生成推理态模型，不修改源模型；
- CPU float32 下训练图与推理图的固定输入前向误差满足 `max_abs_error <= 1e-4`。

本切片只建立模型结构、反向能力和重参数化等价性，不实现 Python checkpoint 导入、训练 DataLoader、VGG19、三种 loss、训练状态恢复、ONNX 导出或视频推理。

## 2. 固定公开契约

生产配置固定为：

- 图像输入：`[B, 6, 160, 160]`；
- FeatherHuBERT 音频窗口输入：`[B, 16, 32, 32]`；
- 输出：`[B, 3, 160, 160]`；
- 输出经过 sigmoid，所有有限输出都位于 `[0, 1]`；
- 主通道：`[32, 64, 128, 256, 512]`；
- 每个 MobileOne 块包含两个卷积分支；
- 上采样继续使用 bilinear、`align_corners=true`，并复用现有安全 padding 行为。

测试使用缩小通道的 `parity_micro()` 配置减少 CPU 成本，但空间尺寸、分支种类、重参数化公式和公开输入输出契约不变。

## 3. 架构边界

### 3.1 可复用 MobileOne 基础模块

现有 `pfld::MobileOneBlock<B>` 已表达训练态多分支结构，但只支持各向同性 stride，且没有融合接口。它将迁移到 crate 内部共享模块，PFLD 保留原公开导出，UNet 复用同一实现。

训练态 `MobileOneBlock<B>` 包含：

- 一个或多个 `Conv2d(bias=false) + BatchNorm` 主分支；
- 当 kernel 大于 1 时存在的 `1x1 Conv2d(bias=false) + BatchNorm` scale 分支；
- 当 stride 为 `[1,1]` 且输入输出通道相同时存在的 BatchNorm skip 分支；
- 可选 ReLU；本切片不引入 SE。

配置支持 `[height, width]` stride，以表达 Python Wenet 分支中的 `[1,2]`，但本切片的生产 UNet 只公开 FeatherHuBERT 模式。PFLD 继续通过方形 stride 构造器使用现有语义。

推理态 `ReparameterizedMobileOneBlock<B>` 只含一个带 bias 的 `Conv2d` 和与训练态一致的可选 ReLU。

### 3.2 UNet 训练图

`MobileOneSeparableBlock` 顺序执行：

1. depthwise MobileOne `3x3`；
2. pointwise MobileOne `1x1`；
3. 仅当显式启用、stride 为 1 且通道相等时，在两块输出之外添加外层 residual。

下采样、融合和上采样的双卷积块都由两个 separable block 构成。模型数据流与 Python 一致：

```text
image -> inc -> down1 -> down2 -> down3 -> down4
                                            +
audio -> MobileOne AudioConvHubert ----------+
                    -> concat -> fuse -> up1 -> up2 -> up3 -> up4 -> outc -> sigmoid
```

音频分支结构固定为：

- separable `16 -> ch[1]`；
- separable `ch[1] -> ch[2]`；
- MobileOne `3x3, stride [2,2]`, `ch[2] -> ch[3]`；
- residual separable `ch[3] -> ch[3]`；
- 普通 `Conv2d + BatchNorm + ReLU`, `3x3, stride [2,2], padding=3`, `ch[3] -> ch[4]`；
- 两个 residual separable `ch[4] -> ch[4]`。

普通 `Conv2d + BatchNorm + ReLU` 不属于 MobileOne 图，因此不在重参数化时改变。

### 3.3 UNet 推理图

推理态完整镜像训练态拓扑，但每个训练态 MobileOne 块都替换为 `ReparameterizedMobileOneBlock<B>`。普通卷积、BatchNorm、上采样和输出层按值复制，源训练模型保持可继续训练且参数不变。

训练态与推理态使用不同 Rust 类型，避免在 Burn `Module` 参数树中原地删除分支，也避免 checkpoint schema 在一次前向后发生变化。

## 4. 重参数化公式

对任意 `Conv(weight=W, bias=0) + BN(gamma, beta, mean, variance, epsilon)`：

```text
std = sqrt(variance + epsilon)
scale = gamma / std
W_fused[o, ...] = W[o, ...] * scale[o]
b_fused[o] = beta[o] - mean[o] * scale[o]
```

scale 分支的 `1x1` kernel 在空间中心补零到主 kernel 大小。skip BN 转换为 grouped identity kernel：

```text
identity[o, o % (in_channels / groups), center_y, center_x] = 1
```

所有主分支、scale 分支和 skip 分支的融合 kernel 与 bias 分别相加，生成最终带 bias 卷积。融合只允许奇数方形 kernel；当前生产图只使用 `1x1` 和 `3x3`。

融合读取 BatchNorm 的 running mean/variance，因此数值等价测试使用非 autodiff CPU backend 的 inference 语义。训练态反向测试单独使用 autodiff backend，不把训练批统计与推理 running statistics 混为一谈。

## 5. 错误与不变量

构造阶段使用断言拒绝：

- 零卷积分支；
- 零通道或零 group；
- 通道不能被 group 整除；
- 非 `[1,1]` / `[2,2]` 等已支持 stride；
- 偶数 kernel 或不匹配的 scale/skip 前提。

公开 UNet `forward` 在执行图之前断言固定图像和音频通道/空间尺寸；batch 大小必须相同。重参数化不执行隐式设备迁移，不允许 WGPU 静默回退 CPU。

## 6. 测试策略

测试按 TDD 添加：

1. 共享 MobileOne 块保持现有 PFLD shape 行为；
2. 各向异性 stride 能产生预期空间尺寸；
3. 无 skip、有 skip、depthwise group、`1x1`、`3x3` 块的训练态与融合态 CPU 输出满足 `max_abs_error <= 1e-4`；
4. `MobileOneUnetConfig::production()` 固定生产参数；
5. production 和 micro 模型输出 shape 为 `[B,3,160,160]` 且值有限并在 sigmoid 范围内；
6. autodiff 模型的输出卷积权重能获得梯度；
7. 完整 micro UNet 训练图与推理图固定输入前向满足 `max_abs_error <= 1e-4`；
8. 重参数化后再次运行源训练模型，输出不变，以证明转换不修改源模型；
9. 现有 Original UNet、PFLD 与 workspace 测试继续通过。

本切片不建立 Python/Burn MobileOne checkpoint parity fixture；该工作在后续受限 checkpoint 导入切片中完成。

## 7. 文件边界

- `rust/crates/feathertalk-models/src/mobileone.rs`：共享训练态块、推理态块和融合公式；
- `rust/crates/feathertalk-models/src/pfld/mobileone.rs`：移除私有实现，改为兼容重导出；
- `rust/crates/feathertalk-models/src/unet/mobileone_blocks.rs`：UNet 训练态/推理态 separable、down、up 和普通音频卷积组件；
- `rust/crates/feathertalk-models/src/unet/mobileone_model.rs`：完整训练态与推理态模型及转换；
- `rust/crates/feathertalk-models/src/unet/config.rs`：`MobileOneUnetConfig`；
- `rust/crates/feathertalk-models/tests/mobileone_reparameterization.rs`：块级数值等价；
- `rust/crates/feathertalk-models/tests/mobileone_unet.rs`：完整模型 shape、反向和转换等价。

## 8. 验收标准

- `MobileOneUnet` 和 `MobileOneUnetInference` 是公开 Burn module；
- 生产输入输出 shape 与 Python 固定契约一致；
- CPU float32 块级与完整 micro 模型重参数化 `max_abs_error <= 1e-4`；
- 源训练模型在转换前后输出不变；
- MobileOne 训练图完成一次输出层反向梯度验证；
- `cargo test --workspace --all-targets`、`cargo check --workspace --all-targets`、`cargo clippy --workspace --all-targets --all-features -- -D warnings`、`cargo fmt --all -- --check` 和 `git diff --check` 通过；
- WGPU 测试只有在认证 adapter 上运行，不允许 CPU fallback。
