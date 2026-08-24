# VGG19 感知特征与三种训练损失设计

日期：2026-08-24
状态：已确认

## 1. 目标

为迁移里程碑三建立模型无关的 Rust 训练损失层，并精确复现现有 Python 训练入口的三种模式：

- Baseline；
- Mouth ROI；
- Mouth ROI + Temporal。

本切片同时实现冻结的 VGG19 `conv3_3` 特征图、独立模型包加载、受限 PyTorch 权重转换和 Python/Burn 数值验证。Original UNet 与 MobileOne UNet 都复用同一损失 API。

本切片不实现 DataLoader、shuffle、随机状态、optimizer checkpoint、epoch/global step 恢复、训练循环、指标持久化或预览图；这些继续作为里程碑三的后续独立切片。

## 2. Crate 与依赖边界

新增 `rust/crates/feathertalk-training`：

- `vgg19`：精确的 `Vgg19Conv3_3<B>` Burn 模块；
- `artifact`：VGG19 独立模型包 schema、校验和加载；
- `perceptual`：冻结特征器接口和感知 MSE；
- `losses`：Baseline、Mouth ROI、Temporal 配置、输入校验和损失分解。

`feathertalk-training` 只依赖通用 Burn/store/serde/hash 能力，不依赖 Python，也不在运行时解析 pickle 或访问网络。后续训练循环可依赖该 crate 和 `feathertalk-models`。

扩展 `feathertalk-weights` 的受限旧权重导入器，增加 `LegacyModelKind::Vgg19Conv3_3`。该 crate 仍不依赖具体 VGG Rust 类型；调用者把待加载的 Burn module 传给通用 `import_into`。

新增 `rust/tools/vgg19-package`。工具同时依赖 `feathertalk-training` 和 `feathertalk-weights`，负责把审核过的 torchvision VGG19 `.pth` 转为独立 safetensors 模型包。应用运行时只读取模型包，不依赖该工具。

## 3. VGG19 `conv3_3` 精确语义

### 3.1 Python 对齐边界

当前 Python `PerceptualLoss` 构造 `torchvision.models.vgg19(...).features`，顺序加入索引 `0..=14` 后停止。因此 Rust 必须实现以下精确行为：

```text
features.0   conv1_1 + bias -> ReLU
features.2   conv1_2 + bias -> ReLU
features.4   MaxPool2d(2, 2)
features.5   conv2_1 + bias -> ReLU
features.7   conv2_2 + bias -> ReLU
features.9   MaxPool2d(2, 2)
features.10  conv3_1 + bias -> ReLU
features.12  conv3_2 + bias -> ReLU
features.14  conv3_3 + bias -> output
```

输出不经过 `features.15` 的 ReLU。对 `[B,3,H,W]` 输入，`H` 和 `W` 必须至少为 4，输出为 `[B,256,floor(H/4),floor(W/4)]`。生产输入 `[B,3,160,160]` 对应 `[B,256,40,40]`。

所有卷积固定为：

- kernel `[3,3]`；
- stride `[1,1]`；
- 四边 padding 1；
- dilation `[1,1]`；
- groups 1；
- bias 开启。

两次池化固定为 2x2 max-pool、stride 2、padding 0、dilation 1、`ceil_mode=false`。

### 3.2 输入兼容语义

为精确复现现有训练行为，VGG 输入保持：

- BGR 通道顺序；
- float32 `[0,1]`；
- 不交换 RGB/BGR；
- 不执行 ImageNet mean/std normalization。

这与常见 torchvision 推理预处理不同，但属于现有 FeatherTalk checkpoint 的训练语义。本迁移不得静默“修正”它。

### 3.3 冻结与梯度

从模型包加载完成后，整个 `Vgg19Conv3_3` 必须调用 Burn `Module::no_grad()`：

- 预测图像经过 VGG 时仍保留到生成器输出的梯度；
- target 特征显式 detach；
- VGG 的 weight/bias 不产生梯度；
- VGG 参数不属于 UNet optimizer，也不写入 UNet checkpoint。

禁止随机初始化 fallback。模型包缺失、损坏或不兼容时，训练必须在第一次 loss 计算前失败。

## 4. VGG19 独立模型包

### 4.1 目录合同

VGG19 包目录只能包含三个普通文件：

```text
vgg19-conv3-3/
  manifest.json
  model.safetensors
  LICENSES.json
```

目录和文件均不得是符号链接。固定读取上限：

- `manifest.json`：64 KiB；
- `LICENSES.json`：1 MiB；
- `model.safetensors`：16 MiB。

### 4.2 Manifest

`manifest.json` 使用 deny-unknown-fields 的 schema one：

```json
{
  "schema_version": 1,
  "model_kind": "vgg19-conv3-3",
  "architecture_version": "torchvision-vgg19-conv3-3-v1",
  "source": {
    "framework": "torchvision",
    "weight_id": "VGG19_Weights.IMAGENET1K_V1",
    "url": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "sha256": "<64 lowercase hex>"
  },
  "input": {
    "channels": 3,
    "color_order": "bgr",
    "value_range": "0..1",
    "normalization": "none"
  },
  "output_layer": "features.14",
  "tensor_count": 14,
  "total_elements": 1735488,
  "model": {
    "file_name": "model.safetensors",
    "bytes": 0,
    "sha256": "<64 lowercase hex>"
  },
  "licenses": {
    "file_name": "LICENSES.json",
    "bytes": 0,
    "sha256": "<64 lowercase hex>"
  }
}
```

`bytes` 在实际文件中必须大于零并等于文件长度。所有 SHA-256 为 64 个小写十六进制字符。source hash 记录输入 `.pth` 的完整内容哈希；包加载器校验格式和模型包自洽性，发行流水线负责选择审核通过的 source hash。

`LICENSES.json` 使用独立版本化 schema，至少包含一个非空组件条目、许可证标识、来源 URL 和 notice。转换工具只复制调用者提供且已通过 schema 校验的许可证文件，不自行推断或伪造权重许可结论。

### 4.3 Safetensors 合同

`model.safetensors` 必须只包含以下 14 个 float32 tensor：

```text
conv1_1.weight  conv1_1.bias
conv1_2.weight  conv1_2.bias
conv2_1.weight  conv2_1.bias
conv2_2.weight  conv2_2.bias
conv3_1.weight  conv3_1.bias
conv3_2.weight  conv3_2.bias
conv3_3.weight  conv3_3.bias
```

加载必须严格拒绝 missing、unused、shape mismatch、dtype mismatch 和额外 tensor。成功加载并冻结后才返回模型。

### 4.4 运行时禁止行为

包加载器不得：

- 自动下载权重；
- 搜索用户 home cache；
- 接受完整 `.pth`；
- 忽略哈希或 tensor 错误；
- 用随机参数继续训练；
- 在 WGPU 不可用时静默切换 CPU。

## 5. 受限 `.pth` 导入与发布工具

### 5.1 Key 映射

`LegacyModelKind::Vgg19Conv3_3` 固定映射：

```text
features.0  -> conv1_1
features.2  -> conv1_2
features.5  -> conv2_1
features.7  -> conv2_2
features.10 -> conv3_1
features.12 -> conv3_2
features.14 -> conv3_3
```

仅该 model kind 可忽略官方 VGG19 中明确位于截断点后的 tensor：

```text
features.{16,19,21,23,25,28,30,32,34}.{weight,bias}
classifier.{0,3,6}.{weight,bias}
```

任何其他 unused tensor 均失败。导入报告必须为：

- applied tensor：14；
- ignored tensor：24；
- applied elements：1,735,488。

通用 BatchNorm `num_batches_tracked` 忽略规则继续保留，但 VGG 忽略集合必须按 model kind 隔离，不能让其他导入器接受 VGG 风格额外 key。

### 5.2 工具输入输出

工具接受：

```text
--source <vgg19-dcbb9e9d.pth>
--licenses <reviewed-LICENSES.json>
--destination <new-directory>
```

source 先复制到不可变临时快照并流式计算哈希。destination 必须不存在。工具在 destination 同一父目录创建 staging，完成以下步骤后才原子 rename：

1. 严格导入 14 个 tensor；
2. 写入 `model.safetensors`；
3. 重新严格加载并逐 tensor 比较；
4. 校验并复制 `LICENSES.json`；
5. 计算文件长度和哈希；
6. 写入并重新读取 `manifest.json`；
7. 通过运行时包加载器加载一次；
8. 原子发布。

任何失败都不创建最终 destination，并清理工具拥有的 staging。source 永不修改。

## 6. 感知损失接口

定义模型无关接口：

```rust
pub trait PerceptualFeatureExtractor<B: Backend> {
    fn forward(&self, image: Tensor<B, 4>) -> Tensor<B, 4>;
}
```

`Vgg19Conv3_3<B>` 实现该接口。测试可用确定性的轻量 extractor 验证纯损失公式，避免所有单元测试都执行 VGG。

感知损失固定为：

```text
mean((features(prediction) - detach(features(target)))^2)
```

prediction 和 target 必须具有相同 `[B,3,H,W]` shape。

## 7. 三种训练损失

### 7.1 配置

公开三个可序列化配置：

```text
BaselineLossConfig:
  perceptual_weight = 0.01

MouthRoiLossConfig:
  mouth_weight      = 4.0
  perceptual_weight = 0.01

TemporalLossConfig:
  mouth_weight          = 4.0
  temporal_weight       = 0.5
  temporal_mouth_weight = 4.0
  perceptual_weight     = 0.01
```

所有权重必须有限且大于等于零。配置校验失败返回结构化错误，不在训练中间静默修正。

### 7.2 公共分量

每个模式返回 `LossBreakdown<B>`：

```text
total:          Tensor<B,1>
full:           Tensor<B,1>
perceptual:     Tensor<B,1>
mouth:          Option<Tensor<B,1>>
temporal:       Option<Tensor<B,1>>
temporal_mouth: Option<Tensor<B,1>>
```

`total` 保留反向图；其他分量也保持 tensor 形式，后续指标层负责在安全边界读取标量。

### 7.3 Baseline

输入 prediction/target 为 `[B,3,H,W]`：

```text
full       = mean(abs(prediction - target))
perceptual = MSE(VGG19 conv3_3)
total      = full + 0.01 * perceptual
```

### 7.4 Mouth ROI

mask 为 `[B,1,H,W]`，通过 broadcast 应用于三个图像通道：

```text
mouth = sum(abs(prediction - target) * mask)
        / (max(sum(mask), 1) * image_channels)

total = full
      + mouth_weight * mouth
      + perceptual_weight * perceptual
```

空 mask 的 mouth loss 必须为有限的 0，不得产生 NaN 或除零。

### 7.5 Temporal

prediction/target 为 `[B,2,3,H,W]`，mask 为 `[B,2,1,H,W]`。pair 长度必须精确为 2：

```text
flat prediction/target/mask = reshape batch 与 pair 为 B*2
full                        = L1(flat prediction, flat target)
mouth                       = mouth L1(flat tensors)
pred_delta                  = prediction[:,1] - prediction[:,0]
target_delta                = target[:,1] - target[:,0]
union_mask                  = elementwise max(mask[:,0], mask[:,1])
temporal                    = L1(pred_delta, target_delta)
temporal_mouth              = mouth L1(pred_delta, target_delta, union_mask)
perceptual                  = MSE(VGG(flat prediction), VGG(flat target))

total = full
      + mouth_weight * mouth
      + temporal_weight * temporal
      + temporal_mouth_weight * temporal_mouth
      + perceptual_weight * perceptual
```

这保持 Python 中“两帧先展平做 full/mouth/perceptual，再单独计算 delta”的语义。

## 8. 输入验证与错误

在执行昂贵 VGG 图之前检查：

- prediction/target shape 完全一致；
- 图像通道为 3；
- mask batch/空间与图像一致且通道为 1；
- Temporal pair 维度精确为 2；
- 空 batch、空间小于 4 或零维度被拒绝；
- 配置权重有限且非负。

Tensor 值范围 `[0,1]` 是数据管线合同，不为验证而把 GPU tensor 同步读回 CPU。非有限 loss 由后续训练步骤的数值检查处理。

错误使用 `TrainingError` 返回，包格式、I/O、哈希、shape/config 和 store apply 错误保持可区分。内部不把外部输入错误表示为不可恢复 panic。

## 9. 测试策略

### 9.1 Rust-only 常规测试

- VGG `16x16 -> 256x4x4` shape；
- 构造负的 `conv3_3` bias，证明输出在索引 14 停止且未经过 ReLU；
- VGG 参数 `no_grad`，prediction 能获得梯度而 VGG weight 不能；
- 相同输入感知 MSE 为 0；
- Baseline、Mouth ROI、Temporal 使用手算小 tensor 验证每个分量和默认权重；
- 空 mask、mask broadcast、union mask、pair length 和 shape 错误；
- 三种 total 都能把梯度传回 prediction；
- 临时模型包的 safetensors round-trip；
- extra/missing tensor、坏 hash、坏长度、未知字段、symlink、额外目录项和超限文件均拒绝；
- 受限 VGG PyTorch fixture 验证 14 applied、24 ignored、key 映射和失败原子性；
- Original UNet、MobileOne UNet 和已有权重导入测试继续通过。

### 9.2 官方权重 parity

提供只依赖 `torch` 的 golden 生成脚本，不依赖 torchvision Python 包。脚本读取官方 VGG19 state dict，使用固定 BGR `[0,1]` 小输入按索引 `0..=14` 执行并写入哈希绑定的 fixture。

外部模型包验收测试通过显式环境变量或命令参数加载刚生成的包，对比 Python 与 Burn：

```text
max_abs_error <= 1e-4
mean_abs_error <= 1e-5
```

该测试不自动下载。里程碑验收必须至少在一个审核过的官方权重包上显式运行一次并记录命令结果。

## 10. 文件边界

预期新增或修改：

- `rust/Cargo.toml`；
- `rust/crates/feathertalk-training/Cargo.toml`；
- `rust/crates/feathertalk-training/src/lib.rs`；
- `rust/crates/feathertalk-training/src/error.rs`；
- `rust/crates/feathertalk-training/src/vgg19.rs`；
- `rust/crates/feathertalk-training/src/artifact.rs`；
- `rust/crates/feathertalk-training/src/perceptual.rs`；
- `rust/crates/feathertalk-training/src/losses.rs`；
- `rust/crates/feathertalk-training/tests/*.rs`；
- `rust/crates/feathertalk-weights/src/key_map.rs`；
- `rust/crates/feathertalk-weights/src/legacy.rs`；
- `rust/crates/feathertalk-weights/tests/*.rs`；
- `rust/tools/vgg19-package/Cargo.toml`；
- `rust/tools/vgg19-package/src/main.rs`；
- `rust/tools/parity/generate_vgg19_golden.py`；
- `rust/tests/golden/vgg19-conv3-3-v1.zip` 及 SHA-256 sidecar；
- `docs/WEIGHTS.md`。

不得修改、删除或提交 `demo/kanghui_training_video_featherhubert_188_latest/`。

## 11. 验收标准

- VGG 图精确截止在 torchvision `features.14` 的 pre-ReLU 输出；
- BGR `[0,1]`、无 normalization 的现有 Python 语义明确且有测试；
- 运行时只加载严格、哈希校验、带许可证记录的独立 safetensors 包；
- VGG 参数冻结，prediction/UNet 仍可通过感知 loss 获得梯度；
- Baseline、Mouth ROI、Temporal 公式和默认值与 Python 一致；
- 官方 VGG19 CPU parity 达到 `max_abs <= 1e-4`、`mean_abs <= 1e-5`；
- 不存在自动联网、随机权重 fallback 或 WGPU 到 CPU 静默回退；
- `cargo test --workspace --all-targets`；
- `cargo check --workspace --all-targets`；
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`；
- `cargo fmt --all -- --check`；
- `git diff --check`；
- 分支只包含本切片文件，受保护 demo 目录不变。
