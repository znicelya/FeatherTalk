# FeatherTalk 标准模型包设计

日期：2026-08-27  
状态：已确认，按推荐方案实施

## 1. 目标与边界

本切片为里程碑四补齐标准模型包的格式、构建器和运行时加载器。首个真实闭环是
FeatherHuBERT：从显式提供的受限 `.pth` checkpoint 导入 Burn 模型，生成可审计的
`model.safetensors`、`manifest.json` 和 `LICENSES.json`，随后在 CPU 上重新加载并逐 tensor
校验。模型包的格式同时为 Original UNet、MobileOne UNet、PFLD 和 SCRFD 保留明确的
模型描述字段，但本切片不凭空生成缺少来源权重或许可证的发布模型。

本切片包含：

- `feathertalk-export` Rust crate，提供标准包 schema、严格校验、原子写入和泛型 safetensors
  加载 API；
- FeatherHuBERT `.pth` 到标准包的 CLI；
- 真实 checkpoint 的导入、哈希和 safetensors round-trip 验证；
- 包目录、许可证、来源、模型配置、I/O contract、训练元数据和最低应用版本记录。

本切片不包含 ONNX 图生成、`ort` 运行、旧 `.npy` 特征迁移或 worker/GPUI 集成；这些作为
后续独立切片继续执行。

## 2. 方案选择

### 方案 A：每种模型各自实现一套包格式

每个模型 crate 自己定义 manifest、文件校验和加载逻辑。实现初期看似直接，但会重复
符号链接检查、哈希、许可证校验和原子发布，容易让不同模型产生不兼容的安全语义。

### 方案 B：只做 FeatherHuBERT 专用工具

只保存 FeatherHuBERT 的三个文件，最快得到真实模型结果，但后续 UNet、PFLD 和 SCRFD
无法共享格式，ONNX 导出和模型页还需要再次定义来源与 I/O contract。

### 方案 C（推荐）：通用包容器 + 类型化模型描述 + FeatherHuBERT 首个构建器

在 `feathertalk-export` 中统一实现容器和文件生命周期；manifest 用严格类型记录模型种类、
架构版本、配置、输入输出 tensor contract、训练元数据、来源、许可证和每个文件的哈希。
构建器通过泛型 `ModuleSnapshot` 支持后续模型，首个 CLI 只暴露已具备真实 checkpoint 的
FeatherHuBERT。这样不会伪造尚未审核的权重，同时把后续 UNet/ONNX 所需的稳定接口一次
固定下来。

采用方案 C。

## 3. 包目录和 schema

推理包的目录必须恰好包含三个非符号链接普通文件：

```text
model-package/
  manifest.json
  model.safetensors
  LICENSES.json
```

训练包可同时包含 `optimizer.safetensors` 和 `training-state.json`；两者必须成对出现，
且目录只能扩展为五个文件。当前构建器只生成推理包，不会把训练状态伪装成部署权重。

`manifest.json` 使用 schema version 1 和 `serde(deny_unknown_fields)`，字段如下：

```text
schema_version
model_type
architecture_version
configuration
inputs[] / outputs[]       # name, shape, dtype；-1 表示动态维度
training                   # mode 和有限、非负 loss 参数
source                     # format、identifier、version、file_name、sha256、可选 URL
created_at
minimum_app_version
tensors[]                  # safetensors 中每个 tensor 的 name、shape、dtype
model                      # file_name、bytes、sha256
licenses                   # file_name、bytes、sha256
optimizer? / training_state?
```

FeatherHuBERT 的固定描述为：

```text
model_type: feather_hubert
architecture_version: feather-hubert-burn-v1
configuration: channels, expansion, num_blocks, output_dim, dropout
input:  waveform [1,-1] f32
output: hidden   [1,-1,1024] f32
```

`configuration` 使用类型化的 FeatherHuBERT/Original UNet/MobileOne UNet 枚举；未知模型
类型不能通过运行时加载器。`tensors` 从实际 Burn module 收集并按完整路径排序，加载时同时
检查名称、shape、dtype、数量和总元素数。

`LICENSES.json` 使用独立 schema version 1，至少包含一个非空组件条目，每条记录组件、
许可证标识、来源 URL 和 notice。工具只验证并复制调用者提供的文件，不推断外部权重是否
获得商业再分发许可。

## 4. 构建与加载数据流

构建 FeatherHuBERT 包的顺序：

1. 校验 source、license、destination 路径和符号链接组件；
2. 将 source `.pth` 复制到不可变临时快照并计算 SHA-256；
3. 用 `feathertalk-weights` 的受限导入器在 CPU `NdArray` 上加载快照；
4. 在 destination 同父目录创建唯一 staging 目录；
5. 写 `model.safetensors` 和 `LICENSES.json`，每个文件 `flush/sync_all` 后重新计算长度与哈希；
6. 从 Burn module 生成排序后的 tensor audit；
7. 写 manifest（最后写入），重新解析并验证 manifest、license 和 safetensors round-trip；
8. 校验 source 快照哈希仍一致后，将 staging 原子 rename 到不存在的 destination，并同步父目录。

任何一步失败都只删除当前进程创建的 staging；不会覆盖既有 destination，也不会修改 source。

加载器先验证目录精确条目、普通文件、manifest JSON、license JSON、声明的字节数和哈希，
再比较调用方提供的模型描述和 tensor audit，最后才调用 Burn `SafetensorsStore`。调用方
提供一个按 manifest 配置创建全新空模型的 factory，加载器只修改这个新实例，成功后才返回；
任何失败都不会触碰调用方已有模型。这里不使用 Burn module 的普通 `Clone`，因为包含
BatchNorm `RunningState` 的模型可能共享内部状态。运行时
不下载、不搜索 cache、不接受 `.pth`，也不静默随机初始化或 CPU fallback。

## 5. 错误与限制

- source 上限 512 MiB，manifest 64 KiB，license 1 MiB；
- 所有 hash 必须是 64 个小写十六进制字符；
- tensor 名称、shape、dtype、数量和总元素数必须自洽；
- `-1` 之外的维度必须为正数，loss 权重必须 finite 且非负；
- 目录、文件和路径组件拒绝符号链接、额外文件、缺失文件和已有 destination；
- schema、模型类型、架构版本、配置、来源和许可证错误均返回结构化 `PackageError`；
- 许可证记录不等于商业授权，发布流水线仍需人工审计。

## 6. 测试与验收

必须包含：

1. manifest/license schema round-trip、未知字段和坏 hash 拒绝；
2. 三文件/五文件目录精确性、额外条目、符号链接、超限文件和已有 destination 拒绝；
3. staging 失败清理与 no-clobber 原子发布；
4. safetensors 缺失、额外、shape/dtype 不匹配在 Burn 加载前拒绝；
5. 泛型 module round-trip 和逐 tensor 数据比较；
6. FeatherHuBERT micro fixture 的导入报告与 manifest；
7. 显式环境变量指向真实 `feather_hubert_188_latest_99.pth` 的 CPU 导入 smoke test，确认
   40,436,613 字节和已知 SHA-256，并确认源文件未被修改；
8. `cargo test --workspace --all-targets`、`cargo check --workspace --all-targets`、
   `cargo clippy --workspace --all-targets -- -D warnings`、`cargo fmt --all -- --check` 和
   `git diff --check`。

## 7. 后续衔接

ONNX 切片直接消费本 manifest 的模型类型、配置和 I/O contract，输出 opset 17 并复用
同一套哈希/原子发布辅助函数。旧模型/`.npy` 迁移 CLI 将把导入结果送入同一标准包容器，
而不再定义第二套文件格式。
