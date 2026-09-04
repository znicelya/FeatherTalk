# 模型检视 worker 命令与 CLI 子命令设计

日期：2026-09-04
状态：已定稿

## 1. 目标与范围

模型侧的库能力早已齐备：`feathertalk-export` 能写、能读标准模型包（`write_model_package`、`read_package_manifest`、`load_model_package`），`feathertalk-training` 能写、能读训练检查点（`save_training_checkpoint`、`read_training_checkpoint`）。上一切片把 `render` 接到线协议上，工程目录里现在会长出 `models/unet/checkpoint-XXXXXXXX/`。缺的是最基础的一句回答：**手里这个目录到底是什么，这一版程序能不能用它。** 今天只能靠人去读 JSON。

本切片交付 13 个 `TaskKind` 中的第 9 个：worker 实现 `Request::InspectModel`，CLI 增加 `inspect-model` 子命令。它正对着迁移设计 §10.4「模型」页要显示的那几件事：模型类型、参数量、输入输出形状、文件哈希、兼容状态。

领域契约早已提交，本切片一个字节都不改：`TaskKind::InspectModel`（slug `inspect_model`）、`InspectModelParams { source }`（带 `deny_unknown_fields`）、`Request::InspectModel`。`commands.rs` 现在把它落在 `other => Failed(unsupported(...))` 分支里。

本切片范围内的改动：

- `feathertalk-worker`：新增 `inspecting.rs`（分类与兼容判定）、`inspect.rs`（命令编排）、`inspect_result.rs`（结果载荷）；`admission.rs` 新增 `check_model_source`；`handshake.rs` 无条件宣告 `InspectModel`；`error_map.rs` 不变，复用 `package_task_error` 与 `training_task_error`。
- `feathertalk-cli`：新增 `inspect-model` 子命令与对应的 `build_request` 分支。
- 顺带修正 `feathertalk-worker/src/lib.rs` 顶部注释的命令清单：上一切片交付了 `render`，那行没有同步。

两个直接结论：本命令**不需要新依赖**，因为两个读取器都在工作区里；也**不需要新环境变量、不需要任何工具链**，因为它只读目录里的 JSON。它是 `validate_project` 之后第二个无条件可用的命令。

范围外的内容集中在 §10。

## 2. 两种可检视的目录

`InspectModelParams` 只有 `source` 一个字段，没有类型标记。这不是遗漏：分类是 worker 的判断，用户不该被要求先说清自己手里的东西是什么。而目录里的文件名足够分类，两种布局互不相交：

```text
标准模型包（推理）        训练检查点
manifest.json            manifest.json
model.safetensors        model.bin
LICENSES.json            optimizer.bin
                         training-state.json
```

分类只看一件事：`manifest.json` 旁边是 `model.safetensors` 还是 `model.bin`。

- 只有 `model.safetensors` → 交给 `read_package_manifest`。
- 只有 `model.bin` → 交给 `read_training_checkpoint`。
- 两个都在、或两个都不在 → 拒绝，`ModelIncompatible`，摘要「无法识别的模型目录」。

分类只用 `symlink_metadata` 看这两个名字是否是普通文件，不打开、不读取。真正的结构校验全部由两个读取器完成，本切片不复制它们的规则，也因此不会与它们不一致：

- `read_package_manifest` 先跑 `validate_package_directory(dir, false)`：目录条目必须精确等于那三个文件名，每一项都必须是普通文件且路径上没有符号链接；再按 64 KiB 上限读 `manifest.json` 并跑 `ModelPackageManifest::validate`。
- `read_training_checkpoint` 先跑 `validate_checkpoint_directory`：目录条目必须精确等于那四个文件名；再读并校验 `manifest.json` 与 `training-state.json` 两份 JSON。

两个读取器都保证**声明的文件一定存在**，这一点在 §4 与 §6 里都被用到。

一个后果要写明：`read_package_manifest` 只接受三文件的推理包。迁移设计 §5.5 里那种额外带 `optimizer.safetensors` 与 `training-state.json` 的五文件训练包，今天没有任何代码写得出来（`write_model_package` 恒定写 `optimizer: None`、`training_state: None`），所以本切片也不接受它：分类看到 `model.safetensors` 就交给 `read_package_manifest`，多出来的条目由它报错。等哪个切片真开始写五文件包，`read_package_manifest` 的 `training` 参数与这里的分类会一起改。

## 3. 只读清单，不读权重

`inspect-model` 不加载模型：不建 device、不建模型模板、不读 safetensors 的张量数据、不读 Burn record。两个读取器的文档都点明「不碰权重」，这正是本命令要的 —— 一次检视是毫秒级的，「模型」页可以在用户点开一个目录时就地调用，不必排队去等适配器锁。

由此产生一处诚实的空缺：**训练检查点报不出参数量。** 检查点 manifest 记的是三个文件的字节数与摘要，没有张量清单；要数参数就得按 `model_kind` 建出模板再读 record，那是渲染做的事。所以 `parameter_count` 与 `tensor_count` 在检查点上是 `null`，在模型包上取自清单里的 `tensors.total_elements` 与 `tensors.tensor_count`。宁可报 `null`，也不用 `model.bin` 的字节数除以 4 去糊一个数。

同理，本命令**不重算文件摘要**。载荷里的 `sha256` 是清单声明的值，不是现场哈希：一个 2 GiB 的模型包重算一遍要十几秒，而检视要回答的是「这份清单说自己是什么」。声明与磁盘是否一致，由真正要用权重的命令验证 —— `load_model_package` 与 `load_training_checkpoint_model` 都会跑 `validate_declared_file`。

但有一件几乎免费的事值得做：每个声明文件多花一次 `symlink_metadata`，把磁盘上的实际字节数一并报出（`bytes_on_disk`）。文件一定存在（§2），所以它只可能与清单不等；不等就说明目录被人动过，见 §4 的 `file_size`。

## 4. 兼容状态

「兼容状态」不是一种感觉，是一句可执行的话：**这一版程序能不能把这个目录用起来。** 载荷统一成 `compatible: bool` 加 `incompatibilities: [string]`，理由用英文标识符（`minimum_app_version`、`model_kind`、`architecture_version`、`model_config_sha256`、`file_size`）。载荷是给程序读的，中文话术留给 CLI 与工作台。

两个读取器已经挡掉的东西不再重复检查：schema 版本、`record_format`、模型类型与配置是否自洽、输入输出契约与配置是否一致、`created_at` 与 `minimum_app_version` 的格式、摘要的字面形状。这些不通过就根本读不出清单，那是**失败**，不是「不兼容」。真正需要判断的是：

模型包：

- `minimum_app_version` 是否高于 worker 自己的版本（`WorkerConfig::worker_version()`，即 `CARGO_PKG_VERSION`）。两侧都是 `validate_version` 保证过的三段数字，按 `(major, minor, patch)` 元组比较即可，不引入 semver 依赖。任一侧解析不出来也记 `minimum_app_version`：无法证明兼容时宁可报不兼容。

训练检查点：

- `model_kind` 能否映射到本版认识的 U-Net 变体。这正是 `rendering::render_variant` 做的事，直接复用：认不出就记 `model_kind`，后两项无从比较，判定就此结束。名字里的 render 是它诞生在渲染切片留下的，本切片不改名，以免制造一次纯改名的 diff。
- 认得出时，用 `checkpoint_descriptor(&variant.configuration())` 造出本版期望的描述符，与检查点清单的 `descriptor()` 逐字段比：`architecture_version` 与 `model_config_sha256` 不等就各记一项。`optimizer_kind` 与 `optimizer_schema_version` 不必比 —— `TrainingCheckpointManifest::validate` 已经要求它们等于本版常量，读得出来就必然相等。

这条检查与 `load_training_checkpoint_model` 的准入完全同源，于是 `compatible: true` 等价于一句可验证的承诺：**`render` 会接受这个检查点。**

两种来源共有：

- 任一声明文件的磁盘字节数与清单不符 → 记 `file_size`。

`compatible` 是 `incompatibilities.is_empty()`，不是另算一遍的第二个判断 —— 两者由同一个函数一次算出，不可能互相矛盾。

## 5. 准入与错误模型

`check_model_source` 与 `admission.rs` 里的 `check_project_dir` 同构，放在它旁边：`source` 必须是绝对路径 —— 工作台与 CLI 传的都是用户选中的完整路径，接受相对路径会把 worker 的工作目录变成一个隐藏参数 —— 且必须是真目录（`symlink_metadata`，不跟随符号链接）。两条失败都报 `MediaInvalid`，摘要「模型目录必须是绝对路径」与「模型目录不可用」。

分类失败不属于准入：它已经是对目录内容的判断，报 `ModelIncompatible`，摘要「无法识别的模型目录」。这个码的恢复建议正是「请重新导入模型文件」，与实际处置一致。

读取器的失败按来源交给已有的映射器，本切片不新增映射函数：

| 失败 | 映射 | 码 |
| --- | --- | --- |
| 准入（非绝对路径、非目录） | `admission::invalid_request` | `MediaInvalid` |
| 分类不出来 | 本切片一个本地构造 | `ModelIncompatible` |
| `PackageError`（任意变体） | `package_task_error` | `ModelIncompatible` |
| `TrainingError`（任意变体） | `training_task_error(&error, TaskStage::Preparing)` | 由映射器决定 |

全部失败的 `stage` 都是 `TaskStage::Preparing`：检视没有第二个阶段，读清单就是它的全部工作。

取消：命令在读清单之前查一次 `token.is_cancelled()`，在返回载荷之前再查一次，与 `validate_project` 同一处置 —— 取消了就 `Cancelled`，不把已经算完的结果当成完成。中间没有第三个检查点，因为两次读取都是有界的小 JSON。

不报进度事件。一次检视没有可分级的工作量，`Progress { completed: 1, total: Some(1) }` 只是噪声。

## 6. 结果载荷

`inspect_to_json` 写出一个扁平对象，字段顺序固定，与 `render_to_json` 同风格。来源答不出的字段是 `null` 而不是缺席：缺席会让工作台分不清「这份清单没写」与「对面是个旧 worker」。

| 字段 | 模型包 | 训练检查点 |
| --- | --- | --- |
| `source_kind` | `"model_package"` | `"training_checkpoint"` |
| `source_path` | 绝对路径原样 | 绝对路径原样 |
| `schema_version` | 清单原样 | 清单原样 |
| `model_kind` | `model_type` | `model_kind` |
| `architecture_version` | 清单原样 | 清单原样 |
| `model_config_sha256` | `null` | 清单原样 |
| `parameter_count` | `tensors.total_elements` | `null` |
| `tensor_count` | `tensors.tensor_count` | `null` |
| `inputs` / `outputs` | `[{name, shape, dtype}]` | `[]` |
| `training_mode` | `training.mode` | `training_config.mode` |
| `epoch` / `global_step` | `null` | 训练状态原样 |
| `created_at` | 清单原样 | `null` |
| `minimum_app_version` | 清单原样 | `null` |
| `files` | model、licenses | model、optimizer、training-state |
| `compatible` | §4 | §4 |
| `incompatibilities` | §4 | §4 |

`files` 的每一项是 `{file_name, bytes, sha256, bytes_on_disk}`，顺序与清单里的顺序一致。`inputs`/`outputs` 在检查点上是空数组而不是 `null`，这让工作台的表格少一个分支；数组里的 `shape` 保留清单写下的 `-1`，它表示这一维是动态的（FeatherHuBERT 的 `waveform [1, -1]` 就是如此），不是错误。

两个 `training_mode` 枚举分别来自两个 crate（模型包那个多一个 `inference`），载荷里都用 `snake_case` 字面量：`inference`、`baseline`、`mouth_roi`、`mouth_roi_temporal`。写法沿用 `train_result.rs`：穷举 `match` 返回 `&'static str`，而不是让 serde 去序列化 —— 这样将来多一种模式是一个编译错误，而不是一个静默改变的字符串。

`source_path` 报的是请求里的路径原样（`Path::display`），不做 canonicalize：用户看到的应该是自己选的那个路径，而不是解析过符号链接与短名的另一个写法。准入已经保证它是绝对路径。

## 7. 命令签名与 CLI 形态

worker 侧的入口与其他命令同形，参数表短到不需要一个 job 结构：

```rust
pub fn execute_inspect_model(
    params: &InspectModelParams,
    config: &WorkerConfig,
    token: &CancellationToken,
) -> CommandOutcome
```

`config` 只为了一件事：`worker_version()`，即 §4 里模型包那条比较的右手边。没有 `reporter` 参数 —— 本命令不报阶段也不报进度（§5），传一个用不到的 `&dyn TaskReporter` 只会让读者去找它在哪里被用了。

`commands.rs` 的分派分支因此也是全篇最短的一个，而且没有工具链守卫：

```rust
Request::InspectModel(params) => execute_inspect_model(params, config, token),
```

CLI 侧一个位置参数：

```text
feathertalk inspect-model <SOURCE>
```

`SOURCE` 是模型包目录或训练检查点目录。`build_request` 只做 `reject_empty(source, "模型目录")`，其余判断全归 worker —— 这是 CLI 一贯的分工，把「是不是目录」在两边各判一次，迟早会得到两个不同的答案。人读输出就是 `run_task` 已有的那段：完成时把载荷按 pretty JSON 打到 stdout，`--json` 时原样透传协议帧。本切片不为它写中文表格。

`render.rs` 的 `UnsupportedCommand` 提示分支不增加：`inspect_model` 无条件出现在握手里，客户端不可能收到「不支持」。

## 8. 握手：无条件宣告

`supported_commands` 现在以 `vec![TaskKind::ValidateProject]` 开头，其余命令都挂在某个工具链的 `is_some()` 上。`InspectModel` 与 `ValidateProject` 同类：它只读文件系统里的 JSON，没有外部工具、没有模型包、没有适配器要求，所以它加在同一个 `vec!` 里，不带任何条件。

这一句话是可测的：一个什么环境变量都没设的 worker，握手里必须同时出现 `validate_project` 与 `inspect_model`，且仍然不出现 `probe_media`。

`Capabilities` 结构不动。它记的是「能不能训练」「有没有 ffmpeg」这类能力位，检视不属于其中任何一项。

## 9. 测试

| 层 | 位置 | 覆盖 |
| --- | --- | --- |
| 分类与兼容 | `worker/tests/inspecting.rs` | 两种布局各认得出；两个都在、都不在各报「无法识别」；`minimum_app_version` 高于本版记一项；未知 `model_kind` 记一项且不再比后两项；`architecture_version` 与 `model_config_sha256` 各自不等各记一项；文件被截断记 `file_size`；兼容时 `incompatibilities` 为空 |
| 载荷 | `worker/tests/inspect_result.rs` | 两种来源的字段齐备与 `null` 位置；`files` 的四个字段；三种 `training_mode` 字面量 |
| 命令 | `worker/tests/inspect.rs` | 相对路径与非目录在读任何文件之前被拒；取消返回 `Cancelled`；真包与真检查点各跑通一次 `execute_inspect_model` |
| 握手 | `worker/tests/handshake.rs` | 空配置的 worker 宣告 `inspect_model` |
| CLI | `cli/src/run.rs` 内联测试 | 空路径报「模型目录不能为空。」；`inspect-model` 装出 `Request::InspectModel` 且路径原样 |
| 端到端 | `cli/tests/real_worker.rs` | 用真的 FeatherHuBERT 模型包跑一次全链路：退出码 0、`source_kind` 为 `model_package`、`model_kind` 为 `feather_hubert`、`parameter_count` 大于 0、`compatible` 为真 |

包与检查点的夹具都已有先例可循：`worker/tests/features.rs` 的 `published_package` 用 `write_model_package` 造一个微型 FeatherHuBERT 包，`worker/tests/render.rs` 的 `unknown_kind_checkpoint` 用 `save_training_checkpoint` 造一个真检查点。本切片把这两个夹具收进 `worker/tests/support/mod.rs`，供新的三个测试文件共用。

端到端测试的门控与训练、渲染那两个一致：缺 `FEATHERTALK_REQUIRE_E2E` 或缺模型包目录就跳过。它不需要 ffmpeg，也不需要 demo 视频 —— 这是本切片唯一一个只靠一个环境变量就能跑起来的端到端测试。

## 10. 范围外

- 其余四个模型命令：`import-legacy-model`、`export-model-package`、`export-onnx`、`migrate-legacy-features`。它们各自要写文件、要跑 `ort` 校验或要解析 pickle，每个都是独立切片。
- 检视旧 `.pth`/`.pth.tar`。本命令的 `source` 是目录；旧权重要先经 `import-legacy-model` 变成本仓库的格式，再被检视。
- 现场重算摘要与「深度校验」（§3）。哪天「模型」页真的需要一个「校验完整性」按钮，那是一个带进度与取消的独立命令，不是给 `inspect-model` 加一个开关。
- 五文件训练包（§2）。
- 参数量按层的分布、张量清单的逐项展示。载荷已经给出 `tensor_count` 与 `total_elements`，逐张量的表格等工作台真的要画时再说。
- GPUI「模型」页本身。本切片只保证 CLI 与协议这一层可用。
