# PFLD Burn 推理与 parity 执行计划

> 执行规则：严格 TDD；每个任务先写最小失败测试，再实现，再运行定向测试。所有生成文件和测试结果必须在当前隔离 worktree 中确认后，才进入下一任务。

## Task 1：定义 runtime manifest、错误和路径安全契约

**目标**：把临时 checkpoint-import manifest 与产品 runtime artifact schema 分开。

**先写测试**

- manifest round-trip 使用 `deny_unknown_fields`；
- schema、model type、architecture、epoch、输入输出 shape/dtype、license 不匹配均拒绝；
- manifest 超过 1 MiB、非 UTF-8、错误/大于 64 位计数、非小写 SHA-256 拒绝；
- artifact 目录缺失文件、额外文件、符号链接、父目录遍历、已有 destination 拒绝；
- 读取权重前检查文件大小上限。

**实现**

- `PfldRuntimeManifest`、`PfldTensorSpec`、`PfldLicense` 和 `PfldRuntimeError`；
- 固定常量及有界 manifest/relative-path/hash 校验；
- 只读 immutable loader 的公共 API 骨架。

**验证**：`cargo test -p feathertalk-pfld --test artifact_contract`、`cargo fmt --all -- --check`。

## Task 2：生成并提交可复现 PFLD artifact

**先写测试**

- generator 使用 epoch-335 checkpoint，生成的 manifest tensor summary 为 `1735/910902`；
- 生成目录只含两个预期文件；
- 第二次生成与第一次 manifest/model 字节完全相同；
- source checkpoint hash 或 candidate tensor 被篡改时不发布。

**实现**

- 在 `feathertalk-pfld` 下增加 artifact generator（测试辅助或受控 CLI）；
- 复用 `feathertalk-weights` 受限 importer，但发布前重新校验 runtime schema；
- 将 `model.safetensors` 与 `manifest.json` 放入 crate artifacts；
- 增加 `.gitattributes` LF byte contract。

**验证**：定向 generator/artifact tests、`git diff --check`、sha256/字节重复生成检查。

## Task 3：实现 strict Burn runtime load/forward

**先写测试**

- committed artifact 在 `CpuBackend` 上能加载；
- forward 只接受 `[1,3,192,192]`，输出为 `[1,220]` 且全为 finite；
- 缺失/篡改/额外 tensor、错误 dtype/shape/hash、额外文件均在图执行前失败；
- 失败加载不得改变已存在 model 或留下半成品；
- WGPU smoke 测试在没有 certified adapter 时 ignored。

**实现**

- immutable `PfldRuntime<B>` 持有 manifest 和 `PfldGhostOne<B>`；
- detached load + strict `SafetensorsStore` apply；
- 输入有限性与固定 shape 检查；
- 将错误映射为可操作的 `PfldRuntimeError`。

**验证**：PFLD runtime 专项测试及 `cargo test -p feathertalk-pfld`。

## Task 4：生成固定 Python fixture

**先写测试/检查**

- fixture manifest 记录输入/输出 shape、模型 artifact hash、源 checkpoint hash、Python 与依赖版本；
- 输入字节固定、无 NaN/Inf，输出恰好 220 个 float32；
- checkpoint 或源码 hash 改变会使 generator 拒绝复用旧 fixture。

**实现**

- `rust/tools/pfld-parity/python/requirements-fixture.txt`；
- `generate_fixture.py`：直接构造确定性 `[1,3,192,192]`，加载 checkpoint `pfld_backbone`，CPU eval 输出；
- 将 fixture 放到 crate `tests/fixtures/pfld_cpu_v1/`，不提交 Python 环境本身。

**验证**：在可用 Python/torch 环境执行 generator；用 Rust fixture contract 验证 schema、哈希和有界读取。若环境缺 torch，记录阻塞并使用已生成 fixture，不把缺依赖伪装成通过。

## Task 5：CPU all-element parity 与 WGPU smoke

**先写测试**

- CPU 对 fixture 输出逐元素比较，报告 max/mean/relative error；
- 输入 shape、非有限值和 fixture hash 变化均拒绝；
- WGPU 测试明确检查 adapter，不允许静默 NdArray fallback。

**实现**

- parity helpers 和稳定错误报告；
- `cpu_parity.rs`、`wgpu_parity.rs`；
- 阈值固定为 CPU `1e-4`、WGPU `1e-3`。

**验证**：`cargo test -p feathertalk-pfld --test cpu_parity`；WGPU 无适配器时仅 ignored。

## Task 6：专项与全 workspace 验收、集成

- `cargo fmt --all -- --check`；
- PFLD 全部定向测试；
- `cargo test --workspace --all-targets`；
- 检查 artifact byte contract、`git diff --check`、保留用户指定未跟踪 demo；
- 提交隔离分支，合并 `main`，在合并后的干净 worktree 复跑关键验收并记录结果。

## 后续自动切片

PFLD 合并后按总迁移设计继续：

1. 实际媒体标准化与音视频元数据；
2. 抽帧、SCRFD/PFLD 管线和无脸/多脸/越界/模糊异常帧处理；
3. 长音频 FeatherHuBERT 分块、拼接、奇数 token 规则；
4. feature/manifest 原子写入与锁定；
5. worker RPC、训练/推理/视频合成与桌面集成；
6. 最终跨平台和发布包验收。
