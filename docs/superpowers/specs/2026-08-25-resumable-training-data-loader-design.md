# 可恢复训练 DataLoader 与随机状态设计

日期：2026-08-25
状态：已确认

## 1. 目标

为迁移里程碑三建立确定性、可恢复、与具体媒体解码实现解耦的 Rust 训练 DataLoader 核心，覆盖：

- Baseline 与 Mouth ROI 的单帧 target + 随机 reference 采样；
- Temporal 的相邻帧对 + 共享随机 reference 采样；
- 每个 epoch 的确定性 shuffle；
- batch 边界、最后一个不完整 batch 和 epoch 推进；
- seed、epoch、下一采样位置和采样合同的序列化恢复；
- 加载失败、训练失败或停止请求时不跳过尚未成功提交的 batch。

本切片不实现 JPEG 解码、face crop、mouth mask、特征文件物化、模型前向、optimizer、checkpoint 目录发布、global step、训练指标或预览图。具体素材读取通过公开 dataset trait 接入；optimizer/checkpoint 与指标/预览继续作为里程碑三的后续独立切片。

## 2. 现有 Python 行为与迁移修正

当前 Python 训练入口使用：

```text
DataLoader(..., shuffle=True, drop_last=False)
random.randint(0, frame_count - 1)
```

Python checkpoint 只保存 model、optimizer 和已完成 epoch，没有保存 DataLoader shuffle 或 `random` 状态。因此旧入口从 checkpoint 恢复后不会重现未中断训练的后续样本顺序和 reference 选择。

Rust 实现保留以下业务语义：

- Baseline 和 Mouth ROI：当前帧为 target，reference 从完整帧范围均匀选择；
- Temporal：target 为 `i` 与 `i + temporal_stride`，两帧共享同一个 reference；
- reference 与 Python 一样允许偶然等于某个 target，不增加“必须不同帧”的新约束；
- `drop_last` 固定为 `false`，最后一个不足 batch size 的 batch 仍参与训练。

Rust 实现新增严格恢复语义：相同状态必须产生完全相同的后续 batch、target 顺序和 reference 选择，并且不受样本物化线程数量或完成顺序影响。

## 3. Crate 与模块边界

实现位于现有 `rust/crates/feathertalk-training`：

```text
src/
  data.rs       公开配置、状态、sample plan、dataset trait 和 DataLoader
  random.rs     私有的版本化随机算法、无偏有界采样和 shuffle
```

`feathertalk-training` 增加直接 `rand` 依赖是禁止的。本切片使用小型、审核可见的固定算法，避免依赖升级改变训练顺序。

`data.rs` 不依赖 `feathertalk-project`、`feathertalk-preprocess`、图像 crate 或文件系统。后续具体资产 dataset 负责把 `TrainingSample` 物化为图像、音频和 mask；DataLoader 只负责顺序、随机决策、batch 与恢复。

## 4. 公开采样合同

### 4.1 配置

公开可序列化值类型：

```rust
pub enum SamplingKind {
    SingleFrame,
    TemporalPair,
}

pub struct SamplingConfig {
    pub kind: SamplingKind,
    pub temporal_stride: u64,
}

pub struct DataLoaderConfig {
    pub batch_size: u64,
    pub seed: u64,
    pub sampling: SamplingConfig,
}
```

固定校验：

- `batch_size > 0`；
- `SingleFrame` 必须使用 `temporal_stride == 0`；
- `TemporalPair` 必须使用 `temporal_stride >= 1`；
- `frame_count > 0`；
- Temporal 必须满足 `temporal_stride < frame_count`；
- `frame_count`、sample count 和 batch size 必须能安全转换到当前平台 `usize`；
- 任何长度、位置或 epoch 运算都使用 checked arithmetic。

提供明确构造器：

```rust
DataLoaderConfig::single_frame(batch_size, seed)
DataLoaderConfig::temporal_pair(batch_size, seed, temporal_stride)
```

Baseline 与 Mouth ROI 使用 `single_frame`；Temporal 使用 `temporal_pair`。

### 4.2 Sample plan

DataLoader 在任何 dataset worker 执行前生成不可变 sample plan：

```rust
pub enum TrainingSample {
    SingleFrame {
        target_index: u64,
        reference_index: u64,
    },
    TemporalPair {
        first_target_index: u64,
        second_target_index: u64,
        reference_index: u64,
    },
}
```

所有索引均在完整帧范围内。Temporal 的 `second_target_index` 必须严格等于 `first_target_index + temporal_stride`，且两帧只有一个 `reference_index` 字段，从类型层面防止 reference 分叉。

单帧 sample count 为 `frame_count`。Temporal sample count 为 `frame_count - temporal_stride`，shuffle 的对象是合法的 first-target 索引范围。

## 5. 固定随机算法

### 5.1 版本标识

训练状态固定记录：

```text
splitmix64_fisher_yates_v1
```

算法标识属于恢复合同。未知标识必须在生成顺序前失败，不得尝试近似恢复或回退到其他 RNG。

### 5.2 Epoch shuffle

每个 epoch 从以下稳定输入派生独立随机流：

```text
base seed + epoch + shuffle domain constant
```

使用 SplitMix64 产生 `u64`，通过无偏 rejection sampling 生成有界整数，再对 `0..sample_count` 执行 Fisher–Yates。顺序只依赖状态合同，不依赖 batch size、worker 数、线程时序、系统熵或进程全局状态。

完整 permutation 只存在于 DataLoader 运行内存，不写入训练状态。恢复时按固定算法重新生成当前 epoch permutation，避免 checkpoint 随数据集长度线性膨胀。

创建 permutation 使用 `Vec::try_reserve_exact`。分配失败返回结构化错误，不 panic，也不留下部分可用 loader。

### 5.3 Reference 选择

每个 shuffled position 从以下稳定输入派生独立随机流：

```text
base seed + epoch + shuffled position + reference domain constant
```

reference 在 `0..frame_count` 上无偏选择。该随机决策不在 dataset 的 `load_sample` 中执行，因此并发、预取、重试或不同 worker 完成顺序不会改变 reference。

不同 epoch 使用不同 shuffle/reference 域；shuffle 消耗多少随机值不会影响 reference 结果。

## 6. 可恢复状态

公开 deny-unknown-fields 的 schema one：

```rust
pub enum RandomAlgorithm {
    Splitmix64FisherYatesV1,
}

pub struct DataLoaderState {
    pub schema_version: u32,
    pub random_algorithm: RandomAlgorithm,
    pub config: DataLoaderConfig,
    pub frame_count: u64,
    pub epoch: u64,
    pub next_position: u64,
}
```

语义：

- `epoch` 是下一次 batch 所属的零基 epoch；
- `next_position` 是该 epoch permutation 中下一条尚未提交样本的位置；
- 状态始终处于 canonical safe boundary；
- 最后一个 batch 成功提交时，状态原子推进到 `epoch + 1, next_position = 0`；
- 因此合法持久化状态要求 `next_position < sample_count`；
- `u64::MAX` epoch 无法完成最后一个 batch，必须在任何状态变更前返回 overflow 错误。

恢复必须校验 schema、算法、配置、frame count、sample count 和游标。调用者提供的 dataset frame count 必须与状态完全一致。任何不匹配都在生成或加载样本前失败。

状态本身不包含 optimizer、global step、模型/素材哈希或训练 loss 配置；后续 checkpoint schema 将把 `DataLoaderState` 作为一个完整字段组合进去，并额外验证这些合同。

## 7. Dataset 与 DataLoader API

### 7.1 Dataset trait

公开最小接口：

```rust
pub trait TrainingDataset {
    type Item;

    fn frame_count(&self) -> u64;

    fn load_sample(
        &self,
        sample: &TrainingSample,
    ) -> Result<Self::Item, TrainingError>;
}
```

dataset 只物化 DataLoader 已决定的索引，不得重新选择 reference 或改变 target。具体 dataset 可以同步实现，也可由后续 worker 层按同一 plan 并行物化。

### 7.2 Prepared batch

```rust
pub struct PreparedBatch<T> { /* private fields */ }
```

只暴露只读 getter：

- `epoch()`；
- `start_position()`；
- `samples()`；
- `items()`。

字段保持私有，调用者不能伪造 commit token 或替换 sample plan。`PreparedBatch` 不实现无条件 `Clone`。

### 7.3 Loader 生命周期

```rust
TrainingDataLoader::new(dataset, config)
TrainingDataLoader::restore(dataset, state)
loader.state()
loader.prepare_next_batch()
loader.commit_batch(prepared_batch)
```

`prepare_next_batch`：

1. 根据当前 permutation 与 reference 域构造当前 batch 的全部 `TrainingSample`；
2. 逐条调用 dataset `load_sample`；
3. 全部成功后返回 `PreparedBatch`；
4. 期间绝不修改 loader 状态。

调用者完成模型前向、loss、反向和 optimizer step 后，才调用 `commit_batch`。如果加载失败、训练失败、OOM、取消或进程在 commit 前退出，持久化状态仍指向同一 batch，恢复后会重试它。

`commit_batch` 消费 prepared batch，并验证它仍对应当前 loader 的 epoch、起点、终点和状态 token：

- 重复 commit；
- commit 旧 batch；
- commit 另一个 loader/config 产生的 batch；
- loader 状态已前进后再 commit；

均返回结构化 stale-batch 错误且不修改状态。

当 commit 最后一个 batch 时，必须先成功生成下一 epoch permutation，再一次性替换 epoch、游标和 permutation。若下一 epoch overflow 或分配失败，当前状态保持不变。

## 8. 错误与安全属性

扩展 `TrainingError`，至少区分：

- 无效 DataLoader 配置；
- 无效或不兼容恢复状态；
- sample count/索引/epoch 算术溢出；
- permutation 分配失败；
- stale/foreign prepared batch；
- dataset 自身返回的现有训练错误。

禁止行为：

- 使用系统时间、系统熵或线程局部 RNG；
- 把完整 permutation 写入 JSON 状态；
- 在 `prepare_next_batch` 成功时提前推进游标；
- 加载失败后跳过样本；
- 自动修正 batch size、stride、frame count、seed 或算法版本；
- 因为未知字段或版本不匹配而静默采用默认值；
- 读取、修改或测试受保护的 `demo/kanghui_training_video_featherhubert_188_latest/`。

## 9. 测试策略

### 9.1 固定算法与采样语义

- 固定小数据集、seed 与 epoch，断言完整字面 permutation 和 reference 序列；
- 不同 epoch 产生固定但不同的顺序；
- 单帧索引和 reference 始终在范围内；
- Temporal second target 等于 first target + stride，并共享一个 reference；
- reference 不实施 target 排除；
- 有界随机采样在非 2 次幂范围工作；
- batch size 大于 sample count 时产生一个完整的尾 batch；
- sample count 不能整除 batch size 时保留最后一个部分 batch。

### 9.2 恢复等价性

- 连续运行 loader A；
- loader B 提交若干 batch 后，将 `state()` JSON round-trip；
- loader C 从该状态恢复；
- 断言 B/C 与 A 的后续 sample plan 跨当前 epoch 和下一 epoch 完全一致；
- 对 SingleFrame 与 TemporalPair 都执行该验证。

### 9.3 安全边界

- dataset 在 batch 中途返回错误时 state 不变；
- prepared batch 未 commit 时再次 prepare 得到相同 plan；
- 成功 commit 才推进游标；
- duplicate/stale/foreign commit 被拒绝且 state 不变；
- 最后一批推进到下一 epoch 的 position 0；
- epoch overflow 在 commit 前失败且 state 不变；
- schema、算法、frame count、batch size、sampling kind、stride、位置和未知字段篡改全部拒绝；
- `Vec` 长度转换和 checked arithmetic 边界不 panic。

### 9.4 回归验证

必须通过：

```powershell
cargo test -p feathertalk-training --all-targets
cargo test --workspace --all-targets
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
git diff --check
```

## 10. 文件边界

预期新增或修改：

- `rust/crates/feathertalk-training/src/data.rs`；
- `rust/crates/feathertalk-training/src/random.rs`；
- `rust/crates/feathertalk-training/src/error.rs`；
- `rust/crates/feathertalk-training/src/lib.rs`；
- `rust/crates/feathertalk-training/tests/data_loader.rs`；
- `rust/crates/feathertalk-training/tests/data_loader_recovery.rs`；
- 必要时仅为测试添加现有 workspace 依赖，不增加运行时随机依赖。

不得修改 Python 训练入口、其他 worktree、现有模型/VGG 权重或受保护 demo。

## 11. 验收标准

- SingleFrame 与 TemporalPair 采样合同精确表达 Python 业务语义；
- Temporal 类型层面只有一个共享 reference；
- 固定 seed/state 生成固定字面顺序，算法版本被持久化；
- checkpoint 无需保存完整 permutation；
- 中断并 JSON 恢复后的后续 batch 跨 epoch 与未中断运行完全一致；
- worker/物化顺序不参与随机决策；
- batch 只在显式 commit 后推进，所有失败路径保持状态不变；
- 恢复严格拒绝不兼容状态，不做静默修正；
- 常规 workspace 测试、check、clippy 和 fmt 全部通过；
- 受保护 demo 状态与本切片开始前一致。
