# FeatherHuBERT 长音频特征与原子素材提交设计

日期：2026-08-24  
状态：已批准执行

## 1. 目标

在已完成媒体标准化和帧脸质量管线的基础上，按现有 Python 实现生成完整 FeatherHuBERT 特征，并以可恢复、不可半提交的方式写入素材包。该切片不实现训练、worker RPC、GPUI 或视频合成。

## 2. Python 兼容边界

输入必须是 16 kHz、单声道、finite `f32` 波形。默认参数固定为：

- HuBERT kernel：400 samples；
- HuBERT stride：320 samples；
- 长音频 chunk：320,000 samples；
- 完整 chunk 的读取范围：`start .. start + chunk_samples - stride + kernel`，即额外 80 samples；
- 完整 chunk 起点：`0, chunk_samples, 2*chunk_samples, ...`；
- tail 从 `chunk_samples * floor(samples / chunk_samples)` 开始，不重复拼接；
- 总目标 token 数：`0`（少于 400 samples）或 `(samples - 80) / 320` 的整数下取整；
- 拼接后不足目标 token 时尾部补零，超过时裁剪；
- token 数为奇数时删除最后一个 token；
- 最终布局为 `[video_frames, 2, 1024]`，其中 `video_frames = even_tokens / 2`。

边界规划必须使用 checked arithmetic，拒绝溢出、空输入、非 finite 波形和错误输出维度。编码器每个 chunk 接收独立的 `&[f32]`，输出按 row-major `[tokens, output_dim]` 展开。

## 3. crate 边界

新增 `feathertalk-audio`：

- 纯 Rust 波形归一化、chunk plan、token 拼接/裁剪/补零和奇数 token 处理；
- `ChunkEncoder` 注入式 seam，便于不依赖 GPU 的确定性测试；
- 带版本 header 的 little-endian `f32` 特征文件读写；
- 生成 feature 临时文件、校验 hash/shape、准备 `assets.json`，并通过现有 `feathertalk-project` 原子 manifest API 完成最后提交。

`feathertalk-models` 依赖 `feathertalk-audio`，提供 `BurnFeatherHubertEncoder<B>` 适配器：将每个 chunk 转成 Burn `[1, samples]` tensor，调用现有 eval 模型并导出 finite `f32` rows。音频 crate 不依赖模型 crate，避免循环依赖。

## 4. 特征文件格式

文件名固定为 `assets/features/feather_hubert.f32`。header 使用固定 ASCII magic 和 little-endian 整数，包含 schema version、token 数、pair width（2）和 feature dimension（1024）；header 后紧跟 row-major `f32` payload。读取时限制文件大小、拒绝 symlink、短读、未知 version、shape 不匹配、NaN/Inf 和尾随字节。

## 5. 原子提交与恢复

所有新文件写入 `assets/.feathertalk-feature-build-{pid}-{counter}` sibling staging：

1. 写并 fsync feature 文件；
2. 计算字节数、SHA-256 和 `[frame_count,2,1024]`；
3. 写 preparing/locked manifest 到 staging；
4. 在最终替换前验证旧 manifest 未被锁定、目标路径和父目录不是 symlink；
5. 以同目录原子 rename 安装 feature，再原子写入 `assets.json`；
6. fsync 支持的平台父目录；
7. 失败时删除本次 staging，旧 feature 和旧 manifest 保持不变；若 feature 已安装而 manifest 安装失败，删除新 feature 并恢复旧 feature（错误同时保留 primary/rollback 信息）。

锁定条件沿用 `feathertalk-project::AssetManifest::validate_locked`，并额外要求 feature header、payload 字节数和 manifest shape 一致。已锁定素材包不可由该 API 覆盖。

## 6. 测试策略

- chunk 边界：399/400/719/720、恰好 320000、跨多个 chunk、tail、不足补零和奇数 token；
- normalization：每批独立 mean/variance、常量/非 finite 拒绝；
- fake encoder：记录精确 chunk slices，验证拼接顺序和稳定输出；
- feature format：round-trip、未知 header、短读、尾随字节、非 finite、大小上限和 symlink；
- atomic commit：成功提交、旧 preparing 替换、旧 locked 拒绝、manifest 晚期失败回滚、staging collision 和旧输出保护；
- Burn adapter：CPU micro model 输出 shape/finiteness，WGPU 没有认证 adapter 时 ignored；
- 全量通过 `cargo test --workspace --all-targets`、`cargo check --workspace --all-targets`、fmt、clippy 和 diff check。

## 7. 非目标

不在本切片实现音频容器解码/重采样（输入由已完成 media crate 提供）、Python 运行时、训练或模型权重导入。真实 Burn adapter 只复用已有 FeatherHuBERT 模型，不新增 checkpoint 格式。
