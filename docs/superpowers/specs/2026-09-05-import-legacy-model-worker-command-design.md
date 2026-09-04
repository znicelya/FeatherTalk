# 旧模型导入 worker 命令设计

## 1. 目标与范围

本切片实现 `Request::ImportLegacyModel`，并在 CLI 暴露 `feathertalk import-legacy-model`。命令读取受限 Rust pickle 导入器支持的旧 `.pth`/`.pth.tar`，将权重转换成标准模型包目录，并返回来源、目标、模型类型、摘要和 tensor 统计。命令不修改旧文件，不覆盖已有目标，不读取 demo 中受保护的 `.MOV`。

## 2. 输入与输出

领域参数保持冻结：

```text
ImportLegacyModelParams { source: PathBuf, kind: LegacyModelKind, destination: PathBuf }
```

`source` 必须是绝对、普通文件，扩展名为 `.pth` 或 `.pth.tar`；`destination` 必须是绝对、尚不存在的目录，父目录必须已存在。许可清单固定从 `source.parent()/LICENSES.json` 读取。创建时间由 worker 生成当前 UTC RFC3339，最低应用版本取 `CARGO_PKG_VERSION`。

当前标准包写入器支持 `FeatherHubert` 与 `OriginalUnet`，因此这两种类型执行导入；`Pfld` 和 `MobileOneUnet` 在缺少对应标准包描述/写入器时返回 `MODEL_INCOMPATIBLE`，不生成部分目录、不回退到随机权重。

## 3. 数据流

```text
admission -> cancellation check -> import legacy weights
          -> build standard package (staging + validation + no-clobber publish)
          -> cancellation check -> completed JSON
```

FeatherHuBERT 复用 `build_feather_hubert_package`，它会先检查 checkpoint 元数据、加载到 `FeatherHubertEncoder`，再由标准包写入器审计 safetensors。Original UNet 使用 `LegacyImportRequest` + `import_into` 将旧键映射到生产配置，再调用 `write_model_package`。两条路径都使用 CPU `NdArray`，不启动外部工具。

## 4. 阶段、取消与错误

命令报告 `Preparing` 后进入 `Importing`，完成时由 runtime 发送 `Completed`。导入器本身是同步操作，token 在开始前和发布前检查；取消不会发布目标。底层 `WeightImportError`、`PackageError` 统一映射为 `ErrorCode::ModelIncompatible`、中文摘要「模型导入失败」、阶段 `Importing`，detail 保留英文技术信息并经过长度限制。路径/许可文件缺失同样属于模型配置错误，不伪装成 worker 崩溃。

## 5. 结果载荷

完成事件携带：

```json
{
  "kind": "import_legacy_model",
  "model_kind": "feather_hubert|original_unet",
  "architecture_version": "...",
  "source": "...",
  "destination": "...",
  "source_sha256": "...",
  "model_sha256": "...",
  "tensor_count": 0,
  "total_elements": 0
}
```

字段来自最终标准包 manifest；目标发布前不会声称成功。`source_sha256` 是旧文件摘要，`model_sha256` 是标准 safetensors 摘要。

## 6. 握手与 CLI

导入不需要 ffmpeg、模型目录或 VGG19，因此 `supported_commands` 无条件宣告 `ImportLegacyModel`，位于 `InspectModel` 之后。CLI 使用三个位置参数：`import-legacy-model <SOURCE> <KIND> <DESTINATION>`，kind 使用 clap 的 `feather-hubert|pfld|original-unet|mobileone-unet`，空路径在客户端拒绝，绝对路径和扩展名由 worker 判断。

## 7. 测试

- worker 单测：空/相对/非文件源、已有目标、取消、未支持 kind；使用临时目录，绝不读取 `.MOV`。
- worker 导入测试：从 weights crate 的 golden fixture 构造 FeatherHuBERT/Original UNet 的成功包，检查 manifest、文件集合和结果 JSON。
- CLI 单测：kind 映射、空路径拒绝和请求字段原样传递。
- CLI gated e2e：若 `FEATHERTALK_WORKER_HUBERT_DIR` 不适用，则使用 demo 目录中的 `.pth` 与同目录 `LICENSES.json`（没有许可文件时跳过）；绝不自动读取视频。

## 8. 不在范围内

不新增协议字段、不改变 `feathertalk-domain`、不迁移旧优化器、不实现 PFLD/MobileOne 标准包写入器、不把 `.pth` 接入 runtime 推理。PFLD 专用 artifact 仍由现有 `feathertalk-weights` API 和独立工具负责。
