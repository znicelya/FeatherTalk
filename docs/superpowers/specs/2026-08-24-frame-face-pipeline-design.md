# 抽帧与 SCRFD/PFLD 质量管线设计

日期：2026-08-24
状态：已批准执行

## 1. 目标

在已验证的 25 FPS 标准化视频上，安全地生成固定编号帧、执行 SCRFD 人脸筛选与 PFLD 关键点解码，分类并持久化每帧质量异常；只有全量帧通过策略后才原子发布帧、landmarks 和审计报告。

本切片不实现 FeatherHuBERT、worker RPC、UI、训练或视频合成。

## 2. 输入与输出契约

输入是 feathertalk-media::NormalizedMediaLayout 中的 video_25fps.mp4，并要求调用方提供已验证的标准化视频 metadata（25/1 FPS、正尺寸）。输出目录由调用方指定，最终布局为：

    frames/000000.jpg ... frames/{N-1:06}.jpg
    landmarks/000000.lms ... landmarks/{N-1:06}.lms
    quality.json

抽帧使用固定 FFmpeg argv，不调用 shell；视频路径和输出路径分别作为独立 OsString 参数。每帧输出先写到 invocation-owned 临时目录，临时文件要求 regular、non-symlink、non-empty，写入后 fsync 并计算 SHA-256/字节数。

帧数由已验证视频 metadata 的正整数 frame_count 决定。每个索引恰好一次，文件名为六位十进制；禁止覆盖已有最终目录。

## 3. 模型组合接口

定义 crate-private/public test seam：

- FrameDecoder::decode(index, path) -> Result<DecodedFrame, PipelineError>
- FaceDetector::detect(&DecodedFrame) -> Result<Vec<FaceDetection>, PipelineError>
- LandmarkPredictor::predict(&DecodedFrame, &FaceDetection) -> Result<PFLDLandmarks, PipelineError>

生产适配器只负责把图像字节转换为模型输入并调用现有 Burn runtime；本切片先实现确定性文件/命令执行层和组合编排，避免在没有图像解码依赖时伪造生产模型调用。

FaceDetection 必须包含 bbox、score、5 个 SCRFD keypoints；组合层按 score 降序、原始索引升序确定主脸。

## 4. 质量策略

固定默认阈值：

- SCRFD confidence：0.50；
- NMS IoU：0.40；
- 主脸数量必须恰好 1；
- bbox 必须为有限正值，且与图像相交面积/图像面积至少 0.10；
  - 勘误（2026-09-02）：分母应为 bbox 自身面积，即「bbox 至少 10% 落在图像内」。按图像面积计算会变成隐含的最小人脸尺寸门槛，实测会拒掉 demo 视频的全部帧。修正记录见 `2026-09-02-frame-model-adapters-design.md` §5。
- 110 个 PFLD 点必须有限，位于图像范围内；
- 模糊判定使用确定性 Laplacian 方差输入接口；默认阈值 20.0，低于阈值分类为 blurred_frame。

异常代码只允许：

- face_not_found
- multiple_faces
- bbox_out_of_bounds
- landmark_invalid
- blurred_frame
- frame_decode_failed
- frame_write_failed
- model_failed

每条异常包含 frame index、code、用户可读 summary、technical detail、recoverable action（exclude_frame 或 rerun_frame）。默认策略不静默排除：任何异常使整次构建失败，旧输出保持不变；调用方可在后续 UI 切片中选择修复后重跑。

## 5. quality.json

    {
      "schema_version": 1,
      "frame_count": N,
      "accepted_count": N,
      "frames": [
        {
          "index": 0,
          "frame_file": "frames/000000.jpg",
          "landmark_file": "landmarks/000000.lms",
          "frame_bytes": 0,
          "frame_sha256": "...",
          "landmark_sha256": "...",
          "face_score": 0.0,
          "bbox": [0.0, 0.0, 0.0, 0.0],
          "blur_variance": 0.0
        }
      ],
      "anomalies": []
    }

JSON 使用 deny_unknown_fields，最大 16 MiB；相对路径仅允许固定 frames/ 和 landmarks/ 前缀。所有 hash 为 64 位小写十六进制。

## 6. 原子发布与恢复

创建 sibling staging directory .feathertalk-frame-build-{pid}-{counter}，拒绝已存在目录和 symlink。成功后：

1. fsync 每个文件和 quality.json；
2. fsync staging 目录；
3. 将旧的 frames/、landmarks/、quality.json 移到 invocation-owned backup；
4. rename staging 子目录和 quality.json 到最终位置；
5. fsync 输出父目录；
6. 删除本次 backup。

任一步失败，反向 rename 已安装项并报告 primary/rollback error；不删除调用方拥有的旧文件或其他 temp。

## 7. 验收边界

必须测试：

- 固定六位命名、精确帧数、命令 argv 和 hostile native path；
- 输出缺失、symlink、空文件、超 16 MiB；
- 无脸、多脸、bbox 越界、关键点异常、模糊、模型失败；
- 异常报告字段和恢复动作；
- 旧输出在抽帧/模型/提交任一失败后不变；
- staging/backup collision、late rename failure、rollback failure；
- 全成功时 frame/landmark hash、byte count、quality.json 可读且严格校验；
- 无 ffmpeg 时返回稳定工具错误。

## 8. 排除项

不添加 image/OpenCV 运行时依赖，不实现真实像素解码；生产图像适配器留在后续跨平台/worker 切片，由 fake decoder 覆盖组合契约。

