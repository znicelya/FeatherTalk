use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// The command line. Help text is Chinese, because the user is.
#[derive(Debug, Parser)]
#[command(
    name = "feathertalk",
    version,
    about = "FeatherTalk 命令行客户端",
    long_about = "通过标准输入输出驱动 feathertalk-worker 执行单个任务。\n\n\
                  标准输出只有结果，进度输出在标准错误，因此可以安全重定向。\n\
                  退出码：0 完成，1 任务失败，2 已取消，3 会话错误。"
)]
pub struct Cli {
    /// 工作进程可执行文件路径，默认依次查找环境变量与本程序同目录
    #[arg(long, global = true, value_name = "PATH")]
    pub worker: Option<PathBuf>,

    /// 按行输出原始协议帧，供程序解析
    #[arg(long, global = true)]
    pub json: bool,

    /// 不输出进度，只保留结果与错误
    #[arg(long, global = true, conflicts_with = "json")]
    pub quiet: bool,

    /// 指定任务 ID：13 位毫秒时间戳、连字符、8 位小写十六进制
    #[arg(long, global = true, value_name = "ID")]
    pub task_id: Option<String>,

    #[command(subcommand)]
    pub command: Command,
}

/// The task commands, kebab-cased by clap: `validate-project`, `probe-media`,
/// `normalize-media`, `capabilities`.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// 校验工程目录
    ValidateProject {
        /// 工程目录
        project_dir: PathBuf,
    },
    /// 探测媒体文件信息
    ProbeMedia {
        /// 输入的音视频文件
        input: PathBuf,
    },
    /// 归一化媒体文件：输出 25fps 视频与 16kHz 单声道音频
    NormalizeMedia {
        /// 输入的音视频文件
        input: PathBuf,
        /// 输出目录，归一化后的视频与音频写入其中
        output_dir: PathBuf,
    },
    /// 打印工作进程的握手信息：后端、设备、支持的命令
    Capabilities,
}
