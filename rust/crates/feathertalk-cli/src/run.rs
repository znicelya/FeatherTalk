//! Locate the worker, run one task, choose the exit code.

use std::path::Path;

use feathertalk_client::{
    CancelToken, EventSink, SessionOptions, SessionOutcome, WorkerLocator, WorkerSession,
    generate_task_id,
};
use feathertalk_domain::{ProbeMediaParams, ProjectDirParams, Request, TaskId};

use crate::cli::{Cli, Command};
use crate::render::{HumanSink, JsonSink, capabilities_report, failure_block, render_client_error};

/// The four exit codes, fixed by the spec. Nothing else is ever returned.
pub const EXIT_COMPLETED: i32 = 0;
pub const EXIT_TASK_FAILED: i32 = 1;
pub const EXIT_CANCELLED: i32 = 2;
pub const EXIT_SESSION_ERROR: i32 = 3;

pub fn run(cli: Cli) -> i32 {
    let request = match build_request(&cli.command) {
        Ok(request) => request,
        Err(message) => {
            eprintln!("{message}");
            return EXIT_SESSION_ERROR;
        }
    };
    let path = match WorkerLocator::from_env(cli.worker.clone()).resolve() {
        Ok(path) => path,
        Err(error) => {
            eprintln!("{}", render_client_error(&error));
            return EXIT_SESSION_ERROR;
        }
    };
    // The worker inherits this process's environment and reads its own
    // configuration; the CLI injects nothing.
    let mut session = match WorkerSession::spawn(&path, SessionOptions::default()) {
        Ok(session) => session,
        Err(error) => {
            eprintln!("{}", render_client_error(&error));
            return EXIT_SESSION_ERROR;
        }
    };
    if cli.json {
        // The handshake is a protocol frame too, so machine consumers get it.
        println!("{}", session.ready_raw());
    }
    let code = match request {
        // `capabilities` needs no task: the handshake already answered it.
        None => {
            if !cli.json {
                println!("{}", capabilities_report(session.ready()));
            }
            EXIT_COMPLETED
        }
        Some(request) => run_task(&mut session, &cli, request),
    };
    let _ = session.shutdown();
    code
}

/// Build the request, or `None` for `capabilities`.
///
/// Only empty arguments are rejected here. Whether a path exists, is a project,
/// or is decodable media is the worker's judgement, and duplicating it in the
/// CLI would produce two answers that can disagree.
fn build_request(command: &Command) -> Result<Option<Request>, String> {
    match command {
        Command::Capabilities => Ok(None),
        Command::ValidateProject { project_dir } => {
            reject_empty(project_dir, "工程目录")?;
            Ok(Some(Request::ValidateProject(ProjectDirParams {
                project_dir: project_dir.clone(),
            })))
        }
        Command::ProbeMedia { input } => {
            reject_empty(input, "输入文件")?;
            Ok(Some(Request::ProbeMedia(ProbeMediaParams {
                input: input.clone(),
            })))
        }
    }
}

fn reject_empty(path: &Path, label: &str) -> Result<(), String> {
    if path.as_os_str().is_empty() {
        return Err(format!("{label}不能为空。"));
    }
    Ok(())
}

fn run_task(session: &mut WorkerSession, cli: &Cli, request: Request) -> i32 {
    let task_id = match resolve_task_id(cli.task_id.as_deref()) {
        Ok(task_id) => task_id,
        Err(message) => {
            eprintln!("{message}");
            return EXIT_SESSION_ERROR;
        }
    };
    let cancel = CancelToken::new();
    install_cancel_handler(&cancel);
    let mut human = HumanSink::new(cli.quiet);
    let mut json = JsonSink;
    let sink: &mut dyn EventSink = if cli.json { &mut json } else { &mut human };
    match session.run(task_id, request, &cancel, sink) {
        SessionOutcome::Completed { result } => {
            // Under --json the completed frame already carried the result.
            if !cli.json {
                let value = result.unwrap_or_else(|| serde_json::json!({}));
                let text =
                    serde_json::to_string_pretty(&value).unwrap_or_else(|_| value.to_string());
                println!("{text}");
            }
            EXIT_COMPLETED
        }
        SessionOutcome::Failed(error) => {
            eprintln!("{}", failure_block(&error));
            EXIT_TASK_FAILED
        }
        SessionOutcome::Cancelled => {
            eprintln!("任务已取消。");
            EXIT_CANCELLED
        }
        SessionOutcome::SessionError(error) => {
            eprintln!("{}", render_client_error(&error));
            EXIT_SESSION_ERROR
        }
    }
}

fn resolve_task_id(requested: Option<&str>) -> Result<TaskId, String> {
    match requested {
        Some(text) => TaskId::parse(text).map_err(|error| {
            format!("任务 ID 无效：{error}\n格式为 13 位毫秒时间戳、连字符、8 位小写十六进制。")
        }),
        None => generate_task_id().map_err(|error| format!("无法生成任务 ID：{error}")),
    }
}

/// Ctrl-C bumps the token and does nothing else: one atomic add, no allocation,
/// no locks. All the escalation lives in the client's run loop.
///
/// Failing to install is reported and ignored — the task can still run, it just
/// cannot be interrupted politely, and refusing to work would be worse.
fn install_cancel_handler(cancel: &CancelToken) {
    let token = cancel.clone();
    if let Err(error) = ctrlc::set_handler(move || token.request()) {
        eprintln!("无法注册 Ctrl-C 处理器：{error}");
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn capabilities_needs_no_task() {
        assert!(
            build_request(&Command::Capabilities)
                .expect("capabilities always builds")
                .is_none()
        );
    }

    #[test]
    fn an_empty_path_is_rejected_in_chinese() {
        // Clap catches this first on the command line, but `run` is a library
        // entry point too, so the guard has to hold on its own.
        let error = build_request(&Command::ValidateProject {
            project_dir: PathBuf::new(),
        })
        .expect_err("an empty project directory is refused");
        assert_eq!(error, "工程目录不能为空。");

        let error = build_request(&Command::ProbeMedia {
            input: PathBuf::new(),
        })
        .expect_err("an empty input is refused");
        assert_eq!(error, "输入文件不能为空。");
    }

    #[test]
    fn a_malformed_task_id_explains_the_format() {
        let error = resolve_task_id(Some("nope")).expect_err("a short id is refused");
        assert!(error.contains("任务 ID 无效"), "{error}");
        assert!(error.contains("13 位毫秒时间戳"), "{error}");
    }
}
