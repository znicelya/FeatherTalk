//! A scripted stand-in for `feathertalk-worker`.
//!
//! Compiled as a `[[bin]]` of `feathertalk-client` so integration tests spawn a
//! real process and talk to it over real pipes. The behaviour is chosen by
//! `FT_FAKE_WORKER_SCENARIO`; one scenario per test case, each one a straight
//! line of writes with no branching, so a failing test names its own script.
//!
//! Only `feathertalk-domain`, `serde_json`, and std are available here, because
//! a `[[bin]]` target cannot use dev-dependencies. Hence the fixed timestamp.

use std::io::{BufReader, StdinLock, Write};
use std::time::Duration;

use feathertalk_domain::{
    AdapterInfo, AdapterKind, Backend, Capabilities, ClientFrame, Event, FrameReader,
    PROTOCOL_VERSION, Progress, ReadyFrame, RejectedFrame, ServerFrame, TaskId, TaskKind,
    TaskStage, encode_line,
};

/// The scenario selector. Tests set it; there is no command line.
const SCENARIO_ENV: &str = "FT_FAKE_WORKER_SCENARIO";

/// A fixed RFC 3339 instant. `Event::validate` only checks the format.
const EMITTED_AT: &str = "2026-09-01T00:00:00Z";

/// A well-formed task id that is never the one the client sends.
const FOREIGN_TASK_ID: &str = "1787900000000-0000beef";

type Reader = FrameReader<BufReader<StdinLock<'static>>>;

fn main() {
    let scenario = std::env::var(SCENARIO_ENV).unwrap_or_else(|_| "ready-complete".to_string());
    let mut reader = FrameReader::new(BufReader::new(std::io::stdin().lock()));
    match scenario.as_str() {
        // Never writes anything, so the handshake has to time out.
        "silent" => park(),
        // A syntactically valid frame that is not `ready`.
        "no-ready" => {
            let task_id = TaskId::parse(FOREIGN_TASK_ID).expect("the constant id is valid");
            write_frame(&ServerFrame::Event(stage_event(
                &task_id,
                TaskStage::Queued,
            )));
            park();
        }
        // Truncated JSON: decodable as a line, not as a frame.
        "invalid-line" => {
            write_line("{\"frame\":\"ready\"");
            park();
        }
        // A structurally valid `ready` frame from a future protocol.
        "bad-version" => {
            let mut value =
                serde_json::to_value(ready(default_commands())).expect("ready serializes");
            value["data"]["protocol_version"] = serde_json::json!(99);
            write_line(&serde_json::to_string(&value).expect("the patched value serializes"));
            park();
        }
        // A worker that advertises no commands at all: `ReadyFrame::validate` rejects it.
        "empty-commands" => {
            let mut value =
                serde_json::to_value(ready(default_commands())).expect("ready serializes");
            value["data"]["supported_commands"] = serde_json::json!([]);
            write_line(&serde_json::to_string(&value).expect("the patched value serializes"));
            park();
        }
        // Refuses the session outright instead of going ready.
        "rejected-handshake" => {
            write_frame(&ServerFrame::Rejected(RejectedFrame {
                protocol_version: PROTOCOL_VERSION,
                reason: "工作进程当前无法接受新会话".to_string(),
            }));
        }
        // Goes ready and then stops reading, so only a kill reaps it.
        "hang-after-ready" => {
            write_frame(&ready(default_commands()));
            park();
        }
        // Floods stderr before going ready, to exercise the tail bound.
        "noisy-stderr" => {
            for index in 0..200 {
                eprintln!("stderr line {index}");
            }
            write_frame(&ready(default_commands()));
            serve_one_task(&mut reader);
        }
        // The happy path: ready, one progress event, then completed.
        "ready-complete" => {
            write_frame(&ready(default_commands()));
            serve_one_task(&mut reader);
        }
        other => {
            eprintln!("unknown fake worker scenario: {other}");
            std::process::exit(97);
        }
    }
}

/// Both task commands the real worker offers when a media toolchain resolved.
fn default_commands() -> Vec<TaskKind> {
    vec![TaskKind::ValidateProject, TaskKind::ProbeMedia]
}

fn ready(commands: Vec<TaskKind>) -> ServerFrame {
    ServerFrame::Ready(ReadyFrame {
        protocol_version: PROTOCOL_VERSION,
        worker_version: "fake-0".to_string(),
        backends: vec![Backend::Cpu],
        adapters: vec![AdapterInfo {
            id: "cpu-0".to_string(),
            name: "Fake CPU".to_string(),
            backend: Backend::Cpu,
            kind: AdapterKind::Cpu,
            certified: true,
            vram_bytes: None,
        }],
        supported_commands: commands,
        capabilities: Capabilities {
            training: false,
            wgpu_training: false,
            onnx_validation: false,
            ffmpeg: true,
        },
    })
}

/// Read frames until a `start` arrives, then run the scripted happy path.
fn serve_one_task(reader: &mut Reader) {
    let Some(task_id) = wait_for_start(reader) else {
        return;
    };
    let mut preparing = stage_event(&task_id, TaskStage::Preparing);
    preparing.progress = Some(Progress {
        completed: 1,
        total: Some(2),
    });
    write_frame(&ServerFrame::Event(preparing));
    write_frame(&ServerFrame::Event(completed(&task_id)));
}

/// Block until the client sends `start`. Returns `None` on shutdown or EOF.
fn wait_for_start(reader: &mut Reader) -> Option<TaskId> {
    loop {
        match reader.read_frame::<ClientFrame>()? {
            Ok(ClientFrame::Start(start)) => return Some(start.task_id),
            Ok(ClientFrame::Cancel(_)) => continue,
            Ok(ClientFrame::Shutdown(_)) => return None,
            Err(error) => {
                eprintln!("fake worker could not decode a client frame: {error}");
                return None;
            }
        }
    }
}

fn stage_event(task_id: &TaskId, stage: TaskStage) -> Event {
    Event::new(task_id.clone(), EMITTED_AT, stage)
}

fn completed(task_id: &TaskId) -> Event {
    let mut event = stage_event(task_id, TaskStage::Completed);
    event.result = Some(serde_json::json!({ "checked": true }));
    event
}

fn write_frame(frame: &ServerFrame) {
    let line = encode_line(frame).expect("the scripted frame serializes");
    write_line(line.trim_end());
}

fn write_line(line: &str) {
    let mut stdout = std::io::stdout().lock();
    stdout
        .write_all(line.as_bytes())
        .expect("stdout accepts a line");
    stdout.write_all(b"\n").expect("stdout accepts a newline");
    stdout.flush().expect("stdout flushes");
}

/// Stay alive until the parent kills us. Several scenarios exist to prove the
/// client's deadlines and reaping actually work.
fn park() -> ! {
    loop {
        std::thread::sleep(Duration::from_millis(50));
    }
}
