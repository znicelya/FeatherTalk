//! The worker child process and the version 2 protocol.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use feathertalk_domain::{
    ClientFrame, DomainError, FrameWriter, MAX_FRAME_BYTES, ReadyFrame, ServerFrame, decode_line,
};

use crate::{ClientError, SessionOptions};

/// A decoded frame together with the exact line it was decoded from.
///
/// `--json` must reprint the worker's own bytes: this workspace compiles
/// `serde_json` without `preserve_order`, so any round trip through `Value`
/// would silently reorder object keys.
#[derive(Debug, Clone)]
pub struct FrameLine {
    pub raw: String,
    pub frame: ServerFrame,
}

/// What one bounded read of the frame channel produced.
enum FrameEvent {
    /// Boxed so the enum is not three hundred bytes wide on every timeout tick;
    /// the run loop polls this type at 100 ms for the life of a task.
    Frame(Box<FrameLine>),
    Timeout,
    Eof,
}

/// Read one newline-terminated line, refusing to buffer past `MAX_FRAME_BYTES`.
///
/// `FrameReader` is not usable here: it hands back a decoded frame and discards
/// the text. An over-long line is drained to its newline before the error is
/// reported, so the stream stays framed.
fn read_line_bounded<R: BufRead>(reader: &mut R) -> Option<Result<String, ClientError>> {
    let mut buffer: Vec<u8> = Vec::new();
    let mut discarded = false;
    loop {
        // The `fill_buf` borrow must end before `consume`, so copy inside.
        let (consumed, finished) = {
            let available = match reader.fill_buf() {
                Ok(available) => available,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Some(Err(ClientError::Io(error))),
            };
            if available.is_empty() {
                break;
            }
            let (chunk, consumed, finished) = match available.iter().position(|byte| *byte == b'\n')
            {
                Some(index) => (&available[..index], index + 1, true),
                None => (available, available.len(), false),
            };
            if buffer.len() + chunk.len() > MAX_FRAME_BYTES {
                discarded = true;
            }
            if !discarded {
                buffer.extend_from_slice(chunk);
            }
            (consumed, finished)
        };
        reader.consume(consumed);
        if finished {
            return Some(finish_line(buffer, discarded));
        }
    }
    if buffer.is_empty() && !discarded {
        return None;
    }
    Some(finish_line(buffer, discarded))
}

fn finish_line(buffer: Vec<u8>, discarded: bool) -> Result<String, ClientError> {
    if discarded {
        return Err(ClientError::Protocol(DomainError::FrameTooLong {
            limit: MAX_FRAME_BYTES,
        }));
    }
    Ok(String::from_utf8_lossy(&buffer)
        .trim_end_matches('\r')
        .to_string())
}

/// Move the worker's stdout onto its own thread.
///
/// The thread decodes but deliberately does **not** validate: validation errors
/// have to be attributed to a protocol phase, and only the main loop knows which
/// phase it is in. The thread stops after forwarding an error, because a stream
/// that has lost framing cannot be resynchronised.
fn spawn_reader(stdout: ChildStdout) -> (Receiver<Result<FrameLine, ClientError>>, JoinHandle<()>) {
    let (sender, receiver) = channel();
    let handle = std::thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        while let Some(line) = read_line_bounded(&mut reader) {
            let message = match line {
                Ok(raw) if raw.trim().is_empty() => continue,
                Ok(raw) => match decode_line::<ServerFrame>(&raw) {
                    Ok(frame) => Ok(FrameLine { raw, frame }),
                    Err(error) => Err(ClientError::Protocol(error)),
                },
                Err(error) => Err(error),
            };
            let fatal = message.is_err();
            if sender.send(message).is_err() || fatal {
                break;
            }
        }
    });
    (receiver, handle)
}

/// Drain the worker's stderr into a bounded ring so a failure report can quote
/// the last few lines. Without this pump a chatty worker would block on a full
/// pipe while the client waited for a frame that could never arrive.
fn spawn_stderr_pump(
    stderr: ChildStderr,
    tail: Arc<Mutex<VecDeque<String>>>,
    limit: usize,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            let Ok(line) = line else {
                break;
            };
            if limit == 0 {
                continue;
            }
            let mut guard = tail.lock().expect("the stderr tail mutex is not poisoned");
            while guard.len() >= limit {
                guard.pop_front();
            }
            guard.push_back(line);
        }
    })
}

/// The child process and its three pipes.
struct Transport {
    child: Child,
    /// `None` once stdin has been closed or has failed; both mean the same to
    /// the worker, which exits on EOF.
    writer: Option<FrameWriter<ChildStdin>>,
    frames: Receiver<Result<FrameLine, ClientError>>,
    reader_thread: Option<JoinHandle<()>>,
    stderr_thread: Option<JoinHandle<()>>,
    stderr_tail: Arc<Mutex<VecDeque<String>>>,
    options: SessionOptions,
}

impl Transport {
    fn spawn(
        path: &Path,
        options: SessionOptions,
        env: &[(String, String)],
    ) -> Result<Self, ClientError> {
        let mut command = Command::new(path);
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (key, value) in env {
            command.env(key, value);
        }
        let mut child = command.spawn().map_err(|source| ClientError::Spawn {
            path: path.to_path_buf(),
            source,
        })?;
        let stdin = child.stdin.take().expect("stdin was piped");
        let stdout = child.stdout.take().expect("stdout was piped");
        let stderr = child.stderr.take().expect("stderr was piped");
        let stderr_tail = Arc::new(Mutex::new(VecDeque::new()));
        let stderr_thread =
            spawn_stderr_pump(stderr, Arc::clone(&stderr_tail), options.stderr_tail_lines);
        let (frames, reader_thread) = spawn_reader(stdout);
        Ok(Self {
            child,
            writer: Some(FrameWriter::new(stdin)),
            frames,
            reader_thread: Some(reader_thread),
            stderr_thread: Some(stderr_thread),
            stderr_tail,
            options,
        })
    }

    fn stderr_tail(&self) -> Vec<String> {
        self.stderr_tail
            .lock()
            .map(|guard| guard.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Validate, then write. A write failure means the worker is gone, which is
    /// a more useful diagnosis than the raw broken-pipe error.
    fn write_frame(&mut self, frame: &ClientFrame) -> Result<(), ClientError> {
        frame.validate().map_err(ClientError::Protocol)?;
        if self.writer.is_none() {
            return Err(self.worker_gone());
        }
        let outcome = self
            .writer
            .as_mut()
            .expect("checked immediately above")
            .write_frame(frame);
        if outcome.is_err() {
            self.writer = None;
            return Err(self.worker_gone());
        }
        Ok(())
    }

    /// Build the `WorkerGone` report, giving the child a short grace period so
    /// the exit status is usually available.
    fn worker_gone(&mut self) -> ClientError {
        // Copy the deadline out first: `wait_for_exit` needs `&mut self`.
        let grace = self.options.shutdown_grace;
        let status = self.wait_for_exit(grace);
        ClientError::WorkerGone {
            status,
            stderr_tail: self.stderr_tail(),
        }
    }

    /// Poll for exit up to `timeout`. `None` means still running or unknown.
    fn wait_for_exit(&mut self, timeout: Duration) -> Option<i32> {
        let deadline = Instant::now() + timeout;
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => return status.code(),
                Ok(None) => {}
                Err(_) => return None,
            }
            if Instant::now() >= deadline {
                return None;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Close stdin. The worker treats EOF as `shutdown`.
    fn close_stdin(&mut self) {
        self.writer = None;
    }

    fn kill_and_reap(&mut self) -> Option<i32> {
        self.writer = None;
        let _ = self.child.kill();
        self.child.wait().ok().and_then(|status| status.code())
    }

    fn next_frame(&self, timeout: Duration) -> Result<FrameEvent, ClientError> {
        match self.frames.recv_timeout(timeout) {
            Ok(Ok(line)) => Ok(FrameEvent::Frame(Box::new(line))),
            Ok(Err(error)) => Err(error),
            Err(RecvTimeoutError::Timeout) => Ok(FrameEvent::Timeout),
            // The sender is dropped when the reader thread sees EOF.
            Err(RecvTimeoutError::Disconnected) => Ok(FrameEvent::Eof),
        }
    }
}

impl std::fmt::Debug for Transport {
    /// Hand-written because `FrameWriter` is not `Debug`. The pipes and the two
    /// threads say nothing useful in a failure report, so this prints only what
    /// identifies the child and whether it can still be written to.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Transport")
            .field("pid", &self.child.id())
            .field("stdin_open", &self.writer.is_some())
            .finish_non_exhaustive()
    }
}

impl Drop for Transport {
    /// Never wait on a worker that may never exit: close stdin, kill, reap, then
    /// join the two threads, which end as soon as their pipes close.
    fn drop(&mut self) {
        self.writer = None;
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(handle) = self.reader_thread.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.stderr_thread.take() {
            let _ = handle.join();
        }
    }
}

/// One worker process that has completed the handshake.
///
/// A session runs at most one task. That is the protocol's rule, not a
/// limitation of this type.
#[derive(Debug)]
pub struct WorkerSession {
    transport: Transport,
    ready: ReadyFrame,
    ready_raw: String,
    foreign_events: usize,
}

impl WorkerSession {
    pub fn spawn(path: &Path, options: SessionOptions) -> Result<Self, ClientError> {
        Self::spawn_with_env(path, options, &[])
    }

    /// Spawn with extra environment variables. Only tests use this: the CLI
    /// never injects configuration into the worker, which reads its own.
    pub fn spawn_with_env(
        path: &Path,
        options: SessionOptions,
        env: &[(String, String)],
    ) -> Result<Self, ClientError> {
        let handshake_timeout = options.handshake_timeout;
        let transport = Transport::spawn(path, options, env)?;
        let line = match transport.next_frame(handshake_timeout) {
            Ok(FrameEvent::Frame(line)) => *line,
            Ok(FrameEvent::Timeout) => {
                return Err(ClientError::Handshake {
                    reason: format!("no ready frame within {} ms", handshake_timeout.as_millis()),
                    stderr_tail: transport.stderr_tail(),
                });
            }
            Ok(FrameEvent::Eof) => {
                return Err(ClientError::Handshake {
                    reason: "the worker closed its output before sending a ready frame".to_string(),
                    stderr_tail: transport.stderr_tail(),
                });
            }
            Err(ClientError::Protocol(error)) => {
                return Err(ClientError::Handshake {
                    reason: format!("the first line was not a decodable frame: {error}"),
                    stderr_tail: transport.stderr_tail(),
                });
            }
            Err(error) => return Err(error),
        };
        // A partial move of `line.frame` is fine: `FrameLine` has no `Drop`.
        match line.frame {
            ServerFrame::Ready(ready) => {
                if let Err(error) = ready.validate() {
                    return Err(match error {
                        DomainError::ProtocolVersion { expected, actual } => {
                            ClientError::ProtocolVersion { expected, actual }
                        }
                        other => ClientError::Handshake {
                            reason: format!("the ready frame is not usable: {other}"),
                            stderr_tail: transport.stderr_tail(),
                        },
                    });
                }
                Ok(Self {
                    transport,
                    ready,
                    ready_raw: line.raw,
                    foreign_events: 0,
                })
            }
            ServerFrame::Rejected(rejected) => Err(ClientError::Rejected {
                reason: rejected.reason,
            }),
            ServerFrame::Event(event) => Err(ClientError::Handshake {
                reason: format!(
                    "the worker sent an event for task {} before its ready frame",
                    event.task_id.as_str()
                ),
                stderr_tail: transport.stderr_tail(),
            }),
        }
    }

    /// The validated handshake frame: backends, adapters, capabilities, and the
    /// command list the capability gate consults.
    pub fn ready(&self) -> &ReadyFrame {
        &self.ready
    }

    /// The handshake line exactly as the worker wrote it, for `--json`.
    pub fn ready_raw(&self) -> &str {
        &self.ready_raw
    }

    /// The worker's most recent stderr lines, oldest first.
    pub fn stderr_tail(&self) -> Vec<String> {
        self.transport.stderr_tail()
    }

    /// How many events for another task were seen and ignored. Read by the run
    /// loop's tests in Task 3; declared here so the field has a reader.
    pub fn foreign_event_count(&self) -> usize {
        self.foreign_events
    }
}
