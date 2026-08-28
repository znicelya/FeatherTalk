use std::io::{BufRead, Write};

use serde::{Serialize, de::DeserializeOwned};

use crate::{DomainError, MAX_FRAME_BYTES, decode_line, encode_line};

pub struct FrameReader<R: BufRead> {
    inner: R,
    buffer: Vec<u8>,
}

impl<R: BufRead> FrameReader<R> {
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            buffer: Vec::new(),
        }
    }

    pub fn read_frame<T: DeserializeOwned>(&mut self) -> Option<Result<T, DomainError>> {
        loop {
            self.buffer.clear();
            let read = match self.inner.read_until(b'\n', &mut self.buffer) {
                Ok(0) => return None,
                Ok(read) => read,
                Err(error) => {
                    return Some(Err(DomainError::MalformedFrame {
                        reason: error.to_string(),
                    }));
                }
            };
            if read > MAX_FRAME_BYTES {
                return Some(Err(DomainError::FrameTooLong {
                    limit: MAX_FRAME_BYTES,
                }));
            }
            let text = match std::str::from_utf8(&self.buffer) {
                Ok(text) => text,
                Err(error) => {
                    return Some(Err(DomainError::MalformedFrame {
                        reason: error.to_string(),
                    }));
                }
            };
            if text.trim().is_empty() {
                continue;
            }
            return Some(decode_line(text));
        }
    }
}

pub struct FrameWriter<W: Write> {
    inner: W,
}

impl<W: Write> FrameWriter<W> {
    pub fn new(inner: W) -> Self {
        Self { inner }
    }

    pub fn write_frame<T: Serialize>(&mut self, value: &T) -> Result<(), DomainError> {
        let line = encode_line(value)?;
        self.inner
            .write_all(line.as_bytes())
            .and_then(|()| self.inner.write_all(b"\n"))
            .and_then(|()| self.inner.flush())
            .map_err(|error| DomainError::MalformedFrame {
                reason: error.to_string(),
            })
    }

    pub fn into_inner(self) -> W {
        self.inner
    }
}
