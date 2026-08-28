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
            buffer: Vec::with_capacity(MAX_FRAME_BYTES),
        }
    }

    pub fn read_frame<T: DeserializeOwned>(&mut self) -> Option<Result<T, DomainError>> {
        loop {
            self.buffer.clear();
            let mut content_len = 0usize;
            let mut saw_input = false;
            let mut too_long = false;
            let mut terminated = false;

            while !terminated {
                let (content_chunk_len, has_newline) = {
                    let available = match self.inner.fill_buf() {
                        Ok(available) => available,
                        Err(error) => {
                            return Some(Err(DomainError::MalformedFrame {
                                reason: error.to_string(),
                            }));
                        }
                    };
                    if available.is_empty() {
                        break;
                    }
                    saw_input = true;
                    let newline = available.iter().position(|byte| *byte == b'\n');
                    let content_chunk_len = newline.unwrap_or(available.len());
                    if !too_long {
                        let remaining = MAX_FRAME_BYTES - content_len;
                        if content_chunk_len > remaining {
                            self.buffer.extend_from_slice(&available[..remaining]);
                            too_long = true;
                        } else {
                            self.buffer
                                .extend_from_slice(&available[..content_chunk_len]);
                            content_len += content_chunk_len;
                        }
                    }
                    let consumed = content_chunk_len + usize::from(newline.is_some());
                    self.inner.consume(consumed);
                    (content_chunk_len, newline.is_some())
                };
                if has_newline {
                    terminated = true;
                } else if !too_long {
                    debug_assert!(content_len >= content_chunk_len);
                }
            }

            if !saw_input {
                return None;
            }
            if too_long {
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
