use serde::{Serialize, de::DeserializeOwned};

use crate::{DomainError, PROTOCOL_VERSION};

pub const MAX_FRAME_BYTES: usize = 1_048_576;

pub fn encode_line<T: Serialize>(value: &T) -> Result<String, DomainError> {
    let line = serde_json::to_string(value).map_err(|error| DomainError::MalformedFrame {
        reason: error.to_string(),
    })?;
    if line.len() > MAX_FRAME_BYTES {
        return Err(DomainError::FrameTooLong {
            limit: MAX_FRAME_BYTES,
        });
    }
    Ok(line)
}

pub fn decode_line<T: DeserializeOwned>(line: &str) -> Result<T, DomainError> {
    let line = line.strip_suffix('\n').unwrap_or(line);
    if line.len() > MAX_FRAME_BYTES {
        return Err(DomainError::FrameTooLong {
            limit: MAX_FRAME_BYTES,
        });
    }
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Err(DomainError::MalformedFrame {
            reason: "empty line".into(),
        });
    }
    serde_json::from_str(trimmed).map_err(|error| DomainError::MalformedFrame {
        reason: error.to_string(),
    })
}

pub fn check_protocol_version(actual: u32) -> Result<(), DomainError> {
    if actual == PROTOCOL_VERSION {
        Ok(())
    } else {
        Err(DomainError::ProtocolVersion {
            expected: PROTOCOL_VERSION,
            actual,
        })
    }
}
