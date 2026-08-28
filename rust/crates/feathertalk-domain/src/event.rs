use serde::{Deserialize, Serialize};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

use crate::{DomainError, PROTOCOL_VERSION, TaskError, TaskId, TaskStage};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Progress {
    pub completed: u64,
    pub total: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metrics {
    pub samples_per_second: Option<f64>,
    pub eta_seconds: Option<f64>,
    pub vram_bytes: Option<u64>,
}

impl Metrics {
    pub fn empty() -> Self {
        Self::default()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Event {
    pub protocol_version: u32,
    pub task_id: TaskId,
    pub emitted_at: String,
    pub stage: TaskStage,
    pub progress: Option<Progress>,
    pub metrics: Metrics,
    pub error: Option<TaskError>,
}

impl Event {
    pub fn new(task_id: TaskId, emitted_at: &str, stage: TaskStage) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            task_id,
            emitted_at: emitted_at.to_owned(),
            stage,
            progress: None,
            metrics: Metrics::empty(),
            error: None,
        }
    }

    pub fn validate(&self) -> Result<(), DomainError> {
        if self.protocol_version != PROTOCOL_VERSION {
            return Err(DomainError::ProtocolVersion {
                expected: PROTOCOL_VERSION,
                actual: self.protocol_version,
            });
        }
        OffsetDateTime::parse(&self.emitted_at, &Rfc3339).map_err(|_| {
            DomainError::InvalidField {
                field: "emitted_at",
                reason: "must be RFC 3339".into(),
            }
        })?;
        if let Some(progress) = self.progress
            && let Some(total) = progress.total
            && progress.completed > total
        {
            return Err(DomainError::InvalidField {
                field: "progress",
                reason: "completed must not exceed total".into(),
            });
        }
        let is_failed = matches!(self.stage, TaskStage::Failed { .. });
        match (&self.error, is_failed) {
            (Some(error), true) => error.validate()?,
            (None, false) => {}
            (None, true) => {
                return Err(DomainError::InvalidField {
                    field: "error",
                    reason: "a failed stage must carry the error payload".into(),
                });
            }
            (Some(_), false) => {
                return Err(DomainError::InvalidField {
                    field: "error",
                    reason: "only a failed stage may carry an error payload".into(),
                });
            }
        }
        Ok(())
    }
}
