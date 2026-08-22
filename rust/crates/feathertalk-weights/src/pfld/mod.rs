use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::source::{DEFAULT_MAX_FILE_BYTES, DEFAULT_MAX_TENSOR_COUNT, DEFAULT_MAX_TOTAL_ELEMENTS};

pub const PFLD_CHECKPOINT_EPOCH: u64 = 335;
pub const PFLD_ARCHITECTURE_VERSION: &str = "burn-pfld-structure-v1";

#[derive(Debug, Clone)]
pub struct PfldImportRequest {
    pub checkpoint: PathBuf,
    pub destination_dir: PathBuf,
    pub max_file_bytes: u64,
    pub max_tensor_count: usize,
    pub max_total_elements: u64,
}

impl Default for PfldImportRequest {
    fn default() -> Self {
        Self {
            checkpoint: PathBuf::new(),
            destination_dir: PathBuf::new(),
            max_file_bytes: DEFAULT_MAX_FILE_BYTES,
            max_tensor_count: DEFAULT_MAX_TENSOR_COUNT,
            max_total_elements: DEFAULT_MAX_TOTAL_ELEMENTS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorAudit {
    pub tensor_count: usize,
    pub total_elements: u64,
    pub keys: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorSummary {
    pub tensor_count: usize,
    pub total_elements: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PfldSourceManifest {
    pub file_name: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PfldModelArtifact {
    pub format: String,
    pub file_name: String,
    pub sha256: String,
    pub tensor_count: usize,
    pub total_elements: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PfldIgnoredTensors {
    pub batch_norm_counters: TensorAudit,
    pub localization: TensorAudit,
    pub auxiliarynet: Option<TensorAudit>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PfldImportManifest {
    pub schema_version: u32,
    pub model_type: String,
    pub architecture_version: String,
    pub source: PfldSourceManifest,
    pub epoch: u64,
    pub backbone: TensorSummary,
    pub model: PfldModelArtifact,
    pub ignored: PfldIgnoredTensors,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PfldImportReport {
    pub destination_dir: PathBuf,
    pub manifest: PfldImportManifest,
    pub applied: Vec<String>,
}
