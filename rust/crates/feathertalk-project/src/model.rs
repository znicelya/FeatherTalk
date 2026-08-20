use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectManifest {
    pub schema_version: u32,
    pub project_id: String,
    pub display_name: String,
    pub asset_package: String,
    pub default_model: ModelSelection,
    pub task_history: Vec<TaskHistoryEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSelection {
    OriginalUnet,
    MobileOneUnet,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskHistoryEntry {
    pub task_id: String,
    pub kind: String,
    pub status: TaskHistoryStatus,
    pub updated_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskHistoryStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssetManifest {
    pub schema_version: u32,
    pub state: AssetPackageState,
    pub video_fps: u32,
    pub audio_sample_rate: u32,
    pub audio_channels: u16,
    pub frame_count: u64,
    pub frame_width: u32,
    pub frame_height: u32,
    pub feature_type: FeatureType,
    pub feature_shape: [u64; 3],
    pub landmark_model_sha256: String,
    pub feature_model_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetPackageState {
    Preparing,
    Locked,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureType {
    FeatherHubert,
}
