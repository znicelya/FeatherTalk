//! SCRFD Burn model artifact and raw inference contract.

mod error;
mod manifest;

pub use error::ScrfdError;
pub use manifest::{
    SCRFD_ANCHORS, SCRFD_ARCHITECTURE_VERSION, SCRFD_INPUT_SHAPE, SCRFD_MODEL_KIND,
    SCRFD_SCHEMA_VERSION, SCRFD_SOURCE_ONNX_BYTES, SCRFD_SOURCE_ONNX_SHA256, SCRFD_SOURCE_OPSET,
    SCRFD_STRIDES, ScrfdArtifactManifest, ScrfdFileManifest, ScrfdGeneratorManifest,
    ScrfdInputManifest, ScrfdLevelManifest, ScrfdLicenseManifest, ScrfdOutputManifest,
    ScrfdSourceManifest, ScrfdWeightManifest,
};
