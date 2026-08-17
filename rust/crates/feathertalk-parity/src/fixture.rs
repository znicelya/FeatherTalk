use ndarray::ArrayD;
use serde_json::Value;
use std::collections::BTreeMap;

use burn::tensor::{Tensor, TensorData};
use feathertalk_models::{
    backend::CpuBackend, feather_hubert::FeatherHubertConfig, unet::OriginalUnetConfig,
};
use feathertalk_weights::{LegacyImportRequest, LegacyModelKind, import_into};

use crate::{
    archive::{FixtureError, GoldenArchive},
    metrics::{ParityError, ParityMetrics, compare_f32},
};

#[derive(Debug)]
pub struct GoldenFixture {
    pub id: String,
    pub schema_version: u32,
    pub fixture_set: String,
    pub kind: String,
    pub weights_entry: String,
    pub config: BTreeMap<String, Value>,
    pub optimizer: Option<BTreeMap<String, Value>>,
    pub loss: Option<String>,
    pub expected_mode: Option<String>,
    pub inputs: BTreeMap<String, ArrayD<f32>>,
    pub expected: BTreeMap<String, ArrayD<f32>>,
    pub metrics: BTreeMap<String, f64>,
    pub scalars: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForwardCase {
    FeatherMicro,
    UnetProduction,
}

pub fn run_cpu_forward(
    archive: &GoldenArchive,
    case: ForwardCase,
) -> Result<ParityMetrics, ParityError> {
    let (fixture_id, weight_kind) = match case {
        ForwardCase::FeatherMicro => ("feather_micro_eval", LegacyModelKind::FeatherHubert),
        ForwardCase::UnetProduction => ("unet_production_eval", LegacyModelKind::OriginalUnet),
    };
    let fixture = archive.load_fixture(fixture_id)?;
    let expected = fixture
        .expected
        .get("output")
        .ok_or_else(|| ParityError::MissingArray("output".to_owned()))?;

    let temp = tempfile::tempdir().map_err(FixtureError::from)?;
    let extracted = temp.path().join("fixture");
    archive.extract_to(&extracted)?;
    let request = LegacyImportRequest {
        path: extracted.join(&fixture.weights_entry),
        kind: weight_kind,
        ..Default::default()
    };
    let device = Default::default();

    let actual = match case {
        ForwardCase::FeatherMicro => {
            let mut model = FeatherHubertConfig::parity_micro().init::<CpuBackend>(&device);
            import_into::<CpuBackend, _>(&mut model, &request)?;
            let input = fixture
                .inputs
                .get("waveform")
                .ok_or_else(|| ParityError::MissingArray("waveform".to_owned()))?;
            let input = tensor_from_array::<2>(input, &device);
            array_from_tensor(model.forward(input))?
        }
        ForwardCase::UnetProduction => {
            let mut model = OriginalUnetConfig::production().init::<CpuBackend>(&device);
            import_into::<CpuBackend, _>(&mut model, &request)?;
            let image = fixture
                .inputs
                .get("image")
                .ok_or_else(|| ParityError::MissingArray("image".to_owned()))?;
            let audio = fixture
                .inputs
                .get("audio")
                .ok_or_else(|| ParityError::MissingArray("audio".to_owned()))?;
            let image = tensor_from_array::<4>(image, &device);
            let audio = tensor_from_array::<4>(audio, &device);
            array_from_tensor(model.forward(image, audio))?
        }
    };

    compare_f32(actual.view(), expected.view())
}

fn tensor_from_array<const D: usize>(
    array: &ArrayD<f32>,
    device: &burn::tensor::Device<CpuBackend>,
) -> Tensor<CpuBackend, D> {
    Tensor::from_data(
        TensorData::new(
            array.iter().copied().collect::<Vec<_>>(),
            array.shape().to_vec(),
        ),
        device,
    )
}

fn array_from_tensor<const D: usize>(
    tensor: Tensor<CpuBackend, D>,
) -> Result<ArrayD<f32>, ParityError> {
    let shape = tensor.dims().to_vec();
    let values = tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| ParityError::TensorData(error.to_string()))?;
    ArrayD::from_shape_vec(shape, values).map_err(|error| ParityError::Array(error.to_string()))
}
