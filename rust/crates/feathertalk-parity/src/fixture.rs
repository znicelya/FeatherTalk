use ndarray::ArrayD;
use serde::Deserialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

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

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct FeatherForwardConfig {
    channels: usize,
    expansion: usize,
    num_blocks: usize,
    output_dim: usize,
    dropout: f64,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct UnetForwardConfig {
    channels: [usize; 5],
    mode: UnetMode,
    n_channels: usize,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
enum UnetMode {
    Hubert,
}

#[derive(Debug, Clone, Copy)]
struct ArrayContract {
    name: &'static str,
    shape: &'static [usize],
}

const FEATHER_WAVEFORM_SHAPE: &[usize] = &[1, 1360];
const FEATHER_OUTPUT_SHAPE: &[usize] = &[1, 4, 64];
const UNET_AUDIO_SHAPE: &[usize] = &[1, 16, 32, 32];
const UNET_IMAGE_SHAPE: &[usize] = &[1, 6, 160, 160];
const UNET_OUTPUT_SHAPE: &[usize] = &[1, 3, 160, 160];

const FEATHER_INPUTS: &[ArrayContract] = &[ArrayContract {
    name: "waveform",
    shape: FEATHER_WAVEFORM_SHAPE,
}];
const FEATHER_OUTPUTS: &[ArrayContract] = &[ArrayContract {
    name: "output",
    shape: FEATHER_OUTPUT_SHAPE,
}];
const UNET_INPUTS: &[ArrayContract] = &[
    ArrayContract {
        name: "audio",
        shape: UNET_AUDIO_SHAPE,
    },
    ArrayContract {
        name: "image",
        shape: UNET_IMAGE_SHAPE,
    },
];
const UNET_OUTPUTS: &[ArrayContract] = &[ArrayContract {
    name: "output",
    shape: UNET_OUTPUT_SHAPE,
}];

impl ForwardCase {
    const fn name(self) -> &'static str {
        match self {
            Self::FeatherMicro => "FeatherMicro",
            Self::UnetProduction => "UnetProduction",
        }
    }

    const fn fixture_kind(self) -> &'static str {
        match self {
            Self::FeatherMicro => "feather_hubert",
            Self::UnetProduction => "original_unet",
        }
    }

    const fn fixture_id(self) -> &'static str {
        match self {
            Self::FeatherMicro => "feather_micro_eval",
            Self::UnetProduction => "unet_production_eval",
        }
    }

    const fn weights_entry(self) -> &'static str {
        match self {
            Self::FeatherMicro => "weights/feather_micro.pth",
            Self::UnetProduction => "weights/unet_production.pth",
        }
    }

    const fn weight_kind(self) -> LegacyModelKind {
        match self {
            Self::FeatherMicro => LegacyModelKind::FeatherHubert,
            Self::UnetProduction => LegacyModelKind::OriginalUnet,
        }
    }

    const fn arrays(self) -> (&'static [ArrayContract], &'static [ArrayContract]) {
        match self {
            Self::FeatherMicro => (FEATHER_INPUTS, FEATHER_OUTPUTS),
            Self::UnetProduction => (UNET_INPUTS, UNET_OUTPUTS),
        }
    }
}

pub fn validate_forward_fixture(
    fixture: &GoldenFixture,
    case: ForwardCase,
) -> Result<(), ParityError> {
    if fixture.id != case.fixture_id() {
        return Err(ParityError::FixtureContract {
            case: case.name(),
            field: "fixture_id",
            expected: case.fixture_id().to_owned(),
            actual: fixture.id.clone(),
        });
    }
    if fixture.kind != case.fixture_kind() {
        return Err(ParityError::FixtureContract {
            case: case.name(),
            field: "kind",
            expected: case.fixture_kind().to_owned(),
            actual: fixture.kind.clone(),
        });
    }
    if fixture.weights_entry != case.weights_entry() {
        return Err(ParityError::FixtureContract {
            case: case.name(),
            field: "weights_entry",
            expected: case.weights_entry().to_owned(),
            actual: fixture.weights_entry.clone(),
        });
    }
    if case == ForwardCase::FeatherMicro {
        let expected = FeatherForwardConfig {
            channels: 32,
            expansion: 2,
            num_blocks: 2,
            output_dim: 64,
            dropout: 0.0,
        };
        let actual = parse_config::<FeatherForwardConfig>(fixture, case, &expected)?;
        if actual != expected {
            return Err(config_mismatch(case, &expected, &actual));
        }
    } else {
        let expected = UnetForwardConfig {
            channels: [32, 64, 128, 256, 512],
            mode: UnetMode::Hubert,
            n_channels: 6,
        };
        let actual = parse_config::<UnetForwardConfig>(fixture, case, &expected)?;
        if actual != expected {
            return Err(config_mismatch(case, &expected, &actual));
        }
    }
    let (inputs, outputs) = case.arrays();
    validate_array_map(case, "input", &fixture.inputs, inputs)?;
    validate_array_map(case, "expected", &fixture.expected, outputs)?;
    Ok(())
}

fn validate_array_map(
    case: ForwardCase,
    role: &'static str,
    arrays: &BTreeMap<String, ArrayD<f32>>,
    contracts: &[ArrayContract],
) -> Result<(), ParityError> {
    let expected_names = contracts
        .iter()
        .map(|contract| contract.name)
        .collect::<BTreeSet<_>>();
    let actual_names = arrays.keys().map(String::as_str).collect::<BTreeSet<_>>();
    if actual_names != expected_names {
        return Err(ParityError::FixtureArraySet {
            case: case.name(),
            role,
            expected: expected_names.into_iter().map(str::to_owned).collect(),
            actual: actual_names.into_iter().map(str::to_owned).collect(),
        });
    }

    for contract in contracts {
        let actual = &arrays[contract.name];
        if actual.shape() != contract.shape {
            return Err(ParityError::FixtureArrayShape {
                case: case.name(),
                role,
                name: contract.name,
                expected: contract.shape.to_vec(),
                actual: actual.shape().to_vec(),
            });
        }
    }
    Ok(())
}

fn parse_config<T>(
    fixture: &GoldenFixture,
    case: ForwardCase,
    expected: &T,
) -> Result<T, ParityError>
where
    T: serde::de::DeserializeOwned + std::fmt::Debug,
{
    let value = Value::Object(fixture.config.clone().into_iter().collect());
    serde_json::from_value(value).map_err(|error| ParityError::FixtureContract {
        case: case.name(),
        field: "config",
        expected: format!("{expected:?}"),
        actual: format!("invalid structured config: {error}"),
    })
}

fn config_mismatch(
    case: ForwardCase,
    expected: &impl std::fmt::Debug,
    actual: &impl std::fmt::Debug,
) -> ParityError {
    ParityError::FixtureContract {
        case: case.name(),
        field: "config",
        expected: format!("{expected:?}"),
        actual: format!("{actual:?}"),
    }
}

pub fn run_cpu_forward(
    archive: &GoldenArchive,
    case: ForwardCase,
) -> Result<ParityMetrics, ParityError> {
    archive
        .verify_sidecar_sha256()
        .map_err(ParityError::ArchiveVerification)?;
    let fixture = archive.load_fixture(case.fixture_id())?;
    validate_forward_fixture(&fixture, case)?;
    let expected = fixture
        .expected
        .get("output")
        .ok_or_else(|| ParityError::MissingArray("output".to_owned()))?;

    let temp = tempfile::tempdir().map_err(FixtureError::from)?;
    let extracted = temp.path().join("fixture");
    archive.extract_to(&extracted)?;
    let request = LegacyImportRequest {
        path: extracted.join(case.weights_entry()),
        kind: case.weight_kind(),
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
            let input = tensor_from_array::<2>("waveform", input, FEATHER_WAVEFORM_SHAPE, &device)?;
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
            let image = tensor_from_array::<4>("image", image, UNET_IMAGE_SHAPE, &device)?;
            let audio = tensor_from_array::<4>("audio", audio, UNET_AUDIO_SHAPE, &device)?;
            array_from_tensor(model.forward(image, audio))?
        }
    };

    compare_f32(actual.view(), expected.view())
}

fn tensor_from_array<const D: usize>(
    name: &'static str,
    array: &ArrayD<f32>,
    expected_shape: &[usize],
    device: &burn::tensor::Device<CpuBackend>,
) -> Result<Tensor<CpuBackend, D>, ParityError> {
    if array.ndim() != D || array.shape() != expected_shape {
        return Err(ParityError::TensorShape {
            name,
            expected: expected_shape.to_vec(),
            actual: array.shape().to_vec(),
        });
    }
    Ok(Tensor::from_data(
        TensorData::new(
            array.iter().copied().collect::<Vec<_>>(),
            array.shape().to_vec(),
        ),
        device,
    ))
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;

    #[test]
    fn fixed_rank_tensor_conversion_rejects_wrong_rank() {
        let device = Default::default();
        let waveform = ArrayD::zeros(IxDyn(&[1360]));

        assert!(matches!(
            tensor_from_array::<2>("waveform", &waveform, FEATHER_WAVEFORM_SHAPE, &device),
            Err(ParityError::TensorShape {
                name: "waveform",
                ..
            })
        ));
    }

    #[test]
    fn fixed_rank_tensor_conversion_rejects_wrong_shape() {
        let device = Default::default();
        let waveform = ArrayD::zeros(IxDyn(&[1, 1359]));

        assert!(matches!(
            tensor_from_array::<2>("waveform", &waveform, FEATHER_WAVEFORM_SHAPE, &device),
            Err(ParityError::TensorShape {
                name: "waveform",
                ..
            })
        ));
    }
}
