//! Reviewed ONNX protobuf subset and strict public-interface validation.

use prost::Message;

pub const ONNX_IR_VERSION: i64 = 8;
pub const ONNX_OPSET_VERSION: i64 = 17;
pub const ONNX_FLOAT_DATA_TYPE: i32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxModelKind {
    FeatherHubert,
    OriginalUnet,
    MobileOneUnet,
}

impl OnnxModelKind {
    fn graph_name(self) -> &'static str {
        match self {
            Self::FeatherHubert => "feathertalk.feather_hubert",
            Self::OriginalUnet => "feathertalk.original_unet",
            Self::MobileOneUnet => "feathertalk.mobileone_unet.reparameterized",
        }
    }

    fn public_contract(self) -> OnnxModelContract {
        match self {
            Self::FeatherHubert => OnnxModelContract::new(
                self,
                vec![OnnxTensorContract::new("waveform", vec![1, -1])],
                vec![OnnxTensorContract::new("hidden", vec![1, -1, 1024])],
            ),
            Self::OriginalUnet | Self::MobileOneUnet => OnnxModelContract::new(
                self,
                vec![
                    OnnxTensorContract::new("input", vec![1, 6, 160, 160]),
                    OnnxTensorContract::new("audio", vec![1, 16, 32, 32]),
                ],
                vec![OnnxTensorContract::new("output", vec![1, 3, 160, 160])],
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnnxTensorContract {
    pub name: String,
    pub shape: Vec<i64>,
}

impl OnnxTensorContract {
    pub fn new(name: impl Into<String>, shape: Vec<i64>) -> Self {
        Self {
            name: name.into(),
            shape,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnnxModelContract {
    pub kind: OnnxModelKind,
    pub inputs: Vec<OnnxTensorContract>,
    pub outputs: Vec<OnnxTensorContract>,
}

impl OnnxModelContract {
    pub fn new(
        kind: OnnxModelKind,
        inputs: Vec<OnnxTensorContract>,
        outputs: Vec<OnnxTensorContract>,
    ) -> Self {
        Self {
            kind,
            inputs,
            outputs,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnnxValue {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: i32,
}

impl From<OnnxTensorContract> for OnnxValue {
    fn from(contract: OnnxTensorContract) -> Self {
        Self {
            name: contract.name,
            shape: contract.shape,
            dtype: ONNX_FLOAT_DATA_TYPE,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnnxGraph {
    pub name: String,
    pub inputs: Vec<OnnxValue>,
    pub outputs: Vec<OnnxValue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnnxModel {
    pub ir_version: i64,
    pub opset_version: i64,
    pub graph_present: bool,
    pub graph: OnnxGraph,
}

impl OnnxModel {
    pub fn new(kind: OnnxModelKind) -> Self {
        let contract = kind.public_contract();
        Self {
            ir_version: ONNX_IR_VERSION,
            opset_version: ONNX_OPSET_VERSION,
            graph_present: true,
            graph: OnnxGraph {
                name: kind.graph_name().to_owned(),
                inputs: contract.inputs.into_iter().map(Into::into).collect(),
                outputs: contract.outputs.into_iter().map(Into::into).collect(),
            },
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum OnnxExportError {
    #[error("ONNX protobuf encode error: {0}")]
    Encode(#[from] prost::EncodeError),
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum OnnxValidationError {
    #[error("ONNX protobuf decode error: {0}")]
    Decode(String),
    #[error("expected ONNX IR version {expected}, got {actual}")]
    IrVersion { expected: i64, actual: i64 },
    #[error("expected one default-domain opset {expected}, got {actual}")]
    Opset { expected: i64, actual: String },
    #[error("ONNX graph is missing")]
    MissingGraph,
    #[error("expected graph kind {expected}, got {actual}")]
    ModelKind {
        expected: String,
        actual: String,
    },
    #[error("expected {expected} {role} tensors, got {actual}")]
    TensorCount {
        role: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("unexpected {role} tensor {index} name: expected {expected}, got {actual}")]
    TensorName {
        role: &'static str,
        index: usize,
        expected: String,
        actual: String,
    },
    #[error("{role} tensor {name} is missing tensor type metadata")]
    MissingTensorType { role: &'static str, name: String },
    #[error("{role} tensor {name} is missing shape metadata")]
    MissingShape { role: &'static str, name: String },
    #[error("{role} tensor {name} must use f32 dtype {expected}, got {actual}")]
    DType {
        role: &'static str,
        name: String,
        expected: i32,
        actual: i32,
    },
    #[error("{role} tensor {name} rank mismatch: expected {expected}, got {actual}")]
    Rank {
        role: &'static str,
        name: String,
        expected: usize,
        actual: usize,
    },
    #[error("{role} tensor {name} dimension {index} is invalid: expected {expected}, got {actual}")]
    Dimension {
        role: &'static str,
        name: String,
        index: usize,
        expected: String,
        actual: String,
    },
}

pub fn serialize_model(model: &OnnxModel) -> Result<Vec<u8>, OnnxExportError> {
    let graph = model.graph_present.then(|| graph_proto(&model.graph));
    let proto = proto::ModelProto {
        ir_version: model.ir_version,
        opset_import: vec![proto::OperatorSetIdProto {
            domain: String::new(),
            version: model.opset_version,
        }],
        producer_name: "FeatherTalk".to_owned(),
        producer_version: env!("CARGO_PKG_VERSION").to_owned(),
        domain: "ai.feathertalk".to_owned(),
        model_version: 1,
        doc_string: String::new(),
        graph,
        metadata_props: Vec::new(),
    };
    let mut bytes = Vec::with_capacity(proto.encoded_len());
    proto.encode(&mut bytes)?;
    Ok(bytes)
}

pub fn validate_model_contract(
    bytes: &[u8],
    expected: &OnnxModelContract,
) -> Result<(), OnnxValidationError> {
    let model = proto::ModelProto::decode(bytes)
        .map_err(|error| OnnxValidationError::Decode(error.to_string()))?;
    if model.ir_version != ONNX_IR_VERSION {
        return Err(OnnxValidationError::IrVersion {
            expected: ONNX_IR_VERSION,
            actual: model.ir_version,
        });
    }
    if model.opset_import.len() != 1
        || !model.opset_import[0].domain.is_empty()
        || model.opset_import[0].version != ONNX_OPSET_VERSION
    {
        let actual = model
            .opset_import
            .iter()
            .map(|opset| format!("{:?}:{}", opset.domain, opset.version))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(OnnxValidationError::Opset {
            expected: ONNX_OPSET_VERSION,
            actual,
        });
    }
    let graph = model.graph.ok_or(OnnxValidationError::MissingGraph)?;
    let expected_graph_name = expected.kind.graph_name();
    if graph.name != expected_graph_name {
        return Err(OnnxValidationError::ModelKind {
            expected: expected_graph_name.to_owned(),
            actual: graph.name,
        });
    }
    validate_values("input", &graph.input, &expected.inputs)?;
    validate_values("output", &graph.output, &expected.outputs)?;
    Ok(())
}

fn graph_proto(graph: &OnnxGraph) -> proto::GraphProto {
    proto::GraphProto {
        node: Vec::new(),
        name: graph.name.clone(),
        initializer: Vec::new(),
        doc_string: String::new(),
        input: graph.inputs.iter().map(value_info_proto).collect(),
        output: graph.outputs.iter().map(value_info_proto).collect(),
        value_info: Vec::new(),
    }
}

fn value_info_proto(value: &OnnxValue) -> proto::ValueInfoProto {
    let dimensions = value
        .shape
        .iter()
        .enumerate()
        .map(|(index, dimension)| proto::DimensionProto {
            value: if *dimension == -1 {
                Some(proto::dimension_proto::Value::DimParam(format!(
                    "{}_dim_{index}",
                    value.name
                )))
            } else {
                Some(proto::dimension_proto::Value::DimValue(*dimension))
            },
        })
        .collect();
    proto::ValueInfoProto {
        name: value.name.clone(),
        r#type: Some(proto::TypeProto {
            tensor_type: Some(proto::TensorTypeProto {
                elem_type: value.dtype,
                shape: Some(proto::TensorShapeProto { dim: dimensions }),
            }),
        }),
        doc_string: String::new(),
    }
}

fn validate_values(
    role: &'static str,
    actual: &[proto::ValueInfoProto],
    expected: &[OnnxTensorContract],
) -> Result<(), OnnxValidationError> {
    if actual.len() != expected.len() {
        return Err(OnnxValidationError::TensorCount {
            role,
            expected: expected.len(),
            actual: actual.len(),
        });
    }
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        if actual.name != expected.name {
            return Err(OnnxValidationError::TensorName {
                role,
                index,
                expected: expected.name.clone(),
                actual: actual.name.clone(),
            });
        }
        let tensor = actual.r#type.as_ref().and_then(|value| value.tensor_type.as_ref()).ok_or_else(
            || OnnxValidationError::MissingTensorType {
                role,
                name: actual.name.clone(),
            },
        )?;
        if tensor.elem_type != ONNX_FLOAT_DATA_TYPE {
            return Err(OnnxValidationError::DType {
                role,
                name: actual.name.clone(),
                expected: ONNX_FLOAT_DATA_TYPE,
                actual: tensor.elem_type,
            });
        }
        let shape = tensor
            .shape
            .as_ref()
            .ok_or_else(|| OnnxValidationError::MissingShape {
                role,
                name: actual.name.clone(),
            })?;
        if shape.dim.len() != expected.shape.len() {
            return Err(OnnxValidationError::Rank {
                role,
                name: actual.name.clone(),
                expected: expected.shape.len(),
                actual: shape.dim.len(),
            });
        }
        for (dimension_index, (actual_dimension, expected_dimension)) in
            shape.dim.iter().zip(&expected.shape).enumerate()
        {
            let valid = match (expected_dimension, &actual_dimension.value) {
                (-1, Some(proto::dimension_proto::Value::DimParam(value))) => !value.is_empty(),
                (expected, Some(proto::dimension_proto::Value::DimValue(actual))) => {
                    *expected > 0 && *actual == *expected
                }
                _ => false,
            };
            if !valid {
                return Err(OnnxValidationError::Dimension {
                    role,
                    name: actual.name.clone(),
                    index: dimension_index,
                    expected: expected_dimension.to_string(),
                    actual: dimension_string(actual_dimension),
                });
            }
        }
    }
    Ok(())
}

fn dimension_string(dimension: &proto::DimensionProto) -> String {
    match &dimension.value {
        Some(proto::dimension_proto::Value::DimValue(value)) => value.to_string(),
        Some(proto::dimension_proto::Value::DimParam(value)) => format!("symbolic({value})"),
        None => "unknown".to_owned(),
    }
}

mod proto {
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct ModelProto {
        #[prost(int64, tag = "1")]
        pub ir_version: i64,
        #[prost(message, repeated, tag = "8")]
        pub opset_import: Vec<OperatorSetIdProto>,
        #[prost(string, tag = "2")]
        pub producer_name: String,
        #[prost(string, tag = "3")]
        pub producer_version: String,
        #[prost(string, tag = "4")]
        pub domain: String,
        #[prost(int64, tag = "5")]
        pub model_version: i64,
        #[prost(string, tag = "6")]
        pub doc_string: String,
        #[prost(message, optional, tag = "7")]
        pub graph: Option<GraphProto>,
        #[prost(message, repeated, tag = "14")]
        pub metadata_props: Vec<StringStringEntryProto>,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    pub struct OperatorSetIdProto {
        #[prost(string, tag = "1")]
        pub domain: String,
        #[prost(int64, tag = "2")]
        pub version: i64,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    pub struct StringStringEntryProto {
        #[prost(string, tag = "1")]
        pub key: String,
        #[prost(string, tag = "2")]
        pub value: String,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    pub struct GraphProto {
        #[prost(message, repeated, tag = "1")]
        pub node: Vec<NodeProto>,
        #[prost(string, tag = "2")]
        pub name: String,
        #[prost(message, repeated, tag = "5")]
        pub initializer: Vec<TensorProto>,
        #[prost(string, tag = "10")]
        pub doc_string: String,
        #[prost(message, repeated, tag = "11")]
        pub input: Vec<ValueInfoProto>,
        #[prost(message, repeated, tag = "12")]
        pub output: Vec<ValueInfoProto>,
        #[prost(message, repeated, tag = "13")]
        pub value_info: Vec<ValueInfoProto>,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    pub struct NodeProto {
        #[prost(string, repeated, tag = "1")]
        pub input: Vec<String>,
        #[prost(string, repeated, tag = "2")]
        pub output: Vec<String>,
        #[prost(string, tag = "3")]
        pub name: String,
        #[prost(string, tag = "4")]
        pub op_type: String,
        #[prost(message, repeated, tag = "5")]
        pub attribute: Vec<AttributeProto>,
        #[prost(string, tag = "6")]
        pub doc_string: String,
        #[prost(string, tag = "7")]
        pub domain: String,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    pub struct AttributeProto {
        #[prost(string, tag = "1")]
        pub name: String,
        #[prost(float, tag = "2")]
        pub f: f32,
        #[prost(int64, tag = "3")]
        pub i: i64,
        #[prost(bytes = "vec", tag = "4")]
        pub s: Vec<u8>,
        #[prost(float, repeated, packed = "false", tag = "7")]
        pub floats: Vec<f32>,
        #[prost(int64, repeated, packed = "false", tag = "8")]
        pub ints: Vec<i64>,
        #[prost(string, tag = "13")]
        pub doc_string: String,
        #[prost(int32, tag = "20")]
        pub r#type: i32,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    pub struct TensorProto {
        #[prost(int64, repeated, packed = "false", tag = "1")]
        pub dims: Vec<i64>,
        #[prost(int32, tag = "2")]
        pub data_type: i32,
        #[prost(string, tag = "8")]
        pub name: String,
        #[prost(bytes = "vec", tag = "9")]
        pub raw_data: Vec<u8>,
        #[prost(string, tag = "12")]
        pub doc_string: String,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    pub struct ValueInfoProto {
        #[prost(string, tag = "1")]
        pub name: String,
        #[prost(message, optional, tag = "2")]
        pub r#type: Option<TypeProto>,
        #[prost(string, tag = "3")]
        pub doc_string: String,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    pub struct TypeProto {
        #[prost(message, optional, tag = "1")]
        pub tensor_type: Option<TensorTypeProto>,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    pub struct TensorTypeProto {
        #[prost(int32, tag = "1")]
        pub elem_type: i32,
        #[prost(message, optional, tag = "2")]
        pub shape: Option<TensorShapeProto>,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    pub struct TensorShapeProto {
        #[prost(message, repeated, tag = "1")]
        pub dim: Vec<DimensionProto>,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    pub struct DimensionProto {
        #[prost(oneof = "dimension_proto::Value", tags = "1, 2")]
        pub value: Option<dimension_proto::Value>,
    }

    pub mod dimension_proto {
        #[derive(Clone, PartialEq, prost::Oneof)]
        pub enum Value {
            #[prost(int64, tag = "1")]
            DimValue(i64),
            #[prost(string, tag = "2")]
            DimParam(String),
        }
    }
}
