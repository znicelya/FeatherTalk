use burn_store::ModuleSnapshot;
use feathertalk_export::onnx::{
    ONNX_FLOAT_DATA_TYPE, OnnxModelContract, OnnxModelKind, OnnxModelProto, OnnxTensorContract,
    export_feather_hubert_onnx, validate_model_contract,
};
use feathertalk_models::feather_hubert::{FeatherHubertConfig, FeatherHubertEncoder};
use prost::Message;

use feathertalk_models::backend::CpuBackend;

fn contract() -> OnnxModelContract {
    OnnxModelContract::new(
        OnnxModelKind::FeatherHubert,
        vec![OnnxTensorContract::new("waveform", vec![1, -1])],
        vec![OnnxTensorContract::new("hidden", vec![1, -1, 1024])],
    )
}

#[test]
fn feather_hubert_export_contains_inference_graph_and_all_weights() {
    let config = FeatherHubertConfig {
        channels: 32,
        expansion: 2,
        num_blocks: 1,
        output_dim: 1024,
        dropout: 0.0,
    };
    let device = Default::default();
    let model: FeatherHubertEncoder<CpuBackend> = config.init(&device);

    let bytes = export_feather_hubert_onnx(&model, &config).unwrap();
    validate_model_contract(&bytes, &contract()).unwrap();
    let proto = OnnxModelProto::decode(bytes.as_slice()).unwrap();
    let graph = proto.graph.unwrap();

    assert_eq!(graph.input[0].name, "waveform");
    assert_eq!(graph.output[0].name, "hidden");
    assert!(
        graph
            .initializer
            .iter()
            .all(|tensor| { tensor.data_type == ONNX_FLOAT_DATA_TYPE || tensor.data_type == 7 })
    );

    let op_types = graph
        .node
        .iter()
        .map(|node| node.op_type.as_str())
        .collect::<Vec<_>>();
    assert!(op_types.iter().filter(|op| **op == "Conv").count() >= 11);
    assert!(
        op_types
            .iter()
            .filter(|op| **op == "InstanceNormalization")
            .count()
            >= 9
    );
    assert!(op_types.iter().any(|op| *op == "Erf"));
    assert!(op_types.iter().any(|op| *op == "Add"));
    assert!(!op_types.iter().any(|op| *op == "Dropout"));

    let initializer_names = graph
        .initializer
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    for snapshot in model.collect(None, None, false) {
        assert!(initializer_names.contains(snapshot.full_path().as_str()));
    }
}
