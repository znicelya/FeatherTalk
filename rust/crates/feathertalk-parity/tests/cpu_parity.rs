use feathertalk_parity::{
    archive::GoldenArchive,
    fixture::{ForwardCase, run_cpu_forward, validate_forward_fixture},
    metrics::{ParityError, compare_f32},
};
use ndarray::{ArrayD, IxDyn, array};
use serde_json::json;
use std::fs::File;
use zip::ZipWriter;

fn golden_archive() -> GoldenArchive {
    let root = env!("CARGO_MANIFEST_DIR");
    GoldenArchive::open(format!("{root}/../../tests/golden/burn-feasibility-v1.zip")).unwrap()
}

#[test]
fn cpu_forward_rejects_a_missing_archive_sidecar_before_fixture_load() {
    let temp = tempfile::tempdir().unwrap();
    let archive_path = temp.path().join("missing-sidecar.zip");
    ZipWriter::new(File::create(&archive_path).unwrap())
        .finish()
        .unwrap();
    let archive = GoldenArchive::open(archive_path).unwrap();

    assert!(matches!(
        run_cpu_forward(&archive, ForwardCase::FeatherMicro),
        Err(ParityError::ArchiveVerification(_))
    ));
}

#[test]
fn feather_contract_rejects_a_kind_mismatch_without_forward() {
    let archive = golden_archive();
    let mut fixture = archive.load_fixture("feather_micro_eval").unwrap();
    fixture.kind = "original_unet".to_owned();

    assert!(matches!(
        validate_forward_fixture(&fixture, ForwardCase::FeatherMicro),
        Err(ParityError::FixtureContract { field: "kind", .. })
    ));
}

#[test]
fn feather_contract_rejects_a_weight_member_mismatch_without_forward() {
    let archive = golden_archive();
    let mut fixture = archive.load_fixture("feather_micro_eval").unwrap();
    fixture.weights_entry = "weights/unet_production.pth".to_owned();

    assert!(matches!(
        validate_forward_fixture(&fixture, ForwardCase::FeatherMicro),
        Err(ParityError::FixtureContract {
            field: "weights_entry",
            ..
        })
    ));
}

#[test]
fn feather_contract_rejects_a_fixture_id_mismatch_without_forward() {
    let archive = golden_archive();
    let mut fixture = archive.load_fixture("feather_micro_eval").unwrap();
    fixture.id = "renamed_feather_eval".to_owned();

    assert!(matches!(
        validate_forward_fixture(&fixture, ForwardCase::FeatherMicro),
        Err(ParityError::FixtureContract {
            field: "fixture_id",
            ..
        })
    ));
}

#[test]
fn feather_contract_rejects_shape_preserving_config_changes_without_forward() {
    let archive = golden_archive();
    let mut fixture = archive.load_fixture("feather_micro_eval").unwrap();
    fixture.config.insert("dropout".to_owned(), json!(0.5));

    assert!(matches!(
        validate_forward_fixture(&fixture, ForwardCase::FeatherMicro),
        Err(ParityError::FixtureContract {
            field: "config",
            ..
        })
    ));
}

#[test]
fn unet_contract_rejects_shape_preserving_config_changes_without_forward() {
    let archive = golden_archive();
    let mut fixture = archive.load_fixture("unet_production_eval").unwrap();
    fixture.config.insert("mode".to_owned(), json!("other"));

    assert!(matches!(
        validate_forward_fixture(&fixture, ForwardCase::UnetProduction),
        Err(ParityError::FixtureContract {
            field: "config",
            ..
        })
    ));
}

#[test]
fn feather_contract_rejects_wrong_waveform_rank_without_forward() {
    let archive = golden_archive();
    let mut fixture = archive.load_fixture("feather_micro_eval").unwrap();
    fixture
        .inputs
        .insert("waveform".to_owned(), ArrayD::zeros(IxDyn(&[1360])));

    assert!(matches!(
        validate_forward_fixture(&fixture, ForwardCase::FeatherMicro),
        Err(ParityError::FixtureArrayShape {
            name: "waveform",
            ..
        })
    ));
}

#[test]
fn unet_contract_rejects_wrong_image_shape_without_forward() {
    let archive = golden_archive();
    let mut fixture = archive.load_fixture("unet_production_eval").unwrap();
    fixture
        .inputs
        .insert("image".to_owned(), ArrayD::zeros(IxDyn(&[1, 6, 160, 159])));

    assert!(matches!(
        validate_forward_fixture(&fixture, ForwardCase::UnetProduction),
        Err(ParityError::FixtureArrayShape { name: "image", .. })
    ));
}

#[test]
fn unet_contract_rejects_wrong_audio_rank_without_forward() {
    let archive = golden_archive();
    let mut fixture = archive.load_fixture("unet_production_eval").unwrap();
    fixture
        .inputs
        .insert("audio".to_owned(), ArrayD::zeros(IxDyn(&[16, 32, 32])));

    assert!(matches!(
        validate_forward_fixture(&fixture, ForwardCase::UnetProduction),
        Err(ParityError::FixtureArrayShape { name: "audio", .. })
    ));
}

#[test]
fn unet_contract_rejects_wrong_output_shape_without_forward() {
    let archive = golden_archive();
    let mut fixture = archive.load_fixture("unet_production_eval").unwrap();
    fixture
        .expected
        .insert("output".to_owned(), ArrayD::zeros(IxDyn(&[1, 3, 160, 159])));

    assert!(matches!(
        validate_forward_fixture(&fixture, ForwardCase::UnetProduction),
        Err(ParityError::FixtureArrayShape { name: "output", .. })
    ));
}

#[test]
fn forward_contract_rejects_extra_array_names_without_forward() {
    let archive = golden_archive();
    let mut fixture = archive.load_fixture("feather_micro_eval").unwrap();
    fixture
        .inputs
        .insert("unused".to_owned(), ArrayD::zeros(IxDyn(&[1])));

    assert!(matches!(
        validate_forward_fixture(&fixture, ForwardCase::FeatherMicro),
        Err(ParityError::FixtureArraySet { role: "input", .. })
    ));
}

#[test]
fn forward_contract_rejects_unknown_config_metadata_without_forward() {
    let archive = golden_archive();
    let mut fixture = archive.load_fixture("feather_micro_eval").unwrap();
    fixture.config.insert("future_flag".to_owned(), json!(true));

    assert!(matches!(
        validate_forward_fixture(&fixture, ForwardCase::FeatherMicro),
        Err(ParityError::FixtureContract {
            field: "config",
            ..
        })
    ));
}

#[test]
fn golden_forward_contracts_validate_without_forward() {
    let archive = golden_archive();
    let feather = archive.load_fixture("feather_micro_eval").unwrap();
    let unet = archive.load_fixture("unet_production_eval").unwrap();

    validate_forward_fixture(&feather, ForwardCase::FeatherMicro).unwrap();
    validate_forward_fixture(&unet, ForwardCase::UnetProduction).unwrap();
}

#[test]
fn exact_arrays_have_zero_error() {
    let values = array![[1.0_f32, -2.0], [3.0, 4.0]].into_dyn();
    let metrics = compare_f32(values.view(), values.view()).unwrap();
    assert_eq!(metrics.max_abs, 0.0);
    assert_eq!(metrics.mean_abs, 0.0);
    assert_eq!(metrics.max_relative, 0.0);
}

#[test]
fn metrics_report_max_mean_and_relative_error() {
    let actual = array![1.0_f32, 3.0].into_dyn();
    let expected = array![1.0_f32, 1.0].into_dyn();
    let metrics = compare_f32(actual.view(), expected.view()).unwrap();
    assert_eq!(metrics.max_abs, 2.0);
    assert_eq!(metrics.mean_abs, 1.0);
    assert_eq!(metrics.max_relative, 2.0);
}

#[test]
fn shape_mismatch_is_an_error() {
    let actual = array![1.0_f32, 2.0].into_dyn();
    let expected = array![[1.0_f32, 2.0]].into_dyn();
    assert!(matches!(
        compare_f32(actual.view(), expected.view()),
        Err(ParityError::ShapeMismatch { .. })
    ));
}

#[test]
fn empty_arrays_are_an_error() {
    let actual = ndarray::Array1::<f32>::zeros(0).into_dyn();
    let expected = ndarray::Array1::<f32>::zeros(0).into_dyn();
    assert!(matches!(
        compare_f32(actual.view(), expected.view()),
        Err(ParityError::EmptyArray)
    ));
}

#[test]
fn non_finite_values_are_an_error() {
    let actual = array![f32::NAN].into_dyn();
    let expected = array![0.0_f32].into_dyn();
    assert!(matches!(
        compare_f32(actual.view(), expected.view()),
        Err(ParityError::NonFinite { .. })
    ));
}

#[test]
fn expected_non_finite_values_are_an_error() {
    for expected_value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let actual = array![0.0_f32].into_dyn();
        let expected = array![expected_value].into_dyn();
        assert!(matches!(
            compare_f32(actual.view(), expected.view()),
            Err(ParityError::NonFinite { .. })
        ));
    }
}

#[test]
fn mean_accumulation_stays_finite_for_finite_f32_metrics() {
    let actual = array![f32::MAX, f32::MAX].into_dyn();
    let expected = array![1.0_f32, 1.0].into_dyn();
    let metrics = compare_f32(actual.view(), expected.view()).unwrap();
    assert!(metrics.mean_abs.is_finite(), "{metrics:?}");
    assert_eq!(metrics.mean_abs, f32::MAX);
}

#[test]
fn finite_subtraction_overflow_is_an_error() {
    let actual = array![f32::MAX].into_dyn();
    let expected = array![-f32::MAX].into_dyn();
    assert!(matches!(
        compare_f32(actual.view(), expected.view()),
        Err(ParityError::MetricOverflow {
            metric: "max_abs",
            ..
        })
    ));
}

#[test]
fn finite_relative_error_overflow_is_an_error() {
    let actual = array![f32::MAX].into_dyn();
    let expected = array![0.0_f32].into_dyn();
    assert!(matches!(
        compare_f32(actual.view(), expected.view()),
        Err(ParityError::MetricOverflow {
            metric: "max_relative",
            ..
        })
    ));
}

#[test]
fn feather_micro_matches_python_on_cpu() {
    let metrics = run_cpu_forward(&golden_archive(), ForwardCase::FeatherMicro).unwrap();
    println!("feather_micro {metrics:?}");
    assert!(metrics.max_abs <= 1e-4, "{metrics:?}");
}

#[test]
fn unet_production_matches_python_on_cpu() {
    let metrics = run_cpu_forward(&golden_archive(), ForwardCase::UnetProduction).unwrap();
    println!("unet_production {metrics:?}");
    assert!(metrics.max_abs <= 1e-4, "{metrics:?}");
}
