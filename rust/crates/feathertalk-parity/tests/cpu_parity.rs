use feathertalk_parity::{
    archive::GoldenArchive,
    fixture::{ForwardCase, run_cpu_forward},
    metrics::{ParityError, compare_f32},
};
use ndarray::array;

fn golden_archive() -> GoldenArchive {
    let root = env!("CARGO_MANIFEST_DIR");
    GoldenArchive::open(format!("{root}/../../tests/golden/burn-feasibility-v1.zip")).unwrap()
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
fn non_finite_values_are_an_error() {
    let actual = array![f32::NAN].into_dyn();
    let expected = array![0.0_f32].into_dyn();
    assert!(matches!(
        compare_f32(actual.view(), expected.view()),
        Err(ParityError::NonFinite { .. })
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
