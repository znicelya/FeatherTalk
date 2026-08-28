use std::path::PathBuf;

use feathertalk_domain::{
    LegacyModelKind, OnnxExportKind, ProbeMediaParams, Request, TaskKind, TrainParams,
    TrainingMode, UnetVariant,
};

#[test]
fn every_task_kind_has_exactly_one_request_variant() {
    let requests = sample_requests();
    assert_eq!(requests.len(), 13);
    let mut kinds: Vec<TaskKind> = requests.iter().map(Request::kind).collect();
    kinds.sort();
    kinds.dedup();
    assert_eq!(kinds.len(), 13, "two requests reported the same TaskKind");
    for kind in TaskKind::ALL {
        assert!(kinds.contains(&kind), "no request maps to {kind:?}");
    }
}

#[test]
fn requests_use_adjacent_tagging_and_round_trip() {
    for request in sample_requests() {
        let json = serde_json::to_string(&request).unwrap();
        assert!(
            json.starts_with(r#"{"command":"#),
            "unexpected wire shape: {json}"
        );
        assert_eq!(serde_json::from_str::<Request>(&json).unwrap(), request);
    }
}

#[test]
fn probe_media_has_the_exact_wire_form() {
    let request = Request::ProbeMedia(ProbeMediaParams {
        input: PathBuf::from("a.mov"),
    });
    assert_eq!(
        serde_json::to_string(&request).unwrap(),
        r#"{"command":"probe_media","params":{"input":"a.mov"}}"#
    );
}

#[test]
fn params_reject_unknown_fields() {
    let bad = r#"{"command":"probe_media","params":{"input":"a.mov","extra":1}}"#;
    assert!(serde_json::from_str::<Request>(bad).is_err());
}

#[test]
fn request_rejects_unknown_outer_fields() {
    let bad = r#"{"command":"probe_media","params":{"input":"a.mov"},"extra":1}"#;
    assert!(serde_json::from_str::<Request>(bad).is_err());
}

#[test]
fn render_treats_preview_as_a_frame_cap_not_a_separate_command() {
    let json = r#"{"command":"render","params":{"project_dir":"p","checkpoint":"c","audio":"a.wav","output":"o.mp4","max_output_frames":120}}"#;
    let request: Request = serde_json::from_str(json).unwrap();
    let Request::Render(params) = &request else {
        panic!("expected Render");
    };
    assert_eq!(params.max_output_frames, Some(120));
    assert_eq!(request.kind(), TaskKind::Render);
}

#[test]
fn training_mode_and_variant_are_independent_dimensions() {
    let params = TrainParams {
        project_dir: PathBuf::from("p"),
        mode: TrainingMode::Temporal,
        variant: UnetVariant::MobileOneUnet,
        epochs: 4,
        resume: true,
    };
    let json = serde_json::to_string(&params).unwrap();
    assert!(json.contains(r#""mode":"temporal""#));
    assert!(json.contains(r#""variant":"mobile_one_unet""#));
    assert_eq!(serde_json::from_str::<TrainParams>(&json).unwrap(), params);
}

fn sample_requests() -> Vec<Request> {
    use feathertalk_domain::{
        ExportModelPackageParams, ExportOnnxParams, ExtractFeaturesParams, ExtractFramesParams,
        ImportLegacyModelParams, InspectModelParams, MigrateLegacyFeaturesParams,
        NormalizeMediaParams, ProjectDirParams, RenderParams,
    };

    let p = || PathBuf::from("p");
    vec![
        Request::ProbeMedia(ProbeMediaParams { input: p() }),
        Request::NormalizeMedia(NormalizeMediaParams {
            input: p(),
            output_dir: p(),
        }),
        Request::ValidateProject(ProjectDirParams { project_dir: p() }),
        Request::LockAssetPackage(ProjectDirParams { project_dir: p() }),
        Request::ExtractFrames(ExtractFramesParams {
            project_dir: p(),
            video: p(),
        }),
        Request::ExtractFeatures(ExtractFeaturesParams {
            project_dir: p(),
            audio: p(),
        }),
        Request::Train(TrainParams {
            project_dir: p(),
            mode: TrainingMode::Baseline,
            variant: UnetVariant::OriginalUnet,
            epochs: 1,
            resume: false,
        }),
        Request::Render(RenderParams {
            project_dir: p(),
            checkpoint: p(),
            audio: p(),
            output: p(),
            max_output_frames: None,
        }),
        Request::InspectModel(InspectModelParams { source: p() }),
        Request::ImportLegacyModel(ImportLegacyModelParams {
            source: p(),
            kind: LegacyModelKind::FeatherHubert,
            destination: p(),
        }),
        Request::ExportModelPackage(ExportModelPackageParams {
            source: p(),
            destination: p(),
        }),
        Request::ExportOnnx(ExportOnnxParams {
            source: p(),
            kind: OnnxExportKind::FeatherHubert,
            destination: p(),
        }),
        Request::MigrateLegacyFeatures(MigrateLegacyFeaturesParams {
            source: p(),
            destination: p(),
        }),
    ]
}
