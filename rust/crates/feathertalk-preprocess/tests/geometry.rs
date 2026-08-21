use feathertalk_preprocess::{
    FaceBoundingBox, MaskRect, PFLD_LANDMARK_COUNT, PreprocessError, compute_face_bbox,
    default_crop_spec, read_landmarks,
};

fn landmarks_file(x1: f32, x31: f32, y52: f32) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("face.lms");
    let mut lines = (0..PFLD_LANDMARK_COUNT)
        .map(|_| "0 0".to_owned())
        .collect::<Vec<_>>();
    lines[1] = format!("{x1} 0");
    lines[31] = format!("{x31} 0");
    lines[52] = format!("0 {y52}");
    std::fs::write(&path, lines.join("\n")).unwrap();
    (dir, path)
}

#[test]
fn computes_square_bbox_with_python_truncation() {
    let (_dir, path) = landmarks_file(10.9, 50.8, 20.7);
    let landmarks = read_landmarks(&path).unwrap();
    let bbox = compute_face_bbox(&landmarks).unwrap();
    assert_eq!(
        bbox,
        FaceBoundingBox {
            xmin: 10,
            ymin: 20,
            xmax: 50,
            ymax: 60
        }
    );
}

#[test]
fn rejects_non_positive_bbox_width() {
    let (_dir, path) = landmarks_file(50.0, 10.0, 20.0);
    let landmarks = read_landmarks(&path).unwrap();
    assert!(matches!(
        compute_face_bbox(&landmarks),
        Err(PreprocessError::InvalidGeometry { .. })
    ));
}

#[test]
fn default_crop_spec_matches_python_constants() {
    let spec = default_crop_spec();
    assert_eq!(spec.crop_size, 168);
    assert_eq!(spec.inner_size, 160);
    assert_eq!(spec.border, 4);
    assert_eq!(
        spec.mouth_mask,
        MaskRect {
            x: 5,
            y: 5,
            width: 150,
            height: 145
        }
    );
    assert_eq!(spec.crop_size, spec.inner_size + 2 * spec.border);
}
