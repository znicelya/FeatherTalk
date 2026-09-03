mod support;

use std::fs;
use std::path::Path;

use feathertalk_inference::MouthMasking;
use feathertalk_training::{TrainingDataset, TrainingSample};
use feathertalk_training_data::{
    FrameSample, ProjectTrainingDataset, TrainingDataError, TrainingItem,
};
use support::{
    GradientFrameReader, INNER_SIZE, downgrade_to_preparing, inner_planes, locked_project,
    mouth_rect, write_features,
};

fn open_dataset(project_dir: &Path) -> ProjectTrainingDataset<GradientFrameReader> {
    ProjectTrainingDataset::open_with_reader(project_dir, GradientFrameReader).unwrap()
}

fn single_frame(item: &TrainingItem) -> &FrameSample {
    match item {
        TrainingItem::SingleFrame(sample) => sample,
    }
}

fn single_frame_sample(target_index: u64, reference_index: u64) -> TrainingSample {
    TrainingSample::SingleFrame {
        target_index,
        reference_index,
    }
}

#[test]
fn frame_count_and_root_come_from_the_locked_project() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let canonical = project_dir.canonicalize().unwrap();
    assert_eq!(dataset.frame_count(), 4);
    assert_eq!(dataset.root(), canonical.as_path());
}

#[test]
fn single_frame_image_is_the_reference_then_the_masked_target() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let item = dataset.load_sample(&single_frame_sample(2, 0)).unwrap();
    let image = single_frame(&item).image();
    let plane = INNER_SIZE * INNER_SIZE;
    let reference = inner_planes(&project_dir, 0, MouthMasking::Keep);
    let masked = inner_planes(&project_dir, 2, MouthMasking::Blackout);
    let kept = inner_planes(&project_dir, 2, MouthMasking::Keep);
    assert_eq!(&image[..3 * plane], reference.as_slice());
    assert_eq!(&image[3 * plane..], masked.as_slice());
    assert_ne!(&image[3 * plane..], kept.as_slice());
}

#[test]
fn the_masked_half_is_black_where_the_mouth_is() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let item = dataset.load_sample(&single_frame_sample(2, 0)).unwrap();
    let image = single_frame(&item).image();
    let plane = INNER_SIZE * INNER_SIZE;
    let mask = mouth_rect(&project_dir, 2);
    let centre_x = (mask.x + mask.width / 2) as usize;
    let centre_y = (mask.y + mask.height / 2) as usize;
    assert_eq!(image[3 * plane + centre_y * INNER_SIZE + centre_x], 0.0);
}

#[test]
fn the_target_is_the_unmasked_target_frame() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let item = dataset.load_sample(&single_frame_sample(2, 0)).unwrap();
    let expected = inner_planes(&project_dir, 2, MouthMasking::Keep);
    assert_eq!(single_frame(&item).target(), expected.as_slice());
}

#[test]
fn every_tensor_has_the_length_the_losses_expect() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let item = dataset.load_sample(&single_frame_sample(2, 0)).unwrap();
    let sample = single_frame(&item);
    let plane = INNER_SIZE * INNER_SIZE;
    assert_eq!(sample.image().len(), 6 * plane);
    assert_eq!(sample.audio().len(), 16 * 32 * 32);
    assert_eq!(sample.target().len(), 3 * plane);
    assert_eq!(sample.mouth_mask().len(), plane);
}

#[test]
fn the_mouth_mask_is_one_inside_the_roi_and_zero_outside() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let item = dataset.load_sample(&single_frame_sample(2, 0)).unwrap();
    let plane = single_frame(&item).mouth_mask();
    let mask = mouth_rect(&project_dir, 2);
    let ones = plane.iter().filter(|value| **value == 1.0).count();
    let inside = (mask.y as usize) * INNER_SIZE + mask.x as usize;
    assert_eq!(ones, (mask.width * mask.height) as usize);
    assert_eq!(plane[inside], 1.0);
    assert_eq!(plane[0], 0.0);
    assert!(plane.iter().all(|value| *value == 0.0 || *value == 1.0));
}

#[test]
fn the_feature_token_count_must_match_the_frame_count() {
    let (_temp, project_dir) = locked_project(4);
    write_features(&project_dir, 3);
    let error =
        ProjectTrainingDataset::open_with_reader(&project_dir, GradientFrameReader).unwrap_err();
    assert!(matches!(
        error,
        TrainingDataError::FeatureShape {
            expected_tokens: 8,
            actual_tokens: 6,
            dims: 1024,
            ..
        }
    ));
}

#[test]
fn a_project_whose_assets_are_not_locked_is_rejected() {
    let (_temp, project_dir) = locked_project(4);
    downgrade_to_preparing(&project_dir);
    let error =
        ProjectTrainingDataset::open_with_reader(&project_dir, GradientFrameReader).unwrap_err();
    assert!(matches!(error, TrainingDataError::Project { .. }));
}

#[test]
fn a_missing_frame_file_names_its_index() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    fs::remove_file(project_dir.join("assets/frames/000002.jpg")).unwrap();
    let error = dataset.load_sample(&single_frame_sample(2, 0)).unwrap_err();
    assert!(error.to_string().contains("unable to read frame 2"));
}

#[test]
fn a_missing_landmark_file_names_its_index() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    fs::remove_file(project_dir.join("assets/landmarks/000002.lms")).unwrap();
    let error = dataset.load_sample(&single_frame_sample(2, 0)).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("unable to read landmarks for frame 2"));
}

#[test]
fn a_corrupt_landmark_file_names_its_index() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let lines = vec![String::from("1 2 3"); 110];
    let path = project_dir.join("assets/landmarks/000002.lms");
    fs::write(path, lines.join("\n")).unwrap();
    let error = dataset.load_sample(&single_frame_sample(2, 0)).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("unable to read landmarks for frame 2"));
}

#[test]
fn a_frame_index_past_the_end_is_rejected() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let error = dataset.load_sample(&single_frame_sample(4, 0)).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("frame index 4 is out of range for 4 frames"));
}
