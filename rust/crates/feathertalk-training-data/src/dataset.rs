use std::path::{Path, PathBuf};

use feathertalk_audio::{FeatureMatrix, read_feature_file};
use feathertalk_inference::{
    BgrFrame, FrameReader, JpegFrameReader, MouthMasking, RenderGeometry, build_face_crop,
    build_inner_image_planes, build_unet_audio_window,
};
use feathertalk_preprocess::{
    CropSpec, MaskRect, MouthRoiSpec, audio_window_indices, compute_face_bbox, default_crop_spec,
    default_mouth_roi_spec, mouth_roi_rect, read_landmarks,
};
use feathertalk_project::validate_project_dir;
use feathertalk_training::{TrainingDataset, TrainingError, TrainingSample};

use crate::TrainingDataError;

const FEATURE_FILE: &str = "assets/features/feather_hubert.f32";
const FEATURE_DIMS: usize = 1024;
const TOKENS_PER_FRAME: usize = 2;
const INNER_SIZE: usize = 160;

/// One training frame, flattened into the planes the losses consume.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameSample {
    image: Vec<f32>,
    audio: Vec<f32>,
    target: Vec<f32>,
    mouth_mask: Vec<f32>,
}

impl FrameSample {
    /// `[6, 160, 160]`: the reference frame's planes followed by the mouth-masked target planes.
    pub fn image(&self) -> &[f32] {
        &self.image
    }

    /// `[16, 32, 32]`: the eight-slot audio window centred on the target frame.
    pub fn audio(&self) -> &[f32] {
        &self.audio
    }

    /// `[3, 160, 160]`: the unmasked target planes.
    pub fn target(&self) -> &[f32] {
        &self.target
    }

    /// `[1, 160, 160]`: one inside the mouth ROI, zero outside it.
    pub fn mouth_mask(&self) -> &[f32] {
        &self.mouth_mask
    }
}

/// What one training sample loads into.
#[derive(Debug, Clone, PartialEq)]
pub enum TrainingItem {
    SingleFrame(FrameSample),
}

/// A locked project directory presented as a training dataset.
#[derive(Debug)]
pub struct ProjectTrainingDataset<R: FrameReader> {
    root: PathBuf,
    frame_count: usize,
    frame_width: u32,
    frame_height: u32,
    features: FeatureMatrix,
    reader: R,
    crop: CropSpec,
    mouth_roi: MouthRoiSpec,
    geometry: RenderGeometry,
}

struct LoadedFrame {
    crop: BgrFrame,
    mouth: MaskRect,
}

fn project_error(path: &Path, message: String) -> TrainingDataError {
    TrainingDataError::Project {
        path: path.to_path_buf(),
        message,
    }
}

fn features_error(path: &Path, message: String) -> TrainingDataError {
    TrainingDataError::Features {
        path: path.to_path_buf(),
        message,
    }
}

fn frame_error(index: usize, path: &Path, message: String) -> TrainingDataError {
    TrainingDataError::Frame {
        index,
        path: path.to_path_buf(),
        message,
    }
}

fn landmark_error(index: usize, path: &Path, message: String) -> TrainingDataError {
    TrainingDataError::Landmarks {
        index,
        path: path.to_path_buf(),
        message,
    }
}

fn sample_error(index: usize, message: String) -> TrainingDataError {
    TrainingDataError::Sample { index, message }
}

impl ProjectTrainingDataset<JpegFrameReader> {
    /// Opens a locked project directory and decodes its frames as JPEG files.
    pub fn open(project_dir: &Path) -> Result<Self, TrainingDataError> {
        Self::open_with_reader(project_dir, JpegFrameReader::default())
    }
}

impl<R: FrameReader> ProjectTrainingDataset<R> {
    /// Opens a locked project directory with a caller-supplied frame reader.
    pub fn open_with_reader(project_dir: &Path, reader: R) -> Result<Self, TrainingDataError> {
        let project = validate_project_dir(project_dir)
            .map_err(|error| project_error(project_dir, error.to_string()))?;
        let manifest = project.asset_package().manifest();
        if manifest.frame_count == 0 {
            return Err(project_error(
                project_dir,
                "the asset package declares zero frames".to_owned(),
            ));
        }
        let Ok(frame_count) = usize::try_from(manifest.frame_count) else {
            return Err(project_error(
                project_dir,
                format!("{} frames do not fit in memory", manifest.frame_count),
            ));
        };
        let Some(expected_tokens) = frame_count.checked_mul(TOKENS_PER_FRAME) else {
            return Err(project_error(
                project_dir,
                format!("{frame_count} frames overflow the feature token count"),
            ));
        };
        let root = project.root().to_path_buf();
        let feature_path = root.join(FEATURE_FILE);
        let features = read_feature_file(&feature_path)
            .map_err(|error| features_error(&feature_path, error.to_string()))?;
        if features.dims() != FEATURE_DIMS || features.tokens() != expected_tokens {
            return Err(TrainingDataError::FeatureShape {
                path: feature_path,
                expected_tokens,
                actual_tokens: features.tokens(),
                dims: features.dims(),
            });
        }
        Ok(Self {
            root,
            frame_count,
            frame_width: manifest.frame_width,
            frame_height: manifest.frame_height,
            features,
            reader,
            crop: default_crop_spec(),
            mouth_roi: default_mouth_roi_spec(),
            geometry: RenderGeometry::standard(),
        })
    }

    /// The canonical project root the dataset reads from.
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn resolve_index(&self, index: u64) -> Result<usize, TrainingDataError> {
        match usize::try_from(index) {
            Ok(resolved) if resolved < self.frame_count => Ok(resolved),
            _ => Err(TrainingDataError::FrameIndexOutOfRange {
                index,
                frame_count: self.frame_count as u64,
            }),
        }
    }

    fn load_frame(&self, index: usize) -> Result<LoadedFrame, TrainingDataError> {
        let frame_path = self.root.join(format!("assets/frames/{index:06}.jpg"));
        let frame = self
            .reader
            .read(index, &frame_path)
            .map_err(|error| frame_error(index, &frame_path, error.to_string()))?;
        if frame.width() != self.frame_width || frame.height() != self.frame_height {
            return Err(frame_error(
                index,
                &frame_path,
                format!(
                    "frame is {}x{} but the asset package declares {}x{}",
                    frame.width(),
                    frame.height(),
                    self.frame_width,
                    self.frame_height
                ),
            ));
        }
        let landmark_path = self.root.join(format!("assets/landmarks/{index:06}.lms"));
        let landmarks = read_landmarks(&landmark_path)
            .map_err(|error| landmark_error(index, &landmark_path, error.to_string()))?;
        let bbox = compute_face_bbox(&landmarks)
            .map_err(|error| landmark_error(index, &landmark_path, error.to_string()))?;
        let mouth = mouth_roi_rect(&landmarks, &self.crop, &self.mouth_roi)
            .map_err(|error| landmark_error(index, &landmark_path, error.to_string()))?;
        let crop = build_face_crop(&frame, &bbox, &self.geometry)
            .map_err(|error| frame_error(index, &frame_path, error.to_string()))?;
        Ok(LoadedFrame { crop, mouth })
    }

    fn inner_planes(
        &self,
        index: usize,
        face_crop: &BgrFrame,
        masking: MouthMasking,
    ) -> Result<Vec<f32>, TrainingDataError> {
        let planes = build_inner_image_planes(face_crop, &self.geometry, masking)
            .map_err(|error| sample_error(index, error.to_string()))?;
        Ok(planes.into_values())
    }

    fn audio_window(&self, index: usize) -> Result<Vec<f32>, TrainingDataError> {
        let window = audio_window_indices(index, self.frame_count)
            .map_err(|error| sample_error(index, error.to_string()))?;
        let audio = build_unet_audio_window(&self.features, &window)
            .map_err(|error| sample_error(index, error.to_string()))?;
        Ok(audio.as_slice().to_vec())
    }

    fn mouth_mask_plane(rect: &MaskRect) -> Vec<f32> {
        let mut plane = vec![0.0; INNER_SIZE * INNER_SIZE];
        let x_start = (rect.x as usize).min(INNER_SIZE);
        let y_start = (rect.y as usize).min(INNER_SIZE);
        let x_end = x_start.saturating_add(rect.width as usize).min(INNER_SIZE);
        let y_end = y_start.saturating_add(rect.height as usize).min(INNER_SIZE);
        let rows = plane
            .chunks_exact_mut(INNER_SIZE)
            .skip(y_start)
            .take(y_end - y_start);
        for row in rows {
            for value in row.iter_mut().skip(x_start).take(x_end - x_start) {
                *value = 1.0;
            }
        }
        plane
    }

    fn build_frame_sample(
        &self,
        index: usize,
        target: &LoadedFrame,
        reference: &[f32],
    ) -> Result<FrameSample, TrainingDataError> {
        let blackout = self.inner_planes(index, &target.crop, MouthMasking::Blackout)?;
        let keep = self.inner_planes(index, &target.crop, MouthMasking::Keep)?;
        let Some(elements) = reference.len().checked_add(blackout.len()) else {
            return Err(sample_error(
                index,
                "image plane count overflows".to_owned(),
            ));
        };
        let mut image = Vec::new();
        image
            .try_reserve_exact(elements)
            .map_err(|_| sample_error(index, format!("cannot allocate {elements} floats")))?;
        image.extend_from_slice(reference);
        image.extend_from_slice(&blackout);
        Ok(FrameSample {
            image,
            audio: self.audio_window(index)?,
            target: keep,
            mouth_mask: Self::mouth_mask_plane(&target.mouth),
        })
    }

    fn load_item(&self, sample: &TrainingSample) -> Result<TrainingItem, TrainingDataError> {
        match sample {
            TrainingSample::SingleFrame {
                target_index,
                reference_index,
            } => {
                let target_index = self.resolve_index(*target_index)?;
                let reference_index = self.resolve_index(*reference_index)?;
                let reference = self.load_frame(reference_index)?;
                let planes =
                    self.inner_planes(reference_index, &reference.crop, MouthMasking::Keep)?;
                let target = self.load_frame(target_index)?;
                let frame = self.build_frame_sample(target_index, &target, &planes)?;
                Ok(TrainingItem::SingleFrame(frame))
            }
            TrainingSample::TemporalPair { .. } => Err(TrainingDataError::Sample {
                index: 0,
                message: "only single-frame samples are implemented".to_owned(),
            }),
        }
    }
}

impl<R: FrameReader> TrainingDataset for ProjectTrainingDataset<R> {
    type Item = TrainingItem;

    fn frame_count(&self) -> u64 {
        self.frame_count as u64
    }

    fn load_sample(&self, sample: &TrainingSample) -> Result<Self::Item, TrainingError> {
        Ok(self.load_item(sample)?)
    }
}
