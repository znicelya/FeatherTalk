use feathertalk_audio::FeatureMatrix;

use crate::{InferenceError, InferenceFramePlan};

const FEATURE_DIMS: usize = 1024;
const TOKENS_PER_FRAME: usize = 2;
const AUDIO_VALUES_PER_SLOT: usize = TOKENS_PER_FRAME * FEATURE_DIMS;
const UNET_AUDIO_VALUES: usize = 16 * 32 * 32;

#[derive(Debug, Clone, PartialEq)]
pub struct UnetAudioInput {
    values: Vec<f32>,
}

impl UnetAudioInput {
    pub fn shape(&self) -> [usize; 4] {
        [1, 16, 32, 32]
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }
}

pub fn build_unet_audio_input(
    features: &FeatureMatrix,
    plan: &InferenceFramePlan,
) -> Result<UnetAudioInput, InferenceError> {
    let tokens = features.tokens();
    let dims = features.dims();
    if dims != FEATURE_DIMS || tokens == 0 || !tokens.is_multiple_of(TOKENS_PER_FRAME) {
        return Err(InferenceError::InvalidFeatureShape { tokens, dims });
    }

    let frame_count = tokens / TOKENS_PER_FRAME;
    if plan.output_index >= frame_count {
        return Err(InferenceError::OutputFrameOutOfRange {
            index: plan.output_index,
            count: frame_count,
        });
    }

    let mut values = Vec::new();
    values
        .try_reserve_exact(UNET_AUDIO_VALUES)
        .map_err(|_| InferenceError::AllocationFailure {
            bytes: UNET_AUDIO_VALUES * std::mem::size_of::<f32>(),
        })?;
    values.resize(UNET_AUDIO_VALUES, 0.0);

    for (slot, frame_index) in plan.audio_window.iter().copied().enumerate() {
        let Some(frame_index) = frame_index else {
            continue;
        };
        if frame_index >= frame_count {
            return Err(InferenceError::InvalidAudioWindowIndex {
                slot,
                index: frame_index,
                frame_count,
            });
        }

        let source_start = frame_index
            .checked_mul(AUDIO_VALUES_PER_SLOT)
            .ok_or(InferenceError::ArithmeticOverflow)?;
        let destination_start = slot
            .checked_mul(AUDIO_VALUES_PER_SLOT)
            .ok_or(InferenceError::ArithmeticOverflow)?;
        let source_end = source_start
            .checked_add(AUDIO_VALUES_PER_SLOT)
            .ok_or(InferenceError::ArithmeticOverflow)?;
        let destination_end = destination_start
            .checked_add(AUDIO_VALUES_PER_SLOT)
            .ok_or(InferenceError::ArithmeticOverflow)?;
        values[destination_start..destination_end]
            .copy_from_slice(&features.values()[source_start..source_end]);
    }

    Ok(UnetAudioInput { values })
}
