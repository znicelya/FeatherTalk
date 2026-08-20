use crate::PreprocessError;

pub fn audio_window_indices(
    frame_index: usize,
    frame_count: usize,
) -> Result<[Option<usize>; 8], PreprocessError> {
    if frame_count == 0 || frame_index >= frame_count {
        return Err(PreprocessError::FrameIndexOutOfRange {
            frame_index,
            frame_count,
        });
    }
    let center = frame_index as i64;
    let count = frame_count as i64;
    Ok(std::array::from_fn(|slot| {
        let index = center + slot as i64 - 4;
        (index >= 0 && index < count).then_some(index as usize)
    }))
}
