use feathertalk_training::{DataLoaderConfig, TrainingConfig, TrainingError, TrainingMode};

/// Derives the data-loader config a training mode needs.
pub fn data_loader_config_for(
    config: &TrainingConfig,
    seed: u64,
) -> Result<DataLoaderConfig, TrainingError> {
    config.validate()?;
    Ok(match config.mode {
        TrainingMode::Baseline | TrainingMode::MouthRoi => {
            DataLoaderConfig::single_frame(config.batch_size, seed)
        }
        TrainingMode::MouthRoiTemporal => {
            DataLoaderConfig::temporal_pair(config.batch_size, seed, config.temporal_stride)
        }
    })
}
