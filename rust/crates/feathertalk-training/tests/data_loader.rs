use feathertalk_training::{
    DATA_LOADER_STATE_SCHEMA_VERSION, DataLoaderConfig, DataLoaderState, RandomAlgorithm,
    SamplingConfig, SamplingKind, TrainingError,
};

#[test]
fn single_and_temporal_configs_expose_the_fixed_contract() {
    assert_eq!(DATA_LOADER_STATE_SCHEMA_VERSION, 1);
    assert_eq!(
        DataLoaderConfig::single_frame(4, 7),
        DataLoaderConfig {
            batch_size: 4,
            seed: 7,
            sampling: SamplingConfig {
                kind: SamplingKind::SingleFrame,
                temporal_stride: 0,
            },
        }
    );
    assert_eq!(
        DataLoaderConfig::temporal_pair(3, 42, 2),
        DataLoaderConfig {
            batch_size: 3,
            seed: 42,
            sampling: SamplingConfig {
                kind: SamplingKind::TemporalPair,
                temporal_stride: 2,
            },
        }
    );
}

#[test]
fn invalid_config_and_state_are_rejected_before_loading() {
    let invalid = [
        DataLoaderConfig::single_frame(0, 7),
        DataLoaderConfig {
            batch_size: 2,
            seed: 7,
            sampling: SamplingConfig {
                kind: SamplingKind::SingleFrame,
                temporal_stride: 1,
            },
        },
        DataLoaderConfig::temporal_pair(2, 7, 0),
    ];
    for config in invalid {
        assert!(matches!(
            config.validate(5),
            Err(TrainingError::InvalidDataLoaderConfig(_))
        ));
    }

    assert!(matches!(
        DataLoaderConfig::single_frame(2, 7).validate(0),
        Err(TrainingError::InvalidDataLoaderConfig(_))
    ));
    assert!(matches!(
        DataLoaderConfig::temporal_pair(2, 7, 5).validate(5),
        Err(TrainingError::InvalidDataLoaderConfig(_))
    ));

    let unsupported_schema = DataLoaderState {
        schema_version: 99,
        random_algorithm: RandomAlgorithm::Splitmix64FisherYatesV1,
        config: DataLoaderConfig::single_frame(2, 7),
        frame_count: 5,
        epoch: 0,
        next_position: 0,
    };
    assert!(matches!(
        unsupported_schema.validate(5),
        Err(TrainingError::InvalidDataLoaderState(_))
    ));

    let cursor_at_end = DataLoaderState {
        schema_version: DATA_LOADER_STATE_SCHEMA_VERSION,
        random_algorithm: RandomAlgorithm::Splitmix64FisherYatesV1,
        config: DataLoaderConfig::single_frame(2, 7),
        frame_count: 5,
        epoch: 0,
        next_position: 5,
    };
    assert!(matches!(
        cursor_at_end.validate(5),
        Err(TrainingError::InvalidDataLoaderState(_))
    ));

    let invalid_temporal_state = DataLoaderState {
        schema_version: DATA_LOADER_STATE_SCHEMA_VERSION,
        random_algorithm: RandomAlgorithm::Splitmix64FisherYatesV1,
        config: DataLoaderConfig::temporal_pair(2, 7, 5),
        frame_count: 5,
        epoch: 0,
        next_position: 0,
    };
    assert!(matches!(
        invalid_temporal_state.validate(5),
        Err(TrainingError::InvalidDataLoaderConfig(_))
    ));
}
