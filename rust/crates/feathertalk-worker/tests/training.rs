use std::path::{Path, PathBuf};

use feathertalk_domain::{TrainParams, TrainingMode as DomainTrainingMode, UnetVariant};
use feathertalk_export::ModelConfiguration;
use feathertalk_models::unet::{MobileOneUnetConfig, OriginalUnetConfig};
use feathertalk_training::TrainingMode;
use feathertalk_worker::{
    DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, MAX_EPOCHS, TRAIN_BACKEND_NAME, TRAINING_SEED,
    TrainingPaths, WORKER_STATE, checkpoint_descriptor, sample_count, training_config,
    training_mode,
};

fn params(mode: DomainTrainingMode, epochs: u32) -> TrainParams {
    TrainParams {
        project_dir: PathBuf::from("C:/tmp/project"),
        mode,
        variant: UnetVariant::OriginalUnet,
        epochs,
        resume: false,
    }
}

#[test]
fn the_three_request_modes_map_onto_the_training_crate() {
    assert_eq!(
        training_mode(DomainTrainingMode::Baseline),
        TrainingMode::Baseline
    );
    assert_eq!(
        training_mode(DomainTrainingMode::MouthRoi),
        TrainingMode::MouthRoi
    );
    // The request has three modes, the training crate three too, but the third
    // pair does not share a name.
    assert_eq!(
        training_mode(DomainTrainingMode::Temporal),
        TrainingMode::MouthRoiTemporal
    );
}

#[test]
fn only_the_temporal_mode_takes_a_stride_and_loses_a_sample() {
    for mode in [DomainTrainingMode::Baseline, DomainTrainingMode::MouthRoi] {
        let config = training_config(&params(mode, 2));
        assert_eq!(config.temporal_stride, 0);
        assert_eq!(sample_count(mode, 188), 188);
    }

    let config = training_config(&params(DomainTrainingMode::Temporal, 2));
    assert_eq!(config.temporal_stride, 1);
    assert_eq!(sample_count(DomainTrainingMode::Temporal, 188), 187);
    // A one-frame project starts no temporal sample at all.
    assert_eq!(sample_count(DomainTrainingMode::Temporal, 1), 0);
    assert_eq!(sample_count(DomainTrainingMode::Temporal, 0), 0);
}

#[test]
fn the_config_takes_five_fields_from_worker_constants() {
    let config = training_config(&params(DomainTrainingMode::MouthRoi, 7));

    assert_eq!(config.total_epochs, 7);
    assert_eq!(config.batch_size, DEFAULT_BATCH_SIZE);
    assert_eq!(config.learning_rate, DEFAULT_LEARNING_RATE);
    assert_eq!(config.mouth_weight, 4.0);
    assert_eq!(config.temporal_weight, 0.5);
    assert_eq!(config.temporal_mouth_weight, 4.0);
    assert_eq!(config.perceptual_weight, 0.01);
    config
        .validate()
        .expect("the assembled config is valid on its own");

    assert_eq!(DEFAULT_BATCH_SIZE, 1);
    assert_eq!(TRAINING_SEED, 1);
    assert_eq!(MAX_EPOCHS, 10_000);
    assert_eq!(WORKER_STATE, "training");
    assert_eq!(TRAIN_BACKEND_NAME, "ndarray-cpu");
}

#[test]
fn the_descriptor_digests_the_model_configuration() {
    let configuration = ModelConfiguration::original_unet(&OriginalUnetConfig::production());
    let descriptor = checkpoint_descriptor(&configuration).expect("the digest is computable");

    descriptor.validate().expect("64 lowercase hex characters");
    assert_eq!(descriptor.model_kind, configuration.model_type());
    assert_eq!(
        descriptor.architecture_version,
        configuration.architecture_version()
    );
    assert_eq!(descriptor.model_config_sha256.len(), 64);
    assert!(
        descriptor
            .model_config_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{}",
        descriptor.model_config_sha256
    );
    // Same configuration, same digest: this value is what a later resume has to
    // match, so it must not depend on anything but the configuration.
    let again = checkpoint_descriptor(&configuration).unwrap();
    assert_eq!(again, descriptor);

    // The training graph is the multi-branch one; a reparameterized descriptor
    // would claim a structure training never builds.
    let mobile = ModelConfiguration::mobileone_unet(&MobileOneUnetConfig::production(), false);
    let mobile_descriptor = checkpoint_descriptor(&mobile).unwrap();
    assert_eq!(mobile_descriptor.model_kind, "mobileone_unet");
    assert_ne!(
        mobile_descriptor.model_config_sha256,
        descriptor.model_config_sha256
    );

    let reparameterized =
        ModelConfiguration::mobileone_unet(&MobileOneUnetConfig::production(), true);
    assert_ne!(
        checkpoint_descriptor(&reparameterized)
            .unwrap()
            .model_config_sha256,
        mobile_descriptor.model_config_sha256
    );

    // A micro configuration digests differently again, which is what makes the
    // offline tests' descriptors distinct from a production run's.
    assert_ne!(
        checkpoint_descriptor(&ModelConfiguration::original_unet(
            &OriginalUnetConfig::parity_micro()
        ))
        .unwrap()
        .model_config_sha256,
        descriptor.model_config_sha256
    );
}

#[test]
fn the_artifact_paths_are_step_numbered() {
    let paths = TrainingPaths::new(Path::new("C:/tmp/project"));

    assert_eq!(
        paths.checkpoints(),
        Path::new("C:/tmp/project").join("models").join("unet")
    );
    assert!(
        paths
            .checkpoint(188)
            .ends_with("models/unet/checkpoint-00000188")
    );
    assert!(
        paths
            .metrics(188)
            .ends_with("outputs/metrics/step-00000188.json")
    );
    assert!(
        paths
            .preview(188)
            .ends_with("outputs/preview/step-00000188")
    );
    // Eight digits, so a nine-digit run does not silently truncate.
    assert!(
        paths
            .checkpoint(123_456_789)
            .ends_with("checkpoint-123456789")
    );
}

#[test]
fn only_step_numbered_checkpoint_names_are_recognised() {
    assert_eq!(
        TrainingPaths::checkpoint_step("checkpoint-00000188"),
        Some(188)
    );
    assert_eq!(
        TrainingPaths::checkpoint_step("checkpoint-00000000"),
        Some(0)
    );
    // `{:08}` pads but never truncates, so a run past eight digits must still
    // find its own checkpoints.
    assert_eq!(
        TrainingPaths::checkpoint_step("checkpoint-123456789"),
        Some(123_456_789)
    );
    for name in [
        "checkpoint-188",
        "checkpoint-0000018x",
        "checkpoint-",
        "checkpoint",
        ".publish-1234-0",
        ".retired-1234-0",
        "last",
    ] {
        assert_eq!(TrainingPaths::checkpoint_step(name), None, "{name}");
    }
}
