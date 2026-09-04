use burn::{module::AutodiffModule, optim::Optimizer, tensor::backend::AutodiffBackend};
use feathertalk_models::unet::TrainableTalkingHead;
use feathertalk_training::{
    PerceptualFeatureExtractor, TrainingConfig, TrainingDataLoader, TrainingDataset, TrainingError,
    TrainingMode,
};
use feathertalk_training_data::{TrainingItem, stack_single_frame_batch, stack_temporal_batch};

use crate::{LossValues, data_loader_config_for, train_single_frame_step, train_temporal_step};

const POISONED: &str = "training runner was poisoned by a failed step";

fn poisoned() -> TrainingError {
    TrainingError::InvalidInput(POISONED.to_owned())
}

fn overflow(operation: &'static str) -> TrainingError {
    TrainingError::DataLoaderOverflow { operation }
}

/// What one committed optimizer step did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepReport {
    pub epoch: u64,
    pub global_step: u64,
    pub samples_in_batch: u64,
    pub losses: LossValues,
}

/// Owns a training run: the loader, the model, the optimizer, and the progress counters.
pub struct TrainingRunner<B, M, O, D>
where
    B: AutodiffBackend,
    D: TrainingDataset<Item = TrainingItem>,
{
    model: Option<M>,
    optimizer: O,
    loader: TrainingDataLoader<D>,
    config: TrainingConfig,
    device: B::Device,
    global_step: u64,
    samples_seen: u64,
}

impl<B, M, O, D> TrainingRunner<B, M, O, D>
where
    B: AutodiffBackend,
    M: TrainableTalkingHead<B> + AutodiffModule<B> + Clone,
    O: Optimizer<M, B> + Clone,
    D: TrainingDataset<Item = TrainingItem>,
{
    pub fn new(
        dataset: D,
        model: M,
        optimizer: O,
        config: TrainingConfig,
        seed: u64,
        device: B::Device,
    ) -> Result<Self, TrainingError> {
        let loader_config = data_loader_config_for(&config, seed)?;
        let loader = TrainingDataLoader::new(dataset, loader_config)?;
        Ok(Self {
            model: Some(model),
            optimizer,
            loader,
            config,
            device,
            global_step: 0,
            samples_seen: 0,
        })
    }

    fn run_step<E>(
        &mut self,
        model: M,
        items: &[TrainingItem],
        extractor: &E,
    ) -> Result<(M, LossValues), TrainingError>
    where
        E: PerceptualFeatureExtractor<B>,
    {
        let config = &self.config;
        match config.mode {
            TrainingMode::Baseline | TrainingMode::MouthRoi => {
                let batch = stack_single_frame_batch::<B>(items, &self.device)?;
                train_single_frame_step(model, &mut self.optimizer, extractor, batch, config)
            }
            TrainingMode::MouthRoiTemporal => {
                let batch = stack_temporal_batch::<B>(items, &self.device)?;
                train_temporal_step(model, &mut self.optimizer, extractor, batch, config)
            }
        }
    }

    /// Prepares one batch, trains on it, and commits the loader position.
    pub fn step<E>(&mut self, extractor: &E) -> Result<StepReport, TrainingError>
    where
        E: PerceptualFeatureExtractor<B>,
    {
        let prepared = self.loader.prepare_next_batch()?;
        let epoch = prepared.epoch();
        let samples_in_batch =
            u64::try_from(prepared.items().len()).map_err(|_| overflow("counting batch items"))?;
        let model = self.model.take().ok_or_else(poisoned)?;
        let (model, losses) = self.run_step(model, prepared.items(), extractor)?;
        self.loader.commit_batch(prepared)?;
        self.model = Some(model);
        self.global_step = self
            .global_step
            .checked_add(1)
            .ok_or_else(|| overflow("counting training steps"))?;
        self.samples_seen = self
            .samples_seen
            .checked_add(samples_in_batch)
            .ok_or_else(|| overflow("counting seen samples"))?;
        Ok(StepReport {
            epoch,
            global_step: self.global_step,
            samples_in_batch,
            losses,
        })
    }

    pub fn epoch(&self) -> u64 {
        self.loader.state().epoch
    }

    pub fn global_step(&self) -> u64 {
        self.global_step
    }

    pub fn samples_seen(&self) -> u64 {
        self.samples_seen
    }

    pub fn is_finished(&self) -> bool {
        self.epoch() >= self.config.total_epochs
    }

    pub fn training_config(&self) -> &TrainingConfig {
        &self.config
    }

    pub fn dataset(&self) -> &D {
        self.loader.dataset()
    }

    pub fn model(&self) -> Result<&M, TrainingError> {
        self.model.as_ref().ok_or_else(poisoned)
    }
}
