use burn::tensor::{ElementConversion, Tensor, backend::Backend};
use feathertalk_training::{LossBreakdown, TrainingError};

/// Scalar view of a `LossBreakdown`, detached from the autodiff graph.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossValues {
    pub total: f64,
    pub full: f64,
    pub perceptual: f64,
    pub mouth: Option<f64>,
    pub temporal: Option<f64>,
    pub temporal_mouth: Option<f64>,
}

impl LossValues {
    pub fn from_breakdown<B: Backend>(breakdown: &LossBreakdown<B>) -> Self {
        Self {
            total: scalar(&breakdown.total),
            full: scalar(&breakdown.full),
            perceptual: scalar(&breakdown.perceptual),
            mouth: breakdown.mouth.as_ref().map(scalar),
            temporal: breakdown.temporal.as_ref().map(scalar),
            temporal_mouth: breakdown.temporal_mouth.as_ref().map(scalar),
        }
    }

    pub fn require_finite(&self) -> Result<(), TrainingError> {
        check("total", Some(self.total))?;
        check("full", Some(self.full))?;
        check("perceptual", Some(self.perceptual))?;
        check("mouth", self.mouth)?;
        check("temporal", self.temporal)?;
        check("temporal_mouth", self.temporal_mouth)
    }
}

fn scalar<B: Backend>(value: &Tensor<B, 1>) -> f64 {
    value.clone().into_scalar().elem::<f64>()
}

fn check(field: &str, value: Option<f64>) -> Result<(), TrainingError> {
    match value {
        Some(value) if !value.is_finite() => {
            let message = format!("training loss {field} is not finite: {value}");
            Err(TrainingError::InvalidInput(message))
        }
        _ => Ok(()),
    }
}
