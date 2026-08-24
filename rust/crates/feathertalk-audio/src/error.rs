use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum AudioError {
    #[error("waveform is empty")]
    EmptyWaveform,
    #[error("waveform contains a non-finite value at index {index}")]
    NonFiniteWaveform { index: usize },
    #[error("waveform has zero variance")]
    ConstantWaveform,
    #[error("chunk size must be greater than zero")]
    InvalidChunkSize,
    #[error("audio arithmetic overflow")]
    ArithmeticOverflow,
    #[error("audio input is too large: {actual} chunks exceeds {limit}")]
    TooManyChunks { actual: usize, limit: usize },
}
