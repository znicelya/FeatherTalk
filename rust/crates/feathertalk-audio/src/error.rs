use thiserror::Error;

#[derive(Debug, Error)]
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
    #[error("feature output dimension must be greater than zero")]
    InvalidFeatureDimension,
    #[error("feature output length {actual} is not divisible by dimension {dimension}")]
    FeatureLengthMismatch { actual: usize, dimension: usize },
    #[error("feature contains a non-finite value at index {index}")]
    NonFiniteFeature { index: usize },
    #[error("feature matrix size overflow")]
    FeatureSizeOverflow,
    #[error("feature file is not a regular non-symlink file: {path}")]
    FeatureNotRegular { path: std::path::PathBuf },
    #[error("feature file exceeds {limit} bytes: {actual}")]
    FeatureTooLarge { limit: u64, actual: u64 },
    #[error("feature file has an invalid magic header")]
    InvalidFeatureMagic,
    #[error("unsupported feature file version {version}")]
    UnsupportedFeatureVersion { version: u32 },
    #[error("feature file header is truncated: {actual} bytes")]
    FeatureHeaderTruncated { actual: usize },
    #[error("feature payload is truncated: expected {expected} bytes, got {actual}")]
    FeaturePayloadTruncated { expected: u64, actual: u64 },
    #[error("feature file has {actual} trailing bytes")]
    FeatureTrailingBytes { actual: usize },
    #[error("feature file has an invalid payload size")]
    InvalidFeaturePayloadSize,
    #[error("feature I/O error during {operation} at {path}: {source}")]
    FeatureIo {
        operation: &'static str,
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
}
