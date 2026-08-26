use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Take, Write},
    path::{Path, PathBuf},
};

use burn::{
    module::AutodiffModule,
    optim::Optimizer,
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::backend::AutodiffBackend,
};
use sha2::{Digest, Sha256};

use crate::{
    CheckpointFileManifest, CHECKPOINT_MANIFEST_FILE_NAME, CHECKPOINT_MODEL_FILE_NAME,
    CHECKPOINT_OPTIMIZER_FILE_NAME, CHECKPOINT_STATE_FILE_NAME, TrainingError,
};

pub(crate) const MANIFEST_MAX_BYTES: u64 = 64 * 1024;
pub(crate) const STATE_MAX_BYTES: u64 = 256 * 1024;

pub(crate) type FullRecorder = BinFileRecorder<FullPrecisionSettings>;

pub(crate) fn write_model_record<B, M>(
    model: &M,
    stem: &Path,
) -> Result<PathBuf, TrainingError>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Clone,
{
    let recorder = FullRecorder::default();
    <FullRecorder as Recorder<B>>::record(
        &recorder,
        model.clone().into_record(),
        stem.to_path_buf(),
    )
    .map_err(|error| TrainingError::Store(format!("write model record: {error}")))?;
    Ok(with_recorder_extension(stem))
}

pub(crate) fn write_optimizer_record<B, M, O>(
    optimizer: &O,
    stem: &Path,
) -> Result<PathBuf, TrainingError>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: Optimizer<M, B> + Clone,
{
    let recorder = FullRecorder::default();
    <FullRecorder as Recorder<B>>::record(
        &recorder,
        optimizer.clone().to_record(),
        stem.to_path_buf(),
    )
    .map_err(|error| TrainingError::Store(format!("write optimizer record: {error}")))?;
    Ok(with_recorder_extension(stem))
}

pub(crate) fn load_model_record<B, M>(
    model: M,
    path: &Path,
    device: &B::Device,
) -> Result<M, TrainingError>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Clone,
{
    let recorder = FullRecorder::default();
    let record = <FullRecorder as Recorder<B>>::load::<M::Record>(
        &recorder,
        path.to_path_buf(),
        device,
    )
    .map_err(|error| TrainingError::Store(format!("load model record: {error}")))?;
    Ok(model.load_record(record))
}

pub(crate) fn load_optimizer_record<B, M, O>(
    optimizer: O,
    path: &Path,
    device: &B::Device,
) -> Result<O, TrainingError>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: Optimizer<M, B> + Clone,
{
    let recorder = FullRecorder::default();
    let record = <FullRecorder as Recorder<B>>::load::<O::Record>(
        &recorder,
        path.to_path_buf(),
        device,
    )
    .map_err(|error| TrainingError::Store(format!("load optimizer record: {error}")))?;
    Ok(optimizer.load_record(record))
}

pub(crate) fn write_synced_bytes(path: &Path, bytes: &[u8]) -> Result<(), TrainingError> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

pub(crate) fn sync_file(path: &Path) -> Result<(), TrainingError> {
    let file = OpenOptions::new().read(true).write(true).open(path)?;
    file.sync_all()?;
    Ok(())
}

pub(crate) fn sync_directory(path: &Path) -> Result<(), TrainingError> {
    #[cfg(unix)]
    {
        File::open(path)?.sync_all()?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

pub(crate) fn sha256_file(path: &Path) -> Result<(u64, String), TrainingError> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut bytes_read = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        bytes_read = bytes_read
            .checked_add(u64::try_from(count).map_err(|_| {
                TrainingError::InvalidCheckpoint("file byte count overflow".to_owned())
            })?)
            .ok_or_else(|| TrainingError::InvalidCheckpoint("file byte count overflow".to_owned()))?;
        digest.update(&buffer[..count]);
    }
    Ok((bytes_read, hex::encode(digest.finalize())))
}

pub(crate) fn read_bounded(path: &Path, max_bytes: u64) -> Result<Vec<u8>, TrainingError> {
    let file = File::open(path)?;
    let mut reader: Take<File> = file.take(max_bytes.saturating_add(1));
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > max_bytes {
        return Err(TrainingError::CheckpointDirectory(format!(
            "file {} exceeds maximum size of {max_bytes} bytes",
            path.display()
        )));
    }
    Ok(bytes)
}

pub(crate) fn validate_declared_file(
    path: &Path,
    declared: &CheckpointFileManifest,
) -> Result<(), TrainingError> {
    validate_regular_file(path)?;
    let (bytes, sha256) = sha256_file(path)?;
    if bytes != declared.bytes {
        return Err(TrainingError::InvalidCheckpoint(format!(
            "{} declares {} bytes but contains {bytes}",
            declared.file_name, declared.bytes
        )));
    }
    if sha256 != declared.sha256 {
        return Err(TrainingError::HashMismatch {
            file: declared.file_name.clone(),
            expected: declared.sha256.clone(),
            actual: sha256,
        });
    }
    Ok(())
}

pub(crate) fn validate_regular_file(path: &Path) -> Result<(), TrainingError> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() {
        return Err(TrainingError::CheckpointDirectory(format!(
            "checkpoint file must not be a symbolic link: {}",
            path.display()
        )));
    }
    if !metadata.is_file() {
        return Err(TrainingError::CheckpointDirectory(format!(
            "checkpoint entry is not a regular file: {}",
            path.display()
        )));
    }
    Ok(())
}

pub(crate) fn validate_checkpoint_directory(path: &Path) -> Result<(), TrainingError> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() {
        return Err(TrainingError::CheckpointDirectory(format!(
            "checkpoint directory must not be a symbolic link: {}",
            path.display()
        )));
    }
    if !metadata.is_dir() {
        return Err(TrainingError::CheckpointDirectory(format!(
            "checkpoint path is not a directory: {}",
            path.display()
        )));
    }

    let mut entries = fs::read_dir(path)?
        .map(|entry| {
            entry.and_then(|entry| {
                entry.file_name().into_string().map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "checkpoint entry name is not valid UTF-8",
                    )
                })
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    entries.sort();
    let mut expected = vec![
        CHECKPOINT_MANIFEST_FILE_NAME.to_owned(),
        CHECKPOINT_MODEL_FILE_NAME.to_owned(),
        CHECKPOINT_OPTIMIZER_FILE_NAME.to_owned(),
        CHECKPOINT_STATE_FILE_NAME.to_owned(),
    ];
    expected.sort();
    if entries != expected {
        return Err(TrainingError::CheckpointDirectory(format!(
            "checkpoint directory entries must be exactly {expected:?}, got {entries:?}"
        )));
    }
    for entry in expected {
        validate_regular_file(&path.join(entry))?;
    }
    Ok(())
}

pub(crate) fn with_recorder_extension(stem: &Path) -> PathBuf {
    let mut path = stem.to_path_buf();
    path.set_extension("bin");
    path
}
