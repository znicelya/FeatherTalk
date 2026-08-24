use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};

use sha2::{Digest, Sha256};

use crate::{AudioError, FeatureMatrix};

const MAGIC: &[u8; 8] = b"FTF32\0\0\0";
const VERSION: u32 = 1;
const HEADER_BYTES: usize = 36;
pub const MAX_FEATURE_FILE_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureArtifact {
    path: PathBuf,
    tokens: usize,
    dims: usize,
    bytes: u64,
    sha256: String,
}

impl FeatureArtifact {
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn tokens(&self) -> usize {
        self.tokens
    }
    pub fn dims(&self) -> usize {
        self.dims
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
}

pub fn write_feature_file(
    path: &Path,
    matrix: &FeatureMatrix,
) -> Result<FeatureArtifact, AudioError> {
    if let Ok(metadata) = fs::symlink_metadata(path)
        && (metadata.file_type().is_symlink() || !metadata.is_file())
    {
        return Err(AudioError::FeatureNotRegular {
            path: path.to_owned(),
        });
    }
    let payload_bytes = matrix
        .values()
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(AudioError::FeatureSizeOverflow)?;
    let total_bytes = HEADER_BYTES
        .checked_add(payload_bytes)
        .ok_or(AudioError::FeatureSizeOverflow)?;
    if total_bytes as u64 > MAX_FEATURE_FILE_BYTES {
        return Err(AudioError::FeatureTooLarge {
            limit: MAX_FEATURE_FILE_BYTES,
            actual: total_bytes as u64,
        });
    }
    let parent = path.parent().ok_or_else(|| AudioError::FeatureIo {
        operation: "parent",
        path: path.to_owned(),
        source: std::io::Error::new(std::io::ErrorKind::InvalidInput, "missing parent"),
    })?;
    fs::create_dir_all(parent).map_err(|source| io("create_parent", parent, source))?;
    let mut bytes = Vec::with_capacity(total_bytes);
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&VERSION.to_le_bytes());
    bytes.extend_from_slice(&(matrix.tokens() as u64).to_le_bytes());
    bytes.extend_from_slice(&(matrix.dims() as u64).to_le_bytes());
    bytes.extend_from_slice(&(payload_bytes as u64).to_le_bytes());
    for value in matrix.values() {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .map_err(|source| io("create", path, source))?;
    file.write_all(&bytes)
        .and_then(|_| file.sync_all())
        .map_err(|source| io("write", path, source))?;
    let (actual_bytes, sha256) = hash_file(path)?;
    Ok(FeatureArtifact {
        path: path.to_owned(),
        tokens: matrix.tokens(),
        dims: matrix.dims(),
        bytes: actual_bytes,
        sha256,
    })
}

pub fn read_feature_file(path: &Path) -> Result<FeatureMatrix, AudioError> {
    let metadata = fs::symlink_metadata(path).map_err(|source| io("stat", path, source))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(AudioError::FeatureNotRegular {
            path: path.to_owned(),
        });
    }
    if metadata.len() > MAX_FEATURE_FILE_BYTES {
        return Err(AudioError::FeatureTooLarge {
            limit: MAX_FEATURE_FILE_BYTES,
            actual: metadata.len(),
        });
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    File::open(path)
        .map_err(|source| io("open", path, source))?
        .read_to_end(&mut bytes)
        .map_err(|source| io("read", path, source))?;
    if bytes.len() < HEADER_BYTES {
        return Err(AudioError::FeatureHeaderTruncated {
            actual: bytes.len(),
        });
    }
    if &bytes[..8] != MAGIC {
        return Err(AudioError::InvalidFeatureMagic);
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if version != VERSION {
        return Err(AudioError::UnsupportedFeatureVersion { version });
    }
    let tokens = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
    let dims = u64::from_le_bytes(bytes[20..28].try_into().unwrap());
    let payload_bytes = u64::from_le_bytes(bytes[28..36].try_into().unwrap());
    let tokens = usize::try_from(tokens).map_err(|_| AudioError::FeatureSizeOverflow)?;
    let dims = usize::try_from(dims).map_err(|_| AudioError::FeatureSizeOverflow)?;
    let expected_payload = tokens
        .checked_mul(dims)
        .and_then(|n| n.checked_mul(4))
        .ok_or(AudioError::FeatureSizeOverflow)? as u64;
    if payload_bytes != expected_payload {
        return Err(AudioError::InvalidFeaturePayloadSize);
    }
    let available = (bytes.len() - HEADER_BYTES) as u64;
    if available < payload_bytes {
        return Err(AudioError::FeaturePayloadTruncated {
            expected: payload_bytes,
            actual: available,
        });
    }
    if available > payload_bytes {
        return Err(AudioError::FeatureTrailingBytes {
            actual: (available - payload_bytes) as usize,
        });
    }
    let mut values = Vec::with_capacity(
        tokens
            .checked_mul(dims)
            .ok_or(AudioError::FeatureSizeOverflow)?,
    );
    for (index, chunk) in bytes[HEADER_BYTES..].chunks_exact(4).enumerate() {
        let value = f32::from_le_bytes(chunk.try_into().unwrap());
        if !value.is_finite() {
            return Err(AudioError::NonFiniteFeature { index });
        }
        values.push(value);
    }
    FeatureMatrix::new(tokens, dims, values)
}

fn hash_file(path: &Path) -> Result<(u64, String), AudioError> {
    let mut file = File::open(path).map_err(|source| io("hash_open", path, source))?;
    let mut digest = Sha256::new();
    let mut bytes = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|source| io("hash_read", path, source))?;
        if read == 0 {
            break;
        }
        bytes = bytes
            .checked_add(read as u64)
            .ok_or(AudioError::FeatureSizeOverflow)?;
        digest.update(&buffer[..read]);
    }
    Ok((bytes, hex::encode(digest.finalize())))
}

fn io(operation: &'static str, path: &Path, source: std::io::Error) -> AudioError {
    AudioError::FeatureIo {
        operation,
        path: path.to_owned(),
        source,
    }
}
