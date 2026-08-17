use crate::fixture::GoldenFixture;
use ndarray::ArrayD;
use ndarray_npy::ReadNpyExt;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{self, BufReader, Cursor, Read},
    path::{Component, Path, PathBuf},
};
use thiserror::Error;
use zip::ZipArchive;

const MAX_ARCHIVE_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const MAX_ENTRY_BYTES: u64 = 512 * 1024 * 1024;
const MAX_EXPANDED_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Error)]
pub enum FixtureError {
    #[error("failed to access fixture file: {0}")]
    Io(#[from] io::Error),
    #[error("invalid ZIP archive: {0}")]
    Zip(#[from] zip::result::ZipError),
    #[error("invalid fixture manifest: {0}")]
    Manifest(#[from] serde_json::Error),
    #[error("invalid NPY fixture: {0}")]
    Npy(#[from] ndarray_npy::ReadNpyError),
    #[error("archive exceeds {limit} bytes: {actual}")]
    ArchiveTooLarge { actual: u64, limit: u64 },
    #[error("archive entry exceeds {limit} bytes: {name} ({actual})")]
    EntryTooLarge {
        name: String,
        actual: u64,
        limit: u64,
    },
    #[error("archive expands beyond {limit} bytes")]
    ExpandedSizeExceeded { limit: u64 },
    #[error("unsafe archive entry path: {0}")]
    UnsafePath(String),
    #[error("archive entry is a symbolic link: {0}")]
    SymbolicLink(String),
    #[error("duplicate archive entry: {0}")]
    DuplicateEntry(String),
    #[error("missing archive entry: {0}")]
    MissingEntry(String),
    #[error("unknown fixture id: {0}")]
    UnknownFixture(String),
    #[error("invalid SHA-256 sidecar")]
    InvalidHashSidecar,
    #[error("fixture SHA-256 mismatch: expected {expected}, actual {actual}")]
    HashMismatch { expected: String, actual: String },
    #[error("extraction destination escapes its root: {0}")]
    DestinationEscape(PathBuf),
}

#[derive(Debug)]
pub struct GoldenArchive {
    path: PathBuf,
    entries: BTreeSet<String>,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    fixtures: BTreeMap<String, FixtureManifest>,
}

#[derive(Debug, Deserialize)]
struct FixtureManifest {
    #[serde(default)]
    inputs: BTreeMap<String, String>,
    #[serde(default)]
    expected: BTreeMap<String, String>,
}

impl GoldenArchive {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, FixtureError> {
        let path = path.into();
        let length = fs::metadata(&path)?.len();
        if length > MAX_ARCHIVE_BYTES {
            return Err(FixtureError::ArchiveTooLarge {
                actual: length,
                limit: MAX_ARCHIVE_BYTES,
            });
        }

        let mut archive = open_zip(&path)?;
        let mut entries = BTreeSet::new();
        for index in 0..archive.len() {
            let entry = archive.by_index(index)?;
            let name = entry.name().to_owned();
            if !entries.insert(name.clone()) {
                return Err(FixtureError::DuplicateEntry(name));
            }
        }
        Ok(Self { path, entries })
    }

    pub fn contains(&self, entry: &str) -> bool {
        self.entries.contains(entry)
    }

    pub fn verify_sidecar_sha256(&self) -> Result<(), FixtureError> {
        let sidecar = self.path.with_extension("sha256");
        let expected = fs::read_to_string(sidecar)?;
        let expected = expected.trim();
        if expected.len() != 64
            || !expected
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(FixtureError::InvalidHashSidecar);
        }

        let mut reader = BufReader::new(File::open(&self.path)?);
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        let actual = hex::encode(hasher.finalize());
        if actual != expected {
            return Err(FixtureError::HashMismatch {
                expected: expected.to_owned(),
                actual,
            });
        }
        Ok(())
    }

    pub fn extract_to(&self, directory: &Path) -> Result<(), FixtureError> {
        self.preflight_extraction()?;
        fs::create_dir_all(directory)?;
        let root = directory.canonicalize()?;
        let mut archive = open_zip(&self.path)?;

        for index in 0..archive.len() {
            let mut entry = archive.by_index(index)?;
            let relative = validate_relative_path(entry.name())?;
            let output = root.join(relative);
            if entry.is_dir() {
                fs::create_dir_all(&output)?;
                ensure_parent_within_root(&root, &output)?;
                continue;
            }

            let parent = output
                .parent()
                .ok_or_else(|| FixtureError::UnsafePath(entry.name().to_owned()))?;
            fs::create_dir_all(parent)?;
            ensure_parent_within_root(&root, parent)?;
            let mut destination = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&output)?;
            io::copy(&mut entry, &mut destination)?;
        }
        Ok(())
    }

    pub fn load_fixture(&self, id: &str) -> Result<GoldenFixture, FixtureError> {
        let manifest_bytes = self.read_entry("manifest.json", MAX_MANIFEST_BYTES)?;
        let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
        let fixture = manifest
            .fixtures
            .get(id)
            .ok_or_else(|| FixtureError::UnknownFixture(id.to_owned()))?;

        Ok(GoldenFixture {
            id: id.to_owned(),
            inputs: self.load_arrays(&fixture.inputs)?,
            expected: self.load_arrays(&fixture.expected)?,
        })
    }

    fn load_arrays(
        &self,
        entries: &BTreeMap<String, String>,
    ) -> Result<BTreeMap<String, ArrayD<f32>>, FixtureError> {
        entries
            .iter()
            .map(|(name, path)| {
                validate_relative_path(path)?;
                let bytes = self.read_entry(path, MAX_ENTRY_BYTES)?;
                let array = ArrayD::<f32>::read_npy(Cursor::new(bytes))?;
                Ok((name.clone(), array))
            })
            .collect()
    }

    fn read_entry(&self, name: &str, limit: u64) -> Result<Vec<u8>, FixtureError> {
        let mut archive = open_zip(&self.path)?;
        let mut entry = archive.by_name(name).map_err(|error| match error {
            zip::result::ZipError::FileNotFound => FixtureError::MissingEntry(name.to_owned()),
            other => FixtureError::Zip(other),
        })?;
        if entry.size() > limit {
            return Err(FixtureError::EntryTooLarge {
                name: name.to_owned(),
                actual: entry.size(),
                limit,
            });
        }
        let mut bytes = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut bytes)?;
        Ok(bytes)
    }

    fn preflight_extraction(&self) -> Result<(), FixtureError> {
        let mut archive = open_zip(&self.path)?;
        let mut expanded = 0_u64;
        for index in 0..archive.len() {
            let entry = archive.by_index(index)?;
            validate_relative_path(entry.name())?;
            if is_symbolic_link(entry.unix_mode()) {
                return Err(FixtureError::SymbolicLink(entry.name().to_owned()));
            }
            if entry.size() > MAX_ENTRY_BYTES {
                return Err(FixtureError::EntryTooLarge {
                    name: entry.name().to_owned(),
                    actual: entry.size(),
                    limit: MAX_ENTRY_BYTES,
                });
            }
            expanded =
                expanded
                    .checked_add(entry.size())
                    .ok_or(FixtureError::ExpandedSizeExceeded {
                        limit: MAX_EXPANDED_BYTES,
                    })?;
            if expanded > MAX_EXPANDED_BYTES {
                return Err(FixtureError::ExpandedSizeExceeded {
                    limit: MAX_EXPANDED_BYTES,
                });
            }
        }
        Ok(())
    }
}

fn open_zip(path: &Path) -> Result<ZipArchive<BufReader<File>>, FixtureError> {
    Ok(ZipArchive::new(BufReader::new(File::open(path)?))?)
}

fn validate_relative_path(path: &str) -> Result<PathBuf, FixtureError> {
    let path = Path::new(path);
    if path.as_os_str().is_empty()
        || path.components().any(|component| {
            matches!(
                component,
                Component::Prefix(_) | Component::RootDir | Component::ParentDir
            )
        })
    {
        return Err(FixtureError::UnsafePath(path.display().to_string()));
    }
    Ok(path.to_path_buf())
}

fn ensure_parent_within_root(root: &Path, path: &Path) -> Result<(), FixtureError> {
    let canonical = path.canonicalize()?;
    if !canonical.starts_with(root) {
        return Err(FixtureError::DestinationEscape(canonical));
    }
    Ok(())
}

fn is_symbolic_link(mode: Option<u32>) -> bool {
    mode.is_some_and(|mode| mode & 0o170000 == 0o120000)
}
