mod error;
mod model;
mod validate;

pub use error::MediaError;
pub use model::{MediaInput, NormalizationSpec, NormalizedMediaLayout, ValidatedInput};
pub use validate::{validate_input, validate_normalization};
