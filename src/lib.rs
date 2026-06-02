pub mod error;
pub mod math;
pub mod types;

pub use error::FerricError;
pub use math::{dot, l2_squared, normalize};
pub use types::{Hit, IndexKind, IndexSpec, Metric, StorageType};

pub type Result<T> = std::result::Result<T, FerricError>;
