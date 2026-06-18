pub mod error;
pub mod indexes;
pub mod math;
pub mod types;

pub use error::FerricError;
pub use indexes::FlatIndex;
pub use math::{dot, l2_squared, normalize};
pub use types::{Hit, IndexKind, IndexSpec, Metric, StorageType};

pub type Result<T> = std::result::Result<T, FerricError>;
