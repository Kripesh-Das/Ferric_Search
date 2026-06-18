#[derive(Debug, thiserror::Error)]
pub enum FerricError {
    #[error("index not trained")]
    NotTrained,
    #[error("invalid dimension: expected {expected}, got {got}")]
    InvalidDimension { expected: usize, got: usize },
    #[error("ids and vectors length mismatch")]
    MismatchedLength,
    #[error("dim {dim} not divisible by M {m}")]
    DimensionNotDivisible { dim: usize, m: usize },
    #[error("ksub {0} must be <= 256")]
    KsubTooLarge(usize),
    #[error("storage error: {0}")]
    Storage(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error)  ,
    #[error("serialization error: {0}")]
    Serialization(#[from] bincode::Error),
}       
