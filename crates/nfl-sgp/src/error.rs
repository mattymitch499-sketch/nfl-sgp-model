#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("data error: {0}")]
    Data(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
