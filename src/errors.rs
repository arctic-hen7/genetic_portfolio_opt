use std::num::ParseFloatError;
use thiserror::Error;

/// An error encountered while trying to parse a CSV file of assets.
#[derive(Error, Debug)]
pub enum CsvParseError {
    #[error("io error occurred while reading csv file")]
    Io(#[from] std::io::Error),
    #[error("failed to parse return at row #{row}, column #{col} as Float")]
    NotFloat {
        row: usize,
        col: usize,
        #[source]
        source: ParseFloatError,
    },
    #[error("found NaN at row #{row}, column #{col}")]
    NaN { row: usize, col: usize },
}

/// Errors that can occur while building a portfolio optimisation engine.
#[derive(Error, Debug)]
pub enum EngineBuildError {
    #[error("no data provided for engine")]
    NoData,
    #[error("no risk-free rate provided for engine")]
    NoRiskFreeRate,
    #[error("no max jitter provided for engine")]
    NoMaxJitter,
    #[error("no strategy provided for engine")]
    NoStrategy,
}

/// Errors that can occur in parsing CLI arguments.
#[derive(Error, Debug)]
pub enum CliError {
    #[error("invalid strategy: {0}")]
    InvalidStrategy(&'static str),
    #[error("invalid selector: {0}")]
    InvalidSelector(&'static str),
    #[error(transparent)]
    CsvParse(#[from] CsvParseError),
}
