use crate::{
    params::Parameters,
    parse_csv::{parse_csv, CsvParseError},
    portfolio::Portfolio,
};
use genetic::Individual;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{io::BufRead, ops::Range, path::Path};

/// Historical data for the assets being optimised. This is likely to be very large in memory.
#[derive(Debug)]
pub struct Data {
    /// The assets we're working with, which are guaranteed to all have the same length (and
    /// semantically should align in time as well).
    ///
    /// Rather than storing these as columns, we store them as rows. The length of the inner vector
    /// is the number of assets, while that of the outer vector is the number of returns. This
    /// allows more efficient iterator zipping for combining returns into weighted portfolios for
    /// fitness computation. It's also easier to read directly from a CSV file.
    assets: Vec<Vec<f64>>,
    /// The names of each asset.
    asset_names: Vec<String>,
    /// The dates for which we have recorded asset data. These are likely to not be continuous, as
    /// holidays and periods where one or more assets aren't reporting will be dropped.
    dates: Vec<String>,
}
impl Data {
    /// Creates a new [`Data`] instance from the given CSV file. See [`Self::from_reader`] for the
    /// details of the expected format.
    pub fn from_file(filename: impl AsRef<Path>) -> Result<Self, CsvParseError> {
        let file = std::fs::File::open(filename).map_err(|e| CsvParseError::Io(e))?;
        Self::from_reader(std::io::BufReader::new(file))
    }

    /// Creates a new [`Data`] instance from the given [`BufRead`] over a CSV file.
    ///
    /// # Asset Data
    ///
    /// The CSV file must be comma-separated and include a header with column names, which will be
    /// read as asset names, with the exception of the first column, which will be read as a date
    /// column. All other rows should contain daily returns, which will be interpreted as
    /// percentage returns (i.e. 0.1 implies a 10% return on the given day, *not* 0.1%).
    ///
    /// Any rows where all assets have 0.0 returns will be considered weekends/trading holidays and
    /// dropped. Only the period from which all assets start reporting non-0.0 values will be
    /// considered (i.e. any rows before that will be dropped, even if they have non-0.0 values for
    /// some assets), as we can only optimise over periods for which we have data for all assets.
    /// Blocks of 0.0s for only some assets *after* non-0.0 values will be accepted.
    ///
    /// While the system is designed to work with daily returns, they can technically be returns
    /// over any period, provided there are sufficiently many observations. The resulting optimal
    /// portfolio weighting can then be analysed in whatever the relevant context may be.
    ///
    /// ## Security
    ///
    /// This function will read each line of `assets_csv` blindly, which can cause memory
    /// exhaustion if an attacker sends a continuous stream of bytes with no newline, or with no
    /// EOF. It is assumed that `assets_csv` is "safe" against these attacks, so great caution
    /// should be taken when allowing unauthorised users to stream it over a network.
    pub fn from_reader(reader: impl BufRead) -> Result<Self, CsvParseError> {
        let (asset_names, dates, assets) = parse_csv(reader)?;
        Ok(Self {
            assets,
            asset_names,
            dates,
        })
    }

    /// Returns a handle to this data for the given range of observations, allowing inference and
    /// testing to be done on a subset of data.
    ///
    /// # Panics
    ///
    /// Panics if the range is empty (e.g. end less than start) or if the end of the range exceeds
    /// the number of observations.
    pub fn handle<'d>(&'d self, range: Range<usize>) -> DataHandle<'d> {
        assert!(!range.is_empty(), "range must be non-empty");
        assert!(
            range.end <= self.assets.len(),
            "range end exceeds number of observations"
        );

        DataHandle {
            data: self,
            start_idx: range.start,
            end_idx: range.end,
        }
    }

    /// Returns a handle to this data for all observations.
    pub fn all<'d>(&'d self) -> DataHandle<'d> {
        DataHandle {
            data: self,
            start_idx: 0,
            end_idx: self.assets.len(), // Exclusive index
        }
    }
}

/// A handle to a subset of some [`Data`]. This allows inference and testing to be easily performed
/// on different windows of historical data.
///
/// This handle implemented [`Copy`] for easy of use.
#[derive(Clone, Copy, Debug)]
pub struct DataHandle<'d> {
    /// The data this handle is valid for.
    data: &'d Data,
    /// The index at which we'll start providing data (inclusive).
    start_idx: usize,
    /// The index at which we'll stop providing data (exclusive). This is guaranteed by the
    /// constructor to be greater than `start_idx`, and less than the total number of observations
    /// in the dataset.
    end_idx: usize,
}
impl<'d> DataHandle<'d> {
    /// Returns the date of the first set of observations in this handle.
    pub fn start_date(&self) -> &str {
        &self.data.dates[self.start_idx]
    }

    /// Returns the date of the last set of observations in this handle.
    pub fn end_date(&self) -> &str {
        &self.data.dates[self.end_idx - 1] // Exclusive index, so we subtract 1
    }

    /// Returns the number of datapoints we have across all assets in this handle (i.e. number of
    /// rows).
    pub fn n_obs(&self) -> usize {
        self.end_idx - self.start_idx
    }

    /// Returns the number of assets we have (i.e. number of columns).
    pub fn n_assets(&self) -> usize {
        self.data.asset_names.len()
    }

    /// Returns the names of the assets in this handle.
    pub fn asset_names(&self) -> impl Iterator<Item = &str> + '_ {
        self.data.asset_names.iter().map(|name| name.as_str())
    }

    /// Returns an iterator over the rows of data we have (i.e. each element is a vector of the
    /// returns for each asset at that point).
    pub fn data(&self) -> impl ParallelIterator<Item = &Vec<f64>> {
        // Return only the rows in the range
        self.data.assets[self.start_idx..self.end_idx].par_iter()
    }

    /// Returns the fitness of a portfolio using *only* each asset. This returns an iterator, where
    /// each element is the name of an asset, and the fitness of the portfolio involving only that
    /// asset. This is useful to comparing how a composite portfolio does relative to singletons.
    ///
    /// This takes in the parameters that should be used for fitness computation.
    pub fn raw_fitnesses<'s, 'p: 's>(
        &'s self,
        params: &'p Parameters,
    ) -> impl Iterator<Item = (&'s str, f64)> + 's {
        self.asset_names()
            .enumerate()
            .map(|(asset_idx, asset_name)| {
                // For each asset, construct a portfolio weighting using only that asset
                let portfolio = Portfolio::new(
                    (0..self.n_assets())
                        .map(|idx| if idx == asset_idx { 1.0 } else { 0.0 })
                        .collect(),
                    params,
                    *self,
                );
                (asset_name, *portfolio.fitness())
            })
    }
}
