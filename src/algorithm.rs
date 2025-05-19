use crate::{
    parse_csv::{parse_csv, CsvParseError},
    portfolio::Portfolio,
};
use genetic::Individual;
use ordered_float::NotNan;
use std::io::BufRead;

/// The context of a run of the portfolio optimisation algorithm, containing hyperparameters and
/// asset return data. Note that this object is likely to be extremely large in memory.
#[derive(Debug)]
pub struct AlgorithmInstance {
    /// The threshold risk-free rate for computing the Omega ratio.
    pub(crate) theta: NotNan<f64>,
    /// The maximum "jitter" for a mutation.
    pub(crate) mutation_max_epsilon: NotNan<f64>,
    /// The assets we're working with, which are guaranteed to all have the same length (and
    /// semantically should align in time as well).
    ///
    /// Rather than storing these as columns, we store them as rows. The length of the inner vector
    /// is the number of assets, while that of the outer vector is the number of returns. This
    /// allows more efficient iterator zipping for combining returns into weighted portfolios for
    /// fitness computation. It's also easier to read directly from a CSV file.
    pub(crate) assets: Vec<Vec<NotNan<f64>>>,
    /// The names of each asset.
    pub(crate) asset_names: Vec<String>,
    /// The dates for which we have recorded asset data. These are likely to not be continuous, as
    /// holidays and periods where one or more assets aren't reporting will be dropped.
    pub(crate) dates: Vec<String>,
}
impl AlgorithmInstance {
    /// Creates a new algorithm instance with the given parameters, and a reader of the CSV file
    /// containing the asset data.
    ///
    /// # Parameters
    ///
    /// The optimisation algorithm takes two parameters:
    ///   - `theta` is the risk-free rate of return, expressed in whatever terms the individual
    ///   asset returns are (typically daily)
    ///   - `max_mutation_epsilon` is the maximum "jitter" experienced by candidate solutions
    ///   during the mutation stage of the genetic algorithm (should generally be small; higher
    ///   values favour exploring new solutions over exploiting known ones, and vice versa; tune as
    ///   needed)
    ///
    /// # Asset Data
    ///
    /// The CSV file must be comma-separated and include a header with column names, which will be
    /// read as asset names, with the exception of the first column, which will be read as a date
    /// column. All other rows should contain daily percentage returns for each asset (e.g. 0.1
    /// implies a 0.1% return on the given day, *not* 10%).
    ///
    /// Any rows where all assets have 0.0 returns will be considered weekends/trading holidays and
    /// dropped. If any asset has more than three 0.0 returns in a row, that period will be dropped
    /// *for all assets* until the asset starts reporting again (i.e. only the intersection where
    /// all assets are reporting values will be considered). The resulting number of observations
    /// will be stored.
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
    ///
    /// # Panics
    ///
    /// This will panic if the given `theta` or `mutation_max_epsilon` is NaN.
    pub fn new(
        theta: f64,
        mutation_max_epsilon: f64,
        assets_csv: impl BufRead,
    ) -> Result<Self, CsvParseError> {
        let theta = NotNan::new(theta).expect("theta must not be NaN");
        let mutation_max_epsilon =
            NotNan::new(mutation_max_epsilon).expect("mutation_max_epsilon must not be NaN");

        let (asset_names, dates, assets) = parse_csv(assets_csv)?;

        Ok(AlgorithmInstance {
            theta,
            mutation_max_epsilon,
            assets,
            asset_names,
            dates,
        })
    }

    /// Returns the number of observations (i.e. the length of each asset's return history).
    pub fn n_obs(&self) -> usize {
        // Outer vector is a vector of rows
        self.assets.len()
    }
    /// Returns the number of assets we have.
    pub fn n_assets(&self) -> usize {
        // Length of the first row
        self.assets[0].len()
    }

    /// Returns the fitness of a portfolio using *only* each asset. This returns an iterator, where
    /// each element is the name of an asset, and the fitness of the portfolio involving only that
    /// asset. This is useful to comparing how a composite portfolio does relative to singletons.
    pub fn raw_fitnesses<'s>(&'s self) -> impl Iterator<Item = (&'s str, f64)> + 's {
        self.asset_names
            .iter()
            .enumerate()
            .map(|(asset_idx, asset_name)| {
                // For each asset, construct a portfolio weighting using only that asset
                let portfolio = Portfolio::new(
                    (0..self.n_assets())
                        .map(|idx| {
                            if idx == asset_idx {
                                NotNan::new(1.0).unwrap()
                            } else {
                                NotNan::new(0.0).unwrap()
                            }
                        })
                        .collect(),
                    self,
                );
                (asset_name.as_str(), *portfolio.fitness())
            })
    }
}
