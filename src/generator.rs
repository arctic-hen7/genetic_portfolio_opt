use crate::{data::DataHandle, params::Parameters, portfolio::Portfolio, Float};
use genetic::{DefaultRng, Generator};
use rand::{Rng, SeedableRng};
use rand_distr::Uniform;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::sync::LazyLock;

/// The uniform distribution from 0 to 1. We keep this global for minor speed improvements.
static UNIFORM_0_1: LazyLock<Uniform<Float>> = LazyLock::new(|| Uniform::new(0.0, 1.0).unwrap());

/// A dedicated and optimised generator for portfolios over the given parameters.
pub struct PortfolioGenerator<'p, 'd> {
    params: &'p Parameters,
    data: DataHandle<'d>,
}
impl<'p, 'd> PortfolioGenerator<'p, 'd> {
    /// Creates a new portfolio generator for portfolios over the given parameters and data.
    pub fn new(params: &'p Parameters, data: DataHandle<'d>) -> Self {
        PortfolioGenerator { params, data }
    }
}
impl<'p, 'd> Generator<Portfolio<'p, 'd>> for PortfolioGenerator<'p, 'd> {
    fn generate(&self, size: usize, rng: &mut impl Rng) -> Vec<Portfolio<'p, 'd>> {
        // Generate in parallel (slightly faster) using a global seed and per-iteration re-seeding
        // TODO: Benchmark against sequential
        let global_seed = rng.random::<u64>();
        (0..size)
            .into_par_iter()
            .map(|idx| {
                let mut rng = DefaultRng::seed_from_u64(idx as u64 ^ global_seed);

                let portfolio = generate_portfolio(self.data.n_assets(), &mut rng);
                Portfolio::new(portfolio, self.params, self.data)
            })
            .collect()
    }
}

/// Generates a single portfolio with the given number of assets.
pub(crate) fn generate_portfolio(n_assets: usize, rng: &mut impl Rng) -> Vec<Float> {
    let mut portfolio = Vec::with_capacity(n_assets);
    for _ in 0..n_assets {
        // Get a uniformly distributed value that's guaranteed not to be zero (so
        // taking the log of it is valid). The chance of getting 0 is miniscule, but it
        // is possible.
        let uniform = loop {
            let uniform = rng.sample(*UNIFORM_0_1);
            if uniform == 0.0 {
                continue;
            } else {
                break uniform;
            }
        };
        // Now this will never fail (`uniform > 0.0` guaranteed)
        debug_assert!(uniform > 0.0, "uniform sample was zero");
        portfolio.push(-uniform.ln());
    }
    let sum: Float = portfolio.iter().sum();
    portfolio.iter_mut().for_each(|x| *x /= sum);

    portfolio
}
