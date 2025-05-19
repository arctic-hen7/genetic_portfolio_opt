use crate::{algorithm::AlgorithmInstance, portfolio::Portfolio};
use genetic::{DefaultRng, Generator};
use ordered_float::NotNan;
use rand::{Rng, SeedableRng};
use rand_distr::Uniform;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::sync::LazyLock;

/// The uniform distribution from 0 to 1. We keep this global for minor speed improvements.
static UNIFORM_0_1: LazyLock<Uniform<f64>> = LazyLock::new(|| Uniform::new(0.0, 1.0).unwrap());

/// A dedicated and optimised generator for portfolios with a given number of assets.
pub struct PortfolioGenerator<'a> {
    handle: &'a AlgorithmInstance,
}
impl<'a> PortfolioGenerator<'a> {
    /// Creates a new portfolio generator for portfolios over the given algorithm handle.
    pub fn new(handle: &'a AlgorithmInstance) -> Self {
        PortfolioGenerator { handle }
    }
}
impl<'a> Generator<Portfolio<'a>> for PortfolioGenerator<'a> {
    fn generate(&self, size: usize, rng: &mut impl Rng) -> Vec<Portfolio<'a>> {
        // Generate in parallel (slightly faster) using a global seed and per-iteration re-seeding
        // TODO: Benchmark against sequential
        let global_seed = rng.random::<u64>();
        (0..size)
            .into_par_iter()
            .map(|idx| {
                let mut rng = DefaultRng::seed_from_u64(idx as u64 ^ global_seed);

                let portfolio = generate_portfolio(self.handle.n_assets(), &mut rng);
                Portfolio::new(portfolio, self.handle)
            })
            .collect()
    }
}

/// Generates a single portfolio with the given number of assets.
pub(crate) fn generate_portfolio(n_assets: usize, rng: &mut impl Rng) -> Vec<NotNan<f64>> {
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
        portfolio.push(unsafe { NotNan::new_unchecked(-uniform.ln()) });
    }
    let sum: NotNan<f64> = portfolio.iter().sum();
    portfolio.iter_mut().for_each(|x| *x /= sum);

    portfolio
}
