use crate::{data::DataHandle, generator::generate_portfolio, params::Parameters};
use core::f64;
use genetic::{Individual, NullTracker};
use ordered_float::NotNan;
use rand::Rng;
use rayon::iter::ParallelIterator;
use std::hash::Hash;

pub(crate) static ZERO: NotNan<f64> = unsafe { NotNan::new_unchecked(0.0) };
pub(crate) static ONE: NotNan<f64> = unsafe { NotNan::new_unchecked(1.0) };

/// A single candidate portfolio. This is a vector of weights that defines an investment strategy
/// over the assets under consideration, where all weights are non-negative and sum to 1.0.
///
/// The lifetime parameter here is used for a reference to the parameter/data context in which this
/// is considered as a candidate solution.
#[derive(Debug)]
pub struct Portfolio<'p, 'd> {
    // NOTE: Guaranteed by `PortfolioGenerator` to have the same length as the number of assets.
    // Needs to be `NotNan<f64>` so we can hash.
    weights: Vec<NotNan<f64>>,
    params: &'p Parameters,
    data: DataHandle<'d>,
}
impl Clone for Portfolio<'_, '_> {
    fn clone(&self) -> Self {
        Self {
            weights: self.weights.clone(),
            params: self.params,
            data: self.data.clone(),
        }
    }
}
impl<'p, 'd> Portfolio<'p, 'd> {
    /// Creates a new [`Portfolio`] from the given weights, data, and parameters
    pub(crate) fn new(
        weights: Vec<NotNan<f64>>,
        params: &'p Parameters,
        data: DataHandle<'d>,
    ) -> Self {
        Self {
            weights,
            params,
            data,
        }
    }

    /// Returns an iterator of the weights this portfolio assigns to each asset, where the first
    /// element is the asset name and the second the weight, which is guaranteed not to be NaN.
    pub fn weights<'s: 'p>(&'s self) -> impl Iterator<Item = (&'p str, f64)> + 's {
        self.data
            .asset_names()
            .zip(self.weights.iter())
            .map(move |(name, weight)| (name, **weight))
    }

    /// Returns a new portfolio with the same weights as this one, over the given data period. This
    /// can be used to easily test a portfolio in a different data period than the one it was
    /// trained in.
    ///
    /// Note: when a portfolio comes out of the training algorithm, it will have the data for the
    /// training period.
    pub fn with_data<'new_d: 'd>(&self, data: DataHandle<'new_d>) -> Portfolio<'p, 'new_d> {
        Portfolio {
            weights: self.weights.clone(),
            params: self.params,
            data,
        }
    }

    /// Returns the cumulative return of this portfolio over the data handle embedded. When a
    /// [`Portfolio`] is returned from the training algorithm, this will return its performance
    /// over the training window, so typically you'll call [`Self::with_data`] first to change this
    /// window.
    ///
    /// Note: the returned `f64` is guaranteed not to be NaN.
    pub fn cumulative_return(&self) -> f64 {
        self.data
            .data()
            // Turn each row into the portfolio's performance at that point
            .map(|row| {
                row.iter()
                    .zip(self.weights.iter())
                    .map(|(r, weight)| r * **weight)
                    .sum::<f64>()
                    + 1.0
            })
            .product::<f64>()
    }
}
// Manual implementations of `PartialEq`, `Eq`, and `Hash` that ignore the `handle`
impl PartialEq for Portfolio<'_, '_> {
    fn eq(&self, other: &Self) -> bool {
        self.weights == other.weights
    }
}
impl Eq for Portfolio<'_, '_> {}
impl Hash for Portfolio<'_, '_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.weights.hash(state);
    }
}

impl<'p, 'd> Individual for Portfolio<'p, 'd> {
    type Fitness = NotNan<f64>;
    type Tracker = NullTracker;

    // We need parameters and a data handle to generate a portfolio
    fn generate(_rng: &mut impl Rng) -> Self {
        unimplemented!("portfolio generation should be done using PortfolioGenerator")
    }

    fn tracker() -> Self::Tracker {
        NullTracker
    }

    // Fitness is the omega ratio of the weighted portfolio of assets
    fn fitness(&self) -> Self::Fitness {
        let (numerator, denominator) = self
            .data
            .data()
            // Now multiply each element of the row by the appropriate weight and sum them all
            // together. This leaves us with an iterator of combined returns
            .map(|row| {
                row.iter()
                    .zip(self.weights.iter())
                    .map(|(r, weight)| r * **weight)
                    .sum::<f64>()
            })
            // Now compute the numerator and denominator of the Omega ratio.
            //   - Numerator = sum_t(max(0, weighted_r - theta))
            //   - Denominator = sum_t(max(0, theta - weighted_r))
            .fold(
                || (0.0, 0.0),
                |(numerator, denominator), weighted_r| {
                    let gain = weighted_r - *self.params.risk_free_rate();
                    let loss = *self.params.risk_free_rate() - weighted_r;

                    (numerator + gain.max(0.0), denominator + loss.max(0.0))
                },
            )
            .reduce(
                || (0.0, 0.0),
                |(num1, den1), (num2, den2)| (num1 + num2, den1 + den2),
            );

        // Handle edge cases
        let omega = if numerator == 0.0 && denominator == 0.0 {
            panic!("assets have zero variation in returns");
        } else if denominator == 0.0 {
            f64::INFINITY
        } else {
            numerator / denominator
        };

        // This could well fail (more performant to fail than to check everywhere though)
        NotNan::new(omega).expect("NaN")
    }

    // Cross with a convex combination (avoids a separate normalisation step)
    fn cross(&self, other: &Self, _tracker: &Self::Tracker, rng: &mut impl Rng) -> Self {
        // SAFETY: A randomly generated number cannot be NaN
        let alpha = unsafe { NotNan::new_unchecked(rng.random_range(0.0..=1.0)) };
        let mut child = Vec::with_capacity(self.weights.len());
        for (a, b) in self.weights.iter().zip(other.weights.iter()) {
            let weight = a * alpha + b * (ONE - alpha);
            debug_assert!(*weight >= 0.0);
            child.push(weight);
        }

        Portfolio {
            weights: child,
            params: self.params,
            data: self.data,
        }
    }

    // Mutate by generating a noise portfolio and jittering ever so slightly toward it.
    fn mutate(&mut self, _tracker: &Self::Tracker, rng: &mut impl Rng) {
        let noise = generate_portfolio(self.weights.len(), rng);
        // SAFETY: We'll never produce a random NaN (`mutation_max_jitter` also guaranteed to be
        // not NaN)
        let epsilon = unsafe {
            NotNan::new_unchecked(rng.random_range(0.0..=*self.params.mutation_max_jitter()))
        };

        self.weights
            .iter_mut()
            .enumerate()
            .for_each(|(idx, weight)| {
                let new_weight = epsilon * noise[idx] + (ONE - epsilon) * *weight;
                debug_assert!(*new_weight >= 0.0);
                *weight = new_weight;
            })
    }
}
