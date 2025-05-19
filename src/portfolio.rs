use genetic::{Individual, NullTracker};
use ordered_float::NotNan;
use rand::Rng;
use std::hash::Hash;

use crate::{algorithm::AlgorithmInstance, generator::generate_portfolio};

pub(crate) static ZERO: NotNan<f64> = unsafe { NotNan::new_unchecked(0.0) };
pub(crate) static ONE: NotNan<f64> = unsafe { NotNan::new_unchecked(1.0) };

/// A single candidate portfolio. This is a vector of weights that defines an investment strategy
/// over the assets under consideration, where all weights are non-negative and sum to 1.0.
///
/// The lifetime parameter here is used for a reference to the algorithm context in which this is
/// considered as a candidate solution.
#[derive(Clone, Debug)]
pub struct Portfolio<'a> {
    // NOTE: Guaranteed by `PortfolioGenerator` to have the same length as the number of assets
    weights: Vec<NotNan<f64>>,
    handle: &'a AlgorithmInstance,
}
impl<'a> Portfolio<'a> {
    /// Creates a new [`Portfolio`] from the given weights and algorithm context handle.
    pub(crate) fn new(weights: Vec<NotNan<f64>>, handle: &'a AlgorithmInstance) -> Self {
        Self { weights, handle }
    }

    /// Returns an iterator of the weights this portfolio assigns to each asset, where the first
    /// element is the asset name and the second the weight, which is guaranteed not to be NaN.
    pub fn weights<'s>(&'s self) -> impl Iterator<Item = (&'a str, f64)> + 's {
        self.handle
            .asset_names
            .iter()
            .zip(self.weights.iter())
            .map(move |(name, weight)| (name.as_str(), **weight))
    }
}
// Manual implementations of `PartialEq`, `Eq`, and `Hash` that ignore the `handle`
impl<'a> PartialEq for Portfolio<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.weights == other.weights
    }
}
impl<'a> Eq for Portfolio<'a> {}
impl<'a> Hash for Portfolio<'a> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.weights.hash(state);
    }
}

impl<'a> Individual for Portfolio<'a> {
    type Fitness = NotNan<f64>;
    type Tracker = NullTracker;

    fn generate(_rng: &mut impl Rng) -> Self {
        unimplemented!("portfolio generation should be done using PortfolioGenerator")
    }

    fn tracker() -> Self::Tracker {
        NullTracker
    }

    // Fitness is the omega ratio of the weighted portfolio of assets
    fn fitness(&self) -> Self::Fitness {
        let (numerator, denominator) = self
            .handle
            .assets
            // Iterate through the *rows*
            .iter()
            // Now multiply each element of the row by the appropriate weight and sum them all
            // together. This leaves us with an iterator of combined returns
            .map(|row| {
                row.iter()
                    .zip(self.weights.iter())
                    .map(|(weight, r)| weight * r)
                    .sum::<NotNan<f64>>()
            })
            // Now compute the numerator and denominator of the Omega ratio.
            //   - Numerator = sum_t(max(0, weighted_r - theta))
            //   - Denominator = sum_t(max(0, theta - weighted_r))
            .fold((ZERO, ZERO), |(numerator, denominator), weighted_r| {
                let gain = weighted_r - self.handle.theta;
                let loss = self.handle.theta - weighted_r;

                (numerator + gain.max(ZERO), denominator + loss.max(ZERO))
            });

        // Handle edge cases
        let omega = if *numerator == 0.0 && *denominator == 0.0 {
            panic!("assets have zero variation in returns");
        } else if *denominator == 0.0 {
            unsafe { NotNan::new_unchecked(f64::INFINITY) }
        } else {
            numerator / denominator
        };

        omega
    }

    // Cross with a convex combination (avoids a separate normalisation step)
    fn cross(&self, other: &Self, _tracker: &Self::Tracker, rng: &mut impl Rng) -> Self {
        let alpha = NotNan::new(rng.random_range(0.0..=1.0)).unwrap();
        let mut child = Vec::with_capacity(self.weights.len());
        for (a, b) in self.weights.iter().zip(other.weights.iter()) {
            let weight = a * alpha + b * (NotNan::new(1.0).unwrap() - alpha);
            debug_assert!(*weight >= 0.0);
            child.push(weight);
        }

        Portfolio {
            weights: child,
            handle: self.handle,
        }
    }

    // Mutate by generating a noise portfolio and jittering ever so slightly toward it.
    fn mutate(&mut self, _tracker: &Self::Tracker, rng: &mut impl Rng) {
        let noise = generate_portfolio(self.weights.len(), rng);
        // SAFETY: We'll never produce a random NaN (`mutation_max_epsilon` also guaranteed to be
        // not NaN)
        let epsilon = unsafe {
            NotNan::new_unchecked(rng.random_range(0.0..=*self.handle.mutation_max_epsilon))
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
