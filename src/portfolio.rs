use crate::{data::DataHandle, generator::generate_portfolio, params::Parameters, Float};
use genetic::{Individual, NullTracker};
use libc::memcmp;
use ordered_float::NotNan;
use rand::Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{ffi::c_void, hash::Hash};

/// A single candidate portfolio. This is a vector of weights that defines an investment strategy
/// over the assets under consideration, where all weights are non-negative and sum to 1.0.
///
/// The lifetime parameter here is used for a reference to the parameter/data context in which this
/// is considered as a candidate solution.
#[derive(Debug)]
pub struct Portfolio<'p, 'd> {
    // NOTE: Guaranteed by `PortfolioGenerator` to have the same length as the number of assets.
    weights: Vec<Float>,
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
    pub(crate) fn new(weights: Vec<Float>, params: &'p Parameters, data: DataHandle<'d>) -> Self {
        Self {
            weights,
            params,
            data,
        }
    }

    /// Returns an iterator of the weights this portfolio assigns to each asset, where the first
    /// element is the asset name and the second the weight, which is guaranteed not to be NaN.
    pub fn weights<'s: 'p>(&'s self) -> impl Iterator<Item = (&'p str, Float)> + 's {
        self.data
            .asset_names()
            .zip(self.weights.iter())
            .map(move |(name, weight)| (name, *weight))
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
    /// Note: the returned `Float` is guaranteed not to be NaN.
    pub fn cumulative_return(&self) -> Float {
        self.data
            .data()
            .into_par_iter()
            // Turn each row into the portfolio's performance at that point
            .map(|row| {
                row.iter()
                    .zip(self.weights.iter())
                    .map(|(r, weight)| r * weight)
                    .sum::<Float>()
                    + 1.0
            })
            .product::<Float>()
    }
}
// Manual implementations of `PartialEq`, `Eq`, and `Hash` that ignore the handles
impl PartialEq for Portfolio<'_, '_> {
    fn eq(&self, other: &Self) -> bool {
        // This is a hot path, so we check lumps of bytes using `memcmp`
        debug_assert!(
            self.weights.len() == other.weights.len(),
            "portfolios must have the same number of assets"
        );

        let byte_len = self.weights.len() * std::mem::size_of::<Float>();
        // SAFETY: We know `self.weights` and `other.weights` point to at least `byte_len` valid
        // bytes.
        unsafe {
            memcmp(
                self.weights.as_ptr() as *const c_void,
                other.weights.as_ptr() as *const c_void,
                byte_len,
            ) == 0
        }
    }
}
impl Eq for Portfolio<'_, '_> {}
impl Hash for Portfolio<'_, '_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // We need a custom hash implementation because we're working with floats, which don't hash
        // (no total order). However, we can reinterpret the floats as integers and hash those
        // happily. This is generic over whatever kind of float we use (`f32` or `Float`).

        // SAFETY: Float is exactly 8 bytes and has the same alignment
        // requirements as u64, so we can re‐interpret the memory
        // as a u8‐byte slice of length len*8.
        //
        // SAFETY: `self.weights` a vector of floats, and each one can be reinterpreted as the
        // same-sized unsigned integer type (same size and alignment requirements).
        let ptr = self.weights.as_ptr() as *const u8;
        // In our array, we're splitting every float into 4/8 bytes, so this is the correct
        // calculation!
        let num_elems = self.weights.len() * std::mem::size_of::<Float>();
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(ptr, num_elems) };
        state.write(bytes);
        // NOTE: We don't write the number of elements because in a given simulation, every
        // portfolio will have the same length. If this ever proves a problem, we can add one more
        // write.
    }
}

impl<'p, 'd> Individual for Portfolio<'p, 'd> {
    type Fitness = NotNan<Float>;
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
        let weights = &self.weights;
        let rfr = self.params.risk_free_rate();

        // single pass: each thread accumulates its own (num, den)
        let (numerator, denominator) = self
            .data
            .data()
            .into_par_iter()
            // PERF: This needs to be faster!
            .fold(
                || (0.0 as Float, 0.0 as Float),
                |(mut numer, mut denom), row| {
                    // 1. dot(row, weights)
                    let len = weights.len();
                    let mut sum = 0.0 as Float;

                    // safe fallback for small or non-x86 targets:
                    // for i in 0..len { sum += row[i] * weights[i]; }

                    // unsafe pointer loop, zero bounds checks:
                    unsafe {
                        let rp = row.as_ptr();
                        let wp = weights.as_ptr();
                        for i in 0..len {
                            sum += *rp.add(i) * *wp.add(i);
                        }
                    }

                    // 2. Omega numerator/denominator increment
                    let diff = sum - rfr;
                    numer += diff.max(0.0);
                    denom += (-diff).max(0.0);

                    (numer, denom)
                },
            )
            .reduce(
                || (0.0 as Float, 0.0 as Float),
                |(n1, d1), (n2, d2)| (n1 + n2, d1 + d2),
            );

        // let (numerator, denominator) = self
        //     .data
        //     .data()
        //     // Now multiply each element of the row by the appropriate weight and sum them all
        //     // together. This leaves us with an iterator of combined returns
        //     .map(|row| {
        //         row.iter()
        //             .zip(self.weights.iter())
        //             .map(|(r, weight)| r * weight)
        //             .sum::<Float>()
        //     })
        //     // Now compute the numerator and denominator of the Omega ratio.
        //     //   - Numerator = sum_t(max(0, weighted_r - theta))
        //     //   - Denominator = sum_t(max(0, theta - weighted_r))
        //     .fold(
        //         || (0.0, 0.0),
        //         |(numerator, denominator), weighted_r| {
        //             let gain = weighted_r - self.params.risk_free_rate();
        //             let loss = self.params.risk_free_rate() - weighted_r;
        //
        //             (numerator + gain.max(0.0), denominator + loss.max(0.0))
        //         },
        //     )
        //     .reduce(
        //         || (0.0, 0.0),
        //         |(num1, den1), (num2, den2)| (num1 + num2, den1 + den2),
        //     );

        // Handle edge cases
        let omega = if numerator == 0.0 && denominator == 0.0 {
            panic!("assets have zero variation in returns");
        } else if denominator == 0.0 {
            Float::INFINITY
        } else {
            numerator / denominator
        };

        // PERF: This could absolutely fail, and we wouldn't know how it had or where it had.
        // However, this is *significantly* more performant than using `NotNan` everywhere, those
        // checks show up as real time spent on a flamegraph. So, we just panic if we see a NaN,
        // and you would haev to debu instrument your way out of it, which isn't reproducible at
        // all, but it hasn't happened yet!
        NotNan::new(omega).expect("NaN")
    }

    // Cross with a convex combination (avoids a separate normalisation step)
    fn cross(&self, other: &Self, _tracker: &Self::Tracker, rng: &mut impl Rng) -> Self {
        let alpha = rng.random_range(0.0..=1.0);
        let mut child = Vec::with_capacity(self.weights.len());
        for (a, b) in self.weights.iter().zip(other.weights.iter()) {
            let weight = a * alpha + b * (1.0 - alpha);
            debug_assert!(weight >= 0.0);
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
        let epsilon = rng.random_range(0.0..=self.params.mutation_max_jitter());

        self.weights
            .iter_mut()
            .enumerate()
            .for_each(|(idx, weight)| {
                let new_weight = epsilon * noise[idx] + (1.0 - epsilon) * *weight;
                debug_assert!(new_weight >= 0.0);
                *weight = new_weight;
            })
    }
}
