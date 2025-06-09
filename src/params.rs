/// The hyperparameters involved in training and testing the optimisation algorithm. These should
/// be set in advance.
#[derive(Debug)]
pub struct Parameters {
    /// The threshold risk-free rate.
    risk_free_rate: f64,
    /// The maximum "jitter" for a mutation.
    mutation_max_jitter: f64,
}
impl Parameters {
    /// Creates a new set of [`Parameters`].
    ///
    /// # Panics
    ///
    /// This will panic if either of the provided values are `NaN`.
    pub fn new(risk_free_rate: f64, mutation_max_jitter: f64) -> Self {
        assert!(!risk_free_rate.is_nan(), "risk_free_rate must not be NaN");
        assert!(
            !mutation_max_jitter.is_nan(),
            "mutation_max_jitter must not be NaN"
        );

        Self {
            risk_free_rate,
            mutation_max_jitter,
        }
    }

    /// Returns the risk-free rate of return.
    pub fn risk_free_rate(&self) -> f64 {
        self.risk_free_rate
    }

    /// Returns the maximum allowed "jitter" for a mutation.
    pub fn mutation_max_jitter(&self) -> f64 {
        self.mutation_max_jitter
    }
}
