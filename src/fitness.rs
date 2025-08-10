use crate::{portfolio::Portfolio, Float};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub fn omega_ratio(portfolio: &Portfolio) -> Float {
    let weights = &portfolio.weights;
    let rfr = portfolio.params.risk_free_rate();

    // single pass: each thread accumulates its own (num, den)
    let (numerator, denominator) = portfolio
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

    // Handle edge cases
    let omega = if numerator == 0.0 && denominator == 0.0 {
        panic!("assets have zero variation in returns");
    } else if denominator == 0.0 {
        Float::INFINITY
    } else {
        numerator / denominator
    };

    omega
}

pub fn sharpe_ratio(portfolio: &Portfolio) -> f64 {
    let weights = &portfolio.weights;
    let rfr = portfolio.params.risk_free_rate();
    let rows = portfolio.data.data();

    // Compute count, mean, and M2 (sum of squared deviations) of excess returns using Welford in parallel
    // Mean is the average excess return
    // M2 accumulates the sum of squared differences from the mean and gives variance when divided by n-1
    let (n, mean, m2) = rows
        .into_par_iter()
        .map(|row| {
            // Compute portfolio return for this row via dot product
            debug_assert_eq!(
                row.len(),
                weights.len(),
                "Row and weight lengths must match"
            );
            let pr: f64 = row.iter().zip(weights).map(|(x, w)| x * w).sum();
            // Convert to excess return by subtracting the risk-free rate
            pr - rfr
        })
        .fold(
            // Initialize local accumulator for each worker
            || (0usize, 0.0f64, 0.0f64),
            |(n, mean, m2), x| {
                // Update Welford accumulator with one new observation
                let n1 = n + 1;
                let delta = x - mean;
                let mean1 = mean + delta / n1 as f64;
                let m2_1 = m2 + delta * (x - mean1);
                (n1, mean1, m2_1)
            },
        )
        .reduce(
            // Identity for reduce
            || (0usize, 0.0f64, 0.0f64),
            |(n1, mean1, m2_1), (n2, mean2, m2_2)| {
                // Merge two Welford accumulators
                if n1 == 0 {
                    return (n2, mean2, m2_2);
                }
                if n2 == 0 {
                    return (n1, mean1, m2_1);
                }
                let n = n1 + n2;
                let delta = mean2 - mean1;
                let mean = mean1 + delta * (n2 as f64 / n as f64);
                let m2 = m2_1 + m2_2 + delta * delta * (n1 as f64 * n2 as f64 / n as f64);
                (n, mean, m2)
            },
        );

    // Require at least two observations for a meaningful standard deviation
    if n < 2 {
        return f64::NAN;
    }

    // Use sample variance which is common in finance
    let mut var = m2 / (n - 1) as f64;

    // Clamp tiny negative variance that can occur from floating point rounding
    if var < 0.0 && var > -1e-15 {
        var = 0.0;
    }

    let std = var.sqrt();

    // Handle zero or non-finite standard deviation
    if !std.is_finite() || std <= 0.0 {
        return f64::NAN;
    }

    // Sharpe is mean excess return over standard deviation of excess returns
    mean / std
}
