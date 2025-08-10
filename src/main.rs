mod data;
mod engine;
mod errors;
mod fitness;
mod generator;
mod params;
mod parse_csv;
mod portfolio;

use clap::Parser;
use data::Data;
use engine::{
    Engine, EngineBuilder, Strategy, DEFAULT_ELITES, DEFAULT_MAX_GENERATIONS,
    DEFAULT_PLATEAU_GENERATIONS, DEFAULT_PLATEAU_SLOPE_THRESHOLD, DEFAULT_PLATEAU_WINDOW,
    DEFAULT_POPULATION_SIZE,
};
use errors::CliError;
use genetic::{selectors::TournamentSelector, PairSelector};
use portfolio::Portfolio;
use std::path::PathBuf;

type Float = f64;

const NUM_TRADING_DAYS: usize = 252;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts = Cli::parse();
    let engine = opts.to_engine()?;
    let runs = engine.par_run()?;

    for (idx, metrics) in runs.iter().enumerate() {
        println!("=== Run #{} ===", idx + 1);
        println!(
            "Training range: {} to {} ({} observations)",
            metrics.training_start_date(),
            metrics.training_end_date(),
            metrics.training_length()
        );
        println!(
            "Testing range: {} to {} ({} observations)",
            metrics.testing_start_date(),
            metrics.testing_end_date(),
            metrics.testing_length()
        );
        println!();
        println!("Best portfolio weights:");
        for (name, weight) in metrics.best_weights() {
            println!("  - {name}: {:.2}%", weight * 100.0);
        }
        println!("Fitness: {:.3}", metrics.fitness());
        println!(
            "Return rate: {:.3}% (cum: {:.3}%)",
            metrics.normalised_return(opts.days) * 100.0,
            (metrics.cumulative_return() - 1.0) * 100.0
        );
        println!();
    }
    // Compute the cumulative return over all runs (assumes non-overlapping)
    let total_return = runs.iter().fold(1.0, |acc, m| acc * m.cumulative_return());
    let total_days = runs.iter().map(|m| m.testing_length()).sum::<usize>();
    let total_periods = total_days as Float / opts.days as Float;
    let annual_return = total_return.powf(opts.days as Float / total_days as Float) - 1.0;
    println!("=== Summary ===");
    println!();
    println!("Seed: {}", runs[0].seed());
    println!(
        "Assuming non-overlapping test periods, total cumulative return: {:.3}% ({:.1} years)",
        (total_return - 1.0) * 100.0,
        total_periods
    );
    println!("ANNUAL RETURN: {:.3}%", annual_return * 100.0);

    Ok(())
}

/// CLI for running portfolio optimisation experiments
#[derive(Parser)]
pub struct Cli {
    /// Path to the CSV file containing asset data
    csv_file: PathBuf,

    /// The number of days in a year in your data (or periods per year)
    #[arg(short = 'd', long = "days", default_value_t = NUM_TRADING_DAYS)]
    days: usize,
    /// A seed for reproducibility (default: randomly generated)
    #[arg(long = "seed")]
    seed: Option<u64>,
    /// The strategy to use for the portfolio optimisation (one of: `full`,
    /// `sliding:<training_len>:<testing_len>`,
    /// `expanding:<initial_training_length>:<testing_length>:<increment_length>`,
    /// `static:<training_fraction>`)
    #[arg(short = 's', long = "strategy", default_value = "static:0.7")]
    strategy: String,
    /// The selector to use for the genetic algorithm (one of: `tournament:<size>`)
    // TODO: More...(Boltzmann needs to impl Clone)
    #[arg(long = "selector", default_value = "tournament:70")]
    selector: String,
    /// The annual risk-free rate to optimise against
    #[arg(short = 'r', long = "risk-free-rate")]
    risk_free_rate: Float,
    /// The maximum "jitter" factor to apply in portflio mutation (higher for more exploration)
    #[arg(short = 'j', long = "max-jitter", default_value_t = 0.25)]
    max_jitter: Float,
    /// The maximum number of generations to run the optimisation for
    #[arg(short = 'g', long = "max-generations", default_value_t = DEFAULT_MAX_GENERATIONS)]
    max_generations: usize,
    /// The number of best-performing portfolios to carry over each generation
    #[arg(short = 'e', long = "elites", default_value_t = DEFAULT_ELITES)]
    elites: usize,
    /// The size of the population to use for the optimisation
    #[arg(short = 'p', long = "population-size", default_value_t = DEFAULT_POPULATION_SIZE)]
    population_size: usize,
    /// Threshold for the slope of the fitness plateau (in Ω) to stop the optimisation (lower to
    /// "try harder")
    #[arg(short = 't', long = "plateau-slope-threshold", default_value_t = DEFAULT_PLATEAU_SLOPE_THRESHOLD)]
    plateau_slope_threshold: Float,
    /// The number of generations to consider for the fitness plateau
    #[arg(long = "plateau-window", default_value_t = DEFAULT_PLATEAU_WINDOW)]
    plateau_window: usize,
    /// The number of generations to wait before stopping the optimisation if the fitness plateaus
    #[arg(long = "plateau-generations", default_value_t = DEFAULT_PLATEAU_GENERATIONS)]
    plateau_generations: usize,
}
impl Cli {
    /// Converts these CLI arguments into an [`Engine`] that can be run to optimise portfolios.
    pub fn to_engine(
        &self,
    ) -> Result<Engine<impl for<'a> PairSelector<Portfolio<'a, 'a>> + Clone>, CliError> {
        // Parse the CSV file into data
        let data = Data::from_file(&self.csv_file)?;
        let strategy =
            Strategy::from_str(&self.strategy).map_err(|err| CliError::InvalidStrategy(err))?;
        // Configure with all the parameters that don't change the type
        let engine = EngineBuilder::new()
            .with_data(data)
            .with_strategy(strategy)
            .with_risk_free_rate(self.risk_free_rate, self.days)
            .with_max_jitter(self.max_jitter)
            .with_max_generations(self.max_generations)
            .with_elites(self.elites)
            .with_population_size(self.population_size)
            .with_halter(
                self.plateau_window,
                self.plateau_slope_threshold,
                self.plateau_generations,
            );
        // And a seed if we have one
        let engine = if let Some(seed) = self.seed {
            engine.with_seed(seed)
        } else {
            engine
        };

        // Now branch on the selector
        match self.selector.as_str() {
            s if s.starts_with("tournament:") => {
                let size = s
                    .strip_prefix("tournament:")
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| {
                        CliError::InvalidSelector("tournament selector requires size")
                    })?;
                let engine = engine.with_selector(TournamentSelector::new(size));

                Ok(engine.build().unwrap())
            }
            // "roulette" => {
            //     let engine = engine.with_selector(RouletteSelector);
            // },
            _ => Err(CliError::InvalidSelector(
                "expected `tournament:<size>` or `roulette`",
            )),
        }
    }
}
