mod data;
mod generator;
mod params;
mod parse_csv;
mod portfolio;

use data::Data;
use generator::PortfolioGenerator;
use genetic::{selectors::TournamentSelector, Individual, SimulationBuilder, StaticElitistUpdater};
use params::Parameters;
use portfolio::Portfolio;

const NUM_TRADING_DAYS: usize = 252;
const RISK_FREE_RATE_PA: f64 = 0.051;
const WINDOW_LENGTH: usize = NUM_TRADING_DAYS * 5;
const TEST_WINDOW_LENGTH: usize = WINDOW_LENGTH / 3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compute the daily risk-free rate and report
    let daily_risk_free_rate = (1.0 + RISK_FREE_RATE_PA).powf(1.0 / NUM_TRADING_DAYS as f64) - 1.0;
    println!(
        "Using daily risk-free rate {:.3}%",
        daily_risk_free_rate * 100.0
    );

    // Prepare the algorithm (parsing the data)
    let params = Parameters::new(daily_risk_free_rate, 0.25);
    let data = Data::from_file("asset_data/daily_returns.csv")?;
    println!("Parsed {} observations.", data.all().n_obs());

    println!();

    // Generate a random seed to use for every optimisation
    let seed = rand::random::<u64>();

    let mut training_range = 0..WINDOW_LENGTH;
    loop {
        // Advance the training range by the length of the testing window
        training_range =
            (training_range.start + TEST_WINDOW_LENGTH)..(training_range.end + TEST_WINDOW_LENGTH);
        // The testing range will then be the *next* testing window after that
        let mut testing_range = training_range.end..(training_range.end + TEST_WINDOW_LENGTH);

        // If the training range is longer than the data, we can't train any more
        if training_range.end > data.all().n_obs() {
            break;
        }
        // But if the training range is within and the testing range is without, we can still
        // train, we'll just be testing on a shorter period
        if testing_range.end > data.all().n_obs() {
            testing_range.end = data.all().n_obs();
        }

        // By the above checks, these won't panic
        let training_data = data.handle(training_range.clone());
        let testing_data = data.handle(testing_range);

        println!(
            "Training on data from {} to {} ({} observations); testing on data from {} to {} ({} observations)",
            training_data.start_date(),
            training_data.end_date(),
            training_data.n_obs(),
            testing_data.start_date(),
            testing_data.end_date(),
            testing_data.n_obs()
        );

        // Train for the best portfolio in the training data
        let solutions = SimulationBuilder::<Portfolio>::new()
            .with_selector(TournamentSelector::new(250))
            .with_updater(StaticElitistUpdater::new_worst(10))
            .with_generator(PortfolioGenerator::new(&params, training_data))
            .with_initial_population_size(100_000)
            .max_generations(250)
            .with_seed(seed)
            .run()?;
        let best_portfolio = solutions.best().unwrap().with_data(testing_data);
        let best_fitness = best_portfolio.fitness();
        let best_cumul_return = best_portfolio.cumulative_return();
        let best_cumul_return_pct = (best_cumul_return - 1.0) * 100.0;
        let best_annual_return_pct =
            (best_cumul_return.powf(252.0 / testing_data.n_obs() as f64) - 1.0) * 100.0;

        // Check if the best portfolio found is better than the best singletons (over the test
        // period)
        let mut outperformers = Vec::new();
        let mut best_singleton = (String::new(), 0.0);
        for (name, fitness) in testing_data.raw_fitnesses(&params) {
            if fitness > *best_fitness {
                outperformers.push(format!("  - {name}: {fitness:.3}Ω"));
            }
            if fitness > best_singleton.1 {
                best_singleton = (name.to_string(), fitness);
            }
        }
        let outperformers_report = if outperformers.is_empty() {
            "No outperforming singletons.".to_string()
        } else {
            format!("Outperforming singletons:\n{}", outperformers.join("\n"))
        };

        // Report the best portfolio found by the algorithm and its fitness
        let best_weights = best_portfolio
            .weights()
            .map(|(name, weight)| format!("  - {name}: {:.2}%", weight * 100.0))
            .collect::<Vec<_>>()
            .join("\n");
        println!("Best Omega ratio: {:.3}Ω", best_fitness);
        println!("Best solution:");
        println!("{best_weights}");
        println!("Omega ratio: {best_fitness:.3}Ω");
        println!("Return rate: {best_annual_return_pct:.3}% (cum: {best_cumul_return_pct:.3}%)");
        println!();
        println!("{outperformers_report}");
    }
    // Report the seed for reproducibility
    println!("Seed: {}", seed);

    Ok(())
}
