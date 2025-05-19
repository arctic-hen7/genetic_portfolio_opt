mod algorithm;
mod generator;
mod parse_csv;
mod portfolio;

use algorithm::AlgorithmInstance;
use generator::PortfolioGenerator;
use genetic::{selectors::TournamentSelector, SimulationBuilder, StaticElitistUpdater};
use portfolio::Portfolio;
use std::{fs::File, io::BufReader};

const NUM_TRADING_DAYS: usize = 252;
const RISK_FREE_RATE_PA: f64 = 0.051;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compute the daily risk-free rate and report
    let daily_risk_free_rate =
        ((1.0 + RISK_FREE_RATE_PA).powf(1.0 / NUM_TRADING_DAYS as f64) - 1.0) * 100.0;
    println!("Using daily risk-free rate {:.3}%", daily_risk_free_rate);

    // Prepare the algorithm (parsing the data)
    let algorithm = AlgorithmInstance::new(
        daily_risk_free_rate,
        0.25,
        BufReader::new(File::open("data/unisuper_clean.csv")?),
    )?;
    println!("Parsed {} observations.", algorithm.n_obs());

    println!();

    // Compute the fitness of each singleton portfolio and report
    let raw_fitnesses = algorithm
        .raw_fitnesses()
        .map(|(name, fitness)| format!("  - {name}: {fitness:.3}Ω"))
        .collect::<Vec<_>>();
    let raw_fitnesses_str = raw_fitnesses.join("\n");
    println!("Omega ratios for singleton portfolios:");
    println!("{raw_fitnesses_str}");

    // Run the algorithm
    let solutions = SimulationBuilder::<Portfolio>::new()
        .with_selector(TournamentSelector::new(150))
        .with_updater(StaticElitistUpdater::new_worst(10))
        .with_generator(PortfolioGenerator::new(&algorithm))
        .with_initial_population_size(10_000)
        .max_generations(100)
        .with_seed(0)
        .run()?;

    println!();

    // Report the best portfolio found by the algorithm and its fitness
    let best_weights = solutions
        .best()
        .unwrap()
        .weights()
        .map(|(name, weight)| format!("  - {name}: {:.2}%", weight * 100.0))
        .collect::<Vec<_>>()
        .join("\n");
    println!(
        "Best Omega ratio: {:.3}Ω",
        solutions.best().unwrap().fitness()
    );
    println!("Best solution:");
    println!("{best_weights}");

    println!();

    // Work out if any of the singletons did better
    let mut outperformers = Vec::new();
    let mut best_singleton = (String::new(), 0.0);
    for (name, fitness) in algorithm.raw_fitnesses() {
        if fitness > *solutions.best().unwrap().fitness() {
            outperformers.push(format!("  - {name}: {fitness:.3}Ω"));
        }
        if fitness > best_singleton.1 {
            best_singleton = (name.to_string(), fitness);
        }
    }
    let outperformers_str = outperformers.join("\n");
    println!("Outperforming singletons:");
    if outperformers.is_empty() {
        let better_by = *solutions.best().unwrap().fitness() / best_singleton.1 - 1.0;
        println!(
            "  - None ({better_by:.2}% better than best singleton {})",
            best_singleton.0
        );
    } else {
        println!("{outperformers_str}");
    }

    println!();

    // Report the seed for reproducibility
    println!("Seed: {}", solutions.seed());

    // TODO: Backtest, also compute Omegas for each individual allocation

    Ok(())
}
