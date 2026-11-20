import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

def generate_scenario(returns_data, size=None, random_state=None):
    """
    Generates an artificial scenario of returns from the given historical data using
    stationary bootstrapping. This effectively selects random days and blocks of days
    in the past and uses them exactly in the new series, thereby preserving correlations
    between asset motions perfectly without requiring any statistical assumptions of
    fit.
    """
    if size is None:
        size = len(returns_data)

    if random_state is not None:
        np.random.seed(random_state)

    index = returns_data.index
    n = len(index)
    bootstrapped_indices = np.empty(size, dtype=np.int64)

    # Initialize the first index
    current_index = np.random.choice(index)
    bootstrapped_indices[0] = current_index

    for i in range(1, size):
        if np.random.rand() < block_prob:
            # Start a new block
            current_index = np.random.choice(index)
        else:
            # Move to the next index, wrapping around if necessary
            current_index = (current_index + 1) % n
        bootstrapped_indices[i] = current_index

    return data.iloc[bootstrapped_indices].reset_index(drop=True)

    n, num_assets = returns_data.shape
    scenario = np.zeros((n, num_assets))


def stationary_bootstrap(returns, block_prob=0.1):
    """
    Generates a new theoretical set of samples for the data using a stationary
    bootstrap on the historical data. There will be as many samples in the return value
    as there were in the original data.
    """
    # Get the number of days and assets
    n, num_assets = returns.shape

    # Generate as many datapoints as we need
    samples = np.zeros((n, num_assets))
    t = 0
    idx = np.zeros(n, dtype=int)
    while t < n:
        # Randomly choose a start point somewhere in the historical data
        start = np.random.randint(0, n)
        # Determine using the given probability and a geometric series how many
        # datapoints we should use in sequence
        run_length = np.random.geometric(p=block_prob)
        for j in range(run_length):
            if t < n:
                # Collect the indices we've chosen
                idx[t] = (start + j) % n
                t += 1

    samples[i] = returns[idx]
    return samples

# Fitness function for the genetic algorithm (negative Sharpe ratio)
def evaluate(weights, returns, risk_free_rate):
    """
    A fitness function for the genetic algorithm used to optimise the portfolio
    weights. The fitness is the negative Sharpe ratio of the portfolio.
    """
    weights = np.array(weights)
    mean_returns = np.mean(returns, axis=0)
    covariance_matrix = np.cov(returns.T)
    portfolio_return = np.dot(mean_returns, weights)
    portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))

    sharpe_ratio = (portfolio_return - risk_free_rate) / np.sqrt(portfolio_variance)

    # Return negative for minimization
    return -sharpe_ratio,

# Genetic algorithm to optimize portfolio weights
def optimize_portfolio(returns, risk_free_rate=0.01, n_generations=50, n_population=100):
    """
    Optimises a portfolio of the given returns using a genetic algorithm with the given
    properties.
    """
    num_assets = returns.shape[1]

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    # Initialize weights using a Dirichlet distribution (sum to 1)
    toolbox.register("attr_float", np.random.dirichlet, np.ones(num_assets))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Crossover with blending
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    # Mutate
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0, up=1.0, eta=0.1, indpb=0.2)
    # Tournament selection
    toolbox.register("select", tools.selTournament, tournsize=3)
    # Fitness evaluation
    toolbox.register("evaluate", evaluate, returns=returns, risk_free_rate=risk_free_rate)

    # Create the initial population
    population = toolbox.population(n=n_population)
    # Run GA
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_generations, verbose=False)

    # Select the best individual
    best_ind = tools.selBest(population, 1)[0]
    return best_ind

df = pd.read_csv("asset_data/daily_returns.csv")
returns = df.drop(columns='Date').values
print(returns)

# Bootstrap samples
bootstrapped_samples = stationary_bootstrap(returns, block_prob=0.1, n_samples=10)
# Find optimal weights
optimal_weights = optimize_portfolio(bootstrapped_samples[0])

print("Optimal Weights:", optimal_weights)
