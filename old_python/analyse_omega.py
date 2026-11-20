#!/usr/bin/python3
# Runs Omega ratio analysis on the provided dataset. This script takes a single filename
# and parses it as a CSV file. The CSV file should have a column for each asset, along with
# a date column (whose format can be set with `--date-format`). Each row should be something
# equivalent to a price or return multiplier for that asset on that date. Any non-trading
# periods or periods where only some assets are trading should be pre-removed, as these will
# be considered valid 0.0s otherwise.

import argparse
import pandas as pd
from scipy import stats
import numpy as np
from scipy.integrate import quad, simpson
import cvxpy as cp
from statsmodels.distributions.empirical_distribution import ECDF
import json

def clean_data(filename, date_format):
    """
    Cleans the data to remove redundant columns and convert cumulative periodic returns to direct
    periodic percentage returns.

    Returns a tuple of the data itself, the first date, and the last date.
    """
    data = pd.read_csv(filename)
    # Extract the date separately
    date = pd.to_datetime(data["Date"], format=date_format)
    data.drop(columns="Date", inplace=True)
    # We have periodic prices or return multipliers (same thing relatively), convert them to
    # percentage periodic returns
    for col in data.columns:
        data[col] = data[col].pct_change() * 100

    # Delete the first row of NaNs
    data = data.iloc[1:]

    return data, date[0], date[len(date) - 1]

def fit_data(data, ks_signif_level = 0.10):
    """
    Fits the data to an appropriate series of theoretical distributions. Returns a dict of column names to
    parameter dicts (tuple of shape parameters, location, and scale).

    This also takes a significance level for the Kolmogorov-Smirnov test to determine if the data fits the
    chosen distribution well enough. If the p-value is less than this value, the null hypothesis *could* be
    rejected, indicating an ill fit. The p-values will also be returned as a dict.
    """
    options = {}
    p_values = {}
    for column in data.columns:
        col_data = data[column]

        dist_params = stats.t.fit(col_data)
        options[column] = dist_params

        # Run a Kolmogorov-Smirnov test to see if the data fits the distribution
        ks_p_value = stats.kstest(col_data, "t", args=dist_params)[1]
        if ks_p_value < ks_signif_level:
            print(f"Warning: {column} does not fit the t-distribution well ({ks_p_value} > {ks_signif_level}).")
        p_values[column] = ks_p_value

    return options, p_values

def omega_ratio_theoretical(threshold, distribution):
    """
    Compute the Omega ratio for the given threshold and the given t-distribution parameters by numerical
    integration. This is relatively fast.
    """
    cdf = lambda x: stats.t.cdf(x, *distribution)
    survival_function = lambda x: 1.0 - cdf(x)

    # Integrate the survival function from threshold to infinity
    numerator, _ = quad(survival_function, threshold, np.inf)

    # Integrate the CDF from -infinity to threshold
    denominator, _ = quad(cdf, -np.inf, threshold)

    # Calculate the Omega ratio
    omega = numerator / denominator if denominator != 0 else np.inf

    return omega

def omega_ratio_empirical(threshold, asset):
    """
    Compute the Omega ratio for the given threshold and asset series by empirical calculation.
    """
    # Compute the empirical CDF
    cdf = ECDF(asset)
    survival_function = lambda x: 1.0 - cdf(x)

    # We need a high, odd number of points for Simpson's rule
    num_points = 10_001
    if len(asset) > num_points:
        num_points = len(asset) if len(asset) % 2 == 1 else len(asset) + 1

    # Integrate the survival function from threshold to infinity
    x_values = np.linspace(threshold, max(asset), num_points)
    numerator = simpson(survival_function(x_values), x=x_values)

    # Integrate the CDF from -infinity to threshold
    x_values = np.linspace(min(asset), threshold, num_points)
    denominator = simpson(cdf(x_values), x=x_values)

    # Calculate the Omega ratio
    omega = numerator / denominator if denominator != 0 else np.inf

    return omega

def _optimise_combination(r, r_bar, threshold):
    """
    Returns an array of weights representing an optimal combination of the options. This takes a dataframe
    of data for all assets and a series of expected returns for each asset, along with a periodic return
    threshold.

    This function implements a method of optimising the Omega ratio using linear programming, devised
    by Kapsos et al.
    """
    n_samples, n_options = r.shape

    s: cp.Variable = cp.Variable(shape=(n_options,), name="s")
    q: cp.Variable = cp.Variable(shape=(n_samples,), name="q")
    z: cp.Variable = cp.Variable(name="z")

    w_min, w_max = (0, 1)
    objective = s.T @ r_bar - z * cp.Constant(
        value=threshold
    )

    constraints = (
        [
            q[i] >= threshold * z - s.T @ r.iloc[i, :]
            for i in range(n_samples)
        ]
        + [q[i] >= 0 for i in range(n_samples)]
        + [cp.sum(q) == 1]
        + [cp.sum(s) == z]
        + [(z * w_min) <= s]
        + [s <= (z * w_max)]
        + [s.T @ r_bar >= threshold * z]
        + [z >= 0.0]
    )
    prob = cp.Problem(
        objective=cp.Maximize(objective), constraints=constraints
    )
    prob.solve()

    if prob.status not in {"optimal", "optimal_inaccurate"}:
        raise ValueError(f"Solver status: {prob.status}")

    weights = s.value / s.value.sum()
    weights_dict = {key: weights[i] for i, key in enumerate(r.columns)}
    return weights_dict

def optimise_combination_theoretical(options, threshold, n_samples):
    """
    Optimises the combination of options to maximise the Omega ratio using some number of samples from
    the theoretical distribution. If correctly specified, this can outperform empirical estimation due
    to the increased number of samples.

    Note that more samples will increase solving runtime linearly.
    """
    # Calculate the expected returns
    r_bar = pd.Series({name: stats.t.mean(*distribution) for name, distribution in options.items()})
    # Calculate the expected returns
    r = pd.DataFrame({name: stats.t.rvs(*distribution, size=n_samples) for name, distribution in options.items()})
    return _optimise_combination(r, r_bar, threshold)

def optimise_combination_empirical(data, threshold):
    """
    Optimises the combination of assets to maximise the Omega ratio using empirical data. The optimisation
    process itself depends on samples, which can come either from the theoretical distribution or directly
    from the data itself. This function uses the latter, and all the data at that. For large datasets,
    this can be slow, however it overcomes any issues of incorrect distributional assumptions.
    """
    return _optimise_combination(data, data.mean(), threshold)

def backtest(weights, data):
    """
    Backtests the given weights on the given data, returning a total return multiplier.
    """
    total_returns = 0
    for col in data.columns:
        # Calculate the cumulative returns (reverses what we did in data processing)
        cumulative_returns = (1 + data[col] / 100).prod()

        total_returns += cumulative_returns * weights[col]

    return total_returns

def total_return_to_per_annum(total_return, start_date, end_date):
    """
    Converts a total return over the periods from the start date to the end date into a per-annum return.
    """
    num_days = (end_date - start_date).days
    num_years = round(num_days / 365)
    return (total_return ** (1 / num_years) - 1) * 100

def float_array(string):
    return list(map(float, string.split(',')))

def main():
    """
    Parses arguments and runs the program.
    """
    parser = argparse.ArgumentParser(description="Runs Omega ratio analysis on the provided dataset.")
    # Most of the time, we'll want to generate some kind of analytical report, but sometimes we'll just
    # want to backtest a weighting strategy
    subparsers = parser.add_subparsers(dest='command', required=True)

    analyse_parser = subparsers.add_parser("analyse", help="Generate a report")
    analyse_parser.add_argument("filename", type=str, help="The filename of the dataset to analyse.")
    analyse_parser.add_argument("-o", "--output", type=str, required=True, help="The filename to output the JSON report to.")
    analyse_parser.add_argument("--date-format", type=str, default="%Y-%m-%d",
                        help="The format of the date column in the dataset.")
    analyse_parser.add_argument(
        "-a",
        "--analysis",
        choices=["theoretical", "empirical", "both"],
        default="theoretical",
        help="The type of analysis to run."
    )
    analyse_parser.add_argument("-s", "--samples", type=int, default=10_000, help="The number of samples to use for theoretical optimisation (higher is far better, but will take much longer).")
    analyse_parser.add_argument("-t", "--thresholds", type=float_array, default=[0.0], help="The per-annum thresholds, comma-separated, for the Omega ratio.")
    analyse_parser.add_argument("--signif-level", type=float, default=0.1, help="The significance level for the Kolmogorov-Smirnov test (lower will accept worse fits).")
    analyse_parser.add_argument("-r", "--ratios", action="store_true", default=True, help="Print the Omega ratios for each asset (default).")
    analyse_parser.add_argument("-z", "--optimise", action="store_true", default=False, help="Optimise the combination of assets.")
    analyse_parser.add_argument("-p", "--period", type=int, default=252, help="The number of periods in a year.")

    backtest_parser = subparsers.add_parser("backtest", help="Backtest a weighting strategy")
    backtest_parser.add_argument("filename", type=str, help="The filename of the dataset to analyse.")
    backtest_parser.add_argument("-w", "--weights", type=float_array, required=True, help="The weights to backtest, comma-separated and in the same order as the columns in the dataset.")
    backtest_parser.add_argument("--date-format", type=str, default="%Y-%m-%d",
                        help="The format of the date column in the dataset.")

    report_parser = subparsers.add_parser("report", help="Interpret a report and display")
    report_parser.add_argument("filename", type=str, help="The filename of the report to interpret.")

    args = parser.parse_args()

    if args.command == "analyse":
        data, start_date, end_date = clean_data(args.filename, args.date_format)
        full_report = {}

        options = {}
        ks_p_values = {}
        if args.analysis in {"theoretical", "both"}:
            options, ks_p_values = fit_data(data, args.signif_level)

        for threshold in args.thresholds:
            report = {
                "threshold": 0.0,
                "theoretical": {
                    "params": {}, # Asset name to dict of df, loc, and scale
                    "ks_p_values": {},
                    "ratios": {},
                    "weights": {},
                    "backtest_performance_total": 0.0,
                    "backtest_performance_per_annum": 0.0
                },
                "empirical": {
                    "ratios": {},
                    "weights": {},
                    "backtest_performance_total": 0.0,
                    "backtest_performance_per_annum": 0.0
                },
            }
            # Adjust the threshold (annual) for how many periods there are in a year
            report["threshold"] = threshold / args.period
            if args.analysis in {"theoretical", "both"}:
                report["theoretical"]["params"] = {key: {"df": value[0], "loc": value[1], "scale": value[2]} for key, value in options.items()}
                report["theoretical"]["ks_p_values"] = ks_p_values

            # Compute the Omega ratios if the user wants them
            if args.ratios:
                for col in data.columns:
                    if args.analysis in {"theoretical", "both"}:
                        report["theoretical"]["ratios"][col] = omega_ratio_theoretical(report["threshold"], options[col])
                    if args.analysis in {"empirical", "both"}:
                        report["empirical"]["ratios"][col] = omega_ratio_empirical(report["threshold"], data[col])

            # Optimise the weights to create a portfolio if the user wants
            if args.optimise:
                theoretical_weights = {}
                theoretical_total_perf = 0.0
                theoretical_pa_perf = 0.0
                empirical_weights = {}
                empirical_total_perf = 0.0
                empirical_pa_perf = 0.0

                if args.analysis in {"theoretical", "both"}:
                    theoretical_weights = optimise_combination_theoretical(options, report["threshold"], args.samples)
                    theoretical_total_perf = backtest(theoretical_weights, data)
                    theoretical_pa_perf = total_return_to_per_annum(theoretical_total_perf, start_date, end_date)
                if args.analysis in {"empirical", "both"}:
                    empirical_weights = optimise_combination_empirical(data, report["threshold"])
                    empirical_total_perf = backtest(empirical_weights, data)
                    empirical_pa_perf = total_return_to_per_annum(empirical_total_perf, start_date, end_date)

                # Our optimisation process is good, but floating-point errors can lead to very slightly negative
                # weights, so get rid of those
                report["theoretical"]["weights"] = {key: max(0, value) for key, value in theoretical_weights.items()}
                report["theoretical"]["backtest_performance_total"] = theoretical_total_perf
                report["theoretical"]["backtest_performance_per_annum"] = theoretical_pa_perf

                report["empirical"]["weights"] = {key: max(0, value) for key, value in empirical_weights.items()}
                report["empirical"]["backtest_performance_total"] = empirical_total_perf
                report["empirical"]["backtest_performance_per_annum"] = empirical_pa_perf

            full_report[threshold] = report

        # Save to a JSON file
        with open(args.output, "w") as f:
            json.dump(full_report, f, indent=4)

    elif args.command == "backtest":
        data, start_date, end_date = clean_data(args.filename, args.date_format)
        weights = {col: weight for col, weight in zip(data.columns, args.weights)}
        performance = backtest(weights, data)
        performance_pa = total_return_to_per_annum(performance, start_date, end_date)

        print(f"Backtest performance: {(performance - 1) * 100:.2f}% total ({performance_pa:.2f}% per annum)")
    elif args.command == "report":
        # Load the JSON file and parse it
        with open(args.filename, "r") as f:
            report = json.load(f)

        # Print the report
        print(report)

if __name__ == "__main__":
    main()
