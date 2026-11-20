#!/usr/bin/env python
# Simple script that cleans data downloaded from Unisuper's public returns table. This
# converts everything into a table of dates against returns for each asset, from the time
# when all assets started trading.

import pandas as pd
import sys

def clean_data(filename, drop_cols):
    """
    1) Reads a Unisuper CSV
    2) Drops the final 3 metadata cols + drop_cols
    3) Converts cumulative % returns → daily % returns
    4) Removes the first NaN row and any row where ALL assets == 0.0
    5) Returns a Date + daily-returns DataFrame

    Note: the final daily returns are already as percentage changes, so
    0.1 indicates a 0.1% change, NOT a 10% change!
    """
    # 1) read and parse dates
    df = pd.read_csv(filename, parse_dates=["Date"], dayfirst=True)
    # 2a) drop the last 3 columns (whatever they are)
    if df.shape[1] >= 3:
        df.drop(columns=df.columns[-3:], inplace=True)
    # 2b) keep Date aside, drop the rest of your unwanted cols
    dates = df["Date"]
    df.drop(columns=["Date", *drop_cols], inplace=True, errors="ignore")

    # 3) cumulative % → multiplier → daily %
    mult = df.div(100.0).add(1.0)       # (cum% / 100 + 1)
    daily = mult.div(mult.shift(1)).sub(1.0)

    # 4a) lose that first NaN row
    daily = daily.iloc[1:].reset_index(drop=True)
    dates = dates.iloc[1:].reset_index(drop=True)

    # 4b) drop rows where *all* assets are exactly zero
    mask = daily.ne(0.0).any(axis=1)   # keep rows where any asset ≠ 0
    daily = daily.loc[mask].reset_index(drop=True)
    dates = dates.loc[mask].reset_index(drop=True)

    # 5) glue back together
    result = pd.concat([dates, daily], axis=1)
    return result

data = clean_data(sys.argv[1], ["Cash", "Conservative", "Conservative Balanced", "Balanced", "Sustainable Balanced", "Growth", "Sustainable High Growth", "High Growth"])
# data = clean_data(sys.argv[1], [])
data.to_csv(sys.argv[2], index=False)
