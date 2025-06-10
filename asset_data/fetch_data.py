import yfinance as yf
import pandas as pd

# tickers = ['GOOG', 'AAPL', 'TSLA', 'MSFT', 'AMZN', 'GE', 'TLT', 'USO', 'GLD']
# tickers = [
#     "AAPL", "MSFT", "AMZN", "GOOG", "META", "TSLA", "NVDA", "BRK-B", "JPM", "V",
#     "UNH", "JNJ", "WMT", "PG", "MA", "HD", "VZ", "DIS", "PFE", "BAC",
#     "XOM", "CVX", "KO", "NKE", "CRM", "ORCL", "NFLX", "ABT", "TSM", "INTC",
#     "SPY", "QQQ", "DIA", "IWM", "EEM", "EFA", "VWO", "XLE", "XLK", "XLF",
#     "XLY", "XLI", "XLB", "XLV", "XLRE", "XLU", "AGG", "LQD", "HYG", "TLT",
#     "USO", "GLD"
# ]
tickers = [
    "AAPL", "MSFT", "AMZN", "GOOG", "NVDA", "BRK-B", "JPM", "V",
    "UNH", "JNJ", "WMT", "PG", "MA", "HD", "VZ", "DIS",
    "PFE", "BAC", "XOM", "CVX", "KO", "NKE", "CRM", "ORCL",
    "NFLX", "ABT", "TSM", "INTC",
    "SPY", "QQQ", "DIA", "IWM", "EEM", "EFA", "VWO",
    "XLE", "XLK", "XLF", "XLY", "XLI", "XLB", "XLV", "XLU",
    "AGG", "LQD", "HYG", "TLT", "USO", "GLD"
]
data = yf.download(tickers, interval="1d")
latest_earliest_date = data.dropna().index[0]
aligned_data = data.loc[latest_earliest_date:]

daily_returns = aligned_data['Close'].pct_change().dropna()
daily_returns.to_csv("daily_returns.csv", index=True)
print(f"Earliest date: {latest_earliest_date}")
