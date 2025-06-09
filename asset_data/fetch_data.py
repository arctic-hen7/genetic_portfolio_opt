import yfinance as yf
import pandas as pd

tickers = ['GOOG', 'AAPL', 'TSLA', 'MSFT', 'AMZN', 'GE', 'TLT', 'USO', 'GLD']
data = yf.download(tickers, interval="1d")
latest_earliest_date = data.dropna().index[0]
aligned_data = data.loc[latest_earliest_date:]

daily_returns = aligned_data['Close'].pct_change().dropna()
daily_returns.to_csv("daily_returns.csv", index=True)
print(f"Earliest date: {latest_earliest_date}")
