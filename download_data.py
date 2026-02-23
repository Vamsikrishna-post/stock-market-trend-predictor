import yfinance as yf
import pandas as pd
import os

def download_stock_data(ticker="AAPL", start="2020-01-01", end="2024-01-01", filename="stock_data.csv"):
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, start=start, end=end)
    
    if data.empty:
        print("No data found.")
        return
    
    # Save to CSV
    data.to_csv(filename)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    download_stock_data()
