import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings
import os
import config

warnings.filterwarnings('ignore')


tickers = config.TICKERS
start_date = config.START_DATE
end_date = config.END_DATE
output_dir = config.OUTPUT_DIR


os.makedirs(output_dir, exist_ok=True)


for ticker in tickers:
    file_path = os.path.join(output_dir, f"{ticker}_stock_data.csv")
    
    
    if os.path.exists(file_path):
        print(f"Data for {ticker} already exists at {file_path}. Skipping download.")
        continue
    
    
    print(f"Downloading data for {ticker}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        
        if data.empty:
            print(f"No data found for {ticker}. Skipping.")
            continue
            
        
        data.reset_index(inplace=True)
        
        
        if 'Adj Close' in data.columns:
            data = data.drop(columns=['Adj Close'])
        

        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Add the ticker column
        data['ticker'] = ticker
        
        # Save to CSV 
        data.to_csv(file_path, index=False)
        
        print(f"Data for {ticker} saved to {file_path}\n")

    except Exception as e:
        print(f"An error occurred while downloading data for {ticker}: {e}\n")

print("Stock data collection process complete.")
