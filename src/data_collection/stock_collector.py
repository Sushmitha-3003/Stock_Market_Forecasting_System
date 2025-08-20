import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings
import os
import config

warnings.filterwarnings('ignore')

# Use the variables from the config file
tickers = config.TICKERS
start_date = config.START_DATE
end_date = config.END_DATE
output_dir = config.OUTPUT_DIR

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through each ticker to check for existing data and download if needed
for ticker in tickers:
    file_path = os.path.join(output_dir, f"{ticker}_stock_data.csv")
    
    # Check if the file already exists
    if os.path.exists(file_path):
        print(f"Data for {ticker} already exists at {file_path}. Skipping download.")
        continue
    
    # If the file does not exist, download the data
    print(f"Downloading data for {ticker}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        
        if data.empty:
            print(f"No data found for {ticker}. Skipping.")
            continue
            
        # Reset the index to make 'Date' a regular column
        data.reset_index(inplace=True)
        
        # Drop the 'Adj Close' column if it exists
        if 'Adj Close' in data.columns:
            data = data.drop(columns=['Adj Close'])
        
        # Force a clean header by assigning a new list of column names.
        # This is a very reliable way to remove any hidden metadata.
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Add the ticker column
        data['ticker'] = ticker
        
        # Save to CSV without writing the index
        data.to_csv(file_path, index=False)
        
        print(f"Data for {ticker} saved to {file_path}\n")

    except Exception as e:
        print(f"An error occurred while downloading data for {ticker}: {e}\n")

print("Stock data collection process complete.")
