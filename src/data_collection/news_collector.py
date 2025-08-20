# news_collector.py

import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import warnings
from dotenv import load_dotenv # Import load_dotenv

# Import configuration from config.py
import config

warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

# Use variables from config.py
API_KEY = os.getenv("FINNHUB_API_KEY")
symbols = config.TICKERS
output_dir = config.OUTPUT_DIR

# Function to fetch news for a date range
def get_news(symbol, start, end):
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={start}&to={end}&token={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error for {symbol}: {response.status_code} - {response.text}")
        return []

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through each symbol in the list
for symbol in symbols:
    # Use the output directory from config for the file path
    file_path = os.path.join(output_dir, f"{symbol}_news.csv")

    # Check if the CSV file already exists
    if os.path.exists(file_path):
        print(f"✅ {file_path} already exists. Skipping download.")
        continue  # Skip the rest of the loop for this symbol and move to the next

    print(f"🚀 Downloading news for {symbol}...")
    
    # Loop month-by-month (~30-day chunks)
    all_data = []
    # Use the start and end dates from config.py
    start_date = datetime.strptime(config.START_DATE, "%Y-%m-%d")
    end_date = datetime.strptime(config.END_DATE, "%Y-%m-%d")
    
    while start_date < end_date:
        range_start = start_date.strftime("%Y-%m-%d")
        range_end = (start_date + timedelta(days=30)).strftime("%Y-%m-%d")
        
        # Check to ensure range_end doesn't exceed end_date for the last chunk
        if datetime.strptime(range_end, "%Y-%m-%d") > end_date:
            range_end = end_date.strftime("%Y-%m-%d")

        print(f"Fetching {range_start} → {range_end} for {symbol}")
        
        data = get_news(symbol, range_start, range_end)
        all_data.extend(data)  # add results to list
        
        start_date += timedelta(days=30)  # move to next chunk

    if not all_data:
        print(f"⚠️ No data found for {symbol}. Skipping file creation.")
        continue

    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Remove rows where datetime is missing or 0
    df = df[df['datetime'].notnull() & (df['datetime'] > 0)]

    # Now safely convert
    df['date'] = pd.to_datetime(df['datetime'], unit='s').dt.date
    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    print(f"✅ CSV file saved as {file_path}")

print("✨ All done!")
