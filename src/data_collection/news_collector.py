# news_collector.py

import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import warnings
from dotenv import load_dotenv 


import config

warnings.filterwarnings('ignore')


load_dotenv()


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


os.makedirs(output_dir, exist_ok=True)


for symbol in symbols:
    
    file_path = os.path.join(output_dir, f"{symbol}_news.csv")

    
    if os.path.exists(file_path):
        print(f"{file_path} already exists. Skipping download.")
        continue  

    print(f"Downloading news for {symbol}...")
    
    
    all_data = []
    
    start_date = datetime.strptime(config.START_DATE, "%Y-%m-%d")
    end_date = datetime.strptime(config.END_DATE, "%Y-%m-%d")
    
    while start_date < end_date:
        range_start = start_date.strftime("%Y-%m-%d")
        range_end = (start_date + timedelta(days=30)).strftime("%Y-%m-%d")
        
        
        if datetime.strptime(range_end, "%Y-%m-%d") > end_date:
            range_end = end_date.strftime("%Y-%m-%d")

        print(f"Fetching {range_start} → {range_end} for {symbol}")
        
        data = get_news(symbol, range_start, range_end)
        all_data.extend(data)  
        
        start_date += timedelta(days=30)  

    if not all_data:
        print(f"No data found for {symbol}. Skipping file creation.")
        continue

    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Remove rows where datetime is missing or 0
    df = df[df['datetime'].notnull() & (df['datetime'] > 0)]

  
    df['date'] = pd.to_datetime(df['datetime'], unit='s').dt.date
    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    print(f"CSV file saved as {file_path}")

print("done")
