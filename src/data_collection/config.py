# config.py
# This file centralizes all configuration variables for the project.

import datetime

# Define the list of company stock tickers to download
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Define the date range for the data
START_DATE = '2020-01-01'
# Set END_DATE to today's date in YYYY-MM-DD format
END_DATE = datetime.date.today().strftime('%Y-%m-%d')

# --- Directory Configuration ---
# Define the directory where the raw data will be saved
OUTPUT_DIR = 'data/raw/'

# Define the directory for the stock data with technical indicators
ENHANCED_DATA_DIR = 'data/enhanced/'

# Define the directory for the final, combined, and preprocessed data
PROCESSED_DATA_DIR = 'data/processed/'

