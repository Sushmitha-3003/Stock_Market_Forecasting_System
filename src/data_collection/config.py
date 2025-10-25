
import datetime

# Define the list of company stock tickers to download
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']


START_DATE = '2020-01-01'

END_DATE = datetime.date.today().strftime('%Y-%m-%d')

OUTPUT_DIR = 'data/raw/'

ENHANCED_DATA_DIR = 'data/enhanced/'

PROCESSED_DATA_DIR = 'data/processed/'

