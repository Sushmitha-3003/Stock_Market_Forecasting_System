import pandas as pd
import os
import sys
import ta


# Get the path to the src/data_collection directory
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_collection'))
config_file_path = os.path.join(config_path, 'config.py')

print(f"Checking for config.py at: {config_file_path}")

# Check if the config file exists  
if not os.path.exists(config_file_path):
    print("Error: The config.py file was not found in the expected location.")
    print(f"Please ensure your config file is located at: {config_file_path}")
    sys.exit(1)

# Add the src/data_collection directory to the system path and import the config module.
sys.path.append(config_path)
import config

print(f"Successfully loaded config from: {config_file_path}")


try:
    import ta
except ImportError:
    print("Error: The 'ta' library is not installed. Please install it using 'pip install ta'.")
    sys.exit(1)


def add_technical_indicators(df):
    """
    Calculates and adds common technical indicators to the stock DataFrame using the 'ta' library.
    """
    # Simple Moving Average (SMA)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)

    # Exponential Moving Average (EMA)
    df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)

    # Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

    # Moving Average Convergence Divergence (MACD)
    macd = ta.trend.MACD(df['Close'], window_fast=12, window_slow=26, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist'] = macd.macd_diff()

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'], window=20)
    df['Bollinger_Upper'] = bollinger.bollinger_hband()
    df['Bollinger_Middle'] = bollinger.bollinger_mavg()
    df['Bollinger_Lower'] = bollinger.bollinger_lband()
    
    # Drop rows with NaN values that result from indicator calculations
    df.dropna(inplace=True)
    return df


def run_feature_engineering():
    """
    Main function to orchestrate the feature engineering process.
    """
    print("Starting feature engineering process...")
    
    
    if not os.path.exists(config.ENHANCED_DATA_DIR):
        os.makedirs(config.ENHANCED_DATA_DIR)
        print(f"Created directory: {config.ENHANCED_DATA_DIR}")

    # Process each ticker's data
    for ticker in config.TICKERS:
        try:
            raw_data_path = os.path.join(config.OUTPUT_DIR, f"{ticker}_stock_data.csv")
            enhanced_output_path = os.path.join(config.ENHANCED_DATA_DIR, f"{ticker}_enhanced_data.csv")
            
            if os.path.exists(enhanced_output_path):
                print(f"Enhanced data for {ticker} already exists. Skipping.")
                continue

            print(f"Loading raw stock data for {ticker} from {raw_data_path}")
            df = pd.read_csv(raw_data_path)
            
            # Convert Date column to datetime objects
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # Add technical indicators
            print(f"Adding technical indicators for {ticker} using the 'ta' library...")
            enhanced_df = add_technical_indicators(df.copy())
            
            # Save the enhanced data
            enhanced_df.to_csv(enhanced_output_path)
            print(f"Enhanced data for {ticker} saved to {enhanced_output_path}")

        except FileNotFoundError:
            print(f"Raw data file not found for {ticker} at {raw_data_path}. Please check your data directory.")
        except Exception as e:
            print(f"An error occurred while processing {ticker}: {e}")

    print("Feature engineering complete.")

if __name__ == "__main__":
    run_feature_engineering()
