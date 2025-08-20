# predict.py
# This script loads a trained LSTM model and a scaler for a given company ticker,
# uses it to predict the next day's stock closing price, and saves the prediction.

import pandas as pd
import numpy as np
import os
import sys
import joblib
from tensorflow.keras.models import load_model
from transformers import pipeline
import warnings
from datetime import date

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
# File path for the trained models and scalers
models_dir = 'models/'
# The number of days used as input for the model
sequence_length = 30
# Preprocessed data directory
processed_data_dir = 'data/processed/'
input_file = os.path.join(processed_data_dir, "combined_features.csv")
# Directory to save predictions
predictions_dir = 'predictions/'
predictions_file = os.path.join(predictions_dir, "predictions.csv")


# Load FinBERT sentiment model (PyTorch backend)
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", framework="pt")

def get_sentiment(text):
    """
    Analyzes sentiment of a given text using a pre-trained model.
    """
    try:
        result = sentiment_pipeline(text[:512])[0]
        # FinBERT provides scores for positive, negative, and neutral. We use a score.
        if result['label'].lower() == 'positive':
            return result['score']
        elif result['label'].lower() == 'negative':
            return -result['score']
        else:
            return 0.0
    except Exception as e:
        # In case of an error (e.g., empty text), return a neutral sentiment
        return 0.0

def get_latest_data(ticker, num_days=sequence_length):
    """
    Mocks fetching the latest stock and news data. In a real-world
    application, this function would call an API (e.g., Alpha Vantage, Finnhub).
    For this example, we'll use the last 'num_days' of data from our
    preprocessed CSV file.
    """
    try:
        df = pd.read_csv(input_file)
        company_df = df[df['ticker'] == ticker].copy()
        
        if len(company_df) < num_days:
            raise ValueError(f"Not enough data for {ticker} in the dataset to make a prediction.")

        # Get the latest `num_days` rows
        latest_data = company_df.tail(num_days)
        return latest_data

    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found. Please run preprocessor.py first.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while fetching data: {e}")
        sys.exit(1)

def save_prediction(ticker, predicted_price):
    """
    Saves the predicted price to a CSV file.
    The file will be created if it doesn't exist, and new data will be appended.
    """
    # Create the predictions directory if it doesn't exist
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    # Create a DataFrame for the new prediction
    today = date.today()
    prediction_data = pd.DataFrame([{
        'Date': today.strftime("%Y-%m-%d"),
        'Ticker': ticker,
        'Predicted_Close_Price': predicted_price
    }])

    # Check if the file already exists
    file_exists = os.path.isfile(predictions_file)

    # Save the data to the CSV file
    prediction_data.to_csv(
        predictions_file,
        mode='a', # 'a' for append mode
        header=not file_exists, # Write header only if the file does not exist
        index=False # Do not write the DataFrame index
    )
    print(f"✅ Prediction saved to {predictions_file}")

def main():
    """
    Main function to handle the prediction process.
    """
    # Ask the user for a ticker
    ticker = input("Please enter a stock ticker (e.g., AAPL): ").upper()

    # Define file paths for the specific model and scaler
    model_path = os.path.join(models_dir, f'stock_price_lstm_model_{ticker}.h5')
    scaler_path = os.path.join(models_dir, f'stock_price_scaler_{ticker}.joblib')

    # Check if the model and scaler files exist
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Error: No model found for {ticker}.")
        print("Please ensure the 'train_model.py' script was run successfully.")
        sys.exit(1)

    print(f"\n--- Making prediction for {ticker} ---")

    # Load the trained model and scaler
    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
    except Exception as e:
        print(f"Error loading model or scaler for {ticker}: {e}")
        sys.exit(1)

    # Get the latest data for the prediction
    latest_data = get_latest_data(ticker)
    
    # Select features for scaling, ensuring they match the training data
    # The 'date' column must be excluded from features used for scaling.
    feature_cols = [col for col in latest_data.columns if col not in ['date', 'ticker', 'target_close_1d', 'target_close_2d', 'target_close_3d', 'target_close_4d', 'target_close_5d']]
    
    # Reshape the data to match the model's input shape
    last_30_days = latest_data[feature_cols].values
    last_30_days_scaled = scaler.transform(last_30_days)
    X_predict = np.reshape(last_30_days_scaled, (1, last_30_days_scaled.shape[0], last_30_days_scaled.shape[1]))
    
    # Make the prediction
    predicted_price_scaled = model.predict(X_predict, verbose=0)
    
    # Inverse transform the prediction to get the actual price
    # Create a dummy array to inverse transform just the 'Close' column
    dummy_array = np.zeros(shape=(1, len(feature_cols)))
    # Assuming 'Close' is the first feature in feature_cols after dropping the non-features
    close_col_index = feature_cols.index('Close')
    dummy_array[0, close_col_index] = predicted_price_scaled[0, 0]
    
    predicted_price = scaler.inverse_transform(dummy_array)[:, close_col_index][0]

    print(f"\n✅ Predicted closing price for the next day for {ticker}: ${predicted_price:.2f}")


if __name__ == "__main__":
    main()
