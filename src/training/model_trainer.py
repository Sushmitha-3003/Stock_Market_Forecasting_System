import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib


# File path for the preprocessed data
processed_data_dir = 'data/processed/'
input_file = os.path.join(processed_data_dir, "combined_features.csv")
sequence_length = 60  # Increased sequence length for a longer look-back period
models_dir = 'models/'
os.makedirs(models_dir, exist_ok=True)

# --- Load and Prepare Data ---
try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    print(f"Error: The file {input_file} was not found.")
    sys.exit(1)

df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df.sort_index(inplace=True)

# Separate data for each company and train a separate model
tickers = df['ticker'].unique()

for ticker in tickers:
    print(f"Training model for {ticker}...")
    company_df = df[df['ticker'] == ticker].copy()
    
    # --- Assigning the explicit list of 17 features ---
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26',
        'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
        'Bollinger_Upper', 'Bollinger_Middle', 'Bollinger_Lower',
        'sentiment_score'
    ]
    
    # Check if all required feature columns exist in the DataFrame
    if not all(col in company_df.columns for col in feature_cols):
        missing_cols = [col for col in feature_cols if col not in company_df.columns]
        print(f"Skipping {ticker} due to missing feature columns: {missing_cols}")
        continue
    
    # Check if there is enough data
    if len(company_df) < sequence_length:
        print(f"Skipping {ticker} due to insufficient data for the chosen sequence length.")
        continue

    data = company_df[feature_cols].values
    
    # --- Scaling Data ---
    # We fit a new scaler for each company's data to ensure the scaling is accurate
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # --- Create Sequences for LSTM ---
    X, y = [], []
    # We will predict the next day's closing price
    target_col_index = feature_cols.index('Close')
    
    for i in range(len(scaled_data) - sequence_length - 1):
        X.append(scaled_data[i:(i + sequence_length), :])
        # The target is the 'Close' price of the next day
        y.append(scaled_data[i + sequence_length, target_col_index])

    X, y = np.array(X), np.array(y)

    # --- Splitting the Data ---
    # We use a time-series split to avoid data leakage
    split_ratio = 0.8
    split_index = int(split_ratio * len(X))

    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]

    # --- LSTM Model for Regression (slightly more complex) ---
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    # --- Train the Model with Callbacks ---
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # --- Evaluate the Model ---
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n--- Model Evaluation for {ticker} ---")
    print(f"Test Loss (MSE): {loss:.4f}\n")

    # --- Save the Model and Scaler ---
    model_path = os.path.join(models_dir, f'stock_price_lstm_model_{ticker}.h5')
    scaler_path = os.path.join(models_dir, f'stock_price_scaler_{ticker}.joblib')

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Model and scaler for {ticker} saved successfully.\n")

print("All models have been trained and saved.")
