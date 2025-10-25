import pandas as pd
import os
import sys
from transformers import pipeline


script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..')
if src_dir not in sys.path:
    sys.path.append(src_dir)

try:
    from data_collection import config
except ImportError as e:
    print(f"Error importing config: {e}")
    sys.exit(1)

# Load FinBERT model 
print("Loading FinBERT sentiment model...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert",
    framework="pt"
)
print("FinBERT model loaded.")

def compute_sentiment_from_news(ticker):
    """
    Reads {ticker}_news.csv from raw data dir, runs FinBERT sentiment,
    returns DataFrame with date and sentiment_score.
    """
    news_path = os.path.join(config.OUTPUT_DIR, f"{ticker}_news.csv")
    if not os.path.exists(news_path):
        print(f"News file not found for {ticker}: {news_path}")
        return pd.DataFrame(columns=['date', 'sentiment_score'])

    print(f"Loading news for {ticker} from {news_path}")
    news_df = pd.read_csv(news_path)

    # Combine headline + summary
    news_df['text'] = news_df.get('headline', '').fillna('') + '. ' + news_df.get('summary', '').fillna('')

    # Apply sentiment
    def get_sentiment(text):
        try:
            result = sentiment_pipeline(text[:512])[0]
            label = result['label'].lower()
            score = float(result['score'])
            return score if label == 'positive' else -score if label == 'negative' else 0.0
        except Exception:
            return 0.0

    news_df['sentiment_score'] = news_df['text'].apply(get_sentiment)

    # Convert date column
    if 'datetime' in news_df.columns:
        news_df['date'] = pd.to_datetime(news_df['datetime'], unit='s').dt.date
    elif 'publishedAt' in news_df.columns:
        news_df['date'] = pd.to_datetime(news_df['publishedAt']).dt.date
    elif 'date' in news_df.columns:
        news_df['date'] = pd.to_datetime(news_df['date']).dt.date
    else:
        raise ValueError(f"No valid date column in news file for {ticker}")

    # Average sentiment per date
    daily_sentiment = news_df.groupby('date', as_index=False)['sentiment_score'].mean()

    return daily_sentiment

def preprocess_data():
    print("Starting data preprocessing process...")

    if not os.path.exists(config.PROCESSED_DATA_DIR):
        os.makedirs(config.PROCESSED_DATA_DIR)
        print(f"Created directory: {config.PROCESSED_DATA_DIR}")

    combined_df = pd.DataFrame()

    for ticker in config.TICKERS:
        try:
            enhanced_path = os.path.join(config.ENHANCED_DATA_DIR, f"{ticker}_enhanced_data.csv")
            print(f"Loading enhanced data for {ticker} from {enhanced_path}")
            df = pd.read_csv(enhanced_path)

            df['Date'] = pd.to_datetime(df['Date'])
            df.rename(columns={'Date': 'date'}, inplace = True)
            df.set_index('date', inplace = True)

            # Get daily sentiment scores
            sentiment_df = compute_sentiment_from_news(ticker)
            sentiment_df.set_index('date', inplace = True)

            # Merge with stock data
            merged_df = df.merge(sentiment_df, how='left', left_index=True, right_index = True)

            # Fill sentiment gaps
            merged_df['sentiment_score'].fillna(method='ffill', inplace = True)
            merged_df['sentiment_score'].fillna(0, inplace = True)

            # Target variable: price 5 days ahead
            merged_df['target'] = merged_df['Close'].shift(-5)
            merged_df.dropna(inplace = True)
            merged_df['ticker'] = ticker

            combined_df = pd.concat([combined_df, merged_df])
            print(f"Finished processing {ticker}")

        except FileNotFoundError:
            print(f"Enhanced data file not found for {ticker}")
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    if not combined_df.empty:
        output_path = os.path.join(config.PROCESSED_DATA_DIR, "combined_features.csv")
        combined_df.to_csv(output_path, index=True)
        print(f"Saved preprocessed data with sentiment to {output_path}")
    else:
        print("No data processed.")

if __name__ == "__main__":
    preprocess_data()
