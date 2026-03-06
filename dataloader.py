import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def fetch_data(tickers, interval='1h', period='730d'):
    """
    Fetch hourly data for given tickers from Yahoo Finance.
    
    Args:
        tickers (list): List of ticker symbols (e.g., ['TRUMP35336-USD', 'BTC-USD']).
        interval (str): Data interval (default: '1h').
        period (str): Data period (default: '730d'). Maximum for '1h' in yf is usually 730d.
        
    Returns:
        pd.DataFrame: Combined DataFrame with all assets, cleanly aligned.
    """
    print(f"Fetching data for {len(tickers)} tickers: {tickers}...")
    
    dfs = {}
    for ticker in tickers:
        try:
            print(f"  -> Fetching {ticker}...")
            # We use auto_adjust=True to adjust for splits/dividends (mostly relevant for equities, not crypto)
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            
            if df.empty:
                print(f"  -> WARNING: No data found for {ticker}.")
                continue
                
            # If the index is tz-aware but not UTC, convert it to UTC.
            # If it's tz-naive, localize it to UTC.
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
                
            # Keep only OHLCV to save memory and avoid multi-index confusion on single pulls
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Flatten multi-index columns if yf.download returned them
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
                
            df.columns = [f"{ticker}_{col}" for col in df.columns]
            dfs[ticker] = df
            
        except Exception as e:
            print(f"  -> ERROR fetching {ticker}: {e}")

    if not dfs:
        raise ValueError("Failed to fetch data for all given tickers.")

    print(f"\nMerging datasets...")
    # Merge all dataframes on the Datetime index
    # We use an outer join to capture all available hours, then sort
    combined = pd.concat(dfs.values(), axis=1).sort_index()
    
    # 1. Forward-fill to cover weekends for traditional markets (equities/futures)
    #    We limit ffill to 72 hours to avoid filling across massive data gaps.
    combined = combined.ffill(limit=72)
    # We NO LONGER drop rows based on a single target asset here, to avoid biasing the resulting 
    # datasets for other assets when we split them. We handle dropping missing data per-asset in preprocess_features.
    
    print(f"Fetched {len(combined)} total hourly observations.")
    return combined

def preprocess_features(df, target_prefix='TRUMP35336-USD'):
    """
    Calculate RL state features (momentum, volatility, volume, time encoding).
    
    Args:
        df (pd.DataFrame): Raw concatenated dataframe from fetch_data.
        target_prefix (str): The prefix of the target asset to build primary features on.
        
    Returns:
        pd.DataFrame: DataFrame with newly computed RL state features appended.
    """
    print(f"\nPreprocessing RL state features for target: {target_prefix}...")
    
    df = df.copy()
    close_col = f'{target_prefix}_Close'
    vol_col = f'{target_prefix}_Volume'
    
    if close_col not in df.columns:
        raise ValueError(f"Target close column '{close_col}' not found in dataframe.")
        
    # Drop rows where the specific asset being preprocessed is missing data
    # This prevents the final dataset generated from incorrectly referencing missing periods
    original_len_before_drop = len(df)
    df = df.dropna(subset=[close_col])
    print(f"  -> Dropped {original_len_before_drop - len(df)} rows where {close_col} was missing before features generation.")

    # 1. Price Momentum Features (Log Returns)
    # Log returns are preferred for financial ML as they are symmetric and additive
    print("  -> Calculating Momentum (1h, 4h, 24h log returns)...")
    df['feature_log_return_1h'] = np.log(df[close_col] / df[close_col].shift(1))
    df['feature_log_return_4h'] = np.log(df[close_col] / df[close_col].shift(4))
    df['feature_log_return_24h'] = np.log(df[close_col] / df[close_col].shift(24))

    # 2. Volatility Features
    # 24-hour rolling standard deviation of 1-hour log returns
    print("  -> Calculating Volatility (24h rolling std of 1h returns)...")
    df['feature_volatility_24h'] = df['feature_log_return_1h'].rolling(window=24).std()

    # 3. Volume Features
    # Log transform volume. We add 1 to avoid log(0) for zero-volume hours.
    print("  -> Calculating Volume features (Log Volume)...")
    if vol_col in df.columns:
        df['feature_log_volume'] = np.log1p(df[vol_col])

    # 4. Intraday Timing (Cyclical Encoding)
    # Using sine/cosine preserves the cyclical nature (e.g., hour 23 is close to hour 0)
    print("  -> Calculating Time features (Cyclical encoding)...")
    
    # Extract UTC Hour (0-23)
    hours = df.index.hour
    df['feature_hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
    df['feature_hour_cos'] = np.cos(2 * np.pi * hours / 24.0)
    
    # Extract Day of Week (0=Monday, 6=Sunday)
    days = df.index.dayofweek
    df['feature_day_sin'] = np.sin(2 * np.pi * days / 7.0)
    df['feature_day_cos'] = np.cos(2 * np.pi * days / 7.0)

    # 5. Clean up NaNs created by rolling windows/shifts
    # We lose the first 24 hours of data due to the 24h shifts, which is acceptable
    # out of ~17,500 rows.
    original_len = len(df)
    features_to_check = [col for col in df.columns if col.startswith('feature_')]
    df = df.dropna(subset=features_to_check)
    print(f"Dropped {original_len - len(df)} warm-up rows (due to NaNs from 24h lag features).")
    
    print(f"Final preprocessed shape: {df.shape}")
    return df

def main():
    # Define assets to pull
    target_ticker = 'TRUMP35336-USD'
    tickers = [
        target_ticker,
        'BTC-USD',  # Major crypto benchmark
        'ETH-USD',  # Major crypto benchmark
        'GC=F',     # Gold futures
        'SI=F',     # Silver futures
        'ES=F'      # S&P 500 futures
    ]
    
    # Base directories
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    RAW_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
    
    # Ensure directories exist
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    try:
        # Step 1: Fetch Data
        # max period in yf for 1h is 730d (approx 2 years)
        raw_df = fetch_data(tickers, interval='1h', period='730d')
        
        # Save raw pull as backup (optional but good practice)
        raw_path = os.path.join(RAW_DIR, 'raw_hourly_pull.csv')
        raw_df.to_csv(raw_path)
        print(f"\nSaved raw data backup to: {raw_path}")
        
        # Step 2: Preprocess RL Features — TRUMP
        processed_df = preprocess_features(raw_df, target_prefix=target_ticker)
        
        # Step 3: Save TRUMP Processed Output
        output_file = os.path.join(PROCESSED_DIR, 'rl_dataset_hourly.csv')
        processed_df.to_csv(output_file)
        
        print(f"\n==============================================")
        print(f"SUCCESS! TRUMP dataset saved to:\n{output_file}")
        print(f"Total Rows: {len(processed_df)}")
        print(f"Date Range: {processed_df.index.min()} to {processed_df.index.max()}")
        print(f"==============================================")

        # Step 4: Preprocess RL Features — BTC
        btc_df = preprocess_features(raw_df, target_prefix='BTC-USD')

        btc_output = os.path.join(PROCESSED_DIR, 'rl_dataset_btc_hourly.csv')
        btc_df.to_csv(btc_output)

        print(f"\n==============================================")
        print(f"SUCCESS! BTC dataset saved to:\n{btc_output}")
        print(f"Total Rows: {len(btc_df)}")
        print(f"Date Range: {btc_df.index.min()} to {btc_df.index.max()}")
        print(f"==============================================")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR in pipeline: {e}")

if __name__ == "__main__":
    main()
