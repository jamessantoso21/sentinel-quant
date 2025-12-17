"""
Sentinel Quant - Data Downloader
Downloads historical BTC data using yfinance (no geo-restrictions)
"""
import pandas as pd
from datetime import datetime, timedelta
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
OUTPUT_FILE = "btc_historical.csv"


def download_historical_data():
    """Download BTC historical data using yfinance"""
    try:
        import yfinance as yf
    except ImportError:
        logger.info("Installing yfinance...")
        import subprocess
        subprocess.check_call(["python", "-m", "pip", "install", "yfinance"])
        import yfinance as yf
    
    logger.info("Downloading BTC-USD data from Yahoo Finance...")
    
    # Download 2 years of hourly data (max for yfinance)
    ticker = yf.Ticker("BTC-USD")
    
    # Get max available data - 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    df = ticker.history(start=start_date, end=end_date, interval="1h")
    
    if df.empty:
        logger.error("No data received from Yahoo Finance")
        return None
    
    logger.info(f"Downloaded {len(df)} hourly candles")
    
    # Rename columns to match expected format
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    
    if 'datetime' in df.columns:
        df = df.rename(columns={'datetime': 'timestamp'})
    elif 'date' in df.columns:
        df = df.rename(columns={'date': 'timestamp'})
    
    # Keep needed columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), OUTPUT_FILE)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Saved to {output_path}")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"Total rows: {len(df)}")
    
    return df


if __name__ == "__main__":
    download_historical_data()
