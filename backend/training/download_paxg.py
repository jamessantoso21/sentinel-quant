"""
Sentinel Quant - PAXG Historical Data Download
Downloads Pax Gold (PAXG) historical data for backtesting
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def download_paxg_data():
    """Download PAXG historical data using yfinance"""
    
    print("Downloading PAXG (Pax Gold) historical data...")
    
    # PAXG-USD ticker on Yahoo Finance
    ticker = "PAXG-USD"
    
    # Download 2 years of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years
    
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    
    try:
        # Create ticker object
        paxg = yf.Ticker(ticker)
        
        # Download hourly data
        df = paxg.history(start=start_date, end=end_date, interval="1h")
        
        if df.empty:
            print("No hourly data. Trying daily data...")
            df = paxg.history(start=start_date, end=end_date, interval="1d")
        
        if df.empty:
            print("ERROR: Could not download PAXG data")
            return None
        
        # Reset index to get timestamp as column
        df = df.reset_index()
        
        # Rename columns to lowercase
        df.columns = [c.lower() if isinstance(c, str) else str(c).lower() for c in df.columns]
        
        # Rename datetime/date column to timestamp
        if 'datetime' in df.columns:
            df = df.rename(columns={'datetime': 'timestamp'})
        elif 'date' in df.columns:
            df = df.rename(columns={'date': 'timestamp'})
        
        # Ensure required columns exist
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        available_cols = [c for c in required_cols if c in df.columns]
        df = df[available_cols]
        
        # Remove any rows with NaN
        df = df.dropna()
        
        print(f"\nDownloaded {len(df)} PAXG candles")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        # Save to CSV
        output_path = os.path.join(os.path.dirname(__file__), "paxg_historical.csv")
        df.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")
        
        return df
        
    except Exception as e:
        print(f"Error downloading PAXG data: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    download_paxg_data()
