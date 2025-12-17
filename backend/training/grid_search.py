"""
Sentinel Quant - Grid Search Optimizer
Comprehensive parameter optimization with more combinations
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import itertools

# Configuration
INITIAL_CAPITAL = 10000

# Parameter ranges to test
CONSENSUS_RANGE = [0.25, 0.50, 0.75, 1.0]  # 1/4, 2/4, 3/4, 4/4
STOP_LOSS_RANGE = [0.01, 0.015, 0.02, 0.025, 0.03]  # 1% to 3%
TAKE_PROFIT_RANGE = [0.02, 0.03, 0.04, 0.05, 0.06]  # 2% to 6%


class FastBacktester:
    """Optimized backtester for grid search"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.precompute_indicators()
    
    def precompute_indicators(self):
        """Precompute all indicators once"""
        # RSI
        delta = self.df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        self.df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        sma20 = self.df['close'].rolling(20).mean()
        std20 = self.df['close'].rolling(20).std()
        self.df['bb_upper'] = sma20 + 2 * std20
        self.df['bb_lower'] = sma20 - 2 * std20
        self.df['bb_position'] = (self.df['close'] - self.df['bb_lower']) / (self.df['bb_upper'] - self.df['bb_lower'])
        
        # SMAs for trend
        self.df['sma20'] = sma20
        self.df['sma50'] = self.df['close'].rolling(50).mean()
        
        # Price changes for momentum
        self.df['pct_change_5'] = self.df['close'].pct_change(5)
    
    def get_votes(self, idx: int) -> dict:
        """Get all votes at index"""
        row = self.df.iloc[idx]
        rsi = row['rsi']
        bb_pos = row['bb_position']
        sma20 = row['sma20']
        sma50 = row['sma50']
        price = row['close']
        pct_change = row['pct_change_5']
        
        votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        # Technical voter
        if rsi < 30 or (rsi < 40 and bb_pos < 0.2):
            votes['BUY'] += 1
        elif rsi > 70 or (rsi > 60 and bb_pos > 0.8):
            votes['SELL'] += 1
        else:
            votes['HOLD'] += 1
        
        # Momentum voter
        if rsi < 30:
            votes['BUY'] += 1
        elif rsi > 70:
            votes['SELL'] += 1
        elif rsi < 40:
            votes['BUY'] += 1
        elif rsi > 60:
            votes['SELL'] += 1
        else:
            votes['HOLD'] += 1
        
        # Trend voter
        if pd.notna(sma50):
            if sma20 > sma50 and price > sma20:
                votes['BUY'] += 1
            elif sma20 < sma50 and price < sma20:
                votes['SELL'] += 1
            else:
                votes['HOLD'] += 1
        else:
            votes['HOLD'] += 1
        
        # Sentiment voter (simulated)
        if pd.notna(pct_change):
            np.random.seed(int(price * 100) % 2**31)
            noise = np.random.uniform(-0.015, 0.015)
            sentiment = pct_change + noise
            if sentiment > 0.01:
                votes['BUY'] += 1
            elif sentiment < -0.01:
                votes['SELL'] += 1
            else:
                votes['HOLD'] += 1
        else:
            votes['HOLD'] += 1
        
        return votes
    
    def run(self, consensus: float, stop_loss: float, take_profit: float) -> dict:
        """Run backtest with given parameters"""
        capital = INITIAL_CAPITAL
        position = None
        entry_price = 0
        trades = []
        
        required_votes = int(4 * consensus)
        
        for idx in range(50, len(self.df)):
            price = self.df['close'].iloc[idx]
            
            # Check exit
            if position:
                if position == 'BUY':
                    pnl_pct = (price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - price) / entry_price
                
                if pnl_pct <= -stop_loss or pnl_pct >= take_profit:
                    pnl = INITIAL_CAPITAL * 0.1 * pnl_pct
                    trades.append({'pnl': pnl, 'win': pnl > 0})
                    capital += pnl
                    position = None
            
            # Check entry
            if not position:
                votes = self.get_votes(idx)
                if votes['BUY'] >= required_votes:
                    position = 'BUY'
                    entry_price = price
                elif votes['SELL'] >= required_votes:
                    position = 'SELL'
                    entry_price = price
        
        if not trades:
            return {'total_trades': 0, 'win_rate': 0, 'return_pct': 0, 'profit_factor': 0}
        
        wins = sum(1 for t in trades if t['win'])
        total_pnl = sum(t['pnl'] for t in trades)
        avg_win = np.mean([t['pnl'] for t in trades if t['win']]) if wins > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in trades if not t['win']]) if wins < len(trades) else 0
        
        return {
            'total_trades': len(trades),
            'win_rate': round(wins / len(trades) * 100, 2),
            'return_pct': round((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2),
            'profit_factor': round(abs(avg_win / avg_loss) if avg_loss != 0 else 0, 2),
            'total_pnl': round(total_pnl, 2)
        }


def main():
    data_path = os.path.join(os.path.dirname(__file__), "btc_historical.csv")
    
    if not os.path.exists(data_path):
        print("Data file not found. Run download_data.py first.")
        return
    
    print("Loading data...")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Data: {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    backtester = FastBacktester(df)
    
    # Generate all combinations
    combinations = list(itertools.product(CONSENSUS_RANGE, STOP_LOSS_RANGE, TAKE_PROFIT_RANGE))
    print(f"\nTesting {len(combinations)} parameter combinations...")
    print("="*70)
    
    results = []
    for i, (consensus, sl, tp) in enumerate(combinations, 1):
        if tp <= sl:  # Skip invalid combinations
            continue
        
        result = backtester.run(consensus, sl, tp)
        result['consensus'] = consensus
        result['stop_loss'] = sl
        result['take_profit'] = tp
        result['rr_ratio'] = round(tp / sl, 2)
        results.append(result)
        
        if i % 20 == 0:
            print(f"Progress: {i}/{len(combinations)}")
    
    # Sort by different metrics
    print("\n" + "="*70)
    print("GRID SEARCH RESULTS")
    print("="*70)
    
    # Filter to only include combos with trades
    results = [r for r in results if r['total_trades'] > 0]
    
    # Top by Return
    print("\nTOP 5 BY RETURN:")
    top_return = sorted(results, key=lambda x: x['return_pct'], reverse=True)[:5]
    for i, r in enumerate(top_return, 1):
        print(f"{i}. Return: {r['return_pct']:+6.2f}% | Win: {r['win_rate']:5.1f}% | Trades: {r['total_trades']:3d} | "
              f"Consensus: {r['consensus']*100:.0f}% | SL: {r['stop_loss']*100:.1f}% | TP: {r['take_profit']*100:.1f}%")
    
    # Top by Win Rate
    print("\nTOP 5 BY WIN RATE:")
    top_wr = sorted(results, key=lambda x: x['win_rate'], reverse=True)[:5]
    for i, r in enumerate(top_wr, 1):
        print(f"{i}. Win: {r['win_rate']:5.1f}% | Return: {r['return_pct']:+6.2f}% | Trades: {r['total_trades']:3d} | "
              f"Consensus: {r['consensus']*100:.0f}% | SL: {r['stop_loss']*100:.1f}% | TP: {r['take_profit']*100:.1f}%")
    
    # Best Balanced (score = win_rate * 0.3 + return * 0.5 + profit_factor * 5)
    for r in results:
        r['score'] = r['win_rate'] * 0.3 + r['return_pct'] * 0.5 + r['profit_factor'] * 5
    
    print("\n" + "="*70)
    print("BEST OVERALL SETTINGS (Balanced Score)")
    print("="*70)
    best = sorted(results, key=lambda x: x['score'], reverse=True)[0]
    print(f"""
Consensus Required:  {best['consensus']*100:.0f}% ({int(best['consensus']*4)}/4 voters)
Stop Loss:           {best['stop_loss']*100:.1f}%
Take Profit:         {best['take_profit']*100:.1f}%
Risk:Reward Ratio:   1:{best['rr_ratio']}
------------------------------------------------------------
Total Trades:        {best['total_trades']}
Win Rate:            {best['win_rate']}%
Total Return:        {best['return_pct']:+.2f}%
Profit Factor:       {best['profit_factor']}
""")
    
    # Save all results
    results_df = pd.DataFrame(results)
    output_path = os.path.join(os.path.dirname(__file__), "grid_search_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"All results saved to: {output_path}")


if __name__ == "__main__":
    main()
