"""
Sentinel Quant - PAXG Grid Search Optimizer
Optimizes trading parameters specifically for PAXG (Gold-backed token)
Based on grid_search.py but uses PAXG data
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import itertools

# Configuration
INITIAL_CAPITAL = 10000

# Parameter ranges to test (adjusted for PAXG's lower volatility)
CONSENSUS_RANGE = [0.25, 0.50, 0.75, 1.0]  # 1/4, 2/4, 3/4, 4/4
STOP_LOSS_RANGE = [0.005, 0.01, 0.015, 0.02, 0.025]  # 0.5% to 2.5% (tighter for PAXG)
TAKE_PROFIT_RANGE = [0.01, 0.015, 0.02, 0.03, 0.04]  # 1% to 4% (smaller for PAXG)


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
        
        # Technical voter (adjusted for PAXG - less extreme thresholds)
        if rsi < 35 or (rsi < 45 and bb_pos < 0.2):  # Less extreme for gold
            votes['BUY'] += 1
        elif rsi > 65 or (rsi > 55 and bb_pos > 0.8):  # Less extreme for gold
            votes['SELL'] += 1
        else:
            votes['HOLD'] += 1
        
        # Momentum voter (adjusted for PAXG)
        if rsi < 35:
            votes['BUY'] += 1
        elif rsi > 65:
            votes['SELL'] += 1
        elif rsi < 45:
            votes['BUY'] += 1
        elif rsi > 55:
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
        
        # Sentiment voter (simulated - gold focused)
        if pd.notna(pct_change):
            np.random.seed(int(price * 100) % 2**31)
            noise = np.random.uniform(-0.01, 0.01)  # Less noise for gold
            sentiment = pct_change + noise
            if sentiment > 0.005:  # Lower threshold for gold
                votes['BUY'] += 1
            elif sentiment < -0.005:
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
    data_path = os.path.join(os.path.dirname(__file__), "paxg_historical.csv")
    
    if not os.path.exists(data_path):
        print("PAXG data file not found. Run download_paxg.py first.")
        print("Command: python download_paxg.py")
        return
    
    print("Loading PAXG data...")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Data: {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    backtester = FastBacktester(df)
    
    # Generate all combinations
    combinations = list(itertools.product(CONSENSUS_RANGE, STOP_LOSS_RANGE, TAKE_PROFIT_RANGE))
    print(f"\nTesting {len(combinations)} parameter combinations for PAXG...")
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
    print("PAXG GRID SEARCH RESULTS")
    print("="*70)
    
    # Filter to only include combos with trades
    results = [r for r in results if r['total_trades'] > 0]
    
    if not results:
        print("No trades executed with any parameter combination.")
        print("PAXG may require different indicator thresholds.")
        return
    
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
    
    # Best Balanced
    for r in results:
        r['score'] = r['win_rate'] * 0.3 + r['return_pct'] * 0.5 + r['profit_factor'] * 5
    
    print("\n" + "="*70)
    print("BEST OVERALL PAXG SETTINGS (Balanced Score)")
    print("="*70)
    best = sorted(results, key=lambda x: x['score'], reverse=True)[0]
    print(f"""
Asset:               PAXG (Pax Gold)
Consensus Required:  {best['consensus']*100:.0f}% ({int(best['consensus']*4)}/4 voters)
Stop Loss:           {best['stop_loss']*100:.2f}%
Take Profit:         {best['take_profit']*100:.2f}%
Risk:Reward Ratio:   1:{best['rr_ratio']}
------------------------------------------------------------
Total Trades:        {best['total_trades']}
Win Rate:            {best['win_rate']}%
Total Return:        {best['return_pct']:+.2f}%
Profit Factor:       {best['profit_factor']}
""")
    
    # Compare with BTC settings
    print("="*70)
    print("COMPARISON: BTC vs PAXG Optimal Settings")
    print("="*70)
    print(f"""
| Parameter      | BTC (Current)    | PAXG (Optimal)   |
|----------------|------------------|------------------|
| Consensus      | 25%              | {best['consensus']*100:.0f}%              |
| Stop Loss      | 2.0%             | {best['stop_loss']*100:.2f}%            |
| Take Profit    | 4.0%             | {best['take_profit']*100:.2f}%            |
| R:R Ratio      | 1:2.0            | 1:{best['rr_ratio']}            |
""")
    
    # Save all results
    results_df = pd.DataFrame(results)
    output_path = os.path.join(os.path.dirname(__file__), "paxg_grid_search_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"All results saved to: {output_path}")


if __name__ == "__main__":
    main()
