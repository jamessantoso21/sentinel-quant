"""
Sentinel Quant - Strategy Optimizer
Tests multiple parameter combinations to find optimal settings
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Import from backtest
from backtest import Backtester, INITIAL_CAPITAL

# Parameter combinations to test
PARAM_SETS = [
    # (consensus, position_size, stop_loss, take_profit)
    (0.75, 0.10, 0.02, 0.04),  # Original
    (0.60, 0.10, 0.02, 0.04),  # Lower consensus
    (0.50, 0.10, 0.02, 0.04),  # 50% consensus (2/4)
    (0.75, 0.10, 0.015, 0.03), # Tighter SL/TP
    (0.75, 0.10, 0.01, 0.02),  # Very tight
    (0.60, 0.10, 0.015, 0.045), # 1:3 risk/reward
    (0.50, 0.10, 0.01, 0.03),  # 1:3 tight
    (0.75, 0.15, 0.02, 0.04),  # Larger position
    (0.60, 0.15, 0.015, 0.03), # Mixed
]


def run_optimization():
    data_path = os.path.join(os.path.dirname(__file__), "btc_historical.csv")
    
    if not os.path.exists(data_path):
        print("Data file not found. Run download_data.py first.")
        return
    
    results = []
    
    print("="*80)
    print("BACKTEST OPTIMIZATION - Testing Multiple Parameter Sets")
    print("="*80)
    print()
    
    for i, (consensus, pos_size, sl, tp) in enumerate(PARAM_SETS, 1):
        # Modify globals in backtest module
        import backtest
        backtest.CONSENSUS_THRESHOLD = consensus
        backtest.POSITION_SIZE_PERCENT = pos_size
        backtest.STOP_LOSS_PERCENT = sl
        backtest.TAKE_PROFIT_PERCENT = tp
        
        # Reload backtester
        backtester = Backtester(data_path)
        result = backtester.run()
        
        result['consensus'] = consensus
        result['pos_size'] = pos_size
        result['stop_loss'] = sl
        result['take_profit'] = tp
        result['risk_reward'] = tp / sl
        
        results.append(result)
        
        print(f"[{i}/{len(PARAM_SETS)}] Consensus: {consensus*100:.0f}% | SL: {sl*100}% | TP: {tp*100}% | R:R = 1:{tp/sl:.1f}")
        print(f"     Trades: {result['total_trades']:3d} | Win Rate: {result['win_rate']:5.1f}% | PnL: ${result.get('total_pnl', 0):+7.2f} | Return: {result.get('return_percent', 0):+5.2f}%")
        print()
    
    # Find best by different metrics
    print("="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    
    # Sort by return
    results_sorted = sorted(results, key=lambda x: x.get('return_percent', 0), reverse=True)
    
    print("\nTop 3 by Return:")
    for i, r in enumerate(results_sorted[:3], 1):
        print(f"  {i}. Return: {r.get('return_percent', 0):+.2f}% | Win Rate: {r['win_rate']:.1f}% | "
              f"Trades: {r['total_trades']} | Consensus: {r['consensus']*100:.0f}% | R:R = 1:{r['risk_reward']:.1f}")
    
    # Sort by win rate
    results_sorted = sorted(results, key=lambda x: x['win_rate'], reverse=True)
    
    print("\nTop 3 by Win Rate:")
    for i, r in enumerate(results_sorted[:3], 1):
        print(f"  {i}. Win Rate: {r['win_rate']:.1f}% | Return: {r.get('return_percent', 0):+.2f}% | "
              f"Trades: {r['total_trades']} | Consensus: {r['consensus']*100:.0f}% | R:R = 1:{r['risk_reward']:.1f}")
    
    # Best overall (balanced)
    for r in results:
        r['score'] = r['win_rate'] * 0.3 + r.get('return_percent', 0) * 0.5 + r.get('profit_factor', 1) * 10
    
    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
    
    print("\nBEST OVERALL (Balanced Score):")
    best = results_sorted[0]
    print(f"  Consensus:    {best['consensus']*100:.0f}%")
    print(f"  Stop Loss:    {best['stop_loss']*100:.1f}%")
    print(f"  Take Profit:  {best['take_profit']*100:.1f}%")
    print(f"  Risk:Reward:  1:{best['risk_reward']:.1f}")
    print(f"  ---")
    print(f"  Win Rate:     {best['win_rate']:.1f}%")
    print(f"  Total Return: {best.get('return_percent', 0):+.2f}%")
    print(f"  Max Drawdown: {best.get('max_drawdown_percent', 0):.1f}%")
    print(f"  Profit Factor: {best.get('profit_factor', 0):.2f}")
    
    print("="*80)


if __name__ == "__main__":
    run_optimization()
