"""
Sentinel Quant - Strategy Backtester
Tests the voting strategy on historical data to estimate win rate
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
from dataclasses import dataclass
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
INITIAL_CAPITAL = 10000
POSITION_SIZE_PERCENT = 0.1  # 10% per trade
STOP_LOSS_PERCENT = 0.02  # 2%
TAKE_PROFIT_PERCENT = 0.04  # 4%
CONSENSUS_THRESHOLD = 0.75  # 75% = 3/4 voters agree


@dataclass
class Trade:
    """Single trade record"""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    side: str  # BUY or SELL
    pnl: float
    pnl_percent: float
    win: bool


class TechnicalVoter:
    """Simplified technical voter for backtesting"""
    
    def vote(self, df: pd.DataFrame, idx: int) -> Tuple[str, float]:
        """Vote based on RSI and Bollinger Bands"""
        if idx < 20:
            return "HOLD", 0.5
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[idx]
        
        # Calculate BB position
        sma = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        current_price = df['close'].iloc[idx]
        bb_position = (current_price - lower_band.iloc[idx]) / (upper_band.iloc[idx] - lower_band.iloc[idx])
        
        # Vote logic
        if current_rsi < 30 or (current_rsi < 40 and bb_position < 0.2):
            return "BUY", 0.8
        elif current_rsi > 70 or (current_rsi > 60 and bb_position > 0.8):
            return "SELL", 0.8
        else:
            return "HOLD", 0.5


class MomentumVoter:
    """Momentum voter based on RSI extremes"""
    
    def vote(self, df: pd.DataFrame, idx: int) -> Tuple[str, float]:
        if idx < 14:
            return "HOLD", 0.5
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[idx]
        
        if current_rsi < 30:
            return "BUY", 0.9
        elif current_rsi > 70:
            return "SELL", 0.9
        elif current_rsi < 40:
            return "BUY", 0.6
        elif current_rsi > 60:
            return "SELL", 0.6
        else:
            return "HOLD", 0.5


class TrendVoter:
    """Trend voter based on moving averages"""
    
    def vote(self, df: pd.DataFrame, idx: int) -> Tuple[str, float]:
        if idx < 50:
            return "HOLD", 0.5
        
        sma20 = df['close'].rolling(20).mean().iloc[idx]
        sma50 = df['close'].rolling(50).mean().iloc[idx]
        current_price = df['close'].iloc[idx]
        
        if sma20 > sma50 and current_price > sma20:
            return "BUY", 0.8
        elif sma20 < sma50 and current_price < sma20:
            return "SELL", 0.8
        else:
            return "HOLD", 0.7


class SentimentVoter:
    """Simulated sentiment voter (random with slight bias based on price action)"""
    
    def vote(self, df: pd.DataFrame, idx: int) -> Tuple[str, float]:
        if idx < 5:
            return "HOLD", 0.5
        
        # Simulate sentiment based on recent price movement
        recent_change = (df['close'].iloc[idx] - df['close'].iloc[idx-5]) / df['close'].iloc[idx-5]
        
        # Add some randomness to simulate news
        np.random.seed(int(df['close'].iloc[idx] * 1000) % 2**31)
        noise = np.random.uniform(-0.02, 0.02)
        sentiment_score = recent_change + noise
        
        if sentiment_score > 0.01:
            return "BUY", 0.7
        elif sentiment_score < -0.01:
            return "SELL", 0.7
        else:
            return "HOLD", 0.6


class VotingSystem:
    """Aggregate votes from all voters"""
    
    def __init__(self):
        self.voters = [
            ("Sentiment", SentimentVoter()),
            ("Technical", TechnicalVoter()),
            ("Momentum", MomentumVoter()),
            ("Trend", TrendVoter()),
        ]
    
    def vote(self, df: pd.DataFrame, idx: int) -> Tuple[str, float, dict]:
        """Get aggregated vote"""
        votes = {}
        for name, voter in self.voters:
            vote, confidence = voter.vote(df, idx)
            votes[name] = {"vote": vote, "confidence": confidence}
        
        # Count votes
        buy_count = sum(1 for v in votes.values() if v["vote"] == "BUY")
        sell_count = sum(1 for v in votes.values() if v["vote"] == "SELL")
        hold_count = sum(1 for v in votes.values() if v["vote"] == "HOLD")
        total = len(self.voters)
        
        # Determine action and consensus
        if buy_count > sell_count and buy_count > hold_count:
            action = "BUY"
            consensus = buy_count / total
        elif sell_count > buy_count and sell_count > hold_count:
            action = "SELL"
            consensus = sell_count / total
        else:
            action = "HOLD"
            consensus = hold_count / total
        
        return action, consensus, votes


class Backtester:
    """Backtest the voting strategy"""
    
    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.voting_system = VotingSystem()
        self.trades: List[Trade] = []
        self.capital = INITIAL_CAPITAL
        self.position = None
        self.entry_price = 0
        self.entry_time = None
        
    def run(self) -> dict:
        """Run backtest"""
        logger.info(f"Starting backtest with {len(self.df)} candles")
        logger.info(f"Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        
        for idx in range(50, len(self.df)):  # Start after warmup
            current_price = self.df['close'].iloc[idx]
            current_time = self.df['timestamp'].iloc[idx]
            
            # Check existing position for exit
            if self.position:
                self._check_exit(idx, current_price, current_time)
            
            # Get vote
            action, consensus, votes = self.voting_system.vote(self.df, idx)
            
            # Check for entry
            if not self.position and consensus >= CONSENSUS_THRESHOLD:
                if action == "BUY":
                    self._enter_position("BUY", current_price, current_time)
                elif action == "SELL":
                    self._enter_position("SELL", current_price, current_time)
        
        # Close any open position at end
        if self.position:
            self._exit_position(
                self.df['close'].iloc[-1],
                self.df['timestamp'].iloc[-1],
                "END"
            )
        
        return self._calculate_metrics()
    
    def _enter_position(self, side: str, price: float, time: datetime):
        """Enter a position"""
        self.position = side
        self.entry_price = price
        self.entry_time = time
    
    def _check_exit(self, idx: int, current_price: float, current_time: datetime):
        """Check if position should be exited"""
        if self.position == "BUY":
            pnl_percent = (current_price - self.entry_price) / self.entry_price
        else:  # SELL
            pnl_percent = (self.entry_price - current_price) / self.entry_price
        
        # Check stop loss or take profit
        if pnl_percent <= -STOP_LOSS_PERCENT:
            self._exit_position(current_price, current_time, "STOP_LOSS")
        elif pnl_percent >= TAKE_PROFIT_PERCENT:
            self._exit_position(current_price, current_time, "TAKE_PROFIT")
    
    def _exit_position(self, price: float, time: datetime, reason: str):
        """Exit current position"""
        if self.position == "BUY":
            pnl_percent = (price - self.entry_price) / self.entry_price
            pnl = INITIAL_CAPITAL * POSITION_SIZE_PERCENT * pnl_percent
        else:  # SELL
            pnl_percent = (self.entry_price - price) / self.entry_price
            pnl = INITIAL_CAPITAL * POSITION_SIZE_PERCENT * pnl_percent
        
        trade = Trade(
            entry_time=self.entry_time,
            exit_time=time,
            entry_price=self.entry_price,
            exit_price=price,
            side=self.position,
            pnl=pnl,
            pnl_percent=pnl_percent * 100,
            win=pnl > 0
        )
        self.trades.append(trade)
        self.capital += pnl
        
        self.position = None
        self.entry_price = 0
        self.entry_time = None
    
    def _calculate_metrics(self) -> dict:
        """Calculate backtest metrics"""
        total_trades = len(self.trades)
        if total_trades == 0:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "message": "No trades executed"
            }
        
        winning_trades = sum(1 for t in self.trades if t.win)
        losing_trades = total_trades - winning_trades
        
        total_pnl = sum(t.pnl for t in self.trades)
        avg_win = np.mean([t.pnl for t in self.trades if t.win]) if winning_trades > 0 else 0
        avg_loss = np.mean([t.pnl for t in self.trades if not t.win]) if losing_trades > 0 else 0
        
        win_rate = (winning_trades / total_trades) * 100
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Calculate max drawdown
        equity_curve = [INITIAL_CAPITAL]
        for trade in self.trades:
            equity_curve.append(equity_curve[-1] + trade.pnl)
        
        peak = equity_curve[0]
        max_drawdown = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "final_capital": round(self.capital, 2),
            "return_percent": round((self.capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_percent": round(max_drawdown, 2),
        }


def main():
    """Run backtest"""
    data_path = os.path.join(os.path.dirname(__file__), "btc_historical.csv")
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        logger.info("Run download_data.py first to get historical data")
        return
    
    backtester = Backtester(data_path)
    results = backtester.run()
    
    print("\n" + "="*60)
    print("BACKTEST RESULTS - Voting Strategy")
    print("="*60)
    print(f"Initial Capital:     ${INITIAL_CAPITAL:,.2f}")
    print(f"Position Size:       {POSITION_SIZE_PERCENT*100}%")
    print(f"Stop Loss:           {STOP_LOSS_PERCENT*100}%")
    print(f"Take Profit:         {TAKE_PROFIT_PERCENT*100}%")
    print(f"Consensus Required:  {CONSENSUS_THRESHOLD*100}%")
    print("-"*60)
    print(f"Total Trades:        {results['total_trades']}")
    print(f"Winning Trades:      {results.get('winning_trades', 0)}")
    print(f"Losing Trades:       {results.get('losing_trades', 0)}")
    print(f"Win Rate:            {results['win_rate']}%")
    print("-"*60)
    print(f"Total P&L:           ${results.get('total_pnl', 0):,.2f}")
    print(f"Final Capital:       ${results.get('final_capital', INITIAL_CAPITAL):,.2f}")
    print(f"Return:              {results.get('return_percent', 0)}%")
    print(f"Max Drawdown:        {results.get('max_drawdown_percent', 0)}%")
    print(f"Profit Factor:       {results.get('profit_factor', 0)}")
    print("-"*60)
    print(f"Avg Win:             ${results.get('avg_win', 0):,.2f}")
    print(f"Avg Loss:            ${results.get('avg_loss', 0):,.2f}")
    print("="*60)
    
    # Save detailed trades
    if backtester.trades:
        trades_df = pd.DataFrame([{
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "side": t.side,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "pnl": t.pnl,
            "pnl_percent": t.pnl_percent,
            "win": t.win
        } for t in backtester.trades])
        
        output_path = os.path.join(os.path.dirname(__file__), "backtest_trades.csv")
        trades_df.to_csv(output_path, index=False)
        print(f"\nTrades saved to: {output_path}")


if __name__ == "__main__":
    main()
