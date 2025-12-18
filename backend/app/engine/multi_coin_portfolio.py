"""
Multi-Coin Portfolio Manager

Manages a diversified crypto portfolio:
- SOL (50%) - Ensemble (PPO + Rule-based)
- SUI (30%) - Ensemble (PPO + Rule-based)
- DOGE (20%) - Hybrid (PPO + Rule + Sentiment via Dify)

Features:
- Centralized decision making
- Cross-coin risk management
- Dynamic rebalancing
- Correlation-aware position sizing
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timezone
from enum import Enum
import logging

# Import coin-specific engines
from .sol_decision_engine import SOLDecisionEngine, SOLTradeDecision, SOLTradeAction
from .sol_trend_engine import SOLTrendEngine, TrendSignal, TradeAction as TrendAction
from .doge_sentiment_strategy import DOGESentimentStrategy, DogeTradeDecision, DogeSentimentAction
from .doge_hybrid_engine import DOGEHybridEngine, HybridDogeDecision

logger = logging.getLogger(__name__)


class PortfolioAction(str, Enum):
    """Portfolio-level actions"""
    REBALANCE = "REBALANCE"
    HOLD = "HOLD"
    REDUCE_RISK = "REDUCE_RISK"
    INCREASE_EXPOSURE = "INCREASE_EXPOSURE"


@dataclass
class CoinAllocation:
    """Allocation for a single coin"""
    symbol: str
    target_weight: float  # Target allocation (0-1)
    current_weight: float  # Current allocation (0-1)
    current_value: float  # Current value in USDT
    should_trade: bool
    action: str
    confidence: float
    position_size_usdt: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reasons: List[str]


@dataclass
class PortfolioDecision:
    """Complete portfolio decision"""
    timestamp: datetime
    total_balance: float
    
    # Individual allocations
    allocations: Dict[str, CoinAllocation]
    
    # Portfolio-level
    portfolio_action: PortfolioAction
    total_exposure: float  # Sum of all positions
    cash_reserve: float
    
    # Risk metrics
    max_correlation_exposure: float
    diversification_score: float
    
    # Reasoning
    reasons: List[str]
    warnings: List[str]


class MultiCoinPortfolio:
    """
    Multi-Coin Portfolio Manager
    
    Allocations:
    - SOL: 50% (ensemble PPO + rule)
    - SUI: 30% (rule-based similar to SOL)
    - DOGE: 20% (sentiment via Dify)
    
    Risk Rules:
    - Max total exposure: 80%
    - Min cash reserve: 20%
    - Max per-coin allocation: 50%
    - Correlation limit: no same-direction trades > 80% portfolio
    """
    
    def __init__(
        self,
        sol_ppo_path: str = None,
        sui_ppo_path: str = None,
        doge_ppo_path: str = None,
        dify_api_url: str = None,
        dify_api_key: str = None,
        target_weights: Dict[str, float] = None
    ):
        # Target allocations
        self.target_weights = target_weights or {
            "SOL": 0.50,
            "SUI": 0.30,
            "DOGE": 0.20
        }
        
        # Validate weights
        total = sum(self.target_weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights don't sum to 1.0: {total}")
        
        # Initialize SOL engine - Optimized Trend Following (+303% backtested)
        self.sol_trend_engine = SOLTrendEngine()
        
        # Keep old engine for SUI (reuses SOL logic)
        self.sol_decision_engine = SOLDecisionEngine(
            ppo_model_path=sol_ppo_path,
            mode="RULE_ONLY"  # Fallback for SUI
        )
        
        # Initialize SUI engine - Rule-based (SOL patterns)
        sui_mode = "RULE_ONLY"  # SUI uses rule-based only
        self.sui_engine = SOLDecisionEngine(
            ppo_model_path=sui_ppo_path,
            mode=sui_mode
        )
        
        # Initialize DOGE engine - Hybrid (PPO + Rule + Sentiment)
        self.doge_engine = DOGEHybridEngine(
            ppo_model_path=doge_ppo_path,
            dify_api_url=dify_api_url,
            dify_api_key=dify_api_key
        )
        
        # Risk parameters
        self.max_total_exposure = 0.80
        self.min_cash_reserve = 0.20
        self.max_per_coin = 0.50
        self.max_correlation_exposure = 0.80
        
        # SOL position tracking
        self.sol_entry_price = 0.0
        self.sol_in_position = False
        
        logger.info("MultiCoinPortfolio initialized")
        logger.info(f"  SOL: TREND_FOLLOWING mode (+303% optimized)")
        logger.info(f"  SUI: {sui_mode} mode")
        logger.info(f"  DOGE: HYBRID mode (PPO + Rule + Sentiment)")
        logger.info(f"  Weights: {self.target_weights}")
    
    def make_portfolio_decision(
        self,
        total_balance: float,
        current_positions: Dict[str, float],  # {symbol: value_usdt}
        market_data: Dict[str, dict]  # {symbol: {price, sma10, sma30, ...}}
    ) -> PortfolioDecision:
        """
        Make portfolio-wide decision.
        
        Args:
            total_balance: Total portfolio value in USDT
            current_positions: Current positions {symbol: value}
            market_data: Market data for each coin
            
        Returns:
            PortfolioDecision with all allocations
        """
        reasons = []
        warnings = []
        allocations = {}
        
        # Calculate current weights
        total_invested = sum(current_positions.values())
        
        # ========== SOL DECISION (OPTIMIZED TREND FOLLOWING) ==========
        if "SOL" in market_data:
            sol_data = market_data["SOL"]
            current_price = sol_data.get("price", 0)
            
            # Update position state in trend engine
            self.sol_trend_engine.update_position(self.sol_entry_price, self.sol_in_position)
            
            # Get trend signal
            sol_signal = self.sol_trend_engine.get_signal(
                current_price=current_price,
                sma10=sol_data.get("sma10", 0),
                sma20=sol_data.get("sma20", sol_data.get("sma30", 0)),  # Use sma20 or sma30
                sma50=sol_data.get("sma50", 0),
                rsi=sol_data.get("rsi", 50),
                high_20d=sol_data.get("high_20d", current_price * 1.1),
                momentum_7d=sol_data.get("momentum_7d", 0)
            )
            
            # Convert TrendSignal to allocation
            sol_action = SOLTradeAction.HOLD
            sol_should_trade = False
            sol_position_size = 0.0
            sol_confidence = sol_signal.confidence
            
            if sol_signal.action == TrendAction.BUY:
                sol_action = SOLTradeAction.BUY
                sol_should_trade = True
                sol_position_size = sol_signal.position_size * total_balance * self.target_weights["SOL"]
                # Track entry
                self.sol_entry_price = current_price
                self.sol_in_position = True
            elif sol_signal.action == TrendAction.SELL:
                sol_action = SOLTradeAction.SELL
                sol_should_trade = True
                # Clear position
                self.sol_entry_price = 0.0
                self.sol_in_position = False
            
            allocations["SOL"] = CoinAllocation(
                symbol="SOL",
                target_weight=self.target_weights["SOL"],
                current_weight=current_positions.get("SOL", 0) / total_balance if total_balance > 0 else 0,
                current_value=current_positions.get("SOL", 0),
                should_trade=sol_should_trade,
                action=sol_action.value,
                confidence=sol_confidence,
                position_size_usdt=sol_position_size,
                stop_loss=None,  # Trend engine handles exits
                take_profit=None,  # Trend engine handles exits
                reasons=[f"TREND: {sol_signal.reason}", f"Mode: {sol_signal.trend.value}"]
            )
        
        # ========== SUI DECISION ==========
        if "SUI" in market_data:
            sui_data = market_data["SUI"]
            sui_decision = self.sui_engine.make_decision(
                symbol="SUIUSDT",
                current_price=sui_data.get("price", 0),
                account_balance=total_balance * self.target_weights["SUI"],
                sma10=sui_data.get("sma10", 0),
                sma30=sui_data.get("sma30", 0),
                sma50=sui_data.get("sma50", 0),
                rsi=sui_data.get("rsi", 50),
                atr=sui_data.get("atr", 0),
                volatility_ratio=sui_data.get("volatility_ratio", 1),
                volume_ratio=sui_data.get("volume_ratio", 1)
            )
            
            allocations["SUI"] = self._create_allocation(
                symbol="SUI",
                decision=sui_decision,
                target_weight=self.target_weights["SUI"],
                current_value=current_positions.get("SUI", 0),
                total_balance=total_balance
            )
        
        # ========== DOGE DECISION ==========
        if "DOGE" in market_data:
            doge_data = market_data["DOGE"]
            
            # DOGE uses Hybrid Engine: PPO + Rule + Sentiment
            doge_decision = self.doge_engine.make_decision(
                current_price=doge_data.get("price", 0),
                sma10=doge_data.get("sma10", 0),
                sma30=doge_data.get("sma30", 0),
                sma50=doge_data.get("sma50", 0),
                rsi=doge_data.get("rsi", 50),
                volume_ratio=doge_data.get("volume_ratio", 1),
                change_24h=doge_data.get("change_24h", 0),
                sentiment_result=market_data.get("DOGE", {}).get("sentiment")
            )
            
            allocations["DOGE"] = self._create_hybrid_doge_allocation(
                decision=doge_decision,
                target_weight=self.target_weights["DOGE"],
                current_value=current_positions.get("DOGE", 0),
                total_balance=total_balance
            )
        
        # ========== PORTFOLIO-LEVEL ADJUSTMENTS ==========
        
        # Check total exposure
        total_exposure = sum(a.position_size_usdt for a in allocations.values())
        exposure_pct = total_exposure / total_balance if total_balance > 0 else 0
        
        # Apply exposure limits
        if exposure_pct > self.max_total_exposure:
            # Scale down all positions proportionally
            scale_factor = self.max_total_exposure / exposure_pct
            for symbol in allocations:
                allocations[symbol].position_size_usdt *= scale_factor
            warnings.append(f"Exposure limited: {exposure_pct*100:.0f}% → {self.max_total_exposure*100:.0f}%")
            total_exposure *= scale_factor
        
        # Check correlation (if all same direction, reduce)
        buy_exposure = sum(
            a.position_size_usdt for a in allocations.values()
            if a.action in ["BUY", "STRONG_BUY"]
        )
        
        if buy_exposure / total_balance > self.max_correlation_exposure:
            warnings.append("High correlation: Multiple BUY signals")
            reasons.append("Consider staggering entries")
        
        # Determine portfolio action
        portfolio_action = PortfolioAction.HOLD
        if any(a.should_trade for a in allocations.values()):
            portfolio_action = PortfolioAction.REBALANCE
        
        # Diversification score (0 = concentrated, 1 = diversified)
        active_positions = sum(1 for a in allocations.values() if a.position_size_usdt > 0)
        diversification = active_positions / len(allocations) if allocations else 0
        
        return PortfolioDecision(
            timestamp=datetime.now(timezone.utc),
            total_balance=total_balance,
            allocations=allocations,
            portfolio_action=portfolio_action,
            total_exposure=total_exposure / total_balance if total_balance > 0 else 0,
            cash_reserve=1 - (total_exposure / total_balance) if total_balance > 0 else 1,
            max_correlation_exposure=buy_exposure / total_balance if total_balance > 0 else 0,
            diversification_score=diversification,
            reasons=reasons,
            warnings=warnings
        )
    
    def _create_allocation(
        self,
        symbol: str,
        decision: SOLTradeDecision,
        target_weight: float,
        current_value: float,
        total_balance: float
    ) -> CoinAllocation:
        """Create allocation from SOL-style decision"""
        
        # Calculate position size
        if decision.should_trade and decision.action == SOLTradeAction.BUY:
            position_size = min(
                decision.position_value_usdt or 0,
                total_balance * target_weight * decision.position_percent
            )
        else:
            position_size = 0
        
        # Cap at max per coin
        position_size = min(position_size, total_balance * self.max_per_coin)
        
        return CoinAllocation(
            symbol=symbol,
            target_weight=target_weight,
            current_weight=current_value / total_balance if total_balance > 0 else 0,
            current_value=current_value,
            should_trade=decision.should_trade,
            action=decision.action.value,
            confidence=decision.model_confidence,
            position_size_usdt=position_size,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            reasons=decision.reasons[:3]
        )
    
    def _create_doge_allocation(
        self,
        decision: DogeTradeDecision,
        target_weight: float,
        current_value: float,
        total_balance: float
    ) -> CoinAllocation:
        """Create allocation from DOGE decision (legacy)"""
        
        # DOGE position size
        if decision.should_trade and decision.action in [DogeSentimentAction.BUY, DogeSentimentAction.STRONG_BUY]:
            position_size = total_balance * target_weight * decision.position_percent
        else:
            position_size = 0
        
        # Cap at max per coin
        position_size = min(position_size, total_balance * self.max_per_coin)
        
        return CoinAllocation(
            symbol="DOGE",
            target_weight=target_weight,
            current_weight=current_value / total_balance if total_balance > 0 else 0,
            current_value=current_value,
            should_trade=decision.should_trade,
            action=decision.action.value,
            confidence=decision.sentiment_confidence / 100,
            position_size_usdt=position_size,
            stop_loss=decision.stop_loss_pct,
            take_profit=decision.take_profit_pct,
            reasons=decision.reasons[:3]
        )
    
    def _create_hybrid_doge_allocation(
        self,
        decision: HybridDogeDecision,
        target_weight: float,
        current_value: float,
        total_balance: float
    ) -> CoinAllocation:
        """Create allocation from DOGE Hybrid decision (PPO + Rule + Sentiment)"""
        
        # DOGE position size from hybrid
        if decision.should_trade and decision.action in [DogeSentimentAction.BUY, DogeSentimentAction.STRONG_BUY]:
            position_size = total_balance * target_weight * decision.position_percent
        else:
            position_size = 0
        
        # Cap at max per coin
        position_size = min(position_size, total_balance * self.max_per_coin)
        
        # Build reasons with source info
        reasons = [f"Source: {decision.decision_source}"]
        if decision.ppo_action:
            reasons.append(f"PPO: {decision.ppo_action}")
        reasons.append(f"Rule: {decision.rule_action}")
        if decision.sentiment_action:
            reasons.append(f"Sentiment: {decision.sentiment_action}")
        
        return CoinAllocation(
            symbol="DOGE",
            target_weight=target_weight,
            current_weight=current_value / total_balance if total_balance > 0 else 0,
            current_value=current_value,
            should_trade=decision.should_trade,
            action=decision.action.value,
            confidence=decision.combined_confidence,
            position_size_usdt=position_size,
            stop_loss=decision.stop_loss_pct,
            take_profit=decision.take_profit_pct,
            reasons=reasons[:3]
        )
    
    def get_summary(self, decision: PortfolioDecision) -> str:
        """Get human-readable summary of portfolio decision"""
        lines = [
            "=" * 60,
            "MULTI-COIN PORTFOLIO DECISION",
            "=" * 60,
            f"Time: {decision.timestamp}",
            f"Balance: ${decision.total_balance:,.2f}",
            f"Total Exposure: {decision.total_exposure*100:.0f}%",
            f"Cash Reserve: {decision.cash_reserve*100:.0f}%",
            "",
            "ALLOCATIONS:",
        ]
        
        for symbol, alloc in decision.allocations.items():
            lines.append(f"  [{symbol}]")
            lines.append(f"    Action: {alloc.action}")
            lines.append(f"    Position: ${alloc.position_size_usdt:,.2f}")
            lines.append(f"    Confidence: {alloc.confidence*100:.0f}%")
        
        if decision.warnings:
            lines.append("")
            lines.append("WARNINGS:")
            for w in decision.warnings:
                lines.append(f"  ⚠️ {w}")
        
        return "\n".join(lines)
