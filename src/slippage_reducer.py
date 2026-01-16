"""Slippage reduction engine for CFD Smart Entry System.

This module implements intelligent order execution strategies
to minimize slippage during trade entry.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from .market_analyzer import MarketCondition, TrendDirection


class ExecutionTier(Enum):
    """Enumeration for execution quality tiers."""

    OPTIMAL = "optimal"
    ACCEPTABLE = "acceptable"
    CAUTION = "caution"
    AVOID = "avoid"


class OrderType(Enum):
    """Enumeration for order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"


@dataclass
class SlippageConfig:
    """Configuration for slippage reduction."""

    max_slippage_pips: float
    execution_timeout_ms: int
    retry_on_requote: bool
    price_tolerance_pips: float


@dataclass
class ExecutionPlan:
    """Execution plan with slippage mitigation strategy."""

    symbol: str
    order_type: OrderType
    tier: ExecutionTier
    recommended_price: float
    price_tolerance: float
    split_orders: bool
    num_splits: int
    wait_for_improvement: bool
    max_wait_ms: int
    expected_slippage_pips: float
    reasoning: str


@dataclass
class ExecutionResult:
    """Result of order execution attempt."""

    success: bool
    executed_price: float
    requested_price: float
    slippage_pips: float
    execution_time_ms: int
    fill_rate: float
    message: str


class SlippageReducer:
    """Engine for reducing slippage during order execution."""

    # Configuration thresholds for order splitting
    LARGE_ORDER_THRESHOLD = 1.0  # Lots - always split orders above this
    MEDIUM_ORDER_THRESHOLD = 0.5  # Lots - split in poor conditions above this

    def __init__(self, config: SlippageConfig | None = None) -> None:
        """Initialize slippage reducer.

        Args:
            config: Slippage configuration. Uses defaults if None.
        """
        self.config = config or SlippageConfig(
            max_slippage_pips=3.0,
            execution_timeout_ms=500,
            retry_on_requote=True,
            price_tolerance_pips=1.0
        )
        self._execution_history: list[ExecutionResult] = []
        self._tier_thresholds = {
            ExecutionTier.OPTIMAL: {"spread": 1.0, "volatility": 0.5},
            ExecutionTier.ACCEPTABLE: {"spread": 2.0, "volatility": 1.0},
            ExecutionTier.CAUTION: {"spread": 3.0, "volatility": 1.5},
        }

        logger.info(
            f"SlippageReducer initialized with max_slippage={self.config.max_slippage_pips} pips"
        )

    def analyze_execution_conditions(
        self,
        condition: MarketCondition
    ) -> ExecutionTier:
        """Analyze current market conditions for execution quality.

        Args:
            condition: Current market condition.

        Returns:
            ExecutionTier indicating recommended execution approach.
        """
        spread = condition.spread
        volatility = condition.volatility

        for tier, thresholds in self._tier_thresholds.items():
            if spread <= thresholds["spread"] and volatility <= thresholds["volatility"]:
                return tier

        return ExecutionTier.AVOID

    def create_execution_plan(
        self,
        symbol: str,
        is_buy: bool,
        quantity: float,
        current_price: float,
        condition: MarketCondition
    ) -> ExecutionPlan:
        """Create an execution plan optimized for slippage reduction.

        Args:
            symbol: Trading symbol.
            is_buy: True for buy order, False for sell.
            quantity: Order quantity in lots.
            current_price: Current market price.
            condition: Current market condition.

        Returns:
            ExecutionPlan with recommended strategy.
        """
        tier = self.analyze_execution_conditions(condition)

        # Determine order type based on tier
        order_type = self._determine_order_type(tier, condition)

        # Calculate recommended price with tolerance
        recommended_price, tolerance = self._calculate_optimal_price(
            is_buy, current_price, condition, tier
        )

        # Determine if order splitting is needed
        split_orders, num_splits = self._should_split_order(
            quantity, tier, condition.volatility
        )

        # Calculate wait strategy
        wait_for_improvement, max_wait = self._determine_wait_strategy(
            tier, condition.trend, is_buy
        )

        # Estimate expected slippage
        expected_slippage = self._estimate_slippage(
            tier, condition.spread, condition.volatility
        )

        # Generate reasoning
        reasoning = self._generate_plan_reasoning(
            tier, order_type, split_orders, wait_for_improvement
        )

        plan = ExecutionPlan(
            symbol=symbol,
            order_type=order_type,
            tier=tier,
            recommended_price=recommended_price,
            price_tolerance=tolerance,
            split_orders=split_orders,
            num_splits=num_splits,
            wait_for_improvement=wait_for_improvement,
            max_wait_ms=max_wait,
            expected_slippage_pips=expected_slippage,
            reasoning=reasoning
        )

        logger.info(
            f"Execution plan for {symbol}: {order_type.value} at {tier.value} tier, "
            f"expected slippage={expected_slippage:.2f} pips"
        )

        return plan

    def _determine_order_type(
        self,
        tier: ExecutionTier,
        condition: MarketCondition
    ) -> OrderType:
        """Determine optimal order type based on conditions.

        Args:
            tier: Execution tier.
            condition: Market condition.

        Returns:
            Recommended OrderType.
        """
        if tier == ExecutionTier.OPTIMAL:
            # Good conditions - market order is fine
            return OrderType.MARKET
        elif tier == ExecutionTier.ACCEPTABLE:
            # Slight concerns - use limit for better control
            return OrderType.LIMIT
        elif tier == ExecutionTier.CAUTION:
            # Poor conditions - use limit with care
            return OrderType.LIMIT
        else:
            # Avoid - if must trade, use limit
            return OrderType.LIMIT

    def _calculate_optimal_price(
        self,
        is_buy: bool,
        current_price: float,
        condition: MarketCondition,
        tier: ExecutionTier
    ) -> tuple[float, float]:
        """Calculate optimal entry price with tolerance.

        Args:
            is_buy: True for buy order.
            current_price: Current market price.
            condition: Market condition.
            tier: Execution tier.

        Returns:
            Tuple of (recommended_price, tolerance).
        """
        # Base tolerance from config
        base_tolerance = self.config.price_tolerance_pips

        # Adjust tolerance based on tier
        tier_multiplier = {
            ExecutionTier.OPTIMAL: 0.5,
            ExecutionTier.ACCEPTABLE: 1.0,
            ExecutionTier.CAUTION: 1.5,
            ExecutionTier.AVOID: 2.0,
        }

        tolerance = base_tolerance * tier_multiplier.get(tier, 1.0)

        # Calculate pip value (assuming standard pip)
        pip_value = self._get_pip_value(condition.symbol, current_price)

        # Adjust recommended price based on trend
        trend_adjustment = 0.0
        if condition.trend == TrendDirection.BULLISH and is_buy:
            # Bullish and buying - price might run up, be more aggressive
            trend_adjustment = -0.5 * pip_value
        elif condition.trend == TrendDirection.BEARISH and not is_buy:
            # Bearish and selling - price might run down, be more aggressive
            trend_adjustment = 0.5 * pip_value

        recommended_price = current_price + trend_adjustment
        tolerance_value = tolerance * pip_value

        return recommended_price, tolerance_value

    def _should_split_order(
        self,
        quantity: float,
        tier: ExecutionTier,
        volatility: float
    ) -> tuple[bool, int]:
        """Determine if order should be split.

        Args:
            quantity: Order quantity in lots.
            tier: Execution tier.
            volatility: Current volatility.

        Returns:
            Tuple of (should_split, number_of_splits).
        """
        # Split large orders or in poor conditions
        should_split = (
            quantity > self.LARGE_ORDER_THRESHOLD or
            (quantity > self.MEDIUM_ORDER_THRESHOLD and
             tier in [ExecutionTier.CAUTION, ExecutionTier.AVOID])
        )

        if should_split:
            # Calculate number of splits
            if quantity > 2.0:
                num_splits = min(int(quantity / 0.5), 5)
            elif quantity > self.LARGE_ORDER_THRESHOLD:
                num_splits = 3
            else:
                num_splits = 2

            return True, num_splits

        return False, 1

    def _determine_wait_strategy(
        self,
        tier: ExecutionTier,
        trend: TrendDirection,
        is_buy: bool
    ) -> tuple[bool, int]:
        """Determine if should wait for price improvement.

        Args:
            tier: Execution tier.
            trend: Current trend direction.
            is_buy: True for buy order.

        Returns:
            Tuple of (wait_for_improvement, max_wait_ms).
        """
        # In optimal conditions, execute immediately
        if tier == ExecutionTier.OPTIMAL:
            return False, 0

        # Check if trend is favorable for waiting
        favorable_trend = (
            (trend == TrendDirection.BEARISH and is_buy) or
            (trend == TrendDirection.BULLISH and not is_buy)
        )

        if favorable_trend and tier == ExecutionTier.ACCEPTABLE:
            # Wait for pullback
            return True, 2000  # 2 seconds

        if tier == ExecutionTier.CAUTION:
            # Wait longer in cautious conditions
            return True, 5000  # 5 seconds

        return False, 0

    def _estimate_slippage(
        self,
        tier: ExecutionTier,
        spread: float,
        volatility: float
    ) -> float:
        """Estimate expected slippage in pips.

        Args:
            tier: Execution tier.
            spread: Current spread.
            volatility: Current volatility.

        Returns:
            Estimated slippage in pips.
        """
        # Base slippage from spread
        base_slippage = spread * 0.5

        # Volatility contribution
        volatility_slippage = volatility * 0.1

        # Tier adjustment
        tier_factor = {
            ExecutionTier.OPTIMAL: 0.5,
            ExecutionTier.ACCEPTABLE: 1.0,
            ExecutionTier.CAUTION: 1.5,
            ExecutionTier.AVOID: 2.5,
        }

        estimated = (base_slippage + volatility_slippage) * tier_factor.get(tier, 1.0)

        return min(estimated, self.config.max_slippage_pips * 2)

    def _get_pip_value(self, symbol: str, price: float) -> float:
        """Get pip value for a symbol.

        Args:
            symbol: Trading symbol.
            price: Current price.

        Returns:
            Pip value.
        """
        # Simplified pip calculation
        symbol_upper = symbol.upper()

        if "JPY" in symbol_upper:
            return 0.01
        elif symbol_upper in ["GOLD", "XAUUSD"]:
            return 0.1
        elif symbol_upper in ["US30", "DJ30"]:
            return 1.0
        else:
            return 0.0001

    def _generate_plan_reasoning(
        self,
        tier: ExecutionTier,
        order_type: OrderType,
        split_orders: bool,
        wait_for_improvement: bool
    ) -> str:
        """Generate reasoning for execution plan.

        Args:
            tier: Execution tier.
            order_type: Recommended order type.
            split_orders: Whether to split orders.
            wait_for_improvement: Whether to wait.

        Returns:
            Reasoning string.
        """
        reasons = [f"Execution conditions: {tier.value}"]

        if order_type == OrderType.LIMIT:
            reasons.append("Using limit order for price control")
        else:
            reasons.append("Market order suitable for current conditions")

        if split_orders:
            reasons.append("Order splitting recommended to reduce market impact")

        if wait_for_improvement:
            reasons.append("Waiting for price improvement advised")

        return " | ".join(reasons)

    def record_execution(self, result: ExecutionResult) -> None:
        """Record execution result for analysis.

        Args:
            result: Execution result to record.
        """
        self._execution_history.append(result)

        # Keep only recent history
        if len(self._execution_history) > 1000:
            self._execution_history = self._execution_history[-500:]

        logger.debug(
            f"Execution recorded: slippage={result.slippage_pips:.2f} pips, "
            f"time={result.execution_time_ms}ms"
        )

    def get_execution_statistics(self) -> dict[str, float]:
        """Get execution statistics from history.

        Returns:
            Dictionary with execution statistics.
        """
        if not self._execution_history:
            return {
                "avg_slippage_pips": 0.0,
                "max_slippage_pips": 0.0,
                "avg_execution_time_ms": 0.0,
                "success_rate": 0.0,
                "sample_size": 0,
            }

        slippages = [r.slippage_pips for r in self._execution_history]
        times = [r.execution_time_ms for r in self._execution_history]
        successes = [1 if r.success else 0 for r in self._execution_history]

        return {
            "avg_slippage_pips": float(np.mean(slippages)),
            "max_slippage_pips": float(np.max(slippages)),
            "avg_execution_time_ms": float(np.mean(times)),
            "success_rate": float(np.mean(successes)),
            "sample_size": len(self._execution_history),
        }

    def should_execute(
        self,
        plan: ExecutionPlan,
        current_price: float
    ) -> tuple[bool, str]:
        """Determine if order should be executed now.

        Args:
            plan: Execution plan.
            current_price: Current market price.

        Returns:
            Tuple of (should_execute, reason).
        """
        # Check if within tolerance
        price_diff = abs(current_price - plan.recommended_price)

        if price_diff <= plan.price_tolerance:
            return True, "Price within tolerance"

        if plan.tier == ExecutionTier.AVOID:
            return False, "Market conditions unfavorable - execution avoided"

        if plan.expected_slippage_pips > self.config.max_slippage_pips:
            return False, f"Expected slippage ({plan.expected_slippage_pips:.2f}) exceeds maximum"

        return True, "Proceeding with adjusted expectations"
