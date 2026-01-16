"""Tests for slippage reducer module."""

import pytest

from src.slippage_reducer import (
    SlippageReducer,
    SlippageConfig,
    ExecutionTier,
    OrderType,
    ExecutionPlan,
    ExecutionResult
)
from src.market_analyzer import MarketCondition, TrendDirection
import pandas as pd


class TestSlippageReducer:
    """Test cases for SlippageReducer class."""

    @pytest.fixture
    def reducer(self) -> SlippageReducer:
        """Create slippage reducer instance."""
        config = SlippageConfig(
            max_slippage_pips=3.0,
            execution_timeout_ms=500,
            retry_on_requote=True,
            price_tolerance_pips=1.0
        )
        return SlippageReducer(config)

    @pytest.fixture
    def optimal_condition(self) -> MarketCondition:
        """Create optimal market condition."""
        return MarketCondition(
            symbol="USDJPY",
            trend=TrendDirection.BULLISH,
            volatility=0.3,
            spread=0.5,
            momentum=30.0,
            support_level=129.0,
            resistance_level=131.0,
            volume_ratio=1.0,
            timestamp=pd.Timestamp.now()
        )

    @pytest.fixture
    def poor_condition(self) -> MarketCondition:
        """Create poor market condition."""
        return MarketCondition(
            symbol="USDJPY",
            trend=TrendDirection.NEUTRAL,
            volatility=2.0,
            spread=4.0,
            momentum=5.0,
            support_level=129.0,
            resistance_level=131.0,
            volume_ratio=0.5,
            timestamp=pd.Timestamp.now()
        )

    def test_analyze_optimal_conditions(
        self,
        reducer: SlippageReducer,
        optimal_condition: MarketCondition
    ) -> None:
        """Test tier analysis for optimal conditions."""
        tier = reducer.analyze_execution_conditions(optimal_condition)
        assert tier == ExecutionTier.OPTIMAL

    def test_analyze_poor_conditions(
        self,
        reducer: SlippageReducer,
        poor_condition: MarketCondition
    ) -> None:
        """Test tier analysis for poor conditions."""
        tier = reducer.analyze_execution_conditions(poor_condition)
        assert tier == ExecutionTier.AVOID

    def test_create_execution_plan_optimal(
        self,
        reducer: SlippageReducer,
        optimal_condition: MarketCondition
    ) -> None:
        """Test execution plan creation for optimal conditions."""
        plan = reducer.create_execution_plan(
            symbol="USDJPY",
            is_buy=True,
            quantity=0.1,
            current_price=130.0,
            condition=optimal_condition
        )

        assert isinstance(plan, ExecutionPlan)
        assert plan.tier == ExecutionTier.OPTIMAL
        assert plan.order_type == OrderType.MARKET
        assert not plan.split_orders
        assert not plan.wait_for_improvement

    def test_create_execution_plan_caution(
        self,
        reducer: SlippageReducer,
        poor_condition: MarketCondition
    ) -> None:
        """Test execution plan for cautious conditions."""
        # Modify condition to be in CAUTION tier
        poor_condition.spread = 2.5
        poor_condition.volatility = 1.2

        plan = reducer.create_execution_plan(
            symbol="USDJPY",
            is_buy=True,
            quantity=0.1,
            current_price=130.0,
            condition=poor_condition
        )

        assert plan.order_type == OrderType.LIMIT

    def test_order_splitting_large_quantity(
        self,
        reducer: SlippageReducer,
        optimal_condition: MarketCondition
    ) -> None:
        """Test order splitting for large quantities."""
        plan = reducer.create_execution_plan(
            symbol="USDJPY",
            is_buy=True,
            quantity=2.5,
            current_price=130.0,
            condition=optimal_condition
        )

        assert plan.split_orders
        assert plan.num_splits > 1

    def test_estimate_slippage(self, reducer: SlippageReducer) -> None:
        """Test slippage estimation."""
        slippage_optimal = reducer._estimate_slippage(
            ExecutionTier.OPTIMAL, 0.5, 0.3
        )
        slippage_avoid = reducer._estimate_slippage(
            ExecutionTier.AVOID, 4.0, 2.0
        )

        assert slippage_optimal < slippage_avoid

    def test_record_execution(self, reducer: SlippageReducer) -> None:
        """Test execution recording."""
        result = ExecutionResult(
            success=True,
            executed_price=130.05,
            requested_price=130.00,
            slippage_pips=0.5,
            execution_time_ms=150,
            fill_rate=1.0,
            message="Success"
        )

        reducer.record_execution(result)
        stats = reducer.get_execution_statistics()

        assert stats["sample_size"] == 1
        assert stats["avg_slippage_pips"] == 0.5

    def test_should_execute_within_tolerance(
        self,
        reducer: SlippageReducer,
        optimal_condition: MarketCondition
    ) -> None:
        """Test execution decision within tolerance."""
        plan = reducer.create_execution_plan(
            symbol="USDJPY",
            is_buy=True,
            quantity=0.1,
            current_price=130.0,
            condition=optimal_condition
        )

        should_exec, reason = reducer.should_execute(plan, 130.0)
        assert should_exec

    def test_pip_value_calculation(self, reducer: SlippageReducer) -> None:
        """Test pip value calculation for different symbols."""
        assert reducer._get_pip_value("USDJPY", 130.0) == 0.01
        assert reducer._get_pip_value("EURUSD", 1.1) == 0.0001
        assert reducer._get_pip_value("GOLD", 1900.0) == 0.1
        assert reducer._get_pip_value("US30", 34000.0) == 1.0
