"""Order execution module for CFD Smart Entry System.

This module orchestrates the order execution process,
combining AI signals, slippage reduction, and MT5 integration.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd
from loguru import logger

from .ai_signal_generator import AISignalGenerator, SignalType, SignalTier, TradingSignal
from .market_analyzer import MarketAnalyzer
from .mt5_connector import MT5Connector, MT5OrderType
from .slippage_reducer import (
    SlippageReducer,
    SlippageConfig,
    ExecutionPlan,
    ExecutionResult
)


class ExecutionStatus(Enum):
    """Enumeration for execution status."""

    PENDING = "pending"
    EXECUTED = "executed"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class OrderRequest:
    """Data class for order request."""

    symbol: str
    signal: TradingSignal
    volume: float
    execution_plan: ExecutionPlan


@dataclass
class OrderResult:
    """Data class for order execution result."""

    request: OrderRequest
    status: ExecutionStatus
    ticket: int | None
    executed_price: float
    executed_volume: float
    slippage_pips: float
    execution_time_ms: int
    message: str
    timestamp: datetime


class OrderExecutor:
    """Orchestrator for order execution with slippage reduction."""

    def __init__(
        self,
        mt5_connector: MT5Connector,
        signal_generator: AISignalGenerator,
        slippage_config: SlippageConfig | None = None
    ) -> None:
        """Initialize order executor.

        Args:
            mt5_connector: MT5 connector instance.
            signal_generator: AI signal generator instance.
            slippage_config: Slippage configuration.
        """
        self.mt5 = mt5_connector
        self.signal_generator = signal_generator
        self.slippage_reducer = SlippageReducer(slippage_config)
        self.market_analyzer = MarketAnalyzer()

        # Execution history
        self._order_history: list[OrderResult] = []

        # Risk parameters
        self.max_positions_per_symbol = 3
        self.max_daily_orders = 50
        self._daily_order_count = 0
        self._last_reset_date: datetime | None = None

        logger.info("OrderExecutor initialized")

    def execute_signal(
        self,
        signal: TradingSignal,
        volume: float | None = None
    ) -> OrderResult:
        """Execute trading signal.

        Args:
            signal: Trading signal to execute.
            volume: Order volume (uses default if None).

        Returns:
            OrderResult with execution details.
        """
        # Reset daily counter if needed
        self._check_daily_reset()

        # Validate signal
        if not self._validate_signal(signal):
            return self._create_failed_result(
                signal, volume or 0.01,
                "Signal validation failed"
            )

        # Check risk limits
        risk_check, risk_message = self._check_risk_limits(signal.symbol)
        if not risk_check:
            return self._create_failed_result(
                signal, volume or 0.01, risk_message
            )

        # Get market data
        ohlcv = self.mt5.get_ohlcv(signal.symbol)
        if ohlcv is None:
            return self._create_failed_result(
                signal, volume or 0.01,
                "Failed to get market data"
            )

        # Analyze current conditions
        tick = self.mt5.get_tick(signal.symbol)
        if tick is None:
            return self._create_failed_result(
                signal, volume or 0.01,
                "Failed to get tick data"
            )

        current_price = tick.bid if signal.signal_type == SignalType.SELL else tick.ask
        condition = self.market_analyzer.analyze(signal.symbol, ohlcv, tick.spread)

        # Create execution plan
        is_buy = signal.signal_type == SignalType.BUY
        execution_plan = self.slippage_reducer.create_execution_plan(
            symbol=signal.symbol,
            is_buy=is_buy,
            quantity=volume or 0.01,
            current_price=current_price,
            condition=condition
        )

        # Create order request
        request = OrderRequest(
            symbol=signal.symbol,
            signal=signal,
            volume=volume or 0.01,
            execution_plan=execution_plan
        )

        # Check if should execute
        should_exec, reason = self.slippage_reducer.should_execute(
            execution_plan, current_price
        )

        if not should_exec:
            logger.warning(f"Execution skipped: {reason}")
            return self._create_failed_result(
                signal, volume or 0.01, reason
            )

        # Execute order
        return self._execute_order(request, current_price)

    def _execute_order(
        self,
        request: OrderRequest,
        current_price: float
    ) -> OrderResult:
        """Execute the actual order.

        Args:
            request: Order request.
            current_price: Current market price.

        Returns:
            OrderResult with execution details.
        """
        start_time = datetime.now()

        # Determine order type
        if request.signal.signal_type == SignalType.BUY:
            order_type = MT5OrderType.BUY
        else:
            order_type = MT5OrderType.SELL

        # Execute order
        result = self.mt5.send_order(
            symbol=request.symbol,
            order_type=order_type,
            volume=request.volume,
            sl=request.signal.stop_loss,
            tp=request.signal.take_profit,
            comment=f"CFD-AI-{request.signal.tier.value}"
        )

        execution_time = int((datetime.now() - start_time).total_seconds() * 1000)

        if result["success"]:
            executed_price = result["price"]
            slippage = self._calculate_slippage(
                request.symbol,
                current_price,
                executed_price
            )

            # Record execution
            exec_result = ExecutionResult(
                success=True,
                executed_price=executed_price,
                requested_price=current_price,
                slippage_pips=slippage,
                execution_time_ms=execution_time,
                fill_rate=1.0,
                message="Order executed successfully"
            )
            self.slippage_reducer.record_execution(exec_result)

            order_result = OrderResult(
                request=request,
                status=ExecutionStatus.EXECUTED,
                ticket=result["ticket"],
                executed_price=executed_price,
                executed_volume=result["volume"],
                slippage_pips=slippage,
                execution_time_ms=execution_time,
                message="Order executed successfully",
                timestamp=datetime.now()
            )

            self._daily_order_count += 1
            self._order_history.append(order_result)

            logger.info(
                f"Order executed: {request.symbol} {order_type.name} "
                f"{request.volume} lots at {executed_price} "
                f"(slippage={slippage:.2f} pips)"
            )

            return order_result

        else:
            return self._create_failed_result(
                request.signal,
                request.volume,
                result.get("error", "Unknown error")
            )

    def close_position(self, ticket: int) -> dict[str, Any]:
        """Close position by ticket.

        Args:
            ticket: Position ticket number.

        Returns:
            Close result dictionary.
        """
        result = self.mt5.close_position(ticket)

        if result["success"]:
            logger.info(f"Position {ticket} closed with profit: {result['profit']}")
        else:
            logger.error(f"Failed to close position {ticket}: {result['error']}")

        return result

    def close_all_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Close all positions.

        Args:
            symbol: Close only positions for this symbol (optional).

        Returns:
            List of close results.
        """
        positions = self.mt5.get_positions(symbol)
        results = []

        for pos in positions:
            result = self.close_position(pos.ticket)
            results.append(result)

        logger.info(f"Closed {len(results)} positions")
        return results

    def _validate_signal(self, signal: TradingSignal) -> bool:
        """Validate trading signal.

        Args:
            signal: Signal to validate.

        Returns:
            True if signal is valid.
        """
        # Check signal type
        if signal.signal_type == SignalType.HOLD:
            logger.debug("Signal is HOLD - no action")
            return False

        # Check tier
        if signal.tier == SignalTier.NO_SIGNAL:
            logger.debug("Signal tier is NO_SIGNAL - below threshold")
            return False

        # Check confidence
        if signal.confidence < self.signal_generator.confidence_threshold:
            logger.debug(f"Signal confidence {signal.confidence} below threshold")
            return False

        return True

    def _check_risk_limits(self, symbol: str) -> tuple[bool, str]:
        """Check risk limits before execution.

        Args:
            symbol: Trading symbol.

        Returns:
            Tuple of (passed, message).
        """
        # Check daily order limit
        if self._daily_order_count >= self.max_daily_orders:
            return False, "Daily order limit reached"

        # Check position count
        position_count = self.mt5.get_symbol_positions_count(symbol)
        if position_count >= self.max_positions_per_symbol:
            return False, f"Max positions for {symbol} reached ({position_count})"

        return True, "Risk check passed"

    def _check_daily_reset(self) -> None:
        """Reset daily counter if new day."""
        today = datetime.now().date()

        if self._last_reset_date != today:
            self._daily_order_count = 0
            self._last_reset_date = today
            logger.debug("Daily order counter reset")

    def _calculate_slippage(
        self,
        symbol: str,
        requested_price: float,
        executed_price: float
    ) -> float:
        """Calculate slippage in pips.

        Args:
            symbol: Trading symbol.
            requested_price: Requested execution price.
            executed_price: Actual execution price.

        Returns:
            Slippage in pips.
        """
        price_diff = abs(executed_price - requested_price)

        # Determine pip value
        symbol_upper = symbol.upper()
        if "JPY" in symbol_upper:
            pip_value = 0.01
        elif symbol_upper in ["GOLD", "XAUUSD"]:
            pip_value = 0.1
        elif symbol_upper in ["US30", "DJ30"]:
            pip_value = 1.0
        else:
            pip_value = 0.0001

        slippage_pips = price_diff / pip_value
        return slippage_pips

    def _create_failed_result(
        self,
        signal: TradingSignal,
        volume: float,
        message: str
    ) -> OrderResult:
        """Create failed order result.

        Args:
            signal: Trading signal.
            volume: Requested volume.
            message: Failure message.

        Returns:
            OrderResult with failed status.
        """
        return OrderResult(
            request=OrderRequest(
                symbol=signal.symbol,
                signal=signal,
                volume=volume,
                execution_plan=None  # type: ignore
            ),
            status=ExecutionStatus.FAILED,
            ticket=None,
            executed_price=0.0,
            executed_volume=0.0,
            slippage_pips=0.0,
            execution_time_ms=0,
            message=message,
            timestamp=datetime.now()
        )

    def get_execution_statistics(self) -> dict[str, Any]:
        """Get execution statistics.

        Returns:
            Dictionary with statistics.
        """
        if not self._order_history:
            return {
                "total_orders": 0,
                "successful_orders": 0,
                "failed_orders": 0,
                "success_rate": 0.0,
                "avg_slippage_pips": 0.0,
                "avg_execution_time_ms": 0.0,
            }

        successful = [o for o in self._order_history if o.status == ExecutionStatus.EXECUTED]
        failed = [o for o in self._order_history if o.status == ExecutionStatus.FAILED]

        slippages = [o.slippage_pips for o in successful]
        exec_times = [o.execution_time_ms for o in successful]

        return {
            "total_orders": len(self._order_history),
            "successful_orders": len(successful),
            "failed_orders": len(failed),
            "success_rate": len(successful) / len(self._order_history) if self._order_history else 0,
            "avg_slippage_pips": sum(slippages) / len(slippages) if slippages else 0,
            "avg_execution_time_ms": sum(exec_times) / len(exec_times) if exec_times else 0,
            "daily_order_count": self._daily_order_count,
        }

    def get_order_history(self, limit: int = 50) -> list[OrderResult]:
        """Get recent order history.

        Args:
            limit: Maximum number of orders to return.

        Returns:
            List of recent OrderResult objects.
        """
        return self._order_history[-limit:]
