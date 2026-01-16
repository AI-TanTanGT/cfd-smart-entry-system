"""Market data analyzer for CFD Smart Entry System.

This module provides technical analysis and market data processing
for generating trading signals.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


class TrendDirection(Enum):
    """Enumeration for trend direction."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class MarketCondition:
    """Data class representing current market condition."""

    symbol: str
    trend: TrendDirection
    volatility: float
    spread: float
    momentum: float
    support_level: float
    resistance_level: float
    volume_ratio: float
    timestamp: pd.Timestamp


class MarketAnalyzer:
    """Analyzer for market data and technical indicators."""

    def __init__(
        self,
        short_period: int = 14,
        medium_period: int = 50,
        long_period: int = 200
    ) -> None:
        """Initialize market analyzer.

        Args:
            short_period: Short-term lookback period.
            medium_period: Medium-term lookback period.
            long_period: Long-term lookback period.
        """
        self.short_period = short_period
        self.medium_period = medium_period
        self.long_period = long_period
        logger.info(
            f"MarketAnalyzer initialized with periods: "
            f"short={short_period}, medium={medium_period}, long={long_period}"
        )

    def analyze(
        self,
        symbol: str,
        ohlcv_data: pd.DataFrame,
        current_spread: float = 0.0
    ) -> MarketCondition:
        """Analyze market data and return current condition.

        Args:
            symbol: Trading symbol.
            ohlcv_data: DataFrame with OHLCV data (columns: open, high, low, close, volume).
            current_spread: Current bid-ask spread in pips.

        Returns:
            MarketCondition object with analysis results.
        """
        if len(ohlcv_data) < self.long_period:
            logger.warning(
                f"Insufficient data for analysis: {len(ohlcv_data)} < {self.long_period}"
            )

        close = ohlcv_data["close"].values
        high = ohlcv_data["high"].values
        low = ohlcv_data["low"].values
        volume = ohlcv_data.get("volume", pd.Series([1.0] * len(close))).values

        trend = self._determine_trend(close)
        volatility = self._calculate_volatility(close)
        momentum = self._calculate_momentum(close)
        support, resistance = self._calculate_support_resistance(high, low, close)
        volume_ratio = self._calculate_volume_ratio(volume)

        condition = MarketCondition(
            symbol=symbol,
            trend=trend,
            volatility=volatility,
            spread=current_spread,
            momentum=momentum,
            support_level=support,
            resistance_level=resistance,
            volume_ratio=volume_ratio,
            timestamp=pd.Timestamp.now()
        )

        logger.debug(f"Market analysis for {symbol}: {trend.value}, volatility={volatility:.4f}")
        return condition

    def _determine_trend(self, close: np.ndarray) -> TrendDirection:
        """Determine trend direction based on moving averages.

        Args:
            close: Array of closing prices.

        Returns:
            TrendDirection enum value.
        """
        if len(close) < self.medium_period:
            return TrendDirection.NEUTRAL

        short_ma = np.mean(close[-self.short_period:])
        medium_ma = np.mean(close[-self.medium_period:])

        if len(close) >= self.long_period:
            long_ma = np.mean(close[-self.long_period:])
        else:
            long_ma = medium_ma

        if short_ma > medium_ma > long_ma:
            return TrendDirection.BULLISH
        elif short_ma < medium_ma < long_ma:
            return TrendDirection.BEARISH
        else:
            return TrendDirection.NEUTRAL

    def _calculate_volatility(self, close: np.ndarray) -> float:
        """Calculate volatility using standard deviation of returns.

        Args:
            close: Array of closing prices.

        Returns:
            Volatility as a percentage.
        """
        if len(close) < 2:
            return 0.0

        returns = np.diff(close) / close[:-1]
        period = min(len(returns), self.short_period)
        volatility = np.std(returns[-period:]) * np.sqrt(252) * 100

        return float(volatility)

    def _calculate_momentum(self, close: np.ndarray) -> float:
        """Calculate price momentum using RSI-like indicator.

        Args:
            close: Array of closing prices.

        Returns:
            Momentum value between -100 and 100.
        """
        period = min(len(close), self.short_period)
        if period < 2:
            return 0.0

        changes = np.diff(close[-period:])
        gains = np.sum(changes[changes > 0])
        losses = -np.sum(changes[changes < 0])

        if losses == 0:
            return 100.0
        if gains == 0:
            return -100.0

        rs = gains / losses
        rsi = 100 - (100 / (1 + rs))

        # Convert RSI (0-100) to momentum (-100 to 100)
        momentum = (rsi - 50) * 2

        return float(momentum)

    def _calculate_support_resistance(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> tuple[float, float]:
        """Calculate support and resistance levels.

        Args:
            high: Array of high prices.
            low: Array of low prices.
            close: Array of closing prices.

        Returns:
            Tuple of (support_level, resistance_level).
        """
        period = min(len(close), self.medium_period)

        recent_low = np.min(low[-period:])
        recent_high = np.max(high[-period:])

        # Use pivot point method
        typical_price = (high[-1] + low[-1] + close[-1]) / 3
        support = 2 * typical_price - recent_high
        resistance = 2 * typical_price - recent_low

        return float(support), float(resistance)

    def _calculate_volume_ratio(self, volume: np.ndarray) -> float:
        """Calculate volume ratio (current vs average).

        Args:
            volume: Array of volume data.

        Returns:
            Volume ratio (1.0 = average volume).
        """
        if len(volume) < 2:
            return 1.0

        period = min(len(volume), self.medium_period)
        avg_volume = np.mean(volume[-period:-1])

        if avg_volume == 0:
            return 1.0

        ratio = volume[-1] / avg_volume
        return float(ratio)

    def calculate_atr(self, ohlcv_data: pd.DataFrame, period: int | None = None) -> float:
        """Calculate Average True Range (ATR).

        Args:
            ohlcv_data: DataFrame with OHLCV data.
            period: ATR period (default: short_period).

        Returns:
            ATR value.
        """
        if period is None:
            period = self.short_period

        if len(ohlcv_data) < 2:
            return 0.0

        high = ohlcv_data["high"].values
        low = ohlcv_data["low"].values
        close = ohlcv_data["close"].values

        true_ranges = []
        for i in range(1, len(close)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )
            true_ranges.append(tr)

        if not true_ranges:
            return 0.0

        period = min(period, len(true_ranges))
        atr = np.mean(true_ranges[-period:])

        return float(atr)

    def get_features(self, ohlcv_data: pd.DataFrame) -> dict[str, float]:
        """Extract features for AI model input.

        Args:
            ohlcv_data: DataFrame with OHLCV data.

        Returns:
            Dictionary of feature names and values.
        """
        close = ohlcv_data["close"].values
        high = ohlcv_data["high"].values
        low = ohlcv_data["low"].values

        features: dict[str, float] = {}

        # Price momentum features
        features["momentum_short"] = self._calculate_momentum(close)

        if len(close) >= self.medium_period:
            features["momentum_medium"] = self._calculate_momentum(close[-self.medium_period:])
        else:
            features["momentum_medium"] = features["momentum_short"]

        # Volatility features
        features["volatility"] = self._calculate_volatility(close)
        features["atr"] = self.calculate_atr(ohlcv_data)

        # Trend features
        trend = self._determine_trend(close)
        features["trend_score"] = {
            TrendDirection.BULLISH: 1.0,
            TrendDirection.NEUTRAL: 0.0,
            TrendDirection.BEARISH: -1.0,
        }[trend]

        # Price position features
        if len(close) >= self.short_period:
            recent_high = np.max(high[-self.short_period:])
            recent_low = np.min(low[-self.short_period:])
            price_range = recent_high - recent_low

            if price_range > 0:
                features["price_position"] = (close[-1] - recent_low) / price_range
            else:
                features["price_position"] = 0.5
        else:
            features["price_position"] = 0.5

        # Volume features
        volume = ohlcv_data.get("volume", pd.Series([1.0] * len(close))).values
        features["volume_ratio"] = self._calculate_volume_ratio(volume)

        return features
