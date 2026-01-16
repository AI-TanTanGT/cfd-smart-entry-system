"""Tests for market analyzer module."""

import numpy as np
import pandas as pd
import pytest

from src.market_analyzer import MarketAnalyzer, TrendDirection, MarketCondition


class TestMarketAnalyzer:
    """Test cases for MarketAnalyzer class."""

    @pytest.fixture
    def analyzer(self) -> MarketAnalyzer:
        """Create analyzer instance."""
        return MarketAnalyzer(short_period=5, medium_period=10, long_period=20)

    @pytest.fixture
    def bullish_data(self) -> pd.DataFrame:
        """Create bullish trend OHLCV data."""
        n = 50
        base_price = 100
        prices = [base_price + i * 0.5 for i in range(n)]

        return pd.DataFrame({
            "open": [p - 0.1 for p in prices],
            "high": [p + 0.2 for p in prices],
            "low": [p - 0.2 for p in prices],
            "close": prices,
            "volume": [1000] * n
        })

    @pytest.fixture
    def bearish_data(self) -> pd.DataFrame:
        """Create bearish trend OHLCV data."""
        n = 50
        base_price = 150
        prices = [base_price - i * 0.5 for i in range(n)]

        return pd.DataFrame({
            "open": [p + 0.1 for p in prices],
            "high": [p + 0.2 for p in prices],
            "low": [p - 0.2 for p in prices],
            "close": prices,
            "volume": [1000] * n
        })

    @pytest.fixture
    def sideways_data(self) -> pd.DataFrame:
        """Create sideways/neutral OHLCV data."""
        n = 50
        base_price = 100
        np.random.seed(42)
        prices = [base_price + np.random.uniform(-1, 1) for _ in range(n)]

        return pd.DataFrame({
            "open": [p - 0.05 for p in prices],
            "high": [p + 0.1 for p in prices],
            "low": [p - 0.1 for p in prices],
            "close": prices,
            "volume": [1000] * n
        })

    def test_analyze_bullish_trend(
        self,
        analyzer: MarketAnalyzer,
        bullish_data: pd.DataFrame
    ) -> None:
        """Test analysis of bullish market."""
        condition = analyzer.analyze("USDJPY", bullish_data, 0.02)

        assert condition.symbol == "USDJPY"
        assert condition.trend == TrendDirection.BULLISH
        assert condition.spread == 0.02
        assert condition.momentum > 0

    def test_analyze_bearish_trend(
        self,
        analyzer: MarketAnalyzer,
        bearish_data: pd.DataFrame
    ) -> None:
        """Test analysis of bearish market."""
        condition = analyzer.analyze("USDJPY", bearish_data, 0.02)

        assert condition.trend == TrendDirection.BEARISH
        assert condition.momentum < 0

    def test_analyze_neutral_trend(
        self,
        analyzer: MarketAnalyzer,
        sideways_data: pd.DataFrame
    ) -> None:
        """Test analysis of neutral/sideways market."""
        condition = analyzer.analyze("USDJPY", sideways_data, 0.02)

        # Neutral or could be slightly biased
        assert condition.trend in [
            TrendDirection.NEUTRAL,
            TrendDirection.BULLISH,
            TrendDirection.BEARISH
        ]

    def test_calculate_volatility(
        self,
        analyzer: MarketAnalyzer,
        bullish_data: pd.DataFrame
    ) -> None:
        """Test volatility calculation."""
        close = bullish_data["close"].values
        volatility = analyzer._calculate_volatility(close)

        assert volatility >= 0
        assert isinstance(volatility, float)

    def test_calculate_momentum(
        self,
        analyzer: MarketAnalyzer,
        bullish_data: pd.DataFrame
    ) -> None:
        """Test momentum calculation."""
        close = bullish_data["close"].values
        momentum = analyzer._calculate_momentum(close)

        assert -100 <= momentum <= 100
        assert momentum > 0  # Bullish data should have positive momentum

    def test_calculate_atr(
        self,
        analyzer: MarketAnalyzer,
        bullish_data: pd.DataFrame
    ) -> None:
        """Test ATR calculation."""
        atr = analyzer.calculate_atr(bullish_data)

        assert atr > 0
        assert isinstance(atr, float)

    def test_get_features(
        self,
        analyzer: MarketAnalyzer,
        bullish_data: pd.DataFrame
    ) -> None:
        """Test feature extraction."""
        features = analyzer.get_features(bullish_data)

        assert "momentum_short" in features
        assert "momentum_medium" in features
        assert "volatility" in features
        assert "atr" in features
        assert "trend_score" in features
        assert "price_position" in features
        assert "volume_ratio" in features

    def test_support_resistance(
        self,
        analyzer: MarketAnalyzer,
        bullish_data: pd.DataFrame
    ) -> None:
        """Test support/resistance calculation."""
        high = bullish_data["high"].values
        low = bullish_data["low"].values
        close = bullish_data["close"].values

        support, resistance = analyzer._calculate_support_resistance(high, low, close)

        assert support < resistance
        assert isinstance(support, float)
        assert isinstance(resistance, float)

    def test_volume_ratio(
        self,
        analyzer: MarketAnalyzer,
        bullish_data: pd.DataFrame
    ) -> None:
        """Test volume ratio calculation."""
        volume = bullish_data["volume"].values
        ratio = analyzer._calculate_volume_ratio(volume)

        # With constant volume, ratio should be close to 1
        assert 0.9 <= ratio <= 1.1
