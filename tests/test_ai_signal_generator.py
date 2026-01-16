"""Tests for AI signal generator module."""

import numpy as np
import pandas as pd
import pytest

from src.ai_signal_generator import (
    AISignalGenerator,
    SignalType,
    SignalTier,
    TradingSignal
)
from src.market_analyzer import TrendDirection


class TestAISignalGenerator:
    """Test cases for AISignalGenerator class."""

    @pytest.fixture
    def generator(self) -> AISignalGenerator:
        """Create signal generator instance."""
        return AISignalGenerator(confidence_threshold=0.65)

    @pytest.fixture
    def bullish_data(self) -> pd.DataFrame:
        """Create bullish trend OHLCV data."""
        n = 100
        base_price = 100
        prices = [base_price + i * 0.3 for i in range(n)]

        return pd.DataFrame({
            "open": [p - 0.1 for p in prices],
            "high": [p + 0.2 for p in prices],
            "low": [p - 0.2 for p in prices],
            "close": prices,
            "volume": [1000 + i * 10 for i in range(n)]
        })

    @pytest.fixture
    def bearish_data(self) -> pd.DataFrame:
        """Create bearish trend OHLCV data."""
        n = 100
        base_price = 150
        prices = [base_price - i * 0.3 for i in range(n)]

        return pd.DataFrame({
            "open": [p + 0.1 for p in prices],
            "high": [p + 0.2 for p in prices],
            "low": [p - 0.2 for p in prices],
            "close": prices,
            "volume": [1000 + i * 10 for i in range(n)]
        })

    def test_generate_signal_bullish(
        self,
        generator: AISignalGenerator,
        bullish_data: pd.DataFrame
    ) -> None:
        """Test signal generation for bullish market."""
        signal = generator.generate_signal(
            symbol="USDJPY",
            ohlcv_data=bullish_data,
            current_price=130.0,
            current_spread=0.02
        )

        assert isinstance(signal, TradingSignal)
        assert signal.symbol == "USDJPY"
        assert signal.signal_type in [SignalType.BUY, SignalType.HOLD]
        assert 0 <= signal.confidence <= 1
        assert signal.entry_price == 130.0

    def test_generate_signal_bearish(
        self,
        generator: AISignalGenerator,
        bearish_data: pd.DataFrame
    ) -> None:
        """Test signal generation for bearish market."""
        signal = generator.generate_signal(
            symbol="USDJPY",
            ohlcv_data=bearish_data,
            current_price=120.0,
            current_spread=0.02
        )

        assert isinstance(signal, TradingSignal)
        assert signal.signal_type in [SignalType.SELL, SignalType.HOLD]

    def test_signal_tiers(self, generator: AISignalGenerator) -> None:
        """Test signal tier determination."""
        # Generator is created with confidence_threshold=0.65
        # Tier 1: >= 0.80
        # Tier 2: >= 0.65 and < 0.80
        # No Signal: < 0.65 (below threshold)
        assert generator._determine_tier(0.90) == SignalTier.TIER_1
        assert generator._determine_tier(0.75) == SignalTier.TIER_2
        assert generator._determine_tier(0.70) == SignalTier.TIER_2  # Above threshold
        assert generator._determine_tier(0.50) == SignalTier.NO_SIGNAL  # Below threshold

    def test_calculate_levels(self, generator: AISignalGenerator) -> None:
        """Test stop loss and take profit calculation."""
        # BUY signal
        sl, tp = generator._calculate_levels(SignalType.BUY, 100.0, 1.0)
        assert sl < 100.0  # Stop loss below entry for buy
        assert tp > 100.0  # Take profit above entry for buy

        # SELL signal
        sl, tp = generator._calculate_levels(SignalType.SELL, 100.0, 1.0)
        assert sl > 100.0  # Stop loss above entry for sell
        assert tp < 100.0  # Take profit below entry for sell

    def test_generate_reasoning(
        self,
        generator: AISignalGenerator,
        bullish_data: pd.DataFrame
    ) -> None:
        """Test reasoning generation."""
        signal = generator.generate_signal(
            symbol="USDJPY",
            ohlcv_data=bullish_data,
            current_price=130.0
        )

        assert signal.reasoning is not None
        assert len(signal.reasoning) > 0
        assert "Signal:" in signal.reasoning

    def test_batch_generate_signals(
        self,
        generator: AISignalGenerator,
        bullish_data: pd.DataFrame,
        bearish_data: pd.DataFrame
    ) -> None:
        """Test batch signal generation."""
        data_dict = {
            "USDJPY": bullish_data,
            "EURUSD": bearish_data
        }
        prices = {"USDJPY": 130.0, "EURUSD": 1.1000}

        signals = generator.batch_generate_signals(
            symbols=["USDJPY", "EURUSD"],
            data_dict=data_dict,
            prices=prices
        )

        assert len(signals) == 2
        assert all(isinstance(s, TradingSignal) for s in signals)
        # Should be sorted by confidence
        assert signals[0].confidence >= signals[1].confidence

    def test_initialize_models(self, generator: AISignalGenerator) -> None:
        """Test model initialization."""
        generator.initialize_models()

        assert "random_forest" in generator._models
        assert "gradient_boosting" in generator._models

    def test_features_used(
        self,
        generator: AISignalGenerator,
        bullish_data: pd.DataFrame
    ) -> None:
        """Test that features are properly extracted and used."""
        signal = generator.generate_signal(
            symbol="USDJPY",
            ohlcv_data=bullish_data,
            current_price=130.0
        )

        assert "momentum_short" in signal.features
        assert "volatility" in signal.features
        assert "trend_score" in signal.features
