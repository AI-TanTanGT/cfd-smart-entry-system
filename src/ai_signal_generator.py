"""AI-based signal generator for CFD Smart Entry System.

This module implements multi-tier AI analysis for generating
trading signals with confidence scoring.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from .market_analyzer import MarketAnalyzer, MarketCondition, TrendDirection


class SignalType(Enum):
    """Enumeration for signal types."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class SignalTier(Enum):
    """Enumeration for signal quality tiers."""

    TIER_1 = "tier_1"  # Highest confidence (>80%)
    TIER_2 = "tier_2"  # High confidence (65-80%)
    TIER_3 = "tier_3"  # Medium confidence (50-65%)
    NO_SIGNAL = "no_signal"  # Below threshold


@dataclass
class TradingSignal:
    """Data class representing a trading signal."""

    symbol: str
    signal_type: SignalType
    tier: SignalTier
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    features: dict[str, float]
    timestamp: pd.Timestamp
    reasoning: str


class AISignalGenerator:
    """Multi-tier AI signal generator for trading decisions."""

    def __init__(
        self,
        confidence_threshold: float = 0.65,
        model_type: str = "ensemble"
    ) -> None:
        """Initialize AI signal generator.

        Args:
            confidence_threshold: Minimum confidence for generating signals.
            model_type: Type of model ('ensemble', 'random_forest', 'gradient_boosting').
        """
        self.confidence_threshold = confidence_threshold
        self.model_type = model_type
        self.market_analyzer = MarketAnalyzer()
        self.scaler = StandardScaler()
        self._models: dict[str, Any] = {}
        self._is_trained = False

        logger.info(
            f"AISignalGenerator initialized with threshold={confidence_threshold}, "
            f"model_type={model_type}"
        )

    def initialize_models(self) -> None:
        """Initialize ML models for signal generation."""
        self._models = {
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
        }
        logger.info("ML models initialized")

    def train(
        self,
        training_data: pd.DataFrame,
        labels: np.ndarray
    ) -> dict[str, float]:
        """Train the AI models with historical data.

        Args:
            training_data: DataFrame with feature columns.
            labels: Array of labels (1=buy, 0=hold, -1=sell).

        Returns:
            Dictionary with training metrics.
        """
        if not self._models:
            self.initialize_models()

        # Prepare features
        X = training_data.values
        y = labels

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train models
        metrics: dict[str, float] = {}
        for name, model in self._models.items():
            model.fit(X_scaled, y)
            score = model.score(X_scaled, y)
            metrics[f"{name}_train_accuracy"] = score
            logger.info(f"Model {name} trained with accuracy: {score:.4f}")

        self._is_trained = True
        return metrics

    def generate_signal(
        self,
        symbol: str,
        ohlcv_data: pd.DataFrame,
        current_price: float,
        current_spread: float = 0.0
    ) -> TradingSignal:
        """Generate trading signal for a symbol.

        Args:
            symbol: Trading symbol.
            ohlcv_data: Historical OHLCV data.
            current_price: Current market price.
            current_spread: Current bid-ask spread.

        Returns:
            TradingSignal object with recommendation.
        """
        # Analyze market
        condition = self.market_analyzer.analyze(symbol, ohlcv_data, current_spread)
        features = self.market_analyzer.get_features(ohlcv_data)

        # Generate signal using rule-based + AI approach
        if self._is_trained:
            signal_type, confidence = self._predict_with_models(features)
        else:
            signal_type, confidence = self._rule_based_signal(condition, features)

        # Determine tier
        tier = self._determine_tier(confidence)

        # Calculate entry levels
        atr = self.market_analyzer.calculate_atr(ohlcv_data)
        stop_loss, take_profit = self._calculate_levels(
            signal_type, current_price, atr
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(condition, features, signal_type, confidence)

        signal = TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            tier=tier,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            features=features,
            timestamp=pd.Timestamp.now(),
            reasoning=reasoning
        )

        logger.info(
            f"Signal generated for {symbol}: {signal_type.value} "
            f"(tier={tier.value}, confidence={confidence:.2%})"
        )

        return signal

    def _predict_with_models(
        self,
        features: dict[str, float]
    ) -> tuple[SignalType, float]:
        """Predict signal using trained models.

        Args:
            features: Feature dictionary.

        Returns:
            Tuple of (SignalType, confidence).
        """
        X = np.array(list(features.values())).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        predictions = []
        probabilities = []

        for name, model in self._models.items():
            pred = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
            predictions.append(pred)
            probabilities.append(np.max(proba))

        # Ensemble voting
        if self.model_type == "ensemble":
            avg_pred = np.mean(predictions)
            confidence = np.mean(probabilities)
        else:
            model = self._models.get(self.model_type, list(self._models.values())[0])
            avg_pred = predictions[0]
            confidence = probabilities[0]

        # Convert prediction to signal type
        if avg_pred > 0.5:
            signal_type = SignalType.BUY
        elif avg_pred < -0.5:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD

        return signal_type, float(confidence)

    def _rule_based_signal(
        self,
        condition: MarketCondition,
        features: dict[str, float]
    ) -> tuple[SignalType, float]:
        """Generate signal using rule-based approach.

        Args:
            condition: Current market condition.
            features: Feature dictionary.

        Returns:
            Tuple of (SignalType, confidence).
        """
        score = 0.0
        weight_sum = 0.0

        # Trend analysis (weight: 30%)
        trend_weight = 0.30
        if condition.trend == TrendDirection.BULLISH:
            score += trend_weight * 1.0
        elif condition.trend == TrendDirection.BEARISH:
            score += trend_weight * -1.0
        weight_sum += trend_weight

        # Momentum analysis (weight: 25%)
        momentum_weight = 0.25
        momentum = features.get("momentum_short", 0)
        normalized_momentum = np.clip(momentum / 100, -1, 1)
        score += momentum_weight * normalized_momentum
        weight_sum += momentum_weight

        # Price position analysis (weight: 20%)
        position_weight = 0.20
        position = features.get("price_position", 0.5)
        # Near support = bullish, near resistance = bearish
        position_signal = (0.5 - position) * 2  # -1 to 1
        score += position_weight * position_signal
        weight_sum += position_weight

        # Volume analysis (weight: 15%)
        volume_weight = 0.15
        volume_ratio = features.get("volume_ratio", 1.0)
        if volume_ratio > 1.5:
            # High volume confirms trend
            score *= 1.1
        weight_sum += volume_weight

        # Volatility filter (weight: 10%)
        volatility_weight = 0.10
        volatility = condition.volatility
        if volatility > 50:  # High volatility
            score *= 0.8  # Reduce confidence
        weight_sum += volatility_weight

        # Normalize score to [-1, 1]
        normalized_score = score / weight_sum if weight_sum > 0 else 0

        # Calculate confidence
        confidence = min(abs(normalized_score) * 0.8 + 0.2, 0.95)

        # Determine signal type
        if normalized_score > 0.2:
            signal_type = SignalType.BUY
        elif normalized_score < -0.2:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD
            confidence = 0.3  # Low confidence for hold

        return signal_type, confidence

    def _determine_tier(self, confidence: float) -> SignalTier:
        """Determine signal tier based on confidence.

        Args:
            confidence: Confidence score (0-1).

        Returns:
            SignalTier enum value.
        """
        if confidence < self.confidence_threshold:
            return SignalTier.NO_SIGNAL
        elif confidence >= 0.80:
            return SignalTier.TIER_1
        elif confidence >= 0.65:
            return SignalTier.TIER_2
        else:
            return SignalTier.TIER_3

    def _calculate_levels(
        self,
        signal_type: SignalType,
        entry_price: float,
        atr: float
    ) -> tuple[float, float]:
        """Calculate stop loss and take profit levels.

        Args:
            signal_type: Type of signal.
            entry_price: Entry price.
            atr: Average True Range.

        Returns:
            Tuple of (stop_loss, take_profit).
        """
        # Use 1.5x ATR for stop loss, 2.5x ATR for take profit
        sl_distance = atr * 1.5 if atr > 0 else entry_price * 0.01
        tp_distance = atr * 2.5 if atr > 0 else entry_price * 0.02

        if signal_type == SignalType.BUY:
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        elif signal_type == SignalType.SELL:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
        else:
            stop_loss = entry_price
            take_profit = entry_price

        return stop_loss, take_profit

    def _generate_reasoning(
        self,
        condition: MarketCondition,
        features: dict[str, float],
        signal_type: SignalType,
        confidence: float
    ) -> str:
        """Generate human-readable reasoning for the signal.

        Args:
            condition: Market condition.
            features: Feature dictionary.
            signal_type: Generated signal type.
            confidence: Confidence score.

        Returns:
            Reasoning string.
        """
        reasons = []

        # Trend reasoning
        trend_str = condition.trend.value
        reasons.append(f"Trend: {trend_str}")

        # Momentum reasoning
        momentum = features.get("momentum_short", 0)
        if momentum > 20:
            reasons.append("Strong bullish momentum")
        elif momentum < -20:
            reasons.append("Strong bearish momentum")
        else:
            reasons.append("Neutral momentum")

        # Volatility reasoning
        if condition.volatility > 30:
            reasons.append(f"High volatility ({condition.volatility:.1f}%)")
        elif condition.volatility < 10:
            reasons.append(f"Low volatility ({condition.volatility:.1f}%)")

        # Volume reasoning
        volume_ratio = features.get("volume_ratio", 1.0)
        if volume_ratio > 1.5:
            reasons.append("Above average volume")
        elif volume_ratio < 0.5:
            reasons.append("Below average volume")

        reasoning = f"Signal: {signal_type.value.upper()} (Confidence: {confidence:.1%}). "
        reasoning += " | ".join(reasons)

        return reasoning

    def batch_generate_signals(
        self,
        symbols: list[str],
        data_dict: dict[str, pd.DataFrame],
        prices: dict[str, float],
        spreads: dict[str, float] | None = None
    ) -> list[TradingSignal]:
        """Generate signals for multiple symbols.

        Args:
            symbols: List of trading symbols.
            data_dict: Dictionary mapping symbols to OHLCV DataFrames.
            prices: Dictionary mapping symbols to current prices.
            spreads: Optional dictionary mapping symbols to spreads.

        Returns:
            List of TradingSignal objects.
        """
        signals = []
        spreads = spreads or {}

        for symbol in symbols:
            if symbol not in data_dict or symbol not in prices:
                logger.warning(f"Missing data for {symbol}, skipping")
                continue

            spread = spreads.get(symbol, 0.0)
            signal = self.generate_signal(
                symbol=symbol,
                ohlcv_data=data_dict[symbol],
                current_price=prices[symbol],
                current_spread=spread
            )
            signals.append(signal)

        # Sort by confidence
        signals.sort(key=lambda s: s.confidence, reverse=True)

        return signals
