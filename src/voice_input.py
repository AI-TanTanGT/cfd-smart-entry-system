"""Voice input handler for CFD Smart Entry System.

This module implements voice recognition for hands-free
trading command input in Japanese (ja-JP) and English (en-US).

Supported commands:
- Buy/Long (買い/ロング or buy/long)
- Sell/Short (売り/ショート or sell/short)
- Close (決済 or close)
- Close All (全決済 or close all)
- Status (状況 or status)
- Stop (停止 or stop)

Supported symbols:
- USDJPY (ドル円, dollar yen)
- EURUSD (ユーロドル, euro dollar)
- GBPUSD (ポンドドル, pound dollar)
- GOLD (ゴールド, gold)
- US30 (ダウ, dow)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from loguru import logger


class VoiceCommand(Enum):
    """Enumeration for recognized voice commands."""

    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"
    CLOSE_ALL = "close_all"
    STATUS = "status"
    STOP = "stop"
    UNKNOWN = "unknown"


@dataclass
class VoiceCommandResult:
    """Result of voice command recognition."""

    command: VoiceCommand
    raw_text: str
    confidence: float
    symbol: str | None
    quantity: float | None
    parameters: dict[str, Any]


class VoiceInputHandler:
    """Handler for voice-based trading commands."""

    def __init__(
        self,
        language: str = "ja-JP",
        enabled: bool = True
    ) -> None:
        """Initialize voice input handler.

        Args:
            language: Primary language for recognition ('ja-JP' or 'en-US').
            enabled: Whether voice input is enabled.
        """
        self.language = language
        self.enabled = enabled
        self._recognizer: Any = None
        self._microphone: Any = None
        self._is_initialized = False
        self._callbacks: dict[VoiceCommand, list[Callable]] = {}

        # Command patterns for different languages
        self._command_patterns = {
            "ja-JP": {
                VoiceCommand.BUY: ["買い", "ロング", "買う", "バイ"],
                VoiceCommand.SELL: ["売り", "ショート", "売る", "セル"],
                VoiceCommand.CLOSE: ["決済", "クローズ", "閉じる"],
                VoiceCommand.CLOSE_ALL: ["全決済", "オールクローズ", "すべて決済"],
                VoiceCommand.STATUS: ["状況", "ステータス", "確認"],
                VoiceCommand.STOP: ["停止", "ストップ", "終了"],
            },
            "en-US": {
                VoiceCommand.BUY: ["buy", "long", "go long"],
                VoiceCommand.SELL: ["sell", "short", "go short"],
                VoiceCommand.CLOSE: ["close", "exit"],
                VoiceCommand.CLOSE_ALL: ["close all", "exit all"],
                VoiceCommand.STATUS: ["status", "check"],
                VoiceCommand.STOP: ["stop", "quit"],
            },
        }

        # Symbol patterns
        self._symbol_patterns = {
            "ja-JP": {
                "USDJPY": ["ドル円", "ドルえん", "usdjpy"],
                "EURUSD": ["ユーロドル", "eurusd"],
                "GBPUSD": ["ポンドドル", "gbpusd"],
                "GOLD": ["ゴールド", "金", "gold"],
                "US30": ["ダウ", "us30"],
            },
            "en-US": {
                "USDJPY": ["dollar yen", "usdjpy", "usd jpy"],
                "EURUSD": ["euro dollar", "eurusd", "eur usd"],
                "GBPUSD": ["pound dollar", "gbpusd", "gbp usd"],
                "GOLD": ["gold", "xauusd"],
                "US30": ["dow", "us30", "dow jones"],
            },
        }

        # Quantity patterns
        self._quantity_patterns = {
            "ja-JP": {
                0.01: ["最小", "ミニ"],
                0.1: ["0.1ロット", "ゼロテンロット"],
                1.0: ["1ロット", "いちロット", "ワンロット"],
            },
            "en-US": {
                0.01: ["minimum", "mini", "micro"],
                0.1: ["point one lot", "0.1 lot"],
                1.0: ["one lot", "1 lot", "standard"],
            },
        }

        logger.info(f"VoiceInputHandler initialized with language={language}")

    def initialize(self) -> bool:
        """Initialize speech recognition system.

        Returns:
            True if initialization successful.
        """
        if not self.enabled:
            logger.warning("Voice input is disabled")
            return False

        try:
            import speech_recognition as sr

            self._recognizer = sr.Recognizer()
            self._microphone = sr.Microphone()

            # Adjust for ambient noise
            with self._microphone as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=1)

            self._is_initialized = True
            logger.info("Voice recognition system initialized successfully")
            return True

        except ImportError:
            logger.error("speech_recognition package not installed")
            return False
        except OSError as e:
            logger.error(f"Microphone initialization failed: {e}")
            return False

    def listen(self, timeout: float = 5.0) -> VoiceCommandResult | None:
        """Listen for voice command.

        Args:
            timeout: Maximum time to wait for speech in seconds.

        Returns:
            VoiceCommandResult if command recognized, None otherwise.
        """
        if not self._is_initialized:
            if not self.initialize():
                return None

        try:
            import speech_recognition as sr

            with self._microphone as source:
                logger.debug("Listening for voice command...")
                audio = self._recognizer.listen(source, timeout=timeout)

            # Try recognition
            text = self._recognize_speech(audio)
            if text:
                return self._parse_command(text)

            return None

        except Exception as e:
            logger.error(f"Voice recognition error: {e}")
            return None

    def _recognize_speech(self, audio: Any) -> str | None:
        """Recognize speech from audio.

        Args:
            audio: Audio data from microphone.

        Returns:
            Recognized text or None.
        """
        try:
            import speech_recognition as sr

            # Try Google Speech Recognition
            if self.language == "ja-JP":
                text = self._recognizer.recognize_google(audio, language="ja-JP")
            else:
                text = self._recognizer.recognize_google(audio, language="en-US")

            logger.debug(f"Recognized text: {text}")
            return text.lower()

        except Exception as e:
            logger.debug(f"Speech recognition failed: {e}")
            return None

    def _parse_command(self, text: str) -> VoiceCommandResult:
        """Parse recognized text into command.

        Args:
            text: Recognized text.

        Returns:
            VoiceCommandResult with parsed command.
        """
        text_lower = text.lower()

        # Detect command
        command = self._detect_command(text_lower)

        # Detect symbol
        symbol = self._detect_symbol(text_lower)

        # Detect quantity
        quantity = self._detect_quantity(text_lower)

        # Calculate confidence based on matches
        confidence = self._calculate_confidence(command, symbol, quantity)

        result = VoiceCommandResult(
            command=command,
            raw_text=text,
            confidence=confidence,
            symbol=symbol,
            quantity=quantity,
            parameters={}
        )

        logger.info(
            f"Voice command parsed: {command.value} "
            f"(symbol={symbol}, qty={quantity}, conf={confidence:.2%})"
        )

        return result

    def _detect_command(self, text: str) -> VoiceCommand:
        """Detect command from text.

        Args:
            text: Recognized text (lowercase).

        Returns:
            Detected VoiceCommand.
        """
        patterns = self._command_patterns.get(self.language, self._command_patterns["en-US"])

        # Check commands in priority order (more specific commands first)
        priority_order = [
            VoiceCommand.CLOSE_ALL,  # Check "全決済" before "決済"
            VoiceCommand.CLOSE,
            VoiceCommand.BUY,
            VoiceCommand.SELL,
            VoiceCommand.STATUS,
            VoiceCommand.STOP,
        ]

        for command in priority_order:
            keywords = patterns.get(command, [])
            for keyword in keywords:
                if keyword.lower() in text:
                    return command

        # Try other language as fallback
        fallback_lang = "en-US" if self.language == "ja-JP" else "ja-JP"
        patterns = self._command_patterns.get(fallback_lang, {})

        for command in priority_order:
            keywords = patterns.get(command, [])
            for keyword in keywords:
                if keyword.lower() in text:
                    return command

        return VoiceCommand.UNKNOWN

    def _detect_symbol(self, text: str) -> str | None:
        """Detect trading symbol from text.

        Args:
            text: Recognized text (lowercase).

        Returns:
            Detected symbol or None.
        """
        patterns = self._symbol_patterns.get(self.language, self._symbol_patterns["en-US"])

        for symbol, keywords in patterns.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    return symbol

        return None

    def _detect_quantity(self, text: str) -> float | None:
        """Detect order quantity from text.

        Args:
            text: Recognized text (lowercase).

        Returns:
            Detected quantity or None.
        """
        patterns = self._quantity_patterns.get(self.language, self._quantity_patterns["en-US"])

        for quantity, keywords in patterns.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    return quantity

        # Try to extract number
        import re
        numbers = re.findall(r'(\d+\.?\d*)\s*(?:ロット|lot)', text)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                pass

        return None

    def _calculate_confidence(
        self,
        command: VoiceCommand,
        symbol: str | None,
        quantity: float | None
    ) -> float:
        """Calculate confidence score for recognition.

        Args:
            command: Detected command.
            symbol: Detected symbol.
            quantity: Detected quantity.

        Returns:
            Confidence score (0-1).
        """
        if command == VoiceCommand.UNKNOWN:
            return 0.0

        confidence = 0.5  # Base confidence for recognized command

        if symbol is not None:
            confidence += 0.25

        if quantity is not None:
            confidence += 0.25

        return min(confidence, 1.0)

    def register_callback(
        self,
        command: VoiceCommand,
        callback: Callable[[VoiceCommandResult], None]
    ) -> None:
        """Register callback for command.

        Args:
            command: Command to register callback for.
            callback: Callback function.
        """
        if command not in self._callbacks:
            self._callbacks[command] = []

        self._callbacks[command].append(callback)
        logger.debug(f"Callback registered for {command.value}")

    def process_command(self, result: VoiceCommandResult) -> bool:
        """Process recognized command and trigger callbacks.

        Args:
            result: Voice command result.

        Returns:
            True if command was processed.
        """
        callbacks = self._callbacks.get(result.command, [])

        if not callbacks:
            logger.debug(f"No callbacks for {result.command.value}")
            return False

        for callback in callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Callback error: {e}")

        return True

    def parse_text_command(self, text: str) -> VoiceCommandResult:
        """Parse text command (for testing or text input).

        Args:
            text: Command text.

        Returns:
            VoiceCommandResult.
        """
        return self._parse_command(text.lower())

    def set_language(self, language: str) -> None:
        """Set recognition language.

        Args:
            language: Language code ('ja-JP' or 'en-US').
        """
        if language in self._command_patterns:
            self.language = language
            logger.info(f"Voice language set to {language}")
        else:
            logger.warning(f"Unsupported language: {language}")

    def get_supported_commands(self) -> dict[str, list[str]]:
        """Get list of supported commands and keywords.

        Returns:
            Dictionary mapping commands to keywords.
        """
        patterns = self._command_patterns.get(self.language, self._command_patterns["en-US"])
        return {cmd.value: keywords for cmd, keywords in patterns.items()}

    def get_supported_symbols(self) -> dict[str, list[str]]:
        """Get list of supported symbols and keywords.

        Returns:
            Dictionary mapping symbols to keywords.
        """
        return self._symbol_patterns.get(self.language, self._symbol_patterns["en-US"])
