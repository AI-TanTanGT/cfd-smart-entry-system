"""Tests for voice input handler module."""

import pytest

from src.voice_input import (
    VoiceInputHandler,
    VoiceCommand,
    VoiceCommandResult
)


class TestVoiceInputHandler:
    """Test cases for VoiceInputHandler class."""

    @pytest.fixture
    def handler_ja(self) -> VoiceInputHandler:
        """Create Japanese language handler."""
        return VoiceInputHandler(language="ja-JP", enabled=False)

    @pytest.fixture
    def handler_en(self) -> VoiceInputHandler:
        """Create English language handler."""
        return VoiceInputHandler(language="en-US", enabled=False)

    def test_parse_buy_command_japanese(
        self,
        handler_ja: VoiceInputHandler
    ) -> None:
        """Test parsing Japanese buy command."""
        result = handler_ja.parse_text_command("ドル円を買い")

        assert result.command == VoiceCommand.BUY
        assert result.symbol == "USDJPY"

    def test_parse_sell_command_japanese(
        self,
        handler_ja: VoiceInputHandler
    ) -> None:
        """Test parsing Japanese sell command."""
        result = handler_ja.parse_text_command("ユーロドル売り")

        assert result.command == VoiceCommand.SELL
        assert result.symbol == "EURUSD"

    def test_parse_buy_command_english(
        self,
        handler_en: VoiceInputHandler
    ) -> None:
        """Test parsing English buy command."""
        result = handler_en.parse_text_command("buy dollar yen")

        assert result.command == VoiceCommand.BUY
        assert result.symbol == "USDJPY"

    def test_parse_sell_command_english(
        self,
        handler_en: VoiceInputHandler
    ) -> None:
        """Test parsing English sell command."""
        result = handler_en.parse_text_command("sell euro dollar")

        assert result.command == VoiceCommand.SELL
        assert result.symbol == "EURUSD"

    def test_parse_close_command(
        self,
        handler_ja: VoiceInputHandler
    ) -> None:
        """Test parsing close command."""
        result = handler_ja.parse_text_command("決済")

        assert result.command == VoiceCommand.CLOSE

    def test_parse_close_all_command(
        self,
        handler_ja: VoiceInputHandler
    ) -> None:
        """Test parsing close all command."""
        result = handler_ja.parse_text_command("全決済")

        assert result.command == VoiceCommand.CLOSE_ALL

    def test_parse_status_command(
        self,
        handler_ja: VoiceInputHandler
    ) -> None:
        """Test parsing status command."""
        result = handler_ja.parse_text_command("状況確認")

        assert result.command == VoiceCommand.STATUS

    def test_parse_quantity_japanese(
        self,
        handler_ja: VoiceInputHandler
    ) -> None:
        """Test parsing Japanese quantity."""
        result = handler_ja.parse_text_command("ドル円を1ロット買い")

        assert result.command == VoiceCommand.BUY
        assert result.symbol == "USDJPY"
        assert result.quantity == 1.0

    def test_parse_unknown_command(
        self,
        handler_ja: VoiceInputHandler
    ) -> None:
        """Test parsing unknown command."""
        result = handler_ja.parse_text_command("不明なコマンド")

        assert result.command == VoiceCommand.UNKNOWN
        assert result.confidence == 0.0

    def test_confidence_calculation(
        self,
        handler_ja: VoiceInputHandler
    ) -> None:
        """Test confidence calculation."""
        # Full match (command + symbol + quantity)
        result_full = handler_ja.parse_text_command("ドル円を1ロット買い")
        assert result_full.confidence == 1.0

        # Partial match (command only)
        result_partial = handler_ja.parse_text_command("買い")
        assert result_partial.confidence == 0.5

    def test_callback_registration(
        self,
        handler_ja: VoiceInputHandler
    ) -> None:
        """Test callback registration."""
        callback_called = [False]

        def test_callback(result: VoiceCommandResult) -> None:
            callback_called[0] = True

        handler_ja.register_callback(VoiceCommand.BUY, test_callback)

        result = handler_ja.parse_text_command("買い")
        handler_ja.process_command(result)

        assert callback_called[0]

    def test_set_language(
        self,
        handler_ja: VoiceInputHandler
    ) -> None:
        """Test language switching."""
        handler_ja.set_language("en-US")
        assert handler_ja.language == "en-US"

    def test_get_supported_commands(
        self,
        handler_ja: VoiceInputHandler
    ) -> None:
        """Test getting supported commands."""
        commands = handler_ja.get_supported_commands()

        assert "buy" in commands
        assert "sell" in commands
        assert "close" in commands

    def test_get_supported_symbols(
        self,
        handler_ja: VoiceInputHandler
    ) -> None:
        """Test getting supported symbols."""
        symbols = handler_ja.get_supported_symbols()

        assert "USDJPY" in symbols
        assert "EURUSD" in symbols
        assert "GOLD" in symbols

    def test_gold_symbol_recognition(
        self,
        handler_ja: VoiceInputHandler
    ) -> None:
        """Test gold symbol recognition."""
        result = handler_ja.parse_text_command("ゴールド買い")

        assert result.command == VoiceCommand.BUY
        assert result.symbol == "GOLD"
