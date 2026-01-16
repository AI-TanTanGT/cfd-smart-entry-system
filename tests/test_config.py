"""Tests for configuration module."""

import os
import tempfile

import pytest
import yaml

from src.config import Config, get_config, reload_config


class TestConfig:
    """Test cases for Config class."""

    def test_default_config(self) -> None:
        """Test loading default configuration."""
        config = Config("/nonexistent/path.yaml")

        assert config.get("mt5.server") == "Demo-Server"
        assert config.get("trading.default_lot_size") == 0.01
        assert config.get("slippage.max_slippage_pips") == 3.0

    def test_load_config_from_file(self) -> None:
        """Test loading configuration from YAML file."""
        test_config = {
            "mt5": {"server": "Test-Server"},
            "trading": {"default_lot_size": 0.05},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_config, f)
            temp_path = f.name

        try:
            config = Config(temp_path)
            assert config.get("mt5.server") == "Test-Server"
            assert config.get("trading.default_lot_size") == 0.05
        finally:
            os.unlink(temp_path)

    def test_get_nested_value(self) -> None:
        """Test getting nested configuration values."""
        config = Config()

        assert config.get("mt5.timeout") == 60000
        symbols = config.get("trading.symbols")
        assert "USDJPY" in symbols
        assert "EURUSD" in symbols

    def test_get_with_default(self) -> None:
        """Test getting value with default."""
        config = Config()

        assert config.get("nonexistent.key", "default") == "default"
        assert config.get("mt5.nonexistent", 100) == 100

    def test_get_section(self) -> None:
        """Test getting configuration section."""
        config = Config()

        mt5_section = config.get_section("mt5")
        assert "server" in mt5_section
        assert "timeout" in mt5_section

    def test_property_accessors(self) -> None:
        """Test property accessors."""
        config = Config()

        assert config.mt5 is not None
        assert config.trading is not None
        assert config.slippage is not None
        assert config.voice is not None
        assert config.risk is not None


class TestGlobalConfig:
    """Test cases for global configuration functions."""

    def test_get_config_singleton(self) -> None:
        """Test get_config returns singleton."""
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_reload_config(self) -> None:
        """Test reload_config creates new instance."""
        config1 = get_config()
        config2 = reload_config()

        assert config1 is not config2
