"""Configuration loader for CFD Smart Entry System.

This module handles loading and validation of configuration settings
from YAML files and environment variables.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from loguru import logger


class Config:
    """Configuration manager for CFD Smart Entry System."""

    def __init__(self, config_path: str | None = None) -> None:
        """Initialize configuration manager.

        Args:
            config_path: Path to the YAML configuration file.
                        If None, uses default path.
        """
        load_dotenv()

        if config_path is None:
            config_path = os.getenv(
                "CFD_CONFIG_PATH",
                str(Path(__file__).parent.parent / "config" / "settings.yaml")
            )

        self._config: dict[str, Any] = {}
        self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        """Load configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file.
        """
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            self._config = self._get_defaults()
            return

        with open(path, encoding="utf-8") as f:
            self._config = yaml.safe_load(f) or {}

        logger.info(f"Configuration loaded from {config_path}")

    def _get_defaults(self) -> dict[str, Any]:
        """Get default configuration values.

        Returns:
            Dictionary containing default configuration.
        """
        return {
            "mt5": {
                "server": "Demo-Server",
                "timeout": 60000,
                "retry_attempts": 3,
                "retry_delay": 1.0,
            },
            "trading": {
                "symbols": ["USDJPY", "EURUSD"],
                "default_lot_size": 0.01,
                "max_lot_size": 1.0,
                "max_positions": 5,
                "magic_number": 123456,
            },
            "slippage": {
                "max_slippage_pips": 3.0,
                "execution_timeout_ms": 500,
                "retry_on_requote": True,
                "price_tolerance_pips": 1.0,
            },
            "ai_signal": {
                "model_type": "ensemble",
                "confidence_threshold": 0.65,
            },
            "voice": {
                "enabled": True,
                "language": "ja-JP",
            },
            "risk": {
                "max_daily_loss_percent": 5.0,
                "max_position_risk_percent": 2.0,
                "stop_loss_pips": 50,
                "take_profit_pips": 100,
            },
            "logging": {
                "level": "INFO",
            },
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Supports nested keys with dot notation (e.g., 'mt5.server').

        Args:
            key: Configuration key, supports dot notation for nested values.
            default: Default value if key not found.

        Returns:
            Configuration value or default.
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_section(self, section: str) -> dict[str, Any]:
        """Get entire configuration section.

        Args:
            section: Section name (e.g., 'mt5', 'trading').

        Returns:
            Dictionary containing section configuration.
        """
        return self._config.get(section, {})

    @property
    def mt5(self) -> dict[str, Any]:
        """Get MT5 configuration section."""
        return self.get_section("mt5")

    @property
    def trading(self) -> dict[str, Any]:
        """Get trading configuration section."""
        return self.get_section("trading")

    @property
    def slippage(self) -> dict[str, Any]:
        """Get slippage configuration section."""
        return self.get_section("slippage")

    @property
    def ai_signal(self) -> dict[str, Any]:
        """Get AI signal configuration section."""
        return self.get_section("ai_signal")

    @property
    def voice(self) -> dict[str, Any]:
        """Get voice configuration section."""
        return self.get_section("voice")

    @property
    def risk(self) -> dict[str, Any]:
        """Get risk management configuration section."""
        return self.get_section("risk")

    @property
    def logging_config(self) -> dict[str, Any]:
        """Get logging configuration section."""
        return self.get_section("logging")


# Global configuration instance
_config: Config | None = None


def get_config(config_path: str | None = None) -> Config:
    """Get global configuration instance.

    Args:
        config_path: Path to configuration file (only used on first call).

    Returns:
        Config instance.
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config


def reload_config(config_path: str | None = None) -> Config:
    """Reload configuration from file.

    Args:
        config_path: Path to configuration file.

    Returns:
        New Config instance.
    """
    global _config
    _config = Config(config_path)
    return _config
