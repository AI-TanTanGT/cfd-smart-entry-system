"""Main application entry point for CFD Smart Entry System.

This module provides the main application class that orchestrates
all components of the smart entry system.
"""

import signal
import sys
import threading
import time
from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger

from .ai_signal_generator import AISignalGenerator, SignalType
from .config import get_config, Config
from .market_analyzer import MarketAnalyzer
from .mt5_connector import MT5Connector
from .order_executor import OrderExecutor
from .slippage_reducer import SlippageConfig
from .voice_input import VoiceInputHandler, VoiceCommand, VoiceCommandResult


class CFDSmartEntrySystem:
    """Main application class for CFD Smart Entry System."""

    def __init__(self, config_path: str | None = None) -> None:
        """Initialize CFD Smart Entry System.

        Args:
            config_path: Path to configuration file.
        """
        self.config = get_config(config_path)
        self._setup_logging()

        # Initialize components
        self.mt5 = MT5Connector(
            server=self.config.get("mt5.server"),
            timeout=self.config.get("mt5.timeout", 60000),
            magic_number=self.config.get("trading.magic_number", 123456)
        )

        self.signal_generator = AISignalGenerator(
            confidence_threshold=self.config.get("ai_signal.confidence_threshold", 0.65),
            model_type=self.config.get("ai_signal.model_type", "ensemble")
        )

        slippage_config = SlippageConfig(
            max_slippage_pips=self.config.get("slippage.max_slippage_pips", 3.0),
            execution_timeout_ms=self.config.get("slippage.execution_timeout_ms", 500),
            retry_on_requote=self.config.get("slippage.retry_on_requote", True),
            price_tolerance_pips=self.config.get("slippage.price_tolerance_pips", 1.0)
        )

        self.order_executor = OrderExecutor(
            mt5_connector=self.mt5,
            signal_generator=self.signal_generator,
            slippage_config=slippage_config
        )

        self.voice_handler = VoiceInputHandler(
            language=self.config.get("voice.language", "ja-JP"),
            enabled=self.config.get("voice.enabled", True)
        )

        # State
        self._running = False
        self._voice_thread: threading.Thread | None = None
        self._monitor_thread: threading.Thread | None = None

        logger.info("CFD Smart Entry System initialized")

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = self.config.get("logging.level", "INFO")
        log_file = self.config.get("logging.file_path", "logs/cfd_smart_entry.log")

        # Remove default handler
        logger.remove()

        # Add console handler
        logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                   "<level>{message}</level>"
        )

        # Add file handler
        try:
            import os
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            logger.add(
                log_file,
                level=log_level,
                rotation=self.config.get("logging.rotation", "1 day"),
                retention=self.config.get("logging.retention", "7 days")
            )
        except Exception as e:
            logger.warning(f"Could not setup file logging: {e}")

    def connect(self, login: int | None = None, password: str | None = None) -> bool:
        """Connect to MT5 platform.

        Args:
            login: MT5 account login.
            password: MT5 account password.

        Returns:
            True if connection successful.
        """
        return self.mt5.connect(login, password)

    def disconnect(self) -> None:
        """Disconnect from MT5 platform."""
        self.mt5.disconnect()

    def start(self) -> None:
        """Start the smart entry system."""
        if self._running:
            logger.warning("System already running")
            return

        self._running = True

        # Setup voice command handlers
        self._setup_voice_handlers()

        # Start voice input thread
        if self.config.get("voice.enabled", True):
            self._voice_thread = threading.Thread(
                target=self._voice_input_loop,
                daemon=True
            )
            self._voice_thread.start()
            logger.info("Voice input thread started")

        # Start market monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._market_monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Market monitoring thread started")

        logger.info("CFD Smart Entry System started")

    def stop(self) -> None:
        """Stop the smart entry system."""
        self._running = False
        logger.info("CFD Smart Entry System stopping...")

        # Wait for threads to finish
        if self._voice_thread and self._voice_thread.is_alive():
            self._voice_thread.join(timeout=2)

        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2)

        logger.info("CFD Smart Entry System stopped")

    def _setup_voice_handlers(self) -> None:
        """Setup voice command callback handlers."""
        self.voice_handler.register_callback(
            VoiceCommand.BUY,
            self._handle_buy_command
        )
        self.voice_handler.register_callback(
            VoiceCommand.SELL,
            self._handle_sell_command
        )
        self.voice_handler.register_callback(
            VoiceCommand.CLOSE,
            self._handle_close_command
        )
        self.voice_handler.register_callback(
            VoiceCommand.CLOSE_ALL,
            self._handle_close_all_command
        )
        self.voice_handler.register_callback(
            VoiceCommand.STATUS,
            self._handle_status_command
        )
        self.voice_handler.register_callback(
            VoiceCommand.STOP,
            self._handle_stop_command
        )

    def _voice_input_loop(self) -> None:
        """Main loop for voice input processing."""
        while self._running:
            try:
                result = self.voice_handler.listen(timeout=3.0)
                if result:
                    self.voice_handler.process_command(result)
            except Exception as e:
                logger.error(f"Voice input error: {e}")

            time.sleep(0.1)

    def _market_monitor_loop(self) -> None:
        """Main loop for market monitoring and signal generation."""
        symbols = self.config.get("trading.symbols", ["USDJPY"])
        check_interval = 60  # seconds

        while self._running:
            try:
                for symbol in symbols:
                    if not self._running:
                        break

                    self._analyze_and_signal(symbol)

                # Wait for next check
                for _ in range(check_interval * 10):
                    if not self._running:
                        break
                    time.sleep(0.1)

            except Exception as e:
                logger.error(f"Market monitor error: {e}")
                time.sleep(5)

    def _analyze_and_signal(self, symbol: str) -> None:
        """Analyze market and generate signal for symbol.

        Args:
            symbol: Trading symbol.
        """
        # Get market data
        ohlcv = self.mt5.get_ohlcv(symbol)
        if ohlcv is None:
            logger.debug(f"No data for {symbol}")
            return

        tick = self.mt5.get_tick(symbol)
        if tick is None:
            return

        current_price = tick.ask

        # Generate signal
        signal = self.signal_generator.generate_signal(
            symbol=symbol,
            ohlcv_data=ohlcv,
            current_price=current_price,
            current_spread=tick.spread
        )

        # Log signal
        logger.debug(
            f"Signal for {symbol}: {signal.signal_type.value} "
            f"(tier={signal.tier.value}, confidence={signal.confidence:.2%})"
        )

        # Auto-execute if high confidence
        if signal.confidence >= 0.8 and signal.signal_type != SignalType.HOLD:
            self._auto_execute_signal(signal)

    def _auto_execute_signal(self, signal: Any) -> None:
        """Auto-execute high confidence signal.

        Args:
            signal: Trading signal.
        """
        default_lot = self.config.get("trading.default_lot_size", 0.01)

        logger.info(
            f"Auto-executing signal: {signal.symbol} {signal.signal_type.value} "
            f"(confidence={signal.confidence:.2%})"
        )

        result = self.order_executor.execute_signal(signal, default_lot)

        if result.status.value == "executed":
            logger.info(f"Auto-trade executed: ticket={result.ticket}")
        else:
            logger.warning(f"Auto-trade failed: {result.message}")

    def _handle_buy_command(self, result: VoiceCommandResult) -> None:
        """Handle buy voice command."""
        symbol = result.symbol or self.config.get("trading.symbols", ["USDJPY"])[0]
        volume = result.quantity or self.config.get("trading.default_lot_size", 0.01)

        logger.info(f"Voice command: BUY {symbol} {volume} lots")

        # Get market data and generate signal
        ohlcv = self.mt5.get_ohlcv(symbol)
        tick = self.mt5.get_tick(symbol)

        if ohlcv is None or tick is None:
            logger.error(f"Cannot get data for {symbol}")
            return

        signal = self.signal_generator.generate_signal(
            symbol=symbol,
            ohlcv_data=ohlcv,
            current_price=tick.ask,
            current_spread=tick.spread
        )

        # Override signal type to BUY
        signal.signal_type = SignalType.BUY

        # Execute
        exec_result = self.order_executor.execute_signal(signal, volume)
        logger.info(f"Buy order result: {exec_result.status.value}")

    def _handle_sell_command(self, result: VoiceCommandResult) -> None:
        """Handle sell voice command."""
        symbol = result.symbol or self.config.get("trading.symbols", ["USDJPY"])[0]
        volume = result.quantity or self.config.get("trading.default_lot_size", 0.01)

        logger.info(f"Voice command: SELL {symbol} {volume} lots")

        # Get market data and generate signal
        ohlcv = self.mt5.get_ohlcv(symbol)
        tick = self.mt5.get_tick(symbol)

        if ohlcv is None or tick is None:
            logger.error(f"Cannot get data for {symbol}")
            return

        signal = self.signal_generator.generate_signal(
            symbol=symbol,
            ohlcv_data=ohlcv,
            current_price=tick.bid,
            current_spread=tick.spread
        )

        # Override signal type to SELL
        signal.signal_type = SignalType.SELL

        # Execute
        exec_result = self.order_executor.execute_signal(signal, volume)
        logger.info(f"Sell order result: {exec_result.status.value}")

    def _handle_close_command(self, result: VoiceCommandResult) -> None:
        """Handle close voice command."""
        symbol = result.symbol

        if symbol:
            logger.info(f"Voice command: CLOSE positions for {symbol}")
            results = self.order_executor.close_all_positions(symbol)
        else:
            logger.info("Voice command: CLOSE - need to specify symbol or position")

    def _handle_close_all_command(self, result: VoiceCommandResult) -> None:
        """Handle close all voice command."""
        logger.info("Voice command: CLOSE ALL positions")
        results = self.order_executor.close_all_positions()
        logger.info(f"Closed {len(results)} positions")

    def _handle_status_command(self, result: VoiceCommandResult) -> None:
        """Handle status voice command."""
        logger.info("Voice command: STATUS")

        # Get account info
        account = self.mt5.get_account_info()
        if account:
            logger.info(
                f"Account: Balance={account['balance']}, "
                f"Equity={account['equity']}, "
                f"Profit={account['profit']}"
            )

        # Get positions
        positions = self.mt5.get_positions()
        logger.info(f"Open positions: {len(positions)}")

        for pos in positions:
            logger.info(
                f"  {pos.symbol} {pos.type.name} {pos.volume} lots: "
                f"P/L={pos.profit}"
            )

        # Get execution stats
        stats = self.order_executor.get_execution_statistics()
        logger.info(
            f"Execution stats: {stats['successful_orders']}/{stats['total_orders']} orders, "
            f"avg slippage={stats['avg_slippage_pips']:.2f} pips"
        )

    def _handle_stop_command(self, result: VoiceCommandResult) -> None:
        """Handle stop voice command."""
        logger.info("Voice command: STOP system")
        self.stop()

    def execute_text_command(self, text: str) -> dict[str, Any]:
        """Execute command from text input.

        Args:
            text: Command text.

        Returns:
            Dictionary with result.
        """
        result = self.voice_handler.parse_text_command(text)

        if result.command == VoiceCommand.UNKNOWN:
            return {"success": False, "message": "Unknown command"}

        self.voice_handler.process_command(result)

        return {
            "success": True,
            "command": result.command.value,
            "symbol": result.symbol,
            "quantity": result.quantity
        }

    def get_status(self) -> dict[str, Any]:
        """Get system status.

        Returns:
            Dictionary with system status.
        """
        account = self.mt5.get_account_info() or {}
        positions = self.mt5.get_positions()
        exec_stats = self.order_executor.get_execution_statistics()
        slippage_stats = self.order_executor.slippage_reducer.get_execution_statistics()

        return {
            "running": self._running,
            "connected": self.mt5.is_connected(),
            "account": account,
            "positions_count": len(positions),
            "total_profit": self.mt5.get_total_profit(),
            "execution_statistics": exec_stats,
            "slippage_statistics": slippage_stats,
            "timestamp": datetime.now().isoformat()
        }


def main() -> None:
    """Main entry point."""
    system = CFDSmartEntrySystem()

    # Setup signal handlers
    def signal_handler(sig: int, frame: Any) -> None:
        logger.info("Shutdown signal received")
        system.stop()
        system.disconnect()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Connect and start
    if system.connect():
        system.start()

        # Keep running
        try:
            while system._running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            system.stop()
            system.disconnect()
    else:
        logger.error("Failed to connect to MT5")
        sys.exit(1)


if __name__ == "__main__":
    main()
