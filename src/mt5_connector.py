"""MT5 connector for CFD Smart Entry System.

This module provides integration with MetaTrader 5 platform
for market data retrieval and order execution.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


class MT5OrderType(Enum):
    """Enumeration for MT5 order types."""

    BUY = 0
    SELL = 1
    BUY_LIMIT = 2
    SELL_LIMIT = 3
    BUY_STOP = 4
    SELL_STOP = 5


class MT5TradeAction(Enum):
    """Enumeration for MT5 trade actions."""

    DEAL = 1
    PENDING = 5
    MODIFY = 6
    REMOVE = 8
    CLOSE_BY = 10


@dataclass
class MT5Position:
    """Data class representing an MT5 position."""

    ticket: int
    symbol: str
    type: MT5OrderType
    volume: float
    price_open: float
    price_current: float
    profit: float
    swap: float
    time: datetime
    magic: int
    comment: str


@dataclass
class MT5Order:
    """Data class representing an MT5 order."""

    ticket: int
    symbol: str
    type: MT5OrderType
    volume: float
    price_open: float
    sl: float
    tp: float
    magic: int
    comment: str


@dataclass
class MT5TickData:
    """Data class for MT5 tick data."""

    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    time: datetime
    spread: float


class MT5Connector:
    """Connector for MetaTrader 5 platform."""

    def __init__(
        self,
        server: str | None = None,
        timeout: int = 60000,
        magic_number: int = 123456
    ) -> None:
        """Initialize MT5 connector.

        Args:
            server: MT5 server name.
            timeout: Connection timeout in milliseconds.
            magic_number: Magic number for orders.
        """
        self.server = server
        self.timeout = timeout
        self.magic_number = magic_number
        self._connected = False
        self._mt5: Any = None

        logger.info(f"MT5Connector initialized (server={server})")

    def connect(self, login: int | None = None, password: str | None = None) -> bool:
        """Connect to MT5 platform.

        Args:
            login: MT5 account login.
            password: MT5 account password.

        Returns:
            True if connection successful.
        """
        try:
            import MetaTrader5 as mt5

            self._mt5 = mt5

            # Initialize MT5
            if not mt5.initialize():
                error = mt5.last_error()
                logger.error(f"MT5 initialization failed: {error}")
                return False

            # Login if credentials provided
            if login and password:
                authorized = mt5.login(
                    login=login,
                    password=password,
                    server=self.server
                )
                if not authorized:
                    error = mt5.last_error()
                    logger.error(f"MT5 login failed: {error}")
                    return False

            self._connected = True
            account_info = mt5.account_info()
            logger.info(
                f"MT5 connected: {account_info.name if account_info else 'Unknown'} "
                f"(Balance: {account_info.balance if account_info else 0})"
            )
            return True

        except ImportError:
            logger.error("MetaTrader5 package not installed")
            return False
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from MT5 platform."""
        if self._connected and self._mt5:
            self._mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected")

    def is_connected(self) -> bool:
        """Check if connected to MT5.

        Returns:
            True if connected.
        """
        return self._connected

    def get_account_info(self) -> dict[str, Any] | None:
        """Get account information.

        Returns:
            Dictionary with account info or None.
        """
        if not self._connected:
            return None

        info = self._mt5.account_info()
        if info is None:
            return None

        return {
            "login": info.login,
            "name": info.name,
            "server": info.server,
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "margin_free": info.margin_free,
            "margin_level": info.margin_level,
            "profit": info.profit,
            "currency": info.currency,
            "leverage": info.leverage,
        }

    def get_symbol_info(self, symbol: str) -> dict[str, Any] | None:
        """Get symbol information.

        Args:
            symbol: Trading symbol.

        Returns:
            Dictionary with symbol info or None.
        """
        if not self._connected:
            return None

        info = self._mt5.symbol_info(symbol)
        if info is None:
            logger.warning(f"Symbol {symbol} not found")
            return None

        return {
            "name": info.name,
            "bid": info.bid,
            "ask": info.ask,
            "spread": info.spread,
            "digits": info.digits,
            "point": info.point,
            "trade_contract_size": info.trade_contract_size,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
        }

    def get_tick(self, symbol: str) -> MT5TickData | None:
        """Get current tick data for symbol.

        Args:
            symbol: Trading symbol.

        Returns:
            MT5TickData or None.
        """
        if not self._connected:
            return None

        tick = self._mt5.symbol_info_tick(symbol)
        if tick is None:
            return None

        return MT5TickData(
            symbol=symbol,
            bid=tick.bid,
            ask=tick.ask,
            last=tick.last,
            volume=tick.volume,
            time=datetime.fromtimestamp(tick.time),
            spread=tick.ask - tick.bid
        )

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "H1",
        count: int = 200
    ) -> pd.DataFrame | None:
        """Get OHLCV data for symbol.

        Args:
            symbol: Trading symbol.
            timeframe: Timeframe string (M1, M5, M15, H1, H4, D1).
            count: Number of bars to retrieve.

        Returns:
            DataFrame with OHLCV data or None.
        """
        if not self._connected:
            return None

        # Map timeframe string to MT5 constant
        timeframe_map = {
            "M1": self._mt5.TIMEFRAME_M1,
            "M5": self._mt5.TIMEFRAME_M5,
            "M15": self._mt5.TIMEFRAME_M15,
            "M30": self._mt5.TIMEFRAME_M30,
            "H1": self._mt5.TIMEFRAME_H1,
            "H4": self._mt5.TIMEFRAME_H4,
            "D1": self._mt5.TIMEFRAME_D1,
            "W1": self._mt5.TIMEFRAME_W1,
        }

        tf = timeframe_map.get(timeframe.upper(), self._mt5.TIMEFRAME_H1)

        rates = self._mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            logger.warning(f"No data retrieved for {symbol}")
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={
            "time": "timestamp",
            "tick_volume": "volume"
        })

        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def get_positions(self, symbol: str | None = None) -> list[MT5Position]:
        """Get open positions.

        Args:
            symbol: Filter by symbol (optional).

        Returns:
            List of MT5Position objects.
        """
        if not self._connected:
            return []

        if symbol:
            positions = self._mt5.positions_get(symbol=symbol)
        else:
            positions = self._mt5.positions_get()

        if positions is None:
            return []

        result = []
        for pos in positions:
            result.append(MT5Position(
                ticket=pos.ticket,
                symbol=pos.symbol,
                type=MT5OrderType(pos.type),
                volume=pos.volume,
                price_open=pos.price_open,
                price_current=pos.price_current,
                profit=pos.profit,
                swap=pos.swap,
                time=datetime.fromtimestamp(pos.time),
                magic=pos.magic,
                comment=pos.comment
            ))

        return result

    def send_order(
        self,
        symbol: str,
        order_type: MT5OrderType,
        volume: float,
        price: float | None = None,
        sl: float | None = None,
        tp: float | None = None,
        comment: str = ""
    ) -> dict[str, Any]:
        """Send trading order.

        Args:
            symbol: Trading symbol.
            order_type: Type of order.
            volume: Order volume in lots.
            price: Order price (for pending orders).
            sl: Stop loss price.
            tp: Take profit price.
            comment: Order comment.

        Returns:
            Dictionary with order result.
        """
        if not self._connected:
            return {"success": False, "error": "Not connected"}

        # Get symbol info
        symbol_info = self._mt5.symbol_info(symbol)
        if symbol_info is None:
            return {"success": False, "error": f"Symbol {symbol} not found"}

        if not symbol_info.visible:
            if not self._mt5.symbol_select(symbol, True):
                return {"success": False, "error": f"Failed to select {symbol}"}

        # Determine price
        if price is None:
            if order_type == MT5OrderType.BUY:
                price = symbol_info.ask
            else:
                price = symbol_info.bid

        # Prepare request
        request = {
            "action": self._mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type.value,
            "price": price,
            "magic": self.magic_number,
            "comment": comment,
            "type_time": self._mt5.ORDER_TIME_GTC,
            "type_filling": self._mt5.ORDER_FILLING_IOC,
        }

        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp

        # Send order
        result = self._mt5.order_send(request)

        if result.retcode != self._mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.comment}")
            return {
                "success": False,
                "error": result.comment,
                "retcode": result.retcode
            }

        logger.info(f"Order executed: {symbol} {order_type.value} {volume} lots at {result.price}")

        return {
            "success": True,
            "ticket": result.order,
            "price": result.price,
            "volume": result.volume,
        }

    def close_position(self, ticket: int) -> dict[str, Any]:
        """Close position by ticket.

        Args:
            ticket: Position ticket number.

        Returns:
            Dictionary with close result.
        """
        if not self._connected:
            return {"success": False, "error": "Not connected"}

        # Get position
        position = self._mt5.positions_get(ticket=ticket)
        if not position:
            return {"success": False, "error": f"Position {ticket} not found"}

        pos = position[0]

        # Determine close order type
        if pos.type == 0:  # Buy position
            close_type = MT5OrderType.SELL
            price = self._mt5.symbol_info_tick(pos.symbol).bid
        else:  # Sell position
            close_type = MT5OrderType.BUY
            price = self._mt5.symbol_info_tick(pos.symbol).ask

        # Prepare close request
        request = {
            "action": self._mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type.value,
            "position": ticket,
            "price": price,
            "magic": self.magic_number,
            "comment": "Close by CFD Smart Entry",
            "type_time": self._mt5.ORDER_TIME_GTC,
            "type_filling": self._mt5.ORDER_FILLING_IOC,
        }

        result = self._mt5.order_send(request)

        if result.retcode != self._mt5.TRADE_RETCODE_DONE:
            logger.error(f"Close failed: {result.comment}")
            return {
                "success": False,
                "error": result.comment,
                "retcode": result.retcode
            }

        logger.info(f"Position {ticket} closed at {result.price}")

        return {
            "success": True,
            "ticket": ticket,
            "close_price": result.price,
            "profit": pos.profit
        }

    def modify_position(
        self,
        ticket: int,
        sl: float | None = None,
        tp: float | None = None
    ) -> dict[str, Any]:
        """Modify position stop loss and take profit.

        Args:
            ticket: Position ticket.
            sl: New stop loss price.
            tp: New take profit price.

        Returns:
            Dictionary with modification result.
        """
        if not self._connected:
            return {"success": False, "error": "Not connected"}

        position = self._mt5.positions_get(ticket=ticket)
        if not position:
            return {"success": False, "error": f"Position {ticket} not found"}

        pos = position[0]

        request = {
            "action": self._mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": ticket,
        }

        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp

        result = self._mt5.order_send(request)

        if result.retcode != self._mt5.TRADE_RETCODE_DONE:
            return {
                "success": False,
                "error": result.comment,
                "retcode": result.retcode
            }

        logger.info(f"Position {ticket} modified: SL={sl}, TP={tp}")

        return {
            "success": True,
            "ticket": ticket,
            "sl": sl,
            "tp": tp
        }

    def get_total_profit(self) -> float:
        """Get total profit from all open positions.

        Returns:
            Total profit.
        """
        positions = self.get_positions()
        return sum(pos.profit for pos in positions)

    def get_symbol_positions_count(self, symbol: str) -> int:
        """Get count of positions for a symbol.

        Args:
            symbol: Trading symbol.

        Returns:
            Number of positions.
        """
        positions = self.get_positions(symbol)
        return len(positions)
