#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smart_entry_controller.py
スマート指値変換システム - メインコントローラー
CFDトレードにおける成行注文スリッページを軽減
"""

import MetaTrader5 as mt5
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import time
import logging

# ロギング設定
logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class OrderUrgency(Enum):
      """注文緊急度"""
      NORMAL = 1      # 通常: 指値待機OK
    URGENT = 2      # 急ぎ: IOC優先
    EMERGENCY = 3   # 緊急: 即時約定必須


@dataclass
class MarketCondition:
      """市況データ"""
      symbol: str
      bid: float
      ask: float
      spread_pips: float
      vpin: float              # 0-1, 高いほど情報非対称
    weekly_pivot: float
    daily_200sma: float
    distance_to_pivot_pips: float
    distance_to_sma_pips: float


@dataclass
class SmartOrder:
      """スマート注文パラメータ"""
      symbol: str
      direction: int           # 1=BUY, -1=SELL
    lot: float
    entry_price: float
    sl: float
    tp: float
    max_slip_pips: float
    order_type: int          # mt5.ORDER_TYPE_*
    expiration_sec: int


class SmartEntryController:
      """スマートエントリーコントローラー"""

    # Exness Zero手数料（往復）
      COMMISSION_PIPS = 0.7
      # 隠れスプレッド推定
      HIDDEN_SPREAD_PIPS = 0.2
      # レイテンシコスト
      LATENCY_COST_PIPS = 0.1
      # 価格追従刻み
      CHASE_STEP_PIPS = 0.1
      # 最大追従回数
      MAX_CHASE_ATTEMPTS = 3

    def __init__(self):
              if not mt5.initialize():
            raise RuntimeError(f"MT5初期化失敗: {mt5.last_error()}")
        logger.info("MT5初期化完了")

    def get_market_condition(self, symbol: str) -> MarketCondition:
              """市況データ取得"""
              tick = mt5.symbol_info_tick(symbol)
              info = mt5.symbol_info(symbol)

        if tick is None or info is None:
                      raise ValueError(f"銘柄データ取得失敗: {symbol}")

        point = info.point
        spread_pips = (tick.ask - tick.bid) / point / 10

        # VPIN計算（簡易版: 直近100ティックから推定）
        ticks = mt5.copy_ticks_from(symbol, tick.time - 60, 100, mt5.COPY_TICKS_ALL)
        vpin = self._calculate_vpin(ticks) if ticks is not None else 0.5

        # ピボット・SMA取得（EA側から取得想定、ここではダミー）
        weekly_pivot = tick.bid  # 要実装
        daily_200sma = tick.bid  # 要実装

        return MarketCondition(
                      symbol=symbol,
                      bid=tick.bid,
                      ask=tick.ask,
                      spread_pips=spread_pips,
                      vpin=vpin,
                      weekly_pivot=weekly_pivot,
                      daily_200sma=daily_200sma,
                      distance_to_pivot_pips=0,
                      distance_to_sma_pips=0
        )

    def _calculate_vpin(self, ticks) -> float:
              """VPIN計算（簡易版）"""
              if len(ticks) < 10:
                            return 0.5

              buy_volume = sum(t['volume'] for t in ticks if t['flags'] & mt5.TICK_FLAG_BUY)
              sell_volume = sum(t['volume'] for t in ticks if t['flags'] & mt5.TICK_FLAG_SELL)
              total = buy_volume + sell_volume

        if total == 0:
                      return 0.5

        return abs(buy_volume - sell_volume) / total

    def calculate_be_pips(self, condition: MarketCondition, slip_pips: float) -> float:
              """BE（損益分岐点）計算"""
              return (
                  self.COMMISSION_PIPS +
                  condition.spread_pips +
                  self.HIDDEN_SPREAD_PIPS +
                  slip_pips +
                  self.LATENCY_COST_PIPS
              )

    def create_smart_order(
              self,
              symbol: str,
              direction: int,
              lot: float,
              urgency: OrderUrgency,
              sl_pips: float,
              tp_pips: float
    ) -> SmartOrder:
              """スマート注文生成"""
              condition = self.get_market_condition(symbol)
              info = mt5.symbol_info(symbol)
              point = info.point

        # 緊急度に応じた許容スリップ設定
              slip_map = {
                  OrderUrgency.NORMAL: 0.3,
                  OrderUrgency.URGENT: 0.5,
                  OrderUrgency.EMERGENCY: 1.0
              }
              max_slip = slip_map[urgency]

        # BE計算 & TP検証
              be = self.calculate_be_pips(condition, max_slip)
        if tp_pips <= be:
                      logger.warning(f"⚠️ TP({tp_pips}pips) <= BE({be:.2f}pips) - 収益性なし")

        # エントリー価格決定
        if direction == 1:  # BUY
                      base_price = condition.ask
                      entry_price = base_price + (max_slip * point * 10)
else:  # SELL
              base_price = condition.bid
              entry_price = base_price - (max_slip * point * 10)

        # SL/TP計算
          if direction == 1:
                        sl = base_price - (sl_pips * point * 10)
                        tp = base_price + (tp_pips * point * 10)
else:
              sl = base_price + (sl_pips * point * 10)
              tp = base_price - (tp_pips * point * 10)

        # 注文タイプ決定
          if urgency == OrderUrgency.EMERGENCY:
                        order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
else:
              order_type = mt5.ORDER_TYPE_BUY_LIMIT if direction == 1 else mt5.ORDER_TYPE_SELL_LIMIT

        return SmartOrder(
                      symbol=symbol,
                      direction=direction,
                      lot=lot,
                      entry_price=round(entry_price, info.digits),
                      sl=round(sl, info.digits),
                      tp=round(tp, info.digits),
                      max_slip_pips=max_slip,
                      order_type=order_type,
                      expiration_sec=30 if urgency == OrderUrgency.URGENT else 60
        )

    def execute_smart_order(self, order: SmartOrder) -> dict:
              """スマート注文実行（段階的価格追従）"""
              info = mt5.symbol_info(order.symbol)
              point = info.point

        for attempt in range(self.MAX_CHASE_ATTEMPTS + 1):
                      # 価格追従
                      chase_offset = attempt * self.CHASE_STEP_PIPS * point * 10
                      if order.direction == 1:
                                        current_price = order.entry_price + chase_offset
else:
                  current_price = order.entry_price - chase_offset

            request = {
                              "action": mt5.TRADE_ACTION_DEAL,
                              "symbol": order.symbol,
                              "volume": order.lot,
                              "type": order.order_type,
                              "price": round(current_price, info.digits),
                              "sl": order.sl,
                              "tp": order.tp,
                              "deviation": int(order.max_slip_pips * 10),
                              "magic": 20250115,
                              "comment": f"SmartEntry_v1_attempt{attempt}",
                              "type_time": mt5.ORDER_TIME_GTC,
                              "type_filling": mt5.ORDER_FILLING_IOC,
            }

            logger.info(f"注文試行 {attempt+1}/{self.MAX_CHASE_ATTEMPTS+1}: {current_price}")
            result = mt5.order_send(request)

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                              logger.info(f"✅ 約定成功: {result.price}")
                              return {
                                  "success": True,
                                  "price": result.price,
                                  "order_id": result.order,
                                  "attempts": attempt + 1
                              }

            logger.warning(f"約定失敗 (retcode={result.retcode}): {result.comment}")
            time.sleep(0.05)  # 50ms待機

        # 最終手段: 成行
        logger.warning("⚠️ 最終手段: 成行注文実行")
        request["type_filling"] = mt5.ORDER_FILLING_FOK
        result = mt5.order_send(request)

        return {
                      "success": result.retcode == mt5.TRADE_RETCODE_DONE,
                      "price": result.price if result.retcode == mt5.TRADE_RETCODE_DONE else None,
                      "order_id": result.order if result.retcode == mt5.TRADE_RETCODE_DONE else None,
                      "attempts": self.MAX_CHASE_ATTEMPTS + 2,
                      "final_retcode": result.retcode
        }

    def shutdown(self):
              """終了処理"""
              mt5.shutdown()
              logger.info("MT5シャットダウン完了")


# 使用例
if __name__ == "__main__":
      controller = SmartEntryController()

    try:
              # 急ぎでEURUSD買い
              order = controller.create_smart_order(
                            symbol="EURUSD",
                            direction=1,  # BUY
                            lot=0.01,
                            urgency=OrderUrgency.URGENT,
                            sl_pips=10.0,
                            tp_pips=15.0
              )

        print(f"生成された注文: {order}")

        # 実行（本番では確認後）
        # result = controller.execute_smart_order(order)
        # print(f"実行結果: {result}")

finally:
        controller.shutdown()
