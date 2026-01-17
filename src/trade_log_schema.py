#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trade_log_schema.py
トレードログスキーマ定義
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from enum import Enum
import json


class TradePhase(Enum):
      """トレードフェーズ"""
      VOICE_INPUT = "voice_input"
      MARKET_ANALYSIS = "market_analysis"
      ORDER_CREATION = "order_creation"
      EXECUTION_ATTEMPT = "execution_attempt"
      FILLED = "filled"
      REJECTED = "rejected"
      CLOSED = "closed"


@dataclass
class ProcessStep:
      """プロセスステップ記録"""
      timestamp: datetime
      phase: TradePhase
      details: dict
      duration_ms: float
      ai_tier_used: Optional[str] = None  # "tier1", "tier2", "tier3"


@dataclass
class TradeScore:
      """トレード採点"""
      timing_score: int          # タイミング精度 0-100
    price_score: int           # 価格精度 0-100
    risk_score: int            # リスク管理 0-100
    result_score: int          # 結果 0-100
    total_score: int           # 総合 0-100
    grade: str                 # S/A/B/C/D/F
    strengths: List[str]       # 良かった点
    improvements: List[str]    # 改善点


@dataclass
class TradeLog:
      """完全トレードログ"""
      # 識別情報
      trade_id: str
      session_id: str
      timestamp_start: datetime
      timestamp_end: Optional[datetime] = None

    # 注文情報
      symbol: str = ""
      direction: str = ""  # "buy" or "sell"
    lot: float = 0.0

    # 価格情報
    intended_price: float = 0.0
    actual_entry_price: float = 0.0
    slippage_pips: float = 0.0
    sl: float = 0.0
    tp: float = 0.0

    # 市況情報
    spread_at_entry: float = 0.0
    vpin_at_entry: float = 0.0
    volatility_at_entry: float = 0.0

    # プロセス記録
    process_steps: List[ProcessStep] = field(default_factory=list)
    total_attempts: int = 0
    final_fill_method: str = ""  # "limit", "ioc", "fok", "market"

    # 結果
    exit_price: Optional[float] = None
    pnl_pips: Optional[float] = None
    pnl_usd: Optional[float] = None
    exit_reason: Optional[str] = None  # "tp", "sl", "manual", "timeout"

    # 採点
    score: Optional[TradeScore] = None

    # AI分析
    ai_analysis: Optional[str] = None
    ai_recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
              """辞書変換"""
              return {
                  "trade_id": self.trade_id,
                  "session_id": self.session_id,
                  "timestamp_start": self.timestamp_start.isoformat(),
                  "timestamp_end": self.timestamp_end.isoformat() if self.timestamp_end else None,
                  "symbol": self.symbol,
                  "direction": self.direction,
                  "lot": self.lot,
                  "intended_price": self.intended_price,
                  "actual_entry_price": self.actual_entry_price,
                  "slippage_pips": self.slippage_pips,
                  "spread_at_entry": self.spread_at_entry,
                  "vpin_at_entry": self.vpin_at_entry,
                  "process_steps": [
                      {
                          "timestamp": s.timestamp.isoformat(),
                          "phase": s.phase.value,
                          "details": s.details,
                          "duration_ms": s.duration_ms,
                          "ai_tier_used": s.ai_tier_used
                      }
                      for s in self.process_steps
                  ],
                  "total_attempts": self.total_attempts,
                  "final_fill_method": self.final_fill_method,
                  "pnl_pips": self.pnl_pips,
                  "pnl_usd": self.pnl_usd,
                  "exit_reason": self.exit_reason,
                  "score": {
                      "timing": self.score.timing_score,
                      "price": self.score.price_score,
                      "risk": self.score.risk_score,
                      "result": self.score.result_score,
                      "total": self.score.total_score,
                      "grade": self.score.grade,
                      "strengths": self.score.strengths,
                      "improvements": self.score.improvements
                  } if self.score else None,
                  "ai_analysis": self.ai_analysis,
                  "ai_recommendations": self.ai_recommendations
              }

    def to_json(self) -> str:
              """JSON文字列に変換"""
              return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
