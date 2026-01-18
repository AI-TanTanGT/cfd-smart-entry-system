#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtester.py
バックテスト連携モジュール - 過去ログからの学習
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
from pathlib import Path


@dataclass
class BacktestResult:
      """バックテスト結果"""
      total_trades: int
      win_rate: float
      avg_pnl_pips: float
      total_pnl_pips: float
      avg_slippage: float
      max_drawdown_pips: float
      sharpe_ratio: float
      avg_score: float
      best_params: Dict
      recommendations: List[str]


class TradeLogAnalyzer:
      """トレードログ分析"""

    def __init__(self, log_dir: str = "trade_logs"):
              self.log_dir = Path(log_dir)
              self.logs: List[Dict] = []

    def load_logs(self, days: int = 30) -> int:
              """ログファイル読み込み"""
              self.logs = []
              cutoff = datetime.now() - timedelta(days=days)

        if not self.log_dir.exists():
                      return 0

        for log_file in self.log_dir.glob("*.json"):
                      try:
                                        with open(log_file, "r", encoding="utf-8") as f:
                                                              log = json.load(f)
                                                              timestamp = datetime.fromisoformat(log["timestamp_start"])
                                                              if timestamp >= cutoff:
                                                                                        self.logs.append(log)
                      except (json.JSONDecodeError, KeyError):
                                        continue

                  return len(self.logs)

    def calculate_metrics(self) -> Dict:
              """メトリクス計算"""
              if not self.logs:
                            return {}

              pnls = [log.get("pnl_pips", 0) for log in self.logs if log.get("pnl_pips") is not None]
              slippages = [log.get("slippage_pips", 0) for log in self.logs]
              scores = [log.get("score", {}).get("total", 0) for log in self.logs if log.get("score")]

        wins = sum(1 for p in pnls if p > 0)

        return {
                      "total_trades": len(self.logs),
                      "win_rate": wins / len(pnls) if pnls else 0,
                      "avg_pnl": np.mean(pnls) if pnls else 0,
                      "total_pnl": sum(pnls),
                      "avg_slippage": np.mean(slippages) if slippages else 0,
                      "max_slippage": max(slippages) if slippages else 0,
                      "avg_score": np.mean(scores) if scores else 0,
                      "std_pnl": np.std(pnls) if pnls else 0
        }

    def analyze_by_symbol(self) -> Dict[str, Dict]:
              """銘柄別分析"""
              by_symbol = {}
              for log in self.logs:
                            symbol = log.get("symbol", "UNKNOWN")
                            if symbol not in by_symbol:
                                              by_symbol[symbol] = []
                                          by_symbol[symbol].append(log)

              results = {}
              for symbol, logs in by_symbol.items():
                            pnls = [log.get("pnl_pips", 0) for log in logs if log.get("pnl_pips")]
                            results[symbol] = {
                                "trades": len(logs),
                                "win_rate": sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0,
                                "avg_pnl": np.mean(pnls) if pnls else 0
                            }
                        return results

    def analyze_by_urgency(self) -> Dict[str, Dict]:
              """緊急度別分析"""
        by_urgency = {"NORMAL": [], "URGENT": [], "EMERGENCY": []}

        for log in self.logs:
                      for step in log.get("process_steps", []):
                                        if step.get("phase") == "voice_input":
                                                              urgency = step.get("details", {}).get("urgency", "NORMAL")
                                                              by_urgency.get(urgency, by_urgency["NORMAL"]).append(log)
                                                              break

                                results = {}
        for urgency, logs in by_urgency.items():
                      if not logs:
                                        continue
                                    slippages = [log.get("slippage_pips", 0) for log in logs]
            results[urgency] = {
                              "trades": len(logs),
                              "avg_slippage": np.mean(slippages) if slippages else 0
            }
        return results


class ParameterOptimizer:
      """パラメータ最適化"""

    def __init__(self, analyzer: TradeLogAnalyzer):
              self.analyzer = analyzer

    def optimize_slip_tolerance(self) -> Dict:
              """許容スリップ最適化"""
        slippages = [log.get("slippage_pips", 0) for log in self.analyzer.logs]

        if not slippages:
                      return {"recommended": 0.5, "reason": "データ不足"}

        percentile_90 = np.percentile(slippages, 90)
        percentile_95 = np.percentile(slippages, 95)

        if percentile_90 < 0.3:
                      recommended = 0.3
            reason = "スリップ小: 許容値を小さく設定可能"
elif percentile_90 < 0.5:
            recommended = 0.5
            reason = "スリップ中: 標準設定を推奨"
else:
            recommended = min(percentile_95, 1.0)
            reason = "スリップ大: 許容値を拡大推奨"

        return {
                      "recommended": round(recommended, 2),
                      "p90": round(percentile_90, 2),
                      "p95": round(percentile_95, 2),
                      "reason": reason
        }

    def optimize_chase_attempts(self) -> Dict:
              """価格追従回数最適化"""
        attempts = []
        for log in self.analyzer.logs:
                      if log.get("total_attempts"):
                                        attempts.append(log["total_attempts"])

        if not attempts:
                      return {"recommended": 3, "reason": "データ不足"}

        avg_attempts = np.mean(attempts)
        success_rate_by_attempt = {}

        for log in self.analyzer.logs:
                      att = log.get("total_attempts", 0)
            pnl = log.get("pnl_pips", 0)
            if att not in success_rate_by_attempt:
                              success_rate_by_attempt[att] = {"wins": 0, "total": 0}
                          success_rate_by_attempt[att]["total"] += 1
            if pnl and pnl > 0:
                              success_rate_by_attempt[att]["wins"] += 1

        if avg_attempts <= 1.5:
                      recommended = 2
            reason = "一発約定率高: 追従回数を減らしてレイテンシ削減"
elif avg_attempts <= 2.5:
            recommended = 3
            reason = "標準的な約定率: 現状維持"
else:
            recommended = 4
            reason = "約定に時間がかかる: 追従回数を増やす"

        return {
                      "recommended": recommended,
                      "avg_attempts": round(avg_attempts, 2),
                      "reason": reason
        }


class Backtester:
      """バックテストエンジン"""

    def __init__(self, log_dir: str = "trade_logs"):
              self.analyzer = TradeLogAnalyzer(log_dir)
        self.optimizer = ParameterOptimizer(self.analyzer)

    def run(self, days: int = 30) -> BacktestResult:
              """バックテスト実行"""
        loaded = self.analyzer.load_logs(days)

        if loaded == 0:
                      return BacktestResult(
                          total_trades=0,
                          win_rate=0,
                          avg_pnl_pips=0,
                          total_pnl_pips=0,
                          avg_slippage=0,
                          max_drawdown_pips=0,
                          sharpe_ratio=0,
                          avg_score=0,
                          best_params={},
                          recommendations=["トレードログがありません"]
        )

        metrics = self.analyzer.calculate_metrics()
        slip_opt = self.optimizer.optimize_slip_tolerance()
        chase_opt = self.optimizer.optimize_chase_attempts()

        # シャープレシオ計算
        if metrics["std_pnl"] > 0:
                      sharpe = metrics["avg_pnl"] / metrics["std_pnl"] * np.sqrt(252)
else:
            sharpe = 0

        # 最大ドローダウン計算
        pnls = [log.get("pnl_pips", 0) for log in self.analyzer.logs if log.get("pnl_pips")]
        cumsum = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumsum)
        drawdowns = running_max - cumsum
        max_dd = max(drawdowns) if len(drawdowns) > 0 else 0

        # 推奨事項生成
        recommendations = self._generate_recommendations(metrics, slip_opt, chase_opt)

        return BacktestResult(
                      total_trades=metrics["total_trades"],
                      win_rate=metrics["win_rate"],
                      avg_pnl_pips=metrics["avg_pnl"],
                      total_pnl_pips=metrics["total_pnl"],
                      avg_slippage=metrics["avg_slippage"],
                      max_drawdown_pips=max_dd,
                      sharpe_ratio=sharpe,
                      avg_score=metrics["avg_score"],
                      best_params={
                                        "max_slip_pips": slip_opt["recommended"],
                                        "chase_attempts": chase_opt["recommended"]
                      },
                      recommendations=recommendations
        )

    def _generate_recommendations(self, metrics: Dict, slip_opt: Dict, chase_opt: Dict) -> List[str]:
              """推奨事項生成"""
        recs = []

        if metrics["win_rate"] < 0.5:
                      recs.append(f"勝率{metrics['win_rate']*100:.1f}%: エントリー条件の見直しを推奨")

        if metrics["avg_slippage"] > 0.5:
                      recs.append(f"平均スリップ{metrics['avg_slippage']:.2f}pips: 指値変換パラメータの調整を推奨")

        if slip_opt["recommended"] != 0.5:
                      recs.append(f"許容スリップ: {slip_opt['recommended']}pips に変更推奨 ({slip_opt['reason']})")

        if chase_opt["recommended"] != 3:
                      recs.append(f"価格追従回数: {chase_opt['recommended']}回 に変更推奨 ({chase_opt['reason']})")

        if metrics["avg_score"] < 70:
                      recs.append(f"平均スコア{metrics['avg_score']:.1f}点: トレード品質の向上が必要")

        if not recs:
                      recs.append("現在のパラメータは最適です")

        return recs

    def export_report(self, result: BacktestResult, filepath: str = "backtest_report.json"):
              """レポート出力"""
        report = {
                      "generated_at": datetime.now().isoformat(),
                      "summary": {
                                        "total_trades": result.total_trades,
                                        "win_rate": f"{result.win_rate*100:.1f}%",
                                        "avg_pnl": f"{result.avg_pnl_pips:.2f} pips",
                                        "total_pnl": f"{result.total_pnl_pips:.2f} pips",
                                        "sharpe_ratio": f"{result.sharpe_ratio:.2f}",
                                        "max_drawdown": f"{result.max_drawdown_pips:.2f} pips"
                      },
                      "optimized_params": result.best_params,
                      "recommendations": result.recommendations,
                      "symbol_analysis": self.analyzer.analyze_by_symbol(),
                      "urgency_analysis": self.analyzer.analyze_by_urgency()
        }

        with open(filepath, "w", encoding="utf-8") as f:
                      json.dump(report, f, ensure_ascii=False, indent=2)

        return filepath


# 使用例
if __name__ == "__main__":
      backtester = Backtester()
    result = backtester.run(days=30)

    print(f"=== バックテスト結果 ===")
    print(f"総トレード数: {result.total_trades}")
    print(f"勝率: {result.win_rate*100:.1f}%")
    print(f"平均PnL: {result.avg_pnl_pips:.2f} pips")
    print(f"シャープレシオ: {result.sharpe_ratio:.2f}")
    print(f"最適パラメータ: {result.best_params}")
    print(f"推奨事項:")
    for rec in result.recommendations:
              print(f"  - {rec}")

    # レポート出力
    backtester.export_report(result)
