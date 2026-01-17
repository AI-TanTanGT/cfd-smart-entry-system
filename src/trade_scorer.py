#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trade_scorer.py
トレード採点エンジン
"""

from typing import List, Tuple


class TradeScorer:
      """トレード採点エンジン"""

    # 採点基準（pips）
      EXCELLENT_SLIP = 0.2
      GOOD_SLIP = 0.5
      ACCEPTABLE_SLIP = 1.0

    # グレード閾値
      GRADE_THRESHOLDS = [
          (90, "S"),
          (80, "A"),
          (70, "B"),
          (60, "C"),
          (50, "D"),
          (0, "F")
      ]

    def score_timing(self, trade_log) -> Tuple[int, List[str], List[str]]:
              """タイミング採点"""
              strengths = []
              improvements = []
              score = 100

        # スプレッド状況
              if trade_log.spread_at_entry < 0.3:
                            strengths.append("低スプレッド時にエントリー")
elif trade_log.spread_at_entry > 1.0:
            score -= 20
            improvements.append(f"スプレッド{trade_log.spread_at_entry:.1f}pips - 広がり時のエントリー回避推奨")

        # VPIN状況
        if trade_log.vpin_at_entry > 0.7:
                      score -= 25
                      improvements.append(f"VPIN={trade_log.vpin_at_entry:.2f} - 情報非対称性高、流動性低下リスク")
elif trade_log.vpin_at_entry < 0.3:
              strengths.append("VPIN低水準 - 流動性良好")

        # 約定までの試行回数
          if trade_log.total_attempts == 1:
                        strengths.append("一発約定")
                        score += 5
elif trade_log.total_attempts > 3:
              score -= 10
              improvements.append(f"約定に{trade_log.total_attempts}回試行 - エントリー価格の見直し推奨")

        return max(0, min(100, score)), strengths, improvements

    def score_price(self, trade_log) -> Tuple[int, List[str], List[str]]:
              """価格精度採点"""
              strengths = []
              improvements = []

        slip = abs(trade_log.slippage_pips)

        if slip <= self.EXCELLENT_SLIP:
                      score = 100
                      strengths.append(f"スリップ{slip:.2f}pips - 優秀")
elif slip <= self.GOOD_SLIP:
              score = 85
              strengths.append(f"スリップ{slip:.2f}pips - 良好")
elif slip <= self.ACCEPTABLE_SLIP:
              score = 70
              improvements.append(f"スリップ{slip:.2f}pips - 許容範囲内だが改善余地あり")
else:
              score = max(0, 70 - (slip - self.ACCEPTABLE_SLIP) * 20)
              improvements.append(f"スリップ{slip:.2f}pips - 指値変換の閾値調整推奨")

        # 約定方法
          if trade_log.final_fill_method == "limit":
                        strengths.append("指値約定成功")
                        score += 5
elif trade_log.final_fill_method == "market":
              score -= 10
              improvements.append("成行約定 - 指値変換パラメータの見直し推奨")

        return max(0, min(100, score)), strengths, improvements

    def score_risk(self, trade_log) -> Tuple[int, List[str], List[str]]:
              """リスク管理採点"""
              strengths = []
              improvements = []
              score = 100

        # SL設定確認
              if trade_log.sl == 0:
                            score -= 50
                            improvements.append("SL未設定 - 必ずSLを設定すること")

              # TP設定確認
              if trade_log.tp == 0:
                            score -= 20
                            improvements.append("TP未設定 - 利確目標の設定推奨")

              # リスクリワード比
              if trade_log.sl != 0 and trade_log.tp != 0:
                            entry = trade_log.actual_entry_price
                            if trade_log.direction == "buy":
                                              risk = entry - trade_log.sl
                                              reward = trade_log.tp - entry
              else:
                                risk = trade_log.sl - entry
                                reward = entry - trade_log.tp

                  if risk > 0:
                                    rr_ratio = reward / risk
                                    if rr_ratio >= 2.0:
                                                          strengths.append(f"RR比={rr_ratio:.2f} - 優秀")
                                                          score += 10
                  elif rr_ratio >= 1.5:
                                        strengths.append(f"RR比={rr_ratio:.2f} - 良好")
elif rr_ratio < 1.0:
                    score -= 20
                    improvements.append(f"RR比={rr_ratio:.2f} - 1.0未満は非推奨")

        return max(0, min(100, score)), strengths, improvements

    def score_result(self, trade_log) -> Tuple[int, List[str], List[str]]:
              """結果採点"""
        strengths = []
        improvements = []

        if trade_log.pnl_pips is None:
                      return 50, [], ["トレード未決済"]

        pnl = trade_log.pnl_pips

        if pnl > 0:
                      # 勝ちトレード
                      if trade_log.exit_reason == "tp":
                                        score = 100
                                        strengths.append(f"TP到達 +{pnl:.1f}pips")
        else:
                score = 80
                          strengths.append(f"利益確定 +{pnl:.1f}pips")
elif pnl == 0:
            score = 60
            improvements.append("収支ゼロ - エントリー精度の向上で改善可能")
else:
            # 負けトレード
              if trade_log.exit_reason == "sl":
                                score = 40
                                improvements.append(f"SL到達 {pnl:.1f}pips - リスク管理は機能")
else:
                score = 30
                improvements.append(f"損失 {pnl:.1f}pips - 損切りルールの見直し推奨")

        return max(0, min(100, score)), strengths, improvements

    def calculate_score(self, trade_log) -> 'TradeScore':
              """総合採点"""
        from trade_log_schema import TradeScore

        timing_score, t_str, t_imp = self.score_timing(trade_log)
        price_score, p_str, p_imp = self.score_price(trade_log)
        risk_score, r_str, r_imp = self.score_risk(trade_log)
        result_score, res_str, res_imp = self.score_result(trade_log)

        # 重み付け平均
        weights = {
                      "timing": 0.20,
                      "price": 0.25,
                      "risk": 0.30,
                      "result": 0.25
        }

        total = (
                      timing_score * weights["timing"] +
                      price_score * weights["price"] +
                      risk_score * weights["risk"] +
                      result_score * weights["result"]
        )
        total = int(round(total))

        # グレード決定
        grade = "F"
        for threshold, g in self.GRADE_THRESHOLDS:
                      if total >= threshold:
                                        grade = g
                                        break

                  # 強み・改善点を統合
                  all_strengths = t_str + p_str + r_str + res_str
        all_improvements = t_imp + p_imp + r_imp + res_imp

        return TradeScore(
                      timing_score=timing_score,
                      price_score=price_score,
                      risk_score=risk_score,
                      result_score=result_score,
                      total_score=total,
                      grade=grade,
                      strengths=all_strengths[:5],  # 上位5つ
                      improvements=all_improvements[:5]  # 上位5つ
        )
