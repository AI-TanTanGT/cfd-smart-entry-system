#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
slack_integration.py
Slack連携モジュール
"""

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dataclasses import dataclass


@dataclass
class SlackConfig:
      """Slack設定"""
      bot_token: str
      channel_alerts: str = "#trade-alerts"
      channel_logs: str = "#trade-logs"
      channel_ai: str = "#ai-analysis"


class SlackIntegration:
      """Slack統合クラス"""

    def __init__(self, config: SlackConfig):
              self.client = WebClient(token=config.bot_token)
              self.config = config

    def send_trade_alert(self, message: str, urgency: str = "normal") -> bool:
              """トレードアラート送信"""
              emoji_map = {
                  "normal": "📊",
                  "warning": "⚠️",
                  "critical": "🚨",
                  "success": "✅",
                  "failure": "❌"
              }
              emoji = emoji_map.get(urgency, "📊")

        try:
                      self.client.chat_postMessage(
                                        channel=self.config.channel_alerts,
                                        text=f"{emoji} {message}",
                                        mrkdwn=True
                      )
                      return True
except SlackApiError as e:
              print(f"Slack送信エラー: {e}")
              return False

    def send_trade_log(self, trade_log: dict) -> bool:
              """トレードログ送信"""
              score = trade_log.get("score", {})
              grade = score.get("grade", "N/A")
              total = score.get("total", 0)

        blocks = [
                      {
                                        "type": "header",
                                        "text": {
                                                              "type": "plain_text",
                                                              "text": f"Trade Log: {trade_log['trade_id']}"
                                        }
                      },
                      {
                                        "type": "section",
                                        "fields": [
                                                              {"type": "mrkdwn", "text": f"*Symbol:* {trade_log['symbol']}"},
                                                              {"type": "mrkdwn", "text": f"*Direction:* {trade_log['direction'].upper()}"},
                                                              {"type": "mrkdwn", "text": f"*Lot:* {trade_log['lot']}"},
                                                              {"type": "mrkdwn", "text": f"*Slippage:* {trade_log['slippage_pips']:.2f}pips"},
                                                              {"type": "mrkdwn", "text": f"*Score:* {grade} ({total})"},
                                                              {"type": "mrkdwn", "text": f"*PnL:* {trade_log.get('pnl_pips', 'N/A')}pips"}
                                        ]
                      }
        ]

        try:
                      self.client.chat_postMessage(
                                        channel=self.config.channel_logs,
                                        blocks=blocks,
                                        text=f"Trade Log: {trade_log['trade_id']}"
                      )
                      return True
except SlackApiError as e:
              print(f"Slack送信エラー: {e}")
              return False

    def send_ai_analysis(self, analysis: str, trade_id: str) -> bool:
              """AI分析結果送信"""
              try:
                            self.client.chat_postMessage(
                                              channel=self.config.channel_ai,
                                              text=f"AI Analysis for {trade_id}\n\n{analysis}",
                                              mrkdwn=True
                            )
                            return True
except SlackApiError as e:
            print(f"Slack送信エラー: {e}")
            return False
