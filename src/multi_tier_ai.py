#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multi_tier_ai.py
多層AI連携モジュール
Tier1: 24H監視 (Gemini Flash)
Tier2: トレード分析 (DeepSeek V3)
Tier3: メイン推論 (Claude 4.5 Opus)
"""

import os
import json
import httpx
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict
from enum import Enum


class AITier(Enum):
      """AI階層"""
      SENTINEL = "tier1"      # 24H監視
    ANALYST = "tier2"       # トレード分析
    INFERENCE = "tier3"     # メイン推論


@dataclass
class AIResponse:
      """AI応答"""
      tier: AITier
      model: str
      content: str
      tokens_used: int
      latency_ms: float
      cost_usd: float


class BaseAIClient(ABC):
      """AI クライアント基底クラス"""

    @abstractmethod
    async def analyze(self, prompt: str, context: Optional[Dict] = None) -> AIResponse:
              pass


class GeminiFlashClient(BaseAIClient):
      """Tier1: Gemini 2.0 Flash (24H監視用)"""

    def __init__(self):
              self.api_key = os.getenv("GOOGLE_API_KEY")
              self.model = "gemini-2.0-flash-exp"
              self.endpoint = "https://generativelanguage.googleapis.com/v1beta/models"
              self.cost_per_1m_input = 0.10
              self.cost_per_1m_output = 0.40

    async def analyze(self, prompt: str, context: Optional[Dict] = None) -> AIResponse:
              import time
              start = time.perf_counter()

        async with httpx.AsyncClient() as client:
                      response = await client.post(
                                        f"{self.endpoint}/{self.model}:generateContent?key={self.api_key}",
                                        json={
                                                              "contents": [{"parts": [{"text": prompt}]}],
                                                              "generationConfig": {
                                                                                        "temperature": 0.1,
                                                                                        "maxOutputTokens": 1024
                                                              }
                                        },
                                        timeout=30.0
                      )
                      data = response.json()

        latency = (time.perf_counter() - start) * 1000
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        tokens = data.get("usageMetadata", {}).get("totalTokenCount", 0)
        cost = tokens / 1_000_000 * self.cost_per_1m_input

        return AIResponse(
                      tier=AITier.SENTINEL,
                      model=self.model,
                      content=content,
                      tokens_used=tokens,
                      latency_ms=latency,
                      cost_usd=cost
        )


class DeepSeekClient(BaseAIClient):
      """Tier2: DeepSeek V3 (GitHub Models経由)"""

    def __init__(self):
              self.token = os.getenv("GITHUB_TOKEN")
              self.model = "deepseek-v3"
              self.endpoint = "https://models.inference.ai.azure.com/chat/completions"
              self.daily_limit = 150
              self.token_limit = 65536

    async def analyze(self, prompt: str, context: Optional[Dict] = None) -> AIResponse:
              import time
              start = time.perf_counter()

        # コンテキスト圧縮（64K制限対策）
              if context:
                            context_str = json.dumps(context, ensure_ascii=False)
                            if len(context_str) > 30000:
                                              context_str = context_str[:30000] + "...[truncated]"
                                          prompt = f"Context:\n{context_str}\n\nTask:\n{prompt}"

              async with httpx.AsyncClient() as client:
                            response = await client.post(
                                              self.endpoint,
                                              headers={
                                                                    "Authorization": f"Bearer {self.token}",
                                                                    "Content-Type": "application/json"
                                              },
                                              json={
                                                                    "model": self.model,
                                                                    "messages": [{"role": "user", "content": prompt}],
                                                                    "temperature": 0.3,
                                                                    "max_tokens": 4096
                                              },
                                              timeout=60.0
                            )
                            data = response.json()

              latency = (time.perf_counter() - start) * 1000
              content = data["choices"][0]["message"]["content"]
              tokens = data.get("usage", {}).get("total_tokens", 0)

        return AIResponse(
                      tier=AITier.ANALYST,
                      model=self.model,
                      content=content,
                      tokens_used=tokens,
                      latency_ms=latency,
                      cost_usd=0.0  # GitHub Models は無料
        )


class ClaudeOpusClient(BaseAIClient):
      """Tier3: Claude 4.5 Opus (メイン推論)"""

    def __init__(self):
              self.api_key = os.getenv("ANTHROPIC_API_KEY")
              self.model = "claude-opus-4-5-20251101"
              self.endpoint = "https://api.anthropic.com/v1/messages"
              self.cost_per_1m_input = 15.0
              self.cost_per_1m_output = 75.0

    async def analyze(self, prompt: str, context: Optional[Dict] = None) -> AIResponse:
              import time
              start = time.perf_counter()

        messages = [{"role": "user", "content": prompt}]

        system_prompt = """あなたは金融トレーディングの専門家AIです。
        トレード分析、リスク評価、改善提案を行います。
        回答は日本語で、具体的な数値と根拠を含めてください。"""

        async with httpx.AsyncClient() as client:
                      response = await client.post(
                                        self.endpoint,
                                        headers={
                                                              "x-api-key": self.api_key,
                                                              "anthropic-version": "2023-06-01",
                                                              "Content-Type": "application/json"
                                        },
                                        json={
                                                              "model": self.model,
                                                              "max_tokens": 4096,
                                                              "system": system_prompt,
                                                              "messages": messages
                                        },
                                        timeout=120.0
                      )
                      data = response.json()

        latency = (time.perf_counter() - start) * 1000
        content = data["content"][0]["text"]
        input_tokens = data["usage"]["input_tokens"]
        output_tokens = data["usage"]["output_tokens"]
        cost = (input_tokens / 1_000_000 * self.cost_per_1m_input +
                                output_tokens / 1_000_000 * self.cost_per_1m_output)

        return AIResponse(
                      tier=AITier.INFERENCE,
                      model=self.model,
                      content=content,
                      tokens_used=input_tokens + output_tokens,
                      latency_ms=latency,
                      cost_usd=cost
        )


class MultiTierAI:
      """多層AI統合コントローラー"""

    def __init__(self):
              self.tier1 = GeminiFlashClient()
              self.tier2 = DeepSeekClient()
              self.tier3 = ClaudeOpusClient()
              self.tier2_daily_count = 0

    async def sentinel_check(self, market_data: Dict) -> AIResponse:
              """Tier1: 24H監視チェック"""
              prompt = f"""以下の市況データを分析し、異常があれば報告してください。

      市況データ:
      {json.dumps(market_data, ensure_ascii=False, indent=2)}

      回答形式:
      - 異常検知: あり/なし
      - 詳細: (あれば)
      - 推奨アクション: (あれば)"""

        return await self.tier1.analyze(prompt)

    async def analyze_trade(self, trade_log: Dict) -> AIResponse:
              """Tier2: トレード分析"""
              # DeepSeek日次制限チェック
              if self.tier2_daily_count >= 150:
                            return await self.tier1.analyze(
                                              f"トレード分析:\n{json.dumps(trade_log, ensure_ascii=False)}"
                            )

        prompt = f"""以下のトレードログを分析し、採点と改善点を提示してください。

        トレードログ:
        {json.dumps(trade_log, ensure_ascii=False, indent=2)}

        回答形式:
        1. プロセス分析: どのように約定に至ったか
        2. 結果評価: 良かった点・悪かった点
        3. 採点: 0-100点
        4. 改善提案: 具体的なアクション"""

        self.tier2_daily_count += 1
        return await self.tier2.analyze(prompt)

    async def main_inference(self, query: str, context: Optional[Dict] = None) -> AIResponse:
              """Tier3: メイン推論（重要判断）"""
              prompt = query
              if context:
                            prompt = f"コンテキスト:\n{json.dumps(context, ensure_ascii=False)}\n\n質問:\n{query}"

              return await self.tier3.analyze(prompt)

    def reset_daily_counts(self):
              """日次カウントリセット"""
              self.tier2_daily_count = 0
