#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
voice_command_parser.py
音声コマンドパーサー（Whisper使用）
"""

import whisper
import sounddevice as sd
import numpy as np
import re
from dataclasses import dataclass
from typing import Optional, Tuple
import queue
import threading


@dataclass
class VoiceCommand:
      """パース済み音声コマンド"""
      symbol: str
      direction: str          # "buy" or "sell"
    is_urgent: bool
    lot: Optional[float]
    raw_text: str


class VoiceCommandParser:
      """音声コマンドパーサー"""

    # 銘柄マッピング
      SYMBOL_MAP = {
          "ユーロドル": "EURUSD",
          "eurusd": "EURUSD",
          "ドル円": "USDJPY",
          "usdjpy": "USDJPY",
          "ポンドドル": "GBPUSD",
          "gbpusd": "GBPUSD",
          "ゴールド": "XAUUSD",
          "gold": "XAUUSD",
          "ビットコイン": "BTCUSD",
          "bitcoin": "BTCUSD",
      }

    # 緊急トリガーワード
      URGENT_TRIGGERS = ["急ぎ", "緊急", "今すぐ", "すぐ", "urgent", "now"]

    def __init__(self, model_size: str = "base"):
              """
                      Args:
                                  model_size: "tiny", "base", "small", "medium", "large"
              """
              self.model = whisper.load_model(model_size)
              self.sample_rate = 16000
              self.audio_queue = queue.Queue()
              self.is_listening = False

    def transcribe(self, audio: np.ndarray) -> str:
              """音声をテキストに変換"""
              result = self.model.transcribe(
                  audio,
                  language="ja",
                  fp16=False
              )
              return result["text"].strip()

    def parse_command(self, text: str) -> Optional[VoiceCommand]:
              """テキストからコマンドをパース"""
              text_lower = text.lower()

        # 緊急度判定
              is_urgent = any(trigger in text_lower for trigger in self.URGENT_TRIGGERS)

        # 銘柄抽出
              symbol = None
              for key, value in self.SYMBOL_MAP.items():
                            if key in text_lower:
                                              symbol = value
                                              break

                        if symbol is None:
                                      return None

        # 方向抽出
        direction = None
        if any(w in text_lower for w in ["買", "ロング", "buy", "long"]):
                      direction = "buy"
elif any(w in text_lower for w in ["売", "ショート", "sell", "short"]):
            direction = "sell"

        if direction is None:
                      return None

        # ロット抽出（オプション）
        lot_match = re.search(r"(\d+\.?\d*)\s*(ロット|lot)", text_lower)
        lot = float(lot_match.group(1)) if lot_match else None

        return VoiceCommand(
                      symbol=symbol,
                      direction=direction,
                      is_urgent=is_urgent,
                      lot=lot,
                      raw_text=text
        )

    def record_audio(self, duration: float = 5.0) -> np.ndarray:
              """音声録音"""
        audio = sd.rec(
                      int(duration * self.sample_rate),
                      samplerate=self.sample_rate,
                      channels=1,
                      dtype=np.float32
        )
        sd.wait()
        return audio.flatten()

    def listen_once(self, duration: float = 5.0) -> Optional[VoiceCommand]:
              """一度だけ聞いてコマンドを返す"""
        print("🎤 録音中...")
        audio = self.record_audio(duration)
        print("📝 文字起こし中...")
        text = self.transcribe(audio)
        print(f"認識結果: {text}")
        return self.parse_command(text)


# 使用例
if __name__ == "__main__":
      parser = VoiceCommandParser(model_size="base")

    # テスト: テキストから直接パース
    test_texts = [
              "ユーロドル買い急ぎ！",
              "ゴールド売りで0.1ロット",
              "USDJPY long now",
    ]

    for text in test_texts:
              cmd = parser.parse_command(text)
        print(f"入力: {text}")
        print(f"結果: {cmd}")
        print("-" * 40)
