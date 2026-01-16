# CFD Smart Entry System

CFD Smart Entry System - Slippage reduction with multi-tier AI, voice input, and MT5 integration

## Overview

CFDスマートエントリーシステムは、CFD取引において最適なエントリータイミングを提供するための高度な取引支援システムです。マルチティアAI分析、音声入力、スリッページ軽減機能、およびMetaTrader 5との統合を特徴としています。

The CFD Smart Entry System is an advanced trading support system designed to provide optimal entry timing for CFD trading. It features multi-tier AI analysis, voice input, slippage reduction, and MetaTrader 5 integration.

## Features

### 🤖 Multi-Tier AI Signal Generation
- **Tier 1 (>80% confidence)**: High-confidence signals for immediate execution
- **Tier 2 (65-80% confidence)**: Standard signals with good probability
- **Tier 3 (50-65% confidence)**: Lower confidence signals requiring additional confirmation

### 📉 Slippage Reduction Engine
- Intelligent order execution strategies
- Dynamic price tolerance adjustment
- Order splitting for large positions
- Market condition-based execution tier assessment

### 🎤 Voice Input (Japanese/English)
- Japanese commands: 買い, 売り, 決済, 状況
- English commands: buy, sell, close, status
- Symbol recognition: ドル円, ユーロドル, ゴールド, etc.

### 📊 Market Analysis
- Trend detection using multiple moving averages
- Volatility measurement
- Support/Resistance level calculation
- Volume analysis

### 🔗 MT5 Integration
- Real-time market data retrieval
- Order execution and management
- Position monitoring
- Account information access

## Installation

```bash
# Clone the repository
git clone https://github.com/AI-TanTanGT/cfd-smart-entry-system.git
cd cfd-smart-entry-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Edit `config/settings.yaml` to customize the system:

```yaml
# MT5 Connection Settings
mt5:
  server: "Your-Broker-Server"
  timeout: 60000

# Trading Parameters
trading:
  symbols:
    - "USDJPY"
    - "EURUSD"
  default_lot_size: 0.01
  max_positions: 5

# AI Signal Settings
ai_signal:
  confidence_threshold: 0.65
  model_type: "ensemble"

# Slippage Settings
slippage:
  max_slippage_pips: 3.0
  execution_timeout_ms: 500

# Voice Settings
voice:
  enabled: true
  language: "ja-JP"
```

## Usage

### Basic Usage

```python
from src.main import CFDSmartEntrySystem

# Initialize the system
system = CFDSmartEntrySystem()

# Connect to MT5
if system.connect(login=12345678, password="your_password"):
    # Start the system
    system.start()

    # Get system status
    status = system.get_status()
    print(status)

    # Execute text command
    system.execute_text_command("ドル円を買い")

    # Stop the system
    system.stop()
    system.disconnect()
```

### Using Individual Components

```python
from src.ai_signal_generator import AISignalGenerator
from src.market_analyzer import MarketAnalyzer
from src.slippage_reducer import SlippageReducer

# Market Analysis
analyzer = MarketAnalyzer()
condition = analyzer.analyze("USDJPY", ohlcv_data, current_spread=0.02)

# Signal Generation
signal_gen = AISignalGenerator(confidence_threshold=0.65)
signal = signal_gen.generate_signal("USDJPY", ohlcv_data, current_price=130.0)

# Slippage Reduction
reducer = SlippageReducer()
plan = reducer.create_execution_plan("USDJPY", is_buy=True, quantity=0.1, 
                                      current_price=130.0, condition=condition)
```

### Voice Commands

| Japanese | English | Action |
|----------|---------|--------|
| 買い / ロング | buy / long | Place buy order |
| 売り / ショート | sell / short | Place sell order |
| 決済 | close | Close position |
| 全決済 | close all | Close all positions |
| 状況 | status | Show status |
| 停止 | stop | Stop system |

## Project Structure

```
cfd-smart-entry-system/
├── config/
│   ├── __init__.py
│   └── settings.yaml        # Configuration file
├── src/
│   ├── __init__.py
│   ├── config.py            # Configuration loader
│   ├── market_analyzer.py   # Market data analysis
│   ├── ai_signal_generator.py  # AI-based signal generation
│   ├── slippage_reducer.py  # Slippage reduction engine
│   ├── voice_input.py       # Voice input handler
│   ├── mt5_connector.py     # MT5 integration
│   ├── order_executor.py    # Order execution orchestration
│   └── main.py              # Main application
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_market_analyzer.py
│   ├── test_ai_signal_generator.py
│   ├── test_slippage_reducer.py
│   └── test_voice_input.py
├── requirements.txt
└── README.md
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_market_analyzer.py -v
```

## Requirements

- Python 3.10+
- MetaTrader 5 terminal (for MT5 integration)
- Microphone (for voice input)

## Dependencies

- MetaTrader5
- pandas
- numpy
- scikit-learn
- SpeechRecognition
- PyAudio
- loguru
- pyyaml
- python-dotenv

## Risk Warning

⚠️ **重要**: このシステムは投資アドバイスを提供するものではありません。CFD取引には大きなリスクが伴い、投資した資金をすべて失う可能性があります。

⚠️ **Important**: This system does not provide investment advice. CFD trading involves significant risks and you may lose all your invested capital.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
