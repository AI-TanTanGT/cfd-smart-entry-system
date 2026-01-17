# CFD Smart Entry System

CFD Smart Entry System - Slippage reduction with multi-tier AI, voice input, and MT5 integration

## Features

- **Slippage Reduction**: Intelligent order execution to minimize slippage
- **Multi-tier AI Analysis**: Layered AI system for market analysis and entry point prediction
- **Voice Input**: Speech recognition for hands-free trading commands
- **MT5 Integration**: Seamless integration with MetaTrader 5 for order execution

## Project Structure

```
cfd-smart-entry-system/
├── src/
│   └── cfd_smart_entry/
│       ├── __init__.py         # Package initialization
│       ├── core/               # Core trading logic and order management
│       ├── ai/                 # Multi-tier AI components
│       ├── voice/              # Voice input handling
│       ├── mt5/                # MetaTrader 5 integration
│       └── utils/              # Shared utilities
├── tests/
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
├── docs/                       # Documentation
├── config/                     # Configuration files
├── pyproject.toml              # Project configuration
├── requirements.txt            # Dependencies
├── .env.example                # Environment variables template
└── README.md                   # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/AI-TanTanGT/cfd-smart-entry-system.git
cd cfd-smart-entry-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your MT5 credentials
```

## Configuration

1. Copy `.env.example` to `.env` and fill in your MT5 credentials
2. Adjust settings in `config/default.yaml` as needed

## Development

```bash
# Run tests
pytest

# Format code
black src tests
isort src tests

# Type checking
mypy src
```

## License

MIT License
