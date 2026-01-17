# CFD Smart Entry System - Handover Document

## Project Overview

**Purpose**: Automated trading system to reduce slippage in CFD market orders

**Axis**: Production Implementation (Axis 3) + SOTA Elements (Axis 2)
- Production: Slippage reduction directly impacts P&L
- - SOTA: VPIN and other advanced monitoring techniques
 
  - ## System Architecture
 
  - ### Processing Flow
  - 1. Voice input "EURUSD buy urgent!" -> Whisper -> Command parse
    2. 2. AI MONITOR LAYER (24H): Spread monitoring, VPIN calculation
       3. 3. INFERENCE LAYER: Order type/price/slip tolerance decision
          4. 4. EXECUTION LAYER (Python): MT5 order_send()
             5. 5. EA LAYER (MT5): Weekly pivot, 200 SMA filters
               
                6. ### Multi-tier AI
                7. | Tier | Purpose | Model | Cost |
                8. |------|---------|-------|------|
                9. | 1 | 24H Monitor | Gemini Flash | ~$0.01/hr |
                10. | 2 | Trade Analysis | DeepSeek V3 | Free (150 RPD) |
                11. | 3 | Main Inference | Claude 4.5 Opus | $15/1M input |
               
                12. ## Implementation Files
               
                13. ### Python Modules (src/)
                14. 1. `smart_entry_controller.py` - Smart limit order controller
                    2. 2. `voice_command_parser.py` - Voice command parser (Whisper)
                       3. 3. `trade_log_schema.py` - Trade log schema
                          4. 4. `trade_scorer.py` - Scoring engine
                             5. 5. `multi_tier_ai.py` - Multi-tier AI integration
                                6. 6. `slack_integration.py` - Slack integration
                                  
                                   7. ## BE Calculation
                                   8. ```
                                      BE(pips) = Commission(0.7) + Spread + Hidden(0.2) + Slip + Latency(0.1)
                                               = ~1.6 pips (EURUSD)
                                      ```

                                      ## Environment Variables
                                      ```bash
                                      SLACK_BOT_TOKEN=xoxb-your-token
                                      GOOGLE_API_KEY=your-google-api-key
                                      GITHUB_TOKEN=your-github-token
                                      ANTHROPIC_API_KEY=your-anthropic-key
                                      ```

                                      ## Setup
                                      ```bash
                                      cd cfd-smart-entry-system
                                      python -m venv .venv
                                      source .venv/bin/activate
                                      pip install MetaTrader5 numpy openai-whisper sounddevice slack-sdk httpx
                                      ```

                                      ## Next Steps (Priority)
                                      1. Backtest integration
                                      2. 2. Prometheus/Grafana dashboard
                                         3. 3. DeepSeek fallback (Gemini Pro)
                                            4. 4. Slack Bot setup guide
                                               5. 5. Voice recognition improvement
                                                  6. 6. Docker Compose
                                                    
                                                     7. ---
                                                     8. Last Updated: 2025-01-18
