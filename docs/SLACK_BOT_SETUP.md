# Slack Bot Setup Guide

## 1. Create Slack App

1. Go to https://api.slack.com/apps
2. 2. Click "Create New App"
   3. 3. Select "From scratch"
      4. 4. Enter App Name: "CFD Smart Entry Bot"
         5. 5. Select your workspace
           
            6. ## 2. Configure Bot Permissions
           
            7. Navigate to "OAuth & Permissions" and add these Bot Token Scopes:
           
            8. - `chat:write` - Send messages
               - - `chat:write.public` - Send to public channels
                 - - `channels:read` - View channel info
                   - - `channels:history` - View message history
                    
                     - ## 3. Install App to Workspace
                    
                     - 1. Click "Install to Workspace"
                       2. 2. Authorize the permissions
                          3. 3. Copy the "Bot User OAuth Token" (starts with `xoxb-`)
                            
                             4. ## 4. Create Channels
                            
                             5. Create these channels in your Slack workspace:
                            
                             6. - `#trade-alerts` - Real-time trade notifications
                                - - `#trade-logs` - Complete trade logs
                                  - - `#ai-analysis` - AI analysis results
                                   
                                    - ## 5. Invite Bot to Channels
                                   
                                    - In each channel, type `/invite @CFD Smart Entry Bot`
                                   
                                    - ## 6. Configure Environment
                                   
                                    - Set the environment variable:
                                   
                                    - ```bash
                                      export SLACK_BOT_TOKEN=xoxb-your-token-here
                                      ```

                                      Or add to `.env` file:

                                      ```
                                      SLACK_BOT_TOKEN=xoxb-your-token-here
                                      ```

                                      ## 7. Test Connection

                                      ```python
                                      from slack_integration import SlackIntegration, SlackConfig

                                      config = SlackConfig(bot_token="xoxb-your-token")
                                      slack = SlackIntegration(config)
                                      slack.send_trade_alert("Test message", "success")
                                      ```

                                      ## Troubleshooting

                                      **Error: `channel_not_found`**
                                      - Invite the bot to the channel
                                     
                                      - **Error: `not_in_channel`**
                                      - - Use `/invite @CFD Smart Entry Bot` in the channel
                                       
                                        - **Error: `invalid_auth`**
                                        - - Check your bot token is correct
                                          - - Reinstall the app if needed
