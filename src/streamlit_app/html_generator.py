# src/streamlit_app/html_generator.py

def create_dual_chatbot_html(session_id: str, model1: str, model2: str, prompt_style: int, api_key: str, system_prompt: str, langsmith_api_key: str, project_name: str) -> str:
    """Create HTML page with dual chatbots that automatically extracts and sends results to Streamlit."""

    prompt_style_text = "Professional" if prompt_style == 0 else "Consultative" if prompt_style == 1 else "Friendly"
    
    # Safely escape backticks, backslashes, and dollars signs for JS template literal
    system_prompt_js = system_prompt.replace("\", "\\\\").replace("`", "\`").replace("$", "\$")

    # Use .format() with named placeholders to avoid f-string conflicts with CSS/JS
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dual Investment Advisor Comparison - Enhanced</title>
        <style>
            * {{ 
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                text-align: center;
                color: white;
                margin-bottom: 30px;
            }}
            .enhanced-badge {{
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
                color: white;
                padding: 10px 20px;
                border-radius: 25px;
                display: inline-block;
                margin-bottom: 10px;
                font-weight: bold;
            }}
            .session-info {{
                background: rgba(255, 255, 255, 0.95);
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            }}
            .sync-controls {{
                background: rgba(255, 255, 255, 0.95);
                border-radius: 15px;
                padding: 20px;
                text-align: center;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
                margin-bottom: 20px;
            }}
            .sync-input {{
                width: 100%;
                max-width: 600px;
                padding: 15px;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                margin-bottom: 15px;
                font-size: 16px;
            }}
            .sync-button {{
                padding: 15px 30px;
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
                margin: 0 10px;
            }}
            .enhanced-button {{
                padding: 15px 30px;
                background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
                margin: 0 10px;
            }}
            .enhanced-button:disabled {{
                background: #6c757d;
                cursor: not-allowed;
            }}
            .status-display {{
                background: rgba(255, 255, 255, 0.9);
                padding: 15px;
                border-radius: 10px;
                margin-top: 15px;
                border-left: 4px solid #007bff;
            }}
            .extraction-display {{
                background: rgba(255, 255, 255, 0.9);
                padding: 15px;
                border-radius: 10px;
                margin-top: 15px;
                border-left: 4px solid #ff6b6b;
            }}
            .chatbots-container {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }}
            .chatbot-panel {{
                background: rgba(255, 255, 255, 0.95);
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
                overflow: hidden;
                height: 600px;
                display: flex;
                flex-direction: column;
            }}
            .chatbot-header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px;
                text-align: center;
                font-weight: bold;
            }}
            .chat-messages {{
                flex: 1;
                overflow-y: auto;
                padding: 20px;
                background: #f8f9fa;
            }}
            .message {{
                margin-bottom: 15px;
                animation: fadeIn 0.3s ease;
            }}
            .message.user {{
                text-align: right;
            }}
            .message-content {{
                display: inline-block;
                max-width: 80%;
                padding: 12px 16px;
                border-radius: 15px;
                line-height: 1.5;
            }}
            .message.user .message-content {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }}
            .message.bot .message-content {{
                background: white;
                border: 1px solid #ddd;
            }}
            .chat-input {{
                padding: 15px;
                background: white;
                border-top: 1px solid #ddd;
            }}
            .input-form {{
                display: flex;
                gap: 10px;
            }}
            .input-form input {{
                flex: 1;
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 25px;
                font-size: 14px;
            }}
            .input-form button {{
                padding: 12px 24px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                font-weight: bold;
            }}
            .typing-indicator {{
                display: none;
                padding: 10px;
                font-style: italic;
                color: #666;
            }}
            .typing-indicator.active {{
                display: block;
            }}
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(10px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="enhanced-badge">Enhanced Analytics</div>
                <h1>Dual Investment Advisor Comparison</h1>
                <p>Session ID: {session_id}</p>
            </div>
            <div class="session-info">
                <h3>Session Configuration</h3>
                <p><strong>Selected Prompt Style:</strong> {prompt_style_num} - {prompt_style_text}</p>
                <p><strong>System Prompt:</strong> {system_prompt}</p>
                <p><strong>Enhanced Analytics:</strong> Enabled - Responses will be extracted for advanced analysis</p>
            </div>
            <div class="sync-controls">
                <h3>Start a New Comparison</h3>
                <input type="text" id="syncInput" class="sync-input" placeholder="Enter your investment question here..." />
                <div>
                    <button class="sync-button" onclick="sendToBoth()">Compare Models</button>
                    <button class="enhanced-button" onclick="downloadResults()">Download Results File</button>
                </div>
                <div class="status-display" id="statusDisplay">
                    Status: Ready to receive your question
                </div>
                <div class="extraction-display" id="extractionDisplay" style="display: none;">
                    Data Extraction: Ready to extract responses for analysis
                </div>
            </div>
            <div class="chatbots-container">
                <div class="chatbot-panel">
                    <div class="chatbot-header">
                        {model1} - Investment Advisor
                    </div>
                    <div class="chat-messages" id="chatMessages1">
                        <div class="message bot">
                            <div class="message-content">
                                Hello! I'm your {model1} investment advisor. I'm ready to help you with your investment questions.
                            </div>
                        </div>
                    </div>
                    <div class="typing-indicator" id="typing1">
                        <span>Advisor is typing...</span>
                    </div>
                    <div class="chat-input">
                        <div class="input-form">
                            <input type="text" id="userInput1" placeholder="Ask about investments..." onkeypress="if(event.key === 'Enter') sendMessage('1')" />
                            <button onclick="sendMessage('1')" id="sendBtn1">Send</button>
                        </div>
                    </div>
                </div>
                <div class="chatbot-panel">
                    <div class="chatbot-header">
                        {model2} - Investment Advisor
                    </div>
                    <div class="chat-messages" id="chatMessages2">
                        <div class="message bot">
                            <div class="message-content">
                                Hello! I'm your {model2} investment advisor. I'm ready to help you with your investment questions.
                            </div>
                        </div>
                    </div>
                    <div class="typing-indicator" id="typing2">
                        <span>Advisor is typing...</span>
                    </div>
                    <div class="chat-input">
                        <div class="input-form">
                            <div class="input-form">
                                <input type="text" id="userInput2" placeholder="Ask about investments..." onkeypress="if(event.key === 'Enter') sendMessage('2')" />
                                <button onclick="sendMessage('2')" id="sendBtn2">Send</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const sessionId = '{session_id}';
            const openAiApiKey = '{api_key}';
            const langsmithApiKey = '{langsmith_api_key}';
            const langsmithProjectName = '{project_name}';
            const models = {{
                '1': '{model1}',
                '2': '{model2}'
            }};
            const systemPrompt = `{system_prompt_js}`;
            
            let extractedData = {{}};
            let responseCount = 0;

            function resetExtractionData() {{
                extractedData = {{
                    userPrompt: '',
                    responses: {{}},
                    models: ['{model1}', '{model2}'],
                    promptStyle: {prompt_style_num_js},
                    sessionId: sessionId,
                    timestamp: ''
                }};
                responseCount = 0;
            }}

            resetExtractionData();

            function updateStatus(message, type = 'info') {{
                const statusDisplay = document.getElementById('statusDisplay');
                statusDisplay.innerHTML = `Status: ${{message}}`;
                
                if (type === 'success') {{
                    statusDisplay.style.borderLeftColor = '#28a745';
                }} else if (type === 'warning') {{
                    statusDisplay.style.borderLeftColor = '#ffc107';
                }} else if (type === 'error') {{
                    statusDisplay.style.borderLeftColor = '#dc3545';
                }} else {{
                    statusDisplay.style.borderLeftColor = '#007bff';
                }}
            }}

            function updateExtraction(message, type = 'info') {{
                const extractionDisplay = document.getElementById('extractionDisplay');
                extractionDisplay.style.display = 'block';
                extractionDisplay.innerHTML = `Data Extraction: ${{message}}`;
                
                if (type === 'success') {{
                    extractionDisplay.style.borderLeftColor = '#28a745';
                }} else if (type === 'warning') {{
                    extractionDisplay.style.borderLeftColor = '#ffc107';
                }} else if (type === 'error') {{
                    extractionDisplay.style.borderLeftColor = '#dc3545';
                }} else {{
                    extractionDisplay.style.borderLeftColor = '#ff6b6b';
                }}
            }}

            async function sendMessage(chatbotId) {{
                const input = document.getElementById(`userInput${{chatbotId}}`);
                const message = input.value.trim();
                
                if (!message) return;

                if (!extractedData.userPrompt) {{
                    extractedData.userPrompt = message;
                    extractedData.timestamp = new Date().toISOString();
                    updateExtraction(`User prompt extracted: "${{message.substring(0, 50)}}"...