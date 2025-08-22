# src/streamlit_app/ui.py
import streamlit as st
import os
import uuid
import webbrowser
import tempfile
import json
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.core.test_suite import InvestmentTestSuite, TestCase, TestCaseType
from src.core.langsmith_analyzer import LangSmithAnalyzer
from src.core.model_manager import EnhancedModelManager
from src.streamlit_app.html_generator import create_dual_chatbot_html

LANGSMITH_AVAILABLE = True
try:
    from langsmith import Client
except ImportError:
    LANGSMITH_AVAILABLE = False

def create_dual_chatbot_interface_with_langsmith(api_key: str, langsmith_api_key: str = None, project_name: str = "investment-chatbot-comparison"):
    """Create the dual chatbot comparison interface with optional LangSmith integration."""
    
    analyzer = None
    if langsmith_api_key and LANGSMITH_AVAILABLE:
        st.header("Real-Time Dual Chatbot Comparison with LangSmith Analytics")
        st.markdown(f"Compare investment advisor responses with advanced LangSmith-powered analysis. Logging to project: **{project_name}**")
        
        # Initialize LangSmith analyzer
        if 'langsmith_analyzer' not in st.session_state or \
           st.session_state.langsmith_analyzer.project_name != project_name:
            try:
                st.session_state.langsmith_analyzer = LangSmithAnalyzer(langsmith_api_key, project_name)
                st.success(f"Successfully connected to LangSmith project: '{project_name}'")
            except (ImportError, ConnectionError) as e:
                st.error(f"LangSmith Initialization Failed: {e}")
                st.session_state.langsmith_analyzer = None
        
        analyzer = st.session_state.langsmith_analyzer
        
        # LangSmith Project Status
        if analyzer:
            with st.expander("LangSmith Project Status"):
                analytics = analyzer.get_project_analytics()
                if "error" in analytics:
                    st.error(analytics["error"])
                else:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Runs", analytics.get("total_runs", 0))
                    with col2:
                        st.metric("Avg Response Length", f"{analytics.get('avg_response_length', 0):.0f} words")
                    with col3:
                        models_used = analytics.get("models_used", {})
                        most_used = max(models_used.items(), key=lambda x: x[1])[0] if models_used else "None"
                        st.metric("Most Used Model", most_used)
                
                if st.button("Fetch Last LangSmith Trace"):
                    with st.spinner("Fetching last trace..."):
                        last_trace = analyzer.get_last_trace()
                        if last_trace:
                            st.subheader("Last Registered Trace")
                            st.json(last_trace.dict())
                        else:
                            st.warning("No traces found in the project.")
    else:
        st.header("Real-Time Dual Chatbot Comparison")
        st.markdown("Compare investment advisor responses with advanced analysis.")
        
        if not LANGSMITH_AVAILABLE:
            st.info("Install LangSmith for enhanced analytics: `pip install langsmith langchain langchain-openai`")
    
    # Initialize session state for dual chatbot
    if 'chatbot_sessions' not in st.session_state:
        st.session_state.chatbot_sessions = {}
    
    # Prompt style selection
    st.subheader("1. Select Prompt Style")
    
    prompt_descriptions = [
        "Professional - Formal, comprehensive investment advice with detailed analysis",
        "Consultative - Structured approach with bullet points and strategic recommendations", 
        "Friendly - Conversational, approachable tone with step-by-step guidance"
    ]
    
    selected_prompt_style = st.selectbox(
        "Choose the prompt style for both models:",
        options=[0, 1, 2],
        format_func=lambda x: prompt_descriptions[x]
    )
    
    # Show selected prompt
    test_suite = InvestmentTestSuite()
    
    with st.expander("Preview Selected Prompt"):
        st.write(test_suite.prompt_variations[selected_prompt_style])
    
    # Model selection
    st.subheader("2. Model Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        model1 = st.selectbox("Model 1:", ["gpt-4o-mini", "gpt-4.1-nano"], index=0)
    
    with col2:
        model2 = st.selectbox("Model 2:", ["gpt-4o-mini", "gpt-4.1-nano"], index=1)
    
    if model1 == model2:
        st.error("Please select different models for comparison.")
        return
    
    # Launch dual chatbots
    st.subheader("3. Launch Dual Chatbots")
    
    if st.button("Launch Dual Chatbot Interface", type="primary"):
        if not langsmith_api_key:
            st.error("Please enter your LangSmith API key to launch the chatbot with tracing.")
            return
        # Generate session ID
        session_id = str(uuid.uuid4())[:8]
        
        # Create HTML content for dual chatbots
        html_content = create_dual_chatbot_html(session_id, model1, model2, selected_prompt_style, api_key, test_suite.prompt_variations[selected_prompt_style], langsmith_api_key, project_name)
        
        # Save to temporary file and open
        temp_dir = tempfile.gettempdir()
        html_file = os.path.join(temp_dir, f"dual_chatbot_{session_id}.html")
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Store session info
        st.session_state.current_session = {
            'id': session_id,
            'models': [model1, model2],
            'prompt_style': selected_prompt_style,
            'html_file': html_file
        }
        
        # Open in browser
        file_url = f"file://{html_file}"
        webbrowser.open(file_url)
        
        st.success(f"Dual chatbot interface launched! Session ID: {session_id}")
        st.info("The chatbot interface opened in your browser. Send the same message to both models, then return here for analysis.")
    
    # Auto-collection section with optional LangSmith
    analysis_header = "4. Automatic Response Collection & Analysis"
    if analyzer:
        analysis_header += " with LangSmith"
    st.subheader(analysis_header)
    
    if 'current_session' in st.session_state:
        session_info = st.session_state.current_session
        st.write(f"Active Session: {session_info['id']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Check for New Results", type="secondary"):
                # Check API for results
                try:
                    import requests
                    response = requests.get(f"http://localhost:8000/get-dual-results/{session_info['id']}")
                    if response.status_code == 200:
                        result_data = response.json()
                        if analyzer:
                            auto_analyze_collected_responses_with_langsmith(result_data, analyzer)
                        else:
                            auto_analyze_collected_responses(result_data)
                    else:
                        st.info("No results found via API. Use manual entry below.")
                        st.rerun()
                except:
                    st.info("API not available. Use manual entry below.")
        
        with col2:
            if st.button("Load from Browser Storage"):
                st.info("If automatic collection isn't working, use the 'Download Results File' button in the HTML interface, then upload the file below.")
        
        # File upload section
        upload_header = "Upload Results File"
        if analyzer:
            upload_header += " for LangSmith Analysis"
        st.subheader(upload_header)
        
        uploaded_file = st.file_uploader(
            "Upload session results JSON file", 
            type=['json'],
            help="Download the results file from the HTML chatbot interface and upload it here"
        )
        
        if uploaded_file is not None:
            try:
                # Read and parse JSON file
                file_contents = uploaded_file.read()
                result_data = json.loads(file_contents)
                
                st.success(f"File uploaded successfully! Session: {result_data.get('session_id', 'Unknown')}")
                
                # Automatically analyze the uploaded data with or without LangSmith
                if analyzer:
                    auto_analyze_collected_responses_with_langsmith(result_data, analyzer)
                else:
                    auto_analyze_collected_responses(result_data)
                
            except json.JSONDecodeError:
                st.error("Invalid JSON file. Please ensure you uploaded the correct results file.")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # Manual input section
    manual_header = "5. Manual Input"
    if analyzer:
        manual_header += " with LangSmith Analysis"
    st.subheader(manual_header)
    
    with st.expander("Manual Response Entry"):
        user_prompt_manual = st.text_input("User Question:", key="manual_prompt_enhanced")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Model 1 Response:")
            response1_manual = st.text_area("Response 1:", height=200, key="manual_resp1_enhanced")
        
        with col2:
            st.write("Model 2 Response:")
            response2_manual = st.text_area("Response 2:", height=200, key="manual_resp2_enhanced")
        
        button_text = "Analyze with LangSmith" if analyzer else "Analyze Responses"
        
        if st.button(button_text, type="primary"):
            if user_prompt_manual and response1_manual and response2_manual and 'current_session' in st.session_state:
                manual_data = {
                    'user_prompt': user_prompt_manual,
                    'responses': {
                        st.session_state.current_session['models'][0]: response1_manual,
                        st.session_state.current_session['models'][1]: response2_manual
                    },
                    'models': st.session_state.current_session['models'],
                    'prompt_style': st.session_state.current_session['prompt_style'],
                    'session_id': st.session_state.current_session['id']
                }
                
                if analyzer:
                    auto_analyze_collected_responses_with_langsmith(manual_data, analyzer)
                else:
                    auto_analyze_collected_responses(manual_data)
            else:
                st.error("Please fill in all fields and ensure you have an active session.")

def auto_analyze_collected_responses_with_langsmith(result_data, analyzer: LangSmithAnalyzer):
    """Automatically analyze collected responses with LangSmith integration."""
    
    st.header("LangSmith-Powered Analysis Results")
    
    # Extract data
    user_prompt = result_data['user_prompt']
    responses = result_data['responses']
    models = result_data['models']
    
    # Display collection info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Session", result_data.get('session_id', 'Manual'))
    with col2:
        st.metric("Models Compared", len(models))
    with col3:
        st.metric("Prompt Style", f"Style {result_data.get('prompt_style', 0) + 1}")
    
    # Show extracted data
    st.subheader("Extracted Data")
    st.write(f"**User Question:** {user_prompt}")
    
    # Display responses
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**{models[0]} Response:**")
        with st.expander(f"{models[0]} Full Response", expanded=True):
            st.write(responses[models[0]])
    
    with col2:
        st.write(f"**{models[1]} Response:**")
        with st.expander(f"{models[1]} Full Response", expanded=True):
            st.write(responses[models[1]])
    
    # Since logging is now automatic, we just need to run the local evaluation for charts.
    st.info("LLM calls are now automatically traced to LangSmith. The analysis below is a local evaluation.")

    # Evaluate responses using LangSmith
    with st.spinner("Running local evaluations for charts..."):
        evaluation_results = analyzer.evaluate_responses(user_prompt, responses, models)
    
    # Display LangSmith evaluation results
    st.subheader("LangSmith Evaluation Results")
    
    # Winner announcement
    winner = max(evaluation_results.items(), key=lambda x: x[1]['overall_score'])
    st.success(f"LangSmith Winner: {winner[0]} with score {winner[1]['overall_score']:.2f}")
    
    # Summary comparison
    col1, col2 = st.columns(2)
    with col1:
        model1_score = evaluation_results[models[0]]['overall_score']
        st.metric(f"{models[0]} LangSmith Score", f"{model1_score:.2f}")
    
    with col2:
        model2_score = evaluation_results[models[1]]['overall_score']
        st.metric(f"{models[1]} LangSmith Score", f"{model2_score:.2f}")
    
    # Generate and display LangSmith charts
    st.subheader("LangSmith Analysis Visualizations")
    
    charts = analyzer.generate_comparison_charts(evaluation_results, models)
    
    # Display charts in organized layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(charts["overall_comparison"], use_container_width=True)
        if "radar_comparison" in charts:
            st.plotly_chart(charts["radar_comparison"], use_container_width=True)
    
    with col2:
        if "quality_breakdown" in charts:
            st.plotly_chart(charts["quality_breakdown"], use_container_width=True)
    
    # Detailed LangSmith evaluation breakdown
    st.subheader("Detailed LangSmith Evaluation Results")
    
    evaluation_df_data = []
    for model in models:
        result = evaluation_results[model]
        row = {
            'Model': model,
            'Overall Score': result['overall_score'],
            'Quality Score': result['quality_score'],
            'Completeness Score': result['completeness_score'],
            'Word Count': result['completeness_details']['word_count'],
            'Structure Bonus': result['completeness_details']['structure_bonus']
        }
        
        # Add quality details
        for criterion, score in result['quality_details'].items():
            row[f'Quality: {criterion.replace("_", " ").title()}'] = score
        
        evaluation_df_data.append(row)
    
    evaluation_df = pd.DataFrame(evaluation_df_data)
    st.dataframe(evaluation_df.round(3), use_container_width=True)
    
    # Detailed breakdown by model
    for model in models:
        with st.expander(f"LangSmith Analysis: {model}"):
            result = evaluation_results[model]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Quality Score", f"{result['quality_score']:.3f}")
                st.write("**Quality Breakdown:**")
                for criterion, score in result["quality_details"].items():
                    st.write(f"- {criterion.replace('_', ' ').title()}: {score:.3f}")
            
            with col2:
                st.metric("Completeness Score", f"{result['completeness_score']:.3f}")
                st.write("**Completeness Details:**")
                details = result["completeness_details"]
                st.write(f"- Word Count: {details['word_count']}")
                st.write(f"- Structure Bonus: +{details['structure_bonus']:.3f}")
    
    # Key insights from LangSmith analysis
    st.subheader("LangSmith Key Insights")
    
    score_diff = abs(model1_score - model2_score)
    if score_diff < 0.1:
        st.info("LangSmith analysis shows the models performed very similarly on this query.")
    elif score_diff < 0.3:
        st.info(f"LangSmith analysis shows {winner[0]} had a slight advantage over the other model.")
    else:
        st.success(f"LangSmith analysis shows {winner[0]} significantly outperformed the other model.")
    
    # Best performing criteria from LangSmith
    for model in models:
        result = evaluation_results[model]
        quality_details = result['quality_details']
        best_criterion = max(quality_details.items(), key=lambda x: x[1])
        worst_criterion = min(quality_details.items(), key=lambda x: x[1])
        
        st.write(f"**LangSmith Analysis - {model}:**")
        st.write(f"- Strongest: {best_criterion[0].replace('_', ' ').title()} ({best_criterion[1]:.3f})")
        st.write(f"- Needs improvement: {worst_criterion[0].replace('_', ' ').title()} ({worst_criterion[1]:.3f})")
    
    # LangSmith project link
    st.subheader("View in LangSmith")
    st.info(f"Visit your LangSmith project '{analyzer.project_name}' to see detailed traces and additional analytics.")
    
    st.success("LangSmith analysis completed!")

def auto_analyze_collected_responses(result_data):
    """Automatically analyze collected responses from dual chatbot (original function)."""
    
    st.header("Automatic Analysis Results")
    
    # Display collection info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Session", result_data.get('session_id', 'Manual'))
    with col2:
        st.metric("Models Compared", len(result_data.get('models', [])))
    with col3:
        st.metric("Prompt Style", f"Style {result_data.get('prompt_style', 0) + 1}")
    
    # Show collected data
    st.subheader("Collected Data")
    st.write(f"**User Question:** {result_data['user_prompt']}")
    
    # Display responses
    col1, col2 = st.columns(2)
    models = result_data['models']
    responses = result_data['responses']
    
    with col1:
        st.write(f"**{models[0]} Response:**")
        with st.expander(f"{models[0]} Full Response", expanded=True):
            st.write(responses[models[0]])
    
    with col2:
        st.write(f"**{models[1]} Response:**")
        with st.expander(f"{models[1]} Full Response", expanded=True):
            st.write(responses[models[1]])
    
    # Run automatic evaluation
    try:
        # Create test case for user query
        user_test_case = TestCase(
            id="auto_collected",
            name="Auto-Collected User Query",
            type=TestCaseType.BASIC,
            variables={
                "user_query": result_data['user_prompt'],
                "prompt_style": str(result_data['prompt_style'])
            },
            expected_elements=[
                "investment advice",
                "specific recommendations",
                "risk considerations"
            ],
            validation_criteria={
                "discusses_risks": True,
                "provides_actionable_advice": True
            },
            difficulty=2,
            tags=["auto_collected", "real_time"]
        )
        
        # Initialize evaluator
        from src.core.evaluators import InvestmentModelEvaluator
        evaluator = InvestmentModelEvaluator()
        
        # Evaluate both responses
        evaluation_results = {}
        
        for model, response in responses.items():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(
                evaluator.evaluate_response(response, user_test_case, model)
            )
            
            evaluation_results[model] = result
            loop.close()
        
        # Display results
        st.subheader("Evaluation Results")
        
        # Winner announcement
        winner = max(evaluation_results.items(), key=lambda x: x[1]['overall_score'])
        st.success(f"Winner: {winner[0]} with score {winner[1]['overall_score']:.1f}/10")
        
        # Summary comparison
        col1, col2 = st.columns(2)
        with col1:
            model1_score = evaluation_results[models[0]]['overall_score']
            st.metric(f"{models[0]} Score", f"{model1_score:.1f}/10")
        
        with col2:
            model2_score = evaluation_results[models[1]]['overall_score']
            st.metric(f"{models[1]} Score", f"{model2_score:.1f}/10")
        
        # Detailed results table
        results_data = []
        for model, result in evaluation_results.items():
            row = {'Model': model, 'Overall Score': result['overall_score']}
            row.update(result['scores'])
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        
        st.subheader("Detailed Score Breakdown")
        st.dataframe(results_df.round(2), use_container_width=True)
        
        # Visualization
        fig = px.bar(
            results_df,
            x='Model',
            y='Overall Score',
            title="Automatic Comparison Results",
            color='Overall Score',
            color_continuous_scale='viridis'
        )
        fig.update_layout(yaxis_range=[0, 10])
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed criteria comparison
        criteria_cols = [col for col in results_df.columns if col not in ['Model', 'Overall Score']]
        
        fig_radar = go.Figure()
        for _, row in results_df.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row[col] for col in criteria_cols],
                theta=criteria_cols,
                fill='toself',
                name=row['Model'],
                opacity=0.7
            ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            title="Detailed Criteria Comparison"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Key insights
        st.subheader("Key Insights")
        
        score_diff = abs(model1_score - model2_score)
        if score_diff < 0.5:
            st.info("The models performed very similarly on this query.")
        elif score_diff < 1.5:
            st.info(f"{winner[0]} had a slight advantage over the other model.")
        else:
            st.success(f"{winner[0]} significantly outperformed the other model.")
        
        # Best performing criteria
        for model, result in evaluation_results.items():
            best_criterion = max(result['scores'].items(), key=lambda x: x[1])
            worst_criterion = min(result['scores'].items(), key=lambda x: x[1])
            
            st.write(f"**{model}:**")
            st.write(f"- Strongest: {best_criterion[0].title()} ({best_criterion[1]:.1f}/10)")
            st.write(f"- Needs improvement: {worst_criterion[0].title()} ({worst_criterion[1]:.1f}/10)")
        
        st.success("Automatic analysis completed!")
        
    except Exception as e:
        st.error(f"Error during automatic analysis: {str(e)}")
        st.info("You can still view the collected responses above.")

def create_standard_comparison_interface(api_key: str):
    """Create the standard test case comparison interface."""
    
    # Initialize components
    test_suite = InvestmentTestSuite()
    model_manager = EnhancedModelManager(api_key)
    
    # Test case selection
    st.sidebar.subheader("Test Scenario")
    test_case_options = {tc.name: tc.id for tc in test_suite.test_cases}
    selected_test_name = st.sidebar.selectbox("Choose test scenario:", list(test_case_options.keys()))
    selected_test_id = test_case_options[selected_test_name]

    # Model selection
    st.sidebar.subheader("Models")
    available_models = list(model_manager.available_models.keys())
    selected_models = st.sidebar.multiselect(
        "Select models to compare:",
        available_models,
        default=available_models
    )
    
    # Prompt variation
    prompt_variation = st.sidebar.selectbox(
        "Prompt style:",
        options=[0, 1, 2],
        format_func=lambda x: ["Professional", "Consultative", "Friendly"][x]
    )
    
    # Display test case details
    selected_test_case = next(tc for tc in test_suite.test_cases if tc.id == selected_test_id)
    
    st.header("Test Scenario Details")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Test Case: {selected_test_case.name}")
        st.write(f"**Type:** {selected_test_case.type.value}")
        st.write(f"**Difficulty:** {selected_test_case.difficulty}/5")
        st.write(f"**Tags:** {', '.join(selected_test_case.tags)}")
    
    with col2:
        st.subheader("Client Profile")
        for key, value in selected_test_case.variables.items():
            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    
    # Run comparison
    if st.button("Run Model Comparison", type="primary"):
        if len(selected_models) < 2:
            st.error("Select at least 2 models for comparison")
            return
        
        with st.spinner("Running model comparison..."):
            try:
                # Run async comparison
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                analysis = loop.run_until_complete(
                    model_manager.run_comparison(selected_models, selected_test_id, prompt_variation)
                )
                loop.close()
                
                # Display results
                st.header("Comparison Results")
                
                # Winner announcement
                winner_result = analysis.results[analysis.winner]
                st.success(f"**Winner: {analysis.winner}** with score {winner_result.overall_score}/10")
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Score", f"{analysis.summary_stats['mean_score']}/10")
                with col2:
                    st.metric("Score Range", f"{analysis.summary_stats['score_range']}")
                with col3:
                    st.metric("Winner Advantage", f"+{analysis.summary_stats['winner_advantage']}")
                with col4:
                    st.metric("Score Std Dev", f"{analysis.summary_stats['score_std']}")
                
                # Detailed results table
                st.subheader("Detailed Scores")
                results_data = []
                for model, result in analysis.results.items():
                    row = {'Model': model, 'Overall Score': result.overall_score}
                    row.update(result.scores)
                    results_data.append(row)
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df.round(2), use_container_width=True)
                
                # Basic visualizations
                st.subheader("Performance Visualizations")
                
                # Overall scores bar chart
                fig_bar = px.bar(
                    results_df, 
                    x='Model', 
                    y='Overall Score',
                    title="Overall Model Performance",
                    color='Overall Score',
                    color_continuous_scale='viridis'
                )
                fig_bar.update_layout(yaxis_range=[0, 10])
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Radar chart for detailed criteria
                criteria_cols = [col for col in results_df.columns if col not in ['Model', 'Overall Score']]
                
                fig_radar = go.Figure()
                for _, row in results_df.iterrows():
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[row[col] for col in criteria_cols],
                        theta=criteria_cols,
                        fill='toself',
                        name=row['Model'],
                        opacity=0.7
                    ))
                
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                    title="Detailed Criteria Comparison"
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                
                # Model responses
                st.subheader("Model Responses")
                for model, result in analysis.results.items():
                    with st.expander(f"{model} Response (Score: {result.overall_score}/10)"):
                        st.write(result.response)
                
                st.success("Model comparison completed successfully!")
                
            except Exception as e:
                st.error(f"Error running comparison: {str(e)}")
