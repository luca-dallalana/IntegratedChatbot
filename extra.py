# integrated_investment_system_with_langsmith.py - Complete System with LangSmith Integration
import asyncio
import json
import re
import time
import os
import tempfile
import webbrowser
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import statistics
import numpy as np

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openai
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Import metrics dashboard functions
from metrics_dashboard import (
    display_advanced_metrics,
    display_response_analysis,
    display_comparative_insights
)

# Import LangSmith integration
from langsmith_integration import (
    LangSmithIntegration,
    integrate_langsmith_with_streamlit,
    display_langsmith_results
)

# [Keep all existing classes: TestCaseType, TestCase, InvestmentTestSuite, InvestmentModelEvaluator]
# ... (keeping all the existing classes from your original file)

# Enhanced Model Manager with LangSmith
class EnhancedModelManagerWithLangSmith:
    """Enhanced model manager with investment focus and LangSmith integration."""
    
    def __init__(self, api_key: str, langsmith_integration: Optional[LangSmithIntegration] = None):
        self.client = openai.OpenAI(api_key=api_key)
        self.available_models = {
            'gpt-4o-mini': 'gpt-4o-mini',
            'gpt-4.1-nano': 'gpt-4.1-nano'
        }
        self.test_suite = InvestmentTestSuite()
        self.evaluator = InvestmentModelEvaluator()
        self.langsmith = langsmith_integration
    
    async def get_model_response(self, model_name: str, prompt: str, max_tokens: int = 500) -> str:
        """Get response from specified model."""
        try:
            response = self.client.chat.completions.create(
                model=self.available_models[model_name],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error with {model_name}: {str(e)}"
    
    async def run_comparison(self, models: List[str], test_case_id: str, 
                           prompt_variation: int = 0) -> ComparisonAnalysis:
        """Run comprehensive model comparison with optional LangSmith tracking."""
        
        # Get test case
        test_case = next((tc for tc in self.test_suite.test_cases if tc.id == test_case_id), None)
        if not test_case:
            raise ValueError(f"Test case {test_case_id} not found")
        
        # Get prompt variation
        if prompt_variation >= len(self.test_suite.prompt_variations):
            prompt_variation = 0
        
        prompt_template = self.test_suite.prompt_variations[prompt_variation]
        formatted_prompt = f"{prompt_template}\n\nClient situation: {test_case.variables}"
        
        # Create LangSmith experiment if integration is available
        experiment_id = None
        if self.langsmith:
            experiment_id = self.langsmith.create_comparison_experiment(
                experiment_name=f"comparison-{test_case_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                models=models,
                test_cases=[{
                    "id": test_case.id,
                    "name": test_case.name,
                    "variables": test_case.variables,
                    "difficulty": test_case.difficulty
                }]
            )
        
        # Get responses from all models
        results = {}
        model_responses = {}
        tasks = [self.get_model_response(model, formatted_prompt) for model in models]
        responses = await asyncio.gather(*tasks)
        
        for model, response in zip(models, responses):
            model_responses[model] = response
        
        # Run LangSmith evaluation if available
        langsmith_results = None
        if self.langsmith:
            langsmith_results = await self.langsmith.run_langsmith_evaluation(
                model_responses=model_responses,
                test_case={
                    "id": test_case.id,
                    "name": test_case.name,
                    "variables": test_case.variables,
                    "difficulty": test_case.difficulty,
                    "expected_elements": test_case.expected_elements,
                    "validation_criteria": test_case.validation_criteria
                },
                prompt=formatted_prompt
            )
        
        # Evaluate each response using the existing evaluator
        for model, response in model_responses.items():
            evaluation_result = await self.evaluator.evaluate_response(response, test_case, model)
            
            # Log to LangSmith if available
            if self.langsmith:
                await self.langsmith.log_comparison_run(
                    model_name=model,
                    prompt=formatted_prompt,
                    response=response,
                    test_case={
                        "id": test_case.id,
                        "name": test_case.name,
                        "variables": test_case.variables,
                        "difficulty": test_case.difficulty
                    },
                    evaluation_scores=evaluation_result['scores'],
                    run_metadata={
                        "experiment_id": experiment_id,
                        "prompt_variation": prompt_variation
                    }
                )
            
            # Create ModelComparisonResult with LangSmith data if available
            result_data = ModelComparisonResult(
                model=model,
                response=evaluation_result['response'],
                scores=evaluation_result['scores'],
                overall_score=evaluation_result['overall_score'],
                test_case_name=test_case.name,
                timestamp=datetime.now().isoformat(),
                metrics=evaluation_result.get('metrics')
            )
            
            # Add LangSmith scores if available
            if langsmith_results and model in langsmith_results:
                result_data.langsmith_evaluation = langsmith_results[model]
            
            results[model] = result_data
        
        # Find winner
        winner = max(results.items(), key=lambda x: x[1].overall_score)[0]
        
        # Calculate summary statistics
        all_scores = [result.overall_score for result in results.values()]
        summary_stats = {
            'mean_score': round(statistics.mean(all_scores), 2),
            'score_std': round(statistics.stdev(all_scores) if len(all_scores) > 1 else 0, 2),
            'score_range': round(max(all_scores) - min(all_scores), 2),
            'winner_advantage': round(results[winner].overall_score - min(all_scores), 2)
        }
        
        # Add LangSmith experiment ID if available
        if experiment_id:
            summary_stats['langsmith_experiment_id'] = experiment_id
        
        return ComparisonAnalysis(
            results=results,
            winner=winner,
            test_case={
                'id': test_case.id,
                'name': test_case.name,
                'type': test_case.type.value,
                'difficulty': test_case.difficulty,
                'tags': test_case.tags
            },
            summary_stats=summary_stats,
            langsmith_results=langsmith_results
        )

# Enhanced Comparison Analysis with LangSmith
class ComparisonAnalysis(BaseModel):
    results: Dict[str, ModelComparisonResult]
    winner: str
    test_case: Dict[str, Any]
    summary_stats: Dict[str, float]
    langsmith_results: Optional[Dict[str, Any]] = None

class ModelComparisonResult(BaseModel):
    model: str
    response: str
    scores: Dict[str, float]
    overall_score: float
    test_case_name: str
    timestamp: str
    metrics: Optional[Dict[str, Any]] = None
    langsmith_evaluation: Optional[Dict[str, Any]] = None

# Enhanced Streamlit Application with LangSmith
def create_streamlit_app():
    """Create enhanced Streamlit application with metrics dashboard and LangSmith."""
    
    st.set_page_config(
        page_title="Investment Chatbot Model Comparison",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("💰 Investment Chatbot Model Comparison")
    st.markdown("Compare AI models on investment advisory scenarios with advanced metrics and LangSmith tracking")
    
    # Sidebar navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Select Mode:",
        ["Standard Test Cases", "Real-Time Dual Chatbot", "Metrics Dashboard", "LangSmith Analytics"]
    )
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    
    # LangSmith integration
    st.sidebar.divider()
    langsmith = integrate_langsmith_with_streamlit(st.sidebar)
    
    if not api_key and page not in ["Metrics Dashboard", "LangSmith Analytics"]:
        st.warning("Enter your OpenAI API key to begin")
        return
    
    # Initialize session state for results storage
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    if 'langsmith_integration' not in st.session_state:
        st.session_state.langsmith_integration = langsmith
    
    if page == "LangSmith Analytics":
        display_langsmith_analytics()
    elif page == "Metrics Dashboard":
        display_metrics_dashboard()
    elif page == "Real-Time Dual Chatbot":
        create_dual_chatbot_interface(api_key, langsmith)
    else:
        create_standard_comparison_interface(api_key, langsmith)

def display_langsmith_analytics():
    """Display LangSmith analytics dashboard."""
    st.header("📊 LangSmith Analytics")
    
    if st.session_state.analysis_results and hasattr(st.session_state.analysis_results, 'langsmith_results'):
        langsmith_results = st.session_state.analysis_results.langsmith_results
        
        if langsmith_results:
            # Display LangSmith evaluation results
            display_langsmith_results(langsmith_results)
            
            # Comparison chart
            st.subheader("LangSmith Score Comparison")
            
            models = list(langsmith_results.keys())
            scores = [result.get('overall_score', 0) for result in langsmith_results.values()]
            
            fig = px.bar(
                x=models,
                y=scores,
                title="LangSmith Overall Scores",
                labels={'x': 'Model', 'y': 'Score'},
                color=scores,
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed criteria breakdown
            st.subheader("Detailed Criteria Analysis")
            
            criteria_data = []
            for model, result in langsmith_results.items():
                if 'evaluations' in result:
                    for criterion, eval_data in result['evaluations'].items():
                        criteria_data.append({
                            'Model': model,
                            'Criterion': criterion.replace('_', ' ').title(),
                            'Score': eval_data['score']
                        })
            
            if criteria_data:
                criteria_df = pd.DataFrame(criteria_data)
                
                fig = px.bar(
                    criteria_df,
                    x='Criterion',
                    y='Score',
                    color='Model',
                    barmode='group',
                    title="LangSmith Criteria Scores by Model"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Export LangSmith results
            if st.button("Export LangSmith Results"):
                export_langsmith_results(langsmith_results)
        else:
            st.info("No LangSmith results available. Run a comparison with LangSmith enabled.")
    else:
        st.info("No analysis results available. Please run a comparison first with LangSmith enabled.")

def export_langsmith_results(langsmith_results):
    """Export LangSmith results to JSON."""
    json_str = json.dumps(langsmith_results, indent=2)
    
    st.download_button(
        label="Download LangSmith Results (JSON)",
        data=json_str,
        file_name=f"langsmith_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def create_standard_comparison_interface(api_key: str, langsmith: Optional[LangSmithIntegration] = None):
    """Create the standard test case comparison interface with LangSmith integration."""
    
    # Initialize components
    test_suite = InvestmentTestSuite()
    model_manager = EnhancedModelManagerWithLangSmith(api_key, langsmith)
    
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
    
    # LangSmith status indicator
    if langsmith:
        st.info("🔍 LangSmith tracking enabled - Results will be sent for advanced evaluation")
    
    # Run comparison
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Run Model Comparison", type="primary"):
            if len(selected_models) < 2:
                st.error("Select at least 2 models for comparison")
                return
            
            with st.spinner("Running model comparison with LangSmith tracking..." if langsmith else "Running model comparison..."):
                try:
                    # Run async comparison
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    analysis = loop.run_until_complete(
                        model_manager.run_comparison(selected_models, selected_test_id, prompt_variation)
                    )
                    loop.close()
                    
                    # Store results in session state
                    st.session_state.analysis_results = analysis
                    
                    # Display results
                    display_comparison_results_with_langsmith(analysis)
                    
                    st.success("Model comparison completed successfully!")
                    
                    if langsmith:
                        st.info("💡 Go to 'LangSmith Analytics' to see detailed LangSmith evaluation")
                    
                    st.info("💡 Go to 'Metrics Dashboard' for advanced analytics")
                    
                except Exception as e:
                    st.error(f"Error running comparison: {str(e)}")
    
    with col2:
        if st.button("View Metrics Dashboard", type="secondary"):
            if st.session_state.analysis_results:
                st.rerun()
            else:
                st.warning("Run a comparison first to generate metrics")
    
    with col3:
        if langsmith and st.button("View LangSmith Analytics", type="secondary"):
            if st.session_state.analysis_results:
                st.rerun()
            else:
                st.warning("Run a comparison first to generate LangSmith data")

def display_comparison_results_with_langsmith(analysis):
    """Display comparison results including LangSmith data if available."""
    st.header("Comparison Results")
    
    # Winner announcement
    winner_result = analysis.results[analysis.winner]
    st.success(f"**Winner: {analysis.winner}** with score {winner_result.overall_score}/10")
    
    # Check if LangSmith results are available
    if hasattr(analysis, 'langsmith_results') and analysis.langsmith_results:
        # Display both scoring systems
        st.subheader("Dual Scoring System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Internal Evaluation Scores:**")
            for model, result in analysis.results.items():
                st.metric(f"{model}", f"{result.overall_score}/10")
        
        with col2:
            st.write("**LangSmith Evaluation Scores:**")
            for model, ls_result in analysis.langsmith_results.items():
                if 'overall_score' in ls_result:
                    st.metric(f"{model}", f"{ls_result['overall_score']:.2%}")
    
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
        
        # Add LangSmith score if available
        if hasattr(result, 'langsmith_evaluation') and result.langsmith_evaluation:
            row['LangSmith Score'] = f"{result.langsmith_evaluation.get('overall_score', 0):.2%}"
        
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df.round(2), use_container_width=True)
    
    # Visualizations
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
    
    # Model responses
    st.subheader("Model Responses")
    for model, result in analysis.results.items():
        with st.expander(f"{model} Response (Score: {result.overall_score}/10)"):
            st.write(result.response)
            
            # Show LangSmith evaluation details if available
            if hasattr(result, 'langsmith_evaluation') and result.langsmith_evaluation:
                st.divider()
                st.write("**LangSmith Evaluation Details:**")
                if 'evaluations' in result.langsmith_evaluation:
                    for criterion, eval_data in result.langsmith_evaluation['evaluations'].items():
                        st.write(f"- {criterion.replace('_', ' ').title()}: {eval_data['score']:.2%} - {eval_data.get('comment', '')}")

def create_dual_chatbot_interface(api_key: str, langsmith: Optional[LangSmithIntegration] = None):
    """Create the dual chatbot comparison interface with LangSmith support."""
    
    st.header("Real-Time Dual Chatbot Comparison")
    st.markdown("Select a prompt style, launch dual chatbots, and automatically collect responses for analysis.")
    
    if langsmith:
        st.info("🔍 LangSmith tracking enabled for dual chatbot comparison")
    
    # [Rest of the dual chatbot interface code remains the same]
    # ... (keeping the existing dual chatbot implementation)

# Keep all other existing classes and functions
# [Include all the TestCase, TestCaseType, InvestmentTestSuite, InvestmentModelEvaluator classes]
# [Include all helper functions like auto_analyze_collected_responses, create_dual_chatbot_html, etc.]

# Main execution function
def main():
    """Main function to run the integrated system with LangSmith."""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "streamlit":
            create_streamlit_app()
        elif sys.argv[1] == "api":
            import uvicorn
            uvicorn.run(app, host="0.0.0.0", port=8000)
        else:
            print("Usage: python integrated_investment_system_with_langsmith.py [streamlit|api]")
    else:
        print("Investment Chatbot Comparison System with LangSmith")
        print("Usage:")
        print("  streamlit run integrated_investment_system_with_langsmith.py")
        print("  python integrated_investment_system_with_langsmith.py api")

if __name__ == "__main__":
    main()