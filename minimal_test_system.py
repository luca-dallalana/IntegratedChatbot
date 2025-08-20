# minimal_test_system.py - Simplified version for testing Phase 1
import asyncio
import json
import re
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import statistics

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import openai

# Simplified Test Case Classes
class TestCaseType(Enum):
    BASIC = "basic"
    EDGE_CASE = "edge_case"
    STRESS_TEST = "stress_test"

@dataclass
class TestCase:
    id: str
    name: str
    type: TestCaseType
    variables: Dict[str, str]
    expected_elements: List[str]
    validation_criteria: Dict[str, Any]
    difficulty: int
    tags: List[str]

class SimpleInvestmentTestSuite:
    """Simplified test suite for Phase 1 testing."""
    
    def __init__(self):
        self.test_cases = []
        self.prompt_variations = []
        self._load_test_cases()
        self._load_prompt_variations()

    def _load_test_cases(self):
        """Load simplified test cases."""
        self.test_cases = [
            TestCase(
                id="etf_beginner",
                name="Beginner investor exploring ETFs",
                type=TestCaseType.BASIC,
                variables={
                    "client_income": "$65,000",
                    "risk_tolerance": "moderate",
                    "investment_goal": "long-term growth",
                    "time_horizon": "15 years",
                    "investment_amount": "$10,000",
                    "preferred_assets": "ETFs"
                },
                expected_elements=[
                    "ETF diversification options",
                    "expense ratio discussion",
                    "risk vs return analysis"
                ],
                validation_criteria={
                    "mentions_diversification": True,
                    "explains_fees": True
                },
                difficulty=1,
                tags=["etf", "beginner"]
            ),
            
            TestCase(
                id="crypto_interest",
                name="Client curious about cryptocurrency",
                type=TestCaseType.BASIC,
                variables={
                    "client_income": "$80,000",
                    "risk_tolerance": "high",
                    "investment_goal": "short-term speculation",
                    "time_horizon": "2 years",
                    "investment_amount": "$5,000",
                    "preferred_assets": "cryptocurrency"
                },
                expected_elements=[
                    "volatility warning",
                    "crypto allocation guidance"
                ],
                validation_criteria={
                    "mentions_volatility": True,
                    "discusses_security": True
                },
                difficulty=2,
                tags=["crypto", "high_risk"]
            )
        ]

    def _load_prompt_variations(self):
        """Load prompt variations."""
        self.prompt_variations = [
            """You are a professional investment advisor. A client has approached you with the following information:

Client Income: {client_income}
Investment Goal: {investment_goal}
Risk Tolerance: {risk_tolerance}
Time Horizon: {time_horizon}
Available Investment: {investment_amount}
Preferred Assets: {preferred_assets}

Please provide comprehensive investment advice including allocation strategies, risk management, and next steps."""
        ]

class SimpleModelEvaluator:
    """Simplified evaluator for Phase 1 testing."""
    
    def __init__(self):
        self.evaluation_criteria = {
            'accuracy': 0.30,
            'completeness': 0.25,
            'helpfulness': 0.20,
            'clarity': 0.15,
            'relevance': 0.10
        }
    
    def evaluate_response(self, response: str, test_case: TestCase, model_name: str) -> Dict[str, Any]:
        """Simple evaluation for testing."""
        # Basic scoring
        scores = {}
        
        # Accuracy - simple keyword matching
        accuracy_score = 0.7  # Base score
        for criterion, expected in test_case.validation_criteria.items():
            if self._check_simple_criterion(response, criterion):
                accuracy_score += 0.1
        scores['accuracy'] = min(10.0, accuracy_score * 10)
        
        # Completeness - check expected elements
        completeness_score = 0.6  # Base score
        for element in test_case.expected_elements:
            if any(word.lower() in response.lower() for word in element.split()):
                completeness_score += 0.1
        scores['completeness'] = min(10.0, completeness_score * 10)
        
        # Helpfulness - look for actionable advice
        helpful_words = ['recommend', 'suggest', 'should', 'consider']
        helpfulness_score = 0.5 + (sum(word in response.lower() for word in helpful_words) * 0.1)
        scores['helpfulness'] = min(10.0, helpfulness_score * 10)
        
        # Clarity - basic readability
        word_count = len(response.split())
        sentences = max(1, response.count('.') + response.count('!') + response.count('?'))
        avg_length = word_count / sentences
        clarity_score = max(0.3, 1.0 - abs(avg_length - 15) * 0.02)
        scores['clarity'] = min(10.0, clarity_score * 10)
        
        # Relevance - investment keywords
        investment_words = ['investment', 'portfolio', 'risk', 'return', 'etf', 'crypto']
        relevance_score = 0.4 + (sum(word in response.lower() for word in investment_words) * 0.1)
        scores['relevance'] = min(10.0, relevance_score * 10)
        
        # Overall score
        overall_score = sum(scores[c] * w for c, w in self.evaluation_criteria.items())
        
        return {
            'scores': scores,
            'overall_score': round(overall_score, 2),
            'response': response,
            'model': model_name,
            'test_case': test_case.name
        }
    
    def _check_simple_criterion(self, response: str, criterion: str) -> bool:
        """Simple criterion checking."""
        response_lower = response.lower()
        
        patterns = {
            "mentions_diversification": "diversif",
            "explains_fees": "fee",
            "mentions_volatility": "volatil",
            "discusses_security": "secur"
        }
        
        pattern = patterns.get(criterion, criterion)
        return pattern in response_lower

class SimpleModelManager:
    """Simplified model manager for Phase 1."""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.available_models = {
            'gpt-4o-mini': 'gpt-4o-mini',
            'gpt-4.1-nano': 'gpt-4.1-nano'
        }
        self.test_suite = SimpleInvestmentTestSuite()
        self.evaluator = SimpleModelEvaluator()
    
    async def get_model_response(self, model_name: str, prompt: str) -> str:
        """Get response from model."""
        try:
            response = self.client.chat.completions.create(
                model=self.available_models[model_name],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error with {model_name}: {str(e)}"
    
    async def run_simple_comparison(self, models: List[str], test_case_id: str) -> Dict[str, Any]:
        """Run simplified comparison."""
        # Get test case
        test_case = next((tc for tc in self.test_suite.test_cases if tc.id == test_case_id), None)
        if not test_case:
            raise ValueError(f"Test case {test_case_id} not found")
        
        # Format prompt
        prompt = self.test_suite.prompt_variations[0].format(**test_case.variables)
        
        # Get responses
        results = {}
        for model in models:
            response = await self.get_model_response(model, prompt)
            evaluation = self.evaluator.evaluate_response(response, test_case, model)
            results[model] = evaluation
        
        # Find winner
        winner = max(results.items(), key=lambda x: x[1]['overall_score'])[0]
        
        # Calculate stats
        all_scores = [result['overall_score'] for result in results.values()]
        summary_stats = {
            'mean_score': round(statistics.mean(all_scores), 2),
            'score_range': round(max(all_scores) - min(all_scores), 2),
            'winner_advantage': round(results[winner]['overall_score'] - min(all_scores), 2)
        }
        
        return {
            'results': results,
            'winner': winner,
            'test_case': {
                'id': test_case.id,
                'name': test_case.name,
                'difficulty': test_case.difficulty
            },
            'summary_stats': summary_stats
        }

def create_simple_streamlit_app():
    """Simple Streamlit app for Phase 1 testing."""
    
    st.set_page_config(
        page_title="Investment Chatbot Test - Phase 1",
        page_icon="🧪",
        layout="wide"
    )
    
    st.title("🧪 Investment Chatbot Test - Phase 1")
    st.markdown("Simplified testing version to verify core functionality")
    
    # Sidebar
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    
    if not api_key:
        st.warning("Enter your OpenAI API key to begin testing")
        return
    
    # Initialize manager
    try:
        model_manager = SimpleModelManager(api_key)
        st.sidebar.success("✅ System initialized")
    except Exception as e:
        st.sidebar.error(f"❌ Initialization error: {str(e)}")
        return
    
    # Test case selection
    st.sidebar.subheader("Test Configuration")
    test_case_options = {tc.name: tc.id for tc in model_manager.test_suite.test_cases}
    selected_test_name = st.sidebar.selectbox("Test scenario:", list(test_case_options.keys()))
    selected_test_id = test_case_options[selected_test_name]
    
    # Model selection
    available_models = list(model_manager.available_models.keys())
    selected_models = st.sidebar.multiselect(
        "Models to compare:",
        available_models,
        default=available_models
    )
    
    # Display test case
    selected_test_case = next(tc for tc in model_manager.test_suite.test_cases if tc.id == selected_test_id)
    
    st.header("Test Scenario")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"📋 {selected_test_case.name}")
        st.write(f"**Difficulty:** {selected_test_case.difficulty}/5")
        st.write(f"**Type:** {selected_test_case.type.value}")
    
    with col2:
        st.subheader("Client Profile")
        for key, value in selected_test_case.variables.items():
            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    
    # Run test button
    if st.button("🚀 Run Simple Test", type="primary"):
        if len(selected_models) < 2:
            st.error("Select at least 2 models for comparison")
            return
        
        with st.spinner("Running simplified test..."):
            try:
                # Run async comparison
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(
                    model_manager.run_simple_comparison(selected_models, selected_test_id)
                )
                loop.close()
                
                # Display results
                st.header("Test Results")
                
                # Winner
                winner_result = results['results'][results['winner']]
                st.success(f"**Winner: {results['winner']}** with score {winner_result['overall_score']:.1f}/10")
                
                # Summary stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Score", f"{results['summary_stats']['mean_score']}/10")
                with col2:
                    st.metric("Score Range", f"{results['summary_stats']['score_range']}")
                with col3:
                    st.metric("Winner Advantage", f"+{results['summary_stats']['winner_advantage']}")
                
                # Detailed results
                st.subheader("Detailed Scores")
                results_data = []
                for model, result in results['results'].items():
                    row = {'Model': model, 'Overall Score': result['overall_score']}
                    row.update(result['scores'])
                    results_data.append(row)
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df.round(2), use_container_width=True)
                
                # Simple bar chart
                fig = px.bar(
                    results_df, 
                    x='Model', 
                    y='Overall Score',
                    title="Model Performance Comparison",
                    color='Overall Score',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(yaxis_range=[0, 10])
                st.plotly_chart(fig, use_container_width=True)
                
                # Model responses
                st.subheader("Model Responses")
                for model, result in results['results'].items():
                    with st.expander(f"{model} Response (Score: {result['overall_score']:.1f}/10)"):
                        st.write(result['response'])
                
                st.success("✅ Phase 1 test completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Test failed: {str(e)}")
                st.write("**Debugging info:**")
                st.write(f"- API key provided: {'Yes' if api_key else 'No'}")
                st.write(f"- Selected models: {selected_models}")
                st.write(f"- Test case: {selected_test_id}")

if __name__ == "__main__":
    create_simple_streamlit_app()