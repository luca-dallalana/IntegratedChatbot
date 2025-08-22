# integrated_investment_system_enhanced.py - Complete System with Metrics Dashboard
# Com langsmith mas falhando na hora de criar chatbot
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

# LangSmith Integration
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Run, Example

# Test Suite Classes
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

class InvestmentTestSuite:
    """Test suite specifically for investment advisor scenarios."""
    
    def __init__(self):
        self.test_cases = []
        self.prompt_variations = []
        self._load_test_cases()
        self._load_prompt_variations()

    def _load_test_cases(self):
        """Load investment-specific test cases."""
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
                    "current_portfolio": "N/A",
                    "investment_amount": "$10,000",
                    "preferred_assets": "ETFs"
                },
                expected_elements=[
                    "ETF diversification options",
                    "expense ratio discussion",
                    "risk vs return analysis",
                    "expected long-term growth",
                    "timeline guidance"
                ],
                validation_criteria={
                    "mentions_diversification": True,
                    "explains_fees": True,
                    "analyzes_risk_return": True
                },
                difficulty=1,
                tags=["etf", "beginner", "long_term"]
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
                    "current_portfolio": "$20,000 in index funds",
                    "investment_amount": "$5,000",
                    "preferred_assets": "cryptocurrency"
                },
                expected_elements=[
                    "volatility warning",
                    "crypto allocation guidance",
                    "security/storage risks",
                    "regulatory considerations",
                    "tax implications"
                ],
                validation_criteria={
                    "mentions_volatility": True,
                    "discusses_security": True,
                    "explains_regulations": True
                },
                difficulty=2,
                tags=["crypto", "speculation", "high_risk"]
            ),
            
            TestCase(
                id="retiree_bonds",
                name="Retiree looking for stable income via bonds",
                type=TestCaseType.EDGE_CASE,
                variables={
                    "client_age": "68",
                    "client_income": "$40,000 (pension)",
                    "risk_tolerance": "low",
                    "investment_goal": "income stability",
                    "time_horizon": "10 years",
                    "current_portfolio": "$200,000 in mixed assets",
                    "investment_amount": "$100,000",
                    "preferred_assets": "bonds"
                },
                expected_elements=[
                    "bond types (treasury, municipal, corporate)",
                    "yield expectations",
                    "interest rate sensitivity",
                    "laddering strategy",
                    "tax considerations"
                ],
                validation_criteria={
                    "mentions_bond_types": True,
                    "explains_yields": True,
                    "discusses_interest_risks": True
                },
                difficulty=3,
                tags=["bonds", "retirement", "income_focus"]
            ),
            
            TestCase(
                id="high_net_worth_portfolio",
                name="High-net-worth client with diversified portfolio",
                type=TestCaseType.STRESS_TEST,
                variables={
                    "net_worth": "$5,000,000",
                    "client_income": "$500,000",
                    "risk_tolerance": "moderate",
                    "investment_goal": "capital preservation with moderate growth",
                    "time_horizon": "25 years",
                    "current_portfolio": "equities, real estate, private equity",
                    "investment_amount": "$1,000,000",
                    "preferred_assets": "ETFs, international bonds, REITs"
                },
                expected_elements=[
                    "asset allocation strategy",
                    "tax efficiency planning",
                    "international diversification",
                    "alternative investments",
                    "risk management"
                ],
                validation_criteria={
                    "mentions_allocation": True,
                    "explains_tax_strategies": True,
                    "discusses_risks": True
                },
                difficulty=5,
                tags=["high_net_worth", "diversification", "tax_planning"]
            )
        ]

    def _load_prompt_variations(self):
        """Load investment advisor prompt variations for testing."""
        self.prompt_variations = [
            "You are a professional investment advisor. Please provide comprehensive investment advice including allocation strategies, risk management, and next steps.",
            
            "As an experienced financial consultant, I want to help you navigate your investment journey. Let me provide you with a detailed analysis of your options, probability of success, and strategic recommendations.",
            
            "Hi there! I'm excited to help you with your investments. I'll walk you through potential strategies, explain how different asset classes fit, and outline what steps we should take to move forward."
        ]

# LangSmith Integration Class
class LangSmithIntegration:
    """Integration layer for LangSmith evaluation and monitoring."""
    
    def __init__(self, api_key: Optional[str] = None, project_name: str = "investment-chatbot-comparison"):
        if api_key:
            os.environ["LANGSMITH_API_KEY"] = api_key
        
        if not os.environ.get("LANGSMITH_API_KEY"):
            raise ValueError("LangSmith API key not provided. Set LANGSMITH_API_KEY environment variable or pass api_key parameter.")
        
        self.client = Client()
        self.project_name = project_name
        self._setup_project()
        
        self.evaluation_criteria = {
            "investment_accuracy": {
                "description": "Evaluates factual accuracy of investment advice",
                "weight": 0.25
            },
            "risk_disclosure": {
                "description": "Checks if risks are properly disclosed",
                "weight": 0.20
            },
            "regulatory_compliance": {
                "description": "Verifies compliance with financial advisory standards",
                "weight": 0.15
            },
            "personalization": {
                "description": "Assesses how well advice is tailored to client profile",
                "weight": 0.15
            },
            "actionability": {
                "description": "Evaluates if advice is clear and actionable",
                "weight": 0.15
            },
            "clarity": {
                "description": "Measures clarity and understandability",
                "weight": 0.10
            }
        }
    
    def _setup_project(self):
        try:
            os.environ["LANGCHAIN_PROJECT"] = self.project_name
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
        except Exception as e:
            print(f"Warning: Could not set up LangSmith project: {e}")
    
    async def log_comparison_run(self, 
                                 model_name: str,
                                 prompt: str,
                                 response: str,
                                 test_case: Dict,
                                 evaluation_scores: Dict,
                                 run_metadata: Optional[Dict] = None) -> str:
        run_id = str(uuid.uuid4())
        
        try:
            metadata = {
                "model": model_name,
                "test_case_id": test_case.get("id", ""),
                "test_case_name": test_case.get("name", ""),
                "difficulty": test_case.get("difficulty", 1),
                "timestamp": datetime.now().isoformat(),
                **(run_metadata or {})
            }
            
            self.client.create_run(
                name=f"{model_name}-{test_case.get('id', 'unknown')}",
                run_type="llm",
                inputs={"prompt": prompt, "client_profile": test_case.get("variables", {})
                },
                outputs={"response": response},
                extra={
                    "metadata": metadata,
                    "evaluation_scores": evaluation_scores
                },
                project_name=self.project_name,
                id=run_id
            )
            
            return run_id
            
        except Exception as e:
            print(f"Error logging run to LangSmith: {e}")
            return None

# Enhanced Model Evaluator with Advanced Scoring
class InvestmentModelEvaluator:
    """Advanced evaluator for investment advisor responses with comprehensive metrics."""
    
    def __init__(self):
        self.evaluation_criteria = {
            'accuracy': 0.30,
            'completeness': 0.25,
            'helpfulness': 0.20,
            'clarity': 0.15,
            'relevance': 0.05,
            'professionalism': 0.05
        }
    
    async def evaluate_response(self, response: str, test_case: TestCase, model_name: str) -> Dict[str, Any]:
        """Comprehensive evaluation of model response."""
        # Convert TestCase to dict format for compatibility
        test_case_dict = {
            'expected_elements': test_case.expected_elements,
            'validation_criteria': test_case.validation_criteria,
            'difficulty': test_case.difficulty,
            'variables': test_case.variables
        }
        
        # Calculate individual criterion scores (0.0 - 1.0)
        criterion_scores = {}
        for criterion in self.evaluation_criteria.keys():
            score = await self._score_criterion(response, test_case_dict, criterion)
            criterion_scores[criterion] = score
        
        # Convert to 0-10 scale for display
        display_scores = {k: round(v * 10, 2) for k, v in criterion_scores.items()}
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(criterion_scores) * 10
        
        # Calculate additional metrics
        metrics = await self._calculate_metrics(response, test_case_dict)
        
        return {
            'scores': display_scores,
            'overall_score': round(overall_score, 2),
            'metrics': metrics,
            'response': response,
            'model': model_name,
            'test_case': test_case.name
        }
    
    async def _score_criterion(self, response: str, test_case: Dict, criterion: str) -> float:
        """Score the response against a specific criterion (0.0 - 1.0)."""
        scorers = {
            "accuracy": self._score_accuracy,
            "clarity": self._score_clarity,
            "completeness": self._score_completeness,
            "helpfulness": self._score_helpfulness,
            "relevance": self._score_relevance,
            "professionalism": self._score_professionalism
        }
        
        if criterion in scorers:
            return await scorers[criterion](response, test_case)
        else:
            return 0.5  # Default neutral score
    
    async def _score_accuracy(self, response: str, test_case: Dict) -> float:
        """Score the accuracy of the response."""
        score = 0.5  # Start with neutral
        
        # Check for factual elements
        expected_elements = test_case.get("expected_elements", [])
        if expected_elements:
            found_elements = sum(1 for elem in expected_elements if self._contains_element(response, elem))
            score = found_elements / len(expected_elements)
        
        # Check for validation criteria
        validation_criteria = test_case.get("validation_criteria", {})
        criteria_met = 0
        total_criteria = 0
        
        for criterion, expected in validation_criteria.items():
            total_criteria += 1
            if self._check_validation_criterion(response, criterion, expected):
                criteria_met += 1
        
        if total_criteria > 0:
            criteria_score = criteria_met / total_criteria
            score = (score + criteria_score) / 2  # Average the two scores
        
        return min(1.0, max(0.0, score))
    
    async def _score_clarity(self, response: str, test_case: Dict) -> float:
        """Score the clarity and readability of the response."""
        readability = self._calculate_readability(response)
        
        # Normalize readability score (0-100) to 0-1
        readability_score = readability / 100
        
        # Check for clear structure
        structure = self._analyze_structure(response)
        structure_score = 0.5
        
        if structure["has_introduction"]:
            structure_score += 0.1
        if structure["has_conclusion"]:
            structure_score += 0.1
        if structure["has_bullet_points"] or structure["has_numbered_list"]:
            structure_score += 0.1
        if structure["paragraph_balance"] == "balanced":
            structure_score += 0.1
        
        # Check for jargon vs. accessible language
        jargon_penalty = self._calculate_jargon_penalty(response)
        
        # Combine scores
        clarity_score = (readability_score * 0.5 + structure_score * 0.3 + (1 - jargon_penalty) * 0.2)
        
        return min(1.0, max(0.0, clarity_score))
    
    async def _score_completeness(self, response: str, test_case: Dict) -> float:
        """Score how complete the response is."""
        score = 0.5
        
        # Check if all expected elements are covered
        expected_elements = test_case.get("expected_elements", [])
        if expected_elements:
            covered_elements = sum(1 for elem in expected_elements if self._contains_element(response, elem))
            element_score = covered_elements / len(expected_elements)
            score = element_score
        
        # Check response length appropriateness
        word_count = len(response.split())
        difficulty = test_case.get("difficulty", 1)
        
        # Expected word count based on difficulty
        expected_min_words = 50 + (difficulty * 25)
        expected_max_words = 200 + (difficulty * 100)
        
        if word_count < expected_min_words:
            length_penalty = (expected_min_words - word_count) / expected_min_words * 0.3
            score -= length_penalty
        elif word_count > expected_max_words * 2:
            # Penalize extremely long responses
            length_penalty = 0.1
            score -= length_penalty
        
        return min(1.0, max(0.0, score))
    
    async def _score_helpfulness(self, response: str, test_case: Dict) -> float:
        """Score how helpful the response is."""
        # Check for actionable advice
        actionable_indicators = [
            'should', 'recommend', 'suggest', 'next step', 'consider', 'contact',
            'apply', 'prepare', 'gather', 'review', 'schedule', 'timeline'
        ]
        
        response_lower = response.lower()
        actionable_count = sum(1 for indicator in actionable_indicators if indicator in response_lower)
        actionable_score = min(1.0, actionable_count / 3)  # Cap at 3 indicators for full score
        
        # Check for specific details vs. generic advice
        specific_indicators = [
            r'\$[\\d,]+',  # Dollar amounts
            r'\\\d+\\.?\\d*\\s*%',  # Percentages
            r'\\\d+\\s+days?',  # Time periods
            r'\\\d+\\s+years?',  # Years
            'specific', 'exactly', 'precisely'
        ]
        
        specific_count = sum(1 for pattern in specific_indicators if re.search(pattern, response_lower))
        specific_score = min(1.0, specific_count / 3)
        
        # Check for addressing potential concerns
        concern_indicators = ['risk', 'consider', 'however', 'important', 'note', 'warning', 'careful']
        concern_count = sum(1 for indicator in concern_indicators if indicator in response_lower)
        concern_score = min(1.0, concern_count / 2)
        
        # Combine scores
        score = (actionable_score * 0.4 + specific_score * 0.4 + concern_score * 0.2)
        
        return min(1.0, max(0.0, score))
    
    async def _score_relevance(self, response: str, test_case: Dict) -> float:
        """Score how relevant the response is to the test case."""
        relevance_analysis = await self._analyze_topic_relevance(response, test_case)
        return relevance_analysis["score"]
    
    async def _score_professionalism(self, response: str, test_case: Dict) -> float:
        """Score the professionalism of the response."""
        score = 0.8  # Start high, deduct for issues
        
        response_lower = response.lower()
        
        # Check for unprofessional language
        unprofessional_terms = ['awesome', 'super', 'totally', 'amazing', 'absolutely']
        for term in unprofessional_terms:
            if term in response_lower:
                score -= 0.1
        
        # Check for appropriate tone
        professional_indicators = ['please', 'recommend', 'suggest', 'advise', 'assist']
        found_indicators = sum(1 for indicator in professional_indicators if indicator in response_lower)
        if found_indicators == 0:
            score -= 0.2
        
        return min(1.0, max(0.0, score))
    
    def _contains_element(self, response: str, element: str) -> bool:
        """Check if response contains the expected element."""
        element_keywords = element.lower().split()
        response_lower = response.lower()
        return any(keyword in response_lower for keyword in element_keywords)
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score (Flesch Reading Ease approximation)."""
        sentences = len(re.findall(r'[.!?]+', text))
        words = len(text.split())
        
        if sentences == 0 or words == 0:
            return 50.0  # Neutral score
        
        # Simplified readability calculation
        avg_sentence_length = words / sentences
        score = max(0, min(100, 100 - (avg_sentence_length - 15) * 2))
        return score
    
    def _analyze_structure(self, text: str) -> Dict[str, Any]:
        """Analyze the structure of the response."""
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        structure = {
            "has_introduction": len(paragraphs) > 0 and len(paragraphs[0]) > 50,
            "has_conclusion": len(paragraphs) > 1 and any(word in paragraphs[-1].lower() 
                                                        for word in ['summary', 'conclusion', 'overall', 'finally']),
            "has_bullet_points": '•' in text or re.search(r'^\s*[-*]\s', text, re.MULTILINE),
            "has_numbered_list": re.search(r'^\s*\\d+\.', text, re.MULTILINE),
            "paragraph_count": len(paragraphs),
            "paragraph_balance": "balanced" if 2 <= len(paragraphs) <= 5 else "unbalanced"
        }
        
        return structure
    
    async def _analyze_topic_relevance(self, response: str, test_case: Dict) -> Dict[str, Any]:
        """Analyze how relevant the response is to the investment topic."""
        # Investment-specific keywords by category
        investment_keywords = {
            'general': ['investment', 'portfolio', 'risk', 'return', 'market'],
            'instruments': ['stock', 'bond', 'etf', 'fund', 'crypto', 'reit'],
            'strategies': ['diversification', 'allocation', 'rebalancing', 'dollar-cost'],
            'metrics': ['yield', 'dividend', 'expense ratio', 'volatility']
        }
        
        response_lower = response.lower()
        relevance_score = 0.0
        category_scores = {}
        
        for category, keywords in investment_keywords.items():
            found_keywords = sum(1 for keyword in keywords if keyword in response_lower)
            category_score = min(1.0, found_keywords / len(keywords))
            category_scores[category] = category_score
            relevance_score += category_score
        
        # Normalize by number of categories
        relevance_score = relevance_score / len(investment_keywords)
        
        return {
            "score": relevance_score,
            "category_scores": category_scores
        }
    
    async def _calculate_metrics(self, response: str, test_case: Dict) -> Dict[str, Any]:
        """Calculate additional metrics for the response."""
        metrics = {
            "token_count": len(response.split()),
            "character_count": len(response),
            "readability_score": self._calculate_readability(response),
            "complexity_score": self._calculate_complexity_score(response, test_case),
            "coverage_score": self._calculate_coverage_score(response, test_case),
            "jargon_penalty": self._calculate_jargon_penalty(response)
        }
        
        return metrics
    
    def _calculate_complexity_score(self, response: str, test_case: Dict) -> float:
        """Calculate how well the response handles complexity."""
        difficulty = test_case.get("difficulty", 1)
        
        # Expected complexity indicators based on difficulty
        complexity_indicators = [
            r'however', r'although', r'consider', r'depending', r'various',
            r'multiple', r'complex', r'several', r'different', r'alternative'
        ]
        
        response_lower = response.lower()
        complexity_count = sum(1 for pattern in complexity_indicators if re.search(pattern, response_lower))
        
        # Score based on difficulty appropriateness
        expected_complexity = difficulty * 2
        if complexity_count >= expected_complexity:
            return 1.0
        else:
            return complexity_count / expected_complexity if expected_complexity > 0 else 0.5
    
    def _calculate_coverage_score(self, response: str, test_case: Dict) -> float:
        """Calculate how well the response covers the test case variables."""
        variables = test_case.get("variables", {})
        if not variables:
            return 1.0
        
        response_lower = response.lower()
        covered_variables = 0
        
        for key, value in variables.items():
            value_str = str(value).lower()
            # Check if the variable value or related terms appear in response
            if value_str in response_lower or any(word in response_lower for word in value_str.split()):
                covered_variables += 1
        
        return covered_variables / len(variables)
    
    def _check_validation_criterion(self, response: str, criterion: str, expected: Any) -> bool:
        """Check if a specific investment validation criterion is met."""
        response_lower = response.lower()
        
        # Boolean criteria
        if isinstance(expected, bool):
            criterion_patterns = {
                "mentions_diversification": r"diversif|spread.*risk|multiple.*asset",
                "explains_fees": r"expense\s+ratio|fund\s+fee|management\s+fee",
                "analyzes_risk_return": r"risk.*return|return.*risk|risk[-\s]?adjusted",
                "mentions_volatility": r"volatil|fluctuat|market.*swing",
                "discusses_security": r"wallet|private\s+key|secure.*crypto",
                "explains_regulations": r"regulat|sec|compliance|oversight",
                "mentions_bond_types": r"treasury|municipal|corporate\s+bond",
                "explains_yields": r"yield|coupon\s+rate|interest.*bond",
                "discusses_interest_risks": r"interest\s+rate\s+risk|duration\s+risk",
                "mentions_allocation": r"allocation|asset\s+mix|diversif.*portfolio",
                "explains_tax_strategies": r"tax\s+efficien|tax[-\s]?advant|capital\s+gain",
                "discusses_risks": r"risk|volatil|uncertain|downside"
            }
            
            if criterion in criterion_patterns:
                pattern = criterion_patterns[criterion]
                return bool(re.search(pattern, response_lower))
        
        # String criteria (check if string is mentioned)
        elif isinstance(expected, str):
            return expected.lower() in response_lower
        
        return False
    
    def _calculate_jargon_penalty(self, response: str) -> float:
        """Calculate penalty for excessive financial jargon use."""
        jargon_terms = [
            'alpha', 'beta', 'sharpe ratio', 'standard deviation', 'liquidity crunch',
            'derivative', 'hedging', 'leverage', 'arbitrage', 'market capitalization',
            'price-to-earnings', 'yield curve', 'basis points', 'quantitative easing'
        ]
        
        response_lower = response.lower()
        word_count = len(response.split())
        jargon_count = sum(1 for term in jargon_terms if term in response_lower)
        
        if word_count == 0:
            return 0.0
        
        jargon_ratio = jargon_count / word_count
        return min(0.5, jargon_ratio * 10)  # Cap at 50% penalty
    
    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate overall score from individual criterion scores."""
        if not scores:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for criterion, score in scores.items():
            weight = self.evaluation_criteria.get(criterion, 0.1)
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0

# Enhanced API Models
class ModelComparisonRequest(BaseModel):
    models: List[str]
    test_case_id: str
    api_key: str
    prompt_variation: int = 0

class ModelComparisonResult(BaseModel):
    model: str
    response: str
    scores: Dict[str, float]
    overall_score: float
    test_case_name: str
    timestamp: str
    metrics: Optional[Dict[str, Any]] = None

class ComparisonAnalysis(BaseModel):
    results: Dict[str, ModelComparisonResult]
    winner: str
    test_case: Dict[str, Any]
    summary_stats: Dict[str, float]

# Enhanced Model Manager
class EnhancedModelManager:
    """Enhanced model manager with investment focus."""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.available_models = {
            'gpt-4o-mini': 'gpt-4o-mini',
            'gpt-4.1-nano': 'gpt-4.1-nano'  # Updated model name
        }
        self.test_suite = InvestmentTestSuite()
        self.evaluator = InvestmentModelEvaluator()
    
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
                           prompt_variation: int = 0, langsmith_client: Optional[LangSmithIntegration] = None) -> ComparisonAnalysis:
        """Run comprehensive model comparison on investment test case."""
        
        # Get test case
        test_case = next((tc for tc in self.test_suite.test_cases if tc.id == test_case_id), None)
        if not test_case:
            raise ValueError(f"Test case {test_case_id} not found")
        
        # Get prompt variation
        if prompt_variation >= len(self.test_suite.prompt_variations):
            prompt_variation = 0
        
        prompt_template = self.test_suite.prompt_variations[prompt_variation]
        formatted_prompt = f"{prompt_template}\n\nClient situation: {test_case.variables}"
        
        # Get responses from all models
        results = {}
        tasks = [self.get_model_response(model, formatted_prompt) for model in models]
        responses = await asyncio.gather(*tasks)
        
        # Evaluate each response using the enhanced async evaluator
        for model, response in zip(models, responses):
            evaluation_result = await self.evaluator.evaluate_response(response, test_case, model)
            
            # Create ModelComparisonResult with correct structure
            results[model] = ModelComparisonResult(
                model=model,
                response=evaluation_result['response'],
                scores=evaluation_result['scores'],
                overall_score=evaluation_result['overall_score'],
                test_case_name=test_case.name,
                timestamp=datetime.now().isoformat(),
                metrics=evaluation_result.get('metrics')
            )

            if langsmith_client:
                await langsmith_client.log_comparison_run(
                    model_name=model,
                    prompt=formatted_prompt,
                    response=response,
                    test_case=test_case.__dict__,
                    evaluation_scores=evaluation_result['scores']
                )
        
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
            summary_stats=summary_stats
        )

# FastAPI Application
app = FastAPI(title="Investment Chatbot Comparison API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced API Models for dual chatbot results
class DualChatbotResult(BaseModel):
    session_id: str
    user_prompt: str
    responses: Dict[str, str]
    models: List[str]
    prompt_style: int
    timestamp: str

# Global storage for dual chatbot results
dual_chatbot_results = {}

@app.post("/store-dual-results")
async def store_dual_results(result: DualChatbotResult):
    """Store results from dual chatbot session."""
    try:
        dual_chatbot_results[result.session_id] = result.dict()
        return {"status": "success", "message": "Results stored successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/get-dual-results/{session_id}")
async def get_dual_results(session_id: str):
    """Get results from dual chatbot session."""
    if session_id in dual_chatbot_results:
        return dual_chatbot_results[session_id]
    else:
        raise HTTPException(status_code=404, detail="Session not found")

# Streamlit Application Functions
def display_metrics_dashboard():
    """Display the metrics dashboard for stored analysis results."""
    st.header("📊 Advanced Metrics Dashboard")
    
    if st.session_state.analysis_results is None:
        st.info("No analysis results available. Please run a comparison first in 'Standard Test Cases' or 'Real-Time Dual Chatbot' mode.")
        
        # Option to load sample data for demonstration
        if st.button("Load Sample Data for Demo"):
            st.session_state.analysis_results = create_sample_analysis_results()
            st.rerun()
    else:
        # Display tabs for different metric views
        tab1, tab2, tab3 = st.tabs(["Advanced Metrics", "Response Analysis", "Comparative Insights"])
        
        with tab1:
            display_advanced_metrics(st.session_state.analysis_results)
        
        with tab2:
            display_response_analysis(st.session_state.analysis_results)
        
        with tab3:
            display_comparative_insights(st.session_state.analysis_results)
        
        # Export functionality
        st.sidebar.subheader("Export Results")
        if st.sidebar.button("Export to CSV"):
            export_results_to_csv(st.session_state.analysis_results)
        
        if st.sidebar.button("Clear Results"):
            st.session_state.analysis_results = None
            st.rerun()

def create_sample_analysis_results():
    """Create sample analysis results for demonstration."""
    # This would create a sample ComparisonAnalysis object with dummy data
    # For brevity, returning None here - in practice, you'd create sample data
    return None

def export_results_to_csv(analysis_results):
    """Export analysis results to CSV."""
    import csv
    from io import StringIO
    
    # Create CSV data
    csv_buffer = StringIO()
    fieldnames = ['Model', 'Overall Score', 'Accuracy', 'Completeness', 'Helpfulness', 
                  'Clarity', 'Relevance', 'Professionalism']
    
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()
    
    for model, result in analysis_results.results.items():
        row = {'Model': model, 'Overall Score': result.overall_score}
        row.update({k.title(): v for k, v in result.scores.items()})
        writer.writerow(row)
    
    # Download button
    st.download_button(
        label="Download CSV",
        data=csv_buffer.getvalue(),
        file_name=f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def create_dual_chatbot_interface(api_key: str, langsmith_client: Optional[LangSmithIntegration] = None):
    """Create the dual chatbot comparison interface with automatic data collection."""
    
    st.header("Real-Time Dual Chatbot Comparison")
    st.markdown("Select a prompt style, launch dual chatbots, and automatically collect responses for analysis.")
    
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
        # Generate session ID
        session_id = str(uuid.uuid4())[:8]
        
        # Create HTML content for dual chatbots
        html_content = create_dual_chatbot_html(session_id, model1, model2, selected_prompt_style, api_key, test_suite.prompt_variations[selected_prompt_style])
        
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
        st.info("The chatbot interface opened in your browser. Send the same message to both models, then click 'Auto-Send Results to Streamlit' button.")
    
    # Auto-collection section
    st.subheader("4. Automatic Response Collection")
    
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
                        auto_analyze_collected_responses(result_data, langsmith_client)
                    else:
                        st.info("No results found via API. Checking browser storage...")
                        st.rerun()
                except:
                    st.info("API not available. Use 'Load from Browser' instead.")
        
        with col2:
            if st.button("Load from Browser Storage"):
                st.info("If automatic collection isn't working, use the 'Download Results File' button in the HTML interface, then upload the file below.")
        
        # File upload section
        st.subheader("Upload Results File")
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
                
                # Automatically analyze the uploaded data
                auto_analyze_collected_responses(result_data, langsmith_client)
                
            except json.JSONDecodeError:
                st.error("Invalid JSON file. Please ensure you uploaded the correct results file.")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # Manual input fallback
    st.subheader("5. Manual Input (Backup)")
    
    with st.expander("Manual Response Entry"):
        user_prompt_manual = st.text_input("User Question:", key="manual_prompt")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Model 1 Response:")
            response1_manual = st.text_area("Response 1:", height=200, key="manual_resp1")
        
        with col2:
            st.write("Model 2 Response:")
            response2_manual = st.text_area("Response 2:", height=200, key="manual_resp2")
        
        if st.button("Analyze Manual Responses"):
            if user_prompt_manual and response1_manual and response2_manual:
                manual_data = {
                    'user_prompt': user_prompt_manual,
                    'responses': {
                        st.session_state.current_session['models'][0]: response1_manual,
                        st.session_state.current_session['models'][1]: response2_manual
                    },
                    'models': st.session_state.current_session['models'],
                    'prompt_style': st.session_state.current_session['prompt_style']
                }
                auto_analyze_collected_responses(manual_data, langsmith_client)
            else:
                st.error("Please fill in all fields.")

def auto_analyze_collected_responses(result_data, langsmith_client: Optional[LangSmithIntegration] = None):
    """Automatically analyze collected responses from dual chatbot and store in session state."""
    
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

            if langsmith_client:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(langsmith_client.log_comparison_run(
                    model_name=model,
                    prompt=result_data['user_prompt'],
                    response=response,
                    test_case=user_test_case.__dict__,
                    evaluation_scores=result['scores']
                ))
                loop.close()
        
        # Create ComparisonAnalysis object for metrics dashboard
        results_dict = {}
        for model, eval_result in evaluation_results.items():
            results_dict[model] = ModelComparisonResult(
                model=model,
                response=eval_result['response'],
                scores=eval_result['scores'],
                overall_score=eval_result['overall_score'],
                test_case_name=user_test_case.name,
                timestamp=datetime.now().isoformat(),
                metrics=eval_result.get('metrics')
            )
        
        winner = max(evaluation_results.items(), key=lambda x: x[1]['overall_score'])[0]
        all_scores = [r['overall_score'] for r in evaluation_results.values()]
        
        analysis = ComparisonAnalysis(
            results=results_dict,
            winner=winner,
            test_case={
                'id': user_test_case.id,
                'name': user_test_case.name,
                'type': user_test_case.type.value,
                'difficulty': user_test_case.difficulty,
                'tags': user_test_case.tags
            },
            summary_stats={
                'mean_score': round(statistics.mean(all_scores), 2),
                'score_std': round(statistics.stdev(all_scores) if len(all_scores) > 1 else 0, 2),
                'score_range': round(max(all_scores) - min(all_scores), 2),
                'winner_advantage': round(evaluation_results[winner]['overall_score'] - min(all_scores), 2)
            }
        )
        
        # Store in session state for metrics dashboard
        st.session_state.analysis_results = analysis
        
        # Display results
        st.subheader("Evaluation Results")
        
        # Winner announcement
        st.success(f"Winner: {winner} with score {evaluation_results[winner]['overall_score']:.1f}/10")
        
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
        
        st.success("Automatic analysis completed!")
        st.info("💡 Go to 'Metrics Dashboard' to see advanced analytics")
        
    except Exception as e:
        st.error(f"Error during automatic analysis: {str(e)}")
        st.info("You can still view the collected responses above.")

def create_dual_chatbot_html(session_id: str, model1: str, model2: str, prompt_style: int, api_key: str, system_prompt: str) -> str:
    """Create HTML page with dual chatbots."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dual Investment Advisor Comparison</title>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/default.min.css">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f0f2f5;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .container {{
            width: 100%;
            max-width: 1200px;
            display: flex;
            gap: 20px;
        }}
        .chat-container {{
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            width: 50%;
        }}
        .chat-header {{
            padding: 16px;
            border-bottom: 1px solid #e0e0e0;
            font-size: 18px;
            font-weight: 600;
            color: #333;
        }}
        .chat-box {{
            flex-grow: 1;
            padding: 16px;
            overflow-y: auto;
            height: 400px;
        }}
        .message {{
            margin-bottom: 12px;
            padding: 8px 12px;
            border-radius: 18px;
            max-width: 80%;
        }}
        .user-message {{
            background-color: #007bff;
            color: white;
            align-self: flex-end;
            margin-left: auto;
        }}
        .bot-message {{
            background-color: #e9e9eb;
            color: #333;
            align-self: flex-start;
        }}
        .input-area {{
            display: flex;
            padding: 16px;
            border-top: 1px solid #e0e0e0;
        }}
        input[type="text"] {{
            flex-grow: 1;
            border: 1px solid #ccc;
            border-radius: 18px;
            padding: 10px 16px;
            font-size: 16px;
            margin-right: 10px;
        }}
        button {{
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 18px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s;
        }}
        button:hover {{
            background-color: #0056b3;
        }}
        .controls {{
            margin-top: 20px;
            display: flex;
            gap: 10px;
        }}
        .main-input-area {{
            width: 100%;
            max-width: 1200px;
            margin-top: 20px;
            display: flex;
            gap: 10px;
        }}
    </style>
</head>
<body>
    <h1>Dual Investment Advisor Comparison</h1>
    <div class="container">
        <div class="chat-container">
            <div class="chat-header" id="model1-name">{model1}</div>
            <div class="chat-box" id="chat-box-1"></div>
        </div>
        <div class="chat-container">
            <div class="chat-header" id="model2-name">{model2}</div>
            <div class="chat-box" id="chat-box-2"></div>
        </div>
    </div>
    <div class="main-input-area">
        <input type="text" id="main-input" placeholder="Ask both models...">
        <button onclick="sendToBoth()">Send</button>
    </div>
    <div class="controls">
        <button onclick="autoSendResults()">Auto-Send Results to Streamlit</button>
        <button onclick="downloadResults()">Download Results File</button>
    </div>

    <script>
        const session_id = "{session_id}";
        const models = ["{models[0]}", "{models[1]}"];
        const prompt_style = {prompt_style};
        const api_key = "{api_key}";
        const system_prompt = `{system_prompt}`;

        let user_prompt = "";
        let responses = {{}};

        async function sendToBoth() {{
            const input = document.getElementById('main-input');
            const messageText = input.value;

            if (!messageText) return;

            user_prompt = messageText;

            // Display user message in both chat boxes
            for (let i = 1; i <= 2; i++) {{
                const chatBox = document.getElementById(`chat-box-${{i}}`);
                const userMessageDiv = document.createElement('div');
                userMessageDiv.className = 'message user-message';
                userMessageDiv.innerText = messageText;
                chatBox.appendChild(userMessageDiv);
                chatBox.scrollTop = chatBox.scrollHeight;
            }}

            input.value = '';

            // Get responses from both models
            for (let i = 0; i < 2; i++) {{
                const modelName = models[i];
                const chatBox = document.getElementById(`chat-box-${{i+1}}`);
                const botMessageDiv = document.createElement('div');
                botMessageDiv.className = 'message bot-message';
                botMessageDiv.innerHTML = 'Thinking...';
                chatBox.appendChild(botMessageDiv);

                const response = await getModelResponse(modelName, messageText);
                
                botMessageDiv.innerHTML = marked.parse(response);
                hljs.highlightAll();
                
                responses[modelName] = response;
                chatBox.scrollTop = chatBox.scrollHeight;
            }}
        }}

        async function getModelResponse(model, prompt) {{
            const full_prompt = `${{system_prompt}}\n\nUser question: ${{prompt}}`;
            const response = await fetch("https://api.openai.com/v1/chat/completions", {{
                method: "POST",
                headers: {{
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${{api_key}}`
                }},
                body: JSON.stringify({{
                    model: model,
                    messages: [{{ "role": "user", "content": full_prompt }}],
                    max_tokens: 500,
                    temperature: 0.7
                }})
            }});
            const data = await response.json();
            return data.choices[0].message.content;
        }}

        function autoSendResults() {{
            const result_data = {{
                session_id,
                user_prompt,
                responses,
                models,
                prompt_style,
                timestamp: new Date().toISOString()
            }};

            fetch("http://localhost:8000/store-dual-results", {{
                method: "POST",
                headers: {{ "Content-Type": "application/json" }},
                body: JSON.stringify(result_data)
            }})
            .then(response => response.json())
            .then(data => {{
                if(data.status === 'success') {{
                    alert("Results sent to Streamlit successfully!");
                }} else {{
                    alert("Failed to send results.");
                }}
            }})
            .catch(err => {{
                console.error("Error sending results:", err);
                alert("Error sending results. Please check the console.");
            }});
        }}

        function downloadResults() {{
            const result_data = {{
                session_id,
                user_prompt,
                responses,
                models,
                prompt_style,
                timestamp: new Date().toISOString()
            }};
            const blob = new Blob([JSON.stringify(result_data, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `session_${{session_id}}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }}
    </script>
</body>
</html>"""

def display_comparison_results(analysis):
    """Display basic comparison results."""
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

def create_standard_comparison_interface(api_key: str, langsmith_client: Optional[LangSmithIntegration] = None):
    """Create the standard test case comparison interface with metrics integration."""
    
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
    col1, col2 = st.columns(2)
    
    with col1:
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
                        model_manager.run_comparison(selected_models, selected_test_id, prompt_variation, langsmith_client)
                    )
                    loop.close()
                    
                    # Store results in session state for metrics dashboard
                    st.session_state.analysis_results = analysis
                    
                    # Display results
                    display_comparison_results(analysis)
                    
                    st.success("Model comparison completed successfully!")
                    st.info("💡 Go to 'Metrics Dashboard' in the sidebar to see advanced analytics")
                    
                except Exception as e:
                    st.error(f"Error running comparison: {str(e)}")
    
    with col2:
        if st.button("View Metrics Dashboard", type="secondary"):
            if st.session_state.analysis_results:
                st.sidebar.selectbox("Select Mode:", ["Metrics Dashboard"])
                st.rerun()
            else:
                st.warning("Run a comparison first to generate metrics")

def integrate_langsmith_with_streamlit(st_component):
    """Add LangSmith integration controls to Streamlit sidebar."""
    st_component.subheader("LangSmith Integration")
    
    # LangSmith API key input
    langsmith_api_key = st_component.text_input(
        "LangSmith API Key",
        type="password",
        help="Enter your LangSmith API key for advanced evaluation tracking"
    )
    
    # Enable/disable LangSmith
    use_langsmith = st_component.checkbox(
        "Enable LangSmith Tracking",
        value=False,
        help="Send evaluation data to LangSmith for organizing evaluations"
    )
    
    if use_langsmith and langsmith_api_key:
        # Project name configuration
        project_name = st_component.text_input(
            "LangSmith Project Name",
            value="investment-chatbot-comparison",
            help="Name of the LangSmith project for organizing evaluations"
        )
        
        # Initialize LangSmith integration
        try:
            langsmith = LangSmithIntegration(
                api_key=langsmith_api_key,
                project_name=project_name
            )
            st_component.success("✅ LangSmith connected")
            return langsmith
        except Exception as e:
            st.error(f"❌ LangSmith connection failed: {e}")
            return None
    
    return None

def create_streamlit_app():
    """Create enhanced Streamlit application with metrics dashboard."""
    
    st.set_page_config(
        page_title="Investment Chatbot Model Comparison",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("💰 Investment Chatbot Model Comparison")
    st.markdown("Compare AI models on investment advisory scenarios with advanced metrics")
    
    # Sidebar navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Select Mode:",
        ["Standard Test Cases", "Real-Time Dual Chatbot", "Metrics Dashboard"]
    )
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    
    langsmith_client = integrate_langsmith_with_streamlit(st.sidebar)

    if not api_key and page != "Metrics Dashboard":
        st.warning("Enter your OpenAI API key to begin")
        return
    
    # Initialize session state for results storage
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    if page == "Metrics Dashboard":
        display_metrics_dashboard()
    elif page == "Real-Time Dual Chatbot":
        create_dual_chatbot_interface(api_key, langsmith_client)
    else:
        create_standard_comparison_interface(api_key, langsmith_client)

# Main execution function
def main():
    """Main function to run the integrated system."""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "streamlit":
            create_streamlit_app()
        elif sys.argv[1] == "api":
            import uvicorn
            uvicorn.run(app, host="0.0.0.0", port=8000)
        else:
            print("Usage: python integrated_investment_system.py [streamlit|api]")
    else:
        print("Investment Chatbot Comparison System")
        print("Usage:")
        print("  streamlit run integrated_investment_system.py streamlit")
        print("  python integrated_investment_system.py api")

if __name__ == "__main__":
    main()