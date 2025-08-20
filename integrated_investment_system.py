# integrated_investment_system.py
import asyncio
import json
import re
import time
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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

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
        
        # Pattern-based matching for common investment concepts
        self.patterns = {
            "diversification": r"diversif(y|ication)|spread.*risk|multiple.*asset",
            "expense_ratio": r"expense\s+ratio|management\s+fee|fund\s+fee",
            "risk_vs_return": r"risk.*return|return.*risk|risk[-\s]?adjusted",
            "volatility": r"volatil|fluctuat|price.*swing",
            "allocation": r"allocat.*portfolio|asset.*mix|weighting",
            "bond_yield": r"bond.*yield|coupon.*rate|fixed\s+income",
            "crypto_security": r"wallet|private\s+key|cold\s+storage|exchange\s+risk",
            "tax_implications": r"tax|capital\s+gain|tax[-\s]?advantaged",
            "cash_flow": r"cash\s+flow|income\s+stream|dividend|interest.*payment",
            "withdrawal_rate": r"withdrawal.*rate|safe.*withdrawal|4%\s+rule"
        }

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
            """You are a professional investment advisor. A client has approached you with the following information:

Client Income: {client_income}
Investment Goal: {investment_goal}
Risk Tolerance: {risk_tolerance}
Time Horizon: {time_horizon}
Available Investment: {investment_amount}
Preferred Assets: {preferred_assets}

Please provide comprehensive investment advice including allocation strategies, risk management, and next steps.""",

            """As an experienced financial consultant, I want to help you navigate your investment journey. Based on your profile:

• Annual Income: {client_income}
• Investment Goal: {investment_goal}
• Risk Tolerance: {risk_tolerance}
• Time Horizon: {time_horizon}
• Available Capital: {investment_amount}
• Preferred Assets: {preferred_assets}

Let me provide you with a detailed analysis of your options, probability of success, and strategic recommendations.""",

            """Hi there! I'm excited to help you with your investments. Let's review your situation:

Income: {client_income}
Goal: {investment_goal}
Risk profile: {risk_tolerance}
Timeline: {time_horizon}
Investment amount: {investment_amount}
Interested in: {preferred_assets}

I'll walk you through potential strategies, explain how different asset classes fit, and outline what steps we should take to move forward."""
        ]

    def check_validation_criterion(self, response: str, criterion: str, expected: Any) -> bool:
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
                "addresses_income_variability": r"irregular\s+income|variable\s+income|unstable\s+earnings",
                "mentions_emergency_fund": r"emergency\s+fund|cash\s+reserve|rainy\s+day",
                "discusses_rebalancing": r"rebalance|adjust.*portfolio|realign.*assets",
                "mentions_allocation": r"allocation|asset\s+mix|diversif.*portfolio",
                "explains_tax_strategies": r"tax\s+efficien|tax[-\s]?advant|capital\s+gain",
                "discusses_risks": r"risk|volatil|uncertain|downside",
                "mentions_savings_rate": r"saving\s+rate|save\s+%|high\s+savings",
                "explains_withdrawal_strategies": r"withdrawal\s+rate|safe\s+withdrawal|4%\s+rule",
                "discusses_tradeoffs": r"trade[-\s]?off|sacrifice|balanc.*goal"
            }
            
            if criterion in criterion_patterns:
                pattern = criterion_patterns[criterion]
                return bool(re.search(pattern, response_lower))
        
        # String criteria (check if string is mentioned)
        elif isinstance(expected, str):
            return expected.lower() in response_lower
            
        return False

    def calculate_jargon_penalty(self, response: str) -> float:
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

# Enhanced Model Evaluator with Advanced Scoring
class InvestmentModelEvaluator:
    """Advanced evaluator for investment advisor responses with comprehensive metrics."""
    
    def __init__(self):
        self.test_suite = InvestmentTestSuite()
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
        score = 0.5
        
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
            r'\$[\d,]+',  # Dollar amounts
            r'\d+\.?\d*\s*%',  # Percentages
            r'\d+\s+days?',  # Time periods
            r'\d+\s+years?',  # Years
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
        
        # Check for proper grammar (simple heuristics)
        grammar_issues = 0
        
        # Check for consistent capitalization
        sentences = re.split(r'[.!?]+', response)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and not sentence[0].isupper():
                grammar_issues += 1
        
        # Penalize excessive grammar issues
        if grammar_issues > len(sentences) * 0.1:
            score -= 0.1
        
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
        syllables = sum(self._count_syllables(word) for word in text.split())
        
        if sentences == 0 or words == 0:
            return 50.0  # Neutral score
        
        # Simplified Flesch Reading Ease
        avg_sentence_length = words / sentences
        avg_syllables_per_word = syllables / words
        
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        return max(0, min(100, score))
    
    def _count_syllables(self, word: str) -> int:
        """Estimate syllables in a word."""
        word = word.lower().strip()
        if not word:
            return 0
        
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent e
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _analyze_structure(self, text: str) -> Dict[str, Any]:
        """Analyze the structure of the response."""
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        sentences = re.split(r'[.!?]+', text)
        
        structure = {
            "has_introduction": len(paragraphs) > 0 and len(paragraphs[0]) > 50,
            "has_conclusion": len(paragraphs) > 1 and any(word in paragraphs[-1].lower() 
                                                        for word in ['summary', 'conclusion', 'overall', 'finally']),
            "has_bullet_points": '•' in text or re.search(r'^\s*[-*]\s', text, re.MULTILINE),
            "has_numbered_list": re.search(r'^\s*\d+\.', text, re.MULTILINE),
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
                "addresses_income_variability": r"irregular\s+income|variable\s+income|unstable\s+earnings",
                "mentions_emergency_fund": r"emergency\s+fund|cash\s+reserve|rainy\s+day",
                "discusses_rebalancing": r"rebalance|adjust.*portfolio|realign.*assets",
                "mentions_allocation": r"allocation|asset\s+mix|diversif.*portfolio",
                "explains_tax_strategies": r"tax\s+efficien|tax[-\s]?advant|capital\s+gain",
                "discusses_risks": r"risk|volatil|uncertain|downside",
                "mentions_savings_rate": r"saving\s+rate|save\s+%|high\s+savings",
                "explains_withdrawal_strategies": r"withdrawal\s+rate|safe\s+withdrawal|4%\s+rule",
                "discusses_tradeoffs": r"trade[-\s]?off|sacrifice|balanc.*goal"
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
        
        # Weight different criteria
        weights = {
            "accuracy": 0.3,
            "completeness": 0.25,
            "helpfulness": 0.2,
            "clarity": 0.15,
            "relevance": 0.05,
            "professionalism": 0.05
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for criterion, score in scores.items():
            weight = weights.get(criterion, 0.1)  # Default weight for unknown criteria
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent e
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _analyze_structure(self, text: str) -> Dict[str, Any]:
        """Analyze the structure of the response."""
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        sentences = re.split(r'[.!?]+', text)
        
        structure = {
            "has_introduction": len(paragraphs) > 0 and len(paragraphs[0]) > 50,
            "has_conclusion": len(paragraphs) > 1 and any(word in paragraphs[-1].lower() 
                                                        for word in ['summary', 'conclusion', 'overall', 'finally']),
            "has_bullet_points": '•' in text or re.search(r'^\s*[-*]\s', text, re.MULTILINE),
            "has_numbered_list": re.search(r'^\s*\d+\.', text, re.MULTILINE),
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
                "addresses_income_variability": r"irregular\s+income|variable\s+income|unstable\s+earnings",
                "mentions_emergency_fund": r"emergency\s+fund|cash\s+reserve|rainy\s+day",
                "discusses_rebalancing": r"rebalance|adjust.*portfolio|realign.*assets",
                "mentions_allocation": r"allocation|asset\s+mix|diversif.*portfolio",
                "explains_tax_strategies": r"tax\s+efficien|tax[-\s]?advant|capital\s+gain",
                "discusses_risks": r"risk|volatil|uncertain|downside",
                "mentions_savings_rate": r"saving\s+rate|save\s+%|high\s+savings",
                "explains_withdrawal_strategies": r"withdrawal\s+rate|safe\s+withdrawal|4%\s+rule",
                "discusses_tradeoffs": r"trade[-\s]?off|sacrifice|balanc.*goal"
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
        
        # Weight different criteria
        weights = {
            "accuracy": 0.3,
            "completeness": 0.25,
            "helpfulness": 0.2,
            "clarity": 0.15,
            "relevance": 0.05,
            "professionalism": 0.05
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for criterion, score in scores.items():
            weight = weights.get(criterion, 0.1)  # Default weight for unknown criteria
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
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
    
    def calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted overall score."""
        overall = sum(scores[criterion] * weight 
                     for criterion, weight in self.evaluation_criteria.items()
                     if criterion in scores)
        return round(overall, 2)

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
            'gpt-4.1-nano': 'gpt-4.1-nano'  # Updated to use GPT-4 nano instead of 3.5-turbo
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
                           prompt_variation: int = 0) -> ComparisonAnalysis:
        """Run comprehensive model comparison on investment test case."""
        
        # Get test case
        test_case = next((tc for tc in self.test_suite.test_cases if tc.id == test_case_id), None)
        if not test_case:
            raise ValueError(f"Test case {test_case_id} not found")
        
        # Get prompt variation
        if prompt_variation >= len(self.test_suite.prompt_variations):
            prompt_variation = 0
        
        prompt_template = self.test_suite.prompt_variations[prompt_variation]
        formatted_prompt = prompt_template.format(**test_case.variables)
        
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
                timestamp=datetime.now().isoformat()
            )
            
            # Add metrics if available
            if 'metrics' in evaluation_result:
                results[model].metrics = evaluation_result['metrics']
        
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

# Serve the HTML file
@app.get("/", response_class=HTMLResponse)
async def serve_chatbot():
    """Serve the investment chatbot HTML."""
    with open("investmentChatBot.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/test-cases")
async def get_test_cases():
    """Get available test cases."""
    test_suite = InvestmentTestSuite()
    return {
        "test_cases": [
            {
                "id": tc.id,
                "name": tc.name,
                "type": tc.type.value,
                "difficulty": tc.difficulty,
                "tags": tc.tags,
                "variables": tc.variables
            }
            for tc in test_suite.test_cases
        ]
    }

@app.post("/compare-models")
async def compare_models(request: ModelComparisonRequest):
    """Compare models on investment test case."""
    try:
        model_manager = EnhancedModelManager(request.api_key)
        analysis = await model_manager.run_comparison(
            request.models, 
            request.test_case_id, 
            request.prompt_variation
        )
        return analysis
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Streamlit Application
def create_streamlit_app():
    """Create enhanced Streamlit application."""
    
    st.set_page_config(
        page_title="Investment Chatbot Model Comparison",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("💰 Investment Chatbot Model Comparison")
    st.markdown("Compare AI models on investment advisory scenarios")
    
    # Sidebar
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    
    if not api_key:
        st.warning("Enter your OpenAI API key to begin")
        return
    
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
        st.subheader(f"📋 {selected_test_case.name}")
        st.write(f"**Type:** {selected_test_case.type.value}")
        st.write(f"**Difficulty:** {selected_test_case.difficulty}/5")
        st.write(f"**Tags:** {', '.join(selected_test_case.tags)}")
    
    with col2:
        st.subheader("Client Profile")
        for key, value in selected_test_case.variables.items():
            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    
    # Run comparison
    if st.button("🚀 Run Model Comparison", type="primary"):
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
                
                # Display results with advanced metrics
                st.header("Comparison Results")
                
                # Winner announcement
                winner_result = analysis.results[analysis.winner]
                st.success(f"**Winner: {analysis.winner}** with score {winner_result.overall_score}/10")
                
                # Try to import and use advanced metrics, fallback to basic if not available
                try:
                    # Check if metrics_dashboard is available
                    import importlib.util
                    spec = importlib.util.find_spec("metrics_dashboard")
                    
                    if spec is not None:
                        from metrics_dashboard import display_advanced_metrics, display_response_analysis, display_comparative_insights
                        
                        # Display advanced metrics dashboard
                        display_advanced_metrics(analysis)
                        
                        # Response analysis
                        display_response_analysis(analysis)
                        
                        # Comparative insights
                        display_comparative_insights(analysis)
                    else:
                        st.info("Advanced metrics dashboard not available. Using basic visualization.")
                        raise ImportError("metrics_dashboard not found")
                        
                except ImportError:
                    # Fallback to basic metrics display
                    st.subheader("Basic Results Display")
                    
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
                
                st.success("✅ Model comparison completed successfully!")
                
                # Detailed results
                st.subheader("Detailed Scores")
                results_data = []
                for model, result in analysis.results.items():
                    row = {'Model': model, 'Overall Score': result.overall_score}
                    row.update(result.scores)
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
                
            except Exception as e:
                st.error(f"Error running comparison: {str(e)}")

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