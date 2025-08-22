# src/core/evaluators.py
import re
from typing import Dict, Any

from .test_suite import TestCase

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
            r'\\d+\\.?\\d*\\s*%',  # Percentages
            r'\\d+\\s+days?',  # Time periods
            r'\\d+\\s+years?',  # Years
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
