# src/core/test_suite.py
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any

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
