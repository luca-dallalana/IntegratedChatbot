# src/core/model_manager.py
import asyncio
import statistics
from datetime import datetime
from typing import Dict, List, Any, Optional

import openai

from .test_suite import InvestmentTestSuite, TestCase
from .evaluators import InvestmentModelEvaluator

# Pydantic models for API data validation (can be moved to a separate models file if they grow)
from pydantic import BaseModel

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

class EnhancedModelManager:
    """Enhanced model manager with investment focus."""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.available_models = {
            'gpt-4o-mini': 'gpt-4o-mini',
            'gpt-4.1-nano': 'gpt-4.1-nano'
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
