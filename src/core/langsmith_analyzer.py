# src/core/langsmith_analyzer.py
import streamlit as st
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional

# LangSmith imports
try:
    from langsmith import Client
    from langsmith.schemas import Run
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

class LangSmithAnalyzer:
    """LangSmith-powered analyzer for investment chatbot responses."""
    
    def __init__(self, langsmith_api_key: str, project_name: str):
        if not LANGSMITH_AVAILABLE:
            raise ImportError("LangSmith not available. Install with: pip install langsmith langchain langchain-openai")
        
        self.client = Client(api_key=langsmith_api_key)
        self.project_name = project_name
        
        try:
            # Check if project exists
            try:
                self.client.read_project(project_name=project_name)
            except Exception: # LangSmith client raises a generic Exception
                # If not, create it
                self.client.create_project(project_name=project_name)
                st.toast(f"Created new LangSmith project: '{project_name}'")
        except Exception as e:
            # Re-raise with a more informative message
            raise ConnectionError(f"Failed to connect or create LangSmith project '{project_name}'. Please check your API key and network connection. Details: {e}")
    
    def get_last_trace(self) -> Optional[Run]:
        """Fetches the most recent run from the LangSmith project.""" 
        try:
            runs = list(self.client.list_runs(project_name=self.project_name, limit=1))
            if runs:
                return runs[0]
            return None
        except Exception as e:
            st.error(f"Failed to fetch last trace from LangSmith: {str(e)}")
            return None

    def test_connection(self) -> bool:
        """Performs a simple write test to the LangSmith project."""
        try:
            test_run = self.client.create_run(
                name="LangSmith Connection Test",
                run_type="tool",
                project_name=self.project_name,
                inputs={"status": "testing..."},
                outputs={"status": "ok"}
            )
            
            if test_run is None:
                raise ConnectionError("Run creation returned None. This often indicates an authentication issue. Please verify your LangSmith API key.")

            # Verify the run was created
            self.client.read_run(test_run.id)
            st.success(f"Connection test successful! A test run was created in project '{self.project_name}'.")
            st.info(f"View the test run here: {test_run.url}")
            return True
        except Exception as e:
            st.error(f"LangSmith connection test failed: {e}")
            return False

    def evaluate_responses(self, user_prompt: str, responses: Dict[str, str], models: List[str]) -> Dict[str, Any]:
        """Evaluate responses using LangSmith's evaluation framework."""
        
        # Define custom evaluators
        def investment_quality_evaluator(run, example):
            """Evaluate investment advice quality."""
            response = run.outputs.get("response", "")
            
            # Investment-specific criteria
            criteria_scores = {}
            
            # Risk discussion (0-1)
            risk_keywords = ["risk", "volatility", "loss", "fluctuation", "uncertainty"]
            risk_score = min(1.0, sum(1 for keyword in risk_keywords if keyword.lower() in response.lower()) / 3)
            criteria_scores["risk_discussion"] = risk_score
            
            # Actionable advice (0-1)
            action_keywords = ["recommend", "suggest", "consider", "should", "allocate", "invest"]
            action_score = min(1.0, sum(1 for keyword in action_keywords if keyword.lower() in response.lower()) / 3)
            criteria_scores["actionable_advice"] = action_score
            
            # Specific details (0-1)
            specific_patterns = [r'\$[\\d,]+', r'\d+\.?\d*\s*%', r'\d+\s+years?']
            specific_score = min(1.0, sum(1 for pattern in specific_patterns if __import__('re').search(pattern, response)) / 2)
            criteria_scores["specificity"] = specific_score
            
            # Professional tone (0-1)
            professional_keywords = ["analysis", "strategy", "portfolio", "diversification", "allocation"]
            professional_score = min(1.0, sum(1 for keyword in professional_keywords if keyword.lower() in response.lower()) / 3)
            criteria_scores["professionalism"] = professional_score
            
            # Overall score
            overall_score = sum(criteria_scores.values()) / len(criteria_scores)
            
            return {
                "key": "investment_quality",
                "score": overall_score,
                "details": criteria_scores
            }
        
        def response_completeness_evaluator(run, example):
            """Evaluate response completeness."""
            response = run.outputs.get("response", "")
            word_count = len(response.split())
            
            # Completeness based on word count and structure
            if word_count < 50:
                completeness_score = 0.3
            elif word_count < 100:
                completeness_score = 0.6
            elif word_count < 200:
                completeness_score = 0.9
            else:
                completeness_score = 1.0
            
            # Check for structure (paragraphs, bullet points)
            structure_bonus = 0
            if '\n' in response:
                structure_bonus += 0.1
            if any(marker in response for marker in ['•', '-', '1.', '2.']):
                structure_bonus += 0.1
            
            final_score = min(1.0, completeness_score + structure_bonus)
            
            return {
                "key": "completeness",
                "score": final_score,
                "details": {"word_count": word_count, "structure_bonus": structure_bonus}
            }
        
        # Run evaluations
        evaluation_results = {}
        
        for model, response in responses.items():
            # Mock run object for evaluation
            class MockRun:
                def __init__(self, response):
                    self.outputs = {"response": response}
            
            mock_run = MockRun(response)
            
            # Run evaluators
            quality_result = investment_quality_evaluator(mock_run, None)
            completeness_result = response_completeness_evaluator(mock_run, None)
            
            evaluation_results[model] = {
                "quality_score": quality_result["score"],
                "quality_details": quality_result["details"],
                "completeness_score": completeness_result["score"],
                "completeness_details": completeness_result["details"],
                "overall_score": (quality_result["score"] + completeness_result["score"]) / 2
            }
        
        return evaluation_results
    
    def generate_comparison_charts(self, evaluation_results: Dict[str, Any], models: List[str]) -> Dict[str, go.Figure]:
        """Generate enhanced comparison charts using evaluation results."""
        charts = {}
        
        # Overall scores comparison
        overall_scores = [evaluation_results[model]["overall_score"] for model in models]
        
        fig_overall = go.Figure(data=[
            go.Bar(
                x=models,
                y=overall_scores,
                marker_color=['#1f77b4', '#ff7f0e'],
                text=[f"{score:.2f}" for score in overall_scores],
                textposition='auto'
            )
        ])
        fig_overall.update_layout(
            title="LangSmith Overall Performance Comparison",
            yaxis_title="Score (0-1)",
            xaxis_title="Model",
            yaxis_range=[0, 1]
        )
        charts["overall_comparison"] = fig_overall
        
        # Detailed criteria radar chart
        criteria = ["quality_score", "completeness_score"]
        
        fig_radar = go.Figure()
        for model in models:
            scores = [evaluation_results[model][criterion] for criterion in criteria]
            fig_radar.add_trace(go.Scatterpolar(
                r=scores,
                theta=[criterion.replace("_", " ").title() for criterion in criteria],
                fill='toself',
                name=model,
                opacity=0.7
            ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="LangSmith Detailed Criteria Comparison"
        )
        charts["radar_comparison"] = fig_radar
        
        # Quality breakdown
        quality_details = {}
        for model in models:
            quality_details[model] = evaluation_results[model]["quality_details"]
        
        quality_criteria = list(quality_details[models[0]].keys())
        fig_quality = go.Figure()
        
        for i, model in enumerate(models):
            scores = [quality_details[model][criterion] for criterion in quality_criteria]
            fig_quality.add_trace(go.Bar(
                name=model,
                x=[criterion.replace("_", " ").title() for criterion in quality_criteria],
                y=scores,
                opacity=0.8
            ))
        
        fig_quality.update_layout(
            title="LangSmith Investment Quality Breakdown",
            yaxis_title="Score (0-1)",
            xaxis_title="Quality Criteria",
            barmode='group',
            yaxis_range=[0, 1]
        )
        charts["quality_breakdown"] = fig_quality
        
        return charts
    
    def get_project_analytics(self) -> Dict[str, Any]:
        """Get analytics from the LangSmith project."""
        try:
            # Get recent runs from the project
            runs = list(self.client.list_runs(project_name=self.project_name, limit=50))
            
            if not runs:
                return {"message": "No runs found in project"}
            
            # Analyze runs
            analytics = {
                "total_runs": len(runs),
                "models_used": {},
                "avg_response_length": 0,
                "recent_activity": []
            }
            
            total_length = 0
            for run in runs:
                # Count models
                model = run.extra.get("model", "unknown") if run.extra else "unknown"
                analytics["models_used"][model] = analytics["models_used"].get(model, 0) + 1
                
                # Calculate average response length
                response = run.outputs.get("response", "") if run.outputs else ""
                total_length += len(response.split())
                
                # Recent activity
                analytics["recent_activity"].append({
                    "timestamp": run.start_time,
                    "model": model,
                    "prompt_preview": run.inputs.get("prompt", "")[:100] + "..." if run.inputs else ""
                })
            
            analytics["avg_response_length"] = total_length / len(runs) if runs else 0
            analytics["recent_activity"] = sorted(
                analytics["recent_activity"], 
                key=lambda x: x["timestamp"],
                reverse=True
            )[:10]
            
            return analytics
            
        except Exception as e:
            return {"error": f"Failed to get analytics: {str(e)}"}
