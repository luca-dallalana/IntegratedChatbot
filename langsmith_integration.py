# langsmith_integration.py - LangSmith Integration for Investment Chatbot System
import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from langsmith import Client
from langsmith.evaluation import evaluate, run_evaluator
from langsmith.schemas import Run, Example
import streamlit as st

class LangSmithIntegration:
    """Integration layer for LangSmith evaluation and monitoring."""
    
    def __init__(self, api_key: Optional[str] = None, project_name: str = "investment-chatbot-comparison"):
        """
        Initialize LangSmith integration.
        
        Args:
            api_key: LangSmith API key (or set LANGSMITH_API_KEY env var)
            project_name: Name of the LangSmith project
        """
        # Set up LangSmith client
        if api_key:
            os.environ["LANGSMITH_API_KEY"] = api_key
        
        # Ensure we have an API key
        if not os.environ.get("LANGSMITH_API_KEY"):
            raise ValueError("LangSmith API key not provided. Set LANGSMITH_API_KEY environment variable or pass api_key parameter.")
        
        self.client = Client()
        self.project_name = project_name
        
        # Create or get project
        self._setup_project()
        
        # Define evaluation criteria specific to investment advice
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
        """Set up or get the LangSmith project."""
        try:
            # Set the project name for tracing
            os.environ["LANGCHAIN_PROJECT"] = self.project_name
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
        except Exception as e:
            print(f"Warning: Could not set up LangSmith project: {e}")
    
    async def create_dataset(self, test_cases: List[Dict], dataset_name: str = None) -> str:
        """
        Create a LangSmith dataset from test cases.
        
        Args:
            test_cases: List of test case dictionaries
            dataset_name: Optional custom dataset name
            
        Returns:
            Dataset ID
        """
        if not dataset_name:
            dataset_name = f"investment-test-cases-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        try:
            # Create dataset
            dataset = self.client.create_dataset(
                dataset_name=dataset_name,
                description="Investment advisor test cases for model comparison"
            )
            
            # Add examples to dataset
            for test_case in test_cases:
                self.client.create_example(
                    dataset_id=dataset.id,
                    inputs={
                        "prompt": test_case.get("prompt", ""),
                        "client_profile": test_case.get("variables", {}),
                        "test_case_id": test_case.get("id", ""),
                        "difficulty": test_case.get("difficulty", 1)
                    },
                    outputs={
                        "expected_elements": test_case.get("expected_elements", []),
                        "validation_criteria": test_case.get("validation_criteria", {})
                    }
                )
            
            return dataset.id
            
        except Exception as e:
            print(f"Error creating dataset: {e}")
            return None
    
    async def log_comparison_run(self, 
                                 model_name: str,
                                 prompt: str,
                                 response: str,
                                 test_case: Dict,
                                 evaluation_scores: Dict,
                                 run_metadata: Optional[Dict] = None) -> str:
        """
        Log a model comparison run to LangSmith.
        
        Args:
            model_name: Name of the model
            prompt: Input prompt
            response: Model response
            test_case: Test case information
            evaluation_scores: Evaluation scores from your system
            run_metadata: Additional metadata
            
        Returns:
            Run ID
        """
        run_id = str(uuid.uuid4())
        
        try:
            # Create run metadata
            metadata = {
                "model": model_name,
                "test_case_id": test_case.get("id", ""),
                "test_case_name": test_case.get("name", ""),
                "difficulty": test_case.get("difficulty", 1),
                "timestamp": datetime.now().isoformat(),
                **(run_metadata or {})
            }
            
            # Log the run
            self.client.create_run(
                name=f"{model_name}-{test_case.get('id', 'unknown')}",
                run_type="llm",
                inputs={"prompt": prompt, "client_profile": test_case.get("variables", {})},
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
    
    def create_investment_evaluators(self):
        """
        Create custom evaluators for investment advice quality.
        
        Returns:
            List of evaluator functions
        """
        evaluators = []
        
        # Risk Disclosure Evaluator
        @run_evaluator
        def evaluate_risk_disclosure(run: Run, example: Optional[Example] = None) -> dict:
            """Check if response properly discloses investment risks."""
            response = run.outputs.get("response", "").lower()
            
            risk_keywords = [
                "risk", "volatility", "loss", "fluctuation", "uncertainty",
                "downturn", "market risk", "not guaranteed", "past performance"
            ]
            
            risk_mentions = sum(1 for keyword in risk_keywords if keyword in response)
            score = min(1.0, risk_mentions / 3)  # Expect at least 3 risk mentions
            
            return {
                "key": "risk_disclosure",
                "score": score,
                "comment": f"Found {risk_mentions} risk-related terms"
            }
        
        evaluators.append(evaluate_risk_disclosure)
        
        # Regulatory Compliance Evaluator
        @run_evaluator
        def evaluate_regulatory_compliance(run: Run, example: Optional[Example] = None) -> dict:
            """Check for regulatory compliance indicators."""
            response = run.outputs.get("response", "").lower()
            
            compliance_indicators = [
                "not financial advice", "consult", "professional", "tax",
                "legal", "disclaimer", "general information", "individual circumstances"
            ]
            
            compliance_mentions = sum(1 for indicator in compliance_indicators if indicator in response)
            score = min(1.0, compliance_mentions / 2)
            
            return {
                "key": "regulatory_compliance",
                "score": score,
                "comment": f"Found {compliance_mentions} compliance indicators"
            }
        
        evaluators.append(evaluate_regulatory_compliance)
        
        # Personalization Evaluator
        @run_evaluator
        def evaluate_personalization(run: Run, example: Optional[Example] = None) -> dict:
            """Check if advice is personalized to client profile."""
            response = run.outputs.get("response", "").lower()
            client_profile = run.inputs.get("client_profile", {})
            
            # Check if response references client-specific details
            references = 0
            for key, value in client_profile.items():
                if str(value).lower() in response:
                    references += 1
            
            score = min(1.0, references / max(1, len(client_profile) * 0.5))
            
            return {
                "key": "personalization",
                "score": score,
                "comment": f"Referenced {references} of {len(client_profile)} client details"
            }
        
        evaluators.append(evaluate_personalization)
        
        # Actionability Evaluator
        @run_evaluator
        def evaluate_actionability(run: Run, example: Optional[Example] = None) -> dict:
            """Check if advice provides clear action items."""
            response = run.outputs.get("response", "").lower()
            
            action_indicators = [
                "should", "recommend", "suggest", "consider", "next step",
                "start by", "begin with", "first", "then", "finally"
            ]
            
            action_mentions = sum(1 for indicator in action_indicators if indicator in response)
            score = min(1.0, action_mentions / 4)
            
            return {
                "key": "actionability",
                "score": score,
                "comment": f"Found {action_mentions} action indicators"
            }
        
        evaluators.append(evaluate_actionability)
        
        return evaluators
    
    async def run_langsmith_evaluation(self,
                                      model_responses: Dict[str, str],
                                      test_case: Dict,
                                      prompt: str) -> Dict[str, Any]:
        """
        Run LangSmith evaluation on model responses.
        
        Args:
            model_responses: Dictionary of model names to responses
            test_case: Test case information
            prompt: Input prompt
            
        Returns:
            Evaluation results dictionary
        """
        results = {}
        evaluators = self.create_investment_evaluators()
        
        for model_name, response in model_responses.items():
            try:
                # Create a run for evaluation
                run_id = await self.log_comparison_run(
                    model_name=model_name,
                    prompt=prompt,
                    response=response,
                    test_case=test_case,
                    evaluation_scores={}
                )
                
                # Run evaluators
                eval_results = {}
                for evaluator in evaluators:
                    # Create a mock run object for evaluation
                    mock_run = Run(
                        id=run_id,
                        name=f"{model_name}-evaluation",
                        run_type="llm",
                        inputs={"prompt": prompt, "client_profile": test_case.get("variables", {})},
                        outputs={"response": response}
                    )
                    
                    result = evaluator(mock_run)
                    eval_results[result["key"]] = {
                        "score": result["score"],
                        "comment": result.get("comment", "")
                    }
                
                # Calculate weighted overall score
                overall_score = sum(
                    eval_results.get(criterion, {}).get("score", 0) * self.evaluation_criteria[criterion]["weight"]
                    for criterion in self.evaluation_criteria
                )
                
                results[model_name] = {
                    "run_id": run_id,
                    "evaluations": eval_results,
                    "overall_score": overall_score,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                results[model_name] = {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        return results
    
    def create_comparison_experiment(self,
                                    experiment_name: str,
                                    models: List[str],
                                    test_cases: List[Dict]) -> str:
        """
        Create a LangSmith experiment for model comparison.
        
        Args:
            experiment_name: Name of the experiment
            models: List of model names
            test_cases: List of test cases
            
        Returns:
            Experiment ID
        """
        try:
            # Create experiment metadata
            experiment_metadata = {
                "name": experiment_name,
                "models": models,
                "num_test_cases": len(test_cases),
                "created_at": datetime.now().isoformat(),
                "evaluation_criteria": list(self.evaluation_criteria.keys())
            }
            
            # Log experiment
            experiment_id = str(uuid.uuid4())
            
            self.client.create_run(
                name=experiment_name,
                run_type="experiment",
                inputs={"test_cases": [tc.get("id") for tc in test_cases]},
                outputs={"models": models},
                extra={"metadata": experiment_metadata},
                project_name=self.project_name,
                id=experiment_id
            )
            
            return experiment_id
            
        except Exception as e:
            print(f"Error creating experiment: {e}")
            return None
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """
        Retrieve results for a specific experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Dictionary of experiment results
        """
        try:
            # Retrieve runs for the experiment
            runs = self.client.list_runs(
                project_name=self.project_name,
                filter=f"eq(extra.experiment_id, '{experiment_id}')"
            )
            
            results = {
                "experiment_id": experiment_id,
                "runs": [],
                "summary": {}
            }
            
            for run in runs:
                results["runs"].append({
                    "run_id": run.id,
                    "model": run.extra.get("metadata", {}).get("model"),
                    "test_case": run.extra.get("metadata", {}).get("test_case_id"),
                    "scores": run.extra.get("evaluation_scores", {})
                })
            
            return results
            
        except Exception as e:
            print(f"Error retrieving experiment results: {e}")
            return {}

def integrate_langsmith_with_streamlit(st_component):
    """
    Add LangSmith integration controls to Streamlit sidebar.
    
    Args:
        st_component: Streamlit component for sidebar
    """
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
        help="Send evaluation data to LangSmith for detailed analysis"
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
            st_component.error(f"❌ LangSmith connection failed: {e}")
            return None
    
    return None

def display_langsmith_results(results: Dict[str, Any]):
    """
    Display LangSmith evaluation results in Streamlit.
    
    Args:
        results: LangSmith evaluation results
    """
    st.header("🔍 LangSmith Evaluation Results")
    
    if not results:
        st.info("No LangSmith results available")
        return
    
    # Display results for each model
    for model_name, model_results in results.items():
        with st.expander(f"{model_name} - LangSmith Analysis"):
            if "error" in model_results:
                st.error(f"Evaluation error: {model_results['error']}")
            else:
                # Overall score
                st.metric(
                    "LangSmith Overall Score",
                    f"{model_results['overall_score']:.2%}"
                )
                
                # Individual evaluations
                st.subheader("Detailed Evaluations")
                
                eval_data = []
                for criterion, eval_result in model_results.get("evaluations", {}).items():
                    eval_data.append({
                        "Criterion": criterion.replace("_", " ").title(),
                        "Score": f"{eval_result['score']:.2%}",
                        "Comment": eval_result.get("comment", "")
                    })
                
                if eval_data:
                    import pandas as pd
                    eval_df = pd.DataFrame(eval_data)
                    st.dataframe(eval_df, use_container_width=True)
                
                # Run ID for traceability
                if "run_id" in model_results:
                    st.caption(f"Run ID: {model_results['run_id']}")
                    st.caption(f"Timestamp: {model_results['timestamp']}")