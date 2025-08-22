# src/api/server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List

# LangSmith imports
try:
    from langsmith import Client
    from langchain.callbacks.tracers import LangChainTracer
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

app = FastAPI(title="Investment Chatbot Comparison API with LangSmith", version="3.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class TracedChatRequest(BaseModel):
    model: str
    prompt: str
    system_prompt: str
    openai_api_key: str
    langsmith_api_key: str
    langsmith_project_name: str
    session_id: str

class DualChatbotResult(BaseModel):
    session_id: str
    user_prompt: str
    responses: Dict[str, str]
    models: List[str]
    prompt_style: int
    timestamp: str

# --- In-memory storage ---
global_results_storage = {}

# --- API Endpoints ---
@app.post("/traced-chat")
async def traced_chat(request: TracedChatRequest):
    """Handles a chat request and ensures it is traced to LangSmith."""
    if not LANGSMITH_AVAILABLE:
        raise HTTPException(status_code=500, detail="LangSmith components are not available. Please install required packages.")
    
    try:
        # Initialize tracer for this specific call
        tracer = LangChainTracer(
            project_name=request.langsmith_project_name,
            client=Client(api_key=request.langsmith_api_key)
        )

        # Initialize ChatOpenAI model
        llm = ChatOpenAI(
            model_name=request.model,
            openai_api_key=request.openai_api_key,
            temperature=0.7,
            max_tokens=500,
            streaming=False # Ensure we get the full response for logging
        )

        # Make the call with the tracer in the callbacks
        response = llm.invoke(
            [
                SystemMessage(content=request.system_prompt),
                HumanMessage(content=request.prompt)
            ],
            config={
                "callbacks": [tracer],
                "metadata": {
                    "session_id": request.session_id,
                    "model": request.model
                }
            }
        )
        
        return {"response": response.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/store-dual-results")
async def store_dual_results(result: DualChatbotResult):
    """Store results from dual chatbot session."""
    global_results_storage[result.session_id] = result.dict()
    return {"status": "success", "message": "Results stored successfully"}

@app.get("/get-dual-results/{session_id}")
async def get_dual_results(session_id: str):
    """Get results from dual chatbot session."""
    if session_id in global_results_storage:
        return global_results_storage[session_id]
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/")
async def root():
    return {
        "message": "Investment Chatbot Comparison API with LangSmith",
        "version": "3.1.0",
        "features": ["LangSmith Integration", "Advanced Analytics", "Data Extraction", "Traced Chat"],
        "endpoints": ["/traced-chat", "/store-dual-results", "/get-dual-results", "/health"]
    }

@app.get("/health")
async def health_check():
    from datetime import datetime
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
