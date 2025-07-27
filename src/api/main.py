# src/intelligent_router_api/main.py

import os
import sys
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import uvicorn

# --- 1. Import Core Logic ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# This is the only AI-related import we need here!
from api.agent_core_logic import HybridClassificationSystem

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- 2. Configuration ---
CONFIG = {
    'BINARY_MODEL_DIR': r'C:\devdub\bidNLP\quantized_models\binary_classifier_onnx',
    'MULTICLASS_MODEL_DIR': r'C:\devdub\bidNLP\quantized_models\multiclass_classifier_onnx',
    'VECTOR_DB_DIR': r'C:\devdub\bidNLP\vector_db',
    'SBERT_MODEL_NAME': 'jhgan/ko-sbert-sts',
    'GEMINI_API_KEY': os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE"),
    'GEMINI_MODEL_ID': 'gemini-2.5-flash',
    'CONFIDENCE_THRESHOLD': 0.8, 
    'NUM_MULTICLASS_LABELS': 22
    }

# --- 3. FastAPI App Initialization & AI System Singleton ---
app = FastAPI(
    title="Intelligent Bid Analysis API",
    description="An adaptive AI system that classifies public procurement bids using a hybrid of fast models and an expert agent system.",
    version="2.0.0"
)

# This global variable will hold our single, powerful AI system instance.
ai_system: HybridClassificationSystem = None

@app.on_event("startup")
def startup_event():
    """
    This event is triggered when the API server starts.
    It loads all AI models into memory once.
    """
    global ai_system
    logger.info("Server startup: Initializing the HybridClassificationSystem...")
    try:
        ai_system = HybridClassificationSystem(CONFIG)
    except Exception as e:
        logger.critical(f"CRITICAL: AI System failed to initialize during startup: {e}", exc_info=True)
        # In a real production system, this might trigger an alert.
        ai_system = None

# --- 4. Pydantic Models for Request & Response ---
# These models define the data structure for our API, providing automatic validation.

class BidItem(BaseModel):
    id: str = Field(..., description="A unique identifier for the bid notice.", example="20240512345-00")
    text: str = Field(..., description="The title or text of the bid notice to classify.", example="세종시 스마트 국가산단 고가차도 타당성 조사")

class ClassificationRequest(BaseModel):
    bids: List[BidItem]

class ClassificationResponseItem(BaseModel):
    """
    여기에 카테고리 분류에대한 confidence를 추가해야됨
    그리고 카테고리 신회도의 임계치도 도입해서 임계치 이하면 미분류로 해야함
    임계치 : 0.65
    """
    id: str
    decision: str = Field(..., description="The final classification: '가능' or '불가능'.")
    category: str = Field(..., description="The detailed service category.")
    confidence: float = Field(..., description="The confidence score of the initial binary prediction.")
    reasoning: str = Field(..., description="The reasoning behind the decision.")
    is_agent_used: bool = Field(..., description="True if the expert agent system was used for this decision.")
    
class ClassificationResponse(BaseModel):
    results: List[ClassificationResponseItem]

# --- 5. API Endpoint ---
@app.post("/classify", response_model=ClassificationResponse)
async def classify_bids(request: ClassificationRequest):
    """
    Classifies a batch of bid notices using the adaptive hybrid AI system.
    """
    if ai_system is None:
        raise HTTPException(
            status_code=503, # Service Unavailable
            detail="AI system is not available due to an initialization error. Please check server logs."
        )

    response_items = []
    for bid in request.bids:
        logger.info(f"Processing bid ID: {bid.id}")
        
        result = await ai_system.classify(bid.text)
        
        # Determine if the agent was used based on the reasoning provided.
        agent_used = "High confidence" not in result.get('reasoning', "")

        response_items.append(
            ClassificationResponseItem(
                id=bid.id,
                decision=result.get('final_decision', 'Error'),
                category=result.get('final_category', 'Unclassified'),
                confidence=result.get('confidence', 0.0),
                reasoning=result.get('reasoning', 'An unknown error occurred.'),
                is_agent_used=agent_used
            )
        )
        
    return ClassificationResponse(results=response_items)

# --- Exception Handler for a cleaner error response ---
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"An unhandled exception occurred: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": "An internal server error occurred. Please contact the administrator."},
    )

# --- 6. Main Entry Point for Running the Server ---
if __name__ == "__main__":
    logger.info("Starting Uvicorn server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)