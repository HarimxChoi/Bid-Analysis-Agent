# src/intelligent_router_api/core.py

import os
import sys
import logging
import json
import torch
import faiss
import numpy as np
from typing import TypedDict, List, Optional, Any

# --- 1. Imports ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
import google.genai as genai

# --- Logger Setup ---
logger = logging.getLogger(__name__)

# --- 2. LangGraph State Definition ---
# This defines the "shared memory" for our agent workflow.
class AgentState(TypedDict):
    query: str
    binary_prediction: Optional[str]
    binary_confidence: Optional[float]
    multiclass_prediction: Optional[str]
    retrieved_docs: Optional[List[dict]]
    final_decision: Optional[str]
    final_reasoning: Optional[str]
    final_category: Optional[str]
    error: Optional[str]

# --- 3. The Core System Class ---
class HybridClassificationSystem:
    """
    The master class that orchestrates all AI components and decision workflows.
    Designed to be initialized once and used repeatedly by the API server.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # --- Load all assets once ---
        self._load_models()
        
        # --- Build the agent graph ---
        self.agent_graph = self._build_agent_graph()
        logger.info("Hybrid Classification System initialized successfully.")

    def _load_models(self):
        """Loads all necessary models and resources into memory."""
        logger.info("Loading all AI models...")
        # Load optimized ONNX models for fast inference
        self.binary_model = ORTModelForSequenceClassification.from_pretrained(self.config['BINARY_MODEL_DIR'], provider="CPUExecutionProvider")
        self.binary_tokenizer = AutoTokenizer.from_pretrained(self.config['BINARY_MODEL_DIR'])
        
        self.multiclass_model = ORTModelForSequenceClassification.from_pretrained(self.config['MULTICLASS_MODEL_DIR'], provider="CPUExecutionProvider")
        self.multiclass_tokenizer = AutoTokenizer.from_pretrained(self.config['MULTICLASS_MODEL_DIR'])
        
        # Load Semantic Search assets
        self.sbert_model = SentenceTransformer(self.config['SBERT_MODEL_NAME'], device=self.device)
        self.faiss_index = faiss.read_index(os.path.join(self.config['VECTOR_DB_DIR'], "faiss.index"))
        with open(os.path.join(self.config['VECTOR_DB_DIR'], "metadata.json"), 'r', encoding='utf-8') as f:
            self.id_to_data = {int(k): v for k, v in json.load(f).items()}

        # Setup LLM client
        genai.configure(api_key=self.config['GEMINI_API_KEY'])
        self.judge_client = genai.GenerativeModel(self.config['GEMINI_MODEL_ID'])
        
    # --- Node Definitions for the Agent Graph ---
    
    def _run_binary_inference(self, text: str) -> tuple[str, float]:
        """Internal helper for binary classification."""
        inputs = self.binary_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        if 'token_type_ids' in inputs: del inputs['token_type_ids']
        
        with torch.no_grad():
            outputs = self.binary_model(**inputs)
        
        logits = torch.from_numpy(outputs.logits)
        probabilities = torch.softmax(logits, dim=-1)[0]
        confidence, pred_id = torch.max(probabilities, dim=-1)
        
        label = self.binary_model.config.id2label[pred_id.item()]
        return label, confidence.item()

    def _run_multiclass_inference(self, text: str) -> str:
        """Internal helper for multi-class classification."""
        inputs = self.multiclass_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        if 'token_type_ids' in inputs: del inputs['token_type_ids']
        
        with torch.no_grad():
            outputs = self.multiclass_model(**inputs)
            
        logits = torch.from_numpy(outputs.logits)
        pred_id = torch.argmax(logits, dim=-1).item()
        return self.multiclass_model.config.id2label[pred_id]

    # --- LangGraph Nodes ---

    def primary_analyst_node(self, state: AgentState) -> AgentState:
        """Node 1: Initial analysis using the fast binary classifier."""
        logger.info(f"[{state['query'][:30]}...] - Running Primary Analyst...")
        pred, conf = self._run_binary_inference(state['query'])
        state['binary_prediction'] = pred
        state['binary_confidence'] = conf
        return state

    def semantic_searcher_node(self, state: AgentState) -> AgentState:
        """Node 2: Retrieve relevant documents if confidence is low."""
        logger.info(f"[{state['query'][:30]}...] - Running Semantic Searcher...")
        query_embedding = self.sbert_model.encode([state['query']]).astype('float32')
        distances, indices = self.faiss_index.search(query_embedding, k=3)
        
        docs = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                docs.append({
                    "text": self.id_to_data[idx]['용역명'],
                    "result": "가능" if self.id_to_data[idx]['label'] == 1 else "불가능"
                })
        state['retrieved_docs'] = docs
        return state

    def final_judge_node(self, state: AgentState) -> AgentState:
        """Node 3: Use LLM to make a final decision based on all evidence."""
        logger.info(f"[{state['query'][:30]}...] - Running Final Judge (LLM)...")
        # This prompt is the core of the Judge's reasoning
        prompt = self._construct_judge_prompt(state)
        try:
            response = self.judge_client.generate_content(prompt)
            json_response = json.loads(response.text.strip().replace("```json", "").replace("```", ""))
            state['final_decision'] = json_response.get("final_decision")
            state['final_reasoning'] = json_response.get("final_reasoning")
        except Exception as e:
            logger.error(f"LLM Judge failed: {e}. Falling back to analyst prediction.")
            state['final_decision'] = state['binary_prediction'] # Fallback
            state['final_reasoning'] = "LLM Error: Fallback to initial analysis."
        return state

    def _construct_judge_prompt(self, state: AgentState) -> str:
        """Helper to create a detailed prompt for the LLM Judge."""
        docs_str = "\n".join([f"- \"{doc['text']}\" (Past Result: {doc['result']})" for doc in state['retrieved_docs']])
        prompt = f"""
        **CONTEXT:** You are a senior bid review officer at a civil engineering firm. You must make a final, reasoned decision based on reports from two junior analysts.

        **BID TITLE:** "{state['query']}"

        **REPORT 1: PROBABILITY ANALYSIS**
        - Initial Prediction: **{state['binary_prediction']}**
        - Confidence Score: {state['binary_confidence']:.2%} (This was deemed too low for an automatic decision).

        **REPORT 2: HISTORICAL CASE ANALYSIS**
        - Top 3 Similar Past Cases:
        {docs_str}

        **YOUR TASK:**
        Synthesize both reports and make a final determination. Your response MUST be a single JSON object with the following format:
        {{"final_decision": "가능 or 불가능", "final_reasoning": "A brief sentence explaining your core logic."}}
        """
        return prompt

    def router_node(self, state: AgentState) -> str:
        """The dynamic router that decides the workflow path."""
        confidence = state.get('binary_confidence', 0)
        if confidence >= self.config['CONFIDENCE_THRESHOLD']:
            return "fast_path"
        else:
            return "agent_path"

    def fast_path_finalizer_node(self, state: AgentState) -> AgentState:
        """Finalizer for the high-confidence path."""
        logger.info(f"[{state['query'][:30]}...] - Executing Fast Path.")
        state['final_decision'] = state['binary_prediction']
        state['final_reasoning'] = f"High confidence ({state['binary_confidence']:.2%}) prediction by FT model."
        # Run multiclass classification for the final category
        state['final_category'] = self._run_multiclass_inference(state['query'])
        return state
        
    def agent_path_finalizer_node(self, state: AgentState) -> AgentState:
        """Finalizer for the low-confidence (agent) path."""
        logger.info(f"[{state['query'][:30]}...] - Executing Agent Path.")
        # The agent's decision is already in 'final_decision' from the judge_node
        # Run multiclass classification for the final category
        state['final_category'] = self._run_multiclass_inference(state['query'])
        return state

    def _build_agent_graph(self) -> Any:
        """Builds and compiles the complete LangGraph workflow."""
        workflow = StateGraph(self.AgentState)
        
        workflow.add_node("primary_analyst", self.primary_analyst_node)
        workflow.add_node("semantic_searcher", self.semantic_searcher_node)
        workflow.add_node("final_judge", self.final_judge_node)
        workflow.add_node("fast_path_finalizer", self.fast_path_finalizer_node)
        workflow.add_node("agent_path_finalizer", self.agent_path_finalizer_node)
        
        workflow.set_entry_point("primary_analyst")
        
        workflow.add_conditional_edges(
            "primary_analyst",
            self.router_node,
            {
                "fast_path": "fast_path_finalizer",
                "agent_path": "semantic_searcher"
            }
        )
        
        workflow.add_edge("semantic_searcher", "final_judge")
        workflow.add_edge("final_judge", "agent_path_finalizer")
        
        workflow.add_edge("fast_path_finalizer", END)
        workflow.add_edge("agent_path_finalizer", END)
        
        return workflow.compile()

    # --- Main Public Method for API ---
    
    async def classify(self, query: str) -> Dict[str, Any]:
        """
        The single entry point for the API to classify a query using the hybrid system.
        """
        # LangGraph's .ainvoke is used for asynchronous execution in FastAPI
        final_state = await self.agent_graph.ainvoke({"query": query})
        
        return {
            "query": query,
            "final_decision": final_state.get('final_decision'),
            "final_category": final_state.get('final_category'),
            "confidence": final_state.get('binary_confidence'),
            "reasoning": final_state.get('final_reasoning'),
            "evidence": final_state.get('retrieved_docs', [])
        }