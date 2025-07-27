# src/shared/vector_db_utils.py

import faiss
import numpy as np
import json
import os
import logging
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

# --- Logger Setup ---
logger = logging.getLogger(__name__)

def load_sbert_model(model_name: str = 'jhgan/ko-sbert-sts', device: str = None) -> SentenceTransformer:
    """
    Loads a pre-trained SentenceTransformer model.

    Args:
        model_name (str): The name of the model to load from Hugging Face Hub.
        device (str, optional): The device to load the model onto ('cuda', 'cpu'). 
                                If None, it will be automatically selected.

    Returns:
        SentenceTransformer: The loaded model.
    """
    logger.info(f"Loading SBERT model: {model_name}...")
    model = SentenceTransformer(model_name, device=device)
    logger.info("SBERT model loaded successfully.")
    return model

def build_and_save_vector_db(
    df: pd.DataFrame, 
    model: SentenceTransformer, 
    output_dir: str, 
    text_column: str, 
    metadata_columns: List[str]
):
    """
    Builds a Faiss vector database from a DataFrame and saves it to disk.

    Args:
        df (pd.DataFrame): The DataFrame containing the source data.
        model (SentenceTransformer): The SBERT model to use for encoding.
        output_dir (str): The directory to save the index and metadata files.
        text_column (str): The name of the column containing the text to be embedded.
        metadata_columns (List[str]): A list of column names to store as metadata.
    """
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in the DataFrame.")
    
    logger.info(f"Starting vector DB build process for {len(df)} documents...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Encode texts into embeddings
    logger.info(f"Encoding texts from '{text_column}' column...")
    embeddings = model.encode(df[text_column].tolist(), show_progress_bar=True, convert_to_tensor=False)
    embeddings = np.array(embeddings, dtype='float32')
    
    # 2. Build Faiss index
    logger.info("Building Faiss index (IndexFlatL2)...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    # 3. Save Faiss index to disk
    index_path = os.path.join(output_dir, "faiss.index")
    faiss.write_index(index, index_path)
    logger.info(f"Fa iss index saved to: {index_path}")

    # 4. Create and save metadata
    # Metadata maps the index position (0, 1, 2...) to the original data.
    id_to_data = df[metadata_columns].to_dict(orient='index')
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(id_to_data, f, ensure_ascii=False, indent=4)
    logger.info(f"Metadata saved to: {metadata_path}")
    logger.info("Vector DB build process completed.")

class SemanticSearcher:
    """
    A class that encapsulates loading a vector DB and performing similarity searches.
    """
    def __init__(self, db_dir: str, model: SentenceTransformer):
        """
        Initializes the searcher by loading the Faiss index and metadata.

        Args:
            db_dir (str): The directory containing 'faiss.index' and 'metadata.json'.
            model (SentenceTransformer): An already loaded SBERT model.
        """
        index_path = os.path.join(db_dir, "faiss.index")
        metadata_path = os.path.join(db_dir, "metadata.json")

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Could not find index or metadata files in '{db_dir}'. Please build the DB first.")

        logger.info(f"Loading vector DB from '{db_dir}'...")
        self.model = model
        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            # JSON keys are strings, so convert them back to integers for indexing.
            self.id_to_data = {int(k): v for k, v in json.load(f).items()}
        logger.info("SemanticSearcher initialized successfully.")

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs a similarity search for a given query.

        Args:
            query (str): The text query to search for.
            k (int): The number of top similar documents to return.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                 contains the retrieved document's data and its similarity score.
        """
        if not query:
            return []
            
        # Encode the query and search the index
        query_embedding = self.model.encode([query], convert_to_tensor=False).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1: # Faiss returns -1 for non-existent indices
                # Combine metadata with the search result
                retrieved_item = self.id_to_data[idx].copy()
                # L2 distance is not bounded. Cosine similarity is more interpretable.
                # For simplicity, we can use the distance directly or calculate cosine similarity.
                # Here, we use a simple transformation for a "score", not true cosine similarity.
                retrieved_item['score'] = 1 / (1 + distances[0][i]) # Simple distance-to-score conversion
                results.append(retrieved_item)
                
        return results

if __name__ == '__main__':
    # --- Test code for direct execution of this script ---
    logging.basicConfig(level=logging.INFO)
    
    # 1. Create dummy data and resources
    test_data = {
        '용역명': ["한강 교량 정밀안전점검", "서울시 도로 설계 용역", "어린이날 기념 행사 기획", "AI 챗봇 소프트웨어 개발"],
        'label': [1, 1, 0, 0]
    }
    test_df = pd.DataFrame(test_data)
    test_model = load_sbert_model()
    test_output_dir = "temp_test_vector_db"

    print("\n" + "="*50)
    print("1. Testing `build_and_save_vector_db` function")
    print("="*50)
    build_and_save_vector_db(
        df=test_df, 
        model=test_model, 
        output_dir=test_output_dir,
        text_column='용역명',
        metadata_columns=['용역명', 'label']
    )

    # 2. Test the SemanticSearcher class
    print("\n" + "="*50)
    print("2. Testing `SemanticSearcher` class")
    print("="*50)
    try:
        searcher = SemanticSearcher(db_dir=test_output_dir, model=test_model)
        
        test_query = "강다리 안전 진단"
        search_results = searcher.search(test_query, k=2)
        
        print(f"Query: '{test_query}'")
        print("Search Results:")
        for result in search_results:
            print(f"  - Score: {result['score']:.4f}, Text: {result['용역명']}, Label: {result['label']}")
            
    except Exception as e:
        print(f"An error occurred during testing: {e}")

    # 3. Clean up temporary directory
    import shutil
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
        print(f"\nCleaned up temporary directory: {test_output_dir}")