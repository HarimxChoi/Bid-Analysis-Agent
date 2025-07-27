# src/shared/data_utils.py

import pandas as pd
import numpy as np
import torch
import logging
import re
import html
from sklearn.utils.class_weight import compute_class_weight
from typing import Union, List, Dict

# --- Logger Setup ---
logger = logging.getLogger(__name__)

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads data from the given file path (.csv or .xlsx).
    Automatically handles common encoding issues for Korean CSV files.

    Args:
        file_path (str): The path to the file to load.

    Returns:
        pd.DataFrame: The loaded DataFrame.
        
    Raises:
        ValueError: If the file format is unsupported or the file cannot be loaded with any encoding.
    """
    logger.info(f"Attempting to load data from '{file_path}'...")
    
    if file_path.endswith('.csv'):
        # --- CSV Loading Logic with Encoding Fallback ---
        encodings_to_try = ['utf-8', 'cp949', 'euc-kr']
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"Success: Loaded CSV file with '{encoding}' encoding.")
                return df
            except UnicodeDecodeError:
                logger.warning(f"Warning: Failed to load with '{encoding}'. Trying next...")
        
        raise ValueError(f"Fatal: Failed to load '{file_path}' with all attempted encodings: {', '.join(encodings_to_try)}.")

    elif file_path.endswith('.xlsx'):
        # --- Excel Loading Logic ---
        df = pd.read_excel(file_path)
        logger.info(f"Success: Loaded Excel file.")
        return df
    else:
        raise ValueError("Unsupported file format. Please use .csv or .xlsx.")

def clean_text(text: str) -> str:
    """
    Cleans an input text string by removing unnecessary whitespace and decoding HTML entities.

    Args:
        text (str): The original string to be cleaned.

    Returns:
        str: The cleaned string. Returns an empty string if input is not a string.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Decode HTML entities (e.g., & -> &)
    cleaned = html.unescape(text)
    
    # 2. Collapse multiple whitespaces, tabs, and newlines into a single space
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

def analyze_class_distribution(df: pd.DataFrame, label_column: str) -> Dict[str, Dict[str, Union[int, float]]]:
    """
    Analyzes and logs the class distribution for a given label column in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        label_column (str): The name of the column containing the labels.

    Returns:
        Dict: A dictionary containing the count and ratio for each class.
    """
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found in DataFrame.")
    
    logger.info(f"--- Analyzing Class Distribution for '{label_column}' ---")
    counts = df[label_column].value_counts()
    ratios = df[label_column].value_counts(normalize=True)
    
    summary = {}
    for label, count in counts.items():
        ratio = ratios[label]
        logger.info(f"  - Class '{label}': {count} samples ({ratio:.2%})")
        summary[label] = {'count': count, 'ratio': ratio}
        
    logger.info("-" * 50)
    return summary
    
def calculate_class_weights(labels: pd.Series) -> torch.Tensor:
    """
    Calculates class weights for handling class imbalance from a pandas Series of labels.
    
    Args:
        labels (pd.Series): A pandas Series containing the class labels.

    Returns:
        torch.Tensor: A tensor of weights suitable for use in PyTorch loss functions.
    """
    unique_labels = np.unique(labels)
    weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float)
    
    logger.info(f"Calculated class weights.")
    logger.info(f"  - Unique Labels: {unique_labels}")
    logger.info(f"  - Corresponding Weights: {class_weights_tensor.numpy()}")
    
    return class_weights_tensor

if __name__ == '__main__':
    # --- Test code for direct execution of this script ---
    # This block allows you to test the functions independently.
    logging.basicConfig(level=logging.INFO) # Set logger level for testing
    
    # Create temporary test data
    test_data = {
        'text': [
            '  2024년 OO대교  정밀안전점검  ',
            '신규 프로젝트 기획 회의',
            '한강 교량 보수공사 설계',
            '소프트웨어 개발자 채용 공고',
            '세종시 스마트 국가산단 타당성 조사'
        ],
        'label': [1, 0, 1, 0, 1]
    }
    test_df = pd.DataFrame(test_data)
    test_csv_path = 'temp_test_data.csv'
    test_df.to_csv(test_csv_path, index=False, encoding='cp949')

    print("\n" + "="*50)
    print("1. Testing `load_dataset` function")
    print("="*50)
    loaded_df = load_dataset(test_csv_path)
    print("Loaded data sample:\n", loaded_df.head())
    
    print("\n" + "="*50)
    print("2. Testing `clean_text` function")
    print("="*50)
    loaded_df['cleaned_text'] = loaded_df['text'].apply(clean_text)
    print("Cleaned text sample:\n", loaded_df[['text', 'cleaned_text']])
    
    print("\n" + "="*50)
    print("3. Testing `analyze_class_distribution` function")
    print("="*50)
    analyze_class_distribution(loaded_df, 'label')
    
    print("\n" + "="*50)
    print("4. Testing `calculate_class_weights` function")
    print("="*50)
    calculate_class_weights(loaded_df['label'])

    # Clean up temporary file
    import os
    os.remove(test_csv_path)