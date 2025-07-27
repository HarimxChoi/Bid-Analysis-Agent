# src/binary_classifier/evaluate.py

import os
import sys
import logging
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# --- 1. Import Shared Modules & Configure Environment ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.shared.data_utils import load_dataset, clean_text

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- 2. Configuration & Paths ---
class EvalConfig:
    # --- Paths ---
    MODEL_DIR = r'[model\best_model_fold_1]' 
    TEST_DATA_PATH = r'[data\custom_testset.csv]' 
    OUTPUT_DIR = r'[binary_classifier_output\evaluation_results]'
    
    # --- Model & Tokenizer ---
    BASE_MODEL_NAME = "klue/roberta-large"
    MAX_LENGTH = 128
    
    # --- Evaluation ---
    BATCH_SIZE = 32

def main():
    """Main evaluation pipeline for the binary classifier."""
    config = EvalConfig()
    
    # --- Create output directory ---
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    logger.info(f"Evaluation results will be saved to: {config.OUTPUT_DIR}")

    # --- 3. Load Model and Tokenizer ---
    logger.info(f"Loading the fine-tuned model from: {config.MODEL_DIR}")
    try:
        # Load the base model with the correct number of labels
        base_model = AutoModelForSequenceClassification.from_pretrained(config.BASE_MODEL_NAME, num_labels=2)
        
        # Apply the LoRA adapter to the base model
        model = PeftModel.from_pretrained(base_model, config.MODEL_DIR)
        
        # It's often recommended to merge the weights for faster inference
        model = model.merge_and_unload()
        
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_DIR)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        logger.info(f"Model loaded successfully and moved to {device}.")
    except Exception as e:
        logger.error(f"Fatal: Failed to load the model. Error: {e}")
        sys.exit(1)

    # --- 4. Load and Prepare Test Data ---
    logger.info(f"Loading test data from: {config.TEST_DATA_PATH}")
    test_df = load_dataset(config.TEST_DATA_PATH)
    test_df['cleaned_text'] = test_df['용역명'].apply(clean_text)
    
    test_queries = test_df['cleaned_text'].tolist()
    test_labels = test_df['label'].tolist()
    
    # --- 5. Run Inference ---
    logger.info(f"Running inference on {len(test_queries)} test samples...")
    all_predictions = []
    all_confidences = []

    for i in tqdm(range(0, len(test_queries), config.BATCH_SIZE), desc="Evaluating"):
        batch_texts = test_queries[i:i + config.BATCH_SIZE]
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=config.MAX_LENGTH
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        
        probabilities = torch.softmax(outputs.logits, dim=-1)
        confidences, predictions = torch.max(probabilities, dim=-1)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_confidences.extend(confidences.cpu().numpy())

    # --- 6. Analyze and Save Results ---
    logger.info("Analyzing and saving results...")

    # --- Classification Report ---
    report_str = classification_report(
        test_labels, 
        all_predictions, 
        target_names=['불가능 (Class 0)', '가능 (Class 1)'],
        digits=4
    )
    logger.info("\n--- Classification Report ---\n" + report_str)
    with open(os.path.join(config.OUTPUT_DIR, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_str)

    # --- Core Metrics ---
    accuracy = accuracy_score(test_labels, all_predictions)
    f1_macro = f1_score(test_labels, all_predictions, average='macro')
    logger.info(f"Overall Accuracy: {accuracy:.4f}")
    logger.info(f"Macro F1-Score: {f1_macro:.4f}")

    metrics_summary = {"accuracy": accuracy, "f1_macro": f1_macro}
    with open(os.path.join(config.OUTPUT_DIR, 'metrics_summary.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=4)
        
    # --- Visualizations ---
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    # 1. Confusion Matrix
    cm = confusion_matrix(test_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['불가능', '가능'], yticklabels=['불가능', '가능'])
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'confusion_matrix.png'), dpi=300)
    plt.close()
    logger.info("Saved confusion matrix plot.")

    # 2. Confidence Distribution
    confidence_df = pd.DataFrame({
        'confidence': all_confidences,
        'correct': np.array(all_predictions) == np.array(test_labels)
    })
    plt.figure(figsize=(12, 6))
    sns.histplot(data=confidence_df, x='confidence', hue='correct', multiple='stack', 
                 bins=30, palette={True: '#3498db', False: '#e74c3c'})
    plt.title('Prediction Confidence Distribution', fontsize=16)
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Correct Prediction', labels=['Yes', 'No'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'confidence_distribution.png'), dpi=300)
    plt.close()
    logger.info("Saved confidence distribution plot.")

    # --- Error Analysis ---
    test_df['prediction'] = all_predictions
    test_df['confidence'] = all_confidences
    error_df = test_df[test_df['label'] != test_df['prediction']]
    error_df.to_csv(os.path.join(config.OUTPUT_DIR, 'error_analysis.csv'), index=False, encoding='utf-8-sig')
    logger.info(f"Saved {len(error_df)} misclassified samples to error_analysis.csv.")
    
    logger.info("Evaluation pipeline finished successfully.")

if __name__ == "__main__":
    main()