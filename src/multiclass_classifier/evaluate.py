# src/multiclass_classifier/evaluate.py

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
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
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
    # This should point to the best model saved by the multiclass train.py
    MODEL_DIR = r'[model\best_model]' 
    TEST_DATA_PATH = r'[data\multiclass_testset.csv]' # Your multiclass test samples
    OUTPUT_DIR = r'[multiclass_classifier_output\evaluation_results]'
    
    # --- Model & Tokenizer ---
    BASE_MODEL_NAME = "klue/roberta-large"
    MAX_LENGTH = 128
    
    # --- Evaluation ---
    BATCH_SIZE = 32

def main():
    """Main evaluation pipeline for the multi-class classifier."""
    config = EvalConfig()
    
    # --- Create output directory ---
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    logger.info(f"Evaluation results will be saved to: {config.OUTPUT_DIR}")

    # --- 3. Load Model and Tokenizer ---
    logger.info(f"Loading the fine-tuned model from: {config.MODEL_DIR}")
    try:
        # Load the configuration from the saved model directory.
        # This ensures we get the correct num_labels, id2label, and label2id mappings.
        model_config = AutoConfig.from_pretrained(config.MODEL_DIR)
        
        # Load the base model with this correct configuration
        base_model = AutoModelForSequenceClassification.from_pretrained(
            config.BASE_MODEL_NAME,
            config=model_config,
        )
        
        # Apply the LoRA adapter to the base model
        model = PeftModel.from_pretrained(base_model, config.MODEL_DIR)
        
        # Merge the weights for faster inference
        model = model.merge_and_unload()
        
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_DIR)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully and moved to {device}.")
        logger.info(f"Model configured for {model.config.num_labels} classes.")

    except Exception as e:
        logger.error(f"Fatal: Failed to load the model. Error: {e}", exc_info=True)
        sys.exit(1)

    # --- 4. Load and Prepare Test Data ---
    logger.info(f"Loading test data from: {config.TEST_DATA_PATH}")
    test_df = load_dataset(config.TEST_DATA_PATH)
    test_df['cleaned_text'] = test_df['text'].apply(clean_text) # Assuming 'text' column
    
    # Map string labels in test set to integer IDs using the model's label map
    label2id = model.config.label2id
    test_df.dropna(subset=['label'], inplace=True)
    test_df['label_id'] = test_df['label'].map(label2id)
    
    # Filter out any labels in the test set that the model wasn't trained on
    test_df.dropna(subset=['label_id'], inplace=True)
    test_df['label_id'] = test_df['label_id'].astype(int)
    
    test_queries = test_df['cleaned_text'].tolist()
    test_labels = test_df['label_id'].tolist()
    
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
    logger.info("Analyzing and saving evaluation results...")

    # --- Classification Report ---
    class_names = list(model.config.id2label.values())
    report_str = classification_report(
        test_labels, 
        all_predictions, 
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    logger.info("\n--- Classification Report ---\n" +report_str")
    with open(os.path.join(config.OUTPUT_DIR, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_str)

    # --- Core Metrics ---
    accuracy = accuracy_score(test_labels, all_predictions)
    f1_macro = f1_score(test_labels, all_predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(test_labels, all_predictions, average='weighted', zero_division=0)
    logger.info(f"Overall Accuracy: {accuracy:.4f}")
    logger.info(f"Macro F1-Score: {f1_macro:.4f}")
    logger.info(f"Weighted F1-Score: {f1_weighted:.4f}")

    metrics_summary = {"accuracy": accuracy, "f1_macro": f1_macro, "f1_weighted": f1_weighted}
    with open(os.path.join(config.OUTPUT_DIR, 'metrics_summary.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=4)
        
    # --- Visualizations ---
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    # 1. Confusion Matrix
    cm = confusion_matrix(test_labels, all_predictions)
    plt.figure(figsize=(20, 16)) # Enlarged for better readability with many classes
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix for Multiclass Classification', fontsize=20)
    plt.xlabel('Predicted Label', fontsize=15)
    plt.ylabel('True Label', fontsize=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'confusion_matrix.png'), dpi=300)
    plt.close()
    logger.info("Saved confusion matrix plot.")

    # 2. Per-class F1-Score Bar Plot
    f1_per_class = f1_score(test_labels, all_predictions, average=None, labels=range(len(class_names)), zero_division=0)
    f1_df = pd.DataFrame({'Class': class_names, 'F1-Score': f1_per_class}).sort_values('F1-Score', ascending=False)
    
    plt.figure(figsize=(12, 10))
    sns.barplot(x='F1-Score', y='Class', data=f1_df, palette='viridis')
    plt.title('F1-Score per Class', fontsize=16)
    plt.xlabel('F1-Score', fontsize=12)
    plt.ylabel('Class', fontsize=12)
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'f1_per_class.png'), dpi=300)
    plt.close()
    logger.info("Saved F1-Score per class plot.")

    # --- Error Analysis ---
    test_df['prediction_id'] = all_predictions
    test_df['prediction_label'] = test_df['prediction_id'].map(model.config.id2label)
    test_df['confidence'] = all_confidences
    error_df = test_df[test_df['label_id'] != test_df['prediction_id']]
    error_df_to_save = error_df[['text', 'label', 'prediction_label', 'confidence']]
    
    error_df_to_save.to_csv(os.path.join(config.OUTPUT_DIR, 'error_analysis.csv'), index=False, encoding='utf-8-sig')
    logger.info(f"Saved {len(error_df)} misclassified samples to error_analysis.csv.")
    
    logger.info("Evaluation pipeline for multiclass classifier finished successfully.")

if __name__ == "__main__":
    main()