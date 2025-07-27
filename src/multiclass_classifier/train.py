# src/multiclass_classifier/train.py

import os
import sys
import logging
import json
import shutil
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import StratifiedKFold

# --- 1. Import Shared Modules & Configure Environment ---
# Add the project root to the Python path to allow importing from `src`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.shared.data_utils import load_dataset, calculate_class_weights
from src.shared.model_utils import UltimateTrainer, compute_metrics, create_optimizer_and_scheduler

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- 2. Configuration & Paths ---
class TrainConfig:
    # --- Paths (MODIFIED FOR MULTICLASS) ---
    DATA_PATH = r'[data\multiclass_data.csv]' # Path to your multiclass labeled data
    OUTPUT_BASE_DIR = r'[multiclass_classifier_output]'
    
    # --- Model & Tokenizer ---
    MODEL_NAME = "klue/roberta-large"
    MAX_LENGTH = 128
    
    # --- Training Parameters ---
    NUM_EPOCHS = 10 # Multiclass tasks might benefit from more epochs
    TRAIN_BATCH_SIZE = 16
    EVAL_BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 2
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    FP16 = True
    N_SPLITS = 5
    RANDOM_STATE = 42
    
    # --- LoRA Parameters (reusable) ---
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = ["query", "value"]

    # --- UltimateTrainer Parameters (reusable) ---
    RDROP_ALPHA = 0.5
    ADV_ALPHA = 0.5
    FOCAL_LOSS_GAMMA = 2.0

def main():
    """Main training pipeline for the multi-class classifier."""
    config = TrainConfig()
    
    # --- Create output directories ---
    results_dir = os.path.join(config.OUTPUT_BASE_DIR, 'results')
    model_save_dir = os.path.join(config.OUTPUT_BASE_DIR, 'model')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # --- 3. Load and Prepare Data (MODIFIED FOR MULTICLASS) ---
    logger.info("Step 1: Loading and preparing multiclass data...")
    df = load_dataset(config.DATA_PATH)
    
    # Drop rows with missing labels or text, which can cause errors
    df.dropna(subset=['text', 'label'], inplace=True)

    # --- Create Label Mappings ---
    # Convert string labels to integer IDs for the model
    unique_labels = sorted(df['label'].unique())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}
    num_labels = len(unique_labels)
    df['label'] = df['label'].map(label2id)
    
    logger.info(f"Found {num_labels} unique classes for classification.")
    
    # Save the label map for later use in evaluation and API
    label_map_path = os.path.join(config.OUTPUT_BASE_DIR, 'label_map.json')
    with open(label_map_path, 'w', encoding='utf-8') as f:
        json.dump({'label2id': label2id, 'id2label': id2label}, f, ensure_ascii=False, indent=4)
    logger.info(f"Label map saved to: {label_map_path}")

    class_weights = calculate_class_weights(df['label'])
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=config.MAX_LENGTH)

    # --- 4. Cross-Validation Training Loop (Structure is identical) ---
    logger.info(f"Step 2: Starting {config.N_SPLITS}-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    
    all_eval_results = []
    best_f1_overall = -1.0
    path_of_best_model_so_far = ""

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
        logger.info(f"===== FOLD {fold+1}/{config.N_SPLITS} =====")
        
        train_df = df.iloc[train_idx]
        eval_df = df.iloc[val_idx]
        train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
        eval_dataset = Dataset.from_pandas(eval_df).map(tokenize_function, batched=True)

        # --- 5. Initialize Model (MODIFIED FOR MULTICLASS) ---
        logger.info("Initializing multiclass model with PEFT...")
        
        # Here we provide num_labels and the label mappings to the model config
        model = AutoModelForSequenceClassification.from_pretrained(
            config.MODEL_NAME, 
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True # Important when loading a binary pre-trained head
        )
        lora_config = LoraConfig(
            r=config.LORA_R, lora_alpha=config.LORA_ALPHA, target_modules=config.LORA_TARGET_MODULES,
            lora_dropout=config.LORA_DROPOUT, bias="none", task_type="SEQ_CLS"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # The rest of the setup is IDENTICAL to the binary classifier's script
        training_args = TrainingArguments(
            output_dir=os.path.join(results_dir, f'fold_{fold+1}'),
            num_train_epochs=config.NUM_EPOCHS,
            per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
            gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
            warmup_ratio=config.WARMUP_RATIO,
            weight_decay=config.WEIGHT_DECAY,
            fp16=config.FP16,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=1,
            logging_strategy="epoch",
            report_to="none",
        )
        
        num_training_steps = (len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)) * training_args.num_train_epochs
        optimizer, lr_scheduler = create_optimizer_and_scheduler(model, training_args, num_training_steps)
        
        trainer = UltimateTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, lr_scheduler),
            class_weights=class_weights,
            rdrop_alpha=config.RDROP_ALPHA,
            adv_alpha=config.ADV_ALPHA,
            focal_loss_gamma=config.FOCAL_LOSS_GAMMA
        )
        
        logger.info("Starting training for the current fold...")
        trainer.train()

        logger.info("Evaluating the best model of the fold...")
        eval_results = trainer.evaluate()
        logger.info(f"Fold {fold+1} Evaluation Results: {eval_results}")
        all_eval_results.append(eval_results)
        
        current_fold_f1 = eval_results['eval_f1']
        if current_fold_f1 > best_f1_overall:
            best_f1_overall = current_fold_f1
            
            # Remove the previous best model directory to save disk space
            if path_of_best_model_so_far and os.path.exists(path_of_best_model_so_far):
                logger.info(f"Removing old best model directory: {path_of_best_model_so_far}")
                shutil.rmtree(path_of_best_model_so_far, ignore_errors=True)

            # Define the path for the new best model
            new_best_model_path = os.path.join(model_save_dir, f'best_model') # Keep only one best model
            path_of_best_model_so_far = new_best_model_path
            
            logger.info(f"*** New overall best model found! (Fold {fold+1}) Saving to: {new_best_model_path} (F1: {best_f1_overall:.4f}) ***")
            trainer.save_model(new_best_model_path)
            tokenizer.save_pretrained(new_best_model_path)

    # --- 8. Final Results Summary ---
    logger.info("\n" + "="*50)
    logger.info("Final Cross-Validation Results Summary")
    logger.info("="*50)
    
    avg_accuracy = np.mean([res['eval_accuracy'] for res in all_eval_results])
    avg_f1 = np.mean([res['eval_f1'] for res in all_eval_results])
    std_f1 = np.std([res['eval_f1'] for res in all_eval_results])

    logger.info(f"Average Accuracy: {avg_accuracy:.4f}")
    logger.info(f"Average Macro F1-Score: {avg_f1:.4f} (Std Dev: {std_f1:.4f})")
    
    final_summary = {
        "average_accuracy": avg_accuracy,
        "average_f1_macro": avg_f1,
        "f1_std": std_f1,
        "best_model_path": path_of_best_model_so_far,
        "all_fold_results": all_eval_results
    }
    
    summary_path = os.path.join(results_dir, 'final_cv_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, indent=4)
        
    logger.info(f"Final summary saved to: {summary_path}")
    logger.info(f"The overall best performing multiclass model is saved at: {path_of_best_model_so_far}")

if __name__ == "__main__":
    main()