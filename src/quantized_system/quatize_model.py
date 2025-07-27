# src/optimization/quantize_models.py

import os
import sys
import logging
import json
import time
import shutil
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from peft import PeftModel
from sklearn.metrics import f1_score
from datasets import Dataset

# --- 1. Import Shared Modules & Optimum Libraries ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.shared.data_utils import load_dataset, clean_text

from optimum.onnxruntime import ORTQuantizer, ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)


# --- 2. Configuration Class ---
class OptimizationConfig:
    # --- Input Paths ---
    BINARY_MODEL_DIR = r'[model\best_model_fold_1]'
    MULTICLASS_MODEL_DIR = r'[model\best_model]'
    BASE_MODEL_NAME = "klue/roberta-large"
    
    # --- Data Paths ---
    BINARY_TEST_DATA_PATH = r'[data\custom_testset]'
    MULTICLASS_TEST_DATA_PATH = r'[data\multiclass_testset.csv]'
    
    # --- Output Paths ---
    OUTPUT_BASE_DIR = r'[quantized_models]'
    REPORT_DIR = r'[optimization_report]'
    
    # --- Optimization Parameters ---
    CALIBRATION_SAMPLE_SIZE = 100 # Number of samples for PTQ calibration
    INFERENCE_REPEATS = 50 # Number of times to repeat inference for stable latency measurement

# --- 3. Helper Functions ---

def get_model_size_mb(path):
    """Calculates the size of a model directory in MB."""
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

def measure_latency(model, tokenizer, texts, repeats):
    """Measures the average inference latency for a batch of texts."""
    latencies = []
    for _ in range(repeats):
        start_time = time.time()
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        _ = model(**inputs)
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000) # in ms
    return np.mean(latencies)

def evaluate_f1(model, tokenizer, df, text_col, label_col, label_map=None):
    """Evaluates the Macro F1-score of a model on a dataframe."""
    texts = df[text_col].tolist()
    true_labels = df[label_col].tolist()
    if label_map:
        true_labels = [label_map.get(lbl) for lbl in true_labels]

    preds = []
    for text in tqdm(texts, desc="Evaluating F1-Score"):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        if 'token_type_ids' in inputs and isinstance(model, ORTModelForSequenceClassification):
            del inputs['token_type_ids'] # ONNX models might not accept this
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
            
        preds.append(torch.argmax(logits, dim=-1).item())
        
    return f1_score(true_labels, preds, average="macro", zero_division=0)


# --- 4. Main Optimization Pipeline Function ---

def run_optimization_pipeline(config: OptimizationConfig, model_type: str):
    """
    Runs the full optimization and evaluation pipeline for a given model type.
    """
    logger.info("\n" + "="*60)
    logger.info(f"STARTING OPTIMIZATION PIPELINE FOR: {model_type.upper()} CLASSIFIER")
    logger.info("="*60)

    # --- Setup paths and data based on model type ---
    if model_type == 'binary':
        pytorch_model_dir = config.BINARY_MODEL_DIR
        test_data_path = config.BINARY_TEST_DATA_PATH
        text_col, label_col = '용역명', 'label'
        label_map = None
    elif model_type == 'multiclass':
        pytorch_model_dir = config.MULTICLASS_MODEL_DIR
        test_data_path = config.MULTICLASS_TEST_DATA_PATH
        text_col, label_col = 'text', 'label'
        model_config = AutoConfig.from_pretrained(pytorch_model_dir)
        label_map = model_config.label2id
    else:
        raise ValueError("model_type must be 'binary' or 'multiclass'")

    onnx_fp32_dir = os.path.join(config.OUTPUT_BASE_DIR, f"{model_type}_onnx_fp32")
    onnx_int8_dir = os.path.join(config.OUTPUT_BASE_DIR, f"{model_type}_onnx_int8")

    # --- Load Data ---
    test_df = load_dataset(test_data_path)
    test_df[text_col] = test_df[text_col].apply(clean_text)
    
    # --- Step 1: Load original PyTorch PEFT model and merge ---
    logger.info("Step 1: Loading and merging original PyTorch PEFT model...")
    base_model_config = AutoConfig.from_pretrained(pytorch_model_dir)
    base_model = AutoModelForSequenceClassification.from_pretrained(config.BASE_MODEL_NAME, config=base_model_config)
    fp32_model = PeftModel.from_pretrained(base_model, pytorch_model_dir).merge_and_unload()
    fp32_tokenizer = AutoTokenizer.from_pretrained(pytorch_model_dir)

    # --- Step 2: Evaluate FP32 PyTorch model (Baseline) ---
    logger.info("Step 2: Evaluating FP32 PyTorch model (Baseline)...")
    baseline_f1 = evaluate_f1(fp32_model, fp32_tokenizer, test_df, text_col, label_col, label_map)
    baseline_size = get_model_size_mb(pytorch_model_dir) # Use original dir size as it's what we load
    baseline_latency = measure_latency(fp32_model, fp32_tokenizer, test_df[text_col].tolist()[:10], config.INFERENCE_REPEATS)
    
    results = {
        'FP32 PyTorch': {
            'f1': baseline_f1,
            'size_mb': baseline_size,
            'latency_ms': baseline_latency
        }
    }
    logger.info(f"  -> Baseline F1: {baseline_f1:.4f}, Size: {baseline_size:.2f} MB, Latency: {baseline_latency:.2f} ms")

    # --- Step 3: Convert to ONNX FP32 ---
    logger.info("Step 3: Converting to ONNX FP32 format...")
    if os.path.exists(onnx_fp32_dir): shutil.rmtree(onnx_fp32_dir)
    onnx_model = ORTModelForSequenceClassification.from_pretrained(pytorch_model_dir, export=True, from_transformers=True)
    onnx_model.save_pretrained(onnx_fp32_dir)
    fp32_tokenizer.save_pretrained(onnx_fp32_dir) # Save tokenizer for consistency

    # --- Step 4: Quantize to ONNX INT8 with Calibration ---
    logger.info("Step 4: Quantizing to ONNX INT8 with Post-Training Quantization...")
    if os.path.exists(onnx_int8_dir): shutil.rmtree(onnx_int8_dir)

    # Create calibration dataset
    calibration_texts = test_df[text_col].sample(n=config.CALIBRATION_SAMPLE_SIZE, random_state=42).tolist()
    calibration_dataset = Dataset.from_dict(fp32_tokenizer(calibration_texts, padding=True, truncation=True))

    quantizer = ORTQuantizer.from_pretrained(onnx_fp32_dir)
    dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=True, per_channel=False)
    
    quantizer.quantize(
        save_dir=onnx_int8_dir,
        quantization_config=dqconfig,
        calibration_dataset=calibration_dataset
    )
    fp32_tokenizer.save_pretrained(onnx_int8_dir)
    logger.info("  -> Quantization complete.")

    # --- Step 5: Evaluate ONNX INT8 model ---
    logger.info("Step 5: Evaluating quantized ONNX INT8 model...")
    int8_model = ORTModelForSequenceClassification.from_pretrained(onnx_int8_dir, provider="CPUExecutionProvider")
    int8_tokenizer = AutoTokenizer.from_pretrained(onnx_int8_dir)

    int8_f1 = evaluate_f1(int8_model, int8_tokenizer, test_df, text_col, label_col, label_map)
    int8_size = get_model_size_mb(onnx_int8_dir)
    int8_latency = measure_latency(int8_model, int8_tokenizer, test_df[text_col].tolist()[:10], config.INFERENCE_REPEATS)

    results['INT8 ONNX'] = {
        'f1': int8_f1,
        'size_mb': int8_size,
        'latency_ms': int8_latency
    }
    logger.info(f"  -> INT8 F1: {int8_f1:.4f}, Size: {int8_size:.2f} MB, Latency: {int8_latency:.2f} ms")
    
    return results


# --- 5. Main Execution Block ---
def main():
    config = OptimizationConfig()
    os.makedirs(config.REPORT_DIR, exist_ok=True)

    # Run pipeline for both models
    binary_results = run_optimization_pipeline(config, 'binary')
    multiclass_results = run_optimization_pipeline(config, 'multiclass')

    # --- Consolidate and display final results ---
    binary_df = pd.DataFrame(binary_results).T.reset_index().rename(columns={'index': 'Format'})
    binary_df['Model'] = 'Binary Classifier'
    multiclass_df = pd.DataFrame(multiclass_results).T.reset_index().rename(columns={'index': 'Format'})
    multiclass_df['Model'] = 'Multiclass Classifier'
    
    final_df = pd.concat([binary_df, multiclass_df])
    
    logger.info("\n\n" + "="*60)
    logger.info("FINAL OPTIMIZATION SUMMARY")
    logger.info("="*60)
    print(final_df.to_markdown(index=False))

    final_df.to_csv(os.path.join(config.REPORT_DIR, 'optimization_summary.csv'), index=False, encoding='utf-8-sig')

    # --- Generate Visualization ---
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle('Model Optimization Performance Comparison', fontsize=20)
    
    # Plot 1: F1-Score
    sns.barplot(data=final_df, x='Model', y='f1', hue='Format', ax=axes[0], palette='coolwarm')
    axes[0].set_title('F1-Score Comparison', fontsize=16)
    axes[0].set_ylabel('Macro F1-Score', fontsize=12)
    axes[0].set_ylim(bottom=max(0, final_df['f1'].min() - 0.01), top=final_df['f1'].max() + 0.01)

    # Plot 2: Model Size
    sns.barplot(data=final_df, x='Model', y='size_mb', hue='Format', ax=axes[1], palette='Greens')
    axes[1].set_title('Model Size Comparison', fontsize=16)
    axes[1].set_ylabel('Size (MB)', fontsize=12)

    # Plot 3: Latency
    sns.barplot(data=final_df, x='Model', y='latency_ms', hue='Format', ax=axes[2], palette='Blues')
    axes[2].set_title('Inference Latency Comparison (CPU)', fontsize=16)
    axes[2].set_ylabel('Latency (ms)', fontsize=12)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(config.REPORT_DIR, 'optimization_comparison_plot.png'), dpi=300)
    
    logger.info(f"Final report and plot saved to: {config.REPORT_DIR}")


if __name__ == "__main__":
    main()