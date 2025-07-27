# src/shared/model_utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import Trainer, TrainingArguments
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Tuple, Optional

# --- Custom Loss Functions ---

class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss to address class imbalance by focusing on hard-to-classify examples.
    It's a dynamically scaled cross-entropy loss.
    """
    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- Custom Trainer ---

class UltimateTrainer(Trainer):
    """
    A comprehensive custom Trainer that integrates multiple advanced training techniques:
    - R-Drop for regularization.
    - Adversarial Training (FGM) for robustness.
    - Focal Loss for handling class imbalance.
    - Class Weights for another layer of imbalance handling.
    """
    def __init__(
        self,
        *args,
        class_weights: Optional[torch.Tensor] = None,
        rdrop_alpha: float = 0.5,
        adv_alpha: float = 0.5,
        adv_epsilon: float = 1.0,
        focal_loss_gamma: float = 2.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.rdrop_alpha = rdrop_alpha
        self.adv_alpha = adv_alpha
        self.adv_epsilon = adv_epsilon
        
        # Move class_weights to the correct device (e.g., 'cuda') if provided.
        weights = class_weights.to(self.args.device) if class_weights is not None else None
        self.focal_loss_fct = FocalLoss(weight=weights, gamma=focal_loss_gamma)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # During evaluation, just use the standard loss.
        if not model.training:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
            return (loss, outputs) if return_outputs else loss
            
        labels = inputs.get("labels")

        # 1. R-Drop: Forward pass twice with different dropout masks.
        outputs1 = model(**inputs)
        outputs2 = model(**inputs)

        # 2. Calculate average Focal Loss and KL-Divergence for R-Drop.
        loss1 = self.focal_loss_fct(outputs1.logits, labels)
        loss2 = self.focal_loss_fct(outputs2.logits, labels)
        loss_ce_avg = (loss1 + loss2) / 2
        
        kl_loss = F.kl_div(
            F.log_softmax(outputs1.logits, dim=-1),
            F.log_softmax(outputs2.logits, dim=-1),
            reduction='batchmean',
            log_target=True
        )

        # 3. Adversarial Training (FGM) logic
        embedding_layer = model.get_input_embeddings()
        original_embeddings = embedding_layer(inputs['input_ids'])
        original_embeddings.requires_grad_() # Enable gradient computation for embeddings

        # Forward pass to get gradients w.r.t. embeddings
        temp_inputs = {k: v for k, v in inputs.items() if k not in ['input_ids', 'labels']}
        outputs_for_grad = model(inputs_embeds=original_embeddings, labels=labels, **temp_inputs)
        loss_for_grad = self.focal_loss_fct(outputs_for_grad.logits, labels)
        
        # Calculate gradients
        grad = torch.autograd.grad(loss_for_grad, original_embeddings, retain_graph=False)[0]

        # Calculate perturbation (delta)
        delta = self.adv_epsilon * grad / (grad.norm() + 1e-8)
        
        # Calculate adversarial loss with perturbed embeddings
        with torch.no_grad():
            model.eval() # Use eval mode for adversarial forward pass to disable dropout
            adv_outputs = model(inputs_embeds=original_embeddings.detach() + delta, **temp_inputs)
            loss_adv = self.focal_loss_fct(adv_outputs.logits, labels)
            model.train() # Switch back to train mode

        # 4. Combine all losses
        loss = loss_ce_avg + self.rdrop_alpha * kl_loss + self.adv_alpha * loss_adv
        
        return (loss, outputs1) if return_outputs else loss

# --- Utility Functions ---

def compute_metrics(p) -> Dict[str, float]:
    """
    Computes and returns accuracy and macro F1-score for evaluation.

    Args:
        p: A tuple containing predictions and labels from the model.

    Returns:
        A dictionary with "accuracy" and "f1" scores.
    """

    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    return {"accuracy": acc, "f1": f1}

def create_optimizer_and_scheduler(
    model: nn.Module, 
    training_args: TrainingArguments, 
    num_training_steps: int
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """
    Creates an AdamW optimizer with differential learning rates and a linear warmup scheduler.
    - Body (pre-trained layers) gets a smaller learning rate.
    - Head (classifier/LoRA adapters) gets a larger learning rate.

    Args:
        model: The model for which to create the optimizer.
        training_args: The TrainingArguments object containing hyperparameters.
        num_training_steps: The total number of training steps.

    Returns:
        A tuple containing the configured optimizer and learning rate scheduler.
    """
    # Define parameters that should not have weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    
    # differential learning rates for different parts of the model
    body_lr = 2e-6  # A very small learning rate for the stable, pre-trained body
    head_lr = 1e-4  # A larger learning rate for the new/adapter layers
    
    optimizer_grouped_parameters = [
        # Parameters for the main body of the model (excluding no_decay ones)
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and "classifier" not in n and "lora" not in n],
            'weight_decay': training_args.weight_decay,
            'lr': body_lr
        },
        # Parameters for the main body of the model (only no_decay ones)
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and "classifier" not in n and "lora" not in n],
            'weight_decay': 0.0,
            'lr': body_lr
        },
        # Parameters for the classifier head and LoRA adapters
        {
            'params': [p for n, p in model.named_parameters() if "classifier" in n or "lora" in n],
            'weight_decay': training_args.weight_decay,
            'lr': head_lr
        }
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters)
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * training_args.warmup_ratio),
        num_training_steps=num_training_steps
    )
    
    return optimizer, lr_scheduler