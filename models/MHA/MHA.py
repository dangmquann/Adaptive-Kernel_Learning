import os
# import logging
from datetime import datetime

import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,f1_score,roc_curve,
    classification_report
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_test(
        data,
        num_epochs=100,
        lr=1e-3,
        random_seed=42,
        checkpoint_dir="checkpoints",
        early_stopping_patience=40,
        early_stopping_delta=0.001, logger=None,
        embed_dim=128, num_heads=4, num_layers=2, dropout=0.1,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(checkpoint_dir, exist_ok=True)
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')


    # Reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    y_test = data.y[test_mask].cpu().numpy()
    # Initialize model and optimizer
    model = MHANodeClassifier(
        in_dim=data.num_features,
        embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [], 'lr': []}

    best_val_loss = float('inf')
    best_loss_epoch = 0
    best_loss_model_state = None

    model = model.to(device)
    data = data.to(device)

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits, attentions = model(data)
        loss = model.get_loss(logits[train_mask], data.y[train_mask], attentions)
        loss.backward()
        optimizer.step()

        if epoch < 50:
            continue

        model.eval()
        with torch.no_grad():
            logits, attentions = model(data)
            val_loss = model.get_loss(logits[val_mask], data.y[val_mask], attentions).item()

        current_lr = optimizer.param_groups[0]['lr']
        history['loss'].append(loss.item())
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        logger.info(
            f"Epoch {epoch}/{num_epochs} - "
            f"LR: {current_lr:.6f}"
        )

        scheduler.step(val_loss)

        loss_improved = val_loss < best_val_loss - early_stopping_delta
        if loss_improved:
            best_val_loss = val_loss
            best_loss_epoch = epoch
            best_loss_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }
            logger.info(f"Saved checkpoint with best val_loss: {best_val_loss:.4f} at epoch {best_loss_epoch}")
            no_improve_count = 0
        else:
            no_improve_count += 1
            logger.info(f"No val_loss improvement for {no_improve_count} epochs (best loss: {best_val_loss:.4f})")



        if no_improve_count >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch} epochs due to no val_loss improvement")
            break

    logger.info("Training completed!")

    if best_loss_model_state is not None:
        model.load_state_dict(best_loss_model_state['model_state_dict'])
        logger.info(f"Loaded best loss model from epoch {best_loss_epoch} with val_loss {best_val_loss:.4f}")

    best_models_state = {
        'best_loss': {
            'epoch': best_loss_epoch,
            'val_loss': best_val_loss,
            'model_state': best_loss_model_state
        }
    }

    result = {
        'best_models': {
            'best_loss_model': model
        },
        'history': history,
        'best_models_state': best_models_state
    }

    model.eval()
    with torch.no_grad():
        logits, attentions = model(data)
        test_logits = logits[test_mask]
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        fpr, tpr, thresholds = roc_curve(y_test, test_probs)
        youden_index = np.argmax(tpr - fpr)
        optimal_threshold_youden = thresholds[youden_index]
        f1_scores = [f1_score(y_test, (test_probs > t).astype(int)) for t in thresholds]
        optimal_threshold_f1 = thresholds[np.argmax(f1_scores)]
        print(f"Optimal threshold (Youden's J): {optimal_threshold_youden}")
        print(f"Optimal threshold (F1 score): {optimal_threshold_f1}")
        test_preds = (test_probs > optimal_threshold_f1).astype(int)

        attention_maps = []
        for layer_attn in attentions:
            if len(layer_attn.shape) > 2:
                layer_attn = layer_attn.mean(dim=0)
            test_attends_to_all = layer_attn[test_mask, :].cpu().numpy()
            test_attends_to_test = layer_attn[test_mask, :][:, test_mask].cpu().numpy()
            attention_maps.append({
                'test_attends_to_all': test_attends_to_all,
                'test_attends_to_test': test_attends_to_test
            })

    prediction_results = {
        'probabilities': test_probs,
        'predictions': test_preds,
        'attention_maps': attention_maps
    }

    if y_test is not None:
        accuracy = accuracy_score(y_test, test_preds)
        cm = confusion_matrix(y_test, test_preds)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        auc = roc_auc_score(y_test, test_probs)

        prediction_results['metrics'] = {
            'confusion_matrix': cm.tolist(),
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'auc': auc,
            'classification_report': classification_report(y_test, test_preds, output_dict=True),
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
        prediction_results['y_test'] = y_test
        prediction_results['y_pred_prob'] = test_probs

        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info(f"Test sensitivity: {sensitivity:.4f}")
        logger.info(f"Test specificity: {specificity:.4f}")
        logger.info(f"Test AUC: {auc:.4f}")
        logger.info(f"Confusion matrix:\n{cm}")

    result['prediction_results'] = prediction_results
    return result


class MHANodeClassifier(nn.Module):
    def __init__(self, in_dim, embed_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        """
        Simple transformer-style encoder using nn.MultiheadAttention
        for node feature encoding.

        Args:
            in_dim: input feature dimension per node
            embed_dim: hidden/embedding dimension
            num_heads: number of attention heads
            num_layers: number of stacked attention blocks
            dropout: dropout rate
        """
        super(MHANodeClassifier, self).__init__()

        self.input_proj = nn.Linear(in_dim, embed_dim)
        self.attn_blocks = nn.ModuleList([
            nn.ModuleDict({
                'mha': nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True),
                'norm1': nn.LayerNorm(embed_dim),
                'ffn': nn.Sequential(
                    nn.Linear(embed_dim, 4 * embed_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(4 * embed_dim, embed_dim)
                ),
                'norm2': nn.LayerNorm(embed_dim),
                'dropout': nn.Dropout(dropout)
            })
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        """
        Args:
            data: [N, D] node features
        Returns:
            logits: [N] classification output
            all_attn_weights: list of attention maps from each layer
        """
        x = data.x
        N = x.size(0)

        # [N, D] -> [1, N, D] for batch_first MultiheadAttention
        x = self.input_proj(x).unsqueeze(0)

        attn_maps = []

        for block in self.attn_blocks:
            attn_out, attn_weights = block['mha'](x, x, x)  # self-attention
            x = block['norm1'](x + block['dropout'](attn_out))

            ffn_out = block['ffn'](x)
            x = block['norm2'](x + block['dropout'](ffn_out))

            attn_maps.append(attn_weights.squeeze(0))  # [N, N] per head

        # Classification head
        x = x.squeeze(0)  # [N, D]
        logits = self.classifier(x).squeeze(-1)  # [N]

        return logits, attn_maps

    def get_loss(self, logits, labels, attentions):
        """
        Compute loss with optional KL regularization between attention layers

        Args:
            logits: Predicted logits [num_nodes]
            labels: Target labels [num_nodes]
            attentions: List of attention tensors from each layer
        """
        # Binary classification loss
        bce = F.binary_cross_entropy_with_logits(logits, labels.float())
        loss = bce
        return loss