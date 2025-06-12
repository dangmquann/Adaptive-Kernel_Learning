import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, f1_score,
    roc_curve, classification_report
)

from linformer import LinformerSelfAttention  # <-- Linformer import

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- Linformer-based Node Classifier ----------
class LinformerNodeClassifier(nn.Module):
    def __init__(self, in_dim, embed_dim=128, num_heads=4, num_layers=2, dropout=0.1, seq_len=500, proj_k=64):
        super(LinformerNodeClassifier, self).__init__()

        self.input_proj = nn.Linear(in_dim, embed_dim)

        self.attn_blocks = nn.ModuleList([
            nn.ModuleDict({
                'mha': LinformerSelfAttention(
                    dim=embed_dim,
                    seq_len=seq_len,
                    heads=num_heads,
                    k=proj_k,
                    dropout=dropout
                ),
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
        x = data.x
        N = x.size(0)
        x = self.input_proj(x).unsqueeze(0)  # shape [1, N, D]
        attn_maps = []

        for block in self.attn_blocks:
            attn_out = block['mha'](x)  # [1, N, D]
            x = block['norm1'](x + block['dropout'](attn_out))
            ffn_out = block['ffn'](x)
            x = block['norm2'](x + block['dropout'](ffn_out))

            # Linformer doesn't return attention weights by default
            attn_maps.append(torch.zeros(N, N).to(x.device))  # placeholder

        x = x.squeeze(0)  # [N, D]
        logits = self.classifier(x).squeeze(-1)  # [N]
        return logits, attn_maps

    def get_loss(self, logits, labels, attentions):
        return F.binary_cross_entropy_with_logits(logits, labels.float())


# ---------- Training Function (unchanged) ----------
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
    os.makedirs(checkpoint_dir, exist_ok=True)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    y_test = data.y[test_mask].cpu().numpy()

    seq_len = data.x.shape[0]

    model = LinformerNodeClassifier(
        in_dim=data.num_features,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        seq_len=seq_len
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=1e-6)
    data = data.to(device)

    best_val_loss = float('inf')
    best_loss_model_state = None
    no_improve_count = 0
    history = {'loss': [], 'val_loss': [], 'lr': []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, attns = model(data)
        loss = model.get_loss(logits[train_mask], data.y[train_mask], attns)
        loss.backward()
        optimizer.step()

        if epoch < 50:
            continue

        model.eval()
        with torch.no_grad():
            logits, attns = model(data)
            val_loss = model.get_loss(logits[val_mask], data.y[val_mask], attns).item()

        scheduler.step(val_loss)
        history['loss'].append(loss.item())
        history['val_loss'].append(val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        if logger:
            logger.info(f"Epoch {epoch}/{num_epochs} - LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_val_loss - early_stopping_delta:
            best_val_loss = val_loss
            best_loss_model_state = model.state_dict()
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= early_stopping_patience:
            if logger:
                logger.info("Early stopping")
            break

    if best_loss_model_state:
        model.load_state_dict(best_loss_model_state)

    model.eval()
    with torch.no_grad():
        logits, attns = model(data)
        test_logits = logits[test_mask]
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        fpr, tpr, thresholds = roc_curve(y_test, test_probs)
        youden_index = np.argmax(tpr - fpr)
        optimal_threshold_youden = thresholds[youden_index]
        optimal_threshold_f1 = thresholds[np.argmax([f1_score(y_test, (test_probs > t).astype(int)) for t in thresholds])]
        test_preds = (test_probs > optimal_threshold_f1).astype(int)

    prediction_results = {
        'probabilities': test_probs,
        'predictions': test_preds,
        'attention_maps': attns,
        'metrics': {
            'confusion_matrix': confusion_matrix(y_test, test_preds).tolist(),
            'accuracy': accuracy_score(y_test, test_preds),
            'sensitivity': (confusion_matrix(y_test, test_preds)[1, 1]) / max((confusion_matrix(y_test, test_preds)[1, :].sum()), 1),
            'specificity': (confusion_matrix(y_test, test_preds)[0, 0]) / max((confusion_matrix(y_test, test_preds)[0, :].sum()), 1),
            'auc': roc_auc_score(y_test, test_probs),
            'classification_report': classification_report(y_test, test_preds, output_dict=True),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
        },
        'y_test': y_test,
        'y_pred_prob': test_probs,
    }

    return {
        'model': model,
        'history': history,
        'prediction_results': prediction_results
    }
