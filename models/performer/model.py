import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, f1_score,
    roc_curve, classification_report
)

from performer_pytorch import Performer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class PerformerSelfAttention(nn.Module):
    def __init__(self, dim, depth=1, heads=4, dropout=0.1, nb_features=256):
        super().__init__()
        self.performer = Performer(
            dim=dim,
            dim_head=dim // heads,
            depth=depth,
            heads=heads,
            causal=False,
            nb_features=nb_features,
            ff_dropout=dropout
        )
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        return self.out_proj(self.performer(x))



# ---------- Performer-based Node Classifier ----------
class PerformerNodeClassifier(nn.Module):
    def __init__(self, in_dim, embed_dim=128, num_heads=4, num_layers=2, dropout=0.1, nb_features=256):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, embed_dim)

        self.attn_blocks = nn.ModuleList([
            nn.ModuleDict({
                'mha': PerformerSelfAttention(
                    dim=embed_dim,
                    heads=num_heads,
                    dropout=dropout,
                    nb_features=nb_features
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
        x = self.input_proj(x).unsqueeze(0)
        attn_maps = []

        for block in self.attn_blocks:
            attn_out = block['mha'](x)
            x = block['norm1'](x + block['dropout'](attn_out))
            ffn_out = block['ffn'](x)
            x = block['norm2'](x + block['dropout'](ffn_out))
            attn_maps.append(torch.zeros(N, N).to(x.device))

        x = x.squeeze(0)
        logits = self.classifier(x).squeeze(-1)
        return logits, attn_maps

    def get_loss(self, logits, labels, attentions):
        return F.binary_cross_entropy_with_logits(logits, labels.float())


# ---------- Training Function ----------
def train_test(
    data,
    num_epochs=100,
    lr=1e-3,
    random_seed=42,
    early_stopping_patience=40,
    early_stopping_delta=0.001,
    logger=None,
    embed_dim=128,
    num_heads=4,
    num_layers=2,
    dropout=0.1
):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    y_test = data.y[test_mask].cpu().numpy()

    model = PerformerNodeClassifier(
        in_dim=data.num_features,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=1e-6)
    data = data.to(device)

    best_val_loss = float('inf')
    best_model_state = None
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

        if val_loss < best_val_loss - early_stopping_delta:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= early_stopping_patience:
            break

    if best_model_state:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        logits, attns = model(data)
        test_logits = logits[test_mask]
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        fpr, tpr, thresholds = roc_curve(y_test, test_probs)
        youden_index = np.argmax(tpr - fpr)
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
            'thresholds': thresholds.tolist()
        },
        'y_test': y_test,
        'y_pred_prob': test_probs
    }

    return {
        'model': model,
        'history': history,
        'prediction_results': prediction_results
    }