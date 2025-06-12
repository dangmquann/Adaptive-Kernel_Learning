import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, f1_score
)
from sklearn.model_selection import train_test_split
import gpytorch
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import VariationalELBO
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as F
from models.DKL.model.DKL import FeatureExtractor, DKLModel


def train_test_DKL(X_train_full, Y_train_full, X_test, Y_test,
                   epochs=100, batch_size=32, lr=0.01,
                patience=15, delta=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === 1. Split train into train/val ===
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_full, Y_train_full, test_size=0.2, stratify=Y_train_full, random_state=42
    )

    # === 2. Convert to tensor ===
    X_train = torch.from_numpy(X_train.astype(np.float32)).to(device)
    X_val = torch.from_numpy(X_val.astype(np.float32)).to(device)
    X_test = torch.from_numpy(X_test.astype(np.float32)).to(device)

    Y_train = torch.from_numpy(np.array(Y_train).astype(np.float32)).squeeze().to(device)
    Y_val = torch.from_numpy(np.array(Y_val).astype(np.float32)).squeeze().to(device)
    Y_test_tensor = torch.from_numpy(np.array(Y_test).astype(np.float32)).squeeze().to(device)

    # === 3. Build model ===
    feat_extractor = FeatureExtractor(input_dim=X_train.size(1))
    inducing_points = X_train[:min(500, len(X_train))]
    model = DKLModel(feat_extractor, inducing_points)
    likelihood = BernoulliLikelihood()

    # Convert model to double precision
    model = model.double().to(device)
    likelihood = likelihood.double().to(device)
    X_train = X_train.double()
    X_val = X_val.double()
    X_test = X_test.double()
    Y_train = Y_train.double()
    Y_val = Y_val.double()
    Y_test_tensor = Y_test_tensor.double()

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()), lr=lr
    )
    mll = VariationalELBO(likelihood, model, num_data=Y_train.size(0))

    # === 4. Training loop ===
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    best_val_loss = float('inf')
    best_model_state = None
    no_improve_counter = 0

    for epoch in range(epochs):
        model.train()
        likelihood.train()
        epoch_loss = 0.
        for xb, yb in train_loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = -mll(output, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # === 5. Evaluate on validation set ===
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            val_preds = likelihood(model(X_val))
            val_probs = val_preds.mean
            val_loss = F.binary_cross_entropy(val_probs, Y_val)

        # === 6. Print and check early stopping ===
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.3f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss - delta:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            no_improve_counter = 0
            print(f"✅ New best validation loss: {val_loss:.4f} at epoch {epoch+1}")
        else:
            no_improve_counter += 1

        if no_improve_counter > patience:
            print("⏹️ Early stopping triggered.")
            break

    # === 7. Final evaluation on test set ===
    model.load_state_dict(best_model_state)
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(X_test))
        test_probs = preds.mean.cpu().numpy()
        Y_test_np = Y_test_tensor.cpu().numpy()

        fpr, tpr, thresholds = roc_curve(Y_test_np, test_probs)
        youden_index = np.argmax(tpr - fpr)
        optimal_threshold_youden = thresholds[youden_index]
        f1_scores = [f1_score(Y_test_np, (test_probs > t).astype(int)) for t in thresholds]
        optimal_threshold_f1 = thresholds[np.argmax(f1_scores)]

        print(f"Optimal threshold (Youden's J): {optimal_threshold_youden:.4f}")
        print(f"Optimal threshold (F1 score): {optimal_threshold_f1:.4f}")

        test_preds = (test_probs > 0.5).astype(int)

        auc = roc_auc_score(Y_test_np, test_probs)
        acc = accuracy_score(Y_test_np, test_preds)
        cf = confusion_matrix(Y_test_np, test_preds)
        tn, fp, fn, tp = cf.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        cr = classification_report(Y_test_np, test_preds, output_dict=True)

        print(f" AUC={auc:.4f}, Acc={acc:.4f}, Sens={sens:.4f}, Spec={spec:.4f}")

        results = {
            'prediction_results': {
                'metrics': {
                    'accuracy': acc,
                    'auc': auc,
                    'sensitivity': sens,
                    'specificity': spec,
                    'confusion_matrix': cf.tolist(),
                    'classification_report': cr,
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': thresholds.tolist(),
                    'optimal_threshold_youden': float(optimal_threshold_youden),
                    'optimal_threshold_f1': float(optimal_threshold_f1)
                },
                'probabilities': test_probs.tolist(),
                'predictions': test_preds.tolist(),
                'y_pred_prob': test_probs.tolist(),
                'y_test': Y_test_np.tolist(),
            }
        }

    return results
