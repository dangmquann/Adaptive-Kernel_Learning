import os
import logging
import numpy as np
import torch
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)


import torch
from torch.nn import functional as F

import numpy as np
import logging
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, f1_score
)

import os
import numpy as np
import torch
import torch.nn.functional as F

cuda = True if torch.cuda.is_available() else False


def cal_sample_weight(labels, num_class, use_sample_weight=True):
    if not use_sample_weight:
        return np.ones(len(labels)) / len(labels)
    count = np.zeros(num_class)
    for i in range(num_class):
        count[i] = np.sum(labels == i)
    sample_weight = np.zeros(labels.shape)
    for i in range(num_class):
        sample_weight[np.where(labels == i)[0]] = count[i] / np.sum(count)

    return sample_weight


def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1, 1), 1)

    return y_onehot


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def to_sparse(x):
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)
    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def cal_adj_mat_parameter(edge_per_node, data, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    parameter = torch.sort(dist.reshape(-1, )).values[edge_per_node * data.shape[0]]
    return parameter.data.cpu().numpy().item()


def graph_from_dist_tensor(dist, parameter, self_dist=True):
    if self_dist:
        assert dist.shape[0] == dist.shape[1], "Input is not pairwise dist matrix"
    g = (dist <= parameter).float()
    if self_dist:
        diag_idx = np.diag_indices(g.shape[0])
        g[diag_idx[0], diag_idx[1]] = 0

    return g


def gen_adj_mat_tensor(data, parameter, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    g = graph_from_dist_tensor(dist, parameter, self_dist=True)
    if metric == "cosine":
        adj = 1 - dist
    else:
        raise NotImplementedError
    adj = adj * g
    adj_T = adj.transpose(0, 1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T * (adj_T > adj).float() - adj * (adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    adj = to_sparse(adj)

    return adj


def gen_test_adj_mat_tensor(data, trte_idx, parameter, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    adj = torch.zeros((data.shape[0], data.shape[0]))
    if cuda:
        adj = adj.cuda()
    num_tr = len(trte_idx["tr"])

    dist_tr2te = cosine_distance_torch(data[trte_idx["tr"]], data[trte_idx["te"]])
    g_tr2te = graph_from_dist_tensor(dist_tr2te, parameter, self_dist=False)
    if metric == "cosine":
        adj[:num_tr, num_tr:] = 1 - dist_tr2te
    else:
        raise NotImplementedError
    adj[:num_tr, num_tr:] = adj[:num_tr, num_tr:] * g_tr2te

    dist_te2tr = cosine_distance_torch(data[trte_idx["te"]], data[trte_idx["tr"]])
    g_te2tr = graph_from_dist_tensor(dist_te2tr, parameter, self_dist=False)
    if metric == "cosine":
        adj[num_tr:, :num_tr] = 1 - dist_te2tr
    else:
        raise NotImplementedError
    adj[num_tr:, :num_tr] = adj[num_tr:, :num_tr] * g_te2tr  # retain selected edges

    adj_T = adj.transpose(0, 1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T * (adj_T > adj).float() - adj * (adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    adj = to_sparse(adj)

    return adj


def save_model_dict(folder, model_dict):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for module in model_dict:
        torch.save(model_dict[module].state_dict(), os.path.join(folder, module + ".pth"))


def load_model_dict(folder, model_dict):
    for module in model_dict:
        if os.path.exists(os.path.join(folder, module + ".pth")):
            #            print("Module {:} loaded!".format(module))
            model_dict[module].load_state_dict(torch.load(os.path.join(folder, module + ".pth"),
                                                          map_location="cuda:{:}".format(torch.cuda.current_device())))
        else:
            print("WARNING: Module {:} from model_dict is not loaded!".format(module))
        if cuda:
            model_dict[module].cuda()
    return model_dict


def cal_sample_weight(labels, num_class, use_sample_weight=True):
    if not use_sample_weight:
        return np.ones(len(labels)) / len(labels)
    count = np.zeros(num_class)
    for i in range(num_class):
        count[i] = np.sum(labels == i)
    sample_weight = np.zeros(labels.shape)
    for i in range(num_class):
        sample_weight[np.where(labels == i)[0]] = count[i] / np.sum(count)
    return sample_weight


def train_test_MOGONET(
    data_list,
    model_dict,
    optimizer_dict,
    num_epochs=100,
    lr=1e-3,
    random_seed=42,
    early_stopping_patience=40,
    early_stopping_delta=0.001,
    logger=None,
    device=None,
):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger.info("MOGONET Training with Multimodal Graph Data")

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    label = data_list[0].y
    train_mask = data_list[0].train_mask
    val_mask = data_list[0].val_mask
    test_mask = data_list[0].test_mask
    y_test = label[test_mask].cpu().numpy()
    num_classes = int(torch.max(label).item() + 1)

    result = {
        'metrics': {},
        'predictions': None,
        'training_history': None,
        'best_model_state': None
    }

    onehot_labels_tr_tensor = F.one_hot(label[train_mask].long(), num_classes=num_classes).float().to(device)
    sample_weight_tr = cal_sample_weight(label[train_mask].cpu().numpy(), num_classes)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr).to(device)

    best_val_loss = float('inf')
    best_model = None
    no_improve_count = 0
    history = {'train_loss': [], 'val_loss': []}

    # ===== Pretrain GCNs (without VCDN) =====
    logger.info("Pretraining GCNs...")

    for epoch in range(500):
        for m in model_dict:
            model_dict[m].train()
        for i in range(len(data_list)):
            optimizer_dict[f"C{i + 1}"].zero_grad()
            ci_loss = 0
            logits = model_dict[f"C{i + 1}"](model_dict[f"E{i + 1}"](
                data_list[i].x[train_mask],
                data_list[i].prior_guide[train_mask][:, train_mask]
            ))
            ci_loss = torch.mean(F.cross_entropy(logits, label[train_mask].long(), reduction='none') * sample_weight_tr)
            ci_loss.backward()
            optimizer_dict[f"C{i + 1}"].step()

    # ===== Main Training with VCDN =====
    for epoch in range(1, num_epochs + 1):
        for m in model_dict:
            model_dict[m].train()

        for i in range(len(data_list)):
            optimizer_dict[f"C{i + 1}"].zero_grad()
            ci_loss = 0
            logits = model_dict[f"C{i + 1}"](model_dict[f"E{i + 1}"](
                data_list[i].x[train_mask],
                data_list[i].prior_guide[train_mask][:, train_mask]
            ))
            ci_loss = torch.mean(F.cross_entropy(logits, label[train_mask].long(), reduction='none') * sample_weight_tr)
            ci_loss.backward()
            optimizer_dict[f"C{i + 1}"].step()

        optimizer_dict["C"].zero_grad()
        c_loss = 0
        ci_list = []
        pred_list = [
            model_dict[f"C{i + 1}"](model_dict[f"E{i + 1}"](
                data_list[i].x[train_mask],
                data_list[i].prior_guide[train_mask][:, train_mask]))
            for i in range(len(data_list))
        ]
        combined_pred = model_dict["C"](pred_list)
        train_loss = torch.mean(F.cross_entropy(combined_pred, label[train_mask].long(), reduction='none') * sample_weight_tr)
        train_loss.backward()
        optimizer_dict["C"].step()
        history['train_loss'].append(train_loss.item())

        # Validation
        with torch.no_grad():
            val_pred_list = [
                model_dict[f"C{i + 1}"](model_dict[f"E{i + 1}"](
                    data_list[i].x[val_mask],
                    data_list[i].prior_guide[val_mask][:, val_mask]))
                for i in range(len(data_list))
            ]
            val_combined_pred = model_dict["C"](val_pred_list)
            val_loss = F.cross_entropy(val_combined_pred, label[val_mask].long())
            history['val_loss'].append(val_loss.item())

        if val_loss.item() < best_val_loss - early_stopping_delta:
            best_val_loss = val_loss.item()
            best_model = {k: v.state_dict() for k, v in model_dict.items()}
            no_improve_count = 0
        else:
            no_improve_count += 1

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Train Loss={train_loss.item():.4f}, Val Loss={val_loss.item():.4f}")

        if no_improve_count >= early_stopping_patience:
            logger.info("Early stopping triggered based on validation loss.")
            break

    result['training_history'] = history
    result['best_model_state'] = best_model

    # Evaluation
    for m in model_dict:
        model_dict[m].eval()

    with torch.no_grad():
        test_pred_list = [
            model_dict[f"C{i + 1}"](model_dict[f"E{i + 1}"](
                data_list[i].x[test_mask],
                data_list[i].prior_guide[test_mask][:, test_mask]))
            for i in range(len(data_list))
        ]
        test_combined_pred = model_dict["C"](test_pred_list)
        test_probs = F.softmax(test_combined_pred, dim=1)[:, 1].cpu().numpy()

        fpr, tpr, thresholds = roc_curve(y_test, test_probs)
        youden_index = np.argmax(tpr - fpr)
        optimal_threshold_youden = thresholds[youden_index]
        f1_scores = [f1_score(y_test, (test_probs > t).astype(int)) for t in thresholds]
        optimal_threshold_f1 = thresholds[np.argmax(f1_scores)]

        test_preds = (test_probs > optimal_threshold_f1).astype(int)
        test_acc = accuracy_score(y_test, test_preds)
        test_auc = roc_auc_score(y_test, test_probs)
        cm = confusion_matrix(y_test, test_preds)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        cr = classification_report(y_test, test_preds, output_dict=True)

        logger.info(f"Test accuracy: {test_acc:.4f}")
        logger.info(f"Test sensitivity: {sensitivity:.4f}")
        logger.info(f"Test specificity: {specificity:.4f}")
        logger.info(f"Test AUC: {test_auc:.4f}")
        logger.info(f"Confusion matrix:\n{cm}")
        logger.info(f"Classification report:\n{cr}")

        result['prediction_results'] = {
            'metrics': {
                'accuracy': test_acc,
                'auc': test_auc,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'confusion_matrix': cm.tolist(),
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
            'y_test': y_test.tolist()
        }

    return result