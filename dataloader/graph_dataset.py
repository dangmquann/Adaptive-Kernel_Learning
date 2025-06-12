
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from scipy import sparse as sp


def build_multimodal_graph_list(
    Xs_train_dict, Y_train, Xs_test_dict, Y_test,
    top_k=None, val_size=0.2, random_state=42, metric="cosine", cuda=False
):
    """
    Build one PyG Data graph per modality using build_graph_data.

    Args:
        Xs_train_dict: dict of modality -> ndarray [N, D]
        Xs_test_dict: same keys as Xs_train_dict
        Y_train, Y_test: label arrays
        Returns: list of PyG Data objects (one per modality)
    """
    graph_list = []
    for modality in Xs_train_dict:
        graph = build_graph_data(
            Xs_train=Xs_train_dict[modality],
            Y_train=Y_train,
            Xs_test=Xs_test_dict[modality],
            Y_test=Y_test,
            top_k=top_k,
            val_size=val_size,
            random_state=random_state,
            metric=metric,
            cuda=cuda
        )
        graph.modality = modality  # optional: attach modality name
        graph_list.append(graph)

    return graph_list


def build_graph_data(Xs_train, Y_train, Xs_test, Y_test, top_k=None, val_size=0.2, random_state=42, metric="cosine", cuda=False):
    # === 1. Train/Val Split ===
    all_train_indices = np.arange(len(Y_train))
    train_idx, val_idx, Y_train_only, Y_val = train_test_split(
        all_train_indices,
        Y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=Y_train
    )

    # === 2. Prepare modality data (single matrix) ===
    X_train_only = Xs_train[train_idx]
    X_val = Xs_train[val_idx]

    # === 3. Concatenate all data ===
    X_all = np.concatenate([X_train_only, X_val, Xs_test], axis=0)  # [N_total, D]

    # === 4. Adjacency matrix ===
    if metric == "cosine":
        adj_matrix = cosine_similarity(X_all)
    else:
        raise NotImplementedError(f"Metric {metric} not supported in build_graph_data.")
    adj_matrix_ = np.copy(adj_matrix)
    # === 5. Labels ===
    Y_test_masked = np.zeros((len(Y_test),), dtype=np.int64) if Y_test is None else Y_test
    Y_all = np.concatenate([Y_train_only, Y_val, Y_test_masked], axis=0)

    # === 6. Build masks ===
    n_train = len(Y_train_only)
    n_val = len(Y_val)
    n_test = len(Y_test)
    N_total = n_train + n_val + n_test

    train_mask = torch.zeros(N_total, dtype=torch.bool)
    val_mask = torch.zeros(N_total, dtype=torch.bool)
    test_mask = torch.zeros(N_total, dtype=torch.bool)

    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True

    # === 7. Normalize and stack features ===
    X_all_norm = (X_all - np.mean(X_all, axis=0)) / (np.std(X_all, axis=0) + 1e-8)
    x = torch.tensor(X_all_norm, dtype=torch.float)

    if cuda and torch.cuda.is_available():
        x = x.cuda()

    # === 8. Adjacency processing (top-k) ===
    if top_k is not None and top_k > 0:
        adj_copy = adj_matrix.copy()
        np.fill_diagonal(adj_copy, 0)
        for i in range(adj_copy.shape[0]):
            idx = np.argpartition(adj_copy[i], -top_k)
            mask = np.ones_like(adj_copy[i], dtype=bool)
            mask[idx[-top_k:]] = False
            adj_copy[i][mask] = 0
        adj_matrix = np.maximum(adj_copy, adj_copy.T)
        np.fill_diagonal(adj_matrix, 1.0)

    # === 9. Sparse adjacency
    adj_sp = sp.coo_matrix(adj_matrix)
    edge_index = torch.tensor(np.vstack((adj_sp.row, adj_sp.col)), dtype=torch.long)
    edge_attr = torch.tensor(adj_sp.data, dtype=torch.float)
    if cuda and torch.cuda.is_available():
        edge_index = edge_index.cuda()
        edge_attr = edge_attr.cuda()

    # === 10. Create PyG Data object
    data = Data(x=x, edge_index=edge_index)
    data.edge_attr = edge_attr
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask


    y = torch.tensor(Y_all, dtype=torch.float)
    if cuda and torch.cuda.is_available():
        y = y.cuda()
    data.y = y

    data.prior_guide = torch.tensor(adj_matrix, dtype=torch.float)
    if cuda and torch.cuda.is_available():
        data.prior_guide = data.prior_guide.cuda()
    data.adj_matrix = adj_matrix_
    return data

def build_graph_data_multimodal_prior(Xs_train_dict, Y_train, Xs_test_dict, Y_test, 
                                     selected_modality, top_k=None, val_size=0.2, 
                                     random_state=42, metric="cosine", cuda=False):
    """
    Build a graph with features from selected_modality but prior guide from all modalities
    
    Args:
        Xs_train_dict: dict of modality -> ndarray [N, D]
        Y_train: labels for training data
        Xs_test_dict: dict of modality -> ndarray [N, D]
        Y_test: labels for test data
        selected_modality: which modality to use for node features
        top_k: number of neighbors to keep
        val_size: validation set size
        random_state: random seed
        metric: similarity metric
        cuda: whether to use GPU
    """
    # === 1. Train/Val Split ===
    all_train_indices = np.arange(len(Y_train))
    train_idx, val_idx, Y_train_only, Y_val = train_test_split(
        all_train_indices,
        Y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=Y_train
    )

    # === 2. Prepare selected modality data for features ===
    X_train_only = Xs_train_dict[selected_modality][train_idx]
    X_val = Xs_train_dict[selected_modality][val_idx]
    X_test = Xs_test_dict[selected_modality]

    # === 3. Concatenate selected modality data for features ===
    X_all = np.concatenate([X_train_only, X_val, X_test], axis=0)  # [N_total, D]
    
    # === 4. Create combined adjacency matrix from all modalities ===
    combined_adj_matrix = None
    
    for modality in Xs_train_dict.keys():
        # Process each modality
        X_train_mod = Xs_train_dict[modality][train_idx]
        X_val_mod = Xs_train_dict[modality][val_idx]
        X_test_mod = Xs_test_dict[modality]
        
        # Concatenate for this modality
        X_all_mod = np.concatenate([X_train_mod, X_val_mod, X_test_mod], axis=0)
        
        # Calculate adjacency for this modality
        if metric == "cosine":
            adj_matrix_mod = cosine_similarity(X_all_mod)
        else:
            raise NotImplementedError(f"Metric {metric} not supported")
            
        # Add to combined adjacency
        if combined_adj_matrix is None:
            combined_adj_matrix = adj_matrix_mod
        else:
            combined_adj_matrix += adj_matrix_mod
            
    # Normalize the combined adjacency if needed
    combined_adj_matrix = combined_adj_matrix / len(Xs_train_dict)  # Average
    adj_matrix_ = np.copy(combined_adj_matrix)
    
    # === 5. Labels ===
    Y_test_masked = np.zeros((len(Y_test),), dtype=np.int64) if Y_test is None else Y_test
    Y_all = np.concatenate([Y_train_only, Y_val, Y_test_masked], axis=0)

    # === 6. Build masks ===
    n_train = len(Y_train_only)
    n_val = len(Y_val)
    n_test = len(Y_test)
    N_total = n_train + n_val + n_test

    train_mask = torch.zeros(N_total, dtype=torch.bool)
    val_mask = torch.zeros(N_total, dtype=torch.bool)
    test_mask = torch.zeros(N_total, dtype=torch.bool)

    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True

    # === 7. Normalize and stack features (using only selected modality) ===
    X_all_norm = (X_all - np.mean(X_all, axis=0)) / (np.std(X_all, axis=0) + 1e-8)
    x = torch.tensor(X_all_norm, dtype=torch.float)

    if cuda and torch.cuda.is_available():
        x = x.cuda()

    # === 8. Adjacency processing (top-k) on the combined adjacency ===
    if top_k is not None and top_k > 0:
        adj_copy = combined_adj_matrix.copy()
        np.fill_diagonal(adj_copy, 0)
        for i in range(adj_copy.shape[0]):
            idx = np.argpartition(adj_copy[i], -top_k)
            mask = np.ones_like(adj_copy[i], dtype=bool)
            mask[idx[-top_k:]] = False
            adj_copy[i][mask] = 0
        combined_adj_matrix = np.maximum(adj_copy, adj_copy.T)
        np.fill_diagonal(combined_adj_matrix, 1.0)

    # === 9. Sparse adjacency
    adj_sp = sp.coo_matrix(combined_adj_matrix)
    edge_index = torch.tensor(np.vstack((adj_sp.row, adj_sp.col)), dtype=torch.long)
    edge_attr = torch.tensor(adj_sp.data, dtype=torch.float)
    if cuda and torch.cuda.is_available():
        edge_index = edge_index.cuda()
        edge_attr = edge_attr.cuda()

    # === 10. Create PyG Data object ===
    data = Data(x=x, edge_index=edge_index)
    data.edge_attr = edge_attr
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    y = torch.tensor(Y_all, dtype=torch.float)
    if cuda and torch.cuda.is_available():
        y = y.cuda()
    data.y = y

    # Use the combined adjacency matrix as prior guide
    data.prior_guide = torch.tensor(combined_adj_matrix, dtype=torch.float)
    if cuda and torch.cuda.is_available():
        data.prior_guide = data.prior_guide.cuda()
        
    data.adj_matrix = adj_matrix_  # Store original for reference
    return data