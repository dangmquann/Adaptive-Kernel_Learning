import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os
from datetime import datetime

def setup_logger(log_dir="logs"):
    """Setup logger configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a unique log file name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    return logging.getLogger(__name__)



class FeatureEmbedder(nn.Module):
    def __init__(self, input_dims, embed_dim):
        '''input_dim: list of input dimensions for each modality [200,200,200]
        embed_dim: embedding dimension for mỗi modality [128,128,128]'''
        super(FeatureEmbedder, self).__init__()
        self.embed_layers = nn.ModuleList([
            nn.Linear(input_dim, embed_dim) for input_dim in input_dims
        ])
    
    def forward(self, x, modalities=None):
        '''
        x: [num_nodes, num_modalities, input_dim]
        modalities: list of modality indices to use (e.g., [0,2]) hoặc None (dùng tất cả)
        '''
        if modalities is None:
            modalities = range(len(self.embed_layers))
        embeds = [self.embed_layers[i](x[:, i, :]) for i in modalities]
        # sum
        return torch.sum(torch.stack(embeds), dim=0)  # [num_nodes, num_modalities * embed_dim]
    
class GCTlayer(nn.Module):
    def __init__(self, embed_dim, num_heads=1, dropout=0.1, multihead_agg='concat'):
        super(GCTlayer, self).__init__()
        out_channels = embed_dim // num_heads if multihead_agg == 'concat' else embed_dim
        self.gat = GATConv(
            in_channels=embed_dim,
            out_channels=out_channels,
            heads=num_heads,
            concat=(multihead_agg == 'concat'),
            dropout=dropout,
            
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x, edge_index):
       # return updated x and attention weights
        (x_att, (edge_index, alpha)) = self.gat(x, edge_index, return_attention_weights=True)
        x = self.norm1(x + x_att)
        x_ffn = self.ffn(x)
        x = self.norm2(x + x_ffn)
        return x, alpha

class GraphConvolutionalTransformer(nn.Module):
    def __init__(self,
                 embed_dim=128,
                 num_layers=3,
                 num_heads=1,
                 dropout=0.1,
                 multihead_agg='concat',
                 reg_coef=0.1,
                 use_prior=True,
                 **kwargs):
        super(GraphConvolutionalTransformer, self).__init__()
        self.layers = nn.ModuleList(
            GCTlayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                multihead_agg=multihead_agg
            ) for _ in range(num_layers)
        )
        self.reg_coef = reg_coef
        self.use_prior = use_prior
    
    def forward(self, x, edge_index):
        attentions = []
        for layer in self.layers:
            x, alpha = layer(x, edge_index)
            attentions.append(alpha)
        return x, attentions

class PatientNodeClassifier(nn.Module):
    def __init__(self, in_dims, embedding_size, gct_params, num_classes=1):
        super(PatientNodeClassifier, self).__init__()
        self.embedder = FeatureEmbedder(in_dims, embedding_size)
        self.batch_norm = nn.BatchNorm1d(embedding_size)
        self.relu = nn.ReLU()
        # Pass reg_coef and use_prior into GCT for loss
        self.gct = GraphConvolutionalTransformer(
            embedding_size=embedding_size,
            num_layers=gct_params.get('num_layers', 3),
            num_heads=gct_params.get('num_heads', 1),
            dropout=gct_params.get('dropout', 0.1),
            multihead_agg=gct_params.get('multihead_agg', 'concat'),
            reg_coef=gct_params.get('reg_coef', 0.0),
            use_prior=gct_params.get('use_prior', False)
        )
        self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, data):
        x = self.embedder(data.x)  # [num_nodes, embedding_size]
        x = self.batch_norm(x)
        x = self.relu(x)
        x, attentions = self.gct(x, data.edge_index)
        logits = self.classifier(x).squeeze(-1)
        return logits, attentions

    def get_loss(self, logits, labels, attentions, prior_guide=None):
        """
        logits: [num_nodes]
        labels: [num_nodes]
        attentions: list of attention tensors per layer, each [num_edges, num_heads]
        prior_guide: optional conditional probability kernel of shape [num_nodes, num_nodes]
        """
        # Binary classification loss
        bce = F.binary_cross_entropy_with_logits(logits, labels.float())
        loss = bce
        # KL-regularization on attention if using prior
        if self.gct.use_prior and prior_guide is not None:
            kl_terms = []
            # assume attentions as list of [E, H] for sequential layers
            for i in range(1, len(attentions)):
                p = attentions[i-1]
                q = attentions[i]
                # convert to dense node-node if needed, or apply directly
                # here we approximate KL on attention weights
                log_p = torch.log(p + 1e-12)
                log_q = torch.log(q + 1e-12)
                kl = p * (log_p - log_q)
                kl = kl.sum(dim=-1).mean()
                kl_terms.append(kl)
            reg_term = torch.stack(kl_terms).mean()
            loss = loss + self.gct.reg_coef * reg_term
        return loss

    def get_predictions(self, model, ):
        '''return logits and attention weights'''
        pass


def graph_from_dist_tensor(dist, parameter, self_dist=True):
    """
    Creates an adjacency matrix from a distance tensor based on a threshold parameter.
    
    Args:
        dist (torch.Tensor): Distance matrix between nodes
        parameter (float or torch.Tensor): Threshold parameter for edge creation
        self_dist (bool): If True, handles self-connections by zeroing diagonal
    
    Returns:
        torch.Tensor: Binary adjacency matrix as float tensor
    """
    if self_dist:
        assert dist.shape[0] == dist.shape[1], "Input is not a pairwise distance matrix"
    
    # Convert parameter to tensor if it's not already
    if not isinstance(parameter, torch.Tensor):
        parameter = torch.tensor(parameter, device=dist.device, dtype=dist.dtype)
    
    # Create adjacency matrix
    g = (dist <= parameter).float()
    
    # Remove self-loops if requested
    if self_dist:
        diag_idx = torch.arange(g.size(0), device=g.device)
        g[diag_idx, diag_idx] = 0
    
    return g


def gen_adj_mat_tensor(data, parameter, metric="cosine", cuda=True):
    # data: [num_nodes, feat_dim]
    assert metric == "cosine", "Only cosine distance implemented"
    # compute cosine distance: 1 - cosine similarity
    data_norm = F.normalize(data, p=2, dim=1)
    sim = torch.mm(data_norm, data_norm.t())  # cosine similarity
    dist = 1 - sim

    mask = graph_from_dist_tensor(dist, parameter, self_dist=True)
    adj = sim * mask
    # symmetrize
    adj_t = adj.t()
    adj = adj + adj_t * (adj_t > adj).float() - adj * (adj_t > adj).float()
    # add self-loop
    I = torch.eye(adj.size(0), device=adj.device)
    adj = adj + I
    # row-normalize
    adj = F.normalize(adj, p=1, dim=1)
    return adj.to_sparse()


def build_data(modality_matrices, adjacency_parameter=None, labels=None,
               metric="cosine", top_k=None, cuda=True):
    """
    Constructs a PyG Data object from per-modality node features and dynamically learned
    adjacency via a distance threshold or top-k filtering.

    Args:
        modality_matrices: list of np.ndarray, each [num_nodes, feat_dim]
        adjacency_parameter: float threshold for cosine distance
        labels: array-like [num_nodes] or None
        metric: distance metric, only 'cosine' supported
        top_k: int or None; if int, keep only top_k neighbors per node
    """

    # Normalize input features
    normalized_matrices = [
        (m - np.mean(m, axis=0)) / (np.std(m, axis=0) + 1e-8)
        for m in modality_matrices
    ]
    x_np = np.stack(normalized_matrices, axis=1)
    # 1) Stack features: [N, M, D]
    x_np = np.stack(modality_matrices, axis=1)
    x = torch.tensor(x_np, dtype=torch.float)
    if cuda and torch.cuda.is_available():
        x = x.cuda()

    # 2) Flatten modalities for adjacency: [N, M*D]
    N, M, D = x.size()
    x_flat = x.view(N, M * D)

    # 3) Build adjacency sparse tensor
    assert adjacency_parameter is not None, "Provide adjacency_parameter for distance threshold"
    adj_sp = gen_adj_mat_tensor(x_flat, adjacency_parameter, metric, cuda=cuda)

    # 4) Optional top_k filtering on dense repr
    if top_k is not None:
        adj_dense = adj_sp.to_dense()
        mask = torch.zeros_like(adj_dense, dtype=torch.bool)
        for i in range(N):
            row = adj_dense[i].clone()
            row[i] = -float('inf')
            topk_idx = torch.topk(row, top_k).indices
            mask[i, topk_idx] = True
        mask = mask | mask.t()
        adj_dense = adj_dense * mask.float()
        adj_sp = adj_dense.to_sparse()

    # 5) Extract edge_index and edge_weight
    edge_index = adj_sp.indices()
    edge_weight = adj_sp.values()

    # 6) Build Data object
    data = Data(x=x, edge_index=edge_index)
    data.edge_attr = edge_weight
    if labels is not None:
        y = torch.tensor(labels, dtype=torch.float)
        if cuda and torch.cuda.is_available():
            y = y.cuda()
        data.y = y
    return data


def train_test(
        Xs_train, Y_train, Xs_test, Y_test,
        chosen_modalities,
        embedding_size=128,
        gct_params=None,
        num_epochs=100,
        lr=1e-3,
        random_seed=42,
        top_k=None,
        log_dir="logs",
        # Added parameters
        checkpoint_dir="checkpoints",
        early_stopping_patience=20,
        early_stopping_delta=0.001,
):
    """
    Train and evaluate PatientNodeClassifier using a k-NN graph constructed via cosine similarity.

    Args:
        Xs_train: list of np.ndarray [n_train, feature_dim] per modality
        Y_train: np.ndarray [n_train]
        Xs_test: list of np.ndarray [n_test, feature_dim] per modality
        Y_test: np.ndarray [n_test]
        chosen_modalities: list of modality indices to use
        top_k: int or None; number of edges per node to keep in adjacency
        checkpoint_dir: directory to save model checkpoints
        early_stopping_patience: number of epochs to wait for improvement before stopping
        early_stopping_delta: minimum change to qualify as improvement
    """

    logger = setup_logger(log_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Log training configuration
    logger.info("Training Configuration:")
    logger.info(f"Embedding size: {embedding_size}")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Random seed: {random_seed}")
    logger.info(f"Top-k neighbors: {top_k}")
    logger.info(f"GCT parameters: {gct_params}")
    logger.info(f"Early stopping patience: {early_stopping_patience}")

    # Reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Concatenate features for chosen modalities
    X_train_concat = np.concatenate([Xs_train[m] for m in chosen_modalities], axis=1)
    X_test_concat  = np.concatenate([Xs_test[m]  for m in chosen_modalities], axis=1) 
    X_all_concat   = np.concatenate([X_train_concat, X_test_concat], axis=0)
    logging.info(f"X_all_concat shape: {X_all_concat.shape}")

    # Compute full adjacency via cosine similarity
    adj_matrix = cosine_similarity(X_all_concat, X_all_concat)
    logging.info(f"Adjacency matrix shape: {adj_matrix.shape}")
    
    # Prepare per-modality node features for graph
    modality_matrices_all = [
        np.concatenate([Xs_train[m], Xs_test[m]], axis=0)
        for m in chosen_modalities
    ]

    # Combine labels
    Y_all = np.concatenate([Y_train, Y_test], axis=0)

    # Build Data with optional top_k filtering
    data = build_data(modality_matrices_all, adj_matrix, labels=Y_all, top_k=top_k)
    logging.info(f"Data object created with {data.num_nodes} nodes and {data.num_edges} edges.")
    
    # Define split indices
    num_train = len(Y_train)
    train_idx = list(range(num_train))
    test_idx  = list(range(num_train, num_train + len(Y_test)))

    # Initialize model and optimizer
    model = PatientNodeClassifier(
        in_dims=[Xs_train[m].shape[1] for m in chosen_modalities],
        embedding_size=embedding_size,
        gct_params=gct_params
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5, #scheduler_factor: factor by which to reduce learning rate
        patience=5,  #scheduler_patience: number of epochs with no improvement after which lr will be reduced
        verbose=True,
        min_lr=1e-6 
    )

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [], 'lr': []}
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    no_improve_count = 0
    best_val_acc = 0.0
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits, attentions = model(data)
        loss = model.get_loss(
            logits[train_idx], data.y[train_idx], attentions,
            prior_guide=adj_matrix
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, attentions = model(data)
            train_acc = (logits[train_idx] > 0.5).float().eq(data.y[train_idx]).sum().item() / len(train_idx)
            val_loss = model.get_loss(
                logits[test_idx], data.y[test_idx], attentions,
                prior_guide=adj_matrix
            ).item()
            val_acc = (logits[test_idx] > 0.5).float().eq(data.y[test_idx]).sum().item() / len(test_idx)

        # Update history
        current_lr = optimizer.param_groups[0]['lr']
        history['loss'].append(loss.item())
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['lr'].append(current_lr)

        # Log training progress
        logger.info(
            f"Epoch {epoch}/{num_epochs} - "
            f"Loss: {loss.item():.4f} - Acc: {train_acc:.4f} - "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - "
            f"LR: {current_lr:.6f}"
        )
        
        # Update learning rate scheduler
        scheduler.step(val_loss)  # Use val_loss for LR scheduling
        
        # Check for improvement based on validation ACCURACY (for model checkpointing)
        acc_improved = val_acc > best_val_acc + early_stopping_delta
        
        # Check for improvement based on validation LOSS (for early stopping)
        loss_improved = val_loss < best_val_loss - early_stopping_delta
        
        # Update best loss for early stopping
        if loss_improved:
            best_val_loss = val_loss
            no_improve_count = 0
        else:
            no_improve_count += 1
            logger.info(f"No val_loss improvement for {no_improve_count} epochs (best loss: {best_val_loss:.4f})")
        
        # Save checkpoint based on best accuracy
        if acc_improved:
            # Save best model based on validation accuracy
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_acc
            }
            
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'model_{timestamp}_best.pt')
            torch.save(best_model_state, checkpoint_path)
            logger.info(f"Saved checkpoint with best val_accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
        
        # Early stopping check based on validation LOSS
        if no_improve_count >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch} epochs due to no val_loss improvement")
            break
            
    logger.info("Training completed!")
    
    # Load best model if available
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state_dict'])
        logger.info(f"Loaded best model from epoch {best_epoch} with val_accuracy {best_val_acc:.4f}")
    
    return model, history