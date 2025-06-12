import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GateNet(nn.Module):
    """Compute modality gate logits from summary statistics of attention maps."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),  # 4 summary features
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, stats: torch.Tensor) -> torch.Tensor:  # stats [M, 4]
        return self.net(stats).squeeze(-1)  # [M]


def _attn_stats(attn: torch.Tensor) -> torch.Tensor:
    """Return simple statistics describing an attention map.

    Args: attn: [H, N, N]
    Returns: [4] – mean, std, entropy, head‑avg sparsity
    """
    # Flatten heads + pairs
    H, N, _ = attn.shape
    flat = attn.reshape(H, -1)  # [H, N*N]
    mean = flat.mean().unsqueeze(0)
    std = flat.std().unsqueeze(0)
    # entropy per head → average
    eps = 1e-9
    p = attn / (attn.sum(dim=-1, keepdim=True) + eps)  # normalise rows
    ent = -(p * (p + eps).log()).sum(dim=-1).mean()  # scalar
    # sparsity = fraction of entries < 1/H*N*N
    thresh = 1.0 / (H * N * N)
    spars = ((attn < thresh).float().mean()).unsqueeze(0)
    return torch.cat([mean, std, ent.unsqueeze(0), spars])  # [4]


class AdaptiveKernelFusion(nn.Module):
    def __init__(self, embed_dim: int, num_modalities: int, num_heads: int,
                 dropout: float = 0.1, residual: bool = True,
                 head_reweight: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.num_heads = num_heads
        self.residual = residual
        self.head_reweight = head_reweight

        # Modality gating
        self.mod_gate = _GateNet()

        # Optional per‑head re‑weighting β_{m,h}
        if head_reweight:
            self.head_weights = nn.Parameter(torch.zeros(num_modalities, num_heads))
        else:
            self.register_parameter('head_weights', None)

        # Graph refinement layer (single‑layer GAT‑style)
        self.refine = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeds, attn_maps):
        # embeds: list M × [N, D]
        # attn_maps: list M × [H, N, N]
        assert len(attn_maps) == self.num_modalities
        N = embeds[0].size(0)

        # --- 1. Compute modality gate weights α ---
        stats = torch.stack([_attn_stats(a[-1]).detach() for a in attn_maps], dim=0)  # [M,4] - use last layer attention
        gate_logits = self.mod_gate(stats)  # [M]
        alpha = F.softmax(gate_logits, dim=0)  # [M]

        # --- 2. Optional head‑wise β re‑weight ---
        if self.head_reweight:
            beta = torch.sigmoid(self.head_weights)  # [M,H]
        else:
            beta = None  # placeholder

        # --- 3. Fuse kernels ---
        # Get the last layer's attention maps from each modality
        last_attns = [a[-1] for a in attn_maps]  # list of [H,N,N]
        
        fused_A = torch.zeros(self.num_heads, N, N, device=last_attns[0].device)
        for m, A in enumerate(last_attns):
            if self.head_reweight:
                A = beta[m].view(-1, 1, 1) * A  # scale heads
            fused_A += alpha[m] * A
            
        # Row‑normalise fused kernel for message passing
        D_inv = fused_A.sum(dim=-1, keepdim=True).clamp(min=1e-8).reciprocal()
        fused_A = fused_A * D_inv  # stochastic attention

        # --- 4. Message passing (single step) ---
        H_split = torch.stack(embeds, dim=0).mean(dim=0)  # use mean of embeds as input features [N,D]
        # aggregate over heads then neighbours
        agg = torch.einsum('hij,jd->id', fused_A, H_split) / self.num_heads  # [N,D]
        out = self.refine(agg)
        out = self.dropout(F.relu(out))
        if self.residual:
            out = out + H_split
        return self.norm(out)
    





class EmbeddingFusion(nn.Module):
    """
    Fusion module that combines embeddings from multiple modalities using various strategies.
    
    Fusion strategies:
    - 'concat': Simple concatenation followed by projection
    - 'mean': Weighted average of embeddings
    - 'attention': Cross-modal attention between embeddings
    - 'gating': Gate-based fusion with modality-specific importance weights
    - 'bilinear': Bilinear fusion for pairwise modality interactions
    - 'tensor': Higher-order tensor fusion (more expressive but computationally intensive)
    """
    
    def __init__(self, embed_dim, num_modalities, 
                 fusion_type='attention', 
                 hidden_size=128,
                 dropout=0.1):
        """
        Initialize the fusion module.
        
        Args:
            embed_dim: Dimension of each modality's embedding
            num_modalities: Number of input modalities
            fusion_type: Type of fusion mechanism to use
            hidden_size: Size of intermediate representations
            dropout: Dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.fusion_type = fusion_type
        
        # Modality-specific projections (can help normalize different embedding spaces)
        self.projections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_modalities)
        ])
        
        # Modality weights for weighted fusion
        self.modality_weights = nn.Parameter(torch.ones(num_modalities))
        
        if fusion_type == 'concat':
            self.fusion_layer = nn.Sequential(
                nn.Linear(embed_dim * num_modalities, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, embed_dim)
            )
        
        elif fusion_type == 'attention':
            # Multi-head cross-attention
            self.attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=4,
                dropout=dropout
            )
            self.layer_norm = nn.LayerNorm(embed_dim)
            self.fusion_layer = nn.Sequential(
                nn.Linear(embed_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, embed_dim)
            )
            
        elif fusion_type == 'gating':
            # Gating networks for each modality
            self.gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                ) for _ in range(num_modalities)
            ])
            self.fusion_layer = nn.Sequential(
                nn.Linear(embed_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, embed_dim)
            )
            
        elif fusion_type == 'bilinear':
            # Pairwise bilinear fusion
            self.bilinear_layers = nn.ModuleList([
                nn.Bilinear(embed_dim, embed_dim, hidden_size)
                for _ in range((num_modalities * (num_modalities - 1)) // 2)
            ])
            self.fusion_layer = nn.Linear(hidden_size * ((num_modalities * (num_modalities - 1)) // 2), embed_dim)
            
        elif fusion_type == 'tensor':
            # Tensor fusion (simplified implementation)
            self.tensor_weights = nn.Parameter(
                torch.Tensor(num_modalities, embed_dim, hidden_size)
            )
            nn.init.xavier_normal_(self.tensor_weights)
            self.tensor_bias = nn.Parameter(torch.zeros(hidden_size))
            self.fusion_layer = nn.Linear(hidden_size, embed_dim)
            
        # Output normalization
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, embeddings):
        """
        Fuse embeddings from multiple modalities.
        
        Args:
            embeddings: List of tensors, each [batch_size, embed_dim]
            
        Returns:
            Fused embedding [batch_size, embed_dim]
        """
        assert len(embeddings) == self.num_modalities, f"Expected {self.num_modalities} modalities, got {len(embeddings)}"
        
        # Apply modality-specific projections
        projected = [self.projections[i](emb) for i, emb in enumerate(embeddings)]
        
        if self.fusion_type == 'concat':
            # Concatenate all embeddings
            concat = torch.cat(projected, dim=-1)
            fused = self.fusion_layer(concat)
            
        elif self.fusion_type == 'mean':
            # Weighted average
            weights = F.softmax(self.modality_weights, dim=0)
            weighted = [emb * w for emb, w in zip(projected, weights)]
            fused = sum(weighted)
            
        elif self.fusion_type == 'attention':
            # Stack embeddings for attention
            stacked = torch.stack(projected, dim=0)  # [num_modalities, batch_size, embed_dim]
            
            # Self attention across modalities
            attended, _ = self.attention(
                query=stacked,
                key=stacked,
                value=stacked
            )
            # Mean across modalities
            fused = torch.mean(attended, dim=0)
            fused = self.layer_norm(fused)
            fused = self.fusion_layer(fused)
            
        elif self.fusion_type == 'gating':
            # Learn importance of each modality dynamically
            gate_values = [gate(emb) for gate, emb in zip(self.gates, projected)]
            gate_weights = torch.cat(gate_values, dim=-1)
            gate_weights = F.softmax(gate_weights, dim=-1)
            
            # Apply gates to modalities
            gated_embeddings = [emb * gate_weights[:, i:i+1] for i, emb in enumerate(projected)]
            fused = sum(gated_embeddings)
            fused = self.fusion_layer(fused)
            
        elif self.fusion_type == 'bilinear':
            # Pairwise bilinear interactions
            bilinear_outputs = []
            idx = 0
            for i in range(self.num_modalities):
                for j in range(i+1, self.num_modalities):
                    interaction = self.bilinear_layers[idx](projected[i], projected[j])
                    bilinear_outputs.append(interaction)
                    idx += 1
            
            bilinear_concat = torch.cat(bilinear_outputs, dim=-1)
            fused = self.fusion_layer(bilinear_concat)
            
        elif self.fusion_type == 'tensor':
            # Higher-order tensor fusion
            tensor_products = []
            for i in range(self.num_modalities):
                # [batch, embed] × [embed, hidden] → [batch, hidden]
                projected_tensor = torch.matmul(projected[i], self.tensor_weights[i])
                tensor_products.append(projected_tensor)
            
            # Element-wise product of all projections
            fused = tensor_products[0]
            for tensor in tensor_products[1:]:
                fused = fused * tensor
            
            fused = fused + self.tensor_bias
            fused = self.fusion_layer(fused)
            
        # Normalize and return
        return self.norm(self.dropout(fused))
    
