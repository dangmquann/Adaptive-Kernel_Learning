import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ATK_late import *

class MultiHeadAttnQK(nn.Module):
    def __init__(self, num_nodes, embed_dim, num_heads=1, dropout=0.1,
                 multihead_agg='concat', attention_normalizer='softmax'):
        """
        Args:
            prior_guide: [N, N] — used to initialize attention head 0
        """
        super(MultiHeadAttnQK, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads if multihead_agg == 'concat' else embed_dim
        self.multihead_agg = multihead_agg
        self.attention_normalizer = attention_normalizer
        self.num_nodes = num_nodes

        # Full learnable attention map tensor
        self.attn_params = nn.Parameter(torch.randn(num_heads, num_nodes, num_nodes))

        # assert prior_guide.shape == (num_nodes, num_nodes), "prior_guide must be [N, N]"
        # with torch.no_grad():
        #     for h in range(num_heads):
        #         self.attn_params.data[h] = prior_guide
        self.scaling = self.head_dim ** -0.5
        self.query_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim if multihead_agg == 'concat' else self.head_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, prior_guide=None):
        """
        Args:
            x: [N, D]
        Returns:
            output: [N, D]
            attn_weights: [H, N, N]
        """

        N = x.size(0)
        assert N == self.num_nodes, "x must have shape [N, D] with correct N"
        # Initialize block 0 with prior_guide
        if prior_guide is not None:
            assert prior_guide.shape == (self.num_nodes, self.num_nodes), "prior_guide must be [N, N]"

        q = self.query_proj(x).view(N, self.num_heads, self.head_dim).permute(1, 0, 2)
        k = self.key_proj(x).view(N, self.num_heads, self.head_dim).permute(1, 0, 2)
        v = self.value_proj(x).view(N, self.num_heads, self.head_dim).permute(1, 0, 2)  # [H, N, D_h]

        scores = torch.matmul(q, k.transpose(1, 2)) * self.scaling

        
        if prior_guide is not None:
            attn_weights = torch.zeros(self.num_heads, N, N, device=x.device)
            for h in range(self.num_heads):
                # Initialize the first blocks with prior_guide
                attn_weights[h] = prior_guide
        else:
            # Normalize attention scores
            if self.attention_normalizer == 'softmax':
                attn_weights = F.softmax(scores, dim=-1)
            else:
                attn_weights = torch.sigmoid(scores)
        
        # Apply attention
        out = torch.matmul(attn_weights, v)  # [H, N, D_h]

        # Aggregate heads
        out = out.permute(1, 0, 2).contiguous()  # [N, H, D_h]
        if self.multihead_agg == 'concat':
            out = out.view(N, self.num_heads * self.head_dim)
        else:
            out = out.mean(dim=1)

        out = self.out_proj(out)
        out = self.norm1(x + self.dropout1(out))

        ffn_out = self.ffn(out)
        out = self.norm2(out + self.dropout2(ffn_out))

        return out, attn_weights

class AttentionKernelQK(nn.Module):
    def __init__(self,
                 num_nodes,
                 embed_dim=128,
                 num_layers=3,
                 num_heads=1,
                 dropout=0.1,
                 multihead_agg='concat',
                 reg_coef=0.1,
                 use_prior=True,
                 attention_normalizer='softmax',
                #  prior_guide=None,
                 **kwargs):
        super(AttentionKernelQK, self).__init__()
        self.layers = nn.ModuleList([
            MultiHeadAttnQK(
                num_nodes=num_nodes,
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                multihead_agg=multihead_agg,
                attention_normalizer=attention_normalizer,
                #prior_guide= prior_guide
            ) for _ in range(num_layers)
        ])
        self.reg_coef = reg_coef
        self.use_prior = use_prior

    def forward(self, x, prior_guide):
        """
        Args:
            x: Node features [num_nodes, embed_dim]
            edge_index: Optional edge indices
            prior_guide: Optional prior knowledge [num_nodes, num_nodes]
            inf_mask: Optional mask to limit attention [num_nodes, num_nodes]
        """
        attentions = []

        # First layer can use prior_guide if provided
        x, attn = self.layers[0](x, prior_guide if self.use_prior else None)
        attentions.append(attn)

        # Subsequent layers
        for i in range(1, len(self.layers)):
            x, attn = self.layers[i](x)
            attentions.append(attn)

        return x, attentions


class ATKQK(nn.Module):
    def __init__(self, num_nodes, in_dim, embedding_size, atk_params=None):#, prior_guide=None
        super(ATKQK, self).__init__()

        self.embedder = nn.Sequential(
            nn.Linear(in_dim, embedding_size),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(),
        )
        self.atk = AttentionKernelQK(
            num_nodes=num_nodes,
            embed_dim=embedding_size,
            num_layers=atk_params.get('num_layers', 3),
            num_heads=atk_params.get('num_heads', 1),
            dropout=atk_params.get('dropout', 0.1),
            multihead_agg=atk_params.get('multihead_agg', 'concat'),
            reg_coef=atk_params.get('reg_coef', 0.1),
            use_prior=atk_params.get('use_prior', True),
            attention_normalizer=atk_params.get('attention_normalizer', 'softmax'),
            # prior_guide=prior_guide
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        """Process graph data through the model.

        Args:
            data: PyG Data object containing node features and graph structure

        Returns:
            logits: Output logits for node classification
            attentions: Attention weights from each atk layer
        """
        # Restructure inputs if needed based on how data is provided
        x = self.embedder(data.x)

        # Extract adjacency information from data
        edge_index = data.edge_index

        prior_guide = data.prior_guide
        x, attentions = self.atk(x, prior_guide)
        logits = self.classifier(x).squeeze(-1)

        return logits, attentions

    def get_loss(self, logits, labels, attentions,train_mask):
        """
        Compute loss with optional KL regularization between attention layers

        Args:
            logits: Predicted logits [num_nodes]
            labels: Target labels [num_nodes]
            attentions: List of attention tensors from each layer
            prior_guide: Optional conditional probability matrix [num_nodes, num_nodes]
        """
        # Binary classification loss
        bce = F.binary_cross_entropy_with_logits(logits, labels.float())
        loss = bce
        # label_kernel = torch.matmul(labels.unsqueeze(1), labels.unsqueeze(0))
        # KL regularization on attention if using prior
        if self.atk.use_prior and len(attentions) > 1:
            kl_terms = []
            for i in range(1, len(attentions)):
                p = attentions[i - 1] + 1e-12  # Add epsilon to avoid log(0)
                q = attentions[i] + 1e-12
                # kl = forget_factor* F.kl_div(p, q, reduction='batchmean')
                # kl_terms.append(kl)
                log_p = torch.log(p)
                log_q = torch.log(q)
                kl = p * (log_p - log_q)
                kl = kl.sum(dim=-1).mean()
                kl_terms.append(kl)
            reg_term = torch.stack(kl_terms).mean() / attentions[0].shape[0] / attentions[0].shape[1]


            # calculate kernel output dot product of labels
            # reg_term = torch.stack(kl_terms).mean()
            # last_attention_train = torch.sum(attentions[-1],axis=0)/ len(attentions[-1])  # Average over heads
            # last_attention_train = last_attention_train[train_mask][:, train_mask]
            # kernel_loss = F.binary_cross_entropy_with_logits(last_attention_train, label_kernel.float())
            kernel_loss = 0
            loss = (1-self.atk.reg_coef)*loss + self.atk.reg_coef*(kernel_loss + reg_term) #(1 - self.atk.reg_coef) *
            print(f"Loss: {loss.item():.4f}, BCE: {bce.item():.4f}, Reg: {reg_term.item():.6f}")
        return loss
    


