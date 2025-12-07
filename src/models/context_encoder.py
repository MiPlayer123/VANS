"""Context Encoder module for VANS.

Encodes 8 context panels with structural positional information.
"""

import torch
import torch.nn as nn


class ContextEncoder(nn.Module):
    """
    Encodes 8 context panels with structural positional information.

    The 8 context panels form a 3x3 grid with the bottom-right missing:
    [0,0] [0,1] [0,2]
    [1,0] [1,1] [1,2]
    [2,0] [2,1]  ???

    Uses row/column embeddings to capture the grid structure.
    """

    def __init__(self, feature_dim=1024, hidden_dim=512, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Positional embeddings for 8 positions
        self.position_embed = nn.Parameter(torch.randn(8, hidden_dim) * 0.02)

        # Row/Column embeddings for structural awareness
        self.row_embed = nn.Parameter(torch.randn(3, hidden_dim // 2) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(3, hidden_dim // 2) * 0.02)

        # Position indices for each of the 8 context panels
        # Order: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1)
        self.register_buffer('row_idx', torch.tensor([0, 0, 0, 1, 1, 1, 2, 2]))
        self.register_buffer('col_idx', torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Context aggregation via learnable query
        self.context_query = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, context_features):
        """
        Args:
            context_features: [B, 8, 1024] - 8 context panel features

        Returns:
            context_repr: [B, 512] - aggregated context representation
            panel_features: [B, 8, 512] - individual panel features
        """
        B = context_features.shape[0]

        # Project to hidden dimension
        x = self.input_proj(context_features)  # [B, 8, 512]

        # Add positional embeddings
        x = x + self.position_embed.unsqueeze(0)

        # Add row/col embeddings for structural awareness
        row_emb = self.row_embed[self.row_idx]  # [8, 256]
        col_emb = self.col_embed[self.col_idx]  # [8, 256]
        struct_emb = torch.cat([row_emb, col_emb], dim=-1)  # [8, 512]
        x = x + struct_emb.unsqueeze(0)

        # Transformer encoding (captures row/column patterns)
        panel_features = self.transformer(x)  # [B, 8, 512]

        # Aggregate to single context vector using learnable query
        query = self.context_query.unsqueeze(0).expand(B, -1, -1)  # [B, 1, 512]
        context_repr, _ = self.cross_attn(query, panel_features, panel_features)
        context_repr = context_repr.squeeze(1)  # [B, 512]

        return context_repr, panel_features
