"""Rule Reasoning module for VANS.

Scores candidate answers based on context using cross-attention.
"""

import torch
import torch.nn as nn


class RuleReasoningModule(nn.Module):
    """
    Scores candidate answers based on context using cross-attention.

    Each candidate attends to the context panels to determine how well
    it completes the pattern. Self-attention among candidates provides
    contrastive awareness.
    """

    def __init__(self, feature_dim=1024, hidden_dim=512, num_heads=8, dropout=0.1):
        super().__init__()

        # Candidate projection
        self.candidate_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Cross-attention: candidates attend to context panels
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # Self-attention: candidates attend to each other (contrastive)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # Layer norms for residual connections
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Scoring MLP
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Learnable temperature for score calibration
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, context_repr, panel_features, candidate_features):
        """
        Args:
            context_repr: [B, 512] - global context representation
            panel_features: [B, 8, 512] - individual context panel features
            candidate_features: [B, 8, 1024] - 8 candidate answer features

        Returns:
            scores: [B, 8] - score for each candidate
        """
        B = candidate_features.shape[0]

        # Project candidates to hidden dimension
        cand = self.candidate_proj(candidate_features)  # [B, 8, 512]

        # Cross-attention: each candidate attends to context panels
        cand_attended, _ = self.cross_attn(cand, panel_features, panel_features)
        cand = self.norm1(cand + cand_attended)

        # Self-attention among candidates (contrastive awareness)
        cand_self, _ = self.self_attn(cand, cand, cand)
        cand = self.norm2(cand + cand_self)

        # Combine with global context for scoring
        context_expanded = context_repr.unsqueeze(1).expand(-1, 8, -1)  # [B, 8, 512]
        combined = torch.cat([cand, context_expanded], dim=-1)  # [B, 8, 1024]

        # Score each candidate
        scores = self.scorer(combined).squeeze(-1)  # [B, 8]

        # Apply learnable temperature
        scores = scores / (self.temperature.abs() + 0.1)

        return scores
