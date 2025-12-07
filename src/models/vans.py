"""VANS: Visual Analogy Network Solver - Complete Model.

Combines ContextEncoder, RuleReasoningModule, and RulePredictor
to solve Raven's Progressive Matrices.
"""

import torch
import torch.nn as nn

from .context_encoder import ContextEncoder
from .rule_reasoning import RuleReasoningModule
from .rule_predictor import RulePredictor


class VANS(nn.Module):
    """
    Visual Analogy Network Solver - Complete Model

    Combines:
    - ContextEncoder: Encodes 8 context panels with positional info
    - RuleReasoningModule: Scores candidates via cross-attention
    - RulePredictor: Auxiliary task for rule prediction
    """

    def __init__(self, feature_dim=1024, hidden_dim=512, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()

        self.context_encoder = ContextEncoder(feature_dim, hidden_dim, num_heads, num_layers, dropout)
        self.rule_reasoning = RuleReasoningModule(feature_dim, hidden_dim, num_heads, dropout)
        self.rule_predictor = RulePredictor(hidden_dim)

    def forward(self, context_features, candidate_features):
        """
        Args:
            context_features: [B, 8, 1024] - pre-extracted DINOv2 features for context
            candidate_features: [B, 8, 1024] - pre-extracted DINOv2 features for candidates

        Returns:
            scores: [B, 8] - logits for each candidate answer
            rule_preds: [B, 20] - rule predictions (auxiliary task)
        """
        # Encode context panels
        context_repr, panel_features = self.context_encoder(context_features)

        # Score candidate answers
        scores = self.rule_reasoning(context_repr, panel_features, candidate_features)

        # Predict rules (auxiliary task)
        rule_preds = self.rule_predictor(context_repr)

        return scores, rule_preds


def create_model(config, device):
    """
    Create VANS model from config.

    Args:
        config: dict with model hyperparameters
        device: torch.device

    Returns:
        VANS model on device
    """
    model = VANS(
        feature_dim=config['feature_dim'],
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n[OK] VANS model created")
    print(f"     Total parameters: {total_params/1e6:.2f}M")
    print(f"     Trainable parameters: {trainable_params/1e6:.2f}M")

    # Test forward pass
    with torch.no_grad():
        dummy_context = torch.randn(2, 8, 1024).to(device)
        dummy_candidates = torch.randn(2, 8, 1024).to(device)
        scores, rules = model(dummy_context, dummy_candidates)
        print(f"\n     Test forward pass:")
        print(f"       Scores shape: {scores.shape}")
        print(f"       Rules shape: {rules.shape}")

    return model
