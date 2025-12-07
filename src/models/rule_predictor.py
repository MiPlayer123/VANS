"""Rule Predictor module for VANS.

Auxiliary task: predicts underlying rules from context representation.
"""

import torch.nn as nn


class RulePredictor(nn.Module):
    """
    Auxiliary task: predicts underlying rules from context representation.

    I-RAVEN has rules across 5 attributes (Type, Size, Color, Position, Number)
    with 4 rule types each (Constant, Progression, Arithmetic, Distribute_Three).
    Total: 20 binary predictions.
    """

    def __init__(self, hidden_dim=512, num_rules=20, dropout=0.2):
        super().__init__()

        self.rule_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, num_rules)
        )

    def forward(self, context_repr):
        """
        Args:
            context_repr: [B, 512] - context representation

        Returns:
            rule_preds: [B, 20] - rule predictions (logits)
        """
        return self.rule_head(context_repr)
