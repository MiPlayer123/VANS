"""Loss functions for VANS training."""

import torch
import torch.nn.functional as F


def primary_loss(scores, targets):
    """
    Cross-entropy loss for answer selection.

    Args:
        scores: [B, 8] - model output scores
        targets: [B] - correct answer indices

    Returns:
        Cross-entropy loss value
    """
    return F.cross_entropy(scores, targets)


def contrastive_margin_loss(scores, targets, margin=0.5):
    """
    Margin loss: correct answer should beat incorrect answers by margin.

    Encourages clear separation between correct and incorrect candidates.

    Args:
        scores: [B, 8] - model output scores
        targets: [B] - correct answer indices
        margin: Minimum margin between correct and best incorrect

    Returns:
        Hinge loss value
    """
    B = scores.shape[0]
    device = scores.device

    # Get score of correct answer
    correct_scores = scores[torch.arange(B, device=device), targets]  # [B]

    # Mask out correct answer to get max incorrect score
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask[torch.arange(B, device=device), targets] = False
    incorrect_scores = scores.clone()
    incorrect_scores[~mask] = float('-inf')
    max_incorrect = incorrect_scores.max(dim=1).values  # [B]

    # Hinge loss: penalize if margin not met
    loss = F.relu(margin - (correct_scores - max_incorrect))
    return loss.mean()


def compute_loss(scores, targets, alpha=1.0, gamma=0.1, margin=0.5):
    """
    Combined loss function.

    Args:
        scores: [B, 8] - model output scores
        targets: [B] - correct answer indices
        alpha: Weight for cross-entropy loss
        gamma: Weight for contrastive margin loss
        margin: Margin for contrastive loss

    Returns:
        total_loss: Combined loss value
        loss_dict: Dictionary with individual loss components
    """
    L_ce = primary_loss(scores, targets)
    L_margin = contrastive_margin_loss(scores, targets, margin)

    total = alpha * L_ce + gamma * L_margin

    return total, {'ce': L_ce.item(), 'margin': L_margin.item()}
