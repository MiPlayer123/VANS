"""Evaluation functions for VANS."""

import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Evaluate model on a dataset.

    Args:
        model: VANS model
        loader: DataLoader
        device: torch.device

    Returns:
        overall_acc: Overall accuracy
        config_acc: Dict mapping config name to accuracy
    """
    model.eval()
    correct = 0
    total = 0
    config_correct = {}
    config_total = {}

    for batch in tqdm(loader, desc='Evaluating', leave=False):
        context = batch['context'].to(device)
        candidates = batch['candidates'].to(device)
        targets = batch['target'].to(device)
        configs = batch['config']

        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            scores, _ = model(context, candidates)

        preds = scores.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        # Per-config tracking
        for i, cfg in enumerate(configs):
            if cfg not in config_correct:
                config_correct[cfg] = 0
                config_total[cfg] = 0
            config_correct[cfg] += (preds[i] == targets[i]).item()
            config_total[cfg] += 1

    overall_acc = correct / total if total > 0 else 0
    config_acc = {cfg: config_correct[cfg] / config_total[cfg]
                  for cfg in config_correct}

    return overall_acc, config_acc


@torch.no_grad()
def confusion_analysis(model, loader, device):
    """
    Analyze model predictions and collect confusion data.

    Args:
        model: VANS model
        loader: DataLoader
        device: torch.device

    Returns:
        confusion_data: Dict with per-config confusion info
        errors: List of error details
    """
    model.eval()
    confusion_data = defaultdict(lambda: {'correct': 0, 'total': 0, 'pred_dist': np.zeros(8)})
    errors = []

    for batch in tqdm(loader, desc='Analyzing', leave=False):
        context = batch['context'].to(device)
        candidates = batch['candidates'].to(device)
        targets = batch['target'].to(device)
        configs = batch['config']

        scores, _ = model(context, candidates)
        preds = scores.argmax(dim=1)

        for i in range(len(targets)):
            cfg = configs[i]
            confusion_data[cfg]['total'] += 1
            confusion_data[cfg]['correct'] += (preds[i] == targets[i]).item()
            confusion_data[cfg]['pred_dist'][preds[i].item()] += 1

            # Collect error info
            if preds[i] != targets[i]:
                probs = torch.softmax(scores[i], dim=0)
                errors.append({
                    'config': cfg,
                    'predicted': preds[i].item(),
                    'actual': targets[i].item(),
                    'confidence': probs[preds[i]].item(),
                    'correct_prob': probs[targets[i]].item(),
                    'margin': (probs[preds[i]] - probs[targets[i]]).item()
                })

    return dict(confusion_data), errors


def print_test_results(test_acc, config_acc, config_short=None):
    """
    Print formatted test results.

    Args:
        test_acc: Overall test accuracy
        config_acc: Dict mapping config name to accuracy
        config_short: Dict mapping full config names to short names
    """
    config_short = config_short or {}

    print("=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"\nOverall Accuracy: {test_acc:.2%}")
    print(f"\nPer-Configuration Breakdown:")
    print("-" * 40)

    for cfg in sorted(config_acc.keys()):
        acc = config_acc[cfg]
        short_name = config_short.get(cfg, cfg[:15])
        bar = '#' * int(acc * 30) + '-' * (30 - int(acc * 30))
        print(f"{short_name:15} {bar} {acc:.1%}")

    print("-" * 40)
    overall_bar = '#' * int(test_acc * 30) + '-' * (30 - int(test_acc * 30))
    print(f"{'OVERALL':15} {overall_bar} {test_acc:.1%}")
    print(f"{'Random guess':15} {'-' * 30} 12.5%")


def print_error_analysis(errors):
    """
    Print error analysis summary.

    Args:
        errors: List of error dicts from confusion_analysis
    """
    if not errors:
        print("\nNo errors to analyze!")
        return

    print(f"\nTotal errors: {len(errors)}")

    # Error distribution by config
    print("\nErrors by configuration:")
    error_by_config = {}
    for e in errors:
        cfg = e['config']
        error_by_config[cfg] = error_by_config.get(cfg, 0) + 1

    for cfg, count in sorted(error_by_config.items(), key=lambda x: -x[1]):
        print(f"  {cfg}: {count} errors")

    # Analyze confidence on errors
    print("\nError confidence analysis:")
    confidences = [e['confidence'] for e in errors]
    margins = [e['margin'] for e in errors]
    print(f"  Mean confidence on wrong answer: {np.mean(confidences):.2%}")
    print(f"  Mean margin (wrong - correct): {np.mean(margins):.2%}")

    # High confidence errors
    high_conf_errors = [e for e in errors if e['confidence'] > 0.5]
    print(f"\nHigh confidence errors (>50%): {len(high_conf_errors)}")
