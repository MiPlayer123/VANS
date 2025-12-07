# -*- coding: utf-8 -*-
"""
RAVEN Dataset Generator - VANS Integration Wrapper

This module provides a clean API for generating RAVEN Progressive Matrices
datasets for use with the VANS training pipeline.

Based on the original RAVEN implementation by Zhang et al. (CVPR 2019).
"""

import copy
import os
import random

import numpy as np
from tqdm import trange

from .build_tree import (build_center_single, build_distribute_four,
                         build_distribute_nine,
                         build_in_center_single_out_center_single,
                         build_in_distribute_four_out_center_single,
                         build_left_center_single_right_center_single,
                         build_up_center_single_down_center_single)
from .const import IMAGE_SIZE
from .rendering import render_panel
from .sampling import sample_attr, sample_attr_avail, sample_rules
from .serialize import dom_problem, serialize_aot, serialize_rules
from .solver import solve


# All available configurations
ALL_CONFIGS = {
    "center_single": build_center_single,
    "distribute_four": build_distribute_four,
    "distribute_nine": build_distribute_nine,
    "left_center_single_right_center_single": build_left_center_single_right_center_single,
    "up_center_single_down_center_single": build_up_center_single_down_center_single,
    "in_center_single_out_center_single": build_in_center_single_out_center_single,
    "in_distribute_four_out_center_single": build_in_distribute_four_out_center_single,
}


def merge_component(dst_aot, src_aot, component_idx):
    """Merge a component from src_aot into dst_aot."""
    src_component = src_aot.children[0].children[component_idx]
    dst_aot.children[0].children[component_idx] = src_component


def generate_single_sample(root, rule_groups):
    """
    Generate a single RAVEN sample given a root tree and rule groups.

    Args:
        root: AoT root node (pruned)
        rule_groups: List of rule groups

    Returns:
        dict with 'image', 'target', 'meta_matrix', 'meta_target',
        'structure', 'meta_structure', 'predicted'
    """
    start_node = root.sample()

    # Generate row 1
    row_1_1 = copy.deepcopy(start_node)
    for l in range(len(rule_groups)):
        rule_group = rule_groups[l]
        rule_num_pos = rule_group[0]
        row_1_2 = rule_num_pos.apply_rule(row_1_1)
        row_1_3 = rule_num_pos.apply_rule(row_1_2)
        for i in range(1, len(rule_group)):
            rule = rule_group[i]
            row_1_2 = rule.apply_rule(row_1_1, row_1_2)
        for i in range(1, len(rule_group)):
            rule = rule_group[i]
            row_1_3 = rule.apply_rule(row_1_2, row_1_3)
        if l == 0:
            to_merge = [row_1_1, row_1_2, row_1_3]
        else:
            merge_component(to_merge[1], row_1_2, l)
            merge_component(to_merge[2], row_1_3, l)
    row_1_1, row_1_2, row_1_3 = to_merge

    # Generate row 2
    row_2_1 = copy.deepcopy(start_node)
    row_2_1.resample(True)
    for l in range(len(rule_groups)):
        rule_group = rule_groups[l]
        rule_num_pos = rule_group[0]
        row_2_2 = rule_num_pos.apply_rule(row_2_1)
        row_2_3 = rule_num_pos.apply_rule(row_2_2)
        for i in range(1, len(rule_group)):
            rule = rule_group[i]
            row_2_2 = rule.apply_rule(row_2_1, row_2_2)
        for i in range(1, len(rule_group)):
            rule = rule_group[i]
            row_2_3 = rule.apply_rule(row_2_2, row_2_3)
        if l == 0:
            to_merge = [row_2_1, row_2_2, row_2_3]
        else:
            merge_component(to_merge[1], row_2_2, l)
            merge_component(to_merge[2], row_2_3, l)
    row_2_1, row_2_2, row_2_3 = to_merge

    # Generate row 3
    row_3_1 = copy.deepcopy(start_node)
    row_3_1.resample(True)
    for l in range(len(rule_groups)):
        rule_group = rule_groups[l]
        rule_num_pos = rule_group[0]
        row_3_2 = rule_num_pos.apply_rule(row_3_1)
        row_3_3 = rule_num_pos.apply_rule(row_3_2)
        for i in range(1, len(rule_group)):
            rule = rule_group[i]
            row_3_2 = rule.apply_rule(row_3_1, row_3_2)
        for i in range(1, len(rule_group)):
            rule = rule_group[i]
            row_3_3 = rule.apply_rule(row_3_2, row_3_3)
        if l == 0:
            to_merge = [row_3_1, row_3_2, row_3_3]
        else:
            merge_component(to_merge[1], row_3_2, l)
            merge_component(to_merge[2], row_3_3, l)
    row_3_1, row_3_2, row_3_3 = to_merge

    # Render context panels
    imgs = [render_panel(row_1_1),
            render_panel(row_1_2),
            render_panel(row_1_3),
            render_panel(row_2_1),
            render_panel(row_2_2),
            render_panel(row_2_3),
            render_panel(row_3_1),
            render_panel(row_3_2),
            np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)]
    context = [row_1_1, row_1_2, row_1_3, row_2_1, row_2_2, row_2_3, row_3_1, row_3_2]

    # Generate candidates (1 correct + 7 distractors)
    modifiable_attr = sample_attr_avail(rule_groups, row_3_3)
    answer_AoT = copy.deepcopy(row_3_3)
    candidates = [answer_AoT]
    for j in range(7):
        component_idx, attr_name, min_level, max_level = sample_attr(modifiable_attr)
        answer_j = copy.deepcopy(answer_AoT)
        answer_j.sample_new(component_idx, attr_name, min_level, max_level, answer_AoT)
        candidates.append(answer_j)

    random.shuffle(candidates)
    answers = []
    for candidate in candidates:
        answers.append(render_panel(candidate))

    # Combine context and answers
    image = imgs[0:8] + answers
    target = candidates.index(answer_AoT)
    predicted = solve(rule_groups, context, candidates)
    meta_matrix, meta_target = serialize_rules(rule_groups)
    structure, meta_structure = serialize_aot(start_node)

    return {
        'image': np.array(image),  # (16, 160, 160)
        'target': target,
        'predicted': predicted,
        'meta_matrix': meta_matrix,
        'meta_target': meta_target,
        'structure': structure,
        'meta_structure': meta_structure,
        'context_aot': context,
        'candidates_aot': candidates,
        'rule_groups': rule_groups,
    }


def generate_single_config(config_name, num_samples, save_dir, seed=42,
                           val_ratio=0.2, test_ratio=0.2, save_xml=False):
    """
    Generate samples for a single configuration.

    Args:
        config_name: Name of the configuration (e.g., 'center_single')
        num_samples: Number of samples to generate
        save_dir: Directory to save generated samples
        seed: Random seed
        val_ratio: Proportion for validation set (default 0.2)
        test_ratio: Proportion for test set (default 0.2)
        save_xml: Whether to save XML annotations (default False)

    Returns:
        dict with accuracy and sample counts
    """
    random.seed(seed)
    np.random.seed(seed)

    if config_name not in ALL_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(ALL_CONFIGS.keys())}")

    # Create output directory
    config_dir = os.path.join(save_dir, config_name)
    os.makedirs(config_dir, exist_ok=True)

    # Build the tree
    root = ALL_CONFIGS[config_name]()

    acc = 0
    train_count = val_count = test_count = 0

    for k in trange(num_samples, desc=config_name[:15]):
        # Determine split (based on index mod 10)
        count_num = k % 10
        train_threshold = int(10 * (1 - val_ratio - test_ratio))
        val_threshold = int(10 * (1 - test_ratio))

        if count_num < train_threshold:
            set_name = "train"
            train_count += 1
        elif count_num < val_threshold:
            set_name = "val"
            val_count += 1
        else:
            set_name = "test"
            test_count += 1

        # Sample rules and prune tree
        while True:
            rule_groups = sample_rules()
            new_root = root.prune(rule_groups)
            if new_root is not None:
                break

        # Generate sample
        sample = generate_single_sample(new_root, rule_groups)

        # Save NPZ file
        np.savez(
            os.path.join(config_dir, f"RAVEN_{k}_{set_name}.npz"),
            image=sample['image'],
            target=sample['target'],
            predict=sample['predicted'],
            meta_matrix=sample['meta_matrix'],
            meta_target=sample['meta_target'],
            structure=sample['structure'],
            meta_structure=sample['meta_structure']
        )

        # Optionally save XML
        if save_xml:
            with open(os.path.join(config_dir, f"RAVEN_{k}_{set_name}.xml"), "w") as f:
                dom = dom_problem(sample['context_aot'] + sample['candidates_aot'],
                                  sample['rule_groups'])
                f.write(dom)

        if sample['target'] == sample['predicted']:
            acc += 1

    accuracy = float(acc) / num_samples
    print(f"[OK] {config_name}: {num_samples} samples (train={train_count}, val={val_count}, test={test_count})")
    print(f"     Solver accuracy: {accuracy:.2%}")

    return {
        'config': config_name,
        'total': num_samples,
        'train': train_count,
        'val': val_count,
        'test': test_count,
        'accuracy': accuracy
    }


def generate_dataset(num_samples, save_dir, seed=42, configs=None,
                     val_ratio=0.2, test_ratio=0.2, save_xml=False):
    """
    Generate a full RAVEN dataset with all configurations.

    Args:
        num_samples: Number of samples per configuration
        save_dir: Root directory to save the dataset
        seed: Random seed
        configs: List of config names to generate (default: all)
        val_ratio: Proportion for validation set (default 0.2)
        test_ratio: Proportion for test set (default 0.2)
        save_xml: Whether to save XML annotations (default False)

    Returns:
        dict with results for each configuration
    """
    os.makedirs(save_dir, exist_ok=True)

    if configs is None:
        configs = list(ALL_CONFIGS.keys())

    print(f"Generating RAVEN dataset:")
    print(f"  Samples per config: {num_samples}")
    print(f"  Configurations: {len(configs)}")
    print(f"  Total samples: {num_samples * len(configs)}")
    print(f"  Output directory: {save_dir}")
    print()

    results = {}
    for config_name in configs:
        result = generate_single_config(
            config_name=config_name,
            num_samples=num_samples,
            save_dir=save_dir,
            seed=seed,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            save_xml=save_xml
        )
        results[config_name] = result

    # Print summary
    total_samples = sum(r['total'] for r in results.values())
    total_train = sum(r['train'] for r in results.values())
    total_val = sum(r['val'] for r in results.values())
    total_test = sum(r['test'] for r in results.values())
    avg_accuracy = sum(r['accuracy'] for r in results.values()) / len(results)

    print()
    print("=" * 50)
    print("DATASET GENERATION COMPLETE")
    print("=" * 50)
    print(f"Total samples: {total_samples}")
    print(f"  Train: {total_train}")
    print(f"  Val:   {total_val}")
    print(f"  Test:  {total_test}")
    print(f"Average solver accuracy: {avg_accuracy:.2%}")
    print(f"Saved to: {save_dir}")
    print("=" * 50)

    return results


if __name__ == "__main__":
    # Quick test
    import argparse

    parser = argparse.ArgumentParser(description="Generate RAVEN dataset")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples per configuration")
    parser.add_argument("--save-dir", type=str, default="./generated_data",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    generate_dataset(
        num_samples=args.num_samples,
        save_dir=args.save_dir,
        seed=args.seed
    )
