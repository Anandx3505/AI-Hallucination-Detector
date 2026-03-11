"""
SelfCheckGPT BERTScore Baseline:
Detects hallucinations by measuring consistency between multiple LLM answers
using cosine similarity of embeddings (BERTScore approximation).

Usage: python selfcheck_bertscore.py --path ../data/ --output_dir ../weights/
"""
import torch, argparse, os, json
import pandas as pd
import numpy as np
import torch.nn.functional as F
from os.path import join as path_join
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix


def compute_consistency_scores(embeddings_per_question):
    """
    For each answer in a question group, compute avg cosine similarity to all other answers.
    High similarity = consistent (likely correct), Low similarity = inconsistent (likely hallucinated).
    """
    scores = []
    for i, emb in enumerate(embeddings_per_question):
        others = [e for j, e in enumerate(embeddings_per_question) if j != i]
        if len(others) == 0:
            scores.append(0.0)
            continue
        others_tensor = torch.stack(others)
        sim = F.cosine_similarity(emb.unsqueeze(0), others_tensor, dim=-1)
        scores.append(sim.mean().item())
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--path", type=str, default="../data/")
    parser.add_argument("--output_dir", type=str, default="../weights/")
    args = parser.parse_args()

    # Fix for running from root directory
    if args.path == "../data/" and not os.path.exists(args.path) and os.path.exists("data/"):
        args.path = "data/"
    if args.output_dir == "../weights/" and not os.path.exists(args.output_dir) and os.path.exists("weights/"):
        args.output_dir = "weights/"

    print("=" * 60)
    print("SELFCHECKGPT BERTSCORE BASELINE")
    print("=" * 60)

    device = torch.device("cpu")

    # Load graph (contains embeddings and labels)
    print("Loading graph...")
    graph = torch.load(path_join(args.path, "graph.pt"), map_location=device, weights_only=False)

    # Load pretrained embedder
    embedder_file = "embedder_act_ReLU_opt_AdamW_lr_0.0001_bs_256_t_0.07.pt"
    in_channels = graph.num_features
    embedder = torch.nn.Sequential(*[torch.nn.Linear(in_channels, in_channels),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(in_channels, 128)])
    embedder.load_state_dict(
        torch.load(path_join(args.output_dir, embedder_file), map_location=device, weights_only=False)["state_dict"])
    embedder.eval()

    print("Computing embeddings...")
    with torch.no_grad():
        all_embeddings = embedder(graph.x)

    # Get labels (sum of one-hot vector gives class: 0=correct, 1-3=hallucinated levels)
    all_labels = torch.sum(graph.y, dim=-1).long().numpy()

    # Process validation set in groups of 5 (each question has 5 answers)
    val_idx = graph.val_idx.numpy()
    val_embeddings = all_embeddings[val_idx]
    val_labels = all_labels[val_idx]

    # Group into question sets of 5
    n_questions = len(val_idx) // 5
    print(f"Processing {n_questions} questions ({len(val_idx)} answers)...")

    all_scores = []
    all_true_labels = []

    for q in range(n_questions):
        start = q * 5
        end = start + 5
        group_embeddings = [val_embeddings[i] for i in range(start, end)]
        group_labels = val_labels[start:end]

        scores = compute_consistency_scores(group_embeddings)
        all_scores.extend(scores)
        all_true_labels.extend(group_labels.tolist())

    all_scores = np.array(all_scores)
    all_true_labels = np.array(all_true_labels)

    # Binary classification: label 0 = correct, label > 0 = hallucinated
    true_binary = (all_true_labels > 0).astype(int)

    # Find optimal threshold
    best_threshold = 0.5
    best_f1 = 0.0
    for threshold in np.arange(0.3, 0.99, 0.01):
        pred_binary = (all_scores < threshold).astype(int)  # low score = hallucinated
        f1 = f1_score(true_binary, pred_binary, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"Optimal threshold: {best_threshold:.2f}")

    # Make predictions with optimal threshold
    pred_binary = (all_scores < best_threshold).astype(int)

    # Multiclass prediction: assign based on score quartiles
    pred_multi = np.zeros_like(all_true_labels)
    sorted_scores = np.sort(all_scores)
    q25 = np.percentile(all_scores, 25)
    q50 = np.percentile(all_scores, 50)
    q75 = np.percentile(all_scores, 75)
    pred_multi[all_scores >= q75] = 0  # highest consistency = correct
    pred_multi[(all_scores >= q50) & (all_scores < q75)] = 1
    pred_multi[(all_scores >= q25) & (all_scores < q50)] = 2
    pred_multi[all_scores < q25] = 3  # lowest consistency = hallucinated

    # Metrics
    final_acc = accuracy_score(all_true_labels, pred_multi)
    final_f1 = f1_score(all_true_labels, pred_multi, average='macro', zero_division=0)

    # Binary recall (hallucinated recall)
    binary_mask = (all_true_labels == 0) | (all_true_labels == 3)
    true_bin_filtered = (all_true_labels[binary_mask] > 0).astype(int)
    pred_bin_filtered = (pred_multi[binary_mask] > 0).astype(int)
    final_bin_recall = recall_score(true_bin_filtered, pred_bin_filtered, zero_division=0)

    conf_mat = confusion_matrix(all_true_labels, pred_multi, labels=[0, 1, 2, 3])

    print("\n" + "=" * 60)
    print("SELFCHECKGPT BERTSCORE RESULTS (Validation Set)")
    print("=" * 60)
    print(f"  Accuracy:        {final_acc:.4f}")
    print(f"  Macro F1:        {final_f1:.4f}")
    print(f"  Binary Recall:   {final_bin_recall:.4f}")
    print(f"\n  Confusion Matrix:\n{conf_mat}")
    print("=" * 60)

    # Save results
    results = {
        "model": "SelfCheckGPT-BERTScore",
        "accuracy": float(final_acc),
        "macro_f1": float(final_f1),
        "binary_recall": float(final_bin_recall),
        "threshold": float(best_threshold),
    }
    results_path = path_join(args.output_dir, "selfcheck_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {results_path}")
