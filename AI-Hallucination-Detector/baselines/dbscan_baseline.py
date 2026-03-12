"""
DBSCAN Baseline:
Detects hallucinations using density-based clustering on SimCLR embeddings.
Cluster members = correct, noise points = hallucinated.

Usage: python dbscan_baseline.py --path ../data/ --output_dir ../weights/
"""
import torch, argparse, os, json
import numpy as np
from os.path import join as path_join
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--path", type=str, default="../data/")
    parser.add_argument("--output_dir", type=str, default="../weights/")
    parser.add_argument("--eps", type=float, default=0.15,
                        help="DBSCAN epsilon (distance threshold)")
    parser.add_argument("--min-samples", type=int, default=2,
                        help="DBSCAN minimum samples for a core point")
    args = parser.parse_args()

    # Fix for running from root directory
    if args.path == "../data/" and not os.path.exists(args.path) and os.path.exists("data/"):
        args.path = "data/"
    if args.output_dir == "../weights/" and not os.path.exists(args.output_dir) and os.path.exists("weights/"):
        args.output_dir = "weights/"

    print("=" * 60)
    print("DBSCAN BASELINE")
    print("=" * 60)

    device = torch.device("cpu")

    # Load graph
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
        all_embeddings = embedder(graph.x).numpy()

    # Get labels
    all_labels = torch.sum(graph.y, dim=-1).long().numpy()

    # Validation set
    val_idx = graph.val_idx.numpy()
    val_embeddings = all_embeddings[val_idx]
    val_labels = all_labels[val_idx]

    # Process per-question groups of 5
    n_questions = len(val_idx) // 5
    print(f"Processing {n_questions} questions ({len(val_idx)} answers)...")
    print(f"DBSCAN params: eps={args.eps}, min_samples={args.min_samples}")

    all_preds = []

    for q in range(n_questions):
        start = q * 5
        end = start + 5
        group_embeddings = val_embeddings[start:end]
        group_labels = val_labels[start:end]

        # Run DBSCAN on this question's 5 answers
        db = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric='cosine')
        cluster_labels = db.fit_predict(group_embeddings)

        # Map DBSCAN results to hallucination classes
        for i, cl in enumerate(cluster_labels):
            if cl == -1:
                # Noise = hallucinated (class 3)
                all_preds.append(3)
            else:
                # Count how many points in this cluster
                cluster_size = np.sum(cluster_labels == cl)
                if cluster_size >= 3:
                    all_preds.append(0)  # Large cluster = correct
                elif cluster_size == 2:
                    all_preds.append(1)  # Small cluster = partially correct
                else:
                    all_preds.append(2)  # Singleton = more hallucinated

    all_preds = np.array(all_preds)

    # Metrics
    final_acc = accuracy_score(val_labels, all_preds)
    final_f1 = f1_score(val_labels, all_preds, average='macro', zero_division=0)

    # Binary recall
    binary_mask = (val_labels == 0) | (val_labels == 3)
    true_bin = (val_labels[binary_mask] > 0).astype(int)
    pred_bin = (all_preds[binary_mask] > 0).astype(int)
    final_bin_recall = recall_score(true_bin, pred_bin, zero_division=0)

    conf_mat = confusion_matrix(val_labels, all_preds, labels=[0, 1, 2, 3])

    print("\n" + "=" * 60)
    print("DBSCAN RESULTS (Validation Set)")
    print("=" * 60)
    print(f"  Accuracy:        {final_acc:.4f}")
    print(f"  Macro F1:        {final_f1:.4f}")
    print(f"  Binary Recall:   {final_bin_recall:.4f}")
    print(f"\n  Confusion Matrix:\n{conf_mat}")
    print("=" * 60)

    # Try multiple eps values to find optimal
    print("\n  Sweep over eps values:")
    best_eps = args.eps
    best_sweep_f1 = final_f1
    for eps_val in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        preds_sweep = []
        for q in range(n_questions):
            start = q * 5
            end = start + 5
            group_emb = val_embeddings[start:end]
            db_s = DBSCAN(eps=eps_val, min_samples=args.min_samples, metric='cosine')
            cl_s = db_s.fit_predict(group_emb)
            for cl in cl_s:
                if cl == -1:
                    preds_sweep.append(3)
                else:
                    cs = np.sum(cl_s == cl)
                    if cs >= 3:
                        preds_sweep.append(0)
                    elif cs == 2:
                        preds_sweep.append(1)
                    else:
                        preds_sweep.append(2)
        preds_sweep = np.array(preds_sweep)
        sweep_acc = accuracy_score(val_labels, preds_sweep)
        sweep_f1 = f1_score(val_labels, preds_sweep, average='macro', zero_division=0)
        marker = " <-- BEST" if sweep_f1 > best_sweep_f1 else ""
        if sweep_f1 > best_sweep_f1:
            best_sweep_f1 = sweep_f1
            best_eps = eps_val
        print(f"    eps={eps_val:.2f} | Acc={sweep_acc:.4f} | F1={sweep_f1:.4f}{marker}")

    # Save results
    results = {
        "model": "DBSCAN",
        "accuracy": float(final_acc),
        "macro_f1": float(final_f1),
        "binary_recall": float(final_bin_recall),
        "eps": args.eps,
        "best_eps": float(best_eps),
    }
    results_path = path_join(args.output_dir, "dbscan_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {results_path}")
