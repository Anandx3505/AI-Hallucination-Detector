"""
Compare All Baselines: GAT vs GCN vs SelfCheckGPT vs DBSCAN

Usage: python compare_baselines.py
"""
import json, os

WEIGHTS_DIR = "weights"

# GAT results (from evaluate_graph.py run)
gat_results = {
    "model": "GAT (Graph Attention)",
    "accuracy": 0.8648,
    "macro_f1": 0.7268,
    "binary_recall": 0.9513,
}

def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

if __name__ == "__main__":
    print("=" * 75)
    print("  COMPARATIVE ANALYSIS: Hallucination Detection Baselines")
    print("=" * 75)

    results = [gat_results]

    # Load other results
    gcn = load_json(os.path.join(WEIGHTS_DIR, "gcn_results.json"))
    if gcn:
        results.append(gcn)
    else:
        print("  [!] GCN results not found. Run: python graph/run_gcn.py")

    selfcheck = load_json(os.path.join(WEIGHTS_DIR, "selfcheck_results.json"))
    if selfcheck:
        results.append(selfcheck)
    else:
        print("  [!] SelfCheck results not found. Run: python baselines/selfcheck_bertscore.py")

    dbscan = load_json(os.path.join(WEIGHTS_DIR, "dbscan_results.json"))
    if dbscan:
        results.append(dbscan)
    else:
        print("  [!] DBSCAN results not found. Run: python baselines/dbscan_baseline.py")

    print()
    # Header
    print(f"  {'Model':<30} {'Accuracy':>10} {'Macro F1':>10} {'Bin. Recall':>12}")
    print("  " + "-" * 65)

    for r in results:
        name = r.get("model", "Unknown")
        acc = r.get("accuracy", 0)
        f1 = r.get("macro_f1", 0)
        br = r.get("binary_recall", 0)
        print(f"  {name:<30} {acc:>10.4f} {f1:>10.4f} {br:>12.4f}")

    print("  " + "-" * 65)

    # Winner
    best = max(results, key=lambda x: x.get("macro_f1", 0))
    print(f"\n  Best model by Macro F1: {best['model']} ({best['macro_f1']:.4f})")

    print()
    print("=" * 75)
    print("  INTERPRETATION")
    print("=" * 75)
    print("  - If GAT >> GCN:   Attention mechanism is essential")
    print("  - If GAT >> DBSCAN: Graph structure adds value beyond clustering")
    print("  - If GAT >> SelfCheck: Message passing beats simple scoring")
    print("=" * 75)
