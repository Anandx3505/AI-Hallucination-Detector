"""
GCN Baseline: Train and evaluate a Graph Convolutional Network.
Mirrors the GAT training pipeline but uses GCNConv (no attention).
Usage: python run_gcn.py --path ../data/ --output_dir ../weights/
"""
import torch, argparse, os, json
from torcheval.metrics import (MulticlassConfusionMatrix, MulticlassAccuracy,
                               MulticlassRecall, MulticlassPrecision, BinaryRecall,
                               MulticlassF1Score)
from torch_geometric.utils import remove_isolated_nodes
from os.path import join as path_join

from GCN import GCN
import utils_graph

torch.set_printoptions(profile="full")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--use-cuda", action="store_true", default=False)
    parser.add_argument("--path", type=str, default="../data/")
    parser.add_argument("--output_dir", type=str, default="../weights/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    # Fix for running from root directory
    if args.path == "../data/" and not os.path.exists(args.path) and os.path.exists("data/"):
        args.path = "data/"
    if args.output_dir == "../weights/" and not os.path.exists(args.output_dir) and os.path.exists("weights/"):
        args.output_dir = "weights/"

    utils_graph.set_seed(42)
    device = torch.device("cuda") if args.use_cuda and torch.cuda.is_available() else torch.device("cpu")

    print("=" * 60)
    print("GCN BASELINE")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")

    # Load graph
    print("Loading graph...")
    graph = torch.load(path_join(args.path, "graph.pt"), map_location=device, weights_only=False)
    graph.to(device)

    # Data leakage prevention (same as GAT training)
    train_idx = graph.train_idx
    val_idx = graph.val_idx
    edge_index = graph.edge_index.T

    print("Loading distances...")
    distances = torch.load(path_join(args.path, "distances.pt"), map_location=device, weights_only=False)
    edge_attr = distances[edge_index[:, 0], edge_index[:, 1]]

    train_mask = (torch.isin(edge_index[:, 0], train_idx)) & (torch.isin(edge_index[:, 1], train_idx))
    val_mask = ((torch.isin(edge_index[:, 0], train_idx)) | (torch.isin(edge_index[:, 0], val_idx))) \
                    & ((torch.isin(edge_index[:, 1], train_idx)) | (torch.isin(edge_index[:, 1], val_idx)))

    train_edge_attr = edge_attr.detach().clone()
    train_edge_attr[~train_mask] = 0.
    val_edge_attr = edge_attr.detach().clone()
    val_edge_attr[~val_mask] = 0.

    # Model
    in_channels = graph.num_features
    out_channels = graph.y.shape[1]

    embedder_file = "embedder_act_ReLU_opt_AdamW_lr_0.0001_bs_256_t_0.07.pt"
    embedder = torch.nn.Sequential(*[torch.nn.Linear(in_channels, in_channels),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(in_channels, 128)])
    embedder.load_state_dict(
        torch.load(path_join(args.output_dir, embedder_file), map_location=device, weights_only=False)["state_dict"])

    gcn = GCN(embedder, n_in=in_channels, hid=32, n_classes=out_channels, dropout=0.2)
    gcn.to(device)

    loss_func = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(gcn.parameters(), lr=args.learning_rate, weight_decay=5e-4)

    # Metrics
    acc = MulticlassAccuracy(num_classes=4)
    conf = MulticlassConfusionMatrix(num_classes=4)
    macro_recall = MulticlassRecall(num_classes=4, average="macro")
    macro_precision = MulticlassPrecision(num_classes=4, average="macro")
    macro_f1 = MulticlassF1Score(num_classes=4, average="macro")
    binary_recall = BinaryRecall()

    print(f"\nTraining GCN for {args.epochs} epochs...")
    best_val_f1 = 0.
    best_epoch = 0
    best_state = None

    for i in range(args.epochs):
        # Train
        train_loss, train_out = utils_graph.train_loop(graph, gcn, loss_func, optimizer, train_edge_attr)
        # Val
        val_loss, val_out = utils_graph.val_loop(graph, gcn, loss_func, val_edge_attr)

        y_pred_val = utils_graph.rewrite_labels(val_out[graph.val_idx].sigmoid().round()).long()
        y_val = torch.sum(graph.y[graph.val_idx], dim=-1).long()
        val_f1 = utils_graph.macro_f1(macro_f1, y_pred_val, y_val).item()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = i
            best_state = {k: v.clone() for k, v in gcn.state_dict().items()}

        if (i + 1) % 10 == 0:
            print(f"  Epoch {i+1:3d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

    # Evaluate best model
    print(f"\nBest epoch: {best_epoch + 1} (Val F1: {best_val_f1:.4f})")
    gcn.load_state_dict(best_state)
    gcn.eval()
    with torch.no_grad():
        out = gcn(graph.x, graph.edge_index, val_edge_attr)

    y_pred = utils_graph.rewrite_labels(out[graph.val_idx].sigmoid().round()).long()
    y = torch.sum(graph.y[graph.val_idx], dim=-1).long()

    final_acc = utils_graph.accuracy(acc, y_pred, y).item()
    final_recall = utils_graph.macro_recall(macro_recall, y_pred, y).item()
    final_precision = utils_graph.macro_precision(macro_precision, y_pred, y).item()
    final_f1 = utils_graph.macro_f1(macro_f1, y_pred, y).item()

    binary_mask = torch.logical_or((y == 0), (y == 3))
    y_binary = utils_graph.rewrite_labels_binary(y[binary_mask])
    y_pred_binary = utils_graph.rewrite_labels_binary(y_pred[binary_mask])
    final_bin_recall = utils_graph.binary_recall(binary_recall, y_pred_binary, y_binary).item()

    conf_mat = utils_graph.confusion_matrix(conf, y_pred, y)

    print("\n" + "=" * 60)
    print("GCN RESULTS (Validation Set)")
    print("=" * 60)
    print(f"  Accuracy:        {final_acc:.4f}")
    print(f"  Macro Recall:    {final_recall:.4f}")
    print(f"  Macro Precision: {final_precision:.4f}")
    print(f"  Macro F1:        {final_f1:.4f}")
    print(f"  Binary Recall:   {final_bin_recall:.4f}")
    print(f"\n  Confusion Matrix:\n{conf_mat.long()}")
    print("=" * 60)

    # Save results
    results = {
        "model": "GCN",
        "best_epoch": best_epoch,
        "accuracy": final_acc,
        "macro_recall": final_recall,
        "macro_precision": final_precision,
        "macro_f1": final_f1,
        "binary_recall": final_bin_recall,
    }
    results_path = path_join(args.output_dir, "gcn_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {results_path}")
