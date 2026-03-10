import torch, argparse
from torcheval.metrics import MulticlassAUPRC, MulticlassConfusionMatrix, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, BinaryRecall, MulticlassF1Score
from torch_geometric.utils import remove_isolated_nodes

from GAT import GAT
import utils_graph
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from os.path import join as path_join
torch.set_printoptions(profile="full")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # command line args for specifying the situation
    parser.add_argument("--use-cuda", action="store_true", default=False,
                        help="Use GPU acceleration if available")
    parser.add_argument("--path", type=str, default="../data/",
                        help="Path to the data folder")
    parser.add_argument("--load-model", type=str, default="../weights/GAT_379.pt",
                        help="GAT model weights to use.")
    parser.add_argument("--mode", type=str, default="val",
                        choices=["train", "val", "test"],
                        help="Mode for evaluation")
    args = parser.parse_args()

    # Fix for running from root directory
    import os
    if args.path == "../data/" and not os.path.exists(args.path) and os.path.exists("data/"):
        args.path = "data/"
    if args.load_model.startswith("../weights/") and not os.path.exists("../weights/") and os.path.exists("weights/"):
        args.load_model = args.load_model.replace("../weights/", "weights/")

    # for reproducibility
    utils_graph.set_seed(42)
    
    print("STARTING...  setup:")
    print(args)
    print("-" * 120)
    print("\n" * 2)

    # Some paramaters
    if args.use_cuda:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")

    # Load graph
    graph = torch.load(path_join(args.path, "graph.pt"), map_location=device, weights_only=False)
    graph.to(device)

    # Removing isolated nodes
    isolated = (remove_isolated_nodes(graph["edge_index"])[2] == False).sum(dim=0).item()
    print(f"Number of isolated nodes = {isolated}\n")

    ######## DATA LEAKAGE PREVENTION WITH SPECIFIC EDGE ATTRIBUTES ########
    train_idx = graph.train_idx # [t, ]
    val_idx = graph.val_idx # [v, ]
    edge_index = graph.edge_index.T # [N, 2]; (i, j) node pairs as rows

    # load the distances
    distances = torch.load(path_join(args.path, "distances.pt"), map_location=device, weights_only=False)
    # get the distances corresponding to the nodes that have edges
    edge_attr = distances[edge_index[:, 0], edge_index[:, 1]] # [N, ]

    # these are all the edges between only train nodes
    train_mask = (torch.isin(edge_index[:, 0], train_idx)) & (torch.isin(edge_index[:, 1], train_idx))
    # these are all the edges only between train and/or validation
    val_mask = ((torch.isin(edge_index[:, 0], train_idx)) | (torch.isin(edge_index[:, 0], val_idx))) \
                    & ((torch.isin(edge_index[:, 1], train_idx)) | (torch.isin(edge_index[:, 1], val_idx)))

    # make all non train attributes zero
    train_edge_attr = edge_attr.detach().clone()
    train_edge_attr[~train_mask] = 0.    

    # make all non train and validation attributes zero
    val_edge_attr = edge_attr.detach().clone()
    val_edge_attr[~val_mask] = 0.
    ######## DATA LEAKAGE PREVENTION WITH SPECIFIC EDGE ATTRIBUTES ########

    # Get the indices for slice to be evaluated
    if args.mode == "train":
        idx = graph.train_idx
        edge_attr = train_edge_attr
    elif args.mode == "val":
        idx = graph.val_idx
        edge_attr = val_edge_attr
    elif args.mode == "test":
        idx = graph.test_idx
    else:
        raise ValueError("Invalid mode")

    # Define model
    in_channels = graph.num_features
    out_channels = graph.y.shape[1] # number of columns
    hidden_channels = 32
    in_head = 2
    dropout = 0.

    embedder = torch.nn.Sequential(*[torch.nn.Linear(in_channels, in_channels), torch.nn.ReLU(), torch.nn.Linear(in_channels, 128)])
    gat = GAT(embedder, n_in=in_channels, hid=hidden_channels,
                     in_head=in_head, 
                     n_classes=out_channels, dropout=dropout)
    state_dict = torch.load(args.load_model, map_location=device, weights_only=False)["state_dict"]
    
    # Patch for compatibility with newer torch_geometric
    if "conv.lin_src.weight" in state_dict:
        print("Patching state dict: renaming conv.lin_src.weight to conv.lin.weight")
        state_dict["conv.lin.weight"] = state_dict.pop("conv.lin_src.weight")
        if "conv.lin_dst.weight" in state_dict:
            state_dict.pop("conv.lin_dst.weight")
            
    gat.load_state_dict(state_dict)
    gat.to(device)

    # Cross entropy loss -- w/ logits
    loss_func = torch.nn.BCEWithLogitsLoss()

    # Evaluation metrics
    acc = MulticlassAccuracy(num_classes=4)
    conf = MulticlassConfusionMatrix(num_classes=4)
    macro_recall = MulticlassRecall(num_classes=4, average="macro")
    macro_precision = MulticlassPrecision(num_classes=4, average="macro")
    macro_f1 = MulticlassF1Score(num_classes=4, average="macro")
    binary_recall = BinaryRecall()
    macro_AUPRC = MulticlassAUPRC(num_classes=4, average="macro")

    gat.eval()
    with torch.no_grad():
        model_output = gat(graph.x, graph.edge_index, edge_attr)

    loss = loss_func(model_output[idx], graph.y[idx].float()).item()

    # Rewrite the labels from vectors to integers
    y_pred, y = utils_graph.rewrite_labels(model_output[idx].sigmoid().round()).long(), torch.sum(graph.y[idx], dim=-1).long()

    # Valuation confusion matrices
    conf_mat = utils_graph.confusion_matrix(conf, y_pred, y)

    # Valuation accuracy
    accuracy = utils_graph.accuracy(acc, y_pred, y)

    # Valuation macro recall
    m_recall = utils_graph.macro_recall(macro_recall, y_pred, y)

    # Valuation macro precision
    m_precision = utils_graph.macro_precision(macro_precision, y_pred, y)

    # Valuation macro F1
    m_f1 = utils_graph.macro_f1(macro_f1, y_pred, y)

    # Valuation macro area under the precision-recall curve
    m_AUPRC = utils_graph.macro_AUPRC(macro_AUPRC, y_pred, y)

    # One frame agreement
    ofa = utils_graph.k_frame_agreement(y_pred, y, k=1)

    # Train and valuation binary accuracy
    binary_mask = torch.logical_or((y == 0), (y == 3))
    y_binary = utils_graph.rewrite_labels_binary(y[binary_mask])
    y_pred_binary = utils_graph.rewrite_labels_binary(y_pred[binary_mask])
    b_recall = utils_graph.binary_recall(binary_recall, y_pred_binary, y_binary)

    # Print valuation loss
    print(f"Loss: {loss}")
    # Print train and valuation confusion matrices
    print(f"Confusion matrix:\n\t{conf_mat.long()}")
    # Print valuation accuracy
    print(f"Accuracy: {accuracy.item()}")
    # Print valuation macro recall
    print(f"Macro recall: {m_recall.item()}")
    # Print valuation macro precision
    print(f"Macro precision: {m_precision.item()}")
    # Print valuation macro F1
    print(f"Macro F1: {m_f1.item()}")
    # Print valuation binary accuracy
    print(f"Binary recall: {b_recall.item()}")
    # Print valuation one frame agreement
    print(f"One frame agreement: {ofa}")
    # Print valuation macro AUPRC
    print(f"Macro AUPRC: {m_AUPRC.item()}")

    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat.cpu().numpy(), annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    
    # Create images directory if it doesn't exist
    os.makedirs(path_join(args.path, "../images"), exist_ok=True)
    save_path = path_join(args.path, "../images/confusion_matrix.png")
    plt.savefig(save_path)
    print(f"Confusion matrix plot saved to {save_path}")
 