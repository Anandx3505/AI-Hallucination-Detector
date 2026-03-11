import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) baseline model.
    Mirrors the GAT architecture but uses isotropic GCNConv (no attention).
    """

    def __init__(self, embedder, n_in=768, hid=32, n_classes=3, dropout=0.):
        super(GCN, self).__init__()

        if embedder[0].in_features != n_in:
            raise ValueError("The embedder does not have the correct input dimension.")

        self.embedder = embedder
        self.linear = Linear(embedder[-1].out_features, hid)
        self.conv = GCNConv(hid, n_classes)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.embedder(x)
        x = self.linear(x)
        # GCN uses unweighted (binary) edges — no edge_weight
        x = self.conv(x, edge_index=edge_index)
        return x
