import torch
from graph.GAT import GAT

embedder = torch.nn.Sequential(*[torch.nn.Linear(768, 768), torch.nn.ReLU(), torch.nn.Linear(768, 128)])
model = GAT(embedder)
print("Current model keys:")
for key, value in model.state_dict().items():
    print(f"{key}: {value.shape}")
