import torch
import sys

try:
    path = "weights/GAT_379.pt"
    state_dict = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    print(f"Keys in {path}:")
    for key, value in state_dict.items():
        print(f"{key}: {value.shape}")
except Exception as e:
    print(e)
