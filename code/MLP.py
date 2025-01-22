import torch.nn as nn
from typing import List


def MLP(dims: List[int], act: str) -> nn.Sequential:
    """
    Create a multi-layer perceptron with the given dimensions and activation function.
    """
    assert len(dims) > 1, "At least two dimensions are required to create MLP"
    in_dim = dims[0]
    layers = []
    for out_dim in dims[1:-1]:
        layers.append(nn.Linear(in_dim, out_dim))
        if act == "relu":
            layers.append(nn.ReLU())
        elif act == "relu6":
            layers.append(nn.ReLU6())
        elif act == "identity":
            pass
        else:
            raise NotImplementedError(f"Activation function {act} not implemented")
        in_dim = out_dim
    layers.append(nn.Linear(in_dim, dims[-1]))
    return nn.Sequential(*layers)
