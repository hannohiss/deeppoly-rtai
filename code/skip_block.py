from torch import Tensor
from torch import nn as nn
from typing import Optional
from MLP import MLP


class SkipBlock(nn.Module):
    def __init__(
        self,
        path: Optional[nn.Sequential] = None,
    ) -> None:
        super(SkipBlock, self).__init__()
        self.path = path

    def lazy_init(self, hidden_dim: int, num_layers: int, act: str) -> None:
        dims = [
            hidden_dim,
        ] * (num_layers + 1)
        self.path = MLP(dims, act)

    def forward(self, x: Tensor) -> Tensor:
        out = self.path(x) + x
        return out
