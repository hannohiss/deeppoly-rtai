"""
Contains the classes for the (absolute) bounds and (relative) constraints of the verification.
"""

import torch  # type: ignore


class BackwardsBounds:
    def __init__(self, ub: torch.Tensor, lb: torch.Tensor):
        self.ub = ub
        self.lb = lb

    def split_in_half(self):
        path = BackwardsBounds(
            self.ub[: self.ub.size(0) // 2],
            self.lb[: self.lb.size(0) // 2],
        )
        skip = BackwardsBounds(
            self.ub[self.ub.size(0) // 2 :],
            self.lb[self.lb.size(0) // 2 :],
        )
        return path, skip


def concat_bounds(
    path: BackwardsBounds,
    skip: BackwardsBounds,
) -> BackwardsBounds:
    ub = torch.cat((path.ub, skip.ub), dim=0)
    lb = torch.cat((path.lb, skip.lb), dim=0)
    return BackwardsBounds(ub, lb)


class ForwardConstraints:
    def __init__(
        self,
        uc: torch.Tensor,
        lc: torch.Tensor,
        uc_bias: torch.Tensor,
        lc_bias: torch.Tensor,
        input_bounds: BackwardsBounds = None,
        to_beginning: bool = True,
    ):
        self.uc = uc
        self.lc = lc
        self.uc_bias = uc_bias
        self.lc_bias = lc_bias
        self.input_bounds = input_bounds
        self.to_beginning = to_beginning

    def split_in_half(self):
        path = ForwardConstraints(
            self.uc[:, : self.uc.size(1) // 2],
            self.lc[:, : self.lc.size(1) // 2],
            torch.zeros_like(self.uc_bias),
            torch.zeros_like(self.lc_bias),
        )
        skip = ForwardConstraints(
            self.uc[:, self.uc.size(1) // 2 :],
            self.lc[:, self.lc.size(1) // 2 :],
            torch.zeros_like(self.uc_bias),
            torch.zeros_like(self.lc_bias),
        )
        return path, skip
