import logging
import math
import time
from typing import Union
import numpy as np
from bounds import (
    BackwardsBounds,
    ForwardConstraints,
    concat_bounds,
)
import torch
import torch.nn as nn

from conv_util import full_toeplitz, transpose
from logger import SimpleLogger
from skip_block import SkipBlock

logger = SimpleLogger("DeepPoly Verifier", logging.NOTSET)


def init_parameters(
    x: BackwardsBounds, method: Union[str, torch.Tensor]
) -> nn.Parameter:
    n = x.ub.size(0)
    if isinstance(method, torch.Tensor):
        params = nn.Parameter(method).requires_grad_()
    elif method == "random":
        params = nn.Parameter(torch.rand(n)).requires_grad_()
    elif method == "zero":
        params = nn.Parameter(torch.zeros(n)).requires_grad_()
    elif method == "ones":
        params = nn.Parameter(torch.ones(n)).requires_grad_()
    elif method == "half":
        params = nn.Parameter(torch.ones(n) * 0.5).requires_grad_()
    elif method == "small":
        params = nn.Parameter(torch.ones(n) * 0.1).requires_grad_()
    elif method == "noisy_half":
        params = nn.Parameter(
            torch.ones(n) * 0.5 + torch.rand(n) * 0.1
        ).requires_grad_()
    elif method == "seesaw":
        params = nn.Parameter((x.ub > -x.lb).float()).requires_grad_()
    elif method == "seesaw_beta":
        params = nn.Parameter((6 - x.lb > x.ub - 6).float()).requires_grad_()
    elif method == "noisy_seesaw":
        params = nn.Parameter(
            (x.ub > -x.lb).float()
            + torch.rand(n) * 0.1 * torch.where(x.ub > -x.lb, -1, 1)
        ).requires_grad_()
    elif method == "noisy_seesaw_beta":
        params = nn.Parameter(
            (6 - x.lb > x.ub - 6).float()
            + torch.rand(n) * 0.1 * torch.where(6 - x.lb > x.ub - 6, -1, 1)
        ).requires_grad_()
    else:
        raise ValueError(f"Unknown method: {method}")
    return params


class VerifyInput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bounds = None

    def forward(self, x: BackwardsBounds) -> BackwardsBounds:
        self.bounds = BackwardsBounds(lb=x.lb.flatten(), ub=x.ub.flatten())
        return self.bounds

    def backsubstitute(self, x: ForwardConstraints) -> None:
        """
        Backsubstitute:
        HERE WE APPLY BOUNDS TO CONSTRAINTS
        In the input layer, as this is the last step in backsubstitution, it is
        important to distinguish negative / positive signs of our constraints.
        positive --> we use the lower bound
        negative --> we use the upper bound

        Args:
            x: Constraints to backsubstitute
            self.bounds: Input (epsilon) Bounds is fixed in init of module

        Returns:
            None (pass by reference)
        """
        # x.uc is false labels, x.lc is true labels
        # first calculate difference between true label and false labels
        # then apply the bounds
        ub = (
            x.uc_bias
            + (torch.where(x.uc > 0, x.uc, 0) * self.bounds.ub).sum(dim=1)
            + (torch.where(x.uc < 0, x.uc, 0) * self.bounds.lb).sum(dim=1)
        )
        lb = (
            x.lc_bias
            + (torch.where(x.lc > 0, x.lc, 0) * self.bounds.lb).sum(dim=1)
            + (torch.where(x.lc < 0, x.lc, 0) * self.bounds.ub).sum(dim=1)
        )
        # x.uc = torch.zeros_like(x.uc)
        # x.lc = torch.zeros_like(x.lc)
        x.input_bounds = BackwardsBounds(ub=ub, lb=lb)


class VerifyLinear(torch.nn.Module):
    def __init__(
        self,
        layer: nn.Linear = None,
        previous: torch.nn.Module = None,
    ):
        super().__init__()
        if layer is not None:
            self.weights = layer.weight.detach()
            self.biases = layer.bias.detach()
            self.out_size = self.weights.size(0)
            self.out_dims = self.weights.size()
        self.previous = previous
        self.bounds = None

    def forward(self, x: BackwardsBounds) -> BackwardsBounds:
        constraints = ForwardConstraints(
            uc=self.weights,
            lc=self.weights,
            uc_bias=self.biases,
            lc_bias=self.biases,
        )
        self.previous.backsubstitute(constraints)
        self.bounds = constraints.input_bounds
        return self.bounds

    def backsubstitute(self, x: ForwardConstraints) -> None:
        """
        Backsubstitute:
        Args:
            x: Constraints to backsubstitute
        """
        x.uc_bias = x.uc_bias + x.uc @ self.biases
        x.lc_bias = x.lc_bias + x.lc @ self.biases
        x.uc = x.uc @ self.weights
        x.lc = x.lc @ self.weights
        self.previous.backsubstitute(x)


class VerifyConv2d(VerifyLinear):
    def __init__(self, layer: nn.Conv2d, previous: torch.nn.Module):
        super().__init__(layer, previous)

        self._stride = layer.stride
        self._padding = layer.padding
        self._in_channels = layer.in_channels
        self._out_channels = layer.out_channels
        self._kernel_size = layer.kernel_size
        self._weights = layer.weight.data
        self._bias = layer.bias.data

        self.previous = previous
        self.init = False

    def _init_linear(self, size: int) -> nn.Linear:
        h, w = int(np.sqrt(size / self._in_channels)), int(
            np.sqrt(size / self._in_channels)
        )
        output_h = (h + 2 * self._padding[0] - self._kernel_size[0]) // self._stride[
            0
        ] + 1
        output_w = (w + 2 * self._padding[1] - self._kernel_size[1]) // self._stride[
            1
        ] + 1
        in_dim = (self._in_channels, h, w)
        out_dim = (self._out_channels, output_h, output_w)
        fc = nn.Linear(in_features=np.prod(in_dim), out_features=np.prod(out_dim))
        weight = transpose(
            full_toeplitz(self._weights, in_dim, self._stride[0], self._padding[0])
        )
        fc.weight = nn.Parameter(weight)
        fc.weight.data = transpose(fc.weight.data)
        bias = torch.repeat_interleave(self._bias, output_w * output_h)
        fc.bias = nn.Parameter(bias)
        fc.bias.data = transpose(fc.bias.data)
        return fc

    def forward(self, x: BackwardsBounds) -> BackwardsBounds:
        if not self.init:
            fc = self._init_linear(x.ub.view(1, -1).size(1))
            super().__init__(fc, self.previous)
            self.init = True
        return super().forward(x)

    def backsubstitute(self, x: ForwardConstraints) -> None:
        if not self.init:
            raise ValueError("Not initialized")
        return super().backsubstitute(x)


class VerifyReLU(torch.nn.Module):
    def __init__(self, layer: nn.Linear, previous: torch.nn.Module):
        super().__init__()
        self.previous = previous
        self.out_size = self.previous.out_size
        self.bounds = None
        self.alpha = None
        self.method = "seesaw"

    def initialize_alpha(self, x: BackwardsBounds, method=None) -> None:
        self.alpha = init_parameters(x, method)

    def forward(self, x: BackwardsBounds) -> BackwardsBounds:
        """
        Propagate

        Args:
            x: Bounds of the previous layer
        """
        if self.alpha is None:
            self.initialize_alpha(x, method=self.method)

        # this is essentially projected gradient descent, right?
        # What about clipped gradient descent?
        clamped_alphas = torch.clamp(self.alpha, min=0, max=1)

        lb, ub = x.lb, x.ub  # get new bounds
        self.slope = torch.clamp(ub / (ub - lb), min=0)

        self.uc_bias = torch.where(lb > 0, 0, -(self.slope * lb))
        self.lc_bias = torch.zeros_like(self.uc_bias)
        self.uc = torch.diag(torch.where(lb > 0, 1.0, self.slope).squeeze())
        self.lc = torch.diag(torch.where(ub < 0, 0, clamped_alphas).squeeze())

        # constraints = ForwardConstraints(
        #     uc=self.uc, lc=self.lc, uc_bias=self.uc_bias, lc_bias=self.lc_bias
        # )
        # self.previous.backsubstitute(constraints)
        # self.bounds = constraints.input_bounds

        # we don't need to backsubstitute here, as we can do it in the next layer
        self.bounds = BackwardsBounds(
            lb=x.lb @ self.lc + self.lc_bias,
            ub=x.ub @ self.uc + self.uc_bias,
        )
        return self.bounds

    def backsubstitute(self, x: ForwardConstraints) -> None:
        """
        Backsubstitute:
        We need to be careful here:
        upper constraints where sign is negative --> use lower bound

        Args:
            x: Constraints to backsubstitute
        """
        x.lc_bias = (
            x.lc_bias
            + torch.where(x.lc > 0, x.lc, 0) @ self.lc_bias
            + torch.where(x.lc < 0, x.lc, 0) @ self.uc_bias
        )
        x.uc_bias = (
            x.uc_bias
            + torch.where(x.uc > 0, x.uc, 0) @ self.uc_bias
            + torch.where(x.uc < 0, x.uc, 0) @ self.lc_bias
        )
        x.uc = (
            torch.where(x.uc > 0, x.uc, 0) @ self.uc
            + torch.where(x.uc < 0, x.uc, 0) @ self.lc
        )
        x.lc = (
            torch.where(x.lc > 0, x.lc, 0) @ self.lc
            + torch.where(x.lc < 0, x.lc, 0) @ self.uc
        )
        # x.lc_bias = x.lc_bias + x.lc @ self.lc_bias
        # x.uc_bias = x.uc_bias + x.uc @ self.uc_bias
        # x.lc = x.lc @ self.lc
        # x.uc = x.uc @ self.uc
        self.previous.backsubstitute(x)

    def reset_parameters(self, method: str) -> None:
        self.alpha = None
        self.method = method

    def get_parameters(self) -> list[torch.Tensor]:
        return [self.alpha]


class VerifyReLU6(torch.nn.Module):
    def __init__(self, previous: torch.nn.Module):
        super().__init__()
        self.previous = previous
        self.out_size = self.previous.out_size
        self.bounds = None
        self.alpha = None
        self.beta = None
        self.method = "seesaw"

    def initialize_alpha(self, x: BackwardsBounds, method=None) -> None:
        self.alpha = init_parameters(x, method)

    def initialize_beta(self, x: BackwardsBounds, method=None) -> None:
        method = method if method != "seesaw" else "seesaw_beta"
        method = method if method != "noisy_seesaw" else "noisy_seesaw_beta"
        self.beta = init_parameters(x, method)

    def forward(self, x: BackwardsBounds) -> BackwardsBounds:
        """
        Propagate

        Args:
            x: Bounds of the previous layer
        """
        if self.alpha is None:
            self.initialize_alpha(x, method=self.method)
        if self.beta is None:
            self.initialize_beta(x, method=self.method)

        lb, ub = x.lb, x.ub  # get new bounds

        # ensure that optimized parameters are in the correct range
        max_alphas = torch.min(
            torch.ones_like(ub), torch.clamp(6 / ub, min=0)
        )  # if ub < 6, this is 1
        max_betas = torch.min(torch.ones_like(lb), 6 / (6 - lb))  # if lb < 0, this is 1

        # this is essentially projected gradient descent, right?
        clamped_alpha = torch.clamp(
            self.alpha, min=torch.zeros_like(ub), max=max_alphas
        )
        clamped_beta = torch.clamp(self.beta, min=torch.zeros_like(lb), max=max_betas)

        # IDEA from Hannes: use same parameter for both bounds

        # calculate slopes
        self.upper_slope = torch.clamp(ub / (ub - lb), min=0, max=1)
        self.lower_slope = torch.clamp((6 - lb) / (ub - lb), min=0, max=1)

        self.lc = torch.diag(
            torch.where(
                lb > 0,
                # no optimization
                torch.where(ub < 6, 1.0, self.lower_slope),
                # optimization
                torch.where(ub < 0, 0, clamped_alpha),
            ).squeeze()
        )
        self.uc = torch.diag(
            torch.where(
                ub < 6,
                # no optimization
                torch.where(lb > 0, 1.0, self.upper_slope),
                # optimization
                torch.where(lb > 6, 0, clamped_beta),
            ).squeeze()
        )

        self.uc_bias = torch.where(
            ub < 6,
            # no optimization (self.upper_slope is 0 when ub < 0)
            torch.where(lb > 0, 0, -(self.upper_slope * lb)),
            # optimization
            torch.where(lb > 6, 6, 6 * (1 - clamped_beta)),
        )
        self.lc_bias = torch.where(
            ub > 6,
            # since lower slope is 0 when lb > 6
            torch.where(lb > 0, 6 - self.lower_slope * ub, 0),
            0.0,
        )

        assert torch.all(self.uc_bias >= 0)
        assert torch.all(self.lc_bias >= 0)

        # constraints = ForwardConstraints(
        #     uc=self.uc, lc=self.lc, uc_bias=self.uc_bias, lc_bias=self.lc_bias
        # )
        # self.previous.backsubstitute(constraints)
        # self.bounds = constraints.input_bounds

        # we don't need to backsubstitute here, as we can do it in the next layer
        self.bounds = BackwardsBounds(
            lb=x.lb @ self.lc + self.lc_bias,
            ub=x.ub @ self.uc + self.uc_bias,
        )
        return self.bounds

    def backsubstitute(self, x: ForwardConstraints) -> None:
        """
        Backsubstitute:
        We need to be careful here:
        upper constraints where sign is negative --> use lower bound

        Args:
            x: Constraints to backsubstitute
        """
        x.lc_bias = (
            x.lc_bias
            + torch.where(x.lc > 0, x.lc, 0) @ self.lc_bias
            + torch.where(x.lc < 0, x.lc, 0) @ self.uc_bias
        )
        x.uc_bias = (
            x.uc_bias
            + torch.where(x.uc > 0, x.uc, 0) @ self.uc_bias
            + torch.where(x.uc < 0, x.uc, 0) @ self.lc_bias
        )
        x.uc = (
            torch.where(x.uc > 0, x.uc, 0) @ self.uc
            + torch.where(x.uc < 0, x.uc, 0) @ self.lc
        )
        x.lc = (
            torch.where(x.lc > 0, x.lc, 0) @ self.lc
            + torch.where(x.lc < 0, x.lc, 0) @ self.uc
        )
        self.previous.backsubstitute(x)

    def reset_parameters(self, method: str) -> None:
        self.alpha = None
        self.beta = None
        self.method = method

    def get_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        return [self.alpha, self.beta]


class VerifyDummy(torch.nn.Module):
    """Dummy class for Skip block input"""

    def __init__(self, previous: torch.nn.Module):
        super().__init__()
        self.previous = previous

    def forward(self, x: BackwardsBounds) -> BackwardsBounds:
        return x

    def backsubstitute(self, x: ForwardConstraints) -> None:
        """brake backsubstitute cycle"""
        if x.to_beginning:
            self.previous.backsubstitute(x)


class VerifySkipMerge(VerifyLinear):
    def __init__(self, previous: torch.nn.Module):
        super().__init__(None, previous)
        self.previous = previous
        self.init = False

    def _init_linear(self, size: int) -> nn.Linear:
        # build linear layer, weights have dim = (out_features,in_features)
        identity_weights = torch.diag(torch.ones(size // 2))
        weights = torch.cat(
            (identity_weights, identity_weights), dim=1
        )  # dim = 1, st. out_features = self.shape[0]
        bias = torch.zeros(size // 2)
        fc = nn.Linear(in_features=size, out_features=size // 2)
        fc.weight = nn.Parameter(weights)
        fc.bias = nn.Parameter(bias)
        return fc

    def forward(self, x: BackwardsBounds) -> BackwardsBounds:
        if not self.init:
            fc = self._init_linear(x.ub.view(1, -1).size(1))
            super().__init__(fc, self.previous)
            self.init = True
        return super().forward(x)

    def backsubstitute(self, x: ForwardConstraints) -> None:
        if not self.init:
            raise ValueError("Not initialized")
        return super().backsubstitute(x)


class VerifySkipSplit(VerifyLinear):
    def __init__(self, previous: torch.nn.Module):
        super().__init__(None, previous)
        self.previous = previous
        self.init = False

    def _init_linear(self, size: int) -> nn.Linear:
        # build linear layer, weights have dim = (out_features,in_features)
        identity_weights = torch.diag(torch.ones(size))
        weights = torch.cat(
            (identity_weights, identity_weights), dim=0
        )  # dim = 0, st. out_features = 2 * self.shape[0]
        bias = torch.zeros(2 * size)
        fc = nn.Linear(in_features=size, out_features=2 * size)
        fc.weight = nn.Parameter(weights)
        fc.bias = nn.Parameter(bias)
        return fc

    def forward(self, x: BackwardsBounds) -> BackwardsBounds:
        if not self.init:
            fc = self._init_linear(x.ub.view(1, -1).size(1))
            super().__init__(fc, self.previous)
            self.init = True
        return super().forward(x)

    def backsubstitute(self, x: ForwardConstraints) -> None:
        if not self.init:
            raise ValueError("Not initialized")
        return super().backsubstitute(x)


class VerifySkipPath(torch.nn.Module):
    """
    Class to verify Skip connections in a network (Hanno et al. 2000)
    """

    def __init__(
        self,
        block: SkipBlock,
        previous_pre_split: torch.nn.Module,
        previous_split: torch.nn.Module,
    ):
        super().__init__()
        self.previous = previous_split
        self.block = block
        self.bounds = None
        self.layers = []

        self.shape = None

        self.layers.append(VerifyDummy(previous_pre_split))
        current_prev = self.layers[-1]
        for layer in block.path:
            if isinstance(layer, nn.Linear):
                self.layers.append(VerifyLinear(layer, current_prev))
            elif isinstance(layer, nn.ReLU):
                self.layers.append(VerifyReLU(layer, current_prev))
            elif isinstance(layer, nn.Conv2d):
                self.layers.append(VerifyConv2d(layer, current_prev))
            elif isinstance(layer, nn.ReLU6):
                self.layers.append(VerifyReLU6(current_prev))
            else:
                raise NotImplementedError(f"Layer {layer} not supported")
            current_prev = self.layers[-1]

        self.path = nn.Sequential(*self.layers)

    def forward(self, x: BackwardsBounds) -> BackwardsBounds:
        """
        Propagate

        Args:
            x: Bounds of the previous layer
        """
        if self.shape is None:
            self.shape = x.ub.shape

        path_connection, skip_connection = x.split_in_half()
        path_connection = self.path(path_connection)
        self.bounds = concat_bounds(path_connection, skip_connection)
        return self.bounds

    def backsubstitute(self, x: ForwardConstraints) -> None:
        """
        Backsubstitute:
        Propagates constraints backward through the skip block and combines them
        with the constraints from the identity (skip) path.

        Args:
            x: ForwardConstraints to backsubstitute
        """

        # 1. split in half [] ForwardConstraints
        path_connection, skip_connection = x.split_in_half()
        # 2. backsubstitute path through path and previous
        path_connection.to_beginning = False
        self.layers[-1].backsubstitute(path_connection)
        # 3. concat splits .cat() ForwardConstraints
        x.uc = torch.cat((path_connection.uc, skip_connection.uc), dim=1)
        x.lc = torch.cat((path_connection.lc, skip_connection.lc), dim=1)
        x.uc_bias = path_connection.uc_bias + x.uc_bias
        x.lc_bias = path_connection.lc_bias + x.lc_bias
        x.to_beginning = True  # backsubstitute to beginning
        self.previous.backsubstitute(x)

    def get_parameters(self) -> list[torch.Tensor]:
        return [
            param
            for layer in self.layers
            if hasattr(layer, "get_parameters")
            for param in layer.get_parameters()
        ]


class DeepPoly(torch.nn.Module):
    def __init__(
        self,
        net: torch.nn.Module,
        true_label: int,
        inputs: torch.Tensor,
        eps: float,
    ):
        super().__init__()

        # build new network using verification layers instead of normal layers
        self.layers = []

        self.layers.append(VerifyInput())

        for layer in net:
            if isinstance(layer, nn.Flatten):
                pass
            elif isinstance(layer, nn.Linear):
                self.layers.append(VerifyLinear(layer, self.layers[-1]))
            elif isinstance(layer, nn.ReLU):
                self.layers.append(VerifyReLU(layer, self.layers[-1]))
            elif isinstance(layer, nn.Conv2d):
                self.layers.append(VerifyConv2d(layer, self.layers[-1]))
            elif isinstance(layer, nn.ReLU6):
                self.layers.append(VerifyReLU6(self.layers[-1]))
            elif isinstance(layer, SkipBlock):
                self.layers.append(VerifySkipSplit(self.layers[-1]))
                self.layers.append(
                    VerifySkipPath(
                        block=layer,
                        previous_pre_split=self.layers[-2],
                        previous_split=self.layers[-1],
                    )
                )
                self.layers.append(VerifySkipMerge(self.layers[-1]))
            else:
                raise NotImplementedError(f"Layer {layer} not supported")

        self.model = nn.Sequential(*self.layers)

        ub_in = torch.Tensor.clamp(inputs + eps, min=0, max=1)
        lb_in = torch.Tensor.clamp(inputs - eps, min=0, max=1)
        input_bounds = BackwardsBounds(ub=ub_in, lb=lb_in)
        self.forward(input_bounds)  # initialize layers

    def forward(self, x: BackwardsBounds) -> BackwardsBounds:
        """
        In every forward pass of the model, we propagate bounds, but also call
        the backsubstitute method to update the constraints multiple times.
        """
        return self.model(x)

    def get_parameters(self) -> list[torch.Tensor]:
        return [
            param
            for layer in self.layers
            if hasattr(layer, "get_parameters")
            for param in layer.get_parameters()
        ]

    def reset_parameters(self, method: str) -> None:
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters(method)
            # if skip block, reset all layers
            if isinstance(layer, VerifySkipPath):
                for l in layer.layers:
                    if hasattr(l, "reset_parameters"):
                        l.reset_parameters(method)

    def verify(self, inputs: torch.Tensor, eps: float, epochs: int, lr: float) -> bool:
        def run_optimization(
            input_bounds: BackwardsBounds, epochs: int, lr: float, index: int = None
        ) -> torch.Tensor:
            params = self.get_parameters()
            opt = torch.optim.Adam(params, lr=lr)

            def lr_lambda(epoch):
                if epoch < epochs * 0.5:
                    return epoch / (epochs * 0.5)
                elif epoch < epochs:
                    return 0.5 * (
                        1 + math.cos(math.pi * (epoch - epochs * 0.5) / (epochs * 0.5))
                    )
                else:
                    return 0.1

            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
            start_time = time.time()
            for i in range(epochs):  # + int(1e6)): TODO: when hand in
                opt.zero_grad()
                final_bound = self.forward(input_bounds).lb
                if index is not None:
                    final_bound = final_bound[index]
                lowest = torch.min(final_bound)
                loss = torch.sum(-torch.clamp(final_bound, max=0))
                if lowest >= 0:
                    logger.debug(f"stopped at iteration: {i}")
                    break
                loss.backward()
                opt.step()
                scheduler.step()
                if i % 10 == 0:
                    logger.debug(
                        f"Epoch {i}, final_bound: {final_bound.detach().numpy()}"
                    )
                if start_time + 60 < time.time():
                    logger.info(f"60s timeout at iteration: {i}")
                    break
            return final_bound

        ub_in = torch.Tensor.clamp(inputs + eps, min=0, max=1)
        lb_in = torch.Tensor.clamp(inputs - eps, min=0, max=1)
        input_bounds = BackwardsBounds(ub=ub_in, lb=lb_in)

        start_time = time.time()
        final_bound = self.forward(input_bounds).lb
        diff = time.time() - start_time
        logger.info(f"Forward pass time: {diff}")
        epochs = min(int(20 // diff), epochs)
        logger.info(f"Using {epochs} epochs and lr={lr}")

        if len(list(self.model.parameters())) != 0:
            final_bound = run_optimization(input_bounds, epochs, lr)
            logger.info(f"Final bound: {final_bound}")

        return torch.min(final_bound) >= 0
