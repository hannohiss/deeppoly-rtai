import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
import numpy as np
from scipy import linalg
from scipy.linalg import toeplitz
import torch

import deeppoly
from networks import get_network
from utils.loading import parse_spec
from logger import SimpleLogger
from typing import Tuple

logger = SimpleLogger("Main Analizer", logging.NOTSET)

DEVICE = "cpu"


def analyze(
    net: torch.nn.Module,
    inputs: torch.Tensor,
    eps: float,
    true_label: int,
    max_epochs: int = 1,
    lr: float = 0.0,
) -> bool:
    logger.info(
        f"Analyzing network {net.__class__.__name__} with DeepPoly using max_epochs={max_epochs}, lr={lr}..."
    )
    net.zero_grad()

    # Last layer eliding
    diff_layer = nn.Linear(10, 9)
    diff_layer.weight.data = (
        torch.eye(10)[true_label] - torch.eye(10)[torch.arange(10) != true_label]
    )
    diff_layer.bias.data = torch.zeros(9)
    diff_model = nn.Sequential(*list(net.children()), diff_layer)

    verifier = deeppoly.DeepPoly(net=diff_model, true_label=true_label, inputs=inputs, eps=eps)  # type: ignore
    return verifier.verify(inputs, eps, epochs=max_epochs, lr=lr)


def main():
    NETZ = []
    CONV_SIZES = []
    parser = argparse.ArgumentParser(
        description="Neural network verification using DeepPoly relaxation."
    )
    parser.add_argument(
        "--net",
        type=str,
        choices=[
            "fc_linear",
            "fc_base",
            "fc_w",
            "fc_d",
            "fc_dw",
            "fc6_base",
            "fc6_w",
            "fc6_d",
            "fc6_dw",
            "conv_linear",
            "conv_base",
            "conv6_base",
            "conv_d",
            "skip",
            "skip_large",
            "skip6",
            "skip6_large",
        ],
        required=True,
        help="Neural network architecture which is supposed to be verified.",
    )
    parser.add_argument("--spec", type=str, required=True, help="Test case to verify.")
    args = parser.parse_args()

    true_label, dataset, image, eps = parse_spec(args.spec)

    if dataset == "mnist":
        in_ch, in_dim, num_class = 1, 28, 10
    elif dataset == "cifar10":
        in_ch, in_dim, num_class = 3, 32, 10
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    net = get_network(
        args.net,
        in_ch=in_ch,
        in_dim=in_dim,
        num_class=num_class,
        weight_path=f"models/{dataset}_{args.net}.pt",
    ).to(DEVICE)

    image = image.to(DEVICE)
    out = net(image.unsqueeze(0))

    pred_label = out.max(dim=1)[1].item()
    assert pred_label == true_label

    # torch set seed
    torch.manual_seed(0)

    verified = False

    verified = analyze(
        net, image, eps, true_label, lr=0.1, max_epochs=500
    )  # adjust max_epochs to your liking for easier debugging

    if verified:
        print("verified")
    else:
        print("not verified")


if __name__ == "__main__":
    main()
