import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import math
from skip_block import SkipBlock
from MLP import MLP


def fc_model(
    act: str,
    layers: List[int],
    in_ch: int = 1,
    in_dim: int = 28,
    num_class: int = 10,
) -> nn.Sequential:
    """
    Create a fully connected network model with the given activation functions, input channels, input dimension and number of classes.

    Args:
        act: List of strings containing the activation functions for each layer, or a single string for the activation function of all layers;
        layers: List of integers containing the number of neurons for each layer;
        in_ch: Number of input channels
        in_dim: Input dimension
        num_class: Number of classes
    """
    model_layers = []
    # Flatten the input and calculate the input dimension as 1D
    model_layers.append(nn.Flatten())
    in_dim = in_ch * in_dim**2
    dims = (
        [
            in_dim,
        ]
        + layers
        + [
            num_class,
        ]
    )
    model_layers += [*MLP(dims, act)]
    return nn.Sequential(*model_layers)


def conv_model(
    act: str,
    convolutions: List[Tuple[int, int, int, int]],
    layers: List[int],
    in_ch: int = 1,
    in_dim: int = 28,
    num_class: int = 10,
) -> nn.Sequential:
    """
    Create a convolutional network model with the given activation functions, convolutional hyperparameters, input channels, input dimension and number of classes.

    Args:
        act: List of strings containing the activation functions for each layer, or a single string for the activation function of all layers;
        convolutions: List of tuples containing the number of output channels, kernel size, stride, padding and slope of the activation function for each convolutional layer;
        layers: List of integers containing the number of neurons for each layer of the MLP following the convolutional layers;
        in_ch: Number of input channels
        in_dim: Input dimension
        num_class: Number of classes
    """
    model_layers = []
    img_dim = in_dim
    prev_channels = in_ch

    for n_channels, kernel_size, stride, padding in convolutions:
        model_layers += [
            nn.Conv2d(
                prev_channels, n_channels, kernel_size, stride=stride, padding=padding
            ),
        ]
        if act == "relu":
            model_layers.append(nn.ReLU())
        elif act == "relu6":
            model_layers.append(nn.ReLU6())
        elif act == "identity":
            pass
        else:
            raise NotImplementedError(f"Activation function {act} not implemented")
        prev_channels = n_channels
        img_dim = math.floor((img_dim - kernel_size + 2 * padding) / stride) + 1

    model_layers.append(nn.Flatten())
    prev_fc_size = prev_channels * img_dim * img_dim
    dims = (
        [
            prev_fc_size,
        ]
        + layers
        + [
            num_class,
        ]
    )
    model_layers += [*MLP(dims, act)]
    return nn.Sequential(*model_layers)


def skip_model(
    act: str,
    num_skip_blocks: int = 2,
    num_layers_each_block: int = 2,
    skip_dim: int = 50,
    in_ch: int = 1,
    in_dim: int = 28,
    num_class: int = 10,
) -> nn.Sequential:
    """
    Create a fully connected model with skip connections using the given base model, number of skip blocks, input channels, input dimension and number of classes.

    Args:
        num_skip_blocks: Number of skip blocks to use
        skip_dim: Number of neurons in the skip blocks
        in_ch: Number of input channels
        in_dim: Input dimension
        num_class: Number of classes
    """
    model_layers = []
    # Flatten the input and calculate the input dimension as 1D
    model_layers.append(nn.Flatten())
    in_dim = in_ch * in_dim**2
    # A first affine layer to match the skip block input dimension
    model_layers.append(nn.Linear(in_dim, skip_dim))
    # Create the skip blocks
    for _ in range(num_skip_blocks):
        block = SkipBlock()
        block.lazy_init(skip_dim, num_layers_each_block, act)
        model_layers.append(block)
    # A final affine layer to match the output dimension
    model_layers.append(nn.Linear(skip_dim, num_class))
    return nn.Sequential(*model_layers)


def get_network(
    name: str,
    in_ch: int = 1,
    in_dim: int = 28,
    num_class: int = 10,
    weight_path: str = "",
    device: str = "cpu",
) -> nn.Sequential:
    """Get network with specific architecture in eval mode.

    Args:
        name (str): Base network architecture
        dataset (str, optional): Dataset used (some architectures have a model for MNIST and
        CIFAR10). Defaults to "mnist".
        weight_path (str, optional): Path to load model weights from. Defaults to "".
        device (str, optional): Device to load model on. Defaults to "cpu".

    Returns:
        nn.Sequential: Resulting model
    """

    model: Optional[nn.Sequential] = None

    if name == "fc_linear":  # linear network
        model = fc_model(
            "identity",
            layers=[
                50,
            ]
            * 3,
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=num_class,
        )
    elif name == "fc_base":  # RELU -- Base
        model = fc_model(
            act="relu",
            layers=[
                50,
            ]
            * 3,
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=num_class,
        )
    elif name == "fc_w":  # RELU -- Wide
        model = fc_model(
            act="relu",
            layers=[
                100,
            ]
            * 3,
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=num_class,
        )
    elif name == "fc_d":  # ReLU -- Deep
        model = fc_model(
            act="relu",
            layers=[
                50,
            ]
            * 7,
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=num_class,
        )
    elif name == "fc_dw":  # ReLU -- Deep and Wide
        model = fc_model(
            act="relu",
            layers=[
                100,
            ]
            * 7,
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=num_class,
        )
    elif name == "fc6_base":  # ReLU6 -- Base
        model = fc_model(
            act="relu6",
            layers=[
                50,
            ]
            * 3,
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=num_class,
        )
    elif name == "fc6_w":  # ReLU6 -- Wide
        model = fc_model(
            act="relu6",
            layers=[
                100,
            ]
            * 3,
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=num_class,
        )
    elif name == "fc6_d":  # ReLU6 -- Deep
        model = fc_model(
            act="relu6",
            layers=[
                50,
            ]
            * 7,
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=num_class,
        )
    elif name == "fc6_dw":  # ReLU6 -- Deep and Wide
        model = fc_model(
            act="relu6",
            layers=[
                100,
            ]
            * 7,
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=num_class,
        )
    elif name == "conv_linear":  # linear conv network
        model = conv_model(
            act="identity",
            convolutions=[(16, 3, 2, 1), (8, 3, 2, 1)],
            layers=[50],
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=num_class,
        )
    elif name == "conv_base":  # Conv -- Base
        model = conv_model(
            act="relu",
            convolutions=[(16, 3, 2, 1), (8, 3, 2, 1)],
            layers=[50],
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=num_class,
        )
    elif name == "conv_w":  # Conv -- Wide
        model = conv_model(
            act="relu",
            convolutions=[(64, 5, 2, 2), (32, 3, 2, 1)],
            layers=[50],
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=num_class,
        )
    elif name == "conv_d":  # Conv -- Deep
        model = conv_model(
            act="relu",
            convolutions=[(16, 3, 2, 1), (8, 3, 2, 1), (8, 3, 1, 1), (8, 3, 1, 1)],
            layers=[
                50,
            ],
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=num_class,
        )
    elif name == "conv_dw":  # Conv -- Deep and Wide
        model = conv_model(
            act="relu",
            convolutions=[(64, 5, 2, 2), (32, 3, 2, 1), (16, 3, 1, 1), (8, 3, 1, 1)],
            layers=[50],
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=num_class,
        )
    elif name == "conv6_base":  # Conv6 -- Base
        model = conv_model(
            act="relu6",
            convolutions=[(16, 3, 2, 1), (8, 3, 2, 1)],
            layers=[50],
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=num_class,
        )
    elif name == "conv6_w":  # Conv6 -- Wide
        model = conv_model(
            act="relu6",
            convolutions=[(64, 5, 2, 2), (32, 3, 2, 1)],
            layers=[50],
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=num_class,
        )
    elif name == "conv6_d":  # Conv6 -- Deep
        model = conv_model(
            act="relu6",
            convolutions=[(16, 3, 2, 1), (8, 3, 2, 1), (8, 3, 1, 1), (8, 3, 1, 1)],
            layers=[50],
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=num_class,
        )
    elif name == "conv6_dw":  # Conv6 -- Deep and Wide
        model = conv_model(
            act="relu6",
            convolutions=[(64, 5, 2, 2), (32, 3, 2, 1), (16, 3, 1, 1), (8, 3, 1, 1)],
            layers=[50],
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=num_class,
        )
    elif name == "skip":  # Skip -- Base
        model = skip_model(
            act="relu",
            num_skip_blocks=2,
            num_layers_each_block=2,
            skip_dim=50,
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=num_class,
        )
    elif name == "skip_large":
        model = skip_model(
            act="relu",
            num_skip_blocks=4,
            num_layers_each_block=3,
            skip_dim=50,
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=num_class,
        )
    elif name == "skip6":
        model = skip_model(
            act="relu6",
            num_skip_blocks=2,
            num_layers_each_block=2,
            skip_dim=50,
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=num_class,
        )
    elif name == "skip6_large":
        model = skip_model(
            act="relu6",
            num_skip_blocks=4,
            num_layers_each_block=3,
            skip_dim=50,
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=num_class,
        )
    else:
        raise NotImplementedError(f"Network {name} not implemented")

    assert model is not None, f"Model is None for {name}"

    if len(weight_path) > 0:
        model.load_state_dict(torch.load(weight_path, map_location="cpu"))

    model.to(device)
    model.eval()

    return model
