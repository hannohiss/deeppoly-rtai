import numpy as np
from scipy.linalg import toeplitz
import torch


def transpose(x: torch.Tensor) -> torch.Tensor:
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))


def padding_matrix(h, w, pad):

    p = pad
    pad_w = w + 2 * p
    pad_h = h + 2 * p
    new_size = pad_h * pad_w

    padd = np.zeros((new_size, h * w))

    for row in range(h):
        for col in range(w):
            o = row * w + col
            pad_row = (row + p) * pad_w + col + p
            padd[pad_row, o] = 1
    return padd


def one_toeplitz(weights_m, input_size):

    in_h, in_w = input_size
    w_h, w_w = weights_m.shape
    out_h = in_h - w_h + 1
    out_w = in_w - w_w + 1

    # Create the first row of the Toeplitz matrix for each weights_m row
    toeplitz_blocks = [
        toeplitz(c=(row[0], *np.zeros(in_w - w_w)), r=(*row, *np.zeros(in_w - w_w)))
        for row in weights_m
    ]

    W_conv = np.zeros(
        (out_h, toeplitz_blocks[0].shape[0], in_h, toeplitz_blocks[0].shape[1])
    )

    for i in range(out_h):
        for j in range(in_h):
            if j >= i and j < i + len(toeplitz_blocks):
                block_idx = j - i
                W_conv[i, :, j, :] = toeplitz_blocks[block_idx]

    return W_conv.reshape(
        out_h * toeplitz_blocks[0].shape[0], in_h * toeplitz_blocks[0].shape[1]
    )


def full_toeplitz(weights_m, input_dim, stride, pad):

    _, h, w = input_dim
    p = pad
    h_out = h - weights_m.shape[2] + 1 + 2 * p
    w_out = w - weights_m.shape[3] + 1 + 2 * p
    output_dim = (weights_m.shape[0], h_out, w_out)

    toep_mat = np.zeros((np.prod(output_dim), np.prod(input_dim)))
    padd = padding_matrix(h, w, pad=p)

    for i, m in enumerate(weights_m):  # loop over output channel
        for j, n in enumerate(m):
            toep_k = one_toeplitz(n, (h + 2 * p, w + 2 * p))
            start_out = i * np.prod(output_dim[1:])
            end_out = (i + 1) * np.prod(output_dim[1:])
            start_in = j * np.prod(input_dim[1:])
            end_in = (j + 1) * np.prod(input_dim[1:])
            toep_mat[start_out:end_out, start_in:end_in] = toep_k @ padd

    TOEPLITZ = toep_mat

    # dimensions of wights matrix
    h_w = weights_m.shape[-2]
    w_w = weights_m.shape[-1]

    # dialiation and stride =1
    h_out = h - h_w + 1 + 2 * p
    w_out = w - w_w + 1 + 2 * p
    out_dim = (h_out, w_out)

    base_pattern = np.zeros(w_out, dtype="float32")
    indices = np.arange(0, len(base_pattern), stride)
    base_pattern[indices] = 1

    full_pattern = np.zeros(out_dim, dtype="float32")
    indices2 = np.arange(0, len(full_pattern), stride)
    full_pattern[indices2] = base_pattern

    full_pattern = np.tile(full_pattern.ravel(), output_dim[0])
    TOEPLITZ = TOEPLITZ[full_pattern > 0].astype("float32")

    return torch.from_numpy(TOEPLITZ)
