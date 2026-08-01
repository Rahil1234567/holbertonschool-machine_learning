#!/usr/bin/env python3
"""Perform back propagation over a pooling layer."""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer.

    Args:
        dA: numpy.ndarray of shape (m, h_new, w_new, c)
            containing the partial derivatives with respect
            to the output of the pooling layer.
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c)
            containing the output of the previous layer.
        kernel_shape: tuple (kh, kw)
        stride: tuple (sh, sw)
        mode: 'max' or 'avg'

    Returns:
        dA_prev: partial derivatives with respect to the
                 previous layer.
    """
    m, h_prev, w_prev, c = A_prev.shape
    _, h_new, w_new, _ = dA.shape

    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(h_new):
        for j in range(w_new):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            for k in range(c):

                if mode == 'max':
                    A_slice = A_prev[
                        :,
                        vert_start:vert_end,
                        horiz_start:horiz_end,
                        k
                    ]

                    mask = (
                        A_slice ==
                        np.max(A_slice, axis=(1, 2), keepdims=True)
                    )

                    dA_prev[
                        :,
                        vert_start:vert_end,
                        horiz_start:horiz_end,
                        k
                    ] += (
                        mask *
                        dA[:, i, j, k][:, None, None]
                    )

                elif mode == 'avg':
                    average = (
                        dA[:, i, j, k] / (kh * kw)
                    )

                    dA_prev[
                        :,
                        vert_start:vert_end,
                        horiz_start:horiz_end,
                        k
                    ] += (
                        np.ones((m, kh, kw)) *
                        average[:, None, None]
                    )

    return dA_prev
