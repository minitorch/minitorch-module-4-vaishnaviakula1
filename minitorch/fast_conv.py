from typing import Tuple
import numpy as np
from numba import njit, prange
from numba.core.types import float64

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Index,
    Shape,
    Strides,
    broadcast_index,
    index_to_position,
    to_index,
)

@njit(inline="always")
def to_index(shape: Shape, position: int) -> Index:
    # Function to convert a flat position in a tensor into a multidimensional index
    # Example implementation; actual function depends on tensor's shape and layout
    pass

@njit(inline="always")
def index_to_position(index: Index, strides: Strides) -> int:
    # Function to convert a multidimensional index in a tensor to a flat position
    # Example implementation; actual function depends on tensor's shape and layout
    pass

@njit(inline="always")
def broadcast_index(index: Index, from_shape: Shape, to_shape: Shape) -> Index:
    # Function to apply NumPy broadcasting rules to an index
    # Example implementation; actual function depends on shapes involved
    pass

@njit(parallel=True)
def _tensor_conv1d(out, out_shape, out_strides, out_size, input, input_shape, input_strides, weight, weight_shape, weight_strides, reverse):
    batch, out_channels, out_width = out_shape
    _, in_channels, width = input_shape
    _, _, kw = weight_shape

    for i in prange(batch):
        for j in prange(out_channels):
            for k in prange(out_width):
                sum = 0.0
                for l in range(in_channels):
                    for m in range(kw):
                        input_index = k + m if not reverse else k - m
                        if 0 <= input_index < width:
                            input_pos = index_to_position((i, l, input_index), input_strides)
                            weight_pos = index_to_position((j, l, m), weight_strides)
                            sum += input[input_pos] * weight[weight_pos]
                out_pos = index_to_position((i, j, k), out_strides)
                out[out_pos] = sum

tensor_conv1d = njit(parallel=True)(_tensor_conv1d)

class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(*output.tuple(), output.size, *input.tuple(), *weight.tuple(), False)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        grad_input, grad_weight = None, None
        # Implement the backward pass
        return grad_input, grad_weight

conv1d = Conv1dFun.apply

@njit(parallel=True, fastmath=True)
def _tensor_conv2d(out, out_shape, out_strides, out_size, input, input_shape, input_strides, weight, weight_shape, weight_strides, reverse):
    batch, out_channels, out_height, out_width = out_shape
    _, in_channels, height, width = input_shape
    _, _, kh, kw = weight_shape

    for i in prange(batch):
        for j in prange(out_channels):
            for k in prange(out_height):
                for l in prange(out_width):
                    sum = 0.0
                    for m in range(in_channels):
                        for n in range(kh):
                            for o in range(kw):
                                input_row = k + n if not reverse else k - n
                                input_col = l + o if not reverse else l - o
                                if 0 <= input_row < height and 0 <= input_col < width:
                                    input_pos = index_to_position((i, m, input_row, input_col), input_strides)
                                    weight_pos = index_to_position((j, m, n, o), weight_strides)
                                    sum += input[input_pos] * weight[weight_pos]
                    out_pos = index_to_position((i, j, k, l), out_strides)
                    out[out_pos] = sum

tensor_conv2d = njit(parallel=True, fastmath=True)(_tensor_conv2d)

class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(*output.tuple(), output.size, *input.tuple(), *weight.tuple(), False)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        grad_input, grad_weight = None, None
        # Implement the backward pass
        return grad_input, grad_weight

conv2d = Conv2dFun.apply
