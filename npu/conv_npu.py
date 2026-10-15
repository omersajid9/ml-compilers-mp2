import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A convolution kernel that you need to implement.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height
out_pool_width = out_width

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""

@nki.jit
def conv2d(X, W, bias):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height
    out_pool_width = out_width
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    c_in_pmax = nl.tile_size.pmax # 128
    n_tiles_c_in = in_channels // c_in_pmax

    c_out_pmax = nl.tile_size.pmax # 128
    n_tiles_c_out = out_channels // c_out_pmax

    w_all = nl.ndarray(
        (filter_height, filter_width, n_tiles_c_out, n_tiles_c_in,
         nl.par_dim(c_in_pmax), c_out_pmax),
        dtype=W.dtype,
        buffer=nl.sbuf,
    )
    for i in nl.affine_range(filter_height):
        for j in nl.affine_range(filter_width):
            for oc_bt in nl.affine_range(n_tiles_c_out):
                for ic_bt in nl.affine_range(n_tiles_c_in):
                    w_io = nl.ndarray((c_out_pmax, c_in_pmax), dtype=W.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=w_io,
                        src=W[oc_bt * c_out_pmax:(oc_bt + 1) * c_out_pmax,
                              ic_bt * c_in_pmax:(ic_bt + 1) * c_in_pmax,
                              i, j]
                    )
                    w_all[i, j, oc_bt, ic_bt] = nl.transpose(w_io)

    bias_re = bias.reshape((n_tiles_c_out, c_out_pmax, 1))

    for b in nl.affine_range(batch_size):
        for row in nl.affine_range(out_height):
            x_all = nl.ndarray(
                (filter_height, filter_width, n_tiles_c_in,
                 nl.par_dim(c_in_pmax), out_width),
                dtype=X.dtype,
                buffer=nl.sbuf,
            )
            for i in nl.affine_range(filter_height):
                for j in nl.affine_range(filter_width):
                    for ic_bt in nl.affine_range(n_tiles_c_in):
                        x_all[i, j, ic_bt] = nl.load(
                            X[b,
                              ic_bt * c_in_pmax:(ic_bt + 1) * c_in_pmax,
                              row + i,
                              j:j + out_width])

            for oc_bt in nl.affine_range(n_tiles_c_out):
                res_psum = nl.zeros((c_out_pmax, out_width), nl.float32, buffer=nl.psum)
                for i in nl.affine_range(filter_height):
                    for j in nl.affine_range(filter_width):
                        for ic_bt in nl.affine_range(n_tiles_c_in):
                            res_psum += nl.matmul(
                                w_all[i, j, oc_bt, ic_bt],
                                x_all[i, j, ic_bt],
                                transpose_x=True)
                res_sb = nl.copy(res_psum, dtype=X_out.dtype)
                bias_tile = nl.load(bias_re[oc_bt]) # (c_out_pmax, 1)
                res_sb = nl.add(res_sb, bias_tile)
                nl.store(X_out[b,
                               oc_bt * c_out_pmax:(oc_bt + 1) * c_out_pmax,
                               row, :],
                         value=res_sb)

    return X_out