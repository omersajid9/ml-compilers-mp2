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

Implementation notes (optimizations):
  1. Weights are staged in SBUF once, transposed into (c_in, c_out) stationary
     layout per tap: no per-row weight DMA, no transpose on the compute path.
  2. The input is viewed flattened (C, H*W) and each band of R output rows is
     fetched as ONE contiguous (128, (R+fh-1)*in_width) tile per ic-tile:
     a single large DMA per band, and every tap window is then a contiguous
     flat slice band[:, (r+i)*in_width + j : ... + out_width] of it, i.e. a
     single-free-dim moving operand with no strided or multi-dim views.
  3. Every loop (batch, band, row, channel tiles, taps) is an affine hardware
     loop; the (128, out_width) PSUM accumulator lives inside the row/output
     -channel loops and accumulates the 9 taps x ic-tiles on-chip, so partial
     sums never leave PSUM until the row is complete.
  4. The drain (PSUM -> SBUF copy, broadcast bias add, HBM store) sits inside
     the affine output-channel loop, one row at a time.
  5. Leftover rows below a full band go through a sequential single-row path.
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

    c_in_pmax = nl.tile_size.pmax
    c_out_pmax = nl.tile_size.pmax
    n_ic = int(in_channels // c_in_pmax)
    n_oc = int(out_channels // c_out_pmax)
    fh = int(filter_height)
    fw = int(filter_width)
    ow = out_width

    X_out = nl.ndarray((batch_size, out_channels, out_height, out_width), dtype=X.dtype, buffer=nl.hbm)

    # flattened input: rows of the image are one contiguous pixel axis
    X_re = X.reshape((batch_size, in_channels, input_height * input_width))

    # weights: one contiguous load per oc-block, transposed per tap
    w_all = nl.ndarray((n_oc, n_ic, filter_height, filter_width, nl.par_dim(c_in_pmax), c_out_pmax), dtype=W.dtype, buffer=nl.sbuf)
    for kt in nl.affine_range(n_oc):
        w_blk = nl.load(W[kt * c_out_pmax:(kt + 1) * c_out_pmax, :, :, :])
        for tc in nl.affine_range(n_ic):
            for ti in nl.affine_range(filter_height):
                for tj in nl.affine_range(filter_width):
                    w_all[kt, tc, ti, tj] = nisa.nc_transpose(w_blk[:, tc * c_in_pmax:(tc + 1) * c_in_pmax, ti, tj])

    # bias as (out_channels, 1) for per-row broadcast load
    bias_re = bias.reshape((out_channels, 1))

    R = 6 # output rows per band
    band_rows = R + fh - 1 # input rows per band
    band_w = band_rows * input_width
    n_bands = out_height // R
    tail0 = n_bands * R

    for b in nl.affine_range(batch_size):
        for bidx in nl.affine_range(n_bands):
            r0 = bidx * R
            # one contiguous DMA per ic-tile for the whole band
            band = nl.ndarray((n_ic, nl.par_dim(c_in_pmax), band_w), dtype=X.dtype, buffer=nl.sbuf)
            for ct in nl.affine_range(n_ic):
                band[ct] = nl.load(
                    X_re[b, ct * c_in_pmax:(ct + 1) * c_in_pmax, r0 * input_width:(r0 + band_rows) * input_width])
            for r in nl.affine_range(R):
                p = r0 + r
                for kt in nl.affine_range(n_oc):
                    res_sum = nl.zeros((c_out_pmax, ow), nl.float32, buffer=nl.psum)
                    for ct in nl.affine_range(n_ic):
                        for i in nl.affine_range(fh):
                            for j in nl.affine_range(fw):
                                off = (r + i) * input_width + j
                                Xf = band[ct][:, off:off + ow]
                                res_sum += nl.matmul(w_all[kt, ct, i, j], Xf, transpose_x=True)
                    res_sb = nl.copy(res_sum, dtype=X_out.dtype)
                    bias_tile = nl.load(bias_re[kt * c_out_pmax:(kt + 1) * c_out_pmax, :])
                    res_sb = nl.add(res_sb, bias_tile)
                    nl.store(X_out[b, kt * c_out_pmax:(kt + 1) * c_out_pmax, p, :], value=res_sb)

        # tail: leftover rows below a full band
        for r in nl.sequential_range(tail0, out_height):
            xw = []
            for ic_bt in nl.static_range(n_ic):
                xw.append([])
                for rel in nl.static_range(fh):
                    xw[ic_bt].append(
                        nl.load(X[b, ic_bt * c_in_pmax:(ic_bt + 1) * c_in_pmax, r + rel, 0:input_width]))
            acc = [[nl.zeros((c_out_pmax, ow), nl.float32, buffer=nl.psum)
                    for _ in nl.static_range(n_ic)]
                   for _ in nl.static_range(n_oc)]
            for i in nl.static_range(fh):
                for j in nl.static_range(fw):
                    for ic_bt in nl.static_range(n_ic):
                        for oc_bt in nl.static_range(n_oc):
                            acc[oc_bt][ic_bt] += nl.matmul(w_all[oc_bt, ic_bt, i, j], xw[ic_bt][i][:, j:j + ow], transpose_x=True)
            for oc_bt in nl.static_range(n_oc):
                res_sb = nl.copy(acc[oc_bt][0], dtype=X_out.dtype)
                for ic_bt in nl.static_range(1, n_ic):
                    res_sb = nl.add(res_sb, nl.copy(acc[oc_bt][ic_bt], dtype=X_out.dtype))
                bias_tile = nl.load(bias_re[oc_bt * c_out_pmax:(oc_bt + 1) * c_out_pmax, :])
                res_sb = nl.add(res_sb, bias_tile)
                nl.store(X_out[b, oc_bt * c_out_pmax:(oc_bt + 1) * c_out_pmax, r, :], value=res_sb)
    return X_out
