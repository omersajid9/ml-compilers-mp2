#!/usr/bin/env python3
"""CPU reference simulation of gpu/myconv_kernel.cu's tiled algorithm.

This is NOT a production convolution -- it mirrors the CUDA kernel step for
step (same TILE_H/TILE_W grid, same shared-memory halo staging with implicit
zero padding, same per-thread PSUM accumulation over the channel loop, same
final single write per output element) so the kernel's correctness can be
validated on a machine without an NVIDIA GPU.  On Google Colab, the real
check is `python3 myconv_interface.py`, which compiles the .cu with nvcc.
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

TILE_H = 8
TILE_W = 8


def cuda_tiled_kernel_sim(x, w, stride, pad):
    """Mirror of tiled_conv2d_kernel + conv_cuda (stride must be 1)."""
    assert stride == 1, "the CUDA kernel implements stride == 1 only"
    N, C_in, H, W = x.shape
    C_out = w.shape[0]
    KH, KW = w.shape[2], w.shape[3]

    out_h = H + 2 * pad - KH + 1
    out_w = W + 2 * pad - KW + 1
    out = np.zeros((N, C_out, out_h, out_w), dtype=np.float32)

    tile_h = TILE_H + KH - 1          # patch size needed by one block
    tile_w = TILE_W + KW - 1

    for n in range(N):
        for out_oh0 in range(0, out_h, TILE_H):
            for out_ow0 in range(0, out_w, TILE_W):
                # input patch origin (back off the padding, stride == 1)
                in_oh0 = out_oh0 - pad
                in_ow0 = out_ow0 - pad

                for c in range(C_in):
                    # ---- s_in: staged input patch (zero == padding) ----
                    s_in = np.zeros((tile_h, tile_w), dtype=np.float32)
                    for sy in range(tile_h):
                        for sx in range(tile_w):
                            ih = in_oh0 + sy
                            iw = in_ow0 + sx
                            if 0 <= ih < H and 0 <= iw < W:
                                s_in[sy, sx] = x[n, c, ih, iw]

                    # ---- s_w: staged per-channel weights (C_out, KH, KW) ----
                    s_w = w[:, c, :, :]

                    # ---- PSUM accumulation over (taps) for this channel ----
                    for ty in range(TILE_H):
                        for tx in range(TILE_W):
                            oh = out_oh0 + ty
                            ow = out_ow0 + tx
                            if oh < out_h and ow < out_w:
                                win = s_in[ty : ty + KH, tx : tx + KW].ravel()
                                for co in range(C_out):
                                    out[n, co, oh, ow] += (
                                        win @ s_w[co].ravel())

    return out


def main():
    torch.manual_seed(0)
    cases = [
        (4, 3, 25, 25, 4, 6, 6, 1),   # launcher default
        (1, 3, 32, 32, 8, 3, 3, 1),
        (2, 4, 22, 22, 8, 7, 7, 1),   # myconv.py default test
        (3, 2, 15, 17, 6, 5, 5, 0),   # no padding; non-square image
        (1, 1, 9, 9,  2, 2, 4, 2),    # non-square kernel
        (2, 5, 20, 20, 3, 1, 1, 0),   # 1x1 kernel
    ]
    all_ok = True
    for N, C_in, H, W, C_out, KH, KW, pad in cases:
        x = torch.randn(N, C_in, H, W)
        w = torch.randn(C_out, C_in, KH, KW)
        ref = F.conv2d(x, w, stride=1, padding=pad)
        sim = torch.from_numpy(cuda_tiled_kernel_sim(x.numpy(), w.numpy(),
                                                     1, pad))
        ok_s = sim.shape == ref.shape
        ok_v = torch.allclose(sim, ref, atol=1e-5, rtol=1e-5)
        all_ok &= ok_s and ok_v
        print(f"N={N} C={C_in} {H}x{W} Co={C_out} k={KH}x{KW} pad={pad} "
              f"-> shape {ok_s}, values {ok_v}")
    print("RESULT:", "all checks PASSED" if all_ok else "FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
