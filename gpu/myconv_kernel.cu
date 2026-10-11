#include <iostream>
#include <cstdlib>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// example
#define TILE_H 8   
#define TILE_W 8   
#define TILE_C 16  

// Kernel declaration
__global__ void gemm_gpu_o4_kernel(
    const float* __restrict__ x,       // input: N x C x H x W
    const float* __restrict__ w,       // weights: C_out x C_in x KH x KW
    float* __restrict__ out,           // output: N x C x H x W
    int N, int C_in, int H, int W,
    int C_out, int KH, int KW,
    int stride, int pad,
    int out_h, int out_w,
    int tile_h, int tile_w, int tile_elems
) {
    extern __shared__ float shmem[];

    // block indexes
    int n = blockIdx.z;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int tid = ty * blockDim.x + tx;
    int nthreads = blockDim.x * blockDim.y;

    // output indexes
    int oh = blockIdx.y * blockDim.y + ty;
    int ow = blockIdx.x * blockDim.x + tx;

    // top left corner of input tile
    int tile_top = blockIdx.y * TILE_H * stride - pad;
    int tile_left = blockIdx.x * TILE_W * stride - pad;

    // tile C_in dimension
    for (int c0 = 0; c0 < C_in; c0 += TILE_C) {

        // load data into shared memory
        for (int idx = tid; idx < TILE_C * tile_elems; idx += nthreads) {
            int c = idx / tile_elems;
            int iy = (idx % tile_elems) / tile_w;
            int ix = idx % tile_w;

            float val = 0.0f;
            if (c0 + c < C_in) {
                int ih = tile_top + iy;
                int iw = tile_left + ix;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                    val = x[((n * C_in + c0 + c) * H + ih) * W + iw];
            }
            shmem[(c * tile_h + iy) * tile_w + ix] = val;
        }
        __syncthreads();

        // calculate convolution into partial sum
        if (oh < out_h && ow < out_w) {
            int nc = (c0 + TILE_C <= C_in) ? TILE_C : C_in - c0;
            for (int co = 0; co < C_out; co++) {
                float sum = 0.0f;
                for (int c = 0; c < nc; c++) {
                    const float* tp = shmem + c * tile_elems + (ty * stride) * tile_w + (tx * stride);
                    const float* wp = w + (co * C_in * KH * KW) + (c0 + c) * KH * KW;
                    for (int kh = 0; kh < KH; kh++) {
                        for (int kw = 0; kw < KW; kw++) {
                            sum += tp[kh * tile_w + kw] * wp[kh * KW + kw];
                        }
                    }
                }
                long o = (((long) n * C_out + co) * out_h + oh) * out_w + ow;
                out[o] += sum;
            }
        }
        __syncthreads();
    }
}

// Function for Python binding
torch::Tensor conv_cuda(torch::Tensor x, torch::Tensor w,
                          int stride, int pad) {
    int N = x.size(0);
    int C_in = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    int C_out = w.size(0);
    int KH = w.size(2);
    int KW = w.size(3);

    int out_h = (H + 2 * pad - KH) / stride + 1;
    int out_w = (W + 2 * pad - KW) / stride + 1;

    auto out = torch::zeros({N, C_out, out_h, out_w}, x.options());

    dim3 block(TILE_W, TILE_H);
    dim3 grid((out_w + block.x - 1)/block.x,
              (out_h + block.y - 1)/block.y,
              N);

    int tile_h = (TILE_H - 1) * stride + KH;
    int tile_w = (TILE_W - 1) * stride + KW;
    size_t shmem_bytes = (size_t)TILE_C * tile_h * tile_w * sizeof(float);

    gemm_gpu_o4_kernel<<<grid, block, shmem_bytes>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C_in, H, W,
        C_out, KH, KW,
        stride, pad,
        out_h, out_w,
        tile_h, tile_w, tile_h * tile_w);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_cuda", &conv_cuda, "Custom Conv2D (CUDA)");
}
