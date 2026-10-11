import jax
import jax.numpy as jnp
from jax import jit
import torch.nn.functional as F
import numpy as np
import torch
from myconv import ConvModel
import jax.profiler

# Create a log directory
logdir = "./jax_trace"

def im2col_manual_jax(x, KH, KW, S, P, out_h, out_w):
    ''' 
        Reimplement the same function (im2col_manual) in myconv.py "for JAX". 
        Hint: Instead of torch tensors, use of jnp arrays is required to leverage JIT compilation and GPU execution in JAX
    '''
    # x: (N, C, H, W)
    N, C, H, W = x.shape

    # Pad input
    x_pad = jnp.pad(x, ((0,0),(0,0),(P,P),(P,P)))

    # TO DO: Convert input (x) into shape (N, out_h*out_w, C*KH*KW). 
    # Refer to Lecture 3 for implementing this operation.
    patches = jnp.stack([
        jnp.stack([
                x_pad[:, :, kh: kh + out_h * S: S, kw: kw + out_w * S: S]
            for kw in range(KW)
        ], axis=0)
        for kh in range(KH)
    ], axis=0) # (KH, KW, N, C, out_h, out_w)
    
    patches = jnp.permute_dims(patches, [2, 4, 5, 3, 0, 1])
    patches = jnp.reshape(patches, (N, out_h * out_w, C * KH * KW))

    return patches

def conv2d_manual_jax(x, weight, bias, stride=1, padding=1):
    '''
        Reimplement the same function (conv2d_manual) in myconv.py "for JAX". 
        Hint: Instead of torch tensors, use of jnp arrays is required to leverage JIT compilation and GPU execution in JAX
        Hint: Unlike PyTorch, JAX arrays are immutable, so you cannot do indexing like out[i:j, :] = ... inside a JIT. You may use .at[].set() instead.
    '''
    N, C, H, W = x.shape
    C_out, _, KH, KW = weight.shape

    # define your helper variables here
    out_h = 1 + (H + 2 * padding - KH) // stride
    out_w = 1 + (W + 2 * padding - KW) // stride
    
    # TO DO: 1) convert input (x) into shape (N, out_h*out_w, C*KH*KW).
    cols = im2col_manual_jax(x, KH, KW, stride, padding, out_h, out_w)

    # TO DO: 2) flatten self.weight into shape (C_out, C*KH*KW).
    w_flat = jnp.transpose(jnp.reshape(weight, (C_out, C * KH * KW)))

    # TO DO: 3) perform tiled matmul after required reshaping is done.
    P = cols.shape[1]
    out = jnp.zeros((N, P, C_out))
    TILE_P = 512
    for t in range(0, P, TILE_P):
        cols_tile = cols[:, t: t + TILE_P, :] # (N, TILE_P, C*KH*KW)
        out = out.at[:, t: t + TILE_P, :].set(jnp.matmul(cols_tile, w_flat))

    # TO DO: 4) Add bias.
    out = out + bias # (N, P, C_out)

    # TO DO: 5) reshape output into shape (N, C_out, out_h, out_w).
    out = jnp.permute_dims(out, [0, 2, 1])
    out = jnp.reshape(out, (N, C_out, out_h, out_w))
    return out

if __name__ == "__main__":
    # Instantiate PyTorch model
    H, W = 33, 33
    model = ConvModel(H, W, in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=1)
    model.eval()

    # Example input
    x_torch = torch.randn(1, 3, H, W)

    # Export weights and biases
    params = {
        "weight": model.weight.detach().cpu().numpy(),  # shape (out_channels, in_channels, KH, KW)
        "bias": model.bias.detach().cpu().numpy()       # shape (out_channels,)
    }

    # Convert model input, weights and bias into jax arrays
    x_jax = jnp.array(x_torch.numpy())
    weight_jax = jnp.array(params["weight"])
    bias_jax = jnp.array(params["bias"])

    # enable JIT compilation
    conv2d_manual_jax_jit = jit(conv2d_manual_jax)

    # call your JAX function
    out_jax = conv2d_manual_jax_jit(x_jax, weight_jax, bias_jax)

    # Test your solution
    conv_ref = F.conv2d(x_torch, model.weight, model.bias, stride=1, padding=1)
    print("JAX --- shape check:", out_jax.shape == conv_ref.shape)
    print("JAX --- correctness check:", torch.allclose(torch.from_numpy(np.array(out_jax)), conv_ref, atol=1e-1))
