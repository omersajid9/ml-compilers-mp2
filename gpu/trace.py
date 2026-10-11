import time, torch
from torch.profiler import profile, ProfilerActivity
from myconv import ConvModel

torch.manual_seed(0)
N, C, H, W = 2, 4, 22, 22
C_out = 8
kernel_size = 7
x = torch.randn(N, C, H, W, device="cuda")
m = ConvModel(H, W, in_channels=C, out_channels=C_out, kernel_size=kernel_size, stride=1, padding=1).cuda().eval()

def snap(name, fn, iters=5):
    t0 = time.time()
    fn()
    torch.cuda.synchronize()
    print(f"{name}: first-call (incl. host compile) = {(time.time()-t0)*1e3:.1f} ms")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True, profile_memory=True) as p:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    p.export_chrome_trace(f"trace_{name}.json.gz")
    print("  wrote", f"trace_{name}.json.gz")
    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=8))

# a) eager
snap("01_eager", lambda: m(x))

# b) inductor
mi = torch.compile(m, backend="inductor")
snap("02_inductor", lambda: mi(x))

# c) jax
import numpy as np
import jax.numpy as jnp
from jax import jit
from myconv_jax import conv2d_manual_jax
xj = jnp.array(x.cpu().numpy())
wj = jnp.array(m.weight.detach().cpu().numpy())
bj = jnp.array(m.bias.detach().cpu().numpy())
fj = jit(conv2d_manual_jax)
snap("03_jax", lambda: fj(xj, wj, bj).block_until_ready())

# d) cuda
t0 = time.time()
from torch.utils.cpp_extension import load
conv = load(name="myconv", sources=["myconv_kernel.cu"], verbose=False)
print(f"cuda: nvcc build = {(time.time()-t0)*1e3:.1f} ms")
wc = torch.randn(m.weight.shape, device="cuda")
snap("04_cuda", lambda: conv.conv_cuda(x, wc, 1, 1))