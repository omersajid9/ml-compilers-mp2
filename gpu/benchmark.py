#!/usr/bin/env python3
"""Wall-time sweeps for the four MP2 Part-3 convolution implementations.

Measures steady-state kernel wall time (warm, mean of --reps runs) for:
    eager   : manual im2col conv2d    (myconv.ConvModel, eager torch)
    inductor: same model torch.compile(backend="inductor")
    jax     : jit(conv2d_manual_jax)
    cuda    : tiled shared-memory kernel (myconv_kernel.cu, GPU only)

Also records host compile time (first call) for inductor and jax.

Usage (on the Colab GPU session, after `%cd mp2/gpu`):
    python3 benchmark.py                         # default sweep
    python3 benchmark.py --sizes 32,64,128,256 --kernels 3,5,7
    python3 benchmark.py --out results.csv --plot results.png

The CUDA backend is measured only when a GPU is present; everything else also
runs on CPU (numbers then reflect the CPU backend, for local smoke tests).
"""
import argparse
import csv
import time

import numpy as np


def timer(device):
    if device == "cuda":
        import torch
        def sync():
            torch.cuda.synchronize()
        return sync
    return lambda: None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--sizes", default="32,64,128,256",
                    help="comma-separated H=W image sizes")
    ap.add_argument("--kernels", default="3,5,7",
                    help="comma-separated kernel sizes")
    ap.add_argument("--channels", default="3,8", help="in,out channels")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default="benchmark_results.csv")
    ap.add_argument("--plot", default="benchmark_plot.png")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    sync = timer(device)

    import myconv
    from myconv import ConvModel
    import myconv_jax
    import jax
    import jax.numpy as jnp
    from jax import jit

    # CUDA backend: lazy-load the extension (skipped when no GPU).
    cuda_mod = None
    if device == "cuda":
        from torch.utils.cpp_extension import load
        cuda_mod = load(name="myconv", sources=["myconv_kernel.cu"],
                        verbose=False)

    C_in, C_out = map(int, args.channels.split(","))
    rows = []

    for size in map(int, args.sizes.split(",")):
        H = W = size
        # Keep batch = 1 so the plot is a clean function of input size.
        N = 1
        x = torch.randn(N, C_in, H, W, device=device)
        x_jax = jnp.array(x.cpu().numpy())
        ref_w = torch.randn(C_out, C_in, 3, 3, device=device)
        for k in map(int, args.kernels.split(",")):
            pad = 1
            model = ConvModel(H, W, C_in, C_out, k, stride=1,
                              padding=pad).to(device).eval()
            weight_jax = jnp.array(model.weight.detach().cpu().numpy())
            bias_jax = jnp.array(model.bias.detach().cpu().numpy())

            entries = {"size": size, "kernel": k,
                       "in_channels": C_in, "out_channels": C_out}

            # ---------------- eager ----------------
            for _ in range(2):          # warmup
                model(x)
            sync()
            t0 = time.perf_counter()
            for _ in range(args.reps):
                model(x)
            sync()
            entries["eager_ms"] = (time.perf_counter() - t0) / args.reps * 1e3

            # ---------------- inductor ----------------
            compiled = torch.compile(model, backend="inductor")
            t0 = time.perf_counter()
            compiled(x)                 # first call: host compilation
            sync()
            t_compile = time.perf_counter() - t0
            for _ in range(2):
                compiled(x)
            sync()
            t0 = time.perf_counter()
            for _ in range(args.reps):
                compiled(x)
            sync()
            entries["inductor_ms"] = (time.perf_counter() - t0) / args.reps * 1e3
            entries["inductor_compile_ms"] = t_compile * 1e3

            # ---------------- jax ----------------
            f = jit(myconv_jax.conv2d_manual_jax)
            t0 = time.perf_counter()
            f(x_jax, weight_jax, bias_jax).block_until_ready()
            t_compile = time.perf_counter() - t0
            for _ in range(2):
                f(x_jax, weight_jax, bias_jax).block_until_ready()
            t0 = time.perf_counter()
            for _ in range(args.reps):
                f(x_jax, weight_jax, bias_jax).block_until_ready()
            entries["jax_ms"] = (time.perf_counter() - t0) / args.reps * 1e3
            entries["jax_compile_ms"] = t_compile * 1e3

            # ---------------- cuda ----------------
            if cuda_mod is not None:
                with torch.no_grad():
                    wc = torch.randn(C_out, C_in, k, k, device=device)
                    for _ in range(2):
                        cuda_mod.conv_cuda(x, wc, 1, pad)
                    sync()
                    t0 = time.perf_counter()
                    for _ in range(args.reps):
                        cuda_mod.conv_cuda(x, wc, 1, pad)
                    sync()
                    entries["cuda_ms"] = \
                        (time.perf_counter() - t0) / args.reps * 1e3

            rows.append(entries)
            print(f"{entries}")

    # Sanity: correctness of each backend on the last config.
    ref = F.conv2d(x, model.weight, model.bias, stride=1, padding=1)
    checks = {
        "eager": model(x),
        "inductor": compiled(x),
        "jax": torch.from_numpy(
            np.array(f(x_jax, weight_jax, bias_jax).block_until_ready())),
    }
    if cuda_mod is not None:
        checks["cuda"] = cuda_mod.conv_cuda(x, wc, 1, pad)
    print("--- correctness on last config ---")
    for name, o in checks.items():
        if o.is_cuda:
            o = o.cpu()
        print(f"  {name}: "
              f"shape {o.shape == ref.shape}, "
              f"values {torch.allclose(o.float().cpu(), ref.cpu(), atol=1e-4)}")

    fields = ["size", "kernel", "in_channels", "out_channels",
              "eager_ms", "inductor_ms", "jax_ms", "cuda_ms",
              "inductor_compile_ms", "jax_compile_ms"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {args.out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, col, title in [
                (axes[0], "kernel", "vary kernel size (H=W=max size)"),
                (axes[1], "size", "vary image size (k=3)")]:
            for backend in ["eager_ms", "inductor_ms", "jax_ms", "cuda_ms"]:
                if not any(backend in r for r in rows):
                    continue
                data = [r for r in rows if r.get(backend) is not None]
                by = {}
                for r in data:
                    by.setdefault(r[col], []).append(r[backend])
                xs = sorted(by)
                ax.plot(xs, [np.mean(by[k]) for k in xs], marker="o",
                        label=backend.replace("_ms", ""))
            ax.set_xlabel(col); ax.set_ylabel("wall time (ms)")
            ax.set_title(title); ax.legend(); ax.grid(True, alpha=.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"wrote {args.plot}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
