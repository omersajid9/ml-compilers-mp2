# Machine Project 2 — Question 3 (GPU) Complete Guide

Everything needed to produce the Question 3 deliverables on the T4 (Google Colab):
the manual-im2col convolution, the four executable variants (eager, Inductor, JAX,
CUDA), correctness checks, profiler traces → Perfetto, the benchmark data, and the
report content.

Target runtime: **Colab, GPU (T4)**. All commands run in notebook cells or a
terminal after `cd gpu`.

---

## 0. Deliverables checklist (what Q3 must show)

| # | Requirement | Where it is produced |
|---|-------------|----------------------|
| 1 | Manual im2col conv2d in PyTorch (no `fold`/`unfold`) | `myconv.py` — `im2col_manual`, `conv2d_manual` |
| 2 | Run it on the GPU with PyTorch interpreter | `python3 myconv.py` |
| 3 | Same conv via PyTorch Inductor | `myconv_inductor.py` (`torch.compile(..., backend="inductor")`) |
| 4 | Same conv via JAX with JIT | `myconv_jax.py` (`jax.jit`) |
| 5 | Tiled CUDA kernel via C++ extension, verified vs `F.conv2d` | `myconv_interface.py` + `myconv_kernel.cu` |
| 6 | Execution traces from all four, loaded in Perfetto | `trace_*.json.gz` exported via `torch.profiler` |
| 7 | Report: kernel walltimes + host compile times | `benchmark.py` output (`.csv` + `.png`) |
| 8 | Report: walltime plot varying input & filter sizes | `benchmark_plot.png` (or your own from the CSV) |
| 9 | Report: justification via Inductor/JAX low-level ops found in Perfetto | analysis in report text |

---

## 1. Repository layout (all files live in `gpu/`)

| File | Role |
|------|------|
| `myconv.py` | **The PyTorch solution.** `ConvModel`, `im2col_manual`, `conv2d_manual` (tiled matmul, bias, reshape). Self-checks vs `F.conv2d`. Used by everything else. |
| `myconv_inductor.py` | Compiles `ConvModel` with `torch.compile(backend="inductor")`, checks vs `F.conv2d`. |
| `myconv_jax.py` | Ports the conv to JAX (`im2col_manual_jax`, `conv2d_manual_jax`), `jit`s it, checks vs `F.conv2d`. |
| `myconv_interface.py` | `torch.utils.cpp_extension.load` → runs `nvcc` on `myconv_kernel.cu`, calls `conv_cuda`, checks vs `F.conv2d`. |
| `myconv_kernel.cu` | The tiled CUDA kernel (shared-memory staging + per-thread PSUM). |
| `myconv_profiler.py` | Small helper wrapping `torch.profiler` with `ProfilerActivity.CUDA` (as the assignment requires). |
| `benchmark.py` | Walltime sweeps of all four variants + host compile times + plot. |
| `cuda_kernel_cpu_sim.py` | CPU mirror of the CUDA kernel for offline validation (no GPU needed). |
| `myconv_kernel_viz.html` | Interactive visual explainer of the `.cu` algorithm (open in any browser). |

---

## 2. Environment setup (Colab, once per session)

```python
!nvidia-smi                     # confirm T4
import torch, jax
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0))
print("jax backend:", jax.default_backend())   # MUST print "gpu"
```

Upload the `gpu/` files (file browser → drag in) or `git clone` the mp2 repo, then:

```python
%cd gpu
```

Rules from the assignment: do **not** `pip install` your own PyTorch/cuDNN/JAX —
Colab's bundled versions are what's graded.

---

## 3. Step 1 — the manual im2col (15 pts)

The 15-point im2col chunk is **done and verified** in `myconv.py`:
- Pad with `F.pad` (allowed).
- Gather patches with `stack` + strided slicing, `permute`, `reshape`
  (exactly the "combination of built-in PyTorch ops" the assignment requires —
  no `unfold`/`fold`).
- `conv2d_manual`: flatten weights → tiled matmul over patches → add bias →
  reshape to `(N, C_out, out_h, out_w)`.

Verify on the **GPU** runtime:

```bash
!python3 myconv.py
# expected:
#   PyTorch --- shape check: True
#   PyTorch --- correctness check: True
```

(Works on CPU too for local smoke tests; the graded run is on the T4.)

---

## 4. Step 2 — create & run the four executables (25 pts)

### 4.1 What "create the executable" means per variant

| Variant | What compiles | Where the artifact goes |
|---------|---------------|-------------------------|
| **Eager** | Nothing — PyTorch runs ops interpretively | — |
| **Inductor** | `torch.compile` — Python/TorchScript → **Triton** kernels at first call | `~/.cache/torch/inductor` (host compile time = first call) |
| **JAX** | `jax.jit` — Python → XLA HLO → LLVM/SASS at first call | `~/.cache/jax` (host compile time = first call) |
| **CUDA** | `nvcc` compiles `myconv_kernel.cu` into a `.so` | `~/.cache/torch_extensions/myconv` (host compile time = `load()`) |

### 4.2 Run each

```bash
!python3 myconv.py              # eager (also runs the conv on GPU)
!python3 myconv_inductor.py     # torch.compile(..., backend="inductor")
!python3 myconv_jax.py          # jax.jit(conv2d_manual_jax)
!python3 myconv_interface.py    # nvcc build (~1-2 min first time) + conv_cuda
```

Expected tails:
```
Inductor --- shape check: True
Inductor --- correctness check: True
JAX --- shape check: ...  JAX --- correctness check: True
CUDA --- shape check: True
CUDA --- correctness check: True
```

Notes:
- `myconv_interface.py` recompiles only if the `.cu` changed (torch caches the
  `.so`). Touching the file → rebuild (that rebuild time is itself a host
  compile-time datapoint for the report).
- First call of Inductor / JAX / CUDA is slow (seconds–minutes); subsequent
  calls are fast. That first-call gap is exactly the **host compilation time**
  the report asks for.
- JAX runner uses `atol=1e-1`; `benchmark.py` uses `atol=1e-4` for its own check.

### 4.3 (Optional) Inline notebook cells instead of scripts

```python
import time, torch
from torch.profiler import profile, ProfilerActivity
from myconv import ConvModel

torch.manual_seed(0)
N, C, H, W = 2, 3, 19, 19
x = torch.randn(N, C, H, W, device="cuda")
m = ConvModel(H, W, in_channels=3, out_channels=8,
              kernel_size=3, stride=1, padding=1).cuda().eval()

def snap(name, fn, iters=5):
    t0 = time.time(); fn(); torch.cuda.synchronize()
    print(f"{name}: first call (incl. compile) = {(time.time()-t0)*1e3:.1f} ms")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, with_stack=True, profile_memory=True) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(f"trace_{name}.json.gz")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    return prof

snap("01_eager", lambda: m(x))

mi = torch.compile(m, backend="inductor")
snap("02_inductor", lambda: mi(x))

import jax.numpy as jnp
from jax import jit
from myconv_jax import conv2d_manual_jax
xj = jnp.array(x.cpu().numpy())
wj = jnp.array(m.weight.detach().cpu().numpy())
bj = jnp.array(m.bias.detach().cpu().numpy())
fj = jit(conv2d_manual_jax)
snap("03_jax", lambda: fj(xj, wj, bj).block_until_ready())

from torch.utils.cpp_extension import load
conv = load(name="myconv", sources=["myconv_kernel.cu"], verbose=False)
wc = torch.randn(8, 3, 3, 3, device="cuda")
snap("04_cuda", lambda: conv.conv_cuda(x, wc, 1, 1))
```

This produces the four trace files **and** the four host compile times in one go.

---

## 5. Step 3 — traces → Perfetto

### 5.1 What the traces contain

`export_chrome_trace` writes a Chrome-format trace (`.json` or `.json.gz` — gzip
is fine and much smaller) that Perfetto renders with lanes:

- **CPU / host lane**: the Python + framework function calls —
  for **Inductor** you'll see `Triton codegen` / generated `triton_*` kernels;
  for **JAX** you'll see XLA / cuBLAS / generated kernels;
  for **eager** you'll see each op (`aten::permute`, `aten::matmul`, …) one by one.
- **CUDA lane**: GPU kernel durations — your `tiled_conv2d_kernel`, Inductor's
  Triton kernels, JAX's XLA kernels, cuDNN/cuBLAS calls.

This is where the "identify compiler optimizations as high-level function calls"
requirement is satisfied.

### 5.2 Export & view

1. The four `trace_*.json.gz` land in `gpu/`. **Download** them (file-browser
   right-click → Download) or push to your repo / Drive — the Colab filesystem is
   wiped on shutdown.
2. Open `https://ui.perfetto.dev/` → **Open trace file** → select one of the `.gz`
   files.
3. Read per trace:
   - **eager**: many small kernels + clear host ops per line.
   - **inductor**: fused Triton kernels; host shows the inductor compile +
     generated `triton_` code; look for fewer, longer GPU kernels (fusion).
   - **jax**: XLA-shaped kernels/cuBLAS; host shows XLA compilation.
   - **cuda**: your single `tiled_conv2d_kernel` per launch.

If the JAX CUDA lane is empty in the torch-profiler trace, additionally export a
JAX-native trace:

```python
import jax.profiler
jax.profiler.start_trace("./jax_trace")
fj(xj, wj, bj).block_until_ready()
jax.profiler.end_trace()
# load ./jax_trace into Perfetto (Open trace file -> folder)
```

---

## 6. Step 4 — benchmark data for the report (20 pts)

`benchmark.py` measures **steady-state warm walltime** (mean of `--reps`) for all
four variants at identical shapes, plus host compile times for Inductor and JAX,
and writes a CSV + a two-panel plot (vary kernel size / vary image size).

```bash
!python3 benchmark.py                            # defaults: sizes 32,64,128,256 · kernels 3,5,7 · channels 3,8 · reps 10
!python3 benchmark.py --sizes 16,32,64,128,256 --kernels 3,5,7 --reps 30
!python3 benchmark.py --out results.csv --plot results.png
```

- Flags: `--sizes`, `--kernels`, `--channels` (comma lists), `--reps`, `--out`,
  `--plot`, `--device auto|cpu|cuda`.
- CSV columns: `size, kernel, in_channels, out_channels, eager_ms, inductor_ms,
  jax_ms, cuda_ms, inductor_compile_ms, jax_compile_ms`.
- `kernel` column = filter size → this is your **"varying input and filter
  sizes"** plot data.
- A final line prints a correctness check for each backend at the last config
  (`shape True, values True`).
- Runtime on T4: a few minutes for defaults (nvcc build adds ~1–2 min on the
  first invocation).

Keep `benchmark_results.csv` + `benchmark_plot.png` — the CSV is the raw
evidence; the PNG is the required plot (re-style freely for the report).

---

## 7. Step 5 — writing the report (20 pts)

Required content, mapped to the rubric:

1. **Kernel execution times on GPU + host compilation times for the four
   variations.** Use the CSV: `eager_ms`, `inductor_ms`, `jax_ms`, `cuda_ms` for
   walltimes; `inductor_compile_ms`, `jax_compile_ms` for host compile; add the
   nvcc build time from the first `myconv_interface.py` / `benchmark.py` run for
   the CUDA entry.
2. **Plot of walltimes while varying input and filter sizes.** From the CSV /
   `benchmark_plot.png`; two axes (image size at fixed k, kernel size at fixed
   image) is the natural presentation.
3. **Justify observed results with Inductor/JAX low-level optimizations.**
   Pull the host-side calls from the Perfetto traces: Inductor's fused Triton
   kernels / codegen, JAX's XLA pass + cuBLAS calls; discuss fusion, tiling,
   avoiding the explicit im2col materialization, memory traffic.
4. **One-paragraph summary** of performance across compilers for different
   input shapes.

Suggested outline (Q3 part of the report only):

```
3. Results
  3.1 Correctness (all four pass vs F.conv2d)
  3.2 Walltimes + host compile times (table from CSV)
  3.3 Plot: walltime vs input size, vs filter size
  3.4 Perfetto analysis: eager vs Inductor vs JAX vs CUDA kernel traces
  3.5 Discussion: why the curves look the way they do (memory-bound im2col,
      fusion, cache, shared-memory tiling)
  3.6 One-paragraph summary
```

---

## 8. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `jax.default_backend()` prints `cpu` | You're not on a GPU runtime: Runtime → Change runtime type → T4 GPU, restart session. |
| CUDA extension rebuilds every run | Expected only when `myconv_kernel.cu` changed; cached `.so` is in `~/.cache/torch_extensions`. |
| `out of memory` at 256² with `profile_memory=True` | Reduce `--sizes` or profile one shape at a time; T4 has 16 GB. |
| Trace file huge | Use `.json.gz` (default suggestion); Perfetto accepts it. |
| Kernel launch looks silent/wrong on a weird shape | Add `cudaGetLastError()` after the launch in `conv_cuda` to surface errors. |
| Inductor/JAX compile very slow at 256² | Expected: host compilation cost is part of the report story — record it as such. |
| Want to validate the `.cu` offline (no GPU) | `python3 cuda_kernel_cpu_sim.py` mirrors the algorithm on CPU. |

---

## 9. Academic-integrity notes

- **Question 3 (GPU)**: full AI/tool help is fine — this guide, the code, the
  traces, and the report are all legitimate for Q3.
- **Questions 1–2 (einsum, TCG)**: the assignment marks these **“[No AI help
  allowed]”** — those answers must remain the student's own work; nothing in this
  guide covers them.
