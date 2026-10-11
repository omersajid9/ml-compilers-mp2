"""Shared profiling helper used by the myconv runner scripts.

Wraps torch.profiler so that:
  * CUDA activity is captured only when a CUDA device is actually used
    (safe on CPU-only machines, e.g. local development without a GPU);
  * the trace is exported to a chrome trace file on context exit, ready to
    load into Perfetto.
"""
from contextlib import contextmanager

from torch.profiler import profile, ProfilerActivity


@contextmanager
def profiler_for(device, trace_path=None):
    activities = [ProfilerActivity.CUDA] if device == "cuda" \
        else [ProfilerActivity.CPU]
    prof = profile(activities=activities, record_shapes=True,
                   with_stack=True, profile_memory=True)
    with prof:
        yield prof
    if trace_path:
        prof.export_chrome_trace(trace_path)
