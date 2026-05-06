"""Optional CUDA/CPU timing for inference and training without editing model code.

Uses forward hooks on existing ``nn.Module`` children (encoder / decoder) and
context-managed sections for greedy decode and AR autoregressive loops.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class CudaTimerStats:
    """Accumulates millisecond timings and call counts per string key."""

    def __init__(self) -> None:
        self._sum_ms: dict[str, float] = defaultdict(float)
        self._count: dict[str, int] = defaultdict(int)

    def add(self, key: str, ms: float) -> None:
        self._sum_ms[key] += float(ms)
        self._count[key] += 1

    def sum_ms(self, key: str) -> float:
        return float(self._sum_ms[key])

    def count(self, key: str) -> int:
        return int(self._count[key])

    def keys(self) -> Iterator[str]:
        return iter(self._sum_ms.keys())


def _tensor_device(args: tuple[object, ...]) -> torch.device | None:
    for a in args:
        if isinstance(a, torch.Tensor):
            return a.device
    return None


def attach_forward_timing_hooks(module: nn.Module, key: str, stats: CudaTimerStats) -> list:
    """Register pre/post forward hooks that record submodule forward duration.

    Returns handles whose ``remove()`` should be called when profiling ends.
    """

    def pre_hook(m: nn.Module, inp: tuple[object, ...]) -> None:
        dev = _tensor_device(inp if isinstance(inp, tuple) else (inp,))
        use_cuda = torch.cuda.is_available() and dev is not None and dev.type == "cuda"
        if use_cuda:
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            ev_start.record()
            m._rtprof_cuda_events = (ev_start, ev_end)
        else:
            m._rtprof_t0 = time.perf_counter()

    def post_hook(m: nn.Module, inp: tuple[object, ...], out: object) -> None:  # noqa: ARG001
        if getattr(m, "_rtprof_cuda_events", None) is not None:
            ev_start, ev_end = m._rtprof_cuda_events
            ev_end.record()
            torch.cuda.synchronize()
            ms = float(ev_start.elapsed_time(ev_end))
            stats.add(key, ms)
            del m._rtprof_cuda_events
        elif getattr(m, "_rtprof_t0", None) is not None:
            ms = (time.perf_counter() - m._rtprof_t0) * 1000.0
            stats.add(key, ms)
            del m._rtprof_t0

    h_pre = module.register_forward_pre_hook(pre_hook)
    h_post = module.register_forward_hook(post_hook)
    return [h_pre, h_post]


def detach_forward_hooks(handles: list) -> None:
    for h in handles:
        h.remove()


def attach_inference_hooks(model: nn.Module, method: str, stats: CudaTimerStats) -> list:
    """Attach encoder (and AR decoder) hooks for validation snapshot collection."""
    handles: list = []
    if method in ("cpl", "bce", "hungarian"):
        handles.extend(attach_forward_timing_hooks(model.transformer, "encoder_ms", stats))
    elif method == "ar":
        handles.extend(attach_forward_timing_hooks(model.encoder, "encoder_ms", stats))
        handles.extend(attach_forward_timing_hooks(model.decoder, "decoder_ms", stats))
    else:
        raise ValueError(f"unknown method {method!r}")
    return handles


@contextmanager
def cuda_timing_section(stats: CudaTimerStats | None, key: str) -> Iterator[None]:
    """Time a region on CUDA (events) or CPU (perf_counter)."""
    if stats is None:
        yield
        return

    if torch.cuda.is_available():
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        try:
            yield
        finally:
            ev_end.record()
            torch.cuda.synchronize()
            stats.add(key, float(ev_start.elapsed_time(ev_end)))
    else:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            stats.add(key, (time.perf_counter() - t0) * 1000.0)


def aggregate_inference_timing(
    stats: CudaTimerStats,
    method: str,
    num_batches: int,
    num_examples: int,
) -> dict[str, float]:
    """Turn accumulated stats into averages suitable for logging / tables."""

    def avg(key: str) -> float:
        c = stats.count(key)
        return stats.sum_ms(key) / float(max(c, 1))

    out: dict[str, float] = {}
    ne = float(max(num_examples, 1))
    nb = float(max(num_batches, 1))

    if stats.count("encoder_ms"):
        out["avg_batch_encoder_ms"] = avg("encoder_ms")
        out["per_example_encoder_ms"] = stats.sum_ms("encoder_ms") / ne

    if method == "ar" and stats.count("decoder_ms"):
        dec_sum = stats.sum_ms("decoder_ms")
        out["avg_batch_decoder_ms"] = dec_sum / nb
        out["per_example_decoder_ms"] = dec_sum / ne

    if stats.count("forward_total_ms"):
        out["avg_batch_forward_total_ms"] = avg("forward_total_ms")
        out["per_example_forward_total_ms"] = stats.sum_ms("forward_total_ms") / ne
        if stats.count("encoder_ms"):
            enc_sum = stats.sum_ms("encoder_ms")
            fwd_sum = stats.sum_ms("forward_total_ms")
            nf = float(max(stats.count("forward_total_ms"), 1))
            out["avg_batch_forward_heads_ms"] = (fwd_sum - enc_sum) / nf
            out["per_example_forward_heads_ms"] = (fwd_sum - enc_sum) / ne

    if stats.count("cpl_greedy_ms"):
        out["avg_batch_cpl_greedy_ms"] = avg("cpl_greedy_ms")
        out["per_example_cpl_greedy_ms"] = stats.sum_ms("cpl_greedy_ms") / ne

    if stats.count("ar_autoreg_ms"):
        out["avg_batch_ar_autoreg_ms"] = avg("ar_autoreg_ms")
        out["per_example_ar_autoreg_ms"] = stats.sum_ms("ar_autoreg_ms") / ne
        if stats.count("decoder_ms"):
            dec_sum = stats.sum_ms("decoder_ms")
            ar_sum = stats.sum_ms("ar_autoreg_ms")
            n_autoreg = float(max(stats.count("ar_autoreg_ms"), 1))
            out["avg_batch_ar_non_decoder_ms"] = (ar_sum - dec_sum) / n_autoreg
            out["per_example_ar_non_decoder_ms"] = (ar_sum - dec_sum) / ne

    if stats.count("ar_inference_total_ms"):
        out["avg_batch_ar_inference_total_ms"] = avg("ar_inference_total_ms")
        out["per_example_ar_inference_total_ms"] = stats.sum_ms("ar_inference_total_ms") / ne

    out["num_batches_timed"] = float(num_batches)
    out["num_examples_timed"] = float(num_examples)
    return out


@torch.no_grad()
def run_inference_warmup_loop(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    method: str,
    *,
    max_batches: int | None = None,
    desc: str = "  warmup",
    cpl_max_selection_steps: int = 100,
    ar_max_selection_steps: int | None = None,
) -> int:
    """Run forward (+ CPL greedy or AR greedy) without timing hooks.

    Mirrors the validation snapshot forward path so CUDA kernels, cuDNN
    autotune, and allocator caches are warm before :func:`attach_inference_hooks`.

    Returns the number of batches processed.
    """
    # Local import avoids import cycle (``inference`` imports this module).
    from inference import ar_greedy, cpl_greedy

    model.eval()
    limit = max_batches if max_batches is not None else len(loader)
    seen = 0
    for X, _cluster_labels, _k, _ds, _gt, mask in tqdm(loader, total=limit, desc=desc, leave=False):
        if max_batches is not None and seen >= max_batches:
            break
        X_dev = X.to(device)
        mask_dev = mask.to(device)
        pad = ~mask_dev
        if method == "cpl":
            theta_full, W = model(X_dev, pad)
            cpl_greedy(theta_full, W, cpl_max_selection_steps, mask_dev)
        elif method == "ar":
            ar_steps = int(ar_max_selection_steps) if ar_max_selection_steps is not None else int(getattr(model, "max_selection_steps", 20))
            ar_greedy(model, X_dev, mask_dev, ar_steps, timing_stats=None)
        elif method in ("bce", "hungarian"):
            model(X_dev, pad)
        else:
            raise ValueError(f"unknown method {method!r}")
        seen += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return seen


def warmup_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_batches: int,
    method: str,
    *,
    cpl_max_selection_steps: int = 100,
    ar_max_selection_steps: int | None = None,
) -> None:
    """Run ``n_batches`` of inference without timing.

    Call *before* :func:`attach_inference_hooks` so warmup never accumulates
    into :class:`CudaTimerStats`.
    """
    if n_batches <= 0:
        return
    run_inference_warmup_loop(
        model,
        loader,
        device,
        method,
        max_batches=n_batches,
        desc="  warmup",
        cpl_max_selection_steps=cpl_max_selection_steps,
        ar_max_selection_steps=ar_max_selection_steps,
    )


def print_inference_timing_report(report: dict[str, float]) -> None:
    """Pretty-print timing aggregates (milliseconds)."""
    print("\n=== Inference timing (ms) ===")
    keys = sorted(k for k in report if not k.startswith("num_"))
    col_w = max(len(k) for k in keys) if keys else 20
    for k in keys:
        print(f"  {k:<{col_w}}  {report[k]:.4f}")
    if "num_batches_timed" in report:
        print(f"  {'num_batches_timed':<{col_w}}  {int(report['num_batches_timed'])}")
    if "num_examples_timed" in report:
        print(f"  {'num_examples_timed':<{col_w}}  {int(report['num_examples_timed'])}")
    print()


__all__ = [
    "CudaTimerStats",
    "aggregate_inference_timing",
    "attach_forward_timing_hooks",
    "attach_inference_hooks",
    "cuda_timing_section",
    "detach_forward_hooks",
    "print_inference_timing_report",
    "run_inference_warmup_loop",
    "warmup_inference",
]
