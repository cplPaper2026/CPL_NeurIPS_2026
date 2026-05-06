"""Optional CUDA/CPU inference timing for road-cpl without editing model code.

The profiler is hook-driven: forward pre/post hooks on the top-level model and
on its sub-modules (``backbone``, ``transformer``, optional ``cpl_head`` /
``ar_head``) record per-region durations. The autoregressive greedy-decode
loop -- which is not part of any ``forward`` -- is timed by externally
monkey-patching ``head.greedy_decode`` with a wrapper that runs the original
inside a ``cuda_timing_section``. The wrapper is exposed via a
``_RevertHandle`` whose ``.remove()`` restores the original method, so cleanup
is uniform with PyTorch ``RemovableHandle`` instances.

This file is intentionally self-contained -- nothing inside ``train.py``'s
inference loop has to change for profiling to work.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Stats accumulator
# ---------------------------------------------------------------------------


class CudaTimerStats:
    """Accumulates per-key millisecond sums + call counts.

    ``num_examples`` is a separate scalar counter bumped by the top-level
    model pre-hook so that per-example averages can be reported without
    threading batch sizes through the call sites.
    """

    def __init__(self) -> None:
        self._sum_ms: dict[str, float] = defaultdict(float)
        self._count: dict[str, int] = defaultdict(int)
        self.num_examples: int = 0

    def add(self, key: str, ms: float) -> None:
        self._sum_ms[key] += float(ms)
        self._count[key] += 1

    def sum_ms(self, key: str) -> float:
        return float(self._sum_ms[key])

    def count(self, key: str) -> int:
        return int(self._count[key])

    def keys(self) -> Iterator[str]:
        return iter(self._sum_ms.keys())


# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------


def _tensor_device(args: tuple[object, ...]) -> torch.device | None:
    for a in args:
        if isinstance(a, torch.Tensor):
            return a.device
    return None


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


# ---------------------------------------------------------------------------
# Forward-hook timing
# ---------------------------------------------------------------------------


def attach_forward_timing_hooks(
    module: nn.Module,
    key: str,
    stats: CudaTimerStats,
    *,
    record_num_examples: bool = False,
) -> list:
    """Register forward pre/post hooks that record submodule duration.

    When ``record_num_examples`` is True the pre-hook also bumps
    ``stats.num_examples`` by the leading dim of the first tensor input. Use
    this on the *top-level* model so per-example averages are derivable.

    Returns handles whose ``.remove()`` should be called when profiling ends.
    """

    def pre_hook(m: nn.Module, inp: tuple[object, ...]) -> None:
        if record_num_examples:
            tup = inp if isinstance(inp, tuple) else (inp,)
            for a in tup:
                if isinstance(a, torch.Tensor):
                    stats.num_examples += int(a.shape[0])
                    break
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


# ---------------------------------------------------------------------------
# Monkey-patch revert handle (uniform .remove() interface)
# ---------------------------------------------------------------------------


class _RevertHandle:
    """Mimics ``torch.utils.hooks.RemovableHandle`` for monkey-patches.

    Storing ``original`` as ``None`` indicates the attribute was set on the
    instance for the first time and should be removed entirely on revert.
    """

    def __init__(self, obj: object, attr: str, original: Any) -> None:
        self._obj = obj
        self._attr = attr
        self._original = original
        self._reverted = False

    def remove(self) -> None:
        if self._reverted:
            return
        if self._original is _NOT_SET:
            try:
                delattr(self._obj, self._attr)
            except AttributeError:
                pass
        else:
            setattr(self._obj, self._attr, self._original)
        self._reverted = True


_NOT_SET = object()


def _wrap_method_with_timing(
    obj: object,
    attr: str,
    key: str,
    stats: CudaTimerStats,
) -> _RevertHandle:
    """Replace ``obj.attr`` with a callable that times the original.

    The original (bound) method is captured in the closure and restored on
    ``handle.remove()``. Works for ``nn.Module`` instances because plain
    callables bypass ``nn.Module.__setattr__``'s parameter / submodule fast
    paths.
    """
    original = obj.__dict__.get(attr, _NOT_SET)
    bound = getattr(obj, attr)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with cuda_timing_section(stats, key):
            return bound(*args, **kwargs)

    setattr(obj, attr, wrapped)
    return _RevertHandle(obj, attr, original)


# ---------------------------------------------------------------------------
# High-level attach / detach
# ---------------------------------------------------------------------------


def attach_inference_hooks(model: nn.Module, stats: CudaTimerStats) -> list:
    """Attach all inference-timing hooks for any road-cpl model variant.

    Records:

    * ``forward_total_ms`` -- top-level ``model.forward`` (and ``num_examples``).
    * ``backbone_ms`` -- ResNet feature extractor.
    * ``encoder_ms`` -- transformer encoder.
    * ``seq_head_forward_ms`` -- ``cpl_head`` or ``ar_head`` forward (if any).
    * ``decode_ms`` -- ``cpl_head.greedy_decode`` / ``ar_head.greedy_decode``
      (if any), via an external monkey-patch.

    Heads time (heatmap + offset MLPs and any multi-hypothesis fusion) is
    derived in :func:`aggregate_inference_timing` as
    ``forward_total_ms - backbone_ms - encoder_ms - seq_head_forward_ms``.
    """
    handles: list = []
    handles.extend(attach_forward_timing_hooks(model, "forward_total_ms", stats, record_num_examples=True))
    backbone = getattr(model, "backbone", None)
    if isinstance(backbone, nn.Module):
        handles.extend(attach_forward_timing_hooks(backbone, "backbone_ms", stats))
    transformer = getattr(model, "transformer", None)
    if isinstance(transformer, nn.Module):
        handles.extend(attach_forward_timing_hooks(transformer, "encoder_ms", stats))

    seq_head: nn.Module | None = None
    cpl_head = getattr(model, "cpl_head", None)
    ar_head = getattr(model, "ar_head", None)
    if isinstance(cpl_head, nn.Module):
        seq_head = cpl_head
    elif isinstance(ar_head, nn.Module):
        seq_head = ar_head
    if seq_head is not None:
        handles.extend(attach_forward_timing_hooks(seq_head, "seq_head_forward_ms", stats))
        handles.append(_wrap_method_with_timing(seq_head, "greedy_decode", "decode_ms", stats))
    return handles


def detach_forward_hooks(handles: list) -> None:
    for h in handles:
        h.remove()


# ---------------------------------------------------------------------------
# Inference loops (warmup + timed)
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_inference_loop(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_cpl: bool,
    use_ar: bool,
    max_batches: int | None = None,
    desc: str = "  inference",
) -> int:
    """Iterate ``loader`` running ``model(images)`` (+ greedy decode for sequence heads).

    No metric work is done -- this is a pure forward + (optional) decode loop.
    When forward hooks attached by :func:`attach_inference_hooks` are present,
    the loop produces the timing measurements; otherwise it is a no-op warmup.

    Returns the number of batches processed.
    """
    model.eval()
    total = max_batches if max_batches is not None else len(loader)
    seen = 0
    for batch in tqdm(loader, total=total, desc=desc, leave=False):
        if max_batches is not None and seen >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        outputs = model(images)
        if use_cpl:
            model.cpl_greedy_decode(outputs)
        elif use_ar:
            model.ar_greedy_decode(outputs)
        seen += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return seen


def warmup_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_batches: int,
    *,
    use_cpl: bool,
    use_ar: bool,
) -> None:
    """Run ``n_batches`` forward (+ greedy-decode) without timing.

    Intended to be called *before* :func:`attach_inference_hooks` so the
    warmup pass never accumulates into ``stats``. Ensures CUDA kernels,
    cuDNN autotune, and allocator caches are warm before measurement.
    """
    if n_batches <= 0:
        return
    run_inference_loop(
        model,
        loader,
        device,
        use_cpl=use_cpl,
        use_ar=use_ar,
        max_batches=n_batches,
        desc="  warmup",
    )


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------


_PER_BATCH_KEYS: tuple[str, ...] = (
    "forward_total_ms",
    "backbone_ms",
    "encoder_ms",
    "seq_head_forward_ms",
    "decode_ms",
)


def aggregate_inference_timing(stats: CudaTimerStats) -> dict[str, float]:
    """Turn raw ``stats`` into a flat dict suitable for printing or JSON.

    Numbers are reported both per-batch (mean over batches that hit each key)
    and per-example (sum / ``num_examples``). The derived ``heads_ms`` covers
    everything inside the model forward not already attributed to backbone,
    encoder, or the sequence head; ``total_inference_ms`` adds the decode
    loop on top.
    """
    out: dict[str, float] = {}
    nb = max(stats.count("forward_total_ms"), 1)
    ne = float(max(stats.num_examples, 1))

    for key in _PER_BATCH_KEYS:
        c = stats.count(key)
        if c == 0:
            continue
        out[f"avg_batch_{key}"] = stats.sum_ms(key) / float(max(c, 1))
        out[f"per_example_{key}"] = stats.sum_ms(key) / ne

    fwd_sum = stats.sum_ms("forward_total_ms")
    bb_sum = stats.sum_ms("backbone_ms")
    enc_sum = stats.sum_ms("encoder_ms")
    seq_sum = stats.sum_ms("seq_head_forward_ms")
    heads_sum = max(fwd_sum - bb_sum - enc_sum - seq_sum, 0.0)
    out["avg_batch_heads_ms"] = heads_sum / float(nb)
    out["per_example_heads_ms"] = heads_sum / ne

    dec_sum = stats.sum_ms("decode_ms")
    total_sum = fwd_sum + dec_sum
    out["avg_batch_total_inference_ms"] = total_sum / float(nb)
    out["per_example_total_inference_ms"] = total_sum / ne

    out["num_batches_timed"] = float(stats.count("forward_total_ms"))
    out["num_examples_timed"] = float(stats.num_examples)
    return out


def print_inference_timing_report(report: dict[str, float]) -> None:
    """Pretty-print timing aggregates (milliseconds) in two grouped columns."""
    print("\n=== Inference timing (ms) ===")
    keys = sorted(k for k in report if not k.startswith("num_"))
    if not keys:
        print("  (no measurements recorded)")
        print()
        return
    col_w = max(len(k) for k in keys)
    for k in keys:
        print(f"  {k:<{col_w}}  {report[k]:.4f}")
    if "num_batches_timed" in report:
        print(f"  {'num_batches_timed':<{col_w}}  {int(report['num_batches_timed'])}")
    if "num_examples_timed" in report:
        print(f"  {'num_examples_timed':<{col_w}}  {int(report['num_examples_timed'])}")
    print()


def save_inference_timing_report(report: dict[str, float], path: str | Path) -> None:
    """Write the aggregated timing dict to ``path`` as JSON (sorted keys)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")


__all__ = [
    "CudaTimerStats",
    "aggregate_inference_timing",
    "attach_forward_timing_hooks",
    "attach_inference_hooks",
    "cuda_timing_section",
    "detach_forward_hooks",
    "print_inference_timing_report",
    "run_inference_loop",
    "save_inference_timing_report",
    "warmup_inference",
]
