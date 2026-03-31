"""
AutoTune: Automatic kernel parameter optimization for Triton kernels on AMD GPU.
Inspired by AutoKernel (https://github.com/OAID/AutoKernel).

Architecture mirrors AutoKernel:
  SearchSpace  — defines the parameter axes (like LoopNest tiling candidates)
  Config       — a single point in the search space (like a Schedule)
  Benchmark    — actual GPU timing as the cost oracle (replaces the learned cost model)
  AutoTuner    — grid or beam search over configs, with result caching
"""

from __future__ import annotations

import itertools
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import triton


# ---------------------------------------------------------------------------
# Config: one point in the search space
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """
    A kernel launch configuration.

    constexprs: dict of compile-time constant arguments (e.g. BLOCK_M=4)
    num_warps:  warps per block
    num_stages: pipeline depth for async copies (AMD: 1–4 is typical)
    """
    constexprs: Dict[str, Any]
    num_warps: int = 4
    num_stages: int = 2

    def to_triton(self) -> triton.Config:
        return triton.Config(self.constexprs, num_warps=self.num_warps, num_stages=self.num_stages)

    def as_dict(self) -> Dict:
        return {"constexprs": self.constexprs, "num_warps": self.num_warps, "num_stages": self.num_stages}

    @staticmethod
    def from_dict(d: Dict) -> "Config":
        return Config(d["constexprs"], d["num_warps"], d["num_stages"])

    def __hash__(self):
        return hash((tuple(sorted(self.constexprs.items())), self.num_warps, self.num_stages))

    def __repr__(self):
        kw = ", ".join(f"{k}={v}" for k, v in self.constexprs.items())
        return f"Config({kw}, warps={self.num_warps}, stages={self.num_stages})"


# ---------------------------------------------------------------------------
# SearchSpace: defines axes and generates all candidate configs
# ---------------------------------------------------------------------------

class SearchSpace:
    """
    Declares the parameter axes to search over.

    Example
    -------
    space = (SearchSpace()
             .add_axis("BLOCK_M", [1, 2, 4, 8, 16])
             .set_warps([1, 2, 4, 8])
             .set_stages([1, 2, 3]))
    configs = space.generate()
    """

    def __init__(self):
        self._axes: Dict[str, List[Any]] = {}
        self._num_warps: List[int] = [4]
        self._num_stages: List[int] = [2]

    def add_axis(self, name: str, values: List[Any]) -> "SearchSpace":
        self._axes[name] = values
        return self

    def set_warps(self, values: List[int]) -> "SearchSpace":
        self._num_warps = values
        return self

    def set_stages(self, values: List[int]) -> "SearchSpace":
        self._num_stages = values
        return self

    def generate(self) -> List[Config]:
        """Cartesian product of all axes × warps × stages."""
        names = list(self._axes.keys())
        value_lists = [self._axes[n] for n in names]
        axes_combos = list(itertools.product(*value_lists)) if names else [()]

        configs = []
        for combo in axes_combos:
            kwargs = dict(zip(names, combo)) if names else {}
            for w in self._num_warps:
                for s in self._num_stages:
                    configs.append(Config(kwargs, w, s))
        return configs

    def __len__(self):
        return len(self.generate())


# ---------------------------------------------------------------------------
# Benchmark: GPU timing utility
# ---------------------------------------------------------------------------

def benchmark(fn: Callable, warmup: int = 25, rep: int = 100) -> float:
    """
    Measure the median execution time of fn() in milliseconds.

    Uses CUDA events for sub-millisecond precision. Sorts timings and
    returns the median to be robust against occasional GPU stalls.
    """
    # Warmup — let the GPU reach steady state and JIT-compile the kernel
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    times: List[float] = []
    for _ in range(rep):
        start_ev.record()
        fn()
        end_ev.record()
        torch.cuda.synchronize()
        times.append(start_ev.elapsed_time(end_ev))

    times.sort()
    return times[len(times) // 2]  # median


# ---------------------------------------------------------------------------
# AutoTuner: search + cache
# ---------------------------------------------------------------------------

class AutoTuner:
    """
    Finds the best Config for a kernel on this GPU.

    Inspired by AutoKernel's approach of exploring a tiling schedule space
    with actual hardware performance as the cost oracle.  For spaces that
    fit in a few hundred configs, exhaustive grid search is used.  For
    larger spaces, beam search prunes to the top-k after a quick pass.

    Parameters
    ----------
    space       : SearchSpace defining candidate configs
    cache_path  : optional JSON file path to persist best configs
    warmup / rep: benchmark timing parameters
    beam_size   : keep top-k configs after the quick pass (None = exhaustive)
    """

    def __init__(
        self,
        space: SearchSpace,
        cache_path: Optional[str] = None,
        warmup: int = 25,
        rep: int = 100,
        beam_size: Optional[int] = None,
    ):
        self.space = space
        self.cache_path = cache_path
        self.warmup = warmup
        self.rep = rep
        self.beam_size = beam_size
        self._cache: Dict[str, Config] = {}

        if cache_path and os.path.exists(cache_path):
            self._load_cache()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_cache(self):
        with open(self.cache_path) as f:
            raw = json.load(f)
        for k, v in raw.items():
            self._cache[k] = Config.from_dict(v)

    def _save_cache(self):
        if not self.cache_path:
            return
        raw = {k: c.as_dict() for k, c in self._cache.items()}
        with open(self.cache_path, "w") as f:
            json.dump(raw, f, indent=2)

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def tune(
        self,
        build_fn: Callable[[Config], Callable],
        cache_key: str,
        verbose: bool = True,
    ) -> Tuple[Config, float]:
        """
        Find the best Config for the kernel described by build_fn.

        Parameters
        ----------
        build_fn  : Given a Config, returns a zero-argument callable that
                    runs the kernel (e.g. a lambda capturing tensors).
                    Raise any exception on invalid configs — they are skipped.
        cache_key : A string that uniquely identifies the problem size,
                    e.g. "quant_M1024_K4096".  Results are cached per key.
        verbose   : Print progress to stdout.

        Returns
        -------
        (best_config, best_time_ms)
        """
        # --- Cache hit ---
        if cache_key in self._cache:
            cfg = self._cache[cache_key]
            if verbose:
                print(f"[AutoTune] Cache hit for '{cache_key}': {cfg}")
            fn = build_fn(cfg)
            t = benchmark(fn, self.warmup, self.rep)
            return cfg, t

        candidates = self.space.generate()
        if verbose:
            print(f"[AutoTune] Searching {len(candidates)} configs for key='{cache_key}' ...")

        # --- Phase 1: quick pass (few reps) to prune bad configs ---
        scored: List[Tuple[float, Config]] = []
        quick_warmup, quick_rep = 3, 10

        for i, cfg in enumerate(candidates):
            try:
                fn = build_fn(cfg)
                t = benchmark(fn, quick_warmup, quick_rep)
                scored.append((t, cfg))
                if verbose:
                    tag = " *" if t == min(s for s, _ in scored) else ""
                    print(f"  [{i+1:3d}/{len(candidates)}] {cfg}  {t:.3f} ms{tag}")
            except Exception as exc:
                if verbose:
                    print(f"  [{i+1:3d}/{len(candidates)}] {cfg}  SKIP ({exc})")

        if not scored:
            raise RuntimeError("[AutoTune] All configs failed.")

        scored.sort(key=lambda x: x[0])

        # --- Phase 2: beam — re-benchmark top-k with full reps ---
        finalists = scored[: self.beam_size] if self.beam_size else scored
        if verbose and self.beam_size:
            print(f"[AutoTune] Re-benchmarking top-{len(finalists)} candidates ...")

        best_time = float("inf")
        best_cfg: Optional[Config] = None

        for t_quick, cfg in finalists:
            try:
                fn = build_fn(cfg)
                t = benchmark(fn, self.warmup, self.rep)
                if verbose and self.beam_size:
                    print(f"  {cfg}  {t:.3f} ms")
                if t < best_time:
                    best_time = t
                    best_cfg = cfg
            except Exception:
                pass

        if best_cfg is None:
            raise RuntimeError("[AutoTune] All finalist configs failed.")

        if verbose:
            print(f"[AutoTune] Best: {best_cfg}  {best_time:.3f} ms")

        self._cache[cache_key] = best_cfg
        self._save_cache()
        return best_cfg, best_time

    # ------------------------------------------------------------------
    # Convenience: tune once per unique input shape and reuse
    # ------------------------------------------------------------------

    def get_best_config(
        self,
        build_fn: Callable[[Config], Callable],
        cache_key: str,
        verbose: bool = True,
    ) -> Config:
        cfg, _ = self.tune(build_fn, cache_key, verbose=verbose)
        return cfg
