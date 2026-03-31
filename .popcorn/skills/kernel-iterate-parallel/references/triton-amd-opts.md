# Triton Optimization Techniques for AMD MI300X/MI355X

## Tiling & Occupancy

- **BLOCK_M / BLOCK_N / BLOCK_K**: start at 64/128/32, tune in powers of 2. MI355X has 256KB LDS per CU.
- **num_warps**: 4–8 typical; 8 often better for memory-bound kernels.
- **num_stages**: pipeline depth. 2–4 for MXFP4 quant; higher risks register spill.
- Use `@triton.autotune` with a grid of configs to find best tile sizes automatically.

## Memory Access

- **Coalesced loads**: ensure the innermost dimension stride is 1 (contiguous). Use `.contiguous()` on inputs.
- **Vectorized loads**: use `tl.load` with `tl.constexpr` width = 4 or 8 (128-bit/256-bit).  
- **LDS reuse**: load a tile into shared memory once, reuse across multiple output elements.
- **Avoid strided stores**: pack results in registers before a single coalesced store.

## Compute

- **`tl.dot`**: maps to MFMA (matrix fused multiply-add) on CDNA. Use for any MxKxN ≥ 16x16x16.  
- **FP32 accumulation**: always accumulate in f32 even when inputs are bf16/fp16 to avoid precision loss.
- **In-register packing**: use bitwise ops (`|`, `<<`, `&`) to pack nibbles without extra memory round-trips.

## AMD-Specific Flags

```python
@triton.jit
def kernel(...):
    ...

# Force wave64 (default on AMD; don't change unless profiling shows otherwise)
# Use triton-profiler or rocprof for roofline analysis
```

- `matrix_instr_nonkdim`: controls MFMA shape, set via `tl.extra.cuda.libdevice` or config.
- Wave32 vs Wave64: MI300X defaults to Wave64; Wave32 can help for small tiles.

## Quantization Kernel Specifics (MXFP4)

- **Group size 32**: one E8M0 scale per 32 elements. Launch grid as `(M, K // 32)`.
- **E8M0 = power-of-2 exponent**: compute `floor(log2(abs_max / fp4_max)) + 127`, clamp `[0, 255]`.
- **FP4 E2M1 thresholds**: 0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0 (midpoints between representable values).
- **Nibble packing**: `lo | (hi << 4)` — confirm byte order matches `gemm_a4w4`'s expected layout.
- **Fused quant+shuffle**: if `e8m0_shuffle` is deterministic, absorb it into the Triton kernel to save a kernel launch.

## Profiling Commands (AMD)

```bash
# Quick latency
rocprof --stats python run_kernel.py

# Full trace
rocprof -d trace/ --hsa-trace python run_kernel.py

# Popcorn built-in benchmark
popcorn run submission.py --benchmark
```

## Common Bottlenecks & Fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| Low TFLOPS, high latency | Memory bound | Increase tile size, vectorize loads |
| Correctness fails | Wrong nibble order or scale bias | Check `lo/hi` packing, E8M0 bias=127 |
| Register spill | Too many pipeline stages | Reduce `num_stages` |
| Low occupancy | Large tiles + many registers | Reduce BLOCK_M or num_warps |
