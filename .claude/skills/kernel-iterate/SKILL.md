---
name: kernel-iterate
description: Autonomous GPU kernel optimization loop for Popcorn leaderboard submissions. Iteratively improves Triton/HIP kernels in submission.py by running popcorn CLI benchmarks, parsing performance results, and applying targeted optimizations informed by AMD MI300X/MI355X architecture and MX format papers. Use when the user says "iterate", "keep optimizing", "run until faster", or "autonomously improve" on a GPU kernel submission.
---

# Kernel Iterate

Autonomous loop: benchmark → analyze → optimize → repeat.

## Workflow

### 1. Establish baseline

```bash
popcorn submit --mode benchmark --no-tui submission.py 2>&1 | tee /tmp/bench_baseline.txt
python3 .claude/skills/kernel-iterate/scripts/parse_benchmark.py /tmp/bench_baseline.txt
```

Record TFLOPS, latency_ms, correctness. This is the target to beat.

### 2. Analyze bottleneck

Read `submission.py` and classify:

- **Memory-bound**: low arithmetic intensity → increase tile size, vectorize loads
- **Compute-bound**: near peak TFLOPS → reduce overhead, check packing
- **Correctness failure**: wrong output → check nibble order, E8M0 bias, scale application

Read `references/triton-amd-opts.md` for the relevant fix pattern.

### 3. Apply one targeted optimization

Edit `submission.py` with a single focused change. Priority order:

| Priority | Optimization |
|---|---|
| 1 | Fix correctness (must pass before tuning) |
| 2 | Fuse scale shuffle into quant kernel |
| 3 | Vectorize loads (128-bit / 256-bit) |
| 4 | Increase BLOCK_M, add `num_stages` pipelining |
| 5 | `@triton.autotune` over tile configs |
| 6 | Wave32 mode or `num_warps` tuning |

### 4. Benchmark and compare

```bash
popcorn submit --mode benchmark --no-tui submission.py 2>&1 | tee /tmp/bench_new.txt
python3 .claude/skills/kernel-iterate/scripts/parse_benchmark.py /tmp/bench_new.txt
```

Compare vs previous best. If correctness regressed, revert and try a different change.

### 5. Decide: continue or stop

**Continue** if: TFLOPS improved AND correctness passes AND more optimizations remain.

**Stop** if:
- Performance plateaued (< 2% gain for 2 consecutive iterations)
- All priority optimizations exhausted
- User's target reached
- User says stop

Print a summary table of all iterations when stopping.

## Iteration Log (keep in memory AND persist to file)

```
| Iter | Change | TFLOPS | Latency | Pass |
|------|--------|--------|---------|------|
| 0    | baseline | --   | --      | ?    |
```

### Log file: `kernel_iterate_log.md`

At the **start** of each session: read `kernel_iterate_log.md` if it exists to restore prior context.

After **each iteration**: append a row to the log file:

```bash
# Append iteration row (example)
cat >> kernel_iterate_log.md << 'EOF'
| 1 | Increase BLOCK_M 64→128 | 12.3 | 4.2ms | ✓ |
EOF
```

At the **end** of the session (stop condition reached): append a session summary block:

```markdown
## Session Summary — <date>

**Experimental direction**: <one sentence describing what was tried this session>

**Best result**: Iter N — <change> — <TFLOPS> TFLOPS (<latency>ms) ✓

**What worked**: <bullet list of changes that improved performance>

**What didn't**: <bullet list of changes that were reverted or had no effect>

**Next directions**: <suggested next steps for future sessions>
```

If `kernel_iterate_log.md` does not exist yet, create it with this header before the first iteration:

```markdown
# Kernel Iterate Log — <submission name>

| Iter | Change | TFLOPS | Latency | Pass |
|------|--------|--------|---------|------|
```

## References

- **AMD Triton opts**: `references/triton-amd-opts.md` — read at start of each iteration
- **Benchmark parser**: `scripts/parse_benchmark.py` — pipe popcorn output through this
- **Iteration log**: `kernel_iterate_log.md` in the working directory

## Constraints

- Never sacrifice correctness for speed
- One change per iteration so causality is clear
- If `popcorn` is unavailable, run `source /Users/zhumingkai/.bashrc` and then try again
- Always write the session summary when stopping, even if only one iteration ran
