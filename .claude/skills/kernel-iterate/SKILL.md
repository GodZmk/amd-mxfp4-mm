---
name: kernel-iterate
description: Autonomous GPU kernel optimization loop for Popcorn leaderboard submissions. Iteratively improves Triton/HIP kernels in submission.py by running popcorn CLI benchmarks, parsing performance results, and applying targeted optimizations informed by AMD MI300X/MI355X architecture and MX format papers. Use when the user says "iterate", "keep optimizing", "run until faster", or "autonomously improve" on a GPU kernel submission.
---

# Kernel Iterate

Autonomous loop: restore context → analyze → optimize → benchmark → save snapshot → log → repeat.

## Workflow

### 1. Restore Context from Log

**Always start here — never re-run a baseline benchmark if the log already exists.**

```bash
cat kernel_iterate_log.md 2>/dev/null || echo "No log yet"
```

Parse the log to extract:
- **CURRENT_ITER**: highest iter number seen (next iteration = CURRENT_ITER + 1)
- **BEST_TFLOPS**: highest TFLOPS in any passing row
- **BEST_ITER**: which iter produced best TFLOPS
- **STRATEGIES_TRIED**: all changes already attempted
- **Last session summary**: what worked, what didn't, suggested next directions

Also scan `submissions/` to find the latest snapshot version and read the most recent session summary:
```bash
ls submissions/ 2>/dev/null | sort
# Read the latest session summary if present
LATEST=$(ls submissions/ 2>/dev/null | sort | tail -1)
cat submissions/${LATEST}/session_summary.md 2>/dev/null || true
```

If `kernel_iterate_log.md` does not exist, create it with this header and treat CURRENT_ITER = 0, BEST_TFLOPS = unknown:

```markdown
# Kernel Iterate Log — <submission name>

| Iter | Change | TFLOPS | Latency | Pass |
|------|--------|--------|---------|------|
```

Then run a single baseline benchmark to establish the starting point:
```bash
popcorn submit --mode benchmark --no-tui submission.py 2>&1 | tee /tmp/bench_baseline.txt
python3 .claude/skills/kernel-iterate/scripts/parse_benchmark.py /tmp/bench_baseline.txt
```

Append baseline row to log:
```
| 0 | baseline | <tflops> | <latency>ms | ✓/✗ |
```

---

### 2. Analyze Bottleneck

Read `submission.py` and the last session summary from the log. Classify current state:

- **Memory-bound**: low arithmetic intensity → increase tile size, vectorize loads
- **Compute-bound**: near peak TFLOPS → reduce overhead, check packing
- **Correctness failure**: wrong output → check nibble order, E8M0 bias, scale application

Read `.claude/skills/kernel-iterate/references/triton-amd-opts.md` for the relevant fix pattern.

Cross-check against `STRATEGIES_TRIED` — do not repeat a strategy that already failed or had no effect.

---

### 3. Apply One Targeted Optimization

Edit `submission.py` with a single focused change. Priority order:

| Priority | Optimization |
|---|---|
| 1 | Fix correctness (must pass before tuning) |
| 2 | Fuse scale shuffle into quant kernel |
| 3 | Vectorize loads (128-bit / 256-bit) |
| 4 | Increase BLOCK_M, add `num_stages` pipelining |
| 5 | `@triton.autotune` over tile configs |
| 6 | Wave32 mode or `num_warps` tuning |

---

### 4. Save Submission Snapshot

Before benchmarking, save the current code as a versioned snapshot:

```bash
ITER=<current_iter_number>
mkdir -p submissions/submission_${ITER}
cp submission.py submissions/submission_${ITER}/submission.py
echo "<brief description of change>" > submissions/submission_${ITER}/notes.txt
```

---

### 5. Benchmark and Compare

```bash
popcorn submit --mode benchmark --no-tui submission.py 2>&1 | tee /tmp/bench_iter${ITER}.txt
python3 .claude/skills/kernel-iterate/scripts/parse_benchmark.py /tmp/bench_iter${ITER}.txt
```

Compare vs `BEST_TFLOPS`. If correctness regressed, revert `submission.py` from the previous snapshot and try a different change:
```bash
cp submissions/submission_${PREV_ITER}/submission.py submission.py
```

---

### 6. Update Log

Append a row to `kernel_iterate_log.md`:

```bash
cat >> kernel_iterate_log.md << 'EOF'
| <N> | <change description> | <tflops> | <latency>ms | ✓/✗ |
EOF
```

---

### 7. Decide: Continue or Stop

**Continue** if: TFLOPS improved AND correctness passes AND more optimizations remain.

**Stop** if:
- Performance plateaued (< 2% gain for 2 consecutive iterations)
- All priority optimizations exhausted
- User's target reached
- User says stop

On stop, write the session summary to **two places**:

**1. A dedicated summary file** (versioned, human-readable):
```bash
mkdir -p submissions/submission_${LAST_ITER}
cat > submissions/submission_${LAST_ITER}/session_summary.md << 'EOF'
# Session Summary — v<LAST_ITER> — <date>

**Version range**: submission_<FIRST_ITER_THIS_SESSION> → submission_<LAST_ITER>
**Session TFLOPS**: <start_tflops> → <best_tflops_this_session> (<delta>%)

**Experimental direction**: <one sentence>

**Best result**: Iter <N> — <change> — <TFLOPS> TFLOPS (<latency>ms) ✓

**What worked**: 
- <change>: +X% TFLOPS

**What didn't**:
- <change>: reverted / no effect

**Next directions**:
- <suggested next steps>
EOF
```

**2. Appended to `kernel_iterate_log.md`** (searchable history):
```markdown
## Session Summary — v<LAST_ITER> — <date>
...same content...
```

Then print the full table from the log as the final output.

---

## References

- **AMD Triton opts**: `.claude/skills/kernel-iterate/references/triton-amd-opts.md`
- **Benchmark parser**: `.claude/skills/kernel-iterate/scripts/parse_benchmark.py`
- **Iteration log**: `kernel_iterate_log.md`
- **Snapshots**: `submissions/submission_<N>/`

## Constraints

- Never sacrifice correctness for speed
- One change per iteration so causality is clear
- If `popcorn` is unavailable, run `source /Users/zhumingkai/.bashrc` and then try again
- Always write the session summary when stopping, even if only one iteration ran
- Always save a snapshot before benchmarking — snapshots are the only way to revert
