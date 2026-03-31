---
name: kernel-iterate-parallel
description: Parallel GPU kernel optimization loop for Popcorn leaderboard. Each iteration: benchmark → analyze → select 3 optimization directions → dispatch 3 parallel sub-agents → compare results → commit winner → update log. Logs every iteration to kernel_iterate_log.md and saves all candidate code snapshots to iterations/. Use when the user wants parallel/multi-agent kernel optimization, says "parallel iterate", "try multiple optimizations", or "dispatch sub-agents to optimize".
---

# Kernel Iterate (Parallel Multi-Agent)

Each round: benchmark → pick 3 strategies → 3 parallel sub-agents → best wins → commit → log → repeat.

## Prerequisites

```bash
source /Users/zhumingkai/.bashrc  # if popcorn not in PATH
mkdir -p iterations
```

---

## Phase 1 — Establish Baseline

```bash
popcorn submit --mode benchmark --no-tui submission.py 2>&1 | tee /tmp/bench_baseline.txt
python3 .claude/skills/kernel-iterate-parallel/scripts/parse_benchmark.py /tmp/bench_baseline.txt
```

Initialize `kernel_iterate_log.md` header row if the file is empty:

```markdown
# Kernel Iterate Log — amd-mxfp4-mm

| Iter | Strategy | TFLOPS | Latency (ms) | vs Ref | Pass | Notes |
|------|----------|--------|--------------|--------|------|-------|
| 0    | baseline | -      | -            | -      | ?    | initial state |
```

Record baseline row immediately. Track `BEST_TFLOPS` and `CURRENT_ITER = 0`.

---

## Phase 2 — Select 3 Optimization Strategies

After each benchmark, classify the bottleneck and choose **3 distinct strategies** not yet tried:

### Bottleneck Classification

| Symptom | Bottleneck | Strategy pool |
|---------|-----------|---------------|
| TFLOPS < 20% peak, high latency | Memory-bound | `vectorize-loads`, `increase-tiling`, `fuse-shuffle` |
| TFLOPS near peak | Compute-bound | `autotune-configs`, `wave32-warps`, `pipeline-stages` |
| Correctness fails | Bug | `fix-nibble-order`, `fix-e8m0-bias`, `fix-scale-apply` |
| Moderate TFLOPS + correctness OK | Mixed | `fuse-shuffle`, `autotune-configs`, `pipeline-stages` |

### Strategy Catalog

| Strategy ID | Description |
|-------------|-------------|
| `vectorize-loads` | Use 128-bit/256-bit vectorized tl.load; BLOCK size multiple of 4/8 |
| `increase-tiling` | Increase BLOCK_M to 2/4/8; add num_stages=2 |
| `fuse-shuffle` | Absorb e8m0_shuffle into Triton quant kernel to eliminate extra kernel launch |
| `autotune-configs` | Add @triton.autotune with grid of BLOCK_M/num_warps/num_stages configs |
| `wave32-warps` | Switch to Wave32 mode; tune num_warps from 4 to 8 |
| `pipeline-stages` | Tune num_stages 1→2→3; add software pipelining |
| `fix-nibble-order` | Verify/fix lo/hi nibble packing order to match gemm_a4w4 expected layout |
| `fix-e8m0-bias` | Fix E8M0 bias (should be 127); verify scale exponent computation |
| `fix-scale-apply` | Fix scale application logic; verify quant_scale formula |
| `reduce-overhead` | Minimize Python-side overhead; optimize kernel launch grid |

**Rules:**
- Never repeat a strategy already tried in a previous winning iteration
- If correctness fails, ALL 3 slots go to correctness-fix strategies
- Choose strategies addressing different aspects of the bottleneck

---

## Phase 3 — Dispatch 3 Parallel Sub-Agents

Spawn **3 sub-agents in parallel** using `run_in_background=True` and `isolation="worktree"`.

Each sub-agent receives this prompt (fill in the placeholders):

```
You are a GPU kernel optimization sub-agent for the amd-mxfp4-mm Popcorn leaderboard project.

Project directory: /Users/zhumingkai/amd_202602/mxfp4-mm
Iteration: <N>
Assigned strategy: <strategy_id>
Current best TFLOPS: <X>

Your task — follow these steps exactly:

1. Read submission.py to understand the current kernel.

2. Apply ONLY the "<strategy_id>" optimization (one focused change).
   Reference: .claude/skills/kernel-iterate-parallel/references/triton-amd-opts.md

3. Create the snapshot directory:
   mkdir -p iterations/iter_<N>_<strategy_id>

4. Save your modified code:
   Write your modified submission.py to: iterations/iter_<N>_<strategy_id>/submission.py

5. Run the benchmark (source bashrc first if needed):
   source /Users/zhumingkai/.bashrc 2>/dev/null; popcorn submit --mode benchmark --no-tui iterations/iter_<N>_<strategy_id>/submission.py 2>&1 | tee /tmp/bench_<N>_<strategy_id>.txt

6. Parse the result:
   python3 .claude/skills/kernel-iterate-parallel/scripts/parse_benchmark.py /tmp/bench_<N>_<strategy_id>.txt

7. Save bench output:
   cp /tmp/bench_<N>_<strategy_id>.txt iterations/iter_<N>_<strategy_id>/bench_result.txt

8. Report back a JSON summary (this is your final output):
   {"iter": <N>, "strategy": "<strategy_id>", "tflops": <float_or_null>, "latency_ms": <float_or_null>, "passed": <true/false>, "notes": "<brief explanation of change made>"}

Constraints:
- Apply exactly ONE optimization change
- Only write to iterations/iter_<N>_<strategy_id>/ — do NOT modify submission.py in the project root
- Never sacrifice correctness for speed
- If benchmark fails entirely: {"iter": <N>, "strategy": "<strategy_id>", "tflops": null, "passed": false, "notes": "benchmark error: <reason>"}
```

---

## Phase 4 — Compare Results & Pick Winner

After all 3 sub-agents complete, collect their JSON results. Selection:

1. **Filter**: keep only `passed == true`
2. **Rank**: by `tflops` descending
3. **Winner**: highest TFLOPS among passing strategies
4. **Tie-break**: lower `latency_ms` wins
5. **If all fail**: keep current `submission.py`; log all 3 as failed; try different strategies next round

Apply winner to project root:
```bash
cp iterations/iter_<N>_<winner_strategy>/submission.py submission.py
```

---

## Phase 5 — Update Log & Commit

### Update kernel_iterate_log.md

Append one row per strategy tried (winner marked):

```markdown
| <N>  | <winner_strategy> ★  | <tflops> | <latency> | <vs_ref> | PASS | +X.X% vs iter<N-1> |
| <N>  | <strategy2>          | <tflops> | <latency> | <vs_ref> | PASS | runner-up          |
| <N>  | <strategy3>          | <tflops> | <latency> | <vs_ref> | FAIL | correctness issue  |
```

### Git Commit

```bash
git add submission.py kernel_iterate_log.md iterations/iter_<N>_*/
git commit -m "iter<N>: <winner_strategy> — <tflops> TFLOPS (+X% vs iter<N-1>)"
git push origin main
```

---

## Phase 6 — Decide: Continue or Stop

**Continue** if:
- Winner TFLOPS > previous best (any improvement)
- Untried strategies remain
- User has not said stop

**Stop** if:
- Performance plateaued: < 2% gain for 2 consecutive rounds
- All strategies in catalog exhausted
- User's TFLOPS target reached
- User says stop

On stop: print the full `kernel_iterate_log.md` table as the final summary.

---

## Iteration State (maintain in memory across rounds)

```
CURRENT_ITER        = 0
BEST_TFLOPS         = <baseline>
STRATEGIES_TRIED    = []   # accumulate winning strategies
CONSECUTIVE_NO_GAIN = 0
```

---

## References

- **AMD Triton opts**: `references/triton-amd-opts.md`
- **Benchmark parser**: `scripts/parse_benchmark.py`
- **Snapshots**: `iterations/iter_N_strategy/` — all candidate code preserved
- **Log**: `kernel_iterate_log.md` — source of truth for all results
