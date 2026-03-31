#!/usr/bin/env python3
"""
Auto-update kernel-iterate-parallel SKILL.md based on kernel_iterate_log.md.
Runs on conversation Stop / autocompact. Updates the "## Learned from This Project"
section with strategy win/fail stats derived from the log.
"""

import re
import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent  # mxfp4-mm/
SKILL_MD = Path(__file__).parent.parent / "SKILL.md"
LOG_MD = PROJECT_ROOT / "kernel_iterate_log.md"
STATE_FILE = Path(__file__).parent / ".last_update_state.json"

SECTION_MARKER_START = "## Learned from This Project"
SECTION_MARKER_END = "<!-- end-learned -->"


def parse_log(log_path: Path) -> list[dict]:
    """Parse kernel_iterate_log.md table rows into dicts."""
    if not log_path.exists():
        return []

    rows = []
    in_table = False
    for line in log_path.read_text().splitlines():
        # Skip header and separator rows
        if line.startswith("| Iter") or line.startswith("|---"):
            in_table = True
            continue
        if not in_table or not line.startswith("|"):
            continue

        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 6:
            continue

        try:
            iter_num = int(re.sub(r"\D", "", parts[0])) if parts[0].strip() else 0
        except ValueError:
            continue

        strategy_raw = parts[1].strip()
        is_winner = "★" in strategy_raw
        strategy = strategy_raw.replace("★", "").strip()

        tflops_str = parts[2].strip().replace("-", "")
        latency_str = parts[3].strip().replace("-", "")
        pass_str = parts[5].strip().upper()

        rows.append({
            "iter": iter_num,
            "strategy": strategy,
            "winner": is_winner,
            "tflops": float(tflops_str) if tflops_str else None,
            "latency_ms": float(latency_str) if latency_str else None,
            "passed": "PASS" in pass_str,
            "notes": parts[6].strip() if len(parts) > 6 else "",
        })
    return rows


def compute_stats(rows: list[dict]) -> dict:
    """Compute per-strategy win/fail/tflops stats."""
    stats: dict[str, dict] = {}
    for row in rows:
        if row["iter"] == 0:
            continue
        s = row["strategy"]
        if s not in stats:
            stats[s] = {"wins": 0, "tries": 0, "fails": 0, "best_tflops": None, "tflops_gains": []}
        stats[s]["tries"] += 1
        if not row["passed"]:
            stats[s]["fails"] += 1
        elif row["winner"]:
            stats[s]["wins"] += 1
            if row["tflops"] is not None:
                if stats[s]["best_tflops"] is None or row["tflops"] > stats[s]["best_tflops"]:
                    stats[s]["best_tflops"] = row["tflops"]

    # Compute best overall TFLOPS across all winning rows
    winning_tflops = [r["tflops"] for r in rows if r["winner"] and r["tflops"] is not None]
    best_overall = max(winning_tflops) if winning_tflops else None
    baseline_rows = [r for r in rows if r["iter"] == 0 and r["tflops"] is not None]
    baseline = baseline_rows[0]["tflops"] if baseline_rows else None

    return {
        "strategies": stats,
        "best_tflops": best_overall,
        "baseline_tflops": baseline,
        "total_iters": max((r["iter"] for r in rows), default=0),
    }


def build_learned_section(stats: dict, rows: list[dict]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        SECTION_MARKER_START,
        f"",
        f"*Auto-updated: {now}*",
        f"",
    ]

    s = stats["strategies"]
    total = stats["total_iters"]
    best = stats["best_tflops"]
    baseline = stats["baseline_tflops"]

    if total > 0:
        lines.append(f"**Progress**: {total} iteration(s) completed")
        if best is not None and baseline is not None and baseline > 0:
            gain = (best - baseline) / baseline * 100
            lines.append(f"**Best TFLOPS**: {best:.2f} ({gain:+.1f}% vs baseline {baseline:.2f})")
        elif best is not None:
            lines.append(f"**Best TFLOPS**: {best:.2f}")
        lines.append("")

    if s:
        lines.append("### Strategy Results")
        lines.append("")
        lines.append("| Strategy | Tries | Wins | Fails | Best TFLOPS | Status |")
        lines.append("|----------|-------|------|-------|-------------|--------|")
        for name, info in sorted(s.items(), key=lambda x: -x[1]["wins"]):
            win_rate = info["wins"] / info["tries"] * 100 if info["tries"] > 0 else 0
            tflops_str = f"{info['best_tflops']:.2f}" if info["best_tflops"] else "-"
            if info["wins"] > 0:
                status = "✓ effective"
            elif info["fails"] == info["tries"]:
                status = "✗ avoid"
            else:
                status = "~ mixed"
            lines.append(
                f"| `{name}` | {info['tries']} | {info['wins']} | {info['fails']} | {tflops_str} | {status} |"
            )
        lines.append("")

    # Recent winning sequence
    winners = [r for r in rows if r["winner"] and r["iter"] > 0]
    if winners:
        lines.append("### Winning Sequence")
        lines.append("")
        for w in winners[-5:]:  # last 5
            tflops_str = f"{w['tflops']:.2f} TFLOPS" if w["tflops"] else "?"
            lines.append(f"- Iter {w['iter']}: `{w['strategy']}` → {tflops_str}")
        lines.append("")

    lines.append(SECTION_MARKER_END)
    return "\n".join(lines)


def update_skill_md(skill_path: Path, new_section: str) -> bool:
    """Replace or append the learned section in SKILL.md. Returns True if changed."""
    content = skill_path.read_text()

    if SECTION_MARKER_START in content:
        # Replace existing section (from marker to end marker or EOF)
        if SECTION_MARKER_END in content:
            pattern = rf"{re.escape(SECTION_MARKER_START)}.*?{re.escape(SECTION_MARKER_END)}"
            new_content = re.sub(pattern, new_section, content, flags=re.DOTALL)
        else:
            # Replace from marker to end of file
            idx = content.index(SECTION_MARKER_START)
            new_content = content[:idx] + new_section
    else:
        # Append at end
        new_content = content.rstrip() + "\n\n---\n\n" + new_section + "\n"

    if new_content == content:
        return False
    skill_path.write_text(new_content)
    return True


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def main():
    # Only run if there's a log to process
    if not LOG_MD.exists():
        return 0

    rows = parse_log(LOG_MD)
    if not rows:
        return 0

    # Check if log changed since last update (avoid no-op updates)
    log_mtime = LOG_MD.stat().st_mtime
    state = load_state()
    if state.get("last_log_mtime") == log_mtime:
        return 0  # nothing new

    stats = compute_stats(rows)
    section = build_learned_section(stats, rows)

    if not SKILL_MD.exists():
        print(f"[update_skill] SKILL.md not found at {SKILL_MD}", file=sys.stderr)
        return 1

    changed = update_skill_md(SKILL_MD, section)
    if changed:
        save_state({"last_log_mtime": log_mtime, "updated_at": datetime.now().isoformat()})
        print(f"[update_skill] SKILL.md updated ({stats['total_iters']} iters, best={stats['best_tflops']})")
    else:
        print("[update_skill] SKILL.md already up-to-date")

    return 0


if __name__ == "__main__":
    sys.exit(main())
