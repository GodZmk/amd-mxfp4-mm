#!/usr/bin/env python3
"""Parse popcorn CLI benchmark output and emit structured results."""

import re
import sys
import json


def parse(output: str) -> dict:
    result = {
        "passed": None,
        "tflops": None,
        "latency_ms": None,
        "vs_reference": None,
        "raw": output,
    }

    # Correctness
    if re.search(r"PASSED|correct|✓", output, re.IGNORECASE):
        result["passed"] = True
    elif re.search(r"FAILED|incorrect|✗|mismatch", output, re.IGNORECASE):
        result["passed"] = False

    # TFLOPS
    m = re.search(r"([\d.]+)\s*TFLOPS", output, re.IGNORECASE)
    if m:
        result["tflops"] = float(m.group(1))

    # Latency
    m = re.search(r"([\d.]+)\s*ms", output, re.IGNORECASE)
    if m:
        result["latency_ms"] = float(m.group(1))

    # vs reference (e.g. "1.23x faster than reference")
    m = re.search(r"([\d.]+)x\s*(faster|slower)", output, re.IGNORECASE)
    if m:
        ratio = float(m.group(1))
        result["vs_reference"] = ratio if "faster" in m.group(2).lower() else -ratio

    return result


def summarize(r: dict) -> str:
    lines = []
    lines.append(f"Correctness : {'PASS' if r['passed'] else 'FAIL' if r['passed'] is False else 'unknown'}")
    if r["tflops"] is not None:
        lines.append(f"Performance : {r['tflops']:.2f} TFLOPS")
    if r["latency_ms"] is not None:
        lines.append(f"Latency     : {r['latency_ms']:.3f} ms")
    if r["vs_reference"] is not None:
        sign = "faster" if r["vs_reference"] > 0 else "slower"
        lines.append(f"vs reference: {abs(r['vs_reference']):.2f}x {sign}")
    return "\n".join(lines)


if __name__ == "__main__":
    text = sys.stdin.read() if len(sys.argv) == 1 else open(sys.argv[1]).read()
    r = parse(text)
    print(summarize(r))
    print("\n--- JSON ---")
    print(json.dumps(r, indent=2))
