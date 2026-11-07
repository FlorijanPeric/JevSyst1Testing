"""Read the logs of run_final.sh and write a PASS/FAIL report (Markdown).

    .venv/bin/python experiments/final_report.py final > final/REPORT.md
The bars were set before the final run, from rounds 3-8 (see CHAT_NOTES.md), slightly below what those reached.
"""
import re
import statistics
import sys
from datetime import date
from pathlib import Path

DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "final")
rows: list[tuple[str, str, str, str]] = []  # (area, check, result, PASS/FAIL/INFO)


def read(name: str) -> str:
    p = DIR / name
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def check(area: str, name: str, result: str, ok: bool | None) -> None:
    rows.append((area, name, result, "INFO" if ok is None else ("PASS" if ok else "FAIL")))


def fraction(pattern: str, text: str) -> tuple[int, int] | None:
    m = re.search(pattern, text, re.MULTILINE)
    return (int(m.group(1)), int(m.group(2))) if m else None


def pct(f: tuple[int, int] | None) -> float:
    return f[0] / f[1] if f and f[1] else 0.0


def show(f: tuple[int, int] | None) -> str:
    return f"{f[0]}/{f[1]} ({pct(f):.0%})" if f else "missing"


# 1. code health
unit = read("unit.log")
passed, failed = re.search(r"(\d+) passed", unit), re.search(r"(\d+) failed", unit)
check("Code", "everything compiles", "yes" if "all compile" in read("compile.log") else "no", "all compile" in read("compile.log"))
check("Code", "unit tests", f"{passed.group(1) if passed else 0} passed, {failed.group(1) if failed else 0} failed",
      bool(passed) and not failed)

# 3. accuracy without memory
for name, log, bar in (("handwritten set", "handwritten.log", 0.88), ("generated set", "generated.log", 0.85)):
    f = fraction(r"^Accuracy:\s+(\d+)/(\d+)", read(log))
    check("Accuracy (fresh, no memory)", f"{name} >= {bar:.0%}", show(f), pct(f) >= bar)
    fast = fraction(r"fast path:\s+(\d+)/(\d+)", read(log))
    check("Accuracy (fresh, no memory)", f"{name}: clear matches (no system one) >= 95% right", show(fast), pct(fast) >= 0.95)

# 4. learning over time
total = {k: fraction(r"^total (\d+)/(\d+)", read(f"curve_{k}.log")) for k in ("none", "corrections", "all")}
check("Learning", "accuracy with memory not below no memory (-1 point)",
      f"{show(total['all'])} vs {show(total['none'])} without memory", pct(total["all"]) >= pct(total["none"]) - 0.01)
mem = fraction(r"^memory decisions: (\d+)/(\d+)", read("curve_all.log"))
check("Learning", "decisions taken from memory >= 90% right", show(mem), pct(mem) >= 0.90)
last = {k: fraction(r"^last 100 emails: (\d+)/(\d+)", read(f"curve_{k}.log")) for k in ("none", "all")}
check("Learning", "last 100 emails: >= 40% need no system one (was ~25% without memory)",
      f"{show(last['all'])} (without memory {show(last['none'])})", pct(last["all"]) >= 0.40)
check("Learning", "with corrections only (no confirmations)", show(total["corrections"]), None)

# 5. one correction carries over
for label, log in (("Qwen3-30B rewordings", "para_qwen_memory-hints.log"),
                   ("Gemma + GaMS rewordings", "para_gemma_gams_memory-hints.log"),
                   ("Qwen3-30B, without hints", "para_qwen_no-memory-hints.log"),
                   ("Gemma + GaMS, without hints", "para_gemma_gams_no-memory-hints.log")):
    text = read(log)
    before = re.search(r"mistakes\s+(\d+)/(\d+).*control\s+(\d+)/(\d+)", text)
    after = re.search(r"^\s+0\.70\s+0\.5\s+(\d+)/(\d+).*?(\d+)/(\d+)", text, re.MULTILINE)
    if not (before and after):
        check("One correction", label, "missing", False)
        continue
    b_mis, b_ctl = (int(before[1]), int(before[2])), (int(before[3]), int(before[4]))
    a_mis, a_ctl = (int(after[1]), int(after[2])), (int(after[3]), int(after[4]))
    check("One correction", f"{label}: mistakes right after 1 fix >= 95%", f"{show(b_mis)} -> {show(a_mis)}", pct(a_mis) >= 0.95)
    check("One correction", f"{label}: control not hurt (-2 points)", f"{show(b_ctl)} -> {show(a_ctl)}",
          pct(a_ctl) >= pct(b_ctl) - 0.02)

# 6. system one down
down = read("down.log")
routed = len(re.findall(r"^Forward to:", down, re.MULTILINE))
fallback = len(re.findall(r"search only", down))
check("Robustness", "system one down: emails still routed (search only)", f"{fallback}/3 via search only, {routed}/3 routed",
      routed == 3 and fallback == 3)

# 7. web API
api = read("api.log")
check("API", "health check", "ok" if "'ok': True" in api else "failed", "'ok': True" in api)
fast_ms = [int(m) for m in re.findall(r"clear match\s+server\s+(\d+) ms", api)]
check("API", "clear match answered < 1 s (median)", f"{statistics.median(fast_ms):.0f} ms" if fast_ms else "missing",
      bool(fast_ms) and statistics.median(fast_ms) < 1000)
llm_ms = [int(m) for m in re.findall(r"system one\s+server\s+(\d+) ms", api)]
check("API", "system one really asked (no cache), answer time depends on hivecore load",
      f"median {statistics.median(llm_ms):.0f} ms over {len(llm_ms)} calls" if llm_ms else "none asked",
      bool(llm_ms) and statistics.median(llm_ms) > 100)
check("API", "unknown person rejected", "400" if "rejected: 400" in api else "not 400", "rejected: 400" in api)
learn = re.search(r"learning: (PASS|FAIL) \((.*)\)", api)
check("API", "fix -> a related, differently worded email follows it", learn.group(2) if learn else "missing",
      bool(learn) and learn.group(1) == "PASS")

# report
n_pass, n_fail = sum(r[3] == "PASS" for r in rows), sum(r[3] == "FAIL" for r in rows)
print(f"# Final test, {date.today():%d.%m.%Y}\n")
print(f"**{n_pass} passed, {n_fail} failed** (INFO rows are measured, not judged). Logs: `{DIR}/`.\n")
print("| Area | Check | Result | |\n|---|---|---|---|")
for area, name, result, verdict in rows:
    mark = {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "INFO": "ℹ️ INFO"}[verdict]
    print(f"| {area} | {name} | {result} | {mark} |")
