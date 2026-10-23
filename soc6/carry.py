"""Carry your A6 copilot into A7.

A6 E.4:  H.export_a6()            -> writes my_a6.py from the cells you ran (build_kg, kg_context,
                                     SYSTEM_PROMPT, answer, the memory code)
A7 0.2:  exec(open("my_a6.py").read())   -> rebuilds the KG and defines your functions again
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["export_a6", "A6_PIECES"]

# (marker that identifies the cell, what it provides) -- the latest executed cell with each marker wins
A6_PIECES = [
    ("def build_kg(", "build_kg + kg"),
    ("def kg_context(", "kg_context"),
    ("SYSTEM_PROMPT =", "SYSTEM_PROMPT, CONTEXT_BUDGET"),
    ("def answer(", "answer"),
    ("def chat_turn(", "memory: DB, POLICY, load_history, rewrite, chat_turn"),
]
HEADER = '''# my_a6.py -- the copilot you built in CS 789 A6, exported by soc6.export_a6().
# A7 runs it with exec(open("my_a6.py").read()) after starting the model server.
import os, re, json, time, typing
import pandas as pd
import soc6 as H
A, hosts, staff, windows = H.load_all()
techniques = pd.read_csv(H.HERE / "data" / "attack_techniques.csv")
'''


def _clean(src: str) -> str:
    keep = []
    for line in src.splitlines():
        s = line.strip()
        if s.startswith(("%", "!")) or s.startswith(("display(", "H.check_", "H.eval_bot", "H.save_kg")):
            continue
        keep.append(line)
    return "\n".join(keep).strip() + "\n"


def export_a6(path: str = "my_a6.py", sources: list[str] | None = None) -> str:
    """Write the latest version of each A6 piece you ran into one Python file. sources: list of cell
    texts (default: this notebook's execution history)."""
    if sources is None:
        from IPython import get_ipython
        ip = get_ipython()
        sources = list(ip.user_ns.get("In", [])) if ip else []
    chosen, missing = [], []
    for marker, what in A6_PIECES:
        cells = [c for c in sources if marker in c and "YOUR CODE HERE" not in c.split("\n")[0]]
        if cells:
            chosen.append((what, cells[-1]))
        else:
            missing.append(what)
    out = HEADER + "".join(f"\n# ---------------------------------------------------------------- {what}\n{_clean(src)}"
                           for what, src in chosen)
    Path(path).write_text(out)
    print(f"wrote {path} ({len(out.splitlines())} lines) with: {[w for w, _ in chosen]}")
    if missing:
        print("NOT FOUND (run those cells first, or A7 will use the reference version):", missing)
    try:
        from google.colab import drive  # noqa: F401
        d = Path("/content/drive/MyDrive")
        if d.exists():
            (d / "cs789").mkdir(exist_ok=True)
            Path(d / "cs789" / Path(path).name).write_text(out)
            print("also copied to Drive: MyDrive/cs789/" + Path(path).name)
    except Exception:
        pass
    return out
