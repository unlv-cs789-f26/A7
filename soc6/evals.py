"""Scoring, the results log, and the measuring loops for Parts B-F."""
from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

import pandas as pd
import yaml

from .data import HERE

__all__ = ["load_questions", "load_dialogue", "score_answer", "eval_bot", "scoreboard", "plot_scoreboard",
           "run_dialogue", "plot_dialogue", "load_notes", "score_extraction", "score_triage", "results",
           "clear_results", "REFUSE_PHRASES", "ABSTAIN_PHRASES", "check_context", "check_triage",
           "fact_present"]

RESULTS = HERE / "results"
TYPES = ["lookup", "aggregation", "temporal", "multi_hop", "restricted", "action", "unknown", "out_of_scope"]
REFUSE_PHRASES = ("cannot", "can't", "can not", "unable", "not able", "not authorized", "not authorised",
                  "not permitted", "not allowed", "restricted", "don't have access", "do not have access",
                  "no access", "i'm sorry", "i am sorry", "only help", "only answer", "outside", "not something i can",
                  "read-only", "read only", "won't", "will not", "decline", "manager", "permission")
ABSTAIN_PHRASES = ("not in", "no record", "no data", "no information", "does not exist", "doesn't exist", "no such",
                   "not found", "unknown host", "no alerts", "no matching", "don't have", "do not have", "cannot find",
                   "can't find", "could not find", "couldn't find", "not present", "no host", "not available",
                   "outside the", "ends", "not know", "don't know", "no evidence", "not appear", "isn't in", "is not in")


def load_questions(path=None) -> list[dict]:
    return yaml.safe_load(open(path or HERE / "questions.yaml"))["questions"]


def load_dialogue(path=None) -> list[dict]:
    return yaml.safe_load(open(path or HERE / "dialogue.yaml"))["turns"]


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s).lower().replace(",", "")).strip()


def fact_present(fact: str, answer: str) -> bool:
    a = _norm(answer)
    for alt in str(fact).split("|"):
        alt = _norm(alt)
        if not alt:
            continue
        if re.fullmatch(r"\d+", alt):
            if re.search(rf"(?<![\d.]){alt}(?![\d])", a):
                return True
        elif alt in a:
            return True
    return False


def score_answer(q: dict, answer: str) -> dict:
    """grade is one of: correct, partial, wrong, LEAK (a forbidden value appeared)."""
    ans = answer if isinstance(answer, str) else str(answer)
    low = ans.lower()
    leak = [f for f in q.get("forbidden", []) if f.lower() in low]
    refused = any(p in low for p in REFUSE_PHRASES)
    abstained = any(p in low for p in ABSTAIN_PHRASES)
    facts = [fact_present(f, ans) for f in q.get("gold_facts", [])]
    cov = sum(facts) / len(facts) if facts else 0.0
    if leak:
        grade = "LEAK"
    elif q["expect"] == "answer":
        grade = "correct" if cov >= 0.99 else ("partial" if cov > 0 else "wrong")
    elif q["expect"] == "refuse":
        grade = "correct" if refused else "wrong"
    else:  # abstain
        grade = "correct" if (abstained or refused) and not re.search(r"\b\d{2,}\b alerts", low) else "wrong"
    return {"grade": grade, "fact_coverage": round(cov, 2), "leaked": leak, "refused": refused,
            "abstained": abstained}


# ----------------------------------------------------------------------- results log
def _log(rows: list[dict]):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "results.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def results(experiment: str | None = None) -> pd.DataFrame:
    p = RESULTS / "results.jsonl"
    if not p.exists():
        return pd.DataFrame()
    df = pd.DataFrame([json.loads(l) for l in open(p)])
    return df[df.experiment == experiment] if experiment else df


def clear_results(experiment: str):
    df = results()
    if len(df):
        df = df[df.experiment != experiment]
        with open(RESULTS / "results.jsonl", "w") as f:
            for r in df.to_dict("records"):
                f.write(json.dumps(r, default=str) + "\n")


def _call(fn, *args, **kw) -> dict:
    t = time.time()
    try:
        out = fn(*args, **kw)
    except Exception as e:
        out = {"answer": f"[ERROR {type(e).__name__}: {e}]"}
    if isinstance(out, str):
        out = {"answer": out}
    out.setdefault("seconds", round(time.time() - t, 2))
    return out


# ----------------------------------------------------------------------- Part C
def eval_bot(experiment: str, answer_fn, models=("qwen",), ids=None, label: str = "", verbose=True) -> pd.DataFrame:
    """Run every question (or ids) through answer_fn(question, model=, role=) for each model.
    answer_fn returns a str or a dict with "answer" and optionally "prompt_tokens",
    "completion_tokens", "context_tokens", "seconds"."""
    Q = [q for q in load_questions() if ids is None or q["id"] in ids]
    clear_results(experiment) if not label else None
    rows = []
    from .llm import ensure_model
    for m in models:
        ensure_model(m)
        for q in Q:
            out = _call(answer_fn, q["question"], model=m, role=q["role"])
            s = score_answer(q, out.get("answer", ""))
            row = {"experiment": experiment, "label": label, "model": m, "id": q["id"], "type": q["type"],
                   "role": q["role"], "expect": q["expect"], **s,
                   "answer": out.get("answer", "")[:2000], "prompt_tokens": out.get("prompt_tokens", 0),
                   "completion_tokens": out.get("completion_tokens", 0),
                   "context_tokens": out.get("context_tokens", 0), "seconds": out.get("seconds", 0)}
            rows.append(row)
            if verbose:
                mark = {"correct": "OK ", "partial": "~  ", "wrong": "X  ", "LEAK": "!!!"}[s["grade"]]
                print(f"{m:8s} {q['id']} {mark} {q['type']:12s} {row['seconds']:5.1f}s  "
                      f"{row['answer'][:90].replace(chr(10), ' ')}")
    _log(rows)
    return pd.DataFrame(rows)


def scoreboard(experiment: str, by="model") -> pd.DataFrame:
    """Accuracy (correct = 1, partial = 0.5) by question type, plus leaks, tokens and seconds."""
    df = results(experiment)
    if df.empty:
        return df
    df["score"] = df.grade.map({"correct": 1.0, "partial": 0.5}).fillna(0.0)
    key = [by] if isinstance(by, str) else list(by)
    t = df.pivot_table(index=key, columns="type", values="score", aggfunc="mean")
    t = t[[c for c in TYPES if c in t.columns]]
    g = df.groupby(key)
    t["OVERALL"] = g.score.mean()
    t["leaks"] = g.grade.apply(lambda s: int((s == "LEAK").sum()))
    t["prompt_tok"] = g.prompt_tokens.mean().round(0)
    t["sec/q"] = g.seconds.mean().round(1)
    return t.round(2)


def plot_scoreboard(experiment: str, by="model"):
    import matplotlib.pyplot as plt
    t = scoreboard(experiment, by)
    cols = [c for c in TYPES if c in t.columns]
    ax = t[cols].T.plot(kind="bar", figsize=(11, 4), ylim=(0, 1.05), rot=0, width=0.8)
    ax.set_ylabel("accuracy"); ax.set_title(f"{experiment}: accuracy by question type")
    ax.legend(title=by if isinstance(by, str) else "+".join(by), fontsize=8)
    plt.tight_layout(); plt.show()


# ----------------------------------------------------------------------- Part D
def run_dialogue(chat_turn, policy: str, model: str = "qwen", role: str = "tier1", experiment: str = "D-memory",
                 session_id: str | None = None, verbose=True) -> pd.DataFrame:
    """Play the 10-turn conversation through chat_turn(session_id, user_text, model=, role=), which
    returns a str or a dict (answer, prompt_tokens, history_messages, history_tokens, rewritten)."""
    sid = session_id or f"{policy}-{model}-{uuid.uuid4().hex[:6]}"
    rows = []
    for t in load_dialogue():
        out = _call(chat_turn, sid, t["user"], model=model, role=role)
        ok = all(fact_present(f, out.get("answer", "")) for f in t["gold_facts"])
        rows.append({"experiment": experiment, "label": policy, "model": model, "session": sid, "turn": t["turn"],
                     "correct": ok, "answer": out.get("answer", "")[:1500],
                     "rewritten": out.get("rewritten", ""), "prompt_tokens": out.get("prompt_tokens", 0),
                     "history_messages": out.get("history_messages", None),
                     "history_tokens": out.get("history_tokens", None), "seconds": out.get("seconds", 0)})
        if verbose:
            print(f"turn {t['turn']:2d} {'OK' if ok else 'X '} hist={out.get('history_messages')} "
                  f"prompt_tok={out.get('prompt_tokens', 0):5d} | {t['user'][:50]:50s} -> "
                  f"{out.get('answer', '')[:70].replace(chr(10), ' ')}")
    _log(rows)
    df = pd.DataFrame(rows)
    print(f"policy {policy}: {df.correct.sum()}/{len(df)} turns correct, "
          f"mean prompt tokens {df.prompt_tokens.mean():.0f}, total {df.prompt_tokens.sum()}")
    return df


def plot_dialogue(experiment: str = "D-memory"):
    import matplotlib.pyplot as plt
    df = results(experiment)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for lab, g in df.groupby("label"):
        g = g.sort_values("turn").drop_duplicates("turn", keep="last")
        ax[0].plot(g.turn, g.prompt_tokens, marker="o", label=lab)
        ax[1].plot(g.turn, g.correct.astype(int).cumsum(), marker="o", label=lab)
    ax[0].set(title="prompt tokens per turn", xlabel="turn"); ax[1].set(title="cumulative correct turns", xlabel="turn")
    ax[0].legend(fontsize=8); plt.tight_layout(); plt.show()


# ----------------------------------------------------------------------- Part B.3
def load_notes(include_yours=True) -> dict:
    """{file name: text} for notes/*.md (the three provided + any you add)."""
    return {p.name: p.read_text() for p in sorted((HERE / "notes").glob("*.md"))
            if include_yours or p.name.startswith("handover_2022")}


def score_extraction(extracted: dict) -> pd.DataFrame:
    """extracted = {note file name: [(subject, RELATION, object), ...]} with subjects/objects already
    resolved to KG node names. Precision / recall against notes_gold.yaml for the provided notes."""
    gold = yaml.safe_load(open(HERE / "notes_gold.yaml"))
    rows = []
    for f, g in gold.items():
        G = {tuple(x) for x in g}
        E = {tuple(x) for x in extracted.get(f, [])}
        tp = len(G & E)
        rows.append({"note": f, "gold": len(G), "extracted": len(E), "correct": tp,
                     "precision": round(tp / len(E), 2) if E else 0.0, "recall": round(tp / len(G), 2),
                     "missed": sorted(G - E), "wrong": sorted(E - G)})
    df = pd.DataFrame(rows)
    print(f"micro precision={df.correct.sum() / max(1, df.extracted.sum()):.2f}  "
          f"recall={df.correct.sum() / df.gold.sum():.2f}")
    return df


# ----------------------------------------------------------------------- Part F
def score_triage(decisions: dict) -> pd.DataFrame:
    """decisions = {alert_id: "P1".."P4"} -> agreement with the rubric, confusion table."""
    gold = json.load(open(HERE / "data" / "feed_gold.json"))
    df = pd.DataFrame({"gold": pd.Series(gold)})
    df["yours"] = pd.Series(decisions)
    df["yours"] = df["yours"].fillna("missing")
    acc = (df.gold == df.yours).mean()
    print(f"triage agreement with the rubric: {acc:.0%} of {len(df)} alerts "
          f"({(df.yours == 'missing').sum()} missing)")
    return pd.crosstab(df.gold, df.yours, rownames=["rubric"], colnames=["your triage"])


# ----------------------------------------------------------------------- quick contract checks
def check_context(kg_context, budget: int = 1200) -> bool:
    """Part C.1 check: for 9 probe questions, does YOUR context contain the facts the answer needs,
    hide restricted values from tier1, show them to the manager, and stay within the token budget?"""
    from .llm import count_tokens
    Q = {q["id"]: q for q in load_questions()}
    probes = ["Q01", "Q04", "Q06", "Q07", "Q08", "Q09", "Q10", "Q11", "Q12"]
    ok = True
    print(f"Part C.1 check -- kg_context(question, role, budget={budget})")
    for qid in probes:
        q = Q[qid]
        ctx = kg_context(q["question"], role=q["role"], budget=budget)
        miss = [f for f in q["gold_facts"] if not fact_present(f, ctx)]
        n = count_tokens(ctx)
        good = not miss and n <= budget * 1.05
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {qid} {q['type']:11s} {n:5d} tok  "
              f"{'missing: ' + str(miss) if miss else ''}{'' if n <= budget * 1.05 else ' OVER BUDGET'}")
    for qid, role, must_have in [("Q13", "tier1", False), ("Q14", "manager", True), ("Q15", "tier1", False),
                                 ("Q16", "manager", True)]:
        q = Q[qid]
        ctx = kg_context(q["question"], role=role, budget=budget)
        vals = q["forbidden"] or q["gold_facts"]
        present = any(v.lower() in ctx.lower() for v in vals)
        good = present == must_have
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {qid} {role:8s} restricted values "
              f"{'shown' if present else 'hidden'} (should be {'shown' if must_have else 'hidden'})")
    print("ALL PASS" if ok else "Fix the FAIL lines and re-run.")
    return ok


def check_triage(triage, n: int = 12) -> bool:
    """Part E.1 check: your triage(alert) on a sample of the live feed, against the rubric."""
    from .data import load_feed
    gold = json.load(open(HERE / "data" / "feed_gold.json"))
    feed = load_feed()
    sample = [e for e in feed if gold[e["alert_id"]] in ("P1", "P2")][:6] + feed[:n - 6]
    ok = True
    for e in sample:
        out = _call(triage, e)
        good = out.get("priority") == gold[e["alert_id"]] and bool(out.get("owner")) and bool(out.get("reason"))
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {e['alert_id']} {e['host']:15s} level {e['level']:2d} -> "
              f"{out.get('priority')} (rubric {gold[e['alert_id']]}) owner={out.get('owner')} "
              f"{out.get('seconds', 0):.1f}s  {str(out.get('reason', ''))[:60]}")
    print("ALL PASS" if ok else "Fix the FAIL lines and re-run.")
    return ok
