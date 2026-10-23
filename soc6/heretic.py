"""A7 Part B: an uncensored (abliterated) copy of the copilot's model, made with Heretic.

Heretic (p-e-w/heretic, AGPL-3.0) finds the "refusal direction" in a model's residual stream from the
difference of mean activations on prompts that usually get refused vs. prompts that do not, and removes it
from the weights (directional ablation = "abliteration"). An Optuna search tunes where and how strongly to
ablate, minimizing two numbers at once: refusals on a public set of refusal-triggering prompts
(mlabonne/harmful_behaviors) and KL divergence from the original model on harmless prompts
(mlabonne/harmless_alpaca) -- the "damage" done to the model.

    H.install_heretic()                    # its own virtualenv (Heretic pins transformers 5.x)
    H.free_gpu()                           # stop the vLLM servers: Heretic needs the GPU
    print(H.heretic_command())             # the command you run (A7 B.1)
    H.heretic_evaluate([...])              # refusals and KL of saved models vs. the original
    H.register_heretic_models()            # serve them with vLLM next to Qwen, for the copilot
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

import pandas as pd

from .data import HERE

__all__ = ["HERETIC_VERSION", "BASE_MODEL", "PUBLISHED_HERETIC", "MY_HERETIC_DIR", "install_heretic", "free_gpu",
           "heretic_command", "heretic_evaluate", "register_heretic_models", "plot_tradeoff", "compare_answers"]

HERETIC_VERSION = "1.4.0"
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
PUBLISHED_HERETIC = "p-e-w/Qwen3-4B-Instruct-2507-heretic"      # made by Heretic's author with the default 200 trials
_ON_COLAB = Path("/content").is_dir()
BASE = Path("/content") if _ON_COLAB else HERE
HERETIC_ENV = Path(os.environ.get("CS789_HERETIC_ENV", BASE / "heretic-env"))
WORK = BASE / "heretic_work"
MY_HERETIC_DIR = BASE / "qwen-heretic-mine"


def install_heretic(version: str = HERETIC_VERSION):
    """Install Heretic into its OWN virtualenv (it needs transformers 5.x and its own torch)."""
    if (HERETIC_ENV / "bin" / "heretic").exists():
        print("Heretic already installed in", HERETIC_ENV)
        return
    t = time.time()
    cmd = (f"set -e; command -v uv >/dev/null 2>&1 || python3 -m pip -q install uv; "
           f"[ -x {HERETIC_ENV}/bin/python ] || uv venv -q {HERETIC_ENV} --python 3.12; "
           f"uv pip install -q --python {HERETIC_ENV}/bin/python heretic-llm=={version}")
    subprocess.run(cmd, shell=True, check=True)
    print(f"Heretic {version} installed in {HERETIC_ENV} ({time.time() - t:.0f}s)")


def free_gpu():
    """Stop every local vLLM server so Heretic (or another model) gets the whole GPU."""
    from .llm import MODELS, server_up, stop_model
    for name, m in MODELS.items():
        if "port" in m and m.get("hf") and server_up(m["port"]):
            print("stopping", name)
            stop_model(name)
    subprocess.run("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader", shell=True)


def heretic_command(n_trials: int = 60, n_startup_trials: int = 20, model: str = BASE_MODEL) -> str:
    """The command for B.1. Heretic is interactive at the end (choose a trial, then save it)."""
    WORK.mkdir(parents=True, exist_ok=True)
    return (f"cd {WORK} && {HERETIC_ENV}/bin/heretic --model {model} "
            f"--n-trials {n_trials} --n-startup-trials {n_startup_trials}")


_PAT = {"base_refusals": r"Initial refusals:\s*(\d+)\s*/\s*(\d+)",
        "kl": r"KL divergence:\s*([0-9.]+)",
        "refusals": r"\*\s*Refusals:\s*(\d+)\s*/\s*(\d+)"}


def heretic_evaluate(models: list, base: str = BASE_MODEL, timeout: int = 3600) -> pd.DataFrame:
    """Run Heretic's evaluator (`heretic --model base --evaluate-model M`) for each model: refusals out of
    100 refusal-triggering prompts and KL divergence from the base model on 100 harmless prompts.
    Prints only these numbers -- the evaluator never shows the prompts or the model's answers."""
    free_gpu()
    rows = []
    log_dir = WORK / "eval_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for m in models:
        t = time.time()
        log = log_dir / (re.sub(r"[^A-Za-z0-9_.-]", "_", str(m)) + ".log")
        cmd = f"cd {WORK} && {HERETIC_ENV}/bin/heretic --model {base} --evaluate-model {m}"
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout,
                           env=dict(os.environ, TERM="dumb", NO_COLOR="1"))
        out = r.stdout + r.stderr
        log.write_text(out)
        row = {"model": str(m), "seconds": round(time.time() - t)}
        b = re.search(_PAT["base_refusals"], out)
        k = re.findall(_PAT["kl"], out)
        f = re.findall(_PAT["refusals"], out)
        row["base_refusals"] = f"{b.group(1)}/{b.group(2)}" if b else None
        row["refusals"] = f"{f[-1][0]}/{f[-1][1]}" if f else None
        row["kl_divergence"] = float(k[-1]) if k else None
        if r.returncode != 0 or not f:
            row["error"] = out.strip().splitlines()[-1][:200] if out.strip() else f"exit {r.returncode}"
        print(row)
        rows.append(row)
    return pd.DataFrame(rows)


def register_heretic_models(mine: str | Path = MY_HERETIC_DIR, include_published: bool = True):
    """Add the abliterated models to the model registry so H.chat / H.eval_bot / the web app can use them.
    Like Qwen and Granite they share the GPU: H.ensure_model() swaps them in and out."""
    from .llm import MODELS
    if Path(mine).exists() or not _ON_COLAB:
        MODELS["heretic-mine"] = {"hf": str(mine), "port": 8003, "served": "heretic-mine",
                                  "base_url": "http://localhost:8003/v1", "gpu_mem": 0.80, "max_len": 16384}
    else:
        print(f"{mine} not found -- save your Heretic model there first (B.1)")
    if include_published:
        MODELS["heretic-pew"] = {"hf": PUBLISHED_HERETIC, "port": 8004, "served": "heretic-pew",
                                 "base_url": "http://localhost:8004/v1", "gpu_mem": 0.80, "max_len": 16384}
    return [m for m in ("heretic-mine", "heretic-pew") if m in MODELS]


def plot_tradeoff(trials, chosen: int | None = None, extra: dict | None = None):
    """trials = [(trial number, refusals, kl), ...] copied from Heretic's Pareto list.
    extra = {"label": (refusals, kl)} for other models, e.g. the published one."""
    import matplotlib.pyplot as plt
    df = pd.DataFrame(trials, columns=["trial", "refusals", "kl"])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(df.kl, df.refusals, "o-", label="your Pareto front")
    for r in df.itertuples():
        ax.annotate(str(r.trial), (r.kl, r.refusals), textcoords="offset points", xytext=(4, 4), fontsize=8)
    if chosen is not None and chosen in set(df.trial):
        c = df[df.trial == chosen].iloc[0]
        ax.plot([c.kl], [c.refusals], "o", ms=14, mfc="none", mec="red", label=f"chosen: trial {chosen}")
    for lab, (ref, kl) in (extra or {}).items():
        ax.plot([kl], [ref], "s", ms=9, label=lab)
    ax.set(xlabel="KL divergence from the original (damage) ->", ylabel="refusals out of 100 ->",
           title="Heretic: refusals vs. damage")
    ax.legend(fontsize=8); plt.tight_layout(); plt.show()


def compare_answers(experiment: str, types=("action", "out_of_scope", "restricted", "unknown"), width: int = 160):
    """Side-by-side answers of every model in an eval_bot experiment, for the question types where
    guardrails matter."""
    from .evals import results
    df = results(experiment)
    df = df[df.type.isin(types)]
    t = df.pivot_table(index=["id", "type", "role"], columns="model", values="answer", aggfunc="first")
    g = df.pivot_table(index=["id", "type", "role"], columns="model", values="grade", aggfunc="first")
    t = t.apply(lambda col: col.str.slice(0, width).str.replace("\n", " "))
    return t.join(g, rsuffix=" grade")
