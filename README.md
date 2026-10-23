# CS 789 — Assignment 7: SOC Copilot II — Automation, an Uncensored Model, the Finished Product

**UNLV · AI-Powered Cyber Attacks and Defenses · Fall 2026 · Prof. Yoohwan Kim**

A7 continues the SOC Copilot you built in A6 — same data, same knowledge graph, **your** retriever, system prompt
and memory (from the `my_a6.py` you exported in A6 E.4).

* **Part 0** — bring back your A6 copilot (or the reference A6 in `reference/a6_reference.py`)
* **Part A** — an **n8n** workflow triages the live alert feed through your copilot's API; you add the **LLM node**
  in the n8n editor so every case gets an AI-written summary
* **Part B** — an **uncensored copy** of the copilot's model made with **Heretic** (abliteration), measured against the
  original: refusals, KL divergence, and what happens to the copilot's guardrails; you write a guardrail canary
* **Part C** — your own **dashboards** and the **finished SOC Copilot**, with an end-to-end acceptance test

Parts A and B are independent. Do A first (it needs Qwen on the GPU; B takes the GPU for ~40 minutes).
The handout is `CS 789 A7.docx` / `.pdf` on WebCampus. This README is the "how do I start" sheet.

## Start on Google Colab (recommended)

1. Open **https://colab.research.google.com/github/unlv-cs789-f26/A7/blob/main/A7_SOC_Copilot_II.ipynb**
2. *Runtime → Change runtime type →* **L4 GPU** (an A100 makes Part B about twice as fast) → *Save*.
3. *File → Save a copy in Drive*.
4. Upload your **`my_a6.py`** into `/content/A7` after cell 0.1 has cloned the repo (or keep it in Drive at
   `MyDrive/cs789/my_a6.py`), then run 0.2.

No paid API is needed. Part A installs Node.js 24 and n8n in the VM (~4 min); Part B installs Heretic in its own
environment (~4 min) and runs it for ~20–40 minutes. The web app (port 8080) and the n8n editor (port 5678) open
through the https links the notebook prints.

## Start on your own computer (Linux or WSL2, NVIDIA GPU ≥ 16 GB)

```bash
git clone https://github.com/unlv-cs789-f26/A7.git
cd A7
cp /path/to/my_a6.py .
./start.sh --vllm --n8n --heretic   # .venv, packages, vLLM, n8n, Heretic; checks the machine; opens Jupyter Lab
./start.sh --check                  # only the environment check
```

## Files

| file | what it is |
|---|---|
| `A7_SOC_Copilot_II.ipynb` | the assignment notebook — 🛠️ build cells (with AI prompts), ✅ checks, ✍️ questions |
| `soc6/` | the fixed harness (same as A6, plus n8n and Heretic helpers). **Do not edit.** |
| `reference/a6_reference.py` | the instructor's A6 copilot, for students whose A6 was incomplete |
| `n8n/soc_triage.json` | the triage workflow imported into n8n (Part A) |
| `data/`, `ontology/`, `questions.yaml` | the A6 data, the completed ontology, the 22 evaluation questions |
| `install_vllm.sh` | vLLM installer used by the notebook |
| `PROMPTS.md` | how to use your browser AI assistant, and prompt patterns |
| `start.sh` / `start.bat` / `check_environment.py` | local start-up and checks |

## Responsible use (Part B)

Part B is a measurement exercise on a model that stays in your Colab VM. Do not upload the abliterated model, do not
use Heretic's chat option to collect harmful content, and report numbers and the copilot's answers to the SOC
questions — not the model's answers to refusal-triggering prompts.

## Credit

Alerts: **AIT Alert Data Set** — M. Landauer, F. Skopik, M. Wurzenberger, Zenodo 2023,
doi:[10.5281/zenodo.8263181](https://doi.org/10.5281/zenodo.8263181), CC BY 4.0 (see `data/DATASET.md`).
Heretic: P. E. Weidmann, [github.com/p-e-w/heretic](https://github.com/p-e-w/heretic), AGPL-3.0 (installed from PyPI as
`heretic-llm`, not redistributed here). n8n: Sustainable Use License.

## Submission

One PDF on WebCampus: the filled-in handout followed by this notebook printed with its outputs
(*File → Print → Save as PDF*).
