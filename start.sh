#!/usr/bin/env bash
# CS 789 A7 -- start script for your OWN computer (Linux, or Windows via WSL2; NVIDIA GPU >= 16 GB).
# On Google Colab you do not need this: open A7_SOC_Copilot_II.ipynb from GitHub in Colab instead.
#
#   ./start.sh            create .venv, install packages, check the machine, open Jupyter Lab
#   ./start.sh --vllm     also install vLLM (every part needs a local model server)
#   ./start.sh --n8n      also install Node.js 24 + n8n (Part A)
#   ./start.sh --heretic  also install Heretic in its own environment (Part B)
#   ./start.sh --check    only run the environment check
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ "$1" = "--check" ]; then $PY check_environment.py; exit 0; fi
[ -d .venv ] || { echo "== creating .venv"; $PY -m venv .venv; }
. .venv/bin/activate
python -m pip -q install --upgrade pip
echo "== installing packages (first time: a few minutes)"
python -m pip -q install -r requirements.txt jupyterlab ipykernel matplotlib pandas
for a in "$@"; do
  [ "$a" = "--vllm" ] && bash install_vllm.sh
  [ "$a" = "--n8n" ] && python -c "import soc6; soc6.install_n8n()"
  [ "$a" = "--heretic" ] && python -c "import soc6; soc6.install_heretic()"
done
python check_environment.py || true
echo "== opening Jupyter Lab: open A7_SOC_Copilot_II.ipynb"
exec jupyter lab A7_SOC_Copilot_II.ipynb
