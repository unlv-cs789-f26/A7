#!/usr/bin/env bash
# Installs vLLM (the local LLM server) into its OWN virtualenv, so it can never break the notebook's
# PyTorch. Default location: /content/vllm-env on Colab, ./vllm-env elsewhere. Same as A4/A5.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -n "$CS789_VLLM_ENV" ]; then ENV="$CS789_VLLM_ENV"
elif [ -d /content ]; then ENV=/content/vllm-env
else ENV="$HERE/vllm-env"; fi
command -v uv >/dev/null 2>&1 || python3 -m pip -q install uv
[ -x "$ENV/bin/python" ] || uv venv -q "$ENV" --python 3.12
# --python pins the target env (Colab sets UV_SYSTEM_PYTHON=true, which would otherwise win)
uv pip install -q --python "$ENV/bin/python" "vllm>=0.20"
"$ENV/bin/python" -c "import vllm; print('vLLM', vllm.__version__, 'installed in $ENV')"
