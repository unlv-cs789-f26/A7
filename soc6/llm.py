"""LLM servers and clients. Three model families, all behind the same OpenAI-compatible API:

    "qwen"     Qwen3-4B-Instruct-2507        vLLM on localhost:8000   (the A4/A5 model)
    "granite"  IBM granite-4.0-h-micro (3B)  vLLM on localhost:8002   (hybrid Mamba-2/transformer)
    "gemini"   Google Gemini flash           Gemini API, OpenAI-compatible endpoint (key in Colab Secrets)

chat(model, messages) returns {"text", "prompt_tokens", "completion_tokens", "seconds"}.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

from .data import HERE

__all__ = ["MODELS", "install_vllm", "start_model", "stop_model", "server_up", "chat", "count_tokens",
           "llamaindex_llm", "langchain_chat", "gpu", "model_available", "gemini_key", "ensure_model", "run_async"]

_ON_COLAB = Path("/content").is_dir()
VLLM_ENV = os.environ.get("CS789_VLLM_ENV") or ("/content/vllm-env" if _ON_COLAB else str(HERE / "vllm-env"))
LOG_DIR = Path("/content") if _ON_COLAB else HERE

MODELS = {
    "qwen":    {"hf": "Qwen/Qwen3-4B-Instruct-2507", "port": 8000, "served": "qwen",
                "base_url": "http://localhost:8000/v1", "gpu_mem": 0.80, "max_len": 16384},
    "granite": {"hf": "ibm-granite/granite-4.0-h-micro", "port": 8002, "served": "granite",
                "base_url": "http://localhost:8002/v1", "gpu_mem": 0.80, "max_len": 16384},
    "gemini":  {"hf": None, "served": os.environ.get("GEMINI_MODEL", "gemini-flash-latest"),
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"},
}
# stub / custom endpoints can be registered at run time, e.g. MODELS["stub"] = {...}


def install_vllm(env: str = VLLM_ENV):
    """Install vLLM into its OWN virtualenv so it cannot break the notebook's torch (same as A4/A5)."""
    if Path(env, "bin", "vllm").exists():
        print("vLLM already installed in", env)
        return
    t = time.time()
    subprocess.run(f"CS789_VLLM_ENV={env} bash {HERE / 'install_vllm.sh'}", shell=True, check=True)
    print(f"vLLM installed in {time.time() - t:.0f}s")


def server_up(port: int) -> bool:
    import requests
    try:
        return requests.get(f"http://localhost:{port}/v1/models", timeout=2).ok
    except Exception:
        return False


def stop_model(name: str):
    port = MODELS[name]["port"]
    subprocess.run(f"pkill -f '[v]llm serve.*--port {port}( |$)' || true", shell=True)
    time.sleep(3)


def start_model(name: str = "qwen", gpu_mem: float | None = None, max_len: int | None = None,
                timeout: int = 1500, extra: str = ""):
    """Start (or reuse) the vLLM server for one local model."""
    import requests
    m = MODELS[name]
    if server_up(m["port"]):
        print(f"{name} already serving on port {m['port']}")
        return
    log = LOG_DIR / f"vllm_{name}.log"
    cmd = (f"{VLLM_ENV}/bin/vllm serve {m['hf']} --served-model-name {m['served']} --port {m['port']} "
           f"--max-model-len {max_len or m['max_len']} --gpu-memory-utilization {gpu_mem or m['gpu_mem']} "
           f"--enable-prefix-caching {extra}")
    env = dict(os.environ, PATH=f"{VLLM_ENV}/bin:/usr/local/cuda/bin:" + os.environ.get("PATH", ""),
               VIRTUAL_ENV=VLLM_ENV)
    proc = subprocess.Popen(cmd, shell=True, stdout=open(log, "w"), stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid, env=env)
    t = time.time()
    while time.time() - t < timeout:
        if proc.poll() is not None:
            print(log.read_text()[-3000:])
            raise RuntimeError(f"vLLM for {name} exited -- see the log above")
        if server_up(m["port"]):
            print(f"{name} ({m['hf']}) up on port {m['port']} in {time.time() - t:.0f}s")
            return
        time.sleep(5)
    raise TimeoutError(f"{name} did not come up; see {log}")


def ensure_model(name: str):
    """Make sure a model answers. Local models share the one GPU, so starting one stops the other
    (about 1-2 minutes per switch). Gemini needs only the key."""
    m = MODELS[name]
    if "port" not in m or m.get("external"):
        if name == "gemini" and not gemini_key():
            raise RuntimeError("no GEMINI_API_KEY in Colab Secrets -- skip gemini or add the key")
        return
    if server_up(m["port"]):
        return
    for other, om in MODELS.items():
        if other != name and "port" in om and not om.get("external") and om.get("hf") and server_up(om["port"]):
            print(f"stopping {other} to free the GPU for {name} ...")
            stop_model(other)
    start_model(name)


def gemini_key():
    k = os.environ.get("GEMINI_API_KEY")
    if not k and _ON_COLAB:
        try:
            from google.colab import userdata
            k = userdata.get("GEMINI_API_KEY")
        except Exception:
            k = None
    return k


def model_available(name: str) -> bool:
    m = MODELS.get(name)
    if not m:
        return False
    if name == "gemini":
        return bool(gemini_key())
    return server_up(m["port"]) if "port" in m else True


def _endpoint(name):
    m = MODELS[name]
    key = gemini_key() if name == "gemini" else m.get("api_key", "EMPTY")
    return m["base_url"].rstrip("/"), m["served"], key


def chat(model: str, messages: list, max_tokens: int = 500, temperature: float = 0.0, retries: int = 3) -> dict:
    """One chat completion. messages = [{"role": "system"|"user"|"assistant", "content": str}, ...]"""
    import requests
    base, served, key = _endpoint(model)
    t = time.time()
    for attempt in range(retries):
        r = requests.post(f"{base}/chat/completions", timeout=600,
                          headers={"Authorization": f"Bearer {key}"},
                          json={"model": served, "messages": messages, "max_tokens": max_tokens,
                                "temperature": temperature})
        if r.status_code == 429 and attempt < retries - 1:        # Gemini free tier: rate limit
            time.sleep(15 * (attempt + 1))
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"{model}: HTTP {r.status_code}: {r.text[:500]}")
        break
    j = r.json()
    text = j["choices"][0]["message"].get("content") or ""
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S)
    u = j.get("usage") or {}
    return {"text": text.strip(), "prompt_tokens": u.get("prompt_tokens", 0),
            "completion_tokens": u.get("completion_tokens", 0), "seconds": round(time.time() - t, 2)}


_TOK = {}


def count_tokens(text, model: str = "qwen") -> int:
    """Exact token count with the model's tokenizer when available (Qwen/Granite), else ~4 chars/token."""
    if not isinstance(text, str):
        text = "\n".join(m["content"] if isinstance(m, dict) else str(getattr(m, "content", m)) for m in text)
    hf = MODELS.get(model, {}).get("hf")
    if hf and hf not in _TOK:
        try:
            from transformers import AutoTokenizer
            _TOK[hf] = AutoTokenizer.from_pretrained(hf)
        except Exception:
            _TOK[hf] = None
    tok = _TOK.get(hf)
    return len(tok.encode(text)) if tok else max(1, len(text) // 4)


def llamaindex_llm(model: str = "qwen", **kw):
    """A LlamaIndex LLM object pointing at one of the endpoints (for SchemaLLMPathExtractor)."""
    from llama_index.llms.openai_like import OpenAILike
    base, served, key = _endpoint(model)
    return OpenAILike(model=served, api_base=base, api_key=key, is_chat_model=True,
                      is_function_calling_model=False, temperature=0.0, max_tokens=kw.pop("max_tokens", 1024),
                      timeout=600, **kw)


def langchain_chat(model: str = "qwen", **kw):
    """A LangChain ChatOpenAI pointing at one of the endpoints (for Part D)."""
    from langchain_openai import ChatOpenAI
    base, served, key = _endpoint(model)
    return ChatOpenAI(base_url=base, api_key=key, model=served, temperature=kw.pop("temperature", 0),
                      max_tokens=kw.pop("max_tokens", 500), **kw)


def gpu():
    out = subprocess.run("nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader",
                         shell=True, capture_output=True, text=True)
    print(out.stdout.strip() or "no NVIDIA GPU visible")


def run_async(coro):
    """Run a coroutine to completion from a notebook cell (Colab already runs an event loop, so
    asyncio.run() fails there). Used for LlamaIndex's async extractors: H.run_async(extractor.acall(nodes))."""
    import asyncio
    import threading
    box = {}

    def _runner():
        try:
            box["result"] = asyncio.run(coro)
        except BaseException as e:  # pragma: no cover
            box["error"] = e
    t = threading.Thread(target=_runner)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box.get("result")
