"""n8n inside the Colab VM: install, start, import the triage workflow, replay the live feed.

    H.install_n8n()        # Node.js 24 + n8n (about 3-4 minutes the first time)
    H.start_n8n()          # n8n on port 5678, imports and publishes n8n/soc_triage.json
    H.public_url(5678)     # open the n8n editor in your browser
    H.replay_feed()        # POST the 178 live-feed alerts to the workflow's webhook
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from .data import HERE, load_feed

__all__ = ["N8N_VERSION", "install_n8n", "start_n8n", "stop_n8n", "n8n_up", "webhook_live", "replay_feed", "n8n_cli",
           "WEBHOOK_URL"]

N8N_VERSION = "2.43.5"
_ON_COLAB = Path("/content").is_dir()
BASE = Path("/content") if _ON_COLAB else HERE
NODE_DIR = BASE / "node24"
N8N_HOME = BASE / "n8n_data"
N8N_PORT = 5678
WEBHOOK_URL = f"http://localhost:{N8N_PORT}/webhook/soc-alert"
WORKFLOW = HERE / "n8n" / "soc_triage.json"
WORKFLOW_ID = "soc6triage000001"


def _node_bin():
    """A Node.js >= 24 bin directory (n8n 2.x needs it), or None."""
    for d in [NODE_DIR / "bin", Path(os.environ.get("CS789_NODE_BIN", "/nonexistent"))]:
        if (d / "node").exists():
            return d
    node = shutil.which("node")
    if node:
        v = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip()
        if v.startswith("v") and int(v[1:].split(".")[0]) >= 24:
            return Path(node).parent
    return None


def _env():
    nb = _node_bin()
    env = dict(os.environ, N8N_USER_FOLDER=str(N8N_HOME), N8N_PORT=str(N8N_PORT), N8N_SECURE_COOKIE="false",
               N8N_DIAGNOSTICS_ENABLED="false", N8N_PERSONALIZATION_ENABLED="false",
               N8N_VERSION_NOTIFICATIONS_ENABLED="false", N8N_HIDE_USAGE_PAGE="true",
               N8N_BLOCK_ENV_ACCESS_IN_NODE="true",
               GENERIC_TIMEZONE="UTC", N8N_LOG_LEVEL="warn", N8N_LISTEN_ADDRESS="0.0.0.0", N8N_PUSH_BACKEND="sse")
    if nb:
        env["PATH"] = f"{nb}:{env.get('PATH', '')}"
    return env


def _n8n_bin():
    for c in [NODE_DIR / "bin" / "n8n", HERE / "node_modules" / ".bin" / "n8n"]:
        if c.exists():
            return str(c)
    return shutil.which("n8n", path=_env()["PATH"])


def install_n8n(version: str = N8N_VERSION):
    """Install Node.js 24 (if this machine has an older one) and n8n into BASE/node24."""
    t = time.time()
    if _node_bin() is None:
        print("installing Node.js 24 ...")
        cmd = (f"set -e; mkdir -p {NODE_DIR}; cd /tmp; "
               "F=$(curl -fsSL https://nodejs.org/dist/latest-v24.x/ | grep -o 'node-v24[0-9.]*-linux-x64.tar.xz' | head -1); "
               "curl -fsSLO https://nodejs.org/dist/latest-v24.x/$F; "
               f"tar -xJf $F -C {NODE_DIR} --strip-components=1")
        subprocess.run(cmd, shell=True, check=True)
    if _n8n_bin() is None:
        print(f"installing n8n {version} (3-4 minutes) ...")
        prefix = NODE_DIR if (NODE_DIR / "bin" / "node").exists() else None
        cmd = f"npm install -g --no-audit --no-fund n8n@{version}" + (f" --prefix {prefix}" if prefix else "")
        r = subprocess.run(cmd, shell=True, env=_env(), capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-3000:])
            raise RuntimeError("n8n install failed (see above)")
    v = subprocess.run([_n8n_bin(), "--version"], env=_env(), capture_output=True, text=True).stdout.strip()
    print(f"n8n {v} ready ({time.time() - t:.0f}s); node at {_node_bin()}")


def n8n_cli(*args) -> str:
    r = subprocess.run([_n8n_bin(), *args], env=_env(), capture_output=True, text=True, timeout=300)
    return (r.stdout + r.stderr).strip()


def n8n_up() -> bool:
    import requests
    try:
        return requests.get(f"http://localhost:{N8N_PORT}/healthz", timeout=2).ok
    except Exception:
        return False


def webhook_live() -> bool:
    """True once the published workflow's production webhook is registered (n8n answers 'healthy'
    a few seconds before that)."""
    import requests
    try:
        r = requests.get(WEBHOOK_URL, timeout=2)
        return "not registered for GET" in r.text
    except Exception:
        return False


def stop_n8n(timeout: int = 40):
    """Stop n8n and wait until the process is really gone (a half-stopped n8n keeps port 5678 and
    serves the old workflow version)."""
    alive = lambda: subprocess.run("pgrep -f '[n]8n start'", shell=True, capture_output=True).stdout.strip()
    if not alive():
        return
    subprocess.run("pkill -f '[n]8n start' || true", shell=True)
    t = time.time()
    while alive() and time.time() - t < timeout:
        time.sleep(1)
    if alive():
        subprocess.run("pkill -9 -f '[n]8n start' || true", shell=True)
        time.sleep(2)


def start_n8n(import_workflow: bool = True, timeout: int = 240):
    """Start n8n on port 5678. The first time it also imports and publishes the triage workflow.
    Re-running it restarts n8n but keeps your edits (they live in N8N_HOME)."""
    N8N_HOME.mkdir(parents=True, exist_ok=True)
    stop_n8n()
    marker = N8N_HOME / ".soc6_imported"
    if import_workflow and not marker.exists():
        print(n8n_cli("import:workflow", f"--input={WORKFLOW}")[-300:])
        print(n8n_cli("publish:workflow", f"--id={WORKFLOW_ID}")[-300:])
        marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    log = open(BASE / "n8n.log", "w")
    subprocess.Popen([_n8n_bin(), "start"], env=_env(), stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    t = time.time()
    while time.time() - t < timeout:
        if n8n_up() and webhook_live():
            print(f"n8n up on port {N8N_PORT} in {time.time() - t:.0f}s; webhook live: {WEBHOOK_URL}")
            return
        time.sleep(3)
    print(Path(BASE / "n8n.log").read_text()[-3000:])
    raise TimeoutError("n8n did not start (log above)")


def replay_feed(limit: int | None = None, delay: float = 0.05, wait: bool = True) -> dict:
    """POST the live-feed alerts (Jan 24 12:00-24:00 UTC) to the n8n webhook, as a SIEM would."""
    import requests
    from .server import triage_log, cases
    feed = load_feed()[:limit]
    before_t, before_c = len(triage_log()), len(cases())
    ok = 0
    for e in feed:
        r = requests.post(WEBHOOK_URL, json=e, timeout=30)
        ok += r.ok
        if not r.ok and ok == 0:
            raise RuntimeError(f"webhook answered {r.status_code}: {r.text[:300]} -- is the workflow published?")
        time.sleep(delay)
    print(f"posted {ok}/{len(feed)} alerts to n8n")
    if wait:  # n8n works asynchronously: wait until the triage log stops growing
        last, still = -1, 0
        for _ in range(600):
            n = len(triage_log())
            still = still + 1 if n == last else 0
            if n - before_t >= ok or still >= 20:
                break
            last = n
            time.sleep(1)
    out = {"posted": ok, "triaged": len(triage_log()) - before_t, "cases_opened": len(cases()) - before_c}
    print(out)
    return out
