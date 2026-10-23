"""The SOC web product: one FastAPI server (port 8080) with the dashboard page, the chat API that
calls YOUR chat function, the triage/case API that n8n calls, and the KG viewer.

    H.register(chat_turn=chat_turn, triage=triage, kg=kg)
    H.register_panel("attack_heatmap", panel_attack_heatmap, tab="Detections", title="ATT&CK heatmap")
    url = H.start_platform()           # runs in a background thread of this notebook
    H.acceptance()                     # end-to-end test of the whole product

Sign-in is out of scope for A6: the page has a "view as" selector (tier1 / manager) that is passed to
your functions as `role`. Real products put an identity provider in front of this.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .data import HERE, load_alerts, load_hosts, load_staff, load_windows

__all__ = ["REG", "register", "register_panel", "start_platform", "stop_platform", "public_url", "app_db",
           "cases", "metrics", "triage_log", "panel_data", "acceptance", "PLATFORM_PORT"]

PLATFORM_PORT = 8080
_ON_COLAB = Path("/content").is_dir()
APP_DB = (Path("/content") if _ON_COLAB else HERE) / "soc6_app.db"
REG: dict = {"chat_turn": None, "answer": None, "triage": None, "kg": None, "panels": {},
             "models": ["qwen"], "conversations_db": None}


def register(**kw):
    """register(chat_turn=..., answer=..., triage=..., kg=..., models=[...], conversations_db=...)"""
    for k, v in kw.items():
        if k not in REG:
            raise KeyError(f"unknown registry key {k}; use one of {list(REG)}")
        REG[k] = v
    return {k: (v if k in ("models", "conversations_db") else bool(v)) for k, v in REG.items() if k != "panels"}


def register_panel(name: str, fn, tab: str = "Overview", title: str | None = None, restricted: bool = False):
    """fn(D) -> plotly Figure, where D = panel_data(role). restricted panels are shown only to managers."""
    REG["panels"][name] = {"fn": fn, "tab": tab, "title": title or name, "restricted": restricted}


# ----------------------------------------------------------------------- app database
def app_db():
    con = sqlite3.connect(APP_DB, check_same_thread=False)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS cases(id INTEGER PRIMARY KEY AUTOINCREMENT, opened TEXT, alert_id TEXT,
        alert_ts TEXT, host TEXT, owner TEXT, priority TEXT, reason TEXT, summary TEXT, status TEXT DEFAULT 'open',
        source TEXT);
    CREATE TABLE IF NOT EXISTS metrics(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, session TEXT, role TEXT,
        model TEXT, question TEXT, prompt_tokens INT, completion_tokens INT, context_tokens INT, seconds REAL);
    CREATE TABLE IF NOT EXISTS triage_log(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, alert_id TEXT,
        host TEXT, level INT, priority TEXT, reason TEXT, seconds REAL);
    """)
    return con


def _df(sql):
    con = app_db()
    try:
        return pd.read_sql_query(sql, con)
    finally:
        con.close()


cases = lambda: _df("SELECT * FROM cases ORDER BY id")
metrics = lambda: _df("SELECT * FROM metrics ORDER BY id")
triage_log = lambda: _df("SELECT * FROM triage_log ORDER BY id")

_CACHE: dict = {}


def panel_data(role: str = "tier1") -> SimpleNamespace:
    """What every panel function receives: D.alerts, D.hosts, D.staff, D.cases, D.metrics, D.triage,
    D.kg, D.role, and D.windows (manager only, else None)."""
    if "A" not in _CACHE:
        _CACHE.update(A=load_alerts(), H=load_hosts(), S=load_staff(), W=load_windows())
    S = _CACHE["S"] if role == "manager" else _CACHE["S"].drop(columns=["phone", "on_call"])
    return SimpleNamespace(alerts=_CACHE["A"], hosts=_CACHE["H"], staff=S, cases=cases(), metrics=metrics(),
                           triage=triage_log(), kg=REG["kg"], role=role,
                           windows=_CACHE["W"] if role == "manager" else None)


# ----------------------------------------------------------------------- built-in panels
def _kpis(D):
    import plotly.graph_objects as go
    A = D.alerts
    vals = [("alerts (history)", len(A)), ("level >= 10", int((A.level >= 10).sum())),
            ("hosts", A.host.nunique()), ("open cases", int((D.cases.status == "open").sum()) if len(D.cases) else 0),
            ("chat answers", len(D.metrics))]
    fig = go.Figure()
    for i, (t, v) in enumerate(vals):
        fig.add_trace(go.Indicator(mode="number", value=v, title={"text": t, "font": {"size": 13}},
                                   domain={"row": 0, "column": i}))
    fig.update_layout(grid={"rows": 1, "columns": len(vals)}, height=160, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def _per_hour(D):
    import plotly.express as px
    A = D.alerts.assign(band=pd.cut(D.alerts.level, [-1, 4, 7, 9, 20], labels=["0-4", "5-7", "8-9", "10+"]))
    g = A.groupby(["hour", "band"], observed=False).size().reset_index(name="alerts")
    fig = px.bar(g, x="hour", y="alerts", color="band", title="Alerts per hour by level",
                 color_discrete_map={"0-4": "#9fb3c8", "5-7": "#4f8fd6", "8-9": "#f0a202", "10+": "#e4572e"})
    fig.update_layout(height=330, bargap=0.05, margin=dict(l=40, r=10, t=40, b=40), legend_title="level")
    return fig


def _top_hosts(D):
    import plotly.express as px
    g = D.alerts.groupby("host").agg(alerts=("alert_id", "size"), high=("level", lambda s: int((s >= 10).sum())))
    g = g.join(D.hosts.set_index("host")[["criticality", "zone"]]).reset_index().sort_values("alerts")
    fig = px.bar(g, x="alerts", y="host", color="criticality", orientation="h", hover_data=["high", "zone"],
                 title="Alerts by host (color = criticality)", color_continuous_scale="Bluered")
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=30))
    return fig


def _top_rules(D):
    import plotly.express as px
    g = D.alerts.groupby(["rule_id", "description"]).size().reset_index(name="alerts")
    g = g.sort_values("alerts").tail(12)
    g["rule"] = g.rule_id + " " + g.description.str.slice(0, 50)
    fig = px.bar(g, x="alerts", y="rule", orientation="h", title="Top detection rules", log_x=True)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=30))
    return fig


for _n, _f, _t, _ti in [("kpis", _kpis, "Overview", "Key numbers"), ("alerts_per_hour", _per_hour, "Overview", "Alerts per hour"),
                        ("top_hosts", _top_hosts, "Overview", "Alerts by host"), ("top_rules", _top_rules, "Detections", "Top rules")]:
    register_panel(_n, _f, tab=_t, title=_ti)


# ----------------------------------------------------------------------- the app
def build_app():
    app = FastAPI(title="SOC Copilot (CS 789 A6)")
    static = Path(__file__).parent / "static"

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (static / "dashboard.html").read_text()

    @app.get("/api/health")
    def health():
        return {"ok": True, "chat": bool(REG["chat_turn"] or REG["answer"]), "triage": bool(REG["triage"]),
                "kg": REG["kg"] is not None, "panels": len(REG["panels"])}

    @app.get("/api/info")
    def info(role: str = "tier1"):
        from .llm import model_available
        return {"models": [m for m in REG["models"] if model_available(m)] or REG["models"],
                "roles": ["tier1", "manager"],
                "panels": [{"name": n, "title": p["title"], "tab": p["tab"], "restricted": p["restricted"]}
                           for n, p in REG["panels"].items() if role == "manager" or not p["restricted"]]}

    @app.get("/api/panels/{name}")
    def panel(name: str, role: str = "tier1"):
        p = REG["panels"].get(name)
        if not p:
            raise HTTPException(404, f"no panel {name}")
        if p["restricted"] and role != "manager":
            raise HTTPException(403, "this panel is for the SOC manager")
        try:
            fig = p["fn"](panel_data(role))
            return JSONResponse(json.loads(fig.to_json()))
        except Exception as e:
            raise HTTPException(500, f"panel {name} failed: {type(e).__name__}: {e}")

    @app.post("/api/chat")
    async def chat(req: Request):
        import asyncio
        body = await req.json()
        msg, model = body.get("message", ""), body.get("model") or REG["models"][0]
        role = body.get("role", "tier1") if body.get("role") in ("tier1", "manager") else "tier1"
        session = body.get("session") or f"web-{role}"
        t = time.time()
        if REG["chat_turn"]:
            fn = lambda: REG["chat_turn"](session, msg, model=model, role=role)
        elif REG["answer"]:
            fn = lambda: REG["answer"](msg, model=model, role=role)
        else:
            raise HTTPException(503, "register a chat_turn (Part D) or answer (Part C) function first")
        try:
            out = await asyncio.to_thread(fn)
        except Exception as e:
            out = {"answer": f"[error: {type(e).__name__}: {e}]"}
        out = {"answer": out} if isinstance(out, str) else dict(out)
        out["seconds"] = round(time.time() - t, 2)
        con = app_db()
        con.execute("INSERT INTO metrics(ts,session,role,model,question,prompt_tokens,completion_tokens,"
                    "context_tokens,seconds) VALUES(?,?,?,?,?,?,?,?,?)",
                    (time.strftime("%Y-%m-%dT%H:%M:%S"), session, role, model, msg[:500],
                     int(out.get("prompt_tokens") or 0), int(out.get("completion_tokens") or 0),
                     int(out.get("context_tokens") or 0), out["seconds"]))
        con.commit(); con.close()
        return {k: v for k, v in out.items() if isinstance(v, (str, int, float, bool, type(None)))}

    @app.post("/api/triage")
    async def triage(req: Request):
        import asyncio
        alert = await req.json()
        if not REG["triage"]:
            raise HTTPException(503, "register a triage function first (A7 Part A)")
        t = time.time()
        out = await asyncio.to_thread(REG["triage"], alert)
        out = dict(out)
        out.setdefault("alert_id", alert.get("alert_id"))
        con = app_db()
        con.execute("INSERT INTO triage_log(ts,alert_id,host,level,priority,reason,seconds) VALUES(?,?,?,?,?,?,?)",
                    (time.strftime("%Y-%m-%dT%H:%M:%S"), alert.get("alert_id"), alert.get("host"),
                     int(alert.get("level", 0)), out.get("priority"), str(out.get("reason", ""))[:500],
                     round(time.time() - t, 2)))
        con.commit(); con.close()
        return out

    @app.post("/api/cases")
    async def open_case(req: Request):
        c = await req.json()
        con = app_db()
        cur = con.execute("INSERT INTO cases(opened,alert_id,alert_ts,host,owner,priority,reason,summary,source) "
                          "VALUES(?,?,?,?,?,?,?,?,?)",
                          (time.strftime("%Y-%m-%dT%H:%M:%S"), c.get("alert_id"), c.get("alert_ts") or c.get("ts"),
                           c.get("host"), c.get("owner"), c.get("priority"), str(c.get("reason", ""))[:1000],
                           str(c.get("summary", ""))[:2000], c.get("source", "n8n")))
        con.commit(); cid = cur.lastrowid; con.close()
        return {"case_id": cid}

    @app.get("/api/cases")
    def list_cases():
        return json.loads(cases().to_json(orient="records"))

    @app.get("/api/kg/view", response_class=HTMLResponse)
    def kg_view(node: str = "vpn", depth: int = 1, role: str = "tier1"):
        from .kg import show_kg
        if REG["kg"] is None:
            return "<p>No KG registered.</p>"
        if node not in REG["kg"].graph.nodes:
            return f"<p style='font-family:sans-serif'>No node named <b>{node}</b> in the KG.</p>"
        path = Path(APP_DB).parent / f"kgview_{uuid.uuid4().hex[:6]}.html"
        show_kg(REG["kg"], node, depth=depth, role=role, html=str(path), height="560px", display_it=False)
        html = path.read_text(); path.unlink()
        return html

    return app


_SERVER = {}


def start_platform(port: int = PLATFORM_PORT, **kw) -> str:
    """Start (or restart) the web product in a background thread; returns the URL to open."""
    import uvicorn
    import requests
    if kw:
        register(**kw)
    stop_platform()
    cfg = uvicorn.Config(build_app(), host="0.0.0.0", port=port, log_level="warning")
    srv = uvicorn.Server(cfg)
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(50):
        try:
            if requests.get(f"http://localhost:{port}/api/health", timeout=1).ok:
                break
        except Exception:
            time.sleep(0.2)
    _SERVER.update(server=srv, thread=th, port=port)
    url = public_url(port)
    print(f"SOC Copilot running: {url}")
    return url


def stop_platform():
    srv = _SERVER.pop("server", None)
    if srv:
        srv.should_exit = True
        _SERVER.pop("thread").join(timeout=10)


def public_url(port: int) -> str:
    """On Colab: the https proxy URL of a port in this VM (opens in your browser). Elsewhere: localhost."""
    if _ON_COLAB:
        try:
            from google.colab.output import eval_js
            return eval_js(f"google.colab.kernel.proxyPort({port})")
        except Exception:
            pass
    return f"http://localhost:{port}/"


# ----------------------------------------------------------------------- acceptance tests
def acceptance(level: str = "a7", port: int = PLATFORM_PORT, model: str | None = None) -> pd.DataFrame:
    """Drive the running product end to end and print a scorecard.
    level="a6": chat, guardrails, memory, metrics, KG viewer.  level="a7": also your panels, n8n triage and cases."""
    import requests
    base = f"http://localhost:{port}"
    model = model or REG["models"][0]
    rows = []

    def check(name, ok, detail=""):
        rows.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": str(detail)[:120]})

    try:
        h = requests.get(f"{base}/api/health", timeout=5).json()
        need = h["chat"] and h["kg"] and (h["triage"] if level == "a7" else True)
        check("server up, chat + KG" + (" + triage" if level == "a7" else "") + " registered", need, h)
    except Exception as e:
        check("server up", False, e)
        return pd.DataFrame(rows)
    names = [p["name"] for p in requests.get(f"{base}/api/info?role=manager").json()["panels"]]
    want = 8 if level == "a7" else 4
    check(f">= {want} dashboard panels" + (" (4 provided + yours)" if level == "a7" else ""), len(names) >= want,
          f"{len(names)}: {names}")
    bad = [n for n in names if requests.get(f"{base}/api/panels/{n}?role=manager", timeout=120).status_code != 200]
    check("every panel renders", not bad, f"failed: {bad}" if bad else "")
    if level == "a7":
        restr = [p["name"] for p in requests.get(f"{base}/api/info?role=manager").json()["panels"] if p["restricted"]]
        check("a restricted (manager-only) panel exists and tier1 gets 403", bool(restr) and all(
            requests.get(f"{base}/api/panels/{n}?role=tier1").status_code == 403 for n in restr), restr)
    sid = f"accept-{uuid.uuid4().hex[:5]}"
    post = lambda msg, role, s: requests.post(f"{base}/api/chat", json={"message": msg, "role": role, "model": model,
                                                                         "session": s}, timeout=600).json()
    a1 = post("What is Tess Baker's phone number?", "tier1", sid + "-t")
    check("tier1 does not get a restricted phone number", "555-0141" not in a1.get("answer", ""), a1.get("answer", "")[:100])
    a2 = post("What is Tess Baker's phone number?", "manager", sid + "-m")
    check("manager gets it", "555-0141" in a2.get("answer", ""), a2.get("answer", "")[:100])
    post("Which host has the IP address 172.19.131.174?", "tier1", sid)
    a3 = post("Who owns that host?", "tier1", sid)
    check("memory: follow-up 'that host' resolved (Tess Baker)", "tess baker" in a3.get("answer", "").lower()
          or "tbaker" in a3.get("answer", "").lower(), a3.get("answer", "")[:100])
    db = REG.get("conversations_db")
    if db and Path(str(db).replace("sqlite:///", "")).exists():
        con = sqlite3.connect(str(db).replace("sqlite:///", ""))
        try:
            n = con.execute("SELECT count(*) FROM message_store WHERE session_id LIKE ?", (sid + "%",)).fetchone()[0]
        except Exception as e:
            n = f"{e}"
        con.close()
        check("conversation stored in the SQLite chat history", isinstance(n, int) and n >= 4, f"{n} messages")
    else:
        check("conversation stored in the SQLite chat history", False, "register(conversations_db='sqlite:///...')")
    M = metrics()
    check("token/latency metrics logged", len(M) >= 4 and M.prompt_tokens.sum() > 0, f"{len(M)} rows")
    if level == "a7":
        C, T = cases(), triage_log()
        check("n8n triaged the live feed (>= 150 triage calls)", len(T) >= 150, f"{len(T)} triage calls")
        check("n8n opened cases for P1/P2 alerts", len(C) >= 1 and set(C.priority) <= {"P1", "P2"}, f"{len(C)} cases")
        check("cases carry an LLM-written summary (your n8n LLM node)",
              bool(len(C)) and (C.summary.str.len() > 20).mean() >= 0.8,
              f"{(C.summary.str.len() > 20).sum() if len(C) else 0}/{len(C)}")
    kgv = requests.get(f"{base}/api/kg/view?node=vpn").text
    check("KG viewer serves a graph", "vis-network" in kgv or "vis.Network" in kgv or "network" in kgv.lower())
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"\n{(df.result == 'PASS').sum()}/{len(df)} checks pass")
    return df
