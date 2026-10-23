"""Knowledge-graph helpers on top of LlamaIndex's SimplePropertyGraphStore.

The store holds LabelledNodes (label = ontology class in UPPER_SNAKE, name = natural key) and
Relations (label = object property in UPPER_SNAKE). These helpers read it; you build it in B.1.

Node naming convention (the B.1 contract):
    HOST            host name               "vpn"
    IP_ADDRESS      the address             "172.19.131.174"
    PERSON          staff username          "tbaker"
    ACCOUNT         user name from alerts   "alyssa.hayes"
    RULE            "rule:<id>"             "rule:31151"
    TECHNIQUE       ATT&CK id               "T1110.001"
    TACTIC          tactic name             "Credential Access"
    ALERT_GROUP     "AG|host|rule_id|src_ip or -|hour"   "AG|vpn|20101|172.19.131.174|2022-01-21T00:00Z"
    EXERCISE_WINDOW window id               "EX-01"
"""
from __future__ import annotations

import gzip
import ipaddress
import re
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .data import HERE, load_alerts, load_hosts, load_staff, load_windows, alert_groups

RESTRICTED_FALLBACK = {"labels": {"EXERCISE_WINDOW"}, "relations": {"OVERLAPS"},
                       "properties": {"phone", "onCall", "phase", "windowStart", "windowEnd"}}
_SCHEMA = None


def schema(path=None, quiet=True) -> dict:
    """The schema from YOUR ontology (ontology/soc.ttl). If it is incomplete, a built-in fallback
    restriction list is used so later parts still work (Part A is graded on its own check)."""
    global _SCHEMA
    from .onto import schema_from_ontology, MUST_BE_RESTRICTED
    S = schema_from_ontology(path)
    if any(v - S["restricted"][k] for k, v in MUST_BE_RESTRICTED.items()):
        if not quiet:
            print("NOTE: ontology/soc.ttl does not mark all restricted items yet -- using the built-in list.")
        S["restricted"] = {k: set(v) | S["restricted"][k] for k, v in RESTRICTED_FALLBACK.items()}
    _SCHEMA = S
    return S


def restricted() -> dict:
    return (_SCHEMA or schema())["restricted"]


def is_internal(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return False


# ----------------------------------------------------------------------- fast read-only index
class KGIndex:
    """A pandas view of a SimplePropertyGraphStore (rebuilt when the store changes size)."""

    def __init__(self, store):
        g = store.graph
        self.nodes = {k: (n.label, dict(n.properties)) for k, n in g.nodes.items()}
        self.edges = pd.DataFrame([(r.source_id, r.label, r.target_id) for r in g.relations.values()],
                                  columns=["src", "rel", "dst"])
        self.by_label: dict[str, list] = {}
        for k, (lab, _) in self.nodes.items():
            self.by_label.setdefault(lab, []).append(k)
        ag = [dict(p, id=k) for k, (lab, p) in self.nodes.items() if lab == "ALERT_GROUP"]
        self.groups = pd.DataFrame(ag)
        if len(self.groups):
            e = self.edges
            for rel, col in [("ON_HOST", "host_n"), ("FROM_IP", "src_n"), ("MATCHES_RULE", "rule_n")]:
                m = e[e.rel == rel].drop_duplicates("src").set_index("src")["dst"]
                self.groups[col] = self.groups["id"].map(m)
            for c in ("count", "maxLevel"):
                if c in self.groups:
                    self.groups[c] = pd.to_numeric(self.groups[c], errors="coerce").fillna(0).astype(int)
        self.size = (len(g.nodes), len(g.relations))

    def neighbors(self, node, rel=None, direction="both"):
        e = self.edges
        out = e[e.src == node] if direction in ("out", "both") else e.iloc[0:0]
        inn = e[e.dst == node] if direction in ("in", "both") else e.iloc[0:0]
        if rel:
            out, inn = out[out.rel == rel], inn[inn.rel == rel]
        return [(r.rel, r.dst, "out") for r in out.itertuples()] + [(r.rel, r.src, "in") for r in inn.itertuples()]


_IDX: dict = {}


def index(store) -> KGIndex:
    key = id(store)
    size = (len(store.graph.nodes), len(store.graph.relations))
    if key not in _IDX or _IDX[key].size != size:
        _IDX[key] = KGIndex(store)
    return _IDX[key]


# ----------------------------------------------------------------------- persistence
def save_kg(store, path="soc_kg.json"):
    store.persist(str(path))
    print("saved", path, f"({Path(path).stat().st_size / 1e6:.1f} MB)")


def load_kg(path="soc_kg.json"):
    from llama_index.core.graph_stores import SimplePropertyGraphStore
    path = Path(path)
    if path.suffix == ".gz":
        tmp = Path(tempfile.mkdtemp()) / path.stem
        with gzip.open(path, "rb") as f, open(tmp, "wb") as g:
            shutil.copyfileobj(f, g)
        path = tmp
    return SimplePropertyGraphStore.from_persist_path(str(path))


def load_reference_kg():
    """The instructor's KG. Use it ONLY if your B.1 does not pass: you lose the B.1 points, but
    Parts C-G will work."""
    print("Loading the REFERENCE KG (B.1 is not credited when you use this).")
    return load_kg(HERE / "data" / "reference_kg.json.gz")


# ----------------------------------------------------------------------- Part B check
def _expected():
    A, Hs, P, W = load_alerts(), load_hosts(), load_staff(), load_windows()
    G = alert_groups(A)
    ips = set(Hs.ip) | set(A.src_ip) | set(A.dst_ip)
    ips.discard("")
    users = (set(A.src_user) | set(A.dst_user)) - {""}
    T = pd.read_csv(HERE / "data" / "attack_techniques.csv")
    tactics = {t for s in T.tactics for t in s.split("|")}
    W2 = W.assign(s=pd.to_datetime(W.windowStart if "windowStart" in W else W.start, utc=True),
                  e=pd.to_datetime(W.windowEnd if "windowEnd" in W else W.end, utc=True))
    fs, ls = pd.to_datetime(G.first_seen, utc=True), pd.to_datetime(G.last_seen, utc=True)
    overl = sum(int(((fs <= w.e) & (ls >= w.s)).sum()) for w in W2.itertuples())
    return {"HOST": len(Hs), "PERSON": len(P), "ALERT_GROUP": len(G), "RULE": A.rule_id.nunique(),
            "IP_ADDRESS": len(ips), "ACCOUNT": len(users), "TECHNIQUE": len(T), "TACTIC": len(tactics),
            "EXERCISE_WINDOW": len(W), "OVERLAPS": overl}


def check_kg(store) -> bool:
    """The Part B.1 check: counts, ontology conformance, spot facts, SHACL on a sample."""
    ok = True

    def line(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  [{'PASS' if good else 'FAIL'}] {name:55s} {detail}")

    S = schema()
    X = index(store)
    exp = _expected()
    print(f"Part B.1 check -- {X.size[0]:,} nodes, {X.size[1]:,} relations")
    for lab in ["HOST", "PERSON", "ALERT_GROUP", "RULE", "IP_ADDRESS", "ACCOUNT", "TECHNIQUE", "TACTIC",
                "EXERCISE_WINDOW"]:
        n = len(X.by_label.get(lab, []))
        line(f"{lab} nodes", n == exp[lab], f"{n} (expected {exp[lab]})")
    bad_lab = sorted(set(X.by_label) - set(S["entity_labels"]))
    line("every node label is an ontology class", not bad_lab, f"not in ontology: {bad_lab}" if bad_lab else "")
    lab_of = {k: v[0] for k, v in X.nodes.items()}
    trip = X.edges.assign(sl=X.edges.src.map(lab_of), tl=X.edges.dst.map(lab_of))
    allowed = set(S["validation"])
    bad = trip[[(a, b, c) not in allowed for a, b, c in zip(trip.sl, trip.rel, trip.tl)]]
    line("every relation matches a domain/range in the ontology", bad.empty,
         "" if bad.empty else f"{len(bad)} bad, e.g. {bad.iloc[0][['sl', 'rel', 'tl']].tolist()}")
    n_ov = int((X.edges.rel == "OVERLAPS").sum())
    line("OVERLAPS edges (alert group time range intersects a window)", n_ov == exp["OVERLAPS"],
         f"{n_ov} (expected {exp['OVERLAPS']})")
    gid = "AG|intranet-server|31151|172.19.131.174|2022-01-24T03:00Z"
    p = X.nodes.get(gid, (None, {}))[1]
    line("spot: the intranet-server 31151 burst group", str(p.get("count")) == "548" and int(p.get("maxLevel", 0)) == 10,
         f"count={p.get('count')} maxLevel={p.get('maxLevel')}")
    has = lambda s, r, d: ((X.edges.src == s) & (X.edges.rel == r) & (X.edges.dst == d)).any()
    line("spot: vpn HAS_IP 172.19.131.174, OWNED_BY tbaker", has("vpn", "HAS_IP", "172.19.131.174") and has("vpn", "OWNED_BY", "tbaker"))
    line("spot: rule:9751 MAPS_TO T1110, T1110 IN_TACTIC Credential Access",
         has("rule:9751", "MAPS_TO", "T1110") and has("T1110", "IN_TACTIC", "Credential Access"))
    line("spot: ON_HOST / MATCHES_RULE / FROM_IP on that group",
         has(gid, "ON_HOST", "intranet-server") and has(gid, "MATCHES_RULE", "rule:31151") and has(gid, "FROM_IP", "172.19.131.174"))
    typed = isinstance(p.get("count"), int) and isinstance(X.nodes.get("vpn", (0, {}))[1].get("criticality"), int)
    line("integers stored as int (count, criticality)", typed)
    try:
        from .onto import validate_shapes
        conf, n, _ = validate_shapes(kg_to_rdf(store, limit=300))
        line("SHACL on a 300-group sample of YOUR KG", conf, f"{n} violations" if not conf else "")
    except Exception as e:
        line("SHACL on your KG", False, f"{type(e).__name__}: {e}")
    print("ALL PASS" if ok else "Fix the FAIL lines (ask your assistant to compare with the contract), re-run.")
    return ok


def kg_to_rdf(store, limit=300):
    """Convert (part of) the property graph to RDF using the ontology names, for SHACL."""
    import rdflib
    from rdflib import RDF, Literal, URIRef, XSD
    from .onto import SOC
    S = schema()
    cls_of = {v: k for k, v in S["classes"].items()}
    rel_of = {v[0]: k for k, v in S["relations"].items()}
    X = index(store)
    keep = set(X.by_label.get("ALERT_GROUP", [])[:limit])
    keep |= {k for k, (lab, _) in X.nodes.items() if lab != "ALERT_GROUP"}
    u = lambda k: URIRef(SOC + "n/" + re.sub(r"[^A-Za-z0-9._-]", "_", k))
    g = rdflib.Graph()
    prop_map = {"ruleLevel": XSD.integer, "criticality": XSD.integer, "count": XSD.integer, "maxLevel": XSD.integer}
    for k in keep:
        lab, props = X.nodes[k]
        if lab in cls_of:
            g.add((u(k), RDF.type, URIRef(SOC + cls_of[lab])))
        for pk, pv in props.items():
            if pk in S["properties"] and S["properties"][pk][0] in ("", lab) and pv not in (None, ""):
                dt = prop_map.get(pk)
                g.add((u(k), URIRef(SOC + pk), Literal(int(pv), datatype=dt) if dt else Literal(str(pv))))
    for r in X.edges.itertuples():
        if r.src in keep and r.dst in keep and r.rel in rel_of:
            g.add((u(r.src), URIRef(SOC + rel_of[r.rel]), u(r.dst)))
    return g


# ----------------------------------------------------------------------- entity linking
_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
TECH_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
RULE_RE = re.compile(r"\brule[ :#]*(\d{3,6})\b", re.I)


def parse_time_window(text: str, default_year=2022):
    """'on Jan 24' -> (2022-01-24T00:00Z, 2022-01-25T00:00Z); 'between 03:00 and 05:00 on 2022-01-24'
    -> (03:00, 05:00); 'before Jan 23' -> (None, 2022-01-23T00:00Z). Returns (since, until) as
    pandas Timestamps or None."""
    t = text.lower()
    day = None
    m = re.search(r"(20\d\d)-(\d\d)-(\d\d)", t)
    if m:
        day = pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)), tz="UTC")
    else:
        m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})\b", t) or \
            re.search(r"\b(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", t)
        if m:
            a, b = m.groups()
            mon, d = (_MONTHS[a[:3]], int(b)) if a[:3] in _MONTHS else (_MONTHS[b[:3]], int(a))
            day = pd.Timestamp(default_year, mon, d, tz="UTC")
    hours = re.findall(r"\b(\d{1,2}):(\d\d)\b", t)
    if day is None:
        return None, None
    if "before" in t and not hours:
        return None, day
    if ("after" in t or "since" in t) and not hours:
        return day + pd.Timedelta(days=1), None
    if len(hours) >= 2:
        h1, h2 = hours[0], hours[1]
        return (day + pd.Timedelta(hours=int(h1[0]), minutes=int(h1[1])),
                day + pd.Timedelta(hours=int(h2[0]), minutes=int(h2[1])))
    if len(hours) == 1:
        s = day + pd.Timedelta(hours=int(hours[0][0]), minutes=int(hours[0][1]))
        if "before" in t:
            return None, s
        return s, s + pd.Timedelta(hours=1)
    return day, day + pd.Timedelta(days=1)


def link_entities(text: str, store) -> dict:
    """Find the KG nodes a question talks about. Returns a dict of lists plus a time window:
    {"hosts", "ips", "persons", "accounts", "rules", "techniques", "tactics", "windows",
     "since", "until", "min_level"}"""
    X = index(store)
    low = text.lower()
    out = {k: [] for k in ("hosts", "ips", "persons", "accounts", "rules", "techniques", "tactics", "windows")}
    for h in sorted(X.by_label.get("HOST", []), key=len, reverse=True):
        if re.search(rf"(?<![\w-]){re.escape(h.lower())}(?![\w-])", low) and not any(h in o for o in out["hosts"]):
            out["hosts"].append(h)
    out["ips"] = [ip for ip in dict.fromkeys(IP_RE.findall(text)) if ip in X.nodes]
    for p in X.by_label.get("PERSON", []):
        props = X.nodes[p][1]
        names = [p.lower()] + [n.lower() for n in str(props.get("fullName", "")).split() if len(n) > 2]
        if any(re.search(rf"\b{re.escape(n)}\b", low) for n in names):
            out["persons"].append(p)
    for a in X.by_label.get("ACCOUNT", []):
        first = a.split(".")[0].lower()
        if a.lower() in low or (len(first) > 3 and re.search(rf"\b{re.escape(first)}\b", low)):
            out["accounts"].append(a)
    rid = set(RULE_RE.findall(text)) | {r for r in re.findall(r"\b(\d{4,5})\b", text) if f"rule:{r}" in X.nodes}
    out["rules"] = [f"rule:{r}" for r in sorted(rid) if f"rule:{r}" in X.nodes]
    out["techniques"] = [t for t in dict.fromkeys(TECH_RE.findall(text)) if t in X.nodes]
    out["tactics"] = [t for t in X.by_label.get("TACTIC", []) if t.lower() in low]
    out["windows"] = [w for w in X.by_label.get("EXERCISE_WINDOW", []) if w.lower() in low]
    out["since"], out["until"] = parse_time_window(text)
    m = re.search(r"level[\s-]*(?:of\s*)?(\d{1,2})", low) or re.search(r"level\s*(?:>=|≥)\s*(\d{1,2})", low)
    out["min_level"] = int(m.group(1)) if m else 0
    return out


# ----------------------------------------------------------------------- context builders
def _fmt_props(props: dict, role: str) -> str:
    R = restricted()["properties"] if role != "manager" else set()
    return ", ".join(f"{k}={v}" for k, v in props.items() if k not in R and v not in ("", None, [])) or "-"


def node_lines(store, node: str, role: str = "tier1", max_neighbors: int = 12) -> list[str]:
    """One line for the node and one per non-alert-group neighbor (alert groups are summarized
    by alert_summary instead). Restricted labels/properties are hidden unless role == 'manager'."""
    X = index(store)
    if node not in X.nodes:
        return []
    R = restricted() if role != "manager" else {"labels": set(), "relations": set(), "properties": set()}
    lab, props = X.nodes[node]
    if lab in R["labels"]:
        return []
    lines = [f"{lab} {node}: {_fmt_props(props, role)}"]
    nb = [(r, o, d) for r, o, d in X.neighbors(node) if X.nodes.get(o, ("", {}))[0] != "ALERT_GROUP"
          and r not in R["relations"] and X.nodes.get(o, ("", {}))[0] not in R["labels"]]
    for rel, other, d in nb[:max_neighbors]:
        olab, oprops = X.nodes[other]
        arrow = f"{node} -{rel}-> {other}" if d == "out" else f"{other} -{rel}-> {node}"
        lines.append(f"  {arrow}   ({olab}: {_fmt_props(oprops, role)})")
    if len(nb) > max_neighbors:
        lines.append(f"  ... {len(nb) - max_neighbors} more neighbors")
    return lines


def alert_summary(store, host=None, ip=None, account=None, technique=None, rule=None,
                  since=None, until=None, min_level=0, top=5, role="tier1") -> list[str]:
    """Aggregate the ALERT_GROUP nodes that match the filters into a few lines of numbers:
    total alerts, by rule, by day, top source IPs, first/last seen. This is how a 37,808-alert
    history fits in a context window."""
    X = index(store)
    G = X.groups
    if G.empty:
        return ["(no alert groups in the KG)"]
    m = pd.Series(True, index=G.index)
    desc = []
    if host:
        m &= G.host_n == host; desc.append(f"host={host}")
    if ip:
        to = set(X.edges[(X.edges.rel == "TO_IP") & (X.edges.dst == ip)].src)
        m &= (G.src_n == ip) | G.id.isin(to); desc.append(f"ip={ip}")
    if account:
        ids = set(X.edges[(X.edges.rel == "INVOLVES_ACCOUNT") & (X.edges.dst == account)].src)
        m &= G.id.isin(ids); desc.append(f"account={account}")
    if technique:
        rules = set(X.edges[(X.edges.rel == "MAPS_TO") & (X.edges.dst == technique)].src)
        m &= G.rule_n.isin(rules); desc.append(f"technique={technique}")
    if rule:
        m &= G.rule_n == rule; desc.append(f"rule={rule}")
    fs, ls = pd.to_datetime(G.firstSeen, utc=True), pd.to_datetime(G.lastSeen, utc=True)
    if since is not None:
        m &= ls >= since; desc.append(f"since={since:%Y-%m-%d %H:%M}")
    if until is not None:
        m &= fs < until; desc.append(f"until={until:%Y-%m-%d %H:%M}")
    if min_level:
        m &= G.maxLevel >= min_level; desc.append(f"level>={min_level}")
    g = G[m]
    head = f"ALERT SUMMARY ({', '.join(desc) or 'all'}):"
    if g.empty:
        return [head, "  no matching alerts in the KG (history ends 2022-01-24 12:00 UTC)"]
    lines = [head, f"  total alerts={int(g['count'].sum())} in {len(g)} alert groups; "
                   f"first={g.firstSeen.min()} last={g.lastSeen.max()}; max level={int(g.maxLevel.max())}"]
    br = g.groupby("rule_n").agg(n=("count", "sum"), d=("description", lambda s: s.mode()[0]),
                                 v=("description", "nunique")).sort_values("n", ascending=False).head(top)
    lines.append("  by rule: " + "; ".join(
        f"{r} '{x.d}'{f' (+{x.v - 1} other signatures)' if x.v > 1 else ''} x{int(x.n)}" for r, x in br.iterrows()))
    if not host:
        bh = g.groupby("host_n")["count"].sum().sort_values(ascending=False).head(top)
        lines.append("  by host: " + "; ".join(f"{h} x{int(c)}" for h, c in bh.items()))
    bd = g.assign(day=g.firstSeen.str[:10]).groupby("day")["count"].sum()
    lines.append("  by day: " + "; ".join(f"{d} x{int(c)}" for d, c in bd.items()))
    bs = g[g.src_n.notna()].groupby("src_n")["count"].sum().sort_values(ascending=False).head(top)
    if len(bs):
        lines.append("  top source IPs: " + "; ".join(f"{s} x{int(c)}" for s, c in bs.items()))
    acc = X.edges[(X.edges.rel == "INVOLVES_ACCOUNT") & X.edges.src.isin(set(g.id))]
    if len(acc):
        cnt = g.set_index("id")["count"]
        ba = acc.assign(n=acc.src.map(cnt)).groupby("dst")["n"].sum().sort_values(ascending=False).head(top)
        lines.append("  accounts: " + "; ".join(f"{a} x{int(c)}" for a, c in ba.items()))
    hi = g[g.maxLevel >= 10].sort_values("firstSeen").head(3)
    for r in hi.itertuples():
        lines.append(f"  high: {r.firstSeen} {r.host_n} {r.rule_n} level {r.maxLevel} x{r.count} from {r.src_n}")
    if role == "manager":
        ov = X.edges[(X.edges.rel == "OVERLAPS") & X.edges.src.isin(set(g.id))]
        if len(ov):
            w = ov.groupby("dst").size()
            lines.append("  overlapping exercise windows (restricted): " +
                         "; ".join(f"{k} {X.nodes[k][1].get('phase', '')} ({n} groups)" for k, n in w.items()))
    return lines


def windows_lines(store, role="tier1") -> list[str]:
    """The exercise schedule -- manager only."""
    if role != "manager":
        return []
    X = index(store)
    return [f"EXERCISE_WINDOW {w}: {_fmt_props(X.nodes[w][1], role)}"
            for w in sorted(X.by_label.get("EXERCISE_WINDOW", []))]


# ----------------------------------------------------------------------- visual
def show_kg(store, center: str, depth: int = 1, limit: int = 80, role: str = "manager",
            html="kg_view.html", height="600px", display_it=True, max_groups: int = 12):
    """pyvis view of the neighborhood of one node. Alert groups are small grey dots (hover for details),
    at most max_groups per node, so hosts with hundreds of groups stay readable."""
    from pyvis.network import Network
    X = index(store)
    colors = {"HOST": "#3333cc", "IP_ADDRESS": "#00aa77", "PERSON": "#f0a202", "ACCOUNT": "#d95d39",
              "RULE": "#7768ae", "TECHNIQUE": "#c2185b", "TACTIC": "#880e4f", "ALERT_GROUP": "#9e9e9e",
              "EXERCISE_WINDOW": "#e4572e"}
    R = restricted() if role != "manager" else {"labels": set(), "relations": set(), "properties": set()}
    seen, frontier, edges = {center}, [center], []
    for _ in range(depth):
        nxt = []
        for n in frontier:
            n_groups = 0
            for rel, o, d in X.neighbors(n):
                olab = X.nodes.get(o, ("", {}))[0]
                if len(seen) >= limit or olab in R["labels"]:
                    continue
                if olab == "ALERT_GROUP":
                    n_groups += 1
                    if n_groups > max_groups:
                        continue
                edges.append((n, rel, o) if d == "out" else (o, rel, n))
                if o not in seen:
                    seen.add(o); nxt.append(o)
        frontier = nxt
    net = Network(height=height, width="100%", directed=True, notebook=True, cdn_resources="remote")
    for n in seen:
        lab, props = X.nodes.get(n, ("?", {}))
        small = lab == "ALERT_GROUP"
        net.add_node(n, label=" " if small else n, title=f"{lab} {n}\n{_fmt_props(props, role)}",
                     color=colors.get(lab, "#555"), size=6 if small else (28 if n == center else 16))
    for a, rel, b in edges:
        if a in seen and b in seen:
            net.add_edge(a, b, title=rel, arrows="to")
    net.write_html(html, notebook=False)
    if display_it:
        from IPython.display import HTML, display
        display(HTML(open(html).read()))
    return html


def kg_stats(store):
    X = index(store)
    nodes = pd.Series({k: len(v) for k, v in X.by_label.items()}).sort_values()
    rels = X.edges.rel.value_counts().sort_values()
    return nodes, rels
