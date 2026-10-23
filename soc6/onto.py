"""Ontology tools: OWL/Turtle -> property-graph schema, SHACL validation, and a picture."""
from __future__ import annotations

import os
import re
from pathlib import Path

from .data import HERE

SOC = "https://cs789.unlv.edu/soc6#"
ONTOLOGY = Path(os.environ.get("SOC6_ONTOLOGY", HERE / "ontology" / "soc.ttl"))
SHAPES = Path(os.environ.get("SOC6_SHAPES", HERE / "ontology" / "shapes.ttl"))


def upper_snake(name: str) -> str:
    """Host -> HOST, IPAddress -> IP_ADDRESS, hasIP -> HAS_IP, loggedInto -> LOGGED_INTO."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    return s.upper()


def _local(u) -> str:
    return str(u).split("#")[-1]


def load_graph(path=None):
    import rdflib
    g = rdflib.Graph()
    g.parse(str(path or ONTOLOGY), format="turtle")
    return g


def schema_from_ontology(path=None) -> dict:
    """Read the ontology and return the vocabulary everything else uses.

    {"classes":   {"Host": "HOST", ...},
     "relations": {"hasIP": ("HAS_IP", "HOST", "IP_ADDRESS"), ...},
     "properties": {"criticality": ("HOST", "integer"), ...},
     "validation": [("HOST", "HAS_IP", "IP_ADDRESS"), ...],      # for SchemaLLMPathExtractor
     "entity_labels": [...], "relation_labels": [...],
     "restricted": {"labels": {...}, "relations": {...}, "properties": {...}}}
    """
    from rdflib import RDF, RDFS, OWL, URIRef
    g = load_graph(path)
    sens = URIRef(SOC + "sensitivity")
    restricted_uris = {s for s, o in g.subject_objects(sens) if str(o).lower() == "restricted"}
    classes = {_local(c): upper_snake(_local(c)) for c in g.subjects(RDF.type, OWL.Class)
               if str(c).startswith(SOC)}
    rels, props = {}, {}
    for p in g.subjects(RDF.type, OWL.ObjectProperty):
        d, r = g.value(p, RDFS.domain), g.value(p, RDFS.range)
        if d is None or r is None:
            continue
        rels[_local(p)] = (upper_snake(_local(p)), upper_snake(_local(d)), upper_snake(_local(r)))
    for p in g.subjects(RDF.type, OWL.DatatypeProperty):
        d, r = g.value(p, RDFS.domain), g.value(p, RDFS.range)
        props[_local(p)] = (upper_snake(_local(d)) if d is not None else "",
                            _local(r) if r is not None else "string")
    restricted = {
        "labels": {classes[_local(u)] for u in restricted_uris if _local(u) in classes},
        "relations": {rels[_local(u)][0] for u in restricted_uris if _local(u) in rels},
        "properties": {_local(u) for u in restricted_uris if _local(u) in props},
    }
    return {"classes": classes, "relations": rels, "properties": props,
            "validation": sorted({(d, l, r) for l, d, r in rels.values()}),
            "entity_labels": sorted(classes.values()),
            "relation_labels": sorted({l for l, _, _ in rels.values()}),
            "restricted": restricted}


# ----------------------------------------------------------------------- RDF views of data
def _rdf_from_tables(sample_groups: int = 40, bad_host: bool = False):
    """Build a small RDF instance graph straight from the CSV tables (independent of your KG)."""
    import rdflib
    from rdflib import RDF, Literal, URIRef, XSD
    from .data import load_alerts, load_hosts, load_staff, alert_groups
    S = rdflib.Namespace(SOC)
    g = rdflib.Graph()
    node = lambda kind, key: URIRef(f"{SOC}{kind}/{re.sub(r'[^A-Za-z0-9._-]', '_', str(key))}")
    H, P = load_hosts(), load_staff()
    for _, p in P.iterrows():
        n = node("person", p.username)
        g.add((n, RDF.type, S.Person))
    for i, h in H.iterrows():
        n = node("host", h.host)
        g.add((n, RDF.type, S.Host))
        crit = 7 if (bad_host and i == 0) else int(h.criticality)
        g.add((n, S.criticality, Literal(crit, datatype=XSD.integer)))
        g.add((n, S.zone, Literal("Basement" if (bad_host and i == 1) else h.zone)))
        g.add((n, S.ownedBy, node("person", h.owner)))
    G = alert_groups(load_alerts()).head(sample_groups)
    for _, r in G.iterrows():
        a, rule = node("ag", r.group_id), node("rule", r.rule_id)
        g.add((a, RDF.type, S.AlertGroup))
        g.add((a, S.onHost, node("host", r.host)))
        g.add((a, S.matchesRule, rule))
        g.add((a, S["count"], Literal(int(r["count"]), datatype=XSD.integer)))
        g.add((a, S.maxLevel, Literal(int(r.max_level), datatype=XSD.integer)))
        g.add((rule, RDF.type, S.Rule))
        g.add((rule, S.ruleLevel, Literal(int(r.max_level), datatype=XSD.integer)))
        g.add((rule, S.description, Literal(r.description)))
    return g


def validate_shapes(data_graph, shapes=None, ontology=None):
    """pySHACL validation. Returns (conforms, number_of_violations, report_text)."""
    from pyshacl import validate
    conforms, rg, text = validate(data_graph, shacl_graph=str(shapes or SHAPES), ont_graph=str(ontology or ONTOLOGY),
                                  inference="rdfs", advanced=True)
    from rdflib import Namespace
    SH = Namespace("http://www.w3.org/ns/shacl#")
    n = len(list(rg.subjects(SH.resultSeverity, None)))
    return conforms, n, text


# ----------------------------------------------------------------------- the Part A check
REQUIRED_CLASSES = ["Host", "IPAddress", "Person", "Account", "Rule", "Technique", "Tactic",
                    "AlertGroup", "ExerciseWindow"]
REQUIRED_RELATIONS = {"onHost": ("ALERT_GROUP", "HOST"), "fromIP": ("ALERT_GROUP", "IP_ADDRESS"),
                      "toIP": ("ALERT_GROUP", "IP_ADDRESS"), "matchesRule": ("ALERT_GROUP", "RULE"),
                      "involvesAccount": ("ALERT_GROUP", "ACCOUNT"), "overlaps": ("ALERT_GROUP", "EXERCISE_WINDOW")}
REQUIRED_PROPS = ["hour", "firstSeen", "lastSeen", "count", "maxLevel", "phase", "windowStart", "windowEnd",
                  "phone", "onCall"]
MUST_BE_RESTRICTED = {"labels": {"EXERCISE_WINDOW"}, "relations": {"OVERLAPS"},
                      "properties": {"phone", "onCall", "phase", "windowStart", "windowEnd"}}


def check_ontology(path=None, shapes=None) -> bool:
    ok = True

    def line(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  [{'PASS' if good else 'FAIL'}] {name:52s} {detail}")

    print("Part A check -- ontology", path or ONTOLOGY)
    try:
        S = schema_from_ontology(path)
    except Exception as e:
        line("Turtle parses", False, f"{type(e).__name__}: {e}")
        return False
    line("Turtle parses", True, f"{len(S['classes'])} classes, {len(S['relations'])} object props, "
                                f"{len(S['properties'])} datatype props")
    miss = [c for c in REQUIRED_CLASSES if c not in S["classes"]]
    line("all 9 classes present", not miss, f"missing: {miss}" if miss else "")
    bad = [f"{p} {S['relations'].get(p)}" for p, (d, r) in REQUIRED_RELATIONS.items()
           if p not in S["relations"] or S["relations"][p][1:] != (d, r)]
    line("AlertGroup / ExerciseWindow relations with domain & range", not bad, f"wrong or missing: {bad}" if bad else "")
    missp = [p for p in REQUIRED_PROPS if p not in S["properties"]]
    line("datatype properties present", not missp, f"missing: {missp}" if missp else "")
    rmiss = {k: sorted(v - S["restricted"][k]) for k, v in MUST_BE_RESTRICTED.items() if v - S["restricted"][k]}
    line('soc:sensitivity "restricted" on window, overlaps, phone, onCall', not rmiss, f"not restricted: {rmiss}" if rmiss else "")
    leak = S["restricted"]["properties"] & {"criticality", "zone", "role", "fullName"}
    line("nothing over-restricted (host facts stay visible)", not leak, f"{sorted(leak)}" if leak else "")
    try:
        conf, n, _ = validate_shapes(_rdf_from_tables(), shapes, path)
        line("SHACL: real sample data conforms", conf, f"{n} violations" if not conf else "")
        conf_bad, n_bad, txt = validate_shapes(_rdf_from_tables(bad_host=True), shapes, path)
        line("SHACL: your HostShape rejects criticality=7 and zone='Basement'", (not conf_bad) and n_bad >= 2,
             f"{n_bad} violations found (need >= 2)")
    except Exception as e:
        line("SHACL validation runs", False, f"{type(e).__name__}: {e}")
    print("ALL PASS" if ok else "Fix the FAIL lines, re-run this cell.")
    return ok


def show_ontology(path=None, html="ontology.html", height="560px"):
    """Draw classes (nodes) and object properties (edges); restricted items in red."""
    from pyvis.network import Network
    from IPython.display import HTML, display
    S = schema_from_ontology(path)
    net = Network(height=height, width="100%", directed=True, notebook=True, cdn_resources="in_line",
                  bgcolor="#ffffff")
    props_by = {}
    for p, (d, t) in S["properties"].items():
        props_by.setdefault(d, []).append(f"{p}: {t}" + (" (restricted)" if p in S["restricted"]["properties"] else ""))
    for c, lab in S["classes"].items():
        red = lab in S["restricted"]["labels"]
        net.add_node(lab, label=c, title="\n".join(props_by.get(lab, [])) or c, shape="box",
                     color="#e4572e" if red else "#3333cc", font={"color": "white"})
    for p, (lab, d, r) in S["relations"].items():
        if d in S["entity_labels"] and r in S["entity_labels"]:
            net.add_edge(d, r, label=p, color="#e4572e" if lab in S["restricted"]["relations"] else "#00aa77",
                         arrows="to")
    net.set_options('{"physics":{"barnesHut":{"springLength":160}}}')
    net.write_html(html, notebook=False)
    display(HTML(open(html).read()))
