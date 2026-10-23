#!/usr/bin/env python3
"""CS 789 A6 -- turn one scenario of the public AIT Alert Data Set (AIT-ADS) into the small,
summary-level tables the assignment uses.

Source: Landauer, Skopik, Wurzenberger, "AIT Alert Data Set", Zenodo, 2023,
        doi:10.5281/zenodo.8263181, license CC BY 4.0.
Changes made here (as CC BY 4.0 asks us to state): one scenario only (russellmitchell, Wazuh
alerts); each alert reduced to summary fields (time, sensor host, rule id/level/description/
groups, ATT&CK ids, IPs, ports, user names, country); the raw log line (full_log) and all
payload fields are dropped; IDs added; split into history and live feed. The host inventory,
staff list and role permissions are an invented overlay for the lab.

Usage (instructor, once):  python tools/prepare_data.py ait_ads/ labels.csv
Writes data/alerts.csv.gz, data/live_feed.jsonl, data/exercise_windows.csv, data/hosts.csv,
data/staff.csv, data/DATASET.md
"""
import collections
import csv
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCENARIO = "russellmitchell"
SPLIT = "2022-01-24T12:00:00Z"        # history < SPLIT <= live feed
HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "data"

# ---------------------------------------------------------------- invented inventory overlay
# sensor IP -> (host, role, zone, owner, criticality 1-5)
HOSTS = {
    "172.19.128.1":    ("inet-firewall",   "Perimeter firewall and network IDS sensor", "Perimeter", "tbaker", 5),
    "172.19.130.4":    ("mail",            "Public mail server (SMTP/IMAP)",            "DMZ",       "rkhan", 4),
    "172.19.130.68":   ("webserver",       "Public web server and reverse proxy",       "DMZ",       "rkhan", 4),
    "172.19.130.106":  ("cloud-share",     "File-sharing service for partners",         "DMZ",       "mlee", 3),
    "172.19.131.174":  ("vpn",             "Remote-access VPN gateway",                 "DMZ",       "tbaker", 5),
    "10.143.2.4":      ("intranet-server", "Internal intranet web portal (CMS)",        "Intranet",  "mlee", 4),
    "10.143.0.103":    ("internal-share",  "Internal file share",                       "Intranet",  "mlee", 3),
    "192.168.231.254": ("inet-dns",        "Internal DNS resolver",                     "Internal",  "tbaker", 4),
    "192.168.231.56":  ("davey-mail",      "Branch office mail relay",                  "Internal",  "rkhan", 2),
    "192.168.231.164": ("morris-mail",     "Branch office mail relay",                  "Internal",  "rkhan", 2),
}
# username, full name, team, role in the SOC app, phone (RESTRICTED), on_call (RESTRICTED)
STAFF = [
    ("tbaker", "Tess Baker", "Network Operations", "asset_owner", "555-0141", "Mon-Wed"),
    ("rkhan", "Rafael Khan", "Messaging & Web", "asset_owner", "555-0177", "Thu-Fri"),
    ("mlee", "Mina Lee", "Collaboration Services", "asset_owner", "555-0102", "Sat-Sun"),
    ("analyst1", "Dana Ortiz", "SOC", "tier1", "555-0123", "Mon-Fri day"),
    ("hunter1", "Sam Whitaker", "SOC", "tier2", "555-0190", "Mon-Fri night"),
    ("manager1", "Lena Yamamoto", "SOC", "manager", "555-0116", "escalation"),
]
KEEP_DATA = ("srcip", "src_ip", "dstip", "dest_ip", "src_port", "srcport", "dest_port", "dstport",
             "proto", "srcuser", "dstuser")


def clean_ip(v):
    return (v or "").split(":")[0]


def summarize(a, n):
    r, d = a["rule"], a.get("data", {}) or {}
    m = r.get("mitre", {}) or {}
    sig = ""
    if isinstance(d.get("alert"), dict):          # Suricata signature, a short public rule name
        sig = d["alert"].get("signature", "")
    agent_ip = a["agent"].get("ip", "")
    host = HOSTS.get(agent_ip, (a.get("predecoder", {}).get("hostname") or agent_ip,))[0]
    return {
        "alert_id": f"AL-{n:06d}",
        "ts": a["@timestamp"][:19] + "Z",
        "host": host,
        "sensor_ip": agent_ip,
        "rule_id": r.get("id", ""),
        "level": int(r.get("level", 0)),
        "description": r.get("description", ""),
        "signature": sig,
        "groups": "|".join(r.get("groups", [])),
        "mitre_ids": "|".join(m.get("id", [])),
        "mitre_tactics": "|".join(m.get("tactic", [])),
        "src_ip": clean_ip(d.get("srcip") or d.get("src_ip")),
        "dst_ip": clean_ip(d.get("dstip") or d.get("dest_ip")),
        "dst_port": str(d.get("dest_port") or d.get("dstport") or ""),
        "proto": d.get("proto", ""),
        "src_user": (d.get("srcuser") or "").strip("<>"),
        "dst_user": (d.get("dstuser") or "").strip("<>"),
        "log_source": a.get("location", "").split("/")[-1],
        "country": (a.get("GeoLocation") or {}).get("country_name", ""),
    }


def main(src_dir, labels):
    OUT.mkdir(exist_ok=True)
    rows = []
    with open(Path(src_dir) / f"{SCENARIO}_wazuh.json") as f:
        for line in f:
            rows.append(json.loads(line))
    rows.sort(key=lambda a: (a["@timestamp"], a.get("id", "")))
    S = [summarize(a, i + 1) for i, a in enumerate(rows)]
    hist = [s for s in S if s["ts"] < SPLIT]
    live = [s for s in S if s["ts"] >= SPLIT]
    cols = list(S[0].keys())
    with gzip.open(OUT / "alerts.csv.gz", "wt", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(hist)
    # live feed for n8n: level >= 5, one event per (rule, host, src) per 10 minutes, with a repeat count
    buckets = collections.OrderedDict()
    for s in live:
        if s["level"] < 5:
            continue
        key = (s["rule_id"], s["host"], s["src_ip"], s["ts"][:15])
        if key in buckets:
            buckets[key]["repeat_count"] += 1
        else:
            buckets[key] = dict(s, repeat_count=1)
    with open(OUT / "live_feed.jsonl", "w") as f:
        for e in buckets.values():
            f.write(json.dumps(e) + "\n")
    # exercise windows (RESTRICTED: the red-team schedule only the SOC manager may see)
    with open(labels) as f, open(OUT / "exercise_windows.csv", "w", newline="") as g:
        w = csv.writer(g)
        w.writerow(["window_id", "phase", "start", "end"])
        k = 0
        for r in csv.DictReader(f):
            if r["scenario"] != SCENARIO:
                continue
            k += 1
            fmt = lambda x: datetime.fromtimestamp(float(x), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            w.writerow([f"EX-{k:02d}", r["attack"], fmt(r["start"]), fmt(r["end"])])
    with open(OUT / "hosts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["host", "ip", "role", "zone", "owner", "criticality"])
        for ip, (h, role, zone, owner, crit) in HOSTS.items():
            w.writerow([h, ip, role, zone, owner, crit])
    with open(OUT / "staff.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["username", "name", "team", "app_role", "phone", "on_call"])
        w.writerows(STAFF)
    print(f"{len(S)} alerts -> history {len(hist)} (alerts.csv.gz), live feed {len(buckets)} events "
          f"from {len(live)} raw alerts (live_feed.jsonl)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE / "ait_ads",
         sys.argv[2] if len(sys.argv) > 2 else HERE / "labels.csv")
