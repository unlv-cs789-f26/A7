"""Data loaders for A6 (summary-level tables derived from the public AIT Alert Data Set)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
SPLIT = "2022-01-24T12:00:00Z"          # history (in the KG) < SPLIT <= live feed (A7, n8n)
ZONES = ["Perimeter", "DMZ", "Intranet", "Internal"]


def load_alerts(path: str | Path | None = None) -> pd.DataFrame:
    """History alerts (Jan 21 00:00 - Jan 24 12:00 UTC), one row per Wazuh alert, summary fields only."""
    A = pd.read_csv(path or DATA / "alerts.csv.gz", dtype=str, keep_default_na=False)
    A["level"] = A["level"].astype(int)
    A["time"] = pd.to_datetime(A["ts"], utc=True)
    A["hour"] = A["ts"].str[:13] + ":00Z"
    A["day"] = A["ts"].str[:10]
    return A


def load_hosts() -> pd.DataFrame:
    H = pd.read_csv(DATA / "hosts.csv", dtype=str, keep_default_na=False)
    H["criticality"] = H["criticality"].astype(int)
    return H


def load_staff() -> pd.DataFrame:
    return pd.read_csv(DATA / "staff.csv", dtype=str, keep_default_na=False)


def load_windows() -> pd.DataFrame:
    """Exercise windows (from AIT-ADS labels.csv). RESTRICTED: only the SOC manager may see them."""
    return pd.read_csv(DATA / "exercise_windows.csv", dtype=str, keep_default_na=False)


def load_feed() -> list[dict]:
    """The live feed for A7 (n8n): 178 deduplicated level>=5 events from Jan 24 12:00-24:00 UTC."""
    return [json.loads(l) for l in open(DATA / "live_feed.jsonl")]


def load_all():
    """A, hosts, staff, windows -- the four tables the KG is built from."""
    return load_alerts(), load_hosts(), load_staff(), load_windows()


def alert_groups(A: pd.DataFrame) -> pd.DataFrame:
    """Reference grouping: one row per (host, rule_id, src_ip, hour). This is what becomes an
    ALERT_GROUP node. Used by the checks; you write your own in B.1."""
    g = (A.assign(src=A["src_ip"].replace("", "-"), dst=A["dst_ip"].replace("", "-"))
           .groupby(["host", "rule_id", "src", "hour"], sort=True)
           .agg(count=("alert_id", "size"), max_level=("level", "max"),
                first_seen=("ts", "min"), last_seen=("ts", "max"),
                description=("description", "first"), dst_ips=("dst", lambda s: sorted(set(s) - {"-"})),
                users=("src_user", lambda s: sorted(set(s) - {""})))
           .reset_index())
    g["group_id"] = "AG|" + g["host"] + "|" + g["rule_id"] + "|" + g["src"] + "|" + g["hour"]
    return g


def overview(A=None):
    """Print a one-screen summary of the dataset."""
    A = load_alerts() if A is None else A
    print(f"{len(A):,} alerts  {A.ts.min()} .. {A.ts.max()}  hosts={A.host.nunique()}  rules={A.rule_id.nunique()}  "
          f"with ATT&CK id={(A.mitre_ids != '').sum():,}")
    return A.pivot_table(index="host", columns="day", values="alert_id", aggfunc="count", fill_value=0)


CITATION = ("M. Landauer, F. Skopik, M. Wurzenberger, \"AIT Alert Data Set\", Zenodo, 2023, "
            "doi:10.5281/zenodo.8263181 (CC BY 4.0). Scenario russellmitchell, Wazuh alerts, reduced to summary fields.")
