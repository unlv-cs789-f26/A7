# Data used in A6

## Source
**AIT Alert Data Set (AIT-ADS)**, version 1 — Max Landauer, Florian Skopik, Markus Wurzenberger
(AIT Austrian Institute of Technology), Zenodo, 2023. doi:10.5281/zenodo.8263181.
License: Creative Commons Attribution 4.0 International (CC BY 4.0).

The data set contains alerts raised by Wazuh, Suricata and AMiner while a simulated enterprise network ran normal
user activity and a scripted multi-step attack (eight scenarios). A6 uses one scenario, **russellmitchell**
(Wazuh alerts, which include the Suricata alerts forwarded to Wazuh), 2022-01-21 00:00 to 2022-01-24 23:59 UTC.

## Changes made for A6 (tools/prepare_data.py)
* Each alert is reduced to summary fields: time, sensor host, rule id / level / description / groups, ATT&CK
  technique ids and tactics, source and destination IP and port, user names, log source, country.
  The raw log line and all payload fields are dropped.
* Alert ids `AL-nnnnnn` are added, in time order.
* `alerts.csv.gz` = history before 2022-01-24 12:00 UTC (37,808 alerts).
  `live_feed.jsonl` = alerts of level ≥ 5 after that time, deduplicated per rule × host × source × 10 minutes
  (178 events, with a `repeat_count`).
* `exercise_windows.csv` = the data set's `labels.csv` (attack phases with start/end) for this scenario, presented
  in the lab as the red-team exercise schedule that only the SOC manager may see.
* `hosts.csv` (owner, criticality, role, zone), `staff.csv` (names, roles, phone numbers, on-call days) and the
  role permissions are **invented** for the lab; they are not part of AIT-ADS.
* `attack_techniques.csv` lists the ATT&CK techniques that occur in the alerts, with names and tactics from MITRE ATT&CK.
