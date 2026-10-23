"""soc6 -- the fixed harness for CS 789 Assignments 6 and 7 (SOC Copilot).

    import soc6 as H
    A, hosts, staff, windows = H.load_all()       # data (AIT Alert Data Set, summary fields)
    H.check_ontology()                            # A6 Part A
    H.check_kg(kg); H.score_extraction(...)       # A6 Part B
    H.check_context(kg_context); H.eval_bot(...)  # A6 Part C
    H.run_dialogue(chat_turn, policy="window-1")  # A6 Part D
    H.start_platform(...); H.acceptance("a6")     # A6 Part E
    H.start_n8n(); H.replay_feed()                # A7 Part A (n8n)
    H.heretic_evaluate(...)                       # A7 Part B (uncensored model)
    H.acceptance("a7")                            # A7 Part C

Everything that must be identical for every student lives here (data, scoring, the web shell,
the n8n and Heretic helpers). Everything you BUILD lives in your notebook. Do not edit this package.
"""
from .data import (HERE, DATA, SPLIT, CITATION, load_alerts, load_hosts, load_staff, load_windows,
                   load_feed, load_all, alert_groups, overview)
from .onto import (schema_from_ontology, check_ontology, show_ontology, validate_shapes, upper_snake)
from .kg import (schema, restricted, index, save_kg, load_kg, load_reference_kg, check_kg, kg_to_rdf,
                 link_entities, parse_time_window, node_lines, alert_summary, windows_lines, show_kg,
                 kg_stats, is_internal)

import importlib as _il

for _m in ("llm", "evals", "server", "carry", "n8n", "heretic"):   # n8n / heretic exist only in the A7 repo
    try:
        _mod = _il.import_module(f".{_m}", __name__)
        globals().update({k: getattr(_mod, k) for k in getattr(_mod, "__all__", [])})
    except ModuleNotFoundError as _e:
        if _e.name not in (f"{__name__}.{_m}",):
            print(f"soc6: {_m} helpers unavailable until Part 0 installs the packages ({_e.name})")
