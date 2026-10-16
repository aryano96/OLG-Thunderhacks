"""Score the engine against the planted ground truth in the synthetic data.

    python -m sentinel.evaluate
"""

import time

import pandas as pd

from .data_generator import generate
from .engine import analyze

AML_SCENARIOS = {"structuring", "minimal_play", "mule_ring", "hedged_betting",
                 "ticket_discounting", "account_takeover", "pep_high_value"}


def main(n_players=5000, seed=7):
    players, tx = generate(n_players, seed)
    t0 = time.time()
    res = analyze(tx, players.drop(columns=["scenario"]))
    secs = time.time() - t0
    truth = players.set_index("player_id")["scenario"]
    cases = res.cases.set_index("player_id")
    escalated = set(cases.index[cases["route"].str.startswith("Escalate")])
    reviewed = set(cases.index[cases["route"].str.startswith("Enhanced")])
    bad = set(truth.index[truth.isin(AML_SCENARIOS)])
    tp = len(escalated & bad)
    print(f"Analysed {len(tx):,} transactions for {len(players):,} players in {secs:.1f}s")
    print(f"Alerts: {len(res.alerts):,}   Cases: {len(cases):,}   "
          f"Escalated to AML: {len(escalated)}   Analyst review: {len(reviewed)}")
    print(f"Planted launderers escalated: {tp}/{len(bad)} (recall {tp / max(1, len(bad)):.0%})")
    print(f"Escalations that are planted launderers: {tp / max(1, len(escalated)):.0%}")
    print()
    rows = []
    for sc, ids in truth.groupby(truth).groups.items():
        ids = set(ids)
        routes = cases.reindex(list(ids))["route"].fillna("Not flagged")
        rows.append({"scenario": sc, "players": len(ids),
                     "escalated_aml": int(routes.str.startswith("Escalate").sum()),
                     "analyst_review": int(routes.str.startswith("Enhanced").sum()),
                     "playsmart": int(routes.str.startswith("Refer").sum()),
                     "filing_only": int(routes.str.startswith("File").sum()),
                     "not_flagged": int((routes == "Not flagged").sum())})
    print(pd.DataFrame(rows).sort_values("players").to_string(index=False))


if __name__ == "__main__":
    main()
