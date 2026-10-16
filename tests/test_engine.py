from datetime import datetime, timedelta

import pandas as pd
import pytest

from sentinel.data_generator import generate
from sentinel.engine import Config, analyze

AML = {"structuring", "minimal_play", "mule_ring", "hedged_betting", "ticket_discounting",
       "account_takeover", "pep_high_value"}


@pytest.fixture(scope="module")
def result():
    players, tx = generate(1500, seed=3)
    return players, analyze(tx, players.drop(columns=["scenario"]))


def test_planted_launderers_are_escalated(result):
    players, res = result
    truth = players.set_index("player_id")["scenario"]
    routes = res.cases.set_index("player_id")["route"]
    bad = truth[truth.isin(AML)].index
    escalated = routes.reindex(bad).fillna("").str.startswith("Escalate")
    assert escalated.mean() >= 0.95


def test_normal_players_rarely_escalated(result):
    players, res = result
    truth = players.set_index("player_id")["scenario"]
    routes = res.cases.set_index("player_id")["route"]
    normal = truth[truth == "normal"].index
    assert routes.reindex(normal).fillna("").str.startswith("Escalate").sum() <= 3


def test_problem_gamblers_go_to_playsmart(result):
    players, res = result
    truth = players.set_index("player_id")["scenario"]
    routes = res.cases.set_index("player_id")["route"]
    pp = routes.reindex(truth[truth == "player_protection"].index).fillna("")
    assert pp.str.startswith("Refer").mean() >= 0.7


def test_every_case_has_name_money_time_and_reason(result):
    _, res = result
    c = res.cases
    assert c["name"].notna().all() and (c["name"] != "").all()
    assert c["top_reason"].str.len().gt(20).all()
    flagged = c[c["route"].str.startswith("Escalate")]
    assert (flagged["suspicious_amount"] > 0).all()
    assert flagged["first_seen"].notna().all() and flagged["last_seen"].notna().all()


def _tx(rows):
    base = datetime(2026, 10, 1, 12)
    return pd.DataFrame([{
        "tx_id": f"T{i}", "timestamp": base + timedelta(hours=h), "player_id": pid, "channel": ch,
        "venue": "Casino Rama", "tx_type": t, "amount": amt, "payment_method": pm,
        "funding_source": "CASH", "game": "Blackjack", "stake_at_risk_pct": 0.9,
        "device_id": "", "ip_address": "", "ip_region": ""} for i, (pid, h, ch, t, amt, pm) in enumerate(rows)])


def _players(ids):
    return pd.DataFrame([{"player_id": p, "name": p, "age": 40, "occupation": "Teacher",
                          "est_annual_income": 80000, "city": "Toronto", "account_created": "2020-01-01",
                          "kyc_status": "Verified", "pep": False, "primary_channel": "Casino"} for p in ids])


def test_structuring_just_under_threshold():
    rows = []
    for day in range(3):
        for k, amt in enumerate([3200, 3100, 3300]):
            rows.append(("S1", day * 48 + k * 2, "Casino", "CASH_BUY_IN", amt, "Cash"))
    res = analyze(_tx(rows), _players(["S1"]), Config())
    rules = set(res.alerts["rule_id"])
    assert "STRUCTURING" in rules
    assert "LCTR" not in rules  # each 24h window stays under $10k


def test_lctr_on_10k_cash_in_24h():
    rows = [("L1", 0, "Casino", "CASH_BUY_IN", 6000, "Cash"), ("L1", 5, "Casino", "CASH_BUY_IN", 4500, "Cash")]
    res = analyze(_tx(rows), _players(["L1"]), Config())
    a = res.alerts[res.alerts["rule_id"] == "LCTR"]
    assert len(a) == 1 and a.iloc[0]["amount"] == 10500
