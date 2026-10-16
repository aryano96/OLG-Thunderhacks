"""Detection engine: turns a raw transaction stream into explained alerts and cases.

Two layers run over every player:

1. Typology rules. Each rule encodes a known way money is laundered through
   gaming (structuring, minimal play, hedged betting, mule rings, ticket
   discounting...) or a mandatory FINTRAC report (LCTR / CDR). Every alert
   carries the exact transactions, dollar amount, time window and a
   plain-English reason, so an investigator can verify it.
2. Unsupervised anomaly detection. An Isolation Forest scores each player's
   behaviour profile against the whole population to surface patterns no rule
   was written for, and explains itself with the features that were most
   unusual.

Alerts are then rolled up into one case per player with a composite risk
score and a recommended routing (AML investigation, mandatory filing, or
player protection).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

MONEY_IN = {"DEPOSIT", "CASH_BUY_IN"}
MONEY_OUT = {"WITHDRAWAL", "CASH_OUT", "PRIZE_CLAIM"}
REPORT_THRESHOLD = 10_000  # FINTRAC LCTR / CDR, 24-hour rule

CATEGORY_AML = "AML"
CATEGORY_REG = "Regulatory"
CATEGORY_ANOMALY = "Anomaly"
CATEGORY_PP = "Player Protection"


@dataclass
class Config:
    structuring_floor: float = 7_500        # 24h money-in between this and $10k = "just under"
    minimal_play_wager_ratio: float = 0.30  # wagered / money-in below this...
    minimal_play_out_ratio: float = 0.70    # ...while cashing out at least this share
    minimal_play_min_in: float = 3_000
    hedge_risk_pct: float = 0.10            # share of stake actually exposed
    hedge_min_total: float = 5_000
    ring_min_accounts: int = 3
    funding_sources_max: int = 2
    affordability_ratio: float = 4.0        # annualised money-in / stated income
    affordability_min_in: float = 5_000
    spike_multiplier: float = 8.0
    spike_min_amount: float = 3_000
    prize_min_amount: float = 1_000
    prize_min_count: int = 3
    pep_min_in: float = 10_000
    anomaly_contamination: float = 0.01


@dataclass
class Alert:
    player_id: str
    rule_id: str
    rule: str
    category: str
    severity: float
    amount: float
    start: pd.Timestamp
    end: pd.Timestamp
    reason: str
    tx_ids: list = field(default_factory=list)
    linked_players: list = field(default_factory=list)


RULES = {
    "STRUCTURING": ("Structuring / smurfing", CATEGORY_AML,
                    "Money-in repeatedly split so each 24h total lands just under the $10,000 FINTRAC threshold."),
    "MINIMAL_PLAY": ("Minimal play / pass-through", CATEGORY_AML,
                     "Large funds in, very little wagered, most of it cashed out as 'winnings' (chip walking)."),
    "HEDGED_BETTING": ("Hedged / low-risk betting", CATEGORY_AML,
                       "Big stakes where almost nothing is at risk (covering both sides of an even-money bet)."),
    "SHARED_DEVICE": ("Shared device / IP ring", CATEGORY_AML,
                      "Several accounts operated from the same device or IP: mule network or multi-accounting."),
    "FUNDING_VELOCITY": ("Many funding sources", CATEGORY_AML,
                         "Deposits from more cards / prepaid instruments than a normal player uses."),
    "AFFORDABILITY": ("Affordability mismatch", CATEGORY_AML,
                      "Money moved is far beyond what the player's stated occupation and income support."),
    "BASELINE_SPIKE": ("Sudden spike vs history", CATEGORY_AML,
                       "Last 7 days of money-in is many times the player's own historical weekly level."),
    "NEW_ACCOUNT_HIGH_VALUE": ("New account, high value", CATEGORY_AML,
                               "Account under 30 days old already moving thousands."),
    "GEO_MASKING": ("Location masking", CATEGORY_AML,
                    "OLG.ca activity from VPN / proxy / out-of-province IPs (play requires being in Ontario)."),
    "PRIZE_FREQUENCY": ("Repeated large prize claims", CATEGORY_AML,
                        "Improbably many big lottery claims: buying winning tickets from real winners for cash."),
    "PEP_HIGH_VALUE": ("PEP high-value activity", CATEGORY_AML,
                       "Politically exposed person moving large sums: enhanced due diligence required."),
    "ML_ANOMALY": ("Behavioural anomaly (ML)", CATEGORY_ANOMALY,
                   "Isolation Forest finds this player's overall profile unlike the rest of the population."),
    "LCTR": ("Large cash transaction (LCTR)", CATEGORY_REG,
             "$10,000+ cash received within 24 hours: mandatory Large Cash Transaction Report."),
    "CDR": ("Casino disbursement (CDR)", CATEGORY_REG,
            "$10,000+ paid out within 24 hours: mandatory Casino Disbursement Report."),
    "LOSS_CHASING": ("Loss chasing / late-night play", CATEGORY_PP,
                     "Escalating repeat deposits in the middle of the night: gambling harm, not laundering."),
}


def money(x: float) -> str:
    return f"${x:,.0f}"


def _alert(pid, rule_id, severity, amount, start, end, reason, tx_ids=(), linked=()):
    name, cat, _ = RULES[rule_id]
    return Alert(pid, rule_id, name, cat, round(float(min(99, severity)), 1), round(float(amount), 2),
                 pd.Timestamp(start), pd.Timestamp(end), reason, list(tx_ids), list(linked))


# --------------------------------------------------------------------- prep
def prepare(tx: pd.DataFrame) -> pd.DataFrame:
    tx = tx.copy()
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce").fillna(0.0)
    for col in ["payment_method", "funding_source", "game", "device_id", "ip_address", "ip_region", "venue"]:
        if col not in tx:
            tx[col] = ""
        tx[col] = tx[col].fillna("").astype(str)
    if "stake_at_risk_pct" not in tx:
        tx["stake_at_risk_pct"] = np.nan
    tx["stake_at_risk_pct"] = pd.to_numeric(tx["stake_at_risk_pct"], errors="coerce")
    tx["direction"] = np.where(tx["tx_type"].isin(MONEY_IN), "IN",
                               np.where(tx["tx_type"].isin(MONEY_OUT), "OUT", "PLAY"))
    return tx.sort_values(["player_id", "timestamp"]).reset_index(drop=True)


def _episodes(df: pd.DataFrame, cond, hours=24):
    """Slide a 24h window over each player's transactions and merge overlapping hits.

    `cond(amounts)` gets the numpy array of amounts in the window. Returns a
    list of (player_id, positional-index-array) episodes.
    """
    out = []
    win = np.timedelta64(int(hours * 3600), "s")
    for pid, g in df.groupby("player_id", sort=False):
        ts = g["timestamp"].values
        amt = g["amount"].values
        idx = g.index.values
        i = 0
        current = None
        for j in range(len(g)):
            while ts[j] - ts[i] > win:
                i += 1
            if cond(amt[i:j + 1]):
                if current is not None and i <= current[1]:
                    current = (current[0], j)
                else:
                    if current is not None:
                        out.append((pid, idx[current[0]:current[1] + 1]))
                    current = (i, j)
        if current is not None:
            out.append((pid, idx[current[0]:current[1] + 1]))
    return out


# ---------------------------------------------------------------- features
def player_features(tx: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    as_of = tx["timestamp"].max()
    period_days = max(1.0, (as_of - tx["timestamp"].min()).total_seconds() / 86400)
    g = tx.groupby("player_id")
    f = pd.DataFrame(index=players["player_id"].values)
    f.index.name = "player_id"

    def by(mask, col="amount", fn="sum"):
        return tx[mask].groupby("player_id")[col].agg(fn)

    f["money_in"] = by(tx["direction"] == "IN")
    f["money_out"] = by(tx["direction"] == "OUT")
    f["wagered"] = by(tx["tx_type"] == "WAGER")
    f["n_tx"] = g.size()
    f["max_in"] = by(tx["direction"] == "IN", fn="max")
    recent = tx["timestamp"] > as_of - timedelta(days=7)
    f["in_last7"] = by((tx["direction"] == "IN") & recent)
    f["in_before7"] = by((tx["direction"] == "IN") & ~recent)
    online = tx["channel"] == "OLG.ca"
    f["n_devices"] = tx[online & (tx["device_id"] != "")].groupby("player_id")["device_id"].nunique()
    f["n_ips"] = tx[online & (tx["ip_address"] != "")].groupby("player_id")["ip_address"].nunique()
    dep = (tx["tx_type"] == "DEPOSIT") & (tx["funding_source"] != "") & (tx["funding_source"] != "CASH")
    f["n_funding"] = tx[dep].groupby("player_id")["funding_source"].nunique()
    night = tx["timestamp"].dt.hour < 5
    f["night_share"] = (tx[night].groupby("player_id").size() / f["n_tx"])
    f["non_ontario_share"] = (tx[online & (tx["ip_region"] != "") & (tx["ip_region"] != "Ontario")]
                              .groupby("player_id").size() / tx[online].groupby("player_id").size())
    big_prize = (tx["tx_type"] == "PRIZE_CLAIM") & (tx["amount"] >= 1000)
    f["n_big_prizes"] = tx[big_prize].groupby("player_id").size()
    w = tx[(tx["tx_type"] == "WAGER") & tx["stake_at_risk_pct"].notna()]
    f["avg_stake_at_risk"] = (w["amount"] * w["stake_at_risk_pct"]).groupby(w["player_id"]).sum() / \
        w.groupby("player_id")["amount"].sum()
    f = f.fillna({c: 0 for c in f.columns if c != "avg_stake_at_risk"})
    f["avg_stake_at_risk"] = f["avg_stake_at_risk"].fillna(1.0)
    f["wager_ratio"] = f["wagered"] / f["money_in"].where(f["money_in"] > 0)
    f["out_ratio"] = f["money_out"] / f["money_in"].where(f["money_in"] > 0)
    p = players.set_index("player_id")
    income = pd.to_numeric(p["est_annual_income"], errors="coerce").reindex(f.index).fillna(0)
    f["income_ratio"] = (f["money_in"] * 365 / period_days) / income.where(income > 0)
    created = pd.to_datetime(p["account_created"], errors="coerce").reindex(f.index)
    f["account_age_days"] = (as_of - created).dt.days.fillna(9999)
    return f


# ------------------------------------------------------------------- rules
def _rule_alerts(tx: pd.DataFrame, players: pd.DataFrame, feats: pd.DataFrame, cfg: Config):
    alerts: list[Alert] = []
    pmeta = players.set_index("player_id")
    as_of = tx["timestamp"].max()
    period_days = max(1, round((as_of - tx["timestamp"].min()).total_seconds() / 86400))
    money_in = tx[tx["direction"] == "IN"]
    money_out = tx[tx["direction"] == "OUT"]

    # Mandatory LCTR: cash received >= $10k in 24h
    cash_in = money_in[money_in["payment_method"].str.lower() == "cash"]
    for pid, ix in _episodes(cash_in, lambda a: a.sum() >= REPORT_THRESHOLD):
        e = tx.loc[ix]
        alerts.append(_alert(pid, "LCTR", 35, e["amount"].sum(), e["timestamp"].min(), e["timestamp"].max(),
                             f"Received {money(e['amount'].sum())} in cash across {len(e)} transaction(s) "
                             f"between {e['timestamp'].min():%b %d %H:%M} and {e['timestamp'].max():%b %d %H:%M}, "
                             f"which meets FINTRAC's $10,000 / 24-hour Large Cash Transaction Report threshold.",
                             e["tx_id"]))

    # Mandatory CDR: disbursements >= $10k in 24h
    for pid, ix in _episodes(money_out, lambda a: a.sum() >= REPORT_THRESHOLD):
        e = tx.loc[ix]
        alerts.append(_alert(pid, "CDR", 30, e["amount"].sum(), e["timestamp"].min(), e["timestamp"].max(),
                             f"Paid out {money(e['amount'].sum())} across {len(e)} disbursement(s) within 24 hours "
                             f"(cheques / EFT / cash-outs). A Casino Disbursement Report is mandatory.",
                             e["tx_id"]))

    # Structuring: 24h totals that land just under the threshold, split into pieces
    def just_under(a):
        s = a.sum()
        return len(a) >= 2 and cfg.structuring_floor <= s < REPORT_THRESHOLD and a.max() < REPORT_THRESHOLD
    eps = {}
    for pid, ix in _episodes(money_in, just_under):
        eps.setdefault(pid, []).append(ix)
    for pid, lst in eps.items():
        ix = np.concatenate(lst)
        e = tx.loc[ix]
        totals = [tx.loc[i, "amount"].sum() for i in lst]
        venues = e["venue"].replace("", np.nan).dropna().nunique()
        n = len(lst)
        alerts.append(_alert(
            pid, "STRUCTURING", 48 + 14 * n, e["amount"].sum(), e["timestamp"].min(), e["timestamp"].max(),
            f"{n} separate 24-hour window(s) where money-in totalled {money(min(totals))}–{money(max(totals))}, "
            f"each split across {min(len(i) for i in lst)}–{max(len(i) for i in lst)} smaller transactions"
            + (f" at {venues} different locations" if venues > 1 else "")
            + ". Every window stays just under the $10,000 reporting threshold, the classic structuring pattern.",
            e["tx_id"]))

    # Minimal play / pass-through: per visit, money comes in, barely gets played, and leaves
    visits = {}
    big_in = tx[(tx["direction"] == "IN") & (tx["amount"] >= 1_000)]
    outs_all = tx[tx["direction"] == "OUT"]
    wag_all = tx[tx["tx_type"] == "WAGER"]
    two_days = np.timedelta64(48, "h")
    for pid, ins in big_in.groupby("player_id", sort=False):
        outs = outs_all[outs_all["player_id"] == pid]
        if outs.empty:
            continue
        o_ts, o_amt, o_idx = outs["timestamp"].values, outs["amount"].values, outs.index.values
        wag = wag_all[wag_all["player_id"] == pid]
        w_ts, w_cum = wag["timestamp"].values, np.concatenate([[0.0], wag["amount"].cumsum().values])
        for i, t_in, amt in zip(ins.index.values, ins["timestamp"].values, ins["amount"].values):
            k = np.searchsorted(o_ts, t_in, side="right")
            if k >= len(o_ts) or o_ts[k] - t_in > two_days:
                continue
            played = w_cum[np.searchsorted(w_ts, o_ts[k], side="right")] - w_cum[np.searchsorted(w_ts, t_in, side="left")]
            if played < cfg.minimal_play_wager_ratio * amt and o_amt[k] >= cfg.minimal_play_out_ratio * amt:
                hours = (o_ts[k] - t_in) / np.timedelta64(1, "h")
                visits.setdefault(pid, []).append((i, o_idx[k], amt, o_amt[k], played, hours))
    for pid, v in visits.items():
        total_in = sum(x[2] for x in v)
        if total_in < cfg.minimal_play_min_in:
            continue
        total_out, played = sum(x[3] for x in v), sum(x[4] for x in v)
        ids = [i for x in v for i in (x[0], x[1])]
        e = tx.loc[ids]
        ratio = played / total_in
        hours = np.median([x[5] for x in v])
        alerts.append(_alert(
            pid, "MINIMAL_PLAY", 66 + 6 * len(v) + 10 * (1 - ratio / cfg.minimal_play_wager_ratio), total_in,
            e["timestamp"].min(), e["timestamp"].max(),
            f"{len(v)} visit(s) where {money(total_in)} came in and {money(total_out)} went back out "
            f"({total_out / total_in:.0%}) within a median of {hours:.1f} hours, with only {money(played)} "
            f"({ratio:.0%}) actually wagered. The funds barely touched a game before leaving as 'winnings' "
            f"(cheque / EFT), which is how gaming is used to make dirty money look clean.",
            e["tx_id"]))

    # Hedged betting
    hw = tx[(tx["tx_type"] == "WAGER") & (tx["stake_at_risk_pct"] < cfg.hedge_risk_pct) & (tx["amount"] >= 500)]
    for pid, e in hw.groupby("player_id"):
        total = e["amount"].sum()
        if total < cfg.hedge_min_total:
            continue
        risk = (e["amount"] * e["stake_at_risk_pct"]).sum() / total
        games = ", ".join(sorted(e["game"].unique()))
        alerts.append(_alert(
            pid, "HEDGED_BETTING", 78 + min(15, len(e) * 2), total, e["timestamp"].min(), e["timestamp"].max(),
            f"{len(e)} {games} session(s) totalling {money(total)} where only {risk:.0%} of the stake was actually "
            f"at risk (bets covering both outcomes, such as red and black). Money cycles through play with almost no "
            f"chance of loss.", e["tx_id"]))

    # Shared device / IP rings
    online = tx[(tx["channel"] == "OLG.ca")]
    for key, label in [("device_id", "device"), ("ip_address", "IP address")]:
        sub = online[online[key] != ""]
        users = sub.groupby(key)["player_id"].unique()
        for ident, pids in users.items():
            if len(pids) < cfg.ring_min_accounts:
                continue
            ring = sub[sub[key] == ident]
            for pid in pids:
                others = [o for o in pids if o != pid]
                names = ", ".join(pmeta.loc[o, "name"] if o in pmeta.index else o for o in others[:6])
                mine = ring[ring["player_id"] == pid]
                ring_in = ring[ring["direction"] == "IN"]["amount"].sum()
                alerts.append(_alert(
                    pid, "SHARED_DEVICE", 58 + 6 * (len(pids) - cfg.ring_min_accounts) + (8 if ring_in > 25_000 else 0),
                    mine[mine["direction"] == "IN"]["amount"].sum(), mine["timestamp"].min(), mine["timestamp"].max(),
                    f"Uses the same {label} ({ident}) as {len(others)} other account(s): {names}. Together the ring "
                    f"deposited {money(ring_in)}. Several people on one {label} points to money-mule accounts "
                    f"controlled by one operator.", mine["tx_id"], others))

    # Funding source velocity
    dep = tx[(tx["tx_type"] == "DEPOSIT") & (tx["funding_source"] != "") & (tx["funding_source"] != "CASH")]
    for pid, e in dep.groupby("player_id"):
        srcs = e["funding_source"].unique()
        prepaid = sum(s.startswith("PREPAID") for s in srcs)
        if len(srcs) <= cfg.funding_sources_max and prepaid < 2:
            continue
        alerts.append(_alert(
            pid, "FUNDING_VELOCITY", 48 + 6 * (len(srcs) - cfg.funding_sources_max) + 6 * prepaid, e["amount"].sum(),
            e["timestamp"].min(), e["timestamp"].max(),
            f"Funded the account from {len(srcs)} different cards/instruments ({', '.join(srcs[:6])})"
            + (f", including {prepaid} prepaid card(s)" if prepaid else "")
            + ". Rotating third-party or prepaid cards is used to layer funds from many sources.",
            e["tx_id"]))

    # Affordability
    af = feats[(feats["money_in"] >= cfg.affordability_min_in) & (feats["income_ratio"] > cfg.affordability_ratio)]
    for pid, r in af.iterrows():
        meta = pmeta.loc[pid]
        e = tx[(tx["player_id"] == pid) & (tx["direction"] == "IN")]
        alerts.append(_alert(
            pid, "AFFORDABILITY", 30 + 10 * np.log2(max(1.0, r["income_ratio"])), r["money_in"],
            e["timestamp"].min(), e["timestamp"].max(),
            f"Moved {money(r['money_in'])} in {period_days} days. Annualised that is {r['income_ratio']:.1f}× the "
            f"stated income of {money(meta['est_annual_income'])} for a {meta['occupation'].lower()}, "
            f"age {meta['age']}. The source of these funds is unexplained.", e["tx_id"]))

    # Baseline spike & new accounts
    prior_days = max(1, period_days - 7)
    for pid, r in feats[feats["in_last7"] >= cfg.spike_min_amount].iterrows():
        e = tx[(tx["player_id"] == pid) & (tx["direction"] == "IN") & (tx["timestamp"] > as_of - timedelta(days=7))]
        if r["account_age_days"] < 30:
            continue
        weekly = r["in_before7"] / prior_days * 7
        if weekly > 0 and r["in_last7"] / weekly >= cfg.spike_multiplier:
            alerts.append(_alert(
                pid, "BASELINE_SPIKE", 62 + min(20, r["in_last7"] / weekly / 5), r["in_last7"],
                e["timestamp"].min(), e["timestamp"].max(),
                f"Deposited {money(r['in_last7'])} in the last 7 days against a personal average of "
                f"{money(weekly)} per week, a {r['in_last7'] / weekly:.0f}× jump. Sudden changes like this follow "
                f"account takeovers or an account being rented out to move money.", e["tx_id"]))
    for pid, r in feats[(feats["account_age_days"] < 30) & (feats["money_in"] >= cfg.affordability_min_in)].iterrows():
        e = tx[(tx["player_id"] == pid) & (tx["direction"] == "IN")]
        alerts.append(_alert(
            pid, "NEW_ACCOUNT_HIGH_VALUE", 55, r["money_in"], e["timestamp"].min(), e["timestamp"].max(),
            f"Account opened {int(r['account_age_days'])} days ago and has already moved {money(r['money_in'])}.",
            e["tx_id"]))

    # Location masking
    geo = online[(online["ip_region"] != "") & (online["ip_region"] != "Ontario")]
    for pid, e in geo.groupby("player_id"):
        regions = ", ".join(sorted(e["ip_region"].unique()))
        alerts.append(_alert(
            pid, "GEO_MASKING", 45 + min(20, len(e)), e[e["direction"] != "PLAY"]["amount"].sum(),
            e["timestamp"].min(), e["timestamp"].max(),
            f"{len(e)} OLG.ca transaction(s) from {regions} IP addresses. OLG.ca requires players to be physically in "
            f"Ontario, and hiding location is a common sign of account takeover or offshore control.", e["tx_id"]))

    # Prize claim frequency
    pc = tx[(tx["tx_type"] == "PRIZE_CLAIM") & (tx["amount"] >= cfg.prize_min_amount)]
    for pid, e in pc.groupby("player_id"):
        if len(e) < cfg.prize_min_count:
            continue
        alerts.append(_alert(
            pid, "PRIZE_FREQUENCY", 68 + 4 * (len(e) - cfg.prize_min_count), e["amount"].sum(),
            e["timestamp"].min(), e["timestamp"].max(),
            f"Claimed {len(e)} lottery prizes of {money(cfg.prize_min_amount)}+ totalling {money(e['amount'].sum())} "
            f"in {period_days} days. Real players almost never win this often. This matches ticket discounting: buying "
            f"winning tickets from actual winners with cash, then claiming them for a clean OLG cheque.", e["tx_id"]))

    # PEP
    for pid in players.loc[players["pep"].astype(str).str.lower().isin(["true", "1", "yes"]), "player_id"]:
        if pid not in feats.index or feats.loc[pid, "money_in"] < cfg.pep_min_in:
            continue
        e = tx[(tx["player_id"] == pid) & (tx["direction"] == "IN")]
        alerts.append(_alert(
            pid, "PEP_HIGH_VALUE", 66, e["amount"].sum(), e["timestamp"].min(), e["timestamp"].max(),
            f"Politically Exposed Person ({pmeta.loc[pid, 'occupation']}) brought in {money(e['amount'].sum())}. "
            f"PCMLTFA requires enhanced due diligence and a source-of-funds check.", e["tx_id"]))

    # Player protection: escalating repeat deposits at night
    nd = money_in[(money_in["channel"] == "OLG.ca") & (money_in["timestamp"].dt.hour < 5)]
    for pid, e in nd.groupby("player_id"):
        if len(e) < 5:
            continue
        gaps = e["timestamp"].diff().dt.total_seconds().fillna(1e9) < 3600
        escal = (e["amount"] > e["amount"].shift(1) * 1.1) & gaps
        if escal.sum() < 3:
            continue
        nights = e["timestamp"].dt.date.nunique()
        alerts.append(_alert(
            pid, "LOSS_CHASING", 55 + min(30, escal.sum() * 2), e["amount"].sum(),
            e["timestamp"].min(), e["timestamp"].max(),
            f"{len(e)} deposits between midnight and 5 AM over {nights} nights, {int(escal.sum())} of them bigger "
            f"than the deposit just before. That is loss-chasing behaviour: a gambling-harm concern for PlaySmart, "
            f"not evidence of laundering.", e["tx_id"]))
    return alerts


# ---------------------------------------------------------------- anomaly
ANOMALY_FEATURES = {
    "money_in": ("total money in", True),
    "money_out": ("total money out", True),
    "max_in": ("largest single deposit / buy-in", True),
    "wager_ratio": ("wagered ÷ money in", False),
    "out_ratio": ("money out ÷ money in", False),
    "n_devices": ("distinct devices", False),
    "n_funding": ("distinct funding sources", False),
    "night_share": ("share of activity 12–5 AM", False),
    "non_ontario_share": ("share of activity outside Ontario", False),
    "n_big_prizes": ("prize claims over $1,000", False),
    "avg_stake_at_risk": ("share of stake at risk", False),
    "income_ratio": ("money in vs. income", False),
}


def anomaly_scores(feats: pd.DataFrame, cfg: Config):
    X = feats[list(ANOMALY_FEATURES)].copy()
    X["wager_ratio"] = X["wager_ratio"].fillna(X["wager_ratio"].median()).clip(0, 20)
    X["out_ratio"] = X["out_ratio"].fillna(0).clip(0, 5)
    X["income_ratio"] = X["income_ratio"].fillna(0).clip(0, 50)
    for c in ["money_in", "money_out", "max_in", "income_ratio"]:
        X[c] = np.log1p(X[c])
    X = X.fillna(0)
    model = IsolationForest(n_estimators=250, contamination=cfg.anomaly_contamination, random_state=0)
    model.fit(X.values)
    raw = -model.score_samples(X.values)
    pct = pd.Series(raw, index=X.index).rank(pct=True) * 100
    flagged = pd.Series(model.predict(X.values) == -1, index=X.index)
    med = X.median()
    mad = (X - med).abs().median() * 1.4826 + X.std() * 0.05 + 1e-9
    z = (X - med) / mad
    return pct.round(1), flagged, z


def _anomaly_alerts(feats, cfg):
    pct, flagged, z = anomaly_scores(feats, cfg)
    alerts = []
    for pid in flagged[flagged].index:
        top = z.loc[pid].abs().sort_values(ascending=False).head(3)
        parts = []
        for col in top.index:
            label, is_money = ANOMALY_FEATURES[col]
            v, typical = feats.loc[pid, col], feats[col].median()
            if is_money:
                parts.append(f"{label} {money(v)} (typical {money(typical)})")
            elif col in ("wager_ratio", "out_ratio", "income_ratio"):
                parts.append(f"{label} {v:.2f} (typical {typical:.2f})")
            elif col in ("night_share", "non_ontario_share", "avg_stake_at_risk"):
                parts.append(f"{label} {v:.0%} (typical {typical:.0%})")
            else:
                parts.append(f"{label} {v:.0f} (typical {typical:.0f})")
        alerts.append(_alert(
            pid, "ML_ANOMALY", 40 + 2 * max(0, pct[pid] - 90), feats.loc[pid, "money_in"],
            pd.NaT, pd.NaT,
            f"The anomaly model ranks this player more unusual than {pct[pid]:.1f}% of all players. Most unusual: "
            + "; ".join(parts) + "."))
    return alerts, pct


# ------------------------------------------------------------------ cases
# Rules that match a specific laundering method. Profile signals (affordability,
# spikes, new accounts, ML anomaly) raise the score but on their own only earn a
# review, because a big-spending or struggling player can trip them too.
TYPOLOGY_RULES = {"STRUCTURING", "MINIMAL_PLAY", "HEDGED_BETTING", "SHARED_DEVICE", "FUNDING_VELOCITY",
                  "PRIZE_FREQUENCY", "GEO_MASKING", "PEP_HIGH_VALUE"}


def _route(rule_ids, cats, risk, pp):
    if rule_ids & TYPOLOGY_RULES and risk >= 65:
        return "Escalate to AML investigation: assess STR to FINTRAC"
    if pp:
        return "Refer to PlaySmart player protection"
    if (CATEGORY_AML in cats or CATEGORY_ANOMALY in cats) and risk >= 40:
        return "Enhanced monitoring: analyst review"
    if CATEGORY_REG in cats:
        return "File mandatory LCTR / CDR"
    return "Monitor"


def _tier(risk):
    return "Critical" if risk >= 85 else "High" if risk >= 65 else "Medium" if risk >= 40 else "Low"


@dataclass
class Result:
    tx: pd.DataFrame
    players: pd.DataFrame
    features: pd.DataFrame
    alerts: pd.DataFrame
    cases: pd.DataFrame
    flagged_tx: set


def analyze(tx: pd.DataFrame, players: pd.DataFrame, cfg: Config | None = None) -> Result:
    cfg = cfg or Config()
    tx = prepare(tx)
    players = players.copy()
    players["player_id"] = players["player_id"].astype(str)
    missing = set(tx["player_id"]) - set(players["player_id"])
    if missing:
        players = pd.concat([players, pd.DataFrame({"player_id": sorted(missing), "name": sorted(missing)})],
                            ignore_index=True)
    for col, default in [("name", ""), ("age", 0), ("occupation", "Unknown"), ("est_annual_income", 0),
                         ("city", ""), ("account_created", pd.NaT), ("kyc_status", "Unknown"), ("pep", False),
                         ("primary_channel", "")]:
        if col not in players:
            players[col] = default
        players[col] = players[col].fillna(default) if default is not pd.NaT else players[col]
    feats = player_features(tx, players)
    alerts = _rule_alerts(tx, players, feats, cfg)
    anom, pct = _anomaly_alerts(feats, cfg)
    alerts += anom
    adf = pd.DataFrame([asdict(a) for a in alerts]) if alerts else pd.DataFrame(
        columns=list(Alert.__dataclass_fields__))
    adf = adf.sort_values("severity", ascending=False).reset_index(drop=True)
    adf.insert(0, "alert_id", [f"AL-{i + 1:05d}" for i in range(len(adf))])

    tx_amt = tx.set_index("tx_id")["amount"]
    tx_ts = tx.set_index("tx_id")["timestamp"]
    tx_dir = tx.set_index("tx_id")["direction"]
    pm = players.set_index("player_id")
    rows = []
    for pid, a in adf.groupby("player_id"):
        scoring = a[a["category"].isin([CATEGORY_AML, CATEGORY_ANOMALY])] \
            .sort_values("severity", ascending=False)["severity"].tolist()
        # Noisy-OR: each extra alert closes part of the remaining gap to 100, with diminishing weight,
        # so many weak alerts cannot outrank one strong, specific one.
        weights = [1.0, 0.35, 0.2] + [0.1] * max(0, len(scoring) - 3)
        risk = 100 * (1 - np.prod([1 - s / 100 * w for s, w in zip(scoring, weights)]))
        risk = min(99.0, risk)
        if (a["category"] == CATEGORY_REG).any():
            # Mandatory reports are routine for big players; they add only a little on top.
            risk = min(99.0, max(risk, 30.0) + 3 * min(3, (a["category"] == CATEGORY_REG).sum()))
        pp = a[a["category"] == CATEGORY_PP]["severity"].max() if (a["category"] == CATEGORY_PP).any() else 0
        evidence = set()
        for ids in a[a["category"].isin([CATEGORY_AML, CATEGORY_REG])]["tx_ids"]:
            evidence.update(ids)
        ev_ids = [t for t in evidence if t in tx_amt.index]
        # Funds involved: money placed in, or money paid out if larger (e.g. prize claims). Wagers excluded
        # so the same dollars are not counted twice.
        ev_dir = tx_dir.loc[ev_ids] if ev_ids else pd.Series(dtype=str)
        susp = float(max(tx_amt.loc[ev_dir.index[ev_dir == "IN"]].sum(),
                         tx_amt.loc[ev_dir.index[ev_dir == "OUT"]].sum())) if ev_ids else 0.0
        ts = tx_ts.loc[ev_ids] if ev_ids else pd.Series(dtype="datetime64[ns]")
        cats = set(a["category"])
        top = a.sort_values("severity", ascending=False).iloc[0]
        meta = pm.loc[pid]
        linked = sorted({x for lst in a["linked_players"] for x in lst})
        rows.append({
            "player_id": pid, "name": meta.get("name", pid), "risk_score": round(risk, 1), "tier": _tier(risk),
            "pp_score": round(float(pp), 1),
            "route": _route(set(a["rule_id"]), cats, risk, pp > 0),
            "suspicious_amount": round(susp, 2),
            "first_seen": ts.min() if len(ts) else a["start"].min(),
            "last_seen": ts.max() if len(ts) else a["end"].max(),
            "n_alerts": len(a), "rules": ", ".join(a.sort_values("severity", ascending=False)["rule"]),
            "top_reason": top["reason"], "channel": meta.get("primary_channel", ""),
            "occupation": meta.get("occupation", ""), "linked_players": linked,
            "anomaly_pct": float(pct.get(pid, 0)),
        })
    cases = pd.DataFrame(rows)
    if len(cases):
        cases = cases.sort_values(["risk_score", "suspicious_amount"], ascending=False).reset_index(drop=True)
    flagged = set()
    for ids in adf.loc[adf["category"] != CATEGORY_REG, "tx_ids"]:
        flagged.update(ids)
    feats = feats.join(pct.rename("anomaly_pct"))
    return Result(tx, players, feats, adf, cases, flagged)


def _when(ts) -> str:
    return "unknown" if pd.isna(ts) else f"{pd.Timestamp(ts):%B %d, %Y %H:%M}"


def str_narrative(case: pd.Series, player: pd.Series, alerts: pd.DataFrame, feats: pd.Series) -> str:
    """Draft narrative in the shape FINTRAC asks for in an STR (who / what / when / where / why / how)."""
    aml = alerts[alerts["category"].isin([CATEGORY_AML, CATEGORY_ANOMALY])]
    lines = [
        f"SUBJECT: {player.get('name', case['player_id'])} (OLG player {case['player_id']}), age {player.get('age', '?')}, "
        f"{player.get('occupation', 'occupation unknown')}, {player.get('city', '')}. Stated income "
        f"{money(float(player.get('est_annual_income', 0) or 0))}. KYC: {player.get('kyc_status', 'Unknown')}. "
        f"PEP: {'Yes' if str(player.get('pep')).lower() in ('true', '1') else 'No'}.",
        "",
        f"WHAT / WHEN: Between {_when(case['first_seen'])} and {_when(case['last_seen'])} the subject "
        f"brought {money(feats['money_in'])} into OLG channels, wagered {money(feats['wagered'])} and received "
        f"{money(feats['money_out'])}. Transactions under review total {money(case['suspicious_amount'])}.",
        "",
        "WHY (grounds for suspicion):",
    ]
    for i, (_, a) in enumerate(aml.iterrows(), 1):
        lines.append(f"  {i}. {a['rule']}: {a['reason']}")
    if case["linked_players"]:
        lines += ["", f"LINKED ACCOUNTS: {', '.join(case['linked_players'])}"]
    lines += [
        "",
        "ACTION TAKEN: Case opened by OLG AML Sentinel; transactions preserved; enhanced monitoring applied.",
        "NOTE: Do not disclose this report or the investigation to the subject (PCMLTFA s.8, tipping-off offence).",
    ]
    return "\n".join(lines)
