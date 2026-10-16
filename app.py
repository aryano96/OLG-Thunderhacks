"""OLG AML Sentinel: SOC-style monitoring dashboard for AML investigators.

    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import asdict, fields

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sentinel.data_generator import generate
from sentinel.engine import (CATEGORY_AML, CATEGORY_ANOMALY, CATEGORY_PP, CATEGORY_REG, RULES, Config,
                             analyze, money, str_narrative)

st.set_page_config(page_title="OLG AML Sentinel", page_icon="🛡️", layout="wide")

TIER_COLORS = {"Critical": "#e5484d", "High": "#f76b15", "Medium": "#ffc53d", "Low": "#8b8d98"}
CAT_COLORS = {CATEGORY_AML: "#e5484d", CATEGORY_ANOMALY: "#8e4ec6", CATEGORY_REG: "#0090ff", CATEGORY_PP: "#30a46c"}
TX_COLORS = {"DEPOSIT": "#0090ff", "CASH_BUY_IN": "#0588f0", "WAGER": "#8b8d98", "WIN": "#30a46c",
             "WITHDRAWAL": "#e5484d", "CASH_OUT": "#f76b15", "PRIZE_CLAIM": "#ffc53d"}
STATUSES = ["New", "Investigating", "STR filed with FINTRAC", "LCTR / CDR filed", "Referred to PlaySmart",
            "Closed: false positive"]

st.markdown("""
<style>
.block-container {padding-top: 3.2rem;}
div[data-testid="stMetric"] {background: rgba(128,128,128,0.08); border: 1px solid rgba(128,128,128,0.18);
  border-radius: 10px; padding: 10px 14px;}
.badge {display:inline-block; padding:2px 10px; border-radius:999px; font-weight:600; font-size:0.85rem; color:#fff;}
.reason {border-left: 4px solid; padding: 8px 12px; margin: 6px 0 10px 0; border-radius: 4px;
  background: rgba(128,128,128,0.07);}
.reason small {opacity: 0.75;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner="Generating transaction stream…")
def synthetic(n_players: int, seed: int):
    return generate(n_players, seed)


@st.cache_data(show_spinner="Analysing every player and transaction…")
def run_engine(tx: pd.DataFrame, players: pd.DataFrame, cfg_items: tuple):
    return analyze(tx, players, Config(**dict(cfg_items)))


def badge(text, color):
    return f'<span class="badge" style="background:{color}">{text}</span>'


def short_money(x):
    for div, suf in [(1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if abs(x) >= div:
            return f"${x / div:.2f}{suf}"
    return money(x)


def fmt_ts(ts):
    return "—" if pd.isna(ts) else pd.Timestamp(ts).strftime("%b %d, %Y %H:%M")


with st.sidebar:
    st.title("🛡️ AML Sentinel")
    st.caption("Ontario Lottery and Gaming · Financial crime monitoring")
    source = st.radio("Data source", ["Synthetic OLG stream", "Upload CSVs"], horizontal=True)
    truth = None
    if source == "Synthetic OLG stream":
        n_players = st.select_slider("Players to simulate", [1000, 2500, 5000, 10000, 20000], value=5000)
        seed = st.number_input("Random seed", 1, 9999, 7)
        players_raw, tx_raw = synthetic(n_players, int(seed))
        truth = players_raw.set_index("player_id")["scenario"]
        players_in = players_raw.drop(columns=["scenario"])
    else:
        st.caption("Columns match `data/` from `python -m sentinel.data_generator`. See README for the schema.")
        tx_file = st.file_uploader("transactions.csv", type="csv")
        pl_file = st.file_uploader("players.csv", type="csv")
        if not tx_file or not pl_file:
            st.info("Upload both files to start.")
            st.stop()
        tx_raw = pd.read_csv(tx_file)
        players_in = pd.read_csv(pl_file)
        if "scenario" in players_in:
            truth = players_in.set_index("player_id")["scenario"]
            players_in = players_in.drop(columns=["scenario"])

    with st.expander("Detection sensitivity"):
        d = Config()
        cfg = Config(
            structuring_floor=st.slider("Structuring: 24h total counts as 'just under' from", 5000, 9500,
                                        int(d.structuring_floor), 250),
            minimal_play_wager_ratio=st.slider("Minimal play: wagered less than this share of funds", 0.05, 0.6,
                                               d.minimal_play_wager_ratio, 0.05),
            ring_min_accounts=st.slider("Shared device ring: minimum accounts", 2, 6, d.ring_min_accounts),
            affordability_ratio=st.slider("Affordability: money-in vs. income (×)", 1.0, 10.0,
                                          d.affordability_ratio, 0.5),
            spike_multiplier=st.slider("Spike: last 7 days vs. own weekly average (×)", 3.0, 20.0,
                                       d.spike_multiplier, 1.0),
            anomaly_contamination=st.slider("ML anomaly: share of players flagged", 0.005, 0.05,
                                            d.anomaly_contamination, 0.005, format="%.3f"),
        )

res = run_engine(tx_raw, players_in, tuple(sorted(asdict(cfg).items())))
tx, cases, alerts, feats = res.tx, res.cases, res.alerts, res.features
pmeta = res.players.set_index("player_id")

if "status" not in st.session_state:
    st.session_state.status = {}
    st.session_state.notes = {}
if len(cases):
    cases["status"] = cases["player_id"].map(st.session_state.status).fillna("New")

with st.sidebar:
    st.divider()
    st.metric("Transactions analysed", f"{len(tx):,}")
    st.caption(f"Window: {tx['timestamp'].min():%b %d} – {tx['timestamp'].max():%b %d, %Y}")
    st.caption("Case statuses and notes live in this browser session only.")

tabs = st.tabs(["🛰️ Command center", "🚨 Alert queue", "🔎 Investigation", "🕸️ Networks", "📡 Live feed",
                "📘 Rules & validation"])

# ======================================================= COMMAND CENTER
with tabs[0]:
    st.subheader("Command center")
    esc = cases[cases["route"].str.startswith("Escalate")] if len(cases) else cases
    c = st.columns(6)
    c[0].metric("Transactions", f"{len(tx):,}")
    c[1].metric("Money moved", short_money(tx.loc[tx["direction"] != "PLAY", "amount"].sum()))
    c[2].metric("Players monitored", f"{tx['player_id'].nunique():,}")
    c[3].metric("Open cases", f"{len(cases):,}")
    c[4].metric("AML escalations", f"{len(esc):,}")
    c[5].metric("$ under suspicion", short_money(esc["suspicious_amount"].sum() if len(esc) else 0))

    if not len(alerts):
        st.success("No alerts in this data.")
    else:
        left, right = st.columns([3, 2])
        with left:
            a = alerts.dropna(subset=["end"]).copy()
            a["day"] = pd.to_datetime(a["end"]).dt.floor("D")
            daily = a.groupby(["day", "category"]).size().reset_index(name="alerts")
            fig = px.bar(daily, x="day", y="alerts", color="category", color_discrete_map=CAT_COLORS,
                         title="Alerts raised per day")
            fig.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=10), legend_title=None,
                              xaxis_title=None, yaxis_title=None)
            st.plotly_chart(fig, width="stretch")
        with right:
            route = cases["route"].value_counts().reset_index()
            route.columns = ["route", "cases"]
            fig = px.bar(route, x="cases", y="route", orientation="h", title="Where cases are routed",
                         color_discrete_sequence=["#0090ff"])
            fig.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=10), yaxis_title=None, xaxis_title=None,
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, width="stretch")

        left, right = st.columns([2, 3])
        with left:
            by_rule = alerts[alerts["category"].isin([CATEGORY_AML, CATEGORY_ANOMALY])].groupby("rule") \
                .agg(players=("player_id", "nunique"), amount=("amount", "sum")).reset_index() \
                .sort_values("players")
            fig = px.bar(by_rule, x="players", y="rule", orientation="h", title="Laundering patterns detected",
                         hover_data={"amount": ":$,.0f"}, color_discrete_sequence=["#e5484d"])
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10), yaxis_title=None,
                              xaxis_title="players")
            st.plotly_chart(fig, width="stretch")
        with right:
            fig = px.scatter(cases, x="suspicious_amount", y="risk_score", color="tier", size="n_alerts",
                             color_discrete_map=TIER_COLORS, hover_name="name",
                             hover_data={"player_id": True, "route": True, "suspicious_amount": ":$,.0f"},
                             title="Cases: risk score vs. money involved", log_x=True)
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10), xaxis_title="suspicious $ (log)",
                              yaxis_title="risk score", legend_title=None)
            st.plotly_chart(fig, width="stretch")

        st.markdown("##### Highest-risk subjects right now")
        top = cases.head(10)[["risk_score", "tier", "name", "player_id", "suspicious_amount", "first_seen",
                              "last_seen", "top_reason"]]
        st.dataframe(top, hide_index=True, width="stretch", column_config={
            "risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.0f"),
            "suspicious_amount": st.column_config.NumberColumn("Suspicious $", format="$%,.0f"),
            "first_seen": st.column_config.DatetimeColumn("First seen", format="MMM DD HH:mm"),
            "last_seen": st.column_config.DatetimeColumn("Last seen", format="MMM DD HH:mm"),
            "top_reason": st.column_config.TextColumn("Main reason", width="large"),
            "name": "Name", "player_id": "Player ID", "tier": "Tier"})

# ========================================================== ALERT QUEUE
with tabs[1]:
    st.subheader("Alert queue")
    if not len(cases):
        st.success("Nothing to triage.")
    else:
        f = st.columns([2, 2, 2, 2, 3])
        tiers = f[0].multiselect("Tier", list(TIER_COLORS), default=["Critical", "High", "Medium"])
        routes = f[1].multiselect("Routing", sorted(cases["route"].unique()))
        rule_pick = f[2].multiselect("Rule", sorted(alerts["rule"].unique()))
        min_amt = f[3].number_input("Min suspicious $", 0, step=1000)
        q = f[4].text_input("Search name or player ID")
        view = cases[cases["tier"].isin(tiers)] if tiers else cases
        if routes:
            view = view[view["route"].isin(routes)]
        if rule_pick:
            ok = alerts[alerts["rule"].isin(rule_pick)]["player_id"].unique()
            view = view[view["player_id"].isin(ok)]
        view = view[view["suspicious_amount"] >= min_amt]
        if q:
            view = view[view["name"].str.contains(q, case=False) | view["player_id"].str.contains(q, case=False)]
        st.caption(f"{len(view)} case(s). Select a row to preview; open **Investigation** for the full file.")
        cols = ["risk_score", "tier", "status", "name", "player_id", "suspicious_amount", "first_seen", "last_seen",
                "n_alerts", "rules", "route"]
        ev = st.dataframe(view[cols], hide_index=True, width="stretch", height=440,
                          on_select="rerun", selection_mode="single-row", column_config={
                              "risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100,
                                                                            format="%.0f"),
                              "suspicious_amount": st.column_config.NumberColumn("Suspicious $", format="$%,.0f"),
                              "first_seen": st.column_config.DatetimeColumn("First seen", format="MMM DD HH:mm"),
                              "last_seen": st.column_config.DatetimeColumn("Last seen", format="MMM DD HH:mm"),
                              "n_alerts": st.column_config.NumberColumn("Alerts"),
                              "rules": st.column_config.TextColumn("Why flagged", width="large"),
                              "route": st.column_config.TextColumn("Recommended action", width="medium"),
                              "name": "Name", "player_id": "Player ID", "tier": "Tier", "status": "Status"})
        st.download_button("Download queue (CSV)", view.drop(columns=["linked_players"]).to_csv(index=False),
                           "aml_alert_queue.csv", "text/csv")
        rows = ev.selection.rows if ev and ev.selection else []
        if rows:
            sel = view.iloc[rows[0]]
            st.session_state.selected = sel["player_id"]
            st.markdown(f"#### {sel['name']} · {sel['player_id']}  "
                        + badge(f"{sel['tier']} · {sel['risk_score']:.0f}", TIER_COLORS[sel["tier"]]),
                        unsafe_allow_html=True)
            st.write(f"**{money(sel['suspicious_amount'])}** under review between **{fmt_ts(sel['first_seen'])}** "
                     f"and **{fmt_ts(sel['last_seen'])}**. {sel['route']}.")
            st.info(sel["top_reason"])

# ======================================================== INVESTIGATION
with tabs[2]:
    if not len(cases):
        st.success("No cases to investigate.")
    else:
        labels = {r.player_id: f"{r.risk_score:>4.0f} · {r.tier} · {r.name} ({r.player_id})"
                  for r in cases.itertuples()}
        ids = list(labels)
        default = st.session_state.get("selected", ids[0])
        pid = st.selectbox("Case", ids, index=ids.index(default) if default in ids else 0,
                           format_func=labels.get, key="case_pick")
        case = cases.set_index("player_id").loc[pid]
        case["player_id"] = pid
        person = pmeta.loc[pid]
        pf = feats.loc[pid]
        pa = alerts[alerts["player_id"] == pid].sort_values("severity", ascending=False)
        ptx = tx[tx["player_id"] == pid].copy()
        ptx["flagged"] = ptx["tx_id"].isin(res.flagged_tx)

        st.markdown(f"## {person.get('name', pid)} "
                    + badge(f"{case['tier']} risk · {case['risk_score']:.0f}/100", TIER_COLORS[case["tier"]]),
                    unsafe_allow_html=True)
        st.markdown(f"**Player ID** `{pid}` &nbsp;·&nbsp; **Recommended action:** {case['route']}")

        m = st.columns(5)
        m[0].metric("Suspicious $", money(case["suspicious_amount"]))
        m[1].metric("Money in", money(pf["money_in"]))
        m[2].metric("Wagered", money(pf["wagered"]),
                    f"{pf['wager_ratio']:.0%} of funds" if pd.notna(pf["wager_ratio"]) else None,
                    delta_color="off")
        m[3].metric("Money out", money(pf["money_out"]))
        m[4].metric("Anomaly rank", f"top {max(0.1, 100 - pf['anomaly_pct']):.1f}%")
        st.caption(f"Activity under review: **{fmt_ts(case['first_seen'])}** → **{fmt_ts(case['last_seen'])}**")

        left, right = st.columns([3, 2])
        with left:
            st.markdown("### Why the system flagged this person")
            reg = pa[pa["category"] == CATEGORY_REG]
            for i, a in enumerate(pa[pa["category"] != CATEGORY_REG].itertuples(), 1):
                color = CAT_COLORS.get(a.category, "#888")
                when = f"{fmt_ts(a.start)} → {fmt_ts(a.end)}" if pd.notna(a.start) else "whole period"
                st.markdown(
                    f'<div class="reason" style="border-color:{color}"><b>{i}. {a.rule}</b> '
                    f'{badge(a.category, color)} <small>severity {a.severity:.0f} · {money(a.amount)} · {when} · '
                    f'{len(a.tx_ids)} transaction(s)</small><br>{a.reason}</div>', unsafe_allow_html=True)
            if len(reg):
                kinds = reg["rule_id"].value_counts()
                summary = ", ".join(f"{n} × {k}" for k, n in kinds.items())
                with st.expander(f"Mandatory FINTRAC reports due: {summary} ({money(reg['amount'].sum())})"):
                    for a in reg.itertuples():
                        st.markdown(f"- **{a.rule}** · {fmt_ts(a.start)} → {fmt_ts(a.end)} · {money(a.amount)}: "
                                    f"{a.reason}")
        with right:
            st.markdown("### Subject profile")
            pep = str(person.get("pep")).lower() in ("true", "1")
            created = pd.to_datetime(person.get("account_created"), errors="coerce")
            profile = {
                "Name": person.get("name", ""), "Age": person.get("age", ""),
                "Occupation": person.get("occupation", ""),
                "Stated income": money(float(person.get("est_annual_income", 0) or 0)),
                "City": person.get("city", ""),
                "Account opened": "—" if pd.isna(created) else f"{created:%b %d, %Y} ({int(pf['account_age_days'])} days)",
                "KYC": person.get("kyc_status", ""), "PEP": "⚠️ Yes" if pep else "No",
                "Main channel": person.get("primary_channel", ""),
                "Devices / IPs": f"{int(pf['n_devices'])} / {int(pf['n_ips'])}",
                "Funding sources": int(pf["n_funding"]),
                "Venues": ", ".join(sorted(v for v in ptx["venue"].unique() if v)[:4]),
            }
            st.table(pd.DataFrame([(k, str(v)) for k, v in profile.items()], columns=["", "value"]).set_index(""))
            if case["linked_players"]:
                st.markdown("### Linked accounts")
                link = cases.set_index("player_id").reindex(case["linked_players"])
                st.dataframe(pd.DataFrame({
                    "player_id": case["linked_players"],
                    "name": [pmeta.loc[x, "name"] if x in pmeta.index else x for x in case["linked_players"]],
                    "risk": link["risk_score"].values,
                    "money_in": [feats.loc[x, "money_in"] if x in feats.index else 0 for x in case["linked_players"]],
                }), hide_index=True, width="stretch",
                    column_config={"money_in": st.column_config.NumberColumn("Money in", format="$%,.0f")})

        st.markdown("### Money timeline")
        plot = ptx[ptx["tx_type"] != "WIN"].copy()
        plot["flag"] = np.where(plot["flagged"], "flagged", "normal")
        fig = px.scatter(plot, x="timestamp", y="amount", color="tx_type", symbol="flag",
                         symbol_map={"flagged": "diamond", "normal": "circle"}, color_discrete_map=TX_COLORS,
                         hover_data={"tx_id": True, "venue": True, "game": True, "payment_method": True,
                                     "amount": ":$,.2f"}, log_y=True)
        fig.add_hline(y=10_000, line_dash="dot", line_color="#e5484d",
                      annotation_text="$10,000 FINTRAC threshold", annotation_position="top left")
        fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color="rgba(0,0,0,0.4)")))
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), legend_title=None,
                          xaxis_title=None, yaxis_title="amount (log $)")
        st.plotly_chart(fig, width="stretch")

        st.markdown("### Transaction ledger")
        only = st.toggle("Only flagged transactions", value=True)
        led = ptx[ptx["flagged"]] if only and ptx["flagged"].any() else ptx
        led_cols = ["flagged", "timestamp", "tx_id", "tx_type", "amount", "channel", "venue", "payment_method",
                    "funding_source", "game", "stake_at_risk_pct", "device_id", "ip_address", "ip_region"]
        st.dataframe(led[led_cols].sort_values("timestamp", ascending=False), hide_index=True,
                     width="stretch", height=320, column_config={
                         "flagged": st.column_config.CheckboxColumn("🚩"),
                         "amount": st.column_config.NumberColumn("Amount", format="$%,.2f"),
                         "stake_at_risk_pct": st.column_config.NumberColumn("Stake at risk", format="%.0f%%"),
                         "timestamp": st.column_config.DatetimeColumn("Time", format="MMM DD YYYY HH:mm")})

        st.markdown("### Case actions")
        a1, a2 = st.columns([1, 2])
        with a1:
            cur = st.session_state.status.get(pid, "New")
            st.session_state.status[pid] = st.selectbox("Case status", STATUSES, index=STATUSES.index(cur),
                                                        key=f"status_{pid}")
            st.session_state.notes[pid] = st.text_area("Investigator notes", st.session_state.notes.get(pid, ""),
                                                       key=f"notes_{pid}", height=140)
        with a2:
            narrative = str_narrative(case, person, pa, pf)
            narrative = st.text_area("Draft STR narrative (edit before filing)", narrative, height=260,
                                     key=f"str_{pid}")
        dossier = {
            "player": {k: (str(v) if not isinstance(v, (int, float, bool)) else v) for k, v in person.items()},
            "player_id": pid,
            "risk_score": float(case["risk_score"]), "tier": case["tier"], "route": case["route"],
            "suspicious_amount": float(case["suspicious_amount"]),
            "first_seen": str(case["first_seen"]), "last_seen": str(case["last_seen"]),
            "status": st.session_state.status[pid], "notes": st.session_state.notes[pid],
            "alerts": json.loads(pa.drop(columns=["linked_players"]).to_json(orient="records", date_format="iso")),
            "linked_players": case["linked_players"], "str_narrative": narrative,
        }
        d = st.columns(3)
        d[0].download_button("⬇️ Case file (JSON)", json.dumps(dossier, indent=2, default=str),
                             f"case_{pid}.json", "application/json", width="stretch")
        d[1].download_button("⬇️ Transactions (CSV)", ptx[led_cols].to_csv(index=False), f"transactions_{pid}.csv",
                             "text/csv", width="stretch")
        d[2].download_button("⬇️ STR draft (TXT)", narrative, f"str_draft_{pid}.txt", "text/plain",
                             width="stretch")
        st.caption("Never tell the subject an STR is being considered or filed. Tipping off is an offence under PCMLTFA.")

# ============================================================= NETWORKS
with tabs[3]:
    st.subheader("Shared device & IP networks")
    st.caption("Each cluster is one device or IP address (centre) and the accounts that used it. "
               "Many accounts on one identifier is a strong sign of mule networks run by one person.")
    online = tx[(tx["channel"] == "OLG.ca")]
    rings = {}
    for key, kind in [("device_id", "Device"), ("ip_address", "IP")]:
        sub = online[online[key] != ""]
        g = sub.groupby(key)["player_id"].unique()
        for ident, pids in g.items():
            if len(pids) >= cfg.ring_min_accounts:
                rings.setdefault(frozenset(pids), []).append(f"{kind} {ident}")
    # One cluster per group of accounts, listing every identifier they share
    edges = [(" + ".join(ids), sorted(p)) for p, ids in rings.items()]
    if not edges:
        st.info("No shared identifiers above the ring threshold.")
    else:
        fig = go.Figure()
        ncol = math.ceil(math.sqrt(len(edges)))
        risk_map = cases.set_index("player_id")["risk_score"] if len(cases) else pd.Series(dtype=float)
        for k, (hub, pids) in enumerate(edges):
            cx, cy = (k % ncol) * 3.0, -(k // ncol) * 3.0
            xs, ys, txt, col = [], [], [], []
            for j, p in enumerate(pids):
                ang = 2 * math.pi * j / len(pids)
                x, y = cx + math.cos(ang), cy + math.sin(ang)
                fig.add_trace(go.Scatter(x=[cx, x], y=[cy, y], mode="lines", line=dict(color="#8b8d98", width=1),
                                         hoverinfo="skip", showlegend=False))
                xs.append(x); ys.append(y)
                nm = pmeta.loc[p, "name"] if p in pmeta.index else p
                txt.append(f"{nm}<br>{p}<br>money in {money(feats.loc[p, 'money_in'])}<br>"
                           f"risk {risk_map.get(p, 0):.0f}")
                col.append(float(risk_map.get(p, 0)))
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(size=16, color=col, cmin=0, cmax=100,
                                     colorscale=[[0, "#8b8d98"], [0.5, "#ffc53d"], [1, "#e5484d"]],
                                     line=dict(width=1, color="#222")),
                                     hovertext=txt, hoverinfo="text", showlegend=False))
            fig.add_trace(go.Scatter(x=[cx], y=[cy], mode="markers+text", text=[" + ".join(h.split(" ")[0] for h in hub.split(" + "))],
                                     textposition="bottom center", hovertext=[hub], hoverinfo="text",
                                     marker=dict(size=22, symbol="hexagon", color="#0090ff"), showlegend=False))
        fig.update_layout(height=max(380, 260 * math.ceil(len(edges) / ncol)), margin=dict(l=10, r=10, t=10, b=10),
                          xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(pd.DataFrame([{
            "identifier": hub, "accounts": len(p),
            "names": ", ".join(pmeta.loc[x, "name"] for x in p if x in pmeta.index),
            "ring money in": sum(feats.loc[x, "money_in"] for x in p if x in feats.index)} for hub, p in edges]),
            hide_index=True, width="stretch",
            column_config={"ring money in": st.column_config.NumberColumn(format="$%,.0f")})

# ============================================================ LIVE FEED
with tabs[4]:
    st.subheader("Live transaction feed (replay)")
    st.caption("Replays the stream in time order, the way a SOC analyst watches it, and raises alerts the moment "
               "the evidence for them is complete.")
    times = tx.sort_values("timestamp").reset_index(drop=True)
    if "feed_pos" not in st.session_state:
        st.session_state.feed_pos = int(len(times) * 0.6)
    ctl = st.columns([1, 1, 2, 3])
    playing = ctl[0].toggle("▶ Play", value=False)
    speed = ctl[1].select_slider("Speed", [25, 100, 400, 1500], value=100)
    if ctl[2].button("⏮ Restart from 60%"):
        st.session_state.feed_pos = int(len(times) * 0.6)

    @st.fragment(run_every=1.0 if playing else None)
    def feed():
        if playing:
            st.session_state.feed_pos = min(len(times) - 1, st.session_state.feed_pos + speed)
        pos = st.session_state.feed_pos
        now = times.loc[pos, "timestamp"]
        raised = alerts[alerts["end"].notna() & (pd.to_datetime(alerts["end"]) <= now)]
        k = st.columns(4)
        k[0].metric("Stream clock", f"{now:%b %d %H:%M}")
        k[1].metric("Transactions seen", f"{pos + 1:,}")
        k[2].metric("Alerts raised", f"{len(raised):,}")
        k[3].metric("AML alerts", f"{(raised['category'] == CATEGORY_AML).sum():,}")
        l, r = st.columns([3, 2])
        recent = times.loc[max(0, pos - 18):pos].iloc[::-1].copy()
        recent["🚩"] = recent["tx_id"].isin(res.flagged_tx)
        recent["name"] = recent["player_id"].map(pmeta["name"])
        l.dataframe(recent[["🚩", "timestamp", "name", "player_id", "tx_type", "amount", "channel", "venue"]],
                    hide_index=True, width="stretch", height=560, column_config={
                        "amount": st.column_config.NumberColumn(format="$%,.2f"),
                        "timestamp": st.column_config.DatetimeColumn("Time", format="MMM DD HH:mm:ss")})
        latest = raised.sort_values("end", ascending=False).head(8)
        with r:
            st.markdown("**Latest alerts**")
            for a in latest.itertuples():
                nm = pmeta.loc[a.player_id, "name"] if a.player_id in pmeta.index else a.player_id
                st.markdown(f'<div class="reason" style="border-color:{CAT_COLORS.get(a.category, "#888")}">'
                            f'<b>{a.rule}</b> · {nm}<br><small>{fmt_ts(a.end)} · {money(a.amount)}</small></div>',
                            unsafe_allow_html=True)
    feed()

# ================================================== RULES & VALIDATION
with tabs[5]:
    st.subheader("How the engine decides")
    st.markdown(
        "Every player runs through **typology rules** (known laundering methods and mandatory FINTRAC reports) and an "
        "**Isolation Forest** anomaly model that compares each player's whole profile with everyone else's. Alerts "
        "roll up into one case per player. Risk = strongest alert + 35% of the next + 15% of the third + 5% of the rest "
        "(capped at 99). Cases that match a laundering method with risk ≥ 65 are escalated for an STR decision; "
        "gambling-harm signals go to PlaySmart instead of AML.")
    st.dataframe(pd.DataFrame([{"rule": n, "category": c, "what it catches": d} for n, c, d in RULES.values()]),
                 hide_index=True, width="stretch")
    st.markdown("**Current thresholds**")
    st.json(asdict(cfg), expanded=False)
    if truth is not None and len(cases):
        st.markdown("### Validation against planted scenarios")
        st.caption("The synthetic stream hides known launderers among normal players. The engine never sees these "
                   "labels; this table checks what it found.")
        routes = cases.set_index("player_id")["route"]
        rows = []
        for sc, ids in truth.groupby(truth).groups.items():
            r = routes.reindex(list(ids)).fillna("Not flagged")
            rows.append({"scenario": sc, "players": len(ids),
                         "escalated to AML": int(r.str.startswith("Escalate").sum()),
                         "analyst review": int(r.str.startswith("Enhanced").sum()),
                         "PlaySmart": int(r.str.startswith("Refer").sum()),
                         "filing only": int(r.str.startswith("File").sum()),
                         "not flagged": int((r == "Not flagged").sum())})
        st.dataframe(pd.DataFrame(rows).sort_values("players"), hide_index=True, width="stretch")
