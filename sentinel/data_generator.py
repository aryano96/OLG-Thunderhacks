"""Synthetic OLG transaction stream.

Generates a month of player activity across OLG.ca, land-based casinos and
lottery retail, and plants known money-laundering typologies inside it so the
detection engine has something real to find. Every planted player carries a
`scenario` label in players.csv; the dashboard never reads it, it is only used
by evaluate.py to measure how well the engine catches them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

END = datetime(2026, 10, 15, 23, 59)
DAYS = 30
START = END - timedelta(days=DAYS)

FIRST = [
    "Liam", "Olivia", "Noah", "Emma", "Ethan", "Ava", "Lucas", "Sophia", "Mason", "Mia",
    "Logan", "Amelia", "Aiden", "Chloe", "Jacob", "Emily", "Owen", "Isla", "Wei", "Priya",
    "Arjun", "Fatima", "Omar", "Hiro", "Mei", "Sanjay", "Aisha", "Mateo", "Lucia", "Diego",
    "Nadia", "Ivan", "Elena", "Tariq", "Zara", "Kwame", "Amara", "Jin", "Yuki", "Rohan",
    "Gabriel", "Leah", "Marcus", "Hannah", "Victor", "Grace", "Daniel", "Sara", "Andre", "Maya",
]
LAST = [
    "Smith", "Brown", "Tremblay", "Martin", "Roy", "Wilson", "MacDonald", "Gagnon", "Johnson",
    "Taylor", "Campbell", "Anderson", "Leblanc", "Lee", "Chen", "Wang", "Patel", "Singh", "Khan",
    "Nguyen", "Kim", "Ali", "Rossi", "Silva", "Kowalski", "Novak", "Murphy", "O'Connor", "Clarke",
    "Stewart", "Thompson", "Reid", "Morin", "Bouchard", "Fraser", "Grant", "Hughes", "Mensah",
    "Okafor", "Haddad", "Sato", "Tanaka", "Ivanov", "Petrov", "Garcia", "Lopez", "Romero", "Dubois",
]
CITIES = [
    "Toronto", "Mississauga", "Brampton", "Ottawa", "Hamilton", "London", "Markham", "Vaughan",
    "Kitchener", "Windsor", "Richmond Hill", "Oakville", "Burlington", "Sudbury", "Oshawa",
    "Barrie", "St. Catharines", "Guelph", "Kingston", "Thunder Bay", "Niagara Falls",
]
OCCUPATIONS = [
    ("Student", 18_000), ("Retail Associate", 32_000), ("Server / Bartender", 30_000),
    ("Warehouse Associate", 42_000), ("Truck Driver", 58_000), ("Electrician", 72_000),
    ("Registered Nurse", 88_000), ("Teacher", 82_000), ("Software Developer", 110_000),
    ("Accountant", 85_000), ("Real Estate Agent", 95_000), ("Small Business Owner", 120_000),
    ("Construction Worker", 55_000), ("Retired", 45_000), ("Unemployed", 12_000),
    ("Physician", 260_000), ("Lawyer", 180_000), ("Corporate Executive", 300_000),
    ("Customer Service Rep", 38_000), ("Graphic Designer", 52_000),
]
CASINOS = [
    "Casino Woodbine", "Fallsview Casino", "Casino Rama", "Great Canadian Casino Pickering",
    "Casino Ajax", "Caesars Windsor", "Casino Niagara", "Shorelines Casino Thousand Islands",
]
RETAILERS = [
    "Mac's #2231 Toronto", "Shoppers Drug Mart #881", "Circle K #1204 Brampton",
    "Rabba Fine Foods #19", "Hasty Market #77 Ottawa", "Loblaws #1092 Hamilton",
]
ONLINE_GAMES = ["Slots", "Blackjack", "Roulette", "Baccarat", "PROLINE+ Sports", "Instant Win", "Poker"]
CASINO_GAMES = ["Slots", "Blackjack", "Baccarat", "Roulette", "Poker", "Craps"]
LOTTERY_GAMES = ["Lotto Max", "Lotto 6/49", "Daily Grand", "Instant Tickets", "Encore"]

TX_COLUMNS = [
    "tx_id", "timestamp", "player_id", "channel", "venue", "tx_type", "amount",
    "payment_method", "funding_source", "game", "stake_at_risk_pct", "device_id",
    "ip_address", "ip_region",
]


@dataclass
class Player:
    player_id: str
    name: str
    age: int
    occupation: str
    est_annual_income: int
    city: str
    account_created: datetime
    kyc_status: str
    pep: bool
    primary_channel: str
    scenario: str = "normal"
    devices: list = field(default_factory=list)
    ips: list = field(default_factory=list)
    cards: list = field(default_factory=list)


class Generator:
    def __init__(self, n_players: int = 5000, seed: int = 7):
        self.rng = np.random.default_rng(seed)
        self.n_players = n_players
        self.rows: list[dict] = []
        self.players: list[Player] = []
        self._tx = 0

    # ------------------------------------------------------------------ helpers
    def _r(self, lo, hi):
        return float(self.rng.uniform(lo, hi))

    def _pick(self, seq):
        return seq[int(self.rng.integers(len(seq)))]

    def _money(self, x):
        return round(max(1.0, float(x)), 2)

    def _device(self):
        return f"DEV-{self.rng.integers(16**8):08X}"

    def _ip(self, region="Ontario"):
        if region == "Ontario":
            return f"99.{self.rng.integers(224, 255)}.{self.rng.integers(256)}.{self.rng.integers(1, 255)}"
        if region == "VPN/Proxy":
            return f"185.{self.rng.integers(100, 200)}.{self.rng.integers(256)}.{self.rng.integers(1, 255)}"
        return f"72.{self.rng.integers(10, 99)}.{self.rng.integers(256)}.{self.rng.integers(1, 255)}"

    def _card(self, kind=None):
        kind = kind or self._pick(["VISA", "MC", "DEBIT", "INTERAC"])
        return f"{kind}-****{self.rng.integers(10000):04d}"

    def _time(self, day_offset=None, hour_lo=10, hour_hi=24):
        day = int(self.rng.integers(DAYS)) if day_offset is None else day_offset
        hour = self._r(hour_lo, hour_hi)
        return START + timedelta(days=day, hours=hour)

    def _emit(self, p: Player, ts, channel, tx_type, amount, *, venue=None, method="",
              funding="", game="", risk=np.nan, device=None, ip=None, region="Ontario"):
        if ts > END:
            ts = END - timedelta(minutes=float(self.rng.uniform(1, 600)))
        if ts < START:
            ts = START + timedelta(minutes=float(self.rng.uniform(1, 600)))
        self._tx += 1
        online = channel == "OLG.ca"
        self.rows.append({
            "tx_id": f"TX-{self._tx:07d}",
            "timestamp": ts.replace(microsecond=0),
            "player_id": p.player_id,
            "channel": channel,
            "venue": venue or ("OLG.ca" if online else ""),
            "tx_type": tx_type,
            "amount": self._money(amount),
            "payment_method": method,
            "funding_source": funding,
            "game": game,
            "stake_at_risk_pct": risk,
            "device_id": (device or (p.devices[0] if p.devices else "")) if online else "",
            "ip_address": (ip or (p.ips[0] if p.ips else "")) if online else "",
            "ip_region": region if online else "",
        })

    def _new_player(self, scenario="normal", **over) -> Player:
        idx = len(self.players) + 1
        occ, base_income = self._pick(OCCUPATIONS)
        age = int(np.clip(self.rng.normal(42, 14), 19, 85))
        if occ == "Student":
            age = int(self.rng.integers(19, 26))
        if occ == "Retired":
            age = int(self.rng.integers(60, 85))
        income = int(base_income * self._r(0.75, 1.3))
        created = END - timedelta(days=int(self.rng.integers(40, 2500)))
        p = Player(
            player_id=f"OLG-{100000 + idx * 7 + int(self.rng.integers(7))}",
            name=f"{self._pick(FIRST)} {self._pick(LAST)}",
            age=age,
            occupation=occ,
            est_annual_income=income,
            city=self._pick(CITIES),
            account_created=created,
            kyc_status="Verified" if self.rng.random() > 0.04 else "Pending",
            pep=False,
            primary_channel=self._pick(["OLG.ca"] * 6 + ["Casino"] * 3 + ["Lottery Retail"] * 2),
            scenario=scenario,
        )
        for k, v in over.items():
            setattr(p, k, v)
        p.devices = [self._device() for _ in range(1 + int(self.rng.random() < 0.3))]
        p.ips = [self._ip() for _ in range(1 + int(self.rng.random() < 0.4))]
        p.cards = [self._card() for _ in range(1 + int(self.rng.random() < 0.35))]
        self.players.append(p)
        return p

    # ------------------------------------------------------- normal behaviour
    def _session_online(self, p: Player, ts, deposit, game=None, night=False):
        game = game or self._pick(ONLINE_GAMES)
        card = self._pick(p.cards)
        method = "Interac e-Transfer" if card.startswith("INTERAC") else (
            "Debit" if card.startswith("DEBIT") else "Credit Card")
        dev, ip = self._pick(p.devices), self._pick(p.ips)
        self._emit(p, ts, "OLG.ca", "DEPOSIT", deposit, method=method, funding=card, device=dev, ip=ip)
        wager = deposit * self._r(1.4, 4.5)
        rtp = self._r(0.80, 1.02)
        self._emit(p, ts + timedelta(minutes=self._r(2, 20)), "OLG.ca", "WAGER", wager,
                   game=game, risk=round(self._r(0.6, 1.0), 2), device=dev, ip=ip)
        ret = wager * rtp - (deposit * self._r(0.2, 0.9))
        if ret > 0:
            self._emit(p, ts + timedelta(minutes=self._r(25, 120)), "OLG.ca", "WIN", ret,
                       game=game, device=dev, ip=ip)
        balance = deposit - wager + max(ret, 0)
        if balance > 40 and self.rng.random() < 0.55:
            self._emit(p, ts + timedelta(hours=self._r(2, 72)), "OLG.ca", "WITHDRAWAL",
                       balance * self._r(0.5, 1.0), method="EFT", funding="BANK-****" + card[-4:],
                       device=dev, ip=ip)

    def _normal_online(self, p: Player):
        weekly = (p.est_annual_income / 52) * self._r(0.004, 0.03)
        n = int(self.rng.poisson(DAYS / self._r(2.5, 7)))
        for _ in range(max(1, n)):
            dep = self.rng.lognormal(np.log(max(20, weekly * self._r(0.5, 2.0))), 0.5)
            self._session_online(p, self._time(hour_lo=11, hour_hi=24), min(dep, 2500))

    def _normal_casino(self, p: Player, high_roller=False):
        visits = int(self.rng.integers(1, 4 if high_roller else 6))
        casino = self._pick(CASINOS)
        for _ in range(visits):
            ts = self._time(hour_lo=14, hour_hi=23)
            if high_roller:
                buy = self._r(3000, 12000)
            else:
                buy = self.rng.lognormal(np.log(max(80, p.est_annual_income / 300)), 0.6)
                buy = min(buy, 3000)
            game = self._pick(CASINO_GAMES)
            self._emit(p, ts, "Casino", "CASH_BUY_IN", buy, venue=casino, method="Cash", funding="CASH")
            wager = buy * self._r(2.0, 6.0)
            self._emit(p, ts + timedelta(minutes=self._r(5, 30)), "Casino", "WAGER", wager,
                       venue=casino, game=game, risk=round(self._r(0.7, 1.0), 2))
            out = buy * self._r(0.0, 1.35)
            if out > 20:
                self._emit(p, ts + timedelta(hours=self._r(1, 5)), "Casino", "CASH_OUT", out,
                           venue=casino, method="Cash" if out < 3000 else "Casino Cheque",
                           funding="CASH")

    def _normal_lottery(self, p: Player):
        n = int(self.rng.integers(3, 14))
        store = self._pick(RETAILERS)
        for _ in range(n):
            ts = self._time(hour_lo=8, hour_hi=22)
            self._emit(p, ts, "Lottery Retail", "WAGER", self._r(3, 60), venue=store,
                       method="Cash", funding="CASH", game=self._pick(LOTTERY_GAMES), risk=1.0)
            if self.rng.random() < 0.08:
                self._emit(p, ts + timedelta(days=self._r(1, 5)), "Lottery Retail", "PRIZE_CLAIM",
                           self._pick([10, 20, 50, 100, 250, 500]), venue=store, method="Cash",
                           game=self._pick(LOTTERY_GAMES))

    def _normal(self, p: Player):
        if p.primary_channel == "OLG.ca":
            self._normal_online(p)
        elif p.primary_channel == "Casino":
            self._normal_casino(p, high_roller=p.est_annual_income > 200_000 and self.rng.random() < 0.5)
        else:
            self._normal_lottery(p)
        if self.rng.random() < 0.25:
            self._normal_lottery(p)

    # ------------------------------------------------------ planted typologies
    def _structuring(self):
        """Cash buy-ins split to stay just under the $10,000 LCTR threshold."""
        p = self._new_player("structuring", primary_channel="Casino")
        days = sorted(self.rng.choice(DAYS, size=int(self.rng.integers(3, 7)), replace=False))
        for d in days:
            target = self._r(8600, 9950)
            parts = int(self.rng.integers(2, 5))
            splits = self.rng.dirichlet(np.ones(parts) * 4) * target
            ts = self._time(int(d), 12, 16)
            for amt in splits:
                casino = self._pick(CASINOS)
                self._emit(p, ts, "Casino", "CASH_BUY_IN", amt, venue=casino, method="Cash", funding="CASH")
                ts += timedelta(hours=self._r(1.0, 3.5))
            self._emit(p, ts, "Casino", "WAGER", target * self._r(0.1, 0.35), venue=casino,
                       game="Baccarat", risk=round(self._r(0.5, 0.9), 2))
            self._emit(p, ts + timedelta(hours=1), "Casino", "CASH_OUT", target * self._r(0.85, 0.97),
                       venue=casino, method="Casino Cheque", funding="CASH")

    def _minimal_play(self):
        """Large buy-in or deposit, almost no play, cash out as 'winnings'."""
        online = self.rng.random() < 0.5
        p = self._new_player("minimal_play", primary_channel="OLG.ca" if online else "Casino")
        self._normal(p)
        for _ in range(int(self.rng.integers(2, 5))):
            amt = self._r(4000, 22000)
            ts = self._time(hour_lo=10, hour_hi=20)
            if online:
                card = self._card("VISA")
                p.cards.append(card)
                self._emit(p, ts, "OLG.ca", "DEPOSIT", amt, method="Credit Card", funding=card)
                self._emit(p, ts + timedelta(minutes=20), "OLG.ca", "WAGER", amt * self._r(0.03, 0.12),
                           game="Roulette", risk=round(self._r(0.3, 0.8), 2))
                self._emit(p, ts + timedelta(hours=self._r(1, 10)), "OLG.ca", "WITHDRAWAL",
                           amt * self._r(0.9, 0.98), method="EFT", funding="BANK-****" + card[-4:])
            else:
                casino = self._pick(CASINOS)
                self._emit(p, ts, "Casino", "CASH_BUY_IN", amt, venue=casino, method="Cash", funding="CASH")
                self._emit(p, ts + timedelta(minutes=30), "Casino", "WAGER", amt * self._r(0.03, 0.12),
                           venue=casino, game="Blackjack", risk=round(self._r(0.5, 1.0), 2))
                self._emit(p, ts + timedelta(hours=self._r(1, 3)), "Casino", "CASH_OUT",
                           amt * self._r(0.9, 0.99), venue=casino, method="Casino Cheque", funding="CASH")

    def _mule_ring(self, size):
        """New accounts sharing one device / IP, funded by prepaid & third-party cards."""
        dev, ip = self._device(), self._ip("VPN/Proxy" if self.rng.random() < 0.5 else "Ontario")
        region = "VPN/Proxy" if ip.startswith("185.") else "Ontario"
        for _ in range(size):
            p = self._new_player("mule_ring", primary_channel="OLG.ca",
                                 occupation="Student", est_annual_income=int(self._r(9000, 22000)),
                                 age=int(self.rng.integers(19, 25)),
                                 account_created=END - timedelta(days=int(self.rng.integers(5, 28))))
            p.devices, p.ips = [dev], [ip]
            p.cards = [self._card(self._pick(["PREPAID", "VISA", "MC"])) for _ in range(int(self.rng.integers(3, 7)))]
            for _ in range(int(self.rng.integers(4, 9))):
                ts = self._time(int(self.rng.integers(DAYS - 25, DAYS)), 0, 24)
                amt = self._r(900, 3500)
                card = self._pick(p.cards)
                self._emit(p, ts, "OLG.ca", "DEPOSIT", amt, method="Prepaid Card" if card.startswith("PREPAID") else "Credit Card",
                           funding=card, device=dev, ip=ip, region=region)
                self._emit(p, ts + timedelta(minutes=15), "OLG.ca", "WAGER", amt * self._r(0.05, 0.2),
                           game="Baccarat", risk=round(self._r(0.2, 0.6), 2), device=dev, ip=ip, region=region)
                self._emit(p, ts + timedelta(hours=self._r(2, 30)), "OLG.ca", "WITHDRAWAL",
                           amt * self._r(0.8, 0.95), method="EFT", funding="BANK-****" + str(self.rng.integers(10000)).zfill(4),
                           device=dev, ip=ip, region=region)

    def _hedged_betting(self):
        """Even-money bets on both sides (red & black) so almost nothing is at risk."""
        p = self._new_player("hedged_betting", primary_channel="OLG.ca")
        self._normal(p)
        for _ in range(int(self.rng.integers(3, 7))):
            amt = self._r(3000, 12000)
            ts = self._time(hour_lo=12, hour_hi=23)
            self._session_hedge(p, ts, amt)

    def _session_hedge(self, p, ts, amt):
        card = self._pick(p.cards)
        game = self._pick(["Roulette", "Baccarat"])
        self._emit(p, ts, "OLG.ca", "DEPOSIT", amt, method="Debit", funding=card)
        self._emit(p, ts + timedelta(minutes=10), "OLG.ca", "WAGER", amt * self._r(1.5, 3.0),
                   game=game, risk=round(self._r(0.02, 0.07), 2))
        self._emit(p, ts + timedelta(minutes=50), "OLG.ca", "WIN", amt * self._r(1.4, 2.9), game=game)
        self._emit(p, ts + timedelta(hours=self._r(3, 20)), "OLG.ca", "WITHDRAWAL",
                   amt * self._r(0.93, 0.99), method="EFT", funding="BANK-****" + card[-4:])

    def _ticket_discounting(self):
        """Buys winning tickets from real winners for cash, claims them for a 'clean' OLG cheque."""
        p = self._new_player("ticket_discounting", primary_channel="Lottery Retail")
        self._normal_lottery(p)
        for _ in range(int(self.rng.integers(4, 10))):
            ts = self._time(hour_lo=9, hour_hi=18)
            self._emit(p, ts, "Lottery Retail", "PRIZE_CLAIM", self._r(1000, 15000),
                       venue="OLG Prize Centre Toronto", method="OLG Prize Cheque",
                       game=self._pick(LOTTERY_GAMES))

    def _account_takeover(self):
        """Quiet low-stakes account suddenly moves big money from a new device / out-of-province IP."""
        p = self._new_player("account_takeover", primary_channel="OLG.ca",
                             est_annual_income=int(self._r(25000, 55000)))
        weekly = 40
        for _ in range(int(self.rng.integers(4, 9))):
            self._session_online(p, self._time(int(self.rng.integers(0, DAYS - 8)), 18, 23),
                                 self._r(10, weekly), game="Slots")
        dev = self._device()
        region = self._pick(["VPN/Proxy", "Outside Canada"])
        ip = self._ip(region)
        new_cards = [self._card("VISA") for _ in range(int(self.rng.integers(2, 4)))]
        p.cards += new_cards
        for _ in range(int(self.rng.integers(3, 6))):
            ts = self._time(int(self.rng.integers(DAYS - 6, DAYS)), 0, 6)
            amt = self._r(1500, 6000)
            card = self._pick(new_cards)
            self._emit(p, ts, "OLG.ca", "DEPOSIT", amt, method="Credit Card", funding=card,
                       device=dev, ip=ip, region=region)
            self._emit(p, ts + timedelta(minutes=10), "OLG.ca", "WAGER", amt * self._r(0.1, 0.3),
                       game="Live Baccarat", risk=round(self._r(0.3, 0.7), 2), device=dev, ip=ip, region=region)
            self._emit(p, ts + timedelta(hours=self._r(1, 6)), "OLG.ca", "WITHDRAWAL", amt * self._r(0.75, 0.9),
                       method="EFT", funding="BANK-****" + str(self.rng.integers(10000)).zfill(4),
                       device=dev, ip=ip, region=region)

    def _pep_high_value(self):
        p = self._new_player("pep_high_value", primary_channel="Casino", pep=True,
                             occupation="Senior Government Official", est_annual_income=int(self._r(140000, 220000)))
        for _ in range(int(self.rng.integers(3, 6))):
            ts = self._time(hour_lo=18, hour_hi=23)
            amt = self._r(6000, 18000)
            casino = self._pick(CASINOS)
            self._emit(p, ts, "Casino", "CASH_BUY_IN", amt, venue=casino, method="Bank Draft", funding="BANK DRAFT")
            self._emit(p, ts + timedelta(minutes=20), "Casino", "WAGER", amt * self._r(0.3, 0.9),
                       venue=casino, game="Baccarat", risk=round(self._r(0.6, 1.0), 2))
            self._emit(p, ts + timedelta(hours=3), "Casino", "CASH_OUT", amt * self._r(0.7, 1.0),
                       venue=casino, method="Casino Cheque", funding="CASH")

    def _problem_gambler(self):
        """Not laundering: loss-chasing, late nights. Routed to player protection, not AML."""
        p = self._new_player("player_protection", primary_channel="OLG.ca")
        for d in range(DAYS - 14, DAYS):
            if self.rng.random() < 0.7:
                ts = START + timedelta(days=d, hours=self._r(1, 4.5))
                dep = self._r(100, 400)
                for k in range(int(self.rng.integers(2, 6))):
                    self._emit(p, ts, "OLG.ca", "DEPOSIT", dep, method="Credit Card", funding=p.cards[0])
                    self._emit(p, ts + timedelta(minutes=5), "OLG.ca", "WAGER", dep * self._r(2.5, 5),
                               game="Instant Win", risk=1.0)
                    ts += timedelta(minutes=self._r(15, 45))
                    dep *= self._r(1.2, 1.7)

    # ------------------------------------------------------------------- build
    def build(self):
        planted = [
            (self._structuring, 12), (self._minimal_play, 12), (self._hedged_betting, 8),
            (self._ticket_discounting, 7), (self._account_takeover, 8), (self._pep_high_value, 3),
            (self._problem_gambler, 10),
        ]
        rings = [5, 4, 6]
        n_planted = sum(n for _, n in planted) + sum(rings)
        for _ in range(max(0, self.n_players - n_planted)):
            self._normal(self._new_player())
        for fn, n in planted:
            for _ in range(n):
                fn()
        for size in rings:
            self._mule_ring(size)

        tx = pd.DataFrame(self.rows, columns=TX_COLUMNS).sort_values("timestamp").reset_index(drop=True)
        tx["tx_id"] = [f"TX-{i + 1:07d}" for i in range(len(tx))]
        players = pd.DataFrame([{
            "player_id": p.player_id, "name": p.name, "age": p.age, "occupation": p.occupation,
            "est_annual_income": p.est_annual_income, "city": p.city,
            "account_created": p.account_created.date(), "kyc_status": p.kyc_status, "pep": p.pep,
            "primary_channel": p.primary_channel, "scenario": p.scenario,
        } for p in self.players])
        return players, tx


def generate(n_players: int = 5000, seed: int = 7):
    """Return (players, transactions) DataFrames."""
    return Generator(n_players, seed).build()


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Write synthetic OLG players.csv and transactions.csv")
    ap.add_argument("--players", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    players, tx = generate(args.players, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    players.to_csv(out / "players.csv", index=False)
    tx.to_csv(out / "transactions.csv", index=False)
    print(f"Wrote {len(players):,} players and {len(tx):,} transactions to {out}/")
