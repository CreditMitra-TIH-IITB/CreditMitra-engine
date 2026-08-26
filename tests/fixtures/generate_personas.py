"""Regenerate the persona fixtures used by the scorecard acceptance test.

Run from the repo root:

    python tests/fixtures/generate_personas.py

Deterministic — a fixed seed per persona, so regenerating produces byte-identical
files and a diff means someone changed the generator, not the RNG.

WHY SIX MONTHS. The first generation of these fixtures spanned two months, which
was too short for the signals the scorecard is built on to move at all:

  * L4 (commitment) needs a merchant paid on a near-monthly cadence. Over two
    months nothing clears that bar, so L4 read ~3/100 for five of six personas
    and the heaviest-weighted index in the model behaved like a constant.
  * L1 (essential stability) measures month-over-month persistence. With two
    months there is exactly one month-pair to compare.
  * The cash-flow block never went negative, because no persona had a bounce or
    a losing month. It contributed +105..+199 to every persona — a flat bias
    rather than a discriminator — and nothing could score below ~669.

Each persona below therefore runs six months (Oct 2025 - Mar 2026) and the set
deliberately spans the failure modes: bounced payments, negative net cash flow,
a shrinking balance, and irregular payment dates.
"""

from __future__ import annotations

import calendar
import json
import random
from pathlib import Path

OUT_DIR = Path(__file__).parent / "personas"

MONTHS: list[tuple[int, int]] = [
    (2025, 10),
    (2025, 11),
    (2025, 12),
    (2026, 1),
    (2026, 2),
    (2026, 3),
]


def inr(value: float) -> str:
    """Indian digit grouping: 1,23,456.78 — the format parse_amount expects."""
    sign = "-" if value < 0 else ""
    whole, _, frac = f"{abs(value):.2f}".partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join([*groups, tail])
    return f"{sign}{whole}.{frac}"


def mon(month: int) -> str:
    return calendar.month_abbr[month].upper()


class Statement:
    """Accumulates rows in date order, tracking a running balance the way a
    real statement does."""

    def __init__(self, opening_balance: float, seed: int) -> None:
        self.balance = opening_balance
        self.rng = random.Random(seed)
        self._events: list[tuple[tuple[int, int, int], dict]] = []

    def _add(
        self,
        year: int,
        month: int,
        day: int,
        narration: str,
        payee: str,
        amount: float,
        direction: str,
        payee_type: str | None,
        enrichment: dict,
    ) -> None:
        day = min(day, calendar.monthrange(year, month)[1])
        self._events.append(
            (
                (year, month, day),
                {
                    "narration": narration,
                    "payee": payee,
                    "amount": round(amount, 2),
                    "direction": direction,
                    "payee_type": payee_type,
                    "enrichment": enrichment,
                },
            )
        )

    def credit(
        self,
        year: int,
        month: int,
        day: int,
        narration: str,
        payee: str,
        amount: float,
        payee_type: str | None = "person",
        **enrichment: object,
    ) -> None:
        self._add(year, month, day, narration, payee, amount, "credit", payee_type, enrichment)

    def debit(
        self,
        year: int,
        month: int,
        day: int,
        narration: str,
        payee: str,
        amount: float,
        payee_type: str | None = "merchant",
        **enrichment: object,
    ) -> None:
        self._add(year, month, day, narration, payee, amount, "debit", payee_type, enrichment)

    def rows(self) -> list[dict]:
        out: list[dict] = []
        for (year, month, day), ev in sorted(self._events, key=lambda e: e[0]):
            amount = ev["amount"]
            if ev["direction"] == "credit":
                self.balance += amount
                deposits, withdrawals = inr(amount), ""
            else:
                self.balance -= amount
                deposits, withdrawals = "", inr(amount)

            enr = ev["enrichment"]
            out.append(
                {
                    "date": f"{day:02d}-{month:02d}-{year}",
                    "particulars": ev["narration"],
                    "deposits": deposits,
                    "withdrawals": withdrawals,
                    "balance": inr(self.balance),
                    "payee": ev["payee"],
                    "payee_type": ev["payee_type"],
                    "payee_confidence": 0.95 if ev["payee_type"] else None,
                    "txn_date": f"{year}-{month:02d}-{day:02d}",
                    "amount": amount,
                    "direction": ev["direction"],
                    "balance_val": round(self.balance, 2),
                    "category": enr.get("category"),
                    "is_essential": enr.get("is_essential"),
                    "risk_flag": enr.get("risk_flag"),
                    "lifestyle_dim": enr.get("lifestyle_dim"),
                    "recurring_type": enr.get("recurring_type"),
                }
            )
        return out


# ---------------------------------------------------------------------------
# Shared spending blocks
# ---------------------------------------------------------------------------

GROCERIES = dict(
    category="groceries", is_essential=True, lifestyle_dim="essential", recurring_type="adhoc"
)
TELECOM = dict(
    category="telecom", is_essential=True, lifestyle_dim="essential", recurring_type="subscription"
)
UTILITIES = dict(
    category="utilities", is_essential=True, lifestyle_dim="essential", recurring_type="adhoc"
)
FUEL = dict(category="fuel", is_essential=True, lifestyle_dim="essential", recurring_type="adhoc")
FOOD_DELIVERY = dict(
    category="food_delivery",
    is_essential=False,
    lifestyle_dim="aspirational",
    recurring_type="adhoc",
)
SHOPPING = dict(
    category="shopping", is_essential=False, lifestyle_dim="aspirational", recurring_type="adhoc"
)
ENTERTAINMENT = dict(
    category="entertainment",
    is_essential=False,
    lifestyle_dim="aspirational",
    recurring_type="subscription",
)
TRAVEL = dict(
    category="travel", is_essential=False, lifestyle_dim="aspirational", recurring_type="adhoc"
)
QUICK_COMMERCE = dict(
    category="quick_commerce",
    is_essential=False,
    lifestyle_dim="aspirational",
    recurring_type="adhoc",
)
INSURANCE = dict(
    category="insurance", is_essential=True, lifestyle_dim="commitment", recurring_type="emi_like"
)
INVESTMENTS = dict(
    category="investments",
    is_essential=False,
    lifestyle_dim="commitment",
    recurring_type="emi_like",
)
BNPL = dict(
    category="bnpl_lending",
    is_essential=False,
    risk_flag="bnpl_lending",
    lifestyle_dim="leverage",
    recurring_type="emi_like",
)
GAMBLING = dict(
    category="gambling",
    is_essential=False,
    risk_flag="gambling",
    lifestyle_dim="risk",
    recurring_type="adhoc",
)
CRYPTO = dict(
    category="crypto",
    is_essential=False,
    risk_flag="crypto",
    lifestyle_dim="risk",
    recurring_type="adhoc",
)
UNRESOLVED = dict(
    category="other", is_essential=False, lifestyle_dim="neutral", recurring_type="adhoc"
)
GIG_PAYOUT = dict(
    category="food_delivery",
    is_essential=False,
    lifestyle_dim="neutral",
    recurring_type="payout_source",
)


def rent(st: Statement, y: int, m: int, day: int, amount: float, landlord: str) -> None:
    """Rent goes to a private landlord, so it arrives from the real pipeline as
    an unenriched person-payee row — no merchant tier can ever resolve it.
    Recognising it is feature_engineering's job, not the fixture's."""
    st.debit(
        y,
        m,
        day,
        f"UPI/DR/5001{day:02d}{m:02d}/LANDLORD {landlord.upper()}/RENT {mon(m)}",
        landlord,
        amount,
        payee_type="person",
    )


def bounce(st: Statement, y: int, m: int, day: int, what: str, charge: float = 590.0) -> None:
    st.debit(
        y,
        m,
        day,
        f"ECS RET CHRG/{what}/INSUFFICIENT FUNDS",
        "",
        charge,
        payee_type=None,
    )


def atm(st: Statement, y: int, m: int, day: int, amount: float) -> None:
    st.debit(
        y,
        m,
        day,
        f"ATM CASH WDL/SBI ATM {1000 + day}/SELF",
        "",
        amount,
        payee_type=None,
    )


# ---------------------------------------------------------------------------
# Personas
# ---------------------------------------------------------------------------


def salaried_saver() -> list[dict]:
    """Steady salary, three sustained commitments paid on the same day every
    month, comfortably positive cash flow. The upper anchor."""
    st = Statement(opening_balance=48_000, seed=101)
    for y, m in MONTHS:
        st.credit(
            y,
            m,
            1,
            f"UPI/CR/5001{m:02d}01/ACME SOFTWARE PVT LTD/SALARY {mon(m)}",
            "Acme Software Pvt Ltd",
            62_000,
        )
        rent(st, y, m, 2, 15_000, "Ravi Kumar")
        st.debit(y, m, 3, f"UPI/DR/5001{m:02d}03/LIC OF INDIA/PREMIUM", "LIC", 4_200, **INSURANCE)
        st.debit(
            y, m, 4, f"UPI/DR/5001{m:02d}04/GROWW INVEST TECH/SIP", "Groww", 8_000, **INVESTMENTS
        )
        st.debit(y, m, 5, f"UPI/DR/5001{m:02d}05/JIO RECHARGE/PREPAID", "Jio", 399, **TELECOM)
        st.debit(y, m, 7, f"UPI/DR/5001{m:02d}07/MSEB ELECTRICITY/BILL", "MSEB", 1_240, **UTILITIES)
        st.debit(y, m, 9, f"UPI/DR/5001{m:02d}09/BIGBASKET/ORDER", "BigBasket", 2_650, **GROCERIES)
        st.debit(y, m, 19, f"UPI/DR/5001{m:02d}19/BIGBASKET/ORDER", "BigBasket", 2_380, **GROCERIES)
        st.debit(
            y,
            m,
            12,
            f"UPI/DR/5001{m:02d}12/NETFLIX INDIA/SUBSCRIPTION",
            "Netflix",
            649,
            **ENTERTAINMENT,
        )
        st.debit(y, m, 15, f"UPI/DR/5001{m:02d}15/SWIGGY/ORDER", "Swiggy", 480, **FOOD_DELIVERY)
        st.debit(
            y, m, 22, f"UPI/DR/5001{m:02d}22/AMAZON PAY INDIA/ORDER", "Amazon", 1_500, **SHOPPING
        )
    return st.rows()


def gig_hustler() -> list[dict]:
    """No single employer — income is platform payouts, irregular in both size
    and timing. Spending is disciplined and entirely digital, and rent is paid
    every month but never on the same date."""
    st = Statement(opening_balance=12_000, seed=202)
    rent_days = [3, 7, 5, 9, 4, 8]
    for i, (y, m) in enumerate(MONTHS):
        for day, platform, base in (
            (4, "SWIGGY", 7_400),
            (11, "ZOMATO", 5_900),
            (18, "SWIGGY", 6_800),
            (25, "UBER INDIA", 4_600),
        ):
            payee = platform.title().replace(" India", "")
            amount = base + st.rng.randint(-1_400, 1_800)
            st.credit(
                y,
                m,
                day,
                f"UPI/CR/5002{m:02d}{day:02d}/{platform}/WEEKLY PAYOUT",
                payee,
                amount,
                payee_type="merchant",
                **GIG_PAYOUT,
            )
        rent(st, y, m, rent_days[i], 9_000, "Sunita Desai")
        st.debit(y, m, 6, f"UPI/DR/5002{m:02d}06/JIO RECHARGE/PREPAID", "Jio", 299, **TELECOM)
        st.debit(y, m, 8, f"UPI/DR/5002{m:02d}08/HP PETROL PUMP/FUEL", "HP Petrol", 2_600, **FUEL)
        st.debit(y, m, 20, f"UPI/DR/5002{m:02d}20/HP PETROL PUMP/FUEL", "HP Petrol", 2_400, **FUEL)
        st.debit(y, m, 10, f"UPI/DR/5002{m:02d}10/DMART/GROCERY", "DMart", 3_500, **GROCERIES)
        st.debit(y, m, 14, f"UPI/DR/5002{m:02d}14/SWIGGY/ORDER", "Swiggy", 420, **FOOD_DELIVERY)
        st.debit(y, m, 24, f"UPI/DR/5002{m:02d}24/MSEB ELECTRICITY/BILL", "MSEB", 980, **UTILITIES)
        if i >= 2:  # starts a SIP once income stabilises
            st.debit(
                y,
                m,
                16,
                f"UPI/DR/5002{m:02d}16/GROWW INVEST TECH/SIP",
                "Groww",
                2_000,
                **INVESTMENTS,
            )
    return st.rows()


def bnpl_heavy_spender() -> list[dict]:
    """Salaried, but four buy-now-pay-later apps carry a third of monthly spend.
    Thin buffer, occasional bounce, slightly negative cash flow."""
    st = Statement(opening_balance=9_000, seed=303)
    for i, (y, m) in enumerate(MONTHS):
        st.credit(
            y,
            m,
            1,
            f"UPI/CR/5003{m:02d}01/VERTEX RETAIL LLP/SALARY {mon(m)}",
            "Vertex Retail Llp",
            38_000,
        )
        rent(st, y, m, 3, 12_000, "Mahesh Iyer")
        st.debit(y, m, 5, f"UPI/DR/5003{m:02d}05/SIMPL/BILL PAYMENT", "Simpl", 4_500, **BNPL)
        st.debit(y, m, 7, f"UPI/DR/5003{m:02d}07/LAZYPAY/DUE", "LazyPay", 3_800, **BNPL)
        st.debit(y, m, 12, f"UPI/DR/5003{m:02d}12/KREDITBEE/EMI", "KreditBee", 5_200, **BNPL)
        st.debit(y, m, 15, f"UPI/DR/5003{m:02d}15/SLICE/DUE", "slice", 2_900, **BNPL)
        st.debit(
            y, m, 9, f"UPI/DR/5003{m:02d}09/FLIPKART INTERNET/ORDER", "Flipkart", 3_200, **SHOPPING
        )
        st.debit(y, m, 18, f"UPI/DR/5003{m:02d}18/MYNTRA/ORDER", "Myntra", 2_600, **SHOPPING)
        st.debit(y, m, 21, f"UPI/DR/5003{m:02d}21/ZOMATO/ORDER", "Zomato", 2_400, **FOOD_DELIVERY)
        st.debit(y, m, 11, f"UPI/DR/5003{m:02d}11/DMART/GROCERY", "DMart", 2_800, **GROCERIES)
        st.debit(y, m, 6, f"UPI/DR/5003{m:02d}06/AIRTEL/POSTPAID", "Airtel", 599, **TELECOM)
        if i in (2, 4):
            bounce(st, y, m, 27, "KREDITBEE")
    return st.rows()


def gambler() -> list[dict]:
    """Salaried, but a large and growing share of spend goes to betting and
    crypto. Balance drains, payments start bouncing."""
    st = Statement(opening_balance=22_000, seed=404)
    for i, (y, m) in enumerate(MONTHS):
        st.credit(
            y,
            m,
            1,
            f"UPI/CR/5004{m:02d}01/NORTHGATE LOGISTICS/SALARY {mon(m)}",
            "Northgate Logistics",
            45_000,
        )
        rent(st, y, m, 4, 10_000, "Prakash Nair")
        escalation = 1.0 + 0.25 * i  # the habit grows over the six months
        for day, platform, base, enr in (
            (6, "DREAM11", 6_500, GAMBLING),
            (13, "DREAM11", 5_200, GAMBLING),
            (19, "MPL GAMING", 4_100, GAMBLING),
            (23, "WAZIRX", 5_800, CRYPTO),
        ):
            st.debit(
                y,
                m,
                day,
                f"UPI/DR/5004{m:02d}{day:02d}/{platform}/DEPOSIT",
                platform.title().replace(" Gaming", ""),
                round(base * escalation),
                **enr,
            )
        st.debit(y, m, 10, f"UPI/DR/5004{m:02d}10/DMART/GROCERY", "DMart", 2_400, **GROCERIES)
        st.debit(y, m, 8, f"UPI/DR/5004{m:02d}08/JIO RECHARGE/PREPAID", "Jio", 399, **TELECOM)
        if i >= 2:
            bounce(st, y, m, 26, "RENT ECS")
        if i >= 4:
            bounce(st, y, m, 28, "DREAM11 MANDATE")
    return st.rows()


def aspirational_overspender() -> list[dict]:
    """Salary is fine, but discretionary spend eats it. No commitments at all —
    no SIP, no insurance — and the buffer never builds."""
    st = Statement(opening_balance=14_000, seed=505)
    for i, (y, m) in enumerate(MONTHS):
        st.credit(
            y,
            m,
            1,
            f"UPI/CR/5005{m:02d}01/BLUEPEAK MEDIA/SALARY {mon(m)}",
            "Bluepeak Media",
            40_000,
        )
        rent(st, y, m, 3, 11_000, "Anita Shah")
        st.debit(y, m, 6, f"UPI/DR/5005{m:02d}06/SWIGGY/ORDER", "Swiggy", 2_900, **FOOD_DELIVERY)
        st.debit(y, m, 13, f"UPI/DR/5005{m:02d}13/ZOMATO/ORDER", "Zomato", 2_700, **FOOD_DELIVERY)
        st.debit(y, m, 21, f"UPI/DR/5005{m:02d}21/SWIGGY/ORDER", "Swiggy", 2_500, **FOOD_DELIVERY)
        st.debit(y, m, 9, f"UPI/DR/5005{m:02d}09/MYNTRA/ORDER", "Myntra", 6_200, **SHOPPING)
        st.debit(
            y, m, 17, f"UPI/DR/5005{m:02d}17/AMAZON PAY INDIA/ORDER", "Amazon", 5_300, **SHOPPING
        )
        st.debit(y, m, 11, f"UPI/DR/5005{m:02d}11/ZEPTO/ORDER", "Zepto", 1_600, **QUICK_COMMERCE)
        st.debit(y, m, 24, f"UPI/DR/5005{m:02d}24/ZEPTO/ORDER", "Zepto", 1_400, **QUICK_COMMERCE)
        st.debit(
            y,
            m,
            15,
            f"UPI/DR/5005{m:02d}15/BOOKMYSHOW/TICKETS",
            "BookMyShow",
            1_900,
            **ENTERTAINMENT,
        )
        st.debit(y, m, 7, f"UPI/DR/5005{m:02d}07/AIRTEL/POSTPAID", "Airtel", 749, **TELECOM)
        st.debit(y, m, 19, f"UPI/DR/5005{m:02d}19/DMART/GROCERY", "DMart", 1_500, **GROCERIES)
        if i in (1, 3, 4):
            st.debit(
                y, m, 26, f"UPI/DR/5005{m:02d}26/MAKEMYTRIP/BOOKING", "MakeMyTrip", 8_500, **TRAVEL
            )
        if i == 4:
            bounce(st, y, m, 28, "AIRTEL POSTPAID")
    return st.rows()


def cash_reliant_informal() -> list[dict]:
    """Earns in cash, withdraws in cash, spends where nothing resolves. Little
    digital footprint for any merchant-derived model to read."""
    st = Statement(opening_balance=7_500, seed=606)
    for _i, (y, m) in enumerate(MONTHS):
        for day, base in ((2, 9_000), (9, 7_500), (17, 10_000), (25, 6_500)):
            st.credit(
                y,
                m,
                day,
                f"CASH DEP/BRANCH 0421/SELF {mon(m)}",
                "Self",
                base + st.rng.randint(-1_500, 2_500),
                payee_type="person",
            )
        atm(st, y, m, 4, 9_000)
        atm(st, y, m, 12, 7_000)
        atm(st, y, m, 20, 7_500)
        atm(st, y, m, 27, 6_000)
        st.debit(y, m, 6, f"UPI/DR/5006{m:02d}06/JIO RECHARGE/PREPAID", "Jio", 239, **TELECOM)
        st.debit(
            y,
            m,
            14,
            f"UPI/DR/5006{m:02d}14/SRI BALAJI STORES/PAYMENT",
            "Sri Balaji Stores",
            2_200,
            **UNRESOLVED,
        )
        st.debit(
            y,
            m,
            22,
            f"UPI/DR/5006{m:02d}22/NEW ARIHANT TRADERS/PAYMENT",
            "New Arihant Traders",
            1_800,
            **UNRESOLVED,
        )
    return st.rows()


def frequent_bouncer() -> list[dict]:
    """The persona the old two-month set had no answer for: lifestyle looks
    unremarkable — no gambling, no BNPL, ordinary spending — and the score has
    to come down on cash-flow evidence alone. Six bounced payments, a balance
    that keeps touching zero, and more going out than coming in."""
    st = Statement(opening_balance=4_200, seed=707)
    for i, (y, m) in enumerate(MONTHS):
        st.credit(
            y,
            m,
            1,
            f"UPI/CR/5007{m:02d}01/HARIOM TRADERS/SALARY {mon(m)}",
            "Hariom Traders",
            35_000,
        )
        rent(st, y, m, 5, 13_000, "Deepak Joshi")
        st.debit(
            y,
            m,
            8,
            f"UPI/DR/5007{m:02d}08/BAJAJ FINANCE/EMI",
            "Bajaj Finance",
            7_500,
            category="loan_emi",
            is_essential=False,
            lifestyle_dim="commitment",
            recurring_type="emi_like",
        )
        st.debit(y, m, 11, f"UPI/DR/5007{m:02d}11/DMART/GROCERY", "DMart", 4_200, **GROCERIES)
        st.debit(
            y, m, 16, f"UPI/DR/5007{m:02d}16/MSEB ELECTRICITY/BILL", "MSEB", 1_450, **UTILITIES
        )
        st.debit(y, m, 6, f"UPI/DR/5007{m:02d}06/AIRTEL/POSTPAID", "Airtel", 649, **TELECOM)
        st.debit(y, m, 19, f"UPI/DR/5007{m:02d}19/SWIGGY/ORDER", "Swiggy", 900, **FOOD_DELIVERY)
        st.debit(
            y,
            m,
            22,
            f"UPI/DR/5007{m:02d}22/FLIPKART INTERNET/ORDER",
            "Flipkart",
            2_600,
            **SHOPPING,
        )
        atm(st, y, m, 24, 4_000)
        # A bounce most months, twice in the worst two.
        bounce(st, y, m, 27, "BAJAJ FINANCE EMI")
        if i in (3, 5):
            bounce(st, y, m, 28, "AIRTEL POSTPAID")
    return st.rows()


def balanced_salaried() -> list[dict]:
    """The ordinary middle. Nothing wrong and nothing exceptional: a salary
    that covers the month with a little left over, rent that gets paid but not
    always on the same date, no savings habit, no red flags, and a cushion of
    about two weeks. Without a persona like this the whole set is extremes, and
    a scorecard that only ever sees extremes can look well-calibrated while
    leaving the middle of its own range untested."""
    st = Statement(opening_balance=8_000, seed=808)
    rent_days = [3, 6, 4, 9, 5, 11]
    for i, (y, m) in enumerate(MONTHS):
        st.credit(
            y,
            m,
            1,
            f"UPI/CR/5008{m:02d}01/KAVERI SYSTEMS PVT LTD/SALARY {mon(m)}",
            "Kaveri Systems Pvt Ltd",
            32_000,
        )
        rent(st, y, m, rent_days[i], 8_000, "Girish Menon")
        st.debit(y, m, 5, f"UPI/DR/5008{m:02d}05/AIRTEL/POSTPAID", "Airtel", 599, **TELECOM)
        st.debit(
            y,
            m,
            8,
            f"UPI/DR/5008{m:02d}08/MSEB ELECTRICITY/BILL",
            "MSEB",
            1_100 + st.rng.randint(-150, 250),
            **UTILITIES,
        )
        st.debit(
            y,
            m,
            10,
            f"UPI/DR/5008{m:02d}10/DMART/GROCERY",
            "DMart",
            2_300 + st.rng.randint(-300, 400),
            **GROCERIES,
        )
        st.debit(
            y,
            m,
            21,
            f"UPI/DR/5008{m:02d}21/BIGBASKET/ORDER",
            "BigBasket",
            1_700 + st.rng.randint(-250, 350),
            **GROCERIES,
        )
        st.debit(
            y,
            m,
            13,
            f"UPI/DR/5008{m:02d}13/SWIGGY/ORDER",
            "Swiggy",
            1_500 + st.rng.randint(-400, 600),
            **FOOD_DELIVERY,
        )
        st.debit(
            y,
            m,
            17,
            f"UPI/DR/5008{m:02d}17/AMAZON PAY INDIA/ORDER",
            "Amazon",
            3_800 + st.rng.randint(-900, 1_200),
            **SHOPPING,
        )
        st.debit(
            y,
            m,
            12,
            f"UPI/DR/5008{m:02d}12/NETFLIX INDIA/SUBSCRIPTION",
            "Netflix",
            649,
            **ENTERTAINMENT,
        )
        st.debit(
            y,
            m,
            9,
            f"UPI/DR/5008{m:02d}09/HP PETROL PUMP/FUEL",
            "HP Petrol",
            2_000 + st.rng.randint(-300, 400),
            **FUEL,
        )
        atm(st, y, m, 23, 3_000)
        # Local shops and one-off payments that never resolve to a known
        # merchant — the long tail every real statement has.
        st.debit(
            y,
            m,
            19,
            f"UPI/DR/5008{m:02d}19/SHREE MEDICAL AND GEN/PAYMENT",
            "Shree Medical And Gen",
            2_400 + st.rng.randint(-400, 900),
            **UNRESOLVED,
        )
        st.debit(
            y,
            m,
            26,
            f"UPI/DR/5008{m:02d}26/RAJESH KIRANA STORE/PAYMENT",
            "Rajesh Kirana Store",
            1_600 + st.rng.randint(-300, 600),
            **UNRESOLVED,
        )
    return st.rows()


PERSONAS = {
    "01_salaried_saver": salaried_saver,
    "02_gig_hustler": gig_hustler,
    "03_bnpl_heavy_spender": bnpl_heavy_spender,
    "04_gambler": gambler,
    "05_aspirational_overspender": aspirational_overspender,
    "06_cash_reliant_informal": cash_reliant_informal,
    "07_frequent_bouncer": frequent_bouncer,
    "08_balanced_salaried": balanced_salaried,
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, builder in PERSONAS.items():
        rows = builder()
        path = OUT_DIR / f"{name}.json"
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        credits = sum(r["amount"] for r in rows if r["direction"] == "credit")
        debits = sum(r["amount"] for r in rows if r["direction"] == "debit")
        print(
            f"{name:<32} {len(rows):>4} rows  "
            f"in {credits:>11,.0f}  out {debits:>11,.0f}  net {credits - debits:>+11,.0f}  "
            f"closing {rows[-1]['balance_val']:>10,.0f}"
        )


if __name__ == "__main__":
    main()
