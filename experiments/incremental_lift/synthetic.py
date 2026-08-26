"""Controlled datasets that validate the measuring instrument.

These do NOT support any claim about real borrowers — the outcome is drawn from
a process this file defines, so measuring lift on it only proves the harness can
recover a signal it was told to plant. That is precisely what makes them useful:
before pointing the experiment at real data, we need to know it reports lift when
lift exists and reports nothing when it does not.

Two controls:

  signal mode   default probability genuinely depends on gambling/BNPL exposure,
                which only Arm C can observe. A working harness must find
                positive lift for C over A.

  null mode     default probability depends only on cash-flow stress; merchant
                categories are assigned at random and carry no information. A
                working harness must report a delta indistinguishable from zero.

The null control is the important one. Arm C has ~25 more columns than Arm A, and
any evaluation that leaks — a shared standardiser, in-fold scoring, an
unregularised fit — will manufacture lift out of pure noise. If null mode shows a
significant delta, the harness is broken and every number it produces is void.
"""

from __future__ import annotations

import calendar
from datetime import date

import numpy as np

from app.schemas.statements import Transaction

from .contract import Dataset, LabelledStatement

MONTHS = [(2025, 10), (2025, 11), (2025, 12), (2026, 1), (2026, 2), (2026, 3)]

# Latent driver distributions. Wide enough that borrowers genuinely differ —
# an earlier version drew exposure so tightly that gambling_share had a
# univariate AUC of 0.56, i.e. the "positive" control planted a signal too faint
# for any honest evaluator to recover, and then blamed the evaluator.
_STRESS_BETA = (2.0, 5.0)
#: Fraction of the discretionary block that goes to gambling/BNPL merchants.
_RISK_SHARE_BETA = (1.6, 3.0)

_E_STRESS = _STRESS_BETA[0] / sum(_STRESS_BETA)
_E_RISK_SHARE = _RISK_SHARE_BETA[0] / sum(_RISK_SHARE_BETA)

#: log(0.15 / 0.85) — a ~15% base rate, in the range real portfolios sit in.
_INTERCEPT = -1.7346

_ESSENTIAL = {"is_essential": True, "lifestyle_dim": "essential", "recurring_type": "adhoc"}
_DISCRETIONARY = {
    "is_essential": False,
    "lifestyle_dim": "aspirational",
    "recurring_type": "adhoc",
}
_BNPL = {
    "is_essential": False,
    "risk_flag": "bnpl_lending",
    "lifestyle_dim": "leverage",
    "recurring_type": "emi_like",
}
_GAMBLING = {
    "is_essential": False,
    "risk_flag": "gambling",
    "lifestyle_dim": "risk",
    "recurring_type": "adhoc",
}


def _row(
    d: date,
    narration: str,
    payee: str,
    amount: float,
    direction: str,
    balance: float,
    payee_type: str | None = "merchant",
    category: str | None = None,
    **enrichment: object,
) -> Transaction:
    return Transaction(
        date=d.strftime("%d-%m-%Y"),
        particulars=narration,
        deposits=f"{amount:.2f}" if direction == "credit" else "",
        withdrawals=f"{amount:.2f}" if direction == "debit" else "",
        balance=f"{balance:.2f}",
        payee=payee,
        payee_type=payee_type,
        payee_confidence=0.95 if payee_type else None,
        txn_date=d,
        amount=round(amount, 2),
        direction=direction,
        balance_val=round(balance, 2),
        category=category,
        **enrichment,  # type: ignore[arg-type]
    )


def _day(year: int, month: int, day: int) -> date:
    return date(year, month, min(day, calendar.monthrange(year, month)[1]))


def _make_borrower(rng: np.random.Generator, risk_share: float, stress: float) -> list[Transaction]:
    """One six-month statement.

    The discretionary block is the crux. Every borrower spends the same
    distribution of rupees on the same dates; `risk_share` decides only what
    fraction of those payments go to a gambling/BNPL merchant rather than a
    retail one. Amounts and timing are drawn independently of risk_share, so the
    cash-flow footprint is identical either way and only an arm that can read
    the merchant can tell two such borrowers apart.

    This matters. A first version scaled gambling spend *on top of* normal
    spending, which drained the balance and let the cash-flow-only arm reach
    AUC 0.856 on a signal it was supposed to be blind to. That is a true and
    useful observation about real statements — money leaving the account is
    visible however it is labelled — but it makes for a worthless control.

    `stress` drives bounces and balance level, and is visible to every arm.
    """
    income = float(rng.uniform(28_000, 65_000))
    rent = income * float(rng.uniform(0.18, 0.32))
    balance = income * float(rng.uniform(0.15, 1.6)) * (1.4 - stress)
    discretionary_budget = income * float(rng.uniform(0.15, 0.35))
    rows: list[Transaction] = []

    for year, month in MONTHS:
        balance += income
        rows.append(
            _row(
                _day(year, month, 1),
                f"UPI/CR/EMPLOYER PVT LTD/SALARY {calendar.month_abbr[month].upper()}",
                "Employer Pvt Ltd",
                income,
                "credit",
                balance,
                payee_type="person",
            )
        )

        balance -= rent
        rows.append(
            _row(
                _day(year, month, 3),
                "UPI/DR/LANDLORD/RENT",
                "Landlord",
                rent,
                "debit",
                balance,
                payee_type="person",
            )
        )

        for day, name, cat, frac in (
            (9, "DMart", "groceries", 0.10),
            (6, "Airtel", "telecom", 0.02),
            (14, "MSEB", "utilities", 0.03),
        ):
            amount = income * frac * float(rng.uniform(0.8, 1.2))
            balance -= amount
            rows.append(
                _row(
                    _day(year, month, day),
                    f"UPI/DR/{name.upper()}/BILL",
                    name,
                    amount,
                    "debit",
                    balance,
                    category=cat,
                    **_ESSENTIAL,
                )
            )

        # Fixed discretionary block: same slots, same amounts for everyone.
        # Only the merchant behind each slot changes with risk_share.
        slots = [(7, 0.20), (11, 0.18), (17, 0.17), (19, 0.16), (22, 0.15), (25, 0.14)]
        n_risky = int(round(len(slots) * risk_share))
        for position, (day, weight) in enumerate(slots):
            amount = discretionary_budget * weight * float(rng.uniform(0.85, 1.15))
            balance -= amount
            if position < n_risky:
                # Alternate the two risk merchants so both a gambling and a BNPL
                # footprint appear rather than one dominating.
                if position % 2 == 0:
                    narration, payee, cat, enr = (
                        "UPI/DR/DREAM11/DEPOSIT",
                        "Dream11",
                        "gambling",
                        _GAMBLING,
                    )
                else:
                    narration, payee, cat, enr = (
                        "UPI/DR/SIMPL/BILL PAYMENT",
                        "Simpl",
                        "bnpl_lending",
                        _BNPL,
                    )
            else:
                narration, payee, cat, enr = (
                    "UPI/DR/AMAZON PAY INDIA/ORDER",
                    "Amazon",
                    "shopping",
                    _DISCRETIONARY,
                )
            rows.append(
                _row(
                    _day(year, month, day),
                    narration,
                    payee,
                    amount,
                    "debit",
                    balance,
                    category=cat,
                    **enr,
                )
            )

        # Cash-flow stress shows up as bounces, visible to every arm.
        if rng.random() < stress * 0.7:
            balance -= 590
            rows.append(
                _row(
                    _day(year, month, 27),
                    "ECS RET CHRG/EMI/INSUFFICIENT FUNDS",
                    "",
                    590.0,
                    "debit",
                    balance,
                    payee_type=None,
                )
            )

    return rows


def make_dataset(n: int = 600, mode: str = "signal", seed: int = 20260826) -> Dataset:
    """`mode="signal"`: merchant exposure genuinely predicts default.
    `mode="null"`: it does not, and is assigned independently of the outcome."""
    if mode not in {"signal", "null"}:
        raise ValueError("mode must be 'signal' or 'null'")

    rng = np.random.default_rng(seed)
    statements: list[LabelledStatement] = []

    for i in range(n):
        stress = float(rng.beta(*_STRESS_BETA))
        risk_share = float(rng.beta(*_RISK_SHARE_BETA))

        # Drivers are centred on their expected value so the intercept alone
        # sets the base rate, and the coefficients control discrimination
        # without dragging the default rate around with them.
        stress_c = stress - _E_STRESS
        risk_c = risk_share - _E_RISK_SHARE

        if mode == "signal":
            # Where the discretionary money goes is the dominant driver, and it
            # is visible only to an arm that can resolve the merchant. Failing
            # to find this is unambiguous evidence the harness is broken.
            logit = _INTERCEPT + 1.5 * stress_c + 7.0 * risk_c
        else:
            # Outcome depends on cash-flow stress alone. risk_share still varies
            # borrower to borrower — it is simply irrelevant to the outcome.
            logit = _INTERCEPT + 6.0 * stress_c

        probability = 1.0 / (1.0 + np.exp(-logit))
        defaulted = bool(rng.random() < probability)

        statements.append(
            LabelledStatement(
                borrower_id=f"{mode}-{i:04d}",
                transactions=_make_borrower(rng, risk_share, stress),
                defaulted=defaulted,
            )
        )

    caveats = (
        "Synthetic. The outcome is drawn from a process defined in this file, so "
        "this validates the harness only — it is not evidence about real borrowers.",
        f"Mode={mode}.",
    )
    return Dataset(name=f"synthetic-{mode}", statements=statements, caveats=caveats)
