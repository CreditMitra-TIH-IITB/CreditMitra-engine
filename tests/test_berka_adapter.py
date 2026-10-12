"""The Berka adapter must give the model only what a lender saw at decision
time. Without the cutoff, the outcome leaks through post-loan balances."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from experiments.incremental_lift import berka

TRANS_HEADER = ";".join(
    f'"{c}"'
    for c in (
        "trans_id",
        "account_id",
        "date",
        "type",
        "operation",
        "amount",
        "balance",
        "k_symbol",
        "bank",
        "account",
    )
)
LOAN_HEADER = '"loan_id";"account_id";"date";"amount";"duration";"payments";"status"'


def _trans(tid: int, account: str, yymmdd: str, balance: float) -> str:
    return f'{tid};{account};{yymmdd};"PRIJEM";"VKLAD";1000.0;{balance};"";"";""'


def _write(tmp_path: Path, loans: list[str], trans: list[str]) -> Path:
    (tmp_path / "trans.asc").write_text(
        "\n".join([TRANS_HEADER, *trans]) + "\n", encoding="latin-1"
    )
    (tmp_path / "loan.asc").write_text("\n".join([LOAN_HEADER, *loans]) + "\n", encoding="latin-1")
    return tmp_path


def _history(account: str, start_tid: int, months: range, year: int, balance: float) -> list[str]:
    return [
        _trans(start_tid + i, account, f"{year % 100:02d}{m:02d}{day:02d}", balance)
        for i, (m, day) in enumerate((m, d) for m in months for d in (5, 15, 25))
    ]


def test_only_transactions_before_the_loan_are_used(tmp_path: Path) -> None:
    # 24 healthy rows in 1995, then the account runs negative after the loan
    # starts on 1996-01-01 - the leak the cutoff exists to stop.
    before = _history("1", 1, range(1, 9), 1995, balance=5000.0)
    after = _history("1", 100, range(1, 9), 1996, balance=-2000.0)
    data = _write(tmp_path, ['1;1;960101;50000;12;4000;"B"'], before + after)

    dataset = berka.load(data, min_transactions=20)

    (statement,) = dataset.statements
    assert statement.defaulted
    assert all(t.txn_date and t.txn_date < date(1996, 1, 1) for t in statement.transactions)
    assert all((t.balance_val or 0) > 0 for t in statement.transactions)


def test_a_loan_without_enough_pre_loan_history_is_skipped(tmp_path: Path) -> None:
    after_only = _history("1", 1, range(1, 9), 1996, balance=100.0)
    data = _write(tmp_path, ['1;1;960101;50000;12;4000;"A"'], after_only)
    assert berka.load(data, min_transactions=20).statements == []


def test_finished_only_keeps_a_versus_b(tmp_path: Path) -> None:
    trans = _history("1", 1, range(1, 9), 1995, 5000.0) + _history(
        "2", 200, range(1, 9), 1995, 5000.0
    )
    loans = ['1;1;960101;50000;12;4000;"A"', '2;2;960101;50000;12;4000;"D"']
    data = _write(tmp_path, loans, trans)

    assert len(berka.load(data, min_transactions=20).statements) == 2
    finished = berka.load(data, min_transactions=20, finished_only=True).statements
    assert [s.meta["status"] for s in finished] == ["A"]
