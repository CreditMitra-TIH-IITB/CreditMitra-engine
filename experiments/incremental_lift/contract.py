"""The dataset contract every adapter produces.

An adapter's only job is to turn some external dataset into a list of
`LabelledStatement` — a borrower's transactions plus whether they defaulted.
Everything downstream (featurisation, modelling, reporting) is dataset-agnostic,
so adding a new source means writing one adapter and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas.statements import Transaction


@dataclass(frozen=True)
class LabelledStatement:
    """One borrower: their statement rows, and the observed outcome.

    `defaulted` is the target. It must come from the dataset, never from
    anything this repo computes — the whole point of the experiment is to check
    our features against an outcome we did not author.
    """

    borrower_id: str
    transactions: list[Transaction]
    defaulted: bool
    meta: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.transactions:
            raise ValueError(f"{self.borrower_id}: a statement with no rows cannot be scored")


@dataclass(frozen=True)
class Dataset:
    name: str
    statements: list[LabelledStatement]
    #: Stated plainly so it reaches the write-up rather than dying in a docstring.
    caveats: tuple[str, ...] = ()

    @property
    def n(self) -> int:
        return len(self.statements)

    @property
    def n_defaults(self) -> int:
        return sum(1 for s in self.statements if s.defaulted)

    @property
    def default_rate(self) -> float:
        return self.n_defaults / self.n if self.n else 0.0

    def summary(self) -> str:
        return (
            f"{self.name}: {self.n} borrowers, {self.n_defaults} defaults ({self.default_rate:.1%})"
        )
