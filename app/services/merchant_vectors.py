"""On-device vector index over the merchant dictionary — the proposal's merchant
vector store (WP6), used as a fuzzy tier behind exact/prefix dictionary lookup.

It catches names that are *near* a known merchant but not a prefix of one: a
typo the payee model introduced ("swigy"), a truncation split mid-word
("connau ght plaza"), a doubled or transposed character. Each dictionary alias
is a character n-gram TF-IDF vector (2-4 grams, word-bounded); a query matches
its nearest alias by cosine similarity when that clears the threshold.

Measured on noisy variants of every dictionary alias that exact/prefix lookup
misses, against real local-shop and person names that must not match:

    threshold 0.75: recovers 67% of the variants, wrong merchant 0.3%,
                    false match on local shops / people 0.3%

High-severity categories (gambling, BNPL/lending, crypto, loan EMI) are left
out of the index entirely: a fuzzy match must never be what tags a borrower as
a gambler.

Why character n-grams, not the Qwen3 embedding the classifier already loads:
on the same sets (250 variants, 250 local shops), at a 0.4% false-match budget
the n-gram index recovered 81.6% of variants and the neural embedding 36.8%,
at 144 ms per name on CPU against microseconds. A sentence-embedding model
encodes what a name *means*; a typo of a brand mostly needs what it is
*spelled* like.

Why no external vector database: the index holds one vector per alias (~500
today, low thousands with curation). Brute-force cosine over that is far under
a millisecond, and the whole index rebuilds in milliseconds when the dictionary
changes, so a separate service would add an operational dependency with no
retrieval benefit at this scale.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.schemas.statements import MerchantEnrichment

_MIN_QUERY_LEN = 4


class MerchantVectorIndex:
    def __init__(
        self,
        entries: Mapping[str, MerchantEnrichment],
        exclude_categories: Iterable[str] = (),
    ) -> None:
        excluded = set(exclude_categories)
        items = [(a, e) for a, e in entries.items() if e.category not in excluded and len(a) >= 3]
        self._enrichments = [e for _, e in items]
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        if items:
            self._vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True
            )
            self._matrix = self._vectorizer.fit_transform([a for a, _ in items])

    def __len__(self) -> int:
        return len(self._enrichments)

    def nearest(self, normalized: str) -> tuple[MerchantEnrichment, float] | None:
        """Closest merchant and its cosine similarity (rows are L2-normalised)."""
        if self._vectorizer is None or self._matrix is None or len(normalized) < _MIN_QUERY_LEN:
            return None
        query = self._vectorizer.transform([normalized])
        if query.nnz == 0:
            return None
        scores = np.asarray((self._matrix @ query.T).todense()).ravel()
        best = int(scores.argmax())
        return self._enrichments[best], float(scores[best])

    def match(self, normalized: str, threshold: float) -> MerchantEnrichment | None:
        hit = self.nearest(normalized)
        if hit is None or hit[1] < threshold:
            return None
        return hit[0]
