# Incremental lift — what is merchant information actually worth?

This is WP4 ("making credit-risk models dynamic by adding real-time merchant
information") expressed as a measurement rather than an assertion.

The proposal's product category is **"Improvements to existing Products with
significant value add."** The claim that category obliges us to defend is not
*"our score is good"* — it is *"adding merchant information to a credit-risk
model improves it, by this much."* That is answerable with a public dataset and
someone else's labels. It does not require us to own a loan book.

## The design

Three nested arms, each given a **masked copy of the same transactions** and run
through the engine's real `build_features` / `build_profile`:

| Arm | Sees | Blanked |
|---|---|---|
| **A** `cashflow_only` | amounts, dates, balances, narration text | `payee`, `payee_type`, all enrichment |
| **B** `plus_payee_identity` | + who was paid, merchant vs person | enrichment fields |
| **C** `plus_merchant_info` | everything | — |

Ablating by *masking inputs* rather than selecting feature columns matters: Arm A
cannot leak merchant knowledge because the fields are literally `None`. And
because every arm runs the production feature code, what gets measured is what
the product computes — not a reimplementation that might flatter itself.

Because the arms are nested, the deltas decompose:

- **C − A** — the whole merchant stack
- **B − A** — payee extraction + merchant/non-merchant classification alone
- **C − B** — the enrichment tier on top

That decomposition is the useful part. It says which half of the stack earns its
keep, which is a question the team currently cannot answer.

Evaluation is 5-fold stratified cross-validation, out-of-fold predictions only,
with a **paired bootstrap over borrowers** for the ΔAUC confidence interval.
Pairing matters: two independently computed AUC intervals would overlap wildly
and say nothing about whether the *difference* is real.

## Validating the instrument first

Before pointing this at real data, it has to be shown to work. Two controls,
both runnable now:

```bash
python -m experiments.incremental_lift.run synthetic --mode signal --n 800
python -m experiments.incremental_lift.run synthetic --mode null   --n 800
```

**Null control** — the outcome depends only on cash-flow stress; merchant
categories are assigned independently and carry no information. Arm C has 15
more columns than Arm A, so any leak (a shared standardiser, in-fold scoring, an
unregularised fit) would manufacture lift out of noise.

```
arm                      features     AUC      KS
A_cashflow_only                16   0.695   0.322
B_plus_payee_identity          19   0.697   0.335
C_plus_merchant_info           31   0.694   0.331

whole merchant stack                      -0.001  [-0.011, +0.010]   p=0.844
payee extraction + classification only    +0.002  [-0.006, +0.011]   p=0.607
enrichment tier on top                    -0.003  [-0.010, +0.003]   p=0.320
```

**Positive control** — the outcome depends on where discretionary money goes.
Critically, every borrower spends the *same distribution of rupees on the same
dates*; only the merchant behind each payment changes. So the cash-flow footprint
is identical and only an arm that can read the merchant can tell borrowers apart.

```
arm                      features     AUC      KS
A_cashflow_only                16   0.535   0.103
B_plus_payee_identity          19   0.785   0.535
C_plus_merchant_info           31   0.811   0.555

whole merchant stack                      +0.276  [+0.220, +0.331]   p=0.000*
payee extraction + classification only    +0.250  [+0.195, +0.304]   p=0.000*
enrichment tier on top                    +0.026  [+0.014, +0.039]   p=0.000*
```

Arm A sits at 0.535 — near chance, correctly blind to a signal carried entirely
by the merchant label. The harness finds real signal and does not invent fake
signal. Both controls run in CI (`tests/test_incremental_lift.py`).

### The dimensionality headwind, and why it decides how to read Berka

Arm C carries 15 more columns than Arm A. Even with the penalty tuned per arm,
those columns cost something in estimation variance when there is nothing in
them. Measured on the null control:

| n | defaults | ΔAUC on pure noise | 95% CI |
|---:|---:|---:|---|
| 200 | 41 | −0.033 | [−0.080, +0.010] |
| 400 | 76 | −0.026 | [−0.055, −0.001] |
| 800 | 162 | −0.001 | [−0.012, +0.009] |
| 1600 | 305 | −0.009 | [−0.016, −0.002] |

The penalty is real below roughly 150 defaults and effectively gone above it.
**Berka has ~70–80 defaults** — the worst row in that table. So the experiment
arrives at the real dataset already carrying a ~0.03 AUC headwind against the
merchant arms, purely for being wider.

The consequence is specific: on Berka, a genuine merchant effect smaller than
about 0.03 AUC cannot be distinguished from the noise floor. **A null result
there is inconclusive, not negative evidence.** Say that in the write-up rather
than reporting "no significant lift" as if it settled anything.

> **These numbers are not evidence about real borrowers.** The outcome is drawn
> from a process defined in `synthetic.py`. They establish that the measuring
> instrument works, and nothing else.

### One finding that came out of building the control

The first version of the positive control scaled gambling spend *on top of*
normal spending. Arm A reached **AUC 0.856** on a signal it was supposed to be
blind to — because money leaving the account is visible however it is labelled.
A borrower who gambles has a thinner balance and worse cash flow whether or not
you know the merchant was Dream11.

That is a genuine and slightly uncomfortable result for the project's thesis:
**a meaningful share of what merchant categorisation appears to tell you is
already in the raw arithmetic.** The incremental value of enrichment is only the
part *not* already implied by the money movement. Any real-data result has to be
read with that in mind, and it is an argument for reporting C − B separately
rather than quoting the headline C − A.

## Running it on real data

```bash
python -m experiments.incremental_lift.run berka --data-dir path/to/pkdd99 \
    --json-out results/berka.json
```

`berka.py` adapts the **PKDD'99 Financial Dataset**, which as far as I can
establish is the only openly available dataset with transaction-level rows *and*
observed loan outcomes for the same accounts. The usual credit benchmarks (UCI
Default of Credit Card Clients, Give Me Some Credit, Home Credit) ship
pre-summarised features with no payee or category field — there is nothing for
enrichment to act on, so no ablation is possible.

Download `trans.asc` and `loan.asc` from
<https://sorry.vse.cz/~berka/challenge/pkdd1999/> and point `--data-dir` at them.

**This adapter has not been run.** The sandbox this was written in has no
outbound network, so the mapping is written from the published schema and needs
a first run against the real files before any number from it is quoted. Expect
to fix small things.

### What the Berka result will and will not mean

Czech retail banking, 1993–1998, ~680 loans of which ~70–80 are defaults. At that
size the AUC confidence interval is wide and a small delta will not clear it.
`k_symbol` is a six-value controlled vocabulary standing in for enrichment
*output* — there is no merchant long tail and no resolution failure, so it tests
a much easier version of the problem than Indian UPI narrations do.

Nothing here transfers quantitatively to India. What it buys is that the method
runs end to end against outcomes **we did not author** — which is exactly what
the persona suite, however well calibrated, can never provide.

One mapping decision is worth challenging: `SANKC. UROK` (sanction interest,
charged on a negative balance) is a bank-generated distress marker of the same
class as a bounced ECS, so its narration includes `UNPAID` and Arm A can see it.
Withholding it would starve the cash-flow baseline of the dataset's clearest
distress signal and inflate apparent merchant lift. `--no-sanction-distress`
runs it the other way; reporting both is the honest thing to do.

## Files

| File | Role |
|---|---|
| `contract.py` | `LabelledStatement` / `Dataset` — the only thing adapters must produce |
| `featurize.py` | the three arms, by input masking |
| `modeling.py` | logistic scorecard, AUC, KS, stratified CV, paired bootstrap (numpy only) |
| `synthetic.py` | the two controls |
| `berka.py` | PKDD'99 adapter |
| `run.py` | CLI |

No new dependencies. An additive logistic model *is* the regulated-industry
standard for scorecards, so keeping it in ~80 auditable lines of numpy is a
feature, not a shortcut — a reviewer can read exactly what produced the number.
