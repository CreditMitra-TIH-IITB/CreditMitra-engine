# Merchant-Derived Lifestyle Profiling

**Status: research contribution and feature generator. Not a validated credit
risk model, and not the product claim.**

Read that line as a scope statement, not modesty. What follows is a defensible
methods contribution — a way to turn merchant-resolved transactions into six
interpretable indices and a behavioural archetype, grounded in published work on
spending-derived traits. What it is *not* is a scorecard anyone should underwrite
against, because its weights are expert-set and have never been fitted to, or
validated against, observed defaults.

The claim the project defends commercially lives elsewhere:
[`experiments/incremental_lift/`](../experiments/incremental_lift/README.md)
measures what merchant information adds to a credit-risk model, against outcome
labels we did not author. That is the number for the TIH report. This document
is the method that generates the features that number is measured on.

---

## 1. Why lifestyle-from-merchants is scientifically legitimate

Four strands of evidence support treating merchant categories as a credit signal.

**(a) Spending categories carry psychological signal.** Gladstone et al.,
*Inferring psychological traits from spending categories and dynamic consumption
patterns* (EPJ Data Science 10:24, 2021) analysed 74M transactions from 127k
customers and found Materialism and Self-Control inferable from spending features
(F1 ≈ 0.42/0.41 against random baselines), with high self-control correlating
with stable, regular spending. Their 54-feature framework — overall spending,
temporal variability and persistence, category dynamics, category profile — is
the backbone of the indices below.

> **What this does and does not license.** F1 ≈ 0.42 is modest. We use their
> *feature framework* as grounding for which behaviours are worth measuring. We
> do not claim to measure personality, and no index here should be described as
> measuring a trait. Where a name suggests otherwise — L4 is informally "the
> self-control proxy" — that is shorthand for "the behaviour their work
> associates with self-control", nothing more.

**(b) Spending categories predict delinquency directly.** A hidden-Markov study
predicts credit-card delinquency from per-category spend plus past behaviour
(*Prediction of consumer credit card risk from an analysis of spending
categories*, 2025). Related: *Financial Risk Assessment via Long-term Payment
Behavior Sequence Folding* (arXiv:2411.15056).

**(c) Cash-flow features dominate application data for thin-file lending.** The
Malaysian MSME study reports transaction features alone at AUROC 0.821 vs 0.672
for application data, with an interpretable logistic model beating RF/GBM
(arXiv:2510.16066); see also NBER w33367, *From FICO to Cash Flow*.

**(d) Interpretable additive scorecards are the regulated standard** (WoE/logistic,
SHAP for ML upgrades).

**Synthesis.** (a)+(b) justify lifestyle-from-merchant-categories as a risk
signal; (c) grounds the cash-flow half; (d) keeps it explainable. The
contribution claimed is the combination: merchant-derived lifestyle profiling
computed **on-device**, behind a differentially-private extraction model, for
Indian UPI data.

---

## 2. The six indices

Computed from enriched transactions. Healthcare is excluded from every index —
see §5.

### L1 — Essential Stability

```
essential_share = ₹(is_essential rows) / ₹(total debits)
persistence     = mean cosine_similarity(c_m, c_{m+1})   # monthly category vectors
L1 = 100 × (0.6 × essential_share + 0.4 × persistence)
```

`persistence` is Gladstone et al.'s spending persistence — does October look
like November. An earlier implementation substituted "did any essential row
appear this month", which saturates at 1.0 for anyone who buys groceries and
therefore carried no information.

`essential_share` reads the `is_essential` field, which the schema documents as
the field that counts toward L1. It deliberately does *not* read `lifestyle_dim`:
the two disagree for insurance (essential, but `lifestyle_dim: commitment`
because it feeds L4) and for rent.

### L2 — Aspirational

```
L2 = 100 × min(discretionary_ratio / 0.5, 1)
```

A **descriptor, not a virtue**, and this matters more than it sounds. L2 is the
share of budget going to discretionary categories. It is not a "higher is
better" quantity, so it is excluded from the scorecard's index ramp and enters
only as a conditional penalty when high discretionary spend coincides with a thin
balance buffer. Scored on the same ramp as the others, it awarded points for
overspending.

### L3 — Digital Maturity

```
L3 = 100 × (0.5 × (1 − min(cash_ratio/0.3, 1)) + 0.5 × merchant_resolution_rate)
```

`merchant_resolution_rate` — the share of debits whose payee resolved to a known
merchant — is a signal unique to this pipeline. Note the reflexivity: it measures
the borrower's digital footprint *and* our dictionary's coverage. A thin
dictionary makes borrowers look less digitally mature than they are.

### L4 — Commitment

```
recurring = payees paid in ≥3 distinct months (rent, SIP, insurance, EMI)
L4 = 100 × min(Σ₹recurring / (0.35 × income), 1) × regularity × reliability
regularity  = 1 − min(mean_std(day_of_month) / 8, 1)
reliability = 1 − min(bounce_count / months_covered, 1)
```

The heaviest-weighted index. Three departures from the literature formula, each
deliberate:

- Cadence is judged **per payee**, not per category, which is what makes "the
  same landlord every month" legible as one sustained obligation.
- A commitment counts only from **three** distinct months. Two payments are a
  coincidence.
- `reliability` is ours. A commitment that bounces is not being sustained,
  whatever the calendar says. Without it, a statement with eight bounced
  payments scored a *perfect* commitment index off the rent line alone. This
  intentionally overlaps with the cash-flow block's bounce penalty: the two
  answer different questions (did you honour obligations vs can you cover
  outflows) and a bounce is evidence for both.

**Rent is recognised from the narration**, not from enrichment. Rent goes to a
private landlord, so `payee_type` is `person`, no merchant tier can ever resolve
it, and its category stays `None`. Before this was handled, the single largest
recurring obligation most Indian renters carry was invisible to L4, to FOIR and
to the essential share.

**Absence of evidence returns neutral, not zero.** With no commitment rows at all
L4 returns the neutral anchor. Someone who pays rent in cash is invisible here;
someone whose rent ECS bounces every month is visible and failing. Scoring both
at zero penalised the informal earner for being unbanked — precisely the
exclusion this project argues against.

### L5 — Leverage (inverse)

```
L5 = 100 − min(bnpl_share/0.15, 1)×70 − min(bnpl_merchant_count/4, 1)×30
```

Hidden leverage — Simpl, LazyPay, KreditBee, slice — appears in merchant names
*before* any bureau records it. This is the clearest case where merchant
resolution sees something a bureau cannot.

### L6 — Risk Appetite (inverse)

```
L6 = 100 × (1 − min(risk_share × 10, 1))
```

Gambling and crypto share of debits. The steep multiplier reflects Indian
underwriting practice.

### Behavioural texture

`category_diversity` (normalised entropy), `merchant_loyalty` (repeat-merchant
share), `burstiness` B = (σ−μ)/(σ+μ) of inter-transaction gaps (Goh–Barabási),
`category_turnover` (a coarse per-month distinct-category rate, not a true
first-appearance count).

---

## 3. Archetypes

Rule-based, priority-ordered, thresholds in `scoring_config.py`. Most-severe
first, so a gambler who also uses BNPL gets the more actionable label.

| Archetype | Trigger | Reading |
|---|---|---|
| Gambler | L6 < 40 | active risk-taking |
| Overextended | ≥0.5 bounces/month | ordinary spending, cannot cover obligations |
| BNPL-Heavy Spender | L5 < 40 | hidden debt |
| Cash-Reliant Informal | L3 < 60 | unverifiable; refer to manual review |
| Gig Hustler | no salary, income present, L3 ≥ 70 | income-volatile, not undisciplined |
| Aspirational Overspender | L2 ≥ 70 and buffer < 30 days | lend with limits |
| Salaried Saver | salary, L4 ≥ 50, L5 ≥ 70 | prime thin-file |
| Balanced | — | nothing dominates |

Gig-payout detection is itself a merchant-pipeline capability: recognising Zomato
as a *payer* identifies livelihood, which no generic cash-flow model can do.

> **Archetype labels are consequential.** "Gambler" attached to a person is a
> claim about them, made by a rule with expert-set thresholds and no measured
> error rate. In any borrower-facing surface these need either a confidence
> treatment or neutral wording. They are currently fit for an internal demo and
> a research figure, not for disclosure to the person being described.

---

## 4. Calibration, and what the persona suite does and does not prove

Eight synthetic personas span six months each. Every archetype classifies
correctly and the score ordering holds:

| persona | score | band | archetype |
|---|---:|---|---|
| salaried_saver | 892 | Excellent | Salaried Saver |
| gig_hustler | 862 | Excellent | Gig Hustler |
| balanced_salaried | 779 | Very Good | Balanced |
| aspirational_overspender | 676 | Good | Aspirational Overspender |
| cash_reliant_informal | 623 | Fair | Cash-Reliant Informal |
| bnpl_heavy_spender | 498 | Poor | BNPL-Heavy Spender |
| frequent_bouncer | 434 | Poor | Overextended |
| gambler | 394 | Poor | Gambler |

Beyond ordering, the suite asserts the model *discriminates*: every index must
vary across the set, both blocks must be able to penalise as well as reward, no
score may pin to a bound, every band must be reachable.

**This is a sanity check, not validation, and the distinction is the whole point
of this section.** The personas were designed by the same people who set the
weights. They encode the same priors. An earlier two-month version of this suite
passed every ordering assertion while four of six indices were effectively
constant, the cash-flow block was positive for every persona regardless of
behaviour, and a statement with eight bounced payments scored "Very Good". Every
ordering test still held, because each persona had exactly one hand-placed dip.

Passing the persona suite means the pipeline is coherent. It says nothing about
whether the score predicts default.

---

## 5. Fair lending

Healthcare is **fully excluded** from every ratio and index — removed from both
numerator and denominator, not zero-weighted — so a month with a large medical
bill cannot even dilute an unrelated ratio. Nobody should score worse for being
ill.

That exclusion is necessary and not sufficient. Merchant categories can proxy for
protected and sensitive attributes well beyond health: religious donations,
prenatal and fertility services, alcohol, political contributions, and merchants
correlated with caste or region. The current taxonomy has no systematic review
for this. Before any deployment touching real lending decisions, the category
list needs a proxy-discrimination audit, and the archetype rules need
disparate-impact testing across whatever demographic axes can be obtained.

---

## 6. Honest limitations

1. **No predictive validation.** Weights and thresholds are expert-set. There is
   no AUC, no KS, no measured relationship to default. The persona suite is a
   consistency check, not evidence.
2. **The EPJ grounding is thinner than it looks.** F1 ≈ 0.42 for trait inference.
   We borrow the feature framework, not the trait claims.
3. **Merchant resolution is self-referential.** L3 partly measures our own
   dictionary coverage (85 entries at time of writing). A borrower in a
   thin-coverage segment looks less digitally mature than they are.
4. **Categorisation may add less than it appears.** The incremental-lift
   experiment found that a cash-flow-only model can recover much of a
   "merchant-driven" signal, because money leaving the account is visible however
   it is labelled. Report the enrichment tier's marginal contribution separately
   from the whole stack's.
5. **Small-sample effects are not neutral.** Adding merchant features costs ~0.03
   AUC in estimation variance below roughly 150 defaults, so a null result on a
   small dataset is inconclusive rather than negative.
6. **Rent detection is regex-based** and will miss narrations that do not say
   rent or name a landlord — likely biased against informal tenancies, which is
   the wrong direction for an inclusion product.
7. **Thresholds are tuned on synthetic Indian-urban-salaried patterns.** Rural,
   agricultural and seasonal income profiles are not represented at all.

---

## 7. References

1. Gladstone, Matz et al., *Inferring psychological traits from spending categories and dynamic consumption patterns*, EPJ Data Science 10:24 (2021). <https://link.springer.com/article/10.1140/epjds/s13688-021-00281-y>
2. *Prediction of consumer credit card risk from an analysis of spending categories: a hidden Markov model* (2025).
3. *Financial Risk Assessment via Long-term Payment Behavior Sequence Folding*, arXiv:2411.15056.
4. *Cash Flow Underwriting with Bank Transaction Data: MSME Financial Inclusion in Malaysia*, arXiv:2510.16066 (2025).
5. *From FICO to Cash Flow*, NBER w33367.
6. *Explainable Machine Learning in Credit Risk Management*, Computational Economics.
7. Goh & Barabási, burstiness measure.
8. CFPB, cash-flow data and credit invisibility.

---

## 8. Where this sits

| Question | Answered by |
|---|---|
| Does merchant data improve a credit model? | `experiments/incremental_lift/` — measured, against external labels |
| What features does merchant data produce? | this document |
| Is the pipeline internally coherent? | `tests/test_persona_ranking.py` |
| Does the score predict default? | **unanswered**; needs a lending partner's outcomes |
