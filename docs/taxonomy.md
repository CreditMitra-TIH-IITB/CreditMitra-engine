# Merchant Taxonomy — CreditMitra

**Issue #4. FROZEN after sign-off.** `india_merchants.json` (#6) and the LLM
fallback (#7b) both validate against this list. Adding a category later means
touching the dictionary, the prompt, and the scorer — so get it right now.

---

## The four fields every merchant carries

| Field | Purpose |
|---|---|
| `category` | what kind of business — drives the spend breakdown |
| `is_essential` | counts toward **L1 Essential Stability** |
| `risk_flag` | `gambling` / `bnpl_lending` / `crypto` / `null` |
| `lifestyle_dim` | which L-index this feeds |
| `recurring_type` | `adhoc` / `subscription` / `emi_like` / `payout_source` |

`category` answers *"what is this business"*; `lifestyle_dim` answers *"which
index does it feed"*. They're different axes — a Groww SIP is category
`investments` but lifestyle_dim `commitment`.

---

## Categories (24)

### Essential — `lifestyle_dim: essential` → L1

| category | is_essential | notes |
|---|---|---|
| `groceries` | true | kirana, BigBasket, DMart, Reliance Fresh |
| `utilities` | true | electricity boards, water, gas, LPG |
| `telecom` | true | Jio, Airtel, Vi — recharge & broadband |
| `fuel` | true | IOCL, HPCL, BPCL, Shell |
| `transport` | true | metro, bus, IRCTC, daily commute |
| `education` | true | school/college fees, Udemy, BYJU'S |
| **`healthcare`** | true | **EXCLUDED FROM SCORING — see below** |

### Aspirational — `lifestyle_dim: aspirational` → L2

| category | is_essential | notes |
|---|---|---|
| `food_delivery` | false | Swiggy, Zomato |
| `quick_commerce` | false | Blinkit, Zepto, Instamart |
| `shopping` | false | Amazon, Flipkart, Myntra, retail |
| `entertainment` | false | Netflix, Hotstar, Spotify, BookMyShow |
| `travel` | false | MakeMyTrip, Goibibo, airlines, hotels |
| `personal_care` | false | salons, Nykaa, gyms |
| `dining` | false | restaurants, cafes, caterers |

### Commitment — `lifestyle_dim: commitment` → L4  *(the self-control proxy)*

| category | is_essential | notes |
|---|---|---|
| `investments` | false | Groww, Zerodha, ICCL, SIP, mutual funds |
| `insurance` | true | LIC, health/term premiums |
| `rent` | true | landlord, housing society |
| `loan_emi` | true | bank EMI, NACH mandates |

> L4 rewards *voluntary sustained* obligations. Rent + SIP + insurance paid on
> schedule for months = demonstrated discipline without any credit history.

### Leverage — `lifestyle_dim: leverage` → L5 (inverse)

| category | risk_flag | notes |
|---|---|---|
| `bnpl_lending` | `bnpl_lending` | Simpl, LazyPay, slice, KreditBee, ZestMoney |

### Risk — `lifestyle_dim: risk` → L6 (inverse)

| category | risk_flag | notes |
|---|---|---|
| `gambling` | `gambling` | Dream11, MPL, RummyCircle, betting |
| `crypto` | `crypto` | WazirX, CoinDCX, Binance |

### Neutral — `lifestyle_dim: neutral` → feeds no index

| category | notes |
|---|---|
| `p2p_transfer` | person-to-person — the classifier's `person` bucket |
| `cash_withdrawal` | ATM — lowers L3 Digital Maturity |
| `gig_platform` | Swiggy/Zomato/Uber/Ola **paying the user** (`recurring_type: payout_source`) |
| `bank_charges` | fees, penalties |
| `other` | unresolved — the safe default |

---

## Rules

**Healthcare is excluded from scoring (fair lending).** It's categorised and
shown in the breakdown, but contributes **zero points** in `credit_scorer.py`
and is skipped by every L-index. Illness must never lower a credit score.
Test-asserted in Issues #11 and #13.

**Unknown → `other` / `neutral` / `adhoc`.** Every failure path (dictionary
miss, LLM timeout, invalid category) returns `MerchantEnrichment.unknown()`.
The pipeline never blocks on enrichment.

**Same merchant, two directions.** Swiggy as a *debit* is `food_delivery` /
`aspirational`. Swiggy as a *credit* is `gig_platform` / `payout_source` — a
gig payout, which is what identifies the Gig Hustler archetype. The dictionary
stores the debit meaning; `feature_engineering.py` (#10) reinterprets on
`direction == "credit"`.

---

## Scorecard weights (initial — team review)

Lifestyle block. Each index scores against **its own neutral anchor**, not a
shared 50, and gains and penalties are sized separately:

```
L >= neutral:  impact = (L − neutral) / (100 − neutral) × max_gain
L <  neutral:  impact = (L − neutral) / neutral         × max_penalty
```

| Index | neutral | max_gain | max_penalty | rationale |
|---|---:|---:|---:|---|
| L4 Commitment | 50 | 72 | 72 | strongest character signal (EPJ 2021) |
| L1 Essential Stability | 50 | 52 | 52 | spending persistence |
| L5 Leverage (inv) | 85 | 20 | 90 | hidden BNPL debt |
| L6 Risk Appetite (inv) | 85 | 15 | 90 | gambling/crypto |
| L3 Digital Maturity | 60 | 24 | 50 | our unique signal |

**Why the anchors differ.** L5 and L6 are inverse indices: they read 100 for
anyone who simply has no BNPL app and places no bets, which is most people.
Anchoring them at 50 paid out ~+90 for the mere absence of two vices, which was
enough to float a statement with eight bounced payments into "Very Good". They
are anchored at 85 with a small upside and a heavy downside, which is how an
underwriter reads them — no gambling is expected, heavy gambling is
disqualifying. L4 and L1 stay symmetric at 50, because sustained commitments and
essential stability are positive evidence in their own right.

**L2 is not in the table.** It measures the *share* of spend that is
discretionary, which is not a "higher is better" quantity — scored on the same
ramp it awarded points for overspending. It is now a conditional penalty
(`ASPIRATIONAL_STRAIN_*`) that applies only when a high L2 coincides with a thin
balance buffer, which is what "neutral alone; risky only w/ low buffer" meant.

Cash-flow block (~±300): FOIR, salary regularity, balance buffer, bounces,
overdrafts. Values live in `scoring_config.py` (#13) — **nowhere else**.

FOIR counts rent, loan EMIs, insurance and BNPL dues — *not* SIPs. A SIP recurs
like an EMI and is tagged `emi_like`, but money moving into savings is not an
obligation against income. It still counts toward L4, which asks a different
question.

The bounce penalty is 35 points per event with a floor of −180, so it keeps
biting past the second one; at the previous −80 floor, two bounces and eight
bounces cost exactly the same.

Bands: <580 Poor · 580-669 Fair · 670-739 Good · 740-799 Very Good · 800+ Excellent

> Weights are **expert-set, not fitted**. Persona ranking (#15) is the
> acceptance test. Predictive validation waits for labelled defaults.