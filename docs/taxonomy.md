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

Lifestyle block, `Σ weightᵢ × (Lᵢ − 50)/50 × max_pointsᵢ`, ~±300 total:

| Index | weight | max_points | rationale |
|---|---:|---:|---|
| L4 Commitment | 0.30 | 90 | strongest character signal (EPJ 2021) |
| L1 Essential Stability | 0.20 | 60 | spending persistence |
| L5 Leverage (inv) | 0.20 | 60 | hidden BNPL debt |
| L6 Risk Appetite (inv) | 0.15 | 45 | gambling/crypto |
| L3 Digital Maturity | 0.10 | 30 | our unique signal |
| L2 Aspirational | 0.05 | 15 | neutral alone; risky only w/ low buffer |

Cash-flow block (~±300): FOIR, salary regularity, balance buffer, bounces,
overdrafts. Values live in `scoring_config.py` (#13) — **nowhere else**.

Bands: <580 Poor · 580-669 Fair · 670-739 Good · 740-799 Very Good · 800+ Excellent

> Weights are **expert-set, not fitted**. Persona ranking (#15) is the
> acceptance test. Predictive validation waits for labelled defaults.