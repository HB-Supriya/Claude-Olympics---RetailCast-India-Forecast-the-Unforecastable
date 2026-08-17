# RetailCast India — Project Blueprint

This is the build guide for the "Forecast 28 Days of Demand" challenge. It's grounded in an
actual inspection of the starter kit data (not guesses) — the numbers below came from running
scripts against `sales_train.csv`, `market_signal.csv`, `vendor_signal.csv`, and `calendar.csv`.
Use this as the spine of your build; don't skip the audit section even if you're in a hurry —
it's worth more points than the model.

---

## 0. How this challenge is actually scored — read this first

The brief and starter kit tell you directly how the AI evaluator will weigh things. Internalize
these before writing any code, because they change what "good work" means here:

1. **"The best honest score is bounded; an impossibly good score is flagged, not rewarded."**
   The evaluator is explicitly checking for suspiciously-low RMSSE and will penalize it, not
   applaud it. This means: if a feature (e.g. a vendor signal) lets you fit the last 28 days
   *too* well, that is evidence you leaked, not evidence you won. Don't chase a heroic number.
2. **"Finding a data problem and saying so (with evidence) scores more than silently getting
   lucky."** Your data audit and its written evidence trail are graded artifacts, not throwaway
   exploration. A mediocre model with a sharp, evidenced audit beats a great-looking model with
   no audit.
3. **Three required artifacts, all graded:** `submission.csv` (accuracy), the **repo**
   (reproducibility, code quality), the **Claude chat export** (your reasoning process —
   "primary evidence of your data judgement"), and the **Approach Summary / Technical Decision
   Log** (≤1,500 words, must answer 7 specific questions, "every claim must be traceable to your
   chat export or your code"). A great model with a weak write-up will score worse than a solid
   model with an airtight write-up, because the write-up is graded as its own artifact.
4. **Metric is `mean_rmsse`** (see `olympics.json`) — RMSSE is the M5-competition metric: RMSE of
   your forecast divided by the RMSE of a seasonal-naive forecast on the training history, scaled
   per series, then averaged. This has two consequences worth designing around:
   - It's **scale-invariant** — a series that sells 0.5 units/day and one that sells 130
     units/day contribute comparably. Don't let your validation loss be dominated by the
     highest-volume series (`GROCERY_3_ATTA_MH_3`, mean ≈135/day) while starving the low-volume
     ones (`ELECTRONICS_1_CABLE_*`, mean <1/day, >75% zero days). Optimize per-series scaled
     error, not global raw error.
   - It's **relative to a naive baseline**. A model has to beat "repeat what happened 7 (or 28,
     or 365) days ago" to score under 1.0. Always report your naive-baseline RMSSE alongside
     your model's, in both your validation output and your write-up — it's the single most
     convincing evidence you can show that you're not overfitting to noise.

### What maximizes points, concretely
- **Do** put a real, timestamped audit trail of what you rejected and why into your Claude chat
  (Phase 1–2 reasoning — the brief says this explicitly). Reasoning "out loud" in chat before you
  write code is not wasted effort here; it's a graded deliverable.
- **Do** ship per-series (or per-segment) uncertainty/trust commentary — Meera asked for "what
  you trust and what you don't," and the seven required questions in the Approach Summary ask for
  this directly. A forecast with no confidence framing under-delivers on the actual ask.
- **Do** validate with a backtest window that mimics the true task: predict the *last* 28 days
  you have, having only seen the days before that, using a **rolling-origin** setup (not a random
  k-fold — sales are a time series, shuffling folds injects leakage from the future into the
  past).
- **Don't** use any feature that isn't legitimately known at prediction time for `d_1914..d_1941`.
  Section 1 below identifies one specific feature in this dataset that fails that test.
- **Don't** apply outlier "cleaning" uniformly. Section 1 explains why: some spikes are Diwali,
  not noise, and squashing them will cost you accuracy on exactly the days replenishment planning
  cares about most.
- **Don't** submit an ensemble-of-everything black box with no reasoning trail. A defensible LightGBM
  or exponential-smoothing model you can explain beats a stacked ensemble you can't.

---

## 1. Data audit findings (do this first, verify independently, don't skip)

The brief warns: *"not every anomaly needs the same treatment, and not every feed you are handed
is safe to use."* Below are the two vendor feeds, audited. Re-run these checks yourself in your
own EDA notebook and cite the numbers in your chat export — that's the paper trail the evaluator
wants.

### 1.1 `market_signal.csv` — reject as a horizon feature (evidence of leakage-shaped behavior)

| Check | Finding |
|---|---|
| Day coverage | `d_1` → `d_1913` only. **Zero coverage of the forecast horizon** (`d_1914`–`d_1941`). |
| Correlation with same-day actual sales (in history) | **r = 0.86–0.95** across every series sampled |
| Scale vs. actual sales | Consistently **~10–11× the mean of actual sales**, series by series (e.g. `ELECTRONICS_1_CHARGER_MH_1`: mean sales 11.95 vs. mean signal 127.6) |

**Verdict: do not use it as a predictive feature for the horizon, at all.** Two independent
reasons, either one sufficient on its own:
- **It doesn't exist where you need it.** It stops exactly at `d_1913`, the last day of history.
  To use it as a horizon feature you'd have to forecast the signal itself for 28 days — which
  means forecasting a near-perfect stand-in for sales in order to help forecast sales. That's
  circular, not useful.
- **Its near-perfect, near-linear correlation with the target it's supposedly independent from is
  itself the anomaly.** A genuine third-party "market demand index" tracking a single retailer's
  per-store, per-SKU daily units at r > 0.9 is not how market signals behave in practice — it
  reads as a scaled/noised copy of the sales series it's meant to accompany, i.e. leakage
  dressed up as a feature. Combined with the sharp cutoff at `d_1913`, the shape of the evidence
  points to a derived signal, not an independent one.
- Where it *is* legitimately useful: as a **history-only diagnostic** (e.g., cross-checking for
  data-entry issues, or comparing implied demand vs. recorded demand at known-anomalous dates) —
  never as a model input for the 28-day forecast.

### 1.2 `vendor_signal.csv` — usable, but only as a weak baseline/benchmark, not a trusted feature

| Check | Finding |
|---|---|
| Day coverage | `d_1` → `d_1941`. **Covers the full forecast horizon** — this is a legitimate vendor-supplied forecast product, available at prediction time. |
| Correlation with same-day actual sales (in history) | **r = 0.02–0.29** — weak, series-dependent |
| Scale vs. actual sales | Mean-matched almost exactly (e.g. `GROCERY_3_ATTA_MH_1`: mean sales 66.78 vs. mean vendor forecast 67.16) |

**Verdict: safe to use, but with modest expectations.** It gets the *average level* right
(it's not junk) but doesn't track day-to-day dynamics — spikes, festivals, SNAP days — well at
all (that's what the low correlation is telling you). Treat it as:
- A **benchmark to beat**, alongside the seasonal-naive baseline, in your validation report.
- A candidate **ensemble input or level-anchor feature** (e.g., `sales / vendor_forecast` ratio,
  or a residual-correction model on top of it) — worth testing, but validate whether it actually
  improves rolling-origin RMSSE before keeping it. Don't include it by default just because it's
  "vendor data" — earn its place in the feature set with a backtest number.
- Something to disclose honestly in the trust write-up: "vendor baseline is directionally useful
  for overall level, unreliable for event-driven spikes."

### 1.3 Anomalies in `sales_train.csv` — do not clean uniformly

Confirmed from the data: this is an **intermittent-demand** panel, not smooth retail volume.

- Average **40% of all daily observations across the 60 series are zero**; 24 of 60 series have
  zero sales on **more than half** their days (e.g. `HOMECARE_2_AGARBATTI_KA_3`: 87.7% zero days).
- High-volume series (`GROCERY_3_ATTA_MH_3`, mean ≈135/day) show occasional spikes up to 4–5×
  mean (max 612) — these coincide with festival calendar events, not random noise.

Implications:
- **Don't impute or smooth the zeros.** For low-volume SKUs, a zero is a legitimate observation
  (nobody bought a phone charger that day), not a missing value or a stockout by default. Blanket
  zero-imputation or interpolation will systematically inflate low-volume forecasts.
- **Do distinguish stockout-zeros from demand-zeros where you can.** Cross-reference
  `sell_prices.csv` — a week with no price row for an (item, store) means it wasn't being sold
  that week (per the data dictionary: "absent rows = not sold that week"). A run of zero sales
  during a week with *no* corresponding price row is a supply-side zero (exclude or flag, don't
  treat as demand signal); a run of zeros during a week *with* a price row is a demand-side zero
  (keep, it's real).
- **Don't clip or de-spike the highs uniformly either.** Cross-reference spikes against
  `calendar.csv`'s `event_name_1/2` and `snap_MH/KA/TN` columns first. A spike on `Diwali` or
  `Ganesh_Chaturthi` is signal your 28-day window may need to reproduce (check the horizon dates
  against the calendar for any festivals falling in `d_1914`–`d_1941` — if one lands there, that
  series-store combination's forecast should reflect it, not be flattened by a global outlier
  filter). A spike with no calendar or price explanation is a better candidate for winsorizing.
- **General rule:** anomaly treatment is a per-anomaly decision backed by a specific
  cross-referenced reason (calendar event / price change / stockout gap), never a global rule
  like "clip everything above the 99th percentile." Document each treated anomaly with its reason
  in your notebook/code comments — this is exactly the kind of traceable claim the write-up asks
  for.

### 1.4 Calendar and price structure (context, not a trap)

- `calendar.csv` covers `d_1`–`d_1969` (i.e., includes the full horizon plus a buffer past it) —
  safe to use for horizon features (day-of-week, month, festival flags, SNAP flags).
- 15 distinct recurring Indian festivals/events across ~5 years (Pongal, Republic Day, Holi,
  Ram Navami, IPL Final, Eid al-Fitr, Independence Day, Raksha Bandhan, Ganesh Chaturthi, Onam,
  Gandhi Jayanti, Dussehra, Diwali, Christmas, New Year) plus `event_type` (National / Cultural /
  Religious / Sporting) — these repeat roughly annually, so with ~5 years of history you have
  4–5 occurrences of each to learn an effect from. Check exactly which (if any) fall inside your
  28-day forecast window (`d_1914`–`d_1941`) — that's your single highest-leverage feature if one
  does.
- `sell_prices.csv` is **weekly**, joined via `wm_yr_wk` from `calendar.csv` (not daily) — don't
  join it naively assuming daily granularity; every day in the same retail week shares one price.
- SNAP flags are state-level (`snap_MH/KA/TN`), roughly 33% of days per state — worth testing as a
  feature, especially for essential categories (`GROCERY_3`, `HOMECARE_1`).

---

## 2. Recommended project structure

```
retailcast-india/
├── data/                     # symlink or copy of the provided data/ (do NOT commit large files if avoidable)
├── notebooks/
│   └── 01_audit.ipynb        # EDA + the feed-verdict evidence (Section 1) — reproduces every number cited in your write-up
├── src/
│   ├── data.py                # loaders: melt sales_train wide->long, join calendar/prices, build feature frame
│   ├── features.py            # calendar features, price features, lags/rolling stats, (validated) vendor_signal features
│   ├── validation.py          # rolling-origin backtest harness + RMSSE implementation
│   ├── models/
│   │   ├── baseline.py         # naive/seasonal-naive, moving average — required for the RMSSE denominator + sanity check
│   │   ├── statistical.py      # Croston/TSB or ETS for intermittent low-volume series (optional escalation)
│   │   └── global_model.py     # single LightGBM/CatBoost trained across all 60 series with series-id + calendar + price features
│   └── predict.py              # generates submission.csv from a trained model
├── tests/
│   └── test_submission_format.py  # unit tests mirroring validate_format.py, run in CI
├── submission.csv
├── approach_summary.md        # ≤1,500 words, answers the 7 required questions (Section 6)
├── requirements.txt           # pinned versions
├── Makefile                   # `make audit`, `make train`, `make submit`, `make validate`
└── README.md                  # run instructions, reproducibility notes
```

Why this shape: the evaluator explicitly grades **reproducibility** (repo is a required
artifact) and **traceability** (write-up claims must trace to code or chat). A flat pile of
notebooks with no pinned environment fails both. A `Makefile`/CLI entry point that regenerates
`submission.csv` from raw `data/` with one command is the strongest signal of reproducibility you
can give.

---

## 3. Modeling strategy — escalate only when the backtest justifies it

Meera's own words are the design spec: *"I don't need the fanciest model... I need numbers I can
actually order stock against."* Build in this order, and only move to the next tier if it
**measurably** improves rolling-origin RMSSE over the previous tier — record every tier's score.

### Tier 0 — Baselines (never skip, always keep in the final report)
- **Seasonal-naive** (repeat value from 7 days prior, or the same weekday last cycle) — this is
  the RMSSE denominator's basis; you must compute it regardless.
- **Simple moving average** (e.g., trailing 28-day mean per series).
These exist to (a) give you the honesty check the evaluator rewards, and (b) catch bugs — if
your "sophisticated" model can't beat these on 3+ backtest folds, that's a red flag on the model,
not the baseline.

### Tier 1 — Classical per-series statistical models (good default for intermittent demand)
- **Croston's method / TSB (Teunter-Syntetos-Babai)** for the low-volume, high-zero-share series
  identified in Section 1.3 — these are purpose-built for intermittent demand and usually beat
  ETS/ARIMA on series with >50% zero days.
- **ETS or auto-ARIMA with exogenous regressors** (festival dummy, SNAP flag, price) for the
  higher-volume, smoother series (`GROCERY_3_ATTA_*`, `HOMECARE_1_DETERGENT_*`).
- Segment series by volume/intermittency (Section 1.3's zero-share stat is your segmentation
  variable) rather than using one model family for all 60 — this is a defensible, explainable
  design choice for the write-up.

### Tier 2 — One global gradient-boosted model (recommended production-grade default)
A single LightGBM/CatBoost model trained across all 60 series simultaneously, with:
- Series identity as categorical features (`item_id`, `store_id`, `state_id`, `cat_id`,
  `dept_id`) so the model can share statistical strength across similar series (this matters most
  for the sparsest, lowest-volume series where per-series models are undertrained).
- Calendar features: day-of-week, month, days-to/since-nearest-festival, `event_type` one-hot,
  `snap_*` flags.
- Price features: current `sell_price`, price change vs. prior week, and — cautiously — a
  "was this even on sale this week" flag derived from row-presence in `sell_prices.csv`.
- Lag/rolling features (lag-7, lag-28, rolling 7/28-day mean and std) — beware: for a 28-day-ahead
  *multi-step* forecast, naive lag features computed from the true series won't exist that far
  into the horizon without recursion. Either (a) train a direct multi-horizon model with
  horizon-index as a feature, or (b) do recursive multi-step forecasting carefully and validate
  that error doesn't compound unacceptably by day 28 — check this explicitly in your backtest.
- **Only after backtesting confirms it helps:** a `vendor_signal`-derived feature (Section 1.2).
- This tier scales trivially to more stores/products (it's already trained as one model over the
  full panel) — see Section 4.

### Tier 3 — Only if Tier 2 clearly wins on backtest, and you can explain it
- LightGBM quantile regression (multiple quantile heads) to natively produce prediction
  intervals — directly answers "what you trust and what you don't" with numbers instead of prose.
- Simple weighted ensemble of Tier 1 (per-segment) + Tier 2 (global), weighted by backtest
  performance per segment.

**Do not** jump straight to Tier 3 or exotic deep learning (N-BEATS/DeepAR/Transformers) for a
60-series, ~5-year panel — insufficient series count for deep learning to reliably outperform
gradient boosting or well-tuned classical methods here, and the brief explicitly discourages
over-engineering ("I don't need the fanciest model"). If you do try one, it must win a backtest
comparison to earn its place, and the complexity cost must be justified in the write-up.

---

## 4. Scaling and efficiency

Even though this challenge is 60 series, build as if it were 6,000 — that's what "how to scale
up" is really asking, and it's also just better engineering:

- **Prefer one global model over 60 independent per-series models.** A global LightGBM trained on
  a long-format panel (one row per series-day) is the same amount of code whether you have 60 or
  60,000 series — it's the scalable pattern. 60 independent ARIMA fits is not (linear compute
  growth, no shared learning, painful to maintain).
- **Vectorize feature engineering** with `pandas`/`polars` group-by operations over
  `(item_id, store_id)`, never per-series Python loops. On this data it doesn't matter for speed,
  but it's the difference between a script that dies at 10,000 series and one that doesn't.
- **Melt wide→long once, cache it.** `sales_train.csv` is wide (`d_1`...`d_1913` as columns) —
  melt to long format (`id, d, sales`) immediately in `src/data.py`, then join calendar/prices/
  features onto the long frame. Don't repeatedly re-melt in notebooks.
- **Backtest efficiently:** rolling-origin evaluation naturally multiplies compute by the number
  of folds. Use 3–5 folds (not 20) with a fixed 28-day validation window each, stepping back by
  28 or 91 days — enough to be statistically meaningful without re-training 20 times per
  experiment. Cache fold splits.
- **Memory:** category-dtype the id columns (`item_id`, `store_id`, `state_id`, etc.) before
  feeding to LightGBM — trivial here, standard practice at scale.
- **Reproducibility as an efficiency multiplier:** pin `requirements.txt`, set random seeds, and
  make `make train && make predict` regenerate `submission.csv` byte-for-comparable results. This
  is graded directly (repo artifact) and also just saves you from your own future confusion.

---

## 5. Validation — make it match the real task, and make it visible

1. **Rolling-origin backtest**, not k-fold: pick 3–5 cutoff points in the history, each time
   training only on data before the cutoff and scoring on the 28 days after it — exactly
   mirroring the real `d_1913 → d_1914..1941` task.
2. **Compute true RMSSE**, not RMSE or MAPE — implement it per the M5 definition (per-series RMSE
   over the horizon, divided by per-series RMSE of the 1-step seasonal-naive forecast computed on
   that series' training history, then averaged — check whether the challenge weights by series
   or by scale; `mean_rmsse` in `olympics.json` implies an unweighted mean across the 60 series).
3. **Report baseline RMSSE alongside your model's** on every fold — this is your strongest
   evidence against the "impossibly good score gets flagged" trap, and it's free to produce.
4. **Report per-segment RMSSE** (by volume tier / intermittency, from Section 1.3) not just an
   overall mean — a single averaged number can hide a model that's great on `GROCERY_3_ATTA` and
   useless on `ELECTRONICS_1_CABLE`. This segment breakdown is also your raw material for the
   "what you trust and what you don't" answer.
5. **Sanity-check the final submission** before anything else: total forecast volume per series
   should be in the same order of magnitude as the last few historical cycles (compare `sum(F1..F28)`
   vs. `sum` of the last 28 historical days, per series) — a model that predicts 10x or 0.1x
   historical scale for a series is broken, not "very confident."
6. Run `validate_format.py` as a final gate — it's the exact structural check the evaluator runs;
   passing it is necessary but not sufficient (it doesn't check accuracy).

---

## 6. The Approach Summary / Technical Decision Log — answer these 7 questions directly

The starter kit says this file is capped at 1,500 words and must answer seven specific questions,
with every claim traceable to your chat export or code. Structure it exactly around these
(inferred from the brief's language — confirm final wording against your actual challenge
portal, but build to this shape now so you're not writing it cold at the deadline):

1. **Audit method** — what you checked, and how (point to Section 1's checks: coverage windows,
   correlation with target, scale comparisons, cross-referencing anomalies against calendar/price).
2. **Data verdicts, including the reading you rejected** — state the `market_signal.csv` verdict
   explicitly, including the specific numbers (coverage cutoff at `d_1913`, r≈0.9 correlation) and
   name the alternative reading you rejected (e.g., "it could be a legitimate independent index
   that happens to correlate with retail demand cycles generally" — and say why the cutoff and
   correlation magnitude together make that reading implausible).
3. **What you left alone** — the anomalies/spikes you decided NOT to treat, and the specific
   evidence (festival date, price row) that justified leaving them.
4. **Modelling choices** — which tier(s) from Section 3 you shipped, and the backtest numbers
   that justified escalating past the baseline (or not escalating further).
5. **Validation you trust** — your rolling-origin setup, number of folds, and why you believe it
   generalizes to the true held-out horizon better than an in-sample fit metric would.
6. **Your least-sure call** — pick one real judgment call (e.g., whether to include the
   `vendor_signal` feature, or how you handled a specific stockout gap) and argue both sides
   honestly. This question is explicitly there to see if you can self-critique — don't pick a
   safe non-answer.
7. **Reproduce and stress** — exact commands to regenerate `submission.csv` from scratch, plus one
   stress scenario you checked (e.g., "what happens to the forecast if I remove the festival
   feature" or "how much does the score change across backtest folds" — evidence of robustness,
   not just a point estimate).

Keep it tight — 1,500 words is roughly 200 words per question. Write it after the work is done,
but draft the skeleton (Section 6 structure) now, and fill numbers in as you get them, so nothing
is reconstructed from memory at the end.

---

## 7. Claude chat export — what needs to be *in* the conversation

Since the chat export is graded as "primary evidence of your data judgement," don't do your real
EDA silently in a notebook and summarize it into chat after the fact — think out loud, in chat,
through Phase 1 (audit) and Phase 2 (data verdicts) specifically:
- State what you're about to check and why, before checking it.
- Report the actual numbers you find (coverage ranges, correlations, zero-share stats) inline.
- State your verdict and reasoning for each feed *in the conversation*, not just in code comments.
- If you change your mind about something (e.g., first assumed `vendor_signal` was safe, then
  found the low correlation), leave that revision visible — a corrected judgment is evidence of
  rigor, not something to edit away.

---

## 8. Pre-submission checklist

- [ ] `market_signal.csv` excluded from horizon-feature set (Section 1.1) — confirmed in code, not just in notes.
- [ ] `vendor_signal.csv` usage (if any) backed by a backtest number showing it helps.
- [ ] Zero-inflation handled per-series-segment, not globally imputed.
- [ ] Anomalies/spikes cross-checked against `calendar.csv` and `sell_prices.csv` before any treatment.
- [ ] Rolling-origin backtest run with ≥3 folds; RMSSE computed per M5 definition.
- [ ] Baseline (seasonal-naive) RMSSE reported alongside model RMSSE, every fold.
- [ ] Per-segment (volume/intermittency) error breakdown produced — this is your "trust" answer.
- [ ] `submission.csv` passes `validate_format.py` with `PASS`.
- [ ] Total forecast volume per series sanity-checked against recent historical volume.
- [ ] Repo runs end-to-end from a clean clone with pinned deps (`make train && make predict` or equivalent).
- [ ] `approach_summary.md` ≤1,500 words, answers all 7 questions, every claim traceable to chat export or code.
- [ ] Claude chat export includes the live audit reasoning (Section 7), not a post-hoc summary.
