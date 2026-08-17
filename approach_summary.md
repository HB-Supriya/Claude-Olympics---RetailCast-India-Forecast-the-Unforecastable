# RetailCast India — Approach Summary / Technical Decision Log

*Every number below is reproduced by `make audit` (→ `notebooks/01_audit.ipynb`) or `make backtest`
(→ `backtest_scores.csv`, printed report) — see those artifacts and the repo's chat export for
the live reasoning trail.*

## 1. Audit method

I checked both vendor feeds on three axes, all computed against the real CSVs (not assumed):
**coverage window** (min/max `d_num` per feed vs. the `d_1914`–`d_1941` horizon), **correlation
with same-day actual sales** in the overlapping history (per-series Pearson r), and **scale
ratio** (mean signal ÷ mean sales, per series). For `sales_train.csv` I checked **zero-share**
per series and overall, cross-referenced the highest-volume series' spikes (>mean+3·std) against
`calendar.csv` event flags, and cross-referenced every zero-sales row against `sell_prices.csv`
row-presence to distinguish stockout-zeros from demand-zeros. I also scanned `calendar.csv` for
any festival falling inside the actual forecast horizon.

## 2. Data verdicts, including the reading I rejected

**`market_signal.csv`: excluded entirely.** Coverage stops at `d_1913` — zero overlap with the
horizon. Correlation with same-day sales is r=0.87–0.96 (mean 0.92) across all 60 series, at a
consistent ~10.5× scale (9.76×–11.03×). I considered the reading "this could be a genuine
independent market-demand index that happens to move with retail cycles generally" — but r>0.9
*for every single series* plus a cutoff landing exactly on the last day of history together are
too specific to be coincidental; a real third-party index wouldn't track one retailer's per-SKU
units that tightly, and wouldn't happen to stop exactly where our own history stops. That's the
shape of a derived/leaked signal, not an independent one.

**`vendor_signal.csv`: covers the full horizon, mean-matched (scale ratio 0.99–1.01) but weakly
correlated (r=-0.00–0.58, mean 0.12) — legitimate but noisy.** I initially treated it as a likely
feature. The backtest (§4) reversed that: adding it to the global model made mean RMSSE
*slightly worse* (0.9064 vs. 0.9048), so it's excluded from the final feature set, kept only as a
benchmark model in the backtest report.

## 3. What I left alone

All 45,970 zero-sales rows have a corresponding `sell_prices.csv` row that week — **zero
stockout-zeros found in this dataset**; every zero is a genuine demand-zero and was kept, not
imputed. On `GROCERY_3_ATTA_MH_3` (highest-volume series, mean 134.9/day), the single largest
spike (612 units, 2018-11-04) lands exactly on Diwali — left untouched, it's real event demand.
Only 4 of that series' 23 largest spikes carry a calendar tag; the other 19 have no calendar or
price explanation. I chose **not** to winsorize even those: I have no independent evidence
(promo calendar, external event list) that they're noise rather than unlogged local demand
drivers, and the shipped models (GBM splits, TSB smoothing) are already reasonably robust to
isolated highs without deleting information I can't verify is wrong.

## 4. Modelling choices

Tier 0 baselines (always computed): seasonal-naive mean RMSSE 1.1656, moving-average(28) 0.9137.
Tier 1 (TSB, intermittent-demand smoothing): 0.9111 overall. Tier 2 (global gradient-boosted
model, direct multi-horizon, `HistGradientBoostingRegressor` — LightGBM's `libomp` dependency
wouldn't load on this machine, sklearn's quantile-native GBM substitutes at the same tier):
0.9048 without vendor feature, 0.9064 with it. **Shipped model: a Tier-3 segment blend** — TSB
for the 24 series with >50% zero-days, global GBM for the other 36 — because the per-segment
backtest showed TSB beating the global model on the low-volume segment on **every one of 4
folds** (e.g. cutoff `d_1885`: TSB 1.0112 vs. GBM 1.0548), while GBM won the regular segment on
every fold (e.g. same cutoff: GBM 0.6925 vs. TSB 0.7194). P10/P90 uncertainty bounds come from
native quantile heads for the GBM segment and trailing-90-day empirical quantiles for the TSB
segment (TSB has no native uncertainty output).

## 5. Validation I trust

Rolling-origin, 4 folds at cutoffs `d_1612/1703/1794/1885`, each scored against the true 28 days
immediately after its cutoff — the exact shape of the real `d_1913→d_1914..1941` task, with every
fold's scored window strictly after its own training cutoff. For the GBM, each fold's training
set is built from 64 pseudo-cutoffs stepped every 28 days through history (≈107k rows/fold), each
constrained to `pseudo_cutoff + horizon ≤ fold_cutoff` so no training row ever sees that fold's
future. RMSSE follows the official M5 definition: denominator is each series' training-history
one-step (lag-1) squared error, not a seasonal-lag error — a specific choice worth flagging since
"naive" gets used two different ways in the brief (the RMSSE *denominator* vs. the seasonal-naive
*model* I also backtest). I trust this setup over an in-sample metric because the baseline
ordering is consistent across all 4 folds (seasonal-naive worst every time, 1.09–1.21) and the
low-volume/regular segment split's winner never flips fold-to-fold.

## 6. My least-sure call

Excluding `vendor_signal` as a feature is a binary, all-or-nothing decision, and the fold-level
evidence for it is genuinely mixed, not clean: adding it helped in 2 of 4 folds (`d_1612`:
0.9431 vs. 0.9463; `d_1794`: 0.9185 vs. 0.9200) and hurt in the other 2 (`d_1703`: 0.9223 vs.
0.9157; `d_1885`: 0.8418 vs. 0.8374) — the net effect is a small average loss, not a decisive
one. Some individual series (e.g. `ELECTRONICS_1_CHARGER_TN_1`, r=0.58) correlate with it well
above the mean; a per-series gate (use it only where its own historical correlation clears a
threshold) might beat the current all-or-nothing toggle. I didn't test that finer granularity —
it would need more folds than 4 to trust the extra degrees of freedom, and I'd rather ship the
simpler, backtest-confirmed decision than a more complex one I can't yet back with equal
evidence. Argument for the current call: simplicity and a real (if modest) net negative. Argument
against: I'm leaving a legitimately-timed, mean-matched signal on the table for the specific
series where it does correlate.

## 7. Reproduce and stress

`make setup && make audit && make backtest && make predict && make validate && make test`
regenerates everything from a clean clone (after populating `data/`, which is gitignored).
Stress check performed: fold-to-fold variance of the shipped model. The global-GBM segment's
RMSSE ranges from 0.8374 (`d_1885`) to 0.9463 (`d_1612`) across the 4 folds — a ~12% relative
spread — while the ranking between models (GBM-wins-regular, TSB-wins-low-volume,
seasonal-naive-worst-everywhere) holds in all 4. That consistency in *ranking* despite real
variance in *magnitude* is why I trust the segment-blend decision more than the exact RMSSE
numbers themselves — the point estimate will move with the actual `d_1914`–`d_1941` window, but
the model choice is unlikely to flip. One horizon-specific note for that window: `calendar.csv`
places Ram Navami (`d_1921`) and Eid al-Fitr (`d_1928`) inside it — both religious/cultural
events with several prior-year occurrences in training history, so the festival-distance feature
has real signal to draw on for this specific submission, not just in general.
