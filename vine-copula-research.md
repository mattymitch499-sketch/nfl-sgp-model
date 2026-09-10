# NFL SGP Model — Research & Planning Notes

Informal research/planning notes, not a spec. `tasks/plan.md` and `METHOD.md`
remain the source of truth for what is actually built and pre-registered;
nothing in this file is implemented yet.

## Date: 2026-09-09

## Core philosophy

- Books price individual legs (singles) well; the thesis is that their real
  weakness is pricing *cross-leg correlation* in same-game parlays.
- Find where the combined implied probability of a multi-leg ticket diverges
  from the joint probability the outcomes actually produce.
- Model-free comparison throughout: measure what actually happens against
  what the book charges, rather than fitting a competing forecast.

## Current 2-leg project status

- The empirical-survival-copula approach (standardise leave-one-out, rank to
  uniforms, read the joint at market marginals) is built and measured for six
  families — see `METHOD.md` and `tasks/plan.md` §1 for the full method.
- Pre-registration, the decision bar, and the §1.9 transport-validation split
  (train ≤2017 / test ≥2018, bootstrapped on the *difference* between the two
  halves) are already in place.
- A vine-copula extension is the natural next step if 3-leg tickets are worth
  pricing — see below.

## 2-leg SGP family evaluation

*(Filled in from what this repo has already measured — `tasks/todo.md`,
2026-09-03 — rather than left blank, since no new evaluation was pasted here.)*

Joint probability at marginals 0.5/0.5 Over/Over; independence = 0.250:

| family | n | joint | reading |
|---|---|---|---|
| `teamtotal_x_passtd` | 7,510 | **0.352** | part-whole pair, the primary family |
| `passyds_x_wr1recyds` | 5,488 | 0.326 | real, but the pair books price best |
| `bothqb_passyds` | 2,250 | 0.266 | weak shootout signal |
| `spread_x_total` | 6,967 | **0.252** | control — ≈ independence, on real lines back to 1999 |
| `rb1rush_x_qbpassunder` | 5,241 | 0.230 | genuinely negative |
| `passyds_x_rb1rush` | 5,241 | 0.230 | same two columns, opposite direction |

All six pass transport validation. `teamtotal_x_passtd` is the strongest and
most era-stable candidate; `spread_x_total` is the end-to-end sanity check,
not a trading candidate.

## 3-leg extension ideas (vine copula)

For a 3-leg ticket, decompose the joint into a **vine**: a cascade of
bivariate copulas, each linking a pair, with later pairs conditioned on the
first. Example structure sketched in discussion:

- Game total Under → RB rushing yards Under → alternate spread direction

Different copula families (empirical, Gaussian, or another) could be fit at
each link if the tail behaviour differs pair to pair — this repo's
`gaussian_joint`/`empirical_joint` split in `src/stats.rs` already gives both
options for a single pair.

**What this would actually need, not yet resolved:**

- A **vine structure** — which pair is modelled directly and which
  conditionally. The choice is not free; it changes which correlations the
  fit can represent at all.
- **Sample size grows fast.** The 2-leg atlas already floors intervals below
  `MIN_CLUSTERS_FOR_INTERVAL` (25) and strata below `MIN_STRATUM_OBSERVATIONS`
  (200) — see `src/atlas.rs`. A 3-way joint at three simultaneous thresholds
  will be sparser by construction; early-season strata may not clear that bar
  at all (see Open Questions below).
- **The bootstrap gets more expensive.** Conditional resampling for a vine
  needs its own construction; the current `cluster_bootstrap_joint*` machinery
  in `src/stats.rs` is built for one pair, not a conditioned cascade.

## Implementation priority

1. Validate the existing 2-leg pipeline against real Week 1 data (in
   progress — `data/processed/sgp_prices.csv` needs real SGP prices from the
   board).
2. Only then evaluate whether a 3-leg vine extension is worth the added
   estimation cost above, on measured 2-leg results.
3. If it is, build the vine fit and a simulation module for backtesting EV
   against realised outcomes.

## Open questions

- Which SGP combinations are actually offered at DraftKings/FanDuel/Fanatics
  on a given week? (This gates which families/legs are even collectible —
  see `crates/nfl-sgp/src/legs.rs` for the parser and `METHOD.md` for the
  pre-registered families.)
- How thin do strata get in early-season data (Weeks 1–4), where the as-of
  Wr1/Rb1 lookback (`SUBJECT_WINDOW_GAMES` = 8, `MIN_SUBJECT_HISTORY_GAMES` =
  4 in `src/games.rs`) has the least history to work with?
- Optimal model-weight calibration if a vine estimate is ever blended with
  market prices — the baseball project measured the market-blend weight at
  0.014 (i.e., essentially zero), but NFL SGP markets are plausibly less
  liquid than Kalshi's MLB strikeout markets, so that number may not transfer.
