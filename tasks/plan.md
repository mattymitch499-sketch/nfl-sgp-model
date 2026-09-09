# Implementation plan: the NFL atlas and scorer on the joint-probability design

Written 2026-09-02 before any code. This is the specification each build phase
is reviewed against. Where this document and the code disagree, the code is
wrong until this document is amended with a reason.

## 0. Goal

An operational pipeline:

```text
nflverse outcomes
  → per-family dependence objects (empirical survival copula on
    within-subject standardised ranks, stratified by a pre-game covariate)
  → evaluated at tonight's devigged consensus marginals
  → P_true(both legs) with a cluster-bootstrap interval
  → EV = P_true / q − 1 against the offered SGP price
  → best book per game, family mean, game-cluster interval, verdict at the
    pre-registered bar
```

plus a held-out validation of the one assumption the design rests on: that
the dependence structure is stable across seasons and transports across
thresholds.

## 1. The mathematics

### 1.1 Objects

A **family** is an ordered pair of legs (A, B). A **leg** is (subject role,
stat, direction). Roles: `Team`, `Qb` (starting quarterback), `Wr1`, `Rb1`,
`OppQb`. Stats are continuous or count per-game quantities from nflverse.
Direction is `Over` or `Under`.

An **observation** is one game-anchor: for same-team families, one row per
(game, anchor team); for cross-team families, one row per game with the home
side as anchor. Each observation carries the two subject values `x_A`, `x_B`,
a pre-game covariate `c`, and one or more cluster ids.

### 1.2 Within-subject standardisation (the between/within trap)

The market's marginal for a leg already prices the subject's level. The parlay
prices only the *within-game co-movement given the subjects*. Pooling raw
values across subjects would count "a good QB and a good WR are both usually
high" as co-movement; it is not, it is level.

For subject `s` in season `y` with games `j = 1..n`, `n ≥ N_MIN = 6`, and game
`i`:

```text
z_i = (x_i − m_{−i}) / s_{−i}
```

where `m_{−i}` and `s_{−i}` are the mean and sample standard deviation (n−2
denominator, since one game is left out) over the other `n − 1` games.
Leave-one-out, so a game never centres itself. If `s_{−i} = 0` the observation
is dropped. Team stats are standardised per team-season the same way.

This treats each subject's distribution as a location–scale shift of a common
shape. That is the assumption a book's simulator makes when it applies one
dependence structure to player-specific means, and it is what makes pooling
across subjects legitimate.

### 1.3 Uniform scores (the distributional transform)

Pool `z` across all subjects in the (family, stratum). Rank the pool with ties
broken by a **seeded uniform jitter** — Rüschendorf's distributional transform
— and set `u_i = (r_i − 0.5) / n`. Ties are the rule for count stats (passing
touchdowns are 0–7), and without the jitter the copula of a discrete margin is
not identified. With it, `u` is exactly uniform and, for a count stat with
distribution function `F`, the event `{Y ≥ k}` equals `{U > F(k−1)}` almost
surely — which is why a market marginal `p_B = P(Y ≥ k)` maps to the region
`{v > 1 − p_B}` with no further assumption. The seed is fixed and recorded;
the jitter dependence is reported by re-running with a second seed.

### 1.4 The empirical survival copula, evaluated at market marginals

Given tonight's devigged marginals `(p_A, d_A)`, `(p_B, d_B)`:

```text
W_A = { u > 1 − p_A }   if d_A = Over
      { u ≤ p_A }       if d_A = Under          (likewise W_B)

P̂(p_A, d_A, p_B, d_B) = (1/n) · #{ i : u_i ∈ W_A  and  v_i ∈ W_B }
```

Required identities, each a test: independent DGP → `p_A · p_B`; comonotone
→ `min(p_A, p_B)`; countermonotone → `max(0, p_A + p_B − 1)`; Fréchet
consistency `P̂(Over, Under) = p_A − P̂(Over, Over at 1 − p_B)` exactly on the
same sample; `P̂` is monotone non-decreasing in each `p`.

### 1.5 The Gaussian copula as smoothed comparator

Normal scores `n_i = Φ⁻¹(u_i)`, `ρ̂ = Pearson(n_A, n_B)`. Then

```text
P_Gauss(Over, Over) = L(h, k; ρ̂),   h = Φ⁻¹(1 − p_A),  k = Φ⁻¹(1 − p_B)
```

where `L` is the upper-orthant probability of the standard bivariate normal.
Other direction combinations follow by symmetry (`Under` on A means the
region below `Φ⁻¹(p_A)`, i.e. negate `h` and `ρ` as appropriate).

Numerics: `Φ` via a high-precision `erfc` (Cody 1969 rational
approximations); `Φ⁻¹` via Wichura's AS241 (PPND16, ~1e-16 relative); `L` via
Genz (2004) / Drezner–Wesolowsky (1990) Gauss–Legendre quadrature. Tests pin:
Sheppard's exact formula `L(0, 0; ρ) = 1/4 + asin(ρ)/(2π)`; `ρ = 0` gives the
product; `ρ → ±1` gives `min` / `max(0, ·)`; symmetry in `(h, k)`;
`L(h, k; ρ) + L(h, −k; −ρ) = 1 − Φ(h)`; and published tabulated values to
1e-7.

The empirical copula is the **primary** estimator (model-free). The Gaussian
one is reported beside it; a large disagreement in a tail is a warning that
the dependence is not elliptical there, not a reason to prefer either.

### 1.6 Summaries per (family, stratum)

`n`, number of clusters, Spearman `ρ_S` (Pearson on `u`), Kendall `τ`,
Gaussian `ρ̂`, `P̂` on the reporting grid
`(p_A, p_B) ∈ {0.3, 0.5, 0.7}²` × four direction combinations, and — for
comparability with the baseball numbers only —
`φ(0.5, 0.5, Over, Over) = (P̂ − 0.25) / 0.25`.

### 1.7 Uncertainty

Cluster bootstrap, `B = 1000`, vendored xorshift64* seeded and recorded.
Each replicate resamples clusters with replacement, **recomputes the pooled
ranks on the resampled pool**, and re-evaluates `P̂`. Percentile 2.5 / 97.5.
Below `MIN_CLUSTERS_FOR_INTERVAL = 25` clusters, no interval. Cluster unit
per family (§2). For a cross-team family, two one-way cluster bootstraps
(home-side seasons, away-side seasons); the **wider** interval is reported.

### 1.8 Stratification

Each family declares a covariate `c` computed strictly from information
available before kickoff. Strata are terciles of `c` over the family's pooled
observations, plus an `all` stratum. The tercile edges are stored in the
atlas so the scorer maps tonight's `c` onto the same buckets.

### 1.9 Validation of the transport assumption

Split by season. Fit (standardise, rank, copula) on seasons `≤ 2017`;
evaluate on `≥ 2018` with the test set's own ranks. For each grid point
(§1.6): predicted `P̂_train` versus realised `P̂_test`, with a cluster-bootstrap
interval on the test side and the difference in percentage points. Report
per family and stratum. Also per-season `P̂(0.5, 0.5, Over, Over)` for the era
check. A family whose realised joints fall outside the predicted intervals at
more than the nominal rate has failed the transport assumption and must not
be scored — the scorer refuses it.

For team legs with historical closing lines (`games.csv`, 1999→), a second
validation uses the **market-centred residuals** `result − spread_line` and
`total − total_line` directly — no standardisation, real historical marginals —
as the end-to-end check of the whole chain.

### 1.10 Decision statistic (unchanged from METHOD.md)

Per row: `q = 1 / decimal(SGP price)`, `EV = P_true / q − 1`, with
`P_true = P̂` at the row's marginals and stratum. Also reported, not decided
on: `EV_indep = p_A p_B / q − 1` and the book's joint relative to independence
`q / (p_A p_B)`. Per (family, game): best-book EV. Per family: mean over
games, game-cluster bootstrap 95% interval (`B = 2000`). Verdict: interval
entirely above zero continues; anything else closes.

## 2. Families (v1)

| key | leg A | leg B | covariate (as-of) | cluster |
|---|---|---|---|---|
| `teamtotal_x_passtd` | Team points Over | Qb passing TDs Over | team's passing share of offensive TDs (prior season + season to date, ≥ 8 games) | team-season |
| `passyds_x_wr1recyds` | Qb passing yards Over | Wr1 receiving yards Over | Wr1's target share over the previous 8 team games (≥ 4; week 1 uses prior season) | qb-season |
| `rb1rush_x_qbpassunder` | Rb1 rushing yards Over | Qb passing yards Under | closing spread, anchor perspective: favourite by ≥ 3 / within 3 / underdog by ≥ 3 | team-season |
| `bothqb_passyds` | home Qb passing yards Over | away Qb passing yards Over | closing total | home team-season and away team-season, wider wins |
| `passyds_x_rb1rush` | Qb passing yards Over | Rb1 rushing yards Over | closing spread | team-season |
| `spread_x_total` | home covers | game Over | closing total | home team-season |

`passyds_x_rb1rush` is measured as a **control candidate** — the pair the
DFS literature reports as near-uncorrelated. `spread_x_total` is the
end-to-end validation family (§1.9) and a calibration family: books have
priced sides-with-totals for decades.

### 2.1 Subject resolution, strictly pre-game

- **Starting Qb**: `games.csv` `home_qb_id` / `away_qb_id`, confirmed present
  in that game's player rows with `attempts ≥ 10`; otherwise the team's QB
  with the most attempts that game (fallback counted and reported).
- **Wr1**: among the team's WR and TE, the player with the most targets over
  the previous 8 team games (season to date, then the prior season's tail),
  requiring ≥ 4 games of history; week 1 uses the prior season's full total.
  Ties → higher prior-season targets.
- **Rb1**: the same rule on carries.
- If the as-of Wr1/Rb1 does not appear in the game's player rows (inactive),
  the observation is **dropped**, never replaced by the realised leader. The
  book prices a named player; choosing the leader after the fact conditions
  on the outcome and inflates the joint.
- Standardisation (§1.2) is per (player, season) over all that player's games
  in the season, regardless of role in any single game.

### 2.2 Data scope

Regular season only. Player and team weekly files 2010–2025; `games.csv`
1999→ for `spread_x_total`. The current season's weekly file is refreshed
each week for the scorer's covariates.

## 3. Leg parsing for the scorer

The sheet's `family` column declares roles; the parser extracts subject name,
stat, line and direction from `leg_desc`, and the scorer trusts the family for
role assignment. Accepted patterns (case-insensitive, punctuation-tolerant):

```text
<Player> over|under <line> pass yds|passing yards
<Player> over|under <line> pass tds|passing touchdowns|passing tds
<Player> over|under <line> rush yds|rushing yards
<Player> over|under <line> rec yds|receiving yards
<Player> over|under <line> receptions|rec
<Team> team total over|under <line>
<Team> ml | <Team> <±spread>
game total over|under <line>
```

Anything else is a loud refusal naming the row. Names go through
`names::normalize_name`.

## 4. Marginal resolution (one function, both legs)

Consensus devigged probability from `outputs/leg_consensus.csv` on
(slate, market, folded subject, line) with `n_books ≥ 2` → else the sheet's
own other side, devigged → else the row is **unscoreable** and says so.
The same function feeds the EV path and any hold derivation, so they cannot
diverge.

## 5. Covariates at scoring time

- closing spread and total: consensus from the leg board's `spreads` and
  `totals` markets for the game (so pull `spread,total` with the board).
- target share, passing-TD share: computed as-of from the current season's
  weekly file plus the prior season, by the same functions the atlas uses.
- an optional `covariate` column on the sheet overrides, and is logged as such.

## 6. Outputs

| file | contents |
|---|---|
| `outputs/nfl_atlas.json` | per (family, stratum): seasons, n, clusters, tercile edges, the `(u, v, cluster ids)` arrays — what the scorer loads |
| `outputs/nfl_atlas.csv` | the human table: summaries and the reporting grid with intervals |
| `outputs/nfl_atlas_validation.csv` | §1.9: predicted vs realised per grid point, per family and stratum |
| `outputs/nfl_atlas_seasons.csv` | per-season `P̂(0.5, 0.5)` and Spearman |
| `outputs/sgp_scored.csv` | row-level: marginals with provenance, stratum, `P_true` + CI, `q`, EV, EV_indep |
| `outputs/sgp_verdict.json` | per family: best-book EV per game, mean, CI, verdict |

## 7. Build phases and review gates

Each phase is built by an agent against this document, then reviewed. A
phase is accepted only when its gate passes and the review finds nothing
wrong; otherwise it goes back with comments.

| phase | files | gate |
|---|---|---|
| 1 | `src/stats.rs` | known-value tests to 1e-7 for `Φ`, `Φ⁻¹`, `L`; Sheppard; Fréchet identities; a synthetic Gaussian-copula DGP (n = 20,000, ρ = 0.4) recovers `ρ̂` within 0.02 and `P̂` within 0.01 of `L` at every grid point; bootstrap deterministic under a fixed seed |
| 2 | `src/games.rs`, `src/families.rs` | fixture tests; a leakage test proving the Wr1/Rb1 choice uses only prior games; dropped-not-replaced pinned; each game appears once per family; real-data counts printed |
| 3 | `src/atlas.rs`, `examples/nfl_atlas.rs` | runs on 2010–2025; writes all four atlas outputs; bit-for-bit reproducible; validation table present for every family and stratum |
| 4 | `src/legs.rs`, `examples/score_sgp.rs` | `--selftest` reproduces hand-computed EV on synthetic rows; every parser pattern pinned; precedence pinned; unscoreable rows loud; refuses families that failed §1.9 |
| 5 | `METHOD.md`, `README.md`, `tasks/todo.md` | pre-registration filled in on the joint-probability statistic; every number in the docs traceable to an output file |

All phases: `cargo test --workspace`, example tests, `clippy -D warnings`,
`fmt --check`. No new dependencies. Randomness only through the vendored
xorshift. No `unwrap` on data paths outside tests.
