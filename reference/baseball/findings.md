# Pre-registered: the five-book pass on team-K and team-hits-under

Written before any post-2026-08-19 collection. The statistic and the bar are
fixed here, ahead of the data — the quarter-Kelly and 12+pp reversals are what
writing the bar afterwards looks like.

## Scope

- ~5 slates, starting 2026-08-23. Families: **k_x_teamk** (primary — the only
  family with a measured under-allowance, +0.087 devigged at n=3) and
  **k_x_teamhitsunder** (secondary — atlas φ +0.198, era-stable, never measured
  on the book side; the one collected row was the wrong quadrant and an
  availability record, DK refused the combo).
- One **indep** row per slate as a pipeline canary. It no longer feeds the
  decision: the statistic below never reads hold.
- k_x_er / k_x_outs / k_x_f5ml stop: overpriced on the devigged basis, and
  strong φ_true did not save deep_start — books' sims price the obvious pairs.
- Per row, from each book that lists the SGP: both leg descriptions, **both
  sides of the team leg**, and the SGP price. Team strikeouts and team hits
  are not carried by the feed, so those stay hand-collected; pitcher-K
  marginals come from the leg board (consensus at 2+ books, sheet fallback).
- Mid-probability lines preferred: the two-leg load grows toward the tails,
  and line choice is the one lever that shrinks the toll before shopping.

## Statistic (fixed)

Per family, per game: the best book's EV = P_true/q − 1 at devigged consensus
marginals (the executable strategy is shopping, so the per-row statistic is
the max over books). Then the mean across games with a game-cluster bootstrap
95% interval.

## Bar (fixed)

- Primary: the k_x_teamk interval sits **entirely above zero** → the thread
  continues to a sizing study. Anything else — below zero or straddling —
  closes it.
- Secondary: same rule on its own slates; it keeps the thread open only by
  clearing the bar itself.
- **No per-cell wins.** ~5 books × ~5 slates × 2 families at ±a few pp per
  cell guarantees a positive cell somewhere even under fair pricing; a single
  row, book, or slate is not evidence. The family mean is what a bet earns.

Prior on record: shopping plus line choice closes at most half of the −6.2%
DraftKings read; the likely result is a clean negative. That still pays for
itself — it retires the thesis on five measured books, not one book's margin.

# The leg board: every book's price, and the φ_book basis was wrong

## Goal

Automate the half of the nightly collection a feed can carry, and put φ_book on
the same scale as φ_true. Two things were throttling the thesis: only 8 of 32
collected rows could be scored at all (EV needs devigged marginals, which needs
both sides typed by hand), and every number came from one book.

## Implementation

- [x] `src/data/odds.rs`: The Odds API client in the library — events (free),
      per-event odds, `x-requests-remaining` tracking, and Over/Under pairing
      into `TwoSidedQuote`, which is **always both sides**. A one-sided outcome
      is dropped, never emitted with a hole, so a vig-loaded number can't be
      mistaken for a fair one. Two-way moneylines yield both perspectives; a
      three-way (listed tie) is skipped. Nothing is cached — a live price is
      the opposite of a completed box score. `book_probe.rs` deliberately
      untouched.
- [x] **`slate_date`**: a game belongs to slate `D` when first pitch falls in
      `[D 10:00Z, D+1 10:00Z)`. A 9:10pm Pacific start is 04:10Z *tomorrow*, so
      a raw `commence_time[..10]` key would have filed every late game under the
      wrong day and silently missed the join. 10:00Z is 6am ET / 3am PT — no
      game starts there, and unlike a timezone conversion it needs no DST table.
- [x] `examples/leg_board.rs`: `--games` required (no selection prints the free
      slate listing and exits), `--dry-run` stops before the first billed call.
      Appends `data/processed/leg_board.csv`, rebuilds `outputs/leg_consensus.csv`
      — median devigged probability per leg, book count, spread. Latest row per
      book wins, so re-pulling a slate refreshes rather than double-weighting.
- [x] `exact_line_phi.rs` joins the board: marginals resolve **consensus (2+
      books) → sheet → unscoreable**, in one function both the EV path and the
      hold derivation call, so they cannot diverge. Provenance is four columns.
      Ambiguous surnames are a loud refusal, not a guess.
- [x] `phi_book_devigged` / `gap_devigged` added beside the raw-basis columns.

## Verification

- [x] Regression gate: with no board present, `exact_line_phi` reproduces the
      previous run **byte-identically on all 26 shared columns**, 28 rows, same
      8 EVs. Only the 6 new columns differ.
- [x] 167 lib tests (+9), leg_board 9, exact_line_phi 10, `pnpm check` 158
      files 0 errors, `pnpm test` 9, strict clippy and fmt clean on every
      touched file.

## Review — the gap was a basis error, and it was the whole story

`phi_book` was inverted against **raw, vig-loaded** leg implieds while
`phi_true` is applied at **devigged** ones. Different scales; their difference
is not an under-allowance. Put on one basis:

| family | gap (raw basis) | gap (devigged) | EV |
|---|---|---|---|
| **indep control** | **+0.193 / +0.105** | **+0.047 / −0.063** | −8.0% / −15.7% |
| k_x_teamk (×3) | +0.203 / +0.229 / +0.247 | **+0.072 / +0.094 / +0.094** | −6.9 / −5.6 / −6.0% |
| k_x_er | +0.110 | **−0.036** | −14.2% |
| k_x_outs | +0.077 | **−0.067** | −15.9% |
| k_x_f5ml | +0.091 | **−0.073** | −15.9% |

**The control proves the fix.** The `indep` family's φ_true is ≈0 by
construction, so its φ_book must be ≈0 too. On the raw basis it read +0.19 and
+0.11 — pure fabricated under-allowance. On the devigged basis it straddles
zero. Everything the raw column ever said was inflated by the same amount.

Consequences:

1. **Team-K's real under-allowance is +0.087, not +0.227** — 2.6x smaller. It
   is still the only family with one.
2. **Every other family goes negative.** Books allow *more* correlation than
   reality on ER, outs and F5. Those are not near-misses to be closed by
   shopping; they are correctly priced or overpriced, and should stop being
   collected as edge candidates.
3. The earlier write-up's `+0.057/+0.076/+0.137 at hold 6.38%` and this file's
   own `+0.090 mean` are **both superseded** — they were the raw basis at a
   hold since re-derived. Quote EV, or quote `gap_devigged`; the raw `gap`
   column is kept only so pre-board rows stay comparable.
4. Nothing about the verdict moves: EV never read φ_book. Team-K still sits at
   **−6.2% mean** against a ~13% two-leg load. The edge is real, measured, and
   about half the size of the toll.

# Exact-line φ_true: the team-K gap is real — and so is the vig load

## Goal

Every collected SGP row judged at its exact leg thresholds, replacing the
nearest-atlas-cell approximation. The Phase 3 decision statistic (the gap)
gets exact inputs.

## Implementation

- [x] `src/atlas.rs`: the dataset-aware measurement layer lifted from the
      atlas example (Obs trait, ViewObs/BatterObs, builders, measure engine).
      correlation.rs stays pure stats. Atlas example refactored to call it —
      regression gate passed: `outputs/correlation_atlas.csv` byte-identical.
- [x] `LegSpec` (12 variants) + desc parser with family cross-check and loud
      refusals (whole lines, garbage, win legs, unsupported). Tests pin every
      desc pattern in the live sheet.
- [x] `examples/exact_line_phi.rs`: per-row exact φ_true + cluster CI + φ_book
      at a hold derived from the sheet's indep rows (6.38% at 3 rows).
      `outputs/exact_line_phi.csv`. 158 tests green.
- [x] Landmarks reproduced in selftest (+0.1976 / +0.1474 / +0.2599 / −0.0131);
      orchestrator recomputed the three team-K cells independently to 1e-3.

## Review — first exact-line verdicts

- indep control −0.006 ≈ 0 (margin calibration works); k_x_outs −0.002
  (**books price the workload pair correctly**, as theorized).
- **k_x_teamk +0.090 mean, all three rows positive** (+0.057/+0.076/+0.137):
  the under-allowance survives exact lines. The earlier +0.15–0.20 read was a
  nearest-cell artifact.
- k_x_batterk +0.067 mean (the Torkelson −620 row carries extreme-price
  inversion leverage — discounted). er/hits/f5 thin (+0.01–0.02).
- **The EV translation tempers every gap.** On the Sánchez row: offered joint
  0.4587, true joint at devigged marginals 0.4242 → EV −7.5%. Compounded leg
  vig (~9%) + SGP margin (~6%) ≈ 17pp of joint load vs ~2pp of under-allowance.
  The φ-gap framing double-counts leg vig; the pre-registered statistic is
  **EV = P_true/q − 1 at devigged marginals**, full stop.
- Consequences: collect **both sides of each leg** (or anchor the K leg to
  Kalshi's mid) so marginals can be devigged; shop SGP prices across books
  (2.4pp of joint between +118 and +130 — same size as the edge); hunt
  mid-probability legs where the load is smallest.

# The correlation atlas: measured φ for every candidate SGP family

## Goal

The φ_true half of the parlay thesis on permanent, tested machinery — binary
correlations between a starter's strikeout over and every candidate partner
leg, replacing the throwaway prototypes.

## Implementation

- [x] `src/correlation.rs`: exact Bernoulli φ (shares `sgp::phi_from_joint` —
      book-side and outcome-side numbers are the same formula inverted, which
      is what makes them comparable), pitcher-year cluster bootstrap
      (percentile, 25-cluster floor matching paper_stats, seeded xorshift,
      deterministic bit-for-bit), era splits.
- [x] `examples/correlation_atlas.rs`: 85 rows — families × fixed rungs
      (5/6/7) × per-pitcher market line (within-season mean, rounded) →
      `outputs/correlation_atlas.csv`. Batter-level joints exist because of the
      hits/team_id extension: starter K over × specific batter 1+ K
      (n=139,896 batter-starter pairs) and × batter hits ≤1.

## Verification

- [x] Landmarks reproduce the prototypes exactly: K≥6×ER≤2 **+0.1976**
      [+0.182, +0.214]; hits≤5 +0.1474; outs≥18 +0.2599; combined ER≤5 +0.115.
- [x] Controls: duel **−0.013** [−0.030, +0.003] (the null replicates);
      part-whole team-Ks +0.48…+0.55.
- [x] 148 tests, strict clippy, fmt, example tests.

## Review — the family table (fixed rung K≥6)

| family | φ | note |
|---|---|---|
| starter ER≤2 | **+0.198** [+0.182, +0.214] | strongest tradable one — but is the leg offered? |
| deep start (18+ outs) | +0.260 [+0.243, +0.276] | strongest overall, most obvious to books' sims |
| team hits ≤7/8/9 | +0.19–0.20 | strong, team props widely offered |
| **batter 1+ K** | **+0.177** [+0.172, +0.182] | n=139,896, era-stable; first measurement |
| outpitch (F5/ML proxy) | +0.166 ties-excluded / +0.124 ties-lost | convention-dependent; leg type decides |
| combined starter ER | +0.108–0.118 | game-total dilution, as predicted |
| batter hits ≤1 | +0.082 | real but thin |
| duel (both starters' Ks) | ≈ 0 | crossed off — null replicated |

Era splits 2023→2026: every carried family stable; no single-season stories.

# Clean-data re-measure: the probe's real number, and Kalshi's in-play trading

## Goal

Every number the correlated-parlay strategy relies on must be provably pregame:
the book-vs-Kalshi anchor across slates, and the paper path's own exposure to
in-play quotes.

## Implementation

- [x] `examples/book_probe_report.rs`: cross-slate analysis of the append-only
      `data/processed/book_probe.csv` — pregame filter re-implemented locally
      (`minutes_to_first_pitch >= 0`, unreadable = not pregame, pinned by test),
      devigged-consensus/model/sim RMS vs the Kalshi mid per slate and pooled,
      the pre-registered ≤2pp verdict printed, `outputs/book_probe_report.json`.
- [x] `examples/kalshi_inplay_audit.rs`: does KXMLBKS trade after first pitch?
      21 settled markets (1 rung/pitcher, ≤3 pitchers/slate, 2026-08-07→08-13),
      hourly candles vs scheduled first pitch parsed from the ticker
      (`26JUL201910` → 19:10 ET; validated by 1.5–6h gaps to close_time).
- [x] **First-pitch gate** (paper.rs): `match_markets` skips any market whose
      scheduled start is past at recording time — no entry, no refresh, no
      position; unparseable time treated as started (book_probe discipline).
      `record_slate_at` threads typed time for tests; an already-logged unlocked
      row keeps its last pregame quote forever.
- [x] `kalshi::scheduled_first_pitch_utc` + hand-rolled US Eastern DST rule
      (second-Sunday-March 07:00 UTC → first-Sunday-November 06:00 UTC;
      chrono-tz deliberately not added).

## Verification

- [x] Clean pooled table (two slates, 149 pregame rows, 37 rungs): book
      consensus **2.01pp** (slates 2.16 / 1.81), model 9.28 / 15.17, sim 7.34
      (one slate; needs posted lineups). The report reproduces the documented
      9.28pp clean figure exactly.
- [x] Audit: 21/21 markets moved >2pp within 2h of first pitch; median max
      early move **43.0pp**; 20/21 railed while the game was certainly on;
      17/21 railed ≥1.5h before close_time. **KXMLBKS trades in-play.**
- [x] Gate test: a post-first-pitch `record_slate_at` leaves the log
      CSV-byte-identical even with a quote that would have opened a YES.
- [x] `cargo test --workspace` (142), strict clippy, fmt.

## Review

- The ≤2pp bar sits at **2.01pp pooled — a tie, not a verdict.** Two slates is
  not a sample; keep accumulating before the divergence thread formally closes.
  The practical content is already clear: books and Kalshi price the same
  information within ~2pp, so **market prices are a trustworthy marginal
  anchor** for the parlay strategy — and the sim (7.34) keeps beating the model
  (9.28+) to the market's number.
- The model's slate swing (9.28 → 15.17) decomposes to a few pitcher-specific
  misses (Bieber −12/−13pp at tight 1–3c spreads), not contamination: 5-start
  form lagging fresher market information. Irrelevant to the strategy —
  marginals were never going to be ours.
- The 2026-08-12 15.64pp figure is retired: 8 in-play rows, now filtered at
  both collection and analysis.

# Batter game lines gain hits and team attribution

## Goal

The correlation atlas needs per-batter hits and an exact join from each batter
to the starter his side faced. The boxscore feed carried both all along; the
pipeline dropped them at the serde boundary.

## Implementation

- [x] `lineups.rs`: `BattingStat` + `BoxscoreBatterLine` gain `hits`, populated
      in `normalize_side`. The fixture (a trimmed payload) gained plausible hits
      values on its seven batting lines.
- [x] `batters.rs`: `BatterGameLine` + `BatterLineRow` gain `hits` and
      `team_id` (team was parsed and discarded all along); `#[serde(default)]`
      back-compat so the pre-extension six-column CSV keeps reading (0 =
      "predates the extension"), the paper.rs pattern.
- [x] Rebuilt `data/processed/batter_lines.csv` (8,705/8,705 games, 100%
      lineup coverage, 156,690 rows): league H/PA 0.2202, K/PA 0.2228.
      `training_starts_v3.csv` rewritten identically modulo the three
      pitch-outcome columns the struct gained in feature schema 2 (empty here
      by design; the v4 join populates them). baseballr_map.md updated.

## Verification

- [x] Smoke test (100 games) before the full run; backups diffed after — the
      original 22 columns are content-identical.
- [x] Tests: fixture hits parse, per-side team attribution, six-column
      back-compat, round-trip. Suite green.

# SGP price inversion tooling

## Goal

Measure a book's correlation allowance from prices alone: offered SGP price vs
the mechanical product of the leg prices. That allowance (φ_book), net of
margin, is the quantity the whole strategy hangs on.

## Implementation

- [x] `src/sgp.rs`: implied probability, devig, decimal, mechanical/offered
      joint, `price_ratio` (model-free headline), `phi_from_joint`,
      `implied_phi_book` (hold-parameterized over [0.10, 0.20]), `ev_per_unit`.
      Pure, no deps.
- [x] `examples/sgp_invert.rs`: reads hand-collected `data/processed/sgp_prices.csv`
      (date, book, region, game, family, legs+prices, sgp price,
      minutes_to_first_pitch, lineups_posted, notes), excludes in-play/untimed
      rows, per-row inversion + (book, family) summary, writes
      `outputs/sgp_inversion.json`. `--selftest` for plumbing.
- [x] Two pre-existing clippy-1.97 lints in paper_stats.rs fixed minimally
      (scoped allow; identical-value literal truncation).

## Verification

- [x] Known-answer tests on the hand-inverted ticket both ways (2-leg:
      −130/−119/+244 → ratio 0.9465; 3-leg decomposition → 0.2311/1.258),
      synthetic φ=0.12 @ hold 0.15 recovered to 1e-9, φ symmetry,
      devig(−115,−115)=0.5.
- [x] Suite green (137).

