# 2026-09-08 — Collection automation, ahead of week 1

Season opens 2026-09-09; week 1 Sunday is 09-13. The measurement side was
finished on 09-03 and has never seen a price, so this session automated
everything around the one number that cannot be automated.

**The honest answer on automation:** the offered SGP price is not sold by any
feed (The Odds API does not carry parlays; OpticOdds is ~$5k/month), so it must
be read off each book's app. Browser-driving DraftKings/FanDuel/Fanatics was
declined — not on terms-of-service grounds but because those are the three books
the bets would be placed at, and a flagged account ends the project before the
measurement does. Everything else is now generated.

## Shipped

- **`examples/slate_sheet.rs`** — emits a pre-filled `sgp_prices.csv`: game,
  family, both leg descriptions with the real player and real line per book,
  both sides of both legs, kickoff timing, covariate. Only `sgp_price` is left
  blank. **Spends no credits** — it reads the board file `leg_board` already
  wrote, so a generation mistake costs nothing and the whole thing is testable
  offline. Verified end to end: 18 tickets (6 families × 3 books) from a
  synthetic DEN @ KC board, 0 skipped, straight through `score_sgp`.
- **`src/depth.rs`** — official depth charts, the user's correction. The
  snapshot file is *not* date-ordered and spans March to today, so the rule is
  newest-snapshot-per-team; taking the file's tail yields an offseason chart
  that parses perfectly and names traded players. Verified against the live
  46 MB 2026 file: 32 teams, all dated today.
- **`games::pre_game`** — the structural gap. `GameTable::build` only emits rows
  for games with final scores, so nothing could compute an as-of covariate for
  an *upcoming* game. Rather than write a second aggregator, `RowContext` now
  accepts `position == slots.len()` ("the next game") with the season supplied.
  Pinned by `pre_game_matches_the_row_the_table_would_have_built`: the two paths
  must agree exactly, or a live row is scored against a stratum its own history
  would not have placed it in.
- **`games::TEAM_NAMES`/`team_code`** — the board says "Kansas City Chiefs", the
  data says "KC", and nothing bridged them. Aliases cover renames
  (Washington ×2, Oakland, San Diego, St. Louis); unknown names refuse.
- **`legs::LegStat::for_stat`** — the market set is now derived from the
  selected families rather than a second hand-maintained list that could drift.
- **`nflverse::download_if_published`** — a season that has not started has no
  weekly files, which must not abort the depth-chart fetch beside them.

## Bug fixed: game totals could never reach the consensus board

`parse_leg` gives a game total no subject (correctly — it names nobody);
`leg_board` files it under `"game"` (correctly — the wire gives its two sides no
description and they must pair under something). Both right locally, and they
never met: the lookup searched `""`, the board held `"game"`, so **every**
`spread_x_total` leg B silently fell back to one book's devig instead of the
three-book median — the exact single-book weakness the leg board exists to
remove. Nothing errored. Reconciled at the one place they meet, pinned by a test
asserting the marginal comes back `Consensus(3)` and not `Sheet`.

## Methodology: three rules pre-registered before any price

An earlier draft cited METHOD.md as already containing the "prefer
mid-probability lines" discipline. **It did not** — that is the *baseball*
study's pre-registration (`reference/baseball/findings.md:22`), and this project
never adopted it. Line choice moves the measured statistic, so building it into
the generator adds a pre-registration clause. Now written into METHOD.md as a
dated amendment, together with two more the generator forces: **anchor on the
home team always** (two anchors from one game would be maximised over together
by `best_by_game`), and **the book names the quarterback** (verified: all
sixteen 2026 week-1 schedule rows carry a spread and total but *empty* QB ids,
so plan §2.1's rule cannot run pre-game).

## Verified, not assumed

- The refactor moved no measurement: at the same 400 reps the atlas reproduces
  every number byte-identically (0.3522 / 0.3256 / 0.2303 / 0.2662 / 0.2523) and
  all six families still transport.
- The spread sign flip is right: a book's "KC −3.0" becomes a covariate of
  **+3.0**, which `spread_bucket` calls a favourite. Backwards, every favourite
  would sit in the underdog stratum with nothing erroring.
- The generated pass-TD covariate (0.6053) matches an independent hand
  computation of KC's 2025 share (0.605) and lands in `t1`.
- 172 tests (162 library + example suites), clippy `-D warnings` and fmt clean.

## Still open

- **Collect week 1.** Reconnaissance scope: 2–3 games, all six families, all
  three books (~36–54 hand-typed prices, ~25–35 min). The point is to learn
  which families each book will *combine at all* — the primary family pairs a
  team total with that team's QB passing TDs, which is exactly the kind of pair
  books block.
- **The depth-versus-usage agreement study.** The atlas defines Wr1/Rb1 by prior
  usage; the live sheet now uses depth charts. That gap is accepted knowingly
  for now, and where the two disagree the target-share covariate is left blank
  so the row scores against `all` rather than against a covariate describing a
  different player. Measure agreement over 2010–2024 (the per-week depth-chart
  format) and rebuild the atlas if it is below ~85%. Touches only three of six
  families; the primary and the control are unaffected.
- `git init` — still never committed.

# 2026-09-03 — The joint-probability build

Reframing agreed with the user before any code: stop transporting φ, transport
the **dependence structure** instead, and state everything in joint
probabilities. `tasks/plan.md` is the specification written before the build;
`METHOD.md` carries the reasoning and the pre-registration.

## Why the reframe

φ for two binary legs is bounded by its own marginals, so "measure φ on history,
rebuild tonight's joint from tonight's marginals plus that φ" is a modelling
assumption, not a measurement. The transportable object is the copula of the
underlying continuous stats. Standardise within subject (the market prices the
level; the parlay pays for co-movement), rank to uniforms, and read the joint at
tonight's marginals. φ is kept only as a summary for comparison with the
baseball numbers.

## Phase 1 — `stats.rs` (done, 32 tests)

- Cody 1969 `erfc` → `norm_cdf`; Wichura AS241 PPND16 → `norm_ppf`; Genz 2004 /
  Drezner–Wesolowsky bivariate upper orthant with the near-singular branch.
- Leave-one-out standardisation, uniform scores with seeded tie-breaking jitter
  (Rüschendorf's distributional transform), empirical survival copula at market
  marginals, Gaussian comparator, Spearman, Kendall τ-b, φ, cluster bootstrap
  that **re-ranks inside every replicate**.
- **Verification, measured not asserted.** `bvn_upper` vs an independent Simpson
  quadrature: worst absolute error **3.0e-14** over eleven (h, k, ρ) points
  spanning both branches and both signs. Sheppard's identity, the Fréchet
  limits, the reflection identity and the ρ = 0 product all pinned. Reference
  values are **derived, not transcribed** — the agent wrote a second-source
  quadrature rather than quote a remembered table digit, and said so.
- **The DGP gate**: 20,000 pairs from a ρ = 0.4 Gaussian copula recovered
  ρ̂ = 0.394, Spearman 0.3789 (truth 0.3846), τ 0.2578 (truth 0.2620), and the
  empirical joint within **0.0026** of the Gaussian one at the worst of 36 grid
  points (bar: 0.01). This is the test that says the four direction conventions
  and the orthant numerics agree with each other.
- The bootstrap widens by more than 2x when a common shift is added inside each
  cluster — the test that proves it respects clustering rather than rows.

## Phase 2 — `games.rs`, `families.rs` (done, 39 tests)

- Subject table 2010–2025: 8,350 team-games, 0 missing from the schedule.
- **QB from the schedule column 98.3%**, fallback 1.7%, 1 unresolved.
- As-of Wr1/Rb1 from the previous 8 team games (week 1 uses the prior season
  entire), **dropped and never substituted** when the named subject did not
  dress: Wr1 resolved 7,368 / dropped 854 (10.4%), Rb1 6,688 / 1,534 (18.7%).
- Leakage tests: a receiver who explodes in game *i* must not be Wr1 *for* game
  *i*. Same for the back. Both pinned.
- 2024 spot check names real starters — St. Brown, London, Chase; Bijan
  Robinson, Najee Harris, Breece Hall; Rodgers, Mayfield, Nix.
- Cluster ids are FNV-1a over a canonical string, **not `DefaultHasher`** (whose
  output is not stable across Rust releases — the predecessor project was bitten
  by exactly that). 3,288 subject-seasons, 0 collisions.
- Relocations canonicalised (OAK→LV, SD→LAC, STL→LA) so prior-season lookups
  survive them; 32 teams after folding.
- Six families built; `spread_x_total` reaches 1999 with 6,967 observations
  because closing lines are the one series that goes back that far.

## Phase 4a — `legs.rs` (done, 15 tests)

The sheet's leg grammar, tolerant of case, punctuation and word order, and
**refusing rather than guessing**. The cross-check that earns it: a row's two
legs must agree with the family it declares on stat *and* direction —
`rb1rush_x_qbpassunder` and `passyds_x_rb1rush` are the same two columns and
differ only by direction, so a mistyped over would otherwise be scored against
the wrong dependence and look fine.

## Phase 5 — docs (done)

`METHOD.md` rewritten on the joint-probability basis, pre-registration filled
in, 14 lessons each tagged with the number that taught it. `README.md` rebuilt.

## Phase 3 — the atlas (done, 12 tests) and the first real measurement

Built on 2010–2025 (`spread_x_total` on 1999–2025): 24 strata across 6
families, reproducible bit-for-bit, jitter-seed sensitivity 0.0027 at worst.

**Measured dependence, `all` stratum, at 0.5/0.5 Over/Over — independence is
0.2500:**

| family | n | clusters | Spearman | joint [95% CI] | φ |
|---|---|---|---|---|---|
| `teamtotal_x_passtd` | 7,510 | 510 | +0.566 | **0.3522** [0.346, 0.357] | +0.409 |
| `passyds_x_wr1recyds` | 5,488 | 527 | +0.431 | 0.3256 [0.319, 0.334] | +0.303 |
| `bothqb_passyds` | 2,250 | 415 | +0.093 | 0.2662 [0.255, 0.275] | +0.065 |
| `spread_x_total` | 6,967 | 861 | +0.016 | 0.2523 [0.247, 0.258] | +0.009 |
| `rb1rush_x_qbpassunder` | 5,241 | 466 | −0.124 | 0.2303 [0.224, 0.238] | −0.079 |
| `passyds_x_rb1rush` | 5,241 | 466 | −0.124 | 0.2303 [0.224, 0.238] | −0.079 |

Reading:

- **The part-whole family is enormous** — φ +0.41, joint 0.352 against 0.250.
  Far stronger than anything in the baseball atlas outside its own part-whole
  pair. This is the primary family and the prior that put it there survives.
- QB × WR1 is +0.30, which is real and is also the pair every engine prices
  best. It is the calibration family, not a candidate.
- The two spread families are the **same numbers** because they are the same
  two columns; they differ only in direction, which is applied at evaluation.
  Both show a genuinely *negative* dependence (joint 0.230 < 0.250) — the DFS
  literature's "QB–RB is uncorrelated" is not quite right at the threshold
  level.
- **`spread_x_total` comes out at φ +0.009**, essentially zero, on real closing
  lines back to 1999. That is the end-to-end control working: two markets a
  book has priced for decades leave no joint residual dependence, which is
  what an efficient pair should look like and what a broken pipeline would not
  produce.

**Era stability of the primary family is excellent.** Sixteen seasons of
`teamtotal_x_passtd`: joint 0.324–0.369, Spearman 0.485–0.638, no trend and no
regime break. This is the family the whole thesis rests on and it does not
wander.

**The stratification earns its place.** `passyds_x_wr1recyds` by as-of target
share: t1 0.318 → t2 0.323 → t3 0.340. Exactly the covariate effect that
motivated stratifying in the first place — a single pooled number would read as
under-allowance on low-share pairs and over-allowance on high-share ones.
`rb1rush_x_qbpassunder` by spread: the negative dependence is strongest when the
anchor is a favourite (0.224) and nearly vanishes when it is a dog (0.239),
which is game script behaving as expected.

**Transport validation.** All six families pass; `bothqb_passyds` is the
weakest at 4 of 36 grid points missing (limit 20%), everything else at 0.

**A second bug, and the worst kind — silent.** The first build standardised
*inside* each stratum as well as ranking there. Standardisation needs `N_MIN`
= 6 games per subject-season, so a family with few rows per subject loses
nearly everything when its rows are split three ways: `bothqb_passyds` anchors
on the home side, giving ~8 rows per team-season, ~2.7 per stratum. Its three
strata came out holding **62, 3 and 121 of 2,250 observations**. Nothing
errored and nothing warned — a joint computed from three games would have been
printed like any other. Fixed by standardising on the whole family and ranking
within the stratum, which is also the more correct scoping: a subject's level
is a property of its season, not of a covariate bucket. Strata are now
781/813/656, and every family's terciles sum to the family. Pinned by
`strata_keep_their_rows_when_subjects_are_thin`.

**A methodological correction, made before reading any verdict.** The first
implementation compared a trained point estimate against a *test-only*
interval, ignoring the training half's sampling error. It failed
`teamtotal_x_passtd` (8/36) and `bothqb_passyds` (12/36) on discrepancies
averaging 0.004, with misses as small as 0.00006 — hairline, and an artifact of
the test rather than the data. Bootstrapping the **difference** across both
halves is the correct statistic; the pre-registered bar was not touched. After
the fix: 0/36 and 4/36.

## Phase 4 — the scorer (done)

`legs.rs` (15 tests) + `scoring.rs` (12 tests) + `examples/score_sgp.rs`.
Verified end to end on a synthetic sheet: parses, cross-checks legs against the
declared family (caught a deliberately mistyped row), excludes a started game,
resolves marginals, reads the atlas stratum, computes EV and the family
verdict. **The EVs in that smoke run are meaningless** — the SGP prices were
invented — it was a plumbing test.

One real bug it surfaced: `normalize_name` folds punctuation to a space, so
`Ja'Marr Chase` → `ja marr chase` but `JaMarr Chase` → `jamarr chase`. Two
books spelling a star differently would each have become a single-book
consensus row, failed the two-book bar, and made the leg silently unscoreable.
Fixed with `names::join_key`, used at every join site.

Another, caught by a test: JSON has no infinity, so the unbounded outer tercile
edges round-tripped to "no range at all" and `stratum_for` would have stopped
matching `t1`/`t3` after a reload. And `serde_json` does not round-trip an
`f64` exactly (one ULP), so the atlas stores float arrays as decimal strings.

## Open

- **Collect week 1.** The pipeline is complete and gated; the only missing
  input is `q`, the offered SGP price, which no feed sells and which has to be
  typed from the board.
- **Covariates at scoring time.** The sheet carries an optional `covariate`
  column and the scorer falls back to the `all` stratum without one. Computing
  target share and passing-TD share as-of for a live slate needs the current
  season's weekly file, which does not exist until games are played; the
  spread and total covariates can come straight off the leg board. Worth
  wiring before the strata matter.
- **Optional, worth one month:** The Odds API historical tier would let
  `P_true` be validated at real historical market marginals rather than at the
  reporting grid. It does not provide a single historical SGP price.
- No `git init` yet; commands are handed over, not run.

## Calendar

- 2026-09-09 Wed — season opens (NE @ SEA). 2026-09-13 — week 1 Sunday.
- Weekly: leg board + SGP sheet in one sitting, pregame only.
- Verdict after week 8 (~2026-11-01).

## Notes for next time

- Two build agents died mid-flight (one stalled, one hit a 529). Both had
  finished their code and their tests; only the report was lost. Check the tree
  before assuming an agent failure means missing work.
- `cargo clippy -D warnings` trips `excessive_precision` on published
  coefficient tables. Scoped `#[allow]` with a comment, same as the baseball
  repo's Lanczos constants — rounding them would break the audit trail.
- The 2026 weekly stats file does not exist until games are played; `games.csv`
  already carries 2026 lines, so week-1 covariates come from the 2025 file.
