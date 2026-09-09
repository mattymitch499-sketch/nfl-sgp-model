# NFL Model — same-game-parlay joint-probability study

Does a sportsbook's same-game-parlay (SGP) price charge for a *smaller* joint
probability than the one the outcomes actually produce? If it does, and by more
than the parlay's margin, the ticket is +EV. This project measures both halves
and fixes the pass/fail bar before any price is scored.

It is the direct successor to the baseball project (`pitcher-next-start`),
which built the method and produced a measured, honest **no** for MLB strikeout
parlays: one family had a real under-allowance, and it was about half the size
of the two-leg load. Everything that transferred is here. Everything that did
not survive contact with a liquid market was left behind on purpose.

## What this is not

There is no forecasting model, and there will not be one. The baseball project
spent most of its effort on Bayesian regressions and a batter-by-batter
simulator; a liquid market beat both at every rung, and the optimal weight of
the model into the market price came out at 0.014. **Market marginals are taken
as given.** Only the joint is questioned.

## The method in one paragraph

Every leg is a threshold on a continuous game stat. For each candidate family,
standardise each subject's season leave-one-out (the market already prices the
subject's *level*; the parlay pays for within-game *co-movement*), rank the
pooled residuals into uniform scores with tie-breaking jitter, and keep the
resulting dependence structure. Tonight's devigged marginals then say which
corner of that structure each leg wins in, and the true joint is the fraction of
history landing in both corners at once — no correlation coefficient, no
invariance assumption, marginals from the market rather than from history. See
[METHOD.md](METHOD.md); the implementation spec is [tasks/plan.md](tasks/plan.md).

## Status (2026-09-03)

**Operational.** Every phase is built, gated and run on real data. The only
missing input is the offered SGP price, which no feed sells and has to be typed
from the board each week.

| phase | what | state |
|---|---|---|
| 1 | `stats.rs` — copula numerics, standardisation, ranks, bootstrap | done, 32 tests |
| 2 | `games.rs`, `families.rs` — subject table and paired observations | done, 39 tests |
| 3 | `atlas.rs`, `examples/nfl_atlas.rs` — dependence, strata, transport validation | done, 12 tests, **run** |
| 4 | `legs.rs`, `scoring.rs`, `examples/score_sgp.rs` — parse, resolve, EV, verdict | done, 30 tests |
| 5 | docs and pre-registration | done |

162 tests (146 library + 16 example), clippy `-D warnings` clean, fmt clean.
The measured dependence per family is in `tasks/todo.md`; the headline is that
the part-whole family sits at a joint of **0.352 against 0.250 independence**,
and the `spread_x_total` control comes out at **0.252** — essentially zero — on
real closing lines back to 1999.

The 2026 season opens Wednesday 2026-09-09; week 1 Sunday is 2026-09-13 and its
lines are already posted. Outcome data for 2010–2025 is downloaded (128 MB).

## Layout

```text
NFL Model/
  README.md                 this file
  METHOD.md                 the statistic, the bar, the pre-registration, early stop, 14 lessons
  tasks/plan.md             the implementation specification, §1–§7
  tasks/todo.md             session log — newest section at the top
  crates/nfl-sgp/src/
    sgp.rs                  price inversion: implied, devig, joints, EV     (verbatim)
    correlation.rs          exact φ + cluster bootstrap                     (verbatim)
    stats.rs                Φ, Φ⁻¹, bivariate orthant, leave-one-out z,
                            uniform scores, empirical + Gaussian copula,
                            cluster bootstrap on a joint
    games.rs                per-game subject table, as-of Wr1/Rb1/QB, clusters
    families.rs             the six families and their paired observations
    atlas.rs                dependence objects, strata, transport validation
    legs.rs                 the SGP sheet's leg grammar, family cross-check
    depth.rs                official depth charts — who is actually starting
    scoring.rs              marginal resolution, EV, family verdict
    odds.rs                 The Odds API client, two-sided NFL quotes
    nflverse.rs             the free outcome data: readers + downloader
    names.rs                accent/punctuation-folded name key
  crates/nfl-sgp/examples/
    fetch_nflverse.rs       download seasons into data/raw/nflverse/
    family_counts.rs        real-data sanity counts per family
    nfl_atlas.rs            build and write the atlas
    leg_board.rs            weekly two-sided marginals across five books
    score_sgp.rs            score the collected sheet at the pre-registered bar
  data/raw/nflverse/        downloaded outcome files (gitignored, 128 MB)
  data/processed/
    sgp_prices.csv          the hand-collected SGP sheet
    leg_board.csv           appended by leg_board
  outputs/                  atlas, validation, consensus marginals, scored rows
  reference/baseball/       the originals ported from, plus worked examples
```

## The weekly loop

1. **Pull the leg board** for the games you will collect.
2. **Generate the sheet** with `slate_sheet`. It fills in everything except the
   parlay price: the game, the family, both leg descriptions naming the real
   player at the real line each book is posting, both sides of both legs'
   prices, and the covariate. It spends no credits.
3. **Type the SGP prices** — one number per ticket, read off each book's app,
   in the same sitting as the board pull so the joint and its marginals are
   contemporaneous. A combination a book refuses is a finding: leave
   `sgp_price` blank and say so in `notes`.
4. **Score**: `score_sgp` resolves each marginal (consensus of ≥ 2 books →
   sheet's own other side → unscoreable), reads the joint out of the atlas at
   the row's stratum, and computes EV.
5. **Verdict** at the bar in METHOD.md — after week 8, or earlier under the
   pre-registered early-stop rule.

## Commands

Download outcomes (already done; 2010–2025 plus the schedule):

```bash
cargo run --release -p nfl-sgp --example fetch_nflverse -- --seasons 2010-2025
```

Sanity-check the subject table and family counts:

```bash
cargo run --release -p nfl-sgp --example family_counts
```

Build the atlas:

```bash
cargo run --release -p nfl-sgp --example nfl_atlas
```

List the coming week's slate for free (no credits spent without `--games`):

```bash
cargo run --release -p nfl-sgp --example leg_board
```

Pull four player markets for two games (8 credits):

```bash
cargo run --release -p nfl-sgp --example leg_board -- --games Chiefs,Ravens
```

Generate this week's sheet (no credits; needs the board and the depth charts):

```bash
cargo run --release -p nfl-sgp --example slate_sheet -- --games Chiefs,Ravens --dry-run
```

Fetch the current season's official depth charts (re-run weekly — the file is a
rolling log of snapshots):

```bash
cargo run --release -p nfl-sgp --example fetch_nflverse -- --seasons 2026 --depth-charts
```

Score the sheet:

```bash
cargo run --release -p nfl-sgp --example score_sgp
```

`leg_board` needs `ODDS_API_KEY` in the environment (free key at
the-odds-api.com, no card). It refuses to spend without `--games`, stops at a
credit floor, and never caches a price.

## Data sources (verified reachable 2026-09-02)

| what | where | size |
|---|---|---|
| player-game stats, per season | nflverse-data release `stats_player` | ~8 MB each, 1999→ |
| team-game stats, per season | nflverse-data release `stats_team` | ~0.2 MB each |
| schedule with closing spread/total, moneylines, starting QBs | nflverse-data release `schedules`, `games.csv` | ~2 MB, all seasons |
| two-sided leg marginals, three target books | The Odds API, `americanfootball_nfl`, region `us` | 500 free credits/month |
| SGP prices | **hand-collected** — no feed carries them at hobbyist prices | — |

## Toolchain and gates

Rust stable (1.97). No Python, no R, and no crates beyond the set the baseball
project already used: `chrono`, `csv`, `serde`, `serde_json`, `thiserror`,
`reqwest`. Randomness is vendored xorshift64*, so no interval can change under a
dependency upgrade.

```bash
cargo test --workspace
```

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

```bash
cargo fmt --all --check
```

Example binaries' own tests need naming explicitly:

```bash
cargo test -p nfl-sgp --example family_counts --example leg_board --example fetch_nflverse --example nfl_atlas --example score_sgp
```

In a fresh PowerShell, cargo may need `$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"` first.

## Reference material

`reference/baseball/` holds the originals this was cut from, unmodified and not
compiled: the baseball atlas and exact-line scorer, the price-inversion and
atlas examples, the 32-row hand-collected price sheet, the scored output with
its devigged columns, the 85-row correlation atlas, and `findings.md` — the
session log's SGP sections, which carry every number quoted in METHOD.md.
