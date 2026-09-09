# Method

The statistic, the bar, and the lessons. Read this before choosing a family or
scoring a row. `tasks/plan.md` is the implementation specification; this is the
reasoning and the pre-registration.

## The question, in one line

For a two-leg SGP with devigged consensus marginals `p_A`, `p_B`, an offered
price whose decimal odds are `d`, and a true joint probability `P_true`:

```text
q      = 1 / d                      the joint the book charges for
EV     = P_true / q − 1             per unit staked
```

Everything else is machinery for estimating `P_true` honestly.

## Why this is stated in joint probabilities, not correlations

The baseball study measured φ — the Pearson correlation of two binary legs —
and transported it: measure φ on history, then rebuild tonight's joint from
tonight's marginals plus that φ. That last step is a modelling assumption
wearing a measurement's clothes. φ for two binary outcomes is **bounded by its
own marginals** (its Fréchet range narrows as the legs move away from 50%), so
it does not transport cleanly across thresholds, and a φ gap of a given size is
worth different money depending on where the lines sit.

The object that does transport is the **dependence structure of the underlying
continuous stats**. Passing yards, receiving yards and team points are
continuous; every leg is a threshold on one of them. So:

1. **Standardise within subject.** For each player-season or team-season,
   replace each game's value with a leave-one-out z-score. The market's
   marginal already prices the subject's level; the parlay pays only for
   within-game co-movement *given* the subjects. Pooling raw values would score
   "a good QB and a good WR are both usually high" as co-movement — it is not,
   it is level, and the baseball atlas partly made this mistake.
2. **Rank to uniforms.** Pool the z-scores and convert to ranks, ties broken by
   seeded jitter (Rüschendorf's distributional transform). This strips the
   marginals entirely and leaves only the copula — the dependence.
3. **Read the joint at tonight's marginals.** Tonight's devigged `p_A` says leg
   A wins in the top `p_A` fraction of its own distribution. `P_true` is the
   fraction of historical pairs where both legs land in their winning corners.
   No φ, no invariance assumption, and the marginals come from the market
   rather than from history.

φ survives in the outputs as a one-number summary beside the joint, so a family
here can be compared with `reference/baseball/`. It decides nothing.

## The statistic (fixed)

Per family, per game: the **best book's EV** at devigged consensus marginals.
Shopping is the executable strategy, so the per-row statistic is the max over
books. Then the **mean across games**, with a **game-cluster bootstrap 95%
interval**.

Reported beside it, deciding nothing: `EV_indep = p_A·p_B/q − 1` (what the
ticket would be worth if the legs were independent) and `q / (p_A·p_B)` (the
book's own correlation allowance, in joint terms).

## The bar (fixed)

- **Primary family:** the interval sits entirely above zero → the thread
  continues to a sizing study. Below zero or straddling → the family closes.
- **Secondary families:** the same rule, each on its own rows.
- **No per-cell wins.** Five books × eighteen weeks × sixteen games × six
  families guarantees a positive cell somewhere under perfectly fair pricing. A
  row, a book, or a week is not evidence. The family mean is what a bet earns.
- **A family that fails the transport validation is not scored at all.** See
  below; the scorer refuses it rather than reporting a number.
- **Written before the data.** The baseball log has two findings that reversed
  sign completely out of sample — a quarter-Kelly rule that swung from −45% to
  +49% on a multiplier, and a "12+ point edges win" rule that went from 47% to
  25% over the next 229 legs. Both were bars written after looking.

## The transport validation, and why it gates everything

The whole design rests on one assumption: that the dependence structure
measured on history is the same structure operating tonight. That is testable,
and it is tested before any row is scored.

Fit the copula on seasons ≤ 2017; evaluate on ≥ 2018, standardised and ranked
independently. At each of 36 grid points (marginals 0.3/0.5/0.7 × four
direction combinations), bootstrap the **difference** between the two halves'
joints, resampling clusters on both sides, and ask whether zero is inside the
interval. A family missing at more than 20% of the grid points has failed, and
`Atlas::failed_families` reports it.

The interval is on the difference, not on the test half alone, and that
distinction decided a family. Comparing a trained point estimate against a
test-only interval ignores the training half's own sampling error; with the two
halves nearly the same size that error is just as large, so the test rejects far
above its nominal rate. Measured here, the naive version failed the primary
family on discrepancies averaging 0.004 — misses as small as 0.00006 — while
the correct version passes it at 0 of 36. The bar (20% of grid points) was not
touched; what was wrong was the interval that defines a miss. `spread_x_total` carries a second,
end-to-end check: its values are market residuals against real closing lines
back to 1999, so it tests the whole chain with no standardisation step at all.

## Pre-registration

Fixed 2026-09-03, before any price was collected and before the atlas was read.

| field | value |
|---|---|
| **primary family** | `teamtotal_x_passtd` — team points Over × QB passing TDs Over |
| **secondary** | `rb1rush_x_qbpassunder`, `bothqb_passyds` |
| **control** | `passyds_x_rb1rush` (DFS literature reports QB–RB near zero) |
| **calibration** | `spread_x_total`, `passyds_x_wr1recyds` (the pairs books price best) |
| stratification | pass-TD share; anchor spread buckets; closing total; target share — one per family, all as-of, listed in `tasks/plan.md` §2 |
| seasons | 2010–2025 regular season; `spread_x_total` 1999–2025 |
| cluster unit | the subject-season that owns the correlation, per family; cross-team families take the wider of two one-way bootstraps |
| books | **DraftKings, FanDuel, Fanatics** — fixed 2026-09-03 |
| weeks | 1–8 of the 2026 season, subject to the early stop below |
| rows per week | one per family per selected game, both sides of every leg |
| **the number that closes it** | the primary family's game-clustered EV interval failing to sit entirely above zero |

### Amendment, 2026-09-08 — three collection-protocol rules

Fixed **before any price was collected** (the sheet holds zero rows as of this
date). These are being written down now because building the sheet generator
forces each of them to be decided, and a decision that moves the measured
statistic belongs in the pre-registration rather than buried in a code comment.

1. **Line selection: mid-probability.** Where a book offers several lines on a
   leg, take the one whose devigged probability is **closest to 0.50**; ties go
   to the lower line, for determinism. Reason: the two-leg load grows toward the
   tails, so line choice is the one lever that shrinks the toll before shopping
   even begins; and the atlas estimates its joint with the most data near the
   middle. The rule reads **only the marginals**, so it cannot be influenced by
   the `q` under test.
   *Provenance, stated because I got it wrong once:* this rule is carried over
   from the **baseball** study's pre-registration
   (`reference/baseball/findings.md:22`). It was never in this project's
   METHOD.md, and an earlier draft of the plan wrongly cited it as though it
   were. It is a new clause here, adopted today.

2. **Anchor side: the home team, always.** Same-team families would otherwise
   admit one row per (game, team), and `scoring::best_by_game` keys on the game,
   so two anchors from one game would be maximised over together — inflating the
   statistic for the same reason adding books would. "One row per family per
   selected game" in the table above already implies this; it is now explicit.

3. **The book names the quarterback; history only checks him.** Plan §2.1
   identifies the starter from `games.csv`'s `home_qb_id`/`away_qb_id`, confirmed
   against realised attempts. **Neither exists before kickoff** — verified: all
   sixteen 2026 week-1 rows carry a spread and a total but empty QB ids. So
   pre-game the subject is taken from the book: the passer it posts a
   passing-yards market on, attributed to a side via the prior-season roster. A
   passer who maps to neither side is **refused and named**, never assigned by
   elimination. Reason: a book posting a market on exactly one quarterback is
   the same kind of external declaration `home_qb_id` is, from a source with
   strictly more current information than a nine-month-old stat file. The
   prior-season attempts leader is computed anyway and any disagreement is
   printed, but the board wins.

**Why the book set is pre-registered.** The feed returns every book in the
region for the same credit, so this is not a cost decision — it is a
measurement one. The statistic is a maximum over books, and a maximum over
more books is larger even under perfectly fair pricing, purely from taking the
best of more draws. Adding a fourth book later would raise the measured edge
without anything changing in the world. The set is therefore fixed here, and
`odds::TARGET_BOOKS` drops every other book at collection so the board file
cannot quietly acquire one.

## The early stop, fixed before any data

Eight weeks was chosen when this looked like a problem of averaging out noise.
It is not: **`EV = P_true/q − 1` contains no game outcome.** Both terms are
computed from prices and the atlas, so a row's EV is known the moment it is
collected and does not get less noisy as games are played. What more weeks buy
is more games and better coverage of the EV distribution — not the resolution
of any uncertainty about who won.

So the collection stops early, in either direction, when the answer is already
in. Written now, before week 1:

- **Stop and close a family** once it has **≥ 12 games** and the mean best-book
  EV interval sits **entirely below −3%**. At that point the family is not
  marginal, and more weeks of the same collection cannot move a bound that far.
- **Stop and continue to a sizing study** at the pre-registered bar, unchanged:
  the interval entirely above zero, with **≥ 20 games** (`MIN_GAMES_FOR_VERDICT`).
- **Otherwise keep collecting to week 8**, then apply the bar as written.

The asymmetry is deliberate. A negative result needs only to be decisive, and
−3% with a bounded interval is decisive against a two-leg load; a positive one
is a decision to risk money and keeps the full sample requirement.

This is a stopping rule, not a licence to look and then decide — it is fixed,
it is symmetric in the sense that both outcomes are specified in advance, and
the alternative (reading the numbers each week and choosing when to stop) is
precisely how the quarter-Kelly and 12-point findings were manufactured.

## What can and cannot be backtested

Worth stating plainly, because it shapes everything above.

**Backtestable now, free, on data already on disk:** every part of `P_true` —
the dependence measurement, the standardisation, the strata, the transport
check, the era stability. Sixteen seasons of outcomes and, for
`spread_x_total`, real closing lines back to 1999. This is the half that
actually needed history, and it is done.

**Not backtestable at hobbyist prices:** `q`, the offered SGP price. No feed
sells historical same-game-parlay prices; OpticOdds is the only source and is
sales-led at a reported ~$5k/month — the same wall the baseball study hit.
Historical *leg* prices are purchasable from The Odds API's paid historical
tier (player props from roughly 2023), which would let `P_true` be validated at
real historical market marginals rather than at the reporting grid. That is a
genuine upgrade and a reasonable thing to buy one month of; it still does not
produce a single historical `q`.

Consequence: the book's side has to be collected forward, and only the book's
side. That is what the early stop is for.

**Prior on record:** a clean, well-measured negative on all six. The reasoning
is arithmetic, not pessimism — a +EV two-leg ticket needs the book's
under-allowance, in joint-probability terms, to exceed the two-leg load, which
measured ~13% on DraftKings in baseball and is generally higher on NFL parlays.
The part-whole family is the one with a mechanism behind it: it was the only
baseball family with a real under-allowance, and a book engine that assigns a
table value per pair type cannot reach the extreme dependence of a leg that is
literally a component of the other. The negative-correlation and cross-team
families are the other two places a table-driven engine is most plausibly
wrong.

## The lessons, each with the number that taught it

1. **Devig before comparing anything.** A raw −115/−115 pair reads 53.5% a side
   and sums to 1.070. Comparing raw implieds to a fair number manufactures
   ~3.5pp of edge on every leg, always in the same direction. The baseball
   control family read a +0.19 correlation allowance on the raw basis and ≈0
   once devigged: the headline finding was 2.6x inflated by this alone.

2. **Both sides of every leg, or the row cannot be scored.** Hand collection
   produced both sides on 8 of 32 baseball rows. The leg board exists so that
   never happens again. Anytime-TD markets are usually listed one-sided — which
   is why no v1 family uses one.

3. **EV, not the φ-gap.** The gap double-counts leg vig and never reads the
   same as money. On one baseball row a φ-gap that looked like +0.18 was an EV
   of −7.5%: compounded leg vig (~9%) plus SGP margin (~6%) against ~2pp of
   joint under-allowance.

4. **Hold is unidentifiable from a single price.** Margin and correlation
   allowance are confounded. The statistic above never reads hold, which is why
   no hold-calibration family is needed this time.

5. **Exclude in-play quotes at collection *and* at analysis.** Eight in-play
   rows once dragged a measured error from 9.28pp to 15.64pp. An append-only
   file outlives any one filter, so filter twice. An unreadable kickoff time
   counts as started.

6. **The slate is not the UTC date.** Sunday Night Football kicks off 00:20Z
   Monday and belongs to Sunday. A game is on slate `D` when kickoff ∈
   `[D 10:00Z, D+1 10:00Z)`.

7. **Cluster by the unit that owns the correlation.** A pitcher-clustered
   interval on an opponent property came out 3.4x too narrow and flipped a
   verdict from "clears" to "not established."

8. **Condition on what the book conditions on.** A population-average
   dependence is not the dependence operating in tonight's game. The QB–WR1
   correlation depends on that receiver's target share; one pooled number would
   read as fake under-allowance on low-share pairs and fake over-allowance on
   high-share ones, and EV would inherit the bias. Every family is stratified
   by the covariate that drives it.

9. **Standardise within subject, leave-one-out.** Between-subject variation is
   level, not co-movement, and the market has already priced it. Leaving the
   game itself out of its own centring keeps the standardisation an
   out-of-sample statement — the same statement a book's simulator makes.

10. **Break ties, or a count stat's copula is not identified.** Passing
    touchdowns take six values. Average ranks would pile a quarter of the
    sample on one score and every threshold question would be answered by the
    arbitrary handling of one atom. The jitter seed is recorded and the
    sensitivity to it is reported.

11. **Era splits are a regime check, not decoration.** NFL rule changes and
    pass-rate drift make this matter more than it did in baseball. A sign flip
    across seasons is a red flag, and per-season joints are published.

12. **Numbers from different pulls are not comparable.** The baseball market
    benchmark scored 6,112 legs one day and 8,284 on the same starts ten days
    later, because the exchange's retention window moved. Only within-run
    ordering is meaningful.

13. **Audit before building.** Reading coefficients out of an artifact
    cancelled a planned feature in minutes. Measure a family's dependence
    before spending a week collecting its prices.

14. **Hand collection is the bottleneck.** The baseball five-book pass was
    pre-registered and never run, because it meant typing a fresh slate every
    night. NFL is one slate a week. Pull the leg board and fill the SGP sheet in
    one sitting so the joint and its marginals are contemporaneous.

## Baseball results, for reference

Full write-ups in `reference/baseball/findings.md`.

| family (starter K over × …) | φ_true | φ_book gap, devigged | EV (DraftKings) |
|---|---|---|---|
| team strikeouts thrown (part-whole) | +0.50 to +0.55 | **+0.087** | **−6.2%** |
| starter earned runs under | +0.20 | −0.036 | −14.2% |
| starter outs over (deep start) | +0.26 | −0.067 | −15.9% |
| first-five moneyline | +0.16 | −0.073 | −15.9% |
| batter 1+ K | +0.18 | one-sided, unscoreable | — |
| opposing starter K over (control) | −0.013 | ≈ 0 | — |

The one family with a real under-allowance was the part-whole pair, where one
leg is largely made of the other. It was still about half the size of the toll.
