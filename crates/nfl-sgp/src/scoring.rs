//! Turning a collected sheet row into an expected value — `tasks/plan.md` §4,
//! §5 and §1.10.
//!
//! Everything here is the half of scoring that does not need the atlas: find
//! each leg's devigged marginal, price the offered joint, and aggregate rows
//! into the pre-registered decision statistic. The atlas supplies one number in
//! the middle — `P_true` — and `examples/score_sgp.rs` wires the two together.
//!
//! ## The marginal precedence, and why it lives in one function
//!
//! A leg's fair probability comes from, in order:
//!
//! 1. the **cross-book consensus** on the leg board, when at least two books
//!    priced it — the median devigged probability, which is a sharper anchor
//!    than any single book and carries its own error bar in the spread;
//! 2. the **sheet's own other side**, devigged — one book's proportional
//!    devig, which is what the baseball study had for every row it could score
//!    at all;
//! 3. nothing. The row is **unscoreable** and says so.
//!
//! [`resolve_marginal`] is the only place that order is written down. The
//! baseball study put the same precedence in one function for the same reason:
//! two code paths that each decide where a marginal comes from will eventually
//! disagree, and the disagreement will be invisible in the output.
//!
//! ## The one conversion that is easy to get backwards
//!
//! The leg board stores `consensus_devigged` as the probability of the **Over**
//! side (or of the named team's side, for a spread). A leg written as an Under
//! is therefore `1 − consensus_devigged`, and [`resolve_marginal`] does that
//! flip in exactly one place. Getting it wrong would not crash anything: it
//! would quietly price every Under family at its complement, and the EV would
//! look plausible. The flip is pinned by a test.

use crate::families::Family;
use crate::legs::{LegStat, ParsedLeg};
use crate::names::join_key;
use crate::sgp;
use crate::stats::{Direction, Marginal};
use crate::{Error, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// Bootstrap replicates for the family mean's interval (plan §1.10).
pub const FAMILY_BOOTSTRAP_REPS: usize = 2000;

/// A family mean below this many games is reported without a verdict: the
/// interval exists but the bar was written for a season, not for a week.
pub const MIN_GAMES_FOR_VERDICT: usize = 20;

/// One row of `outputs/leg_consensus.csv`, as `examples/leg_board.rs` writes it.
#[derive(Debug, Clone, Deserialize)]
pub struct ConsensusRow {
    pub date: String,
    pub market: String,
    pub subject: String,
    pub point: Option<f64>,
    pub n_books: usize,
    /// Median devigged probability **of the Over side** (or of `subject`'s side
    /// on a two-way market).
    pub consensus_devigged: f64,
    pub min_devigged: f64,
    pub max_devigged: f64,
    pub spread: f64,
    pub books: String,
}

/// The consensus board indexed the way a leg asks for it.
pub struct ConsensusIndex {
    by_leg: HashMap<(String, String, String, String), ConsensusRow>,
}

impl ConsensusIndex {
    pub fn read(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self {
                by_leg: HashMap::new(),
            });
        }
        let mut reader = csv::Reader::from_path(path)?;
        let rows: Vec<ConsensusRow> = reader
            .deserialize()
            .collect::<std::result::Result<_, _>>()
            .map_err(Error::Csv)?;
        Ok(Self::from_rows(rows))
    }

    pub fn from_rows(rows: Vec<ConsensusRow>) -> Self {
        let mut by_leg = HashMap::new();
        for row in rows {
            by_leg.insert(key(&row.date, &row.market, &row.subject, row.point), row);
        }
        Self { by_leg }
    }

    pub fn get(&self, slate: &str, leg: &ParsedLeg) -> Option<&ConsensusRow> {
        self.by_leg.get(&key(
            slate,
            leg.stat.market_key(),
            self.subject_of(leg),
            leg.point,
        ))
    }

    /// The subject a leg is looked up under, reconciling the two names a game
    /// total goes by.
    ///
    /// A game total has no subject in the sheet's grammar — `parse_leg` sets it
    /// to the empty string (`legs.rs`), because "game total over 47.5" names
    /// nobody and a leg that *did* carry a team there would be a different bet.
    /// The board writes the same market under [`crate::odds::GAME_SUBJECT`],
    /// i.e. `"game"`, because the wire gives its two sides no description and
    /// they have to pair under *something*.
    ///
    /// Both choices are right locally and they do not meet. Keyed naively, a
    /// game total looks up `""`, the board holds `"game"`, and the lookup can
    /// never hit — so every `spread_x_total` leg B would quietly fall through to
    /// the sheet's own two prices and be devigged at one book instead of three.
    /// Nothing would error; the leg would just stop getting the cross-book
    /// anchor the leg board exists to provide, which is METHOD.md lesson 1.
    fn subject_of<'leg>(&self, leg: &'leg ParsedLeg) -> &'leg str {
        if leg.stat == LegStat::GameTotal {
            crate::odds::GAME_SUBJECT
        } else {
            &leg.subject
        }
    }

    pub fn len(&self) -> usize {
        self.by_leg.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_leg.is_empty()
    }
}

/// The join key. The line is formatted to one decimal so `24.5` and `24.50`
/// are one leg, and the subject is folded so books' spellings agree.
fn key(
    slate: &str,
    market: &str,
    subject: &str,
    point: Option<f64>,
) -> (String, String, String, String) {
    (
        slate.to_string(),
        market.to_string(),
        join_key(subject),
        point.map_or_else(String::new, |value| format!("{value:.1}")),
    )
}

/// Where a resolved marginal came from — carried into the output so a scored
/// row can be audited without re-running anything.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MarginalSource {
    /// Cross-book consensus, with the number of books behind it.
    Consensus(usize),
    /// The sheet's own two prices for this leg, devigged.
    Sheet,
}

/// A resolved marginal, or the reason there isn't one.
#[derive(Debug, Clone, PartialEq)]
pub enum Resolution {
    Resolved {
        marginal: Marginal,
        source: MarginalSource,
        /// Cross-book disagreement, when the consensus supplied it — the
        /// honest error bar on the anchor.
        book_spread: Option<f64>,
    },
    Unscoreable(String),
}

impl Resolution {
    pub fn marginal(&self) -> Option<Marginal> {
        match self {
            Self::Resolved { marginal, .. } => Some(*marginal),
            Self::Unscoreable(_) => None,
        }
    }
}

/// Plan §4's precedence, in the only place it is written down.
///
/// `sheet_price` is the leg's own American price as typed, `sheet_other_price`
/// the other side. Both are needed for the fallback, because a one-sided quote
/// carries the book's margin and cannot be devigged — the baseball study's
/// hardest-won rule.
pub fn resolve_marginal(
    leg: &ParsedLeg,
    slate: &str,
    consensus: &ConsensusIndex,
    sheet_price: Option<f64>,
    sheet_other_price: Option<f64>,
) -> Resolution {
    if let Some(row) = consensus.get(slate, leg)
        && row.n_books >= 2
    {
        // The board stores the Over side; an Under leg is its complement.
        let p = match leg.direction {
            Direction::Over => row.consensus_devigged,
            Direction::Under => 1.0 - row.consensus_devigged,
        };
        if (0.0..=1.0).contains(&p) {
            return Resolution::Resolved {
                marginal: Marginal {
                    p,
                    direction: leg.direction,
                },
                source: MarginalSource::Consensus(row.n_books),
                book_spread: Some(row.spread),
            };
        }
    }
    match (sheet_price, sheet_other_price) {
        (Some(price), Some(other)) => match sgp::devig(price, other) {
            // `devig` returns the first argument's side, which is the leg as
            // typed — so no complement flip here, unlike the board.
            Some((p, _)) => Resolution::Resolved {
                marginal: Marginal {
                    p,
                    direction: leg.direction,
                },
                source: MarginalSource::Sheet,
                book_spread: None,
            },
            None => Resolution::Unscoreable(format!(
                "sheet prices {price} / {other} are not a valid two-sided market"
            )),
        },
        _ => Resolution::Unscoreable(
            "no consensus at 2+ books and the sheet has only one side of this leg".into(),
        ),
    }
}

/// One scored row: everything the output file carries about a single collected
/// ticket at a single book.
#[derive(Debug, Clone)]
pub struct ScoredRow {
    pub slate: String,
    pub game: String,
    pub book: String,
    pub family: Family,
    pub p_a: f64,
    pub p_b: f64,
    pub source_a: MarginalSource,
    pub source_b: MarginalSource,
    pub stratum: String,
    /// The atlas's joint at these marginals.
    pub p_true: f64,
    pub p_true_interval: Option<(f64, f64)>,
    /// What the book charges for the joint, `1 / decimal`.
    pub q: f64,
    /// The decision statistic.
    pub ev: f64,
    /// What the ticket would be worth if the legs were independent — reported,
    /// never decided on.
    pub ev_independent: f64,
    /// The book's own correlation allowance in joint terms, `q / (p_a·p_b)`.
    pub book_allowance: f64,
}

/// Prices one row against a joint the atlas supplied.
///
/// Separate from the atlas lookup so the arithmetic can be tested against hand
/// calculations without building a dependence object.
#[allow(clippy::too_many_arguments)]
pub fn score_row(
    slate: &str,
    game: &str,
    book: &str,
    family: Family,
    a: Marginal,
    b: Marginal,
    source_a: MarginalSource,
    source_b: MarginalSource,
    stratum: &str,
    p_true: f64,
    p_true_interval: Option<(f64, f64)>,
    sgp_price: f64,
) -> Option<ScoredRow> {
    let decimal = sgp::decimal_from_american(sgp_price)?;
    let q = 1.0 / decimal;
    let independent = a.p * b.p;
    Some(ScoredRow {
        slate: slate.to_string(),
        game: game.to_string(),
        book: book.to_string(),
        family,
        p_a: a.p,
        p_b: b.p,
        source_a,
        source_b,
        stratum: stratum.to_string(),
        p_true,
        p_true_interval,
        q,
        ev: p_true * decimal - 1.0,
        ev_independent: independent * decimal - 1.0,
        book_allowance: if independent > 0.0 {
            q / independent
        } else {
            f64::NAN
        },
    })
}

/// The best book's EV for one (family, game) — plan §1.10's per-row statistic,
/// because the executable strategy is shopping.
#[derive(Debug, Clone)]
pub struct BestByGame {
    pub game: String,
    pub book: String,
    pub ev: f64,
}

pub fn best_by_game(rows: &[ScoredRow], family: Family) -> Vec<BestByGame> {
    let mut best: HashMap<&str, (&str, f64)> = HashMap::new();
    for row in rows.iter().filter(|row| row.family == family) {
        let entry = best
            .entry(row.game.as_str())
            .or_insert((row.book.as_str(), f64::NEG_INFINITY));
        if row.ev > entry.1 {
            *entry = (row.book.as_str(), row.ev);
        }
    }
    let mut out: Vec<BestByGame> = best
        .into_iter()
        .map(|(game, (book, ev))| BestByGame {
            game: game.to_string(),
            book: book.to_string(),
            ev,
        })
        .collect();
    // Sorted so the output file and the bootstrap draws are reproducible.
    out.sort_by(|left, right| left.game.cmp(&right.game));
    out
}

/// The pre-registered verdict for one family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// The interval sits entirely above zero: the thread continues.
    Continues,
    /// Below zero or straddling it: the family closes.
    Closes,
    /// Too few games to apply the bar written for a season.
    TooEarly,
    /// The family failed the transport validation and was never scored.
    NotTransportable,
}

impl Verdict {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Continues => "continues",
            Self::Closes => "closes",
            Self::TooEarly => "too early",
            Self::NotTransportable => "not transportable",
        }
    }
}

/// One family's decision statistic.
#[derive(Debug, Clone)]
pub struct FamilySummary {
    pub family: Family,
    pub games: usize,
    pub mean_ev: f64,
    pub interval: Option<(f64, f64)>,
    pub verdict: Verdict,
}

/// Mean best-book EV across games, with a game-cluster bootstrap interval
/// (plan §1.10).
///
/// The cluster is the game and each game contributes exactly one value after
/// the best-book step, so this is a bootstrap over games — which is the point:
/// two families priced in the same game are not independent observations, and
/// the per-family interval must not pretend otherwise by resampling rows.
pub fn summarise(
    family: Family,
    best: &[BestByGame],
    reps: usize,
    seed: u64,
    transportable: bool,
) -> FamilySummary {
    if !transportable {
        return FamilySummary {
            family,
            games: best.len(),
            mean_ev: f64::NAN,
            interval: None,
            verdict: Verdict::NotTransportable,
        };
    }
    let values: Vec<f64> = best.iter().map(|row| row.ev).collect();
    let mean = mean(&values).unwrap_or(f64::NAN);
    let interval = bootstrap_mean(&values, reps, seed);
    let verdict = match interval {
        _ if values.len() < MIN_GAMES_FOR_VERDICT => Verdict::TooEarly,
        Some((low, _)) if low > 0.0 => Verdict::Continues,
        Some(_) => Verdict::Closes,
        None => Verdict::TooEarly,
    };
    FamilySummary {
        family,
        games: values.len(),
        mean_ev: mean,
        interval,
        verdict,
    }
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<f64>() / values.len() as f64)
}

/// Percentile bootstrap of the mean, resampling whole games.
///
/// Uses the crate's vendored xorshift so an interval never moves under a
/// dependency upgrade, and the same linear-interpolation percentile as every
/// other interval in the project.
pub fn bootstrap_mean(values: &[f64], reps: usize, seed: u64) -> Option<(f64, f64)> {
    if values.len() < 2 || reps == 0 {
        return None;
    }
    let mut rng = crate::correlation::Rng::new(seed);
    let mut replicates = Vec::with_capacity(reps);
    for _ in 0..reps {
        let draw: f64 = (0..values.len())
            .map(|_| values[rng.below(values.len())])
            .sum::<f64>()
            / values.len() as f64;
        replicates.push(draw);
    }
    replicates.sort_by(f64::total_cmp);
    Some((
        percentile(&replicates, 0.025),
        percentile(&replicates, 0.975),
    ))
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let rank = (p * (sorted.len() - 1) as f64).clamp(0.0, (sorted.len() - 1) as f64);
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] + (rank - lower as f64) * (sorted[upper] - sorted[lower])
    }
}

/// A leg's market key and line, for the "what is missing from the board"
/// report the scorer prints when rows cannot be scored.
pub fn describe_leg(leg: &ParsedLeg) -> String {
    match (leg.stat, leg.point) {
        (LegStat::GameTotal, Some(point)) => format!("game total {point:.1}"),
        (stat, Some(point)) => format!("{} {} {:.1}", leg.subject, stat.market_key(), point),
        (stat, None) => format!("{} {}", leg.subject, stat.market_key()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::legs::parse_leg;

    fn consensus_row(
        market: &str,
        subject: &str,
        point: f64,
        p: f64,
        books: usize,
    ) -> ConsensusRow {
        ConsensusRow {
            date: "2026-09-13".into(),
            market: market.into(),
            subject: subject.into(),
            point: Some(point),
            n_books: books,
            consensus_devigged: p,
            min_devigged: p - 0.01,
            max_devigged: p + 0.01,
            spread: 0.02,
            books: "draftkings|fanduel".into(),
        }
    }

    #[test]
    fn consensus_wins_when_two_books_priced_the_leg() {
        let index = ConsensusIndex::from_rows(vec![consensus_row(
            "player_pass_yds",
            "Patrick Mahomes",
            274.5,
            0.52,
            3,
        )]);
        let leg = parse_leg("Patrick Mahomes over 274.5 pass yds").unwrap();
        let resolved = resolve_marginal(&leg, "2026-09-13", &index, Some(-115.0), Some(-105.0));
        match resolved {
            Resolution::Resolved {
                marginal,
                source,
                book_spread,
            } => {
                assert!((marginal.p - 0.52).abs() < 1e-12);
                assert_eq!(source, MarginalSource::Consensus(3));
                assert_eq!(book_spread, Some(0.02));
            }
            other => panic!("{other:?}"),
        }
    }

    /// **The flip that would be invisible if it were wrong.** The board stores
    /// the Over side; an Under leg must come back as its complement.
    #[test]
    fn an_under_leg_takes_the_complement_of_the_boards_over() {
        let index = ConsensusIndex::from_rows(vec![consensus_row(
            "player_pass_yds",
            "Josh Allen",
            249.5,
            0.55,
            4,
        )]);
        let leg = parse_leg("Josh Allen under 249.5 pass yds").unwrap();
        let p = resolve_marginal(&leg, "2026-09-13", &index, None, None)
            .marginal()
            .unwrap();
        assert!((p.p - 0.45).abs() < 1e-12, "got {}", p.p);
        assert_eq!(p.direction, Direction::Under);
    }

    /// A single book on the board is not a consensus; the sheet's own two
    /// sides take over, and they are devigged as typed with no flip.
    #[test]
    fn one_book_falls_through_to_the_sheets_own_two_sides() {
        let index = ConsensusIndex::from_rows(vec![consensus_row(
            "player_pass_yds",
            "Patrick Mahomes",
            274.5,
            0.90,
            1,
        )]);
        let leg = parse_leg("Patrick Mahomes under 274.5 pass yds").unwrap();
        let resolved = resolve_marginal(&leg, "2026-09-13", &index, Some(-110.0), Some(-110.0));
        match resolved {
            Resolution::Resolved {
                marginal, source, ..
            } => {
                assert!((marginal.p - 0.5).abs() < 1e-12, "got {}", marginal.p);
                assert_eq!(source, MarginalSource::Sheet);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_one_sided_leg_with_no_consensus_is_unscoreable() {
        let index = ConsensusIndex::from_rows(vec![]);
        let leg = parse_leg("Travis Kelce over 4.5 receptions").unwrap();
        let resolved = resolve_marginal(&leg, "2026-09-13", &index, Some(-120.0), None);
        assert!(matches!(resolved, Resolution::Unscoreable(_)));
        assert!(resolved.marginal().is_none());
    }

    /// The slate is part of the key: last week's board must not answer this
    /// week's question.
    #[test]
    fn the_board_is_keyed_by_slate() {
        let index = ConsensusIndex::from_rows(vec![consensus_row(
            "player_pass_yds",
            "Patrick Mahomes",
            274.5,
            0.52,
            3,
        )]);
        let leg = parse_leg("Patrick Mahomes over 274.5 pass yds").unwrap();
        assert!(matches!(
            resolve_marginal(&leg, "2026-09-20", &index, None, None),
            Resolution::Unscoreable(_)
        ));
    }

    /// And so is the line: 274.5 and 279.5 are different questions.
    #[test]
    fn the_board_is_keyed_by_line() {
        let index = ConsensusIndex::from_rows(vec![consensus_row(
            "player_pass_yds",
            "Patrick Mahomes",
            274.5,
            0.52,
            3,
        )]);
        let leg = parse_leg("Patrick Mahomes over 279.5 pass yds").unwrap();
        assert!(matches!(
            resolve_marginal(&leg, "2026-09-13", &index, None, None),
            Resolution::Unscoreable(_)
        ));
    }

    /// **A game total must find the board's `"game"` subject.**
    ///
    /// The sheet calls a game total subject-less and the board calls it
    /// `"game"`. Before these were reconciled the lookup could never match, and
    /// the failure was invisible: the leg fell through to the sheet's own two
    /// prices and scored at one book's devig instead of the three-book median,
    /// on every `spread_x_total` row, for as long as it went unnoticed.
    #[test]
    fn a_game_total_finds_the_boards_game_subject() {
        let index = ConsensusIndex::from_rows(vec![ConsensusRow {
            date: "2026-09-13".into(),
            market: "totals".into(),
            subject: crate::odds::GAME_SUBJECT.into(),
            point: Some(47.5),
            n_books: 3,
            consensus_devigged: 0.51,
            min_devigged: 0.50,
            max_devigged: 0.52,
            spread: 0.02,
            books: "draftkings|fanduel|fanatics".into(),
        }]);
        let leg = parse_leg("game total over 47.5").unwrap();
        assert_eq!(
            leg.subject_key, "",
            "the sheet's grammar has no subject here"
        );
        let resolved = resolve_marginal(&leg, "2026-09-13", &index, None, None);
        match resolved {
            Resolution::Resolved {
                marginal, source, ..
            } => {
                assert!((marginal.p - 0.51).abs() < 1e-12);
                assert_eq!(
                    source,
                    MarginalSource::Consensus(3),
                    "must come from the board, not fall through to the sheet"
                );
            }
            other => panic!("{other:?}"),
        }
    }

    /// Books spell the same player differently; the folded key joins them.
    #[test]
    fn subjects_join_across_spellings() {
        let index = ConsensusIndex::from_rows(vec![consensus_row(
            "player_reception_yds",
            "Ja'Marr Chase",
            78.5,
            0.48,
            5,
        )]);
        let leg = parse_leg("JaMarr Chase over 78.5 rec yds").unwrap();
        assert!(
            resolve_marginal(&leg, "2026-09-13", &index, None, None)
                .marginal()
                .is_some()
        );
    }

    /// Hand-computed EV. Legs at 0.55 and 0.50 with a true joint of 0.32
    /// against +260 (decimal 3.6): 0.32 × 3.6 − 1 = +0.152. Independence would
    /// be 0.275 × 3.6 − 1 = −0.01, and the book's allowance is
    /// (1/3.6)/0.275 = 1.0101.
    #[test]
    fn score_row_matches_a_hand_calculation() {
        let row = score_row(
            "2026-09-13",
            "Bills @ Chiefs",
            "draftkings",
            Family::TeamtotalXPasstd,
            Marginal::over(0.55),
            Marginal::over(0.50),
            MarginalSource::Consensus(4),
            MarginalSource::Consensus(3),
            "t2",
            0.32,
            Some((0.30, 0.34)),
            260.0,
        )
        .unwrap();
        assert!((row.q - 1.0 / 3.6).abs() < 1e-12);
        assert!((row.ev - 0.152).abs() < 1e-12, "got {}", row.ev);
        assert!((row.ev_independent - (-0.01)).abs() < 1e-12);
        assert!((row.book_allowance - (1.0 / 3.6) / 0.275).abs() < 1e-12);
        assert!(
            score_row(
                "d",
                "g",
                "b",
                Family::TeamtotalXPasstd,
                Marginal::over(0.5),
                Marginal::over(0.5),
                MarginalSource::Sheet,
                MarginalSource::Sheet,
                "all",
                0.3,
                None,
                50.0, // not a valid American price
            )
            .is_none()
        );
    }

    fn scored(game: &str, book: &str, ev: f64) -> ScoredRow {
        ScoredRow {
            slate: "2026-09-13".into(),
            game: game.into(),
            book: book.into(),
            family: Family::TeamtotalXPasstd,
            p_a: 0.5,
            p_b: 0.5,
            source_a: MarginalSource::Sheet,
            source_b: MarginalSource::Sheet,
            stratum: "all".into(),
            p_true: 0.3,
            p_true_interval: None,
            q: 0.3,
            ev,
            ev_independent: 0.0,
            book_allowance: 1.0,
        }
    }

    /// Shopping is the strategy, so the per-game statistic is the best book —
    /// and only rows of the family asked for.
    #[test]
    fn best_by_game_takes_the_maximum_over_books() {
        let rows = vec![
            scored("A", "draftkings", -0.06),
            scored("A", "fanduel", -0.02),
            scored("A", "betmgm", -0.09),
            scored("B", "draftkings", 0.01),
            ScoredRow {
                family: Family::SpreadXTotal,
                ..scored("A", "caesars", 0.99)
            },
        ];
        let best = best_by_game(&rows, Family::TeamtotalXPasstd);
        assert_eq!(best.len(), 2);
        assert_eq!(best[0].game, "A");
        assert_eq!(best[0].book, "fanduel");
        assert!((best[0].ev - (-0.02)).abs() < 1e-12);
        assert!((best[1].ev - 0.01).abs() < 1e-12);
    }

    /// The bar: an interval entirely above zero continues, anything else
    /// closes, and too few games is neither.
    #[test]
    fn the_verdict_follows_the_pre_registered_bar() {
        let clearly_positive: Vec<BestByGame> = (0..40)
            .map(|i| BestByGame {
                game: format!("g{i:02}"),
                book: "dk".into(),
                ev: 0.10 + (i % 3) as f64 * 0.001,
            })
            .collect();
        let summary = summarise(Family::TeamtotalXPasstd, &clearly_positive, 500, 7, true);
        assert_eq!(summary.verdict, Verdict::Continues);
        assert!(summary.interval.unwrap().0 > 0.0);

        let straddling: Vec<BestByGame> = (0..40)
            .map(|i| BestByGame {
                game: format!("g{i:02}"),
                book: "dk".into(),
                ev: if i % 2 == 0 { 0.30 } else { -0.30 },
            })
            .collect();
        assert_eq!(
            summarise(Family::TeamtotalXPasstd, &straddling, 500, 7, true).verdict,
            Verdict::Closes
        );

        let thin = &clearly_positive[..5];
        assert_eq!(
            summarise(Family::TeamtotalXPasstd, thin, 500, 7, true).verdict,
            Verdict::TooEarly
        );
    }

    /// A family that failed the transport validation is never given a number.
    #[test]
    fn an_untransportable_family_is_refused_rather_than_scored() {
        let best: Vec<BestByGame> = (0..40)
            .map(|i| BestByGame {
                game: format!("g{i:02}"),
                book: "dk".into(),
                ev: 0.10,
            })
            .collect();
        let summary = summarise(Family::BothqbPassyds, &best, 500, 7, false);
        assert_eq!(summary.verdict, Verdict::NotTransportable);
        assert!(summary.interval.is_none());
        assert!(summary.mean_ev.is_nan());
    }

    #[test]
    fn the_bootstrap_is_deterministic_and_brackets_the_mean() {
        let values: Vec<f64> = (0..50).map(|i| (i as f64 - 25.0) / 100.0).collect();
        let first = bootstrap_mean(&values, 400, 11).unwrap();
        assert_eq!(Some(first), bootstrap_mean(&values, 400, 11));
        assert_ne!(Some(first), bootstrap_mean(&values, 400, 12));
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        assert!(first.0 < mean && mean < first.1);
        assert!(bootstrap_mean(&[0.1], 400, 1).is_none());
        assert!(bootstrap_mean(&values, 0, 1).is_none());
    }
}
