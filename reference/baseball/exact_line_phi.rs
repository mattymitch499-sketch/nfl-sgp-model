//! Exact-line φ_true for every collected same-game-parlay row.
//!
//! The correlation atlas (`examples/correlation_atlas.rs`) measures φ on a
//! fixed rung grid; this study measures it at the *exact* thresholds the
//! collected prices hang. For each complete, pregame row of
//! `data/processed/sgp_prices.csv`: parse both leg descriptions into
//! [`LegSpec`]s, measure the empirical p1, p2, joint and φ_true at those
//! thresholds straight from the project's datasets (with a pitcher-year
//! cluster-bootstrap interval), invert the prices to φ_book at an assumed
//! hold, and report the gap φ_true − φ_book — the thesis's edge estimate.
//!
//! The hold is the one quantity a single SGP price cannot identify (margin
//! and correlation allowance are confounded), so it comes from a `--hold`
//! flag or, by default, from the sheet itself: the `indep` family's legs are
//! built to be independent, so their mean price ratio R is a pure margin
//! observation and hold = mean(R) − 1. The derived value is printed.
//!
//! Measurements are population-level: "Sanchez over 6.5 K" reads as *a
//! starter's* K ≥ 7 against the whole dataset, the same approximation the
//! atlas makes on its grid — a single named pitcher's sample would be a
//! handful of starts. The outpitch leg (F5/ML proxy) uses full-game starter
//! ER, and tied games drop out; both approximations are documented on
//! [`LegSpec`].
//!
//! Usage:
//! `cargo run --release -p pns-core --example exact_line_phi -- [--hold X] [--selftest] [--dataset PATH] [--batters PATH] [--prices PATH] [--out PATH]`

use chrono::Datelike;
use pns_core::atlas::{
    self, BatterObs, LegSpec, Measurement, ViewObs, batter_start_pairs, parse_leg, personal_lines,
    side_totals, starter_views,
};
use pns_core::data::batters::read_batter_lines;
use pns_core::data::dataset::read_csv;
use pns_core::data::types::PitcherStartLog;
use pns_core::paper::normalize_name;
use pns_core::sgp;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Bootstrap replicates per row. Every row gets an interval — this study has
/// ~20 rows, not an 85-row grid, so the cost is nothing.
const BOOTSTRAP_DRAWS: usize = 300;
/// Fixed bootstrap seed, distinct from the atlas's: identical inputs must
/// reproduce this study bit-for-bit.
const BOOTSTRAP_SEED: u64 = 0xD15E_A5E0_0000_0002;

/// One hand-collected SGP quote, one CSV row.
///
/// Duplicated from `examples/sgp_invert.rs` (~20 lines): examples cannot
/// import examples, and two research drivers sharing a struct would couple
/// them for no benefit. Same discipline: `Option` price fields, because a
/// missing price is an availability record ("ER line not offered", "combo
/// refused"), never a price of zero; and timing stays raw text, because
/// anything that does not parse as a non-negative integer is treated as
/// unknown timing and excluded, never assumed pregame.
#[derive(Debug, Deserialize)]
struct PriceRow {
    date: String,
    book: String,
    region: String,
    game: String,
    family: String,
    leg1_desc: String,
    leg1_price: Option<f64>,
    /// The other side of leg 1 (e.g. the under when leg 1 is the over): what a
    /// proper devig needs. `None` on rows collected before both sides were
    /// recorded — those fall back to the hold-interval treatment.
    /// Read by the devig follow-up, not yet by this study.
    #[serde(default)]
    #[allow(dead_code)]
    leg1_other_price: Option<f64>,
    leg2_desc: String,
    leg2_price: Option<f64>,
    /// Same role as `leg1_other_price` for leg 2.
    #[serde(default)]
    #[allow(dead_code)]
    leg2_other_price: Option<f64>,
    sgp_price: Option<f64>,
    minutes_to_first_pitch: String,
    lineups_posted: String,
    notes: String,
}

/// One collected row, measured and inverted.
struct ExactRow {
    date: String,
    book: String,
    region: String,
    game: String,
    family: String,
    leg1_desc: String,
    leg1_threshold: String,
    leg2_desc: String,
    leg2_threshold: String,
    leg1_price: f64,
    leg2_price: f64,
    sgp_price: f64,
    hold: f64,
    n: usize,
    p1: f64,
    p2: f64,
    joint: f64,
    phi_true: f64,
    ci_low: Option<f64>,
    ci_high: Option<f64>,
    phi_book: Option<f64>,
    gap: Option<f64>,
    /// φ_book re-inverted at **devigged** marginals, and the gap against it.
    ///
    /// `phi_book` above is inverted against raw, vig-loaded leg implieds while
    /// `phi_true` is applied at devigged ones, so the two sit on different
    /// scales and their difference overstates the book's under-allowance —
    /// about threefold on the collected rows. Both bases are kept rather than
    /// one replaced, so rows measured before the board existed stay
    /// comparable. `ev` is unaffected either way: it never reads φ_book.
    phi_book_devigged: Option<f64>,
    gap_devigged: Option<f64>,
    /// Joint and EV at devigged marginals — the pre-registered decision
    /// statistic. `None` when neither the board nor the sheet can supply a
    /// devigged marginal for both legs.
    p_true: Option<f64>,
    ev: Option<f64>,
    /// Where each leg's devigged marginal came from — `consensus`, `sheet`, or
    /// empty — and how many books backed a consensus. Provenance is a column
    /// so a cross-book anchor is never mistaken for one book's devig.
    leg1_source: String,
    leg1_books: Option<usize>,
    leg2_source: String,
    leg2_books: Option<usize>,
    lineups_posted: String,
    notes: String,
}

/// The observation sets a leg pair can live on. Starter legs measure over
/// every start in the file; legs needing the opposing starter or team totals
/// measure over starter views of both-starter games; batter legs measure
/// over batter-start pairs.
#[derive(Clone, Copy)]
enum ObsSet {
    Starts,
    Views,
    Batters,
}

struct Data {
    starts: Vec<PitcherStartLog>,
    views: Vec<ViewObs>,
    batters: Vec<BatterObs>,
}

fn load_data(dataset: &Path, batters: &Path) -> Result<Data, Box<dyn std::error::Error>> {
    let starts = read_csv(dataset)?;
    let batter_rows = read_batter_lines(batters)?;
    eprintln!(
        "{} starts, {} batter lines",
        starts.len(),
        batter_rows.len()
    );
    let line_triples: Vec<(u32, i32, u16)> = starts
        .iter()
        .map(|s| (s.pitcher_id, s.game_date.year(), s.strikeouts))
        .collect();
    let lines = personal_lines(&line_triples);
    let totals = side_totals(&batter_rows);
    let (views, _) = starter_views(&starts, &totals, &lines);
    let pairs = batter_start_pairs(&starts, &batter_rows, &lines);
    eprintln!(
        "{} starter views, {} batter-start pairs",
        views.len(),
        pairs.len()
    );
    Ok(Data {
        starts,
        views,
        batters: pairs,
    })
}

/// Which side of a two-sided market a leg is betting.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Side {
    Over,
    Under,
}

/// One cross-book anchor from `outputs/leg_consensus.csv`, written by
/// `examples/leg_board.rs`.
#[derive(Debug, Clone, Deserialize)]
struct ConsensusRow {
    date: String,
    market: String,
    subject: String,
    point: Option<f64>,
    n_books: usize,
    /// Median devigged probability of the **Over** side across books (or of
    /// `subject`'s side for a moneyline).
    consensus_devigged: f64,
}

fn read_consensus(path: &Path) -> Result<Vec<ConsensusRow>, Box<dyn std::error::Error>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let mut reader = csv::Reader::from_path(path)?;
    Ok(reader
        .deserialize()
        .collect::<Result<Vec<ConsensusRow>, _>>()?)
}

/// The feed market, line and side a [`LegSpec`] corresponds to.
///
/// Half-lines are reconstructed from the integer threshold the spec carries,
/// which is the same direction [`parse_leg`] resolved them in: "over 4.5 K" is
/// `StarterK { over: 5 }`, so the line is `5 − 0.5`. Deriving the number back
/// out of the spec rather than re-reading it from the description means the
/// join and the measurement can never disagree about which bet this is.
///
/// `None` for the families no feed carries — team strikeouts, team hits and
/// combined earned runs. Those legs keep the hand-collected treatment.
fn feed_leg(leg: &LegSpec) -> Option<(&'static str, Option<f64>, Side)> {
    use LegSpec::*;
    Some(match *leg {
        StarterK { over } | DuelK { over } => (
            "pitcher_strikeouts",
            Some(f64::from(over) - 0.5),
            Side::Over,
        ),
        StarterKUnder { under } => (
            "pitcher_strikeouts",
            Some(f64::from(under) + 0.5),
            Side::Under,
        ),
        StarterER { under } => (
            "pitcher_earned_runs",
            Some(f64::from(under) + 0.5),
            Side::Under,
        ),
        StarterHitsAllowed { under } => (
            "pitcher_hits_allowed",
            Some(f64::from(under) + 0.5),
            Side::Under,
        ),
        StarterOuts { over } => ("pitcher_outs", Some(f64::from(over) - 0.5), Side::Over),
        BatterK { over } => ("batter_strikeouts", Some(f64::from(over) - 0.5), Side::Over),
        // The F5 moneyline is two-way and has no line; `subject` is the team.
        Outpitch => ("h2h_1st_5_innings", None, Side::Over),
        TeamK { .. }
        | TeamKUnder { .. }
        | TeamHits { .. }
        | CombinedER { .. }
        | BatterHits { .. } => return None,
    })
}

/// The subject a leg description names, folded for comparison.
///
/// Every description in the sheet leads with who the bet is on — "Wacha over
/// 4.5 K", "Torkelson 1+ K", "Dodgers F5 ml" — so the first token is the
/// subject. Kept deliberately dumb: a cleverer parser would be one more thing
/// that can silently match the wrong player.
fn desc_subject(desc: &str) -> Option<String> {
    let folded = normalize_name(desc);
    folded
        .split(' ')
        .next()
        .map(str::to_string)
        .filter(|token| !token.is_empty())
}

/// True when a consensus subject is the one a leg description names.
///
/// The feed spells players in full ("Michael Wacha") and teams by city ("Los
/// Angeles Dodgers"); the sheet writes a surname or a nickname. Matching any
/// whole token covers both without a name table.
fn subject_matches(consensus_subject: &str, wanted: &str) -> bool {
    normalize_name(consensus_subject)
        .split(' ')
        .any(|token| token == wanted)
}

/// The cross-book devigged marginal for one leg, when the board has it.
///
/// Returns `Ok(None)` when the leg is not carried by the feed, no anchor
/// matches, or only one book priced it — a single book is a quote, not a
/// consensus, and using it here would quietly reintroduce the one-book devig
/// this join exists to replace.
///
/// **Ambiguity is an error, not a guess.** Two pitchers sharing a surname on
/// one slate is the exact case where a silent wrong match would corrupt an EV
/// without ever looking wrong.
fn consensus_marginal(
    anchors: &[ConsensusRow],
    date: &str,
    desc: &str,
    leg: &LegSpec,
    where_: &str,
) -> Result<Option<(f64, usize)>, Box<dyn std::error::Error>> {
    let Some((market, point, side)) = feed_leg(leg) else {
        return Ok(None);
    };
    let Some(wanted) = desc_subject(desc) else {
        return Ok(None);
    };
    let matches: Vec<&ConsensusRow> = anchors
        .iter()
        .filter(|anchor| anchor.date == date && anchor.market == market)
        .filter(|anchor| match (anchor.point, point) {
            (Some(have), Some(want)) => (have - want).abs() < 1e-9,
            (None, None) => true,
            _ => false,
        })
        .filter(|anchor| subject_matches(&anchor.subject, &wanted))
        .collect();
    let distinct: std::collections::BTreeSet<&str> = matches
        .iter()
        .map(|anchor| anchor.subject.as_str())
        .collect();
    if distinct.len() > 1 {
        return Err(format!(
            "{where_}: leg {desc:?} matches {} different subjects in the leg board ({}); \
             disambiguate the description before this row can be scored",
            distinct.len(),
            distinct.into_iter().collect::<Vec<_>>().join(", ")
        )
        .into());
    }
    let Some(anchor) = matches.first() else {
        return Ok(None);
    };
    if anchor.n_books < 2 {
        return Ok(None);
    }
    let over = anchor.consensus_devigged;
    Ok(Some((
        match side {
            Side::Over => over,
            Side::Under => 1.0 - over,
        },
        anchor.n_books,
    )))
}

/// Which observation set hosts both legs of a collected pair. Mirrors the
/// variant coverage of the `eval_on_*` predicates; a pair evaluable nowhere
/// is a skipped row with a printed reason, not a zero.
fn obs_set_for(leg1: &LegSpec, leg2: &LegSpec) -> Option<ObsSet> {
    use LegSpec::*;
    let start_level = |leg: &LegSpec| {
        matches!(
            leg,
            StarterK { .. }
                | StarterKUnder { .. }
                | StarterER { .. }
                | StarterHitsAllowed { .. }
                | StarterOuts { .. }
        )
    };
    let batter_level =
        |leg: &LegSpec| matches!(leg, StarterK { .. } | BatterK { .. } | BatterHits { .. });
    let view_level = |leg: &LegSpec| {
        matches!(
            leg,
            StarterK { .. }
                | StarterKUnder { .. }
                | StarterER { .. }
                | TeamK { .. }
                | TeamKUnder { .. }
                | TeamHits { .. }
                | Outpitch
                | CombinedER { .. }
                | DuelK { .. }
        )
    };
    if start_level(leg1) && start_level(leg2) {
        Some(ObsSet::Starts)
    } else if batter_level(leg1) && batter_level(leg2) {
        Some(ObsSet::Batters)
    } else if view_level(leg1) && view_level(leg2) {
        Some(ObsSet::Views)
    } else {
        None
    }
}

/// Measures φ_true for a leg pair over its observation set, with the
/// cluster-bootstrap interval when `bootstrap` is `Some((reps, seed))`.
fn measure_pair(
    set: ObsSet,
    leg1: LegSpec,
    leg2: LegSpec,
    data: &Data,
    bootstrap: Option<(usize, u64)>,
) -> Option<Measurement> {
    match set {
        ObsSet::Starts => atlas::measure(
            &data.starts,
            |s| Some((leg1.eval_on_start(s)?, leg2.eval_on_start(s)?)),
            bootstrap,
        ),
        ObsSet::Views => atlas::measure(
            &data.views,
            |v| Some((leg1.eval_on_view(v)?, leg2.eval_on_view(v)?)),
            bootstrap,
        ),
        ObsSet::Batters => atlas::measure(
            &data.batters,
            |b| Some((leg1.eval_on_batter(b)?, leg2.eval_on_batter(b)?)),
            bootstrap,
        ),
    }
}

/// None on empty or unparseable text — "we don't know when the game started"
/// is treated as in-play, the same discipline as `sgp_invert` and
/// `book_probe::is_pregame`.
fn parse_minutes(raw: &str) -> Option<i64> {
    raw.parse().ok()
}

/// One screened row: pregame and complete (all three prices present).
struct Screened {
    row: PriceRow,
    leg1_price: f64,
    leg2_price: f64,
    sgp_price: f64,
}

/// Skip accounting, printed at the end so collection gaps stay visible.
#[derive(Default)]
struct Skips {
    unknown_timing: usize,
    incomplete: usize,
    unmeasurable: usize,
    unsupported: usize,
    unparseable: usize,
    no_observation_set: usize,
    degenerate: usize,
}

fn read_screened(
    prices: &Path,
    skips: &mut Skips,
) -> Result<Vec<Screened>, Box<dyn std::error::Error>> {
    let mut reader = csv::ReaderBuilder::new()
        // Hand-edited files pick up stray spaces; trim rather than punish them.
        .trim(csv::Trim::All)
        .from_path(prices)?;
    let mut screened = Vec::new();
    let mut collected = 0usize;
    for (index, record) in reader.deserialize::<PriceRow>().enumerate() {
        // +2: the header row, then 1-based counting, so the number matches the
        // line the collector sees in their editor.
        let record = record.map_err(|error| format!("CSV row {}: {error}", index + 2))?;
        collected += 1;
        let where_ = format!("{} {} ({})", record.date, record.game, record.family);
        match parse_minutes(&record.minutes_to_first_pitch) {
            Some(minutes) if minutes >= 0 => {
                match (record.leg1_price, record.leg2_price, record.sgp_price) {
                    (Some(leg1_price), Some(leg2_price), Some(sgp_price)) => {
                        screened.push(Screened {
                            row: record,
                            leg1_price,
                            leg2_price,
                            sgp_price,
                        });
                    }
                    // A missing price is availability evidence ("ER line not
                    // offered", "combo refused"), not an inversion input.
                    _ => {
                        skips.incomplete += 1;
                        println!("skip {where_}: missing a price (availability record)");
                    }
                }
            }
            _ => {
                skips.unknown_timing += 1;
                println!("skip {where_}: unknown or in-play timing");
            }
        }
    }
    println!("screened {} of {collected} collected rows", screened.len());
    Ok(screened)
}

/// The default hold. Preferred source: indep-family rows with both sides
/// recorded — their offered joint over the *devigged* independence product is
/// a clean margin observation, unpolluted by leg vig. Fallback: mean(R) − 1
/// over raw-implied indep rows (leg vig included, so it reads high). Either
/// way it self-calibrates as the sheet grows.
fn derive_hold(
    screened: &[Screened],
    anchors: &[ConsensusRow],
) -> Result<f64, Box<dyn std::error::Error>> {
    let mut devigged: Vec<f64> = Vec::new();
    for s in screened.iter().filter(|s| s.row.family == "indep") {
        let row = &s.row;
        let where_ = format!("{} {} (indep)", row.date, row.game);
        // Same parse the measurement does, including the indep family's
        // second leg being the *opposing* starter.
        let (Ok(leg1), Ok(leg2_raw)) = (
            parse_leg(&row.family, &row.leg1_desc),
            parse_leg(&row.family, &row.leg2_desc),
        ) else {
            continue;
        };
        let leg2 = match leg2_raw {
            LegSpec::StarterK { over } => LegSpec::DuelK { over },
            other => other,
        };
        let marginal1 = resolve_marginal(
            anchors,
            &row.date,
            &row.leg1_desc,
            &leg1,
            s.leg1_price,
            row.leg1_other_price,
            &where_,
        )?;
        let marginal2 = resolve_marginal(
            anchors,
            &row.date,
            &row.leg2_desc,
            &leg2,
            s.leg2_price,
            row.leg2_other_price,
            &where_,
        )?;
        let (Some(p1_t), Some(p2_t)) = (marginal1.value, marginal2.value) else {
            continue;
        };
        let Some(offered) = sgp::offered_joint(s.sgp_price) else {
            continue;
        };
        devigged.push(offered / (p1_t * p2_t));
    }
    if !devigged.is_empty() {
        let mean = devigged.iter().sum::<f64>() / devigged.len() as f64;
        println!(
            "derived hold {:.4} = mean(offered / devigged independence) - 1 over {} both-sided indep rows",
            mean - 1.0,
            devigged.len()
        );
        return Ok(mean - 1.0);
    }
    let ratios: Vec<f64> = screened
        .iter()
        .filter(|s| s.row.family == "indep")
        .filter_map(|s| {
            let p1 = sgp::implied_probability(s.leg1_price)?;
            let p2 = sgp::implied_probability(s.leg2_price)?;
            sgp::price_ratio(p1, p2, s.sgp_price)
        })
        .collect();
    if ratios.is_empty() {
        return Err("no indep-family rows to calibrate the hold from; pass --hold".into());
    }
    let mean_r = ratios.iter().sum::<f64>() / ratios.len() as f64;
    println!(
        "derived hold {:.4} = mean(R) - 1 over {} raw-implied indep rows (no both-sided rows yet; reads high)",
        mean_r - 1.0,
        ratios.len()
    );
    Ok(mean_r - 1.0)
}

/// One leg's devigged marginal and where it came from.
#[derive(Debug, Clone, PartialEq)]
struct Marginal {
    /// `None` when neither the board nor the sheet can devig this leg — which
    /// is not license to treat it as vig-free, just as unscoreable.
    value: Option<f64>,
    /// `consensus`, `sheet`, or empty when there is no marginal at all.
    source: String,
    /// How many books backed a consensus; `None` for the sheet fallback.
    books: Option<usize>,
}

impl Marginal {
    fn none() -> Self {
        Self {
            value: None,
            source: String::new(),
            books: None,
        }
    }
}

/// One leg's devigged marginal, with its provenance.
///
/// Precedence is fixed here and nowhere else, so the hold derivation and the
/// EV estimate can never end up using different marginals for the same leg:
///
/// 1. the cross-book consensus, when 2+ books priced the leg,
/// 2. the sheet's hand-typed other side,
/// 3. nothing — a one-sided quote is not license to treat the leg as vig-free.
fn resolve_marginal(
    anchors: &[ConsensusRow],
    date: &str,
    desc: &str,
    leg: &LegSpec,
    own_price: f64,
    other_price: Option<f64>,
    where_: &str,
) -> Result<Marginal, Box<dyn std::error::Error>> {
    if let Some((value, books)) = consensus_marginal(anchors, date, desc, leg, where_)? {
        return Ok(Marginal {
            value: Some(value),
            source: "consensus".into(),
            books: Some(books),
        });
    }
    Ok(
        match other_price.and_then(|other| sgp::devigged_probability(own_price, other)) {
            Some(value) => Marginal {
                value: Some(value),
                source: "sheet".into(),
                books: None,
            },
            None => Marginal::none(),
        },
    )
}

/// `(p_true, ev_per_unit)` at devigged marginals: the joint is the devigged
/// independence product plus the empirical correlation applied at those
/// marginals, and EV is against the offered SGP price. This is the
/// pre-registered decision statistic — no hold assumption anywhere in it.
fn ev_estimate(p1_t: f64, p2_t: f64, phi_true: f64, sgp_price: f64) -> Option<(f64, f64)> {
    let denominator = (p1_t * (1.0 - p1_t) * p2_t * (1.0 - p2_t)).sqrt();
    let p_true = p1_t * p2_t + phi_true * denominator;
    let ev = sgp::ev_per_unit(p_true, sgp_price)?;
    Some((p_true, ev))
}

/// Measures and inverts one screened row; `None` (with a printed reason and a
/// counted skip) when the legs refuse to parse, pair, or measure.
fn measure_row(
    screened: &Screened,
    hold: f64,
    data: &Data,
    anchors: &[ConsensusRow],
    skips: &mut Skips,
) -> Result<Option<ExactRow>, Box<dyn std::error::Error>> {
    let row = &screened.row;
    let where_ = format!("{} {} ({})", row.date, row.game, row.family);
    let mut parse = |desc: &str| match parse_leg(&row.family, desc) {
        Ok(leg) => Ok(leg),
        Err(error) => {
            if error.starts_with("unmeasurable:") {
                skips.unmeasurable += 1;
            } else if error.starts_with("unsupported:") {
                skips.unsupported += 1;
            } else {
                skips.unparseable += 1;
            }
            println!("skip {where_}: {error}");
            Err(error)
        }
    };
    let Ok(leg1) = parse(&row.leg1_desc) else {
        return Ok(None);
    };
    let Ok(leg2_raw) = parse(&row.leg2_desc) else {
        return Ok(None);
    };
    // The indep control pairs two opposing starters' K-overs; the sheet's
    // second leg is *their* starter, measured as the duel leg.
    let leg2 = match (row.family.as_str(), leg2_raw) {
        ("indep", LegSpec::StarterK { over }) => LegSpec::DuelK { over },
        _ => leg2_raw,
    };
    let Some(set) = obs_set_for(&leg1, &leg2) else {
        skips.no_observation_set += 1;
        println!("skip {where_}: no observation set hosts both legs");
        return Ok(None);
    };
    let Some(m) = measure_pair(
        set,
        leg1,
        leg2,
        data,
        Some((BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)),
    ) else {
        skips.degenerate += 1;
        println!("skip {where_}: degenerate or empty measurement");
        return Ok(None);
    };
    // Prices are the one thing that must be right: an invalid American price
    // is a hard error naming the row — a typo fixed beats a row dropped.
    let reject = |what: &str| -> Box<dyn std::error::Error> { format!("{where_}: {what}").into() };
    let p1 = sgp::implied_probability(screened.leg1_price)
        .ok_or_else(|| reject("leg1_price is not a valid American price"))?;
    let p2 = sgp::implied_probability(screened.leg2_price)
        .ok_or_else(|| reject("leg2_price is not a valid American price"))?;
    // At extreme leg prices the φ inversion is leverage, not measurement:
    // with p2 = 0.86 a one-cent price wiggle swings the implied correlation
    // by ~0.06, and one-sided batter props (−780) land far past that. φ_true
    // and the devigged EV still measure fine, so the row keeps both; the
    // book-side allowance is reported as unmeasurable rather than as nonsense.
    let phi_book = if p1.max(p2) >= 0.80 {
        None
    } else {
        Some(
            sgp::implied_phi_book(p1, p2, screened.sgp_price, hold)
                .ok_or_else(|| reject("cannot invert the SGP price to a correlation allowance"))?,
        )
    };
    let gap = phi_book.map(|book| m.phi - book);
    // The decision statistic needs devigged marginals. Precedence: the
    // cross-book consensus first (more books, sharper anchor, and it exists
    // for legs a single book only quotes one side of), then the hand-typed
    // other side, then nothing — a row without either is measured but not
    // bet-judged, exactly as before the board existed.
    let marginal1 = resolve_marginal(
        anchors,
        &row.date,
        &row.leg1_desc,
        &leg1,
        screened.leg1_price,
        row.leg1_other_price,
        &where_,
    )?;
    let marginal2 = resolve_marginal(
        anchors,
        &row.date,
        &row.leg2_desc,
        &leg2,
        screened.leg2_price,
        row.leg2_other_price,
        &where_,
    )?;
    let devigged = marginal1.value.zip(marginal2.value);
    let (p_true, ev) = devigged
        .and_then(|(p1_t, p2_t)| ev_estimate(p1_t, p2_t, m.phi, screened.sgp_price))
        .unzip();
    // The book's allowance on the same basis φ_true is applied at. Same
    // leverage guard as the raw-basis inversion: past p = 0.80 a one-cent
    // wiggle swings the implied correlation more than the quantity measured.
    let phi_book_devigged = devigged
        .filter(|(a, b)| a.max(*b) < 0.80)
        .and_then(|(p1_t, p2_t)| sgp::implied_phi_book(p1_t, p2_t, screened.sgp_price, hold));
    let gap_devigged = phi_book_devigged.map(|book| m.phi - book);
    Ok(Some(ExactRow {
        date: row.date.clone(),
        book: row.book.clone(),
        region: row.region.clone(),
        game: row.game.clone(),
        family: row.family.clone(),
        leg1_desc: row.leg1_desc.clone(),
        leg1_threshold: leg1.label(),
        leg2_desc: row.leg2_desc.clone(),
        leg2_threshold: leg2.label(),
        leg1_price: screened.leg1_price,
        leg2_price: screened.leg2_price,
        sgp_price: screened.sgp_price,
        hold,
        n: m.n,
        p1: m.p_a,
        p2: m.p_b,
        joint: m.joint,
        phi_true: m.phi,
        ci_low: m.interval.map(|(low, _)| low),
        ci_high: m.interval.map(|(_, high)| high),
        phi_book,
        gap,
        phi_book_devigged,
        gap_devigged,
        p_true,
        ev,
        leg1_source: marginal1.source,
        leg1_books: marginal1.books,
        leg2_source: marginal2.source,
        leg2_books: marginal2.books,
        lineups_posted: row.lineups_posted.clone(),
        notes: row.notes.clone(),
    }))
}

fn print_table(rows: &[ExactRow]) {
    println!(
        "\n{:<10} {:<8} {:<16} {:<14} {:<16} {:>7} {:>7} {:>7} {:>7} {:>9} {:>19} {:>9} {:>8} {:>8}",
        "date",
        "game",
        "family",
        "leg1",
        "leg2",
        "n",
        "p1",
        "p2",
        "joint",
        "phi_true",
        "95% CI",
        "phi_book",
        "gap",
        "ev"
    );
    for row in rows {
        let ci = match (row.ci_low, row.ci_high) {
            (Some(low), Some(high)) => format!("[{low:+.3}, {high:+.3}]"),
            _ => "—".to_string(),
        };
        let phi_book = row
            .phi_book
            .map_or_else(|| "—".to_string(), |book| format!("{book:+.3}"));
        let gap = row
            .gap
            .map_or_else(|| "—".to_string(), |gap| format!("{gap:+.3}"));
        let ev = row
            .ev
            .map_or_else(|| "—".to_string(), |ev| format!("{:+.1}%", ev * 100.0));
        println!(
            "{:<10} {:<8} {:<16} {:<14} {:<16} {:>7} {:>7.3} {:>7.3} {:>7.3} {:>+9.3} {:>19} {:>9} {:>8} {:>8}",
            row.date,
            row.game,
            row.family,
            row.leg1_threshold,
            row.leg2_threshold,
            row.n,
            row.p1,
            row.p2,
            row.joint,
            row.phi_true,
            ci,
            phi_book,
            gap,
            ev,
        );
    }
}

/// Per-family mean gaps, weighted by each row's measurement n: a φ_true from
/// 130k batter-start pairs is a firmer measurement than one from 15k views,
/// and the weight says so.
fn print_family_summary(rows: &[ExactRow]) {
    let mut groups: BTreeMap<&str, (f64, f64)> = BTreeMap::new();
    for row in rows {
        // Extreme-price rows carry no inversion; pooling their gaps would
        // inject the leverage artifact the guard exists to exclude.
        let Some(gap) = row.gap else { continue };
        let entry = groups.entry(row.family.as_str()).or_default();
        entry.0 += gap * row.n as f64;
        entry.1 += row.n as f64;
    }
    println!("\nper-family n-weighted mean gaps (phi_true - phi_book):");
    for (family, (weighted, n)) in groups {
        println!(
            "  {family:<16} {:+.4}  (total n {})",
            weighted / n,
            n as usize
        );
    }
}

fn write_csv(rows: &[ExactRow], out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = csv::Writer::from_path(out)?;
    writer.write_record([
        "date",
        "book",
        "region",
        "game",
        "family",
        "leg1_desc",
        "leg1_threshold",
        "leg2_desc",
        "leg2_threshold",
        "leg1_price",
        "leg2_price",
        "sgp_price",
        "hold",
        "n",
        "p1",
        "p2",
        "joint",
        "phi_true",
        "ci_low",
        "ci_high",
        "phi_book",
        "gap",
        "phi_book_devigged",
        "gap_devigged",
        "p_true",
        "ev",
        "leg1_source",
        "leg1_books",
        "leg2_source",
        "leg2_books",
        "lineups_posted",
        "notes",
    ])?;
    let number = |value: Option<f64>| value.map_or_else(String::new, |v| format!("{v:.6}"));
    let count = |value: Option<usize>| value.map_or_else(String::new, |v| v.to_string());
    for row in rows {
        writer.write_record([
            row.date.clone(),
            row.book.clone(),
            row.region.clone(),
            row.game.clone(),
            row.family.clone(),
            row.leg1_desc.clone(),
            row.leg1_threshold.clone(),
            row.leg2_desc.clone(),
            row.leg2_threshold.clone(),
            format!("{:.0}", row.leg1_price),
            format!("{:.0}", row.leg2_price),
            format!("{:.0}", row.sgp_price),
            format!("{:.6}", row.hold),
            row.n.to_string(),
            format!("{:.6}", row.p1),
            format!("{:.6}", row.p2),
            format!("{:.6}", row.joint),
            format!("{:.6}", row.phi_true),
            number(row.ci_low),
            number(row.ci_high),
            number(row.phi_book),
            number(row.gap),
            number(row.phi_book_devigged),
            number(row.gap_devigged),
            number(row.p_true),
            number(row.ev),
            row.leg1_source.clone(),
            count(row.leg1_books),
            row.leg2_source.clone(),
            count(row.leg2_books),
            row.lineups_posted.clone(),
            row.notes.clone(),
        ])?;
    }
    writer.flush()?;
    println!("\nwrote {} rows to {}", rows.len(), out.display());
    Ok(())
}

/// Grid reproduction: the exact-line engine at the atlas's own grid
/// thresholds must reproduce the published landmarks (tolerance 1e-3). This
/// is what licenses reading exact-line φ_true as the same measurement the
/// atlas makes, just off the grid.
fn selftest(args: &Args) -> Result<bool, Box<dyn std::error::Error>> {
    let data = load_data(&args.dataset, &args.batters)?;
    let checks: [(&str, LegSpec, LegSpec, ObsSet, f64); 4] = [
        (
            "k>=6 x er<=2",
            LegSpec::StarterK { over: 6 },
            LegSpec::StarterER { under: 2 },
            ObsSet::Starts,
            0.1976,
        ),
        (
            "k>=6 x hits<=5",
            LegSpec::StarterK { over: 6 },
            LegSpec::StarterHitsAllowed { under: 5 },
            ObsSet::Starts,
            0.1474,
        ),
        (
            "k>=6 x outs>=18",
            LegSpec::StarterK { over: 6 },
            LegSpec::StarterOuts { over: 18 },
            ObsSet::Starts,
            0.2599,
        ),
        (
            "duel k>=6 x their_k>=6",
            LegSpec::StarterK { over: 6 },
            LegSpec::DuelK { over: 6 },
            ObsSet::Views,
            -0.013,
        ),
    ];
    let mut all_pass = true;
    for (name, leg1, leg2, set, landmark) in checks {
        let m = measure_pair(set, leg1, leg2, &data, None)
            .ok_or(format!("selftest {name}: degenerate measurement"))?;
        let pass = (m.phi - landmark).abs() <= 1e-3;
        println!(
            "selftest {name}: phi {:+.4} vs landmark {:+.4} [{}]",
            m.phi,
            landmark,
            if pass { "PASS" } else { "FAIL" }
        );
        all_pass &= pass;
    }
    Ok(all_pass)
}

struct Args {
    dataset: PathBuf,
    batters: PathBuf,
    prices: PathBuf,
    /// Cross-book anchors from `examples/leg_board.rs`. Absent is fine — the
    /// study falls back to the sheet's hand-typed second sides, which is how
    /// it worked before the board existed.
    consensus: PathBuf,
    out: PathBuf,
    hold: Option<f64>,
    selftest: bool,
}

impl Args {
    fn defaults(root: &Path) -> Self {
        Self {
            dataset: root.join("data/processed/training_starts.csv"),
            batters: root.join("data/processed/batter_lines.csv"),
            prices: root.join("data/processed/sgp_prices.csv"),
            consensus: root.join("outputs/leg_consensus.csv"),
            out: root.join("outputs/exact_line_phi.csv"),
            hold: None,
            selftest: false,
        }
    }
}

fn parse_args(root: &Path) -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = Args::defaults(root);
    let mut raw = std::env::args().skip(1);
    while let Some(argument) = raw.next() {
        let (name, inline) = match argument.split_once('=') {
            Some((name, value)) => (name.to_string(), Some(value.to_string())),
            None => (argument, None),
        };
        let mut value = || -> Result<String, Box<dyn std::error::Error>> {
            inline
                .clone()
                .or_else(|| raw.next())
                .ok_or_else(|| format!("{name} needs a value").into())
        };
        match name.as_str() {
            "--dataset" => args.dataset = PathBuf::from(value()?),
            "--batters" => args.batters = PathBuf::from(value()?),
            "--prices" => args.prices = PathBuf::from(value()?),
            "--consensus" => args.consensus = PathBuf::from(value()?),
            "--out" => args.out = PathBuf::from(value()?),
            "--hold" => args.hold = Some(value()?.parse()?),
            "--selftest" => args.selftest = true,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    Ok(args)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let args = parse_args(&root)?;
    if args.selftest {
        let pass = selftest(&args)?;
        if !pass {
            return Err("selftest failed: exact-line engine does not reproduce the atlas".into());
        }
        return Ok(());
    }
    let data = load_data(&args.dataset, &args.batters)?;
    let anchors = read_consensus(&args.consensus)?;
    if anchors.is_empty() {
        eprintln!(
            "no leg board at {} — marginals fall back to the sheet's hand-typed second sides",
            args.consensus.display()
        );
    } else {
        eprintln!(
            "{} consensus legs loaded from {} ({} backed by 2+ books)",
            anchors.len(),
            args.consensus.display(),
            anchors.iter().filter(|row| row.n_books >= 2).count()
        );
    }
    let mut skips = Skips::default();
    let screened = read_screened(&args.prices, &mut skips)?;
    let hold = match args.hold {
        Some(hold) => {
            println!("hold {hold:.4} from --hold flag");
            hold
        }
        None => derive_hold(&screened, &anchors)?,
    };
    let mut rows = Vec::new();
    for s in &screened {
        if let Some(row) = measure_row(s, hold, &data, &anchors, &mut skips)? {
            rows.push(row);
        }
    }
    println!(
        "\nskips: {} unknown-timing, {} incomplete, {} unmeasurable, {} unsupported, {} unparseable, {} no-observation-set, {} degenerate",
        skips.unknown_timing,
        skips.incomplete,
        skips.unmeasurable,
        skips.unsupported,
        skips.unparseable,
        skips.no_observation_set,
        skips.degenerate
    );
    print_table(&rows);
    print_family_summary(&rows);
    write_csv(&rows, &args.out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The grid-reproduction gate, run against the committed datasets:
    /// K≥6×ER≤2 = +0.1976, K≥6×hits≤5 = +0.1474, K≥6×outs≥18 = +0.2599,
    /// duel = −0.013, tolerance 1e-3. Chosen over a tiny fixture because the
    /// landmarks are properties of the full dataset — a fixture cannot
    /// reproduce +0.1976, only the real 16,477 starts can.
    #[test]
    fn exact_line_engine_reproduces_the_atlas_landmarks() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let args = Args::defaults(&root);
        assert!(super::selftest(&args).unwrap());
    }

    /// The seed selects every draw: identical inputs and seed must produce
    /// identical intervals, and a different seed must move them.
    #[test]
    fn bootstrap_intervals_are_deterministic_for_a_fixed_seed() {
        let day = chrono::NaiveDate::from_ymd_opt(2025, 6, 1).unwrap();
        // 30 pitcher-year clusters (the interval floor is 25) of patterned
        // views, correlated but nowhere near degenerate.
        let views: Vec<ViewObs> = (0..30u64)
            .flat_map(|cluster| {
                (0..4).map(move |i| ViewObs {
                    strikeouts: ((cluster + i) % 9) as u16,
                    earned_runs: ((cluster + 2 * i) % 5) as u16,
                    their_earned_runs: ((2 * cluster + i) % 4) as u16,
                    their_strikeouts: ((3 * cluster + i) % 8) as u16,
                    opposing_team_ks: Some(((cluster + i) % 14) as u32),
                    opposing_team_hits: Some(((2 * cluster + i) % 11) as u32),
                    personal_line: 5,
                    cluster,
                    date: day,
                })
            })
            .collect();
        let first = measure_pair(
            ObsSet::Views,
            LegSpec::StarterK { over: 5 },
            LegSpec::DuelK { over: 4 },
            &Data {
                starts: Vec::new(),
                views,
                batters: Vec::new(),
            },
            Some((200, 0xBEEF)),
        );
        let views_again: Vec<ViewObs> = (0..30u64)
            .flat_map(|cluster| {
                (0..4).map(move |i| ViewObs {
                    strikeouts: ((cluster + i) % 9) as u16,
                    earned_runs: ((cluster + 2 * i) % 5) as u16,
                    their_earned_runs: ((2 * cluster + i) % 4) as u16,
                    their_strikeouts: ((3 * cluster + i) % 8) as u16,
                    opposing_team_ks: Some(((cluster + i) % 14) as u32),
                    opposing_team_hits: Some(((2 * cluster + i) % 11) as u32),
                    personal_line: 5,
                    cluster,
                    date: day,
                })
            })
            .collect();
        let second = measure_pair(
            ObsSet::Views,
            LegSpec::StarterK { over: 5 },
            LegSpec::DuelK { over: 4 },
            &Data {
                starts: Vec::new(),
                views: views_again,
                batters: Vec::new(),
            },
            Some((200, 0xBEEF)),
        );
        let first = first.and_then(|m| m.interval);
        let second = second.and_then(|m| m.interval);
        assert_eq!(first, second, "same seed must resample identically");
    }

    fn anchor(
        market: &str,
        subject: &str,
        point: Option<f64>,
        over: f64,
        books: usize,
    ) -> ConsensusRow {
        ConsensusRow {
            date: "2026-08-19".into(),
            market: market.into(),
            subject: subject.into(),
            point,
            n_books: books,
            consensus_devigged: over,
        }
    }

    fn resolve(
        anchors: &[ConsensusRow],
        desc: &str,
        leg: &LegSpec,
        own: f64,
        other: Option<f64>,
    ) -> Result<Marginal, Box<dyn std::error::Error>> {
        super::resolve_marginal(anchors, "2026-08-19", desc, leg, own, other, "test row")
    }

    /// −110/−110 devigs to exactly 0.5; a favourite/underdog pair keeps its
    /// lean and both directions sum to one. Anything else imports vig into
    /// the joint. The sheet fallback is the path that does this.
    #[test]
    fn the_sheet_fallback_strips_the_vig() {
        let leg = LegSpec::StarterK { over: 5 };
        let got = resolve(&[], "Wacha over 4.5 K", &leg, -110.0, Some(-110.0)).unwrap();
        assert!((got.value.unwrap() - 0.5).abs() < 1e-12);
        assert_eq!(got.source, "sheet");
        assert_eq!(got.books, None);
        let fav = resolve(&[], "Wacha over 4.5 K", &leg, -150.0, Some(130.0))
            .unwrap()
            .value
            .unwrap();
        let dog = resolve(&[], "Wacha over 4.5 K", &leg, 130.0, Some(-150.0))
            .unwrap()
            .value
            .unwrap();
        assert!(
            (fav + dog - 1.0).abs() < 1e-12,
            "fav {fav} + dog {dog} must sum to one"
        );
        // A missing other side means no devig, not a guess.
        assert_eq!(
            resolve(&[], "Wacha over 4.5 K", &leg, -150.0, None).unwrap(),
            Marginal::none()
        );
    }

    /// The board wins over the sheet, and it reaches legs the sheet cannot:
    /// DraftKings lists one side of batter strikeouts, so those rows have no
    /// hand-typed devig and were unscoreable before the board existed.
    #[test]
    fn the_consensus_outranks_the_sheet_and_reaches_one_sided_legs() {
        let anchors = vec![
            anchor("pitcher_strikeouts", "Michael Wacha", Some(4.5), 0.55, 4),
            anchor("batter_strikeouts", "Spencer Torkelson", Some(0.5), 0.72, 3),
        ];
        let got = resolve(
            &anchors,
            "Wacha over 4.5 K",
            &LegSpec::StarterK { over: 5 },
            -106.0,
            Some(-115.0),
        )
        .unwrap();
        assert!(
            (got.value.unwrap() - 0.55).abs() < 1e-12,
            "board, not sheet"
        );
        assert_eq!(got.source, "consensus");
        assert_eq!(got.books, Some(4));

        let got = resolve(
            &anchors,
            "Torkelson 1+ K",
            &LegSpec::BatterK { over: 1 },
            -620.0,
            None,
        )
        .unwrap();
        assert!((got.value.unwrap() - 0.72).abs() < 1e-12);
        assert_eq!(got.source, "consensus");
    }

    /// An under leg reads the complement of the board's Over probability.
    #[test]
    fn under_legs_take_the_other_side_of_the_consensus() {
        let anchors = vec![anchor(
            "pitcher_earned_runs",
            "Kyle Freeland",
            Some(3.5),
            0.28,
            3,
        )];
        let got = resolve(
            &anchors,
            "Freeland under 3.5 er",
            &LegSpec::StarterER { under: 3 },
            -108.0,
            None,
        )
        .unwrap();
        assert!((got.value.unwrap() - 0.72).abs() < 1e-12);
    }

    /// A single book is a quote, not a consensus — it must fall through to the
    /// sheet rather than quietly reintroducing a one-book devig.
    #[test]
    fn a_single_book_anchor_falls_through_to_the_sheet() {
        let anchors = vec![anchor(
            "pitcher_strikeouts",
            "Michael Wacha",
            Some(4.5),
            0.55,
            1,
        )];
        let got = resolve(
            &anchors,
            "Wacha over 4.5 K",
            &LegSpec::StarterK { over: 5 },
            -110.0,
            Some(-110.0),
        )
        .unwrap();
        assert!((got.value.unwrap() - 0.5).abs() < 1e-12);
        assert_eq!(got.source, "sheet");
    }

    /// Two pitchers sharing a surname on one slate is the case where a silent
    /// wrong match would corrupt an EV without ever looking wrong.
    #[test]
    fn an_ambiguous_surname_is_a_loud_refusal() {
        let anchors = vec![
            anchor("pitcher_strikeouts", "Martin Perez", Some(3.5), 0.60, 3),
            anchor("pitcher_strikeouts", "Eury Perez", Some(3.5), 0.71, 3),
        ];
        let error = resolve(
            &anchors,
            "Perez over 3.5 K",
            &LegSpec::StarterK { over: 4 },
            -150.0,
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("2 different subjects"), "got {error}");
    }

    /// The lines the feed does not carry keep the hand-collected treatment
    /// rather than silently matching something else.
    #[test]
    fn legs_the_feed_does_not_carry_have_no_board_key() {
        assert_eq!(super::feed_leg(&LegSpec::TeamK { over: 10 }), None);
        assert_eq!(super::feed_leg(&LegSpec::TeamKUnder { under: 8 }), None);
        assert_eq!(super::feed_leg(&LegSpec::CombinedER { under: 5 }), None);
        // ...and the ones it does carry resolve to the exact half-line.
        assert_eq!(
            super::feed_leg(&LegSpec::StarterK { over: 5 }),
            Some(("pitcher_strikeouts", Some(4.5), super::Side::Over))
        );
        assert_eq!(
            super::feed_leg(&LegSpec::StarterOuts { over: 17 }),
            Some(("pitcher_outs", Some(16.5), super::Side::Over))
        );
        assert_eq!(
            super::feed_leg(&LegSpec::Outpitch),
            Some(("h2h_1st_5_innings", None, super::Side::Over))
        );
    }

    /// The feed spells players in full and teams by city; the sheet writes a
    /// surname or a nickname.
    #[test]
    fn subjects_match_on_any_whole_token() {
        assert!(super::subject_matches("Michael Wacha", "wacha"));
        assert!(super::subject_matches("Cristopher Sánchez", "sanchez"));
        assert!(super::subject_matches("Los Angeles Dodgers", "dodgers"));
        assert!(!super::subject_matches("Michael Wacha", "wach"));
        assert_eq!(
            super::desc_subject("Wacha over 4.5 K").as_deref(),
            Some("wacha")
        );
        assert_eq!(
            super::desc_subject("Dodgers F5 ml").as_deref(),
            Some("dodgers")
        );
    }

    /// p1_t = p2_t = 0.5 with φ = 0.2 gives P_true = 0.25 + 0.2·0.25 = 0.30;
    /// at +250 (decimal 3.5) EV = 0.30 × 3.5 − 1 = +0.05. Known answer, pinned.
    #[test]
    fn ev_estimate_is_a_known_answer() {
        let (p_true, ev) = super::ev_estimate(0.5, 0.5, 0.2, 250.0).unwrap();
        assert!((p_true - 0.30).abs() < 1e-12, "got {p_true}");
        assert!((ev - 0.05).abs() < 1e-12, "got {ev}");
    }
}
