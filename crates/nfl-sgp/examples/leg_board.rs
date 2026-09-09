//! Pull every book's two-sided price for the legs the SGP study builds tickets
//! from, and reduce them to one consensus marginal per leg.
//!
//! ## Why
//!
//! The same-game-parlay thesis needs `EV = P_true/q − 1` at **devigged**
//! marginals, and devigging needs both sides of both legs. In the baseball
//! study hand collection produced both sides on 8 of 32 rows, so most of the
//! sheet could not be scored at all, and the marginals that could be were one
//! book's proportional devig. This pulls every leg two-sided from every book
//! in region `us`, keeping only the pre-registered books, and takes
//! the median devigged probability across them — a sharper anchor than any
//! single book's, with the cross-book spread as its honest error bar.
//!
//! The SGP price itself is **not** on this feed and stays hand-collected in
//! `data/processed/sgp_prices.csv`. Pull the board at the same sitting the
//! sheet is filled in, so the joint and its marginals are contemporaneous.
//!
//! ## Cost discipline
//!
//! The events list is free; only per-event odds calls bill, at
//! `unique markets returned × regions`. Sixteen games × four markets is 64
//! credits a week on the free 500-credit tier — fits, with the floor as the
//! backstop, as long as nobody pulls "everything" on a whim.
//!
//! Two guards, both deliberate:
//!
//! * **`--games` is required.** With no selection the run prints the coming
//!   week's pregame slate and the per-game cost, then exits without spending.
//! * **`--dry-run`** prints the estimate for the selected games and stops
//!   before the first billed call.
//!
//! ## Output
//!
//! * `data/processed/leg_board.csv` — append-only, one row per
//!   (slate, game, book, market, subject, line) with both sides. This file is
//!   a sample being accumulated; a run that overwrote last week would destroy
//!   it.
//! * `outputs/leg_consensus.csv` — rebuilt from the whole board each run, one
//!   row per (slate, market, subject, line): median devigged probability, how
//!   many books backed it, and their spread.
//!
//! Usage:
//! `cargo run --release -p nfl-sgp --example leg_board -- --games Chiefs,Ravens [--markets pass-yds,rec-yds,rush-yds,pass-tds] [--date YYYY-MM-DD] [--days 7] [--region us] [--floor N] [--dry-run]`

use nfl_sgp::names::join_key;
use nfl_sgp::odds::{DEFAULT_REGION, OddsApiClient, OddsEvent, TARGET_BOOKS, TwoSidedQuote};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Short names for the feed's market keys, so a weekly command stays
/// typable. Every key here was checked against the Odds API market list on
/// 2026-09-02. Which ones a week actually needs depends on the families
/// chosen in the pre-registration — pull only those.
const MARKET_ALIASES: &[(&str, &str)] = &[
    ("pass-yds", "player_pass_yds"),
    ("pass-yds-alt", "player_pass_yds_alternate"),
    ("pass-tds", "player_pass_tds"),
    ("pass-att", "player_pass_attempts"),
    ("pass-comp", "player_pass_completions"),
    ("pass-int", "player_pass_interceptions"),
    ("rush-yds", "player_rush_yds"),
    ("rush-yds-alt", "player_rush_yds_alternate"),
    ("rush-att", "player_rush_attempts"),
    ("rec-yds", "player_reception_yds"),
    ("rec-yds-alt", "player_reception_yds_alternate"),
    ("rec", "player_receptions"),
    ("anytime-td", "player_anytime_td"),
    ("first-td", "player_1st_td"),
    ("kick-pts", "player_kicking_points"),
    ("fg", "player_field_goals"),
    ("sacks", "player_sacks"),
    ("tackles", "player_tackles_assists"),
    ("ml", "h2h"),
    ("spread", "spreads"),
    ("total", "totals"),
    ("team-total", "team_totals"),
    ("team-total-alt", "alternate_team_totals"),
    ("1h-ml", "h2h_h1"),
    ("1h-spread", "spreads_h1"),
    ("1h-total", "totals_h1"),
    ("1h-team-total", "team_totals_h1"),
];

/// A conservative default: the four player markets the obvious families are
/// built from. Team markets are opt-in — each one is another credit per game.
const DEFAULT_MARKETS: &str = "pass-yds,rec-yds,rush-yds,pass-tds";

struct Args {
    /// First slate of the window (inclusive).
    date: chrono::NaiveDate,
    /// Window length in days. NFL weeks run Thursday to Monday, so 7 from a
    /// Tuesday covers one week exactly.
    days: i64,
    /// Case-insensitive substrings matched against `"Away @ Home"`. Empty
    /// means "no selection", which triggers the free slate listing.
    games: Vec<String>,
    markets: Vec<String>,
    region: String,
    /// The books whose quotes are kept — `odds::TARGET_BOOKS` unless overridden.
    books: Vec<String>,
    /// Stop before the remaining credit balance falls to or below this.
    floor: u32,
    dry_run: bool,
    out: PathBuf,
    consensus_out: PathBuf,
}

/// One book's two-sided quote on one leg, as stored.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BoardRow {
    /// The slate date — see `odds::slate_date`. Sunday for SNF even though
    /// its kickoff is Monday in UTC.
    date: String,
    /// `"Away @ Home"` in the feed's own team names — no abbreviation table
    /// to invent, and the same string the consensus and the join key off.
    game: String,
    commence_time: String,
    /// Negative would mean in-play; only pregame rows are ever written, and
    /// the field is kept so analysis can re-filter an accumulated file.
    minutes_to_kickoff: i64,
    book: String,
    market: String,
    subject: String,
    /// Empty for a moneyline or a Yes/No market.
    point: Option<f64>,
    /// American price on the Over / Yes / `subject`'s side.
    price: f64,
    /// American price on the other side.
    other_price: f64,
    /// `price`'s side with the margin removed proportionally.
    devigged: f64,
    overround: f64,
    pulled_at: String,
}

/// The cross-book anchor for one leg.
#[derive(Debug, Serialize)]
struct ConsensusRow {
    date: String,
    market: String,
    /// Display spelling from whichever book sorted first; grouping is on the
    /// folded form, so "Ja'Marr Chase" and "JaMarr Chase" are one leg.
    subject: String,
    point: Option<f64>,
    n_books: usize,
    /// Median devigged probability across books. Median, not mean, so one
    /// stale or wide book cannot drag the anchor.
    consensus_devigged: f64,
    min_devigged: f64,
    max_devigged: f64,
    /// `max − min`: how much the books disagree, which is the honest error bar
    /// on the anchor.
    spread: f64,
    books: String,
}

/// What one run will cost, in credits.
///
/// The feed bills `unique markets returned × regions` per event, and the
/// events list is free. Markets no book offers for a game are not billed, so
/// this is an upper bound rather than an exact charge — which is the right
/// direction for a spending guard to be wrong in.
fn estimate_credits(games: usize, markets: usize, regions: usize) -> usize {
    games * markets * regions
}

fn resolve_market(alias: &str) -> Option<&'static str> {
    MARKET_ALIASES
        .iter()
        .find(|(short, key)| *short == alias || *key == alias)
        .map(|(_, key)| *key)
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let middle = sorted.len() / 2;
    Some(if sorted.len() % 2 == 1 {
        sorted[middle]
    } else {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    })
}

/// Groups the board into one consensus row per leg.
///
/// Two rules that matter:
///
/// * **Latest row per book wins.** The board appends, so pulling the same
///   slate twice leaves two rows per book. Counting both would double-weight
///   whichever book was re-pulled; the later quote is also the better one.
/// * **In-play rows are dropped again here.** Only pregame rows are ever
///   written, but the file accumulates across runs and a filter applied only
///   at collection is one code change away from being bypassed.
fn consensus(rows: &[BoardRow]) -> Vec<ConsensusRow> {
    // (date, market, folded subject, line) -> book -> (pulled_at, row)
    type Key = (String, String, String, String);
    let mut grouped: BTreeMap<Key, BTreeMap<String, (String, BoardRow)>> = BTreeMap::new();
    for row in rows.iter().filter(|row| row.minutes_to_kickoff >= 0) {
        let key = (
            row.date.clone(),
            row.market.clone(),
            join_key(&row.subject),
            row.point
                .map_or_else(String::new, |point| format!("{point:.1}")),
        );
        let per_book = grouped.entry(key).or_default();
        match per_book.get(&row.book) {
            Some((seen, _)) if *seen >= row.pulled_at => {}
            _ => {
                per_book.insert(row.book.clone(), (row.pulled_at.clone(), row.clone()));
            }
        }
    }
    grouped
        .into_values()
        .filter_map(|per_book| {
            let latest: Vec<&BoardRow> = per_book.values().map(|(_, row)| row).collect();
            let first = latest.first()?;
            let devigged: Vec<f64> = latest.iter().map(|row| row.devigged).collect();
            let min = devigged.iter().copied().fold(f64::INFINITY, f64::min);
            let max = devigged.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let books: Vec<String> = per_book.keys().cloned().collect();
            Some(ConsensusRow {
                date: first.date.clone(),
                market: first.market.clone(),
                subject: first.subject.clone(),
                point: first.point,
                n_books: latest.len(),
                consensus_devigged: median(&devigged)?,
                min_devigged: min,
                max_devigged: max,
                spread: max - min,
                books: books.join("|"),
            })
        })
        .collect()
}

fn board_row(event: &OddsEvent, quote: &TwoSidedQuote, pulled_at: &str) -> Option<BoardRow> {
    let (devigged, overround) = quote.devigged()?;
    Some(BoardRow {
        date: event.slate_date()?.to_string(),
        game: event.label(),
        commence_time: event.commence_time.clone(),
        minutes_to_kickoff: event.minutes_to_kickoff()?,
        book: quote.book.clone(),
        market: quote.market.clone(),
        subject: quote.subject.clone(),
        point: quote.point,
        price: quote.price,
        other_price: quote.other_price,
        devigged,
        overround,
        pulled_at: pulled_at.to_string(),
    })
}

fn read_board(path: &Path) -> Result<Vec<BoardRow>, Box<dyn std::error::Error>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let mut reader = csv::Reader::from_path(path)?;
    Ok(reader.deserialize().collect::<Result<Vec<BoardRow>, _>>()?)
}

fn append_board(path: &Path, rows: &[BoardRow]) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let exists = path.exists();
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    let mut writer = csv::WriterBuilder::new()
        .has_headers(!exists)
        .from_writer(file);
    for row in rows {
        writer.serialize(row)?;
    }
    writer.flush()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let args = parse_args(&root)?;
    let markets: Vec<&str> = args
        .markets
        .iter()
        .map(|alias| {
            resolve_market(alias).ok_or_else(|| {
                format!(
                    "unknown market {alias:?}; known aliases: {}",
                    MARKET_ALIASES
                        .iter()
                        .map(|(short, _)| *short)
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
        })
        .collect::<Result<_, _>>()?;

    let mut client = OddsApiClient::from_env()?;
    // Free: this call does not count against the quota, so both the slate
    // listing and the dry run cost nothing.
    let events = client.events()?;
    let window_end = args.date + chrono::Duration::days(args.days);
    let mut underway = 0usize;
    let mut untimed = 0usize;
    let pregame: Vec<&OddsEvent> = events
        .iter()
        .filter(|event| {
            event
                .slate_date()
                .is_some_and(|slate| slate >= args.date && slate < window_end)
        })
        .filter(|event| match event.minutes_to_kickoff() {
            Some(minutes) if minutes >= 0 => true,
            Some(_) => {
                underway += 1;
                false
            }
            None => {
                untimed += 1;
                false
            }
        })
        .collect();
    eprintln!(
        "{} pregame events in [{}, {}) ({underway} underway, {untimed} untimed — both skipped); \
         credits remaining {}",
        pregame.len(),
        args.date,
        window_end,
        client
            .credits_remaining()
            .map_or("unknown".into(), |left| left.to_string())
    );

    if args.games.is_empty() {
        eprintln!(
            "\nno --games given, so nothing was pulled. The window's pregame slate, at {} credits \
             per game for markets [{}]:\n",
            estimate_credits(1, markets.len(), 1),
            args.markets.join(", ")
        );
        for event in &pregame {
            eprintln!(
                "  {:<46} {}  kickoff in {}m",
                event.label(),
                event.slate_date().map_or("?".into(), |d| d.to_string()),
                event.minutes_to_kickoff().unwrap_or_default()
            );
        }
        eprintln!("\nre-run with --games with a substring of the games you want.");
        return Ok(());
    }

    let selected: Vec<&&OddsEvent> = pregame
        .iter()
        .filter(|event| {
            let label = event.label().to_lowercase();
            args.games
                .iter()
                .any(|wanted| label.contains(&wanted.to_lowercase()))
        })
        .collect();
    if selected.is_empty() {
        return Err(format!(
            "none of {:?} matched a pregame game in [{}, {})",
            args.games, args.date, window_end
        )
        .into());
    }
    let estimate = estimate_credits(selected.len(), markets.len(), 1);
    eprintln!(
        "{} games selected x {} markets x 1 region = up to {estimate} credits",
        selected.len(),
        markets.len()
    );
    for event in &selected {
        eprintln!("  {}", event.label());
    }
    if args.dry_run {
        eprintln!("dry run: stopping before the first billed call");
        return Ok(());
    }

    let pulled_at = chrono::Utc::now().to_rfc3339();
    let mut rows: Vec<BoardRow> = Vec::new();
    for event in &selected {
        if let Some(left) = client.credits_remaining()
            && left <= args.floor
        {
            eprintln!(
                "stopping: {left} credits remain, at or below the {} floor",
                args.floor
            );
            break;
        }
        let quotes = match client.event_odds(&event.id, &markets, &args.region) {
            Ok(quotes) => quotes,
            Err(error) => {
                // One game without props is normal; it must not abort the rest
                // of the slate after credits have already been spent.
                eprintln!("  {}: request failed ({error})", event.label());
                continue;
            }
        };
        let before = rows.len();
        let offered = quotes.len();
        // Only the pre-registered books reach the board. The feed returns every
        // book in the region for the same credit and keeping the extras would
        // be free — but the decision statistic is a maximum over books, and a
        // maximum over more books is larger even under fair pricing. The set is
        // part of the measurement, so it is filtered at collection rather than
        // left to whoever reads the file later.
        rows.extend(
            quotes
                .iter()
                .filter(|quote| args.books.contains(&quote.book))
                .filter_map(|quote| board_row(event, quote, &pulled_at)),
        );
        eprintln!(
            "  {}: {} kept from the target books ({offered} offered, {} credits left)",
            event.label(),
            rows.len() - before,
            client
                .credits_remaining()
                .map_or("?".into(), |left| left.to_string())
        );
    }

    if rows.is_empty() {
        return Err("no two-sided quotes collected; nothing written".into());
    }
    append_board(&args.out, &rows)?;
    eprintln!("appended {} rows to {}", rows.len(), args.out.display());

    // Consensus is rebuilt from the whole accumulated board, not just this
    // run, so the file stays a complete index of every leg ever priced.
    let all = read_board(&args.out)?;
    let anchors = consensus(&all);
    if let Some(parent) = args.consensus_out.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut writer = csv::Writer::from_path(&args.consensus_out)?;
    for row in &anchors {
        writer.serialize(row)?;
    }
    writer.flush()?;
    eprintln!(
        "{} consensus legs across {} board rows -> {}",
        anchors.len(),
        all.len(),
        args.consensus_out.display()
    );
    let thin = anchors.iter().filter(|row| row.n_books < 2).count();
    if thin > 0 {
        eprintln!("  {thin} of them are single-book and cannot be used as a consensus anchor");
    }
    Ok(())
}

fn parse_args(root: &Path) -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = Args {
        date: chrono::Local::now().date_naive(),
        days: 7,
        games: Vec::new(),
        markets: DEFAULT_MARKETS.split(',').map(str::to_string).collect(),
        region: DEFAULT_REGION.to_string(),
        books: TARGET_BOOKS.iter().map(|book| book.to_string()).collect(),
        floor: 50,
        dry_run: false,
        out: root.join("data/processed/leg_board.csv"),
        consensus_out: root.join("outputs/leg_consensus.csv"),
    };
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
            "--date" => args.date = chrono::NaiveDate::parse_from_str(&value()?, "%Y-%m-%d")?,
            "--days" => args.days = value()?.parse()?,
            "--games" => {
                args.games = value()?
                    .split(',')
                    .map(str::trim)
                    .filter(|part| !part.is_empty())
                    .map(str::to_string)
                    .collect();
            }
            "--markets" => {
                args.markets = value()?
                    .split(',')
                    .map(str::trim)
                    .filter(|part| !part.is_empty())
                    .map(str::to_string)
                    .collect();
            }
            "--region" => args.region = value()?,
            "--books" => {
                args.books = value()?
                    .split(',')
                    .map(str::trim)
                    .filter(|part| !part.is_empty())
                    .map(str::to_lowercase)
                    .collect();
            }
            "--floor" => args.floor = value()?.parse()?,
            "--dry-run" => args.dry_run = true,
            "--out" => args.out = PathBuf::from(value()?),
            "--consensus-out" => args.consensus_out = PathBuf::from(value()?),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    if args.days < 1 {
        return Err("--days must be at least 1".into());
    }
    Ok(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(book: &str, subject: &str, devigged: f64, pulled_at: &str) -> BoardRow {
        BoardRow {
            date: "2026-09-13".into(),
            game: "Buffalo Bills @ Kansas City Chiefs".into(),
            commence_time: "2026-09-14T00:20:00Z".into(),
            minutes_to_kickoff: 300,
            book: book.into(),
            market: "player_pass_yds".into(),
            subject: subject.into(),
            point: Some(274.5),
            price: -115.0,
            other_price: -105.0,
            devigged,
            overround: 1.0465,
            pulled_at: pulled_at.into(),
        }
    }

    #[test]
    fn credit_estimate_is_markets_times_games_times_regions() {
        assert_eq!(estimate_credits(16, 4, 1), 64);
        assert_eq!(estimate_credits(3, 5, 2), 30);
        assert_eq!(estimate_credits(0, 5, 1), 0);
    }

    #[test]
    fn market_aliases_resolve_both_ways_and_reject_junk() {
        assert_eq!(resolve_market("pass-yds"), Some("player_pass_yds"));
        assert_eq!(resolve_market("player_pass_yds"), Some("player_pass_yds"));
        assert_eq!(resolve_market("team-total"), Some("team_totals"));
        assert_eq!(resolve_market("anytime-td"), Some("player_anytime_td"));
        assert_eq!(resolve_market("k"), None, "that was the other sport");
    }

    #[test]
    fn the_default_market_set_resolves() {
        for alias in DEFAULT_MARKETS.split(',') {
            assert!(resolve_market(alias).is_some(), "{alias} must resolve");
        }
    }

    #[test]
    fn median_handles_both_parities() {
        assert_eq!(median(&[0.5]), Some(0.5));
        assert_eq!(median(&[0.4, 0.6]), Some(0.5));
        assert_eq!(median(&[0.1, 0.5, 0.9]), Some(0.5));
        assert_eq!(median(&[]), None);
    }

    #[test]
    fn consensus_takes_the_median_across_books() {
        let rows = vec![
            row(
                "draftkings",
                "Patrick Mahomes",
                0.52,
                "2026-09-13T18:00:00Z",
            ),
            row("fanduel", "Patrick Mahomes", 0.50, "2026-09-13T18:00:00Z"),
            row("betmgm", "Patrick Mahomes", 0.56, "2026-09-13T18:00:00Z"),
        ];
        let anchors = consensus(&rows);
        assert_eq!(anchors.len(), 1);
        let anchor = &anchors[0];
        assert_eq!(anchor.n_books, 3);
        assert!((anchor.consensus_devigged - 0.52).abs() < 1e-12);
        assert!((anchor.spread - 0.06).abs() < 1e-12);
        assert_eq!(anchor.books, "betmgm|draftkings|fanduel");
    }

    /// The board appends, so a second pull of the same slate must refresh a
    /// book's quote rather than counting it twice — otherwise the re-pulled
    /// book gets double the weight in the median.
    #[test]
    fn a_repulled_book_refreshes_instead_of_double_counting() {
        let rows = vec![
            row(
                "draftkings",
                "Patrick Mahomes",
                0.52,
                "2026-09-13T18:00:00Z",
            ),
            row("fanduel", "Patrick Mahomes", 0.48, "2026-09-13T18:00:00Z"),
            row(
                "draftkings",
                "Patrick Mahomes",
                0.60,
                "2026-09-13T20:00:00Z",
            ),
        ];
        let anchors = consensus(&rows);
        assert_eq!(anchors.len(), 1);
        assert_eq!(anchors[0].n_books, 2);
        assert!((anchors[0].consensus_devigged - 0.54).abs() < 1e-12);
        assert!(
            (anchors[0].max_devigged - 0.60).abs() < 1e-12,
            "the later DraftKings quote is the one that counts"
        );
    }

    /// Books disagree on punctuation; the folded key is what makes
    /// "Ja'Marr Chase" and "JaMarr Chase" one leg rather than two single-book
    /// anchors that both fail the n_books >= 2 bar.
    #[test]
    fn subjects_group_across_punctuation() {
        let rows = vec![
            row("draftkings", "Ja'Marr Chase", 0.48, "2026-09-13T18:00:00Z"),
            row("fanduel", "Ja Marr Chase", 0.50, "2026-09-13T18:00:00Z"),
        ];
        let anchors = consensus(&rows);
        assert_eq!(anchors.len(), 1);
        assert_eq!(anchors[0].n_books, 2);
        assert!((anchors[0].consensus_devigged - 0.49).abs() < 1e-12);
    }

    #[test]
    fn in_play_rows_are_filtered_again_at_analysis() {
        let mut started = row(
            "draftkings",
            "Patrick Mahomes",
            0.52,
            "2026-09-13T18:00:00Z",
        );
        started.minutes_to_kickoff = -20;
        assert!(consensus(&[started]).is_empty());
    }

    #[test]
    fn different_lines_on_one_player_stay_separate_legs() {
        let mut alternate = row("fanduel", "Patrick Mahomes", 0.30, "2026-09-13T18:00:00Z");
        alternate.point = Some(299.5);
        let rows = vec![
            row(
                "draftkings",
                "Patrick Mahomes",
                0.52,
                "2026-09-13T18:00:00Z",
            ),
            alternate,
        ];
        let anchors = consensus(&rows);
        assert_eq!(anchors.len(), 2, "274.5 and 299.5 are different questions");
    }
}
