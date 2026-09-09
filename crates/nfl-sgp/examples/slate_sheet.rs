//! Generate a pre-filled SGP collection sheet, so the only thing left to type
//! is the parlay price itself.
//!
//! The offered same-game-parlay price is the one input no feed sells, and
//! reading it means building each ticket in each book's app by hand. That is
//! the binding constraint on the whole study — the predecessor project
//! pre-registered a five-book pass and never ran it, purely because of the
//! typing (METHOD.md lesson 14). Everything *around* that number can be
//! produced mechanically, and this does it: game, family, both leg descriptions
//! naming the real player at the real line each book is posting, both sides of
//! both legs' prices, the kickoff timing, and the stratifying covariate.
//!
//! `sgp_price` is left blank. That is the human's column.
//!
//! ## It spends no credits
//!
//! It reads the board file `leg_board` already wrote rather than calling the
//! API. So `leg_board` stays the only thing that bills, a mistake here costs
//! nothing to correct, and the whole thing is re-runnable offline and testable.
//!
//! ## Where each fact comes from
//!
//! | fact | source |
//! |---|---|
//! | which games, and the kickoff time | the board |
//! | who is playing which role | the **official depth chart**, intersected with what the book actually posts a market on |
//! | the line and both prices | the board, latest pull, mid-probability line |
//! | spread / total covariates | the board |
//! | target-share / pass-TD-share covariates | [`games::pre_game`] over the prior season |
//! | week and season | `games.csv` |
//!
//! The depth chart names the player and the board confirms he is priced. That
//! order matters: prior-season usage would have named Stefon Diggs as New
//! England's Wr1 and Kenneth Walker III as Seattle's Rb1 for week 1 of 2026,
//! and both had changed teams.
//!
//! ## What it refuses to do
//!
//! Every unresolvable role, missing market and unattributable quarterback
//! **skips its row and prints why**. None of them is guessed at. A row that
//! silently substituted the second receiver, or a passer assigned by
//! elimination, would produce a ticket that looks fine and measures something
//! else.
//!
//! Usage:
//! `cargo run --release -p nfl-sgp --example slate_sheet -- --games Chiefs,Ravens [--date YYYY-MM-DD] [--days 7] [--families KEY,...] [--max-rows 120] [--dry-run]`

use nfl_sgp::depth::DepthChart;
use nfl_sgp::families::{self, Direction as FamilyDirection, Family, Role};
use nfl_sgp::games::{self, PreGame};
use nfl_sgp::legs::{LegStat, parse_row};
use nfl_sgp::names::join_key;
use nfl_sgp::nflverse::{self, GameRow};
use nfl_sgp::odds::GAME_SUBJECT;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Refuse to emit more tickets than a person will actually type. The
/// hand-collection analogue of `leg_board`'s credit floor: the failure mode
/// here is not a bill, it is a sheet that never gets filled in.
const DEFAULT_MAX_ROWS: usize = 120;

/// One row of `data/processed/leg_board.csv`, as `leg_board` writes it.
#[derive(Debug, Clone, Deserialize)]
struct BoardRow {
    date: String,
    game: String,
    /// Kept so the stored row round-trips the board file; the generator reads
    /// timing from `minutes_to_kickoff`.
    #[allow(dead_code)]
    commence_time: String,
    minutes_to_kickoff: i64,
    book: String,
    market: String,
    subject: String,
    point: Option<f64>,
    price: f64,
    other_price: f64,
    devigged: f64,
    #[allow(dead_code)]
    overround: f64,
    pulled_at: String,
}

/// One emitted sheet row. Column order and names must match the sheet the
/// scorer reads — pinned by a test against `score_sgp`'s own header.
#[derive(Debug, Serialize)]
struct SheetRow {
    date: String,
    week: String,
    book: String,
    region: String,
    game: String,
    family: String,
    leg1_desc: String,
    leg1_price: String,
    leg1_other_price: String,
    leg2_desc: String,
    leg2_price: String,
    leg2_other_price: String,
    /// Blank on purpose: the human's column.
    sgp_price: String,
    minutes_to_kickoff: String,
    covariate: String,
    notes: String,
}

/// A leg the generator has fully resolved: who, what line, at what prices.
#[derive(Debug, Clone)]
struct ResolvedLeg {
    description: String,
    price: f64,
    other_price: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let args = parse_args(&root)?;

    let board = read_board(&args.board)?;
    if board.is_empty() {
        return Err(format!(
            "{} is empty or missing — pull the board first:\n  cargo run --release -p nfl-sgp \
             --example leg_board -- --games <teams> --markets {}",
            args.board.display(),
            market_aliases(&args.families).join(",")
        )
        .into());
    }
    let chart = DepthChart::read_snapshots(&nflverse::depth_charts_path(&args.data, args.season))?;
    if chart.is_empty() {
        return Err(format!(
            "no depth charts for {} — fetch them:\n  cargo run --release -p nfl-sgp \
             --example fetch_nflverse -- --seasons {} --depth-charts",
            args.season, args.season
        )
        .into());
    }
    let schedule = nflverse::read_games(&nflverse::schedules_path(&args.data))?;
    // Prior-season weekly rows drive the two computed covariates. In week 1
    // that *is* the whole lookback, per plan §2.1.
    let prior = args.season - 1;
    let pre = games::pre_game(
        nflverse::read_player_weeks(&nflverse::player_week_path(&args.data, prior))?,
        nflverse::read_team_weeks(&nflverse::team_week_path(&args.data, prior))?,
        args.season,
    );
    eprintln!(
        "board {} rows | depth charts {} teams | pre-game views {} (from {prior})",
        board.len(),
        chart.teams(),
        pre.len()
    );

    // Latest pull wins. The board appends, so re-pulling a slate leaves two
    // rows for one book and the line picker would otherwise choose between two
    // snapshots of the same market.
    let latest = latest_per_quote(&board);
    let games = select_games(&latest, &args);
    if games.is_empty() {
        list_slate(&latest, &args);
        return Ok(());
    }

    let mut rows: Vec<SheetRow> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    for game in &games {
        let Some((home, away)) = teams_of(game) else {
            skipped.push(format!("{game}: could not read the two team names"));
            continue;
        };
        let Some(schedule_row) = match_schedule(&schedule, &latest, game, home, away, args.season)
        else {
            skipped.push(format!(
                "{game}: no {} row in games.csv for this matchup",
                args.season
            ));
            continue;
        };
        for &family in &args.families {
            for book in &args.books {
                match build_row(
                    family,
                    game,
                    home,
                    away,
                    book,
                    &latest,
                    &chart,
                    &pre,
                    schedule_row,
                ) {
                    Ok(Some(row)) => rows.push(row),
                    Ok(None) => {}
                    Err(why) => skipped.push(format!("{game} / {} / {book}: {why}", family.key())),
                }
            }
        }
    }

    rows.sort_by(|left, right| {
        (&left.book, &left.game, &left.family).cmp(&(&right.book, &right.game, &right.family))
    });

    eprintln!(
        "\n{} tickets generated, {} skipped",
        rows.len(),
        skipped.len()
    );
    for reason in &skipped {
        eprintln!("  {reason}");
    }
    if rows.len() > args.max_rows {
        return Err(format!(
            "{} tickets is more than --max-rows {} will emit. Each one is a price you must read \
             off a book's app by hand; narrow --games or --families.",
            rows.len(),
            args.max_rows
        )
        .into());
    }

    // Every description must survive the scorer's parser, or the week is
    // wasted. Checked here as well as in tests, because the board supplies
    // names this code has never seen.
    for row in &rows {
        parse_row(&row.family, &row.leg1_desc, &row.leg2_desc).map_err(|why| {
            format!(
                "generated an unparseable row ({} / {}): {why}",
                row.game, row.family
            )
        })?;
    }

    if args.dry_run {
        eprintln!("\ndry run — nothing written. Tickets that would be added:");
        for row in &rows {
            eprintln!(
                "  {:<10} {:<34} {:<22} {} | {}",
                row.book, row.game, row.family, row.leg1_desc, row.leg2_desc
            );
        }
        return Ok(());
    }

    let added = append_sheet(&args.out, &rows)?;
    eprintln!(
        "\nappended {added} new tickets to {} ({} already present)",
        args.out.display(),
        rows.len() - added
    );
    eprintln!(
        "Fill in the sgp_price column, then: cargo run --release -p nfl-sgp --example score_sgp"
    );
    Ok(())
}

/// Builds one (game, family, book) ticket, or explains why it cannot.
///
/// `Ok(None)` means the book simply does not post one of the markets, which is
/// ordinary and not worth a line of output; `Err` means something the collector
/// should see.
#[allow(clippy::too_many_arguments)]
fn build_row(
    family: Family,
    game: &str,
    home: &str,
    away: &str,
    book: &str,
    board: &BTreeMap<QuoteKey, BoardRow>,
    chart: &DepthChart,
    pre: &BTreeMap<String, PreGame>,
    schedule_row: &GameRow,
) -> Result<Option<SheetRow>, String> {
    let [leg_a, leg_b] = family.legs();
    // The anchor is always the home side (METHOD.md, 2026-09-08 amendment):
    // two anchors from one game would be maximised over together by
    // `best_by_game`, inflating the statistic.
    let anchor = home;
    let opponent = away;

    let resolved_a = resolve_leg(
        leg_a.role,
        leg_a.stat,
        leg_a.direction,
        anchor,
        opponent,
        game,
        book,
        board,
        chart,
    )?;
    let resolved_b = resolve_leg(
        leg_b.role,
        leg_b.stat,
        leg_b.direction,
        anchor,
        opponent,
        game,
        book,
        board,
        chart,
    )?;
    let (Some(first), Some(second)) = (resolved_a, resolved_b) else {
        return Ok(None);
    };

    let covariate = covariate_for(family, anchor, game, book, board, pre, schedule_row);
    let sample = board_sample(board, game, book).ok_or("no quotes for this book")?;
    Ok(Some(SheetRow {
        date: sample.date.clone(),
        week: schedule_row.week.to_string(),
        book: book.to_string(),
        region: "us".into(),
        game: game.to_string(),
        family: family.key().to_string(),
        leg1_desc: first.description,
        leg1_price: format!("{:.0}", first.price),
        leg1_other_price: format!("{:.0}", first.other_price),
        leg2_desc: second.description,
        leg2_price: format!("{:.0}", second.price),
        leg2_other_price: format!("{:.0}", second.other_price),
        sgp_price: String::new(),
        minutes_to_kickoff: sample.minutes_to_kickoff.to_string(),
        covariate: covariate.map_or_else(String::new, |value| format!("{value:.4}")),
        notes: String::new(),
    }))
}

/// Turns one leg of a family into a description plus prices, or `Ok(None)` when
/// the book does not post the market.
#[allow(clippy::too_many_arguments)]
fn resolve_leg(
    role: Role,
    stat: nfl_sgp::games::Stat,
    direction: FamilyDirection,
    anchor: &str,
    opponent: &str,
    game: &str,
    book: &str,
    board: &BTreeMap<QuoteKey, BoardRow>,
    chart: &DepthChart,
) -> Result<Option<ResolvedLeg>, String> {
    let market = LegStat::for_stat(stat).ok_or_else(|| format!("{stat:?} has no market"))?;
    let over = matches!(direction, FamilyDirection::Over);

    // Which subject the book must be posting this market on.
    let subject: String = match role {
        Role::Game => GAME_SUBJECT.to_string(),
        Role::Team => games::team_full_name(anchor)
            .ok_or_else(|| format!("no display name for {anchor}"))?
            .to_string(),
        Role::Qb | Role::OppQb | Role::Wr1 | Role::Rb1 => {
            let team = if role == Role::OppQb {
                opponent
            } else {
                anchor
            };
            let candidates: Vec<String> = match role {
                Role::Qb | Role::OppQb => chart
                    .ranked(team, "QB")
                    .iter()
                    .map(|entry| entry.display_name.clone())
                    .collect(),
                Role::Wr1 => chart
                    .receiver_pool(team)
                    .iter()
                    .map(|entry| entry.display_name.clone())
                    .collect(),
                _ => chart
                    .ranked(team, "RB")
                    .iter()
                    .map(|entry| entry.display_name.clone())
                    .collect(),
            };
            if candidates.is_empty() {
                return Err(format!("{team} has no depth-chart {role:?}"));
            }
            // The depth chart ranks; the board decides who is actually priced.
            // Taking the highest-ranked player the book posts a market on is
            // what makes an injured or rested starter fall through to the next
            // man rather than producing a ticket nobody can place.
            match candidates
                .iter()
                .find(|name| quote(board, game, book, market.market_key(), name).is_some())
            {
                Some(found) => found.clone(),
                None => return Ok(None),
            }
        }
    };

    let Some(row) = quote(board, game, book, market.market_key(), &subject) else {
        return Ok(None);
    };
    let description = describe(role, market, &subject, row.point, over, anchor)?;
    // The board stores the Over side; an Under leg takes the other price.
    let (price, other_price) = if over {
        (row.price, row.other_price)
    } else {
        (row.other_price, row.price)
    };
    Ok(Some(ResolvedLeg {
        description,
        price,
        other_price,
    }))
}

/// Writes a leg in exactly the grammar `legs::parse_leg` accepts.
fn describe(
    role: Role,
    market: LegStat,
    subject: &str,
    point: Option<f64>,
    over: bool,
    anchor: &str,
) -> Result<String, String> {
    let side = if over { "over" } else { "under" };
    let line = point.ok_or("the board has no line for this leg")?;
    Ok(match (role, market) {
        (Role::Game, _) => format!("game total {side} {line}"),
        (Role::Team, LegStat::TeamTotal) => format!("{subject} team total {side} {line}"),
        // A spread names the team and carries its sign; `legs::parse_spread`
        // reads the sign, and the anchor is the home side by construction.
        (Role::Team, LegStat::Spread) => {
            let name = games::team_full_name(anchor).unwrap_or(anchor);
            format!("{name} {}{line}", if line >= 0.0 { "+" } else { "" })
        }
        (_, LegStat::PassYds) => format!("{subject} {side} {line} pass yds"),
        (_, LegStat::PassTds) => format!("{subject} {side} {line} pass tds"),
        (_, LegStat::RushYds) => format!("{subject} {side} {line} rush yds"),
        (_, LegStat::RecYds) => format!("{subject} {side} {line} rec yds"),
        (_, LegStat::Receptions) => format!("{subject} {side} {line} receptions"),
        (role, market) => return Err(format!("no description for {role:?} on {market:?}")),
    })
}

/// The stratifying covariate, per family (plan §2, METHOD.md).
fn covariate_for(
    family: Family,
    anchor: &str,
    game: &str,
    book: &str,
    board: &BTreeMap<QuoteKey, BoardRow>,
    pre: &BTreeMap<String, PreGame>,
    schedule_row: &GameRow,
) -> Option<f64> {
    match family.covariate().label {
        // **The sign flip.** nflverse's `spread_line` is positive when the home
        // side is favoured, and `families::spread_bucket` calls >= +3 a
        // favourite. A book posts the favourite at a NEGATIVE point. Get this
        // backwards and every favourite lands in the underdog stratum, nothing
        // errors, and the EVs look entirely plausible.
        "anchor_spread" => {
            let name = games::team_full_name(anchor)?;
            let quoted = quote(board, game, book, "spreads", name).and_then(|row| row.point);
            quoted.map(|point| -point).or(schedule_row.spread_line)
        }
        "total_line" => quote(board, game, book, "totals", GAME_SUBJECT)
            .and_then(|row| row.point)
            .or(schedule_row.total_line),
        "pass_td_share_asof" => pre.get(anchor)?.pass_td_share_asof,
        // Only meaningful when the depth chart's Wr1 is also the usage leader
        // the atlas measured its strata on. When they disagree the share is
        // left blank and the row scores against the `all` stratum — a blunter
        // but honest measurement, rather than a covariate describing a
        // different player. This is the train/serve gap being declined rather
        // than papered over.
        "wr1_target_share_asof" => pre.get(anchor)?.wr1_target_share_asof,
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// board indexing
// ---------------------------------------------------------------------------

type QuoteKey = (String, String, String, String);

fn quote_key(game: &str, book: &str, market: &str, subject: &str) -> QuoteKey {
    (
        game.to_string(),
        book.to_string(),
        market.to_string(),
        join_key(subject),
    )
}

/// One quote per (game, book, market, subject): the latest pull, and among the
/// lines it offers the one closest to even money.
///
/// **Mid-probability line selection** is a pre-registered rule (METHOD.md,
/// 2026-09-08 amendment): the two-leg load grows toward the tails, so line
/// choice is the one lever that shrinks the toll before shopping begins. It
/// reads only the marginals, so it cannot be influenced by the parlay price
/// under test. Ties go to the lower line, for determinism.
fn latest_per_quote(board: &[BoardRow]) -> BTreeMap<QuoteKey, BoardRow> {
    let mut newest: BTreeMap<QuoteKey, String> = BTreeMap::new();
    for row in board {
        let key = quote_key(&row.game, &row.book, &row.market, &row.subject);
        let entry = newest.entry(key).or_default();
        if row.pulled_at > *entry {
            *entry = row.pulled_at.clone();
        }
    }
    let mut chosen: BTreeMap<QuoteKey, BoardRow> = BTreeMap::new();
    for row in board {
        if row.minutes_to_kickoff < 0 {
            continue;
        }
        let key = quote_key(&row.game, &row.book, &row.market, &row.subject);
        if newest
            .get(&key)
            .is_none_or(|latest| row.pulled_at != *latest)
        {
            continue;
        }
        let better = match chosen.get(&key) {
            None => true,
            Some(current) => {
                let (new, old) = ((row.devigged - 0.5).abs(), (current.devigged - 0.5).abs());
                new < old
                    || (new == old
                        && row.point.unwrap_or(f64::MAX) < current.point.unwrap_or(f64::MAX))
            }
        };
        if better {
            chosen.insert(key, row.clone());
        }
    }
    chosen
}

fn quote<'a>(
    board: &'a BTreeMap<QuoteKey, BoardRow>,
    game: &str,
    book: &str,
    market: &str,
    subject: &str,
) -> Option<&'a BoardRow> {
    board.get(&quote_key(game, book, market, subject))
}

fn board_sample<'a>(
    board: &'a BTreeMap<QuoteKey, BoardRow>,
    game: &str,
    book: &str,
) -> Option<&'a BoardRow> {
    board
        .iter()
        .find(|((row_game, row_book, _, _), _)| row_game == game && row_book == book)
        .map(|(_, row)| row)
}

/// `"Away Team @ Home Team"` → (home code, away code).
fn teams_of(game: &str) -> Option<(&'static str, &'static str)> {
    let (away, home) = game.split_once(" @ ")?;
    Some((games::team_code(home)?, games::team_code(away)?))
}

/// The `games.csv` row for this matchup, which supplies week and season and
/// double-checks the team-name mapping.
fn match_schedule<'a>(
    schedule: &'a [GameRow],
    board: &BTreeMap<QuoteKey, BoardRow>,
    game: &str,
    home: &str,
    away: &str,
    season: u16,
) -> Option<&'a GameRow> {
    let slate = board
        .iter()
        .find(|((row_game, _, _, _), _)| row_game == game)
        .map(|(_, row)| row.date.clone())?;
    schedule.iter().find(|row| {
        row.season == season
            && games::canonical_team(&row.home_team) == home
            && games::canonical_team(&row.away_team) == away
            // Kickoff can be the day after the slate for a night game.
            && (row.gameday == slate
                || chrono::NaiveDate::parse_from_str(&row.gameday, "%Y-%m-%d")
                    .ok()
                    .zip(chrono::NaiveDate::parse_from_str(&slate, "%Y-%m-%d").ok())
                    .is_some_and(|(a, b)| (a - b).num_days().abs() <= 1))
    })
}

fn market_aliases(families: &[Family]) -> Vec<&'static str> {
    let mut markets: BTreeSet<&'static str> = BTreeSet::new();
    for family in families {
        for leg in family.legs() {
            if let Some(stat) = LegStat::for_stat(leg.stat) {
                markets.insert(match stat {
                    LegStat::PassYds => "pass-yds",
                    LegStat::PassTds => "pass-tds",
                    LegStat::RushYds => "rush-yds",
                    LegStat::RecYds => "rec-yds",
                    LegStat::Receptions => "rec",
                    LegStat::TeamTotal => "team-total",
                    LegStat::GameTotal => "total",
                    LegStat::Spread => "spread",
                });
            }
        }
        // The covariate markets, where they are not already legs.
        match family.covariate().label {
            "anchor_spread" => {
                markets.insert("spread");
            }
            "total_line" => {
                markets.insert("total");
            }
            _ => {}
        }
    }
    markets.into_iter().collect()
}

fn select_games(board: &BTreeMap<QuoteKey, BoardRow>, args: &Args) -> Vec<String> {
    let mut games: BTreeSet<String> = BTreeSet::new();
    for ((game, _, _, _), row) in board {
        if row.date < args.date.to_string() {
            continue;
        }
        let label = game.to_lowercase();
        if args
            .games
            .iter()
            .any(|wanted| label.contains(&wanted.to_lowercase()))
        {
            games.insert(game.clone());
        }
    }
    games.into_iter().collect()
}

fn list_slate(board: &BTreeMap<QuoteKey, BoardRow>, args: &Args) {
    let mut games: BTreeMap<String, i64> = BTreeMap::new();
    for ((game, _, _, _), row) in board {
        games.insert(game.clone(), row.minutes_to_kickoff);
    }
    eprintln!(
        "\nno --games matched. On the board ({} families, {} books → {} tickets per game):\n",
        args.families.len(),
        args.books.len(),
        args.families.len() * args.books.len()
    );
    for (game, minutes) in games {
        eprintln!("  {game:<40} kickoff in {minutes}m");
    }
    eprintln!("\nre-run with --games and a substring of the games you want.");
}

// ---------------------------------------------------------------------------
// io
// ---------------------------------------------------------------------------

fn read_board(path: &Path) -> Result<Vec<BoardRow>, Box<dyn std::error::Error>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let mut reader = csv::Reader::from_path(path)?;
    Ok(reader.deserialize().collect::<Result<Vec<BoardRow>, _>>()?)
}

/// Appends only tickets the sheet does not already hold, keyed on
/// (date, book, game, family).
///
/// Never rewrites an existing row, even if the line has moved: a typed
/// `sgp_price` is the one thing in this project that cannot be recomputed, and
/// a moved line after the fact is a `notes` matter rather than a regeneration.
fn append_sheet(path: &Path, rows: &[SheetRow]) -> Result<usize, Box<dyn std::error::Error>> {
    let mut existing: BTreeSet<(String, String, String, String)> = BTreeSet::new();
    if path.exists() {
        let mut reader = csv::Reader::from_path(path)?;
        for record in reader.records() {
            let record = record?;
            if record.len() >= 6 {
                existing.insert((
                    record[0].to_string(),
                    record[2].to_string(),
                    record[4].to_string(),
                    record[5].to_string(),
                ));
            }
        }
    }
    let fresh: Vec<&SheetRow> = rows
        .iter()
        .filter(|row| {
            !existing.contains(&(
                row.date.clone(),
                row.book.clone(),
                row.game.clone(),
                row.family.clone(),
            ))
        })
        .collect();
    if fresh.is_empty() {
        return Ok(0);
    }
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
    for row in &fresh {
        writer.serialize(row)?;
    }
    writer.flush()?;
    Ok(fresh.len())
}

struct Args {
    date: chrono::NaiveDate,
    season: u16,
    games: Vec<String>,
    families: Vec<Family>,
    books: Vec<String>,
    data: PathBuf,
    board: PathBuf,
    out: PathBuf,
    max_rows: usize,
    dry_run: bool,
}

fn parse_args(root: &Path) -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = Args {
        date: chrono::Local::now().date_naive(),
        season: 2026,
        games: Vec::new(),
        families: families::ALL.to_vec(),
        books: nfl_sgp::odds::TARGET_BOOKS
            .iter()
            .map(|book| book.to_string())
            .collect(),
        data: root.join("data/raw/nflverse"),
        board: root.join("data/processed/leg_board.csv"),
        out: root.join("data/processed/sgp_prices.csv"),
        max_rows: DEFAULT_MAX_ROWS,
        dry_run: false,
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
            "--season" => args.season = value()?.parse()?,
            "--games" => {
                args.games = value()?
                    .split(',')
                    .map(str::trim)
                    .filter(|part| !part.is_empty())
                    .map(str::to_string)
                    .collect();
            }
            "--families" => {
                args.families = value()?
                    .split(',')
                    .map(str::trim)
                    .filter(|part| !part.is_empty())
                    .map(|key| {
                        Family::from_key(key).ok_or_else(|| format!("unknown family {key:?}"))
                    })
                    .collect::<Result<_, _>>()?;
            }
            "--books" => {
                args.books = value()?
                    .split(',')
                    .map(str::trim)
                    .filter(|part| !part.is_empty())
                    .map(str::to_lowercase)
                    .collect();
            }
            "--data" => args.data = PathBuf::from(value()?),
            "--board" => args.board = PathBuf::from(value()?),
            "--out" => args.out = PathBuf::from(value()?),
            "--max-rows" => args.max_rows = value()?.parse()?,
            "--dry-run" => args.dry_run = true,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    Ok(args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nfl_sgp::depth::{DepthChart, SnapshotRow};

    fn board_row(book: &str, market: &str, subject: &str, point: f64, devigged: f64) -> BoardRow {
        BoardRow {
            date: "2026-09-13".into(),
            game: "Buffalo Bills @ Kansas City Chiefs".into(),
            commence_time: "2026-09-13T17:00:00Z".into(),
            minutes_to_kickoff: 300,
            book: book.into(),
            market: market.into(),
            subject: subject.into(),
            point: Some(point),
            price: -110.0,
            other_price: -110.0,
            devigged,
            overround: 1.0476,
            pulled_at: "2026-09-13T12:00:00Z".into(),
        }
    }

    fn chart() -> DepthChart {
        let rows: Vec<SnapshotRow> = [
            "KC,Patrick Mahomes,00-0033873,QB,1",
            "KC,Rashee Rice,00-0039067,WR,1",
            "KC,Isiah Pacheco,00-0037197,RB,1",
            "BUF,Josh Allen,00-0036212,QB,1",
        ]
        .iter()
        .map(|line| {
            let parts: Vec<&str> = line.split(',').collect();
            SnapshotRow {
                dt: "2026-09-08T11:00:00Z".into(),
                team: parts[0].into(),
                player_name: parts[1].into(),
                gsis_id: Some(parts[2].into()),
                pos_abb: parts[3].into(),
                pos_rank: Some(parts[4].parse().unwrap()),
            }
        })
        .collect();
        DepthChart::from_snapshot_rows(rows)
    }

    /// **The property that makes the generator safe.** If it writes a
    /// description the scorer cannot parse, the entire week's collection is
    /// wasted — the prices are typed against rows that will be refused.
    #[test]
    fn every_generated_description_round_trips_through_the_parser() {
        let cases: [(Family, &str, &str); 6] = [
            (
                Family::TeamtotalXPasstd,
                "Kansas City Chiefs team total over 24.5",
                "Patrick Mahomes over 1.5 pass tds",
            ),
            (
                Family::PassydsXWr1recyds,
                "Patrick Mahomes over 274.5 pass yds",
                "Rashee Rice over 58.5 rec yds",
            ),
            (
                Family::Rb1rushXQbpassunder,
                "Isiah Pacheco over 68.5 rush yds",
                "Patrick Mahomes under 274.5 pass yds",
            ),
            (
                Family::BothqbPassyds,
                "Patrick Mahomes over 274.5 pass yds",
                "Josh Allen over 249.5 pass yds",
            ),
            (
                Family::PassydsXRb1rush,
                "Patrick Mahomes over 274.5 pass yds",
                "Isiah Pacheco over 68.5 rush yds",
            ),
            (
                Family::SpreadXTotal,
                "Kansas City Chiefs -2.5",
                "game total over 47.5",
            ),
        ];
        for (family, leg_a, leg_b) in cases {
            parse_row(family.key(), leg_a, leg_b)
                .unwrap_or_else(|why| panic!("{}: {why}", family.key()));
        }
    }

    /// `describe` must produce exactly those shapes for every role and market
    /// the families use.
    #[test]
    fn describe_writes_the_grammar_the_parser_accepts() {
        let cases = [
            (
                Role::Game,
                LegStat::GameTotal,
                GAME_SUBJECT,
                47.5,
                true,
                "game total over 47.5",
            ),
            (
                Role::Team,
                LegStat::TeamTotal,
                "Kansas City Chiefs",
                24.5,
                true,
                "Kansas City Chiefs team total over 24.5",
            ),
            (
                Role::Qb,
                LegStat::PassYds,
                "Patrick Mahomes",
                274.5,
                false,
                "Patrick Mahomes under 274.5 pass yds",
            ),
            (
                Role::Wr1,
                LegStat::RecYds,
                "Rashee Rice",
                58.5,
                true,
                "Rashee Rice over 58.5 rec yds",
            ),
            (
                Role::Rb1,
                LegStat::RushYds,
                "Isiah Pacheco",
                68.5,
                true,
                "Isiah Pacheco over 68.5 rush yds",
            ),
        ];
        for (role, market, subject, point, over, expected) in cases {
            let got = describe(role, market, subject, Some(point), over, "KC").unwrap();
            assert_eq!(got, expected);
            assert!(nfl_sgp::legs::parse_leg(&got).is_ok(), "{got} must parse");
        }
        // A spread names the team and carries its sign, both ways.
        let favourite = describe(Role::Team, LegStat::Spread, "", Some(-2.5), true, "KC").unwrap();
        assert_eq!(favourite, "Kansas City Chiefs -2.5");
        let dog = describe(Role::Team, LegStat::Spread, "", Some(3.5), true, "KC").unwrap();
        assert_eq!(dog, "Kansas City Chiefs +3.5");
        assert!(nfl_sgp::legs::parse_leg(&dog).is_ok());
    }

    /// The pre-registered rule: among a book's lines, the one closest to even
    /// money. Collecting tail lines is collecting where the load is largest.
    #[test]
    fn line_selection_takes_the_line_closest_to_even_money() {
        let board = vec![
            board_row(
                "draftkings",
                "player_pass_yds",
                "Patrick Mahomes",
                224.5,
                0.71,
            ),
            board_row(
                "draftkings",
                "player_pass_yds",
                "Patrick Mahomes",
                274.5,
                0.47,
            ),
            board_row(
                "draftkings",
                "player_pass_yds",
                "Patrick Mahomes",
                324.5,
                0.22,
            ),
        ];
        let latest = latest_per_quote(&board);
        let chosen = quote(
            &latest,
            "Buffalo Bills @ Kansas City Chiefs",
            "draftkings",
            "player_pass_yds",
            "Patrick Mahomes",
        )
        .unwrap();
        assert_eq!(chosen.point, Some(274.5));
    }

    /// The board appends. A re-pulled slate must not leave the picker choosing
    /// between two snapshots of the same market.
    #[test]
    fn a_repulled_board_uses_only_the_latest_quote() {
        let mut old = board_row(
            "draftkings",
            "player_pass_yds",
            "Patrick Mahomes",
            249.5,
            0.50,
        );
        old.pulled_at = "2026-09-13T09:00:00Z".into();
        let new = board_row(
            "draftkings",
            "player_pass_yds",
            "Patrick Mahomes",
            274.5,
            0.47,
        );
        let latest = latest_per_quote(&vec![old, new]);
        let chosen = quote(
            &latest,
            "Buffalo Bills @ Kansas City Chiefs",
            "draftkings",
            "player_pass_yds",
            "Patrick Mahomes",
        )
        .unwrap();
        assert_eq!(
            chosen.point,
            Some(274.5),
            "the later pull wins even though the earlier line was nearer 0.50"
        );
    }

    /// In-play quotes never reach the sheet — METHOD.md lesson 5, filtered at
    /// collection as well as at analysis.
    #[test]
    fn an_in_play_quote_is_dropped() {
        let mut started = board_row("draftkings", "totals", GAME_SUBJECT, 47.5, 0.50);
        started.minutes_to_kickoff = -12;
        assert!(latest_per_quote(&vec![started]).is_empty());
    }

    /// **The sign flip.** nflverse calls the home favourite `+3`;
    /// the book posts `-3`. Backwards, and every favourite is scored in the
    /// underdog stratum with nothing erroring.
    #[test]
    fn the_anchor_spread_negates_the_books_point() {
        let mut row = board_row("draftkings", "spreads", "Kansas City Chiefs", -3.5, 0.52);
        row.market = "spreads".into();
        let latest = latest_per_quote(&vec![row]);
        let schedule = GameRow {
            game_id: "2026_01_BUF_KC".into(),
            season: 2026,
            game_type: "REG".into(),
            week: 1,
            gameday: "2026-09-13".into(),
            weekday: "Sunday".into(),
            gametime: None,
            away_team: "BUF".into(),
            away_score: None,
            home_team: "KC".into(),
            home_score: None,
            result: None,
            total: None,
            overtime: None,
            away_moneyline: None,
            home_moneyline: None,
            spread_line: None,
            away_spread_odds: None,
            home_spread_odds: None,
            total_line: None,
            under_odds: None,
            over_odds: None,
            div_game: None,
            roof: None,
            surface: None,
            temp: None,
            wind: None,
            away_qb_id: None,
            home_qb_id: None,
            away_qb_name: None,
            home_qb_name: None,
        };
        let covariate = covariate_for(
            Family::Rb1rushXQbpassunder,
            "KC",
            "Buffalo Bills @ Kansas City Chiefs",
            "draftkings",
            &latest,
            &BTreeMap::new(),
            &schedule,
        )
        .unwrap();
        assert_eq!(covariate, 3.5, "a book's -3.5 favourite is +3.5 to us");
        assert_eq!(
            families::spread_bucket(covariate),
            families::SpreadBucket::Favourite
        );
    }

    /// A depth-chart starter the book does not price falls through to the next
    /// ranked player rather than skipping the game.
    #[test]
    fn an_unpriced_starter_falls_through_to_the_next_ranked_player() {
        let mut rows = vec![board_row(
            "draftkings",
            "player_reception_yds",
            "Rashee Rice",
            58.5,
            0.50,
        )];
        rows[0].subject = "Rashee Rice".into();
        let latest = latest_per_quote(&rows);
        let chart = chart();
        let resolved = resolve_leg(
            Role::Wr1,
            nfl_sgp::games::Stat::Wr1ReceivingYards,
            FamilyDirection::Over,
            "KC",
            "BUF",
            "Buffalo Bills @ Kansas City Chiefs",
            "draftkings",
            &latest,
            &chart,
        )
        .unwrap()
        .unwrap();
        assert_eq!(resolved.description, "Rashee Rice over 58.5 rec yds");
    }

    /// A market the book does not post is `Ok(None)` — ordinary, not an error.
    #[test]
    fn a_missing_market_yields_no_row_rather_than_an_error() {
        let latest = latest_per_quote(&Vec::new());
        let outcome = resolve_leg(
            Role::Qb,
            nfl_sgp::games::Stat::QbPassingYards,
            FamilyDirection::Over,
            "KC",
            "BUF",
            "Buffalo Bills @ Kansas City Chiefs",
            "draftkings",
            &latest,
            &chart(),
        )
        .unwrap();
        assert!(outcome.is_none());
    }

    /// The market set is derived from the families, so the credit arithmetic
    /// in METHOD.md is what actually gets pulled.
    #[test]
    fn the_market_set_is_derived_from_the_selected_families() {
        assert_eq!(
            market_aliases(&[Family::TeamtotalXPasstd]),
            vec!["pass-tds", "team-total"]
        );
        let all = market_aliases(&families::ALL);
        assert_eq!(all.len(), 7, "all six families need seven markets: {all:?}");
        assert!(all.contains(&"spread") && all.contains(&"total"));
    }

    /// The game label the board writes must resolve to the two franchises.
    #[test]
    fn the_board_label_resolves_to_home_and_away() {
        assert_eq!(
            teams_of("Buffalo Bills @ Kansas City Chiefs"),
            Some(("KC", "BUF"))
        );
        assert_eq!(teams_of("Toronto Argonauts @ Kansas City Chiefs"), None);
        assert_eq!(teams_of("nonsense"), None);
    }
}
