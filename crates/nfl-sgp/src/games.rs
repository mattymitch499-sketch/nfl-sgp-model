//! The per-game, per-team subject table — `tasks/plan.md` §2.1, §2.2.
//!
//! Three nflverse files describe one football game from three angles: the
//! schedule knows the score and the closing lines, the team-week file knows
//! what the offence did, and the player-week file knows who did it. A parlay
//! family is a pair of numbers drawn from those three, anchored on one team in
//! one game. This module is the join that produces that anchor row, and it
//! carries the one rule the whole study depends on:
//!
//! **Subjects are resolved strictly pre-game.** A book prices a *named*
//! player. If the atlas picks "the receiver who actually led the team that
//! day" it has conditioned on the outcome, and the measured joint is inflated
//! by exactly the selection it performed — a receiver mostly leads his team on
//! days the quarterback threw well, so picking the leader after the fact
//! manufactures the very co-movement the study is trying to measure. Wr1 and
//! Rb1 are therefore chosen from the previous eight team games and nothing
//! else, and if the chosen player did not dress, the row is **dropped rather
//! than replaced**. That single decision is why [`TeamGame::wr1`] is an
//! `Option` and why [`ResolutionCounts`] is part of the public API: a silent
//! fallback here would fabricate the finding, and a drop rate that moves
//! between builds is the first sign the upstream file changed shape.
//!
//! ## The abbreviation trap, and why every team code goes through
//! [`canonical_team`]
//!
//! nflverse is not internally consistent about franchise relocations, and the
//! inconsistency is invisible until a join silently loses three teams' worth
//! of pre-2017 seasons:
//!
//! * `stats_player_week_*.csv` and `stats_team_week_*.csv` write the
//!   **current** code in `team` / `opponent_team` — the 2010 Chargers are
//!   `LAC`, the 2010 Rams are `LA`, the 2010 Raiders are `LV`.
//! * The same files write the **historical** code inside `game_id`:
//!   `2010_01_SD_KC`, `2010_01_ARI_STL`, `2010_01_OAK_TEN`.
//! * `games.csv` writes the historical code in `home_team` / `away_team` and
//!   in `game_id`.
//!
//! So `game_id` joins the three files cleanly, but a team code never compares
//! directly across them. Everything here canonicalises before comparing, which
//! has the second benefit of making a franchise continuous across its move:
//! the prior-season lookback for the 2017 Chargers finds the 2016 Chargers
//! because the stat files call both `LAC`.
//!
//! ## What "one row" means
//!
//! One [`TeamGame`] per (regular-season game, team): two per game, home and
//! away. Team stats come from the team-week file, the score and the closing
//! lines from the schedule, the three subject slots from the player-week file.
//! Everything on the row is either an outcome of that game or a quantity
//! computable before its kickoff; nothing is a quantity from a *later* game,
//! which is what makes the table safe to hand to the standardisation of §1.2.

use crate::nflverse::{self, GameRow, PlayerWeekRow, TeamWeekRow};
use crate::{Error, Result};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

/// The 32 current franchise codes, sorted. Sorted because a fixed order is
/// what makes a team index reproducible, and because the sorted list is the
/// cheapest way to see that the relocations in [`canonical_team`] left exactly
/// 32 teams behind rather than 35.
pub const TEAMS: [&str; 32] = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU",
    "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT",
    "SEA", "SF", "TB", "TEN", "WAS",
];

/// How many previous team games the Wr1 / Rb1 lookback reads (plan §2.1).
pub const SUBJECT_WINDOW_GAMES: usize = 8;

/// Fewer than this many games in the lookback window and the depth chart is
/// not established enough to name a subject; the slot stays `None`.
pub const MIN_SUBJECT_HISTORY_GAMES: usize = 4;

/// The passing-TD share needs at least this many team games behind it (plan
/// §2, `teamtotal_x_passtd`).
pub const MIN_PASS_TD_SHARE_GAMES: usize = 8;

/// The schedule's named starter is only believed if he actually threw this
/// many passes (plan §2.1). Below it he left on the first series, or the
/// column is wrong, and the team's real passer is the one with the attempts.
pub const MIN_QB_ATTEMPTS: f64 = 10.0;

/// Historical team codes mapped to the franchise's current code.
///
/// Identity for everything else, including codes this project has never seen —
/// an unknown code passes through rather than being dropped, so a future
/// relocation shows up as a new team in the counts instead of vanishing.
pub fn canonical_team(raw: &str) -> &str {
    match raw {
        "OAK" => "LV",
        "SD" => "LAC",
        "STL" => "LA",
        other => other,
    }
}

/// The 32 franchises' display names, as the odds feed and the books spell
/// them, paired with the canonical code this crate uses everywhere else.
///
/// Deliberately absent until now. [`crate::odds`] avoids one on purpose ("no
/// abbreviation table to invent and maintain") because the board keys on the
/// feed's own names and the sheet is typed by a human writing those same names,
/// so nothing had to cross the boundary. The sheet *generator* is the first
/// consumer that must: it has to know which side of `"Away @ Home"` a team's
/// history belongs to, and which sign a spread carries.
///
/// It lives here, beside [`TEAMS`] and [`canonical_team`], because this module
/// already owns franchise identity — and splitting one fact across two modules
/// is how the relocation handling would rot.
///
/// Matched through [`crate::names::join_key`], so case, punctuation and
/// spacing do not matter.
pub const TEAM_NAMES: [(&str, &str); 32] = [
    ("ARI", "Arizona Cardinals"),
    ("ATL", "Atlanta Falcons"),
    ("BAL", "Baltimore Ravens"),
    ("BUF", "Buffalo Bills"),
    ("CAR", "Carolina Panthers"),
    ("CHI", "Chicago Bears"),
    ("CIN", "Cincinnati Bengals"),
    ("CLE", "Cleveland Browns"),
    ("DAL", "Dallas Cowboys"),
    ("DEN", "Denver Broncos"),
    ("DET", "Detroit Lions"),
    ("GB", "Green Bay Packers"),
    ("HOU", "Houston Texans"),
    ("IND", "Indianapolis Colts"),
    ("JAX", "Jacksonville Jaguars"),
    ("KC", "Kansas City Chiefs"),
    ("LA", "Los Angeles Rams"),
    ("LAC", "Los Angeles Chargers"),
    ("LV", "Las Vegas Raiders"),
    ("MIA", "Miami Dolphins"),
    ("MIN", "Minnesota Vikings"),
    ("NE", "New England Patriots"),
    ("NO", "New Orleans Saints"),
    ("NYG", "New York Giants"),
    ("NYJ", "New York Jets"),
    ("PHI", "Philadelphia Eagles"),
    ("PIT", "Pittsburgh Steelers"),
    ("SEA", "Seattle Seahawks"),
    ("SF", "San Francisco 49ers"),
    ("TB", "Tampa Bay Buccaneers"),
    ("TEN", "Tennessee Titans"),
    ("WAS", "Washington Commanders"),
];

/// Names a franchise has gone by before, folded onto its current code.
///
/// Renames happen — Washington did it twice inside five years — and a feed that
/// still carries an old name would otherwise become an unmapped team on a
/// Sunday morning. Cheaper to carry the aliases than to debug that live.
const TEAM_NAME_ALIASES: [(&str, &str); 5] = [
    ("WAS", "Washington Football Team"),
    ("WAS", "Washington Redskins"),
    ("LV", "Oakland Raiders"),
    ("LAC", "San Diego Chargers"),
    ("LA", "St. Louis Rams"),
];

/// A full display name to its canonical code. `None` — never a guess — for
/// anything unrecognised, so a feed rename surfaces as a named refusal rather
/// than a game silently anchored on the wrong side.
pub fn team_code(full_name: &str) -> Option<&'static str> {
    let wanted = crate::names::join_key(full_name);
    TEAM_NAMES
        .iter()
        .chain(TEAM_NAME_ALIASES.iter())
        .find(|(_, name)| crate::names::join_key(name) == wanted)
        .map(|(code, _)| *code)
}

/// The canonical code to its current display name.
pub fn team_full_name(code: &str) -> Option<&'static str> {
    let canonical = canonical_team(code);
    TEAM_NAMES
        .iter()
        .find(|(candidate, _)| *candidate == canonical)
        .map(|(_, name)| *name)
}

/// One player's line in one game — only the columns a v1 family can be built
/// from, lifted out of [`PlayerWeekRow`] with blanks already resolved to zero.
#[derive(Debug, Clone, PartialEq)]
pub struct PlayerGame {
    pub player_id: String,
    /// The full name, `player_display_name`, not the abbreviated
    /// `player_name`: this is what a book prints and what
    /// [`crate::names::normalize_name`] folds for the scorer's join.
    pub display_name: String,
    pub position: String,
    pub passing_yards: f64,
    pub passing_tds: f64,
    pub attempts: f64,
    pub rushing_yards: f64,
    pub carries: f64,
    pub receiving_yards: f64,
    pub receptions: f64,
    pub targets: f64,
}

impl PlayerGame {
    fn from_row(row: &PlayerWeekRow) -> Self {
        Self {
            player_id: row.player_id.clone(),
            display_name: row.player_display_name.clone(),
            position: row.position.clone(),
            passing_yards: row.pass_yds(),
            passing_tds: row.pass_tds(),
            attempts: row.pass_atts(),
            rushing_yards: row.rush_yds(),
            carries: row.rush_atts(),
            receiving_yards: row.rec_yds(),
            receptions: row.recs(),
            targets: row.tgts(),
        }
    }
}

/// One team's side of one regular-season game.
#[derive(Debug, Clone)]
pub struct TeamGame {
    pub game_id: String,
    pub season: u16,
    pub week: u8,
    /// Canonical code (see [`canonical_team`]).
    pub team: String,
    /// Canonical code.
    pub opponent: String,
    pub home: bool,
    pub points: i32,
    pub opp_points: i32,
    /// The closing spread **from this team's perspective, positive when this
    /// team is favoured**. `games.csv` states it home-positive, so the away
    /// row carries the negation. `None` when the schedule has no closing line.
    pub anchor_spread: Option<f64>,
    /// The closing total, identical on both rows of a game.
    pub total_line: Option<f64>,
    /// This team's points minus the opponent's. On the home row this equals
    /// `games.csv`'s `result`; on the away row it is its negation.
    pub result_margin: i32,
    pub team_passing_yards: f64,
    pub team_rushing_yards: f64,
    pub team_passing_tds: f64,
    pub team_rushing_tds: f64,
    /// The starting quarterback (plan §2.1). `None` only when no quarterback
    /// appears in the game's player rows for this team at all.
    pub qb: Option<PlayerGame>,
    /// The **as-of** Wr1: the target leader over the lookback window, present
    /// in this game's player rows. `None` when there is not enough history to
    /// name one, or when the named one did not dress — never the realised
    /// leader.
    pub wr1: Option<PlayerGame>,
    /// The as-of Rb1, by carries, under the same rules.
    pub rb1: Option<PlayerGame>,
    /// The as-of Wr1's share of the team's targets over the lookback window.
    ///
    /// Computed for whoever the pre-game rule named, **including when that
    /// player then did not dress**: it describes the depth chart the book
    /// priced, not the game that happened, so it stays meaningful on a row
    /// whose [`TeamGame::wr1`] is `None`.
    pub wr1_target_share_asof: Option<f64>,
    /// The team's passing share of its offensive touchdowns over the prior
    /// season plus the season to date; `None` below
    /// [`MIN_PASS_TD_SHARE_GAMES`] games.
    pub pass_td_share_asof: Option<f64>,
}

impl TeamGame {
    /// The value of `stat` on this row, or `None` when the subject the stat
    /// names did not resolve (or the closing line it is centred on is
    /// missing). One accessor for both the family builder and the
    /// standardisation series, so the two can never read a leg differently.
    pub fn stat(&self, stat: Stat) -> Option<f64> {
        match stat {
            Stat::TeamPoints => Some(f64::from(self.points)),
            Stat::TeamPassingYards => Some(self.team_passing_yards),
            Stat::TeamRushingYards => Some(self.team_rushing_yards),
            Stat::TeamPassingTds => Some(self.team_passing_tds),
            Stat::MarginVsSpread => self
                .anchor_spread
                .map(|spread| f64::from(self.result_margin) - spread),
            Stat::TotalVsLine => self
                .total_line
                .map(|line| f64::from(self.points + self.opp_points) - line),
            Stat::QbPassingYards | Stat::QbPassingTds => {
                self.qb.as_ref().and_then(|player| stat.of_player(player))
            }
            Stat::Wr1ReceivingYards | Stat::Wr1Receptions => {
                self.wr1.as_ref().and_then(|player| stat.of_player(player))
            }
            Stat::Rb1RushingYards => self.rb1.as_ref().and_then(|player| stat.of_player(player)),
        }
    }

    /// The subject that owns `stat` on this row: the team-season for a team
    /// stat, the resolved player's player-season for a player stat.
    pub fn subject(&self, stat: Stat) -> Option<SubjectKey> {
        let player = match stat {
            Stat::TeamPoints
            | Stat::TeamPassingYards
            | Stat::TeamRushingYards
            | Stat::TeamPassingTds
            | Stat::MarginVsSpread
            | Stat::TotalVsLine => return Some(self.team_season()),
            Stat::QbPassingYards | Stat::QbPassingTds => self.qb.as_ref(),
            Stat::Wr1ReceivingYards | Stat::Wr1Receptions => self.wr1.as_ref(),
            Stat::Rb1RushingYards => self.rb1.as_ref(),
        };
        player.map(|found| SubjectKey::Player(found.player_id.clone(), self.season))
    }

    /// The team-season this row is anchored on, whatever the family's stats.
    pub fn team_season(&self) -> SubjectKey {
        SubjectKey::Team(self.team.clone(), self.season)
    }
}

/// The unit a value is standardised within (plan §1.2) and the unit the
/// cluster bootstrap resamples (plan §1.7).
///
/// A player-*season* and a team-*season*, never a player alone: the level of a
/// subject moves between seasons — a receiver's role, a team's pass rate — and
/// the leave-one-out centring of §1.2 only means something inside a stretch
/// over which the level is roughly fixed.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SubjectKey {
    /// `(nflverse player_id, season)`.
    Player(String, u16),
    /// `(canonical team code, season)`.
    Team(String, u16),
}

impl SubjectKey {
    pub fn season(&self) -> u16 {
        match self {
            Self::Player(_, season) | Self::Team(_, season) => *season,
        }
    }

    /// The identifier the atlas stores and the bootstrap groups by.
    ///
    /// FNV-1a 64 over [`SubjectKey::canonical_string`], **not**
    /// [`std::hash::DefaultHasher`]: `DefaultHasher`'s output is explicitly
    /// not guaranteed stable across Rust releases, and a predecessor project
    /// was bitten by exactly that — an atlas written on one toolchain grouped
    /// differently when reloaded on another, silently changing every interval
    /// without changing a single input. FNV-1a is fifteen lines, has published
    /// test vectors (pinned in this module's tests), and will read the same in
    /// ten years.
    pub fn cluster_id(&self) -> u64 {
        fnv1a_64(self.canonical_string().as_bytes())
    }

    /// The exact bytes [`SubjectKey::cluster_id`] hashes. Public because a
    /// stored atlas is only auditable if the pre-image is inspectable.
    pub fn canonical_string(&self) -> String {
        match self {
            Self::Player(id, season) => format!("player:{id}:{season}"),
            Self::Team(team, season) => format!("team:{team}:{season}"),
        }
    }
}

/// FNV-1a, 64-bit.
fn fnv1a_64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET_BASIS;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// A per-game quantity a leg can be written on.
///
/// The `Qb` / `Wr1` / `Rb1` prefixes name the **role that selects the subject**
/// on a [`TeamGame`], not a different statistic. Read off a
/// [`SubjectKey::Player`] series, `Wr1ReceivingYards` is simply that player's
/// receiving yards in every game he played that season — which is what plan
/// §2.1 requires: "standardisation is per (player, season) over all that
/// player's games in the season, regardless of role in any single game".
///
/// [`Stat::MarginVsSpread`] and [`Stat::TotalVsLine`] are two additions beyond
/// the nine raw quantities, forced by the plan's `spread_x_total` family: its
/// legs are already centred on a closing line and so are none of the nine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Stat {
    TeamPoints,
    TeamPassingYards,
    TeamRushingYards,
    TeamPassingTds,
    QbPassingYards,
    QbPassingTds,
    Wr1ReceivingYards,
    Wr1Receptions,
    Rb1RushingYards,
    /// `result − spread_line`, anchor perspective. Market-centred.
    MarginVsSpread,
    /// `total − total_line`. Market-centred.
    TotalVsLine,
}

impl Stat {
    /// A short label for output files.
    pub fn label(self) -> &'static str {
        match self {
            Self::TeamPoints => "team_points",
            Self::TeamPassingYards => "team_pass_yds",
            Self::TeamRushingYards => "team_rush_yds",
            Self::TeamPassingTds => "team_pass_tds",
            Self::QbPassingYards => "qb_pass_yds",
            Self::QbPassingTds => "qb_pass_tds",
            Self::Wr1ReceivingYards => "wr1_rec_yds",
            Self::Wr1Receptions => "wr1_recs",
            Self::Rb1RushingYards => "rb1_rush_yds",
            Self::MarginVsSpread => "margin_vs_spread",
            Self::TotalVsLine => "total_vs_line",
        }
    }

    /// Whether the stat belongs to a [`SubjectKey::Team`] rather than a
    /// [`SubjectKey::Player`].
    pub fn is_team_stat(self) -> bool {
        matches!(
            self,
            Self::TeamPoints
                | Self::TeamPassingYards
                | Self::TeamRushingYards
                | Self::TeamPassingTds
                | Self::MarginVsSpread
                | Self::TotalVsLine
        )
    }

    /// Whether the stat is already expressed as a deviation from a closing
    /// line and must therefore **skip** the leave-one-out standardisation of
    /// plan §1.2: the market has done the centring, on far better information
    /// than a season of leave-one-out means, and centring twice would strip
    /// the very signal the residual carries.
    pub fn is_market_centred(self) -> bool {
        matches!(self, Self::MarginVsSpread | Self::TotalVsLine)
    }

    fn of_player(self, player: &PlayerGame) -> Option<f64> {
        match self {
            Self::QbPassingYards => Some(player.passing_yards),
            Self::QbPassingTds => Some(player.passing_tds),
            Self::Wr1ReceivingYards => Some(player.receiving_yards),
            Self::Wr1Receptions => Some(player.receptions),
            Self::Rb1RushingYards => Some(player.rushing_yards),
            _ => None,
        }
    }
}

/// How often each resolution rule fired.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResolutionCounts {
    /// Rows produced: (regular-season, completed game) × 2.
    pub team_games: usize,
    /// Team-week rows whose game the schedule did not carry, or carried
    /// without a final score. Skipped, not invented.
    pub games_without_schedule: usize,
    /// The schedule's `home_qb_id` / `away_qb_id` was confirmed with at least
    /// [`MIN_QB_ATTEMPTS`] attempts.
    pub qb_from_schedule: usize,
    /// The schedule's id was absent, unconfirmed, or below the attempt floor,
    /// so the team's most-attempting quarterback was used instead.
    pub qb_fallback: usize,
    /// No quarterback at all in the game's player rows for this team.
    pub qb_unresolved: usize,
    pub wr1_resolved: usize,
    /// Not enough lookback: fewer than [`MIN_SUBJECT_HISTORY_GAMES`] games, or
    /// no targeted receiver in the window.
    pub wr1_no_history: usize,
    /// The as-of Wr1 was named but does not appear in this game's player rows.
    /// The row is kept with `wr1 = None` and the family observation is dropped.
    pub wr1_dropped_inactive: usize,
    pub rb1_resolved: usize,
    pub rb1_no_history: usize,
    pub rb1_dropped_inactive: usize,
    /// Rows whose passing-TD share had fewer than [`MIN_PASS_TD_SHARE_GAMES`]
    /// games behind it.
    pub pass_td_share_missing: usize,
}

/// One entry of a player's season series (see [`GameTable::series`]).
#[derive(Debug, Clone)]
struct PlayerSeasonGame {
    week: u8,
    game_id: String,
    line: PlayerGame,
}

/// Row indices of the two sides of one game.
#[derive(Debug, Clone, Copy, Default)]
struct GamePair {
    home: Option<usize>,
    away: Option<usize>,
}

/// The assembled table: the only place Phase 3 reads football out of.
pub struct GameTable {
    /// One row per (game, team), ordered by season, week, game id, then team.
    pub rows: Vec<TeamGame>,
    /// Every **regular-season** row of `games.csv`, whatever season range the
    /// stat files cover. `spread_x_total` is measured on this, because closing
    /// lines reach back eleven seasons further than the weekly stat files this
    /// project downloads (plan §1.9, §2.2).
    pub schedule: Vec<GameRow>,
    /// Seasons the subject table actually covers, ascending.
    pub seasons: Vec<u16>,
    pub counts: ResolutionCounts,
    by_game: HashMap<String, GamePair>,
    team_season_rows: BTreeMap<(String, u16), Vec<usize>>,
    player_season_games: BTreeMap<(String, u16), Vec<PlayerSeasonGame>>,
}

impl GameTable {
    /// Reads the three files for `seasons` out of a raw-data directory.
    ///
    /// `games.csv` is read whole regardless of `seasons`: its extra seasons
    /// cost nothing and the market-centred validation family needs them.
    ///
    /// The lookback for the **first** season in the range has no prior season
    /// to read, so that season's opening weeks resolve no Wr1 or Rb1 and its
    /// first eight games carry no passing-TD share. That is deliberate rather
    /// than papered over: silently loading whatever extra season happens to be
    /// on disk would make the atlas depend on the contents of a directory
    /// instead of on its arguments. Widen the range to widen the history.
    pub fn load(dir: &Path, seasons: std::ops::RangeInclusive<u16>) -> Result<Self> {
        let schedule_path = nflverse::schedules_path(dir);
        if !schedule_path.exists() {
            return Err(Error::Data(format!(
                "{} not found — run the fetch_nflverse example first",
                schedule_path.display()
            )));
        }
        let games = nflverse::read_games(&schedule_path)?;
        let mut players = Vec::new();
        let mut team_weeks = Vec::new();
        for season in seasons {
            let player_path = nflverse::player_week_path(dir, season);
            let team_path = nflverse::team_week_path(dir, season);
            if !player_path.exists() || !team_path.exists() {
                return Err(Error::Data(format!(
                    "season {season} is missing {} or {}",
                    player_path.display(),
                    team_path.display()
                )));
            }
            players.extend(nflverse::read_player_weeks(&player_path)?);
            team_weeks.extend(nflverse::read_team_weeks(&team_path)?);
        }
        Ok(Self::build(games, players, team_weeks))
    }

    /// Assembles the table from already-parsed rows. Post-season rows are
    /// dropped here, once, so no consumer has to remember to (plan §2.2).
    pub fn build(
        games: Vec<GameRow>,
        players: Vec<PlayerWeekRow>,
        team_weeks: Vec<TeamWeekRow>,
    ) -> Self {
        let mut schedule: Vec<GameRow> = games
            .into_iter()
            .filter(GameRow::is_regular_season)
            .collect();
        schedule.sort_by(|left, right| {
            (left.season, left.week, &left.game_id).cmp(&(right.season, right.week, &right.game_id))
        });
        let schedule_by_id: HashMap<&str, usize> = schedule
            .iter()
            .enumerate()
            .map(|(index, game)| (game.game_id.as_str(), index))
            .collect();

        let players: Vec<PlayerWeekRow> = players
            .into_iter()
            .filter(PlayerWeekRow::is_regular_season)
            .collect();
        let team_weeks: Vec<TeamWeekRow> = team_weeks
            .into_iter()
            .filter(TeamWeekRow::is_regular_season)
            .collect();

        let Indexes {
            players_by_game,
            season_totals,
            team_slots,
        } = Indexes::new(&players, &team_weeks);

        let mut counts = ResolutionCounts::default();
        let mut rows: Vec<TeamGame> = Vec::new();
        for (team, slots) in &team_slots {
            for (position, slot) in slots.iter().enumerate() {
                let Some(&schedule_index) = schedule_by_id.get(slot.game_id) else {
                    counts.games_without_schedule += 1;
                    continue;
                };
                let game = &schedule[schedule_index];
                let (Some(home_score), Some(away_score)) = (game.home_score, game.away_score)
                else {
                    counts.games_without_schedule += 1;
                    continue;
                };
                let home = canonical_team(&game.home_team) == *team;
                let context = RowContext {
                    team,
                    slots,
                    position,
                    season: slot.season,
                    players: &players,
                    players_by_game: &players_by_game,
                    team_weeks: &team_weeks,
                    season_totals: &season_totals,
                };
                let (points, opp_points) = if home {
                    (home_score, away_score)
                } else {
                    (away_score, home_score)
                };
                let team_week = &team_weeks[slot.team_week];
                let (qb, qb_source) = context.resolve_qb(if home {
                    game.home_qb_id.as_deref()
                } else {
                    game.away_qb_id.as_deref()
                });
                match qb_source {
                    QbSource::Schedule => counts.qb_from_schedule += 1,
                    QbSource::Fallback => counts.qb_fallback += 1,
                    QbSource::None => counts.qb_unresolved += 1,
                }
                let wr1 = context.resolve_skill(Skill::Receiver);
                let rb1 = context.resolve_skill(Skill::Rusher);
                tally(&mut counts, Skill::Receiver, &wr1);
                tally(&mut counts, Skill::Rusher, &rb1);
                let pass_td_share = context.pass_td_share();
                if pass_td_share.is_none() {
                    counts.pass_td_share_missing += 1;
                }
                counts.team_games += 1;
                rows.push(TeamGame {
                    game_id: slot.game_id.to_string(),
                    season: slot.season,
                    week: slot.week,
                    team: (*team).to_string(),
                    opponent: canonical_team(&team_week.opponent_team).to_string(),
                    home,
                    points,
                    opp_points,
                    anchor_spread: game
                        .spread_line
                        .map(|spread| if home { spread } else { -spread }),
                    total_line: game.total_line,
                    result_margin: points - opp_points,
                    team_passing_yards: team_week.pass_yds(),
                    team_rushing_yards: team_week.rush_yds(),
                    team_passing_tds: team_week.pass_tds(),
                    team_rushing_tds: team_week.rush_tds(),
                    qb,
                    wr1: wr1.player,
                    rb1: rb1.player,
                    wr1_target_share_asof: wr1.share,
                    pass_td_share_asof: pass_td_share,
                });
            }
        }
        rows.sort_by(|left, right| {
            (left.season, left.week, &left.game_id, &left.team).cmp(&(
                right.season,
                right.week,
                &right.game_id,
                &right.team,
            ))
        });

        let mut by_game: HashMap<String, GamePair> = HashMap::new();
        let mut team_season_rows: BTreeMap<(String, u16), Vec<usize>> = BTreeMap::new();
        let mut subjects: BTreeSet<(String, u16)> = BTreeSet::new();
        for (index, row) in rows.iter().enumerate() {
            let pair = by_game.entry(row.game_id.clone()).or_default();
            if row.home {
                pair.home = Some(index);
            } else {
                pair.away = Some(index);
            }
            team_season_rows
                .entry((row.team.clone(), row.season))
                .or_default()
                .push(index);
            for subject in [row.qb.as_ref(), row.wr1.as_ref(), row.rb1.as_ref()]
                .into_iter()
                .flatten()
            {
                subjects.insert((subject.player_id.clone(), row.season));
            }
        }

        // Every game of every player who is a subject somewhere, whatever his
        // role that week — plan §2.1's standardisation pool. Restricted to
        // players who are subjects because nobody else is ever asked for, and
        // the full player file is 300,000 rows.
        let mut player_season_games: BTreeMap<(String, u16), Vec<PlayerSeasonGame>> =
            BTreeMap::new();
        for row in &players {
            let key = (row.player_id.clone(), row.season);
            if !subjects.contains(&key) {
                continue;
            }
            player_season_games
                .entry(key)
                .or_default()
                .push(PlayerSeasonGame {
                    week: row.week,
                    game_id: row.game_id.clone(),
                    line: PlayerGame::from_row(row),
                });
        }
        for games in player_season_games.values_mut() {
            games.sort_by(|left, right| {
                (left.week, &left.game_id).cmp(&(right.week, &right.game_id))
            });
        }

        let mut seasons: Vec<u16> = rows.iter().map(|row| row.season).collect();
        seasons.sort_unstable();
        seasons.dedup();

        Self {
            rows,
            schedule,
            seasons,
            counts,
            by_game,
            team_season_rows,
            player_season_games,
        }
    }

    /// The (home, away) rows of one game, when both are present.
    pub fn game_rows(&self, game_id: &str) -> Option<(&TeamGame, &TeamGame)> {
        let pair = self.by_game.get(game_id)?;
        Some((&self.rows[pair.home?], &self.rows[pair.away?]))
    }

    /// One side of one game. The team code may be historical or current.
    pub fn row(&self, game_id: &str, team: &str) -> Option<&TeamGame> {
        let pair = self.by_game.get(game_id)?;
        let canonical = canonical_team(team);
        [pair.home, pair.away]
            .into_iter()
            .flatten()
            .map(|index| &self.rows[index])
            .find(|row| row.team == canonical)
    }

    /// The ordered `(game_id, value)` series for one subject-season and one
    /// stat — the input to the leave-one-out standardisation of plan §1.2.
    ///
    /// For a [`SubjectKey::Player`] this is **every regular-season game the
    /// player appeared in that season**, not only the games in which he was
    /// the resolved Qb / Wr1 / Rb1: §2.1 standardises a player against his own
    /// season, and a backup week is still one of his games. For a
    /// [`SubjectKey::Team`] it is every regular-season game of that
    /// team-season. Empty when the key and the stat disagree about whether the
    /// subject is a player or a team, or when the subject is unknown.
    pub fn series(&self, key: &SubjectKey, stat: Stat) -> Vec<(String, f64)> {
        match key {
            SubjectKey::Team(team, season) if stat.is_team_stat() => self
                .team_season_rows
                .get(&(team.clone(), *season))
                .into_iter()
                .flatten()
                .filter_map(|&index| {
                    let row = &self.rows[index];
                    row.stat(stat).map(|value| (row.game_id.clone(), value))
                })
                .collect(),
            SubjectKey::Player(id, season) if !stat.is_team_stat() => self
                .player_season_games
                .get(&(id.clone(), *season))
                .into_iter()
                .flatten()
                .filter_map(|game| {
                    stat.of_player(&game.line)
                        .map(|value| (game.game_id.clone(), value))
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Every subject-season the table resolved for a stat, ascending. Phase 3
    /// walks this to build the standardised pool.
    pub fn subjects(&self, stat: Stat) -> Vec<SubjectKey> {
        if stat.is_team_stat() {
            self.team_season_rows
                .keys()
                .map(|(team, season)| SubjectKey::Team(team.clone(), *season))
                .collect()
        } else {
            let mut keys: BTreeSet<SubjectKey> = BTreeSet::new();
            for row in &self.rows {
                if let Some(subject) = row.subject(stat) {
                    keys.insert(subject);
                }
            }
            keys.into_iter().collect()
        }
    }
}

/// A subject the as-of rule names for a game that has **not been played**.
///
/// Deliberately not a [`PlayerGame`]. A `PlayerGame` is a *line* from a
/// completed game; a struct that looked like one but was full of zeroes is
/// exactly the sort of thing that eventually gets used as one.
#[derive(Debug, Clone, PartialEq)]
pub struct NamedSubject {
    pub player_id: String,
    pub display_name: String,
    pub position: String,
    /// Targets (receiver) or carries (rusher) over the lookback window.
    pub window_total: f64,
}

/// What the as-of rules say about a team **before its next game**.
///
/// The covariates here are the same two `TeamGame` carries
/// ([`TeamGame::wr1_target_share_asof`], [`TeamGame::pass_td_share_asof`]),
/// computed by the same code — see [`pre_game`].
#[derive(Debug, Clone, PartialEq)]
pub struct PreGame {
    pub team: String,
    pub season: u16,
    /// Team games already played in `season`. Zero in week 1, when the lookback
    /// is the prior season entire.
    pub games_played: usize,
    /// How many games the lookback window actually spans.
    pub window_games: usize,
    pub wr1: Option<NamedSubject>,
    pub rb1: Option<NamedSubject>,
    pub wr1_target_share_asof: Option<f64>,
    pub pass_td_share_asof: Option<f64>,
}

/// The pre-game state of every team for `season`, from the same weekly files
/// [`GameTable`] reads and by the same rules.
///
/// This closes the one structural gap in the crate: [`GameTable::build`] only
/// emits a row for a game that already has a final score, so nothing could say
/// what was true *before* an upcoming game — even though the inputs were on
/// disk and the rules already handle it. Plan §2.1's lookback in week 1 is "the
/// prior season entire", and that is exactly what this returns.
///
/// **It computes nothing itself.** It positions a [`RowContext`] one slot past
/// the team's last played game and asks the existing rules, so the answer here
/// and the answer on a completed row cannot diverge. The test
/// `pre_game_matches_the_row_the_table_would_have_built` is the proof, and is
/// the whole justification for this design over a second aggregator.
///
/// Quarterbacks are absent on purpose. Plan §2.1 identifies the starter from
/// the schedule's `home_qb_id`/`away_qb_id` confirmed against realised
/// attempts, and pre-game neither exists — the 2026 schedule rows carry a
/// spread and a total but empty QB ids. METHOD.md's 2026-09-08 amendment
/// settles that the *book* names the quarterback pre-game.
pub fn pre_game(
    players: Vec<PlayerWeekRow>,
    team_weeks: Vec<TeamWeekRow>,
    season: u16,
) -> BTreeMap<String, PreGame> {
    let players: Vec<PlayerWeekRow> = players
        .into_iter()
        .filter(PlayerWeekRow::is_regular_season)
        .collect();
    let team_weeks: Vec<TeamWeekRow> = team_weeks
        .into_iter()
        .filter(TeamWeekRow::is_regular_season)
        .collect();
    let indexes = Indexes::new(&players, &team_weeks);

    // A player's display name, for naming the subject the rule chose.
    let mut display: HashMap<&str, (&str, &str)> = HashMap::new();
    for row in &players {
        display.insert(
            row.player_id.as_str(),
            (row.player_display_name.as_str(), row.position.as_str()),
        );
    }

    let mut out = BTreeMap::new();
    for (team, slots) in &indexes.team_slots {
        // One past the last played game: "the team's next game", whose season
        // is supplied rather than read off a row that does not exist.
        let context = RowContext {
            team,
            slots,
            position: slots.len(),
            season,
            players: &players,
            players_by_game: &indexes.players_by_game,
            team_weeks: &team_weeks,
            season_totals: &indexes.season_totals,
        };
        let named = |skill: Skill| -> (Option<NamedSubject>, Option<f64>) {
            match context.window_leader(skill) {
                Some((player_id, window_total, share)) => {
                    let (display_name, position) =
                        display.get(player_id).copied().unwrap_or((player_id, ""));
                    (
                        Some(NamedSubject {
                            player_id: player_id.to_string(),
                            display_name: display_name.to_string(),
                            position: position.to_string(),
                            window_total,
                        }),
                        share,
                    )
                }
                None => (None, None),
            }
        };
        let (wr1, wr1_target_share_asof) = named(Skill::Receiver);
        let (rb1, _) = named(Skill::Rusher);
        out.insert(
            (*team).to_string(),
            PreGame {
                team: (*team).to_string(),
                season,
                games_played: slots.iter().filter(|slot| slot.season == season).count(),
                window_games: context.window().len(),
                wr1,
                rb1,
                wr1_target_share_asof,
                pass_td_share_asof: context.pass_td_share(),
            },
        );
    }
    out
}

/// The three lookup tables every as-of rule reads, built once from the weekly
/// files.
///
/// Extracted from [`GameTable::build`] so [`pre_game`] can build the identical
/// tables rather than a second set that would drift from them. Nothing here is
/// a rule; these are indexes over the raw rows.
struct Indexes<'a> {
    players_by_game: HashMap<&'a str, Vec<u32>>,
    season_totals: BTreeMap<(&'a str, &'a str, u16), (f64, f64)>,
    team_slots: BTreeMap<&'a str, Vec<TeamSlot<'a>>>,
}

impl<'a> Indexes<'a> {
    fn new(players: &'a [PlayerWeekRow], team_weeks: &'a [TeamWeekRow]) -> Self {
        // Player rows by game. Every lookback filters this by canonical team,
        // which is cheap: a game carries about sixty player rows.
        let mut players_by_game: HashMap<&str, Vec<u32>> = HashMap::new();
        for (index, row) in players.iter().enumerate() {
            players_by_game
                .entry(row.game_id.as_str())
                .or_default()
                .push(index as u32);
        }

        // Full-season targets and carries per (player, team, season): the
        // tie-break of plan §2.1, "ties → higher prior-season targets".
        let mut season_totals: BTreeMap<(&str, &str, u16), (f64, f64)> = BTreeMap::new();
        for row in players {
            let entry = season_totals
                .entry((
                    row.player_id.as_str(),
                    canonical_team(&row.team),
                    row.season,
                ))
                .or_default();
            entry.0 += row.tgts();
            entry.1 += row.rush_atts();
        }

        // Each team's games in order. The lookback window is then a slice of
        // this, so "the previous eight team games, crossing into the prior
        // season to fill" needs no special case beyond week 1's.
        let mut team_slots: BTreeMap<&str, Vec<TeamSlot>> = BTreeMap::new();
        for (index, row) in team_weeks.iter().enumerate() {
            team_slots
                .entry(canonical_team(&row.team))
                .or_default()
                .push(TeamSlot {
                    season: row.season,
                    week: row.week,
                    game_id: row.game_id.as_str(),
                    team_week: index,
                });
        }
        for slots in team_slots.values_mut() {
            slots.sort_by_key(|slot| (slot.season, slot.week, slot.game_id));
        }

        Self {
            players_by_game,
            season_totals,
            team_slots,
        }
    }
}

/// One team's slot in its own ordered schedule.
struct TeamSlot<'a> {
    season: u16,
    week: u8,
    game_id: &'a str,
    team_week: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QbSource {
    Schedule,
    Fallback,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Skill {
    Receiver,
    Rusher,
}

impl Skill {
    /// The positions eligible for the role. Plan §2.1 puts tight ends in the
    /// Wr1 pool: books list "receiving yards" on a tight end, and a team whose
    /// leading target is a tight end has no separate Wr1 worth pricing.
    fn positions(self) -> &'static [&'static str] {
        match self {
            Self::Receiver => &["WR", "TE"],
            Self::Rusher => &["RB"],
        }
    }
}

/// What the lookback produced for one skill slot.
struct Resolved {
    /// The as-of subject's line in this game, when he dressed.
    player: Option<PlayerGame>,
    /// Whether a subject was named at all.
    named: bool,
    /// The named subject's share of the team's window targets (receivers only).
    share: Option<f64>,
}

fn tally(counts: &mut ResolutionCounts, skill: Skill, resolved: &Resolved) {
    let (no_history, dropped, resolved_count) = match skill {
        Skill::Receiver => (
            &mut counts.wr1_no_history,
            &mut counts.wr1_dropped_inactive,
            &mut counts.wr1_resolved,
        ),
        Skill::Rusher => (
            &mut counts.rb1_no_history,
            &mut counts.rb1_dropped_inactive,
            &mut counts.rb1_resolved,
        ),
    };
    if !resolved.named {
        *no_history += 1;
    } else if resolved.player.is_none() {
        *dropped += 1;
    } else {
        *resolved_count += 1;
    }
}

/// Everything the three resolution rules need for one (game, team), bundled so
/// each rule reads as its own function rather than a nine-argument call.
struct RowContext<'a> {
    team: &'a str,
    slots: &'a [TeamSlot<'a>],
    /// Index of the game being resolved. **May equal `slots.len()`**, meaning
    /// "the team's next, unplayed game" — which is what [`pre_game`] passes.
    /// Every read of `slots[position]` has been replaced by a read of
    /// [`RowContext::season`] or a checked `get`, so the out-of-range value is
    /// legal by construction rather than by luck.
    position: usize,
    /// The season the resolved game belongs to. Carried separately precisely
    /// so `position == slots.len()` is answerable: the lookback rules need to
    /// know which season "now" is, and with no played game to read it from it
    /// has to be supplied.
    season: u16,
    players: &'a [PlayerWeekRow],
    players_by_game: &'a HashMap<&'a str, Vec<u32>>,
    team_weeks: &'a [TeamWeekRow],
    season_totals: &'a BTreeMap<(&'a str, &'a str, u16), (f64, f64)>,
}

impl<'a> RowContext<'a> {
    /// This team's player rows in this game — empty for an unplayed one, which
    /// is exactly right: nobody has a line in a game that has not happened, and
    /// the "did he dress?" check that consumes this must therefore be skipped
    /// pre-game rather than answered `no`.
    fn game_rows(&self) -> impl Iterator<Item = &'a PlayerWeekRow> {
        self.rows_of(
            self.slots
                .get(self.position)
                .map_or("", |slot| slot.game_id),
        )
    }

    fn rows_of(&self, game_id: &str) -> impl Iterator<Item = &'a PlayerWeekRow> {
        let team = self.team;
        let players = self.players;
        self.players_by_game
            .get(game_id)
            .into_iter()
            .flatten()
            .map(move |&index| &players[index as usize])
            .filter(move |row| canonical_team(&row.team) == team)
    }

    /// The lookback window: the previous [`SUBJECT_WINDOW_GAMES`] team games,
    /// crossing into the prior season to fill; in the team's **first game of a
    /// season**, the prior season entire (plan §2.1's "week 1 uses the prior
    /// season's full total").
    ///
    /// The two cases genuinely differ, which is why the special case exists
    /// rather than being collapsed away: the last eight games of a season
    /// weight the receiver who emerged in December, the full season weights
    /// the one who held the role all year. The plan chose the full season for
    /// week 1 and this honours it.
    fn window(&self) -> &'a [TeamSlot<'a>] {
        let before = &self.slots[..self.position];
        let season_to_date = before
            .iter()
            .filter(|slot| slot.season == self.season)
            .count();
        if season_to_date == 0 {
            let start = before
                .iter()
                .position(|slot| slot.season + 1 == self.season)
                .unwrap_or(before.len());
            &before[start..]
        } else {
            &before[before.len().saturating_sub(SUBJECT_WINDOW_GAMES)..]
        }
    }

    /// Plan §2.1's starting quarterback: the schedule's named starter if he
    /// really threw, else the team's most-attempting quarterback.
    ///
    /// A schedule id that *is* the most-attempting quarterback but threw fewer
    /// than [`MIN_QB_ATTEMPTS`] passes still counts as a fallback: the rule
    /// that fired is the fallback rule, and the count exists to say how often
    /// the schedule column was not enough on its own.
    fn resolve_qb(&self, scheduled: Option<&str>) -> (Option<PlayerGame>, QbSource) {
        if let Some(id) = scheduled
            && let Some(row) = self
                .game_rows()
                .find(|row| row.player_id == id && row.pass_atts() >= MIN_QB_ATTEMPTS)
        {
            return (Some(PlayerGame::from_row(row)), QbSource::Schedule);
        }
        // Strict `>` keeps the first row on a tie; the player rows of a game
        // arrive in file order, which is stable for a released season.
        let mut best: Option<&PlayerWeekRow> = None;
        for row in self.game_rows().filter(|row| row.position == "QB") {
            let better = match best {
                None => true,
                Some(current) => row.pass_atts() > current.pass_atts(),
            };
            if better {
                best = Some(row);
            }
        }
        match best {
            Some(row) => (Some(PlayerGame::from_row(row)), QbSource::Fallback),
            None => (None, QbSource::None),
        }
    }

    /// Plan §2.1's Wr1 / Rb1 leader over the lookback window, and his share of
    /// the team's window usage — **everything the rule decides except whether
    /// he dressed**.
    ///
    /// Split out of [`RowContext::resolve_skill`] so the pre-game path
    /// ([`pre_game`]) can ask the same question about a game that has not been
    /// played. The dressed check is the one clause that cannot exist before
    /// kickoff, and it is the only thing left behind in the caller. One
    /// definition of the rule, two callers — the alternative was a second
    /// implementation that would drift from this one and be scored against a
    /// stratum its history does not belong to.
    fn window_leader(&self, skill: Skill) -> Option<(&'a str, f64, Option<f64>)> {
        let window = self.window();
        if window.len() < MIN_SUBJECT_HISTORY_GAMES {
            return None;
        }
        let prior_season = self.season.saturating_sub(1);
        let mut totals: BTreeMap<&'a str, f64> = BTreeMap::new();
        for slot in window {
            for row in self.rows_of(slot.game_id) {
                if !skill.positions().contains(&row.position.as_str()) {
                    continue;
                }
                let value = match skill {
                    Skill::Receiver => row.tgts(),
                    Skill::Rusher => row.rush_atts(),
                };
                *totals.entry(row.player_id.as_str()).or_default() += value;
            }
        }
        // The map iterates player id ascending and the comparison is strict,
        // so the lowest id wins a tie the prior season could not break — the
        // "stable by player id" the ordering rule asks for.
        let mut chosen: Option<(&str, f64, f64)> = None;
        for (&player_id, &total) in &totals {
            // A player with no touches over eight games is not a depth chart's
            // answer to anything, and picking one would put a zero-usage
            // subject into the pool.
            if total <= 0.0 {
                continue;
            }
            let prior = self
                .season_totals
                .get(&(player_id, self.team, prior_season))
                .map_or(0.0, |&(targets, carries)| match skill {
                    Skill::Receiver => targets,
                    Skill::Rusher => carries,
                });
            let better = match chosen {
                None => true,
                Some((_, best_total, best_prior)) => {
                    total > best_total || (total == best_total && prior > best_prior)
                }
            };
            if better {
                chosen = Some((player_id, total, prior));
            }
        }
        let (player_id, leader_total, _) = chosen?;
        let share = match skill {
            Skill::Receiver => {
                let team_targets: f64 = window
                    .iter()
                    .map(|slot| self.team_weeks[slot.team_week].tgts())
                    .sum();
                (team_targets > 0.0).then(|| leader_total / team_targets)
            }
            Skill::Rusher => None,
        };
        Some((player_id, leader_total, share))
    }

    /// Plan §2.1's Wr1 / Rb1: the window leader, dropped rather than replaced
    /// when he did not dress.
    fn resolve_skill(&self, skill: Skill) -> Resolved {
        let Some((player_id, _, share)) = self.window_leader(skill) else {
            return Resolved {
                player: None,
                named: false,
                share: None,
            };
        };
        // The one line that must never become "or else the game's leader".
        let player = self
            .game_rows()
            .find(|row| row.player_id == player_id)
            .map(PlayerGame::from_row);
        Resolved {
            player,
            named: true,
            share,
        }
    }

    /// The team's passing share of its offensive touchdowns over the prior
    /// season plus the season to date (plan §2, `teamtotal_x_passtd`).
    ///
    /// Not the previous eight games: a passing-TD share is a coaching identity
    /// that moves on a timescale of seasons, and the plan asks for the longer,
    /// steadier window with a floor of [`MIN_PASS_TD_SHARE_GAMES`] games.
    fn pass_td_share(&self) -> Option<f64> {
        let window: Vec<&TeamSlot<'a>> = self.slots[..self.position]
            .iter()
            .filter(|slot| slot.season == self.season || slot.season + 1 == self.season)
            .collect();
        if window.len() < MIN_PASS_TD_SHARE_GAMES {
            return None;
        }
        let mut passing = 0.0;
        let mut rushing = 0.0;
        for slot in window {
            let row = &self.team_weeks[slot.team_week];
            passing += row.pass_tds();
            rushing += row.rush_tds();
        }
        let offensive = passing + rushing;
        (offensive > 0.0).then_some(passing / offensive)
    }
}

#[cfg(test)]
pub(crate) mod fixtures {
    //! A two-team, ten-week mini league with a prior season, in the real
    //! nflverse column names.
    //!
    //! The headers here are the real column *names* but not all ~150 real
    //! columns. The full-header parse — including the quoted `headshot_url`
    //! that contains a comma — is already pinned by the tests in
    //! [`crate::nflverse`]; serde ignores unknown columns and defaults the ones
    //! omitted here; and a fixture whose rows were 150 fields wide could not be
    //! checked by eye, which would defeat the point of a fixture. Every column
    //! any rule in this module consults is present and spelled as nflverse
    //! spells it.

    use super::*;

    pub const PLAYER_HEADER: &str = "player_id,player_name,player_display_name,position,season,week,season_type,game_id,team,opponent_team,attempts,passing_yards,passing_tds,carries,rushing_yards,rushing_tds,receptions,targets,receiving_yards";
    pub const TEAM_HEADER: &str = "season,week,team,season_type,game_id,opponent_team,attempts,passing_yards,passing_tds,carries,rushing_yards,rushing_tds,receptions,targets,receiving_yards";
    pub const GAMES_HEADER: &str = "game_id,season,game_type,week,gameday,weekday,away_team,away_score,home_team,home_score,result,total,spread_line,total_line,away_qb_id,home_qb_id,away_qb_name,home_qb_name";

    /// KC hosts in odd weeks, so both sides of the `anchor_spread` sign appear
    /// for both teams.
    pub const HOME_WEEKS: &[u8] = &[1, 3, 5, 7, 9];
    pub const WEEKS: u8 = 10;
    pub const KC_QB: &str = "00-0000001";
    pub const KC_WR1: &str = "00-0000002";
    pub const KC_WR2: &str = "00-0000003";
    pub const KC_TE: &str = "00-0000004";
    pub const KC_RB1: &str = "00-0000005";
    pub const KC_RB2: &str = "00-0000006";
    pub const KC_QB2: &str = "00-0000007";
    pub const BUF_QB: &str = "00-0000011";
    pub const BUF_WR1: &str = "00-0000012";
    pub const BUF_RB1: &str = "00-0000013";
    /// The tight end's targets every week, and KC's Rb1's — both fixed, so the
    /// share arithmetic in the tests is closed-form.
    pub const KC_TE_TARGETS: f64 = 2.0;
    pub const KC_RB1_TARGETS: f64 = 3.0;

    pub fn game_id(season: u16, week: u8) -> String {
        if HOME_WEEKS.contains(&week) {
            format!("{season}_{week:02}_BUF_KC")
        } else {
            format!("{season}_{week:02}_KC_BUF")
        }
    }

    pub fn home_team(week: u8) -> &'static str {
        if HOME_WEEKS.contains(&week) {
            "KC"
        } else {
            "BUF"
        }
    }

    /// Scores and closing lines: arbitrary but fixed, and chosen so the spread
    /// changes sign across the season and neither side always covers.
    pub fn home_score(week: u8) -> i32 {
        20 + i32::from(week % 4)
    }
    pub fn away_score(week: u8) -> i32 {
        17 + i32::from(week % 3) * 3
    }
    pub fn spread_line(week: u8) -> f64 {
        3.0 - f64::from(week)
    }
    pub fn total_line(week: u8) -> f64 {
        40.0 + f64::from(week)
    }

    /// KC's receivers in the **prior** season are built so the two lookback
    /// rules disagree. Over the full 2023 season WR1 leads (10+10 then 3×8 =
    /// 44 against 4×8 = 32); over its last eight games WR2 leads (32 against
    /// 24). Week 1 of 2024 must therefore name WR1, and code that quietly used
    /// "the last eight" everywhere would name WR2 and fail
    /// `week_one_uses_the_prior_seasons_full_total`.
    pub fn kc_wr1_targets(season: u16, week: u8) -> f64 {
        if season == 2023 {
            if week <= 2 { 10.0 } else { 3.0 }
        } else {
            10.0
        }
    }
    pub fn kc_wr2_targets(season: u16, week: u8) -> f64 {
        if season == 2023 {
            if week <= 2 { 0.0 } else { 4.0 }
        } else if season == 2024 && week == 6 {
            // The leakage trap: WR2 explodes in this one game, and a pre-game
            // rule cannot see it.
            20.0
        } else {
            1.0
        }
    }
    pub fn kc_wr2_yards(season: u16, week: u8) -> f64 {
        if season == 2024 && week == 6 {
            220.0
        } else {
            kc_wr2_targets(season, week) * 8.0
        }
    }
    pub fn kc_rb2_carries(season: u16, week: u8) -> f64 {
        if season == 2024 && week == 6 {
            25.0
        } else {
            2.0
        }
    }
    pub fn kc_rb2_yards(season: u16, week: u8) -> f64 {
        if season == 2024 && week == 6 {
            190.0
        } else {
            kc_rb2_carries(season, week) * 4.0
        }
    }

    /// KC's starter throws 30 times except in 2024 week 4, where he leaves
    /// early (5 attempts) and the backup takes over (22). The schedule still
    /// names the starter, so that week is the fallback case.
    pub fn kc_qb_attempts(season: u16, week: u8) -> f64 {
        if season == 2024 && week == 4 {
            5.0
        } else {
            30.0
        }
    }
    pub fn kc_qb2_attempts(season: u16, week: u8) -> f64 {
        if season == 2024 && week == 4 {
            22.0
        } else {
            0.0
        }
    }
    pub fn kc_qb_tds(week: u8) -> f64 {
        f64::from(week % 3)
    }
    pub fn kc_rb1_tds(week: u8) -> f64 {
        f64::from(week % 2)
    }

    /// 2024 week 8 is the dropped-not-replaced case: KC's as-of Wr1 and Rb1 are
    /// both absent from the player rows entirely, as an inactive player is.
    pub fn kc_wr1_active(season: u16, week: u8) -> bool {
        !(season == 2024 && week == 8)
    }
    pub fn kc_rb1_active(season: u16, week: u8) -> bool {
        !(season == 2024 && week == 8)
    }

    struct Line {
        id: &'static str,
        name: &'static str,
        position: &'static str,
        attempts: f64,
        passing_yards: f64,
        passing_tds: f64,
        carries: f64,
        rushing_yards: f64,
        rushing_tds: f64,
        receptions: f64,
        targets: f64,
        receiving_yards: f64,
    }

    impl Line {
        fn blank(id: &'static str, name: &'static str, position: &'static str) -> Self {
            Self {
                id,
                name,
                position,
                attempts: 0.0,
                passing_yards: 0.0,
                passing_tds: 0.0,
                carries: 0.0,
                rushing_yards: 0.0,
                rushing_tds: 0.0,
                receptions: 0.0,
                targets: 0.0,
                receiving_yards: 0.0,
            }
        }
    }

    fn kc_lines(season: u16, week: u8) -> Vec<Line> {
        let mut lines = Vec::new();
        let mut starter = Line::blank(KC_QB, "Pat Passer", "QB");
        starter.attempts = kc_qb_attempts(season, week);
        starter.passing_yards = starter.attempts * 8.0;
        starter.passing_tds = kc_qb_tds(week);
        lines.push(starter);
        if kc_qb2_attempts(season, week) > 0.0 {
            let mut backup = Line::blank(KC_QB2, "Ben Backup", "QB");
            backup.attempts = kc_qb2_attempts(season, week);
            backup.passing_yards = backup.attempts * 6.0;
            backup.passing_tds = 1.0;
            lines.push(backup);
        }
        if kc_wr1_active(season, week) {
            let mut wr1 = Line::blank(KC_WR1, "Will Wideout", "WR");
            wr1.targets = kc_wr1_targets(season, week);
            wr1.receptions = (wr1.targets * 0.6).round();
            wr1.receiving_yards = wr1.targets * 9.0;
            lines.push(wr1);
        }
        let mut wr2 = Line::blank(KC_WR2, "Larry Late", "WR");
        wr2.targets = kc_wr2_targets(season, week);
        wr2.receptions = (wr2.targets * 0.6).round();
        wr2.receiving_yards = kc_wr2_yards(season, week);
        lines.push(wr2);
        let mut tight_end = Line::blank(KC_TE, "Ted End", "TE");
        tight_end.targets = KC_TE_TARGETS;
        tight_end.receptions = 1.0;
        tight_end.receiving_yards = 14.0;
        lines.push(tight_end);
        if kc_rb1_active(season, week) {
            let mut rb1 = Line::blank(KC_RB1, "Rick Runner", "RB");
            rb1.carries = 18.0;
            rb1.rushing_yards = 84.0;
            rb1.rushing_tds = kc_rb1_tds(week);
            rb1.targets = KC_RB1_TARGETS;
            rb1.receptions = 2.0;
            rb1.receiving_yards = 12.0;
            lines.push(rb1);
        }
        let mut rb2 = Line::blank(KC_RB2, "Sam Sub", "RB");
        rb2.carries = kc_rb2_carries(season, week);
        rb2.rushing_yards = kc_rb2_yards(season, week);
        lines.push(rb2);
        lines
    }

    fn buf_lines(week: u8) -> Vec<Line> {
        let mut quarterback = Line::blank(BUF_QB, "Joe Thrower", "QB");
        quarterback.attempts = 28.0;
        quarterback.passing_yards = 230.0 + f64::from(week);
        quarterback.passing_tds = 2.0;
        let mut receiver = Line::blank(BUF_WR1, "Stan Stretch", "WR");
        receiver.targets = 9.0;
        receiver.receptions = 6.0;
        receiver.receiving_yards = 88.0;
        let mut back = Line::blank(BUF_RB1, "Dan Dash", "RB");
        back.carries = 15.0;
        back.rushing_yards = 70.0;
        back.rushing_tds = 1.0;
        back.targets = 2.0;
        back.receptions = 2.0;
        back.receiving_yards = 10.0;
        vec![quarterback, receiver, back]
    }

    fn lines_for(team: &str, season: u16, week: u8) -> Vec<Line> {
        if team == "KC" {
            kc_lines(season, week)
        } else {
            buf_lines(week)
        }
    }

    /// The three CSV texts for `seasons`, weeks 1..=[`WEEKS`].
    pub fn csvs(seasons: &[u16]) -> (String, String, String) {
        let mut players = format!("{PLAYER_HEADER}\n");
        let mut teams = format!("{TEAM_HEADER}\n");
        let mut games = format!("{GAMES_HEADER}\n");
        for &season in seasons {
            for week in 1..=WEEKS {
                let id = game_id(season, week);
                let home = home_team(week);
                let away = if home == "KC" { "BUF" } else { "KC" };
                let home_qb = if home == "KC" { KC_QB } else { BUF_QB };
                let away_qb = if away == "KC" { KC_QB } else { BUF_QB };
                games.push_str(&format!(
                    "{id},{season},REG,{week},2024-09-08,Sunday,{away},{},{home},{},{},{},{},{},{away_qb},{home_qb},Away QB,Home QB\n",
                    away_score(week),
                    home_score(week),
                    home_score(week) - away_score(week),
                    home_score(week) + away_score(week),
                    spread_line(week),
                    total_line(week),
                ));
                for team in ["KC", "BUF"] {
                    let opponent = if team == "KC" { "BUF" } else { "KC" };
                    let lines = lines_for(team, season, week);
                    let mut totals = Line::blank("", "", "");
                    for line in &lines {
                        players.push_str(&format!(
                            "{},{},{},{},{season},{week},REG,{id},{team},{opponent},{},{},{},{},{},{},{},{},{}\n",
                            line.id,
                            line.name,
                            line.name,
                            line.position,
                            line.attempts,
                            line.passing_yards,
                            line.passing_tds,
                            line.carries,
                            line.rushing_yards,
                            line.rushing_tds,
                            line.receptions,
                            line.targets,
                            line.receiving_yards,
                        ));
                        totals.attempts += line.attempts;
                        totals.passing_yards += line.passing_yards;
                        totals.passing_tds += line.passing_tds;
                        totals.carries += line.carries;
                        totals.rushing_yards += line.rushing_yards;
                        totals.rushing_tds += line.rushing_tds;
                        totals.receptions += line.receptions;
                        totals.targets += line.targets;
                        totals.receiving_yards += line.receiving_yards;
                    }
                    teams.push_str(&format!(
                        "{season},{week},{team},REG,{id},{opponent},{},{},{},{},{},{},{},{},{}\n",
                        totals.attempts,
                        totals.passing_yards,
                        totals.passing_tds,
                        totals.carries,
                        totals.rushing_yards,
                        totals.rushing_tds,
                        totals.receptions,
                        totals.targets,
                        totals.receiving_yards,
                    ));
                }
            }
        }
        (players, teams, games)
    }

    pub fn parse<T: serde::de::DeserializeOwned>(text: &str) -> Vec<T> {
        csv::Reader::from_reader(text.as_bytes())
            .deserialize()
            .collect::<std::result::Result<Vec<T>, _>>()
            .expect("fixture parses")
    }

    /// The standard table: 2023 as history, 2024 as the analysis season.
    pub fn mini_table() -> GameTable {
        table_for(&[2023, 2024])
    }

    pub fn table_for(seasons: &[u16]) -> GameTable {
        let (players, teams, games) = csvs(seasons);
        GameTable::build(parse(&games), parse(&players), parse(&teams))
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::*;
    use super::*;

    fn kc(table: &GameTable, season: u16, week: u8) -> &TeamGame {
        table
            .row(&game_id(season, week), "KC")
            .expect("KC row exists")
    }

    /// One player's raw fixture row for one game, straight out of the CSV the
    /// builder was handed.
    ///
    /// The leakage tests need this rather than [`GameTable::series`], because
    /// a player who is never a resolved subject is deliberately absent from
    /// the table's standardisation index — and the whole point of those tests
    /// is that the exploding non-subject *was* in the input and was still not
    /// chosen.
    fn fixture_row(seasons: &[u16], player_id: &str, game: &str) -> PlayerWeekRow {
        let (players, _, _) = csvs(seasons);
        parse::<PlayerWeekRow>(&players)
            .into_iter()
            .find(|row| row.player_id == player_id && row.game_id == game)
            .unwrap_or_else(|| panic!("{player_id} has a row in {game}"))
    }

    #[test]
    fn fnv_matches_the_published_reference_vectors() {
        // From the FNV reference implementation's own test suite. Pinning
        // against the standard, not against our output, is what makes the
        // cluster ids reproducible on a toolchain nobody has shipped yet.
        assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a_64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a_64(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn cluster_ids_are_stable_constants() {
        assert_eq!(
            SubjectKey::Team("KC".into(), 2024).canonical_string(),
            "team:KC:2024"
        );
        assert_eq!(
            SubjectKey::Player("00-0033873".into(), 2024).canonical_string(),
            "player:00-0033873:2024"
        );
        // Pinned: if either number ever changes, every atlas written before
        // the change has silently re-clustered.
        assert_eq!(
            SubjectKey::Team("KC".into(), 2024).cluster_id(),
            0x1c25_ae84_2bd8_d6e2
        );
        assert_eq!(
            SubjectKey::Player("00-0033873".into(), 2024).cluster_id(),
            0x21e2_9d41_48fa_f983
        );
        assert_ne!(
            SubjectKey::Team("KC".into(), 2024).cluster_id(),
            SubjectKey::Team("KC".into(), 2023).cluster_id()
        );
        assert_eq!(SubjectKey::Team("KC".into(), 2024).season(), 2024);
    }

    /// **The test that justifies reusing `RowContext` instead of writing a
    /// second aggregator.**
    ///
    /// Build the table over both seasons and read what it computed for each
    /// team's *first game of the later season*. Then ask `pre_game` the same
    /// question using only the earlier season's rows. The two must agree
    /// exactly — same named Wr1, same target share, same pass-TD share.
    ///
    /// If they ever diverge, a live row is priced against a stratum its own
    /// history would not have put it in, and nothing would report an error.
    #[test]
    fn pre_game_matches_the_row_the_table_would_have_built() {
        let table = fixtures::table_for(&[2023, 2024]);
        let (players, team_weeks, _) = fixtures::csvs(&[2023]);
        let earlier_players: Vec<PlayerWeekRow> = fixtures::parse(&players);
        let earlier_team_weeks: Vec<TeamWeekRow> = fixtures::parse(&team_weeks);
        let ahead = pre_game(earlier_players, earlier_team_weeks, 2024);

        let mut compared = 0;
        for row in table.rows.iter().filter(|row| row.season == 2024) {
            // Only each team's first 2024 game: after that the window has
            // in-season games the pre-game view was never given.
            let first = table
                .rows
                .iter()
                .filter(|other| other.team == row.team && other.season == 2024)
                .map(|other| other.week)
                .min();
            if Some(row.week) != first {
                continue;
            }
            let view = ahead
                .get(&row.team)
                .unwrap_or_else(|| panic!("no pre-game view for {}", row.team));
            assert_eq!(
                view.wr1.as_ref().map(|subject| subject.player_id.as_str()),
                row.wr1.as_ref().map(|player| player.player_id.as_str()),
                "{} week {}: pre-game named a different Wr1",
                row.team,
                row.week
            );
            assert_eq!(
                view.wr1_target_share_asof, row.wr1_target_share_asof,
                "{} week {}: target share differs",
                row.team, row.week
            );
            assert_eq!(
                view.pass_td_share_asof, row.pass_td_share_asof,
                "{} week {}: pass-TD share differs",
                row.team, row.week
            );
            assert_eq!(view.games_played, 0, "week 1 has no games played yet");
            compared += 1;
        }
        assert!(compared > 0, "the fixture produced nothing to compare");
    }

    /// The dressed check is the one clause that cannot exist before kickoff,
    /// and it is correctly skipped: `pre_game` names a subject even though no
    /// game rows exist for the upcoming game.
    #[test]
    fn pre_game_names_a_subject_although_no_game_rows_exist() {
        let (players, team_weeks, _) = fixtures::csvs(&[2023]);
        let ahead = pre_game(
            fixtures::parse(&players),
            fixtures::parse(&team_weeks),
            2024,
        );
        assert!(!ahead.is_empty());
        assert!(
            ahead.values().any(|view| view.wr1.is_some()),
            "a full prior season must be enough to name a Wr1"
        );
        for view in ahead.values() {
            assert_eq!(view.season, 2024);
            if let Some(subject) = &view.wr1 {
                assert!(
                    !subject.display_name.is_empty(),
                    "the subject must be named"
                );
                assert!(
                    subject.window_total > 0.0,
                    "a zero-usage subject is not a pick"
                );
            }
        }
    }

    /// A team with too little history is refused, not guessed at.
    ///
    /// The case is a *gap*: 2023 rows, asked about 2025. Week 1's lookback is
    /// "the prior season entire", 2024 is missing, and a two-year-old season is
    /// not a substitute — so the window comes back empty and the subject is
    /// `None`. Asking about 2023 itself would **not** test this: after sixteen
    /// played games the window is correctly the last eight of that same season,
    /// which is the rule working, not failing.
    #[test]
    fn pre_game_refuses_a_team_with_too_little_history() {
        let (players, team_weeks, _) = fixtures::csvs(&[2023]);
        let ahead = pre_game(
            fixtures::parse(&players),
            fixtures::parse(&team_weeks),
            2025,
        );
        assert!(!ahead.is_empty(), "the teams should still be listed");
        for view in ahead.values() {
            assert_eq!(
                view.window_games, 0,
                "{}: the window should be empty",
                view.team
            );
            assert!(
                view.wr1.is_none(),
                "{}: named a Wr1 across a missing season",
                view.team
            );
            assert!(view.rb1.is_none());
            assert!(view.wr1_target_share_asof.is_none());
        }
    }

    /// And mid-season the window is the previous eight team games, not the
    /// prior season — the same rule a completed row follows.
    #[test]
    fn pre_game_mid_season_uses_the_previous_eight_games() {
        let (players, team_weeks, _) = fixtures::csvs(&[2023]);
        let ahead = pre_game(
            fixtures::parse(&players),
            fixtures::parse(&team_weeks),
            2023,
        );
        for view in ahead.values() {
            assert!(view.games_played > 0, "{} has played in 2023", view.team);
            assert_eq!(
                view.window_games,
                SUBJECT_WINDOW_GAMES.min(view.games_played),
                "{}: mid-season window should be the last eight games",
                view.team
            );
        }
    }

    /// The table must cover exactly the 32 franchises this crate knows, and
    /// round-trip. A missing or misspelled entry means a game the generator
    /// cannot attribute to a side — on a Sunday morning, with prices waiting.
    #[test]
    fn every_team_name_maps_to_a_code_and_back() {
        assert_eq!(TEAM_NAMES.len(), TEAMS.len());
        for (code, name) in TEAM_NAMES {
            assert!(TEAMS.contains(&code), "{code} is not a known franchise");
            assert_eq!(team_code(name), Some(code), "{name} did not map to {code}");
            assert_eq!(team_full_name(code), Some(name));
        }
        // Every code is covered, so no franchise is missing a name.
        for code in TEAMS {
            assert!(team_full_name(code).is_some(), "{code} has no display name");
        }
    }

    /// Books and feeds spell names loosely, and franchises get renamed. Both
    /// must fold onto the current code; anything genuinely unknown must refuse
    /// rather than guess, so a rename surfaces as a named error.
    #[test]
    fn team_names_tolerate_spelling_and_renames_but_refuse_strangers() {
        assert_eq!(team_code("kansas city chiefs"), Some("KC"));
        assert_eq!(team_code("  Kansas   City  Chiefs "), Some("KC"));
        assert_eq!(team_code("San Francisco 49ers"), Some("SF"));
        // Renames and relocations fold onto the current franchise.
        assert_eq!(team_code("Washington Football Team"), Some("WAS"));
        assert_eq!(team_code("Oakland Raiders"), Some("LV"));
        assert_eq!(team_code("San Diego Chargers"), Some("LAC"));
        assert_eq!(team_code("St. Louis Rams"), Some("LA"));
        // And the old code still canonicalises the same way.
        assert_eq!(team_full_name("OAK"), Some("Las Vegas Raiders"));
        // Not a guess.
        assert_eq!(team_code("Toronto Argonauts"), None);
        assert_eq!(team_code(""), None);
        assert_eq!(team_full_name("XXX"), None);
    }

    #[test]
    fn relocated_franchises_canonicalise_to_one_code() {
        assert_eq!(canonical_team("OAK"), "LV");
        assert_eq!(canonical_team("SD"), "LAC");
        assert_eq!(canonical_team("STL"), "LA");
        assert_eq!(canonical_team("KC"), "KC");
        // An unknown code passes through rather than vanishing.
        assert_eq!(canonical_team("XXX"), "XXX");
        assert_eq!(TEAMS.len(), 32);
        let mut sorted = TEAMS;
        sorted.sort_unstable();
        assert_eq!(sorted, TEAMS, "TEAMS is documented as sorted");
    }

    #[test]
    fn the_table_has_exactly_one_row_per_game_and_team() {
        let table = mini_table();
        assert_eq!(table.seasons, vec![2023, 2024]);
        assert_eq!(table.rows.len(), 2 * usize::from(WEEKS) * 2);
        assert_eq!(table.counts.team_games, table.rows.len());
        let mut seen: BTreeSet<(&str, &str)> = BTreeSet::new();
        for row in &table.rows {
            assert!(
                seen.insert((row.game_id.as_str(), row.team.as_str())),
                "each (game, team) appears exactly once"
            );
        }
        // Ordered by season, week, game id, team — the atlas relies on it.
        let keys: Vec<_> = table
            .rows
            .iter()
            .map(|row| (row.season, row.week, row.game_id.clone(), row.team.clone()))
            .collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
    }

    #[test]
    fn anchor_spread_flips_sign_for_the_away_side_and_margin_agrees() {
        let table = mini_table();
        for week in 1..=WEEKS {
            let id = game_id(2024, week);
            let (home, away) = table.game_rows(&id).expect("both sides present");
            assert_eq!(home.team, home_team(week));
            assert!(home.home);
            assert!(!away.home);
            assert_eq!(home.anchor_spread, Some(spread_line(week)));
            assert_eq!(away.anchor_spread, Some(-spread_line(week)));
            assert_eq!(home.total_line, away.total_line);
            assert_eq!(home.result_margin, home_score(week) - away_score(week));
            assert_eq!(away.result_margin, -home.result_margin);
            assert_eq!(home.points, home_score(week));
            assert_eq!(away.points, away_score(week));
            assert_eq!(home.opp_points, away.points);
            assert_eq!(home.opponent, away.team);
        }
    }

    /// The cover convention has to agree with the one already pinned in
    /// [`crate::nflverse`], or the atlas and the scorer read the spread
    /// differently and nobody notices until a verdict flips.
    #[test]
    fn home_cover_agrees_with_nflverse_home_covered() {
        let table = mini_table();
        let mut checked = 0;
        for game in &table.schedule {
            let Some((home, _)) = table.game_rows(&game.game_id) else {
                continue;
            };
            let (Some(spread), Some(covered)) = (home.anchor_spread, game.home_covered()) else {
                continue;
            };
            assert_eq!(
                f64::from(home.result_margin) > spread,
                covered,
                "{} disagrees on the cover",
                game.game_id
            );
            checked += 1;
        }
        assert!(checked >= 15, "the check actually ran on {checked} games");
    }

    #[test]
    fn the_starting_qb_comes_from_the_schedule_when_he_threw() {
        let table = mini_table();
        let row = kc(&table, 2024, 2);
        let qb = row.qb.as_ref().expect("a quarterback resolved");
        assert_eq!(qb.player_id, KC_QB);
        assert_eq!(qb.attempts, 30.0);
        assert_eq!(qb.display_name, "Pat Passer");
        assert_eq!(qb.position, "QB");
    }

    /// The schedule still names the starter in 2024 week 4, but he threw five
    /// passes; the rule falls back to the backup's 22.
    #[test]
    fn the_qb_falls_back_when_the_scheduled_starter_is_below_the_attempt_floor() {
        let table = mini_table();
        let row = kc(&table, 2024, 4);
        let qb = row.qb.as_ref().expect("a quarterback resolved");
        assert_eq!(qb.player_id, KC_QB2, "the backup actually threw the ball");
        assert_eq!(qb.attempts, 22.0);
        // Exactly one fallback in the whole fixture, and it is this one.
        assert_eq!(table.counts.qb_fallback, 1);
        assert_eq!(table.counts.qb_unresolved, 0);
        assert_eq!(
            table.counts.qb_from_schedule + table.counts.qb_fallback,
            table.counts.team_games
        );
    }

    /// The leakage test. KC's WR2 catches 20 targets for 220 yards in 2024
    /// week 6 — more than WR1 has ever had in a game. Any rule that peeked at
    /// the box score would name him; the pre-game rule names WR1, whose window
    /// targets are far higher.
    #[test]
    fn the_as_of_wr1_ignores_the_game_the_other_receiver_explodes_in() {
        let table = mini_table();
        let row = kc(&table, 2024, 6);
        let wr1 = row.wr1.as_ref().expect("an as-of Wr1 resolved");
        assert_eq!(
            wr1.player_id, KC_WR1,
            "the pre-game leader, not the day's leader"
        );
        assert_ne!(wr1.player_id, KC_WR2);
        // …and the realised leader really did lead, on a row the builder read,
        // so this test fails against a "helpful" fallback to the game's top
        // receiver.
        let exploded = fixture_row(&[2023, 2024], KC_WR2, &game_id(2024, 6));
        assert_eq!(exploded.rec_yds(), 220.0);
        assert_eq!(exploded.tgts(), 20.0);
        assert!(exploded.rec_yds() > wr1.receiving_yards);
        assert!(exploded.tgts() > wr1.targets);
        assert_eq!(row.stat(Stat::Wr1ReceivingYards), Some(wr1.receiving_yards));
    }

    /// The same trap on the ground: RB2 takes 25 carries for 190 in 2024 week
    /// 6 and is still not the Rb1.
    #[test]
    fn the_as_of_rb1_ignores_the_game_the_other_back_explodes_in() {
        let table = mini_table();
        let row = kc(&table, 2024, 6);
        let rb1 = row.rb1.as_ref().expect("an as-of Rb1 resolved");
        assert_eq!(rb1.player_id, KC_RB1);
        assert_eq!(rb1.rushing_yards, 84.0);
        let exploded = fixture_row(&[2023, 2024], KC_RB2, &game_id(2024, 6));
        assert_eq!(exploded.rush_yds(), 190.0);
        assert!(exploded.rush_atts() > rb1.carries);
        assert!(exploded.rush_yds() > rb1.rushing_yards);
    }

    /// Dropped, never replaced: in 2024 week 8 the as-of Wr1 and Rb1 have no
    /// player rows at all, and the slots go empty rather than sliding down the
    /// depth chart to the man who did play.
    #[test]
    fn an_inactive_as_of_subject_drops_the_slot_instead_of_replacing_him() {
        let table = mini_table();
        let row = kc(&table, 2024, 8);
        assert!(row.wr1.is_none(), "no substitute Wr1");
        assert!(row.rb1.is_none(), "no substitute Rb1");
        assert_eq!(row.stat(Stat::Wr1ReceivingYards), None);
        assert_eq!(row.stat(Stat::Rb1RushingYards), None);
        assert_eq!(table.counts.wr1_dropped_inactive, 1);
        assert_eq!(table.counts.rb1_dropped_inactive, 1);
        // The substitutes were there and did play, so the empty slot is a
        // decision rather than an absence of candidates.
        let substitute = fixture_row(&[2023, 2024], KC_WR2, &game_id(2024, 8));
        assert!(substitute.tgts() > 0.0);
        assert!(fixture_row(&[2023, 2024], KC_RB2, &game_id(2024, 8)).rush_atts() > 0.0);
        // The pre-game share survives the drop: it describes the depth chart
        // the book priced, not the game that happened.
        assert!(row.wr1_target_share_asof.is_some());
        // Neighbouring weeks still resolve, so the drop is the fixture's doing
        // and not a broken lookback.
        assert!(kc(&table, 2024, 7).wr1.is_some());
        assert!(kc(&table, 2024, 9).wr1.is_some());
    }

    /// Week 1 reads the prior season entire. The fixture is built so the last
    /// eight games of 2023 favour WR2 while the full season favours WR1; the
    /// rule must name WR1.
    #[test]
    fn week_one_uses_the_prior_seasons_full_total() {
        let table = mini_table();
        let full_wr1: f64 = (1..=WEEKS).map(|week| kc_wr1_targets(2023, week)).sum();
        let full_wr2: f64 = (1..=WEEKS).map(|week| kc_wr2_targets(2023, week)).sum();
        let tail_wr1: f64 = (3..=WEEKS).map(|week| kc_wr1_targets(2023, week)).sum();
        let tail_wr2: f64 = (3..=WEEKS).map(|week| kc_wr2_targets(2023, week)).sum();
        assert!(full_wr1 > full_wr2, "the full season favours WR1");
        assert!(tail_wr2 > tail_wr1, "the last eight games favour WR2");

        let row = kc(&table, 2024, 1);
        assert_eq!(
            row.wr1.as_ref().map(|player| player.player_id.as_str()),
            Some(KC_WR1)
        );
        // The share is that full-season total over the team's own.
        let team_targets: f64 = (1..=WEEKS)
            .map(|week| {
                kc_wr1_targets(2023, week)
                    + kc_wr2_targets(2023, week)
                    + KC_TE_TARGETS
                    + KC_RB1_TARGETS
            })
            .sum();
        let share = row.wr1_target_share_asof.expect("a share");
        assert!((share - full_wr1 / team_targets).abs() < 1e-12);
    }

    /// With no prior season loaded, the first four weeks have no depth chart
    /// to read and the slots stay empty; the fifth has exactly the four games
    /// the floor requires.
    #[test]
    fn a_team_with_no_prior_season_has_no_subject_until_the_window_fills() {
        let table = table_for(&[2024]);
        for week in 1..=4 {
            let row = kc(&table, 2024, week);
            assert!(row.wr1.is_none(), "week {week} has no history");
            assert!(row.rb1.is_none(), "week {week} has no history");
            assert!(row.wr1_target_share_asof.is_none());
            // The quarterback rule needs no history, so he still resolves.
            assert!(row.qb.is_some());
        }
        assert_eq!(
            kc(&table, 2024, 5)
                .wr1
                .as_ref()
                .map(|player| player.player_id.as_str()),
            Some(KC_WR1)
        );
        assert_eq!(table.counts.wr1_no_history, 2 * 4, "both teams, four weeks");
        assert_eq!(table.counts.rb1_no_history, 2 * 4);
    }

    #[test]
    fn the_pass_td_share_needs_eight_games_and_reads_the_prior_season() {
        let with_history = mini_table();
        let row = kc(&with_history, 2024, 1);
        let passing: f64 = (1..=WEEKS).map(kc_qb_tds).sum();
        let rushing: f64 = (1..=WEEKS).map(kc_rb1_tds).sum();
        let share = row.pass_td_share_asof.expect("2023 supplies ten games");
        assert!((share - passing / (passing + rushing)).abs() < 1e-12);

        let alone = table_for(&[2024]);
        for week in 1..=8 {
            assert!(
                kc(&alone, 2024, week).pass_td_share_asof.is_none(),
                "week {week} has fewer than eight games behind it"
            );
        }
        assert!(kc(&alone, 2024, 9).pass_td_share_asof.is_some());
    }

    #[test]
    fn team_stats_come_from_the_team_week_file() {
        let table = mini_table();
        let row = kc(&table, 2024, 2);
        // KC's only passer that week threw for 30 × 8.
        assert_eq!(row.team_passing_yards, 240.0);
        assert_eq!(row.stat(Stat::TeamPassingYards), Some(240.0));
        assert_eq!(row.team_rushing_yards, 84.0 + kc_rb2_yards(2024, 2));
        assert_eq!(row.stat(Stat::TeamPoints), Some(f64::from(row.points)));
        assert_eq!(
            row.stat(Stat::MarginVsSpread),
            Some(f64::from(row.result_margin) - row.anchor_spread.expect("a spread"))
        );
        assert_eq!(
            row.stat(Stat::TotalVsLine),
            Some(f64::from(row.points + row.opp_points) - row.total_line.expect("a total"))
        );
    }

    /// A player's series is every game he played that season, not only the
    /// games in which he was the resolved subject — plan §2.1. KC's starter is
    /// not the resolved Qb in 2024 week 4 (he threw five passes) but that game
    /// is still in his standardisation pool.
    #[test]
    fn a_player_series_spans_every_game_he_played_whatever_his_role() {
        let table = mini_table();
        let series = table.series(
            &SubjectKey::Player(KC_QB.into(), 2024),
            Stat::QbPassingYards,
        );
        assert_eq!(series.len(), usize::from(WEEKS));
        let week_four = series
            .iter()
            .find(|(id, _)| id == &game_id(2024, 4))
            .expect("the benched week is in the pool");
        assert_eq!(week_four.1, 40.0);
        assert_ne!(
            kc(&table, 2024, 4)
                .qb
                .as_ref()
                .map(|player| player.player_id.as_str()),
            Some(KC_QB),
            "…even though he was not the resolved starter that week"
        );
        // Ordered by week: the fixture's ids sort the same way.
        let ids: Vec<&String> = series.iter().map(|(id, _)| id).collect();
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(ids, sorted);
    }

    #[test]
    fn a_team_series_spans_the_season_and_mismatched_keys_are_empty() {
        let table = mini_table();
        let series = table.series(&SubjectKey::Team("KC".into(), 2024), Stat::TeamPoints);
        assert_eq!(series.len(), usize::from(WEEKS));
        assert!(
            table
                .series(&SubjectKey::Team("KC".into(), 2024), Stat::QbPassingYards)
                .is_empty(),
            "a team key cannot answer a player stat"
        );
        assert!(
            table
                .series(&SubjectKey::Player(KC_QB.into(), 2024), Stat::TeamPoints)
                .is_empty()
        );
        assert!(
            table
                .series(&SubjectKey::Team("XXX".into(), 2024), Stat::TeamPoints)
                .is_empty()
        );
    }

    #[test]
    fn stat_flags_split_team_from_player_and_market_centred_from_raw() {
        assert!(Stat::TeamPoints.is_team_stat());
        assert!(Stat::MarginVsSpread.is_team_stat());
        assert!(!Stat::QbPassingYards.is_team_stat());
        assert!(Stat::MarginVsSpread.is_market_centred());
        assert!(Stat::TotalVsLine.is_market_centred());
        assert!(!Stat::TeamPoints.is_market_centred());
        assert_eq!(Stat::Wr1ReceivingYards.label(), "wr1_rec_yds");
    }

    /// A hand-checked real row, the first game in `games.csv`. MIN @ ATL 1999:
    /// the Vikings were favoured by 4 on the road (`spread_line` −4) and won by
    /// 3 (`result` −3), so the home residual is −3 − (−4) = **+1**; the game
    /// totalled 31 against a 49 line, so the total residual is **−18**.
    #[test]
    fn market_centred_values_match_a_hand_checked_game() {
        let text = format!(
            "{GAMES_HEADER}\n1999_01_MIN_ATL,1999,REG,1,1999-09-12,Sunday,MIN,17,ATL,14,-3,31,-4,49,,,,\n"
        );
        let games: Vec<GameRow> = parse(&text);
        let teams = format!(
            "{TEAM_HEADER}\n1999,1,ATL,REG,1999_01_MIN_ATL,MIN,30,240,1,25,90,1,18,30,240\n\
             1999,1,MIN,REG,1999_01_MIN_ATL,ATL,28,250,2,22,80,0,17,28,250\n"
        );
        let table = GameTable::build(games, Vec::new(), parse(&teams));
        let (home, away) = table.game_rows("1999_01_MIN_ATL").expect("both sides");
        assert_eq!(home.team, "ATL");
        assert_eq!(home.stat(Stat::MarginVsSpread), Some(1.0));
        assert_eq!(home.stat(Stat::TotalVsLine), Some(-18.0));
        // The away side sees the mirror image of the spread residual and the
        // same total residual.
        assert_eq!(away.stat(Stat::MarginVsSpread), Some(-1.0));
        assert_eq!(away.stat(Stat::TotalVsLine), Some(-18.0));
    }

    #[test]
    fn subjects_enumerates_resolved_player_seasons() {
        let table = mini_table();
        let quarterbacks = table.subjects(Stat::QbPassingYards);
        assert!(quarterbacks.contains(&SubjectKey::Player(KC_QB.into(), 2024)));
        assert!(quarterbacks.contains(&SubjectKey::Player(KC_QB2.into(), 2024)));
        assert!(!quarterbacks.contains(&SubjectKey::Player(KC_WR1.into(), 2024)));
        let receivers = table.subjects(Stat::Wr1ReceivingYards);
        assert!(receivers.contains(&SubjectKey::Player(BUF_WR1.into(), 2024)));
        assert!(!receivers.contains(&SubjectKey::Player(KC_WR2.into(), 2024)));
        let teams = table.subjects(Stat::TeamPoints);
        assert_eq!(teams.len(), 4, "two teams over two seasons");
    }

    /// The abbreviation trap, end to end: a schedule that says `SD` and stat
    /// files that say `LAC` must produce one team, reachable by either code.
    #[test]
    fn a_relocated_franchise_joins_across_the_two_spellings() {
        let players = format!(
            "{PLAYER_HEADER}\n00-0000021,J.Doe,Jane Doe,QB,2016,1,REG,2016_01_SD_KC,LAC,KC,25,240,2,0,0,0,0,0,0\n"
        );
        let teams =
            format!("{TEAM_HEADER}\n2016,1,LAC,REG,2016_01_SD_KC,KC,25,240,2,20,90,1,15,25,240\n");
        let games = format!(
            "{GAMES_HEADER}\n2016_01_SD_KC,2016,REG,1,2016-09-11,Sunday,SD,17,KC,20,3,37,3,44,00-0000021,00-0000099,A,B\n"
        );
        let table = GameTable::build(parse(&games), parse(&players), parse(&teams));
        let row = table.row("2016_01_SD_KC", "LAC").expect("the Chargers row");
        assert_eq!(row.team, "LAC");
        assert_eq!(row.opponent, "KC");
        assert!(!row.home, "SD was the away side");
        assert_eq!(row.anchor_spread, Some(-3.0));
        assert_eq!(row.points, 17);
        assert_eq!(row.result_margin, -3);
        // The schedule's away_qb_id resolves against a stat row that spells the
        // team the other way.
        assert_eq!(
            row.qb.as_ref().map(|player| player.player_id.as_str()),
            Some("00-0000021")
        );
        assert_eq!(table.counts.qb_from_schedule, 1);
        // The same row is reachable by the historical code.
        assert!(table.row("2016_01_SD_KC", "SD").is_some());
    }

    /// A team-week row whose game the schedule does not carry is counted and
    /// skipped, never given an invented score.
    #[test]
    fn a_game_missing_from_the_schedule_is_counted_not_invented() {
        let teams =
            format!("{TEAM_HEADER}\n2016,1,KC,REG,2016_01_SD_KC,LAC,25,240,2,20,90,1,15,25,240\n");
        let table = GameTable::build(Vec::new(), Vec::new(), parse(&teams));
        assert!(table.rows.is_empty());
        assert_eq!(table.counts.games_without_schedule, 1);
    }

    /// Post-season rows never reach the table (plan §2.2).
    #[test]
    fn post_season_rows_are_dropped_at_the_door() {
        let teams = format!(
            "{TEAM_HEADER}\n2024,1,KC,POST,2024_20_BUF_KC,BUF,25,240,2,20,90,1,15,25,240\n"
        );
        let games = format!(
            "{GAMES_HEADER}\n2024_20_BUF_KC,2024,CON,20,2025-01-26,Sunday,BUF,17,KC,20,3,37,3,44,,,,\n"
        );
        let table = GameTable::build(parse(&games), Vec::new(), parse(&teams));
        assert!(table.rows.is_empty());
        assert!(
            table.schedule.is_empty(),
            "a CON game is not regular season"
        );
    }
}
