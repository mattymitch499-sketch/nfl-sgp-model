//! The free outcome data: nflverse weekly player and team stats, and the
//! schedule with closing spread and total lines.
//!
//! This is the φ_true side of the study. Every candidate parlay family is a
//! pair of game-level outcomes — a quarterback's passing yards and his top
//! receiver's receiving yards, a team's points and its rusher's yards, both
//! quarterbacks' yards in one game — and every one of those numbers is in
//! these three files.
//!
//! ## Sources (all verified reachable 2026-09-02, no key, no login)
//!
//! | file | URL | what | size |
//! |---|---|---|---|
//! | `stats_player_week_{season}.csv` | nflverse-data release `stats_player` | one row per player-game: passing, rushing, receiving, defense, kicking | ~7.5–8.7 MB per season |
//! | `stats_team_week_{season}.csv` | nflverse-data release `stats_team` | one row per team-game, same columns aggregated | ~0.2 MB per season |
//! | `games.csv` | nflverse `schedules` release (mirror of nfldata `games.csv`) | every game 1999→now: scores, `spread_line`, `total_line`, moneylines, starting QBs, roof | ~2.1 MB, all seasons |
//!
//! Weekly files exist from 1999. `games.csv` is a single all-seasons file
//! that nflverse updates through the current season, so it is worth
//! re-downloading during the season while the per-season stat files for
//! completed seasons never change.
//!
//! ## Conventions that matter (from the nflverse data dictionary)
//!
//! * `games.csv`: `result = home_score − away_score`; `spread_line` is
//!   **positive when the home team is favoured** by that many points;
//!   `total = home_score + away_score`; `total_line` is the closing total.
//!   Sanity check on the first two 1999 rows: MIN @ ATL, `spread_line` −4
//!   (Vikings favoured on the road, coming off 15–1), `result` −3 (Falcons
//!   lost 14–17). The convention holds.
//! * Stat files use an empty field for "not applicable", not zero: a kicker's
//!   `passing_epa` is blank. Every numeric column is therefore an `Option`
//!   here, and the accessors treat blank as zero, which is right for counting
//!   stats and wrong for rate stats like `target_share` — which stays an
//!   `Option` on purpose.
//! * `player_name` is abbreviated (`A.Rodgers`); `player_display_name` is the
//!   full name the books use. Join through [`crate::names::normalize_name`]
//!   on the display name.
//! * `season_type` is `REG` or `POST`. Playoff games are a different
//!   population (no resting starters, tighter game scripts) — decide per
//!   family whether they belong in the atlas, and say so in the
//!   pre-registration.

use crate::{Error, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

pub const RELEASES: &str = "https://github.com/nflverse/nflverse-data/releases/download";
pub const FIRST_SEASON: u16 = 1999;

pub fn player_week_url(season: u16) -> String {
    format!("{RELEASES}/stats_player/stats_player_week_{season}.csv")
}

pub fn team_week_url(season: u16) -> String {
    format!("{RELEASES}/stats_team/stats_team_week_{season}.csv")
}

pub fn schedules_url() -> String {
    format!("{RELEASES}/schedules/games.csv")
}

/// Where a downloaded file lives under the raw-data directory.
pub fn player_week_path(dir: &Path, season: u16) -> PathBuf {
    dir.join(format!("stats_player_week_{season}.csv"))
}

pub fn team_week_path(dir: &Path, season: u16) -> PathBuf {
    dir.join(format!("stats_team_week_{season}.csv"))
}

pub fn schedules_path(dir: &Path) -> PathBuf {
    dir.join("games.csv")
}

/// [`download`], but a file the release does not carry yet is `Ok(None)`
/// rather than an error.
///
/// The case this exists for: early in a season the weekly stat files do not
/// exist — no games have been played — while the depth charts and the schedule
/// for that same season very much do. Aborting the whole fetch on the first
/// 404 makes it impossible to pull the charts you need for week 1.
///
/// Only a genuine "not published yet" is swallowed. Any other failure (network,
/// permissions, a 500) still propagates, so a typo in a URL does not quietly
/// become "that season is not out".
pub fn download_if_published(url: &str, dest: &Path) -> Result<Option<u64>> {
    match download(url, dest) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(Error::Data(message)) if message.contains("404") => Ok(None),
        Err(other) => Err(other),
    }
}

/// Official team depth charts for one season — see [`crate::depth`].
///
/// From 2025 these are timestamped snapshots that accumulate all season, so
/// the current season's file grows and must be re-fetched to stay current
/// (~48 MB for 2026). Seasons through 2024 are a different, per-week shape.
pub fn depth_charts_url(season: u16) -> String {
    format!("{RELEASES}/depth_charts/depth_charts_{season}.csv")
}

pub fn depth_charts_path(dir: &Path, season: u16) -> PathBuf {
    dir.join(format!("depth_charts_{season}.csv"))
}

/// Downloads `url` to `dest`, writing to a temporary sibling first so a
/// failed transfer never leaves a truncated CSV that parses as a short
/// season. Returns the byte count. Follows GitHub's release redirect.
pub fn download(url: &str, dest: &Path) -> Result<u64> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .user_agent("nfl-sgp/0.1")
        .build()?;
    let response = client.get(url).send()?;
    if !response.status().is_success() {
        return Err(Error::Data(format!(
            "download of {url} failed ({})",
            response.status()
        )));
    }
    let bytes = response.bytes()?;
    let temporary = dest.with_extension("part");
    std::fs::write(&temporary, &bytes)?;
    std::fs::rename(&temporary, dest)?;
    Ok(bytes.len() as u64)
}

/// One player's line in one game. Only the columns a parlay leg can be built
/// from are kept; the file carries ~150 and serde ignores the rest.
#[derive(Debug, Clone, Deserialize)]
pub struct PlayerWeekRow {
    pub player_id: String,
    /// Abbreviated, e.g. `A.Rodgers`. Not what a book prints.
    pub player_name: String,
    /// Full name, e.g. `Aaron Rodgers`. Join on this, folded.
    pub player_display_name: String,
    pub position: String,
    pub season: u16,
    pub week: u8,
    /// `REG` or `POST`.
    pub season_type: String,
    /// `{season}_{week:02}_{away}_{home}`, e.g. `2024_01_NYJ_SF`.
    pub game_id: String,
    pub team: String,
    pub opponent_team: String,
    #[serde(default)]
    pub completions: Option<f64>,
    #[serde(default)]
    pub attempts: Option<f64>,
    #[serde(default)]
    pub passing_yards: Option<f64>,
    #[serde(default)]
    pub passing_tds: Option<f64>,
    #[serde(default)]
    pub passing_interceptions: Option<f64>,
    #[serde(default)]
    pub sacks_suffered: Option<f64>,
    #[serde(default)]
    pub carries: Option<f64>,
    #[serde(default)]
    pub rushing_yards: Option<f64>,
    #[serde(default)]
    pub rushing_tds: Option<f64>,
    #[serde(default)]
    pub receptions: Option<f64>,
    #[serde(default)]
    pub targets: Option<f64>,
    #[serde(default)]
    pub receiving_yards: Option<f64>,
    #[serde(default)]
    pub receiving_tds: Option<f64>,
    /// Share of the team's targets this game. A rate, so blank stays `None`.
    /// This is the covariate the QB × receiver correlation is expected to
    /// depend on — see METHOD.md on stratifying φ.
    #[serde(default)]
    pub target_share: Option<f64>,
    #[serde(default)]
    pub def_sacks: Option<f64>,
    #[serde(default)]
    pub def_interceptions: Option<f64>,
    #[serde(default)]
    pub def_tackles_solo: Option<f64>,
    #[serde(default)]
    pub def_tackle_assists: Option<f64>,
    #[serde(default)]
    pub fg_made: Option<f64>,
    #[serde(default)]
    pub fg_att: Option<f64>,
    #[serde(default)]
    pub pat_made: Option<f64>,
}

impl PlayerWeekRow {
    /// Blank-as-zero for a counting stat.
    fn count(value: Option<f64>) -> f64 {
        value.unwrap_or(0.0)
    }
    pub fn pass_yds(&self) -> f64 {
        Self::count(self.passing_yards)
    }
    pub fn pass_tds(&self) -> f64 {
        Self::count(self.passing_tds)
    }
    pub fn rush_yds(&self) -> f64 {
        Self::count(self.rushing_yards)
    }
    pub fn rec_yds(&self) -> f64 {
        Self::count(self.receiving_yards)
    }
    pub fn recs(&self) -> f64 {
        Self::count(self.receptions)
    }
    /// Pass attempts. The starting-quarterback rule of `tasks/plan.md` §2.1
    /// confirms the schedule's named starter against this.
    pub fn pass_atts(&self) -> f64 {
        Self::count(self.attempts)
    }
    /// Rushing attempts (`carries`). The Rb1 lookback ranks on this.
    pub fn rush_atts(&self) -> f64 {
        Self::count(self.carries)
    }
    /// Targets. The Wr1 lookback ranks on this, and the target-share
    /// covariate is built from it. Blank means "did not run a route", which
    /// is zero targets — unlike `target_share`, which is a rate and stays
    /// `None`.
    pub fn tgts(&self) -> f64 {
        Self::count(self.targets)
    }
    /// Touchdowns from scrimmage — what an anytime-TD leg settles on.
    /// Return and defensive touchdowns are not in these columns and are rare
    /// enough for skill players that the approximation is documented rather
    /// than fixed.
    pub fn scrimmage_tds(&self) -> f64 {
        Self::count(self.rushing_tds) + Self::count(self.receiving_tds)
    }
    /// The Odds API's "tackles + assists" market settles on solo + assists.
    pub fn tackles_and_assists(&self) -> f64 {
        Self::count(self.def_tackles_solo) + Self::count(self.def_tackle_assists)
    }
    /// Kicking points as a kicker prop settles them: 3 per field goal, 1 per
    /// extra point.
    pub fn kicking_points(&self) -> f64 {
        3.0 * Self::count(self.fg_made) + Self::count(self.pat_made)
    }
    pub fn is_regular_season(&self) -> bool {
        self.season_type == "REG"
    }
}

/// One team's line in one game.
#[derive(Debug, Clone, Deserialize)]
pub struct TeamWeekRow {
    pub season: u16,
    pub week: u8,
    pub team: String,
    pub season_type: String,
    pub game_id: String,
    pub opponent_team: String,
    #[serde(default)]
    pub completions: Option<f64>,
    #[serde(default)]
    pub attempts: Option<f64>,
    #[serde(default)]
    pub passing_yards: Option<f64>,
    #[serde(default)]
    pub passing_tds: Option<f64>,
    #[serde(default)]
    pub passing_interceptions: Option<f64>,
    #[serde(default)]
    pub sacks_suffered: Option<f64>,
    #[serde(default)]
    pub carries: Option<f64>,
    #[serde(default)]
    pub rushing_yards: Option<f64>,
    #[serde(default)]
    pub rushing_tds: Option<f64>,
    /// The team's targets. The denominator of the Wr1 target-share covariate
    /// (`tasks/plan.md` §2.1): a receiver's share only means anything against
    /// the team total over the same window.
    #[serde(default)]
    pub targets: Option<f64>,
    #[serde(default)]
    pub def_sacks: Option<f64>,
    #[serde(default)]
    pub def_interceptions: Option<f64>,
    #[serde(default)]
    pub fg_made: Option<f64>,
    #[serde(default)]
    pub pat_made: Option<f64>,
}

impl TeamWeekRow {
    /// Blank-as-zero, exactly as [`PlayerWeekRow`] treats a counting stat.
    fn count(value: Option<f64>) -> f64 {
        value.unwrap_or(0.0)
    }
    pub fn pass_yds(&self) -> f64 {
        Self::count(self.passing_yards)
    }
    pub fn pass_tds(&self) -> f64 {
        Self::count(self.passing_tds)
    }
    pub fn rush_yds(&self) -> f64 {
        Self::count(self.rushing_yards)
    }
    pub fn rush_tds(&self) -> f64 {
        Self::count(self.rushing_tds)
    }
    pub fn tgts(&self) -> f64 {
        Self::count(self.targets)
    }
    pub fn is_regular_season(&self) -> bool {
        self.season_type == "REG"
    }
}

/// One game from the schedule file, with its closing lines.
#[derive(Debug, Clone, Deserialize)]
pub struct GameRow {
    pub game_id: String,
    pub season: u16,
    /// `REG`, `WC`, `DIV`, `CON`, `SB`.
    pub game_type: String,
    pub week: u8,
    /// `YYYY-MM-DD`.
    pub gameday: String,
    pub weekday: String,
    /// `HH:MM` Eastern, blank on old rows.
    #[serde(default)]
    pub gametime: Option<String>,
    pub away_team: String,
    #[serde(default)]
    pub away_score: Option<i32>,
    pub home_team: String,
    #[serde(default)]
    pub home_score: Option<i32>,
    /// `home_score − away_score`.
    #[serde(default)]
    pub result: Option<i32>,
    /// `home_score + away_score`.
    #[serde(default)]
    pub total: Option<i32>,
    #[serde(default)]
    pub overtime: Option<i32>,
    #[serde(default)]
    pub away_moneyline: Option<i32>,
    #[serde(default)]
    pub home_moneyline: Option<i32>,
    /// Closing spread, **positive when the home team is favoured**.
    #[serde(default)]
    pub spread_line: Option<f64>,
    #[serde(default)]
    pub away_spread_odds: Option<i32>,
    #[serde(default)]
    pub home_spread_odds: Option<i32>,
    /// Closing total.
    #[serde(default)]
    pub total_line: Option<f64>,
    #[serde(default)]
    pub under_odds: Option<i32>,
    #[serde(default)]
    pub over_odds: Option<i32>,
    #[serde(default)]
    pub div_game: Option<i32>,
    #[serde(default)]
    pub roof: Option<String>,
    #[serde(default)]
    pub surface: Option<String>,
    #[serde(default)]
    pub temp: Option<f64>,
    #[serde(default)]
    pub wind: Option<f64>,
    #[serde(default)]
    pub away_qb_id: Option<String>,
    #[serde(default)]
    pub home_qb_id: Option<String>,
    #[serde(default)]
    pub away_qb_name: Option<String>,
    #[serde(default)]
    pub home_qb_name: Option<String>,
}

impl GameRow {
    pub fn kickoff_date(&self) -> Option<chrono::NaiveDate> {
        chrono::NaiveDate::parse_from_str(&self.gameday, "%Y-%m-%d").ok()
    }

    pub fn is_regular_season(&self) -> bool {
        self.game_type == "REG"
    }

    /// Whether the game has a final score.
    pub fn is_final(&self) -> bool {
        self.home_score.is_some() && self.away_score.is_some()
    }

    /// Did the home side cover the closing spread? `None` on a push or when
    /// either the score or the line is missing. `home_covered == Some(false)`
    /// is exactly "the away side covered".
    pub fn home_covered(&self) -> Option<bool> {
        let margin = f64::from(self.result?) - self.spread_line?;
        if margin == 0.0 {
            None
        } else {
            Some(margin > 0.0)
        }
    }

    /// Did the game go over the closing total? `None` on a push or when
    /// either the score or the line is missing.
    pub fn went_over(&self) -> Option<bool> {
        let margin = f64::from(self.total?) - self.total_line?;
        if margin == 0.0 {
            None
        } else {
            Some(margin > 0.0)
        }
    }

    /// Did the home side win outright? `None` on a tie or an unplayed game.
    pub fn home_won(&self) -> Option<bool> {
        let result = self.result?;
        if result == 0 { None } else { Some(result > 0) }
    }
}

pub fn read_player_weeks(path: &Path) -> Result<Vec<PlayerWeekRow>> {
    let mut reader = csv::Reader::from_path(path)?;
    Ok(reader
        .deserialize()
        .collect::<std::result::Result<Vec<_>, _>>()?)
}

pub fn read_team_weeks(path: &Path) -> Result<Vec<TeamWeekRow>> {
    let mut reader = csv::Reader::from_path(path)?;
    Ok(reader
        .deserialize()
        .collect::<std::result::Result<Vec<_>, _>>()?)
}

pub fn read_games(path: &Path) -> Result<Vec<GameRow>> {
    let mut reader = csv::Reader::from_path(path)?;
    Ok(reader
        .deserialize()
        .collect::<std::result::Result<Vec<_>, _>>()?)
}

/// Every season's player file found under `dir`, ascending, so an atlas can
/// say exactly which seasons it was measured on rather than assuming a range.
pub fn downloaded_seasons(dir: &Path) -> Vec<u16> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut seasons: Vec<u16> = entries
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            name.strip_prefix("stats_player_week_")?
                .strip_suffix(".csv")?
                .parse()
                .ok()
        })
        .collect();
    seasons.sort_unstable();
    seasons
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real 2024 header and its first two rows, verbatim from the
    /// release. A quarterback with blanks in rate columns and a kicker with
    /// blanks nearly everywhere — the two shapes the reader has to survive.
    const PLAYER_SAMPLE: &str = "player_id,player_name,player_display_name,position,position_group,headshot_url,season,week,season_type,game_id,team,opponent_team,completions,attempts,passing_yards,passing_tds,passing_interceptions,sacks_suffered,sack_yards_lost,sack_fumbles,sack_fumbles_lost,passing_air_yards,passing_yards_after_catch,passing_first_downs,passing_epa,passing_cpoe,passing_2pt_conversions,pacr,passing_10,passing_16,passing_20,passing_40,carries,rushing_yards,rushing_tds,rushing_fumbles,rushing_fumbles_lost,rushing_first_downs,rushing_epa,rushing_2pt_conversions,rushing_10,rushing_12,rushing_20,rushing_40,receptions,targets,receiving_yards,receiving_tds,receiving_fumbles,receiving_fumbles_lost,receiving_air_yards,receiving_yards_after_catch,receiving_first_downs,receiving_epa,receiving_2pt_conversions,receiving_10,receiving_16,receiving_20,receiving_40,racr,target_share,air_yards_share,wopr,special_teams_tds,def_tackles_solo,def_tackles_with_assist,def_tackle_assists,def_tackles_for_loss,def_tackles_for_loss_yards,def_fumbles_forced,def_sacks,def_sack_yards,def_qb_hits,def_interceptions,def_interception_yards,def_pass_defended,def_tds,def_fumbles,def_safeties,def_punt_blocks,def_pat_blocks,def_fg_blocks,def_2pt_atts,def_2pt_made,misc_yards,fumble_recovery_own,fumble_recovery_yards_own,fumble_recovery_opp,fumble_recovery_yards_opp,fumble_recovery_tds,penalties,penalty_yards,fumbles_forced_by_opp,fumbles_not_forced,fumbles_out_of_bounds,fumbles_total,fumbles_lost_total,punt_returns,punt_return_yards,kickoff_returns,kickoff_return_yards,fg_made,fg_att,fg_missed,fg_blocked,fg_long,fg_pct,fg_made_0_19,fg_made_20_29,fg_made_30_39,fg_made_40_49,fg_made_50_59,fg_made_60_,fg_missed_0_19,fg_missed_20_29,fg_missed_30_39,fg_missed_40_49,fg_missed_50_59,fg_missed_60_,fg_made_list,fg_missed_list,fg_blocked_list,fg_made_distance,fg_missed_distance,fg_blocked_distance,pat_made,pat_att,pat_missed,pat_blocked,pat_pct,gwfg_made,gwfg_att,gwfg_missed,gwfg_blocked,gwfg_distance,pt_att,pt_blocked,pt_long,pt_yards,pt_inside_20,pt_out_of_bounds,pt_downed,pt_touchback,pt_fair_caught,pt_returned,pt_return_yards,pt_return_tds,pt_net_yards,fantasy_points,fantasy_points_ppr
00-0023459,A.Rodgers,Aaron Rodgers,QB,QB,\"https://static.www.nfl.com/image/upload/f_auto,q_auto/league/dypvakakxhccxs67tb0y\",2024,1,REG,2024_01_NYJ_SF,NYJ,SF,13,21,167,1,1,1,-5,0,0,164,93,8,3.25828279805137,-5.89798866283326,0,1.01829268292683,7,3,2,0,1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,,0,0,0,0,0,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,,,0,0,0,0,0,0,0,0,0,0,0,0,,,,0,0,0,0,0,0,0,,0,0,0,0,0,0,0,,0,0,0,0,0,0,0,0,0,0,8.58,8.58
00-0023853,M.Prater,Matt Prater,K,SPEC,\"https://static.www.nfl.com/image/upload/f_auto,q_auto/league/pj981bi535y4jwy6skr1\",2024,1,REG,2024_01_ARI_BUF,ARI,BUF,0,0,0,0,0,0,0,0,0,0,0,0,,,0,,0,0,0,0,0,0,0,0,0,0,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,,0,0,0,0,0,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,2,2,0,0,31,1,0,1,1,0,0,0,0,0,0,0,0,0,29;31,,,60,0,0,2,2,0,0,1,0,0,0,0,0,0,0,,0,0,0,0,0,0,0,0,0,0,0,0
";

    const TEAM_SAMPLE: &str = "season,week,team,season_type,game_id,opponent_team,completions,attempts,passing_yards,passing_tds,passing_interceptions,sacks_suffered,sack_yards_lost,sack_fumbles,sack_fumbles_lost,passing_air_yards,passing_yards_after_catch,passing_first_downs,passing_epa,passing_cpoe,passing_2pt_conversions,passing_10,passing_16,passing_20,passing_40,carries,rushing_yards,rushing_tds,rushing_fumbles,rushing_fumbles_lost,rushing_first_downs,rushing_epa,rushing_2pt_conversions,rushing_10,rushing_12,rushing_20,rushing_40,receptions,targets,receiving_yards,receiving_tds,receiving_fumbles,receiving_fumbles_lost,receiving_air_yards,receiving_yards_after_catch,receiving_first_downs,receiving_epa,receiving_2pt_conversions,receiving_10,receiving_16,receiving_20,receiving_40,special_teams_tds,def_tackles_solo,def_tackles_with_assist,def_tackle_assists,def_tackles_for_loss,def_tackles_for_loss_yards,def_fumbles_forced,def_sacks,def_sack_yards,def_qb_hits,def_interceptions,def_interception_yards,def_pass_defended,def_tds,def_fumbles,def_safeties,def_punt_blocks,def_pat_blocks,def_fg_blocks,def_2pt_atts,def_2pt_made,misc_yards,fumble_recovery_own,fumble_recovery_yards_own,fumble_recovery_opp,fumble_recovery_yards_opp,fumble_recovery_tds,penalties,penalty_yards,timeouts,fumbles_forced_by_opp,fumbles_not_forced,fumbles_out_of_bounds,fumbles_total,fumbles_lost_total,punt_returns,punt_return_yards,kickoff_returns,kickoff_return_yards,fg_made,fg_att,fg_missed,fg_blocked,fg_long,fg_pct,fg_made_0_19,fg_made_20_29,fg_made_30_39,fg_made_40_49,fg_made_50_59,fg_made_60_,fg_missed_0_19,fg_missed_20_29,fg_missed_30_39,fg_missed_40_49,fg_missed_50_59,fg_missed_60_,fg_made_list,fg_missed_list,fg_blocked_list,fg_made_distance,fg_missed_distance,fg_blocked_distance,pat_made,pat_att,pat_missed,pat_blocked,pat_pct,gwfg_made,gwfg_att,gwfg_missed,gwfg_blocked,gwfg_distance,pt_att,pt_blocked,pt_long,pt_yards,pt_inside_20,pt_out_of_bounds,pt_downed,pt_touchback,pt_fair_caught,pt_returned,pt_return_yards,pt_return_tds,pt_net_yards
2024,1,ARI,REG,2024_01_ARI_BUF,BUF,21,31,162,1,0,4,-16,1,1,191,100,10,-1.18484973521212,1.19662813601955,0,4,2,2,0,25,124,1,0,0,7,4.9849497021037,1,4,3,2,0,21,31,162,1,0,0,191,100,10,8.12971816144178,0,4,2,2,0,1,29,14,28,5,18,1,2,10,3,0,0,2,0,0,0,0,0,0,0,0,0,0,0,1,0,0,5,31,1,1,0,0,1,1,1,6,3,123,2,2,0,0,31,1,0,1,1,0,0,0,0,0,0,0,0,0,29;31,,,60,0,0,2,2,0,0,1,0,0,0,0,0,2,0,50,78,0,1,0,0,0,1,7,0,71
";

    const GAMES_SAMPLE: &str = "game_id,season,game_type,week,gameday,weekday,gametime,away_team,away_score,home_team,home_score,location,result,total,overtime,old_game_id,gsis,nfl_detail_id,pfr,pff,espn,ftn,away_rest,home_rest,away_moneyline,home_moneyline,spread_line,away_spread_odds,home_spread_odds,total_line,under_odds,over_odds,div_game,roof,surface,temp,wind,away_qb_id,home_qb_id,away_qb_name,home_qb_name,away_coach,home_coach,referee,stadium_id,stadium
1999_01_MIN_ATL,1999,REG,1,1999-09-12,Sunday,,MIN,17,ATL,14,Home,-3,31,0,1999091210,598,,199909120atl,,190912001,,7,7,,,-4,,,49,,,0,dome,astroturf,,,00-0003761,00-0002876,Randall Cunningham,Chris Chandler,Dennis Green,Dan Reeves,Gerry Austin,ATL00,Georgia Dome
1999_01_KC_CHI,1999,REG,1,1999-09-12,Sunday,,KC,17,CHI,20,Home,3,37,0,1999091206,597,,199909120chi,,190912003,,7,7,,,-3,,,38,,,0,outdoors,grass,80,12,00-0006300,00-0010560,Elvis Grbac,Shane Matthews,Gunther Cunningham,Dick Jauron,Phil Luckett,CHI98,Soldier Field
2026_01_BUF_KC,2026,REG,1,2026-09-13,Sunday,20:20,BUF,,KC,,Home,,,,,,,,,,,7,7,120,-142,2.5,-110,-110,47.5,-112,-108,0,outdoors,grass,,,,,,,,,,KC00,GEHA Field at Arrowhead Stadium
";

    fn parse_players(text: &str) -> Vec<PlayerWeekRow> {
        csv::Reader::from_reader(text.as_bytes())
            .deserialize()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap()
    }

    #[test]
    fn the_real_player_header_parses_with_blanks_as_none() {
        let rows = parse_players(PLAYER_SAMPLE);
        assert_eq!(rows.len(), 2);
        let rodgers = &rows[0];
        assert_eq!(rodgers.player_display_name, "Aaron Rodgers");
        assert_eq!(rodgers.game_id, "2024_01_NYJ_SF");
        assert_eq!(rodgers.pass_yds(), 167.0);
        assert_eq!(rodgers.pass_tds(), 1.0);
        // nflverse writes an explicit 0 target share for a player with no
        // targets; the blank-as-None behaviour is pinned separately below.
        assert_eq!(rodgers.target_share, Some(0.0));
        assert!(rodgers.is_regular_season());
        let prater = &rows[1];
        assert_eq!(prater.position, "K");
        assert_eq!(prater.fg_made, Some(2.0));
        assert_eq!(prater.pat_made, Some(2.0));
        assert_eq!(prater.kicking_points(), 8.0);
        assert_eq!(
            prater.pass_yds(),
            0.0,
            "a blank counting stat reads as zero"
        );
    }

    /// A blank numeric field is "not applicable", never zero, at the parse
    /// layer; only the counting-stat accessors turn it into zero.
    #[test]
    fn a_blank_numeric_field_reads_as_none_and_counts_as_zero() {
        let text = "player_id,player_name,player_display_name,position,season,week,season_type,game_id,team,opponent_team,passing_yards,target_share\n\
                    00-0000001,J.Doe,Jane Doe,WR,2024,1,REG,2024_01_A_B,A,B,,\n";
        let rows = parse_players(text);
        assert_eq!(rows[0].passing_yards, None);
        assert_eq!(rows[0].target_share, None);
        assert_eq!(rows[0].pass_yds(), 0.0);
    }

    #[test]
    fn the_real_team_header_parses() {
        let rows: Vec<TeamWeekRow> = csv::Reader::from_reader(TEAM_SAMPLE.as_bytes())
            .deserialize()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].team, "ARI");
        assert_eq!(rows[0].passing_yards, Some(162.0));
        assert_eq!(rows[0].rushing_yards, Some(124.0));
        assert_eq!(rows[0].def_sacks, Some(2.0));
        // The columns the subject table's covariates are built from.
        assert_eq!(rows[0].targets, Some(31.0));
        assert_eq!(rows[0].pass_yds(), 162.0);
        assert_eq!(rows[0].pass_tds(), 1.0);
        assert_eq!(rows[0].rush_yds(), 124.0);
        assert_eq!(rows[0].rush_tds(), 1.0);
        assert_eq!(rows[0].tgts(), 31.0);
    }

    /// The attempt, carry and target accessors the subject resolution of
    /// `tasks/plan.md` §2.1 ranks on, on the real header.
    #[test]
    fn attempt_carry_and_target_accessors_read_the_real_columns() {
        let rows = parse_players(PLAYER_SAMPLE);
        let rodgers = &rows[0];
        assert_eq!(rodgers.pass_atts(), 21.0);
        assert_eq!(rodgers.rush_atts(), 1.0);
        assert_eq!(rodgers.tgts(), 0.0);
        // A kicker's blanks still read as zero, so a ranking never sees a NaN.
        assert_eq!(rows[1].pass_atts(), 0.0);
        assert_eq!(rows[1].tgts(), 0.0);
    }

    /// The spread convention pinned on real rows. MIN @ ATL: `spread_line`
    /// −4 means the road team was favoured by 4, so the home side was a +4
    /// underdog; it lost by only 3 and therefore **covered**. KC @ CHI: road
    /// team favoured by 3, home side won outright, covered. A future game has
    /// no result.
    #[test]
    fn games_parse_and_the_spread_convention_holds() {
        let rows: Vec<GameRow> = csv::Reader::from_reader(GAMES_SAMPLE.as_bytes())
            .deserialize()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(rows.len(), 3);
        let atl = &rows[0];
        assert_eq!(atl.spread_line, Some(-4.0));
        assert_eq!(atl.result, Some(-3));
        assert_eq!(
            atl.home_covered(),
            Some(true),
            "a +4 dog losing by 3 covers"
        );
        assert_eq!(atl.went_over(), Some(false), "31 under a 49 total");
        assert_eq!(atl.home_won(), Some(false));
        assert_eq!(
            atl.kickoff_date(),
            chrono::NaiveDate::from_ymd_opt(1999, 9, 12)
        );
        let chi = &rows[1];
        assert_eq!(chi.home_covered(), Some(true));
        assert_eq!(chi.home_won(), Some(true));
        assert_eq!(chi.temp, Some(80.0));
        let future = &rows[2];
        assert!(!future.is_final());
        assert_eq!(future.home_covered(), None);
        assert_eq!(future.went_over(), None);
        assert_eq!(future.spread_line, Some(2.5));
        assert_eq!(future.home_moneyline, Some(-142));
        assert_eq!(future.gametime.as_deref(), Some("20:20"));
    }

    #[test]
    fn a_push_is_neither_a_cover_nor_an_over() {
        let mut row: GameRow = csv::Reader::from_reader(GAMES_SAMPLE.as_bytes())
            .deserialize()
            .next()
            .unwrap()
            .unwrap();
        row.spread_line = Some(-3.0);
        assert_eq!(row.home_covered(), None);
        row.total_line = Some(31.0);
        assert_eq!(row.went_over(), None);
    }

    #[test]
    fn urls_and_paths_agree_on_naming() {
        assert_eq!(
            player_week_url(2024),
            "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_2024.csv"
        );
        assert_eq!(
            team_week_url(2010),
            "https://github.com/nflverse/nflverse-data/releases/download/stats_team/stats_team_week_2010.csv"
        );
        assert_eq!(
            schedules_url(),
            "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
        );
        let dir = Path::new("data/raw/nflverse");
        assert_eq!(
            player_week_path(dir, 2024).file_name().unwrap(),
            "stats_player_week_2024.csv"
        );
        assert_eq!(schedules_path(dir).file_name().unwrap(), "games.csv");
    }

    #[test]
    fn downloaded_seasons_lists_only_player_files_in_order() {
        let dir = std::env::temp_dir().join(format!("nfl-sgp-seasons-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for name in [
            "stats_player_week_2016.csv",
            "stats_player_week_2010.csv",
            "stats_team_week_2010.csv",
            "games.csv",
            "stats_player_week_junk.csv",
        ] {
            std::fs::write(dir.join(name), "").unwrap();
        }
        assert_eq!(downloaded_seasons(&dir), vec![2010, 2016]);
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(downloaded_seasons(&dir).is_empty());
    }
}
