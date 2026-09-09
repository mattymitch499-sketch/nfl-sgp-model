//! Official team depth charts — who a team says is starting, before the game.
//!
//! The atlas identifies a team's Wr1 and Rb1 from *prior usage*, because that
//! is what a completed season can support and it is leakage-safe by
//! construction. For a game that has not been played the published depth chart
//! is the better source, and it is decisively better across an offseason: on
//! 2026-09-08 the usage rule names Stefon Diggs as New England's Wr1 and
//! Kenneth Walker III as Seattle's Rb1, while the depth charts say **A.J.
//! Brown** and **Jadarian Price** — Walker having moved to Kansas City. A rule
//! that cannot see a trade is not a rule you want naming the player on a
//! ticket.
//!
//! ## Two formats, and only one is needed here
//!
//! nflverse changed the shape between 2024 and 2025:
//!
//! * **2010–2024** — one row per (season, week, team, player) with an explicit
//!   `depth_team` rank. Per-week, so it is the format a *historical* study
//!   would use.
//! * **2025 onward** — timestamped snapshots (`dt`), no week column: the file
//!   accumulates every scrape of the live chart.
//!
//! This module parses the **snapshot** format only, because the live sheet is
//! the only consumer today. The historical format is what the deferred
//! depth-versus-usage agreement study will need, and is deliberately not
//! written until that study is.
//!
//! ## The snapshot rule
//!
//! The file holds months of scrapes — the 2026 file spans March to today — and
//! they are **not in date order**. Taking the last rows in the file yields a
//! March chart, which in March lists players who have since been traded. So the
//! rule is: per team, keep only rows carrying that team's newest `dt`, and
//! within them order by `pos_rank`. Getting this wrong is silent: an offseason
//! chart parses perfectly and simply names the wrong players.

use crate::games::canonical_team;
use crate::{Error, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

/// Offensive positions the families draw subjects from. Defensive rows use
/// position abbreviations of their own (`LDE`, `WLB`, …) and are dropped.
pub const OFFENSIVE_POSITIONS: [&str; 4] = ["QB", "RB", "WR", "TE"];

/// One player's place on a team's chart.
#[derive(Debug, Clone, PartialEq)]
pub struct DepthEntry {
    /// nflverse `gsis_id`, joining straight to the weekly player rows. Empty
    /// for a player nflverse has no id for (a camp body, usually) — such a row
    /// is kept, because the *name* is still what a book would price.
    pub player_id: String,
    pub display_name: String,
    /// `QB`, `RB`, `WR` or `TE`.
    pub position: String,
    /// 1 is the starter.
    pub rank: u32,
}

/// Every team's current offensive chart.
#[derive(Debug, Clone, Default)]
pub struct DepthChart {
    /// Canonical team code → entries, sorted by (position, rank).
    by_team: BTreeMap<String, Vec<DepthEntry>>,
    /// The snapshot each team's entries came from, for reporting.
    snapshot: BTreeMap<String, String>,
}

impl DepthChart {
    /// Reads the snapshot-format file and keeps each team's newest chart.
    pub fn read_snapshots(path: &Path) -> Result<Self> {
        let mut reader = csv::Reader::from_path(path)?;
        let rows: Vec<SnapshotRow> = reader
            .deserialize()
            .collect::<std::result::Result<_, _>>()
            .map_err(Error::Csv)?;
        Ok(Self::from_snapshot_rows(rows))
    }

    pub fn from_snapshot_rows(rows: Vec<SnapshotRow>) -> Self {
        // Newest timestamp per team first: the file is not date-ordered, and a
        // stale chart is indistinguishable from a current one once parsed.
        let mut newest: BTreeMap<String, String> = BTreeMap::new();
        for row in &rows {
            let team = canonical_team(&row.team).to_string();
            let entry = newest.entry(team).or_default();
            if row.dt > *entry {
                *entry = row.dt.clone();
            }
        }
        let mut by_team: BTreeMap<String, Vec<DepthEntry>> = BTreeMap::new();
        for row in &rows {
            let team = canonical_team(&row.team).to_string();
            if newest.get(&team).is_none_or(|latest| row.dt != *latest) {
                continue;
            }
            if !OFFENSIVE_POSITIONS.contains(&row.pos_abb.as_str()) {
                continue;
            }
            let Some(rank) = row.pos_rank else { continue };
            by_team.entry(team).or_default().push(DepthEntry {
                player_id: row.gsis_id.clone().unwrap_or_default(),
                display_name: row.player_name.clone(),
                position: row.pos_abb.clone(),
                rank,
            });
        }
        for entries in by_team.values_mut() {
            entries.sort_by(|left, right| {
                (&left.position, left.rank, &left.display_name).cmp(&(
                    &right.position,
                    right.rank,
                    &right.display_name,
                ))
            });
        }
        Self {
            by_team,
            snapshot: newest,
        }
    }

    /// Every entry at a position, starter first.
    pub fn ranked(&self, team: &str, position: &str) -> Vec<&DepthEntry> {
        let canonical = canonical_team(team);
        self.by_team
            .get(canonical)
            .map(|entries| {
                entries
                    .iter()
                    .filter(|entry| entry.position == position)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The team's listed starter at a position.
    pub fn starter(&self, team: &str, position: &str) -> Option<&DepthEntry> {
        self.ranked(team, position).into_iter().next()
    }

    /// Receivers and tight ends together, starter-first within each position
    /// and receivers before tight ends — the pool the `Wr1` role draws from,
    /// matching the atlas (`games.rs`'s `Skill::Receiver` covers WR and TE).
    pub fn receiver_pool(&self, team: &str) -> Vec<&DepthEntry> {
        let mut pool = self.ranked(team, "WR");
        pool.extend(self.ranked(team, "TE"));
        pool
    }

    /// The timestamp a team's chart was scraped at.
    pub fn snapshot_of(&self, team: &str) -> Option<&str> {
        self.snapshot.get(canonical_team(team)).map(String::as_str)
    }

    pub fn teams(&self) -> usize {
        self.by_team.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_team.is_empty()
    }
}

/// One row of the 2025+ snapshot file. Only the columns the roles need.
#[derive(Debug, Clone, Deserialize)]
pub struct SnapshotRow {
    /// RFC-3339 scrape time. Lexicographic comparison is chronological for
    /// this format, which is why the newest-per-team pass can compare strings.
    pub dt: String,
    pub team: String,
    pub player_name: String,
    #[serde(default)]
    pub gsis_id: Option<String>,
    #[serde(default)]
    pub pos_abb: String,
    #[serde(default)]
    pub pos_rank: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real rows, in the released column order, including a defensive row that
    /// must be dropped and an older snapshot that must lose.
    const SNAPSHOT: &str = "dt,team,player_name,espn_id,gsis_id,pos_grp_id,pos_grp,pos_id,pos_name,pos_abb,pos_slot,pos_rank\n\
2026-03-22T06:38:42Z,KC,Rashee Rice,4426388,00-0039067,17,3WR 1TE,1,Wide Receiver,WR,1,1\n\
2026-03-22T06:38:42Z,KC,Kenneth Walker III,4373678,00-0038134,17,3WR 1TE,11,Running Back,RB,11,1\n\
2026-03-22T06:38:42Z,KC,Patrick Mahomes,3139477,00-0033873,17,3WR 1TE,9,Quarterback,QB,9,1\n\
2026-09-08T11:56:57Z,KC,Xavier Worthy,4432773,00-0039894,17,3WR 1TE,2,Wide Receiver,WR,2,1\n\
2026-09-08T11:56:57Z,KC,Rashee Rice,4426388,00-0039067,17,3WR 1TE,1,Wide Receiver,WR,1,2\n\
2026-09-08T11:56:57Z,KC,Travis Kelce,15847,00-0030506,17,3WR 1TE,10,Tight End,TE,10,1\n\
2026-09-08T11:56:57Z,KC,Isiah Pacheco,4361579,00-0037197,17,3WR 1TE,11,Running Back,RB,11,1\n\
2026-09-08T11:56:57Z,KC,Patrick Mahomes,3139477,00-0033873,17,3WR 1TE,9,Quarterback,QB,9,1\n\
2026-09-08T11:56:57Z,KC,Josh Sweat,3693166,00-0034381,16,Base 4-3 D,11,Left Defensive End,LDE,1,1\n\
2026-09-08T09:00:00Z,OAK,Brock Bowers,4432665,00-0039910,17,3WR 1TE,10,Tight End,TE,10,1\n";

    fn chart() -> DepthChart {
        let rows: Vec<SnapshotRow> = csv::Reader::from_reader(SNAPSHOT.as_bytes())
            .deserialize()
            .collect::<std::result::Result<_, _>>()
            .unwrap();
        DepthChart::from_snapshot_rows(rows)
    }

    /// **The rule the whole module exists for.** The file is not date-ordered
    /// and spans months; the March chart still lists Kenneth Walker III at
    /// Kansas City, who by September is not there. Taking the newest snapshot
    /// per team is what keeps a traded player off the ticket — and the failure
    /// is silent, because a stale chart parses perfectly well.
    #[test]
    fn only_the_newest_snapshot_per_team_survives() {
        let chart = chart();
        assert_eq!(chart.snapshot_of("KC"), Some("2026-09-08T11:56:57Z"));
        assert_eq!(
            chart
                .starter("KC", "RB")
                .map(|entry| entry.display_name.as_str()),
            Some("Isiah Pacheco"),
            "the March chart's back must not win"
        );
        assert!(
            !chart
                .ranked("KC", "RB")
                .iter()
                .any(|entry| entry.display_name == "Kenneth Walker III"),
            "a player from an older snapshot must not appear at all"
        );
    }

    /// Rank 1 is the starter, and the ordering is by rank rather than file
    /// order — the September chart lists Worthy at rank 1 and Rice at 2.
    #[test]
    fn the_starter_is_rank_one_not_the_first_row() {
        let chart = chart();
        assert_eq!(
            chart
                .starter("KC", "WR")
                .map(|entry| entry.display_name.as_str()),
            Some("Xavier Worthy")
        );
        let receivers = chart.ranked("KC", "WR");
        assert_eq!(receivers.len(), 2);
        assert_eq!(receivers[0].rank, 1);
        assert_eq!(receivers[1].display_name, "Rashee Rice");
    }

    /// Defensive rows use their own abbreviations and are not subjects here.
    #[test]
    fn defensive_rows_are_dropped() {
        let chart = chart();
        assert!(
            !chart
                .ranked("KC", "LDE")
                .iter()
                .any(|entry| entry.display_name == "Josh Sweat")
        );
        for entry in chart.receiver_pool("KC") {
            assert!(OFFENSIVE_POSITIONS.contains(&entry.position.as_str()));
        }
    }

    /// The receiver pool is WR and TE together, matching the atlas's
    /// `Skill::Receiver`, starter-first.
    #[test]
    fn the_receiver_pool_covers_wide_receivers_and_tight_ends() {
        let chart = chart();
        let pool = chart.receiver_pool("KC");
        let names: Vec<&str> = pool.iter().map(|e| e.display_name.as_str()).collect();
        assert_eq!(names, vec!["Xavier Worthy", "Rashee Rice", "Travis Kelce"]);
    }

    /// Relocated franchises fold onto the current code, so a feed still saying
    /// OAK is found under LV — and asked for by either name.
    #[test]
    fn relocated_codes_fold_onto_the_current_franchise() {
        let chart = chart();
        assert_eq!(
            chart.starter("LV", "TE").map(|e| e.display_name.as_str()),
            Some("Brock Bowers")
        );
        assert_eq!(
            chart.starter("OAK", "TE").map(|e| e.display_name.as_str()),
            Some("Brock Bowers")
        );
    }

    /// An unknown team is empty, not a panic and not a guess.
    #[test]
    fn an_unknown_team_yields_nothing() {
        let chart = chart();
        assert!(chart.starter("XXX", "QB").is_none());
        assert!(chart.ranked("XXX", "WR").is_empty());
        assert_eq!(chart.snapshot_of("XXX"), None);
        assert_eq!(chart.teams(), 2);
        assert!(!chart.is_empty());
    }
}
