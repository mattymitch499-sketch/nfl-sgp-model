//! The six v1 families and the paired observations they draw — `tasks/plan.md`
//! §1.1, §2.
//!
//! A **family** is an ordered pair of legs; a **leg** is (role, stat,
//! direction); an **observation** is one game-anchor carrying the two raw
//! values, a pre-game covariate, and the cluster ids the bootstrap resamples.
//! This module is nothing but that translation: it turns the [`GameTable`]
//! into six lists of `(x_a, x_b, c, cluster)` and knows no statistics at all.
//!
//! ## Direction is part of the leg, not part of the value
//!
//! `rb1rush_x_qbpassunder` is "Rb1 rushing yards **Over** × Qb passing yards
//! **Under**", but its observations carry the quarterback's raw yardage, not a
//! flipped or negated number. That is deliberate and it matters: §1.3 ranks the
//! pooled values into uniform scores, and §1.4 then selects the *region*
//! `{v ≤ p_B}` for an Under and `{v > 1 − p_B}` for an Over. The direction is
//! applied once, at evaluation, against the market's marginal. Applying it
//! here as well — negating the values, or ranking them descending — would
//! apply it twice and silently reverse the sign of the measured dependence.
//! The same raw pool therefore serves `rb1rush_x_qbpassunder` and its control
//! `passyds_x_rb1rush`, which is exactly why the plan can list both.
//!
//! ## Which observations exist
//!
//! Same-team families yield **one row per (game, team)** and only where both
//! subjects resolved: a game in which the as-of Wr1 did not dress produces no
//! `passyds_x_wr1recyds` row for that team, because [`crate::games`] refused to
//! substitute the receiver who did play. `bothqb_passyds` is cross-team and
//! yields **one row per game**, always anchored on the home side, so a game is
//! never counted twice under two anchors.
//!
//! `spread_x_total` is the odd one out in two ways. Its values are read
//! straight from `games.csv` rather than from the subject table, so it reaches
//! back to 1999 instead of 2010 — closing lines are the one series that goes
//! back that far. And both its values are already **centred by the market**
//! (`result − spread_line`, `total − total_line`), so it is flagged
//! [`Family::is_market_centred`] and Phase 3 must skip the leave-one-out
//! standardisation of §1.2 for it. Centring a residual against a season of
//! leave-one-out means would throw away the market's own, far better,
//! per-game centring — and that centring is the entire point of using this
//! family as the end-to-end validation of §1.9.

use crate::games::{GameTable, Stat, SubjectKey, TeamGame, canonical_team};

/// The subject a leg is written on (plan §1.1).
///
/// [`Role::Game`] is one variant beyond the five the plan lists. The game
/// total in `spread_x_total` has no team subject, and labelling it `Team`
/// would let the scorer of §3 resolve it against a *team* total market — a
/// different bet at a different number. The plan's own leg grammar in §3
/// already separates `<Team> team total over` from `game total over`, so the
/// sixth role only makes explicit a distinction the plan draws elsewhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    Team,
    Qb,
    Wr1,
    Rb1,
    OppQb,
    Game,
}

impl Role {
    pub fn label(self) -> &'static str {
        match self {
            Self::Team => "Team",
            Self::Qb => "Qb",
            Self::Wr1 => "Wr1",
            Self::Rb1 => "Rb1",
            Self::OppQb => "OppQb",
            Self::Game => "Game",
        }
    }
}

/// Which side of the line a leg is on (plan §1.1). Carried on the leg and
/// applied at evaluation, never to the stored values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    Over,
    Under,
}

impl Direction {
    pub fn label(self) -> &'static str {
        match self {
            Self::Over => "Over",
            Self::Under => "Under",
        }
    }
}

/// One half of a family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Leg {
    pub role: Role,
    pub stat: Stat,
    pub direction: Direction,
}

impl Leg {
    const fn new(role: Role, stat: Stat, direction: Direction) -> Self {
        Self {
            role,
            stat,
            direction,
        }
    }

    /// `"Qb qb_pass_yds Under"` — the form the atlas and the scorer print.
    pub fn label(&self) -> String {
        format!(
            "{} {} {}",
            self.role.label(),
            self.stat.label(),
            self.direction.label()
        )
    }
}

/// How the family's observations are grouped for the cluster bootstrap of
/// plan §1.7.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterScheme {
    /// Every observation belongs to one subject-season, in
    /// [`Observation::cluster_primary`]; [`Observation::cluster_secondary`] is
    /// empty.
    SingleSubjectSeason,
    /// A cross-team family, where no single subject owns the pair. Two one-way
    /// bootstraps are run — one grouping by the home side, one by the away —
    /// and §1.7 reports the **wider** interval. Both ids are carried.
    TwoOneWay,
}

/// How Phase 3 turns the covariate into strata (plan §1.8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CovariateScheme {
    /// Terciles of the pooled covariate, plus `all`. Phase 3 computes the
    /// edges and stores them in the atlas; this module only supplies the raw
    /// value.
    Terciles,
    /// The plan's spread families name their own buckets — favourite by ≥ 3,
    /// within 3, underdog by ≥ 3 — rather than taking terciles. Three points is
    /// a key number, not a distributional quantile, and a tercile edge that
    /// landed on 2.5 would put a field-goal favourite in the same stratum as a
    /// pick'em. Use [`spread_bucket`].
    SpreadBuckets,
}

/// What a family stratifies on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Covariate {
    /// Column name in the outputs.
    pub label: &'static str,
    /// What it is and when it is known, in one line.
    pub description: &'static str,
    pub scheme: CovariateScheme,
}

/// The three spread strata of plan §2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpreadBucket {
    /// The anchor team is favoured by three or more.
    Favourite,
    /// Within three points either way.
    Pickem,
    /// The anchor team is an underdog by three or more.
    Underdog,
}

impl SpreadBucket {
    pub fn label(self) -> &'static str {
        match self {
            Self::Favourite => "fav3+",
            Self::Pickem => "within3",
            Self::Underdog => "dog3+",
        }
    }
}

/// Buckets a closing spread stated **from the anchor team's perspective**
/// (positive when the anchor is favoured), as [`TeamGame::anchor_spread`] is.
///
/// Exactly three is a favourite and exactly minus three an underdog: the plan
/// says "favourite by ≥ 3", and three is the modal NFL spread, so putting it
/// on the boundary either way would move a large block of games. It is written
/// down here rather than left to a reader.
pub fn spread_bucket(anchor_spread: f64) -> SpreadBucket {
    if anchor_spread >= 3.0 {
        SpreadBucket::Favourite
    } else if anchor_spread <= -3.0 {
        SpreadBucket::Underdog
    } else {
        SpreadBucket::Pickem
    }
}

/// The six families of plan §2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Family {
    /// Team points Over × Qb passing TDs Over. The part-whole pair: the
    /// touchdowns are a large part of the points. Baseball's one real
    /// under-allowance was the part-whole family, so this is where to look
    /// first.
    TeamtotalXPasstd,
    /// Qb passing yards Over × Wr1 receiving yards Over. The canonical stack.
    PassydsXWr1recyds,
    /// Rb1 rushing yards Over × Qb passing yards Under. The game-script pair:
    /// a team that runs is a team that is ahead.
    Rb1rushXQbpassunder,
    /// Home Qb passing yards Over × away Qb passing yards Over. Cross-team,
    /// and the only family whose two subjects are on opposite sidelines.
    BothqbPassyds,
    /// Qb passing yards Over × Rb1 rushing yards Over — the **control**. The
    /// DFS literature reports this pair as near-uncorrelated, and a control
    /// family is how the pipeline proves it is not manufacturing dependence
    /// (METHOD.md lesson 4).
    PassydsXRb1rush,
    /// Home covers × game Over. The end-to-end validation family of §1.9 and a
    /// calibration family: books have priced sides-with-totals for decades, so
    /// a wrong answer here indicts the machinery rather than the market.
    SpreadXTotal,
}

/// Every family, in the plan's table order.
pub const ALL: [Family; 6] = [
    Family::TeamtotalXPasstd,
    Family::PassydsXWr1recyds,
    Family::Rb1rushXQbpassunder,
    Family::BothqbPassyds,
    Family::PassydsXRb1rush,
    Family::SpreadXTotal,
];

impl Family {
    /// The key the plan, the atlas files and the SGP sheet all use.
    pub fn key(self) -> &'static str {
        match self {
            Self::TeamtotalXPasstd => "teamtotal_x_passtd",
            Self::PassydsXWr1recyds => "passyds_x_wr1recyds",
            Self::Rb1rushXQbpassunder => "rb1rush_x_qbpassunder",
            Self::BothqbPassyds => "bothqb_passyds",
            Self::PassydsXRb1rush => "passyds_x_rb1rush",
            Self::SpreadXTotal => "spread_x_total",
        }
    }

    pub fn from_key(key: &str) -> Option<Self> {
        ALL.into_iter().find(|family| family.key() == key)
    }

    /// The ordered pair (A, B).
    pub fn legs(self) -> [Leg; 2] {
        use Direction::{Over, Under};
        match self {
            Self::TeamtotalXPasstd => [
                Leg::new(Role::Team, Stat::TeamPoints, Over),
                Leg::new(Role::Qb, Stat::QbPassingTds, Over),
            ],
            Self::PassydsXWr1recyds => [
                Leg::new(Role::Qb, Stat::QbPassingYards, Over),
                Leg::new(Role::Wr1, Stat::Wr1ReceivingYards, Over),
            ],
            Self::Rb1rushXQbpassunder => [
                Leg::new(Role::Rb1, Stat::Rb1RushingYards, Over),
                // The one Under in v1. The values stored are still raw yards.
                Leg::new(Role::Qb, Stat::QbPassingYards, Under),
            ],
            Self::BothqbPassyds => [
                Leg::new(Role::Qb, Stat::QbPassingYards, Over),
                Leg::new(Role::OppQb, Stat::QbPassingYards, Over),
            ],
            Self::PassydsXRb1rush => [
                Leg::new(Role::Qb, Stat::QbPassingYards, Over),
                Leg::new(Role::Rb1, Stat::Rb1RushingYards, Over),
            ],
            Self::SpreadXTotal => [
                Leg::new(Role::Team, Stat::MarginVsSpread, Over),
                Leg::new(Role::Game, Stat::TotalVsLine, Over),
            ],
        }
    }

    /// What the family is stratified by (plan §1.8, §2).
    pub fn covariate(self) -> Covariate {
        match self {
            Self::TeamtotalXPasstd => Covariate {
                label: "pass_td_share_asof",
                description: "the team's passing share of its offensive touchdowns over the prior season plus the season to date, at least 8 games",
                scheme: CovariateScheme::Terciles,
            },
            Self::PassydsXWr1recyds => Covariate {
                label: "wr1_target_share_asof",
                description: "the as-of Wr1's share of the team's targets over the previous 8 team games; week 1 uses the prior season",
                scheme: CovariateScheme::Terciles,
            },
            Self::Rb1rushXQbpassunder | Self::PassydsXRb1rush => Covariate {
                label: "anchor_spread",
                description: "the closing spread from the anchor team's perspective, positive when it is favoured",
                scheme: CovariateScheme::SpreadBuckets,
            },
            Self::BothqbPassyds | Self::SpreadXTotal => Covariate {
                label: "total_line",
                description: "the closing game total",
                scheme: CovariateScheme::Terciles,
            },
        }
    }

    /// How §1.7's bootstrap groups this family's rows.
    pub fn cluster_scheme(self) -> ClusterScheme {
        match self {
            Self::BothqbPassyds => ClusterScheme::TwoOneWay,
            _ => ClusterScheme::SingleSubjectSeason,
        }
    }

    /// What [`Observation::cluster_primary`] holds, in words — the atlas
    /// prints this beside every interval, because METHOD.md lesson 7 cost a
    /// verdict when a cluster unit was written down nowhere.
    pub fn cluster_description(self) -> &'static str {
        match self {
            Self::TeamtotalXPasstd | Self::Rb1rushXQbpassunder | Self::PassydsXRb1rush => {
                "team-season"
            }
            Self::PassydsXWr1recyds => "qb-season",
            Self::BothqbPassyds => "home team-season, and away team-season, wider wins",
            Self::SpreadXTotal => "home team-season",
        }
    }

    /// Whether both of the family's values are already residuals against a
    /// closing line, in which case Phase 3 **must not** apply the
    /// leave-one-out standardisation of §1.2.
    pub fn is_market_centred(self) -> bool {
        self.legs().iter().all(|leg| leg.stat.is_market_centred())
    }

    /// Whether an observation is one per game (cross-team or market-centred)
    /// rather than one per (game, team).
    pub fn is_game_anchored(self) -> bool {
        matches!(self, Self::BothqbPassyds | Self::SpreadXTotal)
    }
}

/// One game-anchor: the unit §1.4 counts.
#[derive(Debug, Clone)]
pub struct Observation {
    pub game_id: String,
    pub season: u16,
    pub week: u8,
    /// The team the row is anchored on — the home side for a game-anchored
    /// family. Canonical code.
    pub anchor_team: String,
    /// Leg A's raw value. Raw: the leg's direction is applied at evaluation.
    pub x_a: f64,
    /// Leg B's raw value.
    pub x_b: f64,
    /// The subject-season leg A's value is standardised within (§1.2).
    pub subject_a: SubjectKey,
    /// The subject-season leg B's value is standardised within.
    pub subject_b: SubjectKey,
    /// The pre-game covariate. `None` is kept, not dropped: the row still
    /// belongs to the `all` stratum, and dropping it would quietly select on
    /// "games where the depth chart was established", which is not a random
    /// subset of games.
    pub covariate: Option<f64>,
    /// The cluster the bootstrap resamples (§1.7).
    pub cluster_primary: u64,
    /// The second one-way cluster, for [`ClusterScheme::TwoOneWay`] only.
    pub cluster_secondary: Option<u64>,
}

/// Every observation of one family, in table order (season, week, game, team).
///
/// Deterministic: the same table always yields the same rows in the same
/// order, which is what makes the atlas bit-for-bit reproducible.
pub fn observations(family: Family, table: &GameTable) -> Vec<Observation> {
    match family {
        Family::SpreadXTotal => market_centred_observations(table),
        Family::BothqbPassyds => cross_team_observations(table),
        _ => same_team_observations(family, table),
    }
}

/// How many observations carry no covariate and therefore reach only the `all`
/// stratum (§1.8).
pub fn missing_covariate(observations: &[Observation]) -> usize {
    observations
        .iter()
        .filter(|observation| observation.covariate.is_none())
        .count()
}

/// One row per (game, team) where both legs' subjects resolved.
fn same_team_observations(family: Family, table: &GameTable) -> Vec<Observation> {
    let [leg_a, leg_b] = family.legs();
    table
        .rows
        .iter()
        .filter_map(|row| {
            // `stat` returns `None` exactly when the leg's subject did not
            // resolve, so this one `?` chain is the whole "both subjects
            // present" rule — including the dropped-not-replaced Wr1.
            let x_a = row.stat(leg_a.stat)?;
            let x_b = row.stat(leg_b.stat)?;
            let subject_a = row.subject(leg_a.stat)?;
            let subject_b = row.subject(leg_b.stat)?;
            let cluster_primary = match family {
                // §2's cluster column: the quarterback owns the stack, the
                // team owns everything else here.
                Family::PassydsXWr1recyds => subject_a.cluster_id(),
                _ => row.team_season().cluster_id(),
            };
            Some(Observation {
                game_id: row.game_id.clone(),
                season: row.season,
                week: row.week,
                anchor_team: row.team.clone(),
                x_a,
                x_b,
                subject_a,
                subject_b,
                covariate: covariate_of(family, row),
                cluster_primary,
                cluster_secondary: None,
            })
        })
        .collect()
}

/// One row per game, home side as anchor.
///
/// Iterating the home rows rather than the games guarantees the "at most one
/// row per game" property structurally: there is exactly one home row per game
/// in the table.
fn cross_team_observations(table: &GameTable) -> Vec<Observation> {
    let [leg_a, leg_b] = Family::BothqbPassyds.legs();
    table
        .rows
        .iter()
        .filter(|row| row.home)
        .filter_map(|row| {
            let (home, away) = table.game_rows(&row.game_id)?;
            let x_a = home.stat(leg_a.stat)?;
            let x_b = away.stat(leg_b.stat)?;
            Some(Observation {
                game_id: home.game_id.clone(),
                season: home.season,
                week: home.week,
                anchor_team: home.team.clone(),
                x_a,
                x_b,
                subject_a: home.subject(leg_a.stat)?,
                subject_b: away.subject(leg_b.stat)?,
                covariate: home.total_line,
                // §2's cluster column for this family is "home team-season and
                // away team-season, wider wins". The two quarterback-seasons
                // are on the row as well, in `subject_a` / `subject_b`, so a
                // bootstrap that prefers them can take
                // `subject_a.cluster_id()` — but the team-season is the
                // coarser grouping and therefore the conservative one, and
                // METHOD.md lesson 7 is about an interval that came out too
                // narrow.
                cluster_primary: home.team_season().cluster_id(),
                cluster_secondary: Some(away.team_season().cluster_id()),
            })
        })
        .collect()
}

/// One row per game, read from `games.csv` rather than the subject table so
/// the family reaches 1999 (plan §1.9, §2.2).
fn market_centred_observations(table: &GameTable) -> Vec<Observation> {
    table
        .schedule
        .iter()
        .filter_map(|game| {
            let result = f64::from(game.result?);
            let total = f64::from(game.total?);
            let spread_line = game.spread_line?;
            let total_line = game.total_line?;
            let anchor = canonical_team(&game.home_team).to_string();
            let subject = SubjectKey::Team(anchor.clone(), game.season);
            Some(Observation {
                game_id: game.game_id.clone(),
                season: game.season,
                week: game.week,
                anchor_team: anchor,
                // Pushes (a zero residual) are kept. The stored value is a
                // continuous residual, not a settled bet, and §1.3's jitter
                // handles the tie; discarding pushes would drop precisely the
                // games that landed on the number the market chose.
                x_a: result - spread_line,
                x_b: total - total_line,
                cluster_primary: subject.cluster_id(),
                subject_a: subject.clone(),
                subject_b: subject,
                covariate: Some(total_line),
                cluster_secondary: None,
            })
        })
        .collect()
}

fn covariate_of(family: Family, row: &TeamGame) -> Option<f64> {
    match family {
        Family::TeamtotalXPasstd => row.pass_td_share_asof,
        Family::PassydsXWr1recyds => row.wr1_target_share_asof,
        Family::Rb1rushXQbpassunder | Family::PassydsXRb1rush => row.anchor_spread,
        Family::BothqbPassyds | Family::SpreadXTotal => row.total_line,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::fixtures::*;

    #[test]
    fn keys_round_trip_and_are_the_plans_own() {
        let keys: Vec<&str> = ALL.iter().map(|family| family.key()).collect();
        assert_eq!(
            keys,
            vec![
                "teamtotal_x_passtd",
                "passyds_x_wr1recyds",
                "rb1rush_x_qbpassunder",
                "bothqb_passyds",
                "passyds_x_rb1rush",
                "spread_x_total",
            ]
        );
        for family in ALL {
            assert_eq!(Family::from_key(family.key()), Some(family));
        }
        assert_eq!(Family::from_key("not_a_family"), None);
    }

    #[test]
    fn the_legs_match_the_plans_table() {
        use Direction::Over;
        assert_eq!(
            Family::TeamtotalXPasstd.legs(),
            [
                Leg::new(Role::Team, Stat::TeamPoints, Over),
                Leg::new(Role::Qb, Stat::QbPassingTds, Over)
            ]
        );
        assert_eq!(
            Family::PassydsXWr1recyds.legs(),
            [
                Leg::new(Role::Qb, Stat::QbPassingYards, Over),
                Leg::new(Role::Wr1, Stat::Wr1ReceivingYards, Over)
            ]
        );
        assert_eq!(
            Family::BothqbPassyds.legs(),
            [
                Leg::new(Role::Qb, Stat::QbPassingYards, Over),
                Leg::new(Role::OppQb, Stat::QbPassingYards, Over)
            ]
        );
        assert_eq!(
            Family::SpreadXTotal.legs(),
            [
                Leg::new(Role::Team, Stat::MarginVsSpread, Over),
                Leg::new(Role::Game, Stat::TotalVsLine, Over)
            ]
        );
        assert_eq!(
            Family::Rb1rushXQbpassunder.legs()[1].label(),
            "Qb qb_pass_yds Under"
        );
    }

    /// The one Under in v1, and the property that keeps it honest: the control
    /// family is the same two stats with the legs swapped and both Over, and
    /// the two families draw the **same raw numbers**. If direction were baked
    /// into the values, they could not.
    #[test]
    fn direction_lives_on_the_leg_and_never_touches_the_values() {
        assert_eq!(
            Family::Rb1rushXQbpassunder.legs()[1].direction,
            Direction::Under
        );
        assert_eq!(Family::PassydsXRb1rush.legs()[1].direction, Direction::Over);
        let table = mini_table();
        let under = observations(Family::Rb1rushXQbpassunder, &table);
        let control = observations(Family::PassydsXRb1rush, &table);
        assert_eq!(under.len(), control.len());
        for (left, right) in under.iter().zip(&control) {
            assert_eq!(left.game_id, right.game_id);
            assert_eq!(left.anchor_team, right.anchor_team);
            // Leg A of one is leg B of the other, raw and unflipped.
            assert_eq!(left.x_a, right.x_b, "Rb1 rushing yards, unmodified");
            assert_eq!(left.x_b, right.x_a, "Qb passing yards, not negated");
        }
    }

    #[test]
    fn cluster_schemes_and_market_centring_match_the_plan() {
        assert_eq!(
            Family::BothqbPassyds.cluster_scheme(),
            ClusterScheme::TwoOneWay
        );
        for family in ALL {
            if family != Family::BothqbPassyds {
                assert_eq!(
                    family.cluster_scheme(),
                    ClusterScheme::SingleSubjectSeason,
                    "{}",
                    family.key()
                );
            }
        }
        assert!(Family::SpreadXTotal.is_market_centred());
        for family in ALL {
            if family != Family::SpreadXTotal {
                assert!(!family.is_market_centred(), "{}", family.key());
            }
        }
        assert_eq!(Family::PassydsXWr1recyds.cluster_description(), "qb-season");
    }

    #[test]
    fn the_spread_families_bucket_rather_than_tercile() {
        for family in [Family::Rb1rushXQbpassunder, Family::PassydsXRb1rush] {
            assert_eq!(family.covariate().scheme, CovariateScheme::SpreadBuckets);
            assert_eq!(family.covariate().label, "anchor_spread");
        }
        for family in [
            Family::TeamtotalXPasstd,
            Family::PassydsXWr1recyds,
            Family::BothqbPassyds,
            Family::SpreadXTotal,
        ] {
            assert_eq!(family.covariate().scheme, CovariateScheme::Terciles);
        }
        // The boundaries, which are on the key number rather than a quantile.
        assert_eq!(spread_bucket(3.0), SpreadBucket::Favourite);
        assert_eq!(spread_bucket(3.5), SpreadBucket::Favourite);
        assert_eq!(spread_bucket(2.5), SpreadBucket::Pickem);
        assert_eq!(spread_bucket(0.0), SpreadBucket::Pickem);
        assert_eq!(spread_bucket(-2.5), SpreadBucket::Pickem);
        assert_eq!(spread_bucket(-3.0), SpreadBucket::Underdog);
        assert_eq!(spread_bucket(-7.0), SpreadBucket::Underdog);
        assert_eq!(SpreadBucket::Favourite.label(), "fav3+");
    }

    /// The counting property the gate asks for: no (game, team) twice in a
    /// same-team family, no game twice in a cross-team one.
    #[test]
    fn every_family_counts_each_anchor_at_most_once() {
        let table = mini_table();
        for family in ALL {
            let rows = observations(family, &table);
            assert!(!rows.is_empty(), "{} produced nothing", family.key());
            let mut seen = std::collections::BTreeSet::new();
            for row in &rows {
                let key = if family.is_game_anchored() {
                    row.game_id.clone()
                } else {
                    format!("{}|{}", row.game_id, row.anchor_team)
                };
                assert!(seen.insert(key), "{} repeated an anchor", family.key());
            }
            if family.is_game_anchored() {
                assert!(
                    rows.iter()
                        .all(|row| row.anchor_team == home_team(row.week)),
                    "{} must anchor on the home side",
                    family.key()
                );
            }
        }
    }

    /// `bothqb_passyds` carries both quarterbacks and both team-seasons, and
    /// the two are different subjects.
    #[test]
    fn the_cross_team_family_is_one_row_per_game_with_both_sides() {
        let table = mini_table();
        let rows = observations(Family::BothqbPassyds, &table);
        assert_eq!(rows.len(), 2 * usize::from(WEEKS));
        let week_two = rows
            .iter()
            .find(|row| row.game_id == game_id(2024, 2))
            .expect("2024 week 2");
        // BUF hosted week 2, so the home quarterback is Buffalo's.
        assert_eq!(week_two.anchor_team, "BUF");
        assert_eq!(
            week_two.subject_a,
            SubjectKey::Player(BUF_QB.into(), 2024),
            "leg A is the home quarterback"
        );
        assert_eq!(week_two.subject_b, SubjectKey::Player(KC_QB.into(), 2024));
        assert_eq!(week_two.x_a, 232.0);
        assert_eq!(week_two.x_b, 240.0);
        assert_eq!(week_two.covariate, Some(total_line(2)));
        assert_eq!(
            week_two.cluster_primary,
            SubjectKey::Team("BUF".into(), 2024).cluster_id()
        );
        assert_eq!(
            week_two.cluster_secondary,
            Some(SubjectKey::Team("KC".into(), 2024).cluster_id())
        );
        assert_ne!(
            week_two.cluster_primary,
            week_two.cluster_secondary.unwrap()
        );
    }

    /// Dropped, not replaced, all the way through to the family: KC's as-of
    /// Wr1 is inactive in 2024 week 8, so there is no `passyds_x_wr1recyds`
    /// row for KC that week — and BUF's row is untouched.
    #[test]
    fn a_dropped_wr1_removes_the_observation_rather_than_substituting() {
        let table = mini_table();
        let rows = observations(Family::PassydsXWr1recyds, &table);
        let missing = game_id(2024, 8);
        assert!(
            !rows
                .iter()
                .any(|row| row.game_id == missing && row.anchor_team == "KC"),
            "the KC anchor is absent, not filled with the day's leader"
        );
        assert!(
            rows.iter()
                .any(|row| row.game_id == missing && row.anchor_team == "BUF"),
            "the opponent's row is unaffected"
        );
        // Same for the Rb1 families.
        for family in [Family::Rb1rushXQbpassunder, Family::PassydsXRb1rush] {
            assert!(
                !observations(family, &table)
                    .iter()
                    .any(|row| row.game_id == missing && row.anchor_team == "KC"),
                "{} kept a row with no Rb1",
                family.key()
            );
        }
        // The team-total family needs no skill subject, so it survives.
        assert!(
            observations(Family::TeamtotalXPasstd, &table)
                .iter()
                .any(|row| row.game_id == missing && row.anchor_team == "KC")
        );
    }

    #[test]
    fn each_family_reads_its_own_covariate_off_the_row() {
        let table = mini_table();
        let anchor = |family: Family, week: u8| -> Observation {
            observations(family, &table)
                .into_iter()
                .find(|row| row.game_id == game_id(2024, week) && row.anchor_team == "KC")
                .unwrap_or_else(|| panic!("{} has a KC row in week {week}", family.key()))
        };
        let row = table.row(&game_id(2024, 5), "KC").expect("the KC row");
        assert_eq!(
            anchor(Family::TeamtotalXPasstd, 5).covariate,
            row.pass_td_share_asof
        );
        assert_eq!(
            anchor(Family::PassydsXWr1recyds, 5).covariate,
            row.wr1_target_share_asof
        );
        assert_eq!(
            anchor(Family::Rb1rushXQbpassunder, 5).covariate,
            row.anchor_spread
        );
        assert_eq!(
            anchor(Family::PassydsXRb1rush, 5).covariate,
            Some(spread_line(5))
        );
        // KC hosts in week 5, so the anchor spread is the schedule's own sign.
        assert_eq!(row.anchor_spread, Some(spread_line(5)));
    }

    /// A missing covariate keeps the row (§1.8's `all` stratum) and is counted.
    #[test]
    fn rows_with_no_covariate_are_kept_and_counted() {
        // 2024 alone: no prior season, so the first eight weeks have no
        // passing-TD share at all.
        let table = table_for(&[2024]);
        let rows = observations(Family::TeamtotalXPasstd, &table);
        assert_eq!(rows.len(), 2 * usize::from(WEEKS), "no row was dropped");
        let missing = missing_covariate(&rows);
        assert_eq!(missing, 2 * 8, "both teams, the first eight weeks");
        assert!(
            rows.iter()
                .filter(|row| row.covariate.is_none())
                .all(|row| row.week <= 8)
        );
        // With 2023 loaded as history, every 2024 row has a covariate — and
        // 2023's own opening weeks still do not, because the first season of
        // any range has nothing behind it.
        let with_history = observations(Family::TeamtotalXPasstd, &table_for(&[2023, 2024]));
        assert!(
            with_history
                .iter()
                .filter(|row| row.season == 2024)
                .all(|row| row.covariate.is_some())
        );
        assert_eq!(
            missing_covariate(&with_history),
            2 * 8,
            "2023's first eight"
        );
    }

    /// Hand-checked. KC hosts week 1 of 2024: it scored 21, allowed 20, so the
    /// margin is +1 against a closing spread of +2, giving `x_a = −1`; the game
    /// totalled 41 against a 41 line, giving `x_b = 0` — a push, which is kept.
    #[test]
    fn spread_x_total_values_are_the_market_residuals() {
        let table = mini_table();
        let rows = observations(Family::SpreadXTotal, &table);
        assert_eq!(rows.len(), 2 * usize::from(WEEKS));
        let opener = rows
            .iter()
            .find(|row| row.game_id == game_id(2024, 1))
            .expect("2024 week 1");
        assert_eq!(home_score(1), 21);
        assert_eq!(away_score(1), 20);
        assert_eq!(spread_line(1), 2.0);
        assert_eq!(total_line(1), 41.0);
        assert_eq!(opener.anchor_team, "KC");
        assert_eq!(opener.x_a, -1.0);
        assert_eq!(opener.x_b, 0.0, "a push is a legitimate zero residual");
        assert_eq!(opener.covariate, Some(41.0));
        assert_eq!(
            opener.subject_a,
            SubjectKey::Team("KC".into(), 2024),
            "the home team-season anchors it"
        );
        assert_eq!(opener.subject_a, opener.subject_b);
        assert_eq!(
            opener.cluster_primary,
            SubjectKey::Team("KC".into(), 2024).cluster_id()
        );
        assert!(opener.cluster_secondary.is_none());
    }

    /// The values agree with the convention pinned in [`crate::nflverse`]: the
    /// sign of `x_a` is exactly "the home side covered".
    #[test]
    fn the_spread_residual_agrees_with_home_covered() {
        let table = mini_table();
        for row in observations(Family::SpreadXTotal, &table) {
            let game = table
                .schedule
                .iter()
                .find(|game| game.game_id == row.game_id)
                .expect("its own schedule row");
            if let Some(covered) = game.home_covered() {
                assert_eq!(row.x_a > 0.0, covered, "{}", row.game_id);
            } else {
                assert_eq!(row.x_a, 0.0, "a push residual is exactly zero");
            }
            assert_eq!(row.x_b > 0.0, game.went_over().unwrap_or(false));
        }
    }

    /// `spread_x_total` reads the schedule, so it covers seasons the weekly
    /// stat files do not.
    #[test]
    fn spread_x_total_reaches_seasons_the_subject_table_does_not() {
        let table = crate::games::GameTable::build(
            parse(&format!(
                "{GAMES_HEADER}\n1999_01_MIN_ATL,1999,REG,1,1999-09-12,Sunday,MIN,17,ATL,14,-3,31,-4,49,,,,\n"
            )),
            Vec::new(),
            Vec::new(),
        );
        assert!(table.rows.is_empty(), "no weekly stats were supplied");
        let rows = observations(Family::SpreadXTotal, &table);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].season, 1999);
        assert_eq!(rows[0].anchor_team, "ATL");
        assert_eq!(rows[0].x_a, 1.0);
        assert_eq!(rows[0].x_b, -18.0);
        // Every other family is empty without a subject table.
        for family in ALL {
            if family != Family::SpreadXTotal {
                assert!(observations(family, &table).is_empty(), "{}", family.key());
            }
        }
    }

    /// A game with no closing line yields no market-centred row, rather than
    /// a row centred on zero.
    #[test]
    fn a_game_without_a_closing_line_yields_no_market_centred_row() {
        let table = crate::games::GameTable::build(
            parse(&format!(
                "{GAMES_HEADER}\n1999_01_MIN_ATL,1999,REG,1,1999-09-12,Sunday,MIN,17,ATL,14,-3,31,,,,,,\n"
            )),
            Vec::new(),
            Vec::new(),
        );
        assert!(observations(Family::SpreadXTotal, &table).is_empty());
    }

    /// The subjects on a same-team row are the two the legs name, and the
    /// cluster is the plan's unit for that family.
    #[test]
    fn same_team_subjects_and_clusters_are_the_plans_units() {
        let table = mini_table();
        let stack = observations(Family::PassydsXWr1recyds, &table)
            .into_iter()
            .find(|row| row.game_id == game_id(2024, 5) && row.anchor_team == "KC")
            .expect("a KC stack row");
        assert_eq!(stack.subject_a, SubjectKey::Player(KC_QB.into(), 2024));
        assert_eq!(stack.subject_b, SubjectKey::Player(KC_WR1.into(), 2024));
        assert_eq!(
            stack.cluster_primary,
            stack.subject_a.cluster_id(),
            "clustered on the quarterback-season"
        );
        assert!(stack.cluster_secondary.is_none());

        let team_total = observations(Family::TeamtotalXPasstd, &table)
            .into_iter()
            .find(|row| row.game_id == game_id(2024, 5) && row.anchor_team == "KC")
            .expect("a KC team-total row");
        assert_eq!(team_total.subject_a, SubjectKey::Team("KC".into(), 2024));
        assert_eq!(team_total.subject_b, SubjectKey::Player(KC_QB.into(), 2024));
        assert_eq!(
            team_total.cluster_primary,
            SubjectKey::Team("KC".into(), 2024).cluster_id(),
            "clustered on the team-season"
        );
    }

    /// Determinism: the same table twice gives byte-identical rows in the same
    /// order. Phase 3's atlas is meant to be bit-for-bit reproducible and this
    /// is the layer that could break it.
    #[test]
    fn observations_are_deterministic_and_in_table_order() {
        let table = mini_table();
        for family in ALL {
            let first = observations(family, &table);
            let second = observations(family, &table);
            let key = |rows: &[Observation]| -> Vec<(String, String, u64, u64)> {
                rows.iter()
                    .map(|row| {
                        (
                            row.game_id.clone(),
                            row.anchor_team.clone(),
                            row.x_a.to_bits(),
                            row.x_b.to_bits(),
                        )
                    })
                    .collect()
            };
            assert_eq!(key(&first), key(&second), "{}", family.key());
            let ordered: Vec<(u16, u8)> = first.iter().map(|row| (row.season, row.week)).collect();
            let mut sorted = ordered.clone();
            sorted.sort_unstable();
            assert_eq!(ordered, sorted, "{} is not in table order", family.key());
        }
    }
}
