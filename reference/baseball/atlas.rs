//! Dataset-aware measurement for the same-game-parlay correlation studies.
//!
//! [`crate::correlation`] is pure statistics — paired Bernoulli observations
//! in, exact φ coefficients out — and knows nothing about pitchers. This
//! module is the layer that knows the project's datasets: what a start log
//! carries, how a both-starter game yields two starter views, how a batter
//! line joins to the starter his side faced, and how a hand-collected SGP leg
//! description parses into a measurable threshold ([`LegSpec`]). Lifted out of
//! `examples/correlation_atlas.rs` so the rung-grid atlas and the exact-line
//! study (`examples/exact_line_phi.rs`) share one engine; the atlas's
//! committed CSV byte-for-byte reproducibility gates the lift.

use crate::correlation::{cluster_bootstrap_phi, era_indices, phi};
use crate::data::batters::BatterLineRow;
use crate::data::types::PitcherStartLog;
use chrono::{Datelike, NaiveDate};
use std::collections::{BTreeMap, HashMap};

/// Anything measurable has a pitcher-year cluster (the bootstrap's resample
/// unit) and a date (the era split's grouping key).
pub trait Obs {
    fn cluster(&self) -> u64;
    fn date(&self) -> NaiveDate;
}

/// One pitcher's season is the unit of independent observation.
pub fn cluster_key(pitcher_id: u32, year: i32) -> u64 {
    u64::from(pitcher_id) * 10_000 + year as u64
}

impl Obs for PitcherStartLog {
    fn cluster(&self) -> u64 {
        cluster_key(self.pitcher_id, self.game_date.year())
    }
    fn date(&self) -> NaiveDate {
        self.game_date
    }
}

/// One starter's view of a both-starter game: his line, the other starter's
/// line, and what his team's whole staff did to the opposing lineup (`None`
/// when the batter file does not cover the game — unknown, not zero).
pub struct ViewObs {
    pub strikeouts: u16,
    pub earned_runs: u16,
    pub their_earned_runs: u16,
    pub their_strikeouts: u16,
    pub opposing_team_ks: Option<u32>,
    pub opposing_team_hits: Option<u32>,
    pub personal_line: u16,
    pub cluster: u64,
    pub date: NaiveDate,
}

impl Obs for ViewObs {
    fn cluster(&self) -> u64 {
        self.cluster
    }
    fn date(&self) -> NaiveDate {
        self.date
    }
}

/// One batter's game line joined to the opposing starter his side faced.
pub struct BatterObs {
    pub starter_strikeouts: u16,
    pub starter_personal_line: u16,
    pub batter_strikeouts: u16,
    pub batter_hits: u16,
    pub plate_appearances: u16,
    pub cluster: u64,
    pub date: NaiveDate,
}

impl Obs for BatterObs {
    fn cluster(&self) -> u64 {
        self.cluster
    }
    fn date(&self) -> NaiveDate {
        self.date
    }
}

/// What [`measure`] found: the exact φ with the 2×2 levels it comes from, an
/// optional pitcher-year cluster-bootstrap interval, and per-season era
/// splits — the regime check that a correlation existing in only one season
/// is a story about that season, not about how legs co-move.
pub struct Measurement {
    pub n: usize,
    pub p_a: f64,
    pub p_b: f64,
    pub joint: f64,
    pub phi: f64,
    pub interval: Option<(f64, f64)>,
    pub eras: BTreeMap<i32, Option<f64>>,
}

/// Filters `obs` to rows where both legs exist, then measures: the exact φ,
/// per-season era splits, and — when `bootstrap` is `Some((reps, seed))` —
/// the pitcher-year cluster-bootstrap interval. Returns `None` when the
/// filtered set is empty or either leg is degenerate; the caller decides
/// whether that is a skipped grid cell or a skipped collection row.
///
/// A leg pair is `None` — and the observation drops out — whenever a leg is
/// undefined for the observation rather than false: an outpitch tie is not a
/// loss, and a game the batter file does not cover is not a zero.
pub fn measure<T: Obs>(
    obs: &[T],
    legs: impl Fn(&T) -> Option<(bool, bool)>,
    bootstrap: Option<(usize, u64)>,
) -> Option<Measurement> {
    let mut pairs = Vec::new();
    let mut clusters = Vec::new();
    let mut dates = Vec::new();
    for observation in obs {
        if let Some(pair) = legs(observation) {
            pairs.push(pair);
            clusters.push(observation.cluster());
            dates.push(observation.date());
        }
    }
    let (value, p_a, p_b, joint) = phi(&pairs)?;
    let interval =
        bootstrap.and_then(|(reps, seed)| cluster_bootstrap_phi(&pairs, &clusters, reps, seed));
    let eras = era_indices(&dates)
        .into_iter()
        .map(|(year, indices)| {
            let subset: Vec<(bool, bool)> = indices.iter().map(|&i| pairs[i]).collect();
            (year, phi(&subset).map(|(era_phi, _, _, _)| era_phi))
        })
        .collect();
    Some(Measurement {
        n: pairs.len(),
        p_a,
        p_b,
        joint,
        phi: value,
        interval,
        eras,
    })
}

/// Baseball notation to outs: 5.1 IP is 16 outs, not 5.1 × 3. The fractional
/// digit counts whole outs past the inning, and the f64 dust around .1/.2
/// rounds away cleanly at one decimal place.
pub fn outs_from_baseball_ip(innings_pitched: f64) -> u32 {
    let full = innings_pitched.trunc() as u32;
    let partial = ((innings_pitched - f64::from(full)) * 10.0).round() as u32;
    full * 3 + partial
}

/// Each pitcher-season's market-line proxy: the mean K/start rounded to the
/// nearest integer, computed strictly within the season (a pitcher's 2026
/// line knows nothing about his 2023 form). Takes (pitcher_id, year,
/// strikeouts) triples so the rule is testable without fabricating whole
/// start logs.
pub fn personal_lines(starts: &[(u32, i32, u16)]) -> HashMap<(u32, i32), u16> {
    let mut totals: HashMap<(u32, i32), (u64, u64)> = HashMap::new();
    for &(pitcher, year, strikeouts) in starts {
        let entry = totals.entry((pitcher, year)).or_default();
        entry.0 += u64::from(strikeouts);
        entry.1 += 1;
    }
    totals
        .into_iter()
        .map(|(key, (ks, n))| (key, (ks as f64 / n as f64).round() as u16))
        .collect()
}

/// Per-(game, team) batting totals from the batter-lines file: (strikeouts,
/// hits). The starter-view join reads the *opposing* side's totals — what the
/// whole staff did to the lineup the starter faced, which is what "team
/// strikeouts thrown" and "team hits" props price.
pub fn side_totals(batter_rows: &[BatterLineRow]) -> HashMap<(u32, u32), (u32, u32)> {
    let mut totals: HashMap<(u32, u32), (u32, u32)> = HashMap::new();
    for row in batter_rows {
        let entry = totals.entry((row.game_pk, row.team_id)).or_default();
        entry.0 += u32::from(row.strikeouts);
        entry.1 += u32::from(row.hits);
    }
    totals
}

/// Starter-views of both-starter games. Each game with both starters in the
/// file contributes two observations, one from each starter's perspective;
/// single-starter games have nothing to cross-join against and are skipped.
/// Returns the views and how many both-starter games they came from (the
/// callers print that count).
pub fn starter_views(
    starts: &[PitcherStartLog],
    totals: &HashMap<(u32, u32), (u32, u32)>,
    lines: &HashMap<(u32, i32), u16>,
) -> (Vec<ViewObs>, usize) {
    let mut by_game: HashMap<u32, Vec<usize>> = HashMap::new();
    for (index, start) in starts.iter().enumerate() {
        by_game.entry(start.game_pk).or_default().push(index);
    }
    let mut views: Vec<ViewObs> = Vec::new();
    let mut both_starter_games = 0usize;
    for indices in by_game.values() {
        if indices.len() != 2 || starts[indices[0]].team_id == starts[indices[1]].team_id {
            continue;
        }
        both_starter_games += 1;
        for (me, them) in [(indices[0], indices[1]), (indices[1], indices[0])] {
            let (my, their) = (&starts[me], &starts[them]);
            let side = totals.get(&(my.game_pk, their.team_id)).copied();
            views.push(ViewObs {
                strikeouts: my.strikeouts,
                earned_runs: my.earned_runs,
                their_earned_runs: their.earned_runs,
                their_strikeouts: their.strikeouts,
                opposing_team_ks: side.map(|t| t.0),
                opposing_team_hits: side.map(|t| t.1),
                personal_line: lines[&(my.pitcher_id, my.game_date.year())],
                cluster: my.cluster(),
                date: my.game_date,
            });
        }
    }
    (views, both_starter_games)
}

/// Batter-start pairs: every batter's line joined to the opposing starter his
/// side faced (same game, different team). The betting question is whether
/// one hitter's night tells you anything about the starter's K total beyond
/// the starter himself.
pub fn batter_start_pairs(
    starts: &[PitcherStartLog],
    batter_rows: &[BatterLineRow],
    lines: &HashMap<(u32, i32), u16>,
) -> Vec<BatterObs> {
    let mut by_game: HashMap<u32, Vec<usize>> = HashMap::new();
    for (index, start) in starts.iter().enumerate() {
        by_game.entry(start.game_pk).or_default().push(index);
    }
    let mut side_batters: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (index, row) in batter_rows.iter().enumerate() {
        side_batters
            .entry((row.game_pk, row.team_id))
            .or_default()
            .push(index);
    }
    let mut pairs: Vec<BatterObs> = Vec::new();
    for indices in by_game.values() {
        if indices.len() != 2 || starts[indices[0]].team_id == starts[indices[1]].team_id {
            continue;
        }
        for (me, them) in [(indices[0], indices[1]), (indices[1], indices[0])] {
            let (my, their) = (&starts[me], &starts[them]);
            let Some(lineup) = side_batters.get(&(my.game_pk, their.team_id)) else {
                continue;
            };
            for &index in lineup {
                let row = &batter_rows[index];
                pairs.push(BatterObs {
                    starter_strikeouts: my.strikeouts,
                    starter_personal_line: lines[&(my.pitcher_id, my.game_date.year())],
                    batter_strikeouts: row.strikeouts,
                    batter_hits: row.hits,
                    plate_appearances: row.plate_appearances,
                    cluster: my.cluster(),
                    date: my.game_date,
                });
            }
        }
    }
    pairs
}

/// One collected SGP leg, reduced to the exact threshold it prices.
///
/// Half-lines resolve to the integer boundary the bet actually needs: over
/// X.5 wins at ⌈X.5⌉ or more, under X.5 at ⌊X.5⌋ or fewer. Whole lines can
/// push and therefore price a different bet, so [`parse_leg`] refuses them
/// rather than guess a side. Every variant names the observation it is
/// evaluated against; the `eval_on_*` methods are the predicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegSpec {
    /// "Sanchez over 6.5 K" — the starter strikes out at least `over`.
    StarterK { over: u16 },
    /// "Wisenhunt under 3.5 K" — the starter strikes out at most `under`.
    StarterKUnder { under: u16 },
    /// "under 1.5 er" — the starter allows at most `under` earned runs.
    StarterER { under: u16 },
    /// "hits allowed under 5.5".
    StarterHitsAllowed { under: u16 },
    /// "over 16.5 outs" — the starter records at least `over` outs.
    StarterOuts { over: u32 },
    /// "team strikeouts over 9.5" / "Strikeouts Thrown over 9.5" / "Phillies
    /// Ks thrown over 9.5" — the whole staff's Ks against the lineup the
    /// starter faced.
    TeamK { over: u32 },
    /// "Cubs Ks thrown under 8.5" — the under-quadrant form of the same leg.
    TeamKUnder { under: u32 },
    /// "team hits under 8.5" — the opposing lineup's total hits.
    TeamHits { under: u32 },
    /// "Cruz 1+ K" — one batter's own strikeouts, at least `over`.
    BatterK { over: u16 },
    /// One batter's hits, at most `under`.
    BatterHits { under: u16 },
    /// F5/ML proxy: my ER < their ER. Tied views drop out of the 2×2 — a tie
    /// is not a loss. Full-game starter ER stands in for the first-five
    /// frame, a documented approximation: the dataset has no F5 splits.
    Outpitch,
    /// Both starters' earned runs combined, at most `under`.
    CombinedER { under: u16 },
    /// Their starter's Ks — the indep control: the two starters' lines share
    /// a game but no mechanism, so this pairing must measure ≈ 0.
    DuelK { over: u16 },
}

impl LegSpec {
    /// The atlas-style threshold label ("k>=7", "er<=1"), used for output
    /// columns and error messages.
    pub fn label(self) -> String {
        match self {
            Self::StarterK { over } => format!("k>={over}"),
            Self::StarterKUnder { under } => format!("k<={under}"),
            Self::StarterER { under } => format!("er<={under}"),
            Self::StarterHitsAllowed { under } => format!("hits<={under}"),
            Self::StarterOuts { over } => format!("outs>={over}"),
            Self::TeamK { over } => format!("team_k>={over}"),
            Self::TeamKUnder { under } => format!("team_k<={under}"),
            Self::TeamHits { under } => format!("team_hits<={under}"),
            Self::BatterK { over } => format!("batter_k>={over}"),
            Self::BatterHits { under } => format!("batter_hits<={under}"),
            Self::Outpitch => "my_er<their_er".to_string(),
            Self::CombinedER { under } => format!("combined_er<={under}"),
            Self::DuelK { over } => format!("their_k>={over}"),
        }
    }

    /// Starter legs against a single start log. `None` for legs whose inputs
    /// a start log does not carry (team totals, the opposing starter, the
    /// batters he faced) — the caller routes those to the right observation
    /// set instead of reading `None` as a lost bet.
    pub fn eval_on_start(&self, start: &PitcherStartLog) -> Option<bool> {
        match *self {
            Self::StarterK { over } => Some(start.strikeouts >= over),
            Self::StarterKUnder { under } => Some(start.strikeouts <= under),
            Self::StarterER { under } => Some(start.earned_runs <= under),
            Self::StarterHitsAllowed { under } => Some(start.hits <= under),
            Self::StarterOuts { over } => {
                Some(outs_from_baseball_ip(start.innings_pitched) >= over)
            }
            _ => None,
        }
    }

    /// View legs against a starter's view of a both-starter game. `None` when
    /// the leg is undefined for the view — an outpitch tie is not a loss, and
    /// a game the batter file does not cover is not a zero.
    pub fn eval_on_view(&self, view: &ViewObs) -> Option<bool> {
        match *self {
            Self::StarterK { over } => Some(view.strikeouts >= over),
            Self::StarterKUnder { under } => Some(view.strikeouts <= under),
            Self::StarterER { under } => Some(view.earned_runs <= under),
            Self::TeamK { over } => Some(view.opposing_team_ks? >= over),
            Self::TeamKUnder { under } => Some(view.opposing_team_ks? <= under),
            Self::TeamHits { under } => Some(view.opposing_team_hits? <= under),
            Self::Outpitch => (view.earned_runs != view.their_earned_runs)
                .then_some(view.earned_runs < view.their_earned_runs),
            Self::CombinedER { under } => Some(view.earned_runs + view.their_earned_runs <= under),
            Self::DuelK { over } => Some(view.their_strikeouts >= over),
            _ => None,
        }
    }

    /// Batter legs against a batter-start pair. The starter's own K leg is
    /// defined here too — the pair carries his line for exactly this join.
    pub fn eval_on_batter(&self, obs: &BatterObs) -> Option<bool> {
        match *self {
            Self::StarterK { over } => Some(obs.starter_strikeouts >= over),
            Self::BatterK { over } => Some(obs.batter_strikeouts >= over),
            Self::BatterHits { under } => Some(obs.batter_hits <= under),
            _ => None,
        }
    }
}

/// Parses one collected leg description into its [`LegSpec`], cross-checked
/// against the collection row's `family` tag.
///
/// The grammar is the one the collection sheet actually speaks: "Name over
/// 6.5 K", "Name under 1.5 er", "Name ER allowed under 1.5", "Name 1+ K",
/// "Team Strikeouts Thrown over 9.5", "Dodgers F5 ml". Anything else is
/// refused loudly — never guessed:
///
/// - `unmeasurable:` the bet is real but the dataset carries no column for
///   it. Win legs are the case in point, and the whole `k_x_win` family is
///   unmeasurable for the same reason.
/// - `whole line (push-capable):` a whole-number line can push; it prices a
///   different bet than the half-line ladder the thresholds assume.
/// - `unsupported:` parseable and measurable in principle, but the enum has
///   no variant for it (a team-hits *over*, say — the over-quadrant quadrant
///   flips both legs, and [`LegSpec`] carries the under form only).
/// - `unparseable:` nothing in the grammar matched.
/// - `family mismatch:` / `unknown family:` the family tag cross-check
///   failed — `k_x_er` expects a starter-K leg and an ER leg, and a row
///   disagreeing with its own tag is a collection error, not a measurement.
pub fn parse_leg(family: &str, desc: &str) -> Result<LegSpec, String> {
    if family == "k_x_win" {
        return Err(format!(
            "unmeasurable: the k_x_win family prices a win, and the dataset has no win column ({desc:?})"
        ));
    }
    let tokens: Vec<String> = desc.split_whitespace().map(|t| t.to_lowercase()).collect();
    if tokens.is_empty() {
        return Err("unparseable: empty leg description".to_string());
    }
    let has = |word: &str| tokens.iter().any(|token| token == word);
    if has("win") {
        return Err(format!(
            "unmeasurable: a win leg has no column in the dataset ({desc:?})"
        ));
    }
    let leg = if has("f5") && has("ml") {
        LegSpec::Outpitch
    } else if has("combined") && has("er") {
        LegSpec::CombinedER {
            under: under_line(&tokens, desc)?,
        }
    } else if has("thrown") || (has("team") && (has("strikeouts") || has("ks"))) {
        // Team-level Ks: "Strikeouts Thrown over 9.5", "team strikeouts over
        // 9.5", "Phillies Ks thrown over 9.5". Checked before the bare "K" so
        // a staff total is never read as a starter leg.
        if has("over") {
            LegSpec::TeamK {
                over: over_line32(&tokens, desc)?,
            }
        } else if has("under") {
            LegSpec::TeamKUnder {
                under: under_line32(&tokens, desc)?,
            }
        } else {
            return Err(format!(
                "unparseable: no direction in team-K leg ({desc:?})"
            ));
        }
    } else if has("team") && has("hits") {
        if !has("under") {
            return Err(format!(
                "unsupported: team hits has no over form — flip both legs of the over-quadrant instead ({desc:?})"
            ));
        }
        LegSpec::TeamHits {
            under: under_line32(&tokens, desc)?,
        }
    } else if has("hits") && has("allowed") {
        if !has("under") {
            return Err(format!(
                "unsupported: hits allowed has no over form ({desc:?})"
            ));
        }
        LegSpec::StarterHitsAllowed {
            under: under_line(&tokens, desc)?,
        }
    } else if has("hits") {
        if !has("under") {
            return Err(format!(
                "unsupported: batter hits has no over form ({desc:?})"
            ));
        }
        LegSpec::BatterHits {
            under: under_line(&tokens, desc)?,
        }
    } else if has("outs") {
        if !has("over") {
            return Err(format!("unsupported: outs has no under form ({desc:?})"));
        }
        LegSpec::StarterOuts {
            over: over_line32(&tokens, desc)?,
        }
    } else if has("er") {
        if !has("under") {
            return Err(format!("unsupported: ER has no over form ({desc:?})"));
        }
        LegSpec::StarterER {
            under: under_line(&tokens, desc)?,
        }
    } else if has("k") {
        // "Cruz 1+ K" is the batter form; "Name over 6.5 K" the starter form.
        if let Some(plus) = tokens
            .iter()
            .find_map(|token| token.strip_suffix('+'))
            .and_then(|prefix| prefix.parse::<u16>().ok())
        {
            LegSpec::BatterK { over: plus }
        } else if has("over") {
            LegSpec::StarterK {
                over: over_line(&tokens, desc)?,
            }
        } else if has("under") {
            LegSpec::StarterKUnder {
                under: under_line(&tokens, desc)?,
            }
        } else {
            return Err(format!("unparseable: no direction in K leg ({desc:?})"));
        }
    } else {
        return Err(format!(
            "unparseable: no known market in leg description ({desc:?})"
        ));
    };
    family_check(family, leg)
}

/// The family tag says what a row is *for*; a leg that cannot belong to its
/// row's family is a collection error, not a measurement.
fn family_check(family: &str, leg: LegSpec) -> Result<LegSpec, String> {
    let allowed = match family {
        // Both legs of the indep control are starter K-overs; the second is
        // the opposing starter's, which the caller converts to a DuelK.
        "indep" => matches!(leg, LegSpec::StarterK { .. }),
        // Under-quadrant rows pair the under forms of the same legs.
        "k_x_teamk" => matches!(
            leg,
            LegSpec::StarterK { .. }
                | LegSpec::StarterKUnder { .. }
                | LegSpec::TeamK { .. }
                | LegSpec::TeamKUnder { .. }
        ),
        "k_x_er" => matches!(leg, LegSpec::StarterK { .. } | LegSpec::StarterER { .. }),
        "k_x_hitsallowed" => matches!(
            leg,
            LegSpec::StarterK { .. } | LegSpec::StarterHitsAllowed { .. }
        ),
        // The dominance quadrant proper: starter K over × opponent team hits
        // under. Distinct from `k_x_hitsallowed`, which prices the starter's
        // own hits-allowed prop.
        "k_x_teamhitsunder" => {
            matches!(leg, LegSpec::StarterK { .. } | LegSpec::TeamHits { .. })
        }
        "k_x_outs" => matches!(leg, LegSpec::StarterK { .. } | LegSpec::StarterOuts { .. }),
        "k_x_batterk" => matches!(leg, LegSpec::StarterK { .. } | LegSpec::BatterK { .. }),
        "k_x_f5ml" => matches!(leg, LegSpec::StarterK { .. } | LegSpec::Outpitch),
        "over-quadrant" => matches!(
            leg,
            LegSpec::StarterKUnder { .. } | LegSpec::TeamHits { .. }
        ),
        other => return Err(format!("unknown family: {other:?}")),
    };
    if allowed {
        Ok(leg)
    } else {
        Err(format!(
            "family mismatch: {family} does not price a {} leg",
            leg.label()
        ))
    }
}

/// The number following "over"/"under", validated as a half-line. Half-lines
/// are exactly representable in f64, so the .5 check is exact.
fn line_value(tokens: &[String], direction: &str, desc: &str) -> Result<f64, String> {
    let position = tokens
        .iter()
        .position(|token| token == direction)
        .ok_or_else(|| format!("unparseable: no {direction:?} in leg description ({desc:?})"))?;
    let value: f64 = tokens[position + 1..]
        .iter()
        .find_map(|token| token.parse().ok())
        .ok_or_else(|| {
            format!("unparseable: no number after {direction:?} in leg description ({desc:?})")
        })?;
    if value.fract() == 0.0 {
        return Err(format!(
            "whole line (push-capable): {direction} {value} in ({desc:?})"
        ));
    }
    if value.fract() != 0.5 {
        return Err(format!(
            "unparseable: not a half line ({direction} {value}) in ({desc:?})"
        ));
    }
    Ok(value)
}

/// Over X.5 wins at ⌈X.5⌉ or more.
fn over_line(tokens: &[String], desc: &str) -> Result<u16, String> {
    Ok((line_value(tokens, "over", desc)? + 0.5) as u16)
}

fn over_line32(tokens: &[String], desc: &str) -> Result<u32, String> {
    Ok((line_value(tokens, "over", desc)? + 0.5) as u32)
}

/// Under X.5 wins at ⌊X.5⌋ or fewer.
fn under_line(tokens: &[String], desc: &str) -> Result<u16, String> {
    Ok((line_value(tokens, "under", desc)? - 0.5) as u16)
}

fn under_line32(tokens: &[String], desc: &str) -> Result<u32, String> {
    Ok((line_value(tokens, "under", desc)? - 0.5) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseball_innings_convert_to_outs() {
        // The fractional digit is outs past the inning, not a decimal: 5.1 IP
        // is 16 outs, and the deep-start rung (6 full innings) is 18.
        assert_eq!(outs_from_baseball_ip(5.0), 15);
        assert_eq!(outs_from_baseball_ip(5.1), 16);
        assert_eq!(outs_from_baseball_ip(5.2), 17);
        assert_eq!(outs_from_baseball_ip(6.0), 18);
        assert_eq!(outs_from_baseball_ip(0.2), 2);
        assert_eq!(outs_from_baseball_ip(9.0), 27);
    }

    #[test]
    fn personal_line_is_a_within_season_mean_rounded() {
        let starts = vec![
            (1u32, 2025i32, 4u16),
            (1, 2025, 5),
            (1, 2025, 6),
            (2, 2025, 4),
            (2, 2025, 4),
            (2, 2025, 5),
            (3, 2025, 4),
            (3, 2025, 5),
            (1, 2026, 9),
        ];
        let lines = personal_lines(&starts);
        assert_eq!(lines[&(1, 2025)], 5, "mean 5.0");
        assert_eq!(lines[&(2, 2025)], 4, "mean 4.33 rounds down");
        assert_eq!(lines[&(3, 2025)], 5, "mean 4.5 rounds up");
        assert_eq!(lines[&(1, 2026)], 9, "a new season gets its own line");
    }

    struct TestObs {
        legs: Option<(bool, bool)>,
        cluster: u64,
        date: NaiveDate,
    }

    impl Obs for TestObs {
        fn cluster(&self) -> u64 {
            self.cluster
        }
        fn date(&self) -> NaiveDate {
            self.date
        }
    }

    #[test]
    fn measure_drops_rows_with_no_leg_pair_and_splits_eras() {
        let day = |year| NaiveDate::from_ymd_opt(year, 6, 1).unwrap();
        let obs = vec![
            TestObs {
                legs: Some((true, true)),
                cluster: 1,
                date: day(2025),
            },
            TestObs {
                legs: Some((true, false)),
                cluster: 2,
                date: day(2025),
            },
            TestObs {
                legs: Some((false, true)),
                cluster: 3,
                date: day(2026),
            },
            TestObs {
                legs: Some((false, false)),
                cluster: 4,
                date: day(2026),
            },
            // An outpitch tie or an uncovered game: excluded, not a loss.
            TestObs {
                legs: None,
                cluster: 5,
                date: day(2026),
            },
        ];
        let m = measure(&obs, |o| o.legs, None).unwrap();
        assert_eq!(m.n, 4, "the None row must not count");
        // p_a = p_b = 0.5, joint = 0.25 → φ = 0 exactly.
        assert_eq!(m.phi, 0.0);
        assert!(m.interval.is_none(), "no bootstrap requested, no interval");
        assert_eq!(m.eras.len(), 2, "one φ per season with data");
    }

    #[test]
    fn measure_is_none_on_empty_or_degenerate_sets() {
        let day = NaiveDate::from_ymd_opt(2025, 6, 1).unwrap();
        assert!(measure::<TestObs>(&[], |_| None, None).is_none());
        // Every leg pair identical on one side: a constant leg has no
        // correlation with anything.
        let obs = vec![
            TestObs {
                legs: Some((true, true)),
                cluster: 1,
                date: day,
            },
            TestObs {
                legs: Some((true, false)),
                cluster: 2,
                date: day,
            },
        ];
        assert!(measure(&obs, |o| o.legs, None).is_none());
    }

    // -- Parser: every desc string in data/processed/sgp_prices.csv. --------

    /// The live collection sheet's leg descriptions, pinned against their
    /// parses. The sweep test below re-reads the sheet itself so a newly
    /// collected row with a new desc shape fails here until it is pinned.
    const SHEET_DESCS: &[(&str, &str, Option<LegSpec>)] = &[
        (
            "indep",
            "Wacha over 4.5 K",
            Some(LegSpec::StarterK { over: 5 }),
        ),
        (
            "indep",
            "Barnett over 3.5 K",
            Some(LegSpec::StarterK { over: 4 }),
        ),
        (
            "indep",
            "Ashcraft over 5.5 K",
            Some(LegSpec::StarterK { over: 6 }),
        ),
        (
            "indep",
            "Montero over 3.5 K",
            Some(LegSpec::StarterK { over: 4 }),
        ),
        (
            "indep",
            "Harrison over 5.5 K",
            Some(LegSpec::StarterK { over: 6 }),
        ),
        (
            "indep",
            "Miller over 5.5 K",
            Some(LegSpec::StarterK { over: 6 }),
        ),
        (
            "k_x_teamk",
            "Sanchez over 6.5 K",
            Some(LegSpec::StarterK { over: 7 }),
        ),
        (
            "k_x_teamk",
            "Phillies Strikeouts Thrown over 9.5 K",
            Some(LegSpec::TeamK { over: 10 }),
        ),
        (
            "k_x_teamk",
            "Rodon over 4.5 K",
            Some(LegSpec::StarterK { over: 5 }),
        ),
        (
            "k_x_teamk",
            "Orioles team strikeouts over 9.5",
            Some(LegSpec::TeamK { over: 10 }),
        ),
        (
            "k_x_teamk",
            "Lauer over 3.5 K",
            Some(LegSpec::StarterK { over: 4 }),
        ),
        (
            "k_x_teamk",
            "Rockies team strikeouts over 7.5",
            Some(LegSpec::TeamK { over: 8 }),
        ),
        // The collector's variant spellings: "Ks thrown" for the staff total,
        // and the under forms for the under-quadrant rows.
        (
            "k_x_teamk",
            "Phillies Ks thrown over 9.5",
            Some(LegSpec::TeamK { over: 10 }),
        ),
        (
            "k_x_teamk",
            "Yankees Ks thrown over 8.5",
            Some(LegSpec::TeamK { over: 9 }),
        ),
        (
            "k_x_teamk",
            "Holmes under 4.5 K",
            Some(LegSpec::StarterKUnder { under: 4 }),
        ),
        (
            "k_x_teamk",
            "Cubs Ks thrown under 8.5",
            Some(LegSpec::TeamKUnder { under: 8 }),
        ),
        // The dominance-quadrant team-hits rows: starter K over × opponent
        // team hits under.
        (
            "k_x_teamhitsunder",
            "Snell over 8.5 K",
            Some(LegSpec::StarterK { over: 9 }),
        ),
        (
            "k_x_teamhitsunder",
            "Pirates team hits under 7.5",
            Some(LegSpec::TeamHits { under: 7 }),
        ),
        (
            "k_x_teamhitsunder",
            "Padres team hits under 8.5",
            Some(LegSpec::TeamHits { under: 8 }),
        ),
        (
            "k_x_er",
            "Freeland under 3.5 er",
            Some(LegSpec::StarterER { under: 3 }),
        ),
        (
            "k_x_outs",
            "Sasaki over 16.5 outs",
            Some(LegSpec::StarterOuts { over: 17 }),
        ),
        (
            "k_x_er",
            "Sanchez over 6.5 K",
            Some(LegSpec::StarterK { over: 7 }),
        ),
        (
            "k_x_er",
            "Sanchez ER allowed under 1.5",
            Some(LegSpec::StarterER { under: 1 }),
        ),
        (
            "k_x_er",
            "Valdez over 5.5 K",
            Some(LegSpec::StarterK { over: 6 }),
        ),
        (
            "k_x_er",
            "Rodon over 4.5 K",
            Some(LegSpec::StarterK { over: 5 }),
        ),
        (
            "k_x_er",
            "Rodon under 1.5 er",
            Some(LegSpec::StarterER { under: 1 }),
        ),
        (
            "k_x_er",
            "Perkins over 3.5 K",
            Some(LegSpec::StarterK { over: 4 }),
        ),
        (
            "k_x_er",
            "Perkins under 2.5 er",
            Some(LegSpec::StarterER { under: 2 }),
        ),
        (
            "k_x_hitsallowed",
            "Wacha over 4.5 K",
            Some(LegSpec::StarterK { over: 5 }),
        ),
        (
            "k_x_hitsallowed",
            "Wacha hits allowed under 5.5",
            Some(LegSpec::StarterHitsAllowed { under: 5 }),
        ),
        (
            "k_x_outs",
            "Wacha over 4.5 K",
            Some(LegSpec::StarterK { over: 5 }),
        ),
        (
            "k_x_outs",
            "Wacha outs over 18.5",
            Some(LegSpec::StarterOuts { over: 19 }),
        ),
        (
            "k_x_outs",
            "Lauer over 3.5 K",
            Some(LegSpec::StarterK { over: 4 }),
        ),
        (
            "k_x_outs",
            "Lauer over 16.5 outs",
            Some(LegSpec::StarterOuts { over: 17 }),
        ),
        (
            "k_x_batterk",
            "Sanchez over 6.5 K",
            Some(LegSpec::StarterK { over: 7 }),
        ),
        (
            "k_x_batterk",
            "Sanoja 1+ K",
            Some(LegSpec::BatterK { over: 1 }),
        ),
        (
            "k_x_batterk",
            "Valdez over 5.5 K",
            Some(LegSpec::StarterK { over: 6 }),
        ),
        (
            "k_x_batterk",
            "Davis 1+ K",
            Some(LegSpec::BatterK { over: 1 }),
        ),
        (
            "k_x_batterk",
            "Montero over 3.5 K",
            Some(LegSpec::StarterK { over: 4 }),
        ),
        (
            "k_x_batterk",
            "Cruz 1+ K",
            Some(LegSpec::BatterK { over: 1 }),
        ),
        (
            "k_x_batterk",
            "Ashcraft over 5.5 K",
            Some(LegSpec::StarterK { over: 6 }),
        ),
        (
            "k_x_batterk",
            "Torkelson 1+ K",
            Some(LegSpec::BatterK { over: 1 }),
        ),
        (
            "k_x_f5ml",
            "Snell over 6.5 K",
            Some(LegSpec::StarterK { over: 7 }),
        ),
        ("k_x_f5ml", "Dodgers F5 ml", Some(LegSpec::Outpitch)),
        (
            "k_x_f5ml",
            "Lauer over 3.5 K",
            Some(LegSpec::StarterK { over: 4 }),
        ),
        (
            "over-quadrant",
            "Wisenhunt under 3.5 K",
            Some(LegSpec::StarterKUnder { under: 3 }),
        ),
        // The k_x_win row: both descs are unmeasurable — the family prices a
        // win, and the dataset has no win column.
        ("k_x_win", "Snell over 6.5 K", None),
        ("k_x_win", "Snell to record a win", None),
        // Team hits *over* is a real line the enum carries no variant for.
        ("over-quadrant", "Guardians team hits over 8.5", None),
    ];

    #[test]
    fn parser_covers_every_desc_in_the_sheet() {
        for &(family, desc, expected) in SHEET_DESCS {
            match expected {
                Some(leg) => assert_eq!(
                    parse_leg(family, desc).as_ref().ok(),
                    Some(&leg),
                    "{family} / {desc:?}"
                ),
                None => assert!(
                    parse_leg(family, desc).is_err(),
                    "{family} / {desc:?} must be refused"
                ),
            }
        }
    }

    /// Re-reads the actual sheet and confirms the pinned table above still
    /// covers it: every desc parses, or fails with one of the sanctioned
    /// reasons (unmeasurable, unsupported, or the one empty desc of the
    /// not-offered ER row).
    #[test]
    fn parser_sweep_matches_the_committed_sheet() {
        #[derive(serde::Deserialize)]
        struct Row {
            family: String,
            leg1_desc: String,
            leg2_desc: String,
        }
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../data/processed/sgp_prices.csv"
        );
        let mut reader = csv::ReaderBuilder::new()
            .trim(csv::Trim::All)
            .from_path(path)
            .unwrap();
        let (mut ok, mut unmeasurable, mut unsupported, mut empty) = (0, 0, 0, 0);
        for row in reader.deserialize::<Row>() {
            let row = row.unwrap();
            for desc in [&row.leg1_desc, &row.leg2_desc] {
                match parse_leg(&row.family, desc) {
                    Ok(_) => ok += 1,
                    Err(error) if error.starts_with("unmeasurable:") => unmeasurable += 1,
                    Err(error) if error.starts_with("unsupported:") => unsupported += 1,
                    Err(_) if desc.is_empty() => empty += 1,
                    Err(error) => panic!("{} / {desc:?}: unexpected refusal {error}", row.family),
                }
            }
        }
        // 64 descs: 60 parse, the k_x_win row's two are unmeasurable, the
        // team-hits-over is unsupported, and the not-offered ER row's empty
        // leg2 is unparseable.
        assert_eq!((ok, unmeasurable, unsupported, empty), (60, 2, 1, 1));
    }

    #[test]
    fn parser_refuses_garbage_whole_lines_and_mismatches_loudly() {
        let error = |family, desc| parse_leg(family, desc).unwrap_err();
        assert!(error("indep", "asdkjf").starts_with("unparseable:"));
        assert!(error("indep", "").starts_with("unparseable:"));
        assert!(error("indep", "Wacha over 6 K").starts_with("whole line (push-capable):"));
        assert!(error("indep", "Wacha over 6.3 K").starts_with("unparseable:"));
        // A win leg is unmeasurable in any family, not just k_x_win.
        assert!(error("k_x_f5ml", "Snell to record a win").starts_with("unmeasurable:"));
        assert!(error("k_x_win", "Snell over 6.5 K").starts_with("unmeasurable:"));
        // The family tag cross-check: an ER leg does not belong in k_x_teamk.
        assert!(error("k_x_teamk", "Rodon under 1.5 er").starts_with("family mismatch:"));
        assert!(parse_leg("k_x_batterk", "Sanchez over 6.5 K").is_ok());
        assert!(error("nonsense", "Wacha over 4.5 K").starts_with("unknown family:"));
    }

    // -- Predicates: boundary values per LegSpec variant. -------------------

    fn start(
        strikeouts: u16,
        earned_runs: u16,
        hits: u16,
        innings_pitched: f64,
    ) -> PitcherStartLog {
        PitcherStartLog {
            pitcher_id: 1,
            pitcher_name: "Test Arm".into(),
            game_pk: 1,
            game_date: NaiveDate::from_ymd_opt(2025, 6, 1).unwrap(),
            team_id: 1,
            opponent_id: 2,
            opponent_name: None,
            home: true,
            innings_pitched,
            strikeouts,
            walks: 2,
            home_runs: 0,
            earned_runs,
            hits,
            batters_faced: 22,
            pitch_count: None,
            ground_ball_pct: None,
            avg_fb_velo: None,
            historical_park_factor: None,
            historical_opponent_offense: None,
            pitch_hand: None,
            opponent_batters: Vec::new(),
            pitches: None,
            swinging_strikes: None,
            called_strikes: None,
        }
    }

    fn view(
        strikeouts: u16,
        earned_runs: u16,
        their_earned_runs: u16,
        their_strikeouts: u16,
        opposing_team_ks: Option<u32>,
        opposing_team_hits: Option<u32>,
    ) -> ViewObs {
        ViewObs {
            strikeouts,
            earned_runs,
            their_earned_runs,
            their_strikeouts,
            opposing_team_ks,
            opposing_team_hits,
            personal_line: 5,
            cluster: 1,
            date: NaiveDate::from_ymd_opt(2025, 6, 1).unwrap(),
        }
    }

    fn batter(starter_strikeouts: u16, batter_strikeouts: u16, batter_hits: u16) -> BatterObs {
        BatterObs {
            starter_strikeouts,
            starter_personal_line: 5,
            batter_strikeouts,
            batter_hits,
            plate_appearances: 4,
            cluster: 1,
            date: NaiveDate::from_ymd_opt(2025, 6, 1).unwrap(),
        }
    }

    #[test]
    fn starter_leg_boundaries_on_start_logs() {
        let leg = LegSpec::StarterK { over: 5 };
        assert_eq!(leg.eval_on_start(&start(4, 0, 0, 6.0)), Some(false));
        assert_eq!(leg.eval_on_start(&start(5, 0, 0, 6.0)), Some(true));
        let leg = LegSpec::StarterKUnder { under: 3 };
        assert_eq!(leg.eval_on_start(&start(3, 0, 0, 6.0)), Some(true));
        assert_eq!(leg.eval_on_start(&start(4, 0, 0, 6.0)), Some(false));
        let leg = LegSpec::StarterER { under: 1 };
        assert_eq!(leg.eval_on_start(&start(0, 1, 0, 6.0)), Some(true));
        assert_eq!(leg.eval_on_start(&start(0, 2, 0, 6.0)), Some(false));
        let leg = LegSpec::StarterHitsAllowed { under: 5 };
        assert_eq!(leg.eval_on_start(&start(0, 0, 5, 6.0)), Some(true));
        assert_eq!(leg.eval_on_start(&start(0, 0, 6, 6.0)), Some(false));
        // "Over 16.5 outs" → 17+: 5.1 IP is 16 and loses, 5.2 is 17 and wins.
        let leg = LegSpec::StarterOuts { over: 17 };
        assert_eq!(leg.eval_on_start(&start(0, 0, 0, 5.1)), Some(false));
        assert_eq!(leg.eval_on_start(&start(0, 0, 0, 5.2)), Some(true));
        // A view/batter leg is undefined on a start log — None, not false.
        assert_eq!(
            LegSpec::TeamK { over: 10 }.eval_on_start(&start(9, 0, 0, 6.0)),
            None
        );
        assert_eq!(
            LegSpec::BatterK { over: 1 }.eval_on_start(&start(9, 0, 0, 6.0)),
            None
        );
    }

    #[test]
    fn view_leg_boundaries_on_starter_views() {
        let covered = view(6, 2, 3, 5, Some(9), Some(8));
        assert_eq!(
            LegSpec::StarterK { over: 7 }.eval_on_view(&view(6, 0, 0, 0, None, None)),
            Some(false)
        );
        assert_eq!(
            LegSpec::StarterK { over: 7 }.eval_on_view(&view(7, 0, 0, 0, None, None)),
            Some(true)
        );
        assert_eq!(
            LegSpec::StarterKUnder { under: 3 }.eval_on_view(&view(3, 0, 0, 0, None, None)),
            Some(true)
        );
        assert_eq!(
            LegSpec::StarterER { under: 2 }.eval_on_view(&covered),
            Some(true)
        );
        assert_eq!(
            LegSpec::TeamK { over: 10 }.eval_on_view(&covered),
            Some(false)
        );
        assert_eq!(
            LegSpec::TeamK { over: 9 }.eval_on_view(&covered),
            Some(true)
        );
        assert_eq!(
            LegSpec::TeamHits { under: 8 }.eval_on_view(&covered),
            Some(true)
        );
        assert_eq!(
            LegSpec::TeamHits { under: 7 }.eval_on_view(&covered),
            Some(false)
        );
        assert_eq!(
            LegSpec::DuelK { over: 6 }.eval_on_view(&covered),
            Some(false)
        );
        assert_eq!(
            LegSpec::DuelK { over: 5 }.eval_on_view(&covered),
            Some(true)
        );
        assert_eq!(
            LegSpec::CombinedER { under: 5 }.eval_on_view(&covered),
            Some(true)
        );
        assert_eq!(
            LegSpec::CombinedER { under: 4 }.eval_on_view(&covered),
            Some(false)
        );
        // Outpitch: a tie is undefined (drops), not a loss.
        assert_eq!(
            LegSpec::Outpitch.eval_on_view(&view(0, 2, 3, 0, None, None)),
            Some(true)
        );
        assert_eq!(
            LegSpec::Outpitch.eval_on_view(&view(0, 3, 2, 0, None, None)),
            Some(false)
        );
        assert_eq!(
            LegSpec::Outpitch.eval_on_view(&view(0, 2, 2, 0, None, None)),
            None
        );
        // A game the batter file does not cover: unknown, not zero.
        let uncovered = view(6, 2, 3, 5, None, None);
        assert_eq!(LegSpec::TeamK { over: 9 }.eval_on_view(&uncovered), None);
        assert_eq!(
            LegSpec::TeamHits { under: 8 }.eval_on_view(&uncovered),
            None
        );
        // A batter leg is undefined on a view.
        assert_eq!(LegSpec::BatterK { over: 1 }.eval_on_view(&covered), None);
    }

    #[test]
    fn batter_leg_boundaries_on_batter_start_pairs() {
        assert_eq!(
            LegSpec::BatterK { over: 1 }.eval_on_batter(&batter(6, 0, 1)),
            Some(false)
        );
        assert_eq!(
            LegSpec::BatterK { over: 1 }.eval_on_batter(&batter(6, 1, 1)),
            Some(true)
        );
        assert_eq!(
            LegSpec::BatterHits { under: 1 }.eval_on_batter(&batter(6, 0, 1)),
            Some(true)
        );
        assert_eq!(
            LegSpec::BatterHits { under: 1 }.eval_on_batter(&batter(6, 0, 2)),
            Some(false)
        );
        assert_eq!(
            LegSpec::StarterK { over: 6 }.eval_on_batter(&batter(6, 0, 0)),
            Some(true)
        );
        assert_eq!(
            LegSpec::StarterK { over: 7 }.eval_on_batter(&batter(6, 0, 0)),
            Some(false)
        );
        // A view leg is undefined on a batter-start pair.
        assert_eq!(
            LegSpec::TeamK { over: 10 }.eval_on_batter(&batter(6, 0, 0)),
            None
        );
    }
}
