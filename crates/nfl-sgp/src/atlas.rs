//! The measurement engine: families → dependence objects, strata, the
//! reporting grid, and the validation that gates scoring — `tasks/plan.md`
//! §1.6, §1.8, §1.9 and the atlas half of §6.
//!
//! [`crate::families`] hands over raw paired values; [`crate::stats`] knows how
//! to standardise, rank and read a joint. This module is what sits between
//! them: it decides *what pool a rank is taken within*, which is the decision
//! the whole design turns on.
//!
//! ## The pool is the stratum, and each leg is standardised against its own
//! subject
//!
//! Two rules, both easy to get subtly wrong and neither of which announces
//! itself when broken:
//!
//! 1. **Each leg is standardised within its own subject-season.** In
//!    `teamtotal_x_passtd`, leg A is a team's points and leg B is that team's
//!    quarterback's passing touchdowns; those are different subjects with
//!    different levels and different spreads. [`crate::families::Observation`]
//!    carries `subject_a` and `subject_b` separately for exactly this reason,
//!    and standardising both against one pooled series would leave the
//!    quarterback's level inside leg B — the level the market has already
//!    priced.
//! 2. **Ranks are taken within the stratum, not within the family.** A stratum
//!    is a claim about a *conditional* dependence: "among games where the
//!    receiver's target share was in the top third, how often did both legs
//!    land in their corners?" Ranking the whole family and then filtering would
//!    answer a different question — it would measure each stratum's position
//!    within the family's overall distribution, so a stratum whose games are
//!    systematically high-scoring would show a spurious joint in the upper
//!    corner. Ranking inside the stratum removes exactly that, and it is why
//!    [`Stratum`] stores its own `z_a` / `z_b` rather than a slice of a shared
//!    pool.
//!
//! ## Why the standardised values are stored, not just the ranks
//!
//! The cluster bootstrap of §1.7 re-ranks inside every replicate: a resample
//! that draws one quarterback-season three times must be ranked as the sample
//! it actually is. That needs the pre-rank values, so [`Stratum`] keeps both.
//!
//! ## The validation gates scoring
//!
//! §1.9 asks whether the dependence measured on history is the dependence
//! operating tonight. Fit on seasons ≤ [`TRANSPORT_SPLIT_SEASON`], evaluate on
//! the rest, and count how many of the 36 grid points have the two halves
//! disagreeing by more than sampling explains — the interval is on the
//! **difference**, resampling both halves, because the trained joint carries
//! sampling error of its own and comparing it against a test-only interval
//! rejects far above its nominal rate. Above
//! [`TRANSPORT_MISS_LIMIT`] the stratum is marked
//! [`Stratum::transport_ok`]` = false`, [`Atlas::failed_families`] names the
//! family, and the scorer refuses to price it. A family that cannot be shown to
//! transport is not a small problem to note in a footnote; it is the assumption
//! the estimate is made of.

use crate::families::{self, ClusterScheme, CovariateScheme, Family, Observation, spread_bucket};
use crate::games::{GameTable, SubjectKey};
use crate::stats::{
    Dependence, Direction, Marginal, N_MIN, cluster_bootstrap_joint,
    cluster_bootstrap_joint_difference, cluster_bootstrap_joints, gaussian_joint, kendall_tau,
    loo_standardize, phi_at,
};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

/// The one recorded jitter seed for the atlas's ranks (§1.3). Recorded because
/// the tie-breaking is part of the measurement: `--jitter-seed-check` re-runs
/// with another and reports the largest change.
pub const ATLAS_SEED: u64 = 0x4E_464C_5F41_544C; // "NFL_ATL"

/// Bootstrap replicates behind every published interval (§1.7).
pub const BOOTSTRAP_REPS: usize = 1000;

/// The marginals the atlas publishes joints at (§1.6). Tonight's marginals are
/// whatever the market says; this grid exists so families can be compared with
/// each other and with the baseball numbers.
pub const REPORTING_GRID: [f64; 3] = [0.3, 0.5, 0.7];

/// Seasons up to and including this one train the transport check; later ones
/// test it (§1.9).
pub const TRANSPORT_SPLIT_SEASON: u16 = 2017;

/// A stratum fails §1.9 when more than this share of the 36 grid points have
/// the trained prediction outside the test interval.
///
/// The nominal miss rate for a correct 95% interval is 5%. The threshold is set
/// at four times that rather than at the nominal rate because the 36 points are
/// heavily overlapping — they read the same sample at nine marginal pairs — so
/// their misses are strongly dependent and a strict 5% cut would fail
/// well-behaved families on one unlucky corner. Fixed here, before any family
/// was measured.
pub const TRANSPORT_MISS_LIMIT: f64 = 0.20;

/// A minimum sample below which a stratum is built but never trusted: its
/// interval would be a count, not a measurement.
pub const MIN_STRATUM_OBSERVATIONS: usize = 200;

/// The four direction combinations of a two-leg ticket.
pub const DIRECTIONS: [(Direction, Direction); 4] = [
    (Direction::Over, Direction::Over),
    (Direction::Over, Direction::Under),
    (Direction::Under, Direction::Over),
    (Direction::Under, Direction::Under),
];

/// One (family, stratum): the dependence, what it was built from, and whether
/// it may be scored.
#[derive(Debug, Clone, PartialEq)]
pub struct Stratum {
    pub family: Family,
    /// `all`, a tercile (`t1`/`t2`/`t3`) or a spread bucket.
    pub key: String,
    /// The covariate interval this stratum covers, `None` for `all`.
    pub range: Option<(f64, f64)>,
    pub n: usize,
    pub clusters: usize,
    pub seasons: (u16, u16),
    pub dependence: Dependence,
    pub cluster_ids: Vec<u64>,
    /// The second one-way cluster ids, for a cross-team family (§1.7).
    pub cluster_ids_secondary: Option<Vec<u64>>,
    /// Standardised values, kept because the bootstrap re-ranks per replicate.
    pub z_a: Vec<f64>,
    pub z_b: Vec<f64>,
    /// §1.9's verdict. `false` means the scorer must refuse this family.
    pub transport_ok: bool,
    /// How many of the 36 grid points missed, when the check could run.
    pub transport_misses: Option<usize>,
}

impl Stratum {
    /// The empirical survival copula at two market marginals — the estimator
    /// the whole study is built on.
    pub fn joint(&self, a: Marginal, b: Marginal) -> Option<f64> {
        self.dependence.joint(a, b)
    }

    /// The Gaussian comparator at the same marginals.
    pub fn gaussian_joint(&self, a: Marginal, b: Marginal) -> Option<f64> {
        Some(gaussian_joint(self.dependence.gaussian_rho()?, a, b))
    }

    /// Cluster-bootstrap interval for [`Stratum::joint`].
    ///
    /// A cross-team family carries two one-way clusterings and neither is the
    /// right one on its own — a game ties a home season to an away season, and
    /// grouping by either alone treats the other as independent. §1.7 takes the
    /// **wider** of the two, which is the conservative reading and the only one
    /// that cannot understate the uncertainty.
    pub fn joint_interval(
        &self,
        a: Marginal,
        b: Marginal,
        reps: usize,
        seed: u64,
    ) -> Option<(f64, f64)> {
        let primary =
            cluster_bootstrap_joint(&self.z_a, &self.z_b, &self.cluster_ids, a, b, reps, seed);
        let Some(secondary_ids) = self.cluster_ids_secondary.as_ref() else {
            return primary;
        };
        let secondary =
            cluster_bootstrap_joint(&self.z_a, &self.z_b, secondary_ids, a, b, reps, seed);
        match (primary, secondary) {
            (Some(first), Some(second)) => Some(if width(second) > width(first) {
                second
            } else {
                first
            }),
            (first, second) => first.or(second),
        }
    }

    /// Intervals for many marginal pairs from one set of resamples — the
    /// batched form of [`Stratum::joint_interval`], and the one the grid uses.
    /// Same two-one-way widening rule for a cross-team family.
    pub fn grid_intervals(
        &self,
        pairs: &[(Marginal, Marginal)],
        reps: usize,
        seed: u64,
    ) -> Vec<Option<(f64, f64)>> {
        if reps == 0 {
            return vec![None; pairs.len()];
        }
        let primary =
            cluster_bootstrap_joints(&self.z_a, &self.z_b, &self.cluster_ids, pairs, reps, seed);
        let secondary = self
            .cluster_ids_secondary
            .as_ref()
            .and_then(|ids| cluster_bootstrap_joints(&self.z_a, &self.z_b, ids, pairs, reps, seed));
        match (primary, secondary) {
            (Some(first), Some(second)) => first
                .into_iter()
                .zip(second)
                .map(|(left, right)| match (left, right) {
                    (Some(a), Some(b)) => Some(if width(b) > width(a) { b } else { a }),
                    (a, b) => a.or(b),
                })
                .collect(),
            (Some(only), None) | (None, Some(only)) => only,
            (None, None) => vec![None; pairs.len()],
        }
    }

    /// Whether this stratum is big enough to be worth reading.
    pub fn is_thin(&self) -> bool {
        self.n < MIN_STRATUM_OBSERVATIONS
    }
}

fn width((low, high): (f64, f64)) -> f64 {
    high - low
}

/// Every family's strata, plus the provenance needed to reproduce them.
#[derive(Debug, Clone, PartialEq)]
pub struct Atlas {
    pub seed: u64,
    pub seasons: (u16, u16),
    pub strata: Vec<Stratum>,
}

impl Atlas {
    /// Measures every family in [`crate::families::ALL`].
    pub fn build(table: &GameTable, seed: u64) -> Self {
        let mut strata = Vec::new();
        for family in families::ALL {
            let observations = families::observations(family, table);
            strata.extend(build_family(family, &observations, seed));
        }
        let seasons = strata
            .iter()
            .fold((u16::MAX, u16::MIN), |(low, high), stratum| {
                (low.min(stratum.seasons.0), high.max(stratum.seasons.1))
            });
        Self {
            seed,
            seasons,
            strata,
        }
    }

    /// Every stratum of one family, `all` first.
    pub fn strata(&self, family: Family) -> Vec<&Stratum> {
        self.strata
            .iter()
            .filter(|stratum| stratum.family == family)
            .collect()
    }

    /// The stratum a live covariate value falls in — the scorer's entry point.
    ///
    /// `None` covariate means the `all` stratum, which is also the fallback
    /// when the value falls outside every stored range (a total beyond
    /// anything in sixteen seasons). Falling back rather than refusing is
    /// deliberate: `all` is a real, if blunter, measurement of the same family,
    /// and the alternative is a row that cannot be scored because one covariate
    /// was unusual.
    pub fn stratum_for(&self, family: Family, covariate: Option<f64>) -> Option<&Stratum> {
        let all = self
            .strata
            .iter()
            .find(|stratum| stratum.family == family && stratum.key == ALL_KEY);
        let Some(value) = covariate else {
            return all;
        };
        self.strata
            .iter()
            .find(|stratum| {
                stratum.family == family
                    && stratum.key != ALL_KEY
                    && stratum
                        .range
                        .is_some_and(|(low, high)| value >= low && value <= high)
            })
            .or(all)
    }

    /// Families whose `all` stratum failed §1.9. The scorer must not price
    /// these.
    pub fn failed_families(&self) -> Vec<Family> {
        families::ALL
            .into_iter()
            .filter(|&family| {
                self.strata.iter().any(|stratum| {
                    stratum.family == family && stratum.key == ALL_KEY && !stratum.transport_ok
                })
            })
            .collect()
    }

    pub fn write_json(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let wire = WireAtlas::from(self);
        let text = serde_json::to_string_pretty(&wire)?;
        std::fs::write(path, text)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let wire: WireAtlas = serde_json::from_str(&text)?;
        wire.into_atlas()
    }
}

/// The `all` stratum's key, in one place so the scorer and the builder cannot
/// disagree about its spelling.
pub const ALL_KEY: &str = "all";

/// Builds one family's strata: `all`, then whatever the covariate scheme cuts.
/// **Standardise on the whole family, rank inside the stratum.** The two steps
/// answer different questions and belong at different scopes:
///
/// * §1.2's standardisation removes a *subject's* level and scale, which are
///   properties of that subject's season and are best estimated from all of its
///   games. Estimating them inside a stratum both shrinks the sample and
///   conditions the estimate on the covariate, which is not what a level
///   adjustment is for.
/// * §1.3's ranking defines the *conditional* distribution the stratum is a
///   claim about, so it must happen within the stratum.
///
/// Doing both inside the stratum — the first version of this — is quietly
/// catastrophic for a family with few rows per subject. `bothqb_passyds`
/// anchors on the home side, so a team-season contributes about eight rows;
/// split three ways that is under three per stratum, below [`N_MIN`], and every
/// one is dropped for want of a leave-one-out scale. Its three strata came out
/// holding 62, 3 and 121 of 2,250 observations. Nothing errored and nothing
/// warned — the strata were simply almost empty, and a joint read off three
/// observations would have been printed like any other.
fn build_family(family: Family, observations: &[Observation], seed: u64) -> Vec<Stratum> {
    let Some((z_a, z_b, kept)) = standardise(family, observations) else {
        return Vec::new();
    };
    let mut strata = Vec::new();
    if let Some(stratum) = assemble(family, ALL_KEY, None, &z_a, &z_b, &kept, seed) {
        strata.push(stratum);
    }
    for (key, range, indices) in partition_indices(family, &kept) {
        let subset_a: Vec<f64> = indices.iter().map(|&index| z_a[index]).collect();
        let subset_b: Vec<f64> = indices.iter().map(|&index| z_b[index]).collect();
        let subset: Vec<Observation> = indices.iter().map(|&index| kept[index].clone()).collect();
        if let Some(stratum) = assemble(
            family,
            &key,
            Some(range),
            &subset_a,
            &subset_b,
            &subset,
            seed,
        ) {
            strata.push(stratum);
        }
    }
    strata
}

/// Which already-standardised rows fall in each stratum (§1.8). Indices, so the
/// caller keeps the values standardised on the full family.
///
/// This is the primary implementation; [`partition`] is the same cut expressed
/// as owned observations, for callers that want the rows themselves.
fn partition_indices(
    family: Family,
    observations: &[Observation],
) -> Vec<(String, (f64, f64), Vec<usize>)> {
    let covariate_of = |index: usize| observations[index].covariate;
    match family.covariate().scheme {
        CovariateScheme::SpreadBuckets => {
            let mut buckets: BTreeMap<&'static str, ((f64, f64), Vec<usize>)> = BTreeMap::new();
            for index in 0..observations.len() {
                let Some(value) = covariate_of(index) else {
                    continue;
                };
                let bucket = spread_bucket(value);
                let range = match bucket {
                    crate::families::SpreadBucket::Underdog => (f64::NEG_INFINITY, -3.0),
                    crate::families::SpreadBucket::Pickem => (-3.0, 3.0),
                    crate::families::SpreadBucket::Favourite => (3.0, f64::INFINITY),
                };
                buckets
                    .entry(bucket.label())
                    .or_insert((range, Vec::new()))
                    .1
                    .push(index);
            }
            buckets
                .into_iter()
                .map(|(key, (range, indices))| (key.to_string(), range, indices))
                .collect()
        }
        CovariateScheme::Terciles => {
            let mut values: Vec<f64> = observations
                .iter()
                .filter_map(|observation| observation.covariate)
                .collect();
            if values.len() < 3 {
                return Vec::new();
            }
            values.sort_by(f64::total_cmp);
            let lower = quantile(&values, 1.0 / 3.0);
            let upper = quantile(&values, 2.0 / 3.0);
            [
                ("t1", (f64::NEG_INFINITY, lower)),
                ("t2", (lower, upper)),
                ("t3", (upper, f64::INFINITY)),
            ]
            .into_iter()
            .map(|(key, range)| {
                let indices: Vec<usize> = (0..observations.len())
                    .filter(|&index| {
                        covariate_of(index).is_some_and(|value| in_range(value, range, key))
                    })
                    .collect();
                (key.to_string(), range, indices)
            })
            .collect()
        }
    }
}

/// Ranks already-standardised values into a stratum. The pool is the stratum —
/// see [`build_family`] for why the standardisation is not.
fn assemble(
    family: Family,
    key: &str,
    range: Option<(f64, f64)>,
    z_a: &[f64],
    z_b: &[f64],
    kept: &[Observation],
    seed: u64,
) -> Option<Stratum> {
    let dependence = Dependence::from_standardised(z_a, z_b, seed)?;
    let cluster_ids: Vec<u64> = kept.iter().map(|row| row.cluster_primary).collect();
    let cluster_ids_secondary = match family.cluster_scheme() {
        ClusterScheme::TwoOneWay => Some(
            kept.iter()
                .map(|row| row.cluster_secondary.unwrap_or(row.cluster_primary))
                .collect(),
        ),
        ClusterScheme::SingleSubjectSeason => None,
    };
    let mut distinct: Vec<u64> = cluster_ids.clone();
    distinct.sort_unstable();
    distinct.dedup();
    let seasons = kept.iter().fold((u16::MAX, u16::MIN), |(low, high), row| {
        (low.min(row.season), high.max(row.season))
    });
    Some(Stratum {
        family,
        key: key.to_string(),
        range,
        n: kept.len(),
        clusters: distinct.len(),
        seasons,
        dependence,
        cluster_ids,
        cluster_ids_secondary,
        z_a: z_a.to_vec(),
        z_b: z_b.to_vec(),
        transport_ok: true,
        transport_misses: None,
    })
}

/// The same cut as [`partition_indices`], as owned rows. Test-only: the build
/// path takes indices so it can keep the family-wide standardised values.
#[cfg(test)]
/// Cuts a family's observations into strata by its covariate (§1.8).
///
/// Terciles are computed from the observations that *have* a covariate, and the
/// edges are returned so the scorer can map tonight's value onto the same
/// buckets. The ranges are closed at both ends and share their endpoints; the
/// lookup in [`Atlas::stratum_for`] takes the first match, so a value exactly
/// on an edge lands in the lower stratum. That convention is arbitrary but it
/// is fixed, stored and tested, which is what matters.
fn partition(
    family: Family,
    observations: &[Observation],
) -> Vec<(String, (f64, f64), Vec<Observation>)> {
    partition_indices(family, observations)
        .into_iter()
        .map(|(key, range, indices)| {
            let subset = indices
                .into_iter()
                .map(|index| observations[index].clone())
                .collect();
            (key, range, subset)
        })
        .collect()
}

/// Membership for a tercile. The edges are shared, so the lower stratum claims
/// a value sitting exactly on one — matching [`Atlas::stratum_for`]'s
/// first-match rule.
fn in_range(value: f64, (low, high): (f64, f64), key: &str) -> bool {
    match key {
        "t1" => value <= high,
        "t3" => value > low,
        _ => value > low && value <= high,
    }
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    let rank = (q * (sorted.len() - 1) as f64).clamp(0.0, (sorted.len() - 1) as f64);
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] + (rank - lower as f64) * (sorted[upper] - sorted[lower])
    }
}

/// Standardises, ranks and packages one stratum.
///
/// Returns `None` when nothing survives standardisation — a stratum whose
/// subjects all played too few games to have a leave-one-out scale is not an
/// empty measurement, it is no measurement.
fn build_stratum(
    family: Family,
    key: &str,
    range: Option<(f64, f64)>,
    observations: &[Observation],
    seed: u64,
) -> Option<Stratum> {
    let (z_a, z_b, kept) = standardise(family, observations)?;
    let dependence = Dependence::from_standardised(&z_a, &z_b, seed)?;
    let cluster_ids: Vec<u64> = kept.iter().map(|row| row.cluster_primary).collect();
    let cluster_ids_secondary = match family.cluster_scheme() {
        ClusterScheme::TwoOneWay => Some(
            kept.iter()
                .map(|row| row.cluster_secondary.unwrap_or(row.cluster_primary))
                .collect(),
        ),
        ClusterScheme::SingleSubjectSeason => None,
    };
    let mut distinct: Vec<u64> = cluster_ids.clone();
    distinct.sort_unstable();
    distinct.dedup();
    let seasons = kept.iter().fold((u16::MAX, u16::MIN), |(low, high), row| {
        (low.min(row.season), high.max(row.season))
    });
    Some(Stratum {
        family,
        key: key.to_string(),
        range,
        n: kept.len(),
        clusters: distinct.len(),
        seasons,
        dependence,
        cluster_ids,
        cluster_ids_secondary,
        z_a,
        z_b,
        transport_ok: true,
        transport_misses: None,
    })
}

/// §1.2 applied per leg, per subject-season — or skipped entirely for a
/// market-centred family.
///
/// Returns the two standardised series and the observations that survived, in
/// the same order. An observation is kept only when **both** legs standardised:
/// a pair with one half missing is not a paired observation.
fn standardise(
    family: Family,
    observations: &[Observation],
) -> Option<(Vec<f64>, Vec<f64>, Vec<Observation>)> {
    if observations.is_empty() {
        return None;
    }
    if family.is_market_centred() {
        // Already residuals against a closing line. Centring them again on a
        // season of leave-one-out means would replace the market's per-game
        // centring with a far worse one — and that centring is the entire
        // reason this family is the end-to-end check of §1.9.
        let z_a = observations.iter().map(|row| row.x_a).collect();
        let z_b = observations.iter().map(|row| row.x_b).collect();
        return Some((z_a, z_b, observations.to_vec()));
    }
    let standardised_a = standardise_leg(observations, |row| (&row.subject_a, row.x_a));
    let standardised_b = standardise_leg(observations, |row| (&row.subject_b, row.x_b));
    let mut z_a = Vec::new();
    let mut z_b = Vec::new();
    let mut kept = Vec::new();
    for (index, observation) in observations.iter().enumerate() {
        if let (Some(a), Some(b)) = (standardised_a[index], standardised_b[index]) {
            z_a.push(a);
            z_b.push(b);
            kept.push(observation.clone());
        }
    }
    if kept.is_empty() {
        None
    } else {
        Some((z_a, z_b, kept))
    }
}

/// One leg's leave-one-out z-scores, grouped by that leg's own subject-season.
///
/// The series a subject is standardised within is *this stratum's* observations
/// for that subject, in the order they appear — which is season/week order,
/// because [`crate::families::observations`] emits table order and the table is
/// sorted. Within a stratum a subject may appear a handful of times, so `N_MIN`
/// does real work here: it is why a tercile stratum can legitimately have far
/// fewer rows than a third of the family.
fn standardise_leg(
    observations: &[Observation],
    pick: impl Fn(&Observation) -> (&SubjectKey, f64),
) -> Vec<Option<f64>> {
    let mut by_subject: BTreeMap<SubjectKey, Vec<usize>> = BTreeMap::new();
    for (index, observation) in observations.iter().enumerate() {
        by_subject
            .entry(pick(observation).0.clone())
            .or_default()
            .push(index);
    }
    let mut out = vec![None; observations.len()];
    for indices in by_subject.values() {
        let values: Vec<f64> = indices
            .iter()
            .map(|&index| pick(&observations[index]).1)
            .collect();
        for (slot, z) in indices.iter().zip(loo_standardize(&values, N_MIN)) {
            out[*slot] = z;
        }
    }
    out
}

/// One row of the reporting grid.
#[derive(Debug, Clone)]
pub struct GridPoint {
    pub p_a: f64,
    pub p_b: f64,
    pub direction_a: Direction,
    pub direction_b: Direction,
    pub empirical: f64,
    pub interval: Option<(f64, f64)>,
    pub gaussian: Option<f64>,
    pub independent: f64,
    pub phi: Option<f64>,
}

impl GridPoint {
    pub fn marginals(&self) -> (Marginal, Marginal) {
        (
            Marginal {
                p: self.p_a,
                direction: self.direction_a,
            },
            Marginal {
                p: self.p_b,
                direction: self.direction_b,
            },
        )
    }

    pub fn label(&self) -> String {
        format!(
            "{:.1}{} x {:.1}{}",
            self.p_a,
            short(self.direction_a),
            self.p_b,
            short(self.direction_b)
        )
    }
}

fn short(direction: Direction) -> &'static str {
    match direction {
        Direction::Over => "O",
        Direction::Under => "U",
    }
}

/// The 36 marginal/direction combinations the atlas publishes, in a fixed
/// order.
pub fn grid_marginals() -> Vec<(Marginal, Marginal)> {
    let mut out = Vec::with_capacity(36);
    for &p_a in &REPORTING_GRID {
        for &p_b in &REPORTING_GRID {
            for &(direction_a, direction_b) in &DIRECTIONS {
                out.push((
                    Marginal {
                        p: p_a,
                        direction: direction_a,
                    },
                    Marginal {
                        p: p_b,
                        direction: direction_b,
                    },
                ));
            }
        }
    }
    out
}

/// Every (marginal, marginal, direction, direction) the atlas publishes, with
/// one shared bootstrap behind all 36 intervals.
///
/// The batching is not an optimisation detail, it is what makes the atlas
/// runnable at all. A bootstrap replicate resamples clusters and re-ranks the
/// drawn pool; **neither step depends on the grid point**, so evaluating the
/// grid inside the replicate loop rather than re-running the loop per point
/// does the same work 36 times fewer. Done the naive way this call was
/// re-ranking eight thousand observations 36,000 times per stratum and did not
/// finish in ten minutes.
///
/// A second consequence, and the reason this is the honest construction as well
/// as the fast one: all 36 intervals now come from the *same* resamples, so
/// they are mutually consistent — two grid points cannot disagree about which
/// draws the sampling distribution was built from.
pub fn grid(stratum: &Stratum, reps: usize, seed: u64) -> Vec<GridPoint> {
    let marginals = grid_marginals();
    let intervals = stratum.grid_intervals(&marginals, reps, seed);
    marginals
        .into_iter()
        .zip(intervals)
        .filter_map(|((a, b), interval)| {
            let empirical = stratum.joint(a, b)?;
            Some(GridPoint {
                p_a: a.p,
                p_b: b.p,
                direction_a: a.direction,
                direction_b: b.direction,
                empirical,
                interval,
                gaussian: stratum.gaussian_joint(a, b),
                independent: a.p * b.p,
                phi: phi_at(empirical, a, b),
            })
        })
        .collect()
}

/// Summary statistics for one stratum (§1.6).
#[derive(Debug, Clone)]
pub struct Summary {
    pub spearman: Option<f64>,
    pub kendall: Option<f64>,
    pub gaussian_rho: Option<f64>,
}

pub fn summarise(stratum: &Stratum) -> Summary {
    Summary {
        spearman: stratum.dependence.spearman(),
        kendall: kendall_tau(&stratum.dependence.u, &stratum.dependence.v),
        gaussian_rho: stratum.dependence.gaussian_rho(),
    }
}

/// One grid point's transport check (§1.9).
#[derive(Debug, Clone)]
pub struct TransportPoint {
    pub label: String,
    pub predicted: f64,
    pub realised: f64,
    pub interval: Option<(f64, f64)>,
    pub missed: bool,
}

/// The §1.9 result for one (family, stratum).
#[derive(Debug, Clone)]
pub struct Transport {
    pub family: Family,
    pub key: String,
    pub train_n: usize,
    pub test_n: usize,
    pub points: Vec<TransportPoint>,
    pub misses: usize,
    pub ok: bool,
}

/// Fits on the early seasons, tests on the late ones, and reports whether the
/// dependence transported.
///
/// Both halves are standardised and ranked **independently**, which is the
/// point: the test set must be treated exactly as a future slate would be, with
/// no knowledge of the training pool's ranks.
pub fn validate(
    family: Family,
    observations: &[Observation],
    key: &str,
    seed: u64,
    reps: usize,
) -> Option<Transport> {
    let (train, test): (Vec<Observation>, Vec<Observation>) = observations
        .iter()
        .cloned()
        .partition(|row| row.season <= TRANSPORT_SPLIT_SEASON);
    let trained = build_stratum(family, key, None, &train, seed)?;
    let tested = build_stratum(family, key, None, &test, seed)?;
    let marginals = grid_marginals();
    // The interval is on the **difference**, resampling both halves, not on the
    // test half alone. A trained joint carries its own sampling error, and with
    // the two halves a similar size ignoring it rejects far above the nominal
    // rate on hairline margins — see
    // [`crate::stats::cluster_bootstrap_joint_difference`].
    let intervals = cluster_bootstrap_joint_difference(
        &trained.z_a,
        &trained.z_b,
        &trained.cluster_ids,
        &tested.z_a,
        &tested.z_b,
        &tested.cluster_ids,
        &marginals,
        reps,
        seed,
    )
    .unwrap_or_else(|| vec![None; marginals.len()]);
    let mut points = Vec::new();
    let mut misses = 0usize;
    for ((a, b), interval) in marginals.into_iter().zip(intervals) {
        let (Some(predicted), Some(realised)) = (trained.joint(a, b), tested.joint(a, b)) else {
            continue;
        };
        // A miss is "zero is outside the interval for train − test": the two
        // halves disagree by more than sampling explains.
        let missed = interval.is_some_and(|(low, high)| low > 0.0 || high < 0.0);
        if missed {
            misses += 1;
        }
        points.push(TransportPoint {
            label: GridPoint {
                p_a: a.p,
                p_b: b.p,
                direction_a: a.direction,
                direction_b: b.direction,
                empirical: realised,
                interval,
                gaussian: None,
                independent: a.p * b.p,
                phi: None,
            }
            .label(),
            predicted,
            realised,
            interval,
            missed,
        });
    }
    if points.is_empty() {
        return None;
    }
    let ok = (misses as f64 / points.len() as f64) <= TRANSPORT_MISS_LIMIT;
    Some(Transport {
        family,
        key: key.to_string(),
        train_n: trained.n,
        test_n: tested.n,
        points,
        misses,
        ok,
    })
}

/// Runs §1.9 for every family's `all` stratum and stamps the verdict onto the
/// atlas.
///
/// The verdict is taken on `all` and applied to the family's every stratum: a
/// family whose overall dependence does not transport is not rescued by one
/// tercile that happens to, and letting a stratum-level pass override a
/// family-level failure would be the multiple-comparison trap the bar in
/// METHOD.md exists to avoid.
pub fn apply_validation(atlas: &mut Atlas, table: &GameTable, reps: usize) -> Vec<Transport> {
    let mut reports = Vec::new();
    for family in families::ALL {
        let observations = families::observations(family, table);
        let Some(report) = validate(family, &observations, ALL_KEY, atlas.seed, reps) else {
            continue;
        };
        for stratum in atlas
            .strata
            .iter_mut()
            .filter(|stratum| stratum.family == family)
        {
            stratum.transport_ok = report.ok;
            stratum.transport_misses = Some(report.misses);
        }
        reports.push(report);
    }
    reports
}

/// Per-season stability for one family (§1.9's era check).
#[derive(Debug, Clone)]
pub struct SeasonRow {
    pub family: Family,
    pub season: u16,
    pub n: usize,
    pub joint: Option<f64>,
    pub spearman: Option<f64>,
}

pub fn seasons(family: Family, observations: &[Observation], seed: u64) -> Vec<SeasonRow> {
    let mut by_season: BTreeMap<u16, Vec<Observation>> = BTreeMap::new();
    for observation in observations {
        by_season
            .entry(observation.season)
            .or_default()
            .push(observation.clone());
    }
    by_season
        .into_iter()
        .map(|(season, subset)| {
            let stratum = build_stratum(family, ALL_KEY, None, &subset, seed);
            let (a, b) = (Marginal::over(0.5), Marginal::over(0.5));
            SeasonRow {
                family,
                season,
                n: stratum.as_ref().map_or(0, |s| s.n),
                joint: stratum.as_ref().and_then(|s| s.joint(a, b)),
                spearman: stratum.as_ref().and_then(|s| s.dependence.spearman()),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// serialization
// ---------------------------------------------------------------------------

/// The stored shape. Primitive arrays only, and the family as its key string,
/// so the file stays readable and nothing depends on a derive in another
/// module.
#[derive(Serialize, Deserialize)]
struct WireAtlas {
    seed: u64,
    season_low: u16,
    season_high: u16,
    strata: Vec<WireStratum>,
}

/// Float arrays are stored as **strings**, and that is not fussiness.
///
/// `serde_json`'s number path does not round-trip an `f64` exactly: writing and
/// reading back the atlas moved a standardised value from
/// `0.9928029079530307` to `...308`, one unit in the last place. Nothing would
/// visibly break — but the ranks are computed from these values, the bootstrap
/// re-ranks from them on every replicate, and the atlas's whole claim to being
/// checkable later is that a rebuild reproduces it bit for bit. An artifact
/// that quietly differs from itself across a save is exactly the kind of thing
/// that makes a number impossible to audit two months on.
///
/// Rust's own `f64` formatting is shortest-round-trip and its parser is
/// correctly rounded, so a string survives the trip exactly while staying
/// readable — unlike a bit pattern, which would also be exact but would make
/// the file impossible to eyeball. The human-facing artifact is
/// `nfl_atlas.csv`; this one is the machine's, and it has to be exact first.
fn floats_to_strings(values: &[f64]) -> Vec<String> {
    values.iter().map(|value| value.to_string()).collect()
}

fn strings_to_floats(values: &[String], what: &str) -> Result<Vec<f64>> {
    values
        .iter()
        .map(|text| {
            text.parse::<f64>()
                .map_err(|_| Error::Data(format!("atlas {what} holds {text:?}, not a number")))
        })
        .collect()
}

/// A stratum's covariate range, in a shape JSON can actually hold.
///
/// The outer terciles are unbounded — `t1` runs from −∞ and `t3` to +∞ — and
/// **JSON has no infinity**: `serde_json` writes one as `null`. Storing the
/// edges as bare `Option<f64>` therefore round-tripped an unbounded edge into
/// "no range at all", which silently turned `t1` and `t3` into strata that
/// [`Atlas::stratum_for`] could never match, sending every extreme covariate to
/// `all` after a reload. Nothing would have failed; the strata would just have
/// quietly stopped being used. So an absent bound is written explicitly and
/// means infinity, and `range: None` keeps its own separate meaning of "this is
/// the `all` stratum". Pinned by the round-trip test.
#[derive(Serialize, Deserialize)]
struct WireRange {
    /// `None` is −∞.
    low: Option<f64>,
    /// `None` is +∞.
    high: Option<f64>,
}

#[derive(Serialize, Deserialize)]
struct WireStratum {
    family: String,
    key: String,
    range: Option<WireRange>,
    n: usize,
    clusters: usize,
    season_low: u16,
    season_high: u16,
    /// Exact decimal strings — see [`floats_to_strings`].
    u: Vec<String>,
    v: Vec<String>,
    z_a: Vec<String>,
    z_b: Vec<String>,
    cluster_ids: Vec<u64>,
    cluster_ids_secondary: Option<Vec<u64>>,
    transport_ok: bool,
    transport_misses: Option<usize>,
}

impl From<&Atlas> for WireAtlas {
    fn from(atlas: &Atlas) -> Self {
        Self {
            seed: atlas.seed,
            season_low: atlas.seasons.0,
            season_high: atlas.seasons.1,
            strata: atlas
                .strata
                .iter()
                .map(|stratum| WireStratum {
                    family: stratum.family.key().to_string(),
                    key: stratum.key.clone(),
                    range: stratum.range.map(|(low, high)| WireRange {
                        low: low.is_finite().then_some(low),
                        high: high.is_finite().then_some(high),
                    }),
                    n: stratum.n,
                    clusters: stratum.clusters,
                    season_low: stratum.seasons.0,
                    season_high: stratum.seasons.1,
                    u: floats_to_strings(&stratum.dependence.u),
                    v: floats_to_strings(&stratum.dependence.v),
                    z_a: floats_to_strings(&stratum.z_a),
                    z_b: floats_to_strings(&stratum.z_b),
                    cluster_ids: stratum.cluster_ids.clone(),
                    cluster_ids_secondary: stratum.cluster_ids_secondary.clone(),
                    transport_ok: stratum.transport_ok,
                    transport_misses: stratum.transport_misses,
                })
                .collect(),
        }
    }
}

impl WireAtlas {
    fn into_atlas(self) -> Result<Atlas> {
        let mut strata = Vec::with_capacity(self.strata.len());
        for wire in self.strata {
            let family = Family::from_key(&wire.family)
                .ok_or_else(|| Error::Data(format!("unknown family {:?} in atlas", wire.family)))?;
            let range = wire.range.map(|range| {
                (
                    range.low.unwrap_or(f64::NEG_INFINITY),
                    range.high.unwrap_or(f64::INFINITY),
                )
            });
            strata.push(Stratum {
                family,
                key: wire.key,
                range,
                n: wire.n,
                clusters: wire.clusters,
                seasons: (wire.season_low, wire.season_high),
                dependence: Dependence {
                    u: strings_to_floats(&wire.u, "u")?,
                    v: strings_to_floats(&wire.v, "v")?,
                },
                cluster_ids: wire.cluster_ids,
                cluster_ids_secondary: wire.cluster_ids_secondary,
                z_a: strings_to_floats(&wire.z_a, "z_a")?,
                z_b: strings_to_floats(&wire.z_b, "z_b")?,
                transport_ok: wire.transport_ok,
                transport_misses: wire.transport_misses,
            });
        }
        Ok(Atlas {
            seed: self.seed,
            seasons: (self.season_low, self.season_high),
            strata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::correlation::Rng;
    use crate::games::SubjectKey;

    /// A synthetic family: `subjects` subject-seasons of `per` games each,
    /// with a Gaussian copula of correlation `rho` inside every subject and a
    /// per-subject level shift that standardisation must remove.
    fn synthetic(subjects: usize, per: usize, rho: f64, seed: u64) -> Vec<Observation> {
        let mut rng = Rng::new(seed);
        let mut out = Vec::new();
        for subject in 0..subjects {
            // A level and scale the market would already have priced.
            let level_a = 200.0 + subject as f64 * 30.0;
            let level_b = 1.0 + subject as f64 * 0.2;
            for game in 0..per {
                let (n1, n2) = box_muller(&mut rng);
                let x = n1;
                let y = rho * n1 + (1.0 - rho * rho).sqrt() * n2;
                out.push(Observation {
                    game_id: format!("s{subject:03}g{game:02}"),
                    season: 2010 + (subject % 16) as u16,
                    week: game as u8 + 1,
                    anchor_team: format!("T{subject:02}"),
                    x_a: level_a + 40.0 * x,
                    x_b: level_b + 0.8 * y,
                    subject_a: SubjectKey::Team(format!("T{subject:02}"), 2010),
                    subject_b: SubjectKey::Player(format!("p{subject:03}"), 2010),
                    covariate: Some(game as f64),
                    cluster_primary: subject as u64,
                    cluster_secondary: None,
                });
            }
        }
        out
    }

    fn box_muller(rng: &mut Rng) -> (f64, f64) {
        let u1 = (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
        let u2 = (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
        let u1 = u1.max(1e-12);
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        (radius * theta.cos(), radius * theta.sin())
    }

    /// The headline property: a known copula is recovered at the grid, and the
    /// per-subject level shifts — which are large — do not leak into it.
    #[test]
    fn a_known_copula_is_recovered_through_the_standardisation() {
        let observations = synthetic(60, 40, 0.5, 99);
        let stratum =
            build_stratum(Family::TeamtotalXPasstd, ALL_KEY, None, &observations, 7).unwrap();
        assert_eq!(stratum.n, 2400);
        assert_eq!(stratum.clusters, 60);
        let rho = stratum.dependence.gaussian_rho().unwrap();
        assert!((rho - 0.5).abs() < 0.05, "rho {rho}");
        let (a, b) = (Marginal::over(0.5), Marginal::over(0.5));
        let joint = stratum.joint(a, b).unwrap();
        let gaussian = stratum.gaussian_joint(a, b).unwrap();
        assert!((joint - gaussian).abs() < 0.02, "{joint} vs {gaussian}");
        // Independence would be 0.25; a ρ = 0.5 copula puts ~1/3 there.
        assert!(joint > 0.30, "joint {joint} should exceed independence");
    }

    /// If the two legs were standardised against one pooled series, the
    /// per-subject level shifts would dominate and the measured dependence
    /// would be far too high. This pins that they are separated.
    #[test]
    fn each_leg_is_standardised_within_its_own_subject() {
        // Zero true within-subject dependence, but leg A's level rises with
        // the subject index and so does leg B's — pooling would read a strong
        // positive correlation that is entirely between-subject.
        let observations = synthetic(60, 40, 0.0, 5);
        let stratum =
            build_stratum(Family::TeamtotalXPasstd, ALL_KEY, None, &observations, 7).unwrap();
        let rho = stratum.dependence.gaussian_rho().unwrap();
        assert!(
            rho.abs() < 0.05,
            "level shifts leaked into the dependence: rho {rho}"
        );
    }

    /// A market-centred family keeps its residuals untouched.
    #[test]
    fn a_market_centred_family_skips_standardisation() {
        let mut observations = synthetic(40, 30, 0.3, 11);
        for observation in &mut observations {
            observation.x_a -= 200.0;
            observation.x_b -= 1.0;
        }
        let (z_a, z_b, kept) = standardise(Family::SpreadXTotal, &observations).unwrap();
        assert_eq!(kept.len(), observations.len());
        for (index, observation) in observations.iter().enumerate() {
            assert_eq!(z_a[index], observation.x_a);
            assert_eq!(z_b[index], observation.x_b);
        }
    }

    /// Strata partition the covariate: every observation with one lands in
    /// exactly one tercile, and `all` holds them all.
    #[test]
    fn terciles_partition_the_observations() {
        let observations = synthetic(40, 30, 0.4, 3);
        let parts = partition(Family::TeamtotalXPasstd, &observations);
        assert_eq!(parts.len(), 3);
        let total: usize = parts.iter().map(|(_, _, subset)| subset.len()).sum();
        assert_eq!(total, observations.len());
        // And no observation is in two strata.
        let mut seen: Vec<&str> = Vec::new();
        for (key, _, subset) in &parts {
            assert!(!subset.is_empty(), "{key} is empty");
            seen.push(key);
        }
        seen.sort_unstable();
        assert_eq!(seen, vec!["t1", "t2", "t3"]);
    }

    /// **The strata must actually keep their rows.** Standardising inside a
    /// stratum instead of across the family silently emptied them: a subject
    /// with few rows per stratum falls under `N_MIN` and every one of its
    /// observations is dropped for want of a leave-one-out scale. On the real
    /// `bothqb_passyds` that left strata of 62, 3 and 121 out of 2,250 — no
    /// error, no warning, just a joint computed from three games.
    ///
    /// Eight rows per subject, split three ways, is exactly that case.
    #[test]
    fn strata_keep_their_rows_when_subjects_are_thin() {
        // 60 subjects x 8 rows, covariate spread across the terciles.
        let mut observations = synthetic(60, 8, 0.4, 31);
        for (index, observation) in observations.iter_mut().enumerate() {
            observation.covariate = Some((index % 9) as f64);
        }
        let strata = build_family(Family::TeamtotalXPasstd, &observations, 7);
        let all = strata.iter().find(|s| s.key == ALL_KEY).unwrap();
        assert_eq!(
            all.n, 480,
            "every row survives standardisation on the family"
        );
        let cut: usize = strata
            .iter()
            .filter(|s| s.key != ALL_KEY)
            .map(|s| s.n)
            .sum();
        assert_eq!(
            cut, 480,
            "the terciles must partition the family, not lose it to N_MIN"
        );
        for stratum in strata.iter().filter(|s| s.key != ALL_KEY) {
            assert!(
                stratum.n > 100,
                "stratum {} holds only {} rows",
                stratum.key,
                stratum.n
            );
        }
    }

    /// Tonight's covariate must land in the stratum it was built from,
    /// including exactly on an edge.
    #[test]
    fn stratum_for_maps_a_value_back_to_its_own_stratum() {
        let observations = synthetic(40, 30, 0.4, 3);
        let atlas = Atlas {
            seed: 1,
            seasons: (2010, 2025),
            strata: build_family(Family::TeamtotalXPasstd, &observations, 1),
        };
        let terciles: Vec<&Stratum> = atlas
            .strata(Family::TeamtotalXPasstd)
            .into_iter()
            .filter(|stratum| stratum.key != ALL_KEY)
            .collect();
        assert_eq!(terciles.len(), 3);
        for stratum in &terciles {
            let (low, high) = stratum.range.unwrap();
            let probe = if low.is_finite() && high.is_finite() {
                (low + high) / 2.0
            } else if low.is_finite() {
                low + 1.0
            } else {
                high - 1.0
            };
            let found = atlas
                .stratum_for(Family::TeamtotalXPasstd, Some(probe))
                .unwrap();
            assert_eq!(
                found.key, stratum.key,
                "probe {probe} in {:?}",
                stratum.range
            );
        }
        // The lower stratum claims a value exactly on the shared edge.
        let edge = terciles[0].range.unwrap().1;
        assert_eq!(
            atlas
                .stratum_for(Family::TeamtotalXPasstd, Some(edge))
                .unwrap()
                .key,
            "t1"
        );
        // No covariate, and a value beyond every range, both fall back to all.
        assert_eq!(
            atlas
                .stratum_for(Family::TeamtotalXPasstd, None)
                .unwrap()
                .key,
            ALL_KEY
        );
    }

    /// The bootstrap interval must widen for a cross-team family, because the
    /// wider of the two one-way clusterings is the reported one.
    #[test]
    fn a_cross_team_family_takes_the_wider_of_two_clusterings() {
        let mut observations = synthetic(60, 40, 0.4, 21);
        // Give every row a second cluster that is far coarser: 3 groups.
        for (index, observation) in observations.iter_mut().enumerate() {
            observation.cluster_secondary = Some((index % 3) as u64);
        }
        let stratum =
            build_stratum(Family::BothqbPassyds, ALL_KEY, None, &observations, 7).unwrap();
        assert!(stratum.cluster_ids_secondary.is_some());
        let (a, b) = (Marginal::over(0.5), Marginal::over(0.5));
        // The secondary clustering has only 3 groups, under the floor, so it
        // returns None and the primary stands — the fallback path.
        assert!(stratum.joint_interval(a, b, 100, 5).is_some());
    }

    #[test]
    fn the_atlas_round_trips_through_json() {
        let observations = synthetic(40, 30, 0.4, 3);
        let atlas = Atlas {
            seed: ATLAS_SEED,
            seasons: (2010, 2025),
            strata: build_family(Family::TeamtotalXPasstd, &observations, ATLAS_SEED),
        };
        let dir = std::env::temp_dir().join(format!("nfl-atlas-{}", std::process::id()));
        let path = dir.join("atlas.json");
        atlas.write_json(&path).unwrap();
        let loaded = Atlas::load(&path).unwrap();

        // Field by field before the whole-struct compare: a mismatch buried in
        // a 1,200-element vector is unreadable as one `assert_eq!` on the
        // whole atlas, and the first version of this test printed both copies
        // in full.
        assert_eq!(atlas.seed, loaded.seed, "seed");
        assert_eq!(atlas.seasons, loaded.seasons, "seasons");
        assert_eq!(atlas.strata.len(), loaded.strata.len(), "stratum count");
        for (before, after) in atlas.strata.iter().zip(loaded.strata.iter()) {
            let where_ = format!("{}/{}", before.family.key(), before.key);
            assert_eq!(before.family, after.family, "family at {where_}");
            assert_eq!(before.key, after.key, "key at {where_}");
            assert_eq!(before.range, after.range, "range at {where_}");
            assert_eq!(before.n, after.n, "n at {where_}");
            assert_eq!(before.clusters, after.clusters, "clusters at {where_}");
            assert_eq!(before.seasons, after.seasons, "seasons at {where_}");
            assert_eq!(
                before.cluster_ids, after.cluster_ids,
                "clusters at {where_}"
            );
            assert_eq!(
                before.cluster_ids_secondary, after.cluster_ids_secondary,
                "secondary clusters at {where_}"
            );
            assert_eq!(
                before.transport_ok, after.transport_ok,
                "transport_ok at {where_}"
            );
            assert_eq!(
                before.transport_misses, after.transport_misses,
                "transport_misses at {where_}"
            );
            for (index, (x, y)) in before.z_a.iter().zip(after.z_a.iter()).enumerate() {
                assert_eq!(x, y, "z_a[{index}] at {where_}");
            }
            for (index, (x, y)) in before.z_b.iter().zip(after.z_b.iter()).enumerate() {
                assert_eq!(x, y, "z_b[{index}] at {where_}");
            }
            for (index, (x, y)) in before
                .dependence
                .u
                .iter()
                .zip(after.dependence.u.iter())
                .enumerate()
            {
                assert_eq!(x, y, "u[{index}] at {where_}");
            }
            for (index, (x, y)) in before
                .dependence
                .v
                .iter()
                .zip(after.dependence.v.iter())
                .enumerate()
            {
                assert_eq!(x, y, "v[{index}] at {where_}");
            }
        }
        assert_eq!(atlas, loaded);

        // Specifically: the unbounded outer terciles must survive as
        // unbounded. JSON cannot hold an infinity, and when this was written
        // as a bare number it came back `null` — which read as "no range",
        // leaving t1 and t3 unmatchable and silently unused.
        let t1 = loaded
            .strata(Family::TeamtotalXPasstd)
            .into_iter()
            .find(|stratum| stratum.key == "t1")
            .unwrap();
        assert_eq!(t1.range.unwrap().0, f64::NEG_INFINITY);
        let t3 = loaded
            .strata(Family::TeamtotalXPasstd)
            .into_iter()
            .find(|stratum| stratum.key == "t3")
            .unwrap();
        assert_eq!(t3.range.unwrap().1, f64::INFINITY);
        // And the reloaded atlas still routes a covariate the same way.
        for probe in [-1e6, 0.0, 5.0, 12.0, 1e6] {
            assert_eq!(
                atlas
                    .stratum_for(Family::TeamtotalXPasstd, Some(probe))
                    .unwrap()
                    .key,
                loaded
                    .stratum_for(Family::TeamtotalXPasstd, Some(probe))
                    .unwrap()
                    .key,
                "probe {probe} routed differently after a reload"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Two builds with one seed are identical; the atlas has to be
    /// reproducible or none of its numbers can be checked later.
    #[test]
    fn building_twice_with_one_seed_is_identical() {
        let observations = synthetic(30, 20, 0.3, 77);
        let first = build_family(Family::TeamtotalXPasstd, &observations, ATLAS_SEED);
        let second = build_family(Family::TeamtotalXPasstd, &observations, ATLAS_SEED);
        assert_eq!(first, second);
    }

    /// **The validation must actually fail something.** Train seasons with no
    /// dependence, test seasons with a lot: the trained prediction should sit
    /// outside the test interval at most of the grid.
    #[test]
    fn validation_catches_a_family_that_does_not_transport() {
        let mut early = synthetic(40, 30, 0.0, 1);
        for observation in &mut early {
            observation.season = 2012;
        }
        let mut late = synthetic(40, 30, 0.8, 2);
        for observation in &mut late {
            observation.season = 2022;
            observation.cluster_primary += 1000;
        }
        early.extend(late);
        let report = validate(Family::TeamtotalXPasstd, &early, ALL_KEY, 7, 200).unwrap();
        assert!(
            !report.ok,
            "a regime change must fail: {} of {} missed",
            report.misses,
            report.points.len()
        );
    }

    /// And it must pass a family that does transport: one DGP throughout.
    #[test]
    fn validation_passes_a_stable_family() {
        let mut observations = synthetic(80, 30, 0.45, 4);
        for (index, observation) in observations.iter_mut().enumerate() {
            observation.season = if index % 2 == 0 { 2012 } else { 2022 };
        }
        let report = validate(Family::TeamtotalXPasstd, &observations, ALL_KEY, 7, 200).unwrap();
        assert!(
            report.ok,
            "a stable DGP must transport: {} of {} missed",
            report.misses,
            report.points.len()
        );
    }

    /// The grid is 36 points and its Fréchet arithmetic holds.
    #[test]
    fn the_grid_is_thirty_six_points_with_sane_values() {
        let observations = synthetic(40, 30, 0.4, 8);
        let stratum =
            build_stratum(Family::TeamtotalXPasstd, ALL_KEY, None, &observations, 7).unwrap();
        let points = grid(&stratum, 100, 3);
        assert_eq!(points.len(), 36);
        for point in &points {
            assert!((0.0..=1.0).contains(&point.empirical), "{}", point.label());
            let (a, b) = point.marginals();
            let upper = a.p.min(b.p);
            let lower = (a.p + b.p - 1.0).max(0.0);
            assert!(
                point.empirical <= upper + 1e-9 && point.empirical >= lower - 1e-9,
                "{} outside Fréchet [{lower}, {upper}]: {}",
                point.label(),
                point.empirical
            );
        }
    }
}
