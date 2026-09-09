//! The measured-correlation half of the same-game-parlay thesis.
//!
//! [`crate::sgp`] recovers the correlation a book's price *allows for*; this
//! module measures the correlation that actually *exists* between two legs,
//! directly from paired outcomes. The gap between the two is the edge the
//! thesis hunts. Everything here is pure: paired Bernoulli observations in,
//! exact φ coefficients and cluster-bootstrap intervals out. Nothing here
//! knows about football; the dataset-aware layer (the NFL atlas, still to be
//! written) is the consumer. Carried over verbatim from the baseball study.

use chrono::NaiveDate;
use std::collections::BTreeMap;

/// Below this many distinct clusters a cluster bootstrap is not trustworthy —
/// the resample has too few moving parts to span the sampling distribution.
/// One floor for the whole crate, so intervals go quiet at the same point
/// rather than each study picking its own.
pub const MIN_CLUSTERS_FOR_INTERVAL: usize = 25;

/// The exact Bernoulli φ of paired 0/1 observations, with the 2×2 levels it
/// comes from: returns `(phi, p_a, p_b, joint)`.
///
/// For two Bernoulli legs the 2×2 table pins the whole joint distribution, so
/// the closed form *is* the correlation, not an estimate of one — the same
/// formula [`crate::sgp::phi_from_joint`] inverts from prices, which is what
/// makes book-side and outcome-side numbers comparable at all.
///
/// `None` on empty input or when either leg is degenerate (p = 0 or 1): a
/// constant outcome has no correlation with anything.
pub fn phi(paired: &[(bool, bool)]) -> Option<(f64, f64, f64, f64)> {
    if paired.is_empty() {
        return None;
    }
    let n = paired.len() as f64;
    let a_wins = paired.iter().filter(|&&(a, _)| a).count() as f64;
    let b_wins = paired.iter().filter(|&&(_, b)| b).count() as f64;
    let both = paired.iter().filter(|&&(a, b)| a && b).count() as f64;
    let (p_a, p_b, joint) = (a_wins / n, b_wins / n, both / n);
    let value = crate::sgp::phi_from_joint(p_a, p_b, joint)?;
    Some((value, p_a, p_b, joint))
}

/// Cluster-bootstrap 95% interval for φ: resamples whole clusters (player-
/// seasons or team-seasons, in the atlas) with replacement and recomputes φ
/// on each draw.
///
/// Games by one quarterback in one season are not independent — one passer's
/// October is one observation repeated, not several — so resampling rows
/// would understate the interval by exactly the within-cluster dependence the
/// measurement is trying to respect. Resampling clusters keeps that
/// dependence intact. **Cluster by the unit that owns the correlation**: the
/// baseball study found a pitcher-clustered interval 3.4x too narrow for a
/// quantity that was really an opponent property.
///
/// Plain percentile, not BCa: this backs a descriptive atlas, and at these
/// cluster counts the bias correction moves the endpoints by less than the
/// rounding they are printed with.
///
/// Deterministic: the seed selects every draw, so unchanged inputs reproduce
/// the interval bit-for-bit. `None` when φ itself is undefined, when there are
/// fewer than [`MIN_CLUSTERS_FOR_INTERVAL`] clusters, or when more than half
/// the draws land on degenerate resamples.
pub fn cluster_bootstrap_phi(
    paired: &[(bool, bool)],
    cluster_ids: &[u64],
    reps: usize,
    seed: u64,
) -> Option<(f64, f64)> {
    if paired.len() != cluster_ids.len() {
        return None;
    }
    phi(paired)?;
    let mut by_cluster: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    for (index, &cluster) in cluster_ids.iter().enumerate() {
        by_cluster.entry(cluster).or_default().push(index);
    }
    if by_cluster.len() < MIN_CLUSTERS_FOR_INTERVAL {
        return None;
    }
    let groups: Vec<&Vec<usize>> = by_cluster.values().collect();
    let mut rng = Rng::new(seed);
    let mut replicates = Vec::with_capacity(reps);
    let mut draw: Vec<(bool, bool)> = Vec::with_capacity(paired.len());
    for _ in 0..reps {
        draw.clear();
        for _ in 0..groups.len() {
            for &index in groups[rng.below(groups.len())] {
                draw.push(paired[index]);
            }
        }
        // A resample can come up degenerate (every drawn cluster has a leg
        // constant); skip it rather than record a divide-by-zero.
        if let Some((value, _, _, _)) = phi(&draw) {
            replicates.push(value);
        }
    }
    if replicates.len() < reps / 2 {
        return None;
    }
    replicates.sort_by(f64::total_cmp);
    Some((
        percentile(&replicates, 0.025),
        percentile(&replicates, 0.975),
    ))
}

/// Observation indices grouped by the calendar year of each date, ascending.
///
/// The atlas's regime check: a correlation that exists only in one season is a
/// story about that season, not about how legs co-move.
pub fn era_indices(dates: &[NaiveDate]) -> BTreeMap<i32, Vec<usize>> {
    use chrono::Datelike;
    let mut eras: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    for (index, date) in dates.iter().enumerate() {
        eras.entry(date.year()).or_default().push(index);
    }
    eras
}

/// Linear-interpolation percentile of a sorted slice.
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

/// xorshift64*, hand-rolled rather than imported — the project vendors its
/// randomness so a bootstrap interval can never change under a dependency
/// upgrade. No `rand` crate, on purpose.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        // Zero is a fixed point for xorshift and would emit only zeros.
        Self(if seed == 0 { 1 } else { seed })
    }

    pub(crate) fn next(&mut self) -> u64 {
        let mut state = self.0;
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        self.0 = state;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform index below `n`. The modulo bias is O(n / 2^64), unmeasurable
    /// at the cluster counts here.
    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// Uniform double in [0, 1), from the top 53 bits — the mantissa width, so
    /// every value the interval can represent at that spacing is reachable and
    /// the draw is exactly uniform on the grid rather than merely close to it.
    /// The jitter that breaks rank ties in [`crate::stats::uniform_scores`]
    /// comes from here, which keeps the whole atlas on one vendored stream: a
    /// set of ranks, like an interval, can never move under a dependency
    /// upgrade.
    pub(crate) fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pairs with a hand-computed φ: both×3, a_only×1, b_only×1, neither×3
    /// gives p_a = p_b = 0.5, joint = 0.375, so φ = (0.375 − 0.25)/0.25 = 0.5.
    fn known_pairs() -> Vec<(bool, bool)> {
        vec![
            (true, true),
            (true, true),
            (true, true),
            (true, false),
            (false, true),
            (false, false),
            (false, false),
            (false, false),
        ]
    }

    /// The atlas reports each family's positive quadrant only; this pins why
    /// that loses nothing. Flipping both legs leaves the 2×2 table's diagonal
    /// sums unchanged, so φ(¬A, ¬B) = φ(A, B) exactly; flipping one leg swaps
    /// a diagonal for an off-diagonal, negating it.
    #[test]
    fn phi_is_identical_on_the_complementary_quadrant() {
        let pairs = known_pairs();
        let (both, p_a, p_b, joint) = phi(&pairs).unwrap();
        assert!((both - 0.5).abs() < 1e-12, "got {both}");
        assert!((p_a - 0.5).abs() < 1e-12);
        assert!((p_b - 0.5).abs() < 1e-12);
        assert!((joint - 0.375).abs() < 1e-12);

        let negated: Vec<(bool, bool)> = pairs.iter().map(|&(a, b)| (!a, !b)).collect();
        let (neither, _, _, _) = phi(&negated).unwrap();
        assert!((both - neither).abs() < 1e-12, "{both} vs {neither}");

        let mixed: Vec<(bool, bool)> = pairs.iter().map(|&(a, b)| (a, !b)).collect();
        let (one_leg_flipped, _, _, _) = phi(&mixed).unwrap();
        assert!(
            (both + one_leg_flipped).abs() < 1e-12,
            "{both} vs {one_leg_flipped}"
        );
    }

    /// A constant leg has no correlation with anything, and empty input is
    /// not a measurement.
    #[test]
    fn degenerate_inputs_have_no_phi() {
        assert!(phi(&[]).is_none());
        assert!(phi(&[(true, true), (true, false)]).is_none());
        assert!(phi(&[(false, true), (false, false)]).is_none());
    }

    /// The seed selects every draw: identical inputs and seed must reproduce
    /// the interval exactly, and a different seed must move it.
    #[test]
    fn bootstrap_is_deterministic_for_a_fixed_seed() {
        // 30 clusters of 40 patterned pairs, correlated but not degenerate.
        let mut pairs = Vec::new();
        let mut clusters = Vec::new();
        for cluster in 0..30u64 {
            for i in 0..40 {
                let a = !(i + cluster as usize).is_multiple_of(3);
                let b = if a {
                    !(i + cluster as usize).is_multiple_of(5)
                } else {
                    i.is_multiple_of(7)
                };
                pairs.push((a, b));
                clusters.push(cluster);
            }
        }
        let first = cluster_bootstrap_phi(&pairs, &clusters, 300, 42);
        let second = cluster_bootstrap_phi(&pairs, &clusters, 300, 42);
        assert_eq!(first, second, "same seed must resample identically");
        let other = cluster_bootstrap_phi(&pairs, &clusters, 300, 43);
        assert_ne!(first, other, "a different seed must resample differently");
    }

    /// Correlated Bernoulli pairs from a known DGP must be recovered inside
    /// the interval. B = A with probability q = 0.7, else ¬A; with
    /// p_a = p_b = 0.5 that gives joint = 0.35 and φ = 2q − 1 = 0.4 exactly.
    #[test]
    fn bootstrap_recovers_a_known_phi() {
        let mut rng = Rng::new(0xDEAD_BEEF);
        let mut pairs = Vec::new();
        let mut clusters = Vec::new();
        for cluster in 0..40u64 {
            for _ in 0..50 {
                // n = 2000 across 40 clusters.
                let a = rng.below(2) == 0;
                let same = rng.below(10) < 7;
                pairs.push((a, if same { a } else { !a }));
                clusters.push(cluster);
            }
        }
        let (point, _, _, _) = phi(&pairs).unwrap();
        assert!((point - 0.4).abs() < 0.05, "point {point}");
        let (low, high) = cluster_bootstrap_phi(&pairs, &clusters, 300, 7).unwrap();
        assert!(low < 0.4 && 0.4 < high, "CI [{low}, {high}] missed 0.4");
        assert!(high - low < 0.2, "CI width {}", high - low);
    }

    /// Fewer clusters than the floor must go quiet, and a cluster vector that
    /// does not match the pairs is a caller bug, not a measurement.
    #[test]
    fn too_few_clusters_or_mismatched_input_refuses_an_interval() {
        let pairs = known_pairs();
        let few = vec![0u64, 0, 1, 1, 2, 2, 3, 3];
        assert_eq!(cluster_bootstrap_phi(&pairs, &few, 300, 1), None);
        let mismatched = vec![0u64; 4];
        assert_eq!(cluster_bootstrap_phi(&pairs, &mismatched, 300, 1), None);
    }

    #[test]
    fn era_indices_group_by_calendar_year_in_order() {
        let dates = [
            NaiveDate::from_ymd_opt(2023, 4, 1).unwrap(),
            NaiveDate::from_ymd_opt(2023, 9, 30).unwrap(),
            NaiveDate::from_ymd_opt(2024, 4, 1).unwrap(),
            NaiveDate::from_ymd_opt(2026, 7, 1).unwrap(),
        ];
        let eras = era_indices(&dates);
        assert_eq!(
            eras.keys().copied().collect::<Vec<_>>(),
            vec![2023, 2024, 2026]
        );
        assert_eq!(eras[&2023], vec![0, 1]);
        assert_eq!(eras[&2024], vec![2]);
        assert_eq!(eras[&2026], vec![3]);
    }
}
