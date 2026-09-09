//! What correlation is a sportsbook's same-game parlay price *allowing for*?
//!
//! A two-leg SGP pays only when both legs win. If the legs were independent,
//! the fair joint probability would be the product of the two leg
//! probabilities — but legs inside one game are usually not independent (a
//! quarterback's passing-yards over and his receiver's yards over move
//! together), and the
//! book shades the parlay price accordingly. That shade is a *correlation
//! allowance*, and it is recoverable from the prices alone. The thesis this
//! module supports: where the book's allowance (φ_book) sits below the
//! correlation measured directly from outcomes (φ_true), the parlay is priced
//! as if the legs co-occur less often than they actually do, and the gap is
//! the edge. Everything here is the inversion half of that comparison — pure
//! functions from prices to implied quantities. Measuring φ_true lives
//! elsewhere.
//!
//! Two confounds are priced in from the start, and the API is shaped around
//! them:
//!
//! 1. **Margin and correlation are inseparable in a single SGP price.** The
//!    offered joint can exceed the independence product because the book
//!    allows for correlation, because the parlay carries extra margin, or
//!    both. [`implied_phi_book`] therefore takes the margin (`hold`) as an
//!    argument and callers report an interval over hold in [0.10, 0.20]; the
//!    `indep` family in the collection sheet exists to measure the hold
//!    itself and collapse that interval.
//! 2. **The leg implieds carry their own vig.** [`mechanical_joint`] takes
//!    whatever probabilities the caller hands it. Feeding it raw leg prices
//!    is deliberate: [`price_ratio`] then captures correlation allowance plus
//!    margin difference, which is the model-free headline — separating the
//!    two is exactly what the hold interval is for.

/// American odds to implied probability, before any vig removal.
///
/// Carried over verbatim from the baseball study (`reference/baseball/`),
/// where the same four lines were pinned by the sportsbook probe's tests.
pub fn implied_probability(american: f64) -> Option<f64> {
    if american >= 100.0 {
        Some(100.0 / (american + 100.0))
    } else if american <= -100.0 {
        Some(-american / (-american + 100.0))
    } else {
        // Between -100 and +100 is not a valid American price.
        None
    }
}

/// Removes the book's margin proportionally across a two-sided market.
///
/// Raw implieds sum to more than one, and that surplus is the book's take,
/// so comparing raw numbers across markets manufactures an edge on every
/// leg, always in the same direction. The baseball study learned this the
/// hard way: on the raw basis its control family read a fabricated +0.19
/// correlation allowance that vanished once devigged. Returns the devigged
/// over and the overround itself — the
/// overround is a per-market margin observation, which is what the hold
/// interval on [`implied_phi_book`] leans on until the `indep` family pins it.
pub fn devig(over_american: f64, under_american: f64) -> Option<(f64, f64)> {
    let over = implied_probability(over_american)?;
    let under = implied_probability(under_american)?;
    let overround = over + under;
    if overround <= 0.0 {
        return None;
    }
    Some((over / overround, overround))
}

/// The devigged probability of the FIRST side, when both sides of a leg are
/// recorded. This is the marginal the joint math wants: a raw implied
/// probability carries the book's vig, and anchoring the covariance estimate
/// to it imports that vig into the joint. `None` on any invalid price — a
/// one-sided quote cannot be devigged, and guessing a side's fair value from
/// itself alone manufactures a margin-free number that does not exist.
pub fn devigged_probability(price: f64, other_price: f64) -> Option<f64> {
    devig(price, other_price).map(|(devigged, _)| devigged)
}

/// American odds to decimal odds: total return per unit staked, stake
/// included. The SGP's expected value is a probability times this number.
pub fn decimal_from_american(american: f64) -> Option<f64> {
    if american >= 100.0 {
        Some(1.0 + american / 100.0)
    } else if american <= -100.0 {
        Some(1.0 + 100.0 / -american)
    } else {
        None
    }
}

/// The joint probability if the legs were independent: `p1 * p2`.
///
/// "Mechanical" because it is what a parlay calculator does — multiply the
/// implieds — and it is the benchmark the offered price is judged against.
/// Not `Option`: with raw leg implieds this is always positive, and keeping
/// the benchmark total makes the gap it leaves explicit in [`price_ratio`].
pub fn mechanical_joint(p1: f64, p2: f64) -> f64 {
    p1 * p2
}

/// The joint probability the SGP price itself implies: `1 / decimal`.
///
/// This is the number that has the margin and the correlation allowance baked
/// in; everything downstream is about separating the two.
pub fn offered_joint(sgp_american: f64) -> Option<f64> {
    Some(1.0 / decimal_from_american(sgp_american)?)
}

/// `offered / mechanical` — the model-free headline statistic.
///
/// R = 1 is independence pricing with no margin difference between legs and
/// parlay. R > 1 means the book prices the joint *higher* than the leg
/// product: a positive correlation allowance, extra parlay margin, or both —
/// one price cannot separate them, which is what the hold interval on
/// [`implied_phi_book`] is for. R < 1 means the parlay pays *better* than
/// independence, which happens when the two legs carry more combined vig than
/// the SGP or the book allowances negative correlation.
pub fn price_ratio(p1: f64, p2: f64, sgp_american: f64) -> Option<f64> {
    let mechanical = mechanical_joint(p1, p2);
    if mechanical <= 0.0 {
        // A zero-probability leg makes the ratio a divide-by-zero, not a
        // measurement.
        return None;
    }
    Some(offered_joint(sgp_american)? / mechanical)
}

/// The φ (Pearson) correlation between two Bernoulli legs, given their joint.
///
/// For two 0/1 outcomes the joint pins the whole 2×2 table, so
/// `(J − p1·p2) / √(p1(1−p1)p2(1−p2))` *is* the correlation, not an estimate
/// of one — and the complementary quadrant (neither leg wins, joint
/// `1 − p1 − p2 + J`) yields the identical value, which the tests pin down.
/// `None` when either leg is degenerate (p = 0 or 1, or outside [0, 1]
/// entirely): a constant outcome has no correlation with anything.
///
/// Nothing clamps the result to the Fréchet bounds. An impossible φ is not a
/// bug in the arithmetic — it is the inputs disagreeing with each other, and
/// the caller should see it.
pub fn phi_from_joint(p1: f64, p2: f64, joint: f64) -> Option<f64> {
    let variance_product = p1 * (1.0 - p1) * p2 * (1.0 - p2);
    // Zero when either leg is a constant outcome; negative when a
    // "probability" outside [0, 1] would make the variance imaginary. Both
    // are None — checked before the square root, so no NaN ever arises.
    if variance_product <= 0.0 {
        return None;
    }
    Some((joint - mechanical_joint(p1, p2)) / variance_product.sqrt())
}

/// The correlation allowance embedded in an SGP price under an assumed hold.
///
/// The offered joint carries the book's margin, so `offered / (1 + hold)` is
/// the joint the book *acts as if* it believes, and the φ of that joint is
/// φ_book. A larger assumed hold strips more margin and therefore yields a
/// smaller φ_book — the interval over hold in [0.10, 0.20] is the honest way
/// to report this until the `indep` family measures the hold directly.
pub fn implied_phi_book(p1: f64, p2: f64, sgp_american: f64, hold: f64) -> Option<f64> {
    if hold <= -1.0 {
        // A hold of -100% or worse makes the margin removal a divide-by-zero
        // (or a sign flip), not a measurement.
        return None;
    }
    phi_from_joint(p1, p2, offered_joint(sgp_american)? / (1.0 + hold))
}

/// Expected profit per unit staked if the true joint probability is
/// `p_true_joint`: `p_true × decimal − 1`. The stake comes back with the
/// winnings when the parlay hits and is lost otherwise.
///
/// This is the number the thesis ultimately cares about — φ_true above
/// φ_book shows up here as positive EV — and it is exactly as honest as the
/// φ_true estimate it is fed. The function measures; it does not vouch.
pub fn ev_per_unit(p_true_joint: f64, sgp_american: f64) -> Option<f64> {
    Some(p_true_joint * decimal_from_american(sgp_american)? - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Worked example, favoured legs: -130 / -119 with the SGP at +244.
    /// Hand-checked: implieds 0.5652 and 0.5434, mechanical 0.3071, offered
    /// 1/3.44 = 0.2907, ratio 0.9465. R < 1 — the parlay pays *better* than
    /// the leg product, because the two favourite legs carry more combined
    /// vig than the SGP does.
    ///
    /// Note: headline numbers of 0.2311 / 1.258 circulate attached to this
    /// SGP price, but those belong to coin-flip legs (~+108/+108), not to
    /// -130/-119. Both versions are pinned here, with the correct leg prices
    /// on each, so the two never get conflated again.
    #[test]
    fn worked_example_favoured_legs_price_below_independence() {
        let p1 = implied_probability(-130.0).unwrap();
        let p2 = implied_probability(-119.0).unwrap();
        let mechanical = mechanical_joint(p1, p2);
        let offered = offered_joint(244.0).unwrap();
        let ratio = price_ratio(p1, p2, 244.0).unwrap();
        assert!((mechanical - 0.3071).abs() < 1e-3, "got {mechanical}");
        assert!((offered - 0.2907).abs() < 1e-3, "got {offered}");
        assert!((ratio - 0.9465).abs() < 1e-3, "got {ratio}");
    }

    /// The same SGP price with balanced legs: +108 / +108, SGP +244. Here the
    /// book prices the joint well above the leg product — R ≈ 1.258, the
    /// signature of a correlation allowance and/or extra parlay margin.
    #[test]
    fn worked_example_balanced_legs_price_above_independence() {
        let p1 = implied_probability(108.0).unwrap();
        let p2 = implied_probability(108.0).unwrap();
        let mechanical = mechanical_joint(p1, p2);
        let offered = offered_joint(244.0).unwrap();
        let ratio = price_ratio(p1, p2, 244.0).unwrap();
        assert!((mechanical - 0.2311).abs() < 1e-3, "got {mechanical}");
        assert!((offered - 0.2907).abs() < 1e-3, "got {offered}");
        assert!((ratio - 1.258).abs() < 1e-3, "got {ratio}");
    }

    /// Construct a book that prices a known φ at a known hold, then check the
    /// inversion hands the φ back. The decimal→American conversion happens
    /// here in the test, so the only code being exercised is the production
    /// inversion path.
    #[test]
    fn synthetic_book_recovers_the_phi_it_was_priced_with() {
        let (p1, p2, phi, hold) = (0.55, 0.48, 0.12, 0.15);
        let fair_joint =
            mechanical_joint(p1, p2) + phi * (p1 * (1.0 - p1) * p2 * (1.0 - p2)).sqrt();
        // The book offers the fair joint inflated by its margin.
        let decimal = 1.0 / (fair_joint * (1.0 + hold));
        // fair_joint ≈ 0.294, so decimal > 2 and the American price is positive.
        let american = (decimal - 1.0) * 100.0;
        let recovered = implied_phi_book(p1, p2, american, hold).unwrap();
        assert!((recovered - phi).abs() < 1e-9, "got {recovered}");
        // A different hold assumption must move the answer — otherwise the
        // hold parameter would be decorative. More assumed margin means less
        // correlation allowance.
        let low = implied_phi_book(p1, p2, american, 0.10).unwrap();
        let high = implied_phi_book(p1, p2, american, 0.20).unwrap();
        assert!(
            low > recovered && recovered > high,
            "expected {low} > {recovered} > {high}"
        );
    }

    /// The constructing formula inverted directly: a joint built from φ must
    /// return exactly that φ.
    #[test]
    fn phi_from_joint_inverts_the_constructing_formula() {
        let (p1, p2, phi) = (0.62, 0.41, -0.07);
        let joint = mechanical_joint(p1, p2) + phi * (p1 * (1.0 - p1) * p2 * (1.0 - p2)).sqrt();
        let recovered = phi_from_joint(p1, p2, joint).unwrap();
        assert!((recovered - phi).abs() < 1e-12, "got {recovered}");
    }

    /// φ is a property of the 2×2 table, not of which cell you happen to
    /// price: the complementary (neither-leg) quadrant yields the identical
    /// value, and the mixed quadrants — exactly one leg wins — yield its
    /// negation, because they describe disagreement.
    #[test]
    fn phi_is_identical_on_the_complementary_quadrant() {
        let (p1, p2, joint) = (0.60, 0.45, 0.30);
        let both = phi_from_joint(p1, p2, joint).unwrap();
        let neither = phi_from_joint(1.0 - p1, 1.0 - p2, 1.0 - p1 - p2 + joint).unwrap();
        assert!((both - neither).abs() < 1e-12, "{both} vs {neither}");
        let first_only = phi_from_joint(p1, 1.0 - p2, p1 - joint).unwrap();
        let second_only = phi_from_joint(1.0 - p1, p2, p2 - joint).unwrap();
        assert!((both + first_only).abs() < 1e-12, "{both} vs {first_only}");
        assert!(
            (both + second_only).abs() < 1e-12,
            "{both} vs {second_only}"
        );
    }

    /// A raw -115/-115 pair is 53.49% a side and
    /// sums to 1.0698, and devigging must return exactly 0.5 — anything less
    /// would invent a 3.5pp edge on every leg.
    #[test]
    fn devig_removes_the_margin_from_a_balanced_pair() {
        let (over, overround) = devig(-115.0, -115.0).unwrap();
        assert!((over - 0.5).abs() < 1e-12, "got {over}");
        assert!(
            (overround - 1.0697674418604652).abs() < 1e-12,
            "got {overround}"
        );
    }

    #[test]
    fn conversions_are_known_answers_not_round_trips() {
        assert!((decimal_from_american(244.0).unwrap() - 3.44).abs() < 1e-12);
        assert!((decimal_from_american(-130.0).unwrap() - (1.0 + 100.0 / 130.0)).abs() < 1e-12);
        assert!((implied_probability(-200.0).unwrap() - 2.0 / 3.0).abs() < 1e-12);
        assert!((offered_joint(244.0).unwrap() - 1.0 / 3.44).abs() < 1e-12);
    }

    /// Nothing legitimately prices between -100 and +100, and every entry
    /// point that takes a price must refuse one.
    #[test]
    fn invalid_american_prices_are_refused_everywhere() {
        assert!(implied_probability(50.0).is_none());
        assert!(implied_probability(-50.0).is_none());
        assert!(implied_probability(0.0).is_none());
        assert!(decimal_from_american(99.99).is_none());
        assert!(offered_joint(0.0).is_none());
        assert!(price_ratio(0.5, 0.5, 10.0).is_none());
        assert!(implied_phi_book(0.5, 0.5, 10.0, 0.15).is_none());
        assert!(ev_per_unit(0.3, 10.0).is_none());
        assert!(devig(-110.0, 0.0).is_none());
    }

    /// A constant outcome has no correlation with anything — and a leg
    /// probability outside [0, 1] is not a probability at all.
    #[test]
    fn degenerate_legs_have_no_correlation() {
        assert!(phi_from_joint(0.0, 0.5, 0.0).is_none());
        assert!(phi_from_joint(1.0, 0.5, 0.5).is_none());
        assert!(phi_from_joint(1.2, 0.5, 0.6).is_none());
        assert!(phi_from_joint(0.5, 0.5, 0.3).is_some());
    }

    /// +244 returns 3.44 per unit staked, so a 30% true joint is worth
    /// 0.3 × 3.44 − 1 = +0.032 per unit, and a 28% true joint is underwater.
    #[test]
    fn ev_per_unit_prices_the_true_joint() {
        let ev = ev_per_unit(0.30, 244.0).unwrap();
        assert!((ev - 0.032).abs() < 1e-9, "got {ev}");
        assert!(ev_per_unit(0.28, 244.0).unwrap() < 0.0);
    }
}
