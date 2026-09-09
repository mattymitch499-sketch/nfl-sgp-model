//! Pure statistics for the joint-probability design — see `tasks/plan.md` §1.
//!
//! [`crate::sgp`] takes the market's side of the comparison apart: what joint
//! probability is a book's same-game-parlay price charging for? This module
//! builds the other side — what joint probability the outcomes actually
//! show — and it does so without ever assuming a parametric joint. The whole
//! chain lives here, and none of it knows about football; the dataset-aware
//! layer is the consumer.
//!
//! 1. **Standardise within subject** ([`loo_standardize`], plan §1.2). The
//!    market's marginal already prices the subject's level, so pooling raw
//!    values across subjects would score "a good quarterback and a good
//!    receiver are both usually high" as co-movement. It is not co-movement,
//!    it is level, and the parlay does not pay for it. Leave-one-out, so a
//!    game never centres itself.
//! 2. **Rank to uniforms** ([`uniform_scores`], plan §1.3). Rüschendorf's
//!    distributional transform, ties broken by seeded jitter. Count stats tie
//!    constantly (passing touchdowns are 0–7) and the copula of a discrete
//!    margin is not identified without the jitter.
//! 3. **Read the survival copula at the market's marginals**
//!    ([`empirical_joint`], plan §1.4): the fraction of history in which both
//!    legs would have won at tonight's probabilities. Model-free, and the
//!    primary estimator.
//! 4. **Keep a smooth comparator beside it** ([`gaussian_rho`],
//!    [`gaussian_joint`], plan §1.5) — what an elliptical simulator of the
//!    kind a book runs would say. A large disagreement in a tail is a warning
//!    that the dependence is not elliptical there, not a reason to prefer
//!    either number.
//! 5. **Put a cluster bootstrap around the answer**
//!    ([`cluster_bootstrap_joint`], plan §1.7). One quarterback's October is
//!    one observation repeated, not sixteen, and the baseball study measured a
//!    wrongly-clustered interval 3.4x too narrow.
//!
//! The numerics ([`norm_cdf`], [`norm_ppf`], [`bvn_upper`]) are the classical
//! high-precision algorithms rather than series or table lookups, because the
//! Gaussian comparator is evaluated in tails where a percent-accurate normal
//! CDF would swamp the quantity being measured.

use crate::correlation::{MIN_CLUSTERS_FOR_INTERVAL, Rng};
use std::collections::BTreeMap;

/// The plan's `N_MIN` (§1.2): fewer games than this in a subject-season and
/// the leave-one-out mean and scale are noise, so the subject contributes
/// nothing rather than contributing garbage.
pub const N_MIN: usize = 6;

// ---------------------------------------------------------------------------
// Normal distribution numerics
// ---------------------------------------------------------------------------

/// The standard normal CDF, `Φ(x)`, to about 1e-15 relative accuracy
/// everywhere including the far tails.
///
/// Computed as `erfc(−x/√2)/2` rather than `(1 + erf(x/√2))/2`: in the left
/// tail the second form is a difference of two nearly equal numbers and loses
/// every significant digit that matters, and the left tail is exactly where
/// this is evaluated — a leg at `p = 0.02` sits at `Φ⁻¹(0.02)`, and the
/// bivariate quadrature calls `Φ` at both thresholds on every evaluation.
///
/// `NaN` in, `NaN` out; `±∞` give 1 and 0.
pub fn norm_cdf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    0.5 * erfc(-x * std::f64::consts::FRAC_1_SQRT_2)
}

/// The complementary error function, Cody's (1969) rational approximations as
/// published in SPECFUN's `CALERF` — three branches, each a ratio of
/// polynomials fitted to near machine precision on its interval.
///
/// The point of using Cody rather than a short series: relative accuracy is
/// held in the tail, not just absolute accuracy near the body. `erfc(5)` is
/// 1.5e-12 and a routine that is merely accurate to 1e-16 *absolutely* there
/// has no correct digits at all.
/// The coefficients are transcribed from the published fit at their full
/// printed precision. Clippy objects that some carry more digits than an `f64`
/// holds, and rounding them to what it prefers would mean the constants in the
/// source no longer match the paper they came from — the next reader could not
/// check them. The truncation clippy asks for is exactly what the compiler
/// already does, so the allow costs nothing and keeps the audit trail.
#[allow(clippy::excessive_precision)]
fn erfc(x: f64) -> f64 {
    /// |x| ≤ THRESH uses the erf branch.
    const THRESH: f64 = 0.46875;
    /// Below this, x² underflows the fit's leading term and erf(x) is linear.
    const XSMALL: f64 = 1.11e-16;
    /// Beyond this exp(−x²) is exactly zero in double precision, and the
    /// argument reduction below would overflow rather than say so.
    const XMAX: f64 = 27.5;
    /// 1/√π.
    const SQRT_1_PI: f64 = 0.5641895835477563;

    // |x| ≤ 0.46875
    const A: [f64; 5] = [
        3.1611237438705656,
        113.864154151050156,
        377.485237685302021,
        3209.37758913846947,
        0.185777706184603153,
    ];
    const B: [f64; 4] = [
        23.6012909523441209,
        244.024637934444173,
        1282.61652607737228,
        2844.23683343917062,
    ];
    // 0.46875 < |x| ≤ 4
    const C: [f64; 9] = [
        0.564188496988670089,
        8.88314979438837594,
        66.1191906371416295,
        298.635138197400131,
        881.952221241769090,
        1712.04761263407058,
        2051.07837782607147,
        1230.33935479799725,
        2.15311535474403846e-8,
    ];
    const D: [f64; 8] = [
        15.7449261107098347,
        117.693950891312499,
        537.181101862009858,
        1621.38957456669019,
        3290.79923573345963,
        4362.61909014324716,
        3439.36767414372164,
        1230.33935480374942,
    ];
    // |x| > 4
    const P: [f64; 6] = [
        0.305326634961232344,
        0.360344899949804439,
        0.125781726111229246,
        0.0160837851487422766,
        6.58749161529837803e-4,
        0.0163153871373020978,
    ];
    const Q: [f64; 5] = [
        2.56852019228982242,
        1.87295284992346047,
        0.527905102951428412,
        0.0605183413124413191,
        2.33520497626869185e-3,
    ];

    if x.is_nan() {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return 0.0;
    }
    if x == f64::NEG_INFINITY {
        return 2.0;
    }
    let y = x.abs();
    if y > XMAX {
        return if x > 0.0 { 0.0 } else { 2.0 };
    }

    if y <= THRESH {
        // erf itself, odd in x, so the sign rides along in the numerator.
        let z = if y > XSMALL { y * y } else { 0.0 };
        let mut num = A[4] * z;
        let mut den = z;
        for i in 0..3 {
            num = (num + A[i]) * z;
            den = (den + B[i]) * z;
        }
        return 1.0 - x * (num + A[3]) / (den + B[3]);
    }

    let value = if y <= 4.0 {
        let mut num = C[8] * y;
        let mut den = y;
        for i in 0..7 {
            num = (num + C[i]) * y;
            den = (den + D[i]) * y;
        }
        scaled_exp(y, (num + C[7]) / (den + D[7]))
    } else {
        let z = 1.0 / (y * y);
        let mut num = P[5] * z;
        let mut den = z;
        for i in 0..4 {
            num = (num + P[i]) * z;
            den = (den + Q[i]) * z;
        }
        let ratio = z * (num + P[4]) / (den + Q[4]);
        scaled_exp(y, (SQRT_1_PI - ratio) / y)
    };
    if x < 0.0 { 2.0 - value } else { value }
}

/// `exp(−y²)·r`, with Cody's argument split: `y` is rounded down to a
/// multiple of 1/16 so `ysq²` is exact, and the small remainder goes through a
/// second exponential. Computing `exp(−y²)` directly would round `y²` first
/// and lose up to a few units in the last place of the answer's *exponent*,
/// which in the tail is a relative error of order 1e-13.
fn scaled_exp(y: f64, r: f64) -> f64 {
    let ysq = (y * 16.0).trunc() / 16.0;
    let del = (y - ysq) * (y + ysq);
    (-ysq * ysq).exp() * (-del).exp() * r
}

/// The standard normal quantile function `Φ⁻¹(p)`: Wichura's AS241 PPND16,
/// three rational branches split at |p − 1/2| = 0.425 and at
/// `√(−ln min(p, 1−p)) = 5`, good to about 1e-16 relative.
///
/// Endpoints and non-probabilities, documented because the atlas will meet
/// them: `p = 0` gives `−∞`, `p = 1` gives `+∞` (the honest limits — a
/// certainty is not a finite quantile), and `p < 0`, `p > 1` or `NaN` give
/// `NaN`. Nothing panics.
#[allow(clippy::excessive_precision)] // AS241's published coefficients — see `erfc`.
pub fn norm_ppf(p: f64) -> f64 {
    const A: [f64; 8] = [
        3.387132872796366608,
        133.14166789178437745,
        1971.5909503065514427,
        13731.693765509461125,
        45921.953931549871457,
        67265.770927008700853,
        33430.575583588128105,
        2509.0809287301226727,
    ];
    const B: [f64; 7] = [
        42.313330701600911252,
        687.1870074920579083,
        5394.1960214247511077,
        21213.794301586595867,
        39307.89580009271061,
        28729.085735721942674,
        5226.495278852854561,
    ];
    const C: [f64; 8] = [
        1.42343711074968357734,
        4.6303378461565452959,
        5.7694972214606914055,
        3.64784832476320460504,
        1.27045825245236838258,
        0.24178072517745061177,
        0.0227238449892691845833,
        7.7454501427834140764e-4,
    ];
    const D: [f64; 7] = [
        2.05319162663775882187,
        1.6763848301838038494,
        0.68976733498510000455,
        0.14810397642748007459,
        0.0151986665636164571966,
        5.475938084995344946e-4,
        1.05075007164441684324e-9,
    ];
    const E: [f64; 8] = [
        6.6579046435011037772,
        5.4637849111641143699,
        1.7848265399172913358,
        0.29656057182850489123,
        0.0265321895265761230930,
        1.2426609473880784386e-3,
        2.71155556874348757815e-5,
        2.01033439929228813265e-7,
    ];
    const F: [f64; 7] = [
        0.5998322065558879377,
        0.13692988092273580531,
        0.0148753612908506148525,
        7.868691311456132591e-4,
        1.8463183175100546818e-5,
        1.4215117583164458887e-7,
        2.04426310338993978564e-15,
    ];

    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }

    let q = p - 0.5;
    if q.abs() <= 0.425 {
        // Central branch: r = 0.425² − q², a polynomial ratio in q.
        let r = 0.180625 - q * q;
        return q * poly(&A, r) / (poly(&B, r) * r + 1.0);
    }
    // Tail branches work on the *smaller* of p and 1 − p, so nothing is ever
    // computed as one-minus-a-tiny-number.
    let tail = if q < 0.0 { p } else { 1.0 - p };
    let r = (-tail.ln()).sqrt();
    let value = if r <= 5.0 {
        let r = r - 1.6;
        poly(&C, r) / (poly(&D, r) * r + 1.0)
    } else {
        let r = r - 5.0;
        poly(&E, r) / (poly(&F, r) * r + 1.0)
    };
    if q < 0.0 { -value } else { value }
}

/// Horner on coefficients given lowest-order first: `c[0] + c[1]x + …`.
fn poly(coefficients: &[f64], x: f64) -> f64 {
    coefficients.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

/// Gauss–Legendre nodes and weights on [−1, 1], positive half only; the
/// negative half is the mirror image with the same weights. Genz switches
/// order by |ρ| band: the integrand sharpens as the correlation rises.
#[allow(clippy::excessive_precision)] // Published quadrature nodes — see `erfc`.
const GL6: [(f64, f64); 3] = [
    (0.9324695142031522, 0.1713244923791705),
    (0.6612093864662647, 0.3607615730481384),
    (0.238619186083197, 0.4679139345726904),
];
#[allow(clippy::excessive_precision)] // Published quadrature nodes — see `erfc`.
const GL12: [(f64, f64); 6] = [
    (0.9815606342467191, 0.04717533638651177),
    (0.904117256370475, 0.1069393259953183),
    (0.769902674194305, 0.1600783285433464),
    (0.5873179542866171, 0.2031674267230659),
    (0.3678314989981802, 0.2334925365383547),
    (0.1252334085114692, 0.2491470458134029),
];
#[allow(clippy::excessive_precision)] // Published quadrature nodes — see `erfc`.
const GL20: [(f64, f64); 10] = [
    (0.9931285991850949, 0.01761400713915212),
    (0.9639719272779138, 0.04060142980038694),
    (0.9122344282513259, 0.06267204833410906),
    (0.8391169718222188, 0.08327674157670475),
    (0.7463319064601508, 0.1019301198172404),
    (0.636053680726515, 0.1181945319615184),
    (0.5108670019508271, 0.1316886384491766),
    (0.3737060887154196, 0.1420961093183821),
    (0.2277858511416451, 0.1491729864726037),
    (0.07652652113349733, 0.1527533871307259),
];

/// The upper-orthant probability of the standard bivariate normal:
/// `L(h, k; ρ) = P(X > h, Y > k)` with `Corr(X, Y) = ρ`.
///
/// This is the whole reason the Gaussian comparator is cheap to evaluate: one
/// number per (threshold, threshold, correlation), and the four direction
/// combinations of a two-leg parlay all reduce to it (see [`gaussian_joint`]).
///
/// Genz's (2004) algorithm, which is Drezner–Wesolowsky (1990) with the
/// singularity handled. For |ρ| < 0.925 it integrates
/// `L = Φ(−h)Φ(−k) + (1/2π)∫₀^{asin ρ} exp(−(h² + k² − 2hk sinθ)/(2cos²θ)) dθ`
/// by Gauss–Legendre, 6/12/20 points as |ρ| crosses 0.3 and 0.75. Above 0.925
/// that integrand is a spike against `cos²θ → 0`, so the second branch
/// substitutes and integrates the analytically-extracted singular part
/// instead; that branch is what keeps the near-comonotone cases (two legs of
/// one drive, say) from returning noise.
///
/// Limits are exact rather than approached: `ρ = 1` gives `1 − Φ(max(h, k))`,
/// `ρ = −1` gives `max(0, 1 − Φ(h) − Φ(k))`, an infinite threshold gives 0 or
/// the univariate tail. `NaN` for `NaN` input or |ρ| > 1 — an impossible
/// correlation is a caller bug, not a probability. The result is clamped to
/// [0, 1], since only rounding can put it outside.
pub fn bvn_upper(h: f64, k: f64, rho: f64) -> f64 {
    if h.is_nan() || k.is_nan() || rho.is_nan() || rho.abs() > 1.0 {
        return f64::NAN;
    }
    if h == f64::INFINITY || k == f64::INFINITY {
        return 0.0;
    }
    if h == f64::NEG_INFINITY {
        return if k == f64::NEG_INFINITY {
            1.0
        } else {
            norm_cdf(-k)
        };
    }
    if k == f64::NEG_INFINITY {
        return norm_cdf(-h);
    }
    if rho == 0.0 {
        return norm_cdf(-h) * norm_cdf(-k);
    }

    let two_pi = std::f64::consts::TAU;
    let nodes: &[(f64, f64)] = if rho.abs() < 0.3 {
        &GL6
    } else if rho.abs() < 0.75 {
        &GL12
    } else {
        &GL20
    };
    // Both branches sum over the mirrored node set {1 − x, 1 + x}, which maps
    // [−1, 1] onto [0, 2] so the quadrature runs over the doubled half-angle.
    let mirrored = |x: f64| [1.0 - x, 1.0 + x];

    let mut k = k;
    let mut hk = h * k;
    let mut bvn = 0.0;

    if rho.abs() < 0.925 {
        let hs = (h * h + k * k) / 2.0;
        let asr = rho.asin() / 2.0;
        let mut sum = 0.0;
        for &(x, w) in nodes {
            for t in mirrored(x) {
                let sn = (asr * t).sin();
                sum += w * ((sn * hk - hs) / (1.0 - sn * sn)).exp();
            }
        }
        bvn = sum * asr / two_pi + norm_cdf(-h) * norm_cdf(-k);
    } else {
        // Near-perfect correlation. Reflect the negative case onto the
        // positive one and put it back at the end.
        if rho < 0.0 {
            k = -k;
            hk = -hk;
        }
        if rho.abs() < 1.0 {
            let variance = 1.0 - rho * rho;
            let scale = variance.sqrt();
            let bs = (h - k) * (h - k);
            let exponent = -(bs / variance + hk) / 2.0;
            let c = (4.0 - hk) / 8.0;
            let d = (12.0 - hk) / 16.0;
            // The two analytic terms of the expansion, guarded against
            // underflow (exp(−100) is already below anything reportable).
            if exponent > -100.0 {
                bvn = scale
                    * exponent.exp()
                    * (1.0 - c * (bs - variance) * (1.0 - d * bs / 5.0) / 3.0
                        + c * d * variance * variance / 5.0);
            }
            if hk > -100.0 {
                let b = bs.sqrt();
                bvn -= (-hk / 2.0).exp()
                    * two_pi.sqrt()
                    * norm_cdf(-b / scale)
                    * b
                    * (1.0 - c * bs * (1.0 - d * bs / 5.0) / 3.0);
            }
            let half = scale / 2.0;
            let mut sum = 0.0;
            for &(x, w) in nodes {
                for t in mirrored(x) {
                    let xs = (half * t) * (half * t);
                    let exponent = -(bs / xs + hk) / 2.0;
                    if exponent > -100.0 {
                        // The two-term expansion of the singular factor
                        // exp(−hk·xs/(2(1+√(1−xs))²))/√(1−xs) = 1 + c·xs +
                        // cd·xs² + O(xs³), which is exactly what the two
                        // closed-form terms above integrate; subtracting it
                        // leaves a smooth remainder for the quadrature.
                        let series = 1.0 + c * xs * (1.0 + d * xs);
                        let root = (1.0 - xs).sqrt();
                        let ep = (-(hk / 2.0) * xs / ((1.0 + root) * (1.0 + root))).exp() / root;
                        sum += w * exponent.exp() * (series - ep);
                    }
                }
            }
            bvn = (half * sum - bvn) / two_pi;
        }
        if rho > 0.0 {
            bvn += norm_cdf(-h.max(k));
        } else if h >= k {
            bvn = -bvn;
        } else {
            // h < −k in the original orientation: the two half-lines overlap.
            let strip = if h < 0.0 {
                norm_cdf(k) - norm_cdf(h)
            } else {
                norm_cdf(-h) - norm_cdf(-k)
            };
            bvn = strip - bvn;
        }
    }
    bvn.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// §1.2 within-subject standardisation
// ---------------------------------------------------------------------------

/// Leave-one-out standardisation (plan §1.2): `z_i = (x_i − m_{−i}) / s_{−i}`,
/// where the mean and the **sample** standard deviation are taken over the
/// other `n − 1` values — denominator `n − 2`, because the leave-out set has
/// `n − 1` members and a sample SD divides by one less than that.
///
/// Leave-one-out, not full-sample, and the difference is not cosmetic: with
/// the full-sample mean every `z` is mechanically pulled towards its own
/// game, so a subject's biggest game partly centres itself and the pooled
/// ranks inherit a bias that runs in the same direction as the correlation
/// being measured. It also makes the standardisation a genuinely out-of-sample
/// statement about game `i`, which is the statement a book's simulator makes.
///
/// `None` per observation where the leave-out scale is zero or not finite (a
/// constant leave-out set says nothing about how unusual the held-out game
/// was), and `None` for every observation when there are fewer than
/// `min_games` values — the plan's [`N_MIN`] — or fewer than three, below
/// which the `n − 2` denominator does not exist.
///
/// O(n²) on purpose: `n` is one subject's games in one season, at most about
/// twenty, and the two-pass form avoids the cancellation that the
/// sum-of-squares shortcut invites when a subject's values are large and
/// tightly grouped (450, 455, 460 passing yards).
pub fn loo_standardize(values: &[f64], min_games: usize) -> Vec<Option<f64>> {
    let n = values.len();
    if n < min_games || n < 3 {
        return vec![None; n];
    }
    let denominator = (n - 2) as f64;
    (0..n)
        .map(|i| {
            let mean = values
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &x)| x)
                .sum::<f64>()
                / (n - 1) as f64;
            let sum_squares = values
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &x)| (x - mean) * (x - mean))
                .sum::<f64>();
            let sd = (sum_squares / denominator).sqrt();
            if sd > 0.0 && sd.is_finite() {
                let z = (values[i] - mean) / sd;
                if z.is_finite() { Some(z) } else { None }
            } else {
                None
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// §1.3 uniform scores
// ---------------------------------------------------------------------------

/// Uniform scores by rank with the distributional transform (plan §1.3):
/// sort by `(value, seeded jitter)`, then `u_i = (rank_i − 0.5) / n` with
/// 1-based ranks, returned in the caller's order.
///
/// The jitter is what makes this legitimate for count stats. Passing
/// touchdowns take five or six distinct values across a whole season; ranking
/// them with average ranks would put a quarter of the sample on one `u` value
/// and the copula of that margin is not identified — every threshold question
/// would be answered by the arbitrary handling of one atom. Breaking ties at
/// random is Rüschendorf's distributional transform: `u` is then exactly
/// uniform on the grid, and for a count variable with CDF `F` the event
/// `{Y ≥ k}` is `{U > F(k−1)}` almost surely, which is the identity that lets
/// a market marginal `p = P(Y ≥ k)` be read off as the region `{u > 1 − p}`
/// with no further assumption.
///
/// The seed is part of the measurement and must be recorded; re-running with
/// a second seed is how the jitter dependence is reported. A different seed
/// can only permute *within* a block of equal values — the block still
/// occupies exactly its own range of ranks — so anything the answer does when
/// the seed changes is a statement about how much of it rests on ties.
///
/// Output is always exactly a permutation of `{(r − 0.5)/n : r = 1..n}`.
pub fn uniform_scores(values: &[f64], seed: u64) -> Vec<f64> {
    let n = values.len();
    let mut rng = Rng::new(seed);
    let jitter: Vec<f64> = (0..n).map(|_| rng.uniform()).collect();
    let mut order: Vec<usize> = (0..n).collect();
    // Index last, so two identical jitters (probability ~2⁻⁵³, but a
    // reproducible atlas may not depend on luck) still order deterministically.
    order.sort_by(|&i, &j| {
        values[i]
            .total_cmp(&values[j])
            .then(jitter[i].total_cmp(&jitter[j]))
            .then(i.cmp(&j))
    });
    let mut scores = vec![0.0; n];
    for (rank, &index) in order.iter().enumerate() {
        scores[index] = (rank as f64 + 0.5) / n as f64;
    }
    scores
}

// ---------------------------------------------------------------------------
// §1.4 marginals and the empirical survival copula
// ---------------------------------------------------------------------------

/// Which side of a leg is being priced.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Direction {
    Over,
    Under,
}

/// One leg's market marginal: the devigged probability that this side wins,
/// and which side that is.
///
/// `p` is always the probability of *this* leg as stated, never of the over:
/// an under leg at `p = 0.6` means the under is a 60% shot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Marginal {
    pub p: f64,
    pub direction: Direction,
}

impl Marginal {
    /// An over leg at probability `p`.
    pub fn over(p: f64) -> Self {
        Self {
            p,
            direction: Direction::Over,
        }
    }

    /// An under leg at probability `p`.
    pub fn under(p: f64) -> Self {
        Self {
            p,
            direction: Direction::Under,
        }
    }

    /// Does an observation with uniform score `u` win this leg? Plan §1.4
    /// exactly: `Over ⇔ u > 1 − p`, `Under ⇔ u ≤ p`.
    ///
    /// The asymmetry in the inequalities is deliberate and the two regions are
    /// exact complements, so `over(p)` and `under(1 − p)` partition the sample
    /// with no observation counted twice or lost. That is what makes the
    /// Fréchet consistency identity of §1.4 hold as an identity in counts
    /// rather than up to a rounding.
    pub fn wins(&self, u: f64) -> bool {
        match self.direction {
            Direction::Over => u > 1.0 - self.p,
            Direction::Under => u <= self.p,
        }
    }
}

/// The empirical survival copula evaluated at two market marginals (plan
/// §1.4): the fraction of paired observations in which both legs would have
/// won.
///
/// This is the **primary** estimator of `P_true`, and its virtue is that it
/// assumes nothing — no elliptical joint, no functional form for the
/// dependence, no smoothing across thresholds. Its cost is granularity: it can
/// only move in steps of `1/n`, and in a tail where only a handful of
/// observations qualify it is a count, not a measurement. That is what the
/// cluster bootstrap is for.
///
/// `None` on empty or mismatched input — a joint over no observations is not
/// zero, it is unmeasured.
pub fn empirical_joint(u: &[f64], v: &[f64], a: Marginal, b: Marginal) -> Option<f64> {
    if u.is_empty() || u.len() != v.len() {
        return None;
    }
    let both = u
        .iter()
        .zip(v.iter())
        .filter(|&(&x, &y)| a.wins(x) && b.wins(y))
        .count();
    Some(both as f64 / u.len() as f64)
}

// ---------------------------------------------------------------------------
// §1.5 the Gaussian copula comparator
// ---------------------------------------------------------------------------

/// The Gaussian copula parameter: Pearson correlation of the normal scores
/// `Φ⁻¹(u)`, `Φ⁻¹(v)` (plan §1.5).
///
/// Fitting the copula on normal scores rather than on the raw values is what
/// makes this comparable with the empirical estimator: both read the same
/// ranks, so a gap between them is a statement about *shape* — the empirical
/// joint says what happened in the corner, the Gaussian one says what an
/// elliptical dependence with the same overall association would have put
/// there.
///
/// `None` on empty, mismatched or degenerate input, including any `u` at
/// exactly 0 or 1 (whose normal score is infinite). [`uniform_scores`] never
/// produces those.
pub fn gaussian_rho(u: &[f64], v: &[f64]) -> Option<f64> {
    if u.len() != v.len() {
        return None;
    }
    let scores_u: Vec<f64> = u.iter().map(|&x| norm_ppf(x)).collect();
    let scores_v: Vec<f64> = v.iter().map(|&y| norm_ppf(y)).collect();
    pearson(&scores_u, &scores_v)
}

/// The Gaussian copula's joint probability for two legs at their market
/// marginals (plan §1.5), through [`bvn_upper`].
///
/// Every direction combination reduces to one upper-orthant call. Write
/// `N_A = Φ⁻¹(U_A)`, standard normal, with `Corr(N_A, N_B) = ρ`, and
/// `h_A = Φ⁻¹(1 − p_A)`:
///
/// * **A over** is `U_A > 1 − p_A`, i.e. `N_A > h_A`. Take the variable as
///   `+N_A` with threshold `h_A`.
/// * **A under** is `U_A ≤ p_A`, i.e. `N_A ≤ Φ⁻¹(p_A) = −h_A`, i.e.
///   `−N_A ≥ h_A`. Take the variable as `−N_A`, again with threshold `h_A` —
///   the threshold is the *same* number, and the sign flip lands on the
///   variable instead. (The boundary moves from `≥` to `>`, which is a null
///   set under a continuous law.)
///
/// Since `Corr(s_A N_A, s_B N_B) = s_A s_B ρ` for signs `s ∈ {+1, −1}`, the
/// four cases are:
///
/// | A | B | value |
/// |---|---|---|
/// | Over | Over | `L(h_A, h_B; ρ)` |
/// | Over | Under | `L(h_A, h_B; −ρ)` |
/// | Under | Over | `L(h_A, h_B; −ρ)` |
/// | Under | Under | `L(h_A, h_B; ρ)` |
///
/// so one sign flip on either leg negates `ρ` and two flips restore it —
/// which is the same symmetry that makes φ identical on the complementary
/// quadrant in [`crate::sgp::phi_from_joint`].
///
/// `h` is computed as `−Φ⁻¹(p)` rather than `Φ⁻¹(1 − p)`: they are the same
/// number, but forming `1 − p` first would throw away the low-order bits of a
/// heavy favourite's marginal.
pub fn gaussian_joint(rho: f64, a: Marginal, b: Marginal) -> f64 {
    let h_a = -norm_ppf(a.p);
    let h_b = -norm_ppf(b.p);
    let signed = match (a.direction, b.direction) {
        (Direction::Over, Direction::Over) | (Direction::Under, Direction::Under) => rho,
        (Direction::Over, Direction::Under) | (Direction::Under, Direction::Over) => -rho,
    };
    bvn_upper(h_a, h_b, signed)
}

// ---------------------------------------------------------------------------
// §1.6 rank summaries and φ
// ---------------------------------------------------------------------------

/// Spearman's ρ: Pearson correlation of the uniform scores.
///
/// Takes the scores from [`uniform_scores`], not raw values — on the exact
/// rank grid Pearson-of-uniforms *is* Spearman, and taking the scores as the
/// input keeps one definition of "the ranks" in the atlas instead of two.
pub fn spearman(u: &[f64], v: &[f64]) -> Option<f64> {
    pearson(u, v)
}

/// Kendall's τ-b: `(C − D)` over the tie-corrected pair count.
///
/// τ and Spearman answer different questions about the same ranks — τ is a
/// probability of concordance, Spearman a correlation of ranks — and for a
/// Gaussian copula they pin ρ by different maps (`τ = (2/π)asin ρ`,
/// `ρ_S = (6/π)asin(ρ/2)`). Reporting both is the cheapest available check
/// that the dependence is roughly elliptical before the Gaussian comparator is
/// read.
///
/// τ-b rather than τ-a, because count stats tie: the denominator discounts
/// pairs tied in either coordinate rather than scoring them as disagreement.
/// O(n log n) by merge-sort inversion counting — the atlas pools tens of
/// thousands of observations per family and the quadratic form would dominate
/// its runtime.
///
/// `None` when the samples differ in length, hold fewer than two observations,
/// or one coordinate is entirely tied.
pub fn kendall_tau(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len();
    if n != y.len() || n < 2 {
        return None;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| x[i].total_cmp(&x[j]).then(y[i].total_cmp(&y[j])));
    let sorted_x: Vec<f64> = order.iter().map(|&i| x[i]).collect();
    let mut sorted_y: Vec<f64> = order.iter().map(|&i| y[i]).collect();

    let total = pair_count(n);
    let tied_x = tie_pairs(&sorted_x);
    // Sorted by (x, y), so observations tied in both are already adjacent.
    let tied_both = tie_pairs_2d(&sorted_x, &sorted_y);
    let tied_y = {
        let mut only_y = sorted_y.clone();
        only_y.sort_by(f64::total_cmp);
        tie_pairs(&only_y)
    };
    // Within a run of tied x the y values are already ascending, so an
    // inversion of y can only come from a pair with x strictly ordered: it is
    // a genuinely discordant pair.
    let mut scratch = Vec::with_capacity(n);
    let discordant = count_inversions(&mut sorted_y, &mut scratch);

    let concordant_minus_discordant =
        total as i64 - tied_x as i64 - tied_y as i64 + tied_both as i64 - 2 * discordant as i64;
    let denominator = ((total - tied_x) as f64).sqrt() * ((total - tied_y) as f64).sqrt();
    if denominator <= 0.0 {
        return None;
    }
    Some(concordant_minus_discordant as f64 / denominator)
}

/// φ at a joint, for comparability with the baseball atlas (plan §1.6).
///
/// The whole NFL study is stated in joint probabilities, which is the quantity
/// the price actually pays on; φ is carried only so that a family here can be
/// put beside `reference/baseball/`'s numbers, where φ was the headline. It is
/// the same inversion the book side uses, so the two remain comparable.
///
/// `None` when either leg is degenerate — a certainty has no correlation with
/// anything.
pub fn phi_at(joint: f64, a: Marginal, b: Marginal) -> Option<f64> {
    crate::sgp::phi_from_joint(a.p, b.p, joint)
}

// ---------------------------------------------------------------------------
// §1.7 uncertainty
// ---------------------------------------------------------------------------

/// Cluster-bootstrap 95% interval for an empirical joint (plan §1.7).
///
/// Takes the **standardised** values of §1.2, not the uniform scores, because
/// the ranks are a property of the pool and the pool changes on every
/// replicate: a resample that happens to draw three copies of one
/// quarterback-season must be ranked as the sample it is. Recomputing the
/// scores inside the loop — including fresh jitter, drawn from the same
/// stream — is what makes the interval reflect the rank estimation as well as
/// the sampling.
///
/// Whole clusters are resampled with replacement. Games by one quarterback in
/// one season are not independent draws, and resampling rows would understate
/// the interval by exactly the within-cluster dependence the measurement
/// exists to respect.
///
/// Plain 2.5/97.5 percentiles, matching [`crate::correlation`]'s φ interval so
/// the two intervals in one atlas mean the same thing. Deterministic: the seed
/// selects every draw and every jitter. `None` below
/// [`MIN_CLUSTERS_FOR_INTERVAL`] distinct clusters, on mismatched or empty
/// input, or when no replicate produced a joint.
pub fn cluster_bootstrap_joint(
    z_a: &[f64],
    z_b: &[f64],
    clusters: &[u64],
    a: Marginal,
    b: Marginal,
    reps: usize,
    seed: u64,
) -> Option<(f64, f64)> {
    cluster_bootstrap_joints(z_a, z_b, clusters, &[(a, b)], reps, seed)?
        .into_iter()
        .next()?
}

/// The same bootstrap, evaluating **many** marginal pairs against one set of
/// resamples.
///
/// Neither the cluster draw nor the re-ranking depends on which marginals are
/// being asked about, so a caller wanting a whole grid of intervals should pay
/// for the resamples once. This is not a micro-optimisation: the atlas asks for
/// 36 points per stratum, and running the loop once per point meant re-ranking
/// eight thousand observations 36,000 times per stratum, which did not finish.
///
/// It is also the more defensible construction. Every returned interval comes
/// from the *identical* resamples, so the grid cannot contain two points whose
/// intervals disagree about the sampling distribution they were built from.
///
/// Returns one interval per input pair, in order; a `None` in a slot means that
/// pair produced no usable replicate. `None` overall means the input was
/// unusable — empty, mismatched, or fewer than [`MIN_CLUSTERS_FOR_INTERVAL`]
/// distinct clusters.
pub fn cluster_bootstrap_joints(
    z_a: &[f64],
    z_b: &[f64],
    clusters: &[u64],
    pairs: &[(Marginal, Marginal)],
    reps: usize,
    seed: u64,
) -> Option<Vec<Option<(f64, f64)>>> {
    if z_a.is_empty() || z_a.len() != z_b.len() || z_a.len() != clusters.len() || pairs.is_empty() {
        return None;
    }
    let mut by_cluster: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    for (index, &cluster) in clusters.iter().enumerate() {
        by_cluster.entry(cluster).or_default().push(index);
    }
    if by_cluster.len() < MIN_CLUSTERS_FOR_INTERVAL {
        return None;
    }
    // BTreeMap, so the group order is the cluster id order and the draws are
    // reproducible across runs and platforms.
    let groups: Vec<&Vec<usize>> = by_cluster.values().collect();
    let mut rng = Rng::new(seed);
    let mut replicates: Vec<Vec<f64>> = vec![Vec::with_capacity(reps); pairs.len()];
    let mut pool_a: Vec<f64> = Vec::with_capacity(z_a.len());
    let mut pool_b: Vec<f64> = Vec::with_capacity(z_b.len());
    for _ in 0..reps {
        pool_a.clear();
        pool_b.clear();
        for _ in 0..groups.len() {
            for &index in groups[rng.below(groups.len())] {
                pool_a.push(z_a[index]);
                pool_b.push(z_b[index]);
            }
        }
        // Two seeds, one per margin: sharing one would tie the two margins'
        // tie-breaking together and manufacture dependence out of ties alone.
        let (seed_a, seed_b) = (rng.next(), rng.next());
        let u = uniform_scores(&pool_a, seed_a);
        let v = uniform_scores(&pool_b, seed_b);
        for (slot, &(a, b)) in pairs.iter().enumerate() {
            if let Some(joint) = empirical_joint(&u, &v, a, b) {
                replicates[slot].push(joint);
            }
        }
    }
    Some(
        replicates
            .into_iter()
            .map(|mut draws| {
                if draws.is_empty() {
                    return None;
                }
                draws.sort_by(f64::total_cmp);
                Some((percentile(&draws, 0.025), percentile(&draws, 0.975)))
            })
            .collect(),
    )
}

/// Cluster-bootstrap interval for the **difference** between two independent
/// samples' joints, evaluated at many marginal pairs from one set of draws.
///
/// This exists because the obvious way to test transport is wrong. Comparing a
/// point estimate fitted on the early seasons against an interval built only
/// from the late ones asks whether the *training* number falls inside the
/// *test* number's uncertainty — but the training number has sampling error of
/// its own, and when the two halves are a similar size that error is just as
/// large. The comparison then rejects far more often than its nominal rate, and
/// the rejections are hairline: measured on the NFL data it failed a family
/// whose predictions were off by an average of 0.004, on misses as small as
/// 0.00006.
///
/// Resampling both sides and bootstrapping the difference puts both sources of
/// error where they belong. A transport check is then "is zero inside the
/// interval for `train − test`", which is the question that was meant all
/// along.
///
/// The two sides are resampled from one RNG stream but independently of each
/// other, which is correct: the seasons are disjoint, so the samples are
/// independent and their difference's variance is the sum of theirs.
///
/// Returns one interval per pair, in order. `None` overall when either side is
/// unusable.
#[allow(clippy::too_many_arguments)]
pub fn cluster_bootstrap_joint_difference(
    train_a: &[f64],
    train_b: &[f64],
    train_clusters: &[u64],
    test_a: &[f64],
    test_b: &[f64],
    test_clusters: &[u64],
    pairs: &[(Marginal, Marginal)],
    reps: usize,
    seed: u64,
) -> Option<Vec<Option<(f64, f64)>>> {
    let train_groups = grouped(train_a, train_b, train_clusters, pairs)?;
    let test_groups = grouped(test_a, test_b, test_clusters, pairs)?;
    let mut rng = Rng::new(seed);
    let mut replicates: Vec<Vec<f64>> = vec![Vec::with_capacity(reps); pairs.len()];
    for _ in 0..reps {
        let train = one_replicate(train_a, train_b, &train_groups, pairs, &mut rng);
        let test = one_replicate(test_a, test_b, &test_groups, pairs, &mut rng);
        for (slot, (left, right)) in train.into_iter().zip(test).enumerate() {
            if let (Some(left), Some(right)) = (left, right) {
                replicates[slot].push(left - right);
            }
        }
    }
    Some(
        replicates
            .into_iter()
            .map(|mut draws| {
                if draws.is_empty() {
                    return None;
                }
                draws.sort_by(f64::total_cmp);
                Some((percentile(&draws, 0.025), percentile(&draws, 0.975)))
            })
            .collect(),
    )
}

/// Cluster index groups, or `None` when the sample cannot support a bootstrap.
fn grouped(
    z_a: &[f64],
    z_b: &[f64],
    clusters: &[u64],
    pairs: &[(Marginal, Marginal)],
) -> Option<Vec<Vec<usize>>> {
    if z_a.is_empty() || z_a.len() != z_b.len() || z_a.len() != clusters.len() || pairs.is_empty() {
        return None;
    }
    let mut by_cluster: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    for (index, &cluster) in clusters.iter().enumerate() {
        by_cluster.entry(cluster).or_default().push(index);
    }
    if by_cluster.len() < MIN_CLUSTERS_FOR_INTERVAL {
        return None;
    }
    Some(by_cluster.into_values().collect())
}

/// One resample: draw whole clusters, re-rank the drawn pool, read every pair.
fn one_replicate(
    z_a: &[f64],
    z_b: &[f64],
    groups: &[Vec<usize>],
    pairs: &[(Marginal, Marginal)],
    rng: &mut Rng,
) -> Vec<Option<f64>> {
    let mut pool_a = Vec::with_capacity(z_a.len());
    let mut pool_b = Vec::with_capacity(z_b.len());
    for _ in 0..groups.len() {
        for &index in &groups[rng.below(groups.len())] {
            pool_a.push(z_a[index]);
            pool_b.push(z_b[index]);
        }
    }
    let (seed_a, seed_b) = (rng.next(), rng.next());
    let u = uniform_scores(&pool_a, seed_a);
    let v = uniform_scores(&pool_b, seed_b);
    pairs
        .iter()
        .map(|&(a, b)| empirical_joint(&u, &v, a, b))
        .collect()
}

// ---------------------------------------------------------------------------
// A dependence object, for the atlas to hold
// ---------------------------------------------------------------------------

/// One (family, stratum)'s dependence: the paired uniform scores, and the
/// questions the atlas asks of them.
///
/// The atlas stores exactly this per stratum (plan §6) — the scorer needs the
/// `(u, v)` arrays themselves, not a summary, because tonight's marginals are
/// not known when the atlas is built and the joint has to be re-read at
/// whatever thresholds the market hands over.
#[derive(Clone, Debug, PartialEq)]
pub struct Dependence {
    pub u: Vec<f64>,
    pub v: Vec<f64>,
}

impl Dependence {
    /// Ranks two standardised series into paired uniform scores. The two
    /// margins get separate jitter streams derived from `seed`, so ties in one
    /// leg cannot line up with ties in the other by construction.
    ///
    /// `None` on mismatched or empty input.
    pub fn from_standardised(z_a: &[f64], z_b: &[f64], seed: u64) -> Option<Self> {
        if z_a.is_empty() || z_a.len() != z_b.len() {
            return None;
        }
        let mut rng = Rng::new(seed);
        let (seed_a, seed_b) = (rng.next(), rng.next());
        Some(Self {
            u: uniform_scores(z_a, seed_a),
            v: uniform_scores(z_b, seed_b),
        })
    }

    /// Paired observations behind the estimate.
    pub fn len(&self) -> usize {
        self.u.len()
    }

    /// No observations, so nothing to evaluate.
    pub fn is_empty(&self) -> bool {
        self.u.is_empty()
    }

    /// [`empirical_joint`] at two marginals.
    pub fn joint(&self, a: Marginal, b: Marginal) -> Option<f64> {
        empirical_joint(&self.u, &self.v, a, b)
    }

    /// [`spearman`] of the pair.
    pub fn spearman(&self) -> Option<f64> {
        spearman(&self.u, &self.v)
    }

    /// [`gaussian_rho`] of the pair.
    pub fn gaussian_rho(&self) -> Option<f64> {
        gaussian_rho(&self.u, &self.v)
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Pearson correlation. `None` on mismatched, too-short, degenerate or
/// non-finite input rather than a NaN that would travel silently into an
/// atlas row.
fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len();
    if n != y.len() || n < 2 {
        return None;
    }
    let mean_x = x.iter().sum::<f64>() / n as f64;
    let mean_y = y.iter().sum::<f64>() / n as f64;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (&a, &b) in x.iter().zip(y.iter()) {
        let (da, db) = (a - mean_x, b - mean_y);
        sxy += da * db;
        sxx += da * da;
        syy += db * db;
    }
    let denominator = (sxx * syy).sqrt();
    if !denominator.is_finite() || denominator <= 0.0 {
        return None;
    }
    let r = sxy / denominator;
    if r.is_finite() { Some(r) } else { None }
}

/// `n(n − 1)/2`, the number of unordered pairs.
fn pair_count(n: usize) -> u64 {
    let n = n as u64;
    n * (n - 1) / 2
}

/// Tied pairs in a sorted slice: `Σ t(t − 1)/2` over runs of equal values.
fn tie_pairs(sorted: &[f64]) -> u64 {
    let mut total = 0;
    let mut run = 1usize;
    for window in sorted.windows(2) {
        if window[0].total_cmp(&window[1]).is_eq() {
            run += 1;
        } else {
            total += pair_count(run);
            run = 1;
        }
    }
    total + pair_count(run)
}

/// Tied pairs in a slice sorted lexicographically by `(x, y)`: runs equal in
/// **both** coordinates, which are adjacent under that order.
fn tie_pairs_2d(x: &[f64], y: &[f64]) -> u64 {
    let mut total = 0;
    let mut run = 1usize;
    for i in 1..x.len() {
        if x[i - 1].total_cmp(&x[i]).is_eq() && y[i - 1].total_cmp(&y[i]).is_eq() {
            run += 1;
        } else {
            total += pair_count(run);
            run = 1;
        }
    }
    if x.is_empty() {
        0
    } else {
        total + pair_count(run)
    }
}

/// Counts pairs `i < j` with `v[i] > v[j]`, sorting `v` ascending on the way.
/// A merge sort: when an element of the right half is taken before the rest of
/// the left half, it is smaller than every one of them, and that is exactly
/// the number of inversions it closes.
fn count_inversions(v: &mut [f64], scratch: &mut Vec<f64>) -> u64 {
    let n = v.len();
    if n < 2 {
        return 0;
    }
    let middle = n / 2;
    let mut count = {
        let (left, right) = v.split_at_mut(middle);
        count_inversions(left, scratch) + count_inversions(right, scratch)
    };
    scratch.clear();
    {
        let (left, right) = v.split_at(middle);
        let (mut i, mut j) = (0, 0);
        while i < left.len() && j < right.len() {
            if left[i] <= right[j] {
                scratch.push(left[i]);
                i += 1;
            } else {
                count += (left.len() - i) as u64;
                scratch.push(right[j]);
                j += 1;
            }
        }
        scratch.extend_from_slice(&left[i..]);
        scratch.extend_from_slice(&right[j..]);
    }
    v.copy_from_slice(scratch);
    count
}

/// Linear-interpolation percentile of a sorted slice.
///
/// The same rule [`crate::correlation`] uses for its φ interval, repeated
/// rather than shared because that one is private to its module: two intervals
/// printed in one atlas row must be built the same way, and this is the
/// cheapest guarantee of that. Panics on an empty slice, which the one caller
/// rules out.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{PI, TAU};

    // -----------------------------------------------------------------
    // helpers shared by the synthetic-DGP tests
    // -----------------------------------------------------------------

    /// Box–Muller on the vendored xorshift: two independent standard normals
    /// per call, no external randomness anywhere in the test suite.
    fn box_muller(rng: &mut Rng) -> (f64, f64) {
        // uniform() can return exactly 0 and ln(0) is −∞; 1 − u lands in (0, 1].
        let u1 = 1.0 - rng.uniform();
        let u2 = rng.uniform();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = TAU * u2;
        (radius * angle.cos(), radius * angle.sin())
    }

    /// `n` draws from a standard bivariate normal with correlation `rho`.
    fn bivariate_normal(n: usize, rho: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut rng = Rng::new(seed);
        let root = (1.0 - rho * rho).sqrt();
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for _ in 0..n {
            let (z1, z2) = box_muller(&mut rng);
            xs.push(z1);
            ys.push(rho * z1 + root * z2);
        }
        (xs, ys)
    }

    /// The standard normal density, for the independent quadrature below.
    fn norm_pdf(x: f64) -> f64 {
        (-0.5 * x * x).exp() / TAU.sqrt()
    }

    /// `P(X > h, Y > k)` by an entirely different route from [`bvn_upper`]:
    /// condition on X, so `Y | X = x ~ N(ρx, 1 − ρ²)` and
    /// `L = ∫_h^∞ φ(x) Φ((ρx − k)/√(1 − ρ²)) dx`, then Simpson's rule on a
    /// fine grid. Shares nothing with Genz's quadrature but the normal CDF,
    /// which is pinned against published values on its own.
    fn bvn_by_quadrature(h: f64, k: f64, rho: f64) -> f64 {
        let root = (1.0 - rho * rho).sqrt();
        // φ(12) is 2e-32; the tails outside this window are far below the
        // 1e-9 the comparison is made at.
        let low = h.max(-12.0);
        let high = 12.0;
        if low >= high {
            return 0.0;
        }
        let steps = 100_000usize; // even, as Simpson requires
        let width = (high - low) / steps as f64;
        let integrand = |x: f64| norm_pdf(x) * norm_cdf((rho * x - k) / root);
        let mut total = integrand(low) + integrand(high);
        for step in 1..steps {
            let x = low + width * step as f64;
            total += integrand(x) * if step % 2 == 1 { 4.0 } else { 2.0 };
        }
        total * width / 3.0
    }

    /// The reporting grid of plan §1.6, all four direction combinations.
    fn reporting_grid() -> Vec<(Marginal, Marginal)> {
        let mut grid = Vec::new();
        for &p_a in &[0.3, 0.5, 0.7] {
            for &p_b in &[0.3, 0.5, 0.7] {
                for a in [Marginal::over(p_a), Marginal::under(p_a)] {
                    for b in [Marginal::over(p_b), Marginal::under(p_b)] {
                        grid.push((a, b));
                    }
                }
            }
        }
        grid
    }

    // -----------------------------------------------------------------
    // Φ and Φ⁻¹
    // -----------------------------------------------------------------

    /// Published values of the standard normal CDF. The tail entries are the
    /// point of the exercise: they are checked in *relative* terms, which a
    /// `(1 + erf)/2` implementation would fail outright.
    #[test]
    fn norm_cdf_matches_published_values() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-16, "{}", norm_cdf(0.0));
        for &(x, expected) in &[
            (1.0, 0.841344746068543),
            (-1.0, 0.158655253931457),
            (-1.96, 0.024997895148220),
            (1.96, 0.975002104851780),
            (3.0, 0.998650101968370),
            (-2.5, 0.006209665325776),
        ] {
            let got = norm_cdf(x);
            assert!(
                (got - expected).abs() < 1e-14,
                "Phi({x}) = {got}, expected {expected}"
            );
        }
        // Deep tails, relative accuracy.
        for &(x, expected) in &[
            (-5.0, 2.866515718791939e-7),
            (-6.0, 9.865876450376946e-10),
            (-8.0, 6.220960574271786e-16),
        ] {
            let got = norm_cdf(x);
            assert!(
                ((got - expected) / expected).abs() < 1e-13,
                "Phi({x}) = {got:e}, expected {expected:e}"
            );
        }
        // The upper tail as the plan states it: Phi(6) = 1 − 9.8658…e-10.
        let upper = norm_cdf(6.0);
        assert!(
            (upper - (1.0 - 9.865876450376946e-10)).abs() < 1e-15,
            "Phi(6) = {upper}"
        );
        assert!((norm_cdf(f64::NEG_INFINITY)).abs() < 1e-300);
        assert!((norm_cdf(f64::INFINITY) - 1.0).abs() < 1e-300);
        assert!(norm_cdf(f64::NAN).is_nan());
    }

    /// Published quantiles, to the last digit a double carries.
    #[test]
    fn norm_ppf_matches_published_quantiles() {
        for &(p, expected) in &[
            (0.5, 0.0),
            (0.975, 1.959963984540054),
            (0.025, -1.959963984540054),
            (0.95, 1.6448536269514722),
            (0.99, 2.3263478740408408),
            (0.001, -3.090232306167813),
            (0.999, 3.090232306167813),
        ] {
            let got = norm_ppf(p);
            assert!(
                (got - expected).abs() < 1e-13,
                "ppf({p}) = {got}, expected {expected}"
            );
        }
    }

    /// Φ⁻¹ inverts Φ across the whole usable range. The grid stops at +4 on
    /// the right: past there `Φ(x)` is one-minus-a-tiny-number and the round
    /// trip measures double-precision subtraction, not either function.
    #[test]
    fn norm_ppf_round_trips_norm_cdf() {
        let mut x = -8.0;
        while x <= 4.0 {
            let round_trip = norm_ppf(norm_cdf(x));
            assert!(
                (round_trip - x).abs() < 1e-12,
                "x = {x} came back as {round_trip}"
            );
            x += 0.05;
        }
        // And the other way, in the lower tail where relative accuracy holds.
        for &p in &[1e-300, 1e-100, 1e-20, 1e-8, 0.001, 0.1, 0.4, 0.5] {
            let back = norm_cdf(norm_ppf(p));
            assert!(
                ((back - p) / p).abs() < 1e-12,
                "p = {p:e} came back as {back:e}"
            );
        }
    }

    /// The endpoints are limits, not errors, and a non-probability is NaN.
    #[test]
    fn norm_ppf_endpoints_and_non_probabilities() {
        assert_eq!(norm_ppf(0.0), f64::NEG_INFINITY);
        assert_eq!(norm_ppf(1.0), f64::INFINITY);
        assert!(norm_ppf(-0.1).is_nan());
        assert!(norm_ppf(1.1).is_nan());
        assert!(norm_ppf(f64::NAN).is_nan());
    }

    // -----------------------------------------------------------------
    // the bivariate normal upper orthant
    // -----------------------------------------------------------------

    /// Sheppard's exact formula for the orthant at the origin:
    /// `L(0, 0; ρ) = 1/4 + asin(ρ)/(2π)`. The ρ = 0.95 case exercises the
    /// near-singular branch, which is otherwise only reached in production by
    /// two legs of the same drive.
    #[test]
    fn bvn_upper_matches_sheppards_formula() {
        for &rho in &[-0.99f64, -0.9, -0.5, -0.25, 0.0, 0.3, 0.7, 0.95, 0.999] {
            let expected = 0.25 + rho.asin() / TAU;
            let got = bvn_upper(0.0, 0.0, rho);
            assert!(
                (got - expected).abs() < 1e-10,
                "L(0,0;{rho}) = {got}, expected {expected}"
            );
        }
    }

    /// Independence is the product of the two tails — the benchmark the whole
    /// study is measured against, so it had better be exact.
    #[test]
    fn bvn_upper_at_zero_correlation_is_the_product() {
        for &(h, k) in &[
            (0.0, 0.0),
            (0.5, -1.25),
            (-2.0, 2.0),
            (1.6448536269514722, 1.2815515655446004),
            (3.0, 3.0),
        ] {
            let expected = norm_cdf(-h) * norm_cdf(-k);
            let got = bvn_upper(h, k, 0.0);
            assert!(
                (got - expected).abs() < 1e-15,
                "L({h},{k};0) = {got}, expected {expected}"
            );
        }
    }

    /// The Fréchet limits: ρ = 1 makes the two events nested, ρ = −1 makes
    /// them a strip.
    #[test]
    fn bvn_upper_at_the_comonotone_and_countermonotone_limits() {
        for &(h, k) in &[
            (0.0, 0.0),
            (0.5, -1.25),
            (-2.0, 2.0),
            (1.0, 1.0),
            (2.0, 1.5),
        ] {
            let comonotone = bvn_upper(h, k, 1.0);
            let expected = 1.0 - norm_cdf(h.max(k));
            assert!(
                (comonotone - expected).abs() < 1e-12,
                "L({h},{k};1) = {comonotone}, expected {expected}"
            );
            let countermonotone = bvn_upper(h, k, -1.0);
            let expected = (1.0 - norm_cdf(h) - norm_cdf(k)).max(0.0);
            assert!(
                (countermonotone - expected).abs() < 1e-12,
                "L({h},{k};-1) = {countermonotone}, expected {expected}"
            );
            // Approaching a limit must not jump, and the gap must close at
            // the right rate: it is O(√(1 − ρ²)), so a ρ a million times
            // closer to 1 leaves a gap a thousand times smaller. Measured:
            // 2.25e-4 then 2.25e-7 at (0, 0). A branch that merely happened
            // to be continuous would not track that rate.
            for (sign, limit) in [(1.0, comonotone), (-1.0, countermonotone)] {
                let near = (bvn_upper(h, k, sign * 0.999_999) - limit).abs();
                let nearer = (bvn_upper(h, k, sign * (1.0 - 1e-12)) - limit).abs();
                assert!(near < 1e-3, "rho -> {sign} at ({h},{k}): gap {near:e}");
                assert!(nearer < 1e-6, "rho -> {sign} at ({h},{k}): gap {nearer:e}");
                if near > 0.0 {
                    assert!(
                        nearer < near / 100.0,
                        "rho -> {sign} at ({h},{k}): {near:e} then {nearer:e} is too slow"
                    );
                }
            }
        }
    }

    /// Swapping the two legs cannot change a joint probability, and the
    /// complement identity `L(h, k; ρ) + L(h, −k; −ρ) = 1 − Φ(h)` is the
    /// same partition of the sample that [`Marginal::wins`] relies on.
    #[test]
    fn bvn_upper_symmetry_and_the_reflection_identity() {
        for &rho in &[-0.95, -0.4, 0.0, 0.2, 0.6, 0.93] {
            for &(h, k) in &[
                (0.0, 0.0),
                (0.4, -0.9),
                (-1.5, 0.75),
                (2.1, 1.3),
                (-2.0, -2.0),
            ] {
                let straight = bvn_upper(h, k, rho);
                let swapped = bvn_upper(k, h, rho);
                assert!(
                    (straight - swapped).abs() < 1e-14,
                    "L({h},{k};{rho}) = {straight} but L({k},{h};{rho}) = {swapped}"
                );
                let reflected = bvn_upper(h, -k, -rho);
                let expected = 1.0 - norm_cdf(h);
                assert!(
                    (straight + reflected - expected).abs() < 1e-12,
                    "L({h},{k};{rho}) + L({h},{};{}) = {}, expected {expected}",
                    -k,
                    -rho,
                    straight + reflected
                );
            }
        }
    }

    /// Independent verification of the orthant at a spread of (h, k, ρ),
    /// including both quadrature branches and both signs of ρ.
    ///
    /// These are **not** transcribed table values: rather than trust a
    /// remembered digit from Genz's or the NBS tables, each point is derived
    /// here by conditioning on X and integrating
    /// `φ(x)·Φ((ρx − k)/√(1 − ρ²))` with Simpson's rule on 100,000 intervals,
    /// which has nothing in common with the algorithm under test.
    #[test]
    fn bvn_upper_matches_independent_numerical_integration() {
        let cases = [
            (0.5, 0.5, 0.5),
            (0.0, 0.0, 0.5),
            (-1.0, 0.5, 0.25),
            (1.0, 1.0, -0.5),
            (0.5, -1.0, 0.8),
            (2.0, 1.0, 0.9),
            (1.5, 1.5, 0.95),
            (-0.5, 2.5, -0.85),
            (2.5, 2.5, 0.99),
            (0.0, 1.0, -0.95),
            (3.0, 0.0, 0.6),
        ];
        let mut worst = 0.0f64;
        for &(h, k, rho) in &cases {
            let got = bvn_upper(h, k, rho);
            let reference = bvn_by_quadrature(h, k, rho);
            let error = (got - reference).abs();
            worst = worst.max(error);
            assert!(
                error < 1e-9,
                "L({h},{k};{rho}) = {got}, quadrature says {reference} (error {error:e})"
            );
        }
        println!("bvn_upper vs Simpson quadrature: worst absolute error {worst:e}");
    }

    /// Fixed reference values, so the orthant cannot drift silently under a
    /// later edit the way it did once already during this build (a wrong
    /// coefficient in the near-singular expansion cost 1e-5 and nothing else
    /// in the suite noticed).
    ///
    /// Provenance, stated because it matters: these are **not** digits
    /// transcribed from a table out of memory. Each was produced by the
    /// independent Simpson quadrature above and cross-checked against SciPy's
    /// `multivariate_normal.cdf` outside the test suite, where the largest
    /// disagreement over the eleven points of the test above was 1.3e-16
    /// absolute. `L(0.5, 0.5; 0.5)` rounds to the 0.1633195 that the classic
    /// bivariate-normal tables print for that argument.
    ///
    /// The plan asks for 1e-7 here; these hold to 1e-12.
    #[test]
    // Quoted to every digit the quadrature produced, which is the point of
    // pinning them — see `erfc`.
    #[allow(clippy::excessive_precision)]
    fn bvn_upper_matches_pinned_reference_values() {
        for &(h, k, rho, expected) in &[
            (0.5, 0.5, 0.5, 0.16331952130906346),
            (0.0, 0.0, 0.5, 1.0 / 3.0),
            (-1.0, 0.5, 0.25, 0.27951123698829289),
            (1.0, 1.0, -0.5, 3.7823020728542378e-3),
            (0.5, -1.0, 0.8, 0.30778363133093523),
            (2.0, 1.0, 0.9, 2.2501572916410861e-2),
            (1.5, 1.5, 0.95, 5.0554204795644653e-2),
            (2.5, 2.5, 0.99, 5.2250605843405795e-3),
            (3.0, 0.0, 0.6, 1.339525091436089e-3),
            (-0.5, 2.5, -0.85, 2.1457121928691869e-6),
            (0.0, 1.0, -0.95, 2.3903160865423878e-5),
        ] {
            let got = bvn_upper(h, k, rho);
            assert!(
                (got - expected).abs() < 1e-12,
                "L({h},{k};{rho}) = {got}, reference {expected}"
            );
        }
    }

    /// Infinite thresholds are the degenerate legs, and an impossible
    /// correlation is refused rather than clamped.
    #[test]
    fn bvn_upper_edge_arguments() {
        assert_eq!(bvn_upper(f64::INFINITY, 0.0, 0.5), 0.0);
        assert_eq!(bvn_upper(0.0, f64::INFINITY, 0.5), 0.0);
        assert_eq!(bvn_upper(f64::NEG_INFINITY, f64::NEG_INFINITY, 0.5), 1.0);
        assert!((bvn_upper(f64::NEG_INFINITY, 1.0, 0.5) - norm_cdf(-1.0)).abs() < 1e-16);
        assert!((bvn_upper(1.0, f64::NEG_INFINITY, 0.5) - norm_cdf(-1.0)).abs() < 1e-16);
        assert!(bvn_upper(0.0, 0.0, 1.5).is_nan());
        assert!(bvn_upper(f64::NAN, 0.0, 0.5).is_nan());
    }

    // -----------------------------------------------------------------
    // §1.2 leave-one-out standardisation
    // -----------------------------------------------------------------

    /// Hand-computed. For 1..6 and i = 0 the leave-out set is 2..6: mean 4,
    /// squared deviations 4+1+0+1+4 = 10 over n−2 = 4, so s = √2.5 and
    /// z = −3/√2.5. For i = 2 the set is {1,2,4,5,6}: mean 3.6, squared
    /// deviations 17.2 over 4, so s = √4.3 and z = −0.6/√4.3.
    ///
    /// Both differ from the full-sample answers (−2.5/√3.5 and −0.5/√3.5),
    /// so this test fails outright against a full-sample implementation.
    #[test]
    fn loo_standardize_matches_hand_computation() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let z = loo_standardize(&values, N_MIN);
        assert_eq!(z.len(), 6);
        let first = z[0].expect("six games is enough");
        assert!(
            (first - (-3.0 / 2.5f64.sqrt())).abs() < 1e-12,
            "z[0] = {first}"
        );
        let third = z[2].expect("six games is enough");
        assert!(
            (third - (-0.6 / 4.3f64.sqrt())).abs() < 1e-12,
            "z[2] = {third}"
        );
        // The full-sample answer for the same observation, for contrast.
        let full_sample = -2.5 / 3.5f64.sqrt();
        assert!(
            (first - full_sample).abs() > 0.5,
            "leave-one-out must not equal the full-sample z"
        );
    }

    /// The signature of leave-one-out: the z values do **not** sum to zero.
    /// A full-sample standardisation makes that sum exactly zero by
    /// construction, so this is the test that catches the substitution.
    #[test]
    fn loo_standardize_is_leave_one_out_not_full_sample() {
        let values = [0.0, 1.0, 2.0, 3.0, 4.0, 20.0];
        let z = loo_standardize(&values, N_MIN);
        let total: f64 = z.iter().map(|entry| entry.expect("all defined")).sum();
        assert!(
            total > 5.0,
            "leave-one-out z should not centre; summed to {total}"
        );
        // The outlier's own leave-one-out score: the other five are 0..4,
        // mean 2, s = √2.5, so z = 18/√2.5.
        let outlier = z[5].unwrap();
        assert!(
            (outlier - 18.0 / 2.5f64.sqrt()).abs() < 1e-12,
            "z[5] = {outlier}"
        );
    }

    /// Too few games, and a leave-out set with no spread, are both refusals.
    #[test]
    fn loo_standardize_refuses_short_series_and_flat_leave_outs() {
        let short = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(loo_standardize(&short, N_MIN).iter().all(Option::is_none));
        assert_eq!(loo_standardize(&[], N_MIN).len(), 0);
        // Five identical values plus one different: leaving out the odd one
        // leaves no spread at all, so that observation alone is undefined.
        let flat = [5.0, 5.0, 5.0, 5.0, 5.0, 7.0];
        let z = loo_standardize(&flat, N_MIN);
        assert!(z[5].is_none(), "a flat leave-out set has no scale");
        for entry in z.iter().take(5) {
            assert!(entry.is_some(), "the other five are perfectly well defined");
        }
        // Even above the floor, three values is the arithmetic minimum.
        assert!(loo_standardize(&[1.0, 2.0], 2).iter().all(Option::is_none));
    }

    // -----------------------------------------------------------------
    // §1.3 uniform scores
    // -----------------------------------------------------------------

    /// Whatever the values, the output is the rank grid, permuted.
    #[test]
    fn uniform_scores_are_a_permutation_of_the_grid() {
        let values = [3.5, -1.0, 3.5, 0.0, 9.9, 3.5, -1.0];
        let n = values.len();
        let scores = uniform_scores(&values, 20260902);
        let mut sorted = scores.clone();
        sorted.sort_by(f64::total_cmp);
        for (rank, &u) in sorted.iter().enumerate() {
            let expected = (rank as f64 + 0.5) / n as f64;
            assert!((u - expected).abs() < 1e-15, "{u} vs {expected}");
        }
        assert_eq!(uniform_scores(&[], 1).len(), 0);
    }

    /// Strictly increasing input keeps its order exactly.
    #[test]
    fn uniform_scores_follow_a_strict_order() {
        let values: Vec<f64> = (0..25).map(|i| i as f64 * 0.5 - 3.0).collect();
        let scores = uniform_scores(&values, 7);
        for (i, &u) in scores.iter().enumerate() {
            let expected = (i as f64 + 0.5) / 25.0;
            assert!((u - expected).abs() < 1e-15, "u[{i}] = {u} vs {expected}");
        }
    }

    /// A block of tied values takes exactly the ranks it is owed — no more,
    /// no fewer — and which member gets which rank inside the block is the
    /// only thing the seed can move.
    #[test]
    fn uniform_scores_break_ties_inside_the_block_only() {
        // One low value, eight ties, one high value.
        let mut values = vec![-1.0];
        values.extend(std::iter::repeat_n(0.0, 8));
        values.push(1.0);
        let n = values.len() as f64;
        let first = uniform_scores(&values, 1);
        let second = uniform_scores(&values, 2);
        assert!((first[0] - 0.05).abs() < 1e-15, "lowest keeps rank 1");
        assert!((first[9] - 0.95).abs() < 1e-15, "highest keeps rank n");
        for scores in [&first, &second] {
            let mut block: Vec<f64> = scores[1..9].to_vec();
            block.sort_by(f64::total_cmp);
            for (offset, &u) in block.iter().enumerate() {
                let expected = (offset as f64 + 1.5) / n;
                assert!((u - expected).abs() < 1e-15, "{u} vs {expected}");
            }
        }
        assert!(
            (first[0] - second[0]).abs() < 1e-15 && (first[9] - second[9]).abs() < 1e-15,
            "untied observations must not move with the seed"
        );
        assert!(
            first[1..9] != second[1..9],
            "a second seed must actually re-break the ties"
        );
        assert_eq!(
            first,
            uniform_scores(&values, 1),
            "same seed, same jitter, same answer"
        );
    }

    // -----------------------------------------------------------------
    // §1.4 the empirical survival copula
    // -----------------------------------------------------------------

    /// The plan's regions, at the boundaries where the inequalities matter.
    #[test]
    fn marginal_regions_follow_the_plan() {
        let over = Marginal::over(0.3);
        assert!(over.wins(0.71));
        assert!(!over.wins(0.7), "Over is strict: u > 1 − p");
        assert!(!over.wins(0.2));
        let under = Marginal::under(0.3);
        assert!(under.wins(0.3), "Under is weak: u <= p");
        assert!(under.wins(0.1));
        assert!(!under.wins(0.31));
        // Exact complements, so nothing is counted twice or lost.
        for step in 0..100 {
            let u = (step as f64 + 0.5) / 100.0;
            assert_ne!(Marginal::over(0.4).wins(u), Marginal::under(0.6).wins(u));
        }
    }

    /// The Fréchet upper and lower bounds, on samples built to attain them.
    #[test]
    fn empirical_joint_attains_the_frechet_bounds() {
        let n = 100;
        let grid: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
        let reversed: Vec<f64> = grid.iter().rev().copied().collect();
        for &p_a in &[0.3, 0.5, 0.7] {
            for &p_b in &[0.3, 0.5, 0.7] {
                let (a, b) = (Marginal::over(p_a), Marginal::over(p_b));
                let comonotone = empirical_joint(&grid, &grid, a, b).unwrap();
                assert!(
                    (comonotone - p_a.min(p_b)).abs() < 1e-12,
                    "comonotone at ({p_a},{p_b}) = {comonotone}"
                );
                let countermonotone = empirical_joint(&grid, &reversed, a, b).unwrap();
                let expected = (p_a + p_b - 1.0).max(0.0);
                assert!(
                    (countermonotone - expected).abs() <= 1.0 / n as f64,
                    "countermonotone at ({p_a},{p_b}) = {countermonotone}, expected {expected}"
                );
            }
        }
    }

    /// Independent inputs land on the product, within sampling error.
    #[test]
    fn empirical_joint_on_an_independent_dgp_is_the_product() {
        let mut rng = Rng::new(11);
        let n = 20_000;
        let x: Vec<f64> = (0..n).map(|_| rng.uniform()).collect();
        let y: Vec<f64> = (0..n).map(|_| rng.uniform()).collect();
        let u = uniform_scores(&x, 1);
        let v = uniform_scores(&y, 2);
        for (a, b) in reporting_grid() {
            let joint = empirical_joint(&u, &v, a, b).unwrap();
            let product = a.p * b.p;
            assert!(
                (joint - product).abs() < 0.015,
                "independent joint {joint} vs product {product}"
            );
        }
    }

    /// Fréchet consistency, stated in counts so it is an identity and not an
    /// approximation: the observations where A goes over split exactly into
    /// those where B is under `p_b` and those where B is over `1 − p_b`.
    #[test]
    fn empirical_joint_frechet_consistency_holds_in_counts() {
        let (x, y) = bivariate_normal(3_000, 0.55, 99);
        let u = uniform_scores(&x, 5);
        let v = uniform_scores(&y, 6);
        let n = u.len() as f64;
        for &p_a in &[0.2, 0.35, 0.5, 0.8] {
            for &p_b in &[0.15, 0.4, 0.65, 0.9] {
                let a = Marginal::over(p_a);
                let a_wins = u.iter().filter(|&&x| a.wins(x)).count() as f64 / n;
                let with_under = empirical_joint(&u, &v, a, Marginal::under(p_b)).unwrap();
                let with_over = empirical_joint(&u, &v, a, Marginal::over(1.0 - p_b)).unwrap();
                assert!(
                    (with_under + with_over - a_wins).abs() < 1e-12,
                    "{with_under} + {with_over} != {a_wins} at ({p_a},{p_b})"
                );
            }
        }
    }

    /// Widening either leg can only add observations, never remove them.
    #[test]
    fn empirical_joint_is_monotone_in_each_marginal() {
        let (x, y) = bivariate_normal(2_000, -0.3, 4242);
        let u = uniform_scores(&x, 8);
        let v = uniform_scores(&y, 9);
        let mut previous = 0.0;
        for step in 0..=20 {
            let p = step as f64 / 20.0;
            let joint = empirical_joint(&u, &v, Marginal::over(p), Marginal::over(0.45)).unwrap();
            assert!(joint >= previous - 1e-15, "{joint} < {previous} at p = {p}");
            previous = joint;
        }
        let mut previous = 0.0;
        for step in 0..=20 {
            let p = step as f64 / 20.0;
            let joint = empirical_joint(&u, &v, Marginal::under(0.6), Marginal::under(p)).unwrap();
            assert!(joint >= previous - 1e-15, "{joint} < {previous} at p = {p}");
            previous = joint;
        }
    }

    #[test]
    fn empirical_joint_refuses_unusable_input() {
        let u = [0.25, 0.75];
        assert!(empirical_joint(&[], &[], Marginal::over(0.5), Marginal::over(0.5)).is_none());
        assert!(empirical_joint(&u, &[0.5], Marginal::over(0.5), Marginal::over(0.5)).is_none());
    }

    // -----------------------------------------------------------------
    // §1.5 the Gaussian comparator, and the plan's phase-1 DGP gate
    // -----------------------------------------------------------------

    /// The four direction cases: first that each one is the orthant call its
    /// doc comment claims (one direction flip negates ρ, two restore it), then
    /// that they satisfy relations no copula can violate — B's two sides
    /// partition A's region, and flipping both legs lands on the
    /// complementary quadrant.
    ///
    /// The real validation of the direction handling is the DGP gate below,
    /// which puts all four combinations against the model-free estimator on a
    /// sample. This test pins the convention so a later "fix" to a sign has to
    /// argue with something.
    #[test]
    fn gaussian_joint_direction_cases_are_internally_consistent() {
        let rho = 0.45;
        for &p_a in &[0.2f64, 0.5, 0.75] {
            for &p_b in &[0.3f64, 0.5, 0.9] {
                let (h_a, h_b) = (-norm_ppf(p_a), -norm_ppf(p_b));
                let over_over = gaussian_joint(rho, Marginal::over(p_a), Marginal::over(p_b));
                let over_under = gaussian_joint(rho, Marginal::over(p_a), Marginal::under(p_b));
                let under_over = gaussian_joint(rho, Marginal::under(p_a), Marginal::over(p_b));
                let under_under = gaussian_joint(rho, Marginal::under(p_a), Marginal::under(p_b));
                assert!((over_over - bvn_upper(h_a, h_b, rho)).abs() < 1e-15);
                assert!((over_under - bvn_upper(h_a, h_b, -rho)).abs() < 1e-15);
                assert!((under_over - bvn_upper(h_a, h_b, -rho)).abs() < 1e-15);
                assert!((under_under - bvn_upper(h_a, h_b, rho)).abs() < 1e-15);

                // `under(p)` and `over(1 − p)` are complements, so each pair
                // of them splits the other leg's region exactly — the same
                // identity §1.4 states in counts for the empirical estimator.
                let split_over = over_under
                    + gaussian_joint(rho, Marginal::over(p_a), Marginal::over(1.0 - p_b));
                assert!(
                    (split_over - p_a).abs() < 1e-12,
                    "B's two sides must partition A's over region: {split_over} vs {p_a}"
                );
                let split_under = under_under
                    + gaussian_joint(rho, Marginal::under(p_a), Marginal::over(1.0 - p_b));
                assert!(
                    (split_under - p_a).abs() < 1e-12,
                    "and its under region: {split_under} vs {p_a}"
                );

                // Both legs flipped: P(¬A, ¬B) = 1 − p_a − p_b + P(A, B).
                let neither =
                    gaussian_joint(rho, Marginal::under(1.0 - p_a), Marginal::under(1.0 - p_b));
                assert!(
                    (neither - (1.0 - p_a - p_b + over_over)).abs() < 1e-12,
                    "complementary quadrant: {neither} vs {}",
                    1.0 - p_a - p_b + over_over
                );
            }
        }
    }

    /// **The plan's phase-1 gate (§7).** 20,000 bivariate normal pairs at
    /// ρ = 0.4, put through the real pipeline (uniform scores, then every
    /// estimator), must recover the truth: the Gaussian parameter within
    /// 0.02, Spearman and Kendall within 0.02 of their exact Gaussian-copula
    /// values, and the model-free empirical joint within 0.01 of the Gaussian
    /// one at all 36 points of the reporting grid.
    ///
    /// This is the test that says the four direction conventions, the region
    /// definitions and the orthant numerics agree with each other. Any sign
    /// error anywhere in §1.4–§1.5 shows up here as a gross failure.
    #[test]
    fn gaussian_copula_dgp_gate() {
        let rho = 0.4;
        let (x, y) = bivariate_normal(20_000, rho, 0xC0FF_EE12_3456_789A);
        let u = uniform_scores(&x, 1_000_003);
        let v = uniform_scores(&y, 1_000_033);

        let fitted = gaussian_rho(&u, &v).unwrap();
        assert!((fitted - rho).abs() < 0.02, "gaussian rho {fitted}");

        let expected_spearman = 6.0 / PI * (rho / 2.0).asin();
        let got_spearman = spearman(&u, &v).unwrap();
        assert!(
            (got_spearman - expected_spearman).abs() < 0.02,
            "spearman {got_spearman} vs {expected_spearman}"
        );

        let expected_tau = 2.0 / PI * rho.asin();
        let got_tau = kendall_tau(&u, &v).unwrap();
        assert!(
            (got_tau - expected_tau).abs() < 0.02,
            "kendall tau {got_tau} vs {expected_tau}"
        );

        let mut worst = 0.0f64;
        let mut worst_at = String::new();
        for (a, b) in reporting_grid() {
            let empirical = empirical_joint(&u, &v, a, b).unwrap();
            let gaussian = gaussian_joint(rho, a, b);
            let error = (empirical - gaussian).abs();
            if error > worst {
                worst = error;
                worst_at = format!("{a:?} x {b:?}");
            }
            assert!(
                error < 0.01,
                "grid point {a:?} x {b:?}: empirical {empirical}, gaussian {gaussian}"
            );
        }
        println!(
            "DGP gate: rho_hat {fitted:.6} (truth {rho}), spearman {got_spearman:.6} \
             (truth {expected_spearman:.6}), tau {got_tau:.6} (truth {expected_tau:.6}), \
             max |empirical - gaussian| over the 36 grid points {worst:.6} at {worst_at}"
        );
    }

    // -----------------------------------------------------------------
    // §1.6 rank summaries and φ
    // -----------------------------------------------------------------

    /// Known answers: perfect agreement, perfect disagreement, and a
    /// hand-counted τ-b with ties in both coordinates.
    #[test]
    fn rank_summaries_on_hand_counted_samples() {
        let ascending = [1.0, 2.0, 3.0, 4.0, 5.0];
        let descending = [5.0, 4.0, 3.0, 2.0, 1.0];
        assert!((kendall_tau(&ascending, &ascending).unwrap() - 1.0).abs() < 1e-12);
        assert!((kendall_tau(&ascending, &descending).unwrap() + 1.0).abs() < 1e-12);
        assert!((spearman(&ascending, &ascending).unwrap() - 1.0).abs() < 1e-12);
        assert!((spearman(&ascending, &descending).unwrap() + 1.0).abs() < 1e-12);

        // x = [1,1,2,3], y = [1,2,2,3]. Six pairs: one tied in x only, one
        // tied in y only, four concordant, none discordant. τ-b is
        // (6 − 1 − 1 + 0 − 0)/√(5·5) = 0.8.
        let x = [1.0, 1.0, 2.0, 3.0];
        let y = [1.0, 2.0, 2.0, 3.0];
        let tau = kendall_tau(&x, &y).unwrap();
        assert!((tau - 0.8).abs() < 1e-12, "tau-b = {tau}");

        assert!(kendall_tau(&[1.0], &[1.0]).is_none());
        assert!(kendall_tau(&ascending, &[1.0, 2.0]).is_none());
        assert!(kendall_tau(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]).is_none());
        assert!(spearman(&[0.5, 0.5], &[0.1, 0.9]).is_none());
    }

    /// φ from a hand-built 2×2 table: 3 both, 1 A-only, 1 B-only, 3 neither.
    /// p_a = p_b = 0.5, joint = 0.375, so φ = (0.375 − 0.25)/0.25 = 0.5 —
    /// the same table [`crate::correlation`]'s tests pin.
    #[test]
    fn phi_at_reproduces_a_hand_built_table() {
        let a = Marginal::over(0.5);
        let b = Marginal::over(0.5);
        assert!((phi_at(0.375, a, b).unwrap() - 0.5).abs() < 1e-12);
        assert!(phi_at(0.25, a, b).unwrap().abs() < 1e-12);
        // And it really is the sgp inversion, degenerate legs included.
        assert!(phi_at(0.0, Marginal::over(0.0), b).is_none());
        let uneven = phi_at(0.30, Marginal::over(0.6), Marginal::under(0.45)).unwrap();
        let expected = (0.30 - 0.27) / (0.6 * 0.4 * 0.45 * 0.55f64).sqrt();
        assert!((uneven - expected).abs() < 1e-12, "{uneven} vs {expected}");
    }

    // -----------------------------------------------------------------
    // §1.7 the cluster bootstrap
    // -----------------------------------------------------------------

    /// 40 clusters of 500 from the ρ = 0.4 DGP; the interval must cover the
    /// truth `L(0, 0; 0.4)` and be narrow enough to be worth printing.
    #[test]
    fn cluster_bootstrap_joint_covers_the_truth() {
        let (x, y) = bivariate_normal(20_000, 0.4, 5150);
        let clusters: Vec<u64> = (0..20_000u64).map(|i| i / 500).collect();
        let (a, b) = (Marginal::over(0.5), Marginal::over(0.5));
        let truth = gaussian_joint(0.4, a, b);
        let (low, high) = cluster_bootstrap_joint(&x, &y, &clusters, a, b, 200, 2026).unwrap();
        assert!(
            low < truth && truth < high,
            "CI [{low}, {high}] missed L(0,0;0.4) = {truth}"
        );
        assert!(high - low < 0.05, "CI width {}", high - low);
    }

    /// The seed selects every cluster draw and every jitter stream, so the
    /// interval reproduces exactly and moves when the seed does.
    #[test]
    fn cluster_bootstrap_joint_is_deterministic_under_a_seed() {
        let (x, y) = bivariate_normal(4_000, 0.35, 17);
        let clusters: Vec<u64> = (0..4_000u64).map(|i| i / 100).collect();
        let (a, b) = (Marginal::over(0.4), Marginal::under(0.6));
        let first = cluster_bootstrap_joint(&x, &y, &clusters, a, b, 150, 42);
        let second = cluster_bootstrap_joint(&x, &y, &clusters, a, b, 150, 42);
        assert_eq!(first, second, "same seed must resample identically");
        let other = cluster_bootstrap_joint(&x, &y, &clusters, a, b, 150, 43);
        assert_ne!(first, other, "a different seed must resample differently");
    }

    /// Below the floor there is no interval, and mismatched input is a caller
    /// bug rather than a measurement.
    #[test]
    fn cluster_bootstrap_joint_refuses_below_the_cluster_floor() {
        let (x, y) = bivariate_normal(600, 0.3, 3);
        let (a, b) = (Marginal::over(0.5), Marginal::over(0.5));
        let too_few: Vec<u64> = (0..600u64)
            .map(|i| i % (MIN_CLUSTERS_FOR_INTERVAL as u64 - 1))
            .collect();
        assert!(cluster_bootstrap_joint(&x, &y, &too_few, a, b, 50, 1).is_none());
        let enough: Vec<u64> = (0..600u64)
            .map(|i| i % MIN_CLUSTERS_FOR_INTERVAL as u64)
            .collect();
        assert!(cluster_bootstrap_joint(&x, &y, &enough, a, b, 50, 1).is_some());
        assert!(cluster_bootstrap_joint(&x, &y[..10], &enough, a, b, 50, 1).is_none());
        assert!(cluster_bootstrap_joint(&[], &[], &[], a, b, 50, 1).is_none());
    }

    /// **The test that proves the bootstrap respects clustering.** Two
    /// samples with the same number of rows, clusters and per-row dependence;
    /// in the second, every row of a cluster carries a common shift, so the
    /// cluster is much less than 100 independent observations. The interval
    /// must widen — a row-resampling bootstrap would return the same width
    /// twice and quietly claim precision it has not got.
    #[test]
    fn cluster_bootstrap_joint_widens_with_within_cluster_dependence() {
        let clusters: Vec<u64> = (0..4_000u64).map(|i| i / 100).collect();
        let (a, b) = (Marginal::over(0.5), Marginal::over(0.5));
        let (x, y) = bivariate_normal(4_000, 0.4, 8080);

        let independent = cluster_bootstrap_joint(&x, &y, &clusters, a, b, 300, 5).unwrap();

        // The same rows, plus one shared shift per cluster on both legs.
        let mut rng = Rng::new(90210);
        let shifts: Vec<f64> = (0..40).map(|_| box_muller(&mut rng).0 * 1.5).collect();
        let shifted_x: Vec<f64> = x
            .iter()
            .zip(clusters.iter())
            .map(|(&value, &cluster)| value + shifts[cluster as usize])
            .collect();
        let shifted_y: Vec<f64> = y
            .iter()
            .zip(clusters.iter())
            .map(|(&value, &cluster)| value + shifts[cluster as usize])
            .collect();
        let dependent =
            cluster_bootstrap_joint(&shifted_x, &shifted_y, &clusters, a, b, 300, 5).unwrap();

        let independent_width = independent.1 - independent.0;
        let dependent_width = dependent.1 - dependent.0;
        assert!(
            dependent_width > 2.0 * independent_width,
            "clustered width {dependent_width} should dwarf the iid width {independent_width}"
        );
    }

    // -----------------------------------------------------------------
    // the Dependence convenience type
    // -----------------------------------------------------------------

    #[test]
    fn dependence_agrees_with_the_free_functions() {
        let (x, y) = bivariate_normal(1_000, 0.5, 61);
        let pair = Dependence::from_standardised(&x, &y, 2026).unwrap();
        assert_eq!(pair.len(), 1_000);
        assert!(!pair.is_empty());
        let (a, b) = (Marginal::over(0.5), Marginal::under(0.4));
        assert_eq!(pair.joint(a, b), empirical_joint(&pair.u, &pair.v, a, b));
        assert_eq!(pair.spearman(), spearman(&pair.u, &pair.v));
        assert_eq!(pair.gaussian_rho(), gaussian_rho(&pair.u, &pair.v));
        assert_eq!(
            pair,
            Dependence::from_standardised(&x, &y, 2026).unwrap(),
            "the seed fixes the whole object"
        );
        assert!(Dependence::from_standardised(&x, &y[..5], 1).is_none());
        assert!(Dependence::from_standardised(&[], &[], 1).is_none());
    }
}
