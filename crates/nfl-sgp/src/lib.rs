//! NFL same-game-parlay correlation study.
//!
//! The thesis, carried over from the baseball project that preceded this one:
//! a sportsbook's same-game-parlay price embeds an allowance for how much the
//! two legs co-move (φ_book). The co-movement that actually exists can be
//! measured directly from paired game outcomes (φ_true). Where the book's
//! allowance sits below reality, the parlay is priced as if the legs co-occur
//! less often than they do — and the pre-registered statistic
//! `EV = P_true / q − 1` at **devigged consensus marginals** says whether that
//! gap survives the parlay's margin.
//!
//! What this crate deliberately does **not** contain: a forecasting model.
//! The baseball project spent most of its effort building forecasters that a
//! liquid market beat at every rung, and the finding that survived was the
//! model-free one. Market marginals are taken as given here; only the joint is
//! questioned.
//!
//! Module map:
//!
//! * [`sgp`] — price inversion. American odds → implied, devig, mechanical vs
//!   offered joint, φ from a joint, EV. Pure arithmetic, carried over verbatim.
//! * [`correlation`] — exact Bernoulli φ from paired outcomes, with a
//!   cluster-bootstrap interval and per-season era splits. Pure, verbatim.
//! * [`odds`] — The Odds API client for two-sided NFL marginals across books.
//! * [`nflverse`] — the free outcome data: weekly player and team stats plus
//!   the schedule with closing spread and total lines, 1999 onward.
//! * [`names`] — accent-folded name key so books spell one player one way.
//!
//! Still to be written, once the target families are chosen: the NFL atlas
//! (the dataset-aware layer that turns [`nflverse`] rows into paired legs and
//! calls [`correlation`]), and the exact-line scorer that joins the
//! hand-collected SGP sheet to the leg board. The baseball originals of both
//! are in `reference/baseball/` as the pattern to port.

pub mod atlas;
pub mod correlation;
pub mod depth;
pub mod error;
pub mod families;
pub mod games;
pub mod legs;
pub mod names;
pub mod nflverse;
pub mod odds;
pub mod scoring;
pub mod sgp;
pub mod stats;

pub use error::{Error, Result};
