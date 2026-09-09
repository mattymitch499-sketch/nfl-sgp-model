//! The empirical correlation atlas for the same-game-parlay thesis.
//!
//! [`pns_core::sgp`] measures the correlation a book's price *allows for*;
//! this atlas measures the correlation that actually *exists* between a
//! starter's strikeout over and each candidate partner leg, straight from the
//! project's own datasets. Every family pairs A = "my starter's strikeouts ≥
//! rung" against a B leg, rung by rung, and reports the exact Bernoulli φ
//! with per-season era splits; headline rows (fixed rung k ≥ 6) also carry a
//! pitcher-year cluster-bootstrap interval.
//!
//! The dataset-aware machinery — observation types, the [`atlas::measure`]
//! engine, the both-starter-game and batter-start joins — lives in
//! [`pns_core::atlas`]; this example owns only the rung grid, the family
//! definitions, and the output schema.
//!
//! Usage:
//! `cargo run --release -p pns-core --example correlation_atlas -- [--dataset PATH] [--batters PATH] [--out PATH]`

use chrono::Datelike;
use pns_core::atlas::{
    self, BatterObs, Obs, ViewObs, batter_start_pairs, outs_from_baseball_ip, personal_lines,
    side_totals, starter_views,
};
use pns_core::data::batters::read_batter_lines;
use pns_core::data::dataset::read_csv;
use pns_core::data::types::PitcherStartLog;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Fixed K-over rungs — the ladder books actually hang.
const FIXED_RUNGS: [u16; 3] = [5, 6, 7];
/// A deep start: 6+ full innings. Baseball notation — 5.1 IP is 16 outs.
const DEEP_START_OUTS: u32 = 18;
/// Bootstrap replicates for the headline rows. A draw aggregates whole
/// pitcher-year clusters, so 300 is cheap and clears the ≥300 bar.
const BOOTSTRAP_DRAWS: usize = 300;
/// Fixed bootstrap seed. Arbitrary but pinned: identical inputs must
/// reproduce the atlas bit-for-bit.
const BOOTSTRAP_SEED: u64 = 0xD15E_A5E0_0000_0001;

/// The A-leg rungs: the three fixed thresholds plus the pitcher's own market
/// line (his season's mean K/start, rounded, computed within the season).
#[derive(Clone, Copy)]
enum ARung {
    Fixed(u16),
    Line,
}

impl ARung {
    fn all() -> [Self; 4] {
        let [a, b, c] = FIXED_RUNGS;
        [Self::Fixed(a), Self::Fixed(b), Self::Fixed(c), Self::Line]
    }

    fn label(self) -> String {
        match self {
            Self::Fixed(k) => format!("k>={k}"),
            Self::Line => "k>=line".to_string(),
        }
    }

    /// Headline rows get the bootstrap interval; the rest get point values
    /// and era splits only, so the atlas runs in seconds.
    fn is_headline(self) -> bool {
        matches!(self, Self::Fixed(6))
    }

    fn over(self, strikeouts: u16, personal_line: u16) -> bool {
        let threshold = match self {
            Self::Fixed(k) => k,
            Self::Line => personal_line,
        };
        strikeouts >= threshold
    }
}

/// One line of the atlas: a family at one A-rung against one B-rung.
struct AtlasRow {
    family: &'static str,
    rung_a: String,
    rung_b: String,
    /// Always "both" — the positive quadrant. φ(¬A, ¬B) = φ(A, B) exactly and
    /// the mixed quadrants negate it (pinned by a correlation.rs unit test),
    /// so one quadrant per family×rung loses nothing.
    quadrant: &'static str,
    n: usize,
    p_a: f64,
    p_b: f64,
    joint: f64,
    phi: Option<f64>,
    ci_low: Option<f64>,
    ci_high: Option<f64>,
    /// Per-season φ, the regime check: a correlation that exists only in one
    /// season is a story about that season, not about how legs co-move.
    eras: BTreeMap<i32, Option<f64>>,
}

/// Thin adapter over [`atlas::measure`]: same engine, atlas row shape. The
/// bootstrap runs on headline rows only, so the atlas stays a seconds-scale
/// run; the rest get point values and era splits.
fn measure<T: Obs>(
    rows: &mut Vec<AtlasRow>,
    family: &'static str,
    rung_a: String,
    rung_b: String,
    headline: bool,
    obs: &[T],
    legs: impl Fn(&T) -> Option<(bool, bool)>,
) {
    let bootstrap = headline.then_some((BOOTSTRAP_DRAWS, BOOTSTRAP_SEED));
    let Some(m) = atlas::measure(obs, legs, bootstrap) else {
        eprintln!("{family} {rung_a} x {rung_b}: degenerate or empty, skipped");
        return;
    };
    rows.push(AtlasRow {
        family,
        rung_a,
        rung_b,
        quadrant: "both",
        n: m.n,
        p_a: m.p_a,
        p_b: m.p_b,
        joint: m.joint,
        phi: Some(m.phi),
        ci_low: m.interval.map(|(low, _)| low),
        ci_high: m.interval.map(|(_, high)| high),
        eras: m.eras,
    });
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let args = parse_args(&root)?;
    let starts = read_csv(&args.dataset)?;
    let batter_rows = read_batter_lines(&args.batters)?;
    eprintln!(
        "{} starts, {} batter lines",
        starts.len(),
        batter_rows.len()
    );

    let line_triples: Vec<(u32, i32, u16)> = starts
        .iter()
        .map(|s| (s.pitcher_id, s.game_date.year(), s.strikeouts))
        .collect();
    let lines = personal_lines(&line_triples);
    let line_of = |s: &PitcherStartLog| lines[&(s.pitcher_id, s.game_date.year())];

    let mut rows: Vec<AtlasRow> = Vec::new();

    // -- Pitcher-level families: the starter against his own line. ----------
    // A = his strikeouts over the rung; B = a quiet day on the scoreboard.
    // These are the legs of the most commonly offered pitcher SGPs.
    for rung in ARung::all() {
        for r in 1..=3u16 {
            measure(
                &mut rows,
                "starter_er",
                rung.label(),
                format!("er<={r}"),
                rung.is_headline(),
                &starts,
                |s: &PitcherStartLog| {
                    Some((rung.over(s.strikeouts, line_of(s)), s.earned_runs <= r))
                },
            );
        }
        for h in 5..=7u16 {
            measure(
                &mut rows,
                "starter_hits_allowed",
                rung.label(),
                format!("hits<={h}"),
                rung.is_headline(),
                &starts,
                |s: &PitcherStartLog| Some((rung.over(s.strikeouts, line_of(s)), s.hits <= h)),
            );
        }
        measure(
            &mut rows,
            "deep_start",
            rung.label(),
            format!("outs>={DEEP_START_OUTS}"),
            rung.is_headline(),
            &starts,
            |s: &PitcherStartLog| {
                Some((
                    rung.over(s.strikeouts, line_of(s)),
                    outs_from_baseball_ip(s.innings_pitched) >= DEEP_START_OUTS,
                ))
            },
        );
    }

    // -- Team/game-level families: starter-views of both-starter games. -----
    // Each game with both starters in the file contributes two observations,
    // one from each starter's perspective; single-starter games have nothing
    // to cross-join against and are skipped.
    let totals = side_totals(&batter_rows);
    let (views, both_starter_games) = starter_views(&starts, &totals, &lines);
    eprintln!(
        "{both_starter_games} both-starter games -> {} starter views",
        views.len()
    );

    for rung in ARung::all() {
        // A tie is not a loss: deadlocked views drop out of the 2×2 entirely.
        measure(
            &mut rows,
            "outpitch",
            rung.label(),
            "my_er<their_er".to_string(),
            rung.is_headline(),
            &views,
            |o: &ViewObs| {
                (o.earned_runs != o.their_earned_runs).then(|| {
                    (
                        rung.over(o.strikeouts, o.personal_line),
                        o.earned_runs < o.their_earned_runs,
                    )
                })
            },
        );
        for t in 4..=6u16 {
            measure(
                &mut rows,
                "combined_starter_er",
                rung.label(),
                format!("combined_er<={t}"),
                rung.is_headline(),
                &views,
                |o: &ViewObs| {
                    Some((
                        rung.over(o.strikeouts, o.personal_line),
                        o.earned_runs + o.their_earned_runs <= t,
                    ))
                },
            );
        }
        for t in 7..=9u32 {
            measure(
                &mut rows,
                "team_k_thrown",
                rung.label(),
                format!("team_k>={t}"),
                rung.is_headline(),
                &views,
                |o: &ViewObs| {
                    Some((
                        rung.over(o.strikeouts, o.personal_line),
                        o.opposing_team_ks? >= t,
                    ))
                },
            );
        }
        for h in 7..=9u32 {
            measure(
                &mut rows,
                "team_hits_allowed",
                rung.label(),
                format!("team_hits<={h}"),
                rung.is_headline(),
                &views,
                |o: &ViewObs| {
                    Some((
                        rung.over(o.strikeouts, o.personal_line),
                        o.opposing_team_hits? <= h,
                    ))
                },
            );
        }
    }

    // -- Batter-level families: every batter's line joined to the opposing --
    // starter he faced (same game, different team). The betting question is
    // whether one hitter's night tells you anything about the starter's K
    // total beyond the starter himself.
    let batter_obs = batter_start_pairs(&starts, &batter_rows, &lines);
    eprintln!("{} batter-start pairs", batter_obs.len());

    for rung in ARung::all() {
        for (family, min_pa) in [("batter_k", 0u16), ("batter_k_pa3", 3)] {
            measure(
                &mut rows,
                family,
                rung.label(),
                "batter_k>=1".to_string(),
                rung.is_headline(),
                &batter_obs,
                |o: &BatterObs| {
                    (o.plate_appearances >= min_pa).then(|| {
                        (
                            rung.over(o.starter_strikeouts, o.starter_personal_line),
                            o.batter_strikeouts >= 1,
                        )
                    })
                },
            );
        }
        for (family, min_pa) in [("batter_hits", 0u16), ("batter_hits_pa3", 3)] {
            measure(
                &mut rows,
                family,
                rung.label(),
                "batter_hits<=1".to_string(),
                rung.is_headline(),
                &batter_obs,
                |o: &BatterObs| {
                    (o.plate_appearances >= min_pa).then(|| {
                        (
                            rung.over(o.starter_strikeouts, o.starter_personal_line),
                            o.batter_hits <= 1,
                        )
                    })
                },
            );
        }
    }

    // -- Controls: two relations with known answers gate the whole atlas. ---
    // (a) The duel: the two starters' K lines share a game but no mechanism,
    //     so my K ≥ 6 × their K ≥ 6 must come out ≈ 0 (prior: −0.013). A
    //     meaningful |φ| here means the game join is attributing lines to the
    //     wrong pitcher.
    measure(
        &mut rows,
        "duel_control",
        "k>=6".to_string(),
        "their_k>=6".to_string(),
        true,
        &views,
        |o: &ViewObs| Some((o.strikeouts >= 6, o.their_strikeouts >= 6)),
    );

    println!(
        "{:<22} {:<8} {:<16} {:>7} {:>7} {:>7} {:>7} {:>8} {:>19} {:>7} {:>7} {:>7} {:>7}",
        "family",
        "rung_a",
        "rung_b",
        "n",
        "p_a",
        "p_b",
        "joint",
        "phi",
        "95% CI",
        "2023",
        "2024",
        "2025",
        "2026"
    );
    for row in &rows {
        let era = |year: i32| row.eras.get(&year).copied().flatten();
        let ci = match (row.ci_low, row.ci_high) {
            (Some(low), Some(high)) => format!("[{low:+.3}, {high:+.3}]"),
            _ => "—".to_string(),
        };
        let era_str =
            |value: Option<f64>| value.map_or_else(|| "—".to_string(), |v| format!("{v:+.3}"));
        println!(
            "{:<22} {:<8} {:<16} {:>7} {:>7.3} {:>7.3} {:>7.3} {:>+8.3} {:>19} {:>7} {:>7} {:>7} {:>7}",
            row.family,
            row.rung_a,
            row.rung_b,
            row.n,
            row.p_a,
            row.p_b,
            row.joint,
            row.phi.unwrap_or(f64::NAN),
            ci,
            era_str(era(2023)),
            era_str(era(2024)),
            era_str(era(2025)),
            era_str(era(2026)),
        );
    }

    // (b) The part-whole: my strikeouts are *part of* the team K total, so
    //     my K ≥ 6 × team Ks ≥ t must come out strongly positive (φ ≳ 0.3).
    //     If either control fails, the joins are wrong — stop rather than
    //     emit numbers nobody should trust.
    let duel = rows
        .iter()
        .find(|row| row.family == "duel_control")
        .and_then(|row| row.phi);
    let part_whole: Vec<f64> = rows
        .iter()
        .filter(|row| row.family == "team_k_thrown" && row.rung_a == "k>=6")
        .filter_map(|row| row.phi)
        .collect();
    let duel_ok = duel.is_some_and(|value| value.abs() <= 0.05);
    let part_whole_ok = part_whole.len() == 3 && part_whole.iter().all(|value| *value >= 0.30);
    println!("\ncontrols:");
    println!(
        "  duel my K>=6 x their K>=6: phi {:+.3} — expect ≈ 0 (prior −0.013) [{}]",
        duel.unwrap_or(f64::NAN),
        if duel_ok { "PASS" } else { "FAIL" }
    );
    println!(
        "  part-whole my K>=6 x team K>=t: phi [{}] — expect all >= 0.30 [{}]",
        part_whole
            .iter()
            .map(|v| format!("{v:+.3}"))
            .collect::<Vec<_>>()
            .join(", "),
        if part_whole_ok { "PASS" } else { "FAIL" }
    );
    if !(duel_ok && part_whole_ok) {
        eprintln!("\nCONTROL FAILURE: the joins are wrong — no atlas written.");
        std::process::exit(1);
    }

    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = csv::Writer::from_path(&args.out)?;
    writer.write_record([
        "family", "rung_a", "rung_b", "quadrant", "n", "p_a", "p_b", "joint", "phi", "ci_low",
        "ci_high", "phi_2023", "phi_2024", "phi_2025", "phi_2026",
    ])?;
    let number = |value: Option<f64>| value.map_or_else(String::new, |v| format!("{v:.6}"));
    for row in &rows {
        let era = |year: i32| row.eras.get(&year).copied().flatten();
        writer.write_record([
            row.family.to_string(),
            row.rung_a.clone(),
            row.rung_b.clone(),
            row.quadrant.to_string(),
            row.n.to_string(),
            format!("{:.6}", row.p_a),
            format!("{:.6}", row.p_b),
            format!("{:.6}", row.joint),
            number(row.phi),
            number(row.ci_low),
            number(row.ci_high),
            number(era(2023)),
            number(era(2024)),
            number(era(2025)),
            number(era(2026)),
        ])?;
    }
    writer.flush()?;
    println!("\nwrote {} rows to {}", rows.len(), args.out.display());
    Ok(())
}

struct Args {
    dataset: PathBuf,
    batters: PathBuf,
    out: PathBuf,
}

fn parse_args(root: &Path) -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = Args {
        dataset: root.join("data/processed/training_starts.csv"),
        batters: root.join("data/processed/batter_lines.csv"),
        out: root.join("outputs/correlation_atlas.csv"),
    };
    let mut raw = std::env::args().skip(1);
    while let Some(argument) = raw.next() {
        let (name, inline) = match argument.split_once('=') {
            Some((name, value)) => (name.to_string(), Some(value.to_string())),
            None => (argument, None),
        };
        let mut value = || -> Result<String, Box<dyn std::error::Error>> {
            inline
                .clone()
                .or_else(|| raw.next())
                .ok_or_else(|| format!("{name} needs a value").into())
        };
        match name.as_str() {
            "--dataset" => args.dataset = PathBuf::from(value()?),
            "--batters" => args.batters = PathBuf::from(value()?),
            "--out" => args.out = PathBuf::from(value()?),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    Ok(args)
}
