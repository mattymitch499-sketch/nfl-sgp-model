//! Score the collected SGP sheet: `EV = P_true/q − 1` at devigged consensus
//! marginals, best book per game, family means with a game-cluster interval,
//! and the pre-registered verdict — `tasks/plan.md` §1.10, §3, §4, §5.
//!
//! Reads `data/processed/sgp_prices.csv` (hand-collected joints),
//! `outputs/leg_consensus.csv` (the leg board's cross-book marginals) and
//! `outputs/nfl_atlas.json` (the measured dependence). Writes
//! `outputs/sgp_scored.csv` and `outputs/sgp_verdict.json`.
//!
//! ## What it refuses to do
//!
//! * A family that failed the transport validation is **not priced**. Its rows
//!   are listed and its verdict is `not transportable`. The alternative —
//!   printing an EV anyway with a footnote — is how a number that nobody
//!   believes ends up quoted six weeks later.
//! * A leg with no consensus at two or more books and no second side on the
//!   sheet is **unscoreable**, and the row is dropped with its reason printed.
//!   A one-sided price cannot be devigged, and a vig-loaded marginal would
//!   quietly inflate every joint built from it.
//! * A row whose legs contradict the family it declares is a **collection
//!   error**, named and dropped rather than guessed at.
//!
//! Usage:
//! `cargo run --release -p nfl-sgp --example score_sgp -- [--sheet PATH] [--consensus PATH] [--atlas PATH] [--out-dir DIR] [--reps N] [--seed N]`

use nfl_sgp::atlas::Atlas;
use nfl_sgp::families;
use nfl_sgp::legs::parse_row;
use nfl_sgp::scoring::{
    ConsensusIndex, FAMILY_BOOTSTRAP_REPS, MarginalSource, Resolution, ScoredRow, best_by_game,
    describe_leg, resolve_marginal, score_row, summarise,
};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// One hand-collected row.
///
/// Prices are `Option` because a missing price is an **availability record**
/// ("the book would not combine these legs", "no ER line offered"), never a
/// price of zero — the discipline the baseball sheet was built on.
#[derive(Debug, Deserialize)]
struct SheetRow {
    date: String,
    #[serde(default)]
    #[allow(dead_code)]
    week: Option<u8>,
    book: String,
    #[serde(default)]
    #[allow(dead_code)]
    region: String,
    game: String,
    family: String,
    leg1_desc: String,
    leg1_price: Option<f64>,
    #[serde(default)]
    leg1_other_price: Option<f64>,
    leg2_desc: String,
    leg2_price: Option<f64>,
    #[serde(default)]
    leg2_other_price: Option<f64>,
    sgp_price: Option<f64>,
    /// Raw text: anything that does not parse as a non-negative integer is
    /// treated as unknown timing and the row is excluded, never assumed
    /// pregame.
    #[serde(default)]
    minutes_to_kickoff: String,
    /// The stratifying covariate, when it was recorded at collection.
    #[serde(default)]
    covariate: Option<f64>,
    #[serde(default)]
    #[allow(dead_code)]
    notes: String,
}

impl SheetRow {
    /// Pregame, by the same rule everywhere else in the project: unreadable
    /// counts as started.
    fn is_pregame(&self) -> bool {
        self.minutes_to_kickoff
            .trim()
            .parse::<i64>()
            .is_ok_and(|minutes| minutes >= 0)
    }
}

struct Args {
    sheet: PathBuf,
    consensus: PathBuf,
    atlas: PathBuf,
    out_dir: PathBuf,
    reps: usize,
    seed: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let args = parse_args(&root)?;

    let atlas = Atlas::load(&args.atlas).map_err(|error| {
        format!(
            "could not read the atlas at {} ({error}). Build it first: \
             cargo run --release -p nfl-sgp --example nfl_atlas",
            args.atlas.display()
        )
    })?;
    let failed = atlas.failed_families();
    let consensus = ConsensusIndex::read(&args.consensus)?;
    eprintln!(
        "atlas: {} strata, seasons {}-{}; consensus board: {} legs",
        atlas.strata.len(),
        atlas.seasons.0,
        atlas.seasons.1,
        consensus.len()
    );
    if !failed.is_empty() {
        eprintln!(
            "families that failed transport validation and will NOT be priced: {}",
            failed
                .iter()
                .map(|family| family.key())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    let mut reader = csv::Reader::from_path(&args.sheet)?;
    let sheet: Vec<SheetRow> = reader
        .deserialize()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("{}: {error}", args.sheet.display()))?;
    if sheet.is_empty() {
        eprintln!(
            "\n{} has no rows yet — collect a slate first.",
            args.sheet.display()
        );
        return Ok(());
    }

    let mut scored: Vec<ScoredRow> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    let mut in_play = 0usize;
    let mut incomplete = 0usize;

    for (index, row) in sheet.iter().enumerate() {
        let line = index + 2; // 1-based, plus the header
        if !row.is_pregame() {
            in_play += 1;
            continue;
        }
        let Some(sgp_price) = row.sgp_price else {
            // No joint price is an availability record, not an error.
            incomplete += 1;
            continue;
        };
        let parsed = match parse_row(&row.family, &row.leg1_desc, &row.leg2_desc) {
            Ok(parsed) => parsed,
            Err(why) => {
                skipped.push(format!("line {line}: {why}"));
                continue;
            }
        };
        if failed.contains(&parsed.family) {
            skipped.push(format!(
                "line {line}: {} failed transport validation and is not priced",
                parsed.family.key()
            ));
            continue;
        }
        let resolution_a = resolve_marginal(
            &parsed.leg_a,
            &row.date,
            &consensus,
            row.leg1_price,
            row.leg1_other_price,
        );
        let resolution_b = resolve_marginal(
            &parsed.leg_b,
            &row.date,
            &consensus,
            row.leg2_price,
            row.leg2_other_price,
        );
        let (marginal_a, source_a, marginal_b, source_b) = match (&resolution_a, &resolution_b) {
            (
                Resolution::Resolved {
                    marginal: a,
                    source: source_a,
                    ..
                },
                Resolution::Resolved {
                    marginal: b,
                    source: source_b,
                    ..
                },
            ) => (*a, source_a.clone(), *b, source_b.clone()),
            _ => {
                for (which, resolution, leg) in [
                    ("A", &resolution_a, &parsed.leg_a),
                    ("B", &resolution_b, &parsed.leg_b),
                ] {
                    if let Resolution::Unscoreable(why) = resolution {
                        skipped.push(format!(
                            "line {line}: leg {which} ({}) {why}",
                            describe_leg(leg)
                        ));
                    }
                }
                continue;
            }
        };
        let Some(stratum) = atlas.stratum_for(parsed.family, row.covariate) else {
            skipped.push(format!(
                "line {line}: the atlas has no stratum for {}",
                parsed.family.key()
            ));
            continue;
        };
        let Some(p_true) = stratum.joint(marginal_a, marginal_b) else {
            skipped.push(format!(
                "line {line}: the {} stratum could not price these marginals",
                stratum.key
            ));
            continue;
        };
        let interval =
            stratum.joint_interval(marginal_a, marginal_b, args.reps.min(400), args.seed);
        let Some(row_scored) = score_row(
            &row.date,
            &row.game,
            &row.book,
            parsed.family,
            marginal_a,
            marginal_b,
            source_a,
            source_b,
            &stratum.key,
            p_true,
            interval,
            sgp_price,
        ) else {
            skipped.push(format!(
                "line {line}: {sgp_price} is not a valid American price"
            ));
            continue;
        };
        scored.push(row_scored);
    }

    eprintln!(
        "\n{} sheet rows: {} scored, {} skipped, {} in-play, {} without an SGP price",
        sheet.len(),
        scored.len(),
        skipped.len(),
        in_play,
        incomplete
    );
    for reason in &skipped {
        eprintln!("  {reason}");
    }

    if scored.is_empty() {
        eprintln!("\nnothing scoreable yet.");
        return Ok(());
    }

    // Row-level output.
    std::fs::create_dir_all(&args.out_dir)?;
    let scored_path = args.out_dir.join("sgp_scored.csv");
    let mut writer = csv::Writer::from_path(&scored_path)?;
    writer.write_record([
        "slate",
        "game",
        "book",
        "family",
        "stratum",
        "p_a",
        "source_a",
        "p_b",
        "source_b",
        "p_true",
        "p_true_ci_low",
        "p_true_ci_high",
        "q",
        "ev",
        "ev_independent",
        "book_allowance",
    ])?;
    for row in &scored {
        writer.write_record([
            row.slate.clone(),
            row.game.clone(),
            row.book.clone(),
            row.family.key().to_string(),
            row.stratum.clone(),
            format!("{:.6}", row.p_a),
            source_label(&row.source_a),
            format!("{:.6}", row.p_b),
            source_label(&row.source_b),
            format!("{:.6}", row.p_true),
            row.p_true_interval
                .map_or_else(String::new, |(low, _)| format!("{low:.6}")),
            row.p_true_interval
                .map_or_else(String::new, |(_, high)| format!("{high:.6}")),
            format!("{:.6}", row.q),
            format!("{:.6}", row.ev),
            format!("{:.6}", row.ev_independent),
            format!("{:.4}", row.book_allowance),
        ])?;
    }
    writer.flush()?;

    // The decision statistic.
    eprintln!("\n== the pre-registered statistic: best book per game, mean across games ==");
    eprintln!(
        "  {:<24} {:>6} {:>9} {:>20} {:>10}",
        "family", "games", "mean EV", "95% CI", "verdict"
    );
    let mut summaries = Vec::new();
    for family in families::ALL {
        let best = best_by_game(&scored, family);
        if best.is_empty() && !failed.contains(&family) {
            continue;
        }
        let summary = summarise(
            family,
            &best,
            args.reps,
            args.seed,
            !failed.contains(&family),
        );
        eprintln!(
            "  {:<24} {:>6} {:>9} {:>20} {:>10}",
            family.key(),
            summary.games,
            if summary.mean_ev.is_nan() {
                "-".into()
            } else {
                format!("{:+.2}%", summary.mean_ev * 100.0)
            },
            summary.interval.map_or_else(
                || "-".into(),
                |(low, high)| format!("[{:+.2}%, {:+.2}%]", low * 100.0, high * 100.0)
            ),
            summary.verdict.label()
        );
        summaries.push((family, summary, best));
    }

    let verdict_path = args.out_dir.join("sgp_verdict.json");
    let json = serde_json::json!({
        "sheet": args.sheet.display().to_string(),
        "atlas_seasons": [atlas.seasons.0, atlas.seasons.1],
        "rows_scored": scored.len(),
        "rows_skipped": skipped.len(),
        "skipped_reasons": skipped,
        "failed_transport": failed.iter().map(|f| f.key()).collect::<Vec<_>>(),
        "families": summaries
            .iter()
            .map(|(family, summary, best)| {
                serde_json::json!({
                    "family": family.key(),
                    "games": summary.games,
                    "mean_ev": if summary.mean_ev.is_nan() { None } else { Some(summary.mean_ev) },
                    "ci_low": summary.interval.map(|(low, _)| low),
                    "ci_high": summary.interval.map(|(_, high)| high),
                    "verdict": summary.verdict.label(),
                    "per_game": best
                        .iter()
                        .map(|row| serde_json::json!({
                            "game": row.game,
                            "book": row.book,
                            "ev": row.ev,
                        }))
                        .collect::<Vec<_>>(),
                })
            })
            .collect::<Vec<_>>(),
    });
    std::fs::write(&verdict_path, serde_json::to_string_pretty(&json)?)?;
    eprintln!(
        "\nwrote:\n  {}\n  {}",
        scored_path.display(),
        verdict_path.display()
    );
    Ok(())
}

fn source_label(source: &MarginalSource) -> String {
    match source {
        MarginalSource::Consensus(books) => format!("consensus/{books}"),
        MarginalSource::Sheet => "sheet".into(),
    }
}

fn parse_args(root: &Path) -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = Args {
        sheet: root.join("data/processed/sgp_prices.csv"),
        consensus: root.join("outputs/leg_consensus.csv"),
        atlas: root.join("outputs/nfl_atlas.json"),
        out_dir: root.join("outputs"),
        reps: FAMILY_BOOTSTRAP_REPS,
        seed: 0x5C0_2ED,
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
            "--sheet" => args.sheet = PathBuf::from(value()?),
            "--consensus" => args.consensus = PathBuf::from(value()?),
            "--atlas" => args.atlas = PathBuf::from(value()?),
            "--out-dir" => args.out_dir = PathBuf::from(value()?),
            "--reps" => args.reps = value()?.parse()?,
            "--seed" => args.seed = value()?.parse()?,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    Ok(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(minutes: &str) -> SheetRow {
        SheetRow {
            date: "2026-09-13".into(),
            week: Some(1),
            book: "draftkings".into(),
            region: "us".into(),
            game: "Bills @ Chiefs".into(),
            family: "teamtotal_x_passtd".into(),
            leg1_desc: "Chiefs team total over 24.5".into(),
            leg1_price: Some(-110.0),
            leg1_other_price: Some(-110.0),
            leg2_desc: "Patrick Mahomes over 1.5 pass tds".into(),
            leg2_price: Some(-120.0),
            leg2_other_price: Some(100.0),
            sgp_price: Some(260.0),
            minutes_to_kickoff: minutes.into(),
            covariate: Some(0.65),
            notes: String::new(),
        }
    }

    /// The in-play rule, which cost the baseball study a retracted finding:
    /// unreadable timing counts as started, never as pregame.
    #[test]
    fn only_readable_pregame_timings_are_scored() {
        assert!(row("300").is_pregame());
        assert!(row("0").is_pregame());
        assert!(!row("-20").is_pregame());
        assert!(!row("").is_pregame());
        assert!(!row("about an hour").is_pregame());
    }

    #[test]
    fn source_labels_carry_the_book_count() {
        assert_eq!(source_label(&MarginalSource::Consensus(3)), "consensus/3");
        assert_eq!(source_label(&MarginalSource::Sheet), "sheet");
    }

    /// The sheet header must deserialize as written, including the optional
    /// columns — otherwise the first real collection fails at the door.
    #[test]
    fn the_committed_sheet_header_deserializes() {
        let text = "date,week,book,region,game,family,leg1_desc,leg1_price,leg1_other_price,\
                    leg2_desc,leg2_price,leg2_other_price,sgp_price,minutes_to_kickoff,covariate,notes\n\
                    2026-09-13,1,draftkings,us,Bills @ Chiefs,teamtotal_x_passtd,\
                    Chiefs team total over 24.5,-110,-110,Patrick Mahomes over 1.5 pass tds,\
                    -120,100,260,300,0.65,\n";
        let rows: Vec<SheetRow> = csv::Reader::from_reader(text.as_bytes())
            .deserialize()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].family, "teamtotal_x_passtd");
        assert_eq!(rows[0].sgp_price, Some(260.0));
        assert_eq!(rows[0].covariate, Some(0.65));
        assert!(rows[0].is_pregame());
        // And a row with the optional fields blank still reads.
        let sparse = "date,week,book,region,game,family,leg1_desc,leg1_price,leg1_other_price,\
                      leg2_desc,leg2_price,leg2_other_price,sgp_price,minutes_to_kickoff,covariate,notes\n\
                      2026-09-13,,dk,us,G,teamtotal_x_passtd,a over 1.5 pass tds,,,b over 1.5 pass tds,,,,300,,\n";
        let rows: Vec<SheetRow> = csv::Reader::from_reader(sparse.as_bytes())
            .deserialize()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(rows[0].sgp_price, None, "a missing joint is availability");
        assert_eq!(rows[0].covariate, None);
    }
}
