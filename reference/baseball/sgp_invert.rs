//! Inverts hand-collected same-game parlay prices into the book's correlation
//! allowance (φ_book), per book and family.
//!
//! Collection is manual: one 2-leg SGP per row in
//! `data/processed/sgp_prices.csv`. The `family` tag says what a row is *for*
//! — `indep` rows pair legs designed to be independent, so their implied φ is
//! a pure margin observation that pins down the hold; `k_x_teamk` is the
//! control family; `k_x_f5ml` and friends are where the thesis expects
//! φ_book < φ_true.
//!
//! The math lives in [`pns_core::sgp`]; this file is plumbing: parse, drop
//! in-play contamination, invert, group, write `outputs/sgp_inversion.json`.
//!
//! Usage:
//! `cargo run --release -p pns-core --example sgp_invert -- [--selftest]`

use pns_core::sgp;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Holds the book could plausibly be working to. A single SGP price confounds
/// margin with correlation, so φ_book is reported across all three — the
/// spread *is* the measurement uncertainty until the `indep` family nails the
/// hold down.
const HOLDS: [f64; 3] = [0.10, 0.15, 0.20];

/// One hand-collected SGP quote, one CSV row.
#[derive(Debug, Deserialize)]
struct PriceRow {
    date: String,
    book: String,
    region: String,
    game: String,
    family: String,
    leg1_desc: String,
    /// Empty when the book did not offer the leg or refused the combo — an
    /// availability record, not an inversion input. The row is counted and
    /// skipped, never read as a price of zero.
    leg1_price: Option<f64>,
    /// The other side of leg 1 (the under to leg 1's over): what a proper
    /// devig needs. `None` on rows collected before both sides were recorded.
    /// Read by the devig follow-up, not yet by the inversion.
    #[serde(default)]
    #[allow(dead_code)]
    leg1_other_price: Option<f64>,
    leg2_desc: String,
    leg2_price: Option<f64>,
    /// Same role as `leg1_other_price` for leg 2.
    #[serde(default)]
    #[allow(dead_code)]
    leg2_other_price: Option<f64>,
    sgp_price: Option<f64>,
    /// Kept as raw text because "unreadable" is a first-class state: anything
    /// that does not parse as an integer is treated as unknown timing and
    /// excluded, never assumed pregame — the same discipline as
    /// `book_probe::is_pregame`, where keeping one in-play quote costs a
    /// corrupted measurement and dropping a good one costs a single row.
    minutes_to_first_pitch: String,
    lineups_posted: String,
    notes: String,
}

/// One pregame row with its inversion attached.
#[derive(Debug, Serialize)]
struct InvertedRow {
    date: String,
    book: String,
    region: String,
    game: String,
    family: String,
    leg1_desc: String,
    leg1_price: f64,
    leg2_desc: String,
    leg2_price: f64,
    sgp_price: f64,
    minutes_to_first_pitch: i64,
    lineups_posted: bool,
    leg1_implied: f64,
    leg2_implied: f64,
    mechanical_joint: f64,
    offered_joint: f64,
    price_ratio: f64,
    phi_book_hold_10: f64,
    phi_book_hold_15: f64,
    phi_book_hold_20: f64,
    notes: String,
}

/// Per-(book, family) means. Books shade parlays by different amounts and
/// each family answers a different question, so pooling across either
/// dimension would average away the signal the sheet was built to measure.
#[derive(Debug, Serialize)]
struct GroupSummary {
    book: String,
    family: String,
    n: usize,
    mean_price_ratio: f64,
    mean_phi_book_hold_10: f64,
    mean_phi_book_hold_15: f64,
    mean_phi_book_hold_20: f64,
}

#[derive(Debug, Serialize)]
struct InversionReport {
    source: String,
    rows_collected: usize,
    /// Rows excluded as in-play or unknown-timing, *before* any math.
    dropped_in_play: usize,
    /// Pregame rows missing at least one price — availability records, kept
    /// out of the math but counted so collection gaps stay visible.
    rows_incomplete: usize,
    rows: Vec<InvertedRow>,
    summary: Vec<GroupSummary>,
}

/// None on empty or unparseable text — "we don't know when the game started"
/// is treated as in-play, per the field comment on [`PriceRow`].
fn parse_minutes(raw: &str) -> Option<i64> {
    raw.parse().ok()
}

/// Truthy spellings a human actually types into a collection sheet; anything
/// else is false. This flag is provenance, not an input to the math.
fn parse_posted(raw: &str) -> bool {
    matches!(
        raw.to_ascii_lowercase().as_str(),
        "yes" | "true" | "1" | "y"
    )
}

/// Inverts one row that has already cleared the timing screen and has all
/// three prices. Prices are the one thing that must be right, so an invalid
/// American price is a hard error naming the row — a typo fixed beats a row
/// silently dropped.
fn invert_row(
    row: &PriceRow,
    minutes: i64,
    leg1_price: f64,
    leg2_price: f64,
    sgp_price: f64,
) -> Result<InvertedRow, String> {
    let reject = |what: &str| format!("{} {} ({}): {what}", row.date, row.game, row.family);
    let p1 = sgp::implied_probability(leg1_price).ok_or_else(|| {
        reject(&format!(
            "leg1_price {leg1_price} is not a valid American price"
        ))
    })?;
    let p2 = sgp::implied_probability(leg2_price).ok_or_else(|| {
        reject(&format!(
            "leg2_price {leg2_price} is not a valid American price"
        ))
    })?;
    let offered = sgp::offered_joint(sgp_price).ok_or_else(|| {
        reject(&format!(
            "sgp_price {sgp_price} is not a valid American price"
        ))
    })?;
    let ratio =
        sgp::price_ratio(p1, p2, sgp_price).ok_or_else(|| reject("cannot form a price ratio"))?;
    let phi_at = |hold: f64| {
        sgp::implied_phi_book(p1, p2, sgp_price, hold)
            .ok_or_else(|| reject("cannot invert to a correlation allowance"))
    };
    Ok(InvertedRow {
        date: row.date.clone(),
        book: row.book.clone(),
        region: row.region.clone(),
        game: row.game.clone(),
        family: row.family.clone(),
        leg1_desc: row.leg1_desc.clone(),
        leg1_price,
        leg2_desc: row.leg2_desc.clone(),
        leg2_price,
        sgp_price,
        minutes_to_first_pitch: minutes,
        lineups_posted: parse_posted(&row.lineups_posted),
        leg1_implied: p1,
        leg2_implied: p2,
        mechanical_joint: sgp::mechanical_joint(p1, p2),
        offered_joint: offered,
        price_ratio: ratio,
        phi_book_hold_10: phi_at(HOLDS[0])?,
        phi_book_hold_15: phi_at(HOLDS[1])?,
        phi_book_hold_20: phi_at(HOLDS[2])?,
        notes: row.notes.clone(),
    })
}

fn summarize(rows: &[InvertedRow]) -> Vec<GroupSummary> {
    // BTreeMap so the JSON and the printed table have a stable order across runs.
    let mut groups: BTreeMap<(String, String), Vec<&InvertedRow>> = BTreeMap::new();
    for row in rows {
        groups
            .entry((row.book.clone(), row.family.clone()))
            .or_default()
            .push(row);
    }
    groups
        .into_iter()
        .map(|((book, family), members)| {
            let n = members.len() as f64;
            let mean = |take: fn(&InvertedRow) -> f64| {
                members.iter().map(|row| take(row)).sum::<f64>() / n
            };
            GroupSummary {
                book,
                family,
                n: members.len(),
                mean_price_ratio: mean(|row| row.price_ratio),
                mean_phi_book_hold_10: mean(|row| row.phi_book_hold_10),
                mean_phi_book_hold_15: mean(|row| row.phi_book_hold_15),
                mean_phi_book_hold_20: mean(|row| row.phi_book_hold_20),
            }
        })
        .collect()
}

/// Parse, screen, invert, summarize. Pure over the CSV text so the tests can
/// drive it with a constructed string.
fn invert(csv_text: &str, source: &str) -> Result<InversionReport, Box<dyn std::error::Error>> {
    let mut reader = csv::ReaderBuilder::new()
        // Hand-edited files pick up stray spaces; trim rather than punish them.
        .trim(csv::Trim::All)
        .from_reader(csv_text.as_bytes());
    let mut rows = Vec::new();
    let mut collected = 0;
    let mut dropped = 0;
    let mut incomplete = 0;
    for (index, record) in reader.deserialize::<PriceRow>().enumerate() {
        // +2: the header row, then 1-based counting, so the number matches the
        // line the collector sees in their editor.
        let record = record.map_err(|error| format!("CSV row {}: {error}", index + 2))?;
        collected += 1;
        match parse_minutes(&record.minutes_to_first_pitch) {
            Some(minutes) if minutes >= 0 => {
                match (record.leg1_price, record.leg2_price, record.sgp_price) {
                    (Some(leg1), Some(leg2), Some(sgp)) => {
                        rows.push(invert_row(&record, minutes, leg1, leg2, sgp)?)
                    }
                    // A missing price is availability evidence ("ER line not
                    // offered", "combo refused"), not an inversion input.
                    _ => incomplete += 1,
                }
            }
            _ => dropped += 1,
        }
    }
    let summary = summarize(&rows);
    Ok(InversionReport {
        source: source.to_string(),
        rows_collected: collected,
        dropped_in_play: dropped,
        rows_incomplete: incomplete,
        rows,
        summary,
    })
}

fn print_summary(report: &InversionReport) {
    println!(
        "{:<12} {:<12} {:>3} {:>8} {:>9} {:>9} {:>9}",
        "book", "family", "n", "mean_R", "phi@0.10", "phi@0.15", "phi@0.20"
    );
    for group in &report.summary {
        println!(
            "{:<12} {:<12} {:>3} {:>8.3} {:>9.3} {:>9.3} {:>9.3}",
            group.book,
            group.family,
            group.n,
            group.mean_price_ratio,
            group.mean_phi_book_hold_10,
            group.mean_phi_book_hold_15,
            group.mean_phi_book_hold_20
        );
    }
}

/// A tiny synthetic sample standing in for the real sheet: two rows in one
/// (book, family) group so the mean is exercised, an `indep` row, one in-play
/// row and one unknown-timing row that must both be dropped.
const SELFTEST_CSV: &str = "\
date,book,region,game,family,leg1_desc,leg1_price,leg2_desc,leg2_price,sgp_price,minutes_to_first_pitch,lineups_posted,notes
2026-08-10,draftkings,national,NYY@BOS,k_x_f5ml,Home SP o7.5 K,-130,BOS F5 ML,-119,244,180,yes,favoured legs
2026-08-10,draftkings,national,LAD@SF,k_x_f5ml,Away SP o6.5 K,108,SF F5 ML,108,244,200,no,balanced legs
2026-08-11,fanduel,national,CHC@STL,indep,Home SP o5.5 K,-110,Game total o8.5,-110,260,150,yes,margin identification
2026-08-11,fanduel,national,NYM@ATL,k_x_teamk,Home SP o8 K,-120,ATL team total o4.5,-105,300,-25,yes,in-play must drop
2026-08-11,fanduel,national,HOU@TEX,k_x_teamk,Away SP o7 K,-115,TEX ML,120,320,,no,unknown timing must drop
";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let selftest = std::env::args().any(|argument| argument == "--selftest");

    let (csv_text, source, out_path) = if selftest {
        (
            SELFTEST_CSV.to_string(),
            "--selftest synthetic sample".to_string(),
            // A separate path: synthetic numbers must never sit where later
            // analysis expects collected ones.
            root.join("outputs/sgp_inversion_selftest.json"),
        )
    } else {
        let csv_path = root.join("data/processed/sgp_prices.csv");
        if !csv_path.exists() {
            println!(
                "no prices collected yet — expected {}. Collect rows there, or run with --selftest to exercise the pipeline on a synthetic sample.",
                csv_path.display()
            );
            return Ok(());
        }
        (
            std::fs::read_to_string(&csv_path)?,
            csv_path.display().to_string(),
            root.join("outputs/sgp_inversion.json"),
        )
    };

    let report = invert(&csv_text, &source)?;
    println!(
        "excluded {} of {} collected rows as in-play or unknown-timing; a started game prices a different question",
        report.dropped_in_play, report.rows_collected
    );
    if report.rows_incomplete > 0 {
        println!(
            "noted {} rows with a missing price (availability records, not inverted)",
            report.rows_incomplete
        );
    }
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out_path, serde_json::to_string_pretty(&report)?)?;
    println!("wrote {} ({} rows)", out_path.display(), report.rows.len());
    print_summary(&report);
    Ok(())
}

#[cfg(test)]
mod tests {
    /// Five collected rows: three keepers across two (book, family) groups,
    /// one in-play row and one row with no timing — both must be excluded,
    /// because a started game prices a different question and an unknown
    /// start time must never be assumed pregame.
    const TEST_CSV: &str = "\
date,book,region,game,family,leg1_desc,leg1_price,leg2_desc,leg2_price,sgp_price,minutes_to_first_pitch,lineups_posted,notes
2026-08-10,draftkings,national,NYY@BOS,k_x_f5ml,Home SP o7.5 K,-130,BOS F5 ML,-119,244,180,yes,favoured legs
2026-08-10,draftkings,national,LAD@SF,k_x_f5ml,Away SP o6.5 K,108,SF F5 ML,108,244,200,no,balanced legs
2026-08-11,fanduel,national,CHC@STL,indep,Home SP o5.5 K,-110,Game total o8.5,-110,260,150,yes,margin identification
2026-08-11,fanduel,national,NYM@ATL,k_x_teamk,Home SP o8 K,-120,ATL team total o4.5,-105,300,-25,yes,in-play must drop
2026-08-11,fanduel,national,HOU@TEX,k_x_teamk,Away SP o7 K,-115,TEX ML,120,320,,no,unknown timing must drop
2026-08-12,draftkings,national,DET@PIT,k_x_er,Away SP o5.5 K,116,,,,468,no,leg not offered — availability record
";

    #[test]
    fn in_play_and_unknown_timing_rows_are_excluded_and_counted() {
        let report = super::invert(TEST_CSV, "test").unwrap();
        assert_eq!(report.rows_collected, 6);
        assert_eq!(report.dropped_in_play, 2);
        assert_eq!(report.rows_incomplete, 1);
        assert_eq!(report.rows.len(), 3);
        assert!(
            report
                .rows
                .iter()
                .all(|row| row.minutes_to_first_pitch >= 0)
        );
    }

    /// First keeper is the favoured-legs worked example pinned in the library
    /// tests: -130/-119 with the SGP at +244.
    #[test]
    fn rows_invert_to_hand_checked_numbers() {
        let report = super::invert(TEST_CSV, "test").unwrap();
        let row = &report.rows[0];
        assert_eq!(row.family, "k_x_f5ml");
        assert!(
            (row.leg1_implied - 0.5652).abs() < 1e-3,
            "got {}",
            row.leg1_implied
        );
        assert!(
            (row.mechanical_joint - 0.3071).abs() < 1e-3,
            "got {}",
            row.mechanical_joint
        );
        assert!(
            (row.offered_joint - 0.2907).abs() < 1e-3,
            "got {}",
            row.offered_joint
        );
        assert!(
            (row.price_ratio - 0.9465).abs() < 1e-3,
            "got {}",
            row.price_ratio
        );
        // More assumed margin must mean less correlation allowance.
        assert!(row.phi_book_hold_10 > row.phi_book_hold_15);
        assert!(row.phi_book_hold_15 > row.phi_book_hold_20);
    }

    /// The draftkings k_x_f5ml group holds the favoured-legs row (R ≈ 0.9465)
    /// and the balanced-legs row (R ≈ 1.2577), so its mean R ≈ 1.1021. The
    /// fanduel indep row stands alone at R ≈ 1.0125.
    #[test]
    fn summary_groups_by_book_and_family_with_hand_checked_means() {
        let report = super::invert(TEST_CSV, "test").unwrap();
        assert_eq!(report.summary.len(), 2);
        let dk = &report.summary[0];
        assert_eq!(
            (dk.book.as_str(), dk.family.as_str()),
            ("draftkings", "k_x_f5ml")
        );
        assert_eq!(dk.n, 2);
        assert!(
            (dk.mean_price_ratio - 1.1021).abs() < 1e-3,
            "got {}",
            dk.mean_price_ratio
        );
        let fd = &report.summary[1];
        assert_eq!((fd.book.as_str(), fd.family.as_str()), ("fanduel", "indep"));
        assert_eq!(fd.n, 1);
        assert!(
            (fd.mean_price_ratio - 1.0125).abs() < 1e-3,
            "got {}",
            fd.mean_price_ratio
        );
    }

    /// The report must actually survive serialization — it is the artifact
    /// later analysis reads.
    #[test]
    fn report_serializes_to_json() {
        let report = super::invert(TEST_CSV, "test").unwrap();
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"dropped_in_play\":2"));
        assert!(json.contains("\"mechanical_joint\""));
        assert!(json.contains("\"mean_phi_book_hold_15\""));
    }
}
