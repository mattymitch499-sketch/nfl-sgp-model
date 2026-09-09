//! What the six families of `tasks/plan.md` §2 actually yield on the real
//! data — the Phase 2 gate's "real-data counts printed".
//!
//! This measures nothing. It prints how many observations each family has,
//! how many clusters they fall into, how often the pre-game subject rules had
//! to fall back or drop a row, how many rows reach only the `all` stratum for
//! want of a covariate, and a five-number summary of each column. Those are
//! the numbers that say whether a family is worth an atlas at all: a family
//! with 400 observations in 30 clusters cannot support the §1.7 interval, and
//! a covariate that is missing on a third of its rows is not a stratification.
//!
//! It also prints the most common Wr1 and Rb1 names of 2024. That is the
//! cheapest possible check that the pre-game resolution of §2.1 is picking
//! real starters rather than, say, the alphabetically-first receiver on the
//! roster — a bug that no unit test on a synthetic fixture would catch.
//!
//! Usage:
//! `cargo run --release -p nfl-sgp --example family_counts [-- --seasons 2010-2025] [--dir data/raw/nflverse] [--out outputs/family_counts.csv]`

use nfl_sgp::families::{self, ClusterScheme, CovariateScheme, Observation};
use nfl_sgp::games::{GameTable, SubjectKey};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// The plan's §2.2 scope: weekly files 2010–2025. `games.csv` is read whole
/// whatever this says, so `spread_x_total` still reaches 1999.
const DEFAULT_FIRST: u16 = 2010;
const DEFAULT_LAST: u16 = 2025;

struct Args {
    first: u16,
    last: u16,
    dir: PathBuf,
    out: PathBuf,
}

/// Min, lower quartile, median, upper quartile, max — with linear
/// interpolation between order statistics, the same convention the bootstrap
/// percentiles use. `None` on an empty slice.
fn five_number(values: &[f64]) -> Option<[f64; 5]> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    Some([
        sorted[0],
        percentile(&sorted, 0.25),
        percentile(&sorted, 0.50),
        percentile(&sorted, 0.75),
        sorted[sorted.len() - 1],
    ])
}

/// Linear-interpolation percentile of an already-sorted slice.
fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let position = quantile * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let weight = position - lower as f64;
    sorted[lower] * (1.0 - weight) + sorted[upper] * weight
}

fn summary_line(label: &str, values: &[f64]) -> String {
    match five_number(values) {
        Some([min, q1, med, q3, max]) => format!(
            "    {label:<5} n {:<6} min {min:>9.2}  q1 {q1:>9.2}  med {med:>9.2}  q3 {q3:>9.2}  max {max:>9.2}",
            values.len()
        ),
        None => format!("    {label:<5} (no values)"),
    }
}

/// Every distinct cluster id a family's rows carry, primary and secondary.
fn clusters(rows: &[Observation]) -> BTreeSet<u64> {
    let mut ids = BTreeSet::new();
    for row in rows {
        ids.insert(row.cluster_primary);
        ids.extend(row.cluster_secondary);
    }
    ids
}

/// The `n` most common display names in a counted map, ties broken by name so
/// the output is stable.
fn most_common(counts: &BTreeMap<String, usize>, n: usize) -> Vec<(String, usize)> {
    let mut ranked: Vec<(String, usize)> = counts
        .iter()
        .map(|(name, count)| (name.clone(), *count))
        .collect();
    ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    ranked.truncate(n);
    ranked
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let args = parse_args(&root)?;

    eprintln!(
        "reading {}..={} from {}",
        args.first,
        args.last,
        args.dir.display()
    );
    let started = std::time::Instant::now();
    let table = GameTable::load(&args.dir, args.first..=args.last)?;
    eprintln!("loaded in {:.1}s", started.elapsed().as_secs_f64());

    let counts = &table.counts;
    println!("== subject table ==");
    println!(
        "seasons {:?}  schedule rows (REG, all seasons) {}",
        table.seasons,
        table.schedule.len()
    );
    println!(
        "team-games {}   games missing from the schedule {}",
        counts.team_games, counts.games_without_schedule
    );
    println!(
        "qb: from schedule {} ({:.1}%)   fallback {} ({:.1}%)   unresolved {}",
        counts.qb_from_schedule,
        pct(counts.qb_from_schedule, counts.team_games),
        counts.qb_fallback,
        pct(counts.qb_fallback, counts.team_games),
        counts.qb_unresolved,
    );
    println!(
        "wr1: resolved {}   no history {}   dropped inactive {} ({:.1}% of named)",
        counts.wr1_resolved,
        counts.wr1_no_history,
        counts.wr1_dropped_inactive,
        pct(
            counts.wr1_dropped_inactive,
            counts.wr1_resolved + counts.wr1_dropped_inactive
        ),
    );
    println!(
        "rb1: resolved {}   no history {}   dropped inactive {} ({:.1}% of named)",
        counts.rb1_resolved,
        counts.rb1_no_history,
        counts.rb1_dropped_inactive,
        pct(
            counts.rb1_dropped_inactive,
            counts.rb1_resolved + counts.rb1_dropped_inactive
        ),
    );
    println!(
        "pass-TD share missing on {} rows ({:.1}%)",
        counts.pass_td_share_missing,
        pct(counts.pass_td_share_missing, counts.team_games)
    );

    // A sanity check the fixtures cannot give: do the pre-game rules name the
    // players a human would call the starters?
    println!("\n== 2024 subject resolution, most common names ==");
    let mut wr1_names: BTreeMap<String, usize> = BTreeMap::new();
    let mut rb1_names: BTreeMap<String, usize> = BTreeMap::new();
    let mut qb_names: BTreeMap<String, usize> = BTreeMap::new();
    for row in table.rows.iter().filter(|row| row.season == 2024) {
        if let Some(player) = &row.wr1 {
            *wr1_names.entry(player.display_name.clone()).or_default() += 1;
        }
        if let Some(player) = &row.rb1 {
            *rb1_names.entry(player.display_name.clone()).or_default() += 1;
        }
        if let Some(player) = &row.qb {
            *qb_names.entry(player.display_name.clone()).or_default() += 1;
        }
    }
    for (label, names) in [("wr1", &wr1_names), ("rb1", &rb1_names), ("qb ", &qb_names)] {
        let top: Vec<String> = most_common(names, 3)
            .into_iter()
            .map(|(name, count)| format!("{name} ({count})"))
            .collect();
        println!("  {label}: {}", top.join(", "));
    }

    // Cluster ids are FNV-1a over a canonical string; a collision would merge
    // two subject-seasons into one bootstrap unit without saying so.
    let mut subject_keys: BTreeSet<SubjectKey> = BTreeSet::new();
    for row in &table.rows {
        subject_keys.insert(row.team_season());
        for player in [row.qb.as_ref(), row.wr1.as_ref(), row.rb1.as_ref()]
            .into_iter()
            .flatten()
        {
            subject_keys.insert(SubjectKey::Player(player.player_id.clone(), row.season));
        }
    }
    let distinct_ids: BTreeSet<u64> = subject_keys.iter().map(SubjectKey::cluster_id).collect();
    println!(
        "\ncluster ids: {} subject-seasons -> {} distinct ids ({} collisions)",
        subject_keys.len(),
        distinct_ids.len(),
        subject_keys.len() - distinct_ids.len()
    );

    let mut writer = csv_writer(&args.out)?;
    writer.write_record([
        "family",
        "season",
        "observations",
        "clusters",
        "missing_covariate",
    ])?;

    for family in families::ALL {
        let rows = families::observations(family, &table);
        let [leg_a, leg_b] = family.legs();
        let covariate = family.covariate();
        println!("\n== {} ==", family.key());
        println!("  A: {}      B: {}", leg_a.label(), leg_b.label());
        println!(
            "  cluster: {} ({})   covariate: {} ({})",
            family.cluster_description(),
            match family.cluster_scheme() {
                ClusterScheme::SingleSubjectSeason => "one-way",
                ClusterScheme::TwoOneWay => "two one-way, wider wins",
            },
            covariate.label,
            match covariate.scheme {
                CovariateScheme::Terciles => "terciles",
                CovariateScheme::SpreadBuckets => "fav3+/within3/dog3+",
            },
        );
        if family.is_market_centred() {
            println!("  market-centred: Phase 3 skips §1.2 standardisation here");
        }
        let ids = clusters(&rows);
        let missing = families::missing_covariate(&rows);
        println!(
            "  observations {}   clusters {}   missing covariate {} ({:.1}%)",
            rows.len(),
            ids.len(),
            missing,
            pct(missing, rows.len())
        );
        let x_a: Vec<f64> = rows.iter().map(|row| row.x_a).collect();
        let x_b: Vec<f64> = rows.iter().map(|row| row.x_b).collect();
        let covariates: Vec<f64> = rows.iter().filter_map(|row| row.covariate).collect();
        println!("{}", summary_line("x_a", &x_a));
        println!("{}", summary_line("x_b", &x_b));
        println!("{}", summary_line("cov", &covariates));

        let mut per_season: BTreeMap<u16, (usize, BTreeSet<u64>, usize)> = BTreeMap::new();
        for row in &rows {
            let entry = per_season.entry(row.season).or_default();
            entry.0 += 1;
            entry.1.insert(row.cluster_primary);
            entry.1.extend(row.cluster_secondary);
            if row.covariate.is_none() {
                entry.2 += 1;
            }
        }
        let line: Vec<String> = per_season
            .iter()
            .map(|(season, (count, _, _))| format!("{season} {count}"))
            .collect();
        println!("  per season: {}", line.join("  "));
        for (season, (count, ids, missing)) in per_season {
            writer.write_record([
                family.key().to_string(),
                season.to_string(),
                count.to_string(),
                ids.len().to_string(),
                missing.to_string(),
            ])?;
        }
    }
    writer.flush()?;
    println!("\nwrote {}", args.out.display());
    Ok(())
}

fn pct(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        0.0
    } else {
        100.0 * part as f64 / whole as f64
    }
}

fn csv_writer(path: &Path) -> Result<csv::Writer<std::fs::File>, Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(csv::Writer::from_path(path)?)
}

fn parse_args(root: &Path) -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = Args {
        first: DEFAULT_FIRST,
        last: DEFAULT_LAST,
        dir: root.join("data/raw/nflverse"),
        out: root.join("outputs/family_counts.csv"),
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
            "--seasons" => {
                let text = value()?;
                let (first, last) = text
                    .split_once('-')
                    .ok_or_else(|| format!("--seasons wants FIRST-LAST, got {text:?}"))?;
                args.first = first.trim().parse()?;
                args.last = last.trim().parse()?;
                if args.last < args.first {
                    return Err(format!("season range {text} runs backwards").into());
                }
            }
            "--dir" => args.dir = PathBuf::from(value()?),
            "--out" => args.out = PathBuf::from(value()?),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    Ok(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_five_number_summary_interpolates_and_survives_edges() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(five_number(&values), Some([1.0, 2.0, 3.0, 4.0, 5.0]));
        // Four points: the quartiles fall between order statistics.
        let quartet = [0.0, 1.0, 2.0, 3.0];
        let summary = five_number(&quartet).expect("four values");
        assert_eq!(summary[0], 0.0);
        assert!((summary[1] - 0.75).abs() < 1e-12);
        assert!((summary[2] - 1.5).abs() < 1e-12);
        assert!((summary[3] - 2.25).abs() < 1e-12);
        assert_eq!(summary[4], 3.0);
        assert_eq!(five_number(&[7.0]), Some([7.0; 5]));
        assert_eq!(five_number(&[]), None);
        // Unsorted input sorts first.
        assert_eq!(
            five_number(&[5.0, 1.0, 3.0]),
            Some([1.0, 2.0, 3.0, 4.0, 5.0])
        );
    }

    #[test]
    fn the_name_ranking_breaks_ties_by_name_so_the_output_is_stable() {
        let counts = BTreeMap::from([
            ("Zeke".to_string(), 5),
            ("Adam".to_string(), 5),
            ("Bo".to_string(), 9),
            ("Cy".to_string(), 1),
        ]);
        assert_eq!(
            most_common(&counts, 3),
            vec![
                ("Bo".to_string(), 9),
                ("Adam".to_string(), 5),
                ("Zeke".to_string(), 5)
            ]
        );
    }

    #[test]
    fn percentages_do_not_divide_by_zero() {
        assert_eq!(pct(0, 0), 0.0);
        assert!((pct(1, 4) - 25.0).abs() < 1e-12);
    }

    /// Every family the counter walks knows its own labels, so a new family
    /// cannot be added without the report gaining a block for it.
    #[test]
    fn every_family_has_the_labels_the_report_prints() {
        for family in families::ALL {
            assert!(!family.key().is_empty());
            assert!(!family.cluster_description().is_empty());
            assert!(!family.covariate().label.is_empty());
            assert!(!family.covariate().description.is_empty());
            for leg in family.legs() {
                assert!(leg.label().split(' ').count() >= 3);
            }
        }
        assert_eq!(families::ALL.len(), 6);
        assert!(Family::SpreadXTotal.is_market_centred());
    }
}
