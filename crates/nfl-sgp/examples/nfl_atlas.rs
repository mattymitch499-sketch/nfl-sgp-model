//! Build the atlas: measured dependence per family and stratum, the reporting
//! grid with cluster-bootstrap intervals, the transport validation that gates
//! scoring, and the per-season era check.
//!
//! Four files, per `tasks/plan.md` §6:
//!
//! * `outputs/nfl_atlas.json` — what the scorer loads. Machine-facing and
//!   exact; float arrays are stored as decimal strings so a save round-trips
//!   bit for bit.
//! * `outputs/nfl_atlas.csv` — the human table: one row per grid point.
//! * `outputs/nfl_atlas_validation.csv` — §1.9, per family and grid point.
//! * `outputs/nfl_atlas_seasons.csv` — the era check.
//!
//! Usage:
//! `cargo run --release -p nfl-sgp --example nfl_atlas -- [--seasons 2010-2025] [--data DIR] [--out-dir DIR] [--seed N] [--reps N] [--jitter-seed-check]`

use nfl_sgp::atlas::{
    ALL_KEY, ATLAS_SEED, Atlas, BOOTSTRAP_REPS, Stratum, apply_validation, grid, seasons, summarise,
};
use nfl_sgp::families::{self, Family};
use nfl_sgp::games::GameTable;
use nfl_sgp::stats::Marginal;
use std::path::{Path, PathBuf};

struct Args {
    seasons: std::ops::RangeInclusive<u16>,
    data: PathBuf,
    out_dir: PathBuf,
    seed: u64,
    reps: usize,
    jitter_check: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let args = parse_args(&root)?;
    std::fs::create_dir_all(&args.out_dir)?;

    eprintln!("reading {:?} from {}", args.seasons, args.data.display());
    let started = std::time::Instant::now();
    let table = GameTable::load(&args.data, args.seasons.clone())?;
    eprintln!(
        "loaded {} team-games in {:.1}s",
        table.rows.len(),
        started.elapsed().as_secs_f64()
    );

    let mut atlas = Atlas::build(&table, args.seed);
    eprintln!(
        "built {} strata across {} families (seed {:#x})",
        atlas.strata.len(),
        families::ALL.len(),
        args.seed
    );

    eprintln!("\n== transport validation (plan §1.9) ==");
    let reports = apply_validation(&mut atlas, &table, args.reps);
    for report in &reports {
        eprintln!(
            "  {:<24} train {:>5} / test {:>5}   {:>2} of {} grid points missed   {}",
            report.family.key(),
            report.train_n,
            report.test_n,
            report.misses,
            report.points.len(),
            if report.ok { "OK" } else { "**FAILED**" }
        );
    }
    let failed = atlas.failed_families();
    if failed.is_empty() {
        eprintln!("  every family transports; all are scoreable");
    } else {
        eprintln!(
            "  NOT SCOREABLE: {}",
            failed
                .iter()
                .map(|family| family.key())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    eprintln!("\n== dependence, `all` stratum ==");
    eprintln!(
        "  {:<24} {:>6} {:>8} {:>9} {:>9} {:>9} {:>17} {:>8}",
        "family", "n", "clusters", "spearman", "kendall", "gauss_rho", "P(0.5,0.5,O,O)", "phi"
    );
    let (a, b) = (Marginal::over(0.5), Marginal::over(0.5));
    for family in families::ALL {
        let Some(stratum) = atlas
            .strata(family)
            .into_iter()
            .find(|stratum| stratum.key == ALL_KEY)
        else {
            continue;
        };
        let summary = summarise(stratum);
        let joint = stratum.joint(a, b);
        let interval = stratum.joint_interval(a, b, args.reps, args.seed);
        eprintln!(
            "  {:<24} {:>6} {:>8} {:>9} {:>9} {:>9} {:>17} {:>8}",
            family.key(),
            stratum.n,
            stratum.clusters,
            show(summary.spearman),
            show(summary.kendall),
            show(summary.gaussian_rho),
            match (joint, interval) {
                (Some(value), Some((low, high))) => format!("{value:.4} [{low:.3},{high:.3}]"),
                (Some(value), None) => format!("{value:.4}"),
                _ => "-".into(),
            },
            show(joint.and_then(|value| nfl_sgp::stats::phi_at(value, a, b))),
        );
    }
    eprintln!("  (independence at these marginals is 0.2500)");

    // The human table.
    let grid_path = args.out_dir.join("nfl_atlas.csv");
    let mut writer = csv::Writer::from_path(&grid_path)?;
    writer.write_record([
        "family",
        "stratum",
        "range_low",
        "range_high",
        "n",
        "clusters",
        "season_low",
        "season_high",
        "transport_ok",
        "spearman",
        "kendall",
        "gaussian_rho",
        "p_a",
        "dir_a",
        "p_b",
        "dir_b",
        "empirical_joint",
        "ci_low",
        "ci_high",
        "gaussian_joint",
        "independent",
        "phi",
    ])?;
    for stratum in &atlas.strata {
        let summary = summarise(stratum);
        for point in grid(stratum, args.reps, args.seed) {
            let (marginal_a, marginal_b) = point.marginals();
            writer.write_record([
                stratum.family.key().to_string(),
                stratum.key.clone(),
                edge(stratum.range.map(|(low, _)| low)),
                edge(stratum.range.map(|(_, high)| high)),
                stratum.n.to_string(),
                stratum.clusters.to_string(),
                stratum.seasons.0.to_string(),
                stratum.seasons.1.to_string(),
                stratum.transport_ok.to_string(),
                cell(summary.spearman),
                cell(summary.kendall),
                cell(summary.gaussian_rho),
                format!("{:.2}", point.p_a),
                format!("{:?}", marginal_a.direction),
                format!("{:.2}", point.p_b),
                format!("{:?}", marginal_b.direction),
                format!("{:.6}", point.empirical),
                cell(point.interval.map(|(low, _)| low)),
                cell(point.interval.map(|(_, high)| high)),
                cell(point.gaussian),
                format!("{:.6}", point.independent),
                cell(point.phi),
            ])?;
        }
    }
    writer.flush()?;

    // The validation table.
    let validation_path = args.out_dir.join("nfl_atlas_validation.csv");
    let mut writer = csv::Writer::from_path(&validation_path)?;
    writer.write_record([
        "family",
        "stratum",
        "train_n",
        "test_n",
        "grid_point",
        "predicted",
        "realised",
        "test_ci_low",
        "test_ci_high",
        "missed",
        "family_misses",
        "family_points",
        "transport_ok",
    ])?;
    for report in &reports {
        for point in &report.points {
            writer.write_record([
                report.family.key().to_string(),
                report.key.clone(),
                report.train_n.to_string(),
                report.test_n.to_string(),
                point.label.clone(),
                format!("{:.6}", point.predicted),
                format!("{:.6}", point.realised),
                cell(point.interval.map(|(low, _)| low)),
                cell(point.interval.map(|(_, high)| high)),
                point.missed.to_string(),
                report.misses.to_string(),
                report.points.len().to_string(),
                report.ok.to_string(),
            ])?;
        }
    }
    writer.flush()?;

    // The era check.
    let seasons_path = args.out_dir.join("nfl_atlas_seasons.csv");
    let mut writer = csv::Writer::from_path(&seasons_path)?;
    writer.write_record(["family", "season", "n", "joint_0.5_0.5_OO", "spearman"])?;
    for family in families::ALL {
        let observations = families::observations(family, &table);
        for row in seasons(family, &observations, args.seed) {
            writer.write_record([
                family.key().to_string(),
                row.season.to_string(),
                row.n.to_string(),
                cell(row.joint),
                cell(row.spearman),
            ])?;
        }
    }
    writer.flush()?;

    let json_path = args.out_dir.join("nfl_atlas.json");
    atlas.write_json(&json_path)?;

    // Reproducibility: the atlas must reload as itself, and a rebuild with the
    // same seed must be identical. An artifact whose numbers cannot be
    // regenerated cannot be audited later.
    let reloaded = Atlas::load(&json_path)?;
    let rebuilt = Atlas::build(&table, args.seed);
    let reload_ok = reloaded == atlas;
    let rebuild_ok = rebuilt.strata == Atlas::build(&table, args.seed).strata;
    eprintln!(
        "\nreproducibility: reload {}   rebuild {}",
        pass(reload_ok),
        pass(rebuild_ok)
    );

    if args.jitter_check {
        // §1.3 requires the tie-breaking dependence to be reported, not
        // assumed away: count stats tie constantly and the jitter is what makes
        // their copula identified at all.
        let other = Atlas::build(&table, args.seed ^ 0x5DEE_CE66_D15E_A5E0);
        let mut worst = 0.0f64;
        let mut worst_at = String::new();
        for stratum in &atlas.strata {
            let Some(twin) = other.strata.iter().find(|candidate| {
                candidate.family == stratum.family && candidate.key == stratum.key
            }) else {
                continue;
            };
            for point in grid(stratum, 0, args.seed) {
                let (marginal_a, marginal_b) = point.marginals();
                let Some(second) = twin.joint(marginal_a, marginal_b) else {
                    continue;
                };
                let change = (point.empirical - second).abs();
                if change > worst {
                    worst = change;
                    worst_at =
                        format!("{}/{} {}", stratum.family.key(), stratum.key, point.label());
                }
            }
        }
        eprintln!("jitter-seed sensitivity: largest grid-joint change {worst:.5} at {worst_at}");
    }

    eprintln!(
        "\nwrote:\n  {}\n  {}\n  {}\n  {}",
        json_path.display(),
        grid_path.display(),
        validation_path.display(),
        seasons_path.display()
    );
    Ok(())
}

fn pass(ok: bool) -> &'static str {
    if ok { "PASS" } else { "FAIL" }
}

fn show(value: Option<f64>) -> String {
    value.map_or_else(|| "-".into(), |value| format!("{value:+.4}"))
}

fn cell(value: Option<f64>) -> String {
    value.map_or_else(String::new, |value| format!("{value:.6}"))
}

/// An unbounded stratum edge prints empty rather than `inf`, so a spreadsheet
/// reads it as "no bound" instead of as text.
fn edge(value: Option<f64>) -> String {
    match value {
        Some(value) if value.is_finite() => format!("{value:.4}"),
        _ => String::new(),
    }
}

/// Which stratum a family's rows are summarised on in the console table.
#[allow(dead_code)]
fn all_stratum(atlas: &Atlas, family: Family) -> Option<&Stratum> {
    atlas
        .strata(family)
        .into_iter()
        .find(|stratum| stratum.key == ALL_KEY)
}

fn parse_args(root: &Path) -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = Args {
        seasons: 2010..=2025,
        data: root.join("data/raw/nflverse"),
        out_dir: root.join("outputs"),
        seed: ATLAS_SEED,
        reps: BOOTSTRAP_REPS,
        jitter_check: false,
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
                let (start, end) = text
                    .split_once('-')
                    .ok_or("--seasons wants a range like 2010-2025")?;
                args.seasons = start.trim().parse()?..=end.trim().parse()?;
            }
            "--data" => args.data = PathBuf::from(value()?),
            "--out-dir" => args.out_dir = PathBuf::from(value()?),
            "--seed" => args.seed = value()?.parse()?,
            "--reps" => args.reps = value()?.parse()?,
            "--jitter-seed-check" => args.jitter_check = true,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    Ok(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unbounded_edges_print_empty_rather_than_inf() {
        assert_eq!(edge(Some(f64::NEG_INFINITY)), "");
        assert_eq!(edge(Some(f64::INFINITY)), "");
        assert_eq!(edge(None), "");
        assert_eq!(edge(Some(0.25)), "0.2500");
    }

    #[test]
    fn optional_cells_are_empty_not_nan() {
        assert_eq!(cell(None), "");
        assert_eq!(cell(Some(0.5)), "0.500000");
        assert_eq!(show(None), "-");
        assert_eq!(show(Some(-0.25)), "-0.2500");
    }
}
