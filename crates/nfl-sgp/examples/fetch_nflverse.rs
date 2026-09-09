//! Download the nflverse files the atlas measures φ_true from.
//!
//! Per season: `stats_player_week_{season}.csv` (~8 MB) and
//! `stats_team_week_{season}.csv` (~0.2 MB). Once: `games.csv` (~2 MB, every
//! season 1999→now, with closing spread and total lines). Sixteen seasons is
//! roughly 135 MB into `data/raw/nflverse/`, which is gitignored.
//!
//! Completed seasons never change, so an existing per-season file is skipped
//! unless `--force`. `games.csv` is one rolling file that nflverse updates
//! through the current season, so it is **always** re-downloaded — a stale
//! copy would silently miss this week's results.
//!
//! Usage:
//! `cargo run --release -p nfl-sgp --example fetch_nflverse -- --seasons 2010-2025 [--dest data/raw/nflverse] [--force] [--schedules-only]`

use nfl_sgp::nflverse::{
    depth_charts_path, depth_charts_url, download_if_published, player_week_path, player_week_url,
    schedules_path, schedules_url, team_week_path, team_week_url,
};
use std::path::{Path, PathBuf};

struct Args {
    seasons: Vec<u16>,
    dest: PathBuf,
    force: bool,
    schedules_only: bool,
    /// Also fetch each season's official depth charts. Always re-downloaded:
    /// the current season's file accumulates snapshots all year, so a cached
    /// copy silently serves an older chart.
    depth_charts: bool,
}

/// `2010-2025`, `2019,2021,2024`, or a mix. Ascending, deduplicated.
fn parse_seasons(text: &str) -> Result<Vec<u16>, String> {
    let mut seasons = Vec::new();
    for part in text
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        match part.split_once('-') {
            Some((start, end)) => {
                let start: u16 = start
                    .trim()
                    .parse()
                    .map_err(|_| format!("bad season {start:?}"))?;
                let end: u16 = end
                    .trim()
                    .parse()
                    .map_err(|_| format!("bad season {end:?}"))?;
                if end < start {
                    return Err(format!("range {part} runs backwards"));
                }
                seasons.extend(start..=end);
            }
            None => seasons.push(part.parse().map_err(|_| format!("bad season {part:?}"))?),
        }
    }
    seasons.sort_unstable();
    seasons.dedup();
    if seasons
        .iter()
        .any(|&season| season < nfl_sgp::nflverse::FIRST_SEASON)
    {
        return Err(format!(
            "nflverse weekly files start in {}",
            nfl_sgp::nflverse::FIRST_SEASON
        ));
    }
    Ok(seasons)
}

fn fetch(
    label: &str,
    url: &str,
    dest: &Path,
    force: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if dest.exists() && !force {
        eprintln!("  {label}: present, skipped ({})", dest.display());
        return Ok(());
    }
    // A season that has not started yet has no weekly files, which is normal in
    // September and must not abort the depth-chart fetch that follows it.
    match download_if_published(url, dest)? {
        Some(bytes) => eprintln!(
            "  {label}: {:.1} MB -> {}",
            bytes as f64 / 1_048_576.0,
            dest.display()
        ),
        None => eprintln!("  {label}: not published yet, skipped"),
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let args = parse_args(&root)?;
    // Validate before the first byte moves, so a typo costs nothing.
    if args.seasons.is_empty() && !args.schedules_only {
        return Err("no --seasons given (e.g. --seasons 2010-2025); pass --schedules-only to fetch only games.csv".into());
    }
    std::fs::create_dir_all(&args.dest)?;

    // The schedule is one rolling file; always refresh it.
    eprintln!("schedules");
    fetch(
        "games.csv",
        &schedules_url(),
        &schedules_path(&args.dest),
        true,
    )?;
    if args.schedules_only {
        return Ok(());
    }
    for &season in &args.seasons {
        eprintln!("{season}");
        fetch(
            "player weeks",
            &player_week_url(season),
            &player_week_path(&args.dest, season),
            args.force,
        )?;
        fetch(
            "team weeks",
            &team_week_url(season),
            &team_week_path(&args.dest, season),
            args.force,
        )?;
        if args.depth_charts {
            // Always forced: from 2025 the file is an accumulating log of
            // snapshots, so a cached copy is a stale depth chart — which parses
            // perfectly and names players who have since been traded.
            fetch(
                "depth charts",
                &depth_charts_url(season),
                &depth_charts_path(&args.dest, season),
                true,
            )?;
        }
    }
    eprintln!(
        "done: seasons on disk {:?}",
        nfl_sgp::nflverse::downloaded_seasons(&args.dest)
    );
    Ok(())
}

fn parse_args(root: &Path) -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = Args {
        seasons: Vec::new(),
        dest: root.join("data/raw/nflverse"),
        force: false,
        schedules_only: false,
        depth_charts: false,
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
            "--seasons" => args.seasons = parse_seasons(&value()?)?,
            "--dest" => args.dest = PathBuf::from(value()?),
            "--force" => args.force = true,
            "--schedules-only" => args.schedules_only = true,
            "--depth-charts" => args.depth_charts = true,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    Ok(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn season_ranges_and_lists_parse_ascending_and_deduplicated() {
        assert_eq!(
            parse_seasons("2010-2013").unwrap(),
            vec![2010, 2011, 2012, 2013]
        );
        assert_eq!(
            parse_seasons("2024, 2019,2021").unwrap(),
            vec![2019, 2021, 2024]
        );
        assert_eq!(
            parse_seasons("2022-2024,2023").unwrap(),
            vec![2022, 2023, 2024]
        );
    }

    #[test]
    fn junk_and_backwards_ranges_and_pre_1999_are_refused() {
        assert!(parse_seasons("twenty").is_err());
        assert!(parse_seasons("2020-2010").is_err());
        assert!(parse_seasons("1998").is_err());
    }
}
