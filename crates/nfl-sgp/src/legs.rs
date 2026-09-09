//! Parsing the hand-collected SGP sheet's leg descriptions — `tasks/plan.md` §3.
//!
//! Every row of `data/processed/sgp_prices.csv` names its two legs in whatever
//! shorthand was quickest to type at the book's board: `"Mahomes over 274.5
//! pass yds"`, `"Chiefs team total over 24.5"`, `"game total over 47.5"`. Two
//! consumers need those strings turned into structure, and they need
//! *different* parts of it:
//!
//! * the **marginal resolver** needs the market key, the subject and the line,
//!   so it can find the same leg on the leg board and devig it;
//! * the **atlas lookup** needs to know which of the family's two legs this is,
//!   so the joint is read at the right coordinate and in the right direction.
//!
//! So a parse is only complete when it agrees with the family the row declares.
//! That cross-check is the whole reason this module refuses rather than
//! guesses: a sheet row whose `family` column says `passyds_x_wr1recyds` but
//! whose leg B reads `"over 68.5 rush yds"` is a collection mistake, and
//! scoring it against the receiving-yards dependence would produce a number
//! that looks fine and means nothing. [`parse_row`] names the row and the
//! disagreement instead.
//!
//! ## What is deliberately not here
//!
//! No fuzzy matching on player names, and no fallback when a stat word is
//! unrecognised. The sheet is typed by one person a handful of times a week;
//! the cost of a loud refusal is re-typing one cell, and the cost of a
//! confident wrong parse is a corrupted row in a study whose whole point is
//! that its numbers can be trusted. The baseball study made the same call and
//! it caught real mistakes.

use crate::families::{Direction as FamilyDirection, Family, Leg, Role};
use crate::games::Stat;
use crate::names::join_key;
use crate::stats::{Direction, Marginal};

/// The quantity a leg is written on, as the *sheet* names it — which is the
/// book's vocabulary, not the dataset's.
///
/// This is a separate type from [`Stat`] on purpose. [`Stat`] is a column in
/// the historical table (`Wr1ReceivingYards` is "the receiving yards of
/// whoever the as-of Wr1 was"); a `LegStat` is a market a book offers on a
/// *named* player. The map between them is [`LegStat::stat`], and it is the
/// only place the two vocabularies meet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LegStat {
    PassYds,
    PassTds,
    RushYds,
    RecYds,
    Receptions,
    /// One team's own points, `team_totals` on the feed.
    TeamTotal,
    /// The game's points, `totals` on the feed. No subject.
    GameTotal,
    /// Against the closing spread, `spreads` on the feed.
    Spread,
}

impl LegStat {
    /// The Odds API market key, which is how the leg board is keyed and
    /// therefore how a marginal is found.
    pub fn market_key(self) -> &'static str {
        match self {
            Self::PassYds => "player_pass_yds",
            Self::PassTds => "player_pass_tds",
            Self::RushYds => "player_rush_yds",
            Self::RecYds => "player_reception_yds",
            Self::Receptions => "player_receptions",
            Self::TeamTotal => "team_totals",
            Self::GameTotal => "totals",
            Self::Spread => "spreads",
        }
    }

    /// The historical column this market settles against.
    ///
    /// The player markets map onto the *role* columns, which is sound only
    /// because [`parse_row`] has already checked that the family assigns this
    /// leg to the matching role: `player_reception_yds` becomes
    /// [`Stat::Wr1ReceivingYards`] because the family said leg B is the Wr1,
    /// not because the parser knows who the player is.
    pub fn stat(self) -> Stat {
        match self {
            Self::PassYds => Stat::QbPassingYards,
            Self::PassTds => Stat::QbPassingTds,
            Self::RushYds => Stat::Rb1RushingYards,
            Self::RecYds => Stat::Wr1ReceivingYards,
            Self::Receptions => Stat::Wr1Receptions,
            Self::TeamTotal => Stat::TeamPoints,
            Self::GameTotal => Stat::TotalVsLine,
            Self::Spread => Stat::MarginVsSpread,
        }
    }

    /// The market a historical column is priced on — the inverse of
    /// [`LegStat::stat`].
    ///
    /// This exists so the set of markets a week needs is **derived from the
    /// families being collected** rather than typed a second time in an
    /// example. A hand-maintained second list is a list that drifts: add a
    /// family whose market nobody remembered to add, and the leg board silently
    /// never fetches it, and every row of that family comes back unscoreable
    /// for a reason that looks like the book's fault.
    ///
    /// [`LegStat::stat`] is injective across all eight variants (each `Stat` is
    /// produced by exactly one), so this really is an inverse and not a guess.
    /// The team-aggregate columns have no market of their own and return
    /// `None`.
    pub fn for_stat(stat: Stat) -> Option<Self> {
        [
            Self::PassYds,
            Self::PassTds,
            Self::RushYds,
            Self::RecYds,
            Self::Receptions,
            Self::TeamTotal,
            Self::GameTotal,
            Self::Spread,
        ]
        .into_iter()
        .find(|candidate| candidate.stat() == stat)
    }

    /// Whether the subject is a named player (as opposed to a team, or the
    /// game itself). Player legs need a name; the others must not carry one
    /// that looks like a player's.
    pub fn is_player(self) -> bool {
        matches!(
            self,
            Self::PassYds | Self::PassTds | Self::RushYds | Self::RecYds | Self::Receptions
        )
    }
}

/// One leg of a collected row, after parsing.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedLeg {
    /// Display spelling as typed, kept for the output file so a scored row can
    /// be read back against the sheet.
    pub subject: String,
    /// [`join_key`] of `subject` — the key the leg board is joined on.
    pub subject_key: String,
    pub stat: LegStat,
    /// The line. `None` only for a moneyline, which no v1 family uses.
    pub point: Option<f64>,
    pub direction: Direction,
}

impl ParsedLeg {
    /// This leg at a devigged probability, ready for the atlas.
    pub fn marginal(&self, p: f64) -> Marginal {
        Marginal {
            p,
            direction: self.direction,
        }
    }
}

/// Both legs of a row, checked against the family it declares.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedRow {
    pub family: Family,
    pub leg_a: ParsedLeg,
    pub leg_b: ParsedLeg,
}

/// Parses one sheet row: the declared family plus its two leg descriptions.
///
/// The family is the authority on *roles*; the descriptions supply subject,
/// line and direction. Both legs must agree with the family on stat and
/// direction, or the row is refused with a message naming the row.
///
/// `Err` carries a human sentence meant to be printed straight into the
/// scorer's output beside the row it came from.
pub fn parse_row(family_key: &str, leg_a: &str, leg_b: &str) -> Result<ParsedRow, String> {
    let family = Family::from_key(family_key).ok_or_else(|| {
        format!(
            "unknown family {family_key:?}; known: {}",
            crate::families::ALL
                .iter()
                .map(|f| f.key())
                .collect::<Vec<_>>()
                .join(", ")
        )
    })?;
    let [expected_a, expected_b] = family.legs();
    let parsed_a = parse_leg(leg_a).map_err(|why| format!("leg A {leg_a:?}: {why}"))?;
    let parsed_b = parse_leg(leg_b).map_err(|why| format!("leg B {leg_b:?}: {why}"))?;
    check(&parsed_a, expected_a, "A", family)?;
    check(&parsed_b, expected_b, "B", family)?;
    Ok(ParsedRow {
        family,
        leg_a: parsed_a,
        leg_b: parsed_b,
    })
}

/// One leg must match the family's declaration on both stat and direction.
///
/// Direction is checked as hard as the stat. `rb1rush_x_qbpassunder` and
/// `passyds_x_rb1rush` are built from the *same* pair of raw columns and differ
/// only in which side of the line each leg sits on (plan §2 lists both on
/// purpose, one as the game-script family and one as its near-null control).
/// A row that declared the first but typed an over on the passing leg would
/// otherwise be silently scored against the wrong family's dependence.
fn check(parsed: &ParsedLeg, expected: Leg, which: &str, family: Family) -> Result<(), String> {
    if parsed.stat.stat() != expected.stat {
        return Err(format!(
            "leg {which} reads as {} but family {} declares {} — one of the two is wrong",
            parsed.stat.stat().label(),
            family.key(),
            expected.stat.label()
        ));
    }
    let expected_direction = match expected.direction {
        FamilyDirection::Over => Direction::Over,
        FamilyDirection::Under => Direction::Under,
    };
    if parsed.direction != expected_direction {
        return Err(format!(
            "leg {which} is an {:?} but family {} declares {:?}",
            parsed.direction,
            family.key(),
            expected_direction
        ));
    }
    if parsed.stat.is_player() && parsed.subject_key.is_empty() {
        return Err(format!(
            "leg {which} is a player market with no player named"
        ));
    }
    if expected.role == Role::Game && parsed.stat != LegStat::GameTotal {
        return Err(format!(
            "leg {which} is on the whole game in family {} but reads as {:?}",
            family.key(),
            parsed.stat
        ));
    }
    Ok(())
}

/// Parses one leg description into subject, stat, line and direction.
///
/// The grammar of plan §3, tolerant of case and punctuation. Recognised, with
/// the subject being everything before the direction word:
///
/// ```text
/// <Player> over|under <line> pass yds | passing yards
/// <Player> over|under <line> pass tds | passing touchdowns
/// <Player> over|under <line> rush yds | rushing yards
/// <Player> over|under <line> rec yds  | receiving yards
/// <Player> over|under <line> receptions | recs | catches
/// <Team>   team total over|under <line>
/// game total over|under <line>
/// <Team> -3.5 | <Team> +3.5            (spread; the sign is the direction)
/// ```
///
/// The line may also precede the direction word (`"Mahomes 274.5 over pass
/// yds"`) because that is how some boards read it back; the stat words may
/// come before the line as well. What is *not* tolerated is a missing stat, a
/// missing number where one is required, or two direction words.
pub fn parse_leg(text: &str) -> Result<ParsedLeg, String> {
    let lowered = text.to_lowercase();
    let cleaned = lowered.replace(['(', ')', ','], " ");
    let tokens: Vec<&str> = cleaned.split_whitespace().collect();
    if tokens.is_empty() {
        return Err("empty description".into());
    }

    // A spread leg is the one shape with no direction word: the sign carries
    // it. Detect it first so "+3.5" is never read as a stray number.
    if let Some(leg) = parse_spread(text, &tokens)? {
        return Ok(leg);
    }

    let direction_at = tokens
        .iter()
        .position(|token| matches!(*token, "over" | "under" | "o" | "u"))
        .ok_or("no over/under in the description")?;
    if tokens[direction_at + 1..]
        .iter()
        .any(|token| matches!(*token, "over" | "under"))
    {
        return Err("two direction words".into());
    }
    let direction = match tokens[direction_at] {
        "over" | "o" => Direction::Over,
        _ => Direction::Under,
    };

    let stat = find_stat(&tokens)?;
    let point = tokens
        .iter()
        .find_map(|token| token.parse::<f64>().ok())
        .ok_or("no line in the description")?;

    // The subject is whatever precedes the direction word, minus the stat
    // words that a team-total or game-total leg puts there ("Chiefs team total
    // over 24.5" — "team total" is the stat, "Chiefs" the subject).
    let subject_tokens: Vec<&str> = tokens[..direction_at]
        .iter()
        .copied()
        .filter(|token| !is_stat_word(token) && token.parse::<f64>().is_err())
        .collect();
    let subject_display = subject_tokens.join(" ");

    if stat == LegStat::GameTotal && !subject_tokens.is_empty() && subject_tokens != ["game"] {
        return Err(format!(
            "a game total has no subject, but {subject_display:?} precedes it"
        ));
    }
    let subject = if stat == LegStat::GameTotal {
        String::new()
    } else {
        // Recover the caller's original casing for display: the subject is the
        // same token span in the untouched text.
        original_span(text, &subject_tokens)
    };
    if !subject_tokens.is_empty() && subject.is_empty() {
        return Err("subject could not be recovered from the description".into());
    }
    if stat != LegStat::GameTotal && subject_tokens.is_empty() {
        return Err("no subject before the direction word".into());
    }

    Ok(ParsedLeg {
        subject_key: join_key(&subject),
        subject,
        stat,
        point: Some(point),
        direction,
    })
}

/// `"Chiefs -2.5"` / `"Bills +2.5"` — a spread leg, where the sign is the
/// direction and there is no over/under word.
///
/// Returns `Ok(None)` when the text is not a spread, so the caller falls
/// through to the over/under grammar. A team laying points is "Over" its
/// margin-versus-spread residual in exactly the sense the atlas measures:
/// [`Stat::MarginVsSpread`] is `result − spread_line`, positive when the team
/// beat the number, and a bettor on either side of a spread is betting that
/// residual's sign. Both sides are therefore `Over` on the *anchor's* residual
/// — which is why the family's anchor is fixed at the home side and the sheet
/// must name the home team here.
fn parse_spread(text: &str, tokens: &[&str]) -> Result<Option<ParsedLeg>, String> {
    if tokens.iter().any(|token| is_direction_word(token)) {
        return Ok(None);
    }
    let Some(last) = tokens.last() else {
        return Ok(None);
    };
    if !(last.starts_with('+') || last.starts_with('-')) {
        return Ok(None);
    }
    let point: f64 = last
        .parse()
        .map_err(|_| format!("{last:?} looks like a spread but is not a number"))?;
    let subject_tokens: Vec<&str> = tokens[..tokens.len() - 1].to_vec();
    if subject_tokens.is_empty() {
        return Err("a spread with no team".into());
    }
    let subject = original_span(text, &subject_tokens);
    Ok(Some(ParsedLeg {
        subject_key: join_key(&subject),
        subject,
        stat: LegStat::Spread,
        point: Some(point),
        direction: Direction::Over,
    }))
}

fn is_direction_word(token: &str) -> bool {
    matches!(token, "over" | "under" | "o" | "u")
}

/// Every word that names a stat rather than a subject.
fn is_stat_word(token: &str) -> bool {
    matches!(
        token,
        "pass"
            | "passing"
            | "rush"
            | "rushing"
            | "rec"
            | "recs"
            | "receiving"
            | "reception"
            | "receptions"
            | "catches"
            | "yds"
            | "yards"
            | "yard"
            | "tds"
            | "td"
            | "touchdown"
            | "touchdowns"
            | "team"
            | "total"
            | "totals"
            | "points"
            | "pts"
            | "game"
    )
}

/// Which stat the description names, from the stat words present.
///
/// Order matters: "team total" and "game total" are checked before the player
/// markets, because "total" alone is ambiguous and the qualifier decides.
fn find_stat(tokens: &[&str]) -> Result<LegStat, String> {
    let has = |word: &str| tokens.contains(&word);
    let total = has("total") || has("totals");
    if total && has("game") {
        return Ok(LegStat::GameTotal);
    }
    if total && has("team") {
        return Ok(LegStat::TeamTotal);
    }
    let passing = has("pass") || has("passing");
    let rushing = has("rush") || has("rushing");
    let receiving = has("rec") || has("receiving") || has("reception");
    let yards = has("yds") || has("yards") || has("yard");
    let touchdowns = has("tds") || has("td") || has("touchdown") || has("touchdowns");
    let catches = has("receptions") || has("recs") || has("catches");

    match (passing, rushing, receiving, yards, touchdowns, catches) {
        (true, false, false, true, false, _) => Ok(LegStat::PassYds),
        (true, false, false, false, true, _) => Ok(LegStat::PassTds),
        (false, true, false, true, false, _) => Ok(LegStat::RushYds),
        (false, false, true, true, false, _) => Ok(LegStat::RecYds),
        (_, _, _, false, false, true) => Ok(LegStat::Receptions),
        (true, false, false, true, true, _) => {
            Err("both yards and touchdowns named on a passing leg".into())
        }
        (false, false, false, _, _, _) if total => {
            Err("a bare \"total\" is ambiguous — write \"team total\" or \"game total\"".into())
        }
        _ => Err(
            "no recognised stat; expected pass/rush/receiving yards, pass tds, \
             receptions, team total, game total or a spread"
                .into(),
        ),
    }
}

/// The original-cased span of `text` covering `wanted`, which are lowercase
/// tokens taken from it in order.
///
/// The sheet's spelling is what a reader will compare against the board, so
/// the output keeps "Ja'Marr Chase" rather than the folded key. Walking the
/// untouched text token-by-token is enough: the tokens came from it, so they
/// appear in the same order.
fn original_span(text: &str, wanted: &[&str]) -> String {
    if wanted.is_empty() {
        return String::new();
    }
    let mut kept: Vec<&str> = Vec::with_capacity(wanted.len());
    let mut next = 0usize;
    for token in text.split_whitespace() {
        if next == wanted.len() {
            break;
        }
        let folded = token.to_lowercase().replace(['(', ')', ','], "");
        if folded == wanted[next] {
            kept.push(token.trim_matches(|c: char| matches!(c, '(' | ')' | ',')));
            next += 1;
        }
    }
    kept.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leg(text: &str) -> ParsedLeg {
        parse_leg(text).unwrap_or_else(|why| panic!("{text:?} should parse: {why}"))
    }

    /// Every pattern the plan's §3 grammar lists, in the spellings a person
    /// actually types at a board.
    #[test]
    fn every_documented_pattern_parses() {
        let cases: [(&str, LegStat, f64, Direction, &str); 12] = [
            (
                "Patrick Mahomes over 274.5 pass yds",
                LegStat::PassYds,
                274.5,
                Direction::Over,
                "patrickmahomes",
            ),
            (
                "Josh Allen under 249.5 passing yards",
                LegStat::PassYds,
                249.5,
                Direction::Under,
                "joshallen",
            ),
            (
                "Patrick Mahomes over 1.5 pass tds",
                LegStat::PassTds,
                1.5,
                Direction::Over,
                "patrickmahomes",
            ),
            (
                "Jalen Hurts over 0.5 passing touchdowns",
                LegStat::PassTds,
                0.5,
                Direction::Over,
                "jalenhurts",
            ),
            (
                "Saquon Barkley over 68.5 rush yds",
                LegStat::RushYds,
                68.5,
                Direction::Over,
                "saquonbarkley",
            ),
            (
                "James Cook under 55.5 rushing yards",
                LegStat::RushYds,
                55.5,
                Direction::Under,
                "jamescook",
            ),
            (
                "Ja'Marr Chase over 78.5 rec yds",
                LegStat::RecYds,
                78.5,
                Direction::Over,
                "jamarrchase",
            ),
            (
                "A.J. Brown over 66.5 receiving yards",
                LegStat::RecYds,
                66.5,
                Direction::Over,
                "ajbrown",
            ),
            (
                "Travis Kelce over 4.5 receptions",
                LegStat::Receptions,
                4.5,
                Direction::Over,
                "traviskelce",
            ),
            (
                "Kansas City Chiefs team total over 24.5",
                LegStat::TeamTotal,
                24.5,
                Direction::Over,
                "kansascitychiefs",
            ),
            (
                "game total over 47.5",
                LegStat::GameTotal,
                47.5,
                Direction::Over,
                "",
            ),
            (
                "Chiefs -2.5",
                LegStat::Spread,
                -2.5,
                Direction::Over,
                "chiefs",
            ),
        ];
        for (text, stat, point, direction, key) in cases {
            let parsed = leg(text);
            assert_eq!(parsed.stat, stat, "{text}");
            assert_eq!(parsed.point, Some(point), "{text}");
            assert_eq!(parsed.direction, direction, "{text}");
            assert_eq!(parsed.subject_key, key, "{text}");
        }
    }

    /// The display spelling survives, because a scored row is read back
    /// against the sheet by eye.
    #[test]
    fn the_typed_spelling_is_kept_beside_the_folded_key() {
        let parsed = leg("Ja'Marr Chase over 78.5 rec yds");
        assert_eq!(parsed.subject, "Ja'Marr Chase");
        assert_eq!(parsed.subject_key, "jamarrchase");
    }

    /// Boards read the line back in either order, and abbreviate the
    /// direction. Neither changes the meaning.
    #[test]
    fn word_order_and_abbreviations_are_tolerated() {
        let a = leg("Mahomes over 274.5 pass yds");
        let b = leg("Mahomes 274.5 over pass yds");
        let c = leg("Mahomes o 274.5 pass yds");
        assert_eq!(a.stat, b.stat);
        assert_eq!(a.point, b.point);
        assert_eq!(a.direction, c.direction);
        assert_eq!(leg("Cook u 55.5 rush yds").direction, Direction::Under);
    }

    /// Market keys are the leg board's own, so the marginal join cannot drift.
    #[test]
    fn market_keys_match_the_feeds_vocabulary() {
        assert_eq!(LegStat::PassYds.market_key(), "player_pass_yds");
        assert_eq!(LegStat::RecYds.market_key(), "player_reception_yds");
        assert_eq!(LegStat::Receptions.market_key(), "player_receptions");
        assert_eq!(LegStat::TeamTotal.market_key(), "team_totals");
        assert_eq!(LegStat::GameTotal.market_key(), "totals");
        assert_eq!(LegStat::Spread.market_key(), "spreads");
    }

    /// And they land on the historical columns the atlas measures.
    #[test]
    fn stats_map_onto_the_atlas_columns() {
        assert_eq!(LegStat::PassYds.stat(), Stat::QbPassingYards);
        assert_eq!(LegStat::PassTds.stat(), Stat::QbPassingTds);
        assert_eq!(LegStat::RushYds.stat(), Stat::Rb1RushingYards);
        assert_eq!(LegStat::RecYds.stat(), Stat::Wr1ReceivingYards);
        assert_eq!(LegStat::TeamTotal.stat(), Stat::TeamPoints);
        assert_eq!(LegStat::GameTotal.stat(), Stat::TotalVsLine);
        assert_eq!(LegStat::Spread.stat(), Stat::MarginVsSpread);
    }

    /// **Every leg of every family must name a market**, or the generator
    /// cannot fetch the prices that leg needs and the whole family comes back
    /// unscoreable for a reason that looks like the book's fault.
    #[test]
    fn every_family_leg_maps_back_to_a_market() {
        for family in crate::families::ALL {
            for leg in family.legs() {
                let market = LegStat::for_stat(leg.stat).unwrap_or_else(|| {
                    panic!(
                        "{} leg {} has no market; the generator could not fetch it",
                        family.key(),
                        leg.label()
                    )
                });
                assert_eq!(
                    market.stat(),
                    leg.stat,
                    "for_stat must invert stat() for {}",
                    leg.label()
                );
            }
        }
    }

    /// The inverse is total over the eight variants and refuses the columns
    /// that genuinely have no market.
    #[test]
    fn for_stat_inverts_stat_and_refuses_team_aggregates() {
        for stat in [
            Stat::QbPassingYards,
            Stat::QbPassingTds,
            Stat::Rb1RushingYards,
            Stat::Wr1ReceivingYards,
            Stat::Wr1Receptions,
            Stat::TeamPoints,
            Stat::TotalVsLine,
            Stat::MarginVsSpread,
        ] {
            assert_eq!(LegStat::for_stat(stat).unwrap().stat(), stat);
        }
        assert_eq!(LegStat::for_stat(Stat::TeamPassingYards), None);
        assert_eq!(LegStat::for_stat(Stat::TeamRushingYards), None);
        assert_eq!(LegStat::for_stat(Stat::TeamPassingTds), None);
    }

    #[test]
    fn junk_is_refused_loudly() {
        for text in [
            "",
            "Mahomes 274.5 pass yds",            // no direction
            "Mahomes over pass yds",             // no line
            "over 47.5",                         // no stat
            "Mahomes over 274.5 sacks",          // unknown stat
            "Chiefs total over 24.5",            // ambiguous "total"
            "Mahomes over under 274.5 pass yds", // two directions
            "Chiefs game total over 47.5",       // a game total with a team
        ] {
            assert!(
                parse_leg(text).is_err(),
                "{text:?} must be refused, not guessed at"
            );
        }
    }

    /// A whole row, with the family agreeing.
    #[test]
    fn a_matching_row_parses() {
        let row = parse_row(
            "passyds_x_wr1recyds",
            "Patrick Mahomes over 274.5 pass yds",
            "Rashee Rice over 58.5 rec yds",
        )
        .unwrap();
        assert_eq!(row.family, Family::PassydsXWr1recyds);
        assert_eq!(row.leg_a.stat, LegStat::PassYds);
        assert_eq!(row.leg_b.subject_key, "rasheerice");
        assert_eq!(row.leg_a.marginal(0.52).direction, Direction::Over);
    }

    /// The cross-check that earns the module its refusals: right family, wrong
    /// stat typed into leg B.
    #[test]
    fn a_row_whose_leg_contradicts_its_family_is_refused() {
        let error = parse_row(
            "passyds_x_wr1recyds",
            "Patrick Mahomes over 274.5 pass yds",
            "Isiah Pacheco over 68.5 rush yds",
        )
        .unwrap_err();
        assert!(error.contains("leg B"), "{error}");
        assert!(error.contains("wr1_rec_yds"), "{error}");
    }

    /// Direction is checked as hard as the stat, because two families in plan
    /// §2 are built from the same two columns and differ only by it.
    #[test]
    fn direction_must_match_the_family_too() {
        // rb1rush_x_qbpassunder wants leg B UNDER; an over is the other family.
        let error = parse_row(
            "rb1rush_x_qbpassunder",
            "Saquon Barkley over 68.5 rush yds",
            "Jalen Hurts over 249.5 pass yds",
        )
        .unwrap_err();
        assert!(error.contains("leg B"), "{error}");
        assert!(error.contains("Under"), "{error}");
        // The same two legs, declared as the control family, are fine.
        assert!(
            parse_row(
                "passyds_x_rb1rush",
                "Jalen Hurts over 249.5 pass yds",
                "Saquon Barkley over 68.5 rush yds",
            )
            .is_ok()
        );
        // And the under really does parse when it is typed.
        assert!(
            parse_row(
                "rb1rush_x_qbpassunder",
                "Saquon Barkley over 68.5 rush yds",
                "Jalen Hurts under 249.5 pass yds",
            )
            .is_ok()
        );
    }

    #[test]
    fn an_unknown_family_names_the_ones_that_exist() {
        let error =
            parse_row("qb_x_kicker", "a over 1.5 pass tds", "b over 1.5 pass tds").unwrap_err();
        assert!(error.contains("unknown family"), "{error}");
        assert!(error.contains("passyds_x_wr1recyds"), "{error}");
    }

    /// The market-centred family: a spread leg and a game total, which are the
    /// two shapes with no player subject.
    #[test]
    fn the_market_centred_family_parses_its_two_odd_legs() {
        let row = parse_row("spread_x_total", "Chiefs -2.5", "game total over 47.5").unwrap();
        assert_eq!(row.leg_a.stat, LegStat::Spread);
        assert_eq!(row.leg_a.point, Some(-2.5));
        assert_eq!(row.leg_b.stat, LegStat::GameTotal);
        assert_eq!(row.leg_b.subject_key, "");
        // A positive spread is the same shape with the other sign.
        let dog = parse_row("spread_x_total", "Bills +3.5", "game total over 47.5").unwrap();
        assert_eq!(dog.leg_a.point, Some(3.5));
    }
}
