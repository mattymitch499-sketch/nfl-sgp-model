//! The Odds API client: two-sided NFL sportsbook quotes for the legs the
//! same-game-parlay study prices.
//!
//! ## What a "quote" is here
//!
//! [`TwoSidedQuote`] is always **both sides of one market**. That is the whole
//! point: a one-sided price cannot be devigged, and a raw implied probability
//! carries the book's margin into every downstream joint. One-sided outcomes
//! are dropped rather than emitted with a missing side, so a consumer can
//! never accidentally treat a vig-loaded number as a fair one.
//!
//! Three market shapes are handled, which covers every NFL leg family a
//! ticket can be built from:
//!
//! * **Over/Under lines** — player props (`player_pass_yds`,
//!   `player_reception_yds`, `player_rush_yds`, `player_pass_tds`, ...), team
//!   totals (`team_totals`, subject = the team) and game totals (`totals`,
//!   which carry no subject on the wire and are keyed under [`GAME_SUBJECT`]
//!   so the two sides still find each other). Paired by (subject, line).
//! * **Yes/No** — `player_anytime_td`, `player_1st_td`. Paired by player.
//!   Most books list only the Yes side, so these usually come back one-sided
//!   and are dropped. That is the honest answer: an anytime-TD leg cannot be
//!   devigged from this feed, so a family built on one has to anchor its
//!   marginal some other way or stay off the board. The baseball study hit
//!   the same wall on batter strikeouts; the lesson is to see it coming.
//! * **Two-way** — moneylines (`h2h`, `h2h_h1`) and spreads (`spreads`,
//!   `spreads_h1`). Each game yields two quotes, one per team's perspective,
//!   and a spread keeps each side's own line. A three-way listing (a priced
//!   tie) is skipped: its margin does not devig two-sidedly.
//!
//! ## Credits
//!
//! The events list is **free** — it does not count against the quota — so
//! game selection costs nothing and only the per-event odds calls bill, at
//! `unique markets returned × regions`. Markets a book does not offer are not
//! billed, so asking for a market that turns out to be absent is safe. Every
//! response carries `x-requests-remaining`; the client records it after each
//! call so a caller can stop *before* a floor rather than discovering the cap
//! by exhausting it. That matters more than it sounds: a spent key returns
//! empty slates that look exactly like a quiet week.
//!
//! **Nothing here is cached.** A live price is the opposite of a completed
//! box score; serving one from disk would silently answer a question about
//! the past.
//!
//! ## Timing
//!
//! [`OddsEvent::is_pregame`] treats an unreadable timestamp as **started**.
//! In-play quotes answer a different question — the market can see the score
//! — and the baseball study once had a measured error inflate from 9.28pp to
//! 15.64pp from eight in-play rows. Filter at collection and again at
//! analysis, because an append-only file outlives any one filter.
//!
//! Needs `ODDS_API_KEY` in the environment. Get a free key at the-odds-api.com
//! (500 credits a month, no card); it is never read from a file or written to
//! one.

use crate::{Error, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::time::Duration as StdDuration;

pub const ODDS_BASE: &str = "https://api.the-odds-api.com/v4";
pub const SPORT: &str = "americanfootball_nfl";

/// Region `us` alone carries every book a ticket can actually be placed at
/// here. Since cost is `markets × regions`, adding `us2` for ESPN Bet and Hard
/// Rock would double the bill for books that are not in the plan.
pub const DEFAULT_REGION: &str = "us";

/// The books the study actually targets, chosen 2026-09-03.
///
/// The feed returns every book in the region at no extra cost, so this is not
/// a spending decision — it is a **measurement** decision, and it has to be
/// fixed in advance. The decision statistic is the best book's EV per game
/// (shopping is the executable strategy), and a maximum over books is biased
/// upward by however many books it ranges over: add a fourth and fifth book and
/// the measured edge rises even under perfectly fair pricing, purely from
/// taking the max of more draws. Which books are in the set therefore has to be
/// pre-registered exactly like the bar, and it has to be the set a bet would
/// really be placed at.
///
/// Quotes from any other book are dropped at collection, so the board file
/// holds only what the statistic is allowed to see.
pub const TARGET_BOOKS: [&str; 3] = ["draftkings", "fanduel", "fanatics"];

/// Whether a bookmaker key is one of [`TARGET_BOOKS`].
pub fn is_target_book(book: &str) -> bool {
    TARGET_BOOKS.contains(&book)
}

/// Subject assigned to a market whose outcomes name no player or team — a
/// game total's Over and Under — so both sides pair under one key instead of
/// splitting into an "Over" subject and an "Under" subject that never meet.
pub const GAME_SUBJECT: &str = "game";

/// One scheduled game as the odds feed sees it.
#[derive(Debug, Clone)]
pub struct OddsEvent {
    pub id: String,
    /// RFC-3339 kickoff, UTC.
    pub commence_time: String,
    pub home_team: String,
    pub away_team: String,
}

impl OddsEvent {
    /// `"Away @ Home"` in the feed's own full team names — the game label
    /// written to the board, so the join never depends on an abbreviation
    /// table this crate would have to invent and maintain.
    pub fn label(&self) -> String {
        format!("{} @ {}", self.away_team, self.home_team)
    }

    /// Minutes until kickoff; negative once the game is underway, `None`
    /// when the timestamp does not parse.
    pub fn minutes_to_kickoff(&self) -> Option<i64> {
        minutes_until(&self.commence_time)
    }

    /// The slate this game belongs to — see [`slate_date`].
    pub fn slate_date(&self) -> Option<chrono::NaiveDate> {
        slate_date(&self.commence_time)
    }

    /// Whether the game has yet to start. An unreadable timestamp counts as
    /// **started**, never as pregame.
    pub fn is_pregame(&self) -> bool {
        matches!(self.minutes_to_kickoff(), Some(minutes) if minutes >= 0)
    }
}

/// Both sides of one book's quote on one market.
///
/// `price` and `other_price` are American odds and are always populated —
/// see the module note on why a one-sided quote is dropped instead.
#[derive(Debug, Clone, PartialEq)]
pub struct TwoSidedQuote {
    /// The feed's bookmaker key, e.g. `draftkings`, `fanduel`, `betmgm`.
    pub book: String,
    /// The feed's market key, e.g. `player_pass_yds`.
    pub market: String,
    /// Player name for props; team name for team totals, moneylines and
    /// spreads; [`GAME_SUBJECT`] for a game total.
    pub subject: String,
    /// The line for Over/Under markets and the spread for `subject`'s side of
    /// a spread market; `None` for a moneyline or a Yes/No market.
    pub point: Option<f64>,
    /// Price on the Over / Yes / `subject`'s side.
    pub price: f64,
    /// Price on the Under / No / the opposing side.
    pub other_price: f64,
}

impl TwoSidedQuote {
    /// The margin-free probability of `price`'s side, and the overround it was
    /// stripped from. Delegates to [`crate::sgp::devig`] so the board and the
    /// hand-collected sheet cannot drift apart on devigging convention.
    pub fn devigged(&self) -> Option<(f64, f64)> {
        crate::sgp::devig(self.price, self.other_price)
    }
}

/// A blocking Odds API client that tracks its own credit balance.
pub struct OddsApiClient {
    http: reqwest::blocking::Client,
    api_key: String,
    base_url: String,
    remaining: Option<u32>,
}

impl OddsApiClient {
    /// Reads `ODDS_API_KEY` from the environment.
    pub fn from_env() -> Result<Self> {
        let key = std::env::var("ODDS_API_KEY").map_err(|_| {
            Error::Data(
                "ODDS_API_KEY is not set. Get a free key at the-odds-api.com and export it.".into(),
            )
        })?;
        Self::with_key(key)
    }

    pub fn with_key(api_key: String) -> Result<Self> {
        let http = reqwest::blocking::Client::builder()
            .timeout(StdDuration::from_secs(20))
            .user_agent("nfl-sgp/0.1")
            .build()?;
        Ok(Self {
            http,
            api_key,
            base_url: ODDS_BASE.into(),
            remaining: None,
        })
    }

    /// Credits left as of the last response, or `None` before any call.
    pub fn credits_remaining(&self) -> Option<u32> {
        self.remaining
    }

    /// Every listed NFL event. **Free** — this endpoint does not count against
    /// the quota, so filtering the week down to the games worth pricing costs
    /// nothing.
    pub fn events(&mut self) -> Result<Vec<OddsEvent>> {
        let request = self
            .http
            .get(format!("{}/sports/{SPORT}/events", self.base_url))
            .query(&[("apiKey", self.api_key.as_str())]);
        let body = self.send(request, "events")?;
        normalize_events(&body)
    }

    /// Two-sided quotes for one event across every book in `region`.
    ///
    /// Billed at `unique markets returned × regions`, so requesting a market
    /// no book offers for this game is free. Returns an empty vector when the
    /// event has no priced markets, which is normal and not an error.
    pub fn event_odds(
        &mut self,
        event_id: &str,
        markets: &[&str],
        region: &str,
    ) -> Result<Vec<TwoSidedQuote>> {
        let markets = markets.join(",");
        let request = self
            .http
            .get(format!(
                "{}/sports/{SPORT}/events/{event_id}/odds",
                self.base_url
            ))
            .query(&[
                ("apiKey", self.api_key.as_str()),
                ("regions", region),
                ("markets", markets.as_str()),
                ("oddsFormat", "american"),
            ]);
        let body = self.send(request, "event odds")?;
        normalize_event_odds(&body)
    }

    /// Sends a request, records the credit header from the response — whether
    /// or not it succeeded, since a rejected call still reports the balance —
    /// and returns the body.
    fn send(&mut self, request: reqwest::blocking::RequestBuilder, what: &str) -> Result<String> {
        let response = request.send()?;
        if let Some(value) = response
            .headers()
            .get("x-requests-remaining")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.trim().parse::<f64>().ok())
        {
            self.remaining = Some(value as u32);
        }
        let status = response.status();
        let body = response.text()?;
        if !status.is_success() {
            return Err(Error::Data(format!(
                "odds {what} request failed ({status}): {body}"
            )));
        }
        Ok(body)
    }
}

#[derive(Deserialize)]
struct WireEvent {
    id: String,
    commence_time: String,
    home_team: String,
    away_team: String,
}

#[derive(Deserialize)]
struct WireEventOdds {
    #[serde(default)]
    bookmakers: Vec<WireBookmaker>,
}

#[derive(Deserialize)]
struct WireBookmaker {
    key: String,
    #[serde(default)]
    markets: Vec<WireMarket>,
}

#[derive(Deserialize)]
struct WireMarket {
    key: String,
    #[serde(default)]
    outcomes: Vec<WireOutcome>,
}

#[derive(Deserialize)]
struct WireOutcome {
    /// `"Over"` / `"Under"` for lines, `"Yes"` / `"No"` for scorer markets,
    /// the team name for a moneyline or spread.
    name: String,
    /// The player a prop is on, or the team a team total is on. Absent on
    /// moneylines, spreads and game totals.
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    price: Option<f64>,
    #[serde(default)]
    point: Option<f64>,
}

/// Parses the events list.
pub fn normalize_events(body: &str) -> Result<Vec<OddsEvent>> {
    let wire: Vec<WireEvent> = serde_json::from_str(body)?;
    Ok(wire
        .into_iter()
        .map(|event| OddsEvent {
            id: event.id,
            commence_time: event.commence_time,
            home_team: event.home_team,
            away_team: event.away_team,
        })
        .collect())
}

/// Parses an event-odds payload into two-sided quotes.
///
/// Emits nothing for a side without a partner, for a price that is missing,
/// or for a market whose shape is none of the three the module handles. Order
/// is deterministic (book, market, subject, line) so the board file is stable
/// across runs on identical input.
pub fn normalize_event_odds(body: &str) -> Result<Vec<TwoSidedQuote>> {
    let wire: WireEventOdds = serde_json::from_str(body)?;
    let mut quotes: Vec<TwoSidedQuote> = Vec::new();
    for bookmaker in &wire.bookmakers {
        for market in &bookmaker.markets {
            quotes.extend(pair_market(&bookmaker.key, market));
        }
    }
    quotes.sort_by(|a, b| {
        (&a.book, &a.market, &a.subject)
            .cmp(&(&b.book, &b.market, &b.subject))
            .then(
                a.point
                    .partial_cmp(&b.point)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });
    Ok(quotes)
}

/// One market's outcomes, paired into two-sided quotes by whichever of the
/// three shapes its outcome names reveal.
fn pair_market(book: &str, market: &WireMarket) -> Vec<TwoSidedQuote> {
    let names = market.outcomes.iter().map(|outcome| outcome.name.as_str());
    if names.clone().any(|name| is_over(name) || is_under(name)) {
        pair_sides(book, market, is_over, is_under, true)
    } else if names.clone().any(|name| is_yes(name) || is_no(name)) {
        pair_sides(book, market, is_yes, is_no, false)
    } else {
        pair_two_way(book, market)
    }
}

/// Pairs a first-side outcome with its second-side partner on the same
/// (subject, line). `needs_point` is true for Over/Under markets, where an
/// outcome with no line is unusable; Yes/No markets carry no line at all.
fn pair_sides(
    book: &str,
    market: &WireMarket,
    is_first: fn(&str) -> bool,
    is_second: fn(&str) -> bool,
    needs_point: bool,
) -> Vec<TwoSidedQuote> {
    // Pair before devigging: a one-sided quote carries no removable margin.
    let mut sides: HashMap<(String, String), (Option<f64>, Option<f64>)> = HashMap::new();
    for outcome in &market.outcomes {
        let Some(price) = outcome.price else {
            continue;
        };
        if needs_point && outcome.point.is_none() {
            continue;
        }
        // Props and team totals name their subject in `description`; a game
        // total does not, and falls back to one shared subject so its two
        // sides pair with each other rather than splitting on "Over"/"Under".
        let subject = outcome
            .description
            .clone()
            .unwrap_or_else(|| GAME_SUBJECT.to_string());
        let point_text = outcome
            .point
            .map_or_else(String::new, |point| format!("{point:.1}"));
        let entry = sides.entry((subject, point_text)).or_insert((None, None));
        if is_first(&outcome.name) {
            entry.0 = Some(price);
        } else if is_second(&outcome.name) {
            entry.1 = Some(price);
        }
    }
    sides
        .into_iter()
        .filter_map(|((subject, point_text), (first, second))| {
            let point = if point_text.is_empty() {
                None
            } else {
                Some(point_text.parse::<f64>().ok()?)
            };
            Some(TwoSidedQuote {
                book: book.to_string(),
                market: market.key.clone(),
                subject,
                point,
                price: first?,
                other_price: second?,
            })
        })
        .collect()
}

/// Moneylines and spreads: exactly two priced sides, each emitted from its
/// own perspective with its own line (a spread's two sides carry opposite
/// points; a moneyline carries none).
fn pair_two_way(book: &str, market: &WireMarket) -> Vec<TwoSidedQuote> {
    let priced: Vec<&WireOutcome> = market
        .outcomes
        .iter()
        .filter(|outcome| outcome.price.is_some())
        .collect();
    // Exactly two sides, or the margin does not devig two-sidedly. A listed
    // tie (three-way) is skipped rather than folded into one of the teams.
    if priced.len() != 2 {
        return Vec::new();
    }
    let (first, second) = (priced[0], priced[1]);
    let quote = |side: &WireOutcome, other: &WireOutcome| TwoSidedQuote {
        book: book.to_string(),
        market: market.key.clone(),
        subject: side.name.clone(),
        point: side.point,
        price: side.price.unwrap_or_default(),
        other_price: other.price.unwrap_or_default(),
    };
    vec![quote(first, second), quote(second, first)]
}

fn is_over(name: &str) -> bool {
    name.eq_ignore_ascii_case("over")
}

fn is_under(name: &str) -> bool {
    name.eq_ignore_ascii_case("under")
}

fn is_yes(name: &str) -> bool {
    name.eq_ignore_ascii_case("yes")
}

fn is_no(name: &str) -> bool {
    name.eq_ignore_ascii_case("no")
}

/// Minutes from now until an RFC-3339 timestamp; negative once it is past.
/// The caller treats `None` as "already started".
pub fn minutes_until(commence_time: &str) -> Option<i64> {
    let start = chrono::DateTime::parse_from_rfc3339(commence_time).ok()?;
    Some((start.with_timezone(&chrono::Utc) - chrono::Utc::now()).num_minutes())
}

/// The **slate** a kickoff belongs to, which is not the same as the UTC date
/// of its timestamp.
///
/// Sunday Night Football kicks off at 8:20pm Eastern, which is 00:20Z on
/// Monday; Monday Night Football at 8:15pm Eastern is 00:15Z on Tuesday.
/// Keying a game off `commence_time[..10]` files both under the wrong day,
/// and the hand-collected sheet — written by a person watching Sunday's board
/// — calls them Sunday and Monday. Joining on a raw UTC prefix silently
/// loses every prime-time game.
///
/// The rule, unchanged from the baseball crate: a game belongs to slate `D`
/// when its kickoff falls in `[D 10:00Z, D+1 10:00Z)`. 10:00Z is 6am Eastern
/// — nothing kicks off in that window (London games start 9:30am Eastern),
/// so the boundary never splits a real slate, and unlike a timezone
/// conversion it needs no DST table.
pub fn slate_date(commence_time: &str) -> Option<chrono::NaiveDate> {
    let start = chrono::DateTime::parse_from_rfc3339(commence_time)
        .ok()?
        .with_timezone(&chrono::Utc);
    Some((start - chrono::Duration::hours(10)).date_naive())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EVENTS: &str = r#"[
        {"id":"abc","commence_time":"2026-09-13T17:00:00Z",
         "home_team":"Pittsburgh Steelers","away_team":"Baltimore Ravens"},
        {"id":"def","commence_time":"2026-09-14T00:20:00Z",
         "home_team":"Kansas City Chiefs","away_team":"Buffalo Bills"}
    ]"#;

    #[test]
    fn events_parse_with_a_readable_label() {
        let events = normalize_events(EVENTS).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].id, "abc");
        assert_eq!(events[0].label(), "Baltimore Ravens @ Pittsburgh Steelers");
    }

    /// Two books and every market shape, plus the cases that must be
    /// dropped: a one-sided player line, an outcome with no price, a Yes with
    /// no No, and a three-way moneyline.
    const ODDS: &str = r#"{
      "bookmakers": [
        {"key":"draftkings","markets":[
          {"key":"player_pass_yds","outcomes":[
            {"name":"Over","description":"Patrick Mahomes","price":-115,"point":274.5},
            {"name":"Under","description":"Patrick Mahomes","price":-105,"point":274.5},
            {"name":"Over","description":"Josh Allen","price":-110,"point":249.5}
          ]},
          {"key":"totals","outcomes":[
            {"name":"Over","price":-108,"point":47.5},
            {"name":"Under","price":-112,"point":47.5}
          ]},
          {"key":"team_totals","outcomes":[
            {"name":"Over","description":"Kansas City Chiefs","price":-110,"point":24.5},
            {"name":"Under","description":"Kansas City Chiefs","price":-110,"point":24.5},
            {"name":"Over","description":"Buffalo Bills","price":-115,"point":23.5},
            {"name":"Under","description":"Buffalo Bills","price":-105,"point":23.5}
          ]},
          {"key":"player_anytime_td","outcomes":[
            {"name":"Yes","description":"Travis Kelce","price":140},
            {"name":"No","description":"Travis Kelce","price":-180},
            {"name":"Yes","description":"James Cook","price":-120}
          ]},
          {"key":"spreads","outcomes":[
            {"name":"Kansas City Chiefs","price":-110,"point":-2.5},
            {"name":"Buffalo Bills","price":-110,"point":2.5}
          ]},
          {"key":"h2h","outcomes":[
            {"name":"Kansas City Chiefs","price":-142},
            {"name":"Buffalo Bills","price":120}
          ]}
        ]},
        {"key":"fanduel","markets":[
          {"key":"player_pass_yds","outcomes":[
            {"name":"Over","description":"Patrick Mahomes","price":-112,"point":274.5},
            {"name":"Under","description":"Patrick Mahomes","price":-108,"point":274.5}
          ]},
          {"key":"h2h_3_way","outcomes":[
            {"name":"Kansas City Chiefs","price":-130},
            {"name":"Draw","price":3000},
            {"name":"Buffalo Bills","price":110}
          ]},
          {"key":"player_rush_yds","outcomes":[
            {"name":"Over","description":"James Cook","point":68.5},
            {"name":"Under","description":"James Cook","price":-115,"point":68.5}
          ]}
        ]}
      ]
    }"#;

    #[test]
    fn player_over_under_pairs_by_player_and_line() {
        let quotes = normalize_event_odds(ODDS).unwrap();
        let mahomes: Vec<&TwoSidedQuote> = quotes
            .iter()
            .filter(|quote| quote.subject == "Patrick Mahomes" && quote.market == "player_pass_yds")
            .collect();
        assert_eq!(mahomes.len(), 2, "one per book");
        let dk = mahomes.iter().find(|q| q.book == "draftkings").unwrap();
        assert_eq!(dk.point, Some(274.5));
        assert_eq!(dk.price, -115.0);
        assert_eq!(dk.other_price, -105.0);
    }

    /// The case that split silently in the baseball crate's untested path: a
    /// game total names no subject, so its Over and Under must share one key.
    #[test]
    fn a_game_total_pairs_under_the_shared_game_subject() {
        let quotes = normalize_event_odds(ODDS).unwrap();
        let totals: Vec<&TwoSidedQuote> = quotes
            .iter()
            .filter(|quote| quote.market == "totals")
            .collect();
        assert_eq!(totals.len(), 1);
        assert_eq!(totals[0].subject, GAME_SUBJECT);
        assert_eq!(totals[0].point, Some(47.5));
        assert_eq!(totals[0].price, -108.0);
        assert_eq!(totals[0].other_price, -112.0);
    }

    #[test]
    fn team_totals_pair_by_team() {
        let quotes = normalize_event_odds(ODDS).unwrap();
        let teams: Vec<&TwoSidedQuote> = quotes
            .iter()
            .filter(|quote| quote.market == "team_totals")
            .collect();
        assert_eq!(teams.len(), 2);
        let bills = teams
            .iter()
            .find(|quote| quote.subject == "Buffalo Bills")
            .unwrap();
        assert_eq!(bills.point, Some(23.5));
        assert_eq!(bills.price, -115.0);
        assert_eq!(bills.other_price, -105.0);
    }

    /// A scorer market with both sides pairs; the far more common Yes-only
    /// listing is dropped, because it cannot be devigged.
    #[test]
    fn yes_no_pairs_by_player_and_a_lone_yes_is_dropped() {
        let quotes = normalize_event_odds(ODDS).unwrap();
        let scorers: Vec<&TwoSidedQuote> = quotes
            .iter()
            .filter(|quote| quote.market == "player_anytime_td")
            .collect();
        assert_eq!(scorers.len(), 1);
        assert_eq!(scorers[0].subject, "Travis Kelce");
        assert_eq!(scorers[0].point, None);
        assert_eq!(scorers[0].price, 140.0);
        assert_eq!(scorers[0].other_price, -180.0);
        assert!(
            !quotes.iter().any(|quote| quote.subject == "James Cook"),
            "a Yes with no No (and an Over with no price) must not appear"
        );
    }

    #[test]
    fn one_sided_and_priceless_outcomes_are_dropped() {
        let quotes = normalize_event_odds(ODDS).unwrap();
        assert!(
            !quotes.iter().any(|quote| quote.subject == "Josh Allen"),
            "an Over with no Under cannot be devigged, so it must not appear"
        );
        assert!(
            !quotes.iter().any(|quote| quote.market == "player_rush_yds"),
            "an outcome missing its price leaves the pair incomplete"
        );
    }

    /// Spreads keep each side's own line; moneylines carry none; a listed
    /// tie yields nothing.
    #[test]
    fn two_way_markets_yield_both_perspectives_with_their_own_lines() {
        let quotes = normalize_event_odds(ODDS).unwrap();
        let spreads: Vec<&TwoSidedQuote> = quotes
            .iter()
            .filter(|quote| quote.market == "spreads")
            .collect();
        assert_eq!(spreads.len(), 2);
        let chiefs = spreads
            .iter()
            .find(|quote| quote.subject == "Kansas City Chiefs")
            .unwrap();
        assert_eq!(chiefs.point, Some(-2.5));
        assert_eq!(chiefs.price, -110.0);
        let bills = spreads
            .iter()
            .find(|quote| quote.subject == "Buffalo Bills")
            .unwrap();
        assert_eq!(bills.point, Some(2.5));

        let moneyline: Vec<&TwoSidedQuote> = quotes
            .iter()
            .filter(|quote| quote.market == "h2h")
            .collect();
        assert_eq!(moneyline.len(), 2);
        let chiefs_ml = moneyline
            .iter()
            .find(|quote| quote.subject == "Kansas City Chiefs")
            .unwrap();
        assert_eq!(chiefs_ml.point, None);
        assert_eq!(chiefs_ml.price, -142.0);
        assert_eq!(chiefs_ml.other_price, 120.0);

        assert!(
            !quotes.iter().any(|quote| quote.market == "h2h_3_way"),
            "a listed tie does not devig two-sidedly, so the market is skipped"
        );
    }

    #[test]
    fn devigging_a_quote_matches_the_sgp_module() {
        let quote = TwoSidedQuote {
            book: "draftkings".into(),
            market: "player_pass_yds".into(),
            subject: "Patrick Mahomes".into(),
            point: Some(274.5),
            price: -110.0,
            other_price: -110.0,
        };
        let (devigged, overround) = quote.devigged().unwrap();
        assert!((devigged - 0.5).abs() < 1e-12);
        assert!(overround > 1.0);
    }

    #[test]
    fn ordering_is_stable_across_runs() {
        let first = normalize_event_odds(ODDS).unwrap();
        let second = normalize_event_odds(ODDS).unwrap();
        assert_eq!(first, second);
        assert!(
            first.windows(2).all(|pair| pair[0].book <= pair[1].book),
            "books must come out sorted so the board file does not churn"
        );
    }

    #[test]
    fn an_unreadable_kickoff_is_never_pregame() {
        let event = OddsEvent {
            id: "x".into(),
            commence_time: "not a timestamp".into(),
            home_team: "Pittsburgh Steelers".into(),
            away_team: "Baltimore Ravens".into(),
        };
        assert_eq!(event.minutes_to_kickoff(), None);
        assert!(!event.is_pregame());
    }

    #[test]
    fn a_past_kickoff_is_not_pregame() {
        let event = OddsEvent {
            id: "x".into(),
            commence_time: "2020-09-13T17:00:00Z".into(),
            home_team: "Pittsburgh Steelers".into(),
            away_team: "Baltimore Ravens".into(),
        };
        assert!(event.minutes_to_kickoff().unwrap() < 0);
        assert!(!event.is_pregame());
    }

    /// Prime-time games are the ones a UTC prefix would file under the wrong
    /// day: SNF is 00:20Z Monday but belongs to Sunday's slate, MNF is 00:15Z
    /// Tuesday but belongs to Monday's.
    #[test]
    fn prime_time_games_stay_on_the_slate_a_human_would_call_them() {
        let day = |text: &str| slate_date(text).unwrap().to_string();
        // 1pm Eastern Sunday.
        assert_eq!(day("2026-09-13T17:00:00Z"), "2026-09-13");
        // 4:25pm Eastern Sunday.
        assert_eq!(day("2026-09-13T20:25:00Z"), "2026-09-13");
        // Sunday Night Football, 8:20pm Eastern — already Monday in UTC.
        assert_eq!(day("2026-09-14T00:20:00Z"), "2026-09-13");
        // Monday Night Football, 8:15pm Eastern — already Tuesday in UTC.
        assert_eq!(day("2026-09-15T00:15:00Z"), "2026-09-14");
        // London game, 9:30am Eastern Sunday.
        assert_eq!(day("2026-10-04T13:30:00Z"), "2026-10-04");
        // The boundary itself, 6am Eastern, belongs to the later slate.
        assert_eq!(day("2026-09-14T10:00:00Z"), "2026-09-14");
        assert_eq!(day("2026-09-14T09:59:00Z"), "2026-09-13");
        assert_eq!(slate_date("not a timestamp"), None);
    }
}
